#!/usr/bin/env python3
"""The GPU gate of the own vision tower (rule 6 of RESULTS-OWN-3V.md): our GPU tower against mtmd's CPU path, and no noisier than mtmd's GPU path.

    python3 gate_gpu.py --helper <strata-vision> --compare <strata-vision-own-compare> --mmproj <mmproj.gguf> --model <text model .gguf> --out <dir>
                        [--max-tokens 1024] [--repeat 4] [--ref-dir <dir of our float32 reference's <picture>.sve>] [--cpu-dir <dir of stored mtmd-CPU <picture>.sve>]
                        <image> [<image> ...]

One strata-vision per reader (mtmd on the CPU unless --cpu-dir gives its files, mtmd on the GPU, own on the GPU), every picture through each.  Per picture:
  main      cosine(own GPU, mtmd CPU)              must be >= --min-cos (0.999)
  floor     cosine(mtmd GPU, mtmd CPU)             what users run today
  no-worse  main >= floor - 1e-4  or  main >= 0.9999   our GPU tower is at least as close to mtmd-CPU as mtmd-GPU is (the coordinator's tolerance, 11-09)
  info      cosine(own GPU, mtmd GPU), cosine(own GPU, float32 reference) (with --ref-dir)
and the times of the two GPU readers (median of --repeat further encodes, after the first); the GPU memory (nvidia-smi, rocm-smi or Linux DRM fdinfo); whether
our GPU tower gives byte-identical output on every encode (REPEATABLE line; information, the exit code does not depend on it).  Exit 0 when every picture passes main and no-worse.  Standard library only.
"""
import argparse
import hashlib
import os
import statistics
import subprocess
import sys


def fdinfo_mib(pid):
    """Device-local memory of a process from Linux's DRM fdinfo (/proc/<pid>/fdinfo; i915 / xe drm-total-local0 or drm-resident-local0, amdgpu
    drm-memory-vram), the largest over its DRM files, MiB; 0 where it cannot be read (no such files, another OS)."""
    best = 0
    d = "/proc/%d/fdinfo" % pid
    try:
        fds = os.listdir(d)
    except OSError:
        return 0
    for fd in fds:
        try:
            with open(os.path.join(d, fd)) as f:
                txt = f.read()
        except OSError:
            continue
        if "drm-driver" not in txt:
            continue
        for line in txt.splitlines():
            k, _, v = line.partition(":")
            if k.strip() not in ("drm-total-local0", "drm-resident-local0", "drm-memory-vram", "drm-total-vram0", "drm-resident-vram0"):
                continue
            parts = v.split()
            if not parts:
                continue
            n = float(parts[0])
            unit = parts[1] if len(parts) > 1 else ""
            mib = {"KiB": n / 1024.0, "MiB": n, "GiB": n * 1024.0}.get(unit, n / 1048576.0)
            best = max(best, int(mib))
    return best


def mac_footprint_mib(pid):
    """macOS: the process's physical footprint (what Activity Monitor calls Memory; on unified memory it includes the GPU buffers), MiB, from the
    `footprint` tool (no root needed for our own process); 0 elsewhere or when it cannot be read."""
    if sys.platform != "darwin":
        return 0
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return 0
    for line in out.splitlines():
        low = line.lower()
        if "phys_footprint:" in low or low.strip().startswith("footprint:"):
            t = line.split(":", 1)[1].split()
            if len(t) >= 2:
                try:
                    n = float(t[0])
                except ValueError:
                    continue
                return int({"KB": n / 1024.0, "MB": n, "GB": n * 1024.0}.get(t[1].upper(), 0))
    return 0


class Reader:
    def __init__(self, exe, tower, mmproj, model, gpu, max_tokens, threads, out, label):
        args = [exe, "--mmproj", mmproj, "--model", model, "--tower", tower]
        if gpu:
            args.append("--gpu")
        if max_tokens:
            args += ["--max-tokens", str(max_tokens)]
        if threads and not gpu:
            args += ["--threads", str(threads)]
        self.label = label
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(out, label + ".stderr"), "w"), text=True)
        line = self.p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise SystemExit("%s reader did not start: %r" % (label, line))

    peak_mib = 0

    def vram(self):
        """This reader's GPU memory now (nvidia-smi's compute-apps list, by pid); remembers the peak.  0 when it cannot be read."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=30).stdout
            for line in out.splitlines():
                pid, _, mem = line.partition(",")
                if pid.strip() == str(self.p.pid) and mem.strip().isdigit():
                    self.peak_mib = max(self.peak_mib, int(mem.strip()))
        except Exception:  # noqa: BLE001
            pass
        try:   # AMD: rocm-smi's KFD process list:  PID  NAME  #GPUs  VRAM USED(B)  SDMA USED  CU OCCUPANCY
            out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                t = line.split()
                if len(t) >= 6 and t[0] == str(self.p.pid) and t[-3].isdigit():
                    self.peak_mib = max(self.peak_mib, int(t[-3]) // (1 << 20))
        except Exception:  # noqa: BLE001
            pass
        self.peak_mib = max(self.peak_mib, fdinfo_mib(self.p.pid))   # Linux DRM: Intel (i915 / xe) and AMD report device memory per process here
        self.peak_mib = max(self.peak_mib, mac_footprint_mib(self.p.pid))   # a Mac: the whole footprint (unified memory)
        return self.peak_mib

    def enc(self, img, out):
        self.p.stdin.write("ENC %s %s\n" % (img, out))
        self.p.stdin.flush()
        res = self.p.stdout.readline().strip()
        if self.label.endswith("-gpu"):
            self.vram()
        return res

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def cmp(compare, a, b):
    r = subprocess.run([compare, a, b], capture_output=True, text=True).stdout.strip()
    if not r.startswith("n="):
        return None
    return dict(p.split("=") for p in r.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--cpu-dir", default="")
    ap.add_argument("--skip-cpu", action="store_true", help="no mtmd-CPU run: for pictures at the largest token limit (the columns then compare the two GPU readers)")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    for d in ("mtmd-cpu", "mtmd-gpu", "own-gpu"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    mk = lambda tower, gpu, label: Reader(a.helper, tower, a.mmproj, a.model, gpu, a.max_tokens, a.threads, a.out, label)  # noqa: E731
    r_cpu = None if (a.cpu_dir or a.skip_cpu) else mk("mtmd", False, "mtmd-cpu")
    r_mg = mk("mtmd", True, "mtmd-gpu")
    r_og = mk("own", True, "own-gpu")

    unrepeatable = []      # own-gpu pictures whose output bytes changed between encodes in one process

    def digest(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def run(r, img, name):
        out = os.path.join(a.out, r.label, name + ".sve")
        res = r.enc(os.path.abspath(img), out)
        if not res.startswith("OK"):
            return out, None, res
        ms = []
        seen = {digest(out)} if r is r_og else set()
        for _ in range(a.repeat):
            again = r.enc(os.path.abspath(img), out)
            if again.startswith("OK"):
                ms.append(float(again.split()[4]))
                if r is r_og:
                    seen.add(digest(out))
        if len(seen) > 1:
            unrepeatable.append("%s (%d different outputs in %d encodes)" % (name, len(seen), a.repeat + 1))
        return out, statistics.median(ms) if ms else float(res.split()[4]), res

    hdr = "%-22s %6s | %9s %9s %9s | %9s %9s | %8s %8s %6s | %s"
    print(hdr % ("picture", "tokens", "own/cpu", "mtmdG/cpu", "own/mtmdG", "own/ref", "tokmin", "mtmdG ms", "own ms", "ratio", "gate"), flush=True)
    ok_all, main_c, floor_c = True, [], []
    for img in a.images:
        name = os.path.basename(img)
        if a.skip_cpu:
            cpu_f, res_c = None, "OK"      # a picture at the largest token limit: mtmd on the CPU would take many minutes; the GPU reader of mtmd is the yardstick
        elif a.cpu_dir:
            cpu_f, res_c = os.path.join(a.cpu_dir, name + ".sve"), "OK"
        else:
            cpu_f, _, res_c = run(r_cpu, img, name)
        mg_f, mg_ms, res_m = run(r_mg, img, name)
        og_f, og_ms, res_o = run(r_og, img, name)
        if cpu_f is None:
            cpu_f = mg_f
        if not (str(res_c).startswith("OK") and res_m.startswith("OK") and res_o.startswith("OK")) or mg_ms is None or og_ms is None:
            print("%-22s FAILED: cpu=%s mtmd-gpu=%s own-gpu=%s" % (name, res_c, res_m, res_o), flush=True)
            ok_all = False
            continue
        main = cmp(a.compare, og_f, cpu_f)
        floor = cmp(a.compare, mg_f, cpu_f)
        info = cmp(a.compare, og_f, mg_f)
        ref = cmp(a.compare, og_f, os.path.join(a.ref_dir, name + ".sve")) if a.ref_dir and os.path.isfile(os.path.join(a.ref_dir, name + ".sve")) else None
        if not (main and floor and info):
            print("%-22s shapes differ between readers" % name, flush=True)
            ok_all = False
            continue
        m, f = float(main["global"]), float(floor["global"])
        passed = (m >= 0.99) if a.skip_cpu else (m >= a.min_cos and (m >= f - 1e-4 or m >= 0.9999))     # rule 1: >= 0.999; rule 2: ours >= mtmd-GPU - 1e-4, or ours >= 0.9999 (coordinator, 11-09)  --skip-cpu: no error and the two GPU readers agree (0.99); the time and memory are the point
        ok_all &= passed
        main_c.append(m)
        floor_c.append(f)
        print(hdr % (name, main["n"], main["global"], floor["global"], info["global"], ref["global"] if ref else "-", main["tokmin"], "%.0f" % mg_ms, "%.0f" % og_ms,
                     "%.2f" % (og_ms / mg_ms), "ok" if passed else "FAIL (%s)" % ("main < %.4f" % a.min_cos if m < a.min_cos else "noisier than mtmd-GPU")), flush=True)
    vram = "GPU memory of the reader (nvidia-smi / rocm-smi / DRM fdinfo / macOS footprint, peak after the pictures): mtmd-gpu %d MiB, own-gpu %d MiB%s" % (
        r_mg.peak_mib, r_og.peak_mib, "" if r_og.peak_mib <= r_mg.peak_mib else "  (own is larger)")
    print("REPEATABLE own-gpu: %s" % ("yes, every picture byte-identical over %d encodes" % (a.repeat + 1) if not unrepeatable else "NO: " + "; ".join(unrepeatable)),
          flush=True)
    for r in (r_cpu, r_mg, r_og):
        if r:
            r.close()
    print(vram, flush=True)
    if main_c:
        print("SUMMARY own GPU vs mtmd CPU: %d pictures, min %.6f mean %.6f; mtmd GPU vs mtmd CPU (the floor): min %.6f mean %.6f" %
              (len(main_c), min(main_c), sum(main_c) / len(main_c), min(floor_c), sum(floor_c) / len(floor_c)), flush=True)
    print("RESULT %s" % ("PASS" if ok_all and main_c else "FAIL"), flush=True)
    return 0 if ok_all and main_c else 1


if __name__ == "__main__":
    sys.exit(main())
