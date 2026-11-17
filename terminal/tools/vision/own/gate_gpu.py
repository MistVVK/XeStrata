#!/usr/bin/env python3
"""The GPU gate of the own vision tower (rule 6 of RESULTS-OWN-3V.md): our GPU tower against the float32 reference, and no noisier than mtmd's GPU path.

    python3 gate_gpu.py --helper <strata-vision> --compare <strata-vision-own-compare> --mmproj <mmproj.gguf> --model <text model .gguf> --out <dir>
                        [--max-tokens 1024] [--repeat 4] [--ref-dir <dir of our float32 reference's <picture>.sve>] [--cpu-dir <dir of stored mtmd-CPU <picture>.sve>]
                        <image> [<image> ...]

The reference is our float32 tower (VitRef: the own tower on the CPU with STRATA_VIT_CPU=ref), from --ref-dir where it holds the picture's file, else made
here.  Until 11-14 the reference was mtmd's CPU path; mtmd itself sits up to 0.0168 below float32 (16 pictures: global cosine min 0.983 on its GPU, 0.998
on its CPU), so it stays only as an information column.  One strata-vision per reader (VitRef on the CPU, mtmd on the CPU unless --cpu-dir gives its
files, mtmd on the GPU, own on the GPU), every picture through each.  Per picture:
  main      cosine(own GPU, float32)               must be >= --min-cos (0.999)
  floor     cosine(mtmd GPU, float32)              what llama.cpp's reader gives
  no-worse  main >= floor - 1e-4  or  main >= 0.9999   our GPU tower is at least as close to float32 as mtmd-GPU is (the coordinator's tolerance, 11-09)
  info      cosine(own GPU, mtmd CPU), cosine(own GPU, mtmd GPU) (printed, no pass / fail)
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
    def __init__(self, exe, tower, mmproj, model, gpu, max_tokens, threads, out, label, env=None):
        args = [exe, "--mmproj", mmproj, "--model", model, "--tower", tower]
        if gpu:
            args.append("--gpu")
        if max_tokens:
            args += ["--max-tokens", str(max_tokens)]
        if threads and not gpu:
            args += ["--threads", str(threads)]
        self.label = label
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(out, label + ".stderr"), "w"), text=True,
                                  env=dict(os.environ, **env) if env else None)
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


def read_sve(path):
    """(n, nx, ny, dim, floats) of a strata-vision output file, or None."""
    import array
    try:
        with open(path, "rb") as f:
            hdr = array.array("i")
            hdr.frombytes(f.read(20))
            if hdr[0] != 0x31455653 or hdr[1] <= 0 or hdr[4] <= 0:
                return None
            v = array.array("f")
            v.frombytes(f.read(4 * hdr[1] * hdr[4]))
            if len(v) != hdr[1] * hdr[4]:
                return None
            return hdr[1], hdr[2], hdr[3], hdr[4], v
    except OSError:
        return None


def py_cmp(a, b):
    """compare_sve's numbers that the gate reads (n, global, tokmin), in Python: for a box without the compiled tool."""
    import math
    import operator
    ra, rb = read_sve(a), read_sve(b)
    if not ra or not rb or ra[:4] != rb[:4]:
        return None
    n, _, _, dim, va, vb = *ra[:4], ra[4], rb[4]
    dot = sum(map(operator.mul, va, vb))
    na, nb = sum(map(operator.mul, va, va)), sum(map(operator.mul, vb, vb))
    tokmin = 1.0
    for t in range(n):
        x, y = va[t * dim:(t + 1) * dim], vb[t * dim:(t + 1) * dim]
        d, p, q = sum(map(operator.mul, x, y)), sum(map(operator.mul, x, x)), sum(map(operator.mul, y, y))
        tokmin = min(tokmin, d / math.sqrt(p * q) if p > 0 and q > 0 else (1.0 if p == q else 0.0))
    g = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0
    return {"n": str(n), "global": "%.6f" % g, "tokmin": "%.6f" % tokmin}


def cmp(compare, a, b):
    if not compare:
        return py_cmp(a, b)
    r = subprocess.run([compare, a, b], capture_output=True, text=True).stdout.strip()
    if not r.startswith("n="):
        return None
    return dict(p.split("=") for p in r.split())


def gate_pass(main, floor, min_cos=0.999, skip_cpu=False):
    """main = cosine(own GPU, float32), floor = cosine(mtmd GPU, float32).  Rule 1: main >= min_cos; rule 2: main >= floor - 1e-4, or main >= 0.9999.
    With --skip-cpu (no float32 reference: pictures at the largest token limit) main compares the two GPU readers and only has to reach 0.99."""
    if skip_cpu:
        return main >= 0.99
    if floor is None:          # no mtmd arm (--mtmd-helper none): rule 1 alone
        return main >= min_cos
    return main >= min_cos and (main >= floor - 1e-4 or main >= 0.9999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--mtmd-helper", default="", help="a strata-vision that still has llama.cpp's mtmd reader (0.2.15 or older; since 0.2.16 --helper has none): "
                                                       "the mtmd-CPU / mtmd-GPU arms run it with --tower mtmd.  'none' leaves them out (rule 1 alone)")
    ap.add_argument("--old-helper", default="", help="an older strata-vision whose own tower is compared with --helper's byte for byte (the IDENTICAL column)")
    ap.add_argument("--compare", default="", help="strata-vision-own-compare; without it the cosines are computed in Python (slower)")
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
    ap.add_argument("--cpu-arm", action="store_true", help="gate the CPU reader instead: 'own' and the mtmd arm run without --gpu (no separate mtmd-CPU arm)")
    a = ap.parse_args()
    for d in ("ref-cpu", "mtmd-cpu", "mtmd-gpu", "own-gpu", "own-old"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    mk = lambda exe, tower, gpu, label, env=None: Reader(exe, tower, a.mmproj, a.model, gpu, a.max_tokens, a.threads, a.out, label, env)  # noqa: E731
    mtmd = "" if a.mtmd_helper == "none" else (a.mtmd_helper or a.helper)
    if mtmd == a.helper:
        print("NOTE: the mtmd arms use --helper itself (no --mtmd-helper); a 0.2.16 or newer strata-vision has no mtmd reader, so they then run ours", flush=True)
    on_gpu = not a.cpu_arm
    ref_of = lambda name: os.path.join(a.ref_dir, name + ".sve") if a.ref_dir and os.path.isfile(os.path.join(a.ref_dir, name + ".sve")) else None  # noqa: E731
    need_ref = not a.skip_cpu and any(ref_of(os.path.basename(i)) is None for i in a.images)
    r_ref = mk(a.helper, "own", False, "ref-cpu", {"STRATA_VIT_CPU": "ref"}) if need_ref else None     # the float32 reference, made here where --ref-dir lacks it
    r_cpu = None if (a.cpu_dir or a.skip_cpu or not mtmd or not on_gpu) else mk(mtmd, "mtmd", False, "mtmd-cpu")
    r_mg = mk(mtmd, "mtmd", on_gpu, "mtmd-gpu") if mtmd else None     # (with --cpu-arm this is mtmd's CPU reader, the bar of that arm)
    r_og = mk(a.helper, "own", on_gpu, "own-gpu")
    r_old = mk(a.old_helper, "own", on_gpu, "own-old") if a.old_helper else None

    unrepeatable = []      # own-gpu pictures whose output bytes changed between encodes in one process

    def digest(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def run(r, img, name, repeat=None):
        out = os.path.join(a.out, r.label, name + ".sve")
        res = r.enc(os.path.abspath(img), out)
        if not res.startswith("OK"):
            return out, None, res
        ms = []
        seen = {digest(out)} if r is r_og else set()
        for _ in range(a.repeat if repeat is None else repeat):
            again = r.enc(os.path.abspath(img), out)
            if again.startswith("OK"):
                ms.append(float(again.split()[4]))
                if r is r_og:
                    seen.add(digest(out))
        if len(seen) > 1:
            unrepeatable.append("%s (%d different outputs in %d encodes)" % (name, len(seen), a.repeat + 1))
        return out, statistics.median(ms) if ms else float(res.split()[4]), res

    hdr = "%-22s %6s | %9s %9s %9s | %9s %9s | %8s %8s %6s | %9s | %s"
    print(hdr % ("picture", "tokens", "own/ref", "mtmdG/ref", "own/mtmdG", "own/mtmdC", "tokmin", "mtmdG ms", "own ms", "ratio", "old own", "gate"), flush=True)
    ok_all, main_c, floor_c, ratios, same_old = True, [], [], [], []
    for img in a.images:
        name = os.path.basename(img)
        res_r, res_c, res_m, cpu_f, mg_f, mg_ms = "OK", "OK", "OK", None, None, None
        if a.skip_cpu:
            ref_f = None      # a picture at the largest token limit: the CPU readers would take many minutes; the GPU reader of mtmd is the yardstick
        else:
            ref_f = ref_of(name)
            if ref_f is None:
                ref_f, _, res_r = run(r_ref, img, name, 0)
            if a.cpu_dir:
                cpu_f = os.path.join(a.cpu_dir, name + ".sve")
            elif r_cpu:
                cpu_f, _, res_c = run(r_cpu, img, name, 0)
        if r_mg:
            mg_f, mg_ms, res_m = run(r_mg, img, name)
        og_f, og_ms, res_o = run(r_og, img, name)
        old = "-"
        if r_old:
            old_f, _, res_x = run(r_old, img, name, 0)
            old = ("same" if digest(old_f) == digest(og_f) else "DIFFERS") if res_x.startswith("OK") and res_o.startswith("OK") else "ERR"
            same_old.append(old == "same")
        if ref_f is None:
            ref_f = mg_f
        if not (str(res_r).startswith("OK") and str(res_c).startswith("OK") and res_m.startswith("OK") and res_o.startswith("OK")) or og_ms is None or (r_mg and mg_ms is None):
            print("%-22s FAILED: ref=%s mtmd-cpu=%s mtmd=%s own=%s" % (name, res_r, res_c, res_m, res_o), flush=True)
            ok_all = False
            continue
        main = cmp(a.compare, og_f, ref_f)
        floor = cmp(a.compare, mg_f, ref_f) if mg_f else None
        info = cmp(a.compare, og_f, mg_f) if mg_f else None
        info_c = cmp(a.compare, og_f, cpu_f) if cpu_f and os.path.isfile(cpu_f) else None
        if not main or (mg_f and not (floor and info)):
            print("%-22s shapes differ between readers" % name, flush=True)
            ok_all = False
            continue
        m = float(main["global"])
        f = float(floor["global"]) if floor else None
        passed = gate_pass(m, f, a.min_cos, a.skip_cpu)
        ok_all &= passed
        main_c.append(m)
        if f is not None:
            floor_c.append(f)
        if mg_ms:
            ratios.append(og_ms / mg_ms)
        print(hdr % (name, main["n"], main["global"], floor["global"] if floor else "-", info["global"] if info else "-", info_c["global"] if info_c else "-",
                     main["tokmin"], "%.0f" % mg_ms if mg_ms else "-", "%.0f" % og_ms, "%.2f" % (og_ms / mg_ms) if mg_ms else "-", old,
                     "ok" if passed else "FAIL (%s)" % ("main < %.4f" % a.min_cos if m < a.min_cos else "noisier than mtmd")), flush=True)
    if r_mg:
        print("GPU memory of the reader (nvidia-smi / rocm-smi / DRM fdinfo / macOS footprint, peak after the pictures): mtmd %d MiB, own %d MiB%s" % (
            r_mg.peak_mib, r_og.peak_mib, "" if r_og.peak_mib <= r_mg.peak_mib else "  (own is larger)"), flush=True)
    print("REPEATABLE own: %s" % ("yes, every picture byte-identical over %d encodes" % (a.repeat + 1) if not unrepeatable else "NO: " + "; ".join(unrepeatable)),
          flush=True)
    if r_old:
        print("OLD OWN (%s): %d of %d pictures byte-identical" % (a.old_helper, sum(same_old), len(same_old)), flush=True)
    for r in (r_ref, r_cpu, r_mg, r_og, r_old):
        if r:
            r.close()
    if main_c:
        print("SUMMARY own vs float32: %d pictures, min %.6f mean %.6f%s%s" % (
            len(main_c), min(main_c), sum(main_c) / len(main_c),
            "; mtmd vs float32 (the floor): min %.6f mean %.6f" % (min(floor_c), sum(floor_c) / len(floor_c)) if floor_c else "",
            "; time own / mtmd: min %.2f median %.2f max %.2f" % (min(ratios), statistics.median(ratios), max(ratios)) if ratios else ""), flush=True)
        print("RULE %.4f: %s (min %.6f)" % (0.9995, "pass" if min(main_c) >= 0.9995 else "MISS", min(main_c)), flush=True)
    print("RESULT %s" % ("PASS" if ok_all and main_c else "FAIL"), flush=True)
    return 0 if ok_all and main_c else 1


if __name__ == "__main__":
    sys.exit(main())
