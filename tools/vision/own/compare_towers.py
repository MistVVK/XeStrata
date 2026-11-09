#!/usr/bin/env python3
"""Runs strata-vision twice (--tower A and --tower B, default mtmd and own) over the same pictures and compares the embeddings.

    python compare_towers.py --helper <strata-vision> --compare <strata-vision-own-compare> --mmproj <mmproj.gguf> --model <text model .gguf>
                             --out <dir> [--a mtmd] [--b own] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]
                             [--repeat K] [--env NAME=VALUE ...] <image> [<image> ...]

Per picture: the tokens, the time of each tower (the first encode, then --repeat K further encodes of the same picture, median reported), and the
cosine of the two embedding files (global = both files as one vector; tokmin / tokmean = per-token cosines), max |difference|.
The last lines are the summary the gate asks for: minimum and mean cosine over the pictures.  Exit 0 when every picture was read by both and
the minimum global cosine is at least --min-cos (default 0.999); 1 otherwise.  Standard library only.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time


class Reader:
    def __init__(self, exe, tower, a, gpu, label):
        args = [exe, "--mmproj", a.mmproj, "--model", a.model, "--tower", tower]
        if gpu:
            args.append("--gpu")
        if a.threads:
            args += ["--threads", str(a.threads)]
        if a.max_tokens:
            args += ["--max-tokens", str(a.max_tokens)]
        if a.min_tokens:
            args += ["--min-tokens", str(a.min_tokens)]
        env = dict(os.environ)
        for kv in a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        self.tower = tower
        self.t0 = time.time()
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(a.out, label + ".stderr"), "w"),
                                  text=True, env=env)
        line = self.p.stdout.readline().strip()
        self.ready = line.startswith("READY")
        self.start_s = time.time() - self.t0
        if not self.ready:
            raise SystemExit("%s reader did not start: %r" % (tower, line))

    def enc(self, img, out):
        self.p.stdin.write("ENC %s %s\n" % (img, out))
        self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", default="mtmd")
    ap.add_argument("--b", default="own")
    ap.add_argument("--a-dir", default="", help="A's embeddings from an earlier run: <dir>/<picture name>.sve (no reader is started for A)")
    ap.add_argument("--helper-b", default="", help="another program for B (e.g. strata-vision-own); default: --helper")
    ap.add_argument("--gpu", action="store_true", help="both readers on the GPU")
    ap.add_argument("--gpu-a", action="store_true")
    ap.add_argument("--gpu-b", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    la = a.a + ("-gpu" if a.gpu or a.gpu_a else "")
    lb = a.b + ("-gpu" if a.gpu or a.gpu_b else "")
    if la == lb:
        lb += "-2"
    for t in (la, lb):
        os.makedirs(os.path.join(a.out, t), exist_ok=True)

    # --a-dir: A's embeddings were made earlier (<dir>/<picture name>.sve, from a run of the same limits); no reader is started for A
    ra = None if a.a_dir else Reader(a.helper, a.a, a, a.gpu or a.gpu_a, la)
    rb = Reader(a.helper_b or a.helper, a.b, a, a.gpu or a.gpu_b, lb)
    if ra:
        ra.label = la
    rb.label = lb
    print("start: %s %.1f s, %s %.1f s" % (la, ra.start_s if ra else 0.0, lb, rb.start_s), flush=True)
    print("%-24s %11s %6s %9s %9s  %9s %9s %9s %10s" % ("picture", "size", "tokens", la + " ms", lb + " ms", "global", "tokmin", "tokmean", "maxabs"), flush=True)
    glob, tmin, ok_all = [], [], True
    for img in a.images:
        name = os.path.basename(img)
        outs, res, times = {}, {}, {}
        for r in (ra, rb):
            if r is None:
                outs[la] = os.path.join(a.a_dir, name + ".sve")
                res[la] = "OK" if os.path.isfile(outs[la]) else "ERR no file " + outs[la]
                times[la] = float("nan")
                continue
            outs[r.label] = os.path.join(a.out, r.label, name + ".sve")
            res[r.label] = r.enc(os.path.abspath(img), outs[r.label])
            ms = [float(res[r.label].split()[4])] if res[r.label].startswith("OK") else []
            for _ in range(a.repeat if ms else 0):
                again = r.enc(os.path.abspath(img), outs[r.label])
                if again.startswith("OK"):
                    ms.append(float(again.split()[4]))
            times[r.label] = statistics.median(ms[1:] if len(ms) > 1 else ms) if ms else float("nan")
        if not (res[la].startswith("OK") and res[lb].startswith("OK")):
            print("%-24s FAILED: %s: %s | %s: %s" % (name, la, res[la], lb, res[lb]), flush=True)
            ok_all = False
            continue
        c = subprocess.run([a.compare, outs[la], outs[lb]], capture_output=True, text=True).stdout.strip()
        if not c.startswith("n="):
            print("%-24s %s" % (name, c), flush=True)
            ok_all = False
            continue
        kv = dict(p.split("=") for p in c.split())
        size = "%sx%s" % (kv["nx"], kv["ny"])
        glob.append(float(kv["global"]))
        tmin.append(float(kv["tokmin"]))
        print("%-24s %11s %6s %9.0f %9.0f  %9s %9s %9s %10s" % (name, size, kv["n"], times[la], times[lb], kv["global"], kv["tokmin"], kv["tokmean"], kv["maxabs"]), flush=True)
    if ra:
        ra.close()
    rb.close()
    if glob:
        print("SUMMARY %s vs %s: %d pictures, global cosine min %.6f mean %.6f, smallest per-token cosine %.6f" % (lb, la, len(glob), min(glob), sum(glob) / len(glob), min(tmin)))
    good = ok_all and glob and min(glob) >= a.min_cos
    print("RESULT %s (minimum cosine %s %.4f)" % ("PASS" if good else "FAIL", ">=" if good else "below or errors,", a.min_cos), flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
