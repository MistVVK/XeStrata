#!/usr/bin/env python3
"""Alternating-pairs speed test of two picture readers of the same helper (default: our fast CPU tower against mtmd): per picture one warm-up encode of each, then N pairs
in the order A, B, A, B, ... so that drift of the machine (clocks, other jobs) hits both the same way.  Reports the median, the minimum and the maximum of each (the `ms` the
helper prints) and the ratio of the medians; also the ratio of every single pair (min / median / max), the spread that decides whether a difference is real.

    python3 speed_pairs.py --helper <strata-vision> --mmproj <mmproj> --model <text model> [--threads N] [--pairs 5] [--max-tokens 1024]
                           [--a own --env-a STRATA_VIT_CPU=fast] [--b mtmd] <image> ...
"""
import argparse
import os
import statistics
import subprocess
import sys


def start(helper, a, tower, envs, label):
    args = [helper, "--mmproj", a.mmproj, "--model", a.model, "--tower", tower, "--max-tokens", str(a.max_tokens)]
    if a.threads:
        args += ["--threads", str(a.threads)]
    env = dict(os.environ)
    for kv in envs:
        k, _, v = kv.partition("=")
        env[k] = v
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    if not p.stdout.readline().startswith("READY"):
        raise SystemExit("%s reader did not start" % label)
    return p


def enc(p, img, out):
    p.stdin.write("ENC %s %s\n" % (img, out))
    p.stdin.flush()
    r = p.stdout.readline().split()
    if not r or r[0] != "OK":
        raise SystemExit("encode failed: %s" % " ".join(r))
    return float(r[4]), int(r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--a", default="own")
    ap.add_argument("--b", default="mtmd")
    ap.add_argument("--env-a", action="append", default=["STRATA_VIT_CPU=fast"])
    ap.add_argument("--env-b", action="append", default=[])
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    pa = start(a.helper, a, a.a, a.env_a, "A")
    pb = start(a.helper, a, a.b, a.env_b, "B")
    print("A = %s %s, B = %s %s, %s threads, %d alternating pairs after one warm-up each (ms of the helper's OK line)" % (a.a, a.env_a, a.b, a.env_b, a.threads or "default", a.pairs))
    print("%-24s %6s | %-26s | %-26s | %6s | %s" % ("picture", "tokens", "A median (min..max)", "B median (min..max)", "A/B", "pair ratios min / median / max"))
    worst = 0.0
    for img in a.images:
        name = os.path.basename(img)
        oa, ob = os.path.join(a.tmp, "sp_a_" + name + ".sve"), os.path.join(a.tmp, "sp_b_" + name + ".sve")
        _, n = enc(pa, img, oa)
        enc(pb, img, ob)
        ta, tb = [], []
        for _ in range(a.pairs):
            ta.append(enc(pa, img, oa)[0])
            tb.append(enc(pb, img, ob)[0])
        ra = [x / y for x, y in zip(ta, tb)]
        ma, mb = statistics.median(ta), statistics.median(tb)
        worst = max(worst, ma / mb)
        print("%-24s %6d | %8.0f (%6.0f..%6.0f) | %8.0f (%6.0f..%6.0f) | %6.3f | %.3f / %.3f / %.3f" %
              (name, n, ma, min(ta), max(ta), mb, min(tb), max(tb), ma / mb, min(ra), statistics.median(ra), max(ra)), flush=True)
        for f in (oa, ob):
            try:
                os.remove(f)
            except OSError:
                pass
    print("largest A/B ratio of medians: %.3f" % worst)
    for p in (pa, pb):
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=60)


if __name__ == "__main__":
    sys.exit(main())
