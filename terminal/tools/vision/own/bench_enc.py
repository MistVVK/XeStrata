#!/usr/bin/env python3
"""Times one reader on one picture: image-to-embedding time (median of --repeat encodes after the first) and the reader's GPU memory.

    python3 bench_enc.py --helper <strata-vision or strata-vision-own> --mmproj <mmproj.gguf> --model <text model or x> --image <picture> [--tower own]
                         [--gpu] [--max-tokens 1024] [--repeat 8] [--out <file for the embeddings>] [--env NAME=VALUE ...]
Prints one line: label, tokens, median ms, min ms, peak GPU MiB (nvidia-smi, by pid).  Used to compare settings of the GPU tower (STRATA_VIT_ROWS, STRATA_VIT_SCORES).
"""
import argparse
import os
import statistics
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--image", required=True)
    ap.add_argument("--tower", default="own")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--out", default="/tmp/bench_enc.sve")
    ap.add_argument("--label", default="")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    args = [a.helper, "--mmproj", a.mmproj, "--model", a.model, "--tower", a.tower, "--max-tokens", str(a.max_tokens)]
    if a.gpu:
        args.append("--gpu")
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    if not p.stdout.readline().startswith("READY"):
        raise SystemExit("reader did not start")
    ms, tokens, peak = [], "?", 0
    for i in range(a.repeat + 1):
        p.stdin.write("ENC %s %s\n" % (os.path.abspath(a.image), a.out))
        p.stdin.flush()
        r = p.stdout.readline().split()
        if not r or r[0] != "OK":
            raise SystemExit("encode failed: %s" % " ".join(r))
        tokens = r[1]
        if i:
            ms.append(float(r[4]))
        try:
            smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
            for line in smi.splitlines():
                pid, _, mem = line.partition(",")
                if pid.strip() == str(p.pid) and mem.strip().isdigit():
                    peak = max(peak, int(mem.strip()))
        except Exception:  # noqa: BLE001
            pass
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=30)
    print("%-28s tokens %s  median %.0f ms  min %.0f ms  peak %d MiB" % (a.label or " ".join(a.env) or "default", tokens, statistics.median(ms), min(ms), peak))


if __name__ == "__main__":
    main()
