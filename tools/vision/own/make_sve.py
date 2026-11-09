#!/usr/bin/env python3
"""Runs one strata-vision reader over pictures and keeps the embeddings: <out dir>/<picture name>.sve.

    python3 make_sve.py --helper <strata-vision> --mmproj <mmproj.gguf> --model <text model or x> --tower own|mtmd [--gpu] [--max-tokens 1024] [--threads N]
                        [--env NAME=VALUE ...] --out <dir> <image> [<image> ...]
Used to make the embedding sets the answers runs read (for instance our tower with STRATA_VIT_REF_ROUND=1).  Standard library only.
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--tower", default="own")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    args = [a.helper, "--mmproj", a.mmproj, "--model", a.model, "--tower", a.tower, "--max-tokens", str(a.max_tokens)]
    if a.gpu:
        args.append("--gpu")
    if a.threads and not a.gpu:
        args += ["--threads", str(a.threads)]
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    if not p.stdout.readline().startswith("READY"):
        raise SystemExit("reader did not start")
    bad = 0
    for img in a.images:
        out = os.path.join(a.out, os.path.basename(img) + ".sve")
        p.stdin.write("ENC %s %s\n" % (os.path.abspath(img), out))
        p.stdin.flush()
        r = p.stdout.readline().strip()
        print("%-26s %s" % (os.path.basename(img), r), flush=True)
        bad += 0 if r.startswith("OK") else 1
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
