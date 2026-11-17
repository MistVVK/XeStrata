#!/usr/bin/env python3
"""Check the prompt kernel's weight tiles against gguf-py.

  moe_tile_test MODEL.gguf OUTDIR > tiles.txt
  tile_check.py MODEL.gguf OUTDIR tiles.txt --gguf-py /path/to/gguf-py

For each line of the dump (tensor, loader type, rows, columns, expert): the f16 values the tile put in shared memory must equal gguf-py's
dequantization of the file's expert, rounded once to f16 (a tile value may differ by one f16 step where the two compute the f32 product in
another order). Strata's Q2_0 (type 42; the loader repacks it to Q4_0 blocks, type 2) is checked against (code - 1) * d read from the
file. Prints one line per tensor and expert; exit 1 on any miss.
"""
import argparse
import os
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("outdir")
    ap.add_argument("dump")
    ap.add_argument("--gguf-py", help="optional: a folder that holds llama.cpp's gguf package, to compare against it (default: tools/strata_gguf.py)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
    if a.gguf_py:
        os.environ["STRATA_GGUF_PY"] = a.gguf_py
    from _paths import gguf_lib
    gguf = gguf_lib()
    quants = gguf.quants

    m = re.match(r"(.*-)(\d{5})(-of-)(\d{5})(\.gguf)$", a.model)
    shards = ["%s%05d%s%s%s" % (m[1], i, m[3], m[4], m[5]) for i in range(1, int(m[4]) + 1)] if m else [a.model]
    tensors = {}
    for s in shards:
        for t in gguf.GGUFReader(s).tensors:
            tensors[t.name] = t

    bad = 0
    for line in open(a.dump):
        p = line.split()
        if len(p) != 5 or not p[1].isdigit():
            continue
        name, ty, nrows, ncols, e = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
        t = tensors[name]
        raw = np.asarray(t.data)[e]
        if t.tensor_type.name == "Q2_0":
            raw = np.ascontiguousarray(raw).reshape(-1, 18)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
            codes = np.stack([(raw[:, 2:] >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(-1, 64)
            ref = ((codes.astype(np.float32) - 1.0) * d).reshape(nrows, ncols)
        else:
            ref = quants.dequantize(raw, t.tensor_type).astype(np.float32).reshape(nrows, ncols)
        ref16 = ref.astype(np.float16)
        got = np.fromfile(os.path.join(a.outdir, "%s.e%d.f16" % (name, e)), dtype=np.float16).reshape(nrows, ncols)
        gi, ri = got.view(np.int16).astype(np.int32), ref16.view(np.int16).astype(np.int32)
        # f16 ordering: compare as sign-magnitude steps
        def key(x):
            return np.where(x < 0, -(x & 0x7FFF), x)
        step = np.abs(key(gi) - key(ri))
        exact = float((step == 0).mean())
        ok = bool(step.max() <= 1 and exact >= 0.999 and not np.isnan(got.astype(np.float32)).any())
        print("%s %-44s %-8s e%-3d exact %.5f  max step %d  max|d| %.3g" % ("ok  " if ok else "FAIL", name, t.tensor_type.name, e, exact, step.max(),
                                                                          float(np.abs(got.astype(np.float32) - ref).max())))
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
