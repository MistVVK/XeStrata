#!/usr/bin/env python3
"""Check the loader's tensor conversion (src/split/src/tensor_convert.hpp) against gguf-py.

  convert_dump MODEL.gguf OUTDIR name... > dump.txt
  convert_check.py MODEL.gguf OUTDIR dump.txt --gguf-py /path/to/llama.cpp/gguf-py

For each tensor in the dump: the values the loader holds now (read back by convert_dump) must equal gguf-py's dequantization of the
file's original tensor to within what the target type allows: exactly (F32 target), or within half a Q8_0 step per 32 values
(amax/254, plus the f16 rounding of the scale). Prints one line per tensor and exits 1 on any miss.
"""
import argparse
import os
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

    shards = [a.model]
    import re
    m = re.match(r"(.*-)(\d{5})(-of-)(\d{5})(\.gguf)$", a.model)
    if m:
        shards = ["%s%05d%s%s%s" % (m[1], i, m[3], m[4], m[5]) for i in range(1, int(m[4]) + 1)]
    tensors = {}
    for s in shards:
        r = gguf.GGUFReader(s)
        for t in r.tensors:
            tensors[t.name] = t

    bad = 0
    for line in open(a.dump):
        p = line.split()
        if len(p) != 3 or p[0] == "converted":
            continue
        name, ty = p[0], int(p[1])
        t = tensors[name]
        if t.tensor_type.name == "Q2_0":   # Strata's type 42, not in gguf-py's dequantizers: (code - 1) * d, value j in byte j / 4 at bit 2 (j % 4)
            raw = np.ascontiguousarray(t.data).reshape(-1, 18)
            d = raw[:, :2].copy().view(np.float16).astype(np.float64).reshape(-1, 1)
            qs = raw[:, 2:]
            codes = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(-1, 64)
            ref = ((codes.astype(np.float64) - 1.0) * d).reshape(-1)
        else:
            ref = quants.dequantize(t.data, t.tensor_type).astype(np.float64).reshape(-1)
        got = np.fromfile(os.path.join(a.outdir, name + ".f32"), dtype=np.float32).astype(np.float64)
        if ref.shape != got.shape:
            print("FAIL %s: %d values, gguf-py %d" % (name, got.size, ref.size))
            bad += 1
            continue
        err = np.abs(got - ref)
        if ty == 0:     # F32: BF16 / F16 / F32 source, exact
            lim = np.zeros_like(err)
        elif ty != 8:   # a type the loader keeps or repacks without changing a value (Q2_0 -> Q4_0, IQ4_NL as is): f32 rounding only
            lim = np.abs(ref) * 1e-6 + 1e-12
        else:           # Q8_0: half a step of its block, the scale stored as f16
            blk = ref.reshape(-1, 32)
            amax = np.abs(blk).max(axis=1)
            # the scale d = amax/127 is stored as f16 (relative error 2^-11), and the value is q*d with |q| <= 127
            # (a block whose d falls below f16's normal range, 6.1e-5, has an absolute step of 2^-25 in d: 127 * 2^-25 on the value;
            # 8 of the 19.9M blocks of token_embd are like that)
            lim = np.repeat(amax * (0.5 / 127.0 + 2.0 ** -10) + 127 * 2.0 ** -25 + 1e-12, 32)
        ok = bool((err <= lim).all())
        rel = float(np.sqrt((err ** 2).sum() / max((ref ** 2).sum(), 1e-30)))
        print("%s %s %s  from %s  max|d|=%.3g  rel_rms=%.3g" % ("ok  " if ok else "FAIL", name, {0: "F32", 8: "Q8_0", 2: "Q4_0", 20: "IQ4_NL"}.get(ty, str(ty)), t.tensor_type.name, err.max(), rel))
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
