#!/usr/bin/env python3
"""P4-SPLIT: check that binaries built without llama.cpp's MMQ prompt kernels (-DSTRATA_MMQ=OFF) hold none of them.

usage: check_no_mmq.py [--nm NM] [--cuobjdump CUOBJDUMP] <binary or library> ...

For every file, three looks, each one where the tool for it exists:
  - the raw bytes: the names and messages only MMQ code carries (kernel names stay readable in an uncompressed fatbin or code object, and
    a Windows .exe keeps the strings even without a symbol table);
  - the symbol table (nm -C; on Linux and the Mac): host functions and the device stubs of launched kernels;
  - the device functions of a CUDA binary (cuobjdump -sass, which also opens a compressed fatbin).
Exit 0 when nothing is found, 1 with the hits listed (at most 20 per look), 2 on a usage error.
Strata's own prompt helpers keep the namespace strata::prefill::mmq (gather_native, swiglu, iota, ...): that name is not a marker."""
import argparse
import os
import re
import shutil
import subprocess
import sys

# llama.cpp's MMQ: the kernels and their templates (mul_mat_q, mul_mat_q_stream_k_fixup, load_tiles_*, vec_dot_*_mma / _dp4a), the q8_1
# quantizers (quantize_mmq_q8_1, quantize_scatter_mmq_q8_1), ggml-cuda's host side (ggml_cuda_*, ggml_backend_cuda_context) and Strata's MMQ
# glue (ggml_cuda_host.cu's error text, the K-quant library's file name)
RAW = [b"mul_mat_q", b"quantize_mmq_q8_1", b"quantize_scatter_mmq", b"ggml_cuda_", b"ggml_backend_cuda", b"ggml-cuda (strata mmq)",
       b"libstrata_kq", b"strata_kq.dll", b"load_tiles_", b"mmq_get_"]
SYM = re.compile(r"mul_mat_q|quantize_mmq_q8_1|quantize_scatter_mmq|ggml_cuda_|ggml_backend_cuda|load_tiles_|mmq_get_|vec_dot_\w+_(mma|dp4a)")


def raw_hits(path):
    data = open(path, "rb").read()
    hits = []
    for m in RAW:
        n = data.count(m)
        if n:
            hits.append("%s x%d" % (m.decode(), n))
    return hits


def tool_lines(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    return r.stdout.splitlines(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", default=shutil.which("nm"))
    ap.add_argument("--cuobjdump", default=shutil.which("cuobjdump"))
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    bad = 0
    for f in a.files:
        if not os.path.isfile(f):
            print("%s: missing" % f)
            return 2
        looks = []
        h = raw_hits(f)
        looks.append(("bytes", h))
        if a.nm:
            lines, err = tool_lines([a.nm, "-C", f])
            if lines is not None:
                looks.append(("nm", sorted({ln.split(None, 2)[-1] for ln in lines if SYM.search(ln)})))
        if a.cuobjdump:
            lines, err = tool_lines([a.cuobjdump, "-sass", f])
            if lines is not None:
                names = [m.group(1) for ln in lines for m in [re.match(r"\s*Function : (\S+)", ln)] if m]
                if names:
                    looks.append(("cuobjdump (%d device functions)" % len(names), sorted({n for n in names if SYM.search(n)})))
        found = [(k, v) for k, v in looks if v]
        print("%s: %s" % (f, "; ".join("%s %s" % (k, "CLEAN" if not v else "%d HITS" % len(v)) for k, v in looks)))
        for k, v in found:
            for x in v[:20]:
                print("    %s: %s" % (k, x[:200]))
        bad += bool(found)
    print("no MMQ symbol in %d of %d files" % (len(a.files) - bad, len(a.files)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
