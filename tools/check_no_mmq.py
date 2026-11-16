#!/usr/bin/env python3
"""P4-SPLIT: check that binaries built without llama.cpp's MMQ prompt kernels (-DSTRATA_MMQ=OFF) hold none of them.

usage: check_no_mmq.py [--nm NM] [--cuobjdump CUOBJDUMP] [--tool-timeout S] <binary or library> ...

For every file, three looks, each one where the tool for it exists:
  - the raw bytes: the names and messages only MMQ code carries (kernel names stay readable in an uncompressed fatbin or code object, and
    a Windows .exe keeps the strings even without a symbol table);
  - the symbol table (nm -C; on Linux and the Mac): host functions and the device stubs of launched kernels;
  - the device functions of a CUDA binary (cuobjdump -sass, which also opens a compressed fatbin).
A look whose tool is there but does not run to the end (missing file, a crash, the time limit, a cuobjdump too old for the binary, which
reads no device function) is printed as NOT RUN with the reason; it is never left out of the line in silence.
Exit 0 when nothing is found, 1 with the hits listed (at most 20 per look), 2 on a usage error, 3 when nothing is found but a look whose tool
was named on the command line (--nm, --cuobjdump) did not run: that file was not fully checked.
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
NO_DEVICE_CODE = "does not contain device code"     # cuobjdump on a file without CUDA code: nothing to look at, not a failure


def raw_hits(path):
    data = open(path, "rb").read()
    hits = []
    for m in RAW:
        n = data.count(m)
        if n:
            hits.append("%s x%d" % (m.decode(), n))
    return hits


def tool_lines(cmd, timeout=900):
    """(stdout lines, None), or (None, why the tool did not run to the end)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timed out after %g s" % timeout
    except OSError as e:
        return None, "could not start %s (%s)" % (cmd[0], e.strerror or e)
    if r.returncode != 0 and not r.stdout.strip() and NO_DEVICE_CODE not in r.stderr:
        first = next((ln.strip() for ln in (r.stderr + "\n" + r.stdout).splitlines() if ln.strip()), "")
        return None, "exit %d%s" % (r.returncode, ": " + first[:200] if first else "")
    return r.stdout.splitlines() + ([NO_DEVICE_CODE] if NO_DEVICE_CODE in r.stderr else []), None


def device_look(cuobjdump, path, timeout=900):
    """("cuobjdump ...", hits) for a look that ran, or (None, why) for one that did not."""
    lines, err = tool_lines([cuobjdump, "-sass", path], timeout)
    if lines is None:
        return None, err
    names = [m.group(1) for ln in lines for m in [re.match(r"\s*Function : (\S+)", ln)] if m]
    if not names:
        if any(NO_DEVICE_CODE in ln for ln in lines):
            return "cuobjdump (no device code)", []
        return None, "it read no device function (a cuobjdump older than the binary's CUDA?)"
    return "cuobjdump (%d device functions)" % len(names), sorted({n for n in names if SYM.search(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", default=None)
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--tool-timeout", type=float, default=900)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    named = {"nm": bool(a.nm), "cuobjdump": bool(a.cuobjdump)}         # named on the command line: a look that must run
    nm = a.nm if a.nm is not None else shutil.which("nm")
    cuobjdump = a.cuobjdump if a.cuobjdump is not None else shutil.which("cuobjdump")
    bad = unchecked = 0
    for f in a.files:
        if not os.path.isfile(f):
            print("%s: missing" % f)
            return 2
        looks, skipped = [], []
        h = raw_hits(f)
        looks.append(("bytes", h))
        if nm:
            lines, err = tool_lines([nm, "-C", f], a.tool_timeout)
            if lines is not None:
                looks.append(("nm", sorted({ln.split(None, 2)[-1] for ln in lines if SYM.search(ln)})))
            else:
                skipped.append(("nm", err))
        if cuobjdump:
            name, result = device_look(cuobjdump, f, a.tool_timeout)
            if name is not None:
                looks.append((name, result))
            else:
                skipped.append(("cuobjdump", result))
        found = [(k, v) for k, v in looks if v]
        parts = ["%s %s" % (k, "CLEAN" if not v else "%d HITS" % len(v)) for k, v in looks]
        parts += ["%s NOT RUN (%s)" % (k, why) for k, why in skipped]
        print("%s: %s" % (f, "; ".join(parts)))
        for k, v in found:
            for x in v[:20]:
                print("    %s: %s" % (k, x[:200]))
        bad += bool(found)
        unchecked += any(named[k] for k, _ in skipped)
    print("no MMQ symbol in %d of %d files" % (len(a.files) - bad, len(a.files)))
    if unchecked and not bad:
        print("%d of %d files not fully checked: a look named on the command line did not run" % (unchecked, len(a.files)))
    return 1 if bad else 3 if unchecked else 0


if __name__ == "__main__":
    sys.exit(main())
