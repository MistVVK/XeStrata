#!/usr/bin/env python3
"""msl_text_check.py - are two builds' kernel libraries the same text?

    STRATA_METAL_MSL_DUMP=<dir> strata ...        (any program that compiles kernels through mtl::Context: the engine, a parity binary)
    python3 metal/tools/msl_text_check.py <dumpdir A> <dumpdir B> [--preprocess "clang -E -P -x c++ -nostdinc -undef"] [--air]

The dump holds one NNN.msl per library (the prelude, <metal_stdlib>, ggml-common, then every kernel file: the exact string handed to
the compiler) and index.txt (NNN.msl: the kernel files of that library).  Libraries are matched by their file list, in order.

For each pair the tool reports, strongest first:
  raw        the two strings are byte-identical (sha256 equal)
  pp         not identical, but the preprocessed texts are (the preprocessor's output with comments, blank lines and runs of blanks
             dropped): a change of comments, #ifndef guards or a macro that expands to the number it replaced
  air        (--air, needs `xcrun metal`) the two compile to byte-identical AIR (same code, whatever the text)
  DIFFERENT  none of the above
Exit code 0 when every pair is raw or pp (or air with --air), 1 otherwise.
"""
import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile


def read_index(d):
    out = []
    p = os.path.join(d, "index.txt")
    if not os.path.exists(p):
        sys.exit("no index.txt in " + d)
    for ln in open(p):
        ln = ln.rstrip("\n")
        if not ln:
            continue
        name, _, files = ln.partition(":")
        out.append((name.strip(), tuple(os.path.basename(f) for f in files.split())))   # (the engine passes absolute paths, the tests relative ones)
    return out


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def preprocess(path, cmd):
    # The C preprocessor of clang over the text, the <...> includes left out: none of the kernel sources branches on a Metal macro, so the
    # macros they and the host's prelude define are all there is to expand (`xcrun metal -E` needs the Metal toolchain component).
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as t:
        data = open(path, "rb").read().decode("utf-8", "replace")
        data = re.sub(r"(?m)^[ 	]*#[ 	]*include[ 	]*<[^>]*>", "", data)   # <metal_stdlib>, <MetalPerformancePrimitives/...>
        t.write(data.encode())
        tmp = t.name
    try:
        r = subprocess.run(shlex.split(cmd) + [tmp], capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr[-400:]
        txt = r.stdout
    finally:
        os.unlink(tmp)
    out = []
    for ln in txt.splitlines():
        if ln.startswith("#"):
            continue   # line markers (the file names differ by temp path)
        ln = re.sub(r"//.*$", "", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            out.append(ln)
    return "\n".join(out), ""


def air(path, tool):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "x.metal")
        open(src, "wb").write(open(path, "rb").read())
        o = os.path.join(td, "x.air")
        r = subprocess.run(shlex.split(tool) + ["-c", src, "-o", o], capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr[-400:]
        return hashlib.sha256(open(o, "rb").read()).hexdigest(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--preprocess", default="clang -E -P -x c++ -nostdinc -undef")
    ap.add_argument("--air", action="store_true")
    ap.add_argument("--air-tool", default="xcrun -sdk macosx metal -std=metal3.2")
    a = ap.parse_args()
    ia, ib = read_index(a.a), read_index(a.b)
    used = set()
    bad = 0
    for na, fa in ia:
        match = None
        for nb, fb in ib:
            if nb not in used and fb == fa:
                match = nb
                break
        if match is None:
            print("%s %s: no library with this file list in B" % (na, " ".join(fa)))
            bad += 1
            continue
        used.add(match)
        pa, pb = os.path.join(a.a, na), os.path.join(a.b, match)
        if sha(pa) == sha(pb):
            print("raw        %s %s  (%d bytes)" % (na, " ".join(fa), os.path.getsize(pa)))
            continue
        ta, ea = preprocess(pa, a.preprocess)
        tb, eb = preprocess(pb, a.preprocess)
        if ta is None or tb is None:
            print("pp-ERROR   %s %s: %s %s" % (na, " ".join(fa), ea, eb))
            bad += 1
            continue
        if ta == tb:
            print("pp         %s %s  (raw differs: %d vs %d bytes; preprocessed %d bytes identical)" %
                  (na, " ".join(fa), os.path.getsize(pa), os.path.getsize(pb), len(ta)))
            continue
        if a.air:
            xa, _ = air(pa, a.air_tool)
            xb, _ = air(pb, a.air_tool)
            if xa and xa == xb:
                print("air        %s %s  (AIR identical)" % (na, " ".join(fa)))
                continue
        la, lb = ta.splitlines(), tb.splitlines()
        first = next((i for i in range(min(len(la), len(lb))) if la[i] != lb[i]), min(len(la), len(lb)))
        print("DIFFERENT  %s %s: first differing preprocessed line %d\n   A: %s\n   B: %s" %
              (na, " ".join(fa), first, la[first][:200] if first < len(la) else "<end>", lb[first][:200] if first < len(lb) else "<end>"))
        bad += 1
    for nb, fb in ib:
        if nb not in used:
            print("%s %s: only in B" % (nb, " ".join(fb)))
            bad += 1
    print("libraries: A %d, B %d; %s" % (len(ia), len(ib), "all the same text" if bad == 0 else "%d differ" % bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
