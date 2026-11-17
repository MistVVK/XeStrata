#!/usr/bin/env python3
"""geo_pins.py - the Metal sources' literals that are Flash-Next's shape, counted per file at two git revisions.

    python metal/tools/geo_pins.py [REV_BEFORE [REV_AFTER]]         (default: the commit before Q4-METAL's geometry work, and HEAD)
    python metal/tools/geo_pins.py --list REV                       every remaining hit of REV, one line each

A "pin" is a code token (comments and string literals are dropped before matching) from the list below.  Specific numbers are the values only
Flash-Next's shape has; ambiguous ones (48 = layers or GDN heads, 24 = query heads, 36, 33, 65) are counted separately because the same digits
also mean thread counts and tile sizes elsewhere.  The tool cannot tell a hit that is a shape from one that is not: the classes of the
remaining hits are in phaseA-tests/qwen4/RESULTS-Q4-METAL.md, section Inventory.
"""
import re
import subprocess
import sys

SPECIFIC = [2560, 10240, 6144, 12288, 2051, 16480, 1280, 640, 320, 248320]
AMBIG = [48, 36, 24, 33, 65, 258]
FILES = ["metal/src/engine/window.cpp", "metal/src/engine/prompt_chain.inc", "metal/src/engine/stream.cpp", "metal/src/engine/dflash2.inc",
         "metal/src/engine/iq_host.cpp", "metal/src/engine/mmvq_host.cpp",
         "metal/kernels/qsa.metal", "metal/kernels/qsa2.metal", "metal/kernels/qsab.metal", "metal/kernels/qsa_prompt.metal",
         "metal/kernels/gdn.metal", "metal/kernels/gdn_prompt.metal", "metal/kernels/gr.metal", "metal/kernels/gr2.metal",
         "metal/kernels/hcf.metal", "metal/kernels/glue.metal", "metal/kernels/router.metal", "metal/kernels/moe.metal",
         "metal/kernels/moe_mm.metal", "metal/kernels/prompt.metal", "metal/kernels/sample.metal", "metal/kernels/gemv.metal",
         "metal/src/engine/msl/ple.metal", "metal/src/engine/msl/mtp.metal", "metal/src/engine/msl/mtp_chain.metal", "metal/src/engine/msl/dflash2.metal"]


def show(rev, path):
    r = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def code_lines(text):
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        ln = re.sub(r'"(?:\\.|[^"\\])*"', '""', ln)
        ln = re.sub(r"//.*$", "", ln)
        out.append((i, ln))
    return out


def count(rev, path, nums):
    pat = re.compile(r"(?<![\w.])(%s)(?![\w.])" % "|".join(str(n) for n in nums))
    n = 0
    hits = []
    for i, ln in code_lines(show(rev, path)):
        m = pat.findall(ln)
        if m:
            n += len(m)
            hits.append((i, ln.strip()))
    return n, hits


def main():
    args = sys.argv[1:]
    if args and args[0] == "--list":
        rev = args[1]
        for f in FILES:
            for tier, nums in (("specific", SPECIFIC), ("ambiguous", AMBIG)):
                n, hits = count(rev, f, nums)
                for i, ln in hits:
                    print("%s:%d [%s] %s" % (f, i, tier, ln[:170]))
        return
    before = args[0] if args else "f6a3b0ce8"
    after = args[1] if len(args) > 1 else "HEAD"
    print("%-44s %17s %17s" % ("file", "specific %s->%s" % (before[:7], after[:7]), "ambiguous"))
    ts = [0, 0, 0, 0]
    for f in FILES:
        a1, _ = count(before, f, SPECIFIC)
        b1, _ = count(after, f, SPECIFIC)
        a2, _ = count(before, f, AMBIG)
        b2, _ = count(after, f, AMBIG)
        if a1 or b1 or a2 or b2:
            print("%-44s %7d -> %-6d %7d -> %-6d" % (f, a1, b1, a2, b2))
        ts = [ts[0] + a1, ts[1] + b1, ts[2] + a2, ts[3] + b2]
    print("%-44s %7d -> %-6d %7d -> %-6d" % ("total", ts[0], ts[1], ts[2], ts[3]))


if __name__ == "__main__":
    main()
