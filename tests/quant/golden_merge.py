"""tests/quant/golden_merge.py - write tests/quant/golden_x86.txt from `quant_parity --write-golden` fragments (tests/quant/make_golden.sh runs it).

    python tests/quant/golden_merge.py --gcc gcc-avx2.txt gcc-avx.txt gcc-none.txt --msvc msvc-avx2.txt msvc-avx.txt msvc-none.txt -o tests/quant/golden_x86.txt

Family gcc: every group of the GCC 13.x fragments.  Family msvc: only the groups whose MSVC 19.4x reference differs from the gcc line; each needs a reason
in WHY below (the merge refuses an unexplained difference, and any group whose INPUTS differ between the toolchains: that is a test bug, not arithmetic).
"""
import argparse
import datetime
import os
import re
import sys

# why a family has its own line for (flavour, section, type), with what was measured (OWN-2A-ORACLE, 11-10: MSVC 19.44 on the owner PC against GCC 13.3 on p1004)
WHY = {
    ("msvc", "avx2", "dot", "q5_K"): "GCC contracts ggml's _mm256_add_ps(_mm256_mul_ps(a, b), c) here into an FMA and MSVC does not; ggml and the layer agree "
                                     "within each toolchain; measured 1-512 ulp (max relative 3.3e-5), n = 512..2048",
    ("msvc", "avx2", "dot", "iq1_s"): "GCC contracts ggml's add-of-mul intrinsics here into an FMA and MSVC does not; ggml and the layer agree within each toolchain; "
                                      "measured 1-16 ulp (max relative 1.6e-6), n = 512..2048",
}
ORDER = ["avx2", "avx", "generic"]


def read_fragment(path):
    made, lines = None, []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\r\n")
            if ln.startswith("# made by "):
                made = ln[len("# made by "):]
            elif ln and not ln.startswith("#"):
                p = ln.split()
                if len(p) != 7:
                    sys.exit("%s: malformed line: %s" % (path, ln))
                lines.append(p)
    if made is None or not lines:
        sys.exit("%s: not a quant_parity --write-golden file" % path)
    return made, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcc", nargs="+", required=True)
    ap.add_argument("--msvc", nargs="+", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    pin = "unknown"
    up = os.path.join(here, "..", "..", "third_party", "strata-ggml", "UPSTREAM.md")
    if os.path.exists(up):
        m = re.search(r"^pin: ([0-9a-f]{40})", open(up, encoding="utf-8").read(), re.M)
        pin = m.group(1) if m else pin
    gcc, msvc, made = {}, {}, {"gcc": set(), "msvc": set()}
    for fam, files, into in (("gcc", a.gcc, gcc), ("msvc", a.msvc, msvc)):
        for path in files:
            who, lines = read_fragment(path)
            if fam == "gcc" and not re.match(r"GCC 13\.", who):
                sys.exit("%s: family gcc needs GCC 13.x, this fragment is made by %s" % (path, who))
            if fam == "msvc" and not re.match(r"MSVC 194", who):
                sys.exit("%s: family msvc needs MSVC 19.4x, this fragment is made by %s" % (path, who))
            made[fam].add(who.split(",")[0])
            fl = lines[0][0]
            into[fl] = lines
    out = []
    head = [
        "# tests/quant/golden_x86.txt - the reference digests of the quant layer's oracle test, leg 1 (tests/quant/quant_parity.cpp says how they are checked).",
        "# Line: <family> <flavour> <section> <type> <n> <rows> <input digest> <reference digest>; digests: golden_fnv1a64 (FNV-1a, 64 bit) in quant_parity.cpp.",
        "# family gcc: ggml-cpu built by %s, every group; every toolchain is held to these lines unless it has its own." % ", ".join(sorted(made["gcc"])),
        "# family msvc: ggml-cpu built by %s, only the groups where its reference differs from the gcc line (%%note says why)." % ", ".join(sorted(made["msvc"])),
        "# ggml: third_party/strata-ggml, llama.cpp pin %s (UPSTREAM.md)." % pin,
        "# made by: tests/quant/make_golden.sh (GCC side, quant_parity --write-golden per flavour) + quant_parity --write-golden under MSVC + tests/quant/golden_merge.py",
        "# date: %s" % a.date,
    ]
    for fl in ORDER:
        if fl not in gcc or fl not in msvc:
            sys.exit("flavour %s is missing from the gcc or the msvc fragments" % fl)
        g, m = gcc[fl], msvc[fl]
        gk = {tuple(p[:4]): p for p in g}
        mk = {tuple(p[:4]): p for p in m}
        if set(gk) != set(mk):
            sys.exit("flavour %s: the toolchains ran different groups: %s" % (fl, sorted(set(gk) ^ set(mk))[:10]))
        own = []
        for p in g:
            q = mk[tuple(p[:4])]
            if q[4] != p[4] or q[5] != p[5]:
                sys.exit("flavour %s group %s: the INPUTS differ between GCC and MSVC (a test bug, not arithmetic)" % (fl, " ".join(p[:4])))
            if q[6] != p[6]:
                if ("msvc", fl, p[1], p[2]) not in WHY:
                    sys.exit("flavour %s group %s: MSVC's reference differs and WHY has no reason for it" % (fl, " ".join(p[:4])))
                own.append(q)
        rows = sum(int(p[4]) for p in g)
        head.append("# flavour %s: %d groups, %d rows; family msvc: %d own lines" % (fl, len(g), rows, len(own)))
        out += ["gcc " + " ".join(p) for p in g]
        for key in sorted({(p[1], p[2]) for p in own}):
            out.append("%%note msvc %s %s %s %s" % (fl, key[0], key[1], WHY[("msvc", fl) + key]))
        out += ["msvc " + " ".join(q) for q in own]
    with open(a.o, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(head + out) + "\n")
    print("golden_merge: wrote %s (%d lines)" % (a.o, len(head) + len(out)))


if __name__ == "__main__":
    main()
