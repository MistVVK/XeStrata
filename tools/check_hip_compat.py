"""tools/check_hip_compat.py - every CUDA runtime / cuBLAS name the AMD (HIP) build compiles must have a mapping in
include/strata/hip_compat/*.h.  The HIP build compiles the same CUDA-shaped sources with those headers forced in
(cmake/hip_backend.cmake), so a CUDA call added without its mapping breaks every AMD build - v1.0.21 did, with three
device queries from #44 (fixed in v1.0.22 by #52).  No ROCm or GPU is needed: it reads the sources.

    python tools/check_hip_compat.py [repo]      # exit 1 and the file:line of each unmapped name

Names inside CUDA-only code (#ifndef STRATA_USE_HIP, #if !defined(STRATA_USE_HIP), the #else of a STRATA_USE_HIP
block), comments and string literals are not counted.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NAME = re.compile(r"\b(cuda[A-Z]\w*|cublas[A-Z]\w*|CUBLAS_[A-Z0-9_]+|CUDA_R_\w+)\b")
SOURCES = ("*.cu", "*.cuh", "*.cpp", "*.hpp", "*.h")


def mapped_names(compat: Path) -> set:
    """What the compat headers define: #define NAME, and the inline functions/templates named like CUDA's."""
    out = set()
    for h in compat.glob("*.h"):
        text = h.read_text(encoding="utf-8", errors="replace")
        out |= set(re.findall(r"^\s*#\s*define\s+(\w+)", text, re.M))
        out |= {m for m in re.findall(r"\b(cuda\w+|cublas\w+)\s*\(", text) if f"{m}(" in text}
    return out


def strip_comments_and_strings(text: str) -> str:
    """The source with comments and string/char literals blanked (newlines kept, so line numbers stay)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(" ")
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def kind(cond: str) -> str:
    """'hip' when the condition holds only in the HIP build, 'cuda' when only in CUDA's, else 'other'."""
    c = cond.replace(" ", "")
    if "STRATA_USE_HIP" not in c or "||" in c:
        return "other"
    if "!defined(STRATA_USE_HIP)" in c or "!STRATA_USE_HIP" in c:
        return "cuda"
    return "hip"


def hip_lines(text: str):
    """(line number, code) for each line the HIP build compiles, as far as STRATA_USE_HIP decides."""
    stack = []          # per open #if: [kind of the condition, in its #else branch]
    for no, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
        if m:
            d, rest = m.group(1), m.group(2)
            if d == "ifdef":
                stack.append(["hip" if rest.strip() == "STRATA_USE_HIP" else "other", False])
            elif d == "ifndef":
                stack.append(["cuda" if rest.strip() == "STRATA_USE_HIP" else "other", False])
            elif d == "if":
                stack.append([kind(rest), False])
            elif d == "elif" and stack:
                stack[-1] = ["other", False]        # (an #elif chain: counted as compiled)
            elif d == "else" and stack:
                stack[-1][1] = True
            elif d == "endif" and stack:
                stack.pop()
            continue
        skipped = any((k == "cuda" and not in_else) or (k == "hip" and in_else) for k, in_else in stack)
        if not skipped:
            yield no, line


def unmapped(repo: Path) -> list:
    """[(file, line, name)] for every name the HIP build compiles without a mapping."""
    compat = repo / "include" / "strata" / "hip_compat"
    have = mapped_names(compat)
    found = []
    for top in ("src", "include", "tests"):
        for pat in SOURCES:
            for f in sorted((repo / top).rglob(pat)):
                if compat in f.parents:
                    continue
                text = strip_comments_and_strings(f.read_text(encoding="utf-8", errors="replace"))
                for no, line in hip_lines(text):
                    for name in NAME.findall(line):
                        if name not in have:
                            found.append((str(f.relative_to(repo)), no, name))
    return found


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    bad = unmapped(repo)
    for f, no, name in bad:
        print(f"{f}:{no}: {name} has no mapping in include/strata/hip_compat/ (the AMD build will not compile)")
    if not bad:
        print("hip_compat: every CUDA name the HIP build compiles has a mapping")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
