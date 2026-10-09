# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Judges XeStrata against upstream from bench-collect.py's matrix.json files: per model file, GPU, tier and
measure (prompt, output), the median of each engine's rounds; a cell passes when XeStrata is at most 1% slower.

    python .claude/skills/upstream-sync/scripts/judge.py OUT/IQ3_XXS OUT/Q2_0 ...   # one folder per model file

A cell where upstream did not finish is left out (not measurable); one where only XeStrata did not finish fails.
Prints a line per cell and, last, "ALL PASS" (exit 0) or "FAIL <n>" (exit 1; also when no cell was compared).
"""
import json
import pathlib
import statistics
import sys

LIMIT = -0.01
TIERS = ["1k", "4k", "32k", "64k", "128k", "262k"]
MEASURES = (("prefill_tok_s", "prompt"), ("decode_tok_s", "output"))


def median(rows, key):
    v = [r[key] for r in rows if r[key]]
    return statistics.median(v) if v else None


fails = judged = 0
for folder in map(pathlib.Path, sys.argv[1:]):
    rows = json.loads((folder / "matrix.json").read_text())
    for gpu in sorted({r["gpu"] for r in rows}):
        for tier in TIERS:
            cell = [r for r in rows if (r["gpu"], r["tier"]) == (gpu, tier)]
            if not cell:
                continue
            for key, name in MEASURES:
                xe = median([r for r in cell if r["arm"] == "xe"], key)
                up = median([r for r in cell if r["arm"] == "up"], key)
                where = f"{folder.name:10s} {gpu:8s} {tier:5s} {name:6s}"
                if up is None:
                    print(f"{where} SKIP  upstream did not finish")
                elif xe is None:
                    fails += 1
                    print(f"{where} FAIL  XeStrata did not finish (upstream {up:.1f})")
                else:
                    judged += 1
                    d = xe / up - 1
                    ok = d >= LIMIT
                    fails += not ok
                    print(f"{where} {'PASS' if ok else 'FAIL'}  {xe:.1f} vs {up:.1f} ({100 * d:+.1f}%)")
fails += judged == 0                          # nothing measured is no win
print("ALL PASS" if fails == 0 else f"FAIL {fails}")
sys.exit(1 if fails else 0)
