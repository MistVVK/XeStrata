# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Judges XeStrata against upstream from sync_collect.py's matrix.json files: per model file, GPU, tier and
measure (prompt, output), the median of each engine's rounds; a cell passes when XeStrata is at most 1% slower.
One folder per model file.

    python .claude/skills/upstream-sync/scripts/judge.py [--prompt-only GPU]... OUT/IQ3_XXS OUT/IQ2_XS ...

A cell where upstream did not finish, ran at SLOW tok/s or less, or was stopped by bench.sh as that slow, is left
out (not measurable); one where only XeStrata did not finish fails.  --prompt-only GPU: that GPU's output cells are
left out too (a GPU no matrix.json names is an error: bench.sh's GPU argument, spelled the same).
Prints a line per cell and, last, "ALL PASS" (exit 0) or "FAIL <n>" (exit 1; also when no cell was compared).
"""
import json
import pathlib
import statistics
import sys

LIMIT = -0.01
SLOW = 10.0                                   # tok/s: upstream this slow is not a speed to compare with
TIERS = ["1k", "4k", "32k", "64k", "128k", "192k", "262k"]
MEASURES = (("prefill_tok_s", "prompt"), ("decode_tok_s", "output"))


def median(rows, key):
    v = [r[key] for r in rows if r[key]]
    return statistics.median(v) if v else None


args = sys.argv[1:]
prompt_only = set()
while len(args) >= 2 and args[0] == "--prompt-only":
    prompt_only.add(args[1])
    args = args[2:]
matrices = [(pathlib.Path(a), json.loads((pathlib.Path(a) / "matrix.json").read_text())) for a in args]
unknown = prompt_only - {r["gpu"] for _, rows in matrices for r in rows}
if unknown:
    sys.exit(f"--prompt-only: no measurements of {', '.join(sorted(unknown))} (GPUs: "
             f"{', '.join(sorted({r['gpu'] for _, rows in matrices for r in rows}))})")
fails = judged = 0
for folder, rows in matrices:
    for gpu in sorted({r["gpu"] for r in rows}):
        for tier in TIERS:
            cell = [r for r in rows if (r["gpu"], r["tier"]) == (gpu, tier)]
            if not cell:
                continue
            for key, name in MEASURES:
                xe = median([r for r in cell if r["arm"] == "xe"], key)
                up = median([r for r in cell if r["arm"] == "up"], key)
                where = f"{folder.name:10s} {gpu:8s} {tier:5s} {name:6s}"
                stopped = {r["stopped"] for r in cell if r["arm"] == "up" and r.get("stopped")}
                if name == "output" and gpu in prompt_only:
                    print(f"{where} SKIP  output not compared on this GPU (--prompt-only)")
                elif up is None and stopped:
                    print(f"{where} SKIP  upstream stopped at {SLOW:g} tok/s or less ({', '.join(sorted(stopped))})")
                elif up is None:
                    print(f"{where} SKIP  upstream did not finish")
                elif up <= SLOW:
                    print(f"{where} SKIP  upstream at {up:.1f} tok/s")
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
