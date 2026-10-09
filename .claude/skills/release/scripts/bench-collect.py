# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reads bench.sh's OUT/runs/*.txt and writes OUT/matrix.json (every run) and, per GPU, the prompt and output tables
(the median of the rounds; a cell no run finished is "-") as Markdown to stdout.

    python .claude/skills/release/scripts/bench-collect.py OUT XE_NAME UP_NAME   # e.g. OUT xe0.1.41 v0.1.41
"""
import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])
NAMES = {"xe": sys.argv[2], "up": sys.argv[3]}
TIERS = ["1k", "4k", "32k", "64k", "128k", "262k"]


def grab(pat, text, cast=float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


rows = []
for f in sorted((out / "runs").glob("*.txt")):
    m = re.fullmatch(r"(.+)-(xe|up)-(\w+)-(\d+)", f.stem)
    if not m:                                 # the warm-ups
        continue
    s = f.read_text(errors="replace")
    rows.append({
        "gpu": m.group(1), "engine": NAMES[m.group(2)], "tier": m.group(3), "round": int(m.group(4)),
        "prompt_tokens": grab(r"^prefill\s+(\d+) tokens", s, int),
        "prefill_tok_s": grab(r"^prefill\s+.*->\s+([\d.]+) tok/s", s),
        "decoded": grab(r"^decode\s+(\d+) tokens", s, int),
        "decode_tok_s": grab(r"^decode\s+.*->\s+([\d.]+) tok/s", s),
        "spec_accept": grab(r"^speculation\s+.*\(([\d.]+)\)", s),
    })
(out / "matrix.json").write_text(json.dumps(rows, indent=1) + "\n")

for gpu in sorted({r["gpu"] for r in rows}):
    for key, title, fmt in (("prefill_tok_s", "Prompt (tokens/s)", "{:,.0f}"),
                            ("decode_tok_s", "Output (tokens/s)", "{:.1f}")):
        print(f"\n### {gpu}: {title}\n\n| Engine | " + " | ".join(t.upper() for t in TIERS) + " |")
        print("| --- |" + " ---: |" * len(TIERS))
        for name in NAMES.values():
            cells = []
            for t in TIERS:
                v = [r[key] for r in rows if (r["gpu"], r["engine"], r["tier"]) == (gpu, name, t) and r[key]]
                cells.append(fmt.format(statistics.median(v)) if v else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")
