# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reads runs/*.txt and writes matrix.json (every run) and the two tables as Markdown to stdout.

    python collect.py
"""
import json
import pathlib
import re

H = pathlib.Path(__file__).parent
MODELS = {"q2_0": "Q2_0", "iq2_xs": "IQ2_XS", "iq3_xxs": "IQ3_XXS", "iq3_s": "IQ3_S", "coder": "Coder"}
TIERS = ["1k", "4k", "32k", "64k", "128k", "262k"]


def grab(pat, text, cast=float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


rows = []
for m in MODELS:
    for t in TIERS:
        f = H / "runs" / f"{m}-{t}.txt"
        if not f.exists():
            continue
        s = f.read_text(errors="replace")
        rows.append({
            "model": MODELS[m], "tier": t,
            "prompt_tokens": grab(r"^prefill\s+(\d+) tokens", s, int),
            "prefill_tok_s": grab(r"^prefill\s+.*->\s+([\d.]+) tok/s", s),
            "ttft_s": (lambda v: v and round(v / 1000, 2))(grab(r"time to first token ([\d.]+) ms", s)),
            "decoded": grab(r"^decode\s+(\d+) tokens", s, int),
            "decode_tok_s": grab(r"^decode\s+.*->\s+([\d.]+) tok/s", s),
            "spec_accept": grab(r"^speculation\s+.*\(([\d.]+)\)", s),
            "tokens_per_round": grab(r"^speculation\s+.*, ([\d.]+) tokens per round", s),
            "vram_slots": grab(r"pre-filled \d+ of (\d+) slots", s, int),
        })
(H / "matrix.json").write_text(json.dumps(rows, indent=1) + "\n")
cell = {(r["model"], r["tier"]): r for r in rows}
for key, fmt in (("prefill_tok_s", "{:,.0f}"), ("decode_tok_s", "{:.1f}")):
    print(f"\n{key}\n\n| Model | " + " | ".join(t.upper() for t in TIERS) + " |")
    print("| --- |" + " ---: |" * len(TIERS))
    for name in MODELS.values():
        vals = [cell.get((name, t), {}).get(key) for t in TIERS]
        print(f"| **{name}** | " + " | ".join(fmt.format(v) if v else "-" for v in vals) + " |")
