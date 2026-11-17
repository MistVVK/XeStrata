"""OWN-5: one line per comparison in pics/cmp/*.json with its worst pair (lowest PSNR), and pics/index.html.
    python -I -B summarize.py <pics>"""
import html
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = []
for f in sorted((root / "cmp").glob("*.json")):
    d = json.loads(f.read_text())
    s, pairs = d["summary"], [p for p in d["pairs"] if "error" not in p]
    worst = min(pairs, key=lambda p: p["psnr_db"]) if pairs else None
    items.append((s, worst))
    w = "-" if worst is None or math.isinf(worst["psnr_db"]) else f"{worst['name']} (PSNR {worst['psnr_db']:.1f}, SSIM {worst['ssim']:.3f}, max {worst['max_abs']:.0f})"
    print(f"{s['label']}: {s['bytes_equal']}/{s['pairs']} identical; worst {w}")
h = ["<!doctype html><meta charset=utf-8><title>OWN-5 same seed</title>",
     "<style>body{font:14px sans-serif;margin:16px;max-width:1100px}li{margin:6px 0}</style>",
     "<h1>OWN-5: the same seed, picture pairs</h1><p>Each link shows A | B | |A-B| x 8 for every pair, worst first is listed here.</p><ul>"]
for s, w in items:
    ws = "" if w is None or math.isinf(w["psnr_db"]) else f"; worst pair {w['name']}: PSNR {w['psnr_db']:.1f} dB, SSIM {w['ssim']:.3f}"
    pm = s.get("psnr_mean")
    h.append(f"<li><a href='cmp/{html.escape(s['label'])}.html'>{html.escape(s['label'])}</a>: {s['bytes_equal']}/{s['pairs']} "
             f"byte-identical, mean |d| {s.get('mean_abs', 0):.2f}, PSNR mean {pm if pm is None or math.isinf(pm) else round(pm, 1)}{ws}</li>")
(root / "index.html").write_text("\n".join(h + ["</ul>"]), encoding="utf-8")
