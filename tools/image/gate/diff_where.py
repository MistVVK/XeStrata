"""tools/image/gate/diff_where.py - TEST-ONLY (OWN-5 gate harness): where two picture sets differ, and what the user
sees after the server's compositing.

    python -I -B diff_where.py <pics root> <tagA> <tagB>

Per pair (p*_s*.png in both): the largest absolute difference over R, G, B and alpha with its pixel (row, column) and
channel, whether that pixel lies within 16 px of the border, the share of pixels whose largest RGB or alpha difference
is above 8 / 255, and the RGB PSNR of the RGBA pictures and of the pictures as the server sends them by default
(serve/images.py flatten_png: on white, 8-bit RGB).  The last lines are the worst picture of the set by RGB PSNR."""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from serve.images import flatten_png  # noqa: E402

CH = "RGBA"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)


def one(pa: Path, pb: Path) -> dict:
    A = np.asarray(Image.open(pa).convert("RGBA")).astype(np.int16)
    B = np.asarray(Image.open(pb).convert("RGBA")).astype(np.int16)
    d = np.abs(A - B)
    y, x, c = np.unravel_index(int(np.argmax(d)), d.shape)
    H, W = d.shape[:2]
    fa = np.asarray(Image.open(io.BytesIO(flatten_png(pa.read_bytes()))).convert("RGB"))
    fb = np.asarray(Image.open(io.BytesIO(flatten_png(pb.read_bytes()))).convert("RGB"))
    return {"name": pa.name, "max": int(d.max()), "at": (int(y), int(x)), "channel": CH[c],
            "border": bool(min(y, x, H - 1 - y, W - 1 - x) < 16),
            "max_rgb": int(d[..., :3].max()), "max_alpha": int(d[..., 3].max()),
            "pct_rgb_over8": float((d[..., :3].max(axis=2) > 8).mean() * 100),
            "pct_alpha_over8": float((d[..., 3] > 8).mean() * 100),
            "psnr_rgb": psnr(A[..., :3], B[..., :3]), "psnr_flat": psnr(fa, fb),
            "flat_max": int(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())}


def main(argv) -> int:
    root, ta, tb = Path(argv[0]), argv[1], argv[2]
    rows = []
    for pa in sorted((root / ta).glob("p*_s*.png")):
        pb = root / tb / pa.name
        if pb.is_file():
            r = one(pa, pb)
            rows.append(r)
            print(f"{r['name']}: max |d| {r['max']} at row {r['at'][0]} col {r['at'][1]} channel {r['channel']}"
                  f"{' (border)' if r['border'] else ''}; RGB max {r['max_rgb']}, alpha max {r['max_alpha']}; "
                  f"pixels > 8: RGB {r['pct_rgb_over8']:.4f}%, alpha {r['pct_alpha_over8']:.4f}%; "
                  f"PSNR RGBA-pictures {r['psnr_rgb']:.2f} dB, flattened on white {r['psnr_flat']:.2f} dB "
                  f"(max |d| {r['flat_max']})")
    if not rows:
        print("no pairs")
        return 1
    w = min(rows, key=lambda r: r["psnr_rgb"])
    print(f"WORST {ta} vs {tb}: {w['name']} PSNR {w['psnr_rgb']:.2f} dB (flattened {w['psnr_flat']:.2f} dB)")
    fl = [r["psnr_flat"] for r in rows if math.isfinite(r["psnr_flat"])]
    print(f"FLATTENED {ta} vs {tb}: n {len(rows)}, PSNR worst {min(fl):.2f} mean {np.mean(fl):.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
