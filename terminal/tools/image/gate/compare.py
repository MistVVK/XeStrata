"""OWN-5: pixel comparison of two picture sets with the same names (p<i>_s<seed>.png).

    python -I -B compare.py <pics_root> <tagA> <tagB> [--label NAME]

Per pair: identical bytes, identical pixels, max and mean absolute difference (0-255, RGB; alpha separately),
PSNR (RGB, dB; inf when equal), SSIM (RGB mean of per-channel SSIM, Gaussian 11x11 sigma 1.5, Wang et al. constants).
Writes <pics_root>/cmp/<label>.json, a diff picture per pair (|A-B| x 8, clipped) and <pics_root>/cmp/<label>.html with
A | B | diff side by side.  Prints a summary line."""
from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def gauss_kernel(n=11, sigma=1.5):
    x = np.arange(n) - n // 2
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def blur(a, k):
    # separable 'valid' convolution
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 0, a)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, a)


def ssim_channel(x, y, k):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x, k), blur(y, k)
    sxx = blur(x * x, k) - mx * mx
    syy = blur(y * y, k) - my * my
    sxy = blur(x * y, k) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def main() -> int:
    a = sys.argv[1:]
    root, ta, tb = Path(a[0]), a[1], a[2]
    label = a[a.index("--label") + 1] if "--label" in a else f"{ta}__vs__{tb}"
    da, db = root / ta, root / tb
    out = root / "cmp"
    (out / label).mkdir(parents=True, exist_ok=True)
    k = gauss_kernel()
    rows = []
    for pa in sorted(da.glob("p*_s*.png")):
        pb = db / pa.name
        if not pb.is_file():
            continue
        ba, bb = pa.read_bytes(), pb.read_bytes()
        ia, ib = Image.open(pa), Image.open(pb)
        A, B = np.asarray(ia.convert("RGBA"), dtype=np.float64), np.asarray(ib.convert("RGBA"), dtype=np.float64)
        if A.shape != B.shape:
            rows.append({"name": pa.name, "error": f"shape {A.shape} vs {B.shape}"})
            continue
        d = np.abs(A[..., :3] - B[..., :3])
        da_ = np.abs(A[..., 3] - B[..., 3])
        mse = float((d * d).mean())
        psnr = math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)
        ssim = 1.0 if mse == 0 else float(np.mean([ssim_channel(A[..., c], B[..., c], k) for c in range(3)]))
        diffp = out / label / pa.name.replace(".png", "_diff.png")
        Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(diffp)
        rows.append({"name": pa.name, "mode": f"{ia.mode}/{ib.mode}", "bytes_equal": ba == bb,
                     "pixels_equal": bool((A == B).all()), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "alpha_max_abs": float(da_.max()), "psnr_db": psnr, "ssim": ssim,
                     "pct_pixels_diff": float((d.max(axis=2) > 0).mean() * 100),
                     "sha_a": hashlib.sha256(ba).hexdigest()[:12], "sha_b": hashlib.sha256(bb).hexdigest()[:12]})
    ok = [r for r in rows if "error" not in r]
    summ = {"label": label, "a": ta, "b": tb, "pairs": len(rows),
            "bytes_equal": sum(r["bytes_equal"] for r in ok), "pixels_equal": sum(r["pixels_equal"] for r in ok)}
    if ok:
        fin = [r["psnr_db"] for r in ok if math.isfinite(r["psnr_db"])]
        summ.update(max_abs=max(r["max_abs"] for r in ok), mean_abs=float(np.mean([r["mean_abs"] for r in ok])),
                    psnr_min=min(fin) if fin else math.inf, psnr_mean=float(np.mean(fin)) if fin else math.inf,
                    ssim_min=min(r["ssim"] for r in ok), ssim_mean=float(np.mean([r["ssim"] for r in ok])))
    (out / f"{label}.json").write_text(json.dumps({"summary": summ, "pairs": rows}, indent=1))
    rel = lambda p: Path(p).relative_to(root).as_posix()  # noqa: E731
    h = [f"<!doctype html><meta charset=utf-8><title>{html.escape(label)}</title>",
         "<style>body{font:14px sans-serif;margin:16px}img{width:32%;max-width:512px}td{vertical-align:top}</style>",
         f"<h1>{html.escape(ta)} vs {html.escape(tb)}</h1><pre>{html.escape(json.dumps(summ, indent=1))}</pre>"]
    for r in rows:
        if "error" in r:
            h.append(f"<p>{r['name']}: {html.escape(r['error'])}</p>")
            continue
        h.append(f"<h3>{r['name']}: bytes equal {r['bytes_equal']}, max |d| {r['max_abs']:.0f}, mean |d| "
                 f"{r['mean_abs']:.3f}, PSNR {r['psnr_db']:.2f} dB, SSIM {r['ssim']:.5f}</h3>"
                 f"<div><img src='../{rel(da / r['name'])}' title='A'> <img src='../{rel(db / r['name'])}' title='B'> "
                 f"<img src='{label}/{r['name'].replace('.png', '_diff.png')}' title='|A-B| x 8'></div>")
    (out / f"{label}.html").write_text("\n".join(h), encoding="utf-8")
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
