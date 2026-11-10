#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""card-5: the trainer's day rations.  Read the preprocessing's caches (the .npz files beside each
image, tools/image/mf_pre.py wrote them) and lay them out again in the dump form the engine's own
tools read, so the trainer binary needs no zip and no npz code.

Nothing here re-runs the VAE or the text reader: the shrink filter is not recorded in the cache, and
changing it moves a latent by a relative L1 of 1.5e-1 (the card-4 record's section 4a).  A missing or
mis-shaped cache is an error, never a rebuild.

  mf_pre_export.py --dir <data dir> --out <run dir>        (48 images -> dumps + manifest)
  mf_pre_export.py --selftest                              (no GPU, no models)

The manifest line per image is tab separated:

  <stem> <w16> <h16> <tokens> <te_rows> <uncond_rows> <txt_dim>
  <crop l t r b> <original w h> <caption_dropout_rate> <caption>

with w16/h16 the bucket's latent sides (tokens = w16*h16 rows of 128 channels), the text rows the
prompt's states after the system turn, the dropout rate the manifest's 10th tab field (SPEC v1
default 0.0; card-3 data carry 0.0 — the trainer reads this column and refuses a line without it),
and the caption the one English line the data folder holds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CACHE_MODEL = "mf"          # Mage-Flow's; MageTrail's caches are "mt" (--cache), their VAEs' latents differ
LATENT_CH = 128             # a token's values (Mage-Flow's and MOSS-Image's; the caches' own count is read)
# the cache's filter, as the card-4 record names it: area average, fixed by tools/image/mf_pre.py
FILTER = "area"


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def npz_path(img: Path, w: int, h: int, suffix: str) -> Path:
    return img.with_name(f"{img.stem}_{w:04d}x{h:04d}_{CACHE_MODEL}{suffix}.npz")


def image_files(data: Path) -> list[Path]:
    out = [p for p in sorted(data.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
    if not out:
        raise SystemExit(f"{data}: no images")
    return out


def export_one(img: Path, out: Path) -> list[str] | None:
    """one image's caches into dumps, a manifest line a latent cache (a bucket); None with a printed reason when a
    file is missing or wrong"""
    lat_fs = [c for c in sorted(img.parent.glob(f"{img.stem}_*_{CACHE_MODEL}.npz")) if "_te." not in c.name]
    te_f: Path | None = img.with_name(f"{img.stem}_{CACHE_MODEL}_te.npz")
    if te_f is not None and not te_f.exists():
        te_f = None
    if not lat_fs or te_f is None:
        print(f"{img.name}: no cache ({'latent' if not lat_fs else 'ok'}/{'te' if te_f is None else 'ok'}) - not "
              "rebuilt here", file=sys.stderr)
        return None
    with np.load(te_f) as z:
        emb = z["prompt_embeds"]                # bf16 bits as uint16, the system turn's rows dropped
        mask = z["attn_mask"]
        unc = z["uncond_embeds"]
        umask = z["uncond_mask"]
        drop = float(z["caption_dropout_rate"])
    if emb.dtype != np.uint16 or emb.ndim != 2:
        print(f"{img.name}: {te_f.name} prompt_embeds are {emb.dtype} {emb.shape}, not u16 [rows][dim]",
              file=sys.stderr)
        return None
    txt_dim = int(emb.shape[1])
    rows = int(np.count_nonzero(mask))          # the mask is the run of 1s, then the padding
    unc_rows = int(np.count_nonzero(umask))
    cap = img.with_suffix(".txt")
    text = cap.read_text(encoding="utf-8").strip().replace("\t", " ").replace("\n", " ") if cap.exists() else ""
    if not text:
        print(f"{img.name}: no caption beside it", file=sys.stderr)
    lines = []
    for lat_f in lat_fs:
        with np.load(lat_f) as z:
            lat = z["latents"]                      # F32 [128][h][w], the posterior's mean
            sd = z["latents_sd"] if "latents_sd" in z else None
            orig = z["original_size"]
            crop = z["crop_ltrb"]
        ch = lat.shape[0] if lat.ndim == 3 else 0
        if lat.dtype != np.float32 or lat.ndim != 3 or ch % 16 != 0:
            print(f"{img.name}: {lat_f.name} latents are {lat.dtype} {lat.shape}, not f32 [channels][h][w]",
                  file=sys.stderr)
            return None
        _, hh, ww = lat.shape
        # a stem a bucket; an image with one bucket keeps its own stem, as the single-budget runs named it
        tok = img.stem if len(lat_fs) == 1 else f"{img.stem}_{ww * 16:04d}x{hh * 16:04d}"
        put(out, f"{tok}_lat", np.ascontiguousarray(lat.reshape(ch, hh * ww).T))   # [tokens][ch] F32
        if sd is not None:   # the posterior's standard deviation, the trainer's draw
            put(out, f"{tok}_lat_sd", np.ascontiguousarray(sd.reshape(ch, hh * ww).T))
        put(out, f"{tok}_te", bf16_to_f32(emb[:rows]))                                     # [rows][txt_dim] F32
        put(out, f"{tok}_uncond", bf16_to_f32(unc[:unc_rows]))
        lines.append("\t".join((tok, str(ww), str(hh), str(ww * hh), str(rows), str(unc_rows), str(txt_dim),
                                " ".join(str(int(v)) for v in crop), " ".join(str(int(v)) for v in orig),
                                f"{drop:g}", text)))
    return lines


def put(out: Path, name: str, a: np.ndarray) -> None:
    a.astype(np.float32).tofile(out / f"{name}.bin")
    (out / f"{name}.json").write_text(json.dumps({"type": "f32", "shape": list(reversed(a.shape))}))


def run(data: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for img in image_files(data):
        got = export_one(img, out)
        if got is not None:
            lines.extend(got)
    if not lines:
        print(f"{data}: nothing exported", file=sys.stderr)
        return 1
    (out / "manifest.tsv").write_text(
        f"# filter\t{FILTER}\n# model\t{CACHE_MODEL}\n# dir\t{data}\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    toks = [int(ln.split("\t")[3]) for ln in lines]
    print(f"{data}: {len(lines)} images, {sum(toks)} tokens, buckets "
          f"{sorted({(ln.split(chr(9))[1], ln.split(chr(9))[2]) for ln in lines})}, "
          f"manifest {out / 'manifest.tsv'}")
    return 0


def selftest() -> int:
    """a fake data dir (two images, two buckets, one missing cache): the dumps come back with the
    shapes and the manifest says the same numbers, and the missing one is refused, not rebuilt"""
    import shutil
    import tempfile
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="mf-pre-export-"))
    try:
        d = tmp / "data"
        d.mkdir()
        for i, (w, h) in enumerate(((64, 96), (96, 64)), start=1):
            Image.new("RGB", (w, h), (i * 40, 90, 160)).save(d / f"0{i}.jpg")
            (d / f"0{i}.txt").write_text(f"stainedglassart, a window with {i} panes\n", encoding="utf-8")
            hh, ww = h // 16, w // 16
            lat = (np.arange(LATENT_CH * hh * ww, dtype=np.float32).reshape(LATENT_CH, hh, ww) / 1e4 - 0.5)
            np.savez(d / f"0{i}_{w:04d}x{h:04d}_{CACHE_MODEL}.npz", latents=lat,
                     original_size=np.array([w, h], np.int32), crop_ltrb=np.array([1, 2, w - 3, h - 4], np.int32))
            f32 = (np.random.default_rng(i).standard_normal((7, 2048)) * 0.1).astype(np.float32)
            bits = ((f32.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)   # the cache's bf16 bits
            np.savez(d / f"0{i}_{CACHE_MODEL}_te.npz", prompt_embeds=bits,
                     attn_mask=np.array([1, 1, 1, 1, 1, 1, 0], np.int32),
                     uncond_embeds=bits[:2], uncond_mask=np.array([1, 1], np.int32),
                     caption_dropout_rate=np.float32(0.0))
        # a third image with no cache at all: refused
        Image.new("RGB", (64, 64), (0, 0, 0)).save(d / "03.jpg")
        (d / "03.txt").write_text("stainedglassart, nothing\n", encoding="utf-8")
        out = tmp / "run"
        rc = run(d, out)
        if rc:
            print(f"selftest: export failed ({rc})")
            return 1
        man = (out / "manifest.tsv").read_text(encoding="utf-8").splitlines()
        bad = 0
        check = [("01", 4, 6, 24, 6, 2048, 1, 2, 61, 92), ("02", 6, 4, 24, 6, 2048, 1, 2, 93, 60)]
        for stem, ww, hh, toks, rows, dim, cl, ct, cr, cb in check:
            ln = next((x for x in man if x.startswith(stem + "\t")), None)
            if ln is None:
                print(f"selftest: {stem} missing from the manifest")
                bad += 1
                continue
            f = ln.split("\t")
            want = (str(ww), str(hh), str(toks), str(rows), "2", str(dim), f"{cl} {ct} {cr} {cb}")
            got = (f[1], f[2], f[3], f[4], f[5], f[6], f[7])
            if got != want:
                print(f"selftest: {stem} manifest {got}, want {want}")
                bad += 1
            a = np.fromfile(out / f"{stem}_lat.bin", dtype=np.float32)
            if a.size != toks * LATENT_CH:
                print(f"selftest: {stem} lat is {a.size} floats, want {toks * LATENT_CH}")
                bad += 1
            if not np.allclose(a.reshape(toks, LATENT_CH)[3],
                               np.load(d / f"{stem}_0064x0096_{CACHE_MODEL}.npz" if stem == "01"
                                       else d / f"{stem}_0096x0064_{CACHE_MODEL}.npz")["latents"][:, 0, 3],
                               atol=0, rtol=0):
                print(f"selftest: {stem} lat row 3 is not the cache's channel 3 (the transpose is wrong)")
                bad += 1
        if (out / "03_lat.bin").exists():
            print("selftest: the image with no cache was exported anyway")
            bad += 1
        for p in out.glob("*.json"):
            j = json.loads(p.read_text(encoding="utf-8"))
            if j["type"] != "f32" or not j["shape"]:
                print(f"selftest: {p.name} is not the engine's dump form")
                bad += 1
        # the mask's row count and the caption survive
        first = next((ln for ln in man if ln.startswith("01\t")), "")
        if "a window with 1 panes" not in first:
            print(f"selftest: the caption is not in the manifest: {first}")
            bad += 1
        print(f"mf_pre_export --selftest: {'OK' if bad == 0 else f'{bad} FAILURE(s)'}")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="mf", choices=("mf", "mt", "ms", "an"),
                    help="the caches' model tag (mf_pre.py --cache)")
    a = ap.parse_args()
    global CACHE_MODEL
    CACHE_MODEL = a.cache
    if a.selftest:
        return selftest()
    if not a.dir or not a.out:
        print("need --dir and --out (or --selftest)", file=sys.stderr)
        return 2
    return run(Path(a.dir), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
