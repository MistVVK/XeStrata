#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi_train_prep.py - a LoRA training set for Qwen-Image-2.1 cut and cached as ai-toolkit's DataLoader
does it, for xestrata-image-qi-train train.

Each image (with its caption: the file of the same name with .txt, read as it is) is taken once a resolution
(512, 768 and 1024 by default, ai-toolkit's three): EXIF-turned, RGB, scaled with PIL's bicubic so it covers its
bucket (get_bucket_for_image_size: no larger than the resolution's pixels, never scaled up, sides a multiple of 32)
and cut from the centre.  The cut images go through the VAE's encoder (the posterior's mean and standard deviation)
and the captions through Qwen3-VL in the chat template serve uses (the system turn dropped; a special token written in
a caption read as the token, as ai-toolkit's processor reads it); a caption's [trigger]
or [name] is the trigger word (or nothing), and a trigger word missing from a caption goes in front.  The caption the
dropout swaps in is the trigger word alone and a space after it, or " " without one.

    qi_train_prep.py --data <folder> --out <dir> --exe <xestrata-image-qi-train> --te <Qwen3-VL GGUF> --vae <VAE>
                     [--resolutions 512,768,1024] [--trigger WORD]

writes <dir>/items.txt for train's --data.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer  # noqa: E402
from serve import images as serve_images  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DIVISIBILITY = 32   # ai-toolkit's Qwen-Image-2.1 get_bucket_divisibility


def bucket(width: int, height: int, resolution: int, divisibility: int = DIVISIBILITY) -> tuple[int, int]:
    """ai-toolkit's get_bucket_for_image_size."""
    total = width * height
    max_pixels = resolution * resolution
    target = min(total, max_pixels)
    scaler = (target / total) ** 0.5
    w_raw, h_raw = width * scaler / divisibility, height * scaler / divisibility
    candidates = [(math.floor(w_raw) * divisibility, math.floor(h_raw) * divisibility),
                  (math.floor(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
                  (math.ceil(w_raw) * divisibility, math.floor(h_raw) * divisibility),
                  (math.ceil(w_raw) * divisibility, math.ceil(h_raw) * divisibility)]
    capped = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= max_pixels]
    if not capped:
        capped = [(max(divisibility, math.floor(w_raw) * divisibility),
                   max(divisibility, math.floor(h_raw) * divisibility))]
    return min(capped, key=lambda wh: abs(wh[0] * wh[1] - target))


def cut(img: Image.Image, resolution: int) -> Image.Image:
    """ai-toolkit's setup_buckets and load_and_process_image: scaled to cover the bucket, the centre cut out."""
    bw, bh = bucket(img.width, img.height, resolution)
    f = max(bw / img.width, bh / img.height)
    sw, sh = int(math.ceil(img.width * f)), int(math.ceil(img.height * f))
    x, y = int((sw - bw) / 2), int((sh - bh) / 2)
    return img.resize((sw, sh), Image.BICUBIC).crop((x, y, x + bw, y + bh))


def caption_of(raw: str, trigger: str) -> str:
    """ai-toolkit's inject_trigger_into_prompt."""
    out = raw.replace("[name]", trigger).replace("[trigger]", trigger)
    if trigger.strip() and out.count(trigger) == 0:
        out = trigger + " " + out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--resolutions", default="512,768,1024")
    ap.add_argument("--trigger", default="")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    tok = strata_tokenizer.Tokenizer.from_gguf(a.te)
    files = sorted(p for p in Path(a.data).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"qi_train_prep: no images in {a.data}")
        return 1
    lines, items = [], []

    def text(name: str, caption: str) -> Path:
        # ai-toolkit's prompt encoder: an empty prompt reads as " "
        # the whole chat template parsed for special tokens, the caption's too: ai-toolkit's processor reads a
        # <|im_end|> written in a caption as the token (serve's prompt_ids reads a user's as text)
        text_ = serve_images.PROMPT_HEAD + (caption if caption else " ") + serve_images.PROMPT_TAIL
        ids = np.array(tok.encode(text_, parse_special=True), np.int32)
        ids_path, states = out / "txt" / f"{name}.ids", out / "txt" / f"{name}.states"
        ids.tofile(ids_path)
        lines.append(f"txt {ids_path} {serve_images.DROP} {states}")
        return states

    empty = text("_dropout", caption_of("", a.trigger))   # get_dropout_caption: "WORD " with a trigger word
    for f in files:
        cap = f.with_suffix(".txt")
        raw = cap.read_text(encoding="utf-8") if cap.exists() else ""
        states = text(f.stem, caption_of(raw, a.trigger))
        img = exif_transpose(Image.open(f)).convert("RGB")
        for res in (int(r) for r in a.resolutions.split(",")):
            c = cut(img, res)
            rgb = out / "img" / f"{f.stem}-{res}.rgb"
            np.asarray(c, np.uint8).tofile(rgb)
            lat = out / "img" / f"{f.stem}-{res}.lat"
            lines.append(f"img {rgb} {c.width} {c.height} {lat}")
            items.append(f"{lat} {states}")
    (out / "list.txt").write_text("\n".join(lines) + "\n")
    r = subprocess.run([a.exe, "prep", "--te", a.te, "--vae", a.vae, "--list", str(out / "list.txt")])
    if r.returncode != 0:
        return r.returncode
    (out / "items.txt").write_text(f"empty {empty}\n" + "\n".join(items) + "\n")
    for p in (out / "img").glob("*.rgb"):
        p.unlink()
    print(f"qi_train_prep: {len(files)} images, {len(items)} items in {out / 'items.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
