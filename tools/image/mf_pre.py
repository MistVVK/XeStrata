# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/mf_pre.py - Mage-Flow's learning preprocessing: the buckets, the VAE latent cache and the text
embedding cache a training step reads back from disk (docs/LORA_TRAIN.md "Aspect buckets", "The latent and
text-embedding disk caches").  The values themselves come from xestrata-image-mf-pre (Mage-VAE's encoder, and the text
reader on the inference's own states path); this file owns the dataset walk, the sizes and the cache shape.

    PY=... python tools/image/mf_pre.py --dir DIR --vae VAE.safetensors --text TEXT.gguf \
        [--pre build/free/xestrata-image-mf-pre] [--toml set.toml] [--budget 512,768,1024] [--step 16] [--scratch DIR]
    PY=... python tools/image/mf_pre.py --check --dir DIR     # the caches beside the images: shape, keys, types
    PY=... python tools/image/mf_pre.py --cache qi --dir DATA --out DIR --vae VAE --text TE.gguf [--trigger WORD]
        # Qwen-Image-2.1's set, cut as ai-toolkit does it: DIR holds the caches and items.txt its trainer reads
    PY=... python tools/image/mf_pre.py --golden GOLDEN_DIR --vae ... --text ...   # both paths against a golden run

`--dir` walks every image the engine's own reader can open (the v1 data mixes .jpg and .png, so nothing is fixed to
one extension); one image, one caption, read from `NN.txt` as it stands (SPEC "The caption policy").  A size goes:

    budgets 512, 768 and 1024 pixels a side (ai-toolkit's Mage-Flow resolutions), step 16; for each budget
    never enlarge - scale to the budget on the image's own ratio, floor both sides to the step,
    cover that on the same ratio and take the centre  ->  bucket, its cover size, and the crop's
    left,top,right,bottom

and the cropped pixels go to the VAE, a latent a bucket (a small image's same bucket once).  One `.npz` per cache,
beside the image, in sd-scripts' shape:

    <base>_<W>x<H>_mf.npz   latents F32 [128][h][w] (the posterior's mean), latents_sd F32 [128][h][w] (its
                            standard deviation: a step draws from them, as ai-toolkit samples latent_dist),
                            original_size I32 [2] (W,H), crop_ltrb I32 [4] (L,T,R,B of the kept part in the cover
                            frame)
    <base>_mf_te.npz        prompt_embeds U16 [L][C] (bf16 bits), attn_mask I32 [L], uncond_embeds / uncond_mask
                            (the empty prompt, always written), caption_dropout_rate F32 scalar

So a run is reproduced from the file name and the two arrays, and nothing is silently re-decided when the trainer
reads.  `--golden` re-runs the same two paths on a golden run's own inputs and prints how far the engine's values are
from the reference's - the match SPEC asks of preprocessing.  `xestrata-image-mf-probe` reports the same two figures for
those runs on the inference side (`mean cos 0.999069 rel 0.0359` of edit-base-scarf, `text cos 0.999634 rel 0.0317` of
turbo-fox-s11), so what this checks is that preprocessing adds nothing to the distance the engine already stands at.
`--scratch` is where the dumps go: a real disk, since one 1024-ish image is about 40 MB of pix and 2 MB of latent
(a tmpfs /tmp overflows and the run fails for the disk).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    from PIL.ImageOps import exif_transpose
except ImportError:            # CTest's mf_pre_cache skips where the image library is not the interpreter's own
    sys.stderr.write("mf_pre: no PIL in this interpreter (CTest skips on exit 77)\n")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "image"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from serve.images import DROP as GENERATION_DROP  # noqa: E402  (the generation prompt's system turn)
from serve.images import PROMPT_HEAD, PROMPT_TAIL, anima_tokens, load_tokenizer, prompt_ids  # noqa: E402
from serve.train_defaults import CAPTION_DROPOUT  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif")      # what third_party/main/stb reads, so what the engine reads
LATENT_TAG = "_mf.npz"
DROP = 34                                      # the generation system turn's rows, serve.images.MF_DROP
TEXT_TAG = "_mf_te.npz"
# the fields each table may hold, per SPEC "The dataset setting (TOML)"; anything else is refused at start
GENERAL_OK = {"caption_extension", "keep_tokens", "shuffle_caption", "caption_dropout_rate"}
DATASET_OK = {"subsets", "resolution", "enable_bucket", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps",
              "bucket_no_upscale", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs",
              "cache_text_encoder_outputs_to_disk", "batch_size"}
SUBSET_OK = {"image_dir", "num_repeats", "caption_extension"}


def bucket(orig_w: int, orig_h: int, budget: int, step: int) -> tuple[int, int, int, int, int, int, int, int]:
    """(bucket_w, bucket_h, cover_w, cover_h, L, T, R, B): the floor-both-sides shape SPEC fixes, the size the image
    is resized to so it covers that shape on its own ratio, and the centre crop taken from it."""
    ar = orig_w / orig_h
    if orig_w * orig_h > budget * budget:
        rw = math.sqrt(budget * budget * ar)
        rh = budget * budget / rw
    else:                                        # never enlarge: the image's own size already is the cover
        rw, rh = float(orig_w), float(orig_h)
    bw, bh = int(rw) // step * step, int(rh) // step * step
    cw, ch = max(int(round(rw)), bw), max(int(round(rh)), bh)
    left, t = (cw - bw) // 2, (ch - bh) // 2
    return bw, bh, cw, ch, left, t, left + bw, t + bh


def vae_input(path: Path, bw: int, bh: int, cover: tuple[int, int], crop: tuple[int, int, int, int]) -> np.ndarray:
    """The image as the encoder takes it: [3][bh][bw] FP32 in -1 .. 1.  Area-resized to the cover (sd-scripts'
    default interpolation when it shrinks, cv2's INTER_AREA; v1 never enlarges) then centre-cropped to the bucket."""
    with Image.open(path) as im:
        px = np.asarray(im.convert("RGB"), np.uint8)
    px = area_resize(px, cover[1], cover[0])
    left, t, r, b = crop
    a = px[t:b, left:r].astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def _area_w(n_in: int, n_out: int) -> np.ndarray:
    """One axis's weights: target pixel i is the area average of the source span [i*n_in/n_out, (i+1)*n_in/n_out),
    each source pixel weighted by its overlap, so a row sums to 1 (INTER_AREA's own rule)."""
    edge = np.arange(n_out + 1, dtype=np.float64) * n_in / n_out
    lo, hi = np.floor(edge[:-1]).astype(np.int64), np.ceil(edge[1:]).astype(np.int64)
    w = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        p0, p1 = edge[i], edge[i + 1]
        for j in range(int(lo[i]), min(int(hi[i]), n_in)):
            w[i, j] = min(p1, j + 1.0) - max(p0, float(j))
    return (w / (edge[1:] - edge[:-1])[:, None]).astype(np.float32)


def area_resize(px: np.ndarray, n_h: int, n_w: int) -> np.ndarray:
    """uint8 [H][W][3] to the area-resized uint8 [n_h][n_w][3], rounded and clipped as cv2's 8-bit result is."""
    h, w = px.shape[:2]
    if (h, w) == (n_h, n_w):
        return px.copy()
    y, x = _area_w(h, n_h), _area_w(w, n_w)
    out = np.einsum("ih,hwc,jw->ijc", y, px.astype(np.float32), x, optimize=True)
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


QI_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")   # what PIL opens of ai-toolkit's datasets
QI_DIVISIBILITY = 32   # ai-toolkit's Qwen-Image-2.1 get_bucket_divisibility


def qi_bucket(width: int, height: int, resolution: int) -> tuple[int, int]:
    """ai-toolkit's get_bucket_for_image_size: the resolution's pixels at most, sides of 32, no enlargement."""
    total = width * height
    max_pixels = resolution * resolution
    target = min(total, max_pixels)
    scaler = (target / total) ** 0.5
    w_raw, h_raw = width * scaler / QI_DIVISIBILITY, height * scaler / QI_DIVISIBILITY
    candidates = [(math.floor(w_raw) * QI_DIVISIBILITY, math.floor(h_raw) * QI_DIVISIBILITY),
                  (math.floor(w_raw) * QI_DIVISIBILITY, math.ceil(h_raw) * QI_DIVISIBILITY),
                  (math.ceil(w_raw) * QI_DIVISIBILITY, math.floor(h_raw) * QI_DIVISIBILITY),
                  (math.ceil(w_raw) * QI_DIVISIBILITY, math.ceil(h_raw) * QI_DIVISIBILITY)]
    capped = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= max_pixels]
    if not capped:
        capped = [(max(QI_DIVISIBILITY, math.floor(w_raw) * QI_DIVISIBILITY),
                   max(QI_DIVISIBILITY, math.floor(h_raw) * QI_DIVISIBILITY))]
    return min(capped, key=lambda wh: abs(wh[0] * wh[1] - target))


def qi_cut(img: Image.Image, resolution: int) -> Image.Image:
    """ai-toolkit's setup_buckets and load_and_process_image: scaled to cover the bucket (PIL's bicubic), the centre cut
    out."""
    bw, bh = qi_bucket(img.width, img.height, resolution)
    f = max(bw / img.width, bh / img.height)
    sw, sh = int(math.ceil(img.width * f)), int(math.ceil(img.height * f))
    x, y = int((sw - bw) / 2), int((sh - bh) / 2)
    return img.resize((sw, sh), Image.BICUBIC).crop((x, y, x + bw, y + bh))


def qi_caption(raw: str, trigger: str) -> str:
    """ai-toolkit's inject_trigger_into_prompt: [name] and [trigger] are the word, and it leads a caption without it."""
    out = raw.replace("[name]", trigger).replace("[trigger]", trigger)
    if trigger.strip() and out.count(trigger) == 0:
        out = trigger + " " + out
    return out


def qi_prep(a) -> int:
    """Qwen-Image-2.1's training set, cut and cached as ai-toolkit's DataLoader does it: every image once a resolution
    (the pixel sides of --budget), EXIF-turned, RGB, its caption read from the .txt beside it with the trigger word in
    it.  The cut images and the captions' ids go to xestrata-image-mf-pre qi-prep (the VAE's posterior mean and
    standard deviation, the text encoder's states before its final norm), and the caches it writes are the items
    file's lines, as qi_trainer reads them (train_sample.hpp).  Both lists are tab-separated: a data folder's path
    can hold spaces."""
    import strata_tokenizer   # tools/strata_tokenizer.py: Qwen3-VL's tokenizer from the GGUF

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    tok = strata_tokenizer.Tokenizer.from_gguf(a.text)
    files = sorted(p for p in Path(a.dir).iterdir() if p.suffix.lower() in QI_EXTS)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"mf_pre: no images in {a.dir}")
    lines, items = [], []

    def field(path: Path) -> str:
        # the lists are tab-separated: a space in a data-folder path is fine, a tab or newline cannot be a field
        s = str(path)
        if any(c in s for c in "\t\r\n"):
            raise SystemExit(f"mf_pre: a cache path cannot hold a tab or newline: {s!r}")
        return s

    def text(name: str, caption: str) -> Path:
        # ai-toolkit's prompt encoder: an empty prompt reads as " ", the chat template parsed for special tokens
        # (a <|im_end|> written in a caption is that token, as its processor reads it)
        ids = np.array(tok.encode(PROMPT_HEAD + (caption if caption else " ") + PROMPT_TAIL, parse_special=True),
                       np.int32)
        ids_path, states = out / "txt" / f"{name}.ids", out / "txt" / f"{name}.states"
        ids.tofile(ids_path)
        lines.append(f"txt\t{field(ids_path)}\t{GENERATION_DROP}\t{field(states)}")
        return states

    empty = text("_dropout", qi_caption("", a.trigger))   # get_dropout_caption: "WORD " with a trigger word
    for f in files:
        cap = f.with_suffix(".txt")
        raw = cap.read_text(encoding="utf-8") if cap.is_file() else ""   # ai-toolkit reads no caption as ""
        states = text(f.stem, qi_caption(raw, a.trigger))
        with Image.open(f) as im:
            img = exif_transpose(im).convert("RGB")
        for res in a.budgets:
            c = qi_cut(img, res)
            rgb = out / "img" / f"{f.stem}-{res}.rgb"
            np.asarray(c, np.uint8).tofile(rgb)
            lat = out / "img" / f"{f.stem}-{res}.lat"
            lines.append(f"img\t{field(rgb)}\t{c.width}\t{c.height}\t{field(lat)}")
            items.append(f"{field(lat)}\t{field(states)}")
        print(f"ok   {f.name} {len(a.budgets)} resolution(s)")
    (out / "list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    r = subprocess.run([a.pre, "qi-prep", a.vae, a.text, str(out / "list.txt")])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    (out / "items.txt").write_text(f"empty\t{field(empty)}\n" + "\n".join(items) + "\n", newline="\n")
    for p in (out / "img").glob("*.rgb"):
        p.unlink()
    print(f"mf_pre: {len(items)} items in {out / 'items.txt'}")
    return 0


def dump(d: Path, name: str, a: np.ndarray) -> None:
    """One array in the image engine's dump form (fastest dim first in the json, the types the probe reads)."""
    a = np.ascontiguousarray(a)
    t = {"float32": "f32", "int32": "i32", "uint16": "bf16"}[str(a.dtype)]
    a.tofile(d / f"{name}.bin")
    (d / f"{name}.json").write_text(json.dumps({"type": t, "shape": list(reversed(a.shape))}))


def undump(d: Path, name: str, dt: np.dtype) -> np.ndarray:
    meta = json.loads((d / f"{name}.json").read_text())
    a = np.fromfile(d / f"{name}.bin", dtype=dt)
    return a.reshape(list(reversed(meta["shape"])))


def run_pre(args, op: str, model: str, work: Path) -> None:
    cmd = [args.pre, op, model, str(work)]
    if op == "text":
        cmd.append(str(args.drop))
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        raise RuntimeError(f"xestrata-image-mf-pre {op}: {out.stderr.strip()}")


def cache_one(args, tok, uncond: np.ndarray, img: Path, work: Path) -> tuple[str, int, int]:
    # one image's scratch: the pix dump is about 12 MB, the latent 2 MB (a tmpfs fills up with 48 of them)
    """One image and its caption to the caches: a latent a budget's bucket (the same bucket once), the text once;
    returns (bucket tags, latent values, embed rows)."""
    with Image.open(img) as im:
        ow, oh = im.size
    work.mkdir(parents=True, exist_ok=True)
    tags: list[str] = []
    n = 0
    for budget in args.budgets:
        bw, bh, cw, ch, left, t, r, b = bucket(ow, oh, budget, args.step)
        tag = f"{bw:04d}x{bh:04d}"
        if tag in tags:
            continue
        for f in work.iterdir():
            f.unlink()
        dump(work, "pix", vae_input(img, bw, bh, (cw, ch), (left, t, r, b)))
        run_pre(args, "anima-vae" if args.cache == "an" else "vae", args.vae, work)
        lat = undump(work, "latent", np.float32)
        # the posterior's standard deviation beside the mean: a training step draws its latent from them
        sd = {"latents_sd": undump(work, "latent_sd", np.float32)} if (work / "latent_sd.json").is_file() else {}
        np.savez(img.parent / f"{img.stem}_{tag}{LATENT_TAG}", latents=lat, original_size=np.array([ow, oh], np.int32),
                 crop_ltrb=np.array([left, t, r, b], np.int32), **sd)
        tags.append(tag)
        n += int(lat.size)
    caption = (img.parent / (img.stem + ".txt")).read_text().strip()
    if args.cache == "an":   # Anima's: the Qwen3 ids, and the T5 ids with their weights for the LLM adapter
        for base, (q, t5, t5w) in (("prompt", anima_tokens(tok, args.t5tok, caption)), ("uncond", uncond)):
            dump(work, f"{base}_ids", np.array(q, np.int32))
            dump(work, f"{base}_t5", np.array(t5, np.int32))
            dump(work, f"{base}_t5w", np.array(t5w, np.float32))
        return " ".join(tags), n, 0
    ids = np.array(prompt_ids(tok, caption, 0, args.family), np.int32)
    dump(work, "prompt_ids", ids)
    dump(work, "uncond_ids", uncond)
    if args.cache == "ms":   # MOSS-Image's text rows: every image's at once afterwards (text_cache, moss-text)
        return " ".join(tags), n, len(ids)
    run_pre(args, "text", args.text, work)
    return " ".join(tags), n, text_cache(args, img, work)


def text_cache(args, img: Path, work: Path) -> int:
    """the text cache beside the image from the states its work dir holds; returns the prompt's rows"""
    emb = undump(work, "prompt_embeds", np.uint16)
    uemb = undump(work, "uncond_embeds", np.uint16)
    np.savez(img.parent / f"{img.stem}{TEXT_TAG}",
             prompt_embeds=emb, attn_mask=np.ones(emb.shape[0], np.int32),
             uncond_embeds=uemb, uncond_mask=np.ones(uemb.shape[0], np.int32),
             caption_dropout_rate=np.array(args.dropout, np.float32))
    return int(emb.shape[0])


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """FP32 -> bf16's round-to-nearest-even, widened back to FP32: what the cache's uint16 bits hold (the engine's
    own `bf16_from_f32`, strata/kernels/bf16_bits.hpp)."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32)
    kept = ((u.astype(np.uint64) + np.uint64(0x7FFF) + r.astype(np.uint64)) & np.uint64(0xFFFF0000)).astype(np.uint32)
    return kept.view(np.float32).reshape(np.shape(a))


def rel(g: np.ndarray, w: np.ndarray) -> float:
    """The relative L1 the image engine's probes report with (mf_probe's `compare`): the sum of the differences over
    the sum of the reference.  A per-element maximum would be the wrong yardstick here - the encoder's own values run
    to 5.9 and cross zero, so one small value's rounding reads as a huge ratio."""
    g, w = g.astype(np.float64), w.astype(np.float64)
    return float(np.abs(g - w).sum() / max(np.abs(w).sum(), 1e-12))


def probe_rel(args, op: str, model: str, golden_dir: Path, tag: str) -> float | None:
    """The relative L1 the inference probe itself reports for the same tensor (`xestrata-image-mf-probe vae-enc`'s
    `mean` line, `text`'s `text` line).  It is the number preprocessing must land on: the same encoder, the same
    states path, the same sums - so the check is the equality of the two, not a bound fitted to what we measured.
    None when the probe is not there to ask."""
    if not args.probe:
        return None
    out = subprocess.run([args.probe, op, model, str(golden_dir)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{args.probe} {op}: {(out.stderr or out.stdout).strip()[:200]}")
    for line in out.stdout.splitlines():
        f = line.split()
        if f and f[0] == tag and "rel" in f:
            return float(f[f.index("rel") + 1])
    return None


def cos(g: np.ndarray, w: np.ndarray) -> float:
    g, w = g.astype(np.float64).ravel(), w.astype(np.float64).ravel()
    return float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w)))


def golden(args, work: Path, g: Path) -> int:
    """The two paths on a golden run's own inputs, against the reference's outputs (SPEC: preprocessing's match is
    checked against the inference goldens).  The latent half runs where the run left `ref0_pix.npy` and `ref0_mean.npy`
    (an edit run's first reference through the encoder); the text half runs for a generation run only - an edit run's
    states come in through the vision encoder and its system turn is 64 ids, not 34, and both are the edit golden's
    own business (src/image/mf_probe.cpp)."""
    bad = 0
    work.mkdir(parents=True, exist_ok=True)
    for f in work.iterdir():
        f.unlink()
    if (g / "ref0_pix.npy").is_file() and (g / "ref0_mean.npy").is_file():
        dump(work, "pix", np.load(g / "ref0_pix.npy").astype(np.float32))
        run_pre(args, "vae", args.vae, work)
        got = undump(work, "latent", np.float32)
        want = np.load(g / "ref0_mean.npy").reshape(got.shape)
        e = rel(got, want)
        p = probe_rel(args, "vae-enc", args.vae, g, "mean")
        ok = abs(e - p) <= 1e-4 if p is not None else e <= args.tol
        bad += 0 if ok else 1
        print(f"latent  {got.shape} cos {cos(got, want):.6f}  rel {e:.3e}  (the probe's own mean line: "
              f"{p if p is None else f'{p:.4f}'}; tol {args.tol:g})  max|d| {np.abs(got - want).max():.3e}")
    elif (g / "ids.npy").is_file() and not (g / "vl_pixel_values.npy").is_file():
        ids = np.load(g / "ids.npy").astype(np.int32)
        lens = np.load(g / "ids_lens.npy") if (g / "ids_lens.npy").is_file() else np.array([len(ids)])
        dump(work, "prompt_ids", ids[:int(lens[0])])
        if (g / "neg.npy").is_file():
            dump(work, "uncond_ids", np.load(g / "neg.npy").astype(np.int32).ravel())
        run_pre(args, "text", args.text, work)
        got = undump(work, "prompt_embeds32", np.float32)
        want = np.load(g / "txt.npy")[: (int(lens[0]) - args.drop)]
        if got.shape != want.shape:
            print(f"FAIL text: the engine's {got.shape} where the golden's txt.npy rows are {want.shape}")
            bad += 1
        else:
            e, bf = rel(got, want), rel(bf16_bits(got), want)
            p = probe_rel(args, "text", args.text, g, "text")
            ok = abs(e - p) <= 1e-4 if p is not None else e <= args.tol
            bad += 0 if ok else 1
            print(f"text    {got.shape} cos {cos(got, want):.6f}  rel {e:.3e}  (the probe's own text line: "
                  f"{p if p is None else f'{p:.4f}'}; tol {args.tol:g})  the cache's bf16 of it {bf:.3e}  "
                  f"max|d| {np.abs(got - want).max():.3e}")
    else:
        raise SystemExit(f"{g}: neither ref0_pix.npy nor a generation run's ids.npy is there")
    return 1 if bad else 0


def check_dir(d: Path, budgets: int | list[int], step: int) -> int:
    """The caches beside the images: names, keys, types, and that the size and the crop recompute the same buckets
    (one a budget, the same bucket once)."""
    if isinstance(budgets, int):
        budgets = [budgets]
    bad = 0
    seen: dict[str, int] = {}
    for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT):
        stem = img.stem
        lat = [p for p in sorted(d.glob(f"{stem}_*{LATENT_TAG}"))]
        txt = d / f"{stem}{TEXT_TAG}"
        cap = d / f"{stem}.txt"
        with Image.open(img) as im:
            ow, oh = im.size
        want: dict[str, tuple[int, ...]] = {}
        for budget in budgets:
            bk = bucket(ow, oh, budget, step)
            want.setdefault(f"{bk[0]:04d}x{bk[1]:04d}", bk)
        why = []
        if sorted(p.name[len(stem) + 1:-len(LATENT_TAG)] for p in lat) != sorted(want):
            why.append(f"{len(lat)} latent caches, {len(want)} buckets")
        if not txt.is_file():
            why.append("no text cache")
        if not cap.is_file():
            why.append("no caption")
        if why:
            bad += 1
            print(f"FAIL {img.name}: " + ", ".join(why))
            continue
        y = np.load(txt)
        ok = True
        for lf in lat:
            tag = lf.name[len(stem) + 1:-len(LATENT_TAG)]
            z = np.load(lf)
            a, orig, crop = z["latents"], z["original_size"], z["crop_ltrb"]
            wb, hb, wc, hc, left, t, r, b = want[tag]
            sd_ok = "latents_sd" not in z or (z["latents_sd"].shape == a.shape
                                              and str(z["latents_sd"].dtype) == "float32")
            ok = ok and (str(a.dtype) == "float32" and a.ndim == 3 and sd_ok
                         and a.shape[1:] == (hb // 16, wb // 16) and list(orig) == [ow, oh]
                         and list(crop) == [left, t, r, b]
                         and int(crop[2] - crop[0]) == wb and int(crop[3] - crop[1]) == hb)
            seen[tag] = seen.get(tag, 0) + 1
        ok = (ok and str(y["prompt_embeds"].dtype) == "uint16" and y["prompt_embeds"].ndim == 2
              and str(y["attn_mask"].dtype) == "int32" and len(y["attn_mask"]) == y["prompt_embeds"].shape[0]
              and str(y["uncond_embeds"].dtype) == "uint16" and y["uncond_embeds"].ndim == 2
              and len(y["uncond_mask"]) == y["uncond_embeds"].shape[0]
              and str(y["caption_dropout_rate"].dtype) == "float32")
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {img.name} {ow}x{oh} -> {' '.join(sorted(want))} latent {a.shape} "
              f"embeds {y['prompt_embeds'].shape} uncond {y['uncond_embeds'].shape}")
    print(f"mf_pre: {sum(seen.values())} latent(s) in {len(seen)} bucket(s), {bad} failure(s)")
    for tag, n in sorted(seen.items()):
        print(f"   {tag}: {n}")
    return 1 if bad else 0


def toml_setting(path: Path) -> dict:
    """The dataset TOML, flattened: the fields v1 takes, the rest refused with their name."""
    import tomllib
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    out: dict = {}
    for k in set(cfg.get("general", {})) - GENERAL_OK:
        raise ValueError(f"[general].{k} is not a field v1 takes")
    out.update(cfg.get("general", {}))
    dss = cfg.get("datasets", [])
    if len(dss) != 1:
        raise ValueError(f"v1 reads exactly one [[datasets]] table, {len(dss)} are there")
    ds = dss[0]
    for k in set(ds) - DATASET_OK:
        raise ValueError(f"[[datasets]].{k} is not a field v1 takes")
    subs = ds.get("subsets", [])
    if len(subs) != 1:
        raise ValueError(f"v1 reads exactly one [[datasets.subsets]] table, {len(subs)} are there")
    sub = subs[0]
    for k in set(sub) - SUBSET_OK:
        raise ValueError(f"[[datasets.subsets]].{k} is not a field v1 takes")
    if int(sub.get("num_repeats", 1)) != 1:
        raise ValueError("num_repeats other than 1: v1 does not repeat a data set")
    out.update({k: v for k, v in ds.items() if k != "subsets"})
    out.update(sub)
    for k, want in (("caption_extension", ".txt"), ("keep_tokens", 1), ("shuffle_caption", False),
                    ("enable_bucket", True), ("bucket_no_upscale", True), ("bucket_reso_steps", 32),
                    ("cache_latents", True), ("cache_latents_to_disk", True),
                    ("cache_text_encoder_outputs", True), ("cache_text_encoder_outputs_to_disk", True),
                    ("batch_size", 1)):
        got = out.get(k, want)
        if got != want:
            raise ValueError(f"{k} = {got!r}: preprocessing takes the SPEC's {want!r} only")
    return out


def write_caches(img: Path, tag: str, crop: tuple[int, int, int, int], orig: tuple[int, int], lat: np.ndarray,
                 emb: np.ndarray, uemb: np.ndarray, dropout: float) -> None:
    """The two `.npz` files beside the image, in sd-scripts' keys and types (SPEC's cache table)."""
    np.savez(img.parent / f"{img.stem}_{tag}{LATENT_TAG}", latents=lat,
             original_size=np.array(orig, np.int32), crop_ltrb=np.array(crop, np.int32))
    np.savez(img.parent / f"{img.stem}{TEXT_TAG}", prompt_embeds=emb, attn_mask=np.ones(emb.shape[0], np.int32),
             uncond_embeds=uemb, uncond_mask=np.ones(uemb.shape[0], np.int32),
             caption_dropout_rate=np.array(dropout, np.float32))


def stub_pre(op: str, work: Path) -> int:
    """`--stub-pre`: the same dumps without a GPU, one 16 x 16 block mean of a channel per latent value and an id's
    own bits per row.  It keeps the cache path (names, keys, shapes, the crop bookkeeping) testable where the VAE and
    the text encoder cannot run.  These are NOT the engine's values, so no parity is ever claimed through it."""
    if op == "vae":
        pix = undump(work, "pix", np.float32)
        _, H, W = pix.shape
        h, w = H // 16, W // 16
        lat = np.stack([pix[c % 3].reshape(h, 16, w, 16).mean(axis=(1, 3)) for c in range(128)]).astype(np.float32)
        dump(work, "latent", np.ascontiguousarray(lat))
        print(f"stub latent 128 x {h} x {w}")
        return 0
    for base in ("prompt", "uncond"):
        if not (work / f"{base}_ids.json").is_file():
            continue
        ids = undump(work, f"{base}_ids", np.int32)
        rows = max(len(ids) - DROP, 0)
        emb = np.zeros((rows, 2560), np.uint16)
        for r in range(rows):
            emb[r, :] = int(ids[DROP + r]) & 0xFFFF
        dump(work, f"{base}_embeds", emb)
        dump(work, f"{base}_embeds32", emb.astype(np.float32))
        print(f"stub {base}: {rows} rows")
    return 0


def selftest(tmp: Path) -> int:
    """The host-side rules and the cache path, no GPU: the bucket rule against SPEC's three worked sizes, the cover
    and crop bookkeeping, the dumps, and the caches read back by `--check`."""
    bad = 0
    for ow, oh, want in ((1792, 2400, (864, 1184)), (2752, 1536, (1344, 736)), (1696, 2528, (832, 1248)),
                         (1024, 1024, (1024, 1024)), (512, 1536, (512, 1536)), (2048, 512, (2048, 512)),
                         (3000, 1000, (1760, 576)), (100, 100, (96, 96))):
        bw, bh, cw, ch, left, t, r, b = bucket(ow, oh, 1024, 32)
        # never enlarged, the cover at least the bucket, and under the budget the image's own size is the cover
        ok = ((bw, bh) == want and bw * bh <= 1024 * 1024 and cw >= bw and ch >= bh and r - left == bw and b - t == bh
              and (cw, ch) == (ow, oh) if ow * oh <= 1024 * 1024 else (cw, ch) != (0, 0))
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} bucket {ow}x{oh} -> {bw}x{bh} (want {want[0]}x{want[1]}), cover {cw}x{ch}, "
              f"crop {left},{t},{r},{b}, {(bw // 16) * (bh // 16)} tokens")
    # the resize: against the area definition written the slow way, on shapes that do not divide evenly
    rng = np.random.default_rng(11)
    for h, w, nh, nw in ((7, 5, 4, 3), (16, 9, 8, 5), (23, 17, 11, 7)):
        px = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
        ey, ex = (np.arange(nh + 1) * h / nh), (np.arange(nw + 1) * w / nw)
        slow = np.zeros((nh, nw, 3))
        for i in range(nh):
            for j in range(nw):
                acc = np.zeros(3)
                for y in range(h):
                    wy = min(ey[i + 1], y + 1) - max(ey[i], y)
                    for x in range(w):
                        wx = min(ex[j + 1], x + 1) - max(ex[j], x)
                        acc += max(wy, 0) * max(wx, 0) * px[y, x]
                slow[i, j] = acc / ((ey[i + 1] - ey[i]) * (ex[j + 1] - ex[j]))
        got = area_resize(px, nh, nw).astype(np.float64)
        ok = np.abs(got - np.clip(np.rint(slow), 0, 255)).max() < 1e-6
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} area {h}x{w} -> {nh}x{nw} matches the definition")
    d = tmp / "data"
    d.mkdir(parents=True, exist_ok=True)
    for n, ext in enumerate(("jpg", "png"), start=1):        # 128 x 96: under the budget, so it stays its own size
        a = (np.arange(3 * 96 * 128, dtype=np.float32).reshape(3, 96, 128) % 251) / 250.0
        Image.fromarray(np.ascontiguousarray(a.transpose(1, 2, 0).astype(np.uint8))).save(d / f"{n:02d}.{ext}")
        (d / f"{n:02d}.txt").write_text(f"stubword, subject number {n}\n")
    for stem, ext in (("01", "jpg"), ("02", "png")):
        img = d / f"{stem}.{ext}"
        work = tmp / "w" / stem
        work.mkdir(parents=True, exist_ok=True)
        with Image.open(img) as im:
            ow, oh = im.size
        bw, bh, cw, ch, left, t, r, b = bucket(ow, oh, 1024, 32)
        dump(work, "pix", vae_input(img, bw, bh, (cw, ch), (left, t, r, b)))
        dump(work, "prompt_ids", np.arange(40, dtype=np.int32))      # 34 of them are the system turn
        dump(work, "uncond_ids", np.arange(35, dtype=np.int32))
        stub_pre("vae", work)
        stub_pre("text", work)
        lat = undump(work, "latent", np.float32)
        emb = undump(work, "prompt_embeds", np.uint16)
        uemb = undump(work, "uncond_embeds", np.uint16)
        ok = lat.shape == (128, bh // 16, bw // 16) and emb.ndim == 2 and emb.shape[1] == 2560
        un = uemb[:1] if len(uemb) else emb[:0]
        write_caches(img, f"{bw:04d}x{bh:04d}", (left, t, r, b), (ow, oh), lat, emb, un, 0.0)
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} cache {img.name} latent {lat.shape} embeds {emb.shape}")
    bad += 0 if check_dir(d, 1024, 32) == 0 else 1
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="the image directory to cache, or to check with --check")
    ap.add_argument("--toml", help="the dataset setting; its buckets, caption extension and dropout are used as read")
    ap.add_argument("--vae", help="Mage-VAE's safetensors")
    ap.add_argument("--text", help="the text encoder's GGUF (Qwen3-VL-4B)")
    ap.add_argument("--pre", default="build/free/xestrata-image-mf-pre")
    ap.add_argument("--budget", default="512,768,1024",
                    help="the pixel budgets' sides, comma separated: a latent a budget's bucket (ai-toolkit's "
                         "Mage-Flow resolutions 512, 768, 1024)")
    ap.add_argument("--step", type=int, default=16, help="the bucket step (ai-toolkit's Mage-Flow: 16)")
    ap.add_argument("--dropout", type=float, default=CAPTION_DROPOUT,
                    help="v1's caption_dropout_rate, recorded in the cache")
    ap.add_argument("--neg", default="", help="the unconditional prompt (empty, as SPEC pins it)")
    ap.add_argument("--scratch", help="where the engine's dumps go")
    ap.add_argument("--check", action="store_true", help="only verify the caches already beside the images")
    ap.add_argument("--golden", help="a golden run's directory: run both paths against it")
    ap.add_argument("--limit", type=int, default=0, help="cache at most this many images (0: all)")
    ap.add_argument("--drop", type=int, default=34, help="the system turn's rows (34 generation, 64 an edit)")
    ap.add_argument("--probe",
                    help="xestrata-image-mf-probe: --golden gates on its own reported rel for the same tensor")
    ap.add_argument("--tol", type=float, default=4e-2,
                    help="the relative-L1 a golden may sit at: the inference probe already stands at 3.6e-2 (the "
                         "encoder's mean) and 3.2e-2 (the text states) on these two runs")
    ap.add_argument("--stub-pre", metavar="OP_DIR", help="act as the engine's tool without a GPU: OP DIR (test only)")
    ap.add_argument("--selftest", action="store_true", help="the host-side rules and the cache path, no GPU")
    ap.add_argument("--cache", default="mf", choices=("mf", "mt", "ms", "an", "qi"),
                    help="the cache files' model tag: mf for Mage-Flow (Mage-VAE), mt for MageTrail (the Flux2 VAE), "
                         "ms for MOSS-Image (its VAE, Qwen3.5-9B and its DiT's text embedder: --dit), an for Anima "
                         "(Wan 2.1's VAE, Qwen3-0.6B and its DiT's LLM adapter: --dit, --t5)")
    ap.add_argument("--dit",
                    help="MOSS-Image's DiT folder or Anima's DiT file (--cache ms, an): the text rows' last part")
    ap.add_argument("--out", help="Qwen-Image-2.1's caches and items.txt (--cache qi)")
    ap.add_argument("--trigger", default="", help="the trigger word --cache qi puts in a caption ([name], [trigger])")
    ap.add_argument("--t5", help="Anima's T5 tokenizer.json (--cache an): the LLM adapter's ids")
    a = ap.parse_args()
    global LATENT_TAG, TEXT_TAG
    LATENT_TAG, TEXT_TAG = f"_{a.cache}.npz", f"_{a.cache}_te.npz"
    a.family = {"ms": "moss-image", "an": "anima"}.get(a.cache, "mage-flow")
    if a.cache in ("ms", "an") and not a.check and not a.selftest and not a.dit:
        raise SystemExit(f"--cache {a.cache} needs --dit (the DiT whose text part makes the text rows)")
    if a.cache == "an" and not a.check and not a.selftest and not a.t5:
        raise SystemExit("--cache an needs --t5 (Anima's T5 tokenizer.json)")
    if a.selftest:
        with tempfile.TemporaryDirectory(prefix="mf-pre-test-") as td:
            return selftest(Path(td))
    if a.stub_pre:
        work = Path(a.stub_pre).expanduser()
        if not work.is_dir():
            raise SystemExit(f"--stub-pre wants a directory of dumps, {a.stub_pre} is not one")
        return stub_pre("vae" if (work / "pix.json").is_file() else "text", work)
    if a.toml:
        s = toml_setting(Path(a.toml))
        a.dropout = float(s.get("caption_dropout_rate", a.dropout))
        if not a.dir:
            a.dir = s.get("image_dir")
        res = s.get("resolution", a.budget)
        a.budget = ",".join(str(int(x)) for x in (res if isinstance(res, (list, tuple)) else [res]))
    a.budgets = [int(x) for x in str(a.budget).split(",") if x.strip()]
    a.budget = max(a.budgets)
    if a.check:
        return check_dir(Path(a.dir), a.budgets, a.step)
    if a.cache == "qi":
        if not (a.dir and a.out and a.vae and a.text):
            raise SystemExit("--cache qi needs --dir, --out, --vae and --text")
        return qi_prep(a)
    if not (a.vae and a.text):
        raise SystemExit("--vae and --text are needed")
    work = Path(a.scratch or tempfile.mkdtemp(prefix="mf-pre-")) / "w"
    if a.golden:
        return golden(a, work, Path(a.golden))
    if not a.dir:
        raise SystemExit("--dir or a [datasets.subsets].image_dir is needed")
    d = Path(a.dir)
    tok = load_tokenizer(a.text, a.family)
    if a.cache == "an":
        import strata_unigram   # the T5 tokenizer serve reads (tools/strata_unigram.py)
        a.t5tok = strata_unigram.UnigramTokenizer.from_hf_json(a.t5)
        uncond = anima_tokens(tok, a.t5tok, a.neg)
    else:
        uncond = np.array(prompt_ids(tok, a.neg, 0, a.family), np.int32)
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    if a.limit:
        imgs = imgs[:a.limit]
    bad = 0
    done: list[Path] = []
    for p in imgs:
        if not (d / (p.stem + ".txt")).is_file():
            print(f"FAIL {p.name}: no caption")
            bad += 1
            continue
        try:
            tag, n, rows = cache_one(a, tok, uncond, p, work / p.stem)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"FAIL {p.name}: {e}")
            bad += 1
            continue
        print(f"ok   {p.name} {tag} latent {n} values, {rows} {'ids' if a.cache == 'ms' else 'rows'}")
        done.append(p)
    if a.cache in ("ms", "an") and done:   # the text encoder and the DiT opened once for every image
        lst = work / "moss-text.list"
        lst.write_text("".join(f"{work / p.stem}\n" for p in done))
        op = "moss-text" if a.cache == "ms" else "anima-text"
        out = subprocess.run([a.pre, op, a.text, a.dit, str(lst)], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"xestrata-image-mf-pre {op}: {out.stderr.strip()}")
        for p in done:
            print(f"ok   {p.name} text {text_cache(a, p, work / p.stem)} rows")
    print(f"mf_pre: {len(imgs)} image(s), {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
