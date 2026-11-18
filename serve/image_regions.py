# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2024 Microsoft and The HuggingFace Inc. team
# SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
"""serve/image_regions.py - words to masks, and the background taken away: the image tools' auxiliary models in the
image engine (src/image: sam3, florence2, birefnet), their weights image_aux's.

Words to masks (segment): SAM 3.1 (Meta's, under the SAM License: fetched only once the user has agreed to it,
image_aux.accept) finds every instance of a short phrase with a score and a mask; Florence-2 (Microsoft's, MIT) large,
or base where the large does not fit, reads a phrase as its referring-expression task and answers with polygons, which
the engine fills.  "auto" takes SAM 3 where it may (agreed, the phrase within CLIP's 32 tokens) and Florence-2
otherwise, and Florence-2 also where SAM 3 has no room on the GPU.  Florence-2 also answers boxes (its open-vocabulary
detection: task "box") and the phrases of a caption (its phrase grounding: task "grounding"), which SAM 3 has no task
for.  Florence-2's answer is read here as its processor reads it (processing_florence2.py's post-processor: the
<loc_*> bins at their centres, (bin + 0.5) / 1000 of the side).

The background (matte): BiRefNet (Peng Zheng's, MIT), one of its versions (BIREFNET_VERSIONS: general, the default, at
1024 x 1024; lite (Swin-T); lite-2k at 2560 x 1440; portrait; matting, for hair, fur and glass; hr at 2048 x 2048;
dynamic at the picture's own size), the alpha at the picture's size, with refine_foreground the foreground's colors
estimated (BiRefNet's blur fusion) so a cut-out laid on another background has no fringe of the old one.  A version
that does not fit the GPU gives way to lite (lite-2k for the large ones).

Both run in the image jobs' GPU turn (images.gpu_turn), synchronously, the engine's AUXNEED its need.  The entry
points: POST /v1/images/segmentations (vLLM-Omni's: its fields and answer, `task` XeStrata's), POST
/v1/images/remove-background (vLLM-Omni's images answer, fal's BiRefNet fields, and background_color, mask_grow and
mask_blur XeStrata's) and Stability's /v2beta/stable-image/edit/remove-background; the chat's find_region,
edit_region and remove_background (images.ImageTools); and the edits' `region` (region_mask: the instances' union
grown a little, as OpenAI's mask).
"""
from __future__ import annotations

import base64
import contextlib
import functools
import io
import re
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from serve import image_aux
from serve import images as img

# ------------------------------------------------------------------------------------------- the models
SAM3 = "sam3.1"
FLORENCE = ("florence-2-large", "florence-2-base")
SEG_MODELS = (SAM3, *FLORENCE)
# what a request may call them (vLLM-Omni's served names, Hugging Face's repositories) -> SEG_MODELS' id
SEG_ALIASES = {"sam3": SAM3, "sam3.1": SAM3, "facebook/sam3": SAM3, "facebook/sam3.1": SAM3, "comfy-org/sam3.1": SAM3,
               "florence-2": FLORENCE[0], "florence-2-large": FLORENCE[0], "microsoft/florence-2-large": FLORENCE[0],
               "florence-2-base": FLORENCE[1], "microsoft/florence-2-base": FLORENCE[1]}
SEG_TASKS = ("mask", "box", "grounding")
SAM3_MAX_IDS = 32            # CLIP's context: SAM 3's text tower reads at most 32 tokens
# SAM 3 finds a concept (a noun phrase: "red car"); a longer phrase is a referring expression ("the girl on the left",
# "the umbrella the man holds"), which Florence-2's task reads and SAM 3's does not: auto gives it to Florence-2
SAM3_MAX_WORDS = 4
# Florence-2's task prompts (processing_florence2.py's task_prompts_with_input) for SEG_TASKS
FLORENCE_PROMPTS = {"mask": "Locate {} in the image with mask", "box": "Locate {} in the image.",
                    "grounding": "Locate the phrases in the caption: {}"}
FLORENCE_BEAMS = 3           # its generation_config's num_beams
FLORENCE_MAX_NEW = 1024
FLORENCE_VOCAB = 50265       # BART's vocabulary; the processor's tokens after it (FLORENCE_ADDED)
FLORENCE_ADDED = (["<od>", "</od>", "<ocr>", "</ocr>"] + [f"<loc_{i}>" for i in range(1000)]
                  + ["<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>",
                     "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>", "<region_to_desciption>",
                     "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>"])
FLORENCE_SPECIAL = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>", 50264: "<mask>"}
# the words Florence-2's phrase grounding does not keep: its processor's black_list_of_phrase_grounding (Microsoft's
# processing_florence2.py, Apache-2.0), as it is
GROUNDING_SKIP = frozenset((
    "it", "I", "me", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "theirs", "one", "oneself", "we", "us", "our", "ours", "its", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves", "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
    "all", "another", "any", "anybody", "anyone", "anything", "each", "everybody", "everyone", "everything", "few",
    "many", "nobody", "none", "several", "some", "somebody", "someone", "something", "each other", "one another",
    "the image", "image", "images", "the", "a", "an", "a group", "other objects", "lots", "a set"))
# an edit's region grown past its mask (the share of the picture's short side, at least REGION_GROW_MIN pixels): the
# masks hug the object, and its edge belongs to the redrawing
REGION_GROW = 0.01
REGION_GROW_MIN = 4

BIREFNET_VERSIONS = ("general", "lite", "lite-2k", "portrait", "matting", "hr", "dynamic")
# fal's BiRefNet v2 `model` names -> the version (its API page's list: "General Use (Heavy)" is BiRefNet_lite there)
BIREFNET_FAL = {"General Use (Light)": "general", "General Use (Light 2K)": "lite-2k", "General Use (Heavy)": "lite",
                "Matting": "matting", "Portrait": "portrait", "General Use (Dynamic)": "dynamic"}
# each version's resolution (width, height), as its model card runs it; dynamic's the picture's own (dynamic_size)
BIREFNET_SIZE = {"general": (1024, 1024), "lite": (1024, 1024), "portrait": (1024, 1024), "matting": (1024, 1024),
                 "hr": (2048, 2048), "lite-2k": (2560, 1440), "dynamic": None}
DYNAMIC_SIDES = (256, 2304)  # BiRefNet_dynamic's trained range
BIREFNET_SMALLER = {"general": "lite", "portrait": "lite", "matting": "lite", "hr": "lite-2k", "dynamic": "lite-2k",
                    "lite-2k": "lite"}
OUTPUT_FORMATS = ("png", "webp", "gif", "jpeg", "jxl")


class MemoryShort(img.ImageJobFailed):
    """The engine had no room for the model on the GPU (its ERROR memory: ...)."""


def root_of(gen) -> Path:
    return image_aux.aux_dir(gen.cfg)


def weights(gen, names: list[str]) -> list[str]:
    """The paths of image_aux's `names`, or AuxPending while any is downloading (its download started)."""
    root = root_of(gen)
    got = [image_aux.ensure(n, root, gen._say) for n in names]
    pending = [g for g in got if g["status"] != "ready"]
    if pending:
        raise img.AuxPending(pending)
    return [g["path"] for g in got]


def ready(gen, names: list[str]) -> bool:
    root = root_of(gen)
    return all(image_aux.ready(n, root) for n in names)


@functools.lru_cache(maxsize=4)
def _clip(path: str):
    return img._tokenizer_module().ClipTokenizer.from_hf(path)


@functools.lru_cache(maxsize=4)
def _bart(path: str):
    return img._tokenizer_module().Tokenizer.from_hf(path)


def seg_files(model: str) -> list[str]:
    """image_aux's names a segmenter reads: its weights, then its tokenizer."""
    if model == SAM3:
        return ["sam3", "clip-tokenizer"]
    return [model, "florence-2-tokenizer"]


# ------------------------------------------------------------------------------------------- words to masks
def segmenters(gen, prompt: str, task: str, model: str | None) -> list[str]:
    """The segmenters to try for `prompt`, in order: `model` alone where one is named; else SAM 3 (where the user has
    agreed to its license, the task is one it has, and the phrase is a concept of at most SAM3_MAX_WORDS words within
    CLIP's 32 tokens) before Florence-2 large and base."""
    if model not in (None, "", "auto"):
        m = SEG_ALIASES.get(str(model).lower())
        if m is None:
            raise img.ImageRequestError(f"model: one of auto, {', '.join(SEG_MODELS)}, not {model!r}")
        if m == SAM3:
            if task == "grounding":
                raise img.ImageRequestError("task grounding: Florence-2's alone (model florence-2-large or -base)")
            if image_aux.needs_consent("sam3", root_of(gen)):
                raise img.ImageRequestError(sam3_consent_words())
        return [m] + ([FLORENCE[1]] if m == FLORENCE[0] else [])
    out = list(FLORENCE)
    if task != "grounding" and len(prompt.split()) <= SAM3_MAX_WORDS \
            and not image_aux.needs_consent("sam3", root_of(gen)):
        tok = ready(gen, ["clip-tokenizer"])
        if not tok or len(_clip(weights(gen, ["clip-tokenizer"])[0]).encode(prompt, 10 ** 6)) <= SAM3_MAX_IDS:
            out.insert(0, SAM3)
    return out


def sam3_consent_words() -> str:
    return ("SAM 3.1 is under the SAM License, which you have not agreed to here: read it and agree on the web page's "
            "Images view (or with xestrata --setup) - Florence-2 (MIT) finds regions until then")


def aux_job(gen, kind: str, path: str, picture: bytes, need_args=()) -> dict:
    """The job gpu_turn takes for an auxiliary model's run on `picture`."""
    w, h = img.image_dims(picture)
    return {"aux": {"kind": kind, "path": path, "need_args": list(need_args)}, "n": 1, "steps": 1, "width": w,
            "height": h, "seed": 0}


@contextlib.contextmanager
def _work(gen):
    work = Path(tempfile.mkdtemp(prefix="regions-", dir=gen.dir))
    try:
        yield work
    finally:
        for f in work.iterdir():
            with contextlib.suppress(OSError):
                f.unlink()
        with contextlib.suppress(OSError):
            work.rmdir()


def _ask(gen, line: str, want: str) -> dict:
    try:
        return gen.aux_ask(line, want)
    except img.ImageJobFailed as e:
        if str(e).startswith("memory: "):
            raise MemoryShort(str(e)[8:]) from e
        if "alloc_device:" in str(e) and "refused" in str(e):
            raise MemoryShort("GPU memory allocation was refused") from e
        raise


@img.on_free_gpu
def segment(svc, gen, picture: bytes, prompt: str, task: str = "mask", model: str | None = None,
            score_threshold: float = 0.5, mask_threshold: float = 0.5, cancel=None) -> dict:
    """The instances of `prompt` in `picture` (an image the engine reads: PNG, JPEG...) -> {"model", "width",
    "height", "task", "instances": [{"score": float or None, "box": [x0, y0, x1, y1], "mask": a bool array or None,
    "label": its phrase (Florence-2's boxes)}], "seconds", "engine_s", "fallback": why another model answered}.
    SAM 3's highest score first; Florence-2's in its answer's order."""
    prompt = " ".join(str(prompt or "").split())
    if not prompt:
        raise img.ImageRequestError("prompt: the words of what to find")
    if task not in SEG_TASKS:
        raise img.ImageRequestError(f"task: one of {', '.join(SEG_TASKS)}, not {task!r}")
    order = segmenters(gen, prompt, task, model)
    fallback = None
    # the first whose files are here, starting the first's download where none is (the next request takes it)
    chosen = next((m for m in order if ready(gen, seg_files(m))), None)
    if chosen is None:
        weights(gen, seg_files(order[0]))
        chosen = order[0]
    elif chosen != order[0]:
        with contextlib.suppress(img.AuxPending):
            weights(gen, seg_files(order[0]))
        fallback = f"{order[0]} is downloading"
    if model in (None, "", "auto") and SAM3 not in order and task != "grounding":
        fallback = ("SAM 3's license not agreed to" if image_aux.needs_consent("sam3", root_of(gen))
                    else "a phrase longer than a concept: Florence-2 reads it")
    W, H = img.image_dims(picture)
    t0 = time.monotonic()
    first = weights(gen, seg_files(chosen))[0]
    job = aux_job(gen, "sam3" if chosen == SAM3 else "florence", first, picture,
                  () if chosen == SAM3 else (FLORENCE_BEAMS, FLORENCE_MAX_NEW))
    result = None
    with img.gpu_turn(svc, gen, job, cancel):
        gen._live(stage="segment", part=0.0)
        with _work(gen) as work:
            pic = work / "picture"
            pic.write_bytes(picture)
            for m in order[order.index(chosen):]:
                if not ready(gen, seg_files(m)):
                    continue
                try:
                    got = (_sam3(gen, m, pic, work, prompt, score_threshold, mask_threshold, W, H) if m == SAM3
                           else _florence(gen, m, pic, work, prompt, task, W, H))
                except MemoryShort as e:
                    gen._say(f"image tools: {m} has no room on the GPU ({e}); the next model")
                    fallback = f"{m} did not fit the GPU"
                    continue
                result = {"model": m, **got}
                break
        if result is None:
            raise img.ImageJobFailed(f"no segmenter fits the GPU now ({fallback})")
    if task == "box":
        for i in result["instances"]:
            i["mask"] = None
    gen.last = {"seconds": round(time.monotonic() - t0, 2)}
    gen.end(None)
    return {"width": W, "height": H, "task": task, **result, "seconds": round(time.monotonic() - t0, 3),
            "fallback": fallback}


def _sam3(gen, m: str, pic: Path, work: Path, prompt: str, score: float, mask: float, W: int, H: int) -> dict:
    import numpy as np
    path, tok = weights(gen, seg_files(m))
    ids = _clip(tok).encode(prompt, SAM3_MAX_IDS)
    prefix = work / "mask"
    got = _ask(gen, f"SEGMENT {path} {pic} {','.join(map(str, ids))} {float(score)!r} {float(mask)!r} {prefix}",
               "SEGMENT")
    out = []
    for i, entry in enumerate(e for e in str(got.get("boxes") or "").split(";") if e):
        x0, y0, x1, y1, s = (float(v) for v in entry.split(","))
        f = Path(f"{prefix}.{i}")
        mk = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(H, W) > 0 if f.is_file() else None
        out.append({"score": round(s, 4), "box": [x0, y0, x1, y1], "mask": mk})
    return {"instances": out, "engine_s": float(got.get("segment_s") or 0.0)}


def florence_ids(tok, prompt: str, task: str) -> list[int]:
    """The task's prompt (FLORENCE_PROMPTS) as Florence-2's processor tokenizes it: <s> ... </s>."""
    return [0] + tok.encode(FLORENCE_PROMPTS[task].format(prompt)) + [2]


def florence_text(tok, ids: list[int]) -> str:
    """Florence-2's answer as its processor decodes it (skip_special_tokens off): BART's tokens as text, the special
    ones and the processor's (FLORENCE_ADDED) by name."""
    out, run = [], []
    for i in ids:
        name = FLORENCE_SPECIAL.get(i) or (FLORENCE_ADDED[i - FLORENCE_VOCAB]
                                           if FLORENCE_VOCAB <= i < FLORENCE_VOCAB + len(FLORENCE_ADDED) else None)
        if name is None:
            run.append(i)
            continue
        if run:
            out.append(tok.decode(run))
            run = []
        out.append(name)
    if run:
        out.append(tok.decode(run))
    return "".join(out)


def _dequant(v: int, side: int) -> float:
    return (v + 0.5) * side / 1000.0


_MARK = re.compile(r"(?:loc_(\d+)|sep|poly|/poly)>")


def florence_pieces(text: str) -> list[tuple[str | None, int | None, str, str]]:
    """Florence-2's answer (its <s>, </s> and <pad> dropped) cut at each "<": for each piece, its mark ("loc", "sep",
    "poly" or "/poly" where the piece starts as one does, else None), the bin of a <loc_N>, what follows the mark (the
    whole piece without one) and the whole piece.  The first piece, before any "<", has no mark."""
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    first, *rest = text.split("<")
    out: list[tuple[str | None, int | None, str, str]] = [(None, None, first, first)]
    for piece in rest:
        m = _MARK.match(piece)
        if m is None:
            out.append((None, None, piece, piece))
        else:
            mark = "loc" if m.group(1) is not None else piece[:m.end() - 1]
            out.append((mark, int(m.group(1)) if mark == "loc" else None, piece[m.end():], piece))
    return out


def florence_polygons(text: str, W: int, H: int) -> list[list[list[float]]]:
    """A referring-expression segmentation's answer -> its instances, each a list of polygons [x0, y0, x1, y1, ...]:
    a stretch of four or more marks with nothing between them (its phrase may be empty) is cut into parts, those between
    <poly> and the next </poly> where it has both, else the whole stretch, and a part with polygons is an instance;
    within a part, each row of bins that a <sep> or the part's end closes is a polygon, its bins in (x, y) pairs."""
    out = []
    stretch: list[tuple[str, int | None]] = []
    for mark, bin_, after, _ in florence_pieces(text) + [(None, None, "", "")]:
        if mark is not None:
            stretch.append((mark, bin_))
            if not after:
                continue
        if len(stretch) >= 4:
            if {"poly", "/poly"} <= {m for m, _ in stretch}:
                parts: list[list[tuple[str, int | None]]] = []
                part: list[tuple[str, int | None]] | None = None
                for m in stretch:
                    if part is None:
                        part = [] if m[0] == "poly" else None
                    elif m[0] == "/poly":
                        parts.append(part)
                        part = None
                    else:
                        part.append(m)
            else:
                parts = [stretch]
            for part in parts:
                polys, row = [], []
                for kind, b in part + [("sep", None)]:
                    if kind == "loc":
                        row.append(cast(int, b))   # a <loc_N> piece always has its bin
                        continue
                    pairs = row[:len(row) // 2 * 2] if kind == "sep" else []
                    if pairs:
                        polys.append([_dequant(v, H if k % 2 else W) for k, v in enumerate(pairs)])
                    row = []
                if polys:
                    out.append(polys)
        stretch = []
    return out


def florence_boxes(text: str, W: int, H: int, grounding: bool = False) -> list[dict]:
    """An open-vocabulary detection's or a phrase grounding's answer -> [{"label", "box"}]: words directly followed by
    four or more bins name the boxes those bins make, four at a time (x0, y0, x1, y1).  The words run from the last
    "<" before the bins, or from the end of the bins a box took before; a piece is words even where it starts as a
    mark no box took, its mark's name kept, as the model's processor reads them.  The label is the words with their
    ends' white space taken off and their non-ASCII letters left out; words that run over a line break (past their
    leading white space) name nothing; a grounding skips the phrases GROUNDING_SKIP lists."""
    out = []
    pieces = florence_pieces(text)
    at, after_box = 0, False
    while at < len(pieces):
        words = pieces[at][2] if after_box else pieces[at][3]
        bins: list[int] = []
        nxt = at + 1
        if words:
            while nxt < len(pieces) and pieces[nxt][0] == "loc":
                bins.append(cast(int, pieces[nxt][1]))   # a <loc_N> piece always has its bin
                if pieces[nxt][2]:
                    break
                nxt += 1
        if len(bins) < 4:
            at, after_box = at + 1, False
            continue
        last = at + len(bins)   # the piece of the last bin: the next words start after its mark
        at, after_box = last, True
        if "\n" in words.lstrip():
            continue
        label = words.strip()
        if grounding and label in GROUNDING_SKIP:
            continue
        label = label.encode("ascii", errors="ignore").decode("ascii")
        for k in range(0, len(bins) - 3, 4):
            x0, y0, x1, y1 = bins[k:k + 4]
            out.append({"label": label, "box": [_dequant(x0, W), _dequant(y0, H), _dequant(x1, W), _dequant(y1, H)]})
    return out


def _florence(gen, m: str, pic: Path, work: Path, prompt: str, task: str, W: int, H: int) -> dict:
    import numpy as np
    path, tok_path = weights(gen, seg_files(m))
    tok = _bart(tok_path)
    ids = florence_ids(tok, prompt, task)
    got = _ask(gen, f"FLORENCE {path} {pic} {','.join(map(str, ids))} {FLORENCE_BEAMS} {FLORENCE_MAX_NEW}",
               "FLORENCE")
    answer = [int(v) for v in str(got.get("ids") or "").split(",") if v]
    text = florence_text(tok, answer)
    engine_s = float(got.get("florence_s") or 0.0)
    out: list[dict] = []
    if task == "mask":
        for polys in florence_polygons(text, W, H):
            f = work / f"poly{len(out)}"
            words = " ".join(",".join(f"{c:.2f}" for c in p) for p in polys if len(p) >= 6)
            if not words:
                continue
            _ask(gen, f"POLYMASK {W} {H} {f} {words}", "POLYMASK")
            mk = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(H, W) > 0
            xs = [c for p in polys for c in p[0::2]]
            ys = [c for p in polys for c in p[1::2]]
            out.append({"score": None, "box": [min(xs), min(ys), max(xs), max(ys)], "mask": mk, "label": prompt})
    else:
        out = [{"score": None, "box": b["box"], "mask": None, "label": b["label"]}
               for b in florence_boxes(text, W, H, grounding=task == "grounding")]
    return {"instances": out, "engine_s": engine_s, "answer": text}


def region_mask(svc, gen, picture: bytes, region: str, model: str | None = None, cancel=None) -> tuple[bytes, dict]:
    """The edits' `region`: OpenAI's mask (a PNG whose transparent pixels change) of every instance of `region` in
    `picture`, their union grown by REGION_GROW of the short side -> (the mask, segment's answer without the masks);
    ImageRequestError when nothing is found."""
    got = segment(svc, gen, picture, region, "mask", model, cancel=cancel)
    out = union_mask(got)
    if out is None:
        raise img.ImageRequestError(f"region: no {region!r} found in the image ({got['model']})")
    info = {k: v for k, v in got.items() if k != "instances"}
    info["instances"] = [{k: v for k, v in i.items() if k != "mask"} for i in got["instances"]]
    info["grow"] = out[1]
    return out[0], info


def union_mask(got: dict) -> tuple[bytes, int] | None:
    """segment's instances as one edit's mask: OpenAI's (transparent where it changes), their union grown by
    REGION_GROW of the short side -> (the mask, the pixels grown), None when none has a mask."""
    import numpy as np
    from PIL import Image
    found = [i["mask"] for i in got["instances"] if i.get("mask") is not None]
    if not found:
        return None
    union = np.logical_or.reduce(found).astype(np.uint8) * 255
    grow = max(REGION_GROW_MIN, round(REGION_GROW * min(got["width"], got["height"])))
    m = spread(union, grow)
    out = Image.new("RGBA", (got["width"], got["height"]), (0, 0, 0, 255))
    out.putalpha(Image.fromarray(255 - m, "L"))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue(), grow


def marked(picture: bytes, instances: list[dict]) -> bytes:
    """The picture with each instance shown (find_region): its mask tinted red, or its box drawn where it has none ->
    a PNG."""
    import numpy as np
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(picture)).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    for i in instances:
        if i.get("mask") is not None:
            m = i["mask"]
            a[m] = a[m] * 0.45 + np.array([255.0, 0.0, 0.0]) * 0.55
    out = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i in instances:
        if i.get("mask") is None:
            d.rectangle([round(v) for v in i["box"]], outline=(255, 0, 0), width=max(2, min(out.size) // 200))
    buf = io.BytesIO()
    out.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------- the background
def birefnet_size(version: str, W: int, H: int, resolution: str | None) -> tuple[int, int]:
    """The resolution BiRefNet runs at: `resolution` ("WxH", fal's operating_resolution) where given, else the
    version's (BIREFNET_SIZE), dynamic's the picture's own shrunk into DYNAMIC_SIDES; multiples of 32."""
    if resolution not in (None, ""):
        m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", str(resolution))
        if m is None:
            raise img.ImageRequestError(f"operating_resolution: WIDTHxHEIGHT, e.g. 1024x1024, not {resolution!r}")
        sw, sh = int(m.group(1)), int(m.group(2))
        if sw % 32 or sh % 32 or not (DYNAMIC_SIDES[0] <= min(sw, sh) and max(sw, sh) <= DYNAMIC_SIDES[1]):
            raise img.ImageRequestError(f"operating_resolution: multiples of 32 from {DYNAMIC_SIDES[0]} to "
                                        f"{DYNAMIC_SIDES[1]}, not {resolution!r}")
        return sw, sh
    fixed = BIREFNET_SIZE[version]
    if fixed is not None:
        return fixed
    lo, hi = DYNAMIC_SIDES
    scale = min(1.0, hi / max(W, H))
    if min(W, H) * scale < lo:   # a long, thin picture: its short side at the least, the long one cut to the most
        scale = lo / max(1, min(W, H))

    def side(v: float) -> int:
        return int(min(hi, max(lo, round(v * scale / 32) * 32)))
    return side(W), side(H)


def birefnet_version(model: str | None) -> str:
    if model in (None, ""):
        return "general"
    v = BIREFNET_FAL.get(str(model)) or str(model).lower().removeprefix("birefnet-").removeprefix("birefnet_")
    v = {"birefnet": "general", "2k": "lite-2k", "lite_2k": "lite-2k"}.get(v, v)
    if v not in BIREFNET_VERSIONS:
        raise img.ImageRequestError(f"model: a BiRefNet version ({', '.join(BIREFNET_VERSIONS)}) or fal's name "
                                    f"({', '.join(BIREFNET_FAL)}), not {model!r}")
    return v


@img.on_free_gpu
def matte(svc, gen, picture: bytes, version: str = "general", resolution: str | None = None, refine: bool = True,
          cancel=None) -> dict:
    """BiRefNet on `picture` -> {"version", "size": [w, h] it ran at, "alpha": uint8 [H][W], "foreground": uint8
    [H][W][3] (refine) or None, "seconds", "engine": the engine's times, "fallback"}.  A version that has no room
    on the GPU gives way to a smaller (BIREFNET_SMALLER) where that one's weights are here."""
    import numpy as np
    W, H = img.image_dims(picture)
    path = weights(gen, [f"birefnet-{version}"])[0]
    sw, sh = birefnet_size(version, W, H, resolution)
    job = aux_job(gen, "birefnet", path, picture, (sw, sh, int(refine)))
    t0 = time.monotonic()
    fallback = None
    with img.gpu_turn(svc, gen, job, cancel):
        gen._live(stage="matte")
        with _work(gen) as work:
            pic, out = work / "picture", work / "alpha"
            pic.write_bytes(picture)
            v = version
            while True:
                try:
                    got = _ask(gen, f"MATTE {path} {pic} {sw} {sh} {out}" + (" REFINE" if refine else ""), "MATTE")
                    break
                except MemoryShort as e:
                    smaller = BIREFNET_SMALLER.get(v)
                    if smaller is None or not ready(gen, [f"birefnet-{smaller}"]):
                        if smaller is not None:
                            with contextlib.suppress(img.AuxPending):
                                weights(gen, [f"birefnet-{smaller}"])
                        raise img.ImageJobFailed(f"BiRefNet {v} has no room on the GPU ({e})"
                                                 + (f"; {smaller} is downloading for the next request"
                                                    if smaller else "")) from e
                    gen._say(f"image tools: BiRefNet {v} has no room on the GPU ({e}); {smaller}")
                    fallback = f"{v} did not fit the GPU"
                    v, path = smaller, weights(gen, [f"birefnet-{smaller}"])[0]
                    sw, sh = birefnet_size(v, W, H, None)
            alpha = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(H, W).copy()
            fg = (np.frombuffer(Path(f"{out}.fg").read_bytes(), dtype=np.uint8).reshape(H, W, 3).copy()
                  if refine else None)
    gen.last = {"seconds": round(time.monotonic() - t0, 2)}
    gen.end(None)
    return {"version": v, "size": [sw, sh], "alpha": alpha, "foreground": fg, "fallback": fallback,
            "seconds": round(time.monotonic() - t0, 3),
            "engine": {k: float(got[k]) for k in ("matte_s", "backbone_s", "decoder_s", "refine_s") if k in got}}


def parse_color(value) -> tuple[int, int, int, int] | None:
    """background_color: "#rrggbb", "#rrggbbaa", "rrggbb", a name Pillow knows (white), or "transparent"/None."""
    if value in (None, "", "transparent", "none"):
        return None
    from PIL import ImageColor
    s = str(value).strip()
    if re.fullmatch(r"[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", s):
        s = "#" + s
    try:
        c = ImageColor.getcolor(s, "RGBA")
    except ValueError:
        raise img.ImageRequestError(f"background_color: #rrggbb, a color name or transparent, not {value!r}") from None
    return cast_rgba(c)


def cast_rgba(c) -> tuple[int, int, int, int]:
    return (int(c[0]), int(c[1]), int(c[2]), int(c[3]) if len(c) > 3 else 255)


def spread(a, r: int):
    """a uint8 [H][W] with each value the largest within r pixels (a square; below 0 the smallest within -r): shifts
    along the rows, then the columns, in time linear in r (Pillow's MaxFilter takes its square)."""
    import numpy as np
    if r == 0:
        return a
    pick, n = (np.maximum, r) if r > 0 else (np.minimum, -r)
    out = a.copy()
    for axis in (1, 0):
        src = out.copy()
        for s in range(1, n + 1):
            if s >= src.shape[axis]:
                break
            lead = [slice(None)] * 2
            tail = [slice(None)] * 2
            lead[axis], tail[axis] = slice(s, None), slice(None, -s)
            out[tuple(lead)] = pick(out[tuple(lead)], src[tuple(tail)])
            out[tuple(tail)] = pick(out[tuple(tail)], src[tuple(lead)])
    return out


def adjust_alpha(alpha, grow: int = 0, blur: float = 0.0):
    """The alpha grown (pixels; below 0 shrunk) and blurred (a Gaussian's radius, pixels): a Pillow L image."""
    from PIL import Image, ImageFilter
    a = Image.fromarray(spread(alpha, int(grow)) if grow else alpha, "L")
    if blur > 0:
        a = a.filter(ImageFilter.GaussianBlur(blur))
    return a


def cut_out(picture: bytes, got: dict, background=None, grow: int = 0, blur: float = 0.0, mask_only: bool = False):
    """matte's answer as a Pillow image: the mask (L) with mask_only, else the picture's (or the refined foreground's)
    colors under the alpha (RGBA), laid on `background` (an RGBA color: RGB out where it is opaque)."""
    from PIL import Image
    a = adjust_alpha(got["alpha"], grow, blur)
    if mask_only:
        return a
    if got.get("foreground") is not None:
        rgb = Image.fromarray(got["foreground"], "RGB")
    else:
        rgb = Image.open(io.BytesIO(picture)).convert("RGB")
    cut = rgb.convert("RGBA")
    cut.putalpha(a)
    if background is None:
        return cut
    bg = Image.new("RGBA", cut.size, background)
    out = Image.alpha_composite(bg, cut)
    return out.convert("RGB") if background[3] == 255 else out


def mask_png(mask) -> str:
    """A bool mask as vLLM-Omni sends one: a base64 single-channel PNG, 255 inside."""
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_rle(mask) -> dict:
    """COCO's uncompressed RLE of a bool mask: column-major run lengths, starting with the background's (vLLM-Omni's
    mask_format rle)."""
    import numpy as np
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], edges, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}


# ------------------------------------------------------------------------------------------- the entry points
def _flag(fields: dict, name: str, default: bool) -> bool:
    return img.parse_flag(fields.get(name), name, default)


def _number(fields: dict, name: str, default: float, lo: float, hi: float) -> float:
    return img.parse_number(fields.get(name), name, lo, hi, default)


def segmentation_response(svc, gen, fields: dict, picture: bytes, cancel=None) -> dict:
    """POST /v1/images/segmentations: vLLM-Omni's form fields (prompt, score_threshold, mask_threshold, mask_format
    png or rle, model) and XeStrata's task (mask, box, grounding) -> its answer (created, model, width, height,
    mask_format, data [{score, box, mask}]; a label where Florence-2 named one; the task and "xestrata" figures)."""
    if fields.get("boxes") not in (None, "", "[]"):
        raise img.ImageRequestError("boxes: exemplar boxes are not taken here (SAM 3's text prompt alone)")
    fmt = str(fields.get("mask_format") or "png").lower()
    if fmt not in ("png", "rle"):
        raise img.ImageRequestError(f"mask_format: png or rle, not {fmt!r}")
    got = segment(svc, gen, picture, str(fields.get("prompt") or ""), str(fields.get("task") or "mask").lower(),
                  fields.get("model"), _number(fields, "score_threshold", 0.5, 0.0, 1.0),
                  _number(fields, "mask_threshold", 0.5, 0.0, 1.0), cancel)
    data = []
    for i in got["instances"]:
        one: dict[str, Any] = {"score": i["score"], "box": [round(v, 2) for v in i["box"]],
                               "mask": None if i["mask"] is None else
                               mask_png(i["mask"]) if fmt == "png" else mask_rle(i["mask"])}
        if i.get("label") and got["model"] != SAM3:
            one["label"] = i["label"]
        data.append(one)
    return {"created": int(time.time()), "model": got["model"], "width": got["width"], "height": got["height"],
            "mask_format": fmt, "task": got["task"], "data": data,
            "xestrata": {**({"gpu": got["gpu"]} if got.get("gpu") else {}),
                         "seconds": got["seconds"], "engine_s": got["engine_s"],
                         **({"fallback": got["fallback"]} if got.get("fallback") else {})}}


def _compression(fields: dict):
    # This endpoint previously ignored the field, including invalid values.
    try:
        return img.parse_number(fields.get("output_compression"), "output_compression", 0, 100, whole=True)
    except img.ImageRequestError:
        return None


def remove_background_request(fields: dict) -> dict:
    """POST /v1/images/remove-background's fields -> matte's and cut_out's arguments: fal's BiRefNet model (a version
    or fal's name; default general), operating_resolution, refine_foreground (default true), mask_only, output_mask,
    output_format (png, webp, gif, jpeg); XeStrata's background_color, mask_grow (pixels, below 0 shrinks) and
    mask_blur (pixels); OpenAI's response_format (b64_json, url)."""
    fmt = str(fields.get("output_format") or "png").lower().replace("jpg", "jpeg")
    if fmt not in OUTPUT_FORMATS:
        raise img.ImageRequestError(f"output_format: one of {', '.join(OUTPUT_FORMATS)}, not {fmt!r}")
    rf = str(fields.get("response_format") or "b64_json")
    if rf not in ("b64_json", "url"):
        raise img.ImageRequestError(f"response_format: b64_json or url, not {rf!r}")
    mask_only = _flag(fields, "mask_only", False)
    return {"version": birefnet_version(fields.get("model")), "resolution": fields.get("operating_resolution"),
            "refine": _flag(fields, "refine_foreground", True) and not mask_only, "mask_only": mask_only,
            "output_mask": _flag(fields, "output_mask", False), "format": fmt, "response_format": rf,
            "compression": _compression(fields),
            "background": parse_color(fields.get("background_color")),
            "grow": int(_number(fields, "mask_grow", 0, -256, 256)), "blur": _number(fields, "mask_blur", 0.0, 0, 256)}


def remove_background_response(svc, gen, req: dict, picture: bytes, base_url: str, cancel=None) -> dict:
    """The cut-out (and with output_mask its mask) as vLLM-Omni's images answer: created, data [{b64_json or url}],
    the version, size and format, and "xestrata" figures."""
    got = matte(svc, gen, picture, req["version"], req["resolution"], req["refine"], cancel)
    im = cut_out(picture, got, req["background"], req["grow"], req["blur"], req["mask_only"])
    files = [img.encode_pillow(im, req["format"], req.get("compression"))]
    if req["output_mask"] and not req["mask_only"]:
        files.append(img.encode_pillow(adjust_alpha(got["alpha"], req["grow"], req["blur"]),
                                       req["format"], req.get("compression")))
    data = []
    for i, b in enumerate(files):
        entry = ({"url": f"{base_url}/v1/images/files/{gen.store(b, req['format'])}"} if req["response_format"] == "url"
                 else {"b64_json": base64.b64encode(b).decode()})
        data.append({**entry, **({"role": "mask"} if i == 1 else {})})
    return {"created": int(time.time()), "model": f"birefnet-{got['version']}", "width": im.width, "height": im.height,
            "output_format": req["format"], "data": data,
            "xestrata": {**({"gpu": got["gpu"]} if got.get("gpu") else {}),
                         "seconds": got["seconds"], "operating_resolution": "{}x{}".format(*got["size"]),
                         **got["engine"], **({"fallback": got["fallback"]} if got.get("fallback") else {})}}


def stability_remove_background(svc, gen, fields: dict, picture: bytes, cancel=None) -> tuple[bytes, str]:
    """Stability's /v2beta/stable-image/edit/remove-background: its output_format (png, webp) -> (the image, its
    format): BiRefNet general, the foreground refined, on transparency."""
    fmt = str(fields.get("output_format") or "png").lower()
    if fmt not in ("png", "webp"):
        raise img.ImageRequestError(f"output_format: png or webp, not {fmt!r}")
    compression = _compression(fields)
    got = matte(svc, gen, picture, "general", None, True, cancel)
    return img.encode_pillow(cut_out(picture, got), fmt, compression), fmt


def models() -> list[dict]:
    """/v1/models' entries of the segmenters and BiRefNet's versions."""
    def resolution(v: str) -> str:
        s = BIREFNET_SIZE[v]
        return f"{s[0]}x{s[1]}" if s is not None else "the picture's own"
    return ([{"id": m, "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["mask"]},
              "segmentation": {"tasks": list(SEG_TASKS) if m != SAM3 else ["mask", "box"]}} for m in SEG_MODELS]
            + [{"id": f"birefnet-{v}", "architecture": {"input_modalities": ["image"], "output_modalities": ["image"]},
                "remove_background": {"operating_resolution": resolution(v)}} for v in BIREFNET_VERSIONS])
