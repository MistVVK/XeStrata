# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/images.py - image generation (Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0, MOSS-Image)
through XeStrata's own image engine, strata-image.

POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url) plus Strata's
own field seed; POST /v1/images/edits the same with reference images (OpenAI's multipart form, image or image[]; or JSON
with images: [{"image_url": ...}]), which the image model edits or takes as its subject (Qwen-Image-2.1's edits: the
vision encoder reads them into the prompt, the VAE's encoder into the DiT's).  Every model takes a mask there (OpenAI's:
a PNG of the first image's size whose transparent pixels are where it changes; the engine keeps the rest, its latent
and its pixels) and a strength (diffusers' img2img strength, above 0 and at most 1: how much of the schedule runs from
the first image's latent, strength_run; 1, the default of an edit, starts from noise).  A model without edits
(MageTrail, Mage-Flow without its edit model, one without its vision file) takes one image as img2img's: its
strength, else img2img_strength (with a mask, 1).  src/image/img_start.hpp.  The engine (src/image/strata_image.cpp,
`strata-image --serve`) starts with the first image and stays, outside the server (serve/image_daemon.py, the next
server of the same image model and GPU finds it: a model switch does not lose it): its kernels built once, the
prompt's text states and the layers that fit in VRAM kept between images, one image a request.  It runs Turbo's
8 steps (CFG 1, the Euler flow sampler, the model's own sigmas), the image the one stable-diffusion.cpp makes with the
model card's command for the same seed (within LPIPS 0.03); the base model ("schedule": "base") the steps asked for
(default the config's "steps", 40) on its scheduler's sigmas (base_sigmas), CFG 1 as well.  A size whose sides are
not multiples of 16 (1920x1080) is made at the sides rounded up (1920x1088) and its middle cut out.

Mage-Flow ("family": "mage-flow", "schedule": "mage"): its own prompt form (MF_PROMPT_HEAD, the 34 ids of its system
turn dropped), the steps asked for (default the config's: Turbo 4, the RL model 20, the base model 30) on its static
shift (mage_sigmas), and guidance by guidance_scale or true_cfg_scale (the one scale it has; default the config's
"cfg_scale": Turbo 1, none; the others 5.0) with negative_prompt (default " ", its pipeline's).  Its edits take up to
MF_MAX_REFS reference images, in its edit form (MF_EDIT_HEAD, "Image j: <pad>" before the instruction, the 64 ids of
the system turn dropped) and the config's "edit_steps" (its edit models': Turbo 4, the others 30), on its edit DiT
("edit_model"); transparent images it has none of.

HunyuanImage 3.0 ("family": "hunyuan-image-3", "schedule": "hunyuanimage3"): its DiT reads the prompt's ids itself
(it has no text encoder), so the sequence is built here, by hunyuanimage3_ids: the model's system prompt (the
en_unified text setup wrote out, the model's own default), "User: " and the prompt, "Assistant: ", then the markers
its tokenizer carries - <answer>, <boi>, <img_size_1024>, <img_ratio_i>, <timestep>, <guidance>, <timestep_r> - the
block's <img> tokens (one a 16 x 16 pixel square) and <eoi>.  Its sizes snap to the model's resolution buckets
(hunyuanimage3_bucket: the config's "resolutions" parameters, the 37 shapes its image processor walks), the bucket's
shape and ratio index going into the sequence and its tokens being what the response's usage counts; the response's
size is the bucket's, and "strata" names the size asked for (requested_size) when it differs.  Its guidance is one
token, not two passes: guidance_scale (true_cfg_scale as its other name), default the config's "guidance" 2.5, and a
negative prompt has nothing to steer (model_defaults says so with guidance_token, so the web page and the estimate
treat a guided step as one).  Its steps are the config's (the Distil model's 8) on the model's own schedule - the
engine builds the sigmas (shift 3.0) from the count, so none are sent.  Its edits take up to the config's
"max_refs" (3) reference images, each a block of its own in the sequence before the prompt (hunyuanimage3_ids: the
image's VAE latent at the bucket nearest its shape and its 1024 SigLIP2 slots, which the engine fills), and need the
model's vision file ("vision": the bf16 shard with vision_model.*, which setup downloads); transparent images it has
none of.  Its weights are under the Tencent Hunyuan Community
License Agreement, which does not apply in the European Union, the United Kingdom or South Korea:
https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE

MOSS-Image ("family": "moss-image", "schedule": "moss"): its prompt in its pipeline's chat form (MOSS_PROMPT_HEAD,
all of it the DiT's, at most MOSS_MAX_IDS ids), the steps asked for (default 50) on its static shift (shift, default
5; the engine builds the sigmas), guidance at any scale but 1 (default 3) with negative_prompt (default empty), the
blocks its negative pass skips (uncond_skip_layers, default [22]) and skip-layer guidance (slg_layers, slg_scale,
slg_start, slg_stop), by its pipeline's names (moss_request).  A layout of boxes (layout: its prompt enhancer's draft,
boxes in the image's pixels) becomes its JSON prompt (moss_layout_prompt).  It has no edits: an image is img2img's.
Its text encoder is Qwen3.5-9B, read by the chat's engine when that reads the same file (text_states).
enhance has the chat's model write the prompt out first (enhance_prompt).

Adapters (serve/loras.py): a request's "lora" field - a list of {"name", "scale"}, one such object, a name alone, or
the same as a form's JSON text - and the prompt's tags, <lora:name:0.8> in A1111's and Forge's form, positive prompt
and negative; they go to the engine as IMG's and NEED's LORA:<strength>:<path> words (the path's whitespace %XX'd:
the engine splits the line on it).  A name the model's LoRA folder ("lora_dir") does not offer is refused where the
field gives it and taken out of the prompt with a warning where a tag does (the response's strata.lora_unknown).
"local_path" is a file inside that folder only.  The response's data[] items carry the adapters applied as "lora":
[{"name", "scale"}], and strata their count and VRAM (lora_n, lora_mib, the IMAGE line's).

The GPU is shared with the text engine one job at a time: an image job takes the service's turn (the FIFO every chat
request takes), so no chat turn runs on the GPU meanwhile.  A config whose "images" section names a card of its own
("gpu_pci", setup --image-gpu) that no text engine of the server takes runs the jobs there (separate): they take only
that GPU's turn (GpuTurn), so chat requests run beside them, and the text engine is left as it is.  The image engine
works in the VRAM it is given: what it holds and what is free, less a reserve; layers that do not fit are streamed from
RAM, so it runs, more slowly, in a few GB.  Before each job it says what the image needs (NEED: the least, and the DiT
resident with the VAE in one piece), and the config's "gpu_policy" says what happens to the text engine during the job
(gpu_turn):
  keep    - it stays loaded and idle (an APU's shared memory, a card with room for both)
  shrink  - an engine started with --vram-elastic gives the image job VRAM from its expert cache and takes it back
            afterwards (POST /v1/vram's mechanism; the conversation stays in the engine)
  unload  - its conversation is saved to a session file, the engine is stopped, and the next start restores it
  auto    - (default) keep when the image fits beside it at full speed, else shrink when the engine can, else keep
            when the image fits at all (its layers streamed), else unload
  separate - not a value of "gpu_policy": the policy of every job on a card of its own (above)
After a job the image engine keeps what it holds (the next image starts warm) only under keep or separate with VRAM to
spare;
otherwise it lets go of everything but its process (about 200 MiB).  The daemon ends after daemon_idle_s (default
1800) without a server.

The config's "images" section (written by setup --images, tools/image/qimage.py):
  {"exe": "engine/strata-image", "diffusion_model": "...Q8_0.gguf", "schedule": "turbo"|"base", "steps": 8|40,
   "vae": "...", "llm": "...Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "quant": "Q8_0", "gpu_policy": "auto",
   "reload": "lazy"|"eager", "max_n": 16, "budget_s": 600, "max_side": 2048, "timeout_s": 3600, "uma": false,
   "out_dir": null, "keep_files": 64,
   "vision": "...mmproj-F16.gguf", "max_refs": 10, "reserve_mib": 512, "spare_mib": 1024, "ssd_offload": false,
   "text_encoder": "stock", "text_quant": "Q4_K",
   "quant_lowered": {"from": "Q8_0", "to": "Q6_K", "text_from": "Q8_0", "text_to": "Q4_K", "need_gb": 20.2,
                     "ram_left_gb": 14.0}}
HunyuanImage 3.0's instead: {"model_name": "hunyuan-image-3-distil", "family": "hunyuan-image-3", "exe",
"quant": "w4a8"|"int8", "diffusion_model", "vae", "tokenizer": ".../tokenizer.json" (the model's own, read as a
byte-level BPE: tools/strata_tokenizer.py's Tokenizer.from_hf), "system_prompt": ".../en_unified.txt",
"vision": ".../model-0032-of-0032.safetensors" (SigLIP2, for the edits), "max_refs": 3, "steps": 8, "guidance": 2.5,
"schedule": "hunyuanimage3", "resolutions": {"base_size": 1024, "step": 64, "align": 16, "extra_resolutions":
[...]}, "transparency": false} - no "llm" (no text encoder).
  lora_dir: the folder of the LoRA adapters the model may apply (setup writes <data>/models/loras; serve/loras.py
  lists what its DiT takes and checks a request's names); without one, "loras" beside the DiT's file
  max_n: the most images one request makes; budget_s: the longest one request's images may take, by the speed this
  server measured (ImageGen.estimate_s; until it has made an image, at most UNMEASURED_N); schedule: Turbo's own
  sigmas or the base model's (base_sigmas; no "schedule" is Turbo), Mage-Flow's static shift (mage_sigmas),
  HunyuanImage 3.0's the engine's own (from the step count: none is sent); steps: the model's default (a request's
  "steps" overrides it; Turbo's 8 are its own); vision: Qwen3-VL's vision encoder (mmproj), for the edits (without it
  they are refused); max_refs: the reference images an edit takes at most; reserve_mib: VRAM the
  image engine leaves free when it sizes itself; spare_mib: the free VRAM that must remain after a job under keep for
  the engine to keep its layers; quant_lowered (only when setup lowered them): the sizes setup took instead of the ones
  asked for, because the RAM the text model leaves was short (note(): the server's log and the web page say so).
"""
from __future__ import annotations

import base64
import collections
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from serve import loras as lora_api

TURBO_STEPS = 8
MAX_STEPS = 100                    # the base model's
CFG_MAX = 30.0                     # guidance_scale / true_cfg_scale (vLLM-Omni: "typically 0.0-20.0")
MODEL_NAME = "qwen-image-2.1-turbo"
POLICIES = ("auto", "keep", "shrink", "unload")
SIZE_STEP = 16                     # the VAE's 8x and the DiT's 2x2 patches: sides made at a multiple of 16, then cut
MIN_SIDE = 256
MAX_N = 16                         # "max_n": the most images one request makes
BUDGET_S = 600                     # "budget_s": the longest one request's images may take (ImageGen.check_n)
UNMEASURED_N = 4                   # the most images a request makes before the server has measured its speed
# img2img's strength when a request gives none (a model without edits, an image and no mask), at most: the denoise of
# ComfyUI's own img2img example (comfyanonymous.github.io/ComfyUI_examples/img2img), lowered for a model of few steps
# (img2img_strength).  An edit's, and a mask's alone, is 1.0, the default denoise of ComfyUI's KSampler (nodes.py)
STRENGTH_IMG2IMG = 0.87
# a step's time against the image's pixels, above the largest size measured (estimate_s): the attention's share grows
# with them (the B70, Qwen-Image Turbo: 6.5 times the time at 4 times the pixels, 1.35); taken higher, to err long
SIZE_EXP = 1.5
FILE_NAME = re.compile(r"^[0-9a-f]{32}\.(png|jpeg|webp|jxl)$")
# OpenAI's output_format, and JPEG XL beside them -> the file's MIME type (Pillow writes jpeg and webp, libjxl's cjxl
# jxl; PNG is png_rgba's)
FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "jxl": "image/jxl"}
# Qwen's recommended prompt for a transparent image (its README's "Transparent Image Generation (RGBA)")
TRANSPARENT_PROMPT = ("This is an RGBA image with transparency. {prompt}. The image has alpha channel and the "
                      "background is transparent.")
# OpenAI's quality -> the base model's steps (None: the config's); Turbo keeps its 8 whatever it says
QUALITY_STEPS = {"low": 20, "medium": 30, "standard": 30, "high": None, "hd": None, "xhigh": None, "max": None,
                 "auto": None}
# the model's prompt (stable-diffusion.cpp's Qwen-Image template): the system turn, whose DROP leading ids' states the
# DiT does not take, then the user's prompt as plain text
PROMPT_HEAD = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
DROP = 14
# Mage-Flow's prompt (its pipeline's "mage-flow" template): the system turn's 34 ids are not the DiT's
MF_PROMPT_HEAD = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
                  "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
MF_DROP = 34
# its edits' (the "mage-flow-edit" template): the system turn's 64 ids dropped, each reference "Image j: <pad>"
MF_EDIT_HEAD = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
                "objects, background), then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency with the "
                "original input where appropriate.<|im_end|>\n<|im_start|>user\n")
MF_EDIT_DROP = 64
MF_EDIT_REF = "Image {i}: <|vision_start|><|image_pad|><|vision_end|>"
MF_MAX_REFS = 3                    # the edit models' training takes up to three references
MF_SHIFT = 6.0                     # Mage-Flow's static time shift
MF_NEG = " "                       # its pipeline's negative prompt when none is given
HUNYUANIMAGE3_STEPS = 8            # HunyuanImage 3.0 Instruct-Distil's own (its generation_config's diff_infer_steps)
HUNYUANIMAGE3_GUIDANCE = 2.5       # its guidance token's default (diff_guidance_scale)
HUNYUANIMAGE3_MAX_REFS = 3         # its edits' reference images at most (each about 5,000 tokens at 1024x1024)
HUNYUANIMAGE3_VIT_PATCH = 16       # SigLIP2's patch, and the slots a reference's features take (its processor's)
HUNYUANIMAGE3_VIT_SLOTS = 1024
# MOSS-Image's prompt (its pipeline's encode_prompt: the chat template with thinking on, the user's turn alone), all of
# it the DiT's (no system turn to drop), at most its max_sequence_length
MOSS_PROMPT_HEAD = "<|im_start|>user\n"
MOSS_PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
MOSS_MAX_IDS = 4096
MOSS_STEPS = 50                    # its pipeline's defaults: 50 steps, guidance 3, shift 5, an empty negative prompt,
MOSS_CFG = 3.0                     # block 22 skipped by the negative pass, no skip-layer guidance (slg_scale 0)
MOSS_SHIFT = 5.0
MOSS_NEG = ""
MOSS_UNCOND_SKIP = [22]
MOSS_BLOCKS = 42                   # its DiT's blocks (the skip lists' range)
MOSS_SHIFT_MAX = 20.0
# a layout (the prompt as MOSS-Image's JSON of boxes): its summary's fields and an item's
MOSS_SUMMARY_KEYS = ("caption", "meta", "canvas", "palette")
MOSS_ITEM_KEYS = ("label", "caption", "text", "bbox", "tags", "palette")
MOSS_LABELS = ("object", "character", "text")
MOSS_CANVASES = ("single", "multi_panel", "collage")
# an edit's reference image in the prompt: one pad, which the engine widens to the image's tokens
REF_PROMPT = "{sep}<image{i}><|vision_start|><|image_pad|><|vision_end|>"
REF_MAX_BYTES = 32 << 20
MAX_REFS = 10                      # Qwen-Image-2.1's own limit (its model card)
RAM_SPARE_MIB = 2048               # RAM an image job leaves free beside its prefix cache (the OS, the server)
START_TIMEOUT_S = 600              # the engine's start: the parts opened and a small image made (kernels built)


class ImageRequestError(ValueError):
    """A request the image endpoint refuses (400)."""


class ImageJobFailed(RuntimeError):
    """The image engine did not make the images (500); the message says why."""


class LoraRejected(ImageJobFailed):
    """The image engine refused an adapter the request names - its file, its forms, its rows against the DiT's (the
    engine's `ERROR lora:` line): the request's own fault (400); the message says why."""


def parse_size(value, max_side: int) -> tuple[int, int]:
    """OpenAI's size, "1024x1024" (or "auto": 1024x1024) -> (width, height); each side from 256 to max_side."""
    if value is None or value == "auto":
        return 1024, 1024
    if not isinstance(value, str):
        raise ImageRequestError("size: a string WIDTHxHEIGHT, e.g. \"1024x1024\"")
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*", value)
    if not m:
        raise ImageRequestError(f"size: WIDTHxHEIGHT, e.g. \"1024x1024\", not {value!r}")
    w, h = int(m.group(1)), int(m.group(2))
    for side in (w, h):
        if side < MIN_SIDE or side > max_side:
            raise ImageRequestError(f"size {w}x{h}: each side from {MIN_SIDE} to {max_side}")
    return w, h


def _int_field(req: dict, key: str, lo: int, hi: int, default):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ImageRequestError(f"{key}: a whole number from {lo} to {hi}")
    return v


def model_defaults(cfg: dict) -> dict:
    """What a request leaves out gets from the image model of the config's "images" section `cfg` (parse_request and
    cfg_request take theirs from here; the web page shows them): steps (Turbo's fixed, steps_fixed; Mage-Flow's edit
    model's for an edit), the least steps,
    the guidance scale (1: none), the negative prompt when none is given (None: none, so no guidance without one),
    whether it makes transparent images, that every model takes a mask, and whether an image it is given is
    img2img's (a model without edits) with the strength it gets when a request gives none (strength: 1 for an
    edit).  HunyuanImage 3.0's guidance is one token
    rather than a second pass (guidance_token: no negative prompt, a guided step costs one), and its sizes are its
    resolution buckets (buckets: [width, height] each; a request's size is made at the one whose shape is nearest).
    MOSS-Image's are its pipeline's: the shift, the blocks the negative pass skips, skip-layer guidance's fields, and
    that it takes a layout of boxes and a prompt enhanced by the chat's model."""
    mage = cfg.get("family") == "mage-flow"
    hunyuan = cfg.get("family") == "hunyuan-image-3"
    moss = cfg.get("family") == "moss-image"
    if cfg.get("schedule") == "base":
        steps, least, fixed = int(cfg.get("steps") or 40), 2, False
    elif cfg.get("schedule") == "mage":
        steps, least, fixed = int(cfg.get("steps") or 4), 1, False
    elif cfg.get("schedule") == "hunyuanimage3":   # HunyuanImage 3.0: any count on the engine's own shift
        steps, least, fixed = int(cfg.get("steps") or HUNYUANIMAGE3_STEPS), 1, False
    elif cfg.get("schedule") == "moss":   # MOSS-Image: any count on its static shift
        steps, least, fixed = int(cfg.get("steps") or MOSS_STEPS), 1, False
    else:
        steps, least, fixed = TURBO_STEPS, TURBO_STEPS, True
    edit = int(cfg.get("edit_steps") or steps) if cfg.get("edit_model") else steps
    # "strength": an img2img's at most, lowered by the steps as img2img_strength (the web page's imgStrength)
    out = {"steps": steps, "edit_steps": edit, "steps_min": least, "steps_max": MAX_STEPS, "steps_fixed": fixed,
           "max_n": int(cfg.get("max_n") or MAX_N), "budget_s": float(cfg.get("budget_s") or BUDGET_S),
           "formats": output_formats(),
           "guidance": (float(cfg.get("guidance") or HUNYUANIMAGE3_GUIDANCE) if hunyuan
                        else float(cfg.get("cfg_scale") or 1.0) if mage
                        else float(cfg.get("cfg_scale") or MOSS_CFG) if moss else 1.0),
           "guidance_token": hunyuan,
           "negative_prompt": MF_NEG if mage else MOSS_NEG if moss else None,
           "transparency": cfg.get("transparency") is not False,
           "mask": True, "img2img": not can_edit(cfg), "strength": 1.0 if can_edit(cfg) else STRENGTH_IMG2IMG}
    if hunyuan:
        out["buckets"] = [[b["width"], b["height"]] for b in hunyuanimage3_buckets(cfg)]
    if moss:
        out.update(shift=float(cfg.get("shift") or MOSS_SHIFT), uncond_skip_layers=list(MOSS_UNCOND_SKIP),
                   slg_layers=[], slg_scale=0.0, slg_start=0.0, slg_stop=1.0, blocks=MOSS_BLOCKS, layout=True,
                   enhance=True)
    return out


_CJXL: list[str | None] = []


def cjxl() -> str | None:
    """libjxl's encoder, cjxl: the system's (the distributions' libjxl-tools / libjxl-utils), else the one the Windows
    7z carries beside the engine; None: no JPEG XL."""
    if not _CJXL:
        beside = Path(__file__).resolve().parents[1] / "engine" / ("cjxl.exe" if sys.platform == "win32" else "cjxl")
        _CJXL.append(shutil.which("cjxl") or (str(beside) if beside.is_file() else None))
    return _CJXL[0]


def output_formats() -> list[str]:
    """The output_formats this server writes: png, jpeg and webp, and jxl where there is a cjxl."""
    return ["png", "jpeg", "webp"] + (["jxl"] if cjxl() else [])


def parse_request(req: dict, cfg: dict) -> dict:
    """The fields of one /v1/images/generations request, checked.  output_format png, jpeg, webp or jxl (where
    output_formats has it; output_compression 0-100, the quality jpeg, webp and jxl are written at, jxl's 100
    lossless); quality low, medium or high (the base model's
    steps, QUALITY_STEPS; an explicit steps wins); background transparent (the prompt in Qwen's form for it, an image
    with alpha; not with jpeg), opaque (alpha flattened on white) or auto.  vLLM-Omni's fields too: num_inference_steps
    (steps by its name), negative_prompt, guidance_scale and true_cfg_scale (cfg_request).  Unknown fields (style,
    user, moderation) are accepted and ignored.  HunyuanImage 3.0's size snaps to the model's resolution bucket nearest
    the shape asked for (the job's "width"/"height" are that bucket's, "bucket" its row of the table, and
    "requested_size" the size asked for when it is another); its size "auto" (or none) is the model's own choice where
    its config has the ratio head (job["choose_size"]: ImageGen.run asks the engine first, the 1024 x 1024 bucket
    standing for it until then - no bucket has more tokens).  The adapters it applies come from "lora" and from the
    two prompts' <lora:name:0.8> tags (serve/loras.py): job["loras"], and for a tag's name the model does not offer
    job["lora_unknown"] (the response's strata.lora_unknown).  A text encoder's adapter is not applied (the human's
    decision of 2026-10-29): "scale_text" and a tag's third field are no fields of the image engine's, and are taken
    as its other unknown fields are, style and user, ignored."""
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageRequestError("prompt: a non-empty string")
    if len(prompt) > 16000:
        raise ImageRequestError("prompt: at most 16000 characters")
    if "\0" in prompt:
        raise ImageRequestError("prompt: no NUL characters")
    # the adapters: the "lora" field, then the tags of both prompts (A1111's and Forge's form); a tag's name the model
    # does not offer is taken out and said in the response's strata (lora_unknown), a field's name is refused
    folder = lora_api.folder(cfg)
    try:
        loras = lora_api.field(req.get("lora"), folder)
        prompt, tagged, unknown = lora_api.strip_tags(prompt, folder)
        negative = req.get("negative_prompt")
        if isinstance(negative, str):
            negative, neg_tagged, neg_unknown = lora_api.strip_tags(negative, folder)
            req = {**req, "negative_prompt": negative}
            tagged += neg_tagged
            unknown += neg_unknown
        for use in tagged:
            if all(u["name"] != use["name"] for u in loras):
                loras.append(use)
        lora_api.limit(loras)
    except lora_api.LoraError as e:
        raise ImageRequestError(str(e)) from None
    if loras and not prompt.strip():
        raise ImageRequestError('prompt: a non-empty string (its LoRA tags are taken out: the "lora" field names them)')
    fmt = req.get("response_format") or "b64_json"
    if fmt not in ("b64_json", "url"):
        raise ImageRequestError("response_format: b64_json or url")
    out_fmt = req.get("output_format") or "png"
    if out_fmt not in output_formats():
        raise ImageRequestError("output_format: " + ", ".join(output_formats()))
    compression = _int_field(req, "output_compression", 0, 100, 100)
    quality = req.get("quality")
    if quality is not None and quality not in QUALITY_STEPS:
        raise ImageRequestError("quality: " + ", ".join(QUALITY_STEPS))
    background = req.get("background") or "auto"
    if background not in ("auto", "opaque", "transparent"):
        raise ImageRequestError("background: auto, opaque or transparent")
    if background == "transparent" and out_fmt == "jpeg":
        raise ImageRequestError("background transparent: png, webp or jxl (jpeg has no alpha channel)")
    if background == "transparent" and cfg.get("transparency") is False:
        raise ImageRequestError("background transparent: this image model does not make transparent images")
    w, h = parse_size(req.get("size"), int(cfg.get("max_side") or 2048))
    bucket = None
    requested: dict = {}
    if cfg.get("family") == "hunyuan-image-3":    # its own table: the bucket nearest the shape asked for
        bucket = hunyuanimage3_bucket(cfg, w, h)
        if req.get("size") in (None, "auto") and cfg.get("ratio_head"):
            requested = {"choose_size": True}
        elif (w, h) != (int(bucket["width"]), int(bucket["height"])):
            requested = {"requested_size": f"{w}x{h}"}
        w, h = int(bucket["width"]), int(bucket["height"])
    n = _int_field(req, "n", 1, int(cfg.get("max_n") or MAX_N), 1)
    if "num_inference_steps" in req:            # vLLM-Omni's name for steps
        if req.get("steps") not in (None, req["num_inference_steps"]):
            raise ImageRequestError("steps and num_inference_steps: give one, or the same number")
        req = {**req, "steps": req["num_inference_steps"]}
    dflt = model_defaults(cfg)
    if cfg.get("schedule") == "base":
        steps = _int_field(req, "steps", 2, MAX_STEPS, QUALITY_STEPS.get(quality or "auto") or dflt["steps"])
    elif cfg.get("schedule") == "mage":   # Mage-Flow: any count on its static shift, the config's by default
        steps = _int_field(req, "steps", 1, MAX_STEPS, dflt["steps"])
    elif cfg.get("schedule") in ("hunyuanimage3", "moss"):   # the count on the engine's own sigmas (its shift)
        steps = _int_field(req, "steps", 1, MAX_STEPS, dflt["steps"])
    else:   # Turbo's own sigmas: steps asked for are ignored, as its diffusers pipeline (sample_sigmas) does
        _int_field(req, "steps", 1, MAX_STEPS, None)
        steps = dflt["steps"]
    cfg_fields = cfg_request(req, cfg)
    seed = _int_field(req, "seed", 0, 2**62, None)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    # OpenAI's streamed partial images: made only for a stream (the server's), none by default
    partial = _int_field(req, "partial_images", 0, 3, 0)
    if req.get("stream") is not None and not isinstance(req.get("stream"), bool):
        raise ImageRequestError("stream: true or false")
    job = {"prompt": prompt, "width": w, "height": h, "n": n, "steps": steps, "seed": seed, "partial_images": partial,
           "quality": quality or "auto", "response_format": fmt, "output_format": out_fmt,
           "output_compression": compression, "background": background, "bucket": bucket, **requested, **cfg_fields}
    if loras:
        job["loras"] = loras
    if unknown:
        job["lora_unknown"] = list(dict.fromkeys(unknown))   # a name in both prompts said once
    if cfg.get("family") == "moss-image":
        job.update(moss_request(req, cfg))
        if req.get("layout") is not None:
            job["layout"] = parse_layout(req["layout"])
            job["layout_prompt"] = moss_layout_prompt(job["layout"], prompt, w, h)
    else:
        for key in ("layout", "enhance"):
            if req.get(key) not in (None, False, "", "false", "off"):
                raise ImageRequestError(f"{key}: MOSS-Image's (this image model has none)")
    return job


def _number(req: dict, key: str, lo: float, hi: float, default: float) -> float:
    """A number field (or a form field's text) from lo to hi, checked; `default` when none."""
    v = req.get(key)
    if v is None or v == "":
        return default
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            raise ImageRequestError(f"{key}: a number from {lo:g} to {hi:g}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ImageRequestError(f"{key}: a number from {lo:g} to {hi:g}")
    return float(v)


def _blocks(req: dict, key: str, default: list[int]) -> list[int]:
    """A list of MOSS-Image's DiT blocks (zero-based, below MOSS_BLOCKS), as its pipeline's skip lists take them: a
    JSON list, or a form field's text ("22", "22,30", "[22]"); null, [] or "none": none; absent: `default`."""
    if key not in req:
        return list(default)
    v = req[key]
    if isinstance(v, str):
        text = v.strip()
        if text.lower() in ("", "none", "null", "[]"):
            return []
        try:
            v = json.loads(text) if text.startswith("[") else [int(x) for x in text.split(",") if x.strip()]
        except ValueError:
            raise ImageRequestError(f"{key}: a list of block numbers from 0 to {MOSS_BLOCKS - 1}") from None
    if v is None:
        return []
    if isinstance(v, int) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, list) or any(isinstance(b, bool) or not isinstance(b, int) or not 0 <= b < MOSS_BLOCKS
                                      for b in v):
        raise ImageRequestError(f"{key}: a list of block numbers from 0 to {MOSS_BLOCKS - 1}")
    return sorted(set(v))


def moss_request(req: dict, cfg: dict) -> dict:
    """MOSS-Image's own fields, by its pipeline's names and defaults (model_defaults): shift (the schedule's static
    shift), uncond_skip_layers (the blocks guidance's negative pass skips; null or [] none), skip-layer guidance's
    slg_layers and slg_scale (a third pass of the prompt with those blocks skipped, slg_scale times the difference
    added; 0 or no layers, none) and both skips' window slg_start .. slg_stop (shares of the steps), and enhance (the
    prompt written out by the chat's model first: true, "caption" or "layout"; enhance_prompt)."""
    dflt = model_defaults(cfg)
    out: dict[str, Any] = {"shift": _number(req, "shift", 0.01, MOSS_SHIFT_MAX, dflt["shift"]),
                           "uncond_skip": _blocks(req, "uncond_skip_layers", dflt["uncond_skip_layers"]),
                           "slg_layers": _blocks(req, "slg_layers", []),
                           "slg_scale": _number(req, "slg_scale", 0.0, CFG_MAX, 0.0),
                           "slg_start": _number(req, "slg_start", 0.0, 1.0, 0.0),
                           "slg_stop": _number(req, "slg_stop", 0.0, 1.0, 1.0)}
    if out["slg_start"] > out["slg_stop"]:
        raise ImageRequestError("slg_start: at most slg_stop")
    enhance = req.get("enhance")
    if isinstance(enhance, str):
        word = enhance.strip().lower()
        enhance = {"true": True, "on": True, "false": False, "off": False, "": False}.get(word, word)
    if enhance not in (None, False, True, "auto", "caption", "layout"):
        raise ImageRequestError('enhance: true (the model chooses), "caption", "layout" or false')
    if enhance not in (None, False):
        if req.get("layout") is not None:
            raise ImageRequestError("enhance: not with a layout (the layout is the prompt already)")
        out["enhance"] = "auto" if enhance is True else enhance
    return out


def parse_layout(value) -> dict:
    """A request's layout (MOSS-Image's boxes), checked: a JSON object (or its text, a form's field) with "items" and
    optionally "summary", as the model's prompt enhancer drafts one: boxes [x_min, y_min, x_max, y_max] in the image's
    pixels from its top-left corner ("bbox_coord_space" "abs" and "bbox_format" "xyxy", which may be given and are the
    only ones taken); its "width" and "height", when given, must be the image's (moss_layout_prompt checks them)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ImageRequestError("layout: a JSON object") from None
    if not isinstance(value, dict):
        raise ImageRequestError('layout: an object with "items" (and "summary")')
    unknown = set(value) - {"summary", "items", "width", "height", "bbox_coord_space", "bbox_format"}
    if unknown:
        raise ImageRequestError(f"layout: unknown fields {sorted(unknown)}")
    if value.get("bbox_coord_space", "abs") != "abs" or value.get("bbox_format", "xyxy") != "xyxy":
        raise ImageRequestError('layout: boxes in the image\'s pixels, [x_min, y_min, x_max, y_max] from its top-left '
                                '(bbox_coord_space "abs", bbox_format "xyxy")')

    def text(v, where: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ImageRequestError(f"layout: {where}: a non-empty string")
        return v

    def strings(v, where: str) -> list:
        if not isinstance(v, list):
            raise ImageRequestError(f"layout: {where}: a list of strings")
        return [text(x, f"{where}[{i}]") for i, x in enumerate(v)]

    summary = value.get("summary")
    if summary is not None:
        if not isinstance(summary, dict):
            raise ImageRequestError("layout: summary: an object")
        unknown = set(summary) - set(MOSS_SUMMARY_KEYS)
        if unknown:
            raise ImageRequestError(f"layout: summary: unknown fields {sorted(unknown)}")
        if "caption" in summary:
            text(summary["caption"], "summary.caption")
        if "canvas" in summary and summary["canvas"] not in MOSS_CANVASES:
            raise ImageRequestError("layout: summary.canvas: " + ", ".join(MOSS_CANVASES))
        for key in ("meta", "palette"):
            if key in summary:
                strings(summary[key], f"summary.{key}")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise ImageRequestError("layout: items: a non-empty list")
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            raise ImageRequestError(f"layout: {where}: an object")
        unknown = set(item) - set(MOSS_ITEM_KEYS)
        if unknown:
            raise ImageRequestError(f"layout: {where}: unknown fields {sorted(unknown)}")
        if item.get("label") not in MOSS_LABELS:
            raise ImageRequestError(f"layout: {where}.label: " + ", ".join(MOSS_LABELS))
        text(item.get("caption"), f"{where}.caption")
        if item["label"] == "text" and "text" not in item:
            raise ImageRequestError(f'layout: {where}: a text item needs its "text" (the exact words)')
        if "text" in item:
            text(item["text"], f"{where}.text")
        for key in ("tags", "palette"):
            if key in item:
                strings(item[key], f"{where}.{key}")
        box = item.get("bbox")
        if (not isinstance(box, list) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            raise ImageRequestError(f"layout: {where}.bbox: [x_min, y_min, x_max, y_max], four numbers")
    return value


def tidy_layout(draft, width: int, height: int):
    """A layout a chat model drafted (enhance_prompt's, generate_image's), into parse_layout's form where it can be
    read so: the summary as text taken for its caption; an item's label from what it has ("text" with words, else
    "object") and its caption from its description, its words or its label; its box kept on the width x height
    image; the canvas' own fields and other fields left out.  Anything else (not an object) is left for parse_layout
    to refuse."""
    if not isinstance(draft, dict):
        return draft
    out: dict = {}
    summary = draft.get("summary")
    if isinstance(summary, str) and summary.strip():
        out["summary"] = {"caption": summary}
    elif isinstance(summary, dict):
        out["summary"] = {k: v for k, v in summary.items() if k in MOSS_SUMMARY_KEYS}
    items = []
    listed = draft.get("items")
    for item in listed if isinstance(listed, list) else ():
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and
                                                                not isinstance(v, bool) for v in box)):
            continue
        x1, y1, x2, y2 = (min(max(float(v), 0.0), float(side)) for v, side in zip(box, (width, height) * 2))
        if not (x1 < x2 and y1 < y2):
            continue
        words = item.get("text") if isinstance(item.get("text"), str) and item["text"].strip() else None
        label = item.get("label") if item.get("label") in MOSS_LABELS else "text" if words else "object"
        caption = next((v for k in ("caption", "description", "object", "character", "content")
                        if isinstance(v := item.get(k), str) and v.strip()), None) or words or label
        tidy = {"label": label, "caption": caption}
        if words:
            tidy["text"] = words
        elif label == "text":
            tidy["label"] = "object"
        tidy["bbox"] = [x1, y1, x2, y2]
        for key in ("tags", "palette"):
            if isinstance(item.get(key), list):
                tidy[key] = [v for v in item[key] if isinstance(v, str) and v.strip()]
        items.append(tidy)
    out["items"] = items
    return out


def moss_layout_prompt(layout: dict, prompt: str, width: int, height: int) -> str:
    """A layout (parse_layout's) as MOSS-Image reads it, its prompt: the JSON of its model card (summary, items, then
    "bbox_coord_space": "abs_centralized" and "bbox_format": "yxyx", as json.dumps writes it with ensure_ascii off),
    each box [y_min, x_min, y_max, x_max] in pixels from the middle of the image the engine makes (made_size; the
    image cut from it keeps the middle), its edges taken outward to whole pixels, as the model's prompt enhancer
    compiles a draft.  The summary's caption is the request's prompt where the layout gives none."""
    for key, side in (("width", width), ("height", height)):
        if key in layout and layout[key] != side:
            raise ImageRequestError(f"layout: {key} {layout[key]} is not the image's {side} (size)")
    mw, mh = made_size({"width": width, "height": height})
    dx, dy = (mw - width) // 2, (mh - height) // 2
    summary = dict(layout.get("summary") or {})
    summary.setdefault("caption", prompt)
    items = []
    for i, item in enumerate(layout["items"]):
        x1, y1, x2, y2 = item["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ImageRequestError(f"layout: items[{i}].bbox {item['bbox']}: x from 0 to {width} and y from 0 to "
                                    f"{height}, each min below its max")
        box = [math.floor(round(y1, 9)) + dy - mh // 2, math.floor(round(x1, 9)) + dx - mw // 2,
               math.ceil(round(y2, 9)) + dy - mh // 2, math.ceil(round(x2, 9)) + dx - mw // 2]
        items.append({k: (box if k == "bbox" else v) for k, v in item.items()})
    return json.dumps({"summary": summary, "items": items, "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False)


def cfg_request(req: dict, cfg: dict | None = None) -> dict:
    """vLLM-Omni's guidance fields, checked: negative_prompt, guidance_scale and true_cfg_scale.  Qwen-Image-2.1 takes
    classifier-free guidance from true_cfg_scale above 1 with a negative_prompt, as diffusers' QwenImage21Pipeline
    does (without one, or at 1, none: "cfg" stays out of the job); guidance_scale it has no use for, and ignores.
    Mage-Flow (cfg "family" mage-flow) has one scale: true_cfg_scale, else guidance_scale, else the config's
    "cfg_scale"; above 1 it guides, with negative_prompt or its pipeline's " ".  MOSS-Image's one scale is taken the
    same way (its config's 3.0), and guides at any scale but 1 (its pipeline's u + scale (c - u)), with negative_prompt
    or its pipeline's empty one.  HunyuanImage 3.0 has one scale as
    well, but as a token rather than a second pass: guidance_scale (true_cfg_scale as its other name), else the
    config's "guidance"; a negative prompt has nothing to steer there, so none is kept."""
    out: dict = {}
    neg = req.get("negative_prompt")
    if neg is not None:
        if not isinstance(neg, str) or len(neg) > 16000 or "\0" in neg:
            raise ImageRequestError("negative_prompt: a string of at most 16000 characters")
        out["negative_prompt"] = neg
    for key in ("guidance_scale", "true_cfg_scale"):
        v = req.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= CFG_MAX:
            raise ImageRequestError(f"{key}: a number from 0 to {CFG_MAX:g}")
        out[key] = float(v)
    if (cfg or {}).get("family") == "hunyuan-image-3":
        out.pop("negative_prompt", None)
        scale = out.get("guidance_scale")
        if scale is None:
            scale = out.get("true_cfg_scale")
        if scale is None:
            scale = float((cfg or {}).get("guidance") or HUNYUANIMAGE3_GUIDANCE)
        out["guidance"] = float(scale)
        return out
    if (cfg or {}).get("family") in ("mage-flow", "moss-image"):
        dflt = model_defaults(cfg or {})
        scale = out.get("true_cfg_scale", out.get("guidance_scale", dflt["guidance"]))
        if scale > 1 or (scale != 1 and (cfg or {}).get("family") == "moss-image"):
            out["cfg"] = scale
            out.setdefault("negative_prompt", dflt["negative_prompt"])
        return out
    if out.get("true_cfg_scale", 1.0) > 1 and out.get("negative_prompt") is not None:
        out["cfg"] = out["true_cfg_scale"]
    return out


def model_prompt(job: dict) -> str:
    """The prompt as the model reads it: a transparent image's in Qwen's form for one (TRANSPARENT_PROMPT); MOSS-Image's
    layout as its JSON (moss_layout_prompt), or the prompt its enhancer wrote (enhance_prompt)."""
    if job.get("layout_prompt"):
        return job["layout_prompt"]
    if job.get("enhanced_prompt"):
        return job["enhanced_prompt"]
    if job.get("background") == "transparent":
        return TRANSPARENT_PROMPT.format(prompt=job["prompt"].strip().rstrip("."))
    return job["prompt"]


def encode_image(png: bytes, fmt: str, compression: int, opaque: bool = False) -> bytes:
    """A made image (png_rgba's PNG) in OpenAI's output_format: PNG as it is, jpeg (no alpha: on white) and webp at
    `compression` as their quality (Pillow), jxl at it as its quality, 100 lossless (cjxl); `opaque` (background
    opaque): its alpha flattened on white."""
    if fmt == "png" and not opaque:
        return png
    if fmt == "jxl" and not opaque:
        return encode_jxl(png, compression)
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if fmt == "jpeg" or opaque:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    out = io.BytesIO()
    if fmt == "jxl":
        im.save(out, format="PNG")
        return encode_jxl(out.getvalue(), compression)
    im.save(out, format=fmt.upper(), **({} if fmt == "png" else {"quality": compression}))
    return out.getvalue()


def encode_jxl(png: bytes, quality: int) -> bytes:
    """A PNG as JPEG XL at `quality` (100: lossless, its alpha kept) by cjxl.  Lossless at effort 3: on an i7-14700 a
    1024x1024 image in 0.04 s and 5% larger than at cjxl's default 7, which takes 0.87 s; the others at 7 (0.12 s)."""
    exe = cjxl()
    if exe is None:
        raise ImageJobFailed("output_format jxl: this computer has no cjxl (libjxl's tools)")
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    r = subprocess.run([exe, "-", "-", "-q", str(quality), "-e", "3" if quality == 100 else "7", "--quiet"],
                       input=png, capture_output=True, timeout=300, creationflags=no_window)
    if r.returncode != 0 or not r.stdout:
        raise ImageJobFailed(f"cjxl failed: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def base_sigmas(steps: int, width: int, height: int) -> list[float]:
    """Qwen-Image-2.1's sigmas for `steps` steps of a width x height image (the size the engine makes), and the
    terminal 0: diffusers' QwenImage21Pipeline with the FlowMatchEulerDiscreteScheduler its scheduler_config sets up
    (linspace(1, 1/steps), the exponential time shift by mu, a line in the image's tokens from base_shift 0.5 at 256 to
    max_shift 0.9 at 8192, then stretched to end at shift_terminal 0.02).  An image token is a 16 x 16 pixel square."""
    seq = (width // SIZE_STEP) * (height // SIZE_STEP)
    mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (seq - 256)
    t = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
    shifted = [math.exp(mu) / (math.exp(mu) + (1.0 / x - 1.0)) for x in t]
    scale = (1.0 - shifted[-1]) / (1.0 - 0.02)
    return [1.0 - (1.0 - x) / scale for x in shifted] + [0.0]


def image_dims(data: bytes) -> tuple[int, int]:
    """(width, height) of a PNG, JPEG, GIF or BMP from its header; ImageRequestError for another format."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return w, abs(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
                i += 1 if marker == 0xFF else 2
                continue
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # a frame header: its size
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    raise ImageRequestError("a reference image: PNG, JPEG, GIF or BMP")


def _qimage():
    """tools/image/qimage.py, the one table of setup's --images choices (setup.py takes it the same way)."""
    tools = str(Path(__file__).resolve().parent.parent / "tools" / "image")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import qimage
    return qimage


def setup_choice(cfg: dict) -> str:
    """The setup --images value that installs the image model of the config's "images" section `cfg` (its
    "model_name" is the model's name, which for Qwen-Image-2.1 is not the choice's)."""
    models = _qimage().IMAGE_MODELS
    name = cfg.get("model_name") or MODEL_NAME
    return next((k for k, v in models.items() if v[0] == name), _qimage().DEFAULT_MODEL)


def not_set_up_message() -> str:
    """What a request for an image is told when the server has no image generation (the 404)."""
    choices = [k for k in _qimage().IMAGE_MODELS if not k.endswith("-uncensored")]
    return ("image generation is not set up on this server: run setup with --images and an image model ("
            + ", ".join(choices) + "; -uncensored after one for its uncensored text encoder), see docs/IMAGE.md")


def parse_strength(req: dict) -> float | None:
    """The request's strength (a number, or a form field's text), checked: above 0 and at most 1; None when none."""
    v = req.get("strength")
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            raise ImageRequestError("strength: a number above 0 and at most 1") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
        raise ImageRequestError("strength: a number above 0 and at most 1")
    return float(v)


def strength_run(steps: int, strength: float) -> int:
    """The steps of a `steps`-step schedule a strength runs: diffusers' get_timesteps, which skips the first
    floor(steps - steps strength) (the engine's strength_skip, src/image/img_start.cpp, counts the same in doubles)."""
    return steps - int(max(steps - min(steps * strength, steps), 0))


def img2img_strength(steps: int) -> float:
    """img2img's default strength for a `steps`-step schedule: STRENGTH_IMG2IMG, or less where that skips none of the
    steps (a few-step model: 4 steps at 0.87 run all 4, the image lost), so that it skips one at least."""
    return STRENGTH_IMG2IMG if steps < 2 else min(STRENGTH_IMG2IMG, 1 - 1 / steps)


def parse_edit_request(req: dict, images: list[bytes], cfg: dict, mask: bytes | None = None) -> dict:
    """An edit (/v1/images/edits): parse_request's fields with `images` (PNG, JPEG, GIF or BMP).  A model that edits
    takes them as its reference images (job["refs"]); one that does not (can_edit) takes one image as img2img's base
    image (job["init"]).  size "auto" (or none): the first image's shape at about 1024 x 1024 pixels, its sides
    multiples of SIZE_STEP.  `mask` (OpenAI's): a PNG of the first image's size whose transparent pixels are where it
    changes, the rest kept as it is, as job["mask"].  strength (parse_strength): how much of the schedule runs from
    the first image (job["strength"] and the steps it runs, job["steps_run"], when below 1); an edit's default 1, an
    img2img's img2img_strength of its steps, or 1 with a mask."""
    mage = cfg.get("family") == "mage-flow"
    hunyuan = cfg.get("family") == "hunyuan-image-3"
    edits = can_edit(cfg)
    strength = parse_strength(req)
    if not images:
        raise ImageRequestError("image: one reference image or more" if edits else "image: the image to start from")
    if edits:
        max_refs = int(cfg.get("max_refs") or (MF_MAX_REFS if mage else HUNYUANIMAGE3_MAX_REFS if hunyuan
                                               else MAX_REFS))
        if len(images) > max_refs:
            raise ImageRequestError(f"image: at most {max_refs} reference images")
    elif len(images) > 1:
        raise ImageRequestError("image: one image (this image model has no edits: it starts from one image, "
                                "img2img)")
    elif strength is None and mask is not None:
        strength = 1.0
    dims = []
    for data in images:
        if len(data) > REF_MAX_BYTES:
            raise ImageRequestError(f"image: at most {REF_MAX_BYTES >> 20} MiB each")
        w, h = image_dims(data)
        if w < 16 or h < 16:
            raise ImageRequestError(f"image: {w}x{h} is too small")
        dims.append((w, h))
    if mask is not None:
        mask = check_mask(mask, dims[0])
    if req.get("size") in (None, "auto") and not (hunyuan and cfg.get("ratio_head")):   # else the model chooses
        w0, h0 = dims[0]
        side = int(cfg.get("max_side") or 2048)
        tw = (1024 * 1024 * w0 / h0) ** 0.5
        th = tw * h0 / w0
        tw, th = (min(side, max(MIN_SIDE, round(v / SIZE_STEP) * SIZE_STEP)) for v in (tw, th))
        req = {**req, "size": f"{tw}x{th}"}
    # Mage-Flow's edit model has its own steps
    job = parse_request(req, {**cfg, "steps": cfg.get("edit_steps") or cfg.get("steps")} if mage and edits else cfg)
    if edits:
        job["refs"] = list(images)
    else:
        job["init"] = images[0]
        if strength is None:
            strength = img2img_strength(job["steps"])
    if mask is not None:
        job["mask"] = mask
    if strength is not None and strength < 1:
        run = strength_run(job["steps"], strength)
        if run < 1:
            raise ImageRequestError(f"strength: {strength:g} runs none of the {job['steps']} steps; raise it (or the "
                                    "steps)")
        job["strength"], job["steps_run"] = strength, run
    return job


def check_mask(data: bytes, size: tuple[int, int]) -> bytes:
    """An edit's mask, checked as OpenAI checks it (a PNG with an alpha channel, the first image's size) -> RGBA PNG
    bytes the engine reads (every model takes one: the configs' "mask": false of before are not read)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageRequestError("mask: a PNG with an alpha channel")
    import io
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ImageRequestError(f"mask: the PNG could not be read ({e})") from None
    if "A" not in im.getbands() and "transparency" not in im.info:
        raise ImageRequestError("mask: a PNG with an alpha channel (its transparent pixels are where the image "
                                "changes)")
    if im.size != size:
        raise ImageRequestError(f"mask: {im.size[0]}x{im.size[1]}, not the image's {size[0]}x{size[1]}")
    out = io.BytesIO()
    im.convert("RGBA").save(out, format="PNG")
    return out.getvalue()


def ref_size(ref_w: int, ref_h: int) -> tuple[int, int]:
    """The size the engine works an edit's reference image at (qi_ref_size in src/image/qi_pipeline.cpp): diffusers'
    calculate_dimensions(1024 * 1024, its width over its height)."""
    ratio = ref_w / ref_h
    vw = math.sqrt(1024 * 1024 * ratio)
    return max(32, round(vw / 32) * 32), max(32, round(vw / ratio / 32) * 32)


def usage(job: dict) -> dict:
    """OpenAI's usage of an image job, in the image engine's tokens: the prompt's ids (the model's chat form) and the
    reference images' DiT tokens in, the made images' DiT tokens out (a token a 16 x 16 pixel square).  HunyuanImage
    3.0's ids carry the image block and the reference images' blocks themselves, so their shares are taken out of the
    text (a reference's: its VAE grid and its 1024 SigLIP2 slots)."""
    mw, mh = made_size(job)
    text = len(job.get("ids") or ())
    block = (mw // SIZE_STEP) * (mh // SIZE_STEP)
    if job.get("bucket") is not None:
        refs = sum(b["tokens_h"] * b["tokens_w"] + HUNYUANIMAGE3_VIT_SLOTS for b in job.get("ref_buckets") or ())
        text -= block + refs
    else:
        sizes = (ref_size(*image_dims(r)) for r in job.get("refs") or ())
        refs = sum((w // SIZE_STEP) * (h // SIZE_STEP) for w, h in sizes)
        if job.get("init"):   # img2img's image, read at the made size
            refs += block
    out = job["n"] * block
    return {"input_tokens": text + refs, "output_tokens": out, "total_tokens": text + refs + out,
            "input_tokens_details": {"text_tokens": text, "image_tokens": refs}}


def made_size(job: dict) -> tuple[int, int]:
    """The size the engine makes for `job`: each side rounded up to a multiple of SIZE_STEP."""
    def up(v) -> int:
        return -(-int(v) // SIZE_STEP) * SIZE_STEP
    return up(job["width"]), up(job["height"])


def png_rgba(rgba: bytes, width: int, height: int, made_w: int, made_h: int) -> bytes:
    """A PNG (8-bit RGBA, as stable-diffusion.cpp writes) of the middle width x height of an RGBA8 image of
    made_w x made_h.  Pillow's adaptive row filters at zlib's level 1: 58 ms and 1.5 MB for a 1024 x 1024 image,
    where filter 0 at level 6 took 100 ms and 2.0 MB (a tenth of a Turbo image's time on a B70)."""
    import io
    from PIL import Image
    x0, y0 = (made_w - width) // 2, (made_h - height) // 2
    im = Image.frombuffer("RGBA", (made_w, made_h), rgba, "raw", "RGBA", 0, 1)
    if (width, height) != (made_w, made_h):
        im = im.crop((x0, y0, x0 + width, y0 + height))
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def prompt_ids(tok, prompt: str, refs: int = 0, family: str = "qwen-image") -> list[int]:
    """The prompt in the model's chat form, tokenized (Qwen3-VL's tokenizer): the user's text plain, so a special
    token spelled in it stays text; an edit's `refs` reference images before it, one pad each (stable-diffusion.cpp's
    prompt for Qwen-Image-2.1's edits, the pads widened by the engine).  Mage-Flow's in its own forms (MF_PROMPT_HEAD;
    an edit's MF_EDIT_HEAD, each reference "Image j: <pad>" before the instruction); MOSS-Image's in its pipeline's
    (the user's turn and the assistant's opened with <think>, MOSS_PROMPT_HEAD and MOSS_PROMPT_TAIL; Qwen3.5-9B's
    tokenizer)."""
    if family == "moss-image":
        text = MOSS_PROMPT_HEAD + prompt + MOSS_PROMPT_TAIL
        at = len(MOSS_PROMPT_HEAD)
        return tok.encode(text, parse_special=True, plain=[(at, at + len(prompt))])
    if family == "mage-flow":
        head = MF_EDIT_HEAD + "".join(MF_EDIT_REF.format(i=i + 1) for i in range(refs)) if refs else MF_PROMPT_HEAD
    else:
        head = PROMPT_HEAD + "".join(REF_PROMPT.format(sep=" " if i else "", i=i + 1) for i in range(refs))
    text = head + prompt + PROMPT_TAIL
    return tok.encode(text, parse_special=True, plain=[(len(head), len(head) + len(prompt))])


def hunyuanimage3_buckets(cfg: dict) -> list[dict]:
    """HunyuanImage 3.0's resolution buckets (ratio_index, height, width, tokens_h, tokens_w, ratio), from the config's
    "resolutions" parameters (tools/image/qimage.py's hunyuanimage3_buckets builds the table; the run config keeps the
    parameters, not the table), or the model's own without them."""
    params = cfg.get("resolutions")
    if not isinstance(params, dict) or not params.get("base_size"):
        params = _qimage().hunyuanimage3_resolutions()
    return _qimage().hunyuanimage3_buckets(params)


def hunyuanimage3_bucket(cfg: dict, width: int, height: int) -> dict:
    """HunyuanImage 3.0's resolution bucket nearest the shape asked for (|ratio - height / width|, the earlier of two
    equally near)."""
    return min(hunyuanimage3_buckets(cfg), key=lambda b: abs(b["ratio"] - height / width))


def _hunyuanimage3_marker(tok, name: str) -> int:
    i = tok.special_ids.get(name)
    if i is None:
        raise ValueError(f"the tokenizer has no {name}: not HunyuanImage 3.0's tokenizer.json")
    return i


def hunyuanimage3_prefix_ids(tok, prompt: str, base_size: int = 1024, system: str | None = None,
                             refs: list[dict] | None = None) -> list[int]:
    """hunyuanimage3_ids up to the generated block's <img_size_base_size>: the sequence the engine's RATIO reads the
    model's own size choice from (the reference's img_ratio task)."""
    def marker(name: str) -> int:
        return _hunyuanimage3_marker(tok, name)

    ids = [marker("<|startoftext|>")]
    if system is not None:
        ids += tok.encode(system, parse_special=False) + tok.encode("\n\n", parse_special=False)
    ids += tok.encode("User: ", parse_special=False)
    for b in refs or ():
        ids += [marker("<boi>"), marker(f"<img_size_{int(base_size)}>"), marker(f"<img_ratio_{int(b['ratio_index'])}>"),
                marker("<timestep>")]
        ids += [marker("<img>")] * (int(b["tokens_h"]) * int(b["tokens_w"]))
        ids += [marker("<joint_img_sep>")] + [marker("<img>")] * HUNYUANIMAGE3_VIT_SLOTS + [marker("<eoi>")]
    for text in (prompt, "\n\n", "Assistant: "):
        ids += tok.encode(text, parse_special=False)
    return ids + [marker("<answer>"), marker("<boi>"), marker(f"<img_size_{int(base_size)}>")]


def hunyuanimage3_ids(tok, prompt: str, bucket: dict, base_size: int = 1024, system: str | None = None,
                      refs: list[dict] | None = None) -> list[int]:
    """HunyuanImage 3.0's sequence for an image of `bucket` (hunyuanimage3_bucket), as its tokenizer's "instruct"
    template writes it: <|startoftext|>, the system prompt (`system`, the model's en_unified text) and a blank line,
    "User: ", an edit's reference images (`refs`: each one's bucket, hunyuanimage3_bucket of its shape - its block
    <boi>, <img_size_base_size>, <img_ratio_i>, <timestep>, its VAE grid's <img> tokens, <joint_img_sep>, its 1024
    SigLIP2 slots' <img> tokens, <eoi>; the reference's `cond_joint_image` before the prompt), the prompt, a blank
    line, "Assistant: ", <answer>, <boi>, <img_size_base_size>, <img_ratio_ratio_index>, the three a step writes over
    (<timestep>, <guidance>, <timestep_r>), the block's <img> tokens, <eoi>, </answer> and a blank line (which the
    engine's hunyuanimage3_trim drops with the rest after <eoi>).  `system` None leaves the system turn out, as the
    CPU golden's batch_system_prompt=[None] did.  Every section is encoded on its own - as one string the pieces
    merge across the boundaries, and the reference encodes each section too - and the prompt plain, so a special
    token spelled in it stays text."""
    def marker(name: str) -> int:
        return _hunyuanimage3_marker(tok, name)

    ids = hunyuanimage3_prefix_ids(tok, prompt, base_size, system, refs)
    ids += [marker(f"<img_ratio_{int(bucket['ratio_index'])}>"), marker("<timestep>"), marker("<guidance>"),
            marker("<timestep_r>")]
    ids += [marker("<img>")] * (int(bucket["tokens_h"]) * int(bucket["tokens_w"]))
    ids += [marker("<eoi>"), marker("</answer>")]
    return ids + tok.encode("\n\n", parse_special=False)


def mage_sigmas(steps: int, shift: float = MF_SHIFT) -> list[float]:
    """Mage-Flow's sigmas for `steps` and the terminal 0: shift t / (1 + (shift - 1) t), t from 1 down to 1 / steps
    (its FlowMatchEulerDiscreteScheduler with a static shift; the engine's own default for Turbo's 4)."""
    out = []
    for i in range(steps):
        t = 1.0 - (1.0 - 1.0 / steps) * i / (steps - 1) if steps > 1 else 1.0
        out.append(shift * t / (1.0 + (shift - 1.0) * t))
    return out + [0.0]


def load_tokenizer(path: str, family: str = "qwen-image"):
    """The prompt's tokenizer (tools/strata_tokenizer.py): HunyuanImage 3.0's own `tokenizer.json` (a byte-level BPE,
    from_hf) or Qwen3-VL's, from the text encoder's GGUF."""
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import strata_tokenizer
    if family == "hunyuan-image-3":
        return strata_tokenizer.Tokenizer.from_hf(path)
    return strata_tokenizer.Tokenizer.from_gguf(path)


def can_edit(cfg: dict) -> bool:
    """Whether the image model of the config section `cfg` edits (takes reference images): Qwen-Image-2.1's and
    Mage-Flow's with their vision encoder (and Mage-Flow's edit model), HunyuanImage 3.0's with its vision model's
    file; MageTrail does not."""
    return bool(cfg.get("vision")) and not (cfg.get("family") == "mage-flow" and not cfg.get("edit_model"))


def _kv(line: str) -> dict:
    out: dict = {}
    for part in line.split()[1:]:
        k, _, v = part.partition("=")
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


# the environment that makes one image engine another: the GPU it takes and the libraries it loads
ENGINE_ENV = ("STRATA_GPU_PCI", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "ONEAPI_DEVICE_SELECTOR",
              "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "STRATA_VRAM_LIMIT_MIB", "STRATA_MAX_ALLOC_MIB",
              "STRATA_NO_XMX", "LD_LIBRARY_PATH")


def daemon_address(cmd: list[str], env: dict) -> str:
    """Where the image daemon (serve/image_daemon.py) of this engine command and GPU listens: a Unix socket's path, on
    Windows the file with its loopback port and token; one per user."""
    import hashlib
    import json
    key = hashlib.sha256(json.dumps([cmd, [env.get(k) for k in ENGINE_ENV]]).encode()).hexdigest()[:16]
    user = str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))
    return str(Path(tempfile.gettempdir()) / f"xestrata-image-{user}-{key}.{'port' if os.name == 'nt' else 'sock'}")


def gpu_key(env: dict) -> str:
    """The GPU an engine environment names (its device selection; none: the system's first), for GpuTurn."""
    import hashlib
    sel = [env.get(k) for k in ("STRATA_GPU_PCI", "CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR",
                                "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")]
    return hashlib.sha256(json.dumps(sel).encode()).hexdigest()[:16]


class GpuTurn:
    """Every server's image jobs on one GPU, one at a time, and the image daemons started on it.  A file lock in the
    temp folder (the GPU's, gpu_key) is held for a job's whole turn, so a job of another server waits for it; a
    file beside it lists the daemons of the GPU, and the job that holds the turn ends every one but its own model's
    (another model's, or another server's: they are idle, having no turn), so two engines do not share the VRAM.
    Servers of the same model share its daemon, which is never ended for them."""

    def __init__(self, env: dict):
        user = str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))
        base = Path(tempfile.gettempdir()) / f"xestrata-image-gpu-{user}-{gpu_key(env)}"
        self.lock_path, self.list_path = Path(str(base) + ".lock"), Path(str(base) + ".daemons")
        self.f: Any = None

    def _try(self, f) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self, cancel: threading.Event | None = None, waiting=None) -> None:
        """Take the GPU's turn, waiting while another server's job holds it (`waiting()` once, when it does);
        ImageJobFailed when `cancel` is set meanwhile."""
        f = open(self.lock_path, "a+b")
        told = False
        while not self._try(f):
            if not told and waiting is not None:
                waiting()
                told = True
            if cancel is not None and cancel.is_set():
                f.close()
                raise ImageJobFailed("cancelled while another server's image job had the GPU")
            time.sleep(0.25)
        self.f = f

    def release(self) -> None:
        f, self.f = self.f, None
        if f is None:
            return
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()                                     # flock's lock goes with the file

    def daemons(self) -> list[str]:
        try:
            got = json.loads(self.list_path.read_text(encoding="utf-8"))
            return [a for a in got if isinstance(a, str)] if isinstance(got, list) else []
        except (OSError, ValueError):
            return []

    def keep_only(self, addr: str) -> list[str]:
        """End every listed daemon but `addr` (the caller holds the turn) and list `addr` -> those that were ended."""
        ended = [a for a in self.daemons() if a != addr and stop_daemon_at(a)]
        tmp = self.list_path.with_suffix(".tmp")
        with contextlib.suppress(OSError):
            tmp.write_text(json.dumps([addr]), encoding="utf-8")
            os.replace(tmp, self.list_path)
        return ended


def stop_daemon_at(addr: str) -> bool:
    """End the image daemon at `addr` and its engine (SHUTDOWN: the engine frees its memory itself) -> whether one
    was there.  Its READY is read first (a connection closed before it ends the daemon's handling unread) and its
    STOPPED, the engine gone, waited for (a daemon without it closes the connection instead)."""
    e = ImageEngine({}, {}, Path(os.devnull), lambda text: None)
    if not e._connect(addr):
        return False
    try:
        e._next(0, 30)
        e.send("SHUTDOWN")
        while e._next(0, 60) != "STOPPED":
            pass
    except ImageJobFailed:
        pass
    e.close()
    return True


class ImageEngine:
    """The image engine, `strata-image --serve`, through the image daemon (serve/image_daemon.py): started outside the
    server the first time and found again by the next server of the same image model and GPU, so a model switch does
    not lose it; one request line at a time (the caller serializes, the daemon between servers)."""

    def __init__(self, cfg: dict, env: dict, log_path: Path, say):
        self.cfg, self.env, self.log_path, self.say = cfg, env, log_path, say
        self.sock: Any = None
        self.f: Any = None
        self.lines: queue.Queue = queue.Queue()
        self.ended = threading.Event()              # the daemon closed this connection

    def command(self) -> list[str]:
        cmd = [str(self.cfg["exe"]), "--dit", str(self.cfg["diffusion_model"]), "--vae", str(self.cfg["vae"])]
        if self.cfg.get("llm"):       # the text encoder's GGUF; HunyuanImage 3.0 has none (its DiT reads the ids)
            cmd += ["--text", str(self.cfg["llm"])]
        if self.cfg.get("vision"):
            cmd += ["--vision", str(self.cfg["vision"])]
        if self.cfg.get("ratio_head"):   # HunyuanImage 3.0's own size choice
            cmd += ["--ratio-head", str(self.cfg["ratio_head"])]
        if self.cfg.get("dit_q8") is not None:   # Mage-Flow's DiT: 8 bits or FP16 as the config says, else by the VRAM
            cmd += ["--dit-q8", "1" if self.cfg["dit_q8"] else "0"]
        if self.cfg.get("edit_model"):
            cmd += ["--edit-dit", str(self.cfg["edit_model"])]
        if self.cfg.get("dit_keep") is False:
            cmd += ["--dit-keep", "0"]
        return cmd + ["--serve"]

    def alive(self) -> bool:
        """Connected, and the daemon has not closed the connection (another server's job may have ended it, GpuTurn):
        a closed one is let go, so the next job starts the engine again."""
        if self.f is not None and self.ended.is_set():
            self.close()
        return self.f is not None

    def _connect(self, addr: str) -> bool:
        import socket
        try:
            if os.name != "nt":
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(addr)
                token = None
            else:
                port, token = Path(addr).read_text(encoding="utf-8").split()
                s = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
                s.settimeout(None)
        except (OSError, ValueError):
            return False
        self.sock = s
        self.f = s.makefile("rw", encoding="utf-8", errors="replace", newline="\n")
        if token is not None:
            self.f.write(token + "\n")
            self.f.flush()
        self.lines = queue.Queue()
        self.ended = threading.Event()

        def pump(f, q, ended):
            try:
                for line in f:
                    q.put(line.rstrip("\r\n"))
            except (OSError, ValueError):
                pass
            ended.set()
            q.put(None)                               # the daemon closed the connection
        threading.Thread(target=pump, args=(self.f, self.lines, self.ended), daemon=True).start()
        return True

    def _spawn(self, addr: str):
        """Start the daemon (its environment the server's engine's), detached from this server."""
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "image_daemon.py"), "--socket", addr,
               "--idle", str(float(self.cfg.get("daemon_idle_s") or 1800)), "--log", str(self.log_path), "--",
               *self.command()]
        kw: dict = {"start_new_session": True} if os.name != "nt" else \
            {"creationflags": 0x00000200 | 0x00000008 | 0x01000000}   # its own group, no console, out of the job
        try:
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=self.env, **kw)
        except OSError:
            if os.name != "nt":
                raise
            kw["creationflags"] = 0x00000200 | 0x00000008                # a job that allows no breakaway
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=self.env, **kw)

    def start(self) -> dict:
        """Connect to the image daemon, starting it when there is none, and wait for READY -> its fields.
        ImageJobFailed when it does not start."""
        addr = daemon_address(self.command(), self.env)
        t0 = time.monotonic()
        proc = None
        lock = addr + ".lock"
        while not self._connect(addr):
            if proc is None:
                try:   # one server starts it; another waits for its socket
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)
                    proc = self._spawn(addr)
                except FileExistsError:
                    with contextlib.suppress(OSError):
                        if time.time() - os.path.getmtime(lock) > START_TIMEOUT_S:
                            os.unlink(lock)               # a start that never finished
                except OSError as e:
                    raise ImageJobFailed(f"the image engine did not start: {e}") from e
            elif proc.poll() is not None:
                with contextlib.suppress(OSError):
                    os.unlink(lock)
                raise ImageJobFailed(f"the image engine did not start; its log: {self.log_path}")
            if time.monotonic() - t0 > START_TIMEOUT_S:
                raise ImageJobFailed(f"the image engine did not start in {START_TIMEOUT_S} s; its log: {self.log_path}")
            time.sleep(0.25)
        if proc is not None:
            with contextlib.suppress(OSError):
                os.unlink(lock)
        line = self._next(0, START_TIMEOUT_S)
        if not line.startswith("READY"):
            self.close()
            raise ImageJobFailed(line[6:] if line.startswith("ERROR") else f"the image engine said: {line}")
        self.say(f"image engine {'started' if proc else 'found'} in {time.monotonic() - t0:.1f} s ({addr})")
        return _kv(line)

    def _next(self, waited: float, limit: float, cancel: threading.Event | None = None) -> str:
        """The engine's next line; ImageJobFailed when the connection ended or `limit` passed."""
        while True:
            try:
                line = self.lines.get(timeout=0.25)
            except queue.Empty:
                if cancel is not None and cancel.is_set():
                    self.send("STOP")
                    cancel = None                     # the engine answers ERROR stopped after its step
                    continue
                if waited > limit:
                    self.close()
                    raise ImageJobFailed(f"the image engine did not answer in {limit:g} s; its log: "
                                         f"{self.log_path}")
                waited += 0.25
                continue
            if line is None:
                self.close()
                raise ImageJobFailed(f"the image engine ended; its log: {self.log_path}")
            return line

    def send(self, line: str) -> None:
        f = self.f
        try:
            if f is None:
                raise OSError("not connected")
            f.write(line + "\n")
            f.flush()
        except (OSError, ValueError) as e:
            self.close()
            raise ImageJobFailed(f"the image engine is gone ({e}); its log: {self.log_path}") from e

    def ask(self, line: str, want: str, timeout: float, cancel: threading.Event | None = None, step=None,
            stage=None, preview=None) -> dict:
        """Send a request line and wait for its answer (the line starting with `want`) -> its fields; STEP lines go to
        `step`, STAGE lines (refs, text, sampling, decode) to `stage`, PREVIEW lines (a partial image: its index, the
        step, the file of its RGBA8 rows) to `preview`.  ImageJobFailed on ERROR."""
        self.send(line)
        t0 = time.monotonic()
        while True:
            got = self._next(time.monotonic() - t0, timeout, cancel)
            if got.startswith("STEP "):
                if step is not None:
                    done, steps = got.split()[1:3]
                    step(int(done), int(steps))
            elif got.startswith("STAGE "):
                if stage is not None:
                    stage(got[6:].strip())
            elif got.startswith("PREVIEW "):
                if preview is not None:
                    index, done, path = got.split(maxsplit=3)[1:4]
                    preview(int(index), int(done), path)
            elif got.startswith(want):
                return _kv(got)
            elif got.startswith("ERROR"):
                why = got[6:]
                if why.startswith("lora: "):
                    raise LoraRejected(why[6:])
                raise ImageJobFailed(why or "the image engine failed")

    def close(self, shutdown: bool = False) -> None:
        """Leave the daemon (it stays for the next server); `shutdown`: end it and its engine."""
        f, sock, self.f, self.sock = self.f, self.sock, None, None
        if f is not None:
            with contextlib.suppress(OSError, ValueError):
                f.write(("SHUTDOWN" if shutdown else "QUIT") + "\n")
                f.flush()
        if sock is not None:   # the reader blocked in the file first: closing the file under it would wait for it
            import socket
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        if f is not None:
            with contextlib.suppress(OSError, ValueError):
                f.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


class ImageGen:
    """The resident image engine, one request at a time (the service's turn, gpu_turn)."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        self.cfg = dict(cfg)
        policy = self.cfg.get("gpu_policy") or "auto"
        if policy not in POLICIES:
            raise ValueError(f'images "gpu_policy" must be one of {", ".join(POLICIES)}, not {policy!r}')
        reload = self.cfg.get("reload") or "lazy"
        if reload not in ("lazy", "eager"):
            raise ValueError(f'images "reload" must be lazy or eager, not {reload!r}')
        self.family = self.cfg.get("family") or "qwen-image"
        for k in self._file_keys():
            if not isinstance(self.cfg.get(k), str) or not self.cfg[k]:
                raise ValueError(f'the config\'s "images" section needs "{k}" (run setup with --images again)')
        if Path(self.cfg["exe"]).name.startswith("sd-cli"):
            raise ValueError('the config\'s "images" section is for stable-diffusion.cpp, which XeStrata no longer '
                             "uses (run setup with --images again)")
        self.policy, self.reload = policy, reload
        # the ids the DiT does not take: the leading system turn's (HunyuanImage 3.0's DiT reads all of its own)
        self.drop = (0 if self.family in ("hunyuan-image-3", "moss-image") else MF_DROP if self.family == "mage-flow"
                     else DROP)
        self.model = self.cfg.get("model_name") or MODEL_NAME
        self.log = log
        out = self.cfg.get("out_dir")
        self.dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="strata-images-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_files = int(self.cfg.get("keep_files") or 64)
        self.reserve = int(self.cfg.get("reserve_mib") or 512)
        self.spare = int(self.cfg.get("spare_mib") or 1024)
        self.lock = threading.Lock()              # one request line at a time even when the service's turn is not taken
        self.loras = lora_api.folder(self.cfg)    # the adapters its DiT takes (serve/loras.py)
        self.engine = ImageEngine(self.cfg, dict(os.environ) if env is None else env, self.dir / "strata-image.log",
                                  self._say)
        self.tok = None
        self._system: str | None = None          # HunyuanImage 3.0's system prompt, read once
        self.totals = {"images": 0, "jobs": 0, "seconds": 0.0, "failed": 0}
        self.last: dict | None = None              # the last job's figures (/v1/status)
        # the measured speed, an edit's apart (its reference images): for each size made (in megapixels), one step's
        # seconds (a guided step two) and the rest's an image (estimate_s)
        self.rates: dict[str, dict[float, list[float]]] = {}
        self.progress: tuple | None = None         # (step, steps) of the running job
        # the job in the GPU's turn as the Monitor shows it (live()), and the jobs that ended, newest last (recent()):
        # the one record of an image job's progress, which /metrics reads
        self.job: dict | None = None
        self.ended: collections.deque[dict] = collections.deque(maxlen=100)

    def note(self) -> str | None:
        """What the user is told about this setup at start and on the web page: the sizes of the image model and its
        text encoder lowered for the RAM (setup's "quant_lowered"), or None."""
        q = self.cfg.get("quant_lowered")
        if not isinstance(q, dict) or not q.get("from") or not q.get("to"):
            return None
        parts = []
        if q["to"] != q["from"]:
            parts.append(f"the image model at {q['to']} instead of {q['from']}")
        if q.get("text_from") and q.get("text_to") and q["text_to"] != q["text_from"]:
            parts.append(f"its text encoder at {q['text_to']} instead of {q['text_from']}")
        if not parts:
            return None
        return (f"Image generation runs {' and '.join(parts)}: the larger sizes need about "
                f"{q.get('need_gb', '?')} GB of RAM beside the text model, which leaves about "
                f"{q.get('ram_left_gb', '?')} GB. Run setup with --image-ssd-offload on to keep them "
                "(what does not fit is read from the SSD, slower).")

    def defaults(self) -> dict:
        """What a request leaves out gets from this image model (model_defaults), the LoRA adapters its DiT takes
        (loras: each with name, path, rank, scale, trigger, targets and size_mib; serve/loras.py), and the speed it was
        measured at (rates: [megapixels, a step's seconds, the rest's] of each size, for the web page's estimate,
        estimate_s)."""
        return {**model_defaults(self.cfg), "loras": self.loras.listing(), "loras_take": self.loras.takes,
                "size_exp": SIZE_EXP,
                "rates": {k: [[mp, *v] for mp, v in sorted(pts.items())] for k, pts in self.rates.items()}}

    @staticmethod
    def _work(job: dict) -> tuple[float, float]:
        """One image's (steps, megapixels): the steps it runs (a strength's), a guided step two (MOSS-Image's
        skip-layer guidance one more in its window), the size the engine makes."""
        mw, mh = made_size(job)
        passes = 2 if job.get("cfg") else 1
        if job.get("slg_layers") and job.get("slg_scale"):
            passes += job.get("slg_stop", 1.0) - job.get("slg_start", 0.0)
        return job.get("steps_run", job["steps"]) * passes, round(mw * mh / 1e6, 3)

    def _measured(self, job: dict, r: dict) -> None:
        """One image's engine figures (its IMAGE line) into the rates of its size, a running mean that follows the
        latest; the eight sizes made last are kept."""
        steps, mp = self._work(job)
        rate = [r.get("sampling_s", 0) / steps, r.get("text_s", 0) + r.get("decode_s", 0)]
        pts = self.rates.setdefault("edit" if job.get("refs") else "generate", {})
        was = pts.pop(mp, None)
        pts[mp] = rate if was is None else [round((a + b) / 2, 4) for a, b in zip(was, rate)]
        while len(pts) > 8:
            del pts[next(iter(pts))]

    def estimate_s(self, job: dict) -> float | None:
        """The seconds `job`'s images take at the speed measured (an edit's, else a generation's; None: none yet): the
        size measured nearest, a step's time scaled to the pixels (to the SIZE_EXP power above it), the rest's by
        them."""
        pts = self.rates.get("edit" if job.get("refs") else "generate") or next(iter(self.rates.values()), None)
        if not pts:
            return None
        steps, mp = self._work(job)
        near = min(pts, key=lambda p: abs(math.log(mp / p)))
        step_s, rest_s = pts[near]
        scale = mp / near
        return job["n"] * (steps * step_s * scale ** (SIZE_EXP if scale > 1 else 1) + rest_s * scale)

    def check_n(self, job: dict) -> None:
        """Refuse more images than budget_s allows at the speed measured (before it is measured, UNMEASURED_N)."""
        n, budget = job["n"], float(self.cfg.get("budget_s") or BUDGET_S)
        est = self.estimate_s(job)
        if n == 1:
            return
        if est is None:
            if n > UNMEASURED_N:
                raise ImageRequestError(f"n: at most {UNMEASURED_N} until this server has made an image (its speed "
                                        "is not measured yet)")
        elif est > budget:
            most = max(1, int(budget // (est / n)))
            raise ImageRequestError(f"n: {n} images of this size and steps take about {est / 60:.1f} min here, over "
                                    f"the {budget / 60:.0f} min limit (budget_s); at most {most}")

    def _file_keys(self) -> tuple[str, ...]:
        """The config keys that name this model's files: HunyuanImage 3.0's tokenizer and system prompt stand where the
        others' text encoder does."""
        if self.family == "hunyuan-image-3":
            return ("exe", "diffusion_model", "vae", "tokenizer", "system_prompt") + \
                (("vision",) if self.cfg.get("vision") else ())
        return ("exe", "diffusion_model", "vae", "llm") + (("vision",) if self.cfg.get("vision") else ())

    def missing(self) -> list[str]:
        return [self.cfg[k] for k in self._file_keys() if not Path(self.cfg[k]).exists()]

    def _tokenizer_path(self) -> str:
        """HunyuanImage 3.0's tokenizer.json or Qwen3-VL's text encoder GGUF, as the config names them."""
        return self.cfg["tokenizer"] if self.family == "hunyuan-image-3" else self.cfg["llm"]

    def system_text(self) -> str:
        """HunyuanImage 3.0's system prompt (setup's en_unified.txt), read once: the ids start with it."""
        if self._system is None:
            self._system = Path(self.cfg["system_prompt"]).read_text(encoding="utf-8")
        return self._system

    def _base_size(self) -> int:
        """HunyuanImage 3.0's image base size: its <img_size_N> token's N (1024 in the model's config), one for every
        bucket, as its image processor takes it from the resolution group."""
        params = self.cfg.get("resolutions")
        return int((params or {}).get("base_size") or 1024)

    def _say(self, text: str) -> None:
        print(f"[strata] {text}", flush=True)
        if self.log is not None:
            try:
                self.log.write(f"[images] {text}\n")
                self.log.flush()
            except (OSError, ValueError):
                pass

    def _prune(self) -> None:
        """Keep the newest keep_files images of response_format url (the oldest go)."""
        files = sorted((p for p in self.dir.iterdir() if FILE_NAME.match(p.name)), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - self.keep_files)]:
            with contextlib.suppress(OSError):
                p.unlink()

    def file(self, name: str) -> Path | None:
        """A stored image (response_format url) by its name, or None."""
        if not FILE_NAME.match(name or ""):
            return None
        p = self.dir / name
        return p if p.is_file() else None

    def _ready(self) -> None:
        if self.tok is None:
            self.tok = load_tokenizer(self._tokenizer_path(), self.family)
        if not self.engine.alive():
            self.engine.start()

    def ids(self, job: dict) -> list[int]:
        if self.tok is None:
            self.tok = load_tokenizer(self._tokenizer_path(), self.family)
        if "ids" not in job:
            if self.family == "hunyuan-image-3":   # the DiT reads the whole sequence, the image blocks in it
                bucket = job.get("bucket") or hunyuanimage3_bucket(self.cfg, job["width"], job["height"])
                # a reference image's block names the bucket nearest its shape: the engine reads it at that size
                job["ref_buckets"] = [hunyuanimage3_bucket(self.cfg, *image_dims(r)) for r in job.get("refs") or ()]
                job["ids"] = hunyuanimage3_ids(self.tok, model_prompt(job), bucket, self._base_size(),
                                               self.system_text(), job["ref_buckets"])
            else:
                ids = prompt_ids(self.tok, model_prompt(job), len(job.get("refs") or ()), self.family)
                neg = None
                if job.get("cfg"):   # guidance's negative prompt, in the same form (its pads as the prompt's)
                    neg = prompt_ids(self.tok, job["negative_prompt"], len(job.get("refs") or ()), self.family)
                if self.family == "moss-image":   # its pipeline's max_sequence_length
                    for what, x in (("prompt", ids), ("negative_prompt", neg or ())):
                        if len(x) > MOSS_MAX_IDS:
                            raise ImageRequestError(f"{what}: {len(x)} tokens in MOSS-Image's form, over its "
                                                    f"{MOSS_MAX_IDS}")
                job["ids"] = ids
                if neg is not None:
                    job["neg_ids"] = neg
        return job["ids"]

    def _choose_size(self, job: dict, timeout: float, cancel: threading.Event | None) -> None:
        """HunyuanImage 3.0's size "auto": the bucket the model writes after the prompt (the engine's RATIO, the
        reference's img_ratio task, greedy), once a job; the job's bucket, size and ids follow it.  The engine keeps
        the sequence's rows for the images that follow."""
        if self.tok is None:
            self.tok = load_tokenizer(self._tokenizer_path(), self.family)
        job["ref_buckets"] = [hunyuanimage3_bucket(self.cfg, *image_dims(r)) for r in job.get("refs") or ()]
        prefix = hunyuanimage3_prefix_ids(self.tok, model_prompt(job), self._base_size(), self.system_text(),
                                          job["ref_buckets"])
        self._live(stage="size")
        r = self.engine.ask(f"RATIO 0 {','.join(str(i) for i in prefix)}{self._refs_words(job)}", "RATIO", timeout,
                            cancel)
        index = int(r["index"])
        bucket = next((b for b in hunyuanimage3_buckets(self.cfg) if b["ratio_index"] == index), None)
        if bucket is None:
            raise ImageJobFailed(f"the image engine chose ratio {index}, which this model's table has not")
        job.update(bucket=bucket, width=int(bucket["width"]), height=int(bucket["height"]), choose_size=False,
                   ratio_s=float(r.get("ratio_s", 0)))
        job.pop("ids", None)
        self._live(size=f"{job['width']}x{job['height']}")
        self._say(f"image size: the model chose {job['width']}x{job['height']} (ratio {index}, "
                  f"{job['ratio_s']:.1f} s)")

    def drop_of(self, job: dict) -> int:
        """The system turn's ids the DiT does not take: Mage-Flow's edit form has its own."""
        return MF_EDIT_DROP if self.family == "mage-flow" and job.get("refs") else self.drop

    @staticmethod
    def _moss_words(job: dict) -> str:
        """MOSS-Image's words of IMG: its steps and shift (the engine makes its sigmas), the blocks the negative pass
        skips, and skip-layer guidance's (moss_request)."""
        words = (f" STEPS:{job['steps']} SHIFT:{job.get('shift', MOSS_SHIFT)!r}"
                 f" USKIP:{','.join(str(b) for b in job.get('uncond_skip', MOSS_UNCOND_SKIP)) or 'none'}")
        if job.get("slg_layers") and job.get("slg_scale"):
            words += (f" SLG:{','.join(str(b) for b in job['slg_layers'])} SLGSCALE:{job['slg_scale']!r}")
        if job.get("slg_start", 0.0) != 0.0 or job.get("slg_stop", 1.0) != 1.0:
            words += f" SLGSTART:{job['slg_start']!r} SLGSTOP:{job['slg_stop']!r}"
        return words

    @staticmethod
    def _lora_words(job: dict) -> str:
        """IMG's and NEED's adapters: LORA:<strength>:<path> for each, in the order the request gave them (the strength
        first, so a Windows path's colons are not read as one; the path's separators %XX'd, as the engine splits the
        line on whitespace - encode_path)."""
        return "".join(f" LORA:{use['scale']!r}:{lora_api.encode_path(use['path'])}"
                       for use in job.get("loras") or ())

    @staticmethod
    def _cfg_words(job: dict) -> str:
        """IMG's guidance words: the negative prompt's ids and the scale (none without guidance)."""
        if not job.get("cfg"):
            return ""
        return f" NEG:{','.join(str(i) for i in job['neg_ids'])} CFG:{job['cfg']:g}"

    def stage(self, job: dict) -> None:
        """An edit's reference images, img2img's image and a mask into files the engine reads (job["ref_paths"],
        job["init_path"], job["mask_path"]); unstage removes them."""
        if (job.get("refs") or job.get("init")) and not job.get("ref_dir"):
            work = Path(tempfile.mkdtemp(prefix="refs-", dir=self.dir))
            job["ref_dir"] = str(work)
            job["ref_paths"] = []
            for i, data in enumerate(job.get("refs") or ()):
                p = work / f"ref_{i}"
                p.write_bytes(data)
                job["ref_paths"].append(str(p))
            for key in ("init", "mask"):
                if job.get(key):
                    p = work / key
                    p.write_bytes(job[key])
                    job[f"{key}_path"] = str(p)

    @staticmethod
    def unstage(job: dict) -> None:
        files = ("init_path", "mask_path", "te_path", "neg_te_path")
        for p in (job.pop("ref_paths", []) or []) + [job.pop(k) for k in files if job.get(k)]:
            with contextlib.suppress(OSError):
                os.unlink(p)
        if job.get("ref_dir"):
            with contextlib.suppress(OSError):
                os.rmdir(job.pop("ref_dir"))

    @staticmethod
    def _refs_words(job: dict, need: bool = False) -> str:
        """The engine's words of a job's files: REF:, MASK: and INIT:, the text states TE: and NEGTE: (text_states),
        and STRENGTH: below 1 (its repr, the double strength_run counted with).  `need`: NEED's, where a strength's
        base image (an edit's first reference) is counted by INIT:- when no mask or init image names it."""
        words = ("".join(f" REF:{p}" for p in job.get("ref_paths") or ())
                 + (f" MASK:{job['mask_path']}" if job.get("mask_path") else "")
                 + (f" INIT:{job['init_path']}" if job.get("init_path") else "")
                 + (f" TE:{job['te_path']}" if job.get("te_path") else "")
                 + (f" NEGTE:{job['neg_te_path']}" if job.get("neg_te_path") else ""))
        if job.get("strength") is not None:
            if not need:
                words += f" STRENGTH:{job['strength']!r}"
            elif not (job.get("mask_path") or job.get("init_path")):
                words += " INIT:-"
        return words

    def need(self, job: dict) -> dict:
        """What the image needs, from the engine (started when it is not running): least_mib, full_mib (every layer
        resident), held_mib (what it holds now), free_mib (the device's free VRAM), cap_mib (the most the engine could
        have) and host_mib (the prefix cache in pinned RAM at cap_mib)."""
        with self.lock:
            self._ready()
            w, h = made_size(job)
            self.stage(job)
            n = len(self.ids(job)) - self.drop_of(job)
            neg = f" NEG:{len(job['neg_ids']) - self.drop_of(job)}" if job.get("cfg") else ""
            return self.engine.ask(f"NEED {w} {h} {n}{self._refs_words(job, need=True)}{neg}{self._lora_words(job)}",
                                   "NEED", 600)

    def free_mib(self) -> int:
        """The device's free VRAM, as the engine sees it (its parts left as they are)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask("FREE", "FREE", 60)["free_mib"])

    def fit(self) -> int:
        """The engine's budget set to what it holds and what is free now, less the reserve -> the budget (MiB)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask(f"FIT {self.reserve}", "BUDGET", 600).get("budget_mib", 0))

    def release(self, everything: bool) -> None:
        """The engine lets go of its cache (layers, activations, buffers); `everything`: its fixed weights too."""
        with self.lock:
            if self.engine.alive():
                try:
                    self.engine.ask("RELEASE ALL" if everything else "RELEASE", "RELEASE", 120)
                except ImageJobFailed as e:
                    self._say(f"image engine: release failed ({e})")

    def close(self, shutdown: bool = False) -> None:
        """Leave the image daemon (it stays, warm, for the next server: a model switch); `shutdown`: end it."""
        with self.lock:
            self.engine.close(shutdown)

    def run(self, job: dict, cancel: threading.Event | None = None, first: int = 0,
            pause=None, preview=None) -> tuple[list[bytes], dict]:
        """Make `job`'s images (parse_request's dict) from image `first` -> (the PNG files' bytes in order, figures).
        The caller holds the GPU turn; `pause()` true after an image ends the call there (make_images gives the turn
        to the requests waiting for it).  ImageJobFailed when the engine fails, times out or the job is cancelled.
        `preview(image, index, png)`: the job's partial images (partial_images), as the engine makes them."""
        timeout = float(self.cfg.get("timeout_s") or 3600)
        t0 = time.monotonic()
        images, parts = [], []
        work = Path(tempfile.mkdtemp(prefix="job-", dir=self.dir))
        try:
            with self.lock:
                self._ready()
                self.stage(job)
                if job.get("loras"):
                    self._say("image job: adapters " + ", ".join(f"{u['name']} at {u['scale']:g}"
                                                                 for u in job["loras"]))
                if job.get("choose_size"):
                    self._choose_size(job, timeout, cancel)
                mw, mh = made_size(job)
                ids = ",".join(str(i) for i in self.ids(job))
                sched = self.cfg.get("schedule")
                sig = (base_sigmas(job["steps"], mw, mh) if sched == "base" else
                       mage_sigmas(job["steps"]) if sched == "mage" else [])
                sigmas = " SIGMAS:" + ",".join(f"{x:.7g}" for x in sig) if sig else ""
                # HunyuanImage 3.0's sigmas are the engine's own (from the step count): its step count and the
                # guidance token's scale go instead
                guide = (f" STEPS:{job['steps']} GUID:{job['guidance']:g}" if sched == "hunyuanimage3" else
                         self._moss_words(job) if self.family == "moss-image" else "")
                for i in range(first, job["n"]):
                    if i > first and pause is not None and pause():
                        break
                    if cancel is not None and cancel.is_set():
                        self.totals["failed"] += 1
                        raise ImageJobFailed("cancelled (the client left)")
                    run = job.get("steps_run", job["steps"])
                    self.progress = (i * run, job["n"] * run)
                    self._live(image=i + 1, step=0, steps=run,
                               stage="refs" if job.get("refs") or job.get("init") else "text", sampling_at=None)
                    out = work / f"img_{i}.rgba"
                    shows = partial_steps(run, job.get("partial_images") or 0) if preview is not None else []
                    shown = " PREVIEW:" + ",".join(str(k) for k in shows) if shows else ""

                    def peek(index, done, path, i=i):
                        try:
                            png = png_rgba(Path(path).read_bytes(), job["width"], job["height"], mw, mh)
                            Path(path).unlink(missing_ok=True)
                        except OSError:
                            return
                        preview(i, index, png)

                    def step(done, steps, i=i):
                        self.progress = (i * steps + done, job["n"] * steps)
                        self._live(step=done, steps=steps)

                    def stage(name):
                        self._live(stage=name, **({"sampling_at": time.monotonic()} if name == "sampling" else {}))
                    try:
                        r = self.engine.ask(f"IMG {mw} {mh} {job['seed'] + i} {self.drop_of(job)} {out} {ids}{sigmas}"
                                            f"{guide}{self._refs_words(job)}{self._cfg_words(job)}{shown}"
                                            f"{self._lora_words(job)}", "IMAGE",
                                            timeout, cancel, step, stage, **({"preview": peek} if shows else {}))
                    except ImageJobFailed as e:
                        self.totals["failed"] += 1
                        why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                        self._say(f"image job failed: {why}")
                        raise (LoraRejected if isinstance(e, LoraRejected) else ImageJobFailed)(why) from e
                    images.append(png_rgba(out.read_bytes(), job["width"], job["height"], mw, mh))
                    parts.append(r)
                    self._measured(job, r)
        finally:
            self.progress = None
            self.unstage(job)
            for p in work.iterdir():
                with contextlib.suppress(OSError):
                    p.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
        secs = time.monotonic() - t0
        figures = {"seconds": round(secs, 2), "steps": job.get("steps_run", job["steps"]), "cfg": job.get("cfg") or 1.0,
                   "n": len(images), "refs": len(job.get("refs") or ()),
                   **({"strength": job["strength"]} if job.get("strength") is not None else {}),
                   **({"img2img": True} if job.get("init") else {}), **({"mask": True} if job.get("mask") else {}),
                   "size": f"{job['width']}x{job['height']}", "seed": job["seed"], "engine": "strata-image",
                   "quant": self.cfg.get("quant"),
                   "text_encode_ms": round(sum(r.get("text_s", 0) for r in parts) * 1000, 1),
                   **({"size_choice_s": job["ratio_s"]} if "ratio_s" in job else {}),
                   "sampling_s": round(sum(r.get("sampling_s", 0) for r in parts), 2),
                   "vae_decode_s": round(sum(r.get("decode_s", 0) for r in parts), 2),
                   "plan": {k: v for k, v in parts[-1].items() if k not in ("text_s", "sampling_s", "decode_s")},
                   **({"guidance": job["guidance"], "ratio": job["bucket"]["ratio_index"]}
                      if job.get("bucket") is not None else {}),
                   **({"layout": True} if job.get("layout_prompt") else {}),
                   **({"enhance": job["enhanced"]} if job.get("enhanced") else {})}
        if (mw, mh) != (job["width"], job["height"]):      # made larger and cut: the web page says so
            figures["made_size"] = f"{mw}x{mh}"
        if job.get("requested_size"):                      # made at the model's bucket instead: the web page says so
            figures["requested_size"] = job["requested_size"]
        if job.get("loras"):                               # the IMAGE line's count and VRAM, and a tag's names left out
            figures.update(lora_n=int(parts[-1].get("lora_n") or 0), lora_mib=parts[-1].get("lora_mib"))
        if job.get("lora_unknown"):
            figures["lora_unknown"] = list(job["lora_unknown"])
        if first + len(images) == job["n"]:
            self.totals["jobs"] += 1
        self.totals["images"] += len(images)
        self.totals["seconds"] += secs
        self.last = figures
        return images, figures

    def store(self, data: bytes, fmt: str = "png") -> str:
        """Keep one image for response_format url -> its file name (32 random hex digits: the name is the access, and
        the output_format)."""
        name = f"{secrets.token_hex(16)}.{fmt}"
        (self.dir / name).write_bytes(data)
        self._prune()
        return name

    def keep(self, data: bytes) -> str:
        """An image a client sends back (a Responses image_generation_call's result) kept as a stored image -> its
        file name; ValueError when it is not a PNG, JPEG, WebP or JPEG XL.  The name is its content's hash: the same
        image, sent again with every turn of a conversation, is one file (its time renewed, so pruning keeps it)."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            fmt = "png"
        elif data[:3] == b"\xff\xd8\xff":
            fmt = "jpeg"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            fmt = "webp"
        elif data[:2] == b"\xff\x0a" or data[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
            fmt = "jxl"
        else:
            raise ValueError("not a PNG, JPEG, WebP or JPEG XL image")
        name = f"{hashlib.sha256(data).hexdigest()[:32]}.{fmt}"
        p = self.dir / name
        if p.is_file():
            p.touch()
        else:
            p.write_bytes(data)
            self._prune()
        return name

    def response(self, images: list[bytes], job: dict, figures: dict, base_url: str) -> dict:
        data = []
        fmt = job.get("output_format") or "png"
        applied = [{"name": u["name"], "scale": u["scale"]} for u in job.get("loras") or ()]
        for i, png in enumerate(images):
            item = {"revised_prompt": model_prompt(job), "seed": job["seed"] + i}
            if applied:   # the adapters this image was made with
                item["lora"] = [dict(u) for u in applied]
            img = encode_image(png, fmt, job.get("output_compression", 100), job.get("background") == "opaque")
            if job["response_format"] == "url":
                item["url"] = f"{base_url}/v1/images/files/{self.store(img, fmt)}"
            else:
                item["b64_json"] = base64.b64encode(img).decode("ascii")
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": self.model, "output_format": fmt,
                "size": f"{job['width']}x{job['height']}", "background": job.get("background") or "auto",
                "quality": job.get("quality") or "auto", "usage": usage(job), "strata": figures}

    # ---- the progress the Monitor shows (/metrics' live.image) and the jobs that ended (its Requests table)
    STAGE_LABELS = {"waiting": "Making room on the GPU", "size": "Choosing the image's size",
                    "refs": "Reading the reference images",
                    "text": "Reading the prompt", "sampling": "Sampling", "decode": "Decoding the image (VAE)"}

    def begin(self, job: dict) -> None:
        """The job has the GPU's turn (gpu_turn): from here it is the running job (again, after make_images gave the
        turn away between its images)."""
        if self.job is not None:
            self.job["stage"] = "waiting"
            return
        self.job = {"started": time.time(), "t0": time.monotonic(), "stage": "waiting", "image": 1, "images": job["n"],
                    "step": 0, "steps": job.get("steps_run", job["steps"]), "size": f"{job['width']}x{job['height']}",
                    "refs": len(job.get("refs") or ()), "edit": bool(job.get("refs")), "sampling_at": None}

    def _live(self, **kw) -> None:
        if self.job is not None:
            self.job.update(kw)

    def live(self) -> dict | None:
        """The running job's stage, step n of the steps, the seconds so far and a step, its size and image i of n, and
        its overall progress (0..1: each image a share; its sampling by the steps, the stages around it at their
        boundaries)."""
        j = dict(self.job) if self.job is not None else None
        if j is None:
            return None
        now = time.monotonic()
        stage, done, steps = j["stage"], j["step"], max(1, j["steps"])
        step_s = round((now - j["sampling_at"]) / done, 2) if j["sampling_at"] is not None and done > 0 else None
        within = {"waiting": 0.0, "size": 0.01, "refs": 0.02, "text": 0.05, "sampling": 0.1 + 0.8 * done / steps,
                  "decode": 0.9}.get(stage, 0.0)
        frac = (j["image"] - 1 + within) / max(1, j["images"])
        return {"stage": stage, "label": self.STAGE_LABELS.get(stage, stage), "step": done, "steps": j["steps"],
                "image": j["image"], "images": j["images"], "size": j["size"], "refs": j["refs"], "edit": j["edit"],
                "elapsed_s": round(now - j["t0"], 1), "step_s": step_s, "progress": round(min(1.0, frac), 3)}

    def end(self, error: str | None, cancelled: bool = False) -> None:
        """The running job ended: one row of the Monitor's Requests table (the time it got the GPU, its size, the
        references, the seconds and each stage's, done or the error)."""
        j, self.job = self.job, None
        if j is None:
            return
        last = self.last if error is None and self.last is not None else {}
        self.ended.append({"kind": "image", "time": j["started"], "duration_s": round(time.monotonic() - j["t0"], 1),
                           "finish": "stop" if error is None else "cancel" if cancelled else "error",
                           "error": None if cancelled else error,
                           "size": j["size"], "n": j["images"], "refs": j["refs"], "edit": j["edit"],
                           "seconds": last.get("seconds"), "text_encode_ms": last.get("text_encode_ms"),
                           "sampling_s": last.get("sampling_s"), "vae_decode_s": last.get("vae_decode_s")})

    def recent(self) -> list[dict]:
        return list(self.ended)

    def status(self) -> dict:
        out = {"enabled": True, "model": self.model, "engine": "strata-image", "quant": self.cfg.get("quant"),
               "gpu_policy": self.policy, "running": self.engine.alive(), **self.totals}
        if self.note():
            out["note"] = self.note()
        if self.progress:
            out["step"], out["steps"] = self.progress
        if self.last:
            out["last"] = self.last
        return out


class ImageModels:
    """The config's image models: its "images" section (the default) and those of "image_models", one engine on the
    GPU at a time.  A request names its model by "model" (pick: an unknown name or none is the default's, as OpenAI's
    clients send their own); its job takes the GPU's turn (GpuTurn: other servers' jobs on the GPU wait for it, and
    it for theirs) and ends every other daemon of the GPU first, this server's other models' and other servers'.
    Anything else is the model of the last job's (the Monitor's, /v1/status)."""

    def __init__(self, cfgs: list[dict], log=None, env: dict | None = None, separate: bool = False):
        self.separate = separate     # `env` names a card no text engine of the server takes (the policy separate)
        self.gens: dict[str, ImageGen] = {}
        for c in cfgs:
            g = ImageGen(c, log=log, env=env)
            if g.model in self.gens:
                raise ValueError(f'two image models are named {g.model!r} ("images" and "image_models")')
            self.gens[g.model] = g
        self.default = self.current = next(iter(self.gens.values()))
        self.gpu = GpuTurn(self.default.engine.env)   # the models share one environment, so one GPU

    def __getattr__(self, name):
        return getattr(self.current, name)

    def pick(self, name) -> ImageGen:
        return self.gens.get(name, self.default) if isinstance(name, str) else self.default

    def editors(self) -> list[ImageGen]:
        """The models that edit (take reference images), the default first."""
        return [g for g in self.gens.values() if can_edit(g.cfg)]

    def pick_edit(self, name, img2img: bool = False) -> ImageGen:
        """The model of an edit (/v1/images/edits): the one named, as pick gives it; none named, the default when it
        edits or the request is img2img's (a strength or a mask, which every model takes), else the first model that
        edits (the chat's edit_image prefers those too), so that an instruction to a default without edits (MageTrail)
        is an edit rather than its img2img.  With no model that edits, the default (img2img)."""
        if isinstance(name, str) and name in self.gens:
            return self.gens[name]
        if can_edit(self.default.cfg) or img2img:
            return self.default
        return next(iter(self.editors()), self.default)

    def activate(self, gen: ImageGen) -> None:
        """Make `gen` the model on the GPU: every other daemon of the GPU ended, this server's other models' and other
        servers' (the caller holds the GPU's turn, GpuTurn)."""
        mine = daemon_address(gen.engine.command(), gen.engine.env)
        names = {daemon_address(g.engine.command(), g.engine.env): g.model for g in self.gens.values()}
        for addr in self.gpu.keep_only(mine):
            gen._say(f"image model: {names.get(addr, 'another server')}'s engine ended for {gen.model}")
        self.current = gen

    def give_back(self) -> None:
        """Before the text engine starts: every model's engine lets go of all it holds.  A job that ran while the text
        engine was unloaded kept its layers (policy keep: nothing to make room from), and the text engine that starts
        after it would find the card full (no expert cache: it stops before the prompt).  Not on a card of their own
        (separate), where a job may be running."""
        if self.separate:
            return
        for g in self.gens.values():
            if g.engine.alive():
                g._say(f"image model: {g.model}'s engine gives the GPU back for the model")
                g.release(True)

    def missing(self) -> list[str]:
        return [f for g in self.gens.values() for f in g.missing()]

    def recent(self) -> list[dict]:
        """The jobs that ended, every model's, oldest first."""
        return sorted((r for g in self.gens.values() for r in g.recent()), key=lambda r: r.get("time", 0))

    def status(self) -> dict:
        return {**self.current.status(), "models": list(self.gens), "default": self.default.model}

    def file(self, name: str) -> Path | None:
        return next((f for g in self.gens.values() if (f := g.file(name)) is not None), None)

    def close(self, shutdown: bool = False) -> None:
        for g in self.gens.values():
            g.close(shutdown)

    def about(self) -> list[dict]:
        """Each model as the web page's About shows it, the default first: its name, the DiT's and the text encoder's
        quantizations, its steps and whether it edits."""
        return [{"id": g.model, "quant": g.cfg.get("quant"), "text_quant": g.cfg.get("text_quant"),
                 "steps": g.cfg.get("steps"), "edits": can_edit(g.cfg)} for g in self.gens.values()]

    def folders(self) -> list:
        """The models' LoRA folders (serve/loras.py), each once."""
        out: list = []
        for g in self.gens.values():
            if all(f is not g.loras for f in out):
                out.append(g.loras)
        return out

    def models(self) -> list[dict]:
        """Each model's name, defaults and whether it edits (the web page's choice), the default first."""
        return [{"id": g.model, "defaults": g.defaults(), "edits": can_edit(g.cfg)} for g in self.gens.values()]


# ------------------------------------------------------------------------------------------- chat's image tools
OPTION_KEYS = ("size", "quality", "output_format", "output_compression", "background")
FILE_REF = re.compile(r"/v1/images/files/([0-9a-f]{32}\.(?:png|jpeg|webp|jxl))")
TOOLS: list[dict[str, Any]] = [
    {"name": "generate_image",
     "description": "Make an image from a description with the image model on this computer. "
                    "Describe it in detail: the subject, the setting, the style, the light, and any text it shows in "
                    "quotes. The image is shown to the user; the result says where it is.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "the image's description"},
         "size": {"type": "string",
                  "description": "WIDTHxHEIGHT, e.g. 1024x1024, 1536x1024 (wide), 1024x1536 (tall)"},
         "seed": {"type": "integer", "description": "the same seed and prompt make the same image"}},
         "required": ["prompt"]}},
    {"name": "edit_image",
     "description": "Change an image, or make a new one from it (its subject, its style): the conversation's last "
                    "image (one the user attached or one made here) unless `image` names another (a "
                    "/v1/images/files/... link from an earlier result). The new image is shown to the user. An image "
                    "model without edits draws the prompt over the image instead (img2img): describe the whole "
                    "picture there.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "what to change, or what to make from the image"},
         "image": {"type": "string", "description": "the image: a /v1/images/files/... link; default the last one"},
         "size": {"type": "string", "description": "WIDTHxHEIGHT; default the image's shape at about 1024x1024"},
         "strength": {"type": "number",
                      "description": "how much of the image may change, above 0 to 1 (lower keeps more of it); "
                                     "default 1 for an edit, 0.87 for img2img (less with few steps)"},
         "seed": {"type": "integer"}},
         "required": ["prompt"]}},
]


class ImageTools:
    """generate_image and edit_image for a chat (or Responses) request that asks for them: the model writes a call,
    the server makes the image in the GPU's turn as /v1/images does, keeps it (/v1/images/files/), and the call's
    result tells the model where it is.  With several image models (ImageModels) the tools take a "model" too,
    each model named and described in it, so the user's words choose one ("draw it with MageTrail", "an anime
    illustration"); none named is the default.  A request that names its image model (`model`) gets that one
    alone; a Responses request's input_image_mask (`mask`) is the mask of its edits.  With MOSS-Image, generate_image
    takes a layout too (its boxes, which the chat's model drafts; parse_layout).  The hub run_with_mcp runs
    (template_tools, routes, call, settings)."""

    def __init__(self, svc, images, messages: list, load, normalize, options: dict | None = None,
                 model: str | None = None, mask: bytes | None = None):
        self.models = images if isinstance(images, ImageModels) else None
        gen = images.default if isinstance(images, ImageModels) else images
        if model is not None:   # the request's own image model (strata_images {"model": ...}): that one alone
            gen = images.pick(model) if isinstance(images, ImageModels) else gen
            if gen.model != model:
                raise ValueError(f"strata_images: model must be one of "
                                 f"{', '.join(images.gens if isinstance(images, ImageModels) else [gen.model])}")
            self.models = None
        self.svc, self.gen, self.messages, self.load, self.normalize = svc, gen, messages, load, normalize
        # a Responses request's image_generation tool: its size, quality, output_format, output_compression and
        # background, over what the model's call says (size "auto": the call's)
        self.options = {k: v for k, v in (options or {}).items()
                        if k in OPTION_KEYS and v is not None and not (k == "size" and v == "auto")}
        self.settings = {"max_rounds": 4}
        self.made: list[str] = []          # the images this request made (file names)
        self.max_calls: int | None = None  # a Responses request's max_tool_calls: the calls that run, at most
        self.calls = 0
        self.mask = mask                   # an edit's mask (Responses' input_image_mask), over the image it edits
        # a streamed Responses request's partial_images (the tool's, 1..3): each call's partial images, put here as
        # (index, the image in the tool's output_format) for run_with_mcp to send while the call runs
        self.partials: queue.Queue | None = None
        self.partial_images = 0

    def stream_partials(self, k) -> None:
        """A streamed request whose image_generation tool asks for partial_images k (1..3): make them."""
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 3:
            self.partial_images, self.partials = k, queue.Queue()

    def _gens(self, edits: bool = False) -> list[ImageGen]:
        """The image models the tools offer, the default first; for edit_image those that edit first (every model
        takes an image: one without edits as img2img's)."""
        gens = list(self.models.gens.values()) if self.models is not None else [self.gen]
        return [g for g in gens if can_edit(g.cfg)] + [g for g in gens if not can_edit(g.cfg)] if edits else gens

    def template_tools(self, exclude=()) -> list[dict]:
        out = []
        for t in TOOLS:
            gens = self._gens(t["name"] == "edit_image")
            if not gens or t["name"] in exclude:
                continue
            t = dict(t)
            moss = [g.model for g in gens if g.family == "moss-image"]
            if t["name"] == "generate_image" and moss:   # MOSS-Image's layout of boxes, drafted by the chat's model
                params = dict(t["parameters"])
                only = "" if len(moss) == len(gens) else f" (only {', '.join(moss)} reads one)"
                params["properties"] = {**params["properties"], "layout": {
                    "type": "object",
                    "description": "where things and text go, when that matters (a poster, a sign, several panels)"
                                   + only + "; the prompt is then a short description of the whole image",
                    "properties": {
                        "summary": {"type": "object", "properties": {"caption": {
                            "type": "string", "description": "the whole image: its scene, style and colours"}}},
                        "items": {"type": "array", "items": {"type": "object", "properties": {
                            "label": {"type": "string", "enum": list(MOSS_LABELS)},
                            "caption": {"type": "string", "description": "how it looks"},
                            "text": {"type": "string", "description": "a text item's exact words"},
                            "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                                     "description": "[x_min, y_min, x_max, y_max] in the image's pixels from its "
                                                    "top-left corner"}},
                            "required": ["label", "caption", "bbox"]}}},
                    "required": ["items"]}}
                t["parameters"] = params
            if len(self._gens()) > 1:   # the model's choice: each one's name and what it is good for
                about = _qimage().ABOUT
                choice = {v[0]: k.removesuffix("-uncensored") for k, v in _qimage().IMAGE_MODELS.items()}
                lines = "; ".join(f"{g.model}: {about.get(choice.get(g.model, ''), '')}".rstrip(": ") for g in gens)
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "model": {
                    "type": "string", "enum": [g.model for g in gens],
                    "description": f"the image model, when the user asks for one or for what one is made for "
                                   f"({lines}); default {gens[0].model}"}}
                t["parameters"] = params
            out.append(t)
        return out

    def routes(self) -> dict:
        return {t["name"]: (None, t["name"]) for t in self.template_tools()}

    def _file(self, name: str) -> bytes | None:
        p = (self.models or self.gen).file(name)
        return p.read_bytes() if p is not None else None

    def has_image(self) -> bool:
        """Is there an image an edit could take (one this request made, a picture in the conversation, or a link to a
        kept one)?  Nothing is read."""
        if self.made:
            return True
        for msg in self.messages:
            content = msg.get("content")
            for item in content if isinstance(content, list) else [content]:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("source"):
                    return True
                text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
                if FILE_REF.search(text or ""):
                    return True
        return False

    def _image(self, which: str) -> bytes | None:
        """The image an edit takes: a /v1/images/files/ link (or its file name), else the conversation's last."""
        m = FILE_REF.search(which or "") or re.fullmatch(r"([0-9a-f]{32}\.(?:png|jpeg|webp|jxl))", which or "")
        if m:
            return self._file(m.group(1))
        if self.made:
            return self._file(self.made[-1])
        for msg in reversed(self.messages):
            content = msg.get("content")
            parts = content if isinstance(content, list) else [content]
            for item in reversed(parts):
                if isinstance(item, dict) and item.get("type") == "image" and item.get("source"):
                    return self.load(item["source"])
                text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
                found = FILE_REF.findall(text or "")
                if found and (data := self._file(found[-1])) is not None:
                    return data
        return None

    def call(self, name: str, arguments, cancel: threading.Event | None = None) -> dict:
        t0 = time.monotonic()
        if self.max_calls is not None and self.calls >= self.max_calls:
            text = f"error: not run: this response may make {self.max_calls} image tool call(s) (max_tool_calls)"
            return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                    "tool": name, "images": []}
        self.calls += 1
        args = arguments if isinstance(arguments, dict) else {}
        urls: list[str] = []
        try:
            req = {**{k: args[k] for k in ("prompt", "size", "seed") if args.get(k) not in (None, "")}, **self.options}
            if args.get("layout") not in (None, "", {}) and name == "generate_image":   # MOSS-Image's boxes
                layout = args["layout"]
                if isinstance(layout, str):
                    try:
                        layout = json.loads(layout)
                    except ValueError:
                        raise ImageRequestError("layout: a JSON object") from None
                w, h = parse_size(req.get("size"), 2048)
                req["layout"] = tidy_layout(layout, w, h)
            if name == "edit_image" and args.get("strength") not in (None, ""):
                req["strength"] = args["strength"]
            gens = self._gens(name == "edit_image")
            asked = args.get("model")
            gen = next((g for g in gens if g.model == asked), gens[0] if gens else self.gen)
            if asked not in (None, "") and gen.model != asked:
                raise ImageRequestError(f"model: one of {', '.join(g.model for g in gens)}, not {asked!r}")
            if name == "generate_image":
                job = parse_request(req, gen.cfg)
            elif name == "edit_image":
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no image in the conversation to edit")
                job = parse_edit_request(req, [self.normalize(data)], gen.cfg, self.mask)
            else:
                raise ImageRequestError(f"there is no tool named {name!r}")
            preview = None
            if self.partials is not None:
                job["partial_images"] = self.partial_images
                fmt, comp, opaque = job["output_format"], job["output_compression"], job["background"] == "opaque"
                q = self.partials
                preview = lambda _i, index, png: q.put((index, encode_image(png, fmt, comp, opaque)))  # noqa: E731
            images, figures, _ = make_images(self.svc, gen, job, cancel, preview)
            fmt = job["output_format"]
            names = [gen.store(encode_image(png, fmt, job["output_compression"], job["background"] == "opaque"),
                               fmt) for png in images]
            self.made += names
            urls = [f"/v1/images/files/{n}" for n in names]
            by = f" with {gen.model}" if len(self._gens()) > 1 else ""
            text = (f"The image is made{by} ({job['width']}x{job['height']}, seed {job['seed']}, "
                    f"{figures['seconds']:.0f} s) and the user sees it: {urls[0]}. To change it, call edit_image "
                    f"(its image: {urls[0]}).")
            ok = True
        except (ValueError, ImageJobFailed) as e:
            text, ok = f"error: {e}", False
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": urls,
                **({"image": {"size": f"{job['width']}x{job['height']}", "quality": job["quality"],
                              "background": job["background"], "output_format": job["output_format"],
                              "revised_prompt": model_prompt(job)}} if ok else {})}


# ------------------------------------------------------------------------------------------- the GPU turn
def resolve_policy(svc, gen: ImageGen, need: dict) -> str:
    """auto -> keep / shrink / unload for this job (see the module's docstring); a fixed policy is kept, except that
    an engine that is not running has nothing to give back (keep).  `need`: ImageGen.need's answer."""
    eng = svc.engine
    running = not hasattr(eng, "alive") or eng.alive()
    if not running or not hasattr(eng, "unload"):
        return "keep"
    if gen.policy != "auto":
        return gen.policy
    if gen.cfg.get("uma"):
        return "keep"                         # an APU: the image job and the engine share the RAM either way
    room = need["free_mib"] + need["held_mib"] - gen.reserve
    if room >= need["full_mib"]:
        return "keep"
    if engine_elastic(eng):
        return "shrink"
    if room >= need["least_mib"]:
        return "keep"                         # the image's layers streamed: slower than unloading only for a big batch
    return "unload"


def engine_elastic(eng) -> bool:
    """An engine that resizes its expert cache on request (--vram-elastic, #533)."""
    if not hasattr(eng, "vram"):
        return False
    if (getattr(eng, "info", None) or {}).get("vram_elastic"):
        return True
    spawn = getattr(eng, "spawn", None)
    args = spawn[1] if isinstance(spawn, (list, tuple)) and len(spawn) > 1 else []
    return "--vram-elastic" in (args or [])


def _client_gone(svc) -> Exception:
    """The server module's ClientGone (it may run as __main__, so not imported by name)."""
    import sys
    return getattr(sys.modules.get(type(svc).__module__), "ClientGone", RuntimeError)()


def check_room(need: dict, ram_mib: int | None) -> None:
    """An image the GPU cannot make even with the text model unloaded (its least above the engine's cap), or whose
    prefix cache would not fit the free RAM (`ram_mib`, None: not known): refused before the job, not shrunk (the
    user's decision, 2026-10-21)."""
    least, cap, host = (int(need.get(k) or 0) for k in ("least_mib", "cap_mib", "host_mib"))
    hint = "use fewer or smaller reference images, or a smaller size"
    if cap and least > cap:
        raise ImageRequestError(f"this image needs at least {least / 1024:.1f} GiB of VRAM and the GPU has "
                                f"{cap / 1024:.1f} GiB for images: {hint}")
    if ram_mib is not None and host > ram_mib - RAM_SPARE_MIB:
        raise ImageRequestError(f"this image keeps {host / 1024:.1f} GiB of its prompt's cache in RAM and "
                                f"{max(ram_mib, 0) / 1024:.1f} GiB is free: {hint}")


def ram_available_mib() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available >> 20)
    except (ImportError, OSError):
        return None


def partial_steps(steps: int, k: int) -> list[int]:
    """The steps after which an image's k partial images (OpenAI's partial_images, 1..3) are made: spread evenly over
    the sampling, never its last (the image itself)."""
    if k <= 0 or steps < 2:
        return []
    return sorted({max(1, min(steps - 1, round(steps * (j + 1) / (k + 1)))) for j in range(k)})


def same_model_file(a: str, b: str) -> bool:
    """Whether two GGUF paths hold the same file: one file on disk (the same path, or a link to it), or the same size
    and the same SHA-256 in setup's finish marks (<file>.done's "sha256 <hex>", written once the file matched its
    pin)."""
    try:
        pa, pb = Path(a), Path(b)
        if os.path.samefile(pa, pb):
            return True
        if pa.stat().st_size != pb.stat().st_size:
            return False

        def pins(p: Path) -> set[str]:
            text = p.with_name(p.name + ".done").read_text(encoding="utf-8", errors="replace")
            return {f[1] for f in (line.split() for line in text.splitlines()) if len(f) == 2 and f[0] == "sha256"}
        return bool(pins(pa) & pins(pb))
    except OSError:
        return False


def text_states(svc, gen: ImageGen, job: dict) -> None:
    """MOSS-Image's text encoder by the chat's engine (its ENCODE command) when that engine reads the file of the image
    model's text encoder (the config's "llm", same_model_file): the prompt's states, and under guidance the negative
    prompt's, into files the IMG line names (job["te_path"] and job["neg_te_path"]: TE: and NEGTE:).  The engine sets
    its conversation aside and puts it back.  Otherwise - another model, no engine running, parallel slots, or an
    engine that refused - nothing: the image engine opens the encoder itself.  Called in the job's turn (gpu_turn),
    before the text engine makes room."""
    eng = getattr(svc, "engine", None)
    te = gen.cfg.get("llm")
    slots = getattr(eng, "batch", 0)
    if (not te or eng is None or not hasattr(eng, "encode") or slots or not eng.alive()
            or not same_model_file(getattr(eng, "model_path", ""), te)):
        return
    made: dict[str, str] = {}
    tokens, ms = 0, 0.0
    try:
        for key, ids in (("te_path", gen.ids(job)), ("neg_te_path", job.get("neg_ids") if job.get("cfg") else None)):
            if ids is None:
                continue
            stem = gen.dir / f"te-{secrets.token_hex(6)}"
            ids_file, out = stem.with_suffix(".ids"), stem.with_suffix(".ste")
            ids_file.write_text(",".join(str(i) for i in ids), encoding="ascii")
            try:
                r = eng.encode(str(ids_file), str(out))
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(ids_file)
            made[key] = str(out)
            tokens += r["tokens"]
            ms += r["ms"]
    except Exception as e:                       # noqa: BLE001 - the image engine encodes the text instead
        for p in made.values():
            with contextlib.suppress(OSError):
                os.unlink(p)
        gen._say(f"image job: the chat's engine did not encode the prompt ({e}); the image engine reads it")
        return
    job.update(made)
    gen._say(f"image job: the prompt's states from the chat's engine ({tokens} tokens, {ms:.0f} ms)")


# MOSS-Image's prompt enhancer: the chat's model writes the image out as a caption or as a layout of boxes.  Written
# after the model's own enhancer (OpenMOSS/MOSS-Image skills/moss-image-prompt-enhancer, Apache-2.0): the same choice
# and the same draft, its boxes in the canvas' pixels from the top-left, which moss_layout_prompt compiles.
MOSS_ENHANCE_SYSTEM = """You turn an idea for an image into a prompt for MOSS-Image, a text-to-image model, for a \
{width}x{height} pixel image. {choice}

Keep everything the idea asks for: its subjects, their number and relations, the medium, the style, what must not \
appear, and every word that must be visible, exactly as written (its language, capitals, punctuation and line \
breaks). Settle the details it leaves open yourself: what the subject looks like and does, the setting, the framing, \
the light, the materials, the colours and the mood. Describe only what is seen: no quality tags, no notes about the \
prompt. Write the description in English unless the idea asks for another language; visible words stay as given.

Answer with one JSON object and nothing else.
A caption: {{"caption": "..."}} - flowing prose; positions in words; visible words in quotes.
A layout: {{"summary": {{"caption": "the whole image: its scene, style and treatment", "meta": ["medium or genre"], \
"canvas": "single"}}, "items": [{{"label": "object", "caption": "how it looks", \
"bbox": [x_min, y_min, x_max, y_max]}}, {{"label": "text", "caption": "its lettering and surface", \
"text": "the exact words", "bbox": [...]}}]}} - label is \
object, character or text; a text item carries its exact words in "text" ("\\n" for a line break); each bbox is in \
pixels from the image's top-left corner, x from 0 to {width} and y from 0 to {height}. A box marks where a thing is, \
not a frame to draw. Give text a box large enough to be read at this size; items may overlap."""
MOSS_ENHANCE_CHOICE = {
    "auto": "Write a caption for an ordinary scene, photograph or illustration; write a layout when the placement of "
            "text, a poster, several panels or the arrangement of several things matters.",
    "caption": "Write a caption.",
    "layout": "Write a layout."}
MOSS_ENHANCE_TOKENS = 3072         # a layout of a dozen items with their captions fits


def _json_object(text: str) -> dict:
    """The first JSON object in a model's answer (a fenced block or bare), or ValueError."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the answer has no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("the answer is not a JSON object")
    return value


def enhance_prompt(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None) -> None:
    """MOSS-Image's prompt enhancer (job["enhance"]: "auto", "caption" or "layout"): the chat's model, in its own turn
    (Service.prepare and Service.run, thinking off), writes the prompt out - a caption becomes job["enhanced_prompt"],
    a layout job["layout"] and its JSON job["layout_prompt"] (its boxes kept on the canvas).  job["enhanced"] says what
    was made (the response's "strata" carries it; the image's revised_prompt is the prompt the model read).  An answer
    that cannot be read, or no chat model, leaves the prompt as it was, and says why there."""
    t0 = time.monotonic()
    mode = job["enhance"]
    info: dict = {"mode": mode}
    text: list[str] = []
    try:
        if not hasattr(svc, "prepare") or getattr(svc, "engine", None) is None:
            raise ValueError("no chat model is loaded")
        system = MOSS_ENHANCE_SYSTEM.format(width=job["width"], height=job["height"], choice=MOSS_ENHANCE_CHOICE[mode])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": job["prompt"]}]
        ids, thinking, max_new = svc.prepare(messages, None, {"enable_thinking": False}, MOSS_ENHANCE_TOKENS,
                                             cancel=cancel)
        for kind, x in svc.run(ids, thinking, None, max_new, {}, cancel or threading.Event()):
            if kind == "event" and getattr(x, "kind", None) == "content":
                text.append(x.text or "")
        draft = _json_object("".join(text))
        if isinstance(draft.get("layout"), dict):   # the layout under a name of its own
            draft = draft["layout"]
        if "items" in draft and mode != "caption":
            layout = parse_layout(tidy_layout(draft, job["width"], job["height"]))
            job["layout_prompt"] = moss_layout_prompt(layout, job["prompt"], job["width"], job["height"])
            job["layout"] = layout
            info.update(made="layout", layout=layout)
        elif isinstance(draft.get("caption"), str) and draft["caption"].strip() and mode != "layout":
            job["enhanced_prompt"] = draft["caption"].strip()
            info.update(made="caption", prompt=job["enhanced_prompt"])   # a stream's events carry no revised_prompt
        else:
            raise ValueError(f"the answer is not a {'layout' if mode == 'layout' else 'caption'}")
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
        info["error"] = str(e)
        if text:   # what the model wrote, to see why
            info["answer"] = "".join(text)[:4000]
        gen._say(f"image job: the prompt enhancer gave nothing usable ({e}); the prompt is used as it is")
    info["seconds"] = round(time.monotonic() - t0, 2)
    job["enhanced"] = info
    job.pop("ids", None)
    job.pop("neg_ids", None)


def make_images(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None,
                preview=None) -> tuple[list[bytes], dict, str]:
    """`job`'s images in the GPU's turn (gpu_turn) -> (the PNG files' bytes, figures, the policy used).  When a request
    waits for the turn after an image, the turn is given to it and taken again for the rest, so a chat is not held up
    by a batch; not under unload (giving it would load the text model and unload it again for each image) nor on a card
    of its own (separate: the chat does not wait for it).  Too many
    images for budget_s are refused first (ImageGen.check_n).  MOSS-Image's prompt enhancer (enhance_prompt) runs
    first, in the chat model's turn."""
    if job.get("enhance") and not job.get("enhanced"):
        enhance_prompt(svc, gen, job, cancel)
    gen.check_n(job)
    images: list[bytes] = []
    parts: list[dict] = []
    used = "keep"
    while len(images) < job["n"]:
        with gpu_turn(svc, gen, job, cancel) as policy:
            used = policy

            def pause() -> bool:
                if used in ("unload", "separate"):   # separate: the chat requests do not wait for the job
                    return False
                with svc.status_lock:
                    return svc.status["queued"] > 0
            made, figures = gen.run(job, cancel, len(images), pause, preview)
        images += made
        parts.append(figures)
        if len(images) < job["n"]:
            gen._live(stage="waiting")
            gen._say(f"image job: {len(images)} of {job['n']} images made; a request waits for the GPU's turn")
    figures = dict(parts[-1])
    for k in ("seconds", "text_encode_ms", "sampling_s", "vae_decode_s"):
        figures[k] = round(sum(p[k] for p in parts), 2)
    figures["n"], figures["seed"] = len(images), job["seed"]
    gen.last = figures
    gen.end(None)
    return images, figures, used


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used.  On a
    card of its own (ImageModels.separate), that card's turn alone (separate_turn)."""
    if isinstance(getattr(svc, "images", None), ImageModels) and svc.images.separate:
        with separate_turn(svc, gen, job, cancel) as policy:
            yield policy
        return
    with svc.status_lock:
        svc.status["queued"] += 1
    queued = True
    error: str | None = None
    held_gpu: GpuTurn | None = None
    try:
        with svc._turn(cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
            queued = False
            models = getattr(svc, "images", None)
            if isinstance(models, ImageModels):   # every server's image jobs on this GPU in turn
                def waiting():
                    gen._say("image job: waiting for another server's image job on this GPU")
                    with svc.status_lock:
                        svc.status["phase"] = "waiting for another server's image job"
                models.gpu.acquire(cancel, waiting)
                held_gpu = models.gpu
                models.activate(gen)
            gen.begin(job)
            ctl = getattr(svc.engine, "ctl", None) if getattr(svc.engine, "batch", 0) else None
            if ctl is not None:
                while not ctl.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        raise _client_gone(svc)
            try:
                if ctl is not None:                   # running parallel requests finish first
                    while getattr(svc, "live_reqs", None):
                        if cancel is not None and cancel.is_set():
                            raise _client_gone(svc)
                        time.sleep(0.25)
                if gen.family == "moss-image":   # its text encoder by the chat's engine, while that holds the GPU
                    text_states(svc, gen, job)
                try:
                    need = gen.need(job)
                except ImageJobFailed:
                    gen.totals["failed"] += 1
                    raise
                check_room(need, ram_available_mib())
                policy = resolve_policy(svc, gen, need)
                with svc.status_lock:
                    svc.status.update(busy=True, phase="generating an image", started=time.time(),
                                      first_token=None, prompt_tokens=None, generated=None, max_tokens=None)
                restore = _make_room(svc, gen, policy, need)
                try:
                    budget = gen.fit()
                    gen._say(f"image job: {policy}, the image engine's budget {budget} MiB (the DiT resident from "
                             f"{need['full_mib']} MiB, at least {need['least_mib']} MiB)")
                    yield policy
                finally:
                    _give_back(svc, gen, policy, restore)
                    with svc.status_lock:
                        svc.status["busy"] = False
                    svc.last_request_at = time.time()
            finally:
                if ctl is not None:
                    ctl.release()
    except BaseException as e:
        error = str(e) or type(e).__name__
        raise
    finally:
        if held_gpu is not None:
            held_gpu.release()
        if error is not None:                     # a job that ends well: make_images ends it, after its last image
            gen.end(error, cancelled=cancel is not None and cancel.is_set())
        if queued:
            with svc.status_lock:
                svc.status["queued"] -= 1


@contextlib.contextmanager
def separate_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """gpu_turn on a card no text engine takes: only that GPU's turn (GpuTurn), so chat requests run meanwhile and the
    text engine is left as it is (the policy separate); the chat's status is not touched.  MOSS-Image's prompt read by
    the chat's engine (text_states) takes the service's turn for that read alone."""
    models: ImageModels = svc.images
    error: str | None = None
    held = False
    try:
        def waiting():
            gen._say("image job: waiting for another server's image job on this GPU")
        models.gpu.acquire(cancel, waiting)
        held = True
        models.activate(gen)
        gen.begin(job)
        if gen.family == "moss-image":
            with svc._turn(cancel):
                text_states(svc, gen, job)
        try:
            need = gen.need(job)
        except ImageJobFailed:
            gen.totals["failed"] += 1
            raise
        check_room(need, ram_available_mib())
        try:
            budget = gen.fit()
            gen._say(f"image job: separate, the image engine's budget {budget} MiB on its own GPU (the DiT resident "
                     f"from {need['full_mib']} MiB, at least {need['least_mib']} MiB)")
            yield "separate"
        finally:
            _give_back(svc, gen, "separate", None)
            svc.last_request_at = time.time()
    except BaseException as e:
        error = str(e) or type(e).__name__
        raise
    finally:
        if held:
            models.gpu.release()
        if error is not None:                     # a job that ends well: make_images ends it, after its last image
            gen.end(error, cancelled=cancel is not None and cancel.is_set())


def _make_room(svc, gen: ImageGen, policy: str, need: dict):
    """Do what `policy` says before the job; -> what _give_back needs."""
    eng = svc.engine
    if policy == "shrink":
        want = max(0, need["full_mib"] - need["held_mib"]) + gen.reserve
        try:
            out = eng.vram(want)
            gen._say(f"image job: the engine's expert cache gave VRAM back ({out.get('vram_free_mib')} MiB free)")
            return "shrunk"
        except Exception as e:                       # noqa: BLE001 - the job still runs, with what is free
            gen._say(f"image job: the engine could not give VRAM back ({e}); the job runs with what is free")
            return None
    if policy == "unload":
        saved = None
        if hasattr(eng, "session_file") and not getattr(eng, "batch", 0):
            path = str(gen.dir / f"llm-session-{secrets.token_hex(6)}.bin")
            try:
                r = eng.session_file("save", path)
                saved = path
                gen._say(f"image job: the conversation ({r.get('tokens')} tokens) saved; unloading the model")
            except Exception as e:                   # noqa: BLE001 - unloading still frees the GPU
                gen._say(f"image job: the conversation could not be saved ({e}); unloading the model")
                with contextlib.suppress(OSError):
                    os.unlink(path)
        eng.unload()
        if svc.vision is not None and hasattr(svc.vision, "unload"):
            svc.vision.unload()
        return saved or "unloaded"
    return None


def _give_back(svc, gen: ImageGen, policy: str, restore) -> None:
    """After the job: the image engine keeps its layers (the next image starts warm) only under keep or separate with
    spare_mib still free; else it lets them go, and the text engine gets back what it gave."""
    keep = policy in ("keep", "separate")
    if keep:
        try:
            keep = gen.engine.alive() and gen.free_mib() >= gen.spare
        except ImageJobFailed:
            keep = False
    if not keep:
        gen.release(True)
    eng = svc.engine
    if restore == "shrunk":
        try:
            eng.vram(svc.vram_reserve)               # the reserve asked for last (None: the start's)
        except Exception as e:                       # noqa: BLE001
            gen._say(f"image job: the expert cache did not grow back ({e}); POST /v1/vram can retry")
    elif policy == "unload" and restore:
        if restore != "unloaded":
            svc.restore_after_load = restore         # Service.ensure_loaded restores it after the next start
        if gen.reload == "eager":
            try:
                svc.ensure_loaded()                  # the caller still holds the FIFO
            except Exception as e:                   # noqa: BLE001 - the next request tries again
                gen._say(f"image job: the model did not load again yet ({e}); the next request loads it")


def restore_session(svc) -> None:
    """After an engine start: restore the conversation an image job saved before it unloaded the engine."""
    path = getattr(svc, "restore_after_load", None)
    if not path:
        return
    svc.restore_after_load = None
    try:
        r = svc.engine.session_file("restore", path)
        print(f"[strata] the conversation from before the image job is back ({r.get('tokens')} tokens)", flush=True)
    except Exception as e:                           # noqa: BLE001 - the engine runs, without the old cache
        print(f"[strata] the conversation from before the image job was not restored ({e}); the next request "
              "reads its prompt again", flush=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
