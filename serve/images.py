# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/images.py - image generation (Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0, MOSS-Image)
through XeStrata's own image engine, xestrata-image.

POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url) plus Strata's
own field seed; POST /v1/images/edits the same with reference images (OpenAI's multipart form, image or image[]; or JSON
with images: [{"image_url": ...}]), which the image model edits or takes as its subject (Qwen-Image-2.1's edits: the
vision encoder reads them into the prompt, the VAE's encoder into the DiT's).  Every model takes a mask there (OpenAI's:
a PNG of the first image's size whose transparent pixels are where it changes; the engine keeps the rest, its latent
and its pixels) and a strength (diffusers' img2img strength, above 0 and at most 1: how much of the schedule runs from
the first image's latent, strength_run; 1, the default of an edit, starts from noise).  A model without edits
(MageTrail, Mage-Flow without its edit model, one without its vision file) takes one image as img2img's: its
strength, else img2img_strength (with a mask, 1).  src/image/img_start.hpp.  The engine (src/image/strata_image.cpp,
`xestrata-image --serve`) starts with the first image and stays, outside the server (serve/image_daemon.py, the next
server of the same image model and GPU finds it: a model switch does not lose it): its kernels built once, the
prompt's text states and the layers that fit in VRAM kept between images, one image a request.  It runs Turbo's
8 steps (CFG 1, the Euler flow sampler, the model's own sigmas), the image the one stable-diffusion.cpp makes with the
model card's command for the same seed (within LPIPS 0.03); the base model ("schedule": "base") the steps asked for
(default the config's "steps", 40) on its scheduler's sigmas (the engine's qi_base_sigmas, from SCHED:base
and STEPS:), CFG 1 as well.  A size whose sides are
not multiples of 16 (1920x1080) is made at the sides rounded up (1920x1088) and its middle cut out.

Mage-Flow ("family": "mage-flow", "schedule": "mage"): its own prompt form (MF_PROMPT_HEAD, the 34 ids of its system
turn dropped), the steps asked for (default the config's: Turbo 4, the RL model 20, the base model 30) on its static
shift (the engine's mage_sigmas, from STEPS:), and guidance by guidance_scale or true_cfg_scale (the one scale it
has; default the config's
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
size is the bucket's, and "xestrata" names the size asked for (requested_size) when it differs.  Its guidance is one
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

Anima ("family": "anima", "schedule": "anima"): its prompt tokenized a part at a time (anima_tokens: the emphasis
"(words:weight)" read, the Qwen3 ids of each part with a weight of 1, the T5 tokenizer's ids with the part's weight and
</s> at the end), the T5 ids and weights going to the engine as T5:/T5W: (and NEGT5:/NEGT5W:), no system turn; its
steps (30; Turbo's config 10) on its schedule's shift 3, the sampler er_sde (the model card's) or euler, guidance at a
scale above 1 (4; Turbo's 1, none) with ANIMA_NEG as its negative prompt by default.  The web page's prompt starts
with its quality tags (prompt_start; none for the Aesthetic model, whose training dropped them).
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
the engine splits the line on it), an adapter's block weights ("blocks") after its strength (@<w0>,<w1>,...).  A
name the model's LoRA folder ("lora_dir") does not offer is refused where the field gives it and taken out of the
prompt with a warning where a tag does (the response's xestrata.lora_unknown).  "local_path" is a file inside that
folder only.  The response's data[] items carry the adapters applied as "lora": [{"name", "scale"}, with "blocks"
where given], and xestrata their count and VRAM (lora_n, lora_mib, the IMAGE line's).

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
  {"exe": "engine/xestrata-image", "diffusion_model": "...Q8_0.gguf", "schedule": "turbo"|"base", "steps": 8|40,
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
  crop_edit: false draws an edit with a mask at the output's size, without cutting the part out (plan_crop; a
  request's crop_edit overrides it)
  MOSS-Image's edit_adapters ({name: {"lora", "format", "vision"}}, setup's chinoll, armv5 and armv6) and
  edit_adapter (the one used; XESTRATA_MOSS_EDIT over it: moss_edit_adapter); edit_format: an edit_lora's layout
  (moss_edit_format)
  max_n: the most images one request makes; budget_s: the longest one request's images may take, by the speed this
  server measured (ImageGen.estimate_s; until it has made an image, at most UNMEASURED_N); schedule: Turbo's own
  sigmas or the base model's (the engine's qi_base_sigmas; no "schedule" is Turbo), Mage-Flow's static shift
  (the engine's mage_sigmas),
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
import functools
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
from typing import Any, Callable, cast

from serve import file_lock, option_values
from serve import downgrades
from serve import image_aux
from serve import loras as lora_api
from serve import envvars
from serve import vram_ledger
from serve import idle_watch
from serve.image_daemon import START_TIMEOUT_S
from serve.model_terms import _qimage

TURBO_STEPS = _qimage().DITS["turbo"][1]
MAX_STEPS = 100                    # the base model's
CFG_MAX = 30.0                     # guidance_scale / true_cfg_scale (vLLM-Omni: "typically 0.0-20.0")
MODEL_NAME = "qwen-image-2.1-turbo"
POLICIES = option_values.GPU_POLICIES
SIZE_STEP = 16                     # the VAE's 8x and the DiT's 2x2 patches: sides made at a multiple of 16, then cut
MIN_SIDE = 256
MAX_N = 16                         # "max_n": the most images one request makes
BUDGET_S = 600                     # "budget_s": the longest one request's images may take (ImageGen.check_n)
UNMEASURED_N = 4                   # the most images a request makes before the server has measured its speed
HELPER_REOPEN_S = 2.0              # an image job closed the text engine's helper: filled again after this long idle
# img2img's strength when a request gives none (a model without edits, an image and no mask), at most: 0.87, lowered
# for a model of few steps (img2img_strength).  An edit's, and a mask's alone, is 1.0, a full denoise
STRENGTH_IMG2IMG = 0.87
# the cut and its return (plan_crop, img_crop.hpp): the width of the soft edge the drawing goes back into the picture
# by, in pixels; the faintest stroke of the mask that counts (the engine's kFaint); and the context drawn around the
# part, on each side, as a share of the part's size (the square root of its box's area) - more for an edit with
# reference images beside the picture (a face from image 2), whose model has to place them in the scene
CROP_BLEND = 32
CROP_FAINT = 0.05
CROP_CONTEXT = 0.125
CROP_CONTEXT_REFS = 0.3
# MOSS-Image's edit with references besides the picture: its edit adapter matches them with the whole picture it was
# trained beside, and a cut part alone takes their content (the face swap of 2026-11-04 drew image 2's body in place of
# the face).  So the whole picture is edited in one pass (its whole
# box: the part outside the mask kept), and with the request's refine the part is then cut out of that and drawn again
# with the references at this strength: the same face swap kept image 2's face and drew its
# eyes and mouth finer; 0.7 turned the hair grey
CROP_REFINE_STRENGTH = 0.5
# an outpaint's (extend) fade of the change into the image from each side that grows, in pixels (at most a third of the
# image's shorter side, extend_canvas)
EXTEND_FEATHER = 48
# the cut's color match by default (color_match_mode): set by the images compared (C4)
COLOR_MATCH = "off"
# an upscale's second pass (parse_upscale): its strength, the low end of the 0.3 to 0.6 a second pass commonly takes
# (less drift from the first image); the largest scale and side; past max_side, drawn again a tile at a time, 1024
# pixels with a context of 128 and a blend of 32: with 32 and 8 a swimsuit's edge across a tile's border came out
# broken (Qwen-Image-2.1-Turbo, 2x of 832x1248, 2026-11-05); 128 and 32 drew it whole
UPSCALE_STRENGTH = 0.35
UPSCALE_MAX_SCALE = 4.0
UPSCALE_MAX_SIDE = 4096
UPSCALE_TILE = 1024
UPSCALE_PAD = 128
UPSCALE_BLUR = 32
# an upscale's first pass, by the engine's upscalers (src/upscale: UPS:, the UPS line): "auto" is RealESRGAN where the
# picture is drawn again after it (its finer edges for the drawing to start from) and SeedVR2 where it is not (its own
# restoration in one step, whatever the picture's style: the user's decision of 2026-11-05), RealESRGAN where SeedVR2
# does not fit the GPU; each method's weights are image_aux's, none for Lanczos
UPSCALE_METHODS = ("auto", "lanczos", "realesrgan", "seedvr2")
UPSCALE_WEIGHTS = {"realesrgan": "realesrgan"}
# SeedVR2's variants: 7B (photos, video, the default), 7B sharp (anime, blurred pictures, crisper detail), 3B (where the
# 7B's 8 bits do not fit the VRAM and the RAM together); each in FP16 (where the GPU holds it) or int8_convrot
SEEDVR2_VARIANTS = ("7b", "7b-sharp", "3b")
SEEDVR2_WORK_MIB = 4096   # what a 7B run takes besides its weights at a couple of megapixels (the DiT's activations)
# the color corrections after it (vLLM-Omni's SeedVR2 color_correction_method's names); none given, the upscaler's
# own: none for RealESRGAN, lab for SeedVR2 (its pipeline's)
UPSCALE_COLORS = ("none", "lab", "adain", "wavelet")
# /v1/images/upscale's models (its `model`, and /v1/models' entries) -> the method and SeedVR2's variant
UPSCALE_MODELS = {"realesrgan-x4plus": ("realesrgan", None), "lanczos": ("lanczos", None),
                  "seedvr2-7b": ("seedvr2", "7b"), "seedvr2-7b-sharp": ("seedvr2", "7b-sharp"),
                  "seedvr2-3b": ("seedvr2", "3b")}
# the parts' fix (fix_image, the edits' `fix`): a detection kept from a score of 0.5; what is redrawn (its box, or a
# person's mask) grown by FIX_GROW of the box's longer side (at least FIX_GROW_MIN pixels); cut out with FIX_MARGIN of
# that side around it on each side; a part whose shorter side is under FIX_MIN pixels left out (too few pixels to
# draw it from), one whose shorter side is FIX_GUIDE or more too (already drawn at the size a model draws, about 1 MP:
# crop_gen_size); drawn again at strength 0.5 and put back with a soft edge of FIX_BLEND pixels; the parts' detectors
# are image_aux's weights
FIX_PARTS = {"face": "yolo-face", "hands": "yolo-hand", "person": "yolo-person"}
FIX_PROMPTS = {"face": "a detailed, well-drawn face", "hands": "detailed, well-drawn hands",
               "person": "a detailed, well-drawn person"}
FIX_STRENGTH = 0.5
FIX_THRESHOLD = 0.5
FIX_GROW = 0.03
FIX_GROW_MIN = 4
FIX_MARGIN = 0.75
# crop_upscale auto: an upscaler (RealESRGAN, the one the redraw's auto takes) for a cut enlarged this many times or
# more, Lanczos below it - the larger the enlargement, the more of the drawing's base a plain resize leaves blurred
CROP_UPS_AUTO = 2.0
FIX_GUIDE = 1024
FIX_MIN = 16
FIX_BLEND = 24
AUX_FIELDS = ("name", "status", "bytes", "total", "license", "source", "error", "retry_s")   # a tool's downloads
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
MF_MAX_REFS = _qimage().FAMILY_CAPABILITIES["mage-flow"]["max_refs"]
MF_SHIFT = 6.0                     # Mage-Flow's static time shift
MF_NEG = " "                       # its pipeline's negative prompt when none is given
MAGETRAIL_QUALITY = "masterpiece, best quality, very aesthetic, aesthetic"   # its README's, after the tags
MF_SAMPLERS = ("euler", "er_sde")  # Mage-Flow's own Euler first; er_sde on request (without edits)
HUNYUANIMAGE3_STEPS = _qimage().DITS["hunyuanimage3"][1]
HUNYUANIMAGE3_GUIDANCE = _qimage().HUNYUANIMAGE3_GUIDANCE
HUNYUANIMAGE3_MAX_REFS = _qimage().HUNYUANIMAGE3_MAX_REFS
HUNYUANIMAGE3_VIT_PATCH = 16       # SigLIP2's patch, and the slots a reference's features take (its processor's)
HUNYUANIMAGE3_VIT_SLOTS = 1024
# MOSS-Image's prompt (its pipeline's encode_prompt: the chat template with thinking on, the user's turn alone), all of
# it the DiT's (no system turn to drop), at most its max_sequence_length
MOSS_PROMPT_HEAD = "<|im_start|>user\n"
MOSS_PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
MOSS_MAX_IDS = 4096
MOSS_STEPS = _qimage().DITS["moss-sft"][1]
MOSS_CFG = _qimage().MOSS_CFG       # block 22 skipped by the negative pass, no skip-layer guidance (slg_scale 0)
MOSS_SHIFT = _qimage().MOSS_SHIFT
MOSS_NEG = ""
MOSS_UNCOND_SKIP = [22]
MOSS_FEW_STEPS = 4                 # the 4-step adapter's step codes (its card: 1 to 4, 4 recommended)
MOSS_BLOCKS = 42                   # its DiT's blocks (the skip lists' range)
MOSS_SHIFT_MAX = 20.0
# MOSS-Image's edits: chinoll's Moss-Image-Edit adapter (the config's "edit_lora", at strength 1: below 1 the scene is
# drawn again rather than edited), as its model card sets it: CFG 5, 30 steps, block 22 skipped by the negative pass,
# up to four references, each at about 1 MP in its own shape (smaller than the image made, the scene zooms out); as it
# was trained, no image to the text encoder, the references' latents to the DiT
MOSS_EDIT_STEPS = 30
MOSS_EDIT_CFG = 5.0
MOSS_MAX_REFS = _qimage().FAMILY_CAPABILITIES["moss-image"]["max_refs"]
# the layouts edit adapters are trained in (moss_edit_format; ai-toolkit's training code and each adapter's model
# card): chinoll's Moss-Image-Edit (the references at timestep 0, seeing only each other); ai-toolkit's (ostris), whose
# text encoder sees the references too (each as "Picture N:" before the prompt, at most MOSS_PICTURE_PIXELS, 384 x
# 384), their latents at most MOSS_AITK_REF_PIXELS (its
# control_image_max_pixels, akhaliq's armV5's) and its steps and guidance MOSS-Image's own; ai-toolkit's with its
# kv_cache off (the references in the sequence at every step, at timestep 0); and "appended" (the references
# appended to the sequence at the image's timestep, no picture to the text encoder; thearkorbit's
# moss-image-edit-lora-poc), its references and steps as ai-toolkit's.  The engine's EDITFMT: word of each but the
# first (QiRefLayout)
MOSS_EDIT_FORMATS = ("moss-image-edit", "ai-toolkit", "ai-toolkit-no-kv", "appended")
MOSS_EDITFMT_WORDS = {"ai-toolkit": "aitk", "ai-toolkit-no-kv": "aitk-nokv", "appended": "appended"}
# the edit adapter setup's config names by default among its "edit_adapters" (tools/image/qimage.py says why); the
# config's "edit_adapter" chooses another, and XESTRATA_MOSS_EDIT over it (moss_edit_adapter)
MOSS_EDIT_DEFAULT = "chinoll"
MOSS_PICTURE_PIXELS = 384 * 384
MOSS_AITK_REF_PIXELS = 524288
ANIMA_STEPS = _qimage().DITS["anima-base"][1]
ANIMA_CFG = _qimage().ANIMA_DEFAULT_CFG
ANIMA_SHIFT = _qimage().ANIMA_SHIFT
ANIMA_SAMPLERS = ("er_sde", "euler")
ANIMA_NEG = ("worst quality, low quality, score_1, score_2, score_3, 6 fingers, 6 toes, ai-generated, bad eyes, "
             "bad pupils, bad iris, bad hands, bad fingers, watermark, patreon logo")
ANIMA_QUALITY = "masterpiece, best quality, score_7, safe, "
ANIMA_PAD = 151643                 # Qwen3's <|endoftext|>: the one token of an empty prompt
T5_END = 1                         # T5's </s>
# a layout (the prompt as MOSS-Image's JSON of boxes): its summary's fields and an item's
MOSS_SUMMARY_KEYS = ("caption", "meta", "canvas", "palette")
MOSS_ITEM_KEYS = ("label", "caption", "text", "bbox", "tags", "palette")
MOSS_LABELS = ("object", "character", "text")
MOSS_CANVASES = ("single", "multi_panel", "collage")
# an edit's reference image in the prompt: one pad, which the engine widens to the image's tokens
REF_PROMPT = "{sep}<image{i}><|vision_start|><|image_pad|><|vision_end|>"
REF_MAX_BYTES = 32 << 20
MAX_REFS = 10                      # Qwen-Image-2.1's own limit (its model card)
GROW_WAIT_S = 10.0                 # the expert cache's growing back after a shrink, tried again for this long
RAM_SPARE_MIB = 2048               # RAM an image job leaves free beside its prefix cache (the OS, the server)


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


# the words vLLM-Omni's form booleans take (pydantic's bool: t, y and the rest, any case), and enable and disable,
# which the image tools' forms have taken
FLAG_TRUE = ("true", "1", "yes", "on", "t", "y", "enable")
FLAG_FALSE = ("false", "0", "no", "off", "f", "n", "disable")


def parse_flag(v, name: str, default=None):
    """A request's true or false, the one reading of every image and video entry's: a JSON boolean (or 1 or 0, 1.0 or
    0.0, as pydantic's bool), or a form's word for one (FLAG_TRUE, FLAG_FALSE, any case) -> `default` when it is not
    given (absent, null or blank text).  Any other value is an ImageRequestError, as the entries' models refuse one:
    vLLM-Omni's form booleans and OpenAI's JSON ones (400 here, as the server's other errors in OpenAI's form; FastAPI
    answers vLLM-Omni's 422)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v in (0.0, 1.0):
        return v == 1.0
    s = str(v).strip().lower()
    if s in FLAG_TRUE:
        return True
    if s in FLAG_FALSE:
        return False
    raise ImageRequestError(f"{name}: true or false, not {v!r}")


def parse_number(v, name: str, lo: float, hi: float, default=None, whole: bool = False):
    """A number from lo to hi (hi math.inf: no bound above), the one reading of every image and video entry's: a JSON
    number, or a form's text of one (whole: an integer, as a JSON integer, a float equal to one or its text, 17.0 or
    "17.0", as pydantic's int takes them for vLLM-Omni's fields; a fraction, 17.5, refused; a float otherwise) ->
    `default` when it is not given (absent, null or blank text); else an ImageRequestError."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    span = f"{lo:g} or more" if hi == math.inf else f"from {lo} to {hi}" if whole else f"from {lo:g} to {hi:g}"
    bad = ImageRequestError(f"{name}: a whole number {span}" if whole else f"{name}: a number {span}")
    if isinstance(v, str):
        s = v.strip()
        try:
            if whole:
                m = re.fullmatch(r"([+-]?\d+)\.0+", s)
                v = int(m.group(1) if m else s)
            else:
                v = float(s)
        except ValueError:
            raise bad from None
    if whole and isinstance(v, float) and math.isfinite(v) and v.is_integer():
        v = int(v)
    # an int wider than a float is out of range, not a crash: math.isfinite() of one raises OverflowError, which no
    # entry's handler catches (a 500 for the user).  Only a float is looked at for infinity and NaN.
    if isinstance(v, bool) or not isinstance(v, int if whole else (int, float)) \
            or (isinstance(v, float) and not math.isfinite(v)) or not lo <= v <= hi:
        raise bad
    return v if whole else float(v)


def _int_field(req: dict, key: str, lo: int, hi: int, default):
    return parse_number(req.get(key), key, lo, hi, default, whole=True)


def _pipeline(family: str | None) -> str | None:
    return _qimage().family_capabilities(family).get("pipeline")


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
    mage = _pipeline(cfg.get("family")) == "flow"
    hunyuan = _pipeline(cfg.get("family")) == "guided_token"
    moss = _pipeline(cfg.get("family")) == "latent_chat"
    anima = _pipeline(cfg.get("family")) == "dual_tokenizer"
    if cfg.get("schedule") == "base":
        steps, least, fixed = int(cfg.get("steps") or _qimage().DITS["base"][1]), 2, False
    elif anima:   # Anima: any count on its shift
        steps, least, fixed = int(cfg.get("steps") or ANIMA_STEPS), 1, False
    elif cfg.get("schedule") == "mage":
        steps, least, fixed = int(cfg.get("steps") or _qimage().DITS["mf-turbo"][1]), 1, False
    elif cfg.get("schedule") == "hunyuanimage3":   # HunyuanImage 3.0: any count on the engine's own shift
        steps, least, fixed = int(cfg.get("steps") or HUNYUANIMAGE3_STEPS), 1, False
    elif cfg.get("schedule") == "moss":   # MOSS-Image: any count on its static shift
        steps, least, fixed = int(cfg.get("steps") or MOSS_STEPS), 1, False
    else:
        steps, least, fixed = TURBO_STEPS, TURBO_STEPS, True
    edit = (int(cfg.get("edit_steps") or steps) if cfg.get("edit_model") else
            int(cfg.get("edit_steps") or MOSS_EDIT_STEPS) if moss and cfg.get("edit_lora") else steps)
    # "strength": an img2img's at most, lowered by the steps as img2img_strength (the web page's imgStrength)
    out = {"steps": steps, "edit_steps": edit, "steps_min": least,
           "steps_max": MOSS_FEW_STEPS if cfg.get("few_step") else MAX_STEPS, "steps_fixed": fixed,
           "max_n": int(cfg.get("max_n") or MAX_N), "budget_s": float(cfg.get("budget_s") or BUDGET_S),
           "formats": output_formats(),
           "guidance": (float(cfg.get("guidance") or HUNYUANIMAGE3_GUIDANCE) if hunyuan
                        else float(cfg.get("cfg_scale") or 1.0) if mage
                        else float(cfg.get("cfg_scale") or MOSS_CFG) if moss
                        else float(cfg.get("cfg_scale") or ANIMA_CFG) if anima else 1.0),
           "guidance_token": hunyuan,
           "negative_prompt": MF_NEG if mage else MOSS_NEG if moss else ANIMA_NEG if anima else None,
           # what the web page's prompt starts as (its model's quality tags); serve adds nothing to a prompt
           "prompt_start": (MAGETRAIL_QUALITY if mage and str(cfg.get("model_name", "")).startswith("magetrail")
                            else (cfg.get("prompt_start", ANIMA_QUALITY) or "") if anima else ""),
           "transparency": cfg.get("transparency") is not False,
           "mask": True, "img2img": not can_edit(cfg), "strength": 1.0 if can_edit(cfg) else STRENGTH_IMG2IMG,
           # an edit with a mask: its part cut out by default (crop_wanted), MOSS-Image's second pass on request
           "crop_edit": cfg.get("crop_edit") is not False, "refine": bool(moss and can_edit(cfg)),
           # the image tools a chat may use with it (and the Images API's operations): the config's image_tools
           "tools": [t["name"] for t in TOOLS if cfg.get("image_tools") is None or t["name"] in cfg["image_tools"]],
           # the control image (image_controls): how it reaches the model (controlnet, reference; null: none) and the
           # kinds made
           "control": ({"mode": _controls().capability(cfg), "types": _controls().kinds_for(cfg),
                        "modes": {k: _controls().mode_of(cfg, k) for k in _controls().kinds_for(cfg)}}
                       if _controls().capability(cfg) else None),
           # the sampling guidance it takes (guidance_request): each method's parameters as default, least and most,
           # and whether it needs classifier-free guidance
           "guidance_methods": {name: {"needs_cfg": name in GUIDANCE_NEEDS_CFG,
                                       "params": {k: {"default": d, "min": lo, "max": hi}
                                                  for k, (d, lo, hi) in guidance_defaults(cfg)[name].items()}}
                                for name in guidance_methods(cfg)}}
    if hunyuan:
        out["buckets"] = [[b["width"], b["height"]] for b in hunyuanimage3_buckets(cfg)]
    if anima:
        out.update(shift=float(cfg.get("shift") or ANIMA_SHIFT), samplers=list(ANIMA_SAMPLERS),
                   sampler=cfg.get("sampler") or ANIMA_SAMPLERS[0])
    elif mage and not can_edit(cfg):   # Mage-Flow's Euler, er_sde on request (MageTrail's author uses either)
        out.update(samplers=list(MF_SAMPLERS), sampler=cfg.get("sampler") or MF_SAMPLERS[0])
    if moss and cfg.get("edit_lora"):   # its edits' guidance (MOSS_EDIT_CFG)
        out["edit_guidance"] = float(cfg.get("edit_cfg_scale") or MOSS_EDIT_CFG)
    if moss and cfg.get("few_step"):   # the 4-step adapter: its sampler (no shift, no skips), no guidance
        out.update(shift=MOSS_SHIFT, uncond_skip_layers=[], slg_layers=[], slg_scale=0.0, slg_start=0.0, slg_stop=1.0,
                   blocks=MOSS_BLOCKS, layout=True, enhance=True, few_step=True, negative_prompt=None)
    elif moss:
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


_DJXL: list[str | None] = []


def djxl() -> str | None:
    """libjxl's decoder, djxl, found as cjxl is (the same packages and the Windows 7z carry both); None: a JPEG XL
    input cannot be read."""
    if not _DJXL:
        beside = Path(__file__).resolve().parents[1] / "engine" / ("djxl.exe" if sys.platform == "win32" else "djxl")
        _DJXL.append(shutil.which("djxl") or (str(beside) if beside.is_file() else None))
    return _DJXL[0]


def is_jxl(data: bytes) -> bool:
    """A JPEG XL file: its bare codestream, or the ISO BMFF container."""
    return data[:2] == b"\xff\x0a" or data[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n"


def decode_jxl(data: bytes) -> bytes:
    """A JPEG XL input (one of this server's jxl outputs sent back, for one) as PNG by djxl: neither Pillow nor the
    engines' decoders read it.  ValueError (the request's 400) without djxl or where it fails."""
    exe = djxl()
    if exe is None:
        raise ValueError("a JPEG XL image needs libjxl's djxl on this computer (Ubuntu's libjxl-tools, Fedora's "
                         "libjxl-utils); send PNG, JPEG or WebP instead")
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    r = subprocess.run([exe, "-", "-", "--output_format", "png", "--quiet"], input=data, capture_output=True,
                       timeout=300, creationflags=no_window)
    if r.returncode != 0 or not r.stdout:
        raise ValueError(f"the JPEG XL image could not be read: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


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
    job["lora_unknown"] (the response's xestrata.lora_unknown).  A text encoder's adapter is not applied (the human's
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
    # does not offer is taken out and said in the response's xestrata (lora_unknown), a field's name is refused
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
    if _pipeline(cfg.get("family")) == "guided_token":    # its own table: the bucket nearest the shape asked for
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
    elif cfg.get("schedule") in ("hunyuanimage3", "moss", "anima"):   # the count on the engine's own sigmas
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
    bmap = cfg.get("lora_block_map")
    if loras and bmap:   # Anima-2.9B: block weights in the release's numbering (its adapters'), to the DiT's blocks
        n_blocks = max(int(b) for b in bmap) + 1
        for use in loras:
            if use.get("blocks") and len(use["blocks"]) == len(bmap):
                # the blocks 2.9B inserted take none of the release's adapter: theirs changes nothing
                weights = [1.0] * n_blocks
                for i, b in enumerate(bmap):
                    weights[int(b)] = use["blocks"][i]
                use["blocks"] = weights
    if loras:
        job["loras"] = loras
    if unknown:
        job["lora_unknown"] = list(dict.fromkeys(unknown))   # a name in both prompts said once
    if "samplers" in dflt:   # Anima's, Mage-Flow's without edits
        sampler = req.get("sampler")
        if sampler not in (None, "") and sampler not in dflt["samplers"]:
            raise ImageRequestError(f"sampler: {' or '.join(dflt['samplers'])}")
        job["sampler"] = sampler or dflt["sampler"]
    elif req.get("sampler") not in (None, ""):
        raise ImageRequestError("sampler: this image model has its own (Anima and Mage-Flow without edits take one)")
    guide = guidance_request(req, cfg, job)
    if guide:
        job["guide"] = guide
    if _pipeline(cfg.get("family")) == "latent_chat":
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
    return parse_number(req.get(key), key, lo, hi, default)


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
    few = cfg.get("few_step")
    if few:   # the 4-step adapter (darask0's DMAD): its own sampler, no guidance, its skips and shift none
        steps = req.get("steps", req.get("num_inference_steps"))
        if steps not in (None, "") and not (isinstance(steps, int) and 1 <= steps <= MOSS_FEW_STEPS):
            raise ImageRequestError(f"steps: 1 to {MOSS_FEW_STEPS} with this model's 4-step adapter")
        for key in ("uncond_skip_layers", "slg_layers", "shift"):
            if req.get(key) not in (None, [], ""):
                raise ImageRequestError(f"{key}: not with this model's 4-step adapter (its own sampler)")
        out.update(few_step=dict(few), uncond_skip=[], slg_layers=[], slg_scale=0.0)
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
        v = parse_number(req.get(key), key, 0, CFG_MAX)
        if v is not None:
            out[key] = v
    if _pipeline((cfg or {}).get("family")) == "guided_token":
        out.pop("negative_prompt", None)
        scale = out.get("guidance_scale")
        if scale is None:
            scale = out.get("true_cfg_scale")
        if scale is None:
            scale = float((cfg or {}).get("guidance") or HUNYUANIMAGE3_GUIDANCE)
        out["guidance"] = float(scale)
        return out
    if _pipeline((cfg or {}).get("family")) in ("flow", "latent_chat", "dual_tokenizer"):
        dflt = model_defaults(cfg or {})
        scale = out.get("true_cfg_scale", out.get("guidance_scale", dflt["guidance"]))
        if scale > 1 or (scale != 1 and _pipeline((cfg or {}).get("family")) == "latent_chat"):
            out["cfg"] = scale
            out.setdefault("negative_prompt", dflt["negative_prompt"])
        return out
    if out.get("true_cfg_scale", 1.0) > 1 and out.get("negative_prompt") is not None:
        out["cfg"] = out["true_cfg_scale"]
    return out


# the sampling guidance (the engine's guidance.hpp), by name: each method's parameters as (default, least, most).  The
# order a request lists them in does not matter: the engine combines the predictions by APG in place of plain CFG,
# then CFGZero*, skip layer guidance and CFG norm, and Detail Daemon to the sigma the model is asked at.  cfg_norm,
# cfg_zero_star and apg act on classifier-free guidance's two predictions: a request without guidance (true_cfg_scale
# at 1, or no negative prompt for Qwen-Image-2.1) is refused them.
GUIDANCE_METHODS: dict[str, dict[str, tuple]] = {
    "cfg_norm": {"strength": (1.0, 0.0, 100.0)},
    "cfg_zero_star": {},
    "apg": {"eta": (1.0, -10.0, 10.0), "norm_threshold": (5.0, 0.0, 50.0), "momentum": (0.0, -5.0, 1.0)},
    "skip_layer_guidance": {"layers": ([7, 8, 9], 0, 255), "scale": (3.0, 0.0, 10.0),
                            "start": (0.01, 0.0, 1.0), "end": (0.15, 0.0, 1.0),
                            "rescaling_scale": (0.0, 0.0, 10.0)},
    "detail_daemon": {"detail_amount": (0.1, -5.0, 5.0), "start": (0.2, 0.0, 1.0), "end": (0.8, 0.0, 1.0),
                      "bias": (0.5, 0.0, 1.0), "exponent": (1.0, 0.0, 10.0), "start_offset": (0.0, -1.0, 1.0),
                      "end_offset": (0.0, -1.0, 1.0), "fade": (0.0, 0.0, 1.0), "smooth": (True, None, None),
                      "cfg_scale_override": (0.0, 0.0, 100.0)},
}
GUIDANCE_NEEDS_CFG = ("cfg_norm", "cfg_zero_star", "apg")
# Skip layer guidance's blocks by model where the default (7, 8, 9, made for SD3's) does not suit it: Mage-Flow has
# 12 blocks, and three of them skipped wash its image out (block 10 alone does too, 4 and 6 do not, 2026-11-05 on the
# B70); Qwen-Image-2.1 keeps the default
GUIDANCE_SLG_LAYERS = {"mage-flow": [6]}


def guidance_defaults(cfg: dict) -> dict[str, dict[str, tuple]]:
    """GUIDANCE_METHODS with this model's own defaults (GUIDANCE_SLG_LAYERS)."""
    layers = GUIDANCE_SLG_LAYERS.get(cfg.get("family") or "")
    if not layers:
        return GUIDANCE_METHODS
    slg = dict(GUIDANCE_METHODS["skip_layer_guidance"])
    slg["layers"] = (list(layers),) + slg["layers"][1:]
    return {**GUIDANCE_METHODS, "skip_layer_guidance": slg}


def guidance_methods(cfg: dict) -> list[str]:
    """The guidance methods this image model's sampling takes (model_defaults' guidance_methods): Qwen-Image-2.1's
    and Mage-Flow's (MageTrail's too), each method checked on that model's sampler; none for
    the others (MOSS-Image keeps its own skip-layer guidance, slg_layers and the rest)."""
    family = cfg.get("family")
    return list(GUIDANCE_METHODS) if family is None or family == "" or \
        _qimage().family_capabilities(family).get("guidance_methods") else []


def chat_guidance_methods(cfg: dict) -> list[str]:
    """The guidance methods a chat's call can use with this model (generate_image's and edit_image's guidance): the
    chat's request leaves guidance's scale and negative prompt at the model's defaults, so those needing them
    (GUIDANCE_NEEDS_CFG) only where the model guides by default (its scale above 1 and a negative prompt of its own)."""
    dflt = model_defaults(cfg)
    guided = dflt["guidance"] > 1 and dflt["negative_prompt"] is not None
    return [m for m in guidance_methods(cfg) if guided or m not in GUIDANCE_NEEDS_CFG]


def guidance_request(req: dict, cfg: dict, job: dict) -> list[dict]:
    """The request's `guidance`, checked: a list of method names (guidance_methods) or {"name": ..., its parameters},
    each with the parameters it leaves out at their defaults (guidance_defaults: GUIDANCE_METHODS, or a model's
    own).  A method the model does not take, one named twice, an unknown parameter or one out of its range, and
    cfg_norm, cfg_zero_star or apg without classifier-free guidance (`job` has no "cfg") are refused rather than left
    out."""
    v = req.get("guidance")
    if v in (None, "", []):
        return []
    if isinstance(v, str):
        text = v.strip()
        try:
            v = json.loads(text) if text[:1] in "[{" else [x.strip() for x in text.split(",") if x.strip()]
        except ValueError:
            raise ImageRequestError("guidance: a JSON list of method names or {\"name\": ...} objects") from None
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, list):
        raise ImageRequestError("guidance: a list of method names or {\"name\": ...} objects")
    takes = guidance_methods(cfg)
    out: list[dict] = []
    for item in v:
        name: Any
        given: dict
        if isinstance(item, str):
            name, given = item, {}
        elif isinstance(item, dict):
            name, given = item.get("name"), {k: x for k, x in item.items() if k != "name"}
        else:
            name, given = None, {}
        if name not in GUIDANCE_METHODS:
            raise ImageRequestError("guidance: methods " + ", ".join(GUIDANCE_METHODS) + f" (not {name!r})")
        if name not in takes:
            raise ImageRequestError(f"guidance {name}: this image model takes none (Qwen-Image-2.1 and Mage-Flow do)")
        if any(m["name"] == name for m in out):
            raise ImageRequestError(f"guidance {name}: named twice")
        if name in GUIDANCE_NEEDS_CFG and not job.get("cfg"):
            raise ImageRequestError(f"guidance {name}: needs classifier-free guidance (true_cfg_scale above 1"
                                    + (" with a negative_prompt" if not cfg.get("family") else "") + ")")
        spec = guidance_defaults(cfg)[name]
        unknown = sorted(set(given) - set(spec))
        if unknown:
            raise ImageRequestError(f"guidance {name}: no parameter {unknown[0]!r} (it takes "
                                    + (", ".join(spec) or "none") + ")")
        m: dict = {"name": name}
        for key, (dflt, lo, hi) in spec.items():
            if key == "layers":
                layers = given.get(key, dflt)
                if isinstance(layers, int) and not isinstance(layers, bool):
                    layers = [layers]
                if (not isinstance(layers, list) or not layers or len(layers) > 64
                        or any(isinstance(b, bool) or not isinstance(b, int) or not lo <= b <= hi for b in layers)):
                    raise ImageRequestError(f"guidance {name}.layers: a list of the model's block numbers, from 0")
                m[key] = sorted(set(layers))
            elif isinstance(dflt, bool):
                b = given.get(key, dflt)
                if not isinstance(b, bool):
                    raise ImageRequestError(f"guidance {name}.{key}: true or false")
                m[key] = b
            else:
                try:
                    m[key] = _number(given, key, lo, hi, dflt)
                except ImageRequestError as e:
                    raise ImageRequestError(f"guidance {name}.{e}") from None
        if m.get("start", 0.0) > m.get("end", 1.0) and name == "skip_layer_guidance":
            raise ImageRequestError(f"guidance {name}: start at most end")
        out.append(m)
    return out


def percent_sigma(cfg: dict, width: int, height: int, percent: float) -> float:
    """The model's sigma at a share of its schedule (skip layer guidance's window): Qwen-Image-2.1's exponential
    shift by its size's mu (base_sigmas', without the terminal stretch), Mage-Flow's static shift."""
    t = min(max(1.0 - percent, 0.0), 1.0)   # the schedule's time, 1 at its start
    if t in (0.0, 1.0):
        return t
    if _pipeline(cfg.get("family")) == "flow":
        return MF_SHIFT * t / (1.0 + (MF_SHIFT - 1.0) * t)
    seq = (width // SIZE_STEP) * (height // SIZE_STEP)
    mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (seq - 256)
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def guidance_words(job: dict, cfg: dict, width: int, height: int) -> str:
    """The engine's words of a job's guidance (strata_image's CFGNORM:, CFGZERO, APG:, DSLG: and DSLGP:, DD:)."""
    words = ""
    for m in job.get("guide") or ():
        name = m["name"]
        if name == "cfg_norm":
            words += f" CFGNORM:{m['strength']!r}"
        elif name == "cfg_zero_star":
            words += " CFGZERO"
        elif name == "apg":
            words += f" APG:{m['eta']!r},{m['norm_threshold']!r},{m['momentum']!r}"
        elif name == "skip_layer_guidance":
            words += (f" DSLG:{','.join(str(b) for b in m['layers'])} DSLGP:{m['scale']!r},"
                      f"{percent_sigma(cfg, width, height, m['start'])!r},"
                      f"{percent_sigma(cfg, width, height, m['end'])!r},{m['rescaling_scale']!r}")
        elif name == "detail_daemon":
            keys = ("detail_amount", "start", "end", "bias", "exponent", "start_offset", "end_offset", "fade")
            words += (" DD:" + ",".join(repr(m[k]) for k in keys)
                      + f",{1 if m['smooth'] else 0},{m['cfg_scale_override']!r}")
    return words


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


def decode_inline_image(value: str, strict: bool) -> bytes:
    """Decode inline image data, preserving each caller's data URL and validation rules."""
    if value.startswith("data:"):
        value = value.split(",", 1)[-1 if strict else 1]
    return base64.b64decode(value, validate=strict)


def png_has_alpha(data: bytes) -> bool:
    """An alpha channel by the PNG's IHDR colour type (4 grey + alpha, 6 RGBA); False for anything unreadable."""
    return len(data) > 25 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and data[25] in (4, 6)


def picture_background(png: bytes, fmt: str, opaque: bool) -> str:
    """What the picture that went out is: "opaque" when the format has no alpha channel or the alpha was put on white,
    "transparent" when it kept one (background auto with an RGBA PNG, or a flatten that failed).  Upstream 13e7188cc."""
    if opaque or fmt in ("jpeg", "jpg"):
        return "opaque"
    return "transparent" if png_has_alpha(png) else "opaque"


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
    return encode_pillow(Image.open(io.BytesIO(png)), fmt, compression, opaque, flatten_jpeg=True)


def encode_pillow(im, fmt: str, compression: int | None = None, opaque: bool = False,
                  flatten_jpeg: bool = False) -> bytes:
    """Encode a Pillow image; absent compression keeps the region API's lossless WebP and JPEG quality 95."""
    import io
    from PIL import Image
    if (fmt == "jpeg" and (flatten_jpeg or im.mode in ("RGBA", "LA"))) or opaque:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    out = io.BytesIO()
    if fmt == "jxl":
        im.save(out, format="PNG")
        return encode_jxl(out.getvalue(), 100 if compression is None else compression)
    if fmt == "jpeg":
        im = im.convert("L" if im.mode == "L" else "RGB")
    options: dict[str, Any] = {}
    if fmt == "webp" and compression is None:
        options["lossless"] = True
    elif fmt == "jpeg" or fmt == "webp":
        options["quality"] = 95 if compression is None else compression
    im.save(out, format=fmt.upper(), **options)
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
    """Qwen-Image-2.1's base schedule, the engine's own now (src/image/sigmas.hpp's qi_base_sigmas): this copy
    remains only for the Turbo redraw's explicit SIGMAS: override (fix_job), which is made at the tile's size.
    diffusers' QwenImage21Pipeline with the FlowMatchEulerDiscreteScheduler its scheduler_config sets up
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


def parse_upscale(req: dict, image: bytes, cfg: dict) -> dict:
    """An upscale (the edits' `upscale`, the chat's upscale_image, /v1/images/upscale) in two passes: `image` made
    larger by `scale` (1 to UPSCALE_MAX_SCALE; the sides multiples of SIZE_STEP) on the GPU by `method`
    (UPSCALE_METHODS: Lanczos, or an upscaler model whose weights image_aux fetches; job["upscaler"]) with its color
    correction (`color`, UPSCALE_COLORS), then with `redraw` (asked for: not the default) drawn again from it at
    `strength` (UPSCALE_STRENGTH), an img2img of the whole picture (job["init"], job["upscale"]); past the model's
    max_side, drawn again a tile at a time (job["tiles"]: UPSCALE_TILE pixels, UPSCALE_PAD of context around each,
    UPSCALE_BLUR of blend), UPSCALE_MAX_SIDE a side at most.  Without redraw, the first pass alone (job["ups_only"]: no
    image model run, one image, the prompt not needed)."""
    v = req["upscale"]
    if isinstance(v, str):   # a form's field
        try:
            v = json.loads(v)
        except ValueError:
            v = None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = {"scale": v}
    if not isinstance(v, dict) or not set(v) <= {"scale", "strength", "method", "redraw", "color", "variant"}:
        raise ImageRequestError('upscale: {"scale": 1.5 or 2, "method": "auto", "redraw": true, "strength": 0.35, '
                                '"color": "lab", "variant": "7b"} (or the scale alone)')
    try:
        scale = float(v.get("scale", 2))
    except (TypeError, ValueError):
        raise ImageRequestError("upscale.scale: a number") from None
    if not 1.0 < scale <= UPSCALE_MAX_SCALE:
        raise ImageRequestError(f"upscale.scale: above 1, at most {UPSCALE_MAX_SCALE:g}")
    method = v.get("method") or "auto"
    if method not in UPSCALE_METHODS:
        raise ImageRequestError(f"upscale.method: one of {', '.join(UPSCALE_METHODS)}, not {method!r}")
    color = v.get("color") or None
    if color is not None and color not in UPSCALE_COLORS:
        raise ImageRequestError(f"upscale.color: one of {', '.join(UPSCALE_COLORS)}, not {color!r}")
    variant = v.get("variant") or None
    if variant is not None and variant not in SEEDVR2_VARIANTS:
        raise ImageRequestError(f"upscale.variant: SeedVR2's {', '.join(SEEDVR2_VARIANTS)}, not {variant!r}")
    redraw = parse_flag(v.get("redraw"), "upscale.redraw", False)
    if not redraw and v.get("strength") is not None:
        raise ImageRequestError("upscale.strength: only with \"redraw\": true")
    w0, h0 = image_dims(image)
    w, h = (max(SIZE_STEP, round(x * scale / SIZE_STEP) * SIZE_STEP) for x in (w0, h0))
    if max(w, h) > UPSCALE_MAX_SIDE:
        raise ImageRequestError(f"upscale: {w}x{h} passes {UPSCALE_MAX_SIDE} pixels a side")
    resolved = method if method != "auto" else "realesrgan" if redraw else "seedvr2"
    if variant is not None and resolved != "seedvr2":
        raise ImageRequestError("upscale.variant: SeedVR2's alone")
    upscaler = upscaler_of(resolved, method, color, variant)
    if not redraw:
        job = ups_only_job({**req, "prompt": str(req.get("prompt") or "").strip() or "upscale"}, cfg, w, h, upscaler)
        job["init"] = image
        return job
    side = int(cfg.get("max_side") or 2048)
    tiled = max(w, h) > side
    job = parse_request({**req, "size": f"{min(w, side)}x{min(h, side)}"}, cfg)
    job["width"], job["height"] = w, h
    job["init"] = image
    job["upscale"] = True
    job["upscaler"] = upscaler
    if tiled:
        job["tiles"] = {"tile": UPSCALE_TILE, "pad": UPSCALE_PAD, "blur": UPSCALE_BLUR}
    strength = parse_strength({"strength": v.get("strength", UPSCALE_STRENGTH)})
    run = strength_run(job["steps"], cast(float, strength))
    if run < 1:
        raise ImageRequestError(f"upscale.strength: {strength:g} runs none of the {job['steps']} steps")
    job["strength"], job["steps_run"] = strength, run
    if cfg.get("schedule") == "turbo":
        # the Turbo's own sigmas start a strength of 0.35 high (its schedule is a picture's from noise): the redraw
        # came out another picture (another face, a bow for a tie, 2026-11-05); the base model's, as fix_job takes
        job["sigmas"] = base_sigmas(job["steps"], *made_size(job))
    return job


def upscaler_of(method: str, asked: str, color: str | None = None, variant: str | None = None, **more) -> dict:
    """An upscaler as the jobs carry it (job["upscaler"], job["crop_ups"]; the engine's UPS words): the method run
    (UPSCALE_METHODS but auto), the one asked for (auto: upscaler_weights may take RealESRGAN for a SeedVR2 that does
    not fit), its color correction (None: the method's own) and SeedVR2's variant (7b where none is asked)."""
    return {"method": method, "asked": asked, "color": color,
            **({"variant": variant or "7b"} if method == "seedvr2" else {}), **more}


def ups_only_job(req: dict, cfg: dict, w: int, h: int, upscaler: dict) -> dict:
    """An upscale without a drawing (an image's, parse_upscale's; a video's, videos.upscale_request): parse_request's
    base fields of `req` (its prompt, seed, answer's format) at the model's size at most (max_side: no image model
    runs, the upscaler makes w x h), one output -> the job with job["ups_only"] and `upscaler` (upscaler_of)."""
    side = int(cfg.get("max_side") or 2048)
    job = parse_request({**req, "n": 1, "size": f"{min(w, side)}x{min(h, side)}"}, cfg)
    job.pop("bucket", None)
    job.update(width=w, height=h, ups_only=True, upscaler=upscaler, n=1)
    return job


def upscale_fields(fields: dict) -> dict:
    """The upscale entries' shared fields, an image's (/v1/images/upscale, LocalAI's) and a video's (/v1/videos,
    vLLM-Omni's): extra_params (a JSON object, or a form's text of one; none or blank: {}), model (UPSCALE_MODELS' ids
    or a method's name; none: auto) and extra_params' color_correction_method (vLLM-Omni's SeedVR2's: UPSCALE_COLORS;
    none or blank: the upscaler's own) -> {"extra", "model", "method", "variant", "color"}; else an
    ImageRequestError."""
    extra = fields.get("extra_params") or {}
    if isinstance(extra, str):   # a form's field
        try:
            extra = json.loads(extra) if extra.strip() else {}
        except ValueError:
            raise ImageRequestError("extra_params: a JSON object") from None
    if not isinstance(extra, dict):
        raise ImageRequestError("extra_params: a JSON object")
    model = str(fields.get("model") or "auto")
    method, variant = UPSCALE_MODELS.get(model, (model, None))
    if method not in UPSCALE_METHODS:
        raise ImageRequestError(f"model: an upscaler, one of {', '.join(['auto', *UPSCALE_MODELS])}, not {model!r}")
    color = extra.get("color_correction_method") or None
    if color is not None and color not in UPSCALE_COLORS:
        raise ImageRequestError(f"color_correction_method: one of {', '.join(UPSCALE_COLORS)}, not {color!r}")
    return {"extra": extra, "model": model, "method": method, "variant": variant, "color": color}


def upscale_request(fields: dict) -> tuple[dict, str | None]:
    """POST /v1/images/upscale's fields (LocalAI's: model, scale; OpenAI's and vLLM-Omni's response_format,
    output_format, output_compression, seed; prompt for a redraw; extra_params: vLLM-Omni's color_correction_method,
    and redraw, redraw_strength and redraw_model, XeStrata's) -> (the edits' request with its `upscale`, the image model
    a redraw names or None).  `model` is an upscaler (UPSCALE_MODELS' ids, or a method's name; none: auto); the scale
    its own where it has one (RealESRGAN x4plus's 4), else 2; no redraw unless extra_params asks for one."""
    u = upscale_fields(fields)
    extra, method = u["extra"], u["method"]
    # the range parse_upscale takes (above 1 it checks); 0 or none: the default, as it has been
    scale = parse_number(fields.get("scale") or None, "scale", 1, UPSCALE_MAX_SCALE,
                         4 if method == "realesrgan" else 2)
    up: dict = {"method": method, "scale": scale, "redraw": extra.get("redraw", False),
                **({"variant": u["variant"]} if u["variant"] else {}), **({"color": u["color"]} if u["color"] else {})}
    if extra.get("redraw_strength") is not None:
        up["strength"] = extra["redraw_strength"]
    req = {k: fields[k] for k in ("prompt", "negative_prompt", "response_format", "output_format",
                                  "output_compression", "seed", "steps", "num_inference_steps") if k in fields}
    req["upscale"] = up
    model_name = extra.get("redraw_model")
    return req, model_name if isinstance(model_name, str) and model_name else None


def refs_limit(cfg: dict) -> int:
    """The images an edit of the model takes at most, the one it edits among them: the config's max_refs, else its
    family's own limit."""
    caps = _qimage().family_capabilities(cfg.get("family"))
    return int(cfg.get("max_refs") or caps.get("max_refs", MAX_REFS))


def parse_edit_request(req: dict, images: list[bytes], cfg: dict, mask: bytes | None = None) -> dict:
    """An edit (/v1/images/edits): parse_request's fields with `images` (PNG, JPEG, GIF or BMP).  A model that edits
    takes them as its reference images (job["refs"]); one that does not (can_edit) takes one image as img2img's base
    image (job["init"]).  size "auto" (or none): the first image's shape at about 1024 x 1024 pixels, its sides
    multiples of SIZE_STEP.  `mask` (OpenAI's): a PNG of the first image's size whose transparent pixels are where it
    changes, the rest kept as it is, as job["mask"].  With a mask and the size left to it (none, "auto" or the first
    image's own), the part is cut out and put back (crop_wanted; not with MOSS-Image's layout): job["crop"] its plan
    (plan_crop), the image the first image's size (job["width"], job["height"]), drawn at the plan's (made_size); a
    mask that changes nothing is job["crop"] {"empty": True} (the picture as it is); a MOSS-Image edit with references
    besides the picture is drawn at the whole picture's plan, and with the request's refine its part cut out of that
    and drawn again (job["refine"], CROP_REFINE_STRENGTH).  strength (parse_strength): how much of the schedule runs
    from the first image (job["strength"] and the steps it runs, job["steps_run"], when below 1); an edit's default 1,
    an img2img's img2img_strength of its steps, or 1 with a mask."""
    mage = _pipeline(cfg.get("family")) == "flow"
    hunyuan = _pipeline(cfg.get("family")) == "guided_token"
    moss = _pipeline(cfg.get("family")) == "latent_chat"
    edits = can_edit(cfg)
    strength = parse_strength(req)
    if not images:
        raise ImageRequestError("image: one reference image or more" if edits else "image: the image to start from")
    if edits:
        limit = refs_limit(cfg)
        if len(images) > limit:
            raise ImageRequestError(f"image: at most {limit} reference images")
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
    if req.get("upscale") is not None:   # the first image drawn again larger (parse_upscale)
        if mask is not None or req.get("extend") is not None or req.get("fix") is not None or len(images) > 1:
            raise ImageRequestError("upscale: one image, without a mask, extend or fix")
        return parse_upscale(req, images[0], cfg)
    if req.get("fix") is not None:   # the parts' fix: parts found and drawn again one by one (fix_images)
        if mask is not None or req.get("extend") is not None or len(images) > 1:
            raise ImageRequestError("fix: one image, without a mask or extend")
        fix = parse_fix(req["fix"])
        w0, h0 = dims[0]
        if not str(req.get("prompt") or "").strip():   # each part's own words
            req = {**req, "prompt": ", ".join(FIX_PROMPTS[p] for p in fix["parts"])}
        gw, gh = crop_gen_size(cfg, w0, h0)
        job = parse_request({**req, "size": f"{gw}x{gh}"}, cfg)
        job.update(fix=fix, picture=images[0], width=w0, height=h0,
                   fix_req={k: v for k, v in req.items() if k not in ("fix", "n", "size", "strength", "mask")})
        return job
    if req.get("extend") is not None:   # an outpaint: the canvas grown, its new area the mask
        if mask is not None:
            raise ImageRequestError("extend: not with a mask (extend the image, then edit the result with one)")
        pads = parse_extend(req["extend"], *dims[0], int(cfg.get("max_side") or 2048))
        images = list(images)
        images[0], mask = extend_canvas(images[0], pads)
        dims[0] = (dims[0][0] + pads[0] + pads[2], dims[0][1] + pads[1] + pads[3])
    if mask is not None:
        mask = check_mask(mask, dims[0])
    crop = None
    w0, h0 = dims[0]
    if (mask is not None and req.get("size") in (None, "auto", f"{w0}x{h0}") and req.get("layout") is None
            and crop_wanted(req, cfg)):
        crop = plan_crop(images[0], mask, cfg, CROP_CONTEXT_REFS if edits and len(images) > 1 else CROP_CONTEXT)
        if crop is not None and not crop.get("empty"):
            req = {**req, "size": "{}x{}".format(*crop["gen"])}
    if ((crop is None or crop.get("empty")) and req.get("size") in (None, "auto")
            and not (hunyuan and cfg.get("ratio_head"))):   # else the model chooses
        side = int(cfg.get("max_side") or 2048)
        tw = (1024 * 1024 * w0 / h0) ** 0.5
        th = tw * h0 / w0
        tw, th = (min(side, max(MIN_SIDE, round(v / SIZE_STEP) * SIZE_STEP)) for v in (tw, th))
        req = {**req, "size": f"{tw}x{th}"}
    # Mage-Flow's edit model has its own steps; MOSS-Image's edits their steps and guidance (MOSS_EDIT_STEPS)
    if mage and edits:
        job = parse_request(req, {**cfg, "steps": cfg.get("edit_steps") or cfg.get("steps")})
    elif moss and edits:   # Moss-Image-Edit's steps and guidance; the other layouts' are MOSS-Image's own
        if req.get("edit_adapter") not in (None, ""):   # the edit adapter the request names (moss_pick_adapter)
            a = moss_pick_adapter(cfg, str(req["edit_adapter"]))
            cfg = {**cfg, "edit_adapter": str(req["edit_adapter"]), "edit_lora": a["lora"], "edit_format": a["format"]}
        dflt = model_defaults(cfg)
        fmt = moss_edit_format(cfg)
        job = parse_request(req, {**cfg, "steps": dflt["edit_steps"], "cfg_scale": dflt["edit_guidance"]}
                            if fmt == "moss-image-edit" else cfg)
        job["edit_format"] = fmt
    else:
        if req.get("edit_adapter") not in (None, ""):
            raise ImageRequestError("edit_adapter: MOSS-Image's edits' alone"
                                    + ("" if moss else f" ({cfg.get('model_name') or 'this model'} is not MOSS-Image)"))
        job = parse_request(req, cfg)
    if edits:
        job["refs"] = list(images)
        # the references at about 1 MP (the cut's first, the picture the engine cuts from, as it is: an
        # ai-toolkit edit's cut again at the references' size beside the drawing's base, _crop_ref)
        if moss:
            keep = 1 if crop is not None and not crop.get("empty") else 0
            if job["edit_format"] == "moss-image-edit":
                job["refs"] = job["refs"][:keep] + [moss_ref(r) for r in job["refs"][keep:]]
            else:
                job["refs"] = job["refs"][:keep] + [moss_ref(r, MOSS_AITK_REF_PIXELS, grow=False)
                                                    for r in job["refs"][keep:]]
            job["edit_lora"] = cfg["edit_lora"]
    else:
        job["init"] = images[0]
        if strength is None:
            strength = img2img_strength(job["steps"])
    if mask is not None:
        job["mask"] = mask
    if crop is not None:   # the image at the first image's size, drawn at the plan's
        job["crop"] = crop
        crop_upscaler(job, req)
        job["color_match"] = color_match_mode(req, cfg)
        job["width"], job["height"] = w0, h0
        if moss and edits and len(images) > 1 and not crop.get("empty") and not crop["whole"]:   # CROP_REFINE_STRENGTH
            job["crop"] = crop_whole(cfg, w0, h0, crop["blend"])
            if parse_flag(req.get("refine"), "refine"):
                job["refine"] = {"crop": crop, "strength": CROP_REFINE_STRENGTH,
                                 "steps_run": strength_run(job["steps"], CROP_REFINE_STRENGTH)}
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


class AuxPending(ImageJobFailed):
    """A tool's weights are not here yet: their download has started (or goes on, or failed).  `downloads`:
    image_aux's status of each."""

    def __init__(self, downloads: list[dict]):
        self.downloads = downloads
        super().__init__(aux_words(downloads))


class LongJobBusy(ImageJobFailed):
    """A request for the image engine while a long job (a video's, serve/videos.py) has it: refused at once rather
    than queued for the job's length (a 503 with Retry-After).  `retry_after`: the job's seconds left, as far as its
    frames tell."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


def job_left(state: dict | None) -> int | None:
    """The seconds a long job's state (videos.upscale's: its start, frames done and frames) says it has left; None
    for none."""
    if not state:
        return None
    done, total = state["done"], max(state["total"], state["done"])
    left = (time.monotonic() - state["started"]) / done * (total - done) if done else 600
    return int(min(max(left, 5), 86400))


def long_job_left(svc) -> int | None:
    """The seconds a long job (svc.long_job: a video's) still holds the image engine; None when none runs."""
    return job_left(getattr(svc, "long_job", None))


def refuse_during_long_job(svc) -> None:
    """LongJobBusy while a long job holds the image engine (another image job would wait for all of it)."""
    left = long_job_left(svc)
    if left is not None:
        raise LongJobBusy(f"a video job has the image engine for about {left} s more: ask again then", left)


class EngineLock:
    """ImageGen's lock (one request line at a time): one that a long job holds (gen.long_job) refuses at once
    (LongJobBusy) rather than keeping the caller for the job's length."""

    def __init__(self, gen: ImageGen):
        self.gen = gen
        self.lock = threading.Lock()

    def __enter__(self) -> EngineLock:
        while not self.lock.acquire(timeout=0.25):
            state = self.gen.long_job
            left = job_left(state) if state and state.get("thread") != threading.get_ident() else None
            if left is not None:
                raise LongJobBusy(f"a video job has the image engine for about {left} s more: ask again then", left)
        return self

    def __exit__(self, *exc) -> None:
        self.lock.release()


def aux_words(downloads: list[dict]) -> str:
    """What a tool's result says of its weights' downloads."""
    out = []
    for d in downloads:
        mib = f"{d.get('bytes', 0) / 2**20:.0f} of {d.get('total', 0) / 2**20:.0f} MiB"
        if d.get("status") == "failed":
            out.append(f"downloading {d['name']} failed: {d.get('error')}")
        else:
            out.append(f"{d['name']} is downloading ({mib}; {d.get('license')}; {d.get('source')})")
    failed = any(d.get("status") == "failed" for d in downloads)
    return "; ".join(out) + ("" if failed else f". Ask again in about {image_aux.RETRY_S} s, in a new request, once it "
                                               "is done; the image is not made until then")


def parse_fix(v) -> dict:
    """The edits' `fix` (the chat's fix_image): {"parts": ["face", "hands", "person"], "strength": 0.5}, or the parts
    alone (a list, or a name) -> {"parts", "strength"}."""
    if isinstance(v, str):   # a form's field: JSON, or a part's name
        try:
            v = json.loads(v)
        except ValueError:
            pass
    if isinstance(v, str):
        v = [v]
    if isinstance(v, list):
        v = {"parts": v}
    if not isinstance(v, dict) or not set(v) <= {"parts", "strength"}:
        raise ImageRequestError('fix: {"parts": ["face", "hands", "person"], "strength": 0.5} (or the parts alone)')
    parts = v.get("parts") or ["face"]
    if isinstance(parts, str):
        parts = [parts]
    if not isinstance(parts, list) or not parts or any(p not in FIX_PARTS for p in parts):
        raise ImageRequestError(f"fix.parts: some of {', '.join(FIX_PARTS)}")
    strength = parse_strength({"strength": v.get("strength", FIX_STRENGTH)})
    return {"parts": list(dict.fromkeys(parts)), "strength": strength}


def fix_crop(box: tuple[float, float, float, float], W: int, H: int, cfg: dict) -> dict:
    """The part of the picture a detection is redrawn in: its box with FIX_MARGIN of its longer side around it on
    every side, in whole pixels, moved the least that puts it inside the picture (all of a side it is longer than),
    then lengthened to the drawing's shape (_crop_to_shape) -> a plan_crop plan, drawn at crop_gen_size (about 1 MP)."""
    margin = FIX_MARGIN * max(box[2] - box[0], box[3] - box[1])

    def span(lo: float, hi: float, side: int) -> tuple[int, int]:
        size = min(side, math.ceil(hi - lo + 2 * margin))
        first = min(max(math.floor(lo - margin), 0), side - size)
        return first, first + size
    ax, bx = span(box[0], box[2], W)
    ay, by = span(box[1], box[3], H)
    gw, gh = crop_gen_size(cfg, bx - ax, by - ay)
    x, y, w, h = _crop_to_shape(ax, ay, bx - ax, by - ay, W, H, gw, gh)
    return {"box": [x, y, w, h], "gen": [gw, gh], "blend": FIX_BLEND, "size": [W, H], "whole": False}


def box_mask(W: int, H: int, box: tuple[int, int, int, int]) -> bytes:
    """OpenAI's mask of a W x H picture that changes inside `box` (x0, y0, x1, y1)."""
    import io
    from PIL import Image, ImageDraw
    m = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    ImageDraw.Draw(m).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=(0, 0, 0, 0))
    buf = io.BytesIO()
    m.save(buf, "PNG")
    return buf.getvalue()


def grow_mask(mask: Any, g: int) -> Any:
    """A boolean mask grown by g pixels on every side: set wherever a set pixel is within g of it along a row, then a
    column (a square of 2 g + 1)."""
    import numpy as np
    out = mask.astype(bool)
    for axis in (0, 1):
        n = out.shape[axis]
        # the running count of set pixels along the axis: a window's count is the difference of two of them
        run = np.concatenate([np.zeros_like(out.sum(axis=axis, keepdims=True), dtype=np.int64),
                              np.cumsum(out, axis=axis, dtype=np.int64)], axis=axis)
        hi = np.minimum(np.arange(n) + g + 1, n)
        lo = np.maximum(np.arange(n) - g, 0)
        out = (np.take(run, hi, axis=axis) - np.take(run, lo, axis=axis)) > 0
    return out


def fix_regions(found: list[dict], W: int, H: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    """The detections (each {"part", "box": [x0, y0, x1, y1], "score", "mask": a bool array or None}) as the regions
    the parts' fix draws, largest first (the same area by the parts' order, then the box's place), each its crop and
    its mask (OpenAI's: transparent where it changes: the box, or a person's mask, grown as FIX_GROW says) ->
    (regions, the ones left out with the reason: smaller than FIX_MIN, or FIX_GUIDE or larger)."""
    import io
    import numpy as np
    from PIL import Image
    order = list(FIX_PARTS)
    keep, left = [], []
    for d in found:
        x0, y0, x1, y1 = d["box"]
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) < FIX_MIN:
            left.append({**{k: d[k] for k in ("part", "box", "score")}, "skipped": f"smaller than {FIX_MIN} px"})
        elif min(bw, bh) >= FIX_GUIDE:
            left.append({**{k: d[k] for k in ("part", "box", "score")}, "skipped": f"larger than {FIX_GUIDE} px"})
        else:
            keep.append(d)
    keep.sort(key=lambda d: (-(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), order.index(d["part"]),
                             d["box"][1], d["box"][0]))
    regions = []
    for d in keep:
        x0, y0, x1, y1 = d["box"]
        g = max(FIX_GROW_MIN, round(FIX_GROW * max(x1 - x0, y1 - y0)))   # the same on every side
        if d.get("mask") is not None:
            edit = grow_mask(d["mask"], g)
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., 3] = np.where(edit, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buf, "PNG")
            mask = buf.getvalue()
        else:
            mask = box_mask(W, H, (max(0, math.floor(x0) - g), max(0, math.floor(y0) - g), min(W, math.ceil(x1) + g),
                                   min(H, math.ceil(y1) + g)))
        regions.append({"part": d["part"], "box": [round(v, 1) for v in d["box"]], "score": round(d["score"], 4),
                        "crop": fix_crop(d["box"], W, H, cfg), "mask": mask})
    return regions, left


def crop_share(mask: bytes) -> Any:
    """An OpenAI mask (a PNG, transparent where the picture changes) as its edit share: float32 [H][W], 1 - alpha."""
    import io
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(mask)).convert("RGBA"), dtype=np.float32)[..., 3]
    return 1.0 - a / 255.0


def crop_reach(blend: int) -> int:
    """How far past the mask's box the engine's mask reaches (img_crop.hpp, CropStitch::prepare): the disc it grows by
    and its three box means' radii, blend // 8 each; its hole fill stays inside the box.  The box planned on the server
    holds every pixel at which the engine's mask is above 0."""
    return 4 * (blend // 8)


def crop_gen_size(cfg: dict, w: float, h: float) -> tuple[int, int]:
    """The size the model draws a box of w x h at: HunyuanImage 3.0's bucket nearest its shape; the others' about
    1024 x 1024 pixels in its shape, sides multiples of 32 that an edit's reference keeps as they are (ref_size's
    fixed point: the reference and the latent the same size), within max_side."""
    if _pipeline(cfg.get("family")) == "guided_token":
        b = hunyuanimage3_bucket(cfg, max(1, round(w)), max(1, round(h)))
        return int(b["width"]), int(b["height"])
    side = int(cfg.get("max_side") or 2048)
    ratio = w / h
    tw0 = (1024 * 1024 * ratio) ** 0.5
    best, score = None, None
    for i in range(-3, 4):
        for j in range(-3, 4):
            gw, gh = round(tw0 / 32) * 32 + 32 * i, round(tw0 / ratio / 32) * 32 + 32 * j
            if not (MIN_SIDE <= gw <= side and MIN_SIDE <= gh <= side) or ref_size(gw, gh) != (gw, gh):
                continue
            s = (abs(math.log(gw / gh / ratio)), abs(gw * gh - 1024 * 1024))
            if score is None or s < score:
                best, score = (gw, gh), s
    if best is None:   # no fixed point near (a shape past the sides' bounds): the nearest in bounds
        best = (min(side, max(MIN_SIDE, round(tw0 / 32) * 32)), min(side, max(MIN_SIDE, round(tw0 / ratio / 32) * 32)))
    return best


def _crop_to_shape(x: int, y: int, w: int, h: int, W: int, H: int, gw: int, gh: int) -> tuple[int, int, int, int]:
    """The box made the drawing's shape (gw : gh) by lengthening its short side about its centre, then moved the least
    that puts it inside the picture; a box longer than the picture is centred on it (reaching past both edges)."""
    def place(start: int, length: int, new: int, side: int) -> int:
        at = start + (length - new) / 2
        return round(min(max(at, 0), side - new)) if new <= side else round((side - new) / 2)
    if w * gh < h * gw:   # too narrow
        nw = max(w, round(h * gw / gh))
        return place(x, w, nw, W), y, nw, h
    nh = max(h, round(w * gh / gw))
    return x, place(y, h, nh, H), w, nh


def plan_crop(image: bytes, mask: bytes, cfg: dict, context: float = CROP_CONTEXT,
              blend: int = CROP_BLEND) -> dict | None:
    """The cut's plan of an edit with a mask (img_crop.hpp, the engine's part, which fills the mask's holes): the box
    of the mask's values from CROP_FAINT on, grown by the engine's reach (crop_reach), by `context` of its size on each
    side and to the drawing's shape (crop_gen_size, _crop_to_shape).  A
    box the drawing would not enlarge is the whole picture's (drawn at the model's size and put back at the picture's,
    its pixels outside the mask kept).  -> {"box": [x, y, w, h], "gen": [gw, gh], "blend", "size": [W, H], "whole":
    the whole picture}; {"empty": True} for a mask that changes nothing; None for one that changes every pixel (the
    edit as it was)."""
    import numpy as np
    W, H = image_dims(image)
    m = crop_share(mask)
    if m.shape != (H, W):
        raise ImageRequestError(f"mask: {m.shape[1]}x{m.shape[0]}, not the image's {W}x{H}")
    rows, cols = np.nonzero(np.any(m >= CROP_FAINT, axis=1))[0], np.nonzero(np.any(m >= CROP_FAINT, axis=0))[0]
    if rows.size == 0:
        return {"empty": True}
    if bool(np.all(m >= 1.0)):
        return None
    reach = crop_reach(blend)
    x0, y0 = max(0, int(cols[0]) - reach), max(0, int(rows[0]) - reach)
    x1, y1 = min(W, int(cols[-1]) + 1 + reach), min(H, int(rows[-1]) + 1 + reach)
    margin = math.ceil(context * math.sqrt((x1 - x0) * (y1 - y0)))   # the same on every side
    x0, y0, x1, y1 = max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin)
    gw, gh = crop_gen_size(cfg, x1 - x0, y1 - y0)
    x, y, w, h = _crop_to_shape(x0, y0, x1 - x0, y1 - y0, W, H, gw, gh)
    if gw <= w:   # the drawing would not enlarge the box: the whole picture instead
        return crop_whole(cfg, W, H, blend)
    return {"box": [x, y, w, h], "gen": [gw, gh], "blend": blend, "size": [W, H], "whole": False}


def crop_upscaler(job: dict, req: dict) -> None:
    """A request's crop_upscale ("auto", "lanczos", "realesrgan", "seedvr2", or {"method", "variant"}: SeedVR2's 7b or
    7b-sharp), how the masked edit's cut is enlarged to the drawing's size -> job["crop_ups"] (an upscaler as
    job["upscaler"] is, and the enlargement) where it is an upscaler; Lanczos is the engine's own resize.  auto: an
    upscaler from CROP_UPS_AUTO times on.  A cut that is not enlarged is resized as it is."""
    v = req.get("crop_upscale")
    if isinstance(v, str) and v.strip().startswith("{"):   # a form's field
        try:
            v = json.loads(v)
        except ValueError:
            raise ImageRequestError("crop_upscale: a JSON object, or the method's name") from None
    if isinstance(v, str):
        v = {"method": v}
    if v is None:
        v = {}
    if not isinstance(v, dict) or not set(v) <= {"method", "variant"}:
        raise ImageRequestError('crop_upscale: "auto", "lanczos", "realesrgan" or "seedvr2" (or {"method", '
                                '"variant"})')
    method = v.get("method") or "auto"
    if method not in CROP_UPSCALE:
        raise ImageRequestError(f"crop_upscale: one of {', '.join(CROP_UPSCALE)}, not {method!r}")
    variant = v.get("variant") or None
    if variant is not None and variant not in SEEDVR2_VARIANTS[:2]:
        raise ImageRequestError(f"crop_upscale's variant: SeedVR2's {' or '.join(SEEDVR2_VARIANTS[:2])}")
    if variant is not None and method not in ("seedvr2", "auto"):
        raise ImageRequestError("crop_upscale's variant: SeedVR2's alone")
    if variant is not None:
        method = "seedvr2"
    crop = job.get("crop") or {}
    if crop.get("empty") or "gen" not in crop:
        return
    (gw, gh), (w, h) = crop["gen"], crop["box"][2:]
    scale = max(gw / max(1, w), gh / max(1, h))
    resolved = method if method != "auto" else "realesrgan" if scale >= CROP_UPS_AUTO else "lanczos"
    if resolved == "lanczos" or scale <= 1.0:
        return
    job["crop_ups"] = upscaler_of(resolved, method, None, variant, scale=round(scale, 2))


def crop_whole(cfg: dict, W: int, H: int, blend: int = CROP_BLEND) -> dict:
    """plan_crop's plan of the whole picture of W x H: drawn at the model's size in its shape and put back at its own,
    the part outside the mask kept."""
    gw, gh = crop_gen_size(cfg, W, H)
    return {"box": list(_crop_to_shape(0, 0, W, H, W, H, gw, gh)), "gen": [gw, gh], "blend": blend, "size": [W, H],
            "whole": True}


def crop_word(crop: dict, ref: tuple[int, int] | None = None) -> str:
    """The engine's CROP: word of a plan_crop plan; `ref` the first reference's size beside the drawing's base (the box
    cut at it: an ai-toolkit MOSS-Image edit's, moss_ref_size)."""
    x, y, w, h = crop["box"]
    word = f" CROP:{x},{y},{w},{h},{crop['blend']},{crop['size'][0]},{crop['size'][1]}"
    return word + (f",{ref[0]},{ref[1]}" if ref else "")


def parse_extend(v, w: int, h: int, side: int) -> tuple[int, int, int, int]:
    """An edit's `extend`, the canvas grown past the first image (w x h): {"left", "top", "right", "bottom"} in pixels,
    or {"aspect": "W:H"} (the canvas grown on the two sides the shape asks for, evenly) -> (left, top, right, bottom);
    ImageRequestError when it is not one of those, grows nothing, or passes `side` pixels a side."""
    if isinstance(v, str):   # a form's field: its JSON
        try:
            v = json.loads(v)
        except ValueError:
            v = None
    if not isinstance(v, dict):
        raise ImageRequestError('extend: {"left", "top", "right", "bottom"} in pixels, or {"aspect": "W:H"}')
    if "aspect" in v:
        if set(v) != {"aspect"}:
            raise ImageRequestError("extend: the aspect alone, or the pixels a side")
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", str(v["aspect"]))
        if not m or float(m.group(1)) <= 0 or float(m.group(2)) <= 0:
            raise ImageRequestError('extend.aspect: "W:H", e.g. "16:9"')
        a = float(m.group(1)) / float(m.group(2))
        nw, nh = max(w, round(h * a)), max(h, round(w / a))
        pads = ((nw - w) // 2, (nh - h) // 2, nw - w - (nw - w) // 2, nh - h - (nh - h) // 2)
    else:
        if not set(v) <= {"left", "top", "right", "bottom"}:
            raise ImageRequestError('extend: "left", "top", "right" and "bottom" in pixels')
        try:
            px = [int(v.get(k) or 0) for k in ("left", "top", "right", "bottom")]
        except (TypeError, ValueError):
            raise ImageRequestError("extend: whole pixels") from None
        pads = (px[0], px[1], px[2], px[3])
        if min(pads) < 0:
            raise ImageRequestError("extend: 0 or more pixels a side")
    if not any(pads):
        raise ImageRequestError("extend: the canvas grows on no side: give aspect (e.g. 16:9 wide, 9:16 tall) or the "
                                "pixels of a side (left, top, right, bottom)")
    if w + pads[0] + pads[2] > side or h + pads[1] + pads[3] > side:
        raise ImageRequestError(f"extend: the canvas would pass {side} pixels a side (max_side)")
    return pads


def extend_canvas(data: bytes, pads: tuple[int, int, int, int], feather: int = EXTEND_FEATHER) -> tuple[bytes, bytes]:
    """The first image grown by `pads` (left, top, right, bottom) for an outpaint -> (the canvas, the mask) as PNG.
    The new area repeats the picture's edge pixels (a base the model continues rather than a flat colour); the mask
    (OpenAI's alpha: 0 where the picture changes) is the new area, and inside the picture the change fades out from
    each side that grows over `feather` pixels (a third of the picture's shorter side where that is less), along a
    smoothstep of the distance."""
    import io
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
    h, w = im.shape[:2]
    left, t, r, b = pads
    canvas = np.pad(im, ((t, b), (left, r), (0, 0)), mode="edge")
    keep = np.ones(canvas.shape[:2], np.float32)   # 1 where the picture stays as it is
    keep[:t] = keep[t + h:] = 0
    keep[:, :left] = keep[:, left + w:] = 0
    feather = min(feather, min(h, w) // 3)
    if feather > 0:
        # each pixel's distance (from its centre) to the nearest side that grows, in rows and in columns
        def gap(n: int, before: int, after: int) -> np.ndarray:
            centre = np.arange(n, dtype=np.float32) + 0.5
            d = np.full(n, np.inf, np.float32)
            if before:
                d = np.minimum(d, centre)
            if after:
                d = np.minimum(d, n - centre)
            return d
        u = np.clip(np.minimum(gap(h, t, b)[:, None], gap(w, left, r)[None, :]) / feather, 0.0, 1.0)
        keep[t:t + h, left:left + w] = u * u * (3.0 - 2.0 * u)
    mask = np.zeros(canvas.shape, np.uint8)
    mask[..., 3] = np.round(keep * 255).astype(np.uint8)
    out = []
    for a in (canvas, mask):
        buf = io.BytesIO()
        Image.fromarray(a, "RGBA").save(buf, format="PNG", compress_level=1)
        out.append(buf.getvalue())
    return out[0], out[1]


def check_tool(cfg: dict, name: str) -> None:
    """ImageRequestError where the config's "image_tools" (every image tool by default) leaves `name` out: the Images
    API's operations are the chat's tools', and a tool turned off is off at every entry."""
    conf = cfg.get("image_tools")
    if conf is not None and name not in conf:
        raise ImageRequestError(f"{name} is turned off on this server (its config's images \"image_tools\")")


def color_match_mode(req: dict, cfg: dict) -> str:
    """The cut's color match as the drawing goes back (the engine's COLOR:): the request's color_match, else the
    config's, else COLOR_MATCH - "auto" by the part kept around the paint, "full" by the whole cut, "off"."""
    v = req.get("color_match")
    if v is None or v == "":
        v = cfg.get("color_match") or COLOR_MATCH
    if v not in ("auto", "full", "off"):
        raise ImageRequestError('color_match: "auto", "full" or "off"')
    return str(v)


def crop_wanted(req: dict, cfg: dict) -> bool:
    """Whether an edit with a mask is drawn on its part cut out and put back: the request's crop_edit, else the
    config's (on by default)."""
    v = parse_flag(req.get("crop_edit"), "crop_edit")
    return cfg.get("crop_edit") is not False if v is None else v


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
    if job.get("ups_only"):   # no image model run: no tokens
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "input_tokens_details": {"text_tokens": 0, "image_tokens": 0}}
    mw, mh = made_size(job)
    text = len(job.get("ids") or ())
    block = (mw // SIZE_STEP) * (mh // SIZE_STEP)
    if job.get("bucket") is not None:
        refs = sum(b["tokens_h"] * b["tokens_w"] + HUNYUANIMAGE3_VIT_SLOTS for b in job.get("ref_buckets") or ())
        text -= block + refs
    else:
        sizes = [ref_size(*image_dims(r)) for r in job.get("refs") or ()]
        if sizes and job.get("crop") and not job["crop"].get("empty"):   # the cut's first, at the drawing's
            sizes[0] = (mw, mh)
        refs = sum((w // SIZE_STEP) * (h // SIZE_STEP) for w, h in sizes)
        if job.get("init"):   # img2img's image, read at the made size
            refs += block
    out = job["n"] * block
    return {"input_tokens": text + refs, "output_tokens": out, "total_tokens": text + refs + out,
            "input_tokens_details": {"text_tokens": text, "image_tokens": refs}}


def made_size(job: dict) -> tuple[int, int]:
    """The size the engine makes for `job`: each side rounded up to a multiple of SIZE_STEP; the cut's, the
    plan's drawing (the engine puts it back into the picture, job["width"] x job["height"])."""
    if job.get("crop") and not job["crop"].get("empty"):
        return job["crop"]["gen"][0], job["crop"]["gen"][1]
    if job.get("tiles"):   # an upscale in tiles: a tile and its context (the engine's upscale_tiles)
        side = -(-(job["tiles"]["tile"] + 2 * job["tiles"]["pad"]) // SIZE_STEP) * SIZE_STEP
        return (min(side, -(-int(job["width"]) // SIZE_STEP) * SIZE_STEP),
                min(side, -(-int(job["height"]) // SIZE_STEP) * SIZE_STEP))

    def up(v) -> int:
        return -(-int(v) // SIZE_STEP) * SIZE_STEP
    return up(job["width"]), up(job["height"])


def as_png(data: bytes) -> bytes:
    """An image (PNG, JPEG, GIF or BMP) as an RGBA PNG, as png_rgba writes one."""
    import io
    from PIL import Image
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGBA").save(out, format="PNG", compress_level=1)
    return out.getvalue()


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
    # ai-toolkit edit pictures each put "Picture N: <pad>" before the prompt
    if _pipeline(family) == "latent_chat":
        head = MOSS_PROMPT_HEAD + "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
                                          for i in range(refs))
        text = head + prompt + MOSS_PROMPT_TAIL
        return tok.encode(text, parse_special=True, plain=[(len(head), len(head) + len(prompt))])
    if _pipeline(family) == "flow":
        head = MF_EDIT_HEAD + "".join(MF_EDIT_REF.format(i=i + 1) for i in range(refs)) if refs else MF_PROMPT_HEAD
    else:
        head = PROMPT_HEAD + "".join(REF_PROMPT.format(sep=" " if i else "", i=i + 1) for i in range(refs))
    text = head + prompt + PROMPT_TAIL
    return tok.encode(text, parse_special=True, plain=[(len(head), len(head) + len(prompt))])


def anima_weights(text: str) -> list[tuple[str, float]]:
    """The prompt's emphasis: the text's parts and their weights, "(words)" 1.1 times its
    surroundings' (nested, multiplied), "(words:2)" 2, "\\(" and "\\)" a plain parenthesis."""
    text = text.replace("\\)", "\0\1").replace("\\(", "\0\2")

    def parse(string: str) -> list[str]:
        out, cur, level = [], "", 0
        for ch in string:
            if ch == "(":
                if level == 0:
                    if cur:
                        out.append(cur)
                    cur = "("
                else:
                    cur += ch
                level += 1
            elif ch == ")":
                level -= 1
                if level == 0:
                    out.append(cur + ")")
                    cur = ""
                else:
                    cur += ch
            else:
                cur += ch
        if cur:
            out.append(cur)
        return out

    def weights(string: str, w: float) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for x in parse(string):
            weight = w
            if len(x) >= 2 and x[0] == "(" and x[-1] == ")":
                x = x[1:-1]
                at = x.rfind(":")
                weight *= 1.1
                if at > 0:
                    try:
                        weight = float(x[at + 1:])
                        x = x[:at]
                    except ValueError:
                        pass
                out += weights(x, weight)
            else:
                out.append((x, w))
        return out

    return [(t.replace("\0\1", ")").replace("\0\2", "("), w) for t, w in weights(text, 1.0)]


def anima_tokens(tok, t5, text: str) -> tuple[list[int], list[int], list[float]]:
    """Anima's prompt tokenized a part at a time: each part (anima_weights) tokenized on its own, the Qwen3
    ids (their weights all 1: the text encoder reads them plainly; one <|endoftext|> for an empty prompt) and the T5
    ids with the part's weight, then T5's </s>."""
    ids: list[int] = []
    t5_ids: list[int] = []
    t5_w: list[float] = []
    for part, w in anima_weights(text):
        if not part:
            continue
        ids += tok.encode(part)
        more = t5.encode(part)
        t5_ids += more
        t5_w += [w] * len(more)
    t5_ids.append(T5_END)
    t5_w.append(1.0)
    return ids or [ANIMA_PAD], t5_ids, t5_w


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


def _tokenizer_module():
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import strata_tokenizer
    return strata_tokenizer


def load_tokenizer(path: str, family: str = "qwen-image"):
    """The prompt's tokenizer (tools/strata_tokenizer.py): HunyuanImage 3.0's own `tokenizer.json` (a byte-level BPE,
    from_hf) or Qwen3-VL's (Anima's Qwen3's), from the text encoder's GGUF."""
    strata_tokenizer = _tokenizer_module()
    if _pipeline(family) == "guided_token":
        return strata_tokenizer.Tokenizer.from_hf(path)
    return strata_tokenizer.Tokenizer.from_gguf(path)


def can_edit(cfg: dict) -> bool:
    """Whether the image model of the config section `cfg` edits (takes reference images): Qwen-Image-2.1's and
    Mage-Flow's with their vision encoder (and Mage-Flow's edit model), HunyuanImage 3.0's with its vision model's
    file, MOSS-Image's with its edit adapter (edit_lora); MageTrail does not."""
    if _pipeline(cfg.get("family")) == "latent_chat":
        return bool(cfg.get("edit_lora"))
    return bool(cfg.get("vision")) and not (_pipeline(cfg.get("family")) == "flow" and not cfg.get("edit_model"))


def moss_edit_adapter(cfg: dict) -> dict:
    """The config section's MOSS-Image edit adapter, chosen among its "edit_adapters" (setup's: {name: {"lora",
    "format", "vision"}}) by XESTRATA_MOSS_EDIT, else its "edit_adapter", else MOSS_EDIT_DEFAULT -> the section's
    fields for it (edit_lora, edit_format, vision: the text encoder's vision encoder where its layout needs one); {}
    for a section without them (its edit_lora as it is).  ValueError for a name it has not, or a file not there."""
    adapters = cfg.get("edit_adapters") or {}
    if not adapters:
        return {}
    name = os.environ.get("XESTRATA_MOSS_EDIT") or cfg.get("edit_adapter") or MOSS_EDIT_DEFAULT
    if name not in adapters:
        raise ValueError(f"MOSS-Image's edit adapter {name!r} (XESTRATA_MOSS_EDIT or the config's \"edit_adapter\"): "
                         f"one of {', '.join(adapters)}")
    a = adapters[name]
    for path in (a.get("lora"), a.get("vision")):
        if path and not Path(path).is_file():
            raise ValueError(f"MOSS-Image's edit adapter {name}: {path} is not there (run setup with --images again)")
    return {"edit_adapter": name, "edit_lora": a["lora"], "edit_format": a.get("format") or "",
            "vision": a.get("vision")}


def moss_edit_format(cfg: dict) -> str:
    """The layout of the config's MOSS-Image edit adapter (MOSS_EDIT_FORMATS): the config's "edit_format", else the
    file's: Moss-Image-Edit's where its metadata's adapter_config says "moss-edit-lora", else ai-toolkit's (an
    ai-toolkit file says only that it is ai-toolkit's, not whether its kv_cache was on: on, as the published edit
    adapters have it (akhaliq's armA to armV7s1, 7err4's and pleiades7's CharRef), though ai-toolkit's own default is
    off - an adapter trained with it off is "ai-toolkit-no-kv" in the config; SimpleTuner's edit adapters, its
    metadata's modelspec.architecture "moss_image-sft-edit/lora", are trained in the kv_cache-on layout as well)."""
    fmt = cfg.get("edit_format")
    if fmt:   # setup's edit_adapters say it; a config of its own may
        if fmt not in MOSS_EDIT_FORMATS:
            raise ImageRequestError(f'images "edit_format": {", ".join(MOSS_EDIT_FORMATS)}')
        return fmt
    return _adapter_format(cfg["edit_lora"])


@functools.lru_cache(maxsize=8)
def _adapter_format(path: str) -> str:
    return "moss-image-edit" if _moss_edit_meta(path) else "ai-toolkit"


def _moss_edit_meta(path: str) -> bool:
    """A safetensors file whose metadata's adapter_config says it is Moss-Image-Edit's ("moss-edit-lora")."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        conf = json.loads(meta.get("adapter_config") or "{}")
    except (OSError, ValueError, struct.error):
        return False
    return isinstance(conf, dict) and conf.get("format") == "moss-edit-lora"


def _aitk_edit(config: Path, stem: str | None) -> bool | None:
    """An ai-toolkit job's config.yaml: its model_kwargs' kv_cache (True or False) where its edit is true, else None;
    `stem` (a shared config.yaml): only where the job's name is the file's ("armV5", or "armV5_000000750" a step's).
    Read by its lines, as ai-toolkit writes it (config: name, process: ... model: model_kwargs: edit, kv_cache)."""
    try:
        lines = config.read_text(errors="replace").splitlines()
    except OSError:
        return None
    name, kw, edit, kv, indent = None, False, False, False, -1
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition(":")
        depth = len(line) - len(line.lstrip())
        value = value.strip().strip("'\"")
        if key == "name" and name is None and depth <= 2:
            name = value
        if kw and depth <= indent:
            kw = False
        if key == "model_kwargs":
            kw, indent = True, depth
        elif kw and key == "edit":
            edit = value.lower() == "true"
        elif kw and key == "kv_cache":
            kv = value.lower() == "true"
    if stem is not None and not (name and (stem == name or re.fullmatch(re.escape(name) + r"_\d+", stem))):
        return None
    return kv if edit else None


def moss_edit_layout(path: Path) -> str | None:
    """The layout (MOSS_EDIT_FORMATS) a file in the LoRA folder is a MOSS-Image edit adapter in, from the files, else
    None (not one, or not one that says so): <name>.json beside it with an "edit_format"; Moss-Image-Edit's metadata;
    an ai-toolkit job's config (<name>.yaml, or the config.yaml ai-toolkit writes beside its output, of the same job
    name) with edit true - kv_cache on "ai-toolkit", off "ai-toolkit-no-kv".  An ai-toolkit file's own metadata says
    neither that it edits nor its kv_cache, and "appended" has no trainer's config to say it."""
    try:
        fmt = json.loads(path.with_suffix(".json").read_text()).get("edit_format")
        if fmt in MOSS_EDIT_FORMATS:
            return fmt
    except (OSError, ValueError, AttributeError):
        pass
    if _moss_edit_meta(str(path)):
        return "moss-image-edit"
    for config, stem in ((path.with_suffix(".yaml"), None), (path.parent / "config.yaml", path.stem)):
        kv = _aitk_edit(config, stem)
        if kv is not None:
            return "ai-toolkit" if kv else "ai-toolkit-no-kv"
    return None


def moss_edit_adapters(cfg: dict) -> dict[str, dict]:
    """The MOSS-Image edit adapters a request may name (edit_adapter): the config's "edit_adapters" (setup's chinoll,
    armv5 and armv6: {"lora", "format", "vision", "about"}), then the LoRA folder's files moss_edit_layout knows, by
    their listing's names - ai-toolkit's with the config's vision encoder (Qwen3.5-9B's mmproj; without it, not
    offered)."""
    out = {k: dict(v) for k, v in (cfg.get("edit_adapters") or {}).items()}
    vision = cfg.get("vision") or next((a.get("vision") for a in out.values() if a.get("vision")), None)
    folder = lora_api.folder(cfg)
    have = {os.path.realpath(a["lora"]) for a in out.values() if a.get("lora")}
    for e in folder.listing():
        if e["name"] in out or os.path.realpath(folder.dir / e["path"]) in have:
            continue
        fmt = moss_edit_layout(folder.dir / e["path"])
        if fmt is None or (fmt.startswith("ai-toolkit") and not vision):
            continue
        out[e["name"]] = {"lora": str(folder.dir / e["path"]), "format": fmt,
                          "vision": vision if fmt.startswith("ai-toolkit") else None, "about": f"LoRA folder, {fmt}"}
    return out


def moss_pick_adapter(cfg: dict, name: str) -> dict:
    """The edit adapter `name` of moss_edit_adapters, or ImageRequestError saying why it cannot be used (not there,
    or a LoRA of the folder that does not say it is an edit adapter, or one that needs the vision encoder)."""
    adapters = moss_edit_adapters(cfg)
    if name in adapters:
        return adapters[name]
    folder = lora_api.folder(cfg)
    known = {e["name"]: e for e in folder.listing()}
    if name in known:
        fmt = moss_edit_layout(folder.dir / known[name]["path"])
        why = (f"{name} is an ai-toolkit edit adapter, and Qwen3.5-9B's vision encoder (mmproj) it needs is not set up "
               "(run setup with --images)" if fmt else
               f"{name} is a LoRA of the folder that does not say it is a MOSS-Image edit adapter: put {name}.json "
               'beside it with {"edit_format": ...} (' + ", ".join(MOSS_EDIT_FORMATS) + ") or its ai-toolkit "
               "config.yaml")
    else:
        why = f"no MOSS-Image edit adapter {name}"
    raise ImageRequestError(f"edit_adapter: {why}; the edit adapters here: {', '.join(adapters) or 'none'}")


def moss_picture(data: bytes, box: list | None = None) -> bytes:
    """A picture the text encoder of an ai-toolkit MOSS-Image edit sees, as the adapter was trained with it:
    the image (or its `box`, x y w h, its parts past the edges the edges' copies) shrunk by area to at most
    MOSS_PICTURE_PIXELS, then its sides rounded to multiples of 32 (bilinear; process_qwen2vl_images) -> a PNG."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    if box is not None:
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > im.width or y + h > im.height:   # the edges copied past them
            import numpy as np
            a = np.asarray(im)
            ys = np.clip(np.arange(y, y + h), 0, im.height - 1)
            xs = np.clip(np.arange(x, x + w), 0, im.width - 1)
            im = Image.fromarray(np.ascontiguousarray(a[ys][:, xs]), "RGBA")
        else:
            im = im.crop((x, y, x + w, y + h))
    k = min(1.0, (MOSS_PICTURE_PIXELS / (im.width * im.height)) ** 0.5)
    if k < 1.0:
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BOX)
    w, h = (max(32, round(v / 32) * 32) for v in im.size)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.BILINEAR)
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def moss_ref_size(w0: int, h0: int, pixels: int = 1024 * 1024, grow: bool = True) -> tuple[int, int]:
    """The size moss_ref makes a w0 x h0 image."""
    if not grow:
        pixels = min(pixels, w0 * h0)
    tw = (pixels * w0 / h0) ** 0.5
    w, h = (max(SIZE_STEP, round(v / SIZE_STEP) * SIZE_STEP) for v in (tw, tw * h0 / w0))
    return w, h


def moss_ref(data: bytes, pixels: int = 1024 * 1024, grow: bool = True) -> bytes:
    """A MOSS-Image edit's reference image at about `pixels` pixels in its shape (`grow` false: at most that many),
    sides multiples of 16 (Lanczos; MOSS_EDIT_STEPS' note), as a PNG; one already that size as it is."""
    import io
    from PIL import Image
    w0, h0 = image_dims(data)
    w, h = moss_ref_size(w0, h0, pixels, grow)
    if (w, h) == (w0, h0):
        return data
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGBA").resize((w, h), Image.LANCZOS).save(out, format="PNG", compress_level=1)
    return out.getvalue()


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
ENGINE_ENV = (*(p + n for n in ("GPU_PCI", "VRAM_LIMIT_MIB", "MAX_ALLOC_MIB", "NO_XMX")
                for p in (envvars.NEW, envvars.OLD)),
              "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "ONEAPI_DEVICE_SELECTOR", "HIP_VISIBLE_DEVICES",
              "ZE_AFFINITY_MASK", "LD_LIBRARY_PATH")


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
    others = ("CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")
    pci = envvars.get("GPU_PCI", env=env)
    # With a PCI address, selectors only narrow visibility; all environments naming that physical card share a turn.
    sel = [pci.lower(), None, None, None, None] if pci else [None] + [env.get(k) for k in others]
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
            file_lock.acquire(f)
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
            file_lock.release(f)
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
    """The image engine, `xestrata-image --serve`, through the image daemon (serve/image_daemon.py): started outside the
    server the first time and found again by the next server of the same image model and GPU, so a model switch does
    not lose it; one request line at a time (the caller serializes, the daemon between servers)."""

    def __init__(self, cfg: dict, env: dict, log_path: Path, say):
        self.cfg, self.env, self.log_path, self.say = cfg, env, log_path, say
        self.sock: Any = None
        self.f: Any = None
        self.lines: queue.Queue = queue.Queue()
        self.ended = threading.Event()              # the daemon closed this connection
        self.pid: int | None = None                 # the engine's process (the daemon's READY), what holds its VRAM
        self.found = False                          # the last start found the daemon running (another server's)

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
        if self.cfg.get("lora_bake") is False:   # the adapters added to each product instead of baked into the weights
            cmd += ["--lora-bake", "0"]
        if self.cfg.get("lora_block_map"):   # Anima-2.9B: the release's adapters bound to its blocks
            cmd += ["--lora-block-map", ",".join(str(int(b)) for b in self.cfg["lora_block_map"])]
        return cmd + ["--serve"]

    def alive(self) -> bool:
        """Connected, and the daemon has not closed the connection (another server's job may have ended it, GpuTurn):
        a closed one is let go, so the next job starts the engine again."""
        if self.f is not None and self.ended.is_set():
            self.close()
        return self.f is not None

    def _connect(self, addr: str) -> bool:
        import socket
        s = None
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
            if s is not None:
                with contextlib.suppress(OSError):
                    s.close()
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
        info = _kv(line)
        self.pid = int(info["pid"]) if isinstance(info.get("pid"), int) else None
        self.found = proc is None
        return info

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
            stage=None, preview=None, pause=None) -> dict:
        """Send a request line and wait for its answer (the line starting with `want`) -> its fields; STEP lines go to
        `step`, STAGE lines (refs, text, sampling, decode) to `stage`, PREVIEW lines (a partial image: its index, the
        step, the file of its RGBA8 rows) to `preview`; at a YIELD (UPSV's YIELD:1) `pause` runs, then RESUME goes
        back.  ImageJobFailed on ERROR."""
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
            elif got == "YIELD":
                try:
                    if pause is not None:
                        pause()
                finally:
                    self.send("RESUME")
            elif downgrades.from_info_line(got, "image engine"):
                pass
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
        if _pipeline(self.family) == "latent_chat":   # its edit adapter, as the config or the environment chooses
            self.cfg.update(moss_edit_adapter(self.cfg))
            # the engine opens the vision encoder where any adapter a request may name needs it (moss_pick_adapter;
            # it reads it only for a request with pictures)
            if not self.cfg.get("vision"):
                self.cfg["vision"] = next((a["vision"] for a in (self.cfg.get("edit_adapters") or {}).values()
                                           if a.get("vision")), None)
        for k in self._file_keys():
            if not isinstance(self.cfg.get(k), str) or not self.cfg[k]:
                raise ValueError(f'the config\'s "images" section needs "{k}" (run setup with --images again)')
        if Path(self.cfg["exe"]).name.startswith("sd-cli"):
            raise ValueError('the config\'s "images" section is for stable-diffusion.cpp, which XeStrata no longer '
                             "uses (run setup with --images again)")
        self.policy, self.reload = policy, reload
        # the ids the DiT does not take: the leading system turn's (HunyuanImage 3.0's DiT reads all of its own)
        self.drop = (0 if _pipeline(self.family) in ("guided_token", "latent_chat", "dual_tokenizer")
                     else MF_DROP if _pipeline(self.family) == "flow" else DROP)
        self.model = self.cfg.get("model_name") or MODEL_NAME
        self.log = log
        out = self.cfg.get("out_dir")
        self.dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="xestrata-images-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_files = int(self.cfg.get("keep_files") or 64)
        self.reserve = int(self.cfg.get("reserve_mib") or 512)
        self.spare = int(self.cfg.get("spare_mib") or 1024)
        self.lock = EngineLock(self)              # one request line at a time even when the service's turn is not taken
        self.long_job: dict | None = None         # a long job's state while it runs (videos.upscale): EngineLock
        self.loras = lora_api.folder(self.cfg)    # the adapters its DiT takes (serve/loras.py)
        self.engine = ImageEngine(self.cfg, dict(os.environ) if env is None else env, self.dir / "xestrata-image.log",
                                  self._say)
        self.tok = None
        self.t5: Any = None   # Anima's T5 tokenizer (tools/strata_unigram.py), read with the first prompt
        self._system: str | None = None          # HunyuanImage 3.0's system prompt, read once
        self.totals = {"images": 0, "jobs": 0, "seconds": 0.0, "failed": 0}
        self.yield_turn: Callable[[], None] | None = None   # gpu_turn's, while a job has the turn
        self.last: dict | None = None              # the last job's figures (/v1/status)
        # the measured speed, an edit's apart (its reference images): for each size made (in megapixels), one step's
        # seconds (a guided step two) and the rest's an image (estimate_s)
        self.rates: dict[str, dict[float, list[float]]] = {}
        self.progress: tuple | None = None         # (step, steps) of the running job
        # the job in the GPU's turn as the Monitor shows it (live()), and the jobs that ended, newest last (recent()):
        # the one record of an image job's progress, which /metrics reads
        self.job: dict | None = None
        self.owner: ImageModels | None = None   # the ImageModels it is one of (gpu_turn's GPU)
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
        skip-layer guidance one more in its window, as is the guidance field's), the size the engine makes."""
        mw, mh = made_size(job)
        passes = 2 if job.get("cfg") else 1
        if job.get("slg_layers") and job.get("slg_scale"):
            passes += job.get("slg_stop", 1.0) - job.get("slg_start", 0.0)
        for m in job.get("guide") or ():   # skip layer guidance's pass of the prompt alone, in its window
            if m["name"] == "skip_layer_guidance" and m["scale"] > 0:
                passes += m["end"] - m["start"]
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
        if _pipeline(self.family) == "guided_token":
            return ("exe", "diffusion_model", "vae", "tokenizer", "system_prompt") + \
                (("vision",) if self.cfg.get("vision") else ())
        if _pipeline(self.family) == "dual_tokenizer":   # its LLM adapter's queries come from the T5 tokenizer
            return ("exe", "diffusion_model", "vae", "llm", "t5_tokenizer")
        return ("exe", "diffusion_model", "vae", "llm") + (("vision",) if self.cfg.get("vision") else ())

    def missing(self) -> list[str]:
        return [self.cfg[k] for k in self._file_keys() if not Path(self.cfg[k]).exists()]

    def _tokenizer_path(self) -> str:
        """HunyuanImage 3.0's tokenizer.json or Qwen3-VL's text encoder GGUF, as the config names them."""
        return self.cfg["tokenizer"] if _pipeline(self.family) == "guided_token" else self.cfg["llm"]

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

    def _lora_ignored(self, job: dict) -> list[dict]:
        """The applied adapters' keys that name no tensor of the model (PEFT's `base_layer.bias`, an auxiliary
        projector, a flow-map sidecar): the response's xestrata-engine.lora_ignored and the log say them (up to 6
        names, and the count), as serve/loras.py's Folder.ignored kept them at the last scan."""
        got = lora_api.folder(self.cfg).ignored
        out = []
        for use in job.get("loras") or ():
            keys = got.get(use["name"]) or []
            if keys:
                out.append({"name": use["name"], "n": len(keys), "keys": keys[:6]})
        return out

    def _say(self, text: str) -> None:
        print(f"[xestrata] {text}", flush=True)
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

    def _ready(self, start: bool = True) -> None:
        if self.tok is None:
            self.tok = load_tokenizer(self._tokenizer_path(), self.family)
        if not self.engine.alive():
            if not start:
                raise ImageJobFailed("the image engine is not running")
            self.engine.start()
            self._enter_ledger()

    def ledger(self, total_mib: int = 0) -> vram_ledger.Ledger:
        """The image GPU's VRAM ledger (serve/vram_ledger.py), `total_mib` the engine's report of the GPU's memory."""
        return vram_ledger.Ledger(gpu_key(self.engine.env), total_mib=total_mib, engine_total=total_mib > 0)

    def _owner(self) -> int:
        """The process whose ledger entry is the engine's: the engine's own (it outlives this server), or this server
        for a daemon that did not say its engine's."""
        return self.engine.pid if self.engine.pid is not None else os.getpid()

    def _enter_ledger(self) -> None:
        """After the engine's start (the caller holds the lock): what it holds entered in the ledger.  A daemon found
        running whose entry the ledger does not have (an older server's, a ledger lost) is entered as in the base:
        the device's report already shows it."""
        try:
            got = self.engine.ask("FREE", "FREE", 60)
        except ImageJobFailed:
            return
        led = self.ledger(int(got.get("total_mib") or 0))
        held = int(got.get("held_mib") or 0)
        if led.find("image", self._owner()) is None and held > 0:
            led.hold("image", held, self._owner(), in_base=self.engine.found, whole=True)

    def ledger_sync(self) -> None:
        """The engine's ledger entry set to what it holds now, its own report (FREE's held_mib: the image model's and
        the image tools' models'), or removed when it holds nothing or has ended: the entry lives as long as the
        engine holds VRAM, past the job and past this server."""
        with self.lock:
            led, owner = self.ledger(), self._owner()
            held = 0
            if self.engine.alive():
                try:
                    held = int(self.engine.ask("FREE", "FREE", 60).get("held_mib") or 0)
                except ImageJobFailed:
                    held = 0
            eid = led.find("image", owner)
            if held > 0:
                led.hold("image", held, owner, whole=True)
            elif eid is not None:
                led.releasing(eid)
                led.remove(eid)

    def ids(self, job: dict) -> list[int]:
        if self.tok is None:
            self.tok = load_tokenizer(self._tokenizer_path(), self.family)
        if "ids" not in job:
            if _pipeline(self.family) == "guided_token":   # the DiT reads the whole sequence, the image blocks in it
                bucket = job.get("bucket") or hunyuanimage3_bucket(self.cfg, job["width"], job["height"])
                # a reference image's block names the bucket nearest its shape: the engine reads it at that size
                job["ref_buckets"] = [hunyuanimage3_bucket(self.cfg, *image_dims(r)) for r in job.get("refs") or ()]
                job["ids"] = hunyuanimage3_ids(self.tok, model_prompt(job), bucket, self._base_size(),
                                               self.system_text(), job["ref_buckets"])
            elif _pipeline(self.family) == "dual_tokenizer":
                if self.t5 is None:
                    _tokenizer_module()   # tools/ on the path
                    import strata_unigram
                    self.t5 = strata_unigram.UnigramTokenizer.from_hf_json(self.cfg["t5_tokenizer"])
                job["ids"], job["t5"], job["t5_w"] = anima_tokens(self.tok, self.t5, model_prompt(job))
                if job.get("cfg"):
                    neg_parts = anima_tokens(self.tok, self.t5, job["negative_prompt"])
                    job["neg_ids"], job["neg_t5"], job["neg_t5_w"] = neg_parts
            else:
                # MOSS-Image's references are pictures of the prompt only for an ai-toolkit adapter
                refs = (len(job.get("refs") or ()) if _pipeline(self.family) != "latent_chat" or
                        job.get("edit_format", "").startswith("ai-toolkit") else 0)
                ids = prompt_ids(self.tok, model_prompt(job), refs, self.family)
                neg = None
                if job.get("cfg"):   # guidance's negative prompt, in the same form (its pads as the prompt's)
                    neg = prompt_ids(self.tok, job["negative_prompt"], refs, self.family)
                if _pipeline(self.family) == "latent_chat":   # its pipeline's max_sequence_length
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
        return MF_EDIT_DROP if _pipeline(self.family) == "flow" and job.get("refs") else self.drop

    @staticmethod
    def _moss_words(job: dict) -> str:
        """MOSS-Image's words of IMG: its steps and shift (the engine makes its sigmas), the blocks the negative pass
        skips, and skip-layer guidance's (moss_request)."""
        few = job.get("few_step")
        if few:   # the 4-step adapter: its time path and data space (its LoRA among _lora_words')
            return (f" STEPS:{min(int(job['steps']), MOSS_FEW_STEPS)} USKIP:none"
                    f" TIMEPATH:{lora_api.encode_path(few['time_path'])}"
                    f" DATASPACE:{lora_api.encode_path(few['dataspace'])}")
        words = (f" STEPS:{job['steps']} SHIFT:{job.get('shift', MOSS_SHIFT)!r}"
                 f" USKIP:{','.join(str(b) for b in job.get('uncond_skip', MOSS_UNCOND_SKIP)) or 'none'}")
        if job.get("slg_layers") and job.get("slg_scale"):
            words += (f" SLG:{','.join(str(b) for b in job['slg_layers'])} SLGSCALE:{job['slg_scale']!r}")
        if job.get("slg_start", 0.0) != 0.0 or job.get("slg_stop", 1.0) != 1.0:
            words += f" SLGSTART:{job['slg_start']!r} SLGSTOP:{job['slg_stop']!r}"
        return words

    @staticmethod
    def _lora_words(job: dict) -> str:
        """IMG's and NEED's adapters: LORA:<strength>[@<block weights>]:<path> for each, in the order the request gave
        them (the strength first, so a Windows path's colons are not read as one; the path's separators %XX'd, as the
        engine splits the line on whitespace - encode_path); a MOSS-Image edit's adapter (job["edit_lora"]) first, at
        1."""
        edit = f" LORA:1.0:{lora_api.encode_path(job['edit_lora'])}" if job.get("edit_lora") else ""
        if job.get("few_step"):   # the 4-step adapter's LoRA, at 1 (the engine adds it to the products)
            edit = f" LORA:1.0:{lora_api.encode_path(job['few_step']['lora'])}" + edit
        return edit + "".join(f" LORA:{use['scale']!r}"
                              + ("@" + ",".join(repr(float(b)) for b in use["blocks"]) if use.get("blocks") else "")
                              + f":{lora_api.encode_path(use['path'])}"
                              for use in job.get("loras") or ())

    @staticmethod
    def _cfg_words(job: dict) -> str:
        """IMG's guidance words: the negative prompt's ids and the scale (none without guidance); Anima's its T5 ids and
        weights too."""
        if not job.get("cfg"):
            return ""
        words = f" NEG:{','.join(str(i) for i in job['neg_ids'])} CFG:{job['cfg']:g}"
        if "neg_t5" in job:
            words += (f" NEGT5:{','.join(str(i) for i in job['neg_t5'])}"
                      f" NEGT5W:{','.join(f'{w:g}' for w in job['neg_t5_w'])}")
        return words

    @staticmethod
    def _anima_words(job: dict) -> str:
        """Anima's words of IMG: its steps and shift (the engine makes the sigmas), the sampler, its T5 ids and
        weights."""
        return (f" STEPS:{job['steps']} SHIFT:{ANIMA_SHIFT!r} SAMPLER:{job.get('sampler') or ANIMA_SAMPLERS[0]}"
                f" T5:{','.join(str(i) for i in job['t5'])} T5W:{','.join(f'{w:g}' for w in job['t5_w'])}")

    def stage(self, job: dict) -> None:
        """An edit's reference images, img2img's image and a mask into files the engine reads (job["ref_paths"],
        job["init_path"], job["mask_path"]); an ai-toolkit MOSS-Image edit's pictures too (job["pic_paths"],
        moss_picture: the first one the cut's box of it); unstage removes them."""
        if (job.get("refs") or job.get("init")) and not job.get("ref_dir"):
            work = Path(tempfile.mkdtemp(prefix="refs-", dir=self.dir))
            job["ref_dir"] = str(work)
            job["ref_paths"] = []
            for i, data in enumerate(job.get("refs") or ()):
                p = work / f"ref_{i}"
                p.write_bytes(data)
                job["ref_paths"].append(str(p))
            if job.get("edit_format", "").startswith("ai-toolkit") and job.get("refs"):
                crop = job.get("crop") or {}
                job["pic_paths"] = []
                for i, data in enumerate(job["refs"]):
                    p = work / f"pic_{i}"
                    p.write_bytes(moss_picture(data, crop.get("box") if i == 0 and crop.get("box") else None))
                    job["pic_paths"].append(str(p))
            for key in ("init", "mask"):
                if job.get(key):
                    p = work / key
                    p.write_bytes(job[key])
                    job[f"{key}_path"] = str(p)
        control = job.get("control") or {}
        mctl = job.get("mctl") or {}   # ControlNet's map, MOSS-Image's color and depth maps (image_controls.img_words)
        for key, data in (("control_path", control.get("map")), ("mctl_color_path", mctl.get("color")),
                          ("mctl_depth_path", mctl.get("depth"))):
            if data and not job.get(key):
                fd, name = tempfile.mkstemp(prefix="control-", dir=self.dir)
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                job[key] = name

    @staticmethod
    def unstage(job: dict) -> None:
        files = ("init_path", "mask_path", "te_path", "neg_te_path", "control_path", "mctl_color_path",
                 "mctl_depth_path")
        for p in ((job.pop("ref_paths", []) or []) + (job.pop("pic_paths", []) or [])
                  + [job.pop(k) for k in files if job.get(k)]):
            with contextlib.suppress(OSError):
                os.unlink(p)
        if job.get("ref_dir"):
            with contextlib.suppress(OSError):
                os.rmdir(job.pop("ref_dir"))

    @staticmethod
    def _crop_ref(job: dict) -> tuple[int, int] | None:
        """An ai-toolkit (or "appended") MOSS-Image edit's first reference beside a cut's drawing: the cut at the
        references' size (MOSS_AITK_REF_PIXELS), as the other references are, where it is smaller than the drawing."""
        if not (job.get("edit_format") in MOSS_EDITFMT_WORDS and job.get("refs")):
            return None
        gw, gh = job["crop"]["gen"]
        ref = moss_ref_size(gw, gh, MOSS_AITK_REF_PIXELS, grow=False)
        return ref if ref != (gw, gh) else None

    @staticmethod
    def _refs_words(job: dict, need: bool = False) -> str:
        """The engine's words of a job's files: REF:, MASK: and INIT:, the text states TE: and NEGTE: (text_states),
        and STRENGTH: below 1 (its repr, the double strength_run counted with).  `need`: NEED's, where a strength's
        base image (an edit's first reference) is counted by INIT:- when no mask or init image names it."""
        words = ("".join(f" REF:{p}" for p in job.get("ref_paths") or ())
                 + (f" MASK:{job['mask_path']}" if job.get("mask_path") else "")
                 + (crop_word(job["crop"], ImageGen._crop_ref(job)) if job.get("crop") and not job["crop"].get("empty")
                    else "")
                 + (f" COLOR:{job['color_match']}" if job.get("crop") and not job["crop"].get("empty")
                    and job.get("color_match") else "")
                 + (ImageGen._ups_words(job, False, "crop_ups") if job.get("crop_ups") and job.get("crop")
                    and not job.get("upscale") else "")
                 + (" UPSCALE" + ImageGen._ups_words(job, need) if job.get("upscale") else "")
                 + ("" if not job.get("tiles") else
                    f" TILES:{job['tiles']['tile']},{job['tiles']['pad']},{job['tiles']['blur']}")
                 + (f" EDITFMT:{MOSS_EDITFMT_WORDS[job['edit_format']]}" if job.get("edit_format") in MOSS_EDITFMT_WORDS
                    else "")
                 + "".join(f" PIC:{p}" for p in job.get("pic_paths") or ())
                 + (f" INIT:{job['init_path']}" if job.get("init_path") else "")
                 + (f" TE:{job['te_path']}" if job.get("te_path") else "")
                 + (f" NEGTE:{job['neg_te_path']}" if job.get("neg_te_path") else ""))
        if job.get("strength") is not None:
            if not need:
                words += f" STRENGTH:{job['strength']!r}"
            elif not (job.get("mask_path") or job.get("init_path")):
                words += " INIT:-"
        return words

    @staticmethod
    def _ups_spec(job: dict, key: str = "upscaler") -> str:
        """The engine's name of an upscale's upscaler: <method>[:<its weights' path, %XX'd as LORA:'s>]
        (job[key]["path"], set by upscaler_weights; key "crop_ups": the masked edit's cut's, crop_upscaler)."""
        u = job[key]
        return u["method"] + (f":{lora_api.encode_path(u['path'])}" if u.get("path") else "")

    @staticmethod
    def _ups_words(job: dict, need: bool = False, key: str = "upscaler") -> str:
        """An upscale's first pass on IMG and NEED: UPS:<spec> and its color correction (UPSCOLOR:), none for Lanczos
        alone (the engine's own resize); NEED's the base image's size too (UPSIN:).  key "crop_ups": the upscaler
        the masked edit's cut is enlarged by (beside CROP:)."""
        u = job.get(key) or {}
        if not u or u["method"] == "lanczos" and u.get("color") in (None, "none"):
            return ""
        words = f" UPS:{ImageGen._ups_spec(job, key)}" + (f" UPSCOLOR:{u['color']}" if u.get("color") else "")
        if need and key == "upscaler":   # an upscale in tiles: the whole size too (NEED's is a tile's)
            words += " UPSIN:{},{}".format(*image_dims(job["init"])) + (
                f",{job['width']},{job['height']}" if job.get("tiles") else "")
        return words

    def need(self, job: dict, *, start: bool = True) -> dict:
        """What the image needs, from the engine (started when it is not running): least_mib, full_mib (every layer
        resident), held_mib (what it holds now), free_mib (the device's free VRAM), cap_mib (the most the engine could
        have) and host_mib (the prefix cache in pinned RAM at cap_mib).  An upscale without a drawing (ups_only): its
        upscaler's (UPSNEED).  An auxiliary model's run (job["aux"]: image_regions' segmentation or background removal):
        its own (AUXNEED)."""
        if job.get("aux"):
            a = job["aux"]
            with self.lock:
                self._ready(start=start)
                return self.engine.ask(f"AUXNEED {a['kind']} {a['path']} {job['width']} {job['height']}"
                                       + "".join(f" {v}" for v in a.get("need_args") or ()), "NEED", 600)
        if job.get("video"):   # a video's upscale (serve/videos.py): its upscaler's on the source's frames
            src = job["source"]
            u = job["upscaler"]
            with self.lock:
                self._ready(start=start)
                return self.engine.ask(f"UPSVNEED {job['width']} {job['height']} {src['width']} {src['height']} "
                                       f"{self._ups_spec(job)}" + (f" COLOR:{u['color']}" if u.get("color") else ""),
                                       "NEED", 600)
        if job.get("ups_only"):
            w0, h0 = image_dims(job["init"])
            u = job["upscaler"]
            with self.lock:
                self._ready(start=start)
                return self.engine.ask(f"UPSNEED {job['width']} {job['height']} {w0} {h0} {self._ups_spec(job)}"
                                       + (f" COLOR:{u['color']}" if u.get("color") else ""), "NEED", 600)
        with self.lock:
            self._ready(start=start)
            w, h = made_size(job)
            self.stage(job)
            n, neg = self._need_counts(job)
            got = self.engine.ask(f"NEED {w} {h} {n}{self._refs_words(job, need=True)}{neg}{self._lora_words(job)}"
                                  f"{_controls().need_words(job)}", "NEED", 600)
            if job.get("refine"):   # the second pass's too: the larger of each
                job2 = self._refine_job(job)
                self.stage(job2)
                try:
                    w2, h2 = made_size(job2)
                    n2, neg2 = self._need_counts(job2)
                    two = self.engine.ask(f"NEED {w2} {h2} {n2}{self._refs_words(job2, need=True)}{neg2}"
                                          f"{self._lora_words(job2)}", "NEED", 600)
                finally:
                    self._unstage_refine(job2)
                got = {k: max(v, two.get(k, v)) if k in ("least_mib", "full_mib") else v for k, v in got.items()}
            return got

    def _need_counts(self, job: dict) -> tuple[int, str]:
        """NEED's prompt ids after the system turn and its NEG: word, an ai-toolkit MOSS-Image edit's pictures counted
        at their tokens (the engine widens each picture's <|image_pad|> to them, in both prompts)."""
        sizes = [image_dims(Path(p).read_bytes()) for p in job.get("pic_paths") or ()]
        wide = sum((pw // 32) * (ph // 32) - 1 for pw, ph in sizes)
        n = len(self.ids(job)) - self.drop_of(job) + wide
        if not job.get("cfg"):   # the negative prompt has the same pictures before it (ids)
            return n, ""
        return n, f" NEG:{len(job['neg_ids']) - self.drop_of(job) + wide}"

    def _refine_job(self, job: dict, picture: bytes | None = None) -> dict:
        """The second pass of a job with "refine" (CROP_REFINE_STRENGTH): its cut's plan and strength, the
        first pass's `picture` as the first reference (the first image's until then), its own files staged."""
        job2 = {k: v for k, v in job.items()
                if k not in ("ref_dir", "ref_paths", "pic_paths", "mask_path", "init_path", "refine")}
        job2.update(crop=job["refine"]["crop"], strength=job["refine"]["strength"],
                    steps_run=job["refine"]["steps_run"])
        if picture is not None:
            job2["refs"] = [picture] + list(job["refs"][1:])
        return job2

    @staticmethod
    def _unstage_refine(job2: dict) -> None:
        """A second pass's own files removed (the text states' files are the first pass's: unstage would take them)."""
        for p in ((job2.get("ref_paths") or []) + (job2.get("pic_paths") or [])
                  + [job2[k] for k in ("mask_path", "init_path") if job2.get(k)]):
            with contextlib.suppress(OSError):
                os.unlink(p)
        if job2.get("ref_dir"):
            with contextlib.suppress(OSError):
                os.rmdir(job2["ref_dir"])

    def _refine(self, job: dict, i: int, picture: bytes, out: Path, ids: str, guide: str, timeout: float, cancel,
                step, stage) -> dict:
        """Image i's second pass (a job with "refine") from the first's `picture` into `out` -> its IMAGE line."""
        job2 = self._refine_job(job, picture)
        self.stage(job2)
        gw, gh = job2["crop"]["gen"]
        self._live(stage="refs", step=0, steps=job2["steps_run"], sampling_at=None)
        try:
            return self.engine.ask(f"IMG {gw} {gh} {job['seed'] + i} {self.drop_of(job)} {out} {ids}{guide}"
                                   f"{self._refs_words(job2)}{self._cfg_words(job2)}{self._lora_words(job2)}",
                                   "IMAGE", timeout, cancel, step, stage)
        except ImageJobFailed as e:
            self.totals["failed"] += 1
            why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
            self._say(f"image job failed: {why}")
            raise (LoraRejected if isinstance(e, LoraRejected) else ImageJobFailed)(why) from e
        finally:
            self._unstage_refine(job2)

    def upscale(self, job: dict, cancel: threading.Event | None = None) -> tuple[bytes, dict]:
        """An upscale without a drawing (job["ups_only"]): the engine's upscaler on job["init"] -> (the PNG, the
        engine's figures: upscale_s and its parts model_s, resize_s, color_s, the color applied, the tiles)."""
        timeout = float(self.cfg.get("timeout_s") or 3600)
        u = job["upscaler"]
        with self.lock:
            self._ready()
            work = Path(tempfile.mkdtemp(prefix="upscale-", dir=self.dir))
            try:
                pic, out = work / "picture", work / "out.rgba"
                pic.write_bytes(job["init"])
                self._live(stage="upscale", sampling_at=None)
                try:
                    got = self.engine.ask(f"UPS {job['width']} {job['height']} {pic} {out} {self._ups_spec(job)}"
                                          + (f" COLOR:{u['color']}" if u.get("color") else "")
                                          + f" SEED:{job['seed']}", "UPSCALED", timeout, cancel)
                except ImageJobFailed as e:
                    self.totals["failed"] += 1
                    why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                    self._say(f"image job failed: {why}")
                    raise ImageJobFailed(why) from e
                png = png_rgba(out.read_bytes(), job["width"], job["height"], job["width"], job["height"])
            finally:
                for f in work.iterdir():
                    with contextlib.suppress(OSError):
                        f.unlink()
                with contextlib.suppress(OSError):
                    work.rmdir()
        return png, got

    def upscale_video(self, job: dict, source: Path, stem: Path, cancel: threading.Event | None = None,
                      progress=None) -> dict:
        """A video's upscale (serve/videos.py): the engine's UPSV on the file `source` into `stem` + its container's
        extension, `progress` told each frame written -> the engine's figures (path, mime, frames, fps, vfr and the
        parts' seconds)."""
        timeout = float(self.cfg.get("video_timeout_s") or 7 * 86400)
        u = job["upscaler"]
        words = (f" COLOR:{u['color']}" if u.get("color") else "") + f" SEED:{job['seed']}" + \
            (f" CHUNK:{job['chunk']}" if job.get("chunk") else "") + \
            (f" OVERLAP:{job['overlap']}" if job.get("overlap") else "") + \
            (f" CODEC:{job['codec']}" if job.get("codec") else "") + \
            (" YIELD:1" if self.yield_turn is not None else "")   # the chat's requests run at its chunks' ends
        with self.lock:
            self._ready()
            self._live(stage="upscale-video", sampling_at=None)
            try:
                return self.engine.ask(f"UPSV {job['width']} {job['height']} {source} {stem} {self._ups_spec(job)}"
                                       + words, "UPSVIDEO", timeout, cancel, progress, pause=self.yield_turn)
            except ImageJobFailed as e:
                self.totals["failed"] += 1
                why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                self._say(f"video job failed: {why}")
                raise ImageJobFailed(why) from e

    def segment_video(self, weights: Path, source: Path, prompts: Path, stem: Path, seg: dict,
                      cancel: threading.Event | None = None, progress=None) -> dict:
        """A video's segmentation (serve/videos.py): the engine's SEGV of SAM 3.1 (`weights`) on the file `source`
        with the prompts file, its JSON lines and videos at `stem` + .jsonl and the rest, `progress` told each frame
        -> the engine's figures (json, objects, masks, mask, overlay, frames, fps, vfr and the parts' seconds)."""
        timeout = float(self.cfg.get("video_timeout_s") or 7 * 86400)
        words = (" MASKS:1" if seg.get("mask_video") else "") + \
            (" MASK:1" if "mask" in seg.get("outputs", ()) else "") + \
            (" OVERLAY:1" if "overlay" in seg.get("outputs", ()) else "") + \
            (" YIELD:1" if self.yield_turn is not None else "")   # the chat's requests run between frames
        with self.lock:
            self._ready()
            self._live(stage="segment-video", sampling_at=None)
            try:
                return self.engine.ask(f"SEGV {weights} {source} {prompts} {stem}" + words, "SEGVIDEO", timeout,
                                       cancel, progress, pause=self.yield_turn)
            except ImageJobFailed as e:
                self.totals["failed"] += 1
                why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                self._say(f"video job failed: {why}")
                raise ImageJobFailed(why) from e

    def aux_ask(self, line: str, want: str, timeout: float = 600) -> dict:
        """One request of the image tools' auxiliary models (image_regions: SEGMENT, FLORENCE, POLYMASK, MATTE) -> its
        answer's fields."""
        with self.lock:
            self._ready()
            return self.engine.ask(line, want, timeout)

    def detect(self, weights: Path, picture: bytes, threshold: float = FIX_THRESHOLD,
               masks: bool = False) -> tuple[list[dict], float]:
        """The engine's detector (`weights`, an image_aux file) on `picture` -> ([{"box": [x0, y0, x1, y1], "score",
        "cls", "mask": a bool array of the picture's size or None}], the engine's seconds)."""
        import numpy as np
        with self.lock:
            self._ready()
            work = Path(tempfile.mkdtemp(prefix="detect-", dir=self.dir))
            try:
                pic = work / "picture"
                pic.write_bytes(picture)
                prefix = work / "mask"
                got = self.engine.ask(f"DETECT {weights} {pic} {threshold!r}" + (f" {prefix}" if masks else ""),
                                      "DETECT", 600)
                W, H = image_dims(picture)
                out = []
                for i, entry in enumerate(e for e in str(got.get("boxes") or "").split(";") if e):
                    x0, y0, x1, y1, score, cls = entry.split(",")
                    m = None
                    if masks and (f := Path(f"{prefix}.{i}")).is_file():
                        m = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(H, W) > 0
                    out.append({"box": [float(x0), float(y0), float(x1), float(y1)], "score": float(score),
                                "cls": int(cls), "mask": m})
                return out, float(got.get("detect_s") or 0.0)
            finally:
                for f in work.iterdir():
                    with contextlib.suppress(OSError):
                        f.unlink()
                with contextlib.suppress(OSError):
                    work.rmdir()

    def free_mib(self) -> int:
        """The device's free VRAM, as the engine sees it (its parts left as they are)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask("FREE", "FREE", 60)["free_mib"])

    def total_mib(self) -> int:
        """The image GPU's memory, as the engine sees it."""
        return int(self.memory().get("total_mib") or 0)

    def memory(self) -> dict:
        """The engine's FREE: the device's free VRAM (free_mib), all the engine holds (held_mib) and the GPU's memory
        (total_mib), in MiB."""
        with self.lock:
            self._ready()
            return self.engine.ask("FREE", "FREE", 60)

    def takes_w8(self) -> bool | None:
        """Whether the image GPU takes the image DiTs' 8-bit form (W8A8), as the engine times it once a GPU (W8): None
        where it could not time it, or an engine that does not answer W8."""
        with self.lock:
            self._ready()
            try:
                take = int(self.engine.ask("W8", "W8", 120).get("take", -1))
            except ImageJobFailed:
                return None
        return None if take < 0 else take == 1

    def fit(self, extra_mib: int = 0) -> int:
        """The engine's budget set to what it holds and what is free now, less the reserve (and `extra_mib`: what the
        GPU's VRAM ledger keeps for others) -> the budget (MiB)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask(f"FIT {self.reserve + max(0, int(extra_mib))}", "BUDGET", 600)
                       .get("budget_mib", 0))

    def release(self, everything: bool) -> None:
        """The engine lets go of its cache (layers, activations, buffers); `everything`: its fixed weights too."""
        with self.lock:
            if self.engine.alive():
                try:
                    self.engine.ask("RELEASE ALL" if everything else "RELEASE", "RELEASE", 120)
                except ImageJobFailed as e:
                    self._say(f"image engine: release failed ({e})")
        self.ledger_sync()

    def close(self, shutdown: bool = False) -> None:
        """Leave the image daemon (it stays, warm, for the next server: a model switch); `shutdown`: end it."""
        with self.lock:
            self.engine.close(shutdown)

    def run(self, job: dict, cancel: threading.Event | None = None, first: int = 0,
            pause=None, preview=None) -> tuple[list[bytes], dict]:
        """Make `job`'s images (parse_request's dict) from image `first` -> (the PNG files' bytes in order, figures).
        The caller holds the GPU turn; `pause()` true after an image ends the call there (make_images gives the turn
        to the requests waiting for it).  ImageJobFailed when the engine fails, times out or the job is cancelled.
        `preview(image, index, png)`: the job's partial images (partial_images), as the engine makes them.  The cut
        parts' images (job["crop"]) come back at the picture's size; a mask that changes nothing gives the picture
        as it is, the engine not asked."""
        timeout = float(self.cfg.get("timeout_s") or 3600)
        crop = job.get("crop") or {}
        t0 = time.monotonic()
        images: list[bytes] = []
        parts: list[dict] = []
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
                # the engine's RGBA rows: the size made (cut to the size asked for), or the cut's picture; an
                # upscale in tiles is asked at its whole size (the engine draws it tile by tile: upscale_tiles)
                whole = crop and not crop.get("empty") or job.get("tiles")
                rw, rh = (job["width"], job["height"]) if whole else (mw, mh)
                lw, lh = (job["width"], job["height"]) if job.get("tiles") else (mw, mh)
                ids = ",".join(str(i) for i in self.ids(job))
                sched = job.get("sched") or self.cfg.get("schedule")
                # the engine owns the schedule now (SCHED:/STEPS: tell it the kind and the settings); an explicit
                # SIGMAS: list (the API's sigmas, the Turbo redraw's override) still overrides it
                sigmas = (" SIGMAS:" + ",".join(f"{x:.7g}" for x in job["sigmas"])) if job.get("sigmas") else ""
                # HunyuanImage 3.0's sigmas are the engine's own (from the step count): its step count and the
                # guidance token's scale go instead
                guide = (f" STEPS:{job['steps']} GUID:{job['guidance']:g}" if sched == "hunyuanimage3" else
                         self._moss_words(job) if _pipeline(self.family) == "latent_chat" else
                         self._anima_words(job) if _pipeline(self.family) == "dual_tokenizer" else
                         f" SCHED:base STEPS:{job['steps']}" if sched == "base" else
                         f" STEPS:{job['steps']}" if sched == "mage" else
                         f" SAMPLER:{job['sampler']}" if job.get("sampler") else "")
                guide += guidance_words(job, self.cfg, mw, mh)
                for i in range(first, job["n"]):
                    if i > first and pause is not None and pause():
                        break
                    if cancel is not None and cancel.is_set():
                        self.totals["failed"] += 1
                        raise ImageJobFailed("cancelled (the client left)")
                    if crop.get("empty"):
                        images.append(as_png(job["refs"][0] if job.get("refs") else job["init"]))
                        parts.append({})
                        continue
                    run = job.get("steps_run", job["steps"])
                    self.progress = (i * run, job["n"] * run)
                    self._live(image=i + 1, step=0, steps=run,
                               stage="refs" if job.get("refs") or job.get("init") else "text", sampling_at=None)
                    out = work / f"img_{i}.rgba"
                    shows = partial_steps(run, job.get("partial_images") or 0) if preview is not None else []
                    shown = " PREVIEW:" + ",".join(str(k) for k in shows) if shows else ""

                    def peek(index, done, path, i=i):
                        try:
                            png = png_rgba(Path(path).read_bytes(), job["width"], job["height"], rw, rh)
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
                        r = self.engine.ask(f"IMG {lw} {lh} {job['seed'] + i} {self.drop_of(job)} {out} {ids}{sigmas}"
                                            f"{guide}{self._refs_words(job)}{self._cfg_words(job)}{shown}"
                                            f"{self._lora_words(job)}"
                                            f"{_controls().img_words(job, self.cfg, lw, lh)}", "IMAGE",
                                            timeout, cancel, step, stage, **({"preview": peek} if shows else {}))
                    except ImageJobFailed as e:
                        self.totals["failed"] += 1
                        why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                        self._say(f"image job failed: {why}")
                        raise (LoraRejected if isinstance(e, LoraRejected) else ImageJobFailed)(why) from e
                    images.append(png_rgba(out.read_bytes(), job["width"], job["height"], rw, rh))
                    parts.append(r)
                    self._measured(job, r)
                    if job.get("refine"):   # its part cut out of the first pass and drawn again (CROP_REFINE_STRENGTH)
                        r = self._refine(job, i, images[-1], work / f"img_{i}.refine.rgba", ids, guide, timeout,
                                         cancel, step, stage)
                        images[-1] = png_rgba((work / f"img_{i}.refine.rgba").read_bytes(), job["width"],
                                              job["height"], job["width"], job["height"])
                        parts.append(r)
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
                   "size": f"{job['width']}x{job['height']}", "seed": job["seed"], "engine": "xestrata-image",
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
        if (mw, mh) != (job["width"], job["height"]) and not crop.get("empty"):   # made larger and cut, or a part
            figures["made_size"] = f"{mw}x{mh}"                                    # drawn larger: the web page says so
        if crop:   # the cut: the box drawn (on the picture, x y w h) or nothing to change
            figures["crop"] = ({"empty": True} if crop.get("empty") else
                               {"box": list(crop["box"]), "whole": bool(crop.get("whole"))})
        if job.get("refine"):   # its second pass: the part cut out of the first and drawn again
            two = job["refine"]["crop"]
            figures["refine"] = {"box": list(two["box"]), "made_size": "{}x{}".format(*two["gen"]),
                                 "strength": job["refine"]["strength"]}
        if job.get("requested_size"):                      # made at the model's bucket instead: the web page says so
            figures["requested_size"] = job["requested_size"]
        if job.get("loras"):                               # the IMAGE line's count and VRAM, and a tag's names left out
            figures.update(lora_n=int(parts[-1].get("lora_n") or 0), lora_mib=parts[-1].get("lora_mib"))
        if job.get("lora_unknown"):
            figures["lora_unknown"] = list(job["lora_unknown"])
        ignored = self._lora_ignored(job)   # keys of an applied file that hold no role: said, not applied in silence
        if ignored:
            figures["lora_ignored"] = ignored
            for one in ignored:
                names = ", ".join(one["keys"]) + (", ..." if one["n"] > len(one["keys"]) else "")
                self._say(f"lora {one['name']}: {one['n']} of its keys hold no adapter tensor and were left out"
                          f" ({names})")
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
        applied = [{"name": u["name"], "scale": u["scale"], **({"blocks": u["blocks"]} if u.get("blocks") else {})}
                   for u in job.get("loras") or ()]
        backgrounds, flatten_s = [], None
        for i, png in enumerate(images):
            item = {"seed": job["seed"] + i} if job.get("ups_only") else \
                {"revised_prompt": model_prompt(job), "seed": job["seed"] + i}
            if applied:   # the adapters this image was made with
                item["lora"] = [dict(u) for u in applied]
            opaque = job.get("background") == "opaque"
            t0 = time.monotonic() if opaque else None
            img = encode_image(png, fmt, job.get("output_compression", 100), opaque)
            if t0 is not None and fmt == "png":   # the flatten's own time (only PNG suffers no re-encode otherwise)
                flatten_s = round((flatten_s or 0.0) + time.monotonic() - t0, 3)
            backgrounds.append(picture_background(png, fmt, opaque))
            if job["response_format"] == "url":
                item["url"] = f"{base_url}/v1/images/files/{self.store(img, fmt)}"
            else:
                item["b64_json"] = base64.b64encode(img).decode("ascii")
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": self.model, "output_format": fmt,
                "size": f"{job['width']}x{job['height']}",
                "background": "transparent" if "transparent" in backgrounds else "opaque",
                "quality": job.get("quality") or "auto", "usage": usage(job),
                "xestrata": ({**figures, "flatten_s": flatten_s} if flatten_s is not None else figures)}

    # ---- the progress the Monitor shows (/metrics' live.image) and the jobs that ended (its Requests table)
    STAGE_LABELS = {"waiting": "Making room on the GPU", "size": "Choosing the image's size",
                    "refs": "Reading the reference images",
                    "text": "Reading the prompt", "sampling": "Sampling", "decode": "Decoding the image (VAE)",
                    "upscale": "Upscaling", "upscale-video": "Upscaling a video",
                    "segment-video": "Finding objects in a video", "segment": "Finding the region",
                    "matte": "Removing the background"}

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
        out = {"enabled": True, "model": self.model, "engine": "xestrata-image", "quant": self.cfg.get("quant"),
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
        # the text engine's helper card, when the jobs run there (server.image_helper_card): closed for each job
        self.helper_card = ""
        self.gpu_pci: str | None = None   # the card of its own (separate), as the server's image_env named it
        self.helper_jobs = 0
        self.helper_lock = threading.Lock()
        self.gens: dict[str, ImageGen] = {}
        for c in cfgs:
            g = ImageGen(c, log=log, env=env)
            if g.model in self.gens:
                raise ValueError(f'two image models are named {g.model!r} ("images" and "image_models")')
            g.owner = self
            self.gens[g.model] = g
        self.default = self.current = next(iter(self.gens.values()))
        self.gpu = GpuTurn(self.default.engine.env)   # the models share one environment, so one GPU
        # the text model's GPU lent to the images (use_text_gpu, tool_parallel): the same models with the text engine's
        # environment, made when first lent (server.py sets lend_cfgs on a card of their own)
        self.lend_cfgs: tuple | None = None
        self._lent: ImageModels | None = None
        self._lent_lock = threading.Lock()

    def lending(self) -> bool:
        """Whether a job runs on the text model's GPU, lent to the images, now (the chat's requests wait meanwhile)."""
        return self._lent is not None and any(g.job is not None for g in self._lent.gens.values())

    def lent(self) -> "ImageModels":
        """The models' engines on the text model's GPU (made once): their jobs take the GPU's turn there as on a
        shared card, the text model making room for them (gpu_turn)."""
        with self._lent_lock:
            if self._lent is None:
                if self.lend_cfgs is None:
                    raise ImageRequestError("use_text_gpu: the images have no card of their own here")
                cfgs, log, env = self.lend_cfgs
                self._lent = ImageModels(cfgs, log=log, env=env, separate=False)
            return self._lent

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
        if self._lent is not None:   # the text model's GPU, lent to the images before: given back too
            self._lent.give_back()
        if self.separate:
            return
        for g in self.gens.values():
            if g.engine.alive():
                g._say(f"image model: {g.model}'s engine gives the GPU back for the model")
                g.release(True)

    def make_room(self, cancel: threading.Event | None = None) -> int:
        """For another engine of this GPU short of room (the translation engine): every model's engine lets go of all
        it holds, in the GPU's turn (so no job runs meanwhile; a job running waits it out) -> how many let go.  An
        engine kept warm after its job (keep, separate with room to spare) holds the card's VRAM until the next job."""
        n = 0
        self.gpu.acquire(cancel)
        try:
            for g in self.gens.values():
                if g.engine.alive():
                    g._say(f"image model: {g.model}'s engine gives the GPU back for the translation model")
                    g.release(True)
                    n += 1
        finally:
            self.gpu.release()
        return n

    def busy(self) -> bool:
        """Whether a job of one of the models runs now."""
        return any(g.job is not None for g in self.gens.values())

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
# an image tool's picture: a kept image's link (a tool's result, or the conversation's), else the conversation's last
IMAGE_ARG = {"type": "string", "description": "a link a result gave; left out: the last image (an attached one too)"}
# how the masked edit enlarges the part it cuts out before it is drawn (crop_upscale), and SeedVR2's version for it
CROP_UPSCALE = ("auto", "lanczos", "realesrgan", "seedvr2")
CROP_ARGS: dict[str, Any] = {"crop_upscale": {"type": "string", "enum": list(CROP_UPSCALE)},
                             "variant": {"type": "string", "enum": ["7b", "7b-sharp"]}}
# what every image and video tool takes besides its own: the plan's step it does (the chart's box), and the group of
# calls run side by side that one failure stops (server.py run_with_mcp)
PLAN_ARGS: dict[str, Any] = {"step": {"type": "string"}, "group": {"type": "string"},
                             "plan": {"type": "string", "description": "the first call's: the steps' mermaid chart"}}
TOOLS: list[dict[str, Any]] = [
    {"name": "generate_image",
     "description": "Make a new image from words alone; the user sees it, and the result gives its link. Not for a "
                    "new image in an attached picture's pose or layout (control), nor a change of a picture "
                    "(edit_image).",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string"},
         "size": {"type": "string", "description": "WxH, e.g. 1024x1024"},
         "seed": {"type": "integer"}},
         "required": ["prompt"]}},
    {"name": "edit_image",
     "description": "Change a whole image, or make a new one from it; the user sees it. Not for: one part only (her "
                    "hair, the clothes, the sky: edit_region), larger or finer (upscale_image), wider, taller or "
                    "another shape (extend_image keeps the picture; this redraws it), a cut-out or plain background "
                    "(remove_background), faces or hands (fix_image), a new subject in a picture's pose (control). "
                    "The same person in another picture's pose: that picture in references. A model without edits "
                    "draws over it (img2img): the prompt then says the whole picture.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "the change"},
         "image": IMAGE_ARG,
         "size": {"type": "string", "description": "WxH; left out: the image's shape (another shape: extend_image)"},
         "strength": {"type": "number", "description": "0 to 1"},
         "references": {"type": "array", "items": {"type": "string"}, "description": "links: images 2, 3, ..."},
         "mask": {"type": "string", "description": "a mask's link (white changes)"},
         "refine": {"type": "boolean", "description": "MOSS-Image: the masked part again, finer"},
         **CROP_ARGS, "seed": {"type": "integer"}},
         "required": ["prompt"]}},
    {"name": "extend_image",
     "description": "Grow an image past its edges (outpaint), the image kept as it is.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "the whole image, its new area too"},
         "image": IMAGE_ARG,
         "left": {"type": "integer"}, "top": {"type": "integer"}, "right": {"type": "integer"},
         "bottom": {"type": "integer"},
         "aspect": {"type": "string", "description": "e.g. 16:9"},
         "seed": {"type": "integer"}},
         "required": ["prompt"]}},
    {"name": "upscale_image",
     "description": "Make an image larger by an upscaler; with redraw, draw it again finer.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "what it shows (with redraw)"},
         "image": IMAGE_ARG,
         "scale": {"type": "number"},
         "method": {"type": "string", "enum": list(UPSCALE_METHODS)},
         "variant": {"type": "string", "enum": list(SEEDVR2_VARIANTS[:2])},
         "redraw": {"type": "boolean"},
         "strength": {"type": "number"},
         "seed": {"type": "integer"}}}},
    {"name": "upscale_video",
     "description": "Make the attached video larger: a job on the server; the answer says where it goes. Do not wait.",
     "parameters": {"type": "object", "properties": {
         "video": {"type": "string", "description": "a /v1/videos/files/ link (default: the last video)"},
         "scale": {"type": "number"},
         "method": {"type": "string", "enum": list(UPSCALE_METHODS)},
         "variant": {"type": "string", "enum": list(SEEDVR2_VARIANTS[:2])},
         "seed": {"type": "integer"}}}},
    {"name": "segment_video",
     "description": "Find and follow things through the attached video (SAM 3.1): a job on the server; the answer says "
                    "where it goes. Do not wait.",
     "parameters": {"type": "object", "properties": {
         "prompts": {"type": "array", "items": {"type": "string"}, "description": "English noun phrases"},
         "video": {"type": "string", "description": "a /v1/videos/files/ link (default: the last video)"},
         "masks": {"type": "boolean"}},
         "required": ["prompts"]}},
    {"name": "fix_image",
     "description": "Find faces, hands or people in an image and draw each again, finer; the rest kept.",
     "parameters": {"type": "object", "properties": {
         "parts": {"type": "array", "items": {"type": "string", "enum": list(FIX_PARTS)}},
         "prompt": {"type": "string"},
         "image": IMAGE_ARG,
         "strength": {"type": "number"},
         **CROP_ARGS, "seed": {"type": "integer"}},
         "required": ["parts"]}},
    {"name": "edit_region",
     "description": "Draw again only the part of an image that words name; the rest kept.",
     "parameters": {"type": "object", "properties": {
         "region": {"type": "string", "description": "English words, e.g. hair"},
         "mask": {"type": "string", "description": "instead of region: a mask's link (find_region's)"},
         "prompt": {"type": "string", "description": "what the part becomes"},
         "image": IMAGE_ARG,
         "strength": {"type": "number"},
         **CROP_ARGS, "seed": {"type": "integer"}},
         "required": ["prompt"]}},
    {"name": "find_region",
     "description": "Find a part of an image by words: the user sees it marked, the result links its mask.",
     "parameters": {"type": "object", "properties": {
         "region": {"type": "string", "description": "English words, e.g. face"},
         "image": IMAGE_ARG},
         "required": ["region"]}},
    {"name": "control",
     "description": "Make a new image whose composition follows a picture's (pose, outlines, depth); the user sees it. "
                    "Nothing of the picture's people but their pose is kept: to keep a given person (their face, "
                    "their clothes), edit_image with the pose's picture in references.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "the whole new image"},
         "type": {"type": "string", "enum": ["canny", "depth", "pose", "lineart", "lineart_coarse", "hed",
                                             "scribble_hed", "scribble", "scribble_xdog", "mlsd", "gray", "color",
                                             "none"]},
         "image": IMAGE_ARG,
         "size": {"type": "string", "description": "WxH"},
         "strength": {"type": "number"},
         "end": {"type": "number"},
         "seed": {"type": "integer"}},
         "required": ["prompt", "type"]}},
    {"name": "remove_background",
     "description": "Take an image's background away (a transparent PNG), or put a plain color behind its subject.",
     "parameters": {"type": "object", "properties": {
         "image": IMAGE_ARG,
         "version": {"type": "string", "enum": ["general", "lite", "lite-2k", "portrait", "matting", "hr", "dynamic"]},
         "background_color": {"type": "string", "description": "e.g. #ffffff"}}}},
    {"name": "train_lora",
     "description": "Open the LoRA training panel in the chat: the user puts the pictures in, picks the kind "
                    "(character, style, other), the name, the trigger word and the model, captions the pictures and "
                    "starts the training there; it runs on the server and its versions come back to the panel. Call "
                    "it when the user wants to make a LoRA; do not wait for the training.",
     "parameters": {"type": "object", "properties": {
         "model": {"type": "string", "description": "The image model requested for the LoRA, if specified"}}}},
]

# the image tools every request declares; the others once the chat's model reads their skills (ImageTools.unlock)
CORE_TOOLS = ("generate_image", "edit_image")
# the tools that start from an image
EDIT_TOOLS = ("edit_image", "extend_image", "upscale_image", "fix_image", "edit_region", "find_region",
              "remove_background", "control")
# the always-shown skill of the image tools (serve/skills/image-tools), and the catalog load_skill reads: what each
# image model offered is made for (the tools' "model" choice)
SKILL_ALWAYS = "image-tools"
SKILL_MODELS = "models"
SKILL_MODELS_WHEN = "choosing the image model: what each one is made for."
SKILL_LORA = "lora"   # how the chat's model chooses the LoRAs, their strength and block weights (generate, edit)
SKILL_COMBINE = "combine-tools"   # its "tool" is "*": offered with two image tools or more
STYLE_DEFAULT = "natural"         # the prompt style of a model whose skill names none
# the copy of a made image an image tool's result shows a chat model that reads pictures: its longer side, and how many
# such copies the image folder's seen/ keeps (a request's later rounds read them)
SEEN_SIDE = 512
SEEN_KEEP = 32


@functools.cache
def _builtin_skills():
    from serve import skills as skills_api
    return skills_api.Skills(None)


TEXT_GPU_MODES = ("off", "ask", "on")


def text_gpu_mode(conf, asked=None) -> str:
    """The use_text_gpu setting in force: the request's (xestrata_images use_text_gpu: off, ask, on, or true / false)
    over the config's, "ask" by default.  ImageRequestError for a value that is none of them."""
    out = "ask"
    for v, where in ((conf, 'the config\'s "use_text_gpu"'), (asked, "xestrata_images.use_text_gpu")):
        if v is None:
            continue
        v = {True: "on", False: "off"}.get(v, v) if isinstance(v, bool) else v
        if v not in TEXT_GPU_MODES:
            raise ImageRequestError(f"{where}: off, ask or on, not {v!r}")
        out = v
    return out


def mask_to_bw(mask: bytes) -> bytes:
    """An edit's mask (OpenAI's: transparent where it changes) as the picture a tool's result links: white where it
    changes, black elsewhere (a grey PNG)."""
    import io
    from PIL import Image
    a = Image.open(io.BytesIO(mask)).convert("RGBA").getchannel("A")
    out = io.BytesIO()
    a.point(lambda v: 255 - v).save(out, format="PNG")
    return out.getvalue()


def mask_from_bw(data: bytes) -> bytes:
    """A mask's picture a tool's argument links -> OpenAI's mask.  A black-and-white one (mask_to_bw's, or any grey
    picture): white changes; one with transparent pixels: as OpenAI reads it."""
    import io
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
    except Exception as e:
        raise ImageRequestError(f"mask: the picture could not be read ({e})") from None
    if im.mode in ("RGBA", "LA", "PA") and im.getchannel("A").getextrema()[0] < 255:
        out = im.convert("RGBA")
    else:
        out = Image.new("RGBA", im.size, (0, 0, 0, 255))
        out.putalpha(im.convert("L").point(lambda v: 255 - v))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def fit_mask(mask: bytes, image: bytes) -> bytes:
    """A mask the web page painted at the size it showed the image, at the image's own size (nearest pixels)."""
    import io
    from PIL import Image
    try:
        m, im = Image.open(io.BytesIO(mask)), Image.open(io.BytesIO(image))
    except Exception as e:
        raise ImageRequestError(f"mask: the PNG could not be read ({e})") from None
    if m.size == im.size:
        return mask
    out = io.BytesIO()
    m.convert("RGBA").resize(im.size, Image.NEAREST).save(out, format="PNG")
    return out.getvalue()


def user_marks(want: dict, load, file) -> dict:
    """xestrata_images' edit and layout, what the user marked on an image in the web page's preview, into ImageTools'
    `user`: {"edit": {"image": a data: URL or a /v1/images/files/ link, "mask": {"image_url": a data: URL}}} - the
    picture and the paint over the part to change; {"layout": {"items": [{"label", "caption", "text", "box": [x0, y0,
    x1, y1] in shares of the sides}], "size": "WxH", "image": the picture they were drawn on}} - MOSS-Image's boxes, in
    pixels of that size.  `load` reads a
    data: URL, `file` a kept image by its file name."""
    out: dict = {}
    edit = want.get("edit")
    if edit is not None:
        mask = edit.get("mask") if isinstance(edit, dict) else None
        url = mask.get("image_url") if isinstance(mask, dict) else None
        which = edit.get("image") if isinstance(edit, dict) else None
        if not (isinstance(url, str) and url.startswith("data:")) or not isinstance(which, str):
            raise ImageRequestError('xestrata_images.edit: {"image": a data: URL or a /v1/images/files/ link, '
                                    '"mask": {"image_url": a data: URL}}')
        kept = FILE_REF.search(which)
        image = file(kept.group(1)) if kept else load(which) if which.startswith("data:") else None
        if image is None:
            raise ImageRequestError("xestrata_images.edit: the image is not here (a kept image the server let go?)")
        out["image"], out["mask"] = image, load(url)
    layout = want.get("layout")
    if layout is not None:
        size = layout.get("size") if isinstance(layout, dict) else None
        items = layout.get("items") if isinstance(layout, dict) else None
        if not isinstance(size, str) or not re.fullmatch(r"\d+x\d+", size) or not isinstance(items, list) or not items:
            raise ImageRequestError('xestrata_images.layout: {"items": [{"label", "caption", "text", "box"}], '
                                    '"size": "WxH"}')
        w, h = (int(v) for v in size.split("x"))
        drawn = []
        for it in items:
            box = it.get("box") if isinstance(it, dict) else None
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and
                                                                    not isinstance(v, bool) for v in box)):
                raise ImageRequestError("xestrata_images.layout: each item's box is [x0, y0, x1, y1] in shares")
            drawn.append({**{k: it[k] for k in ("label", "caption", "text") if isinstance(it.get(k), str)},
                          "bbox": [round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)]})
        out["layout"], out["size"] = {"items": drawn}, (w, h)
        which = layout.get("image")                    # the picture they were drawn on (a mask's edit redraws it)
        if isinstance(which, str) and "image" not in out:
            kept = FILE_REF.search(which)
            image = file(kept.group(1)) if kept else load(which) if which.startswith("data:") else None
            if image is not None:
                out["image"] = image
    return out


def note_marks(messages: list, user: dict) -> None:
    """The marks the user made (user_marks) said in their last message too, after what they wrote: the words alone ("do
    this edit") do not say what to do, and the model asked back instead of calling the tool the marks are for."""
    parts = []
    if user.get("mask") is not None:
        parts.append("a mask painted over the part of the image to change")
    if user.get("layout") is not None:
        parts.append(f"boxes for where things go ({user_layout_words(user['layout'])})")
    if not parts:
        return
    note = (f"[In the page, the user marked the image: {'; '.join(parts)}. They are what to make: call "
            f"{'edit_image' if user.get('mask') is not None else 'generate_image'} now; it applies them.]")
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [*content, {"type": "text", "text": note}]
        else:
            msg["content"] = f"{content or ''}\n\n{note}".lstrip()
        return


def user_layout_words(layout: dict) -> str:
    """The boxes the user drew, as the tool's description names them."""
    return "; ".join(f"{it.get('label', 'object')} \"{it.get('text') or it.get('caption') or ''}\""
                     for it in layout.get("items", []))


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
                 model: str | None = None, mask: bytes | None = None, user: dict | None = None,
                 skills: str = "index", tools=None, text_gpu=None, all_tools=None):
        self.models = images if isinstance(images, ImageModels) else None
        gen = images.default if isinstance(images, ImageModels) else images
        if model is not None:   # the request's own image model (xestrata_images {"model": ...}): that one alone
            gen = images.pick(model) if isinstance(images, ImageModels) else gen
            if gen.model != model:
                raise ValueError(f"xestrata_images: model must be one of "
                                 f"{', '.join(images.gens if isinstance(images, ImageModels) else [gen.model])}")
            self.models = None
        self.user = user or {}
        if self.user.get("layout") is not None:   # the user's boxes are MOSS-Image's: a model of its, chosen or not
            every = list(images.gens.values()) if isinstance(images, ImageModels) else [gen]
            moss = [g for g in every if _pipeline(g.family) == "latent_chat"]
            if not moss:
                raise ImageRequestError("xestrata_images.layout: the boxes are MOSS-Image's, and no MOSS-Image model "
                                        "is installed here")
            if self.models is None and _pipeline(gen.family) != "latent_chat":
                gen = moss[0]
        self.svc, self.gen, self.messages, self.load, self.normalize = svc, gen, messages, load, normalize
        # the skills: "index", the tools' short descriptions, load_skill's (serve/skills.py SkillTools) carrying
        # skill_part and the list of the rest, the chat's model reading those it needs (a chat that makes no image
        # carries no more); "inline", every skill of the models offered in the tools' descriptions (the image button's
        # request, which is an image's for certain, and the Responses route's image_generation tool)
        self.skills = skills
        # the image tools this request may use: the config's "image_tools" (every one by default), narrowed by the
        # request's xestrata_images.tools - a tool the config leaves out cannot be asked for
        offered = [t["name"] for t in TOOLS]
        conf = gen.cfg.get("image_tools")
        allowed = set(offered if conf is None else conf)
        if tools is not None:
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                raise ImageRequestError("xestrata_images.tools: a list of image tools' names")
            unknown = [t for t in tools if t not in offered]
            if unknown:
                raise ImageRequestError(f"xestrata_images.tools: {', '.join(unknown)}: one of {', '.join(offered)}")
            allowed &= set(tools)
        self.allowed = allowed
        # a Responses request's image_generation tool: its size, quality, output_format, output_compression and
        # background, over what the model's call says (size "auto": the call's)
        self.options = {k: v for k, v in (options or {}).items()
                        if k in OPTION_KEYS and v is not None and not (k == "size" and v == "auto")}
        self.settings = {"max_rounds": 4}
        # the text model's GPU lent to the images of a reply's calls (tool_parallel): the config's "use_text_gpu" (off,
        # ask: the chat's model asks for it by a call's use_text_gpu, on), the request's xestrata_images
        # use_text_gpu over it
        self.text_gpu = text_gpu_mode(gen.cfg.get("use_text_gpu"), text_gpu)
        # the tools beyond CORE_TOOLS declared once their skills are read (template_tools); all_tools: every one at
        # once (the image button's request, the Responses route's, xestrata_images' all_tools)
        self.all_tools = skills == "inline" or all_tools is True
        self.unlocked: set[str] = self._unlocked_before()
        self._declared = set(self.unlocked)
        self.made: list[str] = []          # the images this request made (file names)
        self.done_calls: dict[tuple, str] = {}   # the calls that ran well, by name and arguments -> their link
        self.max_calls: int | None = None  # a Responses request's max_tool_calls: the calls that run, at most
        self.calls = 0
        self.mask = mask                   # an edit's mask (Responses' input_image_mask), over the image it edits
        # what the user marked on an image in the web page's preview (xestrata_images' edit and layout, user_marks):
        # {"image": bytes, "mask": bytes} that edit_image takes, {"layout": ..., "size": (w, h)} generate_image does
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
        """The image tools this request declares now: generate_image and edit_image (CORE_TOOLS), and the others once
        the chat's model has read their skills (unlock, through load_skill) - every one from the start for the image
        button's request, the Responses route's and a request that asks for all (xestrata_images all_tools)."""
        offered = self.offered(exclude)
        names = [t["name"] for t in offered]
        if self.all_tools or not any(n in CORE_TOOLS for n in names):
            return offered
        return [t if t["name"] in CORE_TOOLS or t["name"] in self.unlocked or t["name"] == "train_lora"
                else self._stub(t) for t in offered]

    def _stub(self, t: dict) -> dict:
        """A tool not given yet, as its name and one line: called, it runs where its arguments are there, else answers
        with its skill and arguments, and is given whole from the next round on (unlock)."""
        s = self._tool_skill(t["name"])
        when = s.description if s is not None else t["description"]
        return {"name": t["name"], "description": f"{when} Call it; its arguments come with the first call.",
                "parameters": {"type": "object", "properties": {}}}

    def _tool_skill(self, tool: str):
        return next((s for s in self._skills().all() if s.builtin and s.kind == "tool" and s.meta.get("tool") == tool),
                    None)

    def _stub_answer(self, name: str, args: dict, t0: float) -> dict | None:
        """A call of a tool given as a stub: None where its required arguments are there (it runs as it is), else its
        skill and arguments, the tool given whole from the next round on."""
        full = next((t for t in self.offered() if t["name"] == name), None)
        self.unlock(name)
        props = full["parameters"]["properties"] if full is not None else {}
        if full is None or args and all(k in args for k in full["parameters"].get("required", [])) \
                and set(args) & set(props):
            return None   # its arguments are there: it runs as it is (a word it does not take is left out)
        s = self._tool_skill(name)
        listed = "; ".join(f"{k} ({v.get('type', '')}" + (f": {v['description']}" if v.get("description") else "")
                           + ")" for k, v in props.items())
        text = (f"Not run yet: {name} is ready now. Call {name} again at once, with these arguments, to do the step. "
                f"{full['description']}\nArguments: {listed}."
                + (f"\n\n# {s.name}\n\n{s.body}" if s is not None else ""))
        return {"ok": True, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": [], "stub": True}

    def offered(self, exclude=()) -> list[dict]:
        """Every image tool this request may use, declared now or once its skill is read (template_tools)."""
        out = []
        for t in TOOLS:
            gens = self._gens(t["name"] in EDIT_TOOLS)
            if t["name"] == "control":   # the models that take a control image (image_controls.capability)
                gens = [g for g in gens if _controls().capability(g.cfg)]
            if not gens or t["name"] in exclude or t["name"] not in self.allowed:
                continue
            # the user's boxes alone make a new image by them (the picture they were drawn on is the canvas' background
            # only); with a mask too, the painted part is redrawn by them
            if t["name"] == "edit_image" and self.user.get("layout") is not None and self.user.get("mask") is None:
                continue
            if t["name"] in ("upscale_video", "segment_video") and self._video("") is None:
                continue   # offered for a video in the conversation
            if t["name"] == "train_lora":
                from serve import lora_train
                if not any(lora_train.trainer_family(g) is not None for g in gens):
                    continue   # offered where a model with a trainer is installed
            t = dict(t)
            if self.skills == "inline":   # the image button's request: the models' skills in the description
                styles, rules = self._skill_groups(gens)
                store = self._skills()
                t["description"] = "\n\n".join(
                    [t["description"], store.body(SKILL_ALWAYS)]
                    + [f"With {', '.join(models)} ({(s.meta.get('name') if (s := store.get(name)) else '') or name}):"
                       f"\n{text}" for name, models in styles.items() if (text := store.body(name))]
                    + [f"With {', '.join(models)}:\n{text}" for name, models in rules.items()
                       if (text := store.body(name))])
            if t["name"] == "edit_image" and self.attached():
                t["description"] += ("\n\nWithout references, the pictures the user attached to their last message "
                                     "go with the edit after the image it edits: images 2, 3 and so on, in the order "
                                     "attached.")
            if t["name"] == "generate_image" and self.attached() and "control" in self.allowed:
                t["description"] += (" The user attached a picture: a new image in its pose or layout is "
                                     "control's, a change of it edit_image's.")
            if self.attached() and "image" in t["parameters"]["properties"]:   # their names for image, references
                t["description"] += (f" The user's pictures: attached:1 to attached:{len(self.attached())}, in "
                                     "the order attached." if len(self.attached()) > 1 else
                                     " The user's picture: attached:1.")
            if t["name"] == "edit_image" and self.user.get("mask") is not None:
                t["description"] += ("\n\nThe user has painted over an image the part to change (a mask): "
                                     "edit_image changes that part of that image only; its image and mask are set.")
            if self.user.get("layout") is not None:
                t["description"] += ("\n\nThe user has drawn over an image where things go (boxes: "
                                     f"{user_layout_words(self.user['layout'])}): "
                                     + ("edit_image redraws the painted part by them" if t["name"] == "edit_image" else
                                        "generate_image makes a new image by them")
                                     + "; the boxes and the size are set (a layout you write is replaced).")
            guided = [g for g in gens if chat_guidance_methods(g.cfg)]
            if t["name"] in ("generate_image", "edit_image") and guided:   # the sampling's guidance (C4)
                methods = [m for m in GUIDANCE_METHODS if any(m in chat_guidance_methods(g.cfg) for g in guided)]
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "guidance": {
                    "type": "array", "items": {"type": "string", "enum": methods},
                    **({} if len(guided) == len(gens) else
                       {"description": f"{', '.join(g.model for g in guided)} only"})}}
                t["parameters"] = params
            moss = [g.model for g in gens if _pipeline(g.family) == "latent_chat"]
            if t["name"] == "edit_image" and not any(
                    can_edit(g.cfg) for g in gens if _pipeline(g.family) == "latent_chat"):
                params = dict(t["parameters"])   # refine is MOSS-Image's edits' alone
                params["properties"] = {k: v for k, v in params["properties"].items() if k != "refine"}
                t["parameters"] = params
            adapters = {name: a for g in gens if _pipeline(g.family) == "latent_chat" and can_edit(g.cfg)
                        for name, a in moss_edit_adapters(g.cfg).items()}
            if t["name"] == "edit_image" and len(adapters) > 1:   # MOSS-Image's edit adapters, named by the user
                default = next((g.cfg.get("edit_adapter") for g in gens if _pipeline(g.family) == "latent_chat"), None)
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "edit_adapter": {
                    "type": "string",
                    "description": "MOSS-Image's edit LoRA, only when the user names one (their words for it): "
                                   + "; ".join(f"{name}" + (f" ({a['about']})" if a.get("about") else "")
                                               for name, a in adapters.items())
                                   + (f"; default {default}" if default else "")}}
                t["parameters"] = params
            if t["name"] == "generate_image" and moss:   # MOSS-Image's layout of boxes, drafted by the chat's model
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "layout": {
                    "type": "object",
                    "description": f"{', '.join(moss)}: where things and text go",
                    "properties": {
                        "summary": {"type": "object", "properties": {
                            "caption": {"type": "string"},
                            "canvas": {"type": "string", "enum": list(MOSS_CANVASES)}}},
                        "items": {"type": "array", "items": {"type": "object", "properties": {
                            "label": {"type": "string", "enum": list(MOSS_LABELS)},
                            "caption": {"type": "string"},
                            "text": {"type": "string"},
                            "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                            "required": ["label", "caption", "bbox"]}}},
                    "required": ["items"]}}
                t["parameters"] = params
            if t["name"] in ("generate_image", "edit_image") and self._lora_lines(gens):   # the models' LoRAs
                t = self._with_lora(t, gens)
            params = dict(t["parameters"])   # the plan's step and group (combine-tools), every tool's
            params["properties"] = {**params["properties"], **PLAN_ARGS}
            if self.text_gpu == "ask" and t["name"] not in ("upscale_video", "segment_video") and self._can_lend():
                params["properties"]["use_text_gpu"] = {"type": "boolean"}
            t["parameters"] = params
            if len(self._gens()) > 1:   # the model's choice: each one's name and what it is good for
                lines = ("; ".join(self._model_lines(gens)) if self.skills == "inline" else
                         f"the {SKILL_MODELS} skill says what each is for")
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "model": {
                    "type": "string", "enum": [g.model for g in gens],
                    "description": f"default {gens[0].model}" + (f" ({lines})" if self.skills == "inline" else "")}}
                t["parameters"] = params
            out.append(t)
        return out

    @staticmethod
    def _lora_lines(gens: list) -> list[str]:
        """The LoRAs the models of `gens` list, one line each: its name, what it was trained to add and its trigger
        words (serve/loras.py's listing), and its model where several are offered."""
        lines = []
        for g in gens:
            for e in g.loras.listing() if g.loras.takes else ():
                about = ", ".join(x for x in (e.get("kind"), f"trigger {e['trigger']}" if e.get("trigger") else None,
                                              f"step {e['steps']}" if e.get("steps") else None)
                                  if x)
                lines.append(e["name"] + (f" ({about})" if about else "") + (f" [{g.model}]" if len(gens) > 1 else ""))
        return lines

    def _with_lora(self, t: dict, gens: list) -> dict:
        """generate_image's or edit_image's `lora` argument: the models' LoRAs by name, a strength and the block
        weights (BLOCK_PRESETS' names for the models' families, or a number each block with commas); the lora skill
        says how to choose them."""
        # each name with the models whose family has it: a model without the name refuses it (a call wasted on the
        # first try, 2026-11-16: Mage-Flow's character LoRA asked "character", then again without it)
        has: dict[str, list[str]] = {}
        for g in gens:
            for n in lora_api.BLOCK_PRESETS.get(g.family, {}):
                has.setdefault(n, []).append(g.model)
        presets = ["all (every model)"] + [n if len(v) == len(gens) else f"{n} ({', '.join(v)} only)"
                                           for n, v in has.items()]
        names = list(dict.fromkeys(e["name"] for g in gens if g.loras.takes for e in g.loras.listing()))
        params = dict(t["parameters"])
        params["properties"] = {**params["properties"], "lora": {
            "type": "array", "description": "LoRAs: " + "; ".join(self._lora_lines(gens)),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names},
                    "scale": {"type": "number", "description": "strength, default 1"},
                    "blocks": {"type": "string",
                               "description": ("the strength each DiT block: " + "; ".join(presets)
                                               + "; or a number each block with commas")}},
                "required": ["name"]}}}
        t = {**t, "parameters": params}
        if self.skills == "inline" and (body := self._skills().body(SKILL_LORA)):
            t["description"] += "\n\nLoRAs:\n" + body
        return t

    def _model_lines(self, gens: list) -> list[str]:
        """Each model's name and what it is made for (the catalog's ABOUT, else _about)."""
        about = _qimage().ABOUT
        choice = {v[0]: k.removesuffix("-uncensored") for k, v in _qimage().IMAGE_MODELS.items()}
        return [f"{g.model}: {about.get(choice.get(g.model, ''), '') or self._about(g)}".rstrip(": ") for g in gens]

    def _skills(self):
        """The skills' store (serve/skills.py): the server's, else XeStrata's own alone."""
        return getattr(self.svc, "skills", None) or _builtin_skills()

    def _model_skill(self, g):
        """The skill of a model: its own (by its name without -uncensored), else its family's; None when there is
        none.  Its prompt style is read even where the skill is switched off."""
        own = g.model.removesuffix("-uncensored")
        found = [s for s in self._skills().all() if s.builtin and s.kind == "model"]
        return (next((s for s in found if s.meta.get("for") == own), None)
                or next((s for s in found if s.meta.get("for") == g.family), None))

    def _style(self, g) -> str:
        s = self._model_skill(g)
        return (s.meta.get("style") if s is not None else None) or STYLE_DEFAULT

    def _skill_groups(self, gens: list) -> tuple[dict, dict]:
        """The prompt styles and the model skills of `gens`: {skill: [model, ...]} each."""
        styles: dict[str, list[str]] = {}
        rules: dict[str, list[str]] = {}
        for g in gens:
            styles.setdefault(self._style(g), []).append(g.model)
            s = self._model_skill(g)
            if s is not None:
                rules.setdefault(s.name, []).append(g.model)
        return styles, rules

    def skill_index(self) -> list[tuple[str, str, str]]:
        """The image skills the chat's model may read for this request, in the order they are joined: [(name, kind,
        when to read it)] - the catalog (with several models), the styles' and the models' of the models offered, the
        tools' of the tools offered (combine-tools with two of them or more); a skill switched off is left out."""
        store = self._skills()
        tools = [t["name"] for t in self.offered()]
        if not tools:
            return []
        styles, rules = self._skill_groups(self._gens())
        out: list[tuple[str, str, str]] = []
        if len(self._gens()) > 1:
            out.append((SKILL_MODELS, "catalog", SKILL_MODELS_WHEN))
        on = {s.name: s for s in store.enabled(builtin=True)}
        for name in [*styles, *rules]:
            if name in on:
                out.append((name, on[name].kind, on[name].description))
        guided = any(chat_guidance_methods(g.cfg) for g in self._gens())
        for s in on.values():
            want = s.meta.get("tool")
            if s.kind != "tool" or not want:
                continue
            if want in tools and want not in CORE_TOOLS and want != "train_lora" and not self.all_tools:
                continue   # its tool's stub carries its line, and its call answers with the skill (_stub_answer)
            if want == "lora":   # with the LoRA argument only: a model lists LoRAs
                if {"generate_image", "edit_image"} & set(tools) and self._lora_lines(self._gens()):
                    out.append((s.name, "tool", s.description))
                continue
            if (want in tools or want == "*" and len(tools) > 1
                    or want == "guidance" and guided and {"generate_image", "edit_image"} & set(tools)):
                out.append((s.name, "tool", s.description))
        return out

    def skill_part(self) -> str:
        """What always goes with the image tools, in load_skill's description: their rules (the image-tools skill),
        which models take which prompt style, and what this setup can do."""
        rules = self._skills().body(SKILL_ALWAYS)
        kinds: dict[str, list[str]] = {}
        for g in self._gens():   # by their families (a model with a skill of its own, MageTrail, by its name)
            s = self._model_skill(g)
            key = s.meta.get("for", g.family) if s is not None else g.family
            names = kinds.setdefault(self._style(g), [])
            if key not in names:
                names.append(key)
        store = self._skills()
        named = {n: (s.meta.get("name") if (s := store.get(n)) is not None else None) or n for n in kinds}
        style = "; ".join(f"{named[n]} for {', '.join(fams)}" for n, fams in kinds.items())
        return "\n\n".join(p for p in (rules, f"Prompt style: {style}." if style else "", self.can_do()) if p)

    def can_do(self) -> str:
        """What this setup's image tools can do (the image models offered, their edits and references, the
        composition's control, the video tools), one line each: the chat's model plans with it."""
        gens = self._gens()
        tools = {t["name"] for t in self.offered()}
        lines = []
        edits = [g for g in gens if can_edit(g.cfg)]
        if edits and "edit_image" in tools:
            lines.append("edit_image: " + ", ".join(f"{g.model} (up to {refs_limit(g.cfg) - 1} references)"
                                                    for g in edits)
                         + "".join(f"; {g.model} img2img only" for g in gens if not can_edit(g.cfg)))
        if "control" in tools:
            ways: dict[str, list[str]] = {}
            for g in gens:
                mode = _controls().capability(g.cfg)
                if mode == "controlnet":
                    ways.setdefault("ControlNet", []).append(g.model)
                elif mode == "moss":
                    ways.setdefault("color and depth adapter", []).append(g.model)
                elif mode == "reference":
                    ways.setdefault("as a reference", []).append(g.model)
            lines.append("control: " + "; ".join(f"{k} with {', '.join(v)}" for k, v in ways.items()))
        videos = [n for n in ("upscale_video", "segment_video") if n in tools]
        if videos:
            lines.append(f"the attached video: {', '.join(videos)}")
        if getattr(self.svc, "vision", None) is not None:
            lines.append("you see a small copy of each image a tool makes")
        return "This setup:\n" + "\n".join(f"- {line}" for line in lines) if lines else ""

    def skill_body(self, name: str) -> str:
        """load_skill's text of one of skill_index's skills."""
        if name == SKILL_MODELS:
            return "# Image models\n\n" + "\n".join(f"- {line}" for line in self._model_lines(self._gens()))
        return self._skills().body(name)

    @staticmethod
    def _seen(gen, png: bytes) -> Path | None:
        """A copy of `png` at most SEEN_SIDE pixels a side (JPEG) in the image folder's seen/, the newest SEEN_KEEP kept
        -> its path; None where it cannot be written."""
        import io
        from PIL import Image
        try:
            d = Path(gen.cfg.get("out_dir") or tempfile.gettempdir()) / "seen"
            d.mkdir(parents=True, exist_ok=True)
            im = Image.open(io.BytesIO(png)).convert("RGB")
            im.thumbnail((SEEN_SIDE, SEEN_SIDE), Image.LANCZOS)
            path = d / f"{secrets.token_hex(16)}.jpg"
            im.save(path, format="JPEG", quality=85)
            for old in sorted(d.glob("*.jpg"), key=lambda f: f.stat().st_mtime)[:-SEEN_KEEP]:
                with contextlib.suppress(OSError):
                    old.unlink()
            return path
        except OSError:
            return None

    @staticmethod
    def _about(g) -> str:
        """A model the catalog does not describe: a derivative of Anima the user put in its folder."""
        if _pipeline(g.family) == "dual_tokenizer":
            return f"a derivative of Anima ({Path(str(g.cfg.get('diffusion_model', ''))).name})"
        return ""

    def routes(self) -> dict:
        return {t["name"]: (None, t["name"]) for t in self.template_tools()}

    def unlock(self, tool: str) -> bool:
        """A tool whose skill the chat's model read: declared from the next round on (new_tools).  -> whether it is
        new."""
        if tool in self.unlocked or tool not in {t["name"] for t in self.offered()}:
            return False
        self.unlocked.add(tool)
        return True

    def new_tools(self) -> list[dict]:
        """The tools unlocked since the last ask, for run_with_mcp to declare from the next round on."""
        out = [t for t in self.template_tools() if t["name"] in self.unlocked - self._declared]
        self._declared |= {t["name"] for t in out}
        return out

    def _unlocked_before(self) -> set[str]:
        """The tools the conversation's earlier turns unlocked (their load_skill calls, or calls of the tools
        themselves): the chat's model's choice then, kept, so the tools stay as the earlier turn ended them."""
        out: set[str] = set()
        tool_of = {s.name: s.meta.get("tool") for s in self._skills().all() if s.builtin and s.kind == "tool"}
        names = {t["name"] for t in TOOLS}
        for msg in self.messages:
            for call in msg.get("tool_calls") or [] if isinstance(msg, dict) else []:
                fn = call.get("function") if isinstance(call, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(name, str) and name in names:
                    out.add(name)
                elif name == "load_skill" and isinstance(fn, dict):
                    args = fn.get("arguments")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except ValueError:
                        args = None
                    for n in (args or {}).get("names") or [] if isinstance(args, dict) else []:
                        if tool_of.get(n) in names:
                            out.add(tool_of[n])
        return out

    def lane(self, name: str, args: dict) -> tuple[str, str, bool]:
        """Where a call of the reply runs beside the others (serve/tool_parallel.py) -> ("now": no GPU job to wait
        for, or "images": the image GPU's lane; the image model it takes; whether it may run on the text model's GPU,
        lent to it, instead)."""
        if name in ("upscale_video", "segment_video"):   # a job started on the server: the answer comes at once
            return "now", "", False
        if name == "upscale_image" and args.get("method") == "lanczos" and not args.get("redraw"):
            return "now", "lanczos", False
        if name in ("find_region", "remove_background"):
            gens = self._gens()
            return "images", gens[0].model if gens else self.gen.model, False
        gens = self._gens(name in EDIT_TOOLS)
        if name == "control":
            gens = [g for g in gens if _controls().capability(g.cfg)]
        if args.get("layout") not in (None, "", {}):
            gens = [g for g in gens if _pipeline(g.family) == "latent_chat"] or gens
        asked = args.get("model")
        gen = next((g for g in gens if g.model == asked), gens[0] if gens else self.gen)
        return "images", gen.model, self.may_lend(args)

    def _can_lend(self) -> bool:
        models = self.models or getattr(self.svc, "images", None)
        return isinstance(models, ImageModels) and models.separate and models.lend_cfgs is not None

    def may_lend(self, args: dict) -> bool:
        """Whether a call may run on the text model's GPU, lent to the images: the images have a card of their own
        (ImageModels.separate), the setting (the config's "use_text_gpu", else the request's xestrata_images
        use_text_gpu) is on, or ask with the call's use_text_gpu; and no other request's text is being written or waits
        (a lent GPU makes them wait)."""
        models = self.models or getattr(self.svc, "images", None)
        if not isinstance(models, ImageModels) or not models.separate or models.lend_cfgs is None:
            return False
        mode = self.text_gpu
        if mode == "off" or mode == "ask" and args.get("use_text_gpu") is not True:
            return False
        eng = getattr(self.svc, "engine", None)
        if eng is None or not hasattr(eng, "unload"):
            return False
        status = getattr(self.svc, "status", None) or {}
        return not status.get("queued") and not getattr(self.svc, "live_reqs", None)

    def _file(self, name: str) -> bytes | None:
        p = (self.models or self.gen).file(name)
        return p.read_bytes() if p is not None else None

    def attached(self) -> list[bytes]:
        """The pictures in the user's last message: an edit takes them as its reference images after the one it edits
        (as Qwen-Image's edit takes a second and a third picture beside the first), so "the face from the picture I
        attached" reaches the image model - before, only the image edited did, and the prompt alone named the face."""
        for msg in reversed(self.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            return [self.load(item["source"]) for item in (content if isinstance(content, list) else [])
                    if isinstance(item, dict) and item.get("type") == "image" and item.get("source")]
        return []

    def has_image(self) -> bool:
        """Is there an image an edit could take (one this request made, a picture in the conversation, or a link to a
        kept one)?  Nothing is read."""
        if self.made or self.user.get("image") is not None:
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
        """The image an edit takes: a /v1/images/files/ link (or its file name), attached:N, else the conversation's
        last."""
        named = self._attached_named(which)
        if named is not None:
            return named
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

    def _attached_named(self, which: str) -> bytes | None:
        """attached:N, the user's N-th picture in their last message (the name the tools' descriptions give it)."""
        m = re.fullmatch(r"\s*attached:(\d+)\s*", which or "")
        pics = self.attached() if m else []
        return pics[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(pics) else None

    def _linked(self, which: str, arg: str) -> bytes:
        """A picture an argument names by its link (a tool's result, or one in the conversation) or as attached:N: its
        bytes, or an ImageRequestError that says which argument and what it may name."""
        data = self._attached_named(which)
        if data is not None:
            return data
        m = FILE_REF.search(which or "") or re.fullmatch(r"([0-9a-f]{32}\.(?:png|jpeg|webp|jxl))", which or "")
        data = self._file(m.group(1)) if m else None
        if data is None:
            n = len(self.attached())
            raise ImageRequestError(f"{arg}: {which!r} is not an image here: give a link a result gave"
                                    + (f", or the user's pictures attached:1 to attached:{n}" if n else ""))
        return data

    def _video(self, which: str):
        """The video upscale_video takes: a /v1/videos/files/ link, else the conversation's last (the chat's page
        attaches a video as its link) -> its file, or None (none, or the server keeps no video tools)."""
        from serve import videos
        if getattr(self.svc, "images", None) is None or not videos.available():
            return None
        jobs = videos.jobs_of(self.svc)
        m = videos.FILE_REF.search(which or "")
        if m:
            return videos.file_path(jobs, m.group(1))
        for msg in reversed(self.messages):
            content = msg.get("content")
            for item in reversed(content if isinstance(content, list) else [content]):
                text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
                for name in reversed(videos.FILE_REF.findall(text or "")):
                    if (f := videos.file_path(jobs, name)) is not None:
                        return f
        return None

    def _train_lora_tool(self, args: dict, t0: float) -> dict:
        """train_lora: the panel opens in the chat (the page renders it by the tool's name); no job starts here.
        The answer tells the model what the user can do there, with the models this server can train."""
        from serve import lora_train
        all_rows = lora_train.model_rows(self.svc)
        rows = [r for r in all_rows if r["can_train"]]
        requested = str(args.get("model") or "")
        target, choice = lora_train.training_model(requested)
        row = next((r for r in all_rows if r["model"] == target), None) if requested else None
        if requested and (row is None or not row["can_train"]):
            why = (row["why"] if row else
                   f"install the base with {lora_train.setup_command(self.svc, choice)} and restart the server"
                   if choice else f"{target} is not configured on this server")
            text = f"Cannot prepare training for {requested}: {why}"
            return {"ok": False, "text": text, "chars": len(text), "truncated": False,
                    "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": "train_lora", "images": [],
                    "requested_model": requested, "training_model": target, "model_switched": bool(choice)}
        names = ", ".join(f"{r['name']} ({r['model']})" for r in rows) or "none ready (the trainer is not built)"
        text = ("The LoRA training panel is open in the chat. The user puts the training pictures in (drag and drop, "
                "or picks this chat's images), chooses the kind (character, style, other), the name, the trigger "
                f"word and the model ({names}), captions the pictures (the auto button writes them; they can be "
                "edited), sets the steps and how often a version is saved, and starts the training. It runs on the "
                "server; the panel shows its progress and, when it ends, a sample picture of every saved version "
                "side by side. The user picks a version and registers it as a LoRA (the last one by default); it is "
                "usable in the chat's images at once. Tell the user so and answer their questions; do not wait for "
                "the training and do not call this tool again for the same request.")
        if choice:
            text += (f" {requested} is distilled for fewer steps and is not suitable for training. "
                     f"The panel selects {target} instead. Tell the user; the resulting LoRA can also be used "
                     f"with {requested}.")
        return {"ok": True, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": "train_lora", "images": [],
                "requested_model": requested, "training_model": target, "model_switched": bool(choice)}

    def _video_tool(self, args: dict, t0: float) -> dict:
        """upscale_video: a /v1/videos job on the video (videos.start_from_file), its id and where its video will be
        the call's result at once; the job runs on the server past this answer."""
        from serve import videos
        text, ok = "", False
        try:
            f = self._video(str(args.get("video") or ""))
            if f is None:
                raise ImageRequestError("there is no video in the conversation (or it is no longer kept: attach it "
                                        "again)")
            method, variant = str(args.get("method") or "auto"), args.get("variant")
            model = f"seedvr2-{variant}" if method in ("seedvr2", "auto") and variant else method
            if model not in UPSCALE_MODELS and model not in UPSCALE_METHODS:
                model = method
            fields = {"model": model, **{k: args[k] for k in ("scale", "seed") if args.get(k) not in (None, "")}}
            rec = videos.start_from_file(self.svc, videos.jobs_of(self.svc), f, fields, self.gen.cfg)
            src = rec["source"]
            text = (f"The upscale runs on the server as job {rec['id']}: {src['frames']} frames, "
                    f"{src['width']}x{src['height']} to {rec['size']}, with {model}. It takes minutes or more; the "
                    f"user follows it in the Video tab, and the video will be at /v1/videos/{rec['id']}/content once "
                    "done. Tell the user so; the job goes on whatever this chat does.")
            ok = True
        except (ValueError, ImageJobFailed, videos.VideoRequestError) as e:
            text = f"error: {e}"
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": "upscale_video", "images": []}

    def _segment_video_tool(self, args: dict, t0: float) -> dict:
        """segment_video: a segmentation job on the video (videos.start_from_file), the overlay and the answer's links
        the call's result at once; the job runs on the server past this answer."""
        from serve import videos
        text, ok = "", False
        try:
            f = self._video(str(args.get("video") or ""))
            if f is None:
                raise ImageRequestError("there is no video in the conversation (or it is no longer kept: attach it "
                                        "again)")
            prompts = args.get("prompts") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            fields = {"prompts": json.dumps([str(p) for p in prompts]), "outputs": json.dumps(["overlay", "mask"]),
                      "mask_video": "true" if args.get("masks") else "false"}
            check_tool(self.gen.cfg, "segment_video")
            rec = videos.start_from_file(self.svc, videos.jobs_of(self.svc), f, fields, self.gen.cfg, "segment")
            src = rec["source"]
            base = f"/v1/videos/{rec['id']}/content"
            text = (f"The segmentation runs on the server as job {rec['id']}: {src['frames']} frames, finding "
                    f"{', '.join(map(str, prompts))}. The user follows it in the Video tab; once done the video with "
                    f"the objects colored over it will be at {base}?variant=overlay, one mask video of them all at "
                    f"{base}?variant=mask" + (f", each object's at {base}?variant=object_<id>" if args.get("masks")
                                              else "") +
                    f", and every frame's objects (vLLM-Omni's JSON) at {base}. Tell the user so; the job goes on "
                    "whatever this chat does.")
            ok = True
        except (ValueError, ImageJobFailed, videos.VideoRequestError) as e:
            text = f"error: {e}"
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": "segment_video", "images": []}

    def _aux_tool(self, name: str, args: dict, t0: float, cancel: threading.Event | None) -> dict:
        """find_region (the region marked on the image: image_regions.segment) and remove_background (BiRefNet's
        cut-out, image_regions.matte): an image shown to the user, nothing drawn by the image model."""
        from serve import image_regions
        urls: list[str] = []
        seen = None
        try:
            data = self._image(str(args.get("image") or ""))
            if data is None:
                raise ImageRequestError(f"there is no image in the conversation to {name.split('_')[0]} in")
            gens = self._gens()
            gen = gens[0] if gens else self.gen
            check_tool(gen.cfg, name)
            picture = self.normalize(data)
            if name == "find_region":   # `prompt` read as the words too: the name a chat model reaches for first
                region = str(args.get("region") or args.get("prompt") or "").strip()
                got = image_regions.segment(self.svc, gen, picture, region, "mask", cancel=cancel)
                png = image_regions.marked(picture, got["instances"])
                url = f"/v1/images/files/{gen.store(png, 'png')}"
                union = image_regions.union_mask(got)
                mask_url = (f"/v1/images/files/{gen.store(mask_to_bw(union[0]), 'png')}" if union is not None
                            else None)
                scores = [i["score"] for i in got["instances"] if i["score"] is not None]
                text = (f"{len(got['instances'])} {region!r} region(s) found by {got['model']}"
                        + (f" (scores {', '.join(f'{s:.2f}' for s in scores)})" if scores else "")
                        + (f" ({got['fallback']})" if got.get("fallback") else "")
                        + f"; the user sees them marked in red: {url}."
                        + (f" Their mask (white: the regions): {mask_url}, for edit_image's mask." if mask_url else
                           " Boxes only, no mask: edit_region with the same words draws the part."))
                if not got["instances"]:
                    text = f"no {region!r} found by {got['model']}: try other words."
            else:
                version = image_regions.birefnet_version(args.get("version"))
                bg = image_regions.parse_color(args.get("background_color"))
                got = image_regions.matte(self.svc, gen, picture, version, None, True, cancel)
                png = encode_pillow(image_regions.cut_out(picture, got, bg), "png")
                url = f"/v1/images/files/{gen.store(png, 'png')}"
                text = (f"The background is {'replaced' if bg else 'removed'} (BiRefNet {got['version']}, "
                        f"{got['seconds']:.1f} s"
                        + (f"; {got['fallback']}" if got.get("fallback") else "")
                        + f") and the user sees it: {url}.")
            self.made.append(url.rsplit("/", 1)[1])
            urls = [url]
            seen = self._seen(gen, png)
            ok = True
        except AuxPending as e:
            failed = any(d.get("status") == "failed" for d in e.downloads)
            text = f"error: {e}" if failed else f"downloading: {e}"
            return {"ok": False, "text": text, "chars": len(text), "truncated": False,
                    "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": [],
                    "download": [{k: d.get(k) for k in AUX_FIELDS if d.get(k) is not None} for d in e.downloads]}
        except (ValueError, ImageJobFailed) as e:
            text, ok = f"error: {e}", False
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                **({"model_content": [
                    {"type": "text", "text": text + " A small copy is attached: check that it shows what was asked."},
                    {"type": "image", "source": str(seen)}]} if seen is not None else {}),
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": urls}

    def call(self, name: str, arguments, cancel: threading.Event | None = None) -> dict:
        t0 = time.monotonic()
        if name not in self.allowed:   # checked as it runs, not only by the tools offered
            text = (f"error: not run: {name} is not one of this request's image tools "
                    f"({', '.join(sorted(self.allowed))})")
            return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                    "tool": name, "images": []}
        if self.max_calls is not None and self.calls >= self.max_calls:
            text = f"error: not run: this response may make {self.max_calls} image tool call(s) (max_tool_calls)"
            return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                    "tool": name, "images": []}
        args = dict(arguments) if isinstance(arguments, dict) else {}
        args.pop("plan", None)   # the chart, put into the reply by run_with_mcp
        key = (name, json.dumps({k: v for k, v in args.items() if k not in ("step", "group")}, sort_keys=True,
                                ensure_ascii=False))
        done = self.done_calls.get(key)
        if done is not None:   # the same call ran in this request and is done: not again (a model's loop)
            text = (f"not run: this same {name} call already ran in this request and is done ({done}). Do not call "
                    "it again: answer the user, or take the next step.")
            return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                    "tool": name, "images": [], "skipped": True}
        if name not in self.unlocked and name not in CORE_TOOLS and name != "train_lora" and not self.all_tools:
            answer = self._stub_answer(name, args, t0)   # a stub's call: its arguments, or it runs as it is
            if answer is not None:
                return answer
        merged = args.pop("_n", None)        # tool_parallel: generate_image's calls that differ in their seed alone
        merged = merged if isinstance(merged, int) and not isinstance(merged, bool) and merged > 1 else None
        lend = args.pop("_lend", None) is True   # tool_parallel: on the text model's GPU, lent to the images
        self.calls += merged or 1
        if name == "train_lora":
            return self._train_lora_tool(args, t0)
        if name in ("find_region", "remove_background", "upscale_video", "segment_video"):
            r = (self._aux_tool(name, args, t0, cancel) if name in ("find_region", "remove_background") else
                 self._video_tool(args, t0) if name == "upscale_video" else self._segment_video_tool(args, t0))
            if r["ok"]:
                self.done_calls[key] = (r.get("images") or ["done"])[0]
            return r
        urls: list[str] = []
        found: dict | None = None
        try:
            req = {**{k: args[k] for k in ("prompt", "size", "seed") if args.get(k) not in (None, "")}, **self.options}
            if self.user.get("layout") is not None:   # the user's boxes, at their size
                w, h = self.user["size"]
                req["size"] = f"{w}x{h}"
                args = {**args, "layout": self.user["layout"]}
            if args.get("layout") not in (None, "", {}):   # boxes, the user's or the model's: a MOSS-Image model
                moss = [g.model for g in self._gens() if _pipeline(g.family) == "latent_chat"]
                if moss and args.get("model") not in moss:
                    args = {**args, "model": moss[0]}
            if args.get("layout") not in (None, "", {}):   # MOSS-Image's boxes
                layout = args["layout"]
                if isinstance(layout, str):
                    try:
                        layout = json.loads(layout)
                    except ValueError:
                        raise ImageRequestError("layout: a JSON object") from None
                w, h = parse_size(req.get("size"), 2048)
                req["layout"] = tidy_layout(layout, w, h)
            if name in ("edit_image", "edit_region") and args.get("strength") not in (None, ""):
                req["strength"] = args["strength"]
            if name == "edit_image" and args.get("edit_adapter") not in (None, ""):
                req["edit_adapter"] = args["edit_adapter"]
            if name == "edit_image" and args.get("refine") not in (None, ""):
                req["refine"] = args["refine"]
            if name in ("generate_image", "edit_image") and args.get("guidance") not in (None, "", []):
                req["guidance"] = args["guidance"]
            if name in ("generate_image", "edit_image") and args.get("lora") not in (None, "", []):
                req["lora"] = args["lora"]   # serve/loras.py's field: the model's names, strengths, block weights
            if name in ("edit_image", "edit_region", "fix_image") and (args.get("crop_upscale") or args.get("variant")):
                req["crop_upscale"] = {k: v for k, v in (("method", args.get("crop_upscale")),
                                                         ("variant", args.get("variant"))) if v}
            if name == "upscale_image":
                req["upscale"] = {k: args[k] for k in ("scale", "method", "redraw", "strength", "variant")
                                  if args.get(k) not in (None, "")}
            if name == "fix_image":
                req["fix"] = {"parts": args.get("parts") or ["face"],
                              **({"strength": args["strength"]} if args.get("strength") not in (None, "") else {})}
            if name == "extend_image":
                req["extend"] = ({"aspect": args["aspect"]} if args.get("aspect") not in (None, "") else
                                 {k: args[k] for k in ("left", "top", "right", "bottom")
                                  if args.get(k) not in (None, "")})
            gens = self._gens(name in EDIT_TOOLS)
            if name == "control":
                gens = [g for g in gens if _controls().capability(g.cfg)]
            asked = args.get("model")
            gen = next((g for g in gens if g.model == asked), gens[0] if gens else self.gen)
            if asked not in (None, "") and gen.model != asked:
                raise ImageRequestError(f"model: one of {', '.join(g.model for g in gens)}, not {asked!r}")
            if lend:   # the same model's engine on the text model's GPU (ImageModels.lent)
                gen = self.svc.images.lent().gens[gen.model]
            if merged:
                req["n"] = merged
            if req.get("guidance") not in (None, "", []) and not guidance_methods(gen.cfg):
                takes = [g.model for g in gens if guidance_methods(g.cfg)]
                raise ImageRequestError(f"guidance: {gen.model} takes none" + (
                    f"; give model {' or '.join(takes)} for it, or leave guidance out" if takes else
                    "; leave guidance out"))
            if name == "generate_image":
                job = parse_request(req, gen.cfg)
            elif name in ("extend_image", "upscale_image", "fix_image"):
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError(f"there is no image in the conversation to "
                                            f"{name.split('_')[0]}")
                job = parse_edit_request(req, [self.normalize(data)], gen.cfg)
            elif name == "edit_image":
                mask = self.mask
                if self.user.get("image") is not None:   # the image the user painted over or drew boxes on
                    data = self.user["image"]
                    if self.user.get("mask") is not None:
                        mask = fit_mask(self.user["mask"], self.normalize(data))
                else:
                    data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no image in the conversation to edit")
                images = [self.normalize(data)]
                if args.get("mask") not in (None, ""):   # a mask a tool made (find_region's, edit_region's)
                    if mask is not None:
                        raise ImageRequestError("mask: the user painted this edit's mask already; leave mask out")
                    m = self._linked(args["mask"], "mask")
                    mask = fit_mask(mask_from_bw(m), images[0])
                refs = args.get("references")
                if isinstance(refs, str):
                    refs = [refs]
                if refs not in (None, []) and not (isinstance(refs, list) and all(isinstance(r, str) for r in refs)):
                    raise ImageRequestError("references: a list of /v1/images/files/ links")
                if refs and not can_edit(gen.cfg):
                    raise ImageRequestError(f"references: {gen.model} has no edits (img2img takes one image)"
                                            + (f"; give model {self._gens(True)[0].model}"
                                               if can_edit(self._gens(True)[0].cfg) else ""))
                if can_edit(gen.cfg):   # the references named, else the user's attached pictures (attached)
                    limit = refs_limit(gen.cfg)
                    more = ([self.normalize(self._linked(r, "references")) for r in refs] if refs else
                            [self.normalize(p) for p in self.attached() if p != data])
                    if refs and len(more) > limit - 1:
                        raise ImageRequestError(f"references: {gen.model} takes at most {limit - 1}")
                    images += more[:max(0, limit - 1)]
                job = parse_edit_request(req, images, gen.cfg, mask)
            elif name == "edit_region":   # the words' mask (image_regions: SAM 3 or Florence-2), then a masked edit
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no image in the conversation to edit")
                picture = self.normalize(data)
                if args.get("mask") not in (None, ""):   # a mask a tool gave (find_region's) instead of words
                    mask = fit_mask(mask_from_bw(self._linked(str(args["mask"]), "mask")), picture)
                    found = None
                else:
                    if not str(args.get("region") or "").strip():
                        raise ImageRequestError("region: the words of the part to change (or mask: a mask's link)")
                    from serve import image_regions
                    mask, found = image_regions.region_mask(self.svc, gen, picture, str(args["region"]),
                                                            cancel=cancel)
                job = parse_edit_request(req, [picture], gen.cfg, mask)
            elif name == "control":   # a new image by a picture's composition (image_controls)
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no picture in the conversation to take the composition from")
                req["control"] = {k: args[k] for k in ("type", "strength", "end") if args.get(k) not in (None, "")}
                job = _controls().parse_job(req, gen.cfg, picture=self.normalize(data))
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
                    f"{figures['seconds']:.0f} s) and the user sees it: {urls[0]}. This step is done.")
            if found is not None:   # where edit_region drew, and its mask (another tool may take it)
                mask_url = f"/v1/images/files/{gen.store(mask_to_bw(job['mask']), 'png')}"
                text += (f" edit_region redrew {len(found['instances'])} {args['region']!r} region(s) that "
                         f"{found['model']} found" + (f" ({found['fallback']})" if found.get("fallback") else "")
                         + f"; their mask: {mask_url}.")
            if job.get("fix"):   # what the parts' fix found and drew
                fx = figures["fix"]
                regs = fx["regions"]
                drawn = [r for r in regs if "skipped" not in r]
                text += (f" fix_image redrew {len(drawn)} part(s)"
                         + (f" ({', '.join(r['part'] for r in drawn)})" if drawn else
                            f": no {', '.join(job['fix']['parts'])} found, the image is as it was")
                         + "".join(f"; skipped {r['part']} ({r['skipped']})" for r in regs + fx["left_out"]
                                   if "skipped" in r) + ".")
            if job.get("crop_ups"):   # how the masked edit's cut was enlarged (crop_upscale)
                c = job["crop_ups"]
                text += (f" The cut was enlarged {c['scale']:g} times by {c['method']}"
                         + (f" {c['variant']}" if c.get("variant") else "") + seedvr2_3b_note(c) + ".")
            if job.get("upscaler"):   # how it was made larger
                u = job["upscaler"]
                text += (f" upscale_image made it larger by {u['method']}"
                         + (f" {u['variant']}" if u.get("variant") else "") + seedvr2_3b_note(u)
                         + (f" (auto's choice; {u['fallback']})" if u.get("fallback") else
                            " (auto's choice)" if u["asked"] == "auto" else "")
                         + (f", then drew it again at strength {job['strength']:g}." if not job.get("ups_only")
                            else ", not drawn again."))
            if job.get("control"):   # how the composition was taken
                cf = job["control"].get("figures") or {}
                text += (f" control took the picture's {job['control']['type']} map ({cf.get('size', '?')}) "
                         + ("through ControlNet." if cf.get("mode") == "controlnet" else
                            "through MOSS-Image's color and depth adapter." if cf.get("mode") == "moss" else
                            "as a reference image (this model has no ControlNet)."))
                made_map = job["control"].get("map_png")
                if made_map is not None and job["control"]["type"] != "none":
                    map_url = f"/v1/images/files/{gen.store(made_map, 'png')}"
                    text += f" The map: {map_url} (control type none takes it as it is)."
            seen = self._seen(gen, images[0])
            ok = True
        except AuxPending as e:   # the tool's weights are downloading: not an image, not an error to retry now
            failed = any(d.get("status") == "failed" for d in e.downloads)
            text, ok, seen = (f"error: {e}" if failed else f"downloading: {e}"), False, None
            return {"ok": False, "text": text, "chars": len(text), "truncated": False,
                    "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": [],
                    "download": [{k: d.get(k) for k in AUX_FIELDS if d.get(k) is not None} for d in e.downloads]}
        except (ValueError, ImageJobFailed) as e:
            text, ok, seen = f"error: {e}", False, None
        out: dict[str, Any] = {
            "ok": ok, "text": text, "chars": len(text), "truncated": False,
            # a small copy of the image for a chat model that reads pictures (run_with_mcp passes it with one):
            # it sees whether the image shows what was asked, and can fix it with the tools
            **({"model_content": [
                {"type": "text", "text": text + " A small copy is attached: check that it shows what was asked."},
                {"type": "image", "source": str(seen)}]} if seen is not None else {}),
            "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": urls,
            **({"image": {"size": f"{job['width']}x{job['height']}", "quality": job["quality"],
                          "background": job["background"], "output_format": job["output_format"],
                          "revised_prompt": model_prompt(job)}} if ok else {})}
        if ok:
            self.done_calls[key] = urls[0] if urls else "done"
        if merged and ok and len(urls) == merged:   # each merged call its own image, seed and copy to look at
            parts = []
            for i, u in enumerate(urls):
                t = (f"The image is made{by} ({job['width']}x{job['height']}, seed {job['seed'] + i}) and the user "
                     f"sees it: {u}. This step is done.")
                copy = self._seen(gen, images[i])
                parts.append({**out, "text": t, "chars": len(t), "images": [u],
                              **({"model_content": [{"type": "text", "text": t + " A small copy is attached."},
                                                    {"type": "image", "source": str(copy)}]} if copy else {})})
            out["split"] = parts
        return out


# ------------------------------------------------------------------------------------------- the GPU turn
def ledger_fit(gen: ImageGen, need: dict, job: dict | None = None) -> int:
    """The image engine's budget within the GPU's VRAM ledger (serve/vram_ledger.py, shared with the translation
    engine): what it may take besides what it holds is the ledger's room (an engine of the ledger's may hold VRAM the
    device's report does not show yet), and its entry (ImageGen.ledger_sync's) set to the budget for the job, with when
    it should end (the speed measured: ImageGen.estimate_s) -> the budget.  The free VRAM is read now, after the text
    engine made room (`need`'s was read before)."""
    try:
        got = gen.memory()
    except ImageJobFailed:
        got = {}
    led = gen.ledger(int(got.get("total_mib") or 0))
    free = int(got.get("free_mib", need.get("free_mib")) or 0)
    held = int(got.get("held_mib", need.get("held_mib")) or 0)
    owner = gen._owner()
    if held > 0:          # what it holds now, before the room is read: its own MiB are not room for it
        led.hold("image", held, owner, whole=True)
    room = led.room(free)
    extra = max(0, free - gen.reserve - room) if room is not None else 0
    budget = gen.fit(extra)
    est = None
    if job is not None and not job.get("aux") and not job.get("video"):
        with contextlib.suppress(KeyError, TypeError, ValueError, ZeroDivisionError):
            est = gen.estimate_s(job)
    if budget > 0:
        led.hold("image", max(budget, held), owner, exact=False, until=time.time() + est if est else None,
                 whole=True)
    return budget


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
    info = getattr(eng, "info", None) or {}
    if "vram_elastic" in info:   # the engine's own word: a runtime that keeps an unmapped segment's VRAM says 0
        return bool(info["vram_elastic"])
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
    if job.get("pic_paths") or (job.get("edit_format", "").startswith("ai-toolkit") and job.get("refs")):
        return   # its text encoder sees the pictures: the image engine's, with the vision encoder
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
    was made (the response's "xestrata" carries it; the image's revised_prompt is the prompt the model read).  An answer
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


def _controls():
    """serve/image_controls.py (it imports this module)."""
    from serve import image_controls
    return image_controls


def make_images(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None,
                preview=None) -> tuple[list[bytes], dict, str]:
    """`job`'s images in the GPU's turn (gpu_turn) -> (the PNG files' bytes, figures, the policy used).  When a request
    waits for the turn after an image, the turn is given to it and taken again for the rest, so a chat is not held up
    by a batch; not under unload (giving it would load the text model and unload it again for each image) nor on a card
    of its own (separate: the chat does not wait for it).  Too many
    images for budget_s are refused first (ImageGen.check_n).  MOSS-Image's prompt enhancer (enhance_prompt) runs
    first, in the chat model's turn."""
    if job.get("control"):   # the control map first (its own GPU turn), then the model's way of taking it
        _controls().prepare(svc, gen, job, cancel)
    if job.get("fix"):
        return fix_images(svc, gen, job, cancel)
    if job.get("upscaler"):
        upscaler_weights(gen, job)
        seedvr2_or_realesrgan(gen, job)
    if job.get("crop_ups"):   # crop_upscale's upscaler
        upscaler_weights(gen, job, "crop_ups")
    if job.get("ups_only"):
        return upscale_images(svc, gen, job, cancel)
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


def seedvr2_file(gen: ImageGen, variant: str) -> str:
    """The AUX name of SeedVR2's DiT a request takes: the variant's FP16 file where the GPU's memory holds it beside its
    work, else its int8_convrot (the blocks the VRAM does not hold read from the RAM); the 3B where the 7B's int8 fits
    neither the VRAM nor the RAM.  A file already here is taken before one to fetch, where it fits.  Where the engine
    times its 8 bits the faster on this GPU (takes_w8), a file to fetch is the int8_convrot whatever the VRAM: the
    engine would make the FP16 file's weights 8-bit as it opens it (slower to open, the 8-bit rows kept in the RAM),
    the int8 file holds them so in half the bytes.  An FP16 file already here is still taken (made 8-bit on opening)."""
    q = image_aux._qimage()
    root = image_aux.aux_dir(gen.cfg)
    try:   # the GPU's whole memory, not what is free now: the choice the same from request to request
        vram = gen.total_mib()
    except ImageJobFailed:
        vram = 0
    ram = ram_available_mib() or 0

    def mib(name: str) -> int:
        return q.FILES[name][2] >> 20

    def int8_instead(fp16: str, which: str) -> bool:
        """Whether the int8_convrot file is taken where the FP16 one fits: it is not here and the GPU takes 8 bits."""
        if image_aux.ready(fp16, root) or not gen.takes_w8():
            return False
        gen._say(f"image job: SeedVR2 {which}'s int8_convrot file, not the FP16 one the VRAM ({vram} MiB) would hold: "
                 "this GPU runs its DiT faster in 8 bits")
        return True
    if variant != "3b":
        fp16, int8 = f"seedvr2-{variant}-fp16", f"seedvr2-{variant}-int8"
        if vram >= mib(fp16) + SEEDVR2_WORK_MIB and (image_aux.ready(fp16, root) or not image_aux.ready(int8, root)) \
                and not int8_instead(fp16, variant):
            return fp16
        if vram + ram >= mib(int8) + SEEDVR2_WORK_MIB:
            return int8
        gen._say(f"image job: SeedVR2 {variant}'s 8 bits fit neither the VRAM ({vram} MiB) nor the RAM ({ram} MiB): "
                 "its 3B")
    fp16 = "seedvr2-3b-fp16"
    return fp16 if vram >= mib(fp16) + SEEDVR2_WORK_MIB and not int8_instead(fp16, "3b") else "seedvr2-3b-int8"


def seedvr2_3b_note(u: dict) -> str:
    """What a result says when SeedVR2's 7B did not fit and its 3B ran instead (seedvr2_file)."""
    return (" (its 3B: the 7B fits neither this GPU's memory nor the RAM)"
            if str(u.get("file") or "").startswith("seedvr2-3b") and u.get("variant") != "3b" else "")


def upscaler_weights(gen: ImageGen, job: dict, key: str = "upscaler") -> None:
    """An upscale's upscaler weights (image_aux's, UPSCALE_WEIGHTS; SeedVR2's DiT by seedvr2_file and its VAE beside
    it): their path into job[key] ("upscaler", or "crop_ups": the masked edit's cut's), or AuxPending while they
    download (the upscale is not made with another method meanwhile)."""
    u = job[key]
    names = [seedvr2_file(gen, u.get("variant") or "7b"), "seedvr2-vae"] if u["method"] == "seedvr2" else \
        [UPSCALE_WEIGHTS[u["method"]]] if u["method"] in UPSCALE_WEIGHTS else []
    if not names:
        return
    root = image_aux.aux_dir(gen.cfg)
    got = [image_aux.ensure(name, root, gen._say) for name in names]
    pending = [g for g in got if g["status"] != "ready"]
    if pending:
        raise AuxPending(pending)
    u["path"] = got[0]["path"]
    if u["method"] == "seedvr2":
        u["file"] = names[0]


def seedvr2_or_realesrgan(gen: ImageGen, job: dict) -> None:
    """auto's SeedVR2 where the engine says it does not fit the GPU (UPSNEED's least past its cap): RealESRGAN instead,
    said in the result (job["upscaler"]["fallback"])."""
    u = job["upscaler"]
    if u["asked"] != "auto" or u["method"] != "seedvr2":
        return
    try:
        need = gen.need(job) if job.get("ups_only") else None
    except ImageJobFailed:
        need = None
    if need is not None and int(need.get("least_mib", 0)) <= int(need.get("cap_mib", 0)):   # UPSNEED's cap: the GPU's
        return
    gen._say("image job: SeedVR2 does not fit the GPU: RealESRGAN instead")
    u.update(method="realesrgan", fallback="SeedVR2 does not fit the GPU's memory")
    u.pop("variant", None)
    u.pop("path", None)
    u.pop("file", None)
    upscaler_weights(gen, job)


def upscale_images(svc, gen: ImageGen, job: dict,
                   cancel: threading.Event | None = None) -> tuple[list[bytes], dict, str]:
    """An upscale without a drawing (job["ups_only"]) in the GPU's turn: the engine's upscaler alone, no image model
    run (its cache given back where the upscaler needs the room)."""
    t0 = time.monotonic()
    with gpu_turn(svc, gen, job, cancel) as policy:
        png, got = gen.upscale(job, cancel)
    u = job["upscaler"]
    figures = {"seconds": round(time.monotonic() - t0, 2), "n": 1, "seed": job["seed"], "engine": "xestrata-image",
               "size": f"{job['width']}x{job['height']}", "text_encode_ms": 0.0, "sampling_s": 0.0,
               "vae_decode_s": 0.0,
               "upscale": {"method": u["method"], "asked": u["asked"], "color": got.get("color", "none"),
                           **{k: u[k] for k in ("variant", "file", "fallback") if u.get(k)},
                           **{k: float(got[k]) for k in ("upscale_s", "model_s", "resize_s", "color_s") if k in got},
                           **{k: int(got[k]) for k in ("tile", "tiles") if k in got}}}
    gen.last = figures
    gen.end(None)
    return [png], figures, policy


def crop_ups_names(gen: ImageGen, v) -> list[str]:
    """The weights crop_upscale `v` may take (crop_upscaler's choice, before the cut is known)."""
    method = (v.get("method") if isinstance(v, dict) else v) or "auto"
    variant = (v.get("variant") if isinstance(v, dict) else None) or "7b"
    if method == "seedvr2" or isinstance(v, dict) and v.get("variant"):
        return [seedvr2_file(gen, variant), "seedvr2-vae"]
    return [UPSCALE_WEIGHTS["realesrgan"]] if method in ("auto", "realesrgan") else []


def fix_job(gen: ImageGen, job: dict, picture: bytes, region: dict, seed: int) -> dict:
    """One fixed region's img2img: the picture (as it is after the regions before) with the region's mask, drawn
    at the fix's strength in the region's crop (the cut, img_crop.hpp), the image at the picture's size.  A
    model that edits draws it as img2img too, from the cut and the prompt without the picture as a reference (an
    edit's reference would add the vision encoder's pass and its tokens to every
    step)."""
    req = {**job["fix_req"], "strength": job["fix"]["strength"], "seed": seed, "n": 1}
    one = parse_edit_request(req, [picture], {**gen.cfg, "edit_lora": None, "vision": None},
                             region["mask"])
    one.update(crop=region["crop"], width=job["width"], height=job["height"], n=1, seed=seed)
    one.pop("refine", None)
    one.pop("crop_ups", None)
    crop_upscaler(one, req)   # the region's own cut and enlargement
    if one.get("crop_ups"):
        upscaler_weights(gen, one, "crop_ups")
    if gen.cfg.get("schedule") == "turbo":
        # the Turbo's own sigmas put half its steps above 0.89 (its schedule is a picture's from noise): its second half
        # redraws a face anew (the expression lost, 2026-11-05).  The base model's schedule (base_sigmas, diffusers'
        # Qwen-Image-2.1 one) starts its second half at 0.58: the face drawn finer, as it was
        one["sigmas"] = base_sigmas(one["steps"], *region["crop"]["gen"])
    return one


def fix_images(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None) -> tuple[list[bytes], dict, str]:
    """The parts' fix (fix_image, the edits' `fix`): the picture's parts found once
    (the engine's YOLOv8 detectors, image_aux's weights: AuxPending while they download), then each region (fix_regions,
    largest first) drawn again in turn on the picture as the regions before left it, in one GPU turn.  Region k's seed
    is the job's + k (fixed after the detection, whatever is skipped), candidate i's the job's + 1000 i.  budget_s is
    checked before each region, from the turn's start (the detection included): past it the rest are skipped and said
    so; a region running is not cut short.  No region found gives the picture as it is."""
    root = image_aux.aux_dir(gen.cfg)
    downloads = [image_aux.ensure(FIX_PARTS[p], root, gen._say) for p in job["fix"]["parts"]]
    # crop_upscale's upscaler (auto's RealESRGAN where a region is enlarged enough): fetched before the turn, as the
    # detectors are, since a region's own enlargement is known only once it is found
    ups = crop_ups_names(gen, job["fix_req"].get("crop_upscale"))
    pending = [d for d in downloads + [image_aux.ensure(n, root, gen._say) for n in ups] if d["status"] != "ready"]
    if pending:
        raise AuxPending(pending)
    picture = as_png(job["picture"])
    W, H = image_dims(picture)
    budget = float(gen.cfg.get("budget_s") or BUDGET_S)
    # the turn's need: a region drawn at the size the largest takes (about 1 MP), the picture its base
    probe = fix_job(gen, job, picture, {"mask": box_mask(W, H, (W // 4, H // 4, W - W // 4, H - H // 4)),
                                        "crop": crop_whole(gen.cfg, W, H, FIX_BLEND)}, job["seed"])
    images: list[bytes] = []
    report: list[dict] = []
    totals = {"seconds": 0.0, "text_encode_ms": 0.0, "sampling_s": 0.0, "vae_decode_s": 0.0}
    t0 = time.monotonic()
    used = "keep"
    detect_s = 0.0
    last: dict = {}
    left: list[dict] = []
    with gpu_turn(svc, gen, probe, cancel) as policy:
        used = policy
        found = []
        for part in job["fix"]["parts"]:
            dets, secs = gen.detect(Path(downloads[job["fix"]["parts"].index(part)]["path"]), picture, FIX_THRESHOLD,
                                    masks=part == "person")
            detect_s += secs
            found += [{**d, "part": part} for d in dets]
        regions, left = fix_regions(found, W, H, gen.cfg)
        gen.fit()   # the budget again, the detectors' weights now held
        gen._say(f"image job: fix {', '.join(job['fix']['parts'])}: {len(regions)} region(s) to draw, "
                 f"{len(left)} left out, detection {detect_s:.2f} s")
        for i in range(job["n"]):
            current = picture
            done = []
            for k, region in enumerate(regions):
                seed = job["seed"] + 1000 * i + k
                entry = {"part": region["part"], "box": region["box"], "score": region["score"], "seed": seed,
                         "crop": list(region["crop"]["box"]), "made_size": "{}x{}".format(*region["crop"]["gen"])}
                if cancel is not None and cancel.is_set():
                    raise ImageJobFailed("cancelled (the client left)")
                if time.monotonic() - t0 > budget:
                    done.append({**entry, "skipped": f"budget_s {budget:g} s"})
                    continue
                one = fix_job(gen, job, current, region, seed)
                made, fig = gen.run(one, cancel)
                current = made[0]
                last = fig
                for key in totals:
                    totals[key] += fig.get(key, 0)
                done.append({**entry, "seconds": fig["seconds"]})
            images.append(current)
            report.append({"regions": done})
    gen.end(None)
    figures = {**{k: v for k, v in last.items() if k not in ("crop", "made_size", "mask")}, **totals,
               "seconds": round(time.monotonic() - t0, 2), "n": len(images), "seed": job["seed"],
               "size": f"{job['width']}x{job['height']}", "engine": "xestrata-image", "detect_s": round(detect_s, 3),
               "fix": {"parts": job["fix"]["parts"], "strength": job["fix"]["strength"],
                       "regions": report[0]["regions"] if report else [], "left_out": left,
                       **({"candidates": [r["regions"] for r in report]} if len(report) > 1 else {})}}
    for key in ("text_encode_ms", "sampling_s", "vae_decode_s"):
        figures[key] = round(figures[key], 2)
    gen.last = figures
    return images, figures, used


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's GPU turn, with the LoRA training's yield first (serve/lora_train.py): a training on this GPU
    shrinks or pauses at a micro-step boundary so the image goes first (the wait is a micro-step's seconds plus the
    release, never the training's end)."""
    lt = getattr(svc, "lora_jobs", None)
    need = None
    if lt is not None and lt.current is not None and lt.place is not None and gen.engine.alive():
        try:
            need = int(gen.need(job, start=False)["least_mib"])
            if need <= 0:
                need = None
        except (ImageJobFailed, OSError, KeyError, TypeError, ValueError):
            need = None
    waiting = lt.image_begin(need) if lt is not None else None
    try:
        with _gpu_turn(svc, gen, job, cancel) as policy:
            yield policy
    finally:
        if lt is not None and waiting:
            lt.image_end(waiting)


@contextlib.contextmanager
def _gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used.  On a
    card of its own (ImageModels.separate), that card's turn alone (separate_turn)."""
    refuse_during_long_job(svc)
    owner = getattr(gen, "owner", None) or getattr(svc, "images", None)   # the lent text GPU's models: their own
    if isinstance(owner, ImageModels) and owner.separate:
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
            models = owner
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
                if _qimage().family_capabilities(gen.family).get("chat_text_states"):
                    # Its text encoder runs by the chat engine while that holds the GPU.
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
                if policy == "shrink" and restore not in ("shrunk", None):
                    policy = "unload"   # the cache could not shrink far enough: the text engine was unloaded instead

                def yield_turn() -> None:
                    """A long job's pause (a video's chunk out): the requests waiting for the service's turn run on
                    the text engine, then the job goes on.  Under unload that engine is not loaded: they were
                    refused (long_job_left), nothing waits."""
                    if policy == "unload":
                        return
                    with svc.status_lock:
                        if not svc.status["queued"] and not getattr(svc, "live_reqs", None):
                            return
                        svc.status.update(busy=False, phase=None)
                    if ctl is not None:
                        ctl.release()
                    svc.fifo.release()
                    try:
                        for _ in range(240):   # the waiting take the turn (and leave the queue) in a moment
                            with svc.status_lock:
                                if not svc.status["queued"]:
                                    break
                            time.sleep(0.25)
                    finally:
                        svc.fifo.acquire()
                        if ctl is not None:
                            ctl.acquire()
                            while getattr(svc, "live_reqs", None):
                                time.sleep(0.25)
                        with svc.status_lock:
                            svc.status.update(busy=True, phase="generating an image", started=time.time())
                try:
                    budget = ledger_fit(gen, need, job)
                    gen._say(f"image job: {policy}, the image engine's budget {budget} MiB (the DiT resident from "
                             f"{need['full_mib']} MiB, at least {need['least_mib']} MiB)")
                    gen.yield_turn = yield_turn
                    yield policy
                finally:
                    gen.yield_turn = None
                    _give_back(svc, gen, policy, restore)
                    gen.ledger_sync()
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


def _helper_ledger(svc, card_env: dict | None, r: dict | None) -> None:
    """The text engine's helper cache in its card's VRAM ledger (serve/vram_ledger.py; `card_env` the borrower's
    environment, which names the card as the image and translation engines' ledgers do): its entry, under the text
    engine's pid, set to what the engine says the cache holds after HELPER (`r`; None: before the first, its INFO's
    helper_mib).  The cache was on the card before the ledger knew of it: it is entered as in the base, so what it
    gives back is room and what it takes back charged."""
    pid = getattr(getattr(svc.engine, "proc", None), "pid", None)
    if card_env is None or pid is None:
        return
    led = vram_ledger.Ledger(gpu_key(card_env))
    if r is None:
        mib = (getattr(svc.engine, "info", None) or {}).get("helper_mib")
        if isinstance(mib, int) and led.find("helper", pid) is None:
            led.hold("helper", mib, pid, in_base=True)
    elif isinstance(r.get("held_mib"), int) and led.find("helper", pid) is not None:
        led.hold("helper", r["held_mib"], pid)


def helper_borrow(svc, card: str, holder, say, cancel: threading.Event | None = None,
                  card_env: dict | None = None) -> bool:
    """Before work on the text engine's helper card (`card`; an image job's, the translation engine's): the helper
    closed (`HELPER off`, the service's turn for that command alone), so the card is the work's; the text engine
    decodes on its first card meanwhile.  `holder` counts the borrowers (helper_lock, helper_jobs: the image models',
    which the translation engine shares, so the helper stays closed until both are done); `say` writes a log line;
    `card_env` the borrower's environment, for the card's VRAM ledger (_helper_ledger).  False where the engine runs
    no helper there (it chose its first card alone: the card is free)."""
    eng = svc.engine
    info = getattr(eng, "info", None) or {}
    if not card or str(info.get("helper_pci", "")).lower() != card or not hasattr(eng, "helper"):
        return False
    with holder.helper_lock:
        holder.helper_jobs += 1
        holder.helper_env = card_env
    try:
        with svc._turn(cancel):
            _helper_ledger(svc, card_env, None)
            r = eng.helper(False)
            _helper_ledger(svc, card_env, r)
        say(f"the text model's helper GPU handed over ({r.get('vram_free_mib', '?')} MiB free there, "
            f"{r.get('ms', '?')} ms)")
    except Exception as e:                       # noqa: BLE001 - the work still runs, on what the card has free
        say(f"the text model's helper GPU was not handed over ({e})")
    return True


def helper_return_later(svc, holder, say) -> None:
    """After work that closed the helper (helper_borrow): open it again (`HELPER on`, the service's turn for that
    command, a second or two to fill from RAM) once no other borrower has started within HELPER_REOPEN_S, so a run of
    image requests does not close and fill it each time."""
    with holder.helper_lock:
        holder.helper_jobs -= 1

    def reopen():
        with holder.helper_lock:
            if holder.helper_jobs > 0:
                return
            try:
                with svc._turn():
                    r = svc.engine.helper(True)
                    _helper_ledger(svc, getattr(holder, "helper_env", None), r)
                say(f"the text model's helper GPU filled again ({r.get('experts', '?')} experts, "
                    f"{r.get('ms', '?')} ms)")
            except Exception as e:               # noqa: BLE001 - the text engine decodes on its first card meanwhile
                say(f"the text model's helper GPU was not filled again ({e})")
    idle_watch.watch.add(HELPER_REOPEN_S, reopen)


@contextlib.contextmanager
def separate_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """gpu_turn on a card no text engine takes: only that GPU's turn (GpuTurn), so chat requests run meanwhile and the
    text engine is left as it is (the policy separate); the chat's status is not touched.  MOSS-Image's prompt read by
    the chat's engine (text_states) takes the service's turn for that read alone."""
    models: ImageModels = getattr(gen, "owner", None) or svc.images
    error: str | None = None
    held = False
    try:
        def waiting():
            gen._say("image job: waiting for another server's image job on this GPU")
        models.gpu.acquire(cancel, waiting)
        held = True
        models.activate(gen)
        gen.begin(job)
        if _qimage().family_capabilities(gen.family).get("chat_text_states"):
            with svc._turn(cancel):
                text_states(svc, gen, job)
        # the helper handed over before the NEED: the free VRAM it reads, and the ledger's base taken from it, are the
        # card's without the helper's cache
        handed = helper_borrow(svc, getattr(models, "helper_card", ""), models, lambda m: gen._say("image job: " + m),
                               cancel, gen.engine.env)
        try:
            try:
                need = gen.need(job)
            except ImageJobFailed:
                gen.totals["failed"] += 1
                raise
            check_room(need, ram_available_mib())
            try:
                budget = ledger_fit(gen, need, job)
                gen._say(f"image job: separate, the image engine's budget {budget} MiB on its own GPU (the DiT "
                         f"resident from {need['full_mib']} MiB, at least {need['least_mib']} MiB)")
                yield "separate"
            finally:
                _give_back(svc, gen, "separate", None)
                gen.ledger_sync()
                svc.last_request_at = time.time()
        finally:
            if handed:
                helper_return_later(svc, models, lambda m: gen._say("image jobs done: " + m))
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
            got = out.get("vram_free_mib")
            # the cache gives back whole slots and the driver's free count lags: a give-back short of what was asked
            # (2,301 of 2,404 MiB beside Qwen3.6-35B-A3B on the B70, 2026-11-09, which left Florence-2 103 MiB short)
            # is asked once more, by what it lacked
            if isinstance(got, (int, float)) and got < want:
                out = eng.vram(want + (want - int(got)))
            gen._say(f"image job: the engine's expert cache gave VRAM back ({out.get('vram_free_mib')} MiB free)")
            got = out.get("vram_free_mib")
            least = max(0, need["least_mib"] - need["held_mib"]) + gen.reserve
            if isinstance(got, (int, float)) and got < least:
                # at its smallest the cache still leaves the job short (Florence-2 200 MiB short beside
                # Qwen3.6-35B-A3B on the B70, 2026-11-09): back to its size, and the text engine unloaded for the job
                gen._say(f"image job: {got} MiB free with the expert cache at its smallest, the job needs {least}: "
                         "the model is unloaded for it instead")
                with contextlib.suppress(Exception):
                    eng.vram(svc.vram_reserve)
                return _make_room(svc, gen, "unload", need)
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
        kept = getattr(eng, "kept_arena", None)
        if kept is not None:
            kept = _drop_for(eng, gen, need, kept)
        if kept is not None:
            eng.unload(keep_arena=True)
        else:
            eng.unload()
        if svc.vision is not None and hasattr(svc.vision, "unload"):
            svc.vision.unload()
        return saved or "unloaded"
    return None


def model_mib(gen: ImageGen) -> int:
    """The image model's files (its DiT, a folder of them for MOSS-Image, its VAE, text encoder and vision encoder) in
    MiB: what its job reads, from the page cache or a copy in RAM."""
    total = 0
    for k in ("diffusion_model", "vae", "llm", "vision", "edit_model"):
        p = Path(gen.cfg[k]) if isinstance(gen.cfg.get(k), str) else None
        if p is None or not p.exists():
            continue
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return total >> 20


def engine_anon_mib(eng) -> int:
    """The text engine's own RAM, what its unload gives back and the shared arena is not in (Linux's RssAnon, Windows'
    private bytes), 0 unknown."""
    pid = getattr(getattr(eng, "proc", None), "pid", None)
    if os.name == "nt":
        try:
            import psutil
            return int(getattr(psutil.Process(pid).memory_info(), "private")) >> 20   # Windows' field
        except (ImportError, OSError, ValueError, TypeError, AttributeError):
            return 0
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) >> 10
    except (OSError, ValueError, TypeError):
        pass
    return 0


def _drop_for(eng, gen: ImageGen, need: dict, kept):
    """Before an image job's unload that keeps the text engine's experts in RAM (server.py's KeptArena): the RAM the
    job wants - its prefix cache, its model's files, RAM_SPARE_MIB and the RAM reserve the user asked for - against
    what is available once the engine's own RAM is back; what is short, the engine marks dropped from its coldest
    experts (arena_drop), punched out after it ends.  -> `kept`, or None when the whole arena is to go (the engine
    could not, or the RAM is not known)."""
    avail = ram_available_mib()
    if avail is None:
        gen._say("image job: the free RAM is not known; the expert arena goes with the model")
        return None
    reserve = int(str(envvars.get("RAM_RESERVE_MIB", env=gen.engine.env) or "0") or 0)
    want = int(need.get("host_mib") or 0) + model_mib(gen) + RAM_SPARE_MIB + reserve
    short = want - avail - engine_anon_mib(eng)
    if short <= 0:
        gen._say(f"image job: the experts stay in RAM ({want} MiB wanted, enough is free)")
        return kept
    path = str(gen.dir / f"arena-drop-{secrets.token_hex(6)}.txt")
    try:
        r = eng.arena_drop(short, path)
    except Exception as e:                           # noqa: BLE001 - the unload gives the whole arena back
        gen._say(f"image job: the engine could not drop experts from its arena ({e}); the whole arena goes")
        with contextlib.suppress(OSError):
            os.unlink(path)
        return None
    kept.pending = path
    gen._say(f"image job: {r.get('dropped_mib')} MiB of the coldest experts leave the RAM for the image "
             f"({short} MiB short); the rest stay")
    return kept


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
        # the reserve asked for last (None: the start's).  The image engine's VRAM reaches the device a moment after
        # it lets go (ROCm 10.1's HIP refused the first segments mapped again at once, 2026-11-02): tried again for
        # GROW_WAIT_S while the cache is short of its full size
        end, why = time.monotonic() + GROW_WAIT_S, ""
        while True:
            try:
                out = eng.vram(svc.vram_reserve)
                why = "" if out.get("expert_slots", 0) >= out.get("expert_slots_full", 0) else \
                    f"{out.get('expert_slots')} of {out.get('expert_slots_full')} slots"
            except Exception as e:                   # noqa: BLE001
                why = str(e)
            if not why or time.monotonic() >= end:
                break
            time.sleep(0.25)
        if why:
            gen._say(f"image job: the expert cache did not grow back ({why}); POST /v1/vram can retry")
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
        print(f"[xestrata] the conversation from before the image job is back ({r.get('tokens')} tokens)", flush=True)
    except Exception as e:                           # noqa: BLE001 - the engine runs, without the old cache
        print(f"[xestrata] the conversation from before the image job was not restored ({e}); the next request "
              "reads its prompt again", flush=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
