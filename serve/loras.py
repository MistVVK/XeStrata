# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/loras.py - the adapters an image model can apply: its LoRA folder's files, and a request's "lora" field and
prompt tags checked against them (the image engine reads and applies them: src/image/image_lora.cpp).

The files are read by their headers only (a safetensors' JSON, a GGUF's metadata and tensor infos): each tensor's name
and shape and the file's string metadata.  A key is brought to the DiT's name the way the engine brings it (split_key:
the file's prefix off, ".weight" off, the role's suffix off; sd-scripts' and the LyCORIS form's names, which write
every dot as an underscore, by the model's own names - kohya_name, and the engine's lora_key_names), and a file is
this model's when every weight it changes is one the DiT's products take (TARGETS: the names of qi_dit.cpp's,
moss_dit.cpp's and mf_dit.cpp's build_lora_targets, written here a second time; the record's check sets them
against the probe's `map` on the models, name for name), in a form the engine applies today: a plain LoRA, LoKr or
LoHa (a DoRA: a plain LoRA with a magnitude over the weight's rows, PEFT's composed row norms rescaled - `dora_scale`,
or the same vector as PEFT writes it (`lora_magnitude_vector`, its `.default` form too) or as ai-toolkit writes it
(`.magnitude`), and LyCORIS's [rows][1] shape too), a whole weight's difference (diff, set_weight, w_norm), or an
OFT's turn of the weight's rows (oft_blocks, a BOFT's four dimensions too; the file's alpha is the constraint on the
block's skew part, not a scale), and a GLoRA (a1/a2/b1/b2: the weight's own product on the input
its factors make, beside a plain form's pair), and no text encoder's weight.  A bias's delta (diff_b, or b_norm beside
w_norm) goes with any of them or alone, where the weight adds a bias of its own (BIASED: Mage-Flow's DiT's weights all
do; MOSS-Image's outside its blocks and in its text embedder; Qwen-Image-2.1's none).  The engine refuses the forms it
does not apply when it binds a set (QiDit::set_loras, MossDit::set_loras, MfDit::set_loras), so no image is made with
part of a file.

A listing entry - ImageGen.defaults' "loras" (the engine's capabilities: /health's image_defaults and image_models,
/v1/models' image model):
  name      the file without its extension; a subfolder's is its path there ("style/gouache")
  path      the same with the extension, relative to the folder (never an absolute path)
  rank      the highest rank of its weights
  scale     the strength its metadata recommends (recommended_strength_model, ...), else 1.0
  trigger   its trigger words (modelspec.trigger_phrase, ss_trigger_words, ...), else None
  kind      what it was trained to add, "character", "style" or "other" (ss_xestrata_kind), else None
  targets   the kinds of weights it changes ("attn.to_q", "img_mlp.gate_layer", "modulation.1")
  size_mib  the file's size

A file's keys that name no tensor of an adapter (PEFT's `base_layer.bias`, an auxiliary projector, a sidecar) are not
dropped in silence: the scan keeps them in Folder.ignored, a request that applies such a file says them in its
response's xestrata-engine (lora_ignored) and in the log, and so does the engine (image_lora.cpp, to its stderr).
A file whose metadata says rsLoRA is not listed: its scale is alpha / sqrt(rank), which the engine does not apply
(docs).

A request's adapters come from the "lora" field - a list of {"name", "scale"} or one such object (vLLM-Omni's shape), a
name alone, or one of those as JSON text (a multipart form's field) - and from the prompt's tags, <lora:name:0.8> in
A1111's and Forge's form, in the positive prompt and the negative one.  An adapter's "blocks" (a tag's
<lora:name:0.8:lbw=...>, LoRA Block Weight's) scales its strength a DiT block: a number each block, or a name
(BLOCK_PRESETS: "character", "style").  "local_path" names a file inside the folder,
and no other path is taken (a path is not a capability).  A text encoder's adapter is not applied (the human's
decision of 2026-10-29): "scale_text" and a tag's third field, A1111's text encoder's weight, are taken as the
engine's other unknown fields are, ignored.  A name the listing does not have is refused where the field
gives it, and taken out of the prompt with a warning where a tag gives it (the response's xestrata.lora_unknown and the
server's log; the human's decision of 2026-10-29, as A1111 and Forge do).  A model whose DiT takes no adapter at all
takes neither the field nor a tag: both are refused, so no image is made without what was asked for.

The folder is the config's "lora_dir" (setup writes it: <data>/models/loras), else "loras" beside the DiT's file.  Its
listing is kept CACHE_S seconds and read again at once when the folder's mtime changes: /health asks for it with each
page load.
"""
from __future__ import annotations

import collections
import errno
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import shutil
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from serve import download as download_core
from serve import file_lock, model_terms

MAX_LORA = 4                       # the most adapters one request stacks
SCALE_MIN, SCALE_MAX = -2.0, 2.0   # a request's strength
CACHE_S = 10.0                     # how long a listing is kept before the folder is read again
EXTS = (".safetensors", ".gguf")
MAX_HEADER = 64 << 20              # the longest header read (a safetensors' JSON is a file's tensors; a GGUF's too)
TAG = re.compile(r"<lora:([^<>:]+)(?::([^<>:]+))?(?::([^<>:]+))?>", re.IGNORECASE)
FAMILY_NAMES = {"qwen-image": "Qwen-Image-2.1", "moss-image": "MOSS-Image", "mage-flow": "Mage-Flow",
                "hunyuan-image-3": "HunyuanImage 3.0", "anima": "Anima"}

# The weights adapters may take, by the name a file's key is left with after split_key: the DiTs' target tables
# (src/image/qi_dit.cpp's, moss_dit.cpp's and mf_dit.cpp's build_lora_targets write these names).  "{b}" is a block's
# number - the same names for every block, so a listing needs no block count.  Qwen-Image-2.1's MLP's first weight is
# one [gate | up] on the GPU: a file may name it fused (gate_up, the dot form's and a GGUF's) or as diffusers' two
# halves (gate_layer is its rows [0, inter), proj the rest).  MOSS-Image's text embedder holds one qkv of three C-row
# weights (one name on the file's side), and up and gate apart on the file's side where the GPU has one [up | gate].
# Mage-Flow's and MageTrail's blocks are the same 14 weights (the two streams' modulations, the attention's projections,
# the MLPs); MageTrail's low-rank modulation is F32 and the engine binds no adapter to it (set_loras says so).
_QWEN_BLOCKS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.gate_layer",
                "img_mlp.proj", "img_mlp.out")
_MOSS_TEXT_BLOCKS = ("qkv", "proj", "ffn_up_gate", "ffn_up", "ffn_gate", "ffn_down")
_MOSS_BLOCKS = ("proj.q", "proj.k", "proj.v", "proj.qkv", "proj.o", "proj.ffn_up_gate", "proj.ffn_up",
                "proj.ffn_gate", "proj.ffn_down")
_MOSS_OUTSIDE = ("video_embedder", "time_embedding.0", "time_embedding.2", "shared_block_modulation",
                 "video_head_modulation", "final_linear_video", "text_embedder.proj")
_MAGE_BLOCKS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                "attn.to_out.0", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
                "txt_mlp.net.2", "img_mod.1", "txt_mod.1")
_MAGE_OUTSIDE = ("img_in", "txt_in", "time_text_embed.timestep_embedder.linear_1",
                 "time_text_embed.timestep_embedder.linear_2", "norm_out.linear", "proj_out")
# Anima's (src/image/anima_dit.cpp's build_lora_targets): the blocks' attentions, MLP and modulations, the LLM
# adapter's, and the parts outside them.  These names hold underscores of their own, which is what the underscore
# table (_KOHYA, kohya_name) is for: sd-scripts' keys reach this model with every dot an underscore.
_ANIMA_BLOCKS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                 "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                 "mlp.layer1", "mlp.layer2", "adaln_modulation_self_attn.1", "adaln_modulation_self_attn.2",
                 "adaln_modulation_cross_attn.1", "adaln_modulation_cross_attn.2", "adaln_modulation_mlp.1",
                 "adaln_modulation_mlp.2")
_ANIMA_ADAPTER = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "cross_attn.q_proj",
                  "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.o_proj", "mlp.0", "mlp.2")
_ANIMA_OUTSIDE = ("x_embedder.proj.1", "t_embedder.1.linear_1", "t_embedder.1.linear_2", "final_layer.linear",
                  "final_layer.adaln_modulation.1", "final_layer.adaln_modulation.2", "llm_adapter.out_proj")
_ANIMA = (_ANIMA_OUTSIDE + tuple("blocks.{b}." + n for n in _ANIMA_BLOCKS) +
          tuple("llm_adapter.blocks.{b}." + n for n in _ANIMA_ADAPTER))
_ANIMA_BIASED = ("llm_adapter.out_proj", "llm_adapter.blocks.{b}.mlp.0", "llm_adapter.blocks.{b}.mlp.2")
TARGETS: dict[str, tuple[str, ...]] = {
    "qwen-image": ("img_in", "time_text_embed.timestep_embedder.linear_1",
                   "time_text_embed.timestep_embedder.linear_2", "modulation.1", "norm_out.linear", "proj_out",
                   "txt_in.in_layer", "txt_in.out_layer") + tuple("transformer_blocks.{b}." + n for n in _QWEN_BLOCKS),
    "moss-image": (_MOSS_OUTSIDE + tuple("text_embedder.blocks.{b}." + n for n in _MOSS_TEXT_BLOCKS) +
                   tuple("layers.{b}." + n for n in _MOSS_BLOCKS)),
    "mage-flow": (_MAGE_OUTSIDE + tuple("transformer_blocks.{b}." + n for n in _MAGE_BLOCKS)),
    "anima": _ANIMA,
}

# The weights that add a bias of their own, where a bias's delta (diff_b, or b_norm beside w_norm) goes: Mage-Flow's
# DiT adds one under every product, MOSS-Image's outside the blocks and in its text embedder (the 42 blocks' shift is
# their modulation's), and Qwen-Image-2.1's DiT has none (the engine refuses a bias's delta for it).
_BIASED_NAMES: dict[str, tuple[str, ...]] = {
    "qwen-image": (),
    "moss-image": ("video_embedder", "time_embedding.0", "time_embedding.2", "shared_block_modulation",
                   "video_head_modulation", "text_embedder.proj") + tuple(
                       "text_embedder.blocks.{b}." + n for n in _MOSS_TEXT_BLOCKS),
    "mage-flow": TARGETS["mage-flow"],
    "anima": _ANIMA_BIASED,
}


def _target_pattern(name: str) -> re.Pattern[str]:
    """A TARGETS name as the pattern a file's key is matched with ("{b}" a block's number)."""
    return re.compile("^" + re.escape(name).replace(r"\{b\}", r"\d+") + "$")


_MATCH = {family: tuple(_target_pattern(n) for n in names) for family, names in TARGETS.items()}
_BIASED = {family: tuple(_target_pattern(n) for n in names) for family, names in _BIASED_NAMES.items()}

# A file's key -> the DiT's weight and the tensor's role, as the engine reads them (src/image/image_lora.cpp's
# split_key): the longest prefix first, ".weight" off every form but PEFT's bare lora_A and lora_B, then the role's
# suffix, longest first too.  sd-scripts' keys (`lora_unet_`) and the LyCORIS form's (`lycoris_`) write a weight's
# name with every dot an underscore; those two prefixes leave the name for kohya_name below to bring to the model's
# own (the engine matches the same two forms against its target table, image_lora's lora_key_names).
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "lora_unet_", "lycoris_")
UNDERSCORE_PREFIXES = ("lora_unet_", "lycoris_")
ROLES = (
    (".lora_A.default", "lora_A"), (".lora_B.default", "lora_B"),
    (".lora.down", "lora_A"), (".lora.up", "lora_B"),
    (".lora_A", "lora_A"), (".lora_B", "lora_B"),
    (".lora_down", "lora_A"), (".lora_up", "lora_B"),
    (".lora_mid", "lora_mid"), (".lora_linear_layer.down", "lora_A"), (".lora_linear_layer.up", "lora_B"),
    (".alpha", "alpha"), (".dora_scale", "dora_scale"),
    # PEFT's and ai-toolkit's names for the same thing (a DoRA's magnitude, over the weight's rows)
    (".lora_magnitude_vector.default", "dora_scale"), (".lora_magnitude_vector", "dora_scale"),
    (".magnitude", "dora_scale"),
    (".lokr_w1_a", "lokr_w1_a"), (".lokr_w1_b", "lokr_w1_b"), (".lokr_w2_a", "lokr_w2_a"), (".lokr_w2_b", "lokr_w2_b"),
    (".lokr_w1", "lokr_w1"), (".lokr_w2", "lokr_w2"), (".lokr_t2", "lokr_t2"),
    (".hada_w1_a", "hada_w1_a"), (".hada_w1_b", "hada_w1_b"), (".hada_w2_a", "hada_w2_a"), (".hada_w2_b", "hada_w2_b"),
    (".hada_t1", "hada_t1"), (".hada_t2", "hada_t2"),
    (".oft_blocks", "oft_blocks"), (".rescale", "rescale"),
    (".a1", "a1"), (".a2", "a2"), (".b1", "b1"), (".b2", "b2"),
    (".diff_b", "diff_b"), (".diff", "diff"), (".set_weight", "set_weight"),
    (".w_norm", "w_norm"), (".b_norm", "b_norm"),
    (".reshape_weight", "reshape_weight"),
)
# The strength a file's metadata may recommend, and the words it may name a trigger with (kohya's and peft's saves
# carry no strength; civitai's, SimpleTuner's and ai-toolkit's may recommend one).
STRENGTH_KEYS = ("recommended_strength_model", "recommended_strength", "ss_strength", "modelspec.strength")
TRIGGER_KEYS = ("modelspec.trigger_phrase", "trigger_phrase", "trigger", "trigger_words", "ss_trigger_words")
# What an adapter was trained to add (XeStrata's trainers write it: ss_xestrata_kind), the listing's "kind"
KIND_KEY = "ss_xestrata_kind"
KINDS = ("character", "style", "other")

# A request's block weights by name ("blocks": "character"): the strength's factor each DiT block, one a block of the
# family's DiT (the engine's LORA:<strength>@<w0>,...: word, LoraUse's blocks), where a character's look and where
# a picture's style go in that DiT.  Read from images of a character LoRA with its blocks switched on a group at a
# time, the same prompt and seed (2026-11-16, the RX 9060 XT): Qwen-Image-2.1 Turbo's blocks 8-15 alone drew the
# character (its hair, eyes and hairpin) with the base's light and background, 0-7 alone the LoRA's drawing without
# the character, the rest little; Anima Turbo's 14-20 alone drew the character in the base's drawing, the others
# none of it.  "style" is the other blocks.  Mage-Flow's blocks 0-5 carried both, a block or a weight at a time
# alike, so it has no table (nor MOSS-Image and MageTrail, not looked at yet).  "all" is every block at 1.  A family
# with no table takes the numbers' form only.


def _span(n: int, first: int, last: int, inside: float) -> tuple[float, ...]:
    return tuple(inside if first <= b <= last else 1.0 - inside for b in range(n))


BLOCK_PRESETS: dict[str, dict[str, tuple[float, ...]]] = {
    "qwen-image": {"character": _span(32, 8, 15, 1.0), "style": _span(32, 8, 15, 0.0)},
    "anima": {"character": _span(28, 14, 20, 1.0), "style": _span(28, 14, 20, 0.0)},
}


class LoraError(ValueError):
    """A request's adapters that cannot be taken (the image endpoint's 400)."""


def split_key(raw: str) -> tuple[str, str, bool]:
    """A key as a file writes it -> (the weight's name, the tensor's role, whether the name is one's dots are written
    as underscores); ("", "", False) for a key of no role (a file's own: a quantization's, a config's)."""
    s, under = raw, False
    for p in PREFIXES:
        if s.startswith(p):
            under = p in UNDERSCORE_PREFIXES
            s = s[len(p):]
            break
    if s.endswith(".weight"):
        s = s[: -len(".weight")]
    role = ""
    for suffix, r in ROLES:
        if s.endswith(suffix):
            s, role = s[: -len(suffix)], r
            break
    return s, role, under


def _kohya_forms(name: str) -> list[tuple[str, str]]:
    """A weight's underscore-written name as the chances of finding it in the table: itself, and itself with each
    run of digits standing for a block's number (`transformer_blocks_12_attn_to_q` as
    `transformer_blocks_{b}_attn_to_q`).  A name's own digits stay where they are (`img_mlp_net_0_proj`,
    `modulation_1`), so trying each run once is enough and no pattern scan is needed."""
    parts = re.split(r"(\d+)", name)
    got = [(name, "")]
    for i in range(1, len(parts), 2):
        got.append(("".join("{b}" if j == i else p for j, p in enumerate(parts)), parts[i]))
    return got


# The underscore form of every weight a DiT takes, made from the model's own names (each name's dots written as
# underscores, not the file's underscores read back as dots).  A name's own underscores stay underscores
# (`attn.to_q`, `proj.ffn_up_gate`, `img_mlp.net.0.proj`), which a blanket turn of the file's underscores into dots
# missed: that read only names with no underscore at all (Anima's names were matched both ways for it).
_KOHYA = {family: {n.replace(".", "_"): n for n in names} for family, names in TARGETS.items()}


def kohya_name(name: str, family: str) -> str:
    """The weight a key left by `lora_unet_` or `lycoris_` names (every dot an underscore), as `family` names it;
    "" when that model names no such weight (the engine answers the same way, lora_key_names)."""
    table = _KOHYA.get(family) or {}
    for chance, block in _kohya_forms(name):
        got = table.get(chance)
        if got is not None:
            return got.replace("{b}", block)
    return ""


def dora_way(shape: tuple[int, ...], rows: int, cols: int) -> str:
    """Which way a DoRA's magnitude runs: "rows" (what the engine scales), "columns", or "" (neither).  LyCORIS and
    PEFT write the rows' vector as [rows][1] and the columns' as [1][cols], so a 1 says which; a bare [n] takes n,
    and a square weight's magnitude reads as the rows."""
    if len(shape) == 1:
        return "rows" if shape[0] == rows else ("columns" if shape[0] == cols else "")
    if len(shape) == 2 and shape[1] == 1 and shape[0] == rows:
        return "rows"
    if len(shape) == 2 and shape[0] == 1 and shape[1] == cols:
        return "columns"
    return ""


def form_of(roles) -> str:
    """The form a weight's tensors are in, as the engine's kind_of reads them: lora, lokr, loha, glora, oft, diff (a
    whole delta: diff, set_weight or w_norm); "" when they are none of those."""
    for role, form in (("lora_A", "lora"), ("lora_B", "lora"), ("lokr_w1", "lokr"), ("lokr_w2", "lokr"),
                       ("lokr_w1_a", "lokr"), ("lokr_w2_a", "lokr"), ("hada_w1_a", "loha"), ("hada_w2_a", "loha"),
                       ("a1", "glora"), ("b1", "glora"), ("oft_blocks", "oft"), ("diff", "diff"),
                       ("set_weight", "diff"), ("w_norm", "diff")):
        if role in roles:
            return form
    return ""


def _lokr_rank(roles: dict[str, tuple[int, ...]]) -> tuple[int, str]:
    """A LoKr weight's tensors -> (the factorization's rank, "") or (0, why they make no Kronecker product): its
    two matrices whole (lokr_w1 and lokr_w2) or each from two factors (lokr_w1_a [rows][rank] and lokr_w1_b
    [rank][cols]), w2's under a Tucker core (lokr_t2, a linear product's: a kernel of 1 x 1).  The rank is the one
    the file's alpha is divided by (w2's factors' rows, else w1's); the engine reads the tensors the same way
    (src/image/image_lora.cpp's make_lokr)."""
    rank = 0
    for name in ("w1", "w2"):
        whole, fa, fb = f"lokr_{name}", f"lokr_{name}_a", f"lokr_{name}_b"
        if whole in roles:   # the whole matrix: a Tucker core beside w2 stays unread, the matrix taken
            if fa in roles or fb in roles:
                return 0, f"both {whole} and its factors {fa}/{fb}"
            if len(roles[whole]) != 2:
                return 0, f"{whole} must be a matrix"
            continue
        if fa not in roles or fb not in roles:
            return 0, f"a LoKr weight needs {whole} or both {fa} and {fb}"
        a, b = roles[fa], roles[fb]
        if len(a) != 2 or len(b) != 2:
            return 0, f"{fa} and {fb} must be matrices"
        if name == "w2" and "lokr_t2" in roles:
            t2 = roles["lokr_t2"]
            if not (len(t2) == 2 or (len(t2) == 4 and t2[2] == 1 and t2[3] == 1)):
                return 0, "a LoKr Tucker core with a convolution's kernel dimensions"
            if a[0] != t2[0] or b[0] != t2[1]:
                return 0, f"the Tucker core is {t2[0]} x {t2[1]}; w2_a's rows {a[0]}, w2_b's {b[0]}"
        elif a[1] != b[0]:
            return 0, f"{fb} is {b[0]} x {b[1]}, not the rank its {fa} has ({a[1]})"
        rank = b[0]
    return rank, ""


def _loha_rank(roles: dict[str, tuple[int, ...]]) -> tuple[int, str]:
    """A LoHa weight's tensors -> (the rank the file's alpha is divided by, "") or (0, why they make no delta):
    two low-rank products, (hada_w1_a . hada_w1_b) * (hada_w2_a . hada_w2_b), each side's under a Tucker core
    (hada_t1/hada_t2; a linear product's, a kernel of 1 x 1), which writes the side's factors the core's way (w_a
    [core][out], w_b [rank][in]).  The rank is w1_b's rows; the engine makes the delta when it
    binds the entry (src/image/image_lora.cpp's make_loha and loha_delta)."""
    if ("hada_t1" in roles) != ("hada_t2" in roles):
        return 0, "a LoHa with one Tucker core (hada_t1/hada_t2): both or neither"
    sides = []
    for name in ("w1", "w2"):
        fa, fb, tc = f"hada_{name}_a", f"hada_{name}_b", f"hada_t{name[-1]}"
        if fa not in roles or fb not in roles:
            return 0, f"a LoHa weight needs {fa} and {fb}"
        a, b = roles[fa], roles[fb]
        if len(a) != 2 or len(b) != 2:
            return 0, f"{fa} and {fb} must be matrices"
        if tc in roles:
            t = roles[tc]
            if not (len(t) == 2 or (len(t) == 4 and t[2] == 1 and t[3] == 1)):
                return 0, f"a {tc} with a convolution's kernel dimensions"
            if t[0] != a[0] or t[1] != b[0]:
                return 0, f"{tc} is {t[0]} x {t[1]}; {fa}'s rows {a[0]}, {fb}'s {b[0]}"
            sides.append((a[1], b[1]))
        else:
            if a[1] != b[0]:
                return 0, f"{fb} is {b[0]} x {b[1]}, not the rank its {fa} has ({a[1]})"
            sides.append((a[0], b[1]))
    if sides[0] != sides[1]:
        return 0, (f"its two sides' products are {sides[0][0]} x {sides[0][1]} and {sides[1][0]} x {sides[1][1]}: "
                   f"not the weight's one shape")
    return roles["hada_w1_b"][0], ""


def _oft_shape(roles: dict[str, tuple[int, ...]]) -> tuple[int, str]:
    """An OFT weight's tensors -> (a block's rows, "") or (0, why they make no turn): oft_blocks [blocks][b][b] (an
    OFT) or [stages][blocks][b][b] (a BOFT, whose stage i reads the rows 2^(i-1) blocks at a time), the blocks squares,
    and a rescale of the weight's rows where the file has one.  The engine reads them the same way
    (src/image/image_lora.cpp's make_oft), and checks the rows against the weight it binds them to."""
    bl = roles["oft_blocks"]
    if len(bl) == 3:
        stages, blocks, b = 1, bl[0], bl[1]
    elif len(bl) == 4:
        stages, blocks, b = bl[0], bl[1], bl[2]
    else:
        return 0, (f"oft_blocks must be [blocks][b][b] (an OFT) or [stages][blocks][b][b] (a BOFT), not {len(bl)} "
                   f"dimensions")
    if b <= 0 or blocks <= 0 or stages <= 0 or bl[-1] != b:
        return 0, f"oft_blocks' blocks are {b} x {bl[-1]}: not squares"
    if stages > 1:
        if b % 2 != 0:
            return 0, f"a BOFT's block of {b} rows: not an even number"
        for i in range(stages):
            if blocks % (2 ** i) != 0:
                return 0, (f"a BOFT's stage {i} reads the rows {blocks} blocks at a time: not a multiple of {2 ** i}")
    rs = roles.get("rescale")
    if rs is not None and tuple(rs) != (blocks * b,):
        return 0, f"a rescale of {rs} values, not the weight's {blocks * b} rows"
    return b, ""


def _glora_rank(roles: dict[str, tuple[int, ...]]) -> tuple[int, str]:
    """A GLoRA weight's tensors -> (the rank the file's alpha is divided by, "") or (0, why they make no GLoRA): the
    four matrices of one of the form's two readings - the newer a1 [cols][rank], a2 [rank][cols], b1 [rows][rank], b2
    [rank][cols], or LyCORIS's older one (a1 [rank][cols], a2 [cols][rank], b1 [rank][cols], b2 [rows][rank]).  A
    square weight's four fit both, and the weight's rows and columns choose between them in the engine (make_glora's
    `glora_other`, the entry the DiT resolves with lora_glora_for_weight); the listing has no weight to ask, so it
    reports the smaller of the two readings' ranks (the one a GLoRA of a weight that shape has: the other reading would
    make the weight's rank its rows' count)."""
    names = ("a1", "a2", "b1", "b2")
    missing = [n for n in names if n not in roles]
    if missing:
        return 0, f"a GLoRA weight needs a1, a2, b1 and b2 ({missing[0]} is missing)"
    a1, a2, b1, b2 = (roles[n] for n in names)
    if any(len(t) != 2 for t in (a1, a2, b1, b2)):
        return 0, "a1, a2, b1 and b2 must be matrices"
    ranks = []
    why = ""
    # the older reading first, as the engine takes it where both fit
    if b2[1] == b1[0] == a1[0] == a2[1]:
        ranks.append(a1[0])
        want = {"a1": (a1[0], a1[1]), "a2": (a1[1], a1[0]), "b1": (a1[0], a1[1]), "b2": (b2[0], a1[0])}
        why = why or _glora_shapes(want, {"a1": a1, "a2": a2, "b1": b1, "b2": b2})
    if b2[0] == b1[1] == a1[1] == a2[0]:
        ranks.append(a1[1])
        want = {"a1": (a1[0], a1[1]), "a2": (a1[1], a1[0]), "b1": (b1[0], a1[1]), "b2": (a1[1], a1[0])}
        why = why or _glora_shapes(want, {"a1": a1, "a2": a2, "b1": b1, "b2": b2})
    if not ranks:
        return 0, (f"a1/a2/b1/b2 of {a1}, {a2}, {b1} and {b2}: neither of a GLoRA's two readings (a1 [cols][rank], a2 "
                   f"[rank][cols], b1 [rows][rank], b2 [rank][cols], or LyCORIS's with a1/a2 and b1/b2 the other way "
                   f"round)")
    if why:
        return 0, why
    return min(ranks), ""


def _glora_shapes(want: dict[str, tuple[int, int]], have: dict[str, tuple[int, ...]]) -> str:
    """Why a reading's four are not the shapes it takes (the other tensors' rows and columns), "" where they are."""
    for name, t in have.items():
        if tuple(t) != want[name]:
            return (f"its {name} is {t[0]} x {t[1]}, not the {want[name][0]} x {want[name][1]} the form's other "
                    f"tensors take")
    return ""


def is_text_encoder(key: str) -> bool:
    """A weight of the text encoder rather than the DiT (image_lora's lora_is_text_encoder_key)."""
    return key.startswith(("lora_te", "text_encoder", "te."))


# The models whose text encoder takes adapters (src/image/text_encoder.cpp's set_loras: Qwen3-VL's and Qwen3's layers;
# text_encoder.cpp's: Qwen3.5-9B's, its GDN's projections too),
# and its weights as image_lora's lora_text_encoder_name writes a key: the prefixes off, every underscore a dot
TEXT_ENCODER_FAMILIES = ("qwen-image", "mage-flow", "anima", "moss-image")
_TE_NAME = re.compile(r"^layers\.\d+\.(self\.attn\.[qkvo]\.proj|mlp\.(gate|up|down)\.proj|"
                      r"linear\.attn\.(in\.proj\.(qkv|z|a|b)|out\.proj))$")


def text_encoder_name(key: str) -> str:
    """A text encoder's key as the engine names its weight ("lora_te_layers_0_self_attn_q_proj" ->
    "layers.0.self.attn.q.proj")."""
    for p in ("lora_te1_", "lora_te2_", "lora_te_", "text_encoders.", "text_encoder.", "te."):
        if key.startswith(p):
            key = key[len(p):]
            break
    key = key.replace("_", ".")
    at = key.find("layers.")
    return key[at:] if at > 0 and key[at - 1] == "." else key


def is_training_aid(key: str) -> bool:
    """SimpleTuner's auxiliary projector: trained beside the adapter, not one of the model's weights."""
    return "crepa_projector" in key


def kind_of(name: str) -> str:
    """The kind of weight a target's name is, a block's without its number: "transformer_blocks.7.attn.to_q" ->
    "attn.to_q", "layers.3.proj.q" -> "proj.q", "text_embedder.blocks.1.qkv" -> "qkv", "modulation.1" as it is."""
    return re.sub(r"^(?:transformer_blocks|layers|text_embedder\.blocks|blocks|llm_adapter\.blocks)\.\d+\.", "", name)


def encode_path(path: str) -> str:
    """A file's path as a request line word (IMG's LORA:<strength>:<path>): the line is split on whitespace, so the
    separators a path may hold - a space, a '#' - are written %XX; the engine decodes them (strata_image.cpp's
    read_lora).  A colon and the path's own slashes stay as they are."""
    return quote(path, safe="/\\:")


# ----------------------------------------------------------------------------------------------- the files' headers
def _gguf_string(f, limit: int) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    if n > limit:
        raise ValueError("a string of the header is too long")
    return f.read(n).decode("utf-8", "replace")


def _gguf_value(f, kind: int, depth: int = 0):
    """One GGUF metadata value skipped (a string's read: the listing wants those), of its type's number."""
    if kind in (0, 1, 7):                       # u8, i8, bool
        return f.read(1)
    if kind in (2, 3):                          # u16, i16
        return f.read(2)
    if kind in (4, 5, 6):                       # u32, i32, f32
        return f.read(4)
    if kind in (10, 11, 12):                    # u64, i64, f64
        return f.read(8)
    if kind == 8:                               # string
        return _gguf_string(f, MAX_HEADER)
    if kind == 9:                               # array: its elements' type, its count, its values
        if depth > 4:
            raise ValueError("an array inside an array")
        inner, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n):
            _gguf_value(f, inner, depth + 1)
        return None
    raise ValueError(f"a metadata field of type {kind}")


def _gguf_header(f) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    f.seek(4)
    version = struct.unpack("<I", f.read(4))[0]
    if version not in (2, 3):
        raise ValueError(f"a GGUF of version {version}")
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    if n_tensors > (1 << 24) or n_kv > (1 << 24):
        raise ValueError("a header with more tensors than a model has")
    meta: dict[str, str] = {}
    for _ in range(n_kv):
        key = _gguf_string(f, MAX_HEADER)
        kind = struct.unpack("<I", f.read(4))[0]
        value = _gguf_value(f, kind)
        if kind == 8:
            meta[key] = value
    tensors: dict[str, tuple[int, ...]] = {}
    for _ in range(n_tensors):
        name = _gguf_string(f, MAX_HEADER)
        n_dims = struct.unpack("<I", f.read(4))[0]
        if n_dims > 4:
            raise ValueError(f"{name}: more than four dimensions")
        dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
        f.read(4 + 8)                           # the tensor's type; its offset in the data
        tensors[name] = tuple(int(d) for d in reversed(dims))   # [rows, columns], the engine's order
    return tensors, meta


def read_header(path: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """A safetensors' or a GGUF's header -> (its tensors' shapes as [rows, columns], its string metadata).  The data is
    not read.  ValueError for a file of neither form."""
    with open(path, "rb") as f:
        if f.read(4) == b"GGUF":
            return _gguf_header(f)
        f.seek(0)
        head = f.read(8)
        if len(head) < 8:
            raise ValueError("shorter than a header")
        n = struct.unpack("<Q", head)[0]
        if n < 2 or n > MAX_HEADER:
            raise ValueError("not a safetensors header")
        raw = f.read(n)
        if len(raw) < n:
            raise ValueError("shorter than its header")
    head = json.loads(raw)
    if not isinstance(head, dict):
        raise ValueError("not a safetensors header")
    meta = head.pop("__metadata__", None)
    tensors: dict[str, tuple[int, ...]] = {}
    for name, t in head.items():
        shape = t.get("shape") if isinstance(t, dict) else None
        if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
            raise ValueError(f"{name}: no shape")
        tensors[name] = tuple(shape)
    if not isinstance(meta, dict):
        meta = {}
    return tensors, {str(k): str(v) for k, v in meta.items()}


# --------------------------------------------------------------------------------------------------- the folder
def _finite(value) -> float | None:
    """A number, or a metadata value's text (a safetensors' fields are strings), or None."""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _recommended(meta: dict[str, str]) -> float:
    """The strength a file's metadata recommends, else 1.0 (the engine's default: alpha / rank at a request's 1)."""
    for key in STRENGTH_KEYS:
        v = _finite(meta.get(key))
        if v is not None and SCALE_MIN <= v <= SCALE_MAX:
            return v
    return 1.0


def _peft_meta(meta: dict[str, str]) -> dict:
    """PEFT's and diffusers' adapter metadata (the JSON string they keep in "lora_adapter_metadata"), as a dict.
    Its keys carry the model's own prefix ("transformer.lora_alpha"), so they are looked up by their end."""
    raw = meta.get("lora_adapter_metadata") or ""
    if not raw:
        return {}
    try:
        got = json.loads(raw)
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def _meta_flag(peft: dict, key: str) -> bool:
    """Whether PEFT's metadata says `key` (a diffusers-form file names it "transformer.use_rslora")."""
    for name, value in peft.items():
        if (name == key or name.endswith("." + key)) and value is True:
            return True
    return False


def _meta_alphas(peft: dict) -> list[float]:
    """Every lora_alpha the PEFT metadata keeps: PEFT's plain one, and the one a diffusers file gives each component
    ("text_encoder.lora_alpha" beside "transformer.lora_alpha", as its _pack_dict_with_prefix writes them)."""
    got: list[float] = []
    for name, value in peft.items():
        if name == "lora_alpha" or name.endswith(".lora_alpha"):
            v = _finite(value)
            if v is not None:
                got.append(v)
    return got


def _trigger(meta: dict[str, str]) -> str | None:
    for key in TRIGGER_KEYS:
        v = (meta.get(key) or "").strip()
        if v:
            return v
    return None


class Folder:
    """One image model's adapter folder: the files its DiT takes, as the listing, and the lookup a request's names go
    through.  Read once per CACHE_S seconds, and at once when the folder's mtime changes."""

    def __init__(self, cfg: dict):
        self.family = cfg.get("family") or "qwen-image"
        self.takes = self.family in TARGETS
        given = cfg.get("lora_dir")
        if isinstance(given, str) and given:
            self.dir = Path(given)
        else:   # "loras" beside the DiT's file (a folder of shards: beside the folder)
            dit = cfg.get("diffusion_model")
            self.dir = (Path(dit).parent if isinstance(dit, str) and dit else Path(".")) / "loras"
        self._listing: list[dict] = []
        self._by_name: dict[str, dict] = {}
        self._by_path: dict[str, dict] = {}
        self.skipped: dict[str, str] = {}   # a file that is not this model's, and why (a request's error says so)
        self.ignored: dict[str, list[str]] = {}   # an adapter's keys that hold no role: said, never applied silently
        self._mtime: int | None = None
        self._read = 0.0
        self._scan(force=True)

    def __repr__(self) -> str:
        return f"Folder({self.family}, {str(self.dir)!r}, {len(self._listing)} adapters)"

    def _scan(self, force: bool = False) -> None:
        try:
            mtime = self.dir.stat().st_mtime_ns if self.dir.is_dir() else None
        except OSError:
            mtime = None
        if not force and mtime == self._mtime and time.monotonic() - self._read < CACHE_S:
            return
        self._mtime, self._read = mtime, time.monotonic()
        made: list[dict] = []
        skipped: dict[str, str] = {}
        if self.takes and mtime is not None:
            for p in sorted(self.dir.rglob("*")):
                parts = p.relative_to(self.dir).parts
                if p.suffix.lower() not in EXTS or not p.is_file() or any(x.startswith(".") for x in parts):
                    continue
                rel = "/".join(parts)
                name = rel[: -len(p.suffix)]
                try:
                    entry = self._entry(p, rel, name)
                except (OSError, ValueError, struct.error) as exc:
                    skipped[name] = f"its header could not be read ({exc})"
                    continue
                if isinstance(entry, str):
                    skipped[name] = entry
                    continue
                made.append(entry)
        # the same stem twice (a.safetensors beside a.gguf): each keeps its extension
        repeated = collections.Counter(x["name"] for x in made)
        self._listing, self._by_name, self._by_path, self.ignored = [], {}, {}, {}
        for entry in made:
            left = entry.pop("ignored", [])
            if repeated[entry["name"]] > 1:
                entry = {**entry, "name": entry["path"]}
            if left:
                self.ignored[entry["name"]] = left   # a request says them (images.py: xestrata-engine.lora_ignored)
            self._listing.append(entry)
            self._by_name[entry["name"]] = entry
            self._by_path[entry["path"]] = entry
        self.skipped = skipped

    def _entry(self, path: Path, rel: str, name: str) -> dict | str:
        """One file's listing entry, or why this model does not take it."""
        tensors, meta = read_header(path)
        weights: dict[str, dict[str, tuple[int, ...]]] = {}
        ignored: list[str] = []
        under: dict[str, str] = {}   # an underscore name resolved once, not once a tensor of that weight
        for raw, shape in tensors.items():
            key, role, underscore = split_key(raw)
            if not role:   # the file's own (a quantization's, a config's) or one this engine applies not yet
                ignored.append(raw)
                continue
            if underscore:   # sd-scripts' and the LyCORIS form's names, brought to this model's
                if key not in under:
                    under[key] = kohya_name(key, self.family) or key
                key = under[key]
            if is_training_aid(key):   # trained beside the adapter, not one of the model's weights
                ignored.append(raw)
                continue
            weights.setdefault(key, {})[role] = shape
        if not weights:
            return "it holds no adapter tensors"
        peft = _peft_meta(meta)
        if _meta_flag(peft, "use_rslora"):
            return ("the adapter is rsLoRA (use_rslora), whose scale is alpha / sqrt(rank), not the plain "
                    "alpha / rank the image engine applies")
        alphas = set(_meta_alphas(peft))
        # a weight without its own ".alpha" is the one that would take the metadata's, so a file whose weights all
        # carry an ".alpha" needs no answer here either (the engine's rule, apply_peft_metadata, is the same one)
        if len(alphas) > 1 and any("alpha" not in roles for roles in weights.values()):
            return (f"its metadata names {len(alphas)} different lora_alpha values (a text encoder's and a "
                    "transformer's in one file): which one the image weights take is not said")
        targets: set[str] = set()
        rank = 0
        for key, roles in sorted(weights.items()):
            if is_text_encoder(key):
                if self.family not in TEXT_ENCODER_FAMILIES:
                    return f"{key}: a text encoder's weight, which the image engine applies not yet for this model"
                if not _TE_NAME.match(text_encoder_name(key)):
                    return f"{key} is not one of this image model's text encoder's weights"
                te_form = form_of(roles)
                if te_form not in ("lora", "lokr", "loha", "diff") or "dora_scale" in roles or \
                        "diff_b" in roles or "b_norm" in roles or "set_weight" in roles:
                    return f"{key}: the text encoder takes a plain LoRA, a LoKr, a LoHa or a whole delta"
                if te_form == "lora" and roles.get("lora_A") is not None and len(roles["lora_A"]) == 2:
                    rank = max(rank, roles["lora_A"][0])
                targets.add("text_encoder." + re.sub(r"^layers\.\d+\.", "", text_encoder_name(key)))
                continue
            # a bias's delta (.diff_b, or .b_norm beside the weight's w_norm, which .diff_b overrides): the
            # engine adds it where the DiT adds the weight's own bias, and only the weights that add one take it
            bias = roles.get("diff_b", roles.get("b_norm"))
            if bias is not None:
                if len(bias) != 1:
                    return f"{key}: a bias's delta (diff_b/b_norm) must be the bias's vector"
                if "diff_b" not in roles and "w_norm" not in roles:
                    return (f"{key}: a bias's delta (b_norm) without the weight's w_norm, beside which the common "
                            f"readers take it")
                if not any(m.match(key) for m in _BIASED[self.family]):
                    return f"{key}: a bias's delta (diff_b/b_norm), and this model's weight adds no bias"
            form = form_of(roles)
            if not form:
                if bias is None:
                    return f"{key}: a non-LoRA adapter, which the image engine applies not yet"
                if not any(m.match(key) for m in _MATCH[self.family]):   # a bias's delta alone: it changes no weight
                    return f"{key} is not one of this image model's weights"
                targets.add(kind_of(key))
                continue
            if form not in ("lora", "lokr", "loha", "glora", "diff", "oft"):
                return f"{key}: a {form} adapter, which the image engine applies not yet"
            if "dora_scale" in roles and form != "lora":
                return f"{key}: a DoRA (dora_scale) on a {form} adapter, which the image engine applies not yet"
            if not any(m.match(key) for m in _MATCH[self.family]):
                return f"{key} is not one of this image model's weights"
            if form == "diff":
                # the weight's own matrix (set_weight is the weight itself, diff and w_norm a delta), one a key, read
                # in that order; its rows and columns are the DiT's, which the engine checks
                whole = roles.get("set_weight", roles.get("diff", roles.get("w_norm")))
                if whole is None or len(whole) != 2:
                    return f"{key}: a whole delta (diff/set_weight) must be the weight's matrix"
            elif form == "lokr":
                r, why = _lokr_rank(roles)
                if why:
                    return f"{key}: {why}"
                rank = max(rank, r)
            elif form == "loha":
                r, why = _loha_rank(roles)
                if why:
                    return f"{key}: {why}"
                rank = max(rank, r)
            elif form == "glora":
                r, why = _glora_rank(roles)
                if why:
                    return f"{key}: {why}"
                rank = max(rank, r)
            elif form == "oft":
                # a block's rows (the listing's rank is the blocks' factorization's, which is none)
                _, why = _oft_shape(roles)
                if why:
                    return f"{key}: {why}"
                if self.family == "qwen-image" and kind_of(key) == "modulation.1":
                    # the DiT writes the modulation a band of rows at a time, and a turn is of a part's rows whole
                    return f"{key}: an OFT of the modulation, which the image engine does not turn"
            else:
                a, b = roles.get("lora_A"), roles.get("lora_B")
                if a is None or b is None:
                    return f"{key}: a LoRA weight needs both lora_A (down) and lora_B (up)"
                if len(a) != 2 or len(b) != 2:
                    return f"{key}: lora_A and lora_B must be matrices"
                if b[1] != a[0]:
                    return f"{key}: lora_B is {b[0]} x {b[1]}, not the rank its lora_A has ({a[0]})"
                way = dora_way(roles["dora_scale"], b[0], a[1]) if "dora_scale" in roles else "rows"
                if way == "columns":   # the engine's DoRA scales a product's rows, c = dora_scale / (|W| a row + eps)
                    return (f"{key}: a DoRA (dora_scale) over the weight's columns, which the engine's DoRA "
                            f"does not take (it scales a product's rows)")
                if way != "rows":
                    return f"{key}: dora_scale is neither the weight's rows nor its columns"
                rank = max(rank, a[0])
            targets.add(kind_of(key))
        kind = (meta.get(KIND_KEY) or "").strip().lower()
        try:
            steps = int(meta.get("ss_steps", ""))
        except ValueError:
            steps = None
        return {"name": name, "path": rel, "rank": rank, "scale": _recommended(meta), "trigger": _trigger(meta),
                "kind": kind if kind in KINDS else None, "steps": steps,
                "targets": sorted(targets), "size_mib": round(path.stat().st_size / 1048576, 1),
                "ignored": sorted(ignored)}

    def listing(self) -> list[dict]:
        """This model's adapters, each a listing entry (a copy: a caller may not change the kept one)."""
        self._scan()
        return [dict(e) for e in self._listing]

    def find(self, name: str) -> dict | None:
        """The adapter a request's name is, or None (see skipped for why a file of the folder is not one)."""
        self._scan()
        return self._by_name.get(name.strip())

    def file_of(self, entry: dict) -> str:
        """The file a listing entry names: an absolute path (the engine reads it from another process, whose working
        directory is not this one's)."""
        return str(self.dir.absolute() / entry["path"])

    def local(self, local_path: str) -> tuple[dict | None, str]:
        """A request's local_path -> (the listing entry, "") or (None, why it is not taken).  Only a file inside the
        folder is taken: the path is resolved - a link's target too - and must stay there."""
        try:
            root = self.dir.resolve()
        except OSError:
            return None, "the LoRA folder could not be read"
        p = Path(local_path)
        try:
            full = (p if p.is_absolute() else root / p).resolve()
        except (OSError, RuntimeError):
            return None, "the path could not be read"
        if root not in full.parents:
            return None, "it is not inside the model's LoRA folder"
        if not full.is_file():
            return None, "it is not a file inside the model's LoRA folder"
        rel = full.relative_to(root).as_posix()
        self._scan()
        entry = self._by_path.get(rel)
        if entry is not None:
            return entry, ""
        return None, self.skipped.get(rel[: -len(full.suffix)], "it is not an adapter this model applies")


_FOLDERS: dict[tuple[str, str], Folder] = {}


def folder(cfg: dict) -> Folder:
    """The folder of `cfg`'s image model, kept: a server asks for it with each request and /health."""
    fam = cfg.get("family") or "qwen-image"
    given = cfg.get("lora_dir")
    dirname = given if isinstance(given, str) and given else str(
        (Path(cfg["diffusion_model"]).parent if isinstance(cfg.get("diffusion_model"), str)
         and cfg["diffusion_model"] else Path(".")) / "loras")
    key = (str(Path(dirname).absolute()), fam)
    f = _FOLDERS.get(key)
    if f is None:
        f = _FOLDERS[key] = Folder(cfg)
    return f


# ------------------------------------------------------------------------------------------------- a request's
def _no_loras(f: Folder) -> str:
    name = FAMILY_NAMES.get(f.family, f.family)
    return f"lora: {name} takes no adapters (its DiT has no adapter path in the image engine)"


def _unknown(f: Folder) -> str:
    names = [e["name"] for e in f.listing()]
    if not names:
        return "the model's LoRA folder holds no file its DiT takes"
    return "the model's adapters are " + ", ".join(names[:6]) + (f" (and {len(names) - 6} more)" if len(names) > 6
                                                                 else "")


def _strength(value, where: str, default: float) -> float:
    """A strength (or a form field's text, or a tag's) from SCALE_MIN to SCALE_MAX; `default` when none."""
    if value is None:
        return default
    v = _finite(value)
    if v is None or not SCALE_MIN <= v <= SCALE_MAX:
        raise LoraError(f"{where}: a number from {SCALE_MIN:g} to {SCALE_MAX:g}")
    return v


def _items(value) -> list:
    """A "lora" field as the adapters' list: a list or one object, or one of those as JSON text (a form's field), or a
    name alone."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                value = json.loads(text)
            except ValueError:
                raise LoraError('lora: a JSON list of {"name", "scale"} or one such object') from None
        else:
            return [text] if text else []
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise LoraError('lora: a list of {"name", "scale"} or one such object')


def blocks(value, f: Folder, where: str) -> list[float] | None:
    """A request's block weights -> the strength's factor each DiT block (None: every block at 1): a list of numbers,
    one a block (the engine checks their count against the DiT's; as text, with commas), or a name of BLOCK_PRESETS
    for the model's family ("all": none)."""
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+0-9.eE, ]+", text):   # "0,0,1,..." (a tag's, a tool call's)
            text = f"[{text}]"
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError:
                raise LoraError(f"{where}: a list of numbers, one a block, or a name") from None
        else:
            presets = BLOCK_PRESETS.get(f.family, {})
            if text == "all":
                return None
            if text not in presets:
                names = ", ".join(["all", *presets])
                raise LoraError(f"{where}: {text!r} is not one of {FAMILY_NAMES.get(f.family, f.family)}'s block "
                                f"weights ({names}); a list of numbers, one a block, gives them")
            return list(presets[text])
    if not isinstance(value, (list, tuple)) or not value:
        raise LoraError(f"{where}: a list of numbers, one a block, or a name")
    out = []
    for v in value:
        x = _finite(v)
        if x is None or isinstance(v, bool) or not SCALE_MIN <= x <= SCALE_MAX:
            raise LoraError(f"{where}: each block's weight a number from {SCALE_MIN:g} to {SCALE_MAX:g}")
        out.append(x)
    return out


def one(item, f: Folder) -> dict:
    """One adapter a request names -> {"name", "path", "scale"} and "blocks" where it gives them ("path": its file,
    absolute)."""
    if isinstance(item, str):
        item = {"name": item}
    if not isinstance(item, dict):
        raise LoraError('lora: an object {"name": "gouache", "scale": 0.8} (a string is a name alone)')
    name, local = item.get("name"), item.get("local_path")
    entry, why = None, ""
    if local is not None:
        if not isinstance(local, str) or not local.strip():
            raise LoraError("lora: local_path: a file's path inside the model's LoRA folder")
        entry, why = f.local(local)
        if entry is None:
            raise LoraError(f"lora: local_path {local!r}: {why}")
    elif isinstance(name, str) and name.strip():
        entry = f.find(name)
        if entry is None:
            raise LoraError(f"lora: {name.strip()!r}: {f.skipped.get(name.strip()) or _unknown(f)}")
    else:
        raise LoraError('lora: each adapter needs a "name" (one the model lists) or a "local_path"')
    use = {"name": entry["name"], "path": f.file_of(entry),
           "scale": _strength(item.get("scale"), f"lora: {entry['name']}: scale", 1.0)}
    weights = blocks(item.get("blocks"), f, f"lora: {entry['name']}: blocks")
    if weights is not None:
        use["blocks"] = weights
    return use


def limit(uses: list[dict]) -> None:
    """At most MAX_LORA adapters stack in one request."""
    if len(uses) > MAX_LORA:
        raise LoraError(f"lora: at most {MAX_LORA} adapters at once ({len(uses)} are named)")


def field(value, f: Folder) -> list[dict]:
    """A request's "lora" field -> the adapters it asks for (one()'s dicts, in the order given); LoraError for a name
    the model does not list, a repeat, more than MAX_LORA, or a strength outside SCALE_MIN..SCALE_MAX."""
    items = _items(value)
    if not items:
        return []
    if not f.takes:
        raise LoraError(_no_loras(f))
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        use = one(item, f)
        if use["name"] in seen:
            raise LoraError(f"lora: {use['name']!r} is named twice")
        seen.add(use["name"])
        out.append(use)
    limit(out)
    return out


def strip_tags(text: str, f: Folder) -> tuple[str, list[dict], list[str]]:
    """A prompt's <lora:name:weight> tags out of `text` -> (the text, the adapters the tags name in order - the first
    naming of a name wins, one()'s dicts - and the names the model does not list, which are left out and said in the
    response).  <lora:name:weight:weight> is A1111's text encoder's weight: not applied, and taken as the engine's other
    unknown fields are (the human's decision of 2026-10-29).  A weight that is not a number, or a tag on a model that
    takes no adapter at all: LoraError."""
    if "<" not in text:
        return text, [], []
    found = list(TAG.finditer(text))
    if not found:
        return text, [], []
    if not f.takes:
        raise LoraError(_no_loras(f) + ": take the prompt's tag out, or name another image model")
    uses: list[dict] = []
    unknown: list[str] = []
    for m in found:
        name = m.group(1).strip()
        entry = f.find(name)
        scale = _strength(m.group(2), f"<lora:{name}:weight>", 1.0)
        third = (m.group(3) or "").strip()
        weights = None
        if third.lower().startswith("lbw="):   # LoRA Block Weight's form: a name, or the numbers with commas
            weights = blocks(third[4:].strip(), f, f"<lora:{name}:weight:lbw=...>")
        if entry is None:
            unknown.append(name)
            continue
        if any(u["name"] == entry["name"] for u in uses):
            continue
        use = {"name": entry["name"], "path": f.file_of(entry), "scale": scale}
        if weights is not None:
            use["blocks"] = weights
        uses.append(use)
    return TAG.sub("", text), uses, unknown


# ------------------------------------------------------------------------------------------- the page's own adapters
# The web app adds an adapter to the image models' LoRA folders - a file it uploads, or one the server fetches from
# Hugging Face - and takes one away.  A file is kept only where an image model of this server takes it (Folder._entry,
# the listing's own check), under its own name made safe, and never in the place of one that is there; a file no model
# takes is removed again, with each model's reason.  The human's decision of 2026-11-02: an upload, a remove and
# Hugging Face's files (no other site).
ADD_MAX = 8 << 30                  # the largest adapter added (a rank-256 LoRA of a 20B DiT is about 2.5 GiB in bf16)
ADD_SPARE = 1 << 30                # free disk an added file leaves
_UNSAFE = re.compile(r"[^A-Za-z0-9._ +-]+")
HF_FILE = re.compile(r"https://huggingface\.co/(?!datasets/|spaces/)([\w.-]+/[\w.-]+)/(?:resolve|blob)/([^/?#]+)/"
                     r"([^?#]+\.(?:safetensors|gguf))(?:[?#].*)?", re.IGNORECASE)


def file_name(name: str) -> str:
    """An added adapter's file name: the name given, without folders, its characters made safe."""
    base = Path(str(name or "").replace("\\", "/")).name.strip()
    stem, ext = os.path.splitext(base)
    if ext.lower() not in EXTS:
        raise LoraError(f"a LoRA is a {' or '.join(EXTS)} file, not {base or 'nothing'}")
    stem = _UNSAFE.sub("_", stem).strip(" ._")[:120]
    if not stem:
        raise LoraError("the file needs a name")
    return stem + ext.lower()


def _dirs(folders: list[Folder]) -> list[Path]:
    seen: list[Path] = []
    for f in folders:
        if f.dir.absolute() not in [d.absolute() for d in seen]:
            seen.append(f.dir)
    return seen


def staging(folders: list[Folder], size: int | None = None) -> Path:
    """Where an adapter on its way in is written: a hidden file in the first LoRA folder (the listing skips it), on
    the disk it will stay on.  LoraError when there is no folder, the file is too large or the disk too full."""
    takes = [f for f in folders if f.takes]
    if not takes:
        raise LoraError("no image model of this server takes a LoRA")
    if size is not None and size > ADD_MAX:
        raise LoraError(f"the file is {size >> 20} MiB; an added LoRA is at most {ADD_MAX >> 20} MiB")
    d = takes[0].dir
    d.mkdir(parents=True, exist_ok=True)
    if size is not None and shutil.disk_usage(d).free < size + ADD_SPARE:
        raise LoraError(f"the disk of {d} has too little room for {size >> 20} MiB")
    return d / f".adding-{secrets.token_hex(8)}.part"


def add(folders: list[Folder], tmp: Path, name: str) -> dict:
    """Keep the file at `tmp` as `name` where an image model takes it (moved into its folder, copied into a second
    folder that takes it too), or remove it: {"path", "name", "models": the families that take it}.  LoraError with
    each model's reason when none does, or when the name is there already."""
    try:
        final = file_name(name)
        stem = final[: -len(Path(final).suffix)]
        takes, why = [], []
        for f in folders:
            if not f.takes:
                why.append(f"{FAMILY_NAMES.get(f.family, f.family)} takes no LoRA")
                continue
            try:
                entry = f._entry(tmp, final, stem)
            except (OSError, ValueError, struct.error) as exc:
                entry = f"its header could not be read ({exc})"
            if isinstance(entry, str):
                why.append(f"{FAMILY_NAMES.get(f.family, f.family)}: {entry}")
            else:
                takes.append(f)
        if not takes:
            raise LoraError("no image model here takes this file - " + "; ".join(why))
        dirs = _dirs(takes)
        for d in dirs:
            if (d / final).exists():
                raise LoraError(f"{final} is in the LoRA folder already (remove it first, or give the file another "
                                "name)")
        for i, d in enumerate(dirs):
            d.mkdir(parents=True, exist_ok=True)
            if i + 1 < len(dirs):
                shutil.copyfile(tmp, d / final)
            else:
                try:
                    os.replace(tmp, d / final)
                except OSError as e:
                    if e.errno != errno.EXDEV:
                        raise
                    # Another model's folder can be on a different filesystem.
                    local = d / f".adding-{secrets.token_hex(8)}.part"
                    try:
                        shutil.copyfile(tmp, local)
                        os.replace(local, d / final)
                    finally:
                        local.unlink(missing_ok=True)
    finally:
        if tmp.exists():
            tmp.unlink()
    for f in folders:
        f._scan(force=True)
    return {"path": final, "name": stem, "models": sorted({f.family for f in takes})}


def remove(folders: list[Folder], path: str) -> dict:
    """Take an adapter away: the file `path` (a listing entry's) from every LoRA folder that has it."""
    gone = []
    for d in _dirs(folders):
        try:
            root = d.resolve()
            full = (root / str(path or "")).resolve()
        except (OSError, RuntimeError):
            continue
        if root in full.parents and full.suffix.lower() in EXTS and full.is_file():
            full.unlink()
            gone.append(str(full.relative_to(root).as_posix()))
    if not gone:
        raise LoraError(f"there is no LoRA {path!r} in the LoRA folder")
    for f in folders:
        f._scan(force=True)
    return {"removed": gone[0]}


def hf_file(url: str) -> tuple[str, str]:
    """A Hugging Face file's address (a /blob/ or /resolve/ link of a model repository) -> (its download URL, its file
    name).  LoraError for anything else."""
    m = HF_FILE.fullmatch(str(url or "").strip())
    if not m:
        raise LoraError("a Hugging Face file's link: https://huggingface.co/<owner>/<repo>/blob/<revision>/<file>"
                        f"{' or '.join(EXTS)}")
    repo, rev, path = m.groups()
    return f"{download_core.hf_endpoint()}/{repo}/resolve/{rev}/{path}", Path(path).name


class Fetch:
    """One adapter the server fetches from Hugging Face, on its own thread: its state for the page to read."""
    CHUNK = 1 << 20

    def __init__(self, folders: list[Folder], url: str):
        self.folders = folders
        self.url, self.name = hf_file(url)
        self.id = secrets.token_hex(8)
        self.state, self.got = "running", 0
        self.total: int | None = None
        self.error: str | None = None
        self.added: dict | None = None
        self.cancel = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="xestrata-lora-fetch").start()

    def view(self) -> dict:
        return {"id": self.id, "name": self.name, "state": self.state, "bytes": self.got, "total": self.total,
                "error": self.error, "added": self.added}

    def _download(self, tmp: Path, tag_path: Path) -> None:
        for attempt in range(5):
            if self.cancel.is_set():
                raise LoraError("cancelled")
            have = tmp.stat().st_size if tmp.exists() else 0
            tag = tag_path.read_text(encoding="utf-8") if have and tag_path.exists() else ""
            if not tag or tag.startswith("W/"):
                have = 0
            headers = {"User-Agent": "XeStrata"}
            if have:
                headers.update({"Range": f"bytes={have}-", "If-Range": tag})
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             (r.headers.get("Content-Range") or "").strip())
                        if not match or not download_core.range_matches(r.headers.get("Content-Range"), have) or \
                                r.headers.get("ETag") != tag:
                            tmp.unlink(missing_ok=True)
                            tag_path.unlink(missing_ok=True)
                            raise OSError("the server answered a different file or range")
                        self.total = int(match[3])
                    elif r.status == 200:
                        have = 0
                        length = r.headers.get("Content-Length")
                        self.total = int(length) if length and length.isdigit() else None
                        tag = r.headers.get("ETag") or ""
                        # The old bytes must be gone before a new validator can name this partial file.
                        tmp.write_bytes(b"")
                        if tag and not tag.startswith("W/"):
                            tag_path.write_text(tag, encoding="utf-8")
                        else:
                            tag_path.unlink(missing_ok=True)
                    else:
                        raise OSError(f"unexpected HTTP {r.status}")
                    if self.total is not None and self.total > ADD_MAX:
                        raise LoraError(f"the file is {self.total >> 20} MiB; an added LoRA is at most "
                                        f"{ADD_MAX >> 20} MiB")
                    staging(self.folders, max(0, self.total - have) if self.total is not None else None)
                    self.got = have

                    def report(done):
                        self.got = done
                        if done > ADD_MAX:
                            raise LoraError(f"the file is over {ADD_MAX >> 20} MiB, the most an added LoRA may be")

                    with open(tmp, "ab" if have else "wb") as out:
                        self.got = download_core.copy_stream(r, out, have=have, chunk=self.CHUNK,
                                                             progress=report, stop=self.cancel.is_set)
                    if self.cancel.is_set():
                        raise LoraError("cancelled")
                    if self.total is not None and self.got != self.total:
                        raise OSError(f"the download ended at {self.got} of {self.total} bytes")
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 416 or not have:
                    raise
                match = re.fullmatch(r"bytes \*/(\d+)", (exc.headers.get("Content-Range") or "").strip())
                if match and int(match[1]) == have and exc.headers.get("ETag") == tag:
                    self.got = self.total = have
                    return
                # The completed part cannot be identified as this version; fetch it again.
                tmp.unlink(missing_ok=True)
                tag_path.unlink(missing_ok=True)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def _run(self) -> None:
        tmp = tag_path = None
        lock = None
        try:
            stage = staging(self.folders)
            key = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:32]
            tmp = stage.parent / f".adding-{key}.part"
            tag_path = stage.parent / f".adding-{key}.etag"
            lock = open(stage.parent / f".adding-{key}.lock", "a+b")
            while True:
                try:
                    file_lock.acquire(lock)
                    break
                except OSError:
                    if self.cancel.is_set():
                        raise LoraError("cancelled")
                    time.sleep(0.25)
            self._download(tmp, tag_path)
            self.added = add(self.folders, tmp, self.name)
            tag_path.unlink(missing_ok=True)
            model_terms.record_lora_source([f.dir / self.added["path"] for f in self.folders
                                            if (f.dir / self.added["path"]).is_file()], self.url)
            self.state = "done"
        except urllib.error.HTTPError as exc:
            self.state, self.error = "failed", f"Hugging Face answered {exc.code} ({exc.reason}) for {self.url}"
        except (urllib.error.URLError, OSError, LoraError, ValueError, http.client.HTTPException) as exc:
            self.state, self.error = "failed", str(getattr(exc, "reason", None) or exc)
        finally:
            if lock is not None:
                try:
                    file_lock.release(lock)
                except OSError:
                    pass
                lock.close()
            if tag_path is not None and (tmp is None or not tmp.exists()):
                tag_path.unlink(missing_ok=True)


class Fetches:
    """The fetches of this server, the newest 16 kept for their pages."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items: dict[str, Fetch] = {}

    def start(self, folders: list[Folder], url: str) -> Fetch:
        f = Fetch(folders, url)
        with self.lock:
            self.items[f.id] = f
            while len(self.items) > 16:
                old = next((k for k, v in self.items.items() if v.state != "running"), None)
                if old is None:
                    break
                del self.items[old]
        return f

    def get(self, fid: str) -> Fetch | None:
        with self.lock:
            return self.items.get(fid)


FETCHES = Fetches()
