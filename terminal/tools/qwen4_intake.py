#!/usr/bin/env python3
"""Intake for a new model: read GGUF headers (never the weights) and say what Strata would have to do to run them.

    python tools/qwen4_intake.py PATH [PATH ...]                 a folder, or one or more .gguf files (any shard of a set)
    python tools/qwen4_intake.py --hf REPO [--rev REV] [--dir D]  Hugging Face: HTTP range reads of each shard's header only
    options: --out FILE   (the JSON report; default qwen4_intake.json)   --sha256 (local files: hash them, slow)
             --no-tokenizer-build   --quiet

For each set of files (a split model is one set) it prints, and writes as JSON:

  * the architecture and every geometry key, against both families Strata runs (qwen4exp = include/strata/core/geometry.hpp's
    kCompiledGeometry; qwen35moe = the run-time shapes of src/qwen35 with its fixed limits);
  * the tensor layout against the layout tables of include/strata/core/family_layout.hpp: missing, reshaped and unknown tensors;
  * every quant type against what each backend (CUDA, HIP, SYCL, Metal) has kernels for, by tensor role;
  * the tokenizer (model, pre-tokenizer, vocabulary, special tokens, BOS/EOS) against Flash-Next's, and whether
    tools/strata_tokenizer.py can build it; the chat template's hash, tool-call form and thinking tags;
  * the MTP block, the vision projector, split shards, sizes, sha256 where known, the general.* names and licence;
  * a verdict: case A (runs now: which path and backends), B (known layer types at a new shape or under a new name: the fields
    that differ) or C (each blocker), and a DRAFT catalog entry and setup config.  Nothing here is wired into setup.py.

The tables below are ports of C++ and kernel-source lists.  tools/test_qwen4_intake.py reads the C++ files and fails when a table
here no longer matches them, so they cannot drift quietly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402
import quant_formats as QF  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

TOOL_VERSION = 1

# ====================================================================================================== geometry (port)
# include/strata/core/geometry.hpp: ModelGeometry.  The defaults are Qwen3.8-Flash-Next's, the shape the kernels are compiled for.


@dataclasses.dataclass
class Geometry:
    n_embd: int = 2560
    n_layers: int = 48
    qsa_interval: int = 4
    n_vocab: int = 248320
    rms_eps: float = 1e-6
    ssm_state_size: int = 128
    ssm_k_heads: int = 16
    ssm_v_heads: int = 48
    ssm_d_conv: int = 4
    ssm_conv_channels: int = 10240
    ssm_value_dim: int = 6144
    n_head: int = 24
    n_head_kv: int = 2
    head_dim: int = 256
    n_rot: int = 64
    idx_q_heads: int = 4
    idx_key_dim: int = 128
    idx_block: int = 4
    idx_top_k: int = 2048
    hc: int = 4
    hc_lr: int = 320
    n_expert: int = 512
    n_expert_used: int = 10
    n_ff: int = 640
    n_ff_shexp: int = 640
    family: str = "qwen4exp"
    n_mtp_layers: int = 0

    @property
    def has_indexer(self) -> bool:
        return self.idx_q_heads > 0

    @property
    def has_hc(self) -> bool:
        return self.hc > 0

    @property
    def hc_dim(self) -> int:
        return self.hc * self.n_embd

    def is_qsa(self, layer: int) -> bool:
        return layer % self.qsa_interval == self.qsa_interval - 1

    def kind(self, layer: int) -> str:
        if layer >= self.n_layers:
            return "mtp"
        return "attention" if self.is_qsa(layer) else "gdn"


COMPILED = Geometry()

# include/strata/artifact/model_family.hpp
FAMILY_ARCH = {"qwen4exp": "qwen4exp", "qwen35moe": "qwen35moe"}
FAMILY_TITLE = {"qwen4exp": "Qwen3.8-Flash-Next", "qwen35moe": "Qwen3.6-35B-A3B"}
# model_family_runs() is true for qwen4exp only; qwen35moe has the src/qwen35 path the CUDA and HIP builds switch to.  The SYCL
# build refuses it (model_family_not_supported); Metal has no such path (docs: "Metal and Intel Arc have no such path").
FAMILY_BACKENDS = {"qwen4exp": ["cuda", "hip", "sycl", "metal"], "qwen35moe": ["cuda", "hip"]}
# Other architectures Strata runs with an engine of its own: never a Qwen4 path.
OTHER_ENGINES = {"deepseek41": "strata-dsv41 (setup.py MULTI_FILE_FAMILIES): a separate engine, not the Qwen3.8 / Qwen3.6 paths"}

# read_geometry() (gguf_reader.hpp): key -> (field, required), under "<arch>."
QWEN4_KEYS = [
    ("block_count", "n_layers", True), ("embedding_length", "n_embd", True), ("expert_count", "n_expert", True),
    ("expert_used_count", "n_expert_used", True), ("attention.head_count", "n_head", True),
    ("attention.head_count_kv", "n_head_kv", True), ("full_attention_interval", "qsa_interval", False),
    ("ssm.state_size", "ssm_state_size", False), ("ssm.group_count", "ssm_k_heads", False),
    ("ssm.time_step_rank", "ssm_v_heads", False), ("ssm.conv_kernel", "ssm_d_conv", False),
    ("ssm.inner_size", "ssm_value_dim", False), ("attention.key_length", "head_dim", False),
    ("rope.dimension_count", "n_rot", False), ("attention.indexer.head_count", "idx_q_heads", False),
    ("attention.indexer.key_length", "idx_key_dim", False), ("attention.indexer.top_k", "idx_top_k", False),
    ("hyper_connection.count", "hc", False), ("hyper_connection.low_rank", "hc_lr", False),
    ("expert_feed_forward_length", "n_ff", False),
]
# geometry_unsupported() pins + compiled_geometry_mismatch(): the fields the kernels are compiled for, with the GGUF key they come from
PINNED = [
    ("block_count", "n_layers"), ("embedding_length", "n_embd"), ("full_attention_interval", "qsa_interval"),
    ("ssm.state_size", "ssm_state_size"), ("ssm.group_count", "ssm_k_heads"), ("ssm.time_step_rank", "ssm_v_heads"),
    ("ssm.conv_kernel", "ssm_d_conv"), ("ssm.inner_size", "ssm_value_dim"), ("attention.head_count", "n_head"),
    ("attention.head_count_kv", "n_head_kv"), ("attention.key_length", "head_dim"), ("rope.dimension_count", "n_rot"),
    ("attention.indexer.head_count", "idx_q_heads"), ("attention.indexer.key_length", "idx_key_dim"),
    ("attention.compress_ratios", "idx_block"), ("attention.indexer.top_k", "idx_top_k"),
    ("hyper_connection.count", "hc"), ("hyper_connection.low_rank", "hc_lr"), ("expert_feed_forward_length", "n_ff"),
    ("expert_shared_feed_forward_length", "n_ff_shexp"),
]
QWEN35_REQUIRED = [
    "block_count", "embedding_length", "attention.head_count", "attention.head_count_kv", "attention.key_length",
    "attention.value_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
    "expert_shared_feed_forward_length", "ssm.conv_kernel", "ssm.state_size", "ssm.group_count", "ssm.time_step_rank",
    "ssm.inner_size", "full_attention_interval", "nextn_predict_layers",
]
# src/qwen35/q36_engine.cu load(): "a geometry this driver's kernels do not take"
QWEN35_LIMITS = [("attention.key_length", "head_dim", "==", 256), ("ssm.state_size", "ssm_state_size", "==", 128),
                 ("ssm.conv_kernel", "ssm_d_conv", "==", 4), ("expert_count", "n_expert", "<=", 512),
                 ("expert_used_count", "n_expert_used", "<=", 15)]


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int_key(md: dict, key: str, required: bool, default, problems: list):
    v = md.get(key)
    if v is None:
        if required:
            problems.append("missing " + key)
        return default
    if not isinstance(v, int) or isinstance(v, bool):
        problems.append(key + " is not an integer")
        return default
    if v <= 0:
        problems.append("%s = %d, it must be positive" % (key, v))
        return default
    return v


def geometry_qwen4(md: dict, arch: str, ntokens: int | None) -> tuple[Geometry, list]:
    """gguf_reader.hpp read_geometry(), for the keys under "<arch>." (the real qwen4exp guard when arch is 'qwen4exp')."""
    g, problems, p = Geometry(), [], arch + "."
    for key, field, required in QWEN4_KEYS:
        setattr(g, field, _int_key(md, p + key, required, getattr(g, field), problems))
    vlen = _int_key(md, p + "attention.value_length", False, g.head_dim, problems)
    shexp = _int_key(md, p + "expert_shared_feed_forward_length", False, g.n_ff, problems)
    if problems:
        return g, problems
    if vlen != g.head_dim:
        problems.append("%sattention.value_length = %d and attention.key_length = %d: the attention kernels use one head width for both"
                        % (p, vlen, g.head_dim))
    g.n_ff_shexp = shexp
    if shexp != g.n_ff:
        problems.append("%sexpert_shared_feed_forward_length = %d and expert_feed_forward_length = %d: the shared expert is read "
                        "with the routed experts' width" % (p, shexp, g.n_ff))
    eps = md.get(p + "attention.layer_norm_rms_epsilon")
    if eps is not None:
        if not _num(eps) or not eps > 0:
            problems.append(p + "attention.layer_norm_rms_epsilon must be a positive number")
        else:
            g.rms_eps = float(eps)
    if ntokens:
        g.n_vocab = ntokens
    if g.n_head % g.n_head_kv:
        problems.append("%sattention.head_count = %d is not a multiple of head_count_kv = %d" % (p, g.n_head, g.n_head_kv))
    if g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads:
        problems.append("%sssm.inner_size = %d, but ssm.state_size x ssm.time_step_rank = %d: the GDN kernels take square heads"
                        % (p, g.ssm_value_dim, g.ssm_state_size * g.ssm_v_heads))
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim
    if g.n_expert_used > g.n_expert or g.n_expert_used > 64:
        problems.append("%sexpert_used_count = %d, the router takes 1..%d of %d experts"
                        % (p, g.n_expert_used, min(g.n_expert, 64), g.n_expert))
    if g.n_rot > g.head_dim or g.n_rot > g.idx_key_dim or g.n_rot % 2:
        problems.append("%srope.dimension_count = %d is not an even slice of the head widths" % (p, g.n_rot))
    ratios = md.get(p + "attention.compress_ratios")
    if ratios is not None:
        if not isinstance(ratios, list) or len(ratios) != g.n_layers:
            problems.append("%sattention.compress_ratios must list one entry for each of the %d layers" % (p, g.n_layers))
        elif any(ratios):                       # an all-zero list is a converter's placeholder (#2396): treated as absent
            block = 0
            for l, r in enumerate(ratios):
                if (r != 0) != g.is_qsa(l) or r < 0 or (block and r and r != block):
                    problems.append("%sattention.compress_ratios[%d] = %d, but full_attention_interval = %d makes layer %d %s"
                                    % (p, l, r, g.qsa_interval, l, "full attention" if g.is_qsa(l) else "linear attention"))
                    break
                if r:
                    block = r
            if block:
                g.idx_block = block
    return g, problems


def geometry_qwen35(md: dict, arch: str, ntokens: int | None) -> tuple[Geometry, list]:
    """family_layout.hpp geometry_from_gguf() after check_architecture_qwen35moe()."""
    p, problems = arch + ".", []
    for k in QWEN35_REQUIRED:
        v = md.get(p + k)
        if v is None:
            problems.append("missing " + p + k)
        elif not _num(v):
            problems.append(p + k + " is not a number")
    g = Geometry(family="qwen35moe")
    if problems:
        return g, problems
    mtp, nl = int(md[p + "nextn_predict_layers"]), int(md[p + "block_count"])
    if mtp >= nl:
        problems.append(p + "nextn_predict_layers is not smaller than " + p + "block_count")
    if int(md[p + "full_attention_interval"]) < 2:
        problems.append(p + "full_attention_interval must be at least 2 (the GDN / attention pattern)")
    g.n_embd, g.n_head, g.n_head_kv = int(md[p + "embedding_length"]), int(md[p + "attention.head_count"]), int(md[p + "attention.head_count_kv"])
    g.head_dim, g.n_expert, g.n_expert_used = int(md[p + "attention.key_length"]), int(md[p + "expert_count"]), int(md[p + "expert_used_count"])
    g.n_ff, g.n_ff_shexp = int(md[p + "expert_feed_forward_length"]), int(md[p + "expert_shared_feed_forward_length"])
    g.ssm_d_conv, g.ssm_state_size, g.ssm_k_heads = int(md[p + "ssm.conv_kernel"]), int(md[p + "ssm.state_size"]), int(md[p + "ssm.group_count"])
    g.ssm_v_heads, g.ssm_value_dim, g.qsa_interval = int(md[p + "ssm.time_step_rank"]), int(md[p + "ssm.inner_size"]), int(md[p + "full_attention_interval"])
    g.n_mtp_layers, g.hc, g.hc_lr, g.idx_q_heads, g.idx_key_dim = mtp, 0, 0, 0, 0
    g.n_layers = nl - mtp
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim
    if ntokens:
        g.n_vocab = ntokens
    v = md.get(p + "rope.dimension_count")
    if _num(v):
        g.n_rot = int(v)
    if (g.n_layers <= 0 or g.n_embd <= 0 or g.n_head <= 0 or g.n_head_kv <= 0 or g.head_dim <= 0 or g.n_expert <= 0 or
            g.n_expert_used <= 0 or g.n_expert_used > g.n_expert or g.n_ff <= 0 or g.ssm_state_size <= 0 or g.ssm_k_heads <= 0 or
            g.ssm_v_heads <= 0 or g.ssm_d_conv <= 0 or g.qsa_interval < 2):
        problems.append("a geometry key is zero or out of range")
        return g, problems
    if int(md[p + "attention.value_length"]) != g.head_dim:
        problems.append("attention.key_length != attention.value_length")
    if g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads:
        problems.append("ssm.inner_size != ssm.state_size * ssm.time_step_rank")
    if g.ssm_v_heads % g.ssm_k_heads:
        problems.append("ssm.time_step_rank is not a multiple of ssm.group_count")
    if g.n_head % g.n_head_kv:
        problems.append("attention.head_count is not a multiple of attention.head_count_kv")
    if g.n_layers % g.qsa_interval:
        problems.append("the layer count is not a multiple of full_attention_interval")
    return g, problems


def shape_differences(g: Geometry, arch: str) -> list[dict]:
    """Every pinned field that differs from the compiled shape (the engine's guard names only the first)."""
    out = []
    for key, field in PINNED:
        got, want = getattr(g, field), getattr(COMPILED, field)
        if field == "idx_block" and not g.has_indexer:
            continue
        if got != want:
            out.append({"key": arch + "." + key, "field": field, "got": got, "kernels_built_for": want})
    if abs(g.rms_eps - COMPILED.rms_eps) > 1e-6 * COMPILED.rms_eps:
        out.append({"key": arch + ".attention.layer_norm_rms_epsilon", "field": "rms_eps", "got": g.rms_eps,
                    "kernels_built_for": COMPILED.rms_eps})
    return out


def qwen35_limit_violations(g: Geometry, arch: str) -> list[dict]:
    out = []
    for key, field, op, want in QWEN35_LIMITS:
        got = getattr(g, field)
        if (op == "==" and got != want) or (op == "<=" and got > want):
            out.append({"key": arch + "." + key, "field": field, "got": got, "limit": "%s %d" % (op, want)})
    return out


# ====================================================================================================== layout tables (port)
# include/strata/core/family_layout.hpp layer_tensors() / global_tensors(); shapes are GGUF order (dim 0 contiguous).


def layer_tensors(g: Geometry, layer: int) -> list[tuple[str, list[int]]]:
    kind = g.kind(layer)
    N, NE, FF, FS = g.n_embd, g.n_expert, g.n_ff, g.n_ff_shexp
    t = [("ffn_gate_inp.weight", [N, NE]), ("ffn_gate_inp_shexp.weight", [N]), ("ffn_gate_shexp.weight", [N, FS]),
         ("ffn_up_shexp.weight", [N, FS]), ("ffn_down_shexp.weight", [FS, N]), ("ffn_gate_exps.weight", [N, FF, NE]),
         ("ffn_up_exps.weight", [N, FF, NE]), ("ffn_down_exps.weight", [FF, N, NE])]
    if g.has_hc:
        H = g.hc_dim
        t += [("hc_attn_down.weight", [H, g.hc_lr]), ("hc_attn_up.weight", [g.hc_lr, H]), ("hc_attn_inject.weight", [H, g.hc]),
              ("hc_attn_norm.weight", [H]), ("hc_ffn_down.weight", [H, g.hc_lr]), ("hc_ffn_up.weight", [g.hc_lr, H]),
              ("hc_ffn_inject.weight", [H, g.hc]), ("hc_ffn_norm.weight", [H])]
    else:
        t += [("attn_norm.weight", [N]), ("post_attention_norm.weight", [N])]
    if kind == "gdn":
        t += [("attn_qkv.weight", [N, g.ssm_conv_channels]), ("attn_gate.weight", [N, g.ssm_value_dim]),
              ("ssm_out.weight", [g.ssm_value_dim, N]), ("ssm_conv1d.weight", [g.ssm_d_conv, g.ssm_conv_channels]),
              ("ssm_alpha.weight", [N, g.ssm_v_heads]), ("ssm_beta.weight", [N, g.ssm_v_heads]), ("ssm_a", [g.ssm_v_heads]),
              ("ssm_dt.bias", [g.ssm_v_heads]), ("ssm_norm.weight", [g.ssm_state_size])]
    else:
        t += [("attn_q.weight", [N, 2 * g.n_head * g.head_dim]), ("attn_k.weight", [N, g.n_head_kv * g.head_dim]),
              ("attn_v.weight", [N, g.n_head_kv * g.head_dim]), ("attn_output.weight", [g.n_head * g.head_dim, N]),
              ("attn_q_norm.weight", [g.head_dim]), ("attn_k_norm.weight", [g.head_dim])]
        if g.has_indexer and kind == "attention":
            t += [("indexer.q_proj.weight", [N, g.idx_q_heads * g.idx_key_dim]), ("indexer.k_proj.weight", [N, g.idx_key_dim]),
                  ("indexer.q_norm.weight", [g.idx_key_dim]), ("indexer.k_norm.weight", [g.idx_key_dim])]
    if kind == "mtp":
        t += [("nextn.eh_proj.weight", [2 * N, N]), ("nextn.enorm.weight", [N]), ("nextn.hnorm.weight", [N]),
              ("nextn.shared_head_norm.weight", [N])]
    return t


def global_tensors(g: Geometry) -> list[tuple[str, list[int]]]:
    t = [("token_embd.weight", [g.n_embd, g.n_vocab]), ("output.weight", [g.n_embd, g.n_vocab])]
    if g.has_hc:
        t += [("output_hc_down.weight", [g.hc_dim, g.hc_lr]), ("output_hc_up.weight", [g.hc_lr, g.hc_dim]),
              ("output_hc_norm.weight", [g.hc_dim])]
    else:
        t.append(("output_norm.weight", [g.n_embd]))
    return t


# layout_table_complete(): only qwen35moe's table names every tensor.  Flash-Next's published files also carry these (its PLE and
# n-gram tables, which "the loaders check"): known, not errors.  Anything else is an unknown tensor.
QWEN4_UNLISTED = re.compile(r"^(per_layer_token_embd\.weight|blk\.\d+\.ple_(conv1d|key|value|norm_conv|norm_key|norm_query)\.weight)$")


def check_layout(tensors: dict[str, list[int]], g: Geometry, complete: bool) -> dict:
    wanted: dict[str, list[int]] = dict(global_tensors(g))
    for l in range(g.n_layers + g.n_mtp_layers):
        for name, shape in layer_tensors(g, l):
            wanted["blk.%d.%s" % (l, name)] = shape
    missing = sorted(n for n in wanted if n not in tensors)
    reshaped = [{"name": n, "got": tensors[n], "want": s} for n, s in wanted.items() if n in tensors and list(tensors[n]) != s]
    extra, known = [], []
    for n in sorted(tensors):
        if n in wanted:
            continue
        if not complete and QWEN4_UNLISTED.match(n):
            known.append(n)
        else:
            extra.append(n)
    return {"wanted": len(wanted), "found": len(wanted) - len(missing), "missing": missing, "reshaped": reshaped,
            "unknown": extra, "known_unlisted": len(known), "ok": not (missing or reshaped or extra)}


# ====================================================================================================== quant support (port)
# ggml type ids -> names come from tools/gguf_reader.py.  The lists below are copied from the kernel sources named beside them;
# tools/test_qwen4_intake.py parses those files and compares.

TYPE_NAME = G.GGML_TYPES
SMALL_OK = {0, 1, 30}            # F32, F16, BF16: norms, router, biases, the SSM vectors (q36_engine.cu f32(): F32 / BF16 / F16 / Q8_0)
MMVQ = {2, 6, 7, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29}        # native_mmvq_supported(): CUDA, SYCL and Metal host copy
BACKEND_LISTS = {
    # src/kernels/cuda/iq_kernels.cu: STRATA_GU_FMTS / STRATA_D_FMTS (STRATA_Q6K_EXPERTS=ON, the default) and is_iq().  The HIP
    # build compiles this same file: AMD gfx11 / gfx12 / gfx1151 differ in kernel modes and speed, not in this list.
    "cuda": {"gu": {16, 17, 18, 21, 22, 23, 29, 42, 12, 13, 14, 6, 2, 3, 8, 40, 144, 145, 147, 148, 149},
             "down": {20, 23, 42, 7, 6, 2, 3, 8, 13, 14, 40, 144, 146, 147, 148, 149},
             "is_iq": {16, 17, 18, 20, 21, 22, 23, 29, 42, 11, 12, 13, 14, 7, 6, 2, 3, 8, 40, 144, 145, 146, 147, 148, 149},
             "mmvq": MMVQ, "embed_extra": {30, 14}},
    # sycl/src/kernels/cuda/iq_kernels.dp.cpp (the SYCL port; it has no NVFP4 or the Gyro-L pair-trellis types)
    "sycl": {"gu": {16, 17, 18, 21, 22, 23, 29, 42, 12, 13, 14, 6, 2, 3, 8, 144, 145},
             "down": {20, 23, 42, 7, 6, 2, 3, 8, 13, 14, 144, 146},
             "is_iq": {144, 145, 146, 16, 17, 18, 20, 21, 22, 23, 29, 42, 11, 12, 13, 14, 7, 6, 2, 3, 8},
             "mmvq": MMVQ, "embed_extra": {30}},
    # metal/src/engine/iq_host.cpp (host helpers; the kernels are mac-kern's, metal/src/engine/window.cpp: not audited)
    "metal": {"gu": {16, 17, 18, 21, 22, 23, 29, 12, 13, 14, 42, 144, 145, 147, 148, 149, 8},
              "down": {23, 42, 144, 146, 147, 148, 149, 20, 7, 6, 8, 11, 12, 13, 14, 21},
              "is_iq": {16, 17, 18, 20, 21, 22, 23, 29, 42, 11, 12, 13, 14, 7, 6, 2, 3, 8, 144, 145, 146, 147, 148, 149},
              "mmvq": MMVQ, "embed_extra": {30}},
}
BACKEND_LISTS["hip"] = BACKEND_LISTS["cuda"]
BACKEND_SOURCES = {
    "cuda": "src/kernels/cuda/iq_kernels.cu (native_expert_supported), src/kernels/cuda/native_mmvq.cu (native_mmvq_supported)",
    "hip": "the same two files as cuda (the HIP build compiles them)",
    "sycl": "sycl/src/kernels/cuda/iq_kernels.dp.cpp, sycl/src/kernels/cuda/native_mmvq.dp.cpp",
    "metal": "metal/src/engine/iq_host.cpp, metal/src/engine/mmvq_host.cpp (host helpers; Metal GEMV kernels not audited)",
}


def type_name(t: int) -> str:
    return TYPE_NAME.get(t, "type%d" % t)


def block_elems(t: int) -> int:
    geom = G.BLOCK_GEOMETRY.get(type_name(t))
    return geom[0] if geom else 0


def expert_pair_ok(be: str, gu: int, down: int, n_embd: int, n_ff: int) -> bool:
    """native_expert_supported(gu_type, d_type, n_embd, n_ff)."""
    L = BACKEND_LISTS[be]
    qg = block_elems(gu) if gu in L["gu"] else 0
    qd = block_elems(down) if down in L["down"] else 0
    return bool(qg > 0 and qd > 0 and gu in L["is_iq"] and down in L["is_iq"] and n_embd % qg == 0 and n_ff % qd == 0 and
                n_embd % 256 == 0 and (n_ff * n_embd) % 256 == 0)


ROLE_PATTERNS = [
    ("routed_gate", re.compile(r"ffn_gate_exps\.weight$")), ("routed_up", re.compile(r"ffn_up_exps\.weight$")),
    ("routed_down", re.compile(r"ffn_down_exps\.weight$")),
    ("token_embedding", re.compile(r"^(token_embd|per_layer_token_embd)\.weight$")),
    ("indexer", re.compile(r"\.indexer\.(q|k)_proj\.weight$")),
]


def tensor_role(name: str, shape: list[int]) -> str:
    for role, pat in ROLE_PATTERNS:
        if pat.search(name):
            return role
    if len(shape) == 1 or re.search(r"(norm\.weight|_norm\.weight|ssm_a|ssm_dt\.bias|ffn_gate_inp\.weight|ffn_gate_inp_shexp\.weight|"
                                    r"ssm_alpha\.weight|ssm_beta\.weight|ssm_conv1d\.weight|ple_conv1d\.weight|hc_(attn|ffn)_inject\.weight)$", name):
        return "small"
    return "dense"


def analyse_quants(tensors: list[tuple[str, list[int], int]], g: Geometry | None, family: str | None) -> dict:
    """Each quant type against what each backend has kernels for, by role.  `tensors`: (name, shape, ggml type)."""
    by_type: dict[str, int] = {}
    roles: dict[str, dict[str, int]] = {}
    per_layer: dict[int, dict[str, int]] = {}
    for name, shape, ty in tensors:
        tn = type_name(ty)
        by_type[tn] = by_type.get(tn, 0) + 1
        role = tensor_role(name, shape)
        roles.setdefault(role, {})
        roles[role][tn] = roles[role].get(tn, 0) + 1
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if m:
            per_layer.setdefault(int(m.group(1)), {})[m.group(2)] = ty
    backends = ["cuda", "hip", "sycl", "metal"]
    result = {"types": dict(sorted(by_type.items(), key=lambda kv: -kv[1])), "roles": roles, "backends": {}, "unknown_types": [],
              "expert_combinations": []}
    combos: dict[tuple, list[int]] = {}
    for l, d in sorted(per_layer.items()):
        combos.setdefault((d.get("gate"), d.get("up"), d.get("down")), []).append(l)
    for (gt, ut, dt), layers in combos.items():
        result["expert_combinations"].append({"gate": type_name(gt) if gt is not None else None, "up": type_name(ut) if ut is not None else None,
                                              "down": type_name(dt) if dt is not None else None, "layers": len(layers),
                                              "first_layer": layers[0]})
    seen_unknown = {ty for _, _, ty in tensors if ty not in G.GGML_TYPES}
    anywhere = set(SMALL_OK) | {8}
    for L in BACKEND_LISTS.values():
        anywhere |= L["is_iq"] | L["mmvq"] | L["embed_extra"] | L["gu"] | L["down"]
    nk: dict[int, dict] = {}
    for name, shape, ty in tensors:
        if ty not in anywhere and ty in G.GGML_TYPES:
            e = nk.setdefault(ty, {"type": type_name(ty), "tensors": 0, "example": name})
            e["tensors"] += 1
    result["no_kernel_anywhere"] = list(nk.values())
    result["unknown_types"] = sorted(seen_unknown)
    allowed = FAMILY_BACKENDS.get(family or "", backends)
    for be in backends:
        L = BACKEND_LISTS[be]
        bad: list[dict] = []
        qwen35 = family == "qwen35moe"
        for name, shape, ty in tensors:
            role = tensor_role(name, shape)
            why = None
            if role == "small":
                # qwen35moe (q36_engine.cu upload_f32): F32, BF16, F16 or Q8_0.  Flash-Next dequantizes small tensors with the generic
                # dequantizer (dq_dispatch: the is_iq() list), which is why Gyro's Q5_K / Q6_K hyper-connection and SSM tensors run.
                if ty not in SMALL_OK and ty != 8 and (qwen35 or ty not in L["is_iq"]):
                    why = "a small tensor must be F32, F16, BF16 or Q8_0" + ("" if qwen35 else " or a type the generic dequantizer takes")
            elif role == "indexer":
                if ty not in SMALL_OK and ty not in L["mmvq"]:
                    why = "no GEMV kernel for this type"
            elif role == "token_embedding":
                if ty not in L["is_iq"] and ty not in L["embed_extra"]:
                    why = "not an embedding type (embed_type_supported)"
            elif role == "dense":
                ok = ty in L["mmvq"] or (not qwen35 and ty in SMALL_OK)       # Flash-Next also runs BF16 / F16 dense (bf16_gemv)
                if not ok:
                    why = "no native MMVQ kernel (native_mmvq_supported)"
            if why:
                bad.append({"tensor": name, "type": type_name(ty), "role": role, "why": why})
        if g is not None:
            for l, d in sorted(per_layer.items()):
                gt, ut, dt = d.get("gate"), d.get("up"), d.get("down")
                if None in (gt, ut, dt):
                    continue
                if gt != ut:
                    bad.append({"tensor": "blk.%d.ffn_gate_exps / ffn_up_exps" % l, "type": type_name(gt) + "/" + type_name(ut),
                                "role": "routed", "why": "gate and up differ in type"})
                elif not expert_pair_ok(be, gt, dt, g.n_embd, g.n_ff):
                    bad.append({"tensor": "blk.%d.ffn_*_exps" % l, "type": "%s/%s" % (type_name(gt), type_name(dt)), "role": "routed",
                                "why": "no expert kernel for this gate+up / down pair at n_embd %d, n_ff %d (native_expert_supported)"
                                       % (g.n_embd, g.n_ff)})
        # group the identical complaints
        grouped: dict[tuple, dict] = {}
        for b in bad:
            k = (b["type"], b["role"], b["why"])
            e = grouped.setdefault(k, {"type": b["type"], "role": b["role"], "why": b["why"], "tensors": 0, "example": b["tensor"]})
            e["tensors"] += 1
        result["backends"][be] = {"family_runs_here": be in allowed, "all_types_supported": not bad,
                                  "unsupported": list(grouped.values()), "source": BACKEND_SOURCES[be]}
    return result


# ====================================================================================================== tokenizer, template
# Flash-Next's tokenizer (recorded from unsloth/Qwen3.8-Flash-Next-GGUF; Qwen3.6-35B-A3B's is the same vocabulary)
REF_TOKENIZER = {
    "n_vocab": 248320, "n_merges": 247587, "pre": "qwen35",
    "tokens_sha256": "f307bf31f9c6130c016bc12538b9eea8877540d705167d1add3a2cb5605bf9a0",
    "merges_sha256": "4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332",
    "bos": 248044, "eos": 248046, "pad_flash_next": 248044, "pad_qwen36": 248055,
    "key_tokens": {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "<|vision_start|>": 248053,
                   "<|vision_end|>": 248054, "<|image_pad|>": 248056, "<tool_call>": 248058, "</tool_call>": 248059,
                   "<tool_response>": 248066, "</tool_response>": 248067, "<think>": 248068, "</think>": 248069},
}
KNOWN_TEMPLATES = {
    "12827f24b742": "Qwen3.8-Flash-Next (serve/chat_template.jinja; also the Unsloth, GSQ-RCO, Coder and Gyro files)",
    "c3cf9e34abf4": "Swift 1.5",
    "55d4931433fe": "Qwen3.6-35B-A3B (Unsloth MTP files)",
    "06cada875e3a": "DeepSeek-V4.1-Flash (Unsloth)",
}


def tokenizer_report(md: dict, build: bool, text_probe: bool = True) -> dict:
    import strata_tokenizer as ST
    toks = md.get("tokenizer.ggml.tokens")
    merges = md.get("tokenizer.ggml.merges")
    types = md.get("tokenizer.ggml.token_type")
    pre = md.get("tokenizer.ggml.pre")
    r: dict = {"model": md.get("tokenizer.ggml.model"), "pre": pre, "pre_rule": ST.pre_tokenizer_kind(pre),
               "n_tokens": len(toks) if isinstance(toks, list) else None, "n_merges": len(merges) if isinstance(merges, list) else None,
               "bos": md.get("tokenizer.ggml.bos_token_id"), "eos": md.get("tokenizer.ggml.eos_token_id"),
               "pad": md.get("tokenizer.ggml.padding_token_id"), "add_bos": md.get("tokenizer.ggml.add_bos_token")}
    if not isinstance(toks, list):
        r["problems"] = ["the file carries no tokenizer.ggml.tokens"]
        return r
    r["tokens_sha256"] = hashlib.sha256("\n".join(toks).encode()).hexdigest()
    if isinstance(merges, list):
        r["merges_sha256"] = hashlib.sha256("\n".join(merges).encode()).hexdigest()
    special = []
    counts: dict[int, int] = {}
    if isinstance(types, list):
        for i, ty in enumerate(types):
            counts[ty] = counts.get(ty, 0) + 1
            if ty in (3, 4) and i < len(toks):
                special.append((i, toks[i], ty))
    r["token_type_counts"] = {str(k): v for k, v in sorted(counts.items())}
    r["n_special"] = len(special)
    r["same_vocabulary_as_flash_next"] = (r["tokens_sha256"] == REF_TOKENIZER["tokens_sha256"] and
                                          r.get("merges_sha256") == REF_TOKENIZER["merges_sha256"])
    by_text = {t: i for i, t, _ in special}
    diffs = []
    for text, want in REF_TOKENIZER["key_tokens"].items():
        got = by_text.get(text)
        if got != want:
            diffs.append({"token": text, "flash_next_id": want, "this_file": got})
    r["key_token_differences"] = diffs
    r["end_of_turn_id"] = by_text.get("<|im_end|>")
    problems = []
    if r["model"] != "gpt2":
        problems.append("tokenizer.ggml.model is %r; tools/strata_tokenizer.py builds the byte-level BPE ('gpt2') only" % (r["model"],))
    if r["pre_rule"] is None:
        problems.append("tokenizer.ggml.pre is %r: no splitting rule for it in tools/strata_tokenizer.py (knows qwen35, deepseek*)" % (pre,))
    if not isinstance(merges, list):
        problems.append("the file carries no tokenizer.ggml.merges")
    r["problems"] = problems
    r["builds"] = None
    if build and not problems:
        t0 = time.time()
        try:
            sp = {k: int(md[k]) for k in md if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
            tk = ST.Tokenizer(list(toks), list(merges), list(types) if isinstance(types, list) else None, pre, sp)
            entry = {"ok": True, "seconds": round(time.time() - t0, 2)}
            if text_probe and all(ch in tk.ids for ch in G_BYTE_CHARS):
                probe = "def f(x):\n\treturn x  # café 你好\n"
                ids = tk.encode(probe)
                entry["roundtrip"] = tk.decode(ids) == probe
                entry["probe_tokens"] = len(ids)
            else:
                entry["roundtrip"] = "skipped (the vocabulary lacks the 256 byte-level tokens)"
            r["builds"] = entry
        except Exception as e:  # noqa: BLE001 - the report names it
            r["builds"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    return r


def _byte_chars():
    import strata_tokenizer as ST
    return sorted(ST.BYTE_TO_UNICODE.values())


G_BYTE_CHARS = _byte_chars()


def template_report(md: dict) -> dict:
    from serve import toolformat
    src = md.get("tokenizer.chat_template")
    if not isinstance(src, str):
        return {"present": False}
    h = hashlib.sha256(src.encode()).hexdigest()
    known = KNOWN_TEMPLATES.get(h[:12])
    r = {"present": True, "sha256": h, "chars": len(src), "known_as": known, "tool_call_format": toolformat.tool_format(src)}
    r.update(toolformat.thinking_tags(src))
    local = ROOT / "serve" / "chat_template.jinja"
    if local.is_file():
        r["same_as_serve_chat_template"] = hashlib.sha256(local.read_bytes()).hexdigest() == h
    return r


# ====================================================================================================== reading the headers
SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)


@dataclasses.dataclass
class FileInfo:
    path: str                       # a local path, or "repo:path" for Hugging Face
    name: str
    size: int
    header: object                  # a gguf_reader.GGUFFile / quant_formats.Header
    sha256: str | None = None
    sha256_source: str | None = None
    url: str | None = None

    @property
    def md(self) -> dict:
        return self.header.metadata

    @property
    def tensors(self):
        return self.header.tensors


def read_local(path: pathlib.Path) -> FileInfo:
    h = G.GGUFFile(path)
    return FileInfo(str(path), path.name, path.stat().st_size, h)


def file_sha256(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            d.update(b)
    return d.hexdigest()


def hf_list(repo: str, rev: str, subdir: str = "") -> list[dict]:
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = "%s/api/models/%s/tree/%s/%s?recursive=true" % (base, repo, urllib.parse.quote(rev, safe=""), urllib.parse.quote(subdir.strip("/")))
    out = []
    while url:
        req = urllib.request.urlopen(url, timeout=60)
        out += json.load(req)
        link = req.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return [e for e in out if e.get("type") == "file" and e["path"].lower().endswith(".gguf")]


def read_hf_entry(repo: str, rev: str, e: dict) -> FileInfo:
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = "%s/%s/resolve/%s/%s" % (base, repo, rev, urllib.parse.quote(e["path"]))
    h = QF.Header(e["path"], QF.RangeFile(url), int(e["size"]))
    oid = (e.get("lfs") or {}).get("oid")
    return FileInfo("%s:%s" % (repo, e["path"]), e["path"], int(e["size"]), h, oid, "Hugging Face LFS metadata" if oid else None, url)


def group_sets(files: list[FileInfo]) -> list[list[FileInfo]]:
    """A split model is one set (same folder, stem and shard count); everything else is a set of one."""
    sets: dict[tuple, list[FileInfo]] = {}
    order = []
    for f in files:
        m = SHARD_RE.match(os.path.basename(f.name))
        key = (os.path.dirname(f.path), m.group("stem"), m.group("n")) if m else (f.path,)
        if key not in sets:
            sets[key] = []
            order.append(key)
        sets[key].append(f)
    out = []
    for key in order:
        out.append(sorted(sets[key], key=lambda f: f.name))
    return out


# ====================================================================================================== the analysis
def _arch_of(md: dict) -> str | None:
    a = md.get("general.architecture")
    return a if isinstance(a, str) else None


def projector_report(f: FileInfo) -> dict:
    try:
        from strata_control.models import OUR_PROJECTOR_TYPES as ours
    except Exception:  # noqa: BLE001 - the report must run from a bare tools/ copy
        ours = {"qwen3vl_merger"}
    md = f.md
    pt = md.get("clip.projector_type")
    return {"file": f.name, "size": f.size, "projector_type": pt, "ours": sorted(ours), "supported": pt in ours,
            "projection_dim": md.get("clip.vision.projection_dim"), "image_size": md.get("clip.vision.image_size"),
            "block_count": md.get("clip.vision.block_count"), "tensors": len(f.tensors)}


def mtp_report(md: dict, arch: str | None, names: set[str], siblings: list[str]) -> dict:
    p = (arch or "") + "."
    n = md.get(p + "nextn_predict_layers")
    in_file = bool(n) or any(".nextn." in x for x in names)
    sep = [s for s in siblings if re.search(r"mtp|draft", s, re.I)]
    return {"in_file_layers": int(n) if isinstance(n, int) else 0, "nextn_tensors": sorted(x for x in names if ".nextn." in x)[:8],
            "in_file": in_file, "separate_drafter_files": sep}


def general_report(md: dict) -> dict:
    keys = ["name", "basename", "size_label", "quantized_by", "license", "license.link", "license.name", "repo_url", "url",
            "file_type", "description", "type", "version", "finetune"]
    out = {k: md["general." + k] for k in keys if ("general." + k) in md}
    bases = {k: v for k, v in md.items() if k.startswith("general.base_model.")}
    if bases:
        out["base_model"] = bases
    return out


def is_drafter(files: list[FileInfo]) -> bool:
    """A separate MTP drafter file (Unsloth's MTP/ folder, Gyro's mtp-*.gguf): the same architecture name as the model, but no
    trunk layers, only the draft block (Unsloth's mtp-*.gguf: token_embd, output and blk.48 of a 49-block file)."""
    names = {t.name for f in files for t in f.tensors}
    return any(n.startswith("blk.") for n in names) and not any(n.startswith("blk.0.") for n in names)


def drafter_report(files: list[FileInfo]) -> dict:
    md = files[0].md
    names = [t.name for f in files for t in f.tensors]
    blks = sorted({int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    types: dict[str, int] = {}
    for f in files:
        for t in f.tensors:
            types[t.type_name] = types.get(t.type_name, 0) + 1
    return {"files": [{"name": f.name, "path": f.path, "size": f.size, "tensors": len(f.tensors), "sha256": f.sha256} for f in files],
            "architecture": _arch_of(md), "drafter": True, "size_bytes": sum(f.size for f in files), "tensor_count": len(names),
            "block_indices": blks, "types": types, "general": general_report(md),
            "verdict": {"case": "-", "reading": "a separate MTP drafter file, not a model", "path": None, "backends": [], "blockers": [],
                        "warnings": [], "shape_fields_that_differ": [],
                        "notes": ["drafter blocks %s of arch '%s': the Flash-Next drafter is packed by tools/mtp_pack.py (not checked here)" % (blks, _arch_of(md))]}}


def analyse_set(files: list[FileInfo], siblings: list[str] | None = None, build_tokenizer: bool = True,
                projectors: list[dict] | None = None) -> dict:
    """The report for one set of files (the shards of one model).  Reads nothing but what the headers already hold."""
    md = files[0].md
    arch = _arch_of(md)
    names: dict[str, list[int]] = {}
    tlist: list[tuple[str, list[int], int]] = []
    for f in files:
        for t in f.tensors:
            names[t.name] = list(t.shape)
            tlist.append((t.name, list(t.shape), t.type_id))
    rep: dict = {"files": [], "architecture": arch, "tensor_count": len(names), "size_bytes": sum(f.size for f in files)}
    for f in files:
        rep["files"].append({"name": f.name, "path": f.path, "size": f.size, "tensors": len(f.tensors), "gguf_version": f.header.version,
                             "sha256": f.sha256, "sha256_source": f.sha256_source,
                             "split_no": f.md.get("split.no"), "split_count": f.md.get("split.count")})
    # ---- split completeness
    split = {"count": md.get("split.count"), "found": len(files), "complete": True, "problems": []}
    sc = md.get("split.count")
    if isinstance(sc, int) and sc > 1:
        nos = sorted(f.md.get("split.no") for f in files)
        if nos != list(range(sc)):
            split["complete"] = False
            split["problems"].append("shards present: %s of %d" % ([n + 1 for n in nos if isinstance(n, int)], sc))
        total = md.get("split.tensors.count")
        if isinstance(total, int) and total != len(names):
            split["complete"] = False
            split["problems"].append("split.tensors.count is %d, the shards hold %d" % (total, len(names)))
    elif len(files) > 1:
        split["problems"].append("several files but no split.count key")
    rep["split"] = split
    rep["general"] = general_report(md)
    # ---- the tokenizer and the template are in the metadata shard
    rep["tokenizer"] = tokenizer_report(md, build_tokenizer)
    rep["chat_template"] = template_report(md)
    ntok = rep["tokenizer"].get("n_tokens")
    # ---- family, geometry, layout
    cand = []                               # (family, geometry, problems, layout)
    pref = (arch or "") + "."
    for fam in ("qwen4exp", "qwen35moe"):
        if arch == fam:
            order = [fam]
        elif arch in OTHER_ENGINES or arch is None:
            order = []
        else:
            order = ["qwen4exp", "qwen35moe"]                  # an unknown name: do the keys and tensors fit a known family?
        if fam not in order:
            continue
        g, problems = (geometry_qwen4 if fam == "qwen4exp" else geometry_qwen35)(md, arch, ntok)
        layout = check_layout(names, g, complete=(fam == "qwen35moe")) if not problems else None
        cand.append((fam, g, problems, layout))
    best = None
    if cand:
        scored = []
        for fam, g, problems, layout in cand:
            score = (len(problems) > 0, 0 if layout is None else len(layout["missing"]) + len(layout["reshaped"]) + len(layout["unknown"]))
            scored.append((score, fam, g, problems, layout))
        scored.sort(key=lambda x: x[0])
        _, fam, g, problems, layout = scored[0]
        best = (fam, g, problems, layout)
    rep["family"] = best[0] if best else None
    rep["geometry"] = None
    rep["geometry_problems"] = best[2] if best else []
    rep["layout"] = best[3] if best else None
    rep["shape_differences"] = []
    rep["qwen35_limit_violations"] = []
    if best:
        fam, g, problems, layout = best
        rep["geometry"] = dataclasses.asdict(g)
        rep["geometry"]["keys"] = {k[len(pref):]: v for k, v in md.items() if k.startswith(pref) and not isinstance(v, list)}
        if not problems:
            rep["shape_differences"] = shape_differences(g, arch) if fam == "qwen4exp" else []
            if fam == "qwen35moe":
                rep["qwen35_limit_violations"] = qwen35_limit_violations(g, arch)
                ref = Geometry(n_embd=2048, n_layers=40, ssm_v_heads=32, ssm_value_dim=4096, n_head=16, n_ff=512, n_ff_shexp=512,
                               n_expert=256, n_expert_used=8, n_mtp_layers=1, family="qwen35moe", hc=0, hc_lr=0, idx_q_heads=0, idx_key_dim=0)
                rep["differs_from_qwen36_35b"] = [f.name for f in dataclasses.fields(Geometry)
                                                   if f.name not in ("n_vocab", "rms_eps", "n_rot", "idx_block", "idx_top_k", "ssm_conv_channels")
                                                   and getattr(g, f.name) != getattr(ref, f.name)]
    rep["quant"] = analyse_quants(tlist, best[1] if best and not best[2] else None, rep["family"])
    # embedding rows must match the vocabulary (a file whose tokenizer and embedding disagree is not usable)
    emb = names.get("token_embd.weight")
    rep["vocab_matches_embedding"] = None if not emb or not ntok else (emb[-1] == ntok)
    rep["mtp"] = mtp_report(md, arch, set(names), siblings or [])
    rep["projectors"] = projectors or []
    # ---- the verdict, the draft catalog entry and the setup config
    rep["verdict"] = verdict(rep, arch)
    rep["draft_catalog_entry"], rep["draft_setup_config"] = draft_entry(rep, files, arch)
    return rep


def verdict(rep: dict, arch: str | None) -> dict:
    blockers, warnings, notes = [], [], []
    fam = rep["family"]
    case, path, backends = "C", None, []
    tk, qt = rep["tokenizer"], rep["quant"]
    if arch is None:
        blockers.append("the file has no general.architecture")
    elif arch in OTHER_ENGINES:
        blockers.append("architecture '%s' is %s" % (arch, OTHER_ENGINES[arch]))
    # ---- the model part
    shape_fields = rep["shape_differences"]
    if fam is not None:
        if arch not in FAMILY_ARCH and (rep["geometry_problems"] or (rep["layout"] is not None and not rep["layout"]["ok"])):
            blockers.append("architecture '%s' is not a name Strata knows, and it does not fit %s either" % (arch, FAMILY_TITLE[fam]))
        if rep["geometry_problems"]:
            blockers += ["geometry: " + p for p in rep["geometry_problems"][:8]]
        lay = rep["layout"]
        if lay is not None and not lay["ok"]:
            if lay["unknown"]:
                blockers.append("%d tensors the %s layout does not name: %s" % (len(lay["unknown"]), FAMILY_TITLE[fam], ", ".join(lay["unknown"][:6])))
            if lay["missing"]:
                blockers.append("%d tensors the %s layout wants are missing: %s" % (len(lay["missing"]), FAMILY_TITLE[fam], ", ".join(lay["missing"][:6])))
            if lay["reshaped"]:
                r = lay["reshaped"][0]
                blockers.append("%d tensors have another shape than the %s layout (first: %s is %s, wants %s)"
                                % (len(lay["reshaped"]), FAMILY_TITLE[fam], r["name"], r["got"], r["want"]))
    elif arch is not None and arch not in OTHER_ENGINES:
        blockers.append("architecture '%s' is not a family Strata knows and no key set fits qwen4exp or qwen35moe" % arch)
    # ---- dense / expert-less models
    md_experts = rep["geometry"]["n_expert"] if rep["geometry"] else None
    if rep["quant"]["roles"].get("routed_gate") is None and rep["tensor_count"] and fam is not None:
        blockers.append("no routed expert tensors: a dense model is a new engine path")
    # ---- quant support
    allowed = FAMILY_BACKENDS.get(fam or "", [])
    usable = [b for b in allowed if qt["backends"][b]["all_types_supported"]]
    if fam is not None and not rep["geometry_problems"] and not usable:
        merged: dict[tuple, list] = {}
        for b in allowed:
            for u in qt["backends"][b]["unsupported"][:3]:
                merged.setdefault((u["role"], u["type"], u["tensors"], u["example"], u["why"]), []).append(b)
        for (role, ty, n, ex, why), bs in merged.items():
            blockers.append("%s: %s %s (%d tensors, e.g. %s): %s" % ("/".join(bs), role, ty, n, ex, why))
    elif fam is not None:
        for b in allowed:
            if b not in usable:
                us = qt["backends"][b]["unsupported"]
                notes.append("%s cannot run this file: %s" % (b, "; ".join("%s %s (%d)" % (u["role"], u["type"], u["tensors"]) for u in us[:4])))
    for u in qt["no_kernel_anywhere"]:
        blockers.append("quant type %s (%d tensors, e.g. %s) has no kernel in any backend's list" % (u["type"], u["tensors"], u["example"]))
    if qt["unknown_types"]:
        blockers.append("ggml type ids this tools/gguf_reader.py does not know: %s" % qt["unknown_types"])
    # ---- the tokenizer
    for p in tk.get("problems", []):
        blockers.append("tokenizer: " + p)
    if tk.get("builds") and not tk["builds"].get("ok"):
        blockers.append("tokenizer: tools/strata_tokenizer.py cannot build it: " + tk["builds"].get("error", ""))
    if tk.get("builds") and tk["builds"].get("ok") and tk["builds"].get("roundtrip") is False:
        blockers.append("tokenizer: encode + decode of a probe string does not return the string")
    if rep["vocab_matches_embedding"] is False:
        blockers.append("tokenizer: %s tokens but the embedding has another number of rows" % tk.get("n_tokens"))
    if tk.get("n_tokens") and not tk.get("same_vocabulary_as_flash_next") and not tk.get("problems"):
        notes.append("the vocabulary is not Flash-Next's (%d tokens): check the special tokens and stop ids; %d key tokens differ"
                     % (tk["n_tokens"], len(tk["key_token_differences"])))
    # ---- the template
    ct = rep["chat_template"]
    if not ct.get("present"):
        warnings.append("no chat template in the file")
    elif ct["tool_call_format"] in ("other",):
        warnings.append("the chat template documents tool calls in a form no parser handles (tool_call_format: other)")
    elif ct["tool_call_format"] == "json":
        notes.append("tool calls are the JSON form: the server's JSON parser is used (serve/frontend.py parser_for_template)")
    elif ct["tool_call_format"] == "dsml":
        warnings.append("DSML tool calls belong to the DeepSeek engine")
    # ---- split, projector
    if not rep["split"]["complete"]:
        blockers.append("split set incomplete: " + "; ".join(rep["split"]["problems"]))
    for pr in rep["projectors"]:
        if not pr["supported"]:
            warnings.append("picture reader %s has projector type %r, not %s: pictures off (a new vision tower)" % (pr["file"], pr["projector_type"], pr["ours"]))
    # ---- the case
    if blockers:
        case = "C"
    else:
        if fam == "qwen4exp":
            case = "B" if shape_fields else "A"
        elif fam == "qwen35moe":
            case = "B" if rep["qwen35_limit_violations"] else "A"
        if arch not in FAMILY_ARCH and fam is not None:
            case = "B"
            notes.append("known layer types under a new name: architecture '%s' must be added to model_family_from_arch (and its %s guard)" % (arch, fam))
        if fam == "qwen35moe":
            shape_fields = rep["qwen35_limit_violations"]
        path = {"qwen4exp": "the Flash-Next engine path (src/program, src/kernels)", "qwen35moe": "src/qwen35 (strata-q36)"}[fam]
        backends = usable
    if case == "B" and fam == "qwen4exp":
        notes.append("kernels are compiled for the Flash-Next shape (include/strata/core/geometry.hpp): a rebuild with another geometry header, "
                     "every platform; the engine refuses the file until then")
    if case == "A" and fam == "qwen4exp":
        # native router kernel: the (512 experts, top-10) or (256, top-8) fast path only; other counts take the generic router
        g = rep["geometry"]
        if (g["n_expert"], g["n_expert_used"]) not in ((512, 10), (256, 8)):
            notes.append("the native router kernel covers 512/top-10 and 256/top-8; this file has %d/top-%d (native_router.cu native_router_topk_supported)"
                         % (g["n_expert"], g["n_expert_used"]))
    return {"case": case, "path": path, "backends": backends, "blockers": blockers if case != "A" else [],
            "shape_fields_that_differ": shape_fields if case == "B" else [], "warnings": warnings, "notes": notes,
            "reading": {"A": "runs now", "B": "known layer types at a new shape or name", "C": "port first"}[case]}


# ====================================================================================================== the draft entry
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")


def draft_entry(rep: dict, files: list[FileInfo], arch: str | None) -> tuple[dict, dict]:
    """A catalog entry in the shape of setup.py's OWN_PATH_FAMILIES / the Unsloth tables, and the config it implies.  DRAFT: filled from
    the headers only; the numbers marked 'estimate' are not measured; nothing here is wired into setup.py."""
    gen = rep["general"]
    name = gen.get("name") or (arch or "model")
    fam = rep["family"]
    g = rep["geometry"] or {}
    shard_re = SHARD_RE.match(files[0].name)
    pattern = files[0].name
    if shard_re:
        pattern = "%s-{i:05d}-of-%s.gguf" % (shard_re.group("stem"), shard_re.group("n"))
    size = rep["size_bytes"]
    sha = {f["name"]: [f["size"], f["sha256"]] for f in rep["files"]}
    hf_repo = None
    if files[0].url:
        m = re.match(r"https?://[^/]+/(.+?)/resolve/", files[0].url)
        hf_repo = m.group(1) if m else None
    entry = {
        "_status": "DRAFT: made from GGUF headers by tools/qwen4_intake.py; not wired into setup.py; every value to be checked by a person",
        "family_key": "FILL-IN (a short id, e.g. 'qwen4')", "title": name, "by": gen.get("quantized_by") or "FILL-IN",
        "about": "FILL-IN (one line; numbers only)", "arch": arch,
        "hf": ("hf(%r)" % hf_repo) if hf_repo else "FILL-IN (the repository at a pinned revision)",
        "file": pattern, "shards": len(files), "tag": "FILL-IN-", "name": slug(name),
        "license": gen.get("license.link") or gen.get("license") or "FILL-IN", "sha256": sha, "size_bytes": size,
        "vision": any(p["supported"] for p in rep["projectors"]) if rep["projectors"] else False,
        "mmproj": rep["projectors"][0]["file"] if rep["projectors"] else None,
        "mtp": (2 if rep["mtp"]["in_file"] else 0),
    }
    cfg: dict = {"_status": "DRAFT: the settings the verdict implies, not a config setup.py would write", "arch": arch, "case": rep["verdict"]["case"],
                 "engine_path": rep["verdict"]["path"], "backends": rep["verdict"]["backends"], "model_name": slug(name),
                 "gguf_first_shard": files[0].name}
    if fam and g:
        n_attn = g["n_layers"] // g["qsa_interval"]
        cfg["geometry"] = {k: g[k] for k in ("n_embd", "n_layers", "n_expert", "n_expert_used", "n_ff", "n_head", "n_head_kv", "head_dim", "qsa_interval")}
        if fam == "qwen35moe":
            # setup.py OWN_PATH_FAMILIES['qwen36']: "kv_bytes_per_token": 11 * 2 * 256 * 2 * 4 and "state_gb" (the GDN recurrent state)
            n_gdn = g["n_layers"] - n_attn
            entry["kv_bytes_per_token"] = (n_attn + g["n_mtp_layers"]) * 2 * g["head_dim"] * g["n_head_kv"] * 4
            entry["state_gb"] = round((n_gdn * g["ssm_v_heads"] * g["ssm_state_size"] ** 2 * 4 +
                                       n_gdn * (g["ssm_d_conv"] - 1) * g["ssm_conv_channels"] * 4) / 1e9, 2)
            cfg["kv_bytes_per_token"], cfg["state_gb"] = entry["kv_bytes_per_token"], entry["state_gb"]
        entry["models"] = {"FILL-IN-QUANT": {"about": "FILL-IN", "ram_gb": math.ceil(size / 1e9 * 1.4),
                                              "_ram_gb": "estimate: file size x 1.4, rounded up; measure before use"}}
        cfg["context_length_trained"] = rep["geometry"]["keys"].get("context_length")
    return entry, cfg


# ====================================================================================================== driving it
def collect(args) -> tuple[list[FileInfo], list[str]]:
    infos: list[FileInfo] = []
    notes: list[str] = []
    if args.hf:
        rev = args.rev or "main"
        entries = hf_list(args.hf, rev, args.dir or "")
        if not entries:
            raise SystemExit("no .gguf files in %s@%s/%s" % (args.hf, rev, args.dir or ""))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
            infos = list(ex.map(lambda e: read_hf_entry(args.hf, rev, e), entries))
        return infos, notes
    paths: list[pathlib.Path] = []
    for p in args.paths:
        pp = pathlib.Path(p)
        if pp.is_dir():
            paths += sorted(x for x in pp.rglob("*.gguf") if len(x.relative_to(pp).parts) <= 3)
        elif pp.is_file():
            paths.append(pp)
            m = SHARD_RE.match(pp.name)
            if m:                                   # any shard of a set stands for the set
                for q in sorted(pp.parent.glob("%s-*-of-%s.gguf" % (m.group("stem"), m.group("n")))):
                    if q not in paths:
                        paths.append(q)
        else:
            raise SystemExit("not a file or folder: %s" % p)
    if not paths:
        raise SystemExit("no .gguf files found")
    seen, uniq = set(), []
    for p in paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        infos = list(ex.map(read_local, uniq))
    if args.sha256:
        for f in infos:
            f.sha256, f.sha256_source = file_sha256(f.path), "computed"
    return infos, notes


def run(infos: list[FileInfo], build_tokenizer: bool = True) -> dict:
    models, projectors = [], []
    for f in infos:
        (projectors if _arch_of(f.md) == "clip" else models).append(f)
    names = [f.name for f in infos]
    reps = []
    for s in group_sets(models):
        if is_drafter(s):
            reps.append(drafter_report(s))
            continue
        pr = [projector_report(p) for p in projectors]
        reps.append(analyse_set(s, siblings=names, build_tokenizer=build_tokenizer, projectors=pr))
    if not models:
        reps = [{"architecture": "clip", "projectors": [projector_report(p) for p in projectors],
                 "verdict": {"case": "-", "reading": "picture readers only: no model in this input", "blockers": [], "warnings": [], "notes": []}}]
    return {"tool": "tools/qwen4_intake.py", "tool_version": TOOL_VERSION, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sets": reps, "picture_readers": [projector_report(p) for p in projectors]}


def render(report: dict) -> str:
    out = []
    for s in report["sets"]:
        v = s["verdict"]
        out.append("=" * 100)
        if "files" in s:
            out.append("%s   %d file(s)   %.1f GB   arch %s%s" % (s["files"][0]["name"], len(s["files"]), s["size_bytes"] / 1e9, s["architecture"],
                                                              "   (drafter)" if s.get("drafter") else ""))
        out.append("CASE %s: %s%s" % (v["case"], v["reading"], ("   path: " + v["path"]) if v.get("path") else ""))
        if v.get("backends"):
            out.append("  runs on: " + ", ".join(v["backends"]))
        for b in v.get("blockers", []):
            out.append("  BLOCKER  " + b)
        for d in v.get("shape_fields_that_differ", []):
            out.append("  SHAPE    %s is %s, kernels built for %s" % (d["key"], d["got"], d.get("kernels_built_for", d.get("limit"))))
        for w in v.get("warnings", []):
            out.append("  WARNING  " + w)
        for n in v.get("notes", []):
            out.append("  note     " + n)
        if "quant" in s:
            q = s["quant"]
            out.append("  quant types: " + ", ".join("%s x%d" % kv for kv in q["types"].items()))
            for c in q["expert_combinations"]:
                out.append("    experts %s/%s/%s: %d layers" % (c["gate"], c["up"], c["down"], c["layers"]))
            out.append("  backends: " + "; ".join("%s %s" % (b, "ok" if d["all_types_supported"] else "NOT ALL") for b, d in q["backends"].items()))
            tk, ct = s["tokenizer"], s["chat_template"]
            out.append("  tokenizer: %s / pre %s, %s tokens, %s merges, same as Flash-Next: %s, builds: %s"
                       % (tk["model"], tk["pre"], tk["n_tokens"], tk["n_merges"], tk.get("same_vocabulary_as_flash_next"),
                          (tk.get("builds") or {}).get("ok", "not tried")))
            out.append("  chat template: %s, tool calls: %s, thinking tags: %s"
                       % (ct.get("known_as") or ct.get("sha256", "none")[:12], ct.get("tool_call_format"), ct.get("has_think_tags")))
            out.append("  MTP: in file %s, drafter files %s; picture readers: %s"
                       % (s["mtp"]["in_file"], s["mtp"]["separate_drafter_files"] or "none",
                          [(p["file"], p["projector_type"], "ours" if p["supported"] else "NOT ours") for p in s["projectors"]] or "none"))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="folders or .gguf files")
    ap.add_argument("--hf", metavar="REPO", help="a Hugging Face repository (headers by HTTP range reads)")
    ap.add_argument("--rev", help="the revision for --hf (default main)")
    ap.add_argument("--dir", help="a folder inside the repository")
    ap.add_argument("--out", default="qwen4_intake.json", help="the JSON report")
    ap.add_argument("--sha256", action="store_true", help="hash local files (slow)")
    ap.add_argument("--no-tokenizer-build", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.hf and not a.paths:
        ap.error("give a folder, .gguf files or --hf REPO")
    t0 = time.time()
    infos, _ = collect(a)
    t_read = time.time() - t0
    report = run(infos, build_tokenizer=not a.no_tokenizer_build)
    report["seconds"] = {"headers": round(t_read, 1), "total": round(time.time() - t0, 1)}
    pathlib.Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if not a.quiet:
        print(render(report))
        print("\nJSON: %s   (%.1f s, headers %.1f s)" % (a.out, report["seconds"]["total"], t_read))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
