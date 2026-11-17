#!/usr/bin/env python3
"""Synthetic qwen4exp GGUF files with random weights, at any shape, in the per-tensor type mix of an Unsloth UD file.

Why: the engine is compiled for one model shape (include/strata/core/geometry.hpp).  To check that a build for another shape
loads and runs, there has to be a file of that shape.  The writer in gguf_writer.py holds every tensor in memory and knows
only F32, F16 and Q2_0; this one streams, and writes the types the UD files use (Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, IQ4_NL,
IQ4_XS, IQ3_S, BF16, F32).  The weights are random bytes with the scale fields of every block set to a fixed small number, so
the dequantized values stay in a few hundredths and the activations stay finite; nothing here quantizes anything.

    python tools/q4geo_synth.py --shape q4_lite --layers 8 --vocab 8192 --recipe ud-q4k --out /data/synth/lite.gguf
    python tools/q4geo_synth.py --shape q4_lite --layers 8 --header out.hpp          # the geometry header for that file
    python tools/q4geo_synth.py --from-gguf some.gguf --header out.hpp              # day one: a geometry header from a real file
    python tools/q4geo_synth.py --list

The layout (tensor names and shapes) is core/family_layout.hpp's for qwen4exp.  The PLE / n-gram tables of Flash-Next are NOT
written (their hash constants are compiled in and the table is 28 GB): run the engine with --no-ple on these files.

The geometry header written by --header is what -DSTRATA_GEOMETRY_FILE=<path> takes.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter, GGUF_MAGIC, DEFAULT_ALIGNMENT  # noqa: E402

# ggml type id, elements per block, bytes per block, fields to set: (byte offset, fp16 value)
QTYPES = {
    "F32": (0, 1, 4, []),
    "F16": (1, 1, 2, []),
    "BF16": (30, 1, 2, []),
    "Q5_0": (6, 32, 22, [(0, 0.0024)]),
    "Q5_1": (7, 32, 24, [(0, 0.0012), (2, -0.018)]),
    "Q8_0": (8, 32, 34, [(0, 0.0003)]),
    "Q4_K": (12, 256, 144, [(0, 3e-5), (2, 3e-5)]),
    "Q5_K": (13, 256, 176, [(0, 1.5e-5), (2, 1.5e-5)]),
    "Q6_K": (14, 256, 210, [(208, 5e-6)]),
    "IQ4_NL": (20, 32, 18, [(0, 0.0003)]),
    "IQ3_S": (21, 256, 110, [(0, 1e-4)]),
    "IQ4_XS": (23, 256, 136, [(0, 1e-5)]),
}

ROUTER_SCALE = 1.0   # --router-scale: the router's weights times this. Random weights give router logits packed so closely that two
                     # implementations of one model (llama.cpp CPU and GPU among them) pick different experts at a few positions, and one
                     # expert flip moves a position's KL by 100x; a larger scale spreads the logits like a trained router's
EXACT_PARTS: set = set()   # --exact-parts router,f32,q8 (--exact-bf16 = all three): which tensors are written BF16-exact
TOKENIZER_FROM = None   # --tokenizer-from FILE: the tokenizer.* keys (vocabulary, merges, special tokens, chat template) and the PLE token ids of a real GGUF, so a Terminal/server can encode text for the synthetic model
EXACT_BF16 = False   # --exact-bf16: every F32 tensor holds BF16-representable values and the Q8_0 scale is a power of two, so the pack's
                     # --compat-bf16 conversion (router, shared-expert gate, HC down/up, ...) is lossless and llama.cpp sees the engine's weights

# Flash-Next's values for every field a shape does not name
FLASH = dict(n_embd=2560, n_layers=48, n_head=24, n_head_kv=2, head_dim=256, n_rot=64, ssm_state=128, ssm_k_heads=16,
             ssm_v_heads=48, ssm_d_conv=4, idx_q_heads=4, idx_key_dim=128, idx_top_k=2048, idx_block=4, hc=4, hc_lr=320,
             n_expert=512, n_expert_used=10, n_ff=640, qsa_interval=4)
SHAPES = {
    "flash_next": {},
    "q4_deep": dict(n_layers=64),
    "q4_lite": dict(n_embd=2048, n_layers=40, n_head=16, ssm_v_heads=32, hc_lr=256, n_expert=256, n_expert_used=8, n_ff=512),
    "q4_mid": dict(n_embd=3072, n_layers=48, n_head=32, ssm_v_heads=64, hc_lr=384, n_expert=256, n_expert_used=8, n_ff=1024),
    "q4_big": dict(n_embd=4096, n_layers=60, n_head=32, ssm_v_heads=64, hc_lr=512, n_expert=512, n_expert_used=10, n_ff=1024),
}


def shape(name: str, layers: int | None = None, **over) -> dict:
    s = dict(FLASH)
    s.update(SHAPES[name])
    if layers:
        s["n_layers"] = layers
    s.update(over)
    s["conv_ch"] = 2 * s["ssm_state"] * s["ssm_k_heads"] + s["ssm_state"] * s["ssm_v_heads"]
    s["value_dim"] = s["ssm_state"] * s["ssm_v_heads"]
    return s


# ------------------------------------------------------------------ layout (core/family_layout.hpp)
def layer_tensors(s: dict, layer: int):
    """(name, shape, role) of one layer."""
    N, NE, FF = s["n_embd"], s["n_expert"], s["n_ff"]
    H = s["hc"] * N
    t = [("ffn_gate_inp.weight", [N, NE], "router"), ("ffn_gate_inp_shexp.weight", [N], "small"),
         ("ffn_gate_shexp.weight", [N, FF], "shexp"), ("ffn_up_shexp.weight", [N, FF], "shexp"),
         ("ffn_down_shexp.weight", [FF, N], "shexp"),
         ("ffn_gate_exps.weight", [N, FF, NE], "gate"), ("ffn_up_exps.weight", [N, FF, NE], "up"),
         ("ffn_down_exps.weight", [FF, N, NE], "down")]
    for blk in ("attn", "ffn"):
        t += [(f"hc_{blk}_down.weight", [H, s["hc_lr"]], "hc"), (f"hc_{blk}_up.weight", [s["hc_lr"], H], "hc"),
              (f"hc_{blk}_inject.weight", [H, s["hc"]], "inject"), (f"hc_{blk}_norm.weight", [H], "norm")]
    if (layer % s["qsa_interval"]) != s["qsa_interval"] - 1:
        t += [("attn_qkv.weight", [N, s["conv_ch"]], "dense"), ("attn_gate.weight", [N, s["value_dim"]], "dense"),
              ("ssm_out.weight", [s["value_dim"], N], "dense"), ("ssm_conv1d.weight", [s["ssm_d_conv"], s["conv_ch"]], "conv"),
              ("ssm_alpha.weight", [N, s["ssm_v_heads"]], "small_mat"), ("ssm_beta.weight", [N, s["ssm_v_heads"]], "small_mat"),
              ("ssm_a", [s["ssm_v_heads"]], "ssm_a"), ("ssm_dt.bias", [s["ssm_v_heads"]], "bias"),
              ("ssm_norm.weight", [s["ssm_state"]], "norm")]
    else:
        hd, nh, nkv = s["head_dim"], s["n_head"], s["n_head_kv"]
        t += [("attn_q.weight", [N, 2 * nh * hd], "dense"), ("attn_k.weight", [N, nkv * hd], "dense"),
              ("attn_v.weight", [N, nkv * hd], "dense"), ("attn_output.weight", [nh * hd, N], "dense"),
              ("attn_q_norm.weight", [hd], "norm"), ("attn_k_norm.weight", [hd], "norm"),
              ("indexer.q_proj.weight", [N, s["idx_q_heads"] * s["idx_key_dim"]], "indexer"),
              ("indexer.k_proj.weight", [N, s["idx_key_dim"]], "indexer"),
              ("indexer.q_norm.weight", [s["idx_key_dim"]], "norm"), ("indexer.k_norm.weight", [s["idx_key_dim"]], "norm")]
    return t


PLE_ROWS = 320001536
PLE_VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107,
             20000147, 20000153, 20000159, 20000161, 20000171]
PLE_MULT = [23703573157769, 20109073645365, 8052911324071]


def all_tensors(s: dict, vocab: int, ple: bool = True):
    N = s["n_embd"]
    H = s["hc"] * N
    out = [("token_embd.weight", [N, vocab], "embd"), ("output.weight", [N, vocab], "output"),
           ("output_hc_down.weight", [H, s["hc_lr"]], "hc"), ("output_hc_up.weight", [s["hc_lr"], H], "hc"),
           ("output_hc_norm.weight", [H], "norm")]
    for l in range(s["n_layers"]):
        out += [(f"blk.{l}.{n}", sh, r) for n, sh, r in layer_tensors(s, l)]
    if ple:
        # the engine's native path always reads the PLE key; its hash constants are compiled in (kernels/ngram.cpp), so the table has
        # Flash-Next's 320001536 rows of n_embd/16; it is written SPARSE (zeros: IQ4_NL blocks with a zero scale are zero vectors)
        out += [("per_layer_token_embd.weight", [N // 16, PLE_ROWS], "ple_table"),
                ("blk.1.ple_conv1d.weight", [4, H], "conv"), ("blk.1.ple_key.weight", [N, H], "dense"),
                ("blk.1.ple_norm_conv.weight", [H], "norm"), ("blk.1.ple_norm_key.weight", [H], "norm"),
                ("blk.1.ple_norm_query.weight", [H], "norm"), ("blk.1.ple_value.weight", [N, N], "dense")]
    return out


# ------------------------------------------------------------------ recipes (the type per role, as Unsloth's UD files)
def recipe_type(recipe: str, role: str, layer: int, s: dict) -> str:
    ff256 = s["n_ff"] % 256 == 0
    if role in ("router", "small", "norm", "inject", "small_mat", "conv", "ssm_a", "bias"):
        return "F32"
    if role == "indexer":
        return "BF16"
    if role in ("dense", "shexp", "hc"):
        return "Q8_0"
    if role == "embd":
        return "Q8_0"
    if role == "ple_table":
        return "IQ4_NL"
    if role == "output":
        return "Q6_K" if recipe == "ud-iq4xs" else "Q8_0"
    first = layer == 0
    if recipe == "ud-q4k":           # UD-Q4_K_XL: Q4_K gate/up; down Q5_K, or Q5_1 where llama.cpp falls back (n_ff % 256); Q8_0 in a few
        if role in ("gate", "up"):
            return "Q5_K" if first else "Q4_K"
        return "Q8_0" if first else ("Q5_K" if ff256 else "Q5_1")
    if recipe == "ud-q4k-q6":        # as ud-q4k with Q6_K down on every 8th layer (Unsloth's Qwen3.6 mix: 38 x Q5_K, 3 x Q6_K)
        if role in ("gate", "up"):
            return "Q5_K" if first else "Q4_K"
        if first:
            return "Q8_0"
        return ("Q6_K" if layer % 8 == 1 else "Q5_K") if ff256 else "Q5_1"
    if recipe == "ud-iq4xs":         # UD-IQ4_XS: IQ3_S gate/up (IQ4_XS in one layer), IQ4_XS down or IQ4_NL by the fallback
        if role in ("gate", "up"):
            return "IQ4_XS" if first else "IQ3_S"
        return "Q8_0" if first else ("IQ4_XS" if ff256 else "IQ4_NL")
    if recipe == "ud-q5_0down":      # the v31 mix: Q5_0 down layers (what llama.cpp makes of Q4_K down at a non-256 width)
        if role in ("gate", "up"):
            return "Q4_K"
        return "Q5_0"
    if recipe == "fused-kq":         # Q4_K gate/up (Q5_K in layer 0) with Q5_1 / Q8_0 down at ANY expert width: the pairs the AMD fused / F16 prompt kernels take (Q4-HIPKQ)
        if role in ("gate", "up"):
            return "Q5_K" if first else "Q4_K"
        return "Q8_0" if layer % 2 == 0 else "Q5_1"
    if recipe == "mix-kq":           # one file with every K-quant expert path of the AMD prompt kernels: Q5_K/Q8_0 (fused), Q4_K/Q5_K and Q4_K/Q6_K (MMQ), Q4_K/Q5_1 (fused F16), by layer % 4 (Q4-HIPKQ)
        if role in ("gate", "up"):
            return "Q5_K" if layer % 4 == 0 else "Q4_K"
        return ("Q8_0", "Q5_K", "Q5_1", "Q6_K")[layer % 4]
    if recipe == "fused-iq":         # IQ3_S gate/up (IQ4_XS in layer 0) with IQ4_NL down at any expert width: the native IQ pack's fused prompt kernels (Q4-HIPKQ)
        if role in ("gate", "up"):
            return "IQ4_XS" if first else "IQ3_S"
        return "IQ4_NL"
    raise SystemExit(f"unknown recipe {recipe}")


RECIPES = ["ud-q4k", "ud-q4k-q6", "ud-iq4xs", "ud-q5_0down", "fused-kq", "fused-iq", "mix-kq"]


def row_ok(ty: str, ne0: int) -> bool:
    return ne0 % QTYPES[ty][1] == 0


# ------------------------------------------------------------------ data
def bf16_round(a: np.ndarray) -> np.ndarray:
    """float32 rounded to the nearest BF16 (ties to even), returned as float32."""
    u = a.astype("<f4").view("<u4").astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype("<u4").view("<f4")


def small_f32(role: str, shape: list[int], rng: np.random.Generator) -> np.ndarray:
    a = _small_f32(role, shape, rng)
    if role == "router":
        a = (a * np.float32(ROUTER_SCALE)).astype("<f4")
    if role == "router":
        return bf16_round(a) if ("router" in EXACT_PARTS) else a
    return bf16_round(a) if ("f32" in EXACT_PARTS) else a


def _small_f32(role: str, shape: list[int], rng: np.random.Generator) -> np.ndarray:
    n = int(np.prod(shape))
    if role == "norm":
        return (1.0 + 0.05 * rng.standard_normal(n)).astype("<f4")
    if role == "ssm_a":
        return (-rng.uniform(0.1, 1.5, n)).astype("<f4")
    if role == "bias":
        return (0.1 * rng.standard_normal(n)).astype("<f4")
    if role == "conv":
        return (0.1 * rng.standard_normal(n)).astype("<f4")
    if role == "inject":
        return (0.25 + 0.05 * rng.standard_normal(n)).astype("<f4")
    return (0.02 * rng.standard_normal(n)).astype("<f4")


def tensor_bytes(ty: str, shape: list[int]) -> int:
    _, blk, tsz, _ = QTYPES[ty]
    n = int(np.prod(shape))
    assert n % blk == 0, (ty, shape)
    return n // blk * tsz


def emit_data(fh, ty: str, role: str, shape: list[int], rng: np.random.Generator) -> int:
    tid, blk, tsz, fields = QTYPES[ty]
    total = tensor_bytes(ty, shape)
    if role == "ple_table":              # a hole: zeros, no disk
        fh.seek(total - 1, 1)
        fh.write(bytes(1))
        return total
    if ty == "F32":
        fh.write(small_f32(role, shape, rng).tobytes())
        return total
    if ty == "BF16":
        a = (0.02 * rng.standard_normal(int(np.prod(shape)))).astype("<f4")
        fh.write((a.view("<u4") >> 16).astype("<u2").tobytes())
        return total
    nblk = total // tsz
    step = max(1, (64 << 20) // tsz)
    done = 0
    while done < nblk:
        m = min(step, nblk - done)
        b = rng.integers(0, 256, size=(m, tsz), dtype=np.uint8)
        for off, val in fields:
            if "q8" in EXACT_PARTS and ty == "Q8_0":
                val = 2.0 ** -12
            b[:, off:off + 2] = np.frombuffer(np.float16(val).tobytes(), dtype=np.uint8)
        fh.write(b.tobytes())
        done += m
    return total


# ------------------------------------------------------------------ metadata
def metadata(s: dict, vocab: int, name: str) -> GGUFWriter:
    w = GGUFWriter()
    p = "qwen4exp."
    w.add("general.architecture", "qwen4exp")
    w.add("general.type", "model")
    w.add("general.name", name)
    w.add("general.description", "synthetic random weights (tools/q4geo_synth.py), not a model")
    w.add(p + "block_count", s["n_layers"], "u32")
    w.add(p + "context_length", 262144, "u32")
    w.add(p + "embedding_length", s["n_embd"], "u32")
    w.add(p + "attention.head_count", s["n_head"], "u32")
    w.add(p + "attention.head_count_kv", s["n_head_kv"], "u32")
    w.add(p + "rope.dimension_sections", [11, 11, 10, 0], "array:i32")
    w.add(p + "rope.freq_base", 10000000.0, "f32")
    w.add(p + "attention.layer_norm_rms_epsilon", 1e-6, "f32")
    w.add(p + "expert_count", s["n_expert"], "u32")
    w.add(p + "expert_used_count", s["n_expert_used"], "u32")
    w.add(p + "attention.key_length", s["head_dim"], "u32")
    w.add(p + "attention.value_length", s["head_dim"], "u32")
    w.add(p + "expert_feed_forward_length", s["n_ff"], "u32")
    w.add(p + "expert_shared_feed_forward_length", s["n_ff"], "u32")
    w.add(p + "ssm.conv_kernel", s["ssm_d_conv"], "u32")
    w.add(p + "ssm.state_size", s["ssm_state"], "u32")
    w.add(p + "ssm.group_count", s["ssm_k_heads"], "u32")
    w.add(p + "ssm.time_step_rank", s["ssm_v_heads"], "u32")
    w.add(p + "ssm.inner_size", s["value_dim"], "u32")
    w.add(p + "full_attention_interval", s["qsa_interval"], "u32")
    w.add(p + "rope.dimension_count", s["n_rot"], "u32")
    w.add(p + "hyper_connection.count", s["hc"], "u32")
    w.add(p + "hyper_connection.low_rank", s["hc_lr"], "u32")
    w.add(p + "attention.indexer.head_count", s["idx_q_heads"], "u32")
    w.add(p + "attention.indexer.key_length", s["idx_key_dim"], "u32")
    w.add(p + "attention.indexer.top_k", s["idx_top_k"], "u32")
    w.add(p + "attention.compress_ratios",
          [s["idx_block"] if l % s["qsa_interval"] == s["qsa_interval"] - 1 else 0 for l in range(s["n_layers"])], "array:i32")
    offs = [sum(PLE_VOCAB[:i]) for i in range(16)]
    w.add(p + "ple.layers", [1], "array:i32")
    w.add(p + "ple.ngram_size", 3, "u32")
    w.add(p + "ple.heads_per_ngram", 8, "u32")
    w.add(p + "ple.conv_kernel", 4, "u32")
    w.add(p + "ple.eos_token_id", vocab - 2, "u32")
    w.add(p + "embedding_length_per_layer_input", s["n_embd"] // 16, "u32")
    w.add(p + "ple.layer_multipliers", PLE_MULT, "array:u64")
    w.add(p + "ple.head_offsets", offs, "array:u64")
    w.add(p + "ple.head_vocab_sizes", PLE_VOCAB, "array:u64")
    w.add(p + "ple.image_token_id", vocab - 3, "u32")
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "qwen35")
    w.add("tokenizer.ggml.tokens", ["a", "b", "ab"] + [f"<t{i}>" for i in range(3, vocab)], "array:string")
    w.add("tokenizer.ggml.token_type", [1] * vocab, "array:i32")
    w.add("tokenizer.ggml.merges", ["a b"], "array:string")
    w.add("tokenizer.ggml.eos_token_id", vocab - 2, "u32")
    w.add("tokenizer.ggml.padding_token_id", vocab - 1, "u32")
    w.add("tokenizer.ggml.bos_token_id", vocab - 1, "u32")
    w.add("general.quantization_version", 2, "u32")
    if TOKENIZER_FROM is not None:
        for k, v in TOKENIZER_FROM.metadata.items():
            if k.startswith("tokenizer."):
                if k.endswith("_token_id"):
                    w.add(k, int(v), "u32")
                elif k == "tokenizer.ggml.tokens" or k == "tokenizer.ggml.merges":
                    w.add(k, list(v), "array:string")
                elif k == "tokenizer.ggml.token_type":
                    w.add(k, [int(x) for x in v], "array:i32")
                elif isinstance(v, bool):
                    w.add(k, v, "bool")
                elif isinstance(v, str):
                    w.add(k, v, "string")
            elif k.endswith(".ple.eos_token_id") or k.endswith(".ple.image_token_id"):
                w.add(p + "ple." + k.rsplit(".ple.", 1)[1], int(v), "u32")
    return w


def write_gguf(path: pathlib.Path, s: dict, vocab: int, recipe: str, seed: int, name: str) -> dict:
    rng = np.random.default_rng(seed)
    w = metadata(s, vocab, name)
    tl = all_tensors(s, vocab)
    plan = []
    for n, sh, role in tl:
        layer = int(n.split(".")[1]) if n.startswith("blk.") else -1
        ty = recipe_type(recipe, role, layer, s)
        if not row_ok(ty, sh[0]):
            raise SystemExit(f"{n} {sh}: row length {sh[0]} is not a multiple of {QTYPES[ty][1]} ({ty})")
        plan.append((n, sh, role, ty))
    # the expert pack (tools/iq_pack.py -> native_experts.txt) wants every layer's gate, up, down tensors back to back and the layers one after
    # the other, so the experts come first in the file (a stable sort keeps the layer order)
    plan.sort(key=lambda x: 0 if "_exps.weight" in x[0] else 1)
    body = bytearray()
    body += struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3) + struct.pack("<Q", len(plan)) + struct.pack("<Q", len(w.metadata))
    for k, (t, v) in w.metadata.items():
        body += w._kv_bytes(k, t, v)
    off = 0
    for n, sh, role, ty in plan:
        body += struct.pack("<Q", len(n)) + n.encode() + struct.pack("<I", len(sh)) + struct.pack(f"<{len(sh)}Q", *sh)
        body += struct.pack("<I", QTYPES[ty][0]) + struct.pack("<Q", off)
        off += tensor_bytes(ty, sh)
        off = (off + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT * DEFAULT_ALIGNMENT
    body += b"\0" * ((-len(body)) % DEFAULT_ALIGNMENT)
    types: dict[str, int] = {}
    with open(path, "wb") as fh:
        fh.write(body)
        for n, sh, role, ty in plan:
            nb = emit_data(fh, ty, role, sh, rng)
            fh.write(b"\0" * ((-nb) % DEFAULT_ALIGNMENT))
            types[ty] = types.get(ty, 0) + 1
    return {"tensors": len(plan), "bytes": path.stat().st_size, "types": types}


# ------------------------------------------------------------------ geometry header
def header_text(s: dict, label: str, source: str) -> str:
    f = FLASH
    lines = [f"// {label}: written by tools/q4geo_synth.py from {source}.", "// Take it with -DSTRATA_GEOMETRY_FILE=<this file>.",
             "#pragma once", "", "namespace strata::core {", "",
             f'inline constexpr const char* kGeometryName = "{label}";',
             "inline constexpr ModelGeometry kCompiledGeometry = [] {", "    ModelGeometry g;"]
    m = [("n_embd", "n_embd"), ("n_layers", "n_layers"), ("qsa_interval", "qsa_interval"), ("n_head", "n_head"),
         ("n_head_kv", "n_head_kv"), ("head_dim", "head_dim"), ("n_rot", "n_rot"), ("ssm_state", "ssm_state_size"),
         ("ssm_k_heads", "ssm_k_heads"), ("ssm_v_heads", "ssm_v_heads"), ("ssm_d_conv", "ssm_d_conv"),
         ("idx_q_heads", "idx_q_heads"), ("idx_key_dim", "idx_key_dim"), ("idx_block", "idx_block"), ("idx_top_k", "idx_top_k"),
         ("hc", "hc"), ("hc_lr", "hc_lr"), ("n_expert", "n_expert"), ("n_expert_used", "n_expert_used"), ("n_ff", "n_ff")]
    for k, c in m:
        lines.append(f"    g.{c} = {s[k]};")
    lines += ["    g.n_ff_shexp = g.n_ff;",
              "    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;",
              "    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;", "    return g;", "}();", "", "}  // namespace strata::core", ""]
    return "\n".join(lines)


def shape_from_gguf(path: str) -> dict:
    from gguf_reader import GGUFFile
    g = GGUFFile(pathlib.Path(path))
    md = g.metadata
    p = "qwen4exp."
    gi = lambda k, d=None: int(md[p + k]) if (p + k) in md else d
    s = dict(FLASH)
    s.update(n_embd=gi("embedding_length"), n_layers=gi("block_count"), n_head=gi("attention.head_count"),
             n_head_kv=gi("attention.head_count_kv"), head_dim=gi("attention.key_length", 256), n_rot=gi("rope.dimension_count", 64),
             ssm_state=gi("ssm.state_size", 128), ssm_k_heads=gi("ssm.group_count", 16), ssm_v_heads=gi("ssm.time_step_rank", 48),
             ssm_d_conv=gi("ssm.conv_kernel", 4), idx_q_heads=gi("attention.indexer.head_count", 4),
             idx_key_dim=gi("attention.indexer.key_length", 128), idx_top_k=gi("attention.indexer.top_k", 2048),
             hc=gi("hyper_connection.count", 4), hc_lr=gi("hyper_connection.low_rank", 320), n_expert=gi("expert_count"),
             n_expert_used=gi("expert_used_count"), n_ff=gi("expert_feed_forward_length"),
             qsa_interval=gi("full_attention_interval", 4))
    cr = md.get(p + "attention.compress_ratios")
    if cr:
        nz = [int(x) for x in cr if int(x)]
        if nz:
            s["idx_block"] = nz[0]
    s["conv_ch"] = 2 * s["ssm_state"] * s["ssm_k_heads"] + s["ssm_state"] * s["ssm_v_heads"]
    s["value_dim"] = s["ssm_state"] * s["ssm_v_heads"]
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="flash_next")
    ap.add_argument("--layers", type=int, help="block_count of the file (a build for it needs a header with the same count)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--recipe", choices=RECIPES, default="ud-q4k")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="the GGUF to write")
    ap.add_argument("--header", help="write the geometry header for the shape (or --from-gguf) here")
    ap.add_argument("--label", help="kGeometryName in the header (default: the shape, plus _<layers>l)")
    ap.add_argument("--from-gguf", help="read the shape from this file's metadata instead of --shape")
    ap.add_argument("--exact-bf16", action="store_true", help="F32 tensors rounded to BF16 values and Q8_0 scale 2^-12: the pack's BF16 conversion is then lossless")
    ap.add_argument("--router-scale", type=float, default=1.0, help="multiply the router weights (default 1: the files of Q4-GEO2)")
    ap.add_argument("--exact-parts", default="", help="router, f32 (the other F32 tensors) and/or q8 (Q8_0 scale 2^-12): the BF16-exact parts of --exact-bf16")
    ap.add_argument("--tokenizer-from", help="a real qwen4exp GGUF (any shard 1): its tokenizer.* keys and PLE token ids go into the file, and --vocab becomes its vocabulary size")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--small-ple", action="store_true",
                    help="PLE head sizes of 16 small primes (a table of ~1.6M rows, ~50 MB) instead of Flash-Next's 320001536 rows (28 GB): the hash moduli are read from the file, so the engine runs it; "
                         "for hosts where the sparse 28 GB table cannot be written (Windows)")
    ap.add_argument("--set", default="", help="comma list of field=value overrides of the shape (a bisection of one field), e.g. n_ff=1024,hc_lr=384")
    a = ap.parse_args()
    global EXACT_BF16
    EXACT_BF16 = a.exact_bf16
    global EXACT_PARTS
    EXACT_PARTS = {"router", "f32", "q8"} if a.exact_bf16 else set(x for x in a.exact_parts.split(",") if x)
    global ROUTER_SCALE
    ROUTER_SCALE = a.router_scale
    if a.tokenizer_from:
        global TOKENIZER_FROM
        from gguf_reader import GGUFFile
        TOKENIZER_FROM = GGUFFile(pathlib.Path(a.tokenizer_from))
        a.vocab = len(TOKENIZER_FROM.metadata["tokenizer.ggml.tokens"])
    if a.list:
        for n in sorted(SHAPES):
            s = shape(n)
            print(n, {k: s[k] for k in ("n_embd", "n_layers", "n_head", "n_head_kv", "ssm_v_heads", "hc_lr", "n_expert", "n_expert_used", "n_ff")})
        return 0
    if a.from_gguf:
        s = shape_from_gguf(a.from_gguf)
        label = a.label or pathlib.Path(a.from_gguf).stem.replace("-", "_").lower()
        src = a.from_gguf
    else:
        s = shape(a.shape, a.layers, **{k: int(v) for k, v in (kv.split("=") for kv in a.set.split(",") if kv)})
        label = a.label or (a.shape + (f"_{a.layers}l" if a.layers else ""))
        src = f"shape {a.shape}" + (f", {a.layers} layers" if a.layers else "")
    if a.small_ple:
        global PLE_ROWS
        primes, n = [], 100003
        while len(primes) < 16:
            if all(n % d for d in range(2, int(n ** 0.5) + 1)):
                primes.append(n)
            n += 2
        PLE_VOCAB[:] = primes
        PLE_ROWS = sum(primes)
    if a.header:
        pathlib.Path(a.header).write_text(header_text(s, label, src), encoding="utf8", newline="\n")
        print(f"wrote {a.header} ({label})")
    if a.out:
        if a.from_gguf:
            raise SystemExit("--from-gguf is for --header")
        info = write_gguf(pathlib.Path(a.out), s, a.vocab, a.recipe, a.seed, f"synthetic {label} ({a.recipe})")
        print(f"wrote {a.out}: {info['tensors']} tensors, {info['bytes'] / 1e9:.2f} GB, types {info['types']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
