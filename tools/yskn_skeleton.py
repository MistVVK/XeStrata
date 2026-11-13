# SPDX-FileCopyrightText: 2026 the YASAKANI authors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
"""tools/yskn_skeleton.py - the GGUF skeleton of a YASAKANI .yskn model (Qwen3.8-27B YASAKANI quality / speed).

The engine runs a .yskn model as the plain engine's qwen35 (the dense Qwen3.5 block): everything but the coded
matrices comes from a small GGUF beside the .yskn, which this writes from the model's Hugging Face files -

* the metadata: llama.cpp's qwen35 keys from config.json (its text_config), the tokenizer from tokenizer.json
  (tokens, merges, token types and the pre-tokenizer `qwen35`, as llama.cpp's converter writes them), the chat
  template from chat_template.jinja, and `xestrata.yskn.file`, the .yskn's file name;
* the small tensors from the .yskn's vectors (BF16), converted as llama.cpp's converter converts Hugging Face's: the
  norms plus one (not the GDN's output norm), A = -exp(A_log), dt_bias as ssm_dt.bias, the conv squeezed, and the GDN's
  V heads from grouped to tiled order (in_proj_a / b's rows, A, dt, the conv's V channels).

The coded matrices (the projections, the head, the embedding) stay in the .yskn: the engine reads them there
(PlainModel::open_yskn).  The serving side reads the tokenizer and the template from the skeleton as from any GGUF.

    python tools/yskn_skeleton.py MODEL_DIR [--yskn NAME.yskn] [--out NAME.gguf]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402

# llama.cpp's gguf token types
NORMAL, USER_DEFINED, CONTROL = 1, 4, 3


def read_header(path: pathlib.Path) -> dict:
    """The .yskn's JSON header (strata/artifact/yskn.hpp: "YSKN", version 1, the JSON's length at 8, the JSON at 0x20)"""
    with open(path, "rb") as f:
        head = f.read(0x20)
        if len(head) < 0x20 or head[:4] != b"YSKN":
            raise ValueError(f"{path} is not a YSKN container")
        version, length = struct.unpack_from("<II", head, 4)
        if version != 1:
            raise ValueError(f"{path}: version {version} (1 is read)")
        return json.loads(f.read(length))


def read_vector(path: pathlib.Path, v: dict) -> np.ndarray:
    """A vector of the .yskn as FP32"""
    dtype = v.get("dtype", "f16")
    size = {"bf16": 2, "f16": 2, "f32": 4}[dtype]
    with open(path, "rb") as f:
        f.seek(int(v["off"]))
        raw = f.read(int(v["n"]) * size)
    if len(raw) != int(v["n"]) * size:
        raise ValueError(f"{path}: vector {v['name']} is truncated")
    if dtype == "bf16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, "<f2" if dtype == "f16" else "<f4").astype(np.float32)


def v_tiled(n_k: int, v_per_k: int, dim: int) -> np.ndarray:
    """llama.cpp's _reorder_v_heads as an index: tiled[i] = grouped[index[i]], the heads (k, v per k) -> (v per k, k)"""
    return np.arange(n_k * v_per_k * dim).reshape(n_k, v_per_k, dim).transpose(1, 0, 2).reshape(-1)


def tokenizer_meta(tok_path: pathlib.Path, vocab_size: int) -> dict:
    """tokens, merges and token types as llama.cpp's converter writes a byte-level BPE (get_vocab_base): the ids
    without a token are [PAD<id>] (user-defined), an added token is a control token when special, else user-defined"""
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    model = tok.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise ValueError(f"{tok_path}: a byte-level BPE tokenizer.json is expected")
    by_id: dict[int, str] = {i: t for t, i in model["vocab"].items()}
    added: dict[int, bool] = {}
    for a in tok.get("added_tokens", []):
        by_id[int(a["id"])] = a["content"]
        added[int(a["id"])] = bool(a.get("special", False))
    if max(by_id) >= vocab_size:
        raise ValueError(f"{tok_path}: token id {max(by_id)} is past the model's vocabulary ({vocab_size})")
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in by_id:
            tokens.append(f"[PAD{i}]")
            types.append(USER_DEFINED)
        elif i in added:
            tokens.append(by_id[i])
            types.append(CONTROL if added[i] else USER_DEFINED)
        else:
            tokens.append(by_id[i])
            types.append(NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", [])]
    ids = {t: i for i, t in enumerate(tokens)}
    meta = {"tokens": tokens, "types": types, "merges": merges}
    for key, name in (("eos", "<|im_end|>"), ("pad", "<|endoftext|>")):
        if name in ids:
            meta[key] = ids[name]
    return meta


def build(model_dir: pathlib.Path, yskn: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    head = read_header(yskn)
    m = head["model"]
    if m["arch"] != "qwen3_5":
        raise ValueError(f"{yskn}: arch {m['arch']} (qwen3_5 is read)")
    H, L, F = int(tc["hidden_size"]), int(tc["num_hidden_layers"]), int(tc["intermediate_size"])
    if (H, L, F) != (int(m["hidden"]), int(m["layers"]), int(m["ffn"])):
        raise ValueError("config.json and the .yskn's model differ")
    n_k, n_v = int(tc["linear_num_key_heads"]), int(tc["linear_num_value_heads"])
    dk, dv = int(tc["linear_key_head_dim"]), int(tc["linear_value_head_dim"])
    conv_k = int(tc["linear_conv_kernel_dim"])
    if n_v % n_k:
        raise ValueError("the GDN's V heads are not a multiple of its K heads")
    vpk = n_v // n_k
    interval = int(tc.get("full_attention_interval", 4))
    types = m.get("layer_types") or tc.get("layer_types")
    want = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(L)]
    if types is not None and list(types) != want:
        raise ValueError("the layers' kinds are not every %d-th full attention" % interval)
    rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    theta = float(rope.get("rope_theta", tc.get("rope_theta", m["rope_theta"])))
    partial = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
    sections = list(rope.get("mrope_section", [11, 11, 10]))
    sections = (sections + [0, 0, 0, 0])[:4]
    head_dim = int(tc.get("head_dim", H // int(tc["num_attention_heads"])))

    w = GGUFWriter()
    a = "qwen35"
    w.add("general.architecture", a)
    w.add("general.type", "model")
    w.add("general.name", model_dir.name)
    w.add("general.license", "apache-2.0")
    w.add(f"{a}.block_count", L, "u32")
    w.add(f"{a}.context_length", int(tc.get("max_position_embeddings", 262144)), "u32")
    w.add(f"{a}.embedding_length", H, "u32")
    w.add(f"{a}.feed_forward_length", F, "u32")
    w.add(f"{a}.attention.head_count", int(tc["num_attention_heads"]), "u32")
    w.add(f"{a}.attention.head_count_kv", int(tc["num_key_value_heads"]), "u32")
    w.add(f"{a}.rope.dimension_sections", sections, "array:i32")
    w.add(f"{a}.rope.freq_base", theta, "f32")
    w.add(f"{a}.attention.layer_norm_rms_epsilon", float(tc.get("rms_norm_eps", m["eps"])), "f32")
    w.add(f"{a}.attention.key_length", head_dim, "u32")
    w.add(f"{a}.attention.value_length", head_dim, "u32")
    w.add(f"{a}.ssm.conv_kernel", conv_k, "u32")
    w.add(f"{a}.ssm.state_size", dk, "u32")
    w.add(f"{a}.ssm.group_count", n_k, "u32")
    w.add(f"{a}.ssm.time_step_rank", n_v, "u32")
    w.add(f"{a}.ssm.inner_size", n_v * dv, "u32")
    w.add(f"{a}.full_attention_interval", interval, "u32")
    w.add(f"{a}.rope.dimension_count", int(head_dim * partial), "u32")
    t = tokenizer_meta(model_dir / "tokenizer.json", int(tc.get("vocab_size", m["vocab"])))
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "qwen35")
    w.add("tokenizer.ggml.tokens", t["tokens"], "array:string")
    w.add("tokenizer.ggml.token_type", t["types"], "array:i32")
    w.add("tokenizer.ggml.merges", t["merges"], "array:string")
    if "eos" in t:
        w.add("tokenizer.ggml.eos_token_id", t["eos"], "u32")
    if "pad" in t:
        w.add("tokenizer.ggml.padding_token_id", t["pad"], "u32")
    tpl = model_dir / "chat_template.jinja"
    if tpl.exists():
        w.add("tokenizer.chat_template", tpl.read_text(encoding="utf-8"))
    w.add("xestrata.yskn.file", yskn.name)

    vectors = {v["name"]: v for v in head["vectors"]}

    def vec(name: str, n: int) -> np.ndarray:
        if name not in vectors:
            raise ValueError(f"{yskn}: no vector {name}")
        x = read_vector(yskn, vectors[name])
        if x.size != n:
            raise ValueError(f"{yskn}: vector {name} has {x.size} values, {n} expected")
        return x

    def put(name: str, x: np.ndarray, shape: list[int]) -> None:   # shape in GGUF order: dim 0 fastest
        a32 = np.ascontiguousarray(x, dtype="<f4")
        if a32.size != int(np.prod(shape)):
            raise ValueError(f"{name}: {a32.size} values for {shape}")
        w.tensors.append(_Tensor(name, shape, "F32", a32.tobytes()))

    put("output_norm.weight", vec("norm", H) + 1.0, [H])
    v_rows = v_tiled(n_k, vpk, dv)
    v_heads = v_tiled(n_k, vpk, 1)
    qk = 2 * n_k * dk
    for i in range(L):
        p, b = f"layers.{i}.", f"blk.{i}."
        put(b + "attn_norm.weight", vec(p + "input_layernorm", H) + 1.0, [H])
        put(b + "post_attention_norm.weight", vec(p + "post_attention_layernorm", H) + 1.0, [H])
        if want[i] == "full_attention":
            put(b + "attn_q_norm.weight", vec(p + "self_attn.q_norm", head_dim) + 1.0, [head_dim])
            put(b + "attn_k_norm.weight", vec(p + "self_attn.k_norm", head_dim) + 1.0, [head_dim])
            continue
        la = p + "linear_attn."
        put(b + "ssm_a", -np.exp(vec(la + "A_log", n_v))[v_heads], [n_v])
        put(b + "ssm_dt.bias", vec(la + "dt_bias", n_v)[v_heads], [n_v])
        conv = vec(la + "conv1d", (qk + n_v * dv) * conv_k).reshape(qk + n_v * dv, conv_k)
        conv = np.concatenate([conv[:qk], conv[qk:][v_rows]])
        put(b + "ssm_conv1d.weight", conv, [conv_k, qk + n_v * dv])
        put(b + "ssm_norm.weight", vec(la + "norm", dv), [dv])
        for g, y in (("ssm_alpha.weight", "in_proj_a"), ("ssm_beta.weight", "in_proj_b")):
            put(b + g, vec(la + y, n_v * H).reshape(n_v, H)[v_heads], [H, n_v])
    w.write(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument("--yskn", help="the .yskn's name in MODEL_DIR (default: the only one there)")
    ap.add_argument("--out", type=pathlib.Path, help="the skeleton (default: MODEL_DIR/<the .yskn's stem>.gguf)")
    args = ap.parse_args()
    if args.yskn:
        yskn = args.model_dir / args.yskn
    else:
        found = [f for f in args.model_dir.glob("*.yskn") if not f.name.startswith("bdraft")]
        if len(found) != 1:
            ap.error(f"{args.model_dir}: {len(found)} model .yskn files; name one with --yskn")
        yskn = found[0]
    out = args.out or args.model_dir / (yskn.stem + ".gguf")
    if out.resolve().parent != yskn.resolve().parent:
        ap.error("the skeleton names the .yskn by its file name: write it beside the .yskn")
    build(args.model_dir, yskn, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
