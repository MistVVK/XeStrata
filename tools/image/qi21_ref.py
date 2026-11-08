# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi21_ref.py - Qwen-Image-2.1's DiT in NumPy (FP32), the reference the image engine's kernels are
checked against.  Written from the model's structure: a single stream of the text tokens then the image tokens,
32 blocks of LayerNorm with a time scale, attention with RMS-normed q and k and a 3-axis rotary embedding (text
tokens see the text before them, image tokens see everything), and a SwiGLU MLP with a time gate.

    PY=... XESTRATA_GGUF_PY=<llama.cpp>/gguf-py python tools/image/qi21_ref.py DIT.gguf DUMP_DIR [STEP] [LAYERS]
    ... python tools/image/qi21_ref.py --dump-block DIT.gguf DUMP_DIR OUT_DIR STEP BLOCK

DUMP_DIR holds a run's dit_x_STEP / dit_t_STEP / dit_ctx_STEP (.bin + .json); prints how far this DiT's output is
from dit_out_STEP.  LAYERS (default all) runs only the first blocks (then nothing to compare: it prints the norm).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, for serve/envvars.py
from serve import envvars  # noqa: E402

HEAD = 128
AXES = (16, 56, 56)            # the rotary embedding's dims per axis: frame / text position, height, width
EPS = 1e-6


def load(d: Path, name: str) -> np.ndarray:
    meta = json.loads((d / f"{name}.json").read_text())
    a = np.fromfile(d / f"{name}.bin", np.int32 if meta["type"] == "i32" else np.float32)
    return a.reshape(list(reversed(meta["shape"])))   # ggml's ne[0] is the fastest axis


class Weights:
    def __init__(self, path: str):
        sys.path.insert(0, envvars.get("GGUF_PY", ""))
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.r, self.deq = GGUFReader(path), dequantize
        self.by = {t.name.removeprefix("model.diffusion_model."): t for t in self.r.tensors}

    def __call__(self, name: str) -> np.ndarray:
        t = self.by[name]
        w = self.deq(t.data, t.tensor_type).astype(np.float32)
        return w.reshape([int(x) for x in reversed(t.shape)]) if w.ndim == 1 and len(t.shape) > 1 else w


def layer_norm(x):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS)


def rms_norm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def timestep_embedding(t: float, dim=256, max_period=10000.0):
    half = dim // 2
    f = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    a = np.float32(t) * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def rope_tables(pos: np.ndarray):
    """pos [L, 3] -> cos, sin [L, HEAD/2]: each axis's pairs take 1/10000^(2j/dim) of that axis's position."""
    cols = []
    for axis, dim in enumerate(AXES):
        omega = 1.0 / np.power(np.float32(10000.0), np.linspace(0, (dim - 2) / dim, dim // 2, dtype=np.float32))
        cols.append(pos[:, axis:axis + 1] * omega[None, :])
    ang = np.concatenate(cols, 1)
    return np.cos(ang), np.sin(ang)


def rope(x, cos, sin):
    """x [L, H, HEAD]; the pairs (2j, 2j + 1) rotate by the angle of pair j."""
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def positions(n_text: int, h: int, w: int) -> np.ndarray:
    p = [[i, i, i] for i in range(n_text)]
    for y in range(h):
        for x in range(w):
            p.append([n_text, y - (h - h // 2), x - (w - w // 2)])
    return np.array(p, np.float32)


def attention(q, k, v, n_text):
    """q, k, v [L, H, HEAD]: text rows see the text up to themselves, image rows see every row."""
    L, H, _ = q.shape
    out = np.empty_like(q)
    scale = 1.0 / np.sqrt(HEAD)
    for hh in range(H):
        s = (q[:, hh] @ k[:, hh].T) * scale
        s[:n_text, n_text:] = -np.inf
        s[:n_text, :n_text][np.triu_indices(n_text, 1)] = -np.inf
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        out[:, hh] = p @ v[:, hh]
    return out


def block(W, i, x, mods, n_text, cos, sin):
    """One transformer block; mods [4][2, 4096]: row 0 the image tokens' time, row 1 the text's (time 0)."""
    p = f"transformer_blocks.{i}."

    def mod(h, m, gate=False):
        f = (lambda r: np.tanh(r)) if gate else (lambda r: 1 + r)
        out = h.copy()
        out[:n_text] *= f(m[1])
        out[n_text:] *= f(m[0])
        return out
    L = x.shape[0]
    h = mod(layer_norm(x), mods[0])
    q = (h @ W(p + "attn.to_q.weight").T).reshape(L, -1, HEAD)
    k = (h @ W(p + "attn.to_k.weight").T).reshape(L, -1, HEAD)
    v = (h @ W(p + "attn.to_v.weight").T).reshape(L, -1, HEAD)
    q = rope(rms_norm(q, W(p + "attn.norm_q.weight")), cos, sin)
    k = rope(rms_norm(k, W(p + "attn.norm_k.weight")), cos, sin)
    a = attention(q, k, v, n_text).reshape(L, -1) @ W(p + "attn.to_out.0.weight").T
    x = x + mod(a, mods[1], gate=True)
    h = mod(layer_norm(x), mods[2])
    gu = h @ W(p + "img_mlp.gate_up.weight").T
    inter = gu.shape[1] // 2
    h = (gu[:, inter:] * silu(gu[:, :inter])) @ W(p + "img_mlp.out.weight").T
    return x + mod(h, mods[3], gate=True)


def dit(W, latent, t: float, ctx, layers=None):
    """latent [C=64, H, W], ctx [n_text, 4096] -> the DiT's output [64, H, W]."""
    C, H, Wd = latent.shape
    n_text = ctx.shape[0]
    temb = np.stack([timestep_embedding(t), timestep_embedding(0.0)])            # [2, 256]
    temb = silu(temb @ W("time_text_embed.timestep_embedder.linear_1.weight").T) @ \
        W("time_text_embed.timestep_embedder.linear_2.weight").T
    temb = silu(temb)
    mod = temb @ W("modulation.1.weight").T                                       # [2, 16384]
    mods = [mod[:, k * 4096:(k + 1) * 4096] for k in range(4)]
    text = rms_norm(ctx, W("txt_in.text_norm.weight") + 1)
    text = gelu_tanh(text @ W("txt_in.in_layer.weight").T) @ W("txt_in.out_layer.weight").T
    img = latent.reshape(C, H * Wd).T @ W("img_in.weight").T
    x = np.concatenate([text, img]).astype(np.float32)
    cos, sin = rope_tables(positions(n_text, H, Wd))
    n = 32 if layers is None else layers
    for i in range(n):
        x = block(W, i, x, mods, n_text, cos, sin).astype(np.float32)
        print(f"  block {i}: |x| {np.abs(x).mean():.4f}", flush=True)
    if layers is not None:
        return x
    x = x[n_text:]
    scale = temb[0] @ W("norm_out.linear.weight").T
    x = layer_norm(x) * (1 + scale)
    x = x @ W("proj_out.weight").T
    return x.T.reshape(C, H, Wd)


def save(d: Path, name: str, a: np.ndarray) -> None:
    a = np.ascontiguousarray(a)
    a.tofile(d / f"{name}.bin")
    (d / f"{name}.json").write_text(json.dumps({"type": "i32" if a.dtype == np.int32 else "f32",
                                                "shape": list(reversed(a.shape))}))


def dump_block(W, src: Path, out: Path, step: int, index: int) -> None:
    """Block `index`'s input and output at step `step` of a run in `src`, for the image engine's block probe:
    x_in / x_out [T, C], mods [4, 2, C], rope_cos / rope_sin [T, HEAD / 2], attn [T, C] (the attention before
    to_out), n_text."""
    out.mkdir(parents=True, exist_ok=True)
    latent = load(src, f"dit_x_{step}")[0]
    t = float(load(src, f"dit_t_{step}")[0])
    ctx = load(src, f"dit_ctx_{step}")
    C, H, Wd = latent.shape
    n_text = ctx.shape[0]
    temb = np.stack([timestep_embedding(t), timestep_embedding(0.0)])
    temb = silu(temb @ W("time_text_embed.timestep_embedder.linear_1.weight").T) @ \
        W("time_text_embed.timestep_embedder.linear_2.weight").T
    mod = silu(temb) @ W("modulation.1.weight").T
    mods = np.stack([mod[:, k * 4096:(k + 1) * 4096] for k in range(4)]).astype(np.float32)
    text = rms_norm(ctx, W("txt_in.text_norm.weight") + 1)
    text = gelu_tanh(text @ W("txt_in.in_layer.weight").T) @ W("txt_in.out_layer.weight").T
    x = np.concatenate([text, latent.reshape(C, H * Wd).T @ W("img_in.weight").T]).astype(np.float32)
    cos, sin = rope_tables(positions(n_text, H, Wd))
    for i in range(index):
        x = block(W, i, x, mods, n_text, cos, sin).astype(np.float32)
    save(out, "x_in", x)
    save(out, "mods", mods)
    save(out, "rope_cos", cos.astype(np.float32))
    save(out, "rope_sin", sin.astype(np.float32))
    save(out, "n_text", np.array([n_text], np.int32))
    global attention
    plain, seen = attention, {}

    def keep(q, k, v, n):
        seen["attn"] = plain(q, k, v, n)
        return seen["attn"]
    attention = keep
    y = block(W, index, x, mods, n_text, cos, sin).astype(np.float32)
    attention = plain
    save(out, "attn", seen["attn"].reshape(x.shape[0], -1).astype(np.float32))
    save(out, "x_out", y)
    print(f"block {index} of step {step}: T {x.shape[0]}, n_text {n_text} -> {out}")


def main() -> None:
    if sys.argv[1] == "--dump-block":
        dump_block(Weights(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        return
    gguf, d = sys.argv[1], Path(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    W = Weights(gguf)
    x = load(d, f"dit_x_{step}")[0]                     # [C, H, W]
    t = float(load(d, f"dit_t_{step}")[0])
    ctx = load(d, f"dit_ctx_{step}")
    out = dit(W, x, t, ctx, layers)
    if layers is not None:
        print("norm", float(np.abs(out).mean()))
        return
    ref = load(d, f"dit_out_{step}")[0]
    diff = np.abs(out - ref)
    cos = np.sum(out * ref) / np.linalg.norm(out) / np.linalg.norm(ref)
    print(f"step {step}: max {diff.max():.4g} mean {diff.mean():.4g} ref |mean| {np.abs(ref).mean():.4g} "
          f"rel {diff.mean() / np.abs(ref).mean():.4g} cos {cos:.6f}")


if __name__ == "__main__":
    main()
