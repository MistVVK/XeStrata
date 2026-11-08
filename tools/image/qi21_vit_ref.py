# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi21_vit_ref.py - a NumPy reference of Qwen-Image-2.1's vision encoder (Qwen3-VL's ViT, the mmproj
GGUF) in double precision (the attention in single), to tell which of xestrata-image and stable-diffusion.cpp is the
nearer one where they differ.

    XESTRATA_GGUF_PY=<llama.cpp>/gguf-py python tools/image/qi21_vit_ref.py MMPROJ.gguf DUMP_DIR [OUT_DIR]

DUMP_DIR: a stable-diffusion.cpp run's dump (vit_in_0.bin, [3][H][W] FP32 in -1 .. 1; vit_out_0_K.bin, K = 0 the
embeddings, 1.. the DeepStack blocks).  Prints how far the run's values are from the reference; OUT_DIR gets the
reference's as ref_out_K.bin (FP32), which xestrata-image-vision-probe can read in place of the run's.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, for serve/envvars.py
from serve import envvars  # noqa: E402

sys.path.insert(0, envvars.get("GGUF_PY", ""))
import gguf  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    r = gguf.GGUFReader(sys.argv[1])
    dump = Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    t = {x.name: x for x in r.tensors}

    def w(name):   # a weight as float64, in GGUF's row order ([rows][cols] for a matrix)
        x = t[name]
        return np.asarray(x.data, dtype=np.float64).reshape([int(d) for d in reversed(x.shape)])

    def field(key, default):
        f = r.fields.get(key)
        return default if f is None else f.parts[f.data[0]][0].item()

    C = int(field("clip.vision.embedding_length", 1152))
    heads = int(field("clip.vision.attention.head_count", 16))
    layers = int(field("clip.vision.block_count", 27))
    eps = float(field("clip.vision.attention.layer_norm_epsilon", 1e-6))
    P, M = 16, 2
    hd = C // heads

    rgb = np.fromfile(dump / "vit_in_0.bin", dtype=np.float32).astype(np.float64)
    side = int(round(math.sqrt(rgb.size / 3)))
    rgb = rgb.reshape(3, side, side)
    ph = pw = side // P
    # the patches in the merge's order
    order = [(y2 * M + dy, x2 * M + dx) for y2 in range(ph // M) for x2 in range(pw // M) for dy in range(M)
             for dx in range(M)]
    patches = np.stack([rgb[:, py * P:(py + 1) * P, px * P:(px + 1) * P].reshape(-1) for py, px in order])
    pw0 = w("v.patch_embd.weight").reshape(C, -1)
    pw1 = w("v.patch_embd.weight.1").reshape(C, -1) if "v.patch_embd.weight.1" in t else 0
    x = patches @ (pw0 + pw1).T + w("v.patch_embd.bias")
    # the position embeddings, bilinear with the corners aligned
    pos = w("v.position_embd.weight")
    g = int(round(math.sqrt(pos.shape[0])))
    pos = pos.reshape(g, g, C)

    def interp(i, n):
        f = i * (g - 1) / (n - 1) if n > 1 else 0.0
        lo = int(math.floor(f))
        return lo, min(lo + 1, g - 1), f - lo
    for k, (py, px) in enumerate(order):
        y0, y1, fy = interp(py, ph)
        x0, x1, fx = interp(px, pw)
        x[k] += ((1 - fy) * ((1 - fx) * pos[y0, x0] + fx * pos[y0, x1])
                 + fy * ((1 - fx) * pos[y1, x0] + fx * pos[y1, x1]))
    # the rotary embedding: pairs (j, j + hd/2), the first hd/4 by row, the rest by column
    half, quarter = hd // 2, hd // 4
    inv = 10000.0 ** (-2.0 * np.arange(quarter) / half)
    ys = np.array([p[0] for p in order], dtype=np.float64)
    xs = np.array([p[1] for p in order], dtype=np.float64)
    ang = np.concatenate([ys[:, None] * inv, xs[:, None] * inv], axis=1)   # [T][half]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(v):   # v [T][heads][hd]
        a, b = v[..., :half], v[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def ln(v, name):
        m = v.mean(-1, keepdims=True)
        var = ((v - m) ** 2).mean(-1, keepdims=True)
        return (v - m) / np.sqrt(var + eps) * w(name + ".weight") + w(name + ".bias")

    def lin(v, name):
        y = v @ w(name + ".weight").T
        return y + w(name + ".bias") if name + ".bias" in t else y

    def gelu_tanh(v):
        return 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))

    def gelu_erf(v):
        from math import erf
        return 0.5 * v * (1 + np.vectorize(erf)(v / math.sqrt(2)))

    T = x.shape[0]
    outs = {}
    ds = 0
    for i in range(layers):
        p = f"v.blk.{i}."
        h = ln(x, p + "ln1")
        qkv = lin(h, p + "attn_qkv").reshape(T, 3, heads, hd)
        q, k, v = rope(qkv[:, 0]), rope(qkv[:, 1]), qkv[:, 2]
        att = np.empty((T, heads, hd))
        for hh in range(heads):
            s = (q[:, hh].astype(np.float32) @ k[:, hh].astype(np.float32).T) / np.float32(math.sqrt(hd))
            s -= s.max(1, keepdims=True)
            e = np.exp(s)
            att[:, hh] = (e / e.sum(1, keepdims=True)) @ v[:, hh].astype(np.float32)
        x = x + lin(att.reshape(T, C), p + "attn_out")
        h = ln(x, p + "ln2")
        x = x + lin(gelu_tanh(lin(h, p + "ffn_up")), p + "ffn_down")
        d = f"v.deepstack.{i}."
        if d + "fc1.weight" in t:
            ds += 1
            m = ln(x.reshape(T // 4, 4 * C), d + "norm")
            outs[ds] = lin(gelu_erf(lin(m, d + "fc1")), d + "fc2")
        print(f"layer {i}", flush=True, file=sys.stderr)
    m = ln(x, "v.post_ln").reshape(T // 4, 4 * C)
    outs[0] = lin(gelu_erf(lin(m, "mm.0")), "mm.2")
    for k in sorted(outs):
        ref = outs[k].reshape(-1)
        run = np.fromfile(dump / f"vit_out_0_{k}.bin", dtype=np.float32).astype(np.float64)
        cos_ = float(ref @ run / np.linalg.norm(ref) / np.linalg.norm(run))
        rel = float(np.abs(ref - run).sum() / np.abs(ref).sum())
        print(f"out {k}: stable-diffusion.cpp against the reference: cos {cos_:.6f} rel {rel:.3g}")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            outs[k].astype(np.float32).tofile(out_dir / f"ref_out_{k}.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
