# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/mf_train_ref.py - the training side's ops in NumPy (FP64), what the engine's gradient kernels are
checked against (include/strata/image/mf_train.hpp, src/image/mf_train.cpp).

    PY=... python tools/image/mf_train_ref.py --probe build/free/xestrata-image-mf-train-probe [--dir /tmp/mf-train]
        [--op lnmod]

Each case writes the op's inputs into <dir>/<case> the way the image engine dumps them (<name>.bin with <name>.json,
the fastest dim first; "f16" holds FP16 bits), runs the probe for it, and compares what came back with the same
gradient in FP64.  A line prints the largest relative difference over the tensor and its tolerance; the exit code is
the number of lines over it.

The ops are the block's own math: a LayerNorm with a modulation, the tanh GELU, a gated residual, the time's SiLU, the
VAE's 2x2 packing with its batch norm, the flow loss, and a product's dgrad / wgrad / bias sum through FP16 rows with
their power-of-two scales - the forms src/image/mf_dit.cpp reads in the forward.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

EPS_LN = 1e-6        # the LayerNorm's and the q/k norms' eps, as dit_rows takes it
ROW_LIM = 16384.0    # dit_rows' kRowLim: a row's power-of-two scale brings its largest value under this


def pow2_scale(amax) -> float:
    amax = float(amax)
    return float(2.0 ** np.ceil(np.log2(amax / ROW_LIM))) if amax > ROW_LIM else 1.0


def save(d: Path, name: str, a: np.ndarray) -> None:
    a = np.ascontiguousarray(a)
    t = "f16" if a.dtype == np.float16 else "i32" if a.dtype == np.int32 else "f32"
    a.tofile(d / f"{name}.bin")
    (d / f"{name}.json").write_text(json.dumps({"type": t, "shape": list(reversed(a.shape))}))


def load(d: Path, name: str) -> np.ndarray:
    meta = json.loads((d / f"{name}.json").read_text())
    dt = {"f16": np.float16, "i32": np.int32}.get(meta["type"], np.float32)
    a = np.fromfile(d / f"{name}.bin", dt)
    return a.reshape(list(reversed(meta["shape"]))) if "shape" in meta else a


def rel(got: np.ndarray, want: np.ndarray) -> float:
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.max(np.abs(g - w)) / max(float(np.max(np.abs(w))), 1e-12))


# ----------------------------------------------------------------------- the cases
# each: write the inputs into d, return the FP64 gradients keyed by the probe's output file name


def case_lnmod(d: Path, r: np.random.Generator):
    T, C = 320, 192
    x = r.standard_normal((T, C)) * 3.0
    scale, dy = r.standard_normal(C) * 0.2, r.standard_normal((T, C))
    mean, inv = x.mean(1), 1.0 / np.sqrt(x.var(1) + EPS_LN)
    xhat = (x - mean[:, None]) * inv[:, None]
    for name, a in (("x", x), ("mean", mean), ("inv", inv), ("scale", scale), ("dy", dy)):
        save(d, name, a.astype(np.float32))
    m = dy * (1.0 + scale[None, :])
    return {"dx": inv[:, None] * (m - (m.sum(1, keepdims=True) + xhat * (m * xhat).sum(1, keepdims=True)) / C),
            "dshift": dy.sum(0), "dscale": (dy * xhat).sum(0)}


def _element(d: Path, r: np.random.Generator, kind: str):
    n = 4096
    v = r.standard_normal(n) * 2.0
    dy = r.standard_normal(n)
    save(d, "v" if kind == "gelu" else "y", v.astype(np.float32))
    save(d, "dy", dy.astype(np.float32))
    if kind == "gelu":
        k, c = 0.7978845608028654, 0.044715
        th = np.tanh(k * (v + c * v**3))
        return {"d": dy * (0.5 * (1 + th) + 0.5 * v * (1 - th**2) * k * (1 + 3 * c * v**2))}
    s = 1.0 / (1.0 + np.exp(-v))
    return {"d": dy * s * (1.0 + v * (1.0 - s))}


def case_gelu(d: Path, r: np.random.Generator):
    return _element(d, r, "gelu")


def case_silu(d: Path, r: np.random.Generator):
    return _element(d, r, "silu")


def case_gate(d: Path, r: np.random.Generator):
    T, C = 256, 128
    h, gate, dx = r.standard_normal((T, C)) * 1.7, r.standard_normal(C) * 0.5, r.standard_normal((T, C))
    for name, a in (("h", h), ("gate", gate), ("dx", dx)):
        save(d, name, a.astype(np.float32))
    return {"dh": dx * gate[None, :], "dgate": (dx * h).sum(0)}


def case_patch(d: Path, r: np.random.Generator):
    h, w = 6, 5                        # tokens: the latent is 2h x 2w positions of 32 channels
    m = np.zeros((2 * h, 2 * w, 64))            # the posterior's rows: the mean's 32, then the log-variance's
    mean_m = r.standard_normal((2 * h, 2 * w, 32))
    m[:, :, :32] = mean_m
    mean = r.standard_normal(128) * 0.3
    sd = 1.0 + np.abs(r.standard_normal(128)) * 0.5
    dz = r.standard_normal((h, w, 128))
    for name, a in (("m", m), ("mean", mean), ("sd", sd), ("dz", dz)):
        save(d, name, a.astype(np.float32))
    # the forward: z[t][k] = (m[Y][X][c] - mean[k]) / sd[k] with k = 4c + 2(Y%2) + X%2, t = (Y/2) w + X/2
    dm, dmean, dsd = np.zeros_like(mean_m), np.zeros(128), np.zeros(128)
    for ay in (0, 1):
        for ax in (0, 1):
            for c in range(32):
                k = 4 * c + 2 * ay + ax
                g = dz[:, :, k] / sd[k]
                dm[ay::2, ax::2, c] = g
                dmean[k] = -g.sum()
                dsd[k] = -(dz[:, :, k] * (mean_m[ay::2, ax::2, c] - mean[k]) / sd[k] ** 2).sum()
    return {"dm": dm, "dmean": dmean, "dsd": dsd}


def case_flow(d: Path, r: np.random.Generator):
    n = 5000
    vp, x0, eps = r.standard_normal(n) * 1.3, r.standard_normal(n), r.standard_normal(n)
    for name, a in (("vp", vp), ("x0", x0), ("eps", eps)):
        save(d, name, a.astype(np.float32))
    diff = vp - (eps - x0)
    return {"dv": 2 * diff / n, "dx0": -2 * diff / n, "deps": 2 * diff / n, "loss": np.array([(diff**2).mean()])}


def case_products(d: Path, r: np.random.Generator):
    """One product's rows and their power-of-two scales (what dit_rows' rows16 writes), its dgrad, its wgrad and its
    bias sum.  The wgrad sums over the rows, so its side of the scales becomes one global power of two G."""
    T, N, K = 256, 128, 96
    x = r.standard_normal((T, K)) * 400.0
    rs = np.array([pow2_scale(np.abs(row).max()) for row in x], dtype=np.float32)
    x16 = (x / rs[:, None]).astype(np.float16)
    w = (r.standard_normal((N, K)) * 0.05).astype(np.float16)
    dy = r.standard_normal((T, N)) * 2.0
    rsY = np.array([pow2_scale(np.abs(row).max()) for row in dy], dtype=np.float32)
    dy16 = (dy / rsY[:, None]).astype(np.float16)
    A = dy * rs[:, None]                       # the row factors that cancel against the stored X rows
    G = pow2_scale(np.abs(A).max())
    save(d, "x16", x16)
    save(d, "x16t", np.ascontiguousarray(x16.T.astype(np.float16)))
    save(d, "wt", np.ascontiguousarray(w.T.astype(np.float16)))
    save(d, "dy16", dy16)
    save(d, "rsY", rsY)
    save(d, "a16t", np.ascontiguousarray((A / G).T.astype(np.float16)))
    save(d, "gscale", np.array([G], dtype=np.float32))
    save(d, "dy", dy.astype(np.float32))
    dY = dy16.astype(np.float64) * rsY[:, None]
    X = x16.astype(np.float64) * rs[:, None]
    # the bias sum reads the gradient's own FP32 rows (the tape holds them), not the FP16 form
    return {"dx": dY @ w.astype(np.float64), "dw": dY.T @ X, "db": dy.astype(np.float64).sum(0)}


def case_attn(d: Path, r: np.random.Generator):
    """The joint attention of a block at the small probe's shape."""
    return attn_rows(d, r, 96, 4, 128)


def case_attnreal(d: Path, r: np.random.Generator):
    """The same at Mage-Flow's real widths - 24 heads of 128 - and 1024 rows, a quarter of what a 1024 x 1024 image
    reaches: enough for the head count, the head width and the row scales to bite without the reference holding a
    4096 x 4096 FP64 score matrix per head."""
    return attn_rows(d, r, 1024, 24, 128)


def case_attncausal(d: Path, r: np.random.Generator):
    """The first 48 rows causal (a block's text rows beside its image rows): the library backward's text rows go
    through the fused kernel, and where it cannot run (XESTRATA_NO_XMX=1) the whole step must fall to the products
    without a throw and with the same gradients."""
    return attn_rows(d, r, 192, 4, 128, nt=48)


def attn_rows(d: Path, r: np.random.Generator, rows: int, heads: int, hd: int, nt: int = 0):
    """Each head's rows are FP16 with the head's power of two already folded out, and one row sees every key;
    the rows below nt (causal) see the keys up to themselves."""
    q = (r.standard_normal((rows, heads * hd)) * 0.6).astype(np.float16)
    k = (r.standard_normal((rows, heads * hd)) * 0.6).astype(np.float16)
    v = (r.standard_normal((rows, heads * hd)) * 0.8).astype(np.float16)
    save(d, "q16", q)
    save(d, "k16", k)
    save(d, "v16", v)
    save(d, "head_dim", np.array([hd], dtype=np.float32))
    if nt:
        save(d, "n_causal", np.array([nt], dtype=np.float32))
    dO = r.standard_normal((rows, heads * hd)).astype(np.float32)
    save(d, "dO", dO)
    Q = q.astype(np.float64).reshape(rows, heads, hd).transpose(1, 0, 2)
    K = k.astype(np.float64).reshape(rows, heads, hd).transpose(1, 0, 2)
    V = v.astype(np.float64).reshape(rows, heads, hd).transpose(1, 0, 2)
    DO = dO.reshape(rows, heads, hd).transpose(1, 0, 2)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)
    if nt:
        S[:, :nt, :][:, np.arange(nt)[:, None] < np.arange(rows)[None, :]] = -np.inf
    P = np.exp(S - S.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    oo = P @ V
    lse = np.log(np.exp(S - S.max(-1, keepdims=True)).sum(-1)) + S.max(-1)
    D = (DO * oo).sum(-1)
    dS = P * ((DO @ V.transpose(0, 2, 1)) - D[:, :, None]) / np.sqrt(hd)
    o16 = oo.transpose(1, 0, 2).reshape(rows, heads * hd).astype(np.float16)
    o32 = oo.transpose(1, 0, 2).reshape(rows, heads * hd)
    want = {"lse": lse.transpose(1, 0).ravel(), "o16": o16.astype(np.float32).ravel(),
            "o32": o32.astype(np.float32).ravel(),
            "dQ": (dS @ K).transpose(1, 0, 2).reshape(-1),
            "dK": (dS.transpose(0, 2, 1) @ Q).transpose(1, 0, 2).reshape(-1),
            "dV": (P.transpose(0, 2, 1) @ DO).transpose(1, 0, 2).reshape(-1)}
    return want


def case_productsreal(d: Path, r: np.random.Generator):
    """`case_products` at the shapes a real Mage-Flow block gives a product: 128 rows of 3072 through a 12288-wide
    mlp1 (the block's widest dgrad) and its wgrad back.  Inference runs these shapes every step, so a fault here is
    the training side's own buffers, not `prefill::Gemm`."""
    T, N, K = 128, 12288, 3072
    x = r.standard_normal((T, K)) * 400.0
    rs = np.array([pow2_scale(np.abs(row).max()) for row in x], dtype=np.float32)
    x16 = (x / rs[:, None]).astype(np.float16)
    w = (r.standard_normal((N, K)) * 0.02).astype(np.float16)
    dy = r.standard_normal((T, N)) * 2.0
    rsY = np.array([pow2_scale(np.abs(row).max()) for row in dy], dtype=np.float32)
    dy16 = (dy / rsY[:, None]).astype(np.float16)
    A = dy * rs[:, None]
    G = pow2_scale(np.abs(A).max())
    save(d, "x16", x16)
    save(d, "x16t", np.ascontiguousarray(x16.T.astype(np.float16)))
    save(d, "wt", np.ascontiguousarray(w.T.astype(np.float16)))
    save(d, "dy16", dy16)
    save(d, "dy", dy.astype(np.float32))   # the bias sum reads the gradient's own rows, in the dump's FP32
    save(d, "rsY", rsY)
    save(d, "a16t", np.ascontiguousarray((A / G).T.astype(np.float16)))
    save(d, "gscale", np.array([G], dtype=np.float32))
    return {"dx": dy.astype(np.float64) @ w.astype(np.float64),
            "dw": (dy.astype(np.float64).T @ x).astype(np.float32).ravel(),
            "db": dy.astype(np.float64).sum(0).astype(np.float32)}


def case_rows16(d: Path, r: np.random.Generator):
    """A gradient's rows as a product reads them: each divided by the power of two that brings its largest value
    under FP16's reach (dit_rows' rule), so the reference and the kernel pick the same scale."""
    T, C = 192, 160
    y = r.standard_normal((T, C)).astype(np.float32)
    y[:, :8] *= 4000.0                     # the outlier channels that make the scale necessary
    save(d, "y", y)
    rs = np.array([pow2_scale(np.abs(row).max()) for row in y], dtype=np.float32)
    return {"rs": rs, "y16": (y / rs[:, None]).astype(np.float16).astype(np.float32).ravel()}


def case_qkrope(d: Path, r: np.random.Generator):
    """The attention's q and k: each head RMS-normed with its own weights and rotated pairwise.  The reference undoes
    the rotation, then the norm, and sums the norm weights' own gradient over the rows and heads."""
    rows, heads, hd = 64, 3, 128
    half = hd // 2
    q = r.standard_normal((rows, heads, hd)).astype(np.float32)
    k = (r.standard_normal((rows, heads, hd)) * 1.4).astype(np.float32)
    wq = (1.0 + r.standard_normal(hd) * 0.1).astype(np.float32)
    wk = (1.0 + r.standard_normal(hd) * 0.1).astype(np.float32)
    ang = r.standard_normal((rows, half)).astype(np.float32)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    dgq = r.standard_normal((rows, heads, hd)).astype(np.float32)
    dgk = r.standard_normal((rows, heads, hd)).astype(np.float32)
    save(d, "q", q.reshape(rows, heads * hd))
    save(d, "k", k.reshape(rows, heads * hd))
    save(d, "wq", wq)
    save(d, "wk", wk)
    save(d, "cos", cos)
    save(d, "sin", sin)
    save(d, "dgq", dgq.reshape(rows, heads * hd))
    save(d, "dgk", dgk.reshape(rows, heads * hd))
    save(d, "head_dim", np.array([hd], np.float32))
    save(d, "lse", np.array([1.0], np.float32))        # the rope flag: 1 rotates, 0 leaves the norm alone

    def full(x, w, g):   # the two steps are the same for q and k: undo the rotation, then the norm
        rr = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + EPS_LN)
        ga_ = g[..., 0::2] * cos[:, None, :] + g[..., 1::2] * sin[:, None, :]      # even channels
        gb_ = -g[..., 0::2] * sin[:, None, :] + g[..., 1::2] * cos[:, None, :]     # odd channels
        dn = np.empty_like(x)
        dn[..., 0::2] = ga_ * w[None, None, 0::2]
        dn[..., 1::2] = gb_ * w[None, None, 1::2]
        s_ = (dn * x).sum(-1, keepdims=True)
        dw = np.zeros(hd)   # y = x r w turned: the weight's own gradient is the de-rotated one times x and r
        dw[0::2] = (ga_ * x[..., 0::2] * rr).sum((0, 1))
        dw[1::2] = (gb_ * x[..., 1::2] * rr).sum((0, 1))
        return dn * rr - x * (rr**3) * s_ / hd, dw
    dq, dwqn = full(q, wq, dgq)
    dk, dkwn = full(k, wk, dgk)
    return {"dq": dq.reshape(-1), "dk": dk.reshape(-1), "wqn": dwqn, "wkn": dkwn}


def case_lora(d: Path, r: np.random.Generator):
    """One weight's plain LoRA: h = x A^T kept as FP16 rows with their power of two, delta = scale (h B^T).  The
    gradients are read the way train::lora_bwd reads them, from the same FP16 rows the GPU sees."""
    T, rows, cols, rank = 96, 64, 48, 8
    scale = 0.7
    x = r.standard_normal((T, cols)) * 3.0
    xr = np.array([pow2_scale(v.max()) for v in np.abs(x)], np.float32)
    x16 = (x / xr[:, None]).astype(np.float16)
    A = (r.standard_normal((rank, cols)) * 0.2).astype(np.float16)
    B = (r.standard_normal((rows, rank)) * 0.3).astype(np.float16)
    h = x @ A.astype(np.float64).T
    hr = np.array([pow2_scale(v) for v in np.abs(h).max(1)], np.float32)
    h16 = (h / hr[:, None]).astype(np.float16)
    dY = (r.standard_normal((T, rows)) * 1.3).astype(np.float32)
    for n, a in (("dY", dY), ("x16", x16), ("x16t", np.ascontiguousarray(x16.T.astype(np.float16))),
                 ("xr", xr), ("h16", h16), ("h16t", np.ascontiguousarray(h16.T.astype(np.float16))),
                 ("hr", hr), ("b16", B), ("bt16", np.ascontiguousarray(B.T.astype(np.float16))),
                 ("scale", np.array([scale], np.float32))):
        save(d, n, a)
    Bd, hd = B.astype(np.float64), h16.astype(np.float64) * hr[:, None]
    dh = scale * (dY.astype(np.float64) @ Bd)
    return {"dh": dh.astype(np.float32).ravel(),
            "dB": (scale * dY.astype(np.float64).T @ hd).astype(np.float32).ravel(),
            # dA reads dh, which already carries scale (dL/dh = scale * dY B): no second scale here
            "dA": (dh.T @ (x16.astype(np.float64) * xr[:, None])).astype(np.float32).ravel()}


def case_lora_zero_rows(d: Path, r: np.random.Generator):
    """Zero and small gradient rows together: zero rows must not set the transposed product's FP16 scale."""
    case_lora(d, r)
    dy = (r.standard_normal((96, 64)) * 1e-7).astype(np.float32)
    dy[:24] = 0
    save(d, "dY", dy)
    h = load(d, "h16").astype(np.float64) * load(d, "hr").astype(np.float64)[:, None]
    return {"dB": (float(load(d, "scale")[0]) * dy.astype(np.float64).T @ h).astype(np.float32).ravel()}


def case_lora_zero(d: Path, r: np.random.Generator):
    """All gradient rows zero: both backward paths return finite zeros."""
    case_lora(d, r)
    save(d, "dY", np.zeros((96, 64), np.float32))
    return {"dh": np.zeros(96 * 8, np.float32), "dB": np.zeros(64 * 8, np.float32),
            "dA": np.zeros(8 * 48, np.float32)}


def case_lokr(d: Path, r: np.random.Generator):
    """A LoKr's factors: the delta is kron(w1, w2_a w2_b) over a (out_l out_k) x (in_m in_n) weight, so each factor is
    read through the whole delta's gradient by its partial trace, and w2 splits as a product does."""
    ol, ok, im, inn, rank = 3, 4, 2, 6, 5
    dDelta = (r.standard_normal((ol * ok, im * inn)) * 0.4).astype(np.float32)
    w1 = (r.standard_normal((ol, im)) * 0.3).astype(np.float16)
    w2 = (r.standard_normal((ok, inn)) * 0.25).astype(np.float16)
    w2a = (r.standard_normal((ok, rank)) * 0.3).astype(np.float16)
    w2b = (r.standard_normal((rank, inn)) * 0.2).astype(np.float16)
    save(d, "dDelta", dDelta)
    save(d, "w1", w1)
    save(d, "w2", w2)
    save(d, "parts", np.array([ol, ok, im, inn], np.float32))
    save(d, "w2a", w2a)
    save(d, "w2b", w2b)
    save(d, "parts_f", np.array([ok, inn, rank], np.float32))   # the second op reads dw2 and this shape
    G = dDelta.astype(np.float64).reshape(ol, ok, im, inn)
    W1, W2 = w1.astype(np.float64), w2.astype(np.float64)
    dw1 = np.einsum("lkmn,kn->lm", G, W2)
    dw2 = np.einsum("lkmn,lm->kn", G, W1)
    W2A, W2B = w2a.astype(np.float64), w2b.astype(np.float64)
    dw2a = dw2 @ W2B.T
    dw2b = W2A.T @ dw2
    return {"dw1": dw1, "dw2": dw2, "dw2a": dw2a, "dw2b": dw2b}


def case_dora(d: Path, r: np.random.Generator):
    """A DoRA's magnitude, PEFT's definition at strength 1 (tuners/lora/dora.py): y = c (W x) + s c (BA x) with
    c = m / (n + eps_FP32) and n = ||W + s BA|| the composed weight's row norm, detached (no gradient through it).
    So dm = sum_t dY (p + s d) / (n + eps), da = c and db = s c.  The old form (a = 1 + s (c - 1) with n = |W|)
    was the engine's own blend, not PEFT's: a LoRA trained under it read differently in inference."""
    T, rows, scale, eps = 64, 40, 0.8, 1.1920929e-07
    dY = (r.standard_normal((T, rows)) * 1.1).astype(np.float32)
    pp = (r.standard_normal((T, rows)) * 0.9).astype(np.float32)
    dd = (r.standard_normal((T, rows)) * 0.4).astype(np.float32)
    norms = (0.4 + np.abs(r.standard_normal(rows))).astype(np.float32)
    m = (norms * (1.0 + 0.2 * r.standard_normal(rows))).astype(np.float32)
    for n, a in (("dY", dY), ("p", pp), ("d", dd), ("m", m), ("norms", norms),
                 ("scale", np.array([scale], np.float32))):
        save(d, n, a)
    c = m.astype(np.float64) / (norms.astype(np.float64) + eps)
    return {"dm": ((dY.astype(np.float64) * (pp.astype(np.float64) + scale * dd.astype(np.float64))).sum(0)
                   / (norms.astype(np.float64) + eps)).astype(np.float32),
            "da": c.astype(np.float32), "db": (scale * c).astype(np.float32)}


def dora_norms_ref(d: Path, r: np.random.Generator, N: int, K: int, rank: int, scale: float):
    """PEFT's get_weight_norm: the composed weight W + s BA made, its rows' norm directly (FP64 over the FP16
    values the products read).  norms0 is |W| alone, the base term the engine's expansion starts from."""
    w = (r.standard_normal((N, K)) * 0.1).astype(np.float16)
    a = (r.standard_normal((rank, K)) * 0.2).astype(np.float16)
    b = (r.standard_normal((N, rank)) * 0.3).astype(np.float16)
    w0 = w.astype(np.float64)
    norms0 = np.linalg.norm(w0, axis=1).astype(np.float32)
    for n, arr in (("w16", w), ("a16", a), ("b16", b), ("norms0", norms0),
                   ("scale", np.array([scale], np.float32))):
        save(d, n, arr)
    composed = w0 + scale * (b.astype(np.float64) @ a.astype(np.float64))
    return {"norms": np.linalg.norm(composed, axis=1).astype(np.float32)}


def case_doranorm(d: Path, r: np.random.Generator):
    return dora_norms_ref(d, r, 40, 48, 8, 0.8)


def case_doranormreal(d: Path, r: np.random.Generator):
    """The same at Mage-Flow's attention width and a large rank (the scratch's N rank + rank rank)."""
    return dora_norms_ref(d, r, 3072, 3072, 96, 2.0)


def dora_ab_ref(d: Path, r: np.random.Generator, T: int, N: int, K: int, rank: int, scale: float):
    """One DoRA product whole: the forward's rows and the backward's dm, dA, dB and dX, PEFT's definition with the
    norm detached.  The reference reads the same FP16 bits the GPU does (h rounded once, as the engine keeps it)."""
    eps = 1.1920929e-07
    x = r.standard_normal((T, K)) * 3.0
    xr = np.array([pow2_scale(np.abs(row).max()) for row in x], np.float32)
    x16 = (x / xr[:, None]).astype(np.float16)
    w = (r.standard_normal((N, K)) * 0.05).astype(np.float16)
    a = (r.standard_normal((rank, K)) * 0.2).astype(np.float16)
    b = (r.standard_normal((N, rank)) * 0.3).astype(np.float16)
    norms0 = np.linalg.norm(w.astype(np.float64), axis=1).astype(np.float32)
    m = (norms0 * (1.0 + 0.2 * r.standard_normal(N))).astype(np.float32)
    dY = (r.standard_normal((T, N)) * 1.3).astype(np.float32)
    for n, arr in (("x16", x16), ("xr", xr), ("w16", w), ("a16", a), ("b16", b), ("norms0", norms0), ("m", m),
                   ("dY", dY), ("scale", np.array([scale], np.float32))):
        save(d, n, arr)
    X = x16.astype(np.float64) * xr[:, None]
    W, A, B = w.astype(np.float64), a.astype(np.float64), b.astype(np.float64)
    norms = np.linalg.norm(W + scale * (B @ A), axis=1)
    c = m.astype(np.float64) / (norms + eps)
    h = X @ A.T
    hr = np.array([pow2_scale(v) for v in np.abs(h).max(1)], np.float32)
    hd = (h / hr[:, None]).astype(np.float16).astype(np.float64) * hr[:, None]
    p, delta = X @ W.T, hd @ B.T
    dYd = dY.astype(np.float64)
    dYb = dYd * (scale * c)[None, :]
    dh = dYb @ B
    return {"y": (c[None, :] * p + (scale * c)[None, :] * delta).astype(np.float32),
            "dx": ((dYd * c[None, :]) @ W + dh @ A).astype(np.float32),
            "dm": (dYd * (p + scale * delta)).sum(0) / (norms + eps),
            "dA": (dh.T @ X).astype(np.float32), "dB": (dYb.T @ hd).astype(np.float32),
            "norms": norms.astype(np.float32)}


def case_doraab(d: Path, r: np.random.Generator):
    return dora_ab_ref(d, r, 64, 40, 48, 8, 0.8)


def case_doraabreal(d: Path, r: np.random.Generator):
    return dora_ab_ref(d, r, 64, 3072, 3072, 32, 2.0)


def lokr_ab_ref(d: Path, r: np.random.Generator, T: int, ol: int, ok: int, im: int, inn: int, rank: int,
                scale: float):
    """One LoKr product whole: delta = s kron(w1, w2a w2b) with w2b nonzero (the second update's state, where the
    transposed reads bite), the forward's rows and every factor's gradient through the whole delta's gradient."""
    N, K = ol * ok, im * inn
    x = r.standard_normal((T, K)) * 3.0
    xr = np.array([pow2_scale(np.abs(row).max()) for row in x], np.float32)
    x16 = (x / xr[:, None]).astype(np.float16)
    w = (r.standard_normal((N, K)) * 0.05).astype(np.float16)
    w1 = (r.standard_normal((ol, im)) * 0.3).astype(np.float16)
    w2a = (r.standard_normal((ok, rank)) * 0.3).astype(np.float16)
    w2b = (r.standard_normal((rank, inn)) * 0.2).astype(np.float16)
    dY = (r.standard_normal((T, N)) * 1.3).astype(np.float32)
    for n, arr in (("x16", x16), ("xr", xr), ("w16", w), ("w1", w1), ("w2a", w2a), ("w2b", w2b), ("dY", dY),
                   ("parts", np.array([ol, ok, im, inn, rank], np.float32)),
                   ("scale", np.array([scale], np.float32))):
        save(d, n, arr)
    X = x16.astype(np.float64) * xr[:, None]
    W = w.astype(np.float64)
    D = scale * np.kron(w1.astype(np.float64), w2a.astype(np.float64) @ w2b.astype(np.float64))
    dYd = dY.astype(np.float64)
    W1, W2 = w1.astype(np.float64), w2a.astype(np.float64) @ w2b.astype(np.float64)
    G = (scale * (dYd.T @ X)).reshape(ol, ok, im, inn)
    dw2 = np.einsum("lkmn,lm->kn", G, W1)
    return {"y": (X @ (W + D).T).astype(np.float32), "dx": (dYd @ (W + D)).astype(np.float32),
            "dW1": np.einsum("lkmn,kn->lm", G, W2).astype(np.float32),
            "dW2a": (dw2 @ w2b.astype(np.float64).T).astype(np.float32),
            "dW2b": (w2a.astype(np.float64).T @ dw2).astype(np.float32)}


def case_lokrab(d: Path, r: np.random.Generator):
    return lokr_ab_ref(d, r, 8, 3, 4, 2, 6, 5, 0.7)


def case_lokrabreal(d: Path, r: np.random.Generator):
    """Mage-Flow's mlp1 (12288 x 3072) at rank 96: the shared scratch's bound the old sizing crossed at rank 65."""
    return lokr_ab_ref(d, r, 64, 96, 128, 48, 64, 96, 0.5)


def case_lokrabattn(d: Path, r: np.random.Generator):
    """The attention-only target (N = K = 3072): the other shape the old scratch crossed at every rank."""
    return lokr_ab_ref(d, r, 64, 48, 64, 48, 64, 32, 0.5)


CASES = {"lnmod": case_lnmod, "gelu": case_gelu, "silu": case_silu, "gate": case_gate, "patch": case_patch,
         "flow": case_flow, "products": case_products, "productsreal": case_productsreal,
         "attn": case_attn, "attnreal": case_attnreal, "attncausal": case_attncausal,
         "rows16": case_rows16,
         "qkrope": case_qkrope, "lora": case_lora, "lora_zero_rows": case_lora_zero_rows,
         "lora_zero": case_lora_zero, "lokr": case_lokr, "dora": case_dora,
         "doranorm": case_doranorm, "doranormreal": case_doranormreal,
         "doraab": case_doraab, "doraabreal": case_doraabreal,
         "lokrab": case_lokrab, "lokrabreal": case_lokrabreal, "lokrabattn": case_lokrabattn}
# the probe calls of each case: (op, its output files)
RUNS = {"lnmod": [("lnmod", ["dx", "dshift", "dscale"])], "gelu": [("gelu", ["d"])], "silu": [("silu", ["d"])],
        "gate": [("gate", ["dh", "dgate"])], "patch": [("patch", ["dm", "dmean", "dsd"])],
        "flow": [("flow", ["dv", "dx0", "deps", "loss"])],
        "products": [("dgrad", ["dx"]), ("wgrad", ["dw"]), ("bias", ["db"])],
        "productsreal": [("dgrad", ["dx"]), ("wgrad", ["dw"]), ("bias", ["db"])],
        "attn": [("attnf", ["o16", "o32", "lse"]), ("attnb", ["dQ", "dK", "dV"])],
        "attnreal": [("attnf", ["o16", "o32", "lse"]), ("attnb", ["dQ", "dK", "dV"])],
        "attncausal": [("attnf", ["o16", "o32", "lse"]), ("attnb", ["dQ", "dK", "dV"])],
        "rows16": [("rows16", ["rs", "y16"])],
        "qkrope": [("qkrope", ["dq", "dk", "wqn", "wkn"])],
        "lora": [("lora", ["dh", "dB", "dA"])], "lora_zero_rows": [("lora", ["dB"])],
        "lora_zero": [("lora", ["dh", "dB", "dA"])],
        "lokr": [("lokr", ["dw1", "dw2"]), ("lokrf", ["dw2a", "dw2b"])],
        "dora": [("dora", ["dm", "da", "db"])],
        "doranorm": [("doranorm", ["norms"])], "doranormreal": [("doranorm", ["norms"])],
        "doraab": [("doraab", ["y", "dx", "dm", "dA", "dB", "norms"])],
        "doraabreal": [("doraab", ["y", "dx", "dm", "dA", "dB", "norms"])],
        "lokrab": [("lokrab", ["y", "dx", "dW1", "dW2a", "dW2b"])],
        "lokrabreal": [("lokrab", ["y", "dx", "dW1", "dW2a", "dW2b"])],
        "lokrabattn": [("lokrab", ["y", "dx", "dW1", "dW2a", "dW2b"])]}
# the attention is the shared FP16 kernel (kernels::flash_attn, inference's) and its backward the shared FP16
# products: P and dS are rounded to FP16 once each, so its gradients and its FP32 output carry FP16's own error
# (dQ, dK, dV 2.7-3.4e-4 and o32 2.4e-4 measured on the B70, 2026-11-09)
TOL = {"lora": 5e-4, "lora_zero_rows": 5e-4, "lora_zero": 0.0, "lokr": 1e-5, "dora": 1e-6, "qkrope": 2e-4,
       "attn": 5e-4, "attnreal": 5e-4,
       # the causal case takes the FP16 class (1e-3, TOL_LIBRARY's) under every form: the fused backward rounds P
       # and dS to FP16 (flash_attn_bwd.cpp's ph[] = half) and the products form's maps likewise (tile_maps) - the
       # same one effective FP16 rounding as the library's gradients.  The causal rows measure dQ 5.10e-4 / dK
       # 5.09e-4 on the RTX 4070 and dQ 5.10e-4 on the RX 9060 XT (fused, products and library alike, 2026-11-18),
       # above the non-causal empirical 5e-4 (B70 2.7-3.4e-4, RTX 2.8-3.4e-4) but within the FP16 class
       "attncausal": 1e-3,
       "rows16": 1e-3, "lnmod": 2e-5, "gelu": 5e-5, "silu": 5e-5, "gate": 2e-5, "patch": 1e-6, "flow": 1e-5,
       "products": 3e-3, "productsreal": 3e-3,
       # the composed norms: FP16 factors into FP32 products, the base |W| FP32 (the expansion's own rounding)
       "doranorm": 1e-4, "doranormreal": 1e-4,
       # the whole adapter paths: dgrad and wgrad read FP16 gradient rows (the products' own 3e-3)
       "doraab": 3e-3, "doraabreal": 3e-3, "lokrab": 3e-3, "lokrabreal": 3e-3, "lokrabattn": 3e-3}


# an FP16 tensor can only carry FP16's own error: an output written as FP16 is judged at two of its units
TOL_OUT = {("attn", "o16"): 2e-3, ("attn", "o32"): 5e-4, ("attnreal", "o16"): 2e-3, ("attnreal", "o32"): 5e-4,
           ("attncausal", "o16"): 2e-3, ("attncausal", "o32"): 5e-4, ("products", "db"): 1e-3,
           # the DoRA product's FP32 outputs (its FP16 gradient rows' dgrad/wgrad keep the case's 3e-3).  y at the
           # real width carries the kept h's FP16 rounding through the delta (measured 1.3e-4 on the RX 9060 XT)
           **{(n, o): 1e-4 for n in ("doraab", "doraabreal") for o in ("dm", "norms")},
           ("doraab", "y"): 1e-4, ("doraabreal", "y"): 5e-4}
# the attention through the GPU maker's library (oneDNN's, cuDNN's or aotriton's SDPA for training), whose output
# and gradients come out in FP16: one rounding stage at the library's output, 2-6e-4 alone (oneDNN on the B70:
# dQ 6.3e-4, dK 5.1e-4, dV 5.6e-4, o32 4.3e-4 at the real shapes, 2026-11-09; cuDNN on the RTX 4070: attncausal
# dQ 5.10e-4, dK 5.09e-4, 2026-11-18; aotriton on the RX 9060 XT: attnreal dQ 5.24e-4, dK 5.07e-4, dV 5.48e-4,
# 2026-11-18), so they are judged at 1e-3.  The own forms (fused, products) keep FP32 gradients and take the 5e-4
# class (RX 9060 XT fused: dQ 3.1e-4).  Which class applies is decided by the form that actually ran, not by the
# case name: the probe reports it (XESTRATA_ATTN_BWD_REPORT, "attention backward: <form>"), and the auto choice
# takes the library on the RTX 4070 (cuDNN) and the RX 9060 XT (aotriton).  A causal case's text rows fall back to
# the products (FP32) when the fused kernel cannot run, but the image rows' library FP16 dominates the tensor-wide
# relative error, so the whole tensor takes the library class when the library ran.
TOL_LIBRARY = {(n, o): 1e-3 for n in ("attn", "attnreal", "attncausal") for o in ("o32", "dQ", "dK", "dV")}
ATTN_CASES = ("attn", "attnreal", "attncausal")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dir", default="/tmp/mf-train")
    ap.add_argument("--op", action="append", default=[])
    ap.add_argument("--attn", choices=["own", "library"], default="",
                    help="the attention's forms: the engine's own kernels (the forward's, the backward's fused one) or "
                         "the GPU maker's library (empty: whichever the probe chooses)")
    a = ap.parse_args()
    env = dict(os.environ)
    env["XESTRATA_ATTN_BWD_REPORT"] = "1"   # the probe says which backward form ran; the tolerance follows it
    tol_out = dict(TOL_OUT)
    if a.attn == "own":
        env.update(XESTRATA_ATTN_TRAIN_FWD="own", XESTRATA_ATTN_BWD_FORM="fused")
    elif a.attn == "library":
        env.update(XESTRATA_ATTN_TRAIN_FWD="library", XESTRATA_ATTN_BWD_FORM="library")
        tol_out.update(TOL_LIBRARY)
    root = Path(a.dir)
    bad = 0
    for name in (a.op or list(CASES)):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        want = CASES[name](d, np.random.default_rng(20261103))
        for cmd, outs in RUNS[name]:
            modes = ("auto", "own") if name in ("lora_zero_rows", "lora_zero") else ("",)
            for mode in modes:
                run_env = env.copy()
                if mode:
                    run_env["XESTRATA_BLAS"] = "0"
                    run_env["XESTRATA_NO_BLAS"] = "1" if mode == "own" else "0"
                out = subprocess.run([a.probe, cmd, str(d)], capture_output=True, text=True, env=run_env)
                if out.returncode != 0:
                    print(f"{name} {cmd:7} {mode} FAILED to run: {out.stderr.strip()}", flush=True)
                    bad += 1
                    continue
                if mode:
                    used_library = any("f16-batch-ex-tn" in line and "path=oneMath" in line
                                       for line in out.stderr.splitlines())
                    path = "transposed oneMath" if used_library else "own"
                    print(f"{name} {mode}: {path}", flush=True)
                    if mode == "own" and used_library:
                        print(f"{name}: own mode used oneMath: FAIL", flush=True)
                        bad += 1
                # the backward's form as it ran (the auto choice takes the maker's library on some GPUs): its
                # FP16 gradients take TOL_LIBRARY, the own forms' FP32 ones the case's tolerance
                lib_ran = False
                if name in ATTN_CASES and cmd == "attnb":
                    fm = re.search(r"attention backward: (\w+)", out.stderr)
                    lib_ran = fm is not None and fm.group(1) == "library"
                    if fm is not None:
                        print(f"{name:9} {cmd:7} {mode:4} form: {fm.group(1)}", flush=True)
                for o in outs:
                    got = load(d, o)
                    e = rel(got, want[o])
                    tol = tol_out.get((name, o), TOL[name])
                    if lib_ran:
                        tol = TOL_LIBRARY.get((name, o), tol)
                    ok = np.isfinite(got).all() and e <= tol
                    if name == "lora_zero":
                        ok = ok and np.all(got == 0)
                    bad += 0 if ok else 1
                    print(f"{name:9} {cmd:7} {mode:4} {o:7} rel {e:.2e} "
                          f"(tol {tol:.0e}) {'ok' if ok else 'FAIL'}", flush=True)
    print(f"mf_train_ref: {bad} over tolerance")
    return bad


if __name__ == "__main__":
    sys.exit(main())
