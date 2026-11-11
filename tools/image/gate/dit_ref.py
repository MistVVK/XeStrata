"""tools/image/gate/dit_ref.py - TEST-ONLY (OWN-5 gate harness): a numpy reference of one Qwen-Image-2.1 DiT call as
stable-diffusion.cpp 2988060 computes it (src/model/diffusion/qwen_image_2_1.hpp, QwenImage21Runner, no reference
pictures, text prefix + image tokens in one sequence), from the stage-hook dumps of that call: the latent and the
timestep it got (dit_x_<k>, dit_t_<k>) and the condition (cond).  The oracle of step 4 (our own DiT GEMMs and runner).

The model (read from the pinned tree):
  time   = [t, 0] -> sinusoid 256 (cos half, then sin half; freq = exp(-ln 10000 * j / 128)) -> linear_1 -> SiLU ->
           linear_2 -> SiLU;  mod = modulation.1(time) -> 4 chunks m0..m3 of 4096, row 0 for image tokens, row 1 (t = 0)
           for text tokens;  scale = norm_out.linear(time row 0)
  text   = txt_in: RMSNorm(eps 1e-6) x (1 + w) -> in_layer -> GELU (tanh form) -> out_layer;  image = img_in(latent
           tokens, patch 1, token h*W + w);  joint = [text ; image]
  block  = h = LN(x) (1 + m0);  q k v = h W^T, per-head RMSNorm (norm_q / norm_k, eps 1e-6), RoPE (interleaved pairs,
           axes 16 / 56 / 56, theta 10000, positions text (p, p, p), image (L, h - (H - H/2), w - (W - W/2)));
           attention scale 1/sqrt(128): text rows causal over the text, image rows over everything;
           x += to_out(attn) tanh(m1);  h = LN(x) (1 + m2);  gate | up = h Wgu^T;  x += out(up silu(gate)) tanh(m3)
  head   = LN(image rows) (1 + scale) -> proj_out -> velocity [64, H, W]
  (LN: no affine, eps 1e-6; no linear has a bias)

Modes: "f32" (weights dequantized exactly, every GEMM and the attention in float64 rounded to float32) and "f16op"
(step 4's arithmetic: f16 weights and GEMM inputs, exact accumulation, Q / K / V and the probabilities f16).

    python -I -B dit_ref.py <dit.gguf> --dumps D [D ...] [--call 0] [--mode f32,f16op] [--out-dir O]
                            [--cmp NAME=<file with {dump}>]  e.g. --cmp stock={dump}/vel_0.bin
        each D is a stage dump dir with dit_x_<call>, dit_t_<call>, cond; writes O/<dump name>_<mode>.npy"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from gguf_reader import GGUFFile  # noqa: E402
from te_ref import DEQ, compare, f16r  # noqa: E402

PFX = "model.diffusion_model."
AXES = (16, 56, 56)
F32 = np.float32


class DitWeights:
    def __init__(self, path: Path):
        self.g = GGUFFile(path)
        self.t = {t.name: t for t in self.g.tensors}
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.n_layer = 1 + max(int(n[len(PFX + "transformer_blocks."):].split(".")[0]) for n in self.t
                               if n.startswith(PFX + "transformer_blocks."))

    def mat(self, name: str) -> np.ndarray:
        t = self.t[PFX + name]
        a = self.g.data_start + t.offset
        r = np.asarray(self.mm[a:a + t.expected_bytes()])
        ne0 = t.shape[0]
        return DEQ[t.type_name](r).reshape(-1, ne0)


# ------------------------------------------------------------------ positions and RoPE (Rope::embed_nd, QwenImage21Layout)
def layout_positions(text_len: int, H: int, W: int) -> np.ndarray:
    pos = [[float(p)] * 3 for p in range(text_len)]
    for h in range(H):
        for w in range(W):
            pos.append([float(text_len), float(h - (H - H // 2)), float(w - (W - W // 2))])
    return np.asarray(pos, dtype=F32)


def rope_table(pos: np.ndarray, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [n, 64] in the order sd.cpp concatenates the axes (axis 0: 8 pairs, axis 1 and 2: 28 each)."""
    cs, sn = [], []
    for ax, dim in enumerate(AXES):
        half = dim // 2
        start, end = F32(0), F32((dim * 1.0 - 2) / dim)
        step = F32((end - start) / F32(half - 1))
        scale = np.array([F32(start + F32(i) * step) for i in range(half)], dtype=F32)
        omega = (F32(1.0) / np.power(F32(theta), scale).astype(F32)).astype(F32)
        ang = (pos[:, ax:ax + 1].astype(F32) * omega[None, :]).astype(F32)
        cs.append(np.cos(ang).astype(F32))
        sn.append(np.sin(ang).astype(F32))
    return np.concatenate(cs, 1), np.concatenate(sn, 1)


def apply_rope(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """x [n, H, 128]; interleaved pairs (2j, 2j+1): [[cos, -sin], [sin, cos]]."""
    a = x[..., 0::2].astype(np.float64)
    b = x[..., 1::2].astype(np.float64)
    cc, ss = c[:, None, :].astype(np.float64), s[:, None, :].astype(np.float64)
    out = np.empty(x.shape, np.float64)
    out[..., 0::2] = a * cc - b * ss
    out[..., 1::2] = a * ss + b * cc
    return out.astype(F32)


# ------------------------------------------------------------------ pieces
def layer_norm(x):
    x64 = x.astype(np.float64)
    m = x64.mean(-1, keepdims=True)
    v = ((x64 - m) ** 2).mean(-1, keepdims=True)
    return ((x64 - m) / np.sqrt(v + 1e-6)).astype(F32)


def rms_norm(x, w):
    x64 = x.astype(np.float64)
    return ((x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + np.float64(F32(1e-6)))).astype(F32) * w).astype(F32)


def gelu_tanh(x):
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + np.tanh(0.7978845608028654 * x64 * (1.0 + 0.044715 * x64 * x64)))).astype(F32)


def silu(x):
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(F32)


def timestep_embedding(t: np.ndarray, dim: int = 256, max_period: int = 10000) -> np.ndarray:
    half = dim // 2
    j = np.arange(half, dtype=np.float64)
    freq = np.exp(-math.log(max_period) * j / half)
    arg = t[:, None].astype(np.float64) * freq[None, :]
    return np.concatenate([np.cos(arg), np.sin(arg)], 1).astype(F32)


class Mode:
    def __init__(self, f16: bool):
        self.f16 = f16

    def w(self, m: np.ndarray) -> np.ndarray:
        return (f16r(m) if self.f16 else m).astype(np.float64)

    def gemm(self, x: np.ndarray, W64: np.ndarray) -> np.ndarray:
        xi = f16r(x) if self.f16 else x
        return (xi.astype(np.float64) @ W64.T).astype(F32)


def attention(q, k, v, L: int, f16: bool) -> np.ndarray:
    """q k v [n, H, D] (after norms and rope); rows < L (text) causal over the text, the rest over all n."""
    n, H, D = q.shape
    if f16:
        q, k, v = f16r(q), f16r(k), f16r(v)
    out = np.empty((n, H, D), F32)
    scale = 1.0 / math.sqrt(D)
    for h in range(H):
        s = q[:, h].astype(np.float64) @ k[:, h].astype(np.float64).T
        s = (s.astype(F32) * F32(scale)).astype(np.float64) if f16 else s * scale
        s[:L, L:] = -np.inf                                      # text rows see only the text
        s[:L, :L] += np.triu(np.full((L, L), -np.inf), 1)        # and causally
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        if f16:
            p = f16r(p.astype(F32)).astype(np.float64)
        out[:, h] = (p @ v[:, h].astype(np.float64)).astype(F32)
    return out


def prelude(Wt: DitWeights, m: Mode, t: float, cond: np.ndarray, x: np.ndarray):
    tm = timestep_embedding(np.array([t, 0.0], dtype=F32))
    tm = silu(m.gemm(tm, m.w(Wt.mat("time_text_embed.timestep_embedder.linear_1.weight"))))
    tm = silu(m.gemm(tm, m.w(Wt.mat("time_text_embed.timestep_embedder.linear_2.weight"))))
    mod = m.gemm(tm, m.w(Wt.mat("modulation.1.weight")))                  # [2, 4C]
    scale = m.gemm(tm[:1], m.w(Wt.mat("norm_out.linear.weight")))[0]     # [C]
    txt = rms_norm(cond, (Wt.mat("txt_in.text_norm.weight").reshape(-1) + F32(1)).astype(F32))
    txt = gelu_tanh(m.gemm(txt, m.w(Wt.mat("txt_in.in_layer.weight"))))
    txt = m.gemm(txt, m.w(Wt.mat("txt_in.out_layer.weight")))
    C, Hh, Ww = x.shape
    tok = x.reshape(C, Hh * Ww).T.copy()                                  # token h*W + w, channels
    img = m.gemm(tok, m.w(Wt.mat("img_in.weight")))
    return mod, scale, np.concatenate([txt, img], 0)


def block(Wt: DitWeights, m: Mode, i: int, xj: np.ndarray, rows: np.ndarray, mods, L: int, c, s, wcache=None):
    p = f"transformer_blocks.{i}."
    C = xj.shape[1]
    H, D = C // 128, 128
    W = wcache if wcache is not None else {}
    if not W:
        for n in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"):
            W[n] = m.w(Wt.mat(p + n + ".weight"))
        W["nq"] = Wt.mat(p + "attn.norm_q.weight").reshape(-1)
        W["nk"] = Wt.mat(p + "attn.norm_k.weight").reshape(-1)
    m0, m1, m2, m3 = (mods[:, k * C:(k + 1) * C][rows] for k in range(4))   # per token rows
    h = (layer_norm(xj) * (F32(1) + m0)).astype(F32)
    n = xj.shape[0]
    q = rms_norm(m.gemm(h, W["attn.to_q"]).reshape(n, H, D), W["nq"])
    k = rms_norm(m.gemm(h, W["attn.to_k"]).reshape(n, H, D), W["nk"])
    v = m.gemm(h, W["attn.to_v"]).reshape(n, H, D)
    q, k = apply_rope(q, c, s), apply_rope(k, c, s)
    a = attention(q, k, v, L, m.f16).reshape(n, C)
    xj = (xj + (m.gemm(a, W["attn.to_out.0"]) * np.tanh(m1.astype(np.float64)).astype(F32))).astype(F32)
    h = (layer_norm(xj) * (F32(1) + m2)).astype(F32)
    gu = m.gemm(h, W["img_mlp.gate_up"])
    F = gu.shape[1] // 2
    act = (gu[:, F:] * silu(gu[:, :F])).astype(F32)
    xj = (xj + (m.gemm(act, W["img_mlp.out"]) * np.tanh(m3.astype(np.float64)).astype(F32))).astype(F32)
    return xj


def head(Wt: DitWeights, m: Mode, xj: np.ndarray, L: int, scale: np.ndarray, Hh: int, Ww: int) -> np.ndarray:
    img = (layer_norm(xj[L:]) * (F32(1) + scale)).astype(F32)
    out = m.gemm(img, m.w(Wt.mat("proj_out.weight")))                     # [T, 64]
    return out.T.reshape(-1, Hh, Ww).astype(F32)


def load_dump(d: Path, name: str):
    shape = [int(v) for v in (d / f"{name}.shape").read_text().split()]
    a = np.fromfile(d / f"{name}.bin", dtype="<f4")
    return a, shape


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--mode", default="f32,f16op")
    ap.add_argument("--out-dir")
    ap.add_argument("--cmp", action="append", default=[])
    ap.add_argument("--layers", type=int, default=0)
    a = ap.parse_args(argv)
    Wt = DitWeights(Path(a.gguf))
    modes = a.mode.split(",")
    jobs = {}
    for dd in a.dumps:
        d = Path(dd)
        xv, xs = load_dump(d, f"dit_x_{a.call}")
        tv, _ = load_dump(d, f"dit_t_{a.call}")
        cv, cs = load_dump(d, "cond")
        Ww, Hh, C = xs[0], xs[1], xs[2]
        jobs[d] = {"x": xv.reshape(C, Hh, Ww), "t": float(tv[0]), "cond": cv.reshape(cs[1], cs[0]), "H": Hh, "W": Ww}
    state = {}
    t0 = time.time()
    for mname in modes:
        m = Mode(mname == "f16op")
        for d, j in jobs.items():
            mods, scale, xj = prelude(Wt, m, j["t"], j["cond"], j["x"])
            L = j["cond"].shape[0]
            c, s = rope_table(layout_positions(L, j["H"], j["W"]))
            rows = np.array([1] * L + [0] * (xj.shape[0] - L))
            state[(d, mname)] = [mods, scale, xj, L, c, s, rows]
    nl = a.layers or Wt.n_layer
    for i in range(nl):
        for mname in modes:
            m = Mode(mname == "f16op")
            wc = {}
            for d in jobs:
                st = state[(d, mname)]
                st[2] = block(Wt, m, i, st[2], st[6], st[0], st[3], st[4], st[5], wc)
        print(f"block {i} done {time.time() - t0:.0f} s", flush=True)
    res = {}
    out = Path(a.out_dir) if a.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for d, j in jobs.items():
        r = {"t": j["t"], "text_tokens": int(j["cond"].shape[0]), "image_tokens": j["H"] * j["W"]}
        vel = {}
        for mname in modes:
            st = state[(d, mname)]
            vel[mname] = head(Wt, Mode(mname == "f16op"), st[2], st[3], st[1], j["H"], j["W"])
            if out:
                np.save(out / f"{d.name}_{mname}.npy", vel[mname])
        if "f32" in vel and "f16op" in vel:
            r["f16op_vs_f32"] = compare(vel["f16op"].reshape(64, -1).T, vel["f32"].reshape(64, -1).T)
        for spec in a.cmp:
            name, pat = spec.split("=", 1)
            p = Path(pat.replace("{dump}", str(d)).replace("{name}", d.name))
            if not p.is_file():
                continue
            ref = np.fromfile(p, dtype="<f4").reshape(64, -1).T
            for mname in modes:
                r[f"{name}_vs_{mname}"] = compare(ref, vel[mname].reshape(64, -1).T)
        res[d.name] = r
        print(d.name, json.dumps({k: ({kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                      if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    keys = sorted({k for r in res.values() for k, v in r.items() if isinstance(v, dict)})
    for k in keys:
        rows = [r[k] for r in res.values() if k in r]
        print(f"SUMMARY {k}: n {len(rows)} rel_l2 mean {np.mean([x['rel_l2'] for x in rows]):.6f} max "
              f"{max(x['rel_l2'] for x in rows):.6f} cos min {min(x['cos'] for x in rows):.8f}")
    if out:
        (out / "dit_ref.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
