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
(f16 weights and GEMM inputs, exact accumulation, Q / K / V and the probabilities f16) and "f16acc32" (as f16op but
every GEMM sums in float32, as the GPU does; the stage rule's comparison since 11-11, coordinator: on prompt 5 the
exact-sum emulation moves 1.6x from inputs 3e-6 apart; the nine-case table: dit_acc32_cases.py).

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


class ModeAcc32(Mode):
    """"f16acc32": f16 weights and GEMM inputs, FLOAT32 accumulation (numpy float32 matmul, the BLAS's summation
    order), as cuBLAS GemmEx with compute 32F sums; everything else as "f16op"."""
    def __init__(self):
        super().__init__(True)

    def w(self, m: np.ndarray) -> np.ndarray:
        return f16r(m).astype(F32)

    def gemm(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        return (f16r(x).astype(F32) @ W.astype(F32, copy=False).T).astype(F32)


MODES = ("f32", "f16op", "f16acc32")


def make_mode(name: str) -> Mode:
    if name not in MODES:
        raise SystemExit(f"dit_ref: unknown mode {name!r} (known: {', '.join(MODES)})")
    return ModeAcc32() if name == "f16acc32" else Mode(name == "f16op")


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


def block(Wt: DitWeights, m: Mode, i: int, xj: np.ndarray, rows: np.ndarray, mods, L: int, c, s, wcache=None, mx=None,
          cap=None):
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
    qr, kr = m.gemm(h, W["attn.to_q"]), m.gemm(h, W["attn.to_k"])
    q = rms_norm(qr.reshape(n, H, D), W["nq"])
    k = rms_norm(kr.reshape(n, H, D), W["nk"])
    v = m.gemm(h, W["attn.to_v"]).reshape(n, H, D)
    q, k = apply_rope(q, c, s), apply_rope(k, c, s)
    a = attention(q, k, v, L, m.f16).reshape(n, C)
    if mx is not None:                                   # largest |value| of every GEMM / attention input, text and image rows
        for nm, arr in (("qkv_in", h), ("q", q), ("k", k), ("v", v), ("to_out_in", a)):
            mx[f"txt {i} {nm}"] = float(np.abs(arr[:L]).max())
            mx[f"img {i} {nm}"] = float(np.abs(arr[L:]).max())
    xj = (xj + (m.gemm(a, W["attn.to_out.0"]) * np.tanh(m1.astype(np.float64)).astype(F32))).astype(F32)
    h = (layer_norm(xj) * (F32(1) + m2)).astype(F32)
    if cap is not None:                                   # one block in detail (image rows; K / V over all rows)
        cap.update(qkv=np.concatenate([qr, kr, v.reshape(n, C)], 1)[L:], q16=q[L:], k16=k, v16=v, att=a[L:],
                   x_mid=xj[L:].copy(), ln2=h[L:])
    gu = m.gemm(h, W["img_mlp.gate_up"])
    F = gu.shape[1] // 2
    act = (gu[:, F:] * silu(gu[:, :F])).astype(F32)
    if mx is not None:
        for nm, arr in (("gate_up_in", h), ("out_in", act)):
            mx[f"txt {i} {nm}"] = float(np.abs(arr[:L]).max())
            mx[f"img {i} {nm}"] = float(np.abs(arr[L:]).max())
    xj = (xj + (m.gemm(act, W["img_mlp.out"]) * np.tanh(m3.astype(np.float64)).astype(F32))).astype(F32)
    if cap is not None:
        cap.update(act=act[L:])
    return xj


def head(Wt: DitWeights, m: Mode, xj: np.ndarray, L: int, scale: np.ndarray, Hh: int, Ww: int) -> np.ndarray:
    img = (layer_norm(xj[L:]) * (F32(1) + scale)).astype(F32)
    out = m.gemm(img, m.w(Wt.mat("proj_out.weight")))                     # [T, 64]
    return out.T.reshape(-1, Hh, Ww).astype(F32)


TRACE: dict = {}
MAXIMA: dict = {}
STATES: dict = {}
CAPS: dict = {}


def trace_cmp(pat: str, d: Path, mode: str, stage: str, parts: dict, normalise: bool = False) -> None:
    """rel L2 of our run (the trace files) against this mode's value at the same stage, kept in TRACE."""
    tdir = Path(pat.replace("{name}", d.name))
    for part, (ref, fname) in parts.items():
        f = tdir / fname
        if not f.is_file():
            continue
        ours = np.fromfile(f, dtype="<f4").reshape(ref.shape)
        if normalise:
            ours = layer_norm(ours)
        r = compare(ours, ref)
        TRACE.setdefault((d.name, stage, part), {})[mode] = r["rel_l2"]


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
    ap.add_argument("--trace-block", type=int, default=-1, help="with --trace: compare the inside of this block (image rows)")
    ap.add_argument("--trace", help="per-stage comparison with a STRATA_DIT_TRACE_DIR dump; {name} = the dump dir name")
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
        m = make_mode(mname)
        for d, j in jobs.items():
            mods, scale, xj = prelude(Wt, m, j["t"], j["cond"], j["x"])
            L = j["cond"].shape[0]
            c, s = rope_table(layout_positions(L, j["H"], j["W"]))
            rows = np.array([1] * L + [0] * (xj.shape[0] - L))
            state[(d, mname)] = [mods, scale, xj, L, c, s, rows]
            if a.trace:
                trace_cmp(a.trace, d, mname, "prelude", {"mod": (mods, "mod.bin"), "scale": (scale[None, :], "scale.bin"),
                                                         "txt_in": (xj[:L], "txtin.bin"), "img_in": (xj[L:], "imgin.bin")})
    nl = a.layers or Wt.n_layer
    for i in range(nl):
        for mname in modes:
            m = make_mode(mname)
            wc = {}
            for d in jobs:
                st = state[(d, mname)]
                mx = MAXIMA.setdefault((d.name, mname), {}) if a.trace else None
                cap = CAPS.setdefault((d.name, mname), {}) if (a.trace and i == a.trace_block) else None
                st[2] = block(Wt, m, i, st[2], st[6], st[0], st[3], st[4], st[5], wc, mx, cap)
                if a.trace:
                    L = st[3]
                    trace_cmp(a.trace, d, mname, f"block {i:02d}", {"txt": (st[2][:L], f"txt_{i:02d}.bin"),
                                                                      "img": (st[2][L:], f"img_{i:02d}.bin")})
                    # what the next step reads of the stream is LayerNorm(x): the same comparison per token-normalised
                    trace_cmp(a.trace, d, mname, f"block {i:02d}", {"txt LN": (layer_norm(st[2][:L]), f"txt_{i:02d}.bin"),
                                                                      "img LN": (layer_norm(st[2][L:]), f"img_{i:02d}.bin")},
                              normalise=True)
                    STATES[(d.name, mname, i)] = st[2].copy()
        if a.trace and "f32" in modes and "f16op" in modes:      # the emulation against the reference, same stage
            for d in jobs:
                L = state[(d, "f32")][3]
                for part, sl in (("txt", slice(0, L)), ("img", slice(L, None))):
                    TRACE.setdefault((d.name, f"block {i:02d}", part), {})["emu_vs_f32"] = \
                        compare(state[(d, "f16op")][2][sl], state[(d, "f32")][2][sl])["rel_l2"]
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
            vel[mname] = head(Wt, make_mode(mname), st[2], st[3], st[1], j["H"], j["W"])
            if a.trace:
                trace_cmp(a.trace, d, mname, "head", {"velocity": (vel[mname].reshape(64, -1).T, "out.bin")})
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
    if a.trace:
        for dn in sorted({k[0] for k in TRACE}):
            print(f"TRACE {dn}: stage | part | ours vs f32 | ours vs f16op | f16op vs f32")
            for (n, stage, part), v in TRACE.items():
                if n == dn:
                    f = lambda key: f"{v[key]:.6f}" if key in v else "-"   # noqa: E731
                    print(f"TRACE {dn} | {stage} | {part} | {f('f32')} | {f('f16op')} | {f('emu_vs_f32')}")
        # alignment check: our block-i dump against the reference after blocks i-1, i, i+1 (f32 mode, image rows)
        for dn in sorted({k[0] for k in STATES}):
            tdir = Path(a.trace.replace("{name}", dn))
            for i in (0, 1, 2, 5, 15, 30):
                f = tdir / f"img_{i:02d}.bin"
                if not f.is_file() or (dn, "f32", i) not in STATES:
                    continue
                L = next(st[3] for (dd, mm), st in state.items() if dd.name == dn and mm == "f32")
                ours = np.fromfile(f, dtype="<f4").reshape(-1, STATES[(dn, "f32", i)].shape[1])
                cells = []
                for j in (i - 1, i, i + 1):
                    if (dn, "f32", j) in STATES:
                        cells.append(f"ref after block {j}: {compare(ours, STATES[(dn, 'f32', j)][L:])['rel_l2']:.5f}")
                print(f"ALIGN {dn} ours img after block {i}: " + "; ".join(cells))
        for (dn, mname), cp in sorted(CAPS.items()):
            if mname != "f32" or not cp:
                continue
            tdir = Path(a.trace.replace("{name}", dn))
            emu = CAPS.get((dn, "f16op"), {})
            n = cp["att"].shape[0]
            Hh, D = cp["q16"].shape[1], cp["q16"].shape[2]
            def ours(name, shape, dt="<f4"):
                f = tdir / name
                return np.fromfile(f, dtype=dt).astype(np.float32).reshape(shape) if f.is_file() else None
            q16 = ours("blk_q16.bin", (Hh, n, D), "<f2")
            kvt = cp["k16"].shape[0]
            got = {"qkv": ours("blk_qkv.bin", (n, -1)), "q16": None if q16 is None else q16.transpose(1, 0, 2),
                   "k16": ours("blk_k16.bin", (Hh, kvt, D), "<f2"), "v16": ours("blk_v16.bin", (Hh, kvt, D), "<f2"),
                   "att": ours("blk_att.bin", (n, -1)), "x_mid": ours("blk_x_mid.bin", (n, -1)), "ln2": ours("blk_ln2.bin", (n, -1)),
                   "act": ours("blk_act.bin", (n, -1))}
            for key in ("k16", "v16"):
                if got[key] is not None:
                    got[key] = got[key].transpose(1, 0, 2)
            print(f"BLOCK {a.trace_block} {dn} | stage | ours vs f32 | ours vs f16op | f16op vs f32")
            for key in ("qkv", "q16", "k16", "v16", "att", "x_mid", "ln2", "act"):
                g = got[key]
                if g is None:
                    continue
                r = cp[key].reshape(g.shape[0], -1)
                gg = g.reshape(g.shape[0], -1)
                e = emu[key].reshape(g.shape[0], -1) if key in emu else None
                f = lambda x, y: "%.6f" % compare(x, y)["rel_l2"]   # noqa: E731
                ge = f(gg, e) if e is not None else "-"
                er = f(e, r) if e is not None else "-"
                print(f"BLOCK {a.trace_block} {dn} | {key} | {f(gg, r)} | {ge} | {er}")
        for (dn, mname), mx in MAXIMA.items():
            if mname == "f32":
                print(f"MAXIMA {dn} f32 " + json.dumps({k: round(x, 4) for k, x in mx.items()}))
        if out:
            (out / "trace.json").write_text(json.dumps({"trace": {" | ".join(k): v for k, v in TRACE.items()},
                                                        "maxima": {" | ".join(k): v for k, v in MAXIMA.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
