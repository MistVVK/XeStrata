"""tools/image/gate/dit_replay.py - TEST-ONLY (OWN-5B trace): replay one DiT block of dit_ref.py from OUR stream.

Our test build (STRATA_DIT_TRACE_DIR, STRATA_DIT_TRACE_BLOCK=B) dumps the stream after every block (txt_NN.bin,
img_NN.bin) and the inside of block B (blk_*.bin).  This tool:
  1. runs dit_ref's block B in f32 and in f16op starting from OUR stream after block B-1, and compares each stage with
     our dump of the same stage: same input, so what is left is the difference in the block's own arithmetic;
  2. runs the reference up to block B-1 (f32 and f16op), then block B in f32 from three inputs: the f32 state, the
     f16op state and our state; the attention output's distance from the f32-from-f32 one against the input's
     distance tells how much block B amplifies a given input difference, per input;
  3. the same with random input perturbations of the size of ours (three trials), to see whether block B of this
     prompt amplifies any small difference or only ours;
  4. a K / V layout check (rows of our blk_k16 / blk_v16 against the reference's text and image rows).

    python -I -B dit_replay.py <dit.gguf> --dump D --trace T --block B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from dit_ref import DitWeights, Mode, block, layer_norm, layout_positions, load_dump, prelude, rope_table  # noqa: E402
from te_ref import compare  # noqa: E402

F32 = np.float32
KEYS = ("qkv", "q16", "att", "x_mid", "ln2", "act")


def rel(a, b) -> float:
    return compare(np.asarray(a, F32).reshape(a.shape[0], -1), np.asarray(b, F32).reshape(b.shape[0], -1))["rel_l2"]


def ours_block(t: Path, n: int, H: int, D: int) -> dict:
    def rd(name, shape, dt="<f4"):
        f = t / name
        return np.fromfile(f, dtype=dt).astype(F32).reshape(shape) if f.is_file() else None
    q = rd("blk_q16.bin", (H, n, D), "<f2")
    return {"qkv": rd("blk_qkv.bin", (n, -1)), "q16": None if q is None else q.transpose(1, 0, 2),
            "att": rd("blk_att.bin", (n, -1)), "x_mid": rd("blk_x_mid.bin", (n, -1)), "ln2": rd("blk_ln2.bin", (n, -1)),
            "act": rd("blk_act.bin", (n, -1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args(argv)
    B = a.block
    d, t = Path(a.dump), Path(a.trace)
    name = d.name
    Wt = DitWeights(Path(a.gguf))
    xv, xs = load_dump(d, "dit_x_0")
    tv, _ = load_dump(d, "dit_t_0")
    cv, cs = load_dump(d, "cond")
    Ww, Hh, C = xs[0], xs[1], xs[2]
    x, tt, cond = xv.reshape(C, Hh, Ww), float(tv[0]), cv.reshape(cs[1], cs[0])
    L = cond.shape[0]
    c, s = rope_table(layout_positions(L, Hh, Ww))
    ours_prev = np.concatenate([np.fromfile(t / f"txt_{B - 1:02d}.bin", dtype="<f4").reshape(L, -1),
                                np.fromfile(t / f"img_{B - 1:02d}.bin", dtype="<f4").reshape(Hh * Ww, -1)], 0)
    n = Hh * Ww
    H, D = ours_prev.shape[1] // 128, 128
    got = ours_block(t, n, H, D)
    st = {}
    for mname in ("f32", "f16op"):
        m = Mode(mname == "f16op")
        mods, scale, xj = prelude(Wt, m, tt, cond, x)
        rows = np.array([1] * L + [0] * (xj.shape[0] - L))
        for i in range(B):
            xj = block(Wt, m, i, xj, rows, mods, L, c, s)
        st[mname] = (m, mods, rows, xj)
        print(f"{name}: {mname} state after block {B - 1} done", flush=True)
    for nm, (m, mods, rows, xj) in st.items():
        print(f"INPUT {name} | {nm} state vs ours after block {B - 1} | img {rel(xj[L:], ours_prev[L:]):.6f} | "
              f"img LN {rel(layer_norm(xj[L:]), layer_norm(ours_prev[L:])):.6f} | txt {rel(xj[:L], ours_prev[:L]):.6f}")
    # 1. same input (ours), the block's arithmetic only
    caps = {}
    for nm, (m, mods, rows, _) in st.items():
        cap = {}
        block(Wt, m, B, ours_prev.copy(), rows, mods, L, c, s, None, None, cap)
        caps[nm] = cap
    print(f"REPLAY {name} block {B} from OUR input | stage | ours vs f32 | ours vs f16op | f16op vs f32")
    for k in KEYS:
        g = got[k]
        if g is None:
            continue
        print(f"REPLAY {name} | {k} | {rel(g, caps['f32'][k]):.6f} | {rel(g, caps['f16op'][k]):.6f} | "
              f"{rel(caps['f16op'][k], caps['f32'][k]):.6f}")
    # 2. block B in f32 from three inputs
    m32, mods32, rows32, x32 = st["f32"]

    def att_from(xin):
        cap = {}
        block(Wt, m32, B, xin.copy(), rows32, mods32, L, c, s, None, None, cap)
        return cap

    base = att_from(x32)
    print(f"AMPLIFY {name} block {B} (f32 arithmetic) | input | input img LN dist | att dist | act dist | att / input")
    for nm, xin in (("f16op state", st["f16op"][3]), ("our state", ours_prev)):
        cp = att_from(xin)
        di = rel(layer_norm(xin[L:]), layer_norm(x32[L:]))
        da, dact = rel(cp["att"], base["att"]), rel(cp["act"], base["act"])
        print(f"AMPLIFY {name} | {nm} | {di:.6f} | {da:.6f} | {dact:.6f} | {da / max(di, 1e-30):.2f}")
    # 3. random perturbations of our input's size (image rows only, per-row relative)
    di_ours = rel(ours_prev[L:], x32[L:])
    rng = np.random.default_rng(1)
    for k in range(a.trials):
        e = rng.standard_normal(x32[L:].shape).astype(F32)
        e *= (np.linalg.norm(x32[L:], axis=1, keepdims=True) / np.linalg.norm(e, axis=1, keepdims=True)) * F32(di_ours)
        xin = x32.copy()
        xin[L:] += e
        cp = att_from(xin)
        di = rel(layer_norm(xin[L:]), layer_norm(x32[L:]))
        da, dact = rel(cp["att"], base["att"]), rel(cp["act"], base["act"])
        print(f"AMPLIFY {name} | random {k} | {di:.6f} | {da:.6f} | {dact:.6f} | {da / max(di, 1e-30):.2f}")
    # 4. K / V layout of our dump
    for key, f in (("k16", "blk_k16.bin"), ("v16", "blk_v16.bin")):
        p = t / f
        if not p.is_file():
            continue
        arr = np.fromfile(p, dtype="<f2").astype(F32)
        kvt = arr.size // (H * D)
        o = arr.reshape(H, kvt, D).transpose(1, 0, 2)
        r = caps["f16op"][key]                                   # [L + n, H, D]
        cells = [f"kvt {kvt} (L {L} + n {n} = {L + n})"]
        if kvt >= L + n:
            cells.append(f"[text;image] {rel(o[:L + n], r):.6f}")
            cells.append(f"text rows {rel(o[:L], r[:L]):.6f} image rows {rel(o[L:L + n], r[L:]):.6f}")
            cells.append(f"[image;text] {rel(np.concatenate([o[n:n + L], o[:n]], 0), r):.6f}")
        print(f"LAYOUT {name} | {key} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
