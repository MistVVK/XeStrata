"""tools/image/gate/dit_replay2.py - TEST-ONLY (OWN-5B trace, step 2): where does our stream pick up what makes a prompt
far from the reference?

  --from K [K ...]: start the f16op emulation from OUR stream after block K (txt_KK.bin + img_KK.bin of a
      STRATA_DIT_TRACE_DIR dump; K = -1 = our prelude outputs txtin.bin + imgin.bin), run the rest of the blocks and the
      head in f16op, and give the velocity's rel L2 to the f32 reference velocity (dit_ref --out-dir npy).  The first K
      where this jumps from the emulation's own value to ours is where our arithmetic put the difference in.
  --acc32: also run the whole call with f16 inputs and FLOAT32 accumulation in every GEMM (numpy float32 matmul,
      OpenBLAS summation order), i.e. the emulation with the accumulation our GPU really does, and give its distance.

    python -I -B dit_replay2.py <dit.gguf> --dump D --trace T --ref R.npy [--from K ...] [--acc32]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from dit_ref import DitWeights, Mode, ModeAcc32, block, head, layout_positions, load_dump, prelude, rope_table  # noqa: E402
from te_ref import compare  # noqa: E402

F32 = np.float32


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ref", required=True, help="the f32 reference velocity (.npy, [64, H, W])")
    ap.add_argument("--from", dest="starts", type=int, nargs="*", default=[])
    ap.add_argument("--acc32", action="store_true")
    a = ap.parse_args(argv)
    d, t = Path(a.dump), Path(a.trace)
    name = d.name
    Wt = DitWeights(Path(a.gguf))
    xv, xs = load_dump(d, "dit_x_0")
    tv, _ = load_dump(d, "dit_t_0")
    cv, cs = load_dump(d, "cond")
    Ww, Hh, C = xs[0], xs[1], xs[2]
    x, tt, cond = xv.reshape(C, Hh, Ww), float(tv[0]), cv.reshape(cs[1], cs[0])
    L, n = cond.shape[0], Hh * Ww
    c, s = rope_table(layout_positions(L, Hh, Ww))
    ref = np.load(a.ref).reshape(64, -1).T
    ours_v = np.fromfile(t / "out.bin", dtype="<f4").reshape(-1, 64)
    print(f"RESTART {name} | ours (no restart) | {compare(ours_v, ref)['rel_l2']:.6f}", flush=True)
    nl = Wt.n_layer
    m = Mode(True)
    mods, scale, _ = prelude(Wt, m, tt, cond, x)
    rows = np.array([1] * L + [0] * n)
    t0 = time.time()
    for k in a.starts:
        if k < 0:
            tx, im = "txtin.bin", "imgin.bin"
        else:
            tx, im = f"txt_{k:02d}.bin", f"img_{k:02d}.bin"
        xj = np.concatenate([np.fromfile(t / tx, dtype="<f4").reshape(L, -1),
                             np.fromfile(t / im, dtype="<f4").reshape(n, -1)], 0)
        for i in range(k + 1, nl):
            xj = block(Wt, m, i, xj, rows, mods, L, c, s)
        v = head(Wt, m, xj, L, scale, Hh, Ww).reshape(64, -1).T
        print(f"RESTART {name} | f16op from our stream after {'the prelude' if k < 0 else 'block %d' % k} | "
              f"{compare(v, ref)['rel_l2']:.6f} | {compare(v, ours_v)['rel_l2']:.6f} to ours | {time.time() - t0:.0f} s",
              flush=True)
    if a.acc32:
        m = ModeAcc32()
        mods, scale, xj = prelude(Wt, m, tt, cond, x)
        for i in range(nl):
            xj = block(Wt, m, i, xj, rows, mods, L, c, s)
        v = head(Wt, m, xj, L, scale, Hh, Ww).reshape(64, -1).T
        print(f"ACC32 {name} | f16 inputs, float32 accumulation, whole call | {compare(v, ref)['rel_l2']:.6f} | "
              f"{compare(v, ours_v)['rel_l2']:.6f} to ours | {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
