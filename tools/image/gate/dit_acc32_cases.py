"""tools/image/gate/dit_acc32_cases.py - TEST-ONLY (OWN-5B): the stage check's nine cases against the emulation with
float32 accumulation (dit_ref mode "f16acc32", ModeAcc32: f16 weights and GEMM inputs, float32 sums as cuBLAS GemmEx compute 32F
does) next to the exact-accumulation emulation (dit_ref f16op) it replaces.

    python -I -B dit_acc32_cases.py <dit.gguf> --root R --out O
R = the Spark own5b folder (pics/, d0ref/); writes O/<case>_acc32.npy and prints the table and the rule lines.
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
from dit_ref import DitWeights, ModeAcc32, block, head, layout_positions, load_dump, prelude, rope_table  # noqa: E402
from te_ref import compare  # noqa: E402


def vel(path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(64, -1).T


def run_acc32(Wt, d: Path) -> np.ndarray:
    xv, xs = load_dump(d, "dit_x_0")
    tv, _ = load_dump(d, "dit_t_0")
    cv, cs = load_dump(d, "cond")
    Ww, Hh, C = xs[0], xs[1], xs[2]
    x, tt, cond = xv.reshape(C, Hh, Ww), float(tv[0]), cv.reshape(cs[1], cs[0])
    L, n = cond.shape[0], Hh * Ww
    c, s = rope_table(layout_positions(L, Hh, Ww))
    m = ModeAcc32()
    mods, scale, xj = prelude(Wt, m, tt, cond, x)
    rows = np.array([1] * L + [0] * n)
    for i in range(Wt.n_layer):
        xj = block(Wt, m, i, xj, rows, mods, L, c, s)
    return head(Wt, m, xj, L, scale, Hh, Ww)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Wt = DitWeights(Path(a.gguf))
    cases = []
    for p in range(8):
        dn = f"dump_p{p}_s42"
        lo = "d0" if p < 4 else "z"
        ours = R / ("pics/d2_g15" if p < 4 else "pics/z_ours_512") / dn / "vel_0.bin"
        cases.append((f"512 p{p}", R / f"pics/{lo}_stock_512" / dn, R / "d0ref" / dn,
                      R / f"pics/{lo}_stock_512" / dn / "vel_0.bin", R / f"pics/{lo}_native_512" / dn / "vel_0.bin", ours))
    dn = "dump_p0_s42"
    cases.append(("1024 p0", R / "pics/d0_stock_1024" / dn, R / "d0ref/r1024" / dn, R / "pics/d0_stock_1024" / dn / "vel_0.bin",
                  R / "pics/d0_native_1024" / dn / "vel_0.bin", R / "pics/z_ours_1024" / dn / "vel_0.bin"))
    t0 = time.time()
    rows = []
    for name, dump, ref, st, nat, ours in cases:
        f32 = np.load(str(ref) + "_f32.npy").reshape(64, -1).T
        exact = compare(np.load(str(ref) + "_f16op.npy").reshape(64, -1).T, f32)["rel_l2"]
        npy = out / (name.replace(" ", "_") + "_acc32.npy")
        if npy.is_file():
            v = np.load(npy)
        else:
            v = run_acc32(Wt, dump)
            np.save(npy, v)
        acc = compare(v.reshape(64, -1).T, f32)["rel_l2"]
        o = compare(vel(ours), f32)["rel_l2"]
        s = compare(vel(st), f32)["rel_l2"]
        nv = compare(vel(nat), f32)["rel_l2"]
        oa = compare(vel(ours), v.reshape(64, -1).T)["rel_l2"]
        rows.append((name, o, exact, acc, s, nv, oa))
        print(f"case {name} done {time.time() - t0:.0f} s", flush=True)
    print("| case | ours | emulation, exact sums (before) | ratio | emulation, f32 sums (corrected) | ratio | stock | native | ours vs corrected emulation |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, o, e, ac, s, nv, oa in rows:
        print(f"| {name} | {o:.5f} | {e:.5f} | {o / e:.3f} | {ac:.5f} | {o / ac:.3f} | {s:.5f} | {nv:.5f} | {oa:.5f} |")
    O = np.mean([r[1] for r in rows])
    for label, k in (("exact sums", 2), ("f32 sums", 3)):
        E = np.mean([r[k] for r in rows])
        worst = max(rows, key=lambda r: r[1] / r[k])
        print(f"RULE vs emulation with {label}: (i) mean ours {O:.5f} / emulation {E:.5f} = {O / E:.3f} "
              f"({'PASS' if O <= 1.25 * E else 'FAIL'}); (ii) largest ratio {worst[1] / worst[k]:.3f} at {worst[0]} "
              f"({'PASS' if all(r[1] <= 1.5 * r[k] for r in rows) else 'FAIL'}); (iii) "
              f"{'PASS' if all(r[1] < r[4] and r[1] < r[5] for r in rows) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
