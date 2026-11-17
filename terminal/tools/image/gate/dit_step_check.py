"""tools/image/gate/dit_step_check.py - TEST-ONLY (OWN-5B D4): strata-dit-step's velocities against the reference.

For each case: velocity rel L2 to the f32 reference (dit_ref.py) for ours (strata-dit-step), the emulation with
float32 sums (dit_ref mode f16acc32, the stage rule's comparison since 11-11) and with exact sums (f16op), the D1+D2
test build (sd.cpp's native runner with our GEMMs), stock sd.cpp and its native runner; and ours against the D1+D2
build directly.  Then the stage rule over the cases given: (i) mean ours <= 1.25 x mean emulation, (ii) no case above
1.5 x its emulation value, (iii) every case closer than stock and native - against both emulations.

    python -I -B dit_step_check.py --root R --ours O [--cases "512 p0,512 p5"]
R = the Spark own5b folder (pics/, d0ref/ with d0ref/acc32 from dit_acc32_cases.py); O = strata-dit-step's out dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from te_ref import compare  # noqa: E402


def vel(path) -> np.ndarray:
    """a velocity file ([C][H][W] float32, sd.cpp's vel_<k>.bin layout) as [H*W][C] rows"""
    return np.fromfile(path, dtype="<f4").reshape(64, -1).T


def case_paths(root: Path, name: str) -> dict:
    size, p = name.split()
    p = int(p[1:])
    dn = f"dump_p{p}_s42"
    if size == "1024":
        return {"dump": root / "pics/d0_stock_1024" / dn, "ref": root / "d0ref/r1024" / dn,
                "acc32": root / "d0ref/acc32/1024_p0_acc32.npy", "stock": root / "pics/d0_stock_1024" / dn / "vel_0.bin",
                "native": root / "pics/d0_native_1024" / dn / "vel_0.bin",
                "d1d2": root / "pics/z_ours_1024" / dn / "vel_0.bin"}
    lo = "d0" if p < 4 else "z"
    return {"dump": root / f"pics/{lo}_stock_512" / dn, "ref": root / "d0ref" / dn,
            "acc32": root / f"d0ref/acc32/512_p{p}_acc32.npy", "stock": root / f"pics/{lo}_stock_512" / dn / "vel_0.bin",
            "native": root / f"pics/{lo}_native_512" / dn / "vel_0.bin",
            "d1d2": root / ("pics/d2_g15" if p < 4 else "pics/z_ours_512") / dn / "vel_0.bin"}


def rule(rows: list, k: int) -> str:
    O = np.mean([r["ours"] for r in rows])
    E = np.mean([r[k] for r in rows])
    worst = max(rows, key=lambda r: r["ours"] / r[k])
    ii = all(r["ours"] <= 1.5 * r[k] for r in rows)
    iii = all(r["ours"] < r["stock"] and r["ours"] < r["native"] for r in rows)
    return (f"(i) mean {O:.5f} / {E:.5f} = {O / E:.3f} ({'PASS' if O <= 1.25 * E else 'FAIL'}); (ii) largest "
            f"{worst['ours'] / worst[k]:.3f} at {worst['name']} ({'PASS' if ii else 'FAIL'}); (iii) "
            f"{'PASS' if iii else 'FAIL'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--cases", default="512 p0,512 p5")
    a = ap.parse_args(argv)
    root, ours_dir = Path(a.root), Path(a.ours)
    rows = []
    for name in [c.strip() for c in a.cases.split(",") if c.strip()]:
        P = case_paths(root, name)
        f32 = np.load(str(P["ref"]) + "_f32.npy").reshape(64, -1).T
        o = vel(ours_dir / P["dump"].parent.name / P["dump"].name / "vel.bin")
        r = {"name": name, "ours": compare(o, f32)["rel_l2"],
             "f16acc32": compare(np.load(P["acc32"]).reshape(64, -1).T, f32)["rel_l2"],
             "f16op": compare(np.load(str(P["ref"]) + "_f16op.npy").reshape(64, -1).T, f32)["rel_l2"],
             "d1d2": compare(vel(P["d1d2"]), f32)["rel_l2"], "stock": compare(vel(P["stock"]), f32)["rel_l2"],
             "native": compare(vel(P["native"]), f32)["rel_l2"], "ours_vs_d1d2": compare(o, vel(P["d1d2"]))["rel_l2"]}
        rows.append(r)
    print("| case | ours (strata-dit-step) | f32-sum emulation | ratio | exact-sum emulation | ratio | D1+D2 test build "
          "| stock | native | ours vs D1+D2 build |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['ours']:.5f} | {r['f16acc32']:.5f} | {r['ours'] / r['f16acc32']:.3f} | "
              f"{r['f16op']:.5f} | {r['ours'] / r['f16op']:.3f} | {r['d1d2']:.5f} | {r['stock']:.5f} | "
              f"{r['native']:.5f} | {r['ours_vs_d1d2']:.5f} |")
    print("RULE vs the float32-sum emulation: " + rule(rows, "f16acc32"))
    print("RULE vs the exact-sum emulation: " + rule(rows, "f16op"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
