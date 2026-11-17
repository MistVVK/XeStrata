"""tools/image/gate/stage_cmp.py - TEST-ONLY (OWN-5 gate harness): compare stage dumps of two runs (run_set.py --dump).

    python -I -B stage_cmp.py <pics root> <tagA> <tagB> <stage> [<stage> ...]     stages: noise cond vel_0 ... latent

Per stage over all pictures both runs dumped: identical bits, max |a-b|, relative L2 ||a-b|| / ||b||, cosine of the
whole tensor; for cond also the lowest per-token cosine (rows of the text-encoder output)."""
import json
import sys
from pathlib import Path

import numpy as np


def load(d: Path, stage: str):
    b, s = d / f"{stage}.bin", d / f"{stage}.shape"
    if not b.is_file():
        return None
    a = np.fromfile(b, dtype="<f4").astype(np.float64)
    shape = [int(x) for x in s.read_text().split()]
    return a, shape


root, ta, tb = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out = {}
for stage in sys.argv[4:]:
    rows = []
    for da in sorted((root / ta).glob("dump_p*_s*")):
        db = root / tb / da.name
        A, B = load(da, stage), load(db, stage)
        if A is None or B is None:
            continue
        (a, sa), (b, _) = A, B
        if a.shape != b.shape:
            rows.append({"pic": da.name, "error": "shape"})
            continue
        r = {"pic": da.name, "identical": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a - b).max()),
             "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
             "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))}
        if stage == "cond" and len(sa) >= 2:
            dim = sa[0]                       # sd.cpp tensors are [ne0 = hidden, ne1 = tokens, ...]
            ma, mb = a.reshape(-1, dim), b.reshape(-1, dim)
            c = (ma * mb).sum(1) / np.maximum(np.linalg.norm(ma, axis=1) * np.linalg.norm(mb, axis=1), 1e-30)
            r["token_cos_min"] = float(c.min())
        rows.append(r)
    ok = [r for r in rows if "error" not in r]
    summ = {"n": len(rows), "identical": sum(r["identical"] for r in ok)}
    if ok:
        summ.update(max_abs=max(r["max_abs"] for r in ok), rel_l2_max=max(r["rel_l2"] for r in ok),
                    rel_l2_mean=float(np.mean([r["rel_l2"] for r in ok])), cos_min=min(r["cos"] for r in ok))
        if stage == "cond":
            summ["token_cos_min"] = min(r.get("token_cos_min", 1.0) for r in ok)
    out[stage] = {"summary": summ, "pictures": rows}
    print(f"{ta} vs {tb} [{stage}]: {json.dumps(summ)}")
(root / "cmp").mkdir(exist_ok=True)
(root / "cmp" / f"stage_{ta}__vs__{tb}.json").write_text(json.dumps(out, indent=1))
