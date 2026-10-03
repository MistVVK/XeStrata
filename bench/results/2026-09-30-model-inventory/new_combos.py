"""The non-expert (name role, type) pairs of each inventoried file that the IQ2_XS model does not have.

    python new_combos.py CSV_DIR
"""
import csv, glob, pathlib, re, sys
def role(n):
    return re.sub(r"^blk\.\d+\.", "blk.N.", n)
def combos(rows):
    return {(role(r["name"]), r["type"]) for r in rows if "_exps." not in r["name"]}
base = combos(csv.DictReader(open(str(pathlib.Path(__file__).resolve().parent.parent / "2026-09-30-b70" / "iq2-xs-tensors.csv"))))
base_types = {t for _, t in base}
for f in sorted(glob.glob(sys.argv[1] + "/*.csv")):
    c = combos(csv.DictReader(open(f)))
    new = sorted(c - base)
    newt = sorted({t for _, t in c} - base_types)
    print(f"== {f.split('/')[-1]}: {len(new)} role/type pairs not in IQ2_XS; types not in IQ2_XS: {newt}")
    by = {}
    for r, t in new: by.setdefault(t, []).append(r)
    for t, rs in sorted(by.items()): print(f"   {t}: {sorted(set(rs))[:8]}{' ...' if len(set(rs)) > 8 else ''}")
