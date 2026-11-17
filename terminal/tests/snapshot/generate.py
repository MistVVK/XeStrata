"""Write the snapshot of every (machine class, model set) as golden/<machine>/<set>.json, or print a table.

  python -m tests.snapshot.generate --out tests/snapshot/golden            (rewrite the golden files; the only way they change)
  python -m tests.snapshot.generate --out DIR --machine nv_resident --set iq3_s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .driver import drive
from .matrix import ENTRY_POINTS, MACHINES, MODEL_SETS

GOLDEN = Path(__file__).resolve().parent / "golden"


def cell(machine_key: str, set_key: str, entries=ENTRY_POINTS) -> dict:
    return {"machine": machine_key, "set": set_key, "entries": {e: drive(machine_key, set_key, e) for e in entries}}


def dump(data: dict) -> str:
    return json.dumps(data, indent=1, sort_keys=False, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GOLDEN))
    ap.add_argument("--machine", action="append")
    ap.add_argument("--set", action="append")
    a = ap.parse_args(argv)
    out = Path(a.out)
    for m in MACHINES:
        if a.machine and m.key not in a.machine:
            continue
        for s in MODEL_SETS:
            if a.set and s.key not in a.set:
                continue
            path = out / m.key / f"{s.key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump(cell(m.key, s.key)), encoding="utf-8", newline="\n")
            print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
