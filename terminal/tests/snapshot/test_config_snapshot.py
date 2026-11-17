"""The config snapshot test: for every machine class x model set x entry point, what the engine would be started with must equal the golden file.

  python -m pytest tests/snapshot -q                          (or: python -m unittest tests.snapshot.test_config_snapshot)
  STRATA_SNAPSHOT_UPDATE=1 python -m pytest tests/snapshot    (rewrite the golden files after an intended change; the diff of golden/ is the review)

No GPU, no model, no download, no network and no real user folder is needed (see tests/snapshot/README.md).  The app entry points run the host's own code in Node,
so `node` (24 or newer) must be installed; STRATA_SNAPSHOT_SKIP_APP=1 skips those two entries by name and the test says so.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive  # noqa: E402
from tests.snapshot.generate import GOLDEN, dump  # noqa: E402
from tests.snapshot.matrix import ENTRY_POINTS, MACHINES, MODEL_SETS  # noqa: E402

UPDATE = os.environ.get("STRATA_SNAPSHOT_UPDATE") == "1"
SKIP_APP = os.environ.get("STRATA_SNAPSHOT_SKIP_APP") == "1"
APP_ENTRIES = ("app_plain", "app_models")


def readable_diff(want: dict, got: dict, label: str) -> str:
    a = json.dumps(want, indent=1, ensure_ascii=False).splitlines()
    b = json.dumps(got, indent=1, ensure_ascii=False).splitlines()
    lines = list(difflib.unified_diff(a, b, "golden " + label, "now " + label, lineterm="", n=2))
    head = [f"changed top-level fields: {sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))}"]
    return "\n".join(head + lines[:160] + (["... (%d more diff lines)" % (len(lines) - 160)] if len(lines) > 160 else []))


class ConfigSnapshot(unittest.TestCase):
    maxDiff = None

    def check_cell(self, machine_key: str, set_key: str) -> None:
        path = GOLDEN / machine_key / f"{set_key}.json"
        want = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        got_entries = {}
        for entry in ENTRY_POINTS:
            if entry in APP_ENTRIES and SKIP_APP:
                print(f"SKIPPED (STRATA_SNAPSHOT_SKIP_APP=1): {machine_key}/{set_key}/{entry}")
                continue
            if entry in APP_ENTRIES and not shutil.which(os.environ.get("STRATA_SNAPSHOT_NODE") or "node"):
                self.fail("node is not installed: the app entry points cannot be driven (install Node 24, or set STRATA_SNAPSHOT_SKIP_APP=1 to skip them by name)")
            got_entries[entry] = drive(machine_key, set_key, entry)
        if UPDATE:
            merged = dict(want["entries"]) if want else {}
            merged.update(got_entries)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump({"machine": machine_key, "set": set_key, "entries": {e: merged[e] for e in ENTRY_POINTS if e in merged}}), encoding="utf-8", newline="\n")
            return
        self.assertIsNotNone(want, f"no golden file {path}: run with STRATA_SNAPSHOT_UPDATE=1 and review the new file")
        for entry, got in got_entries.items():
            with self.subTest(machine=machine_key, set=set_key, entry=entry):
                self.assertNotIn("failed", got, f"{machine_key}/{set_key}/{entry} could not be driven: {got.get('failed')}")
                if want["entries"].get(entry) != got:
                    self.fail(f"\n{machine_key} / {set_key} / {entry}: what the engine would be started with changed.\n"
                              + readable_diff(want["entries"].get(entry, {}), got, f"{machine_key}/{set_key}/{entry}")
                              + "\nIf the change is intended: STRATA_SNAPSHOT_UPDATE=1 python -m pytest tests/snapshot, then review the golden/ diff.")

    def test_every_golden_file_belongs_to_the_matrix(self):
        have = {(p.parent.name, p.stem) for p in GOLDEN.glob("*/*.json")}
        want = {(m.key, s.key) for m in MACHINES for s in MODEL_SETS}
        self.assertEqual(sorted(have - want), [], "golden files for a class or set that is no longer in the matrix")
        if not UPDATE:
            self.assertEqual(sorted(want - have), [], "a class x set without a golden file")


def _add(machine_key: str, set_key: str) -> None:
    def test(self):
        self.check_cell(machine_key, set_key)
    test.__name__ = f"test_{machine_key}__{set_key}"
    setattr(ConfigSnapshot, test.__name__, test)


for _m in MACHINES:
    for _s in MODEL_SETS:
        _add(_m.key, _s.key)

if __name__ == "__main__":
    unittest.main()
