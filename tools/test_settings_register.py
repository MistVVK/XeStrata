"""The settings register must list exactly the STRATA_* settings the engine reads (tools/settings_register.py, terminal/data/settings-register.json).

Fails when
  * the engine reads a setting that is not in the register (add it: python tools/settings_register.py, then set its status);
  * the register lists a setting that no engine code reads, unless its status is "retired" and it names the version in "retired_in";
  * an entry has no status, an unknown status, or a retire-candidate / unsure entry without a note that cites its evidence.

Run:  python -m unittest tools.test_settings_register   (or pytest tools/test_settings_register.py)
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import settings_register as sr   # noqa: E402


def _register():
    return json.loads(sr.REGISTER.read_text(encoding="utf-8"))["settings"]


class SettingsRegister(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sr.engine_names()
        cls.entries = _register()
        cls.registered = {e["name"] for e in cls.entries}

    def test_engine_reads_nothing_unregistered(self):
        missing = sorted(self.engine - self.registered)
        self.assertEqual(missing, [], "the engine reads settings that are not in terminal/data/settings-register.json "
                                      "(run python tools/settings_register.py and give each a status): " + ", ".join(missing[:20]))

    def test_register_has_no_orphans(self):
        retired = {e["name"] for e in self.entries if e.get("status") == "retired" and e.get("retired_in")}
        orphans = sorted(self.registered - self.engine - retired)
        self.assertEqual(orphans, [], "listed in the register but read by no engine code (remove them, or mark them "
                                      "status=retired with retired_in=<version>): " + ", ".join(orphans[:20]))

    def test_names_are_unique_and_sorted(self):
        names = [e["name"] for e in self.entries]
        self.assertEqual(names, sorted(set(names)))

    def test_every_entry_has_a_valid_status(self):
        bad = [e["name"] for e in self.entries if e.get("status") not in sr.STATUSES]
        self.assertEqual(bad, [], "entries without a valid status: " + ", ".join(bad[:20]))

    def test_judgement_entries_cite_evidence(self):
        bad = [e["name"] for e in self.entries
               if e.get("status") in ("retire-candidate", "unsure", "rollback") and not str(e.get("note", "")).strip()]
        self.assertEqual(bad, [], "retire-candidate, unsure and rollback entries need a note with the evidence: " + ", ".join(bad[:20]))

    def test_retire_candidates_cite_a_file_and_line(self):
        import re
        bad = [e["name"] for e in self.entries
               if e.get("status") == "retire-candidate" and not re.search(r"[\w./\\-]+\.\w+:\d+", str(e.get("note", "")))]
        self.assertEqual(bad, [], "retire-candidate notes must cite file:line: " + ", ".join(bad[:20]))

    def test_retired_entries_name_a_version(self):
        bad = [e["name"] for e in self.entries if e.get("status") == "retired" and not e.get("retired_in")]
        self.assertEqual(bad, [], "retired entries need retired_in: " + ", ".join(bad))

    def test_scan_sees_the_known_shapes(self):
        # guards the scanner itself: a plain getenv, a helper, and an off-switch
        self.assertIn("STRATA_HIPBLASLT_TUNING", self.engine)
        self.assertIn("STRATA_KVA", self.engine)
        self.assertGreater(len(self.engine), 900)


if __name__ == "__main__":
    unittest.main()
