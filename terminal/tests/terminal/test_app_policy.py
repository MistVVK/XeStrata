"""The app's host reads its launch-policy tables from apps/host/server/policy-generated.mjs, which strata_control/app_policy.py writes from the Python that owns them.
A table changed in Python and not regenerated, or edited by hand in the JavaScript, fails here."""
from __future__ import annotations

import unittest

from strata_control import app_policy, drafting, speed_switches


class GeneratedPolicy(unittest.TestCase):
    def test_the_committed_file_is_what_the_python_makes_now(self):
        have = app_policy.TARGET.read_text(encoding="utf-8")
        self.assertEqual(have, app_policy.render(), "run: python -m strata_control.app_policy")

    def test_the_committed_vectors_are_what_the_python_answers_now(self):
        self.assertEqual(app_policy.VECTORS.read_text(encoding="utf-8"), app_policy.render_vectors(), "run: python -m strata_control.app_policy")

    def test_the_app_shows_every_switch_but_the_terminal_only_one(self):
        ids = [r["id"] for r in app_policy.table()]
        self.assertEqual(ids, [s.id for s in speed_switches.SWITCHES if not s.terminal_only])
        self.assertNotIn("qwen_defaults_026", ids)

    def test_a_row_carries_exactly_what_the_python_switch_holds(self):
        for row in app_policy.table():
            sw = speed_switches.BY_ID[row["id"]]
            self.assertEqual((row["label"], row["hint"], row["default"], row["env"], row.get("legacy", {})),
                             (sw.label, sw.effect, sw.default, dict(sw.env), dict(sw.legacy)))

    def test_the_drafting_families_and_names_are_drafting_py_own(self):
        text = app_policy.render()
        for family in drafting.SUPPORTED_FAMILIES:
            self.assertIn(f'"{family}"', text)
        self.assertIn(drafting._SUPPORTED_NAME.pattern.replace("\\", "\\\\"), text)


if __name__ == "__main__":
    unittest.main()
