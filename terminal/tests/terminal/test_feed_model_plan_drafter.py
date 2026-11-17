"""`strata launch --plan` for a feed model whose entry says `drafter: none` (found by the Q4-P8 rehearsal, G14: the plan said "Drafting: MTP" while setup fetches no draft layer)."""

from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch import settings as S
from tests.terminal.launch_support import HW_12, Home, catalog, not_intel


class PlanLineOfAFeedModel(unittest.TestCase):
    def setUp(self):
        not_intel(self)

    def runner(self, home, family):
        from strata_cli.launch.runner import Runner
        cat = catalog(HW_12)
        models, families, contexts, source = cat.tables()
        fam = dict(families["qwen"], feed=True, **({"drafter": "none"} if family == "none" else {"drafter": "mtp"}))
        tables = (models, {**families, "feedmodel": fam}, contexts, source)
        p = mock.patch.object(type(cat), "tables", lambda self: tables)
        p.start()
        self.addCleanup(p.stop)
        return Runner(S.LaunchSettings(family="feedmodel", model="Q2_0", context=32768, vision="off"), catalog=cat)

    def test_drafter_none_is_not_drafting_mtp(self):
        with Home() as home:
            line = self.runner(home, "none").drafting_plan_line()
            self.assertEqual(line, "Drafting: Off")
            self.assertNotIn("MTP", line)

    def test_a_feed_model_with_a_drafter_keeps_the_usual_line(self):
        with Home() as home:
            self.assertTrue(self.runner(home, "mtp").drafting_plan_line().startswith("Drafting:"))
            self.assertNotEqual(self.runner(home, "mtp").drafting_plan_line(), "Drafting: Off")


if __name__ == "__main__":
    unittest.main()
