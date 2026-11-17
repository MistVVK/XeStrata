"""#5232: `strata tune auto --plan` lists what the tune would try and leave out without starting it, from the one planner the run uses
(autotune.plan_candidates); `strata tune auto NAME` also takes an installed model (IQ3_S, qwen/IQ3_S, a unique part of its name) and refuses a name that fits
several or none with the names."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import tune_commands as TC
from strata_control import autotune, profiles, tune_knobs
from tests.tuning.test_autotune import make_ctx, run_tuner, speed_with


class OnePlanner(unittest.TestCase):
    def test_the_run_and_the_preview_use_plan_candidates(self):
        run_src = inspect.getsource(autotune.Tuner._run_stages)
        self.assertIn("plan_candidates(self.ctx, o,", run_src)
        self.assertNotIn("plan_arms(", run_src)
        self.assertIn("plan_candidates(ctx, o,", inspect.getsource(autotune.preview))

    def test_the_preview_lists_what_the_run_then_measures(self):
        model = speed_with({})
        seen = {}
        real = autotune.plan_candidates

        def spy(*a, **kw):
            out = real(*a, **kw)
            seen.setdefault("run", out)
            return out
        with mock.patch.object(autotune, "plan_candidates", side_effect=spy):
            _t, _s, result = run_tuner(model)
        tried = [p.arm.id for p in seen["run"][0]]
        plan = autotune.preview(make_ctx(), autotune.Options(), avg_run_s=150.0)
        self.assertEqual([t["id"] for t in plan["try"]], tried[:len(plan["try"])])
        self.assertGreater(plan["engine_runs"], 1)
        self.assertEqual(plan["budget_minutes"], 20)
        self.assertTrue(set(t["id"] for t in plan["try"]).isdisjoint(s["id"] for s in plan["left_out"]))

    def test_thorough_keeps_every_candidate_and_quick_says_what_the_time_leaves_out(self):
        quick = autotune.preview(make_ctx(), autotune.Options(mode="quick", budget_s=600.0), avg_run_s=150.0)
        thorough = autotune.preview(make_ctx(), autotune.Options(mode="thorough"), avg_run_s=150.0)
        planned, _ = tune_knobs.plan_arms(make_ctx(), mode="thorough")
        self.assertEqual(len(thorough["try"]), len(planned))
        if len(planned) > len(quick["try"]):
            self.assertTrue(any(s["reason"] == "left out to stay inside the time you chose" for s in quick["left_out"]))

    def test_the_words(self):
        plan = {"mode": "quick", "budget_minutes": 20, "model": "IQ3_S", "config": "/c.json", "engine_runs": 9, "run_minutes": 2.5, "minutes": 22,
                "try": [{"id": "a", "label": "Keep more of the conversation in graphics memory"}],
                "left_out": [{"id": "b", "label": "Expert cache split", "reason": "it can change answers slightly"}]}
        text = TC._plan_words(plan)
        self.assertIn("Tuning plan for IQ3_S (quick, up to 20 minutes). Nothing was started.", text)
        self.assertIn("  Keep more of the conversation in graphics memory", text)
        self.assertIn("  Expert cache split: it can change answers slightly", text)
        self.assertIn("About 9 engine runs of about 2.5 minutes each, about 22 minutes, more than the quick tune's 20, so the last settings may not be reached. "
                      "The first run measures", text)
        within = TC._plan_words({**plan, "engine_runs": 7, "minutes": 18})
        self.assertIn("About 7 engine runs of about 2.5 minutes each, about 18 minutes. The first run measures", within)
        self.assertNotIn("may not be reached", within)


class FakeCatalog:
    def __init__(self, rows):
        self.rows = rows

    def resolve(self, size, family=None):
        sizes = {"iq3_s": "IQ3_S", "q2_0": "Q2_0", "iq1_m": "IQ1_M"}
        if size.lower() not in sizes:
            raise ValueError("no such size")
        return (family or ("coder" if size.lower() == "iq1_m" else "qwen")), sizes[size.lower()]

    def tag_of(self, family, model):
        return ("coder-" if family == "coder" else "") + model.lower()

    def installed_config(self, tag):
        hit = [r for r in self.rows if Path(r["path"]).name == f"strata-{tag}.json"]
        return Path(hit[0]["path"]) if hit else None

    def installed_configs(self):
        return list(self.rows)


ROWS = [{"path": "/i/strata-iq3_s.json", "model_name": "qwen3.8-flash-next-iq3_s"},
        {"path": "/i/strata-q2_0.json", "model_name": "qwen3.8-flash-next-q2_0"},
        {"path": "/i/strata-unsloth-ud-q4_k_xl.json", "model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl"}]


class Names(unittest.TestCase):
    def find(self, name):
        return TC.installed_model_config(name, catalog=FakeCatalog(ROWS))

    def test_a_size_and_family_size(self):
        self.assertEqual(self.find("IQ3_S"), Path("/i/strata-iq3_s.json"))
        self.assertEqual(self.find("iq3_s"), Path("/i/strata-iq3_s.json"))
        self.assertEqual(self.find("qwen/IQ3_S"), Path("/i/strata-iq3_s.json"))

    def test_a_unique_part_of_the_name(self):
        self.assertEqual(self.find("q4_k_xl"), Path("/i/strata-unsloth-ud-q4_k_xl.json"))

    def test_several_or_none_are_refused_with_the_names(self):
        with self.assertRaises(profiles.ProfileError) as cm:
            self.find("flash-next")
        self.assertIn("fits 3 installed models", str(cm.exception))
        with self.assertRaises(profiles.ProfileError) as cm:
            self.find("IQ1_M")                              # a size the table knows, not installed, and no name holds it
        self.assertIn("No saved profile or installed model is called 'IQ1_M'. Installed models: ", str(cm.exception))

    def test_a_json_path_and_a_saved_profile_come_first(self):
        with mock.patch.object(profiles, "resolve", return_value=(Path("/p/x.json"), None, None)) as res, \
                mock.patch.object(profiles, "read_run_config", return_value={"args": []}):
            self.assertEqual(TC._load_target("/p/x.json")[0], Path("/p/x.json"))
            res.assert_called_once()

    def test_a_name_that_is_nothing_keeps_its_old_exit_code(self):
        with mock.patch.object(TC, "installed_model_config", side_effect=profiles.ProfileError("No saved profile or installed model is called 'Zz'.", 5)):
            with self.assertRaises(profiles.ProfileError) as cm:
                TC._load_target("Zz")                       # not a valid profile name: exit 2 as before
            self.assertEqual(cm.exception.exit_code, 2)
            self.assertIn("installed model", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
