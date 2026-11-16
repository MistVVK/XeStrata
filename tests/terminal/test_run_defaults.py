"""#5858: run-defaults.json beside launch-presets/ holds user-wide server settings for every `strata launch` (the reporter's example:
"undeclared_tool_calls": "call"). Lowest precedence: the model's installed config, the Settings view's saved values and this launch's own
--set come first; no file is today's behaviour; a bad key is skipped and said, the rest apply. Throw-away config folders only."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner
from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config, read_user_defaults


BASE = {"exe": "component:strata-engine", "model_name": "qwen3.8-flash-next-iq3_s", "args": ["--max-context", "32768"]}


class RunDefaults(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-run-defaults-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        env = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.dir / "profiles")})
        env.start()
        self.addCleanup(env.stop)
        self.path = self.dir / "run-defaults.json"

    def write(self, data) -> None:
        self.path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def derive(self, base=BASE, s=None, prev=None):
        defaults, notes = read_user_defaults()
        return derive_config(base, s or S.LaunchSettings(), bool_flags=frozenset(), prev=prev, defaults=defaults), notes

    def test_the_file_sits_beside_launch_presets(self):
        self.assertEqual(runner.user_defaults_path(), self.path)

    def test_no_file_is_todays_behaviour(self):
        cfg, notes = self.derive()
        self.assertEqual(notes, [])
        self.assertNotIn("undeclared_tool_calls", cfg)
        self.assertNotIn(runner.USER_DEFAULTS_KEY, cfg)

    def test_every_launch_gets_the_defaults(self):
        self.write({"undeclared_tool_calls": "call", "reasoning_budget_tokens": 4000, "sampling": {"temperature": 0.4}})
        cfg, notes = self.derive()
        self.assertEqual(notes, [])
        self.assertEqual((cfg["undeclared_tool_calls"], cfg["reasoning_budget_tokens"], cfg["sampling"]["temperature"]), ("call", 4000, 0.4))
        other, _ = self.derive(base={**BASE, "model_name": "swift-1.5-iq2_xs"})
        self.assertEqual(other["undeclared_tool_calls"], "call")                      # another model: the same defaults

    def test_the_models_own_setting_comes_first(self):
        self.write({"undeclared_tool_calls": "call", "idle_unload_s": 300})
        cfg, _ = self.derive(base={**BASE, "undeclared_tool_calls": "text"})
        self.assertEqual((cfg["undeclared_tool_calls"], cfg["idle_unload_s"]), ("text", 300))

    def test_this_launchs_set_comes_first(self):
        self.write({"idle_unload_s": 300})
        cfg, _ = self.derive(s=S.LaunchSettings(extra={"--idle-unload": "60"}))
        self.assertEqual(cfg["idle_unload_s"], 60)
        cfg, _ = self.derive(s=S.LaunchSettings(extra={"--idle-unload": "off"}))
        self.assertNotIn("idle_unload_s", cfg)

    def test_a_settings_view_change_stays_and_a_changed_defaults_file_takes_effect(self):
        self.write({"reasoning_budget_tokens": 4000, "api_monitor": True})
        first, _ = self.derive()
        prev = {**first, "api_monitor": False}                                        # the Settings view changed one of them since
        self.write({"reasoning_budget_tokens": 8000, "api_monitor": True})            # and the person edited the defaults file
        second, _ = self.derive(prev=prev)
        self.assertEqual(second["reasoning_budget_tokens"], 8000)                     # from the file as it is now, not frozen by the carry-over
        self.assertIs(second["api_monitor"], False)                                   # the Settings view's own change stays

    def test_a_bad_key_is_skipped_and_said_the_rest_apply(self):
        self.write({"undeclared_tool_calls": "maybe", "mtp": False, "open_browser": True, "--spec": 4, "idle_unload_s": 120,
                    "fit_max_tokens": "yes"})
        cfg, notes = self.derive()
        self.assertEqual(cfg["idle_unload_s"], 120)
        for key in ("undeclared_tool_calls", "mtp", "open_browser", "--spec", "fit_max_tokens"):
            self.assertTrue(any(f": {key} is skipped (" in n for n in notes), key)
        for key in ("undeclared_tool_calls", "fit_max_tokens", "--spec"):
            self.assertNotIn(key, cfg)
        self.assertNotIn("--spec", cfg["args"])

    def test_a_file_that_is_not_an_object_is_skipped(self):
        self.write("[1, 2]")
        cfg, notes = self.derive()
        self.assertEqual(notes, [f"{self.path} is skipped: it must hold one JSON object of settings."])
        self.write("{not json")
        _, notes = self.derive()
        self.assertTrue(notes[0].startswith(f"{self.path} is skipped: it is not valid JSON ("))


if __name__ == "__main__":
    unittest.main()
