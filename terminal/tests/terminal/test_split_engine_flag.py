"""`strata launch --split-engine auto|on|off`: the owner's switch for the all-cards split engine (strata_control/split_engine.py).
auto (nothing written) is the default; the choice is listed in --help, in the copyable flags, in the launch config and in a saved preset, so an
"off" survives updates and restarts; the launcher's apply_split_engine honours it (off: nothing, auto: silent on a refusal, on: one line)."""

from __future__ import annotations

import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import split_engine as SE
from tests.terminal.launch_support import run_cli


class Flag(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--pack", "/d/packs/x", "--mtp", "/d/mtp/rt"], "port": 8080, "model_name": "m"}

    def test_the_flag_is_listed_checked_copyable_and_saved_in_the_launch_config(self):
        code, out, _ = run_cli(["launch", "--help"])
        self.assertIn("--split-engine", out)
        self.assertIn("auto|on|off", out)
        for mode in S.SPLIT_MODES:
            cfg = derive_config(self.BASE, S.LaunchSettings(split_engine=mode), bool_flags=frozenset())
            self.assertEqual(cfg["split_engine"], mode)
            self.assertEqual(SE.mode(cfg, {}), mode)
        self.assertNotIn("split_engine", derive_config(self.BASE, S.LaunchSettings(), bool_flags=frozenset()), "nobody chose: auto writes nothing")
        self.assertEqual(SE.mode(derive_config(self.BASE, S.LaunchSettings(), bool_flags=frozenset()), {}), "auto")
        self.assertIn("--split-engine off", S.flags_text(S.LaunchSettings(split_engine="off")))
        self.assertNotIn("--split-engine", S.flags_text(S.LaunchSettings()))
        code, out, err = run_cli(["launch", "--plan", "--split-engine", "sometimes"])
        self.assertNotEqual(code, 0)

    def test_a_saved_off_survives_a_new_launch_config_and_a_preset_round_trip(self):
        s = S.LaunchSettings.from_dict(S.LaunchSettings(split_engine="off").to_dict())
        self.assertEqual(s.split_engine, "off")
        self.assertIsNone(S.LaunchSettings.from_dict({}).split_engine)
        # a config written earlier with "off" is kept when this launch names nothing
        cfg = derive_config({**self.BASE, "split_engine": "off"}, S.LaunchSettings(), bool_flags=frozenset())
        self.assertEqual(cfg["split_engine"], "off")
        # and the person's flag of this launch wins over the old file
        cfg = derive_config({**self.BASE, "split_engine": "off"}, S.LaunchSettings(split_engine="on"), bool_flags=frozenset())
        self.assertEqual(cfg["split_engine"], "on")


if __name__ == "__main__":
    unittest.main()
