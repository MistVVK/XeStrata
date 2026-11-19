"""0.2.17 (owner, 11-19): the route tail skip (STRATA_ROUTE_TAIL_SKIP) is OFF by default.

Engine 0.1.62 skips with rank 7 when the variable is unset (CUDA, HIP gfx1200/gfx1201), and 0.2.17 ships on it, so Control sets the variable at every
engine start: "0" when neither the person's own environment nor the model's run config names it, so an install made before 0.2.17 starts Off without a
new setup.  STRATA_ROUTE_TAIL_SKIP=7 (On) in the run config, or any value in the person's own environment, is kept as it is.
Order: the person's own environment, then the run config's "env" over it, then Control's default for what is still unset.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import server  # noqa: E402
from strata_control import speed_switches as ss  # noqa: E402
from strata_control import tune_engine  # noqa: E402

NAME = "STRATA_ROUTE_TAIL_SKIP"


def start_env(cfg, own=None):
    """child_env as a server started with this process environment (`own`) builds it."""
    with mock.patch.dict(os.environ, own or {}, clear=False):
        if not (own and NAME in own):
            os.environ.pop(NAME, None)
        return server.child_env(cfg)


class ServerStartEnv(unittest.TestCase):
    def test_no_setting_anywhere_starts_the_engine_with_zero(self):
        for backend in ("cuda", "hip", "sycl", "metal", "cpu", None):
            cfg = {"backend": backend, "env": {"PATH": "/native/bin"}} if backend else {"env": {}}
            with mock.patch.object(server, "hip_auto_devices", return_value=None):
                self.assertEqual(start_env(cfg)[NAME], "0", backend)

    def test_a_config_made_before_0_2_17_with_no_entry_starts_off(self):
        # 0.2.16's Automatic (and its On, which was stored as no entry) wrote nothing: that is the same file as a fresh default now
        self.assertEqual(start_env({"backend": "cuda"})[NAME], "0")
        self.assertEqual(start_env({"backend": "cuda", "env": {}})[NAME], "0")

    def test_the_run_configs_explicit_on_is_kept(self):
        self.assertEqual(start_env({"backend": "cuda", "env": {NAME: "7"}})[NAME], "7")
        self.assertEqual(start_env({"backend": "cuda", "env": {NAME: 5}})[NAME], "5")

    def test_the_run_configs_zero_stays_zero(self):
        self.assertEqual(start_env({"backend": "cuda", "env": {NAME: "0"}})[NAME], "0")

    def test_the_persons_own_environment_wins_over_the_default(self):
        self.assertEqual(start_env({"backend": "cuda"}, {NAME: "7"})[NAME], "7")
        self.assertEqual(start_env({"backend": "cuda"}, {NAME: "4"})[NAME], "4")
        self.assertEqual(start_env({"backend": "cuda"}, {NAME: "0"})[NAME], "0")

    def test_the_run_config_is_applied_over_the_persons_environment_as_before(self):
        self.assertEqual(start_env({"backend": "cuda", "env": {NAME: "0"}}, {NAME: "7"})[NAME], "0")

    def test_the_default_changes_nothing_else(self):
        env = start_env({"backend": "cuda", "env": {"STRATA_KVA": "1"}})
        self.assertEqual(env["STRATA_KVA"], "1")
        self.assertEqual({k for k in env if k.startswith("STRATA_ROUTE")}, {NAME})


class DefaultHelper(unittest.TestCase):
    def test_it_sets_an_unset_value_in_place_and_returns_the_dict(self):
        given = {"A": "1"}
        self.assertIs(ss.engine_env_defaults(given), given)
        self.assertEqual(given, {"A": "1", NAME: "0"})

    def test_a_value_that_is_there_is_never_changed_not_even_an_empty_one(self):
        for value in ("7", "0", "", "3"):
            self.assertEqual(ss.engine_env_defaults({NAME: value})[NAME], value)


class TuningRunsMeasureWhatALaunchRuns(unittest.TestCase):
    def test_the_measurement_engine_starts_off_unless_told_otherwise(self):
        harness = tune_engine.EngineHarness(Path("strata"), {}, env={"PATH": "/bin"})
        self.assertEqual(harness._start_env()[NAME], "0")
        on = tune_engine.EngineHarness(Path("strata"), {}, env={"PATH": "/bin", NAME: "7"})
        self.assertEqual(on._start_env()[NAME], "7")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(NAME, None)
            with mock.patch("strata_control.tuning.engine_env", return_value=None):
                inherit = tune_engine.EngineHarness(Path("strata"), {})
            self.assertEqual(inherit._start_env()[NAME], "0")


if __name__ == "__main__":
    unittest.main()
