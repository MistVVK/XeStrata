"""#4824: a runtime host left in "failed" (its server exited, the host did not) is cleared before the tuning and before the start, a start that still
meets one says so and what to do, and a launch preset asked for with --save-preset is kept when the start fails after the tuning."""

import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import store
from strata_cli.launch import settings as S
from strata_cli.launch.runner import LEFTOVER_STATES, LaunchError, Runner, _log_hint
from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, run_cli


def bare_runner():
    class Bare(Runner):
        profile = "launch-x"
    r = Bare.__new__(Bare)
    r.s = S.LaunchSettings(name="x")
    r.restart_after = True
    r._stop_first = []
    r.said = []
    r.emit = lambda *a, **k: r.said.append(a[1])
    return r


class Leftover(unittest.TestCase):
    def test_every_state_that_blocks_a_start_is_one_that_launch_stop_ends(self):
        self.assertEqual(set(LEFTOVER_STATES), {"ready", "starting", "failed", "stale"})

    def test_a_failed_host_is_cleared_before_the_tuning_and_the_line_says_so(self):
        for state in ("failed", "stale"):
            r = bare_runner()
            with mock.patch("strata_control.runtime.status", return_value={"state": state}), mock.patch("strata_control.runtime.stop") as stop:
                r.stop_own_server_for_tuning()
            stop.assert_called_once()
            self.assertEqual(r.said, ["Clearing a stopped server of this preset before measuring."])

    def test_a_running_server_keeps_its_own_line(self):
        r = bare_runner()
        with mock.patch("strata_control.runtime.status", return_value={"state": "ready"}), mock.patch("strata_control.runtime.stop"):
            r.stop_own_server_for_tuning()
        self.assertEqual(r.said, ["Stopping the model to measure; it starts again afterwards."])

    def test_the_start_clears_a_failed_host_before_it_starts(self):
        r = bare_runner()
        r.cancel = mock.Mock(is_set=lambda: False)
        r.config_path = None
        r.stop_others = lambda: None
        r.check_port_free = lambda: None
        r.log_marks = lambda: {}
        r.start_wait = lambda: 5
        r.report_drafting = lambda: None
        calls = []
        with mock.patch("strata_control.runtime.status", return_value={"state": "failed", "error": "server_exited"}), \
                mock.patch("strata_control.runtime.stop", side_effect=lambda *a, **k: calls.append("stop")), \
                mock.patch("strata_control.runtime.start", side_effect=lambda *a, **k: calls.append("start") or {"ok": True}):
            r.start()
        self.assertEqual(calls, ["stop", "start"])
        self.assertTrue(any("stopped earlier and was never cleared" in line for line in r.said), r.said)

    def test_the_message_for_a_profile_that_is_still_held_names_the_cure_and_the_retune(self):
        hint = _log_hint(None, "this profile already has a live managed runtime")
        self.assertIn("`strata launch --stop`", hint)
        self.assertIn("without --retune", hint)
        self.assertNotIn("Run the command again", hint)


class SavedPreset(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def test_a_start_that_fails_after_the_tuning_keeps_the_preset_that_was_asked_for(self):
        def fails(self_, **_):
            self_.config_path = Path("/x")                              # the settings were written: the failure came later
            raise LaunchError("The server could not start: busy.", "Run `strata launch --stop`.", 6)
        with mock.patch.object(Runner, "run", fails):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--save-preset", "Office"])
        self.assertEqual(code, 6, err)
        self.assertEqual([p.name for p in store.list_presets()], ["Office"])
        self.assertIn("saved as the launch preset Office", out)
        self.assertIn("strata launch --preset Office", out)
        self.assertIn("The server could not start", err)

    def test_a_failure_before_the_settings_were_written_saves_nothing(self):
        with mock.patch.object(Runner, "run", side_effect=LaunchError("The engine could not be installed.", "Check your internet.", 6)):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--save-preset", "Office"])
        self.assertEqual(code, 6, err)
        self.assertEqual(store.list_presets(), [])


if __name__ == "__main__":
    unittest.main()
