"""Round 0213 TUICFG, step 2: `strata launch --stop` exit status and message (#4398).
Four cases scripted against a runtime double that tracks which fake PIDs are still alive: empty, successful, failed, mixed."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch import access
from tests.terminal.launch_support import run_cli

NOTHING = "Nothing from `strata launch` is running."


class Runtime:
    """profile id -> (alive?, stop succeeds?)"""

    def __init__(self, profiles):
        self.alive = {p: ok is not None for p, ok in profiles.items()}
        self.stops_ok = {p: bool(ok) for p, ok in profiles.items()}

    def list_profiles(self):
        return [{"id": p} for p in self.alive]

    def status(self, prof):
        return {"state": "ready" if self.alive.get(prof) else "stopped"}

    def stop(self, prof, timeout=None):
        if self.stops_ok.get(prof):
            self.alive[prof] = False
            return {"ok": True}
        return {"ok": False, "error": "leftover_process_not_ended"}

    def live(self):
        return sorted(p for p, a in self.alive.items() if a)


class LaunchStopExit(unittest.TestCase):
    def _run(self, profiles, *args):
        rt = Runtime(profiles)
        with mock.patch("strata_control.profiles.list_profiles", rt.list_profiles), \
             mock.patch("strata_control.runtime.status", rt.status), \
             mock.patch("strata_control.runtime.stop", rt.stop), \
             mock.patch.object(access, "stop_tunnel"):
            code, out, err = run_cli(["launch", "--stop", *args])
        return rt, code, out, err

    def test_empty_says_nothing_and_exits_zero(self):
        rt, code, out, _ = self._run({})
        self.assertEqual(code, 0)
        self.assertIn(NOTHING, out)
        rt, code, out, _ = self._run({"launch-office": None})          # a profile exists but nothing runs
        self.assertEqual((code, rt.live()), (0, []))
        self.assertIn(NOTHING, out)

    def test_successful_stop_exits_zero_and_leaves_nothing_alive(self):
        rt, code, out, _ = self._run({"launch-office": True, "launch-home": True})
        self.assertEqual((code, rt.live()), (0, []))
        self.assertIn("Stopped office.", out)
        self.assertNotIn(NOTHING, out)

    def test_failed_stop_exits_nonzero_and_does_not_claim_nothing_runs(self):
        for extra in ([], ["--preset", "Office"]):
            with self.subTest(extra=extra):
                rt, code, out, _ = self._run({"launch-office": False}, *extra)
                self.assertEqual(code, 1)
                self.assertEqual(rt.live(), ["launch-office"])
                self.assertIn("Could not stop office: leftover_process_not_ended.", out)
                self.assertNotIn(NOTHING, out)

    def test_mixed_exits_nonzero_reports_both_and_only_the_failed_one_stays(self):
        rt, code, out, _ = self._run({"launch-office": False, "launch-home": True})
        self.assertEqual(code, 1)
        self.assertEqual(rt.live(), ["launch-office"])
        self.assertIn("Could not stop office", out)
        self.assertIn("Stopped home.", out)
        self.assertNotIn(NOTHING, out)


if __name__ == "__main__":
    unittest.main()
