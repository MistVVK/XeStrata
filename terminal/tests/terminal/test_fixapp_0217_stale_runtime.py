"""#3977 / #6239: "A model is running, so nothing was changed" with no model loaded.

A managed-runtime record whose owner host is gone (the app was closed or crashed with a model loaded) reads as state "stale". `serving_now()` counted it
as an engine that serves, so `strata update` refused with exit 7 and the app said a model is running. The person's workaround was to load a model once and
unload it, which clears the record. `serving_now()` now clears a stale record the way `strata runtime stop` does for a lost host; a record that cannot
be cleared, or whose owner is alive, still counts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import runtime, updater  # noqa: E402


class StaleRuntimeRecord(unittest.TestCase):
    def serving(self, managed, recovered):
        with mock.patch.object(runtime, "active_managed_instances", return_value=managed), \
                mock.patch.object(runtime, "recover_lost", side_effect=recovered) as recover, \
                mock.patch.object(runtime, "acquire_activity_lease", side_effect=OSError("no lease in this test")):
            return updater.serving_now(), recover

    def test_a_stale_record_is_cleared_and_nothing_is_serving(self):
        stopped = {"ok": True, "profile": "app-1", "state": "stopped", "recovered": True, "exit_code": 0}
        answer, recover = self.serving([{"profile": "app-1", "state": "stale"}], lambda name: stopped)
        self.assertEqual(answer, [])
        recover.assert_called_once_with("app-1")

    def test_a_stale_record_that_cannot_be_cleared_still_counts(self):
        stuck = {"ok": False, "profile": "app-1", "state": "failed", "error": "leftover_process_not_ended", "exit_code": 6}
        answer, _ = self.serving([{"profile": "app-1", "state": "stale"}], lambda name: stuck)
        self.assertEqual(answer, [{"profile": "app-1", "state": "stale"}])

    def test_a_record_that_raises_still_counts(self):
        def boom(name):
            raise runtime.ProfileError("lost host has no valid child ownership record; inspect before recovery", 6)
        answer, _ = self.serving([{"profile": "app-1", "state": "stale"}], boom)
        self.assertEqual(answer, [{"profile": "app-1", "state": "stale"}])

    def test_an_owner_that_is_alive_again_counts_and_is_not_touched(self):
        answer, _ = self.serving([{"profile": "app-1", "state": "stale"}], lambda name: None)
        self.assertEqual(answer, [{"profile": "app-1", "state": "stale"}])

    def test_a_model_that_is_really_serving_is_never_recovered(self):
        answer, recover = self.serving([{"profile": "app-1", "state": "ready"}], lambda name: self.fail("recovered a live model"))
        self.assertEqual(answer, [{"profile": "app-1", "state": "ready"}])
        recover.assert_not_called()

    def test_only_the_stale_one_goes_when_two_are_listed(self):
        stopped = {"ok": True, "profile": "app-1", "state": "stopped", "exit_code": 0}
        answer, _ = self.serving([{"profile": "app-1", "state": "stale"}, {"profile": "app-2", "state": "ready"}], lambda name: stopped)
        self.assertEqual(answer, [{"profile": "app-2", "state": "ready"}])

    def test_recover_lost_never_asks_a_live_owner_to_stop(self):
        with mock.patch.object(runtime, "_read_state", return_value={"token": "t"}), \
                mock.patch.object(runtime, "_host_lease_active", return_value=True), \
                mock.patch.object(runtime, "_rpc", side_effect=AssertionError("asked a live owner")):
            self.assertIsNone(runtime.recover_lost("app-1"))


if __name__ == "__main__":
    unittest.main()
