"""#5956 (revised) and #5958: the restart plan says what the stop phase can take at worst (teardown_limit_s: the drain, the kill and the reap
of Host._stop_child) and, apart from it, what the last stop here took (last_teardown_s, informational, never a bound). A fake clock, no
processes: fast then slow stop, a stop that times out and is reaped, and a failed load with rollback. Also: the status answer already carries
what a swap gate needs (#5958), and a restart planned on one instance is refused once another instance runs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import runtime


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class FakeChild:
    """A server child: a stop takes `took` seconds of fake time, or never ends by itself (took=None: the wait times out)."""

    def __init__(self, clock: Clock, took: float | None, pid: int = 4242):
        self.clock, self.took, self.pid, self.returncode = clock, took, pid, None
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, _sig):
        pass

    def wait(self, timeout=None):
        if self.took is not None and not self.killed:
            if self.took > timeout:
                self.clock.now += timeout
                raise subprocess.TimeoutExpired("server", timeout)
            self.clock.now += self.took
            self.returncode = 0
            return 0
        if self.killed:
            self.clock.now += timeout                      # the reap takes its whole wait (the worst case)
            self.returncode = -9
            return -9
        self.clock.now += timeout
        raise subprocess.TimeoutExpired("server", timeout)


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class TeardownBudget(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-teardown-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(self.dir / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        self.clock = Clock()
        patch = mock.patch.object(runtime.time, "monotonic", self.clock.monotonic)
        patch.start()
        self.addCleanup(patch.stop)
        self.host = runtime.Host("local", shutdown_timeout=30)
        self.host._close_owner_pipe = lambda: None
        self.host._kill_child_tree = lambda: setattr(self.host.child, "killed", True)
        self.source = self.dir / "run.json"
        self.source.write_bytes(b"{}")
        self.host.args, self.host.source, self.host.source_bytes, self.host.revision = [], self.source, b"{}", "r" * 16
        self.host._status = lambda: {"ok": True, "state": "ready", "owner_id": "o" * 32, "instance_id": "i" * 32}
        self.host._plan = lambda: ({"profile_revision": "r" * 16}, [], self.source, b"{}")
        self.host._source_change_check = lambda *a: (False, False, None)
        self.limit = 30 + runtime.STOP_KILL_GRACE + runtime.STOP_REAP_WAIT

    def plan(self):
        return self.host._plan_restart()

    def stop(self, took):
        self.host.child = FakeChild(self.clock, took)
        return self.host._stop_child()

    def test_before_any_stop_the_plan_has_the_limit_and_no_observation(self):
        plan = self.plan()
        self.assertEqual((plan["teardown_limit_s"], plan["last_teardown_s"]), (round(self.limit, 1), None))

    def test_a_fast_stop_then_a_slow_one_never_lowers_the_limit(self):
        self.assertTrue(self.stop(1.0)["ok"])
        fast = self.plan()
        self.assertEqual((fast["last_teardown_s"], fast["teardown_limit_s"]), (1.0, round(self.limit, 1)))
        self.assertTrue(self.stop(29.0)["ok"])                  # the next stop is slow: the earlier fast one was no bound
        slow = self.plan()
        self.assertEqual((slow["last_teardown_s"], slow["teardown_limit_s"]), (29.0, round(self.limit, 1)))
        self.assertGreater(slow["last_teardown_s"], fast["last_teardown_s"] + 5)   # the old "last drain + 5 s" floor would have been short

    def test_a_stop_that_times_out_and_is_reaped_takes_the_whole_limit(self):
        result = self.stop(None)
        self.assertEqual(result["error"], "server_stop_deadline_expired")
        plan = self.plan()
        self.assertEqual(plan["last_teardown_s"], round(self.limit, 1))
        self.assertLessEqual(plan["last_teardown_s"], plan["teardown_limit_s"])

    def test_a_failed_load_with_rollback_stays_inside_two_teardowns_and_two_waits(self):
        wait = 120.0
        launches = []

        def launch(args, source, source_bytes, revision):
            launches.append(revision)
            self.host.child = FakeChild(self.clock, None if len(launches) == 1 else 2.0, pid=5000 + len(launches))

        answers = iter([{"ok": False, "state": "failed", "error": "server_exited"}, {"ok": True, "state": "ready", "instance_id": "n" * 32}])

        def wait_ready(timeout):
            self.clock.now += timeout                          # each wait runs to its end: the worst case
            return next(answers)

        self.host._launch, self.host._wait_ready = launch, wait_ready
        self.host.child = FakeChild(self.clock, 30 + runtime.STOP_KILL_GRACE)   # the old server drains as late as it may (a kill there ends the restart)
        began = self.clock.now
        result = self.host._restart(wait)
        self.assertTrue(result.get("rolled_back"), result)
        self.assertEqual(len(launches), 2)                      # the new launch and the rollback
        self.assertLessEqual(self.clock.now - began, 2 * self.limit + 2 * wait)   # the documented budget holds at its worst

    def test_the_status_answer_already_has_what_a_swap_gate_reads(self):
        # #5958: no new action; the existing status of a ready instance carries state, instance, revision, model state and address
        host = runtime.Host("local", shutdown_timeout=30)
        host.child = FakeChild(self.clock, 1.0)
        host.revision = "r" * 16
        host.ready_path = self.dir / "ready.json"
        host.ready_path.write_text(runtime.json.dumps({"pid": host.child.pid, "instance_id": "i" * 32, "model_state": "ready"}))
        host.probe_key = (host.child.pid, "i" * 32)
        host.probe_result = {"api_identity_verified": True, "launch_attested": True, "generation_ready": True,
                             "model_state": "ready", "api_url": "http://127.0.0.1:8080"}
        with mock.patch.object(runtime, "_ready_pid_is_child", lambda pid, child: pid == child):
            status = host._status()
        for key in ("state", "instance_id", "profile_revision", "model_state", "api_url"):
            self.assertIn(key, status)
        self.assertEqual((status["state"], status["api_url"]), ("ready", "http://127.0.0.1:8080"))

    def test_a_ready_file_of_an_earlier_instance_gives_no_address(self):
        host = runtime.Host("local", shutdown_timeout=30)
        host.child = FakeChild(self.clock, 1.0, pid=7001)
        host.ready_path = self.dir / "ready.json"
        host.ready_path.write_text(runtime.json.dumps({"pid": 6999, "instance_id": "e" * 32, "api_url": "http://127.0.0.1:9"}))
        with mock.patch.object(runtime, "_ready_pid_is_child", lambda pid, child: pid == child):
            status = host._status()
        self.assertEqual(status["state"], "starting")
        self.assertNotIn("api_url", status)

    def test_a_restart_planned_on_one_instance_is_refused_after_a_swap(self):
        planned = self.plan()["current_instance_id"]
        self.host._status = lambda: {"ok": True, "state": "ready", "owner_id": "o" * 32, "instance_id": "x" * 32}   # swapped meanwhile
        self.host.child = FakeChild(self.clock, 1.0)
        result = self.host._restart(60, expected_instance_id=planned)
        self.assertEqual((result["ok"], result["error"]), (False, "instance_changed_after_restart_plan"))
        self.assertIsNone(self.host.child.returncode)            # nothing was stopped


class PlanReadersTakeTheNewFields(unittest.TestCase):
    """Every reader of the restart plan accepts the two added fields (additive: none of them checks for unknown keys)."""

    PLAN = {"ok": True, "state": "ready", "profile": "local", "current_owner_id": "o" * 32, "current_instance_id": "b" * 32,
            "current_profile_revision": "a" * 16, "candidate_profile_revision": "a" * 16, "candidate_source_sha256": "s" * 64,
            "changed": False, "shutdown_timeout": 30, "teardown_limit_s": 48.0, "last_teardown_s": 2.5,
            "would_drain": False, "would_start": False}

    def test_bench_managed_match(self):
        from strata_cli import bench
        owner = {"ok": True, "state": "ready", "launch_attested": True, "api_identity_verified": True, "generation_ready": True,
                 "instance_id": "b" * 32, "profile_revision": "a" * 16, "api_url": "http://127.0.0.1:8080"}
        with mock.patch.object(runtime, "status", return_value=owner), mock.patch.object(runtime, "plan_restart", return_value=dict(self.PLAN)):
            got = bench._managed_match({"profile": "local", "revision": "a" * 16}, "http://127.0.0.1:8080", {"instance": {"id": "b" * 32}})
        self.assertTrue(got["matched"], got)

    def test_recipes_plan_and_run_preconditions(self):
        from strata_control import recipes
        recipe = {"kind": "runtime.restart", "params": {"startup_wait_s": 60}}
        with mock.patch.object(recipes, "load", return_value=(recipe, "v" * 16)), \
                mock.patch.object(recipes.profiles, "load", return_value=({}, "a" * 16)), \
                mock.patch.object(runtime, "plan_restart", return_value=dict(self.PLAN)):
            planned = recipes.plan("local", "restart")
        self.assertEqual(planned["operation"]["preflight"]["teardown_limit_s"], 48.0)
        self.assertTrue(planned["operation"]["ok"])


if __name__ == "__main__":
    unittest.main()
