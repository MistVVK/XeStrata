"""#5400: an engine update loaded the model "for testing" and pushed a busy PC out of memory. When the memory free now is less than a test load of the
model needs, the new engine gets the check an install without a model gets (it runs and answers --version), the update goes on and the person is told;
that engine is "unproven" until a real start reaches ready. If its first real start fails (the engine itself, not a refusal such as no room), the
previous engine is used again by itself, once, and the version is not switched to again without the person asking (--force). Unknown memory: the full
test as before. No model installed: as before."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from strata_control import runtime, updater
from tests.terminal import update_fixture as fx
from tests.terminal.test_updater import UpdaterCase

SHORT = (20.0, 58.0)


class LowMemoryUpdate(UpdaterCase):
    def update(self, memory, **over):
        events: list = []
        out = updater.run_update(self.ctx(memory_check=lambda cfg: memory, events=events, **over), control=False)
        return out, events

    def engine_state(self):
        return updater.State(self.components).section("engine")

    def test_short_memory_switches_without_loading_the_model_and_says_so(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.object(runtime, "start", side_effect=AssertionError("the model must not be loaded")):
            out, events = self.update(SHORT)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        self.assertEqual(self.resolved(), "0.1.46")
        said = [e.get("message", "") for e in events]
        self.assertIn("Only 20 GB of memory is free right now and fake needs about 58 GB, so the new engine was checked without loading the model. "
                      "It loads your model the next time it starts.", said)
        u = self.engine_state()["unproven"]
        self.assertEqual((u["version"], u["previous"]), ("0.1.46", f"0.1.45-{self.platform}-{self.tag}"))
        self.assertEqual(updater.State(self.components).data["history"][-1]["detail"], updater.LOW_MEMORY_MODE)

    def test_enough_memory_is_the_full_test_and_nothing_is_unproven(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        out, _events = self.update(None)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        self.assertNotIn("unproven", self.engine_state())
        self.assertEqual(updater.State(self.components).data["history"][-1]["detail"], "full")

    def test_a_failed_first_start_goes_back_once_and_the_version_waits_for_the_person(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.object(runtime, "start", side_effect=AssertionError("no test load")):
            self.update(SHORT)
        self.assertIsNotNone(updater.unproven_in_use(self.components, self.hardware))
        back = updater.fall_back_from_unproven(self.components, hardware=self.hardware, serving=lambda: [])
        self.assertEqual((back["from"], back["to"]), ("0.1.46", "0.1.45"))
        self.assertEqual(back["message"], "The new engine 0.1.46 could not start your model, so Strata went back to engine 0.1.45. "
                                          "Your settings are unchanged; the update will be offered again.")
        self.assertEqual(self.resolved(), "0.1.45")
        st = self.engine_state()
        self.assertNotIn("unproven", st)
        self.assertIn("0.1.46", st["failed_unproven"])
        self.assertNotIn("0.1.46", st.get("skipped", []), "not hidden from the update")
        self.assertIsNone(updater.fall_back_from_unproven(self.components, hardware=self.hardware, serving=lambda: []), "once: nothing more to do")
        # the next update offers it, but does not switch to it by itself again (no loop)
        again, _ = self.update(SHORT)
        self.assertEqual(again["results"][0]["status"], "failed-before", again)
        self.assertIn("could not start your model after the last update", again["results"][0].get("note", "") + json.dumps(again))
        self.assertEqual(self.resolved(), "0.1.45")
        # the person asks for it: it is used, and still unproven (a failed first start would go back once more)
        forced, _ = self.update(SHORT, force=True)
        self.assertEqual(self.resolved(), "0.1.46", forced)
        self.assertEqual(self.engine_state()["unproven"]["version"], "0.1.46")

    def test_a_ready_start_proves_it(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.object(runtime, "start", side_effect=AssertionError("no test load")):
            self.update(SHORT)
        self.assertTrue(updater.mark_proven(self.components, self.hardware))
        self.assertNotIn("unproven", self.engine_state())
        self.assertIsNone(updater.fall_back_from_unproven(self.components, hardware=self.hardware, serving=lambda: []))
        self.assertEqual(self.resolved(), "0.1.46")

    def test_an_engine_the_person_chose_is_never_unproven(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.object(runtime, "start", side_effect=AssertionError("no test load")):
            self.update(SHORT)
        updater.rollback_engine(self.ctx())                 # the person goes back by hand (components rollback / use)
        self.assertNotIn("unproven", self.engine_state())
        self.assertIsNone(updater.unproven_in_use(self.components, self.hardware))

    def test_no_model_installed_is_unchanged(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        check = mock.Mock(return_value=SHORT)
        with mock.patch.object(updater, "discover_run_config", return_value=None):
            out = updater.run_update(self.ctx(run_config=None, memory_check=check), control=False)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        check.assert_not_called()
        self.assertNotIn("unproven", self.engine_state())


class MemoryRule(unittest.TestCase):
    def test_unknown_memory_or_need_is_the_full_test(self):
        with mock.patch.object(updater, "free_ram_gb", return_value=None), mock.patch.object(updater, "test_ram_need_gb", return_value=58.0):
            self.assertIsNone(updater.low_memory({}))
        with mock.patch.object(updater, "free_ram_gb", return_value=10.0), mock.patch.object(updater, "test_ram_need_gb", return_value=None):
            self.assertIsNone(updater.low_memory({}))
        with mock.patch.object(updater, "free_ram_gb", return_value=60.0), mock.patch.object(updater, "test_ram_need_gb", return_value=58.0):
            self.assertIsNone(updater.low_memory({}))
        with mock.patch.object(updater, "free_ram_gb", return_value=20.0), mock.patch.object(updater, "test_ram_need_gb", return_value=58.0):
            self.assertEqual(updater.low_memory({}), (20.0, 58.0))

    def test_the_need_is_the_test_starts_own_plan(self):
        table = {"IQ3_S": {"ram_gb": 62, "arena_gb": 50.3}, "Q2_0": {"ram_gb": 48}}
        with mock.patch.object(updater, "_setup_models", return_value=table):
            iq3 = {"model_name": "qwen3.8-flash-next-iq3_s", "args": ["--expert-cache", "auto", "--max-context", "4096"]}
            self.assertAlmostEqual(updater.test_ram_need_gb(iq3), 50.3 + 5.3 + 4.0)   # every expert in RAM: the arena + the engine base + 4 GB
            self.assertAlmostEqual(updater.test_ram_need_gb({**iq3, "args": ["--mmap-experts"]}), 9.3)      # mapped: the OS cache
            self.assertAlmostEqual(updater.test_ram_need_gb({**iq3, "args": ["--resident-experts"]}), 9.3)  # copied only when they fit
            self.assertAlmostEqual(updater.test_ram_need_gb({**iq3, "args": ["--resident-budget-gib", "10"]}), 10 * 1.073741824 + 9.3)
            self.assertEqual(updater.test_ram_need_gb({"model_name": "qwen3.8-flash-next-q2_0", "args": []}), 44.0)   # no arena figure: the PC figure
            self.assertIsNone(updater.test_ram_need_gb({"model_name": "something-else", "args": []}))
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as tmp:
                ple = Path(tmp) / "ple.gguf"
                ple.write_bytes(b"x" * 2_000_000)
                held = {**iq3, "args": ["--ple-gguf", str(ple), "--ple-io", "ram"]}
                self.assertAlmostEqual(updater.test_ram_need_gb(held), 50.3 + 5.3 + 4.0 + 0.002)        # the PLE table when it stays in RAM
                self.assertAlmostEqual(updater.test_ram_need_gb({**held, "args": updater.light_test_args(held["args"])}), 50.3 + 9.3)

    def test_the_light_test_reads_the_ple_table_from_the_file(self):
        self.assertEqual(updater.light_test_args(["--ple-gguf", "p", "--ple-io", "ram", "--max-context", "131072"]),
                         ["--ple-gguf", "p", "--ple-io", "direct", "--max-context", "4096"])
        self.assertEqual(updater.TEST_ENV, {"STRATA_PLE_AUTO_RAM": "0"}, "the engine's own choice of ram is off in the test start")
        self.assertEqual(updater.light_test_args(["--ple-io", "mmap"]), ["--ple-io", "mmap"])


class StartFallsBack(unittest.TestCase):
    """runtime.start (the Terminal's launch and the app's `strata runtime start` both come here)."""

    def run_start(self, results, *, unproven=True, back=True):
        calls = []

        def once(name, **kw):
            calls.append(kw["wait"])
            return dict(results[len(calls) - 1])
        fall = {"from": "0.1.46", "to": "0.1.45", "message": "went back"} if back else None
        with mock.patch.object(runtime, "_start_once", side_effect=once), \
                mock.patch.object(runtime, "stop", return_value={"ok": True}), \
                mock.patch.object(updater, "unproven_in_use", return_value={"version": "0.1.46"} if unproven else None), \
                mock.patch.object(updater, "fall_back_from_unproven", return_value=fall) as fb, \
                mock.patch.object(updater, "mark_proven") as proven:
            out = runtime.start("p", wait=300)
        return out, calls, fb, proven

    FAILED = {"ok": False, "state": "failed", "error": "server_exited", "log_tail": "engine stopped: illegal instruction", "exit_code": 6}
    READY = {"ok": True, "state": "ready", "exit_code": 0}

    def test_an_engine_failure_goes_back_and_starts_once_more(self):
        out, calls, fb, _p = self.run_start([self.FAILED, self.READY])
        self.assertEqual(len(calls), 2)
        self.assertEqual((out["state"], out["note"], out["engine_fallback"]["to"]), ("ready", "went back", "0.1.45"))
        fb.assert_called_once()

    def test_never_a_loop(self):
        out, calls, fb, _p = self.run_start([self.FAILED, self.FAILED])
        self.assertEqual(len(calls), 2, "one switch back, one more start, then the failure is the answer")
        self.assertEqual(out["state"], "failed")
        fb.assert_called_once()

    def test_a_refusal_never_switches(self):
        for tail in ("CUDA error: out of memory", "cudaErrorMemoryAllocation", "address already in use"):
            out, calls, fb, _p = self.run_start([{**self.FAILED, "log_tail": tail}])
            self.assertEqual(len(calls), 1, tail)
            fb.assert_not_called()

    def test_a_proven_engine_and_a_wait_that_ran_out_never_switch(self):
        out, calls, fb, _p = self.run_start([self.FAILED], unproven=False)
        self.assertEqual((len(calls), out["state"]), (1, "failed"))
        fb.assert_not_called()
        out, calls, fb, _p = self.run_start([{"ok": False, "state": "starting", "error": "startup_wait_expired", "exit_code": 8}])
        self.assertEqual(len(calls), 1)
        fb.assert_not_called()

    def test_ready_proves_it(self):
        out, calls, fb, proven = self.run_start([self.READY])
        proven.assert_called_once()
        fb.assert_not_called()

    def test_the_second_start_keeps_inside_the_first_wait(self):
        out, calls, _fb, _p = self.run_start([self.FAILED, self.READY])
        self.assertLessEqual(calls[1], 300)


class TheAppsStartPath(unittest.TestCase):
    """The app starts a model with `strata runtime start --profile P --wait N --output json` (app/host/server/control-runtime.mjs): the same
    runtime.start, and its JSON carries the fallback and its line."""

    def test_runtime_start_from_the_command_line(self):
        import tempfile
        from pathlib import Path
        from strata_control import profiles
        from tests.terminal.launch_support import Home, run_cli
        results = [StartFallsBack.FAILED, StartFallsBack.READY]
        calls = []

        def once(name, **kw):
            calls.append(name)
            return dict(results[len(calls) - 1], profile=name)
        with Home(), tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "run.json"
            (Path(tmp) / "tok").mkdir()
            cfg.write_text(json.dumps({"exe": "/nowhere/strata", "args": [], "model_name": "fake", "port": 18555, "cwd": tmp,
                                       "tokenizer": str(Path(tmp) / "tok")}), encoding="utf-8")
            profiles.create("appslot", str(cfg))
            back = {"from": "0.1.46", "to": "0.1.45", "message": "went back"}
            with mock.patch.object(runtime, "_start_once", side_effect=once), mock.patch.object(runtime, "stop", return_value={"ok": True}), \
                    mock.patch.object(updater, "unproven_in_use", return_value={"version": "0.1.46"}), \
                    mock.patch.object(updater, "fall_back_from_unproven", return_value=back), mock.patch.object(updater, "mark_proven"):
                code, out, err = run_cli(["runtime", "start", "--profile", "appslot", "--wait", "60", "--output", "json"])
        self.assertEqual(len(calls), 2, err)
        doc = json.loads(out)
        self.assertEqual((doc["state"], doc["note"], doc["engine_fallback"]["to"]), ("ready", "went back", "0.1.45"))


if __name__ == "__main__":
    unittest.main()
