"""#5078 (what is left after fe4a03e2f): a tune whose engine could not run the current settings

1. a full graphics card is said in graphics-card words and names the one next step (--goal room:2048), not only "the worker refused the request (503)";
2. the log of the engine that failed is kept (last-failed-engine.log in the presets folder) instead of going with the run's scratch folder;
3. the next engine starts only after the previous one gave its graphics memory back (up to 30 s), so a candidate is not judged on memory the last engine still held.
Doubles only."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import tune_commands as TC
from strata_control import autotune, tune_session
from strata_control.tune_session import StartFailed, WorkerSession
from tests.tuning.stub_session import FakeClock, StubSession
from tests.tuning.test_autotune import BASE, make_ctx, speed_with


def base_failed(status, ctx=None):
    def fail(args, env):
        return StartFailed(status, "the worker refused the request (503) | engine: cudaMalloc: out of memory")
    clock = FakeClock()
    session = StubSession(model=speed_with({}), clock=clock, fail=fail)
    t = autotune.Tuner(session, BASE, {}, ctx or make_ctx(), autotune.Options(), clock=clock, sleep=clock.sleep, system=session.system,
                       total_ram_mib=64000, total_vram_mib=12000, quiet=lambda: None)
    return t.run()


class FullCardIsSaidAsSuch(unittest.TestCase):
    def test_an_oom_on_the_base_names_the_card_and_the_next_step(self):
        r = base_failed("OOM")
        self.assertEqual(r.outcome, "base_failed")
        self.assertIn("cudaMalloc", r.message)
        self.assertTrue(r.message.endswith(autotune.GRAPHICS_MEMORY_HINT))
        self.assertIn("--goal room:2048", r.message)

    def test_other_failures_and_other_cards_keep_the_old_message(self):
        self.assertNotIn(autotune.GRAPHICS_MEMORY_HINT, base_failed("NO_TOKENS").message)
        ctx = make_ctx()
        ctx.backend = "sycl"
        self.assertNotIn(autotune.GRAPHICS_MEMORY_HINT, base_failed("OOM", ctx).message)


class FailedLogIsKept(unittest.TestCase):
    def test_the_failed_engines_log_is_copied_to_the_presets_folder(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as presets_dir:
            log = Path(tmp) / "engine-2.log"
            log.write_text("loading\nggml-cuda: out of memory: cudaMalloc\n")
            session = mock.Mock(failed_log=log)
            with mock.patch.dict(os.environ, {"STRATA_PRESET_DIR": presets_dir}):
                kept = TC._keep_failed_log(session)
            self.assertEqual(kept, Path(presets_dir) / "last-failed-engine.log")
            self.assertIn("cudaMalloc", kept.read_text())
            if os.name != "nt":
                self.assertEqual(kept.stat().st_mode & 0o777, 0o600)
            else:
                from strata_control import private_files
                self.assertTrue(private_files.is_private(kept))

    def test_no_failed_engine_keeps_nothing(self):
        self.assertIsNone(TC._keep_failed_log(mock.Mock(failed_log=None)))

    def test_the_session_remembers_the_log_of_an_engine_that_did_not_start(self):
        def popen(argv, name, **kw):
            raise OSError("not started in this test")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("serve.lifeline.popen", popen), \
                mock.patch.object(tune_session.tune_machine, "gpu_used_mib", return_value=None):
            s = WorkerSession(base_config={"args": []}, serve_dir=tmp, workdir=tmp)
            with self.assertRaises(StartFailed):
                with s.engine([], {}):
                    pass
            self.assertIsNone(s.failed_log, "an engine that never spawned has no log to keep")

    def test_a_failure_while_the_engine_runs_is_remembered(self):
        class Proc:
            pid = 0
            returncode = 0

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0
        with tempfile.TemporaryDirectory() as tmp, mock.patch("serve.lifeline.popen", return_value=Proc()), \
                mock.patch.object(tune_session.tune_machine, "gpu_used_mib", return_value=None), \
                mock.patch.object(tune_session, "WorkerEngine", return_value=object()), \
                mock.patch.object(tune_session.loopback, "urlopen") as up, mock.patch.object(tune_session.time, "sleep"):
            up.return_value.__enter__.return_value.status = 200
            s = WorkerSession(base_config={"args": []}, serve_dir=tmp, workdir=tmp)
            with self.assertRaises(StartFailed):
                with s.engine([], {}):
                    raise StartFailed("OOM", "x")
            self.assertEqual(s.failed_log, Path(tmp) / "engine-1.log")


class GraphicsMemoryComesBack(unittest.TestCase):
    def session(self):
        with tempfile.TemporaryDirectory() as tmp:
            return WorkerSession(base_config={"args": []}, serve_dir=tmp, workdir=tmp)

    def wait(self, used, before=1000):
        s = self.session()
        seen = iter(used)
        clock = [0.0]

        def sleep(n):
            clock[0] += n
        with mock.patch.object(tune_session.tune_machine, "gpu_used_mib", side_effect=lambda: next(seen)) as g, \
                mock.patch.object(tune_session.time, "sleep", side_effect=sleep), \
                mock.patch.object(tune_session.time, "monotonic", side_effect=lambda: clock[0]):
            s._wait_graphics_memory_back(before)
        return g.call_count, clock[0]

    def test_it_waits_until_the_memory_is_back_and_no_longer(self):
        calls, waited = self.wait([9000, 9000, 4000, 1100])
        self.assertEqual((calls, waited), (4, 3.0))

    def test_it_is_back_at_once_when_the_card_is_where_it_was(self):
        self.assertEqual(self.wait([1100]), (1, 0.0))

    def test_it_gives_up_after_thirty_seconds(self):
        calls, waited = self.wait([9000] * 100)
        self.assertEqual(waited, WorkerSession.GRAPHICS_BACK_WAIT_S)
        self.assertLess(calls, 40)

    def test_no_card_reading_or_no_mark_means_no_wait(self):
        self.assertEqual(self.wait([None]), (1, 0.0))
        self.assertEqual(self.wait([], before=None), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
