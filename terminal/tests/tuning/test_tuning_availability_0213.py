"""Round 0213 TUICFG, step 6: tuning stays available (#5056 STRATA_API_KEY, #5122 RAM guard, #5078 engine abort shown as a 503).
Doubles only: a stub session, a tiny HTTP worker, a recorded spawn."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from strata_control import autotune, tune_session
from strata_control.tune_session import StartFailed, WorkerEngine, WorkerSession, native_failure
from tests.tuning.stub_session import FakeClock, StubSession, flag
from tests.tuning.test_autotune import BASE, make_ctx, speed_with


class WorkerKey(unittest.TestCase):
    """#5056: the worker must answer to the key the tuner made, whatever STRATA_API_KEY says."""

    def spawned(self, env):
        seen = {}

        def popen(argv, name, **kw):
            seen["env"], seen["argv"] = dict(kw["env"]), argv
            raise OSError("not started in this test")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env, clear=False), \
                mock.patch("serve.lifeline.popen", popen), mock.patch.object(tune_session.tune_machine, "gpu_used_mib", return_value=None):
            if "STRATA_API_KEY" not in env:
                os.environ.pop("STRATA_API_KEY", None)
            s = WorkerSession(base_config={"args": [], "api_key": "from-the-preset"}, serve_dir=tmp, workdir=tmp)
            with self.assertRaises(StartFailed):
                with s.engine([], {}):
                    pass
            cfg = json.loads((Path(tmp) / "run-1.json").read_text())
        return seen, cfg

    def test_the_variable_is_not_passed_to_the_worker_and_the_config_key_is_the_tuners_own(self):
        for env in ({"STRATA_API_KEY": "client-shared-key"}, {"STRATA_API_KEY": ""}, {}):
            with self.subTest(env=env):
                seen, cfg = self.spawned(env)
                self.assertNotIn("STRATA_API_KEY", seen["env"])
                self.assertEqual(len(cfg["api_key"]), 32)
                self.assertNotEqual(cfg["api_key"], "from-the-preset")
                self.assertNotIn(cfg["api_key"], " ".join(seen["argv"]), "the key is not on the command line")


class FailingWorker(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: ANN002
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        body = json.dumps({"error": {"message": "the engine stopped unexpectedly (exit code -6); the next request restarts it"}}).encode()
        self.send_response(503)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeProc:
    pid = 0

    def poll(self):
        return None


class EngineAbort(unittest.TestCase):
    """#5078: a generic 503 from the worker must carry the engine's own last error line, and a memory failure is an OOM."""

    def engine(self, log_text):
        srv = HTTPServer(("127.0.0.1", 0), FailingWorker)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "engine.log"
        log.write_text(log_text)
        session = WorkerSession(base_config={"args": []}, serve_dir=".", workdir=tmp.name)
        return WorkerEngine(session, FakeProc(), srv.server_address[1], "k", [log], 1.0, None, lambda x: None)

    def test_a_cuda_oom_in_prefill_is_reported_as_an_oom_with_the_native_line(self):
        eng = self.engine("strata serve: 687 MiB of VRAM free with everything loaded\n"
                          "ggml-cuda (strata mmq): out of memory: cudaMalloc(&p, size)\n  in alloc at ggml_cuda_host.cu:127\n")
        with self.assertRaises(StartFailed) as cm:
            eng.chat("hello", 8)
        self.assertEqual(cm.exception.status, "OOM")
        self.assertIn("the worker refused the request (503)", cm.exception.reason)
        self.assertIn("out of memory: cudaMalloc", cm.exception.reason)

    def test_another_abort_keeps_its_line_and_is_not_an_oom(self):
        eng = self.engine("loading\nterminate called after throwing an instance of 'std::runtime_error'\n")
        with self.assertRaises(StartFailed) as cm:
            eng.chat("hello", 8)
        self.assertEqual(cm.exception.status, "NO_TOKENS")
        self.assertIn("terminate called", cm.exception.reason)

    def test_a_quiet_log_keeps_the_old_message(self):
        eng = self.engine("loading\nready\n")
        with self.assertRaises(StartFailed) as cm:
            eng.chat("hello", 8)
        self.assertEqual((cm.exception.status, "engine:" in cm.exception.reason), ("NO_TOKENS", False))

    def test_the_memory_guards_information_line_is_not_an_error(self):
        self.assertEqual(native_failure("memory guard: limit 20 GiB, available 30 GiB\nready\n"), "")

    def test_a_crashing_candidate_is_failed_alone_and_the_baseline_is_kept(self):
        def fail(args, env):
            if flag(args, "--kv-resident") == "65536":
                return StartFailed("NO_TOKENS", "the worker refused the request (503) | engine: terminate called")
            return None
        clock = FakeClock()
        session = StubSession(model=speed_with({("--kv-resident", "65536"): 1.10}), clock=clock, fail=fail)
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(), clock=clock, sleep=clock.sleep, system=session.system,
                           total_ram_mib=64000, total_vram_mib=12000, quiet=lambda: None)
        r = t.run()
        failed = [a for a in r.arms if a["status"] == "failed"]
        self.assertTrue(failed and "terminate called" in failed[0]["reason"])
        self.assertNotEqual(r.outcome, "base_failed")
        self.assertNotIn("--kv-resident", r.settings["args"], "the crashed candidate is never applied")
        self.assertIn(r.outcome, ("default_confirmed", "inconclusive"), "the tune ends with a result, not as a failure")

    def test_a_base_that_aborts_names_the_native_failure(self):
        def fail(args, env):
            return StartFailed("OOM", "the worker refused the request (503): b'..' | engine: ggml-cuda: out of memory: cudaMalloc")
        clock = FakeClock()
        session = StubSession(model=speed_with({}), clock=clock, fail=fail)
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(), clock=clock, sleep=clock.sleep, system=session.system,
                           total_ram_mib=64000, total_vram_mib=12000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(r.outcome, "base_failed")
        self.assertIn("cudaMalloc", r.message)


class RamGuard(unittest.TestCase):
    """#5122: the restart candidate needs what the base run had, not the base's peak RSS plus the floor."""

    def tuner(self, free_before, free_after, rss=23429.0, wait_ram_s=30, big_card=False, gain=1.10):
        clock = FakeClock()
        session = StubSession(model=speed_with({("--kv-resident", "65536"): gain}), clock=clock, avail=free_before, rss=lambda a, e: rss)
        real = session.system

        def system():
            out = real()
            out["memory"]["available_mib"] = free_before if not session.runs else free_after
            return out
        ctx = make_ctx()
        if big_card:                     # a 24 GB card also plans the 32K prompt chunk, which asks for 4 GiB more RAM than the base
            ctx.vram_mib = 24000
        t = autotune.Tuner(session, BASE, {}, ctx, autotune.Options(wait_ram_s=wait_ram_s, per_request=False), clock=clock, sleep=clock.sleep,
                           system=system, total_ram_mib=32000, quiet=lambda: None)
        return t, session

    def test_a_candidate_that_adds_nothing_is_tried_when_the_base_started_with_the_same_free_memory(self):
        t, session = self.tuner(free_before=23654, free_after=24680)      # 23.1 GiB at the base start, 24.1 GiB free later (the report)
        r = t.run()
        self.assertTrue(any(flag(a, "--kv-resident") == "65536" for a, _ in session.runs), "the 64K resident-KV candidate (no extra RAM) was started")
        self.assertFalse([a for a in r.arms if a["status"] == "skipped"], r.arms)

    def test_less_free_memory_than_the_base_had_is_still_skipped(self):
        t, session = self.tuner(free_before=23654, free_after=18000)
        r = t.run()
        self.assertFalse(any(flag(a, "--kv-resident") == "65536" for a, _ in session.runs))
        self.assertTrue(any(a["status"] == "skipped" and "memory" in a["reason"] for a in r.arms))

    def test_a_candidate_that_needs_more_than_the_base_asks_for_the_difference(self):
        t, _ = self.tuner(free_before=23654, free_after=24680)
        t.base_runs.append(tune_session.RunResult(status="PASS", rss_peak_mib=23429.0))
        t._base_start_avail = 23654.0
        t.system = lambda: {"memory": {"available_mib": 24680, "total_mib": 32000}}
        self.assertIsNone(t._ram_guard(0, "x", wait_s=0))
        self.assertIn("not enough free memory", t._ram_guard(4096, "x", wait_s=0) or "")     # 23.1 + 4 GiB needed, 24.1 free

    def test_skipped_for_memory_is_said_instead_of_already_the_fastest(self):
        """Owner 11-04: 'N settings could not be tried: not enough free memory', only when some were skipped for memory, in place of 'already the fastest'."""
        t, session = self.tuner(free_before=23654, free_after=24680, big_card=True, gain=1.0)
        r = t.run()
        self.assertEqual([(a["id"], a["status"]) for a in r.arms if a["status"] == "skipped"], [("prefill_32k", "skipped")], r.arms)
        self.assertEqual(r.outcome, "default_confirmed")
        self.assertEqual(r.not_tried, 1)
        self.assertEqual(r.message, "1 setting could not be tried: not enough free memory")
        self.assertNotIn("fastest", r.message)
        self.assertEqual(r.values["not_tried"], 1)
        from strata_cli.launch.runner import tune_words
        word, sentence = tune_words(r.to_dict())
        self.assertEqual(sentence, "1 setting could not be tried: not enough free memory")
        self.assertNotEqual(word, "already fastest")

    def test_the_plural(self):
        from strata_cli.launch.runner import tune_words
        self.assertEqual(tune_words({"outcome": "default_confirmed", "not_tried": 2})[1], "2 settings could not be tried: not enough free memory")

    def test_nothing_skipped_keeps_the_old_sentences(self):
        t, session = self.tuner(free_before=23654, free_after=24680, gain=1.0)
        r = t.run()
        self.assertEqual(r.not_tried, 0)
        self.assertEqual(r.message, "Your current settings are already the fastest we found. Nothing was changed.")
        from strata_cli.launch.runner import tune_words
        self.assertEqual(tune_words(r.to_dict())[0], "already fastest")

    def test_the_old_rule_still_holds_when_the_base_had_plenty(self):
        t, session = self.tuner(free_before=40000, free_after=9000)
        t.run()
        self.assertEqual(len(session.runs), 1, "only the base was started")


if __name__ == "__main__":
    unittest.main()
