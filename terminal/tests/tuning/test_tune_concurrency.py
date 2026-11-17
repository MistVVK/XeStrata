"""#4821 (opt-in): `strata tune auto --concurrency N` judges every setting on N requests decoding at once, not on one request at a time.

The solo tune is untouched (clients 1, no loaded phase, the same metric). With N > 1 the engine run gets a loaded phase (all requests' tokens per second,
median of three), the metric becomes that figure, and the report names N. Doubles only: a stub session and a tiny threaded HTTP worker."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_control import autotune
from strata_control.tune_session import RequestSet, StartFailed, WorkerEngine, WorkerSession
from tests.tuning.stub_session import FakeClock, StubSession, flag
from tests.tuning.test_autotune import BASE, make_ctx, speed_with


class Worker(BaseHTTPRequestHandler):
    delay = 0.2

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
        time.sleep(self.delay)
        body = json.dumps({"timings": {"predicted_n": 200, "predicted_per_second": 1000.0 / self.delay}}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeProc:
    pid = 0

    def poll(self):
        return None


class LoadedPhase(unittest.TestCase):
    def engine(self, log_text="loading\n"):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "engine.log"
        log.write_text(log_text)
        session = WorkerSession(base_config={"args": []}, serve_dir=".", workdir=tmp.name)
        return WorkerEngine(session, FakeProc(), srv.server_address[1], "k", [log], 1.0, None, lambda x: None)

    def test_the_requests_run_together_and_all_their_tokens_count(self):
        rate = self.engine().load_once(3, 200)
        self.assertGreater(rate, 1800, "3 x 200 tokens in about 0.2 s, not one after the other (that would be about 1000 per second)")
        self.assertLess(rate, 3100)

    def test_a_request_that_gave_too_little_makes_the_run_unmeasurable(self):
        eng = self.engine()
        self.assertIsNone(eng.load_once(2, 500), "200 tokens of 500 asked is below half: not a measurement")

    def test_a_solo_set_has_no_loaded_phase_and_a_loaded_one_has(self):
        eng = self.engine()
        solo = eng.measure(RequestSet(warmups=0, decodes=1, decode_tokens=64, pp_tokens=0))
        self.assertIsNone(solo.load_tok_s)
        loaded = eng.measure(RequestSet(warmups=0, decodes=1, decode_tokens=64, pp_tokens=0, clients=3, load_rounds=2))
        self.assertGreater(loaded.load_tok_s, 100)
        self.assertEqual(len(loaded.load_runs), 2)

    def test_an_engine_with_fewer_slots_than_asked_is_refused_with_the_number(self):
        eng = self.engine("INFO engine=0.1.60 batch_slots=2\n")
        eng.facts = {}
        with self.assertRaises(StartFailed) as cm:
            eng.measure(RequestSet(warmups=0, decodes=1, decode_tokens=64, pp_tokens=0, clients=4, load_rounds=1))
        self.assertIn("runs 2 requests at once, not the 4 asked for", cm.exception.reason)


class JudgedOnTheLoad(unittest.TestCase):
    def test_the_metric_is_the_loaded_rate_when_there_is_one(self):
        self.assertEqual(autotune.metric({"decode_tok_s": 60.0, "load_tok_s": 180.0}, "chat"), 180.0)
        self.assertEqual(autotune.metric({"decode_tok_s": 60.0, "load_tok_s": 180.0}, "coding"), 180.0)
        self.assertEqual(autotune.metric({"decode_tok_s": 60.0}, "chat"), 60.0)
        self.assertEqual(autotune.metric({"decode_tok_s": 60.0, "load_tok_s": None, "pp_tok_s": 1500.0}, "chat"), 60.0)
        self.assertEqual(autotune.metric({"decode_tok_s": 60.0, "load_tok_s": 180.0, "pp_tok_s": 1500.0}, "prompt"), 1500.0)

    def tune(self, concurrency):
        # --kv-resident 65536 is the same solo and 12 % faster with 4 requests at once (the issue's point: a solo run cannot see it)
        solo = speed_with({})
        load = lambda args, env: 3.0 * 60.0 * (1.12 if flag(args, "--kv-resident") == "65536" else 1.0)  # noqa: E731
        clock = FakeClock()
        session = StubSession(model=solo, clock=clock, load=load)
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(concurrency=concurrency), clock=clock, sleep=clock.sleep, system=session.system,
                           total_ram_mib=64000, total_vram_mib=12000, quiet=lambda: None)
        return t.run(), session

    def test_a_setting_only_the_load_shows_is_taken_with_concurrency_and_not_without(self):
        solo_r, _s = self.tune(1)
        self.assertNotEqual(solo_r.outcome, "improved")
        self.assertNotIn("--kv-resident", solo_r.settings["args"])
        loaded_r, _s = self.tune(4)
        self.assertEqual(loaded_r.outcome, "improved")
        self.assertEqual(loaded_r.settings["args"].get("--kv-resident"), 65536)
        self.assertGreater(loaded_r.gain_pct, 8.0)
        self.assertEqual(loaded_r.concurrency, 4)
        self.assertGreater(loaded_r.base["load_tok_s"], 0)

    def test_the_solo_result_is_as_it_was(self):
        r, _s = self.tune(1)
        self.assertEqual(r.concurrency, 1)
        self.assertIsNone(r.base.get("load_tok_s"))


class TheCommand(unittest.TestCase):
    def run_auto(self, *extra, tuner_result=None):
        import argparse
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from strata_cli import tune_commands as TC
        from tests.tuning.test_autotune import run_tuner
        res = tuner_result or run_tuner(speed_with({}), ctx=None)[2]
        seen = {}

        class FakeTuner(autotune.Tuner):
            def __init__(self, session, base_args, env, ctx, opts, **kw):
                self.result = res
                seen["opts"] = opts

            def run(self):
                return res

        def session(**kw):
            seen["session"] = kw
            return mock.Mock(failed_log=None)

        def create(result, *, choices, **kw):
            seen["choices"] = dict(choices)
            return {"id": "i", "name": "n"}
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            (t / "engine.exe").write_bytes(b"x")
            (t / "model.gguf").write_bytes(b"gguf" * 100)
            (t / "tok").mkdir()
            cfg = t / "strata-model.json"
            cfg.write_text(json.dumps({"exe": str(t / "engine.exe"), "tokenizer": str(t / "tok"), "model_name": "iq3_s-test",
                                       "args": ["--native", str(t / "model.gguf"), "--max-context", "65536"]}))
            ap = argparse.ArgumentParser()
            TC.add_tune_auto(ap.add_subparsers(dest="c"))
            a = ap.parse_args(["auto", str(cfg), "--work-dir", str(t / "work"), *extra])
            out, err = io.StringIO(), io.StringIO()
            import os
            with mock.patch.dict(os.environ, {"STRATA_PRESET_DIR": str(t / "presets")}), \
                    mock.patch("strata_control.tuning.engine_flag_inventory", return_value=None), \
                    mock.patch("strata_control.tune_session.WorkerSession", side_effect=session), mock.patch.object(autotune, "Tuner", FakeTuner), \
                    mock.patch("strata_control.presets.create", side_effect=create), redirect_stdout(out), redirect_stderr(err):
                code = TC.auto(a)
        return code, out.getvalue(), err.getvalue(), seen

    def test_a_number_outside_two_to_eight_is_refused_in_plain_words(self):
        for n in ("0", "9", "-3"):
            code, _out, err, seen = self.run_auto("--concurrency", n)
            self.assertEqual(code, 2, n)
            self.assertIn("--concurrency is a number of requests from 2 to 8", err)
            self.assertNotIn("session", seen, "nothing was started")

    def test_without_the_flag_nothing_is_different(self):
        code, _out, _err, seen = self.run_auto()
        self.assertEqual(code, 0)
        self.assertEqual(seen["opts"].concurrency, 1)
        self.assertNotIn("parallel", seen["session"]["base_config"])
        self.assertNotIn("concurrency", seen["choices"])

    def test_with_the_flag_the_worker_gets_the_slots_and_the_preset_records_it(self):
        _t, _s, res = autotune_result_improved()
        res.concurrency = 4                                  # (what a real Tuner records from its Options)
        code, out, _err, seen = self.run_auto("--concurrency", "4", tuner_result=res)
        self.assertEqual(code, 0)
        self.assertEqual(seen["opts"].concurrency, 4)
        self.assertEqual(seen["session"]["base_config"]["parallel"], 4)
        self.assertEqual(seen["choices"]["concurrency"], 4)
        self.assertIn("Measured with 4 requests at once.", out)


def autotune_result_improved():
    from tests.tuning.test_autotune import run_tuner
    return run_tuner(speed_with({("--kv-resident", "65536"): 1.10}))


class TheAccountNamesIt(unittest.TestCase):
    def test_the_line_comes_after_the_solo_one(self):
        from strata_control import tune_report
        r = {"outcome": "default_confirmed", "concurrency": 4, "base": {"decode_tok_s": 60.0, "load_tok_s": 180.0},
             "noise": {"threshold_pct": 3.0, "aa_gap": 0.001, "base_runs": [179.0, 180.0, 181.0]}, "arms": [], "skipped": []}
        lines = tune_report.build(r).splitlines()
        self.assertEqual(lines[0], "Your settings: answers at 60.0 tok/s")
        self.assertEqual(lines[1], "With 4 requests at once: 180.0 tok/s in all (measured 3 times: 179, 180, 181)")
        self.assertTrue(lines[2].startswith("To count, a setting had to be at least +3.0% faster"))
        self.assertNotIn("requests at once", tune_report.build({**r, "concurrency": 1}))


class AppliedToTheRightStart(unittest.TestCase):
    """A preset tuned on N requests at once is applied to starts that run requests together, not to a one-request start (where its solo predecessor,
    if any, stands in); a solo preset behaves as before on every start."""

    def setUp(self):
        import copy
        import os
        from strata_control import presets
        from tests.tuning.test_presets import CFG, RESULT, identity
        self.presets, self.CFG, self.RESULT, self.identity, self.copy = presets, CFG, RESULT, identity, copy
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["STRATA_PRESET_DIR"] = str(Path(self.tmp.name) / "presets")
        self.addCleanup(lambda: os.environ.pop("STRATA_PRESET_DIR", None))
        self.addCleanup(self.tmp.cleanup)
        self.choices = presets.choices_from_config(CFG)

    def save(self, args, **choices):
        res = self.copy.deepcopy(self.RESULT)
        res["settings"]["args"] = args
        return self.presets.create(res, choices={**self.choices, **choices}, identity=self.identity())

    def at(self, cfg):
        out, info = self.presets.overlay(cfg, now=self.identity())
        return out["args"], info

    def parallel(self, n):
        cfg = self.copy.deepcopy(self.CFG)
        cfg["parallel"] = n
        return cfg

    def test_how_many_requests_a_start_runs(self):
        batch = self.copy.deepcopy(self.CFG)
        batch["args"] += ["--batch", "4"]
        self.assertEqual([self.presets.start_requests(c) for c in (self.CFG, self.parallel(1), self.parallel(4), batch)], [1, 1, 4, 4])

    def test_a_load_tuned_preset_is_skipped_on_a_solo_start_and_its_solo_predecessor_stands_in(self):
        self.save({"--host-core": "last"})
        self.save({"--kv-resident": 65536}, concurrency=4)
        args, info = self.at(self.CFG)
        self.assertTrue(info["applied"])
        self.assertIn("--host-core", args)
        self.assertNotIn("--kv-resident", args)

    def test_it_is_applied_where_requests_run_together(self):
        self.save({"--host-core": "last"})
        self.save({"--kv-resident": 65536}, concurrency=4)
        for cfg in (self.parallel(4), self.parallel(2)):
            args, info = self.at(cfg)
            self.assertTrue(info["applied"], cfg.get("parallel"))
            self.assertIn("--kv-resident", args)
            self.assertNotIn("--host-core", args)

    def test_with_no_solo_predecessor_a_solo_start_gets_nothing_and_is_told_why(self):
        self.save({"--kv-resident": 65536}, concurrency=4)
        args, info = self.at(self.CFG)
        self.assertFalse(info["applied"])
        self.assertEqual(info["why_not"], "not applied: it was tuned for 4 requests at once, and this start runs one request at a time")
        self.assertEqual(args, self.CFG["args"])
        self.assertTrue(self.at(self.parallel(4))[1]["applied"])

    def test_a_solo_preset_is_applied_to_every_start_as_before(self):
        self.save({"--host-core": "last"})
        for cfg in (self.CFG, self.parallel(1), self.parallel(4)):
            args, info = self.at(cfg)
            self.assertTrue(info["applied"])
            self.assertIn("--host-core", args)

    def test_effective_says_so_in_words(self):
        import io
        from contextlib import redirect_stdout
        from strata_cli import tune_commands as TC
        why = "not applied: it was tuned for 4 requests at once, and this start runs one request at a time"
        out = io.StringIO()
        with mock.patch.object(TC, "_load_target", return_value=(Path("x.json"), self.CFG)), \
                mock.patch.object(self.presets, "overlay", return_value=(self.CFG, {"applied": False, "why_not": why})), redirect_stdout(out):
            TC._effective(type("A", (), {"target": "x.json"})(), False)
        self.assertIn(why, out.getvalue())
        self.assertIn("No saved tuning is applied; the standard settings run.", out.getvalue())


if __name__ == "__main__":
    unittest.main()
