"""#5232: the account of a tune (strata_control/tune_report.py), the file it is saved in, `strata tune report`, and the run counts in the progress lines.
Doubles only."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import autotune, tune_report
from tests.tuning.stub_session import FakeClock, StubSession
from tests.tuning.test_autotune import BASE, make_ctx, run_tuner, speed_with

ROOT = Path(__file__).resolve().parents[2]


def result(**kw):
    r = {"outcome": "improved", "message": "", "base": {"decode_tok_s": 60.0, "pp_tok_s": 1500.0},
         "noise": {"base_runs": [60.0, 60.04, 59.98], "aa_gap": 0.001, "threshold_pct": 3.0},
         "arms": [{"id": "kv", "label": "Keep more of the conversation in graphics memory", "status": "tested", "ratio_decode": 1.10, "pairs": 2,
                   "faster": 2, "won": True, "reason": ""},
                  {"id": "pf", "label": "Bigger prompt chunks", "status": "tested", "ratio_decode": 1.012, "pairs": 2, "faster": 1, "won": False, "reason": ""},
                  {"id": "sp", "label": "Longer drafts", "status": "dropped", "ratio_decode": 0.9, "pairs": 1, "faster": 0, "won": False, "reason": ""},
                  {"id": "x", "label": "Another setting", "status": "failed", "pairs": 0, "faster": 0, "won": False, "reason": "the engine stopped"}],
         "skipped": [{"label": "Faster but different answers", "reason": "it can change the answers; allowed with --allow-small-differences"}]}
    r.update(kw)
    return r


class Account(unittest.TestCase):
    def test_the_account_has_every_part(self):
        text = tune_report.build(result(), "IQ3_XXS 128K")
        self.assertEqual(text.splitlines(), [
            "Your settings: answers at 60.0 tok/s, prompts read at 1500.0 tok/s (measured 3 times: 60, 60.04, 59.98)",
            "To count, a setting had to be at least +3.0% faster; identical runs on this PC differed by 0.1%.",
            "What was tried:",
            "  Keep more of the conversation in graphics memory  faster, +10.0% against your settings, faster in 2 of 2 rounds",
            "  Bigger prompt chunks  not faster by enough: +1.2% against your settings, faster in 1 of 2 rounds",
            "  Longer drafts  slower, stopped after one round",
            "  Another setting  could not run: the engine stopped",
            "Left out:",
            "  Faster but different answers: it can change the answers; allowed with --allow-small-differences",
            "Next: 'IQ3_XXS 128K' is applied at every start. See it with 'strata presets show', or go back with 'strata presets undo'."])

    def test_one_next_step_for_every_way_a_tune_ends(self):
        said = {o: tune_report.build(result(outcome=o, arms=[], skipped=[]), None).splitlines()[-1] for o in
                ("improved", "default_confirmed", "inconclusive", "base_failed", "cancelled")}
        self.assertEqual(said["default_confirmed"], "Next: Nothing was changed. Your settings are already the fastest of what was tried.")
        self.assertEqual(said["inconclusive"], "Next: Nothing was changed. Close other programs and tune again.")
        self.assertEqual(said["base_failed"], "Next: Nothing was changed. Fix the problem above and tune again.")
        self.assertEqual(said["cancelled"], "Next: Nothing was changed.")
        self.assertTrue(said["improved"].startswith("Next: The faster settings were not saved."))
        self.assertEqual(len(set(said.values())), 5)

    def test_a_failed_baseline_says_why_and_holds_no_paths_or_keys(self):
        text = tune_report.build(result(outcome="base_failed", base={}, noise={}, arms=[], skipped=[],
                                        message="Your current settings could not be measured: the engine stopped"))
        self.assertIn("could not be measured", text)
        self.assertNotIn("args", text.lower())


class SavedAndShown(unittest.TestCase):
    def test_saved_privately_and_read_again(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"STRATA_PRESET_DIR": d}):
            self.assertIsNone(tune_report.read())
            path = tune_report.save("an account\n")
            self.assertEqual(path, Path(d) / "last-tune-report.txt")
            self.assertEqual(tune_report.read(), "an account\n")
            tune_report.save("a newer one\n")
            self.assertEqual(tune_report.read(), "a newer one\n")
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            else:
                from strata_control import private_files
                self.assertTrue(private_files.is_private(path))

    def test_the_command_prints_it_or_says_no_tune_has_been_run(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, STRATA_PRESET_DIR=d, PYTHONPATH=str(ROOT))
            run = lambda: subprocess.run([sys.executable, str(ROOT / "strata.py"), "tune", "report"], capture_output=True, text=True, env=env,  # noqa: E731
                                         cwd=str(ROOT), timeout=120)
            first = run()
            self.assertEqual((first.returncode, first.stdout.strip()), (0, "No tune has been run yet."))
            as_json = subprocess.run([sys.executable, str(ROOT / "strata.py"), "tune", "report", "--output", "json"], capture_output=True, text=True,
                                     env=env, cwd=str(ROOT), timeout=120)
            self.assertIn('"report": null', as_json.stdout)
            (Path(d) / "last-tune-report.txt").write_text("Your settings: x\n", encoding="utf-8")
            self.assertEqual(run().stdout, "Your settings: x\n")


class AutoKeepsTheAccount(unittest.TestCase):
    """`strata tune auto` saves the account after the run and says where to read it; the JSON answer is not given the extra line."""

    def run_auto(self, *extra, outcome_stub=None):
        import argparse
        import io
        import json
        from contextlib import redirect_stdout
        from strata_cli import tune_commands as TC
        _t, _s, res = run_tuner(speed_with({("--kv-resident", "65536"): 1.10}))

        class FakeTuner(autotune.Tuner):
            def __init__(self, *a, **kw):
                self.result = res

            def run(self):
                return res
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            (t / "engine.exe").write_bytes(b"x")
            (t / "model.gguf").write_bytes(b"gguf" * 100)
            (t / "tok").mkdir()
            cfg = t / "strata-model.json"
            cfg.write_text(json.dumps({"exe": str(t / "engine.exe"), "tokenizer": str(t / "tok"), "model_name": "iq3_s-test",
                                       "args": ["--native", str(t / "model.gguf"), "--max-context", "65536", "--kv-resident", "32768"]}))
            ap = argparse.ArgumentParser()
            TC.add_tune_auto(ap.add_subparsers(dest="c"))
            a = ap.parse_args(["auto", str(cfg), "--no-save", "--work-dir", str(t / "work"), *extra])
            out = io.StringIO()
            with mock.patch.dict(os.environ, {"STRATA_PRESET_DIR": str(t / "presets")}), \
                    mock.patch("strata_control.tuning.engine_flag_inventory", return_value=None), \
                    mock.patch("strata_control.tune_session.WorkerSession", return_value=mock.Mock(failed_log=None)), \
                    mock.patch.object(autotune, "Tuner", FakeTuner), \
                    redirect_stdout(out):
                code = TC.auto(a)
            saved = (t / "presets" / "last-tune-report.txt")
            return code, out.getvalue(), (saved.read_text(encoding="utf-8") if saved.exists() else None)

    def test_the_account_is_saved_and_the_answer_points_to_it(self):
        code, said, saved = self.run_auto()
        self.assertEqual(code, 0)
        self.assertIn("Your settings: answers at ", saved)
        self.assertIn("What was tried:", saved)
        self.assertTrue(said.rstrip().endswith("Why each setting was or was not chosen: strata tune report"))

    def test_the_json_answer_stays_as_it_was(self):
        import json
        code, said, saved = self.run_auto("--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(saved)
        self.assertTrue(json.loads(said)["ok"])
        self.assertNotIn("strata tune report", said)


class Progress(unittest.TestCase):
    def test_the_noise_carries_the_bar_a_setting_had_to_clear(self):
        _t, _s, r = run_tuner(speed_with({("--kv-resident", "65536"): 1.10}))
        self.assertEqual(r.noise["threshold_pct"], round((autotune.accept_threshold(0.03, r.noise["aa_gap"], r.noise["drift"]) - 1.0) * 100, 1))
        self.assertGreaterEqual(r.noise["threshold_pct"], 3.0)

    def test_progress_lines_say_which_run_of_how_many(self):
        from strata_cli import tune_commands as TC
        self.assertEqual(TC.run_words({"runs_done": 2, "runs_planned": 9}), "run 3 of 9")
        self.assertEqual(TC.run_words({"runs_done": 9, "runs_planned": 9}), "run 9 of 9")
        self.assertEqual(TC.run_words({"runs_done": 0, "runs_planned": 0}), "")
        self.assertEqual(TC.run_words({}), "")


if __name__ == "__main__":
    unittest.main()
