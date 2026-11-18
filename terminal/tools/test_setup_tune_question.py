"""P12 (0.2.16): setup's tuning question defaults to No, as the Terminal's Tuning step does on a first launch (FIX-TUI T5).

USERTEST-LIN L2: `./setup.sh` asked "Tune Strata for this PC now? ... [y]": a person pressing Enter waited 15-30 minutes before the model started.

    python -m unittest tools.test_setup_tune_question
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class TuneQuestion(unittest.TestCase):
    def test_the_default_shown_and_taken_is_no(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        i = text.index("ask(TUNE_QUESTION.format(")
        self.assertIn('["y", "n"], "n", a.yes, shown=TUNE_DEFAULT_SHOWN)', text[i:i + 160])
        shown = []
        with mock.patch("builtins.input", side_effect=lambda prompt: shown.append(prompt) or ""), mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            answer = setup.ask(setup.TUNE_QUESTION.format(again="./setup.sh"), ["y", "n"], "n", False, shown=setup.TUNE_DEFAULT_SHOWN)
        self.assertEqual(answer, "n")                                   # Enter alone: no tuning

    def test_f3_the_prompt_names_the_default_in_words_and_the_command_stands_alone(self):
        # 0.2.16 retest F3: "... with: START-HERE.bat --calibrate [n]: " read like the command took an argument "[n]".
        shown = []
        with mock.patch("builtins.input", side_effect=lambda prompt: shown.append(prompt) or ""), mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            setup.ask(setup.TUNE_QUESTION.format(again="START-HERE.bat"), ["y", "n"], "n", False, shown=setup.TUNE_DEFAULT_SHOWN)
        lines = shown[0].split("\n")
        self.assertEqual(lines[-1], "Tune now? (y = yes, n = no) [Enter = no]: ")
        self.assertNotIn("[n]", shown[0])
        self.assertTrue(any(ln.strip().endswith("START-HERE.bat --calibrate") for ln in lines[:-1]))   # the command ends its line
        self.assertEqual(setup.ask("Q", ["y", "n"], "n", True, shown="x"), "n")                      # --yes: still the default
        with mock.patch("builtins.input", side_effect=lambda prompt: shown.append(prompt) or ""), mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            setup.ask("Pick", ["1", "2"], "1", False)
        self.assertTrue(shown[-1].endswith("Pick [1]: "))                # every other question keeps its own look

    def test_y_still_tunes(self):
        with mock.patch("builtins.input", return_value="y"), mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            self.assertEqual(setup.ask(setup.TUNE_QUESTION.format(again="./setup.sh"), ["y", "n"], "n", False, shown=setup.TUNE_DEFAULT_SHOWN), "y")

    def test_the_words_say_how_long_what_no_means_and_how_to_tune_later(self):
        text = setup.TUNE_QUESTION.format(again="./setup.sh")
        self.assertIn("15-30 minutes", text)
        self.assertIn("you can start sooner", text)
        self.assertIn("tune at any time later by running:  ./setup.sh --calibrate", text)
        with mock.patch.object(setup, "WIN", True):
            self.assertEqual(setup.setup_command(), "START-HERE.bat")
        with mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.setup_command(), "./setup.sh")

    def test_yes_and_scripts_still_never_tune(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        i = text.index("ask(TUNE_QUESTION.format(")
        line = text[text.rindex("\n", 0, i):i]
        for guard in ("cal is None", "not hip", "not a.no_start", "not a.yes"):
            self.assertIn(guard, line)


if __name__ == "__main__":
    unittest.main()
