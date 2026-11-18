"""P13 (0.2.16): with --gguf-dir, setup does not ask "Which model?" and "Which size?" when the files name exactly one model.

USERTEST-LIN L2: `./setup.sh --gguf-dir <IQ3_XXS folder>` still asked both questions (answered 1 and 3) before it used the files.
setup.gguf_dir_choices reads the folder: one published first-shard name = that family and size, no question; several models or nothing
recognisable = the questions stay.

    python -m unittest tools.test_setup_gguf_dir_choice
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

IQ3_XXS = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
IQ3_S = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"


class Folder(unittest.TestCase):
    def folder(self, *names):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        for n in names:
            (Path(d.name) / n).write_bytes(b"x")
        return Path(d.name)

    def test_the_published_names_are_found_and_the_other_shards_do_not_count(self):
        self.assertIsNotNone(setup.gguf_choice(IQ3_XXS), "the fixture name must be a published first shard")
        folder = self.folder(IQ3_XXS, IQ3_XXS.replace("00001", "00002"), "mmproj-F16.gguf", IQ3_XXS + ".done")
        got = setup.gguf_dir_choices(folder)
        self.assertEqual(got, [setup.gguf_choice(IQ3_XXS)])
        self.assertEqual(got[0][1], "IQ3_XXS")

    def test_two_models_in_one_folder_leave_the_questions_to_the_person(self):
        self.assertEqual(len(setup.gguf_dir_choices(self.folder(IQ3_XXS, IQ3_S))), 2)

    def test_a_folder_with_nothing_recognisable_or_nothing_at_all(self):
        self.assertEqual(setup.gguf_dir_choices(self.folder("some-other-model-Q4_K_M.gguf")), [])
        self.assertEqual(setup.gguf_dir_choices(self.folder()), [])
        self.assertEqual(setup.gguf_dir_choices(Path("/no/such/folder/at/all")), [])

    def test_a_family_or_size_already_given_narrows_the_choice(self):
        folder = self.folder(IQ3_XXS, IQ3_S)
        fam = setup.gguf_choice(IQ3_XXS)[0]
        self.assertEqual(setup.gguf_dir_choices(folder, None, "IQ3_S"), [setup.gguf_choice(IQ3_S)])
        self.assertEqual(setup.gguf_dir_choices(folder, fam, "IQ3_XXS"), [setup.gguf_choice(IQ3_XXS)])
        self.assertEqual(setup.gguf_dir_choices(folder, None, "IQ2_XS"), [])        # the size asked for is not in the folder: the old error path says so

    def test_an_unsupported_quant_is_not_a_choice(self):
        self.assertEqual(setup.gguf_dir_choices(self.folder("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00002.gguf")), [])


class WholeSetup(unittest.TestCase):
    """setup.main() on a mocked PC (test_setup_golden.install), the folder given and no --family / --model: the questions are not asked."""

    def run_setup(self, names, answers=""):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["64GB-1x32GB"]
        with tempfile.TemporaryDirectory() as tmp:
            for n in names:
                (Path(tmp) / n).write_bytes(b"")
            return install(ram, found, ["--gguf-dir", tmp, "--no-start"], answers=answers,
                           extra=[mock.patch.object(setup, "whole_shard", lambda s: True),
                                  mock.patch.object(setup, "feed_models_setup", lambda *a, **k: None)])      # no network: the signed feed lookup


    def test_one_model_in_the_folder_asks_neither_question(self):
        code, out, cfg, asked = self.run_setup([f"Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{i:05d}-of-00002.gguf" for i in (1, 2)])
        self.assertEqual(code, 0, out)
        self.assertIn(": using them (no question to answer).", out)
        self.assertIn("Q2_0", out.split("using them")[0].splitlines()[-1])
        self.assertFalse([q for q in asked if "Which model?" in q or "Which size?" in q], asked)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-q2_0")

    def test_two_models_in_the_folder_still_ask(self):
        names = [f"Qwen3.8-Flash-Next-GSQ-RCO-{q}-{i:05d}-of-00002.gguf" for q in ("Q2_0", "IQ3_XXS") for i in (1, 2)]
        code, out, cfg, asked = self.run_setup(names, answers="")  # Enter for every question: the run then stops at the shard the default size lacks; only the questions matter here
        self.assertTrue([q for q in asked if "Which model?" in q], asked)
        self.assertNotIn("using them (no question to answer)", out)


class Wiring(unittest.TestCase):
    def test_step_2_uses_it_before_the_menus_and_only_when_a_choice_is_missing(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        i = text.index("named = gguf_dir_choices(Path(a.gguf_dir), a.family, a.model)")
        self.assertIn("if a.gguf_dir and not (a.family and a.model):", text[i - 200:i])
        self.assertIn("if len(named) == 1:", text[i:i + 120])
        self.assertLess(i, text.index('family = allf[int(ask("Which model?"'))
        self.assertLess(i, text.index('model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]'))

    def test_the_answer_is_said_in_one_line(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("are {FAMILIES[a.family]['title']} {a.model}: using them (no question to answer).", text)


if __name__ == "__main__":
    unittest.main()
