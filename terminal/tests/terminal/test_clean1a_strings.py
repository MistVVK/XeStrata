"""The four strings the owner approved word for word (11-07), and the rule for the third."""
from __future__ import annotations

import argparse
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from strata_control import drafting
from strata_cli.launch import cmd as launch_cmd

ROOT = Path(__file__).resolve().parents[2]


class Wording(unittest.TestCase):
    def test_launch_help_and_the_docs_row_say_the_size_depends_on_the_model(self):
        top = argparse.ArgumentParser()
        parsers = launch_cmd.add_parsers(top.add_subparsers(dest="command"))
        helptext = " ".join(" ".join(p.format_help() for p in parsers).split())
        self.assertIn("dflash2 (downloads its drafter once, 0.3 to 1.2 GB depending on the model; the size is shown before the download)", helptext)
        self.assertNotIn("1.2 GB drafter once", helptext)
        docs = (ROOT.parent / "docs" / "TERMINAL_LAUNCH.md").read_text(encoding="utf-8")
        self.assertIn("and downloads its drafter once (0.34 GB for the models that have a tuned drafter, 0.52 GB on Intel Arc, else 1.2 GB).", docs)

    def test_the_arc_launch_line_names_what_was_not_downloaded(self):
        src = (ROOT / "strata_cli" / "launch" / "runner.py").read_text(encoding="utf-8")
        self.assertIn("({path.name}). No model files were downloaded.\")", src)
        self.assertNotIn("({path.name}). Nothing was downloaded.", src)


class FetchHint(unittest.TestCase):
    NONE = drafting.Decision("auto", "off", drafting.NONE_HERE)

    def hint(self, data, decision=None, **kw):
        kw = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda", "archs": ["sm_120"], "args": [], "vram_mib": 73728.0, **kw}
        return drafting.fetch_hint(decision or self.NONE, data_dir=data, **kw)

    def miss(self, data, age=0):
        mark = Path(data) / drafting.FOLDER / drafting.MISS_MARK
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text("x")
        if age:
            old = time.time() - age
            os.utime(mark, (old, old))

    def test_only_with_a_recent_miss_and_a_tuned_set(self):
        with TemporaryDirectory() as d:
            self.assertEqual(self.hint(d), "", "no miss mark")
            self.miss(d)
            self.assertEqual(self.hint(d), "Run strata drafting fetch to get the DFlash2 drafter now.")
            self.assertEqual(self.hint(d, drafting.Decision("dflash2", "off", drafting.NONE_HERE)), "Run strata drafting fetch to get the DFlash2 drafter now.")
            self.assertEqual(self.hint(d, drafting.Decision("mtp", "off", drafting.NONE_HERE)), "", "MTP chosen: nothing to fetch")
            self.assertEqual(self.hint(d, drafting.Decision("auto", "mtp", "")), "", "a drafter runs")
            self.assertEqual(self.hint(d, drafting.Decision("auto", "off", drafting.DIDNT_FIT)), "", "off for another reason")
            self.assertEqual(self.hint(d, model_name="qwen3.8-flash-next-coder-iq1_m"), "", "a set with no tuned drafter")

    def test_an_old_miss_says_nothing(self):
        with TemporaryDirectory() as d:
            self.miss(d, age=drafting.RETRY_AFTER_S + 60)
            self.assertEqual(self.hint(d), "")


if __name__ == "__main__":
    unittest.main()
