"""0.2.17 lane PLAT's three one-hunk changes in strata_cli/launch/* (lane TUI's files), with their tests:
  1. cmd._can_ask: `strata launch < /dev/null` shows no "Start it now? [Y/n]" (it asked, got end of input and ended with exit 2 anyway);
  2. modelstate.reader_present: the plan knows a picture reader in the person's own folder under another name (setup.own_reader_by_content);
  3. catalog.model_folder_trace: a data folder below a folder this account may not read (PermissionError) is "not there", not a crash that
     silently stopped the Save screen from repainting (the boston-only StoragePath failure)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_cli.launch import cmd, modelstate
from strata_cli.launch.runner import Runner
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

GOOD = b"reader-bytes " * 200
SHA = hashlib.sha256(GOOD).hexdigest()


class NoKeyboard(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_dev_null_is_never_asked_even_where_a_terminal_name_is_set(self):
        env = {"TERM": "xterm-256color", "CI": "", "STRATA_NO_PROMPT": ""}
        with mock.patch.dict(os.environ, env), open(os.devnull) as null, mock.patch.object(sys, "stdin", null):
            if os.isatty(null.fileno()):
                self.skipTest("this Python reports the null device as a terminal")
            self.assertFalse(cmd._can_ask())
        with mock.patch.dict(os.environ, env), mock.patch.object(sys, "stdin", io.StringIO("y\n")):
            self.assertTrue(cmd._can_ask(), "a pipe or a replaced stdin may carry answers: unchanged")

    def test_launch_from_dev_null_names_what_to_start_and_shows_no_question(self):
        env = {"TERM": "xterm-256color", "CI": "", "STRATA_NO_PROMPT": ""}
        with mock.patch.dict(os.environ, env), open(os.devnull) as null, mock.patch.object(sys, "stdin", null), \
                mock.patch("builtins.input", side_effect=AssertionError("no question")), \
                mock.patch.object(Runner, "run", side_effect=AssertionError("must not run")):
            if os.isatty(null.fileno()):
                self.skipTest("this Python reports the null device as a terminal")
            code, out, err = run_cli(["launch"])
        self.assertEqual(code, 2)
        self.assertIn("not a terminal", err)
        self.assertIn("--preset NAME", err)
        self.assertNotIn("Start it now", out + err)
        self.assertNotIn("Going on with plain questions", out + err)


class PlanSeesARenamedReader(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        t = tempfile.TemporaryDirectory(prefix="strata-plan-reader-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)
        self.cat = catalog(HW_12)
        self.mod = self.cat.strata.setup_module()
        if not hasattr(self.mod, "own_reader_by_content"):
            self.skipTest("needs setup.own_reader_by_content (0217-PLAT's setup.py; present after the merge)")
        self.pinned = self.mod.MMPROJ
        for p in (mock.patch.dict(self.mod.ISTA_FILES, {self.pinned: (len(GOOD), SHA)}),
                  mock.patch.object(self.mod, "ROOT", self.base / "strata-root")):
            p.start()
            self.addCleanup(p.stop)
        self.own = self.base / "mine" / "IQ3_XXS"
        self.own.mkdir(parents=True)

    def plan(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return modelstate.reader_present(self.cat, "qwen", [self.base / "data"], own=self.own)

    def test_the_same_bytes_under_another_name_count_and_other_bytes_do_not(self):
        (self.own / "q38-mmproj-BF16.gguf").write_bytes(GOOD)
        self.assertTrue(self.plan())
        (self.own / "q38-mmproj-BF16.gguf").write_bytes(bytes(reversed(GOOD)))
        self.assertFalse(self.plan(), "a reader of another model: the plan lists the download")
        (self.own / "q38-mmproj-BF16.gguf").write_bytes(GOOD[:-1])
        self.assertFalse(self.plan())


@unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs POSIX permissions and an ordinary account")
class UnreadableFolder(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        t = tempfile.TemporaryDirectory(prefix="strata-unreadable-")
        self.addCleanup(t.cleanup)
        self.locked = Path(t.name) / "locked"
        self.locked.mkdir()
        os.chmod(self.locked, 0)
        self.addCleanup(os.chmod, self.locked, 0o700)

    def test_a_data_folder_below_a_folder_that_may_not_be_read_is_just_not_there(self):
        cat = catalog(HW_12)
        trace = cat.model_folder_trace(self.locked / "Strata-data", "qwen", "IQ3_S")
        self.assertEqual(trace["rule"], "fallback")
        self.assertFalse(any(t["exists"] for t in trace["tried"]))
        st = modelstate.assess(cat, "qwen", "IQ3_S", data_dir=self.locked / "Strata-data")      # the Save screen's plan: no PermissionError
        self.assertFalse(st.installed)


if __name__ == "__main__":
    unittest.main()
