"""Round 0213 QATUI: `strata update --check` on a SOURCE checkout printed `Control  ?: available (0.1.56)`: no installed Control and no VERSION file, so the version
was unknown. A checkout runs its own tree: what is PRINTED is the project version in CMakeLists.txt, as `strata --version` says (build 0.1.56).
Display only (owner 11-04): `decide_control`, which decides whether a Control copy is staged, works exactly as before.

    python -m pytest tests/terminal/test_update_control_source_0213_qatui.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli import update_cmd  # noqa: E402
from strata_control import engine_component as ec, updater  # noqa: E402


class SourceTreeVersion(unittest.TestCase):
    def test_it_is_the_cmake_project_version(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(strata VERSION 0.1.56 LANGUAGES CXX)\n", encoding="utf-8")
            self.assertEqual(updater.source_tree_version(Path(t)), "0.1.56")

    def test_nothing_to_read_is_none(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(updater.source_tree_version(Path(t)))
            (Path(t) / "CMakeLists.txt").write_text("project(other LANGUAGES C)\n", encoding="utf-8")
            self.assertIsNone(updater.source_tree_version(Path(t)))

    def test_this_checkout_has_a_version(self):
        self.assertRegex(updater.source_tree_version() or "", r"^\d+(\.\d+)+$")


def _report(installed):
    return {"feed": {"sequence": 1, "channel": "stable", "key_id": "k", "trusted_by": "test"},
            "engine": {"installed": None, "offered": None, "status": "not-in-feed", "download_bytes": 0},
            "control": {"installed": installed, "offered": "0.1.56", "status": "available", "download_bytes": 1},
            "app": {"status": "not-in-feed", "installed": None, "offered": None}}


class PrintedLine(unittest.TestCase):
    def _line(self, installed, source):
        out = io.StringIO()
        with mock.patch.object(update_cmd, "_source_version", return_value=source), contextlib.redirect_stdout(out):
            update_cmd._print_check(_report(installed))
        return next(ln for ln in out.getvalue().splitlines() if ln.startswith("  Control"))

    def test_a_checkout_shows_its_own_version(self):
        self.assertEqual(self._line(None, "0.1.56"), "  Control  0.1.56: available (0.1.56)")

    def test_an_installed_control_is_shown_as_it_was(self):
        self.assertEqual(self._line("0.1.55", "0.1.56"), "  Control  0.1.55: available (0.1.56)")

    def test_nothing_known_is_still_a_question_mark(self):
        self.assertEqual(self._line(None, None), "  Control  ?: available (0.1.56)")


class DecisionIsUnchanged(unittest.TestCase):
    def test_the_staging_decision_does_not_look_at_the_source_tree(self):
        feed = SimpleNamespace(control=[SimpleNamespace(platform="linux-x86_64", version="0.1.56", size=123)])
        ctx = SimpleNamespace(hardware={"platform": "linux-x86_64"}, components=Path("."))
        with mock.patch.object(updater, "active_control", return_value=None), mock.patch.object(ec, "runtime_version", return_value=None), \
                mock.patch.object(updater, "source_tree_version", side_effect=AssertionError("decide_control must not read the source tree")):
            report = updater.decide_control(feed, ctx, None)
        self.assertEqual((report["installed"], report["status"]), (None, "available"))


if __name__ == "__main__":
    unittest.main()
