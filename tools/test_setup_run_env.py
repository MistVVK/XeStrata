# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""setup's child processes get EXIT_WITH_PARENT before they start, on both output paths (upstream #3257).

    python -m unittest tools.test_setup_run_env
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class RunEnv(unittest.TestCase):
    def test_a_plain_run_sets_it_before_the_child(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen.update(kw["env"])
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(setup.subprocess, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            setup.run(["true"])
        self.assertEqual(seen.get("XESTRATA_EXIT_WITH_PARENT"), "1")

    def test_a_json_run_sets_it_before_the_child(self):
        seen = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                seen.update(kw["env"])
                self.args, self.returncode, self.stdout = cmd, 0, []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(setup, "JSON_OUT", True), \
                mock.patch.object(setup.subprocess, "Popen", FakePopen), contextlib.redirect_stdout(io.StringIO()):
            setup.run(["true"])
        self.assertEqual(seen.get("XESTRATA_EXIT_WITH_PARENT"), "1")

    def test_a_value_from_the_caller_is_kept(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen.update(kw["env"])
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(setup.subprocess, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            setup.run(["true"], env={"XESTRATA_EXIT_WITH_PARENT": "0"})
        self.assertEqual(seen.get("XESTRATA_EXIT_WITH_PARENT"), "0")


if __name__ == "__main__":
    unittest.main()
