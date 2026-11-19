"""run-<model>.bat is written with plain CRLF line ends: setup writes the \\r\\n itself, so text mode on Windows must not
translate them again (it made "@echo off\\r\\r\\n").  The test writes the script into a temporary folder with WIN set,
so it runs on Linux too (where the doubling cannot happen, but the line ends are still checked).

    python -m unittest tools.test_setup_run_script_newlines
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class RunScriptNewlines(unittest.TestCase):
    def test_bat_has_single_crlf(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "ROOT", Path(d)):
                script = setup.write_run_script("IQ3_XXS", Path(d) / "strata-iq3_xxs.json", 8080)
            data = script.read_bytes()
        self.assertNotIn(b"\r\r\n", data)
        self.assertTrue(data.startswith(b"@echo off\r\ntitle Strata IQ3_XXS\r\n"), data[:40])
        self.assertTrue(data.endswith(b"if errorlevel 1 pause\r\n"))


if __name__ == "__main__":
    unittest.main()
