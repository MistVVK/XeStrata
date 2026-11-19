"""The Hy4-Flash engine as a component (strata_control/shape_engine.py, shape "hy4"): its name, its program, the test start that names it, and that the Strata engine's own checks are
unchanged.  The program is strata-hy4 (not a compile of the Strata engine), so it has no geometry: `--version` names it and that is the check."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from strata_control import engine_component as ec
from strata_control import shape_engine as SE


def fake_program(folder: Path, name: str, first_line: str, code: int = 0) -> Path:
    if os.name == "nt":
        p = folder / (name + ".cmd")
        p.write_text(f"@echo {first_line}\r\n@exit /b {code}\r\n")
    else:
        p = folder / name
        p.write_text(f"#!/bin/sh\necho '{first_line}'\nexit {code}\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


class Names(unittest.TestCase):
    def test_the_shape_is_valid_and_the_component_is_strata_hy4(self):
        self.assertTrue(SE.valid("hy4"))
        self.assertEqual(SE.component_id("hy4"), "strata-hy4")
        self.assertEqual(SE.reference("hy4"), ec.REFERENCE_PREFIX + "strata-hy4")
        self.assertEqual(SE.shape_of_component("strata-hy4"), "hy4")

    def test_the_program_names(self):
        suffix = ".exe" if os.name == "nt" else ""
        self.assertEqual(SE.exe_name("hy4"), "strata-hy4" + suffix)
        self.assertEqual(SE.exe_name(), "strata" + suffix)                      # the Strata engine's and the Qwen4 shapes': unchanged
        self.assertEqual(SE.exe_name("q4_lite"), "strata" + suffix)


class TestStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_the_hy4_engine_is_accepted_by_its_version_line(self):
        exe = fake_program(self.dir, "strata-hy4", "strata-hy4 0.2.18-dev")
        ok, why = SE._test_start(exe, shape="hy4")
        self.assertTrue(ok, why)
        self.assertEqual(why, "strata-hy4 0.2.18-dev")

    def test_a_program_that_is_not_the_hy4_engine_is_refused(self):
        exe = fake_program(self.dir, "strata-other", "strata 0.2.18")
        ok, why = SE._test_start(exe, shape="hy4")
        self.assertFalse(ok)
        self.assertEqual(why, SE.NO_SHAPE_ANSWER)

    def test_a_failing_program_is_refused(self):
        exe = fake_program(self.dir, "strata-hy4", "strata-hy4 0.2.18-dev", code=3)
        ok, _why = SE._test_start(exe, shape="hy4")
        self.assertFalse(ok)

    def test_the_qwen4_shape_still_asks_for_its_geometry(self):
        exe = fake_program(self.dir, "strata", "strata 0.2.18")                 # answers --version, and --geometry with the same line: not a shape name
        ok, why = SE._test_start(exe, shape="q4_lite")
        self.assertFalse(ok)
        self.assertEqual(why, SE.NO_SHAPE_ANSWER)


if __name__ == "__main__":
    unittest.main()
