# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_engine_nice.py - the engine's start nice value: off off Linux, the XESTRATA_ENGINE_NICE override, and the
automatic +5 only when a KFD GPU node is integrated (its cpu_cores_count, not a gfx name).

    python -m unittest serve.test_engine_nice -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as srv  # noqa: E402


class EngineNice(unittest.TestCase):
    def test_not_linux(self):
        self.assertEqual(srv.engine_nice({}, "win32"), 0)
        self.assertEqual(srv.engine_nice({}, "darwin"), 0)

    def test_env_override(self):
        for val, want in (("0", 0), ("7", 7), ("99", 19), ("-3", 0), ("x", 0)):
            self.assertEqual(srv.engine_nice({"XESTRATA_ENGINE_NICE": val}, "linux"), want)
        self.assertEqual(srv.engine_nice({"STRATA_ENGINE_NICE": "3"}, "linux"), 3)   # the old name still reads

    def test_auto_by_igpu(self):
        with mock.patch.object(srv, "hip_igpu_present", return_value=True):
            self.assertEqual(srv.engine_nice({}, "linux"), 5)
            self.assertEqual(srv.engine_nice({"XESTRATA_ENGINE_NICE": "0"}, "linux"), 0)   # off wins over the detection
        with mock.patch.object(srv, "hip_igpu_present", return_value=False):
            self.assertEqual(srv.engine_nice({}, "linux"), 0)

    def test_igpu_detection(self):
        discrete = "gfx_target_version 110000\nsimd_count 40\ncpu_cores_count 0\n"
        with mock.patch.object(srv.os, "listdir", return_value=["0", "1"]), \
                mock.patch("builtins.open", mock.mock_open(read_data=discrete)):
            self.assertFalse(srv.hip_igpu_present())   # a discrete card: its node reports no CPU cores
        apu = "gfx_target_version 115100\nsimd_count 40\ncpu_cores_count 16\n"
        with mock.patch.object(srv.os, "listdir", return_value=["0"]), \
                mock.patch("builtins.open", mock.mock_open(read_data=apu)):
            self.assertTrue(srv.hip_igpu_present())    # an APU: CPU cores on the GPU node


if __name__ == "__main__":
    unittest.main()
