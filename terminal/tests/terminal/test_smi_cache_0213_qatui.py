"""Round 0213 QATUI (final pass, owner PC): with an engine installed, one `strata launch --plan` on Windows started `nvidia-smi` 19 times (10 x compute_cap, 9 x
driver_version): every caller asked again. The answer is kept for 30 s per field now.

    python -m pytest tests/terminal/test_smi_cache_0213_qatui.py
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import engine_component as ec  # noqa: E402


class SmiCache(unittest.TestCase):
    def setUp(self):
        ec._SMI_CACHE.clear()
        self.addCleanup(ec._SMI_CACHE.clear)

    def _smi(self, run):
        return (mock.patch.object(ec.shutil, "which", lambda name: "nvidia-smi"), mock.patch.object(ec.subprocess, "run", run))

    def test_many_callers_start_nvidia_smi_once_per_field(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[1])
            return subprocess.CompletedProcess(cmd, 0, "12.0\n" if "compute_cap" in cmd[1] else "610.47\n", "")
        p1, p2 = self._smi(run)
        with p1, p2:
            for _ in range(10):
                self.assertEqual(ec.cuda_archs_present(), ["sm_120"])
                self.assertEqual(ec.nvidia_driver_version(), "610.47")
        self.assertEqual(sorted(calls), ["--query-gpu=compute_cap", "--query-gpu=driver_version"])

    def test_a_stuck_nvidia_smi_costs_its_timeout_once(self):
        calls = []

        def run(cmd, **kw):
            calls.append(kw.get("timeout"))
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        p1, p2 = self._smi(run)
        with p1, p2:
            for _ in range(10):
                self.assertEqual(ec.cuda_archs_present(), [])
        self.assertEqual(calls, [8])

    def test_it_is_asked_again_after_the_time_is_up(self):
        calls = []

        def run(cmd, **kw):
            calls.append(1)
            return subprocess.CompletedProcess(cmd, 0, "8.6\n", "")
        p1, p2 = self._smi(run)
        with p1, p2:
            ec.cuda_archs_present()
            stamp, found = ec._SMI_CACHE["compute_cap"]
            ec._SMI_CACHE["compute_cap"] = (stamp - ec.SMI_CACHE_SECONDS - 1, found)
            ec.cuda_archs_present()
        self.assertEqual(len(calls), 2)

    def test_a_caller_cannot_change_the_next_answer(self):
        p1, p2 = self._smi(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "8.6\n8.9\n", ""))
        with p1, p2:
            first = ec._smi_query("compute_cap")
            first.append("x")
            self.assertEqual(ec._smi_query("compute_cap"), ["8.6", "8.9"])


if __name__ == "__main__":
    unittest.main()
