"""Round 0213 QATUI: the Windows display-adapter CIM query is asked once per process, not once per caller (#4975 family).

Found on the owner PC (packaged Windows Terminal, `strata launch --plan`): the same PowerShell `Get-CimInstance Win32_VideoController` command ran 18 times in one
command (26 child processes, 17 s). With PowerShell/WMI stuck (a fake `powershell` that never answers) the plan was still running after 300 s: 18 times the
query's 15 s timeout, far beyond the 60 s the app waits for Control. After the change a stuck query costs its timeout once.

    python -m pytest tests/terminal/test_cim_cache_0213_qatui.py
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

GOOD = "PCI\\VEN_10DE&DEV_2F04&SUBSYS_1|NVIDIA GeForce RTX 5070|12884901888\nPCI\\VEN_1002&DEV_164E&SUBSYS_2|AMD Radeon(TM) Graphics|536870912\n"


class CimCache(unittest.TestCase):
    def setUp(self):
        ec._CIM_CACHE.clear()
        self.addCleanup(ec._CIM_CACHE.clear)

    def _windows(self, run):
        return (mock.patch.object(ec.os, "name", "nt"), mock.patch.object(ec.shutil, "which", lambda name: "powershell"),
                mock.patch.object(ec.subprocess, "run", run))

    def test_many_callers_ask_powershell_once(self):
        calls = []

        def run(*a, **k):
            calls.append(a)
            return subprocess.CompletedProcess(a, 0, GOOD, "")
        p1, p2, p3 = self._windows(run)
        with p1, p2, p3:
            first = ec._windows_adapters_cim()
            for _ in range(17):
                self.assertEqual(ec._windows_adapters_cim(), first)
        self.assertEqual(len(calls), 1)
        self.assertEqual({a["vendor"] for a in first}, {"10de", "1002"})

    def test_a_stuck_query_costs_its_timeout_once(self):
        calls = []

        def run(*a, **k):
            calls.append(k.get("timeout"))
            raise subprocess.TimeoutExpired(a, k.get("timeout"))
        p1, p2, p3 = self._windows(run)
        with p1, p2, p3:
            for _ in range(18):
                self.assertEqual(ec._windows_adapters_cim(), [])
        self.assertEqual(calls, [15])

    def test_the_answer_is_asked_again_after_it_expires(self):
        calls = []

        def run(*a, **k):
            calls.append(1)
            return subprocess.CompletedProcess(a, 0, GOOD, "")
        p1, p2, p3 = self._windows(run)
        with p1, p2, p3:
            ec._windows_adapters_cim()
            stamp, found = ec._CIM_CACHE["adapters"]
            ec._CIM_CACHE["adapters"] = (stamp - ec.CIM_CACHE_SECONDS - 1, found)
            ec._windows_adapters_cim()
        self.assertEqual(len(calls), 2)

    def test_a_caller_cannot_change_what_the_next_one_gets(self):
        p1, p2, p3 = self._windows(lambda *a, **k: subprocess.CompletedProcess(a, 0, GOOD, ""))
        with p1, p2, p3:
            a = ec._windows_adapters_cim()
            a[0]["vendor"] = "changed"
            self.assertNotEqual(ec._windows_adapters_cim()[0]["vendor"], "changed")

    def test_not_windows_asks_nothing(self):
        with mock.patch.object(ec.os, "name", "posix"), mock.patch.object(ec.subprocess, "run", side_effect=AssertionError("must not run")):
            self.assertEqual(ec._windows_adapters_cim(), [])
        self.assertNotIn("adapters", ec._CIM_CACHE)


if __name__ == "__main__":
    unittest.main()
