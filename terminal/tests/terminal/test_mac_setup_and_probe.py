"""Two Mac fixes found by the Mac test run (11-16):
M1: `setup.py --family deepseek41` (or a DeepSeek model name) on a Mac printed nothing and stopped with metal/setup_mac.py's own argument error; the
    retired names are now checked before the Mac hand-over: the same sentence and exit 2 as on Linux and Windows.
M2: a plan on a Mac read the IOKit registry (`ioreg`) six times; the registry's static facts are read once per process and a Mac's card memory
    is the Metal working set, so one read is left.
Both run anywhere: the Mac is named with sys.platform and the probes are counted, nothing real is started."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest import mock

from serve import telemetry
from strata_control import drafting


class RetiredNamesOnAMac(unittest.TestCase):
    def test_the_same_sentence_and_exit_before_the_mac_hand_over(self):
        import setup
        for argv in (("--family", "deepseek41"), ("--model", "IST-DAS-2.25bpw"), ("--model=ISTA-DAS-Coder",)):
            out = io.StringIO()
            with mock.patch.object(sys, "argv", ["setup.py", *argv, "--yes"]), mock.patch.object(setup.sys, "platform", "darwin"), \
                    mock.patch.object(setup, "CONTROL", False), \
                    mock.patch.object(setup, "mac_setup", side_effect=AssertionError("the Mac setup must not be reached")), \
                    contextlib.redirect_stdout(out):
                code = setup.main()
            self.assertEqual(code, 2, argv)
            self.assertIn("Strata no longer offers DeepSeek models. Run `strata launch` to choose one of the Qwen models.", out.getvalue())

    def test_a_qwen_run_still_goes_to_the_mac_setup(self):
        import setup
        with mock.patch.object(sys, "argv", ["setup.py", "--yes"]), mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup, "CONTROL", False), mock.patch.object(setup, "mac_setup", return_value=0) as mac:
            self.assertEqual(setup.main(), 0)
        mac.assert_called_once_with(["--yes"])


REGISTRY = ('+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X>\n    "model" = "Apple M5 Max"\n'
            '    "PerformanceStatistics" = {"Device Utilization %"=12,"In use system memory"=123456}\n')


class OneRegistryRead(unittest.TestCase):
    def test_a_plan_reads_the_registry_once(self):
        calls = []

        def run(argv, *a, **kw):
            calls.append(list(argv))
            return mock.Mock(stdout=REGISTRY, returncode=0)
        with mock.patch.object(telemetry.sys, "platform", "darwin"), mock.patch.object(drafting.sys, "platform", "darwin"), \
                mock.patch.object(telemetry._Metal, "_first", [], create=True), mock.patch.object(telemetry._Metal, "_recent", [], create=True), mock.patch.object(telemetry.subprocess, "run", side_effect=run), \
                mock.patch.object(telemetry, "metal_working_set", return_value=48 << 30):
            for _ in range(3):                              # the plan asks for the card's memory more than once
                self.assertEqual(drafting.card_vram_mib({"gpu": 0}, "metal", []), float(48 << 10))
        ioreg = [c for c in calls if c and c[0] == "ioreg"]
        self.assertEqual(len(ioreg), 1, ioreg)

    def test_live_readings_still_read_the_registry_each_time(self):
        calls = []

        def run(argv, *a, **kw):
            calls.append(list(argv))
            return mock.Mock(stdout=REGISTRY, returncode=0)
        with mock.patch.object(telemetry.sys, "platform", "darwin"), mock.patch.object(telemetry._Metal, "_first", [], create=True), mock.patch.object(telemetry._Metal, "_recent", [], create=True), \
                mock.patch.object(telemetry.subprocess, "run", side_effect=run), mock.patch.object(telemetry, "metal_working_set", return_value=1):
            clock = [1000.0]
            with mock.patch.object(telemetry.time, "monotonic", side_effect=lambda: clock[0]):
                g = telemetry.gpu_reader(0)
                self.assertEqual((g.ok(), g.name()), (True, "Apple M5 Max"))
                before = len(calls)
                for _ in range(2):                          # the server's polls, seconds apart
                    clock[0] += 2.0
                    self.assertEqual(g.read()["util"], 12)
        self.assertEqual(len(calls) - before, 2, "the server's telemetry keeps reading the load live")


if __name__ == "__main__":
    unittest.main()
