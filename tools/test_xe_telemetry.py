"""tools/test_xe_telemetry.py - the Intel Arc load read from DRM fdinfo (#5994), against a fake /proc; no card, no Level Zero.

    python -m pytest tools/test_xe_telemetry.py   (py-tools runs it)

What the load means (#5994 review): the share of the card's cycles used by the DRM clients this process can see; unknown (None)
until a client has two readings whose counters moved forward. A client seen for the first time, or whose counters went back, is
never taken as idle; a client open in several file descriptors counts once; other cards and drivers are left out.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sycl" / "serve"))
import xe_telemetry  # noqa: E402

PCI = "0000:03:00.0"


def fdinfo(cid, rcs, ccs, total, pdev=PCI, driver="xe", cap_ccs=1):
    return (f"pos:\t0\nflags:\t02100002\ndrm-driver:\t{driver}\ndrm-client-id:\t{cid}\ndrm-pdev:\t{pdev}\n"
            f"drm-cycles-rcs:\t{rcs}\ndrm-total-cycles-rcs:\t{total}\ndrm-cycles-ccs:\t{ccs}\ndrm-total-cycles-ccs:\t{total}\n"
            f"drm-engine-capacity-ccs:\t{cap_ccs}\n")


class FakeProc:
    def __init__(self, root):
        self.root = Path(root)
        self.n = 0

    def clear(self):
        for p in sorted(self.root.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()

    def put(self, pid, fd, text):
        d = self.root / str(pid) / "fdinfo"
        d.mkdir(parents=True, exist_ok=True)
        (d / str(fd)).write_text(text, encoding="ascii")


class Load(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proc = FakeProc(self.tmp.name)
        self.gpu = xe_telemetry._XeGpu(index=99)       # no such card: no sysfs, no Level Zero
        self.gpu.pci = PCI
        self.gpu._proc = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, clients):
        self.proc.clear()
        for i, (pid, fd, text) in enumerate(clients):
            self.proc.put(pid, fd, text)
        return self.gpu._busy()

    def test_first_reading_is_unknown_then_a_delta_gives_the_load(self):
        self.assertIsNone(self.read([(10, 5, fdinfo(1, 0, 0, 1000))]))
        self.assertAlmostEqual(self.read([(10, 5, fdinfo(1, 250, 250, 2000))]), 50.0)

    def test_an_idle_client_with_a_delta_is_zero_not_unknown(self):
        self.read([(10, 5, fdinfo(1, 100, 0, 1000))])
        self.assertEqual(self.read([(10, 5, fdinfo(1, 100, 0, 2000))]), 0.0)

    def test_only_new_clients_is_unknown_not_idle(self):
        self.read([(10, 5, fdinfo(1, 0, 0, 1000))])
        self.assertIsNone(self.read([(11, 5, fdinfo(2, 900, 0, 1000))]), "a client seen for the first time says nothing yet")

    def test_a_counter_that_went_back_is_a_new_client(self):
        self.read([(10, 5, fdinfo(1, 800, 0, 5000))])
        self.assertIsNone(self.read([(10, 5, fdinfo(1, 10, 0, 100))]), "a reset is not a negative or zero load")
        self.assertAlmostEqual(self.read([(10, 5, fdinfo(1, 60, 0, 200))]), 50.0)

    def test_a_client_open_in_two_descriptors_counts_once(self):
        self.read([(10, 5, fdinfo(1, 0, 0, 1000)), (10, 6, fdinfo(1, 0, 0, 1000))])
        self.assertAlmostEqual(self.read([(10, 5, fdinfo(1, 300, 0, 2000)), (10, 6, fdinfo(1, 300, 0, 2000))]), 30.0)

    def test_two_clients_add_up_and_stay_within_100(self):
        self.read([(10, 5, fdinfo(1, 0, 0, 1000)), (11, 3, fdinfo(2, 0, 0, 1000))])
        self.assertAlmostEqual(self.read([(10, 5, fdinfo(1, 400, 0, 2000)), (11, 3, fdinfo(2, 300, 0, 2000))]), 70.0)
        self.assertEqual(self.read([(10, 5, fdinfo(1, 2400, 0, 3000)), (11, 3, fdinfo(2, 2300, 0, 3000))]), 100.0)

    def test_other_cards_drivers_and_unreadable_entries_are_left_out(self):
        clients = [(10, 5, fdinfo(1, 0, 0, 1000)), (12, 1, fdinfo(7, 0, 0, 1000, pdev="0000:04:00.0")),
                   (13, 1, fdinfo(8, 0, 0, 1000, driver="i915")), (14, 1, "not drm\n")]
        self.read(clients)
        later = [(10, 5, fdinfo(1, 100, 0, 2000)), (12, 1, fdinfo(7, 1000, 0, 2000, pdev="0000:04:00.0")),
                 (13, 1, fdinfo(8, 1000, 0, 2000, driver="i915")), (14, 1, "not drm\n")]
        self.assertAlmostEqual(self.read(later), 10.0)

    def test_engine_capacity_divides_a_class_s_cycles(self):
        self.read([(10, 5, fdinfo(1, 0, 0, 1000, cap_ccs=4))])
        self.assertAlmostEqual(self.read([(10, 5, fdinfo(1, 0, 400, 2000, cap_ccs=4))]), 10.0)

    def test_no_clients_is_unknown(self):
        self.assertIsNone(self.read([]))
        self.assertIsNone(self.read([]))


class OldSampler(unittest.TestCase):
    """A PC that still runs the root sampler (sycl/tools/gpustat.py) keeps its readings for what this process cannot read."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gpu = xe_telemetry._XeGpu(index=99)
        self.gpu.pci = PCI
        self.gpu._proc = os.path.join(self.tmp.name, "proc")
        self.gpu.STAT = os.path.join(self.tmp.name, "gpustat.json")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, ts):
        Path(self.gpu.STAT).write_text(json.dumps({"ts": ts, "busy_pct": 37, "vram_used_mb": 2048, "vram_total_mb": 16384}))

    def test_a_fresh_sampler_fills_what_is_missing(self):
        self.write(time.time())
        out = self.gpu.read()
        self.assertEqual(out["util"], 37)
        self.assertEqual(out["mem_used"], 2048 * 2**20)
        self.assertEqual(out["mem_total"], 16384 * 2**20)

    def test_an_old_or_missing_sampler_leaves_them_unknown(self):
        self.write(time.time() - 60)
        out = self.gpu.read()
        self.assertIsNone(out["util"])
        self.assertIsNone(out["mem_used"])
        os.unlink(self.gpu.STAT)
        self.assertIsNone(self.gpu.read()["mem_used"])


if __name__ == "__main__":
    unittest.main()
