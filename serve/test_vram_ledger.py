# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The VRAM ledger (serve/vram_ledger.py): reservations of two processes against a device whose report does not move
with their allocations, a failed allocation, the boundary of a release, a dead owner.

    .venv/bin/python -m unittest discover -s serve -p test_vram_ledger.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vram_ledger  # noqa: E402


def sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = vram_ledger.Ledger("gpu0", margin_mib=100, settle_s=0.5, folder=self.tmp.name)
        self.procs: list[subprocess.Popen] = []

    def tearDown(self):
        for p in self.procs:
            p.kill()
            p.wait()
        self.tmp.cleanup()

    def other(self) -> int:
        p = sleeper()
        self.procs.append(p)
        return p.pid

    def test_report_that_does_not_move(self):
        # 8000 MiB free, and the device keeps saying so: an allocated entry is still charged
        a = self.ledger.reserve(5000, "translate", 8000, pid=self.other())
        self.assertIsNotNone(a)
        self.assertTrue(self.ledger.allocated(a))
        self.assertIsNone(self.ledger.reserve(3000, "image", 8000, pid=self.other()))   # 5000 + 3000 + 100 > 8000
        self.assertIsNotNone(self.ledger.reserve(2800, "image", 8000, pid=self.other()))

    def test_outsider_in_the_report(self):
        # the chat's engine takes 3000 MiB outside the ledger: the report's term counts it
        self.assertEqual(self.ledger.room(8000), 7900)
        self.assertEqual(self.ledger.room(5000), 4900)

    def test_failed_allocation(self):
        a = self.ledger.reserve(6000, "translate", 8000, pid=self.other())
        self.assertIsNone(self.ledger.reserve(3000, "image", 8000, pid=self.other()))
        self.ledger.remove(a)   # the allocation failed: the entry goes, the caller plans again
        self.assertEqual(self.ledger.entries(), [])
        self.assertIsNotNone(self.ledger.reserve(3000, "image", 8000, pid=self.other()))

    def test_release_boundary(self):
        a = self.ledger.reserve(6000, "translate", 8000, pid=self.other())
        self.ledger.allocated(a)
        self.ledger.releasing(a)   # being given back: another process reading now still sees it charged
        self.assertIsNone(self.ledger.reserve(3000, "image", 8000, pid=self.other()))
        self.ledger.remove(a)
        # right after the release the report still shows the allocation (2000 free): the base is not taken from it,
        # and the report's term keeps the reservation within what is free
        self.assertIsNone(self.ledger.reserve(3000, "image", 2000, pid=self.other()))
        self.assertEqual(self.ledger.room(2000), 1900)
        time.sleep(0.6)   # the report caught up after the settling time: the base is taken again
        self.assertEqual(self.ledger.room(8000), 7900)

    def test_base_not_lowered_early(self):
        self.assertEqual(self.ledger.room(8000), 7900)   # the base: 8000
        a = self.ledger.reserve(1000, "image", 8000, pid=self.other())
        self.ledger.allocated(a)
        self.ledger.remove(a)
        self.ledger.room(3000)   # a stale report right after the removal: not the new base
        self.assertEqual(self.ledger.room(9000), 7900)   # the old base still bounds it

    def test_dead_owner(self):
        p = sleeper()
        a = self.ledger.reserve(6000, "translate", 8000, pid=p.pid)
        self.assertIsNotNone(a)
        p.kill()
        p.wait()
        self.assertEqual(self.ledger.entries(), [])   # gone with its process
        self.assertIsNotNone(self.ledger.reserve(6000, "image", 8000, pid=self.other()))

    def test_reused_pid(self):
        pid = self.other()
        a = self.ledger.reserve(6000, "translate", 8000, pid=pid)
        state = self.ledger._read()   # the same pid, started at another time: another process
        state["entries"][0]["start"] -= 100.0
        self.ledger._write(state)
        self.assertEqual(self.ledger.entries(), [])
        self.assertFalse(self.ledger.allocated(a))

    def test_owner_moves_to_the_engine(self):
        a = self.ledger.reserve(1000, "translate", 8000)
        engine = self.other()
        self.assertTrue(self.ledger.allocated(a, pid=engine))
        self.assertEqual(self.ledger.entries()[0]["pid"], engine)
        self.assertNotEqual(engine, os.getpid())


if __name__ == "__main__":
    unittest.main()
