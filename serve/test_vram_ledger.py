# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The VRAM ledger (serve/vram_ledger.py): reservations of two processes against a device whose report does not move
with their allocations, a failed allocation, the boundary of a release, a dead owner, holders in the base.

    .venv/bin/python -m unittest discover -s serve -p test_vram_ledger.py
"""
import os
from pathlib import Path
import subprocess
from unittest import mock
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

    def test_shared_margin_and_device_totals(self):
        # These are core::vram_room(total, 1, 256 MiB, 768 MiB), rounded down to MiB.
        for total, expected in ((0, 256), (5783, 256), (8192, 256), (12288, 384),
                                (16304, 509), (31023, 768)):
            self.assertEqual(vram_ledger.device_margin_mib(total), expected)
        image = vram_ledger.Ledger("shared", total_mib=11902, engine_total=True, folder=self.tmp.name)
        translation = vram_ledger.Ledger("shared", total_mib=12282, folder=self.tmp.name)
        self.assertEqual(translation.room(10000), 9617)   # telemetry before the first image engine
        self.assertEqual(image.room(10000), 9629)         # the engine's 371 MiB replaces its 383 MiB
        self.assertEqual(translation.room(None), 9629)    # telemetry cannot overwrite the engine's total
        self.assertEqual(image.room(None), 9629)
        self.assertEqual(vram_ledger.Ledger("unknown", folder=self.tmp.name).room(10000), 9744)

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

    def test_report_not_known(self):
        # the translation engine's server reads no report: no limit until a base is known, then the base's term alone,
        # and the base is not taken from it
        self.assertIsNone(self.ledger.room(None))
        a = self.ledger.reserve(3000, "translate", None, pid=self.other())
        self.assertIsNotNone(a)
        self.assertEqual(self.ledger.room(8000), 4900)          # an image job's report: 8000 - 3000 reserved - 100
        self.ledger.remove(a)
        time.sleep(0.6)
        self.assertEqual(self.ledger.room(8000), 7900)          # the base: 8000
        self.ledger.reserve(5000, "image", 8000, pid=self.other())
        self.assertEqual(self.ledger.room(None), 2900)          # 8000 - 5000 - 100
        self.assertIsNone(self.ledger.reserve(3000, "translate", None, pid=self.other()))

    def test_owner_moves_to_the_engine(self):
        a = self.ledger.reserve(1000, "translate", 8000)
        engine = self.other()
        self.assertTrue(self.ledger.allocated(a, pid=engine))
        self.assertEqual(self.ledger.entries()[0]["pid"], engine)
        self.assertNotEqual(engine, os.getpid())

    def test_helper_card_lent_and_filled_again(self):
        # the text engine's helper cache on its card before the ledger knew of it, handed over for an image job, filled
        # again, handed over for a translation: the translation gets the card less what the image engine keeps
        text = self.other()
        self.ledger.hold("helper", 9000, text, in_base=True)
        self.assertIsNone(self.ledger.room(None))               # the base is read again at the next report
        self.ledger.hold("helper", 0, text)                     # handed over
        self.assertEqual(self.ledger.room(11000), 10900)        # the image job's report: the base 11000
        image = self.other()
        self.ledger.hold("image", 8000, image, exact=False)     # the job's budget
        self.assertEqual(self.ledger.room(None), 2900)
        self.ledger.hold("image", 1000, image)                  # kept after the job, its own report
        self.ledger.hold("helper", 9000, text)                  # filled again: charged
        self.assertEqual(self.ledger.room(None), 900)
        self.ledger.hold("helper", 0, text)                     # handed over to the translation
        self.assertEqual(self.ledger.room(None), 9900)

    def test_give_back_of_a_holder_in_the_base(self):
        # the text engine's expert cache (6000) is in the base; what it gives back is room without a new report, what
        # it takes back is charged
        text = self.other()
        self.ledger.hold("text", 6000, text, in_base=True)
        self.assertEqual(self.ledger.room(4000), 3900)
        self.ledger.hold("text", 2000, text)
        self.assertEqual(self.ledger.room(None), 7900)
        self.assertEqual(self.ledger.room(4000), 3900)          # a report that has not caught up still bounds it
        self.ledger.hold("text", 6000, text)
        self.assertEqual(self.ledger.room(None), 3900)

    def test_base_read_again_only_with_exact_entries(self):
        self.assertEqual(self.ledger.room(8000), 7900)
        image = self.other()
        self.ledger.hold("image", 3000, image, exact=False)     # a budget: what it holds is not known
        time.sleep(0.6)
        self.assertEqual(self.ledger.room(5000), 4900)
        self.assertEqual(self.ledger.room(9000), 4900)          # the base 8000 kept: 8000 - 3000 - 100
        self.ledger.hold("image", 3000, image)                  # its own report
        time.sleep(0.6)
        self.assertEqual(self.ledger.room(5000), 4900)          # the base read again: 5000, with it in it
        self.assertEqual(self.ledger.room(None), 4900)

    def test_base_read_with_a_new_holder(self):
        # an old base (another model's day), then an image engine started and entered with what it holds: the first
        # report reads the base again with it in it, as an empty ledger's would
        self.assertEqual(self.ledger.room(20000), 19900)
        time.sleep(0.6)
        self.ledger.hold("image", 5000, self.other())
        self.assertEqual(self.ledger.room(12000), 11900)
        self.assertEqual(self.ledger.room(None), 11900)

    def test_whole_holder_in_the_base_lets_go(self):
        # an image engine kept warm (5000, in the base), asked to let go of all it holds for a translation: its MiB are
        # room at once, without a new report
        image = self.other()
        self.ledger.hold("image", 5000, image, whole=True)
        self.assertEqual(self.ledger.room(6000), 5900)
        self.ledger.remove(self.ledger.find("image", image))
        self.assertEqual(self.ledger.room(None), 10900)

    def test_holder_in_the_base_ends(self):
        p = sleeper()
        self.ledger.hold("text", 6000, p.pid, in_base=True)
        self.assertEqual(self.ledger.room(4000), 3900)
        p.kill()
        p.wait()
        self.assertIsNone(self.ledger.room(None))               # what it gave back is not known: read again
        self.assertEqual(self.ledger.room(10000), 9900)
        self.assertIsNone(self.ledger.find("text", p.pid))

    def test_part_of_the_base_given_back_settles(self):
        # a text engine in the base ends: what part of the base it held is not known, and a report read right after
        # the release still shows the memory - it must not become the base, whose too small a value would then bound
        # every later room until the next settling
        p = sleeper()
        self.ledger.hold("text", 6000, p.pid, in_base=True)
        self.assertEqual(self.ledger.room(4000), 3900)          # the base read with the holder in it: 4000
        p.kill()
        p.wait()
        self.assertEqual(self.ledger.entries(), [])             # gone: no base known any more
        self.assertEqual(self.ledger.room(4000), 3900)          # a report that has not caught up: still no base
        self.assertEqual(self.ledger.room(10000), 9900)         # bounded by itself, not by that stale reading
        time.sleep(0.6)
        self.assertEqual(self.ledger.room(10000), 9900)         # the base read again once the settling time passed
        self.assertIsNotNone(self.ledger.reserve(9000, "image", 10000, pid=self.other()))


class LedgerFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)

    def test_files_live_in_a_private_folder_and_temporary_names_are_unique(self):
        led = vram_ledger.Ledger("gpu0", folder=str(self.folder))
        led.reserve(100, "translate", 8000, pid=os.getpid())
        root = self.folder / f"xestrata-vram-{vram_ledger._user()}"
        self.assertTrue(root.is_dir())
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        self.assertTrue((root / "gpu0.json").is_file())
        self.assertTrue((root / "gpu0.lock").is_file())
        self.assertEqual(list(root.glob("*.tmp")), [])
        seen = []
        real = os.replace

        def spy(src, dst):
            seen.append(Path(src).name)
            return real(src, dst)

        with mock.patch.object(vram_ledger.os, "replace", spy):
            led.entries()
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith("gpu0.json.") and seen[0].endswith(".tmp"))

    def test_a_planted_folder_is_refused(self):
        root = self.folder / f"xestrata-vram-{vram_ledger._user()}"
        (self.folder / "someone-else").mkdir()
        root.symlink_to(self.folder / "someone-else")
        with self.assertRaises(PermissionError):
            vram_ledger.Ledger("gpu0", folder=str(self.folder))


if __name__ == "__main__":
    unittest.main()
