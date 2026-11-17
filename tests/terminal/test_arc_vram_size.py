"""An Intel Arc card is sized from its own memory (follow-up to #5111): the kernel's size file, else Level Zero sysman, and the id table only
when neither says. Before, an Arc B-series whose PCI id is not in the table counted as a 10 GB card whatever it had (a 24 GB card got the
context and reserve of a 10 GB one), and an A770 counted as 8 GB also when it has 16. Fake /sys/class/drm trees; sysman is replaced."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import catalog as C

GIB = 2 ** 30


class FakeZes:
    """serve/telemetry.py _Zes for one card: (used, total) bytes for the PCI address it knows, (None, None) otherwise."""
    sizes: dict = {}
    asked: list = []

    def __init__(self, pci, lib=None):
        FakeZes.asked.append(pci)
        self.pci = pci

    def read(self):
        total = FakeZes.sizes.get(self.pci)
        return (0, total) if total else (None, None)


class ArcSize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        FakeZes.sizes, FakeZes.asked = {}, []
        cache = getattr(C, "_SYSMAN_TOTAL", {})                 # (absent before this change: the test then fails on its figures)
        cache.clear()
        self.addCleanup(cache.clear)
        patch = mock.patch("serve.telemetry._Zes", FakeZes)
        patch.start()
        self.addCleanup(patch.stop)

    def card(self, name, did, files=None, pci=None):
        """A card entry; `pci`: the device folder is a link to a folder named by that PCI address (as the kernel's is)."""
        if pci:
            target = self.root / "pci" / pci
            target.mkdir(parents=True)
            (self.root / name).mkdir(parents=True)
            os.symlink(target, self.root / name / "device")
        else:
            (self.root / name / "device").mkdir(parents=True)
        dev = self.root / name / "device"
        (dev / "vendor").write_text("0x8086\n")
        (dev / "device").write_text(did + "\n")
        for rel, text in (files or {}).items():
            p = self.root / name / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def sizes(self):
        return [(c["name"], c["vram_gb"]) for c in C.intel_arc_cards(str(self.root))]

    def test_an_unknown_battlemage_reporting_24_gb_in_sysfs_is_a_24_gb_card(self):
        self.card("card0", "0xe2f0", {"device/tile0/physical_vram_size_bytes": f"{24 * GIB}\n"})
        self.assertEqual(self.sizes(), [("Intel Arc", 24.0)])
        self.assertEqual(FakeZes.asked, [], "a size file answers: sysman is not asked")

    @unittest.skipUnless(hasattr(os, "symlink") and os.name != "nt", "a PCI-named device link needs a POSIX file system")
    def test_an_unknown_battlemage_reporting_24_gb_through_sysman_is_a_24_gb_card(self):
        self.card("card0", "0xe2f0", pci="0000:03:00.0")
        FakeZes.sizes = {"0000:03:00.0": 24 * GIB}
        self.assertEqual(self.sizes(), [("Intel Arc", 24.0)])
        self.assertEqual(C.intel_arc_cards(str(self.root))[0]["vram_gb"], 24.0)
        self.assertEqual(FakeZes.asked, ["0000:03:00.0"], "sysman is asked once per card")

    def test_a_listed_card_takes_its_real_size_over_the_table(self):
        self.card("card0", "0x56a0", {"lmem_total_bytes": f"{16 * GIB}\n"})       # A770: the table says 8 (the 8 GB model shares the id)
        self.assertEqual(self.sizes(), [("Intel Arc A770", 16.0)])

    def test_the_table_stands_when_nothing_says(self):
        self.card("card0", "0xe2f0")
        self.card("card1", "0xe222", {"device/tile0/physical_vram_size_bytes": "0\n"})
        self.card("card2", "0x56a1", {"device/lmem_total_bytes": "not a number\n"})
        self.assertEqual(self.sizes(), [("Intel Arc", 10.0), ("Intel Arc Pro B65", 32.0), ("Intel Arc A750", 8.0)])
        self.assertEqual(FakeZes.asked, [], "a device folder that is not a PCI address is never looked up")

    @unittest.skipUnless(hasattr(os, "symlink") and os.name != "nt", "a PCI-named device link needs a POSIX file system")
    def test_sysman_without_an_answer_keeps_the_table(self):
        self.card("card0", "0xe20b", pci="0000:04:00.0")
        self.assertEqual(self.sizes(), [("Intel Arc B580", 12.0)])

    def test_the_processor_graphics_stay_out_whatever_they_report(self):
        self.card("card0", "0x7d55", {"device/tile0/physical_vram_size_bytes": f"{16 * GIB}\n"})
        self.assertEqual(self.sizes(), [])


if __name__ == "__main__":
    unittest.main()
