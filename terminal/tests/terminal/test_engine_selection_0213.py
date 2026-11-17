"""Round 0213 TUICFG, step 4: choosing the engine on a PC with cards of several makes (#4971, #5111).

Every case is a temporary /sys/class/drm tree (a real two-vendor sysfs cannot be made on a test box), in both card orders."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import catalog as C
from strata_control import engine_component as ec

AMD, INTEL, NVIDIA = "0x1002", "0x8086", "0x10de"
REAL_ARC = C.intel_arc_cards


def build(root: Path, *cards) -> None:
    """cards: (name, vendor, device id or None, own memory in MiB or None); a name like card0 / card10 / card1-DP-1 is the directory."""
    for name, vendor, did, mib in cards:
        dev = root / name / "device"
        dev.mkdir(parents=True)
        if vendor:
            (dev / "vendor").write_text(vendor + "\n", encoding="utf-8")
        if did:
            (dev / "device").write_text(did + "\n", encoding="utf-8")
        if mib is not None:
            (dev / "mem_info_vram_total").write_text(str(mib * 1024 * 1024) + "\n", encoding="utf-8")


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def backend(self, *cards):
        build(self.root, *cards)
        return ec._drm_vendor_backend(str(self.root))

    def arcs(self, *cards):
        build(self.root, *cards)
        return REAL_ARC(str(self.root))


APU = ("card0", AMD, "0x164e", 512)
ARC = ("card1", INTEL, "0xe211", None)


class Backend(Tree):
    def test_an_apu_beside_an_arc_is_not_an_amd_pc_in_either_order(self):
        self.assertEqual(self.backend(APU, ARC), "sycl")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(("card0", INTEL, "0xe211", None), ("card1", AMD, "0x164e", 512)), "sycl")

    def test_an_arc_of_unknown_size_beside_an_apu(self):
        self.assertEqual(self.backend(APU, ("card1", INTEL, "0xe2f0", None)), "sycl")

    def test_a_sole_apu_stays_amd(self):
        self.assertEqual(self.backend(APU), "hip")

    def test_a_sole_arc_is_intel(self):
        self.assertEqual(self.backend(ARC), "sycl")

    def test_a_discrete_amd_card_and_nvidia_keep_their_order(self):
        self.assertEqual(self.backend(("card0", AMD, "0x7550", 16384), ARC), "hip")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(APU, ("card1", NVIDIA, "0x2204", None)), "cuda")

    def test_a_large_unified_memory_apu_is_a_card(self):
        self.assertEqual(self.backend(("card0", AMD, "0x1586", 32768)), "hip")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(("card0", AMD, "0x1586", 32768), ARC), "hip", "the older order; an explicit backend decides otherwise")

    def test_connectors_and_unreadable_entries_are_ignored(self):
        self.assertEqual(self.backend(("card0-DP-1", None, None, None), APU, ARC), "sycl")

    def test_no_card_is_none(self):
        self.assertIsNone(ec._drm_vendor_backend(str(self.root / "missing")))


class ArcCards(Tree):
    def test_both_orders_and_a_card_beside_a_connector(self):
        for cards in ([APU, ARC], [("card0", INTEL, "0xe211", None), ("card1", AMD, "0x164e", 512)], [("card0-DP-1", None, None, None), APU, ARC]):
            with self.subTest(cards=[c[0] for c in cards]):
                self.tearDown(); self.setUp()
                got = self.arcs(*cards)
                self.assertEqual([(g["name"], g["vram_gb"], g["index"]) for g in got], [("Intel Arc Pro B60", 24.0, 0)])

    def test_a_card_number_above_nine_and_an_unreadable_neighbour(self):
        got = self.arcs(("card0", INTEL, None, None), ("card2", INTEL, "0xe20b", None), ("card10", INTEL, "0xe211", None))
        self.assertEqual([g["name"] for g in got], ["Intel Arc B580", "Intel Arc Pro B60"], "a card without a device file does not hide the rest")

    def test_an_unlisted_battlemage_or_alchemist_id_still_counts_the_processor_graphics_do_not(self):
        got = self.arcs(("card0", INTEL, "0xe2f0", None), ("card1", INTEL, "0x56b1", None))
        self.assertEqual(len(got), 2)
        self.assertTrue(all(g["usable"] and g["vram_gb"] >= 8.0 for g in got))
        self.tearDown(); self.setUp()
        self.assertEqual(self.arcs(("card0", INTEL, "0x7d55", None), ("card1", INTEL, "0x46a6", None), ("card2", INTEL, "0xa7a0", None)), [])

    def test_a_sole_apu_has_no_arc(self):
        self.assertEqual(self.arcs(APU), [])


class LauncherChoice(Tree):
    """The launcher's own view: the cards, the engine's make, and what is listed as usable."""

    def computer(self, shared_gpus, env=None):
        from tests.terminal.launch_support import Home
        with Home():
            cat = C.Catalog()
            hw = {"ram_gb": 64.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": shared_gpus, "disk_free_gb": 500}
            fake = lambda root="/sys/class/drm": REAL_ARC(str(self.root))      # noqa: E731
            with mock.patch.dict(os.environ, env or {}, clear=False), mock.patch.object(cat.strata, "hardware", return_value=hw), \
                    mock.patch.object(C, "intel_arc_cards", fake), mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
                if not env:
                    os.environ.pop("STRATA_ENGINE_BACKEND", None)
                comp = cat.computer()
                return comp.engine_vendor, [g["name"] for g in comp.usable_gpus], [g["name"] for g in comp.ignored_gpus]

    IGPU = {"index": 0, "name": "AMD Radeon Graphics", "vram_gb": 0.5, "usable": False, "vendor": "amd"}
    RTX = {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}

    def test_the_arc_is_used_beside_an_apu_igpu(self):
        build(self.root, APU, ARC)
        vendor, usable, ignored = self.computer([self.IGPU])
        self.assertEqual((vendor, usable), ("intel", ["Intel Arc Pro B60"]))

    def test_an_explicit_sycl_beside_nvidia_uses_the_arc_also_when_its_model_is_not_listed(self):
        for did in ("0xe211", "0xe2f0"):
            with self.subTest(did=did):
                self.tearDown(); self.setUp()
                build(self.root, ("card0", NVIDIA, "0x2204", None), ("card1", INTEL, did, None))
                vendor, usable, ignored = self.computer([self.RTX], {"STRATA_ENGINE_BACKEND": "sycl"})
                self.assertEqual(vendor, "intel")
                self.assertEqual(len(usable), 1)
                self.assertIn("NVIDIA GeForce RTX 3090", ignored)

    def test_without_a_choice_nvidia_stays_the_first_make(self):
        build(self.root, ("card0", NVIDIA, "0x2204", None), ARC)
        vendor, usable, _ = self.computer([self.RTX])
        self.assertEqual((vendor, usable), ("nvidia", ["NVIDIA GeForce RTX 3090"]))


if __name__ == "__main__":
    unittest.main()
