"""sycl/setup_intel.py: an Intel GPU built into the processor (Lunar Lake, Arrow Lake on xe) is never taken for a discrete Arc card.

Before, an xe GPU outside the id table was listed as a card named "Intel GPU <id> (xe)", sized by its PCI BAR 2 (the graphics
aperture, not memory) and sent down the container route. Now Level Zero's integrated flag decides (ZE_DEVICE_PROPERTY_FLAG_INTEGRATED,
read through sysman), the discrete id ranges when Level Zero cannot say, never the PCI slot; setup then says integrated Intel
graphics are not supported yet and points at ./setup.sh (the CPU or another card). The A750 is unchanged. Fake sysfs trees, a fake
Level Zero; no GPU.

    python -m unittest tools.test_setup_intel_igpu
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BAR_256M = "0x0000000000000000 0x0000000000000000 0x0000000000000000\n" * 2 + "0x0000004000000000 0x000000400fffffff 0x000000000014220c\n"


def setup_intel():
    spec = importlib.util.spec_from_file_location("setup_intel_igpu_test", ROOT.parent / "engine" / "sycl" / "setup_intel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(os.name == "nt", "the fixtures are POSIX sysfs trees with links")
class IntegratedIntel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.drm = self.root / "drm"
        self.drm.mkdir()
        self.m = setup_intel()
        self.asked = []

    def gpu(self, card, pci, devid, driver, resource=BAR_256M):
        """A card entry whose device is a link to a folder named by its PCI address, with a driver link, as the kernel's."""
        dev = self.root / "pci" / pci
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x8086\n")
        (dev / "device").write_text(devid + "\n")
        (dev / "resource").write_text(resource)
        drv = self.root / "drivers" / driver
        drv.mkdir(parents=True, exist_ok=True)
        os.symlink(drv, dev / "driver")
        (self.drm / card).mkdir()
        os.symlink(dev, self.drm / card / "device")

    def ask(self, answers):
        def level_zero(pci):
            self.asked.append(pci)
            return answers.get(pci)
        return level_zero

    def devices(self, answers=None):
        return self.m.intel_devices(self.drm, ask=self.ask(answers or {}))

    def test_lunar_lake_on_xe_is_integrated_when_level_zero_says_so(self):
        self.gpu("card0", "0000:00:02.0", "0x64a0", "xe")
        cards, igpus = self.devices({"0000:00:02.0": True})
        self.assertEqual(cards, [])
        self.assertEqual(igpus, ["PCI id 64a0, xe"])

    def test_arrow_lake_on_xe_is_integrated_also_without_level_zero(self):
        self.gpu("card1", "0000:00:02.0", "0x7d51", "xe")             # the Arc 140T
        cards, igpus = self.devices()                                 # no runtime: the discrete id ranges decide
        self.assertEqual((cards, igpus), ([], ["PCI id 7d51, xe"]))

    def test_the_pci_slot_does_not_decide(self):
        self.gpu("card0", "0000:00:04.0", "0x7dd1", "xe")             # an integrated GPU at another address
        self.gpu("card1", "0000:00:02.0", "0xe20b", "xe")             # a B580 at the integrated GPU's usual address
        cards, igpus = self.devices()
        self.assertEqual([c["name"] for c in cards], ["Intel Arc B580"])
        self.assertEqual(igpus, ["PCI id 7dd1, xe"])

    def test_level_zero_wins_over_the_id_ranges(self):
        self.gpu("card0", "0000:03:00.0", "0xe2f0", "xe")             # a Battlemage id outside the table
        self.gpu("card1", "0000:04:00.0", "0x7d55", "xe")
        cards, igpus = self.devices({"0000:03:00.0": True, "0000:04:00.0": False})
        self.assertEqual([c["devid"] for c in cards], ["7d55"])
        self.assertEqual(igpus, ["PCI id e2f0, xe"])

    def test_an_unlisted_battlemage_is_still_a_card_sized_as_before(self):
        self.gpu("card0", "0000:03:00.0", "0xe2f0", "xe")
        cards, igpus = self.devices()
        self.assertEqual(igpus, [])
        self.assertIn("e2f0", cards[0]["name"])
        self.assertEqual(cards[0]["vram_gb"], 0.25, "a discrete card outside the table: its BAR, as before")

    def test_the_a750_is_unchanged_and_level_zero_is_not_asked(self):
        self.gpu("card0", "0000:2d:00.0", "0x56a1", "i915")
        cards, igpus = self.devices()
        self.assertEqual(cards, [{"index": 0, "name": "Intel Arc A750", "vram_gb": 8.0, "arch": "i915", "driver": "i915", "devid": "56a1"}])
        self.assertEqual((igpus, self.asked), ([], []))
        self.assertEqual(self.m.intel_gpus(self.drm, ask=self.ask({})), cards)

    def test_an_a750_beside_an_arrow_lake_igpu_keeps_the_card(self):
        self.gpu("card0", "0000:00:02.0", "0x7d51", "xe")
        self.gpu("card1", "0000:2d:00.0", "0x56a1", "i915")
        cards, igpus = self.devices({"0000:00:02.0": True})
        self.assertEqual([c["name"] for c in cards], ["Intel Arc A750"])
        self.assertEqual(igpus, ["PCI id 7d51, xe"])

    def test_setup_on_an_igpu_only_pc_says_so_and_points_at_setup_sh(self):
        self.gpu("card0", "0000:00:02.0", "0x64a0", "xe")
        found = self.devices({"0000:00:02.0": True})
        with mock.patch.object(self.m, "intel_devices", return_value=found), mock.patch.object(self.m.S, "WIN", False), \
                mock.patch.object(self.m.S, "fail", side_effect=SystemExit) as fail:
            with self.assertRaises(SystemExit):
                self.m.install([])
        what, nxt = fail.call_args[0][:2]
        self.assertIn("built into the processor", what)
        self.assertIn("PCI id 64a0, xe", what)
        self.assertIn("./setup.sh", nxt)

    def test_no_intel_gpu_at_all_keeps_the_old_message(self):
        with mock.patch.object(self.m, "intel_devices", return_value=([], [])), mock.patch.object(self.m.S, "WIN", False), \
                mock.patch.object(self.m.S, "fail", side_effect=SystemExit) as fail:
            with self.assertRaises(SystemExit):
                self.m.install([])
        self.assertIn("no Intel Arc found", fail.call_args[0][0])


class LevelZeroFlag(unittest.TestCase):
    """serve/telemetry.py _Zes.integrated(): ZE_DEVICE_PROPERTY_FLAG_INTEGRATED from zesDeviceGetProperties, through a fake loader."""

    def zes(self, vendor=0x8086, flags=0, rc=0):
        from serve import telemetry

        def get_properties(dev, buf):
            assert ctypes.c_uint32.from_buffer(buf, 0).value == 0x1 and ctypes.c_uint32.from_buffer(buf, 16).value == 0x3
            ctypes.c_uint32.from_buffer(buf, 36).value = vendor
            ctypes.c_uint32.from_buffer(buf, 44).value = flags
            return rc
        z = telemetry._Zes.__new__(telemetry._Zes)
        z.lib, z.dev, z.mems = mock.Mock(zesDeviceGetProperties=get_properties), ctypes.c_void_p(1), []
        return z

    def test_the_flag_says_integrated_or_not(self):
        self.assertIs(self.zes(flags=0x1).integrated(), True)
        self.assertIs(self.zes(flags=0x4).integrated(), False)

    def test_an_error_or_a_foreign_layout_says_nothing(self):
        self.assertIsNone(self.zes(rc=0x78000001).integrated())
        self.assertIsNone(self.zes(vendor=0).integrated())

    def test_no_runtime_says_nothing(self):
        from serve import telemetry
        z = telemetry._Zes.__new__(telemetry._Zes)
        z.lib, z.dev, z.mems = None, None, []
        self.assertIsNone(z.integrated())


if __name__ == "__main__":
    unittest.main()
