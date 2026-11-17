"""Setup's low-RAM choice compares the RAM (ram_gb(): GiB) with the expert arena (MODELS arena_gb: GB of 10^9 bytes)
plus the headroom in ONE unit, GiB.  IQ3_S: arena 50.3 GB = 46.85 GiB, so the line moves at 56.85 GiB.

    python -m unittest tools.test_setup_low_ram_units
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

POD_GIB = 61_998_571_520 / 2**30      # the rented RTX 4090 container's memory.max: 57.74 GiB
PC_64GB_GIB = 62.6                    # what a 64 GB PC's MemTotal shows


class LowRamUnits(unittest.TestCase):
    def test_the_iq3s_arena_in_gib(self):
        self.assertAlmostEqual(setup.MODELS["IQ3_S"]["arena_gb"] * 1e9 / 2**30, 46.85, places=2)

    def test_the_57_7_gib_container_keeps_the_ram_copy_line(self):
        # measured there: line B peaks at 50.72 GiB anon + shmem with no OOM in six starts, and is the faster line
        self.assertFalse(setup.low_ram_needed("IQ3_S", POD_GIB))

    def test_a_48_gib_container_gets_the_low_ram_line(self):
        self.assertTrue(setup.low_ram_needed("IQ3_S", 48.0))

    def test_a_64_gb_pc_keeps_the_ram_copy_line_as_before(self):
        self.assertFalse(setup.low_ram_needed("IQ3_S", PC_64GB_GIB))

    def test_56_gib_gets_the_low_ram_line(self):
        self.assertTrue(setup.low_ram_needed("IQ3_S", 56.0))

    def test_the_line_moves_at_arena_gib_plus_headroom(self):
        edge = 50.3e9 / 2**30 + setup.LOW_RAM_HEADROOM_GB
        self.assertTrue(setup.ram_short_of_arena(edge - 0.01, 50.3))
        self.assertFalse(setup.ram_short_of_arena(edge, 50.3))

    def test_the_old_mixed_unit_band_moves(self):
        # 56.85-60.3 GiB took the low-RAM line only because GiB was compared with GB
        for ram in (57.0, 58.5, 60.29):
            self.assertFalse(setup.low_ram_needed("IQ3_S", ram), ram)


if __name__ == "__main__":
    unittest.main()
