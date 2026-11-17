"""G16 (Q4-P8, 2026-11-08): a Windows Strix Halo config with no picture reader (or pictures off) has no --vram-reserve-mib, and the engine then applies its
Windows floor of 2560 MiB, which on the shared pool only moves memory from the expert cache into the RAM budget's page-locked complement: UD-Q4_K_XL never
started (the prompt path's own buffers found 0 MiB).  Setup writes the engine's own default, 700, on every single-card Windows Strix Halo config that has no other reserve; every other cell is pinned by the golden files.

  python -m pytest tests/snapshot/test_strix_reserve.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive  # noqa: E402


def reserve(record: dict):
    args = record["config_file"]["args"]
    return args[args.index("--vram-reserve-mib") + 1] if "--vram-reserve-mib" in args else None


class StrixReserve(unittest.TestCase):
    def test_every_single_card_strix_set_without_pictures_gets_the_engines_own_700(self):
        # one rule for every size: the big expert set that hung (UD-Q4_K_XL, 77 GB) and the small ones (UD-IQ4_XS-v31 59.5 GB, IQ3_S 50.3 GB)
        for model in ("ud_q4_k_xl", "ud_iq4_xs_v31", "iq3_s"):
            with self.subTest(model=model):
                rec = drive("strix_halo_win", model, "setup")
                self.assertEqual(rec["inputs"]["vision"], "none")
                self.assertEqual(reserve(rec), "700")
                self.assertEqual(rec["config_file"]["args"].count("--vram-reserve-mib"), 1)
                self.assertIn("700", rec["engine"]["argv"])


if __name__ == "__main__":
    unittest.main()
