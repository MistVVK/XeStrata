"""0.2.10: UD-Q4_K_XL on a Strix Halo is warned about (warn only). Remove with the stall fix."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import gpu_fit

NV = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 5060", "vram_gb": 8.0, "usable": True, "index": 0}


class StrixQ4(unittest.TestCase):
    STRIX = {"vendor": "amd", "name": "AMD Radeon 8060S", "arch": "gfx1151", "vram_gb": 96.0}

    def test_warns_for_the_one_model_on_the_one_chip_and_never_otherwise(self):
        note = gpu_fit.strix_q4_stall_note("ud-q4_k_xl", [self.STRIX])
        self.assertIn("7 of 30", note)
        self.assertIn("UD-IQ4_XS", note)
        self.assertIsNone(gpu_fit.strix_q4_stall_note("UD-IQ4_XS", [self.STRIX]))
        self.assertIsNone(gpu_fit.strix_q4_stall_note("UD-Q4_K_XL", [NV]))
        self.assertIsNone(gpu_fit.strix_q4_stall_note(None, None))

    def test_the_launch_notice(self):
        from strata_cli.launch import cmd, settings as S
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home():
            cat = catalog({**HW_12, "gpus": [self.STRIX]})
            s, notes = cmd.apply_defaults(S.LaunchSettings(family="qwen", model="UD-Q4_K_XL"), cat)
            self.assertTrue([n for n in notes if "7 of 30" in n.text], [n.text for n in notes])
            self.assertTrue(all(n.level == "warn" for n in notes if "7 of 30" in n.text))


if __name__ == "__main__":
    unittest.main()
