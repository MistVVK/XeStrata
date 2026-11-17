"""An Arc card is listed once (found with #5111's A750 check, p1006 11-16): on a PC whose only card is an Arc and whose engine is the Intel one,
the shared hardware reader (tools/strata_mcp.py through strata_control/intel_app.py cards_to_add) already holds the card, and the Terminal's
Catalog.hardware added it a second time: "Arc A750 8 GB x2", two usable cards with the same number. Fake reports; no card."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from strata_cli.launch import catalog as C

ARC = {"vendor": "intel", "name": "Intel Arc A750", "vram_gb": 7.9, "usable": True, "index": 0}
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}


class ArcListedOnce(unittest.TestCase):
    def hardware(self, shared_gpus, arc_cards, env=None):
        from tests.terminal.launch_support import Home
        with Home():
            cat = C.Catalog()
            hw = {"ram_gb": 64.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": [dict(g) for g in shared_gpus], "disk_free_gb": 500}
            with mock.patch.dict(os.environ, env or {}), mock.patch.object(cat.strata, "hardware", return_value=hw), \
                    mock.patch.object(C, "intel_arc_cards", lambda root="/sys/class/drm": [dict(c) for c in arc_cards]), \
                    mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
                comp = cat.computer()
                return [g["name"] for g in comp.gpus], comp.line()

    def test_an_arc_the_shared_reader_already_holds_is_not_added_again(self):
        names, line = self.hardware([ARC], [ARC], {"STRATA_ENGINE_BACKEND": "sycl"})
        self.assertEqual(names, ["Intel Arc A750"])
        self.assertNotIn("x2", line)

    def test_an_arc_beside_nvidia_is_still_added(self):
        names, _ = self.hardware([RTX], [ARC], {"STRATA_ENGINE_BACKEND": "sycl"})
        self.assertEqual(names, ["NVIDIA GeForce RTX 3090", "Intel Arc A750"])

    def test_an_arc_on_its_own_is_added_when_the_shared_reader_left_it_out(self):
        names, _ = self.hardware([], [ARC])
        self.assertEqual(names, ["Intel Arc A750"])


if __name__ == "__main__":
    unittest.main()
