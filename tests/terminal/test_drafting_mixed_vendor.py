"""#5109: on a computer with an idle NVIDIA card beside the AMD card the engine serves from (an RTX 3060 next to an RX 7900 XTX; the lent
RTX 3080 + RX 6700 XT test box), detect_hardware() answers "cuda" with the NVIDIA card first.  The drafter's support line, card class and
confidence must come from the engine's own family (the config's "backend"), not from the idle card.  No GPU needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import drafting  # noqa: E402

MIXED = ("cuda", ["sm_86", "gfx1100"])


class HardwareFor(unittest.TestCase):
    def test_the_config_family_wins_with_its_own_cards(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            self.assertEqual(drafting.hardware_for("hip"), ("hip", ["gfx1100"]))
            self.assertEqual(drafting.hardware_for("cuda"), ("cuda", ["sm_86"]))
            self.assertEqual(drafting.hardware_for("HIP"), ("hip", ["gfx1100"]))

    def test_no_family_or_another_family_keeps_detect_hardware(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            self.assertEqual(drafting.hardware_for(None), MIXED)
            self.assertEqual(drafting.hardware_for(""), MIXED)
            self.assertEqual(drafting.hardware_for("metal"), MIXED)
            self.assertEqual(drafting.hardware_for("sycl"), MIXED)

    def test_one_vendor_computers_are_unchanged(self):
        for hw, fam in ((("cuda", ["sm_120"]), "cuda"), (("hip", ["gfx1201"]), "hip"), (("hip", ["gfx1151"]), "hip")):
            with mock.patch.object(drafting, "detect_hardware", return_value=hw):
                self.assertEqual(drafting.hardware_for(fam), hw)

    def test_class_and_support_follow_the_serving_card(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            backend, archs = drafting.hardware_for("hip")
        self.assertEqual(drafting.card_class(archs), drafting.card_class(["gfx1100"]))
        self.assertNotEqual(drafting.card_class(archs), drafting.card_class(["sm_86"]))
        self.assertEqual(drafting.hardware_support(backend, archs), drafting.hardware_support("hip", ["gfx1100"]))


class ServerStart(unittest.TestCase):
    """serve/server.py engine_args hands drafting.apply the serving card's family, cards and class."""

    def test_engine_args_on_a_mixed_computer(self):
        from serve import server
        seen = {}

        def apply(shaped, **kw):
            seen.update(kw)
            return mock.Mock(note="", flags=[], vocab="", drafter="off")
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(drafting, "apply", side_effect=apply), \
                mock.patch.object(server, "drafting_vram_mib", return_value=24560.0):
            try:
                server.engine_args({"args": ["--max-context", "4096"], "backend": "hip"}, announce=False)
            except Exception:                                 # noqa: BLE001 - only the drafting call is under test here
                pass
        self.assertEqual(seen.get("backend"), "hip")
        self.assertEqual(seen.get("archs"), ["gfx1100"])
        self.assertEqual(seen.get("klass"), drafting.card_class(["gfx1100"]))


if __name__ == "__main__":
    unittest.main()
