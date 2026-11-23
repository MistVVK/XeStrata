# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A card that can run alone but is not in this run is said, not asked (a line, with the way to use it); the cards
used do not change, and a card named with --gpu or --gpus asks nothing.

    python -m unittest tools.test_setup_cards
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def card(i, name, gb, integrated=False, problem=None):
    return {"index": i, "name": name, "vram_gb": gb, "integrated": integrated, "problem": problem}


class UnusedCardSaid(unittest.TestCase):
    def notes(self, found, used):
        with mock.patch.object(setup, "gpu_problem", side_effect=lambda g: g["problem"]):
            return setup.unused_card_notes(found, used)

    def test_the_card_left_out_is_named_with_the_way_to_use_it(self):
        lines = self.notes([card(0, "NVIDIA GeForce RTX 4090", 24.0), card(1, "NVIDIA GeForce RTX 2060", 5.9)], {0})
        self.assertEqual(len(lines), 1)
        self.assertIn("GPU 1 (NVIDIA GeForce RTX 2060, 6 GB) is not used", lines[0])
        self.assertIn("--gpus 0,1", lines[0])

    def test_a_card_chosen_for_a_split_says_nothing(self):
        self.assertEqual(self.notes([card(0, "a", 24.0), card(1, "b", 12.0)], {0, 1}), [])

    def test_the_processors_graphics_is_not_called_unused(self):
        found = [card(0, "Intel Arc Pro B70", 31.0), card(1, "UHD 770", 0.0, integrated=True)]
        self.assertEqual(self.notes(found, {0}), [])

    def test_a_card_that_cannot_be_used_is_not_called_unused(self):
        self.assertEqual(self.notes([card(0, "a", 24.0), card(1, "b", 6.0, problem="too old")], {0}), [])


class PciProductName(unittest.TestCase):
    """A product name in udev's PCI database beats the die name in pci.ids (the B70's "Battlemage G31")."""

    def test_product_name_and_fallback(self):
        from serve import hardware
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "pci.hwdb"
            db.write_text("\n".join(("pci:v00008086d0000E223*",
                                     " ID_MODEL_FROM_DATABASE=Battlemage G31 [Arc Pro B70]", "",
                                     "pci:v000010DEd00002786*", " ID_MODEL_FROM_DATABASE=AD104 [GeForce RTX 4070]", "",
                                     "pci:v00000001d00000002*", " ID_MODEL_FROM_DATABASE=a die [Intel Graphics]", "",
                                     "pci:v00000001d00000003*", " ID_MODEL_FROM_DATABASE=a die without a product", "")))
            with mock.patch.object(hardware, "UDEV_PCI_HWDB", (str(db),)):
                self.assertEqual(setup.pci_name(0x8086, 0xe223), "Intel Arc Pro B70")
                self.assertEqual(hardware.pci_product_name("0x8086", "0xe223"), "Arc Pro B70")
                self.assertEqual(hardware.pci_product_name(0x10de, 0x2786), "GeForce RTX 4070")
                self.assertIsNone(hardware.pci_product_name(1, 2))               # no product in the brackets
                self.assertIsNone(hardware.pci_product_name(1, 3))               # no brackets
                self.assertIsNone(hardware.pci_product_name(9, 9))               # no entry


if __name__ == "__main__":
    unittest.main()
