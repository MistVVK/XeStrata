# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A card that can run alone but is not in this run is said, not asked (a line, with the way to use it); the cards
used do not change, and a card named with --gpu or --gpus asks nothing.

    python -m unittest tools.test_setup_cards
"""
from __future__ import annotations

import sys
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


if __name__ == "__main__":
    unittest.main()
