"""strata_control/hy4_fit.py: the Hy4-Flash fit rule setup, launch and serve read, pinned to the model author's numbers
(tests/data/hy4_squant_fit_cases.json, the same file test_hy4_squant_fit.py reads) and to the picks the author's hardware table names."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from strata_control import hy4_fit as F

SPEC = json.loads((Path(__file__).resolve().parents[1] / "data" / "hy4_squant_fit_cases.json").read_text(encoding="utf-8"))


class AuthorsRule(unittest.TestCase):
    def test_the_constants_and_the_table_are_the_authors(self):
        for key, value in SPEC["constants"].items():
            self.assertEqual(getattr(F, key), value, key)
        for name, info in SPEC["models"].items():
            mine = F.BUILDS[name]
            self.assertEqual(mine["file"], info["file_name"])
            for k in ("footprint_gb", "dense_gb", "bodies_gb"):
                self.assertEqual(mine[k], info[k], (name, k))
            self.assertEqual(F.ENGRAM_GB, info["engram_gb"])
            self.assertAlmostEqual(mine["dense_gb"] + mine["bodies_gb"], mine["footprint_gb"], delta=0.02)

    def test_the_seven_cases(self):
        for c in SPEC["cases"]:
            res = F.compute(c["model"], c["vram_gib"], c["ram_gib"])
            self.assertAlmostEqual(res["ssd_gb"], c["expected_ssd_gb"], delta=0.05, msg=str(c))
            self.assertEqual(F.mode(res["ssd_gb"]), c["mode"], c)
            self.assertAlmostEqual(res["vram_gb"] + res["ram_gb"] + res["ssd_gb"], F.BUILDS[c["model"]]["bodies_gb"], delta=0.02)

    def test_home_pc_numbers_of_the_docs(self):
        r = F.compute("SQ-64", 12, 64)                       # the author's home PC: 5.08 GB in the card, 50.96 GB in RAM
        self.assertEqual((r["vram_gb"], r["ram_gb"], r["ssd_gb"]), (5.08, 50.96, 0.0))


class Recommendation(unittest.TestCase):
    def test_the_authors_hardware_classes(self):
        for vram, ram, want in ((12, 32, "SQ-32"), (16, 32, "SQ-32"), (8, 64, "SQ-64"), (12, 64, "SQ-64"), (16, 64, "SQ-64"),
                                (8, 96, "SQ-96"), (12, 96, "SQ-96"), (12, 128, "SQ-128"), (24, 128, "SQ-128")):
            got = F.recommend(vram, ram)
            self.assertEqual(got["build"], want, (vram, ram, got))
            self.assertTrue(got["fits"])
            self.assertEqual(got["mode"], "ram_vram_zero_ssd")

    def test_a_small_pc_still_gets_a_pick_and_the_words_say_it_streams(self):
        got = F.recommend(8, 16)
        self.assertEqual(got["build"], "SQ-32")               # the least read from the SSD, never a refusal
        self.assertFalse(got["fits"])
        self.assertEqual(got["mode"], "cross_tier_ssd")
        self.assertGreater(got["placement"]["ssd_gb"], 5)

    def test_no_card(self):
        got = F.recommend(None, 64)
        self.assertIn(got["build"], F.BUILDS)

    def test_unified_memory(self):
        # Strix Halo: the GPU maps ~107.9 GiB (measured by HY4-STRIX), SQ-96 fits, SQ-128 would leave the system no room
        self.assertEqual(F.recommend(None, 107.87, unified=True)["build"], "SQ-96")
        self.assertFalse(F.fits("SQ-128", None, 107.87, unified=True))
        # a 128 GiB Mac or Spark pool holds SQ-128
        self.assertEqual(F.recommend(None, 128, unified=True)["build"], "SQ-128")
        self.assertEqual(F.place("SQ-64", None, 128, unified=True)["ssd_gb"], 0.0)

    def test_modes(self):
        self.assertEqual(F.mode(0.0), "ram_vram_zero_ssd")
        self.assertEqual(F.mode(3.03), "near_resident_ssd")
        self.assertEqual(F.mode(22.6), "cross_tier_ssd")

    def test_rows_are_best_first(self):
        rows = F.rows(12, 64)
        self.assertEqual([r["build"] for r in rows], ["SQ-128", "SQ-96", "SQ-64", "SQ-32"])
        self.assertEqual([r["mode"] for r in rows], ["cross_tier_ssd", "cross_tier_ssd", "ram_vram_zero_ssd", "ram_vram_zero_ssd"])


class Disk(unittest.TestCase):
    def test_disk(self):
        self.assertAlmostEqual(F.disk_gb("SQ-64"), 62.03 + 67.58, places=2)
        self.assertAlmostEqual(F.disk_gb("SQ-64", engram_present=True), 62.03, places=2)
        self.assertAlmostEqual(F.disk_gb("SQ-96", vision=True), 89.58 + 67.58 + 0.934, places=2)

    def test_model_name(self):
        self.assertEqual(F.build_of_model_name("hy4-flash-sq-64"), "SQ-64")
        self.assertIsNone(F.build_of_model_name("qwen3.6-35b-a3b-ud-q4_k_xl"))
        self.assertIsNone(F.build_of_model_name("hy4-flash-sq-7"))


if __name__ == "__main__":
    unittest.main()
