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
            res = F.compute(c["model"], c["vram_gib"], c["ram_gib"], F.AUTHOR_CONTEXT)
            self.assertAlmostEqual(res["ssd_gb"], c["expected_ssd_gb"], delta=0.05, msg=str(c))
            self.assertEqual(F.mode(res["ssd_gb"]), c["mode"], c)
            self.assertAlmostEqual(res["vram_gb"] + res["ram_gb"] + res["ssd_gb"], F.BUILDS[c["model"]]["bodies_gb"], delta=0.02)

    def test_home_pc_numbers_of_the_docs(self):
        r = F.compute("SQ-64", 12, 64, F.AUTHOR_CONTEXT)                     # the author's home PC: 5.08 GB in the card, 50.96 GB in RAM
        self.assertEqual((r["vram_gb"], r["ram_gb"], r["ssd_gb"]), (5.08, 50.96, 0.0))


class Recommendation(unittest.TestCase):
    def test_the_authors_hardware_classes(self):
        for vram, ram, want in ((12, 32, "SQ-32"), (16, 32, "SQ-32"), (8, 64, "SQ-64"), (12, 64, "SQ-64"), (16, 64, "SQ-64"),
                                (8, 96, "SQ-96"), (12, 96, "SQ-96"), (12, 128, "SQ-128"), (24, 128, "SQ-128")):
            got = F.recommend(vram, ram, False, F.AUTHOR_CONTEXT)
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


class Defaults0218(unittest.TestCase):
    """The owner's 0.2.18 defaults (2026-11-19): 128K context, the drafter on (its 0.44 GB is inside every build's dense weights), images on (the 0.93 GB Vision shard,
    loaded on demand).  The KV of a longer context is the figure HY4-STRIX measured on Strix Halo: 66.3 GiB resident at 128K against 62.2 GiB at 4K, SQ-64."""

    def test_the_default_is_128k_and_the_authors_20k_is_the_base(self):
        self.assertEqual(F.DEFAULT_CONTEXT, 131072)
        self.assertEqual(F.AUTHOR_CONTEXT, 20480)
        self.assertEqual(F.kv_extra_gb(F.AUTHOR_CONTEXT), 0.0)
        self.assertEqual(F.kv_extra_gb(8192), 0.0)                           # never below the author's figure
        self.assertEqual(F.non_expert_gb(F.AUTHOR_CONTEXT), F.VRAM_NON_EXPERT_GB)

    def test_the_kv_figure_is_the_measured_one(self):
        measured_gib = 66.3 - 62.2                                           # Strix, 128K against 4K
        per_token = measured_gib * F.GIB_TO_GB / (131072 - 4096)
        self.assertAlmostEqual(F.KV_GB_PER_TOKEN, per_token, delta=per_token * 0.02)
        self.assertAlmostEqual(F.kv_extra_gb(131072) + (20480 - 4096) * F.KV_GB_PER_TOKEN, measured_gib * F.GIB_TO_GB, delta=0.05)    # the author's 20K plus the rest is the measured 4.1 GiB
        self.assertAlmostEqual(F.non_expert_gb(131072), 2.0 + 3.84, delta=0.02)

    def test_the_authors_seven_cases_still_hold_at_their_context(self):
        for c in SPEC["cases"]:
            res = F.compute(c["model"], c["vram_gib"], c["ram_gib"], F.AUTHOR_CONTEXT)
            self.assertAlmostEqual(res["ssd_gb"], c["expected_ssd_gb"], delta=0.05, msg=str(c))

    def test_the_seven_cases_at_128k_with_images(self):
        # (the author's own cases are at 20K; these are what the 0.2.18 defaults make of the same seven PCs)
        for c in SPEC["cases"]:
            res = F.compute(c["model"], c["vram_gib"], c["ram_gib"], 131072, True)
            self.assertAlmostEqual(res["vram_gb"] + res["ram_gb"] + res["ssd_gb"], F.BUILDS[c["model"]]["bodies_gb"], delta=0.02)
        # the resident cases stay resident (RAM takes what the card gives up) ...
        for model, vram, ram in (("SQ-64", 8, 64), ("SQ-64", 12, 64), ("SQ-96", 12, 96), ("SQ-128", 12, 128)):
            self.assertEqual(F.compute(model, vram, ram, 131072, True)["ssd_gb"], 0.0, (model, vram, ram))
        self.assertEqual(F.mode(F.compute("SQ-32", 12, 32, 131072, True)["ssd_gb"]), "near_resident_ssd")        # the 32 GiB class: the author's 0 GB from the SSD becomes about 3.5 GB at 128K with images
        # ... and the ones that read from the SSD read a little more
        self.assertAlmostEqual(F.compute("SQ-32", 8, 32, 131072, True)["ssd_gb"], 3.03 + 1.34 + 0.93, delta=0.05)           # the 8 GiB card had 1.34 GB of room for experts; the cache takes it all (and does not fit: context_fits says so)
        self.assertAlmostEqual(F.compute("SQ-64", 12, 32, 131072, True)["ssd_gb"], 22.60 + 3.84 + 0.93, delta=0.1)      # RAM is full, so what the cache and the image reader take from the card and the RAM comes from the SSD side

    def test_the_home_pc_12_gib_card_and_64_gib_of_ram_at_128k(self):
        r = F.compute("SQ-64", 12, 64, 131072, True)
        self.assertEqual((r["vram_gb"], r["ram_gb"], r["ssd_gb"]), (1.25, 54.79, 0.0))      # the card keeps the core weights, the 128K cache and 1.2 GB of hot experts; RAM the rest
        self.assertEqual(F.recommend(12, 64, False, 131072, True)["build"], "SQ-64")

    def test_the_images_cost_the_ram_only(self):
        with_images = F.compute("SQ-64", 12, 64, 131072, True)
        without = F.compute("SQ-64", 12, 64, 131072, False)
        self.assertEqual(with_images["vram_gb"], without["vram_gb"])
        self.assertAlmostEqual(F.compute("SQ-96", 12, 64, 131072, False)["ram_gb"] - F.compute("SQ-96", 12, 64, 131072, True)["ram_gb"], F.VISION_GB, delta=0.011)

    def test_the_drafter_is_not_added_twice(self):
        self.assertEqual(F.DRAFTER_GB, 0.44)
        self.assertGreater(F.BUILDS["SQ-32"]["dense_gb"], F.DRAFTER_GB)      # it is part of every build's dense weights (Section 1)

    def test_an_8_gib_card_is_planned_at_128k_like_every_other(self):
        # owner, 2026-11-19: 128K is the default on every card and people change it themselves: no smaller context is recommended and nothing warns.
        # The planner still places the experts: the core weights (5.8 GB) and the 128K cache and workspace (5.84 GB) do not fit 8.6 GB, so no expert sits in the card.
        self.assertFalse(hasattr(F, "recommend_context"))
        self.assertFalse(hasattr(F, "context_fits"))
        r = F.compute("SQ-64", 8, 64, 131072, True)
        self.assertEqual(r["vram_gb"], 0.0)
        self.assertEqual(r["ssd_gb"], 0.0)                                    # 64 GiB of RAM still holds every expert body
        self.assertAlmostEqual(r["ram_gb"], F.BUILDS["SQ-64"]["bodies_gb"], delta=0.01)
        self.assertEqual(F.recommend(8, 64, False, 131072, True)["build"], "SQ-64")
        self.assertGreater(F.compute("SQ-64", 8, 64, 20480, True)["vram_gb"], 0.0)       # at the author's 20K the card did keep some

    def test_unified_memory_pays_the_cache_and_the_images_from_its_pool(self):
        strix = 107.87                                                        # what the GPU maps on Strix Halo (HY4-STRIX)
        self.assertEqual(F.recommend(None, strix, True, 131072, True)["build"], "SQ-96")
        self.assertEqual(F.place("SQ-96", None, strix, True, 131072, True)["ssd_gb"], 0.0)
        self.assertGreater(F.place("SQ-128", None, strix, True, 131072, True)["ssd_gb"], 5.0)
        self.assertEqual(F.recommend(None, 128, True, 131072, True)["build"], "SQ-128")


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
