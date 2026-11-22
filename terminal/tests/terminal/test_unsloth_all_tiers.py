"""Comprehensive Unsloth tier planning tests across ALL VRAM/RAM tiers and Unsloth quants.

Verifies:
  - Discrete tiers: 4, 6, 8, 10, 12, 16, 24, 32 GB across NVIDIA CUDA, AMD HIP (RDNA2/3/4), and Intel Arc SYCL
  - Unified memory tiers: NVIDIA DGX Spark (128 GB), Apple Mac Metal (16/24/32/48/64/128 GB), AMD Strix Halo (64/128 GB)
  - Primary Unsloth quants (UD-Q4_K_XL, Q8_0, UD-IQ4_XS, UD-IQ3_XXS) + full Unsloth catalog
    (UD-Q6_K_XL, UD-Q5_K_XL, UD-Q3_K_XL, UD-Q2_K_XL, UD-IQ2_XXS, UD-IQ2_XS, UD-Q2_0, UD-IQ1_M, UD-IQ1_S)
  - Issue #6354: auto --kv int8 + elastic --kv-resident on <= 16 GB GPUs and <= 24 GB GPUs at >= 64K context
  - Issue #4870: coverage-based adaptive defaults (2 / 192 / 0.92) for 15%..65% VRAM coverage
  - Clean 4 GB & 6 GB GPU unlock (0-slot / dense-streaming / RAM+SSD cold tiers, never refuses or OOMs)
  - Heterogeneous Unsloth expert slot sizing outperforms uniform max_blob sizing on mixed-quant packs
"""

from __future__ import annotations

import unittest

import setup
from strata_control import gpu_fit as G

PRIMARY_QUANTS = ("UD-Q4_K_XL", "Q8_0", "UD-IQ4_XS", "UD-IQ3_XXS")
ALL_UNSLOTH_QUANTS = (
    "UD-Q4_K_XL", "Q8_0", "UD-IQ4_XS", "UD-IQ4_XS-v31", "UD-IQ3_XXS",
    "UD-Q6_K_XL", "UD-Q5_K_XL", "UD-Q3_K_XL", "UD-Q2_K_XL",
    "UD-IQ2_XXS", "UD-IQ2_XS", "UD-Q2_0", "UD-IQ1_M", "UD-IQ1_S",
)
DISCRETE_TIERS = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0)


class DiscreteTiersAllQuants(unittest.TestCase):
    def test_every_discrete_tier_and_primary_quant_is_runnable_across_backends(self):
        backends = (
            ("cuda", "NVIDIA GeForce RTX", "sm_89"),
            ("hip", "AMD Radeon RX RDNA3", "gfx1100"),
            ("hip", "AMD Radeon RX RDNA4", "gfx1201"),
            ("hip", "AMD Radeon RX RDNA2", "gfx1030"),
            ("sycl", "Intel Arc B580", "xe2"),
        )
        for backend, name, arch in backends:
            for vram in DISCRETE_TIERS:
                for model in PRIMARY_QUANTS:
                    plan = G.unsloth_tier_plan(model, vram_gb=vram, ram_gb=64.0, ctx=32768,
                                               backend=backend, gpu_name=name, arch=arch)
                    self.assertTrue(plan["runnable"], (backend, vram, model))
                    self.assertEqual(plan["backend"], backend)
                    self.assertEqual(plan["model"], model)
                    self.assertFalse(plan["uma"])
                    self.assertIn("--kv", plan["flags"])

    def test_all_14_unsloth_quants_supported_across_all_discrete_tiers(self):
        for model in ALL_UNSLOTH_QUANTS:
            self.assertIn(model, G.UNSLOTH_QUANTS)
            for vram in DISCRETE_TIERS:
                plan = G.unsloth_tier_plan(model, vram_gb=vram, ram_gb=64.0, ctx=32768)
                self.assertTrue(plan["runnable"], (model, vram))
                self.assertEqual(plan["tier"], f"discrete_{int(vram)}gb")

    def test_4gb_and_6gb_unlock_cleanly_never_refuse_or_oom(self):
        for vram in (4.0, 6.0):
            for ram in (16.0, 32.0, 64.0):
                for model in PRIMARY_QUANTS:
                    plan = G.unsloth_tier_plan(model, vram_gb=vram, ram_gb=ram, ctx=32768)
                    self.assertTrue(plan["runnable"], (vram, ram, model))
                    self.assertFalse(plan["mtp"], f"MTP must stay off on {vram} GB to protect VRAM")
                    self.assertEqual(plan["kv"], "int8")
                    self.assertIsNotNone(plan["kv_resident"])
                    if vram <= 4.5:
                        self.assertEqual(plan["expert_cache_mode"], "zero_slot")
                        self.assertEqual(plan["estimated_slots"], 0)
                        self.assertIn("--expert-cache", plan["flags"])
                        self.assertIn("--no-prefill-borrow", plan["flags"])
                    if ram == 16.0:
                        self.assertEqual(plan["ram_tier"], "ssd_streaming")
                        self.assertIn("--ssd-experts", plan["flags"])
        # setup.unsloth_small_card_plan also never returns None for 4 GB / 6 GB cards
        for vram in (4.0, 6.0):
            sp = setup.unsloth_small_card_plan(vram, ctx=32768, kv="int8", reserve_mib=None)
            self.assertIsNotNone(sp, f"setup.unsloth_small_card_plan refused {vram} GB")
            c0, k0, mtp0, slack0 = sp
            self.assertFalse(mtp0)
            self.assertGreaterEqual(slack0, 0.0)
            lines = setup.unsloth_small_card_lines("UD-Q4_K_XL", vram, sp, start_ok=True)
            self.assertTrue(any("0-slot" in ln for ln in lines))

    def test_heterogeneous_sizing_beats_uniform_max_blob(self):
        for model in ("UD-Q4_K_XL", "UD-IQ4_XS", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ1_M"):
            plan = G.unsloth_tier_plan(model, vram_gb=16.0, ram_gb=64.0, ctx=32768)
            self.assertGreater(plan["estimated_slots"], plan["uniform_max_blob_slots"],
                               f"{model} heterogeneous sizing should fit strictly more slots than uniform max_blob")


class KvInt8AndElasticResident6354(unittest.TestCase):
    def test_auto_int8_kv_on_16gb_and_below_at_any_context(self):
        for vram in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0):
            for ctx in (4096, 8192, 16384, 32768, 65536):
                self.assertEqual(G.unsloth_kv_default(vram, ctx), "int8", (vram, ctx))

    def test_auto_int8_and_elastic_kv_resident_on_24gb_at_64k(self):
        self.assertEqual(G.unsloth_kv_default(24.0, 65536), "int8")
        self.assertEqual(G.unsloth_kv_resident_default(24.0, 64.0, 65536), 32768)
        self.assertIsNone(G.unsloth_kv_resident_default(24.0, 64.0, 32768))
        # Explicit user override is always respected
        self.assertEqual(G.unsloth_kv_default(12.0, 32768, explicit_kv="q4_0"), "q4_0")
        self.assertEqual(G.unsloth_kv_resident_default(12.0, 64.0, 32768, explicit_kv_resident=8192), 8192)

    def test_12gb_and_16gb_enable_elastic_kv_resident_at_32k_and_64k(self):
        for vram in (8.0, 10.0, 12.0, 16.0):
            p32 = G.unsloth_tier_plan("UD-Q4_K_XL", vram_gb=vram, ram_gb=64.0, ctx=32768)
            self.assertEqual(p32["kv"], "int8")
            self.assertEqual(p32["kv_resident"], 16384, (vram, p32))
            self.assertIn("--kv-resident", p32["flags"])
        p16_64 = G.unsloth_tier_plan("UD-IQ4_XS", vram_gb=16.0, ram_gb=64.0, ctx=65536)
        self.assertEqual(p16_64["kv"], "int8")
        self.assertEqual(p16_64["kv_resident"], 32768)

    def test_setup_kv_streaming_wanted_works_for_unsloth_budget_models(self):
        # 12 GB + 32 GB RAM running UD-Q4_K_XL at 32K or 64K must allow KV streaming (budget subtracts KV RAM)
        self.assertTrue(setup.kv_streaming_wanted("UD-Q4_K_XL", 32768, "int8", 32.0, vram_gb=12.0))
        self.assertTrue(setup.kv_streaming_wanted("UD-Q4_K_XL", 65536, "int8", 32.0, vram_gb=12.0))
        self.assertTrue(setup.kv_streaming_wanted("Q8_0", 65536, "int8", 64.0, vram_gb=24.0))
        self.assertFalse(setup.kv_streaming_wanted("UD-Q4_K_XL", 65536, "int8", 64.0, choice="off", vram_gb=12.0))


class AdaptiveDefaults4870(unittest.TestCase):
    def test_mid_coverage_15_to_65_percent_uses_2_192_092(self):
        import math
        for cov_pct in (15, 20, 30, 45, 65):
            slots = int(math.ceil(G.UNSLOTH_TOTAL_EXPERT_PAIRS * cov_pct / 100.0)) if cov_pct == 15 else int(G.UNSLOTH_TOTAL_EXPERT_PAIRS * cov_pct / 100.0)
            d = G.adaptive_cache_defaults(slots)
            self.assertEqual(d["mode"], "mid_coverage_fast", cov_pct)
            self.assertEqual((d["adapt_every"], d["adapt_swaps"], d["adapt_decay"]), (2, 192, 0.92))

    def test_low_and_high_coverage_bands_and_explicit_overrides(self):
        low = G.adaptive_cache_defaults(int(G.UNSLOTH_TOTAL_EXPERT_PAIRS * 0.10))
        self.assertEqual(low["mode"], "low_vram_dampened")
        self.assertEqual((low["adapt_every"], low["adapt_swaps"], low["adapt_decay"]), (4, 16, 0.7))

        high = G.adaptive_cache_defaults(int(G.UNSLOTH_TOTAL_EXPERT_PAIRS * 0.80))
        self.assertEqual(high["mode"], "high_coverage_standard")
        self.assertEqual((high["adapt_every"], high["adapt_swaps"], high["adapt_decay"]), (4, 96, 0.7))

        # User CLI overrides are preserved
        ovr = G.adaptive_cache_defaults(int(G.UNSLOTH_TOTAL_EXPERT_PAIRS * 0.35),
                                        adapt_every=6, adapt_swaps=64, adapt_decay=0.8)
        self.assertEqual((ovr["adapt_every"], ovr["adapt_swaps"], ovr["adapt_decay"]), (6, 64, 0.8))

    def test_tier_plan_emits_4870_flags_in_mid_coverage_tiers(self):
        # 16 GB / 24 GB / 32 GB running UD-Q4_K_XL / UD-IQ4_XS / UD-IQ3_XXS land squarely in 15%-65% coverage
        plan = G.unsloth_tier_plan("UD-Q4_K_XL", vram_gb=24.0, ram_gb=64.0, ctx=32768)
        self.assertEqual(plan["adaptive"]["mode"], "mid_coverage_fast")
        self.assertIn("--adapt-every", plan["flags"])
        self.assertIn("2", plan["flags"])
        self.assertIn("192", plan["flags"])
        self.assertIn("0.92", plan["flags"])


class UnifiedMemoryTiers(unittest.TestCase):
    def test_dgx_spark_128gb_all_primary_quants(self):
        for model in PRIMARY_QUANTS:
            plan = G.unsloth_tier_plan(model, vram_gb=128.0, ram_gb=128.0, ctx=32768,
                                       backend="cuda", uma=True, gpu_name="NVIDIA GB10 DGX Spark", arch="sm_121")
            self.assertEqual(plan["tier"], "dgx_spark")
            self.assertTrue(plan["runnable"])
            self.assertTrue(plan["uma"])
            self.assertEqual(plan["kv"], "int8")
            self.assertIsNone(plan["kv_resident"])
            self.assertIn("--mmap-experts", plan["flags"])
            self.assertIn("auto:32768", plan["flags"])
            if model == "Q8_0":
                self.assertEqual(plan["expert_cache_mode"], "spark_mmap_budget")
                self.assertGreater(plan["resident_budget_gib"], 0)
            else:
                self.assertEqual(plan["expert_cache_mode"], "spark_mmap_resident")

    def test_mac_metal_16_to_128gb_all_primary_quants(self):
        for unified in (16.0, 24.0, 32.0, 48.0, 64.0, 128.0):
            for model in PRIMARY_QUANTS:
                plan = G.unsloth_tier_plan(model, vram_gb=0.0, ram_gb=unified, ctx=32768,
                                           backend="metal", uma=True, gpu_name="Apple M4 Max")
                self.assertEqual(plan["tier"], f"mac_metal_{int(unified)}gb")
                self.assertTrue(plan["runnable"])
                self.assertEqual(plan["backend"], "metal")
                self.assertEqual(plan["kv"], "int8")
                self.assertIsNone(plan["kv_resident"])
                if unified <= 64.0 and model in ("UD-Q4_K_XL", "Q8_0"):
                    self.assertEqual(plan["expert_cache_mode"], "metal_stream")
                    self.assertIn("--mmap-experts", plan["flags"])
                if unified == 128.0 and model in ("UD-Q4_K_XL", "UD-IQ4_XS", "UD-IQ3_XXS"):
                    self.assertEqual(plan["expert_cache_mode"], "metal_resident")
                    self.assertTrue(plan["mtp"])

    def test_strix_halo_64gb_and_128gb_all_primary_quants(self):
        for total_ram in (64.0, 128.0):
            for model in PRIMARY_QUANTS:
                plan = G.unsloth_tier_plan(model, vram_gb=total_ram * 0.75, ram_gb=total_ram, ctx=32768,
                                           backend="hip", uma=True, gpu_name="AMD Radeon 8060S (Strix Halo)", arch="gfx1151")
                self.assertEqual(plan["tier"], f"strix_halo_{int(total_ram)}gb")
                self.assertTrue(plan["runnable"])
                self.assertEqual(plan["backend"], "hip")
                self.assertEqual(plan["kv"], "int8")
                self.assertIsNone(plan["kv_resident"])
                if model == "UD-Q4_K_XL":
                    self.assertIn(G.STRIX_Q4_NOTE, plan["notes"])
                if total_ram == 64.0:
                    # 64 GB Strix Halo must use bounded cache + budget to avoid 2x pin OOM
                    self.assertEqual(plan["expert_cache_mode"], "uma_bounded_cache")
                    self.assertIn("--resident-budget-gib", plan["flags"])
                elif model != "Q8_0":
                    self.assertEqual(plan["expert_cache_mode"], "full_uma")


if __name__ == "__main__":
    unittest.main()
