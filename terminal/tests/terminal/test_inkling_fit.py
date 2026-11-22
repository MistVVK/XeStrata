"""Unit tests for Inkling-2-Mini hardware auto-selection, shape_engine registration, and HF_TOKEN auth downloader."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import inkling_family, inkling_fit, pictures, shape_engine, transfer


class TestInklingFitAndHardwareSelection(unittest.TestCase):
    def test_discrete_24gb_plus_selects_sq20_full_vram(self):
        rec = inkling_fit.recommend(vram_gib=24, ram_gib=64)
        self.assertEqual(rec["build"], "SQ-20")
        self.assertEqual(rec["mode"], "full_vram")
        self.assertTrue(rec["fits"])
        self.assertEqual(rec["placement"]["ram_gb"], 0.0)
        self.assertEqual(rec["placement"]["ssd_gb"], 0.0)

        rec_5090 = inkling_fit.recommend(vram_gib=32, ram_gib=128)
        self.assertEqual(rec_5090["build"], "SQ-20")
        self.assertEqual(rec_5090["mode"], "full_vram")

    def test_uma_32gb_plus_selects_sq20(self):
        for pool in (32, 36, 48, 64, 128):
            rec = inkling_fit.recommend(vram_gib=None, ram_gib=pool, unified=True)
            self.assertEqual(rec["build"], "SQ-20", f"pool={pool} GiB UMA should pick SQ-20")
            self.assertTrue(rec["fits"])

    def test_discrete_16gb_and_24gb_mac_select_sq15(self):
        # #6613: 16 GiB discrete GPU at 32K plans SQ-15 as safe hybrid (never OOMing on KV growth),
        # while speed preference selects 100%-VRAM SQ-10 and 24 GiB Mac UMA fits SQ-15 100% in memory.
        rec_16 = inkling_fit.recommend(vram_gib=16, ram_gib=32)
        self.assertEqual(rec_16["build"], "SQ-15")
        self.assertEqual(rec_16["mode"], "ram_vram_zero_ssd")
        self.assertTrue(rec_16["fits"])
        self.assertLessEqual(rec_16["placement"]["vram_gb"], 10.6)

        rec_16_speed = inkling_fit.recommend(vram_gib=16, ram_gib=32, preference="speed")
        self.assertEqual(rec_16_speed["build"], "SQ-10")
        self.assertEqual(rec_16_speed["mode"], "full_vram")
        self.assertTrue(rec_16_speed["fits"])

        rec_mac24 = inkling_fit.recommend(vram_gib=None, ram_gib=24, unified=True)
        self.assertEqual(rec_mac24["build"], "SQ-15")
        self.assertEqual(rec_mac24["mode"], "full_vram")
        self.assertTrue(rec_mac24["fits"])

    def test_discrete_12gb_quality_vs_speed(self):
        rec_q = inkling_fit.recommend(vram_gib=12, ram_gib=32, preference="quality")
        self.assertEqual(rec_q["build"], "SQ-15")
        self.assertEqual(rec_q["mode"], "ram_vram_zero_ssd")
        self.assertTrue(rec_q["fits"])

        rec_s = inkling_fit.recommend(vram_gib=12, ram_gib=32, preference="speed")
        self.assertEqual(rec_s["build"], "SQ-10")
        self.assertEqual(rec_s["mode"], "full_vram")
        self.assertTrue(rec_s["fits"])

        # 12 GiB GPU with only 16 GiB RAM drops to SQ-10 automatically
        rec_low_ram = inkling_fit.recommend(vram_gib=12, ram_gib=16, preference="quality")
        self.assertEqual(rec_low_ram["build"], "SQ-10")

    def test_8gb_gpu_16gb_mac_and_amd_igpus_select_sq10(self):
        rec_8 = inkling_fit.recommend(vram_gib=8, ram_gib=32)
        self.assertEqual(rec_8["build"], "SQ-10")
        self.assertTrue(rec_8["fits"])

        rec_mac16 = inkling_fit.recommend(vram_gib=None, ram_gib=16, unified=True)
        self.assertEqual(rec_mac16["build"], "SQ-10")
        self.assertTrue(rec_mac16["fits"])

        # AMD Radeon 780M (gfx1103) with small 2 GiB BIOS carve-out + 32 GiB RAM -> SQ-10 via coherent UMA
        rec_780m = inkling_fit.recommend(vram_gib=2, ram_gib=32, arch="gfx1103")
        self.assertEqual(rec_780m["build"], "SQ-10")
        self.assertTrue(rec_780m["unified"])
        self.assertTrue(rec_780m["fits"])

        # AMD Strix Point 880M/890M (gfx1150) -> SQ-10 via coherent UMA
        rec_strix = inkling_fit.recommend(vram_gib=4, ram_gib=32, arch="gfx1150")
        self.assertEqual(rec_strix["build"], "SQ-10")
        self.assertTrue(rec_strix["unified"])
        self.assertTrue(rec_strix["fits"])

    def test_4tier_host_kv_offload_and_low_vram_contract(self):
        # Lossless FP16 KV stays default on 8/12/16/24 GiB GPUs via Host KV Offload (#552)!
        p8 = inkling_fit.compute("SQ-10", 8, 32, 32768)
        self.assertEqual(p8["kv_dtype"], "fp16")
        self.assertEqual(p8["kv_offload"], "on")
        self.assertGreater(p8["host_kv_gb"], 0.9)
        self.assertGreater(p8["vram_gb"], 3.5)
        self.assertEqual(p8["ssd_gb"], 0.0)

        # Pastebin CA1JEeAx: RTX 5050 Laptop 8 GiB + SQ-15 + 64K + Vision + DSpark must NEVER drop below 0 VRAM or OOM!
        p8_vis = inkling_fit.compute("SQ-15", 8, 32, 65536, vision=True, dspark=True)
        self.assertGreater(p8_vis["vram_gb"], 3.5)
        self.assertEqual(p8_vis["kv_offload"], "on")
        self.assertEqual(p8_vis["ssd_gb"], 0.0)
        # Vision stays on CPU/host on <= 12.5 GiB GPUs (0 permanent VRAM stolen)
        p8_novis = inkling_fit.compute("SQ-15", 8, 32, 65536, vision=False, dspark=True)
        self.assertEqual(p8_vis["vram_gb"], p8_novis["vram_gb"])

        # 12 GiB + SQ-15 + 32K keeps > 7.5 GB hot 3x-looped experts in VRAM with lossless FP16 Host KV Offload
        p12 = inkling_fit.compute("SQ-15", 12, 32, 32768)
        self.assertEqual(p12["kv_dtype"], "fp16")
        self.assertGreater(p12["vram_gb"], 7.5)
        self.assertEqual(p12["ssd_gb"], 0.0)

        # #6613: 16 GiB (and 15.92 GiB Radeon RX 9060 XT / RX 9070 XT) + SQ-15 + 32K must NOT plan full_vram=True
        # (12.28 GB experts leaves <100 MiB for HIP kv grow_pool); it keeps safe VRAM headroom (~10.44-10.53 GB experts)
        # while SQ-10 fits 100% in VRAM at 32K and SQ-15 fits 100% in VRAM at 8K.
        for v_16, bkd in ((15.92, "hip"), (15.92, None), (16.0, "cuda")):
            p16 = inkling_fit.compute("SQ-15", v_16, 64, 32768, backend=bkd)
            self.assertEqual(p16["kv_dtype"], "fp16")
            self.assertFalse(p16["full_vram"], (v_16, bkd, p16))
            self.assertLessEqual(p16["vram_gb"], 10.6, (v_16, bkd, p16))
            self.assertGreater(p16["ram_gb"], 1.5, (v_16, bkd, p16))
            self.assertEqual(p16["ssd_gb"], 0.0)
            self.assertEqual(p16["host_kv_gb"], 0.0)

        p16_radeon_vis = inkling_fit.compute("SQ-15", 15.92, 64, 32768, vision=True, backend="hip")
        self.assertFalse(p16_radeon_vis["full_vram"])
        self.assertLessEqual(p16_radeon_vis["vram_gb"], 10.0)
        self.assertEqual(p16_radeon_vis["ssd_gb"], 0.0)

        p16_sq10 = inkling_fit.compute("SQ-10", 15.92, 64, 32768, vision=True, backend="hip")
        self.assertTrue(p16_sq10["full_vram"])
        self.assertEqual(p16_sq10["ram_gb"], 0.0)

        p16_sq15_8k = inkling_fit.compute("SQ-15", 16.0, 32, 8192)
        self.assertTrue(p16_sq15_8k["full_vram"])

        # 4 GiB & 6 GiB GPUs never produce negative VRAM; low-RAM 4 GiB auto-activates SSD tier
        p4 = inkling_fit.compute("SQ-10", 4, 8, 32768)
        self.assertGreaterEqual(p4["vram_gb"], 0.0)
        self.assertGreater(p4["ssd_gb"], 0.0)
        ssd_f = inkling_fit.ssd_mode_flags("SQ-10", 4, 8, ctx=32768)
        self.assertIn("--ssd-experts", ssd_f["flags"])

        # Multi-GPU layer-split auto-activates for 2..4 GPUs so KV + experts span combined VRAM (#6592, #6631)
        f_2x16 = inkling_fit.multi_gpu_flags([{"vram_gb": 16, "arch": "sm_120"}] * 2, windows=False)
        self.assertEqual(f_2x16, ["--gpus", "2", "--layer-split", "auto"])
        f_3p100 = inkling_fit.multi_gpu_flags([{"vram_gb": 16, "arch": "sm_60"}] * 3, windows=True)
        self.assertEqual(f_3p100, ["--gpus", "3", "--layer-split", "auto"])
        f_mixed = inkling_fit.multi_gpu_flags([{"vram_gb": 24, "arch": "sm_86"}, {"vram_gb": 12, "arch": "sm_89"}])
        self.assertEqual(f_mixed, ["--gpus", "2", "--layer-split", "auto"])
        # 2x 16 GiB GPUs (e.g. RTX 5080 + RTX 5070 Ti) fit SQ-15, SQ-20, UD-Q4_K_XL, UD-Q8_K_XL 100% in VRAM
        for build in ("SQ-15", "SQ-20", "UD-Q4_K_XL", "UD-Q8_K_XL"):
            p_dual = inkling_fit.place(build, 16.0, 32.0, ctx=32768, extra_vram_gib=16.0)
            self.assertTrue(p_dual["full_vram"], (build, p_dual))
            self.assertEqual(p_dual["ram_gb"], 0.0)
            self.assertEqual(p_dual["ssd_gb"], 0.0)

    def test_all_10_unsloth_gguf_and_strata_builds_across_all_vram_and_uma_tiers(self):
        unsloth_quants = (
            "UD-Q8_K_XL", "UD-Q4_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL",
            "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S",
        )
        self.assertEqual(tuple(inkling_fit.GGUF_ORDER), unsloth_quants)
        all_builds = (*inkling_fit.STRATA_ORDER, *unsloth_quants)

        # Discrete tiers: 4, 6, 8, 10, 12, 16, 24, 32 GiB
        for vram, ram in ((4, 16), (6, 16), (8, 32), (10, 32), (12, 32), (16, 64), (24, 64), (32, 128)):
            for b in all_builds:
                p = inkling_fit.compute(b, vram, ram, 32768)
                self.assertGreaterEqual(p["vram_gb"], 0.0, (vram, b, p))
                self.assertGreaterEqual(p["ram_gb"], 0.0, (vram, b, p))
                self.assertGreaterEqual(p["ssd_gb"], 0.0, (vram, b, p))
                total_exp = round(p["vram_gb"] + p["ram_gb"] + p["ssd_gb"], 2)
                self.assertAlmostEqual(total_exp, inkling_fit.BUILDS[b]["bodies_gb"], delta=0.03, msg=(vram, b, p))
                if vram == 32:
                    self.assertTrue(p["full_vram"], (vram, b, p))
                if vram == 16 and b in ("SQ-15", "UD-Q8_K_XL", "UD-Q4_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL"):
                    self.assertFalse(p["full_vram"], (vram, b, p))
                    self.assertEqual(p["ssd_gb"], 0.0, (vram, b, p))
                if vram == 16 and b in ("SQ-10", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S"):
                    self.assertTrue(p["full_vram"], (vram, b, p))
                if vram == 24 and b == "UD-Q8_K_XL":
                    self.assertFalse(p["full_vram"], (vram, b, p))
                    self.assertEqual(p["ssd_gb"], 0.0, (vram, b, p))
                if vram == 24 and b == "UD-Q4_K_XL":
                    self.assertTrue(p["full_vram"], (vram, b, p))

        # Unified tiers: Mac Metal (16/24/36/48/64/128), Strix Halo (96/128), DGX Spark (128)
        for pool in (16, 24, 36, 48, 64, 96, 128):
            for b in all_builds:
                p = inkling_fit.compute_unified(b, pool, 32768)
                self.assertGreaterEqual(p["vram_gb"], 0.0, (pool, b, p))
                self.assertEqual(p["ram_gb"], 0.0)
                if pool >= 36:
                    self.assertTrue(p["full_vram"], (pool, b, p))
                elif pool == 24 and b in ("SQ-15", "SQ-10", "UD-Q4_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S"):
                    self.assertTrue(p["full_vram"], (pool, b, p))
                elif pool == 16 and b in ("SQ-10", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S"):
                    self.assertTrue(p["full_vram"], (pool, b, p))

    def test_kv_cache_scales_for_parallel_1_to_64(self):
        k1 = inkling_fit.kv_cache_gb(32768, parallel=1)
        k8 = inkling_fit.kv_cache_gb(32768, parallel=8)
        k64 = inkling_fit.kv_cache_gb(32768, parallel=64)
        self.assertGreater(k8, k1)
        self.assertGreater(k64, k1)
        # 64 slots with FP8 KV + realistic active occupancy remains well inside 24 GiB headroom
        self.assertLess(k64, 12.0)

    def test_shape_engine_and_family_wiring(self):
        self.assertEqual(shape_engine.INKLING_SHAPE, "inkling2")
        self.assertIn("inkling2", shape_engine.CUSTOM_SHAPES)
        self.assertEqual(shape_engine.component_id("inkling2"), "strata-inkling")
        self.assertEqual(shape_engine.shape_of_component("strata-inkling"), "inkling2")
        self.assertEqual(shape_engine.exe_name("inkling2"), "strata-inkling.exe" if os.name == "nt" else "strata-inkling")
        self.assertEqual(shape_engine.exe_name("inkling2", target="linux-x86_64"), "strata-inkling")
        self.assertEqual(shape_engine.exe_name("inkling2", target="linux-aarch64"), "strata-inkling")
        self.assertEqual(shape_engine.exe_name("inkling2", target="macos-arm64"), "strata-inkling")
        self.assertEqual(shape_engine.exe_name("inkling2", target="windows-x86_64"), "strata-inkling.exe")
        import json
        from strata_cli.launch import harness
        t = harness.Target("http://127.0.0.1:8080", "strata-local", "inkling-2-mini-sq-20", 32768)
        oc_ink = json.loads(harness.opencode_file(t))
        self.assertFalse(oc_ink["providers"]["strata"]["models"]["inkling-2-mini-sq-20"]["capabilities"]["reasoning"]["supported"])
        self.assertIn("reasoning: false", harness.omp_models(t))
        self.assertIn("defaultThinkingLevel: off", harness.omp_config(t))
        self.assertIn('"defaultThinkingLevel": "off"', harness.pi_settings(t))
        self.assertEqual(inkling_fit.build_of_model_name("inkling-2-mini-sq-20"), "SQ-20")
        self.assertEqual(inkling_fit.build_of_model_name("inkling-2-mini-sq-15"), "SQ-15")
        self.assertEqual(inkling_fit.build_of_model_name("inkling-2-mini-sq-10"), "SQ-10")
        self.assertEqual(pictures.family_of({"arch": "inkling2_loop", "model_name": "inkling-2-mini-sq-20"}, "Inkling-2-Mini-SQ-20.strata"), "inkling-2-mini")

    def test_sources_json_hashes_and_sizes(self):
        srcs = inkling_family.sources()
        self.assertGreaterEqual(len(srcs), 1)
        self.assertIn("strata-engine/Inkling-2-Mini-Strata", srcs[0]["base"])
        for key in ("SQ-20", "SQ-15", "SQ-10"):
            fname = inkling_family.build_file(key)
            size, sha = inkling_family.FILES[fname]
            self.assertEqual(len(sha), 64)
            self.assertGreater(size, 10_000_000_000)
        vsize, vsha = inkling_family.FILES[inkling_family.VISION_FILE]
        self.assertEqual(len(vsha), 64)
        self.assertEqual(vsize, 636526592)

    def test_hf_token_auth_headers_only_sent_to_huggingface(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_secret_token"}, clear=False):
            h_hf = transfer.hf_auth_headers("https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main/SHA256SUMS.txt")
            self.assertEqual(h_hf, {"Authorization": "Bearer hf_test_secret_token"})
            h_other = transfer.hf_auth_headers("https://cdn-lfs.s3.amazonaws.com/bucket/key")
            self.assertEqual(h_other, {})
        with mock.patch.dict(os.environ, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""}, clear=False):
            h_pub = transfer.hf_auth_headers("https://huggingface.co/strata-engine/Hy4-Flash-Strata/resolve/main/README.md")
            self.assertEqual(h_pub, {})

    def test_dgx_spark_gb10_linux_aarch64_auto_selects_sq20(self):
        from strata_control import targets
        import setup as setup_mod
        from tools import strata_mcp

        ok, why = targets.backend_supported("cuda", "linux-aarch64")
        self.assertTrue(ok, why)
        self.assertTrue(inkling_fit.is_big_uma_arch("121", "NVIDIA GB10"))
        self.assertTrue(inkling_fit.is_big_uma_arch("sm_121", "NVIDIA GB10"))
        self.assertTrue(inkling_fit.is_big_uma_arch("gfx1151", "AMD Radeon Graphics"))
        self.assertFalse(inkling_fit.is_big_uma_arch("gfx1103", "Radeon 780M"))

        # Even if hardware probe tags GB10 with integrated=True, unified=True, or vram_gb=0, it MUST pick SQ-20 (never SQ-10)
        for gpu in (
            {"name": "NVIDIA GB10", "vendor": "nvidia", "arch": "121", "vram_gb": 120.0, "uma": True},
            {"name": "NVIDIA GB10", "vendor": "nvidia", "arch": "sm_121", "vram_gb": 0.0, "unified": True, "integrated": True},
            {"name": "NVIDIA GB10", "vendor": "nvidia", "arch": "121", "vram_gb": None, "integrated": True},
        ):
            vram_gib, ram_gib, unified, igpu, arch = setup_mod.inkling_pc(gpu, 128.0)
            self.assertTrue(unified)
            self.assertFalse(igpu)
            rec = inkling_fit.recommend(vram_gib, ram_gib, unified, igpu=igpu, arch=arch)
            self.assertEqual(rec["build"], "SQ-20")
            self.assertTrue(rec["fits"])
            self.assertEqual(rec["mode"], "full_vram")
            row = strata_mcp.Strata.inkling_fit_row(None, "SQ-20", 128.0, [gpu])
            self.assertIsNotNone(row)
            self.assertTrue(row["recommended"])
            self.assertTrue(row["unified"])
            self.assertEqual(row["mode"], "full_vram")


if __name__ == "__main__":
    unittest.main()
