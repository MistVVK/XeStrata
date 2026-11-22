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
        rec_16 = inkling_fit.recommend(vram_gib=16, ram_gib=32)
        self.assertEqual(rec_16["build"], "SQ-15")
        self.assertEqual(rec_16["mode"], "full_vram")
        self.assertTrue(rec_16["fits"])

        rec_mac24 = inkling_fit.recommend(vram_gib=None, ram_gib=24, unified=True)
        self.assertEqual(rec_mac24["build"], "SQ-15")
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
        from strata_cli.launch import harness
        t = harness.Target("http://127.0.0.1:8080", "strata-local", "inkling-2-mini-sq-20", 32768)
        self.assertIn('"reasoning": false', harness.opencode_file(t))
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
