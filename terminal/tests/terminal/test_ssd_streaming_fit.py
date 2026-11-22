"""Tests for first-class GPU + SSD Only and CPU + SSD Only fitting & CLI flag planning across all engines."""

from __future__ import annotations

import unittest

from strata_control import gpu_fit, hy4_fit, inkling_fit


class SsdStreamingFitTests(unittest.TestCase):
    def test_gpu_ssd_only_big_moe_on_small_ram_host(self):
        # 12 GB RTX 5070 + 12 GB RAM running a 78 GB MoE -> GPU + SSD Only (bounded ~1-2 GB DMA ring, 0 swap!)
        p = gpu_fit.ssd_streaming_plan(78.0, ram_gb=12.0, vram_gb=12.0)
        self.assertEqual(p["mode"], "gpu_ssd_only")
        self.assertTrue(p["runnable"])
        self.assertLessEqual(p["bounce_ring_gb"], 2.0)
        self.assertGreaterEqual(p["bounce_ring_gb"], 1.0)
        self.assertIn("--ssd-experts", p["flags"])
        self.assertIn("0", p["flags"])
        self.assertGreater(p["ssd_gb"], 50.0)

    def test_cpu_ssd_only_big_moe_with_no_gpu(self):
        # No GPU + 16 GB RAM running a 78 GB MoE -> CPU + SSD Only (bounded RAM LRU, never blocks!)
        p = gpu_fit.ssd_streaming_plan(78.0, ram_gb=16.0, vram_gb=0.0)
        self.assertEqual(p["mode"], "cpu_ssd_only")
        self.assertTrue(p["runnable"])
        self.assertIn("--cpu", p["flags"])
        self.assertIn("--ssd-experts", p["flags"])
        self.assertLessEqual(p["ram_use_gb"], 8.0)
        self.assertGreater(p["ssd_gb"], 60.0)

    def test_hy4_gpu_ssd_only_and_cpu_ssd_only(self):
        # 12 GiB GPU + 12 GiB RAM running 72 GB Hy4-Flash SQ-64 -> gpu_ssd_only
        g = hy4_fit.ssd_mode_flags("SQ-64", vram_gib=12.0, ram_gib=12.0)
        self.assertEqual(g["tier"], "gpu_ssd_only")
        self.assertTrue(g["runnable"])
        self.assertIn("--ssd-experts", g["flags"])

        # No GPU + 16 GiB RAM -> cpu_ssd_only
        c = hy4_fit.ssd_mode_flags("SQ-64", vram_gib=0.0, ram_gib=16.0)
        self.assertEqual(c["tier"], "cpu_ssd_only")
        self.assertTrue(c["runnable"])
        self.assertIn("--ssd-experts", c["flags"])

    def test_inkling_gpu_ssd_only_and_cpu_ssd_only(self):
        # 8 GiB GPU + 10 GiB RAM running 20 GB Inkling-2-Mini SQ-20 -> gpu_ssd_only
        g = inkling_fit.ssd_mode_flags("SQ-20", vram_gib=8.0, ram_gib=10.0)
        self.assertEqual(g["tier"], "gpu_ssd_only")
        self.assertTrue(g["runnable"])
        self.assertIn("--ssd-experts", g["flags"])

        # No GPU + 8 GiB RAM running 20 GB SQ-20 -> cpu_ssd_only
        c = inkling_fit.ssd_mode_flags("SQ-20", vram_gib=0.0, ram_gib=8.0)
        self.assertEqual(c["tier"], "cpu_ssd_only")
        self.assertTrue(c["runnable"])
        self.assertIn("--cpu", c["flags"])
        self.assertIn("--ssd-experts", c["flags"])

        # 24 GiB GPU + 32 GiB RAM running SQ-20 -> in_memory
        ok = inkling_fit.ssd_mode_flags("SQ-20", vram_gib=24.0, ram_gib=32.0)
        self.assertEqual(ok["tier"], "in_memory")
        self.assertEqual(ok["flags"], [])


if __name__ == "__main__":
    unittest.main()
