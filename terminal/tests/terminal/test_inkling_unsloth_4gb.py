"""Tests for Track 7A:
- Direct download & load of unsloth/Inkling-2-Mini-GGUF quants in GUI & TUI catalog + setup
- Unlocking 4 GB & 6 GB GPUs across all MoE families (Inkling-2-Mini, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Hy4-Flash)
- Realistic safe --vram-experts-gb sizing in inkling_fit.py (Issue #6587)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import gpu_fit, gpu_roles, hy4_fit, inkling_family, inkling_fit, models, own_path_fit


EXPECTED_UNSLOTH_QUANTS = (
    "UD-IQ1_S",
    "UD-IQ1_M",
    "UD-IQ2_XXS",
    "UD-Q2_K_XL",
    "UD-IQ3_XXS",
    "UD-Q3_K_XL",
    "UD-IQ4_XS",
    "UD-Q4_K_M",
    "UD-Q4_K_XL",
    "UD-Q8_K_XL",
)


class TestInklingUnslothDirectCatalog(unittest.TestCase):
    def test_all_unsloth_quants_and_companions_pinned_with_sha256(self):
        for q in EXPECTED_UNSLOTH_QUANTS:
            self.assertIn(q, inkling_fit.BUILDS)
            self.assertTrue(inkling_fit.is_gguf_build(q))
            fname = inkling_family.build_file(q)
            self.assertEqual(fname, f"Inkling-2-Mini-{q}.gguf")
            size, sha = inkling_family.FILES[fname]
            self.assertGreater(size, 7_000_000_000)
            self.assertEqual(len(sha), 64)
            self.assertIn(q, inkling_family.BUILD_ABOUT)

        for aux in (inkling_family.DSPARK_GGUF_FILE, inkling_family.MMPROJ_BF16_FILE, inkling_family.MMPROJ_F16_FILE):
            size, sha = inkling_family.FILES[aux]
            self.assertGreater(size, 300_000_000)
            self.assertEqual(len(sha), 64)

        # Primary .strata builds remain first in ORDER
        self.assertEqual(inkling_fit.ORDER[:3], ("SQ-20", "SQ-15", "SQ-10"))

    def test_sources_for_routes_gguf_to_unsloth_and_strata_to_strata_engine(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_INKLING_SOURCE", None)
            os.environ.pop("STRATA_INKLING_GGUF_SOURCE", None)
            s_strata = inkling_family.sources_for("Inkling-2-Mini-SQ-10.strata")
            s_gguf = inkling_family.sources_for("Inkling-2-Mini-UD-Q2_K_XL.gguf")
            s_dspark = inkling_family.sources_for(inkling_family.DSPARK_GGUF_FILE)
            self.assertIn("strata-engine/Inkling-2-Mini-Strata", s_strata[0]["base"])
            self.assertIn("unsloth/Inkling-2-Mini-GGUF", s_gguf[0]["base"])
            self.assertIn("unsloth/Inkling-2-Mini-GGUF", s_dspark[0]["base"])

        # STRATA_INKLING_SOURCE overrides both for local mirrors / offline testing
        with mock.patch.dict(os.environ, {"STRATA_INKLING_SOURCE": "http://127.0.0.1:19999/mirror"}, clear=False):
            self.assertEqual(inkling_family.sources_for("Inkling-2-Mini-UD-IQ1_S.gguf")[0]["base"], "http://127.0.0.1:19999/mirror")
            self.assertEqual(inkling_family.sources_for("Inkling-2-Mini-SQ-10.strata")[0]["base"], "http://127.0.0.1:19999/mirror")

    def test_files_of_gguf_includes_dspark_and_vision(self):
        files = inkling_family.files_of("UD-Q2_K_XL", vision=True, dspark=True)
        self.assertEqual(files, [
            "Inkling-2-Mini-UD-Q2_K_XL.gguf",
            inkling_family.DSPARK_GGUF_FILE,
            inkling_family.VISION_FILE,
        ])
        files_novis = inkling_family.files_of("UD-IQ1_S", vision=False, dspark=True)
        self.assertEqual(files_novis, ["Inkling-2-Mini-UD-IQ1_S.gguf", inkling_family.DSPARK_GGUF_FILE])
        # .strata builds already embed DSpark
        self.assertEqual(inkling_family.files_of("SQ-15", vision=True), ["Inkling-2-Mini-SQ-15.strata", inkling_family.VISION_FILE])

    def test_mcp_catalog_and_tui_rows_expose_strata_and_unsloth_gguf(self):
        sys.path.insert(0, str(TERMINAL / "tools"))
        import strata_mcp
        from strata_cli.launch.catalog import Catalog

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hw = {"ram_gb": 16.0, "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3050 4GB", "vram_gb": 3.9, "usable": True, "vendor": "nvidia"}]}
            s = strata_mcp.Strata(root, TERMINAL)
            with mock.patch.object(s, "inkling_visible", return_value=True):
                table = s.model_table(hw)
            ink = next((f for f in table if f["family"] == "inkling-2-mini"), None)
            self.assertIsNotNone(ink)
            names = [row["model"] for row in ink["sizes"]]
            self.assertEqual(names[:3], ["SQ-20", "SQ-15", "SQ-10"])
            for q in EXPECTED_UNSLOTH_QUANTS:
                self.assertIn(q, names)
            ud_row = next(r for r in ink["sizes"] if r["model"] == "UD-Q2_K_XL")
            self.assertEqual(ud_row["id"], "inkling-2-mini-ud-q2_k_xl")
            self.assertEqual(ud_row["format"], "gguf")
            self.assertEqual(ud_row["on_this_pc"], "fits")
            self.assertEqual(ud_row["inkling"]["placement"]["vram_gb"], 0.0)

            # TUI Catalog.rows() surfaces them too
            cat = Catalog(src=TERMINAL, root=root, hardware=hw)
            with mock.patch.object(cat.strata, "inkling_visible", return_value=True):
                rows = cat.rows(hw)
            ink_rows = [r for r in rows if r.family == "inkling-2-mini"]
            self.assertGreaterEqual(len(ink_rows), 13)
            tui_ud = next(r for r in ink_rows if r.model == "UD-IQ1_S")
            self.assertEqual(tui_ud.verdict, "fits")

    def test_setup_direct_download_unsloth_gguf_on_4gb_gpu(self):
        import setup
        from tests.terminal.test_inkling_gguf_import import make_inkling_gguf, write_fake_o200k_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            install_dir = root / "install"
            eng_dir = root / "eng"
            data_dir.mkdir()
            install_dir.mkdir()
            eng_dir.mkdir()
            (eng_dir / inkling_family.EXE).write_bytes(b"exe")

            args_ns = argparse.Namespace(
                model="UD-Q2_K_XL",
                family="inkling-2-mini",
                context=32768,
                vision="yes",
                yes=True,
                no_start=True,
                gguf_dir=None,
                models_dir=str(data_dir / "models"),
                data_dir=str(data_dir),
                backend=None,
                gpu=None,
                host=None,
                api_key=None,
                browser=None,
                port=8080,
            )
            fetched: list[str] = []

            def fake_fetch(_a, name, folder, _own):
                fetched.append(name)
                folder.mkdir(parents=True, exist_ok=True)
                p = folder / name
                if name.endswith(".gguf"):
                    p.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
                else:
                    p.write_bytes(b"STRATA\x00\x01" + b"\x00" * 64)
                write_fake_o200k_files(folder)
                return p

            gpu_4gb = {"index": 0, "name": "NVIDIA GeForce GTX 1650", "vendor": "nvidia", "vram_gb": 3.9, "count": 1}
            with mock.patch.object(setup, "home", return_value=install_dir), \
                 mock.patch.object(setup, "data_folder_for", return_value=(data_dir, "explicit")), \
                 mock.patch.object(setup, "pip_install"), \
                 mock.patch.object(setup, "free_gb", return_value=500.0), \
                 mock.patch.object(setup, "inkling_fetch", side_effect=fake_fetch), \
                 mock.patch.object(setup, "engine_step", return_value=(None, eng_dir, {"component": "strata-inkling"}, [], gpu_4gb, "none")), \
                 mock.patch.object(setup, "engine_exe", return_value="component:strata-inkling"), \
                 mock.patch.object(setup, "write_run_script", return_value=install_dir / "run.bat"):
                rc = setup.install_inkling2(args_ns, "inkling-2-mini", gpu_4gb, False, False, None, False, 8080, 16.0, None)

            self.assertEqual(rc, 0)
            self.assertEqual(fetched, [
                "Inkling-2-Mini-UD-Q2_K_XL.gguf",
                inkling_family.DSPARK_GGUF_FILE,
                inkling_family.VISION_FILE,
            ])
            cfg_path = install_dir / "strata-inkling-2-mini-ud-q2_k_xl.json"
            self.assertTrue(cfg_path.is_file())
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["drafting"], "dspark")
            self.assertIn("--dspark", cfg["args"])
            self.assertIn("--vision", cfg["args"])
            # 4 GB GPU gets --vram-experts-gb 0.00 so trunk sits in VRAM and experts stream from RAM
            vi = cfg["args"].index("--vram-experts-gb")
            self.assertEqual(cfg["args"][vi + 1], "0.00")


class TestUnlock4GbAndIssue6587(unittest.TestCase):
    def test_issue_6587_rtx3060_12gb_sq15_128k_vision_safe_vram_experts(self):
        placed = inkling_fit.compute("SQ-15", vram_gib=12.0, ram_gib=32.0, ctx=131072, vision=True, parallel=1, dspark=True)
        # With Host KV Offload + SWA-1025 DSpark + CPU Vision on <= 12.5 GiB, 128K Global KV spills to mapped host RAM
        # while >= 2.5 GB VRAM headroom remains alongside the 2.07 GB dense trunk for runner + DSpark + desktop (#6587, #552).
        vram_total_gb = 12.0 * inkling_fit.GIB_TO_GB
        dense_gb = inkling_fit.BUILDS["SQ-15"]["dense_gb"]
        self.assertGreaterEqual(vram_total_gb - dense_gb - placed["vram_gb"], 2.5, placed)
        self.assertEqual(placed["kv_offload"], "on")
        self.assertGreater(placed["host_kv_gb"], 4.0)
        self.assertEqual(placed["ssd_gb"], 0.0)
        self.assertFalse(placed["full_vram"])
        # When Host KV Offload is explicitly disabled, KV steps down to FP8 while keeping >= 4.5 GB VRAM headroom.
        placed_vram_fp8 = inkling_fit.compute(
            "SQ-15", vram_gib=12.0, ram_gib=32.0, ctx=131072, vision=True, parallel=1, dspark=True, kv_offload="off"
        )
        self.assertEqual(placed_vram_fp8["kv_dtype"], "fp8")
        self.assertGreaterEqual(vram_total_gb - dense_gb - placed_vram_fp8["vram_gb"], 4.5, placed_vram_fp8)

    def test_4gb_and_6gb_gpus_unlocked_across_all_moe_families(self):
        # 1. gpu_fit & gpu_roles admit 4 GB discrete GPUs reporting 3.8-4.0 GiB (GTX 1650, RTX 3050 4 GB, T500, Arc A310)
        for vram in (3.8, 3.9, 4.0, 6.0):
            c = {"index": 0, "name": "NVIDIA GeForce GTX 1650", "vram_gb": vram, "usable": True, "vendor": "nvidia"}
            keep, ignored = gpu_fit.usable_cards([c])
            self.assertEqual(len(keep), 1, f"vram={vram} was ignored: {ignored}")
            rec = gpu_roles.recommend([c])
            self.assertEqual(rec["roles"], {0: "main"})
            self.assertEqual(rec["main"], 0)
            opts = gpu_fit.options(44.0, 64.0, [c])
            self.assertEqual(opts["recommended"], "single")

        # 2. Inkling-2-Mini on 4 GB + 16 GB RAM: --vram-experts-gb 0.00, 100% experts in Host RAM, zero SSD spill
        for build in ("SQ-10", "UD-IQ1_S", "UD-Q2_K_XL"):
            p4 = inkling_fit.compute(build, vram_gib=3.9, ram_gib=16.0, ctx=32768, vision=True)
            self.assertEqual(p4["vram_gb"], 0.0, (build, p4))
            self.assertGreater(p4["ram_gb"], 0.0, (build, p4))
            self.assertEqual(p4["ssd_gb"], 0.0, (build, p4))
            self.assertEqual(inkling_fit.mode(p4), "ram_vram_zero_ssd")

        # 3. Inkling-2-Mini on 6 GB + 16 GB RAM: puts trunk + partial experts in VRAM, rest in RAM
        p6 = inkling_fit.compute("SQ-10", vram_gib=6.0, ram_gib=16.0, ctx=32768, vision=True)
        self.assertGreater(p6["vram_gb"], 0.0)
        self.assertEqual(p6["ssd_gb"], 0.0)

        # 4. Hy4-Flash on 4 GB + 64 GB RAM: clamps vram_gb to 0.0, never blocks
        h4 = hy4_fit.place("SQ-64", vram_gib=4.0, ram_gib=64.0)
        self.assertEqual(h4["vram_gb"], 0.0)
        self.assertGreater(h4["ram_gb"], 40.0)

        # 5. Qwen3.6-35B-A3B (own_path_fit) on 4 GB GPU: recommends smallest context, never refuses
        fit36 = own_path_fit.inputs("qwen3.6-35b-a3b", "UD-Q4_K_XL")
        self.assertEqual(own_path_fit.recommended(fit36, 4.0, own_path_fit.MENU), 8192)
        self.assertIsNone(own_path_fit.warning(fit36, 4.0, 8192, own_path_fit.MENU))

        # 6. Local import maps Unsloth GGUF filename to its exact catalog build
        self.assertEqual(models._inkling_build_for_bytes(9_138_715_328, "Inkling-2-Mini-UD-Q2_K_XL.gguf"), "UD-Q2_K_XL")
        self.assertEqual(models._inkling_build_for_bytes(7_764_291_776, "Inkling-2-Mini-UD-IQ1_S.gguf"), "UD-IQ1_S")


if __name__ == "__main__":
    unittest.main()
