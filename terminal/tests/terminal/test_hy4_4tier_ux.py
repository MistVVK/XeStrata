"""Regression tests for HY-4 4-tier UX, missing vocab.json recovery, #6607/#6592, #6589, #6588, #6606, and PR#6610."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TERM_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TERM_DIR.parent
for _p in (str(TERM_DIR), str(TERM_DIR / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _sample_hy4_hf_tokenizer() -> dict:
    return {
        "model": {
            "type": "BPE",
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {"hello": 0, "world": 1, " ": 2},
            "merges": ["h e"],
        },
        "added_tokens": [
            {"id": 3, "content": "<｜hy_begin▁of▁sentence｜>", "special": True},
            {"id": 4, "content": "<think:opensource>", "special": False},
        ],
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated"},
                {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated"},
                {
                    "type": "Split",
                    "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\s+"},
                    "behavior": "Isolated",
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
        "normalizer": None,
    }


class TestHy4FitHelpers(unittest.TestCase):
    def test_wddm_floor_and_elastic_kv_and_layer_split(self):
        from strata_control import hy4_fit

        self.assertGreaterEqual(hy4_fit.wddm_vram_floor_gb(12.0, windows=True), 1.0)
        self.assertGreaterEqual(hy4_fit.wddm_vram_floor_gb(8.0, windows=True), 0.6)
        self.assertGreaterEqual(hy4_fit.wddm_vram_floor_gb(4.0, windows=True), 0.45)

        self.assertEqual(hy4_fit.elastic_kv_init_slots(8.0, 131072), 4096)
        self.assertEqual(hy4_fit.elastic_kv_init_slots(12.0, 131072), 8192)
        self.assertEqual(hy4_fit.elastic_kv_init_slots(16.0, 131072), 16384)
        self.assertEqual(hy4_fit.elastic_kv_init_slots(64.0, 131072), 131072)

        self.assertTrue(hy4_fit.should_layer_split([{"vram_gb": 24}, {"vram_gb": 24}], windows=True))
        self.assertTrue(hy4_fit.should_layer_split([{"vram_gb": 24}, {"vram_gb": 12}], windows=False))
        self.assertFalse(hy4_fit.should_layer_split([{"vram_gb": 24, "nvlink": True}, {"vram_gb": 24, "nvlink": True}], windows=False))


class TestTokenizerRecovery(unittest.TestCase):
    def test_setup_hy4_tokenizer_dir_companion_and_hf_fallback(self):
        import setup as strata_setup

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "Hy4-Flash-SQ-64.strata"
            model.write_bytes(b"NOT_A_VALID_STRATA_FOR_ZSTD_FALLBACK")
            (root / "tokenizer.json").write_text(json.dumps(_sample_hy4_hf_tokenizer()), encoding="utf-8")
            (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            out = root / "packs" / "hy4-flash-sq-64" / "tokenizer"
            got = strata_setup.hy4_tokenizer_dir(model, out, None, folder=root)
            self.assertEqual(got, out)
            self.assertTrue((out / "vocab.json").is_file())
            self.assertTrue((out / "chat_template.jinja").is_file())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "Hy4-Flash-SQ-64.strata"
            model.write_bytes(b"NO_ZSTD")
            out = root / "packs" / "hy4-flash-sq-64" / "tokenizer"

            def fake_download(url, dst, label):
                if dst.name == "tokenizer.json":
                    dst.write_text(json.dumps(_sample_hy4_hf_tokenizer()), encoding="utf-8")
                elif dst.name == "chat_template.jinja":
                    dst.write_text("{{ messages }}", encoding="utf-8")

            with mock.patch.object(strata_setup, "download", side_effect=fake_download):
                got = strata_setup.hy4_tokenizer_dir(model, out, None, a=SimpleNamespace(), folder=root, own=False)
            self.assertEqual(got, out)
            self.assertTrue((out / "vocab.json").is_file())

    def test_server_autocorrect_and_ensure_shape_tokenizer(self):
        from serve import server

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "Hy4-Flash-SQ-64.strata"
            model.write_bytes(b"STUB")
            (root / "tokenizer.json").write_text(json.dumps(_sample_hy4_hf_tokenizer()), encoding="utf-8")
            (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            tpath = root / "packs" / "hy4-flash-sq-64" / "tokenizer"
            cfg = {
                "exe": "component:strata-engine",
                "arch": "qwen4exp",
                "cwd": str(root),
                "args": ["--native", str(model), "--engram", str(root / "Hy4-Flash-Engram-Q8.strata")],
            }
            server._autocorrect_shape_config(cfg)
            self.assertEqual(cfg["arch"], "hy4flash")
            self.assertEqual(cfg["exe"], "component:strata-hy4")
            ok = server._ensure_shape_tokenizer(cfg, tpath)
            self.assertTrue(ok)
            self.assertTrue((tpath / "vocab.json").is_file())


class TestMultiGpuIntNormalization6607(unittest.TestCase):
    def test_multi_gpu_cards_handles_int_indices_and_dicts(self):
        import setup as strata_setup

        fake_nv = [
            {"index": 0, "name": "RTX 3090", "vram_gb": 24.0, "arch": "sm_86"},
            {"index": 1, "name": "RTX 3090", "vram_gb": 24.0, "arch": "sm_86"},
        ]
        with mock.patch.object(strata_setup, "gpus", return_value=fake_nv):
            cards = strata_setup._multi_gpu_cards([0, 1], fake_nv[0], False, [])
        self.assertEqual(len(cards), 2)
        self.assertEqual([c["index"] for c in cards], [0, 1])
        self.assertEqual(sum(c["vram_gb"] for c in cards), 48.0)


class TestUpdaterSkipsShapeEngine6589(unittest.TestCase):
    def test_updater_detects_hy4_and_inkling_shape_configs(self):
        from strata_control import updater

        self.assertTrue(updater._is_shape_engine_config({"arch": "hy4flash", "args": ["--engram", "e.strata"]}))
        self.assertTrue(updater._is_shape_engine_config({"arch": "inkling2_loop", "args": ["--dspark", "d.gguf"]}))
        self.assertTrue(updater._is_shape_engine_config({"exe": "component:strata-hy4", "args": []}))
        self.assertTrue(updater._is_shape_engine_config({"exe": "C:/bin/strata-hy4.exe", "args": []}))
        self.assertFalse(updater._is_shape_engine_config({"arch": "qwen4exp", "exe": "component:strata-engine", "args": ["--model", "x.gguf"]}))

    def test_discover_run_config_skips_hy4_when_updating_strata_engine(self):
        from strata_control import models, profiles, retired, updater

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hy4_cfg = root / "strata-hy4-flash-sq-64.json"
            hy4_cfg.write_text(json.dumps({"arch": "hy4flash", "exe": "component:strata-hy4", "args": ["--engram", "e.strata"]}), encoding="utf-8")
            qwen_cfg = root / "strata-qwen.json"
            qwen_cfg.write_text(json.dumps({"arch": "qwen4exp", "exe": "component:strata-engine", "args": ["--model", "m.gguf"]}), encoding="utf-8")
            listed = {"models": [
                {"id": "a-hy4", "component_status": "present", "source_config": str(hy4_cfg)},
                {"id": "b-qwen", "component_status": "present", "source_config": str(qwen_cfg)},
            ]}
            with mock.patch.object(models, "list_installed", return_value=listed), \
                 mock.patch.object(profiles, "list_profiles", return_value=[{"source_config": str(hy4_cfg)}]), \
                 mock.patch.object(retired, "config_file_reason", return_value=None):
                picked = updater.discover_run_config()
            self.assertEqual(picked, qwen_cfg)


class TestBuildShapeEngine6588(unittest.TestCase):
    def test_shape_engine_step_builds_locally_when_build_flag_set(self):
        import setup as strata_setup

        with tempfile.TemporaryDirectory() as td:
            eng_dir = Path(td) / "engine-inkling2"
            eng_dir.mkdir(parents=True)
            exe_name = "strata-inkling.exe" if os.name == "nt" else "strata-inkling"
            (eng_dir / exe_name).write_bytes(b"ELF")
            (eng_dir / "BUILD.json").write_text(json.dumps({"version": "local", "backend": "cuda", "source": "local", "shape": "inkling2"}), encoding="utf-8")
            strata_setup.SHAPE_ENGINE.clear()
            strata_setup.SHAPE_ENGINE["shape"] = "inkling2"
            a = SimpleNamespace(build=True, yes=True)
            with mock.patch.object(strata_setup, "CONTROL", False), \
                 mock.patch.object(strata_setup, "build_shape_engine", return_value=eng_dir) as bmock, \
                 mock.patch.object(strata_setup, "cuda_lib_dirs", return_value=[]):
                _llama, eng, meta, _libs, _gpu, _vis = strata_setup.shape_engine_step(a, {"index": 0}, "none", hip=False)
                self.assertTrue(bmock.called)
                self.assertEqual(eng, eng_dir)
                self.assertEqual(meta["source"], "local")
                self.assertEqual(strata_setup.engine_exe(eng), str(eng_dir / exe_name))
            strata_setup.SHAPE_ENGINE.clear()


class TestGfx1150IgpuResolution6606(unittest.TestCase):
    def test_gfx_covers_normalizes_feature_suffix_and_selects_igpu_variant(self):
        from strata_control import feed

        self.assertTrue(feed.gfx_covers(("gfx1103", "gfx1150"), ["gfx1150:xnack-"]))
        self.assertTrue(feed.gfx_covers(("gfx1103:xnack+", "gfx1150"), ["gfx1103"]))
        pkg_igpu = feed.EnginePackage(
            version="0.1.66", platform="linux-x86_64", backend="hip", variant="igpu",
            archs=("gfx1103", "gfx1150"), min_driver={}, min_runtime=None, build_id="b1",
            url="https://example.invalid/igpu.tar.gz", format="tar.gz", size=100,
            sha256="a" * 64, manifest_sha256="b" * 64, installed_bytes=200,
        )
        pkg_halo = feed.EnginePackage(
            version="0.1.66", platform="linux-x86_64", backend="hip", variant="halo",
            archs=("gfx1151",), min_driver={}, min_runtime=None, build_id="b2",
            url="https://example.invalid/halo.tar.gz", format="tar.gz", size=100,
            sha256="c" * 64, manifest_sha256="d" * 64, installed_bytes=200,
        )
        chosen, _reasons = feed.select_engine([pkg_halo, pkg_igpu], {
            "platform": "linux-x86_64", "backend": "hip", "gfx": ["gfx1150:xnack-"],
            "cuda_archs": [], "nvidia_driver": None, "runtime": "0.2.20",
        })
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.variant, "igpu")


class TestServerIntelSamplingKeys6610(unittest.TestCase):
    def test_server_intel_sampling_keys_accepts_keywords_subprocess(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT / 'engine' / 'sycl' / 'serve')!r}); "
            "import server_intel; "
            "server_intel.install_logprobs(); "
            "out = server_intel.S.StrataEngine.sampling_keys({'temperature': 0.5, 'logprobs': True, 'top_logprobs': 3}, hy4=False, inkling=False); "
            "out_h = server_intel.S.StrataEngine.sampling_keys({'temperature': 0.5}, hy4=True); "
            "out_i = server_intel.S.StrataEngine.sampling_keys({'temperature': 0.5}, inkling=True); "
            "assert 'logprobs=3' in out and 'temperature=0.5' in out_h and 'temperature=0.5' in out_i, (out, out_h, out_i); "
            "print('OK', out)"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, msg=done.stderr + done.stdout)
        self.assertIn("OK", done.stdout)


if __name__ == "__main__":
    unittest.main()
