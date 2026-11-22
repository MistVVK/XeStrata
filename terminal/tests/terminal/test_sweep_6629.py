"""Regression tests for sweep #6614-#6629:
- #4891: doctor.py & models.py resolve pruned engine / shape-engine / control component paths for vision.exe, --expert-profile, and cwd.
- #6623: updater.py never tests `strata-engine` against a Hy4-Flash (`--engram`) or Inkling (`--dspark`) config and matches shape components cleanly.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_control import doctor, engine_component as ec, models, updater  # noqa: E402


class PrunedComponentPathTests(unittest.TestCase):
    """#4891: models with vision or expert-profile must stay `present` after an update prunes the old component folder."""

    def _make_install(self, root: Path) -> tuple[Path, Path, Path]:
        comp_root = root / "components"
        # Active engine folder has a different version AND variant suffix than the pruned legacy folder
        eng_dir = comp_root / "engine" / "0.1.65-windows-x86_64-cuda-cuda13"
        eng_dir.mkdir(parents=True)
        (eng_dir / "strata.exe").write_bytes(b"MZ" + b"\x00" * 64)
        (eng_dir / "strata-vision.exe").write_bytes(b"MZ" + b"\x00" * 64)
        (eng_dir / "BUILD.json").write_text(json.dumps({"version": "0.1.65"}), encoding="utf-8")
        (comp_root / "engine" / "active.json").write_text(
            json.dumps({"version": "0.1.65-windows-x86_64-cuda-cuda13"}), encoding="utf-8"
        )

        # Active control folder
        ctl_dir = comp_root / "control" / "0.2.20"
        (ctl_dir / "profiles").mkdir(parents=True)
        (ctl_dir / "profiles" / "qwen.profile").write_text("rank\n", encoding="utf-8")
        (comp_root / "control" / "active.json").write_text(json.dumps({"version": "0.2.20"}), encoding="utf-8")

        # Model files (tokenizer, pack, mmproj, model gguf)
        data_dir = root / "data"
        tok_dir = data_dir / "tokenizer"
        tok_dir.mkdir(parents=True)
        for fn in ("vocab.json", "merges.txt", "token_type.json"):
            (tok_dir / fn).write_text("ok", encoding="utf-8")
        pack_dir = data_dir / "pack"
        pack_dir.mkdir(parents=True)
        (pack_dir / "index.txt").write_text("ok", encoding="utf-8")
        (pack_dir / "native_experts.txt").write_text("ok", encoding="utf-8")
        mmproj = data_dir / "mmproj.gguf"
        mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
        model_gguf = data_dir / "model.gguf"
        model_gguf.write_bytes(b"GGUF" + b"\x00" * 16)

        # Legacy run config referencing PRUNED 0.1.52 cuda12 engine and PRUNED 0.1.56 control paths
        pruned_eng = comp_root / "engine" / "0.1.52-windows-x86_64-cuda-cuda12"
        pruned_ctl = comp_root / "control" / "0.1.56"
        cfg = {
            "model_name": "Qwen3.8-Flash-Next IQ2_XS",
            "backend": "cuda",
            "exe": str(pruned_eng / "strata.exe"),
            "cwd": str(pruned_ctl),
            "tokenizer": str(tok_dir),
            "mtp": False,
            "args": [
                "--pack", str(pack_dir),
                "--expert-profile", "profiles/qwen.profile",
            ],
            "vision": {
                "exe": str(pruned_eng / "strata-vision.exe"),
                "mmproj": str(mmproj),
                "model": str(model_gguf),
            },
        }
        cfg_path = root / "strata-iq2_xs.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return comp_root, eng_dir, cfg_path

    def test_doctor_and_models_rebase_pruned_vision_profile_and_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp_root, eng_dir, cfg_path = self._make_install(root)
            with mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(comp_root)}):
                report = doctor.inspect(cfg_path)
                self.assertTrue(report["ok"], f"unexpected issues: {report['issues']}")
                by_name = {c["name"]: c for c in report["components"]}
                self.assertEqual(by_name["engine"]["status"], "ok")
                self.assertEqual(Path(by_name["engine"]["path"]), eng_dir / "strata.exe")
                self.assertEqual(by_name["vision/exe"]["status"], "ok")
                self.assertEqual(Path(by_name["vision/exe"]["path"]), eng_dir / "strata-vision.exe")
                self.assertEqual(by_name["expert-profile"]["status"], "ok")
                self.assertTrue(Path(by_name["expert-profile"]["path"]).is_file())

                listed = models.list_installed(override=str(root))
                self.assertEqual(len(listed["models"]), 1)
                m = listed["models"][0]
                self.assertEqual(m["component_status"], "present")
                self.assertEqual(m["missing"], [])


class UpdaterSmokeComponentRoutingTests(unittest.TestCase):
    """#6623: updater smoke test must match configs to their engine component and never pass --engram to `strata`."""

    def test_config_component_id_distinguishes_general_hy4_and_inkling(self):
        self.assertEqual(updater._config_component_id({
            "exe": "component:strata-engine",
            "args": ["--pack", "/tmp/pack", "--max-context", "32768"],
        }), ec.COMPONENT_ID)
        self.assertEqual(updater._config_component_id({
            "exe": "component:strata-hy4",
            "arch": "hy4flash",
            "args": ["--native", "Hy4-Flash-SQ-64.strata", "--engram", "Hy4-Flash-Engram-Q8.strata"],
        }), "strata-hy4")
        # Even if a legacy Hy4 config had a plain exe path or missing arch, --engram routes it to strata-hy4
        self.assertEqual(updater._config_component_id({
            "exe": "/opt/strata/strata",
            "args": ["--native", "Hy4-Flash-SQ-64.strata", "--engram", "Hy4-Flash-Engram-Q8.strata"],
        }), "strata-hy4")
        self.assertEqual(updater._config_component_id({
            "exe": "component:strata-inkling",
            "arch": "inkling2_loop",
            "args": ["--native", "Inkling-2-Mini-SQ-15.strata", "--dspark", "dspark.gguf"],
        }), "strata-inkling")
        self.assertTrue(updater._is_shape_engine_config({"args": ["--engram", "x.strata"]}))
        self.assertTrue(updater._is_shape_engine_config({"args": ["--dspark", "d.gguf"]}))
        self.assertFalse(updater._is_shape_engine_config({"exe": "component:strata-engine", "args": ["--pack", "p"]}))

    def test_discover_run_config_filters_by_component_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen_cfg = root / "strata-qwen.json"
            hy4_cfg = root / "strata-hy4-flash-sq-64.json"
            ink_cfg = root / "strata-inkling-2-mini-sq-15.json"
            qwen_cfg.write_text(json.dumps({"exe": "component:strata-engine", "args": ["--pack", "p"]}), encoding="utf-8")
            hy4_cfg.write_text(json.dumps({"exe": "component:strata-hy4", "arch": "hy4flash", "args": ["--engram", "e.strata"]}), encoding="utf-8")
            ink_cfg.write_text(json.dumps({"exe": "component:strata-inkling", "arch": "inkling2_loop", "args": ["--dspark", "d.gguf"]}), encoding="utf-8")

            fake_listed = {
                "models": [
                    {"id": "hy4-flash-sq-64", "component_status": "present", "source_config": str(hy4_cfg)},
                    {"id": "inkling-2-mini-sq-15", "component_status": "present", "source_config": str(ink_cfg)},
                    {"id": "qwen-iq2_xs", "component_status": "present", "source_config": str(qwen_cfg)},
                ]
            }
            with mock.patch("strata_control.models.list_installed", return_value=fake_listed), \
                 mock.patch("strata_control.profiles.list_profiles", return_value=[]):
                self.assertEqual(updater.discover_run_config(component_id=ec.COMPONENT_ID), qwen_cfg)
                self.assertEqual(updater.discover_run_config(component_id="strata-hy4"), hy4_cfg)
                self.assertEqual(updater.discover_run_config(component_id="strata-inkling"), ink_cfg)

    def test_default_test_start_falls_back_to_binary_smoke_when_only_hy4_config_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hy4_cfg = root / "strata-hy4-flash-sq-64.json"
            hy4_cfg.write_text(json.dumps({
                "exe": "component:strata-hy4",
                "arch": "hy4flash",
                "args": ["--native", "Hy4-Flash-SQ-64.strata", "--engram", "Hy4-Flash-Engram-Q8.strata"],
            }), encoding="utf-8")
            comp = mock.MagicMock(spec=ec.EngineComponent)
            comp.id = ec.COMPONENT_ID
            comp.version = "0.1.65"
            comp.backend = "cuda"
            ctx = updater.Context(
                source="",
                trusted=[],
                hardware={},
                components=root / "components",
                run_config=hy4_cfg,
                allow_no_model_smoke=True,
            )
            with mock.patch.object(updater, "discover_run_config", return_value=None), \
                 mock.patch.object(updater, "binary_smoke", return_value={"mode": "binary_smoke", "ok": True}) as smoke:
                res = updater.default_test_start(root, comp, ctx)
            self.assertEqual(res["mode"], "binary_smoke")
            smoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
