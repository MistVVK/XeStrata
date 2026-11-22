"""Tests for Inkling-2-Mini local .gguf (unsloth + ggml-org, single & split) and .strata import, setup, adopt, and MCP plan."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_cli.launch import adopt
from strata_control import models
from strata_control.profiles import ProfileError


def _gguf_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make_inkling_gguf(
    *,
    architecture: str | None = "inkling2",
    tokenizer: bool = True,
    expert_type: int = 12,  # Q4_K (256 elements -> 144 bytes)
    include_merges: bool = True,
) -> bytes:
    pairs = []
    if architecture is not None:
        pairs.append(("general.architecture", 8, _gguf_str(architecture)))
    if tokenizer:
        pairs.append(("tokenizer.ggml.model", 8, _gguf_str("gpt2")))
        pairs.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + _gguf_str("a")))
        if include_merges:
            pairs.append(("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + _gguf_str("a b")))
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, len(pairs))
        + b"".join(_gguf_str(k) + struct.pack("<I", kind) + val for k, kind, val in pairs)
    )
    # Q4_K=12 (256 -> 144), Q8_0=8 (32 -> 34), IQ4_NL=20 (32 -> 18)
    geom = {12: (256, 144), 8: (32, 34), 20: (32, 18), 42: (64, 18)}
    elements, payload_bytes = geom[expert_type]
    header += _gguf_str("blk.0.ffn_gate_exps.weight") + struct.pack("<IQIQ", 1, elements, expert_type, 0)
    return header + b"\x00" * (-len(header) % 32) + b"\x00" * payload_bytes


def make_fake_strata() -> bytes:
    # 8-byte magic + 8-byte reserved + 8-byte manifest off + 8-byte manifest len
    return b"STRATA\x00\x01" + struct.pack("<QQQ", 0, 4096, 16) + b"\x00" * 4096


def write_fake_o200k_files(folder: Path) -> None:
    tok = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
        "added_tokens": [{"id": 200006, "content": "<|content_model_end_sampling|>", "special": True}],
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"\p{N}{1,3}|[^\s]+"}, "behavior": "Isolated"},
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
    }
    (folder / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")


class InklingGgufImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_plan_import_unsloth_gguf_with_aux_files(self):
        folder = self.root / "unsloth-inkling"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
        (folder / "dspark-Inkling-2-Mini-Q8_0.gguf").write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=8))
        (folder / "mmproj-BF16.gguf").write_bytes(make_inkling_gguf(architecture="clip", tokenizer=False, expert_type=8))
        (folder / "imatrix.gguf").write_bytes(b"GGUF\x03\x00\x00\x00")

        # Folder import ignores dspark / mmproj / imatrix and picks Inkling-2-Mini cleanly
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["file_set_status"], "plausible", plan)
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["family"], "inkling-2-mini")
        self.assertEqual(plan["candidate_selection"]["basis"], "inkling2_gguf")
        self.assertTrue(plan["reads_pictures"])
        self.assertTrue(any("DSpark" in m for m in plan["messages"]))

        # Direct single-file import works too
        plan_file = models.plan_import(str(gguf))
        self.assertEqual(plan_file["file_set_status"], "plausible")
        self.assertEqual(plan_file["candidate_selection"]["family"], "inkling-2-mini")

        # Picking dspark or imatrix directly is refused with a helpful message
        with self.assertRaises(ProfileError):
            models.plan_import(str(folder / "dspark-Inkling-2-Mini-Q8_0.gguf"))
        with self.assertRaises(ProfileError):
            models.plan_import(str(folder / "imatrix.gguf"))

    def test_plan_import_ggml_org_inkling2_loop_gguf(self):
        folder = self.root / "ggml-inkling"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-IQ4_NL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=20))
        (folder / "mmproj-Inkling-2-Mini-BF16.gguf").write_bytes(make_inkling_gguf(architecture="clip", tokenizer=False, expert_type=8))

        plan = models.plan_import(str(gguf))
        self.assertEqual(plan["file_set_status"], "plausible", plan)
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["metadata"]["architecture"], "inkling2_loop")
        self.assertEqual(plan["candidate_selection"]["family"], "inkling-2-mini")
        self.assertEqual(plan["candidate_selection"]["model"], "SQ-10")
        self.assertTrue(plan["reads_pictures"])

    def test_plan_import_split_inkling_gguf_shards(self):
        folder = self.root / "split-inkling"
        folder.mkdir()
        s1 = folder / "Inkling-2-Mini-Q8_0-00001-of-00002.gguf"
        s2 = folder / "Inkling-2-Mini-Q8_0-00002-of-00002.gguf"
        s1.write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=8))
        s2.write_bytes(make_inkling_gguf(architecture=None, tokenizer=False, expert_type=8))

        plan = models.plan_import(str(s2))
        self.assertEqual(plan["file_set_status"], "plausible", plan)
        self.assertEqual(plan["file_count"], 2)
        self.assertEqual(plan["candidate_selection"]["family"], "inkling-2-mini")

    def test_plan_import_local_strata_file_and_folder(self):
        folder = self.root / "local-strata"
        folder.mkdir()
        model_file = folder / "Inkling-2-Mini-SQ-20.strata"
        vision_file = folder / "Inkling-2-Mini-Vision.strata"
        model_file.write_bytes(make_fake_strata())
        vision_file.write_bytes(make_fake_strata())

        self.assertIsNone(models.own_path_problem(str(model_file)))
        self.assertIsNone(models.own_path_problem(str(folder)))

        with mock.patch("tools.inkling_tokenizer.read_manifest", side_effect=RuntimeError("no zstd")):
            plan_file = models.plan_import(str(model_file))
            self.assertEqual(plan_file["file_set_status"], "plausible", plan_file)
            self.assertEqual(plan_file["candidate_selection"]["family"], "inkling-2-mini")
            self.assertEqual(plan_file["candidate_selection"]["model"], "SQ-20")
            self.assertTrue(plan_file["reads_pictures"])

            plan_dir = models.plan_import(str(folder))
            self.assertEqual(plan_dir["file_set_status"], "plausible", plan_dir)
            self.assertEqual(plan_dir["candidate_selection"]["model"], "SQ-20")

        with self.assertRaises(ProfileError):
            models.plan_import(str(vision_file))

    def test_setup_local_source_and_install_inkling2_gguf(self):
        import argparse
        import setup

        folder = self.root / "user-gguf"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
        dspark = folder / "dspark-Inkling-2-Mini-Q8_0.gguf"
        dspark.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=8))
        mmproj = folder / "mmproj-BF16.gguf"
        mmproj.write_bytes(b"GGUF\x03\x00\x00\x00")
        write_fake_o200k_files(folder)

        data_dir = self.root / "Strata-data"
        install_dir = self.root / "install"
        install_dir.mkdir()
        data_dir.mkdir()

        args_ns = argparse.Namespace(
            model="SQ-15",
            family="inkling-2-mini",
            context=32768,
            vision="yes",
            yes=True,
            no_start=True,
            gguf_dir=str(gguf),
            models_dir=str(data_dir / "models"),
            data_dir=str(data_dir),
            backend=None,
            gpu=None,
            host=None,
            api_key=None,
            browser=None,
            port=8080,
        )
        src = setup.inkling_local_source(args_ns, "SQ-15")
        self.assertIsNotNone(src)
        self.assertEqual(src[0], gguf)
        self.assertTrue(src[2])
        self.assertEqual(setup.inkling_find_local_vision(folder, data_dir, True), mmproj)
        self.assertEqual(setup.inkling_find_local_dspark(folder), dspark)

        fake_eng = self.root / "eng"
        fake_eng.mkdir()
        (fake_eng / setup.inkling_family.EXE).write_bytes(b"exe")
        gpu = {"index": 0, "name": "RTX 5070", "vendor": "nvidia", "vram_gb": 12.0, "count": 1}

        with mock.patch.object(setup, "home", return_value=install_dir), \
             mock.patch.object(setup, "data_folder_for", return_value=(data_dir, "explicit")), \
             mock.patch.object(setup, "pip_install"), \
             mock.patch.object(setup, "engine_step", return_value=(None, fake_eng, {"component": "strata-inkling"}, [], gpu, "none")), \
             mock.patch.object(setup, "engine_exe", return_value="component:strata-inkling"), \
             mock.patch.object(setup, "write_run_script", return_value=install_dir / "run.bat"):
            rc = setup.install_inkling2(args_ns, "inkling-2-mini", gpu, False, False, None, False, 8080, 32.0, None)
        self.assertEqual(rc, 0)
        cfg_path = install_dir / "strata-inkling-2-mini-sq-15.json"
        self.assertTrue(cfg_path.is_file())
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["exe"], "component:strata-inkling")
        self.assertEqual(cfg["arch"], "inkling2_loop")
        self.assertEqual(cfg["drafting"], "dspark")
        self.assertNotIn("reasoning_effort", cfg)
        self.assertIn(str(gguf), cfg["args"])
        self.assertIn("--vision", cfg["args"])
        self.assertIn(str(mmproj), cfg["args"])
        self.assertIn("--dspark", cfg["args"])
        self.assertIn(str(dspark), cfg["args"])
        self.assertTrue((Path(cfg["tokenizer"]) / "vocab.json").is_file())

    def test_setup_install_inkling2_gguf_without_dspark_or_vision(self):
        import argparse
        import setup

        folder = self.root / "ggml-text-only"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-IQ4_NL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=20))
        write_fake_o200k_files(folder)
        data_dir = self.root / "Strata-data-2"
        install_dir = self.root / "install-2"
        install_dir.mkdir()
        data_dir.mkdir()

        args_ns = argparse.Namespace(
            model="SQ-10",
            family="inkling-2-mini",
            context=32768,
            vision="yes",
            yes=True,
            no_start=True,
            gguf_dir=str(folder),
            models_dir=str(data_dir / "models"),
            data_dir=str(data_dir),
            backend=None,
            gpu=None,
            host=None,
            api_key=None,
            browser=None,
            port=8080,
        )
        fake_eng = self.root / "eng2"
        fake_eng.mkdir()
        gpu = {"index": 0, "name": "RTX 5070", "vendor": "nvidia", "vram_gb": 12.0, "count": 1}

        with mock.patch.object(setup, "home", return_value=install_dir), \
             mock.patch.object(setup, "data_folder_for", return_value=(data_dir, "explicit")), \
             mock.patch.object(setup, "pip_install"), \
             mock.patch.object(setup, "engine_step", return_value=(None, fake_eng, {"component": "missing"}, [], gpu, "none")), \
             mock.patch.object(setup, "engine_exe", return_value="component:strata-inkling"), \
             mock.patch.object(setup, "write_run_script", return_value=install_dir / "run.bat"):
            rc = setup.install_inkling2(args_ns, "inkling-2-mini", gpu, False, False, None, False, 8080, 32.0, None)
        self.assertEqual(rc, 0)
        cfg = json.loads((install_dir / "strata-inkling-2-mini-sq-10.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["drafting"], "off")
        self.assertNotIn("--vision", cfg["args"])
        self.assertNotIn("--dspark", cfg["args"])
        self.assertNotIn("reasoning_effort", cfg)

    def test_adopt_plan_supports_inkling_gguf_and_strata(self):
        folder = self.root / "adopt-ink"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-IQ4_NL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=20))
        data_dir = self.root / "adopt-data"
        tok = data_dir / "packs" / "inkling-2-mini-sq-10" / "tokenizer"
        tok.mkdir(parents=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")

        cfg, problems = adopt.plan(
            family="inkling-2-mini",
            model="SQ-10",
            model_dir=folder,
            data_dir=data_dir,
            src=ROOT,
            context=32768,
            kv=None,
            model_name="inkling-2-mini-sq-10",
            family_def={},
            log=self.root / "ink.log",
            backend="sycl",
        )
        self.assertEqual(problems, [])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["exe"], "component:strata-inkling")
        self.assertEqual(cfg["arch"], "inkling2_loop")
        self.assertEqual(cfg["drafting"], "off")
        self.assertIn(str(gguf), cfg["args"])
        self.assertNotIn("reasoning_effort", cfg)

    def test_strata_mcp_install_plan_accepts_inkling_gguf_file(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_mcp

        folder = self.root / "mcp-ink"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
        install_dir = self.root / "mcp-install"
        install_dir.mkdir()
        strata = strata_mcp.Strata(install_dir, ROOT)
        tools = strata_mcp.Tools(strata)
        with mock.patch.object(strata, "hardware", return_value={"ram_gb": 32, "gpus": [{"index": 0, "name": "RTX 5070", "vram_gb": 12.0, "usable": True, "vendor": "nvidia"}]}), \
             mock.patch.object(strata, "inkling_visible", return_value=True):
            plan, argv = tools.install_plan("inkling-2-mini", "SQ-15", 32768, "no", None, None, None, None, None, None, None, gguf_dir=str(gguf))
        self.assertEqual(plan["download_gb"], 0.0)
        self.assertIn("--gguf-dir", argv)
        self.assertIn(str(gguf), argv)


if __name__ == "__main__":
    unittest.main()
