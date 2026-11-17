"""strata_control/retired.py: DeepSeek configs, names, folders and files are recognised locally and at once; the words name the folder
and its size; a Qwen config never matches; nothing raises on unreadable input."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import retired  # noqa: E402


def gguf(path: Path, arch: str, extra_kv_first: bool = False) -> Path:
    """A GGUF header with general.architecture (optionally behind a u32 key) and no tensors."""
    def kv_str(k: str, v: str) -> bytes:
        kb, vb = k.encode(), v.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    kvs = []
    if extra_kv_first:
        kb = b"general.quantization_version"
        kvs.append(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", 2))
    kvs.append(kv_str("general.architecture", arch))
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs))
    return path


QWEN = {"exe": "/x/components/engine/0.1.60-linux-x86_64-cuda/strata", "args": ["--model", "/m/q.gguf"], "arch": "qwen3next",
        "model_name": "qwen3.8-flash-next-iq3_s"}
DS = {"exe": "/x/engine-dsv41/0.1.59-linux-x86_64-cuda-cuda13/strata-dsv41", "args": ["--pack", "/nowhere/deepseek"],
      "arch": "deepseek41", "model_name": "deepseek-v4.1-flash"}


class ConfigReason(unittest.TestCase):
    def test_each_reason(self):
        self.assertEqual(retired.config_reason(DS), "arch")
        self.assertEqual(retired.config_reason({"exe": "C:\\S\\engine-dsv41\\v\\strata-dsv41.exe", "args": []}), "engine")
        self.assertEqual(retired.config_reason({"exe": "/x/strata", "model_name": "deepseek-v4.1-flash-ist-das-2.00bpw"}), "name")
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "tokenizer"
            tok.mkdir()
            (tok / "tokenizer.json").write_text(json.dumps({"pre": "deepseek3"}), encoding="utf-8")
            self.assertEqual(retired.config_reason({"exe": "/x/strata", "tokenizer": str(tok)}), "tokenizer")
            (tok / "tokenizer.json").write_text(json.dumps({"pre": "qwen35"}), encoding="utf-8")
            self.assertIsNone(retired.config_reason({"exe": "/x/strata", "tokenizer": str(tok)}))

    def test_qwen_configs_never_match(self):
        self.assertIsNone(retired.config_reason(QWEN))
        for arch in ("qwen3next", "qwen35moe", "", None):
            self.assertIsNone(retired.config_reason({**QWEN, "arch": arch}))

    def test_bad_input_never_raises(self):
        for bad in (None, 3, "x", [], {"exe": 5, "arch": 7, "args": None, "tokenizer": 9}):
            self.assertIsNone(retired.config_reason(bad))
        self.assertIsNone(retired.config_file_reason("/no/such/strata-x.json"))
        self.assertEqual(retired.config_file_reason("/no/such/strata-deepseek-v4.1-flash-ist-das-2.00bpw.json"), "name")

    def test_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-anything.json"
            p.write_text(json.dumps(DS), encoding="utf-8")
            self.assertEqual(retired.config_file_reason(p), "arch")
            p.write_text(json.dumps(QWEN), encoding="utf-8")
            self.assertIsNone(retired.config_file_reason(p))


class Names(unittest.TestCase):
    def test_typed_names(self):
        for w in ("IST-DAS-2.00bpw", "ist-das-2.25bpw", "IST-DAS-1.75bpw", "ISTA-DAS-Coder", "deepseek41", "DeepSeek", "UD-Q8_K_XL"):
            self.assertTrue(retired.is_name(w), w)
        for w in ("IQ3_S", "UD-IQ1_M", "UD-IQ4_XS", "coder", "qwen", "", None, 4):
            self.assertFalse(retired.is_name(w), w)

    def test_file_names(self):
        for n in ("DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf", "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
                  "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf", "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"):
            self.assertTrue(retired.is_file_name(n), n)
        self.assertFalse(retired.is_file_name("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"))


class Files(unittest.TestCase):
    def test_gguf_arch(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(retired.gguf_arch(gguf(Path(d) / "a.gguf", "deepseek4")), "deepseek4")
            self.assertEqual(retired.gguf_arch(gguf(Path(d) / "b.gguf", "qwen3next", extra_kv_first=True)), "qwen3next")
            (Path(d) / "c.gguf").write_bytes(b"not a gguf")
            self.assertIsNone(retired.gguf_arch(Path(d) / "c.gguf"))
            self.assertIsNone(retired.gguf_arch(Path(d) / "missing.gguf"))

    def test_folder_reason(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "byname").mkdir()
            (root / "byname" / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").write_bytes(b"x")
            self.assertEqual(retired.folder_reason(root / "byname"), "name")
            (root / "byarch").mkdir()
            gguf(root / "byarch" / "model-00001-of-00004.gguf", "deepseek4")
            self.assertEqual(retired.folder_reason(root / "byarch"), "gguf")
            self.assertEqual(retired.folder_reason(root / "byarch" / "model-00001-of-00004.gguf"), "gguf")
            (root / "qwen").mkdir()
            gguf(root / "qwen" / "Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf", "qwen3next")
            self.assertIsNone(retired.folder_reason(root / "qwen"))
            self.assertIsNone(retired.folder_reason(root / "none"))

    def test_bounded_look_is_quick(self):
        """A folder with many entries is looked at, not walked: the answer comes at once."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(600):
                (root / f"f{i:04d}.bin").write_bytes(b"")
            sub = root / "deep"
            sub.mkdir()
            (sub / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").write_bytes(b"x")
            t = time.monotonic()
            self.assertIsNone(retired.folder_reason(root))          # the DeepSeek file is one level down: not entered
            self.assertIsNotNone(retired.where([root]))
            self.assertLess(time.monotonic() - t, 2.0)


class Words(unittest.TestCase):
    def test_refusal_names_folder_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            models = Path(d) / "models"
            models.mkdir()
            (models / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").write_bytes(b"\0" * 2_500_000)
            pack = Path(d) / "packs" / "deepseek"
            pack.mkdir(parents=True)
            (pack / "ds4.txt").write_text(f"n_layer 43\nformat gguf\nmodel_dir {models}\n", encoding="utf-8")
            cfg = {**DS, "args": ["--pack", str(pack)]}
            text = retired.refusal(cfg, "DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw")
            self.assertTrue(text.startswith("DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw is a DeepSeek model. This version of Strata no longer runs "
                                            "DeepSeek models, so it was not started."), text)
            self.assertIn(f"Its files are still on this computer: 0.0 GB in {models} and {pack}. Nothing was changed or deleted.", text)
            self.assertTrue(text.endswith("Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that "
                                          "folder to free the space."), text)
            self.assertNotIn("[DRAFT]", text)

    def test_refusal_without_files(self):
        text = retired.refusal(DS)
        self.assertNotIn("Its files", text)
        self.assertIn("deepseek-v4.1-flash is a DeepSeek model.", text)

    def test_sizes(self):
        self.assertEqual(retired.size_words(257_000_000_000), "257 GB")
        self.assertEqual(retired.size_words(2_500_000_000), "2.5 GB")


if __name__ == "__main__":
    unittest.main()
