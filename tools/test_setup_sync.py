# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sync's pinned files are reused, checked, and shared between model sizes."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import setup
from serve import sync_models


class SyncFiles(unittest.TestCase):
    def test_reuse_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = b"model bytes"
            pin = (len(data), hashlib.sha256(data).hexdigest())
            packs = {"Sync-XS": {"pack.bin": pin}, "Sync-S": {"pack.bin": pin}}
            shared = {"native/common.bin": pin}
            calls = []

            def fetch(url, dst, label):
                calls.append(url)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)

            with (mock.patch.object(setup, "STATE", root / "state"),
                  mock.patch.object(sync_models, "PACK_FILES", packs),
                  mock.patch.object(sync_models, "SHARED_FILES", shared),
                  mock.patch.object(setup, "download_file", side_effect=fetch)):
                pack, common = setup.fetch_sync_files(root, "Sync-XS")
                setup.fetch_sync_files(root, "Sync-XS")
                setup.fetch_sync_files(root, "Sync-S")
                self.assertEqual(len(calls), 3)
                self.assertTrue((common / "native/common.bin").is_file())
                (pack / "pack.bin").write_bytes(b"bad content")
                with self.assertRaises(SystemExit):
                    setup.fetch_sync_files(root, "Sync-XS")
                self.assertEqual((pack / "pack.bin").read_bytes(), b"bad content")

    def test_setup_wires_pack_and_model_alias(self):
        a = SimpleNamespace(gguf_dir=None, custom=None, drafting="mtp", draft_vocab=None, models_dir="/models",
                            resident_budget_gib=None, ple_io=None, vram_reserve_mib=None, ram_reserve_gib=None,
                            host=None, api_key=None, browser=None, parallel=None, images=None)
        gpu = {"pci": "0000:01:00.0"}
        with (mock.patch.object(setup, "prepare_engine", return_value=(Path("/llama"), Path("/engine"), [])),
              mock.patch.object(setup, "fetch_sync_files", return_value=(Path("/pack"), Path("/shared"))),
              mock.patch.object(setup.model_terms, "fetch"), mock.patch.object(setup, "setup_translate"),
              mock.patch.object(setup, "replicas_config"),
              mock.patch.object(setup, "finish_setup") as finish):
            setup.setup_sync(a, "sync", "Sync-XS", gpu, [0], [gpu], True, 64, [], 8095, None, 8192, "fp16",
                             "none", True, True, None, 1.0)
        cfg = finish.call_args.args[1]
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-sync-xs")
        self.assertEqual(cfg["aliases"], ["Sync-XS", "sync-xs"])
        self.assertIn("--resident-experts", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--pack") + 1], "/pack")
        self.assertEqual(cfg["tokenizer"], "/shared/tokenizer")

    def test_draft_subset_does_not_change_the_shared_files(self):
        with tempfile.TemporaryDirectory() as temp:
            shared = Path(temp) / "sync-shared"
            source = shared / "mtp-rt"
            source.mkdir(parents=True)
            for name in ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin"):
                (source / name).write_bytes(name.encode())
            self.assertEqual(setup.sync_mtp(shared, "cjk"), source)
            derived = setup.sync_mtp(shared, "en")
            self.assertEqual((source / "draft_vocab.bin").read_bytes(), b"draft_vocab.bin")
            self.assertEqual((derived / "draft_vocab.bin").read_bytes(),
                             (setup.ROOT / "data" / setup.DRAFT_VOCABS["en"]).read_bytes())
            for name in ("dense.bin", "dense.txt", "experts.bin"):
                self.assertTrue((source / name).samefile(derived / name))
            self.assertEqual(setup.sync_mtp(shared, "en"), derived)

    def test_hash_direct_and_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "file"
            data = b"abcdefg" * ((32 << 20) // 7 + 1)
            p.write_bytes(data)
            self.assertEqual(sync_models.sha256(p, len(data)), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
