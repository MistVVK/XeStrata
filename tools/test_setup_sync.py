# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sync's pinned files are reused, checked, and shared between model sizes."""
import hashlib
import tempfile
import unittest
from pathlib import Path
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

    def test_hash_direct_and_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "file"
            data = b"abcdefg" * ((32 << 20) // 7 + 1)
            p.write_bytes(data)
            self.assertEqual(sync_models.sha256(p, len(data)), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
