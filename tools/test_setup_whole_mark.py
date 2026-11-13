# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A full-length shard with no finish mark and a pinned SHA-256 is not marked "whole" (upstream c3109534d, #4088).

A run killed between the rename of a finished .part and the end of its hash leaves exactly that state; the mark the
next run wrote made done() true, download() skipped the file, the pinned hash was never checked and a shard damaged
by that crash stayed in place.

    python -m unittest tools.test_setup_whole_mark
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class WholeMarkNeedsThePin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.repo = next(r for r, files in setup.REPO_PINS.items() if files)
        self.name = next(iter(setup.REPO_PINS[self.repo]))
        self.hf = f"https://huggingface.co/{self.repo}/resolve/main/"
        patch = mock.patch.object(setup, "whole_shard", return_value=True)   # about the mark, not the header
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_pinned_shard_is_left_for_the_download_check(self):
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        self.assertIsNotNone(setup.file_pin(self.hf + self.name),
                             "the file needs a pin for this test to mean anything")
        setup.mark_whole_unless_pinned([shard], self.hf, own=False)
        self.assertFalse(setup.done(shard), "no finish mark: download() hashes it and repairs it")
        self.assertIsNone(setup.mark_text(shard))

    def test_a_file_without_a_pin_keeps_the_whole_mark(self):
        shard = self.folder / "hand-copied.gguf"
        shard.write_bytes(b"x")
        self.assertIsNone(setup.file_pin(self.hf + shard.name))
        setup.mark_whole_unless_pinned([shard], self.hf, own=False)
        self.assertTrue(setup.done(shard))
        self.assertIn("whole", setup.mark_text(shard) or "")

    def test_the_persons_own_file_keeps_the_mark(self):
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        setup.mark_whole_unless_pinned([shard], self.hf, own=True)          # --gguf-dir: not a download
        self.assertTrue(setup.done(shard))

    def test_a_family_without_a_hugging_face_folder_keeps_the_mark(self):
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        setup.mark_whole_unless_pinned([shard], None, own=False)          # a file list, not shards: no pin lookup
        self.assertTrue(setup.done(shard))

    def test_a_marked_or_missing_file_is_not_touched(self):
        shard = self.folder / "marked.gguf"
        shard.write_bytes(b"x")
        setup.mark(shard, "sha256 abc")
        setup.mark_whole_unless_pinned([shard, self.folder / "absent.gguf"], self.hf, own=False)
        self.assertEqual((setup.mark_text(shard) or "").strip(), "sha256 abc")


if __name__ == "__main__":
    unittest.main()
