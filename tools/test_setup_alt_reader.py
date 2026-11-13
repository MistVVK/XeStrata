# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""An Unsloth mmproj-F16.gguf the person already has is accepted as the Flash-Next picture reader (setup.py's
ALT_READERS / alt_reader_places / accepted_alt_reader): only with the pinned size and SHA-256, only when the pinned
BF16 file is found nowhere, never downloaded, and the SHA-256 is remembered in the state folder, never beside the
person's file.  A file replaced by another of the same size is hashed again.  No GPU, no download.

    python -m unittest tools.test_setup_alt_reader
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

SIZE = 904004000


class AltReader(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.models = self.dir / "models"
        self.models.mkdir()
        # the pinned SHA-256 is of a real 0.9 GB file; the test takes a digest of its own and pins it for the run
        self.file = self.models / "mmproj-F16.gguf"
        with open(self.file, "wb") as f:
            f.write(b"unsloth reader")
        os.truncate(self.file, SIZE)
        self.sha = self.hash_now()
        self.patch = mock.patch.dict(setup.ALT_READERS,
                                     {setup.MMPROJ: ("mmproj-F16.gguf", SIZE, self.sha)})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.state = self.dir / "state"
        patcher = mock.patch.object(setup, "STATE", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def hash_now(self) -> str:
        h = hashlib.sha256()
        with open(self.file, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def accept(self):
        return setup.accepted_alt_reader([self.file])

    def test_a_pinned_file_is_accepted_and_its_hash_remembered_in_the_state_folder(self):
        self.assertEqual(self.accept(), self.file)
        marks = list((self.state / "marks").glob("*.done"))
        self.assertEqual(len(marks), 1)
        self.assertIn(self.sha, marks[0].read_text(encoding="utf-8"))
        self.assertEqual(list(self.models.glob("*.done")), [], "nothing is written beside the person's file")

    def test_the_mark_saves_the_second_hash_of_the_same_file(self):
        self.assertEqual(self.accept(), self.file)
        with mock.patch.object(setup, "_hash_file", side_effect=AssertionError("hashed again")):
            self.assertEqual(self.accept(), self.file)

    def test_a_replaced_file_of_the_same_size_is_hashed_again_and_refused(self):
        self.assertEqual(self.accept(), self.file)
        with open(self.file, "r+b") as f:              # another file, the same size: a new modification time
            f.write(b"another reader!!")
            f.flush()
            os.fsync(f.fileno())
        st = self.file.stat()
        os.utime(self.file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        self.addCleanup(setattr, self, "sha", self.sha)
        with mock.patch.object(setup, "_hash_file", return_value="0" * 64) as hashed:
            self.assertIsNone(self.accept())
        hashed.assert_called_once()

    def test_another_size_is_not_hashed(self):
        os.truncate(self.file, SIZE - 1)
        with mock.patch.object(setup, "_hash_file", side_effect=AssertionError("hashed")):
            self.assertIsNone(self.accept())

    def test_another_name_is_ignored(self):
        other = self.models / "mmproj-BF16.gguf"
        os.link(self.file, other)
        with mock.patch.object(setup, "_hash_file", side_effect=AssertionError("hashed")):
            self.assertIsNone(setup.accepted_alt_reader([other]))

    def test_the_places_are_the_pinned_ones(self):
        gguf = self.dir / "gguf" / "shards"
        places = setup.alt_reader_places(self.dir / "data" / "models", [self.dir / "old"], gguf)
        self.assertEqual([str(p) for p in places], [
            str(self.dir / "data" / "models" / "mmproj-F16.gguf"),
            str(self.dir / "old" / "models" / "mmproj-F16.gguf"),
            str(gguf / "mmproj-F16.gguf"),
            str(gguf.parent / "mmproj-F16.gguf"),
            str(gguf.parent / "models" / "mmproj-F16.gguf"),
        ])


if __name__ == "__main__":
    unittest.main()
