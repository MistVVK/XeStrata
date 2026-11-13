# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""setup.py and the person's own model files (--gguf-dir): setup checks them, it never deletes them (0.2.2).

    python -m unittest tools.test_setup_ownfiles
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def run(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn(*args, **kw)
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class PictureFileBeside(unittest.TestCase):
    """The picture file of a person's own folder (--gguf-dir), or of the folder above it and that folder's models/ (a
    data folder's layout), at the pinned size only for the two outer places (upstream 434c6221b)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.gdir = self.tmp / "models" / "IQ3_XXS"
        self.gdir.mkdir(parents=True)

    def test_the_folder_itself_is_taken_at_any_size(self):
        f = self.gdir / "mmproj-x.gguf"
        f.write_bytes(b"ab")
        self.assertEqual(setup.mmproj_beside(str(self.gdir), "mmproj-x.gguf", 99), f)

    def test_the_folder_above_and_its_models_are_taken_at_the_pinned_size(self):
        up = self.tmp / "models" / "mmproj-x.gguf"
        up.write_bytes(b"abc")
        self.assertEqual(setup.mmproj_beside(str(self.gdir), "mmproj-x.gguf", 3), up)
        self.assertIsNone(setup.mmproj_beside(str(self.gdir), "mmproj-x.gguf", 4))   # half copied: fetched again
        up.unlink()
        sub = self.tmp / "models" / "models" / "mmproj-x.gguf"
        sub.parent.mkdir()
        sub.write_bytes(b"abcd")
        self.assertEqual(setup.mmproj_beside(str(self.gdir), "mmproj-x.gguf", 4), sub)

    def test_no_gguf_dir_is_none(self):
        self.assertIsNone(setup.mmproj_beside(None, "mmproj-x.gguf", 1))


class OwnFileChecks(unittest.TestCase):
    def test_the_draft_layer_beside_a_model_is_not_a_second_model(self):
        # upstream #4693: the folder held "several models" for a model and the mtp-q2_0.gguf beside it
        folder = Path(self.tmp.name) / "own"
        folder.mkdir()
        (folder / "My-Finetune-IQ3_S-00001-of-00001.gguf").write_bytes(b"abc")
        (folder / "mtp-q2_0.gguf").write_bytes(b"abc")
        self.assertEqual([p.name for p in setup.model_firsts(folder)], ["My-Finetune-IQ3_S-00001-of-00001.gguf"])
        code, text = run(setup.gguf_files, folder)
        self.assertEqual(code, 0, text)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "model-00001-of-00002.gguf"
        self.file.write_bytes(b"abcdefgh" * 100)
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()

    def test_a_downloaded_file_of_the_wrong_size_asks_to_delete_it(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha)
        self.assertEqual(code, 1)
        self.assertIn("delete it and run setup again", text)

    def test_an_own_file_of_the_wrong_size_is_left_and_the_text_says_so(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha, own=True)
        self.assertEqual(code, 1)
        self.assertIn("your file was left as it is", text)
        self.assertTrue(self.file.exists())

    def test_an_own_file_with_another_hash_is_not_deleted(self):
        code, text = run(setup.verify_sha256, self.file, 800, "0" * 64, own=True)
        self.assertEqual(code, 1)
        self.assertIn("does not match the published file", text)
        self.assertTrue(self.file.exists())

    def test_a_downloaded_file_with_another_hash_is_deleted_as_before(self):
        code, _text = run(setup.verify_sha256, self.file, 800, "0" * 64)
        self.assertEqual(code, 1)
        self.assertFalse(self.file.exists())

    def test_a_matching_own_file_passes(self):
        code, _text = run(setup.verify_sha256, self.file, 800, self.sha, own=True)
        self.assertEqual(code, 0)


class CheckingShowsItsProgress(unittest.TestCase):
    def test_a_big_file_is_hashed_with_a_progress_line_of_its_own(self):
        """Hashing a 60 GB shard takes minutes: setup shows it as 'checking', with a figure (not a download stuck at
        99%)."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "big-00001-of-00002.gguf"
            size = (1 << 30) + 4096
            with open(f, "wb") as g:
                g.truncate(size)
            sha = hashlib.sha256()
            with open(f, "rb") as g:
                for block in iter(lambda: g.read(16 << 20), b""):
                    sha.update(block)
            import time
            original = time.time
            ticks = iter(range(0, 10_000, 3))
            setup.time.time = lambda: float(next(ticks))     # every read looks 3 s later: the progress line prints
            try:
                code, text = run(setup.verify_sha256, f, size, sha.hexdigest())
            finally:
                setup.time.time = original
        self.assertEqual(code, 0)
        self.assertIn("checking big-00001-of-00002.gguf (SHA-256", text)
        self.assertRegex(text, r"checking big-00001-of-00002\.gguf: +\d+\.\d\d / 1\.07 GB \(\d+%\)")


if __name__ == "__main__":
    unittest.main()
