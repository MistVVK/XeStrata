"""0.2.17 item 4: a picture reader the person already has in their own model folder under ANOTHER name (q38-mmproj-BF16.gguf) is used when it is
the right file for the model, found by its content (size and SHA-256 against the pin), and never when it is a reader of another model.

Two places decide this and both are tested: setup (setup.resolve_reader, the launch / install) and the app's picture step (pictures.fetch).
The tests use a small stand-in file and patch the pin."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_control import pictures
from tests.terminal.test_pictures import BODY, NAME, TINY, Layout

OTHER_NAME = "q38-mmproj-BF16.gguf"
WRONG_SAME_SIZE = bytes(reversed(BODY))                          # a reader of another model: same size, other bytes
assert len(WRONG_SAME_SIZE) == len(BODY) and WRONG_SAME_SIZE != BODY
SHA = hashlib.sha256(BODY).hexdigest()


class SetupRule(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix="strata-readercontent-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)
        p = mock.patch.object(setup, "ROOT", self.base / "strata-root")
        p.start()
        self.addCleanup(p.stop)
        self.own = self.base / "mine" / "IQ3_XXS"
        self.own.mkdir(parents=True)
        self.models = self.base / "data" / "models"
        self.models.mkdir(parents=True)
        self.pinned = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"

    def resolve(self, gguf_dir=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kind, path = setup.resolve_reader(self.pinned, self.models / self.pinned, self.models, [], gguf_dir or self.own, len(BODY), SHA)
        return kind, path, out.getvalue()

    def test_the_same_file_under_another_name_is_used_where_it_is(self):
        f = self.own / OTHER_NAME
        f.write_bytes(BODY)
        kind, path, said = self.resolve()
        self.assertEqual((kind, path), ("renamed", f))
        self.assertEqual(sorted(p.name for p in self.own.iterdir()), [OTHER_NAME], "nothing is written beside the person's file")
        self.assertEqual(f.read_bytes(), BODY)

    def test_the_folder_above_and_its_models_folder_count_too(self):
        for where in (self.own.parent, self.own.parent / "models"):
            where.mkdir(parents=True, exist_ok=True)
            f = where / OTHER_NAME
            f.write_bytes(BODY)
            self.assertEqual(self.resolve()[:2], ("renamed", f), str(where))
            f.unlink()
        self.assertEqual(self.resolve()[0], "fetch")

    def test_a_reader_of_another_model_is_never_taken_same_size(self):
        (self.own / OTHER_NAME).write_bytes(WRONG_SAME_SIZE)
        kind, path, _said = self.resolve()
        self.assertEqual((kind, path), ("fetch", self.models / self.pinned), "the pinned file is downloaded")
        self.assertEqual((self.own / OTHER_NAME).read_bytes(), WRONG_SAME_SIZE, "the person's file is left as it is")

    def test_a_reader_of_another_model_is_never_taken_other_size_and_is_not_even_read(self):
        (self.own / OTHER_NAME).write_bytes(BODY + b"!")
        (self.own / "mmproj-F16.gguf").write_bytes(BODY[:-1])
        hashed = []
        real = setup._hash_file
        with mock.patch.object(setup, "_hash_file", lambda p, s: (hashed.append(Path(p).name), real(p, s))[1]):
            kind, _path, _said = self.resolve()
        self.assertEqual(kind, "fetch")
        self.assertEqual(hashed, [], "another size is not hashed")

    def test_without_the_pinned_hash_nothing_is_taken_by_content(self):
        (self.own / OTHER_NAME).write_bytes(BODY)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kind, _path = setup.resolve_reader(self.pinned, self.models / self.pinned, self.models, [], self.own, len(BODY), None)
        self.assertEqual(kind, "fetch")

    def test_the_pinned_name_still_wins_and_a_model_shard_is_not_a_reader(self):
        (self.own / OTHER_NAME).write_bytes(BODY)
        (self.own / "Qwen-00001-of-00002.gguf").write_bytes(WRONG_SAME_SIZE)
        pinned = self.own / self.pinned
        pinned.write_bytes(b"x")
        self.assertEqual(self.resolve()[:2], ("near", pinned))
        pinned.unlink()
        self.assertEqual(self.resolve()[:2], ("renamed", self.own / OTHER_NAME))

    def test_the_hash_is_remembered_so_the_next_start_does_not_read_the_file_again(self):
        (self.own / OTHER_NAME).write_bytes(BODY)
        self.assertEqual(self.resolve()[0], "renamed")
        hashed = []
        with mock.patch.object(setup, "_hash_file", lambda p, s: hashed.append(p) or "x"):
            self.assertEqual(self.resolve()[0], "renamed")
        self.assertEqual(hashed, [])

    def test_the_unsloth_f16_under_another_name_counts_too(self):
        f = self.own / "my-vision-tower.gguf"
        f.write_bytes(BODY)
        alt = {self.pinned: ("mmproj-F16.gguf", len(BODY), SHA)}
        with mock.patch.dict(setup.ALT_READERS, alt, clear=True):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                kind, path = setup.resolve_reader(self.pinned, self.models / self.pinned, self.models, [], self.own, 10, "0" * 64)
        self.assertEqual((kind, path), ("alt", f))

    def test_no_own_folder_means_nothing(self):
        self.assertIsNone(setup.own_reader_by_content(None, self.pinned, len(BODY), SHA))
        self.assertIsNone(setup.own_reader_by_content(self.base / "missing", self.pinned, len(BODY), SHA))


class AppStep(unittest.TestCase):
    """pictures.fetch: the same rule for a model that is already installed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_a_renamed_copy_beside_the_model_is_used_and_nothing_is_downloaded(self):
        lay = Layout(self.base)
        mine = lay.model_dir / OTHER_NAME
        mine.write_bytes(BODY)
        listing = sorted(p.name for p in lay.model_dir.iterdir())
        result = lay.fetch(online=lambda url: self.fail("no download"))
        self.assertFalse(result["fetched"])
        self.assertEqual(Path(result["mmproj"]), mine)
        self.assertEqual(sorted(p.name for p in lay.model_dir.iterdir()), listing, "nothing written beside the person's files")
        self.assertFalse((lay.data / "models" / NAME).exists())

    def test_a_renamed_copy_one_folder_up_counts(self):
        lay = Layout(self.base)
        mine = lay.model_dir.parent / OTHER_NAME
        mine.write_bytes(BODY)
        result = lay.fetch(online=lambda url: self.fail("no download"))
        self.assertEqual((result["fetched"], Path(result["mmproj"])), (False, mine))

    def test_a_reader_of_another_model_is_not_used_and_the_pinned_file_is_downloaded(self):
        lay = Layout(self.base)
        for size_ok in (True, False):
            mine = lay.model_dir / OTHER_NAME
            mine.write_bytes(WRONG_SAME_SIZE if size_ok else BODY[:-3])
            result = lay.fetch()
            self.assertTrue(result["fetched"], f"same size: {size_ok}")
            self.assertEqual(Path(result["mmproj"]), lay.data / "models" / NAME)
            self.assertEqual(mine.read_bytes(), WRONG_SAME_SIZE if size_ok else BODY[:-3], "the person's file is untouched")
            (lay.data / "models" / NAME).unlink()
            (lay.data / "models" / (NAME + ".done")).unlink()


if __name__ == "__main__":
    unittest.main()
