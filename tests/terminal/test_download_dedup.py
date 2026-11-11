"""A shared picture file is found where the person already has it and never fetched a second time (setup.mmproj_beside, setup.share_sibling_file)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import setup

NAME = "mmproj-X.gguf"


class MmprojBeside(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gguf = self.root / "models" / "IQ3_S"
        self.gguf.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_without_a_folder(self):
        self.assertIsNone(setup.mmproj_beside(None, NAME, 4))

    def test_the_folder_itself_first_at_any_size(self):
        (self.gguf / NAME).write_bytes(b"12")
        (self.root / "models" / NAME).write_bytes(b"1234")
        self.assertEqual(setup.mmproj_beside(self.gguf, NAME, 4), self.gguf / NAME)

    def test_the_folder_above_in_a_data_folder_layout(self):
        (self.root / "models" / NAME).write_bytes(b"1234")
        self.assertEqual(setup.mmproj_beside(self.gguf, NAME, 4), self.root / "models" / NAME)

    def test_a_models_folder_next_to_the_size_folder(self):
        shards = self.root / "IQ3_S"
        shards.mkdir()
        (self.root / "models" / NAME).write_bytes(b"1234")
        self.assertEqual(setup.mmproj_beside(shards, NAME, 4), self.root / "models" / NAME)

    def test_a_half_copied_file_above_is_not_used(self):
        (self.root / "models" / NAME).write_bytes(b"12")
        self.assertIsNone(setup.mmproj_beside(self.gguf, NAME, 4))

    def test_the_pinned_size_is_optional(self):
        (self.root / "models" / NAME).write_bytes(b"12")
        self.assertEqual(setup.mmproj_beside(self.gguf, NAME), self.root / "models" / NAME)


class ShareSibling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)
        for q in ("qwen36-UD-Q4_K_XL", "qwen36-UD-Q6_K_XL"):
            (self.models / q).mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_another_size_s_copy_is_linked_not_fetched(self):
        src = self.models / "qwen36-UD-Q4_K_XL" / NAME
        src.write_bytes(b"1234")
        dst = self.models / "qwen36-UD-Q6_K_XL" / NAME
        self.assertTrue(setup.share_sibling_file(dst, self.models, NAME, 4))
        self.assertEqual(dst.read_bytes(), b"1234")
        self.assertTrue(src.exists())

    def test_a_copy_of_the_wrong_size_is_not_taken(self):
        (self.models / "qwen36-UD-Q4_K_XL" / NAME).write_bytes(b"12")
        dst = self.models / "qwen36-UD-Q6_K_XL" / NAME
        self.assertFalse(setup.share_sibling_file(dst, self.models, NAME, 4))
        self.assertFalse(dst.exists())

    def test_no_sibling_means_nothing_is_made(self):
        dst = self.models / "qwen36-UD-Q6_K_XL" / NAME
        self.assertFalse(setup.share_sibling_file(dst, self.models, NAME, 4))

    def test_a_file_already_there_is_left_alone(self):
        dst = self.models / "qwen36-UD-Q6_K_XL" / NAME
        dst.write_bytes(b"mine")
        (self.models / "qwen36-UD-Q4_K_XL" / NAME).write_bytes(b"1234")
        self.assertTrue(setup.share_sibling_file(dst, self.models, NAME, 4))
        self.assertEqual(dst.read_bytes(), b"mine")

    def test_a_copy_when_a_link_is_not_possible(self):
        (self.models / "qwen36-UD-Q4_K_XL" / NAME).write_bytes(b"1234")
        dst = self.models / "qwen36-UD-Q6_K_XL" / NAME
        orig = os.link
        os.link = lambda *a, **k: (_ for _ in ()).throw(OSError("no links here"))
        try:
            self.assertTrue(setup.share_sibling_file(dst, self.models, NAME, 4))
        finally:
            os.link = orig
        self.assertEqual(dst.read_bytes(), b"1234")
        self.assertFalse(dst.with_name(dst.name + ".copying").exists())



if __name__ == "__main__":
    unittest.main()
