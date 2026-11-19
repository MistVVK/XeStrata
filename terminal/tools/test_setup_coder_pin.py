"""The Coder model's pin (#3445): the revision of ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF that setup downloads and the
size and SHA-256 of its first shard belong together. Both were read from Hugging Face on 2026-11-19: the repository's
newest commit is bba88856 (2026-10-19, "Improved Coder build ... Shard 2 unchanged"), and for that commit the file
IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf is 29,608,446,496 bytes with the LFS SHA-256 below (the
header X-Linked-ETag of the resolve address). The old pin 5348543e had e11083ba..., so a pin moved without its hash
(or the other way) is a start that fails its own check.

    python -m unittest tools/test_setup_coder_pin.py -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"


class CoderPin(unittest.TestCase):
    def test_revision_and_first_shard_are_the_2026_10_19_build(self):
        self.assertEqual(setup.HF_REVISIONS[REPO], "bba88856f8c3c0c9e40e45f82d05e228b814ac43")
        size, sha = setup.CODER_FILES[SHARD1]
        self.assertEqual(size, 29608446496)
        self.assertEqual(sha, "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211")

    def test_the_second_shard_and_the_picture_reader_are_unchanged(self):
        # "Shard 2 unchanged" in the repository's commit; the picture reader is the original's file
        self.assertEqual(setup.CODER_FILES["Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"],
                         (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"))
        self.assertEqual(setup.CODER_FILES["mmproj-Qwen3.8-Flash-Next-BF16.gguf"],
                         (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"))


if __name__ == "__main__":
    unittest.main()
