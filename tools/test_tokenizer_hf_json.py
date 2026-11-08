# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools/strata_tokenizer.py's `from_hf`: Hugging Face's `tokenizer.json` read as it is.

An image model that reads its own prompt (HunyuanImage 3.0) ships a `tokenizer.json` and no GGUF, so the pack's
`from_dir` and a GGUF's `from_gguf` do not reach it.  `from_hf` reads the file's `model.vocab`, `model.merges`
and `added_tokens`, and classifies its `pre_tokenizer` regex against the shapes XeStrata knows BY WHAT THEY SPLIT
(`(?i:'s|'t|...)` and the spelled-out alternatives are one machine written twice).

    python -m unittest tools.test_tokenizer_hf_json -v

The fixtures are written here, so the real file is not needed.  With `$XESTRATA_HUNYUANIMAGE3_TOKENIZER` pointing at the
folder that holds HunyuanImage 3.0's `tokenizer.json`, `RealHunyuanImage3` runs against it and compares the ids
with the ones the file's own BPE gives - checked against Hugging Face's loader of the same file on 2026-10-24.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

# The regex HunyuanImage 3.0's file carries (its `pre_tokenizer`'s Split), verbatim.
HUNYUANIMAGE3_REGEX = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                       "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")


def fixture_doc(with_normalizer=False, ignore_merges=True, pattern=HUNYUANIMAGE3_REGEX, dropout=None,
                continuing=None, end_of_word=None):
    """A byte-level BPE small enough to check by hand: the 256 byte tokens (GPT-2's mapping), then `Ġa` and `Ġab`
    (what " a" and " ab" become) and the whole word `abc`, with the two merges that make them."""
    from strata_tokenizer import bytes_to_unicode
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["Ġa"] = 256
    vocab["Ġab"] = 257
    vocab["abc"] = 258
    return {
        "version": "test",
        "added_tokens": [
            {"id": 259, "content": "<boi>", "special": True},
            {"id": 260, "content": "<img>", "special": True},
            {"id": 261, "content": "<not_special>", "special": False},
        ],
        "normalizer": {"type": "NFC"} if with_normalizer else None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False,
                           "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": dropout, "unk_token": None,
            "continuing_subword_prefix": continuing, "end_of_word_suffix": end_of_word,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": ignore_merges,
            "vocab": vocab, "merges": ["Ġ a", "Ġa b"],
        },
    }


def write_fixture(dirpath: Path, doc) -> Path:
    p = dirpath / "tokenizer.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


class FromHfReadsTheFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def tok(self, **kwargs):
        return ST.Tokenizer.from_hf(write_fixture(self.dir, fixture_doc(**kwargs)))

    def test_the_shapes_the_literals_and_the_types(self):
        tok = self.tok()
        self.assertEqual(tok.pre, ST.HUNYUANIMAGE3_PRE)
        self.assertTrue(tok.ignore_merges, "the file sets model.ignore_merges")
        self.assertEqual(len(tok.tokens), 262)
        self.assertEqual(tok.tokens[259:262], ["<boi>", "<img>", "<not_special>"])
        # `special: true` is a CONTROL (matched only with parse_special); anything else an ordinary piece
        self.assertEqual(tok.token_types[259], 3)
        self.assertEqual(tok.token_types[261], 1)
        self.assertEqual(tok.special_ids["<img>"], 260)

    def test_a_piece_that_is_a_word_is_taken_whole(self):
        """ignore_merges: a pre-token piece that is itself in the vocabulary is that one token."""
        self.assertEqual(self.tok().encode("abc", parse_special=False), [258])
        self.assertNotEqual(self.tok(ignore_merges=False).encode("abc", parse_special=False), [258])

    def test_the_merges_of_the_file_run_lowest_rank_first(self):
        tok = self.tok(ignore_merges=False)
        self.assertEqual(tok.encode(" ab", parse_special=False), [257], "Ġ+a (rank 0), then Ġa+b (rank 1)")

    def test_specials_split_only_with_parse_special(self):
        tok = self.tok()
        head, tail = tok.encode("a", parse_special=True), tok.encode("b", parse_special=True)
        self.assertEqual(tok.encode("a<boi>b", parse_special=True), head + [259] + tail)
        self.assertNotIn(259, tok.encode("a<boi>b", parse_special=False), "a CONTROL is text without it")

    def test_the_folder_and_the_file_are_the_same_call(self):
        p = write_fixture(self.dir, fixture_doc())
        a, b = ST.Tokenizer.from_hf(p), ST.Tokenizer.from_hf(self.dir)
        self.assertEqual(a.encode("abc", parse_special=True), b.encode("abc", parse_special=True))

    def test_an_unknown_shape_is_refused(self):
        """A regex no known shape splits with is refused, rather than tokenized with another's pieces."""
        with self.assertRaises(ValueError):
            self.tok(pattern=r"\w+|[^\w\s]+")

    def test_the_ignore_merges_flag_picks_between_two_shapes_of_one_pattern(self):
        """`qwen2` and `hunyuanimage3` share their pattern; the file's own flag says which this is."""
        for im, want in ((True, ST.HUNYUANIMAGE3_PRE), (False, "qwen2")):
            self.assertEqual(self.tok(ignore_merges=im).pre, want)

    def test_a_normalizer_is_refused(self):
        with self.assertRaises(ValueError):
            self.tok(with_normalizer=True)

    def test_a_dropout_or_a_word_suffix_is_refused(self):
        for kwargs in ({"dropout": 0.1}, {"continuing": "__"}, {"end_of_word": "</s>"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self.tok(**kwargs)


REAL = os.environ.get("XESTRATA_HUNYUANIMAGE3_TOKENIZER")

_USER_TURN_IDS = [1502, 25, 362, 2579, 39935, 304, 279, 12056, 271, 72803, 25, 220]

# The ids HunyuanImage 3.0's `tokenizer.json` gives its own BPE (read by Hugging Face's
# `PreTrainedTokenizerFast(tokenizer_file=...)`, and the same through `AutoTokenizer` and
# `PreTrainedTokenizerFast.from_pretrained`, 2026-10-24).  Digits are one a token because the pattern keeps
# `\p{N}` alone; a CJK string that is in the vocabulary is one token because of ignore_merges.
HUNYUANIMAGE3_IDS = {
    "A red fox in the snow": [32, 2579, 39935, 304, 279, 12056],
    "Hello, world!": [9906, 11, 1917, 0],
    "numbers 12345 and 3.1415": [38478, 220, 16, 17, 18, 19, 20, 323, 220, 18, 13, 16, 19, 16, 20],
    "日本語のテキストとEnglish": [101694, 125726, 16144, 57933, 62903, 71634, 19732, 23392],
    "User: A red fox in the snow\n\nAssistant: ": _USER_TURN_IDS,
}


@unittest.skipUnless(REAL and Path(REAL).is_dir(),
                     "set $XESTRATA_HUNYUANIMAGE3_TOKENIZER to the folder with the model's files")
class RealHunyuanImage3(unittest.TestCase):
    def setUp(self):
        self.tok = ST.Tokenizer.from_hf(REAL)

    def test_the_classified_shape_and_the_size(self):
        self.assertEqual(self.tok.pre, ST.HUNYUANIMAGE3_PRE)
        self.assertTrue(self.tok.ignore_merges)
        self.assertEqual(len(self.tok.tokens), 130108)
        self.assertEqual(len(self.tok.ranks), 264306)

    def test_the_embedded_ids(self):
        for text, want in HUNYUANIMAGE3_IDS.items():
            self.assertEqual(self.tok.encode(text, parse_special=True), want, text)

    def test_the_system_prompt_of_the_record(self):
        p = Path(REAL) / "en_unified.txt"
        if not p.is_file():
            self.skipTest("no en_unified.txt next to the tokenizer")
        self.assertEqual(len(self.tok.encode(p.read_text(encoding="utf-8"), parse_special=True)), 1222)

    def test_the_markers_of_a_generated_sequence(self):
        ids = self.tok.encode("<boi><img_size_1024><img_ratio_16><timestep><guidance><timestep_r>"
                              "<img><img><eoi>", parse_special=True)
        self.assertEqual(ids, [128000, 128037, 128060, 128017, 130082, 130107, 128006, 128006, 128001])


if __name__ == "__main__":
    unittest.main()
