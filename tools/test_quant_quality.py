# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""quant_quality: an ids file with a token outside the model's vocabulary fails at once, naming the segment and the id
(upstream b3e73dbfb).

    python -m unittest tools.test_quant_quality
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quant_quality as QQ  # noqa: E402


def ids_file(vocab_size=None):
    st = {"version": "v1", "seg_tokens": 3, "segments": [{"name": "code", "ids": [1, 2, 3]}]}
    if vocab_size is not None:
        st["vocab_size"] = vocab_size
    return st


class IdsVocab(unittest.TestCase):
    def test_an_id_outside_the_recorded_vocabulary_stops(self):
        with self.assertRaises(SystemExit) as e:
            QQ.check_ids_vocab({"segments": [{"name": "code", "ids": [1, 32001]}], "vocab_size": 32000}, {},
                               "set.ids.json")
        self.assertIn("segment 'code' has token id 32001", str(e.exception))
        self.assertIn("another tokenizer", str(e.exception))

    def test_the_config_s_tokenizer_says_the_size(self):
        tmp = Path(tempfile.mkdtemp(prefix="xestrata-qq-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        (tmp / "vocab.json").write_text(json.dumps({str(i): i for i in range(5)}))
        QQ.check_ids_vocab(ids_file(), {"tokenizer": str(tmp)}, "x")      # 3 < 5
        with self.assertRaises(SystemExit):
            QQ.check_ids_vocab({"segments": [{"name": "code", "ids": [5]}]}, {"tokenizer": str(tmp)}, "x")

    def test_nothing_is_checked_where_no_vocabulary_is_known(self):
        QQ.check_ids_vocab(ids_file(), {}, "x")


if __name__ == "__main__":
    unittest.main()
