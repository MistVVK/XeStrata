# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The catalog's Unsloth v3.1 set UD-IQ1_M (upstream 9ed3c7789, 199fbc7a4): offered with its three pinned shards whose
names and sizes are the fingerprint file's, its own GGUF accepted, UD-IQ1_S still refused with the reason.

    python -m unittest tools.test_setup_unsloth_v31
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

FINGERPRINTS = json.loads((ROOT / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8"))
FAM = setup.FAMILIES["unsloth"]


class UDIQ1M(unittest.TestCase):
    def test_the_1_bit_choice_is_offered_with_three_shards(self):
        self.assertIn("UD-IQ1_M", setup.MODELS)
        self.assertIn("UD-IQ1_M", setup.family_sizes("unsloth"))
        self.assertEqual(setup.model_shards(FAM, "UD-IQ1_M"), 3)
        self.assertEqual(setup.MODELS["UD-IQ1_M"]["families"], ("unsloth",))

    def test_its_shard_names_and_sizes_are_the_fingerprint_s(self):
        fp = FINGERPRINTS["unsloth/UD-IQ1_M"]
        names = [setup.model_file(FAM, "UD-IQ1_M", i) for i in (1, 2, 3)]
        self.assertEqual([f.rsplit("/", 1)[-1] for f in fp["files"]], names)
        pins = sum(FAM["sha256"][n][0] for n in names)
        self.assertEqual(round(pins / 1e9, 1), setup.MODELS["UD-IQ1_M"]["download_gb"])

    def test_the_last_shard_is_the_ngram_table_shared_with_the_other_three_shard_sets(self):
        last = setup.model_file(FAM, "UD-IQ1_M", 3)
        for other in ("UD-Q2_0", "UD-IQ2_XS"):
            self.assertEqual(FAM["sha256"][last], FAM["sha256"][setup.model_file(FAM, other, 3)])

    def test_its_own_gguf_is_taken_and_ud_iq1_s_is_still_refused(self):
        self.assertIsNone(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"))
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"), "UD-IQ1_S")
        tmp = Path(tempfile.mkdtemp(prefix="xestrata-gguf-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        first = tmp / "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
        first.write_bytes(b"x")
        msg, hint = setup.gguf_dir_problem(tmp, first, FAM, "UD-Q4_K_XL")
        self.assertIn("cannot run", msg)
        self.assertIn("UD-IQ1_S mixes two expert formats", hint)


if __name__ == "__main__":
    unittest.main()
