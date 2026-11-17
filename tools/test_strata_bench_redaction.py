# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The benchmark redaction (upstream 4be6a003a, 0481347ec): a credential of "none" is no key, the key is replaced as a
whole token in its own case, a key next to a hyphen is replaced, and redacting twice gives what once gives.

    python -m unittest tools.test_strata_bench_redaction
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_bench  # noqa: E402

KEY = "qatui-bench-key-0213-xyz"  # gitleaks:allow -- synthetic credential for redaction testing


class UsableKey(unittest.TestCase):
    def test_none_is_no_key(self):
        for value in (None, "", " ", "none", "NONE", "None"):
            self.assertEqual(strata_bench.usable_key(value), "")
        self.assertEqual(strata_bench.usable_key(" sk-abc "), "sk-abc")
        self.assertEqual(strata_bench.usable_key(["sk-a", "sk-b"]), "sk-a")


class KeyNextToHyphens(unittest.TestCase):
    def test_a_key_beside_hyphens_is_replaced(self):
        clean = strata_bench.redactor(key=KEY)
        self.assertEqual(clean(f"pack-{KEY}"), "pack-<key>")
        self.assertEqual(clean(f"{KEY}-1"), "<key>-1")
        self.assertEqual(clean(f"a-{KEY}-b and {KEY}"), "a-<key>-b and <key>")

    def test_a_key_inside_a_longer_token_is_not_the_key(self):
        key = "a1b2c3d4"
        clean = strata_bench.redactor(key=key)
        self.assertEqual(clean(f"sha{key}ff x_{key} {key}_y {key}9"), f"sha{key}ff x_{key} {key}_y {key}9")

    def test_a_short_key_beside_a_hyphen_is_replaced_but_a_longer_word_is_not_shortened(self):
        clean = strata_bench.redactor(key="pass")
        self.assertEqual(clean("the passage and the pass-word"), "the passage and the <key>-word")

    def test_a_key_of_none_is_not_redacted_as_a_word(self):
        clean = strata_bench.redactor(key="none")
        self.assertEqual(clean("none of the runners"), "none of the runners")

    def test_the_command_line_of_a_run_is_clean(self):
        clean = strata_bench.redactor(key=KEY)
        line = f"xestrata bench run --profile p --custom-pack /home/alice/pack-{KEY} --track-notes 'key {KEY}'"
        out = clean(line)
        self.assertNotIn(KEY, out)
        self.assertNotIn("alice", out)


class TwiceIsOnce(unittest.TestCase):
    def test_a_user_called_user_is_not_labelled_twice(self):
        with mock.patch.object(strata_bench.getpass, "getuser", return_value="user"):
            clean = strata_bench.redactor(key=KEY)
        for text in ("/home/alice/m", r"C:\Users\alice\m", "owner user wrote this", "/home/user/m"):
            once = clean(text)
            self.assertEqual(clean(once), once, text)
            self.assertNotIn("<<", clean(once))

    def test_a_host_called_host_is_not_labelled_twice(self):
        with mock.patch.object(strata_bench.socket, "gethostname", return_value="host"):
            clean = strata_bench.redactor()
        once = clean("running on host with 3 cards")
        self.assertEqual(clean(once), once)
        self.assertIn("<host>", once)

    def test_the_user_name_is_still_replaced_as_a_word(self):
        with mock.patch.object(strata_bench.getpass, "getuser", return_value="alice9"):
            clean = strata_bench.redactor()
        self.assertEqual(clean("by alice9 at alice90"), "by <user> at alice90")


if __name__ == "__main__":
    unittest.main()
