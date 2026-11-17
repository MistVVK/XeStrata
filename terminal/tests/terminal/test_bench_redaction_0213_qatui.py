"""Round 0213 QATUI: the benchmark redaction (found on l620 in a real results.json).

- A key written next to a hyphen (`--custom-pack /home/x/pack-<key>`) stayed in results.json: the whole-word rule treated the hyphen as part of the word. Now a hyphen
  is a boundary; letters, digits and the underscore still are not (a key inside `sha<key>ff` is not the key).
- `run.command` is redacted when it is built and the whole document again when it is written. For a user called "user" the second pass turned the label `<user>`
  into `<<user>>`. Redacting twice now gives what redacting once gives.

    python -m pytest tests/terminal/test_bench_redaction_0213_qatui.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli import bench_facts  # noqa: E402

KEY = "qatui-bench-key-0213-xyz"


class KeyNextToHyphens(unittest.TestCase):
    def test_a_key_beside_hyphens_is_replaced(self):
        clean = bench_facts.redactor(KEY)
        self.assertEqual(clean(f"pack-{KEY}"), "pack-<key>")
        self.assertEqual(clean(f"{KEY}-1"), "<key>-1")
        self.assertEqual(clean(f"a-{KEY}-b and {KEY}"), "a-<key>-b and <key>")

    def test_a_key_inside_a_longer_token_is_not_the_key(self):
        key = "a1b2c3d4"
        clean = bench_facts.redactor(key)
        self.assertEqual(clean(f"sha{key}ff x_{key} {key}_y {key}9"), f"sha{key}ff x_{key} {key}_y {key}9")

    def test_a_short_key_beside_a_hyphen_is_replaced_but_a_longer_word_is_not_shortened(self):
        clean = bench_facts.redactor("pass")
        self.assertEqual(clean("the passage and the pass-word"), "the passage and the <key>-word")

    def test_the_command_line_of_a_run_is_clean(self):
        clean = bench_facts.redactor(KEY)
        line = f"strata bench run --profile p --custom-pack /home/alice/pack-{KEY} --track-notes 'key {KEY}'"
        out = clean(line)
        self.assertNotIn(KEY, out)
        self.assertNotIn("alice", out)


class TwiceIsOnce(unittest.TestCase):
    def test_a_user_called_user_is_not_labelled_twice(self):
        with mock.patch.object(bench_facts.getpass, "getuser", return_value="user"):
            clean = bench_facts.redactor(KEY)
        for text in ("/home/alice/m", r"C:\Users\alice\m", "owner user wrote this", "/home/user/m"):
            once = clean(text)
            self.assertEqual(clean(once), once, text)
            self.assertNotIn("<<", clean(once))

    def test_a_host_called_host_is_not_labelled_twice(self):
        with mock.patch.object(bench_facts.socket, "gethostname", return_value="host"):
            clean = bench_facts.redactor("")
        once = clean("running on host with 3 cards")
        self.assertEqual(clean(once), once)
        self.assertIn("<host>", once)

    def test_the_user_name_is_still_replaced_as_a_word(self):
        with mock.patch.object(bench_facts.getpass, "getuser", return_value="alice9"):
            clean = bench_facts.redactor("")
        self.assertEqual(clean("by alice9 at alice90"), "by <user> at alice90")


if __name__ == "__main__":
    unittest.main()
