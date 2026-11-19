"""#5133: an engine that ended at the start after its own Windows commit-limit warning (#1607) is told in the error that the page file is the thing to change."""
import tempfile
import unittest
from pathlib import Path

from serve.server import start_failure_hint

WARNING = ("strata serve: WARNING the Windows commit limit is 127.7 GiB (RAM + page file) and 1.8 GiB of it is left, but this run plans 2.0 GiB of host "
           "allocations and keeps 4.0 GiB free: it can end at the limit with no message (exit code 0xC0000409, #1607) while RAM is still free. Raise the page file "
           "(System managed, or RAM-sized) or lower --conversation-cache-mib / --max-context / --prefill")


class CommitHint(unittest.TestCase):
    def hint(self, text, before=""):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text(before + text, encoding="utf-8")
            return start_failure_hint(str(log), len(before.encode("utf-8")))

    def test_a_start_that_ended_after_the_warning_names_the_page_file(self):
        got = self.hint("strata serve: Windows commit: 1.8 GiB left of a 127.7 GiB limit (RAM + page file)\n" + WARNING +
                        "\nstrata serve: prompt chunk auto: 8192 tokens, a 96-slot ring\nfatal   : Memory allocation failure\n")
        self.assertIn("memory:", got)
        self.assertIn("1.8 GiB more memory (RAM + page file)", got)
        self.assertIn("plans 2.0 GiB", got)
        self.assertIn('"System managed"', got)

    def test_a_start_without_the_warning_adds_nothing(self):
        self.assertEqual(self.hint("strata serve: prompt chunk auto: 8192 tokens\nfatal   : Memory allocation failure\n"), "")

    def test_an_earlier_starts_warning_is_not_this_starts(self):
        self.assertEqual(self.hint("strata serve: ready\n", before=WARNING + "\n"), "")

    def test_the_draft_head_hint_still_comes_first(self):
        got = self.hint("strata mtp: the draft head does not fit\n" + WARNING + "\n")
        self.assertTrue(got.startswith(". mtp: the draft head does not fit"))
        self.assertNotIn("page file", got)


if __name__ == "__main__":
    unittest.main()
