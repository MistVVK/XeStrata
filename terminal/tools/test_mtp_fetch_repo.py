"""mtp_fetch.py --repo (#6058): read the MTP tensors from another BF16 checkpoint's repository (a finetune that keeps them
in its own repo). The URL gets its closing slash, and a run on another repository is not "pinned": the pinned
revision's tensor hashes do not apply to it.

    python -m unittest tools/test_mtp_fetch_repo.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mtp_fetch as M  # noqa: E402


class RepoOption(unittest.TestCase):
    def run_main(self, *extra):
        old = M.REPO
        seen = {}
        try:
            with mock.patch.object(sys, "argv", ["mtp_fetch.py", "inventory", "--out", "unused", *extra]), \
                    mock.patch.object(M, "exit_with_parent"), \
                    mock.patch.object(M, "resolve_repo", lambda: seen.update(repo=M.REPO)), \
                    mock.patch.object(M, "inventory", lambda out: seen.update(inventory=out)):
                M.main()
            return seen, M.pinned()
        finally:
            M.REPO = old

    def test_another_repository_gets_its_closing_slash_and_is_not_pinned(self):
        for given in ("https://huggingface.co/u/finetune/resolve/main", "https://huggingface.co/u/finetune/resolve/main/"):
            with self.subTest(given=given):
                seen, pinned = self.run_main("--repo", given)
                self.assertEqual(seen["repo"], "https://huggingface.co/u/finetune/resolve/main/")
                self.assertFalse(pinned)

    def test_without_the_option_the_pinned_revision_is_read(self):
        seen, pinned = self.run_main()
        self.assertEqual(seen["repo"], M.PINNED)
        self.assertTrue(pinned)


if __name__ == "__main__":
    unittest.main()
