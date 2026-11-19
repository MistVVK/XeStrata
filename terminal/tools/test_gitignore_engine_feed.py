"""The folder setup's ready-made engine download goes to (engine-feed, beside the install: terminal/ in a checkout) is not
reported by git as 705 MB of untracked files (#4231).

    python -m unittest tools/test_gitignore_engine_feed.py -v
"""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")


@unittest.skipUnless(GIT and (ROOT / ".git").exists() and (ROOT / ".gitignore").is_file(), "needs a git checkout")
class EngineFeedIsIgnored(unittest.TestCase):
    def ignored(self, path: str) -> bool:
        r = subprocess.run([GIT, "-C", str(ROOT), "check-ignore", "-q", "--no-index", path], capture_output=True)
        return r.returncode == 0

    def test_the_download_folder_is_ignored(self):
        for path in ("terminal/engine-feed/strata-engine/strata.exe", "engine-feed/strata-engine/strata.exe"):
            with self.subTest(path=path):
                self.assertTrue(self.ignored(path))

    def test_the_engine_sources_are_not(self):
        self.assertFalse(self.ignored("engine/src/program/generate.cpp"))


if __name__ == "__main__":
    unittest.main()
