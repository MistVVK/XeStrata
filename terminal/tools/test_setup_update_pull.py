"""MAP-0215 Q6 (owner 11-16): `setup.py --update` gets the newest code itself, as UPDATE.bat / update.sh did; those two files go with the
0.2.16 layout.  With real git repositories in a temp folder: a clone that is behind is pulled and the update then runs in the NEW
setup.py (also when the pull moved setup.py into terminal/); a clone that is up to date goes on in this run; a zip download says how
to get the newest one; an installed Terminal and an update started by a pull do not pull.

    python -m unittest tools.test_setup_update_pull
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GIT = shutil.which("git")


def git(cwd: Path, *args) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run([GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


@unittest.skipUnless(GIT, "git is not installed")
class UpdatePull(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.origin = self.tmp / "origin"
        self.origin.mkdir()
        git(self.origin, "init", "-q", "-b", "main")
        (self.origin / "setup.py").write_text("# v1\n", encoding="utf-8")
        (self.origin / "CMakeLists.txt").write_text("project(strata VERSION 0.1.0)\n", encoding="utf-8")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-q", "-m", "v1")
        self.clone = self.tmp / "Strata"
        subprocess.run([GIT, "clone", "-q", str(self.origin), str(self.clone)], check=True, capture_output=True)

    def update(self, root: Path, file: Path, env=None):
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "__file__", str(file)), \
                mock.patch.object(setup, "CONTROL", False), mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, contextlib.redirect_stdout(out):
            os.environ.pop(setup.PULLED_ENV, None) if not env else None
            rc = setup.update_pull(["--update", "--yes"])
        return rc, call, out.getvalue()

    def test_a_clone_that_is_up_to_date_goes_on_in_this_run(self):
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertIsNone(rc)
        call.assert_not_called()
        self.assertIn("git pull", out)

    def test_u1_the_pull_is_one_summary_line_not_gits_change_list(self):
        """U1: a release that moves folders made git print about 15,000 lines of renames."""
        (self.origin / "src").mkdir()
        for i in range(60):
            (self.origin / "src" / f"f{i}.txt").write_text(f"file number {i}\n" * 20, encoding="utf-8")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-q", "-m", "many files")
        git(self.origin, "mv", "src", "lib")
        git(self.origin, "commit", "-q", "-m", "renamed")
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("\n"), 2, out)                               # "Getting the newest ..." + one summary line
        self.assertIn("Got the newest Strata (60 files changed).", out)
        for noise in ("rename ", "create mode", "Fast-forward", "Updating ", "|"):
            self.assertNotIn(noise, out)
        self.assertTrue((self.clone / "lib" / "f59.txt").is_file())

    def test_u1_up_to_date_says_so_in_one_line(self):
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertIn("Strata is up to date.", out)
        self.assertNotIn("Already up to date", out)

    def test_u1_a_failing_pull_still_shows_gits_own_words(self):
        (self.clone / "setup.py").write_text("# my edit\n", encoding="utf-8")
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 1)
        self.assertIn("setup.py", out)                                          # git names the file that stops the pull
        self.assertIn("overwritten", out)
        self.assertIn("git pull did not succeed (the reason is above)", out)

    def test_u1_a_remote_that_cannot_be_reached_says_why(self):
        git(self.clone, "remote", "set-url", "origin", str(self.tmp / "gone"))
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 1)
        self.assertIn("fatal", out.lower())

    def test_a_clone_that_is_behind_is_pulled_and_the_update_runs_in_the_new_setup(self):
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 0)
        self.assertEqual((self.clone / "setup.py").read_text(encoding="utf-8"), "# v2\n")
        argv, kw = call.call_args
        self.assertEqual(argv[0][1:], [str((self.clone / "setup.py").resolve()), "--update", "--yes"])
        self.assertEqual(kw["env"][setup.PULLED_ENV], "1", "the new setup.py does not pull again")

    def test_the_pull_to_the_0216_layout_runs_terminal_setup_py(self):
        git(self.origin, "mv", "setup.py", "terminal-setup.py")
        (self.origin / "terminal").mkdir()
        git(self.origin, "mv", "terminal-setup.py", "terminal/setup.py")
        (self.origin / "engine").mkdir()
        git(self.origin, "mv", "CMakeLists.txt", "engine/CMakeLists.txt")
        git(self.origin, "commit", "-q", "-m", "0.2.16 layout")
        rc, call, _ = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 0)
        self.assertFalse((self.clone / "setup.py").exists())
        self.assertEqual(call.call_args[0][0][1], str(self.clone / "terminal" / "setup.py"))
        self.assertEqual(call.call_args[1]["cwd"], str(self.clone / "terminal"))

    def test_in_the_0216_layout_the_clone_is_the_folder_above_terminal(self):
        (self.origin / "terminal").mkdir()
        git(self.origin, "mv", "setup.py", "terminal/setup.py")
        (self.origin / "engine").mkdir()
        git(self.origin, "mv", "CMakeLists.txt", "engine/CMakeLists.txt")
        git(self.origin, "commit", "-q", "-m", "0.2.16 layout")
        git(self.clone, "pull", "-q", "--ff-only")
        with mock.patch.object(setup, "ROOT", self.clone / "terminal"):
            self.assertEqual(setup.source_checkout(), self.clone)
        (self.origin / "terminal" / "setup.py").write_text("# v3\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v3")
        rc, call, _ = self.update(self.clone / "terminal", self.clone / "terminal" / "setup.py")
        self.assertEqual(rc, 0)
        self.assertEqual((self.clone / "terminal" / "setup.py").read_text(encoding="utf-8"), "# v3\n")

    def test_a_failed_pull_stops_the_update(self):
        (self.clone / "setup.py").write_text("# my edit\n", encoding="utf-8")
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 1)
        call.assert_not_called()
        self.assertIn("git status", out)

    def test_a_clone_without_origin_main_is_not_taken_for_the_old_history(self):
        """#4390: a failed pull on a clone that has no origin/main (the remote branch is gone from its refs) has no old history to
        move off: the answer is the general one (git pull did not succeed), not the move to "the new history" that cannot work."""
        git(self.clone, "update-ref", "-d", "refs/remotes/origin/main")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = setup.pull_onto_new_history(GIT, self.clone)
        self.assertEqual(rc, 1)
        self.assertIn("git pull did not succeed", out.getvalue())
        self.assertNotIn("old history", out.getvalue())
        self.assertEqual(git(self.clone, "branch", "--list", "pre-cleanup-backup*"), "")      # no backup branch was made

    def test_a_clone_on_unrelated_history_is_still_moved(self):
        """The case the function is for (#1276) is unchanged: origin/main exists and shares no commit with HEAD."""
        orphan = self.tmp / "orphan"
        orphan.mkdir()
        git(orphan, "init", "-q", "-b", "main")
        (orphan / "setup.py").write_text("# unrelated\n", encoding="utf-8")
        git(orphan, "add", "-A")
        git(orphan, "commit", "-q", "-m", "unrelated")
        git(self.clone, "fetch", "-q", str(orphan), "main:refs/remotes/origin/main", "--force")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = setup.pull_onto_new_history(GIT, self.clone)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("Moved to the new history", out.getvalue())

    def test_a_clone_on_a_fixed_release_says_so_and_stops(self):
        """P6: a clone on a tag or a commit used to get "Files you changed here can stop it: git status lists them"."""
        git(self.origin, "tag", "v1")
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        git(self.clone, "fetch", "-q", "--tags")
        git(self.clone, "checkout", "-q", "v1")
        for windows, again in ((False, "./setup.sh --update"), (True, "START-HERE.bat --update")):
            with mock.patch.object(setup, "WIN", windows):
                rc, call, out = self.update(self.clone, self.clone / "setup.py")
            self.assertEqual(rc, 1)
            call.assert_not_called()
            self.assertIn("This folder is on a fixed release, not a branch", out)
            self.assertIn("git checkout main", out)
            self.assertIn(again, out)
            self.assertNotIn("git status", out)
            self.assertNotIn("Files you changed", out)
            self.assertNotIn("Getting the newest", out)                         # nothing was tried
        self.assertEqual((self.clone / "setup.py").read_text(encoding="utf-8"), "# v1\n")

    def test_a_detached_commit_is_the_same_and_a_branch_is_not(self):
        sha = git(self.clone, "rev-parse", "HEAD")
        self.assertFalse(setup.detached_head(GIT, self.clone))
        git(self.clone, "checkout", "-q", "--detach", sha)
        self.assertTrue(setup.detached_head(GIT, self.clone))
        self.assertFalse(setup.detached_head(GIT, self.tmp))                    # not a repository: not claimed as a fixed release

    def test_after_git_checkout_main_the_update_goes_on(self):
        git(self.clone, "checkout", "-q", "--detach", "HEAD")
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        git(self.clone, "checkout", "-q", "main")
        rc, call, out = self.update(self.clone, self.clone / "setup.py")
        self.assertEqual(rc, 0)
        self.assertEqual((self.clone / "setup.py").read_text(encoding="utf-8"), "# v2\n")

    def test_an_update_started_by_a_pull_or_by_the_app_does_not_pull(self):
        (self.origin / "setup.py").write_text("# v2\n", encoding="utf-8")
        git(self.origin, "commit", "-q", "-am", "v2")
        rc, call, out = self.update(self.clone, self.clone / "setup.py", env={setup.PULLED_ENV: "1"})
        self.assertIsNone(rc)
        self.assertEqual((self.clone / "setup.py").read_text(encoding="utf-8"), "# v1\n")
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "ROOT", self.clone):
            self.assertIsNone(setup.update_pull(["--update"]))
        self.assertEqual((self.clone / "setup.py").read_text(encoding="utf-8"), "# v1\n")

    def test_a_zip_download_says_how_to_get_the_newest_and_goes_on(self):
        zipped = self.tmp / "zip"
        zipped.mkdir()
        (zipped / "CMakeLists.txt").write_text("project(strata VERSION 0.1.0)\n", encoding="utf-8")
        rc, call, out = self.update(zipped, zipped / "setup.py")
        self.assertIsNone(rc)
        self.assertIn("archive/refs/heads/main.zip", out)

    def test_an_installed_terminal_neither_pulls_nor_talks(self):
        installed = self.tmp / "install" / "terminal" / "0.2.16"
        installed.mkdir(parents=True)
        rc, call, out = self.update(installed, installed / "setup.py")
        self.assertIsNone(rc)
        self.assertEqual(out, "")


class MainPullsFirst(unittest.TestCase):
    def test_main_runs_the_pull_before_anything_else_on_update(self):
        with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                mock.patch.object(setup, "update_pull", return_value=7) as pull, \
                mock.patch.object(setup, "installed_configs", side_effect=AssertionError("ran after the pull")):
            self.assertEqual(setup.main(), 7)
        pull.assert_called_once_with(["--update"])


if __name__ == "__main__":
    unittest.main()
