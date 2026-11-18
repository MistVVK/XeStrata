"""P4b (0.2.16 retest): the next start says what happened to a last run that did not end normally (a systemd-oomd kill, or just an unclean end).

    python -m unittest serve.test_lastrun
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import lastrun


class JournalOut:
    def __init__(self, text="", rc=0):
        self.stdout, self.returncode = text, rc


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
class LastRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.said = []

    def begin(self, journal=lambda epoch: False, profile="launch-x"):
        return lastrun.begin(profile, state_base=self.base, say=self.said.append, journal=journal)

    def leave(self, profile="launch-x", pid=2 ** 22 + 12345, started=None, token="l:1"):
        """A marker of a server that is gone (a pid no process has)."""
        path = lastrun.marker_path(profile, self.base)
        path.write_text(json.dumps({"pid": pid, "token": token, "started": started or time.time() - 600, "profile": profile}), encoding="utf-8")
        return path

    def test_a_first_start_says_nothing_and_leaves_a_marker_that_end_removes(self):
        path = self.begin()
        self.assertEqual(self.said, [])
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["pid"], os.getpid())
        lastrun.end(path)
        self.assertFalse(path.exists())
        self.begin()
        self.assertEqual(self.said, [])                        # a normal end leaves nothing to report

    def test_a_marker_left_behind_says_the_last_run_did_not_end_normally(self):
        self.leave()
        self.begin()
        self.assertEqual(len(self.said), 1)
        self.assertEqual(self.said[0], lastrun.UNCLEAN_WORDS)
        self.assertNotIn("oomd", self.said[0])

    def test_an_oomd_line_in_the_journal_names_it_and_says_what_to_do(self):
        seen = []
        self.leave(started=1_700_000_000.0)
        self.begin(journal=lambda epoch: seen.append(epoch) or True)
        self.assertEqual(self.said, [lastrun.OOMD_WORDS])
        self.assertEqual(seen, [1_700_000_000.0])              # the journal is read from the old run's start
        for want in ("systemd-oomd", "20 seconds", "close other programs", "text console or over ssh"):
            self.assertIn(want, lastrun.OOMD_WORDS)

    def test_the_words_are_plain(self):
        for words in (lastrun.OOMD_WORDS, lastrun.UNCLEAN_WORDS):
            self.assertTrue(words.startswith("[strata] "))
            for jargon in ("SIGKILL", "cgroup", "PSI", "pid", "exit code", "-9"):
                self.assertNotIn(jargon, words)

    def test_a_marker_of_a_server_that_still_runs_is_left_alone(self):
        from strata_control import procs
        mine = lastrun.marker_path("launch-x", self.base)
        mine.write_text(json.dumps({"pid": os.getpid(), "token": procs.start_token(os.getpid()), "started": time.time(), "profile": "launch-x"}), encoding="utf-8")
        self.assertIsNone(self.begin())
        self.assertEqual(self.said, [])
        self.assertEqual(json.loads(mine.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_a_reused_pid_counts_as_gone(self):
        self.leave(pid=os.getpid(), token="l:not-this-process")
        self.begin()
        self.assertEqual(self.said, [lastrun.UNCLEAN_WORDS])

    def test_profiles_do_not_see_each_others_markers(self):
        self.leave(profile="launch-a")
        self.begin(profile="launch-b")
        self.assertEqual(self.said, [])

    def test_a_journal_that_cannot_be_read_is_just_the_unclean_end(self):
        def broken(*a, **k):
            raise OSError("no journalctl")
        self.assertFalse(lastrun.oomd_killed_since(1.0, run=broken))
        self.assertFalse(lastrun.oomd_killed_since(1.0, run=lambda *a, **k: JournalOut("", 1)))
        self.assertFalse(lastrun.oomd_killed_since(1.0, run=lambda *a, **k: JournalOut("nothing about oomd here\n")))
        self.assertTrue(lastrun.oomd_killed_since(1.0, run=lambda *a, **k: JournalOut("x.scope: systemd-oomd killed 6 process(es) in this unit.\n")))
        self.assertFalse(lastrun.oomd_killed_since(1.0, run=mock.Mock(side_effect=subprocess.TimeoutExpired("journalctl", 5))))

    def test_it_never_raises_and_the_marker_is_not_ours_to_remove_when_another_wrote_it(self):
        lastrun.end(None)
        path = self.leave()
        lastrun.end(path)                                       # a marker of another pid: kept
        self.assertTrue(path.exists())
        with mock.patch.object(lastrun, "marker_path", side_effect=RuntimeError("x")):
            self.assertIsNone(lastrun.begin("p", state_base=self.base))

    def test_only_a_real_engine_server_and_only_when_not_turned_off(self):
        self.assertTrue(lastrun.wants(["--engine", "strata", "--config", "c"]))
        self.assertTrue(lastrun.wants(["--engine=strata"]))
        self.assertFalse(lastrun.wants(["--engine", "mock"]))
        self.assertFalse(lastrun.wants([]))
        with mock.patch.dict(os.environ, {"STRATA_LASTRUN": "0"}):
            self.assertIsNone(self.begin())
        with mock.patch.object(lastrun.sys, "platform", "win32"):
            self.assertIsNone(self.begin())


if __name__ == "__main__":
    unittest.main()
