# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent one-shot and recurring idle actions share the same scheduler."""
import contextlib
import io
import threading
import time
import unittest

from serve.idle_watch import FAIL_LIMIT, Watch


class IdleWatchTest(unittest.TestCase):
    def test_one_shot_does_not_delay_other_actions(self):
        watch = Watch()
        started, release, other = threading.Event(), threading.Event(), threading.Event()

        def slow():
            started.set()
            release.wait(2)

        watch.add(0.01, slow)
        self.assertTrue(started.wait(1))
        try:
            watch.add(0.01, other.set)
            self.assertTrue(other.wait(1))
        finally:
            release.set()

    def test_repeats_do_not_overlap(self):
        watch = Watch()
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def action():
            calls.append(1)
            if len(calls) == 1:
                started.set()
                release.wait(1)
            else:
                finished.set()
                return False
            return None

        watch.add(0.01, action, repeat=True)
        self.assertTrue(started.wait(1))
        self.assertFalse(finished.wait(0.1))
        release.set()
        self.assertTrue(finished.wait(1))

    def test_a_repeating_failure_is_said_and_dropped(self):
        # an action that always raises: each round is said in the log, and the watch drops it after FAIL_LIMIT
        # instead of printing a traceback every interval forever; the other actions keep running
        watch = Watch()
        calls = []

        def failing():
            calls.append(1)
            raise OSError("the daemon's log is closed")

        other = threading.Event()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            watch.add(0.01, failing, repeat=True)
            watch.add(0.01, other.set)
            self.assertTrue(other.wait(2))
            deadline = time.monotonic() + 2
            while len(calls) < FAIL_LIMIT and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(calls), FAIL_LIMIT)
            time.sleep(0.1)
            self.assertEqual(len(calls), FAIL_LIMIT)   # not re-armed any more
            self.assertIn("the daemon's log is closed", log.getvalue())
            self.assertIn("failing", log.getvalue())

    def test_a_success_between_failures_resets_the_count(self):
        # three failures with a good round between them are not three failures in a row
        watch = Watch()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) in (1, 3, 5, 6, 7):
                raise OSError("busy")
            return None

        with contextlib.redirect_stdout(io.StringIO()):
            watch.add(0.01, flaky, repeat=True)
            deadline = time.monotonic() + 3
            while len(calls) < 7 and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)
        self.assertEqual(len(calls), 7)   # dropped at 5, 6, 7: the three in a row

    def test_a_lone_failure_keeps_the_repeat(self):
        # one round that raises is said and re-armed: an action that trips once still gets its next rounds
        watch = Watch()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("the engine was busy")
            return False

        with contextlib.redirect_stdout(io.StringIO()) as log:
            watch.add(0.01, flaky, repeat=True)
            deadline = time.monotonic() + 2
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(calls), 2)
        self.assertIn("the engine was busy", log.getvalue())
