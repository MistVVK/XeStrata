# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent one-shot and recurring idle actions share the same scheduler."""
import threading
import unittest

from serve.idle_watch import Watch


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
