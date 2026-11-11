"""Lane NET: the callers of the update feed that run by themselves (a launch, setup's start) say nothing when the feed cannot be reached, in each of the
three ways: no network at all, a refused port, a server that never answers (tests/terminal/net_dead.py). The callers that need a fixture of their own
have their test beside their other tests:

  dsv41_engine.upgrade   tests/terminal/test_dsv41_upgrade.py   test_an_unreachable_feed_is_silent_and_the_installed_engine_stays
  shape_engine.ensure    tests/terminal/test_shape_engine.py     test_an_unreachable_feed_leaves_an_installed_engine_alone_and_says_nothing
  drafting_fetch         tests/terminal/test_drafting_tuned.py   test_a_feed_nobody_answers_for_is_silent_offline_refused_or_never_answering
  the daily check        tests/terminal/test_update_notice.py    test_dns_is_blocked / test_a_proxy_that_refuses / test_a_connection_that_never_answers

and here: feed_models.load (the model list at setup's start) and driver_floor.from_feed. A run that DOES explicit work (`strata update`) keeps its
sentence: feed.FeedUnreachable carries the approved words, and no caller decides by those words (only by the class)."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from strata_control import driver_floor, feed as feedmod, feed_models
from tests.terminal import net_dead


class QuietCallers(unittest.TestCase):
    def test_the_model_list_read_is_silent_and_keeps_what_it_has(self):
        for kind in net_dead.KINDS:
            with net_dead.dead_feed(kind) as dead, tempfile.TemporaryDirectory() as root:
                lines = []
                started = time.monotonic()
                loaded = feed_models.load(dead, fetch=True, root=Path(root), trusted={"k": b"0" * 32}, say=lines.append)
                self.assertEqual(list(loaded.models), [], kind)
                self.assertEqual(lines, [], kind)                    # (the waiting line comes after 5 s; these end sooner)
                self.assertLess(time.monotonic() - started, 5, kind)

    def test_the_driver_floor_read_is_silent(self):
        for kind in net_dead.KINDS:
            with net_dead.dead_feed(kind) as dead:
                self.assertIsNone(driver_floor.from_feed(dead, "linux-x86_64", seconds=5.0), kind)

    def test_an_explicit_update_check_names_the_host_and_is_the_class_the_quiet_callers_catch(self):
        for kind in net_dead.KINDS:
            with net_dead.dead_feed(kind) as dead:
                with self.assertRaises(feedmod.FeedError) as caught:
                    feedmod.fetch(dead, {"k": b"0" * 32}, timeout=0.6)
                self.assertIsInstance(caught.exception, feedmod.FeedUnreachable, kind)
                self.assertEqual(caught.exception.exit_code, 6)
                self.assertIn("so the update check could not run", str(caught.exception), kind)


if __name__ == "__main__":
    unittest.main()
