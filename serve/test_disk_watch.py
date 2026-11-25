# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The disk watcher's levels, hysteresis, repeat backoff and quiet failures.

    python -m unittest serve.test_disk_watch
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import disk_watch  # noqa: E402

G = 1 << 30
TOTAL = 464 * G


class Clock:
    def __init__(self):
        self.wall = 1_700_000_000.0
        self.mono = 1000.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono


class Watched(unittest.TestCase):
    def setUp(self):
        self.lines: list[str] = []
        self.free = TOTAL // 2
        self.clock = Clock()
        patches = [
            mock.patch.object(disk_watch, "disk_usage",
                              side_effect=lambda p: SimpleNamespace(total=TOTAL, free=self.free)),
            mock.patch.object(disk_watch, "_device", return_value=1),
            mock.patch.object(disk_watch, "_mount_kind", return_value="disk"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def watcher(self, free: int) -> disk_watch.Watcher:
        self.free = free
        return disk_watch.Watcher([("data", Path("/tmp"))], say=self.lines.append, clock=self.clock,
                                  save=lambda snap: None)   # no test writes a real state file

    def test_thresholds(self):
        warn, danger = disk_watch.thresholds(TOTAL)
        self.assertEqual(warn, int(0.10 * TOTAL))
        self.assertEqual(danger, int(0.03 * TOTAL))
        small_warn, small_danger = disk_watch.thresholds(32 * G)
        self.assertEqual(small_warn, 8 * G)      # capped at 25 % of a small volume
        self.assertEqual(small_danger, int(0.10 * 32 * G))

    def test_config_without_images(self):
        self.assertEqual(disk_watch.image_paths({}), [])
        p = disk_watch.paths({}, "/tmp/xestrata-none.json")
        self.assertTrue(p)
        self.assertTrue(all(isinstance(path, Path) for _, path in p))

    def test_warning_hysteresis_and_recovery(self):
        w = self.watcher(int(0.09 * TOTAL))
        self.assertTrue(w.check())
        self.assertEqual(w.level, disk_watch.WARN)
        self.assertEqual(len(self.lines), 1)
        self.assertIn("nearly full", self.lines[0])
        # a little more space is still inside the hysteresis: no second line
        self.free = int(0.10 * TOTAL) + 2 * G
        w.check()
        self.assertEqual(w.level, disk_watch.WARN)
        self.assertEqual(len(self.lines), 1)
        self.free = TOTAL // 2
        w.check()
        self.assertEqual(w.level, disk_watch.OK)
        self.assertEqual(len(self.lines), 2)
        self.assertIn("recovered", self.lines[1])

    def test_warning_can_fall_straight_to_danger(self):
        w = self.watcher(int(0.09 * TOTAL))
        w.check()
        self.free = int(0.01 * TOTAL)
        w.check()
        self.assertEqual(w.level, disk_watch.DANGER)
        self.assertIn("critically full", self.lines[-1])

    def test_console_severity(self):
        self.assertEqual(disk_watch.severity("disk is nearly full: ..."), "WARNING")
        self.assertEqual(disk_watch.severity("disk is critically full: ..."), "ERROR")
        self.assertEqual(disk_watch.severity("disk space recovered: ..."), "INFO")

    def test_danger_repeats_with_backoff(self):
        w = self.watcher(int(0.01 * TOTAL))
        w.check()
        self.assertEqual(w.level, disk_watch.DANGER)
        self.assertEqual(len(self.lines), 1)
        for advance, count in ((59, 1), (1, 2), (119, 2), (1, 3), (239, 3), (1, 4), (299, 4), (1, 5)):
            self.clock.mono += advance
            w.check()
            self.assertEqual(len(self.lines), count, f"after {advance} s")
        self.assertIn("repeat 1", self.lines[1])
        self.assertIn("repeat 4", self.lines[-1])

    def test_leaving_danger_then_recovering(self):
        w = self.watcher(int(0.01 * TOTAL))
        w.check()
        self.free = int(0.055 * TOTAL)
        w.check()
        self.assertEqual(w.level, disk_watch.WARN)
        self.assertIn("no longer critically full", self.lines[-1])
        self.assertIn("still under the warning level", self.lines[-1])
        self.free = TOTAL // 2
        w.check()
        self.assertEqual(w.level, disk_watch.OK)
        self.assertIn("recovered", self.lines[-1])

    def test_a_big_change_resets_the_backoff(self):
        w = self.watcher(int(0.01 * TOTAL))
        w.check()
        self.clock.mono += 30
        self.free -= 3 * G                      # a real fall, not a nibble at the threshold
        w.check()
        self.assertEqual(len(self.lines), 1)   # no repeat yet: the interval restarted
        self.clock.mono += 59
        w.check()
        self.assertEqual(len(self.lines), 1)
        self.clock.mono += 1
        w.check()
        self.assertEqual(len(self.lines), 2)

    def test_a_failed_reading_is_unmeasured_and_silent(self):
        with mock.patch.object(disk_watch, "disk_usage", side_effect=OSError(5, "input/output error")):
            w = self.watcher(0)
            self.assertTrue(w.check())
            self.assertEqual(w.level, disk_watch.UNMEASURED)
            self.assertEqual(self.lines, [])
            self.assertFalse(w.view_data["measured"])

    def test_view_never_carries_a_path(self):
        w = self.watcher(int(0.01 * TOTAL))
        w.check()
        self.assertNotIn("path", w.view_data)
        self.assertNotIn("paths", w.view_data)
        self.assertEqual(w.view_data["volume"], "data")

    def test_a_broken_state_writer_does_not_stop_the_watch(self):
        def bad_save(snap):
            raise RuntimeError("the state folder is gone")

        self.free = int(0.09 * TOTAL)
        w = disk_watch.Watcher([("data", Path("/tmp"))], say=self.lines.append, clock=self.clock, save=bad_save)
        self.assertTrue(w.check())
        self.assertIn("nearly full", self.lines[0])

    def test_event_time_does_not_move_on_a_plain_sample(self):
        w = self.watcher(int(0.01 * TOTAL))
        w.check()
        first = w.view_data["event_at"]
        self.clock.mono += 3
        self.clock.wall += 3
        w.check()
        self.assertEqual(w.view_data["event_at"], first)   # no new toast for a steady full disk
        self.clock.mono += 57
        w.check()
        self.assertGreater(w.view_data["event_at"], first)  # the danger repeat is an event

    def test_public_health_line(self):
        view = {**disk_watch._EMPTY, "level": disk_watch.OK, "free_gib": 120.0, "measured": True}
        with mock.patch.object(disk_watch, "view", return_value=view):
            self.assertIsNone(disk_watch.warning_line())
        view["level"] = disk_watch.DANGER
        view["free_gib"] = 8.3
        with mock.patch.object(disk_watch, "view", return_value=view):
            line = disk_watch.warning_line()
        self.assertEqual(line, "Disk is critically full: 8.3 GiB free")
        self.assertNotIn("/tmp", line)


if __name__ == "__main__":
    unittest.main()
