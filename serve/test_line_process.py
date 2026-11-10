# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Line-engine shutdown and Windows application-control errors."""
import io
import subprocess
import unittest
from unittest import mock

from serve import line_process


class LineProcess(unittest.TestCase):
    def test_quit_then_terminate_then_kill(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.stdin = io.StringIO()
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 1), subprocess.TimeoutExpired("engine", 1), 0]
        self.assertTrue(line_process.shutdown(proc, 1, 1, 1, close_stdin=True))
        self.assertTrue(proc.stdin.closed)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 3)

    def test_windows_block_explained_for_image_engine(self):
        blocked = OSError("blocked")
        blocked.winerror = 4551
        with mock.patch.object(line_process.subprocess, "Popen", side_effect=blocked):
            with self.assertRaisesRegex(OSError, "Smart App Control"):
                line_process.popen("the XeStrata image engine", ["xestrata-image.exe"])
