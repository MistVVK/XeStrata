# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The settings file is read the same way by setup and the server: a BOM is not a reason to lose the user's choices,
and setup never overwrites a file it cannot read.

    python -m unittest serve.test_settings
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402
from serve import settings  # noqa: E402


class SettingsFile(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "settings.json"

    def test_a_bom_is_read_by_both_readers(self):
        self.path.write_bytes(b'\xef\xbb\xbf{"data_dir": "/tmp/data", "search_engine": "duckduckgo"}')
        expected = {"data_dir": "/tmp/data", "search_engine": "duckduckgo"}
        self.assertEqual(settings.load(self.path, strict=False), expected)
        self.assertEqual(settings.load(self.path), expected)
        with mock.patch.object(setup, "settings_path", return_value=self.path):
            self.assertEqual(setup.load_settings(), expected)

    def test_setup_keeps_a_broken_file(self):
        broken = b'{"data_dir": "/tmp/data",}'
        self.path.write_bytes(broken)
        with mock.patch.object(setup, "settings_path", return_value=self.path), \
                mock.patch.object(setup, "warn") as warn:
            setup.save_settings({"data_dir": "/tmp/other"})
        self.assertEqual(self.path.read_bytes(), broken)
        warn.assert_called_once()
        self.assertIn("kept until the file is fixed", warn.call_args.args[0])

    def test_a_missing_file_is_written(self):
        with mock.patch.object(setup, "settings_path", return_value=self.path):
            setup.save_settings({"data_dir": "/tmp/data"})
        self.assertEqual(settings.load(self.path), {"data_dir": "/tmp/data"})


if __name__ == "__main__":
    unittest.main()
