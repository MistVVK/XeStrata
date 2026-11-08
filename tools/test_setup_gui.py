# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's part of the launcher's window on Linux: what `xestrata gui` says is missing
(gui_runtime_problem), by the distribution and by what is installed.  No .NET is run.

    python -m unittest tools.test_setup_gui
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def listed(out: str):
    return mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, out, ""))


class GuiRuntime(unittest.TestCase):
    def setUp(self):
        self.which = mock.patch.object(setup.shutil, "which", return_value="/usr/bin/dotnet")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_dotnet_10_or_later(self):
        with listed("Microsoft.AspNetCore.App 10.0.14 [x]\nMicrosoft.NETCore.App 10.0.14 [x]\n"):
            self.assertTrue(setup.dotnet_has("runtime"))
        with listed("Microsoft.NETCore.App 11.0.1 [x]\n"):           # RollForward Major: a later one runs it
            self.assertTrue(setup.dotnet_has("runtime"))
        with listed("Microsoft.NETCore.App 8.0.20 [x]\n"):
            self.assertFalse(setup.dotnet_has("runtime"))
        with listed("10.0.114 [/usr/lib/dotnet/sdk]\n"):
            self.assertTrue(setup.dotnet_has("sdk"))

    def test_what_to_install_by_distribution(self):
        with mock.patch.object(setup, "PACKAGED", None), mock.patch.object(setup, "dotnet_has", return_value=False):
            with mock.patch.object(setup, "distro_id", return_value="ubuntu"):
                self.assertIn("sudo apt install dotnet-runtime-10.0", setup.gui_runtime_problem())
                self.assertIn("sudo apt install dotnet-sdk-10.0", setup.gui_runtime_problem(sdk=True))
            with mock.patch.object(setup, "distro_id", return_value="fedora"):
                self.assertIn("sudo dnf install dotnet-runtime-10.0", setup.gui_runtime_problem())
            with mock.patch.object(setup, "distro_id", return_value="debian"):
                why = setup.gui_runtime_problem()
                self.assertIn("no official package", why)
                self.assertIn("web page", why)

    def test_the_package_named_where_it_is_missing(self):
        with mock.patch.object(setup, "PACKAGED", {"engine": "/usr/lib/xestrata/engine"}), \
                mock.patch.object(setup, "GUI_PACKAGED", Path("/nonexistent/xestrata-gui")):
            self.assertIn("install xestrata-gui", setup.gui_runtime_problem())

    def test_none_with_the_runtime(self):
        with mock.patch.object(setup, "PACKAGED", None), mock.patch.object(setup, "dotnet_has", return_value=True):
            self.assertIsNone(setup.gui_runtime_problem(sdk=True))


if __name__ == "__main__":
    unittest.main()
