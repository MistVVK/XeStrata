# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""--packages on Fedora 45: the same list as 44's except NVIDIA's CUDA toolkit, whose name NVIDIA's repository makes
per Fedora version (44: cuda-toolkit-13-4, 45: cuda-toolkit-13-5, as the packages build with), and the install
command setup prints for it.

    python -m unittest tools.test_setup_packages
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Fedora45(unittest.TestCase):
    def toolkits(self, ver: str) -> list:
        return setup.packages_for("fedora", {"nvidia"}, "contrib", ver)[0]

    def test_44_and_45_get_their_own_cuda_toolkit(self):
        self.assertIn("cuda-toolkit-13-4", self.toolkits("44"))
        self.assertIn("cuda-toolkit-13-5", self.toolkits("45"))

    def test_a_fedora_not_listed_takes_the_newest_known_toolkit(self):
        self.assertIn(setup.PACKAGES["fedora"]["nvidia"], self.toolkits("46"))

    def test_the_install_command_names_the_toolkit_of_the_fedora_asked_for(self):
        lines = []
        with mock.patch.object(setup, "say", lines.append):
            setup.print_packages("fedora45/nvidia,amd", "contrib")
        self.assertTrue(any("sudo dnf install" in line and "cuda-toolkit-13-5" in line for line in lines))
        self.assertTrue(any("rocm-hip-devel" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
