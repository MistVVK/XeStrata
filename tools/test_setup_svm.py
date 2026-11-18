# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""svm_broken_kernel against the kernel releases it must and must not catch (setup.py's check_svm_kernel and the
engine's svm_broken_kernel in src/core/device.cpp are the same rule): Ubuntu's 7.0.0 ABI 40 and later, nothing
else (upstream stable 7.2.y/7.3.y and Fedora 44/45's 7.2.15 and 7.3.3/7.3.4 carry the whole series, another series,
another distribution).

    python -m unittest tools.test_setup_svm
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class SvmBrokenKernel(unittest.TestCase):
    def test_it_catches_ubuntu_7_0_0_abi_40_and_later_only(self):
        for r in ("7.0.0-40-generic", "7.0.0-42-generic", "7.0.0-43-generic", "7.0.0-44-generic",
                  "7.0.0-44-lowlatency", "7.0.0-1015-oem", "7.0.0-43"):
            self.assertTrue(setup.svm_broken_kernel(r), r)
        for r in ("7.0.0-14-generic", "7.0.0-38-generic", "7.0.0-39-generic", "7.0.0", "7.0.0-rc2",
                  "7.0.0-generic", "7.1.0-rc3", "7.0.1-45-generic", "6.17.0-100-generic", "7.0.0-2-amd64",
                  "7.0.13-300.fc45.x86_64", "7.0.9-arch1-1", "7.2.15", "7.3.4",
                  "7.2.15-200.fc44", "7.3.3-200.fc44", "7.3.4-200.fc44", "7.2.15-300.fc45",
                  "7.3.3-300.fc45", "7.3.4-300.fc45", "", None):
            self.assertFalse(setup.svm_broken_kernel(r), r)


if __name__ == "__main__":
    unittest.main()
