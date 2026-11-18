# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Ubuntu kernels at or above the upstream fix are not said to need an update.

    python -m unittest serve.test_kernel_check
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import kernel_check  # noqa: E402


class UbuntuKernel(unittest.TestCase):
    def answer(self, sig: str):
        with mock.patch.object(kernel_check.Path, "read_text", return_value=sig):
            return kernel_check.ubuntu_fixed()

    def test_series_with_the_upstream_fix(self):
        for series in ("7.2.0", "7.3.0", "8.0.0"):
            self.assertIs(self.answer(f"Ubuntu {series}-38.38-generic 7.2.14"), True, series)

    def test_the_7_0_table_still_applies(self):
        self.assertIs(self.answer("Ubuntu 7.0.0-38.38-generic 7.0.14"), True)
        self.assertIs(self.answer("Ubuntu 7.0.0-37.10-generic 7.0.14"), False)

    def test_a_series_before_the_bug_is_not_affected(self):
        self.assertIs(self.answer("Ubuntu 6.8.0-40.40-generic 6.8.0"), True)


if __name__ == "__main__":
    unittest.main()
