# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""A child process that writes non-UTF-8 bytes (a GBK display adapter name) must not kill setup.

    python -m unittest tools.test_setup_gbk_adapters
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GBK_LINE = b"Microsoft \xbb\xf9\xb1\xbe\xcf\xd4\xca\xbe\xca\xca\xc5\xe4\xc6\xf7|PCI/VEN_10DE&DEV_0A60&SUBSYS|0\r\n"


class GbkOutput(unittest.TestCase):
    def test_out_survives_bytes_that_are_not_utf8(self):
        code = "import sys; sys.stdout.buffer.write(%r)" % (GBK_LINE,)
        text = setup.out([sys.executable, "-X", "utf8", "-c", code])
        self.assertIsInstance(text, str)
        self.assertIn("Microsoft", text)
        self.assertIn("VEN_10DE", text)

    def test_run_survives_bytes_that_are_not_utf8(self):
        code = "import sys; sys.stdout.buffer.write(%r)" % (GBK_LINE,)
        r = setup.run([sys.executable, "-X", "utf8", "-c", code], quiet=True, check=False)
        self.assertIn("Microsoft", r.stdout)
        self.assertIn("VEN_10DE", r.stdout)


if __name__ == "__main__":
    unittest.main()
