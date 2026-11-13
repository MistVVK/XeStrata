# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A download that cannot reach its server says what to set behind a proxy (upstream 61b39f08b).

    python -m unittest tools.test_setup_proxy
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

NAMES = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


class ProxyHint(unittest.TestCase):
    def test_the_hint_names_the_variable_and_notices_one_already_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for n in NAMES:
                os.environ.pop(n, None)
            self.assertIn("set HTTPS_PROXY", setup.proxy_hint())
            os.environ["HTTPS_PROXY"] = "http://user:secret@proxy:3128"
            hint = setup.proxy_hint()
            self.assertIn("a proxy is set", hint)
            self.assertNotIn("secret", hint)


if __name__ == "__main__":
    unittest.main()
