#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The settings register's check (tools/settings_register.py): every name the engine and its entries read is in
data/settings-register.json, and every name in it is read.  A new setting that is not registered fails here, as a stale
entry does; running the tool fixes both.  upstream 89f4fa18."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import settings_register as R  # noqa: E402


class Register(unittest.TestCase):
    def test_the_register_matches_the_code(self):
        reads = R.scan()
        old = R.load_register()
        self.assertIsNotNone(old, "data/settings-register.json is missing; run tools/settings_register.py")
        self.assertEqual(sorted(reads), sorted(old["names"]))

    def test_known_settings_are_found(self):
        reads = R.scan()
        # one from each reader: the engine's env_get, a local helper, serve's envvars.get, and a shell script
        for name in ("PREFILL_TIMING", "PF_ALLOC_TRACE", "MANIFEST"):
            self.assertIn(name, reads, name)
        self.assertTrue(any(s.startswith("serve/") for sites in reads.values() for s in sites),
                        "no envvars.get site was scanned")

    def test_check_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "settings_register.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
