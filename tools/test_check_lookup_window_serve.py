# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The lookup-window serve gate check (tools/check_lookup_window_serve.py) against a fake engine: a clean answer with
drafts passes, a refused window fails, and --need-drafts fails when lookup windows did not run.  No GPU, no model.

    python -m unittest tools.test_check_lookup_window_serve
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_lookup_window_serve as C  # noqa: E402

FAKE = r'''import sys
mode = sys.argv[-1]
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("GEN"):
        print("T 5", flush=True)
        if mode == "refuse":
            print("ERR metal window: window size out of range", flush=True)
            break
        print("DONE 1 78 1.0 1.0 length %d %d 0 0 0 0 0 0.0 78 0"
              % ((0, 0) if mode == "nodrafts" else (3, 6)), flush=True)
    elif line.startswith("QUIT"):
        break
'''


class LookupWindowGate(unittest.TestCase):
    def run_fake(self, mode, **kw):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "fake.py"
            f.write_text(FAKE)
            real = subprocess.Popen
            try:   # the gate starts [exe, --serve, *args]; the fake is run as python fake.py --serve <mode>
                subprocess.Popen = lambda cmd, **k: real([cmd[0], str(f), *cmd[1:], mode], **k)
                return C.check(sys.executable, [], [1, 2, 3], start_timeout=20, request_timeout=20, **kw)
            finally:
                subprocess.Popen = real

    def test_pass_and_drafts(self):
        ok, lines = self.run_fake("ok", need_drafts=True)
        self.assertTrue(ok, lines)
        self.assertEqual(len(lines), 2)

    def test_refused_window_fails_the_gate(self):
        ok, lines = self.run_fake("refuse")
        self.assertFalse(ok)
        self.assertIn("window size out of range", lines[0])

    def test_no_drafts_fails_only_when_asked(self):
        ok, _ = self.run_fake("ok")
        self.assertTrue(ok)
        ok, lines = self.run_fake("nodrafts", need_drafts=True)
        self.assertFalse(ok)
        self.assertIn("drafts offered 0", lines[0])


if __name__ == "__main__":
    unittest.main()
