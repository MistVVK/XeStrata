import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_lookup_window_serve as C

FAKE = r'''import sys
mode = sys.argv[-1]
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("GEN"):
        print("T 5", flush=True)
        if mode == "refuse":
            print("ERR metal window: window size out of range", flush=True)
            break
        print("DONE 1 78 1.0 1.0 length 3 6 0 0 0 0 0 0.0 78 0", flush=True)
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


if __name__ == "__main__":
    unittest.main()
