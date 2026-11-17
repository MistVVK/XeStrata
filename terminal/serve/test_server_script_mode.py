"""serve/server.py runs as a SCRIPT as well as a module (setup.py starts it as `python serve/server.py --engine strata --config ...`, the run-<model>
scripts do the same; Control, `strata launch` and the tuning start use `-m serve.server`).

    python -m pytest serve/test_server_script_mode.py

The 0.2.16 file split put its re-export block above the file's own `sys.path.insert(0, str(ROOT))` line, so a script start ended with
"ModuleNotFoundError: No module named 'serve'" while every module-mode test passed. These tests start the file by path from a folder that is
not the Terminal's, with no PYTHONPATH, and talk to a mock engine.
"""
from __future__ import annotations

import http.client
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parent / "server.py"


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class ScriptMode(unittest.TestCase):
    def test_help_runs_from_a_foreign_folder(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(SERVER), "--help"], cwd=d, env=clean_env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("usage:", r.stdout)
        self.assertNotIn("ModuleNotFoundError", r.stderr)

    def test_the_server_starts_as_a_script_and_answers(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen([sys.executable, str(SERVER), "--engine", "mock", "--host", "127.0.0.1", "--port", str(port)], cwd=d, env=clean_env(),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                status = None
                t0 = time.time()
                while time.time() - t0 < 45 and status is None:
                    if p.poll() is not None:
                        self.fail("the server exited with %s: %s" % (p.returncode, (p.stdout.read() or "")[-800:]))
                    try:
                        c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                        c.request("GET", "/health")
                        status = c.getresponse().status
                        c.close()
                    except OSError:
                        time.sleep(0.3)
                self.assertEqual(status, 200)
            finally:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=15)
                if p.stdout:
                    p.stdout.close()

    def test_setup_py_starts_the_server_by_that_path(self):
        # the two places setup.py starts it; if setup.py changes how it starts the server these tests must follow
        text = (SERVER.parents[1] / "setup.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count('str(ROOT / "serve" / "server.py")'), 2)


if __name__ == "__main__":
    unittest.main()
