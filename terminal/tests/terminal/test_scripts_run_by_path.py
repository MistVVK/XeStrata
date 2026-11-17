"""The programs a source checkout starts BY PATH (`python <file>`) import their packages: started from a folder that is not theirs, with no PYTHONPATH.

    python -m pytest tests/terminal/test_scripts_run_by_path.py

The 0.2.16 layout move put setup.py, strata_control and serve under terminal/ and the engine's own setup scripts under engine/. engine/metal/setup_mac.py,
engine/sycl/setup_intel.py and engine/sycl/serve/server_intel.py still put only engine/ on sys.path and stopped at their first import
(ModuleNotFoundError: No module named 'setup' / 'serve'), and the split of serve/server.py did the same to `python serve/server.py`. No suite ran any of
them as a script (their tests import them as modules). Each file runs with --help; a start that needs the person's PC (setup_intel.py asks for an
Intel card first) only has to get past its imports.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # terminal/
REPO = ROOT.parent
# (path, the run must end with exit code 0: a parser answers --help)
SCRIPTS = [
    ("terminal/setup.py", True),
    ("terminal/strata.py", True),
    ("terminal/serve/server.py", True),
    ("engine/metal/setup_mac.py", True),
    ("engine/sycl/serve/server_intel.py", True),
    ("engine/sycl/setup_intel.py", False),
]


class ScriptsRunByPath(unittest.TestCase):
    def test_each_script_gets_past_its_imports_from_a_foreign_folder(self):
        env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "PYTHONHOME")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for rel, needs_zero in SCRIPTS:
            with self.subTest(script=rel), tempfile.TemporaryDirectory() as d:
                path = REPO / rel
                self.assertTrue(path.is_file(), rel)
                r = subprocess.run([sys.executable, str(path), "--help"], cwd=d, env=env, capture_output=True, text=True, timeout=120,
                                   stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
                self.assertNotIn("ModuleNotFoundError", r.stderr, r.stderr[-600:])
                self.assertNotIn("ImportError", r.stderr, r.stderr[-600:])
                if needs_zero:
                    self.assertEqual(r.returncode, 0, r.stderr[-600:])


if __name__ == "__main__":
    unittest.main()
