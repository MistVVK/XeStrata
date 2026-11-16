"""#4312 / #4309: bin/strata and docker/entrypoint.sh ask the interpreter itself when `test -x` says no (a container runtime's seccomp profile
can deny access(X_OK) for a file the shell still runs: Docker 19.03.15's default profile in the reporter's case). The helper is taken out of
each script and run on its own: a directory and a file that does not run are refused, a runnable interpreter is taken. (The seccomp case
itself needs that Docker; not reproduced here.)"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def helper(script: str, name: str) -> str:
    text = (ROOT / script).read_text(encoding="utf-8")
    m = re.search(r"^%s\(\) \{\n.*?^\}\n" % re.escape(name), text, re.M | re.S)
    assert m, f"{name} is not in {script}"
    return m.group(0)


@unittest.skipIf(os.name == "nt" or not shutil.which("sh"), "POSIX shell scripts")
class Runnable(unittest.TestCase):
    def check(self, script: str, name: str, path: str) -> bool:
        code = helper(script, name) + f'{name} "$1"\n'
        return subprocess.run(["sh", "-c", code, "sh", path], capture_output=True).returncode == 0

    def test_both_scripts(self):
        for script, name in (("bin/strata", "is_runnable"), ("docker/entrypoint.sh", "python_usable")):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as d:
                folder = Path(d)
                self.assertFalse(self.check(script, name, str(folder)))                       # a directory is never a Python
                plain = folder / "python3"
                plain.write_text("not a program\n")
                self.assertFalse(self.check(script, name, str(plain)))                        # not runnable at all
                self.assertTrue(self.check(script, name, sys.executable))                     # a real interpreter
                self.assertFalse(self.check(script, name, str(folder / "missing")))


if __name__ == "__main__":
    unittest.main()
