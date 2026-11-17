"""The Windows build scripts (.bat) name only paths that exist in the repository: tools/hip/build_windows.bat (the HIP engine and zip) and
packaging/build_vision_windows.bat (the CUDA picture reader).

    python -m unittest tools.test_build_windows_bats -v

Nothing here runs a script (they need Windows, a toolkit and hours). The tests read the script, work out what its SRC variable is from the
`for %%I in ("%~dp0..\\..") do set "SRC=%%~fI"` line, and check every `%SRC%\\...` path and every `cmake -S` folder against the checkout, on
every OS. The 0.2.16 layout move left build_windows.bat with SRC one level too low (terminal\\, no CMakeLists.txt) and nothing in a suite
reads that file's paths.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # terminal/
REPO = ROOT.parent
SCRIPTS = [REPO / "terminal" / "tools" / "hip" / "build_windows.bat", REPO / "tools" / "packaging" / "build_vision_windows.bat"]
MADE_BY_THE_SCRIPT = ("build", "dist", ".rocm", "out-")        # folders the script creates: not in a checkout
SRC_LINE = re.compile(r'for %%I in \("%~dp0((?:\.\.\\)*\.\.)"\) do set "SRC=%%~fI"')
SRC_PATH = re.compile(r'%SRC%((?:\\[A-Za-z0-9_.+-]+)+)')
CMAKE_S = re.compile(r'-S "%SRC%((?:\\[^"\\]+)*)"')


def read(script: Path) -> str:
    return script.read_text(encoding="utf-8", errors="replace")


def src_of(script: Path) -> Path:
    m = SRC_LINE.search(read(script))
    if not m:
        raise AssertionError("%s: no `set SRC` line of the known form" % script.name)
    return (script.parent / m.group(1).replace("\\", "/")).resolve()


class BatSourcePaths(unittest.TestCase):
    def test_the_scripts_are_where_the_tests_look(self):
        for s in SCRIPTS:
            self.assertTrue(s.is_file(), s)

    def test_src_is_the_repository_root(self):
        for s in SCRIPTS:
            with self.subTest(script=s.name):
                self.assertEqual(src_of(s), REPO.resolve())

    def test_every_path_under_src_exists(self):
        for s in SCRIPTS:
            src = src_of(s)
            found = 0
            for rel in SRC_PATH.findall(read(s)):
                parts = rel.strip("\\").split("\\")
                if parts[0].startswith(MADE_BY_THE_SCRIPT):
                    continue
                found += 1
                with self.subTest(script=s.name, path=rel):
                    self.assertTrue((src / "/".join(parts)).exists(), rel)
            self.assertGreaterEqual(found, 1, s.name)

    def test_cmake_is_pointed_at_folders_that_have_a_cmakelists(self):
        for s in SCRIPTS:
            src = src_of(s)
            found = CMAKE_S.findall(read(s))
            self.assertGreaterEqual(len(found), 1, s.name)
            for rel in found:
                folder = src / rel.strip("\\").replace("\\", "/") if rel.strip("\\") else src
                with self.subTest(script=s.name, source=rel or "%SRC%"):
                    self.assertTrue((folder / "CMakeLists.txt").is_file(), "cmake -S %%SRC%%%s: no CMakeLists.txt in %s" % (rel, folder))

    def test_the_hip_engine_is_configured_from_engine(self):
        text = read(SCRIPTS[0])
        self.assertIn('-S "%SRC%\\engine"', text)
        self.assertTrue((REPO / "engine" / "CMakeLists.txt").is_file())


if __name__ == "__main__":
    unittest.main()
