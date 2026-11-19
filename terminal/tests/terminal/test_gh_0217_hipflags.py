"""0.2.17 GH-LIST #6255 / #6256 (iMilnb): every HIP build the Terminal starts names the chosen ROCm's own include folder before the system's
(`-isystem <root>/include` first in CMAKE_HIP_FLAGS), so a distro's older HIP headers in /usr/include never pair with the chosen ROCm's compiler.
The flag strings are read from the source (no ROCm, no compiler here)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]


class HipFlagsNameTheRootFirst(unittest.TestCase):
    def flags(self, relative: str) -> list[str]:
        text = (TERMINAL / relative).read_text(encoding="utf-8")
        return re.findall(r'CMAKE_HIP_FLAGS=([^"\n]*)"', text)

    def test_setup_py_has_the_engine_and_the_picture_reader_builds(self):
        found = self.flags("setup.py")
        self.assertEqual(len(found), 2, found)
        for flags in found:
            self.assertTrue(flags.startswith("-isystem {root}/include --rocm-path={root}"), flags)

    def test_the_image_tool_builds_the_same_way(self):
        found = self.flags("tools/image/qimage.py")
        self.assertEqual(len(found), 1, found)
        self.assertTrue(found[0].startswith("-isystem {rocm}/include --rocm-path={rocm}"), found[0])


if __name__ == "__main__":
    unittest.main()
