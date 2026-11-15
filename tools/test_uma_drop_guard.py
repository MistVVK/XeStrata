"""The shared-memory page drop (STRATA_UMA_DROP_PAGES) runs and prints only in a Linux CUDA build.

src/program/generate.cpp gives the GGUF's cached expert pages back on a GPU that reports cudaDevAttrIntegrated (the DGX Spark).
The same attribute is true on AMD APUs under HIP and on every Mac under the Metal layer, the gfx906 layer has no such attribute,
and Windows has no page drop, so the whole block (the call and its log line) must sit inside the one guard that excludes them.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "src" / "program" / "generate.cpp"
GUARD = ("#if defined(__linux__) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)"
         "   // SPARK-UMA-DROP")


def guarded_span(lines):
    """(first, last) line index of the guarded block: the guard line and its matching #endif."""
    start = next(i for i, l in enumerate(lines) if l.strip() == GUARD.strip())
    depth = 0
    for i in range(start, len(lines)):
        s = lines[i].lstrip()
        if re.match(r"#\s*if", s):
            depth += 1
        elif re.match(r"#\s*endif", s):
            depth -= 1
            if depth == 0:
                return start, i
    raise AssertionError("the guard has no matching #endif")


class UmaDropGuard(unittest.TestCase):
    def setUp(self):
        self.lines = GEN.read_text(encoding="utf-8").splitlines()

    def test_guard_present_once(self):
        self.assertEqual(sum(1 for l in self.lines if l.strip() == GUARD.strip()), 1)

    def test_call_and_line_inside_the_guard(self):
        first, last = guarded_span(self.lines)
        for needle in ("drop_cached_pages(", "holds every expert, so the", "getenv(\"STRATA_UMA_DROP_PAGES\")"):
            at = [i for i, l in enumerate(self.lines) if needle in l]
            self.assertTrue(at, needle)
            for i in at:
                self.assertTrue(first < i < last, f"{needle!r} at line {i + 1} is outside the Linux-CUDA guard ({first + 1}-{last + 1})")

    def test_line_only_when_pages_were_given_back(self):
        first, last = guarded_span(self.lines)
        block = "\n".join(self.lines[first:last + 1])
        self.assertRegex(block, r"if \(freed > 0\)\s*\n\s*std::fprintf\(stderr, \"strata serve: this GPU shares")


if __name__ == "__main__":
    unittest.main()
