"""The picture reader's context retry (tools/vision/own/vit_gpu.cu, set_device_retrying) exists only in a Linux CUDA build.

A new CUDA context on a shared-memory GPU (DGX Spark) can fail with "out of memory" for a moment while the kernel has no free 2 MiB blocks; the reader
tries again up to 12 times, one second apart.  The same cudaDevAttrIntegrated is true on AMD APUs (HIP build, same source file) and on Macs (a different
reader), and Windows has no such case, so the retry loop sits inside one guard, and the pure rule (vit_gpu.h) says no for everything but integrated + out of memory.
"""
import re
import unittest
from pathlib import Path

OWN = Path(__file__).resolve().parent / "vision" / "own"
CU = OWN / "vit_gpu.cu"
GUARD = "#if defined(__linux__) && !defined(STRATA_VIT_HIP)   // SPARK-CTX-RETRY"


def guarded_span(lines):
    start = next(i for i, l in enumerate(lines) if l.strip() == GUARD)
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


class CtxRetryGuard(unittest.TestCase):
    def setUp(self):
        self.lines = CU.read_text(encoding="utf-8").splitlines()

    def test_guard_present_once(self):
        self.assertEqual(sum(1 for l in self.lines if l.strip() == GUARD), 1)

    def test_attribute_and_retry_inside_the_guard(self):
        first, last = guarded_span(self.lines)
        for needle in ("cudaDevAttrIntegrated", "cudaErrorMemoryAllocation", "vit_gpu_retry_context(", "sleep_for("):
            at = [i for i, l in enumerate(self.lines) if needle in l]
            self.assertTrue(at, needle)
            for i in at:
                self.assertTrue(first < i < last, f"{needle!r} at line {i + 1} is outside the Linux-CUDA guard ({first + 1}-{last + 1})")

    def test_the_context_is_made_through_the_retrying_call(self):
        text = "\n".join(self.lines)
        self.assertIn("set_device_retrying(device)", text)
        # the only cudaSetDevice left in load() is the one inside set_device_retrying, and the later ones in encode / alloc ignore their result or need no retry
        load = text[text.index("bool VitGpu::load("):text.index("bool VitGpu::encode(")]
        self.assertNotIn("cudaSetDevice(", load)


if __name__ == "__main__":
    unittest.main()
