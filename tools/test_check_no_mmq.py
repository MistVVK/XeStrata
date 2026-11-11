"""P4-SPLIT: tools/check_no_mmq.py finds llama.cpp's MMQ names in a file and passes a file without them (the bytes look; nm and
cuobjdump are left out here, the ctest no_mmq_symbols runs them on a real -DSTRATA_MMQ=OFF build)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "check_no_mmq.py"


def run(*files):
    return subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", "", *files], capture_output=True, text=True)


class CheckNoMmq(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_a_kernel_name_is_found(self):
        p = self.write("eng", b"\x7fELF...._Z9mul_mat_qIL9ggml_type23ELi8ELb0EEvPKcPKiS4_...")
        r = run(p)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("mul_mat_q x1", r.stdout)

    def test_the_host_glue_is_found(self):
        p = self.write("eng.exe", b"MZ....ggml-cuda (strata mmq): %s: %s\n....")
        self.assertEqual(run(p).returncode, 1)

    def test_strata_helpers_in_the_mmq_namespace_are_not_markers(self):
        p = self.write("eng", b"_ZN6strata7prefill3mmq13gather_nativeEPKvS3_mS3_mPvS4_S4_S3_S4_m strata::prefill::mmq::iota prefill mmq: swiglu")
        r = run(p)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("no MMQ symbol in 1 of 1 files", r.stdout)

    def test_a_missing_file_is_a_usage_error(self):
        self.assertEqual(run(os.path.join(self.dir, "nothing")).returncode, 2)


if __name__ == "__main__":
    unittest.main()
