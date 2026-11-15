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

    # The next-release item of 11-15: the device-function look was skipped in silence when the named cuobjdump was not there (the jammy
    # chroot had no CUDA 13 cuobjdump) or hit its time limit (Windows CUDA 12): the file read as checked.
    def tool(self, name, body):
        """A stand-in for nm / cuobjdump: a Python script, started through a .cmd on Windows and a #! line elsewhere."""
        script = self.write(name + ".py", body.encode())
        if os.name == "nt":
            return self.write(name + ".cmd", ('@"%s" "%s" %%*\r\n' % (sys.executable, script)).encode())
        p = self.write(name, ("#!%s\nexec(open(%r).read())\n" % (sys.executable, script)).encode())
        os.chmod(p, 0o755)
        return p

    def test_a_named_cuobjdump_that_is_not_there_is_not_a_pass(self):
        p = self.write("eng", b"\x7fELF clean")
        r = subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", os.path.join(self.dir, "no-cuobjdump"), p],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("cuobjdump NOT RUN (could not start", r.stdout)
        self.assertIn("1 of 1 files not fully checked", r.stdout)

    def test_a_cuobjdump_that_runs_out_of_time_or_reads_nothing_is_named(self):
        p = self.write("eng", b"\x7fELF clean")
        slow = self.tool("slow", "import time\ntime.sleep(5)\n")
        r = subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", slow, "--tool-timeout", "1", p],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("cuobjdump NOT RUN (timed out after 1 s)", r.stdout)
        old = self.tool("old", "import sys\nprint('cuobjdump info    : File version 9.0 not supported')\n")
        r = subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", old, p], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("cuobjdump NOT RUN (it read no device function", r.stdout)

    def test_cuobjdump_on_a_file_without_device_code_and_a_clean_device_look_pass(self):
        p = self.write("eng", b"\x7fELF clean")
        host = self.tool("host", "import sys\nsys.stderr.write(\"cuobjdump info    : File 'x' does not contain device code\\n\")\nsys.exit(255)\n")
        r = subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", host, p], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("cuobjdump (no device code) CLEAN", r.stdout)
        dev = self.tool("dev", "print('\\t\\tFunction : _Z13gather_nativePKv')\nprint('\\t\\tFunction : _Z9mul_mat_qILi1EEvPKc')\n")
        r = subprocess.run([sys.executable, str(TOOL), "--nm", "", "--cuobjdump", dev, p], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("cuobjdump (2 device functions) 1 HITS", r.stdout)


if __name__ == "__main__":
    unittest.main()
