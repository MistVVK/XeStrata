"""packaging/build_cuda_windows.py: the Windows NVIDIA engine package recipe (the release's configure line, the targets the build tree has).

    python -m unittest tools.test_build_cuda_windows -v
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
import build_cuda_windows as B  # noqa: E402

CUDA13 = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
CUDA12 = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")


class ConfigureLine(unittest.TestCase):
    def test_cuda13_is_the_release_line(self):
        # the 0.1.59 zips' line: 75;86;89;120 real + 120 PTX, portable, native experts, no tests (the K-quant and Q6_K options at their defaults)
        got = B.configure_args(Path("S"), Path("Bd"), "cuda13", CUDA13, [])
        self.assertEqual(got, ["-G", "Ninja", "-S", "S", "-B", "Bd", "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
                               "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_PORTABLE=ON",
                               "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;120-real;120-virtual",
                               "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe",
                               "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"])

    def test_cuda12_is_the_pascal_and_volta_build(self):
        got = B.configure_args(Path("S"), Path("Bd"), "cuda12", CUDA12, [])
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", got)
        # #5784: machine code for sm_120 too (a Blackwell card beside a Pascal / Volta one, or on a driver under 580.65, runs this package
        # and compiled its compute_89 PTX for minutes at every start); the PTX stays compute_89, the one sm_90 and other cards between use
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=60-real;61-real;70-real;75-real;80-real;86-real;120-real;89-real;89-virtual", got)
        self.assertEqual(B.VARIANTS["cuda12"][2], "576.02")      # the 0.1.59 cuda12 zip's min_driver
        self.assertEqual(B.VARIANTS["cuda13"][2], "580.65")

    def test_every_cuda12_recipe_names_the_same_archs(self):
        # the engine (Windows, Linux) and the Windows picture reader: package_engine.py refuses a reader that does not cover the engine
        want = ";".join(str(a) for a in B.VARIANTS["cuda12"][1])
        self.assertEqual(want, "60;61;70;75;80;86;120;89")
        linux = (ROOT / "packaging" / "build_cuda_linux.sh").read_text(encoding="utf-8")
        self.assertIn("cuda12) archs=${ARCHS:-%s};" % want, linux)
        bat = (ROOT / "packaging" / "build_vision_windows.bat").read_text(encoding="utf-8")
        self.assertIn('set "ARCHS=%s"' % want, bat)

    def test_extra_arguments_come_last_and_the_override_is_named(self):
        got = B.configure_args(Path("S"), Path("Bd"), "cuda13", CUDA13, ["-DSTRATA_WINDOW_MAX_T=16"], ggml_dir="C:\\ref\\llama.cpp")
        self.assertEqual(got[-2:], ["-DSTRATA_GGML_DIR=C:/ref/llama.cpp", "-DSTRATA_WINDOW_MAX_T=16"])
        self.assertEqual([a for a in got if "MMQ" in a], [])   # 0.2.16: no MMQ option exists any more

    def test_an_arch_override_keeps_ptx_for_the_last(self):
        self.assertEqual(B.parse_archs("120"), (120,))
        self.assertEqual(B.parse_archs("86; 89,120"), (86, 89, 120))
        got = B.configure_args(Path("S"), Path("Bd"), "cuda13", CUDA13, [], archs=(120,))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual", got)
        for bad in ("", "sm_120", "120;x"):
            with self.assertRaises(SystemExit):
                B.parse_archs(bad)

    def test_it_refuses_to_run_off_windows(self):
        if os.name == "nt":
            self.skipTest("Linux only")
        with self.assertRaises(SystemExit):
            B.main(["cuda13"])
        self.assertEqual(B.main(["cuda11"]), 2)


FAKE_CMAKE = """#!/bin/sh
case "$FAKE_MODE" in
  kq) exit 0 ;;
  nokq) echo "ninja: error: unknown target 'strata_kq'" >&2; exit 1 ;;
  *) echo "ninja: error: loading 'build.ninja': No such file or directory" >&2; exit 1 ;;
esac
"""


@unittest.skipIf(os.name == "nt", "POSIX stand-in cmake")
class Targets(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cmake = self.tmp / "cmake"
        self.cmake.write_text(FAKE_CMAKE, encoding="utf-8", newline="\n")
        self.cmake.chmod(self.cmake.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, mode):
        return B.engine_targets(str(self.cmake), self.tmp, env=dict(os.environ, FAKE_MODE=mode))

    def test_with_and_without_strata_kq(self):
        self.assertEqual(self.call("kq"), ["strata", "strata_kq"])
        self.assertEqual(self.call("nokq"), ["strata"])

    def test_any_other_failure_stops(self):
        with self.assertRaises(SystemExit):
            self.call("broken")

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"), "cmake and ninja")
    def test_real_ninja_trees(self):
        for with_kq in (True, False):
            src, build = self.tmp / ("s%d" % with_kq), self.tmp / ("b%d" % with_kq)
            src.mkdir()
            text = "cmake_minimum_required(VERSION 3.16)\nproject(t NONE)\nadd_custom_target(strata)\n" + ("add_custom_target(strata_kq)\n" if with_kq else "")
            (src / "CMakeLists.txt").write_text(text, encoding="utf-8")
            subprocess.run(["cmake", "-G", "Ninja", "-S", str(src), "-B", str(build)], check=True, capture_output=True, timeout=600)
            self.assertEqual(B.engine_targets("cmake", build), ["strata", "strata_kq"] if with_kq else ["strata"])


if __name__ == "__main__":
    unittest.main()
