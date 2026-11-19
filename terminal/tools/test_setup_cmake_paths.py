"""#5694: a source build on Windows hands CMake its paths with forward slashes.

START-HERE.bat compiled with `-DCMAKE_CUDA_COMPILER=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\nvcc.exe`; CMake 3.26 wrote that
string into CMakeCUDACompiler.cmake as it was and stopped with "Invalid character escape '\\P'".

    python -m unittest tools.test_setup_cmake_paths
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

NVCC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.EXE"
TOOLKIT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
NINJA = r"C:\Users\x y\AppData\Local\Strata\tools\ninja.exe"


class ForwardSlashes(unittest.TestCase):
    DEFS = [f"-DCMAKE_CUDA_COMPILER={NVCC}", f"-DCUDAToolkit_ROOT={TOOLKIT}", r"-DSTRATA_GGML_DIR=D:\src\llama.cpp",
            "-DCMAKE_CUDA_ARCHITECTURES=86;120", "-DSTRATA_ENABLE_CUDA=ON"]

    def test_windows_paths_in_definitions_use_forward_slashes(self):
        got = setup.forward_slash_defs(self.DEFS, win=True)
        self.assertEqual(got[0], "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.EXE")
        self.assertEqual(got[1], "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
        self.assertEqual(got[2], "-DSTRATA_GGML_DIR=D:/src/llama.cpp")
        self.assertEqual(got[3:], self.DEFS[3:])
        self.assertFalse(any("\\" in d for d in got))

    def test_other_systems_are_left_alone(self):
        self.assertEqual(setup.forward_slash_defs(self.DEFS, win=False), self.DEFS)

    def test_only_definitions_change(self):
        self.assertEqual(setup.forward_slash_defs([r"-UCUDAToolkit_ROOT", r"C:\x\y"], win=True), [r"-UCUDAToolkit_ROOT", r"C:\x\y"])


class BuildScript(unittest.TestCase):
    def test_the_configure_line_cmake_build_writes_has_no_back_slash_in_a_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build").mkdir()
            seen = []
            with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup.acquisition, "require"), \
                    mock.patch.object(setup, "ensure_build_tools", return_value=("cmake", NINJA)), \
                    mock.patch.object(setup, "fresh_if_other_generator"), \
                    mock.patch.object(setup, "run", side_effect=lambda *a, **k: seen.append(a[0]) or mock.Mock(returncode=0)):
                setup.cmake_build(root / "src", root / "build", "strata",
                                  [f"-DCMAKE_CUDA_COMPILER={NVCC}", f"-DCUDAToolkit_ROOT={TOOLKIT}"], r"C:\VS\vcvars64.bat", "b.bat")
            script = (root / "b.bat").read_text(encoding="utf-8")
        configure = next(line for line in script.splitlines() if " -G Ninja " in line)
        for raw in (NVCC, TOOLKIT, NINJA):
            self.assertNotIn(raw, configure)
        self.assertIn("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.EXE", configure)
        self.assertIn("C:/Users/x y/AppData/Local/Strata/tools/ninja.exe", configure)


if __name__ == "__main__":
    unittest.main()
