"""#5717: a build folder configured with one CUDA toolkit keeps that toolkit's library paths in its CMake cache
(CUDA_cudart_LIBRARY and the other CUDA_* / CUDAToolkit_* entries): after the toolkit moved, a new
-DCUDAToolkit_ROOT alone does not search them again and ninja stops on the old libcudart.so.  setup's configure
must notice a changed toolkit and drop those entries, and must leave an unchanged folder alone.

No real toolkit is needed: the cache and the toolkit folders are fakes, and the configure/build commands are
recorded instead of run."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import setup  # noqa: E402


def fake_toolkit(root: Path) -> Path:
    """A toolkit folder with the files the cache entries point at; returns its nvcc."""
    for sub in ("bin", "include", "targets/x86_64-linux/lib"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "bin" / "nvcc").write_text("", encoding="utf-8")
    (root / "targets/x86_64-linux/lib/libcudart.so").write_text("", encoding="utf-8")
    (root / "targets/x86_64-linux/lib/libcublas.so").write_text("", encoding="utf-8")
    return root / "bin" / "nvcc"


def write_cache(bdir: Path, root: Path, nvcc: str) -> None:
    """A CMakeCache.txt shaped like the reporter's: the cudart / cuBLAS paths and the toolkit dirs of `root`."""
    bdir.mkdir(parents=True, exist_ok=True)
    r = root.as_posix()
    (bdir / "CMakeCache.txt").write_text("\n".join([
        "# This is the CMakeCache file.",
        "CMAKE_GENERATOR:INTERNAL=Ninja",
        "CMAKE_BUILD_TYPE:STRING=Release",
        f"CMAKE_CUDA_COMPILER:STRING={nvcc}",
        f"CUDAToolkit_ROOT:UNINITIALIZED={r}",
        f"CUDAToolkit_BIN_DIR:PATH={r}/bin",
        f"CUDAToolkit_NVCC_EXECUTABLE:FILEPATH={r}/bin/nvcc",
        f"CUDA_cudart_LIBRARY:FILEPATH={r}/targets/x86_64-linux/lib/libcudart.so",
        f"CUDA_cublas_LIBRARY:FILEPATH={r}/targets/x86_64-linux/lib/libcublas.so",
        "CUDA_nvml_LIBRARY:FILEPATH=CUDA_nvml_LIBRARY-NOTFOUND",
        "STRATA_ENABLE_CUDA:BOOL=ON",
        ""]), encoding="utf-8")
    det = bdir / "CMakeFiles" / "3.31.6"
    det.mkdir(parents=True, exist_ok=True)
    (det / "CMakeCUDACompiler.cmake").write_text(f'set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "{r}")\n', encoding="utf-8")
    (det / "CMakeCXXCompiler.cmake").write_text("", encoding="utf-8")
    obj = bdir / "CMakeFiles" / "strata.dir"
    obj.mkdir(parents=True, exist_ok=True)
    (obj / "main.cpp.o").write_bytes(b"obj")


class Issue5717StaleToolkitCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.bdir = self.base / "build"
        self.runs = []
        for p in (mock.patch.object(setup, "ensure_build_tools", return_value=("cmake", "ninja")),
                  mock.patch.object(setup.acquisition, "require", return_value=None),
                  mock.patch.object(setup, "WIN", False),
                  mock.patch.object(setup, "run", side_effect=self._run)):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, cmd, *a, **k):
        self.runs.append([str(c) for c in cmd])
        return mock.Mock(returncode=0)

    def configure(self, nvcc: Path) -> list:
        self.runs.clear()
        setup.cmake_build(self.base, self.bdir, "strata",
                          ["-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}", *setup.toolkit_root_defs(nvcc)],
                          None, "build-strata.bat")
        return self.runs[0]

    def test_a_moved_toolkit_drops_the_cached_cuda_paths(self):
        """The reporter's case: the old toolkit's folder is gone, the new one is passed as CUDAToolkit_ROOT."""
        old, new = self.base / "cuda-13.4", self.base / "cuda-13.5"
        fake_toolkit(old)
        write_cache(self.bdir, old.resolve(), str(old / "bin" / "nvcc"))
        new_nvcc = fake_toolkit(new)
        import shutil
        shutil.rmtree(old)
        conf = self.configure(new_nvcc)
        self.assertIn("-UCUDA_*", conf)
        self.assertIn("-UCUDAToolkit_*", conf)
        # the -U comes before the new -DCUDAToolkit_ROOT, or CMake would remove the new value too
        root_def = next(i for i, a in enumerate(conf) if a.startswith("-DCUDAToolkit_ROOT="))
        self.assertLess(conf.index("-UCUDAToolkit_*"), root_def)
        self.assertLess(conf.index("-UCUDA_*"), root_def)
        # CMake detects the CUDA compiler again (its toolkit root is in that file); the objects stay for the rebuild
        self.assertEqual(list(self.bdir.glob("CMakeFiles/*/CMakeCUDACompiler.cmake")), [])
        self.assertTrue((self.bdir / "CMakeFiles" / "3.31.6" / "CMakeCXXCompiler.cmake").exists())
        self.assertTrue((self.bdir / "CMakeFiles" / "strata.dir" / "main.cpp.o").exists())
        self.assertTrue((self.bdir / "CMakeCache.txt").exists())

    def test_another_installed_toolkit_also_counts(self):
        """Both toolkits still installed, setup now builds with the other one: the cached paths are valid files
        of the wrong toolkit."""
        old, new = self.base / "cuda-12.9", self.base / "cuda-13.0"
        fake_toolkit(old)
        write_cache(self.bdir, old.resolve(), str(old / "bin" / "nvcc"))
        conf = self.configure(fake_toolkit(new))
        self.assertIn("-UCUDA_*", conf)
        self.assertIn("-UCUDAToolkit_*", conf)

    def test_an_unchanged_toolkit_changes_nothing(self):
        tk = self.base / "cuda-13.5"
        nvcc = fake_toolkit(tk)
        write_cache(self.bdir, tk.resolve(), str(nvcc))
        before = (self.bdir / "CMakeCache.txt").read_bytes()
        conf = self.configure(nvcc)
        self.assertFalse([a for a in conf if a.startswith("-U")], conf)
        self.assertEqual((self.bdir / "CMakeCache.txt").read_bytes(), before)
        self.assertTrue((self.bdir / "CMakeFiles" / "3.31.6" / "CMakeCUDACompiler.cmake").exists())

    def test_a_new_folder_and_a_cpu_folder_change_nothing(self):
        nvcc = fake_toolkit(self.base / "cuda-13.5")
        conf = self.configure(nvcc)                      # never configured: no cache
        self.assertFalse([a for a in conf if a.startswith("-U")], conf)
        self.bdir.mkdir(exist_ok=True)
        (self.bdir / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\nSTRATA_ENABLE_CUDA:BOOL=OFF\n",
                                                  encoding="utf-8")
        self.runs.clear()
        setup.cmake_build(self.base, self.bdir, "strata-vision", ["-DSTRATA_VISION_CUDA=OFF"], None, "x.bat")
        self.assertFalse([a for a in self.runs[0] if a.startswith("-U")], self.runs[0])


if __name__ == "__main__":
    unittest.main()
