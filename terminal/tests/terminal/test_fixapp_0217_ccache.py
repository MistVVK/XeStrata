"""#6172: a source build failed in Strata's own CUDA files when sccache was on PATH. The vendored ggml finds ccache or sccache and sets it as the
global compiler launcher, so every nvcc compile of Strata went through sccache and stopped at the fatbinary step. setup's configure now passes
-DGGML_CCACHE=OFF (before the caller's own definitions, so one of theirs still wins). The configure command is recorded, not run."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import setup  # noqa: E402


class ConfigureKeepsTheLauncherOff(unittest.TestCase):
    def configure(self, defs):
        runs = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup, "ensure_build_tools", return_value=("cmake", "ninja")), \
                mock.patch.object(setup.acquisition, "require", return_value=None), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "run", side_effect=lambda cmd, *a, **k: runs.append([str(c) for c in cmd]) or mock.Mock(returncode=0)):
            setup.cmake_build(Path(tmp), Path(tmp) / "build", "strata", defs, None, "x.bat")
        return runs[0]

    def test_the_configure_turns_ggml_s_compiler_launcher_off(self):
        self.assertIn("-DGGML_CCACHE=OFF", self.configure(["-DSTRATA_ENABLE_CUDA=ON"]))

    def test_the_vision_and_hip_configures_get_it_too(self):
        self.assertIn("-DGGML_CCACHE=OFF", self.configure(["-DSTRATA_VISION_CUDA=OFF"]))

    def test_a_definition_of_the_caller_comes_after_and_wins(self):
        conf = self.configure(["-DGGML_CCACHE=ON"])
        self.assertLess(conf.index("-DGGML_CCACHE=OFF"), conf.index("-DGGML_CCACHE=ON"))


if __name__ == "__main__":
    unittest.main()
