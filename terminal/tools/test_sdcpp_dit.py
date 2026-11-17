"""Tests for the patch that puts Strata's DiT GEMMs into the native Qwen-Image-2.1 runner of sd-cli
(tools/image/sdcpp_dit/apply.py), on a stand-in checkout with the native runner applied by its own patch: the sources it
copies (byte for byte), the CMake target, the runner's switch (strata_dit_gemm=1, off without it) and its gemm() branch,
that a second run changes nothing, and that a checkout without the native runner is refused.

    python -m unittest tools.test_sdcpp_dit
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = load("strata_sdcpp_dit_apply_test", ROOT / "tools" / "image" / "sdcpp_dit" / "apply.py")
N = load("strata_sdcpp_native_apply_for_dit_test", ROOT / "tools" / "image" / "sdcpp_native" / "apply.py")

CMAKE = "add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})\ntarget_include_directories(${SD_LIB} PUBLIC . src include)\n"
BUILDERS = '#include "model/diffusion/minit2i.hpp"\n' + N.BUILDER_OLD


def fake_tree(d: Path, native: bool = True) -> Path:
    sd = d / "stable-diffusion.cpp"
    for p in ("src/pipeline", "src/model/diffusion", "ggml/src/ggml-cuda"):
        (sd / p).mkdir(parents=True)
    (sd / "src" / "pipeline" / "model_builders.cpp").write_text(BUILDERS)
    (sd / "CMakeLists.txt").write_text(CMAKE)
    if native:
        N.apply(sd)
    return sd


class DitApply(unittest.TestCase):
    def test_apply(self):
        with tempfile.TemporaryDirectory() as td:
            sd = fake_tree(Path(td))
            D.apply(sd)
            for src, rel in D.FILES:
                self.assertEqual((sd / "src" / "strata_dit" / rel).read_bytes(), (ROOT.parent / "engine" / src).read_bytes(), rel)
            self.assertTrue((sd / "src" / "strata_dit" / D.HIP_COMPAT / "cublas_v2.h").is_file())
            nat = (sd / "src" / "model" / "diffusion" / "qwen_image_2_1_native.hpp").read_text()
            self.assertEqual(nat.count('#include "strata/image/dit_gemm.hpp"'), 1)
            self.assertEqual(nat.count('std::strstr(model_args, "strata_dit_gemm=1")'), 1)
            self.assertEqual(nat.count("strata::image::dit_gemm("), 2)          # gemm() and the prelude / head helper
            for part in ("timestep MLP, modulation and the final scale", ": txt_in", ": the output head"):
                self.assertEqual(nat.count(part), 1, part)
            self.assertEqual(nat.count("} else\n#endif\n"), 3)                  # each falls back to the ggml block
            self.assertEqual(nat.count("bool strata_gemm = false;"), 1)
            # without the argument the old ggml_mul_mat path is still there, right after the branch
            self.assertIn("#endif\n            ggml_tensor* y = ggml_mul_mat(g.ctx, weight, rows(g.ctx, src, n));", nat)
            cm = (sd / "CMakeLists.txt").read_text()
            self.assertEqual(cm.count("add_library(strata_dit STATIC"), 2)          # CUDA and HIP branches
            self.assertEqual(cm.count("target_compile_definitions(${SD_LIB} PRIVATE SD_STRATA_DIT)"), 1)
            before = {p: p.read_bytes() for p in sd.rglob("*") if p.is_file()}
            D.apply(sd)
            after = {p: p.read_bytes() for p in sd.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_needs_native_runner(self):
        with tempfile.TemporaryDirectory() as td:
            sd = fake_tree(Path(td), native=False)
            with self.assertRaises(SystemExit):
                D.apply(sd)


if __name__ == "__main__":
    unittest.main()
