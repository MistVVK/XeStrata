"""Tests for the patch that puts Strata's VAE decoder into sd-cli (tools/image/sdcpp_vae/apply.py), on stand-in files
with the code it edits at the pinned stable-diffusion.cpp commit: the sources it copies (byte for byte), the CMake
target, the decode hook with its switch and its fallback (one warning that names the reason, then sd.cpp's own
decoder), that a second run changes nothing, that it works before or after the native runner's and the noise
generator's patches, and that a checkout at another commit is refused.

    python -m unittest tools.test_sdcpp_vae
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


V = load("strata_sdcpp_vae_apply_test", ROOT / "tools" / "image" / "sdcpp_vae" / "apply.py")
N = load("strata_sdcpp_native_apply_for_vae_test", ROOT / "tools" / "image" / "sdcpp_native" / "apply.py")
Z = load("strata_sdcpp_noise_apply_for_vae_test", ROOT / "tools" / "image" / "sdcpp_noise" / "apply.py")

ENGINE = """#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"
#include "core/util.h"

bool StableDiffusionGGML::init(const sd_ctx_params_t* sd_ctx_params) {
    rng = get_rng(sd_ctx_params->rng_type);
    if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
        sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
    } else {
        sampler_rng = rng;
    }
}

sd::Tensor<float> StableDiffusionGGML::decode_first_stage(const sd::Tensor<float>& x, bool decode_video) {
    auto latents                      = first_stage_model->diffusion_to_vae_latents(x);
    auto tiling_params                = first_stage_model->resolve_tiling_params(vae_tiling_params);
    return {};
}
"""
CMAKE = "add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})\ntarget_include_directories(${SD_LIB} PUBLIC . src include)\n"
BUILDERS = '#include "model/diffusion/minit2i.hpp"\n' + N.BUILDER_OLD


def fake_tree(d: Path) -> Path:
    sd = d / "stable-diffusion.cpp"
    (sd / "src" / "pipeline").mkdir(parents=True)
    (sd / "src" / "core").mkdir(parents=True)
    (sd / "src" / "model" / "diffusion").mkdir(parents=True)
    (sd / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
    (sd / "src" / "pipeline" / "diffusion_engine.cpp").write_text(ENGINE)
    (sd / "src" / "pipeline" / "model_builders.cpp").write_text(BUILDERS)
    (sd / "CMakeLists.txt").write_text(CMAKE)
    return sd


class Apply(unittest.TestCase):
    def test_copies_edits_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            sd = fake_tree(Path(t))
            V.apply(sd)
            for src, rel in V.FILES:
                self.assertEqual((sd / "src" / "strata_vae" / rel).read_bytes(), (ROOT.parent / "engine" / src).read_bytes(), rel)
            self.assertTrue((sd / "src" / "strata_vae" / "include" / "strata" / "hip_compat" / "cublas_v2.h").is_file())
            eng = (sd / "src" / "pipeline" / "diffusion_engine.cpp").read_text()
            cm = (sd / "CMakeLists.txt").read_text()
            self.assertIn('std::strstr(config_->params.model_args, "strata_vae=1")', eng)
            self.assertIn("version == VERSION_QWEN_IMAGE_2_1", eng)
            self.assertIn("#ifdef SD_STRATA_VAE", eng)
            # the fallback: the reason goes into the warning, then stable-diffusion.cpp's decoder runs (no return)
            self.assertEqual(eng.count('LOG_WARN("strata VAE decoder: %s; using stable-diffusion.cpp\'s decoder", '
                                       'strata_err.c_str());'), 2)
            # a non-finite picture: the warning says that sd.cpp's decoder (the same f16 operands) may fail the same way
            self.assertIn('if (strata_err.find("not finite") != std::string::npos)', eng)
            self.assertIn("using stable-diffusion.cpp's decoder, which may fail the same way", eng)
            cu = (ROOT.parent / "engine" / "src" / "image" / "vae" / "wan_vae.cu").read_text(encoding="utf-8")
            self.assertIn('"wan_vae: the picture has values that are not finite', cu)      # the reason the hook looks for
            self.assertLess(eng.index("return decoded;"), eng.rindex("LOG_WARN"))
            self.assertLess(eng.rindex("LOG_WARN"), eng.index("auto tiling_params"))
            self.assertIn("add_library(strata_vae STATIC", cm)
            self.assertIn("target_compile_definitions(${SD_LIB} PRIVATE SD_STRATA_VAE)", cm)
            self.assertNotIn("use_fast_math", cm)
            # HIP: the library defined before enable_language(HIP) needs the architectures too (p620, 11-10)
            self.assertIn('set_target_properties(strata_vae ${SD_LIB} PROPERTIES HIP_ARCHITECTURES '
                          '"${CMAKE_HIP_ARCHITECTURES}")', cm)
            self.assertLess(cm.index("if (SD_CUDA OR SD_HIPBLAS)"), cm.index(V.CMAKE_OLD))
            before = (eng, cm)
            V.apply(sd)
            self.assertEqual(((sd / "src" / "pipeline" / "diffusion_engine.cpp").read_text(),
                              (sd / "CMakeLists.txt").read_text()), before)

    def test_with_native_and_noise_in_either_order(self):
        results = []
        for order in ((V, N, Z), (N, Z, V)):
            with tempfile.TemporaryDirectory() as t:
                sd = fake_tree(Path(t))
                for m in order:
                    if m is N:
                        N.edit(sd / "src" / "pipeline" / "model_builders.cpp", N.INCLUDE_OLD, N.INCLUDE_NEW)
                        N.edit(sd / "src" / "pipeline" / "model_builders.cpp", N.BUILDER_OLD, N.BUILDER_NEW)
                        N.edit(sd / "CMakeLists.txt", N.CMAKE_OLD, N.CMAKE_NEW)
                    elif m is Z:
                        Z.apply(sd)
                    else:
                        V.apply(sd)
                eng = (sd / "src" / "pipeline" / "diffusion_engine.cpp").read_text()
                cm = (sd / "CMakeLists.txt").read_text()
                self.assertIn(V.MARK, eng)
                self.assertIn(Z.MARK, eng)
                self.assertIn("SD_QI_NATIVE", cm)
                self.assertIn("SD_STRATA_VAE", cm)
                results.append(eng.count(V.MARK))
        self.assertEqual(results[0], results[1])

    def test_other_commit_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            sd = fake_tree(Path(t))
            (sd / "src" / "pipeline" / "diffusion_engine.cpp").write_text(ENGINE.replace("diffusion_to_vae_latents(x)",
                                                                                          "to_vae(x)"))
            with self.assertRaises(SystemExit):
                V.apply(sd)


if __name__ == "__main__":
    unittest.main()
