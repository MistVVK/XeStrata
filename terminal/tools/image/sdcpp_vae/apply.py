"""tools/image/sdcpp_vae/apply.py - put Strata's own VAE decoder (src/image/vae, CUDA and HIP) into a
stable-diffusion.cpp checkout at the pinned commit (tools/image/qimage.py SD_CPP_COMMIT).  Idempotent; sd-cli keeps
stable-diffusion.cpp's decoder unless it is started with --model-args strata_vae=1, and only CUDA / HIP builds have it.

    python tools/image/sdcpp_vae/apply.py third_party/stable-diffusion.cpp

What it does:
  * copies the decoder (src/image/vae/wan_vae.cu, safetensors_lite.hpp, convert.hpp) and the headers it uses
    (include/strata/image/wan_vae.hpp, include/strata/kernels/{f16,bf16}_bits.hpp, include/strata/hip_compat/*) into
    src/strata_vae/ of the checkout, unchanged;
  * CMakeLists.txt: a static library strata_vae of our own (CUDA or HIP language, its own flags: none of ggml's
    -use_fast_math) linked into the stable-diffusion library, with the cuBLAS / hipBLAS that ggml links already;
  * src/pipeline/diffusion_engine.cpp: in decode_first_stage, for Qwen-Image-2.1 pictures with strata_vae=1, our
    decoder takes the VAE-space latent and returns the picture in sd.cpp's 0-1 range; sd.cpp's latent statistics, its
    8-bit conversion and its PNG writer stay as they are.  If our decoder cannot open the VAE file or fails, sd-cli logs
    one warning and decodes with stable-diffusion.cpp's decoder."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if (HERE.parents[2] / "src").is_dir() else HERE.parents[3] / "engine"   # the engine's sources (../engine in a 0.2.16 checkout)
MARK = "// strata: own VAE decoder"

# (repo path, path under src/strata_vae/)
FILES = [
    ("src/image/vae/wan_vae.cu", "wan_vae.cu"),
    ("src/image/vae/safetensors_lite.hpp", "safetensors_lite.hpp"),
    ("src/image/vae/convert.hpp", "convert.hpp"),
    ("include/strata/image/wan_vae.hpp", "include/strata/image/wan_vae.hpp"),
    ("include/strata/kernels/f16_bits.hpp", "include/strata/kernels/f16_bits.hpp"),
    ("include/strata/kernels/bf16_bits.hpp", "include/strata/kernels/bf16_bits.hpp"),
]
HIP_COMPAT = "include/strata/hip_compat"

INCLUDE_OLD = '#include "core/rng_mt19937.hpp"\n'
INCLUDE_NEW = INCLUDE_OLD + ("#ifdef SD_STRATA_VAE\n#include <cstring>\n#include \"strata/image/wan_vae.hpp\"  " + MARK +
                             "\n#endif\n")
DECODE_OLD = """    auto latents                      = first_stage_model->diffusion_to_vae_latents(x);
    auto tiling_params                = first_stage_model->resolve_tiling_params(vae_tiling_params);
"""
DECODE_NEW = """    auto latents                      = first_stage_model->diffusion_to_vae_latents(x);
#ifdef SD_STRATA_VAE
    """ + MARK + """ (--model-args strata_vae=1): the one-pass picture, f32 accumulation
    if (version == VERSION_QWEN_IMAGE_2_1 && !decode_video && config_->params.model_args != nullptr &&
        std::strstr(config_->params.model_args, "strata_vae=1") != nullptr && latents.dim() == 4 &&
        latents.shape()[2] == 64 && latents.shape()[3] == 1) {
        static strata::image::WanVae* strata_vae = nullptr;
        static bool strata_vae_failed            = false;
        std::string strata_err;
        if (strata_vae == nullptr && !strata_vae_failed) {
            strata_vae        = strata::image::wan_vae_open(SAFE_STR(config_->params.vae_path), -1, &strata_err,
                                                            n_threads);   // sd-cli's own thread setting
            strata_vae_failed = strata_vae == nullptr;
            if (strata_vae_failed) {
                LOG_WARN("strata VAE decoder: %s; using stable-diffusion.cpp's decoder", strata_err.c_str());
            }
        }
        if (strata_vae != nullptr) {
            const int lw              = static_cast<int>(latents.shape()[0]);
            const int lh              = static_cast<int>(latents.shape()[1]);
            sd::Tensor<float> decoded = sd::zeros<float>({static_cast<int64_t>(lw) * 16, static_cast<int64_t>(lh) * 16, 4, 1});
            strata::image::WanVaeStats strata_stats;
            strata_err = strata::image::wan_vae_decode(strata_vae, latents.data(), lh, lw, decoded.data(), &strata_stats);
            if (strata_err.empty()) {
                for (int64_t i = 0; i < decoded.numel(); ++i) {   // VAE::scale_tensor_to_0_1
                    float value = (decoded[i] + 1.0f) * 0.5f;
                    decoded[i]  = std::max(0.0f, std::min(1.0f, value));
                }
                LOG_INFO("strata VAE decoder: %.2fs, peak %.0f MiB", strata_stats.decode_ms / 1000.0,
                         strata_stats.peak_bytes / 1048576.0);
                return decoded;
            }
            if (strata_err.find("not finite") != std::string::npos) {   // sd.cpp's decoder has the same f16 operands
                LOG_WARN("strata VAE decoder: %s; using stable-diffusion.cpp's decoder, which may fail the same way",
                         strata_err.c_str());
            } else {
                LOG_WARN("strata VAE decoder: %s; using stable-diffusion.cpp's decoder", strata_err.c_str());
            }
        }
    }
#endif
    auto tiling_params                = first_stage_model->resolve_tiling_params(vae_tiling_params);
"""
CMAKE_OLD = "target_include_directories(${SD_LIB} PUBLIC . src include)\n"
CMAKE_NEW = """if (SD_CUDA OR SD_HIPBLAS)  # """ + MARK.lstrip("/ ") + """: src/strata_vae (tools/image/sdcpp_vae/apply.py)
    set(STRATA_VAE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/strata_vae)
    if (SD_CUDA)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(STRATA_VAE_ARCH native)
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_library(strata_vae STATIC ${STRATA_VAE_DIR}/wan_vae.cu)
        if (STRATA_VAE_ARCH)
            set_target_properties(strata_vae PROPERTIES CUDA_ARCHITECTURES ${STRATA_VAE_ARCH})
        endif()
        if (GGML_STATIC AND NOT WIN32)   # the same cuBLAS / runtime as ggml-cuda
            target_link_libraries(strata_vae PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        elseif (GGML_STATIC)
            target_link_libraries(strata_vae PRIVATE CUDA::cudart_static CUDA::cublas)
        else()
            target_link_libraries(strata_vae PRIVATE CUDA::cudart CUDA::cublas)
        endif()
    else()
        if (AMDGPU_TARGETS AND NOT GPU_TARGETS)
            set(GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
        endif()
        enable_language(HIP)
        find_package(hip REQUIRED)
        find_package(hipblas REQUIRED)
        set_source_files_properties(${STRATA_VAE_DIR}/wan_vae.cu PROPERTIES LANGUAGE HIP)
        add_library(strata_vae STATIC ${STRATA_VAE_DIR}/wan_vae.cu)
        target_link_libraries(strata_vae PRIVATE hip::host roc::hipblas)
        # HIP is enabled here, after the stable-diffusion library was defined: give both targets the architectures
        # (else CMake stops with "HIP_ARCHITECTURES is empty for target stable-diffusion")
        set_target_properties(strata_vae ${SD_LIB} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    target_include_directories(strata_vae PRIVATE ${STRATA_VAE_DIR} ${STRATA_VAE_DIR}/include)
    target_compile_features(strata_vae PRIVATE cxx_std_17)
    set_target_properties(strata_vae PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${SD_LIB} PRIVATE strata_vae)
    target_include_directories(${SD_LIB} PRIVATE ${STRATA_VAE_DIR}/include)
    target_compile_definitions(${SD_LIB} PRIVATE SD_STRATA_VAE)
endif()
""" + CMAKE_OLD


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"{path}: the expected code is not there (another stable-diffusion.cpp commit?)")
    path.write_bytes(text.replace(old, new).encode("utf-8"))


def copy_sources(dst: Path) -> None:
    """The decoder's files into dst (src/strata_vae of a checkout), byte for byte."""
    for src, rel in FILES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / src, dst / rel)
    hc = dst / HIP_COMPAT
    hc.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / HIP_COMPAT).iterdir()):
        if f.is_file():
            shutil.copyfile(f, hc / f.name)


def apply(sd: Path) -> None:
    sd = Path(sd)
    copy_sources(sd / "src" / "strata_vae")
    engine = sd / "src" / "pipeline" / "diffusion_engine.cpp"
    edit(engine, INCLUDE_OLD, INCLUDE_NEW)
    edit(engine, DECODE_OLD, DECODE_NEW)
    edit(sd / "CMakeLists.txt", CMAKE_OLD, CMAKE_NEW)


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        apply(Path(sys.argv[1]))
        print("Strata VAE decoder applied")
    else:
        raise SystemExit(__doc__)
