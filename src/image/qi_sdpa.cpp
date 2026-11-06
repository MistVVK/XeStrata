// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa.cpp - see include/strata/image/qi_kernels.hpp (qi_sdpa): the image attention's unmasked rows through
// a GPU maker's kernel, the tensors' strides the engine's [rows][heads * head_dim] as they are, a plan built once a shape
// (the query rows, the key rows) and kept:
//   * Intel GPUs: oneDNN's SDPA through its graph API (Q . K^T, the scale, the softmax, . V fused into one partition);
//   * NVIDIA GPUs (contrib-llvm): cuDNN's SDPA through cudnn-frontend, cuDNN opened at the first attention
//     (the user's installation, not free software: open_cudnn), on the queue's CUDA stream;
//   * AMD GPUs: aotriton's flash attention (MIT; its library and its kernels' images beside the engine), on the
//     queue's HIP stream, in a module the engine opens on an AMD GPU only (aotriton needs AMD's HIP runtime).
// A GPU or a build without one, or a failure, leaves it to the engine's own kernel.

#include "strata/image/qi_kernels.hpp"

#include "qi_sdpa.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/sdpa_onednn.hpp"

#include <sycl/sycl.hpp>

#if STRATA_AOTRITON
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#endif

namespace strata::image {

#if STRATA_AOTRITON
namespace {

using AotFn = bool (*)(void*, const uint16_t*, const uint16_t*, const uint16_t*, int64_t, int64_t, int64_t, int64_t,
                       int64_t, uint16_t*, bool);

// The aotriton module (src/image/qi_sdpa_aotriton.cpp), opened at an AMD GPU's first attention from the engine's
// folder or its lib/ (the packages'), with the libraries it needs beside it; null when it does not open (no HIP
// runtime here, a build without it), said once.
AotFn aotriton_module() {
    static AotFn fn = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        namespace fs = std::filesystem;
        std::error_code ec;
#if defined(_WIN32)
        char exe[MAX_PATH] = {};
        const fs::path dir = GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0 ? fs::path(exe).parent_path() : fs::path();
        const char* name = "strata_sdpa_aotriton.dll";
#else
        const fs::path dir = fs::read_symlink("/proc/self/exe", ec).parent_path();
        const char* name = "libstrata_sdpa_aotriton.so";
#endif
        std::string why = "not found";
        for (const fs::path& p : {dir / name, dir / "lib" / name}) {
            if (!fs::exists(p, ec)) continue;
#if defined(_WIN32)
            // its own folder first for the DLLs it imports (aotriton_v2.dll; amdhip64_7.dll from the driver's)
            HMODULE h = LoadLibraryExA(p.string().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (h == nullptr) {
                why = p.string() + ": error " + std::to_string(GetLastError());
                continue;
            }
            fn = reinterpret_cast<AotFn>(reinterpret_cast<void*>(GetProcAddress(h, "strata_sdpa_aotriton")));
#else
            void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (h == nullptr) {
                const char* e = dlerror();
                why = e != nullptr ? e : p.string();
                continue;
            }
            fn = reinterpret_cast<AotFn>(dlsym(h, "strata_sdpa_aotriton"));
#endif
            if (fn != nullptr) return;
            why = p.string() + ": no strata_sdpa_aotriton";
        }
        std::fprintf(stderr, "strata-image: aotriton's attention module did not open (%s): the engine's own kernel\n",
                     why.c_str());
    });
    return fn;
}

}  // namespace
#endif

bool qi_sdpa_causal([[maybe_unused]] void* stream) {
    if (kernels::no_xmx()) return false;   // as qi_sdpa
#if STRATA_AOTRITON
    return core::Runtime::get().stream(stream).get_device().get_backend() == sycl::backend::ext_oneapi_hip &&
           aotriton_module() != nullptr;
#else
    return false;
#endif
}

bool qi_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1, int64_t T,
             int64_t heads, int64_t head_dim, uint16_t* out16, void* stream, bool causal) {
    if (q1 <= q0) return true;
    // every maker's kernel here runs on the matrix engines: STRATA_NO_XMX=1 (a GPU without them, played on one with
    // them) leaves the attention to the engine's own kernels, which take their paths without them
    if (kernels::no_xmx()) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    [[maybe_unused]] const sycl::backend be = q.get_device().get_backend();
#if STRATA_AOTRITON
    if (be == sycl::backend::ext_oneapi_hip) {
        const AotFn fn = aotriton_module();
        return fn != nullptr && fn(&q, q16, k16, v16, q0, q1, T, heads, head_dim, out16, causal);
    }
#endif
    if (causal) return false;   // the other makers' kernels take the unmasked rows only
#if STRATA_CUDNN
    if (be == sycl::backend::ext_oneapi_cuda) return sdpa::cudnn(q, q16, k16, v16, q0, q1, T, heads, head_dim, out16);
#endif
#if STRATA_ONEDNN
    // oneDNN's GPU kernels are Intel's: its Level Zero and OpenCL devices
    if (be == sycl::backend::ext_oneapi_level_zero || be == sycl::backend::opencl)
        return kernels::onednn_sdpa(q16 + q0 * heads * head_dim, k16, v16, q1 - q0, T, heads, heads, head_dim, out16,
                                    stream);
#endif
    (void) k16, (void) v16, (void) T, (void) heads, (void) head_dim, (void) out16, (void) q16;
    return false;
}

}  // namespace strata::image
