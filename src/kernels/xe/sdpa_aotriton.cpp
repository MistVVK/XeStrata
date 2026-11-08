// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sdpa_aotriton.cpp - see include/strata/kernels/sdpa_aotriton.hpp: aotriton's module opened on an AMD
// GPU's first attention (STRATA_AOTRITON: a build with it; without, nothing taken).

#include "strata/kernels/sdpa_aotriton.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"

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

namespace strata::kernels {

#if STRATA_AOTRITON
namespace {

using AotFn = bool (*)(void*, const uint16_t*, const uint16_t*, const uint16_t*, int64_t, int64_t, int64_t, int64_t,
                       int64_t, uint16_t*, bool);   // strata_sdpa_aotriton's

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
        const char* name = "xestrata_sdpa_aotriton.dll";
#else
        const fs::path dir = fs::read_symlink("/proc/self/exe", ec).parent_path();
        const char* name = "libxestrata_sdpa_aotriton.so";
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
        std::fprintf(stderr, "xestrata: aotriton's attention module did not open (%s): the engine's own kernel\n",
                     why.c_str());
    });
    return fn;
}

}  // namespace
#endif

bool aotriton_sdpa_ready([[maybe_unused]] void* stream) {
    if (no_xmx()) return false;   // its kernels run on the matrix engines (STRATA_NO_XMX=1 plays a GPU without them)
#if STRATA_AOTRITON
    return core::Runtime::get().stream(stream).get_device().get_backend() == sycl::backend::ext_oneapi_hip &&
           aotriton_module() != nullptr;
#else
    return false;
#endif
}

bool aotriton_sdpa([[maybe_unused]] const uint16_t* q16, [[maybe_unused]] const uint16_t* k16,
                   [[maybe_unused]] const uint16_t* v16, [[maybe_unused]] int64_t rows, [[maybe_unused]] int64_t keys,
                   [[maybe_unused]] int64_t heads, [[maybe_unused]] int64_t kv_heads, [[maybe_unused]] int64_t hd,
                   [[maybe_unused]] uint16_t* out16, [[maybe_unused]] void* stream, [[maybe_unused]] bool causal) {
#if STRATA_AOTRITON
    if (!aotriton_sdpa_ready(stream)) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    return aotriton_module()(&q, q16, k16, v16, rows, keys, heads, kv_heads, hd, out16, causal);
#else
    return false;
#endif
}

}  // namespace strata::kernels
