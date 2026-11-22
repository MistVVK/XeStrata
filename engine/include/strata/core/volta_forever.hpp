// SPDX-License-Identifier: MIT
// include/strata/core/volta_forever.hpp - runtime detection and launch/module-loading policy for NVIDIA sm_70
// (Tesla V100 / Titan V) running CUDA 13.x binaries via the external volta-forever libcuda shim.
//
// Why this exists:
//   1. CUDA 13.0+ dropped sm_70 compilation targets.  The community volta-forever shim interposes libcuda.so.1,
//      rewriting sm_75/sm_80 cubins and JIT-translating PTX to sm_70 on module load.
//   2. Forcing CUDA_MODULE_LOADING=EAGER (Strata's default to reserve ~30 MiB of code VRAM upfront) makes a
//      volta-forever shim translate EVERY compiled kernel in the binary before main() even prints a banner, causing
//      multi-minute startup hangs or JIT failures on sm_80+/sm_90+/sm_120 kernels that sm_70 never calls.
//      Under volta-forever, CUDA_MODULE_LOADING=LAZY must be used unless the operator explicitly set CUDA_MODULE_LOADING.
//   3. volta-forever's cuGraphAddKernelNode translation on sm_70 rejects 1024-thread blocks (max 512 threads per block
//      in captured CUDA Graphs), whereas direct stream launches accept 1024 threads normally.
#pragma once

#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace strata::core {

/// Queries `volta_forever_version()` via `dlsym(RTLD_DEFAULT, ...)` on POSIX without linking against any library.
/// Returns nullptr when volta-forever is not loaded in the current process.
inline const char* volta_forever_version_str() noexcept {
#if !defined(_WIN32)
    using Fn = const char* (*)();
    void* sym = dlsym(RTLD_DEFAULT, "volta_forever_version");
    if (sym != nullptr) {
        Fn fn = nullptr;
        std::memcpy(&fn, &sym, sizeof(sym));
        return fn ? fn() : "loaded";
    }
#endif
    return nullptr;
}

/// Pure helper (testable with injected strings): detects volta-forever from symbol version, LD_PRELOAD, or explicit
/// STRATA_VOLTA_FOREVER override ("1" forces on, "0" forces off).
inline bool volta_forever_active_from(const char* explicit_env,
                                      const char* ld_preload_env,
                                      const char* dlsym_version,
                                      const char* vf_env = nullptr) noexcept {
    if (explicit_env != nullptr && explicit_env[0] != '\0') {
        return explicit_env[0] != '0';
    }
    if (vf_env != nullptr && vf_env[0] != '\0' && std::strcmp(vf_env, "0") != 0) {
        return true;
    }
    if (dlsym_version != nullptr && dlsym_version[0] != '\0') {
        return true;
    }
    if (ld_preload_env != nullptr && std::strstr(ld_preload_env, "libvolta_forever") != nullptr) {
        return true;
    }
    return false;
}

inline bool volta_forever_active() noexcept {
    return volta_forever_active_from(std::getenv("STRATA_VOLTA_FOREVER"),
                                     std::getenv("LD_PRELOAD"),
                                     volta_forever_version_str(),
                                     std::getenv("VOLTA_FOREVER"));
}

/// Under volta-forever on V100 (sm_70 masquerading as sm_75 to CUDA 13), native dp4a MMQ prefill runs ~1.95x faster
/// than emulated sm_75 mma.m8n8k16.s8 tensor cores (582 vs 299 tok/s).
constexpr bool prefer_dp4a_mmq(bool vf_active) noexcept {
    return vf_active;
}

/// Returns the default CUDA_MODULE_LOADING value when the user has not set CUDA_MODULE_LOADING:
/// "LAZY" when volta-forever is active (avoids eager JIT translation of unused sm_80+/sm_90+/sm_120 cubins),
/// "EAGER" on native NVIDIA drivers (reserves ~30 MiB kernel code upfront before expert cache fills VRAM).
constexpr const char* default_cuda_module_loading(bool vf_active) noexcept {
    return vf_active ? "LAZY" : "EAGER";
}

/// Sets CUDA_MODULE_LOADING if unset in the environment.  Returns true if volta-forever was detected and LAZY was set.
inline bool apply_default_cuda_module_loading() noexcept {
    const bool vf = volta_forever_active();
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        const char* mode = default_cuda_module_loading(vf);
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", mode);
#else
        setenv("CUDA_MODULE_LOADING", mode, 0);
#endif
        return vf;
    }
    return false;
}

/// Maximum safe thread block size inside CUDA Graph capture: 512 on sm_70 under volta-forever, 1024 otherwise.
constexpr int safe_cuda_graph_max_threads(int cc_major, int cc_minor, bool vf_active) noexcept {
    return (vf_active && cc_major == 7 && cc_minor == 0) ? 512 : 1024;
}

/// Whether automatic CUDA Graph capture should be enabled by default for kernels that launch 1024-thread blocks.
constexpr bool allow_1024_thread_cuda_graphs(int cc_major, int cc_minor, bool vf_active) noexcept {
    return safe_cuda_graph_max_threads(cc_major, cc_minor, vf_active) >= 1024;
}

}  // namespace strata::core
