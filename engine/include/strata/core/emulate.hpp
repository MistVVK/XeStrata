// strata/core/emulate.hpp - STRATA_EMULATE_CC: another GPU generation's code paths, run on this card (tests only).
//
// A build whose device code is PTX for an older architecture (CMAKE_CUDA_ARCHITECTURES=86-virtual) runs on a newer
// card through the driver's JIT, and every kernel then sees __CUDA_ARCH__ = 860.  What the host decides from the
// device must agree with that, or it picks paths and shared-memory sizes the device code was not built for:
// STRATA_EMULATE_CC=75|80|86|89 makes each such query (compute capability, shared memory per block) answer as that
// card would.  Speed under it says nothing about the real card; results do - the kernels are the real card's.
#pragma once

#include <cstdlib>
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_GPU_METAL)
#include <dlfcn.h>
#include <string>
#endif

namespace strata {

/// nbald/volta-forever version string when running under the CUDA 13 Volta compatibility proxy (libcuda.so.1
/// exports volta_forever_version()), or "" when not loaded.  Cached once per process.
inline const char* volta_forever_version() {
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_GPU_METAL)
    static const std::string ver = []() -> std::string {
        using Fn = const char* (*)();
        auto probe = [](void* handle) -> std::string {
            if (!handle) return {};
            void* sym = dlsym(handle, "volta_forever_version");
            if (!sym) return {};
            const char* s = reinterpret_cast<Fn>(sym)();
            return s ? std::string(s) : std::string();
        };
        std::string out;
#ifdef RTLD_DEFAULT
        out = probe(RTLD_DEFAULT);
        if (!out.empty()) return out;
#endif
#ifdef RTLD_NOLOAD
        if (void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD)) {
            out = probe(h);
            dlclose(h);
            if (!out.empty()) return out;
        }
#endif
        if (void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL)) {
            out = probe(h);
            dlclose(h);
            if (!out.empty()) return out;
        }
        return {};
    }();
    return ver.c_str();
#else
    return "";
#endif
}

inline bool volta_forever_active() { return volta_forever_version()[0] != '\0'; }

/// the emulated compute capability as 10 * major + minor, or 0 (off)
inline int emulated_cc() {
    static const int cc = [] {
        const char* e = std::getenv("STRATA_EMULATE_CC");
        const int v = e ? std::atoi(e) : 0;
        return v >= 75 && v < 120 ? v : 0;
    }();
    return cc;
}
/// Effective compute capability (10 * major + minor).  STRATA_EMULATE_CC wins; otherwise under
/// nbald/volta-forever (which spoofs physical Volta 7.0 as 7.5 for CUDA 13 binaries), remap 7.5 -> 7.0.
inline int cc_of(int real_major, int real_minor, bool vf_active) {
    if (const int e = emulated_cc()) return e;
    if (vf_active && real_major == 7 && real_minor == 5) return 70;
    return 10 * real_major + real_minor;
}
inline int cc_of(int real_major, int real_minor) {
    return cc_of(real_major, real_minor, volta_forever_active());
}
inline int cc_major_of(int real_major) { return emulated_cc() ? emulated_cc() / 10 : real_major; }
inline int cc_minor_of(int real_minor) {
    if (const int e = emulated_cc()) return e % 10;
    if (volta_forever_active() && real_minor == 5) return 0;
    return real_minor;
}
/// cudaDevAttrMaxSharedMemoryPerBlockOptin as the emulated card reports it: 64 KB on Turing, 163 KB on the A100,
/// 99 KB on the other Ampere and Ada cards
inline int smem_optin_of(int real) {
    switch (emulated_cc()) {
        case 0: return real;
        case 75: return 65536;
        case 80: return 166912;
        default: return real < 101376 ? real : 101376;
    }
}

}  // namespace strata
