// tools/vision/own/cpu_caps.h - the instruction-set probe and the 16-bit float helpers of the CPU tower.
//
// TODO(0.2.15): use strata_quant.  This is a stand-in with the same names as the OWN-2A layer (quant::caps(), best_isa(), isa_ok(), the STRATA_FORCE_ISA cap,
// the fp16 helpers), written here because f-own-3v is not based on f-own-2a; when the branches meet the file is deleted and the includes point at
// include/strata/quant/.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#include <intrin.h>
#endif

namespace strata_vit {
namespace quant {

enum class Isa { Scalar = 0, Avx2 = 1, Avx512 = 2, Neon = 3 };   // Neon: AArch64 only (baseline Advanced SIMD); the x86 tiers are 0 .. 2

struct Caps {
    bool avx2 = false, fma = false, f16c = false, avx512 = false;   // avx512: F, BW, DQ, VL (what the tower's kernels use) and the OS saves the state
};

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
inline Caps probe() {
    Caps c;
    int r[4];
    __cpuid(r, 0);
    const int n = r[0];
    if (n < 7) return c;
    __cpuid(r, 1);
    const bool osxsave = (r[2] >> 27) & 1, fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1;
    unsigned long long xcr = osxsave ? _xgetbv(0) : 0;
    const bool ymm = (xcr & 6) == 6, zmm = (xcr & 0xe6) == 0xe6;
    __cpuidex(r, 7, 0);
    c.avx2 = ymm && ((r[1] >> 5) & 1);
    c.fma = ymm && fma;
    c.f16c = ymm && f16c;
    c.avx512 = zmm && ((r[1] >> 16) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 30) & 1) && ((r[1] >> 31) & 1);
    return c;
}
#else
inline Caps probe() {
    Caps c;
    __builtin_cpu_init();
    c.avx2 = __builtin_cpu_supports("avx2");
    c.fma = __builtin_cpu_supports("fma");
    c.f16c = __builtin_cpu_supports("f16c");
    c.avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
    return c;
}
#endif
#else
inline Caps probe() { return Caps(); }
#endif

inline const Caps& caps() {
    static const Caps c = probe();
    return c;
}

inline bool isa_ok(Isa i) {
    const Caps& c = caps();
    if (i == Isa::Avx512) return c.avx512 && c.avx2 && c.fma;
    if (i == Isa::Avx2) return c.avx2 && c.fma && c.f16c;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (i == Isa::Neon) return true;             // part of every AArch64 CPU
#else
    if (i == Isa::Neon) return false;
#endif
    return true;
}

// the best tier this CPU runs, capped by STRATA_FORCE_ISA=scalar|avx2|avx512 (on AArch64: neon, or scalar)
inline Isa best_isa() {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (const char* e = std::getenv("STRATA_FORCE_ISA"))
        if (std::string(e) == "scalar") return Isa::Scalar;
    return Isa::Neon;
#endif
    Isa best = isa_ok(Isa::Avx512) ? Isa::Avx512 : isa_ok(Isa::Avx2) ? Isa::Avx2 : Isa::Scalar;
    if (const char* e = std::getenv("STRATA_FORCE_ISA")) {
        const std::string s(e);
        const Isa cap = s == "scalar" ? Isa::Scalar : s == "avx2" ? Isa::Avx2 : Isa::Avx512;
        if ((int) cap < (int) best) best = cap;
    }
    return best;
}

inline const char* isa_name(Isa i) { return i == Isa::Neon ? "neon" : i == Isa::Avx512 ? "avx512" : i == Isa::Avx2 ? "avx2" : "scalar"; }

// ---- 16-bit floats (bit exact, no hardware needed) ----
inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) {
        if (m == 0) u = sign;
        else {   // subnormal: renormalise
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            u = sign | (e << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace quant
}  // namespace strata_vit
