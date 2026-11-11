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

// ranked: an x86 tier includes every lower one (STRATA_FORCE_ISA caps by rank).  Neon: AArch64 only (baseline Advanced SIMD), outside the x86 ranking.
enum class Isa { Scalar = 0, Sse41 = 1, Avx = 2, Avx2 = 3, Avx512 = 4, Neon = 5 };

struct Caps {
    bool sse41 = false, avx = false;   // avx: the CPU has it and the OS saves the ymm state
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
    const bool osxsave = (r[2] >> 27) & 1, fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1, avx = (r[2] >> 28) & 1;
    c.sse41 = (r[2] >> 19) & 1;
    unsigned long long xcr = osxsave ? _xgetbv(0) : 0;
    const bool ymm = (xcr & 6) == 6, zmm = (xcr & 0xe6) == 0xe6;
    c.avx = ymm && avx;
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
    c.sse41 = __builtin_cpu_supports("sse4.1");
    c.avx = __builtin_cpu_supports("avx");      // libgcc tests the OS ymm state for it
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

// the same two questions for any feature set (the tests give the sets of the CPU classes; the program asks caps())
inline bool isa_ok_for(const Caps& c, Isa i) {
    if (i == Isa::Avx512) return c.avx512 && c.avx2 && c.fma;
    if (i == Isa::Avx2) return c.avx2 && c.fma && c.f16c;
    if (i == Isa::Avx) return c.avx && c.sse41;
    if (i == Isa::Sse41) return c.sse41;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (i == Isa::Neon) return true;             // part of every AArch64 CPU
#else
    if (i == Isa::Neon) return false;
#endif
    return true;
}

inline bool isa_ok(Isa i) { return isa_ok_for(caps(), i); }

// the best tier a CPU with feature set `c` runs, capped by `force` (STRATA_FORCE_ISA's value, may be null): scalar|sse41|avx|avx2|avx512 (on AArch64: neon, or scalar)
inline Isa best_isa_for(const Caps& c, const char* force) {
#if defined(__aarch64__) || defined(_M_ARM64)
    (void) c;
    if (force && std::string(force) == "scalar") return Isa::Scalar;
    return Isa::Neon;
#else
    Isa best = isa_ok_for(c, Isa::Avx512) ? Isa::Avx512 : isa_ok_for(c, Isa::Avx2) ? Isa::Avx2 : isa_ok_for(c, Isa::Avx) ? Isa::Avx : isa_ok_for(c, Isa::Sse41) ? Isa::Sse41 : Isa::Scalar;
    if (force) {
        const std::string s(force);
        const Isa cap = s == "scalar" ? Isa::Scalar : s == "sse41" ? Isa::Sse41 : s == "avx" ? Isa::Avx : s == "avx2" ? Isa::Avx2 : Isa::Avx512;
        if ((int) cap < (int) best) best = cap;
    }
    return best;
#endif
}

// the best tier this CPU runs, capped by STRATA_FORCE_ISA
inline Isa best_isa() { return best_isa_for(caps(), std::getenv("STRATA_FORCE_ISA")); }

inline const char* isa_name(Isa i) {
    return i == Isa::Neon ? "neon" : i == Isa::Avx512 ? "avx512" : i == Isa::Avx2 ? "avx2" : i == Isa::Avx ? "avx" : i == Isa::Sse41 ? "sse41" : "scalar";
}

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
