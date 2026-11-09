// src/quant/isa.cpp - the CPU probe of the quant layer (own-stack phase 2).  No ggml.
#include "strata/quant/isa.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define STRATA_QUANT_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define STRATA_QUANT_ARM64 1
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

namespace strata::quant {
namespace {

#ifdef STRATA_QUANT_X86
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int a[4];
    __cpuidex(a, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) a[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
unsigned long long xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long) edx << 32) | eax;
#endif
}
#endif

// STRATA_FORCE_ISA caps the probes like the pool's own (expert_layout.cpp): avx2 = no AVX-512, avx = AVX only, sse = SSE4.2 only.
int isa_cap() {
    const char* f = std::getenv("STRATA_FORCE_ISA");
    if (f == nullptr || *f == 0) return 3;
    if (!std::strcmp(f, "avx2")) return 2;
    if (!std::strcmp(f, "avx")) return 1;
    if (!std::strcmp(f, "sse")) return 0;
    std::fprintf(stderr, "strata: STRATA_FORCE_ISA=%s is not avx2, avx or sse; ignored\n", f);
    return 3;
}

Caps probe() {
    Caps c;
#ifdef STRATA_QUANT_X86
    unsigned r[4] = {0, 0, 0, 0};
    cpuid(0, 0, r);
    const unsigned maxleaf = r[0];
    cpuid(1, 0, r);
    const unsigned ecx1 = r[2];
    c.sse42 = (ecx1 >> 20) & 1;
    const bool osxsave = (ecx1 >> 27) & 1;
    const unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
    const bool ymm = (xcr0 & 6) == 6;
    const bool zmm = ymm && (xcr0 & 0xE0) == 0xE0;
    c.avx = ymm && ((ecx1 >> 28) & 1);
    const bool fma = (ecx1 >> 12) & 1, f16c = (ecx1 >> 29) & 1;
    if (maxleaf >= 7) {
        cpuid(7, 0, r);
        const unsigned ebx = r[1], ecx = r[2];
        c.avx2 = c.avx && fma && f16c && ((ebx >> 5) & 1);
        c.bmi2 = (ebx >> 8) & 1;
        c.avx512 = zmm && ((ebx >> 16) & 1) && ((ebx >> 17) & 1) && ((ebx >> 30) & 1) && ((ebx >> 31) & 1);   // F, DQ, BW, VL
        c.avx512_vnni = c.avx512 && ((ecx >> 11) & 1);
        cpuid(7, 1, r);
        c.avx_vnni = c.avx2 && ((r[0] >> 4) & 1);
    }
    const int cap = isa_cap();
    if (cap < 3) c.avx512 = c.avx512_vnni = false;
    if (cap < 2) c.avx2 = c.avx_vnni = c.bmi2 = false;
    if (cap < 1) c.avx = false;
#elif defined(STRATA_QUANT_ARM64)
    c.neon = true;
#if defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP);
    c.dotprod = (hw >> 20) & 1;   // HWCAP_ASIMDDP
    const unsigned long hw2 = getauxval(AT_HWCAP2);
    c.i8mm = (hw2 >> 13) & 1;     // HWCAP2_I8MM
#elif defined(__APPLE__)
    int v = 0;
    size_t sz = sizeof v;
    c.dotprod = sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, nullptr, 0) == 0 && v;
    v = 0;
    sz = sizeof v;
    c.i8mm = sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, nullptr, 0) == 0 && v;
#endif
#endif
    return c;
}

}  // namespace

const Caps& caps() noexcept {
    static const Caps c = probe();
    return c;
}

Isa best_isa() noexcept {
    const Caps& c = caps();
    if (c.avx512 && c.avx512_vnni) return Isa::Avx512;
    if (c.avx2) return Isa::Avx2;
    if (c.avx) return Isa::Avx1;
    if (c.neon) return Isa::Neon;
    return Isa::Scalar;
}

Arith default_arith() noexcept {
    const Caps& c = caps();
    if (c.neon) return Arith::Neon;
    if (c.avx2) return Arith::Avx2;
    if (c.avx) return Arith::Avx;
    return Arith::Generic;
}

bool isa_ok(Isa i) noexcept {
    const Caps& c = caps();
    switch (i) {
        case Isa::Scalar: return true;
        case Isa::Neon: return c.neon;
        case Isa::Avx1: return c.avx;
        case Isa::Avx2: return c.avx2;
        case Isa::Avx512: return c.avx512 && c.avx512_vnni;
    }
    return false;
}

const char* isa_name(Isa i) noexcept {
    switch (i) {
        case Isa::Scalar: return "scalar";
        case Isa::Neon: return "neon";
        case Isa::Avx1: return "avx";
        case Isa::Avx2: return "avx2";
        case Isa::Avx512: return "avx512";
    }
    return "?";
}

const char* arith_name(Arith a) noexcept {
    switch (a) {
        case Arith::Generic: return "generic";
        case Arith::Avx: return "avx";
        case Arith::Avx2: return "avx2";
        case Arith::Neon: return "neon";
    }
    return "?";
}

}  // namespace strata::quant
