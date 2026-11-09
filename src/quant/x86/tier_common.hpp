// src/quant/x86/tier_common.hpp - what every x86 tier file of the layer needs before it includes ggml's ported x86 code (x86_dots.inc, x86_act.inc).
//
// The tier files (dot_avx.cpp, dot_avx2.cpp, act_avx.cpp, act_avx2.cpp) are compiled with their own instruction-set flags (cmake/backends/quant.cmake),
// so the #if __AVX2__ / __AVX__ branches of the ported code select exactly what ggml-cpu selects for that flavour.  Never include this header from a
// file compiled without those flags.
#pragma once

#define STRATA_QUANT_TABLES
#include "../internal.hpp"

#include <immintrin.h>

#include <cassert>
#include <cstdlib>

#define UNUSED(x) (void) (x)

// Keeps a float product a separate rounded value, so the compiler cannot fuse it with the add that follows (where ggml-cpu's build does not).
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define STRATA_NOFUSE(v) __asm__("" : "+x"(v))
#else
#define STRATA_NOFUSE(v) (void) (v)   // MSVC: checked by the Windows run of tests/quant/quant_parity (RESULTS-OWN-2A.md)
#endif

// ggml's own choice: F16C hardware conversions where the flavour has them (same values as the software ones for every finite input), software otherwise.
#if defined(__F16C__) || (defined(_MSC_VER) && defined(__AVX2__))
namespace strata::quant {
#if defined(_MSC_VER)   // MSVC has no scalar _cvtsh_ss / _cvtss_sh: the packed ones on one lane (the same VCVTPH2PS / VCVTPS2PH)
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t f2h(float f) { return (uint16_t) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), 0)); }
#else
inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
#endif
}  // namespace strata::quant
#else
namespace strata::quant {
inline float h2f(uint16_t h) { return fp16_to_fp32(h); }
inline uint16_t f2h(float f) { return fp32_to_fp16(f); }
}  // namespace strata::quant
#endif
