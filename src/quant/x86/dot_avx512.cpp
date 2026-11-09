// src/quant/x86/dot_avx512.cpp - the Avx2 contract on CPUs with AVX-512 VL + VNNI: the same ported ggml code as dot_avx2.cpp, built so that its
// mul_sum_us8_pairs_float helper takes the VPDPBUSD branch ggml has for `__AVX512VNNI__ && __AVX512VL__` (one instruction for maddubs + madd).  The integer sums are
// exact either way and the float order is the AVX2 one, so the bits are the AVX2 tier's (quant_parity runs these kernels against ggml's AVX2 dots).
// Compiled with the AVX-512 flags of cmake/backends/quant.cmake; only ever called when caps() says AVX-512 F/DQ/BW/VL + VNNI (Isa::Avx512).
// Only the formats whose dot goes through that helper are registered; the others gain nothing here.
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX512VNNI__) || !defined(__AVX512VL__))
#error "dot_avx512.cpp needs AVX-512 VL and VNNI"
#endif

namespace strata::quant {
namespace avx512 {

#include "x86_helpers.inc"
#include "x86_dots.inc"

}  // namespace avx512

namespace {
#define ENTRY(T, ACT, NAME) {avx512::dot_##NAME, Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0), ENTRY(Q4_1, Q8_1, q4_1_q8_1), ENTRY(Q5_0, Q8_0, q5_0_q8_0), ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0),
};
#undef ENTRY
}  // namespace

DotTable avx512_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
