// src/quant/x86/dot_avx2.cpp - the Avx2 contract: ggml-cpu's dot products as its AVX2 + FMA + F16C flavour compiles them (the shipped release flavour).
// Compiled with -mavx2 -mfma -mf16c (/arch:AVX2).  Types ggml has no SIMD branch for (Q2_0) run the Generic code under this contract.
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {
namespace avx2 {

#include "x86_helpers.inc"
#include "x86_dots.inc"

}  // namespace avx2

namespace {
#define ENTRY(T, ACT, NAME) {avx2::dot_##NAME, Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0),   ENTRY(Q4_1, Q8_1, q4_1_q8_1),   ENTRY(Q5_0, Q8_0, q5_0_q8_0),   ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY(Q8_0, Q8_0, q8_0_q8_0),   ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0), ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
    ENTRY(Q2_K, Q8_K, q2_K_q8_K),   ENTRY(Q3_K, Q8_K, q3_K_q8_K),   ENTRY(Q4_K, Q8_K, q4_K_q8_K),   ENTRY(Q5_K, Q8_K, q5_K_q8_K),
    ENTRY(Q6_K, Q8_K, q6_K_q8_K),   ENTRY(IQ1_S, Q8_K, iq1_s_q8_K),  ENTRY(IQ1_M, Q8_K, iq1_m_q8_K),  ENTRY(IQ2_XXS, Q8_K, iq2_xxs_q8_K),
    ENTRY(IQ2_XS, Q8_K, iq2_xs_q8_K), ENTRY(IQ2_S, Q8_K, iq2_s_q8_K), ENTRY(IQ3_XXS, Q8_K, iq3_xxs_q8_K), ENTRY(IQ3_S, Q8_K, iq3_s_q8_K),
    ENTRY(IQ4_XS, Q8_K, iq4_xs_q8_K),
    {generic::dot_q2_0_q8_0_generic, Type::Q2_0, Type::Q8_0, Arith::Avx2, Isa::Scalar, "avx2_q2_0_generic"},
};
#undef ENTRY
}  // namespace

DotTable avx2_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
