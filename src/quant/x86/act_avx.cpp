// src/quant/x86/act_avx.cpp - the Avx contract's activation quantizers: ggml's AVX Q8_0 / Q8_1 (128-bit integer halves), and Q8_K (ggml's scalar `_ref`, here in
// 128-bit lanes, byte for byte the same; the engine's q8k_quant_avx1, moved into the layer).
// Compiled with -mavx and AVX2, FMA and F16C off (/arch:AVX).
#include "tier_common.hpp"

#include "../registry.hpp"
#include "strata/quant/act.hpp"

#if defined(__AVX2__) || defined(__FMA__) || defined(__F16C__)
#error "act_avx.cpp is the AVX-only tier: compile it without AVX2, FMA and F16C"
#endif

namespace strata::quant {
namespace avx {

#include "x86_helpers.inc"
#include "x86_act.inc"
#include "q8k_avx1.inc"

}  // namespace avx

namespace {
const ActKernel kTable[] = {
    {avx::quantize_q8_0, Type::Q8_0, Arith::Avx, Isa::Avx1, "avx_q8_0"},
    {avx::quantize_q8_1, Type::Q8_1, Arith::Avx, Isa::Avx1, "avx_q8_1"},
    {avx::quantize_q8_K_avx1, Type::Q8_K, Arith::Avx, Isa::Avx1, "avx_q8_K"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Avx, Isa::Scalar, "avx_q8_K_ref"},
};
}  // namespace

ActTable avx_act_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
