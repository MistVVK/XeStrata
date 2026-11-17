// src/quant/x86/act_avx2.cpp - the Avx2 contract's activation quantizers: ggml's AVX2 Q8_0 / Q8_1 (round half to even), and Q8_K, which is ggml's scalar
// `_ref` on every x86 flavour and is quantized here byte for byte the same in AVX2 (the engine's q8k_quant_avx2, moved into the layer).
// Compiled with -mavx2 -mfma -mf16c (/arch:AVX2).
#include "tier_common.hpp"

#include "../registry.hpp"
#include "strata/quant/act.hpp"

namespace strata::quant {
namespace avx2 {

#include "x86_helpers.inc"
#include "x86_act.inc"
#include "q8k_avx2.inc"

}  // namespace avx2

namespace {
const ActKernel kTable[] = {
    {avx2::quantize_q8_0, Type::Q8_0, Arith::Avx2, Isa::Avx2, "avx2_q8_0"},
    {avx2::quantize_q8_1, Type::Q8_1, Arith::Avx2, Isa::Avx2, "avx2_q8_1"},
    {avx2::quantize_q8_K_avx2, Type::Q8_K, Arith::Avx2, Isa::Avx2, "avx2_q8_K"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Avx2, Isa::Scalar, "avx2_q8_K_ref"},
};
}  // namespace

ActTable avx2_act_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
