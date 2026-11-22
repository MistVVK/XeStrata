// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/arm/act_neon.cpp - the Neon contract's activation quantizers: NEON Q8_0 / Q8_1 (round half to even, vcvtnq),
// and SIMD NEON Q8_K (bit-exact to quantize_q8_K_ref on both GCC and Clang ARM64 builds).
#include "tier_common.hpp"

#include "../registry.hpp"
#include "strata/quant/act.hpp"

namespace strata::quant {
namespace neon {

#include "arm_helpers.inc"
#include "arm_act.inc"

}  // namespace neon

namespace {
const ActKernel kTable[] = {
    {neon::quantize_q8_0, Type::Q8_0, Arith::Neon, Isa::Neon, "neon_q8_0"},
    {neon::quantize_q8_1, Type::Q8_1, Arith::Neon, Isa::Neon, "neon_q8_1"},
    {neon::quantize_q8_K, Type::Q8_K, Arith::Neon, Isa::Neon, "neon_q8_K"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Neon, Isa::Scalar, "neon_q8_K_ref"},
};
}  // namespace

ActTable neon_act_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
