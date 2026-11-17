// include/strata/quant/act.hpp - activation quantizers (float row -> Q8_0 / Q8_1 / Q8_K blocks), own-stack phase 2.
//
// An activation is quantized once per token and layer and every weight row's dot reads it, so the quantizer's rounding is part of the
// arithmetic contract (isa.hpp): ggml's AVX / AVX2 Q8_0 and Q8_1 quantizers round half to even (`_mm256_round_ps(NEAREST)`), its generic
// `_ref` code rounds half away from zero (`roundf`), and the two differ on exact ties.  Q8_K is ggml's scalar `quantize_row_q8_K_ref` on
// every x86 flavour (the Arm flavour has its own).  Each contract's quantizer below reproduces that flavour's bytes.
#pragma once

#include <cstdint>

#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

namespace strata::quant {

/// x (n floats; n a whole number of blocks: 32 for Q8_0 / Q8_1, 256 for Q8_K) -> blocks y.
using ActQuantFn = void (*)(const float* x, void* y, int64_t n);

struct ActKernel {
    ActQuantFn fn = nullptr;
    Type act = Type::Q8_0;
    Arith arith = Arith::Generic;
    Isa isa = Isa::Scalar;
    const char* name = "";
};

/// The best quantizer of contract `a` for the activation format `act` that runs on a CPU of level `max_isa` (default: this CPU), or nullptr.
const ActKernel* act_kernel(Type act, Arith a, Isa max_isa) noexcept;
const ActKernel* act_kernel(Type act, Arith a) noexcept;

/// The reference quantizers (the Generic contract: ggml's `_ref` functions).
void quantize_q8_0_ref(const float* x, void* y, int64_t n);
void quantize_q8_1_ref(const float* x, void* y, int64_t n);
void quantize_q8_K_ref(const float* x, void* y, int64_t n);
/// Q4_0 from floats, ggml's `quantize_row_q4_0_ref` (the `--mtp-q4` requantizer's output format).
void quantize_q4_0_ref(const float* x, void* y, int64_t n);

}  // namespace strata::quant
