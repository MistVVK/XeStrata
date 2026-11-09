// include/strata/quant/dequant.hpp - one row of blocks -> floats (own-stack phase 2).
//
// For the host-side readers of a quantized tensor: the --mtp-q4 requantizer (Q8_0 and the draft head's format), the DeepSeek embedding rows.  The same
// arithmetic as ggml's `dequantize_row_*`, scalar; not a hot path.
#pragma once

#include <cstdint>

#include "strata/quant/types.hpp"

namespace strata::quant {

/// True for the types dequant_row() handles: Q2_0, Q4_0/1, Q5_0/1, Q8_0, MXFP4, NVFP4, Q2_K..Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS.
bool can_dequant(Type t) noexcept;

/// n elements (a whole number of the type's blocks) of the row at `src` -> dst.  False (nothing written) for a type it does not handle.
bool dequant_row(Type t, const void* src, float* dst, int64_t n);

}  // namespace strata::quant
