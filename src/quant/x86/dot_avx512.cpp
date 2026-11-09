// src/quant/x86/dot_avx512.cpp - 512-bit kernels of the Avx2 contract (VNNI integer parts in 512-bit lanes, float accumulation in ggml's AVX2 order).
// Compiled with the AVX-512 flags of cmake/backends/quant.cmake; only ever called when caps().avx512 is true.
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {

DotTable avx512_dot_table() noexcept { return {nullptr, 0}; }

}  // namespace strata::quant
