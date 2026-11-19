// include/strata/kernels/cpu/kq_avx512.hpp - AVX-512 VNNI multi-token dot products for Unsloth UD-Q4_K_XL's expert formats
// (Q4_K gate/up against Q8_K activations; Q5_1 down against Q8_1, Q8_0 down against Q8_0), bit-exact against the AVX2 contract's
// one-token dot for every token (the same rows kq_avx2.hpp gives).  See src/kernels/cpu/kq_avx512.cpp.
//
// The caller checks cpu_avx512_ok() (AVX-512 F / BW / VL / DQ / VNNI / VBMI) before it calls anything here: the translation unit is
// compiled for those, so on any other CPU the functions are an illegal instruction.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q4_K (12) gate/up, Q5_1 (7) and Q8_0 (8) down: the formats the AVX-512 rows cover.
bool kq512_supported(int ggml_type) noexcept;

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.  Q4_K only; any `nt`.
void kq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).  Q5_1 / Q8_0 (and Q4_K for the tests); any `nt`.
void kq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1);

}  // namespace strata::kernels::cpu
