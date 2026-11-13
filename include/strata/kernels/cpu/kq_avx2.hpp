// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/kq_avx2.hpp - the routing-aware prefetch's router dot product (AVX2).  Upstream's file
// also holds UD-Q4_K_XL's multi-token expert rows, which XeStrata does not take (measured no faster).
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// out[t * rows + r] = w_r . x_t for BF16 rows (bits) and `nt` <= 8 fp32 tokens (x: nt rows of cols, cols % 8 == 0):
/// the router of RouterLookahead (an estimate only).  Needs AVX2 and FMA (cpu_avx2_ok).
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

}  // namespace strata::kernels::cpu
