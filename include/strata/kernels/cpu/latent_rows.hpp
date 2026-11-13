// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/kernels/cpu/latent_rows.hpp - the latent MoE's CPU-share rows: the q8_1 image of h and the
// U product on the host (XESTRATA_LATENT_U_GPU=off).  Written after edge0-strata's kq_avx2.cpp lat_* rows.
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// The engine's q8_1 rule on the host (canon_math.hpp, the same bits the GPU's quantize_q8_1_rows writes):
/// `rows` rows of `n` values -> 36-byte blocks (fp16 d, fp16 sum, 32 int8).  A row of whole 256s shares one
/// scale across its eight blocks, as the GPU's does.
void latent_quant_q8_1_rows(const float* x, int64_t rows, int64_t n, uint8_t* y);

/// One entry's U product: `H` rows of `ELL` values (Q8_0, fmt 8, or Q4_0, fmt 4) against the entry's q8_1 row
/// (`hq`, ELL/32 blocks of 36 B).  out[h] = U[h] . h.  AVX2 where the CPU has it, scalar otherwise.
void latent_u_row(const uint8_t* u_group, int H, int ELL, int fmt, const uint8_t* hq, float* out);

}  // namespace strata::kernels::cpu