// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/q2_avxvnni.cpp - q2_avx2.cpp's Q2_0 rows with AVX-VNNI's vpdpbusd for the integer dot (Intel Core 12th
// gen and later, AMD Zen 5), chosen at run time (cpu_avxvnni_ok).  Bitwise the same rows (rows256.hpp).
#include "strata/kernels/cpu/expert.hpp"
#include "rows256.hpp"

#include <immintrin.h>

namespace strata::kernels::cpu {
namespace {
struct DotAvxVnni {
    static __m256i dot(__m256i u, __m256i s) { return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s); }
};
}  // namespace

void q2_0_gguf_rows_multi_avxvnni(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                  float* const* out, int r0, int r1) {
    rows256::q2_rows_nt<DotAvxVnni>(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
