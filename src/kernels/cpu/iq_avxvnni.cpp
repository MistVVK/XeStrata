// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/iq_avxvnni.cpp - iq_avx2.cpp's IQ4_NL down rows with AVX-VNNI's vpdpbusd for the integer dot, chosen
// at run time (cpu_avxvnni_ok).  Bitwise the same rows (rows256.hpp).  The other i-quant rows fold an int16 scale
// into madd, which vpdpbusd has no place for: they stay on iq_avx2.cpp.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "iq4nl_rows256.hpp"

#include <immintrin.h>

namespace strata::kernels::cpu {
namespace {
struct DotAvxVnni {
    static __m256i dot(__m256i u, __m256i s) { return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s); }
};
}  // namespace

void iq4nl256_down_rows_avxvnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                                float* const* out, int r0, int r1) {
    rows256::iq4nl_down_rows<DotAvxVnni>(w, row_bytes, n, hq, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
