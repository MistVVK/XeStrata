// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/aux_attn.cpp - see include/strata/image/aux_attn.hpp.
#include "strata/image/aux_attn.hpp"

#include "strata/image/aux_kernels.hpp"

namespace strata::image {

void aux_attn_matrix(prefill::Gemm& gemm, const uint16_t* qh, const uint16_t* kh, const uint16_t* v,
                     int64_t batch, int64_t heads, int64_t lq, int64_t lk, int64_t kdim, int64_t outrows,
                     int64_t outcols, bool channel, float scale, const float* bias, const float* mask,
                     int64_t mask_n, const float* grid, int64_t G, int64_t skip, float* sc, uint16_t* p16,
                     float* o, void* stream) {
    const int64_t B = batch * heads;
    gemm.f16_batch(qh, kh, sc, lq, lk, kdim, B, lq * kdim, lk * kdim, lq * lk);
    aux_softmax_attn16(sc, batch, heads, lq, lk, scale, bias, mask, mask_n, grid, G, skip, p16, stream);
    if (channel)   // the channels attend to each other over the tokens: v . p^T
        gemm.f16_batch(v, p16, o, outrows, lq, lk, B, outrows * lk, lq * lk, outrows * lq);
    else   // a query's attention: p . v (v the values transposed)
        gemm.f16_batch(p16, v, o, lq, outcols, lk, B, lq * lk, outcols * lk, lq * outcols);
}

}  // namespace strata::image