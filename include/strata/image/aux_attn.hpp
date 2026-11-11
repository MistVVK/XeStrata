// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/aux_attn.hpp - the auxiliary models' attention on the GPU's matrix engines (a Gemm Q . K^T,
// the softmax, a Gemm for the output), the part SAM 3's grid attention, Florence-2's channel attention and
// qi_sdpa_ex's biased large shapes share (src/image/aux_attn.cpp).  The scores are held in the caller's scratch.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>

namespace strata::image {

/// A batched attention on the matrix engines.  The scores are qh . kh^T (qh [B][lq][kdim], kh [B][lk][kdim],
/// B = batch * heads), the softmax (aux_softmax_attn16's terms), then the output:
///   channel = false: o [B][lq][outcols] = p . v, v the values transposed [B][outcols][lk] (a query's attention);
///   channel = true:  o [B][outrows][lq] = v . p^T, v [B][outrows][lk] (Florence-2's channel attention).
/// sc [B][lq][lk] FP32, p16 [B][lq][lk] FP16 and o FP32 are the caller's scratch.
void aux_attn_matrix(prefill::Gemm& gemm, const uint16_t* qh, const uint16_t* kh, const uint16_t* v,
                     int64_t batch, int64_t heads, int64_t lq, int64_t lk, int64_t kdim, int64_t outrows,
                     int64_t outcols, bool channel, float scale, const float* bias, const float* mask,
                     int64_t mask_n, const float* grid, int64_t G, int64_t skip, float* sc, uint16_t* p16,
                     float* o, void* stream);

}  // namespace strata::image