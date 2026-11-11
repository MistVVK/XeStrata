// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_sdpa_ex.hpp - attention of any shape the image tools' auxiliary models ask for
// (src/image/qi_sdpa_ex.cpp): SAM 3's windows, its text tower and DETR layers, Florence-2's DaViT and BART, BiRefNet's
// Swin.
#pragma once

#include <cstdint>

namespace strata::image {

/// `batch` independent attentions of `heads` heads, each lq query rows over lk keys; head h's channels h * hd .. of a
/// row (hd up to 128).  The strides are in elements: a row's and a batch's for q, k, v and out (FP16).  Batch b reads
/// the keys of batch b / kv_div (a decoder's beams share the encoder's keys).  Masks, each optional: key_len[b] keys
/// at most (padding); `causal`, row i the keys up to i + causal_off; additive terms `bias` [heads][lq][lk] (a relative
/// position bias, every batch's) and `mask` [mask_n][lq][lk] (batch b's mask_n's b % mask_n: the shifted windows').
/// A row that reads no key comes out zero.
struct QiSdpa {
    const uint16_t *q = nullptr, *k = nullptr, *v = nullptr;
    uint16_t* out = nullptr;
    int64_t batch = 1, heads = 1, lq = 0, lk = 0, hd = 0;
    int64_t q_row = 0, q_batch = 0, k_row = 0, k_batch = 0, v_row = 0, v_batch = 0, o_row = 0, o_batch = 0;
    int64_t kv_div = 1;
    float scale = 1.f;
    bool causal = false;
    int64_t causal_off = 0;
    const int32_t* key_len = nullptr;
    const float* bias = nullptr;
    const float* mask = nullptr;
    int64_t mask_n = 1;
};

/// Through qi_sdpa (a GPU maker's kernel) where its contract takes the shape (no mask or bias, rows laid out one
/// after another, the scale 1 / sqrt(hd); batches whose keys differ one call each), else the engine's own kernels; a
/// needed mask is never dropped for a faster path.
void qi_sdpa_ex(const QiSdpa& a, void* stream);
/// The library part alone: false where qi_sdpa does not take the shape (the parity probe's check of both paths).
bool qi_sdpa_ex_library(const QiSdpa& a, void* stream);
/// The engine's own kernels alone (the matrix-engine paths where they take the shape, else the FMA kernels): the
/// parity probe's check of the own path forced, without the library's choice.
void qi_sdpa_ex_own(const QiSdpa& a, void* stream);

}  // namespace strata::image
