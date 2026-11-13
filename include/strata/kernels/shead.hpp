// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// SHEAD's low-rank shortlist and exact int8 head, after YASAKANI's bdraft head.
#pragma once

#include <cstdint>

namespace strata::kernels {

// The caller owns the device buffers. bt [rank][hidden] is FP32; a [vocab][rank] and w [vocab][hidden] are signed
// int8 with the row scales as and ws. h [rows][hidden] is FP32. Work: q [rows][rank], score [rows][vocab],
// candidates and exact [rows][512], and sort_tmp [2][rows][ceil(vocab / 1024)][512]. Output: top_id and top_value [rows][16], descending with token-id tie breaks.
// The model's .shead must match its tokenizer and drafter. Its rows are not interchangeable with a DFlash2 head.
void shead_top512(const float* h, const float* bt, const int8_t* a, const float* as, const int8_t* w,
                  const float* ws, int rows, int hidden, int rank, int vocab, float* q, float* score,
                  int32_t* candidates, float* exact, int32_t* sort_tmp, int32_t* top_id, float* top_value, void* stream);

}  // namespace strata::kernels
