// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/draft_head.hpp - a draft head's coarse pass: the vocabulary's rows at 2 bits, the few best
// of their products chosen, and those rows taken again from the head itself (the idea of vllm-radiance's int2 draft
// head with an exact rerank, written anew).  A draft only proposes; the verify window decides with the model's own
// head, so a draft head may be approximate: what it costs is acceptance.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The int2 form's group: 128 values with their minimum and their step in FP16.
constexpr int kInt2Group = 128;

/// The bytes of one row of `n_in` values (a positive multiple of kInt2Group): the codes, 4 a byte, then each group's
/// minimum and step.
std::size_t int2_row_bytes(int n_in);

/// Host: `n_rows` rows of `n_in` floats to the int2 form, each group's minimum and step from its least and largest
/// value (rounded to the nearest of the four levels).
void int2_pack_rows(const float* rows, int64_t n_rows, int n_in, uint8_t* out);

/// y[r] = the int2 row r of `w` . x for the `n_out` rows (`x`: `n_in` floats).  Device pointers.
void int2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int64_t n_out, void* stream);

/// The largest topk_max values a call can choose.
constexpr int kTopKMax = 64;

/// The scratch topk needs for `n` values.
std::size_t topk_scratch_bytes(int64_t n, int k);

/// The indices of the `k` (1..kTopKMax) largest of the `n` values in `v`, in no order, into `ids`.  Device pointers;
/// `scratch` holds topk_scratch_bytes(n, k).
void topk(const float* v, int64_t n, int k, int32_t* ids, void* scratch, void* stream);

/// y[0..n) = -1.7e38 (a probability of 0) when the device value *pos is `from` or more, else nothing: a decision the
/// device makes, so a captured graph takes it at every replay.
void int2_pad_from(const int32_t* pos, int32_t from, float* y, int64_t n, void* stream);

/// y[ids[j]] = vals[j] for the `k` values (the chosen rows' products written back over their coarse ones).
void scatter_values(const float* vals, const int32_t* ids, int k, float* y, void* stream);

}  // namespace strata::kernels
