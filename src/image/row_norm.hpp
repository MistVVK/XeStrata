// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/row_norm.hpp - a row's mean and deviation by one work-group, shared by the auxiliary networks' norms and
// the depthwise convolution's fused norm.
#pragma once

#include "strata/kernels/row_stats.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace strata::image::rows {

// A row's norm by a work-group of L items, each holding its kRowVals channels c = l + k L in registers: L about a
// quarter of the channels (32 to 256), so a narrow row is not 256 items of one channel and two group reductions (the
// B70's Florence-2 large vision, 2026-11-07: its 256-channel rows' fused depthwise convolution and norm 476 us at
// 256 items, at 170 GB/s).  Rows wider than kRowVals x 256 channels take the loops over memory (row_vals() 0).
constexpr int64_t kRowVals = 8;
inline int64_t row_items(int64_t C) { return std::clamp<int64_t>((C / 4 + 31) / 32 * 32, 32, 256); }
inline bool row_vals(int64_t C) { return C <= kRowVals * row_items(C); }

// The mean and the inverse deviation of a row's values held as above (two passes: the variance about the mean)
template <int N>
inline sycl::float2 row_stats(const sycl::nd_item<N>& it, const float (&v)[kRowVals], int64_t l, int64_t L, int64_t C,
                              float eps) {
    const auto st = kernels::row_stats::calculate<kernels::row_stats::Kind::variance, false, kRowVals>(
        l, L, C, eps, [&](int64_t c) { return v[(c - l) / L]; },
        [&](float x) { return sycl::reduce_over_group(it.get_group(), x, sycl::plus<float>()); });
    return {st.mean, st.inv};
}

}  // namespace strata::image::rows
