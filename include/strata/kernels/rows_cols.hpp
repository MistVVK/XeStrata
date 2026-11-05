// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/rows_cols.hpp - a work-item a (row, column), whatever the number of rows (device code: SYCL
// files only).
//
// A GPU's first dimension takes 65,535 work-groups at most (CUDA's and HIP's grid y; the RX 9060 XT failed the launch
// with SIGFPE past it), and a prompt's or an image's rows pass that.  Up to it the two-dimensional range, as the
// kernels always launched; past it one dimension, the row and column out of the index by a division.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels {

constexpr int64_t kMaxFirstDim = 65535;

/// f(row, column) for rows x cols work-items
template <class F>
sycl::event for_rows_cols(sycl::queue& q, int64_t rows, int64_t cols, const F& f) {
    if (rows <= kMaxFirstDim)
        return q.parallel_for(sycl::range<2>((size_t) rows, (size_t) cols),
                              [=](sycl::id<2> i) { f((int64_t) i[0], (int64_t) i[1]); });
    return q.parallel_for(sycl::range<1>((size_t) (rows * cols)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / cols;
        f(r, (int64_t) i[0] - r * cols);
    });
}

}  // namespace strata::kernels
