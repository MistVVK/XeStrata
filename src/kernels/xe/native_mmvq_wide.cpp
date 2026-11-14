// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_mmvq_wide.cpp - the native MMVQ's 9..16 columns (a window of 9..16 rows), a source of their
// own so their device code is an image loaded only when a wide window first runs (native_mmvq_kernels.hpp).
#include "native_mmvq_kernels.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_mmvq_wide
#include "strata/kernels/device_arch_tag.hpp"

#include <stdexcept>

namespace strata::kernels {
namespace {
// native_mmvq.cpp's launch_n for NCOLS of 9..16
template <typename F, int NCOLS>
void launch_wide(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, bool row,
                 bool exact) {
    if (!exact) {
        launch_kernel<F, NCOLS, 2, 2>(q, weights, x_q8_1, y, n_in, n_out, (std::size_t(n_out) + 1) / 2);
        return;
    }
    if (row) launch_row<F, NCOLS, 8>(q, weights, x_q8_1, y, n_in, n_out);
    else launch_group<F, NCOLS>(q, weights, x_q8_1, y, n_in, n_out);
}

template <typename F>
void wide(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, bool row,
          bool exact) {
    switch (ncols) {
        case 9: launch_wide<F, 9>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 10: launch_wide<F, 10>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 11: launch_wide<F, 11>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 12: launch_wide<F, 12>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 13: launch_wide<F, 13>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 14: launch_wide<F, 14>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 15: launch_wide<F, 15>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        case 16: launch_wide<F, 16>(q, weights, x_q8_1, y, n_in, n_out, row, exact); break;
        default: throw std::invalid_argument("native MMVQ: the wide window's products take 9 to 16 columns");
    }
}
}  // namespace
}  // namespace strata::kernels

namespace strata::kernels::mmvq_xe {
void mmvq_wide(int fmt, sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
               int ncols, bool row, bool exact) {
    switch (fmt) {
        case kWideFmt<Q5KTraits>: wide<Q5KTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q4KTraits>: wide<Q4KTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q20Traits>: wide<Q20Traits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q3KTraits>: wide<Q3KTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q2KTraits>: wide<Q2KTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q6KTraits>: wide<Q6KTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q6KRowTraits>: wide<Q6KRowTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<IQ4XSTraits>: wide<IQ4XSTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q40Traits>: wide<Q40Traits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q50Traits>: wide<Q50Traits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q80Traits>: wide<Q80Traits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<Q80RowTraits>: wide<Q80RowTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        case kWideFmt<IQ4NLTraits>: wide<IQ4NLTraits>(q, weights, x_q8_1, y, n_in, n_out, ncols, row, exact); break;
        default: throw std::invalid_argument("native MMVQ: a format without wide products");
    }
}
}  // namespace strata::kernels::mmvq_xe
