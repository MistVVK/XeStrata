// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_conv.hpp - the VAEs' convolutions through a GPU maker's library where the build has one for the GPU
// (oneDNN on Intel GPUs, cuDNN on NVIDIA GPUs in contrib-llvm), false where it does not take the convolution (the
// engine's im2col and products then).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::image::conv {

/// y [OH][OW][O] (FP32) = the k x k convolution of stride `stride` of x [H][W][C] (FP16) by w [O][k][k][C] (FP16)
/// plus `bias` [O] (null: none), zero padding pt, pl before the rows and columns and pb, pr after them; `add`: added to
/// y (y += ...) instead of written.
bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, float* y);

/// The same through cuDNN (contrib-llvm, NVIDIA GPUs): stride 1, padding k / 2.
bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, float* y);

}  // namespace strata::image::conv
