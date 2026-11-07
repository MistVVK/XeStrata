// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_conv.hpp - the image engine's convolutions (the VAEs', RealESRGAN's) through a GPU maker's library where
// the build has one for the GPU (oneDNN on Intel GPUs, cuDNN on NVIDIA GPUs in contrib-llvm), false where it does not
// take the convolution (the engine's im2col and products then).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::image::conv {

/// Where a convolution reads and writes beyond the plain form (the VAEs'): RealESRGAN's dense blocks read a prefix of a
/// wider tensor's channels and write their output into its next channels, LeakyReLU'd, in FP16.
struct Form {
    int64_t ldx = 0;       // x's elements between two pixels (0: C)
    int64_t ldy = 0;       // y's (0: O)
    bool y16 = false;      // y FP16 (uint16_t bits) rather than FP32
    float leaky = -1.0f;   // 0 or more: LeakyReLU of that slope after the bias (not with `add`)
    bool gelu = false;     // GELU (erf's) after the bias (not with `add` or `leaky`; oneDNN alone)
};

/// y [OH][OW][O] (FP32) = the k x k convolution of stride `stride` of x [H][W][C] (FP16) by w [O][k][k][C] (FP16)
/// plus `bias` [O] (null: none), zero padding pt, pl before the rows and columns and pb, pr after them; `add`: added to
/// y (y += ...) instead of written.
bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, float* y);
/// The same in `f`'s form: y is FP32 or FP16 by f.y16.
bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, void* y, const Form& f);

/// The same through cuDNN (contrib-llvm, NVIDIA GPUs): stride 1, padding k / 2.
bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, float* y);
/// The same in `f`'s form.
bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, void* y, const Form& f);

}  // namespace strata::image::conv
