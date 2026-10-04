// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/dp4a_gemm.hpp - xmx_gemm's products without the matrix engines, for a device or a SYCL runtime that
// reports no FP16 / BF16 joint_matrix combination (intel/llvm 6.2's runtime lists none for the B70).  Both sides are
// quantized to int8 in blocks of 32 values with a float scale each (W as q8_0, X as q8_1 without its sum) while their
// tiles load, and multiplied with DP4a: 4 times the vector engines' FP32 rate, at about 1% relative error.
//
// In a source file of its own, so its kernels are in a device image without joint_matrix: the runtime refuses every
// kernel of an image that needs matrix hardware the device does not report.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::xe {

/// The sub-group size of the kernels written for 16 lanes (this file's, xmx_gemm's gemm_rows): 16 where the device
/// lists it (Intel GPUs), else 32 (the size every GPU the engine accepts has).  The engine drives one device, so the
/// first call's answer holds.
int narrow_sub_group(const sycl::queue& q);

/// A row count measured on the B70 (256 compute units) as the point where a product has tiles enough to fill it,
/// scaled to `q`'s device by its compute units: the tile choices of xmx_gemm and dp4a_gemm.
int64_t fill_rows(const sycl::queue& q, int64_t b70_rows);

/// Y[T, N] (row stride ldy) = X[T, K] . W[N, K]^T, 16-bit inputs (`bf16`: BF16, else FP16), K a multiple of 32.
sycl::event dp4a_gemm(sycl::queue& q, bool bf16, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy);

/// xmx_gemm_grouped's G experts (FP16): rows bounds[e] .. bounds[e + 1] of X and Y (ldy = N) by W + e * w_stride.
sycl::event dp4a_gemm_grouped(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                              const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K);

}  // namespace strata::kernels::xe
