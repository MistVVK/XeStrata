// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/mma_gemm.hpp - xmx_gemm's products on the matrix engines of GPUs other than Intel's (NVIDIA's tensor
// cores), through joint_matrix's portable API only: the tile shape and the sub-group size are the ones the device
// reports in its matrix combinations.  xmx_gemm.cpp's kernels use Intel's extensions (checked loads, prefetches, the
// large register file) and run on Intel GPUs; these run where those do not.
//
// In a source file of its own, so its kernels are in a device image apart from the Intel-only ones.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::xe {

/// Whether mma_gemm runs on `q`'s device for 16-bit inputs (`bf16`: BF16, else FP16): one of its tile shapes with FP32
/// accumulators in the device's matrix combinations, at the sub-group size that shape takes.  STRATA_NO_XMX=1 says no
/// (it asks for the products without matrix engines); STRATA_MMA=1 says yes wherever the shape is reported, also
/// where xmx_gemm's kernels would run, to check these on an Intel GPU.
bool mma_usable(const sycl::queue& q, bool bf16);

/// STRATA_MMA=1: take mma_gemm before xmx_gemm's own kernels.
bool mma_forced();

/// Whether mma_usable(q, false) with the FP16 tile shape m x n x k on sg lanes (a kernel written for that shape alone,
/// such as the prompt attention's portable one, asks this).
bool mma_shape_is(const sycl::queue& q, int m, int n, int k, int sg);

/// Y[T, N] (row stride ldy) = X[T, K] . W[N, K]^T (`accumulate`: Y +=), N a multiple of 16, K of 32.
sycl::event mma_gemm(sycl::queue& q, bool bf16, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                     int64_t K, int64_t ldy, bool accumulate);

/// xmx_gemm_grouped's G experts (FP16): rows bounds[e] .. bounds[e + 1] of X and Y (ldy = N) by W + e * w_stride.
sycl::event mma_gemm_grouped(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                             const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K);

}  // namespace strata::kernels::xe
