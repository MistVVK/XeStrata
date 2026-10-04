// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/xmx_gemm.hpp - the prompt path's matrix products on the XMX engines (SYCL joint_matrix).
//
// Y[T, N] (fp32, row stride ldy) = X[T, K] . W[N, K]^T with X and W row-major in FP16 or BF16 and FP32 accumulation.
#pragma once

#include <cstdint>

namespace strata::kernels {

enum class XmxType { f16, bf16 };

/// Whether the compute device runs the XMX kernels for inputs of type `t` (joint_matrix 8 x 16 x 16 with FP32
/// accumulators and 16-wide sub-groups, as it reports them; STRATA_NO_XMX=1 says no).  Without them the products run
/// through DP4a, and other kernels on the matrix engines (the prompt attention) take their own paths.
bool xmx_available(XmxType t);

/// The path xmx_gemm takes for inputs of type `t` on the compute device, for the startup report: "XMX", "joint_matrix"
/// (mma_gemm, the matrix engines a GPU without the XMX kernels reports) or "DP4a".
const char* gemm_path(XmxType t);

/// Whether xmx_gemm takes the product: N a multiple of 16, K of 32, ldy of 16, and X, W, Y 64-byte aligned (the 2D
/// block loads and stores).  The others go to gemm_rows.
bool xmx_gemm_ok(const void* X, const void* W, const float* Y, int64_t N, int64_t K, int64_t ldy) noexcept;

/// `accumulate`: Y += X . W^T (Y read before the product is added, in FP32); BF16 only on the XMX engines (without
/// them the sum goes through gemm_rows).
void xmx_gemm(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream, bool accumulate = false);

/// The products of G experts in one launch (FP16): expert e multiplies rows bounds[e] .. bounds[e + 1] of X (and
/// writes them in Y, ldy = N) by its weights at W + e * w_stride.  `bounds` is on the device; `max_rows` bounds every
/// expert's row count (it sizes the launch).
void xmx_gemm_grouped(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds, int G,
                      int64_t max_rows, int64_t N, int64_t K, void* stream);

/// Y = X . W^T, one sub-group an output: the small products xmx_gemm does not take.
void gemm_rows(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream, bool accumulate = false);

}  // namespace strata::kernels
