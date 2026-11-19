// skhy4.hip.h - the one include for the Hy4-Flash linear algebra on gfx1151: linear_h (production dispatcher), linear_f32, embed_rows, moe_experts.
//
//   linear_h(stream, fmt, W, N, K, X, Y, M):  Y[m][n] = sum_k W[n][k] X[m][k]
//     S2..S8_K weights (K % 256 == 0, X fp16 rows 16-byte aligned):  M <= 4 -> bandwidth GEMV (skfast), 5..16 -> small-tile WMMA (linear_h_gemm_s), M > 16 -> WMMA GEMM (skwmma)
//     Q8ROW weights: reference kernel (small tensors only)
//   linear_h_ref: the straightforward reference kernels (any shape).
#pragma once
#include "skgemm.hip.h"
#include "skfast.hip.h"
#include "skwmma.hip.h"
#include "skmoe.hip.h"

namespace hy4 {

inline void linear_h(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (fmt == FMT_Q8ROW) linear_h_ref(s, fmt, W, N, K, X, Y, M);
    else if (M <= 4) linear_h_gemv(s, fmt, W, N, K, X, Y, M);
    else if (M <= 16) linear_h_gemm_s(s, fmt, W, N, K, X, Y, M);
    else linear_h_gemm(s, fmt, W, N, K, X, Y, M);
}

}  // namespace hy4
