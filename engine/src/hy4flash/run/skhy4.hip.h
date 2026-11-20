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
#if HY4_MMA && !HY4_CUDA
#include "sk_int8_rdna4.hip.h"
#endif
#include "skmoe.hip.h"

namespace hy4 {

namespace skd {
// tuning knob: M <= lin_gemv_max_m() (valid up to 16) uses the bandwidth GEMV, whose per-row result does not depend on M (speculative verify windows, DFly2)
inline int& lin_gemv_max_m() { static int v = 4; return v; }
}  // namespace skd

inline void linear_h(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (fmt == FMT_Q8ROW) linear_h_ref(s, fmt, W, N, K, X, Y, M);
    else if (M <= skd::lin_gemv_max_m() && M <= 16) linear_h_gemv(s, fmt, W, N, K, X, Y, M);
#if HY4_MMA
    else if (M <= 16) {
#if !HY4_CUDA
        const int i8m = skd::int8_wmma_mode();
        if (i8m == 1) { skd::launch_gemm_s_i8_rdna4<false>(s, fmt, W, N, K, X, Y, M); return; }
        if (i8m == 2) { skd::launch_gemm_s_i8_rdna4<true>(s, fmt, W, N, K, X, Y, M); return; }
#endif
        linear_h_gemm_s(s, fmt, W, N, K, X, Y, M);
    } else {
#if !HY4_CUDA
        const int i8m = skd::int8_wmma_mode();
        if (i8m == 1) { skd::launch_gemm_i8_rdna4<false>(s, fmt, W, N, K, X, Y, M); return; }
        if (i8m == 2) { skd::launch_gemm_i8_rdna4<true>(s, fmt, W, N, K, X, Y, M); return; }
#endif
        linear_h_gemm(s, fmt, W, N, K, X, Y, M);
    }
#else
    else if (M <= 16) linear_h_gemv(s, fmt, W, N, K, X, Y, M);
    else linear_h_ref(s, fmt, W, N, K, X, Y, M);
#endif
}

}  // namespace hy4
