// sk_cublas_cuda.hip.h - Side-stream double-buffered S_K -> FP16 dequant + cuBLAS Tensor Core GEMM for prefill (CUDA only).
#pragma once
#include "sk_cublas_ctx.hip.h"
#include "skwmma.hip.h"

#if HY4_CUDA
namespace hy4 {
namespace skd {

template <int FMT>
__global__ void __launch_bounds__(256) k_dequant_sk_to_fp16(const uint8_t* __restrict__ W, __half* __restrict__ out, int n0, int n_tile, int nsb) {
    if constexpr (FMT == FMT_S1_25) {
        s125_lut_init();
        __syncthreads();
    }
    const int r = blockIdx.x;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sb = blockIdx.y * 8 + wave;
    if (r >= n_tile || sb >= nsb) return;
    constexpr int SB = sk_sb_bytes(FMT);
    const uint8_t* blk = W + ((size_t)(n0 + r) * (size_t)nsb + (size_t)sb) * (size_t)SB;
    const float D = h2f_bits(ld16(blk));
    const int dg = lane >> 3;
    float eff = D * ((float)blk[2 + dg] * (1.0f / 255.0f));
    if constexpr (FMT == FMT_S1_25) eff *= 0.5f + (float)((blk[46 + (dg >> 1)] >> ((dg & 1) * 4)) & 15u) / 15.0f;
    uint4 v4;
    dequant8_h<FMT>(blk, lane, eff, v4);
    reinterpret_cast<uint4*>(out + ((size_t)r * (size_t)nsb + (size_t)sb) * 256)[lane] = v4;
}

inline void launch_dequant_sk_to_fp16(cudaStream_t s, int fmt, const void* W, __half* out, int n0, int n_tile, int K) {
    const int nsb = K >> 8;
    dim3 grid(n_tile, (nsb + 7) / 8);
    dim3 block(256);
    const uint8_t* w = (const uint8_t*)W;
    switch (fmt) {
        case FMT_S1_25: k_dequant_sk_to_fp16<FMT_S1_25><<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S1_5:  k_dequant_sk_to_fp16<FMT_S1_5> <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S2:    k_dequant_sk_to_fp16<FMT_S2>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S3:    k_dequant_sk_to_fp16<FMT_S3>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S4:    k_dequant_sk_to_fp16<FMT_S4>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S5:    k_dequant_sk_to_fp16<FMT_S5>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        case FMT_S6:    k_dequant_sk_to_fp16<FMT_S6>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
        default:        k_dequant_sk_to_fp16<FMT_S8>   <<<grid, block, 0, s>>>(w, out, n0, n_tile, nsb); break;
    }
}

inline bool linear_h_cublas(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (fmt == FMT_Q8ROW || (K & 255) != 0) return false;
    CublasPfState& st = cublas_pf();
    if (!st.ensure(s)) return false;
    const int max_rows = (int)(st.half_elems / (size_t)K);
    if (max_rows < 16) return false;
    const int tile_n = std::min(N, (max_rows >= 64) ? (max_rows & ~63) : max_rows);
    if (cublasSetStream(st.handle, s) != CUBLAS_STATUS_SUCCESS) return false;
    const float alpha = 1.0f, beta = 0.0f;
    int b_cur = st.cur;
    int n0 = 0;
    int nt = std::min(tile_n, N);
    cudaStreamWaitEvent(st.side_st, st.ev_free[b_cur], 0);
    launch_dequant_sk_to_fp16(st.side_st, fmt, W, st.buf(b_cur), n0, nt, K);
    cudaEventRecord(st.ev_ready[b_cur], st.side_st);
    while (n0 < N) {
        const int next_n0 = n0 + nt;
        const int b_next = b_cur ^ 1;
        if (next_n0 < N) {
            const int next_nt = std::min(tile_n, N - next_n0);
            cudaStreamWaitEvent(st.side_st, st.ev_free[b_next], 0);
            launch_dequant_sk_to_fp16(st.side_st, fmt, W, st.buf(b_next), next_n0, next_nt, K);
            cudaEventRecord(st.ev_ready[b_next], st.side_st);
        }
        cudaStreamWaitEvent(s, st.ev_ready[b_cur], 0);
        cublasStatus_t cst = cublasGemmEx(
            st.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            nt, M, K,
            &alpha,
            st.buf(b_cur), CUDA_R_16F, K,
            X, CUDA_R_16F, K,
            &beta,
            Y + n0, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(st.ev_free[b_cur], s);
        if (cst != CUBLAS_STATUS_SUCCESS) return false;
        n0 = next_n0;
        nt = std::min(tile_n, N - n0);
        b_cur = b_next;
    }
    st.cur = b_cur;
    return true;
}

}  // namespace skd
}  // namespace hy4
#endif  // HY4_CUDA
