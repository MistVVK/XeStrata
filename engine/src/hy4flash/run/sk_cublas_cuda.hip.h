// sk_cublas_cuda.hip.h - Double-buffered side-stream S_K -> FP16 dequant + cublasGemmEx (Tensor Cores) for dense prefill (M >= 128).
// Compiled when -DHY4_CUBLAS is passed (link with -lcublas).
// Uses two reusable 64 MiB FP16 scratch buffers on a side stream with event ordering so dequantization of tile t+1
// (and the next dense layer) overlaps cuBLAS Tensor Core execution of tile t with zero bubble.
#pragma once
#if HY4_CUDA && defined(HY4_CUBLAS)
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace hy4 {
namespace skd {

// Coalesced S_K -> FP16 dequantizer: 1 warp per (row, super-block of 256 weights), 8 weights/lane -> uint4 store (16 B).
// Matches skwmma.hip.h dequant8_h<FMT> rounding bit-for-bit.
template <int FMT>
__global__ void __launch_bounds__(256) k_dequant_sk_f16(const uint8_t* __restrict__ W, __half* __restrict__ dst, int N_tile, int K, int row0) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int nsb = K >> 8;
    const int warp_global = (blockIdx.x << 3) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int total_warps = N_tile * nsb;
    if constexpr (FMT == FMT_S1_25) {
        s125_lut_init();
        __syncthreads();
    }
    if (warp_global >= total_warps) return;
    const int lr = warp_global / nsb;
    const int sb = warp_global - lr * nsb;
    const uint8_t* blk = W + ((size_t)(row0 + lr) * nsb + sb) * SB;
    const float eff = sk_eff<FMT>(blk, lane >> 3);
    uint4 out;
    dequant8_h<FMT>(blk, lane, eff, out);
    *(uint4*)(dst + ((size_t)lr * K + (size_t)sb * 256 + lane * 8)) = out;
}

struct CublasPrefillCtx {
    cublasHandle_t handle = nullptr;
    cudaStream_t side_st = nullptr;
    cudaEvent_t ev_ready[2] = {};
    cudaEvent_t ev_free[2] = {};
    __half* scratch[2] = {};
    size_t buf_elems = 0;  // per scratch buffer (in __half elements; 32M = 64 MiB)
    int cur = 0;
    int dev_id = -1;
    bool inited = false;
    bool ok = false;

    bool ensure_init() {
        int cur_dev = 0;
        cudaGetDevice(&cur_dev);
        if (inited && cur_dev == dev_id) return ok;
        if (inited) destroy();
        inited = true;
        dev_id = cur_dev;
        ok = false;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { handle = nullptr; return false; }
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        if (cudaStreamCreateWithFlags(&side_st, cudaStreamNonBlocking) != cudaSuccess) { destroy(); return false; }
        for (int i = 0; i < 2; ++i) {
            if (cudaEventCreateWithFlags(&ev_ready[i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&ev_free[i], cudaEventDisableTiming) != cudaSuccess) { destroy(); return false; }
        }
        // Default 64 MiB per buffer (33,554,432 halfs); fall back to 32 MiB if VRAM is tight
        size_t want_mb = getenv("HY4_CUBLAS_SCRATCH_MB") ? (size_t)std::max(16, atoi(getenv("HY4_CUBLAS_SCRATCH_MB"))) : 64u;
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if (fr < (want_mb * 2 + 256) * (1ull << 20)) want_mb = 32u;
        if (fr < (want_mb * 2 + 128) * (1ull << 20)) { destroy(); return false; }
        buf_elems = (want_mb << 20) / sizeof(__half);
        for (int i = 0; i < 2; ++i) {
            if (cudaMalloc(&scratch[i], buf_elems * sizeof(__half)) != cudaSuccess) {
                (void)cudaGetLastError();
                destroy();
                return false;
            }
        }
        ok = true;
        return true;
    }

    void destroy() {
        for (int i = 0; i < 2; ++i) {
            if (scratch[i]) { cudaFree(scratch[i]); scratch[i] = nullptr; }
            if (ev_ready[i]) { cudaEventDestroy(ev_ready[i]); ev_ready[i] = nullptr; }
            if (ev_free[i]) { cudaEventDestroy(ev_free[i]); ev_free[i] = nullptr; }
        }
        if (side_st) { cudaStreamDestroy(side_st); side_st = nullptr; }
        if (handle) { cublasDestroy(handle); handle = nullptr; }
        buf_elems = 0;
        ok = false;
        inited = false;
    }
    ~CublasPrefillCtx() { destroy(); }
};

inline CublasPrefillCtx& cublas_prefill_ctx() {
    static CublasPrefillCtx ctx;
    return ctx;
}

inline int cublas_prefill_min_m() {
    static int v = -1;
    if (v < 0) {
        if (const char* e = getenv("HY4_CUBLAS_PREFILL")) {
            int x = atoi(e);
            v = (x <= 0) ? 0 : (x == 1 ? 128 : x);
        } else {
            v = 128;  // ON by default for M >= 128 when compiled with -DHY4_CUBLAS
        }
    }
    return v;
}

inline bool launch_dequant_sk_f16(cudaStream_t s, int fmt, const uint8_t* W, __half* dst, int N_tile, int K, int row0) {
    const int nsb = K >> 8;
    const int total_warps = N_tile * nsb;
    dim3 grid((total_warps + 7) / 8), block(256);
    switch (fmt) {
        case FMT_S1_25: k_dequant_sk_f16<FMT_S1_25><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S1_5:  k_dequant_sk_f16<FMT_S1_5><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S2:    k_dequant_sk_f16<FMT_S2><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S3:    k_dequant_sk_f16<FMT_S3><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S4:    k_dequant_sk_f16<FMT_S4><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S5:    k_dequant_sk_f16<FMT_S5><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S6:    k_dequant_sk_f16<FMT_S6><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        case FMT_S8:    k_dequant_sk_f16<FMT_S8><<<grid, block, 0, s>>>(W, dst, N_tile, K, row0); return true;
        default: return false;
    }
}

// Y[M, N] (fp32 row-major) = X[M, K] (fp16 row-major) @ W[N, K]^T (S_K).
// Row-tiles W into two 64 MiB FP16 buffers on side_st while cublasGemmEx executes on compute stream s.
inline bool linear_h_cublas_prefill(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    const int min_m = cublas_prefill_min_m();
    if (min_m <= 0 || M < min_m || (K & 255) != 0) return false;
    if (fmt == FMT_Q8ROW) return false;
    CublasPrefillCtx& ctx = cublas_prefill_ctx();
    if (!ctx.ensure_init()) return false;
    int max_rows = (int)(ctx.buf_elems / (size_t)K);
    max_rows = (max_rows / 64) * 64;
    if (max_rows < 64) return false;
    cublasSetStream(ctx.handle, s);
    const float alpha = 1.0f, beta = 0.0f;
    const uint8_t* Wb = (const uint8_t*)W;
    // Queue row tiles with double-buffered side-stream dequantization:
    // For each tile: side_st waits for ev_free[b], dequants into scratch[b], records ev_ready[b];
    // compute stream s waits for ev_ready[b], runs cublasGemmEx, records ev_free[b].
    for (int r0 = 0; r0 < N; r0 += max_rows) {
        const int nr = std::min(max_rows, N - r0);
        const int b = ctx.cur;
        ctx.cur ^= 1;
        cudaStreamWaitEvent(ctx.side_st, ctx.ev_free[b], 0);
        if (!launch_dequant_sk_f16(ctx.side_st, fmt, Wb, ctx.scratch[b], nr, K, r0)) return false;
        cudaEventRecord(ctx.ev_ready[b], ctx.side_st);
        cudaStreamWaitEvent(s, ctx.ev_ready[b], 0);
        cublasStatus_t st = cublasGemmEx(
            ctx.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            nr, M, K,
            &alpha,
            ctx.scratch[b], CUDA_R_16F, K,
            X, CUDA_R_16F, K,
            &beta,
            Y + r0, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(ctx.ev_free[b], s);
        if (st != CUBLAS_STATUS_SUCCESS) {
            // Fallback on cuBLAS failure
            return false;
        }
    }
    return true;
}

}  // namespace skd
}  // namespace hy4
#endif  // HY4_CUDA && defined(HY4_CUBLAS)
