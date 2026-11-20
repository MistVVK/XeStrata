// sk_cublas_ctx.hip.h - Shared side-stream double-buffered 64 MiB FP16 scratch + cuBLAS handle for prefill (CUDA only).
// Kept separate from skwmma.hip.h / skgemm.hip.h so both skhy4.hip.h and mla_attn.hip.cpp share one singleton with zero duplicate kernels.
#pragma once
#include "hy4_gpu.h"

#if HY4_CUDA
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace hy4 {
namespace skd {

inline int& cublas_pf_mode() {
    static int v = getenv("HY4_CUBLAS_PF") ? atoi(getenv("HY4_CUBLAS_PF")) : 1;
    return v;
}
inline int& cublas_pf_min_m() {
    static int v = getenv("HY4_CUBLAS_MIN_M") ? atoi(getenv("HY4_CUBLAS_MIN_M")) : 32;
    return v;
}

struct CublasPfState {
    cublasHandle_t handle = nullptr;
    cudaStream_t side_st = nullptr;
    cudaEvent_t ev_ready[2] = {nullptr, nullptr};
    cudaEvent_t ev_free[2] = {nullptr, nullptr};
    __half* base = nullptr;      // contiguous 64 MiB (2 x 32 MiB) FP16 scratch (or 2 x 16 MiB if VRAM is tight)
    size_t half_elems = 0;       // elements per half-buffer (buf[0] = base, buf[1] = base + half_elems)
    size_t total_bytes = 0;
    int cur = 0;
    int dev = -1;
    bool tried = false;
    bool ok = false;

    bool ensure(cudaStream_t s) {
        if (cublas_pf_mode() <= 0) return false;
        int d = 0;
        if (cudaGetDevice(&d) != cudaSuccess) return false;
        if (tried && d == dev) return ok;
        if (tried && d != dev) return false;  // primary GPU only
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(s, &cap_st) == cudaSuccess && cap_st != cudaStreamCaptureStatusNone) return false;
        tried = true;
        dev = d;
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return false;
        cudaDeviceProp dp{};
        const bool is_uma = (cudaGetDeviceProperties(&dp, d) == cudaSuccess && dp.integrated != 0);
        if (is_uma && !getenv("HY4_CUBLAS_PF")) {
            fprintf(stderr, "[cublas-pf] auto-disabled on integrated UMA GPU (fused skwmma avoids LPDDR5X round-trip; set HY4_CUBLAS_PF=1 to override)\n");
            return false;
        }
        size_t want_bytes = getenv("HY4_CUBLAS_SCRATCH_MB") ? ((size_t)std::max(16, atoi(getenv("HY4_CUBLAS_SCRATCH_MB"))) << 20) : ((size_t)64 << 20);
        if (!is_uma && fr < want_bytes + ((size_t)96 << 20)) want_bytes = (size_t)32 << 20;
        if (!is_uma && fr < want_bytes + ((size_t)64 << 20)) {
            fprintf(stderr, "[cublas-pf] disabled: low free VRAM (%.0f MiB)\n", fr / 1048576.0);
            return false;
        }
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            handle = nullptr;
            return false;
        }
        (void)cublasSetMathMode(handle, dp.major == 7 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
        if (cudaMalloc((void**)&base, want_bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            cublasDestroy(handle);
            handle = nullptr;
            return false;
        }
        if (cudaStreamCreateWithFlags(&side_st, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ready[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ready[1], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_free[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_free[1], cudaEventDisableTiming) != cudaSuccess) {
            cudaFree(base); base = nullptr;
            cublasDestroy(handle); handle = nullptr;
            return false;
        }
        total_bytes = want_bytes;
        half_elems = (want_bytes / sizeof(__half)) / 2;
        ok = true;
        // Warm up cuBLAS Tensor Core kernel loading once at init so Layer 0 prefill pays zero cold-start latency
        (void)cublasSetStream(handle, s);
        const float one = 1.0f, zero = 0.0f;
        (void)cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 16, 16, 16, &one, base, CUDA_R_16F, 16, base + 256, CUDA_R_16F, 16, &zero, (float*)(base + 512), CUDA_R_32F, 16, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        (void)cudaStreamSynchronize(s);
        fprintf(stderr, "[cublas-pf] side-stream double-buffered dequant + cuBLAS enabled (2 x %.0f MiB scratch, M >= %d)\n",
                (total_bytes / 2) / 1048576.0, cublas_pf_min_m());
        return true;
    }
    __half* buf(int b) const { return base + (size_t)b * half_elems; }
    void release_full(cudaStream_t s) {
        cudaEventRecord(ev_free[0], s);
        cudaEventRecord(ev_free[1], s);
    }
};

inline CublasPfState& cublas_pf() {
    static CublasPfState st;
    return st;
}

}  // namespace skd
}  // namespace hy4
#endif  // HY4_CUDA
