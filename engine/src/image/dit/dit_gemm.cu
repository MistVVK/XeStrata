// src/image/dit/dit_gemm.cu - see include/strata/image/dit_gemm.hpp.
#include "strata/image/dit_gemm.hpp"

#include "strata/image/te_dequant.hpp"
#include "strata/kernels/f16_bits.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#include "strata/hip_compat/cublas_v2.h"
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <mutex>

namespace strata::image {
namespace {

// rows of f32 (stride ldx) -> packed f16 rows, round to nearest even (f16_bits.hpp)
__global__ void rows_to_f16(const float* __restrict__ x, int64_t ldx, int64_t K, int64_t n, uint16_t* __restrict__ out) {
    const int64_t total = n * K;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / K, c = i % K;
        out[i] = strata::kernels::f16_from_f32(x[r * ldx + c]);
    }
}

struct State {
    std::mutex mu;
    int device = -1;
    cublasHandle_t blas = nullptr;
    uint16_t* w16 = nullptr;
    size_t w_cap = 0;
    uint16_t* x16 = nullptr;
    size_t x_cap = 0;
};
State& st() {
    static State s;
    return s;
}

std::string grow(uint16_t*& p, size_t& cap, size_t elems) {
    if (elems <= cap) return std::string();
    if (p) cudaFree(p);
    p = nullptr;
    cap = 0;
    const cudaError_t e = cudaMalloc((void**) &p, elems * sizeof(uint16_t));
    if (e != cudaSuccess) return std::string("dit_gemm: cudaMalloc: ") + cudaGetErrorString(e);
    cap = elems;
    return std::string();
}

}  // namespace

namespace {

// x (f32 rows, converted here) or x16 (f16 rows, used in place)
std::string gemm_impl(void* stream, uint32_t type, const void* w, int64_t K, int64_t N, const float* x,
                      const uint16_t* x16, int64_t ldx, int64_t n, float* y, int64_t ldy) {
    if (n <= 0) return std::string();
    State& s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    int dev = 0;
    cudaGetDevice(&dev);
    if (s.blas == nullptr || s.device != dev) {
        if (s.blas) cublasDestroy(s.blas);
        s.blas = nullptr;
        if (cublasCreate(&s.blas) != CUBLAS_STATUS_SUCCESS) return "dit_gemm: cannot create the BLAS handle";
        s.device = dev;
    }
    std::string err;
    if (type != TE_F16) err = grow(s.w16, s.w_cap, (size_t) N * (size_t) K);
    if (err.empty() && x16 == nullptr) err = grow(s.x16, s.x_cap, (size_t) n * (size_t) K);
    if (!err.empty()) return err;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    const uint16_t* wa = static_cast<const uint16_t*>(w);
    if (type != TE_F16) {
        err = te_dequant_f16(type, w, K, N, s.w16, stream);
        if (!err.empty()) return "dit_gemm: " + err;
        wa = s.w16;
    }
    const uint16_t* xa = x16;
    int64_t lda = ldx;
    if (x16 == nullptr) {
        const int64_t total = n * K;
        const int64_t blocks = (total + 255) / 256;
        rows_to_f16<<<(unsigned) (blocks < 65535 * 8 ? blocks : 65535 * 8), 256, 0, cs>>>(x, ldx, K, n, s.x16);
        if (cudaGetLastError() != cudaSuccess) return "dit_gemm: the f16 conversion kernel failed";
        xa = s.x16;
        lda = K;
    }
    if (cublasSetStream(s.blas, cs) != CUBLAS_STATUS_SUCCESS) return "dit_gemm: cublasSetStream failed";
    const float one = 1.f, zero = 0.f;
    const auto r = cublasGemmEx(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &one, wa, CUDA_R_16F,
                                (int) K, xa, CUDA_R_16F, (int) lda, &zero, y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) return "dit_gemm: GemmEx status " + std::to_string((int) r);
    return std::string();
}

}  // namespace

std::string dit_gemm(void* stream, uint32_t type, const void* w, int64_t K, int64_t N, const float* x, int64_t ldx,
                     int64_t n, float* y, int64_t ldy) {
    return gemm_impl(stream, type, w, K, N, x, nullptr, ldx, n, y, ldy);
}

std::string dit_gemm_x16(void* stream, uint32_t type, const void* w, int64_t K, int64_t N, const uint16_t* x16,
                         int64_t ldx, int64_t n, float* y, int64_t ldy) {
    return gemm_impl(stream, type, w, K, N, nullptr, x16, ldx, n, y, ldy);
}

size_t dit_gemm_buffer_bytes() {
    State& s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    return (s.w_cap + s.x_cap) * sizeof(uint16_t);
}

void dit_gemm_release() {
    State& s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.w16) cudaFree(s.w16);
    if (s.x16) cudaFree(s.x16);
    if (s.blas) cublasDestroy(s.blas);
    s.w16 = s.x16 = nullptr;
    s.w_cap = s.x_cap = 0;
    s.blas = nullptr;
    s.device = -1;
}

}  // namespace strata::image
