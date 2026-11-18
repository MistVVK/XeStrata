// include/strata/quant/sk_cuda.cuh - Native CUDA/HIP warp-level S_K dequant & DP4A dot kernels for Hy4-Flash
// Matches tools/squant/codecs.py & src/quant/sk_codecs.cpp bit-for-bit on [1536, 1536] LatentMoE experts.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_fp16.h>

namespace strata::quant::sk_gpu {

constexpr int QK_S  = 256;
constexpr int SUB_G = 64;

__device__ __constant__ int8_t kS125CodebookDev[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1,  1}, {-1, -1,  1, -1}, {-1, -1,  1,  1},
    {-1,  1, -1, -1}, {-1,  1, -1,  1}, {-1,  1,  1, -1}, {-1,  1,  1,  1},
    { 1, -1, -1, -1}, { 1, -1, -1,  1}, { 1, -1,  1, -1}, { 1, -1,  1,  1},
    { 1,  1, -1, -1}, { 1,  1, -1,  1}, { 1,  1,  1, -1}, { 1,  1,  1,  1},
    { 2,  0,  0,  0}, {-2,  0,  0,  0}, { 0,  2,  0,  0}, { 0, -2,  0,  0},
    { 0,  0,  2,  0}, { 0,  0, -2,  0}, { 0,  0,  0,  2}, { 0,  0,  0, -2},
    { 0,  0,  0,  0}, { 2,  2,  0,  0}, {-2, -2,  0,  0}, { 0,  0,  2,  2},
    { 0,  0, -2, -2}, { 2, -2,  0,  0}, { 0,  0,  2, -2}, { 2,  0,  2,  0},
};

struct BlockQ8_1_G64_Dev {
    int8_t qs[64];
    float d;
    float s;
};

__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, c);
#else
    int res = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int8_t va = static_cast<int8_t>((a >> (8 * i)) & 0xFF);
        int8_t vb = static_cast<int8_t>((b >> (8 * i)) & 0xFF);
        res += static_cast<int>(va) * static_cast<int>(vb);
    }
    return res;
#endif
}

// Warp-per-row GEMV kernel for S4_K (Band 0 Hot experts in VRAM on RTX 5070 / 5080 / 5090 / PRO 6000!)
// Each warp (32 lanes) processes 1 row of ncols=1536 (6 super-blocks of 256 weights).
__global__ void gemv_s4k_q8_1_kernel(
    const uint8_t* __restrict__ weights,
    const BlockQ8_1_G64_Dev* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (row >= nrows) return;

    const int n_sb = ncols / QK_S;
    const uint8_t* row_ptr = weights + static_cast<size_t>(row) * (n_sb * 134);
    float acc = 0.0f;

    for (int s = 0; s < n_sb; ++s) {
        const uint8_t* blk = row_ptr + s * 134;
        __half D_h;
        memcpy(&D_h, blk, 2);
        const float D = __half2float(D_h);
        const uint8_t* u = blk + 2;
        const uint8_t* pl = blk + 6;  // 128 B = 256 x 4-bit nibbles

        // Each of 32 lanes handles 8 weights = 4 packed bytes (2 bytes = 4 nibbles x 2)
        const int g = lane >> 3;          // sub-group 0..3 (8 lanes per 64-weight sub-group)
        const int in_g = (lane & 7) * 8;  // offset 0,8,...,56 within sub-group
        const int byte_off = (g * 32) + (in_g >> 1);
        const BlockQ8_1_G64_Dev& xb = x[s * 4 + g];
        const float eff = D * (static_cast<float>(u[g]) / 255.0f);

        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            uint8_t b0 = pl[byte_off + 2 * k + 0];
            uint8_t b1 = pl[byte_off + 2 * k + 1];
            int w4 = (int)(b0 & 0x0F) | ((int)(b0 >> 4) << 8) | ((int)(b1 & 0x0F) << 16) | ((int)(b1 >> 4) << 24);
            int x4;
            memcpy(&x4, xb.qs + in_g + 4 * k, 4);
            isum = dp4a_i8(w4, x4, isum);
        }
        // Warp-reduce across the 8 lanes belonging to sub-group g
        #pragma unroll
        for (int off = 4; off >= 1; off >>= 1) {
            isum += __shfl_xor_sync(0xFFFFFFFFu, isum, off);
        }
        if ((lane & 7) == 0) {
            acc += eff * (xb.d * static_cast<float>(isum) - 7.5f * xb.s);
        }
    }
    // Reduce across the 4 sub-group leaders (lanes 0, 8, 16, 24)
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 8);
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 16);
    if (lane == 0) out[row] = acc;
}

}  // namespace strata::quant::sk_gpu
#endif
