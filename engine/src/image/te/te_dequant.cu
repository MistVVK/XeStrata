// src/image/te/te_dequant.cu - GGUF block formats -> f16 rows on CUDA / HIP (include/strata/image/te_dequant.hpp).
//
// One thread decodes 8 consecutive values (a quarter of a 32-value group; a K-quant / IQ4_XS super-block has 8 groups,
// a Q8_0 block is one) and writes them with one 16-byte store, so a warp writes 512 contiguous bytes.  Every multiply and subtract is __fmul_rn / __fsub_rn: the compilers would
// otherwise fuse "d1 * q - m1" into one FMA, which rounds once instead of twice and gives other f16 bits than ggml's
// (and dequant.hpp's) f32 values.  The f16 rounding is f16_bits.hpp's (round to nearest even), not __float2half.
#include "strata/image/te_dequant.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#else
#include <cuda_runtime.h>
#endif

namespace strata::image {
namespace {

using strata::kernels::f16_from_f32;

__constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float h2f_at(const uint8_t* p) {
    return strata::kernels::f32_from_f16((uint16_t) (p[0] | (p[1] << 8)));
}

// get_scale_min_k4 (ggml-quants.c), j = 0..7
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ void put8(uint16_t* out, const float* v) {     // 16-byte store: consecutive threads, consecutive 16 B
    uint4 o;
    o.x = (uint32_t) f16_from_f32(v[0]) | ((uint32_t) f16_from_f32(v[1]) << 16);
    o.y = (uint32_t) f16_from_f32(v[2]) | ((uint32_t) f16_from_f32(v[3]) << 16);
    o.z = (uint32_t) f16_from_f32(v[4]) | ((uint32_t) f16_from_f32(v[5]) << 16);
    o.w = (uint32_t) f16_from_f32(v[6]) | ((uint32_t) f16_from_f32(v[7]) << 16);
    *reinterpret_cast<uint4*>(out) = o;
}

template <uint32_t TYPE>
__device__ __forceinline__ void decode8(const uint8_t* row, int g, int l0, float* v) {   // values l0 .. l0+7 of group g
    if constexpr (TYPE == TE_Q4_K) {                    // d dmin scales[12] qs[128]; group g: 32 values
        const uint8_t* b = row + (size_t) (g >> 3) * 144;
        const int sub = g & 7, j64 = sub >> 1, hi = sub & 1;
        int sc, m;
        scale_min_k4(sub, b + 4, sc, m);
        const float d1 = __fmul_rn(h2f_at(b), (float) sc), m1 = __fmul_rn(h2f_at(b + 2), (float) m);
        const uint8_t* q = b + 16 + 32 * j64;
#pragma unroll
        for (int i = 0; i < 8; ++i) { const int l = l0 + i; v[i] = __fsub_rn(__fmul_rn(d1, (float) (hi ? (q[l] >> 4) : (q[l] & 0x0F))), m1); }
    } else if constexpr (TYPE == TE_Q5_K) {             // d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row + (size_t) (g >> 3) * 176;
        const int sub = g & 7, j64 = sub >> 1, hi = sub & 1;
        int sc, m;
        scale_min_k4(sub, b + 4, sc, m);
        const float d1 = __fmul_rn(h2f_at(b), (float) sc), m1 = __fmul_rn(h2f_at(b + 2), (float) m);
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const int u = 1 << sub;                         // u1 = 1, 4, 16, 64 and u2 = 2, 8, 32, 128
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int l = l0 + i;
            const int q = (hi ? (ql[l] >> 4) : (ql[l] & 0x0F)) + ((qh[l] & u) ? 16 : 0);
            v[i] = __fsub_rn(__fmul_rn(d1, (float) q), m1);
        }
    } else if constexpr (TYPE == TE_Q6_K) {             // ql[128] qh[64] scales[16] (int8) d
        const uint8_t* b = row + (size_t) (g >> 3) * 210;
        const int sub = g & 7, n = sub >> 2, k = sub & 3;   // values 128 n + 32 k + l
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f_at(b + 208);
        const uint8_t* L = ql + ((k & 1) ? 32 : 0);
        const int shift_l = (k & 2) ? 4 : 0, shift_h = 2 * k;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int l = l0 + i;
            const int q = (int) (((L[l] >> shift_l) & 0x0F) | (((qh[l] >> shift_h) & 3) << 4)) - 32;
            v[i] = __fmul_rn(__fmul_rn(d, (float) sc[(l >> 4) + 2 * k]), (float) q);
        }
    } else if constexpr (TYPE == TE_IQ4_XS) {           // d scales_h scales_l[4] qs[128]
        const uint8_t* b = row + (size_t) (g >> 3) * 136;
        const int ib = g & 7;
        const int sh = b[2] | (b[3] << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = __fmul_rn(h2f_at(b), (float) (ls - 32));
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int i = 0; i < 8; ++i) {                 // values 0-15: low nibbles of qs[0..15], 16-31: high nibbles
            const int l = l0 + i;
            v[i] = __fmul_rn(dl, (float) kIq4nl[l < 16 ? (qs[l] & 0x0F) : (qs[l - 16] >> 4)]);
        }
    } else if constexpr (TYPE == TE_Q8_0) {             // d qs[32]
        const uint8_t* b = row + (size_t) g * 34;
        const float d = h2f_at(b);
        const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __fmul_rn((float) q[l0 + i], d);
    } else if constexpr (TYPE == TE_F32) {
        const float* p = reinterpret_cast<const float*>(row) + 32 * (size_t) g;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = p[l0 + i];
    } else if constexpr (TYPE == TE_BF16) {
        const uint8_t* p = row + 64 * (size_t) g;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = strata::kernels::f32_from_bf16((uint16_t) (p[2 * (l0 + i)] | (p[2 * (l0 + i) + 1] << 8)));
    }
}

template <uint32_t TYPE>
__global__ void dequant_kernel(const uint8_t* __restrict__ src, size_t row_bytes, int64_t ne0, int64_t rows,
                               uint16_t* __restrict__ dst) {
    const int64_t ppr = ne0 / 8;                          // pieces of 8 values per row
    const int64_t total = ppr * rows;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / ppr, pc = i % ppr;
        float v[8];
        decode8<TYPE>(src + (size_t) r * row_bytes, (int) (pc >> 2), (int) (pc & 3) * 8, v);
        put8(dst + r * ne0 + pc * 8, v);
    }
}

__global__ void copy_f16_kernel(const uint16_t* __restrict__ src, size_t n, uint16_t* __restrict__ dst) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

template <uint32_t TYPE>
std::string launch(const void* src, size_t rb, int64_t ne0, int64_t rows, uint16_t* dst, cudaStream_t s) {
    const int64_t total = ne0 / 8 * rows;
    const int threads = 256;
    const int64_t want = (total + threads - 1) / threads;
    const int blocks = (int) (want < 65535 * 8 ? want : 65535 * 8);
    dequant_kernel<TYPE><<<blocks, threads, 0, s>>>(static_cast<const uint8_t*>(src), rb, ne0, rows, dst);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? std::string() : std::string("te_dequant: launch: ") + cudaGetErrorString(e);
}

struct Geo { int elems, bytes; };
bool geo(uint32_t type, Geo& g) {
    switch (type) {
    case TE_F32: g = {1, 4}; return true;
    case TE_F16: g = {1, 2}; return true;
    case TE_BF16: g = {1, 2}; return true;
    case TE_Q8_0: g = {32, 34}; return true;
    case TE_Q4_K: g = {256, 144}; return true;
    case TE_Q5_K: g = {256, 176}; return true;
    case TE_Q6_K: g = {256, 210}; return true;
    case TE_IQ4_XS: g = {256, 136}; return true;
    default: return false;
    }
}

}  // namespace

bool te_type_supported(uint32_t type) {
    Geo g;
    return geo(type, g);
}

size_t te_row_bytes(uint32_t type, int64_t ne0) {
    Geo g;
    if (!geo(type, g) || ne0 <= 0 || ne0 % 32 != 0 || ne0 % g.elems != 0) return 0;   // the device path decodes groups of 32
    return (size_t) (ne0 / g.elems) * (size_t) g.bytes;
}

std::string te_dequant_f16(uint32_t type, const void* d_src, int64_t ne0, int64_t rows, uint16_t* d_dst, void* stream) {
    const size_t rb = te_row_bytes(type, ne0);
    if (rb == 0) return "te_dequant: type " + std::to_string(type) + " with rows of " + std::to_string(ne0) + " values is not supported";
    if (rows <= 0) return std::string();
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (type) {
    case TE_Q4_K: return launch<TE_Q4_K>(d_src, rb, ne0, rows, d_dst, s);
    case TE_Q5_K: return launch<TE_Q5_K>(d_src, rb, ne0, rows, d_dst, s);
    case TE_Q6_K: return launch<TE_Q6_K>(d_src, rb, ne0, rows, d_dst, s);
    case TE_IQ4_XS: return launch<TE_IQ4_XS>(d_src, rb, ne0, rows, d_dst, s);
    case TE_Q8_0: return launch<TE_Q8_0>(d_src, rb, ne0, rows, d_dst, s);
    case TE_F32: return launch<TE_F32>(d_src, rb, ne0, rows, d_dst, s);
    case TE_BF16: return launch<TE_BF16>(d_src, rb, ne0, rows, d_dst, s);
    case TE_F16: {
        const size_t n = (size_t) ne0 * (size_t) rows;
        copy_f16_kernel<<<(unsigned) ((n + 255) / 256 < 65535 * 8 ? (n + 255) / 256 : 65535 * 8), 256, 0, s>>>(
            static_cast<const uint16_t*>(d_src), n, d_dst);
        const cudaError_t e = cudaGetLastError();
        return e == cudaSuccess ? std::string() : std::string("te_dequant: launch: ") + cudaGetErrorString(e);
    }
    default: return "te_dequant: type " + std::to_string(type) + " is not supported";
    }
}

}  // namespace strata::image
