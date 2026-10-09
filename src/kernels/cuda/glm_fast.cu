// src/kernels/cuda/glm_fast.cu - see include/strata/kernels/glm_fast.hpp.
//
// The dot products are llama.cpp's vecdotq (ggml/src/ggml-cuda/vecdotq.cuh, MIT, third_party/ggml/LICENSE),
// the same expressions iq_kernels.cu and native_mmvq.cu transcribe; the block structs and codebooks come
// from ggml-common.h.  Every reduction below is parallel, so the summation ORDER differs from the
// serial reference kernels - the values agree to float rounding, which the fast-vs-reference check in
// glm_pack_test pins on the real model.
#include "strata/kernels/glm_fast.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/kernels/glm_expert_bench.hpp"
#include <cstring>
#endif

#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "dsa_topk.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_HIP)
// The HC grid exchanges FP32 partials between workgroups. An agent-scope
// acquire load preserves visibility after their publication fences; HIP only
// supplies CUDA's __ldcg spelling for half types.
__device__ __forceinline__ float __ldcg(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

namespace strata::kernels::glmf {
namespace {

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; ++i) t = fmaxf(t, sh[i]);
    return t;
}
// one q8_1 block from 32 values held one per lane (quantize.cu's arithmetic)
__device__ __forceinline__ void q8_1_store_warp(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    // the scale and the sum stay finite in FP16 (Strata #1448: an activation past ~8.3M overflowed the scale to inf,
    // and the dot products to NaN); below that nothing changes
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// ---------------------------------------------------------------- llama.cpp vecdotq helpers
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// --- K quants (bq8_1 = the q8_1 blocks of this weight block's 256 values)
__device__ __forceinline__ float dot_q4_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q4_K* bq4 = (const block_q4_K*) vbq + kbx;
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];
    const uint16_t* scales = (const uint16_t*) bq4->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float dot_q5_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q5_K* bq5 = (const block_q5_K*) vbq + kbx;
    int vl[2], vh[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = (const uint16_t*) bq5->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(bq5->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float dot_q6_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q6_K* w = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = get_int_b2(w->ql, iqs);
    const int vh = get_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = w->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int u = ((const int*) bq8_1[bq8_offset + 2 * i].qs)[iqs % 8];
        const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (__dp4a(vi, u, 0) * sc);
    }
    return __half2float(w->d) * sumf;
}

__device__ __forceinline__ float dot_q8_0(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q8_0* w = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b2(w->qs, iqs + i);
        const int u = ((const int*) bq8_1->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    return __half2float(w->d) * __low2float(bq8_1->ds) * float(sumi);
}

// --- i-quants (iq_kernels.cu's transcriptions)
// --- Q2_K / Q3_K (the NextN block's experts): llama.cpp's vec_dot_q2_K_q8_1 / vec_dot_q3_K_q8_1, VDR 1 - 16 calls per
// 256-block, call iqs reading int (iqs % 8) of the 4 q8_1 blocks 4 * (iqs / 8) .. + 3
__device__ __forceinline__ float dot_q2_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q2_K* b = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8 * (__dp4a(vi, u, 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8 * __dp4a(m, u, 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
__device__ __forceinline__ float dot_q3_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q3_K* b = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = __half2float(b->d);
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += d8 * (__dp4a((int) __vsubss4(vil, vih), u, 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float dot_iq2_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq1_s(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq4_xs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// qk = values per block, ipb = dot calls per block, step = iqs stride, bytes = block bytes
template<int T> struct F;
template<> struct F<12> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q4_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q4_K(v, y, kbx, iqs); } };
template<> struct F<13> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q5_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q5_K(v, y, kbx, iqs); } };
template<> struct F<14> { static constexpr int qk = 256, ipb = 32, step = 1, bytes = sizeof(block_q6_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q6_K(v, y, kbx, iqs); } };
template<> struct F<8> { static constexpr int qk = 32, ipb = 4, step = 2, bytes = sizeof(block_q8_0);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q8_0(v, y, kbx, iqs); } };
template<> struct F<16> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq2_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq2_xxs(v, y, kbx, iqs); } };
template<> struct F<18> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq3_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq3_xxs(v, y, kbx, iqs); } };
template<> struct F<19> { static constexpr int qk = 256, ipb = 8, step = 1, bytes = sizeof(block_iq1_s);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq1_s(v, y, kbx, iqs); } };
template<> struct F<10> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q2_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q2_K(v, y, kbx, iqs); } };
template<> struct F<11> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q3_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q3_K(v, y, kbx, iqs); } };
template<> struct F<23> { static constexpr int qk = 256, ipb = 8, step = 4, bytes = sizeof(block_iq4_xs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq4_xs(v, y, kbx, iqs); } };

// one row against one q8_1 activation, the whole warp (lane-strided calls)
template<int T>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int n_in, int lane) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += Fm::dot(row, x + kbx * (Fm::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_f32(const float* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const float4* r4 = (const float4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 a = r4[k], b = x4[k];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_bf16(const uint16_t* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const uint4* r8 = (const uint4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float4 a = x4[2 * k], b = x4[2 * k + 1];
        s += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
             __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
             __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
             __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
    }
    return warp_sum(s);
}

template<int T>
__device__ __forceinline__ size_t rbytes(int n_in) { return (size_t) (n_in / F<T>::qk) * F<T>::bytes; }

// ---- the same i-quant dots with the codebook in SHARED memory (the expert kernels): the grid lookups are
// scattered across the 32 lanes, and from global memory every one of them replays through L1
__device__ __forceinline__ float dot_iq1_s_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs,
                                             const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq2_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
template<int T> __device__ __forceinline__ float dot_t(const void* v, const block_q8_1* y, int kbx, int iqs,
                                                       const uint32_t* tab);
template<> __device__ __forceinline__ float dot_t<19>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq1_s_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<16>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq2_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<18>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq3_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<23>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t*) { return dot_iq4_xs(v, y, kbx, iqs); }
// u32 words of the type's codebook (0 = none needed)
template<int T> struct TabWords { static constexpr int n = 0; };
template<> struct TabWords<19> { static constexpr int n = 2048; };
template<> struct TabWords<16> { static constexpr int n = 512; };
template<> struct TabWords<18> { static constexpr int n = 256; };
template<> struct TabWords<17> { static constexpr int n = 1024; };
template<> struct TabWords<22> { static constexpr int n = 2048; };
template<> struct TabWords<21> { static constexpr int n = 512; };
template<> struct TabWords<29> { static constexpr int n = 2048; };
template<int T>
__device__ __forceinline__ void load_tab(uint32_t* s_tab) {
    const uint32_t* src = T == 19 || T == 29 ? (const uint32_t*) iq1s_grid_gpu
                          : T == 16 ? (const uint32_t*) iq2xxs_grid
                          : T == 18 ? (const uint32_t*) iq3xxs_grid
                          : T == 17 ? (const uint32_t*) iq2xs_grid
                          : T == 22 ? (const uint32_t*) iq2s_grid
                          : T == 21 ? (const uint32_t*) iq3s_grid : nullptr;
    for (int i = threadIdx.x; i < TabWords<T>::n; i += blockDim.x) s_tab[i] = src[i];
}
// ---- the activation half of a dot, loaded ONCE and reused across rows: every i-quant call k reads exactly one
// q8_1 block (index kbx * qk/32 + k % ipb) and all 8 of its ints
struct XV {
    int u[8];
    float d, s;
};
__device__ __forceinline__ XV load_xv(const block_q8_1* xb) {
    XV v;
    const int* q = (const int*) xb->qs;
#pragma unroll
    for (int i = 0; i < 8; ++i) v.u[i] = q[i];
    const float2 ds = __half22float2(xb->ds);
    v.d = ds.x;
    v.s = ds.y;
    return v;
}
template<int T> __device__ __forceinline__ float dw(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab);
template<> __device__ __forceinline__ float dw<19>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        sumi = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi);
        sumi = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    return d1q * (x.d * sumi + x.s * delta);
}
template<> __device__ __forceinline__ float dw<16>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[k0 + 0], sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<18>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<23>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t*) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) row + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = __dp4a(v.x, x.u[j + 0], sumi);
        sumi = __dp4a(v.y, x.u[j + 4], sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    return __half2float(bq4->d) * x.d * sumi;
}
// ---- the other i-quants of the Maya quants (docs/QUANT-PLAN.md): vecdotq's IQ2_XS, IQ2_S, IQ3_S and IQ1_M (the
// expressions iq_kernels.cu transcribes) in the same activation-preloaded, table-in-shared-memory form
template<> __device__ __forceinline__ float dw<17>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<22>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<21>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_s* bq3 = (const block_iq3_s*) row + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        sumi = __dp4a(grid_l, x.u[l0 + 0], sumi);
        sumi = __dp4a(grid_h, x.u[l0 + 1], sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<29>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_m* bq1 = (const block_iq1_m*) row + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = tab[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        sumi[l0 / 4] = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi[l0 / 4]);
        sumi[l0 / 4] = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = __dp4a(x.u[l0 + 0], 0x01010101, sumy);
        sumy = __dp4a(x.u[l0 + 1], 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * x.d;
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}
// HIP sign decode adapted from Strata src/prefill/moe_fused_iq.cu (MIT).
// Every IQ2/IQ3 grid byte is nonzero and <= 62, so two's-complement negation
// can use a whole-word add without carries across byte boundaries. Multiplying
// a four-bit sign nibble by 0x10204080 puts its bits at each byte's high bit;
// after masking/shifting, these are the +1 corrections and *255 gives the masks.
// The integer scale divisions and floating-point accumulation order stay intact.
#if defined(STRATA_USE_HIP)
__device__ __forceinline__ uint4 direct_signs(unsigned b, bool parity) {
    b &= parity ? 127u : 255u;
    if (parity) b |= (__popc(b) & 1) << 7;
    const unsigned lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    const unsigned hi = ((((b >> 4) & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return make_uint4(lo * 255u, lo, hi * 255u, hi);
}
__device__ __forceinline__ uint2 quant_load8(const void* p) {
    uint2 words;
    __builtin_memcpy(&words, p, sizeof(words)); // GGUF super-blocks are only 2-byte aligned.
    return words;
}
template<int T> constexpr int sign_count = T == 22 ? 256 : 128;
template<int T>
__device__ __forceinline__ void load_signs(uint4* signs) {
    for (int i = threadIdx.x; i < sign_count<T>; i += blockDim.x) {
        unsigned b = i;
        if constexpr (T != 22) b |= (__popc(b) & 1) << 7;
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= (0u - ((b >> j) & 1u)) & (255u << (8 * j));
            hi |= (0u - ((b >> (j + 4)) & 1u)) & (255u << (8 * j));
        }
        signs[i] = make_uint4(lo, lo & 0x01010101u, hi, hi & 0x01010101u);
    }
}
template<int T>
__device__ __forceinline__ float dw_rdna(const uint8_t* row, int kbx, int iqs, const XV& x,
                                        const uint32_t* tab, const uint4* signs, bool direct, bool wide);
template<> __device__ __forceinline__ float dw_rdna<16>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = wide ? quant_load8(bq2->qs + 2 * iqs).x : get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = wide ? quant_load8(bq2->qs + 2 * iqs).y : get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * k0 / 2), true) : sign_tab[(aux32 >> (7 * k0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[k0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<18>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const uint2 q3_wide = wide ? quant_load8(bq3->qs + 4 * iqs) : make_uint2(0, 0);
    const int2 q3_packed = wide ? make_int2(q3_wide.x, q3_wide.y) :
        make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * l0 / 2), true) : sign_tab[(aux32 >> (7 * l0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<22>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    (void) wide;
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const uint4 signs = direct ? direct_signs(signs_packed_8[l0 / 2], false) : sign_tab[signs_packed_8[l0 / 2]];
        const int grid_l = ((grid_pos.x ^ signs.x) + signs.y);
        const int grid_h = ((grid_pos.y ^ signs.z) + signs.w);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<int T, int NR>
__device__ __forceinline__ void rows_dot_rdna(const uint8_t* const* rows, const block_q8_1* x,
                                             int n_in, int lane, const uint32_t* tab,
                                             const uint4* signs, float* s, bool direct = false, bool wide = false) {
    float acc[NR] = {};
    for (int k = lane; k < n_in / 32; k += 32) {
        const XV xv = load_xv(x + k);
#pragma unroll
        for (int r = 0; r < NR; ++r)
            acc[r] += dw_rdna<T>(rows[r], k / 8, 2 * (k % 8), xv, tab, signs, direct, wide);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}
#endif

// their F entries (the dense GEMV and row_dot read the codebook from global memory; the call's q8_1 block is
// iqs / step of the super-block's)
#define GLMF_IQ_F(T, BLOCK, STEP, TABLE)                                                                          \
    template<> struct F<T> { static constexpr int qk = 256, ipb = 8, step = STEP, bytes = sizeof(BLOCK);         \
        __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {                      \
            return dw<T>((const uint8_t*) v, kbx, iqs, load_xv(y + iqs / STEP), (const uint32_t*) TABLE); } };
GLMF_IQ_F(17, block_iq2_xs, 2, iq2xs_grid)
GLMF_IQ_F(22, block_iq2_s, 2, iq2s_grid)
GLMF_IQ_F(21, block_iq3_s, 2, iq3s_grid)
GLMF_IQ_F(29, block_iq1_m, 1, iq1s_grid_gpu)
#undef GLMF_IQ_F
// NR rows against ONE activation, the whole warp; s[r] gets each row's full sum
// the K-quant calls read 4 q8_1 blocks' int (iqs % 8) each: loaded ONCE per call position for all rows
struct XK {
    int u[4];
    float d8[4];
};
__device__ __forceinline__ XK load_xk(const block_q8_1* xb, int iqs) {
    XK v;
    const int off = 4 * (iqs / 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v.u[i] = get_int_b4(xb[off + i].qs, iqs % 8);
        v.d8[i] = __low2float(xb[off + i].ds);
    }
    return v;
}
template<int T> __device__ __forceinline__ float dwk(const uint8_t* row, int kbx, int iqs, const XK& x);
template<> __device__ __forceinline__ float dwk<10>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q2_K* b = (const block_q2_K*) row + kbx;
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sc = scales[2 * i];
        sumf_d += x.d8[i] * (__dp4a((v >> (2 * i)) & 0x03030303, x.u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += x.d8[i] * __dp4a(m, x.u[i], 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
template<> __device__ __forceinline__ float dwk<11>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q3_K* b = (const block_q3_K*) row + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += x.d8[i] * (__dp4a((int) __vsubss4(vil, vih), x.u[i], 0) * ((sc_low | sc_high) - 32));
    }
    return __half2float(b->d) * sumf;
}
template<int T, int NR>
__device__ __forceinline__ void rows_dot(const uint8_t* const* rows, const block_q8_1* x, int n_in, int lane,
                                         const uint32_t* tab, float* s) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    if constexpr (T == 10 || T == 11) {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = k % Fm::ipb;
            const XK xk = load_xk(x + kbx * (Fm::qk / 32), iqs);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dwk<T>(rows[r], kbx, iqs, xk);
        }
    } else if constexpr (T == 12 || T == 13 || T == 14) {
        // Q4_K / Q5_K / Q6_K experts (K-quant GGUFs other than Maya's): no shared-activation form yet - each row
        // takes the dense GEMV's dot (row_dot), the activation re-read per row
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += Fm::dot(rows[r], x + kbx * (Fm::qk / 32), kbx, iqs);
        }
    } else {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
            const XV xv = load_xv(x + kbx * (Fm::qk / 32) + ki);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dw<T>(rows[r], kbx, iqs, xv, tab);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}

template<int T>
__device__ __forceinline__ float row_dot_t(const uint8_t* row, const block_q8_1* x, int n_in, int lane,
                                           const uint32_t* tab) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
#pragma unroll 4
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += dot_t<T>(row, x + kbx * (Fm::qk / 32), kbx, iqs, tab);
    }
    return warp_sum(s);
}

__device__ __forceinline__ float any_row_dot(int type, const void* w, int row, const void* xq, const float* xf,
                                             int n_in, int lane) {
    switch (type) {
        case kTypeF32: return row_dot_f32((const float*) w + (size_t) row * n_in, xf, n_in, lane);
        case kTypeBF16: return row_dot_bf16((const uint16_t*) w + (size_t) row * n_in, xf, n_in, lane);
#define GF_CASE(T) case T: return row_dot<T>((const uint8_t*) w + (size_t) row * rbytes<T>(n_in), (const block_q8_1*) xq, n_in, lane);
        GF_CASE(12) GF_CASE(13) GF_CASE(14) GF_CASE(8) GF_CASE(16) GF_CASE(18) GF_CASE(19) GF_CASE(23) GF_CASE(10) GF_CASE(11)
        GF_CASE(17) GF_CASE(22) GF_CASE(21) GF_CASE(29)
#undef GF_CASE
        default: return 0.0f;
    }
}

// ---------------------------------------------------------------- multi-job GEMV
struct MvBatch {
    MvJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
};
constexpr int MV_ROWS = 8;   // rows (warps) per block

__global__ void __launch_bounds__(256) mv_kernel(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const float s = any_row_dot(J.type, J.w, row, J.xq, J.xf, J.n_in, lane);
    if (lane == 0) J.y[row] = J.alpha * s + (J.bias ? J.bias[row] : 0.0f);
}

// ... the same for jobs of ONE quantized type: compiled for that type alone (the switch over every type costs the
// generic kernel registers and occupancy - a Q6_K 8192 x 4096 GEMV on a V100: 53.4 -> 47.8 us, the same arithmetic)
template<int T>
__global__ void __launch_bounds__(256) mv_kernel_t(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const float s = row_dot<T>((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), (const block_q8_1*) J.xq,
                               J.n_in, lane);
    if (lane == 0) J.y[row] = J.alpha * s + (J.bias ? J.bias[row] : 0.0f);
}

__global__ void quantize_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0: whole warps only
    q8_1_store_warp(x[i], y + i / 32, threadIdx.x & 31);
}

// ---------------------------------------------------------------- mHC
constexpr int HC_CHUNK = 512;
constexpr int HC_THREADS = 256;
constexpr int HC_MIX = 24;      // (2 + hc) * hc for hc = 4

__global__ void __launch_bounds__(HC_THREADS) hc_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float sR[HC_CHUNK];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ bool s_last;

    // ---- phase 1: this block's chunk of R_new, its sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < HC_CHUNK; i += HC_THREADS) {
        const int idx = b * HC_CHUNK + i;
        float r;
        if (a.block_out != nullptr) {
            const int d = idx / n_embd, e = idx - d * n_embd;
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[idx] = r;
        } else {
            r = a.R_old[idx];
        }
        sR[i] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const uint16_t* wr = a.w_fn + (size_t) m * hc_dim + (size_t) b * HC_CHUNK;
        float acc = 0.0f;
        for (int i = lane; i < HC_CHUNK; i += 32) acc += sR[i] * bf(wr[i]);
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;

    // ---- phase 2 (the last block): the gates, the Sinkhorn, mixed, the norm, q8_1
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[4096];
    if (tid == 0) *a.counter = 0;
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) gridDim.x; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            const float post_v = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            a.pre[lane] = pre_v;
            s_pre[lane] = pre_v;
            a.post[lane] = post_v;
        }
        // comb[d][s] on lane d*4 + s (lanes 0..15); lanes 16..31 shadow harmlessly
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        // softmax over dst (d) per column s: lanes sharing s differ in bits 2..3
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        // dst-normalise: sum over s (bits 0..1); src-normalise: sum over d (bits 2..3)
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    const float* Rn = a.block_out != nullptr ? a.R_new : a.R_old;
    float ss2 = 0.0f;
    for (int e = tid; e < n_embd; e += HC_THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * __ldcg(&Rn[s * n_embd + e]);
        s_mixed[e] = acc;
        ss2 += acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    const float inv2 = rsqrtf(ss2 / (float) n_embd + a.norm_eps);
    for (int e = tid; e < n_embd; e += HC_THREADS) {   // warp w holds 32 consecutive e at every step
        const float xv = s_mixed[e] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take two: the grid splits by EMBEDDING range (all 4 streams of 128 values per block), so after the partial
// dots every block can finish its own range - two grid barriers (the blocks are co-resident: 32 blocks, checked
// at launch) instead of one block doing the whole tail.  Same arithmetic as hc_kernel.
constexpr int HC2_EB = 128;
__device__ __forceinline__ void grid_barrier(unsigned long long* ctr, unsigned int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long v = atomicAdd(ctr, 1ull);
        const unsigned long long target = (v / nb + 1) * nb;
        while (*(volatile unsigned long long*) ctr < target) {
        }
        __threadfence();
    }
    __syncthreads();
}

__global__ void __launch_bounds__(HC_THREADS) hc2_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const unsigned int nb = gridDim.x;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];

    // ---- phase 1: this block's 4 x 128 values of R_new, their sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const uint16_t* wr = a.w_fn + (size_t) m * hc_dim + e0;
        float acc = 0.0f;
        for (int i = lane; i < 4 * HC2_EB; i += 32) {
            const int d = i / HC2_EB, j = i - d * HC2_EB;
            acc += sR[d][j] * bf(wr[(size_t) d * n_embd + j]);
        }
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    grid_barrier(ctr, nb);

    // ---- phase 2 (every block, redundantly): the gates and the Sinkhorn from the partial sums
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) nb; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    // ---- phase 3: mixed for this block's 128 values, the norm's sum of squares across the grid, x and q8_1
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    float* part2 = a.part + (size_t) nb * 25;
    if (tid == 0) part2[b] = ss2;
    grid_barrier(ctr, nb);
    float tot = 0.0f;
    for (int bb = 0; bb < (int) nb; ++bb) tot += __ldcg(&part2[bb]);
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {   // warps 0..3: 32 consecutive values each
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take three: hc2's arithmetic with the latency taken out - every block issues its 24 weight rows' loads
// (8 bf16 per lane per stream segment, uint4) BEFORE it computes its R values, the partial sums are reduced by whole
// warps (lane = block) instead of one thread walking 32 of them, and the second barrier's total likewise.
// 32 blocks of 128 embedding values; n_embd == 4096 only.
__global__ void __launch_bounds__(HC_THREADS) hc3_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const unsigned int nb = gridDim.x;   // 32
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];
    // the weights first: warp w owns mix rows w, w + 8, w + 16; per row and stream segment d a lane loads the 4 bf16
    // values j = 4 * lane .. 4 * lane + 3 of [e0, e0 + 128)
    uint2 wv[3][4];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        const uint16_t* wr = a.w_fn + (size_t) m * (hc * n_embd) + e0 + 4 * lane;
#pragma unroll
        for (int d = 0; d < 4; ++d) wv[r][d] = *(const uint2*) (wr + (size_t) d * n_embd);
    }
    // R_new for this block's 4 x 128 values (thread t: stream t / 64... two values per thread)
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    float* part = a.part;   // [m][block]: 25 rows of nb partials (the 25th: the sum of squares)
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const uint2 w = wv[r][d];
            const float* x = &sR[d][4 * lane];
            acc += __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1] +
                   __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
        }
        acc = warp_sum(acc);
        if (lane == 0) part[(size_t) m * nb + b] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid == 0) part[(size_t) HC_MIX * nb + b] = ss;
    grid_barrier(ctr, nb);
    // the 25 totals: warp w reduces rows w, w + 8, w + 16 (and warp 0 row 24), one partial per lane
    for (int m = warp; m < HC_MIX + 1; m += 8) {
        const float v = lane < (int) nb ? __ldcg(&part[(size_t) m * nb + lane]) : 0.0f;
        const float t = warp_sum(v);
        if (lane == 0) smix[m] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) (hc * n_embd) + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    float* part2 = a.part + (size_t) (HC_MIX + 1) * nb;
    if (tid == 0) part2[b] = ss2;
    grid_barrier(ctr, nb);
    const float tot = warp_sum(lane < (int) nb ? __ldcg(&part2[lane]) : 0.0f);
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

__global__ void hc_post_kernel(const float* __restrict__ block_out, const float* __restrict__ R_old,
                               const float* __restrict__ post, const float* __restrict__ comb, int n_embd,
                               float* __restrict__ R_new) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    const int d = idx / n_embd, e = idx - d * n_embd;
    float acc = block_out[e] * post[d];
#pragma unroll
    for (int s = 0; s < 4; ++s) acc += comb[d * 4 + s] * R_old[s * n_embd + e];
    R_new[idx] = acc;
}

__global__ void head_prep_kernel(const float* __restrict__ R, const float* __restrict__ norm_w, float eps,
                                 int n_embd, float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    __shared__ float sm[4096];
    const int tid = threadIdx.x, lane = tid & 31;
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < 4; ++s) acc += R[e + (size_t) n_embd * s];
        acc = acc / 4.0f;
        sm[e] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float v = sm[e] * inv * norm_w[e];
        x[e] = v;
        q8_1_store_warp(v, xq + e / 32, lane);
    }
}

__global__ void embed_streams_kernel(const float* __restrict__ emb, float* __restrict__ R, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    R[idx] = emb[idx % n_embd];
}

// ---------------------------------------------------------------- KDA
__global__ void __launch_bounds__(128) kda_prep_kernel(const __grid_constant__ KdaPrepArgs a) {
    const int h = blockIdx.x, role = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int hd = a.head_dim;
    const int d_inner = a.n_head * hd;
    __shared__ float sred[32];
    __shared__ float sv[128];
    if (role < 3) {
        const int c = h * hd + tid;
        const int hist = a.d_conv - 1;
        float* st = a.conv_state + (size_t) role * d_inner * hist + (size_t) hist * c;
        const float* w = a.conv_w[role] + (size_t) a.d_conv * c;
        const float xp = a.proj[role][c];
        float acc = 0.0f;
        for (int tap = 0; tap < hist; ++tap) acc += w[tap] * st[tap];
        acc += w[hist] * xp;
        float y = acc / (1.0f + __expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) st[j] = st[j + 1];
        st[hist - 1] = xp;
        if (role < 2) {
            const float ss = block_sum(y * y, sred);
            y = y * rsqrtf(ss + 1e-6f);
        }
        a.out[role][c] = y;
        return;
    }
    // roles 3..6: g1 = lb * sigmoid(-(f_b . fa + dt_bias) * ssm_a[h]) for a quarter of the head's rows each;
    // roles 7..10: g2 = g_b . ga, likewise (a quarter = 32 rows: 8 per warp)
    const bool is_g1 = role < 7;
    const int quarter = (role - 3) & 3;
    const float* in = is_g1 ? a.fa : a.ga;
    const uint16_t* W = is_g1 ? a.f_b : a.g_b;
    sv[tid] = in[tid];
    __syncthreads();
    const int rbase = quarter * (hd / 4) + warp * (hd / 16);
    for (int r = rbase; r < rbase + hd / 16; ++r) {
        const int c = h * hd + r;
        const uint16_t* wr = W + (size_t) c * hd;
        float acc = 0.0f;
        for (int j = lane; j < hd; j += 32) acc += bf(wr[j]) * sv[j];
        acc = warp_sum(acc);
        if (lane == 0) {
            if (is_g1) {
                const float v = acc + a.dt_bias[c];
                a.g1[c] = a.lower_bound * dsigmoid(-v * a.ssm_a[h]);
            } else {
                a.g2[c] = acc;
            }
        }
    }
}

__global__ void __launch_bounds__(256) kda_rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ g1,
                                                      const float* __restrict__ beta_raw, float* __restrict__ state,
                                                      const float* __restrict__ g2, const float* __restrict__ norm_w,
                                                      float eps, int n_head, block_q8_1* __restrict__ out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, j = t & (HD - 1), half = t >> 7, lane = t & 31;
    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[32];
    if (t < HD) {
        sq[t] = q[h * HD + t];
        sk[t] = k[h * HD + t];
        sdec[t] = __expf(g1[h * HD + t]);
    }
    const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    __syncthreads();
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
    float A = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s = Sh[i * HD + j] * sdec[i];
        reg[ii] = s;
        A += s * sk[i];
    }
    sacc[half][j] = A;
    __syncthreads();
    if (half == 0) sdel[j] = b * (v[h * HD + j] - (sacc[0][j] + sacc[1][j]));
    __syncthreads();
    const float dl = sdel[j];
    float O = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + sk[i] * dl;
        Sh[i * HD + j] = s2;
        O += s2 * sq[i];
    }
    __syncthreads();
    sacc[half][j] = O;
    __syncthreads();
    const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
    // out gate over the head (threads 0..127 own j; the upper half mirrors and does not write)
    float ss = half == 0 ? o * o : 0.0f;
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) HD + eps);
    if (half == 0) {
        const float gv = o * inv * norm_w[j] * dsigmoid(g2[h * HD + j]);
        q8_1_store_warp(gv, out + h * (HD / 32) + (j >> 5), lane);
    }
}

// the DSA latent cache is FP16 (uint16_t bits): read / written in F32
__device__ __forceinline__ float lat_f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t lat_h(float v) { return __half_as_ushort(__float2half(v)); }
// the INT8 latent records (lat8_rec_bytes): value c of the record at rec
__device__ __forceinline__ float lat8_f(const uint8_t* rec, int kv_lora, int c) {
    return (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
}
// latent value c of cell: FP16 rows or INT8 records
__device__ __forceinline__ float lat_any(const uint16_t* lat, bool q8, int kv_lora, int cell, int c) {
    return q8 ? lat8_f((const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell, kv_lora, c)
              : lat_f(lat[(size_t) kv_lora * cell + c]);
}
// one latent row into its INT8 record: lane l of the warp holding values [32 w, 32 w + 32) stores value c
__device__ __forceinline__ void lat8_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half d16 = __float2half(amax / 127.0f);
    const float d = __half2float(d16);
    const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0;
    ((int8_t*) rec)[c] = (int8_t) q;
    if (lane == 0) ((__half*) (rec + kv_lora))[c >> 5] = d16;
}
// ---------------------------------------------------------------- DSA
__global__ void __launch_bounds__(512) dsa_prep_kernel(const __grid_constant__ DsaPrepArgs a) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    if (blockIdx.x == 0) {
        // q_a norm -> qr (f32) and qr_q (q8_1); q_lora <= 2048 (stride 512 keeps warps on 32-runs)
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            vals[n] = a.qr_raw[e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            a.qr[e] = y;
            q8_1_store_warp(y, (block_q8_1*) a.qr_q + e / 32, lane);
        }
        return;
    }
    if (blockIdx.x == 1) {
        float v = tid < a.kv_lora ? a.kv_raw[tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (tid < a.kv_lora) {   // (kv_lora % 32 == 0: whole warps)
            const float y = v * inv * a.kv_norm[tid];
            if (a.lat_q8)
                lat8_store_warp(y, (uint8_t*) a.lat + (size_t) lat8_rec_bytes(a.kv_lora) * a.p, a.kv_lora, tid, tid & 31);
            else
                a.lat[(size_t) a.kv_lora * a.p + tid] = lat_h(y);
        }
        return;
    }
    // block 2: the indexer key (layer norm) and the compressor gate into their caches; the pool
    const int K = a.idx_key;
    const float v = tid < K ? a.ik_raw[tid] : 0.0f;
    const float s1 = block_sum(v, sred);
    const float s2 = block_sum(v * v, sred);
    const float mu = s1 / (float) K;
    const float inv = rsqrtf(s2 / (float) K - mu * mu + a.eps);
    if (tid < K) {
        a.ik_cache[(size_t) K * (a.p % a.ring) + tid] = (v - mu) * inv * a.k_norm_w[tid] + a.k_norm_b[tid];
        a.ig_cache[(size_t) K * (a.p % a.ring) + tid] = a.ig_raw[tid];
    }
    if ((a.p + 1) % a.kpool != 0) return;
    __syncthreads();
    __threadfence_block();
    const int pi = (a.p + 1) / a.kpool - 1;
    if (tid < K) {
        float lg[16];
        float mx = -INFINITY;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = a.ig_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)] + a.ape[tid + K * m];
            mx = fmaxf(mx, lg[m]);
        }
        float den = 0.0f;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = expf(lg[m] - mx);
            den += lg[m];
        }
        float acc = 0.0f;
        for (int m = 0; m < a.kpool; ++m)
            acc += (lg[m] / den) * a.ik_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)];
        a.pooled[(size_t) tid + (size_t) K * pi] = acc;
    }
}

__global__ void __launch_bounds__(256) dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                        const float* __restrict__ iw, int key_dim, int idx_heads,
                                                        int n_pools, float* __restrict__ score) {
    extern __shared__ float s_iq[];   // idx_heads * key_dim, then idx_heads weights
    float* s_iw = s_iq + idx_heads * key_dim;
    for (int i = threadIdx.x; i < idx_heads * key_dim; i += blockDim.x) s_iq[i] = iq[i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_iw[i] = iw[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (p >= n_pools) return;
    const float* pk = pooled + (size_t) key_dim * p;
    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        float dot = 0.0f;
        for (int e = lane; e < key_dim; e += 32) dot += s_iq[h * key_dim + e] * pk[e];
        dot = warp_sum(dot);
        acc += fmaxf(dot, 0.0f) * s_iw[h];
    }
    if (lane == 0) score[p] = acc;
}

__global__ void __launch_bounds__(1024) dsa_select_kernel(const float* __restrict__ score, int n_vis, int kpool,
                                                          int top_pools, int n_sel, int pos, int* __restrict__ cells) {
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) cells[i] = -1;
    __syncthreads();
    // the top pools in rank order (score descending, ties by the lower index), O(n_vis) - src/kernels/cuda/dsa_topk.cuh
    if (n_vis > 0) strata::kernels::dsa::select_top_pools(score, n_vis, min(top_pools, n_vis), kpool, cells);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos) cells[top_pools * kpool + m] = cell;
        }
    }
}

// One block per head: q_abs = wk_b_h . q_h (kv_lora rows of qk_nope), scores over the selected latents,
// softmax, ctx, out = wv_b_h . ctx (v_head rows of kv_lora), q8_1 of out.
__global__ void __launch_bounds__(256) mla_kernel(const float* __restrict__ q, const uint16_t* __restrict__ wk_b,
                                                  const uint16_t* __restrict__ wv_b, const uint16_t* __restrict__ lat,
                                                  const int* __restrict__ cells, int n_sel, int qk_nope, int kv_lora,
                                                  int v_head, block_q8_1* __restrict__ out, bool q8) {
    extern __shared__ float smem[];
    float* s_q = smem;                    // qk_nope
    float* s_qa = s_q + qk_nope;          // kv_lora
    float* s_ctx = s_qa + kv_lora;        // kv_lora
    float* s_out = s_ctx + kv_lora;       // v_head
    float* s_p = s_out + v_head;          // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < qk_nope; i += blockDim.x) s_q[i] = q[(size_t) qk_nope * h + i];
    __syncthreads();
    const uint16_t* wk = wk_b + (size_t) qk_nope * kv_lora * h;
    for (int c = warp; c < kv_lora; c += nw) {
        const uint16_t* wr = wk + (size_t) qk_nope * c;
        float acc = 0.0f;
        for (int e = lane; e < qk_nope; e += 32) acc += bf(wr[e]) * s_q[e];
        acc = warp_sum(acc);
        if (lane == 0) s_qa[c] = acc;
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        s_ctx[c] = acc;
    }
    __syncthreads();
    const uint16_t* wv = wv_b + (size_t) kv_lora * v_head * h;
    for (int vv = warp; vv < v_head; vv += nw) {
        const uint16_t* wr = wv + (size_t) kv_lora * vv;
        float acc = 0.0f;
        for (int c = lane; c < kv_lora; c += 32) acc += bf(wr[c]) * s_ctx[c];
        acc = warp_sum(acc);
        if (lane == 0) s_out[vv] = acc;
    }
    __syncthreads();
    for (int vv = tid; vv < v_head; vv += blockDim.x)   // warp-aligned runs of 32
        q8_1_store_warp(s_out[vv], out + ((size_t) v_head * h + vv) / 32, lane);
}

// ---- MLA, take two: the absorbed projections as head-wise GEMVs over the whole GPU (warp per row, 32 rows per
// block - 4 per warp) and the attention proper in between
// out[h * R + r] = W[h][r][:] . in[h * K : h * K + K] (W BF16, rows of K); q8 != nullptr also writes the q8_1
__global__ void __launch_bounds__(256) headwise_gemv_kernel(const uint16_t* __restrict__ W, const float* __restrict__ in,
                                                           int R, int K, float* __restrict__ out,
                                                           block_q8_1* __restrict__ q8) {
    __shared__ float s_o[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32;   // R % 32 == 0: a block never straddles heads
    const int h = row0 / R;
    const float* x = in + (size_t) h * K;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const float s = row_dot_bf16(W + (size_t) row * K, x, K, lane);
        if (lane == 0) s_o[warp * 4 + j] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = s_o[lane];
        if (out) out[row0 + lane] = v;
        if (q8) q8_1_store_warp(v, q8 + row0 / 32, lane);
    }
}

__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                       const int* __restrict__ cells, int n_sel, int qk_nope,
                                                       int kv_lora, float* __restrict__ ctx, bool q8) {
    extern __shared__ float smem[];
    float* s_qa = smem;               // kv_lora
    float* s_p = s_qa + kv_lora;      // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < kv_lora; i += blockDim.x) s_qa[i] = q_abs[(size_t) kv_lora * h + i];
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        ctx[(size_t) kv_lora * h + c] = acc;
    }
}

// ---- MLA attention, split over the selected cells (flash-decoding): every head reads the SAME latents, so a block
// loads a chunk of MLA_CHUNK latent rows into shared memory once and scores it against ALL heads; per head it keeps
// its chunk's max, sum of exponentials and exp-weighted context, and the combine kernel merges the chunks.  At 2k
// selected cells this reads the latents once per layer instead of once per head (64x less).
// The grid is (chunks, head groups of MLA_HPB): ~20 chunks at 600 cells would leave most SMs idle with every head in
// one block (208 us a call on a V100); a group re-reads its chunk's latents, which is cheap next to the arithmetic.
// The latents stay FP16 in shared memory, as in the cache (34 KB at kv_lora 512; F32 rows were over Turing's 64 KB).
constexpr int MLA_CHUNK = 32;
constexpr int MLA_HPB = 16;
__global__ void __launch_bounds__(256) mla_split_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                        const int* __restrict__ cells, int n_sel, int n_head, int qk_nope,
                                                        int kv_lora, float* __restrict__ part, bool q8) {
    extern __shared__ float smem[];
    uint16_t* sL = (uint16_t*) smem;                   // MLA_CHUNK x kv_lora, FP16
    float* sS = smem + MLA_CHUNK * kv_lora / 2;        // MLA_HPB x MLA_CHUNK scores -> weights
    __shared__ int s_cell[MLA_CHUNK];
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const int h0 = blockIdx.y * MLA_HPB, nh = min(MLA_HPB, n_head - h0);
    const int s0 = b * MLA_CHUNK;
    const int ns = min(MLA_CHUNK, n_sel - s0);
    if (tid < MLA_CHUNK) s_cell[tid] = tid < ns ? cells[s0 + tid] : -1;
    __syncthreads();
    for (int i = tid; i < MLA_CHUNK * kv_lora; i += blockDim.x) {
        const int s = i / kv_lora, c = i - s * kv_lora;
        const int cell = s_cell[s];
        sL[i] = cell < 0 ? (uint16_t) 0 : q8 ? lat_h(lat_any(lat, true, kv_lora, cell, c)) : lat[(size_t) kv_lora * cell + c];
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    // scores: a warp per head, the head's q in registers (kv_lora / 32 per lane), one warp reduction per cell
    for (int hh = warp; hh < nh; hh += nw) {
        float q[16];
        const float* qh = q_abs + (size_t) kv_lora * (h0 + hh);
#pragma unroll
        for (int j = 0; j < 16; ++j) q[j] = (lane + 32 * j) < kv_lora ? qh[lane + 32 * j] : 0.0f;
        for (int s = 0; s < MLA_CHUNK; ++s) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < kv_lora) acc += q[j] * lat_f(sL[s * kv_lora + lane + 32 * j]);
            acc = warp_sum(acc);
            if (lane == 0) sS[hh * MLA_CHUNK + s] = s_cell[s] >= 0 ? acc * scale : -INFINITY;
        }
    }
    __syncthreads();
    // per head: the chunk's max and sum; the scores become exp-weights
    float* pm = part + ((size_t) b * n_head + h0) * (kv_lora + 2);
    for (int hh = warp; hh < nh; hh += nw) {
        const float v = lane < MLA_CHUNK ? sS[hh * MLA_CHUNK + lane] : -INFINITY;
        const float m = warp_max(v);
        const float e = (v == -INFINITY) ? 0.0f : expf(v - m);
        const float l = warp_sum(e);
        if (lane < MLA_CHUNK) sS[hh * MLA_CHUNK + lane] = e;
        if (lane == 0) {
            pm[(size_t) hh * (kv_lora + 2) + kv_lora] = m;
            pm[(size_t) hh * (kv_lora + 2) + kv_lora + 1] = l;
        }
    }
    __syncthreads();
    // exp-weighted context per head over this chunk
    for (int i = tid; i < nh * kv_lora; i += blockDim.x) {
        const int hh = i / kv_lora, c = i - hh * kv_lora;
        float acc = 0.0f;
#pragma unroll 8
        for (int s = 0; s < MLA_CHUNK; ++s) acc += sS[hh * MLA_CHUNK + s] * lat_f(sL[s * kv_lora + c]);
        pm[(size_t) hh * (kv_lora + 2) + c] = acc;
    }
}

__global__ void mla_combine_kernel(const float* __restrict__ part, int n_chunks, int n_head, int kv_lora,
                                   float* __restrict__ ctx) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int b = 0; b < n_chunks; ++b) M = fmaxf(M, part[((size_t) b * n_head + h) * (kv_lora + 2) + kv_lora]);
    float L = 0.0f;
    for (int b = 0; b < n_chunks; ++b) {
        const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
        if (pb[kv_lora] != -INFINITY) L += expf(pb[kv_lora] - M) * pb[kv_lora + 1];
    }
    const float invL = L > 0.0f ? 1.0f / L : 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < n_chunks; ++b) {
            const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
            if (pb[kv_lora] != -INFINITY) acc += expf(pb[kv_lora] - M) * pb[c];
        }
        ctx[(size_t) kv_lora * h + c] = acc * invL;
    }
}

__global__ void swiglu_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up, float limit, int n,
                                 block_q8_1* __restrict__ hq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + i / 32, threadIdx.x & 31);
}

// ---------------------------------------------------------------- routed experts
constexpr int ROUTE_THREADS = 320;

// the k largest of sel[0..E) in descending order, ties by the lower index - exactly the stable-descending rank rule
// (rank(e) = #{f: sel f > sel e} + #{f < e: sel f == sel e}) - as k rounds of a block argmax; sel is consumed
// (the picked entries become -inf).  All threads of the block call it.
__device__ void topk_argmax(float* sel, int E, int k, int* out, float* s_bv, int* s_bi) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = s_bv[0];
            int i = s_bi[0];
            for (int w = 1; w < nw; ++w)
                if (s_bv[w] > v || (s_bv[w] == v && s_bi[w] < i)) {
                    v = s_bv[w];
                    i = s_bi[w];
                }
            out[r] = i;
            if (i >= 0 && i < E) sel[i] = -INFINITY;
        }
        __syncthreads();
    }
}

struct RouteArgs {
    const float* logits;
    const float* bias;
    int n_expert, k;
    float w_scale;
    int norm_w;
    int layer;
    const float* x;
    int n_embd;
    MoeDev d;
    const float* sh_gate;
    const float* sh_up;
    float sh_limit;
    int n_ff_sh;
    block_q8_1* sh_hq;
    const float* pred_logits;   // the next layer's router on this layer's input (nullptr: no prediction)
    const float* pred_bias;
    int max_pf;                 // prefetch at most this many predicted non-resident experts of the next layer
    const float* ahead_logits;  // n_ahead x n_expert: the next layers' routers on this layer's input (LOOKAHEAD)
    const float* ahead_bias[kAhead];
    int n_ahead;
    int skip_from;              // experts not in VRAM at route rank >= skip_from are left out (no fetch, no wait):
                                // the draft block (0: all of them; >= k: none)
    unsigned long long cpu_plan;   // the CPU LANE: of f RAM-tier experts, (plan >> 4 f) & 15 go to the host (0: off)
    int promote_min;            // STRATA_GLM_PROMOTE_MIN: keep a fetched expert only when its aged route count
                                // clears this; 0 keeps the old rule (a spare, if one is free)
};

__global__ void __launch_bounds__(ROUTE_THREADS) moe_route_kernel(const __grid_constant__ RouteArgs a) {
    const int tid = threadIdx.x, lane = tid & 31;
    if (blockIdx.x > 0) {
        // the shared expert's swiglu + q8_1 (warp-aligned 32-runs)
        const int i = (blockIdx.x - 1) * ROUTE_THREADS + tid;
        if (i >= a.n_ff_sh) return;
        const float g = fminf(a.sh_gate[i], a.sh_limit);
        const float u = fminf(fmaxf(a.sh_up[i], -a.sh_limit), a.sh_limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, a.sh_hq + i / 32, lane);
        return;
    }
    const int E = a.n_expert;
    __shared__ float s_p[512], s_sel[512];
    __shared__ int s_ids[8];
    __shared__ float s_w[8];
    __shared__ unsigned int s_mask, s_seq;
    // the CPU lane's frequency counts age: every 4096 routes all of them halve (the whole block, before any use)
    if (a.cpu_plan != 0ull && a.d.dcnt != nullptr && ((*a.d.seq + 1) & 4095u) == 0u)
        for (int i = tid; i < a.d.n_keys; i += ROUTE_THREADS) a.d.dcnt[i] >>= 1;
    for (int e = tid; e < E; e += ROUTE_THREADS) {
        const float p = 1.0f / (1.0f + expf(-a.logits[e]));
        s_p[e] = p;
        s_sel[e] = a.bias ? p + a.bias[e] : p;
    }
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    __syncthreads();
    topk_argmax(s_sel, E, a.k, s_ids, s_bv, s_bi);
    // the next layer's predicted top-k (same selection rule, its own bias)
    __shared__ int s_pred[8];
    if (tid < 8) s_pred[tid] = -1;
    if (a.pred_logits != nullptr) {
        __syncthreads();
        for (int e = tid; e < E; e += ROUTE_THREADS) {
            const float p = 1.0f / (1.0f + expf(-a.pred_logits[e]));
            s_sel[e] = a.pred_bias ? p + a.pred_bias[e] : p;   // s_sel is free again: the selection is done
        }
        __syncthreads();
        topk_argmax(s_sel, E, a.k, s_pred, s_bv, s_bi);
    }
    // LOOKAHEAD: one warp per later layer, its top-k in registers (E <= 512; the same selection rule and ties)
    __shared__ short s_ah[kAhead][8];
    if ((tid >> 5) < a.n_ahead) {
        const int w = tid >> 5;
        const float* lg = a.ahead_logits + (size_t) w * E;
        const float* bs = a.ahead_bias[w];
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            float x = -INFINITY;
            if (e < E) {
                const float p = 1.0f / (1.0f + expf(-lg[e]));
                x = bs ? p + bs[e] : p;
            }
            v[j] = x;
        }
        for (int r = 0; r < a.k; ++r) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (v[j] > bv || (v[j] == bv && lane + 32 * j < bi)) {
                    bv = v[j];
                    bi = lane + 32 * j;
                }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j == bi) v[j] = -INFINITY;
            if (lane == 0) s_ah[w][r] = (short) (bi < E ? bi : -1);
        }
    }
    __syncthreads();
    __shared__ unsigned int s_fetch, s_promo, s_cpu;
    __shared__ unsigned long long s_pp[8], s_cs[8];
    __shared__ int s_npf, s_pfid[8];
    __shared__ unsigned long long s_pfptr[8];
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < a.k; ++i) sum += (double) s_p[s_ids[i]];
        const float inv = (float) (a.norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        unsigned int mask = 0, fetch = 0, promo = 0, cpu = 0;
        unsigned long long* rtab = a.d.tab + a.d.n_keys;
        unsigned long long* spares = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) a.layer * kSpares;
        int next_spare = 0;
        // the CPU LANE: of this route's f RAM-tier experts, cpu_take(f) (the plan's split table) are computed by the
        // host from their RAM-tier blobs while the device pulls the others over PCIe - the COLDEST by the route counts
        // (ties: the later in route order), so the hot ones are still promoted into VRAM
        unsigned int host_set = 0;
        if (a.cpu_plan != 0ull && a.skip_from > 0) {
            int f_ram = 0, ri[8];
            unsigned int rcnt[8];
            for (int i = 0; i < a.k; ++i) {
                const size_t key = (size_t) a.layer * E + s_ids[i];
                const unsigned int c = a.d.dcnt != nullptr ? ++a.d.dcnt[key] : 0u;
                if (i < a.skip_from && a.d.tab[key] == 0ull && rtab[key] != 0ull) {
                    ri[f_ram] = i;
                    rcnt[f_ram++] = c;
                }
            }
            const int to_cpu = (int) ((a.cpu_plan >> (4 * f_ram)) & 15ull);
            for (int t = 0; t < to_cpu && t < f_ram; ++t) {
                int best = -1;
                for (int j = 0; j < f_ram; ++j)
                    if (((host_set >> ri[j]) & 1u) == 0u && (best < 0 || rcnt[j] <= rcnt[best])) best = j;
                host_set |= 1u << ri[best];
            }
        }
        for (int i = 0; i < a.k; ++i) {
            const float w = (float) ((double) s_p[s_ids[i]] * inv * a.w_scale);
            s_w[i] = w;
            const size_t key = (size_t) a.layer * E + s_ids[i];
            unsigned long long ptr = a.d.tab[key];
            unsigned long long src = 0ull;
            s_pp[i] = 0ull;
            s_cs[i] = 0ull;
            if (ptr == 0ull && i >= a.skip_from) {
                // left out: the expert kernels skip a null plan entry (a draft only has to be a good guess)
            } else if (ptr == 0ull) {
                src = rtab[key];
                if (src != 0ull && ((host_set >> i) & 1u)) {
                    // to the host: a null plan entry (the expert kernels skip it), no spare, no fetch
                    cpu |= 1u << i;
                    s_cs[i] = src;
                    src = 0ull;
                } else {
                    // not in VRAM: land it in a spare slot of this layer (it becomes resident: the table says so
                    // from here on) or, with none left, in this entry's scratch slot.  promote_min > 1: keep it
                    // only when its aged route count clears the bar - one-offs take scratch and do not churn the tier
                    const unsigned int rc = a.d.dcnt != nullptr ? a.d.dcnt[key] : 0u;
                    while (next_spare < kSpares && spares[next_spare] == 0ull) ++next_spare;
                    if (next_spare < kSpares && (a.promote_min <= 1 || rc >= (unsigned int) a.promote_min)) {
                        ptr = spares[next_spare];
                        spares[next_spare] = 0ull;
                        a.d.tab[key] = ptr;
                        promo |= 1u << i;
                        s_pp[i] = ptr;
                    } else {
                        ptr = a.d.scratch[i];
                    }
                    if (src != 0ull) fetch |= 1u << i;
                    else mask |= 1u << i;   // only on disk: the host provides the source
                }
            }
            a.d.plan_ptr[i] = ptr;
            a.d.fetch_src[i] = src;
            a.d.plan_w[i] = w;
            a.d.plan_id[i] = s_ids[i];
        }
        for (int i = a.k; i < 8; ++i) a.d.fetch_src[i] = 0ull;
        s_mask = mask;
        s_fetch = fetch;
        s_promo = promo;
        s_cpu = cpu;
        // PREFETCH: the predicted experts of the next layer that are not in VRAM but are in the RAM tier claim
        // that layer's spares now (one spare stays for its real misses); the side stream copies them while this
        // layer and the next one's attention compute, and the next route sees them as resident
        int npf = 0;
        if (a.pred_logits != nullptr && a.max_pf > 0) {
            const int L1 = a.layer + 1;
            unsigned long long* sp1 = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) L1 * kSpares;
            for (int i = 0; i < a.k && npf < a.max_pf; ++i) {
                const int e = s_pred[i];
                if (e < 0) continue;
                const size_t key = (size_t) L1 * E + e;
                if (a.d.tab[key] != 0ull) continue;
                const unsigned long long src = rtab[key];
                if (src == 0ull) continue;
                int navail = 0, first = -1;
                for (int j = 0; j < kSpares; ++j)
                    if (sp1[j] != 0ull) {
                        ++navail;
                        if (first < 0) first = j;
                    }
                if (navail < 2) break;
                const unsigned long long dst = sp1[first];
                sp1[first] = 0ull;
                a.d.tab[key] = dst;
                a.d.pf_src[npf] = src;
                a.d.pf_dst[npf] = dst;
                s_pfid[npf] = e;
                s_pfptr[npf] = dst;
                ++npf;
            }
        }
        *a.d.pf_n = npf;
        s_npf = npf;
        const unsigned int seq = *a.d.seq + 1;
        *a.d.seq = seq;
        s_seq = seq;
        *a.d.wait_seq = mask ? seq : 0u;
        if (a.d.cpu_seq != nullptr) *a.d.cpu_seq = cpu ? seq : 0u;
        *a.d.cpu_flag = 0;
    }
    __syncthreads();
    MoeRequest* rq = (MoeRequest*) a.d.ring + (s_seq % kRingSize);
    if (s_cpu != 0u)
        for (int e = tid; e < a.n_embd; e += ROUTE_THREADS) rq->x[e] = a.x[e];
    if (tid == 0) {
        rq->layer = a.layer;
        rq->miss_mask = s_mask;
        rq->fetch_mask = s_fetch;
        rq->promo_mask = s_promo;
        rq->cpu_mask = s_cpu;
        for (int i = 0; i < a.k; ++i) {
            rq->ids[i] = s_ids[i];
            rq->w[i] = s_w[i];
            rq->promo_ptr[i] = s_pp[i];
            rq->pred[i] = s_pred[i];
            rq->cpu_src[i] = s_cs[i];
        }
        rq->pf_n = s_npf;
        for (int i = 0; i < s_npf; ++i) {
            rq->pf_ids[i] = s_pfid[i];
            rq->pf_ptr[i] = s_pfptr[i];
        }
        for (int d2 = 0; d2 < kAhead; ++d2)
            for (int i = 0; i < 8; ++i) rq->ahead[d2][i] = d2 < a.n_ahead && i < a.k ? s_ah[d2][i] : (short) -1;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        rq->seq = s_seq;
#if defined(__HIPCC__)
        // The CPU polls this signal without a stream query. As in Strata #697,
        // publish the signal itself after publishing the request payload.
        __threadfence_system();
#endif
    }
}

__global__ void moe_wait_kernel(MoeDev d, int n_embd) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = *d.wait_seq;
    __syncthreads();
    if (s_need == 0) return;
    const MoeResponse* r = (const MoeResponse*) d.resp;
    if (tid == 0) {
        // spin on the host's answer (bounded: a dead host must not hang the GPU forever - ~60 s)
        const long long t0 = clock64();
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    // the disk-only experts: the host read each into host-mapped memory and answers with that SOURCE; the
    // fetch kernel copies it into the slot the route already planned
    const unsigned int pm = ((volatile const MoeResponse*) r)->ptr_mask;
    if (tid < 8 && (pm >> tid) & 1u) d.fetch_src[tid] = ((volatile const MoeResponse*) r)->ptr[tid];
    const int nu = ((volatile const MoeResponse*) r)->n_upd;
    for (int u = tid; u < nu && u < 64; u += blockDim.x)
        d.tab[((volatile const MoeResponse*) r)->upd_key[u]] = ((volatile const MoeResponse*) r)->upd_val[u];
    if (((volatile const MoeResponse*) r)->cpu) {
        for (int e = tid; e < n_embd; e += blockDim.x) d.cpu_part[e] = ((volatile const MoeResponse*) r)->cpu_part[e];
        if (tid == 0) *d.cpu_flag = 1;
    }
}

// the CPU LANE's answer for the last route (when it sent experts to the host): spin, then out += its weighted sum
__global__ void moe_cpu_wait_kernel(MoeDev d, int n_embd, float* out) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = *d.cpu_seq;
    __syncthreads();
    if (s_need == 0) return;
    const CpuAnswer* r = (const CpuAnswer*) d.cpu_ans;
    if (tid == 0) {
        const long long t0 = clock64();   // bounded like moe_wait (~60 s)
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int e = tid; e < n_embd; e += blockDim.x) out[e] += ((volatile const CpuAnswer*) r)->part[e];
}

__global__ void __launch_bounds__(256) moe_prefetch_kernel(MoeDev d, size_t n16) {
    const int n = *d.pf_n;
    for (int i = 0; i < n; ++i) {
        const uint4* s = (const uint4*) d.pf_src[i];
        uint4* t = (uint4*) d.pf_dst[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

// one block per SM, 16-byte loads: the PCIe pull measured 11.5 GB/s (a little over cudaMemcpy's 10.7)
__global__ void __launch_bounds__(256) moe_fetch_kernel(MoeDev d, int k, size_t n16) {
    for (int i = 0; i < k; ++i) {
        const unsigned long long src = d.fetch_src[i];
        if (src == 0ull) continue;
        const uint4* s = (const uint4*) src;
        uint4* t = (uint4*) d.plan_ptr[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

constexpr int GU_ROWS = 32;   // h values (gate rows = up rows) per block: one q8_1 block of h
constexpr int GU_WARPS = 16;  // ... 2 gate + 2 up rows per warp (8 per warp measured 90 vs 65 us a layer: occupancy)

// grid (n_ff / GU_ROWS, k [+ 1]): the extra row of blocks (blockIdx.y == k) computes the SHARED expert's down rows
// into sh_out while the routed experts' blocks run - in moe_down it was a serial tail (~25 us of 72)
#if defined(STRATA_USE_HIP)
template<int TG, int NW = GU_WARPS, bool LUT = false, bool GLOBAL = false, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0>
__attribute__((amdgpu_waves_per_eu(MINWAVES > 0 ? MINWAVES : 1)))
__global__ void __launch_bounds__(NW * 32)
#else
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32)
#endif
moe_gate_up_kernel(MoeDev d, int k, int n_embd, int n_ff, float limit,
                                                                    const block_q8_1* __restrict__ xq,
                                                                    block_q8_1* __restrict__ hq,
                                                                    const uint8_t* __restrict__ sh_down, int sh_type,
                                                                    const block_q8_1* __restrict__ sh_hq, int n_ff_sh,
                                                                    float* __restrict__ sh_out) {
#if !defined(STRATA_USE_HIP)
    constexpr int NW = GU_WARPS;
#endif
    const int ei = blockIdx.y, chunk = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#if defined(STRATA_USE_HIP)
    if constexpr (!SPLIT)
#endif
    if (ei == k) {
        const int per = (n_embd + (int) gridDim.x - 1) / (int) gridDim.x;
        const int r1 = min(n_embd, (chunk + 1) * per);
        for (int r = chunk * per + warp; r < r1; r += NW) {
            const float v = any_row_dot(sh_type, sh_down, r, sh_hq, nullptr, n_ff_sh, lane);
            if (lane == 0) sh_out[r] = v;
        }
        return;
    }
    const unsigned long long ptr = d.plan_ptr[ei];
    if (ptr == 0ull) return;
    __shared__ float sg[GU_ROWS], su[GU_ROWS];
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TG> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TG>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TG>(s_tab);
    __syncthreads();
    const size_t rb = rbytes<TG>(n_embd);
    const uint8_t* blob = (const uint8_t*) ptr;
    // each warp: 2 gate + 2 up rows in ONE pass (one activation load per call for all four)
    constexpr int RPW = GU_ROWS / NW;
    const uint8_t* rows[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) n_ff * rb + (size_t) r * rb;
    }
    float s[2 * RPW];
    const uint32_t* tab = s_tab;
#if defined(STRATA_USE_HIP)
    if constexpr (GLOBAL) tab = TG == 16 ? (const uint32_t*) iq2xxs_grid :
                                TG == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
    if constexpr (LUT) {
        rows_dot_rdna<TG, 2 * RPW>(rows, xq, n_embd, lane, tab, s_sign, s, DIRECT, WIDE);
    } else
#endif
    rows_dot<TG, 2 * RPW>(rows, xq, n_embd, lane, tab, s);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < RPW; ++j) {
            sg[warp * RPW + j] = s[j];
            su[warp * RPW + j] = s[RPW + j];
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float g = fminf(sg[lane], limit);
        const float u = fminf(fmaxf(su[lane], -limit), limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + (size_t) ei * (n_ff / 32) + chunk, lane);
    }
}

constexpr int DOWN_ROWS = 4;

#if defined(STRATA_USE_HIP)
template<int TD, int NR = DOWN_ROWS, bool LUT = false, bool GLOBAL = false, bool DIRECT = false, bool WIDE = false>
#else
template<int TD>
#endif
__global__ void __launch_bounds__(256) moe_down_kernel(MoeDev d, int k, int n_embd, int n_ff, size_t down_off,
                                                       const block_q8_1* __restrict__ hq,
                                                       const float* __restrict__ sh_out, float* __restrict__ out) {
#if !defined(STRATA_USE_HIP)
    constexpr int NR = DOWN_ROWS;
#endif
    // NR output rows per block; warp w computes those rows for expert w (one activation load per call
    // for all of them), the combine runs in plan order like the reference's axpy chain
    __shared__ unsigned long long s_ptr[8];
    __shared__ float s_w[8];
    __shared__ float s_part[8][NR];
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TD> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TD>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TD>(s_tab);
    if (threadIdx.x < 8) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? d.plan_ptr[threadIdx.x] : 0ull;
        s_w[threadIdx.x] = (int) threadIdx.x < k ? d.plan_w[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * NR;
    const size_t drb = rbytes<TD>(n_ff);
    {
        float s[NR];
        if (warp < k && s_ptr[warp] != 0ull) {
            const uint8_t* rows[NR];
#pragma unroll
            for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) s_ptr[warp] + down_off + (size_t) (r0 + j) * drb;
            const uint32_t* tab = s_tab;
#if defined(STRATA_USE_HIP)
            if constexpr (GLOBAL) tab = TD == 16 ? (const uint32_t*) iq2xxs_grid :
                                        TD == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
            if constexpr (LUT) {
                rows_dot_rdna<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s_sign, s, DIRECT, WIDE);
            } else
#endif
            rows_dot<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s);
        } else {
#pragma unroll
            for (int j = 0; j < NR; ++j) s[j] = 0.0f;
        }
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < NR; ++j) s_part[warp][j] = s[j];
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        const int j = threadIdx.x, r = r0 + j;
        float m = 0.0f;
        for (int i = 0; i < k; ++i) m += s_w[i] * s_part[i][j];
        if (*d.cpu_flag) m += d.cpu_part[r];
        out[r] = m + (sh_out != nullptr ? sh_out[r] : 0.0f);
    }
}

// ---------------------------------------------------------------- the prompt path's lightly routed experts
// An expert with a few rows (tokens) of a prompt chunk: its weights read ONCE, each 32-value sub-block decoded once
// and dotted with up to LR_MT tokens at a time (MMQ computes tiles of 64-128 tokens - mostly empty at ~10 rows an
// expert, as at a 300-token prompt).  The sub-block decode is the decode path's dw<T>, split from its dot.
constexpr int LR_MT = 4;
#ifndef GLMF_LR_MT2
#define GLMF_LR_MT2 8
#endif
constexpr int LR_MT2 = GLMF_LR_MT2;   // tokens a pass of the light kernels (rows_multi2)
template<int T>
__device__ __forceinline__ void dwm(const uint8_t* row, int kbx, int iqs, const XV* x, int nt, const uint32_t* tab,
                                    float* acc) {
    int g[8];
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            g[l0] = (grid >> 0) & 0x0F0F0F0F;
            g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            acc[t] += d1q * (x[t].d * sumi + x[t].s * delta);
        }
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 28;
        const float d = __half2float(bq3->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = (ls * sumi + sumi / 2) / 2;
            acc[t] += d * x[t].d * sumi;
        }
    } else {   // 16: IQ2_XXS
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 27 | 1;
        const float d = __half2float(bq2->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = sumi * ls / 8;
            acc[t] += d * x[t].d * sumi;
        }
    }
}

// one 32-value sub-block decoded (dwm's first half), then dotted with any number of activations
struct Dec {
    int g[8];
    float a, b;   // IQ1_S: d1q, delta; IQ3_XXS / IQ2_XXS: d, -
    int ls;
};
template<int T>
__device__ __forceinline__ void dec_sub(const uint8_t* row, int kbx, int iqs, const uint32_t* tab, Dec& o) {
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            o.g[l0] = (grid >> 0) & 0x0F0F0F0F;
            o.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        o.a = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        o.b = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 28;
        o.a = __half2float(bq3->d);
    } else {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 27 | 1;
        o.a = __half2float(bq2->d);
    }
}
template<int T>
__device__ __forceinline__ float dec_dot(const Dec& o, const XV& x) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumi = __dp4a(o.g[i], x.u[i], sumi);
    if constexpr (T == 19) return o.a * (x.d * sumi + x.s * o.b);
    else if constexpr (T == 18) return o.a * x.d * ((o.ls * sumi + sumi / 2) / 2);
    else return o.a * x.d * (sumi * o.ls / 8);
}

// NR weight rows x MT tokens a pass: each k step decodes the NR rows' sub-blocks once, then every token's activation
// is loaded once and dotted with all NR (dwm decoded per row and per token group)
template<int T, int NR, int MT, typename XF>
__device__ __forceinline__ void rows_multi2(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                            XF x_of, float (&out)[NR][MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        Dec dr[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) dec_sub<T>(rows[r], kbx, iqs, tab, dr[r]);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nt) break;
            const XV xv = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
            for (int r = 0; r < NR; ++r) out[r][t] += dec_dot<T>(dr[r], xv);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// NR weight rows (pointers) x the rows' tokens, LR_MT tokens a pass; x(t) = the q8_1 activation of token t
template<int T, int NR, typename XF>
__device__ __forceinline__ void rows_multi(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                           XF x_of, float (&out)[NR][LR_MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(LR_MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        XV xv[LR_MT];
#pragma unroll
        for (int t = 0; t < LR_MT; ++t)
            if (t < nt) xv[t] = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
        for (int r = 0; r < NR; ++r) dwm<T>(rows[r], kbx, iqs, xv, nt, tab, out[r]);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// light[i] = {slot, r0, nr}: grid (n_ff / 32, n); 16 warps x (1 gate + 1 up row)... as the decode kernel: 2 + 2
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_gate_up_kernel(const uint8_t* __restrict__ base, size_t stride,
                                                                     const int* __restrict__ light,
                                                                     const int* __restrict__ row_tok,
                                                                     const block_q8_1* __restrict__ xq, int n_embd,
                                                                     int n_ff, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    load_tab<TG>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = base + (size_t) slot * stride;
    const size_t rb = rbytes<TG>(n_embd);
    constexpr int RPW = GU_ROWS / GU_WARPS;
    const uint8_t* rows[2 * RPW];
    int col[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) (n_ff + r) * rb;
        col[j] = r;
        col[RPW + j] = n_ff + r;
    }
    const int nblk = n_embd / 32;
    const auto x_of = [&](int t) { return xq + (size_t) row_tok[R0 + t] * nblk; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[2 * RPW][LR_MT2];
        rows_multi2<TG, 2 * RPW, LR_MT2>(rows, n_embd, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < 2 * RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// h = swiglu(gate, up) of the light rows [rlo, rhi) (GU at out rows, ld floats apart) -> q8_1 rows (n_ff / 32 each)
__global__ void rows_swiglu_q8_kernel(const float* __restrict__ gu, int ld, int rlo, int rhi, int n_ff, float limit,
                                      block_q8_1* __restrict__ hq) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n_ff / 32;
    const int64_t r = rlo + w / per, b = w % per;
    if (r >= rhi) return;
    const float* row = gu + (size_t) r * ld;
    const float gv = fminf(row[b * 32 + lane], limit);
    const float uv = fminf(fmaxf(row[n_ff + b * 32 + lane], -limit), limit);
    q8_1_store_warp((gv / (1.0f + expf(-gv))) * uv, hq + (size_t) (r - rlo) * per + b, lane);
}

// the down rows: grid (n_embd / 32, n); 16 warps x 2 rows; out row r (ld floats) cols [0, n_embd)
template<int TD>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_down_kernel(const uint8_t* __restrict__ base, size_t stride,
                                                                  size_t down_off, const int* __restrict__ light,
                                                                  const block_q8_1* __restrict__ hq, int rlo, int n_ff,
                                                                  int n_embd, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = base + (size_t) slot * stride + down_off;
    const size_t rb = rbytes<TD>(n_ff);
    constexpr int RPW = 2;
    const uint8_t* rows[RPW];
    int col[RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        col[j] = chunk * (GU_WARPS * RPW) + warp * RPW + j;
        rows[j] = blob + (size_t) col[j] * rb;
    }
    const int per = n_ff / 32;
    const auto x_of = [&](int t) { return hq + (size_t) (R0 + t - rlo) * per; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[RPW][LR_MT2];
        rows_multi2<TD, RPW, LR_MT2>(rows, n_ff, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// x rows (ld floats apart) -> q8_1 rows (n / 32 blocks each): a warp a block
__global__ void rows_q8_kernel(const float* __restrict__ x, int ld, int rows, int n, block_q8_1* __restrict__ q) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n / 32;
    if (w >= (int64_t) rows * per) return;
    const int64_t r = w / per, b = w % per;
    q8_1_store_warp(x[(size_t) r * ld + b * 32 + lane], q + (size_t) r * per + b, lane);
}

// ---------------------------------------------------------------- the NextN block's glue
// cat = [rms(emb) * enorm, rms(h) * hnorm] as q8_1 (eh_proj's input); n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) mtp_in_kernel(const float* __restrict__ emb, const float* __restrict__ h,
                                                      const float* __restrict__ enorm, const float* __restrict__ hnorm,
                                                      float eps, int n, block_q8_1* __restrict__ catq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float a[4], b[4];
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        a[i] = e < n ? emb[e] : 0.0f;
        b[i] = e < n ? h[e] : 0.0f;
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sred);
    sb = block_sum(sb, sred);
    const float ia = rsqrtf(sa / (float) n + eps), ib = rsqrtf(sb / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        q8_1_store_warp(a[i] * ia * enorm[e], catq + e / 32, lane);
        q8_1_store_warp(b[i] * ib * hnorm[e], catq + (n + e) / 32, lane);
    }
}
// h += add (when given), then x = rms(h) * w and its q8_1; n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) rms_q8_kernel(float* __restrict__ h, const float* __restrict__ add,
                                                      const float* __restrict__ w, float eps, int n,
                                                      float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        float t = e < n ? h[e] : 0.0f;
        if (add != nullptr && e < n) {
            t += add[e];
            h[e] = t;
        }
        v[i] = t;
        ss += t * t;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        const float xv = v[i] * inv * w[e];
        x[e] = xv;
        q8_1_store_warp(xv, xq + e / 32, lane);
    }
}

// one block: float4 loads, four in flight per thread (the vocabulary is ~0.6 MB; scalar loads ran at ~10 GB/s)
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    const auto take = [&](float v, int i) {
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    };
    const int B = (int) blockDim.x;
    int tail = 0;
    if ((((uintptr_t) x) & 15u) == 0) {
        const float4* x4 = (const float4*) x;
        const int n4 = n >> 2;
        int i = threadIdx.x;
        for (; i + 3 * B < n4; i += 4 * B) {
            float4 a[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) a[u] = x4[i + u * B];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = 4 * (i + u * B);
                take(a[u].x, b);
                take(a[u].y, b + 1);
                take(a[u].z, b + 2);
                take(a[u].w, b + 3);
            }
        }
        for (; i < n4; i += B) {
            const float4 a = x4[i];
            take(a.x, 4 * i);
            take(a.y, 4 * i + 1);
            take(a.z, 4 * i + 2);
            take(a.w, 4 * i + 3);
        }
        tail = 4 * n4;
    }
    for (int i = tail + threadIdx.x; i < n; i += B) take(x[i], i);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int) (blockDim.x >> 5); ++i)
            if (sv[i] > best || (sv[i] == best && si[i] < bi)) { best = sv[i]; bi = si[i]; }
        *out = bi;
    }
}

int mv_type_ok(int t) {
    switch (t) {
        case kTypeF32: case kTypeBF16: case 12: case 13: case 14: case 8: case 16: case 18: case 19: case 23: case 10: case 11:
        case 17: case 22: case 21: case 29: return 1;
        default: return 0;
    }
}

// a failed launch is STICKY: the forward that issued it reports an error (launch_errors) instead of computing on
std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_fast %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace

bool mv_supported(int type) { return mv_type_ok(type) != 0; }

bool moe_supported(int type) {   // moe_gate_up's and moe_down's instantiations
    return type == 10 || type == 11 || type == 12 || type == 13 || type == 14 || type == 16 || type == 17 ||
           type == 18 || type == 19 || type == 21 || type == 22 || type == 23 || type == 29;
}

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

size_t row_bytes(int type, int64_t n_in) {
    switch (type) {
        case kTypeF32: return (size_t) n_in * 4;
        case kTypeBF16: return (size_t) n_in * 2;
        case 12: return (size_t) (n_in / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n_in / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n_in / 256) * sizeof(block_q6_K);
        case 8: return (size_t) (n_in / 32) * sizeof(block_q8_0);
        case 16: return (size_t) (n_in / 256) * sizeof(block_iq2_xxs);
        case 18: return (size_t) (n_in / 256) * sizeof(block_iq3_xxs);
        case 19: return (size_t) (n_in / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n_in / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n_in / 256) * sizeof(block_q2_K);
        case 11: return (size_t) (n_in / 256) * sizeof(block_q3_K);
        case 17: return (size_t) (n_in / 256) * sizeof(block_iq2_xs);
        case 22: return (size_t) (n_in / 256) * sizeof(block_iq2_s);
        case 21: return (size_t) (n_in / 256) * sizeof(block_iq3_s);
        case 29: return (size_t) (n_in / 256) * sizeof(block_iq1_m);
        default: return 0;
    }
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    for (int i = 0; i < n; ++i)
        if (!mv_type_ok(jobs[i].type)) {
            std::fprintf(stderr, "glm_fast mv: type %d unsupported\n", jobs[i].type);
            return false;
        }
    // the jobs of the common dense types go to a kernel compiled for their type (one launch per type present), the
    // rest to the generic one; the jobs are independent, so their order does not matter
    static const bool typed = getenv("STRATA_GLM_MV_GENERIC") == nullptr;
    constexpr int kTyped[] = {14, 8, 12, 13};
    bool done[kMaxMvJobs] = {};
    if (typed) {
        for (const int T : kTyped) {
            MvBatch b{};
            int acc = 0, m = 0;
            for (int i = 0; i < n; ++i)
                if (jobs[i].type == T) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            switch (T) {
                case 14: mv_kernel_t<14><<<acc, 256, 0, s>>>(b); break;
                case 8: mv_kernel_t<8><<<acc, 256, 0, s>>>(b); break;
                case 12: mv_kernel_t<12><<<acc, 256, 0, s>>>(b); break;
                case 13: mv_kernel_t<13><<<acc, 256, 0, s>>>(b); break;
            }
        }
    }
    MvBatch b{};
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        b.j[m] = jobs[i];
        acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
        b.blk_end[m++] = acc;
    }
    if (m > 0) {
        b.n = m;
        mv_kernel<<<acc, 256, 0, s>>>(b);
    }
    launch_check("mv");
    return true;
}

void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s) {
    quantize_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, (block_q8_1*) xq, n);
    launch_check("quantize");
}

void hc(const HcArgs& a, cudaStream_t s) {
    // the grid-barrier kernel needs every block resident at once; checked once per process (and per device count
    // of SMs), the single-tail kernel otherwise.  STRATA_GLM_HC1=1 forces the old one (A/B).
    static int mode = -1;
    if (mode < 0) {
        int dev = 0, sms = 0, per = 0, per3 = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, hc2_kernel, HC_THREADS, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per3, hc3_kernel, HC_THREADS, 0);
        mode = (getenv("STRATA_GLM_HC1") == nullptr && per * sms >= a.n_embd / HC2_EB) ? 2 : 1;
        if (mode == 2 && getenv("STRATA_GLM_HC2") == nullptr && a.n_embd == 4096 && per3 * sms >= a.n_embd / HC2_EB)
            mode = 3;
    }
    if (mode == 3) {
        hc3_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(a);
        launch_check("hc3");
        return;
    }
    if (mode == 2 && a.n_embd % HC2_EB == 0) {
        hc2_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(a);
        launch_check("hc2");
        return;
    }
    const int blocks = 4 * a.n_embd / HC_CHUNK;
    hc_kernel<<<blocks, HC_THREADS, 0, s>>>(a);
    launch_check("hc");
}

void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd,
             float* R_new, cudaStream_t s) {
    hc_post_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(block_out, R_old, post, comb, n_embd, R_new);
    launch_check("hc_post");
}

void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s) {
    head_prep_kernel<<<1, 1024, 0, s>>>(R, norm_w, eps, n_embd, x, (block_q8_1*) xq);
    launch_check("head_prep");
}

void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s) {
    embed_streams_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(emb, R, n_embd);
    launch_check("embed_streams");
}

void kda_prep(const KdaPrepArgs& a, cudaStream_t s) {
    kda_prep_kernel<<<dim3(a.n_head, 11), 128, 0, s>>>(a);   // 3 conv roles + 2 x 4 gate quarters
    launch_check("kda_prep");
}

void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
             const float* g2, const float* norm_w, float eps, int n_head, int head_dim, void* out_q8_1,
             cudaStream_t s) {
    if (head_dim != 128) {
        std::fprintf(stderr, "glm_fast kda_rec: head_dim %d (the kernel is built for 128)\n", head_dim);
        return;
    }
    kda_rec_kernel<<<n_head, 256, 0, s>>>(q, k, v, g1, beta_raw, state, g2, norm_w, eps, n_head,
                                          (block_q8_1*) out_q8_1);
    launch_check("kda_rec");
}

void dsa_prep(const DsaPrepArgs& a, cudaStream_t s) {
    dsa_prep_kernel<<<3, 512, 0, s>>>(a);
    launch_check("dsa_prep");
}

void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s) {
    if (n_pools <= 0) return;
    const size_t smem = ((size_t) idx_heads * key_dim + idx_heads) * sizeof(float);
    dsa_score_kernel<<<(n_pools + 7) / 8, 256, smem, s>>>(iq, pooled, iw, key_dim, idx_heads, n_pools, score);
    launch_check("dsa_score");
}

void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s) {
    dsa_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, kpool, top_pools, n_sel, pos, cells);
    launch_check("dsa_select");
}

void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8) {
    // scratch for q_abs and ctx (n_head * kv_lora each), per device
    static float* scratch[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (getenv("STRATA_GLM_MLA1") != nullptr || dev >= 16 || kv_lora % 32 || v_head % 32) {
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8);
        launch_check("mla");
        return;
    }
    if (scratch[dev] == nullptr && cudaMalloc(&scratch[dev], (size_t) 2 * n_head * kv_lora * sizeof(float)) != cudaSuccess) {
        scratch[dev] = nullptr;
        cudaGetLastError();
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8);
        launch_check("mla");
        return;
    }
    float* q_abs = scratch[dev];
    float* ctx = q_abs + (size_t) n_head * kv_lora;
    headwise_gemv_kernel<<<n_head * kv_lora / 32, 256, 0, s>>>(wk_b, q, kv_lora, qk_nope, q_abs, nullptr);
    // the split attention: chunks of MLA_CHUNK cells, partials per (chunk, head), then the merge
    static float* part[16] = {};
    static int part_chunks[16] = {};
    const int n_chunks = (n_sel + MLA_CHUNK - 1) / MLA_CHUNK;
    const size_t smem = (size_t) MLA_CHUNK * kv_lora * sizeof(uint16_t) + (size_t) MLA_HPB * MLA_CHUNK * sizeof(float);
    // the opt-in must be EXACTLY what the launch needs (or at most the device's opt-in minus the kernel's static
    // shared memory): asking for the whole 96 KB fails on a V100 (s_cell is static), and every launch then failed
    // with "invalid argument" - the attention silently contributed nothing (fixed 2026-10-06)
    static int attr_smem[16] = {};
    static bool split_ok[16] = {true, true, true, true, true, true, true, true,
                                true, true, true, true, true, true, true, true};
    if (split_ok[dev] && attr_smem[dev] < (int) smem) {
        if (cudaFuncSetAttribute(mla_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) == cudaSuccess) {
            attr_smem[dev] = (int) smem;
        } else {
            cudaGetLastError();
            split_ok[dev] = false;
            std::fprintf(stderr, "glm_fast mla: the split kernel cannot get %zu bytes of shared memory - the one-pass "
                                 "attention instead\n", smem);
        }
    }
    if (split_ok[dev] && kv_lora <= 512 && n_head <= 128 && getenv("STRATA_GLM_MLA_ATTN1") == nullptr) {
        if (part_chunks[dev] < n_chunks) {
            if (part[dev]) cudaFree(part[dev]);
            const int want = std::max(n_chunks, 80);
            if (cudaMalloc(&part[dev], (size_t) want * n_head * (kv_lora + 2) * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                part[dev] = nullptr;
                part_chunks[dev] = 0;
            } else {
                part_chunks[dev] = want;
            }
        }
    }
    if (split_ok[dev] && part[dev] != nullptr && part_chunks[dev] >= n_chunks && kv_lora <= 512 &&
        getenv("STRATA_GLM_MLA_ATTN1") == nullptr) {
        mla_split_kernel<<<dim3(n_chunks, (n_head + MLA_HPB - 1) / MLA_HPB), 256, smem, s>>>(
            q_abs, lat, cells, n_sel, n_head, qk_nope, kv_lora, part[dev], lat_q8);
        mla_combine_kernel<<<n_head, 256, 0, s>>>(part[dev], n_chunks, n_head, kv_lora, ctx);
    } else {
        mla_attn_kernel<<<n_head, 256, ((size_t) kv_lora + n_sel) * sizeof(float), s>>>(q_abs, lat, cells, n_sel,
                                                                                        qk_nope, kv_lora, ctx, lat_q8);
    }
    headwise_gemv_kernel<<<n_head * v_head / 32, 256, 0, s>>>(wv_b, ctx, v_head, kv_lora, nullptr,
                                                              (block_q8_1*) out_q8_1);
    launch_check("mla2");
}

void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s) {
    swiglu_q8_kernel<<<(n + 255) / 256, 256, 0, s>>>(gate, up, limit, n, (block_q8_1*) hq);
    launch_check("swiglu_q8");
}

void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int layer,
               const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up, float sh_limit,
               int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits, const float* pred_bias,
               int max_prefetch, const float* ahead_logits, const float* const* ahead_bias, int n_ahead,
               int skip_from, unsigned long long cpu_plan, int promote_min) {
    RouteArgs a{logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, layer, x, n_embd, d,
                sh_gate, sh_up, sh_limit, n_ff_sh, (block_q8_1*) sh_hq, pred_logits, pred_bias, max_prefetch,
                ahead_logits, {}, 0, skip_from, d.cpu_seq != nullptr ? cpu_plan : 0ull, promote_min};
    if (ahead_logits != nullptr && ahead_bias != nullptr && n_expert <= 512) {
        a.n_ahead = std::min(n_ahead, kAhead);
        for (int i = 0; i < a.n_ahead; ++i) a.ahead_bias[i] = ahead_bias[i];
    }
    const int blocks = 1 + (n_ff_sh + ROUTE_THREADS - 1) / ROUTE_THREADS;
    moe_route_kernel<<<blocks, ROUTE_THREADS, 0, s>>>(a);
    launch_check("moe_route");
}

void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s) {
    moe_wait_kernel<<<1, 256, 0, s>>>(d, n_embd);
    launch_check("moe_wait");
}

void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s) {
    if (d.cpu_seq == nullptr) return;
    moe_cpu_wait_kernel<<<1, 256, 0, s>>>(d, n_embd, out);
    launch_check("moe_cpu_wait");
}

void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    // half the SMs: it runs beside the main stream's kernels
    moe_prefetch_kernel<<<std::max(1, sms / 2), 256, 0, s>>>(d, (blob_bytes + 15) / 16);
    launch_check("moe_prefetch");
}

void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    moe_fetch_kernel<<<sms, 256, 0, s>>>(d, k, (blob_bytes + 15) / 16);
    launch_check("moe_fetch");
}

#if defined(STRATA_USE_HIP)
namespace {
// Limit automatic dispatch to the measured decode geometry and RDNA3/3.5
// targets. RDNA4 and all other shapes retain the original expert kernels.
bool rdna3_expert_device(bool allow_gfx1151 = true) {
    static const bool legacy = [] {
        const char* v = std::getenv("STRATA_HIP_EXPERTS_LEGACY");
        return v && v[0] == '1';
    }();
    if (legacy) return false;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool gfx1100 = false;
    static thread_local bool gfx1151 = false;
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        gfx1100 = std::strncmp(prop.gcnArchName, "gfx1100", 7) == 0;
        gfx1151 = std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        cached_device = device;
    }
    return gfx1100 || (allow_gfx1151 && gfx1151);
}
// Splitting shared down removes its dynamic type switch from routed gate/up's
// register budget. The routed grid keeps exactly the original dot/reduction order.
__global__ void __launch_bounds__(256) rdna_shared_down_kernel(const uint8_t* weights, int type,
                                                         const block_q8_1* h, int n_in,
                                                         int n_out, float* out) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + wave;
    if (row < n_out) {
        const float v = any_row_dot(type, weights, row, h, nullptr, n_in, lane);
        if (lane == 0) out[row] = v;
    }
}
template<int T, int NW, bool GLOBAL = false, bool LUT = true, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0>
void launch_expert_gu(const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
               const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s) {
    const bool shared = sh_down && sh_out;
    const dim3 grid(n_ff / GU_ROWS, k + (shared && !SPLIT ? 1 : 0));
    moe_gate_up_kernel<T, NW, LUT, GLOBAL, SPLIT, DIRECT, WIDE, MINWAVES><<<grid, NW * 32, 0, s>>>(d, shared ? k : k + 1, n_embd, n_ff,
        limit, (const block_q8_1*) xq, (block_q8_1*) hq, (const uint8_t*) sh_down, sh_type,
        (const block_q8_1*) sh_hq, n_ff_sh, sh_out);
    if constexpr (SPLIT) {
        if (shared) rdna_shared_down_kernel<<<(n_embd + 7) / 8, 256, 0, s>>>(
            (const uint8_t*) sh_down, sh_type, (const block_q8_1*) sh_hq, n_ff_sh, n_embd, sh_out);
    }
}
template<int T, int NR, bool GLOBAL = false, bool LUT = true, bool DIRECT = false, bool WIDE = false>
void launch_expert_down(const MoeDev& d, int k, int n_embd, int n_ff, size_t off, const void* hq,
               const float* sh, float* out, cudaStream_t s) {
    moe_down_kernel<T, NR, LUT, GLOBAL, DIRECT, WIDE><<<n_embd / NR, 256, 0, s>>>(d, k, n_embd, n_ff, off,
        (const block_q8_1*) hq, sh, out);
}
} // namespace
#endif

void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    if (gu_type == 16 && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device()) {
        // Packed sign expansion, original 16-wave/four-row geometry, grid in LDS.
        launch_expert_gu<16, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
            xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s);
        launch_check("moe_gate_up RDNA3");
        return;
    }
#endif
    const bool sh = sh_down != nullptr && sh_out != nullptr;
    const dim3 grid((unsigned) (n_ff / GU_ROWS), (unsigned) (k + (sh ? 1 : 0)));
    const auto* X = (const block_q8_1*) xq;
    auto* H = (block_q8_1*) hq;
    const auto* SD = (const uint8_t*) sh_down;
    const auto* SH = (const block_q8_1*) sh_hq;
    const int kk = sh ? k : k + 1;   // no shared expert: no block has blockIdx.y == kk
    switch (gu_type) {
#define GLMF_GU(T) \
        case T: moe_gate_up_kernel<T><<<grid, GU_WARPS * 32, 0, s>>>(d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, \
                                                                     n_ff_sh, sh_out); break;
        GLMF_GU(16) GLMF_GU(18) GLMF_GU(19) GLMF_GU(23) GLMF_GU(10) GLMF_GU(11) GLMF_GU(17) GLMF_GU(22) GLMF_GU(21) GLMF_GU(29)
        GLMF_GU(12) GLMF_GU(13) GLMF_GU(14)
#undef GLMF_GU
        default: std::fprintf(stderr, "glm_fast moe_gate_up: type %d unsupported\n", gu_type); return;
    }
    launch_check("moe_gate_up");
}

#if defined(STRATA_USE_HIP)
namespace hip_expert_bench {

void gate_up(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, float limit,
             const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq,
             int n_ff_sh, float* sh_out, cudaStream_t s) {
    if (v == Variant::production) return moe_gate_up(type, d, k, n_embd, n_ff, limit, xq, hq,
                                                   sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s);
#define BENCH_GU(T) case T: \
    if (v == Variant::baseline) launch_expert_gu<T,16,false,false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_lds) launch_expert_gu<T,16,false,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_signs) launch_expert_gu<T,16,true,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::vector_load) launch_expert_gu<T,16,true,true,false,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::occupancy8) launch_expert_gu<T,16,true,true,true,false,false,8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::split_shared) launch_expert_gu<T, 16, true, true, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_plain) launch_expert_gu<T, 16, true, false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_waves8) launch_expert_gu<T, 8, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::waves8) launch_expert_gu<T, 8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_grid) launch_expert_gu<T, 16, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else launch_expert_gu<T, 16>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); break;
    switch (type) { BENCH_GU(16) BENCH_GU(18) BENCH_GU(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_GU
    launch_check("expert bench gate/up");
}

void down(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, size_t off,
          const void* hq, const float* sh, float* out, cudaStream_t s) {
    if (v == Variant::production) return moe_down(type,d,k,n_embd,n_ff,off,hq,sh,out,s);
#define BENCH_DN(T) case T: \
    if (v == Variant::baseline) launch_expert_down<T,4,false,false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds) launch_expert_down<T,2,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_rows4) launch_expert_down<T,4,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds_rows4) launch_expert_down<T,4,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_signs) launch_expert_down<T,2,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::vector_load) launch_expert_down<T,2,true,true,false,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_plain) launch_expert_down<T, 4, true, false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows2) launch_expert_down<T, 2, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows8) launch_expert_down<T, 8, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows2) launch_expert_down<T, 2>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows8) launch_expert_down<T, 8>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_grid) launch_expert_down<T, 4, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else launch_expert_down<T, 4>(d,k,n_embd,n_ff,off,hq,sh,out,s); break;
    switch (type) { BENCH_DN(16) BENCH_DN(18) BENCH_DN(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_DN
    launch_check("expert bench down");
}
} // namespace hip_expert_bench
#endif

bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n,
                  const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi,
                  void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s) {
    const auto ok = [](int t) { return t == 16 || t == 18 || t == 19; };
    if (!ok(gu_type) || !ok(d_type) || n_embd % 256 || n_ff % (GU_ROWS) || n_embd % (GU_WARPS * 2)) return false;
    if (n <= 0 || rhi <= rlo) return true;
    auto* xq = (block_q8_1*) xq_scratch;
    auto* hq = (block_q8_1*) hq_scratch;
    rows_q8_kernel<<<(unsigned) (((int64_t) n_tok * (n_embd / 32) * 32 + 255) / 256), 256, 0, s>>>(x, n_embd, n_tok,
                                                                                                 n_embd, xq);
    const dim3 ggu((unsigned) (n_ff / GU_ROWS), (unsigned) n), gdn((unsigned) (n_embd / (GU_WARPS * 2)), (unsigned) n);
    switch (gu_type) {
        case 16: rows_gate_up_kernel<16><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        case 18: rows_gate_up_kernel<18><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        default: rows_gate_up_kernel<19><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
    }
    rows_swiglu_q8_kernel<<<(unsigned) (((int64_t) (rhi - rlo) * (n_ff / 32) * 32 + 255) / 256), 256, 0, s>>>(
        out, ld, rlo, rhi, n_ff, limit, hq);
    switch (d_type) {
        case 16: rows_down_kernel<16><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        case 19: rows_down_kernel<19><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        default: rows_down_kernel<18><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
    }
    launch_check("rows_experts");
    return true;
}

void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    // IQ3_XXS down regresses on gfx1151; let it use the original dispatch below.
    if ((d_type == 22 || d_type == 18) && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device(d_type == 22)) {
        // Two rows lower the register budget; IQ2_S prefers global grid reads,
        // while the smaller IQ3_XXS grid is faster in LDS on gfx1100.
        if (d_type == 22) launch_expert_down<22, 2, true, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s);
        else launch_expert_down<18, 2, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s);
        launch_check("moe_down RDNA3");
        return;
    }
#endif
    const int blocks = n_embd / DOWN_ROWS;   // n_embd % DOWN_ROWS == 0 (4096)
    const auto* H = (const block_q8_1*) hq;
    switch (d_type) {
#define GLMF_DN(T) \
        case T: moe_down_kernel<T><<<blocks, 256, 0, s>>>(d, k, n_embd, n_ff, down_off, H, sh_out, out); break;
        GLMF_DN(16) GLMF_DN(18) GLMF_DN(19) GLMF_DN(23) GLMF_DN(10) GLMF_DN(11) GLMF_DN(17) GLMF_DN(22) GLMF_DN(21) GLMF_DN(29)
        GLMF_DN(12) GLMF_DN(13) GLMF_DN(14)
#undef GLMF_DN
        default: std::fprintf(stderr, "glm_fast moe_down: type %d unsupported\n", d_type); return;
    }
    launch_check("moe_down");
}

namespace {
__global__ void tab_update_kernel(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tab[keys[i]] = vals[i];
}
}  // namespace

void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s) {
    if (n <= 0) return;
    tab_update_kernel<<<(n + 255) / 256, 256, 0, s>>>(tab, keys, vals, n);
    launch_check("tab_update");
}

void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s) {
    mtp_in_kernel<<<1, 1024, 0, s>>>(emb, h, enorm, hnorm, eps, n, (block_q8_1*) cat_q8_1);
    launch_check("mtp_in");
}

void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s) {
    rms_q8_kernel<<<1, 1024, 0, s>>>(h, add, w, eps, n, x, (block_q8_1*) xq);
    launch_check("rms_q8");
}

void argmax(const float* x, int n, int* out, cudaStream_t s) {
    argmax_kernel<<<1, 1024, 0, s>>>(x, n, out);
    launch_check("argmax");
}

}  // namespace strata::kernels::glmf
