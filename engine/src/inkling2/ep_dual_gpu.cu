// engine/src/inkling2/ep_dual_gpu.cu - Dual-GPU Expert Parallelism (--gpus 2 / --ep-split / IK2_EP_SPLIT)
// + Single-GPU Hybrid Graph-Doorbell NUMA Offload + Native SF4/MX8/S_K/GGUF_RAW + K8V4/K8V8 CUDA Backend.
//
// MIT License
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Copyright (c) 2026 Baptiste Rajaut, merbanan, and the Strata contributors
#include "ep_dual_gpu.h"
#include "hybrid_doorbell.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#define EP_CK(call)                                                                                                  \
    do {                                                                                                             \
        cudaError_t _e = (call);                                                                                     \
        if (_e != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "[inkling2-ep] CUDA error %d (%s) at %s:%d\n", (int)_e, cudaGetErrorString(_e),     \
                         __FILE__, __LINE__);                                                                        \
            return false;                                                                                            \
        }                                                                                                            \
    } while (0)

#define EP_CK_VOID(call)                                                                                             \
    do {                                                                                                             \
        cudaError_t _e = (call);                                                                                     \
        if (_e != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "[inkling2-ep] CUDA error %d (%s) at %s:%d\n", (int)_e, cudaGetErrorString(_e),     \
                         __FILE__, __LINE__);                                                                        \
        }                                                                                                            \
    } while (0)

namespace ink2 {
namespace {

using namespace strata::inkling2;
using namespace strata::quant::sk;

// ---- Device math helpers (portable sm_70 .. sm_121a) ----
__device__ __forceinline__ float d_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ float d_fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float d_e2m1_nib(uint32_t nib) {
    const uint32_t mag = nib & 7u;
    const uint32_t x2 = (0xC8643210u >> (mag * 4u)) & 0xFu;
    float v = 0.5f * static_cast<float>(x2);
    return (nib & 8u) ? -v : v;
}

__device__ __forceinline__ float d_e4m3(uint8_t c) {
    const uint32_t sgn = (static_cast<uint32_t>(c) & 0x80u) << 24;
    const uint32_t mag = static_cast<uint32_t>(c) & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    if (mag == 0u) return __uint_as_float(sgn);
    const uint32_t exp = mag >> 3;
    const uint32_t man = mag & 7u;
    if (exp == 0u) {
        float v = static_cast<float>(man) * (1.0f / 512.0f);
        return sgn ? -v : v;
    }
    return __uint_as_float(sgn | ((exp + 120u) << 23) | (man << 20));
}

__device__ __forceinline__ float d_ue8m0(uint8_t e) {
    if (e == 255u) return 0.0f;
    if (e == 0u) return 5.877471754111438e-39f;
    return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

__device__ __forceinline__ float d_situ(float g, float u) {
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float ga = 4.0f * tanhf(g * 0.25f) * sig;
    const float ua = 25.0f * tanhf(u * 0.04f);
    return ga * ua;
}

// IQ4_NL codebook (ggml MIT)
__device__ __constant__ int8_t c_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// S1.25 codebook (exact match to sk_codecs.cpp)
__device__ __constant__ int8_t c_s125[32][4] = {
    {-1,-1,-1,-1},{-1,-1,-1, 1},{-1,-1, 1,-1},{-1,-1, 1, 1},
    {-1, 1,-1,-1},{-1, 1,-1, 1},{-1, 1, 1,-1},{-1, 1, 1, 1},
    { 1,-1,-1,-1},{ 1,-1,-1, 1},{ 1,-1, 1,-1},{ 1,-1, 1, 1},
    { 1, 1,-1,-1},{ 1, 1,-1, 1},{ 1, 1, 1,-1},{ 1, 1, 1, 1},
    { 2, 0, 0, 0},{-2, 0, 0, 0},{ 0, 2, 0, 0},{ 0,-2, 0, 0},
    { 0, 0, 2, 0},{ 0, 0,-2, 0},{ 0, 0, 0, 2},{ 0, 0, 0,-2},
    { 0, 0, 0, 0},{ 2, 2, 0, 0},{-2,-2, 0, 0},{ 0, 0, 2, 2},
    { 0, 0,-2,-2},{ 2,-2, 0, 0},{ 0, 0, 2,-2},{ 2, 0, 2, 0},
};

// ============================================================================
// Expert Device Descriptor & GPU Kernels (SF4, MX8, S_K, GGUF_RAW Q8_0/Q4_K/Q6_K/IQ4_NL/IQ4_XS/MXFP4/BF16/F16)
// ============================================================================
struct DevExpert {
    const uint8_t* body;
    uint32_t dn_off;
    uint8_t gu_code;
    uint8_t dn_code;
    uint8_t owner_gpu;  // 0 or 1
    uint8_t resident;   // 1 = VRAM on owner_gpu, 0 = host (CPU doorbell)
};

// Single-expert SF4 w13 [2*inter, hidden] + SITU -> act [inter] (1 warp per (gate, up) pair!)
__global__ void k_expert_sf4_gu_situ(
    const uint8_t* __restrict__ body,
    const float* __restrict__ x,  // [m, hidden]
    float* __restrict__ act,      // [m, inter]
    int hidden,
    int inter) {
    const int j = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (j >= inter) return;
    const int lane = threadIdx.x & 31;

    float hdr0 = *reinterpret_cast<const float*>(body + 0);
    float hdr1 = *reinterpret_cast<const float*>(body + 4);
    const float outer = hdr0 / hdr1;
    const uint8_t* w_p = body + 16;
    const uint8_t* w_s = w_p + static_cast<size_t>(2 * inter) * (hidden / 2);

    const uint8_t* pg = w_p + static_cast<size_t>(2 * j + 0) * (hidden / 2);
    const uint8_t* sg = w_s + static_cast<size_t>(2 * j + 0) * (hidden / 16);
    const uint8_t* pu = w_p + static_cast<size_t>(2 * j + 1) * (hidden / 2);
    const uint8_t* su = w_s + static_cast<size_t>(2 * j + 1) * (hidden / 16);
    const float* xt = x + static_cast<size_t>(t) * hidden;

    float ag = 0.0f, au = 0.0f;
    for (int blk = lane; blk < hidden / 16; blk += 32) {
        const float scg = d_e4m3(sg[blk]);
        const float scu = d_e4m3(su[blk]);
        float dg = 0.0f, du = 0.0f;
        const uint8_t* bg = pg + blk * 8;
        const uint8_t* bu = pu + blk * 8;
        const float* xb = xt + blk * 16;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint8_t mg = bg[k], mu = bu[k];
            float x0 = xb[2 * k], x1 = xb[2 * k + 1];
            dg += d_e2m1_nib(mg & 0xFu) * x0 + d_e2m1_nib(mg >> 4) * x1;
            du += d_e2m1_nib(mu & 0xFu) * x0 + d_e2m1_nib(mu >> 4) * x1;
        }
        ag += dg * scg;
        au += du * scu;
    }
    ag = d_warp_sum(ag) * outer;
    au = d_warp_sum(au) * outer;
    if (lane == 0) act[static_cast<size_t>(t) * inter + j] = d_situ(ag, au);
}

__global__ void k_expert_sf4_dn(
    const uint8_t* __restrict__ body,
    uint32_t dn_off,
    const float* __restrict__ act, // [m, inter]
    float* __restrict__ y,         // [m, hidden]
    int hidden,
    int inter) {
    const int r = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (r >= hidden) return;
    const int lane = threadIdx.x & 31;

    float hdr2 = *reinterpret_cast<const float*>(body + 8);
    float hdr3 = *reinterpret_cast<const float*>(body + 12);
    const float outer = hdr2 / hdr3;
    const uint8_t* w_p = body + dn_off;
    const uint8_t* w_s = w_p + static_cast<size_t>(hidden) * (inter / 2);
    const uint8_t* pr = w_p + static_cast<size_t>(r) * (inter / 2);
    const uint8_t* sr = w_s + static_cast<size_t>(r) * (inter / 16);
    const float* at = act + static_cast<size_t>(t) * inter;

    float acc = 0.0f;
    for (int blk = lane; blk < inter / 16; blk += 32) {
        const float sc = d_e4m3(sr[blk]);
        float d = 0.0f;
        const uint8_t* bp = pr + blk * 8;
        const float* ab = at + blk * 16;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint8_t m = bp[k];
            d += d_e2m1_nib(m & 0xFu) * ab[2 * k] + d_e2m1_nib(m >> 4) * ab[2 * k + 1];
        }
        acc += d * sc;
    }
    acc = d_warp_sum(acc) * outer;
    if (lane == 0) y[static_cast<size_t>(t) * hidden + r] = acc;
}

// MX8 (MXFP8) expert kernels
__global__ void k_expert_mx8_gu_situ(
    const uint8_t* __restrict__ body,
    const float* __restrict__ x,
    float* __restrict__ act,
    int hidden,
    int inter) {
    const int j = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (j >= inter) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* w_w = body;
    const uint8_t* w_s = w_w + static_cast<size_t>(2 * inter) * hidden;
    const uint8_t* pg = w_w + static_cast<size_t>(2 * j + 0) * hidden;
    const uint8_t* sg = w_s + static_cast<size_t>(2 * j + 0) * (hidden / 32);
    const uint8_t* pu = w_w + static_cast<size_t>(2 * j + 1) * hidden;
    const uint8_t* su = w_s + static_cast<size_t>(2 * j + 1) * (hidden / 32);
    const float* xt = x + static_cast<size_t>(t) * hidden;

    float ag = 0.0f, au = 0.0f;
    for (int blk = lane; blk < hidden / 32; blk += 32) {
        const float scg = d_ue8m0(sg[blk]);
        const float scu = d_ue8m0(su[blk]);
        float dg = 0.0f, du = 0.0f;
        const uint8_t* bg = pg + blk * 32;
        const uint8_t* bu = pu + blk * 32;
        const float* xb = xt + blk * 32;
        #pragma unroll 4
        for (int k = 0; k < 32; ++k) {
            float xv = xb[k];
            dg += d_e4m3(bg[k]) * xv;
            du += d_e4m3(bu[k]) * xv;
        }
        ag += dg * scg;
        au += du * scu;
    }
    ag = d_warp_sum(ag);
    au = d_warp_sum(au);
    if (lane == 0) act[static_cast<size_t>(t) * inter + j] = d_situ(ag, au);
}

__global__ void k_expert_mx8_dn(
    const uint8_t* __restrict__ body,
    uint32_t dn_off,
    const float* __restrict__ act,
    float* __restrict__ y,
    int hidden,
    int inter) {
    const int r = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (r >= hidden) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* w_w = body + dn_off;
    const uint8_t* w_s = w_w + static_cast<size_t>(hidden) * inter;
    const uint8_t* pr = w_w + static_cast<size_t>(r) * inter;
    const uint8_t* sr = w_s + static_cast<size_t>(r) * (inter / 32);
    const float* at = act + static_cast<size_t>(t) * inter;

    float acc = 0.0f;
    for (int blk = lane; blk < inter / 32; blk += 32) {
        const float sc = d_ue8m0(sr[blk]);
        float d = 0.0f;
        const uint8_t* bp = pr + blk * 32;
        const float* ab = at + blk * 32;
        #pragma unroll 4
        for (int k = 0; k < 32; ++k) d += d_e4m3(bp[k]) * ab[k];
        acc += d * sc;
    }
    acc = d_warp_sum(acc);
    if (lane == 0) y[static_cast<size_t>(t) * hidden + r] = acc;
}

// Generic S_K single-superblock (256 weights) dot helper on GPU (supports S1.25..S8_K for SQ-15 & SQ-10!)
__device__ __forceinline__ size_t d_sk_sb_bytes(uint8_t code) {
    switch (code) {
        case 0: return 48;
        case 1: return 58;
        case 2: return 70;
        case 3: return 102;
        case 4: return 134;
        case 5: return 166;
        case 6: return 198;
        case 8: return 262;
        default: return 0;
    }
}

__device__ float d_sk_sb_dot_elem(const uint8_t* blk, uint8_t code, int idx_in_256) {
    uint16_t db = static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8);
    const float D = d_fp16_to_f32(db);
    const int g = idx_in_256 >> 6;
    const float eff = D * (static_cast<float>(blk[2 + g]) * (1.0f / 255.0f));
    const uint8_t* pl = blk + 6;
    if (code == 8) { // S8_K
        return static_cast<float>(reinterpret_cast<const int8_t*>(pl)[idx_in_256]) * eff;
    }
    if (code == 1) { // S1_5
        int b = idx_in_256 / 5;
        int rem = idx_in_256 - b * 5;
        int v = pl[b];
        for (int k = 0; k < rem; ++k) v /= 3;
        return static_cast<float>((v % 3) - 1) * eff;
    }
    if (code == 0) { // S1_25
        int grp = idx_in_256 >> 5;  // 0..7 (32 weights per 5 bytes)
        int j = (idx_in_256 >> 2) & 7;
        int lane4 = idx_in_256 & 3;
        const uint8_t* p = pl + grp * 5;
        uint64_t bits40 = static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
                          (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
                          (static_cast<uint64_t>(p[4]) << 32);
        uint32_t ci = static_cast<uint32_t>((bits40 >> (5 * j)) & 0x1Fu);
        uint8_t fb = pl[40 + (g >> 1)];
        uint32_t fn = (g & 1) ? (fb >> 4) : (fb & 0x0Fu);
        float fine = 0.5f + static_cast<float>(fn) * (1.0f / 15.0f);
        return static_cast<float>(c_s125[ci][lane4]) * eff * fine;
    }
    // S2_K..S6_K symmetric half-integer
    uint32_t q = 0;
    float half = 0.0f;
    if (code == 2) {
        q = (pl[idx_in_256 >> 2] >> ((idx_in_256 & 3) * 2)) & 3u;
        half = 1.5f;
    } else if (code == 3) {
        q = (pl[idx_in_256 >> 2] >> ((idx_in_256 & 3) * 2)) & 3u;
        q |= ((pl[64 + (idx_in_256 >> 3)] >> (idx_in_256 & 7)) & 1u) << 2;
        half = 3.5f;
    } else if (code == 4) {
        q = (pl[idx_in_256 >> 1] >> ((idx_in_256 & 1) * 4)) & 0xFu;
        half = 7.5f;
    } else if (code == 5) {
        q = (pl[idx_in_256 >> 1] >> ((idx_in_256 & 1) * 4)) & 0xFu;
        q |= ((pl[128 + (idx_in_256 >> 3)] >> (idx_in_256 & 7)) & 1u) << 4;
        half = 15.5f;
    } else if (code == 6) {
        q = (pl[idx_in_256 >> 1] >> ((idx_in_256 & 1) * 4)) & 0xFu;
        q |= ((pl[128 + (idx_in_256 >> 2)] >> ((idx_in_256 & 3) * 2)) & 3u) << 4;
        half = 31.5f;
    }
    return (static_cast<float>(q) - half) * eff;
}

__global__ void k_expert_sk_gu_situ(
    const uint8_t* __restrict__ body,
    uint8_t gu_code,
    const float* __restrict__ x,
    float* __restrict__ act,
    int hidden,
    int inter) {
    const int j = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (j >= inter) return;
    const int lane = threadIdx.x & 31;

    const size_t sb_b = d_sk_sb_bytes(gu_code);
    const size_t row_b = (hidden / 256) * sb_b;
    const uint8_t* gu_base = body + static_cast<size_t>(inter) * 2u;  // skip FP16 sq_xd_dn[inter]
    const uint8_t* rg = gu_base + static_cast<size_t>(2 * j + 0) * row_b;
    const uint8_t* ru = gu_base + static_cast<size_t>(2 * j + 1) * row_b;
    const float* xt = x + static_cast<size_t>(t) * hidden;

    float ag = 0.0f, au = 0.0f;
    for (int col = lane; col < hidden; col += 32) {
        const int s = col >> 8;
        const int idx = col & 255;
        const float xv = xt[col];
        ag += d_sk_sb_dot_elem(rg + s * sb_b, gu_code, idx) * xv;
        au += d_sk_sb_dot_elem(ru + s * sb_b, gu_code, idx) * xv;
    }
    ag = d_warp_sum(ag);
    au = d_warp_sum(au);
    if (lane == 0) act[static_cast<size_t>(t) * inter + j] = d_situ(ag, au);
}

__global__ void k_expert_sk_dn_paired2(
    const uint8_t* __restrict__ body,
    uint32_t dn_off,
    uint8_t dn_code,
    const float* __restrict__ act,
    float* __restrict__ y,
    int hidden,
    int inter) {
    // 1 warp computes 1 paired-2-row group (rows 2p, 2p+1)
    const int p = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (2 * p >= hidden) return;
    const int lane = threadIdx.x & 31;

    const size_t sb_b = d_sk_sb_bytes(dn_code);
    const size_t pair_sb = (2 * inter) / 256;
    const uint8_t* pair_ptr = body + dn_off + static_cast<size_t>(p) * pair_sb * sb_b;
    const uint16_t* sq_xd = reinterpret_cast<const uint16_t*>(body);
    const float* at = act + static_cast<size_t>(t) * inter;

    float a0 = 0.0f, a1 = 0.0f;
    for (int col = lane; col < inter; col += 32) {
        const float av = at[col] / d_fp16_to_f32(sq_xd[col]);
        // Row 0 sits at [0..inter-1] inside the paired 2*inter stream
        int i0 = col;
        a0 += d_sk_sb_dot_elem(pair_ptr + (i0 >> 8) * sb_b, dn_code, i0 & 255) * av;
        // Row 1 sits at [inter..2*inter-1]
        int i1 = inter + col;
        a1 += d_sk_sb_dot_elem(pair_ptr + (i1 >> 8) * sb_b, dn_code, i1 & 255) * av;
    }
    a0 = d_warp_sum(a0);
    a1 = d_warp_sum(a1);
    if (lane == 0) {
        y[static_cast<size_t>(t) * hidden + 2 * p + 0] = a0;
        if (2 * p + 1 < hidden) y[static_cast<size_t>(t) * hidden + 2 * p + 1] = a1;
    }
}

// Native GGUF_RAW Q8_0 / Q4_K / Q6_K / IQ4_NL / IQ4_XS / MXFP4 / BF16 / F16 GPU dot row helper
__device__ float d_gguf_row_dot(uint32_t ggml_type, const uint8_t* row, const float* x, int K, int lane) {
    float acc = 0.0f;
    if (ggml_type == 8) { // Q8_0: 32 elems = 2B fp16 d + 32B int8 qs (34 B)
        for (int b = lane; b < K / 32; b += 32) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 34;
            uint16_t db = static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8);
            const float d = d_fp16_to_f32(db);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            const float* xb = x + b * 32;
            float s = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < 32; ++i) s += static_cast<float>(qs[i]) * xb[i];
            acc += s * d;
        }
    } else if (ggml_type == 12) { // Q4_K: 256 elems = 144 B (d fp16, dmin fp16, scales[12], qs[128])
        for (int b = 0; b < K / 256; ++b) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 144;
            const float d = d_fp16_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const float dmin = d_fp16_to_f32(static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8));
            const uint8_t* sc = blk + 4;
            const uint8_t* qs = blk + 16;
            const float* xb = x + b * 256;
            for (int idx = lane; idx < 256; idx += 32) {
                int j64 = idx >> 6;          // 0..3 (which 64-elem chunk)
                int in64 = idx & 63;
                int sub = j64 * 2 + (in64 >> 5); // 0..7
                uint8_t sd = 0, sm = 0;
                if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
                else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
                uint8_t byte = qs[j64 * 32 + (in64 & 31)];
                uint8_t q = (in64 < 32) ? (byte & 0xFu) : (byte >> 4);
                float w = d * static_cast<float>(sd) * static_cast<float>(q) - dmin * static_cast<float>(sm);
                acc += w * xb[idx];
            }
        }
    } else if (ggml_type == 14) { // Q6_K: 256 elems = 210 B (ql[128], qh[64], scales[16], d fp16)
        for (int b = 0; b < K / 256; ++b) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            const float d = d_fp16_to_f32(static_cast<uint16_t>(blk[208]) | (static_cast<uint16_t>(blk[209]) << 8));
            const float* xb = x + b * 256;
            for (int idx = lane; idx < 256; idx += 32) {
                int half = idx >> 7;        // 0 or 1 (128-elem half)
                int r = idx & 127;
                int qgrp = r >> 5;          // 0..3
                int l = r & 31;
                const uint8_t* qlh = ql + half * 64;
                const uint8_t* qhh = qh + half * 32;
                const int8_t* sch = sc + half * 8;
                int is = l >> 4;
                int q = 0;
                int sc_i = 0;
                if (qgrp == 0)      { q = static_cast<int>((qlh[l] & 0xFu) | (((qhh[l] >> 0) & 3u) << 4)) - 32; sc_i = is + 0; }
                else if (qgrp == 1) { q = static_cast<int>((qlh[l + 32] & 0xFu) | (((qhh[l] >> 2) & 3u) << 4)) - 32; sc_i = is + 2; }
                else if (qgrp == 2) { q = static_cast<int>((qlh[l] >> 4) | (((qhh[l] >> 4) & 3u) << 4)) - 32; sc_i = is + 4; }
                else                { q = static_cast<int>((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3u) << 4)) - 32; sc_i = is + 6; }
                acc += d * static_cast<float>(sch[sc_i]) * static_cast<float>(q) * xb[idx];
            }
        }
    } else if (ggml_type == 20) { // IQ4_NL: 32 elems = 18 B (d fp16, qs[16])
        for (int b = lane; b < K / 32; b += 32) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 18;
            const float d = d_fp16_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const float* xb = x + b * 32;
            float s = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < 16; ++j) {
                s += static_cast<float>(c_iq4nl[qs[j] & 0xFu]) * xb[j] +
                     static_cast<float>(c_iq4nl[qs[j] >> 4]) * xb[j + 16];
            }
            acc += s * d;
        }
    } else if (ggml_type == 23) { // IQ4_XS: 256 elems = 136 B (d fp16, scales_h u16, scales_l[4], qs[128])
        for (int b = 0; b < K / 256; ++b) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 136;
            const float d = d_fp16_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint16_t sh = static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8);
            const uint8_t* sl = blk + 4;
            const uint8_t* qs = blk + 8;
            const float* xb = x + b * 256;
            for (int idx = lane; idx < 256; idx += 32) {
                int ib = idx >> 5;
                int in32 = idx & 31;
                int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xFu) | (((sh >> (2 * ib)) & 3u) << 4);
                float dl = d * static_cast<float>(ls - 32);
                uint8_t byte = qs[ib * 16 + (in32 & 15)];
                uint8_t q = (in32 < 16) ? (byte & 0xFu) : (byte >> 4);
                acc += dl * static_cast<float>(c_iq4nl[q]) * xb[idx];
            }
        }
    } else if (ggml_type == 39) { // MXFP4: 32 elems = 17 B (e u8, qs[16])
        for (int b = lane; b < K / 32; b += 32) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 17;
            uint8_t e = blk[0];
            float d = (e < 2) ? __uint_as_float(0x00200000u << e) : __uint_as_float(static_cast<uint32_t>(e - 1) << 23);
            const uint8_t* qs = blk + 1;
            const float* xb = x + b * 32;
            float s = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < 16; ++j) {
                s += (2.0f * d_e2m1_nib(qs[j] & 0xFu)) * xb[j] + (2.0f * d_e2m1_nib(qs[j] >> 4)) * xb[j + 16];
            }
            acc += s * d;
        }
    } else if (ggml_type == 30) { // BF16
        const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
        for (int i = lane; i < K; i += 32) acc += d_bf16_to_f32(w[i]) * x[i];
    } else if (ggml_type == 1) { // F16
        const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
        for (int i = lane; i < K; i += 32) acc += d_fp16_to_f32(w[i]) * x[i];
    } else if (ggml_type == 0) { // F32
        const float* w = reinterpret_cast<const float*>(row);
        for (int i = lane; i < K; i += 32) acc += w[i] * x[i];
    }
    return d_warp_sum(acc);
}

__device__ __forceinline__ size_t d_gguf_row_bytes(uint32_t t, int K) {
    switch (t) {
        case 0:  return static_cast<size_t>(K) * 4u;
        case 1:
        case 30: return static_cast<size_t>(K) * 2u;
        case 8:  return static_cast<size_t>(K / 32) * 34u;
        case 12: return static_cast<size_t>(K / 256) * 144u;
        case 14: return static_cast<size_t>(K / 256) * 210u;
        case 20: return static_cast<size_t>(K / 32) * 18u;
        case 23: return static_cast<size_t>(K / 256) * 136u;
        case 39: return static_cast<size_t>(K / 32) * 17u;
        default: return 0u;
    }
}

__global__ void k_expert_gguf_gu_situ(
    const uint8_t* __restrict__ body,
    const float* __restrict__ x,
    float* __restrict__ act,
    int hidden,
    int inter) {
    const int j = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (j >= inter) return;
    const int lane = threadIdx.x & 31;

    const GgufExpertHeader* hdr = reinterpret_cast<const GgufExpertHeader*>(body);
    const size_t rb = d_gguf_row_bytes(hdr->gu_type, hidden);
    const uint8_t* base = body + sizeof(GgufExpertHeader);
    const bool interleaved = (hdr->flags & 1u) != 0;
    const uint8_t* rg = base + static_cast<size_t>(interleaved ? (2 * j) : j) * rb;
    const uint8_t* ru = base + static_cast<size_t>(interleaved ? (2 * j + 1) : (inter + j)) * rb;
    const float* xt = x + static_cast<size_t>(t) * hidden;

    float g = d_gguf_row_dot(hdr->gu_type, rg, xt, hidden, lane) * hdr->gate_scale;
    float u = d_gguf_row_dot(hdr->gu_type, ru, xt, hidden, lane) * hdr->up_scale;
    if (lane == 0) act[static_cast<size_t>(t) * inter + j] = d_situ(g, u);
}

__global__ void k_expert_gguf_dn(
    const uint8_t* __restrict__ body,
    uint32_t dn_off,
    const float* __restrict__ act,
    float* __restrict__ y,
    int hidden,
    int inter) {
    const int r = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (r >= hidden) return;
    const int lane = threadIdx.x & 31;

    const GgufExpertHeader* hdr = reinterpret_cast<const GgufExpertHeader*>(body);
    const size_t rb = d_gguf_row_bytes(hdr->dn_type, inter);
    const uint8_t* rr = body + dn_off + static_cast<size_t>(r) * rb;
    const float* at = act + static_cast<size_t>(t) * inter;

    float v = d_gguf_row_dot(hdr->dn_type, rr, at, inter, lane) * hdr->dn_scale;
    if (lane == 0) y[static_cast<size_t>(t) * hidden + r] = v;
}

// Check if a GGUF type has a native CUDA device kernel above (otherwise falls back cleanly to CPU/host dequant)
inline bool gguf_type_has_gpu_kernel(uint32_t t) noexcept {
    switch (t) {
        case 0: case 1: case 8: case 12: case 14: case 20: case 23: case 30: case 39:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Dense BF16 GEMV/GEMM & Unembed Kernels on GPU 0
// ============================================================================
__global__ void k_bf16_matmul(
    const float* __restrict__ x,      // [M, K]
    const uint16_t* __restrict__ w,   // [N, K] BF16
    float* __restrict__ y,            // [M, N]
    int M, int N, int K) {
    const int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const int lane = threadIdx.x & 31;
    const float* xm = x + static_cast<size_t>(m) * K;
    const uint16_t* wn = w + static_cast<size_t>(n) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc += xm[k] * d_bf16_to_f32(wn[k]);
    acc = d_warp_sum(acc);
    if (lane == 0) y[static_cast<size_t>(m) * N + n] = acc;
}

__global__ void k_unembed_softcap_bf16(
    const float* __restrict__ h,
    const uint16_t* __restrict__ w,
    float* __restrict__ out,
    int M, int V, int D, float cap) {
    const int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int m = blockIdx.y;
    if (v >= V || m >= M) return;
    const int lane = threadIdx.x & 31;
    const float* hm = h + static_cast<size_t>(m) * D;
    const uint16_t* wv = w + static_cast<size_t>(v) * D;
    float acc = 0.0f;
    for (int d = lane; d < D; d += 32) acc += hm[d] * d_bf16_to_f32(wv[d]);
    acc = d_warp_sum(acc);
    if (lane == 0) out[static_cast<size_t>(m) * V + v] = cap * tanhf(acc / cap);
}

// ============================================================================
// Dual-GPU Expert Parallelism + Hybrid Doorbell CUDA Backend Implementation
// ============================================================================
struct EpSlot {
    bool allocated = false;
    int max_ctx = 0;
    SlotKvPages kv;
    SlotShortConvState sconv;
    std::vector<SlotShortConvState> sconv_snaps;
    int snap_seq_len = -1;
    bool recording_snaps = false;
    std::vector<std::vector<uint16_t>> dspark_k;
    std::vector<std::vector<uint16_t>> dspark_v;
    std::vector<uint16_t> dspark_frnn;
    int dspark_len = 0;
};

class EpDualGpuCudaBackend final : public InkBackend {
public:
    ~EpDualGpuCudaBackend() override { cleanup(); }

    const char* backend_name() const noexcept override {
        if (n_gpus_ >= 2) return "cuda-ep2";
        if (doorbell_active_) return "cuda-hybrid-doorbell";
        return "cuda-ep1";
    }

    std::string device_info() const override {
        std::string info;
        for (int g = 0; g < n_gpus_; ++g) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, gpu_ids_[g]) == cudaSuccess) {
                if (!info.empty()) info += " + ";
                info += std::string(prop.name) + " (" + std::to_string(prop.totalGlobalMem >> 20) + " MiB, sm_" +
                        std::to_string(prop.major * 10 + prop.minor) + ")";
            }
        }
        if (n_gpus_ >= 2) {
            info += " [EP-2GPU split=" + std::to_string(ep_split_) + ", p2p=" + (p2p_ok_ ? "on" : "pinned") +
                    ", vram_exp=" + std::to_string(vram_experts_total_) + "/" + std::to_string(total_experts_) + "]";
        } else if (doorbell_active_) {
            info += " [Graph-Doorbell NUMA=" + std::to_string(db_.n_nodes) +
                    ", vram_exp=" + std::to_string(vram_experts_total_) + "/" + std::to_string(total_experts_) + "]";
        }
        return info;
    }

    BackendStats stats() const override {
        BackendStats st;
        st.vram_experts_loaded = vram_experts_total_;
        st.ram_experts_loaded = host_experts_total_;
        st.total_experts = total_experts_;
        st.vram_used_bytes = vram_used_bytes_;
        st.vram_total_bytes = vram_total_bytes_;
        st.max_slots = static_cast<int>(slots_.size());
        for (const auto& s : slots_) if (s.allocated) ++st.active_slots;
        st.kv_dtype = default_kv_dtype_;
        return st;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        cleanup();
        model_ = &model;
        opts_ = opts;
        c_ = model.m.c;
        dspark_cfg_ = model.m.dspark;
        default_kv_dtype_ = opts.kv_dtype;

        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count < 1) {
            err = "cuda-ep: no CUDA devices visible";
            return false;
        }
        n_gpus_ = (opts.gpus >= 2 && dev_count >= 2) ? 2 : 1;
        gpu_ids_[0] = std::max(0, std::min(opts.gpu_index, dev_count - 1));
        gpu_ids_[1] = (n_gpus_ >= 2) ? ((gpu_ids_[0] == 0) ? 1 : 0) : gpu_ids_[0];
        ep_split_ = std::clamp(opts.ep_split, 0.1, 0.9);

        // Setup streams & P2P across GPUs
        for (int g = 0; g < n_gpus_; ++g) {
            EP_CK(cudaSetDevice(gpu_ids_[g]));
            EP_CK(cudaStreamCreate(&streams_[g]));
        }
        p2p_ok_ = false;
        if (n_gpus_ >= 2) {
            int p01 = 0, p10 = 0;
            cudaDeviceCanAccessPeer(&p01, gpu_ids_[0], gpu_ids_[1]);
            cudaDeviceCanAccessPeer(&p10, gpu_ids_[1], gpu_ids_[0]);
            if (p01 && p10) {
                cudaSetDevice(gpu_ids_[0]);
                cudaError_t e1 = cudaDeviceEnablePeerAccess(gpu_ids_[1], 0);
                if (e1 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                cudaSetDevice(gpu_ids_[1]);
                cudaError_t e2 = cudaDeviceEnablePeerAccess(gpu_ids_[0], 0);
                if (e2 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                p2p_ok_ = ((e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled) &&
                           (e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled));
            }
            // Hardware 16-byte roundtrip P2P probe: some AMD Threadripper/EPYC/Xeon
            // PCIe root complexes with IOMMU/ACS report CanAccessPeer=1 and EnablePeerAccess=ok
            // yet silently transfer all zeroes on BAR-to-BAR cudaMemcpyPeer.
            if (p2p_ok_) {
                uint32_t h_tx[4] = {0x12345678u, 0xCAFEBABEu, 0xA5A55A5Au, 0xDEADBEEFu};
                uint32_t h_rx01[4] = {0, 0, 0, 0};
                uint32_t h_rx10[4] = {0, 0, 0, 0};
                uint32_t *d_p0 = nullptr, *d_p1 = nullptr;
                cudaSetDevice(gpu_ids_[0]);
                cudaError_t pa0 = cudaMalloc(&d_p0, sizeof(h_tx));
                cudaSetDevice(gpu_ids_[1]);
                cudaError_t pa1 = cudaMalloc(&d_p1, sizeof(h_tx));
                bool hw_ok = false;
                if (pa0 == cudaSuccess && pa1 == cudaSuccess) {
                    cudaSetDevice(gpu_ids_[0]);
                    cudaMemcpy(d_p0, h_tx, sizeof(h_tx), cudaMemcpyHostToDevice);
                    cudaSetDevice(gpu_ids_[1]);
                    cudaMemset(d_p1, 0, sizeof(h_tx));
                    cudaError_t ep01 = cudaMemcpyPeer(d_p1, gpu_ids_[1], d_p0, gpu_ids_[0], sizeof(h_tx));
                    cudaMemcpy(h_rx01, d_p1, sizeof(h_tx), cudaMemcpyDeviceToHost);
                    cudaSetDevice(gpu_ids_[0]);
                    cudaMemset(d_p0, 0, sizeof(h_tx));
                    cudaError_t ep10 = cudaMemcpyPeer(d_p0, gpu_ids_[0], d_p1, gpu_ids_[1], sizeof(h_tx));
                    cudaMemcpy(h_rx10, d_p0, sizeof(h_tx), cudaMemcpyDeviceToHost);
                    hw_ok = (ep01 == cudaSuccess && ep10 == cudaSuccess &&
                             std::memcmp(h_tx, h_rx01, sizeof(h_tx)) == 0 &&
                             std::memcmp(h_tx, h_rx10, sizeof(h_tx)) == 0);
                }
                if (d_p0) { cudaSetDevice(gpu_ids_[0]); cudaFree(d_p0); }
                if (d_p1) { cudaSetDevice(gpu_ids_[1]); cudaFree(d_p1); }
                if (!hw_ok) {
                    std::fprintf(stderr,
                                 "[inkling2-ep] Hardware P2P roundtrip probe FAILED "
                                 "(rx01=0x%08x rx10=0x%08x, expected 0x%08x); "
                                 "disabling peer access and using pinned-host async DMA fallback\n",
                                 h_rx01[0], h_rx10[0], h_tx[0]);
                    cudaSetDevice(gpu_ids_[0]);
                    if (cudaDeviceDisablePeerAccess(gpu_ids_[1]) != cudaSuccess) cudaGetLastError();
                    cudaSetDevice(gpu_ids_[1]);
                    if (cudaDeviceDisablePeerAccess(gpu_ids_[0]) != cudaSuccess) cudaGetLastError();
                    p2p_ok_ = false;
                }
            }
        }
        EP_CK(cudaSetDevice(gpu_ids_[0]));

        // 1. Load Section 1 trunk + DSpark
        std::vector<uint8_t> sec1;
        if (!model.read_section1(sec1, err)) return false;

        const std::string embed_k = "model.llm.embed.weight";
        const std::string unemb_k = "model.llm.unembed.weight";
        for (const auto& kv : model.m.trunk) {
            const auto& ti = kv.second;
            if (kv.first == embed_k || kv.first == unemb_k) {
                packed_trunk_[kv.first] = {ti, std::vector<uint8_t>(sec1.begin() + ti.off, sec1.begin() + ti.off + ti.bytes)};
            } else {
                trunk_f32_[kv.first] = decode_section1_f32(sec1.data(), ti);
            }
        }

        // Upload BF16 dense matrices & unembed to GPU 0 for fast GPU projection
        if (auto it = packed_trunk_.find(unemb_k); it != packed_trunk_.end() && it->second.info.fmt == F_BF16) {
            const size_t nb = it->second.bytes.size();
            EP_CK(cudaMalloc(&d_unembed_bf16_, nb));
            EP_CK(cudaMemcpy(d_unembed_bf16_, it->second.bytes.data(), nb, cudaMemcpyHostToDevice));
            vram_used_bytes_ += nb;
        }
        // Upload layer dense matrices (wq, wk, wv, wr, gate_attn, sink, wo, router) as BF16 on GPU 0
        for (int il = 0; il < c_.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            auto up_bf16 = [&](const std::string& name) -> bool {
                auto it = trunk_f32_.find(name);
                if (it == trunk_f32_.end()) return true;
                std::vector<uint16_t> b16(it->second.size());
                for (size_t i = 0; i < b16.size(); ++i) b16[i] = f32_to_bf16(it->second[i]);
                uint16_t* dp = nullptr;
                EP_CK(cudaMalloc(&dp, b16.size() * 2));
                EP_CK(cudaMemcpy(dp, b16.data(), b16.size() * 2, cudaMemcpyHostToDevice));
                d_mats_bf16_[name] = dp;
                vram_used_bytes_ += b16.size() * 2;
                return true;
            };
            if (!up_bf16(p + ".attn.wq_du.weight") || !up_bf16(p + ".attn.wk_dv.weight") ||
                !up_bf16(p + ".attn.wv_dv.weight") || !up_bf16(p + ".attn.wr_du.weight") ||
                !up_bf16(p + ".attn.attn_sink_proj.weight") || !up_bf16(p + ".attn.gate_attn.weight") ||
                !up_bf16(p + ".attn.wo_ud.weight") || !up_bf16(p + ".mlp.gate.weight")) {
                err = "cuda-ep: VRAM alloc failed for dense trunk";
                return false;
            }
        }

        // Precompute msg_queries & static ShortConv
        if (trunk_f32_.count("model.llm.middle_cycle_cross_pass_queries") &&
            trunk_f32_.count("model.llm.middle_cycle_cross_pass_kv_queries")) {
            const auto& q_cp = trunk_f32_.at("model.llm.middle_cycle_cross_pass_queries");
            const auto& kv_cp = trunk_f32_.at("model.llm.middle_cycle_cross_pass_kv_queries");
            const int core_len = c_.core_end - c_.core_start;
            msg_queries_.resize(c_.n_phys);
            for (int i = 0; i < core_len; ++i) {
                auto& mq = msg_queries_[c_.core_start + i];
                mq.resize(4u * c_.hidden);
                std::memcpy(mq.data() + 0 * c_.hidden, q_cp.data()  + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 1 * c_.hidden, kv_cp.data() + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 2 * c_.hidden, kv_cp.data() + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 3 * c_.hidden, q_cp.data()  + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
            }
        }
        for (int il = 0; il < c_.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            static_sconv_[p + ".attn_sconv"] = trunk_f32_.at(p + ".attn_sconv.weight");
            static_sconv_[p + ".mlp_sconv"]  = trunk_f32_.at(p + ".mlp_sconv.weight");
            if (c_.is_swa[il]) {
                static_sconv_[p + ".k_sconv"] = trunk_f32_.at(p + ".attn.k_sconv.weight");
                static_sconv_[p + ".v_sconv"] = trunk_f32_.at(p + ".attn.v_sconv.weight");
            } else {
                const int kv_dim = c_.n_kv * c_.head_dim;
                const auto& kw = trunk_f32_.at(p + ".attn.k_sconv.static_w");
                const auto& vw = trunk_f32_.at(p + ".attn.v_sconv.static_w");
                std::vector<float> kt(kv_dim * 4), vt(kv_dim * 4);
                for (int c = 0; c < kv_dim; ++c) {
                    for (int lag = 0; lag < 4; ++lag) {
                        kt[c * 4 + (3 - lag)] = kw[lag * kv_dim + c];
                        vt[c * 4 + (3 - lag)] = vw[lag * kv_dim + c];
                    }
                }
                static_sconv_[p + ".k_sconv"] = std::move(kt);
                static_sconv_[p + ".v_sconv"] = std::move(vt);
            }
        }

        // Optional DSpark tensors
        has_dspark_ = opts.enable_dspark && dspark_cfg_.present;
        if (has_dspark_) {
            for (const auto& kv : model.m.dspark_tensors) {
                std::string k = kv.first;
                if (k.rfind("dspark.", 0) == 0) k = k.substr(7);
                dspark_f32_[k] = decode_section1_f32(sec1.data(), kv.second);
            }
        }

        // 2. Expert Placement: Dual-GPU EP (--gpus 2) OR Single-GPU Resident / Hybrid Doorbell Offload!
        const auto& dir = model.expert_directory();
        total_experts_ = dir.size();
        const int n_total = static_cast<int>(dir.size());
        experts_.assign(n_total, DevExpert{});
        host_descs_.assign(n_total, hybrid::HostExpertDesc{});

        // Query free VRAM on each GPU
        size_t free_b[2] = {0, 0}, tot_b[2] = {0, 0};
        vram_total_bytes_ = 0;
        for (int g = 0; g < n_gpus_; ++g) {
            EP_CK(cudaSetDevice(gpu_ids_[g]));
            EP_CK(cudaMemGetInfo(&free_b[g], &tot_b[g]));
            vram_total_bytes_ += tot_b[g];
        }
        const size_t reserve0 = (opts.vram_reserve_mib >= 0)
            ? (static_cast<size_t>(opts.vram_reserve_mib) << 20) + (512ull << 20)
            : (1536ull << 20);
        const size_t reserve1 = (768ull << 20);
        size_t budget[2] = {
            (free_b[0] > reserve0) ? (free_b[0] - reserve0) : 0,
            (n_gpus_ >= 2 && free_b[1] > reserve1) ? (free_b[1] - reserve1) : 0
        };
        if (opts.vram_experts_gb >= 0.0) {
            size_t cap = static_cast<size_t>(opts.vram_experts_gb * 1073741824.0);
            if (n_gpus_ == 1) budget[0] = std::min(budget[0], cap);
        }

        // Compute deterministic EP ownership per expert (balanced by ep_split_ and VRAM capacity)
        std::vector<uint8_t> target_gpu(n_total, 0);
        std::vector<uint8_t> in_vram(n_total, 0);
        size_t need_bytes[2] = {0, 0};
        size_t max_eb = 0;
        vram_experts_total_ = 0;
        host_experts_total_ = 0;

        for (uint16_t di : model.vram_load_order) {
            if (di >= dir.size()) continue;
            const auto& e = dir[di];
            const size_t eb_aligned = (static_cast<size_t>(e.body_bytes) + 255u) & ~size_t(255u);
            max_eb = std::max(max_eb, eb_aligned);

            int pref = 0;
            if (n_gpus_ >= 2) {
                // Hash-balanced split across GPU 0 & GPU 1 (#6506 IK2_EP_SPLIT), falling back to whichever GPU has room!
                const int il = di / c_.n_experts;
                const int eid = e.expert_id;
                uint32_t h = (static_cast<uint32_t>(eid) * 2654435761u + static_cast<uint32_t>(il) * 40503u) >> 8;
                pref = ((h % 10000u) < static_cast<uint32_t>(ep_split_ * 10000.0)) ? 1 : 0;
                if (need_bytes[pref] + eb_aligned > budget[pref] && need_bytes[1 - pref] + eb_aligned <= budget[1 - pref]) {
                    pref = 1 - pref;
                }
            }
            target_gpu[di] = static_cast<uint8_t>(pref);
            if (need_bytes[pref] + eb_aligned <= budget[pref]) {
                in_vram[di] = 1;
                need_bytes[pref] += eb_aligned;
                ++vram_experts_total_;
            } else {
                ++host_experts_total_;
            }
        }

        // Allocate contiguous VRAM expert pool on each GPU
        for (int g = 0; g < n_gpus_; ++g) {
            if (need_bytes[g] > 0) {
                EP_CK(cudaSetDevice(gpu_ids_[g]));
                EP_CK(cudaMalloc(&d_expert_pool_[g], need_bytes[g]));
                vram_used_bytes_ += need_bytes[g];
            }
        }

        // Allocate host storage for non-resident experts (NUMA-placed if doorbell active, else regular vector)
        doorbell_active_ = (n_gpus_ == 1 && host_experts_total_ > 0 && opts.cpu_threads > 0);
        const int n_numa = doorbell_active_ ? hybrid::numa_node_count() : 1;
        host_bodies_.resize(n_total);

        size_t cur_off[2] = {0, 0};
        std::vector<uint8_t> tmp;
        for (int di = 0; di < n_total; ++di) {
            const auto& e = dir[di];
            const int il = di / c_.n_experts;
            const int key = il * c_.n_experts + e.expert_id;
            if (!model.read_expert_body(di, tmp, err)) return false;

            const size_t eb_aligned = (tmp.size() + 255u) & ~size_t(255u);
            bool gpu_supported = true;
            if (e.gu_code_id == static_cast<uint8_t>(SkCode::GGUF_RAW)) {
                GgufExpertHeader gh{};
                std::memcpy(&gh, tmp.data(), sizeof(gh));
                gpu_supported = gguf_type_has_gpu_kernel(gh.gu_type) && gguf_type_has_gpu_kernel(gh.dn_type);
            }

            auto& de = experts_[key];
            de.dn_off = e.dn_offset;
            de.gu_code = e.gu_code_id;
            de.dn_code = e.dn_code_id;
            de.owner_gpu = target_gpu[di];

            if (in_vram[di] && gpu_supported) {
                const int g = target_gpu[di];
                EP_CK(cudaSetDevice(gpu_ids_[g]));
                EP_CK(cudaMemcpy(d_expert_pool_[g] + cur_off[g], tmp.data(), tmp.size(), cudaMemcpyHostToDevice));
                de.body = d_expert_pool_[g] + cur_off[g];
                de.resident = 1;
                cur_off[g] += eb_aligned;
            } else {
                if (in_vram[di] && !gpu_supported) {
                    --vram_experts_total_;
                    ++host_experts_total_;
                }
                de.resident = 0;
                de.body = nullptr;
                host_bodies_[key] = tmp;
                auto& hd = host_descs_[key];
                hd.body = host_bodies_[key].data();
                hd.body_bytes = static_cast<uint32_t>(host_bodies_[key].size());
                hd.dn_offset = e.dn_offset;
                hd.gu_code = e.gu_code_id;
                hd.dn_code = e.dn_code_id;
                hd.numa_node = static_cast<uint8_t>(key % n_numa);
            }
        }

        // Allocate per-GPU MoE scratch buffers (supports up to 64 batched tokens!)
        const size_t max_tok = kMaxParallelSlots;
        for (int g = 0; g < n_gpus_; ++g) {
            EP_CK(cudaSetDevice(gpu_ids_[g]));
            EP_CK(cudaMalloc(&d_x_buf_[g], max_tok * c_.hidden * sizeof(float)));
            EP_CK(cudaMalloc(&d_act_buf_[g], max_tok * c_.inter * sizeof(float)));
            EP_CK(cudaMalloc(&d_y_buf_[g], max_tok * c_.hidden * sizeof(float)));
        }
        EP_CK(cudaSetDevice(gpu_ids_[0]));
        const size_t max_dense_k = static_cast<size_t>(std::max(c_.hidden, c_.n_head * c_.head_dim));
        const size_t max_dense_n = static_cast<size_t>(std::max(std::max(c_.vocab_unpadded, c_.n_head * c_.head_dim),
                                                                std::max(c_.hidden, c_.n_experts)));
        EP_CK(cudaMalloc(&d_dense_x_, max_tok * max_dense_k * sizeof(float)));
        EP_CK(cudaMalloc(&d_dense_y_, max_tok * max_dense_n * sizeof(float)));

        // Start NUMA Doorbell CPU pool if hybrid doorbell offload is active
        if (doorbell_active_) {
            db_.n_layers = c_.n_phys;
            db_.n_experts = c_.n_experts;
            db_.n_nodes = n_numa;
            const size_t bell_bytes = (sizeof(hybrid::InkDoorbell) + 4095u) & ~size_t(4095u);
            for (int nd = 0; nd < n_numa; ++nd) {
                EP_CK(cudaHostAlloc(&db_.h_bells[nd], bell_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
                std::memset(db_.h_bells[nd], 0, sizeof(hybrid::InkDoorbell));
                EP_CK(cudaHostGetDevicePointer(&db_.d_bells[nd], db_.h_bells[nd], 0));
            }
            hybrid::DoorbellPoolConfig pcfg{};
            pcfg.n_layers = c_.n_phys;
            pcfg.n_experts = c_.n_experts;
            pcfg.hidden = c_.hidden;
            pcfg.inter = c_.inter;
            pcfg.gate_cap = c_.situ_beta;
            pcfg.up_cap = c_.situ_lb;
            pcfg.experts = host_descs_.data();
            for (int nd = 0; nd < n_numa; ++nd) pcfg.bells[nd] = db_.h_bells[nd];
            pcfg.threads_per_node = opts.cpu_threads;
            hybrid::start_doorbell_pool(pcfg);
        }

        const int n_slots = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        slots_.resize(n_slots);

        std::fprintf(stderr,
            "[inkling2-ep] Loaded %s (%s) | gpus=%d (p2p=%s, ep_split=%.2f) | experts: %zu VRAM (GPU0=%.2f GiB, GPU1=%.2f GiB), %zu host (doorbell=%s, numa=%d, thr=%d) | kv=%s\n",
            source_kind_name(model.kind), model.m.build.c_str(), n_gpus_, p2p_ok_ ? "on" : "pinned", ep_split_,
            vram_experts_total_, cur_off[0] / 1073741824.0, cur_off[1] / 1073741824.0, host_experts_total_,
            doorbell_active_ ? "on" : "off", n_numa, opts.cpu_threads, kv_dtype_name(default_kv_dtype_));
        return true;
    }

    bool alloc_slot(int slot, int max_ctx, KvDtype kv_dtype, std::string& err) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) {
            err = "slot index out of range";
            return false;
        }
        auto& sl = slots_[slot];
        sl.allocated = true;
        sl.max_ctx = max_ctx;
        sl.kv.init(c_.n_swa_visits, c_.n_global_visits, c_.n_kv * c_.head_dim, c_.swa_win, kv_dtype);
        sl.sconv.init(c_.n_visits, c_.n_kv * c_.head_dim, c_.hidden);
        sl.sconv_snaps.clear();
        sl.snap_seq_len = -1;
        sl.recording_snaps = false;
        sl.dspark_k.assign(dspark_cfg_.n_layers, {});
        sl.dspark_v.assign(dspark_cfg_.n_layers, {});
        sl.dspark_frnn.clear();
        sl.dspark_len = 0;
        return true;
    }

    void free_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) return;
        slots_[slot] = EpSlot{};
    }

    void reset_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        slots_[slot].kv.reset();
        slots_[slot].sconv.reset();
        slots_[slot].recording_snaps = false;
        slots_[slot].snap_seq_len = -1;
        slots_[slot].dspark_len = 0;
    }

    void snapshot_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        if (sl.sconv_snaps.empty()) {
            sl.sconv_snaps.resize(9);
            for (auto& sn : sl.sconv_snaps) sn.init(c_.n_visits, c_.n_kv * c_.head_dim, c_.hidden);
        }
        sl.sconv_snaps[0] = sl.sconv;
        sl.snap_seq_len = sl.kv.seq_len;
        sl.recording_snaps = true;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        sl.kv.seq_len = keep_len;
        sl.dspark_len = std::min(sl.dspark_len, keep_len);
        if (sl.snap_seq_len >= 0 && keep_len >= sl.snap_seq_len) {
            int step = keep_len - sl.snap_seq_len;
            if (step >= 0 && step < static_cast<int>(sl.sconv_snaps.size())) sl.sconv = sl.sconv_snaps[step];
        }
        sl.recording_snaps = false;
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        const int B = batch.n_tokens;
        if (B <= 0) return true;
        const int D = c_.hidden;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const int q_dim = c_.n_head * c_.head_dim;

        for (int b = 0; b < B; ++b) {
            int s = batch.slot_ids[b];
            if (s < 0 || s >= static_cast<int>(slots_.size()) || !slots_[s].allocated) {
                err = "unallocated slot " + std::to_string(s);
                return false;
            }
        }

        // 1. Embeddings
        std::vector<float> cur(static_cast<size_t>(B) * D);
        const float* emb_nw = trunk_f32_.at("model.llm.embed_norm.weight").data();
        const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
        std::vector<float> raw_row(D);
        for (int b = 0; b < B; ++b) {
            if (batch.img_embed_idx[b] >= 0 && batch.img_embeds != nullptr) {
                std::memcpy(cur.data() + static_cast<size_t>(b) * D,
                            batch.img_embeds + static_cast<size_t>(batch.img_embed_idx[b]) * D, D * sizeof(float));
            } else {
                decode_packed_row(emb_pk, batch.token_ids[b], raw_row.data());
                rms_norm_f32(raw_row.data(), emb_nw, cur.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
            }
        }

        std::vector<float> tau(B);
        for (int b = 0; b < B; ++b) tau[b] = log_pos_tau(batch.pos[b], c_.log_scale_floor, c_.log_scale_alpha);

        std::vector<std::vector<float>> blocks, prelude_blocks, bank;
        std::vector<float> prelude, bank_out;

        const bool need_taps = (taps_out != nullptr) || (has_dspark_ && batch.capture_taps);
        std::vector<float> taps[4];
        if (need_taps) for (int k = 0; k < 4; ++k) taps[k].resize(static_cast<size_t>(B) * D);
        auto maybe_tap = [&](int visit_id, const std::vector<float>& vec) {
            if (!need_taps) return;
            for (int k = 0; k < 4 && k < static_cast<int>(dspark_cfg_.target_visits.size()); ++k) {
                if (dspark_cfg_.target_visits[k] == visit_id) {
                    std::memcpy(taps[k].data(), vec.data(), static_cast<size_t>(B) * D * sizeof(float));
                }
            }
        };

        int visit = 0;
        auto run_visit = [&](int ip, int /*pass_idx*/, const std::vector<float>* msgs) -> bool {
            maybe_tap(visit, cur);
            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = c_.is_swa[ip] != 0;
            const int kv_slot = c_.visit_kv_slot[visit];

            std::vector<float> residual = cur;
            std::vector<float> in0(B * D), in1(B * D), in2(B * D);

            if (ip > 0) {
                const auto& q_res = trunk_f32_.at("model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
                const int nq = is_swa ? 1 : 3;
                for (int b = 0; b < B; ++b) {
                    std::vector<const float*> blk_ptrs(blocks.size());
                    for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
                    std::vector<float> mx(nq * D);
                    attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, q_res.data(), nq, D, mx.data());
                    if (is_swa) {
                        std::memcpy(in0.data() + static_cast<size_t>(b) * D, mx.data(), D * 4);
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, mx.data(), D * 4);
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, mx.data(), D * 4);
                    } else {
                        std::memcpy(in0.data() + static_cast<size_t>(b) * D, mx.data() + 0 * D, D * 4);
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, mx.data() + 1 * D, D * 4);
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, mx.data() + 2 * D, D * 4);
                    }
                }
            } else {
                in0 = cur; in1 = cur; in2 = cur;
            }

            if (msgs != nullptr) {
                const char* cnames[3] = {"attn", "k", "v"};
                std::vector<float>* ins[3] = {&in0, &in1, &in2};
                for (int j = 0; j < 3; ++j) {
                    const std::string cp = p + ".controllers." + cnames[j];
                    const float* wd = trunk_f32_.at(cp + ".down_weight").data();
                    const float* wg = trunk_f32_.at(cp + ".gain_weight").data();
                    const float* wf = trunk_f32_.at(cp + ".forget_weight").data();
                    for (int b = 0; b < B; ++b) {
                        std::vector<float> out_c(D);
                        loop_control_f32(ins[j]->data() + static_cast<size_t>(b) * D,
                                         msgs->data() + (static_cast<size_t>(b) * 4 + j) * D,
                                         wd, wg, wf, D, c_.gate_rank, out_c.data());
                        std::memcpy(ins[j]->data() + static_cast<size_t>(b) * D, out_c.data(), D * 4);
                    }
                }
            }

            if (is_swa) {
                const float* an = trunk_f32_.at(p + ".attn_norm.weight").data();
                for (int b = 0; b < B; ++b) {
                    rms_norm_f32(in0.data() + static_cast<size_t>(b) * D, an, in0.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    if (msgs != nullptr) {
                        rms_norm_f32(in1.data() + static_cast<size_t>(b) * D, an, in1.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                        rms_norm_f32(in2.data() + static_cast<size_t>(b) * D, an, in2.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    } else {
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, in0.data() + static_cast<size_t>(b) * D, D * 4);
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, in0.data() + static_cast<size_t>(b) * D, D * 4);
                    }
                }
            } else {
                const float* qn = trunk_f32_.at(p + ".q_input_norm.weight").data();
                const float* kn = trunk_f32_.at(p + ".k_input_norm.weight").data();
                const float* vn = trunk_f32_.at(p + ".v_input_norm.weight").data();
                for (int b = 0; b < B; ++b) {
                    rms_norm_f32(in0.data() + static_cast<size_t>(b) * D, qn, in0.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    rms_norm_f32(in1.data() + static_cast<size_t>(b) * D, kn, in1.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    rms_norm_f32(in2.data() + static_cast<size_t>(b) * D, vn, in2.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                }
            }

            // GPU 0 dense projections
            std::vector<float> q(B * q_dim), k_raw(B * kv_dim), v_raw(B * kv_dim), r_raw(B * c_.n_head * c_.d_rel);
            std::vector<float> sinks(B * c_.n_head), gate_a(B * q_dim);
            if (!gpu_dense_proj(in0.data(), p + ".attn.wq_du.weight", q.data(), B, q_dim, D, err)) return false;
            if (!gpu_dense_proj(in1.data(), p + ".attn.wk_dv.weight", k_raw.data(), B, kv_dim, D, err)) return false;
            if (!gpu_dense_proj(in2.data(), p + ".attn.wv_dv.weight", v_raw.data(), B, kv_dim, D, err)) return false;
            if (!gpu_dense_proj(in0.data(), p + ".attn.wr_du.weight", r_raw.data(), B, c_.n_head * c_.d_rel, D, err)) return false;
            if (!gpu_dense_proj(in0.data(), p + ".attn.attn_sink_proj.weight", sinks.data(), B, c_.n_head, D, err)) return false;
            if (!gpu_dense_proj(in0.data(), p + ".attn.gate_attn.weight", gate_a.data(), B, q_dim, D, err)) return false;

            std::vector<float> k_dyn, v_dyn;
            if (!is_swa) {
                k_dyn.resize(B * c_.n_kv * 4);
                v_dyn.resize(B * c_.n_kv * 4);
                cpu_matmul(in1.data(), trunk_f32_.at(p + ".attn.k_sconv.weight_proj").data(), k_dyn.data(), B, c_.n_kv * 4, D);
                cpu_matmul(in2.data(), trunk_f32_.at(p + ".attn.v_sconv.weight_proj").data(), v_dyn.data(), B, c_.n_kv * 4, D);
            }

            const float* k_sw = static_sconv_.at(p + ".k_sconv").data();
            const float* v_sw = static_sconv_.at(p + ".v_sconv").data();
            const auto& q_clip_vec = trunk_f32_.at(p + ".attn.q_l2_clip.scale");
            const auto& r_clip_vec = trunk_f32_.at(p + ".attn.rel_l2_clip.scale");
            const bool q_clip_ph = q_clip_vec.size() >= static_cast<size_t>(c_.n_head * c_.head_dim);
            const bool r_clip_ph = r_clip_vec.size() >= static_cast<size_t>(c_.n_head * c_.d_rel);
            const float* rel_proj = trunk_f32_.at(p + ".attn.rel_logits_proj.proj").data();

            std::vector<float> attn_out(B * q_dim, 0.0f);
            std::vector<float> k_conv(kv_dim), v_conv(kv_dim), h_tmp(3 * kv_dim);
            std::vector<float> k_mem(kv_dim), v_mem(kv_dim);
            std::vector<float> rel_tab(c_.n_head * (c_.rel_extent + 1));
            const int n_rep = c_.n_head / c_.n_kv;
            const float attn_scale = 1.0f / std::sqrt(static_cast<float>(c_.head_dim));

            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                const int pos_b = batch.pos[b];

                uint16_t* kh_ptr = sl.sconv.k_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim;
                uint16_t* vh_ptr = sl.sconv.v_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim;
                SlotShortConvState::load_3lag(kh_ptr, h_tmp.data(), kv_dim);
                shortconv4_step_f32(k_raw.data() + static_cast<size_t>(b) * kv_dim, k_sw,
                                    is_swa ? nullptr : (k_dyn.data() + static_cast<size_t>(b) * c_.n_kv * 4),
                                    kv_dim, c_.n_kv, h_tmp.data(), k_conv.data());
                SlotShortConvState::store_3lag(h_tmp.data(), kh_ptr, kv_dim);

                SlotShortConvState::load_3lag(vh_ptr, h_tmp.data(), kv_dim);
                shortconv4_step_f32(v_raw.data() + static_cast<size_t>(b) * kv_dim, v_sw,
                                    is_swa ? nullptr : (v_dyn.data() + static_cast<size_t>(b) * c_.n_kv * 4),
                                    kv_dim, c_.n_kv, h_tmp.data(), v_conv.data());
                SlotShortConvState::store_3lag(h_tmp.data(), vh_ptr, kv_dim);

                float* qb = q.data() + static_cast<size_t>(b) * q_dim;
                float* rb = r_raw.data() + static_cast<size_t>(b) * c_.n_head * c_.d_rel;
                for (int h = 0; h < c_.n_head; ++h) {
                    const float* qsc = q_clip_ph ? (q_clip_vec.data() + h * c_.head_dim) : q_clip_vec.data();
                    const float* rsc = r_clip_ph ? (r_clip_vec.data() + h * c_.d_rel) : r_clip_vec.data();
                    l2_clip_f32(qb + h * c_.head_dim, qsc, qb + h * c_.head_dim, c_.head_dim, c_.qk_l2_clip);
                    l2_clip_f32(rb + h * c_.d_rel, rsc, rb + h * c_.d_rel, c_.d_rel, c_.rel_l2_clip);
                    if (!is_swa) {
                        for (int j = 0; j < c_.head_dim; ++j) qb[h * c_.head_dim + j] *= tau[b];
                        for (int j = 0; j < c_.d_rel; ++j) rb[h * c_.d_rel + j] *= tau[b];
                    }
                }
                for (int kh = 0; kh < c_.n_kv; ++kh) {
                    l2_clip_f32(k_conv.data() + kh * c_.head_dim, nullptr, k_conv.data() + kh * c_.head_dim, c_.head_dim, c_.qk_l2_clip);
                    rms_norm_f32(v_conv.data() + kh * c_.head_dim, nullptr, v_conv.data() + kh * c_.head_dim, c_.head_dim, c_.rms_eps);
                }

                if (is_swa) sl.kv.store_swa(kv_slot, pos_b, k_conv.data(), v_conv.data());
                else        sl.kv.store_global(kv_slot, pos_b, k_conv.data(), v_conv.data());

                const int k_start = is_swa ? std::max(0, pos_b - c_.swa_win + 1) : 0;
                const int max_d = std::min(pos_b, c_.rel_extent - 1);
                for (int h = 0; h < c_.n_head; ++h) {
                    const float* rh = rb + h * c_.d_rel;
                    float* th = rel_tab.data() + h * (c_.rel_extent + 1);
                    for (int d = 0; d <= max_d; ++d) {
                        float s = 0.0f;
                        for (int rc = 0; rc < c_.d_rel; ++rc) s += rh[rc] * rel_proj[rc * c_.rel_extent + d];
                        th[d] = s;
                    }
                    th[c_.rel_extent] = 0.0f;
                }

                std::vector<float> h_max(c_.n_head), h_sum(c_.n_head, 1.0f);
                for (int h = 0; h < c_.n_head; ++h) { h_max[h] = sinks[b * c_.n_head + h]; h_sum[h] = 1.0f; }
                float* ao = attn_out.data() + static_cast<size_t>(b) * q_dim;
                for (int kp = k_start; kp <= pos_b; ++kp) {
                    if (is_swa) sl.kv.load_swa(kv_slot, kp, k_mem.data(), v_mem.data());
                    else        sl.kv.load_global(kv_slot, kp, k_mem.data(), v_mem.data());
                    const int dist = std::min(pos_b - kp, c_.rel_extent);
                    for (int h = 0; h < c_.n_head; ++h) {
                        const int kvh = h / n_rep;
                        const float* qh = qb + h * c_.head_dim;
                        const float* kh = k_mem.data() + kvh * c_.head_dim;
                        const float* vh = v_mem.data() + kvh * c_.head_dim;
                        float dot = 0.0f;
                        for (int j = 0; j < c_.head_dim; ++j) dot += qh[j] * kh[j];
                        const float sc = dot * attn_scale + rel_tab[h * (c_.rel_extent + 1) + dist];
                        float* oh = ao + h * c_.head_dim;
                        if (sc > h_max[h]) {
                            const float eo = std::exp(h_max[h] - sc);
                            h_sum[h] = h_sum[h] * eo + 1.0f;
                            for (int j = 0; j < c_.head_dim; ++j) oh[j] = oh[j] * eo + vh[j];
                            h_max[h] = sc;
                        } else {
                            const float w = std::exp(sc - h_max[h]);
                            h_sum[h] += w;
                            for (int j = 0; j < c_.head_dim; ++j) oh[j] += w * vh[j];
                        }
                    }
                }
                for (int h = 0; h < c_.n_head; ++h) {
                    const float inv = 1.0f / h_sum[h];
                    float* oh = ao + h * c_.head_dim;
                    const float* gh = gate_a.data() + (static_cast<size_t>(b) * q_dim + h * c_.head_dim);
                    for (int j = 0; j < c_.head_dim; ++j) oh[j] = (oh[j] * inv) / (1.0f + std::exp(-gh[j]));
                }
            }

            std::vector<float> o_proj(B * D);
            if (!gpu_dense_proj(attn_out.data(), p + ".attn.wo_ud.weight", o_proj.data(), B, D, q_dim, err)) return false;

            const float* attn_sw = static_sconv_.at(p + ".attn_sconv").data();
            std::vector<float> a_sconv(B * D), h_d(3 * D);
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                uint16_t* ah_ptr = sl.sconv.attn_hist.data() + static_cast<size_t>(visit) * 3 * D;
                SlotShortConvState::load_3lag(ah_ptr, h_d.data(), D);
                shortconv4_step_f32(o_proj.data() + static_cast<size_t>(b) * D, attn_sw, nullptr, D, 0, h_d.data(),
                                    a_sconv.data() + static_cast<size_t>(b) * D);
                SlotShortConvState::store_3lag(h_d.data(), ah_ptr, D);
            }

            if (ip % c_.block_size == 0) {
                blocks.push_back(residual);
                cur = std::move(a_sconv);
            } else {
                for (size_t i = 0; i < cur.size(); ++i) cur[i] = residual[i] + a_sconv[i];
            }

            // MoE sublayer: AttnRes + optional loop controller + mlp_norm + router + DUAL-GPU EP / HYBRID DOORBELL MoE!
            residual = cur;
            const auto& mlp_q = trunk_f32_.at("model.llm.attn_res.mlp_queries." + std::to_string(ip));
            std::vector<float> mlp_in(B * D);
            for (int b = 0; b < B; ++b) {
                std::vector<const float*> blk_ptrs(blocks.size());
                for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
                attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, mlp_q.data(), 1, D,
                                 mlp_in.data() + static_cast<size_t>(b) * D);
            }
            if (msgs != nullptr) {
                const std::string cp = p + ".controllers.mlp";
                const float* wd = trunk_f32_.at(cp + ".down_weight").data();
                const float* wg = trunk_f32_.at(cp + ".gain_weight").data();
                const float* wf = trunk_f32_.at(cp + ".forget_weight").data();
                for (int b = 0; b < B; ++b) {
                    std::vector<float> out_c(D);
                    loop_control_f32(mlp_in.data() + static_cast<size_t>(b) * D,
                                     msgs->data() + (static_cast<size_t>(b) * 4 + 3) * D,
                                     wd, wg, wf, D, c_.gate_rank, out_c.data());
                    std::memcpy(mlp_in.data() + static_cast<size_t>(b) * D, out_c.data(), D * 4);
                }
            }
            const float* mn = trunk_f32_.at(p + ".mlp_norm.weight").data();
            for (int b = 0; b < B; ++b) {
                rms_norm_f32(mlp_in.data() + static_cast<size_t>(b) * D, mn, mlp_in.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
            }

            std::vector<float> r_logits(B * c_.n_experts);
            if (!gpu_dense_proj(mlp_in.data(), p + ".mlp.gate.weight", r_logits.data(), B, c_.n_experts, D, err)) return false;
            const float* r_bias = trunk_f32_.at(p + ".mlp.gate.bias").data();

            std::vector<std::vector<std::pair<int, float>>> exp_tokens(c_.n_experts);
            for (int b = 0; b < B; ++b) {
                std::vector<std::pair<float, int>> sc(c_.n_experts);
                std::vector<float> probs(c_.n_experts);
                for (int e = 0; e < c_.n_experts; ++e) {
                    float pr = 1.0f / (1.0f + std::exp(-r_logits[b * c_.n_experts + e]));
                    probs[e] = pr;
                    sc[e] = {pr + r_bias[e], e};
                }
                std::partial_sort(sc.begin(), sc.begin() + c_.top_k, sc.end(),
                                  [](const auto& a, const auto& c) { return a.first > c.first || (a.first == c.first && a.second < c.second); });
                float wsum = 0.0f;
                for (int k = 0; k < c_.top_k; ++k) wsum += probs[sc[k].second];
                const float inv_w = c_.route_scale / (wsum + 1e-20f);
                for (int k = 0; k < c_.top_k; ++k) {
                    int eid = sc[k].second;
                    exp_tokens[eid].push_back({b, probs[eid] * inv_w});
                }
            }

            // Execute active experts in parallel across GPU 0, GPU 1 (Dual-GPU EP!), and/or CPU Doorbell pool!
            std::vector<float> moe_out(static_cast<size_t>(B) * D, 0.0f);
            if (!dispatch_moe_ep(ip, mlp_in, exp_tokens, moe_out, err)) return false;

            const float* mlp_sw = static_sconv_.at(p + ".mlp_sconv").data();
            std::vector<float> m_sconv(B * D);
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                uint16_t* mh_ptr = sl.sconv.mlp_hist.data() + static_cast<size_t>(visit) * 3 * D;
                SlotShortConvState::load_3lag(mh_ptr, h_d.data(), D);
                shortconv4_step_f32(moe_out.data() + static_cast<size_t>(b) * D, mlp_sw, nullptr, D, 0, h_d.data(),
                                    m_sconv.data() + static_cast<size_t>(b) * D);
                SlotShortConvState::store_3lag(h_d.data(), mh_ptr, D);
            }
            for (size_t i = 0; i < cur.size(); ++i) cur[i] = residual[i] + m_sconv[i];
            ++visit;
            return true;
        };

        // Prelude (0..core_start-1)
        for (int ip = 0; ip < c_.core_start; ++ip) if (!run_visit(ip, 0, nullptr)) return false;
        // Looped Core (core_start..core_end-1 x num_passes)
        prelude = cur;
        prelude_blocks = blocks;
        for (int p_idx = 0; p_idx < c_.num_passes; ++p_idx) {
            if (p_idx > 0) {
                bank.assign(blocks.begin() + static_cast<ptrdiff_t>(prelude_blocks.size() + 1), blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = c_.core_start; ip < c_.core_end; ++ip) {
                std::vector<float> msgs;
                const std::vector<float>* msgs_ptr = nullptr;
                if (p_idx > 0) {
                    msgs.resize(static_cast<size_t>(B) * 4 * D);
                    const float* mq = msg_queries_[ip].data();
                    for (int b = 0; b < B; ++b) {
                        std::vector<const float*> bank_ptrs(bank.size());
                        for (size_t k = 0; k < bank.size(); ++k) bank_ptrs[k] = bank[k].data() + static_cast<size_t>(b) * D;
                        attn_res_mix_f32(bank_ptrs, bank_out.data() + static_cast<size_t>(b) * D,
                                         mq, 4, D, msgs.data() + static_cast<size_t>(b) * 4 * D);
                    }
                    msgs_ptr = &msgs;
                }
                if (!run_visit(ip, p_idx, msgs_ptr)) return false;
            }
        }
        // Coda (core_end..n_phys-1)
        for (int ip = c_.core_end; ip < c_.n_phys; ++ip) if (!run_visit(ip, 0, nullptr)) return false;

        // Output depth mix + output_norm
        const auto& out_q = trunk_f32_.at("model.llm.attn_res.output_query");
        const auto& out_nw = trunk_f32_.at("model.llm.norm.weight");
        std::vector<float> h_norm(static_cast<size_t>(B) * D);
        for (int b = 0; b < B; ++b) {
            std::vector<const float*> blk_ptrs(blocks.size());
            for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
            std::vector<float> h_out(D);
            attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, out_q.data(), 1, D, h_out.data());
            rms_norm_f32(h_out.data(), out_nw.data(), h_norm.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
        }
        maybe_tap(c_.n_visits, h_norm);

        for (int b = 0; b < B; ++b) {
            auto& sl = slots_[batch.slot_ids[b]];
            sl.kv.seq_len = std::max(sl.kv.seq_len, batch.pos[b] + 1);
        }
        if (taps_out) for (int k = 0; k < 4; ++k) taps_out->taps[k] = std::move(taps[k]);

        // Unembed + softcap(30.0) on GPU 0 (or CPU fallback for non-BF16 unembed)
        if (logits_out != nullptr) {
            std::vector<int> want_idx;
            for (int b = 0; b < B; ++b) if (batch.want_logits[b]) want_idx.push_back(b);
            const int n_want = static_cast<int>(want_idx.size());
            const int V = c_.vocab_unpadded;
            if (n_want > 0) {
                if (d_unembed_bf16_ != nullptr && n_want <= kMaxParallelSlots) {
                    std::vector<float> h_want(static_cast<size_t>(n_want) * D);
                    for (int wi = 0; wi < n_want; ++wi) {
                        std::memcpy(h_want.data() + static_cast<size_t>(wi) * D,
                                    h_norm.data() + static_cast<size_t>(want_idx[wi]) * D, D * 4);
                    }
                    EP_CK(cudaSetDevice(gpu_ids_[0]));
                    EP_CK(cudaMemcpyAsync(d_dense_x_, h_want.data(), h_want.size() * 4, cudaMemcpyHostToDevice, streams_[0]));
                    dim3 grid((V + 7) / 8, n_want);
                    k_unembed_softcap_bf16<<<grid, 256, 0, streams_[0]>>>(
                        d_dense_x_, d_unembed_bf16_, d_dense_y_, n_want, V, D, c_.logit_softcap);
                    EP_CK(cudaMemcpyAsync(logits_out, d_dense_y_, static_cast<size_t>(n_want) * V * 4, cudaMemcpyDeviceToHost, streams_[0]));
                    EP_CK(cudaStreamSynchronize(streams_[0]));
                } else {
                    const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
                    const float cap = c_.logit_softcap, inv_cap = 1.0f / cap;
                    std::vector<float> wrow(D);
                    for (int v = 0; v < V; ++v) {
                        decode_packed_row(unemb_pk, v, wrow.data());
                        for (int wi = 0; wi < n_want; ++wi) {
                            const float* hb = h_norm.data() + static_cast<size_t>(want_idx[wi]) * D;
                            float z = 0.0f;
                            for (int d = 0; d < D; ++d) z += hb[d] * wrow[d];
                            logits_out[static_cast<size_t>(wi) * V + v] = cap * std::tanh(z * inv_cap);
                        }
                    }
                }
            }
        }
        return true;
    }

    bool selfcheck(std::string& err) override {
        char buf[256]{};
        if (!hybrid::run_doorbell_and_kv_cuda_selfcheck(gpu_ids_[0], buf, sizeof(buf))) {
            err = std::string("hybrid doorbell / K8V4 CUDA selfcheck failed: ") + buf;
            return false;
        }
        // Verify GPU 0 & GPU 1 SF4 expert parity against exact CPU reference!
        const int D = 2304, F = 1152;
        const size_t w13_p = static_cast<size_t>(2 * F) * (D / 2);
        const size_t w13_s = static_cast<size_t>(2 * F) * (D / 16);
        const size_t w2_p  = static_cast<size_t>(D) * (F / 2);
        const size_t w2_s  = static_cast<size_t>(D) * (F / 16);
        std::vector<uint8_t> body(16 + w13_p + w13_s + w2_p + w2_s);
        float hdr[4] = {0.02f, 1.0f, 0.02f, 1.0f};
        std::memcpy(body.data(), hdr, 16);
        for (size_t i = 0; i < w13_p; ++i) body[16 + i] = static_cast<uint8_t>((i * 19 + 3) & 0xFFu);
        for (size_t i = 0; i < w13_s; ++i) body[16 + w13_p + i] = static_cast<uint8_t>(0x30u + (i & 15));
        const uint32_t dn_off = static_cast<uint32_t>(16 + w13_p + w13_s);
        for (size_t i = 0; i < w2_p; ++i) body[dn_off + i] = static_cast<uint8_t>((i * 23 + 7) & 0xFFu);
        for (size_t i = 0; i < w2_s; ++i) body[dn_off + w2_p + i] = static_cast<uint8_t>(0x30u + (i & 15));

        std::vector<float> x(D), y_ref(D), y_gpu(D);
        for (int i = 0; i < D; ++i) x[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.5f;
        eval_inkling2_expert(body.data(), SkCode::SF4, SkCode::SF4, dn_off, x.data(), D, F, y_ref.data());

        int check_gpus = std::max(1, n_gpus_);
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (opts_.gpus >= 2 && dev_count >= 2) check_gpus = 2;

        for (int g = 0; g < check_gpus; ++g) {
            int gid = (g < n_gpus_) ? gpu_ids_[g] : g;
            EP_CK(cudaSetDevice(gid));
            uint8_t* db = nullptr;
            float *dx = nullptr, *da = nullptr, *dy = nullptr;
            EP_CK(cudaMalloc(&db, body.size()));
            EP_CK(cudaMalloc(&dx, D * 4));
            EP_CK(cudaMalloc(&da, F * 4));
            EP_CK(cudaMalloc(&dy, D * 4));
            EP_CK(cudaMemcpy(db, body.data(), body.size(), cudaMemcpyHostToDevice));
            EP_CK(cudaMemcpy(dx, x.data(), D * 4, cudaMemcpyHostToDevice));
            k_expert_sf4_gu_situ<<<dim3((F + 7) / 8, 1), 256>>>(db, dx, da, D, F);
            k_expert_sf4_dn<<<dim3((D + 7) / 8, 1), 256>>>(db, dn_off, da, dy, D, F);
            EP_CK(cudaDeviceSynchronize());
            EP_CK(cudaMemcpy(y_gpu.data(), dy, D * 4, cudaMemcpyDeviceToHost));
            cudaFree(db); cudaFree(dx); cudaFree(da); cudaFree(dy);

            double max_d = 0.0;
            for (int i = 0; i < D; ++i) max_d = std::max(max_d, std::abs(double(y_gpu[i]) - double(y_ref[i])));
            if (max_d > 5e-4) {
                err = "GPU " + std::to_string(gid) + " SF4 expert parity mismatch: max_diff=" + std::to_string(max_d);
                return false;
            }
        }
        return true;
    }

private:
    struct PackedTensor {
        TInfo info;
        std::vector<uint8_t> bytes;
    };

    const InkModelFile* model_ = nullptr;
    BackendOpts opts_;
    InkCfg c_;
    DsparkCfg dspark_cfg_;
    KvDtype default_kv_dtype_ = KvDtype::FP16;
    bool has_dspark_ = false;

    int n_gpus_ = 1;
    int gpu_ids_[2] = {0, 0};
    double ep_split_ = 0.5;
    bool p2p_ok_ = false;
    bool doorbell_active_ = false;
    cudaStream_t streams_[2] = {nullptr, nullptr};

    uint8_t* d_expert_pool_[2] = {nullptr, nullptr};
    float* d_x_buf_[2] = {nullptr, nullptr};
    float* d_act_buf_[2] = {nullptr, nullptr};
    float* d_y_buf_[2] = {nullptr, nullptr};
    float* d_dense_x_ = nullptr;
    float* d_dense_y_ = nullptr;
    uint16_t* d_unembed_bf16_ = nullptr;
    std::unordered_map<std::string, uint16_t*> d_mats_bf16_;

    size_t vram_experts_total_ = 0;
    size_t host_experts_total_ = 0;
    size_t total_experts_ = 0;
    size_t vram_used_bytes_ = 0;
    size_t vram_total_bytes_ = 0;

    std::vector<DevExpert> experts_;
    std::vector<std::vector<uint8_t>> host_bodies_;
    std::vector<hybrid::HostExpertDesc> host_descs_;
    hybrid::DoorbellDeviceState db_{};

    std::unordered_map<std::string, std::vector<float>> trunk_f32_;
    std::unordered_map<std::string, PackedTensor> packed_trunk_;
    std::unordered_map<std::string, std::vector<float>> static_sconv_;
    std::vector<std::vector<float>> msg_queries_;
    std::unordered_map<std::string, std::vector<float>> dspark_f32_;
    std::vector<EpSlot> slots_;

    void cleanup() {
        if (doorbell_active_) {
            hybrid::stop_doorbell_pool();
            for (int nd = 0; nd < hybrid::kMaxNumaNodes; ++nd) {
                if (db_.h_bells[nd]) {
                    cudaFreeHost(db_.h_bells[nd]);
                    db_.h_bells[nd] = nullptr;
                    db_.d_bells[nd] = nullptr;
                }
            }
            doorbell_active_ = false;
        }
        if (n_gpus_ > 0) cudaSetDevice(gpu_ids_[0]);
        for (auto& kv : d_mats_bf16_) if (kv.second) cudaFree(kv.second);
        d_mats_bf16_.clear();
        if (d_unembed_bf16_) { cudaFree(d_unembed_bf16_); d_unembed_bf16_ = nullptr; }
        if (d_dense_x_) { cudaFree(d_dense_x_); d_dense_x_ = nullptr; }
        if (d_dense_y_) { cudaFree(d_dense_y_); d_dense_y_ = nullptr; }
        for (int g = 0; g < 2; ++g) {
            if (d_expert_pool_[g]) { cudaSetDevice(gpu_ids_[g]); cudaFree(d_expert_pool_[g]); d_expert_pool_[g] = nullptr; }
            if (d_x_buf_[g]) { cudaSetDevice(gpu_ids_[g]); cudaFree(d_x_buf_[g]); d_x_buf_[g] = nullptr; }
            if (d_act_buf_[g]) { cudaSetDevice(gpu_ids_[g]); cudaFree(d_act_buf_[g]); d_act_buf_[g] = nullptr; }
            if (d_y_buf_[g]) { cudaSetDevice(gpu_ids_[g]); cudaFree(d_y_buf_[g]); d_y_buf_[g] = nullptr; }
            if (streams_[g]) { cudaSetDevice(gpu_ids_[g]); cudaStreamDestroy(streams_[g]); streams_[g] = nullptr; }
        }
        vram_used_bytes_ = 0;
    }

    static std::vector<float> decode_section1_f32(const uint8_t* sec1, const TInfo& ti) {
        const size_t n = static_cast<size_t>(ti.numel());
        std::vector<float> out(n, 0.0f);
        const uint8_t* src = sec1 + ti.off;
        if (ti.fmt == F_F32) std::memcpy(out.data(), src, n * 4);
        else if (ti.fmt == F_BF16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        } else if (ti.fmt == F_F16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(p[i]);
        } else if (ti.fmt == F_Q8ROW) {
            const size_t rows = static_cast<size_t>(ti.rows()), cols = static_cast<size_t>(ti.cols());
            const float* sc = reinterpret_cast<const float*>(src);
            const int8_t* qs = reinterpret_cast<const int8_t*>(src + rows * 4);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c) out[r * cols + c] = static_cast<float>(qs[r * cols + c]) * sc[r];
        } else {
            dequant_sk_row(src, static_cast<SkCode>(ti.fmt), out.data(), n);
        }
        return out;
    }

    static void decode_packed_row(const PackedTensor& pk, int row, float* dst) noexcept {
        const size_t cols = static_cast<size_t>(pk.info.cols());
        const uint8_t* base = pk.bytes.data();
        if (pk.info.fmt == F_BF16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + static_cast<size_t>(row) * cols;
            for (size_t c = 0; c < cols; ++c) dst[c] = bf16_to_f32(p[c]);
        } else if (pk.info.fmt == F_F32) {
            std::memcpy(dst, base + static_cast<size_t>(row) * cols * 4, cols * 4);
        } else if (pk.info.fmt == F_Q8ROW) {
            const size_t rows = static_cast<size_t>(pk.info.rows());
            const float sc = reinterpret_cast<const float*>(base)[row];
            const int8_t* q = reinterpret_cast<const int8_t*>(base + rows * 4) + static_cast<size_t>(row) * cols;
            for (size_t c = 0; c < cols; ++c) dst[c] = static_cast<float>(q[c]) * sc;
        } else {
            const size_t sb_b = superblock_bytes(static_cast<SkCode>(pk.info.fmt));
            const size_t row_b = (cols / QK_S) * sb_b;
            dequant_sk_row(base + static_cast<size_t>(row) * row_b, static_cast<SkCode>(pk.info.fmt), dst, cols);
        }
    }

    static void cpu_matmul(const float* x, const float* w, float* y, int M, int N, int K) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<size_t>(m) * K;
            float* ym = y + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                const float* wn = w + static_cast<size_t>(n) * K;
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += xm[k] * wn[k];
                ym[n] = s;
            }
        }
    }

    bool gpu_dense_proj(const float* x, const std::string& wname, float* y, int M, int N, int K, std::string& err) {
        auto it = d_mats_bf16_.find(wname);
        const int max_k = std::max(c_.hidden, c_.n_head * c_.head_dim);
        const int max_n = std::max(std::max(c_.vocab_unpadded, c_.n_head * c_.head_dim), std::max(c_.hidden, c_.n_experts));
        if (it == d_mats_bf16_.end() || K > max_k || N > max_n) {
            cpu_matmul(x, trunk_f32_.at(wname).data(), y, M, N, K);
            return true;
        }
        EP_CK(cudaSetDevice(gpu_ids_[0]));
        for (int m0 = 0; m0 < M; m0 += kMaxParallelSlots) {
            const int cm = std::min(kMaxParallelSlots, M - m0);
            EP_CK(cudaMemcpyAsync(d_dense_x_, x + static_cast<size_t>(m0) * K,
                                  static_cast<size_t>(cm) * K * sizeof(float), cudaMemcpyHostToDevice, streams_[0]));
            dim3 grid((N + 7) / 8, cm);
            k_bf16_matmul<<<grid, 256, 0, streams_[0]>>>(d_dense_x_, it->second, d_dense_y_, cm, N, K);
            EP_CK(cudaMemcpyAsync(y + static_cast<size_t>(m0) * N, d_dense_y_,
                                  static_cast<size_t>(cm) * N * sizeof(float), cudaMemcpyDeviceToHost, streams_[0]));
            EP_CK(cudaStreamSynchronize(streams_[0]));
        }
        (void)err;
        return true;
    }

    // Launch one resident GPU expert on `g` (GPU 0 or GPU 1)
    void launch_gpu_expert(int g, const DevExpert& de, int m) {
        const int D = c_.hidden, F = c_.inter;
        cudaStream_t st = streams_[g];
        dim3 g_gu((F + 7) / 8, m);
        dim3 g_dn((D + 7) / 8, m);

        if (de.gu_code == static_cast<uint8_t>(SkCode::SF4)) {
            k_expert_sf4_gu_situ<<<g_gu, 256, 0, st>>>(de.body, d_x_buf_[g], d_act_buf_[g], D, F);
        } else if (de.gu_code == static_cast<uint8_t>(SkCode::MX8)) {
            k_expert_mx8_gu_situ<<<g_gu, 256, 0, st>>>(de.body, d_x_buf_[g], d_act_buf_[g], D, F);
        } else if (de.gu_code == static_cast<uint8_t>(SkCode::GGUF_RAW)) {
            k_expert_gguf_gu_situ<<<g_gu, 256, 0, st>>>(de.body, d_x_buf_[g], d_act_buf_[g], D, F);
        } else {
            k_expert_sk_gu_situ<<<g_gu, 256, 0, st>>>(de.body, de.gu_code, d_x_buf_[g], d_act_buf_[g], D, F);
        }

        if (de.dn_code == static_cast<uint8_t>(SkCode::SF4)) {
            k_expert_sf4_dn<<<g_dn, 256, 0, st>>>(de.body, de.dn_off, d_act_buf_[g], d_y_buf_[g], D, F);
        } else if (de.dn_code == static_cast<uint8_t>(SkCode::MX8)) {
            k_expert_mx8_dn<<<g_dn, 256, 0, st>>>(de.body, de.dn_off, d_act_buf_[g], d_y_buf_[g], D, F);
        } else if (de.dn_code == static_cast<uint8_t>(SkCode::GGUF_RAW)) {
            k_expert_gguf_dn<<<g_dn, 256, 0, st>>>(de.body, de.dn_off, d_act_buf_[g], d_y_buf_[g], D, F);
        } else {
            dim3 g_dn2((D / 2 + 7) / 8, m);
            k_expert_sk_dn_paired2<<<g_dn2, 256, 0, st>>>(de.body, de.dn_off, de.dn_code, d_act_buf_[g], d_y_buf_[g], D, F);
        }
    }

    // Dual-GPU EP + Hybrid Doorbell MoE Dispatch
    bool dispatch_moe_ep(int ip, const std::vector<float>& mlp_in,
                         const std::vector<std::vector<std::pair<int, float>>>& exp_tokens,
                         std::vector<float>& moe_out, std::string& err) {
        const int D = c_.hidden;
        struct Job {
            int eid;
            int m;
            std::vector<float> xe;
            std::vector<float> ye;
        };
        std::vector<Job> gpu_jobs[2], host_jobs;

        for (int e = 0; e < c_.n_experts; ++e) {
            if (exp_tokens[e].empty()) continue;
            const int key = ip * c_.n_experts + e;
            const auto& de = experts_[key];
            Job jb;
            jb.eid = e;
            jb.m = static_cast<int>(exp_tokens[e].size());
            jb.xe.resize(static_cast<size_t>(jb.m) * D);
            jb.ye.resize(static_cast<size_t>(jb.m) * D);
            for (int t = 0; t < jb.m; ++t) {
                std::memcpy(jb.xe.data() + static_cast<size_t>(t) * D,
                            mlp_in.data() + static_cast<size_t>(exp_tokens[e][t].first) * D, D * 4);
            }
            if (de.resident) {
                gpu_jobs[de.owner_gpu].push_back(std::move(jb));
            } else {
                host_jobs.push_back(std::move(jb));
            }
        }

        // Ring NUMA Doorbell CPU workers BEFORE GPU launch when B=1 decode has missed experts (#6520 overlap!)
        bool used_doorbell = false;
        uint32_t bell_seq[hybrid::kMaxNumaNodes] = {};
        int bell_node_of_job[hybrid::kDoorbellTopK] = {};
        int bell_slot_of_job[hybrid::kDoorbellTopK] = {};
        if (doorbell_active_ && !host_jobs.empty() && host_jobs.size() <= static_cast<size_t>(hybrid::kDoorbellTopK) &&
            D <= hybrid::kDoorbellHidden) {
            bool all_single = true;
            for (const auto& jb : host_jobs) {
                if (jb.m != 1 || exp_tokens[jb.eid][0].first != exp_tokens[host_jobs[0].eid][0].first) {
                    all_single = false;
                    break;
                }
            }
            if (all_single) {
                used_doorbell = true;
                static uint32_t host_bell_counter = 1;
                const uint32_t seq = ++host_bell_counter;
                int node_cnt[hybrid::kMaxNumaNodes] = {};
                for (size_t ji = 0; ji < host_jobs.size(); ++ji) {
                    const int key = ip * c_.n_experts + host_jobs[ji].eid;
                    const int nd = std::min(static_cast<int>(host_descs_[key].numa_node), db_.n_nodes - 1);
                    hybrid::InkDoorbell* bell = db_.h_bells[nd];
                    const int slot = node_cnt[nd]++;
                    bell->expert_id[slot] = host_jobs[ji].eid;
                    bell->slot_k[slot] = slot;
                    bell_node_of_job[ji] = nd;
                    bell_slot_of_job[ji] = slot;
                }
                for (int nd = 0; nd < db_.n_nodes; ++nd) {
                    if (node_cnt[nd] == 0) continue;
                    hybrid::InkDoorbell* bell = db_.h_bells[nd];
                    bell->layer = ip;
                    bell->n_jobs = node_cnt[nd];
                    std::memcpy(bell->x, host_jobs[0].xe.data(), static_cast<size_t>(D) * sizeof(float));
                    bell_seq[nd] = seq;
#if defined(_MSC_VER)
                    std::atomic_thread_fence(std::memory_order_release);
                    bell->ring = seq;
#else
                    __atomic_store_n(&bell->ring, seq, __ATOMIC_RELEASE);
#endif
                }
            }
        }

        // Launch GPU 0 and GPU 1 jobs concurrently (each GPU runs its own experts in VRAM, chunked to kMaxParallelSlots)
        size_t max_jobs = std::max(gpu_jobs[0].size(), gpu_jobs[1].size());
        for (size_t idx = 0; idx < max_jobs; ++idx) {
            int max_chunks = 0;
            for (int g = 0; g < n_gpus_; ++g) {
                if (idx < gpu_jobs[g].size()) {
                    const int nc = (gpu_jobs[g][idx].m + kMaxParallelSlots - 1) / kMaxParallelSlots;
                    if (nc > max_chunks) max_chunks = nc;
                }
            }
            for (int ci = 0; ci < max_chunks; ++ci) {
                const int off = ci * kMaxParallelSlots;
                for (int g = 0; g < n_gpus_; ++g) {
                    if (idx >= gpu_jobs[g].size()) continue;
                    auto& jb = gpu_jobs[g][idx];
                    if (off >= jb.m) continue;
                    const int cm = std::min(kMaxParallelSlots, jb.m - off);
                    const auto& de = experts_[ip * c_.n_experts + jb.eid];
                    EP_CK(cudaSetDevice(gpu_ids_[g]));
                    EP_CK(cudaMemcpyAsync(d_x_buf_[g], jb.xe.data() + static_cast<size_t>(off) * D,
                                          static_cast<size_t>(cm) * D * sizeof(float), cudaMemcpyHostToDevice, streams_[g]));
                    launch_gpu_expert(g, de, cm);
                    EP_CK(cudaMemcpyAsync(jb.ye.data() + static_cast<size_t>(off) * D, d_y_buf_[g],
                                          static_cast<size_t>(cm) * D * sizeof(float), cudaMemcpyDeviceToHost, streams_[g]));
                }
                for (int g = 0; g < n_gpus_; ++g) {
                    if (idx < gpu_jobs[g].size() && off < gpu_jobs[g][idx].m) {
                        EP_CK(cudaSetDevice(gpu_ids_[g]));
                        EP_CK(cudaStreamSynchronize(streams_[g]));
                    }
                }
            }
        }

        // Collect NUMA Doorbell results (or run direct SIMD for multi-token prefill host experts)
        if (used_doorbell) {
            for (int nd = 0; nd < db_.n_nodes; ++nd) {
                if (bell_seq[nd] == 0) continue;
                hybrid::InkDoorbell* bell = db_.h_bells[nd];
                for (;;) {
#if defined(_MSC_VER)
                    const uint32_t cur_ack = bell->ack[nd];
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (cur_ack == bell_seq[nd]) break;
#else
                    if (__atomic_load_n(&bell->ack[nd], __ATOMIC_ACQUIRE) == bell_seq[nd]) break;
#endif
                    std::this_thread::yield();
                }
            }
            for (size_t ji = 0; ji < host_jobs.size(); ++ji) {
                hybrid::InkDoorbell* bell = db_.h_bells[bell_node_of_job[ji]];
                const int slot = bell_slot_of_job[ji];
                std::memcpy(host_jobs[ji].ye.data(), bell->y[slot], static_cast<size_t>(D) * sizeof(float));
            }
        } else {
            for (auto& jb : host_jobs) {
                const int key = ip * c_.n_experts + jb.eid;
                const auto& hd = host_descs_[key];
                if (hd.body == nullptr) {
                    err = "cuda-ep: missing host body for non-resident expert " + std::to_string(key);
                    return false;
                }
                eval_inkling2_expert_batch(hd.body, static_cast<SkCode>(hd.gu_code), static_cast<SkCode>(hd.dn_code),
                                           hd.dn_offset, jb.xe.data(), static_cast<size_t>(jb.m), D, c_.inter, jb.ye.data());
            }
        }

        // Deterministic scatter-add into moe_out
        auto scatter = [&](const std::vector<Job>& jobs) {
            for (const auto& jb : jobs) {
                const auto& tl = exp_tokens[jb.eid];
                for (int t = 0; t < jb.m; ++t) {
                    const int b = tl[t].first;
                    const float w = tl[t].second;
                    float* dst = moe_out.data() + static_cast<size_t>(b) * D;
                    const float* src = jb.ye.data() + static_cast<size_t>(t) * D;
                    for (int d = 0; d < D; ++d) dst[d] += src[d] * w;
                }
            }
        };
        scatter(gpu_jobs[0]);
        scatter(gpu_jobs[1]);
        scatter(host_jobs);
        (void)err;
        return true;
    }
};

}  // namespace

std::unique_ptr<InkBackend> make_ep_cuda_backend() {
    return std::make_unique<EpDualGpuCudaBackend>();
}

}  // namespace ink2
