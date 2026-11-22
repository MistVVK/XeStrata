// src/kernels/cuda/vhead_bitplane.cu - Clean-room MIT exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived strictly from NVIDIA PTX ISA mma.sync.aligned.m16n8k32 / m16n8k16 (sm_80+) and Cauchy-Schwarz interval
// bounds on exact bit-plane decompositions:
//   - Q8_0 (int8 w in [-128..127]): w = 16 * hi4 + 8 + lo_c, hi4 = w >> 4 in [-8..7], lo_c = (w & 15) - 8 in [-8..7].
//   - Q6_K (int6 q in [-32..31]):   q =  4 * hi4 + 2 + lo_c, hi4 = q >> 2 in [-8..7], lo_c = (q &  3) - 2 in [-2..1].
//
// Overflow-free 2-stage tile-bound architecture (zero candidate list, zero atomic contention):
//   - Stage 1 streams ONLY the coalesced 4-bit hi_tiled plane (~half bandwidth) using INT8 Tensor Cores across all
//     T = 1..16 tokens simultaneously, computing exact per-tile upper bound `tile_hi[t][tile] = max_{r in tile} hi(r,t)`
//     (only 993 KB for 248K vocab x 16 tokens -> stays resident in L2!) and global lower bound
//     `g_max_lo[t] = max_r (approx(r, t) - rho[r] * ||x_t||_2)`.
//   - Stage 2 scans `tile_hi[t][tile] >= g_max_lo[t]` across all SMs (in L2, ~3 us) and evaluates the exact dot
//     product ONLY on the surviving 16-row tiles (typically 1..4 tiles per token), reducing the exact argmax via an
//     order-preserving 64-bit atomicMax (lowest row index wins ties). Zero false negatives, zero overflow regardless
//     of where the argmax row sits in the vocabulary.
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

struct BitplaneHeadHandle {
    int type = 0;            // 8 = Q8_0, 14 = Q6_K
    int n_in = 0;
    int n_out = 0;
    int n_tiles = 0;         // (n_out + 15) / 16
    int nb32 = 0;            // n_in / 32
    int n_scales = 0;        // Q8_0: n_in / 32; Q6_K: n_in / 16
    const void* raw_w = nullptr;
    void* hi_tiled = nullptr;      // Q8_0: uint2[n_tiles * nb32 * 32]; Q6_K: uint32_t[n_tiles * n_scales * 32]
    uint16_t* d_tiled = nullptr;   // uint16_t[n_tiles * n_scales * 16]
    float* rho = nullptr;          // float[n_out] exact residual L2 norm * (1 + safety)
    static constexpr int kMaxT = 16;
    float* d_x_norm = nullptr;     // [16]
    float* d_x_sum = nullptr;      // [16][n_scales]
    int32_t* d_max_lo = nullptr;   // [16] ordered-int32 for atomicMax
    float* d_tile_hi = nullptr;    // [16][n_tiles] (~993 KB for 248K vocab x 16 tokens)
    unsigned long long* d_best = nullptr; // [16] packed (ordered_u32(val) << 32) | (~row)
    size_t total_bytes = 0;
};

namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeQ6_K = 14;
constexpr float kBoundSafety = 1.0005f;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define STRATA_VHEAD_SM80 1
#else
#define STRATA_VHEAD_SM80 0
#endif

__device__ __forceinline__ int32_t float_to_ordered(float f) {
    const int32_t i = __float_as_int(f);
    return i >= 0 ? i : (i ^ 0x7fffffff);
}

__device__ __forceinline__ float ordered_to_float(int32_t o) {
    return __int_as_float(o >= 0 ? o : (o ^ 0x7fffffff));
}

__device__ __forceinline__ uint32_t float_to_u32_ordered(float f) {
    return (uint32_t) float_to_ordered(f) ^ 0x80000000u;
}

__device__ __forceinline__ unsigned long long pack_val_row(float val, int32_t row) {
    return ((unsigned long long) float_to_u32_ordered(val) << 32) | (unsigned long long) (0xFFFFFFFFu - (uint32_t) row);
}

__device__ __forceinline__ void atomic_max_float(int32_t* addr, float val) {
    atomicMax(addr, float_to_ordered(val));
}

__device__ __forceinline__ uint32_t unpack_s4x4(uint16_t p) {
    const uint32_t v = (uint32_t) (p & 0x000Fu) |
                       ((uint32_t) (p & 0x00F0u) << 4) |
                       ((uint32_t) (p & 0x0F00u) << 8) |
                       ((uint32_t) (p & 0xF000u) << 12);
    return v | ((v & 0x08080808u) * 0x1Eu);
}

#if STRATA_VHEAD_SM80
__device__ __forceinline__ void mma_s8_m16n8k32(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8_m16n8k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
#endif

// ---- Load-time lossless tiled repack kernels ----

__device__ __forceinline__ uint16_t pack_q8_hi4(const int8_t* qs, int k0) {
    const uint16_t h0 = (uint16_t) ((qs[k0 + 0] >> 4) & 0xF);
    const uint16_t h1 = (uint16_t) ((qs[k0 + 1] >> 4) & 0xF);
    const uint16_t h2 = (uint16_t) ((qs[k0 + 2] >> 4) & 0xF);
    const uint16_t h3 = (uint16_t) ((qs[k0 + 3] >> 4) & 0xF);
    return (uint16_t) (h0 | (h1 << 4) | (h2 << 8) | (h3 << 12));
}

__global__ void repack_q8_0_tiled_kernel(const uint8_t* __restrict__ src, uint2* __restrict__ hi_tiled,
                                         uint16_t* __restrict__ d_tiled, float* __restrict__ rho,
                                         int n_in, int n_out) {
    const int tile = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    if (tile >= n_tiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nb = n_in >> 5;
    const int r0 = tile * 16 + g;
    const int r1 = r0 + 8;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const uint8_t* src0 = src + (size_t) (v0 ? r0 : 0) * (size_t) nb * 34u;
    const uint8_t* src1 = src + (size_t) (v1 ? r1 : 0) * (size_t) nb * 34u;

    double err_sq0 = 0.0, err_sq1 = 0.0;
    for (int b = 0; b < nb; ++b) {
        uint16_t p00 = 0, p01 = 0, p10 = 0, p11 = 0;
        if (v0) {
            const uint8_t* blk = src0 + (size_t) b * 34u;
            const uint16_t d_bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            p00 = pack_q8_hi4(qs, tig * 4);
            p01 = pack_q8_hi4(qs, 16 + tig * 4);
            if (tig == 0) d_tiled[((size_t) tile * nb + b) * 16 + g] = d_bits;
            const float d = __half2float(__ushort_as_half(d_bits));
            int lsq = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ((int) qs[tig * 4 + j] & 15) - 8;
                const int c1 = ((int) qs[16 + tig * 4 + j] & 15) - 8;
                lsq += c0 * c0 + c1 * c1;
            }
            err_sq0 += (double) d * (double) d * (double) lsq;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * nb + b) * 16 + g] = 0;
        }
        if (v1) {
            const uint8_t* blk = src1 + (size_t) b * 34u;
            const uint16_t d_bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            p10 = pack_q8_hi4(qs, tig * 4);
            p11 = pack_q8_hi4(qs, 16 + tig * 4);
            if (tig == 0) d_tiled[((size_t) tile * nb + b) * 16 + 8 + g] = d_bits;
            const float d = __half2float(__ushort_as_half(d_bits));
            int lsq = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ((int) qs[tig * 4 + j] & 15) - 8;
                const int c1 = ((int) qs[16 + tig * 4 + j] & 15) - 8;
                lsq += c0 * c0 + c1 * c1;
            }
            err_sq1 += (double) d * (double) d * (double) lsq;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * nb + b) * 16 + 8 + g] = 0;
        }
        hi_tiled[((size_t) tile * nb + b) * 32 + lane] =
            make_uint2((uint32_t) p00 | ((uint32_t) p10 << 16), (uint32_t) p01 | ((uint32_t) p11 << 16));
    }

    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 1);
    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 2);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 1);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 2);
    if (tig == 0) {
        if (v0) rho[r0] = (float) sqrt(err_sq0) * kBoundSafety;
        if (v1) rho[r1] = (float) sqrt(err_sq1) * kBoundSafety;
    }
}

__device__ __forceinline__ int decode_q6k_elem(const uint8_t* blk, int sub, int j) {
    const int n = sub >> 3;
    const int rem = sub & 7;
    const int l = (rem & 1) * 16 + j;
    const int qid = rem >> 1;
    const uint8_t* ql_n = blk + 64 * n;
    const uint8_t* qh_n = blk + 128 + 32 * n;
    uint8_t low4 = 0;
    if (qid == 0) low4 = ql_n[l] & 0x0Fu;
    else if (qid == 1) low4 = ql_n[l + 32] & 0x0Fu;
    else if (qid == 2) low4 = ql_n[l] >> 4;
    else low4 = ql_n[l + 32] >> 4;
    const uint8_t hi2 = (qh_n[l] >> (2 * qid)) & 0x03u;
    return (int) (low4 | (hi2 << 4)) - 32;
}

__global__ void repack_q6_k_tiled_kernel(const uint8_t* __restrict__ src, uint32_t* __restrict__ hi_tiled,
                                         uint16_t* __restrict__ d_tiled, float* __restrict__ rho,
                                         int n_in, int n_out) {
    const int tile = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    if (tile >= n_tiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nsb = n_in >> 8;
    const int n_scales = n_in >> 4;
    const int r0 = tile * 16 + g;
    const int r1 = r0 + 8;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const uint8_t* src0 = src + (size_t) (v0 ? r0 : 0) * (size_t) nsb * 210u;
    const uint8_t* src1 = src + (size_t) (v1 ? r1 : 0) * (size_t) nsb * 210u;

    double err_sq0 = 0.0, err_sq1 = 0.0;
    for (int gsub = 0; gsub < n_scales; ++gsub) {
        const int sb = gsub >> 4;
        const int sub = gsub & 15;
        uint16_t p0 = 0, p1 = 0;

        if (v0) {
            const uint8_t* blk = src0 + (size_t) sb * 210u;
            const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
            const float d_all = __half2float(__ushort_as_half((uint16_t) blk[208] | ((uint16_t) blk[209] << 8)));
            const float d_sub = d_all * (float) sc;
            const __half d_h = __float2half_rn(d_sub);
            const float d_q = __half2float(d_h);
            if (tig == 0) d_tiled[((size_t) tile * n_scales + gsub) * 16 + g] = __half_as_ushort(d_h);
            const double d_err = fabs((double) d_sub - (double) d_q);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int v = decode_q6k_elem(blk, sub, tig * 4 + j);
                p0 |= (uint16_t) (((v >> 2) & 0xF) << (4 * j));
                const int c = (v & 3) - 2;
                const double e = (double) d_q * (double) c + d_err * (double) abs(v);
                err_sq0 += e * e;
            }
        } else if (tig == 0) {
            d_tiled[((size_t) tile * n_scales + gsub) * 16 + g] = 0;
        }

        if (v1) {
            const uint8_t* blk = src1 + (size_t) sb * 210u;
            const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
            const float d_all = __half2float(__ushort_as_half((uint16_t) blk[208] | ((uint16_t) blk[209] << 8)));
            const float d_sub = d_all * (float) sc;
            const __half d_h = __float2half_rn(d_sub);
            const float d_q = __half2float(d_h);
            if (tig == 0) d_tiled[((size_t) tile * n_scales + gsub) * 16 + 8 + g] = __half_as_ushort(d_h);
            const double d_err = fabs((double) d_sub - (double) d_q);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int v = decode_q6k_elem(blk, sub, tig * 4 + j);
                p1 |= (uint16_t) (((v >> 2) & 0xF) << (4 * j));
                const int c = (v & 3) - 2;
                const double e = (double) d_q * (double) c + d_err * (double) abs(v);
                err_sq1 += e * e;
            }
        } else if (tig == 0) {
            d_tiled[((size_t) tile * n_scales + gsub) * 16 + 8 + g] = 0;
        }

        hi_tiled[((size_t) tile * n_scales + gsub) * 32 + lane] = (uint32_t) p0 | ((uint32_t) p1 << 16);
    }

    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 1);
    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 2);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 1);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 2);
    if (tig == 0) {
        if (v0) rho[r0] = (float) sqrt(err_sq0) * kBoundSafety;
        if (v1) rho[r1] = (float) sqrt(err_sq1) * kBoundSafety;
    }
}

// ---- Per-step prep kernel ----
template <bool IS_Q6K>
__global__ void vhead_prep_kernel(const Q81Block* __restrict__ xq, int n_in, int T,
                                  float* __restrict__ x_norm, float* __restrict__ x_sum,
                                  int32_t* __restrict__ d_max_lo, unsigned long long* __restrict__ d_best) {
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int tid = (int) threadIdx.x;
    const int nb32 = n_in >> 5;
    const Q81Block* xt = xq + (size_t) t * (size_t) nb32;
    const int n_scales = IS_Q6K ? (n_in >> 4) : nb32;

    if (tid == 0) {
        d_max_lo[t] = float_to_ordered(-1e30f);
        d_best[t] = 0ull;
    }

    double sq = 0.0;
    for (int b = tid; b < nb32; b += blockDim.x) {
        const float dx = __low2float(xt[b].ds);
        const int8_t* q = xt[b].qs;
        int s_lo = 0, s_hi = 0, q_sq = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int v = (int) q[j];
            s_lo += v;
            q_sq += v * v;
        }
#pragma unroll
        for (int j = 16; j < 32; ++j) {
            const int v = (int) q[j];
            s_hi += v;
            q_sq += v * v;
        }
        if constexpr (IS_Q6K) {
            x_sum[(size_t) t * n_scales + 2 * b + 0] = (float) s_lo * dx;
            x_sum[(size_t) t * n_scales + 2 * b + 1] = (float) s_hi * dx;
        } else {
            x_sum[(size_t) t * n_scales + b] = (float) (s_lo + s_hi) * dx;
        }
        sq += (double) dx * (double) dx * (double) q_sq;
    }
    __shared__ double s_red[8];
    const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        double v = lane < (blockDim.x >> 5) ? s_red[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) x_norm[t] = (float) sqrt(v);
    }
}

// ---- Stage 1 Screen: High-occupancy K-chunked INT8 TC streaming over hi_tiled (~5 KB smem -> 4-6 CTAs/SM!) ----
// 4 warps (128 threads) per block, 64 output rows per block, non-persistent full grid (matching dense_i8_tc's
// 621 GB/s bandwidth utilization!), writing L2-resident tile_hi[T][n_tiles] with zero candidate-list atomics.
template <bool IS_Q6K, int MAX_NTILES>
__global__ void __launch_bounds__(128) vhead_stage1_tc_kernel(
    const void* __restrict__ hi_tiled, const uint16_t* __restrict__ d_tiled, const float* __restrict__ rho,
    const Q81Block* __restrict__ xq, const float* __restrict__ x_norm, const float* __restrict__ x_sum,
    int n_in, int n_out, int T,
    int32_t* __restrict__ d_max_lo, float* __restrict__ d_tile_hi) {
#if STRATA_VHEAD_SM80
    constexpr int WARPS = 4;
    constexpr int MAX_T = MAX_NTILES * 8;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nb32 = n_in >> 5;
    const int n_scales = IS_Q6K ? (n_in >> 4) : nb32;
    const int n_tiles = (n_out + 15) >> 4;
    const int tile = (int) blockIdx.x * WARPS + warp;
    if (tile >= n_tiles) return;

    // K-chunked 256-elem (8 block_q8_1) shared memory: only ~5.1 KB (T<=8) or ~10.2 KB (T<=16) -> max occupancy!
    __shared__ alignas(16) int8_t s_xq[MAX_T * 256];
    __shared__ float s_dx[MAX_T * 8];
    __shared__ float s_xsum[MAX_T * (IS_Q6K ? 16 : 8)];
    __shared__ float s_xnorm[MAX_T];

    if (tid < MAX_T) s_xnorm[tid] = tid < T ? x_norm[tid] : 0.0f;

    const int r_base = tile << 4;
    const int r0 = r_base + g;
    const int r1 = r_base + 8 + g;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    float acc[MAX_NTILES][4];
#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    const uint2* hi2_base = reinterpret_cast<const uint2*>(hi_tiled) + (size_t) tile * nb32 * 32u + lane;
    const uint32_t* hi1_base = reinterpret_cast<const uint32_t*>(hi_tiled) + (size_t) tile * n_scales * 32u + lane;
    const uint16_t* dt_base = d_tiled + (size_t) tile * n_scales * 16u;

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        // Cooperatively load MAX_T * 8 Q81Blocks into ~5 KB smem, pre-scaling dx and xsum ONCE!
        for (int idx = tid; idx < MAX_T * 8; idx += 128) {
            const int t = idx >> 3;
            const int b = idx & 7;
            if (t < T) {
                const Q81Block& blk = xq[(size_t) t * nb32 + k_blk0 + b];
                const float dx = __low2float(blk.ds);
                s_dx[idx] = IS_Q6K ? (4.0f * dx) : (16.0f * dx);
                if constexpr (IS_Q6K) {
                    s_xsum[t * 16 + 2 * b + 0] = 2.0f * x_sum[(size_t) t * n_scales + (k_blk0 + b) * 2 + 0];
                    s_xsum[t * 16 + 2 * b + 1] = 2.0f * x_sum[(size_t) t * n_scales + (k_blk0 + b) * 2 + 1];
                } else {
                    s_xsum[idx] = 8.0f * x_sum[(size_t) t * n_scales + (k_blk0 + b)];
                }
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(blk.qs);
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(src32[0], src32[1], src32[2], src32[3]);
                dst4[1] = make_uint4(src32[4], src32[5], src32[6], src32[7]);
            } else {
                s_dx[idx] = 0.0f;
                if constexpr (IS_Q6K) {
                    s_xsum[t * 16 + 2 * b + 0] = 0.0f;
                    s_xsum[t * 16 + 2 * b + 1] = 0.0f;
                } else {
                    s_xsum[idx] = 0.0f;
                }
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        if constexpr (!IS_Q6K) {
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint2 pk = __ldg(hi2_base + (size_t) (k_blk0 + b) * 32u);
                const uint32_t a0 = unpack_s4x4((uint16_t) (pk.x & 0xFFFFu));
                const uint32_t a1 = unpack_s4x4((uint16_t) (pk.x >> 16));
                const uint32_t a2 = unpack_s4x4((uint16_t) (pk.y & 0xFFFFu));
                const uint32_t a3 = unpack_s4x4((uint16_t) (pk.y >> 16));
                const uint16_t* db = dt_base + (size_t) (k_blk0 + b) * 16u;
                const float dw0 = __half2float(__ushort_as_half(__ldg(db + g)));
                const float dw1 = __half2float(__ushort_as_half(__ldg(db + 8 + g)));

#pragma unroll
                for (int nt = 0; nt < MAX_NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(dw0, fmaf(s_dx[t0 * 8 + b], (float) d[0], s_xsum[t0 * 8 + b]), acc[nt][0]);
                    acc[nt][1] = fmaf(dw0, fmaf(s_dx[t1 * 8 + b], (float) d[1], s_xsum[t1 * 8 + b]), acc[nt][1]);
                    acc[nt][2] = fmaf(dw1, fmaf(s_dx[t0 * 8 + b], (float) d[2], s_xsum[t0 * 8 + b]), acc[nt][2]);
                    acc[nt][3] = fmaf(dw1, fmaf(s_dx[t1 * 8 + b], (float) d[3], s_xsum[t1 * 8 + b]), acc[nt][3]);
                }
            }
        } else {
#pragma unroll
            for (int sb = 0; sb < 16; ++sb) {
                const int b = sb >> 1;
                const int half16 = sb & 1;
                const int gsub = k_blk0 * 2 + sb;
                const uint32_t pk = __ldg(hi1_base + (size_t) gsub * 32u);
                const uint32_t a0 = unpack_s4x4((uint16_t) (pk & 0xFFFFu));
                const uint32_t a1 = unpack_s4x4((uint16_t) (pk >> 16));
                const uint16_t* db = dt_base + (size_t) gsub * 16u;
                const float dw0 = __half2float(__ushort_as_half(__ldg(db + g)));
                const float dw1 = __half2float(__ushort_as_half(__ldg(db + 8 + g)));

#pragma unroll
                for (int nt = 0; nt < MAX_NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + half16 * 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k16(d, a0, a1, b0);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(dw0, fmaf(s_dx[t0 * 8 + b], (float) d[0], s_xsum[t0 * 16 + sb]), acc[nt][0]);
                    acc[nt][1] = fmaf(dw0, fmaf(s_dx[t1 * 8 + b], (float) d[1], s_xsum[t1 * 16 + sb]), acc[nt][1]);
                    acc[nt][2] = fmaf(dw1, fmaf(s_dx[t0 * 8 + b], (float) d[2], s_xsum[t0 * 16 + sb]), acc[nt][2]);
                    acc[nt][3] = fmaf(dw1, fmaf(s_dx[t1 * 8 + b], (float) d[3], s_xsum[t1 * 16 + sb]), acc[nt][3]);
                }
            }
        }
        __syncthreads();
    }

    const float rho0 = v0 ? __ldg(rho + r0) : 0.0f;
    const float rho1 = v1 ? __ldg(rho + r1) : 0.0f;

#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
        const int t0 = nt * 8 + 2 * tig;
        const int t1 = t0 + 1;
        const int ts[2] = {t0, t1};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int tc = ts[c];
            const bool active = tc < T;
            const float xn = active ? s_xnorm[tc] : 0.0f;
            const float ap0 = acc[nt][c];
            const float rad0 = rho0 * xn;
            const float lo0 = (active && v0) ? (ap0 - rad0) : -1e30f;
            const float hi0 = (active && v0) ? (ap0 + rad0) : -1e38f;
            const float ap1 = acc[nt][2 + c];
            const float rad1 = rho1 * xn;
            const float lo1 = (active && v1) ? (ap1 - rad1) : -1e30f;
            const float hi1 = (active && v1) ? (ap1 + rad1) : -1e38f;

            float tile_lo = fmaxf(lo0, lo1);
            float tile_hi = fmaxf(hi0, hi1);
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                tile_lo = fmaxf(tile_lo, __shfl_xor_sync(0xffffffffu, tile_lo, o));
                tile_hi = fmaxf(tile_hi, __shfl_xor_sync(0xffffffffu, tile_hi, o));
            }
            if (active && g == 0) {
                d_tile_hi[(size_t) tc * (size_t) n_tiles + (size_t) tile] = tile_hi;
                if (tile_lo > ordered_to_float(*reinterpret_cast<volatile int32_t*>(d_max_lo + tc))) {
                    atomic_max_float(d_max_lo + tc, tile_lo);
                }
            }
        }
    }
#else
    (void) hi_tiled; (void) d_tiled; (void) rho; (void) xq; (void) x_norm; (void) x_sum;
    (void) n_in; (void) n_out; (void) T; (void) d_max_lo; (void) d_tile_hi;
#endif
}

// ---- Exact dot helpers for Stage 2 ----
__device__ __forceinline__ float exact_dot_q8_0_warp(const uint8_t* __restrict__ raw_w, int row,
                                                     const Q81Block* __restrict__ xt, int nb32, int lane) {
    const uint8_t* rsrc = raw_w + (size_t) row * (size_t) nb32 * 34u;
    float sum = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t* blk = rsrc + (size_t) b * 34u;
        const float dw = __half2float(__ushort_as_half((uint16_t) blk[0] | ((uint16_t) blk[1] << 8)));
        const float dx = __low2float(xt[b].ds);
        const int* x4 = reinterpret_cast<const int*>(xt[b].qs);
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint8_t* p = blk + 2 + j * 4;
            const int wv = (int) ((uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24));
            dot = STRATA_DP4A(wv, x4[j], dot);
        }
        sum = fmaf(dw * dx, (float) dot, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}

__device__ __forceinline__ float exact_dot_q6_k_warp(const uint8_t* __restrict__ raw_w, int row,
                                                     const Q81Block* __restrict__ xt, int n_in, int lane) {
    const int nsb = n_in >> 8;
    const uint8_t* rsrc = raw_w + (size_t) row * (size_t) nsb * 210u;
    float sum = 0.0f;
    const int total_subs = nsb * 16;
    for (int gsub = lane; gsub < total_subs; gsub += 32) {
        const int sb = gsub >> 4;
        const int sub = gsub & 15;
        const uint8_t* blk = rsrc + (size_t) sb * 210u;
        const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
        const float d_all = __half2float(__ushort_as_half((uint16_t) blk[208] | ((uint16_t) blk[209] << 8)));
        const float d_sub = d_all * (float) sc;

        const int b32 = gsub >> 1;
        const int half16 = gsub & 1;
        const float dx = __low2float(xt[b32].ds);
        const int8_t* xq16 = xt[b32].qs + half16 * 16;

        int dot = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) dot += decode_q6k_elem(blk, sub, j) * (int) xq16[j];
        sum = fmaf(d_sub * dx, (float) dot, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}

// ---- Stage 2 Multi-CTA Exact Tile Refinement (grid = (32, T), 4 warps/CTA = 128 warps/token) ----
template <bool IS_Q6K>
__global__ void __launch_bounds__(128) vhead_stage2_tiles_kernel(
    const uint8_t* __restrict__ raw_w, const Q81Block* __restrict__ xq, int n_in, int n_out, int n_tiles,
    const int32_t* __restrict__ d_max_lo, const float* __restrict__ d_tile_hi,
    unsigned long long* __restrict__ d_best) {
    const int t = (int) blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp = (int) blockIdx.x * 4 + warp;
    const int total_warps = (int) gridDim.x * 4;

    const float final_lo = ordered_to_float(d_max_lo[t]);
    const float* th = d_tile_hi + (size_t) t * (size_t) n_tiles;
    const int nb32 = n_in >> 5;
    const Q81Block* xt = xq + (size_t) t * (size_t) nb32;

    unsigned long long best = 0ull;

    for (int tile = global_warp; tile < n_tiles; tile += total_warps) {
        if (th[tile] < final_lo) continue;
        const int r0 = tile << 4;
        const int r_end = min(r0 + 16, n_out);
        for (int row = r0; row < r_end; ++row) {
            const float val = IS_Q6K ? exact_dot_q6_k_warp(raw_w, row, xt, n_in, lane)
                                     : exact_dot_q8_0_warp(raw_w, row, xt, nb32, lane);
            if (lane == 0) {
                const unsigned long long cand = pack_val_row(val, row);
                if (cand > best) best = cand;
            }
        }
    }
    if (lane == 0 && best != 0ull) {
        atomicMax(d_best + t, best);
    }
}

__global__ void vhead_extract_tokens_kernel(const unsigned long long* __restrict__ d_best,
                                            const float* __restrict__ d_x_norm, int T,
                                            int32_t* __restrict__ out_tokens,
                                            uint32_t* __restrict__ out_nonfinite, int nf_stride) {
    const int t = (int) threadIdx.x;
    if (t < T) {
        const unsigned long long packed = d_best[t];
        out_tokens[t] = (int32_t) (0xFFFFFFFFu - (uint32_t) (packed & 0xFFFFFFFFull));
        if (out_nonfinite != nullptr && nf_stride > 0) {
            const float best_v = ordered_to_float((int32_t) ((uint32_t) (packed >> 32) ^ 0x80000000u));
            const uint32_t bad = (packed != 0ull && isfinite(d_x_norm[t]) && isfinite(best_v)) ? 0u : 1u;
            uint32_t* dst = out_nonfinite + (size_t) t * (size_t) nf_stride;
            dst[0] = bad;
            for (int b = 1; b < nf_stride; ++b) dst[b] = 0u;
        }
    }
}

int current_cc() {
#if defined(__HIPCC__)
    return 0;
#else
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) { (void) cudaGetLastError(); return 0; }
    int mj = 0, mn = 0;
    if (cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return mj * 10 + mn;
#endif
}

}  // namespace

bool vhead_bitplane_supported() {
    return current_cc() >= 80;
}

bool vhead_bitplane_enabled() {
    // Opt-in via STRATA_VHEAD_BITPLANE=1: keeps ~400 MiB VRAM available for the expert cache on partial-VRAM cards
    // and avoids Stage-2 tile-survival spikes on outlier-heavy LLM activations (especially Q6_K).
    static const bool env = [] {
        const char* v = std::getenv("STRATA_VHEAD_BITPLANE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return env && vhead_bitplane_supported();
}

BitplaneHeadHandle* vhead_bitplane_create(int ggml_type, const void* d_weights, int n_in, int n_out) {
    if (!vhead_bitplane_supported() || d_weights == nullptr || n_in <= 0 || n_out <= 0 || (n_in & 255) != 0)
        return nullptr;
    if (ggml_type != kTypeQ8_0 && ggml_type != kTypeQ6_K) return nullptr;

    auto* h = new BitplaneHeadHandle();
    h->type = ggml_type;
    h->n_in = n_in;
    h->n_out = n_out;
    h->n_tiles = (n_out + 15) >> 4;
    h->nb32 = n_in >> 5;
    h->n_scales = (ggml_type == kTypeQ6_K) ? (n_in >> 4) : h->nb32;
    h->raw_w = d_weights;

    const size_t hi_bytes = (ggml_type == kTypeQ8_0)
                                ? (size_t) h->n_tiles * (size_t) h->nb32 * 32u * sizeof(uint2)
                                : (size_t) h->n_tiles * (size_t) h->n_scales * 32u * sizeof(uint32_t);
    const size_t d_bytes = (size_t) h->n_tiles * (size_t) h->n_scales * 16u * sizeof(uint16_t);
    const size_t rho_bytes = (size_t) n_out * sizeof(float);
    const size_t scratch_xnorm = BitplaneHeadHandle::kMaxT * sizeof(float);
    const size_t scratch_xsum = (size_t) BitplaneHeadHandle::kMaxT * (size_t) h->n_scales * sizeof(float);
    const size_t scratch_maxlo = BitplaneHeadHandle::kMaxT * sizeof(int32_t);
    const size_t scratch_tilehi = (size_t) BitplaneHeadHandle::kMaxT * (size_t) h->n_tiles * sizeof(float);
    const size_t scratch_best = BitplaneHeadHandle::kMaxT * sizeof(unsigned long long);

    bool ok = cudaMalloc(&h->hi_tiled, hi_bytes) == cudaSuccess &&
              cudaMalloc(&h->d_tiled, d_bytes) == cudaSuccess &&
              cudaMalloc(&h->rho, rho_bytes) == cudaSuccess &&
              cudaMalloc(&h->d_x_norm, scratch_xnorm) == cudaSuccess &&
              cudaMalloc(&h->d_x_sum, scratch_xsum) == cudaSuccess &&
              cudaMalloc(&h->d_max_lo, scratch_maxlo) == cudaSuccess &&
              cudaMalloc(&h->d_tile_hi, scratch_tilehi) == cudaSuccess &&
              cudaMalloc(&h->d_best, scratch_best) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        vhead_bitplane_destroy(h);
        return nullptr;
    }

    h->total_bytes = hi_bytes + d_bytes + rho_bytes + scratch_xnorm + scratch_xsum +
                     scratch_maxlo + scratch_tilehi + scratch_best;

    const unsigned grid = (unsigned) ((h->n_tiles + 7) / 8);
    if (ggml_type == kTypeQ8_0) {
        repack_q8_0_tiled_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                reinterpret_cast<uint2*>(h->hi_tiled),
                                                h->d_tiled, h->rho, n_in, n_out);
    } else {
        repack_q6_k_tiled_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                reinterpret_cast<uint32_t*>(h->hi_tiled),
                                                h->d_tiled, h->rho, n_in, n_out);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        vhead_bitplane_destroy(h);
        return nullptr;
    }
    return h;
}

void vhead_bitplane_destroy(BitplaneHeadHandle* h) {
    if (h == nullptr) return;
    if (h->hi_tiled) cudaFree(h->hi_tiled);
    if (h->d_tiled) cudaFree(h->d_tiled);
    if (h->rho) cudaFree(h->rho);
    if (h->d_x_norm) cudaFree(h->d_x_norm);
    if (h->d_x_sum) cudaFree(h->d_x_sum);
    if (h->d_max_lo) cudaFree(h->d_max_lo);
    if (h->d_tile_hi) cudaFree(h->d_tile_hi);
    if (h->d_best) cudaFree(h->d_best);
    delete h;
}

size_t vhead_bitplane_bytes(const BitplaneHeadHandle* h) {
    return h ? h->total_bytes : 0;
}

bool vhead_bitplane_argmax(BitplaneHeadHandle* h, const void* x_q8_1, int T, int32_t* out_tokens, void* stream,
                           uint32_t* out_nonfinite, int nonfinite_stride) {
    if (h == nullptr || x_q8_1 == nullptr || out_tokens == nullptr || T < 1 || T > BitplaneHeadHandle::kMaxT)
        return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* xq = reinterpret_cast<const Q81Block*>(x_q8_1);
    const bool is_q6k = (h->type == kTypeQ6_K);

    if (is_q6k) {
        vhead_prep_kernel<true><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, h->d_x_norm, h->d_x_sum,
                                                             h->d_max_lo, h->d_best);
    } else {
        vhead_prep_kernel<false><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, h->d_x_norm, h->d_x_sum,
                                                              h->d_max_lo, h->d_best);
    }

    const unsigned grid_s1 = (unsigned) ((h->n_tiles + 3) / 4);
    const int ntiles = T <= 8 ? 1 : 2;

    if (is_q6k) {
        if (ntiles == 1)
            vhead_stage1_tc_kernel<true, 1><<<grid_s1, 128, 0, s>>>(h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm,
                                                                    h->d_x_sum, h->n_in, h->n_out, T, h->d_max_lo, h->d_tile_hi);
        else
            vhead_stage1_tc_kernel<true, 2><<<grid_s1, 128, 0, s>>>(h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm,
                                                                    h->d_x_sum, h->n_in, h->n_out, T, h->d_max_lo, h->d_tile_hi);
    } else {
        if (ntiles == 1)
            vhead_stage1_tc_kernel<false, 1><<<grid_s1, 128, 0, s>>>(h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm,
                                                                     h->d_x_sum, h->n_in, h->n_out, T, h->d_max_lo, h->d_tile_hi);
        else
            vhead_stage1_tc_kernel<false, 2><<<grid_s1, 128, 0, s>>>(h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm,
                                                                     h->d_x_sum, h->n_in, h->n_out, T, h->d_max_lo, h->d_tile_hi);
    }

    const dim3 grid_s2(32u, (unsigned) T);
    if (is_q6k) {
        vhead_stage2_tiles_kernel<true><<<grid_s2, 128, 0, s>>>(
            reinterpret_cast<const uint8_t*>(h->raw_w), xq, h->n_in, h->n_out, h->n_tiles,
            h->d_max_lo, h->d_tile_hi, h->d_best);
    } else {
        vhead_stage2_tiles_kernel<false><<<grid_s2, 128, 0, s>>>(
            reinterpret_cast<const uint8_t*>(h->raw_w), xq, h->n_in, h->n_out, h->n_tiles,
            h->d_max_lo, h->d_tile_hi, h->d_best);
    }
    vhead_extract_tokens_kernel<<<1, 32, 0, s>>>(h->d_best, h->d_x_norm, T, out_tokens, out_nonfinite, nonfinite_stride);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels
