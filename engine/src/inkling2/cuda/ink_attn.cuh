// engine/src/inkling2/cuda/ink_attn.cuh - Fused RelPos + Per-Query Sink Flash-GQA (24:8, head_dim=128) + Paged SWA/Global KV (FP16 & FP8)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Exact Inkling-2-Mini Attention Invariants:
//   - 24 Q heads, 8 KV heads (3:1 GQA), head_dim = 128, zero RoPE
//   - 40 SWA-1024 visits (sliding window = 1024 -> valid keys have 0 <= q_pos - k_pos < 1024)
//   -  8 Global visits (phys layers 5, 11, 17, 23 -> valid keys have 0 <= k_pos <= q_pos)
//   - Relative position bias: rel_proj_pad [16, 1025] (col 1024 = 0.0f), dist = clamp(q_pos - k_pos, 0, 1024),
//     bias = dot16(r[q, h], rel_proj_pad[:, dist])!
//   - Per-query attention sink folded algebraically EXACTLY into online softmax:
//     Initialize m = sink[q, h], l = 1.0f, acc = 0.0f -> exp(s - m) / (exp(sink - m) + sum exp(s_k - m))!
//   - Supports:
//       * Single-slot & multi-slot continuous batching (B = 1..64) with FP16 or FP8 E4M3 KV!
//       * Tree-16 speculative verify (each tree node attends committed prefix [0..p0-1] + its exact tree ancestors!)
//       * Fast Split-K FlashDecoding for high SM occupancy on RTX 5090 (170 SMs)!
#pragma once

#include "ink_cuda.cuh"
#include <cuda_fp8.h>

namespace strata::ink {

// ---- KV Cache Representation (Native GPU FP16, FP8 E4M3, K8V8 INT8, and K8V4 INT8/INT4!) ----
enum class InkKvDtype : int {
    FP16 = 0,
    FP8  = 1,
    K8V8 = 2,
    K8V4 = 3,
};

// Per-visit KV cache descriptor passed by value to attention kernels (CUDA-Graph friendly!)
// Supports zero-copy Host KV Offloading: global tokens [0..dev_cap-1] live in VRAM and [dev_cap..cap_pos-1]
// live in mapped pinned host RAM (cudaHostAllocMapped + cudaHostGetDevicePointer) so lossless FP16 KV never OOMs!
struct InkKvVisitDesc {
    __half*  k_f16;      // [max_slots, dev_cap, 8, 128] (kv_fmt == 0)
    __half*  v_f16;
    uint8_t* k_f8;       // [max_slots, dev_cap, 8, 128] (FP8 E4M3 / K8V8 int8 / K8V4 int8)
    uint8_t* v_f8;       // [max_slots, dev_cap, 8, 128] (FP8 / K8V8) or [max_slots, dev_cap, 8, 64] (K8V4 packed 4-bit)
    float*   k_sc;       // [max_slots, dev_cap, 8] per-head FP32 scale
    float*   v_sc;
    __half*  h_k_f16;    // Mapped pinned host K [max_slots, host_cap, 8, 128] (global visits only)
    __half*  h_v_f16;
    uint8_t* h_k_f8;
    uint8_t* h_v_f8;
    float*   h_k_sc;
    float*   h_v_sc;
    int      cap_pos;    // SWA: swa_ring; Global: dev_cap + host_cap (total allocated elastic capacity <= max_ctx)
    int      dev_cap;    // Global: VRAM-resident prefix [0..dev_cap-1]; 0 when 100% host-offloaded
    int      host_cap;   // Global: mapped pinned host tail [dev_cap..dev_cap+host_cap-1]; 0 when VRAM-only
    bool     is_swa;
    uint8_t  kv_fmt;     // 0=FP16, 1=FP8, 2=K8V8, 3=K8V4
};

// Unified inline 4-dim KV loader for all 4 formats (FP16 / FP8 / K8V8 / K8V4) + transparent Host KV Offload!
__device__ __forceinline__ void ink_kv_load4(
    const InkKvVisitDesc& kv, int sl, int p_idx, int kh, int lane, float4& kv4, float4& vv4) {
    const bool use_host = (!kv.is_swa && kv.host_cap > 0 && p_idx >= kv.dev_cap);
    const int stride = use_host ? kv.host_cap : (kv.host_cap > 0 ? kv.dev_cap : kv.cap_pos);
    const int pos_l  = use_host ? (p_idx - kv.dev_cap) : p_idx;
    const size_t hbase = ((size_t)sl * stride + pos_l) * kInkKVHeads + kh;
    const size_t base  = hbase * kInkHeadDim + (size_t)lane * 4;
    if (kv.kv_fmt == 0) {
        const __half* pk = use_host ? kv.h_k_f16 : kv.k_f16;
        const __half* pv = use_host ? kv.h_v_f16 : kv.v_f16;
        const uint2 k2 = *(const uint2*)(pk + base);
        const uint2 v2 = *(const uint2*)(pv + base);
        const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
        const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
        const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
        const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
        kv4 = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
        vv4 = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
    } else if (kv.kv_fmt == 1) {
        const uint8_t* pk = use_host ? kv.h_k_f8 : kv.k_f8;
        const uint8_t* pv = use_host ? kv.h_v_f8 : kv.v_f8;
        const float sk = (use_host ? kv.h_k_sc : kv.k_sc)[hbase];
        const float sv = (use_host ? kv.h_v_sc : kv.v_sc)[hbase];
        const uint32_t k4 = *(const uint32_t*)(pk + base);
        const uint32_t v4 = *(const uint32_t*)(pv + base);
        const float2 kf0 = e4m3x2_to_f32x2((uint16_t)(k4 & 0xFFFFu));
        const float2 kf1 = e4m3x2_to_f32x2((uint16_t)(k4 >> 16));
        const float2 vf0 = e4m3x2_to_f32x2((uint16_t)(v4 & 0xFFFFu));
        const float2 vf1 = e4m3x2_to_f32x2((uint16_t)(v4 >> 16));
        kv4 = make_float4(kf0.x * sk, kf0.y * sk, kf1.x * sk, kf1.y * sk);
        vv4 = make_float4(vf0.x * sv, vf0.y * sv, vf1.x * sv, vf1.y * sv);
    } else if (kv.kv_fmt == 2) {
        const uint8_t* pk = use_host ? kv.h_k_f8 : kv.k_f8;
        const uint8_t* pv = use_host ? kv.h_v_f8 : kv.v_f8;
        const float sk = (use_host ? kv.h_k_sc : kv.k_sc)[hbase];
        const float sv = (use_host ? kv.h_v_sc : kv.v_sc)[hbase];
        const uint32_t k4 = *(const uint32_t*)(pk + base);
        const uint32_t v4 = *(const uint32_t*)(pv + base);
        kv4 = make_float4((float)(int8_t)(k4 & 0xFFu) * sk, (float)(int8_t)((k4 >> 8) & 0xFFu) * sk,
                          (float)(int8_t)((k4 >> 16) & 0xFFu) * sk, (float)(int8_t)(k4 >> 24) * sk);
        vv4 = make_float4((float)(int8_t)(v4 & 0xFFu) * sv, (float)(int8_t)((v4 >> 8) & 0xFFu) * sv,
                          (float)(int8_t)((v4 >> 16) & 0xFFu) * sv, (float)(int8_t)(v4 >> 24) * sv);
    } else {
        const uint8_t* pk = use_host ? kv.h_k_f8 : kv.k_f8;
        const uint8_t* pv = use_host ? kv.h_v_f8 : kv.v_f8;
        const float sk = (use_host ? kv.h_k_sc : kv.k_sc)[hbase];
        const float sv = (use_host ? kv.h_v_sc : kv.v_sc)[hbase];
        const uint32_t k4 = *(const uint32_t*)(pk + base);
        kv4 = make_float4((float)(int8_t)(k4 & 0xFFu) * sk, (float)(int8_t)((k4 >> 8) & 0xFFu) * sk,
                          (float)(int8_t)((k4 >> 16) & 0xFFu) * sk, (float)(int8_t)(k4 >> 24) * sk);
        const uint16_t vp = *(const uint16_t*)(pv + hbase * (kInkHeadDim / 2) + (size_t)lane * 2);
        const int v0 = (int)(vp & 0xFu) - 8;
        const int v1 = (int)((vp >> 4) & 0xFu) - 8;
        const int v2 = (int)((vp >> 8) & 0xFu) - 8;
        const int v3 = (int)((vp >> 12) & 0xFu) - 8;
        vv4 = make_float4((float)v0 * sv, (float)v1 * sv, (float)v2 * sv, (float)v3 * sv);
    }
}

// Unified 128-thread per-head KV store (all 4 formats + transparent Host KV Offload + FP16 overflow clamp!)
__device__ __forceinline__ void ink_kv_store_128(
    const InkKvVisitDesc& kv, int sl, int p_idx, int kh, int tid, float val, bool is_k, float sh[4]) {
    const bool use_host = (!kv.is_swa && kv.host_cap > 0 && p_idx >= kv.dev_cap);
    const int stride = use_host ? kv.host_cap : (kv.host_cap > 0 ? kv.dev_cap : kv.cap_pos);
    const int pos_l  = use_host ? (p_idx - kv.dev_cap) : p_idx;
    const size_t hbase = ((size_t)sl * stride + pos_l) * kInkKVHeads + kh;
    const size_t base  = hbase * kInkHeadDim;
    if (kv.kv_fmt == 0) {
        __half* dst = is_k ? (use_host ? kv.h_k_f16 : kv.k_f16) : (use_host ? kv.h_v_f16 : kv.v_f16);
        dst[base + tid] = __float2half_rn(fminf(fmaxf(val, -65504.0f), 65504.0f));
        return;
    }
    float mv = warp_max(fabsf(val));
    if ((tid & 31) == 0) sh[tid >> 5] = mv;
    __syncthreads();
    const float am = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    uint8_t* dst8 = is_k ? (use_host ? kv.h_k_f8 : kv.k_f8) : (use_host ? kv.h_v_f8 : kv.v_f8);
    float* dst_sc = is_k ? (use_host ? kv.h_k_sc : kv.k_sc) : (use_host ? kv.h_v_sc : kv.v_sc);
    if (kv.kv_fmt == 1) {
        const float sc = am > 0.0f ? (am * (1.0f / 448.0f)) : 1.0f;
        const __nv_fp8_e4m3 qv(fminf(fmaxf(val / sc, -448.0f), 448.0f));
        dst8[base + tid] = qv.__x;
        if (tid == 0) dst_sc[hbase] = sc;
    } else if (kv.kv_fmt == 2 || is_k) {
        const float sc = am > 0.0f ? (am * (1.0f / 127.0f)) : 1.0f;
        const float inv = am > 0.0f ? (127.0f / am) : 0.0f;
        const int8_t qv = (int8_t)max(-127, min(127, __float2int_rn(val * inv)));
        dst8[base + tid] = (uint8_t)qv;
        if (tid == 0) dst_sc[hbase] = sc;
    } else {
        // K8V4: V quantized to signed 4-bit [-7..+7] biased by +8 -> [1..15], two dims packed per byte!
        const float sc = am > 0.0f ? (am * (1.0f / 7.0f)) : 1.0f;
        const float inv = am > 0.0f ? (7.0f / am) : 0.0f;
        const uint32_t q = (uint32_t)(max(-7, min(7, __float2int_rn(val * inv))) + 8);
        const uint32_t q_hi = __shfl_xor_sync(0xFFFFFFFFu, q, 1);
        if ((tid & 1) == 0) {
            dst8[hbase * (kInkHeadDim / 2) + (tid >> 1)] = (uint8_t)((q & 0xFu) | ((q_hi & 0xFu) << 4));
        }
        if (tid == 0) dst_sc[hbase] = sc;
    }
}

// Write K, V [M, 8, 128] FP32 into KV cache at positions pos_arr[m] for slot_arr[m]
// (For SWA, writes to ring index pos % cap_pos; for Global, guards 0 <= pos < cap_pos to prevent CUDA error 700!).
__global__ void __launch_bounds__(128) k_ink_kv_write(
    const float* __restrict__ k_in,     // [M, 8, 128]
    const float* __restrict__ v_in,     // [M, 8, 128]
    const int32_t* __restrict__ slots,  // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ pos,    // [M]
    InkKvVisitDesc kv,
    int M) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;  // 0..7
    const int d = threadIdx.x; // 0..127
    if (m >= M || h >= kInkKVHeads) return;

    const int sl = slots ? slots[m] : slot0;
    const int p = pos[m];
    if (p < 0 || (!kv.is_swa && p >= kv.cap_pos)) return;
    const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
    const float kv_val = k_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];
    const float vv_val = v_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];
    __shared__ float sk[4], sv[4];
    ink_kv_store_128(kv, sl, p_idx, h, d, kv_val, true, sk);
    ink_kv_store_128(kv, sl, p_idx, h, d, vv_val, false, sv);
}

// ---- FUSED 1-Launch Post-QKV Kernel (K/V ShortConv4 + Q/K/V/R Clip + V RMSNorm + Global Tau + KV Write/Stage!) ----
// Collapses 4 launches + 2 D2D memcpys into ONE launch for M=1 decode, B=1..64 batched, and Tree-16 verify!
// Grid: (64 blocks = 24 Q + 8 K + 8 V + 24 R, M tokens), Block: 128 threads (1 dim/thread!)
__global__ void __launch_bounds__(128) k_ink_post_qkv_fused(
    float* __restrict__ q,                 // [M, 24, 128] in/out
    const float* __restrict__ q_scale, int q_scale_elems,
    const float* __restrict__ k_raw,       // [M, 8, 128]
    const float* __restrict__ v_raw,       // [M, 8, 128]
    const float* __restrict__ sw_k,        // [1024, 4]
    const float* __restrict__ sw_v,        // [1024, 4]
    const float* __restrict__ k_dyn,       // [M, 8, 4] or nullptr
    const float* __restrict__ v_dyn,       // [M, 8, 4] or nullptr
    float* __restrict__ hist_k_pool,       // [max_slots, 3, 1024]
    float* __restrict__ hist_v_pool,       // [max_slots, 3, 1024]
    float* __restrict__ k_out,             // [M, 8, 128] (written in tree mode for tree_k!)
    float* __restrict__ v_out,             // [M, 8, 128] (written in tree mode for tree_v!)
    float* __restrict__ r,                 // [M, 24, 16] in/out
    const float* __restrict__ r_scale, int r_scale_elems,
    const int32_t* __restrict__ slots,     // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ pos_arr,   // [M]
    const int8_t* __restrict__ lag_idx,    // [M, 3] (tree mode only!)
    float* __restrict__ stage_k,           // [16, 1024] (tree mode only!)
    float* __restrict__ stage_v,           // [16, 1024] (tree mode only!)
    float* __restrict__ stage_sconv,       // [16, kInkSconvCh] (tree mode only!)
    InkKvVisitDesc kv,
    bool is_global,
    int mode,                              // 0 = single-tok M=1, 1 = batched B=1..64, 2 = Tree-16 verify
    int M) {
    const int m = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;  // 0..127
    if (m >= M) return;

    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(pos_arr[m] + 1) * (1.0f / (float)kInkLogFloor);
        tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
    }

    __shared__ float sh[8];
    if (blk < 24) {
        const int h = blk;
        float* qp = q + ((size_t)m * kInkHeads + h) * kInkHeadDim;
        float val = qp[tid];
        if (q_scale) {
            const int si = (q_scale_elems == kInkHeadDim) ? tid : (h * kInkHeadDim + tid);
            val *= q_scale[si];
        }
        float ss = block_sum(val * val, sh);
        const float factor = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau;
        qp[tid] = val * factor;
    } else if (blk < 40) {
        const bool is_k = (blk < 32);
        const int kh = is_k ? (blk - 24) : (blk - 32);
        const int c  = kh * kInkHeadDim + tid;
        const int sl = (mode == 1 && slots) ? slots[m] : slot0;
        const float* raw = is_k ? k_raw : v_raw;
        const float* swp = is_k ? sw_k : sw_v;
        const float* dyn = is_k ? k_dyn : v_dyn;
        float* hist = (is_k ? hist_k_pool : hist_v_pool) + (size_t)sl * 3 * kInkKVDim;

        const float cur = raw[(size_t)m * kInkKVDim + c];
        float h0, h1, h2;
        if (mode == 2) {
            auto fetch = [&](int idx) -> float {
                return idx >= 0 ? raw[(size_t)idx * kInkKVDim + c] : hist[(size_t)(3 + idx) * kInkKVDim + c];
            };
            h2 = fetch((int)lag_idx[m * 3 + 0]);
            h1 = fetch((int)lag_idx[m * 3 + 1]);
            h0 = fetch((int)lag_idx[m * 3 + 2]);
            if (stage_sconv) stage_sconv[(size_t)m * kInkSconvCh + (is_k ? 0 : kInkKVDim) + c] = cur;
        } else {
            h0 = hist[0 * kInkKVDim + c];
            h1 = hist[1 * kInkKVDim + c];
            h2 = hist[2 * kInkKVDim + c];
            hist[0 * kInkKVDim + c] = h1;
            hist[1 * kInkKVDim + c] = h2;
            hist[2 * kInkKVDim + c] = cur;
        }
        const float4 sw = reinterpret_cast<const float4*>(swp)[c];
        float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        const float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
        if (dyn) {
            const float4 dt = reinterpret_cast<const float4*>(dyn + ((size_t)m * kInkKVHeads + kh) * 4)[0];
            acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
        }
        float val = (cur + acc) * sc_conv;

        // K: L2 clip; V: parameterless RMSNorm
        float ss = block_sum(val * val, sh);
        if (is_k) {
            val *= fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f));
        } else {
            val *= rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
        }

        if (mode == 2) {
            // Round to FP16 when KV cache is FP16 so tree_k/tree_v are 100% bitwise identical to committed KV reads!
            const float kv_val = (kv.kv_fmt != 0) ? val : __half2float(__float2half_rn(fminf(fmaxf(val, -65504.0f), 65504.0f)));
            (is_k ? k_out : v_out)[(size_t)m * kInkKVDim + c] = kv_val;
            (is_k ? stage_k : stage_v)[(size_t)m * kInkKVDim + c] = val;
        } else {
            // Direct write into KV cache (FP16 / FP8 / K8V8 / K8V4) with defensive OOB guard!
            const int p = pos_arr[m];
            if (p >= 0 && (kv.is_swa || p < kv.cap_pos)) {
                const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
                ink_kv_store_128(kv, sl, p_idx, kh, tid, val, is_k, sh);
            }
        }
    } else if (blk < 64) {
        if (tid < kInkDRel) {
            const int h = blk - 40;
            float* rp = r + ((size_t)m * kInkHeads + h) * kInkDRel;
            float val = rp[tid];
            if (r_scale) {
                const int si = (r_scale_elems == kInkDRel) ? tid : (h * kInkDRel + tid);
                val *= r_scale[si];
            }
            float ss = val * val;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0x0000FFFFu, ss, o);
            const float factor = fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau;
            rp[tid] = val * factor;
        }
    }
}

// ---- Precompute relative-position bias table for a batch of M queries:
//      rel_table[m, h, dist] = sum_{c=0..15} r[m, h, c] * rel_proj_pad[c, dist] for dist in 0..1024!
//      Grid: (24 heads, M tokens) — wait: 1025 cols = 4 iterations of 256 threads! Only 16 FMAs/thread -> < 5 us!
__global__ void __launch_bounds__(256) k_ink_rel_table(
    const float* __restrict__ r,          // [M, 24, 16]
    const float* __restrict__ rel_proj,   // [16, 1024] F32 (note: col 1024 is implicitly 0.0f!)
    float* __restrict__ table,            // [M, 24, 1025] F32
    int M) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;  // 0..23
    if (m >= M || h >= kInkHeads) return;
    __shared__ float sr[kInkDRel];
    if (threadIdx.x < kInkDRel) sr[threadIdx.x] = r[((size_t)m * kInkHeads + h) * kInkDRel + threadIdx.x];
    __syncthreads();

    float* dst = table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
    for (int d = threadIdx.x; d < kInkRelExtent; d += 256) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < kInkDRel; ++c) {
            acc = fmaf(sr[c], rel_proj[(size_t)c * kInkRelExtent + d], acc);
        }
        dst[d] = acc;
    }
    if (threadIdx.x == 0) dst[kInkRelExtent] = 0.0f;
}

// ---- High-Speed Flash-GQA Kernel (Unified for Decode, Batched Decode B=1..64, Prefill, AND Tree-16 Verify!) ----
// One block of 128 threads (4 warps) per (query_idx m, q_head h) — OR Split-K when kv_len is large!
// Each thread owns 1 dimension d in 0..127 (or 4 warps cooperate over K tiles, 4 dims/lane)!
// WAIT: Using 1 warp (32 lanes, 4 dims/lane = float4!) per K-split -> 4 warps = 4-way K-parallelism inside ONE block!
// Let's verify that!
//   - 4 warps = 128 threads per block:
//   - Each warp holds its OWN independent online-softmax state (m_w, l_w, acc_w[4]) over stride-4 keys (k = k0 + warp, k0 + warp + 4, ...)!
//   - Each lane (0..31) holds 4 contiguous dims (d = 4*lane .. 4*lane+3) -> 128-bit float4 / half4 loads!!
//   - Inside the inner loop over keys, dot(q, k) is a single 5-step warp reduction (NO __syncthreads() in the hot loop!!)!
//   - At the end of the K-loop, ONE tiny 4-warp online-softmax merge in shared memory + fold per-query sink + multiply by sigmoid(gate_attn)!
// THAT IS INSANELY FAST!! Zero shared-memory syncs inside the inner loop, 100% coalesced 128-bit loads, and exact online softmax!

__global__ void __launch_bounds__(128) k_ink_flash_gqa(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    const float* __restrict__ gate_attn,  // [M, 24, 128] (optional: fused * sigmoid(gate_attn)!)
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,    // [M] (null -> slot 0)
    const int32_t* __restrict__ q_pos,    // [M] query position of each token m
    // Optional tree verify fields (nullptr for normal decode/prefill/batched):
    const float* __restrict__ tree_k,     // [M, 8, 128] current batch's K (uncommitted tree nodes)
    const float* __restrict__ tree_v,     // [M, 8, 128] current batch's V
    const uint8_t* __restrict__ tree_anc, // [M, 16] ancestor node indices for node m (depth+1 entries)
    const uint8_t* __restrict__ tree_dep, // [M] depth of node m in tree (0..15)
    int tree_p0,                          // committed prefix length before tree root (prefix keys = [0..tree_p0-1])
    float* __restrict__ out,              // [M, 24, 128]
    int M) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;  // 0..23
    if (m >= M || h >= kInkHeads) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..3
    const int lane = tid & 31;   // 0..31 -> dims 4*lane .. 4*lane+3
    const int kh   = h / 3;      // 24:8 GQA (3 Q heads per KV head!)
    const int sl   = slots ? slots[m] : 0;
    const int qp   = q_pos[m];
    const float scale = 0.08838834764831845f;  // 1 / sqrt(128)

    const float4 qv = *(const float4*)(q + ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4);
    const float* rt = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);

    const int tp0 = tree_k ? q_pos[0] : tree_p0;
    // Determine valid committed KV range [k_start, k_end)
    const int k_start = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const int k_end   = tree_k ? tp0 : (qp + 1);

    // Each of 4 warps maintains its own online-softmax accumulator!
    // Warp 0 initializes with the per-query sink (m = sink, l = 1.0f); warps 1..3 start empty (m = -inf, l = 0.0f)!
    float mw = (warp == 0) ? sinks[(size_t)m * kInkHeads + h] : -1e30f;
    float lw = (warp == 0) ? 1.0f : 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    auto attend_kv = [&](float4 kv4, float4 vv4, int k_pos) {
        float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
        dot = warp_sum(dot) * scale;
        const int dist = min(max(qp - k_pos, 0), kInkRelExtent);
        const float sc = dot + rt[dist];
        const float m_new = fmaxf(mw, sc);
        const float alpha = __expf(mw - m_new);
        const float p = __expf(sc - m_new);
        lw = lw * alpha + p;
        a0 = a0 * alpha + p * vv4.x;
        a1 = a1 * alpha + p * vv4.y;
        a2 = a2 * alpha + p * vv4.z;
        a3 = a3 * alpha + p * vv4.w;
        mw = m_new;
    };

    // 1. Committed KV cache loop (stride 4 across the 4 warps)
    const int k_end_safe = kv.is_swa ? k_end : min(k_end, kv.cap_pos);
    for (int kp = k_start + warp; kp < k_end_safe; kp += 4) {
        const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
        float4 kv4, vv4;
        ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        attend_kv(kv4, vv4, kp);
    }

    // 2. Tree-32 ancestor keys (only when tree_k != nullptr)
    if (tree_k) {
        const int dep = (int)tree_dep[m];
        for (int d_i = warp; d_i <= dep; d_i += 4) {
            const int anc_node = (int)tree_anc[m * kInkMaxTreeDepth + d_i];
            const int kp = tp0 + d_i;
            if (kv.is_swa && (qp - kp) >= kInkSwaWin) continue;
            const size_t tbase = ((size_t)anc_node * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            const float4 kv4 = *(const float4*)(tree_k + tbase);
            const float4 vv4 = *(const float4*)(tree_v + tbase);
            attend_kv(kv4, vv4, kp);
        }
    }

    // 3. Merge the 4 warps' online-softmax accumulators in shared memory!
    __shared__ float sm_m[4];
    __shared__ float sm_l[4];
    __shared__ float sm_a[4][kInkHeadDim];
    if (lane == 0) {
        sm_m[warp] = mw;
        sm_l[warp] = lw;
    }
    sm_a[warp][lane * 4 + 0] = a0;
    sm_a[warp][lane * 4 + 1] = a1;
    sm_a[warp][lane * 4 + 2] = a2;
    sm_a[warp][lane * 4 + 3] = a3;
    __syncthreads();

    if (warp == 0) {
        float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
        float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
        float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
        float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
        const float l_tot = sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3;
        const float inv_l = 1.0f / l_tot;

        const size_t obase = ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4;
        float o0 = (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv_l;
        float o1 = (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv_l;
        float o2 = (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv_l;
        float o3 = (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv_l;

        if (gate_attn) {
            const float4 gv = *(const float4*)(gate_attn + obase);
            o0 *= 1.0f / (1.0f + __expf(-gv.x));
            o1 *= 1.0f / (1.0f + __expf(-gv.y));
            o2 *= 1.0f / (1.0f + __expf(-gv.z));
            o3 *= 1.0f / (1.0f + __expf(-gv.w));
        }
        *(float4*)(out + obase) = make_float4(o0, o1, o2, o3);
    }
}

// ============================================================================
// Adaptive Split-K FlashDecoding with 3-Head GQA KV Sharing (up to 32 splits -> 256 CTAs on M=1 -> saturates sm_75..sm_120!)
// ============================================================================
constexpr int kInkAttnSplits = 8;
constexpr int kInkAttnMaxSplits = 32;

__global__ void __launch_bounds__(128) k_ink_flash_gqa_splitk(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,    // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ q_pos,    // [M]
    const float* __restrict__ tree_k,     // [M, 8, 128] optional
    const float* __restrict__ tree_v,     // [M, 8, 128] optional
    const uint8_t* __restrict__ tree_anc, // [M, 16]
    const uint8_t* __restrict__ tree_dep, // [M]
    int tree_p0,
    float* __restrict__ part_m,           // [M, 24, n_splits]
    float* __restrict__ part_l,           // [M, 24, n_splits]
    float* __restrict__ part_o,           // [M, 24, n_splits, 128]
    int M,
    int n_splits = kInkAttnSplits) {
    const int kh = blockIdx.x;  // 0..7
    const int sp = blockIdx.y;  // 0..n_splits-1
    const int m  = blockIdx.z;  // 0..M-1
    if (kh >= kInkKVHeads || sp >= n_splits || m >= M) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;  // 0..3
    const int lane = tid & 31;  // 0..31 -> dims 4*lane .. 4*lane+3
    const int sl   = slots ? slots[m] : slot0;
    const int qp   = q_pos[m];
    const float scale = 0.08838834764831845f;

    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + h0 + 0) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + h0 + 1) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + h0 + 2) * (kInkRelExtent + 1);

    const int tp0 = tree_k ? q_pos[0] : tree_p0;
    const int k_start = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const int k_end   = qp + 1;
    const int k_len   = max(0, k_end - k_start);
    const int per_sp  = (k_len + n_splits - 1) / n_splits;
    const int s_beg   = k_start + sp * per_sp;
    const int s_end   = min(k_end, s_beg + per_sp);

    // Split 0 / warp 0 folds the per-query sink!
    float mw[3] = {
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 0] : -1e30f,
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 1] : -1e30f,
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 2] : -1e30f
    };
    float lw[3] = {
        (sp == 0 && warp == 0) ? 1.0f : 0.0f,
        (sp == 0 && warp == 0) ? 1.0f : 0.0f,
        (sp == 0 && warp == 0) ? 1.0f : 0.0f
    };
    float a[3][4] = {};

    auto attend3 = [&](float4 kv4, float4 vv4, int k_pos) {
        const int dist = min(max(qp - k_pos, 0), kInkRelExtent);
        float d0 = warp_sum(qv0.x * kv4.x + qv0.y * kv4.y + qv0.z * kv4.z + qv0.w * kv4.w) * scale + rt0[dist];
        float d1 = warp_sum(qv1.x * kv4.x + qv1.y * kv4.y + qv1.z * kv4.z + qv1.w * kv4.w) * scale + rt1[dist];
        float d2 = warp_sum(qv2.x * kv4.x + qv2.y * kv4.y + qv2.z * kv4.z + qv2.w * kv4.w) * scale + rt2[dist];
        const float sc[3] = {d0, d1, d2};
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const float mn = fmaxf(mw[r], sc[r]);
            const float al = __expf(mw[r] - mn);
            const float p  = __expf(sc[r] - mn);
            lw[r] = lw[r] * al + p;
            a[r][0] = a[r][0] * al + p * vv4.x;
            a[r][1] = a[r][1] * al + p * vv4.y;
            a[r][2] = a[r][2] * al + p * vv4.z;
            a[r][3] = a[r][3] * al + p * vv4.w;
            mw[r] = mn;
        }
    };

    // Unified bit-exact loop: every key kp (committed OR tree ancestor!) lands in the EXACT same (split, warp) as M=1!
    for (int kp = s_beg + warp; kp < s_end; kp += 4) {
        float4 kv4, vv4;
        if (tree_k && kp >= tp0) {
            const int anc_node = (int)tree_anc[m * kInkMaxTreeDepth + (kp - tp0)];
            const size_t tbase = ((size_t)anc_node * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            kv4 = *(const float4*)(tree_k + tbase);
            vv4 = *(const float4*)(tree_v + tbase);
        } else {
            if (!kv.is_swa && (kp < 0 || kp >= kv.cap_pos)) continue;
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        }
        attend3(kv4, vv4, kp);
    }

    // Merge 4 warps inside this split for each of the 3 Q heads -> write to part_m / part_l / part_o!
    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
        sm_a[warp][lane * 4 + 0] = a[r][0];
        sm_a[warp][lane * 4 + 1] = a[r][1];
        sm_a[warp][lane * 4 + 2] = a[r][2];
        sm_a[warp][lane * 4 + 3] = a[r][3];
        __syncthreads();
        if (warp == 0) {
            float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
            float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
            float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
            float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
            const float lt = sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3;
            const size_t idx_sp = ((size_t)m * kInkHeads + h0 + r) * n_splits + sp;
            if (lane == 0) { part_m[idx_sp] = mg; part_l[idx_sp] = lt; }
            *(float4*)(part_o + idx_sp * kInkHeadDim + lane * 4) = make_float4(
                sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3,
                sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3,
                sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3,
                sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3);
        }
        __syncthreads();
    }
}

// Merge n_splits (8..32) + fused * sigmoid(gate_attn)! Grid: (24 heads, M), Block: 128 (1 dim/thread!)
__global__ void __launch_bounds__(128) k_ink_flash_gqa_merge(
    const float* __restrict__ part_m,     // [M, 24, n_splits]
    const float* __restrict__ part_l,     // [M, 24, n_splits]
    const float* __restrict__ part_o,     // [M, 24, n_splits, 128]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    float* __restrict__ out,              // [M, 24, 128]
    int M,
    int n_splits = kInkAttnSplits) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;  // 0..127
    if (h >= kInkHeads || m >= M) return;

    const size_t base_sp = ((size_t)m * kInkHeads + h) * n_splits;
    float mg = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        if (part_l[base_sp + s] > 0.0f) mg = fmaxf(mg, part_m[base_sp + s]);
    }
    float lt = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float ls = part_l[base_sp + s];
        if (ls > 0.0f) {
            const float w = __expf(part_m[base_sp + s] - mg);
            lt += ls * w;
            acc = fmaf(part_o[(base_sp + s) * kInkHeadDim + d], w, acc);
        }
    }
    float val = acc / lt;
    const size_t obase = ((size_t)m * kInkHeads + h) * kInkHeadDim + d;
    if (gate_attn) {
        val *= 1.0f / (1.0f + __expf(-gate_attn[obase]));
    }
    out[obase] = val;
}

// Prefill / Large-M Flash-GQA with 3-Head GQA KV Sharing (reads K and V ONCE for all 3 Q heads -> 3x faster prefill!)
// Grid: (8 KV heads, M tokens), Block: 128 threads (4 warps)
__global__ void __launch_bounds__(128) k_ink_flash_gqa_gqa3(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,    // [M]
    int slot0,
    const int32_t* __restrict__ q_pos,    // [M]
    float* __restrict__ out,              // [M, 24, 128]
    int M) {
    const int kh = blockIdx.x;  // 0..7
    const int m  = blockIdx.y;  // 0..M-1
    if (kh >= kInkKVHeads || m >= M) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sl   = slots ? slots[m] : slot0;
    const int qp   = q_pos[m];
    const float scale = 0.08838834764831845f;

    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + h0 + 0) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + h0 + 1) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + h0 + 2) * (kInkRelExtent + 1);

    const int k_start = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const int k_end   = qp + 1;

    float mw[3] = {
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 0] : -1e30f,
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 1] : -1e30f,
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 2] : -1e30f
    };
    float lw[3] = { (warp == 0) ? 1.0f : 0.0f, (warp == 0) ? 1.0f : 0.0f, (warp == 0) ? 1.0f : 0.0f };
    float a[3][4] = {};

    const int k_end_safe = kv.is_swa ? k_end : min(k_end, kv.cap_pos);
    for (int kp = k_start + warp; kp < k_end_safe; kp += 4) {
        const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
        float4 kv4, vv4;
        ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        const int dist = min(max(qp - kp, 0), kInkRelExtent);
        const float sc[3] = {
            warp_sum(qv0.x * kv4.x + qv0.y * kv4.y + qv0.z * kv4.z + qv0.w * kv4.w) * scale + rt0[dist],
            warp_sum(qv1.x * kv4.x + qv1.y * kv4.y + qv1.z * kv4.z + qv1.w * kv4.w) * scale + rt1[dist],
            warp_sum(qv2.x * kv4.x + qv2.y * kv4.y + qv2.z * kv4.z + qv2.w * kv4.w) * scale + rt2[dist]
        };
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const float mn = fmaxf(mw[r], sc[r]);
            const float al = __expf(mw[r] - mn);
            const float p  = __expf(sc[r] - mn);
            lw[r] = lw[r] * al + p;
            a[r][0] = a[r][0] * al + p * vv4.x;
            a[r][1] = a[r][1] * al + p * vv4.y;
            a[r][2] = a[r][2] * al + p * vv4.z;
            a[r][3] = a[r][3] * al + p * vv4.w;
            mw[r] = mn;
        }
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
        sm_a[warp][lane * 4 + 0] = a[r][0]; sm_a[warp][lane * 4 + 1] = a[r][1];
        sm_a[warp][lane * 4 + 2] = a[r][2]; sm_a[warp][lane * 4 + 3] = a[r][3];
        __syncthreads();
        if (warp == 0) {
            float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
            float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
            float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
            float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
            const float inv = 1.0f / (sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3);
            const size_t ob = ((size_t)m * kInkHeads + h0 + r) * kInkHeadDim + lane * 4;
            float o0 = (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv;
            float o1 = (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv;
            float o2 = (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv;
            float o3 = (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv;
            if (gate_attn) {
                const float4 gv = *(const float4*)(gate_attn + ob);
                o0 *= 1.0f / (1.0f + __expf(-gv.x)); o1 *= 1.0f / (1.0f + __expf(-gv.y));
                o2 *= 1.0f / (1.0f + __expf(-gv.z)); o3 *= 1.0f / (1.0f + __expf(-gv.w));
            }
            *(float4*)(out + ob) = make_float4(o0, o1, o2, o3);
        }
        __syncthreads();
    }
}

// ============================================================================
// Tensor Core Prefill Flash-GQA Helpers (cuBLAS BF16 on sm_80+ and FP16 on sm_60..sm_75 -> 6-15x faster prefill!)
// ============================================================================
template <bool USE_FP16>
__device__ __forceinline__ uint16_t pf_cvt_from_f32(float v) {
    if constexpr (USE_FP16) {
        v = fminf(fmaxf(v, -65504.0f), 65504.0f);
        __half h = __float2half_rn(v);
        return *reinterpret_cast<const uint16_t*>(&h);
    } else {
        __nv_bfloat16 b = __float2bfloat16_rn(v);
        return *reinterpret_cast<const uint16_t*>(&b);
    }
}

template <bool USE_FP16 = false>
__global__ void __launch_bounds__(128) k_ink_pf_pack_q(
    const float* __restrict__ q,        // [M, 24, 128]
    void* __restrict__ q_out_v,         // [24, M, 128] BF16 or FP16
    int M) {
    const int h = blockIdx.x;  // 0..23
    const int m = blockIdx.y;  // 0..M-1
    const int d = threadIdx.x; // 0..127
    if (h >= kInkHeads || m >= M) return;
    uint16_t* q_out = reinterpret_cast<uint16_t*>(q_out_v);
    q_out[((size_t)h * M + m) * kInkHeadDim + d] = pf_cvt_from_f32<USE_FP16>(q[((size_t)m * kInkHeads + h) * kInkHeadDim + d]);
}

template <bool USE_FP16 = false>
__global__ void __launch_bounds__(128) k_ink_pf_pack_kv(
    InkKvVisitDesc kv,
    int slot0, int k_start, int K_len,
    void* __restrict__ k_out_v,         // [24, K_len, 128] BF16 or FP16
    void* __restrict__ v_out_v) {       // [24, K_len, 128] BF16 or FP16
    const int kh = blockIdx.x;  // 0..7
    const int ki = blockIdx.y;  // 0..K_len-1
    const int d  = threadIdx.x; // 0..127
    if (kh >= kInkKVHeads || ki >= K_len) return;
    const int kp = k_start + ki;
    const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : min(max(kp, 0), kv.cap_pos - 1);
    const bool use_host = (!kv.is_swa && kv.host_cap > 0 && p_idx >= kv.dev_cap);
    const int stride = use_host ? kv.host_cap : (kv.host_cap > 0 ? kv.dev_cap : kv.cap_pos);
    const int pos_l  = use_host ? (p_idx - kv.dev_cap) : p_idx;
    const size_t hbase = ((size_t)slot0 * stride + pos_l) * kInkKVHeads + kh;
    const size_t base  = hbase * kInkHeadDim + d;
    const __half*  pk16 = use_host ? kv.h_k_f16 : kv.k_f16;
    const __half*  pv16 = use_host ? kv.h_v_f16 : kv.v_f16;
    const uint8_t* pk8  = use_host ? kv.h_k_f8  : kv.k_f8;
    const uint8_t* pv8  = use_host ? kv.h_v_f8  : kv.v_f8;
    const float*   pksc = use_host ? kv.h_k_sc  : kv.k_sc;
    const float*   pvsc = use_host ? kv.h_v_sc  : kv.v_sc;
    uint16_t kb, vb;
    if constexpr (USE_FP16) {
        if (kv.kv_fmt == 0) {
            kb = *reinterpret_cast<const uint16_t*>(&pk16[base]);
            vb = *reinterpret_cast<const uint16_t*>(&pv16[base]);
        } else {
            float kf, vf;
            if (kv.kv_fmt == 1) {
                kf = e4m3_to_f32(pk8[base]) * pksc[hbase];
                vf = e4m3_to_f32(pv8[base]) * pvsc[hbase];
            } else if (kv.kv_fmt == 2) {
                kf = (float)(int8_t)pk8[base] * pksc[hbase];
                vf = (float)(int8_t)pv8[base] * pvsc[hbase];
            } else {
                kf = (float)(int8_t)pk8[base] * pksc[hbase];
                const uint8_t pb = pv8[hbase * (kInkHeadDim / 2) + (d >> 1)];
                const int qv = (int)((d & 1) ? (pb >> 4) : (pb & 0xFu)) - 8;
                vf = (float)qv * pvsc[hbase];
            }
            kb = pf_cvt_from_f32<true>(kf);
            vb = pf_cvt_from_f32<true>(vf);
        }
    } else {
        float kf, vf;
        if (kv.kv_fmt == 0) {
            kf = __half2float(pk16[base]);
            vf = __half2float(pv16[base]);
        } else if (kv.kv_fmt == 1) {
            kf = e4m3_to_f32(pk8[base]) * pksc[hbase];
            vf = e4m3_to_f32(pv8[base]) * pvsc[hbase];
        } else if (kv.kv_fmt == 2) {
            kf = (float)(int8_t)pk8[base] * pksc[hbase];
            vf = (float)(int8_t)pv8[base] * pvsc[hbase];
        } else {
            kf = (float)(int8_t)pk8[base] * pksc[hbase];
            const uint8_t pb = pv8[hbase * (kInkHeadDim / 2) + (d >> 1)];
            const int qv = (int)((d & 1) ? (pb >> 4) : (pb & 0xFu)) - 8;
            vf = (float)qv * pvsc[hbase];
        }
        kb = pf_cvt_from_f32<false>(kf);
        vb = pf_cvt_from_f32<false>(vf);
    }
    uint16_t* k_out = reinterpret_cast<uint16_t*>(k_out_v);
    uint16_t* v_out = reinterpret_cast<uint16_t*>(v_out_v);
    const int h0 = 3 * kh;
    k_out[((size_t)(h0 + 0) * K_len + ki) * kInkHeadDim + d] = kb;
    k_out[((size_t)(h0 + 1) * K_len + ki) * kInkHeadDim + d] = kb;
    k_out[((size_t)(h0 + 2) * K_len + ki) * kInkHeadDim + d] = kb;
    v_out[((size_t)(h0 + 0) * K_len + ki) * kInkHeadDim + d] = vb;
    v_out[((size_t)(h0 + 1) * K_len + ki) * kInkHeadDim + d] = vb;
    v_out[((size_t)(h0 + 2) * K_len + ki) * kInkHeadDim + d] = vb;
}

// 1-Pass Register-Cached RelPos + Causal/SWA Mask + Exact Per-Query Sink Softmax -> BF16/FP16 probs [24, C, K_sub]!
template <bool USE_FP16 = false>
__global__ void __launch_bounds__(256) k_ink_pf_softmax_rel_sink(
    const float* __restrict__ scores,     // [24, C, K_len] F32 (already scaled by 1/sqrt(128))
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    void* __restrict__ probs_v,           // [24, C, K_len] BF16 or FP16
    int q0, int C, int k_start, int K_len, int p0, bool is_swa) {
    const int ci = blockIdx.x;  // 0..C-1
    const int h  = blockIdx.y;  // 0..23
    if (ci >= C || h >= kInkHeads) return;
    const int tid = threadIdx.x;
    const int m = q0 + ci;
    const int qp = p0 + m;
    const float* sc_row = scores + ((size_t)h * C + ci) * K_len;
    const float* rt = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
    const float sk = sinks[(size_t)m * kInkHeads + h];
    uint16_t* pr_row = reinterpret_cast<uint16_t*>(probs_v) + ((size_t)h * C + ci) * K_len;
    __shared__ float sh[32];

    if (K_len <= 2048) {
        float reg_v[8];
        uint8_t reg_ok[8];
        float mx = (tid == 0) ? sk : -1e30f;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int ki = t * 256 + tid;
            if (ki < K_len) {
                const int kp = k_start + ki;
                const int dist = qp - kp;
                const bool ok = (kp <= qp) && (!is_swa || dist < kInkSwaWin);
                reg_ok[t] = ok ? 1 : 0;
                if (ok) {
                    const float v = sc_row[ki] + rt[min(dist, kInkRelExtent)];
                    reg_v[t] = v;
                    mx = fmaxf(mx, v);
                }
            } else {
                reg_ok[t] = 0;
            }
        }
        mx = warp_max(mx);
        if ((tid & 31) == 0) sh[tid >> 5] = mx;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < 8) ? sh[tid] : -1e30f;
            v = warp_max(v);
            if (tid == 0) sh[0] = v;
        }
        __syncthreads();
        const float mg = sh[0];
        __syncthreads();

        float lsum = (tid == 0) ? __expf(sk - mg) : 0.0f;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (reg_ok[t]) {
                const float e = __expf(reg_v[t] - mg);
                reg_v[t] = e;
                lsum += e;
            }
        }
        lsum = block_sum(lsum, sh);
        const float inv_l = 1.0f / lsum;

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int ki = t * 256 + tid;
            if (ki < K_len) {
                pr_row[ki] = pf_cvt_from_f32<USE_FP16>(reg_ok[t] ? (reg_v[t] * inv_l) : 0.0f);
            }
        }
        return;
    }

    // General fallback for K_len > 2048
    float mx = (tid == 0) ? sk : -1e30f;
    for (int ki = tid; ki < K_len; ki += 256) {
        const int kp = k_start + ki;
        const int dist = qp - kp;
        if ((kp <= qp) && (!is_swa || dist < kInkSwaWin)) {
            mx = fmaxf(mx, sc_row[ki] + rt[min(dist, kInkRelExtent)]);
        }
    }
    mx = warp_max(mx);
    if ((tid & 31) == 0) sh[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < 8) ? sh[tid] : -1e30f;
        v = warp_max(v);
        if (tid == 0) sh[0] = v;
    }
    __syncthreads();
    const float mg = sh[0];

    float lsum = (tid == 0) ? __expf(sk - mg) : 0.0f;
    for (int ki = tid; ki < K_len; ki += 256) {
        const int kp = k_start + ki;
        const int dist = qp - kp;
        if ((kp <= qp) && (!is_swa || dist < kInkSwaWin)) {
            lsum += __expf(sc_row[ki] + rt[min(dist, kInkRelExtent)] - mg);
        }
    }
    lsum = block_sum(lsum, sh);
    const float inv_l = 1.0f / lsum;
    for (int ki = tid; ki < K_len; ki += 256) {
        const int kp = k_start + ki;
        const int dist = qp - kp;
        float p = 0.0f;
        if ((kp <= qp) && (!is_swa || dist < kInkSwaWin)) {
            p = __expf(sc_row[ki] + rt[min(dist, kInkRelExtent)] - mg) * inv_l;
        }
        pr_row[ki] = pf_cvt_from_f32<USE_FP16>(p);
    }
}

// Unpack O [24, C, 128] F32 -> out [q0..q0+C-1, 24, 128] F32 with fused * sigmoid(gate_attn)!
__global__ void __launch_bounds__(128) k_ink_pf_unpack_o_gate(
    const float* __restrict__ o_in,       // [24, C, 128]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    float* __restrict__ out,              // [M, 24, 128]
    int q0, int C) {
    const int h  = blockIdx.x;  // 0..23
    const int ci = blockIdx.y;  // 0..C-1
    const int d  = threadIdx.x; // 0..127
    if (h >= kInkHeads || ci >= C) return;
    const size_t ob = ((size_t)(q0 + ci) * kInkHeads + h) * kInkHeadDim + d;
    float v = o_in[((size_t)h * C + ci) * kInkHeadDim + d];
    if (gate_attn) v *= 1.0f / (1.0f + __expf(-gate_attn[ob]));
    out[ob] = v;
}

// ---- O(1)-VRAM KV-Tiled Online-Softmax Tensor Core Prefill Helpers (Fixes #6609 28k OOM & #6605 27k hang!) ----
// Bounded to KV_TILE <= 2048 so prefill workspace is strictly <= ~114 MiB forever (even at 256K..1M context)!
__global__ void __launch_bounds__(128) k_ink_pf_online_init(
    const float* __restrict__ sinks,      // [M, 24]
    float* __restrict__ m_acc,            // [24, C]
    float* __restrict__ l_acc,            // [24, C]
    float* __restrict__ o_acc,            // [24, C, 128]
    int q0, int C) {
    const int h  = blockIdx.x;  // 0..23
    const int ci = blockIdx.y;  // 0..C-1
    const int d  = threadIdx.x; // 0..127
    if (h >= kInkHeads || ci >= C) return;
    if (d == 0) {
        const size_t idx = (size_t)h * C + ci;
        m_acc[idx] = sinks[(size_t)(q0 + ci) * kInkHeads + h];
        l_acc[idx] = 1.0f;
    }
    o_acc[((size_t)h * C + ci) * kInkHeadDim + d] = 0.0f;
}

template <bool USE_FP16 = false>
__global__ void __launch_bounds__(256) k_ink_pf_softmax_tile_online(
    const float* __restrict__ scores,     // [24, C, K_t] F32 (K_t <= 2048)
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    void* __restrict__ probs_v,           // [24, C, K_t] BF16 or FP16 (unnormalized exp(s - m_new))
    float* __restrict__ m_acc,            // [24, C] in/out
    float* __restrict__ l_acc,            // [24, C] in/out
    float* __restrict__ alpha_out,        // [24, C] exp(m_old - m_new)
    int q0, int C, int k_tile_start, int K_t, int p0, bool is_swa) {
    const int ci = blockIdx.x;  // 0..C-1
    const int h  = blockIdx.y;  // 0..23
    if (ci >= C || h >= kInkHeads) return;
    const int tid = threadIdx.x;
    const int m   = q0 + ci;
    const int qp  = p0 + m;
    const size_t st_idx = (size_t)h * C + ci;
    const float* sc_row = scores + st_idx * K_t;
    const float* rt     = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
    uint16_t* pr_row    = reinterpret_cast<uint16_t*>(probs_v) + st_idx * K_t;
    __shared__ float sh[32];

    const float m_old = m_acc[st_idx];
    float reg_v[8];
    uint8_t reg_ok[8];
    float mx = (tid == 0) ? m_old : -1e30f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int ki = t * 256 + tid;
        if (ki < K_t) {
            const int kp = k_tile_start + ki;
            const int dist = qp - kp;
            const bool ok = (kp <= qp) && (!is_swa || dist < kInkSwaWin);
            reg_ok[t] = ok ? 1 : 0;
            if (ok) {
                const float v = sc_row[ki] + rt[min(dist, kInkRelExtent)];
                reg_v[t] = v;
                mx = fmaxf(mx, v);
            }
        } else {
            reg_ok[t] = 0;
        }
    }
    mx = warp_max(mx);
    if ((tid & 31) == 0) sh[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < 8) ? sh[tid] : -1e30f;
        v = warp_max(v);
        if (tid == 0) sh[0] = v;
    }
    __syncthreads();
    const float m_new = sh[0];
    __syncthreads();

    const float alpha = __expf(m_old - m_new);
    float tile_l = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int ki = t * 256 + tid;
        if (ki < K_t) {
            const float e = reg_ok[t] ? __expf(reg_v[t] - m_new) : 0.0f;
            tile_l += e;
            pr_row[ki] = pf_cvt_from_f32<USE_FP16>(e);
        }
    }
    tile_l = block_sum(tile_l, sh);
    if (tid == 0) {
        m_acc[st_idx]     = m_new;
        l_acc[st_idx]     = l_acc[st_idx] * alpha + tile_l;
        alpha_out[st_idx] = alpha;
    }
}

__global__ void __launch_bounds__(128) k_ink_pf_accum_tile(
    const float* __restrict__ o_tile,     // [24, C, 128]
    const float* __restrict__ alpha_in,   // [24, C]
    const float* __restrict__ l_acc,      // [24, C]
    float* __restrict__ o_acc,            // [24, C, 128]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    float* __restrict__ out,              // [M, 24, 128] (written on is_last!)
    int q0, int C, bool is_last) {
    const int h  = blockIdx.x;  // 0..23
    const int ci = blockIdx.y;  // 0..C-1
    const int d  = threadIdx.x; // 0..127
    if (h >= kInkHeads || ci >= C) return;
    const size_t st_idx = (size_t)h * C + ci;
    const size_t oi = st_idx * kInkHeadDim + d;
    const float v = o_acc[oi] * alpha_in[st_idx] + o_tile[oi];
    if (!is_last) {
        o_acc[oi] = v;
    } else {
        float res = v / l_acc[st_idx];
        const size_t ob = ((size_t)(q0 + ci) * kInkHeads + h) * kInkHeadDim + d;
        if (gate_attn) res *= 1.0f / (1.0f + __expf(-gate_attn[ob]));
        out[ob] = res;
    }
}

}  // namespace strata::ink
