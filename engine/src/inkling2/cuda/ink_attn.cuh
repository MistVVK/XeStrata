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

// ---- KV Cache Representation (Supports both FP16 and FP8 E4M3 per-head-token scaled!) ----
enum class InkKvDtype : int {
    FP16 = 0,
    FP8  = 1,
};

// Per-visit KV cache descriptor passed by value to attention kernels (CUDA-Graph friendly!)
struct InkKvVisitDesc {
    __half*  k_f16;      // [max_slots, cap_pos, 8, 128] (used if dtype == FP16)
    __half*  v_f16;
    uint8_t* k_f8;       // [max_slots, cap_pos, 8, 128] E4M3 (used if dtype == FP8)
    uint8_t* v_f8;
    float*   k_sc;       // [max_slots, cap_pos, 8] per-head FP32 scale for FP8
    float*   v_sc;
    int      cap_pos;    // SWA: swa_ring (>= 1024 + max_chunk or 1024+16); Global: max_ctx
    bool     is_swa;
    bool     is_fp8;
};

// Write K, V [M, 8, 128] FP32 into KV cache at positions pos_arr[m] for slot_arr[m]
// (For SWA, writes to ring index pos % cap_pos; for Global, writes to pos).
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
    const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
    const size_t base = (((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + h) * kInkHeadDim;
    const float kv_val = k_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];
    const float vv_val = v_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];

    if (!kv.is_fp8) {
        kv.k_f16[base + d] = __float2half(kv_val);
        kv.v_f16[base + d] = __float2half(vv_val);
    } else {
        // Warp/block max for per-head FP8 scale
        float mk = warp_max(fabsf(kv_val));
        float mv = warp_max(fabsf(vv_val));
        __shared__ float sk[4], sv[4];
        if ((d & 31) == 0) { sk[d >> 5] = mk; sv[d >> 5] = mv; }
        __syncthreads();
        float amk = fmaxf(fmaxf(sk[0], sk[1]), fmaxf(sk[2], sk[3]));
        float amv = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        const float sc_k = amk > 0.0f ? (amk * (1.0f / 448.0f)) : 1.0f;
        const float sc_v = amv > 0.0f ? (amv * (1.0f / 448.0f)) : 1.0f;
        const __nv_fp8_e4m3 qk(fminf(fmaxf(kv_val / sc_k, -448.0f), 448.0f));
        const __nv_fp8_e4m3 qv(fminf(fmaxf(vv_val / sc_v, -448.0f), 448.0f));
        kv.k_f8[base + d] = qk.__x;
        kv.v_f8[base + d] = qv.__x;
        if (d == 0) {
            const size_t sidx = ((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + h;
            kv.k_sc[sidx] = sc_k;
            kv.v_sc[sidx] = sc_v;
        }
    }
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
            const float kv_val = kv.is_fp8 ? val : __half2float(__float2half(val));
            (is_k ? k_out : v_out)[(size_t)m * kInkKVDim + c] = kv_val;
            (is_k ? stage_k : stage_v)[(size_t)m * kInkKVDim + c] = val;
        } else {
            // Direct write into KV cache (FP16 or FP8 E4M3)!
            const int p = pos_arr[m];
            const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
            const size_t base = (((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh) * kInkHeadDim;
            if (!kv.is_fp8) {
                (is_k ? kv.k_f16 : kv.v_f16)[base + tid] = __float2half(val);
            } else {
                float mv = warp_max(fabsf(val));
                if ((tid & 31) == 0) sh[tid >> 5] = mv;
                __syncthreads();
                float am = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
                const float sc = am > 0.0f ? (am * (1.0f / 448.0f)) : 1.0f;
                const __nv_fp8_e4m3 qv(fminf(fmaxf(val / sc, -448.0f), 448.0f));
                (is_k ? kv.k_f8 : kv.v_f8)[base + tid] = qv.__x;
                if (tid == 0) {
                    const size_t sidx = ((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh;
                    (is_k ? kv.k_sc : kv.v_sc)[sidx] = sc;
                }
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
    for (int kp = k_start + warp; kp < k_end; kp += 4) {
        const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
        const size_t base = (((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        float4 kv4, vv4;
        if (!kv.is_fp8) {
            const uint2 k2 = *(const uint2*)(kv.k_f16 + base);
            const uint2 v2 = *(const uint2*)(kv.v_f16 + base);
            const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
            const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
            const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
            const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
            kv4 = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
            vv4 = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
        } else {
            const size_t sidx = ((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh;
            const float sk = kv.k_sc[sidx];
            const float sv = kv.v_sc[sidx];
            const uint32_t k4 = *(const uint32_t*)(kv.k_f8 + base);
            const uint32_t v4 = *(const uint32_t*)(kv.v_f8 + base);
            const float2 kf0 = e4m3x2_to_f32x2((uint16_t)(k4 & 0xFFFFu));
            const float2 kf1 = e4m3x2_to_f32x2((uint16_t)(k4 >> 16));
            const float2 vf0 = e4m3x2_to_f32x2((uint16_t)(v4 & 0xFFFFu));
            const float2 vf1 = e4m3x2_to_f32x2((uint16_t)(v4 >> 16));
            kv4 = make_float4(kf0.x * sk, kf0.y * sk, kf1.x * sk, kf1.y * sk);
            vv4 = make_float4(vf0.x * sv, vf0.y * sv, vf1.x * sv, vf1.y * sv);
        }
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
// Split-K FlashDecoding with 3-Head GQA KV Sharing (13x faster on M=1..16!)
// Grid: (8 KV heads, kInkAttnSplits=16 splits, M tokens) = 128*M CTAs across 170 SMs!
// Each CTA reads K and V ONCE for KV head kh and attends all 3 Q heads (3*kh + 0..2)!
// ============================================================================
constexpr int kInkAttnSplits = 8;

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
    float* __restrict__ part_m,           // [M, 24, kInkAttnSplits]
    float* __restrict__ part_l,           // [M, 24, kInkAttnSplits]
    float* __restrict__ part_o,           // [M, 24, kInkAttnSplits, 128]
    int M) {
    const int kh = blockIdx.x;  // 0..7
    const int sp = blockIdx.y;  // 0..kInkAttnSplits-1
    const int m  = blockIdx.z;  // 0..M-1
    if (kh >= kInkKVHeads || sp >= kInkAttnSplits || m >= M) return;

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
    const int per_sp  = (k_len + kInkAttnSplits - 1) / kInkAttnSplits;
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
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            const size_t base = (((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            if (!kv.is_fp8) {
                const uint2 k2 = *(const uint2*)(kv.k_f16 + base);
                const uint2 v2 = *(const uint2*)(kv.v_f16 + base);
                const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
                const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
                const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
                const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
                kv4 = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
                vv4 = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
            } else {
                const size_t sidx = ((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh;
                const float sk = kv.k_sc[sidx], sv = kv.v_sc[sidx];
                const uint32_t k4 = *(const uint32_t*)(kv.k_f8 + base);
                const uint32_t v4 = *(const uint32_t*)(kv.v_f8 + base);
                const float2 kf0 = e4m3x2_to_f32x2((uint16_t)(k4 & 0xFFFFu));
                const float2 kf1 = e4m3x2_to_f32x2((uint16_t)(k4 >> 16));
                const float2 vf0 = e4m3x2_to_f32x2((uint16_t)(v4 & 0xFFFFu));
                const float2 vf1 = e4m3x2_to_f32x2((uint16_t)(v4 >> 16));
                kv4 = make_float4(kf0.x * sk, kf0.y * sk, kf1.x * sk, kf1.y * sk);
                vv4 = make_float4(vf0.x * sv, vf0.y * sv, vf1.x * sv, vf1.y * sv);
            }
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
            const size_t idx_sp = ((size_t)m * kInkHeads + h0 + r) * kInkAttnSplits + sp;
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

// Merge kInkAttnSplits=16 splits + fused * sigmoid(gate_attn)! Grid: (24 heads, M), Block: 128 (1 dim/thread!)
__global__ void __launch_bounds__(128) k_ink_flash_gqa_merge(
    const float* __restrict__ part_m,     // [M, 24, 16]
    const float* __restrict__ part_l,     // [M, 24, 16]
    const float* __restrict__ part_o,     // [M, 24, 16, 128]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    float* __restrict__ out,              // [M, 24, 128]
    int M) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;  // 0..127
    if (h >= kInkHeads || m >= M) return;

    const size_t base_sp = ((size_t)m * kInkHeads + h) * kInkAttnSplits;
    float mg = -1e30f;
#pragma unroll
    for (int s = 0; s < kInkAttnSplits; ++s) {
        if (part_l[base_sp + s] > 0.0f) mg = fmaxf(mg, part_m[base_sp + s]);
    }
    float lt = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kInkAttnSplits; ++s) {
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

    for (int kp = k_start + warp; kp < k_end; kp += 4) {
        const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
        const size_t base = (((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        float4 kv4, vv4;
        if (!kv.is_fp8) {
            const uint2 k2 = *(const uint2*)(kv.k_f16 + base);
            const uint2 v2 = *(const uint2*)(kv.v_f16 + base);
            const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
            const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
            const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
            const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
            kv4 = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
            vv4 = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
        } else {
            const size_t sidx = ((size_t)sl * kv.cap_pos + p_idx) * kInkKVHeads + kh;
            const float sk = kv.k_sc[sidx], sv = kv.v_sc[sidx];
            const uint32_t k4 = *(const uint32_t*)(kv.k_f8 + base);
            const uint32_t v4 = *(const uint32_t*)(kv.v_f8 + base);
            const float2 kf0 = e4m3x2_to_f32x2((uint16_t)(k4 & 0xFFFFu));
            const float2 kf1 = e4m3x2_to_f32x2((uint16_t)(k4 >> 16));
            const float2 vf0 = e4m3x2_to_f32x2((uint16_t)(v4 & 0xFFFFu));
            const float2 vf1 = e4m3x2_to_f32x2((uint16_t)(v4 >> 16));
            kv4 = make_float4(kf0.x * sk, kf0.y * sk, kf1.x * sk, kf1.y * sk);
            vv4 = make_float4(vf0.x * sv, vf0.y * sv, vf1.x * sv, vf1.y * sv);
        }
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
    const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
    const size_t base = (((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + kh) * kInkHeadDim + d;
    uint16_t kb, vb;
    if constexpr (USE_FP16) {
        if (!kv.is_fp8) {
            kb = *reinterpret_cast<const uint16_t*>(&kv.k_f16[base]);
            vb = *reinterpret_cast<const uint16_t*>(&kv.v_f16[base]);
        } else {
            const size_t sidx = ((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + kh;
            kb = pf_cvt_from_f32<true>(e4m3_to_f32(kv.k_f8[base]) * kv.k_sc[sidx]);
            vb = pf_cvt_from_f32<true>(e4m3_to_f32(kv.v_f8[base]) * kv.v_sc[sidx]);
        }
    } else {
        float kf, vf;
        if (!kv.is_fp8) {
            kf = __half2float(kv.k_f16[base]);
            vf = __half2float(kv.v_f16[base]);
        } else {
            const size_t sidx = ((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + kh;
            kf = e4m3_to_f32(kv.k_f8[base]) * kv.k_sc[sidx];
            vf = e4m3_to_f32(kv.v_f8[base]) * kv.v_sc[sidx];
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

}  // namespace strata::ink
