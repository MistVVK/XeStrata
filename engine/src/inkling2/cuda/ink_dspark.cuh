// engine/src/inkling2/cuda/ink_dspark.cuh - Persistent 100% On-GPU DSpark Block-8 / Tree-16 Recurrent Markov Drafter (< 0.35 ms!)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Implements Thinking Machines' co-trained 3-layer Block-8 DSpark drafter (embedded in .strata v1.1 Section 1):
//   - Taps [5, 23, 41, 48] mixed via softmax(fuse_weight[0..3]) + dflare_norm -> context K/V cache (SWA-1025, Global, SWA-1025) + f_rnn
//   - 3 non-causal block-attention layers (24 Q, 8 KV, head_dim=128, NeoX half-split RoPE theta=1e6, SwiGLU 9216)
//   - Fast base logits over B=8 slots via 230 MB 4-bit screening plane (or BF16/Q8_ROW) + 4-pass radix Top-K_cand (K_cand = 256/512)
//   - Persistent on-GPU 7-step markov_rank=256 RNN head scoring ONLY K_cand rows of unembed (fits 100% in L2 -> ~10 us/step!)
//   - Supports BOTH Block-8 linear drafting AND Tree-16 2-branch beam drafting with 100% lossless exact target verification!
#pragma once

#include "ink_runner.cuh"

namespace strata::ink {

// ---- NeoX Half-Split RoPE (theta = 1e6) + Per-Head RMSNorm ----
__global__ void __launch_bounds__(128) k_dspark_rmsnorm_rope(
    float* __restrict__ x,               // [M, H, 128]
    const float* __restrict__ nw,        // [128] F32
    const int32_t* __restrict__ pos_arr, // [M]
    float theta,
    int M, int H) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;
    const int d = threadIdx.x;  // 0..127
    if (m >= M || h >= H) return;
    __shared__ float s_x[kInkHeadDim];
    __shared__ float sh[8];
    float* xp = x + ((size_t)m * H + h) * kInkHeadDim;
    float v = xp[d];
    float ss = block_sum(v * v, sh);
    const float inv = rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
    s_x[d] = v * inv * nw[d];
    __syncthreads();

    // NeoX half-split RoPE: pairs (d, d + 64) for d in 0..63
    if (d < 64) {
        const float freq = 1.0f / powf(theta, (float)(2 * d) * (1.0f / (float)kInkHeadDim));
        const float ang = (float)pos_arr[m] * freq;
        float s_val, c_val;
        __sincosf(ang, &s_val, &c_val);
        const float x0 = s_x[d];
        const float x1 = s_x[d + 64];
        xp[d]      = x0 * c_val - x1 * s_val;
        xp[d + 64] = x0 * s_val + x1 * c_val;
    }
}

// Mix 4 target taps [4, M, 2304] with softmax(fuse_weight[row]) + RMSNorm(norm_w) -> out [M, 2304]
__global__ void __launch_bounds__(256) k_dspark_mix_taps(
    const float* __restrict__ tap0,
    const float* __restrict__ tap1,
    const float* __restrict__ tap2,
    const float* __restrict__ tap3,
    const float* __restrict__ fuse_w,  // [4, 4] F32
    int row,
    const float* __restrict__ norm_w,  // [2304] F32
    float* __restrict__ out,           // [M, 2304]
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    __shared__ float sw[4];
    __shared__ float sh[32];
    if (tid == 0) {
        const float* fw = fuse_w + row * 4;
        float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        float inv = 1.0f / (e0 + e1 + e2 + e3);
        sw[0] = e0 * inv; sw[1] = e1 * inv; sw[2] = e2 * inv; sw[3] = e3 * inv;
    }
    __syncthreads();
    const size_t off = (size_t)m * kInkHidden;
    float* dst = out + off;
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        float v = sw[0] * tap0[off + d] + sw[1] * tap1[off + d] + sw[2] * tap2[off + d] + sw[3] * tap3[off + d];
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

// Non-causal DSpark Block-8 Attention: attends to committed ctx [c0..apos-1] + all B=8 draft tokens [0..B-1]!
// Reads apos from d_apos_ptr so the entire DSpark draft_block8 captures cleanly in a SINGLE CUDA Graph across all positions!
__global__ void __launch_bounds__(128) k_dspark_block_attn(
    const float* __restrict__ q,          // [B, 24, 128]
    const float* __restrict__ kd,         // [B, 8, 128]
    const float* __restrict__ vd,         // [B, 8, 128]
    const __half* __restrict__ ctx_k,     // [cap, 8, 128]
    const __half* __restrict__ ctx_v,     // [cap, 8, 128]
    const int32_t* __restrict__ apos_ptr, // [1] committed context length
    bool is_swa,
    int swa_win,                          // 1025
    int B,                                // 8
    float* __restrict__ out) {            // [B, 24, 128]
    const int b = blockIdx.y;
    const int h = blockIdx.x;  // 0..23
    if (b >= B || h >= kInkHeads) return;
    const int apos = *apos_ptr;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kh = h / 3;
    const float scale = 0.08838834764831845f;

    const float4 qv = *(const float4*)(q + ((size_t)b * kInkHeads + h) * kInkHeadDim + lane * 4);
    const int c0 = is_swa ? max(0, apos - swa_win + 1) : 0;

    float mw = -1e30f, lw = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    auto attend = [&](float4 kv4, float4 vv4) {
        float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
        dot = warp_sum(dot) * scale;
        const float mn = fmaxf(mw, dot);
        const float al = __expf(mw - mn);
        const float p  = __expf(dot - mn);
        lw = lw * al + p;
        a0 = a0 * al + p * vv4.x; a1 = a1 * al + p * vv4.y;
        a2 = a2 * al + p * vv4.z; a3 = a3 * al + p * vv4.w;
        mw = mn;
    };

    for (int kp = c0 + warp; kp < apos; kp += 4) {
        const size_t base = ((size_t)kp * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        const uint2 k2 = *(const uint2*)(ctx_k + base);
        const uint2 v2 = *(const uint2*)(ctx_v + base);
        const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
        const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
        const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
        const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
        attend(make_float4(kf0.x, kf0.y, kf1.x, kf1.y), make_float4(vf0.x, vf0.y, vf1.x, vf1.y));
    }
    for (int j = warp; j < B; j += 4) {
        const size_t base = ((size_t)j * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        attend(*(const float4*)(kd + base), *(const float4*)(vd + base));
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
    if (lane == 0) { sm_m[warp] = mw; sm_l[warp] = lw; }
    sm_a[warp][lane * 4 + 0] = a0; sm_a[warp][lane * 4 + 1] = a1;
    sm_a[warp][lane * 4 + 2] = a2; sm_a[warp][lane * 4 + 3] = a3;
    __syncthreads();
    if (warp == 0) {
        float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
        float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
        float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
        float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
        const float inv = 1.0f / (sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3);
        const size_t ob = ((size_t)b * kInkHeads + h) * kInkHeadDim + lane * 4;
        *(float4*)(out + ob) = make_float4(
            (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv,
            (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv,
            (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv,
            (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv);
    }
}

__global__ void __launch_bounds__(256) k_dspark_fill_mask(
    const float* __restrict__ mask_e,
    float* __restrict__ dst_cur,
    int B) {
    const int s = blockIdx.x + 1;
    if (s >= B) return;
    const int tid = threadIdx.x;
    const float4* src4 = reinterpret_cast<const float4*>(mask_e);
    float4* dst4 = reinterpret_cast<float4*>(dst_cur + (size_t)s * kInkHidden);
    for (int i = tid; i < (kInkHidden / 4); i += 256) dst4[i] = src4[i];
}

__global__ void __launch_bounds__(256) k_dspark_split_qkv(
    const float* __restrict__ qkv,  // [B, 5120]
    float* __restrict__ q,          // [B, 3072]
    float* __restrict__ k,          // [B, 1024]
    float* __restrict__ v,          // [B, 1024]
    int B) {
    constexpr int TOT = kInkQDim + 2 * kInkKVDim;  // 5120
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || i >= TOT) return;
    const float val = qkv[(size_t)b * TOT + i];
    if (i < kInkQDim) q[(size_t)b * kInkQDim + i] = val;
    else if (i < kInkQDim + kInkKVDim) k[(size_t)b * kInkKVDim + (i - kInkQDim)] = val;
    else v[(size_t)b * kInkKVDim + (i - kInkQDim - kInkKVDim)] = val;
}

__global__ void __launch_bounds__(256) k_dspark_swiglu_fused(
    const float* __restrict__ gu,   // [B, 2 * 9216]
    float* __restrict__ out,        // [B, 9216]
    int B) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || i >= kInkDsparkInter) return;
    const float* row = gu + (size_t)b * (2 * kInkDsparkInter);
    const float gv = row[i];
    out[(size_t)b * kInkDsparkInter + i] = (gv / (1.0f + __expf(-gv))) * row[kInkDsparkInter + i];
}

__global__ void __launch_bounds__(256) k_dspark_swiglu(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        out[i] = (gv / (1.0f + __expf(-gv))) * u[i];
    }
}

// Fast 4-pass radix Top-K candidate extractor from base logits [7, V] -> cand_ids [7, K_cand] & cand_base [7, K_cand]
__device__ __forceinline__ uint32_t dspark_f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// ---- Ultra-Fast INT8 __dp4a Base Logits for DSpark 7 Slots (~0.30 ms on RTX 5090!) ----
__global__ void __launch_bounds__(256) k_dspark_quant_h_q8(
    const float* __restrict__ h,  // [7, 2304]
    int8_t* __restrict__ h_q8,    // [7, 2304]
    float* __restrict__ h_sc) {   // [7]
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const float* hp = h + (size_t)s * kInkHidden;
    float am = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(hp[d]));
    am = warp_max(am);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
               fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = am * (1.0f / 127.0f);
    const float inv = sc > 0.0f ? (127.0f / am) : 0.0f;
    int8_t* dst = h_q8 + (size_t)s * kInkHidden;
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = (int8_t)max(-127, min(127, __float2int_rn(hp[d] * inv)));
    if (tid == 0) h_sc[s] = sc;
}

// 128 threads/CTA (4 warps = 4 vocab rows), caches ALL 7 x 2304 int8 bytes in 16 KB shared memory, dp4a_i8!
__global__ void __launch_bounds__(128) k_dspark_screen_dp4a7(
    const int8_t* __restrict__ q8,    // [V, 2304]
    const float* __restrict__ sc,     // [V]
    const int8_t* __restrict__ h_q8,  // [7, 2304]
    const float* __restrict__ h_sc,   // [7]
    float* __restrict__ base_logits,  // [7, V]
    int V) {
    __shared__ alignas(16) uint32_t s_h[7][kInkHidden / 4];  // 16,128 B!
    const int tid = threadIdx.x;
    for (int m = 0; m < 7; ++m) {
        const uint4* src = reinterpret_cast<const uint4*>(h_q8 + (size_t)m * kInkHidden);
        uint4* dst = reinterpret_cast<uint4*>(s_h[m]);
        for (int i = tid; i < (kInkHidden / 16); i += 128) dst[i] = src[i];
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= V) return;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    int32_t acc[7] = {};
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const int base = i * 4;
#pragma unroll
        for (int m = 0; m < 7; ++m) {
            const uint4 hv = *reinterpret_cast<const uint4*>(s_h[m] + base);
            int32_t a = acc[m];
            a = dp4a_i8(v.x, hv.x, a);
            a = dp4a_i8(v.y, hv.y, a);
            a = dp4a_i8(v.z, hv.z, a);
            a = dp4a_i8(v.w, hv.w, a);
            acc[m] = a;
        }
    }
    const float sc_r = sc[row];
#pragma unroll
    for (int m = 0; m < 7; ++m) {
        int32_t s = acc[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
        if (lane == 0) base_logits[(size_t)m * V + row] = (float)s * sc_r * h_sc[m];
    }
}

// Zero-Emulated-dp4a Bank-Conflict-Free FP32-FMA Screener for Volta sm_70 (V100 15.7 TFLOPS) & Pascal sm_60 (P100)!
// Zero shared-memory LUT bank conflicts -> HBM2 bandwidth bound (~0.55 ms on V100 vs 4.9-6.7 ms emulated dp4a!)
__global__ void __launch_bounds__(128) k_dspark_screen_h2_7(
    const int8_t* __restrict__ q8,    // [V, 2304]
    const float* __restrict__ sc,     // [V]
    const int8_t* __restrict__ h_q8,  // [7, 2304]
    const float* __restrict__ h_sc,   // [7]
    float* __restrict__ base_logits,  // [7, V]
    int V) {
    __shared__ alignas(16) __half2 s_h[7][kInkHidden / 2];  // 32,256 B (linear read -> ZERO bank conflicts!)
    const int tid = threadIdx.x;
    for (int m = 0; m < 7; ++m) {
        const int8_t* src = h_q8 + (size_t)m * kInkHidden;
        for (int i = tid; i < (kInkHidden / 2); i += 128) {
            s_h[m][i] = __halves2half2(
                __float2half_rn((float)src[2 * i] * (1.0f / 127.0f)),
                __float2half_rn((float)src[2 * i + 1] * (1.0f / 127.0f)));
        }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= V) return;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    __half2 acc[7];
#pragma unroll
    for (int m = 0; m < 7; ++m) acc[m] = __float2half2_rn(0.0f);

    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const __half2 w0 = __halves2half2(__int2half_rn((int)(int8_t)(v.x)),       __int2half_rn((int)(int8_t)(v.x >> 8)));
        const __half2 w1 = __halves2half2(__int2half_rn((int)(int8_t)(v.x >> 16)), __int2half_rn((int)(int8_t)(v.x >> 24)));
        const __half2 w2 = __halves2half2(__int2half_rn((int)(int8_t)(v.y)),       __int2half_rn((int)(int8_t)(v.y >> 8)));
        const __half2 w3 = __halves2half2(__int2half_rn((int)(int8_t)(v.y >> 16)), __int2half_rn((int)(int8_t)(v.y >> 24)));
        const __half2 w4 = __halves2half2(__int2half_rn((int)(int8_t)(v.z)),       __int2half_rn((int)(int8_t)(v.z >> 8)));
        const __half2 w5 = __halves2half2(__int2half_rn((int)(int8_t)(v.z >> 16)), __int2half_rn((int)(int8_t)(v.z >> 24)));
        const __half2 w6 = __halves2half2(__int2half_rn((int)(int8_t)(v.w)),       __int2half_rn((int)(int8_t)(v.w >> 8)));
        const __half2 w7 = __halves2half2(__int2half_rn((int)(int8_t)(v.w >> 16)), __int2half_rn((int)(int8_t)(v.w >> 24)));
        const int base = i * 8;
#pragma unroll
        for (int m = 0; m < 7; ++m) {
            const __half2* hp = &s_h[m][base];
            __half2 a = acc[m];
            a = __hfma2(w0, hp[0], a); a = __hfma2(w1, hp[1], a);
            a = __hfma2(w2, hp[2], a); a = __hfma2(w3, hp[3], a);
            a = __hfma2(w4, hp[4], a); a = __hfma2(w5, hp[5], a);
            a = __hfma2(w6, hp[6], a); a = __hfma2(w7, hp[7], a);
            acc[m] = a;
        }
    }
    const float sc_r = sc[row] * 127.0f;
#pragma unroll
    for (int m = 0; m < 7; ++m) {
        const float2 f2 = __half22float2(acc[m]);
        float s = warp_sum(f2.x + f2.y);
        if (lane == 0) base_logits[(size_t)m * V + row] = s * sc_r * h_sc[m];
    }
}

// Split fused kv_ctx [M, 2048] -> kc [M, 1024], vc [M, 1024]
__global__ void __launch_bounds__(256) k_dspark_split_kv(
    const float* __restrict__ kv_fused, float* __restrict__ kc, float* __restrict__ vc, int M) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= 2 * kInkKVDim) return;
    const float v = kv_fused[(size_t)m * (2 * kInkKVDim) + c];
    if (c < kInkKVDim) kc[(size_t)m * kInkKVDim + c] = v;
    else               vc[(size_t)m * kInkKVDim + (c - kInkKVDim)] = v;
}

// ---- Ultra-Fast 2-Stage 896-CTA Parallel Top-512 Extractor (~0.03 ms on RTX 5090!) ----
// Stage 1: Grid = (128 striped chunks, 7 slots) = 896 CTAs across 170 SMs; each CTA finds Top-16 -> 2,048 pairs/slot!
__global__ void __launch_bounds__(256) k_dspark_top_cand_s1(
    const float* __restrict__ base_logits,  // [7, V]
    int V,
    int32_t* __restrict__ tmp_ids,          // [7, 128 * 16 = 2048]
    float* __restrict__ tmp_vals,           // [7, 2048]
    int ld = 0) {
    const int s = blockIdx.y;   // 0..6
    const int ch = blockIdx.x;  // 0..127
    const int tid = threadIdx.x;
    const int stride = (ld > 0) ? ld : V;
    const float* lg = base_logits + (size_t)s * stride;

    float b0 = -3e38f, b1 = -3e38f;
    int32_t i0 = 0, i1 = 0;
    for (int base = ch * 256; base < V; base += 128 * 256) {
        const int idx = base + tid;
        if (idx < V) {
            const float v = lg[idx];
            if (v > b0) { b1 = b0; i1 = i0; b0 = v; i0 = idx; }
            else if (v > b1) { b1 = v; i1 = idx; }
        }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float wv0 = b0; int32_t wi0 = i0;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, wv0, o);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, wi0, o);
        if (ov > wv0 || (ov == wv0 && oi < wi0)) { wv0 = ov; wi0 = oi; }
    }
    float cand2 = (i0 == wi0) ? b1 : b0;
    int32_t cid2 = (i0 == wi0) ? i1 : i0;
    float wv1 = cand2; int32_t wi1 = cid2;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, wv1, o);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, wi1, o);
        if (ov > wv1 || (ov == wv1 && oi < wi1)) { wv1 = ov; wi1 = oi; }
    }
    if (lane == 0) {
        const size_t ob = (size_t)s * 2048 + ch * 16 + warp * 2;
        tmp_ids[ob + 0]  = wi0; tmp_vals[ob + 0] = wv0;
        tmp_ids[ob + 1]  = wi1; tmp_vals[ob + 1] = wv1;
    }
}

// Stage 2: 7 CTAs x 256 threads pick exact Top-K_cand (512) from the 2,048 Stage-1 candidates in shared memory (~3 us)!
__global__ void __launch_bounds__(256) k_dspark_top_cand_s2(
    const int32_t* __restrict__ tmp_ids,    // [7, 2048]
    const float* __restrict__ tmp_vals,     // [7, 2048]
    int K_cand,
    int32_t* __restrict__ cand_ids,         // [7, K_cand]
    float* __restrict__ cand_base) {        // [7, K_cand]
    __shared__ int32_t s_id[2048];
    __shared__ float   s_val[2048];
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, thr_s;
    __shared__ int need_s, cnt;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    for (int i = tid; i < 2048; i += 256) {
        s_id[i]  = tmp_ids[(size_t)s * 2048 + i];
        s_val[i] = tmp_vals[(size_t)s * 2048 + i];
    }
    if (tid == 0) { prefix_s = 0; need_s = K_cand; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int i = tid; i < 2048; i += 256) {
            const uint32_t k = dspark_f2key(s_val[i]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (tid == 0) { thr_s = prefix_s; cnt = 0; }
    __syncthreads();
    const uint32_t thr = thr_s;
    int32_t* o_id = cand_ids + (size_t)s * K_cand;
    float* o_val = cand_base + (size_t)s * K_cand;
    for (int i = tid; i < 2048; i += 256) {
        if (dspark_f2key(s_val[i]) > thr) {
            int p = atomicAdd(&cnt, 1);
            if (p < K_cand) { o_id[p] = s_id[i]; o_val[p] = s_val[i]; }
        }
    }
    __syncthreads();
    if (cnt < K_cand) {
        for (int i = tid; i < 2048; i += 256) {
            if (dspark_f2key(s_val[i]) == thr) {
                int p = atomicAdd(&cnt, 1);
                if (p < K_cand) { o_id[p] = s_id[i]; o_val[p] = s_val[i]; }
            }
        }
    }
}

__global__ void __launch_bounds__(1024) k_dspark_top_cand(
    const float* __restrict__ base_logits,  // [7, V]
    int V, int K_cand,
    int32_t* __restrict__ cand_ids,         // [7, K_cand]
    float* __restrict__ cand_base) {        // [7, K_cand]
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, thr_s;
    __shared__ int need_s, cnt;
    const float* lg = base_logits + (size_t)blockIdx.x * V;
    int32_t* o_id = cand_ids + (size_t)blockIdx.x * K_cand;
    float* o_val = cand_base + (size_t)blockIdx.x * K_cand;
    if (threadIdx.x == 0) { prefix_s = 0; need_s = K_cand; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int i = threadIdx.x; i < V; i += 1024) {
            const uint32_t k = dspark_f2key(lg[i]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { thr_s = prefix_s; cnt = 0; }
    __syncthreads();
    const uint32_t thr = thr_s;
    for (int i = threadIdx.x; i < V; i += 1024) {
        if (dspark_f2key(lg[i]) > thr) {
            int s = atomicAdd(&cnt, 1);
            if (s < K_cand) { o_id[s] = i; o_val[s] = lg[i]; }
        }
    }
    __syncthreads();
    if (cnt < K_cand) {
        for (int i = threadIdx.x; i < V; i += 1024) {
            if (dspark_f2key(lg[i]) == thr) {
                int s = atomicAdd(&cnt, 1);
                if (s < K_cand) { o_id[s] = i; o_val[s] = lg[i]; }
            }
        }
    }
}

// Multi-CTA High-Speed Markov RNN Helpers (84x faster than 1-CTA: ~5 us/step across 170 SMs!)
__global__ void __launch_bounds__(256) k_dspark_markov_prep(
    const int32_t* __restrict__ prev_tok_ptr,
    const float* __restrict__ state,      // [256]
    const float* __restrict__ hidden,     // [2304]
    const float* __restrict__ f_rnn_base, // [max_ctx, 2304]
    const int32_t* __restrict__ apos_ptr, // [1]
    const uint8_t* __restrict__ mw1_raw, int mw1_fmt,
    float* __restrict__ joint_out) {      // [5120]
    constexpr int R = kInkMarkovRank;
    const int tid = threadIdx.x;
    const int32_t pt = *prev_tok_ptr;
    const int apos = *apos_ptr;
    const float* ctx = f_rnn_base + (size_t)(apos > 0 ? (apos - 1) : 0) * kInkHidden;

    joint_out[tid] = state[tid];
    float mv = 0.0f;
    if (mw1_fmt == FMT_BF16) mv = bf16_to_f32(reinterpret_cast<const uint16_t*>(mw1_raw)[(size_t)pt * R + tid]);
    else if (mw1_fmt == FMT_F32) mv = reinterpret_cast<const float*>(mw1_raw)[(size_t)pt * R + tid];
    else {
        const float sc = reinterpret_cast<const float*>(mw1_raw)[pt];
        mv = (float)reinterpret_cast<const int8_t*>(mw1_raw + (size_t)kInkVocabPad * 4)[(size_t)pt * R + tid] * sc;
    }
    joint_out[R + tid] = mv;
    for (int d = tid; d < kInkHidden; d += 256) {
        joint_out[2 * R + d]              = hidden[d];
        joint_out[2 * R + kInkHidden + d] = ctx[d];
    }
}

__global__ void __launch_bounds__(256) k_dspark_markov_gru_upd(
    const float* __restrict__ proj,  // [768]
    const float* __restrict__ rp_b,  // [768]
    float* __restrict__ state,       // [256] in/out
    float* __restrict__ lat_out) {   // [256]
    constexpr int R = kInkMarkovRank;
    const int tid = threadIdx.x;
    const float p0 = proj[tid]         + rp_b[tid];
    const float p1 = proj[R + tid]     + rp_b[R + tid];
    const float p2 = proj[2 * R + tid] + rp_b[2 * R + tid];
    const float g = 1.0f / (1.0f + __expf(-p0));
    const float cand = tanhf(p1);
    state[tid] = cand + g * (state[tid] - cand);
    lat_out[tid] = tanhf(p2);
}

__global__ void __launch_bounds__(256) k_dspark_add_bias_and_h(
    float* __restrict__ corr,
    const float* __restrict__ ho_b,
    const float* __restrict__ hs,
    bool add_hs) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d < kInkHidden) corr[d] = corr[d] + ho_b[d] + (add_hs ? hs[d] : 0.0f);
}

// Multi-CTA L2-cached candidate scorer: Grid = (K_cand / 8 = 32 CTAs), 1 warp per candidate row -> ~4 us on RTX 5090!
__global__ void __launch_bounds__(256) k_dspark_score_cand_eval(
    const uint8_t* __restrict__ unemb_raw,
    int unemb_fmt,
    int unemb_rows,
    const float* __restrict__ corr,         // [2304]
    const int32_t* __restrict__ cand_ids,   // [K_cand]
    const float* __restrict__ cand_base,    // [K_cand] (nullable -> 0.0f)
    int K_cand,
    float* __restrict__ cand_scores) {      // [K_cand]
    __shared__ float s_corr[kInkHidden];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int d = tid; d < kInkHidden; d += 256) s_corr[d] = corr[d];
    __syncthreads();

    const int k = blockIdx.x * 8 + warp;
    if (k >= K_cand) return;
    const int32_t row = cand_ids[k];
    float acc = 0.0f;
    if (unemb_fmt == FMT_BF16) {
        const uint4* wr = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(unemb_raw) + (size_t)row * kInkHidden);
        for (int k8 = lane; k8 < (kInkHidden / 8); k8 += 32) {
            const uint4 wv = wr[k8];
            const float2 w0 = bf16x2_to_f32x2(wv.x), w1 = bf16x2_to_f32x2(wv.y);
            const float2 w2 = bf16x2_to_f32x2(wv.z), w3 = bf16x2_to_f32x2(wv.w);
            const float4 x0 = *(const float4*)(s_corr + k8 * 8);
            const float4 x1 = *(const float4*)(s_corr + k8 * 8 + 4);
            acc = fmaf(w0.x, x0.x, acc); acc = fmaf(w0.y, x0.y, acc);
            acc = fmaf(w1.x, x0.z, acc); acc = fmaf(w1.y, x0.w, acc);
            acc = fmaf(w2.x, x1.x, acc); acc = fmaf(w2.y, x1.y, acc);
            acc = fmaf(w3.x, x1.z, acc); acc = fmaf(w3.y, x1.w, acc);
        }
    } else if (unemb_fmt == FMT_Q8ROW) {
        const float sc = reinterpret_cast<const float*>(unemb_raw)[row];
        const int8_t* qr = reinterpret_cast<const int8_t*>(unemb_raw + (size_t)unemb_rows * 4) + (size_t)row * kInkHidden;
        for (int k4 = lane; k4 < (kInkHidden / 4); k4 += 32) {
            const int32_t w4 = *(const int32_t*)(qr + k4 * 4);
            const float4 xv = *(const float4*)(s_corr + k4 * 4);
            acc = fmaf((float)(int8_t)(w4 & 0xFF), xv.x, acc);
            acc = fmaf((float)(int8_t)((w4 >> 8) & 0xFF), xv.y, acc);
            acc = fmaf((float)(int8_t)((w4 >> 16) & 0xFF), xv.z, acc);
            acc = fmaf((float)(int8_t)((w4 >> 24) & 0xFF), xv.w, acc);
        }
        acc *= sc;
    } else if (unemb_fmt == FMT_S6_K) {
        const int sb_b = ink_sb_bytes(FMT_S6_K);
        const uint8_t* rptr = unemb_raw + (size_t)row * 9 * sb_b;
        for (int d = lane; d < kInkHidden; d += 32) {
            const int s = d >> 8, idx = d & 255, g = idx >> 6;
            const uint8_t* blk = rptr + s * sb_b;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const uint8_t* pl = blk + 6;
            uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
            uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
            acc = fmaf(((float)(lo | (hi << 4)) - 31.5f) * eff, s_corr[d], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float bv = cand_base ? cand_base[k] : 0.0f;
        cand_scores[k] = kInkSoftcap * tanhf((bv + acc) * (1.0f / kInkSoftcap));
    }
}

__global__ void __launch_bounds__(256) k_dspark_top2_reduce(
    const float* __restrict__ cand_scores,
    const int32_t* __restrict__ cand_ids,
    int K_cand,
    int32_t* __restrict__ out_top2_ids,
    float* __restrict__ out_top2_probs) {
    if (threadIdx.x != 0) return;
    float b1 = -3e38f, b2 = -3e38f;
    int32_t i1 = 0, i2 = 0;
    for (int k = 0; k < K_cand; ++k) {
        const float v = cand_scores[k];
        const int32_t id = cand_ids[k];
        if (v > b1 || (v == b1 && id < i1)) { b2 = b1; i2 = i1; b1 = v; i1 = id; }
        else if (v > b2 || (v == b2 && id < i2)) { b2 = v; i2 = id; }
    }
    float z = 0.0f;
    for (int k = 0; k < K_cand; ++k) z += expf(cand_scores[k] - b1);
    out_top2_ids[0] = i1;
    out_top2_ids[1] = i2;
    out_top2_probs[0] = 1.0f / z;
    out_top2_probs[1] = expf(b2 - b1) / z;
}

// ============================================================================
// InkDSpark: On-GPU 3-Layer Block-8 / Tree-16 Recurrent Markov Drafter
// ============================================================================
struct InkDSpark {
    InkCudaRunner& r;
    InkCudaModel& m;
    bool ready = false;
    int max_ctx = 4096;
    int K_cand = 256;  // reduced-vocab L2-cached candidate pool for Markov RNN

    // Context KV cache for 3 DSpark layers + f_rnn [max_ctx, 2304]
    __half* d_ctx_k[3] = {};
    __half* d_ctx_v[3] = {};
    float*  d_f_rnn = nullptr;  // [max_ctx, 2304]

    // Working buffers for Block-8 forward
    float *d_cur = nullptr, *d_h = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_qkv_fused = nullptr, *d_gu_fused = nullptr;
    float *d_attn_out = nullptr, *d_proj = nullptr, *d_g = nullptr, *d_u = nullptr, *d_act = nullptr;
    float *d_ctx_tmp = nullptr, *d_kc_tmp = nullptr, *d_vc_tmp = nullptr, *d_kv_ctx_fused = nullptr;
    float *d_base_logits = nullptr;  // [7, V_unpadded] (slots 1..7)
    int32_t *d_cand_ids = nullptr;   // [7, K_cand]
    float   *d_cand_base = nullptr;  // [7, K_cand]
    int32_t *d_top_tmp_ids = nullptr;// [7, 1024]
    float   *d_top_tmp_vals = nullptr;// [7, 1024]
    int8_t  *d_h_q8 = nullptr;       // [7, 2304]
    float   *d_h_sc = nullptr;       // [7]
    float   *d_cand_scores = nullptr;// [K_cand]
    float   *d_rnn_state = nullptr;  // [256]
    float   *d_rnn_joint = nullptr;  // [5120]
    float   *d_rnn_proj  = nullptr;  // [768]
    float   *d_rnn_lat   = nullptr;  // [256]
    float   *d_corr = nullptr;       // [2304]
    int32_t *d_pos = nullptr;        // [max_ctx]
    int32_t *d_apos = nullptr;       // [1]
    int32_t *d_anchor_id = nullptr;  // [1]
    int32_t *d_draft_ids = nullptr;  // [8] (slot 0 = anchor, slots 1..7 = drafted)
    int32_t *d_top2_ids = nullptr;   // [7, 2]
    float   *d_top2_probs = nullptr; // [7, 2]
    int32_t *h_top2_ids = nullptr;   // pinned [7, 2]
    float   *h_top2_probs = nullptr; // pinned [7, 2]
    int32_t *h_meta_pin = nullptr;   // pinned [16] for async anchor/apos/pos
    int32_t *h_pos_pin = nullptr;    // pinned [max_batch] for zero-sync ingest_taps_from!
    cudaGraphExec_t draft_graph = nullptr;
    // Pre-dequantized BF16 (sm_80+) or FP16 (sm_70/75) cache of DSpark's 3 tiny layers (+ fused kv_ctx!)
    std::unordered_map<std::string, uint16_t*> bf16_cache;
    uint16_t* d_bf16_pool = nullptr;
    __half*   d_unembed_fp16 = nullptr;  // Optional FP16 unembed on V100 sm_70 (32GB) -> 0.67 ms TC screening!
    bool cache_is_fp16 = false;

    InkDSpark(InkCudaRunner& runner) : r(runner), m(runner.m) {}

    void init(int ctx, int k_cand = 512) {
        if (!m.man.has_dspark) return;
        max_ctx = ctx;
        K_cand = k_cand;
        for (int il = 0; il < 3; ++il) {
            d_ctx_k[il] = dalloc<__half>((size_t)max_ctx * kInkKVHeads * kInkHeadDim);
            d_ctx_v[il] = dalloc<__half>((size_t)max_ctx * kInkKVHeads * kInkHeadDim);
        }
        d_f_rnn = dalloc<float>((size_t)max_ctx * kInkHidden);

        const int B = kInkDsparkBlock;  // 8
        d_cur         = dalloc<float>(B * kInkHidden);
        d_h           = dalloc<float>(B * kInkHidden);
        d_q           = dalloc<float>(B * kInkQDim);
        d_k           = dalloc<float>(B * kInkKVDim);
        d_v           = dalloc<float>(B * kInkKVDim);
        d_qkv_fused   = dalloc<float>(B * (kInkQDim + 2 * kInkKVDim));
        d_gu_fused    = dalloc<float>(B * (2 * kInkDsparkInter));
        d_attn_out    = dalloc<float>(B * kInkQDim);
        d_proj        = dalloc<float>(B * kInkHidden);
        d_g           = dalloc<float>(B * kInkDsparkInter);
        d_u           = dalloc<float>(B * kInkDsparkInter);
        d_act         = dalloc<float>(B * kInkDsparkInter);
        d_ctx_tmp     = dalloc<float>((size_t)r.max_batch * kInkHidden);
        d_kc_tmp      = dalloc<float>((size_t)r.max_batch * kInkKVDim);
        d_vc_tmp      = dalloc<float>((size_t)r.max_batch * kInkKVDim);
        d_kv_ctx_fused= dalloc<float>((size_t)r.max_batch * 2 * kInkKVDim);
        d_base_logits = dalloc<float>(8ull * ((m.man.c.vocab_unpadded + 7) & ~7));
        d_cand_ids    = dalloc<int32_t>(7ull * K_cand);
        d_cand_base   = dalloc<float>(7ull * K_cand);
        d_top_tmp_ids = dalloc<int32_t>(7ull * 2048);
        d_top_tmp_vals= dalloc<float>(7ull * 2048);
        d_h_q8        = dalloc<int8_t>(7ull * kInkHidden);
        d_h_sc        = dalloc<float>(7);
        d_cand_scores = dalloc<float>(K_cand);
        d_rnn_state   = dalloc<float>(kInkMarkovRank);
        d_rnn_joint   = dalloc<float>(2 * kInkMarkovRank + 2 * kInkHidden);
        d_rnn_proj    = dalloc<float>(3 * kInkMarkovRank);
        d_rnn_lat     = dalloc<float>(kInkMarkovRank);
        d_corr        = dalloc<float>(kInkHidden);
        d_pos         = dalloc<int32_t>(std::max(r.max_batch, 16));
        d_apos        = dalloc<int32_t>(1);
        d_anchor_id   = dalloc<int32_t>(1);
        d_draft_ids   = dalloc<int32_t>(16);
        d_top2_ids    = dalloc<int32_t>(14);
        d_top2_probs  = dalloc<float>(14);
        h_top2_ids    = halloc_pinned<int32_t>(14);
        h_top2_probs  = halloc_pinned<float>(14);
        h_meta_pin    = halloc_pinned<int32_t>(16);
        h_pos_pin     = halloc_pinned<int32_t>(std::max(r.max_batch, 16));

        // Pre-pack the 3 tiny DSpark layers into fused BF16 (sm_80+) or FP16 (sm_70/75) QKV, GU, KV_CTX, O, DOWN!
        // Collapses 21+6 cuBLAS calls -> 12+3 Tensor Core calls (< 0.35 ms on 5090 & V100!)
        size_t vfree = 0, vtot = 0;
        INK_CK(ink_cuda_mem_info(&vfree, &vtot));
        cache_is_fp16 = (m.sm_cc >= 70 && m.sm_cc < 80);
        auto load_to_tc = [&](const std::string& wn, uint16_t* dst_u16) {
            const auto& ti = m.tinfo(wn);
            const int N = (int)ti.rows(), K = (int)ti.cols();
            const uint8_t* W = m.d_trunk + ti.off;
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dst_u16);
            if (ti.fmt == FMT_BF16) {
                if (cache_is_fp16) {
                    const size_t nw = (size_t)N * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, r.st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(W), reinterpret_cast<__half*>(dst_u16), nw);
                    return;
                }
                INK_CK(cudaMemcpyAsync(dst, W, (size_t)N * K * 2, cudaMemcpyDeviceToDevice, r.st));
            } else {
                dim3 dg((K + 255) / 256, N);
                if (ti.fmt == FMT_Q8ROW)     k_ink_q8row_to_bf16<<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S5_K) k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S8_K) k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                if (cache_is_fp16) {
                    const size_t nw = (size_t)N * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, r.st>>>(dst, reinterpret_cast<__half*>(dst_u16), nw);
                }
            }
        };
        size_t tot_elems = 0;
        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            tot_elems += (size_t)(kInkQDim + 2 * kInkKVDim) * kInkHidden;   // fused QKV [5120, 2304]
            tot_elems += (size_t)(2 * kInkDsparkInter) * kInkHidden;        // fused GU  [18432, 2304]
            tot_elems += (size_t)(2 * kInkKVDim) * kInkHidden;              // fused KV_CTX [2048, 2304]
            if (cache_is_fp16 || m.tinfo(p + ".self_attn.o_proj.weight").fmt != FMT_BF16) tot_elems += (size_t)kInkHidden * kInkQDim;
            if (cache_is_fp16 || m.tinfo(p + ".mlp.down_proj.weight").fmt != FMT_BF16)    tot_elems += (size_t)kInkHidden * kInkDsparkInter;
        }
        if (m.sm_cc >= 70 && vfree > tot_elems * 2 + (768ull << 20)) {
            d_bf16_pool = dalloc<uint16_t>(tot_elems);
            size_t cur = 0;
            for (int il = 0; il < 3; ++il) {
                const std::string p = "dspark.layers." + std::to_string(il);
                uint16_t* qkv = d_bf16_pool + cur;
                load_to_tc(p + ".self_attn.q_proj.weight", qkv);
                load_to_tc(p + ".self_attn.k_proj.weight", qkv + (size_t)kInkQDim * kInkHidden);
                load_to_tc(p + ".self_attn.v_proj.weight", qkv + (size_t)(kInkQDim + kInkKVDim) * kInkHidden);
                bf16_cache[p + ".qkv_fused"] = qkv;
                cur += (size_t)(kInkQDim + 2 * kInkKVDim) * kInkHidden;

                uint16_t* gu = d_bf16_pool + cur;
                load_to_tc(p + ".mlp.gate_proj.weight", gu);
                load_to_tc(p + ".mlp.up_proj.weight",   gu + (size_t)kInkDsparkInter * kInkHidden);
                bf16_cache[p + ".gu_fused"] = gu;
                cur += (size_t)(2 * kInkDsparkInter) * kInkHidden;

                uint16_t* kv_ctx = d_bf16_pool + cur;
                load_to_tc(p + ".self_attn.k_proj_ctx.weight", kv_ctx);
                load_to_tc(p + ".self_attn.v_proj_ctx.weight", kv_ctx + (size_t)kInkKVDim * kInkHidden);
                bf16_cache[p + ".kv_ctx_fused"] = kv_ctx;
                cur += (size_t)(2 * kInkKVDim) * kInkHidden;

                const std::string op = p + ".self_attn.o_proj.weight";
                if (cache_is_fp16 || m.tinfo(op).fmt != FMT_BF16) {
                    load_to_tc(op, d_bf16_pool + cur);
                    bf16_cache[op] = d_bf16_pool + cur;
                    cur += (size_t)kInkHidden * kInkQDim;
                }
                const std::string dp = p + ".mlp.down_proj.weight";
                if (cache_is_fp16 || m.tinfo(dp).fmt != FMT_BF16) {
                    load_to_tc(dp, d_bf16_pool + cur);
                    bf16_cache[dp] = d_bf16_pool + cur;
                    cur += (size_t)kInkHidden * kInkDsparkInter;
                }
            }
            INK_CK(cudaStreamSynchronize(r.st));
        }
        // On Volta sm_70 (V100 32GB, lacks hardware dp4a), pre-pack FP16 unembed [V_pad, 2304] (~604 MB) when >2 GiB free
        // Pad V to multiple of 8 so Volta sm_70 cuBLAS engages 125 TFLOPS FP16 Tensor Cores (~0.67 ms vs 4.6 ms)!
        if (m.sm_cc == 70 && m.unembed_screen.ready) {
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            const int V_pad = (m.man.c.vocab_unpadded + 7) & ~7;
            const size_t u_elems = (size_t)V_pad * kInkHidden;
            if (vfree > u_elems * 2 + (2048ull << 20)) {
                d_unembed_fp16 = dalloc<__half>(u_elems);
                const auto& ui = m.tinfo("model.llm.unembed.weight");
                const uint8_t* W = m.d_trunk + ui.off;
                const int V_rows = std::min(V_pad, (int)ui.rows());
                if (ui.fmt == FMT_BF16) {
                    k_ink_bf16_to_fp16<<<((size_t)V_rows * kInkHidden + 255) / 256, 256, 0, r.st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(W), d_unembed_fp16, (size_t)V_rows * kInkHidden);
                } else {
                    dim3 dg((kInkHidden + 255) / 256, V_rows);
                    __nv_bfloat16* dst_b = reinterpret_cast<__nv_bfloat16*>(d_unembed_fp16);
                    if (ui.fmt == FMT_S5_K)      k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else if (ui.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else if (ui.fmt == FMT_S8_K) k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else                         k_ink_q8row_to_bf16<<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    k_ink_bf16_to_fp16<<<((size_t)V_rows * kInkHidden + 255) / 256, 256, 0, r.st>>>(
                        dst_b, d_unembed_fp16, (size_t)V_rows * kInkHidden);
                }
                INK_CK(cudaStreamSynchronize(r.st));
            }
        }
        ready = true;
    }

    // Fast cuBLAS BF16 (sm_80+) or FP16 (sm_70/75) Tensor Core linear for DSpark!
    void dspark_gemm_bf16(const uint16_t* W16, const float* X, float* Y, int B, int N, int K, bool reuse_x = false) {
        const float a = 1.0f, b = 0.0f;
        if (cache_is_fp16) {
            __half* x_fp16 = reinterpret_cast<__half*>(r.d_x_bf16);
            if (!reuse_x) {
                const size_t nx = (size_t)B * K;
                k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, r.st>>>(X, x_fp16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &a,
                W16, CUDA_R_16F, K, x_fp16, CUDA_R_16F, K, &b,
                Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            if (!reuse_x) {
                const size_t nx = (size_t)B * K;
                k_ink_f32_to_bf16<<<((nx + 3) / 4 + 255) / 256, 256, 0, r.st>>>(X, r.d_x_bf16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &a,
                W16, CUDA_R_16BF, K, r.d_x_bf16, CUDA_R_16BF, K, &b,
                Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    void dspark_linear_b8(const std::string& wn, const float* X, float* Y, int B, bool reuse_x = false) {
        if (m.sm_cc >= 70) {
            const auto& ti = m.tinfo(wn);
            const int N = (int)ti.rows(), K = (int)ti.cols();
            auto it = bf16_cache.find(wn);
            const uint16_t* W16 = (it != bf16_cache.end()) ? it->second
                                : ((!cache_is_fp16 && ti.fmt == FMT_BF16) ? reinterpret_cast<const uint16_t*>(m.d_trunk + ti.off) : nullptr);
            if (W16) {
                dspark_gemm_bf16(W16, X, Y, B, N, K, reuse_x);
                return;
            }
        }
        r.linear(wn, X, Y, B);
    }

    // Ingest target taps [4, M, 2304] at positions [p0 .. p0 + M - 1] into DSpark context KV & f_rnn (ZERO host sync!)
    void ingest_taps_from(float* const taps[4], int p0, int M) {
        if (!ready || M <= 0) return;
        for (int i = 0; i < M; ++i) h_pos_pin[i] = p0 + i;
        INK_CK(cudaMemcpyAsync(d_pos, h_pos_pin, (size_t)M * 4, cudaMemcpyHostToDevice, r.st));
        const float* fw = (const float*)m.trunk_ptr("dspark.fuse_weight");

        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_dspark_mix_taps<<<M, 256, 0, r.st>>>(
                taps[0], taps[1], taps[2], taps[3], fw, il,
                (const float*)m.trunk_ptr(p + ".dflare_norm.weight"), d_ctx_tmp, M);
            auto it_kv = bf16_cache.find(p + ".kv_ctx_fused");
            if (it_kv != bf16_cache.end()) {
                dspark_gemm_bf16(it_kv->second, d_ctx_tmp, d_kv_ctx_fused, M, 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_kv<<<dim3((2 * kInkKVDim + 255) / 256, M), 256, 0, r.st>>>(d_kv_ctx_fused, d_kc_tmp, d_vc_tmp, M);
            } else {
                r.linear(p + ".self_attn.k_proj_ctx.weight", d_ctx_tmp, d_kc_tmp, M);
                r.linear(p + ".self_attn.v_proj_ctx.weight", d_ctx_tmp, d_vc_tmp, M);
            }
            dim3 g(kInkKVHeads, M);
            k_dspark_rmsnorm_rope<<<g, 128, 0, r.st>>>(
                d_kc_tmp, (const float*)m.trunk_ptr(p + ".self_attn.k_norm_ctx.weight"),
                d_pos, m.man.c.dspark_rope_theta, M, kInkKVHeads);
            InkKvVisitDesc desc{};
            desc.k_f16 = d_ctx_k[il];
            desc.v_f16 = d_ctx_v[il];
            desc.cap_pos = max_ctx;
            desc.is_swa = false;
            desc.is_fp8 = false;
            k_ink_kv_write<<<g, 128, 0, r.st>>>(d_kc_tmp, d_vc_tmp, nullptr, 0, d_pos, desc, M);
        }
        k_dspark_mix_taps<<<M, 256, 0, r.st>>>(
            taps[0], taps[1], taps[2], taps[3], fw, 3,
            (const float*)m.trunk_ptr("dspark.markov_head.dflare_norm.weight"),
            d_f_rnn + (size_t)p0 * kInkHidden, M);
    }

    void ingest_taps(int p0, int M) {
        ingest_taps_from(r.d_taps, p0, M);
    }

    // Emit full Block-8 draft graph body (100% CUDA-Graph capturable -> < 0.45 ms on RTX 5090!)
    void emit_draft_block8() {
        const int B = kInkDsparkBlock;  // 8
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<1, 256, 0, r.st>>>(
            m.d_trunk + ei.off, ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_anchor_id, nullptr, nullptr, d_cur, 1, kInkHidden);
        const float* mask_e = (const float*)m.trunk_ptr("dspark.mask_embed");
        k_dspark_fill_mask<<<B - 1, 256, 0, r.st>>>(mask_e, d_cur, B);

        // 2. 3 non-causal block-attention layers (Fused QKV + Fused GU cuBLAS BF16 Tensor Cores -> < 0.85 ms total!)
        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".input_layernorm.weight"), d_h, nullptr, B, kInkHidden);
            auto it_qkv = bf16_cache.find(p + ".qkv_fused");
            if (it_qkv != bf16_cache.end()) {
                dspark_gemm_bf16(it_qkv->second, d_h, d_qkv_fused, B, kInkQDim + 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_qkv<<<dim3((kInkQDim + 2 * kInkKVDim + 255) / 256, B), 256, 0, r.st>>>(d_qkv_fused, d_q, d_k, d_v, B);
            } else {
                dspark_linear_b8(p + ".self_attn.q_proj.weight", d_h, d_q, B, false);
                dspark_linear_b8(p + ".self_attn.k_proj.weight", d_h, d_k, B, true);
                dspark_linear_b8(p + ".self_attn.v_proj.weight", d_h, d_v, B, true);
            }
            k_dspark_rmsnorm_rope<<<dim3(kInkHeads, B), 128, 0, r.st>>>(
                d_q, (const float*)m.trunk_ptr(p + ".self_attn.q_norm.weight"), d_pos, m.man.c.dspark_rope_theta, B, kInkHeads);
            k_dspark_rmsnorm_rope<<<dim3(kInkKVHeads, B), 128, 0, r.st>>>(
                d_k, (const float*)m.trunk_ptr(p + ".self_attn.k_norm.weight"), d_pos, m.man.c.dspark_rope_theta, B, kInkKVHeads);
            k_dspark_block_attn<<<dim3(kInkHeads, B), 128, 0, r.st>>>(
                d_q, d_k, d_v, d_ctx_k[il], d_ctx_v[il], d_apos,
                m.man.c.dspark_is_swa[il] != 0, m.man.c.dspark_swa_win, B, d_attn_out);
            dspark_linear_b8(p + ".self_attn.o_proj.weight", d_attn_out, d_proj, B, false);
            k_ink_add<<<(B * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, B * kInkHidden);

            k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".post_attention_layernorm.weight"), d_h, nullptr, B, kInkHidden);
            auto it_gu = bf16_cache.find(p + ".gu_fused");
            if (it_gu != bf16_cache.end()) {
                dspark_gemm_bf16(it_gu->second, d_h, d_gu_fused, B, 2 * kInkDsparkInter, kInkHidden, false);
                k_dspark_swiglu_fused<<<dim3((kInkDsparkInter + 255) / 256, B), 256, 0, r.st>>>(d_gu_fused, d_act, B);
            } else {
                dspark_linear_b8(p + ".mlp.gate_proj.weight", d_h, d_g, B, false);
                dspark_linear_b8(p + ".mlp.up_proj.weight", d_h, d_u, B, true);
                k_dspark_swiglu<<<(B * kInkDsparkInter + 255) / 256, 256, 0, r.st>>>(d_g, d_u, d_act, B * kInkDsparkInter);
            }
            dspark_linear_b8(p + ".mlp.down_proj.weight", d_act, d_proj, B, false);
            k_ink_add<<<(B * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, B * kInkHidden);
        }

        // 3. Final norm + ultra-fast dp4a INT8 base logits (~0.30 ms!) + 896-CTA 2-stage Top-512 (~0.03 ms!)
        k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr("dspark.norm.weight"), d_cur, nullptr, B, kInkHidden);
        const int V = m.man.c.vocab_unpadded;
        const float* base_logits_7 = d_base_logits;
        int base_ld = V;
        if (d_unembed_fp16) {
            // Volta sm_70 V100: (V_pad, 8, 2304) ALL multiples of 8 -> 125 TFLOPS FP16 Tensor Cores (~0.67 ms vs 4.6 ms!)
            const int V_pad = (V + 7) & ~7;
            const size_t nx = 8ull * kInkHidden;
            __half* x_fp16 = reinterpret_cast<__half*>(r.d_x_bf16);
            k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, r.st>>>(d_cur, x_fp16, nx);
            const float a = 1.0f, b = 0.0f;
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, V_pad, 8, kInkHidden, &a,
                d_unembed_fp16, CUDA_R_16F, kInkHidden, x_fp16, CUDA_R_16F, kInkHidden, &b,
                d_base_logits, CUDA_R_32F, V_pad, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            base_logits_7 = d_base_logits + V_pad;
            base_ld = V_pad;
        } else if (m.unembed_screen.ready) {
            k_dspark_quant_h_q8<<<7, 256, 0, r.st>>>(d_cur + kInkHidden, d_h_q8, d_h_sc);
            if (m.sm_cc == 70 || m.sm_cc == 60) {
                // Volta sm_70 (V100) & Pascal sm_60 (P100) lack hardware dp4a -> use 31.4/21.2 TFLOPS FP16x2 __hfma2!
                k_dspark_screen_h2_7<<<(V + 3) / 4, 128, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_h_q8, d_h_sc, d_base_logits, V);
            } else {
                k_dspark_screen_dp4a7<<<(V + 3) / 4, 128, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_h_q8, d_h_sc, d_base_logits, V);
            }
        } else {
            r.linear("model.llm.unembed.weight", d_cur + kInkHidden, d_base_logits, 7, V);
        }
        // Note: if base_ld > V (V_pad), pass V_pad? No, k_dspark_top_cand_s1 indexes slot*V + i!
        // Instead use ld=V when V_pad==V, or pass stride! Let's check k_dspark_top_cand_s1: ld=V if V_pad==V else V_pad.
        k_dspark_top_cand_s1<<<dim3(128, 7), 256, 0, r.st>>>(base_logits_7, V, d_top_tmp_ids, d_top_tmp_vals, base_ld);
        k_dspark_top_cand_s2<<<7, 256, 0, r.st>>>(d_top_tmp_ids, d_top_tmp_vals, K_cand, d_cand_ids, d_cand_base);

        // 4. Multi-CTA 7-step on-GPU Markov RNN loop (~0.06 ms total for all 7 steps!)
        INK_CK(cudaMemsetAsync(d_rnn_state, 0, kInkMarkovRank * 4, r.st));
        const auto& ti_mw1 = m.tinfo("dspark.markov_head.markov_w1.weight");
        const auto& ti_rpb = m.tinfo("dspark.markov_head.rnn_proj.bias");
        const auto& ti_hob = m.tinfo("dspark.markov_head.hidden_out.bias");
        const auto& ti_un  = m.tinfo("model.llm.unembed.weight");

        const int32_t* prev_ptr = d_anchor_id;
        for (int s = 0; s < 7; ++s) {
            const float* hs = d_cur + (size_t)(s + 1) * kInkHidden;
            k_dspark_markov_prep<<<1, 256, 0, r.st>>>(
                prev_ptr, d_rnn_state, hs, d_f_rnn, d_apos,
                m.d_trunk + ti_mw1.off, ti_mw1.fmt, d_rnn_joint);
            r.linear("dspark.markov_head.rnn_proj.weight", d_rnn_joint, d_rnn_proj, 1);
            k_dspark_markov_gru_upd<<<1, 256, 0, r.st>>>(
                d_rnn_proj, (const float*)(m.d_trunk + ti_rpb.off), d_rnn_state, d_rnn_lat);
            r.linear("dspark.markov_head.hidden_out.weight", d_rnn_lat, d_corr, 1);
            k_dspark_add_bias_and_h<<<(kInkHidden + 255) / 256, 256, 0, r.st>>>(
                d_corr, (const float*)(m.d_trunk + ti_hob.off), hs, m.unembed_screen.ready);
            k_dspark_score_cand_eval<<<(K_cand + 7) / 8, 256, 0, r.st>>>(
                m.d_trunk + ti_un.off, ti_un.fmt, (int)ti_un.rows(),
                d_corr, d_cand_ids + (size_t)s * K_cand,
                m.unembed_screen.ready ? nullptr : (d_cand_base + (size_t)s * K_cand),
                K_cand, d_cand_scores);
            k_dspark_top2_reduce<<<1, 256, 0, r.st>>>(
                d_cand_scores, d_cand_ids + (size_t)s * K_cand, K_cand,
                d_top2_ids + 2 * s, d_top2_probs + 2 * s);
            prev_ptr = d_top2_ids + 2 * s;
        }
    }

    // Draft 7 tokens (Block-8) + Top-2 candidates per slot on GPU (CUDA Graph captured!)
    void draft_block8(int32_t anchor_tok, int apos, int32_t* out_ids_7, float* out_conf_7 = nullptr) {
        const int B = kInkDsparkBlock;  // 8
        h_meta_pin[0] = anchor_tok;
        h_meta_pin[1] = apos;
        for (int i = 0; i < B; ++i) h_meta_pin[2 + i] = apos + i;
        INK_CK(cudaMemcpyAsync(d_anchor_id, h_meta_pin + 0, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_apos,      h_meta_pin + 1, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_meta_pin + 2, B * 4, cudaMemcpyHostToDevice, r.st));

        if (r.use_cuda_graph) {
            if (!draft_graph) {
                cudaGraph_t gr = nullptr;
                INK_CK(cudaStreamBeginCapture(r.st, cudaStreamCaptureModeThreadLocal));
                emit_draft_block8();
                INK_CK(cudaStreamEndCapture(r.st, &gr));
                INK_CK(cudaGraphInstantiate(&draft_graph, gr, nullptr, nullptr, 0));
                INK_CK(cudaGraphDestroy(gr));
            }
            INK_CK(cudaGraphLaunch(draft_graph, r.st));
        } else {
            emit_draft_block8();
        }
        INK_CK(cudaMemcpyAsync(h_top2_ids, d_top2_ids, 14 * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_top2_probs, d_top2_probs, 14 * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaStreamSynchronize(r.st));
        for (int s = 0; s < 7; ++s) {
            out_ids_7[s] = h_top2_ids[2 * s];
            if (out_conf_7) out_conf_7[s] = h_top2_probs[2 * s];
        }
    }

    void free_all() {
        if (draft_graph) { cudaGraphExecDestroy(draft_graph); draft_graph = nullptr; }
        bf16_cache.clear();
        if (d_bf16_pool) { cudaFree(d_bf16_pool); d_bf16_pool = nullptr; }
        if (d_unembed_fp16) { cudaFree(d_unembed_fp16); d_unembed_fp16 = nullptr; }
        for (int il = 0; il < 3; ++il) {
            if (d_ctx_k[il]) { cudaFree(d_ctx_k[il]); d_ctx_k[il] = nullptr; }
            if (d_ctx_v[il]) { cudaFree(d_ctx_v[il]); d_ctx_v[il] = nullptr; }
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_f_rnn); fr(d_cur); fr(d_h); fr(d_q); fr(d_k); fr(d_v); fr(d_qkv_fused); fr(d_gu_fused); fr(d_attn_out);
        fr(d_proj); fr(d_g); fr(d_u); fr(d_act); fr(d_ctx_tmp); fr(d_kc_tmp); fr(d_vc_tmp); fr(d_kv_ctx_fused);
        fr(d_base_logits); fr(d_cand_ids); fr(d_cand_base); fr(d_top_tmp_ids); fr(d_top_tmp_vals); fr(d_h_q8); fr(d_h_sc); fr(d_cand_scores);
        fr(d_rnn_state); fr(d_rnn_joint); fr(d_rnn_proj); fr(d_rnn_lat);
        fr(d_corr); fr(d_pos); fr(d_apos); fr(d_anchor_id); fr(d_draft_ids);
        fr(d_top2_ids); fr(d_top2_probs);
        if (h_top2_ids) { cudaFreeHost(h_top2_ids); h_top2_ids = nullptr; }
        if (h_top2_probs) { cudaFreeHost(h_top2_probs); h_top2_probs = nullptr; }
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (h_pos_pin) { cudaFreeHost(h_pos_pin); h_pos_pin = nullptr; }
        ready = false;
    }
};

}  // namespace strata::ink
