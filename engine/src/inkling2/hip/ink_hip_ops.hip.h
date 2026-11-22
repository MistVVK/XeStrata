// engine/src/inkling2/hip/ink_hip_ops.hip.h
// Fused 48-Visit Inkling-2-Mini HIP Kernels (k1..k5), Multi-Request Paged Flash-SWA/GQA (B=1..64, FP16 & FP8),
// Persistent On-GPU DSpark Block-8 Recurrent Markov Drafter (markov_rank=256), and 23-Layer BF16 Vision ViT.
// Works across RDNA4 (gfx120x), RDNA3/3.5 (gfx110x, gfx1103 780M, gfx1150 890M, gfx1151 Strix Halo), and RDNA2 (gfx103x).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_codecs.hip.h"

namespace strata::inkling2::hip {

constexpr float kRmsEps        = 1e-6f;
constexpr float kQkL2Clip      = 22.627416997969522f;
constexpr float kRelL2Clip     = 8.0f;
constexpr float kLogScaleAlpha = 0.1f;
constexpr int   kLogScaleFloor = 8192;
constexpr float kLogitSoftcap  = 30.0f;

// =================================================================================================
// k1: Fused Attention-Residual Depth Mix + Rank-64 Tanh Loop Message Controller
// =================================================================================================

// Pass up to 12 depth block pointers BY VALUE in kernel arguments -> ZERO H2D hipMemcpyAsync & 100% HIP-Graph safe!
struct BlockPtrs12 {
    const float* p[12];
};

// Fused Attention-Residual Depth Mix:
//   states = [bp.p[0..n_blocks-1], cur]  (n_states = n_blocks + 1 <= 13)
//   For each token m in [0..M-1] and query q in [0..n_queries-1]:
//     score[s] = dot(rms_norm(state[s], eps=1e-6), queries[q]) / sqrt(D)
//     w = softmax(score)
//     out[m, q, d] = sum_s w[s] * rms_norm(state[s])[m, d]
// Grid: (n_queries, M), Block: 256 (8 waves, Wave32 DPP reductions!)
__global__ void __launch_bounds__(256) k1_attn_res_mix(
    BlockPtrs12 bp,
    int n_blocks,
    const float* __restrict__ cur,                // [M, D]
    const float* __restrict__ queries,            // [n_queries, D]
    float* __restrict__ out,                      // [M, n_queries, D]
    int M, int n_queries, int D) {
    const int q = blockIdx.x;
    const int m = blockIdx.y;
    if (q >= n_queries || m >= M) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int n_states = n_blocks + 1;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const float* qv = queries + static_cast<size_t>(q) * D;

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    __shared__ float s_scores[16];
    __shared__ float s_inv_rms[16];

    // Pass 1: vectorized float4 loads + merged ss/dot reduction (2x fewer __syncthreads()!)
    const int d4 = D >> 2;
    const float4* qv4 = reinterpret_cast<const float4*>(qv);
    for (int s = 0; s < n_states; ++s) {
        const float4* sv4 = reinterpret_cast<const float4*>(
            (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                           : (cur + static_cast<size_t>(m) * D));
        float ss = 0.0f;
        float dot = 0.0f;
        for (int i = tid; i < d4; i += 256) {
            const float4 x = sv4[i];
            const float4 q4 = qv4[i];
            ss  = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss))));
            dot = fmaf(x.x, q4.x, fmaf(x.y, q4.y, fmaf(x.z, q4.z, fmaf(x.w, q4.w, dot))));
        }
        ss = wave32_sum(ss);
        dot = wave32_sum(dot);
        if (lane == 0) { s_ss[wave] = ss; s_dot[wave] = dot; }
        __syncthreads();
        if (tid == 0) {
            float tss = 0.0f, tdot = 0.0f;
            for (int w = 0; w < 8; ++w) { tss += s_ss[w]; tdot += s_dot[w]; }
            const float ir = rsqrtf(tss / static_cast<float>(D) + kRmsEps);
            s_inv_rms[s] = ir;
            s_scores[s] = tdot * ir * inv_sqrt_d;
        }
        __syncthreads();
    }

    // Softmax over n_states (<= 16)
    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s < n_states; ++s) mx = fmaxf(mx, s_scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            s_scores[s] = __expf(s_scores[s] - mx);
            sum += s_scores[s];
        }
        const float inv = 1.0f / fmaxf(sum, 1e-30f);
        for (int s = 0; s < n_states; ++s) s_scores[s] = (s_scores[s] * inv) * s_inv_rms[s];
    }
    __syncthreads();

    // Pass 2: vectorized float4 weighted sum across RMS-normalized depth states
    float4* dst4 = reinterpret_cast<float4*>(out + (static_cast<size_t>(m) * n_queries + q) * D);
    for (int i = tid; i < d4; i += 256) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            const float4* sv4 = reinterpret_cast<const float4*>(
                (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                               : (cur + static_cast<size_t>(m) * D));
            const float4 x = sv4[i];
            const float w = s_scores[s];
            a0 = fmaf(w, x.x, a0); a1 = fmaf(w, x.y, a1);
            a2 = fmaf(w, x.z, a2); a3 = fmaf(w, x.w, a3);
        }
        dst4[i] = make_float4(a0, a1, a2, a3);
    }
}

// Wave-6C Fused AttnRes(nq=1) + RMSNorm: computes depth mix in registers (float4 reg[4]), normalizes, and writes ONCE!
// Deletes 45 kernel launches + 45 full [B, D] DRAM round-trips per forward! Grid: (M,), Block: 256
__global__ void __launch_bounds__(256) k1_attn_res_mix_norm_fused(
    BlockPtrs12 bp,
    int n_blocks,
    const float* __restrict__ cur,       // [M, D]
    const float* __restrict__ query,     // [D]
    const float* __restrict__ norm_w,    // [D]
    float* __restrict__ out,             // [M, D]
    int M, int D, float rms_eps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int n_states = n_blocks + 1;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int d4 = D >> 2;
    const float4* qv4 = reinterpret_cast<const float4*>(query);

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    __shared__ float s_scores[16];
    __shared__ float s_inv_rms[16];
    __shared__ float s_out_inv;

    for (int s = 0; s < n_states; ++s) {
        const float4* sv4 = reinterpret_cast<const float4*>(
            (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                           : (cur + static_cast<size_t>(m) * D));
        float ss = 0.0f, dot = 0.0f;
        for (int i = tid; i < d4; i += 256) {
            const float4 x = sv4[i];
            const float4 q4 = qv4[i];
            ss  = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss))));
            dot = fmaf(x.x, q4.x, fmaf(x.y, q4.y, fmaf(x.z, q4.z, fmaf(x.w, q4.w, dot))));
        }
        ss = wave32_sum(ss);
        dot = wave32_sum(dot);
        if (lane == 0) { s_ss[wave] = ss; s_dot[wave] = dot; }
        __syncthreads();
        if (tid == 0) {
            float tss = 0.0f, tdot = 0.0f;
            for (int w = 0; w < 8; ++w) { tss += s_ss[w]; tdot += s_dot[w]; }
            const float ir = rsqrtf(tss / static_cast<float>(D) + kRmsEps);
            s_inv_rms[s] = ir;
            s_scores[s] = tdot * ir * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s < n_states; ++s) mx = fmaxf(mx, s_scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            s_scores[s] = __expf(s_scores[s] - mx);
            sum += s_scores[s];
        }
        const float inv = 1.0f / fmaxf(sum, 1e-30f);
        for (int s = 0; s < n_states; ++s) s_scores[s] = (s_scores[s] * inv) * s_inv_rms[s];
    }
    __syncthreads();

    // Mix into registers + accumulate out_ss for fused RMSNorm (D <= 4096 -> <= 4 float4/thread!)
    float4 reg[4];
    float oss = 0.0f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int i = tid + it * 256;
        if (i < d4) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                const float4* sv4 = reinterpret_cast<const float4*>(
                    (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                                   : (cur + static_cast<size_t>(m) * D));
                const float4 x = sv4[i];
                const float w = s_scores[s];
                a0 = fmaf(w, x.x, a0); a1 = fmaf(w, x.y, a1);
                a2 = fmaf(w, x.z, a2); a3 = fmaf(w, x.w, a3);
            }
            reg[it] = make_float4(a0, a1, a2, a3);
            oss = fmaf(a0, a0, fmaf(a1, a1, fmaf(a2, a2, fmaf(a3, a3, oss))));
        }
    }
    oss = wave32_sum(oss);
    if (lane == 0) s_ss[wave] = oss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_ss[w];
        s_out_inv = rsqrtf(tot / static_cast<float>(D) + rms_eps);
    }
    __syncthreads();
    const float oinv = s_out_inv;
    const float4* nw4 = reinterpret_cast<const float4*>(norm_w);
    float4* dst4 = reinterpret_cast<float4*>(out + static_cast<size_t>(m) * D);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int i = tid + it * 256;
        if (i < d4) {
            const float4 v = reg[it];
            const float4 ww = nw4[i];
            dst4[i] = make_float4(v.x * oinv * ww.x, v.y * oinv * ww.y, v.z * oinv * ww.z, v.w * oinv * ww.w);
        }
    }
}

// Single-Pass Multi-Query AttnRes Depth Mix (nq in 1..4): reads each depth state ONCE for all nq queries!
// Cuts DRAM traffic 3x-4x on Global layers (nq=3) and Cross-Pass Messages (nq=4) at B=512..2048!
// Grid: (M,), Block: 256
template <int NQ>
__global__ void __launch_bounds__(256) k1_attn_res_mix_multi(
    BlockPtrs12 bp,
    int n_blocks,
    const float* __restrict__ cur,
    const float* __restrict__ queries,  // [NQ, D]
    float* __restrict__ out,            // [M, NQ, D]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int n_states = n_blocks + 1;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int d4 = D >> 2;

    __shared__ float s_ss[8];
    __shared__ float s_dot[NQ][8];
    __shared__ float s_w[NQ][16];
    __shared__ float s_ir[16];

    for (int s = 0; s < n_states; ++s) {
        const float4* sv4 = reinterpret_cast<const float4*>(
            (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                           : (cur + static_cast<size_t>(m) * D));
        float ss = 0.0f;
        float dot[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) dot[q] = 0.0f;

        for (int i = tid; i < d4; i += 256) {
            const float4 x = sv4[i];
            ss = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss))));
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 q4 = reinterpret_cast<const float4*>(queries + static_cast<size_t>(q) * D)[i];
                dot[q] = fmaf(x.x, q4.x, fmaf(x.y, q4.y, fmaf(x.z, q4.z, fmaf(x.w, q4.w, dot[q]))));
            }
        }
        ss = wave32_sum(ss);
        if (lane == 0) s_ss[wave] = ss;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            dot[q] = wave32_sum(dot[q]);
            if (lane == 0) s_dot[q][wave] = dot[q];
        }
        __syncthreads();
        if (tid == 0) {
            float tss = 0.0f;
            for (int w = 0; w < 8; ++w) tss += s_ss[w];
            const float ir = rsqrtf(tss / static_cast<float>(D) + kRmsEps);
            s_ir[s] = ir;
            for (int q = 0; q < NQ; ++q) {
                float td = 0.0f;
                for (int w = 0; w < 8; ++w) td += s_dot[q][w];
                s_w[q][s] = td * ir * inv_sqrt_d;
            }
        }
        __syncthreads();
    }

    if (tid < NQ) {
        const int q = tid;
        float mx = -1e30f;
        for (int s = 0; s < n_states; ++s) mx = fmaxf(mx, s_w[q][s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            s_w[q][s] = __expf(s_w[q][s] - mx);
            sum += s_w[q][s];
        }
        const float inv = 1.0f / fmaxf(sum, 1e-30f);
        for (int s = 0; s < n_states; ++s) s_w[q][s] = (s_w[q][s] * inv) * s_ir[s];
    }
    __syncthreads();

    for (int i = tid; i < d4; i += 256) {
        float a0[NQ], a1[NQ], a2[NQ], a3[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) { a0[q] = 0.0f; a1[q] = 0.0f; a2[q] = 0.0f; a3[q] = 0.0f; }
        for (int s = 0; s < n_states; ++s) {
            const float4* sv4 = reinterpret_cast<const float4*>(
                (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                               : (cur + static_cast<size_t>(m) * D));
            const float4 x = sv4[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float w = s_w[q][s];
                a0[q] = fmaf(w, x.x, a0[q]); a1[q] = fmaf(w, x.y, a1[q]);
                a2[q] = fmaf(w, x.z, a2[q]); a3[q] = fmaf(w, x.w, a3[q]);
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            reinterpret_cast<float4*>(out + (static_cast<size_t>(m) * NQ + q) * D)[i] = make_float4(a0[q], a1[q], a2[q], a3[q]);
        }
    }
}

// Multi-CU Parallel Rank-64 Tanh Loop Message Controller (3 fast kernels across ALL CUs -> 20x faster at B=1!):
//   Step 1: k1_loop_ctrl_rms   <<<B, 256>>>                  -> inv_rms(cur)[B], inv_rms(msg)[B]
//   Step 2: k1_loop_ctrl_down  <<<dim3((rank+7)/8, B), 256>>> -> gate[B, rank] (8 blocks x 8 waves = 64 waves!)
//   Step 3: k1_loop_ctrl_up    <<<dim3((D+255)/256, B), 256>>> -> out[B, D] = (1 - tanh(W_f@g))*cur + tanh(W_g@g)*msg
__global__ void __launch_bounds__(256) k1_loop_ctrl_rms(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg_base, // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    float* __restrict__ inv_c_out,      // [M]
    float* __restrict__ inv_m_out,      // [M]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float* c_row = cur + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D;
    __shared__ float s_w[8];

    float ssc = 0.0f, ssm = 0.0f;
    for (int d = tid; d < D; d += 256) {
        ssc = fmaf(c_row[d], c_row[d], ssc);
        ssm = fmaf(m_row[d], m_row[d], ssm);
    }
    ssc = wave32_sum(ssc);
    ssm = wave32_sum(ssm);
    if (lane == 0) s_w[wave] = ssc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        inv_c_out[m] = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
    __syncthreads();
    if (lane == 0) s_w[wave] = ssm;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        inv_m_out[m] = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
}

template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_ctrl_down(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg_base, // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ inv_c_in, // [M]
    const float* __restrict__ inv_m_in, // [M]
    const void* __restrict__ w_down,    // [rank, 2*D]
    float* __restrict__ gate_out,       // [M, rank]
    int M, int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const int m = blockIdx.y;
    if (r >= rank || m >= M) return;

    const float* c_row = cur + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D;
    const float inv_c = inv_c_in[m];
    const float inv_m = inv_m_in[m];

    float dot = 0.0f;
    if constexpr (BF16_W) {
        const uint16_t* wr = static_cast<const uint16_t*>(w_down) + static_cast<size_t>(r) * (2 * D);
        for (int d = lane * 4; d < D; d += 128) {
            const uint2 wc = *reinterpret_cast<const uint2*>(wr + d);
            const uint2 wm = *reinterpret_cast<const uint2*>(wr + D + d);
            const float4 xc = *reinterpret_cast<const float4*>(c_row + d);
            const float4 xm = *reinterpret_cast<const float4*>(m_row + d);
            dot = fmaf(bf16_to_f32(wc.x & 0xFFFFu), xc.x * inv_c, dot);
            dot = fmaf(bf16_to_f32(wc.x >> 16),     xc.y * inv_c, dot);
            dot = fmaf(bf16_to_f32(wc.y & 0xFFFFu), xc.z * inv_c, dot);
            dot = fmaf(bf16_to_f32(wc.y >> 16),     xc.w * inv_c, dot);
            dot = fmaf(bf16_to_f32(wm.x & 0xFFFFu), xm.x * inv_m, dot);
            dot = fmaf(bf16_to_f32(wm.x >> 16),     xm.y * inv_m, dot);
            dot = fmaf(bf16_to_f32(wm.y & 0xFFFFu), xm.z * inv_m, dot);
            dot = fmaf(bf16_to_f32(wm.y >> 16),     xm.w * inv_m, dot);
        }
    } else {
        const float* wr = static_cast<const float*>(w_down) + static_cast<size_t>(r) * (2 * D);
        for (int d = lane * 4; d < D; d += 128) {
            const float4 wc = *reinterpret_cast<const float4*>(wr + d);
            const float4 wm = *reinterpret_cast<const float4*>(wr + D + d);
            const float4 xc = *reinterpret_cast<const float4*>(c_row + d);
            const float4 xm = *reinterpret_cast<const float4*>(m_row + d);
            dot = fmaf(wc.x, xc.x * inv_c, fmaf(wc.y, xc.y * inv_c, fmaf(wc.z, xc.z * inv_c, fmaf(wc.w, xc.w * inv_c, dot))));
            dot = fmaf(wm.x, xm.x * inv_m, fmaf(wm.y, xm.y * inv_m, fmaf(wm.z, xm.z * inv_m, fmaf(wm.w, xm.w * inv_m, dot))));
        }
    }
    dot = wave32_sum(dot);
    if (lane == 0) gate_out[static_cast<size_t>(m) * rank + r] = dot;
}

// Wave-6D High-Occupancy loop_ctrl_down for B<=4: 1 block (256 threads) per rank r=0..63 -> 64 blocks (8x CU occupancy!)
// Grid: (rank, M), Block: 256
template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_ctrl_down_hiocc(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg_base, // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ inv_c_in, // [M]
    const float* __restrict__ inv_m_in, // [M]
    const void* __restrict__ w_down,    // [rank, 2*D]
    float* __restrict__ gate_out,       // [M, rank]
    int M, int D, int rank) {
    const int r = blockIdx.x;
    const int m = blockIdx.y;
    if (r >= rank || m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    const float* c_row = cur + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D;
    const float inv_c = inv_c_in[m];
    const float inv_m = inv_m_in[m];

    float dot_c = 0.0f, dot_m = 0.0f;
    const int d4 = D >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(c_row);
    const float4* m4 = reinterpret_cast<const float4*>(m_row);
    if constexpr (BF16_W) {
        const uint2* wc2 = reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(w_down) + static_cast<size_t>(r) * (2 * D));
        const uint2* wm2 = reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(w_down) + static_cast<size_t>(r) * (2 * D) + D);
        for (int i = tid; i < d4; i += 256) {
            const uint2 wc = wc2[i];
            const uint2 wm = wm2[i];
            const float4 xc = c4[i];
            const float4 xm = m4[i];
            dot_c = fmaf(bf16_to_f32(wc.x & 0xFFFFu), xc.x, fmaf(bf16_to_f32(wc.x >> 16), xc.y,
                    fmaf(bf16_to_f32(wc.y & 0xFFFFu), xc.z, fmaf(bf16_to_f32(wc.y >> 16), xc.w, dot_c))));
            dot_m = fmaf(bf16_to_f32(wm.x & 0xFFFFu), xm.x, fmaf(bf16_to_f32(wm.x >> 16), xm.y,
                    fmaf(bf16_to_f32(wm.y & 0xFFFFu), xm.z, fmaf(bf16_to_f32(wm.y >> 16), xm.w, dot_m))));
        }
    } else {
        const float4* wc4 = reinterpret_cast<const float4*>(static_cast<const float*>(w_down) + static_cast<size_t>(r) * (2 * D));
        const float4* wm4 = reinterpret_cast<const float4*>(static_cast<const float*>(w_down) + static_cast<size_t>(r) * (2 * D) + D);
        for (int i = tid; i < d4; i += 256) {
            const float4 wc = wc4[i];
            const float4 wm = wm4[i];
            const float4 xc = c4[i];
            const float4 xm = m4[i];
            dot_c = fmaf(wc.x, xc.x, fmaf(wc.y, xc.y, fmaf(wc.z, xc.z, fmaf(wc.w, xc.w, dot_c))));
            dot_m = fmaf(wm.x, xm.x, fmaf(wm.y, xm.y, fmaf(wm.z, xm.z, fmaf(wm.w, xm.w, dot_m))));
        }
    }
    const float wsum = wave32_sum(dot_c * inv_c + dot_m * inv_m);
    __shared__ float s_w[8];
    if (lane == 0) s_w[wave] = wsum;
    __syncthreads();
    if (tid == 0) {
        gate_out[static_cast<size_t>(m) * rank + r] =
            s_w[0] + s_w[1] + s_w[2] + s_w[3] + s_w[4] + s_w[5] + s_w[6] + s_w[7];
    }
}

template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_ctrl_up(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg_base, // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ gate_in,  // [M, rank]
    const void* __restrict__ w_gain,    // [D, rank]
    const void* __restrict__ w_forget,  // [D, rank]
    float* __restrict__ out,            // [M, D]
    int M, int D, int rank) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (m >= M) return;

    __shared__ float s_gate[64];
    if (static_cast<int>(threadIdx.x) < rank) s_gate[threadIdx.x] = gate_in[static_cast<size_t>(m) * rank + threadIdx.x];
    __syncthreads();
    if (d >= D) return;

    float g_acc = 0.0f, f_acc = 0.0f;
    if constexpr (BF16_W) {
        const uint16_t* wg = static_cast<const uint16_t*>(w_gain) + static_cast<size_t>(d) * rank;
        const uint16_t* wf = static_cast<const uint16_t*>(w_forget) + static_cast<size_t>(d) * rank;
        for (int r = 0; r < rank; r += 4) {
            const uint2 g4 = *reinterpret_cast<const uint2*>(wg + r);
            const uint2 f4 = *reinterpret_cast<const uint2*>(wf + r);
            g_acc = fmaf(bf16_to_f32(g4.x & 0xFFFFu), s_gate[r + 0], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.x >> 16),     s_gate[r + 1], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.y & 0xFFFFu), s_gate[r + 2], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.y >> 16),     s_gate[r + 3], g_acc);
            f_acc = fmaf(bf16_to_f32(f4.x & 0xFFFFu), s_gate[r + 0], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.x >> 16),     s_gate[r + 1], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.y & 0xFFFFu), s_gate[r + 2], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.y >> 16),     s_gate[r + 3], f_acc);
        }
    } else {
        const float* wg = static_cast<const float*>(w_gain) + static_cast<size_t>(d) * rank;
        const float* wf = static_cast<const float*>(w_forget) + static_cast<size_t>(d) * rank;
        for (int r = 0; r < rank; ++r) {
            const float gr = s_gate[r];
            g_acc = fmaf(wg[r], gr, g_acc);
            f_acc = fmaf(wf[r], gr, f_acc);
        }
    }
    const float c = cur[static_cast<size_t>(m) * D + d];
    const float mv = msg_base[(static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D + d];
    out[static_cast<size_t>(m) * D + d] = fmaf(1.0f - tanhf(f_acc), c, tanhf(g_acc) * mv);
}

// Wave-6E Batched 3-Channel (attn, k, v) Loop Controllers + 3-Channel RMSNorm for B<=4:
// Deletes 232 launches/D2D copies per token and boosts k1_loop_ctrl_up occupancy 3x (36 -> 108 blocks)!
struct Ptr3    { const void*  p[3]; };
struct FPtr3   { const float* p[3]; };
struct MutFPtr3 { float*      p[3]; };

// Grid: (3, M), Block: 256
__global__ void __launch_bounds__(256) k1_loop_ctrl_rms3(
    FPtr3 cur_in,
    const float* __restrict__ msg_base, // [M, 4, D]
    float* __restrict__ inv_c_out,      // [3, M]
    float* __restrict__ inv_m_out,      // [3, M]
    int M, int D) {
    const int ch = blockIdx.x;
    const int m  = blockIdx.y;
    if (ch >= 3 || m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float* c_row = cur_in.p[ch] + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * 4 + ch) * D;
    float ssc = 0.0f, ssm = 0.0f;
    const int d4 = D >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(c_row);
    const float4* m4 = reinterpret_cast<const float4*>(m_row);
    for (int i = tid; i < d4; i += 256) {
        const float4 cv = c4[i];
        const float4 mv = m4[i];
        ssc = fmaf(cv.x, cv.x, fmaf(cv.y, cv.y, fmaf(cv.z, cv.z, fmaf(cv.w, cv.w, ssc))));
        ssm = fmaf(mv.x, mv.x, fmaf(mv.y, mv.y, fmaf(mv.z, mv.z, fmaf(mv.w, mv.w, ssm))));
    }
    ssc = wave32_sum(ssc);
    ssm = wave32_sum(ssm);
    __shared__ float s_wc[8], s_wm[8];
    if (lane == 0) { s_wc[wave] = ssc; s_wm[wave] = ssm; }
    __syncthreads();
    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f;
        for (int w = 0; w < 8; ++w) { tc += s_wc[w]; tm += s_wm[w]; }
        inv_c_out[ch * M + m] = rsqrtf(tc / static_cast<float>(D) + kRmsEps);
        inv_m_out[ch * M + m] = rsqrtf(tm / static_cast<float>(D) + kRmsEps);
    }
}

// Grid: (rank, M, 3), Block: 256 -> 192 blocks at B=1!
template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_ctrl_down_hiocc3(
    FPtr3 cur_in,
    const float* __restrict__ msg_base, // [M, 4, D]
    const float* __restrict__ inv_c_in, // [3, M]
    const float* __restrict__ inv_m_in, // [3, M]
    Ptr3 w_down,
    float* __restrict__ gate_out,       // [3, M, rank]
    int M, int D, int rank) {
    const int r  = blockIdx.x;
    const int m  = blockIdx.y;
    const int ch = blockIdx.z;
    if (r >= rank || m >= M || ch >= 3) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    const float* c_row = cur_in.p[ch] + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * 4 + ch) * D;
    const float inv_c = inv_c_in[ch * M + m];
    const float inv_m = inv_m_in[ch * M + m];

    float dot_c = 0.0f, dot_m = 0.0f;
    const int d4 = D >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(c_row);
    const float4* m4 = reinterpret_cast<const float4*>(m_row);
    if constexpr (BF16_W) {
        const uint16_t* wd = static_cast<const uint16_t*>(w_down.p[ch]) + static_cast<size_t>(r) * (2 * D);
        const uint2* wc2 = reinterpret_cast<const uint2*>(wd);
        const uint2* wm2 = reinterpret_cast<const uint2*>(wd + D);
        for (int i = tid; i < d4; i += 256) {
            const uint2 wc = wc2[i];
            const uint2 wm = wm2[i];
            const float4 xc = c4[i];
            const float4 xm = m4[i];
            dot_c = fmaf(bf16_to_f32(wc.x & 0xFFFFu), xc.x, fmaf(bf16_to_f32(wc.x >> 16), xc.y,
                    fmaf(bf16_to_f32(wc.y & 0xFFFFu), xc.z, fmaf(bf16_to_f32(wc.y >> 16), xc.w, dot_c))));
            dot_m = fmaf(bf16_to_f32(wm.x & 0xFFFFu), xm.x, fmaf(bf16_to_f32(wm.x >> 16), xm.y,
                    fmaf(bf16_to_f32(wm.y & 0xFFFFu), xm.z, fmaf(bf16_to_f32(wm.y >> 16), xm.w, dot_m))));
        }
    } else {
        const float* wd = static_cast<const float*>(w_down.p[ch]) + static_cast<size_t>(r) * (2 * D);
        const float4* wc4 = reinterpret_cast<const float4*>(wd);
        const float4* wm4 = reinterpret_cast<const float4*>(wd + D);
        for (int i = tid; i < d4; i += 256) {
            const float4 wc = wc4[i];
            const float4 wm = wm4[i];
            const float4 xc = c4[i];
            const float4 xm = m4[i];
            dot_c = fmaf(wc.x, xc.x, fmaf(wc.y, xc.y, fmaf(wc.z, xc.z, fmaf(wc.w, xc.w, dot_c))));
            dot_m = fmaf(wm.x, xm.x, fmaf(wm.y, xm.y, fmaf(wm.z, xm.z, fmaf(wm.w, xm.w, dot_m))));
        }
    }
    const float wsum = wave32_sum(dot_c * inv_c + dot_m * inv_m);
    __shared__ float s_w[8];
    if (lane == 0) s_w[wave] = wsum;
    __syncthreads();
    if (tid == 0) {
        gate_out[(static_cast<size_t>(ch) * M + m) * rank + r] =
            s_w[0] + s_w[1] + s_w[2] + s_w[3] + s_w[4] + s_w[5] + s_w[6] + s_w[7];
    }
}

// Grid: ((D + 63) / 64, M, 3), Block: 64 -> 108 blocks at B=1!
template <bool BF16_W>
__global__ void __launch_bounds__(64) k1_loop_ctrl_up3(
    FPtr3 cur_in,
    const float* __restrict__ msg_base, // [M, 4, D]
    const float* __restrict__ gate_in,  // [3, M, rank]
    Ptr3 w_gain,
    Ptr3 w_forget,
    MutFPtr3 out_ptrs,
    int M, int D, int rank) {
    const int d  = blockIdx.x * blockDim.x + threadIdx.x;
    const int m  = blockIdx.y;
    const int ch = blockIdx.z;
    if (m >= M || ch >= 3) return;

    __shared__ float s_gate[64];
    if (static_cast<int>(threadIdx.x) < rank) {
        s_gate[threadIdx.x] = gate_in[(static_cast<size_t>(ch) * M + m) * rank + threadIdx.x];
    }
    __syncthreads();
    if (d >= D) return;

    float g_acc = 0.0f, f_acc = 0.0f;
    if constexpr (BF16_W) {
        const uint16_t* wg = static_cast<const uint16_t*>(w_gain.p[ch]) + static_cast<size_t>(d) * rank;
        const uint16_t* wf = static_cast<const uint16_t*>(w_forget.p[ch]) + static_cast<size_t>(d) * rank;
        for (int r = 0; r < rank; r += 4) {
            const uint2 g4 = *reinterpret_cast<const uint2*>(wg + r);
            const uint2 f4 = *reinterpret_cast<const uint2*>(wf + r);
            g_acc = fmaf(bf16_to_f32(g4.x & 0xFFFFu), s_gate[r + 0], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.x >> 16),     s_gate[r + 1], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.y & 0xFFFFu), s_gate[r + 2], g_acc);
            g_acc = fmaf(bf16_to_f32(g4.y >> 16),     s_gate[r + 3], g_acc);
            f_acc = fmaf(bf16_to_f32(f4.x & 0xFFFFu), s_gate[r + 0], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.x >> 16),     s_gate[r + 1], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.y & 0xFFFFu), s_gate[r + 2], f_acc);
            f_acc = fmaf(bf16_to_f32(f4.y >> 16),     s_gate[r + 3], f_acc);
        }
    } else {
        const float* wg = static_cast<const float*>(w_gain.p[ch]) + static_cast<size_t>(d) * rank;
        const float* wf = static_cast<const float*>(w_forget.p[ch]) + static_cast<size_t>(d) * rank;
        for (int r = 0; r < rank; ++r) {
            const float gr = s_gate[r];
            g_acc = fmaf(wg[r], gr, g_acc);
            f_acc = fmaf(wf[r], gr, f_acc);
        }
    }
    const float c  = cur_in.p[ch][static_cast<size_t>(m) * D + d];
    const float mv = msg_base[(static_cast<size_t>(m) * 4 + ch) * D + d];
    out_ptrs.p[ch][static_cast<size_t>(m) * D + d] = fmaf(1.0f - tanhf(f_acc), c, tanhf(g_acc) * mv);
}

// Batched 3-Channel RMSNorm: Grid: (3, M), Block: 256 (replaces 3 sequential 1-block launches!)
__global__ void __launch_bounds__(256) k_rms_norm_rows3(
    MutFPtr3 io_ptrs,
    FPtr3 norm_w,
    int M, int D, float eps) {
    const int ch = blockIdx.x;
    const int m  = blockIdx.y;
    if (ch >= 3 || m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    float* row = io_ptrs.p[ch] + static_cast<size_t>(m) * D;
    const float* nw = norm_w.p[ch];
    const int d4 = D >> 2;
    float4* r4 = reinterpret_cast<float4*>(row);
    const float4* w4 = reinterpret_cast<const float4*>(nw);
    float ss = 0.0f;
    for (int i = tid; i < d4; i += 256) {
        const float4 v = r4[i];
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = wave32_sum(ss);
    __shared__ float s_w[8];
    __shared__ float s_inv;
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        s_inv = rsqrtf(t / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int i = tid; i < d4; i += 256) {
        const float4 v = r4[i];
        const float4 ww = w4[i];
        r4[i] = make_float4(v.x * inv * ww.x, v.y * inv * ww.y, v.z * inv * ww.z, v.w * inv * ww.w);
    }
}

// Batched WMMA helpers for k1_loop_ctrl when B > 4 (avoids reading controller weights B separate times!)
__global__ void __launch_bounds__(256) k1_loop_ctrl_pack_in(
    const float* __restrict__ cur,       // [M, D]
    const float* __restrict__ msg_base,  // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ inv_c_arr, // [M]
    const float* __restrict__ inv_m_arr, // [M]
    float* __restrict__ packed_out,      // [M, 2*D]
    int M, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (m >= M || d >= D) return;
    const float ic = inv_c_arr[m];
    const float im = inv_m_arr[m];
    float* dst = packed_out + static_cast<size_t>(m) * (2 * D);
    dst[d]     = cur[static_cast<size_t>(m) * D + d] * ic;
    dst[D + d] = msg_base[(static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D + d] * im;
}

__global__ void __launch_bounds__(256) k1_loop_ctrl_apply_out(
    const float* __restrict__ cur,        // [M, D]
    const float* __restrict__ msg_base,   // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ gain_raw,   // [M, D]
    const float* __restrict__ forget_raw, // [M, D]
    float* __restrict__ out,              // [M, D] (may alias cur!)
    int M, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (m >= M || d >= D) return;
    const size_t idx = static_cast<size_t>(m) * D + d;
    const float c = cur[idx];
    const float mv = msg_base[(static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D + d];
    out[idx] = fmaf(1.0f - tanhf(forget_raw[idx]), c, tanhf(gain_raw[idx]) * mv);
}

// Pack normalized [cur; msg] directly to BF16 (2x less DRAM than FP32!)
__global__ void __launch_bounds__(256) k1_loop_ctrl_pack_in_bf16(
    const float* __restrict__ cur,       // [M, D]
    const float* __restrict__ msg_base,  // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ inv_c_arr, // [M]
    const float* __restrict__ inv_m_arr, // [M]
    uint16_t* __restrict__ packed_bf16,  // [M, 2*D] BF16
    int M, int D) {
    const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    const int d = d4 * 4;
    if (m >= M || d >= D) return;
    const float ic = inv_c_arr[m];
    const float im = inv_m_arr[m];
    const float4 cv = *reinterpret_cast<const float4*>(cur + static_cast<size_t>(m) * D + d);
    const float4 mv = *reinterpret_cast<const float4*>(msg_base + (static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D + d);
    uint16_t* dst = packed_bf16 + static_cast<size_t>(m) * (2 * D);
    *reinterpret_cast<uint2*>(dst + d) = make_uint2(
        static_cast<uint32_t>(f32_to_bf16(cv.x * ic)) | (static_cast<uint32_t>(f32_to_bf16(cv.y * ic)) << 16),
        static_cast<uint32_t>(f32_to_bf16(cv.z * ic)) | (static_cast<uint32_t>(f32_to_bf16(cv.w * ic)) << 16));
    *reinterpret_cast<uint2*>(dst + D + d) = make_uint2(
        static_cast<uint32_t>(f32_to_bf16(mv.x * im)) | (static_cast<uint32_t>(f32_to_bf16(mv.y * im)) << 16),
        static_cast<uint32_t>(f32_to_bf16(mv.z * im)) | (static_cast<uint32_t>(f32_to_bf16(mv.w * im)) << 16));
}

// Fused WMMA Loop Controller Down-Projection (BM=64, BN=64, BK=64): reads cur & msg directly -> ZERO 37.7 MB cin DRAM!
// Grid: (1, (M + 63)/64) [since rank=64!], Block: 128 (4 waves: W_M=2, W_N=2, 18.4 KB LDS -> 3 WG/CU!)
__global__ void __launch_bounds__(128) k1_loop_ctrl_down_wmma_fused(
    const float* __restrict__ cur,       // [M, D]
    const float* __restrict__ msg_base,  // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ inv_c_arr, // [M]
    const float* __restrict__ inv_m_arr, // [M]
    const uint16_t* __restrict__ w_down, // [64, 2*D] BF16
    float* __restrict__ gate_out,        // [M, 64] FP32
    int M, int D) {
#if INK_HAS_WMMA
    constexpr int BM = 64, BN = 64, BK = 64, LD = BK + 8;
    __shared__ alignas(16) uint16_t sX[BM * LD];   // 9,216 B
    __shared__ alignas(16) uint16_t sW[BN * LD];   // 9,216 B -> 18,432 B (18.0 KB!)
    __shared__ float sInvC[BM];
    __shared__ float sInvM[BM];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int m_blk = blockIdx.y * BM;
    const int tm = (M - m_blk < BM) ? (M - m_blk) : BM;

    if (tid < BM) {
        sInvC[tid] = (tid < tm) ? inv_c_arr[m_blk + tid] : 0.0f;
        sInvM[tid] = (tid < tm) ? inv_m_arr[m_blk + tid] : 0.0f;
    }
    sync_lds();

    fv8 acc[2][2];
    acc[0][0] = fv8_zero(); acc[0][1] = fv8_zero();
    acc[1][0] = fv8_zero(); acc[1][1] = fv8_zero();

    // Two halves: h=0 reads cur * inv_c with w_down[:, 0..D-1], h=1 reads msg * inv_m with w_down[:, D..2D-1]
    for (int half = 0; half < 2; ++half) {
        const int w_col_off = half * D;
        for (int k0 = 0; k0 < D; k0 += BK) {
            for (int idx = tid; idx < BM * 16; idx += 128) {
                const int r = idx >> 4, c4 = idx & 15;
                uint16_t* dst = &sX[r * LD + c4 * 4];
                if (r < tm) {
                    const int gm = m_blk + r;
                    const float sc = (half == 0) ? sInvC[r] : sInvM[r];
                    const float* src_row = (half == 0)
                        ? (cur + static_cast<size_t>(gm) * D)
                        : (msg_base + (static_cast<size_t>(gm) * msg_stride_ch + msg_ch) * D);
                    const float4 v = *reinterpret_cast<const float4*>(src_row + k0 + c4 * 4);
                    *reinterpret_cast<uint2*>(dst) = make_uint2(
                        static_cast<uint32_t>(f32_to_bf16(v.x * sc)) | (static_cast<uint32_t>(f32_to_bf16(v.y * sc)) << 16),
                        static_cast<uint32_t>(f32_to_bf16(v.z * sc)) | (static_cast<uint32_t>(f32_to_bf16(v.w * sc)) << 16));
                } else {
                    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
                }
            }
            for (int idx = tid; idx < BN * 8; idx += 128) {
                const int r = idx >> 3, c8 = idx & 7;
                *reinterpret_cast<uint4*>(&sW[r * LD + c8 * 8]) =
                    *reinterpret_cast<const uint4*>(w_down + static_cast<size_t>(r) * (2 * D) + w_col_off + k0 + c8 * 8);
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    wmma_bf_t b0 = wmma_ld_bf(&sW[(wn * 32 + l16) * LD + kk * 16], hi);
                    wmma_bf_t b1 = wmma_ld_bf(&sW[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
                    wmma_bf_t a0 = wmma_ld_bf(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                    acc[0][0] = wmma_bf16(a0, b0, acc[0][0]);
                    acc[0][1] = wmma_bf16(a0, b1, acc[0][1]);
                    if (do_m1) {
                        wmma_bf_t a1 = wmma_ld_bf(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                        acc[1][0] = wmma_bf16(a1, b0, acc[1][0]);
                        acc[1][1] = wmma_bf16(a1, b1, acc[1][1]);
                    }
                }
            }
            sync_lds();
        }
    }

    if (wm * 32 < tm) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int gn = wn * 32 + ni * 16 + wmma_acc_n(0, lane);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) gate_out[static_cast<size_t>(gm) * 64 + gn] = acc[mi][ni][r];
                }
            }
        }
    }
#else
    (void)cur; (void)msg_base; (void)msg_stride_ch; (void)msg_ch; (void)inv_c_arr; (void)inv_m_arr; (void)w_down; (void)gate_out; (void)M; (void)D;
#endif
}

// Fused WMMA Loop Controller Up-Projection + Tanh Apply (K = rank = 64 -> ONE single BK=64 step, ZERO K-loop!)
// Computes cg = gate @ W_gain^T & cf = gate @ W_forget^T in registers and updates cur in-place -> ZERO cg/cf DRAM traffic!
// Grid: ((D + 63)/64, (M + 63)/64), Block: 128 (4 waves: 2x2 tiles/wave, 27.6 KB LDS < 32 KB -> 2 WG/CU!)
__global__ void __launch_bounds__(128) k1_loop_ctrl_up_wmma_fused(
    float* __restrict__ cur,              // [M, D] (updated in-place!)
    const float* __restrict__ msg_base,   // [M, msg_stride_ch, D]
    int msg_stride_ch, int msg_ch,
    const float* __restrict__ gate_in,    // [M, 64] FP32
    const uint16_t* __restrict__ w_gain,  // [D, 64] BF16
    const uint16_t* __restrict__ w_forget,// [D, 64] BF16
    int M, int D) {
#if INK_HAS_WMMA
    constexpr int BM = 64, BN = 64, BK = 64, LD = BK + 8;
    __shared__ alignas(16) uint16_t sG[BM * LD];   // 9,216 B
    __shared__ alignas(16) uint16_t sWg[BN * LD];  // 9,216 B
    __shared__ alignas(16) uint16_t sWf[BN * LD];  // 9,216 B -> 27,648 B (27.0 KB!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;  // 0..1 (32 M)
    const int wn = wave & 1;   // 0..1 (32 N)
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int tm = (M - m_blk < BM) ? (M - m_blk) : BM;

    // Load gate[64, 64] FP32 -> BF16 (8 iters/thread)
    for (int idx = tid; idx < BM * 16; idx += 128) {
        const int r = idx >> 4, c4 = idx & 15;
        uint16_t* dst = &sG[r * LD + c4 * 4];
        if (r < tm) {
            const float4 gv = *reinterpret_cast<const float4*>(gate_in + static_cast<size_t>(m_blk + r) * 64 + c4 * 4);
            *reinterpret_cast<uint2*>(dst) = make_uint2(
                static_cast<uint32_t>(f32_to_bf16(gv.x)) | (static_cast<uint32_t>(f32_to_bf16(gv.y)) << 16),
                static_cast<uint32_t>(f32_to_bf16(gv.z)) | (static_cast<uint32_t>(f32_to_bf16(gv.w)) << 16));
        } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
        }
    }
    // Load w_gain[64, 64] & w_forget[64, 64] BF16 via uint4 (4 iters/thread)
    for (int idx = tid; idx < BN * 8; idx += 128) {
        const int r = idx >> 3, c8 = idx & 7;
        const int gn = n_blk + r;
        if (gn < D) {
            *reinterpret_cast<uint4*>(&sWg[r * LD + c8 * 8]) = *reinterpret_cast<const uint4*>(w_gain   + static_cast<size_t>(gn) * 64 + c8 * 8);
            *reinterpret_cast<uint4*>(&sWf[r * LD + c8 * 8]) = *reinterpret_cast<const uint4*>(w_forget + static_cast<size_t>(gn) * 64 + c8 * 8);
        } else {
            *reinterpret_cast<uint4*>(&sWg[r * LD + c8 * 8]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sWf[r * LD + c8 * 8]) = make_uint4(0, 0, 0, 0);
        }
    }
    sync_lds();

    if (wm * 32 < tm) {
        fv8 ag[2][2], af[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) { ag[mi][ni] = fv8_zero(); af[mi][ni] = fv8_zero(); }
        const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_bf_t a0 = wmma_ld_bf(&sG[(wm * 32 + l16) * LD + kk * 16], hi);
            wmma_bf_t g0 = wmma_ld_bf(&sWg[(wn * 32 + l16) * LD + kk * 16], hi);
            wmma_bf_t g1 = wmma_ld_bf(&sWg[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
            wmma_bf_t f0 = wmma_ld_bf(&sWf[(wn * 32 + l16) * LD + kk * 16], hi);
            wmma_bf_t f1 = wmma_ld_bf(&sWf[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
            ag[0][0] = wmma_bf16(a0, g0, ag[0][0]);
            ag[0][1] = wmma_bf16(a0, g1, ag[0][1]);
            af[0][0] = wmma_bf16(a0, f0, af[0][0]);
            af[0][1] = wmma_bf16(a0, f1, af[0][1]);
            if (do_m1) {
                wmma_bf_t a1 = wmma_ld_bf(&sG[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                ag[1][0] = wmma_bf16(a1, g0, ag[1][0]);
                ag[1][1] = wmma_bf16(a1, g1, ag[1][1]);
                af[1][0] = wmma_bf16(a1, f0, af[1][0]);
                af[1][1] = wmma_bf16(a1, f1, af[1][1]);
            }
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                if (gn < D) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int gm = m_blk + wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (gm < M) {
                            const size_t cidx = static_cast<size_t>(gm) * D + gn;
                            const float cv = cur[cidx];
                            const float mv = msg_base[(static_cast<size_t>(gm) * msg_stride_ch + msg_ch) * D + gn];
                            cur[cidx] = fmaf(1.0f - tanhf(af[mi][ni][r]), cv, tanhf(ag[mi][ni][r]) * mv);
                        }
                    }
                }
            }
        }
    }
#else
    (void)cur; (void)msg_base; (void)msg_stride_ch; (void)msg_ch; (void)gate_in; (void)w_gain; (void)w_forget; (void)M; (void)D;
#endif
}

// Legacy single-block k1_loop_control kept for standalone tests
template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_control(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg_base, // [M, msg_stride_ch, D]
    int msg_stride_ch,                  // 4 when reading from msgs[M, 4, D], or 1
    int msg_ch,                         // 0=attn, 1=k, 2=v, 3=mlp
    const void* __restrict__ w_down,    // [rank, 2*D]
    const void* __restrict__ w_gain,    // [D, rank]
    const void* __restrict__ w_forget,  // [D, rank]
    float* __restrict__ out,            // [M, D]
    int M, int D, int rank) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    const float* c_row = cur + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * msg_stride_ch + msg_ch) * D;
    float* o_row = out + static_cast<size_t>(m) * D;

    __shared__ float s_wave[8];
    __shared__ float s_inv_c, s_inv_m;
    __shared__ float s_gate[64];

    // 1. RMSNorm scales for cur and msg
    float ssc = 0.0f, ssm = 0.0f;
    for (int d = tid; d < D; d += 256) {
        ssc = fmaf(c_row[d], c_row[d], ssc);
        ssm = fmaf(m_row[d], m_row[d], ssm);
    }
    ssc = wave32_sum(ssc);
    ssm = wave32_sum(ssm);
    if (lane == 0) s_wave[wave] = ssc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_wave[w];
        s_inv_c = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
    __syncthreads();
    if (lane == 0) s_wave[wave] = ssm;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_wave[w];
        s_inv_m = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
    __syncthreads();

    const float inv_c = s_inv_c;
    const float inv_m = s_inv_m;

    // 2. gate[r] = w_down[r] @ [rms(cur); rms(msg)] (8 waves process 8 rows of rank=64 at a time -> 8 iters!)
    for (int r0 = 0; r0 < rank; r0 += 8) {
        const int r = r0 + wave;
        if (r < rank) {
            float dot = 0.0f;
            if constexpr (BF16_W) {
                const uint16_t* wr = static_cast<const uint16_t*>(w_down) + static_cast<size_t>(r) * (2 * D);
                for (int d = lane; d < D; d += 32) {
                    dot = fmaf(bf16_to_f32(wr[d]), c_row[d] * inv_c, dot);
                    dot = fmaf(bf16_to_f32(wr[D + d]), m_row[d] * inv_m, dot);
                }
            } else {
                const float* wr = static_cast<const float*>(w_down) + static_cast<size_t>(r) * (2 * D);
                for (int d = lane; d < D; d += 32) {
                    dot = fmaf(wr[d], c_row[d] * inv_c, dot);
                    dot = fmaf(wr[D + d], m_row[d] * inv_m, dot);
                }
            }
            dot = wave32_sum(dot);
            if (lane == 0) s_gate[r] = dot;
        }
    }
    __syncthreads();

    // 3. gain/forget + gated mix out[d] = (1 - tanh(W_forget @ gate)) * cur[d] + tanh(W_gain @ gate) * msg[d]
    for (int d = tid; d < D; d += 256) {
        float g_acc = 0.0f, f_acc = 0.0f;
        if constexpr (BF16_W) {
            const uint16_t* wg = static_cast<const uint16_t*>(w_gain) + static_cast<size_t>(d) * rank;
            const uint16_t* wf = static_cast<const uint16_t*>(w_forget) + static_cast<size_t>(d) * rank;
            for (int r = 0; r < rank; ++r) {
                const float gr = s_gate[r];
                g_acc = fmaf(bf16_to_f32(wg[r]), gr, g_acc);
                f_acc = fmaf(bf16_to_f32(wf[r]), gr, f_acc);
            }
        } else {
            const float* wg = static_cast<const float*>(w_gain) + static_cast<size_t>(d) * rank;
            const float* wf = static_cast<const float*>(w_forget) + static_cast<size_t>(d) * rank;
            for (int r = 0; r < rank; ++r) {
                const float gr = s_gate[r];
                g_acc = fmaf(wg[r], gr, g_acc);
                f_acc = fmaf(wf[r], gr, f_acc);
            }
        }
        const float gain = tanhf(g_acc);
        const float forget = tanhf(f_acc);
        o_row[d] = fmaf(1.0f - forget, c_row[d], gain * m_row[d]);
    }
}

// =================================================================================================
// k2: Fused 4-Tap Causal ShortConv (Static + Optional Global Dynamic Per-Head Taps) + L2 Clip + V Norm + Tau
// =================================================================================================

// Single-step batched ShortConv4 across M distinct slots (slots[m] indexes into pool history[slot, 3, C]):
//   static_w: [C, 4] (index 0=t-3, 1=t-2, 2=t-1, 3=t)
//   dynamic_taps: nullptr (scale 1/sqrt(5)) OR [M, n_heads, 4] (lag 0=t, 1=t-1, 2=t-2, 3=t-3; scale 1/3)
__global__ void __launch_bounds__(256) k2_shortconv4_batched(
    const float* __restrict__ cur,          // [M, C]
    const float* __restrict__ static_w,     // [C, 4]
    const float* __restrict__ dyn_taps,     // nullptr or [M, n_heads, 4]
    const int32_t* __restrict__ slot_ids,   // [M]
    uint16_t* __restrict__ hist_pool,       // [max_slots, 3, C] (compact BF16 ring state matching CPU!)
    float* __restrict__ out,                // [M, C]
    int M, int C, int n_heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (c >= C || m >= M) return;

    const int slot = slot_ids[m];
    uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * C;
    const float h0 = bf16_to_f32(h_slot[0 * C + c]);  // t-3
    const float h1 = bf16_to_f32(h_slot[1 * C + c]);  // t-2
    const float h2 = bf16_to_f32(h_slot[2 * C + c]);  // t-1
    const float x0 = cur[static_cast<size_t>(m) * C + c];    // t

    const float* sw = static_w + static_cast<size_t>(c) * 4;
    float y = x0 + sw[0] * h0 + sw[1] * h1 + sw[2] * h2 + sw[3] * x0;

    if (dyn_taps != nullptr) {
        const int h_dim = C / n_heads;
        const int head = c / h_dim;
        const float* dt = dyn_taps + (static_cast<size_t>(m) * n_heads + head) * 4;
        y += dt[0] * x0 + dt[1] * h2 + dt[2] * h1 + dt[3] * h0;
        y *= (1.0f / 3.0f);
    } else {
        y *= 0.44721359549995793f;  // 1 / sqrt(5)
    }

    // Shift 3-tap history ring in-place (BF16 exact match to SlotShortConvState)
    h_slot[0 * C + c] = f32_to_bf16(h1);
    h_slot[1 * C + c] = f32_to_bf16(h2);
    h_slot[2 * C + c] = f32_to_bf16(x0);
    out[static_cast<size_t>(m) * C + c] = y;
}

// Causal multi-token ShortConv4 for batches where one slot has multiple consecutive tokens (prefill chunks & DSpark verify!)
// Processes one slot's token sequence [tok_indices[0..n_tok-1]] in registers — zero per-token launch overhead!
// Grid: ((C + 255) / 256, n_groups), Block: 256
__global__ void __launch_bounds__(256) k2_shortconv4_grouped_seq(
    const float* __restrict__ cur,          // [M, C]
    const float* __restrict__ static_w,     // [C, 4]
    const float* __restrict__ dyn_taps,     // nullptr or [M, n_heads, 4]
    const int32_t* __restrict__ grp_slot,   // [n_groups] slot id for this group
    const int32_t* __restrict__ grp_off,    // [n_groups + 1] offsets into grp_tok_idx
    const int32_t* __restrict__ grp_tok_idx,// [M] batch indices in ascending pos order per group
    uint16_t* __restrict__ hist_pool,       // [max_slots, 3, C] BF16
    float* __restrict__ out,                // [M, C]
    int n_groups, int C, int n_heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (c >= C || g >= n_groups) return;

    const int slot = grp_slot[g];
    const int beg = grp_off[g];
    const int end = grp_off[g + 1];
    uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * C;
    float h0 = bf16_to_f32(h_slot[0 * C + c]);
    float h1 = bf16_to_f32(h_slot[1 * C + c]);
    float h2 = bf16_to_f32(h_slot[2 * C + c]);

    const float* sw = static_w + static_cast<size_t>(c) * 4;
    const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
    const int head = (dyn_taps != nullptr && n_heads > 0) ? (c / (C / n_heads)) : 0;

    for (int i = beg; i < end; ++i) {
        const int m = grp_tok_idx[i];
        const float x0 = cur[static_cast<size_t>(m) * C + c];
        float y = x0 + sw0 * h0 + sw1 * h1 + sw2 * h2 + sw3 * x0;
        if (dyn_taps != nullptr) {
            const float* dt = dyn_taps + (static_cast<size_t>(m) * n_heads + head) * 4;
            y += dt[0] * x0 + dt[1] * h2 + dt[2] * h1 + dt[3] * h0;
            y *= (1.0f / 3.0f);
        } else {
            y *= 0.44721359549995793f;
        }
        out[static_cast<size_t>(m) * C + c] = y;
        // Round through BF16 between steps to stay bit-exact with SlotShortConvState!
        h0 = bf16_to_f32(f32_to_bf16(h1));
        h1 = bf16_to_f32(f32_to_bf16(h2));
        h2 = bf16_to_f32(f32_to_bf16(x0));
    }
    h_slot[0 * C + c] = f32_to_bf16(h0);
    h_slot[1 * C + c] = f32_to_bf16(h1);
    h_slot[2 * C + c] = f32_to_bf16(h2);
}

// Fused Q/K/Rel per-head L2 clip + optional per-head scale [n_heads, head_dim] (or [head_dim]) + optional log-position tau
// Grid: (n_heads, M), Block: 128 (4 waves for head_dim=128 or 16)
__global__ void __launch_bounds__(128) k2_head_l2_clip(
    float* __restrict__ x,              // [M, n_heads, head_dim]
    const float* __restrict__ scale,    // nullptr, [head_dim], or [n_heads, head_dim]
    int scale_head_stride,              // head_dim for [n_heads, head_dim], 0 for [head_dim]
    const int32_t* __restrict__ pos,    // nullptr or [M] (apply log_pos_tau when non-null)
    float threshold,
    int M, int n_heads, int head_dim) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (h >= n_heads || m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    float* xv = x + (static_cast<size_t>(m) * n_heads + h) * head_dim;
    const float* sc = scale ? (scale + static_cast<size_t>(h) * scale_head_stride) : nullptr;
    __shared__ float s_wave[4];
    __shared__ float s_factor;

    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += 128) {
        const float v = sc ? (xv[d] * sc[d]) : xv[d];
        xv[d] = v;
        ss = fmaf(v, v, ss);
    }
    ss = wave32_sum(ss);
    if (lane == 0) s_wave[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
        const float norm = sqrtf(tot) + 1e-6f;
        float fac = fminf(1.0f, threshold / norm);
        if (pos != nullptr) {
            const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
            const float tau = 1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r));
            fac *= tau;
        }
        s_factor = fac;
    }
    __syncthreads();
    const float fac = s_factor;
    for (int d = tid; d < head_dim; d += 128) xv[d] *= fac;
}

// Wave-6B Fused K+V ShortConv4 for distinct-slot decode (B <= 64): halves ShortConv launches (48 fewer/tok)!
// Grid: ((kv_dim + 255) / 256, 2 * B), Block: 256 (blockIdx.y < B -> K, >= B -> V)
__global__ void __launch_bounds__(256) k2_shortconv4_kv_pair(
    const float* __restrict__ k_cur,
    const float* __restrict__ k_sw,
    const float* __restrict__ k_dyn,
    uint16_t* __restrict__ k_hist_pool,
    float* __restrict__ k_out,
    const float* __restrict__ v_cur,
    const float* __restrict__ v_sw,
    const float* __restrict__ v_dyn,
    uint16_t* __restrict__ v_hist_pool,
    float* __restrict__ v_out,
    const int32_t* __restrict__ slot_ids,
    int M, int C, int n_heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y;
    const bool is_v = (by >= M);
    const int m = is_v ? (by - M) : by;
    if (c >= C || m >= M) return;

    const float* cur = is_v ? v_cur : k_cur;
    const float* static_w = is_v ? v_sw : k_sw;
    const float* dyn_taps = is_v ? v_dyn : k_dyn;
    uint16_t* hist_pool = is_v ? v_hist_pool : k_hist_pool;
    float* out = is_v ? v_out : k_out;

    const int slot = slot_ids[m];
    uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * C;
    const float h0 = bf16_to_f32(h_slot[0 * C + c]);
    const float h1 = bf16_to_f32(h_slot[1 * C + c]);
    const float h2 = bf16_to_f32(h_slot[2 * C + c]);
    const float x0 = cur[static_cast<size_t>(m) * C + c];

    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(c) * 4);
    float y = x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0;
    if (dyn_taps != nullptr) {
        const int head = c / (C / n_heads);
        const float4 dt = *reinterpret_cast<const float4*>(dyn_taps + (static_cast<size_t>(m) * n_heads + head) * 4);
        y += dt.x * x0 + dt.y * h2 + dt.z * h1 + dt.w * h0;
        y *= (1.0f / 3.0f);
    } else {
        y *= 0.44721359549995793f;
    }
    out[static_cast<size_t>(m) * C + c] = y;
    h_slot[0 * C + c] = h_slot[1 * C + c];
    h_slot[1 * C + c] = h_slot[2 * C + c];
    h_slot[2 * C + c] = f32_to_bf16(x0);
}

// Wave-6C Fused Post-Attn ShortConv4 + Residual Add into cur (deletes 36 launches + asc DRAM round-trips!)
// Grid: ((D + 255) / 256, B), Block: 256
__global__ void __launch_bounds__(256) k2_sconv_add_inplace(
    const float* __restrict__ in_x,       // [M, D]
    const float* __restrict__ static_w,   // [D, 4]
    const int32_t* __restrict__ slot_ids, // [M]
    uint16_t* __restrict__ hist_pool,     // [max_slots, 3, D]
    float* __restrict__ cur,              // [M, D] (cur += sconv(in_x))
    int M, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (c >= D || m >= M) return;
    const int slot = slot_ids[m];
    uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * D;
    const float h0 = bf16_to_f32(h_slot[0 * D + c]);
    const float h1 = bf16_to_f32(h_slot[1 * D + c]);
    const float h2 = bf16_to_f32(h_slot[2 * D + c]);
    const float x0 = in_x[static_cast<size_t>(m) * D + c];
    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(c) * 4);
    const float y = (x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0) * 0.44721359549995793f;
    h_slot[0 * D + c] = h_slot[1 * D + c];
    h_slot[1 * D + c] = h_slot[2 * D + c];
    h_slot[2 * D + c] = f32_to_bf16(x0);
    cur[static_cast<size_t>(m) * D + c] += y;
}

// Wave-6B Fused Q/R/K L2-Clip + V RMSNorm (1 launch instead of 4 -> 144 fewer launches/token, single-read register path!)
// Grid: (2 * n_head + 2 * n_kv, M), Block: 128
__global__ void __launch_bounds__(128) k2_qrkv_norm_clip_fused(
    float* __restrict__ Q,
    const float* __restrict__ q_sc, int q_sc_stride,
    float* __restrict__ R,
    const float* __restrict__ r_sc, int r_sc_stride,
    float* __restrict__ K,
    float* __restrict__ V,
    const int32_t* __restrict__ pos, // nullptr for SWA, non-null for Global log-pos tau
    float qk_clip, float rel_clip, float rms_eps,
    int M, int n_head, int n_kv, int head_dim, int d_rel) {
    const int bx = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_wave[4];
    __shared__ float s_fac;

    if (bx < n_head) {
        // Q head L2 clip + scale + optional tau (head_dim=128 -> 1 float per thread!)
        float* xv = Q + (static_cast<size_t>(m) * n_head + bx) * head_dim;
        const float* sc = q_sc + static_cast<size_t>(bx) * q_sc_stride;
        float v = (tid < head_dim) ? (xv[tid] * sc[tid]) : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            float fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
            if (pos != nullptr) {
                const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
                fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
            }
            s_fac = fac;
        }
        __syncthreads();
        if (tid < head_dim) xv[tid] = v * s_fac;
    } else if (bx < 2 * n_head) {
        // R head L2 clip + scale + optional tau (d_rel=32 -> 1 single wave, ZERO __syncthreads()!)
        if (tid >= 32) return;
        const int h = bx - n_head;
        float* xv = R + (static_cast<size_t>(m) * n_head + h) * d_rel;
        const float* sc = r_sc + static_cast<size_t>(h) * r_sc_stride;
        float v = (tid < d_rel) ? (xv[tid] * sc[tid]) : 0.0f;
        float tot = wave32_sum(v * v);
        float fac = fminf(1.0f, rel_clip / (sqrtf(tot) + 1e-6f));
        if (pos != nullptr) {
            const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
            fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
        }
        if (tid < d_rel) xv[tid] = v * fac;
    } else if (bx < 2 * n_head + n_kv) {
        // K head L2 clip (head_dim=128)
        const int h = bx - 2 * n_head;
        float* xv = K + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            s_fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
        }
        __syncthreads();
        if (tid < head_dim) xv[tid] = v * s_fac;
    } else {
        // V head RMSNorm (head_dim=128)
        const int h = bx - 2 * n_head - n_kv;
        if (h >= n_kv) return;
        float* xv = V + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            s_fac = rsqrtf((s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) / static_cast<float>(head_dim) + rms_eps);
        }
        __syncthreads();
        if (tid < head_dim) xv[tid] = v * s_fac;
    }
}

// Wave-6F Super-Fused Q/R/K/V Norm+Clip + Direct FP16 Paged KV Store + Direct RelPos Bias Projection!
// Replaces 3 kernels (k2_qrkv_norm_clip_fused + k3_kv_paged_store<false> + k_relpos_bias_proj) with ONE launch!
// Deletes 96 launches/tok AND eliminates K/V/R DRAM round-trips across B=1..64!
// Grid: (2 * n_head + 2 * n_kv, M), Block: 128
__global__ void __launch_bounds__(128) k2_qrkv_kvstore_relpos_fused(
    float* __restrict__ Q,
    const float* __restrict__ q_sc, int q_sc_stride,
    const float* __restrict__ R,
    const float* __restrict__ r_sc, int r_sc_stride,
    const float* __restrict__ rel_proj,       // [d_rel, rel_extent]
    float* __restrict__ rel_bias,             // [M, n_head, rel_extent]
    const float* __restrict__ K_in,
    const float* __restrict__ V_in,
    const int32_t* __restrict__ slot_ids,     // [M]
    const int32_t* __restrict__ pos,          // [M]
    const int32_t* __restrict__ page_table,   // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    half_t* __restrict__ k_pool,
    half_t* __restrict__ v_pool,
    bool apply_tau,
    float qk_clip, float rel_clip, float rms_eps,
    int M, int n_head, int n_kv, int head_dim, int d_rel, int swa_window, int rel_extent) {
    const int bx = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_wave[4];
    __shared__ float s_fac;
    __shared__ float s_r[32];

    if (bx < n_head) {
        // Q head L2 clip + scale + optional tau
        float* xv = Q + (static_cast<size_t>(m) * n_head + bx) * head_dim;
        const float* sc = q_sc + static_cast<size_t>(bx) * q_sc_stride;
        float v = (tid < head_dim) ? (xv[tid] * sc[tid]) : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            float fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
            if (apply_tau) {
                const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
                fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
            }
            s_fac = fac;
        }
        __syncthreads();
        if (tid < head_dim) xv[tid] = v * s_fac;
    } else if (bx < 2 * n_head) {
        // R head L2 clip + scale + optional tau -> register-fused RelPos bias projection across ALL 128 threads!
        const int h = bx - n_head;
        if (tid < 32) {
            const float* xv = R + (static_cast<size_t>(m) * n_head + h) * d_rel;
            const float* sc = r_sc + static_cast<size_t>(h) * r_sc_stride;
            float v = (tid < d_rel) ? (xv[tid] * sc[tid]) : 0.0f;
            float tot = wave32_sum(v * v);
            float fac = fminf(1.0f, rel_clip / (sqrtf(tot) + 1e-6f));
            if (apply_tau) {
                const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
                fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
            }
            if (tid < d_rel) s_r[tid] = v * fac;
        }
        __syncthreads();
        const int lim = min(pos[m] + 1, rel_extent);
        const int dr = min(d_rel, 16);
        float rloc[16];
#pragma unroll
        for (int rc = 0; rc < 16; ++rc) rloc[rc] = (rc < dr) ? s_r[rc] : 0.0f;
        float* dst = rel_bias + (static_cast<size_t>(m) * n_head + h) * rel_extent;
        const int lim4 = lim >> 2;
        for (int d4 = tid; d4 < lim4; d4 += 128) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int rc = 0; rc < 16; ++rc) {
                if (rc < dr) {
                    const float rv = rloc[rc];
                    const float4 pv = *reinterpret_cast<const float4*>(rel_proj + static_cast<size_t>(rc) * rel_extent + d4 * 4);
                    a0 = fmaf(rv, pv.x, a0); a1 = fmaf(rv, pv.y, a1);
                    a2 = fmaf(rv, pv.z, a2); a3 = fmaf(rv, pv.w, a3);
                }
            }
            *reinterpret_cast<float4*>(dst + d4 * 4) = make_float4(a0, a1, a2, a3);
        }
        const int d_rem = (lim4 << 2) + tid;
        if (d_rem < lim) {
            float acc = 0.0f;
            for (int rc = 0; rc < dr; ++rc) acc = fmaf(rloc[rc], rel_proj[static_cast<size_t>(rc) * rel_extent + d_rem], acc);
            dst[d_rem] = acc;
        }
    } else if (bx < 2 * n_head + n_kv) {
        // K head L2 clip -> direct FP16 write to Paged KV Pool (ZERO k_cv DRAM write/read!)
        const int h = bx - 2 * n_head;
        const float* xv = K_in + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            s_fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
        }
        __syncthreads();
        if (tid < head_dim) {
            const int slot = slot_ids[m];
            const int p = (swa_window > 0) ? (pos[m] % swa_window) : pos[m];
            const int phys = page_table[slot * max_pages_per_slot + (p >> 6)];
            const size_t dst = (static_cast<size_t>(phys) * 64 + (p & 63)) * (n_kv * head_dim) + h * head_dim + tid;
            k_pool[dst] = static_cast<half_t>(v * s_fac);
        }
    } else {
        // V head RMSNorm -> direct FP16 write to Paged KV Pool (ZERO v_cv DRAM write/read!)
        const int h = bx - 2 * n_head - n_kv;
        if (h >= n_kv) return;
        const float* xv = V_in + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            s_fac = rsqrtf((s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) / static_cast<float>(head_dim) + rms_eps);
        }
        __syncthreads();
        if (tid < head_dim) {
            const int slot = slot_ids[m];
            const int p = (swa_window > 0) ? (pos[m] % swa_window) : pos[m];
            const int phys = page_table[slot * max_pages_per_slot + (p >> 6)];
            const size_t dst = (static_cast<size_t>(phys) * 64 + (p & 63)) * (n_kv * head_dim) + h * head_dim + tid;
            v_pool[dst] = static_cast<half_t>(v * s_fac);
        }
    }
}

// =================================================================================================
// k3: RelPos + Sink Flash-SWA (40 Visits, Window=1024) & Global Flash-GQA (8 Visits, 24:2)
// Supports Paged 64-Token KV Blocks in BOTH FP16 and FP8_E4M3 across B=1..64 Concurrent Slots!
// Clean Wave32 online-softmax implementation — works identically on RDNA4, RDNA3/3.5, and RDNA2!
// =================================================================================================

// Store newly computed K/V (head_dim=128, n_kv_heads=8) into paged KV pool (page_size=64).
// For FP8_KV, computes exact per-token scale = amax / 448.0f (matching SlotKvPages in ink_batch.h!).
// Grid: (M,), Block: 256
template <bool FP8_KV>
__global__ void __launch_bounds__(256) k3_kv_paged_store(
    const float* __restrict__ K_in,         // [M, n_kv_heads, head_dim]
    const float* __restrict__ V_in,         // [M, n_kv_heads, head_dim]
    const int32_t* __restrict__ slot_ids,   // [M]
    const int32_t* __restrict__ pos,        // [M]
    const int32_t* __restrict__ page_table, // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    void* __restrict__ k_pool,              // [n_phys_pages, 64, n_kv_heads, head_dim] (half_t or uint8_t)
    void* __restrict__ v_pool,              // [n_phys_pages, 64, n_kv_heads, head_dim] (half_t or uint8_t)
    float* __restrict__ k_scales,           // nullptr or [n_phys_pages, 64] (for FP8_KV)
    float* __restrict__ v_scales,           // nullptr or [n_phys_pages, 64] (for FP8_KV)
    int M, int n_kv_heads, int head_dim, int swa_window) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int elems = n_kv_heads * head_dim;

    const int slot = slot_ids[m];
    const int p = (swa_window > 0) ? (pos[m] % swa_window) : pos[m];
    const int lpage = p >> 6;
    const int poff = p & 63;
    const int phys = page_table[slot * max_pages_per_slot + lpage];
    const size_t tok_base = (static_cast<size_t>(phys) * 64 + poff);
    const size_t dst_base = tok_base * elems;
    const float* ks = K_in + static_cast<size_t>(m) * elems;
    const float* vs = V_in + static_cast<size_t>(m) * elems;

    if constexpr (FP8_KV) {
        const int wave = tid >> 5;
        const int lane = tid & 31;
        __shared__ float s_w[8];
        __shared__ float s_inv_k, s_inv_v;

        float km = 0.0f, vm = 0.0f;
        for (int i = tid; i < elems; i += 256) {
            km = fmaxf(km, fabsf(ks[i]));
            vm = fmaxf(vm, fabsf(vs[i]));
        }
        km = wave32_max(km);
        vm = wave32_max(vm);
        if (lane == 0) s_w[wave] = km;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f;
            for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (k_scales) k_scales[tok_base] = sc;
            s_inv_k = 1.0f / sc;
        }
        __syncthreads();
        if (lane == 0) s_w[wave] = vm;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f;
            for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (v_scales) v_scales[tok_base] = sc;
            s_inv_v = 1.0f / sc;
        }
        __syncthreads();
        const float ik = s_inv_k, iv = s_inv_v;
        uint8_t* kd = static_cast<uint8_t*>(k_pool) + dst_base;
        uint8_t* vd = static_cast<uint8_t*>(v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = f32_to_e4m3fn(ks[i] * ik);
            vd[i] = f32_to_e4m3fn(vs[i] * iv);
        }
    } else {
        half_t* kd = static_cast<half_t*>(k_pool) + dst_base;
        half_t* vd = static_cast<half_t*>(v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = static_cast<half_t>(ks[i]);
            vd[i] = static_cast<half_t>(vs[i]);
        }
    }
}

// 100% Parallel 4-tap ShortConv for single-slot contiguous prefill (B >= 16):
// Wave-5G: Tiled 16-token sliding register window per thread -> loads static_w[c,0..3] ONCE and in_x 1.18x instead of 4.0x!
// Grid: ((C + 255) / 256, (B + 15) / 16), Block: 256
__global__ void __launch_bounds__(256) k2_shortconv4_prefill_par(
    const float* __restrict__ in_x,       // [B, C]
    const float* __restrict__ static_w,   // [C, 4] (oldest lag t-3 at [0] .. current t at [3])
    const float* __restrict__ dyn_taps,   // nullptr or [B, n_heads, 4] (lag 0=t..3=t-3)
    const uint16_t* __restrict__ hist_in, // [3, C] BF16 for this slot (pre-batch history)
    float* __restrict__ out_y,            // [B, C]
    int B, int C, int n_heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y * 16;
    if (c >= C || t0 >= B) return;

    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(c) * 4);
    float x0 = (t0 >= 3) ? bf16_to_f32(f32_to_bf16(in_x[static_cast<size_t>(t0 - 3) * C + c])) : bf16_to_f32(hist_in[t0 * C + c]);
    float x1 = (t0 >= 2) ? bf16_to_f32(f32_to_bf16(in_x[static_cast<size_t>(t0 - 2) * C + c])) : bf16_to_f32(hist_in[(t0 + 1) * C + c]);
    float x2 = (t0 >= 1) ? bf16_to_f32(f32_to_bf16(in_x[static_cast<size_t>(t0 - 1) * C + c])) : bf16_to_f32(hist_in[2 * C + c]);

    const bool has_dyn = (dyn_taps != nullptr && n_heads > 0);
    const int h = has_dyn ? (c / (C / n_heads)) : 0;
    const int tend = (t0 + 16 < B) ? (t0 + 16) : B;

    for (int t = t0; t < tend; ++t) {
        const float x3 = in_x[static_cast<size_t>(t) * C + c];
        float y = x3 + sw.x * x0 + sw.y * x1 + sw.z * x2 + sw.w * x3;
        if (has_dyn) {
            const float4 dt = *reinterpret_cast<const float4*>(dyn_taps + (static_cast<size_t>(t) * n_heads + h) * 4);
            y += x3 * dt.x + x2 * dt.y + x1 * dt.z + x0 * dt.w;
            y *= (1.0f / 3.0f);
        } else {
            y *= 0.44721359549995793f;  // 1 / sqrt(5)
        }
        out_y[static_cast<size_t>(t) * C + c] = y;
        x0 = x1;
        x1 = x2;
        x2 = bf16_to_f32(f32_to_bf16(x3));
    }
}

__global__ void __launch_bounds__(256) k2_shortconv4_update_hist(
    const float* __restrict__ in_x,       // [B, C]
    uint16_t* __restrict__ hist_slot,     // [3, C] BF16 for this slot
    int B, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    if (B >= 3) {
        hist_slot[0 * C + c] = f32_to_bf16(in_x[static_cast<size_t>(B - 3) * C + c]);
        hist_slot[1 * C + c] = f32_to_bf16(in_x[static_cast<size_t>(B - 2) * C + c]);
        hist_slot[2 * C + c] = f32_to_bf16(in_x[static_cast<size_t>(B - 1) * C + c]);
    } else if (B == 2) {
        hist_slot[0 * C + c] = hist_slot[2 * C + c];
        hist_slot[1 * C + c] = f32_to_bf16(in_x[0 * C + c]);
        hist_slot[2 * C + c] = f32_to_bf16(in_x[1 * C + c]);
    } else if (B == 1) {
        hist_slot[0 * C + c] = hist_slot[1 * C + c];
        hist_slot[1 * C + c] = hist_slot[2 * C + c];
        hist_slot[2 * C + c] = f32_to_bf16(in_x[0 * C + c]);
    }
}

// Flash-SWA / Flash-GQA Attention Kernel with RelPos Bias + Per-Query Attention Sinks + Sigmoid Gate!
// Wave32 Zero-Barrier Design: 1 Wave32 per (qh, m), 4 dims per lane (32*4 = head_dim 128), 4 Q-heads per 128-thread block!
// Coalesced uint2 (FP16) / uint32_t (FP8) KV loads + pure register DPP wave32_sum -> ZERO __syncthreads(), ZERO LDS!
// Grid: ((n_q_heads + 3) / 4, M), Block: 128
template <bool FP8_KV>
__global__ void __launch_bounds__(128) k3_flash_swa_gqa(
    const float* __restrict__ Q,            // [M, n_q_heads, 128]
    const float* __restrict__ gate_attn,    // nullptr or [M, n_q_heads, 128] (sigmoid gate)
    const float* __restrict__ rel_bias,     // nullptr or [M, n_q_heads, max_rel] (pre-projected rel_logits_proj)
    const float* __restrict__ sink_logit,   // nullptr or [M, n_q_heads] (per-query sink logit)
    const int32_t* __restrict__ slot_ids,   // [M]
    const int32_t* __restrict__ pos,        // [M]
    const int32_t* __restrict__ page_table, // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    const void* __restrict__ k_pool,        // [n_phys_pages, 64, n_kv_heads, 128]
    const void* __restrict__ v_pool,        // [n_phys_pages, 64, n_kv_heads, 128]
    const float* __restrict__ k_scales,     // nullptr or [n_phys_pages, 64]
    const float* __restrict__ v_scales,     // nullptr or [n_phys_pages, 64]
    float* __restrict__ O_out,              // [M, n_q_heads, 128]
    int M, int n_q_heads, int n_kv_heads, int swa_window, int max_rel) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x * 4 + wave;
    const int m = blockIdx.y;
    if (qh >= n_q_heads || m >= M) return;

    const int d0 = lane * 4;  // 0, 4, ..., 124 (4 dims per lane across 32 lanes = 128!)
    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int slot = slot_ids[m];
    const int cur_pos = pos[m];
    const int k_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;
    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = sink_logit ? sink_logit[static_cast<size_t>(m) * n_q_heads + qh] : -1e30f;
    float sum_exp = sink_logit ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int elems = n_kv_heads * 128;
    const int kv_off = kvh * 128 + d0;
    const float* rb_row = rel_bias ? (rel_bias + (static_cast<size_t>(m) * n_q_heads + qh) * max_rel) : nullptr;
    const int32_t* pt_row = page_table + slot * max_pages_per_slot;

    for (int kp = k_start; kp <= cur_pos; ++kp) {
        const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
        const int phys = pt_row[ring_p >> 6];
        const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
        const size_t base = tok_base * elems + kv_off;

        float k0, k1, k2, k3, v0, v1, v2, v3;
        if constexpr (FP8_KV) {
            const float ks = k_scales ? k_scales[tok_base] : 1.0f;
            const float vs = v_scales ? v_scales[tok_base] : 1.0f;
            const uint32_t kw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(k_pool) + base);
            const uint32_t vw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(v_pool) + base);
            k0 = e4m3fn_to_f32(static_cast<uint8_t>(kw & 0xFFu)) * ks;
            k1 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 8) & 0xFFu)) * ks;
            k2 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 16) & 0xFFu)) * ks;
            k3 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 24) & 0xFFu)) * ks;
            v0 = e4m3fn_to_f32(static_cast<uint8_t>(vw & 0xFFu)) * vs;
            v1 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 8) & 0xFFu)) * vs;
            v2 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 16) & 0xFFu)) * vs;
            v3 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 24) & 0xFFu)) * vs;
        } else {
            const uint2 kw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(k_pool) + base);
            const uint2 vw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(v_pool) + base);
            h2_unpack(kw.x, k0, k1); h2_unpack(kw.y, k2, k3);
            h2_unpack(vw.x, v0, v1); h2_unpack(vw.y, v2, v3);
        }

        float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const int dist = cur_pos - kp;
        if (rb_row && dist < max_rel) sc += rb_row[dist];

        const float new_max = fmaxf(max_l, sc);
        const float alpha = __expf(max_l - new_max);
        const float beta = __expf(sc - new_max);
        o0 = fmaf(o0, alpha, beta * v0);
        o1 = fmaf(o1, alpha, beta * v1);
        o2 = fmaf(o2, alpha, beta * v2);
        o3 = fmaf(o3, alpha, beta * v3);
        sum_exp = fmaf(sum_exp, alpha, beta);
        max_l = new_max;
    }

    const float inv_s = 1.0f / fmaxf(sum_exp, 1e-30f);
    o0 *= inv_s; o1 *= inv_s; o2 *= inv_s; o3 *= inv_s;
    if (gate_attn != nullptr) {
        const float4 gv = *reinterpret_cast<const float4*>(gate_attn + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
        o0 *= 1.0f / (1.0f + __expf(-gv.x));
        o1 *= 1.0f / (1.0f + __expf(-gv.y));
        o2 *= 1.0f / (1.0f + __expf(-gv.z));
        o3 *= 1.0f / (1.0f + __expf(-gv.w));
    }
    *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0) = make_float4(o0, o1, o2, o3);
}

// 4-Wave Split-KV Flash-SWA / Flash-GQA for small batches (M <= 8: B=1 decode & B=8 DSpark verify!):
// Launches 1 block per (qh, m) = 24 blocks x 4 waves = 96 waves at B=1 (4x-8x faster at P=256..2048!).
// Zero barrier in the entire KV loop; 1 single __syncthreads() at the end merges the 4 waves' online softmax!
// Grid: (n_q_heads, M), Block: 128
template <bool FP8_KV>
__global__ void __launch_bounds__(128) k3_flash_swa_gqa_splitkv(
    const float* __restrict__ Q,            // [M, n_q_heads, 128]
    const float* __restrict__ gate_attn,    // nullptr or [M, n_q_heads, 128]
    const float* __restrict__ rel_bias,     // nullptr or [M, n_q_heads, max_rel]
    const float* __restrict__ sink_logit,   // nullptr or [M, n_q_heads]
    const int32_t* __restrict__ slot_ids,   // [M]
    const int32_t* __restrict__ pos,        // [M]
    const int32_t* __restrict__ page_table, // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ O_out,
    int M, int n_q_heads, int n_kv_heads, int swa_window, int max_rel) {
    const int wave = threadIdx.x >> 5;  // 0..3 split across KV sequence!
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x;
    const int m = blockIdx.y;
    if (qh >= n_q_heads || m >= M) return;

    const int d0 = lane * 4;
    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int slot = slot_ids[m];
    const int cur_pos = pos[m];
    const int k_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;
    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = (wave == 0 && sink_logit) ? sink_logit[static_cast<size_t>(m) * n_q_heads + qh] : -1e30f;
    float sum_exp = (wave == 0 && sink_logit) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int elems = n_kv_heads * 128;
    const int kv_off = kvh * 128 + d0;
    const float* rb_row = rel_bias ? (rel_bias + (static_cast<size_t>(m) * n_q_heads + qh) * max_rel) : nullptr;
    const int32_t* pt_row = page_table + slot * max_pages_per_slot;

    for (int kp = k_start + wave; kp <= cur_pos; kp += 4) {
        const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
        const int phys = pt_row[ring_p >> 6];
        const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
        const size_t base = tok_base * elems + kv_off;

        float k0, k1, k2, k3, v0, v1, v2, v3;
        if constexpr (FP8_KV) {
            const float ks = k_scales ? k_scales[tok_base] : 1.0f;
            const float vs = v_scales ? v_scales[tok_base] : 1.0f;
            const uint32_t kw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(k_pool) + base);
            const uint32_t vw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(v_pool) + base);
            k0 = e4m3fn_to_f32(static_cast<uint8_t>(kw & 0xFFu)) * ks;
            k1 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 8) & 0xFFu)) * ks;
            k2 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 16) & 0xFFu)) * ks;
            k3 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 24) & 0xFFu)) * ks;
            v0 = e4m3fn_to_f32(static_cast<uint8_t>(vw & 0xFFu)) * vs;
            v1 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 8) & 0xFFu)) * vs;
            v2 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 16) & 0xFFu)) * vs;
            v3 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 24) & 0xFFu)) * vs;
        } else {
            const uint2 kw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(k_pool) + base);
            const uint2 vw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(v_pool) + base);
            h2_unpack(kw.x, k0, k1); h2_unpack(kw.y, k2, k3);
            h2_unpack(vw.x, v0, v1); h2_unpack(vw.y, v2, v3);
        }

        float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const int dist = cur_pos - kp;
        if (rb_row && dist < max_rel) sc += rb_row[dist];

        const float new_max = fmaxf(max_l, sc);
        const float alpha = __expf(max_l - new_max);
        const float beta = __expf(sc - new_max);
        o0 = fmaf(o0, alpha, beta * v0);
        o1 = fmaf(o1, alpha, beta * v1);
        o2 = fmaf(o2, alpha, beta * v2);
        o3 = fmaf(o3, alpha, beta * v3);
        sum_exp = fmaf(sum_exp, alpha, beta);
        max_l = new_max;
    }

    // 1 single __syncthreads() to merge the 4 waves' online-softmax accumulators!
    __shared__ float s_max[4];
    __shared__ float s_sum[4];
    __shared__ alignas(16) float4 s_o[4][32];
    if (lane == 0) { s_max[wave] = max_l; s_sum[wave] = sum_exp; }
    s_o[wave][lane] = make_float4(o0, o1, o2, o3);
    __syncthreads();

    if (wave == 0) {
        const float gmax = fmaxf(fmaxf(s_max[0], s_max[1]), fmaxf(s_max[2], s_max[3]));
        float w0 = (s_sum[0] > 0.0f) ? __expf(s_max[0] - gmax) : 0.0f;
        float w1 = (s_sum[1] > 0.0f) ? __expf(s_max[1] - gmax) : 0.0f;
        float w2 = (s_sum[2] > 0.0f) ? __expf(s_max[2] - gmax) : 0.0f;
        float w3 = (s_sum[3] > 0.0f) ? __expf(s_max[3] - gmax) : 0.0f;
        const float tot_s = w0 * s_sum[0] + w1 * s_sum[1] + w2 * s_sum[2] + w3 * s_sum[3];
        const float inv_s = 1.0f / fmaxf(tot_s, 1e-30f);
        w0 *= inv_s; w1 *= inv_s; w2 *= inv_s; w3 *= inv_s;
        const float4 p0 = s_o[0][lane], p1 = s_o[1][lane], p2 = s_o[2][lane], p3 = s_o[3][lane];
        float r0 = w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x;
        float r1 = w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y;
        float r2 = w0 * p0.z + w1 * p1.z + w2 * p2.z + w3 * p3.z;
        float r3 = w0 * p0.w + w1 * p1.w + w2 * p2.w + w3 * p3.w;
        if (gate_attn != nullptr) {
            const float4 gv = *reinterpret_cast<const float4*>(gate_attn + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
            r0 *= 1.0f / (1.0f + __expf(-gv.x));
            r1 *= 1.0f / (1.0f + __expf(-gv.y));
            r2 *= 1.0f / (1.0f + __expf(-gv.z));
            r3 *= 1.0f / (1.0f + __expf(-gv.w));
        }
        *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0) = make_float4(r0, r1, r2, r3);
    }
}

// 2-Stage FlashDecoding Split-KV Attention for B <= 4 (Saturates ALL 64 CUs on R9700 & 40 CUs on Strix Halo!):
// Stage 1: Grid = (n_q_heads, M, kSplits=4), Block = 256 (8 waves -> 32 parallel KV walkers per head = 768 waves at B=1!)
// Writes tiny partials [M, n_q_heads, kSplits, 130] (o[128], max_l, sum_e) -> 100% L2-resident!
template <bool FP8_KV, int KSPLITS = 4>
__global__ void __launch_bounds__(256) k3_flash_splitkv_part(
    const float* __restrict__ Q,
    const float* __restrict__ rel_bias,
    const float* __restrict__ sink_logit,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ pos,
    const int32_t* __restrict__ page_table,
    int max_pages_per_slot,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ partials,           // [M, n_q_heads, KSPLITS, 130]
    int M, int n_q_heads, int n_kv_heads, int swa_window, int max_rel) {
    const int wave = threadIdx.x >> 5;      // 0..7
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x;
    const int m = blockIdx.y;
    const int sp = blockIdx.z;              // 0..KSPLITS-1
    if (qh >= n_q_heads || m >= M || sp >= KSPLITS) return;

    const int d0 = lane * 4;
    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int slot = slot_ids[m];
    const int cur_pos = pos[m];
    const int k_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;
    const int total_kv = cur_pos - k_start + 1;
    const int chunk = (total_kv + KSPLITS - 1) / KSPLITS;
    const int sp_beg = k_start + sp * chunk;
    const int sp_end = min(cur_pos + 1, sp_beg + chunk);

    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = (sp == 0 && wave == 0 && sink_logit) ? sink_logit[static_cast<size_t>(m) * n_q_heads + qh] : -1e30f;
    float sum_exp = (sp == 0 && wave == 0 && sink_logit) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int elems = n_kv_heads * 128;
    const int kv_off = kvh * 128 + d0;
    const float* rb_row = rel_bias ? (rel_bias + (static_cast<size_t>(m) * n_q_heads + qh) * max_rel) : nullptr;
    const int32_t* pt_row = page_table + slot * max_pages_per_slot;

    for (int kp = sp_beg + wave; kp < sp_end; kp += 8) {
        const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
        const int phys = pt_row[ring_p >> 6];
        const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
        const size_t base = tok_base * elems + kv_off;

        float k0, k1, k2, k3, v0, v1, v2, v3;
        if constexpr (FP8_KV) {
            const float ks = k_scales ? k_scales[tok_base] : 1.0f;
            const float vs = v_scales ? v_scales[tok_base] : 1.0f;
            const uint32_t kw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(k_pool) + base);
            const uint32_t vw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(v_pool) + base);
            k0 = e4m3fn_to_f32(static_cast<uint8_t>(kw & 0xFFu)) * ks;
            k1 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 8) & 0xFFu)) * ks;
            k2 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 16) & 0xFFu)) * ks;
            k3 = e4m3fn_to_f32(static_cast<uint8_t>((kw >> 24) & 0xFFu)) * ks;
            v0 = e4m3fn_to_f32(static_cast<uint8_t>(vw & 0xFFu)) * vs;
            v1 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 8) & 0xFFu)) * vs;
            v2 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 16) & 0xFFu)) * vs;
            v3 = e4m3fn_to_f32(static_cast<uint8_t>((vw >> 24) & 0xFFu)) * vs;
        } else {
            const uint2 kw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(k_pool) + base);
            const uint2 vw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(v_pool) + base);
            h2_unpack(kw.x, k0, k1); h2_unpack(kw.y, k2, k3);
            h2_unpack(vw.x, v0, v1); h2_unpack(vw.y, v2, v3);
        }

        float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const int dist = cur_pos - kp;
        if (rb_row && dist < max_rel) sc += rb_row[dist];

        const float new_max = fmaxf(max_l, sc);
        const float alpha = __expf(max_l - new_max);
        const float beta = __expf(sc - new_max);
        o0 = fmaf(o0, alpha, beta * v0);
        o1 = fmaf(o1, alpha, beta * v1);
        o2 = fmaf(o2, alpha, beta * v2);
        o3 = fmaf(o3, alpha, beta * v3);
        sum_exp = fmaf(sum_exp, alpha, beta);
        max_l = new_max;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ alignas(16) float4 s_o[8][32];
    if (lane == 0) { s_max[wave] = max_l; s_sum[wave] = sum_exp; }
    s_o[wave][lane] = make_float4(o0, o1, o2, o3);
    __syncthreads();

    if (wave == 0) {
        float gmax = -1e30f;
#pragma unroll
        for (int w = 0; w < 8; ++w) gmax = fmaxf(gmax, s_max[w]);
        float tot_s = 0.0f;
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (s_sum[w] > 0.0f) {
                const float wt = __expf(s_max[w] - gmax);
                tot_s += wt * s_sum[w];
                const float4 p = s_o[w][lane];
                r0 = fmaf(wt, p.x, r0); r1 = fmaf(wt, p.y, r1);
                r2 = fmaf(wt, p.z, r2); r3 = fmaf(wt, p.w, r3);
            }
        }
        float* dst = partials + ((static_cast<size_t>(m) * n_q_heads + qh) * KSPLITS + sp) * 130;
        *reinterpret_cast<float4*>(dst + d0) = make_float4(r0, r1, r2, r3);
        if (lane == 0) { dst[128] = gmax; dst[129] = tot_s; }
    }
}

// Stage 2: Merges KSPLITS partials + applies sigmoid(gate_attn) (< 5 us!)
// Grid = (n_q_heads, M), Block = 32 (1 wave = 128 dims!)
template <int KSPLITS = 4>
__global__ void __launch_bounds__(32) k3_flash_splitkv_reduce(
    const float* __restrict__ partials,
    const float* __restrict__ gate_attn,
    float* __restrict__ O_out,
    int M, int n_q_heads) {
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x;
    const int m = blockIdx.y;
    if (qh >= n_q_heads || m >= M) return;
    const int d0 = lane * 4;
    const float* base = partials + (static_cast<size_t>(m) * n_q_heads + qh) * KSPLITS * 130;

    float gmax = -1e30f;
#pragma unroll
    for (int s = 0; s < KSPLITS; ++s) gmax = fmaxf(gmax, base[s * 130 + 128]);
    float tot_s = 0.0f;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
    for (int s = 0; s < KSPLITS; ++s) {
        const float ls = base[s * 130 + 129];
        if (ls > 0.0f) {
            const float wt = __expf(base[s * 130 + 128] - gmax);
            tot_s += wt * ls;
            const float4 p = *reinterpret_cast<const float4*>(base + s * 130 + d0);
            r0 = fmaf(wt, p.x, r0); r1 = fmaf(wt, p.y, r1);
            r2 = fmaf(wt, p.z, r2); r3 = fmaf(wt, p.w, r3);
        }
    }
    const float inv_s = 1.0f / fmaxf(tot_s, 1e-30f);
    r0 *= inv_s; r1 *= inv_s; r2 *= inv_s; r3 *= inv_s;
    if (gate_attn != nullptr) {
        const float4 gv = *reinterpret_cast<const float4*>(gate_attn + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0);
        r0 *= 1.0f / (1.0f + __expf(-gv.x));
        r1 *= 1.0f / (1.0f + __expf(-gv.y));
        r2 *= 1.0f / (1.0f + __expf(-gv.z));
        r3 *= 1.0f / (1.0f + __expf(-gv.w));
    }
    *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d0) = make_float4(r0, r1, r2, r3);
}

// =================================================================================================
// Wave32 WMMA FlashAttention-2 Prefill Kernel (BQ=64, BK=64, head_dim=128) for Single-Slot Prefill!
// Uses hardware wmma_f16 for BOTH S = Q @ K^T and O += P @ V with half-wave 16-lane register online softmax!
// Exact causal mask + SWA-1024 window + pre-projected rel_bias + per-query sink_logit + sigmoid(gate_attn).
// Total LDS = 27.0 KB (< 32 KB -> 2 WG/CU!). Guarded #if INK_HAS_WMMA (zero WMMA emitted on RDNA2!).
// Grid: (n_q_heads, (M + 63) / 64), Block: 128 (4 waves -> 16 Q rows per wave)
// =================================================================================================
template <bool FP8_KV>
__global__ void __launch_bounds__(128) k3_flash_swa_gqa_prefill_wmma(
    const float* __restrict__ Q,            // [M, n_q_heads, 128]
    const float* __restrict__ gate_attn,    // nullptr or [M, n_q_heads, 128]
    const float* __restrict__ rel_bias,     // nullptr or [M, n_q_heads, max_rel]
    const float* __restrict__ sink_logit,   // nullptr or [M, n_q_heads]
    int slot,
    int base_pos,                           // pos[0] (assumes contiguous single-slot pos[m] = base_pos + m!)
    const int32_t* __restrict__ page_table, // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ O_out,
    int M, int n_q_heads, int n_kv_heads, int swa_window, int max_rel) {
#if INK_HAS_WMMA
    constexpr int BQ = 64, BK = 64, HD = 128;
    constexpr int LDK = HD + 8;   // 136 -> 64 * 136 * 2 = 17,408 B
    constexpr int LDV = BK + 8;   //  72 -> 128 * 72 * 2 = 18,432 B (16B-aligned for ds_read_b128 wmma_ld_h!)
    constexpr int LDP = BK + 8;   //  72 -> 64 * 72 * 2  =  9,216 B

    union KvBuf {
        alignas(16) half_t sK[BQ * LDK];   // [64, 136] for Q (init) and K tiles
        alignas(16) half_t sVt[HD * LDV];  // [128, 72] for transposed V tiles
    };
    __shared__ KvBuf kv;
    __shared__ alignas(16) half_t sP[BQ * LDP];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int qh = blockIdx.x;
    const int q_blk = blockIdx.y * BQ;
    if (qh >= n_q_heads || q_blk >= M) return;

    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int elems = n_kv_heads * HD;
    const int kv_off = kvh * HD;
    const int32_t* pt_row = page_table + slot * max_pages_per_slot;
    const float scale = rsqrtf(128.0f);

    // 1. Load Q[64, 128] scaled FP16 into kv.sK once, then keep wave's [16, 128] permanently in registers!
    for (int idx = tid; idx < BQ * (HD / 4); idx += 128) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        const int gm = q_blk + r;
        half_t* dst = &kv.sK[r * LDK + c4 * 4];
        if (gm < M) {
            const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(gm) * n_q_heads + qh) * HD + c4 * 4);
            dst[0] = static_cast<half_t>(qv.x * scale);
            dst[1] = static_cast<half_t>(qv.y * scale);
            dst[2] = static_cast<half_t>(qv.z * scale);
            dst[3] = static_cast<half_t>(qv.w * scale);
        } else {
            dst[0] = static_cast<half_t>(0.0f); dst[1] = static_cast<half_t>(0.0f);
            dst[2] = static_cast<half_t>(0.0f); dst[3] = static_cast<half_t>(0.0f);
        }
    }
    sync_lds();

    wmma_h_t q_frag[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        q_frag[d] = wmma_ld_h(&kv.sK[(wave * 16 + l16) * LDK + d * 16], hi);
    }
    sync_lds();

    // 2. Initialize online softmax state (including per-query sink_logit!) and FP32 O accumulators [8][fv8]
    fv8 o_acc[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) o_acc[d] = fv8_zero();

    float row_max[8];
    float row_sum[8];
    int row_gm[8];
    int row_pos[8];
    const float* rb_ptrs[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int gm = q_blk + wave * 16 + wmma_acc_m(r, lane);
        row_gm[r] = gm;
        row_pos[r] = base_pos + gm;
        rb_ptrs[r] = (rel_bias && gm < M) ? (rel_bias + (static_cast<size_t>(gm) * n_q_heads + qh) * max_rel) : nullptr;
        if (sink_logit && gm < M) {
            row_max[r] = sink_logit[static_cast<size_t>(gm) * n_q_heads + qh];
            row_sum[r] = 1.0f;
        } else {
            row_max[r] = -1e30f;
            row_sum[r] = 0.0f;
        }
    }

    const int q_min_pos = base_pos + q_blk;
    const int q_max_pos = base_pos + min(q_blk + BQ - 1, M - 1);
    const int k_begin = (swa_window > 0 && q_min_pos >= swa_window) ? (q_min_pos - swa_window + 1) : 0;
    const int k_blk_start = (k_begin / BK) * BK;

    // 3. Main FlashAttention-2 loop over KV blocks [kb .. kb + 63]
    for (int kb = k_blk_start; kb <= q_max_pos; kb += BK) {
        // 3A. Load K[64, 128] FP16 into kv.sK[64 * LDK]
        for (int idx = tid; idx < BK * (HD / 4); idx += 128) {
            const int r = idx >> 5;
            const int c4 = idx & 31;
            const int kp = kb + r;
            half_t* dst = &kv.sK[r * LDK + c4 * 4];
            if (kp <= q_max_pos) {
                const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
                const int phys = pt_row[ring_p >> 6];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
                const size_t base = tok_base * elems + kv_off + c4 * 4;
                if constexpr (FP8_KV) {
                    const float ks = k_scales ? k_scales[tok_base] : 1.0f;
                    const uint32_t kw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(k_pool) + base);
                    dst[0] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(kw & 0xFFu)) * ks);
                    dst[1] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 8) & 0xFFu)) * ks);
                    dst[2] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 16) & 0xFFu)) * ks);
                    dst[3] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 24) & 0xFFu)) * ks);
                } else {
                    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(k_pool) + base);
                }
            } else {
                *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
            }
        }
        sync_lds();

        // 3B. Compute S[16, 64] = Q[16, 128] @ K[64, 128]^T via 4x8 hardware wmma_f16!
        fv8 s_acc[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) s_acc[kt] = fv8_zero();
#pragma unroll
        for (int d = 0; d < 8; ++d) {
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                wmma_h_t k_f = wmma_ld_h(&kv.sK[(kt * 16 + l16) * LDK + d * 16], hi);
                s_acc[kt] = wmma_f16(q_frag[d], k_f, s_acc[kt]);
            }
        }
        sync_lds();

        // 3C. Register-only 16-lane online softmax + mask + hoisted rel_bias skip -> write FP16 P[64, 64] to sP!
        const int col16 = wmma_acc_n(0, lane);
        const bool blk_has_rb = (rel_bias != nullptr && (kb + BK - 1) > (q_min_pos - max_rel));
        const bool blk_full_causal = (kb + BK - 1 <= q_min_pos) && (swa_window <= 0 || (q_max_pos - kb < swa_window));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gm = row_gm[r];
            const int qp = row_pos[r];
            const float* rbr = blk_has_rb ? rb_ptrs[r] : nullptr;
            float sv[4];
            float lmax = -1e30f;
            if (blk_full_causal && gm < M && !rbr) {
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const float val = s_acc[kt][r];
                    sv[kt] = val;
                    lmax = fmaxf(lmax, val);
                }
            } else {
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int kp = kb + kt * 16 + col16;
                    const int dist = qp - kp;
                    if (gm < M && kp <= qp && (swa_window <= 0 || dist < swa_window)) {
                        float val = s_acc[kt][r];
                        if (rbr && dist < max_rel) val += rbr[dist];
                        sv[kt] = val;
                        lmax = fmaxf(lmax, val);
                    } else {
                        sv[kt] = -1e30f;
                    }
                }
            }
            // Half-wave 16-lane max reduction (exact per-row max on BOTH RDNA3/3.5 and RDNA4!)
            lmax = fmaxf(lmax, __shfl_xor(lmax, 8, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 4, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 2, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 1, 32));

            const float nm = fmaxf(row_max[r], lmax);
            const float alpha = (row_sum[r] > 0.0f) ? __expf(row_max[r] - nm) : 0.0f;
#pragma unroll
            for (int d = 0; d < 8; ++d) o_acc[d][r] *= alpha;
            row_sum[r] *= alpha;
            row_max[r] = nm;

            float lsum = 0.0f;
            const int local_row = wave * 16 + wmma_acc_m(r, lane);
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const float pv = (sv[kt] > -1e20f) ? __expf(sv[kt] - nm) : 0.0f;
                lsum += pv;
                sP[local_row * LDP + kt * 16 + col16] = static_cast<half_t>(pv);
            }
            lsum += __shfl_xor(lsum, 8, 32);
            lsum += __shfl_xor(lsum, 4, 32);
            lsum += __shfl_xor(lsum, 2, 32);
            lsum += __shfl_xor(lsum, 1, 32);
            row_sum[r] += lsum;
        }

        // 3D. Load V[64, 128] transposed into kv.sVt[128 * LDV] (reusing kv LDS!)
        for (int idx = tid; idx < BK * (HD / 4); idx += 128) {
            const int r = idx >> 5;
            const int c4 = idx & 31;
            const int kp = kb + r;
            const int d0 = c4 * 4;
            if (kp <= q_max_pos) {
                const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
                const int phys = pt_row[ring_p >> 6];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
                const size_t base = tok_base * elems + kv_off + d0;
                if constexpr (FP8_KV) {
                    const float vs = v_scales ? v_scales[tok_base] : 1.0f;
                    const uint32_t vw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(v_pool) + base);
                    kv.sVt[(d0 + 0) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(vw & 0xFFu)) * vs);
                    kv.sVt[(d0 + 1) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 8) & 0xFFu)) * vs);
                    kv.sVt[(d0 + 2) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 16) & 0xFFu)) * vs);
                    kv.sVt[(d0 + 3) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 24) & 0xFFu)) * vs);
                } else {
                    const uint2 vw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(v_pool) + base);
                    const hv2 v01 = __builtin_bit_cast(hv2, vw.x);
                    const hv2 v23 = __builtin_bit_cast(hv2, vw.y);
                    kv.sVt[(d0 + 0) * LDV + r] = v01[0];
                    kv.sVt[(d0 + 1) * LDV + r] = v01[1];
                    kv.sVt[(d0 + 2) * LDV + r] = v23[0];
                    kv.sVt[(d0 + 3) * LDV + r] = v23[1];
                }
            } else {
                kv.sVt[(d0 + 0) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 1) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 2) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 3) * LDV + r] = static_cast<half_t>(0.0f);
            }
        }
        sync_lds();

        // 3E. Compute O[16, 128] += P[16, 64] @ V[64, 128] via 4x8 hardware wmma_f16!
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t p_f = wmma_ld_h(&sP[(wave * 16 + l16) * LDP + kk * 16], hi);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                wmma_h_t v_f = wmma_ld_h(&kv.sVt[(d * 16 + l16) * LDV + kk * 16], hi);
                o_acc[d] = wmma_f16(p_f, v_f, o_acc[d]);
            }
        }
        sync_lds();
    }

    // 4. Normalize by row_sum + apply optional sigmoid(gate_attn) + store O_out[M, n_q_heads, 128]
    const int d_col = wmma_acc_n(0, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int gm = row_gm[r];
        if (gm < M) {
            const float inv_s = 1.0f / fmaxf(row_sum[r], 1e-30f);
            const size_t row_off = (static_cast<size_t>(gm) * n_q_heads + qh) * HD;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const int col = d * 16 + d_col;
                float val = o_acc[d][r] * inv_s;
                if (gate_attn != nullptr) {
                    const float gv = gate_attn[row_off + col];
                    val *= 1.0f / (1.0f + __expf(-gv));
                }
                O_out[row_off + col] = val;
            }
        }
    }
#else
    (void)Q; (void)gate_attn; (void)rel_bias; (void)sink_logit; (void)slot; (void)base_pos;
    (void)page_table; (void)max_pages_per_slot; (void)k_pool; (void)v_pool; (void)k_scales; (void)v_scales;
    (void)O_out; (void)M; (void)n_q_heads; (void)n_kv_heads; (void)swa_window; (void)max_rel;
#endif
}

// =================================================================================================
// k4: Fused MoE Router (Top-K Sigmoid with F32 Bias + route_scale=8.0 Renormalization)
// =================================================================================================

// Grid: (M,), Block: 128 (1 thread per expert e=0..127, exact deterministic tie-break by smallest eid!)
__global__ void __launch_bounds__(128) k4_moe_router_topk(
    const float* __restrict__ logits,  // [M, 128]
    const float* __restrict__ bias,    // nullptr or [128] (F32 gate.bias)
    int32_t* __restrict__ topk_ids,    // [M, K_top]
    float* __restrict__ topk_weights,  // [M, K_top]
    int M, int K_top, float route_scale = 8.0f) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int e = threadIdx.x;
    const int wave = e >> 5;
    const int lane = e & 31;

    __shared__ float s_prob[128];
    __shared__ float s_wmax[4];
    __shared__ int32_t s_wid[4];
    __shared__ int32_t s_chosen[8];

    const float lg = logits[static_cast<size_t>(m) * 128 + e];
    // Inkling-2-Mini exact sigmoid router! (NaN-safe so top-K always picks K_top distinct experts!)
    float p = 1.0f / (1.0f + __expf(-lg));
    if (!(p == p)) p = 0.0f;
    s_prob[e] = p;
    float my_sc = p + (bias ? bias[e] : 0.0f);
    if (!(my_sc == my_sc)) my_sc = -1e29f;
    __syncthreads();

    // 4-Wave parallel deterministic Top-K argmax (strictly greater, smallest eid wins ties -> 8x faster than 1-thread!)
    for (int k = 0; k < K_top; ++k) {
        float bv = my_sc;
        int32_t bi = e;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 32);
            const int32_t oi = __shfl_xor(bi, off, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_wmax[wave] = bv; s_wid[wave] = bi; }
        __syncthreads();
        if (e == 0) {
            float gv = s_wmax[0]; int32_t gi = s_wid[0];
            for (int w = 1; w < 4; ++w) {
                if (s_wmax[w] > gv || (s_wmax[w] == gv && s_wid[w] < gi)) { gv = s_wmax[w]; gi = s_wid[w]; }
            }
            s_chosen[k] = gi;
        }
        __syncthreads();
        if (e == s_chosen[k]) my_sc = -1e30f;
    }
    if (e == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < K_top; ++k) {
            const int best_e = s_chosen[k];
            topk_ids[m * K_top + k] = best_e;
            const float w = s_prob[best_e];
            topk_weights[m * K_top + k] = w;
            wsum += w;
        }
        const float inv_w = route_scale / (wsum + 1e-20f);
        for (int k = 0; k < K_top; ++k) topk_weights[m * K_top + k] *= inv_w;
    }
}

// Wave-6C Fused Router Top-K + FP32->FP16 mlpin conversion (deletes 48 k_f32_to_f16 launches/tok!)
// Grid: (M,), Block: 128
__global__ void __launch_bounds__(128) k4_moe_router_topk_f16(
    const float* __restrict__ logits,  // [M, 128]
    const float* __restrict__ bias,    // nullptr or [128]
    const float* __restrict__ mlpin,   // [M, D]
    half_t* __restrict__ mlpf16,       // [M, D]
    int32_t* __restrict__ topk_ids,    // [M, K_top]
    float* __restrict__ topk_weights,  // [M, K_top]
    int M, int D, int K_top, float route_scale = 8.0f) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int e = threadIdx.x;
    const int wave = e >> 5;
    const int lane = e & 31;

    // Convert mlpin[m, 0..D-1] -> mlpf16[m, 0..D-1] via float4/uint2 coalesced writes!
    const int d4 = D >> 2;
    const float4* src4 = reinterpret_cast<const float4*>(mlpin + static_cast<size_t>(m) * D);
    uint2* dst2 = reinterpret_cast<uint2*>(mlpf16 + static_cast<size_t>(m) * D);
    for (int i = e; i < d4; i += 128) {
        const float4 v = src4[i];
        dst2[i] = make_uint2(h2_pack(v.x, v.y), h2_pack(v.z, v.w));
    }

    __shared__ float s_prob[128];
    __shared__ float s_wmax[4];
    __shared__ int32_t s_wid[4];
    __shared__ int32_t s_chosen[8];

    const float lg = logits[static_cast<size_t>(m) * 128 + e];
    float p = 1.0f / (1.0f + __expf(-lg));
    if (!(p == p)) p = 0.0f;
    s_prob[e] = p;
    float my_sc = p + (bias ? bias[e] : 0.0f);
    if (!(my_sc == my_sc)) my_sc = -1e29f;
    __syncthreads();

    for (int k = 0; k < K_top; ++k) {
        float bv = my_sc;
        int32_t bi = e;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 32);
            const int32_t oi = __shfl_xor(bi, off, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_wmax[wave] = bv; s_wid[wave] = bi; }
        __syncthreads();
        if (e == 0) {
            float gv = s_wmax[0]; int32_t gi = s_wid[0];
            for (int w = 1; w < 4; ++w) {
                if (s_wmax[w] > gv || (s_wmax[w] == gv && s_wid[w] < gi)) { gv = s_wmax[w]; gi = s_wid[w]; }
            }
            s_chosen[k] = gi;
        }
        __syncthreads();
        if (e == s_chosen[k]) my_sc = -1e30f;
    }
    if (e == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < K_top; ++k) {
            const int best_e = s_chosen[k];
            topk_ids[m * K_top + k] = best_e;
            const float w = s_prob[best_e];
            topk_weights[m * K_top + k] = w;
            wsum += w;
        }
        const float inv_w = route_scale / (wsum + 1e-20f);
        for (int k = 0; k < K_top; ++k) topk_weights[m * K_top + k] *= inv_w;
    }
}

// =================================================================================================
// k5: Persistent On-GPU DSpark Block-8 Recurrent Markov Head (markov_rank=256, zero host sync!)
// =================================================================================================
// Exact match to strata::inkling2::dspark_markov_step_f32:
//   joint = [state (rank); markov_w1[prev_token] (rank); hidden (D); context (D)]
//   proj = rnn_proj_w @ joint + rnn_proj_b  (3*rank)
//   gate = sigmoid(proj[0..rank-1]), cand = tanh(proj[rank..2*rank-1]), latent = tanh(proj[2*rank..3*rank-1])
//   state = cand + gate * (state - cand)
//   corr = hidden_out_w @ latent + hidden_out_b  (D)
//   logits[v] = softcap * tanh((base_logits[v] + dot(unembed_w[v], corr)) / softcap)
//   next_token = argmax_v(logits[v])  (strictly greater -> deterministic first-max tie-break!)

// Step A: Update RNN state & compute correction vector `corr[D]` on GPU (1 block of 256 threads!)
__global__ void __launch_bounds__(256) k5_dspark_markov_rnn_update(
    const int32_t* __restrict__ prev_tok_ptr,  // [1] device pointer (eliminates host round-trip!)
    float* __restrict__ state,                 // [rank] updated in-place
    const float* __restrict__ hidden,          // [D]
    const float* __restrict__ context,         // [D]
    const float* __restrict__ markov_w1,       // [V, rank]
    const float* __restrict__ rnn_proj_w,      // [3*rank, 2*rank + 2*D]
    const float* __restrict__ rnn_proj_b,      // [3*rank]
    const float* __restrict__ hidden_out_w,    // [D, rank]
    const float* __restrict__ hidden_out_b,    // [D]
    float* __restrict__ corr_out,              // [D]
    int D, int rank) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int prev_tok = *prev_tok_ptr;
    const int in_dim = 2 * rank + 2 * D;
    const float* mw_row = markov_w1 + static_cast<size_t>(prev_tok) * rank;

    __shared__ float s_state[256];
    __shared__ float s_latent[256];
    if (tid < rank) s_state[tid] = state[tid];
    __syncthreads();

    // 1. Compute 3*rank projections using 8 waves (each wave handles 1 row per iteration)
    for (int r0 = 0; r0 < rank; r0 += 8) {
        const int r = r0 + wave;
        if (r < rank) {
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            const float* w0 = rnn_proj_w + static_cast<size_t>(r) * in_dim;
            const float* w1 = rnn_proj_w + static_cast<size_t>(rank + r) * in_dim;
            const float* w2 = rnn_proj_w + static_cast<size_t>(2 * rank + r) * in_dim;
            for (int i = lane; i < rank; i += 32) {
                const float sv = s_state[i];
                const float mv = mw_row[i];
                d0 = fmaf(w0[i], sv, fmaf(w0[rank + i], mv, d0));
                d1 = fmaf(w1[i], sv, fmaf(w1[rank + i], mv, d1));
                d2 = fmaf(w2[i], sv, fmaf(w2[rank + i], mv, d2));
            }
            for (int i = lane; i < D; i += 32) {
                const float hv = hidden[i];
                const float cv = context[i];
                d0 = fmaf(w0[2 * rank + i], hv, fmaf(w0[2 * rank + D + i], cv, d0));
                d1 = fmaf(w1[2 * rank + i], hv, fmaf(w1[2 * rank + D + i], cv, d1));
                d2 = fmaf(w2[2 * rank + i], hv, fmaf(w2[2 * rank + D + i], cv, d2));
            }
            d0 = wave32_sum(d0) + rnn_proj_b[r];
            d1 = wave32_sum(d1) + rnn_proj_b[rank + r];
            d2 = wave32_sum(d2) + rnn_proj_b[2 * rank + r];
            if (lane == 0) {
                const float gate = 1.0f / (1.0f + __expf(-d0));
                const float cand = tanhf(d1);
                state[r] = fmaf(gate, s_state[r] - cand, cand);
                s_latent[r] = tanhf(d2);
            }
        }
    }
    __syncthreads();

    // 2. corr_out[d] = hidden_out_w[d] @ latent + hidden_out_b[d]
    for (int d = tid; d < D; d += 256) {
        const float* wh = hidden_out_w + static_cast<size_t>(d) * rank;
        float acc = hidden_out_b[d];
        for (int r = 0; r < rank; ++r) acc = fmaf(wh[r], s_latent[r], acc);
        corr_out[d] = acc;
    }
}

// Multi-CU Parallel Markov RNN Step A1: 32 blocks x 8 waves = 256 waves (1 wave per r=0..255!)
// Eliminates single-CU bottleneck on [768, 5120] F32 rnn_proj_w (25x faster than 1-block version!).
__global__ void __launch_bounds__(256) k5_dspark_markov_proj(
    const int32_t* __restrict__ prev_tok_ptr,
    const float* __restrict__ state_in,        // [rank]
    float* __restrict__ state_out,             // [rank] (double-buffered -> zero race!)
    float* __restrict__ latent_out,            // [rank]
    const float* __restrict__ hidden,          // [D]
    const float* __restrict__ context,         // [D]
    const float* __restrict__ markov_w1,       // [V, rank]
    const float* __restrict__ rnn_proj_w,      // [3*rank, 2*rank + 2*D]
    const float* __restrict__ rnn_proj_b,      // [3*rank]
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    if (r >= rank) return;

    const int prev_tok = *prev_tok_ptr;
    const int in_dim = 2 * rank + 2 * D;
    const float* mw_row = markov_w1 + static_cast<size_t>(prev_tok) * rank;
    const float* w0 = rnn_proj_w + static_cast<size_t>(r) * in_dim;
    const float* w1 = rnn_proj_w + static_cast<size_t>(rank + r) * in_dim;
    const float* w2 = rnn_proj_w + static_cast<size_t>(2 * rank + r) * in_dim;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    for (int i = lane * 4; i < rank; i += 128) {
        const float4 sv = *reinterpret_cast<const float4*>(state_in + i);
        const float4 mv = *reinterpret_cast<const float4*>(mw_row + i);
        const float4 a0 = *reinterpret_cast<const float4*>(w0 + i);
        const float4 b0 = *reinterpret_cast<const float4*>(w0 + rank + i);
        const float4 a1 = *reinterpret_cast<const float4*>(w1 + i);
        const float4 b1 = *reinterpret_cast<const float4*>(w1 + rank + i);
        const float4 a2 = *reinterpret_cast<const float4*>(w2 + i);
        const float4 b2 = *reinterpret_cast<const float4*>(w2 + rank + i);
        d0 = fmaf(a0.x, sv.x, fmaf(a0.y, sv.y, fmaf(a0.z, sv.z, fmaf(a0.w, sv.w, d0))));
        d0 = fmaf(b0.x, mv.x, fmaf(b0.y, mv.y, fmaf(b0.z, mv.z, fmaf(b0.w, mv.w, d0))));
        d1 = fmaf(a1.x, sv.x, fmaf(a1.y, sv.y, fmaf(a1.z, sv.z, fmaf(a1.w, sv.w, d1))));
        d1 = fmaf(b1.x, mv.x, fmaf(b1.y, mv.y, fmaf(b1.z, mv.z, fmaf(b1.w, mv.w, d1))));
        d2 = fmaf(a2.x, sv.x, fmaf(a2.y, sv.y, fmaf(a2.z, sv.z, fmaf(a2.w, sv.w, d2))));
        d2 = fmaf(b2.x, mv.x, fmaf(b2.y, mv.y, fmaf(b2.z, mv.z, fmaf(b2.w, mv.w, d2))));
    }
    const int off_h = 2 * rank;
    const int off_c = 2 * rank + D;
    for (int i = lane * 4; i < D; i += 128) {
        const float4 hv = *reinterpret_cast<const float4*>(hidden + i);
        const float4 cv = *reinterpret_cast<const float4*>(context + i);
        const float4 a0 = *reinterpret_cast<const float4*>(w0 + off_h + i);
        const float4 b0 = *reinterpret_cast<const float4*>(w0 + off_c + i);
        const float4 a1 = *reinterpret_cast<const float4*>(w1 + off_h + i);
        const float4 b1 = *reinterpret_cast<const float4*>(w1 + off_c + i);
        const float4 a2 = *reinterpret_cast<const float4*>(w2 + off_h + i);
        const float4 b2 = *reinterpret_cast<const float4*>(w2 + off_c + i);
        d0 = fmaf(a0.x, hv.x, fmaf(a0.y, hv.y, fmaf(a0.z, hv.z, fmaf(a0.w, hv.w, d0))));
        d0 = fmaf(b0.x, cv.x, fmaf(b0.y, cv.y, fmaf(b0.z, cv.z, fmaf(b0.w, cv.w, d0))));
        d1 = fmaf(a1.x, hv.x, fmaf(a1.y, hv.y, fmaf(a1.z, hv.z, fmaf(a1.w, hv.w, d1))));
        d1 = fmaf(b1.x, cv.x, fmaf(b1.y, cv.y, fmaf(b1.z, cv.z, fmaf(b1.w, cv.w, d1))));
        d2 = fmaf(a2.x, hv.x, fmaf(a2.y, hv.y, fmaf(a2.z, hv.z, fmaf(a2.w, hv.w, d2))));
        d2 = fmaf(b2.x, cv.x, fmaf(b2.y, cv.y, fmaf(b2.z, cv.z, fmaf(b2.w, cv.w, d2))));
    }
    d0 = wave32_sum(d0) + rnn_proj_b[r];
    d1 = wave32_sum(d1) + rnn_proj_b[rank + r];
    d2 = wave32_sum(d2) + rnn_proj_b[2 * rank + r];
    if (lane == 0) {
        const float gate = 1.0f / (1.0f + __expf(-d0));
        const float cand = tanhf(d1);
        state_out[r] = fmaf(gate, state_in[r] - cand, cand);
        latent_out[r] = tanhf(d2);
    }
}

// Multi-CU Parallel Markov RNN Step A2: (D+7)/8 blocks x 8 waves -> 1 wave per output dim d! (< 10 us!)
__global__ void __launch_bounds__(256) k5_dspark_markov_corr(
    const float* __restrict__ latent,          // [rank]
    const float* __restrict__ hidden_out_w,    // [D, rank]
    const float* __restrict__ hidden_out_b,    // [D]
    float* __restrict__ corr_out,              // [D]
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = blockIdx.x * 8 + wave;
    if (d >= D) return;
    const float* wh = hidden_out_w + static_cast<size_t>(d) * rank;
    float acc = 0.0f;
    for (int r = lane * 4; r < rank; r += 128) {
        const float4 wv = *reinterpret_cast<const float4*>(wh + r);
        const float4 lv = *reinterpret_cast<const float4*>(latent + r);
        acc = fmaf(wv.x, lv.x, fmaf(wv.y, lv.y, fmaf(wv.z, lv.z, fmaf(wv.w, lv.w, acc))));
    }
    acc = wave32_sum(acc) + hidden_out_b[d];
    if (lane == 0) corr_out[d] = acc;
}

// BF16-weight Multi-CU Parallel Markov RNN Step A1 & A2 (2x less memory bandwidth -> 2x faster markov7x!)
__global__ void __launch_bounds__(256) k5_dspark_markov_proj_bf16(
    const int32_t* __restrict__ prev_tok_ptr,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    float* __restrict__ latent_out,
    const float* __restrict__ hidden,
    const float* __restrict__ context,
    const float* __restrict__ markov_w1,
    const uint16_t* __restrict__ rnn_proj_w,
    const float* __restrict__ rnn_proj_b,
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    if (r >= rank) return;

    const int prev_tok = *prev_tok_ptr;
    const int in_dim = 2 * rank + 2 * D;
    const float* mw_row = markov_w1 + static_cast<size_t>(prev_tok) * rank;
    const uint16_t* w0 = rnn_proj_w + static_cast<size_t>(r) * in_dim;
    const uint16_t* w1 = rnn_proj_w + static_cast<size_t>(rank + r) * in_dim;
    const uint16_t* w2 = rnn_proj_w + static_cast<size_t>(2 * rank + r) * in_dim;

    auto dot4_bf = [](uint2 w4, float4 xv, float acc) -> float {
        acc = fmaf(bf16_to_f32(w4.x & 0xFFFFu), xv.x, acc);
        acc = fmaf(bf16_to_f32(w4.x >> 16),     xv.y, acc);
        acc = fmaf(bf16_to_f32(w4.y & 0xFFFFu), xv.z, acc);
        acc = fmaf(bf16_to_f32(w4.y >> 16),     xv.w, acc);
        return acc;
    };

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    for (int i = lane * 4; i < rank; i += 128) {
        const float4 sv = *reinterpret_cast<const float4*>(state_in + i);
        const float4 mv = *reinterpret_cast<const float4*>(mw_row + i);
        d0 = dot4_bf(*reinterpret_cast<const uint2*>(w0 + i), sv, dot4_bf(*reinterpret_cast<const uint2*>(w0 + rank + i), mv, d0));
        d1 = dot4_bf(*reinterpret_cast<const uint2*>(w1 + i), sv, dot4_bf(*reinterpret_cast<const uint2*>(w1 + rank + i), mv, d1));
        d2 = dot4_bf(*reinterpret_cast<const uint2*>(w2 + i), sv, dot4_bf(*reinterpret_cast<const uint2*>(w2 + rank + i), mv, d2));
    }
    const int off_h = 2 * rank;
    const int off_c = 2 * rank + D;
    for (int i = lane * 4; i < D; i += 128) {
        const float4 hv = *reinterpret_cast<const float4*>(hidden + i);
        const float4 cv = *reinterpret_cast<const float4*>(context + i);
        d0 = dot4_bf(*reinterpret_cast<const uint2*>(w0 + off_h + i), hv, dot4_bf(*reinterpret_cast<const uint2*>(w0 + off_c + i), cv, d0));
        d1 = dot4_bf(*reinterpret_cast<const uint2*>(w1 + off_h + i), hv, dot4_bf(*reinterpret_cast<const uint2*>(w1 + off_c + i), cv, d1));
        d2 = dot4_bf(*reinterpret_cast<const uint2*>(w2 + off_h + i), hv, dot4_bf(*reinterpret_cast<const uint2*>(w2 + off_c + i), cv, d2));
    }
    d0 = wave32_sum(d0) + rnn_proj_b[r];
    d1 = wave32_sum(d1) + rnn_proj_b[rank + r];
    d2 = wave32_sum(d2) + rnn_proj_b[2 * rank + r];
    if (lane == 0) {
        const float gate = 1.0f / (1.0f + __expf(-d0));
        const float cand = tanhf(d1);
        state_out[r] = fmaf(gate, state_in[r] - cand, cand);
        latent_out[r] = tanhf(d2);
    }
}

__global__ void __launch_bounds__(256) k5_dspark_markov_corr_bf16(
    const float* __restrict__ latent,
    const uint16_t* __restrict__ hidden_out_w,
    const float* __restrict__ hidden_out_b,
    float* __restrict__ corr_out,
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = blockIdx.x * 8 + wave;
    if (d >= D) return;
    const uint16_t* wh = hidden_out_w + static_cast<size_t>(d) * rank;
    float acc = 0.0f;
    for (int r = lane * 8; r < rank; r += 256) {
        const uint4 w8 = *reinterpret_cast<const uint4*>(wh + r);
        const float4 l0 = *reinterpret_cast<const float4*>(latent + r);
        const float4 l1 = *reinterpret_cast<const float4*>(latent + r + 4);
        acc = fmaf(bf16_to_f32(w8.x & 0xFFFFu), l0.x, fmaf(bf16_to_f32(w8.x >> 16), l0.y, acc));
        acc = fmaf(bf16_to_f32(w8.y & 0xFFFFu), l0.z, fmaf(bf16_to_f32(w8.y >> 16), l0.w, acc));
        acc = fmaf(bf16_to_f32(w8.z & 0xFFFFu), l1.x, fmaf(bf16_to_f32(w8.z >> 16), l1.y, acc));
        acc = fmaf(bf16_to_f32(w8.w & 0xFFFFu), l1.z, fmaf(bf16_to_f32(w8.w >> 16), l1.w, acc));
    }
    acc = wave32_sum(acc) + hidden_out_b[d];
    if (lane == 0) corr_out[d] = acc;
}

// Step B: Fused unembed bias + softcap + two-stage deterministic first-max argmax across V=200058!
// Stage 1: Grid = (n_blocks,), Block = 256 (8 waves, 1 vocab row per wave -> block reduces to best (val, id))
template <bool BF16_UNEMBED>
__global__ void __launch_bounds__(256) k5_dspark_logits_argmax_stage1(
    const float* __restrict__ base_logits,   // [V]
    const void* __restrict__ unembed_w,      // [V, D]
    const float* __restrict__ corr,          // [D]
    float* __restrict__ out_logits,          // optional [V] (may be nullptr)
    float* __restrict__ blk_max_val,         // [gridDim.x]
    int32_t* __restrict__ blk_max_id,        // [gridDim.x]
    int D, int V, float softcap) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float inv_cap = 1.0f / softcap;

    float best_v = -1e30f;
    int32_t best_i = V;

    const int stride = gridDim.x * 8;
    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float dot = 0.0f;
        if constexpr (BF16_UNEMBED) {
            const uint16_t* wu = static_cast<const uint16_t*>(unembed_w) + static_cast<size_t>(v) * D;
            for (int d = lane; d < D; d += 32) dot = fmaf(bf16_to_f32(wu[d]), corr[d], dot);
        } else {
            const float* wu = static_cast<const float*>(unembed_w) + static_cast<size_t>(v) * D;
            for (int d = lane; d < D; d += 32) dot = fmaf(wu[d], corr[d], dot);
        }
        dot = wave32_sum(dot);
        if (lane == 0) {
            const float val = softcap * tanhf((base_logits[v] + dot) * inv_cap);
            if (out_logits) out_logits[v] = val;
            if (val > best_v || (val == best_v && v < best_i)) {
                best_v = val;
                best_i = v;
            }
        }
    }

    __shared__ float s_val[8];
    __shared__ int32_t s_id[8];
    if (lane == 0) {
        s_val[wave] = best_v;
        s_id[wave] = best_i;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f;
        int32_t bi = V;
        for (int w = 0; w < 8; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_id[w] < bi)) {
                bv = s_val[w];
                bi = s_id[w];
            }
        }
        blk_max_val[blockIdx.x] = bv;
        blk_max_id[blockIdx.x] = bi;
    }
}

__global__ void __launch_bounds__(256) k5_dspark_argmax_stage2(
    const float* __restrict__ blk_max_val,
    const int32_t* __restrict__ blk_max_id,
    int n_blocks,
    int32_t* __restrict__ out_token) {
    if (threadIdx.x != 0) return;
    float bv = -1e30f;
    int32_t bi = 0x7FFFFFFF;
    for (int i = 0; i < n_blocks; ++i) {
        const float v = blk_max_val[i];
        const int32_t id = blk_max_id[i];
        if (v > bv || (v == bv && id < bi)) {
            bv = v;
            bi = id;
        }
    }
    *out_token = bi;
}

// =================================================================================================
// GPU Vision (Inkling-2-Mini-Vision.strata): 23-Layer BF16 ViT LayerNorm + LayerScale + GELU + 2x2 Merge
// =================================================================================================

__global__ void __launch_bounds__(256) k_vit_layernorm_f32(
    const float* __restrict__ x,      // [N, C]
    const float* __restrict__ w,      // [C]
    const float* __restrict__ b,      // [C]
    float* __restrict__ out,          // [N, C]
    int N, int C, float eps) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + static_cast<size_t>(row) * C;
    float* yr = out + static_cast<size_t>(row) * C;
    __shared__ float s_wave[8];
    __shared__ float s_mean, s_inv;

    float sum = 0.0f;
    for (int c = tid; c < C; c += 256) sum += xr[c];
    sum = wave32_sum(sum);
    if (lane == 0) s_wave[wave] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_wave[i];
        s_mean = t / static_cast<float>(C);
    }
    __syncthreads();

    const float mean = s_mean;
    float var = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float d = xr[c] - mean;
        var = fmaf(d, d, var);
    }
    var = wave32_sum(var);
    if (lane == 0) s_wave[wave] = var;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_wave[i];
        s_inv = rsqrtf(t / static_cast<float>(C) + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    for (int c = tid; c < C; c += 256) {
        yr[c] = fmaf((xr[c] - mean) * inv, w[c], b ? b[c] : 0.0f);
    }
}

__global__ void __launch_bounds__(256) k_vit_layerscale_res(
    float* __restrict__ x,          // [N, C] updated in-place: x += ls * delta
    const float* __restrict__ delta,// [N, C]
    const float* __restrict__ ls,   // [C] (ls_1 or ls_2)
    int N, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    x[idx] = fmaf(ls[idx % C], delta[idx], x[idx]);
}

__global__ void __launch_bounds__(256) k_vit_gelu_exact(float* __restrict__ x, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float v = x[idx];
    x[idx] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475244f));
}

// =================================================================================================
// Additional Core Forward, DSpark, and Vision GPU Kernels (zero host round-trips!)
// =================================================================================================

// Batched RMSNorm: out[r, d] = x[r, d] * rsqrt(mean(x^2) + eps) * (w ? w[d] : 1)
// Wave-5G: float4 single-read register-cached fast path (2x less DRAM traffic across 96+ launches!)
// Grid: (rows,), Block: 256
__global__ void __launch_bounds__(256) k_rms_norm_rows(
    const float* __restrict__ x,    // [rows, D]
    const float* __restrict__ w,    // nullptr or [D]
    float* __restrict__ out,        // [rows, D] (may alias x!)
    int rows, int D, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + static_cast<size_t>(r) * D;
    float* yr = out + static_cast<size_t>(r) * D;
    __shared__ float s_w[8];
    __shared__ float s_inv;

    if ((D & 3) == 0 && D <= 4096) {
        const int D4 = D >> 2;
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const float4* w4  = reinterpret_cast<const float4*>(w);
        float4* yr4 = reinterpret_cast<float4*>(yr);
        float4 reg[4];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d4 = tid + i * 256;
            if (d4 < D4) {
                const float4 v = xr4[d4];
                reg[i] = v;
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
        ss = wave32_sum(ss);
        if (lane == 0) s_w[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
            for (int i = 0; i < 8; ++i) t += s_w[i];
            s_inv = rsqrtf(t / static_cast<float>(D) + eps);
        }
        __syncthreads();
        const float inv = s_inv;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d4 = tid + i * 256;
            if (d4 < D4) {
                const float4 v = reg[i];
                if (w4) {
                    const float4 ww = w4[d4];
                    yr4[d4] = make_float4(v.x * inv * ww.x, v.y * inv * ww.y, v.z * inv * ww.z, v.w * inv * ww.w);
                } else {
                    yr4[d4] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
                }
            }
        }
        return;
    }

    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = xr[d];
        ss = fmaf(v, v, ss);
    }
    ss = wave32_sum(ss);
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_inv = rsqrtf(t / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int d = tid; d < D; d += 256) {
        yr[d] = xr[d] * inv * (w ? w[d] : 1.0f);
    }
}

// Elementwise add: out[i] = a[i] + b[i]
__global__ void __launch_bounds__(256) k_vec_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = a[idx] + b[idx];
}

// Split-into-3 copy for AttnRes nq=3 output [M, 3, D] -> in0[M, D], in1[M, D], in2[M, D]
__global__ void __launch_bounds__(256) k_split_attnres3(
    const float* __restrict__ mx,   // [M, 3, D]
    float* __restrict__ in0,        // [M, D]
    float* __restrict__ in1,        // [M, D]
    float* __restrict__ in2,        // [M, D]
    int M, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    const int m = idx / D;
    const int d = idx % D;
    in0[idx] = mx[(static_cast<size_t>(m) * 3 + 0) * D + d];
    in1[idx] = mx[(static_cast<size_t>(m) * 3 + 1) * D + d];
    in2[idx] = mx[(static_cast<size_t>(m) * 3 + 2) * D + d];
}

// Gather rows X_out[t, :] = X_in[tok_idx[t], :] as FP16 for coalesced batched MoE GEMM!
__global__ void __launch_bounds__(256) k_gather_rows_f16(
    const float* __restrict__ X_in,
    const int32_t* __restrict__ tok_idx,
    half_t* __restrict__ X_out,
    int m_rows, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m_rows * D) return;
    const int t = idx / D;
    const int d = idx % D;
    X_out[idx] = static_cast<half_t>(X_in[static_cast<size_t>(tok_idx[t]) * D + d]);
}

// Deterministic scatter-add for a single expert's batched output Y_e[m_rows, D] -> moe_out[B, D]
__global__ void __launch_bounds__(256) k_scatter_add_expert(
    const float* __restrict__ Y_e,
    const int32_t* __restrict__ tok_idx,
    const float* __restrict__ tok_w,
    float* __restrict__ moe_out,
    int m_rows, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    for (int t = 0; t < m_rows; ++t) {
        moe_out[static_cast<size_t>(tok_idx[t]) * D + d] =
            fmaf(Y_e[static_cast<size_t>(t) * D + d], tok_w[t], moe_out[static_cast<size_t>(tok_idx[t]) * D + d]);
    }
}

// Project RelPos bias: rel_bias[m, h, d] = sum_{rc=0..d_rel-1} r[m, h, rc] * rel_proj[rc, d]
// Register-caches r[m, h, 0..15] once + vectorized float4 loads on rel_proj -> 3x faster at P=256..2048!
// Grid: (n_heads, M), Block: 256
__global__ void __launch_bounds__(256) k_relpos_bias_proj(
    const float* __restrict__ r_in,     // [M, n_heads, d_rel]
    const float* __restrict__ rel_proj, // [d_rel, rel_extent]
    const int32_t* __restrict__ pos,    // [M]
    float* __restrict__ rel_bias,       // [M, n_heads, rel_extent]
    int M, int n_heads, int d_rel, int rel_extent) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (h >= n_heads || m >= M) return;
    const int lim = pos ? min(pos[m] + 1, rel_extent) : rel_extent;
    const float* rh = r_in + (static_cast<size_t>(m) * n_heads + h) * d_rel;
    float* dst = rel_bias + (static_cast<size_t>(m) * n_heads + h) * rel_extent;

    float rloc[16];
    const int dr = min(d_rel, 16);
#pragma unroll
    for (int rc = 0; rc < 16; ++rc) rloc[rc] = (rc < dr) ? rh[rc] : 0.0f;

    const int lim4 = lim >> 2;
    const int tid = threadIdx.x;
    if ((rel_extent & 3) == 0 && d_rel <= 16) {
        for (int d4 = tid; d4 < lim4; d4 += 256) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int rc = 0; rc < 16; ++rc) {
                if (rc < dr) {
                    const float rv = rloc[rc];
                    const float4 pv = *reinterpret_cast<const float4*>(rel_proj + static_cast<size_t>(rc) * rel_extent + d4 * 4);
                    a0 = fmaf(rv, pv.x, a0); a1 = fmaf(rv, pv.y, a1);
                    a2 = fmaf(rv, pv.z, a2); a3 = fmaf(rv, pv.w, a3);
                }
            }
            *reinterpret_cast<float4*>(dst + d4 * 4) = make_float4(a0, a1, a2, a3);
        }
        const int d_rem = (lim4 << 2) + tid;
        if (d_rem < lim) {
            float acc = 0.0f;
            for (int rc = 0; rc < dr; ++rc) acc = fmaf(rloc[rc], rel_proj[static_cast<size_t>(rc) * rel_extent + d_rem], acc);
            dst[d_rem] = acc;
        }
    } else {
        for (int d = tid; d < lim; d += 256) {
            float acc = 0.0f;
            for (int rc = 0; rc < d_rel; ++rc) acc = fmaf(rh[rc], rel_proj[static_cast<size_t>(rc) * rel_extent + d], acc);
            dst[d] = acc;
        }
    }
}

// Multi-format embedding lookup + embed_norm (with image embedding bypass!)
// fmt: 102=BF16, 101=F32, 100=Q8_ROW, 0..8=S_K
// Grid: (B,), Block: 256
__global__ void __launch_bounds__(256) k_embed_lookup(
    const uint8_t* __restrict__ emb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ emb_norm_w,
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ img_idx,
    const float* __restrict__ img_embeds,
    float* __restrict__ out,
    int B, int D, float eps) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    float* dst = out + static_cast<size_t>(b) * D;

    if (img_idx && img_idx[b] >= 0 && img_embeds) {
        const float* src = img_embeds + static_cast<size_t>(img_idx[b]) * D;
        for (int d = tid; d < D; d += 256) dst[d] = src[d];
        return;
    }

    const int tok = token_ids[b];
    __shared__ float s_w[8];
    __shared__ float s_inv;
    float ss = 0.0f;

    if (fmt == 102) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(emb_raw) + static_cast<size_t>(tok) * D;
        for (int d = tid; d < D; d += 256) {
            const float v = bf16_to_f32(row[d]);
            dst[d] = v;
            ss = fmaf(v, v, ss);
        }
    } else if (fmt == 101) {
        const float* row = reinterpret_cast<const float*>(emb_raw) + static_cast<size_t>(tok) * D;
        for (int d = tid; d < D; d += 256) {
            const float v = row[d];
            dst[d] = v;
            ss = fmaf(v, v, ss);
        }
    } else if (fmt == 100) {
        const float sc = reinterpret_cast<const float*>(emb_raw)[tok];
        const int8_t* row = reinterpret_cast<const int8_t*>(emb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(tok) * D;
        for (int d = tid; d < D; d += 256) {
            const float v = static_cast<float>(row[d]) * sc;
            dst[d] = v;
            ss = fmaf(v, v, ss);
        }
    } else {
        const int sb_b = sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const uint8_t* row = emb_raw + static_cast<size_t>(tok) * nsb * sb_b;
        for (int d = tid; d < D; d += 256) {
            const uint8_t* blk = row + static_cast<size_t>(d >> 8) * sb_b;
            const float v = decode_sk_w_f32(blk, fmt, d & 255);
            dst[d] = v;
            ss = fmaf(v, v, ss);
        }
    }

    ss = wave32_sum(ss);
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_inv = rsqrtf(t / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int d = tid; d < D; d += 256) {
        dst[d] = dst[d] * inv * emb_norm_w[d];
    }
}

// Multi-format Unembed + softcap(30.0) over vocab_unpadded for up to 16 hidden rows per block in ONE weight pass!
// Wave-6: LDS-cached FP16 superblock tiles (8 KB) + zero-switch S6_K hot loop + 16 tokens/pass -> 2x faster B=16 & 1.5x faster B=1!
// Grid: ((V + 7) / 8, (n_want + 15) / 16), Block: 256 (8 waves = 8 vocab rows; up to 16 tokens share every weight load!)
__global__ void __launch_bounds__(256) k_unembed_softcap(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ h_norm,       // [B, D]
    const int32_t* __restrict__ want_idx,   // [n_want]
    float* __restrict__ logits_out,         // [n_want, V]
    int n_want, int V, int D, float softcap) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int v = blockIdx.x * 8 + wave;
    const int wi0 = blockIdx.y * 16;
    if (wi0 >= n_want) return;
    const int n_loc = (n_want - wi0 < 16) ? (n_want - wi0) : 16;

    const float* hbs[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        hbs[t] = (t < n_loc) ? (h_norm + static_cast<size_t>(want_idx[wi0 + t]) * D) : nullptr;
    }

    float acc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    if (fmt == 102) {
        if (v < V) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 8; k < D; k += 256) {
                const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
                const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
                const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
                const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
                for (int t = 0; t < n_loc; ++t) {
                    const float4 x0 = *reinterpret_cast<const float4*>(hbs[t] + k);
                    const float4 x1 = *reinterpret_cast<const float4*>(hbs[t] + k + 4);
                    acc[t] = fmaf(f0, x0.x, fmaf(f1, x0.y, fmaf(f2, x0.z, fmaf(f3, x0.w, acc[t]))));
                    acc[t] = fmaf(f4, x1.x, fmaf(f5, x1.y, fmaf(f6, x1.z, fmaf(f7, x1.w, acc[t]))));
                }
            }
        }
    } else if (fmt == 101) {
        if (v < V) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
                for (int t = 0; t < n_loc; ++t) {
                    const float4 xv = *reinterpret_cast<const float4*>(hbs[t] + k);
                    acc[t] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[t]))));
                }
            }
        }
    } else if (fmt == 100) {
        if (v < V) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
                const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
                const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
                const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
                for (int t = 0; t < n_loc; ++t) {
                    const float4 xv = *reinterpret_cast<const float4*>(hbs[t] + k);
                    acc[t] = fmaf(f0, xv.x, fmaf(f1, xv.y, fmaf(f2, xv.z, fmaf(f3, xv.w, acc[t]))));
                }
            }
            for (int t = 0; t < n_loc; ++t) acc[t] *= sc;
        }
    } else {
        // S_K (especially S6_K on SQ-10/SQ-15/SQ-20!): cache each 256-elem superblock of h_norm as FP16 in LDS ONCE for all 8 waves!
        __shared__ alignas(16) half_t sH[16 * 256];  // 8,192 B
        const int sb_b = sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const uint8_t* wr = (v < V) ? (unemb_raw + static_cast<size_t>(v) * nsb * sb_b) : nullptr;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        if (fmt == CODE_S6_K) {
            for (int s = 0; s < nsb; ++s) {
                const int sbase = s << 8;
                for (int t = 0; t < n_loc; ++t) {
                    sH[t * 256 + tid] = static_cast<half_t>(hbs[t][sbase + tid]);
                }
                __syncthreads();
                if (wr) {
                    const uint8_t* blk = wr + static_cast<size_t>(s) * 198;
                    const float eff = decode_sk_sub_eff_i8(blk, CODE_S6_K, sub);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S6_K, j0);
                    for (int t = 0; t < n_loc; ++t) {
                        const uint4 hv = *reinterpret_cast<const uint4*>(&sH[t * 256 + j0]);
                        float d = fdot2_f32(wv.x, hv.x, 0.0f);
                        d = fdot2_f32(wv.y, hv.y, d);
                        d = fdot2_f32(wv.z, hv.z, d);
                        d = fdot2_f32(wv.w, hv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
                __syncthreads();
            }
        } else {
            for (int s = 0; s < nsb; ++s) {
                const int sbase = s << 8;
                for (int t = 0; t < n_loc; ++t) {
                    sH[t * 256 + tid] = static_cast<half_t>(hbs[t][sbase + tid]);
                }
                __syncthreads();
                if (wr) {
                    const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                    const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                    for (int t = 0; t < n_loc; ++t) {
                        const uint4 hv = *reinterpret_cast<const uint4*>(&sH[t * 256 + j0]);
                        float d = fdot2_f32(wv.x, hv.x, 0.0f);
                        d = fdot2_f32(wv.y, hv.y, d);
                        d = fdot2_f32(wv.z, hv.z, d);
                        d = fdot2_f32(wv.w, hv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
                __syncthreads();
            }
        }
    }

    if (v >= V) return;
    const float inv_cap = 1.0f / softcap;
    for (int t = 0; t < n_loc; ++t) {
        const float z = wave32_sum(acc[t]);
        if (lane == 0) {
            logits_out[static_cast<size_t>(wi0 + t) * V + v] = softcap * tanhf(z * inv_cap);
        }
    }
}

// Multi-format DSpark combined unembed argmax stage 1: computes dot(unemb[v], combined) for all V, deterministic first-max!
// Uses LDS-cached FP16 combined vector + fast 32-bit S_K unpack + fdot2_f32 for 3x faster Markov drafting!
__global__ void __launch_bounds__(256) k5_dspark_packed_argmax_stage1(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ cur_step,    // [D]
    const float* __restrict__ corr,        // [D]
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_id,
    int D, int V) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ alignas(16) float s_comb[2304];
    __shared__ alignas(16) half_t s_comb_h[2304];
    for (int d = tid; d < D; d += 256) {
        const float c = cur_step[d] + corr[d];
        s_comb[d] = c;
        s_comb_h[d] = static_cast<half_t>(c);
    }
    __syncthreads();

    float best_v = -1e30f;
    int32_t best_i = V;
    const int stride = gridDim.x * 8;

    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc = 0.0f;
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint2 w4 = *reinterpret_cast<const uint2*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(s_comb + k);
                acc = fmaf(bf16_to_f32(w4.x & 0xFFFFu), cv.x, acc);
                acc = fmaf(bf16_to_f32(w4.x >> 16), cv.y, acc);
                acc = fmaf(bf16_to_f32(w4.y & 0xFFFFu), cv.z, acc);
                acc = fmaf(bf16_to_f32(w4.y >> 16), cv.w, acc);
            }
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(s_comb + k);
                acc = fmaf(wv.x, cv.x, fmaf(wv.y, cv.y, fmaf(wv.z, cv.z, fmaf(wv.w, cv.w, acc))));
            }
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(s_comb + k);
                acc = fmaf(static_cast<float>(static_cast<int8_t>(w4 & 0xFFu)), cv.x, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu)), cv.y, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu)), cv.z, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu)), cv.w, acc);
            }
            acc *= sc;
        } else {
            const int sb_b = sk_sb_bytes(fmt);
            const int nsb = D >> 8;
            const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const int j0 = lane * 8;
                const float eff = decode_sk_sub_eff_i8(blk, fmt, j0 >> 6);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                const uint4 xv = *reinterpret_cast<const uint4*>(s_comb_h + (s << 8) + j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        }
        const float z = wave32_sum(acc);
        if (lane == 0) {
            if (z > best_v || (z == best_v && v < best_i)) {
                best_v = z;
                best_i = v;
            }
        }
    }

    __shared__ float s_val[8];
    __shared__ int32_t s_id[8];
    if (lane == 0) { s_val[wave] = best_v; s_id[wave] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f;
        int32_t bi = V;
        for (int w = 0; w < 8; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_id[w] < bi)) { bv = s_val[w]; bi = s_id[w]; }
        }
        blk_max_val[blockIdx.x] = bv;
        blk_max_id[blockIdx.x] = bi;
    }
}

// Single-Pass Multi-Step DSpark Base Unembed Top-2 Candidate Screen (7x LESS UNEMBED BANDWIDTH!!):
// Reads unembed[V, D] ONCE across ALL n_draft (<= 7) cur[1..n_draft] rows simultaneously!
// Grid: (512,), Block: 256 (8 waves/block) -> writes top-2 (val, id) per block = 1,024 top candidates per step!
__global__ void __launch_bounds__(256) k5_dspark_base_top2_screen(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ cur_steps,    // [n_draft, D] (cur + 1 * D)
    float* __restrict__ cand_val,           // [n_draft, gridDim.x * 2]
    int32_t* __restrict__ cand_id,          // [n_draft, gridDim.x * 2]
    int n_draft, int D, int V) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    // Union keeps total LDS = 32,256 B <= 32 KB -> 2 WG/CU occupancy (2x vs 33,152 B)!
    // Caches FP16 for S_K AND BF16 for SQ-20 BF16 unembed -> ZERO global reads of cur_steps in inner loop!
    union ScreenLds {
        half_t   s_cur_h[7 * 2304];
        uint16_t s_cur_bf[7 * 2304];
        struct {
            float   v[7 * 16];
            int32_t i[7 * 16];
        } red;
    };
    __shared__ alignas(16) ScreenLds u;
    if (fmt < 100) {
        const int tot = n_draft * D;
        for (int idx = tid; idx < tot; idx += 256) {
            u.s_cur_h[idx] = static_cast<half_t>(cur_steps[idx]);
        }
        __syncthreads();
    } else if (fmt == 102) {
        const int tot = n_draft * D;
        for (int idx = tid; idx < tot; idx += 256) {
            u.s_cur_bf[idx] = f32_to_bf16(cur_steps[idx]);
        }
        __syncthreads();
    }

    float best1_v[7], best2_v[7];
    int32_t best1_i[7], best2_i[7];
#pragma unroll
    for (int t = 0; t < 7; ++t) {
        best1_v[t] = -1e30f; best2_v[t] = -1e30f;
        best1_i[t] = V;      best2_i[t] = V;
    }

    const int stride = gridDim.x * 8;
    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc[7] = {0, 0, 0, 0, 0, 0, 0};
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 8; k < D; k += 256) {
                const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
                const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
                const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
                const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n_draft) {
                        const uint4 c8 = *reinterpret_cast<const uint4*>(u.s_cur_bf + t * 2304 + k);
                        acc[t] = fmaf(f0, bf16_to_f32(c8.x & 0xFFFFu), fmaf(f1, bf16_to_f32(c8.x >> 16),
                                 fmaf(f2, bf16_to_f32(c8.y & 0xFFFFu), fmaf(f3, bf16_to_f32(c8.y >> 16), acc[t]))));
                        acc[t] = fmaf(f4, bf16_to_f32(c8.z & 0xFFFFu), fmaf(f5, bf16_to_f32(c8.z >> 16),
                                 fmaf(f6, bf16_to_f32(c8.w & 0xFFFFu), fmaf(f7, bf16_to_f32(c8.w >> 16), acc[t]))));
                    }
                }
            }
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n_draft) {
                        const float4 cv = *reinterpret_cast<const float4*>(cur_steps + static_cast<size_t>(t) * D + k);
                        acc[t] = fmaf(wv.x, cv.x, fmaf(wv.y, cv.y, fmaf(wv.z, cv.z, fmaf(wv.w, cv.w, acc[t]))));
                    }
                }
            }
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
                const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
                const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
                const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n_draft) {
                        const float4 cv = *reinterpret_cast<const float4*>(cur_steps + static_cast<size_t>(t) * D + k);
                        acc[t] = fmaf(f0, cv.x, fmaf(f1, cv.y, fmaf(f2, cv.z, fmaf(f3, cv.w, acc[t]))));
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < 7; ++t) acc[t] *= sc;
        } else if (fmt == CODE_S6_K) {
            const int nsb = D >> 8;
            const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * 198;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * 198;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S6_K, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S6_K, j0);
                const int base = (s << 8) + j0;
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n_draft) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(u.s_cur_h + t * 2304 + base);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d);
                        d = fdot2_f32(wv.z, xv.z, d);
                        d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        } else {
            const int sb_b = sk_sb_bytes(fmt);
            const int nsb = D >> 8;
            const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                const int base = (s << 8) + j0;
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n_draft) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(u.s_cur_h + t * 2304 + base);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d);
                        d = fdot2_f32(wv.z, xv.z, d);
                        d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            if (t < n_draft) {
                const float z = wave32_sum(acc[t]);
                if (lane == 0) {
                    if (z > best1_v[t] || (z == best1_v[t] && v < best1_i[t])) {
                        best2_v[t] = best1_v[t]; best2_i[t] = best1_i[t];
                        best1_v[t] = z;          best1_i[t] = v;
                    } else if (z > best2_v[t] || (z == best2_v[t] && v < best2_i[t])) {
                        best2_v[t] = z;          best2_i[t] = v;
                    }
                }
            }
        }
    }

    __syncthreads();
    if (lane == 0) {
        for (int t = 0; t < n_draft; ++t) {
            u.red.v[t * 16 + wave * 2 + 0] = best1_v[t];
            u.red.i[t * 16 + wave * 2 + 0] = best1_i[t];
            u.red.v[t * 16 + wave * 2 + 1] = best2_v[t];
            u.red.i[t * 16 + wave * 2 + 1] = best2_i[t];
        }
    }
    __syncthreads();
    if (tid < n_draft) {
        const int t = tid;
        float b1 = -1e30f, b2 = -1e30f;
        int32_t i1 = V, i2 = V;
        for (int k = 0; k < 16; ++k) {
            const float val = u.red.v[t * 16 + k];
            const int32_t id = u.red.i[t * 16 + k];
            if (val > b1 || (val == b1 && id < i1)) {
                b2 = b1; i2 = i1;
                b1 = val; i1 = id;
            } else if (val > b2 || (val == b2 && id < i2)) {
                b2 = val; i2 = id;
            }
        }
        const int out_stride = gridDim.x * 2;
        cand_val[t * out_stride + blockIdx.x * 2 + 0] = b1;
        cand_id[t * out_stride + blockIdx.x * 2 + 0]  = i1;
        cand_val[t * out_stride + blockIdx.x * 2 + 1] = b2;
        cand_id[t * out_stride + blockIdx.x * 2 + 1]  = i2;
    }
}

// Fast 1,024-Candidate Markov Argmax Stage 1: scores ONLY the shortlisted candidates (< 0.02 ms/step!)
// Grid: ((n_cand + 7) / 8,), Block: 256 (8 waves = 8 candidates/block)
__global__ void __launch_bounds__(256) k5_dspark_cand_argmax_stage1(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ cand_base_val, // [n_cand]
    const int32_t* __restrict__ cand_id,     // [n_cand]
    const float* __restrict__ corr,          // [D]
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_id,
    int n_cand, int D, int V) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ alignas(16) float s_corr[2304];
    __shared__ alignas(16) half_t s_corr_h[2304];
    for (int d = tid; d < D; d += 256) {
        const float c = corr[d];
        s_corr[d] = c;
        s_corr_h[d] = static_cast<half_t>(c);
    }
    __syncthreads();

    const int ci = blockIdx.x * 8 + wave;
    float my_v = -1e30f;
    int32_t my_i = V;
    if (ci < n_cand) {
        const int v = cand_id[ci];
        if (v >= 0 && v < V) {
            float acc = 0.0f;
            if (fmt == 102) {
                const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
                for (int k = lane * 8; k < D; k += 256) {
                    const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                    const float4 c0 = *reinterpret_cast<const float4*>(s_corr + k);
                    const float4 c1 = *reinterpret_cast<const float4*>(s_corr + k + 4);
                    acc = fmaf(bf16_to_f32(w8.x & 0xFFFFu), c0.x, fmaf(bf16_to_f32(w8.x >> 16), c0.y, acc));
                    acc = fmaf(bf16_to_f32(w8.y & 0xFFFFu), c0.z, fmaf(bf16_to_f32(w8.y >> 16), c0.w, acc));
                    acc = fmaf(bf16_to_f32(w8.z & 0xFFFFu), c1.x, fmaf(bf16_to_f32(w8.z >> 16), c1.y, acc));
                    acc = fmaf(bf16_to_f32(w8.w & 0xFFFFu), c1.z, fmaf(bf16_to_f32(w8.w >> 16), c1.w, acc));
                }
            } else if (fmt == 101) {
                const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
                for (int k = lane * 4; k < D; k += 128) {
                    const float4 wv = *reinterpret_cast<const float4*>(wr + k);
                    const float4 cv = *reinterpret_cast<const float4*>(s_corr + k);
                    acc = fmaf(wv.x, cv.x, fmaf(wv.y, cv.y, fmaf(wv.z, cv.z, fmaf(wv.w, cv.w, acc))));
                }
            } else if (fmt == 100) {
                const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
                const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
                for (int k = lane * 4; k < D; k += 128) {
                    const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                    const float4 cv = *reinterpret_cast<const float4*>(s_corr + k);
                    acc = fmaf(static_cast<float>(static_cast<int8_t>(w4 & 0xFFu)), cv.x, acc);
                    acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu)), cv.y, acc);
                    acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu)), cv.z, acc);
                    acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu)), cv.w, acc);
                }
                acc *= sc;
            } else {
                const int sb_b = sk_sb_bytes(fmt);
                const int nsb = D >> 8;
                const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
                for (int s = 0; s < nsb; ++s) {
                    const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                    const int j0 = lane * 8;
                    const float eff = decode_sk_sub_eff_i8(blk, fmt, j0 >> 6);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                    const uint4 xv = *reinterpret_cast<const uint4*>(s_corr_h + (s << 8) + j0);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d);
                    d = fdot2_f32(wv.z, xv.z, d);
                    d = fdot2_f32(wv.w, xv.w, d);
                    acc = fmaf(d, eff, acc);
                }
            }
            my_v = cand_base_val[ci] + wave32_sum(acc);
            my_i = v;
        }
    }

    __shared__ float s_val[8];
    __shared__ int32_t s_id[8];
    if (lane == 0) { s_val[wave] = my_v; s_id[wave] = my_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f;
        int32_t bi = V;
        for (int w = 0; w < 8; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_id[w] < bi)) { bv = s_val[w]; bi = s_id[w]; }
        }
        blk_max_val[blockIdx.x] = bv;
        blk_max_id[blockIdx.x] = bi;
    }
}

// Half-RoPE (used by DSpark context K and draft Q/K): grid (n_heads, M), block (head_dim/2)
__global__ void __launch_bounds__(128) k_half_rope(
    float* __restrict__ x,          // [M, n_heads, head_dim]
    const int32_t* __restrict__ pos,// nullptr -> uses base_pos + m
    int base_pos, int M, int n_heads, int head_dim, float theta) {
    const int j = threadIdx.x;
    const int half = head_dim >> 1;
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (j >= half || h >= n_heads || m >= M) return;
    const int p = pos ? pos[m] : (base_pos + m);
    const float freq = 1.0f / powf(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
    const float ang = static_cast<float>(p) * freq;
    float s, c;
    __sincosf(ang, &s, &c);
    float* xh = x + (static_cast<size_t>(m) * n_heads + h) * head_dim;
    const float a = xh[j], b = xh[j + half];
    xh[j]        = a * c - b * s;
    xh[j + half] = a * s + b * c;
}

// DSpark 4-tap fuse + RMSNorm: out[m, d] = rms(sum_{i=0..3} w[i]*tap[i][m, d]) * norm_w[d]
__global__ void __launch_bounds__(256) k_dspark_fuse_mix(
    const float* __restrict__ t0,
    const float* __restrict__ t1,
    const float* __restrict__ t2,
    const float* __restrict__ t3,
    float w0, float w1, float w2, float w3,
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int M, int D, float eps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const size_t off = static_cast<size_t>(m) * D;
    float* dst = out + off;
    __shared__ float s_w[8];
    __shared__ float s_inv;
    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = w0 * t0[off + d] + w1 * t1[off + d] + w2 * t2[off + d] + w3 * t3[off + d];
        dst[d] = v;
        ss = fmaf(v, v, ss);
    }
    ss = wave32_sum(ss);
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_inv = rsqrtf(t / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int d = tid; d < D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

// DSpark Non-Causal Block-8 Attention (Zero-Barrier Wave32: 1 wave per (hh, qi), 4 dims/lane, 4 heads/block!)
// Grid: ((H + 3) / 4, B), Block: 128 -> ZERO LDS, ZERO __syncthreads(), coalesced uint2 BF16 / float4 loads!
__global__ void __launch_bounds__(128) k_dspark_block_attn(
    const float* __restrict__ Q,          // [B, H, 128]
    const uint16_t* __restrict__ ctx_K,   // [max_ctx, KV, 128] BF16
    const uint16_t* __restrict__ ctx_V,   // [max_ctx, KV, 128] BF16
    const float* __restrict__ drf_K,      // [B, KV, 128] F32
    const float* __restrict__ drf_V,      // [B, KV, 128] F32
    float* __restrict__ O_out,            // [B, H, 128]
    int B, int H, int KV, int c0, int c_len) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.x * 4 + wave;
    const int qi = blockIdx.y;
    if (hh >= H || qi >= B) return;
    const int d0 = lane * 4;
    const int kvh = hh / (H / KV);
    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(qi) * H + hh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = -1e30f;
    float sum_e = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int tot = c_len + B;
    for (int t = 0; t < tot; ++t) {
        float k0, k1, k2, k3, v0, v1, v2, v3;
        if (t < c_len) {
            const size_t idx = (static_cast<size_t>(c0 + t) * KV + kvh) * 128 + d0;
            const uint2 kw = *reinterpret_cast<const uint2*>(ctx_K + idx);
            const uint2 vw = *reinterpret_cast<const uint2*>(ctx_V + idx);
            k0 = bf16_to_f32(kw.x & 0xFFFFu); k1 = bf16_to_f32(kw.x >> 16);
            k2 = bf16_to_f32(kw.y & 0xFFFFu); k3 = bf16_to_f32(kw.y >> 16);
            v0 = bf16_to_f32(vw.x & 0xFFFFu); v1 = bf16_to_f32(vw.x >> 16);
            v2 = bf16_to_f32(vw.y & 0xFFFFu); v3 = bf16_to_f32(vw.y >> 16);
        } else {
            const int di = t - c_len;
            const size_t idx = (static_cast<size_t>(di) * KV + kvh) * 128 + d0;
            const float4 kw = *reinterpret_cast<const float4*>(drf_K + idx);
            const float4 vw = *reinterpret_cast<const float4*>(drf_V + idx);
            k0 = kw.x; k1 = kw.y; k2 = kw.z; k3 = kw.w;
            v0 = vw.x; v1 = vw.y; v2 = vw.z; v3 = vw.w;
        }
        const float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const float nm = fmaxf(max_l, sc);
        const float a = __expf(max_l - nm);
        const float b = __expf(sc - nm);
        o0 = fmaf(o0, a, b * v0); o1 = fmaf(o1, a, b * v1);
        o2 = fmaf(o2, a, b * v2); o3 = fmaf(o3, a, b * v3);
        sum_e = fmaf(sum_e, a, b);
        max_l = nm;
    }
    const float inv = 1.0f / fmaxf(sum_e, 1e-30f);
    *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(qi) * H + hh) * 128 + d0) =
        make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv);
}

// DSpark SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void __launch_bounds__(256) k_swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// Wave-6E Graph-Friendly Fused DSpark Kernels (reads anchor_pos from device scalar -> 1-launch HIP Graph!)
// Fused split(QKV) + Q/K RMSNorm + Half-RoPE: Grid (H + 2*KV, B), Block 128 (assumes hd <= 128)
__global__ void __launch_bounds__(128) k_dspark_split_qkv_norm_rope(
    const float* __restrict__ qkv,         // [B, q_dim + 2*kv_dim]
    const float* __restrict__ q_norm_w,    // [hd]
    const float* __restrict__ k_norm_w,    // [hd]
    const int32_t* __restrict__ anchor_ptr,// [1] device anchor_pos
    float* __restrict__ Q_out,             // [B, H, hd]
    float* __restrict__ K_out,             // [B, KV, hd]
    float* __restrict__ V_out,             // [B, KV, hd]
    int B, int H, int KV, int hd, float eps, float theta) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= B || tid >= hd) return;
    const int q_dim = H * hd;
    const int kv_dim = KV * hd;
    const float* row = qkv + static_cast<size_t>(b) * (q_dim + 2 * kv_dim);

    if (h >= H + KV) {
        const int vh = h - H - KV;
        if (vh < KV) {
            V_out[(static_cast<size_t>(b) * KV + vh) * hd + tid] = row[q_dim + kv_dim + vh * hd + tid];
        }
        return;
    }

    const bool is_q = (h < H);
    const float val = is_q ? row[h * hd + tid] : row[q_dim + (h - H) * hd + tid];
    const float nw  = is_q ? q_norm_w[tid] : k_norm_w[tid];
    const int wave = tid >> 5;
    const int lane = tid & 31;
    float ss = wave32_sum(val * val);
    __shared__ float s_w[4];
    __shared__ float s_x[128];
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    const int n_waves = (hd + 31) >> 5;
    float tot = 0.0f;
    for (int w = 0; w < n_waves; ++w) tot += s_w[w];
    const float inv = rsqrtf(tot / static_cast<float>(hd) + eps);
    s_x[tid] = val * inv * nw;
    __syncthreads();

    const int half = hd >> 1;
    if (tid < half) {
        const int p = anchor_ptr[0] + b;
        const float freq = 1.0f / powf(theta, static_cast<float>(2 * tid) / static_cast<float>(hd));
        float s, c;
        __sincosf(static_cast<float>(p) * freq, &s, &c);
        const float a = s_x[tid], bv = s_x[tid + half];
        float* dst = is_q ? (Q_out + (static_cast<size_t>(b) * H + h) * hd)
                          : (K_out + (static_cast<size_t>(b) * KV + (h - H)) * hd);
        dst[tid]        = a * c - bv * s;
        dst[tid + half] = a * s + bv * c;
    }
}

// Dynamic-position DSpark Non-Causal Block-8 Attention (reads anchor_pos from device scalar -> 100% HIP-Graphable!)
// Grid: ((H + 3) / 4, B), Block: 128
__global__ void __launch_bounds__(128) k_dspark_block_attn_dyn(
    const float* __restrict__ Q,           // [B, H, 128]
    const uint16_t* __restrict__ ctx_K,    // [max_ctx, KV, 128] BF16
    const uint16_t* __restrict__ ctx_V,    // [max_ctx, KV, 128] BF16
    const float* __restrict__ drf_K,       // [B, KV, 128] F32
    const float* __restrict__ drf_V,       // [B, KV, 128] F32
    const int32_t* __restrict__ anchor_ptr,// [1]
    float* __restrict__ O_out,             // [B, H, 128]
    int B, int H, int KV, int swa_win) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.x * 4 + wave;
    const int qi = blockIdx.y;
    if (hh >= H || qi >= B) return;
    const int anchor_pos = anchor_ptr[0];
    const int c0 = (swa_win > 0 && anchor_pos >= swa_win) ? (anchor_pos - swa_win + 1) : 0;
    const int c_len = anchor_pos - c0;

    const int d0 = lane * 4;
    const int kvh = hh / (H / KV);
    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(qi) * H + hh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = -1e30f;
    float sum_e = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int tot = c_len + B;
    for (int t = 0; t < tot; ++t) {
        float k0, k1, k2, k3, v0, v1, v2, v3;
        if (t < c_len) {
            const size_t idx = (static_cast<size_t>(c0 + t) * KV + kvh) * 128 + d0;
            const uint2 kw = *reinterpret_cast<const uint2*>(ctx_K + idx);
            const uint2 vw = *reinterpret_cast<const uint2*>(ctx_V + idx);
            k0 = bf16_to_f32(kw.x & 0xFFFFu); k1 = bf16_to_f32(kw.x >> 16);
            k2 = bf16_to_f32(kw.y & 0xFFFFu); k3 = bf16_to_f32(kw.y >> 16);
            v0 = bf16_to_f32(vw.x & 0xFFFFu); v1 = bf16_to_f32(vw.x >> 16);
            v2 = bf16_to_f32(vw.y & 0xFFFFu); v3 = bf16_to_f32(vw.y >> 16);
        } else {
            const int di = t - c_len;
            const size_t idx = (static_cast<size_t>(di) * KV + kvh) * 128 + d0;
            const float4 kw = *reinterpret_cast<const float4*>(drf_K + idx);
            const float4 vw = *reinterpret_cast<const float4*>(drf_V + idx);
            k0 = kw.x; k1 = kw.y; k2 = kw.z; k3 = kw.w;
            v0 = vw.x; v1 = vw.y; v2 = vw.z; v3 = vw.w;
        }
        const float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const float nm = fmaxf(max_l, sc);
        const float a = __expf(max_l - nm);
        const float b = __expf(sc - nm);
        o0 = fmaf(o0, a, b * v0); o1 = fmaf(o1, a, b * v1);
        o2 = fmaf(o2, a, b * v2); o3 = fmaf(o3, a, b * v3);
        sum_e = fmaf(sum_e, a, b);
        max_l = nm;
    }
    const float inv = 1.0f / fmaxf(sum_e, 1e-30f);
    *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(qi) * H + hh) * 128 + d0) =
        make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv);
}

// Fused SwiGLU on concatenated [gate(inter); up(inter)] rows: Grid ((B*inter+255)/256), Block 256
__global__ void __launch_bounds__(256) k_swiglu_split_f32(
    const float* __restrict__ gu,  // [B, 2 * inter]
    float* __restrict__ out,       // [B, inter]
    int B, int inter) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * inter) return;
    const int b = idx / inter;
    const int j = idx - b * inter;
    const float g = gu[static_cast<size_t>(b) * (2 * inter) + j];
    const float u = gu[static_cast<size_t>(b) * (2 * inter) + inter + j];
    out[idx] = (g / (1.0f + __expf(-g))) * u;
}

// Fused dynamic f_rnn[anchor_pos-1] load + Markov state zero: Grid ((D+255)/256), Block 256
__global__ void __launch_bounds__(256) k_dspark_load_frnn_zero(
    const uint16_t* __restrict__ frnn_base,
    const int32_t* __restrict__ anchor_ptr,
    float* __restrict__ dst_frnn,
    float* __restrict__ dst_state,
    int D, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        const int pos = anchor_ptr[0] - 1;
        dst_frnn[i] = bf16_to_f32(frnn_base[static_cast<size_t>(pos) * D + i]);
    }
    if (i < rank) dst_state[i] = 0.0f;
}

// Vision ViT Pairwise Axial 2D RoPE on Q and K: grid (H, S), block (hd/2)
__global__ void __launch_bounds__(64) k_vit_rope_2d(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ rope_cos,  // [S, hd]
    const float* __restrict__ rope_sin,  // [S, hd]
    int S, int H, int hd) {
    const int pair = threadIdx.x;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (pair * 2 + 1 >= hd || h >= H || t >= S) return;
    const int j = pair * 2;
    const float rc0 = rope_cos[static_cast<size_t>(t) * hd + j];
    const float rc1 = rope_cos[static_cast<size_t>(t) * hd + j + 1];
    const float rs0 = rope_sin[static_cast<size_t>(t) * hd + j];
    const float rs1 = rope_sin[static_cast<size_t>(t) * hd + j + 1];
    float* qh = Q + (static_cast<size_t>(t) * H + h) * hd;
    float* kh = K + (static_cast<size_t>(t) * H + h) * hd;
    const float q0 = qh[j], q1 = qh[j + 1];
    qh[j]     = q0 * rc0 - q1 * rs0;
    qh[j + 1] = q1 * rc1 + q0 * rs1;
    const float k0 = kh[j], k1 = kh[j + 1];
    kh[j]     = k0 * rc0 - k1 * rs0;
    kh[j + 1] = k1 * rc1 + k0 * rs1;
}

// Vision ViT Full Non-Causal Self-Attention (hd=64, H=16): Grid (H, S), Block 64 (2 waves!)
__global__ void __launch_bounds__(64) k_vit_full_attn(
    const float* __restrict__ Q,   // [S, H, 64]
    const float* __restrict__ K,   // [S, H, 64]
    const float* __restrict__ V,   // [S, H, 64]
    float* __restrict__ O,         // [S, H, 64]
    int S, int H, int hd) {
    const int h = blockIdx.x;
    const int qi = blockIdx.y;
    if (h >= H || qi >= S) return;
    const int d = threadIdx.x;  // 0..63
    const int wave = d >> 5;
    const int lane = d & 31;
    const float scale = rsqrtf(static_cast<float>(hd));
    const float qv = Q[(static_cast<size_t>(qi) * H + h) * hd + d] * scale;

    __shared__ float s_w[2];
    __shared__ float s_sc;
    float max_l = -1e30f;
    float sum_e = 0.0f;
    float o_acc = 0.0f;

    for (int ki = 0; ki < S; ++ki) {
        const size_t idx = (static_cast<size_t>(ki) * H + h) * hd + d;
        const float dot = wave32_sum(qv * K[idx]);
        if (lane == 0) s_w[wave] = dot;
        __syncthreads();
        if (d == 0) s_sc = s_w[0] + s_w[1];
        __syncthreads();
        const float sc = s_sc;
        const float nm = fmaxf(max_l, sc);
        const float a = __expf(max_l - nm);
        const float b = __expf(sc - nm);
        o_acc = fmaf(o_acc, a, b * V[idx]);
        sum_e = fmaf(sum_e, a, b);
        max_l = nm;
    }
    O[(static_cast<size_t>(qi) * H + h) * hd + d] = o_acc / fmaxf(sum_e, 1e-30f);
}

}  // namespace strata::inkling2::hip
