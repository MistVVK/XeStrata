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

    __shared__ float s_wave[8];
    __shared__ float s_scores[16];
    __shared__ float s_inv_rms[16];

    // Pass 1: compute RMSNorm inv_rms and dot(norm(state[s]), qv) for each depth state s
    for (int s = 0; s < n_states; ++s) {
        const float* sv = (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                                         : (cur + static_cast<size_t>(m) * D);
        float ss = 0.0f;
        float dot = 0.0f;
        for (int d = tid; d < D; d += 256) {
            const float x = sv[d];
            ss = fmaf(x, x, ss);
            dot = fmaf(x, qv[d], dot);
        }
        ss = wave32_sum(ss);
        dot = wave32_sum(dot);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        float total_ss = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < 8; ++w) total_ss += s_wave[w];
            s_inv_rms[s] = rsqrtf(total_ss / static_cast<float>(D) + kRmsEps);
        }
        __syncthreads();
        if (lane == 0) s_wave[wave] = dot;
        __syncthreads();
        if (tid == 0) {
            float total_dot = 0.0f;
            for (int w = 0; w < 8; ++w) total_dot += s_wave[w];
            s_scores[s] = total_dot * s_inv_rms[s] * inv_sqrt_d;
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

    // Pass 2: weighted sum across RMS-normalized depth states (matches strata::inkling2::attn_res_mix_f32!)
    float* dst = out + (static_cast<size_t>(m) * n_queries + q) * D;
    for (int d = tid; d < D; d += 256) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            const float* sv = (s < n_blocks) ? (bp.p[s] + static_cast<size_t>(m) * D)
                                             : (cur + static_cast<size_t>(m) * D);
            acc = fmaf(s_scores[s], sv[d], acc);
        }
        dst[d] = acc;
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

// Flash-SWA / Flash-GQA Attention Kernel with RelPos Bias + Per-Query Attention Sinks + Sigmoid Gate!
// Grid: (n_q_heads, M), Block: 128 (4 waves = 128 threads, 1 thread per head_dim element d=0..127!)
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
    const int qh = blockIdx.x;
    const int m = blockIdx.y;
    if (qh >= n_q_heads || m >= M) return;

    const int d = threadIdx.x;  // 0..127 (head_dim = 128!)
    const int wave = d >> 5;
    const int lane = d & 31;
    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int slot = slot_ids[m];
    const int cur_pos = pos[m];
    const int k_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;
    const float scale = rsqrtf(128.0f);
    const float q_val = Q[(static_cast<size_t>(m) * n_q_heads + qh) * 128 + d] * scale;

    __shared__ float s_wave[4];
    __shared__ float s_score;

    // Initialize online softmax state with per-query attention sink (acts as virtual zero-value token!)
    float max_l = sink_logit ? sink_logit[static_cast<size_t>(m) * n_q_heads + qh] : -1e30f;
    float sum_exp = sink_logit ? 1.0f : 0.0f;
    float o_acc = 0.0f;

    const int elems = n_kv_heads * 128;
    const int kv_off = kvh * 128 + d;
    const float* rb_row = rel_bias ? (rel_bias + (static_cast<size_t>(m) * n_q_heads + qh) * max_rel) : nullptr;

    for (int kp = k_start; kp <= cur_pos; ++kp) {
        const int ring_p = (swa_window > 0) ? (kp % swa_window) : kp;
        const int phys = page_table[slot * max_pages_per_slot + (ring_p >> 6)];
        const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
        const size_t base = tok_base * elems + kv_off;

        float k_val, v_val;
        if constexpr (FP8_KV) {
            const float ks = k_scales ? k_scales[tok_base] : 1.0f;
            const float vs = v_scales ? v_scales[tok_base] : 1.0f;
            k_val = e4m3fn_to_f32(static_cast<const uint8_t*>(k_pool)[base]) * ks;
            v_val = e4m3fn_to_f32(static_cast<const uint8_t*>(v_pool)[base]) * vs;
        } else {
            k_val = static_cast<float>(static_cast<const half_t*>(k_pool)[base]);
            v_val = static_cast<float>(static_cast<const half_t*>(v_pool)[base]);
        }

        float dot = wave32_sum(q_val * k_val);
        if (lane == 0) s_wave[wave] = dot;
        __syncthreads();
        if (d == 0) {
            float sc = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
            const int dist = cur_pos - kp;
            if (rb_row && dist < max_rel) sc += rb_row[dist];
            s_score = sc;
        }
        __syncthreads();

        const float sc = s_score;
        const float new_max = fmaxf(max_l, sc);
        const float alpha = __expf(max_l - new_max);
        const float beta = __expf(sc - new_max);
        o_acc = fmaf(o_acc, alpha, beta * v_val);
        sum_exp = fmaf(sum_exp, alpha, beta);
        max_l = new_max;
    }

    float res = o_acc / fmaxf(sum_exp, 1e-30f);
    if (gate_attn != nullptr) {
        const float g = gate_attn[(static_cast<size_t>(m) * n_q_heads + qh) * 128 + d];
        res *= 1.0f / (1.0f + __expf(-g));
    }
    O_out[(static_cast<size_t>(m) * n_q_heads + qh) * 128 + d] = res;
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

    __shared__ float s_prob[128];
    __shared__ float s_score[128];

    const float lg = logits[static_cast<size_t>(m) * 128 + e];
    // Inkling-2-Mini exact sigmoid router!
    const float p = 1.0f / (1.0f + __expf(-lg));
    s_prob[e] = p;
    s_score[e] = p + (bias ? bias[e] : 0.0f);
    __syncthreads();

    if (e == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < K_top; ++k) {
            int best_e = 0;
            float best_s = -1e30f;
            for (int i = 0; i < 128; ++i) {
                if (s_score[i] > best_s) {  // strictly greater scanning 0..127 -> smallest eid wins ties!
                    best_s = s_score[i];
                    best_e = i;
                }
            }
            topk_ids[m * K_top + k] = best_e;
            const float w = s_prob[best_e];
            topk_weights[m * K_top + k] = w;
            wsum += w;
            s_score[best_e] = -1e30f;
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

    __shared__ float s_latent[256];

    // 1. Compute 3*rank projections using 8 waves (each wave handles 1 row per iteration)
    for (int r0 = 0; r0 < rank; r0 += 8) {
        const int r = r0 + wave;
        if (r < rank) {
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            const float* w0 = rnn_proj_w + static_cast<size_t>(r) * in_dim;
            const float* w1 = rnn_proj_w + static_cast<size_t>(rank + r) * in_dim;
            const float* w2 = rnn_proj_w + static_cast<size_t>(2 * rank + r) * in_dim;
            for (int i = lane; i < rank; i += 32) {
                const float sv = state[i];
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
                state[r] = fmaf(gate, state[r] - cand, cand);
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
// Only computes d in [0 .. active_extent-1] where active_extent = min(pos[m]+1, rel_extent) -> up to 16x faster!
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
    for (int d = threadIdx.x; d < lim; d += 256) {
        float acc = 0.0f;
        for (int rc = 0; rc < d_rel; ++rc) {
            acc = fmaf(rh[rc], rel_proj[static_cast<size_t>(rc) * rel_extent + d], acc);
        }
        dst[d] = acc;
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

// Multi-format Unembed + softcap(30.0) over vocab_unpadded for `n_want` hidden rows!
// Grid: ((V + 7) / 8, n_want), Block: 256 (8 waves = 8 vocab rows!)
__global__ void __launch_bounds__(256) k_unembed_softcap(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ h_norm,       // [B, D]
    const int32_t* __restrict__ want_idx,   // [n_want]
    float* __restrict__ logits_out,         // [n_want, V]
    int n_want, int V, int D, float softcap) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + wave;
    const int wi = blockIdx.y;
    if (v >= V || wi >= n_want) return;

    const float* hb = h_norm + static_cast<size_t>(want_idx[wi]) * D;
    float acc = 0.0f;

    if (fmt == 102) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const uint2 w4 = *reinterpret_cast<const uint2*>(wr + k);
            const float4 xv = *reinterpret_cast<const float4*>(hb + k);
            acc = fmaf(bf16_to_f32(w4.x & 0xFFFFu), xv.x, acc);
            acc = fmaf(bf16_to_f32(w4.x >> 16), xv.y, acc);
            acc = fmaf(bf16_to_f32(w4.y & 0xFFFFu), xv.z, acc);
            acc = fmaf(bf16_to_f32(w4.y >> 16), xv.w, acc);
        }
    } else if (fmt == 101) {
        const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(wr + k);
            const float4 xv = *reinterpret_cast<const float4*>(hb + k);
            acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
        }
    } else if (fmt == 100) {
        const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
        const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
            const float4 xv = *reinterpret_cast<const float4*>(hb + k);
            acc = fmaf(static_cast<float>(static_cast<int8_t>(w4 & 0xFFu)), xv.x, acc);
            acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu)), xv.y, acc);
            acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu)), xv.z, acc);
            acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu)), xv.w, acc);
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
            const uint8_t* pl = blk + 6;
            const float* xp = hb + (s << 8) + j0;
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                d = fmaf(static_cast<float>(decode_sk_w_i8(pl, fmt, j0 + i)), xp[i], d);
            }
            acc = fmaf(d, eff, acc);
        }
    }

    const float z = wave32_sum(acc);
    if (lane == 0) {
        logits_out[static_cast<size_t>(wi) * V + v] = softcap * tanhf(z / softcap);
    }
}

// Multi-format DSpark combined unembed argmax stage 1: computes dot(unemb[v], combined) for all V, deterministic first-max!
// (Exploits exact identity: softcap(z) is strictly monotone AND base_logits + unembed@corr == unembed@(cur+corr)!)
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

    float best_v = -1e30f;
    int32_t best_i = V;
    const int stride = gridDim.x * 8;

    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc = 0.0f;
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint2 w4 = *reinterpret_cast<const uint2*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(cur_step + k);
                const float4 dv = *reinterpret_cast<const float4*>(corr + k);
                acc = fmaf(bf16_to_f32(w4.x & 0xFFFFu), cv.x + dv.x, acc);
                acc = fmaf(bf16_to_f32(w4.x >> 16), cv.y + dv.y, acc);
                acc = fmaf(bf16_to_f32(w4.y & 0xFFFFu), cv.z + dv.z, acc);
                acc = fmaf(bf16_to_f32(w4.y >> 16), cv.w + dv.w, acc);
            }
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(cur_step + k);
                const float4 dv = *reinterpret_cast<const float4*>(corr + k);
                acc = fmaf(wv.x, cv.x + dv.x, fmaf(wv.y, cv.y + dv.y, fmaf(wv.z, cv.z + dv.z, fmaf(wv.w, cv.w + dv.w, acc))));
            }
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float4 cv = *reinterpret_cast<const float4*>(cur_step + k);
                const float4 dv = *reinterpret_cast<const float4*>(corr + k);
                acc = fmaf(static_cast<float>(static_cast<int8_t>(w4 & 0xFFu)), cv.x + dv.x, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu)), cv.y + dv.y, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu)), cv.z + dv.z, acc);
                acc = fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu)), cv.w + dv.w, acc);
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
                const uint8_t* pl = blk + 6;
                const int base = (s << 8) + j0;
                float d = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    d = fmaf(static_cast<float>(decode_sk_w_i8(pl, fmt, j0 + i)), cur_step[base + i] + corr[base + i], d);
                }
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

// DSpark Non-Causal Block-8 Attention (attending to ctx [c0..anchor_pos-1] in BF16 + B=8 draft tokens in F32)
// Grid: (n_heads, B), Block: 128 (head_dim=128)
__global__ void __launch_bounds__(128) k_dspark_block_attn(
    const float* __restrict__ Q,          // [B, H, 128]
    const uint16_t* __restrict__ ctx_K,   // [max_ctx, KV, 128] BF16
    const uint16_t* __restrict__ ctx_V,   // [max_ctx, KV, 128] BF16
    const float* __restrict__ drf_K,      // [B, KV, 128] F32
    const float* __restrict__ drf_V,      // [B, KV, 128] F32
    float* __restrict__ O_out,            // [B, H, 128]
    int B, int H, int KV, int c0, int c_len) {
    const int hh = blockIdx.x;
    const int qi = blockIdx.y;
    if (hh >= H || qi >= B) return;
    const int d = threadIdx.x;
    const int wave = d >> 5;
    const int lane = d & 31;
    const int kvh = hh / (H / KV);
    const float scale = rsqrtf(128.0f);
    const float qv = Q[(static_cast<size_t>(qi) * H + hh) * 128 + d] * scale;

    __shared__ float s_wave[4];
    __shared__ float s_sc;
    float max_l = -1e30f;
    float sum_e = 0.0f;
    float o_acc = 0.0f;

    const int tot = c_len + B;
    for (int t = 0; t < tot; ++t) {
        float kv, vv;
        if (t < c_len) {
            const size_t idx = (static_cast<size_t>(c0 + t) * KV + kvh) * 128 + d;
            kv = bf16_to_f32(ctx_K[idx]);
            vv = bf16_to_f32(ctx_V[idx]);
        } else {
            const int di = t - c_len;
            const size_t idx = (static_cast<size_t>(di) * KV + kvh) * 128 + d;
            kv = drf_K[idx];
            vv = drf_V[idx];
        }
        float dot = wave32_sum(qv * kv);
        if (lane == 0) s_wave[wave] = dot;
        __syncthreads();
        if (d == 0) s_sc = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
        __syncthreads();
        const float sc = s_sc;
        const float nm = fmaxf(max_l, sc);
        const float a = __expf(max_l - nm);
        const float b = __expf(sc - nm);
        o_acc = fmaf(o_acc, a, b * vv);
        sum_e = fmaf(sum_e, a, b);
        max_l = nm;
    }
    O_out[(static_cast<size_t>(qi) * H + hh) * 128 + d] = o_acc / fmaxf(sum_e, 1e-30f);
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
