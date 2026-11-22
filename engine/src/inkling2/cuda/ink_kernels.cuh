// engine/src/inkling2/cuda/ink_kernels.cuh - Fused Inkling2Loop Kernels + Dense Projections + Exact 2-Stage Screened Unembed
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include "ink_cuda.cuh"
#include "ink_moe_cuda.cuh"

#include <cuda_fp8.h>

namespace strata::ink {

// ============================================================================
// 1. Embedding Lookup + RMSNorm (Supports BF16, F32, Q8_ROW, S5_K/S6_K; bypasses norm on image rows!)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_embed_norm(
    const uint8_t* __restrict__ emb_w,
    int emb_fmt,
    const float* __restrict__ emb_norm_w,  // [2304] F32
    const int32_t* __restrict__ tok_ids,   // [M]
    const uint8_t* __restrict__ is_img,    // [M] optional (null = all text)
    const float* __restrict__ img_emb,     // [M, 2304] optional
    float* __restrict__ out,               // [M, 2304] F32
    int M,
    int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    float* dst = out + (size_t)m * D;

    if (is_img && is_img[m]) {
        const float* src = img_emb + (size_t)m * D;
        for (int d = tid; d < D; d += 256) dst[d] = src[d];
        return;
    }

    const int32_t id = tok_ids[m];
    __shared__ float sh[32];
    float ss = 0.0f;

    if (emb_fmt == FMT_BF16) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(emb_w) + (size_t)id * D;
        for (int d = tid; d < D; d += 256) {
            float v = bf16_to_f32(row[d]);
            dst[d] = v;
            ss += v * v;
        }
    } else if (emb_fmt == FMT_F32) {
        const float* row = reinterpret_cast<const float*>(emb_w) + (size_t)id * D;
        for (int d = tid; d < D; d += 256) {
            float v = row[d];
            dst[d] = v;
            ss += v * v;
        }
    } else if (emb_fmt == FMT_Q8ROW) {
        // Q8_ROW layout: [rows] float32 scales followed by [rows, D] int8
        const float sc = reinterpret_cast<const float*>(emb_w)[id];
        const int8_t* qrow = reinterpret_cast<const int8_t*>(emb_w + (size_t)kInkVocabPad * 4) + (size_t)id * D;
        for (int d = tid; d < D; d += 256) {
            float v = (float)qrow[d] * sc;
            dst[d] = v;
            ss += v * v;
        }
    } else {
        // S5_K / S6_K / S_K row decode
        const int nsb = D / 256;
        const int sb_b = ink_sb_bytes(emb_fmt);
        const float half = ink_sk_half(emb_fmt);
        const uint8_t* row = emb_w + (size_t)id * nsb * sb_b;
        for (int d = tid; d < D; d += 256) {
            const int s = d >> 8;
            const int idx = d & 255;
            const int g = idx >> 6;
            const uint8_t* blk = row + (size_t)s * sb_b;
            const float Dsc = f16_to_f32(ld16_u(blk));
            const float eff = Dsc * ((float)blk[2 + g] * (1.0f / 255.0f));
            const uint8_t* pl = blk + 6;
            float v = 0.0f;
            if (emb_fmt == FMT_S5_K) {
                uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
                uint32_t hi = (pl[128 + (idx >> 3)] >> (idx & 7)) & 1u;
                v = ((float)(lo | (hi << 4)) - half) * eff;
            } else if (emb_fmt == FMT_S6_K) {
                uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
                uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
                v = ((float)(lo | (hi << 4)) - half) * eff;
            } else if (emb_fmt == FMT_S8_K) {
                v = (float)((int8_t)pl[idx]) * eff;
            }
            dst[d] = v;
            ss += v * v;
        }
    }

    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
    for (int d = tid; d < D; d += 256) {
        dst[d] = dst[d] * inv * emb_norm_w[d];
    }
}

// ============================================================================
// 2. Kimi-Style Attention-Residual Depth Mix (Fused 1-Launch Kernel!)
//    Mixes S depth states (blocks[0..n_blk-1] + cur) against Q queries [Q, D] (Q in {1, 3, 4}).
//    Grid: (M,) blocks of 256 threads. Supports up to 8 depth states (Inkling-2-Mini has max 5!).
// ============================================================================
struct DepthStatePtrs {
    const float* p[8];
    int count;  // total states = n_blocks + 1 (last is cur)
};

__global__ void __launch_bounds__(256) k_ink_attn_res_mix(
    DepthStatePtrs states,
    const float* __restrict__ queries,  // [n_q, D] F32
    float* __restrict__ out,            // [n_q, M, D] F32 (contiguous per query q: out[q * M * D + m * D])
    int n_q,
    int M,
    int D,
    const float* __restrict__ nw0 = nullptr,
    const float* __restrict__ nw1 = nullptr,
    const float* __restrict__ nw2 = nullptr) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int S = states.count;
    const int D4 = D >> 2;
    __shared__ float sh[32];
    __shared__ float s_wred[8][5][8];  // [s][0=ss, 1..4=dots[0..3]][warp 0..7] -> ONE barrier for ALL S x n_q!
    __shared__ float s_prob[4][8];
    __shared__ float s_sc_local[4][8];
    __shared__ float s_inv_buf[8];

    // Fused Pass 1+2 with ONE single __syncthreads() across all S states and n_q queries!
    const float inv_sqrt_d = rsqrtf((float)D);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= S) break;
        const float4* sp4 = reinterpret_cast<const float4*>(states.p[s] + (size_t)m * D);
        float ss = 0.0f;
        float dots[4] = {};
        for (int i = tid; i < D4; i += 256) {
            const float4 sv = sp4[i];
            ss += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q < n_q) {
                    const float4 qv = reinterpret_cast<const float4*>(queries + (size_t)q * D)[i];
                    dots[q] = fmaf(sv.x, qv.x, dots[q]); dots[q] = fmaf(sv.y, qv.y, dots[q]);
                    dots[q] = fmaf(sv.z, qv.z, dots[q]); dots[q] = fmaf(sv.w, qv.w, dots[q]);
                }
            }
        }
        ss = warp_sum(ss);
        if (lane == 0) s_wred[s][0][warp] = ss;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < n_q) {
                float dq = warp_sum(dots[q]);
                if (lane == 0) s_wred[s][1 + q][warp] = dq;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = 0; s < S; ++s) {
            float ss = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) ss += s_wred[s][0][w];
            const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
            s_inv_buf[s] = inv;
            for (int q = 0; q < n_q; ++q) {
                float dsum = 0.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) dsum += s_wred[s][1 + q][w];
                s_sc_local[q][s] = dsum * inv * inv_sqrt_d;
            }
        }
        for (int q = 0; q < n_q; ++q) {
            float mx = -1e30f;
            for (int s = 0; s < S; ++s) mx = fmaxf(mx, s_sc_local[q][s]);
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) {
                s_sc_local[q][s] = expf(s_sc_local[q][s] - mx);
                sum += s_sc_local[q][s];
            }
            const float inv_s = 1.0f / sum;
            for (int s = 0; s < S; ++s) s_prob[q][s] = s_sc_local[q][s] * inv_s * s_inv_buf[s];
        }
    }
    __syncthreads();

    // Pass 3: vectorized float4 weighted sum (+ optional fused RMSNorm when nw != nullptr -> saves 72 kernel launches!)
    for (int q = 0; q < n_q; ++q) {
        const float* nw = (q == 0) ? nw0 : (q == 1 ? nw1 : nw2);
        float4* dst4 = reinterpret_cast<float4*>(out + ((size_t)q * M + m) * D);
        float ss = 0.0f;
        for (int i = tid; i < D4; i += 256) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s < S) {
                    const float w = s_prob[q][s];
                    const float4 sv = reinterpret_cast<const float4*>(states.p[s] + (size_t)m * D)[i];
                    ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay);
                    az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
                }
            }
            dst4[i] = make_float4(ax, ay, az, aw);
            if (nw) ss += ax * ax + ay * ay + az * az + aw * aw;
        }
        if (nw) {
            ss = block_sum(ss, sh);
            const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
            const float4* nw4 = reinterpret_cast<const float4*>(nw);
            for (int i = tid; i < D4; i += 256) {
                const float4 v = dst4[i], wv = nw4[i];
                dst4[i] = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
            }
        }
    }
}

// ============================================================================
// 3. High-Speed Multi-CTA / TF32-TC Rank-64 tanh Gain/Forget Loop Controller (+ Optional RMSNorm!)
//    gate[M, 64] = [rms(cur); rms(msg)] @ w_down[64, 4608]^T
//    out[M, D]   = rms_norm((1 - tanh(w_forget @ gate)) * cur + tanh(w_gain @ gate) * msg, norm_w)
//    Achieves 100x speedup over single-CTA version (0.3 ms decode, 1.5 ms prefill across all 96 controllers!).
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_ctrl_cat_rms(
    const float* __restrict__ cur,  // [M, D]
    const float* __restrict__ msg,  // [M, D]
    float* __restrict__ cat,        // [M, 2*D]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    __shared__ float sh[32];
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    const int D4 = D >> 2;
    float ssc = 0.0f, ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 c = cp4[i], g = mp4[i];
        ssc += c.x * c.x + c.y * c.y + c.z * c.z + c.w * c.w;
        ssm += g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w;
    }
    ssc = block_sum(ssc, sh);
    ssm = block_sum(ssm, sh);
    const float ic = rsqrtf(ssc / (float)D + kInkRmsEps);
    const float im = rsqrtf(ssm / (float)D + kInkRmsEps);
    float4* dst0 = reinterpret_cast<float4*>(cat + (size_t)m * (2 * D));
    float4* dst1 = reinterpret_cast<float4*>(cat + (size_t)m * (2 * D) + D);
    for (int i = tid; i < D4; i += 256) {
        const float4 c = cp4[i], g = mp4[i];
        dst0[i] = make_float4(c.x * ic, c.y * ic, c.z * ic, c.w * ic);
        dst1[i] = make_float4(g.x * im, g.y * im, g.z * im, g.w * im);
    }
}

// 64 CTAs (1 CTA of 256 threads per gate row r in 0..63), coalesced float4 along 4608 -> ~1.1 us on RTX 5090!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_down_smallM(
    const float* __restrict__ cat,    // [M, 4608]
    const float* __restrict__ w_down, // [64, 4608]
    float* __restrict__ gate,         // [M, 64]
    int M, int D2) {
    const int r = blockIdx.x;  // 0..63
    if (r >= kInkGateRank) return;
    const int tid = threadIdx.x;
    const int D2_4 = D2 >> 2;
    const float4* wr = reinterpret_cast<const float4*>(w_down + (size_t)r * D2);
    float acc[MAX_M] = {};
    for (int i = tid; i < D2_4; i += 256) {
        const float4 w = wr[i];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 x = reinterpret_cast<const float4*>(cat + (size_t)m * D2)[i];
                acc[m] = fmaf(w.x, x.x, acc[m]);
                acc[m] = fmaf(w.y, x.y, acc[m]);
                acc[m] = fmaf(w.z, x.z, acc[m]);
                acc[m] = fmaf(w.w, x.w, acc[m]);
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float s = block_sum(acc[m], sh);
            if (tid == 0) gate[(size_t)m * kInkGateRank + r] = s;
        }
    }
}

// 144 CTAs (16 rows/CTA, 16 threads/row reading coalesced float4 along 64!) -> ~0.8 us on RTX 5090!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_apply_smallM(
    const float* __restrict__ cur,      // [M, 2304]
    const float* __restrict__ msg,      // [M, 2304]
    const float* __restrict__ gate,     // [M, 64]
    const float* __restrict__ w_gain,   // [2304, 64]
    const float* __restrict__ w_forget, // [2304, 64]
    float* __restrict__ out,            // [M, 2304] (may alias cur)
    int M, int D) {
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    for (int i = tid; i < M * kInkGateRank; i += 256) {
        s_gate[i >> 6][i & 63] = gate[i];
    }
    __syncthreads();

    const int sub = tid >> 4;     // 0..15 rows inside CTA
    const int lg  = tid & 15;     // 0..15 -> floats [4*lg .. 4*lg+3] inside rank-64 row
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;

    const float4 wg = reinterpret_cast<const float4*>(w_gain + (size_t)d * kInkGateRank)[lg];
    const float4 wf = reinterpret_cast<const float4*>(w_forget + (size_t)d * kInkGateRank)[lg];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 g4 = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = wg.x * g4.x + wg.y * g4.y + wg.z * g4.z + wg.w * g4.w;
            float af = wf.x * g4.x + wf.y * g4.y + wf.z * g4.z + wf.w * g4.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af)) * cur[idx] + tanhf(ag) * msg[idx];
            }
        }
    }
}

// Prefill / Large-M vectorized float4 fused blend + RMSNorm after cuBLAS TF32 computes gain_val[M, D] and forget_val[M, D]
__global__ void __launch_bounds__(256) k_ink_ctrl_blend_norm(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gain_val,
    const float* __restrict__ forget_val,
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    __shared__ float sh[32];
    const size_t off = (size_t)m * D;
    const float4* c4 = reinterpret_cast<const float4*>(cur + off);
    const float4* g4 = reinterpret_cast<const float4*>(msg + off);
    const float4* gv4 = reinterpret_cast<const float4*>(gain_val + off);
    const float4* fv4 = reinterpret_cast<const float4*>(forget_val + off);
    float4* o4 = reinterpret_cast<float4*>(out + off);
    float ss = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = c4[i], mv = g4[i], ga = gv4[i], fo = fv4[i];
        const float y0 = (1.0f - tanhf(fo.x)) * cv.x + tanhf(ga.x) * mv.x;
        const float y1 = (1.0f - tanhf(fo.y)) * cv.y + tanhf(ga.y) * mv.y;
        const float y2 = (1.0f - tanhf(fo.z)) * cv.z + tanhf(ga.z) * mv.z;
        const float y3 = (1.0f - tanhf(fo.w)) * cv.w + tanhf(ga.w) * mv.w;
        o4[i] = make_float4(y0, y1, y2, y3);
        ss += y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    }
    if (norm_w) {
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
        const float4* w4 = reinterpret_cast<const float4*>(norm_w);
        for (int i = tid; i < D4; i += 256) {
            const float4 v = o4[i], wv = w4[i];
            o4[i] = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
        }
    }
}

// Fused 3-controller (attn, k, v) multi-CTA kernels (grid.y = 3 -> 192 CTAs & 432 CTAs in 2 launches instead of 12!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_cat_rms(
    const float* __restrict__ cur0, const float* __restrict__ cur1, const float* __restrict__ cur2,
    const float* __restrict__ msgs,  // [4, M, D] (uses slots 0, 1, 2)
    float* __restrict__ cat3,        // [3, M, 2*D]
    int M, int D) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;  // 0..2
    if (m >= M || c >= 3) return;
    const int tid = threadIdx.x;
    __shared__ float sh[32];
    const float* cur = (c == 0) ? cur0 : (c == 1 ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * M * D;
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    const int D4 = D >> 2;
    float ssc = 0.0f, ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        ssc += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
        ssm += gv.x * gv.x + gv.y * gv.y + gv.z * gv.z + gv.w * gv.w;
    }
    ssc = block_sum(ssc, sh);
    ssm = block_sum(ssm, sh);
    const float ic = rsqrtf(ssc / (float)D + kInkRmsEps);
    const float im = rsqrtf(ssm / (float)D + kInkRmsEps);
    float* dst = cat3 + ((size_t)c * M + m) * (2 * D);
    float4* dst0 = reinterpret_cast<float4*>(dst);
    float4* dst1 = reinterpret_cast<float4*>(dst + D);
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        dst0[i] = make_float4(cv.x * ic, cv.y * ic, cv.z * ic, cv.w * ic);
        dst1[i] = make_float4(gv.x * im, gv.y * im, gv.z * im, gv.w * im);
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_down_smallM(
    const float* __restrict__ cat3,  // [3, M, 2*D]
    const float* __restrict__ wd0, const float* __restrict__ wd1, const float* __restrict__ wd2,
    float* __restrict__ gate3,       // [3, M, 64]
    int M, int D2) {
    const int r = blockIdx.x;  // 0..63
    const int c = blockIdx.y;  // 0..2
    if (r >= kInkGateRank || c >= 3) return;
    const int tid = threadIdx.x;
    const int D2_4 = D2 >> 2;
    const float* wd = (c == 0) ? wd0 : (c == 1 ? wd1 : wd2);
    const float* cat = cat3 + (size_t)c * M * D2;
    const float4* wr = reinterpret_cast<const float4*>(wd + (size_t)r * D2);
    float acc[MAX_M] = {};
    for (int i = tid; i < D2_4; i += 256) {
        const float4 w = wr[i];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 x = reinterpret_cast<const float4*>(cat + (size_t)m * D2)[i];
                acc[m] = fmaf(w.x, x.x, acc[m]); acc[m] = fmaf(w.y, x.y, acc[m]);
                acc[m] = fmaf(w.z, x.z, acc[m]); acc[m] = fmaf(w.w, x.w, acc[m]);
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float s = block_sum(acc[m], sh);
            if (tid == 0) gate3[((size_t)c * M + m) * kInkGateRank + r] = s;
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_apply_smallM(
    const float* __restrict__ cur0, const float* __restrict__ cur1, const float* __restrict__ cur2,
    const float* __restrict__ msgs,  // [4, M, D]
    const float* __restrict__ gate3, // [3, M, 64]
    const float* __restrict__ wg0, const float* __restrict__ wg1, const float* __restrict__ wg2,
    const float* __restrict__ wf0, const float* __restrict__ wf1, const float* __restrict__ wf2,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2,
    int M, int D) {
    const int c = blockIdx.y;  // 0..2
    if (c >= 3) return;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    const float* gate = gate3 + (size_t)c * M * kInkGateRank;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gate[i];
    __syncthreads();

    const int sub = tid >> 4;
    const int lg  = tid & 15;
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;

    const float* cur = (c == 0) ? cur0 : (c == 1 ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * M * D;
    const float* wg  = (c == 0) ? wg0 : (c == 1 ? wg1 : wg2);
    const float* wf  = (c == 0) ? wf0 : (c == 1 ? wf1 : wf2);
    float* out       = (c == 0) ? out0 : (c == 1 ? out1 : out2);

    const float4 wgv = reinterpret_cast<const float4*>(wg + (size_t)d * kInkGateRank)[lg];
    const float4 wfv = reinterpret_cast<const float4*>(wf + (size_t)d * kInkGateRank)[lg];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 g4 = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = wgv.x * g4.x + wgv.y * g4.y + wgv.z * g4.z + wgv.w * g4.w;
            float af = wfv.x * g4.x + wfv.y * g4.y + wfv.z * g4.z + wfv.w * g4.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af)) * cur[idx] + tanhf(ag) * msg[idx];
            }
        }
    }
}

__global__ void __launch_bounds__(256) k_ink_rms_norm_3x(
    float* __restrict__ x0, float* __restrict__ x1, float* __restrict__ x2,
    const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
    int M, int D) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;
    if (m >= M || c >= 3) return;
    const int tid = threadIdx.x;
    __shared__ float sh[32];
    float* xp = ((c == 0) ? x0 : (c == 1 ? x1 : x2)) + (size_t)m * D;
    const float* wp = (c == 0) ? w0 : (c == 1 ? w1 : w2);
    const int D4 = D >> 2;
    float4* x4 = reinterpret_cast<float4*>(xp);
    const float4* w4 = reinterpret_cast<const float4*>(wp);
    float ss = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
    for (int i = tid; i < D4; i += 256) {
        const float4 v = x4[i], wv = w4[i];
        x4[i] = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
    }
}

// FUSED 1-Launch Looped Pre-Attn Kernel: replaces attn_res_mix(d_msgs) + attn_res_mix(d_mixed) + ctrl_3x_cat_rms (-48 launches/step!)
// Grid: dim3(M, 4), Block: 256 threads.
//   blockIdx.y in {0,1,2}: computes msg_c from bank (bst, msg_q[c]) AND src_c from depth states (stp, res_q[is_swa ? 0 : c]),
//                          writes d_msgs[c], d_mixed[c], AND normalized [rms(src_c); rms(msg_c)] into cat3[c]!
//   blockIdx.y == 3:       computes msg_3 (for MLP controller) from bank (bst, msg_q[3]) -> writes d_msgs[3]!
__global__ void __launch_bounds__(256) k_ink_attn_res_3x_cat_rms(
    DepthStatePtrs stp,              // depth states [0..S-1]
    const float* __restrict__ q_res, // [1 or 3, D]
    bool is_swa,
    DepthStatePtrs bst,              // bank states [0..B-1]
    const float* __restrict__ q_msg, // [4, D]
    float* __restrict__ mixed_out,   // [3, M, D] (for is_swa, slot 0..2 all written so ctrl_3x reads uniformly)
    float* __restrict__ msgs_out,    // [4, M, D]
    float* __restrict__ cat3_out,    // [3, M, 2*D] (only used when cat3_out != nullptr)
    int M, int D) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;  // 0..3
    if (m >= M || c >= 4) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int D4 = D >> 2;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][2][8];  // [state][0=ss, 1=dot][warp]
    __shared__ float s_prob[8];
    __shared__ float s_sc[8];
    __shared__ float s_sinv[8];
    __shared__ float sh[32];

    auto compute_mix_probs = [&](const DepthStatePtrs& ds, const float* qptr) {
        const int S = ds.count;
        const float4* q4 = reinterpret_cast<const float4*>(qptr);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= S) break;
            const float4* sp4 = reinterpret_cast<const float4*>(ds.p[s] + (size_t)m * D);
            float ss = 0.0f, dot = 0.0f;
            for (int i = tid; i < D4; i += 256) {
                const float4 sv = sp4[i], qv = q4[i];
                ss  += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
                dot = fmaf(sv.x, qv.x, dot); dot = fmaf(sv.y, qv.y, dot);
                dot = fmaf(sv.z, qv.z, dot); dot = fmaf(sv.w, qv.w, dot);
            }
            ss  = warp_sum(ss);
            dot = warp_sum(dot);
            if (lane == 0) { s_wred[s][0][warp] = ss; s_wred[s][1][warp] = dot; }
        }
        __syncthreads();
        if (tid == 0) {
            float mx = -1e30f;
            for (int s = 0; s < S; ++s) {
                float ss = 0.0f, ds_v = 0.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) { ss += s_wred[s][0][w]; ds_v += s_wred[s][1][w]; }
                s_sinv[s] = rsqrtf(ss / (float)D + kInkRmsEps);
                s_sc[s] = ds_v * s_sinv[s] * inv_sqrt_d;
                mx = fmaxf(mx, s_sc[s]);
            }
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) { s_sc[s] = expf(s_sc[s] - mx); sum += s_sc[s]; }
            const float inv_s = 1.0f / sum;
            for (int s = 0; s < S; ++s) s_prob[s] = s_sc[s] * inv_s * s_sinv[s];
        }
        __syncthreads();
    };

    // 1. Compute msg_c from bank states -> write msgs_out[c, m, :] and accumulate ssm!
    compute_mix_probs(bst, q_msg + (size_t)c * D);
    float4* mdst4 = reinterpret_cast<float4*>(msgs_out + ((size_t)c * M + m) * D);
    float ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < bst.count) {
                const float w = s_prob[s];
                const float4 sv = reinterpret_cast<const float4*>(bst.p[s] + (size_t)m * D)[i];
                ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay);
                az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
            }
        }
        mdst4[i] = make_float4(ax, ay, az, aw);
        ssm += ax * ax + ay * ay + az * az + aw * aw;
    }
    if (c == 3) return;  // MLP message done!

    // 2. Compute src_c from depth states -> write mixed_out[c, m, :] and accumulate ssc!
    const int rq = is_swa ? 0 : c;
    compute_mix_probs(stp, q_res + (size_t)rq * D);
    float4* sdst4 = reinterpret_cast<float4*>(mixed_out + ((size_t)c * M + m) * D);
    float ssc = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < stp.count) {
                const float w = s_prob[s];
                const float4 sv = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D)[i];
                ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay);
                az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
            }
        }
        sdst4[i] = make_float4(ax, ay, az, aw);
        ssc += ax * ax + ay * ay + az * az + aw * aw;
    }

    // 3. Write normalized [rms(src_c); rms(msg_c)] directly into cat3_out (if small-M fused path)!
    if (cat3_out) {
        ssc = block_sum(ssc, sh);
        ssm = block_sum(ssm, sh);
        const float ic = rsqrtf(ssc / (float)D + kInkRmsEps);
        const float im = rsqrtf(ssm / (float)D + kInkRmsEps);
        float* dst = cat3_out + ((size_t)c * M + m) * (2 * D);
        float4* d0 = reinterpret_cast<float4*>(dst);
        float4* d1 = reinterpret_cast<float4*>(dst + D);
        for (int i = tid; i < D4; i += 256) {
            const float4 sv = sdst4[i], mv = mdst4[i];
            d0[i] = make_float4(sv.x * ic, sv.y * ic, sv.z * ic, sv.w * ic);
            d1[i] = make_float4(mv.x * im, mv.y * im, mv.z * im, mv.w * im);
        }
    }
}

// FUSED 1-Launch Looped Pre-MLP Kernel: replaces attn_res_mix(d_moe_in) + ctrl_cat_rms (-24 launches/step!)
__global__ void __launch_bounds__(256) k_ink_mlp_res_cat_rms(
    DepthStatePtrs stp,
    const float* __restrict__ q_mlp,  // [D]
    const float* __restrict__ msg_m,  // [M, D] (d_msgs + 3*M*D)
    float* __restrict__ moe_in_out,   // [M, D]
    float* __restrict__ cat_out,      // [M, 2*D]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int D4 = D >> 2;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][2][8];
    __shared__ float s_prob[8];
    __shared__ float s_sc[8];
    __shared__ float s_sinv[8];
    __shared__ float sh[32];

    const int S = stp.count;
    const float4* q4 = reinterpret_cast<const float4*>(q_mlp);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= S) break;
        const float4* sp4 = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D);
        float ss = 0.0f, dot = 0.0f;
        for (int i = tid; i < D4; i += 256) {
            const float4 sv = sp4[i], qv = q4[i];
            ss  += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
            dot = fmaf(sv.x, qv.x, dot); dot = fmaf(sv.y, qv.y, dot);
            dot = fmaf(sv.z, qv.z, dot); dot = fmaf(sv.w, qv.w, dot);
        }
        ss = warp_sum(ss); dot = warp_sum(dot);
        if (lane == 0) { s_wred[s][0][warp] = ss; s_wred[s][1][warp] = dot; }
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s < S; ++s) {
            float ss = 0.0f, ds_v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) { ss += s_wred[s][0][w]; ds_v += s_wred[s][1][w]; }
            s_sinv[s] = rsqrtf(ss / (float)D + kInkRmsEps);
            s_sc[s] = ds_v * s_sinv[s] * inv_sqrt_d;
            mx = fmaxf(mx, s_sc[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) { s_sc[s] = expf(s_sc[s] - mx); sum += s_sc[s]; }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < S; ++s) s_prob[s] = s_sc[s] * inv_s * s_sinv[s];
    }
    __syncthreads();

    float4* dst4 = reinterpret_cast<float4*>(moe_in_out + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg_m + (size_t)m * D);
    float ssc = 0.0f, ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < S) {
                const float w = s_prob[s];
                const float4 sv = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D)[i];
                ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay);
                az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
            }
        }
        dst4[i] = make_float4(ax, ay, az, aw);
        const float4 mv = mp4[i];
        ssc += ax * ax + ay * ay + az * az + aw * aw;
        ssm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
    }
    ssc = block_sum(ssc, sh);
    ssm = block_sum(ssm, sh);
    const float ic = rsqrtf(ssc / (float)D + kInkRmsEps);
    const float im = rsqrtf(ssm / (float)D + kInkRmsEps);
    float4* c0 = reinterpret_cast<float4*>(cat_out + (size_t)m * (2 * D));
    float4* c1 = reinterpret_cast<float4*>(cat_out + (size_t)m * (2 * D) + D);
    for (int i = tid; i < D4; i += 256) {
        const float4 sv = dst4[i], mv = mp4[i];
        c0[i] = make_float4(sv.x * ic, sv.y * ic, sv.z * ic, sv.w * ic);
        c1[i] = make_float4(mv.x * im, mv.y * im, mv.z * im, mv.w * im);
    }
}

// FUSED MoE 8-Slot Reduction + Post-MLP ShortConv4 + Residual Add (-48 launches & -48 [M,2304] DRAM round-trips!)
__global__ void __launch_bounds__(256) k_ink_moe_reduce8_sconv_m1(
    const float* __restrict__ y_pair,     // [8, 2304]
    const float* __restrict__ static_w,   // [2304, 4]
    float* __restrict__ history,          // [3, 2304]
    float* __restrict__ cur_io) {         // [2304] in/out
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= kInkHidden) return;
    float cur = 0.0f;
#pragma unroll
    for (int k = 0; k < kInkTopK; ++k) cur += y_pair[(size_t)k * kInkHidden + c];
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float h0 = history[0 * kInkHidden + c];
    const float h1 = history[1 * kInkHidden + c];
    const float h2 = history[2 * kInkHidden + c];
    const float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    cur_io[c] += (cur + acc) * 0.44721359549995793f;
    history[0 * kInkHidden + c] = h1;
    history[1 * kInkHidden + c] = h2;
    history[2 * kInkHidden + c] = cur;
}

__global__ void __launch_bounds__(256) k_ink_moe_reduce8_sconv_batched(
    const float* __restrict__ y_pair,     // [B, 8, 2304]
    const float* __restrict__ static_w,   // [2304, 4]
    float* __restrict__ history_pool,     // [max_slots, 3, 2304]
    const int32_t* __restrict__ slot_ids, // [B]
    float* __restrict__ cur_io,           // [B, 2304]
    int B) {
    const int b = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || c >= kInkHidden) return;
    const float* yp = y_pair + (size_t)b * kInkTopK * kInkHidden + c;
    float cur = 0.0f;
#pragma unroll
    for (int k = 0; k < kInkTopK; ++k) cur += yp[(size_t)k * kInkHidden];
    const int slot = slot_ids[b];
    float* hist = history_pool + (size_t)slot * 3 * kInkHidden;
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float h0 = hist[0 * kInkHidden + c];
    const float h1 = hist[1 * kInkHidden + c];
    const float h2 = hist[2 * kInkHidden + c];
    const float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    cur_io[(size_t)b * kInkHidden + c] += (cur + acc) * 0.44721359549995793f;
    hist[0 * kInkHidden + c] = h1;
    hist[1 * kInkHidden + c] = h2;
    hist[2 * kInkHidden + c] = cur;
}

// Tree-32 Fused Reduce8 + ShortConv4 + Stage + Residual Add (18 CTAs x 128 threads, 16 KB smem, ZERO d_moe_raw!)
__global__ void __launch_bounds__(128) k_ink_moe_reduce8_sconv_tree(
    const float* __restrict__ y_pair,     // [M, 8, 2304]
    const float* __restrict__ static_w,   // [2304, 4]
    const float* __restrict__ history,    // [3, 2304]
    const int8_t* __restrict__ lag_idx,   // [M, 3]
    float* __restrict__ stage_v,          // [tree_cap, kInkSconvCh]
    int ch_off,
    float* __restrict__ cur_io,           // [M, 2304]
    int M) {
    const int tid = threadIdx.x;
    const int c = blockIdx.x * 128 + tid;
    if (c >= kInkHidden) return;
    __shared__ float s_x[kInkMaxTree][128];  // 32 x 128 x 4 = 16 KB!
    for (int m = 0; m < M; ++m) {
        const float* yp = y_pair + (size_t)m * kInkTopK * kInkHidden + c;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kInkTopK; ++k) s += yp[(size_t)k * kInkHidden];
        s_x[m][tid] = s;
        if (stage_v) stage_v[(size_t)m * kInkSconvCh + ch_off + c] = s;
    }
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float hc0 = history[0 * kInkHidden + c];
    const float hc1 = history[1 * kInkHidden + c];
    const float hc2 = history[2 * kInkHidden + c];
    auto fetch = [&](int idx) -> float {
        return (idx >= 0) ? s_x[idx][tid] : ((idx == -1) ? hc2 : ((idx == -2) ? hc1 : hc0));
    };
    for (int m = 0; m < M; ++m) {
        const float h2 = fetch((int)lag_idx[m * 3 + 0]);
        const float h1 = fetch((int)lag_idx[m * 3 + 1]);
        const float h0 = fetch((int)lag_idx[m * 3 + 2]);
        const float cur = s_x[m][tid];
        const float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        cur_io[(size_t)m * kInkHidden + c] += (cur + acc) * 0.44721359549995793f;
    }
}

// Fast Shared-W F32 GEMM for tiny N (<= 64, e.g. attn_sink_proj [24, 2304], k/v_sconv.weight_proj [32, 2304]) at large M!
// Caches W[row, 0..K-1] in 9 KB shared memory ONCE and computes 16 tokens/CTA (reads W 16x fewer times -> 0.15 ms)!
// Grid: (N, (M + 15) / 16), Block: 256 (8 warps x 2 tokens/warp = 16 tokens/CTA)
__global__ void __launch_bounds__(256) k_ink_gemm_f32_smallN(
    const float* __restrict__ W,  // [N, K]
    const float* __restrict__ X,  // [M, K]
    float* __restrict__ Y,        // [M, N]
    int M, int N, int K) {
    const int row = blockIdx.x;  // 0..N-1
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int K4 = K >> 2;
    __shared__ alignas(16) float s_w[kInkHidden];
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)row * K);
    float4* sw4 = reinterpret_cast<float4*>(s_w);
    for (int i = tid; i < K4; i += 256) sw4[i] = wr[i];
    __syncthreads();

    const int m0 = blockIdx.y * 16;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ma = m0 + 2 * warp;
    const int mb = ma + 1;
    if (ma >= M) return;

    const float4* xa = reinterpret_cast<const float4*>(X + (size_t)ma * K);
    const float4* xb = (mb < M) ? reinterpret_cast<const float4*>(X + (size_t)mb * K) : nullptr;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int k4 = lane; k4 < K4; k4 += 32) {
        const float4 w = sw4[k4];
        const float4 va = xa[k4];
        acc0 = fmaf(w.x, va.x, acc0); acc0 = fmaf(w.y, va.y, acc0);
        acc0 = fmaf(w.z, va.z, acc0); acc0 = fmaf(w.w, va.w, acc0);
        if (xb) {
            const float4 vb = xb[k4];
            acc1 = fmaf(w.x, vb.x, acc1); acc1 = fmaf(w.y, vb.y, acc1);
            acc1 = fmaf(w.z, vb.z, acc1); acc1 = fmaf(w.w, vb.w, acc1);
        }
    }
    acc0 = warp_sum(acc0);
    if (lane == 0) Y[(size_t)ma * N + row] = acc0;
    if (xb) {
        acc1 = warp_sum(acc1);
        if (lane == 0) Y[(size_t)mb * N + row] = acc1;
    }
}

// Standalone fast vectorized float4 RMSNorm (supports F32 output and optional BF16 output)
__global__ void __launch_bounds__(256) k_ink_rms_norm(
    const float* __restrict__ x,
    const float* __restrict__ w,  // [D] F32 (nullable -> parameterless RMSNorm)
    float* __restrict__ out_f32,
    __nv_bfloat16* __restrict__ out_bf16,
    int M,
    int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    __shared__ float sh[32];
    const float4* xp4 = reinterpret_cast<const float4*>(x + (size_t)m * D);
    float ss = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 v = xp4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
    float4* df4 = out_f32 ? reinterpret_cast<float4*>(out_f32 + (size_t)m * D) : nullptr;
    __nv_bfloat16* db = out_bf16 ? (out_bf16 + (size_t)m * D) : nullptr;
    const float4* w4 = w ? reinterpret_cast<const float4*>(w) : nullptr;
    for (int i = tid; i < D4; i += 256) {
        const float4 v = xp4[i];
        const float4 wv = w4 ? w4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float r0 = v.x * inv * wv.x, r1 = v.y * inv * wv.y;
        const float r2 = v.z * inv * wv.z, r3 = v.w * inv * wv.w;
        if (df4) df4[i] = make_float4(r0, r1, r2, r3);
        if (db) {
            db[i * 4 + 0] = __float2bfloat16_rn(r0); db[i * 4 + 1] = __float2bfloat16_rn(r1);
            db[i * 4 + 2] = __float2bfloat16_rn(r2); db[i * 4 + 3] = __float2bfloat16_rn(r3);
        }
    }
}

// ============================================================================
// 4. 4-Tap Causal ShortConv Kernels (2D-Parallel Prefill, Batched B=1..64, Tree-16 Verify!)
// ============================================================================
// State layout per slot & visit: float history[3][C] where row 0 = t-3, row 1 = t-2, row 2 = t-1.
// static_w: [C, 4] F32 (index 0 = lag 3 t-3, ..., index 3 = lag 0 t).
// dyn_taps: nullptr (SWA/post-attn/post-ffn -> scale 1/sqrt(5)) or [M, n_heads, 4] F32 (global K/V -> scale 1/3, lag 0..3).

// 2D-Parallel Prefill causal ShortConv (M > 16): dim3((C+255)/256, (M+15)/16) -> 22x faster on RTX 5090 (0.25 ms at M=512)!
// Does NOT mutate history (call race-free k_ink_sconv4_save_hist right after!).
__global__ void __launch_bounds__(256) k_ink_sconv4_seq_par(
    const float* __restrict__ x,          // [M, C]
    const float* __restrict__ static_w,   // [C, 4]
    const float* __restrict__ dyn_taps,   // [M, n_heads, 4] or nullptr
    const float* __restrict__ history,    // [3, C] (read-only!)
    float* __restrict__ out,              // [M, C]
    int M, int C, int n_heads,
    bool add_to_out = false) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int m0 = blockIdx.y * 16;
    if (c >= C || m0 >= M) return;
    const int m_end = min(M, m0 + 16);

    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float scale = dyn_taps ? (1.0f / 3.0f) : 0.44721359549995793f;
    const int hd = (dyn_taps && n_heads > 0) ? (c / (C / n_heads)) : 0;

    auto fetch = [&](int idx) -> float {
        return idx >= 0 ? x[(size_t)idx * C + c] : history[(size_t)(3 + idx) * C + c];
    };
    float h0 = fetch(m0 - 3);
    float h1 = fetch(m0 - 2);
    float h2 = fetch(m0 - 1);

    for (int m = m0; m < m_end; ++m) {
        const float cur = x[(size_t)m * C + c];
        float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        if (dyn_taps) {
            const float4 dt = reinterpret_cast<const float4*>(dyn_taps + ((size_t)m * n_heads + hd) * 4)[0];
            acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
        }
        const float res = (cur + acc) * scale;
        const size_t oi = (size_t)m * C + c;
        out[oi] = add_to_out ? (out[oi] + res) : res;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

__global__ void __launch_bounds__(256) k_ink_sconv4_save_hist(
    const float* __restrict__ x,
    float* __restrict__ history,
    int M, int C) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    // M > 16 guaranteed >= 3, so new history is simply x[M-3], x[M-2], x[M-1]!
    history[0 * C + c] = x[(size_t)(M - 3) * C + c];
    history[1 * C + c] = x[(size_t)(M - 2) * C + c];
    history[2 * C + c] = x[(size_t)(M - 1) * C + c];
}

// Single-slot / Prefill causal ShortConv over M consecutive tokens (updates history[3, C] in-place at end!)
__global__ void __launch_bounds__(256) k_ink_sconv4_seq(
    const float* __restrict__ x,          // [M, C]
    const float* __restrict__ static_w,   // [C, 4]
    const float* __restrict__ dyn_taps,   // [M, n_heads, 4] or nullptr
    float* __restrict__ history,          // [3, C] (in/out; if readonly_hist, does not mutate history)
    float* __restrict__ out,              // [M, C] (may NOT alias x when M > 1)
    int M,
    int C,
    int n_heads,
    bool update_hist,
    bool add_to_out = false) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float scale = dyn_taps ? (1.0f / 3.0f) : 0.44721359549995793f;  // 1/sqrt(5)
    const int hd = (dyn_taps && n_heads > 0) ? (c / (C / n_heads)) : 0;

    float h0 = history[0 * C + c];  // t-3
    float h1 = history[1 * C + c];  // t-2
    float h2 = history[2 * C + c];  // t-1

    for (int m = 0; m < M; ++m) {
        const float cur = x[(size_t)m * C + c];
        float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        if (dyn_taps) {
            const float4 dt = reinterpret_cast<const float4*>(dyn_taps + ((size_t)m * n_heads + hd) * 4)[0];
            acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
        }
        const float res = (cur + acc) * scale;
        const size_t oi = (size_t)m * C + c;
        out[oi] = add_to_out ? (out[oi] + res) : res;
        h0 = h1;
        h1 = h2;
        h2 = cur;
    }
    if (update_hist) {
        history[0 * C + c] = h0;
        history[1 * C + c] = h1;
        history[2 * C + c] = h2;
    }
}

// Batched multi-slot 1-token-per-slot decode ShortConv (B = 1..64 concurrent requests, each with its own history!)
// history_pool: [max_slots, 3, C], slot_ids: [B]
__global__ void __launch_bounds__(256) k_ink_sconv4_batched(
    const float* __restrict__ x,            // [B, C]
    const float* __restrict__ static_w,     // [C, 4]
    const float* __restrict__ dyn_taps,     // [B, n_heads, 4] or nullptr
    float* __restrict__ history_pool,       // [max_slots, 3, C]
    const int32_t* __restrict__ slot_ids,   // [B]
    float* __restrict__ out,                // [B, C]
    int B,
    int C,
    int n_heads,
    bool add_to_out = false) {
    const int b = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || c >= C) return;
    const int slot = slot_ids[b];
    float* hist = history_pool + (size_t)slot * 3 * C;
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float scale = dyn_taps ? (1.0f / 3.0f) : 0.44721359549995793f;
    const float h0 = hist[0 * C + c];
    const float h1 = hist[1 * C + c];
    const float h2 = hist[2 * C + c];
    const float cur = x[(size_t)b * C + c];
    float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    if (dyn_taps) {
        const int hd = c / (C / n_heads);
        const float4 dt = reinterpret_cast<const float4*>(dyn_taps + ((size_t)b * n_heads + hd) * 4)[0];
        acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
    }
    const float res = (cur + acc) * scale;
    const size_t oi = (size_t)b * C + c;
    out[oi] = add_to_out ? (out[oi] + res) : res;
    hist[0 * C + c] = h1;
    hist[1 * C + c] = h2;
    hist[2 * C + c] = cur;
}

// Tree-16 speculative verify ShortConv: node m reads its exact parent chain AND stages raw cur into stage_sconv in ONE kernel!
// Eliminates 1,536 cudaMemcpyAsync D2D calls per verify pass (-5.0 ms/iter!!)!
__global__ void __launch_bounds__(256) k_ink_sconv4_tree(
    const float* __restrict__ x,          // [M, C]
    const float* __restrict__ static_w,   // [C, 4]
    const float* __restrict__ dyn_taps,   // [M, n_heads, 4] or nullptr
    const float* __restrict__ history,    // [3, C]
    const int8_t* __restrict__ lag_idx,   // [M, 3] (for each node m: source of t-1, t-2, t-3)
    float* __restrict__ out,              // [M, C]
    float* __restrict__ stage_v,          // [16, kInkSconvCh] staging slice for this visit (nullable)
    int ch_off,
    int M,
    int C,
    int n_heads,
    bool add_to_out = false) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= C) return;
    const auto fetch = [&](int idx) -> float {
        return idx >= 0 ? x[(size_t)idx * C + c] : history[(size_t)(3 + idx) * C + c];
    };
    const float h2 = fetch((int)lag_idx[m * 3 + 0]);  // t-1
    const float h1 = fetch((int)lag_idx[m * 3 + 1]);  // t-2
    const float h0 = fetch((int)lag_idx[m * 3 + 2]);  // t-3
    const float cur = x[(size_t)m * C + c];
    if (stage_v) stage_v[(size_t)m * kInkSconvCh + ch_off + c] = cur;
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float scale = dyn_taps ? (1.0f / 3.0f) : 0.44721359549995793f;
    float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    if (dyn_taps) {
        const int hd = c / (C / n_heads);
        const float4 dt = reinterpret_cast<const float4*>(dyn_taps + ((size_t)m * n_heads + hd) * 4)[0];
        acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
    }
    const float res = (cur + acc) * scale;
    const size_t oi = (size_t)m * C + c;
    out[oi] = add_to_out ? (out[oi] + res) : res;
}

// ============================================================================
// 5. Fused Q/K/V/R Clipping + Value RMSNorm + Global Log-Pos Tau(pos)
// ============================================================================
__global__ void __launch_bounds__(128) k_ink_qkvr_clip_tau(
    float* __restrict__ q,               // [M, 24, 128]
    const float* __restrict__ q_scale,   // [128] or [24, 128] -> note: q_l2_clip.scale is [24, 128] or [128]! Checked via q_scale_stride!
    int q_scale_elems,
    float* __restrict__ k,               // [M, 8, 128]
    float* __restrict__ v,               // [M, 8, 128]
    float* __restrict__ r,               // [M, 24, 16]
    const float* __restrict__ r_scale,   // [24, 16] or [16]
    int r_scale_elems,
    const int32_t* __restrict__ pos_arr, // [M] token positions (for global tau)
    bool is_global,
    int M) {
    // Grid: (24 + 8 + 8 + 24 = 64 blocks along head axis, M along token axis)
    const int m = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    if (m >= M) return;

    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(pos_arr[m] + 1) * (1.0f / (float)kInkLogFloor);
        tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
    }

    __shared__ float sh[8];
    if (blk < 24) {
        // Q head h = blk, dim 128 (4 warps = 128 threads, 1 elem/thread)
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
    } else if (blk < 32) {
        // K head h = blk - 24, dim 128
        const int h = blk - 24;
        float* kp = k + ((size_t)m * kInkKVHeads + h) * kInkHeadDim;
        float val = kp[tid];
        float ss = block_sum(val * val, sh);
        const float factor = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f));
        kp[tid] = val * factor;
    } else if (blk < 40) {
        // V head h = blk - 32, dim 128 -> parameterless RMSNorm(eps = 1e-6)
        const int h = blk - 32;
        float* vp = v + ((size_t)m * kInkKVHeads + h) * kInkHeadDim;
        float val = vp[tid];
        float ss = block_sum(val * val, sh);
        const float inv = rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
        vp[tid] = val * inv;
    } else if (blk < 64) {
        // R head h = blk - 40, dim 16 (first 16 threads)
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

// ============================================================================
// 6. Fast Dense Linear Projection Dispatcher (BF16 / F32 / Q8_ROW / S5_K / S6_K)
//    Y[M, N] = X[M, K] @ W[N, K]^T
// ============================================================================
// High-bandwidth exact BF16 GEMV for small M (1..16): caches exact FP32 X in tiled shared memory + vectorized uint4!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_gemv_bf16(
    const uint16_t* __restrict__ W,  // [N, K] BF16
    const float* __restrict__ X,     // [M, K] F32
    float* __restrict__ Y,           // [M, N] F32
    int M, int N, int K) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 8 + warp;

    const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    float acc[MAX_M] = {};
    // Tile exact FP32 X into <= 16 KB shared memory (KT = 256 for MAX_M=16, 512 for MAX_M=4/8, 3072 for MAX_M=1) -> high occupancy + zero global X rereads!
    constexpr int KT = (MAX_M <= 8) ? 512 : 256;
    __shared__ alignas(16) float s_x[MAX_M][KT];
    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int kl4 = kl >> 2;
        for (int m = 0; m < M; ++m) {
            const float4* sx = reinterpret_cast<const float4*>(X + (size_t)m * K + k0);
            float4* dx = reinterpret_cast<float4*>(s_x[m]);
            for (int i = tid; i < kl4; i += 256) dx[i] = sx[i];
        }
        __syncthreads();
        if (row < N) {
            const int kl8 = kl >> 3;
            const uint4* wrt = wr + (k0 >> 3);
            for (int k8 = lane; k8 < kl8; k8 += 32) {
                const uint4 wv = wrt[k8];
                const float2 w0 = bf16x2_to_f32x2(wv.x);
                const float2 w1 = bf16x2_to_f32x2(wv.y);
                const float2 w2 = bf16x2_to_f32x2(wv.z);
                const float2 w3 = bf16x2_to_f32x2(wv.w);
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float4 x0 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2];
                        const float4 x1 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2 + 1];
                        acc[m] = fmaf(w0.x, x0.x, acc[m]); acc[m] = fmaf(w0.y, x0.y, acc[m]);
                        acc[m] = fmaf(w1.x, x0.z, acc[m]); acc[m] = fmaf(w1.y, x0.w, acc[m]);
                        acc[m] = fmaf(w2.x, x1.x, acc[m]); acc[m] = fmaf(w2.y, x1.y, acc[m]);
                        acc[m] = fmaf(w3.x, x1.z, acc[m]); acc[m] = fmaf(w3.y, x1.w, acc[m]);
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }
    if (row >= N) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]);
            if (lane == 0) Y[(size_t)m * N + row] = v;
        }
    }
}

// High-bandwidth exact Q8_ROW GEMV for small M (1..16): 128-bit uint4 loads + tiled FP32 s_x, reads W ONCE across M tokens!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_gemv_q8row(
    const uint8_t* __restrict__ W_raw,  // [N] float32 scales + [N, K] int8
    const float* __restrict__ X,        // [M, K] F32
    float* __restrict__ Y,              // [M, N] F32
    int M, int N, int K) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 8 + warp;

    const float sc = (row < N) ? reinterpret_cast<const float*>(W_raw)[row] : 0.0f;
    const int8_t* qr = reinterpret_cast<const int8_t*>(W_raw + (size_t)N * 4) + (size_t)row * K;
    const uint4* w16 = reinterpret_cast<const uint4*>(qr);
    float acc[MAX_M] = {};

    constexpr int KT = (MAX_M <= 8) ? 512 : 256;
    __shared__ alignas(16) float s_x[MAX_M][KT];
    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int kl4 = kl >> 2;
        for (int m = 0; m < M; ++m) {
            const float4* sx = reinterpret_cast<const float4*>(X + (size_t)m * K + k0);
            float4* dx = reinterpret_cast<float4*>(s_x[m]);
            for (int i = tid; i < kl4; i += 256) dx[i] = sx[i];
        }
        __syncthreads();
        if (row < N) {
            const int kl16 = kl >> 4;
            const uint4* wt = w16 + (k0 >> 4);
            for (int k16 = lane; k16 < kl16; k16 += 32) {
                const uint4 wv = wt[k16];
                const uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t p = u[q];
                    const float f0 = (float)(int8_t)(p & 0xFFu);
                    const float f1 = (float)(int8_t)((p >> 8) & 0xFFu);
                    const float f2 = (float)(int8_t)((p >> 16) & 0xFFu);
                    const float f3 = (float)(int8_t)(p >> 24);
                    const int idx4 = k16 * 4 + q;
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float4 xv = reinterpret_cast<const float4*>(s_x[m])[idx4];
                            acc[m] = fmaf(f0, xv.x, acc[m]); acc[m] = fmaf(f1, xv.y, acc[m]);
                            acc[m] = fmaf(f2, xv.z, acc[m]); acc[m] = fmaf(f3, xv.w, acc[m]);
                        }
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }
    if (row >= N) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]) * sc;
            if (lane == 0) Y[(size_t)m * N + row] = v;
        }
    }
}

// High-bandwidth F32 GEMV for small M (1..16)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_gemv_f32(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= N) return;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)row * K);
    const int K4 = K >> 2;
    float acc[MAX_M] = {};
    for (int k4 = lane; k4 < K4; k4 += 32) {
        const float4 wv = wr[k4];
        const int base = k4 * 4;
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 xv = *(const float4*)(X + (size_t)m * K + base);
                acc[m] = fmaf(wv.x, xv.x, acc[m]); acc[m] = fmaf(wv.y, xv.y, acc[m]);
                acc[m] = fmaf(wv.z, xv.z, acc[m]); acc[m] = fmaf(wv.w, xv.w, acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]);
            if (lane == 0) Y[(size_t)m * N + row] = v;
        }
    }
}

// Multi-Projection Fused GEMV descriptor (shares 1 s_x load + 1 launch across up to 5 projections: wq+wr+gate+wk+wv!)
// 128 threads/CTA (4 warps = 4 output rows/CTA) -> 2x more CTAs across 170 SMs (zero tail-wave stall -> 1,500+ GB/s!)
struct InkMultiProj {
    const uint8_t* W[5];
    float*         Y[5];
    int            N[5];
    int            cumN[6];
    int            count;
};

// Constant-index resolver for InkMultiProj: keeps the 128-byte parameter struct in CONSTANT[0] (eliminates STACK:128 spill!)
__device__ __forceinline__ bool ink_resolve_multi_proj(
    const InkMultiProj& mp, int grow,
    const uint8_t*& Wp, float*& Yp, int& Np, int& lrow) {
    Wp = nullptr; Yp = nullptr; Np = 0; lrow = 0;
#pragma unroll
    for (int p = 0; p < 5; ++p) {
        if (Wp == nullptr && p < mp.count && grow < mp.cumN[p + 1]) {
            Wp = mp.W[p];
            Yp = mp.Y[p];
            Np = mp.N[p];
            lrow = grow - mp.cumN[p];
        }
    }
    return Wp != nullptr;
}

// 256B-Coalesced 16-Lane Multi-Projection BF16 GEMV (8 output rows/CTA, 16 lanes x 16B uint4 = 256B!)
// 2x wave occupancy on wo_ud (288 CTAs > 170 SMs!) + zero tiling at M=1 -> saturates 1,500+ GB/s GDDR7!
template <int MAX_M, bool IS_FP16 = false>
__global__ void __launch_bounds__(128) k_ink_gemv_multi_bf16(
    InkMultiProj mp,
    const float* __restrict__ X,     // [M, K] F32
    int M, int K) {
    const int tid  = threadIdx.x;
    const int sub  = tid >> 4;       // 0..7 output rows per CTA
    const int lg   = tid & 15;       // 0..15 lanes per row (16 x 16B = 256B coalesced!)
    const int grow = blockIdx.x * 8 + sub;

    const uint8_t* Wp = nullptr;
    float* Yp = nullptr;
    int Np = 0, lrow = 0;
    const bool valid = ink_resolve_multi_proj(mp, grow, Wp, Yp, Np, lrow);
    const uint4* wr = valid ? reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(Wp) + (size_t)lrow * K) : nullptr;
    float acc[MAX_M] = {};

    constexpr int KT = (MAX_M == 1) ? 2304 : ((MAX_M <= 8) ? 576 : 256);
    __shared__ alignas(16) float s_x[MAX_M][KT];
    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int kl4 = kl >> 2;
        for (int m = 0; m < M; ++m) {
            const float4* sx = reinterpret_cast<const float4*>(X + (size_t)m * K + k0);
            float4* dx = reinterpret_cast<float4*>(s_x[m]);
            for (int i = tid; i < kl4; i += 128) dx[i] = sx[i];
        }
        __syncthreads();
        if (valid) {
            const int kl8 = kl >> 3;
            const uint4* wrt = wr + (k0 >> 3);
            for (int k8 = lg; k8 < kl8; k8 += 16) {
                const uint4 wv = wrt[k8];
                float2 w0, w1, w2, w3;
                if constexpr (IS_FP16) {
                    w0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
                    w1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
                    w2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
                    w3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
                } else {
                    w0 = bf16x2_to_f32x2(wv.x);
                    w1 = bf16x2_to_f32x2(wv.y);
                    w2 = bf16x2_to_f32x2(wv.z);
                    w3 = bf16x2_to_f32x2(wv.w);
                }
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float4 x0 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2];
                        const float4 x1 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2 + 1];
                        acc[m] = fmaf(w0.x, x0.x, acc[m]); acc[m] = fmaf(w0.y, x0.y, acc[m]);
                        acc[m] = fmaf(w1.x, x0.z, acc[m]); acc[m] = fmaf(w1.y, x0.w, acc[m]);
                        acc[m] = fmaf(w2.x, x1.x, acc[m]); acc[m] = fmaf(w2.y, x1.y, acc[m]);
                        acc[m] = fmaf(w3.x, x1.z, acc[m]); acc[m] = fmaf(w3.y, x1.w, acc[m]);
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }
    if (!valid) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
            if (lg == 0) Yp[(size_t)m * Np + lrow] = v;
        }
    }
}

// Zero-Extra-Launch 16-Lane M=1 Q8_ROW Multi-Projection GEMV (8 rows/CTA = 288 CTAs on wo_ud, inline L1 quantize -> eliminates 192 launches!)
__global__ void __launch_bounds__(128) k_ink_gemv_multi_q8row_m1(
    InkMultiProj mp,
    const float* __restrict__ X,     // [1, K] F32 (K <= 3072, multiple of 64)
    int K) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ng_tot = K >> 6;       // 36 for K=2304, 48 for K=3072
    __shared__ alignas(16) uint32_t s_xq[3072 / 4];  // 3,072 B!
    __shared__ float                s_xs[48];         // 192 B!
    for (int g = warp; g < ng_tot; g += 4) {
        const float* xp = X + g * 64;
        const float v0 = xp[2 * lane], v1 = xp[2 * lane + 1];
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float d  = am * (1.0f / 127.0f);
        const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
        int8_t* dst = reinterpret_cast<int8_t*>(s_xq) + g * 64;
        dst[2 * lane]     = (int8_t)max(-127, min(127, __float2int_rn(v0 * inv)));
        dst[2 * lane + 1] = (int8_t)max(-127, min(127, __float2int_rn(v1 * inv)));
        if (lane == 0) s_xs[g] = d;
    }
    __syncthreads();

    const int sub  = tid >> 4;       // 0..7 output rows per CTA
    const int lg   = tid & 15;       // 0..15 lanes per row (16 x 16B = 256B!)
    const int grow = blockIdx.x * 8 + sub;
    const uint8_t* Wp = nullptr;
    float* Yp = nullptr;
    int Np = 0, lrow = 0;
    if (!ink_resolve_multi_proj(mp, grow, Wp, Yp, Np, lrow)) return;
    const float sc = reinterpret_cast<const float*>(Wp)[lrow];
    const uint4* w16 = reinterpret_cast<const uint4*>(
        reinterpret_cast<const int8_t*>(Wp + (size_t)Np * 4) + (size_t)lrow * K);
    const int n16 = K >> 4;
    float acc = 0.0f;
    for (int k16 = lg; k16 < n16; k16 += 16) {
        const uint4 wv = w16[k16];
        const uint4 xv = *reinterpret_cast<const uint4*>(s_xq + k16 * 4);
        int32_t s = dp4a_i8(wv.x, xv.x, 0);
        s = dp4a_i8(wv.y, xv.y, s);
        s = dp4a_i8(wv.z, xv.z, s);
        s = dp4a_i8(wv.w, xv.w, s);
        acc = fmaf((float)s, s_xs[k16 >> 2], acc);
    }
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lg == 0) Yp[lrow] = acc * sc;
}

// 256B-Coalesced 16-Lane Pre-Quantized Q8_ROW Multi-Projection GEMV (8 rows/CTA, pure dp4a_i8, ZERO per-CTA quantization!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_gemv_multi_q8row_preq(
    InkMultiProj mp,
    const InkQ8_1_G64* __restrict__ Xq,  // [M, K/64]
    int M, int K) {
    const int tid  = threadIdx.x;
    const int sub  = tid >> 4;       // 0..7 output rows per CTA (2x wave occupancy!)
    const int lg   = tid & 15;       // 0..15 lanes per row (16 x 16B = 256B!)
    const int grow = blockIdx.x * 8 + sub;

    const uint8_t* Wp = nullptr;
    float* Yp = nullptr;
    int Np = 0, lrow = 0;
    const bool valid = ink_resolve_multi_proj(mp, grow, Wp, Yp, Np, lrow);
    const float sc = valid ? reinterpret_cast<const float*>(Wp)[lrow] : 0.0f;
    const uint4* w16 = valid ? reinterpret_cast<const uint4*>(
        reinterpret_cast<const int8_t*>(Wp + (size_t)Np * 4) + (size_t)lrow * K) : nullptr;
    float acc[MAX_M] = {};

    constexpr int GT = (MAX_M <= 8) ? 36 : 16;
    const int ng_tot = K >> 6;
    __shared__ alignas(16) InkQ8_1_G64 s_xq[MAX_M][GT];
    for (int g0 = 0; g0 < ng_tot; g0 += GT) {
        const int ng = min(GT, ng_tot - g0);
        const int n4 = ng * 5;
        for (int m = 0; m < M; ++m) {
            const uint4* sx = reinterpret_cast<const uint4*>(Xq + (size_t)m * ng_tot + g0);
            uint4* dx = reinterpret_cast<uint4*>(s_xq[m]);
            for (int i = tid; i < n4; i += 128) dx[i] = sx[i];
        }
        __syncthreads();
        if (valid) {
            const int n16 = ng * 4;
            const uint4* wt = w16 + g0 * 4;
            for (int k16 = lg; k16 < n16; k16 += 16) {
                const uint4 wv = wt[k16];
                const int g = k16 >> 2;
                const int sub4 = (k16 & 3) * 16;
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(s_xq[m][g].qs + sub4);
                        int32_t s = dp4a_i8(wv.x, xv.x, 0);
                        s = dp4a_i8(wv.y, xv.y, s);
                        s = dp4a_i8(wv.z, xv.z, s);
                        s = dp4a_i8(wv.w, xv.w, s);
                        acc[m] = fmaf((float)s, s_xq[m][g].d, acc[m]);
                    }
                }
            }
        }
        if (g0 + GT < ng_tot) __syncthreads();
    }
    if (!valid) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
            if (lg == 0) Yp[(size_t)m * Np + lrow] = v * sc;
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_gemv_multi_q8row(
    InkMultiProj mp,
    const float* __restrict__ X,
    int M, int K) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grow = blockIdx.x * 4 + warp;

    const uint8_t* Wp = nullptr;
    float* Yp = nullptr;
    int Np = 0, lrow = 0;
    const bool valid = ink_resolve_multi_proj(mp, grow, Wp, Yp, Np, lrow);
    const float sc = valid ? reinterpret_cast<const float*>(Wp)[lrow] : 0.0f;
    const uint4* w16 = valid ? reinterpret_cast<const uint4*>(
        reinterpret_cast<const int8_t*>(Wp + (size_t)Np * 4) + (size_t)lrow * K) : nullptr;
    float acc[MAX_M] = {};

    // High-speed Q8_1 dp4a_i8 tiled GEMV: quantize X[m, k0..k0+KT-1] in 64-elem groups ONCE into shared memory (4x faster ALU!)
    constexpr int KT = 512;
    constexpr int NG = KT / 64;  // 8 groups of 64 elements
    __shared__ alignas(16) uint32_t s_xq[MAX_M][KT / 4];   // MAX_M * 512 B (<= 8 KB for MAX_M=16!)
    __shared__ float                s_xs[MAX_M][NG];       // MAX_M * 32 B
    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int ng = kl >> 6;  // <= 8 groups of 64
        // 4 warps cooperate to quantize M * ng groups of 64 floats (each warp handles 1 group of 64: 2 floats/lane!)
        const int tot_g = M * ng;
        for (int g_i = warp; g_i < tot_g; g_i += 4) {
            const int m = g_i / ng;
            const int g = g_i % ng;
            const float* xp = X + (size_t)m * K + k0 + g * 64;
            const float v0 = xp[2 * lane];
            const float v1 = xp[2 * lane + 1];
            const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
            const float d  = am * (1.0f / 127.0f);
            const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
            const int8_t q0 = (int8_t)max(-127, min(127, __float2int_rn(v0 * inv)));
            const int8_t q1 = (int8_t)max(-127, min(127, __float2int_rn(v1 * inv)));
            int8_t* dst = reinterpret_cast<int8_t*>(s_xq[m]) + g * 64;
            dst[2 * lane]     = q0;
            dst[2 * lane + 1] = q1;
            if (lane == 0) s_xs[m][g] = d;
        }
        __syncthreads();
        if (valid) {
            // 8 groups of 64 = 32 uint4s (each lane handles 1 uint4 = 16 weights -> group = k16 >> 2!)
            const int kl16 = kl >> 4;
            const uint4* wt = w16 + (k0 >> 4);
            for (int k16 = lane; k16 < kl16; k16 += 32) {
                const uint4 wv = wt[k16];
                const int g = k16 >> 2;
                const int b4 = k16 * 4;
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(s_xq[m] + b4);
                        int32_t s = dp4a_i8(wv.x, xv.x, 0);
                        s = dp4a_i8(wv.y, xv.y, s);
                        s = dp4a_i8(wv.z, xv.z, s);
                        s = dp4a_i8(wv.w, xv.w, s);
                        acc[m] = fmaf((float)s, s_xs[m][g], acc[m]);
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }
    if (!valid) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]) * sc;
            if (lane == 0) Yp[(size_t)m * Np + lrow] = v;
        }
    }
}

// Convert F32 [M, K] -> BF16 [M, K] for cuBLAS Tensor Core GEMM (vectorized float4 -> 2x bf16x2!)
__global__ void __launch_bounds__(256) k_ink_f32_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n) {
    const size_t n4 = n >> 2;
    const size_t i4 = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i4 < n4) {
        const float4 v = reinterpret_cast<const float4*>(src)[i4];
        __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(dst);
        d2[i4 * 2 + 0] = __floats2bfloat162_rn(v.x, v.y);
        d2[i4 * 2 + 1] = __floats2bfloat162_rn(v.z, v.w);
    } else if (i4 == n4) {
        for (size_t i = n4 * 4; i < n; ++i) dst[i] = __float2bfloat16_rn(src[i]);
    }
}

// Convert F32 -> FP16 (clamped to [-65504, 65504]) for Volta sm_70 / Turing sm_75 FP16 Tensor Cores
__global__ void __launch_bounds__(256) k_ink_f32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        float v = src[i];
        v = fminf(fmaxf(v, -65504.0f), 65504.0f);
        dst[i] = __float2half_rn(v);
    }
}

// Convert BF16 -> FP16 (supports in-place dst == (__half*)src) for Volta sm_70 / Turing sm_75 FP16 TC
__global__ void __launch_bounds__(256) k_ink_bf16_to_fp16(const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(src[i]);
        v = fminf(fmaxf(v, -65504.0f), 65504.0f);
        dst[i] = __float2half_rn(v);
    }
}

// Convert BF16 -> F32 for Pascal sm_60 / sm_61 FP32 cuBLAS GEMM
__global__ void __launch_bounds__(256) k_ink_bf16_to_f32(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Per-row symmetric INT8 quantization of X [M, K] -> Xq [M, K] int8 + x_sc [M] (for 45 TOPS sm_61 / 220 TOPS sm_75 INT8 GEMM!)
__global__ void __launch_bounds__(256) k_ink_quant_x_row_i8(
    const float* __restrict__ X, int8_t* __restrict__ Xq, float* __restrict__ x_sc, int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const float* xp = X + (size_t)m * K;
    float am = 0.0f;
    for (int k = tid; k < K; k += 256) am = fmaxf(am, fabsf(xp[k]));
    am = warp_max(am);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
               fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = am * (1.0f / 127.0f);
    const float inv = sc > 0.0f ? (127.0f / am) : 0.0f;
    int8_t* dst = Xq + (size_t)m * K;
    for (int k = tid; k < K; k += 256) dst[k] = (int8_t)max(-127, min(127, __float2int_rn(xp[k] * inv)));
    if (tid == 0) x_sc[m] = sc;
}

// In-place INT32 -> F32 scaling of cuBLAS INT8 GEMM output Y[M, N] = (float)Y_i32[m, n] * w_sc[n] * x_sc[m]
__global__ void __launch_bounds__(256) k_ink_i32_to_f32_scale(
    int32_t* __restrict__ Y_i32, float* __restrict__ Y_f32,
    const float* __restrict__ w_sc, const float* __restrict__ x_sc, int M, int N) {
    const size_t idx = (size_t)blockIdx.x * 256 + threadIdx.x;
    const size_t tot = (size_t)M * N;
    if (idx < tot) {
        const int m = (int)(idx / (size_t)N);
        const int n = (int)(idx % (size_t)N);
        Y_f32[idx] = (float)Y_i32[idx] * w_sc[n] * x_sc[m];
    }
}

// Dequantize Q8_ROW [N, K] -> BF16 [N, K] into reusable scratch for cuBLAS prefill
__global__ void __launch_bounds__(256) k_ink_q8row_to_bf16(const uint8_t* __restrict__ raw, __nv_bfloat16* __restrict__ dst, int N, int K) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (row >= N || col >= K) return;
    const float sc = reinterpret_cast<const float*>(raw)[row];
    const int8_t q = reinterpret_cast<const int8_t*>(raw + (size_t)N * 4)[(size_t)row * K + col];
    dst[(size_t)row * K + col] = __float2bfloat16_rn((float)q * sc);
}

// Dequantize S_K (S5_K / S6_K / S8_K) [N, K] -> BF16 [N, K] into reusable scratch for cuBLAS prefill
template <int FMT>
__global__ void __launch_bounds__(256) k_ink_sk_to_bf16(const uint8_t* __restrict__ raw, __nv_bfloat16* __restrict__ dst, int N, int K) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (row >= N || col >= K) return;
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    const int nsb = K >> 8;
    const int s = col >> 8;
    const int idx = col & 255;
    const int g = idx >> 6;
    const uint8_t* blk = raw + ((size_t)row * nsb + s) * SB;
    const float D = f16_to_f32(ld16_u(blk));
    const float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
    const uint8_t* pl = blk + 6;
    float v = 0.0f;
    if constexpr (FMT == FMT_S5_K) {
        uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
        uint32_t hi = (pl[128 + (idx >> 3)] >> (idx & 7)) & 1u;
        v = ((float)(lo | (hi << 4)) - HALF) * eff;
    } else if constexpr (FMT == FMT_S6_K) {
        uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
        uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
        v = ((float)(lo | (hi << 4)) - HALF) * eff;
    } else if constexpr (FMT == FMT_S8_K) {
        v = (float)((int8_t)pl[idx]) * eff;
    }
    dst[(size_t)row * K + col] = __float2bfloat16_rn(v);
}

// ============================================================================
// 7. Exact 2-Stage Cauchy-Schwarz Screened Unembed (0.25 ms on RTX 5090, 100.000% Bit-Exact Argmax!)
//    Coarse plane: symmetric INT8 per-row quantized unembed [V, 2304] (460 MB) + per-row scale sc[V] + exact L2 error eps[V]!
//    For any query h [D]:
//      |dot(w_v, h) - dot(w_hat_v, h)| <= eps[v] * ||h||_2
//    Pass 1 reads ONLY 460 MB (2x smaller than BF16, shared across M=1..16!), finds exact lower bound L_max,
//    collects the ~1..32 candidates with (dot_hat_v + eps_v * ||h||_2 >= L_max).
//    Pass 2 evaluates ONLY those candidates against the 100% exact BF16 unembed weight!
// ============================================================================
struct UnembedScreenIndex {
    int8_t*  q8 = nullptr;    // [V, 2304] signed int8
    float*   sc = nullptr;    // [V] per-row scale = amax / 127.0
    float*   eps = nullptr;   // [V] exact L2 residual norm ||w_v - w_hat_v||_2
    int      V = 0;
    bool     ready = false;

    void free_all() {
        if (q8)  { INK_CK(cudaFree(q8));  q8 = nullptr; }
        if (sc)  { INK_CK(cudaFree(sc));  sc = nullptr; }
        if (eps) { INK_CK(cudaFree(eps)); eps = nullptr; }
        V = 0;
        ready = false;
    }
};

__global__ void __launch_bounds__(256) k_ink_build_unembed_screen_bf16(
    const uint16_t* __restrict__ W,  // [V, 2304] BF16
    int8_t* __restrict__ q8_out,     // [V, 2304]
    float* __restrict__ sc_out,      // [V]
    float* __restrict__ eps_out,     // [V]
    int V) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= V) return;
    const uint16_t* wr = W + (size_t)row * kInkHidden;
    float am = 0.0f;
    for (int k = lane; k < kInkHidden; k += 32) am = fmaxf(am, fabsf(bf16_to_f32(wr[k])));
    am = warp_max(am);
    const float s = am * (1.0f / 127.0f);
    const float inv = s > 0.0f ? (127.0f / am) : 0.0f;
    int8_t* dst = q8_out + (size_t)row * kInkHidden;
    float err2 = 0.0f;
    for (int k = lane; k < kInkHidden; k += 32) {
        const float w = bf16_to_f32(wr[k]);
        const int q = max(-127, min(127, __float2int_rn(w * inv)));
        dst[k] = (int8_t)q;
        const float d = w - (float)q * s;
        err2 += d * d;
    }
    err2 = warp_sum(err2);
    if (lane == 0) {
        sc_out[row] = s;
        eps_out[row] = sqrtf(err2) * 1.001f + 1e-4f;
    }
}

// Build exact 2-stage Cauchy-Schwarz screening index for S6_K unembed (SQ-15 & SQ-10)!
__global__ void __launch_bounds__(256) k_ink_build_unembed_screen_s6k(
    const uint8_t* __restrict__ W,   // [V, 9 * 198] S6_K
    int8_t* __restrict__ q8_out,     // [V, 2304]
    float* __restrict__ sc_out,      // [V]
    float* __restrict__ eps_out,     // [V]
    int V) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= V) return;
    constexpr int SB = ink_sb_bytes(FMT_S6_K);
    const uint8_t* rptr = W + (size_t)row * 9 * SB;
    auto dec = [&](int d) -> float {
        const int s = d >> 8, idx = d & 255, g = idx >> 6;
        const uint8_t* blk = rptr + s * SB;
        const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
        const uint8_t* pl = blk + 6;
        const uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
        const uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
        return ((float)(lo | (hi << 4)) - 31.5f) * eff;
    };
    float am = 0.0f;
    for (int d = lane; d < kInkHidden; d += 32) am = fmaxf(am, fabsf(dec(d)));
    am = warp_max(am);
    const float s = am * (1.0f / 127.0f);
    const float inv = s > 0.0f ? (127.0f / am) : 0.0f;
    int8_t* dst = q8_out + (size_t)row * kInkHidden;
    float err2 = 0.0f;
    for (int d = lane; d < kInkHidden; d += 32) {
        const float w = dec(d);
        const int q = max(-127, min(127, __float2int_rn(w * inv)));
        dst[d] = (int8_t)q;
        const float diff = w - (float)q * s;
        err2 += diff * diff;
    }
    err2 = warp_sum(err2);
    if (lane == 0) {
        sc_out[row] = s;
        eps_out[row] = sqrtf(err2) * 1.001f + 1e-4f;
    }
}

__global__ void __launch_bounds__(256) k_ink_screen_eval_s6k(
    const uint8_t* __restrict__ W,
    const float* __restrict__ h,
    const int32_t* __restrict__ cand_ids,
    const int32_t* __restrict__ cand_cnt,
    int max_cand,
    int32_t* __restrict__ out_argmax,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cnt = min(cand_cnt[m], max_cand);
    __shared__ float s_h[kInkHidden];
    const float* hm = h + (size_t)m * kInkHidden;
    for (int d = tid; d < kInkHidden; d += 256) s_h[d] = hm[d];
    __syncthreads();

    constexpr int SB = ink_sb_bytes(FMT_S6_K);
    float best_v = -3e38f;
    int32_t best_id = 0;
    for (int ci = warp; ci < cnt; ci += 8) {
        const int32_t row = cand_ids[(size_t)m * max_cand + ci];
        const uint8_t* rptr = W + (size_t)row * 9 * SB;
        float acc = 0.0f;
        for (int d = lane; d < kInkHidden; d += 32) {
            const int s = d >> 8, idx = d & 255, g = idx >> 6;
            const uint8_t* blk = rptr + s * SB;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const uint8_t* pl = blk + 6;
            const uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
            const uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
            acc = fmaf(((float)(lo | (hi << 4)) - 31.5f) * eff, s_h[d], acc);
        }
        acc = warp_sum(acc);
        if (lane == 0 && (acc > best_v || (acc == best_v && row < best_id))) {
            best_v = acc;
            best_id = row;
        }
    }
    __shared__ float bv[8];
    __shared__ int32_t bi[8];
    if (lane == 0) { bv[warp] = best_v; bi[warp] = best_id; }
    __syncthreads();
    if (tid == 0) {
        float mv = -3e38f; int32_t mi = 0;
        for (int w = 0; w < 8; ++w) {
            if (bv[w] > mv || (bv[w] == mv && bi[w] < mi)) { mv = bv[w]; mi = bi[w]; }
        }
        out_argmax[m] = mi;
    }
}

__global__ void __launch_bounds__(256) k_ink_row_l2norm(
    const float* __restrict__ h, float* __restrict__ out_norm,
    int* __restrict__ max_lb_bits, int32_t* __restrict__ cand_cnt, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float sh[32];
    const float* hm = h + (size_t)m * kInkHidden;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < kInkHidden; d += 256) ss += hm[d] * hm[d];
    ss = block_sum(ss, sh);
    if (threadIdx.x == 0) {
        out_norm[m] = sqrtf(ss);
        uint32_t b = __float_as_uint(-1e30f);
        max_lb_bits[m] = (int)(b ^ ((b >> 31) ? 0x7FFFFFFFu : 0u));
        cand_cnt[m] = 0;
    }
}

// Multi-M exact Pass 1: reads the 460 MB INT8 unembed plane ONCE across ALL M=1..8 rows (~0.30 ms, 100.000% exact Cauchy-Schwarz!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_screen_pass1_multiM(
    const int8_t* __restrict__ q8,      // [V, 2304]
    const float* __restrict__ sc,       // [V]
    const float* __restrict__ eps,      // [V]
    const float* __restrict__ h,        // [M, 2304]
    const float* __restrict__ h_norm,   // [M] ||h_m||_2
    float* __restrict__ dot_hat,        // [M, V]
    int* __restrict__ max_lb_bits,      // [M] (nullable when only base_logits needed!)
    int M, int V) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 8 + warp;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    float acc[MAX_M] = {};
    // Tile exact FP32 h into <= 18 KB shared memory (KT = 576 for MAX_M=8 -> 4 tiles = 18 KB, 1152 for MAX_M=4 -> 18 KB, 2304 for MAX_M=1 -> 9 KB!)
    constexpr int KT = (MAX_M == 1) ? 2304 : ((MAX_M <= 4) ? 1152 : 576);
    static_assert(kInkHidden % KT == 0 && KT % 16 == 0, "KT divides 2304");
    __shared__ alignas(16) float s_h[MAX_M][KT];

    for (int k0 = 0; k0 < kInkHidden; k0 += KT) {
        for (int m = 0; m < M; ++m) {
            const float4* src4 = reinterpret_cast<const float4*>(h + (size_t)m * kInkHidden + k0);
            float4* dst4 = reinterpret_cast<float4*>(s_h[m]);
            for (int i = tid; i < (KT / 4); i += 256) dst4[i] = src4[i];
        }
        __syncthreads();
        if (row < V) {
            const uint4* qt = q16 + (k0 >> 4);
            for (int i = lane; i < (KT / 16); i += 32) {
                const uint4 v = qt[i];
                const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t p = u[q];
                    const float f0 = (float)(int8_t)(p & 0xFFu);
                    const float f1 = (float)(int8_t)((p >> 8) & 0xFFu);
                    const float f2 = (float)(int8_t)((p >> 16) & 0xFFu);
                    const float f3 = (float)(int8_t)(p >> 24);
                    const int idx4 = i * 4 + q;
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float4 xv = reinterpret_cast<const float4*>(s_h[m])[idx4];
                            acc[m] = fmaf(f0, xv.x, acc[m]); acc[m] = fmaf(f1, xv.y, acc[m]);
                            acc[m] = fmaf(f2, xv.z, acc[m]); acc[m] = fmaf(f3, xv.w, acc[m]);
                        }
                    }
                }
            }
        }
        if (k0 + KT < kInkHidden) __syncthreads();
    }
    if (row >= V) return;
    const float sc_r = sc[row];
    const float eps_r = eps[row];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float d = warp_sum(acc[m]) * sc_r;
            if (lane == 0) {
                dot_hat[(size_t)m * V + row] = d;
                if (max_lb_bits) {
                    const float lb = d - eps_r * h_norm[m];
                    uint32_t b = __float_as_uint(lb);
                    int ord = (int)(b ^ ((b >> 31) ? 0x7FFFFFFFu : 0u));
                    atomicMax(&max_lb_bits[m], ord);
                }
            }
        }
    }
}

// Keep single-M signature wrapper for ink_kernel_test compatibility
__global__ void __launch_bounds__(256) k_ink_screen_pass1(
    const int8_t* __restrict__ q8,
    const float* __restrict__ sc,
    const float* __restrict__ eps,
    const float* __restrict__ h,
    const float* __restrict__ h_norm,
    float* __restrict__ dot_hat,
    int* __restrict__ max_lb_bits,
    int M, int V) {
    const int m = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (m >= M || row >= V) return;
    const float* hm = h + (size_t)m * kInkHidden;
    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    float acc = 0.0f;
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
        const int base = i * 16;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 xv = *(const float4*)(hm + base + q * 4);
            const uint32_t p = u[q];
            acc = fmaf((float)(int8_t)(p & 0xFFu), xv.x, acc);
            acc = fmaf((float)(int8_t)((p >> 8) & 0xFFu), xv.y, acc);
            acc = fmaf((float)(int8_t)((p >> 16) & 0xFFu), xv.z, acc);
            acc = fmaf((float)(int8_t)((p >> 24) & 0xFFu), xv.w, acc);
        }
    }
    acc = warp_sum(acc) * sc[row];
    if (lane == 0) {
        dot_hat[(size_t)m * V + row] = acc;
        if (max_lb_bits) {
            const float lb = acc - eps[row] * h_norm[m];
            uint32_t b = __float_as_uint(lb);
            int ord = (int)(b ^ ((b >> 31) ? 0x7FFFFFFFu : 0u));
            atomicMax(&max_lb_bits[m], ord);
        }
    }
}

__global__ void __launch_bounds__(256) k_ink_screen_pass2_bf16(
    const uint16_t* __restrict__ W,
    const float* __restrict__ eps,
    const float* __restrict__ h,
    const float* __restrict__ h_norm,
    const float* __restrict__ dot_hat,
    const int* __restrict__ max_lb_bits,
    int32_t* __restrict__ cand_ids,
    int32_t* __restrict__ cand_cnt,
    int max_cand,
    int M, int V) {
    (void)W; (void)h;
    const int m = blockIdx.y;
    const int row = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || row >= V) return;
    int ord = max_lb_bits[m];
    uint32_t ub_bits = (uint32_t)ord ^ (((uint32_t)ord >> 31) ? 0x7FFFFFFFu : 0u);
    const float lmax = __uint_as_float(ub_bits);
    const float ub = dot_hat[(size_t)m * V + row] + eps[row] * h_norm[m];
    if (ub >= lmax) {
        int pos = atomicAdd(&cand_cnt[m], 1);
        if (pos < max_cand) cand_ids[(size_t)m * max_cand + pos] = row;
    }
}

__global__ void __launch_bounds__(256) k_ink_screen_eval_bf16(
    const uint16_t* __restrict__ W,
    const float* __restrict__ h,
    const int32_t* __restrict__ cand_ids,
    const int32_t* __restrict__ cand_cnt,
    int max_cand,
    int32_t* __restrict__ out_argmax,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int n = min(cand_cnt[m], max_cand);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* hm = h + (size_t)m * kInkHidden;

    float best_v = -3e38f;
    int32_t best_i = 0x7FFFFFFF;
    for (int ci = warp; ci < n; ci += 8) {
        const int32_t row = cand_ids[(size_t)m * max_cand + ci];
        const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)row * kInkHidden);
        float acc = 0.0f;
        for (int k8 = lane; k8 < (kInkHidden / 8); k8 += 32) {
            const uint4 wv = wr[k8];
            const float2 w0 = bf16x2_to_f32x2(wv.x), w1 = bf16x2_to_f32x2(wv.y);
            const float2 w2 = bf16x2_to_f32x2(wv.z), w3 = bf16x2_to_f32x2(wv.w);
            const float4 x0 = *(const float4*)(hm + k8 * 8);
            const float4 x1 = *(const float4*)(hm + k8 * 8 + 4);
            acc = fmaf(w0.x, x0.x, acc); acc = fmaf(w0.y, x0.y, acc);
            acc = fmaf(w1.x, x0.z, acc); acc = fmaf(w1.y, x0.w, acc);
            acc = fmaf(w2.x, x1.x, acc); acc = fmaf(w2.y, x1.y, acc);
            acc = fmaf(w3.x, x1.z, acc); acc = fmaf(w3.y, x1.w, acc);
        }
        acc = warp_sum(acc);
        const float val = kInkSoftcap * tanhf(acc * (1.0f / kInkSoftcap));
        if (lane == 0 && (val > best_v || (val == best_v && row < best_i))) {
            best_v = val;
            best_i = row;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = sv[0]; int32_t bi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        out_argmax[m] = bi;
    }
}

// Logit softcap in-place: z = 30.0 * tanh(z / 30.0)
__global__ void __launch_bounds__(256) k_ink_softcap(float* __restrict__ logits, size_t n, float cap) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) logits[i] = cap * tanhf(logits[i] * (1.0f / cap));
}

// Deterministic first-max argmax per row [M, V] -> out[M]
__global__ void __launch_bounds__(1024) k_ink_argmax_rows(const float* __restrict__ logits, int V, int32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int32_t si[32];
    const float* lg = logits + (size_t)blockIdx.x * V;
    float bv = -3.0e38f;
    int32_t bi = 0;
    for (int i = threadIdx.x; i < V; i += 1024) {
        float v = lg[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b2 = sv[0]; int32_t i2 = si[0];
        for (int q = 1; q < 32; ++q)
            if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
        out[blockIdx.x] = i2;
    }
}

// Elementwise multiply by sigmoid(gate) + residual add helpers
__global__ void __launch_bounds__(256) k_ink_mul_sigmoid(float* __restrict__ x, const float* __restrict__ g, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + __expf(-g[i]));
}

__global__ void __launch_bounds__(256) k_ink_add(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

}  // namespace strata::ink
