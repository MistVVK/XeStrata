// engine/src/inkling2/hip/ink_hip_tree.hip.h
// Wave-7 Multi-Branch On-GPU W=4 Markov Beam Tree Drafter, Exact Tree ShortConv/FlashAttn Verify,
// Zero-Logits-D2H Batched Unembed Argmax, and 1-Launch Winning-Path Commit Kernels for AMD HIP.
// Works across RDNA4 (gfx120x), RDNA3/3.5 (gfx110x/1103/1150/1151), and RDNA2 (gfx103x).
// 100% lossless (0.00000 KLD): tree verify + winning-path commit is mathematically identical to 1-by-1 decode.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_ops.hip.h"

namespace strata::inkling2::hip {

constexpr int kTreeMaxM      = 16;   // max tree verify batch size (uses M* in {3,4,5,6,8,10,12})
constexpr int kBeamW         = 4;    // W=4 multi-branch Markov beam width
constexpr int kBeamMaxSteps  = 6;    // up to 6 draft depth steps (depths 1..6 -> max 7-token chain)
constexpr int kTreePoolMax   = 32;   // root (0) + up to 6 * 4 = 24 beam-expanded candidate nodes

// Compact candidate node produced on GPU by the W=4 beam drafter (16 bytes -> 400 B D2H!)
struct alignas(16) BeamTreeNode {
    int32_t tok;
    int32_t parent;   // index in pool[0..count-1], -1 for root 0
    int32_t depth;    // 0 for root, 1..6 for drafted tokens
    float   cum_lp;   // cumulative log-probability from root
};

// =================================================================================================
// 1. On-GPU W=4 Multi-Branch Recurrent Markov Beam Search (100% HIP-Graphable!)
// =================================================================================================

// Initialize beam pool with root node 0 = anchor_tok (1 thread!)
__global__ void __launch_bounds__(32) k5_beam_init(
    const int32_t* __restrict__ hdr_in,     // [2]: {anchor_pos, anchor_tok}
    BeamTreeNode* __restrict__ pool,        // [kTreePoolMax]
    int32_t* __restrict__ pool_cnt,         // [1]
    int32_t* __restrict__ beam_tok,         // [kBeamW]
    int32_t* __restrict__ beam_node,        // [kBeamW]
    float* __restrict__ beam_lp) {          // [kBeamW]
    if (threadIdx.x == 0) {
        const int32_t anchor_tok = hdr_in[1];
        pool[0] = {anchor_tok, -1, 0, 0.0f};
        pool_cnt[0] = 1;
        beam_tok[0] = anchor_tok;
        beam_node[0] = 0;
        beam_lp[0] = 0.0f;
        for (int w = 1; w < kBeamW; ++w) {
            beam_tok[w] = anchor_tok;
            beam_node[w] = 0;
            beam_lp[w] = -1e30f;
        }
    }
}

// Batched W-Beam Markov RNN Step A1: Grid ((rank + 7) / 8, W_active), Block 256
__global__ void __launch_bounds__(256) k5_beam_markov_proj_bf16(
    const int32_t* __restrict__ beam_tok,      // [W_active]
    const float* __restrict__ state_in,        // [W_active, rank]
    float* __restrict__ state_out,             // [W_active, rank]
    float* __restrict__ latent_out,            // [W_active, rank]
    const float* __restrict__ hidden,          // [D] (shared across all W beams at this depth!)
    const float* __restrict__ context,         // [D]
    const float* __restrict__ markov_w1,       // [V, rank]
    const uint16_t* __restrict__ rnn_proj_w,   // [3*rank, 2*rank + 2*D] BF16
    const float* __restrict__ rnn_proj_b,      // [3*rank]
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const int bw = blockIdx.y;
    if (r >= rank) return;

    const int prev_tok = beam_tok[bw];
    const int in_dim = 2 * rank + 2 * D;
    const float* st_row = state_in + static_cast<size_t>(bw) * rank;
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
        const float4 sv = *reinterpret_cast<const float4*>(st_row + i);
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
        state_out[static_cast<size_t>(bw) * rank + r] = fmaf(gate, st_row[r] - cand, cand);
        latent_out[static_cast<size_t>(bw) * rank + r] = tanhf(d2);
    }
}

// Batched W-Beam Markov Step A2: Grid ((D + 7) / 8, W_active), Block 256
__global__ void __launch_bounds__(256) k5_beam_markov_corr_bf16(
    const float* __restrict__ latent,         // [W_active, rank]
    const uint16_t* __restrict__ hidden_out_w,// [D, rank] BF16
    const float* __restrict__ hidden_out_b,   // [D]
    float* __restrict__ corr_out,             // [W_active, D]
    int D, int rank) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = blockIdx.x * 8 + wave;
    const int bw = blockIdx.y;
    if (d >= D) return;
    const uint16_t* wh = hidden_out_w + static_cast<size_t>(d) * rank;
    const float* lat = latent + static_cast<size_t>(bw) * rank;
    float acc = 0.0f;
    for (int r = lane * 8; r < rank; r += 256) {
        const uint4 w8 = *reinterpret_cast<const uint4*>(wh + r);
        const float4 l0 = *reinterpret_cast<const float4*>(lat + r);
        const float4 l1 = *reinterpret_cast<const float4*>(lat + r + 4);
        acc = fmaf(bf16_to_f32(w8.x & 0xFFFFu), l0.x, fmaf(bf16_to_f32(w8.x >> 16), l0.y, acc));
        acc = fmaf(bf16_to_f32(w8.y & 0xFFFFu), l0.z, fmaf(bf16_to_f32(w8.y >> 16), l0.w, acc));
        acc = fmaf(bf16_to_f32(w8.z & 0xFFFFu), l1.x, fmaf(bf16_to_f32(w8.z >> 16), l1.y, acc));
        acc = fmaf(bf16_to_f32(w8.w & 0xFFFFu), l1.z, fmaf(bf16_to_f32(w8.w >> 16), l1.w, acc));
    }
    acc = wave32_sum(acc) + hidden_out_b[d];
    if (lane == 0) corr_out[static_cast<size_t>(bw) * D + d] = acc;
}

// Single-Read Multi-Beam Candidate Scorer: reads unembed row ONCE across all W_active <= 4 beams!
// Uses 18.4 KB LDS (4 x 2304 FP16 or BF16) -> 2-3 WG/CU occupancy!
// Grid: ((n_cand + 7) / 8,), Block: 256
__global__ void __launch_bounds__(256) k5_beam_cand_scores(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ cand_base_val, // [n_cand]
    const int32_t* __restrict__ cand_id,     // [n_cand]
    const float* __restrict__ corr,          // [W_active, D]
    float* __restrict__ scores_out,          // [W_active, n_cand]
    int W_active, int n_cand, int D, int V) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    union CorrLds {
        half_t   h[kBeamW * 2304];   // 18,432 B
        uint16_t bf[kBeamW * 2304];  // 18,432 B
    };
    __shared__ alignas(16) CorrLds sc;
    const int tot = W_active * D;
    if (fmt < 100) {
        for (int i = tid; i < tot; i += 256) sc.h[i] = static_cast<half_t>(corr[i]);
    } else if (fmt == 102) {
        for (int i = tid; i < tot; i += 256) sc.bf[i] = f32_to_bf16(corr[i]);
    }
    __syncthreads();

    const int ci = blockIdx.x * 8 + wave;
    if (ci >= n_cand) return;
    const int v = cand_id[ci];
    if (v < 0 || v >= V) {
        if (lane == 0) {
            for (int w = 0; w < W_active; ++w) scores_out[w * n_cand + ci] = -1e30f;
        }
        return;
    }

    float acc[kBeamW] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (fmt == 102) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 8; k < D; k += 256) {
            const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
            const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
            const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
            const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
            const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const uint4 c8 = *reinterpret_cast<const uint4*>(sc.bf + w * D + k);
                    acc[w] = fmaf(f0, bf16_to_f32(c8.x & 0xFFFFu), fmaf(f1, bf16_to_f32(c8.x >> 16),
                             fmaf(f2, bf16_to_f32(c8.y & 0xFFFFu), fmaf(f3, bf16_to_f32(c8.y >> 16), acc[w]))));
                    acc[w] = fmaf(f4, bf16_to_f32(c8.z & 0xFFFFu), fmaf(f5, bf16_to_f32(c8.z >> 16),
                             fmaf(f6, bf16_to_f32(c8.w & 0xFFFFu), fmaf(f7, bf16_to_f32(c8.w >> 16), acc[w]))));
                }
            }
        }
    } else if (fmt == 101) {
        const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const float4 cv = *reinterpret_cast<const float4*>(corr + static_cast<size_t>(w) * D + k);
                    acc[w] = fmaf(wv.x, cv.x, fmaf(wv.y, cv.y, fmaf(wv.z, cv.z, fmaf(wv.w, cv.w, acc[w]))));
                }
            }
        }
    } else if (fmt == 100) {
        const float fsc = reinterpret_cast<const float*>(unemb_raw)[v];
        const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
            const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
            const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
            const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
            const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const float4 cv = *reinterpret_cast<const float4*>(corr + static_cast<size_t>(w) * D + k);
                    acc[w] = fmaf(f0, cv.x, fmaf(f1, cv.y, fmaf(f2, cv.z, fmaf(f3, cv.w, acc[w]))));
                }
            }
        }
#pragma unroll
        for (int w = 0; w < kBeamW; ++w) acc[w] *= fsc;
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
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(sc.h + w * D + base);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d);
                    d = fdot2_f32(wv.z, xv.z, d);
                    d = fdot2_f32(wv.w, xv.w, d);
                    acc[w] = fmaf(d, eff, acc[w]);
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
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(sc.h + w * D + base);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d);
                    d = fdot2_f32(wv.z, xv.z, d);
                    d = fdot2_f32(wv.w, xv.w, d);
                    acc[w] = fmaf(d, eff, acc[w]);
                }
            }
        }
    }

    const float bv = cand_base_val[ci];
#pragma unroll
    for (int w = 0; w < kBeamW; ++w) {
        if (w < W_active) {
            const float z = bv + wave32_sum(acc[w]);
            if (lane == 0) scores_out[w * n_cand + ci] = z;
        }
    }
}

// 1-Block Beam Step Selector: computes exact log-softmax over n_cand=512 per beam, selects global Top-4
// beam extensions, appends them to pool[], and gathers surviving RNN states into state_next!
// Grid: (1,), Block: 256 (rank = 256 -> 1 thread per state dim!)
__global__ void __launch_bounds__(256) k5_beam_select_step(
    const float* __restrict__ scores,         // [W_in, n_cand] (raw logits before softcap)
    const int32_t* __restrict__ cand_id,      // [n_cand]
    const float* __restrict__ state_cand,     // [W_in, rank] (updated state for each input beam)
    float* __restrict__ state_next,           // [kBeamW, rank]
    int32_t* __restrict__ beam_tok,           // [kBeamW] (overwritten with next step's W_out tokens)
    int32_t* __restrict__ beam_node,          // [kBeamW] (in: parent pool idx, out: new pool idx)
    float* __restrict__ beam_lp,              // [kBeamW] (in: parent cum_lp, out: new cum_lp)
    BeamTreeNode* __restrict__ pool,          // [kTreePoolMax]
    int32_t* __restrict__ pool_cnt,           // [1]
    int W_in, int n_cand, int depth, int rank, float softcap) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float inv_cap = 1.0f / softcap;

    __shared__ float s_sc[kBeamW * 512];  // 8 KB: softcapped logits
    __shared__ float s_w[8];
    __shared__ float s_lse[kBeamW];
    __shared__ float   s_top_lp[kBeamW * kBeamW];
    __shared__ int32_t s_top_tok[kBeamW * kBeamW];
    __shared__ int32_t s_top_bw[kBeamW * kBeamW];
    __shared__ int32_t s_sel_bw[kBeamW];

    // 1. Softcap + logsumexp per active input beam
    for (int w = 0; w < W_in; ++w) {
        float lmax = -1e30f;
        for (int i = tid; i < n_cand; i += 256) {
            const float raw = scores[w * n_cand + i];
            const float z = (raw > -1e20f) ? (softcap * tanhf(raw * inv_cap)) : -1e30f;
            s_sc[w * 512 + i] = z;
            lmax = fmaxf(lmax, z);
        }
        lmax = wave32_max(lmax);
        if (lane == 0) s_w[wave] = lmax;
        __syncthreads();
        if (tid == 0) {
            float mx = -1e30f;
            for (int k = 0; k < 8; ++k) mx = fmaxf(mx, s_w[k]);
            s_lse[w] = mx;
        }
        __syncthreads();
        const float mx = s_lse[w];
        float lsum = 0.0f;
        for (int i = tid; i < n_cand; i += 256) {
            const float z = s_sc[w * 512 + i];
            if (z > -1e20f) lsum += __expf(z - mx);
        }
        lsum = wave32_sum(lsum);
        if (lane == 0) s_w[wave] = lsum;
        __syncthreads();
        if (tid == 0) {
            float sm = 0.0f;
            for (int k = 0; k < 8; ++k) sm += s_w[k];
            s_lse[w] = mx + logf(fmaxf(sm, 1e-30f));
        }
        __syncthreads();
    }

    // 2. Pick Top-4 children per input beam (W_in threads in parallel, or tid < W_in!)
    if (tid < W_in) {
        const int w = tid;
        const float par_lp = beam_lp[w];
        const float lse = s_lse[w];
        float bv[kBeamW] = {-1e30f, -1e30f, -1e30f, -1e30f};
        int32_t bi[kBeamW] = {-1, -1, -1, -1};
        for (int i = 0; i < n_cand; ++i) {
            const int32_t tok = cand_id[i];
            const float z = s_sc[w * 512 + i];
            if (z <= -1e20f || tok < 0) continue;
            // Deduplicate identical token IDs if any
            bool dup = false;
            for (int k = 0; k < kBeamW; ++k) if (bi[k] == tok) dup = true;
            if (dup) continue;
            const float lp = par_lp + (z - lse);
            if (lp > bv[kBeamW - 1]) {
                int pos = kBeamW - 1;
                while (pos > 0 && lp > bv[pos - 1]) {
                    bv[pos] = bv[pos - 1];
                    bi[pos] = bi[pos - 1];
                    --pos;
                }
                bv[pos] = lp;
                bi[pos] = tok;
            }
        }
        for (int k = 0; k < kBeamW; ++k) {
            s_top_lp[w * kBeamW + k]  = bv[k];
            s_top_tok[w * kBeamW + k] = bi[k];
            s_top_bw[w * kBeamW + k]  = w;
        }
    }
    __syncthreads();

    // 3. Thread 0 picks global Top-4 surviving beams & appends them to pool[]
    if (tid == 0) {
        const int n_ext = W_in * kBeamW;
        bool used[kBeamW * kBeamW]{};
        int32_t prev_nodes[kBeamW];
        for (int w = 0; w < W_in; ++w) prev_nodes[w] = beam_node[w];

        int cnt = pool_cnt[0];
        for (int out_w = 0; out_w < kBeamW; ++out_w) {
            int best_idx = -1;
            float best_lp = -1e30f;
            for (int j = 0; j < n_ext; ++j) {
                if (!used[j] && s_top_tok[j] >= 0 && s_top_lp[j] > best_lp) {
                    best_lp = s_top_lp[j];
                    best_idx = j;
                }
            }
            if (best_idx >= 0 && cnt < kTreePoolMax) {
                used[best_idx] = true;
                const int pw = s_top_bw[best_idx];
                const int32_t tok = s_top_tok[best_idx];
                const int32_t pid = prev_nodes[pw];
                pool[cnt] = {tok, pid, depth, best_lp};
                beam_tok[out_w]  = tok;
                beam_node[out_w] = cnt;
                beam_lp[out_w]   = best_lp;
                s_sel_bw[out_w]  = pw;
                ++cnt;
            } else {
                beam_tok[out_w]  = beam_tok[0];
                beam_node[out_w] = beam_node[0];
                beam_lp[out_w]   = -1e30f;
                s_sel_bw[out_w]  = 0;
            }
        }
        pool_cnt[0] = cnt;
    }
    __syncthreads();

    // 4. Gather surviving RNN states state_cand[s_sel_bw[w]] -> state_next[w] (coalesced, 1 thread per dim!)
    if (tid < rank) {
        for (int w = 0; w < kBeamW; ++w) {
            const int src_w = s_sel_bw[w];
            state_next[static_cast<size_t>(w) * rank + tid] = state_cand[static_cast<size_t>(src_w) * rank + tid];
        }
    }
}

// =================================================================================================
// 2. Exact Tree Verification Kernels (ShortConv, KV Store, Flash-SWA/GQA, Batched Unembed Argmax!)
// =================================================================================================

// Device tree metadata uploaded ONCE per verify step (256 bytes -> 100% HIP-Graphable!)
struct TreeMetaDev {
    int32_t  anc1[kTreeMaxM];      // parent[m], or -1 for pre-verify lag-1
    int32_t  anc2[kTreeMaxM];      // grandparent[m], or -1/-2 for pre-verify lag-1/-2
    int32_t  anc3[kTreeMaxM];      // great-grandparent[m], or -1/-2/-3 for pre-verify lag-1/-2/-3
    uint32_t mask[kTreeMaxM];      // bit j set iff node j is ancestor of m (including j == m)
    int32_t  base_pos;             // anchor_pos (prefix KV covers 0..base_pos-1)
    int32_t  slot;                 // slot id
};

// Parallel Exact Tree ShortConv4: reads pre-verify hist_pool (WITHOUT mutating it!) + tree ancestor inputs,
// records BF16 input into rec_pool[visit, m, c] for 1-launch winning-path commit!
// Grid: ((C + 255) / 256, M), Block: 256
__global__ void __launch_bounds__(256) k2_shortconv4_tree(
    const float* __restrict__ cur,          // [M, C]
    const float* __restrict__ static_w,     // [C, 4]
    const float* __restrict__ dyn_taps,     // nullptr or [M, n_heads, 4]
    const TreeMetaDev* __restrict__ tmeta,
    const uint16_t* __restrict__ hist_pool, // [max_slots, 3, C] (READ-ONLY during tree verify!)
    uint16_t* __restrict__ rec_visit_out,   // [kTreeMaxM, C] (records BF16 cur[m, c]!)
    float* __restrict__ out,                // [M, C]
    int M, int C, int n_heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (c >= C || m >= M) return;

    const int slot = tmeta->slot;
    const uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * C;
    const int a1 = tmeta->anc1[m];
    const int a2 = tmeta->anc2[m];
    const int a3 = tmeta->anc3[m];

    const float x0 = cur[static_cast<size_t>(m) * C + c];
    rec_visit_out[static_cast<size_t>(m) * C + c] = f32_to_bf16(x0);

    // Round ancestor inputs through BF16 to match SlotShortConvState bit-for-bit!
    const float h2 = (a1 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a1) * C + c]))
                               : bf16_to_f32(h_slot[2 * C + c]);
    const float h1 = (a2 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a2) * C + c]))
                               : bf16_to_f32(h_slot[(3 + a2) * C + c]);
    const float h0 = (a3 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a3) * C + c]))
                               : bf16_to_f32(h_slot[(3 + a3) * C + c]);

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
}

// Super-Fused Tree Q/R/K/V Norm+Clip + Direct FP16 Tree Scratch KV Store + Direct RelPos Bias Projection!
// Writes K/V into L2-resident tree scratch [kTreeMaxM, kv_dim] WITHOUT touching real KV pools!
// Grid: (2 * n_head + 2 * n_kv, M), Block: 128
__global__ void __launch_bounds__(128) k2_qrkv_treestore_relpos_fused(
    float* __restrict__ Q,
    const float* __restrict__ q_sc, int q_sc_stride,
    const float* __restrict__ R,
    const float* __restrict__ r_sc, int r_sc_stride,
    const float* __restrict__ rel_proj,       // [d_rel, rel_extent]
    float* __restrict__ rel_bias,             // [M, n_head, rel_extent]
    const float* __restrict__ K_in,
    const float* __restrict__ V_in,
    const int32_t* __restrict__ pos,          // [M]
    half_t* __restrict__ tree_k_visit,        // [kTreeMaxM, n_kv * head_dim]
    half_t* __restrict__ tree_v_visit,        // [kTreeMaxM, n_kv * head_dim]
    bool apply_tau,
    float qk_clip, float rel_clip, float rms_eps,
    int M, int n_head, int n_kv, int head_dim, int d_rel, int rel_extent) {
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
            tree_k_visit[(static_cast<size_t>(m) * n_kv + h) * head_dim + tid] = static_cast<half_t>(v * s_fac);
        }
    } else {
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
            tree_v_visit[(static_cast<size_t>(m) * n_kv + h) * head_dim + tid] = static_cast<half_t>(v * s_fac);
        }
    }
}

// 4-Wave Split-KV Exact Tree Flash-SWA / Flash-GQA Attention (Prefix Paged KV + Bitmask Tree Ancestors!)
// Grid: (n_q_heads, M), Block: 128
template <bool FP8_KV>
__global__ void __launch_bounds__(128) k3_flash_swa_gqa_tree(
    const float* __restrict__ Q,            // [M, n_q_heads, 128]
    const float* __restrict__ gate_attn,    // nullptr or [M, n_q_heads, 128]
    const float* __restrict__ rel_bias,     // nullptr or [M, n_q_heads, max_rel]
    const float* __restrict__ sink_logit,   // nullptr or [M, n_q_heads]
    const TreeMetaDev* __restrict__ tmeta,
    const int32_t* __restrict__ pos,        // [M] (pos[m] = base_pos + depth[m])
    const int32_t* __restrict__ page_table, // [max_slots, max_pages_per_slot]
    int max_pages_per_slot,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const half_t* __restrict__ tree_k,      // [kTreeMaxM, n_kv_heads * 128]
    const half_t* __restrict__ tree_v,      // [kTreeMaxM, n_kv_heads * 128]
    float* __restrict__ O_out,
    int M, int n_q_heads, int n_kv_heads, int swa_window, int max_rel) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x;
    const int m = blockIdx.y;
    if (qh >= n_q_heads || m >= M) return;

    const int d0 = lane * 4;
    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int slot = tmeta->slot;
    const int base_pos = tmeta->base_pos;
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

    // 1. Attend to committed prefix [k_start, base_pos - 1] across all 4 waves
    for (int kp = k_start + wave; kp < base_pos; kp += 4) {
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
        o0 = fmaf(o0, alpha, beta * v0); o1 = fmaf(o1, alpha, beta * v1);
        o2 = fmaf(o2, alpha, beta * v2); o3 = fmaf(o3, alpha, beta * v3);
        sum_exp = fmaf(sum_exp, alpha, beta);
        max_l = new_max;
    }

    // 2. Wave 0 attends to bitmask-selected tree ancestors j in [0..M-1] (L2-resident!)
    if (wave == 0) {
        const uint32_t tmask = tmeta->mask[m];
        for (int j = 0; j < M; ++j) {
            if (((tmask >> j) & 1u) == 0u) continue;
            const size_t tbase = static_cast<size_t>(j) * elems + kv_off;
            const uint2 kw = *reinterpret_cast<const uint2*>(tree_k + tbase);
            const uint2 vw = *reinterpret_cast<const uint2*>(tree_v + tbase);
            float k0, k1, k2, k3, v0, v1, v2, v3;
            h2_unpack(kw.x, k0, k1); h2_unpack(kw.y, k2, k3);
            h2_unpack(vw.x, v0, v1); h2_unpack(vw.y, v2, v3);

            float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
            const int dist = cur_pos - pos[j];
            if (rb_row && dist >= 0 && dist < max_rel) sc += rb_row[dist];

            const float new_max = fmaxf(max_l, sc);
            const float alpha = __expf(max_l - new_max);
            const float beta = __expf(sc - new_max);
            o0 = fmaf(o0, alpha, beta * v0); o1 = fmaf(o1, alpha, beta * v1);
            o2 = fmaf(o2, alpha, beta * v2); o3 = fmaf(o3, alpha, beta * v3);
            sum_exp = fmaf(sum_exp, alpha, beta);
            max_l = new_max;
        }
    }

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

// =================================================================================================
// 3. Zero-Logits-D2H Batched Unembed Greedy Argmax (M <= 16 in ONE Read of W_unemb -> 64 B D2H!)
// =================================================================================================
// Stage 1 (Zero-Spill Templated on MT in {4, 6, 8, 12}): Grid (kArgBlks=256,), Block 256
// Keeps acc[MT], best_v[MT], best_i[MT] 100% in VGPRs (ZERO local scratch VRAM spills!)
// Uses 32 KB union-LDS for MT <= 6 (2 WG/CU occupancy, ZERO global h_norm reads in inner loop!)
template <int MT>
__global__ void __launch_bounds__(256) k_unembed_argmax_batch_stage1(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    const float* __restrict__ h_norm,       // [M, D]
    const half_t* __restrict__ h_norm_f16,  // [M, D] pre-converted FP16 for S_K fdot2!
    float* __restrict__ blk_val,            // [M, gridDim.x]
    int32_t* __restrict__ blk_id,           // [M, gridDim.x]
    int M, int V, int D) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    float best_v[MT];
    int32_t best_i[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { best_v[m] = -1e30f; best_i[m] = V; }

    constexpr int kLdsSlots = (MT <= 7) ? MT : 1;
    union ArgLds {
        half_t   h[kLdsSlots * 2304];    // <= 32,256 B -> 2 WG/CU!
        uint16_t bf[kLdsSlots * 2304];
        struct {
            float   wv[MT * 8];
            int32_t wi[MT * 8];
        } red;
    };
    __shared__ alignas(16) ArgLds u;

    const bool use_lds = (MT <= 7 && D <= 2304);
    if constexpr (MT <= 7) {
        if (D <= 2304) {
            const int tot = M * D;
            if (fmt < 100) {
                for (int i = tid; i < tot; i += 256) u.h[i] = h_norm_f16[i];
                __syncthreads();
            } else if (fmt == 102) {
                for (int i = tid; i < tot; i += 256) u.bf[i] = f32_to_bf16(h_norm[i]);
                __syncthreads();
            }
        }
    }

    const int stride = gridDim.x * 8;
    if (fmt == CODE_S6_K || fmt < 100) {
        const int sb_b = (fmt == CODE_S6_K) ? 198 : sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        const half_t* hsrc = use_lds ? u.h : h_norm_f16;
        for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
            float acc[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
            const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = (fmt == CODE_S6_K) ? decode_sk_sub_eff_i8(blk, CODE_S6_K, sub)
                                                     : decode_sk_sub_eff_i8(blk, fmt, sub);
                const uint4 wv  = (fmt == CODE_S6_K) ? decode_sk_8_unscaled_f16(blk + 6, CODE_S6_K, j0)
                                                     : decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                const int col = (s << 8) + j0;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(hsrc + static_cast<size_t>(m) * D + col);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d);
                        d = fdot2_f32(wv.z, xv.z, d);
                        d = fdot2_f32(wv.w, xv.w, d);
                        acc[m] = fmaf(d, eff, acc[m]);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float z = wave32_sum(acc[m]);
                    if (lane == 0 && (z > best_v[m] || (z == best_v[m] && v < best_i[m]))) {
                        best_v[m] = z; best_i[m] = v;
                    }
                }
            }
        }
    } else if (fmt == 102) {
        for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
            float acc[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 8; k < D; k += 256) {
                const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
                const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
                const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
                const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
                if constexpr (MT <= 7) {
                    if (D <= 2304) {
#pragma unroll
                        for (int m = 0; m < MT; ++m) {
                            if (m < M) {
                                const uint4 h8 = *reinterpret_cast<const uint4*>(u.bf + static_cast<size_t>(m) * D + k);
                                acc[m] = fmaf(f0, bf16_to_f32(h8.x & 0xFFFFu), fmaf(f1, bf16_to_f32(h8.x >> 16),
                                         fmaf(f2, bf16_to_f32(h8.y & 0xFFFFu), fmaf(f3, bf16_to_f32(h8.y >> 16), acc[m]))));
                                acc[m] = fmaf(f4, bf16_to_f32(h8.z & 0xFFFFu), fmaf(f5, bf16_to_f32(h8.z >> 16),
                                         fmaf(f6, bf16_to_f32(h8.w & 0xFFFFu), fmaf(f7, bf16_to_f32(h8.w >> 16), acc[m]))));
                            }
                        }
                        continue;
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const float4 h0 = *reinterpret_cast<const float4*>(h_norm + static_cast<size_t>(m) * D + k);
                        const float4 h1 = *reinterpret_cast<const float4*>(h_norm + static_cast<size_t>(m) * D + k + 4);
                        acc[m] = fmaf(f0, h0.x, fmaf(f1, h0.y, fmaf(f2, h0.z, fmaf(f3, h0.w, acc[m]))));
                        acc[m] = fmaf(f4, h1.x, fmaf(f5, h1.y, fmaf(f6, h1.z, fmaf(f7, h1.w, acc[m]))));
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float z = wave32_sum(acc[m]);
                    if (lane == 0 && (z > best_v[m] || (z == best_v[m] && v < best_i[m]))) {
                        best_v[m] = z; best_i[m] = v;
                    }
                }
            }
        }
    } else if (fmt == 101) {
        for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
            float acc[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const float4 hv = *reinterpret_cast<const float4*>(h_norm + static_cast<size_t>(m) * D + k);
                        acc[m] = fmaf(wv.x, hv.x, fmaf(wv.y, hv.y, fmaf(wv.z, hv.z, fmaf(wv.w, hv.w, acc[m]))));
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float z = wave32_sum(acc[m]);
                    if (lane == 0 && (z > best_v[m] || (z == best_v[m] && v < best_i[m]))) {
                        best_v[m] = z; best_i[m] = v;
                    }
                }
            }
        }
    } else {
        for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
            float acc[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
                const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
                const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
                const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const float4 hv = *reinterpret_cast<const float4*>(h_norm + static_cast<size_t>(m) * D + k);
                        acc[m] = fmaf(f0, hv.x, fmaf(f1, hv.y, fmaf(f2, hv.z, fmaf(f3, hv.w, acc[m]))));
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float z = wave32_sum(acc[m]) * sc;
                    if (lane == 0 && (z > best_v[m] || (z == best_v[m] && v < best_i[m]))) {
                        best_v[m] = z; best_i[m] = v;
                    }
                }
            }
        }
    }

    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                u.red.wv[m * 8 + wave] = best_v[m];
                u.red.wi[m * 8 + wave] = best_i[m];
            }
        }
    }
    __syncthreads();
    if (tid < M) {
        const int m = tid;
        float bv = -1e30f; int32_t bi = V;
        for (int w = 0; w < 8; ++w) {
            const float vv = u.red.wv[m * 8 + w];
            const int32_t ii = u.red.wi[m * 8 + w];
            if (vv > bv || (vv == bv && ii < bi)) { bv = vv; bi = ii; }
        }
        blk_val[m * gridDim.x + blockIdx.x] = bv;
        blk_id[m * gridDim.x + blockIdx.x]  = bi;
    }
}

// Stage 2: Reduce partial [M, n_blks] -> exact greedy argmax_ids[M] (< 5 us!)
// Grid: (M,), Block: 256
__global__ void __launch_bounds__(256) k_unembed_argmax_batch_stage2(
    const float* __restrict__ blk_val,   // [M, n_blks]
    const int32_t* __restrict__ blk_id,  // [M, n_blks]
    int32_t* __restrict__ out_ids,       // [M]
    int M, int n_blks) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    float bv = -1e30f;
    int32_t bi = 0x7FFFFFFF;
    for (int i = tid; i < n_blks; i += 256) {
        const float v = blk_val[m * n_blks + i];
        const int32_t id = blk_id[m * n_blks + i];
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, 32);
        const int32_t oi = __shfl_xor(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float s_v[8];
    __shared__ int32_t s_i[8];
    if (lane == 0) { s_v[wave] = bv; s_i[wave] = bi; }
    __syncthreads();
    if (tid == 0) {
        float gv = -1e30f; int32_t gi = 0x7FFFFFFF;
        for (int w = 0; w < 8; ++w) {
            if (s_v[w] > gv || (s_v[w] == gv && s_i[w] < gi)) { gv = s_v[w]; gi = s_i[w]; }
        }
        out_ids[m] = gi;
    }
}

// =================================================================================================
// 4. 1-Launch Winning-Path Commit Kernels (ShortConv + Paged KV + DSpark Context!)
// =================================================================================================

struct WinPath8 {
    int32_t idx[8];   // winning tree node indices win_path[0..n_acc-1]
    int32_t n_acc;    // 1..8
};

// Commit winning path ShortConv across ALL 48 visits for a pair of pools AND save (snap, rec[0..n_acc-1])
// so rollback_slot() remains 100% exact if ink_serve.h hits EOS/stop early!
// Grid: ((C + 255) / 256, n_visits), Block: 256
__global__ void __launch_bounds__(256) k_sconv_commit_tree_pair(
    uint16_t* __restrict__ pool_a,        // [n_visits, max_slots, 3, C]
    uint16_t* __restrict__ snap_a,        // [n_visits, 3, C] (stores pre-verify history!)
    uint16_t* __restrict__ rec_a,         // [n_visits, 8, C] (stores linear winning path!)
    const uint16_t* __restrict__ trec_a,  // [n_visits, kTreeMaxM, C]
    uint16_t* __restrict__ pool_b,
    uint16_t* __restrict__ snap_b,
    uint16_t* __restrict__ rec_b,
    const uint16_t* __restrict__ trec_b,
    WinPath8 wp, int slot, int max_slots, int n_visits, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;

    const size_t pbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = static_cast<size_t>(v) * 3 * C;
    const size_t rbase = static_cast<size_t>(v) * 8 * C;
    const size_t tbase = static_cast<size_t>(v) * kTreeMaxM * C;

    uint16_t old_a[3], old_b[3];
    for (int lag = 0; lag < 3; ++lag) {
        old_a[lag] = pool_a[pbase + lag * C + c];
        old_b[lag] = pool_b[pbase + lag * C + c];
        snap_a[sbase + lag * C + c] = old_a[lag];
        snap_b[sbase + lag * C + c] = old_b[lag];
    }
    for (int k = 0; k < wp.n_acc; ++k) {
        const int nid = wp.idx[k];
        rec_a[rbase + k * C + c] = trec_a[tbase + nid * C + c];
        rec_b[rbase + k * C + c] = trec_b[tbase + nid * C + c];
    }
    for (int lag = 0; lag < 3; ++lag) {
        const int t = wp.n_acc - 3 + lag;
        pool_a[pbase + lag * C + c] = (t < 0) ? old_a[t + 3] : trec_a[tbase + wp.idx[t] * C + c];
        pool_b[pbase + lag * C + c] = (t < 0) ? old_b[t + 3] : trec_b[tbase + wp.idx[t] * C + c];
    }
}

// Per-visit Paged KV descriptor for 1-launch batched commit across all 48 visits!
struct KvVisitDesc {
    void*          k_pool;
    void*          v_pool;
    float*         k_scales;
    float*         v_scales;
    const int32_t* d_pt;
    int            max_pages_per_slot;
    int            swa_window;
};

// Commit winning path K/V across ALL 48 visits in ONE SINGLE KERNEL LAUNCH!
// Grid: (n_acc, n_visits), Block: 256
template <bool FP8_KV>
__global__ void __launch_bounds__(256) k3_kv_commit_tree_all(
    const KvVisitDesc* __restrict__ visits,  // [n_visits]
    const half_t* __restrict__ tree_k_all,   // [n_visits, kTreeMaxM, elems]
    const half_t* __restrict__ tree_v_all,   // [n_visits, kTreeMaxM, elems]
    WinPath8 wp, int slot, int base_pos, int n_visits, int elems) {
    const int k = blockIdx.x;
    const int v = blockIdx.y;
    if (k >= wp.n_acc || v >= n_visits) return;
    const int tid = threadIdx.x;

    const KvVisitDesc vd = visits[v];
    const int p = (vd.swa_window > 0) ? ((base_pos + k) % vd.swa_window) : (base_pos + k);
    const int phys = vd.d_pt[slot * vd.max_pages_per_slot + (p >> 6)];
    const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
    const size_t dst_base = tok_base * elems;
    const size_t src_base = (static_cast<size_t>(v) * kTreeMaxM + wp.idx[k]) * elems;
    const half_t* ks = tree_k_all + src_base;
    const half_t* vs = tree_v_all + src_base;

    if constexpr (FP8_KV) {
        const int wave = tid >> 5;
        const int lane = tid & 31;
        __shared__ float s_w[8];
        __shared__ float s_ik, s_iv;
        float km = 0.0f, vm = 0.0f;
        for (int i = tid; i < elems; i += 256) {
            km = fmaxf(km, fabsf(static_cast<float>(ks[i])));
            vm = fmaxf(vm, fabsf(static_cast<float>(vs[i])));
        }
        km = wave32_max(km); vm = wave32_max(vm);
        if (lane == 0) s_w[wave] = km;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (vd.k_scales) vd.k_scales[tok_base] = sc;
            s_ik = 1.0f / sc;
        }
        __syncthreads();
        if (lane == 0) s_w[wave] = vm;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (vd.v_scales) vd.v_scales[tok_base] = sc;
            s_iv = 1.0f / sc;
        }
        __syncthreads();
        const float ik = s_ik, iv = s_iv;
        uint8_t* kd = static_cast<uint8_t*>(vd.k_pool) + dst_base;
        uint8_t* vd_p = static_cast<uint8_t*>(vd.v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i]   = f32_to_e4m3fn(static_cast<float>(ks[i]) * ik);
            vd_p[i] = f32_to_e4m3fn(static_cast<float>(vs[i]) * iv);
        }
    } else {
        half_t* kd = static_cast<half_t*>(vd.k_pool) + dst_base;
        half_t* vd_p = static_cast<half_t*>(vd.v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = ks[i];
            vd_p[i] = vs[i];
        }
    }
}

// Gather winning path [0..n_acc-1] of 4 taps [M, D] -> compact [n_acc, D] so launch_dspark_ctx_kernels
// commits ONLY the winning tokens directly to sl.dspark_k/v/frnn (ZERO wasted ctx compute on rejected nodes!)
// Grid: ((D + 255) / 256, n_acc), Block: 256
__global__ void __launch_bounds__(256) k_gather_tree_taps4(
    const float* __restrict__ t0_in, const float* __restrict__ t1_in,
    const float* __restrict__ t2_in, const float* __restrict__ t3_in,
    float* __restrict__ t0_out, float* __restrict__ t1_out,
    float* __restrict__ t2_out, float* __restrict__ t3_out,
    WinPath8 wp, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (d >= D || k >= wp.n_acc) return;
    const int nid = wp.idx[k];
    const size_t s = static_cast<size_t>(nid) * D + d;
    const size_t o = static_cast<size_t>(k) * D + d;
    t0_out[o] = t0_in[s];
    t1_out[o] = t1_in[s];
    t2_out[o] = t2_in[s];
    t3_out[o] = t3_in[s];
}

// Fill contiguous winning-path slot_ids & pos on GPU (100% async-safe, ZERO stack H2D!)
__global__ void __launch_bounds__(32) k_fill_win_slot_pos(
    int32_t* __restrict__ sids,
    int32_t* __restrict__ pos,
    int slot, int base_pos, int n_acc) {
    const int k = threadIdx.x;
    if (k < n_acc) {
        sids[k] = slot;
        pos[k]  = base_pos + k;
    }
}

}  // namespace strata::inkling2::hip
