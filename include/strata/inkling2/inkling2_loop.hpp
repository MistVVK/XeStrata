// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/inkling2/inkling2_loop.hpp - Native ModelFamily::Inkling2Loop + DSpark Block-8 Markov primitives
#pragma once

#include "strata/quant/sk_codecs.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::inkling2 {

constexpr float kInklingRmsEps = 1e-6f;
constexpr float kSituBeta      = 4.0f;
constexpr float kSituLb        = 25.0f;
constexpr float kQkL2Clip      = 22.627416997969522f;
constexpr float kRelL2Clip     = 8.0f;
constexpr float kLogScaleAlpha = 0.1f;
constexpr int   kLogScaleFloor = 8192;
constexpr float kLogitSoftcap  = 30.0f;

// Physical layer mapping for 24 physical layers -> 48 virtual visits (core [6, 18) x 3 passes)
inline int physical_layer_for_visit(int visit, int core_start = 6, int core_end = 18, int num_passes = 3) noexcept {
    const int core_len = core_end - core_start;
    if (visit < core_start) return visit;
    if (visit < core_start + num_passes * core_len) {
        return core_start + ((visit - core_start) % core_len);
    }
    return core_end + (visit - core_start - num_passes * core_len);
}

inline int pass_for_visit(int visit, int core_start = 6, int core_end = 18, int num_passes = 3) noexcept {
    const int core_len = core_end - core_start;
    if (visit < core_start || visit >= core_start + num_passes * core_len) return 0;
    return (visit - core_start) / core_len;
}

float situ_scalar(float g, float u, float beta = kSituBeta, float lb = kSituLb) noexcept;

void rms_norm_f32(const float* x, const float* w, float* out, size_t n, float eps = kInklingRmsEps) noexcept;

void l2_clip_f32(const float* x, const float* scale, float* out, size_t n, float threshold) noexcept;

float log_pos_tau(int pos, int n_floor = kLogScaleFloor, float alpha = kLogScaleAlpha) noexcept;

// Attention-Residual depth mix: mixes [blocks..., cur] (RMSNormed, eps=1e-6) against n_queries vectors [n_queries, D]
void attn_res_mix_f32(
    const std::vector<const float*>& blocks,
    const float* cur,
    const float* queries,  // [n_queries, D]
    size_t n_queries,
    size_t D,
    float* out) noexcept;  // [n_queries, D]

// Rank-64 tanh Gain/Forget Loop Message Controller:
// gate = down([rms(cur); rms(msg)]), gain = tanh(W_gain @ gate), forget = tanh(W_forget @ gate)
// out = (1 - forget) * cur + gain * msg
void loop_control_f32(
    const float* cur,       // [D]
    const float* msg,       // [D]
    const float* w_down,    // [rank, 2*D]
    const float* w_gain,    // [D, rank]
    const float* w_forget,  // [D, rank]
    size_t D,
    size_t rank,
    float* out) noexcept;   // [D]

// Single-token 4-tap causal ShortConv step (updates history[3, C] in-place)
// static_w: [C, 4] (oldest lag t-3 at index 0 .. current t at index 3)
// dynamic_taps (optional, global layers): [n_heads, 4] (lag 0=t, lag 1=t-1, lag 2=t-2, lag 3=t-3)
void shortconv4_step_f32(
    const float* cur,          // [C]
    const float* static_w,     // [C, 4]
    const float* dynamic_taps, // nullptr or [n_heads, 4]
    size_t C,
    size_t n_heads,
    float* history,            // [3, C] (t-3, t-2, t-1) updated in-place
    float* out) noexcept;      // [C]

// Execute one Inkling-2-Mini expert (w13 interleaved even=gate/odd=up + SITU + w2 down) from raw Section 4 bytes.
// Supports SF4 (code_id=15), MX8 (code_id=16), a GGUF file's Q8_0 body (GGUF_RAW, code_id=20, interleaved rows) and
// S1.5..S8_K (with paired 2-row w2 SB_0..SB_8 & FP16 sq_xd_dn).
void eval_inkling2_expert(
    const uint8_t* body,
    strata::quant::sk::SkCode gu_code,
    strata::quant::sk::SkCode dn_code,
    uint32_t dn_offset,
    const float* x_in,         // [hidden]
    size_t hidden,             // 2304
    size_t inter,              // 1152
    float* y_out) noexcept;    // [hidden]

// Batched multi-token expert evaluation for m tokens (m = 1..64) sharing one expert body read!
void eval_inkling2_expert_batch(
    const uint8_t* body,
    strata::quant::sk::SkCode gu_code,
    strata::quant::sk::SkCode dn_code,
    uint32_t dn_offset,
    const float* x_in,         // [m, hidden]
    size_t m,
    size_t hidden,             // 2304
    size_t inter,              // 1152
    float* y_out) noexcept;    // [m, hidden]

// DSpark Block-8 Recurrent Markov head step (one slot i in 1..block_size-1):
// Updates state [rank] in-place and produces softcapped logits [V_unpadded] + deterministic first-max argmax token.
int32_t dspark_markov_step_f32(
    int32_t prev_token,
    float* state,                 // [rank] in/out
    const float* hidden,          // [D]
    const float* context,         // [D] (f_rnn from 4x4 dspark_fuse row 3)
    const float* base_logits,     // [V_unpadded]
    const float* markov_w1,       // [V_padded, rank]
    const float* rnn_proj_w,      // [3*rank, 2*rank + 2*D]
    const float* rnn_proj_b,      // [3*rank]
    const float* hidden_out_w,    // [D, rank]
    const float* hidden_out_b,    // [D]
    const float* unembed_w,       // [V_unpadded, D]
    size_t D,
    size_t rank,
    size_t V_unpadded,
    float softcap,
    float* out_logits) noexcept;  // [V_unpadded]

}  // namespace strata::inkling2
