// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/glm5.hpp - GLM-5.3-Flash's (glm5-next) own pieces: the mHC streams, the sigmoid router, the
// KDA gate and step, and the clamped SwiGLU.  The arithmetic follows llama.cpp's src/models/glm5-next.cpp and the ggml
// CPU operators it uses (ggml_dsv4_hc_comb, ggml_gated_delta_net with a per-channel gate, ggml_swiglu_clamp) at
// f9a95ae; MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.
//
// Every buffer is a device pointer; `stream` null runs synchronously.  T is the number of tokens, N the model width.
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <cstdint>

namespace strata::kernels {

/// The mHC streams' read before a block (llama.cpp's build_hc_pre), for the hc = 4 streams.
///   x      [T][4][N]   the streams
///   mixes  [T][24]     hc_fn applied to the token's 4N stream values as they are: their RMS normalisation (eps
///                      `rms_eps`, no weight) is a factor of the product and is applied here
///   scale  [3], base [24]   hc_*_scale and hc_*_base
/// Out: y [T][N] = sum_s pre[s] x[s], pre = sigmoid(mix[0..4) scale[0] + base[0..4)) + hc_eps;
///      post [T][4] = 2 sigmoid(mix[4..8) scale[1] + base[4..8));
///      comb [T][16], comb[dst + 4 src]: mix[8..24) scale[2] + base[8..24), softmax over dst, + hc_eps, then the
///      columns (sum over src) normalised, then `iters` - 1 rounds of rows (sum over dst) and columns, each sum + hc_eps.
/// With streams = 2 (SenseNova-Loop), mixes [T][8] are pre[2], post[2], row-major comb[2][2]; base [8] is
/// added after RMS normalization, scale and hc_eps are unused, and pre is sigmoid times pre_scale. The comb is
/// normalized in log space, rows then columns for each iteration, and exponentiated at the end.
void glm5_hc_pre(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps, float hc_eps,
                 int iters, float* y, float* post, float* comb, int64_t n_embd, int64_t n_tokens, void* stream,
                 int streams = 4, float pre_scale = 1.0f);

/// The streams before the first block: every stream the token's embedding, hs[t][s] = x[t] (llama.cpp's hc_init).
void glm5_hc_init(const float* x, float* hs, int64_t n_embd, int64_t n_tokens, void* stream, int streams = 4);

/// dst[b][a][d] = src[a][b][d] for A x B blocks of D floats: [token][head][dim] to [head][token][dim] and back, so a
/// head's rows of every token lie together for its own matrix (the MLA's wk_b and wv_b).
void glm5_transpose_abd(const float* src, float* dst, int64_t a, int64_t b, int64_t d, void* stream);

/// The mHC streams' write after a block (llama.cpp's build_hc_post): out[t][d] = f[t] post[t][d] + sum_s
/// residual[t][s] comb[t][d + 4 s]. For 2 streams, comb is row-major [dst][src]. `out` may be `residual`.
void glm5_hc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
                  int64_t n_embd, int64_t n_tokens, void* stream, int streams = 4);

/// The streams' mean before the output norm: y[t] = (x[t][0] + x[t][1] + x[t][2] + x[t][3]) / 4.
void glm5_hc_mean(const float* x, float* y, int64_t n_embd, int64_t n_tokens, void* stream, int streams = 4);

/// The sigmoid router (llama.cpp's build_moe_ffn with gating SIGMOID and exp_probs_b): p = sigmoid(logits); the k
/// experts with the largest p + bias (ties to the lower index), weights p of those, divided by their sum (clamped
/// below at 2**-14) when `norm`, times `scale`.  logits [T][E], bias [E]; ids, weights [T][k].  k <= 32.
void glm5_router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm,
                         float scale, int32_t* ids, float* weights, void* stream);

/// KDA's per-channel decay (llama.cpp's build_kda_layer with a gate lower bound): g = lb sigmoid(-(f + dt_bias) a),
/// a = ssm_a (it holds -exp(A_log)) per head.  f, g [T][H][S]; dt_bias [H S]; ssm_a [H].  lb finite and below 0.
void glm5_kda_gate(const float* f, const float* dt_bias, const float* ssm_a, float lb, float* g, int64_t n_heads,
                   int64_t head_dim, int64_t n_tokens, void* stream);

/// KDA's short convolution of one projection (q, k or v) with its SiLU (llama.cpp's glm5_conv1d, ggml_ssm_conv):
/// y[t][c] = silu(sum_j w[c][j] window_t[j]), window_t the 3 inputs before token t then its own, oldest first.  The
/// inputs before the window are state[c][0..3) (oldest first); `update` leaves the window's last 3 there.
/// x, y [T][C]; w [C][4] (GGUF [4, 1, C]).  d_conv = 4.
void glm5_kda_conv(const float* x, float* state, const float* w, float* y, int64_t n_channels, int64_t n_tokens,
                   bool update, void* stream);

/// x[t][h] /= max(|x[t][h]|, eps) in place (ggml_l2_norm per head; FLA's eps 1e-6).  x [T][H][S], S = 128.
void glm5_l2_heads(float* x, int64_t n_heads, float eps, int64_t n_tokens, void* stream);

/// KDA's output before attn_output: y = RMSNorm(o[t][h]) w sigmoid(g2[t][h]) per head (eps `eps`).  o, g2, y
/// [T][H][S], w [S], S = 128.
void glm5_kda_out(const float* o, const float* g2, const float* w, float eps, float* y, int64_t n_heads,
                  int64_t n_tokens, void* stream);

/// One KDA step (ggml_gated_delta_net with a gate per key channel), IN PLACE on `state`, for S = 128:
///   state (S, H, S) f32 as Strata's GDN state, state[(i H + h) S + j] = S_h[i][j] (i the key channel)
///   q, k  (H, S) L2-normalised, q NOT scaled (1/sqrt(S) is applied to the readout)
///   v     (H, S);  g (H, S) the PRE-exp decay of each key channel;  beta (H,) after its sigmoid
///   out   (H, S) = (S_h' q) / sqrt(S), S_h' = diag(exp g) S_h + k delta^T, delta = (v - (diag(exp g) S_h)^T k) beta
void glm5_kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, void* stream);
/// glm5_kda_step for n_tokens in order in one launch: q, k, v, g, out [T][H][S], beta [T][H].  A state column's
/// recurrence depends on no other column, so each walks the tokens by itself; every value is the per-token calls'.
/// `snaps` (n_snaps of them, at most kGlm5MaxSnaps): the state after rows 0 .. n_snaps - 1 written there too, for a
/// speculative window to go back to the last accepted row.
inline constexpr int kGlm5MaxSnaps = kWindowMaxT;   ///< (a window of up to 8 rows passes the 8 it always did)
void glm5_kda_scan(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, int64_t n_tokens, void* stream, float* const* snaps = nullptr,
                   int n_snaps = 0);

// ---- the DSA layers: the k-pool indexer and the MLA attention over its selection.  Cells are sequence positions
// from 0 (the step records' kStepPos, qsa.hpp); a pool is the `kpool` cells [kpool m, kpool m + kpool), and a token at
// position p chooses among the pools that end at or before it, (p + 1) / kpool of them; with indexer.kpool_select_tail
// (llama.cpp's default where the file has no such key, as GLM-5.3-Flash's) it also attends the cells of its own
// unfinished pool up to itself, [kpool ((p + 1) / kpool), p].

/// The selection's step records from the window's (qsa.hpp's layout): kStepPos copied, kStepNBid = (pos + 1) / kpool
/// complete pools, kStepNKv = kpool times that (so the unfinished pool weighs no cell in qsa_block_topk), and
/// kStepWidth = min(kStepNKv, top_k).  steps_in, steps_out [T][kStepCount].
void glm5_idx_steps(const int32_t* steps_in, int64_t n_tokens, int kpool, int top_k, int32_t* steps_out, void* stream);

/// The indexer's key and pool gate of each token into the cache rows of its cell: key = LayerNorm(k_raw) w + b (eps
/// `ln_eps`), gate = g_raw.  k_raw, g_raw [T][D]; cache [cells][2 D] = key | gate, row steps[t][kStepPos].  D = 128.
void glm5_idx_append(const float* k_raw, const float* g_raw, const float* ln_w, const float* ln_b, float ln_eps,
                     const int32_t* steps, int64_t n_tokens, float* cache, void* stream);

/// The pooled key of every pool a token of the window completes ((pos + 1) % kpool == 0): pooled[m][c] =
/// sum_j softmax_j(gate[kpool m + j][c] + ape[j][c]) key[kpool m + j][c].  ape [kpool][D] (GGUF ne0 = D);
/// pooled [pools][D].  D = 128.
void glm5_idx_pool(const float* cache, const float* ape, int kpool, const int32_t* steps, int64_t n_tokens,
                   float* pooled, void* stream);

/// The pools' scores for each token: score[t][b] = sum_h relu(q[t][h] . pooled[b]) w[t][h] for b below the token's
/// kStepNBid (glm5_idx_steps), 0 at b = kStepNBid.  q [T][H][D] (indexer_attn_q_b applied to the normed q_a), w [T][H]
/// (indexer_proj applied to the block input and scaled by 1/sqrt(D H)).  scores [T][max_blocks].  D = 128.
void glm5_idx_scores(const float* pooled, const float* q, const float* w, int n_heads, const int32_t* steps,
                     int64_t n_tokens, int64_t max_blocks, float* scores, void* stream);

/// The MLA latent of each token into its cell's FP16 row: latent [cells][L] from kv [T][L] (L = kv_lora_rank).
void glm5_latent_store(const float* kv, const int32_t* steps, int64_t n_tokens, int64_t rank, uint16_t* latent,
                       void* stream);

/// The absorbed MLA attention over each token's selection: for head h, p_j = softmax_j(q[t][h] . latent[c_j] scale),
/// out[t][h] = sum_j p_j latent[c_j], over the steps[t][kStepWidth] cells ids[t][0 ..) (qsa_block_topk's output, cap
/// apart) and, with `tail_kpool` (the pool size; 0 for none), the token's unfinished pool's cells up to it.  A token
/// with no cell gets zeros.  q, out [T][H][L] (q already through wk_b); latent [cells][L] FP16.  L = 512.
/// `compact` > 0: `latent` is glm5_latent_gather's [T][compact][L] rows instead, token t's cell j at row t compact + j.
void glm5_mla_attn(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                   int n_heads, float scale, int64_t n_tokens, float* out, void* stream, int tail_kpool = 0,
                   int64_t compact = 0);

/// The cells glm5_mla_attn reads for each token (its selection, then its unfinished pool's cells up to it) copied
/// from `latent` [cells][L] into out [T][stride][L], in that order: one read of each over PCIe when the latent lives
/// in host memory (--kv-ram), where the attention would read each cell twice for each head.  stride >= cap +
/// tail_kpool - 1.  L = 512.
void glm5_latent_gather(const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                        int64_t n_tokens, int tail_kpool, int64_t stride, uint16_t* out, void* stream);

/// GLM's clamped SwiGLU (ggml_swiglu_clamp): out = silu(min(gate, limit)) clamp(up, -limit, limit), n values.
/// `out` may be `gate` or `up`.
void glm5_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream);

}  // namespace strata::kernels
