// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/plain_prompt_glm5.cpp - GLM-5.3-Flash's prompt in chunks of batched products (PlainModel::prompt).
//
// The block of plain_glm5.cpp (after llama.cpp's glm5-next graph at f9a95ae; MIT License, Copyright (c) 2023-2026 The
// ggml authors, see third_party/main/ggml/LICENSE), T tokens at a time: every projection a matrix product of the
// chunk's rows (the int8 products where the weight's type and shape are theirs, else the weight dequantized to FP16),
// the mHC, the KDA's convolutions, gates and norms, the indexer, the MLA attention and the router the window's
// kernels over the chunk's rows (they take any number of rows), the KDA recurrence in one scan, the routed
// experts grouped by expert (prompt_routed).  The state is left as the windows leave it: the KDA recurrence and
// convolution histories, the indexer's cells and pooled keys and the MLA latent of the chunk's positions.
#include "strata/core/plain_model.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/platform/env.hpp"
#include "plain_prompt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

namespace {
constexpr int kHC = 4;
constexpr float kL2Eps = 1e-6f;   // llama.cpp's FLA l2 norm
int file_type(int t) { return t == k::kNativeQ6KRows ? 14 : t == k::kNativeQ8Rows ? 8 : t; }
}  // namespace

bool PlainModel::prompt_chunks_glm5(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                                    std::string& err, int64_t c_begin, int64_t c_end, const float* hand_in) {
    PromptBufs& P = pb_;
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, S = s.kda_head_dim, C = H * S, E = s.n_expert, K = s.n_expert_used;
    const int64_t QLR = s.q_lora_rank, KVR = s.kv_lora_rank, QK = s.key_length_mla, VD = s.value_length_mla;
    const int64_t IH = s.idx_n_head, ID = s.idx_key_length;
    const float eps = (float) s.rms_eps, hc_eps = (float) s.hc_eps;
    const int iters = (int) s.hc_sinkhorn_iters;
    const k::QsaShapes qs = attn_shapes();   // the step records' positions (glm5_idx_steps reads kStepPos only)
    k::QsaShapes sel;
    sel.idx_block = s.idx_kpool;
    sel.idx_top_k = s.idx_top_k;
    strata::gpu::Stream cs = cs_;
    PromptGemm& pg = prompt_gemm();
    prefill::Gemm& gm = pg.gemm;
    // x [T][n_in] (ldx a row) through the weight into y [T][n_out] (ldy a row): int8 where the weight's type and shape
    // are the int8 products', else from an FP16 copy of x (contiguous rows)
    static const bool dense_env = [] {   // STRATA_PLAIN_DENSE_MMQ=0: every dense product from FP16
        const char* v = strata::env_get("PLAIN_DENSE_MMQ");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool dense_on = dense_env && prefill::mmq::chosen();
    auto prod = [&](int type, const void* w, int64_t n_in, int64_t n_out, const float* x, int64_t ldx, float* y,
                    int64_t T, int64_t ldy) {
        if (dense_on && prefill::mmq::supported(type) && n_out % 128 == 0 && n_in % 32 == 0) {
            prefill::mmq::quantize(x, nullptr, P.xq, type, n_in, ldx, T, cs);
            prefill::mmq::Product p;
            p.w = w; p.type = type; p.w_rows = n_out; p.w_cols = n_in; p.n = 1; p.xq = P.xq;
            p.bounds = P.dense_bounds; p.ids = P.identity; p.total_rows = T; p.max_rows = T;
            p.dst = y; p.ld_dst = ldy;
            pg.mmq.run(p, cs);
            return;
        }
        if (ldx != n_in) throw std::runtime_error("plain prompt (GLM): an FP16 product of strided rows");
        kernels::f32_to_f16_bulk(x, P.x16, T * n_in, cs);
        gm.native(P.x16, type, w, y, T, n_out, n_in, ldy);
    };
    auto dense = [&](const W& w, const float* x, float* y, int64_t T) {
        prod(w.type, w.data, w.n_in, w.n_out, x, w.n_in, y, T, w.n_out);
    };
    // the routed experts' CPU share, as the Qwen-family chunks take it (plain_prompt.cpp)
    static const bool cpu_env = [] {
        const char* v = strata::env_get("PLAIN_CPU_SHARE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const double share_setting = strata::prefill::cpu_share_setting();
    const bool cpu_share = cpu_env && share_setting != 0.0 &&
                           (share_setting > 0.0 || (split_.c > 0.0 && split_.p > 0.0)) && P.cpu_cap > 0;
    static const bool profile = strata::env_get("PROMPT_PROFILE") != nullptr;
    // 0 KDA (its parts 5 projections, 6 convolutions and gates, 7 recurrence, 8 output), 1 DSA, 2 the grouping, 3 the
    // experts, 4 the shared expert, mHC and dense FFN
    double stage_ms[12] = {};   // 9 the chunk's staging, 10 the head, 11 what is left at the end
    auto t_mark = std::chrono::steady_clock::now();
    auto mark = [&](int stage) {
        if (!profile) return;
        strata::gpu::stream_sync(cs);
        if (pg.xs != nullptr) strata::gpu::stream_sync(pg.xs);   // the experts' copies too
        const auto now = std::chrono::steady_clock::now();
        stage_ms[stage] += std::chrono::duration<double, std::milli>(now - t_mark).count();
        t_mark = now;
    };
    PromptRouted rst;
    if (!pg.identity) {
        prefill::mmq::iota(P.identity, P.T * K, cs);
        pg.identity = true;
    }
    try {
        const int64_t L = n_moe_;
        std::vector<int32_t> routes(disp_.usage.empty() ? 0 : (size_t) (L * P.T * K));
        const int64_t c_last = std::min(n, c_end);
        if (!first())   // the rows handed on, into this stage's own pinned memory (and its rows out from there)
            std::memcpy(hand_host_, hand_in, (size_t) ((c_last - c_begin) * kHC * N) * 4);
        for (int64_t c0 = c_begin; c0 < c_last; c0 += P.T) {
            const int64_t T = std::min<int64_t>(P.T, c_last - c0), p0 = pos + c0;
            float* hand = hand_host_ != nullptr ? hand_host_ + (c0 - c_begin) * kHC * N : nullptr;
            if (first()) embed_ahead(tokens + c0, T);
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = tokens[c0 + t];
                if (tok < 0 || tok >= n_vocab_ ||
                    (first() && !embed(tok, p0 + t, P.emb + t * N))) {
                    err = "plain prompt: a token is out of the vocabulary";
                    return false;
                }
                k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + t, qs);
            }
            P.h_dense_bounds[0] = 0;
            P.h_dense_bounds[1] = (int32_t) T;
            // the four streams from the embeddings, or as the stage before left them
            if (!(first() ? strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs)
                          : strata::gpu::copy_async(P.g_hs, hand, (size_t) (T * kHC * N) * 4, cs)) ||
                !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                !strata::gpu::copy_async(P.dense_bounds, P.h_dense_bounds, 2 * sizeof(int32_t), cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            k::glm5_idx_steps(P.steps, T, (int) s.idx_kpool, (int) s.idx_top_k, P.g_steps, cs);
            if (first()) k::glm5_hc_init(P.x, P.g_hs, N, T, cs);
            mark(9);
            // hc_pre: the four streams' mixes, then the block's row, RMS-normed with `norm` into hn
            auto hc_read = [&](const W& fn, const float* scale, const float* base, const float* norm) {
                dense(fn, P.g_hs, P.g_mixes, T);
                k::glm5_hc_pre(P.g_hs, P.g_mixes, scale, base, eps, hc_eps, iters, P.g_y, P.g_post, P.g_comb, N, T, cs);
                k::native_qsa_rms_norm_weighted(P.g_y, norm, P.hn, (int) N, (int) T, eps, cs);
            };
            auto swiglu = [&](const GBlock& b, float* dst) {   // gate / up from hn, the clamp, down
                const int64_t FW = b.ffn_gate.n_out;
                dense(b.ffn_gate, P.hn, P.g_ffg, T);
                dense(b.ffn_up, P.hn, P.g_ffu, T);
                k::glm5_swiglu_clamp(P.g_ffg, P.g_ffu, P.g_ffg, T * FW, b.clamp, cs);
                dense(b.ffn_down, P.g_ffg, dst, T);
            };
            for (int64_t l = lb_; l < le_; ++l) {
                const GBlock& b = gblocks_[(size_t) l];
                hc_read(b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, b.attn_norm);
                if (b.kda) {
                    float* state = gdn_state_[(size_t) b.idx];
                    float* conv = gdn_conv_[(size_t) b.idx];
                    const int64_t hist = C * (s.ssm_d_conv - 1);
                    dense(b.q, P.hn, P.g_q, T);
                    dense(b.k, P.hn, P.g_k, T);
                    dense(b.v, P.hn, P.g_v, T);
                    dense(b.f_a, P.hn, P.g_fa, T);
                    dense(b.g_a, P.hn, P.g_ga, T);
                    dense(b.beta, P.hn, P.g_beta, T);
                    dense(b.f_b, P.g_fa, P.g_f, T);
                    dense(b.g_b, P.g_ga, P.g_g2, T);
                    mark(5);
                    k::glm5_kda_conv(P.g_q, conv, b.conv_q, P.g_qc, C, T, true, cs);
                    k::glm5_kda_conv(P.g_k, conv + hist, b.conv_k, P.g_kc, C, T, true, cs);
                    k::glm5_kda_conv(P.g_v, conv + 2 * hist, b.conv_v, P.g_vc, C, T, true, cs);
                    k::glm5_l2_heads(P.g_qc, H, kL2Eps, T, cs);
                    k::glm5_l2_heads(P.g_kc, H, kL2Eps, T, cs);
                    k::glm5_kda_gate(P.g_f, b.dt, b.a, (float) s.kda_gate_lower_bound, P.g_gate, H, S, T, cs);
                    // the gate takes at most 65535 values a launch: the chunk's H T in slices (64 heads: a chunk past 1023
                    // tokens refused every prompt longer than that)
                    for (int64_t r = 0; r < H * T; r += 65535)
                        k::native_gdn_beta_gate(P.g_beta + r, (int) std::min<int64_t>(65535, H * T - r), cs);
                    mark(6);
                    k::glm5_kda_scan(state, P.g_qc, P.g_kc, P.g_vc, P.g_gate, P.g_beta, P.g_o, H, T, cs);
                    mark(7);
                    k::glm5_kda_out(P.g_o, P.g_g2, b.o_norm, eps, P.g_oy, H, T, cs);
                    dense(b.o, P.g_oy, P.mix, T);
                    mark(8);
                } else {
                    const int64_t d = b.idx;
                    dense(b.q_a, P.hn, P.g_qa, T);
                    dense(b.kv_a, P.hn, P.g_absT, T);   // the raw latent, normed into g_kv below
                    dense(b.idx_k, P.hn, P.g_ik, T);
                    dense(b.idx_gate, P.hn, P.g_ig, T);
                    k::f32_gemm_rows(P.hn, N, b.idx_proj, P.g_iw, IH, N, IH, T, cs);
                    k::scale_inplace(P.g_iw, T * IH, 1.0f / std::sqrt((float) (ID * IH)), cs);
                    k::native_qsa_rms_norm_weighted(P.g_absT, b.kv_a_norm, P.g_kv, (int) KVR, (int) T, eps, cs);
                    k::native_qsa_rms_norm_weighted(P.g_qa, b.q_a_norm, P.g_qr, (int) QLR, (int) T, eps, cs);
                    dense(b.q_b, P.g_qr, P.g_qb, T);
                    dense(b.idx_q_b, P.g_qr, P.g_iq, T);
                    k::glm5_idx_append(P.g_ik, P.g_ig, b.idx_k_norm, b.idx_k_norm_b,
                                       s.norm_eps > 0 ? (float) s.norm_eps : 1e-6f, P.g_steps, T,
                                       g_idx_cache_[(size_t) d], cs);
                    k::glm5_idx_pool(g_idx_cache_[(size_t) d], b.idx_ape, (int) s.idx_kpool, P.g_steps, T,
                                     g_pooled_[(size_t) d], cs);
                    k::glm5_idx_scores(g_pooled_[(size_t) d], P.g_iq, P.g_iw, (int) IH, P.g_steps, T, g_max_blocks_,
                                       P.g_scores, cs);
                    k::qsa_block_topk(P.g_scores, P.g_steps, T, g_max_blocks_, g_cap_, sel, P.g_ids, cs);
                    k::glm5_latent_store(P.g_kv, P.g_steps, T, KVR, g_latent_[(size_t) d], cs);
                    // each head's q through its wk_b into the latent, the attention, each head's wv_b
                    const size_t kb = k::native_mmvq_weight_bytes(file_type(b.k_b.type), (int) QK, (int) KVR);
                    const size_t vb = k::native_mmvq_weight_bytes(file_type(b.v_b.type), (int) KVR, (int) VD);
                    k::glm5_transpose_abd(P.g_qb, P.g_qbT, T, H, QK, cs);
                    for (int64_t h = 0; h < H; ++h)
                        prod(b.k_b.type, (const uint8_t*) b.k_b.data + h * kb, QK, KVR, P.g_qbT + h * T * QK, QK,
                             P.g_absT + h * T * KVR, T, KVR);
                    k::glm5_transpose_abd(P.g_absT, P.g_abs, H, T, KVR, cs);
                    // --kv-ram: the layer's cells up to the chunk's last into VRAM, read there
                    const uint16_t* lat = g_latent_[(size_t) d];
                    if (g_lat_stage_ != nullptr) {
                        if (!strata::gpu::copy_async(g_lat_stage_, lat, (size_t) ((p0 + T) * KVR) * 2, cs)) {
                            err = std::string("plain prompt: --kv-ram's cells: ") + strata::gpu::last_error();
                            return false;
                        }
                        lat = g_lat_stage_;
                    }
                    k::glm5_mla_attn(P.g_abs, lat, P.g_ids, g_cap_, P.g_steps, (int) H,
                                     1.0f / std::sqrt((float) QK), T, P.g_lat, cs,
                                     s.idx_kpool_select_tail ? (int) s.idx_kpool : 0);
                    k::glm5_transpose_abd(P.g_lat, P.g_latT, T, H, KVR, cs);
                    for (int64_t h = 0; h < H; ++h)
                        prod(b.v_b.type, (const uint8_t*) b.v_b.data + h * vb, KVR, VD, P.g_latT + h * T * KVR, KVR,
                             P.g_voT + h * T * VD, T, VD);
                    k::glm5_transpose_abd(P.g_voT, P.g_vo, H, T, VD, cs);
                    dense(b.o, P.g_vo, P.mix, T);
                    mark(1);
                }
                k::glm5_hc_post(P.mix, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs);

                // the FFN
                hc_read(b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, b.ffn_norm);
                const float* ffn_out = P.mix;
                if (l < s.n_dense_lead) {
                    swiglu(b, P.mix);
                    mark(4);
                } else {
                    k::f32_gemm_rows(P.hn, N, b.router, P.rlog, E, N, E, T, cs);
                    const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
                    k::glm5_router_sigmoid(P.rlog, b.router_b, (int) T, (int) E, (int) K, s.expert_weights_norm, scale,
                                           P.ids, P.w, cs);
                    if (P.f16) kernels::f32_to_f16_bulk(P.hn, P.x16, T * N, cs);   // the FP16 path's routed rows
                    const int64_t ml = s.moe_index(l);
                    if (!prompt_routed_tuned(ml, T, cpu_share, routes.empty() ? nullptr : routes.data() + ml * P.T * K,
                                       [&](int stage) { mark(stage == 1 ? 2 : 3); }, rst, err))
                        return false;
                    swiglu(b, P.shared);   // the shared expert (no gate)
                    k::native_moe_combine_multi(P.parts, P.w, P.shared, P.moe_out, N, K, (int) T, cs);
                    ffn_out = P.moe_out;
                    mark(4);
                }
                k::glm5_hc_post(ffn_out, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs);
                tap_rows(l, P.g_hs, kHC, T, p0, false, cs);   // DFlash2's prompt taps: the four streams' mean
            }
            if (!prompt_taps_deliver(ptap_dev_, T, p0, cs, err)) return false;
            // the MTP layer after the trunk, its caches a cell ahead: row t pairs the trunk's normed row t with the
            // token after it (the prompt's last row with a stand-in, rewritten by spec_start once the token is picked).
            // Only its caches are wanted here, so of its block only the latent and the indexer's keys are made.
            if (!last() && !strata::gpu::copy_async(hand, P.g_hs, (size_t) (T * kHC * N) * 4, cs)) {
                err = std::string("plain prompt: the rows to the next GPU: ") + strata::gpu::last_error();
                return false;
            }
            const bool mtp_chunk = has_mtp_ && p0 + T < opt_.max_cells;
            const int64_t rows = mtp_chunk ? T : 1;   // the trunk's normed rows: every row for the MTP layer
            if (last() && (mtp_chunk || c0 + T >= n)) {
                k::glm5_hc_mean(P.g_hs + (T - rows) * kHC * N, P.x, N, rows, cs);
                k::native_qsa_rms_norm_weighted(P.x, out_norm_, P.hn, (int) N, (int) rows, eps, cs);
            }
            if (last() && c0 + T >= n) {   // the last token's logits, by the decode's head
                const float* last = P.hn + (rows - 1) * N;
                k::native_quantize_q8_1(last, xq_, head_.n_in, 1, cs);
                k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, 1, cs);
                pick(logits_, 1, p0 + T - 1, m_out_, cs);
                if (mtp_chunk && !strata::gpu::copy_async(mtp_h_, last, (size_t) N * 4, cs)) {
                    err = std::string("plain prompt: the MTP layer's input: ") + strata::gpu::last_error();
                    return false;
                }
            }
            if (mtp_chunk) {
                const GBlock& b = gmtp_;
                const int64_t d = b.idx;
                k::native_qsa_rms_norm_weighted(P.hn, mtp_hnorm_, P.mix, (int) N, (int) T, eps, cs);
                if (!strata::gpu::stream_sync(cs)) {   // the host's staging is rewritten below
                    err = std::string("plain prompt: ") + strata::gpu::last_error();
                    return false;
                }
                for (int64_t t = 0; t < T; ++t) {
                    const int32_t tok = c0 + t + 1 < n ? tokens[c0 + t + 1] : tokens[n - 1];
                    if (!embed(tok, p0 + 1 + t, P.emb + t * N)) {
                        err = "plain prompt: a token is out of the vocabulary";
                        return false;
                    }
                    k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + 1 + t, qs);
                }
                if (!strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) ||
                    !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs)) {
                    err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                    return false;
                }
                k::glm5_idx_steps(P.steps, T, (int) s.idx_kpool, (int) s.idx_top_k, P.g_steps, cs);
                k::native_qsa_rms_norm_weighted(P.x, mtp_enorm_, P.hn, (int) N, (int) T, eps, cs);
                k::concat_rows(P.g_qbT, P.hn, P.mix, N, T, cs);
                dense(mtp_eh_, P.g_qbT, P.x, T);
                k::native_qsa_rms_norm_weighted(P.x, b.attn_norm, P.hn, (int) N, (int) T, eps, cs);
                dense(b.kv_a, P.hn, P.g_absT, T);
                dense(b.idx_k, P.hn, P.g_ik, T);
                dense(b.idx_gate, P.hn, P.g_ig, T);
                k::native_qsa_rms_norm_weighted(P.g_absT, b.kv_a_norm, P.g_kv, (int) KVR, (int) T, eps, cs);
                k::glm5_latent_store(P.g_kv, P.g_steps, T, KVR, g_latent_[(size_t) d], cs);
                k::glm5_idx_append(P.g_ik, P.g_ig, b.idx_k_norm, b.idx_k_norm_b,
                                   s.norm_eps > 0 ? (float) s.norm_eps : 1e-6f, P.g_steps, T, g_idx_cache_[(size_t) d],
                                   cs);
                k::glm5_idx_pool(g_idx_cache_[(size_t) d], b.idx_ape, (int) s.idx_kpool, P.g_steps, T,
                                 g_pooled_[(size_t) d], cs);
            }
            mark(10);
            if (!routes.empty()) {   // the prompt's routings, as plain_prompt.cpp counts them
                for (int64_t ml : moe_local_)   // this stage's layers' routes (seed_usage at the prompt's end)
                    for (int64_t i = 0; i < T * K; ++i) ++prompt_routed_[(size_t) (ml * E + routes[(size_t) (ml * P.T * K + i)])];
            }
            if (prompt_progress_ && c_begin == 0 && c_end >= n) prompt_progress_(c0 + T, n);   // as plain_prompt.cpp
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: ") + e.what();
        return false;
    }
    mark(11);
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: ") + strata::gpu::last_error();
        return false;
    }
    if (c_end < n) return true;   // a layer split's chunk: the stage's part of the prompt goes on with the next
    windows_ += (n + opt_.max_t - 1) / opt_.max_t;
    if (profile)
        std::fprintf(stderr, "xestrata plain: GLM prompt of %lld tokens: KDA %.1f ms (projections %.1f, convolutions and "
                             "gates %.1f, recurrence %.1f, output %.1f), DSA %.1f, grouping %.1f, experts %.1f, shared and "
                             "dense FFN %.1f, staging %.1f, head %.1f, the rest %.1f; experts not in VRAM %lld, on the CPU %lld "
                             "(%lld rows)\n",
                     (long long) n, stage_ms[5] + stage_ms[6] + stage_ms[7] + stage_ms[8], stage_ms[5], stage_ms[6],
                     stage_ms[7], stage_ms[8], stage_ms[1], stage_ms[2], stage_ms[3], stage_ms[4], stage_ms[9], stage_ms[10], stage_ms[11],
                     (long long) rst.miss_n, (long long) rst.cpu_n, (long long) rst.cpu_rows);
    mtp_h_set_ = has_mtp_ && pos + n < opt_.max_cells;   // the trunk's last row is in mtp_h_ for spec_start
    if (!last()) return true;
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
