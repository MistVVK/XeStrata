// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/plain_glm5.cpp - GLM-5.3-Flash (glm5-next) in PlainModel (plain_model.hpp).
//
// The block follows llama.cpp's glm5-next graph (src/models/glm5-next.cpp at f9a95ae; MIT License, Copyright (c)
// 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE) step for step:
//
//   streams:  four copies of the embedding (hc_init); before the mixer and before the FFN, hc_pre reads them into one
//             row (the mixes hc_fn . flat, RMS-normalized; pre / post / the Sinkhorn comb), after each hc_post writes
//             the block's row back into the four (glm5.hpp); the head reads their mean
//   KDA:      q, k, v = attn_q / attn_k / attn_v (h), each through its short conv and SiLU, q and k l2-normed per head;
//             decay g = lb sigmoid(-(f_b(f_a(h)) + dt) ssm_a) per channel, beta = sigmoid(ssm_beta(h)); the delta rule;
//             y = rms_norm(o) ssm_norm sigmoid(g_b(g_a(h))); out = attn_output(y)
//   DSA:      qr = rms_norm(q_a(h)); q = q_b(qr) per head, through wk_b into the latent (absorbed); kv = rms_norm(kv_a(h)),
//             the latent the cell keeps; the k-pool indexer (key LayerNorm(indexer.attn_k(h)), pool gate, pools of 4,
//             32 heads of indexer.attn_q_b(qr) weighted by indexer.proj(h)) selects the top 512 complete pools; softmax
//             over their cells and the token's unfinished pool up to it (kpool_select_tail); wv_b per head;
//             out = attn_output(.)
//   FFN:      the leading layers a dense SwiGLU; then the sigmoid router (bias for the choice, weights normalized,
//             x expert_weights_scale), the routed experts, the shared expert (no gate); every SwiGLU clamped
//
// The prompt runs in chunks (src/prefill/plain_prompt_glm5.cpp); the MTP layer does not draft yet.
#include "strata/core/plain_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/swiglu_clamp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace strata::core {
namespace {

namespace k = strata::kernels;

constexpr int kHC = 4, kMix = (2 + kHC) * kHC;
constexpr int kBF16 = 30;
constexpr int kQ6K = 14, kQ80 = 8;
constexpr float kL2Eps = 1e-6f;   // llama.cpp's FLA l2 norm

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

/// A projection's GGUF type before the engine's row order (the byte count of its rows)
int file_type(int t) { return t == k::kNativeQ6KRows ? kQ6K : t == k::kNativeQ8Rows ? kQ80 : t; }

}  // namespace

bool PlainModel::load_gblock(const GgufModel& m, int64_t l, GBlock& b, std::string& err, bool mtp) {
    const ModelSpec& s = spec_;
    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    const int64_t H = s.n_head, S = s.kda_head_dim, C = H * S;
    const int64_t QLR = s.q_lora_rank, KVR = s.kv_lora_rank, QK = s.key_length_mla, VD = s.value_length_mla;
    const int64_t IH = s.idx_n_head, ID = s.idx_key_length;
    const std::string p = "blk." + std::to_string(l) + ".";
    auto bad = [&](const char* what) {
        err = "plain model: block " + std::to_string(l) + "'s " + what + " have other shapes";
        return false;
    };
    if (!upload_f32(m, p + "attn_norm.weight", N, b.attn_norm, err) ||
        !upload_f32(m, p + "ffn_norm.weight", N, b.ffn_norm, err))
        return false;
    if (!mtp && (!upload(m, p + "hc_attn_fn.weight", b.hc_attn_fn, err) ||
        !upload_f32(m, p + "hc_attn_base.weight", kMix, b.hc_attn_base, err) ||
        !upload_f32(m, p + "hc_attn_scale.weight", 3, b.hc_attn_scale, err) ||
        !upload(m, p + "hc_ffn_fn.weight", b.hc_ffn_fn, err) ||
        !upload_f32(m, p + "hc_ffn_base.weight", kMix, b.hc_ffn_base, err) ||
        !upload_f32(m, p + "hc_ffn_scale.weight", 3, b.hc_ffn_scale, err)))
        return false;
    for (const W* w : {&b.hc_attn_fn, &b.hc_ffn_fn})
        if (!mtp && ((uint64_t) w->n_in != kHC * N || w->n_out != kMix)) return bad("mHC mixes");
    if (b.kda) {
        if (!upload(m, p + "attn_q.weight", b.q, err) || !upload(m, p + "attn_k.weight", b.k, err) ||
            !upload(m, p + "attn_v.weight", b.v, err) || !upload(m, p + "ssm_f_a.weight", b.f_a, err) ||
            !upload(m, p + "ssm_f_b.weight", b.f_b, err) || !upload(m, p + "ssm_beta.weight", b.beta, err) ||
            !upload(m, p + "ssm_g_a.weight", b.g_a, err) || !upload(m, p + "ssm_g_b.weight", b.g_b, err) ||
            !upload(m, p + "attn_output.weight", b.o, err) ||
            !upload_f32(m, p + "ssm_conv1d_q.weight", (uint64_t) (C * s.ssm_d_conv), b.conv_q, err) ||
            !upload_f32(m, p + "ssm_conv1d_k.weight", (uint64_t) (C * s.ssm_d_conv), b.conv_k, err) ||
            !upload_f32(m, p + "ssm_conv1d_v.weight", (uint64_t) (C * s.ssm_d_conv), b.conv_v, err) ||
            !upload_f32(m, p + "ssm_dt.bias", (uint64_t) C, b.dt, err) ||
            !upload_f32(m, p + "ssm_a", (uint64_t) H, b.a, err) ||
            !upload_f32(m, p + "ssm_norm.weight", (uint64_t) S, b.o_norm, err))
            return false;
        for (const W* w : {&b.q, &b.k, &b.v})
            if ((uint64_t) w->n_in != N || w->n_out != C) return bad("KDA projections");
        if (b.f_a.n_out != S || b.g_a.n_out != S || b.f_b.n_in != S || b.g_b.n_in != S || b.f_b.n_out != C ||
            b.g_b.n_out != C || b.beta.n_out != H || b.o.n_in != C || (uint64_t) b.o.n_out != N)
            return bad("KDA gates");
    } else {
        if (!upload(m, p + "attn_q_a.weight", b.q_a, err) || !upload(m, p + "attn_q_b.weight", b.q_b, err) ||
            !upload(m, p + "attn_kv_a_mqa.weight", b.kv_a, err) || !upload(m, p + "attn_k_b.weight", b.k_b, err) ||
            !upload(m, p + "attn_v_b.weight", b.v_b, err) || !upload(m, p + "attn_output.weight", b.o, err) ||
            !upload_f32(m, p + "attn_q_a_norm.weight", (uint64_t) QLR, b.q_a_norm, err) ||
            !upload_f32(m, p + "attn_kv_a_norm.weight", (uint64_t) KVR, b.kv_a_norm, err) ||
            !upload(m, p + "indexer.attn_q_b.weight", b.idx_q_b, err) ||
            !upload(m, p + "indexer.attn_k.weight", b.idx_k, err) ||
            !upload(m, p + "indexer_compressor_gate.weight", b.idx_gate, err) ||
            !upload_f32(m, p + "indexer.k_norm.weight", (uint64_t) ID, b.idx_k_norm, err) ||
            !upload_f32(m, p + "indexer.k_norm.bias", (uint64_t) ID, b.idx_k_norm_b, err) ||
            !upload_f32(m, p + "indexer.proj.weight", N * (uint64_t) IH, b.idx_proj, err) ||
            !upload_f32(m, p + "indexer_compressor_ape.weight", (uint64_t) (ID * s.idx_kpool), b.idx_ape, err))
            return false;
        if ((uint64_t) b.q_a.n_in != N || b.q_a.n_out != QLR || b.q_b.n_in != QLR || b.q_b.n_out != H * QK ||
            (uint64_t) b.kv_a.n_in != N || b.kv_a.n_out != KVR || b.k_b.n_in != QK || b.k_b.n_out != KVR * H ||
            b.v_b.n_in != KVR || b.v_b.n_out != VD * H || b.o.n_in != H * VD || (uint64_t) b.o.n_out != N)
            return bad("MLA projections");
        if (b.idx_q_b.n_in != QLR || b.idx_q_b.n_out != IH * ID || (uint64_t) b.idx_k.n_in != N || b.idx_k.n_out != ID ||
            (uint64_t) b.idx_gate.n_in != N || b.idx_gate.n_out != ID)
            return bad("indexer projections");
    }
    // the FFN: a leading layer's dense SwiGLU, else the router and the shared expert; the clamp the file gives
    const bool dense = l < s.n_dense_lead;
    const int64_t FW = dense ? s.n_ff : s.n_ff_shexp;
    if (!upload(m, p + (dense ? "ffn_gate.weight" : "ffn_gate_shexp.weight"), b.ffn_gate, err) ||
        !upload(m, p + (dense ? "ffn_up.weight" : "ffn_up_shexp.weight"), b.ffn_up, err) ||
        !upload(m, p + (dense ? "ffn_down.weight" : "ffn_down_shexp.weight"), b.ffn_down, err))
        return false;
    if ((uint64_t) b.ffn_gate.n_in != N || b.ffn_gate.n_out != FW || (uint64_t) b.ffn_up.n_in != N ||
        b.ffn_up.n_out != FW || b.ffn_down.n_in != FW || (uint64_t) b.ffn_down.n_out != N)
        return bad(dense ? "dense FFN" : "shared expert");
    if (!dense && (!upload_f32(m, p + "ffn_gate_inp.weight", N * E, b.router, err) ||
                   !upload_f32(m, p + "exp_probs_b.bias", E, b.router_b, err)))
        return false;
    b.clamp = (size_t) l < s.swiglu_clamp_shexp.size() ? (float) s.swiglu_clamp_shexp[(size_t) l] : 0.0f;
    // the native matrix-vector kernels only (no BF16 GLM file is taken yet)
    for (const W* w : {&b.hc_attn_fn, &b.hc_ffn_fn, &b.q, &b.k, &b.v, &b.f_a, &b.f_b, &b.beta, &b.g_a, &b.g_b, &b.q_a,
                       &b.q_b, &b.kv_a, &b.k_b, &b.v_b, &b.idx_q_b, &b.idx_k, &b.idx_gate, &b.o, &b.ffn_gate, &b.ffn_up,
                       &b.ffn_down})
        if (w->data != nullptr && w->type == kBF16) {
            err = "plain model: a BF16 GLM-5.3-Flash file is not run yet";
            return false;
        }
    return true;
}

bool PlainModel::load_glm5(std::string& err) {
    const ModelSpec& s = spec_;
    glm_ = true;
    // its MLA latent and indexer caches have no such kernels yet
    if (opt_.kv_int8 || opt_.kv_q4 != 0 || opt_.kv_tcq != 0) {
        err = "plain model: --kv int8, q4_0, k8v4, tcq4 and tcq5 are not available for GLM-5.3-Flash yet";
        return false;
    }
    // the shapes the kernels are written for (glm5.hpp: four streams, KDA heads of 128, the MLA's 512-wide latent and
    // 256-wide heads, the indexer's 128-wide keys in pools of 4, at most 32 routed experts)
    if (s.hc != 4 || s.kda_head_dim != 128 || s.ssm_d_conv != 4 || s.kv_lora_rank != 512 || s.key_length_mla != 256 ||
        s.value_length_mla != 256 || s.idx_key_length != 128 || s.idx_kpool != 4 ||
        !(s.kda_gate_lower_bound < 0) || s.n_expert_used > 15 || s.gating != ExpertGating::Sigmoid ||
        s.n_ff_shexp <= 0 || s.n_embd % 256 != 0 || s.q_lora_rank % 32 != 0) {
        err = "plain model: a GLM-5.3-Flash shape this engine's kernels are not built for";
        return false;
    }
    // the MTP layer (blk.<n_layer>, nextn.*): drafts for speculative windows, as the Qwen-family files'; the last
    // stage's
    model_mtp_ = opt_.spec >= 2 && s.n_nextn == 1 &&
                 model_->find("blk." + std::to_string(s.n_layer) + ".nextn.eh_proj.weight") != nullptr;
    has_mtp_ = model_mtp_ && last();
    // its decode windows keep their graphs, as the other MoE models' (load_qwen): a window answers the CPU's rings
    // between its graph segments, which the straight form does not.  Timed against it on the one-row windows, a
    // one-row window ran straight, its experts' ids were garbage and the GPU faulted on them (B70, UD-IQ3_XXS:
    // --draft off stopped at its first token, the MTP's windows at a one-row window after 63 or 127 tokens)
    step_form_.fix(0);
    opt_.max_cells = (opt_.max_cells + s.idx_kpool - 1) / s.idx_kpool * s.idx_kpool;
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    // the routed experts' SwiGLU limit: one for every MoE layer (a process setting, before any graph)
    {
        float lim = 0.0f;
        bool one = true;
        for (int64_t l = s.n_dense_lead; l < s.n_layer; ++l) {
            const float v = (size_t) l < s.swiglu_clamp_exp.size() ? (float) s.swiglu_clamp_exp[(size_t) l] : 0.0f;
            if (l == s.n_dense_lead) lim = v;
            else one = one && v == lim;
        }
        if (!one) { err = "plain model: the experts' SwiGLU limits differ between layers"; return false; }
        k::expert_swiglu_set_limit(lim);
    }

    const GgufModel& m = *model_;
    if (!load_embedding(err)) return false;
    if (last()) {   // the head: the last stage's
        if (!upload(m, "output.weight", head_, err) ||
            !upload_f32(m, "output_norm.weight", (uint64_t) s.n_embd, out_norm_, err))
            return false;
        if (head_.n_in != s.n_embd || head_.n_out != n_vocab_ || head_.type == kBF16) {
            err = "plain model: output.weight's shape or type";
            return false;
        }
    }
    gblocks_.resize((size_t) s.n_layer);
    int64_t n_kda = 0, n_dsa = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {   // the indices count every block; this stage loads its own
        GBlock& b = gblocks_[(size_t) l];
        b.kda = s.recurrent[(size_t) l];
        b.idx = b.kda ? n_kda++ : n_dsa++;
        if (l >= lb_ && l < le_ && !load_gblock(m, l, b, err)) return false;
    }
    if (n_dsa == 0) { err = "plain model: the model has no DSA block"; return false; }
    if (!setup_taps(err)) return false;   // DFlash2's (GLM-5.3-Flash's drafter reads the four streams' mean)
    if (has_mtp_) {   // a DSA block of its own caches, after the trunk's
        const std::string p = "blk." + std::to_string(s.n_layer) + ".nextn.";
        mtp_allocs_[0] = allocs_.size();   // the layer's own weights (release_mtp)
        gmtp_.kda = false;
        gmtp_.idx = n_dsa++;
        if (!load_gblock(m, s.n_layer, gmtp_, err, true) || !upload(m, p + "eh_proj.weight", mtp_eh_, err) ||
            !upload_f32(m, p + "enorm.weight", (uint64_t) s.n_embd, mtp_enorm_, err) ||
            !upload_f32(m, p + "hnorm.weight", (uint64_t) s.n_embd, mtp_hnorm_, err))
            return false;
        if (m.find(p + "shared_head_norm.weight") == nullptr) mtp_head_norm_ = out_norm_;
        else if (!upload_f32(m, p + "shared_head_norm.weight", (uint64_t) s.n_embd, mtp_head_norm_, err)) return false;
        if (mtp_eh_.n_in != 2 * s.n_embd || mtp_eh_.n_out != s.n_embd || mtp_eh_.type == kBF16) {
            err = "plain model: the MTP layer's eh_proj has another shape or type";
            return false;
        }
        mtp_allocs_[1] = allocs_.size();
    }

    if (!tune_uploaded(err)) return false;
    // per-sequence state: per KDA block its recurrence and its three convolutions' last 3 inputs; per DSA block the
    // indexer's key | gate rows, its pooled keys and the MLA latent (FP16), one row a cell
    const int64_t H = s.n_head, S = s.kda_head_dim, C = H * S, KVR = s.kv_lora_rank, ID = s.idx_key_length;
    const int64_t cells = opt_.max_cells;
    state_bytes_ = (size_t) (H * S * S) * 4;
    conv_bytes_ = (size_t) (3 * C * (s.ssm_d_conv - 1)) * 4;
    g_max_blocks_ = cells / s.idx_kpool + 1;
    auto mine = [&](bool kda, int64_t idx) {   // a KDA or DSA block of this stage's (the MTP layer's on the last)
        if (!kda && has_mtp_ && idx == gmtp_.idx) return true;
        for (int64_t l = lb_; l < le_; ++l)
            if (gblocks_[(size_t) l].kda == kda && gblocks_[(size_t) l].idx == idx) return true;
        return false;
    };
    gdn_state_.assign((size_t) n_kda, nullptr);
    gdn_conv_.assign((size_t) n_kda, nullptr);
    for (int64_t i = 0; i < n_kda; ++i) {
        if (!mine(true, i)) continue;
        float *st = nullptr, *cv = nullptr;
        if (!strata::gpu::alloc_device(&st, state_bytes_) || !strata::gpu::alloc_device(&cv, conv_bytes_)) {
            err = "plain model: no VRAM for the KDA states";
            return false;
        }
        allocs_.push_back(st);
        allocs_.push_back(cv);
        gdn_state_[(size_t) i] = st;
        gdn_conv_[(size_t) i] = cv;
    }
    // the KDA states after a window's rows but its last, for going back to the last accepted draft (every stage's)
    if (model_mtp_ || opt_.draft_windows > 0) {   // (--draft dflash2: its windows' copies)
        if (model_mtp_) opt_.spec = std::min({opt_.spec, opt_.max_t, k::kGlm5MaxSnaps + 1});
        n_mtp_ = model_mtp_ ? opt_.spec - 1 : 0;
        // those past the MTP layer's are the drafter's (load_qwen's rule: set_window_drafts may free them)
        n_snap_load_ = n_mtp_;
        n_snap_ = std::max(n_mtp_, std::min({opt_.draft_windows, opt_.max_t - 1, k::kGlm5MaxSnaps}));
        gdn_snap_.assign(gdn_state_.size(), {});
        conv_snap_.assign(gdn_state_.size(), {});
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            for (int t = 0; t < n_snap_ && gdn_state_[g] != nullptr; ++t) {
                float *st = nullptr, *cv = nullptr;
                if (!strata::gpu::alloc_device(&st, state_bytes_) || !strata::gpu::alloc_device(&cv, conv_bytes_)) {
                    strata::gpu::free(st);
                    err = "plain model: no VRAM for the KDA states' copies (--spec)";
                    return false;
                }
                if (t < n_snap_load_) {
                    allocs_.push_back(st);
                    allocs_.push_back(cv);
                }
                gdn_snap_[g].push_back(st);
                conv_snap_[g].push_back(cv);
            }
    }
    g_idx_cache_.assign((size_t) n_dsa, nullptr);
    g_pooled_.assign((size_t) n_dsa, nullptr);
    g_latent_.assign((size_t) n_dsa, nullptr);
    for (int64_t i = 0; i < n_dsa; ++i) {
        if (!mine(false, i)) continue;
        float *ic = nullptr, *pl = nullptr;
        uint16_t* lat = nullptr;
        void* lat_host = nullptr;   // --kv-ram: the cells in host memory the GPU reads (lat its device address)
        if (!strata::gpu::alloc_device(&ic, (size_t) (cells * 2 * ID) * 4) ||
            !strata::gpu::alloc_device(&pl, (size_t) (g_max_blocks_ * ID) * 4) ||
            !(opt_.kv_ram ? mapped((size_t) (cells * KVR) * 2, &lat_host, (void**) &lat)
                          : strata::gpu::alloc_device(&lat, (size_t) (cells * KVR) * 2))) {
            err = "plain model: no " + std::string(opt_.kv_ram ? "RAM" : "VRAM") + " for the DSA caches (" +
                  std::to_string(cells) + " cells)";
            return false;
        }
        allocs_.push_back(ic);
        allocs_.push_back(pl);
        allocs_.push_back(opt_.kv_ram ? lat_host : (void*) lat);
        g_idx_cache_[(size_t) i] = ic;
        g_pooled_[(size_t) i] = pl;
        g_latent_[(size_t) i] = lat;
    }
    {
        k::QsaShapes qs;
        qs.idx_block = s.idx_kpool;
        qs.idx_top_k = s.idx_top_k;
        g_cap_ = k::qsa_selection_width(k::kTopkMaxCells, qs);
    }
    if (opt_.kv_ram && n_dsa > 0) {   // the window's gathered rows and the prompt's layer of cells, in VRAM
        g_gather_stride_ = g_cap_ + (s.idx_kpool_select_tail && s.idx_kpool > 1 ? s.idx_kpool - 1 : 0);
        if (!strata::gpu::alloc_device(&g_lat_gather_, (size_t) (opt_.max_t * g_gather_stride_ * KVR) * 2) ||
            !strata::gpu::alloc_device(&g_lat_stage_, (size_t) (cells * KVR) * 2)) {
            err = "plain model: no VRAM for --kv-ram's buffers";
            return false;
        }
        allocs_.push_back(g_lat_gather_);
        allocs_.push_back(g_lat_stage_);
        std::fprintf(stderr, "xestrata plain: --kv-ram: the DSA latent cells in RAM (%.1f MiB), %.1f MiB of VRAM for them\n",
                     (double) (n_dsa * cells * KVR * 2) / 1048576.0,
                     (double) ((opt_.max_t * g_gather_stride_ + cells) * KVR * 2) / 1048576.0);
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    const uint64_t K = (uint64_t) s.n_expert_used, NH = (uint64_t) H;
    const uint64_t QLR = (uint64_t) s.q_lora_rank, QK = (uint64_t) s.key_length_mla, VD = (uint64_t) s.value_length_mla;
    const uint64_t IH = (uint64_t) s.idx_n_head, FW = (uint64_t) std::max<int64_t>(s.n_ff, s.n_ff_shexp);
    hand_w_ = kHC * s.n_embd;   // a stage hands the four streams on
    if (!mapped(T * N * 4, (void**) &h_emb_, (void**) &m_emb_) ||
        !mapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !mapped(T * (NH + 1) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) || !mapped(16, (void**) &h_prob_, (void**) &m_prob_) ||
        (!first() && !mapped(T * kHC * N * 4, (void**) &h_in_, (void**) &m_in_)) ||
        (!last() && !mapped(T * kHC * N * 4, (void**) &h_hand_, (void**) &m_hand_))) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    // the q8_1 rows: a token's 4N stream values, the widest input, or the MLA's per-head rows of every token
    const size_t xq = std::max({k::native_q8_1_bytes((int) (kHC * N), (int) T), k::native_q8_1_bytes((int) FW, (int) T),
                                k::native_q8_1_bytes((int) (NH * VD), (int) T),
                                k::native_q8_1_rows_bytes((int) KVR, (int64_t) (NH * T)),
                                k::native_q8_1_rows_bytes((int) QK, (int64_t) (NH * T))});
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    const uint64_t UC = T * (uint64_t) C, L4 = T * NH * (uint64_t) KVR;
    if (!dev(&step_, T * k::kStepCount * 4) || !dev(&g_steps_, T * k::kStepCount * 4) || !dev(&x_, T * N * 4) ||
        !dev(&hn_, T * N * 4) || !dev(&mix_, T * N * 4) || !dev(&g_xq_, xq) || !dev(&g_hs_, T * kHC * N * 4) ||
        !dev(&g_mixes_, T * kMix * 4) || !dev(&g_post_, T * kHC * 4) || !dev(&g_comb_, T * kHC * kHC * 4) ||
        !dev(&g_y_, T * N * 4) ||
        // KDA
        !dev(&g_q_, UC * 4) || !dev(&g_k_, UC * 4) || !dev(&g_v_, UC * 4) || !dev(&g_qc_, UC * 4) ||
        !dev(&g_kc_, UC * 4) || !dev(&g_vc_, UC * 4) || !dev(&g_fa_, T * (uint64_t) S * 4) || !dev(&g_f_, UC * 4) ||
        !dev(&g_gate_, UC * 4) || !dev(&g_beta_, T * NH * 4) || !dev(&g_ga_, T * (uint64_t) S * 4) ||
        !dev(&g_g2_, UC * 4) || !dev(&g_o_, UC * 4) || !dev(&g_oy_, UC * 4) ||
        // DSA
        !dev(&g_qa_, T * QLR * 4) || !dev(&g_qr_, T * QLR * 4) || !dev(&g_qb_, T * NH * QK * 4) ||
        !dev(&g_qbT_, T * NH * QK * 4) || !dev(&g_absT_, L4 * 4) || !dev(&g_abs_, L4 * 4) ||
        !dev(&g_kv_, T * (uint64_t) KVR * 4) || !dev(&g_lat_, L4 * 4) || !dev(&g_latT_, L4 * 4) ||
        !dev(&g_voT_, T * NH * VD * 4) || !dev(&g_vo_, T * NH * VD * 4) ||
        !dev(&g_iq_, T * IH * (uint64_t) ID * 4) || !dev(&g_ik_, T * (uint64_t) ID * 4) ||
        !dev(&g_ig_, T * (uint64_t) ID * 4) || !dev(&g_iw_, T * IH * 4) ||
        !dev(&g_scores_, T * (uint64_t) g_max_blocks_ * 4) || !dev(&g_ids_, T * (uint64_t) g_cap_ * 4) ||
        // the FFN
        !dev(&g_ffg_, T * FW * 4) || !dev(&g_ffu_, T * FW * 4) || !dev(&rlog_, T * E * 4) || !dev(&ids_, T * K * 4) ||
        !dev(&w_, T * K * 4) || !dev(&shared_, T * N * 4) || !dev(&moe_out_, T * N * 4) ||
        !dev(&logits_, T * (uint64_t) n_vocab_ * 4) || (has_mtp_ && !dev(&mtp_h_, T * N * 4))) {
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    xq_ = g_xq_;   // the head's q8_1 rows (head_logits)
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    setup_int2();   // before the expert tier takes what VRAM is left
    if (!open_experts(err)) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    mexec_.assign((size_t) opt_.max_t + 1, nullptr);
    msegs_.assign((size_t) opt_.max_t + 1, {});
    if (root_ == this)
        std::fprintf(stderr, "xestrata plain: GLM-5.3-Flash, %lld KDA and %lld DSA blocks, %lld cells; the prompt in chunks "
                             "of %lld%s\n", (long long) n_kda, (long long) (has_mtp_ ? n_dsa - 1 : n_dsa),
                     (long long) cells, (long long) opt_.prompt_chunk, model_mtp_ ? "; the MTP layer drafts" : "");
    return true;
}

bool PlainModel::reset_glm5(std::string& err) {
    bool ok = true;
    for (float* p : gdn_state_) ok = ok && (p == nullptr || strata::gpu::memset(p, 0, state_bytes_));
    for (float* p : gdn_conv_) ok = ok && (p == nullptr || strata::gpu::memset(p, 0, conv_bytes_));
    cur_ = -1;
    drafts_.clear();
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

void PlainModel::gproj(const W& w, const float* x, float* y, int T, bool quantized, strata::gpu::Stream cs) {
    if (!quantized) k::native_quantize_q8_1(x, g_xq_, w.n_in, T, cs);
    k::native_mmvq(w.type, w.data, g_xq_, y, w.n_in, w.n_out, T, cs);
}

bool PlainModel::record_gblock(const GBlock& b, int64_t l, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                               strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, S = s.kda_head_dim, C = H * S;
    const float eps = (float) s.rms_eps, hc_eps = (float) s.hc_eps;
    const int iters = (int) s.hc_sinkhorn_iters;
    // hc_pre: the mixes of the four streams as they are (their RMS is applied in glm5_hc_pre), then the block's row
    auto hc_read = [&](const W& fn, const float* scale, const float* base, const float* norm) {
        k::native_quantize_q8_1(g_hs_, g_xq_, (int) (kHC * N), T, cs);
        k::native_mmvq(fn.type, fn.data, g_xq_, g_mixes_, fn.n_in, fn.n_out, T, cs);
        k::glm5_hc_pre(g_hs_, g_mixes_, scale, base, eps, hc_eps, iters, g_y_, g_post_, g_comb_, N, T, cs);
        k::native_qsa_rms_norm_weighted_q8_1(g_y_, norm, hn_, g_xq_, (int) N, T, eps, cs);   // hn_ and its q8_1
    };

    // ---- the mixer
    hc_read(b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, b.attn_norm);
    if (b.kda) {
        float* state = gdn_state_[(size_t) b.idx];
        float* conv = gdn_conv_[(size_t) b.idx];
        const int64_t hist = C * (s.ssm_d_conv - 1);
        const std::pair<const W*, float*> from_row[] = {{&b.q, g_q_},   {&b.k, g_k_},   {&b.v, g_v_},
                                                        {&b.f_a, g_fa_}, {&b.g_a, g_ga_}, {&b.beta, g_beta_}};
        for (const auto& [w, y] : from_row) gproj(*w, hn_, y, T, true, cs);
        gproj(b.f_b, g_fa_, g_f_, T, false, cs);   // the low-rank gates' second halves (their inputs quantized in turn)
        gproj(b.g_b, g_ga_, g_g2_, T, false, cs);
        // the states after rows 0 .. n_snaps - 1, for spec_step: the convolutions then go row by row, their history
        // copied after each of those rows
        const int n_snaps = std::min(T - 1, n_snap_);
        if (n_snaps <= 0) {
            k::glm5_kda_conv(g_q_, conv, b.conv_q, g_qc_, C, T, true, cs);
            k::glm5_kda_conv(g_k_, conv + hist, b.conv_k, g_kc_, C, T, true, cs);
            k::glm5_kda_conv(g_v_, conv + 2 * hist, b.conv_v, g_vc_, C, T, true, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                const size_t r = (size_t) t * (size_t) C;
                k::glm5_kda_conv(g_q_ + r, conv, b.conv_q, g_qc_ + r, C, 1, true, cs);
                k::glm5_kda_conv(g_k_ + r, conv + hist, b.conv_k, g_kc_ + r, C, 1, true, cs);
                k::glm5_kda_conv(g_v_ + r, conv + 2 * hist, b.conv_v, g_vc_ + r, C, 1, true, cs);
                if (t < n_snaps)
                    k::copy_from_mapped(conv_snap_[(size_t) b.idx][(size_t) t], conv, (int64_t) (conv_bytes_ / 4), cs);
            }
        }
        float* snaps[k::kGlm5MaxSnaps] = {};
        for (int t = 0; t < n_snaps; ++t) snaps[t] = gdn_snap_[(size_t) b.idx][(size_t) t];
        k::glm5_l2_heads(g_qc_, H, kL2Eps, T, cs);
        k::glm5_l2_heads(g_kc_, H, kL2Eps, T, cs);
        k::glm5_kda_gate(g_f_, b.dt, b.a, (float) s.kda_gate_lower_bound, g_gate_, H, S, T, cs);
        k::native_gdn_beta_gate(g_beta_, H * T, cs);
        k::glm5_kda_scan(state, g_qc_, g_kc_, g_vc_, g_gate_, g_beta_, g_o_, H, T, cs, snaps,   // the recurrence
                         std::max(n_snaps, 0));
        k::glm5_kda_out(g_o_, g_g2_, b.o_norm, eps, g_oy_, H, T, cs);
        gproj(b.o, g_oy_, mix_, T, false, cs);
    } else {
        record_dsa(b, T, cs);
    }
    k::glm5_hc_post(mix_, g_hs_, g_post_, g_comb_, g_hs_, N, T, cs);

    // ---- the FFN
    hc_read(b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, b.ffn_norm);
    auto swiglu = [&](float* out) {   // gate / up from hn_'s q8_1, the clamp, down
        const int FW = b.ffn_gate.n_out;
        if (b.ffn_gate.type != b.ffn_up.type ||
            !k::native_mmvq_pair(b.ffn_gate.type, b.ffn_gate.data, b.ffn_up.data, g_xq_, g_ffg_, g_ffu_, (int) N, FW, T,
                                 cs)) {
            k::native_mmvq(b.ffn_gate.type, b.ffn_gate.data, g_xq_, g_ffg_, (int) N, FW, T, cs);
            k::native_mmvq(b.ffn_up.type, b.ffn_up.data, g_xq_, g_ffu_, (int) N, FW, T, cs);
        }
        k::glm5_swiglu_clamp(g_ffg_, g_ffu_, g_ffg_, (int64_t) T * FW, b.clamp, cs);
        gproj(b.ffn_down, g_ffg_, out, T, false, cs);
    };
    const float* ffn_out = mix_;
    if (l < s.n_dense_lead) {
        swiglu(mix_);
    } else {
        if (!record_moe(b, s.moe_index(l), T, ring, segs, cs, err)) return false;
        ffn_out = moe_out_;
    }
    k::glm5_hc_post(ffn_out, g_hs_, g_post_, g_comb_, g_hs_, N, T, cs);
    tap_rows(l, g_hs_, kHC, T, 0, true, cs);   // DFlash2's taps: the mean of the four streams (sglang's capture)
    return true;
}

void PlainModel::record_dsa(const GBlock& b, int T, strata::gpu::Stream cs) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, QLR = s.q_lora_rank, KVR = s.kv_lora_rank, QK = s.key_length_mla,
                  VD = s.value_length_mla, IH = s.idx_n_head, ID = s.idx_key_length;
    const float eps = (float) s.rms_eps;
        const int64_t d = b.idx;
        // from the block row: q_a, kv_a, the indexer's key and pool gate (its q8_1), its head weights (F32)
        gproj(b.q_a, hn_, g_qa_, T, true, cs);
        gproj(b.kv_a, hn_, g_absT_, T, true, cs);   // the raw latent, normed into g_kv_ below
        gproj(b.idx_k, hn_, g_ik_, T, true, cs);
        gproj(b.idx_gate, hn_, g_ig_, T, true, cs);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.idx_proj, g_iw_, IH, N, IH, T, cs);
        k::scale_inplace(g_iw_, (int64_t) T * IH, 1.0f / std::sqrt((float) (ID * IH)), cs);
        k::native_qsa_rms_norm_weighted(g_absT_, b.kv_a_norm, g_kv_, (int) KVR, T, eps, cs);
        k::native_qsa_rms_norm_weighted_q8_1(g_qa_, b.q_a_norm, g_qr_, g_xq_, (int) QLR, T, eps, cs);
        gproj(b.q_b, g_qr_, g_qb_, T, true, cs);
        gproj(b.idx_q_b, g_qr_, g_iq_, T, true, cs);
        // the indexer: the window's keys and pools, the pools' scores, the selection
        k::glm5_idx_append(g_ik_, g_ig_, b.idx_k_norm, b.idx_k_norm_b, s.norm_eps > 0 ? (float) s.norm_eps : 1e-6f, g_steps_, T,
                           g_idx_cache_[(size_t) d], cs);
        k::glm5_idx_pool(g_idx_cache_[(size_t) d], b.idx_ape, (int) s.idx_kpool, g_steps_, T, g_pooled_[(size_t) d], cs);
        k::glm5_idx_scores(g_pooled_[(size_t) d], g_iq_, g_iw_, (int) IH, g_steps_, T, g_max_blocks_, g_scores_, cs);
        k::QsaShapes qs;
        qs.idx_block = s.idx_kpool;
        qs.idx_top_k = s.idx_top_k;
        k::qsa_block_topk(g_scores_, g_steps_, T, g_max_blocks_, g_cap_, qs, g_ids_, cs);
        // the MLA: the latent into the cells, q through each head's wk_b, the attention, each head's wv_b
        k::glm5_latent_store(g_kv_, g_steps_, T, KVR, g_latent_[(size_t) d], cs);
        const size_t kb = k::native_mmvq_weight_bytes(file_type(b.k_b.type), (int) QK, (int) KVR);
        const size_t vb = k::native_mmvq_weight_bytes(file_type(b.v_b.type), (int) KVR, (int) VD);
        k::glm5_transpose_abd(g_qb_, g_qbT_, T, H, QK, cs);
        k::native_quantize_q8_1_rows(g_qbT_, g_xq_, (int) QK, H * T, cs);
        for (int64_t h = 0; h < H; ++h)
            k::native_mmvq(b.k_b.type, (const uint8_t*) b.k_b.data + h * kb,
                           g_xq_ + k::native_q8_1_rows_bytes((int) QK, h * T), g_absT_ + h * T * KVR, (int) QK,
                           (int) KVR, T, cs);
        k::glm5_transpose_abd(g_absT_, g_abs_, H, T, KVR, cs);
        const int tail = s.idx_kpool_select_tail ? (int) s.idx_kpool : 0;
        if (g_lat_gather_ != nullptr) {   // --kv-ram: each token's cells from RAM once, the attention over them
            k::glm5_latent_gather(g_latent_[(size_t) d], g_ids_, g_cap_, g_steps_, T, tail, g_gather_stride_, g_lat_gather_,
                                  cs);
            k::glm5_mla_attn(g_abs_, g_lat_gather_, g_ids_, g_cap_, g_steps_, (int) H, 1.0f / std::sqrt((float) QK), T,
                             g_lat_, cs, tail, g_gather_stride_);
        } else {
            k::glm5_mla_attn(g_abs_, g_latent_[(size_t) d], g_ids_, g_cap_, g_steps_, (int) H,
                             1.0f / std::sqrt((float) QK), T, g_lat_, cs, tail);
        }
        k::glm5_transpose_abd(g_lat_, g_latT_, T, H, KVR, cs);
        k::native_quantize_q8_1_rows(g_latT_, g_xq_, (int) KVR, H * T, cs);
        for (int64_t h = 0; h < H; ++h)
            k::native_mmvq(b.v_b.type, (const uint8_t*) b.v_b.data + h * vb,
                           g_xq_ + k::native_q8_1_rows_bytes((int) KVR, h * T), g_voT_ + h * T * VD, (int) KVR,
                           (int) VD, T, cs);
        k::glm5_transpose_abd(g_voT_, g_vo_, H, T, VD, cs);
        gproj(b.o, g_vo_, mix_, T, false, cs);
}

bool PlainModel::record_moe(const GBlock& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                            strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, E = s.n_expert, K = s.n_expert_used;
    k::f32_gemv_fp32_mmvf_multi(hn_, N, b.router, rlog_, E, N, E, T, cs);
    const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
    k::glm5_router_sigmoid(rlog_, b.router_b, T, (int) E, (int) K, s.expert_weights_norm, scale, ids_, w_, cs);
    moe_ring(ml, T, ring, nullptr, cs);
    {   // the shared expert while the host serves: gate / up from hn_'s q8_1, the clamp, down
        const int FW = b.ffn_gate.n_out;
        if (b.ffn_gate.type != b.ffn_up.type ||
            !k::native_mmvq_pair(b.ffn_gate.type, b.ffn_gate.data, b.ffn_up.data, g_xq_, g_ffg_, g_ffu_, (int) N, FW, T,
                                 cs)) {
            k::native_mmvq(b.ffn_gate.type, b.ffn_gate.data, g_xq_, g_ffg_, (int) N, FW, T, cs);
            k::native_mmvq(b.ffn_up.type, b.ffn_up.data, g_xq_, g_ffu_, (int) N, FW, T, cs);
        }
        k::glm5_swiglu_clamp(g_ffg_, g_ffu_, g_ffg_, (int64_t) T * FW, b.clamp, cs);
        gproj(b.ffn_down, g_ffg_, shared_, T, false, cs);
    }
    if (!moe_experts(ml, T, ring, segs, cs, err)) return false;
    win_.combine(0, 0, T, w_, shared_, moe_out_, nullptr, cs);
    return true;
}

bool PlainModel::record_mtp_glm5(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd;
    const float eps = (float) s.rms_eps;
    const GBlock& b = gmtp_;
    k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::glm5_idx_steps(step_, T, (int) s.idx_kpool, (int) s.idx_top_k, g_steps_, cs);
    // llama.cpp's glm5-next graph_mtp: x = eh_proj([rms_norm(e) enorm | rms_norm(h) hnorm]), then the block with
    // plain residuals, and the shared head's norm
    k::native_qsa_rms_norm_weighted(x_, mtp_enorm_, hn_, (int) N, T, eps, cs);
    k::native_qsa_rms_norm_weighted(mtp_h_, mtp_hnorm_, mix_, (int) N, T, eps, cs);
    k::concat_rows(g_qbT_, hn_, mix_, N, T, cs);
    gproj(mtp_eh_, g_qbT_, x_, T, false, cs);
    k::native_qsa_rms_norm_weighted_q8_1(x_, b.attn_norm, hn_, g_xq_, (int) N, T, eps, cs);
    record_dsa(b, T, cs);
    k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    k::native_qsa_rms_norm_weighted_q8_1(x_, b.ffn_norm, hn_, g_xq_, (int) N, T, eps, cs);
    uint32_t ring = 0;
    if (!record_moe(b, n_moe_ - 1, T, ring, msegs_[(size_t) T], cs, err)) return false;
    k::add_inplace(x_, moe_out_, (int64_t) T * N, cs);
    // the last row only: its norm is the next step's trunk row, and its logits pick the draft
    k::native_qsa_rms_norm_weighted(x_ + (size_t) (T - 1) * N, mtp_head_norm_, hn_, (int) N, 1, eps, cs);
    k::copy_from_mapped(mtp_h_, hn_, N, cs);
    draft_logits(step_ + (size_t) (T - 1) * k::kStepCount + k::kStepPos, cs);
    return true;
}

bool PlainModel::record_window_glm5(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd;
    const float eps = (float) s.rms_eps;
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    // the selection's step records, shared by every DSA block of the window
    k::glm5_idx_steps(step_, T, (int) s.idx_kpool, (int) s.idx_top_k, g_steps_, cs);
    if (first()) {   // the four streams from the embeddings, or as the stage before left them
        k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
        k::glm5_hc_init(x_, g_hs_, N, T, cs);
    } else {
        k::copy_from_mapped(g_hs_, m_in_, (int64_t) T * kHC * N, cs);
    }
    uint32_t ring = 0;
    for (int64_t l = lb_; l < le_; ++l)
        if (!record_gblock(gblocks_[(size_t) l], l, T, ring, segs_[(size_t) T], cs, err)) return false;
    if (!last()) {
        k::copy_from_mapped(m_hand_, g_hs_, (int64_t) T * kHC * N, cs);
        return true;
    }
    k::glm5_hc_mean(g_hs_, x_, N, T, cs);
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, eps, cs);
    head_logits(hn_, T, cs);   // the tokens are picked after the graph (forward), with the run's sampler
    return true;
}

}  // namespace strata::core
