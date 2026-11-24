// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_runner.h - Inkling-2-Mini's 48-visit forward on the GPU (upstream's SYCL runner, on
// 16 logical lanes: ink_sy_common.h): SQ-20 (SF4 + MX8), SQ-15 (SF4 + S3/S4/S5_K), SQ-10 (S2/S3_K + SF4); 1..64 slots;
// the experts placed by vram_load_order; the DSpark Block-8 drafter on the GPU; ShortConv rolled back from a ring with
// no replay (the exact rollback of a rejected draft).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_sy_codecs.h"
#include "ink_sy_xmx.h"
#include "ink_sy_common.h"
#include "ink_sy_dspark.h"
#include "ink_sy_hybrid.h"
#include "ink_sy_ops.h"
#include "ink_sy_qsa.h"
#include "ink_sy_vision.h"

namespace ink_sycl {

struct InkSyRunner {
    InkSyModel* mod = nullptr;
    int max_slots = 1;
    int max_ctx = 4096;
    int max_batch = 256;
    bool use_dspark = true;

    std::vector<int> visit_kv_stride;
    std::vector<half*> kv_k;
    std::vector<half*> kv_v;
    std::vector<float*> sc_hist_k;
    std::vector<float*> sc_hist_v;
    std::vector<float*> sc_hist_attn;
    std::vector<float*> sc_hist_mlp;

    // Pre-verify snapshot buffers + per-step verify ring [48] x [8, 3, C] for O(1) zero-replay rollback!
    std::vector<float*> sc_snap_k;
    std::vector<float*> sc_snap_v;
    std::vector<float*> sc_snap_attn;
    std::vector<float*> sc_snap_mlp;
    std::vector<float*> sc_ring_k;
    std::vector<float*> sc_ring_v;
    std::vector<float*> sc_ring_attn;
    std::vector<float*> sc_ring_mlp;
    int last_ver_slot = -1, last_ver_pos0 = -1, last_ver_m = 0;

    // DSpark persistent state
    half* dsp_ctx_k[3]{};
    half* dsp_ctx_v[3]{};
    float* dsp_frnn_slot = nullptr;   // [max_slots, D]
    float* dsp_frnn_tmp = nullptr;    // [max_batch, D]
    float* bd_cur = nullptr;
    float* bd_res = nullptr;
    float* bd_h = nullptr;
    float* bd_q = nullptr;
    float* bd_k = nullptr;
    float* bd_v = nullptr;
    float* bd_ao = nullptr;
    float* bd_proj = nullptr;
    float* bd_mg = nullptr;
    float* bd_mu = nullptr;
    float* bd_mh = nullptr;
    float* bd_logits = nullptr;       // [vocab_padded] (reused across the 7 Markov steps via linearity!)
    float* bd_state = nullptr;        // [256]
    float* bd_joint = nullptr;        // [5120]
    float* bd_rproj = nullptr;        // [768]
    float* bd_latent = nullptr;       // [256]
    float* bd_corr = nullptr;         // [D]
    float* bd_comb = nullptr;         // [D] = cur[s] + corr
    float* bd_part_v = nullptr;       // [256]
    int* bd_part_i = nullptr;         // [256]
    int* bd_tok_chain = nullptr;      // [8]

    // Per-step activations
    int* b_slot_ids = nullptr;
    int* b_pos = nullptr;
    int* b_tok_ids = nullptr;
    uint8_t* b_is_img = nullptr;
    float* b_img_emb = nullptr;
    int* b_active_slots = nullptr;
    int* b_slot_tok_off = nullptr;
    int* b_slot_tok_cnt = nullptr;
    int* b_slot_tok_idx = nullptr;

    float* b_cur = nullptr;
    float* b_res = nullptr;
    float* b_blocks = nullptr;
    float* b_prelude = nullptr;
    float* b_bank_blk = nullptr;
    float* b_bank_out = nullptr;
    float* b_msgs = nullptr;
    float* b_in3 = nullptr;
    float* b_cat = nullptr;
    float* b_ctrl_gate = nullptr;

    float* b_q = nullptr;
    float* b_k_raw = nullptr;
    float* b_v_raw = nullptr;
    float* b_k = nullptr;
    float* b_v = nullptr;
    float* b_r = nullptr;
    float* b_kd_taps = nullptr;
    float* b_vd_taps = nullptr;
    float* b_sinks = nullptr;
    float* b_gate_attn = nullptr;
    float* b_attn_out = nullptr;
    float* b_wo_raw = nullptr;
    float* b_wo = nullptr;

    float* b_mlp_in = nullptr;
    float* b_r_logits = nullptr;
    int* b_topk_idx = nullptr;
    float* b_topk_w = nullptr;
    int* b_moe_uexp = nullptr;
    int* b_moe_uoff = nullptr;
    int* b_moe_ucnt = nullptr;
    int* b_moe_pairs = nullptr;
    int* b_moe_n_uniq = nullptr;
    float* b_moe_h = nullptr;
    float* b_moe_yp = nullptr;
    float* b_moe_raw = nullptr;
    float* b_moe_out = nullptr;

    float* b_h_out = nullptr;
    float* b_h_norm = nullptr;
    float* b_logits = nullptr;
    int* b_argmax = nullptr;
    float* b_taps4 = nullptr;

    int8_t* xq_s8_dev = nullptr;
    float* xsc_dev = nullptr;

    InkSyVision vis;
    InkSyHybridEngine hybrid;
    InkSyQsa qsa;

    ~InkSyRunner() { release(); }

    void release() {
        hybrid.release();
        qsa.release();
        for (auto* p : kv_k) dfree(p);
        for (auto* p : kv_v) dfree(p);
        for (auto* p : sc_hist_k) dfree(p);
        for (auto* p : sc_hist_v) dfree(p);
        for (auto* p : sc_hist_attn) dfree(p);
        for (auto* p : sc_hist_mlp) dfree(p);
        for (auto* p : sc_snap_k) dfree(p);
        for (auto* p : sc_snap_v) dfree(p);
        for (auto* p : sc_snap_attn) dfree(p);
        for (auto* p : sc_snap_mlp) dfree(p);
        for (auto* p : sc_ring_k) dfree(p);
        for (auto* p : sc_ring_v) dfree(p);
        for (auto* p : sc_ring_attn) dfree(p);
        for (auto* p : sc_ring_mlp) dfree(p);
        kv_k.clear(); kv_v.clear();
        sc_hist_k.clear(); sc_hist_v.clear(); sc_hist_attn.clear(); sc_hist_mlp.clear();
        sc_snap_k.clear(); sc_snap_v.clear(); sc_snap_attn.clear(); sc_snap_mlp.clear();
        sc_ring_k.clear(); sc_ring_v.clear(); sc_ring_attn.clear(); sc_ring_mlp.clear();

        for (int i = 0; i < 3; ++i) { dfree(dsp_ctx_k[i]); dfree(dsp_ctx_v[i]); dsp_ctx_k[i] = dsp_ctx_v[i] = nullptr; }
        dfree(dsp_frnn_slot); dfree(dsp_frnn_tmp);
        dfree(bd_cur); dfree(bd_res); dfree(bd_h); dfree(bd_q); dfree(bd_k); dfree(bd_v);
        dfree(bd_ao); dfree(bd_proj); dfree(bd_mg); dfree(bd_mu); dfree(bd_mh);
        dfree(bd_logits); dfree(bd_state); dfree(bd_joint); dfree(bd_rproj); dfree(bd_latent);
        dfree(bd_corr); dfree(bd_comb); dfree(bd_part_v); dfree(bd_part_i); dfree(bd_tok_chain);

        dfree(b_slot_ids); dfree(b_pos); dfree(b_tok_ids); dfree(b_is_img); dfree(b_img_emb);
        dfree(b_active_slots); dfree(b_slot_tok_off); dfree(b_slot_tok_cnt); dfree(b_slot_tok_idx);
        dfree(b_cur); dfree(b_res); dfree(b_blocks); dfree(b_prelude); dfree(b_bank_blk); dfree(b_bank_out);
        dfree(b_msgs); dfree(b_in3); dfree(b_cat); dfree(b_ctrl_gate);
        dfree(b_q); dfree(b_k_raw); dfree(b_v_raw); dfree(b_k); dfree(b_v); dfree(b_r);
        dfree(b_kd_taps); dfree(b_vd_taps); dfree(b_sinks); dfree(b_gate_attn); dfree(b_attn_out);
        dfree(b_wo_raw); dfree(b_wo);
        dfree(b_mlp_in); dfree(b_r_logits); dfree(b_topk_idx); dfree(b_topk_w);
        dfree(b_moe_uexp); dfree(b_moe_uoff); dfree(b_moe_ucnt); dfree(b_moe_pairs); dfree(b_moe_n_uniq);
        dfree(b_moe_h); dfree(b_moe_yp); dfree(b_moe_raw); dfree(b_moe_out);
        dfree(b_h_out); dfree(b_h_norm); dfree(b_logits); dfree(b_argmax); dfree(b_taps4);
        dfree(xq_s8_dev); dfree(xsc_dev);
        dsp_frnn_slot = nullptr; b_cur = nullptr;
    }

    bool init(InkSyModel& m, int slots, int ctx_len, int batch_cap, bool enable_dspark, std::string& err) {
        release();
        mod = &m;
        max_slots = std::max(1, std::min(64, slots));
        max_ctx = std::max(256, ctx_len);
        max_batch = std::max(64, batch_cap);
        use_dspark = enable_dspark && m.man.c.has_dspark;

        const auto& c = m.man.c;
        kv_k.resize(c.n_visits); kv_v.resize(c.n_visits); visit_kv_stride.resize(c.n_visits);
        sc_hist_k.resize(c.n_visits); sc_hist_v.resize(c.n_visits);
        sc_hist_attn.resize(c.n_visits); sc_hist_mlp.resize(c.n_visits);
        sc_snap_k.resize(c.n_visits); sc_snap_v.resize(c.n_visits);
        sc_snap_attn.resize(c.n_visits); sc_snap_mlp.resize(c.n_visits);
        sc_ring_k.resize(c.n_visits); sc_ring_v.resize(c.n_visits);
        sc_ring_attn.resize(c.n_visits); sc_ring_mlp.resize(c.n_visits);

        size_t kv_tot = 0, sc_tot = 0;
        for (int v = 0; v < c.n_visits; ++v) {
            const int ip = c.phys_for_visit(v);
            const int stride = c.is_swa[ip] ? c.swa_win : max_ctx;
            visit_kv_stride[v] = stride;
            const size_t kve = (size_t)max_slots * stride * c.n_kv * c.head_dim;
            kv_k[v] = dalloc<half>(kve); kv_v[v] = dalloc<half>(kve);
            kv_tot += 2 * kve * sizeof(half);

            const size_t sc_kv = (size_t)max_slots * 3 * 1024;
            const size_t sc_d  = (size_t)max_slots * 3 * c.hidden;
            sc_hist_k[v] = dalloc<float>(sc_kv); sc_hist_v[v] = dalloc<float>(sc_kv);
            sc_hist_attn[v] = dalloc<float>(sc_d); sc_hist_mlp[v] = dalloc<float>(sc_d);
            sc_snap_k[v] = dalloc<float>(sc_kv); sc_snap_v[v] = dalloc<float>(sc_kv);
            sc_snap_attn[v] = dalloc<float>(sc_d); sc_snap_mlp[v] = dalloc<float>(sc_d);
            sc_ring_k[v] = dalloc<float>(8ul * 3 * 1024); sc_ring_v[v] = dalloc<float>(8ul * 3 * 1024);
            sc_ring_attn[v] = dalloc<float>(8ul * 3 * c.hidden); sc_ring_mlp[v] = dalloc<float>(8ul * 3 * c.hidden);
            dzero(sc_hist_k[v], sc_kv * 4); dzero(sc_hist_v[v], sc_kv * 4);
            dzero(sc_hist_attn[v], sc_d * 4); dzero(sc_hist_mlp[v], sc_d * 4);
            sc_tot += (2 * sc_kv + 2 * sc_d) * 4;
        }

        const size_t D = (size_t)c.hidden, MB = (size_t)max_batch;
        b_slot_ids = dalloc<int>(MB); b_pos = dalloc<int>(MB); b_tok_ids = dalloc<int>(MB);
        b_is_img = dalloc<uint8_t>(MB); b_img_emb = dalloc<float>(MB * D);
        b_active_slots = dalloc<int>(max_slots); b_slot_tok_off = dalloc<int>(max_slots);
        b_slot_tok_cnt = dalloc<int>(max_slots); b_slot_tok_idx = dalloc<int>(MB);

        b_cur = dalloc<float>(MB * D); b_res = dalloc<float>(MB * D);
        b_blocks = dalloc<float>(6 * MB * D); b_prelude = dalloc<float>(MB * D);
        b_bank_blk = dalloc<float>(2 * MB * D); b_bank_out = dalloc<float>(MB * D);
        b_msgs = dalloc<float>(4 * MB * D); b_in3 = dalloc<float>(3 * MB * D);
        b_cat = dalloc<float>(MB * 2 * D); b_ctrl_gate = dalloc<float>(MB * c.gate_rank);

        b_q = dalloc<float>(MB * 3072); b_k_raw = dalloc<float>(MB * 1024); b_v_raw = dalloc<float>(MB * 1024);
        b_k = dalloc<float>(MB * 1024); b_v = dalloc<float>(MB * 1024); b_r = dalloc<float>(MB * 384);
        b_kd_taps = dalloc<float>(MB * 32); b_vd_taps = dalloc<float>(MB * 32);
        b_sinks = dalloc<float>(MB * 24); b_gate_attn = dalloc<float>(MB * 3072);
        b_attn_out = dalloc<float>(MB * 3072); b_wo_raw = dalloc<float>(MB * D); b_wo = dalloc<float>(MB * D);

        b_mlp_in = dalloc<float>(MB * D); b_r_logits = dalloc<float>(MB * 128);
        b_topk_idx = dalloc<int>(MB * 8); b_topk_w = dalloc<float>(MB * 8);
        b_moe_uexp = dalloc<int>(128); b_moe_uoff = dalloc<int>(128); b_moe_ucnt = dalloc<int>(128);
        b_moe_pairs = dalloc<int>(MB * 8); b_moe_n_uniq = dalloc<int>(1);
        b_moe_h = dalloc<float>(MB * 8 * 1152); b_moe_yp = dalloc<float>(MB * 8 * D);
        b_moe_raw = dalloc<float>(MB * D); b_moe_out = dalloc<float>(MB * D);

        b_h_out = dalloc<float>(MB * D); b_h_norm = dalloc<float>(MB * D);
        b_logits = dalloc<float>((size_t)std::min(max_batch, 64) * c.vocab_padded);
        b_argmax = dalloc<int>(MB); b_taps4 = dalloc<float>(4 * MB * D);
        xq_s8_dev = dalloc<int8_t>(MB * 9216); xsc_dev = dalloc<float>(MB);

        if (use_dspark) {
            const size_t dkv = (size_t)max_slots * max_ctx * 1024;
            for (int l = 0; l < 3; ++l) { dsp_ctx_k[l] = dalloc<half>(dkv); dsp_ctx_v[l] = dalloc<half>(dkv); }
            dsp_frnn_slot = dalloc<float>((size_t)max_slots * D);
            dsp_frnn_tmp = dalloc<float>(MB * D);
            bd_cur = dalloc<float>(8 * D); bd_res = dalloc<float>(8 * D); bd_h = dalloc<float>(8 * D);
            bd_q = dalloc<float>(8 * 3072); bd_k = dalloc<float>(8 * 1024); bd_v = dalloc<float>(8 * 1024);
            bd_ao = dalloc<float>(8 * 3072); bd_proj = dalloc<float>(8 * D);
            bd_mg = dalloc<float>(8 * 9216); bd_mu = dalloc<float>(8 * 9216); bd_mh = dalloc<float>(8 * 9216);
            bd_logits = dalloc<float>(c.vocab_padded);
            bd_state = dalloc<float>(256); bd_joint = dalloc<float>(5120); bd_rproj = dalloc<float>(768);
            bd_latent = dalloc<float>(256); bd_corr = dalloc<float>(D); bd_comb = dalloc<float>(D);
            bd_part_v = dalloc<float>(256); bd_part_i = dalloc<int>(256); bd_tok_chain = dalloc<int>(8);
        }

        wait();
        hybrid.init(m);
        fprintf(stderr, "[ink_sycl] runner ready: slots=%d ctx=%d batch=%d dspark=%d hybrid=%d | KV=%.1f MiB SC=%.1f MiB DevAlloc=%.2f GiB\n",
                max_slots, max_ctx, max_batch, (int)use_dspark, (int)hybrid.active,
                kv_tot / (1024.0 * 1024.0), sc_tot / (1024.0 * 1024.0),
                dev_bytes_allocated() / (1024.0 * 1024.0 * 1024.0));
        (void)err;
        return true;
    }

    void reset_slot(int s) {
        if (s < 0 || s >= max_slots) return;
        const size_t sc_kv = 3ul * 1024 * 4, sc_d = 3ul * INK_D * 4;
        for (int v = 0; v < mod->man.c.n_visits; ++v) {
            dzero(sc_hist_k[v] + (size_t)s * 3 * 1024, sc_kv);
            dzero(sc_hist_v[v] + (size_t)s * 3 * 1024, sc_kv);
            dzero(sc_hist_attn[v] + (size_t)s * 3 * INK_D, sc_d);
            dzero(sc_hist_mlp[v] + (size_t)s * 3 * INK_D, sc_d);
        }
        wait();
    }

    void snapshot_slot(int s) {
        if (s < 0 || s >= max_slots) return;
        const size_t sc_kv = 3ul * 1024 * 4, sc_d = 3ul * INK_D * 4;
        for (int v = 0; v < mod->man.c.n_visits; ++v) {
            d2d(sc_snap_k[v] + (size_t)s * 3 * 1024, sc_hist_k[v] + (size_t)s * 3 * 1024, sc_kv);
            d2d(sc_snap_v[v] + (size_t)s * 3 * 1024, sc_hist_v[v] + (size_t)s * 3 * 1024, sc_kv);
            d2d(sc_snap_attn[v] + (size_t)s * 3 * INK_D, sc_hist_attn[v] + (size_t)s * 3 * INK_D, sc_d);
            d2d(sc_snap_mlp[v] + (size_t)s * 3 * INK_D, sc_hist_mlp[v] + (size_t)s * 3 * INK_D, sc_d);
        }
    }

    void rollback_slot(int s, int keep_len) {
        if (s < 0 || s >= max_slots) return;
        const size_t sc_kv = 3ul * 1024 * 4, sc_d = 3ul * INK_D * 4;
        if (s == last_ver_slot && keep_len > last_ver_pos0 && keep_len <= last_ver_pos0 + last_ver_m) {
            const int step_idx = keep_len - last_ver_pos0 - 1;
            for (int v = 0; v < mod->man.c.n_visits; ++v) {
                d2d(sc_hist_k[v] + (size_t)s * 3 * 1024, sc_ring_k[v] + (size_t)step_idx * 3 * 1024, sc_kv);
                d2d(sc_hist_v[v] + (size_t)s * 3 * 1024, sc_ring_v[v] + (size_t)step_idx * 3 * 1024, sc_kv);
                d2d(sc_hist_attn[v] + (size_t)s * 3 * INK_D, sc_ring_attn[v] + (size_t)step_idx * 3 * INK_D, sc_d);
                d2d(sc_hist_mlp[v] + (size_t)s * 3 * INK_D, sc_ring_mlp[v] + (size_t)step_idx * 3 * INK_D, sc_d);
            }
            if (use_dspark) {
                d2d(dsp_frnn_slot + (size_t)s * INK_D, dsp_frnn_tmp + (size_t)step_idx * INK_D, (size_t)INK_D * 4);
            }
            return;
        }
        for (int v = 0; v < mod->man.c.n_visits; ++v) {
            d2d(sc_hist_k[v] + (size_t)s * 3 * 1024, sc_snap_k[v] + (size_t)s * 3 * 1024, sc_kv);
            d2d(sc_hist_v[v] + (size_t)s * 3 * 1024, sc_snap_v[v] + (size_t)s * 3 * 1024, sc_kv);
            d2d(sc_hist_attn[v] + (size_t)s * 3 * INK_D, sc_snap_attn[v] + (size_t)s * 3 * INK_D, sc_d);
            d2d(sc_hist_mlp[v] + (size_t)s * 3 * INK_D, sc_snap_mlp[v] + (size_t)s * 3 * INK_D, sc_d);
        }
    }

    void dispatch_linear(const InkTensorInfo& ti, const float* X, float* Y, int M, int N_override = -1) {
        const int N = (N_override > 0) ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        switch (ti.fmt) {
            case FMT_BF16:
                if (ink_xmx_linear_bf16(reinterpret_cast<const uint16_t*>(ti.dev_ptr), N, K, X, Y, M)) break;
                if (M >= 16) k_gemm_bf16_sg16(reinterpret_cast<const uint16_t*>(ti.dev_ptr), N, K, X, Y, M);
                else k_linear_bf16_sg16(reinterpret_cast<const uint16_t*>(ti.dev_ptr), N, K, X, Y, M);
                break;
            case FMT_Q8ROW:
                if (ink_xmx_linear_q8row(ti.dev_ptr, N, K, X, Y, M, (int)ti.rows())) break;
                if (M >= 16) {
                    k_gemm_q8row_sg16(ti.dev_ptr, N, K, X, Y, M, (int)ti.rows());
                } else {
                    k_quantize_x_q8_sg16(X, M, K, xq_s8_dev, xsc_dev);
                    k_linear_q8row_dp4a_sg16(ti.dev_ptr, N, K, xq_s8_dev, xsc_dev, Y, M, (int)ti.rows());
                }
                break;
            case FMT_Q8_0:
                if (!ink_xmx_linear_q8_0(ti.dev_ptr, N, K, X, Y, M, (int)ti.rows()))
                    k_linear_q8_0_sg16(ti.dev_ptr, N, K, X, Y, M, (int)ti.rows());
                break;
            case FMT_F32:
                k_linear_f32_sg16(reinterpret_cast<const float*>(ti.dev_ptr), N, K, X, Y, M);
                break;
            case FMT_S2: if (!ink_xmx_linear_sk<FMT_S2>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S2>(ti.dev_ptr, N, K, X, Y, M); break;
            case FMT_S3: if (!ink_xmx_linear_sk<FMT_S3>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S3>(ti.dev_ptr, N, K, X, Y, M); break;
            case FMT_S4: if (!ink_xmx_linear_sk<FMT_S4>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S4>(ti.dev_ptr, N, K, X, Y, M); break;
            case FMT_S5: if (!ink_xmx_linear_sk<FMT_S5>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S5>(ti.dev_ptr, N, K, X, Y, M); break;
            case FMT_S6: if (!ink_xmx_linear_sk<FMT_S6>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S6>(ti.dev_ptr, N, K, X, Y, M); break;
            case FMT_S8: if (!ink_xmx_linear_sk<FMT_S8>(ti.dev_ptr, N, K, X, Y, M)) k_linear_sk_sg16<FMT_S8>(ti.dev_ptr, N, K, X, Y, M); break;
            default:
                fprintf(stderr, "[ink_sycl] unsupported linear fmt %d for %s\n", ti.fmt, ti.name.c_str());
                std::exit(2);
        }
    }

    const float* tf32(const std::string& n) const { return reinterpret_cast<const float*>(mod->man.get(n).dev_ptr); }
    const float* df32(const std::string& n) const { return reinterpret_cast<const float*>(mod->man.get_d(n).dev_ptr); }

    bool forward_batch(const int* tok_ids, const int* slot_ids, const int* positions, int M,
                       const uint8_t* is_img_host, const float* img_emb_host,
                       int logits_mode,  // 0=none, 1=last token only, 2=all M tokens (or want_idx_host[0..n_want-1])
                       float* out_logits_host, int* out_argmax_host,
                       bool record_dspark_ctx,
                       const int* want_idx_host = nullptr, int n_want = 0) {
        if (M <= 0 || M > max_batch) return false;
        hybrid.decay_step();
        const auto& man = mod->man;
        const auto& c = man.c;
        const int D = c.hidden;
        const int block_stride = max_batch * D;
        const int M_stride = M * D;

        // Upload metadata & build per-slot token index lists for race-free multi-slot ShortConv
        std::unordered_map<int, std::vector<int>> by_slot;
        for (int m = 0; m < M; ++m) by_slot[slot_ids[m]].push_back(m);
        std::vector<int> h_as, h_off, h_cnt, h_idx;
        int cur_t = 0;
        for (const auto& kv : by_slot) {
            h_as.push_back(kv.first);
            h_off.push_back(cur_t);
            h_cnt.push_back((int)kv.second.size());
            for (int m : kv.second) { h_idx.push_back(m); ++cur_t; }
        }
        const int n_active_slots = (int)h_as.size();
        const int save_ring = (n_active_slots == 1 && M > 1 && M <= 8) ? 1 : 0;
        if (save_ring) {
            last_ver_slot = slot_ids[0];
            last_ver_pos0 = positions[0];
            last_ver_m = M;
        }

        if (qsa.loaded) qsa.plan(slot_ids, positions, M);
        h2d(b_tok_ids, tok_ids, (size_t)M * 4);
        h2d(b_slot_ids, slot_ids, (size_t)M * 4);
        h2d(b_pos, positions, (size_t)M * 4);
        h2d_sync(b_active_slots, h_as.data(), (size_t)n_active_slots * 4);
        h2d_sync(b_slot_tok_off, h_off.data(), (size_t)n_active_slots * 4);
        h2d_sync(b_slot_tok_cnt, h_cnt.data(), (size_t)n_active_slots * 4);
        h2d_sync(b_slot_tok_idx, h_idx.data(), (size_t)M * 4);
        if (is_img_host && img_emb_host) {
            h2d(b_is_img, is_img_host, (size_t)M);
            h2d(b_img_emb, img_emb_host, (size_t)M * D * 4);
        } else {
            dzero(b_is_img, (size_t)M);
        }

        // 1. Embedding + embed_norm (bypassed for image tokens)
        k_embed_sg16(man.get("model.llm.embed.weight"), tf32("model.llm.embed_norm.weight"),
                     b_tok_ids, b_is_img, b_img_emb, b_cur, M, c.vocab_padded);

        static const bool dbg_env = [] { const char* e = strata::env_get("INKLING_DEBUG"); return e && e[0] == '1'; }();
        const bool dbg = dbg_env && M == 1;   // XESTRATA_INKLING_DEBUG=1: each stage's statistics, one token
        auto dbg_buf = [&](const char* tag, const float* p, int n) {
            if (!dbg) return;
            std::vector<float> h(n);
            d2h(h.data(), p, (size_t)n * 4);
            wait();
            int nans = 0, infs = 0; float mn = 1e30f, mx = -1e30f; double ss = 0;
            for (float v : h) {
                if (std::isnan(v)) nans++;
                else if (!std::isfinite(v)) infs++;
                else { mn = std::min(mn, v); mx = std::max(mx, v); ss += (double)v * v; }
            }
            fprintf(stderr, "  [DBG %s] n=%d nan=%d inf=%d min=%.4g max=%.4g l2=%.4g\n", tag, n, nans, infs, mn, mx, std::sqrt(ss));
        };
        dbg_buf("embed", b_cur, M * D);

        int n_blocks = 0;
        int prelude_n_blocks = 0;

        auto run_visit = [&](int ip, int visit, int visit_pass, bool has_msgs) {
            if (record_dspark_ctx && use_dspark) {
                for (int ti = 0; ti < 3; ++ti) {
                    if (visit == c.dspark_taps[ti]) {
                        d2d(b_taps4 + (size_t)ti * M_stride, b_cur, (size_t)M_stride * 4);
                    }
                }
            }

            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = (c.is_swa[ip] != 0);

            d2d(b_res, b_cur, (size_t)M_stride * 4);

            // Attention-Residual mix before attention (ip > 0)
            if (ip > 0) {
                const int nq = is_swa ? 1 : 3;
                k_attn_res_mix_sg16(b_blocks, b_cur, tf32("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)),
                                    b_in3, n_blocks, block_stride, M, nq);
                if (is_swa && has_msgs) {
                    d2d(b_in3 + M_stride, b_in3, (size_t)M_stride * 4);
                    d2d(b_in3 + 2 * M_stride, b_in3, (size_t)M_stride * 4);
                }
            } else {
                d2d(b_in3, b_cur, (size_t)M_stride * 4);
            }

            // Cross-pass loop controllers (attn, k, v) + input RMSNorm
            if (has_msgs) {
                const char* cnames[3] = {"attn", "k", "v"};
                const char* nnames[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
                for (int j = 0; j < 3; ++j) {
                    float* inp = b_in3 + (size_t)j * M_stride;
                    const float* mp = b_msgs + (size_t)j * M_stride;
                    k_ctrl_cat_rms_sg16(inp, mp, b_cat, M);
                    dispatch_linear(man.get(p + ".controllers." + cnames[j] + ".down_weight"), b_cat, b_ctrl_gate, M);
                    const std::string nkey = is_swa ? (p + ".attn_norm.weight") : (p + "." + nnames[j]);
                    k_ctrl_apply_norm_sg16(inp, mp, b_ctrl_gate,
                                           tf32(p + ".controllers." + cnames[j] + ".gain_weight"),
                                           tf32(p + ".controllers." + cnames[j] + ".forget_weight"),
                                           tf32(nkey), inp, M, 1);
                }
            } else {
                if (is_swa) {
                    k_rms_norm_sg16(b_in3, tf32(p + ".attn_norm.weight"), b_in3, D, M);
                } else {
                    const char* nnames[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
                    for (int j = 0; j < 3; ++j) {
                        float* inp = b_in3 + (size_t)j * M_stride;
                        k_rms_norm_sg16(inp, tf32(p + "." + nnames[j]), inp, D, M);
                    }
                }
            }

            const float* in_q = b_in3;
            const float* in_k = (is_swa && !has_msgs) ? b_in3 : (b_in3 + M_stride);
            const float* in_v = (is_swa && !has_msgs) ? b_in3 : (b_in3 + 2 * M_stride);

            // Q, K, V, R, Sinks, GateAttn
            dispatch_linear(man.get(p + ".attn.wq_du.weight"), in_q, b_q, M);
            dispatch_linear(man.get(p + ".attn.wk_dv.weight"), in_k, b_k_raw, M);
            dispatch_linear(man.get(p + ".attn.wv_dv.weight"), in_v, b_v_raw, M);
            dispatch_linear(man.get(p + ".attn.wr_du.weight"), in_q, b_r, M);
            dispatch_linear(man.get(p + ".attn.attn_sink_proj.weight"), in_q, b_sinks, M);
            dispatch_linear(man.get(p + ".attn.gate_attn.weight"), in_q, b_gate_attn, M);
            if (!is_swa) {
                dispatch_linear(man.get(p + ".attn.k_sconv.weight_proj"), in_k, b_kd_taps, M);
                dispatch_linear(man.get(p + ".attn.v_sconv.weight_proj"), in_v, b_vd_taps, M);
            }

            // 4-tap ShortConv on K & V
            const float* sc_layer = mod->static_sconv_dev + (size_t)ip * InkSyModel::kSconvLayerFloats;
            const int has_dyn = is_swa ? 0 : 1;
            k_sconv4_multi_sg16(b_k_raw, sc_layer + InkSyModel::kSconvKOff, b_kd_taps, sc_hist_k[visit],
                                b_active_slots, b_slot_tok_off, b_slot_tok_cnt, b_slot_tok_idx,
                                b_k, n_active_slots, 1024, has_dyn, sc_ring_k[visit], save_ring);
            k_sconv4_multi_sg16(b_v_raw, sc_layer + InkSyModel::kSconvVOff, b_vd_taps, sc_hist_v[visit],
                                b_active_slots, b_slot_tok_off, b_slot_tok_cnt, b_slot_tok_idx,
                                b_v, n_active_slots, 1024, has_dyn, sc_ring_v[visit], save_ring);

            // Q/R L2 clip + tau, K L2 clip + V RMSNorm + KV store
            k_qr_clip_tau_sg16(b_q, tf32(p + ".attn.q_l2_clip.scale"),
                               b_r, tf32(p + ".attn.rel_l2_clip.scale"),
                               b_pos, M, is_swa ? 1 : 0);
            k_kv_norm_store_f16_sg16(b_k, b_v, kv_k[visit], kv_v[visit],
                                     b_slot_ids, b_pos, M, is_swa ? 1 : 0, visit_kv_stride[visit]);

            // The QSA indexer's keys and, where a query's context passes its budget, its selection (ink_sy_qsa.h)
            const int qf = (qsa.loaded && !is_swa) ? qsa.fidx_of_visit[(size_t)visit] : -1;
            const bool qsel = qf >= 0 && qsa.any_sparse;
            if (qf >= 0) qsa.keys(qf, in_k, M);
            if (qsel) qsa.select(qf, in_q, M);

            // Fused RelPos + Sink 24:8 GQA Flash Attention + sigmoid(gate_attn)
            k_relpos_sink_gqa_sg16(b_q, b_r, mod->rel_proj_pad_dev + (size_t)ip * 16 * 1025,
                                   b_sinks, b_gate_attn, kv_k[visit], kv_v[visit],
                                   b_slot_ids, b_pos, b_attn_out, M, is_swa ? 1 : 0,
                                   c.swa_win, c.rel_extent, visit_kv_stride[visit],
                                   qsel ? qsa.ids : nullptr, qsel ? qsa.steps : nullptr, qsel ? qsa.use : nullptr,
                                   qsa.cap);
            if (visit == 0) {
                dbg_buf("v0_q", b_q, 3072); dbg_buf("v0_k", b_k, 1024); dbg_buf("v0_v", b_v, 1024);
                dbg_buf("v0_sinks", b_sinks, 24); dbg_buf("v0_attn", b_attn_out, 3072);
            }

            // Output projection + attn_sconv
            dispatch_linear(man.get(p + ".attn.wo_ud.weight"), b_attn_out, b_wo_raw, M);
            k_sconv4_multi_sg16(b_wo_raw, sc_layer + InkSyModel::kSconvAttnOff, nullptr, sc_hist_attn[visit],
                                b_active_slots, b_slot_tok_off, b_slot_tok_cnt, b_slot_tok_idx,
                                b_wo, n_active_slots, D, 0, sc_ring_attn[visit], save_ring);
            if (visit == 0) { dbg_buf("v0_wo_raw", b_wo_raw, D); dbg_buf("v0_wo", b_wo, D); }

            // Attention-Residual block boundary rule (matches inkling2_loop.py:418!)
            if (ip % c.block_size == 0) {
                d2d(b_blocks + (size_t)n_blocks * block_stride, b_res, (size_t)M_stride * 4);
                ++n_blocks;
                d2d(b_cur, b_wo, (size_t)M_stride * 4);
            } else {
                k_add_sg16(b_res, b_wo, b_cur, M_stride);
            }

            // ---- MoE Sublayer ----
            d2d(b_res, b_cur, (size_t)M_stride * 4);
            k_attn_res_mix_sg16(b_blocks, b_cur, tf32("model.llm.attn_res.mlp_queries." + std::to_string(ip)),
                                b_mlp_in, n_blocks, block_stride, M, 1);
            if (has_msgs) {
                const float* mp = b_msgs + 3 * M_stride;
                k_ctrl_cat_rms_sg16(b_mlp_in, mp, b_cat, M);
                dispatch_linear(man.get(p + ".controllers.mlp.down_weight"), b_cat, b_ctrl_gate, M);
                k_ctrl_apply_norm_sg16(b_mlp_in, mp, b_ctrl_gate,
                                       tf32(p + ".controllers.mlp.gain_weight"),
                                       tf32(p + ".controllers.mlp.forget_weight"),
                                       tf32(p + ".mlp_norm.weight"), b_mlp_in, M, 1);
            } else {
                k_rms_norm_sg16(b_mlp_in, tf32(p + ".mlp_norm.weight"), b_mlp_in, D, M);
            }

            dispatch_linear(man.get(p + ".mlp.gate.weight"), b_mlp_in, b_r_logits, M);
            k_router_topk8_sg16(b_r_logits, tf32(p + ".mlp.gate.bias"), b_topk_idx, b_topk_w, c.route_scale, M);

            if (hybrid.active) {
                hybrid.run_moe_hybrid(ip, visit_pass, M,
                                      b_mlp_in, b_topk_idx, b_topk_w,
                                      b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq,
                                      b_moe_h, b_moe_yp, b_moe_raw);
            } else if (xmx_enabled() && M >= 32) {
                const int P = M * 8;
                std::vector<int32_t> h_topk((size_t)P);
                d2h(h_topk.data(), b_topk_idx, (size_t)P * sizeof(int32_t));
                wait();
                std::vector<std::vector<int>> by_exp((size_t)c.n_experts);
                for (int pid = 0; pid < P; ++pid) { const int e = h_topk[pid]; if (e >= 0 && e < c.n_experts) by_exp[e].push_back(pid); }
                std::vector<int> all_pairs, sm_uexp, sm_uoff, sm_ucnt;
                struct XmxItem { int e, off, cnt; };
                std::vector<XmxItem> xmx_items;
                const int xmx_min = xmx_moe_min_n();
                for (int e = 0; e < c.n_experts; ++e) {
                    const int cnt = (int)by_exp[e].size();
                    if (cnt == 0) continue;
                    const int off = (int)all_pairs.size();
                    for (int pid : by_exp[e]) all_pairs.push_back(pid);
                    if (cnt >= xmx_min) xmx_items.push_back({e, off, cnt});
                    else { sm_uexp.push_back(e); sm_uoff.push_back(off); sm_ucnt.push_back(cnt); }
                }
                h2d_sync(b_moe_pairs, all_pairs.data(), all_pairs.size() * sizeof(int));
                const InkSyExpertDev* ex_layer = mod->expert_dev + (size_t)ip * c.n_experts;
                for (const auto& it : xmx_items) {
                    ink_xmx_moe_single(mod->h_ed[(size_t)ip * c.n_experts + it.e], b_mlp_in, b_topk_w, b_moe_pairs, it.off, it.cnt, b_moe_yp);
                }
                if (!sm_uexp.empty()) {
                    const int n_su = (int)sm_uexp.size();
                    h2d_sync(b_moe_uexp, sm_uexp.data(), (size_t)n_su * sizeof(int));
                    h2d_sync(b_moe_uoff, sm_uoff.data(), (size_t)n_su * sizeof(int));
                    h2d_sync(b_moe_ucnt, sm_ucnt.data(), (size_t)n_su * sizeof(int));
                    h2d_sync(b_moe_n_uniq, &n_su, sizeof(int));
                    k_moe_gu_situ_sg16(ex_layer, b_mlp_in, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_su, b_moe_h);
                    k_moe_dn_sg16(ex_layer, b_topk_w, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_su, b_moe_h, b_moe_yp);
                }
                k_moe_reduce8_sg16(b_moe_yp, b_moe_raw, M);
            } else {
                const int P = M * 8;
                const int max_uniq = std::min(128, P);
                k_moe_plan_sg16(b_topk_idx, P, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq);
                const InkSyExpertDev* ex_layer = mod->expert_dev + (size_t)ip * c.n_experts;
                k_moe_gu_situ_sg16(ex_layer, b_mlp_in, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, max_uniq, b_moe_h);
                k_moe_dn_sg16(ex_layer, b_topk_w, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, max_uniq, b_moe_h, b_moe_yp);
                k_moe_reduce8_sg16(b_moe_yp, b_moe_raw, M);
            }

            k_sconv4_multi_sg16(b_moe_raw, sc_layer + InkSyModel::kSconvMlpOff, nullptr, sc_hist_mlp[visit],
                                b_active_slots, b_slot_tok_off, b_slot_tok_cnt, b_slot_tok_idx,
                                b_moe_out, n_active_slots, D, 0, sc_ring_mlp[visit], save_ring);
            k_add_sg16(b_res, b_moe_out, b_cur, M_stride);
            if (dbg) {
                char tb[64];
                std::snprintf(tb, sizeof(tb), "v%d_ip%d_moeh", visit, ip); dbg_buf(tb, b_moe_h, 8 * 1152);
                std::snprintf(tb, sizeof(tb), "v%d_ip%d_cur", visit, ip); dbg_buf(tb, b_cur, D);
            }
        };

        // 2. Prelude (0..5)
        int visit = 0;
        for (int ip = 0; ip < c.core_start; ++ip) run_visit(ip, visit++, 0, false);

        // 3. Looped Core (6..17 x 3 passes)
        d2d(b_prelude, b_cur, (size_t)M_stride * 4);
        prelude_n_blocks = n_blocks;
        for (int p_idx = 0; p_idx < c.num_passes; ++p_idx) {
            int bank_n = 0;
            if (p_idx > 0) {
                bank_n = n_blocks - (prelude_n_blocks + 1);
                for (int bi = 0; bi < bank_n; ++bi) {
                    d2d(b_bank_blk + (size_t)bi * block_stride,
                        b_blocks + (size_t)(prelude_n_blocks + 1 + bi) * block_stride, (size_t)M_stride * 4);
                }
                d2d(b_bank_out, b_cur, (size_t)M_stride * 4);
                d2d(b_cur, b_prelude, (size_t)M_stride * 4);
                n_blocks = prelude_n_blocks;
            }
            for (int ip = c.core_start; ip < c.core_end; ++ip) {
                if (p_idx > 0) {
                    k_attn_res_mix_sg16(b_bank_blk, b_bank_out,
                                        mod->msg_queries_dev + (size_t)(ip - c.core_start) * 4 * D,
                                        b_msgs, bank_n, block_stride, M, 4);
                }
                run_visit(ip, visit++, p_idx, p_idx > 0);
            }
        }

        // 4. Coda (18..23)
        for (int ip = c.core_end; ip < c.n_phys; ++ip) run_visit(ip, visit++, 0, false);

        // 5. Final Output Attention-Residual + model.llm.norm
        k_attn_res_mix_sg16(b_blocks, b_cur, tf32("model.llm.attn_res.output_query"), b_h_out, n_blocks, block_stride, M, 1);
        k_rms_norm_sg16(b_h_out, tf32("model.llm.norm.weight"), b_h_norm, D, M);

        // 6. Optional DSpark taps -> context KV + f_rnn
        if (record_dspark_ctx && use_dspark) {
            d2d(b_taps4 + 3 * M_stride, b_h_norm, (size_t)M_stride * 4);
            const float* fw = df32("dspark.fuse_weight");
            for (int dl = 0; dl < 3; ++dl) {
                std::string dp = "dspark.layers." + std::to_string(dl);
                k_dspark_mix_taps_sg16(b_taps4, fw, df32(dp + ".dflare_norm.weight"), b_mlp_in, dl, M, M);
                dispatch_linear(man.get_d(dp + ".self_attn.k_proj_ctx.weight"), b_mlp_in, b_k_raw, M);
                dispatch_linear(man.get_d(dp + ".self_attn.v_proj_ctx.weight"), b_mlp_in, b_v_raw, M);
                k_dspark_ctx_kv_store_sg16(b_k_raw, b_v_raw, df32(dp + ".self_attn.k_norm_ctx.weight"),
                                           dsp_ctx_k[dl], dsp_ctx_v[dl], b_slot_ids, b_pos,
                                           M, max_ctx, c.dspark_rope_theta);
            }
            k_dspark_mix_taps_sg16(b_taps4, fw, df32("dspark.markov_head.dflare_norm.weight"), dsp_frnn_tmp, 3, M, M);
            // Scatter latest f_rnn per slot on GPU (in ascending token order)
            for (int m = 0; m < M; ++m) {
                d2d(dsp_frnn_slot + (size_t)slot_ids[m] * D, dsp_frnn_tmp + (size_t)m * D, (size_t)D * 4);
            }
        }

        // 7. Unembed + softcap(30.0) + argmax (supports arbitrary want_idx_host rows & >64 rows via chunking!)
        if (logits_mode > 0) {
            const int V_un = c.vocab_unpadded;
            if (logits_mode == 1) {
                const float* x_un = b_h_norm + (size_t)(M - 1) * D;
                dispatch_linear(man.get("model.llm.unembed.weight"), x_un, b_logits, 1, V_un);
                k_softcap_sg16(b_logits, V_un, c.logit_softcap);
                if (out_argmax_host) {
                    k_argmax_rows_sg16(b_logits, b_argmax, V_un, 1);
                    d2h(out_argmax_host, b_argmax, 4);
                }
                if (out_logits_host) {
                    d2h(out_logits_host, b_logits, (size_t)V_un * 4);
                }
            } else {
                const int total_out = (want_idx_host && n_want > 0) ? n_want : M;
                const int logit_cap = std::min(max_batch, 64);
                for (int r0 = 0; r0 < total_out; r0 += logit_cap) {
                    const int rc = std::min(logit_cap, total_out - r0);
                    const float* x_src = b_h_norm + (size_t)r0 * D;
                    if (want_idx_host && n_want > 0) {
                        for (int j = 0; j < rc; ++j) {
                            d2d(b_h_out + (size_t)j * D, b_h_norm + (size_t)want_idx_host[r0 + j] * D, (size_t)D * 4);
                        }
                        x_src = b_h_out;
                    }
                    dispatch_linear(man.get("model.llm.unembed.weight"), x_src, b_logits, rc, V_un);
                    k_softcap_sg16(b_logits, rc * V_un, c.logit_softcap);
                    if (out_argmax_host) {
                        k_argmax_rows_sg16(b_logits, b_argmax, V_un, rc);
                        d2h(out_argmax_host + r0, b_argmax, (size_t)rc * 4);
                    }
                    if (out_logits_host) {
                        d2h(out_logits_host + (size_t)r0 * V_un, b_logits, (size_t)rc * V_un * 4);
                    }
                    wait();
                }
            }
        }

        wait();
        return true;
    }

    // Persistent 100% on-GPU DSpark Block-8 Recurrent Markov drafter!
    // Uses linearity: unembed(cur[s]) + unembed(corr) == unembed(cur[s] + corr) -> ONLY 7 unembed calls (vs 15)!
    int dspark_draft_block8(int slot, int anchor_tok, int anchor_pos, int out_drafts[7]) {
        if (!use_dspark || anchor_pos <= 0 || anchor_pos + 8 >= max_ctx) return 0;
        const auto& man = mod->man;
        const auto& c = man.c;
        const int D = c.hidden;
        const int V_un = c.vocab_unpadded;

        dzero(bd_state, 256 * 4);
        h2d_sync(bd_tok_chain, &anchor_tok, 4);

        k_dspark_init_block_sg16(man.get("model.llm.embed.weight"), tf32("model.llm.embed_norm.weight"),
                                 df32("dspark.mask_embed"), bd_cur, anchor_tok, c.vocab_padded);

        for (int il = 0; il < 3; ++il) {
            std::string lp = "dspark.layers." + std::to_string(il);
            d2d(bd_res, bd_cur, 8 * D * 4);
            k_rms_norm_sg16(bd_cur, df32(lp + ".input_layernorm.weight"), bd_h, D, 8);
            dispatch_linear(man.get_d(lp + ".self_attn.q_proj.weight"), bd_h, bd_q, 8);
            dispatch_linear(man.get_d(lp + ".self_attn.k_proj.weight"), bd_h, bd_k, 8);
            dispatch_linear(man.get_d(lp + ".self_attn.v_proj.weight"), bd_h, bd_v, 8);
            k_dspark_qk_norm_rope_sg16(bd_q, df32(lp + ".self_attn.q_norm.weight"), 24, anchor_pos, c.dspark_rope_theta);
            k_dspark_qk_norm_rope_sg16(bd_k, df32(lp + ".self_attn.k_norm.weight"), 8, anchor_pos, c.dspark_rope_theta);
            k_dspark_block_attn_sg16(bd_q, bd_k, bd_v, dsp_ctx_k[il], dsp_ctx_v[il], bd_ao,
                                     slot, anchor_pos, c.dspark_is_swa[il], c.dspark_swa_win, max_ctx);
            dispatch_linear(man.get_d(lp + ".self_attn.o_proj.weight"), bd_ao, bd_proj, 8);
            k_add_sg16(bd_res, bd_proj, bd_cur, 8 * D);

            d2d(bd_res, bd_cur, 8 * D * 4);
            k_rms_norm_sg16(bd_cur, df32(lp + ".post_attention_layernorm.weight"), bd_h, D, 8);
            dispatch_linear(man.get_d(lp + ".mlp.gate_proj.weight"), bd_h, bd_mg, 8);
            dispatch_linear(man.get_d(lp + ".mlp.up_proj.weight"), bd_h, bd_mu, 8);
            k_dspark_swiglu_sg16(bd_mg, bd_mu, bd_mh, 8 * c.dspark_inter);
            dispatch_linear(man.get_d(lp + ".mlp.down_proj.weight"), bd_mh, bd_proj, 8);
            k_add_sg16(bd_res, bd_proj, bd_cur, 8 * D);
        }

        k_rms_norm_sg16(bd_cur, df32("dspark.norm.weight"), bd_cur, D, 8);

        const float* ctx_vec = dsp_frnn_slot + (size_t)slot * D;
        const auto& mw1 = man.get_d("dspark.markov_head.markov_w1.weight");
        const auto& rnn_w = man.get_d("dspark.markov_head.rnn_proj.weight");
        const float* rnn_b = df32("dspark.markov_head.rnn_proj.bias");
        const auto& ho_w = man.get_d("dspark.markov_head.hidden_out.weight");
        const float* ho_b = df32("dspark.markov_head.hidden_out.bias");
        const auto& unemb = man.get("model.llm.unembed.weight");

        for (int s = 1; s < 8; ++s) {
            k_dspark_markov_joint_sg16(bd_state, mw1, bd_cur, ctx_vec, bd_tok_chain, bd_joint, s, c.vocab_padded);
            dispatch_linear(rnn_w, bd_joint, bd_rproj, 1);
            k_dspark_markov_rnn_update_sg16(bd_rproj, rnn_b, bd_state, bd_latent);
            dispatch_linear(ho_w, bd_latent, bd_corr, 1);
            k_dspark_add_bias_sg16(bd_corr, ho_b, D);
            // Linearity fusion: unembed(cur[s] + corr) == base_logits[s] + bias_logits!
            k_add_sg16(bd_cur + (size_t)s * D, bd_corr, bd_comb, D);
            dispatch_linear(unemb, bd_comb, bd_logits, 1, V_un);
            // Argmax directly on un-softcapped logits (softcap(30*tanh(x/30)) is strictly monotonic!)
            k_argmax_rows_sg16(bd_logits, bd_tok_chain + s, V_un, 1);
        }

        int chain[8]{};
        d2h(chain, bd_tok_chain, sizeof(chain));
        wait();
        for (int i = 0; i < 7; ++i) out_drafts[i] = chain[i + 1];
        return 7;
    }
};

}  // namespace ink_sycl
