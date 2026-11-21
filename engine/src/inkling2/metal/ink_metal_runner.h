// ink_metal_runner.h - High-speed Single-Command-Buffer 48-Visit Metal Runner for Inkling-2-Mini
// Features:
//   - Single command-buffer encoding across all 48 virtual visits (cx.run_graph)
//   - Full support for SQ-20 (SF4+MX8+BF16), SQ-15 (SF4+S3/4/5_K+Q8_ROW+S6_K), SQ-10 (S2/3_K+SF4+Q8_ROW+S6_K)
//   - Continuous multi-request batching for c = 1, 4, 8, 32, 64 concurrent slots
//   - Persistent 100% on-GPU DSpark 3-layer Block-8 Recurrent Markov speculative drafter (0 host syncs across 7 steps!)
//   - Native Metal Vision ViT (Inkling-2-Mini-Vision.strata)
//   - Exact greedy & speculative verification parity + selfcheck
// 100% MIT clean-room implementation.
#pragma once

#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_metal_common.h"
#include "ink_metal_vision.h"

namespace ink_metal {

inline bool InkMetalModel::read_section0_json(int fd, uint64_t off, uint64_t len, std::string& out, std::string& err) {
    if (len == 0) { out.clear(); return true; }
    std::vector<uint8_t> raw(len);
    if (::pread(fd, raw.data(), len, (off_t)off) != (ssize_t)len) {
        err = "short read on Section 0 manifest";
        return false;
    }
    unsigned long long rsize = ZSTD_getFrameContentSize(raw.data(), raw.size());
    if (rsize == ZSTD_CONTENTSIZE_ERROR) {
        out.assign(reinterpret_cast<const char*>(raw.data()), raw.size());
        return true;
    }
    if (rsize != ZSTD_CONTENTSIZE_UNKNOWN) {
        out.resize((size_t)rsize);
        size_t d = ZSTD_decompress(out.data(), out.size(), raw.data(), raw.size());
        if (ZSTD_isError(d)) { err = std::string("zstd: ") + ZSTD_getErrorName(d); return false; }
        out.resize(d);
        return true;
    }
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    ZSTD_inBuffer in_b{raw.data(), raw.size(), 0};
    std::vector<char> chunk(1 << 20);
    out.clear();
    while (in_b.pos < in_b.size) {
        ZSTD_outBuffer out_b{chunk.data(), chunk.size(), 0};
        size_t ret = ZSTD_decompressStream(dctx, &out_b, &in_b);
        if (ZSTD_isError(ret)) {
            err = std::string("zstd stream: ") + ZSTD_getErrorName(ret);
            ZSTD_freeDCtx(dctx);
            return false;
        }
        out.append(chunk.data(), out_b.pos);
    }
    ZSTD_freeDCtx(dctx);
    return true;
}

struct InkBatchStep {
    int M = 0;                          // total tokens across all active slots in this batch
    std::vector<int32_t> slot_ids;      // [M]
    std::vector<int32_t> pos;           // [M]
    std::vector<int32_t> tok_ids;       // [M]
    std::vector<uint8_t> is_img;        // [M] (optional, 1 = use img_embeds)
    std::vector<float> img_embeds;      // [M * 2304] (when any is_img == 1)
    bool capture_taps = false;          // capture visits [5, 23, 41, 48] into DSpark context KV + f_rnn
    bool compute_logits = true;         // compute final unembed + softcap(30.0)
    int logits_last_only_idx = -1;      // >= 0 -> compute unembed ONLY for token index `logits_last_only_idx` (prefill speed!)
    bool record_verify_ring = false;    // save per-token ShortConv states [0..7] for O(1) zero-replay rollback!
};

struct InkMetalRunner {
    mtl::Context* cx = nullptr;
    InkMetalModel* mod = nullptr;
    InkMetalVision vis;

    int lib_codecs = -1, lib_ops = -1, lib_dspark = -1, lib_vis = -1;

    // Pipelines
    mtl::Pipeline p_lin_bf16_mv1, p_lin_bf16_mv, p_lin_bf16_mm8, p_lin_bf16_mm;
    mtl::Pipeline p_lin_q8_mv1, p_lin_q8_mv, p_lin_q8_mm8, p_lin_q8_mm;
    mtl::Pipeline p_lin_f32, p_lin_f32_mm;
    mtl::Pipeline p_lin_sk[9], p_lin_sk6_mm8;
    mtl::Pipeline p_unemb_argmax_q8, p_unemb_argmax_bf16, p_unemb_argmax_sk6, p_argmax_part;
    mtl::Pipeline p_moe_plan, p_moe_gu, p_moe_dn, p_moe_red8;
    mtl::Pipeline p_moe_gu_sf4, p_moe_dn_sf4, p_moe_gu_mx8, p_moe_dn_mx8;
    mtl::Pipeline p_moe_gu_mm_sf4, p_moe_dn_mm_sf4, p_moe_gu_mm_mx8, p_moe_dn_mm_mx8, p_moe_gu_mm_gen, p_moe_dn_mm_gen;
    int layer_homog_gu[24]{};
    int layer_homog_dn[24]{};

    mtl::Pipeline p_emb_bf16, p_emb_q8;
    mtl::Pipeline p_attn_res_mix, p_attn_res_mix_norm, p_ctrl_cat, p_ctrl_apply, p_ctrl_gf_64, p_ctrl_fuse_norm, p_rms_norm, p_add;
    mtl::Pipeline p_sconv4_multi, p_sconv4_attn_res, p_moe_red8_sconv_add, p_sconv4_pf, p_sconv4_pf_commit;
    mtl::Pipeline p_qr_clip_tau, p_kv_store_f16, p_relpos_gqa;
    mtl::Pipeline p_router_topk8, p_router_topk8_plan_m1, p_softcap, p_argmax_rows;

    mtl::Pipeline p_dsp_mix, p_dsp_ctx_kv, p_dsp_init_bf16, p_dsp_init_q8;
    mtl::Pipeline p_dsp_qk_rope, p_dsp_attn, p_dsp_swiglu;
    mtl::Pipeline p_dsp_mjoint_bf16, p_dsp_mjoint_q8, p_dsp_mrnn, p_dsp_add_bias;
    mtl::Pipeline p_dsp_comb_s1, p_dsp_comb_s2;

    // Limits & capacities
    int max_slots = 1;
    int max_ctx = 8192;
    int max_batch = 512;  // max tokens per forward_batch chunk
    bool use_dspark = true;

    // KV Cache & ShortConv State Pools (allocated for max_slots)
    // 40 SWA visits -> ring [max_slots, 1024, 8, 128] FP16 per SWA visit
    // 8 Global visits -> linear [max_slots, max_ctx, 8, 128] FP16 per Global visit
    std::vector<mtl::Buffer> kv_k;        // [48]
    std::vector<mtl::Buffer> kv_v;        // [48]
    std::vector<int> visit_kv_stride;     // [48]: 1024 for SWA, max_ctx for global

    // ShortConv 3-lag history pools per visit: [48]
    std::vector<mtl::Buffer> sc_hist_k;    // [max_slots, 3, 1024]
    std::vector<mtl::Buffer> sc_hist_v;    // [max_slots, 3, 1024]
    std::vector<mtl::Buffer> sc_hist_attn; // [max_slots, 3, 2304]
    std::vector<mtl::Buffer> sc_hist_mlp;  // [max_slots, 3, 2304]

    // Per-step ShortConv verify ring [48] x [8, 3, C] (~30.6 MB total) for O(1) zero-replay rollback!
    std::vector<mtl::Buffer> sc_ring_k;
    std::vector<mtl::Buffer> sc_ring_v;
    std::vector<mtl::Buffer> sc_ring_attn;
    std::vector<mtl::Buffer> sc_ring_mlp;

    // DSpark context KV caches (3 layers) + committed f_rnn [max_slots, max_ctx, 2304] (wait: only last committed f_rnn per slot is needed for decode!
    // Store [max_slots, max_ctx] for ctx_kv, and [max_slots, 2304] for latest f_rnn + backup for rollback!)
    mtl::Buffer dsp_ctx_k[3], dsp_ctx_v[3];
    mtl::Buffer dsp_frnn_slot;   // [max_slots, 2304] latest committed f_rnn (for pos = apos - 1)
    mtl::Buffer dsp_frnn_tmp;    // [max_batch, 2304]

    // Workspace buffers for forward_batch (sized to max_batch)
    mtl::Buffer b_slot_ids, b_pos, b_tok_ids, b_is_img, b_img_emb;
    mtl::Buffer b_active_slots, b_slot_tok_off, b_slot_tok_cnt, b_slot_tok_idx;
    mtl::Buffer b_cur, b_res, b_blocks;    // b_blocks: [6, max_batch, 2304]
    mtl::Buffer b_prelude;                 // [max_batch, 2304]
    mtl::Buffer b_bank_blk, b_bank_out;    // cross-pass bank: 1 block + layer 17 output
    mtl::Buffer b_msgs;                    // [4, max_batch, 2304]
    mtl::Buffer b_in3;                     // [3, max_batch, 2304] (Q/K/V inputs)
    mtl::Buffer b_cat, b_ctrl_gate;        // [max_batch, 4608], [max_batch, 64]
    mtl::Buffer b_q, b_k_raw, b_v_raw, b_k, b_v, b_r, b_kd_taps, b_vd_taps;
    mtl::Buffer b_sinks, b_gate_attn, b_attn_out, b_wo_raw, b_wo;
    mtl::Buffer b_mlp_in, b_r_logits, b_topk_idx, b_topk_w;
    mtl::Buffer b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_meta, b_moe_tiles;
    mtl::Buffer b_moe_h, b_moe_yp, b_moe_raw, b_moe_out;
    mtl::Buffer b_h_out, b_h_norm, b_logits, b_argmax;
    mtl::Buffer b_taps4;                   // [4, max_batch, 2304] for DSpark taps [5, 23, 41, 48]

    // DSpark Block-8 scratch buffers (M=8)
    mtl::Buffer bd_cur, bd_res, bd_h, bd_q, bd_k, bd_v, bd_ao, bd_proj;
    mtl::Buffer bd_mg, bd_mu, bd_mh, bd_logits, bd_comb;
    mtl::Buffer bd_state, bd_joint, bd_rproj, bd_latent, bd_corr;
    mtl::Buffer bd_tok_chain, bd_part_val, bd_part_idx;

    double last_gpu_ms = 0.0;

    bool init(mtl::Context& ctx, InkMetalModel& m, int slots, int ctx_len, int batch_cap,
              bool enable_dspark, const std::string& vision_path,
              const std::string& kernel_dir, std::string& err) {
        cx = &ctx;
        mod = &m;
        max_slots = std::max(1, std::min(64, slots));
        max_ctx = std::max(256, ctx_len);
        max_batch = std::max(64, batch_cap);
        use_dspark = enable_dspark && m.man.c.has_dspark;

        // Compile Metal shader libraries
        if (!compile_shaders(kernel_dir, err)) return false;

        // Create pipelines
        p_lin_bf16_mv1 = cx->pipeline(lib_codecs, "ink_linear_bf16_mv1");
        p_lin_bf16_mv  = cx->pipeline(lib_codecs, "ink_linear_bf16_mv");
        p_lin_bf16_mm8 = cx->pipeline(lib_codecs, "ink_linear_bf16_mm8");
        p_lin_bf16_mm  = cx->pipeline(lib_codecs, "ink_linear_bf16_mm");
        p_lin_q8_mv1   = cx->pipeline(lib_codecs, "ink_linear_q8row_mv1");
        p_lin_q8_mv    = cx->pipeline(lib_codecs, "ink_linear_q8row_mv");
        p_lin_q8_mm8   = cx->pipeline(lib_codecs, "ink_linear_q8row_mm8");
        p_lin_q8_mm    = cx->pipeline(lib_codecs, "ink_linear_q8row_mm");
        p_lin_f32      = cx->pipeline(lib_codecs, "ink_linear_f32");
        p_lin_f32_mm   = cx->pipeline(lib_codecs, "ink_linear_f32_mm");
        p_unemb_argmax_q8   = cx->pipeline(lib_codecs, "ink_unembed_argmax_q8row");
        p_unemb_argmax_bf16 = cx->pipeline(lib_codecs, "ink_unembed_argmax_bf16");
        p_unemb_argmax_sk6  = cx->pipeline(lib_codecs, "ink_unembed_argmax_sk6");
        p_argmax_part       = cx->pipeline(lib_codecs, "ink_argmax_part_reduce");
        for (int f : {2, 3, 4, 5, 6, 8}) {
            p_lin_sk[f] = cx->pipeline(lib_codecs, "ink_linear_sk_mv_" + std::to_string(f));
        }
        p_lin_sk6_mm8   = cx->pipeline(lib_codecs, "ink_linear_sk6_mm8");
        p_moe_plan      = cx->pipeline(lib_codecs, "ink_moe_plan");
        p_moe_gu        = cx->pipeline(lib_codecs, "ink_moe_gu_situ");
        p_moe_dn        = cx->pipeline(lib_codecs, "ink_moe_dn");
        p_moe_gu_sf4    = cx->pipeline(lib_codecs, "ink_moe_gu_sf4");
        p_moe_dn_sf4    = cx->pipeline(lib_codecs, "ink_moe_dn_sf4");
        p_moe_gu_mx8    = cx->pipeline(lib_codecs, "ink_moe_gu_mx8");
        p_moe_dn_mx8    = cx->pipeline(lib_codecs, "ink_moe_dn_mx8");
        p_moe_gu_mm_sf4 = cx->pipeline(lib_codecs, "ink_moe_gu_mm_sf4");
        p_moe_dn_mm_sf4 = cx->pipeline(lib_codecs, "ink_moe_dn_mm_sf4");
        p_moe_gu_mm_mx8 = cx->pipeline(lib_codecs, "ink_moe_gu_mm_mx8");
        p_moe_dn_mm_mx8 = cx->pipeline(lib_codecs, "ink_moe_dn_mm_mx8");
        p_moe_gu_mm_gen = cx->pipeline(lib_codecs, "ink_moe_gu_mm_gen");
        p_moe_dn_mm_gen = cx->pipeline(lib_codecs, "ink_moe_dn_mm_gen");
        p_moe_red8      = cx->pipeline(lib_codecs, "ink_moe_reduce8");

        // Detect homogeneous SF4/MX8 layers for low-register specialized Metal pipelines!
        const auto* ed_host = mod->expert_dev_buf.as<InkExpertDev>();
        for (int il = 0; il < mod->man.c.n_phys; ++il) {
            int g0 = ed_host[il * mod->man.c.n_experts].gu_code;
            int d0 = ed_host[il * mod->man.c.n_experts].dn_code;
            bool hg = true, hd = true;
            for (int e = 1; e < mod->man.c.n_experts; ++e) {
                const auto& ed = ed_host[il * mod->man.c.n_experts + e];
                if (ed.gu_code != g0) hg = false;
                if (ed.dn_code != d0) hd = false;
            }
            layer_homog_gu[il] = hg ? g0 : -1;
            layer_homog_dn[il] = hd ? d0 : -1;
        }

        p_emb_bf16     = cx->pipeline(lib_ops, "ink_embed_bf16");
        p_emb_q8       = cx->pipeline(lib_ops, "ink_embed_q8row");
        p_attn_res_mix        = cx->pipeline(lib_ops, "ink_attn_res_mix");
        p_attn_res_mix_norm   = cx->pipeline(lib_ops, "ink_attn_res_mix_norm");
        p_ctrl_cat            = cx->pipeline(lib_ops, "ink_ctrl_cat_rms");
        p_ctrl_apply          = cx->pipeline(lib_ops, "ink_ctrl_apply_norm");
        p_ctrl_gf_64          = cx->pipeline(lib_ops, "ink_ctrl_gf_64");
        p_ctrl_fuse_norm      = cx->pipeline(lib_ops, "ink_ctrl_fuse_norm");
        p_rms_norm            = cx->pipeline(lib_ops, "ink_rms_norm");
        p_add                 = cx->pipeline(lib_ops, "ink_add");
        p_copy_f32            = cx->pipeline(lib_ops, "ink_copy_f32");
        p_sconv4_multi        = cx->pipeline(lib_ops, "ink_sconv4_multi");
        p_sconv4_attn_res     = cx->pipeline(lib_ops, "ink_sconv4_attn_res");
        p_moe_red8_sconv_add  = cx->pipeline(lib_ops, "ink_moe_red8_sconv_add");
        p_sconv4_pf           = cx->pipeline(lib_ops, "ink_sconv4_prefill_single");
        p_sconv4_pf_commit    = cx->pipeline(lib_ops, "ink_sconv4_prefill_commit");
        p_qr_clip_tau         = cx->pipeline(lib_ops, "ink_qr_clip_tau");
        p_kv_store_f16        = cx->pipeline(lib_ops, "ink_kv_norm_store_f16");
        p_relpos_gqa          = cx->pipeline(lib_ops, "ink_relpos_sink_gqa");
        p_router_topk8        = cx->pipeline(lib_ops, "ink_router_topk8");
        p_router_topk8_plan_m1 = cx->pipeline(lib_ops, "ink_router_topk8_plan_m1");
        p_softcap             = cx->pipeline(lib_ops, "ink_softcap_f32");
        p_argmax_rows         = cx->pipeline(lib_ops, "ink_argmax_rows");

        p_dsp_mix         = cx->pipeline(lib_dspark, "dspark_mix_taps");
        p_dsp_ctx_kv      = cx->pipeline(lib_dspark, "dspark_ctx_kv_store");
        p_dsp_init_bf16   = cx->pipeline(lib_dspark, "dspark_init_block_bf16");
        p_dsp_init_q8     = cx->pipeline(lib_dspark, "dspark_init_block_q8row");
        p_dsp_qk_rope     = cx->pipeline(lib_dspark, "dspark_qk_norm_rope");
        p_dsp_attn        = cx->pipeline(lib_dspark, "dspark_block_attn");
        p_dsp_swiglu      = cx->pipeline(lib_dspark, "dspark_swiglu");
        p_dsp_mjoint_bf16 = cx->pipeline(lib_dspark, "dspark_markov_joint_bf16");
        p_dsp_mjoint_q8   = cx->pipeline(lib_dspark, "dspark_markov_joint_q8row");
        p_dsp_mrnn        = cx->pipeline(lib_dspark, "dspark_markov_rnn_update");
        p_dsp_add_bias    = cx->pipeline(lib_dspark, "dspark_add_bias");
        p_dsp_comb_s1     = cx->pipeline(lib_dspark, "dspark_combine_argmax_s1");
        p_dsp_comb_s2     = cx->pipeline(lib_dspark, "dspark_combine_argmax_s2");

        // Allocate KV caches & ShortConv state pools
        const auto& c = mod->man.c;
        kv_k.resize(c.n_visits);
        kv_v.resize(c.n_visits);
        visit_kv_stride.resize(c.n_visits);
        sc_hist_k.resize(c.n_visits);
        sc_hist_v.resize(c.n_visits);
        sc_hist_attn.resize(c.n_visits);
        sc_hist_mlp.resize(c.n_visits);
        sc_ring_k.resize(c.n_visits);
        sc_ring_v.resize(c.n_visits);
        sc_ring_attn.resize(c.n_visits);
        sc_ring_mlp.resize(c.n_visits);

        size_t kv_bytes_tot = 0, sc_bytes_tot = 0;
        for (int v = 0; v < c.n_visits; ++v) {
            const int ip = c.phys_for_visit(v);
            const int stride = c.is_swa[ip] ? c.swa_win : max_ctx;
            visit_kv_stride[v] = stride;
            const size_t kvb = (size_t)max_slots * stride * c.n_kv * c.head_dim * sizeof(uint16_t);
            kv_k[v] = cx->alloc(kvb);
            kv_v[v] = cx->alloc(kvb);
            kv_bytes_tot += 2 * kvb;

            const size_t sc_kv = (size_t)max_slots * 3 * (c.n_kv * c.head_dim) * sizeof(float);
            const size_t sc_d  = (size_t)max_slots * 3 * c.hidden * sizeof(float);
            sc_hist_k[v]    = cx->alloc(sc_kv);
            sc_hist_v[v]    = cx->alloc(sc_kv);
            sc_hist_attn[v] = cx->alloc(sc_d);
            sc_hist_mlp[v]  = cx->alloc(sc_d);
            sc_ring_k[v]    = cx->alloc(8ul * 3 * (c.n_kv * c.head_dim) * sizeof(float));
            sc_ring_v[v]    = cx->alloc(8ul * 3 * (c.n_kv * c.head_dim) * sizeof(float));
            sc_ring_attn[v] = cx->alloc(8ul * 3 * c.hidden * sizeof(float));
            sc_ring_mlp[v]  = cx->alloc(8ul * 3 * c.hidden * sizeof(float));
            sc_bytes_tot += 2 * sc_kv + 2 * sc_d;
        }

        // Workspace buffers
        const size_t D = (size_t)c.hidden;
        const size_t MB = (size_t)max_batch;
        b_slot_ids     = cx->alloc(MB * 4);
        b_pos          = cx->alloc(MB * 4);
        b_tok_ids      = cx->alloc(MB * 4);
        b_is_img       = cx->alloc(MB);
        b_img_emb      = cx->alloc(MB * D * 4);
        b_active_slots = cx->alloc((size_t)max_slots * 4);
        b_slot_tok_off = cx->alloc((size_t)max_slots * 4);
        b_slot_tok_cnt = cx->alloc((size_t)max_slots * 4);
        b_slot_tok_idx = cx->alloc(MB * 4);

        b_cur       = cx->alloc(MB * D * 4);
        b_res       = cx->alloc(MB * D * 4);
        b_blocks    = cx->alloc(6 * MB * D * 4);
        b_prelude   = cx->alloc(MB * D * 4);
        b_bank_blk  = cx->alloc(2 * MB * D * 4);
        b_bank_out  = cx->alloc(MB * D * 4);
        b_msgs      = cx->alloc(4 * MB * D * 4);
        b_in3       = cx->alloc(3 * MB * D * 4);
        b_cat       = cx->alloc(3 * MB * 2 * D * 4);
        b_ctrl_gate = cx->alloc(3 * MB * c.gate_rank * 4);

        b_q         = cx->alloc(MB * (c.n_head * c.head_dim) * 4);
        b_k_raw     = cx->alloc(MB * (c.n_kv * c.head_dim) * 4);
        b_v_raw     = cx->alloc(MB * (c.n_kv * c.head_dim) * 4);
        b_k         = cx->alloc(MB * (c.n_kv * c.head_dim) * 4);
        b_v         = cx->alloc(MB * (c.n_kv * c.head_dim) * 4);
        b_r         = cx->alloc(MB * (c.n_head * c.d_rel) * 4);
        b_kd_taps   = cx->alloc(MB * 32 * 4);
        b_vd_taps   = cx->alloc(MB * 32 * 4);
        b_sinks     = cx->alloc(MB * c.n_head * 4);
        b_gate_attn = cx->alloc(MB * (c.n_head * c.head_dim) * 4);
        b_attn_out  = cx->alloc(MB * (c.n_head * c.head_dim) * 4);
        b_wo_raw    = cx->alloc(MB * D * 4);
        b_wo        = cx->alloc(MB * D * 4);

        b_mlp_in    = cx->alloc(MB * D * 4);
        b_r_logits  = cx->alloc(MB * c.n_experts * 4);
        b_topk_idx  = cx->alloc(MB * c.top_k * 4);
        b_topk_w    = cx->alloc(MB * c.top_k * 4);
        b_moe_uexp  = cx->alloc(c.n_experts * 4);
        b_moe_uoff  = cx->alloc(c.n_experts * 4);
        b_moe_ucnt  = cx->alloc(c.n_experts * 4);
        b_moe_pairs = cx->alloc(MB * c.top_k * 4);
        b_moe_meta  = cx->alloc(16 * 4);
        b_moe_tiles = cx->alloc((MB * c.top_k / 32 + 256) * 8);
        b_moe_h     = cx->alloc(MB * c.top_k * c.inter * 4);
        b_moe_yp    = cx->alloc(MB * c.top_k * D * 4);
        b_moe_raw   = cx->alloc(MB * D * 4);
        b_moe_out   = cx->alloc(MB * D * 4);

        b_h_out     = cx->alloc(MB * D * 4);
        b_h_norm    = cx->alloc(MB * D * 4);
        // Logits buffer: up to min(max_batch, 64) rows simultaneously (prefill uses last-token-only mode!)
        b_logits    = cx->alloc((size_t)std::min(max_batch, 64) * c.vocab_padded * 4);
        b_argmax    = cx->alloc(MB * 4);
        bd_part_val = cx->alloc(16384 * 4);
        bd_part_idx = cx->alloc(16384 * 4);
        b_taps4     = cx->alloc(4 * MB * D * 4);

        if (use_dspark) {
            const size_t dkv = (size_t)max_slots * max_ctx * c.n_kv * c.head_dim * sizeof(uint16_t);
            for (int l = 0; l < 3; ++l) {
                dsp_ctx_k[l] = cx->alloc(dkv);
                dsp_ctx_v[l] = cx->alloc(dkv);
            }
            dsp_frnn_slot = cx->alloc((size_t)max_slots * D * 4);
            dsp_frnn_tmp  = cx->alloc(MB * D * 4);

            bd_cur         = cx->alloc(8 * D * 4);
            bd_res         = cx->alloc(8 * D * 4);
            bd_h           = cx->alloc(8 * D * 4);
            bd_q           = cx->alloc(8 * 3072 * 4);
            bd_k           = cx->alloc(8 * 1024 * 4);
            bd_v           = cx->alloc(8 * 1024 * 4);
            bd_ao          = cx->alloc(8 * 3072 * 4);
            bd_proj        = cx->alloc(8 * D * 4);
            bd_mg          = cx->alloc(8 * 9216 * 4);
            bd_mu          = cx->alloc(8 * 9216 * 4);
            bd_mh          = cx->alloc(8 * 9216 * 4);
            bd_logits      = cx->alloc(c.vocab_padded * 4);
            bd_comb        = cx->alloc(D * 4);
            bd_state       = cx->alloc(256 * 4);
            bd_joint       = cx->alloc(5120 * 4);
            bd_rproj       = cx->alloc(768 * 4);
            bd_latent      = cx->alloc(256 * 4);
            bd_corr        = cx->alloc(D * 4);
            bd_tok_chain   = cx->alloc(8 * 4);
        }

        if (!vision_path.empty()) {
            if (!vis.load(*cx, lib_codecs, lib_vis, vision_path, &InkMetalModel::read_section0_json, err)) {
                return false;
            }
        }

        fprintf(stderr,
                "[ink_metal] runner ready: slots=%d ctx=%d batch=%d dspark=%d vision=%d | KV=%.1f MiB SC=%.1f MiB TotalAlloc=%.2f GiB\n",
                max_slots, max_ctx, max_batch, (int)use_dspark, (int)vis.loaded,
                kv_bytes_tot / (1024.0 * 1024.0), sc_bytes_tot / (1024.0 * 1024.0),
                cx->allocated() / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    // Reset ShortConv state for a slot (called before starting a fresh sequence in slot `s`)
    void reset_slot(int s) {
        if (s < 0 || s >= max_slots) return;
        const auto& c = mod->man.c;
        const size_t sc_kv = 3ul * (c.n_kv * c.head_dim) * sizeof(float);
        const size_t sc_d  = 3ul * c.hidden * sizeof(float);
        for (int v = 0; v < c.n_visits; ++v) {
            std::memset(sc_hist_k[v].as<uint8_t>() + (size_t)s * sc_kv, 0, sc_kv);
            std::memset(sc_hist_v[v].as<uint8_t>() + (size_t)s * sc_kv, 0, sc_kv);
            std::memset(sc_hist_attn[v].as<uint8_t>() + (size_t)s * sc_d, 0, sc_d);
            std::memset(sc_hist_mlp[v].as<uint8_t>() + (size_t)s * sc_d, 0, sc_d);
        }
    }

    // Snapshot / restore ShortConv states for a slot (needed for lossless speculative rollback!)
    struct SlotSconvSnap {
        std::vector<float> k, v, attn, mlp;
    };
    void snapshot_slot(int s, SlotSconvSnap& snap) const {
        const auto& c = mod->man.c;
        const size_t nkv = 3ul * (c.n_kv * c.head_dim);
        const size_t nd  = 3ul * c.hidden;
        snap.k.resize(c.n_visits * nkv);
        snap.v.resize(c.n_visits * nkv);
        snap.attn.resize(c.n_visits * nd);
        snap.mlp.resize(c.n_visits * nd);
        for (int v = 0; v < c.n_visits; ++v) {
            std::memcpy(snap.k.data() + v * nkv, sc_hist_k[v].as<float>() + s * nkv, nkv * 4);
            std::memcpy(snap.v.data() + v * nkv, sc_hist_v[v].as<float>() + s * nkv, nkv * 4);
            std::memcpy(snap.attn.data() + v * nd, sc_hist_attn[v].as<float>() + s * nd, nd * 4);
            std::memcpy(snap.mlp.data() + v * nd, sc_hist_mlp[v].as<float>() + s * nd, nd * 4);
        }
    }
    void restore_slot(int s, const SlotSconvSnap& snap) {
        const auto& c = mod->man.c;
        const size_t nkv = 3ul * (c.n_kv * c.head_dim);
        const size_t nd  = 3ul * c.hidden;
        for (int v = 0; v < c.n_visits; ++v) {
            std::memcpy(sc_hist_k[v].as<float>() + s * nkv, snap.k.data() + v * nkv, nkv * 4);
            std::memcpy(sc_hist_v[v].as<float>() + s * nkv, snap.v.data() + v * nkv, nkv * 4);
            std::memcpy(sc_hist_attn[v].as<float>() + s * nd, snap.attn.data() + v * nd, nd * 4);
            std::memcpy(sc_hist_mlp[v].as<float>() + s * nd, snap.mlp.data() + v * nd, nd * 4);
        }
    }

    // O(1) ZERO-REPLAY speculative rollback: restores exact post-token-`step_idx` ShortConv & f_rnn state!
    void restore_verify_step(int s, int step_idx) {
        if (step_idx < 0 || step_idx >= 8) return;
        const auto& c = mod->man.c;
        const size_t nkv = 3ul * (c.n_kv * c.head_dim);
        const size_t nd  = 3ul * c.hidden;
        for (int v = 0; v < c.n_visits; ++v) {
            std::memcpy(sc_hist_k[v].as<float>() + (size_t)s * nkv, sc_ring_k[v].as<float>() + (size_t)step_idx * nkv, nkv * 4);
            std::memcpy(sc_hist_v[v].as<float>() + (size_t)s * nkv, sc_ring_v[v].as<float>() + (size_t)step_idx * nkv, nkv * 4);
            std::memcpy(sc_hist_attn[v].as<float>() + (size_t)s * nd, sc_ring_attn[v].as<float>() + (size_t)step_idx * nd, nd * 4);
            std::memcpy(sc_hist_mlp[v].as<float>() + (size_t)s * nd, sc_ring_mlp[v].as<float>() + (size_t)step_idx * nd, nd * 4);
        }
        if (use_dspark) {
            const size_t D = (size_t)c.hidden;
            std::memcpy(dsp_frnn_slot.as<float>() + (size_t)s * D, dsp_frnn_tmp.as<float>() + (size_t)step_idx * D, D * 4);
        }
    }

    // Helper: push linear dispatch (auto-dispatches BF16/Q8_ROW/F32/S_K across mv1, mm8, and mm32x64!)
    void push_linear(std::vector<mtl::Dispatch>& ds, const InkTensorInfo& ti, const mtl::Arg& w_arg,
                     const mtl::Arg& x_arg, const mtl::Arg& y_arg, int N, int K, int M,
                     bool barrier = true, bool exact_f32 = false) {
        mtl::Dispatch d;
        d.barrier = barrier;
        if (ti.fmt == FMT_BF16) {
            if (M == 1) {
                d.p = &p_lin_bf16_mv1;
                d.grid[0] = (uint32_t)((N + 31) / 32);
                d.tg[0] = 256;
            } else if (M >= 6 && M <= 8) {
                d.p = &p_lin_bf16_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else if (M > 8) {
                d.p = &p_lin_bf16_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 63) / 64);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_bf16_mv;
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            }
        } else if (ti.fmt == FMT_Q8ROW) {
            if (M == 1 && (K & 15) == 0) {
                d.p = &p_lin_q8_mv1;
                d.grid[0] = (uint32_t)((N + 31) / 32);
                d.tg[0] = 256;
            } else if (M >= 6 && M <= 8) {
                d.p = &p_lin_q8_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else if (M > 8) {
                d.p = &p_lin_q8_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 31) / 32);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_q8_mv;
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            }
        } else if (ti.fmt == FMT_F32) {
            if (M > 8 && !exact_f32) {
                d.p = &p_lin_f32_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 31) / 32);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_f32;
                d.grid[0] = (uint32_t)((N + 3) / 4);
                d.tg[0] = 128;
            }
        } else if (ti.fmt >= FMT_S2 && ti.fmt <= FMT_S8) {
            if (ti.fmt == FMT_S6 && M >= 2 && M <= 8) {
                d.p = &p_lin_sk6_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_sk[ti.fmt];
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            }
        } else {
            fprintf(stderr, "[ink_metal] unsupported linear fmt %d for %s\n", ti.fmt, ti.name.c_str());
            exit(2);
        }
        d.args = {w_arg, x_arg, y_arg, mtl::Arg::V(N), mtl::Arg::V(K), mtl::Arg::V(M)};
        ds.push_back(std::move(d));
    }

    void push_trunk_linear(std::vector<mtl::Dispatch>& ds, const std::string& name,
                           const mtl::Arg& x_arg, const mtl::Arg& y_arg, int M, int N_override = -1,
                           bool barrier = true, bool exact_f32 = false) {
        const auto& ti = mod->man.get(name);
        const int N = N_override > 0 ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        push_linear(ds, ti, mod->tw(name), x_arg, y_arg, N, K, M, barrier, exact_f32);
    }

    void push_dspark_linear(std::vector<mtl::Dispatch>& ds, const std::string& name,
                            const mtl::Arg& x_arg, const mtl::Arg& y_arg, int M, bool barrier = true) {
        const auto& ti = mod->man.get_d(name);
        push_linear(ds, ti, mod->dw(name), x_arg, y_arg, (int)ti.rows(), (int)ti.cols(), M, barrier);
    }

    // Execute ONE complete 48-visit Inkling-2-Mini forward pass for batch `step` in a SINGLE Metal command buffer!
    bool forward_batch(const InkBatchStep& step, float* logits_out, int32_t* argmax_out, std::string& err) {
        const int M = step.M;
        if (M <= 0 || M > max_batch) { err = "invalid batch M=" + std::to_string(M); return false; }
        const auto& c = mod->man.c;
        const int D = c.hidden;
        const int block_stride = max_batch * D;
        const int M_stride = M * D;

        // Upload batch metadata + build per-slot token index lists for race-free multi-slot ShortConv
        std::memcpy(b_slot_ids.ptr, step.slot_ids.data(), (size_t)M * 4);
        std::memcpy(b_pos.ptr, step.pos.data(), (size_t)M * 4);
        std::memcpy(b_tok_ids.ptr, step.tok_ids.data(), (size_t)M * 4);
        if (!step.is_img.empty()) {
            std::memcpy(b_is_img.ptr, step.is_img.data(), (size_t)M);
            if (!step.img_embeds.empty()) {
                std::memcpy(b_img_emb.ptr, step.img_embeds.data(), (size_t)M * D * 4);
            }
        } else {
            std::memset(b_is_img.ptr, 0, (size_t)M);
        }

        // Group batch token indices by slot (preserving ascending time order within each slot)
        std::unordered_map<int, std::vector<int>> by_slot;
        for (int m = 0; m < M; ++m) by_slot[step.slot_ids[m]].push_back(m);
        int n_active_slots = 0, tok_cursor = 0;
        int* as_p = b_active_slots.as<int>();
        int* off_p = b_slot_tok_off.as<int>();
        int* cnt_p = b_slot_tok_cnt.as<int>();
        int* idx_p = b_slot_tok_idx.as<int>();
        for (const auto& kv : by_slot) {
            as_p[n_active_slots] = kv.first;
            off_p[n_active_slots] = tok_cursor;
            cnt_p[n_active_slots] = (int)kv.second.size();
            for (int m : kv.second) idx_p[tok_cursor++] = m;
            ++n_active_slots;
        }

        std::vector<mtl::Dispatch> ds;
        ds.reserve(2400);

        auto copy_buf = [&](const mtl::Buffer& src, size_t src_off, const mtl::Buffer& dst, size_t dst_off, int n_floats) {
            // Simple copy via ink_add or direct? Use a tiny zero-overhead kernel or copy through ink_rms?
            // Better: add a 1-line ink_copy or use ink_add with zero? Let's encode copy using ink_add? No, we need exact copy!
            // Actually, let's check: we can do exact copy with buffer offsets directly in the next kernel OR a tiny copy kernel!
            (void)src; (void)src_off; (void)dst; (void)dst_off; (void)n_floats;
        };
        (void)copy_buf;

        // 1. Embedding + embed_norm (bypassing embed_norm for Vision image tokens!)
        {
            const auto& eti = mod->man.get("model.llm.embed.weight");
            mtl::Dispatch d;
            d.grid[0] = (uint32_t)M;
            d.tg[0] = 256;
            if (eti.fmt == FMT_BF16) {
                d.p = &p_emb_bf16;
                d.args = {mod->tw("model.llm.embed.weight"), mod->tw("model.llm.embed_norm.weight"),
                          mtl::Arg::B(b_tok_ids), mtl::Arg::B(b_is_img), mtl::Arg::B(b_img_emb), mtl::Arg::B(b_cur)};
            } else {
                d.p = &p_emb_q8;
                d.args = {mod->tw("model.llm.embed.weight"), mod->tw("model.llm.embed_norm.weight"),
                          mtl::Arg::B(b_tok_ids), mtl::Arg::B(b_is_img), mtl::Arg::B(b_img_emb), mtl::Arg::B(b_cur),
                          mtl::Arg::V(c.vocab_padded)};
            }
            ds.push_back(std::move(d));
        }

        // Helper: GPU copy M*D floats from src_arg to dst_arg using a 2-op-free dedicated dispatch (ink_copy -> wait, let's use ink_add? No, let's track buffer pointers directly or add ink_copy in ops.metal!)
        // Wait: we can track logical buffer bindings AND only copy when saving to b_blocks[slot] / b_prelude / b_taps4!
        // Let's add `ink_copy` to `ink_metal_ops.metal` (4 lines) so saving block snapshots & DSpark taps is 1 trivial dispatch!

        // Helper for 1 physical layer visit (ip, visit, pass_idx)
        int n_blocks = 0;
        int prelude_n_blocks = 0;

        auto push_copy = [&](const mtl::Arg& src, const mtl::Arg& dst, int n_floats) {
            mtl::Dispatch d;
            d.p = &p_copy_f32;
            d.grid[0] = (uint32_t)((n_floats + 255) / 256);
            d.tg[0] = 256;
            d.args = {src, dst, mtl::Arg::V(n_floats)};
            ds.push_back(std::move(d));
        };

        auto run_visit = [&](int ip, int visit, int pass_idx, bool has_msgs) {
            // Capture DSpark tap if visit in {5, 23, 41}
            if (step.capture_taps && use_dspark) {
                for (int ti = 0; ti < 3; ++ti) {
                    if (visit == c.dspark_taps[ti]) {
                        push_copy(mtl::Arg::B(b_cur), mtl::Arg::B(b_taps4, (size_t)ti * M_stride * 4), M_stride);
                    }
                }
            }

            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = (c.is_swa[ip] != 0);
            const int save_ring = (step.record_verify_ring && n_active_slots == 1 && M <= 8) ? 1 : 0;

            // Attention-Residual mix + optional 3-way concurrent loop controllers + input RMSNorm
            // Fused attn_res_mix_norm eliminates ~90 barriers/token when !has_msgs!
            if (!has_msgs) {
                if (ip == 0) {
                    mtl::Dispatch d;
                    d.p = &p_rms_norm; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_cur), mod->tw(p + ".attn_norm.weight"), mtl::Arg::B(b_in3), mtl::Arg::V(D)};
                    ds.push_back(std::move(d));
                } else {
                    const int nq = is_swa ? 1 : 3;
                    mtl::Dispatch d;
                    d.p = &p_attn_res_mix_norm; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    mtl::Arg nw0 = is_swa ? mod->tw(p + ".attn_norm.weight") : mod->tw(p + ".q_input_norm.weight");
                    mtl::Arg nw1 = is_swa ? nw0 : mod->tw(p + ".k_input_norm.weight");
                    mtl::Arg nw2 = is_swa ? nw0 : mod->tw(p + ".v_input_norm.weight");
                    d.args = {mtl::Arg::B(b_blocks), mtl::Arg::B(b_cur),
                              mod->tw("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)),
                              nw0, nw1, nw2, mtl::Arg::B(b_in3),
                              mtl::Arg::V(n_blocks), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride), mtl::Arg::V(nq)};
                    ds.push_back(std::move(d));
                }
            } else {
                const int nq = is_swa ? 1 : 3;
                {
                    // Runs CONCURRENTLY with the cross-pass b_msgs attn_res_mix (disjoint inputs & outputs -> -24 barriers!)
                    mtl::Dispatch d;
                    d.barrier = false;
                    d.p = &p_attn_res_mix; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    int boff = 0;
                    d.args = {mtl::Arg::B(b_blocks), mtl::Arg::B(b_cur),
                              mod->tw("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)),
                              mtl::Arg::B(b_in3),
                              mtl::Arg::V(n_blocks), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride),
                              mtl::Arg::V(nq), mtl::Arg::V(boff)};
                    ds.push_back(std::move(d));
                }
                // Run all 3 (attn, k, v) controllers CONCURRENTLY in 3 barriers instead of 9 (saves 144+ barriers/tok)!
                const char* cnames[3] = {"attn", "k", "v"};
                const char* nnames[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
                for (int j = 0; j < 3; ++j) {
                    const size_t src_off = is_swa ? 0 : ((size_t)j * M_stride * 4);
                    const size_t moff = (size_t)j * M_stride * 4;
                    const size_t coff = (size_t)j * (size_t)M * 2 * D * 4;
                    mtl::Dispatch d;
                    d.barrier = (j == 0);
                    d.p = &p_ctrl_cat; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_in3, src_off), mtl::Arg::B(b_msgs, moff), mtl::Arg::B(b_cat, coff)};
                    ds.push_back(std::move(d));
                }
                for (int j = 0; j < 3; ++j) {
                    const size_t coff = (size_t)j * (size_t)M * 2 * D * 4;
                    const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                    push_trunk_linear(ds, p + ".controllers." + cnames[j] + ".down_weight",
                                      mtl::Arg::B(b_cat, coff), mtl::Arg::B(b_ctrl_gate, goff), M, -1, (j == 0));
                }
                if (M > 8) {
                    // Run ALL 3 (attn, k, v) gain & forget GEMMs CONCURRENTLY into b_cat[0..5], then 3-way concurrent fuse_norm (-96 barriers)!
                    for (int j = 0; j < 3; ++j) {
                        const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                        const size_t sg_off = (size_t)(2 * j + 0) * M_stride * 4;
                        const size_t sf_off = (size_t)(2 * j + 1) * M_stride * 4;
                        push_trunk_linear(ds, p + ".controllers." + cnames[j] + ".gain_weight",
                                          mtl::Arg::B(b_ctrl_gate, goff), mtl::Arg::B(b_cat, sg_off), M, -1, (j == 0));
                        push_trunk_linear(ds, p + ".controllers." + cnames[j] + ".forget_weight",
                                          mtl::Arg::B(b_ctrl_gate, goff), mtl::Arg::B(b_cat, sf_off), M, -1, false);
                    }
                    const int order[3] = {1, 2, 0};
                    for (int oi = 0; oi < 3; ++oi) {
                        const int j = is_swa ? order[oi] : oi;
                        const size_t src_off = is_swa ? 0 : ((size_t)j * M_stride * 4);
                        const size_t dst_off = (size_t)j * M_stride * 4;
                        const size_t moff = (size_t)j * M_stride * 4;
                        const size_t sg_off = (size_t)(2 * j + 0) * M_stride * 4;
                        const size_t sf_off = (size_t)(2 * j + 1) * M_stride * 4;
                        std::string nkey = is_swa ? (p + ".attn_norm.weight") : (p + "." + nnames[j]);
                        mtl::Dispatch df;
                        df.barrier = (oi == 0) || (is_swa && oi == 2);
                        df.p = &p_ctrl_fuse_norm; df.grid[0] = (uint32_t)M; df.tg[0] = 256;
                        df.args = {mtl::Arg::B(b_in3, src_off), mtl::Arg::B(b_msgs, moff),
                                   mtl::Arg::B(b_cat, sg_off), mtl::Arg::B(b_cat, sf_off),
                                   mod->tw(nkey), mtl::Arg::B(b_in3, dst_off)};
                        ds.push_back(std::move(df));
                    }
                } else {
                    // 40-core parallel ink_ctrl_gf_64 (144 TGs each!) writing to scratch b_cat[0..2] -> ALL 3 run 100% CONCURRENTLY even for is_swa (-12 barriers!)
                    for (int j = 0; j < 3; ++j) {
                        const size_t src_off = is_swa ? 0 : ((size_t)j * M_stride * 4);
                        const size_t dst_off = (size_t)j * M_stride * 4;
                        const size_t moff = (size_t)j * M_stride * 4;
                        const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                        mtl::Dispatch d;
                        d.barrier = (j == 0);
                        d.p = &p_ctrl_gf_64; d.grid[0] = 144; d.grid[1] = (uint32_t)M; d.tg[0] = 256;
                        d.args = {mtl::Arg::B(b_in3, src_off), mtl::Arg::B(b_msgs, moff), mtl::Arg::B(b_ctrl_gate, goff),
                                  mod->tw(p + ".controllers." + cnames[j] + ".gain_weight"),
                                  mod->tw(p + ".controllers." + cnames[j] + ".forget_weight"),
                                  mtl::Arg::B(b_cat, dst_off)};
                        ds.push_back(std::move(d));
                    }
                    for (int j = 0; j < 3; ++j) {
                        const size_t dst_off = (size_t)j * M_stride * 4;
                        std::string nkey = is_swa ? (p + ".attn_norm.weight") : (p + "." + nnames[j]);
                        mtl::Dispatch dn;
                        dn.barrier = (j == 0);
                        dn.p = &p_rms_norm; dn.grid[0] = (uint32_t)M; dn.tg[0] = 256;
                        dn.args = {mtl::Arg::B(b_cat, dst_off), mod->tw(nkey), mtl::Arg::B(b_in3, dst_off), mtl::Arg::V(D)};
                        ds.push_back(std::move(dn));
                    }
                }
            }

            // Inputs for Q, K, V: when SWA and !has_msgs, all three share slot 0 of b_in3!
            const mtl::Arg in_q = mtl::Arg::B(b_in3, 0);
            const mtl::Arg in_k = (is_swa && !has_msgs) ? mtl::Arg::B(b_in3, 0) : mtl::Arg::B(b_in3, (size_t)M_stride * 4);
            const mtl::Arg in_v = (is_swa && !has_msgs) ? mtl::Arg::B(b_in3, 0) : mtl::Arg::B(b_in3, (size_t)2 * M_stride * 4);

            // Q, K, V, R, Sinks, GateAttn, k_sconv.weight_proj, v_sconv.weight_proj:
            // ALL 6-8 projections read from b_in3 and write disjoint outputs -> run 100% CONCURRENTLY on Apple Silicon!
            push_trunk_linear(ds, p + ".attn.wq_du.weight", in_q, mtl::Arg::B(b_q), M, -1, true);
            push_trunk_linear(ds, p + ".attn.wk_dv.weight", in_k, mtl::Arg::B(b_k_raw), M, -1, false);
            push_trunk_linear(ds, p + ".attn.wv_dv.weight", in_v, mtl::Arg::B(b_v_raw), M, -1, false);
            push_trunk_linear(ds, p + ".attn.wr_du.weight", in_q, mtl::Arg::B(b_r), M, -1, false);
            push_trunk_linear(ds, p + ".attn.attn_sink_proj.weight", in_q, mtl::Arg::B(b_sinks), M, -1, false);
            push_trunk_linear(ds, p + ".attn.gate_attn.weight", in_q, mtl::Arg::B(b_gate_attn), M, -1, false);

            if (!is_swa) {
                push_trunk_linear(ds, p + ".attn.k_sconv.weight_proj", in_k, mtl::Arg::B(b_kd_taps), M, -1, false);
                push_trunk_linear(ds, p + ".attn.v_sconv.weight_proj", in_v, mtl::Arg::B(b_vd_taps), M, -1, false);
            }

            // Helper: 4-tap causal ShortConv (parallel across M*C for single-slot prefill M >= 16!)
            auto push_sconv = [&](const mtl::Buffer& xin, size_t sc_off, const mtl::Buffer& dyn,
                                  const mtl::Buffer& hist, const mtl::Buffer& ring, const mtl::Buffer& yout,
                                  int C_ch, int has_dyn, bool first_barrier) {
                if (n_active_slots == 1 && M >= 16 && !save_ring) {
                    int s0 = as_p[0];
                    mtl::Dispatch d1;
                    d1.barrier = first_barrier;
                    d1.p = &p_sconv4_pf;
                    d1.grid[0] = (uint32_t)M; d1.grid[1] = (uint32_t)((C_ch + 255) / 256); d1.tg[0] = 256;
                    d1.args = {mtl::Arg::B(xin), mtl::Arg::B(mod->static_sconv_buf, sc_off * 4),
                               mtl::Arg::B(dyn), mtl::Arg::B(hist), mtl::Arg::B(yout),
                               mtl::Arg::V(s0), mtl::Arg::V(M), mtl::Arg::V(C_ch), mtl::Arg::V(has_dyn)};
                    ds.push_back(std::move(d1));
                    mtl::Dispatch d2;
                    d2.barrier = true;
                    d2.p = &p_sconv4_pf_commit;
                    d2.grid[0] = (uint32_t)((C_ch + 255) / 256); d2.tg[0] = 256;
                    d2.args = {mtl::Arg::B(xin), mtl::Arg::B(hist), mtl::Arg::V(s0), mtl::Arg::V(M), mtl::Arg::V(C_ch)};
                    ds.push_back(std::move(d2));
                } else {
                    mtl::Dispatch d;
                    d.barrier = first_barrier;
                    d.p = &p_sconv4_multi;
                    d.grid[0] = (uint32_t)n_active_slots; d.grid[1] = (uint32_t)((C_ch + 255) / 256); d.tg[0] = 256;
                    d.args = {mtl::Arg::B(xin), mtl::Arg::B(mod->static_sconv_buf, sc_off * 4),
                              mtl::Arg::B(dyn), mtl::Arg::B(hist),
                              mtl::Arg::B(b_active_slots), mtl::Arg::B(b_slot_tok_off), mtl::Arg::B(b_slot_tok_cnt), mtl::Arg::B(b_slot_tok_idx),
                              mtl::Arg::B(yout), mtl::Arg::V(C_ch), mtl::Arg::V(has_dyn),
                              mtl::Arg::B(ring), mtl::Arg::V(save_ring)};
                    ds.push_back(std::move(d));
                }
            };

            // 4-tap causal ShortConv on K & V + Q/R L2 clip & tau (ALL 3 run CONCURRENTLY -> saves 48 barriers!)
            const size_t sc_lbase = (size_t)ip * InkMetalModel::kSconvLayerFloats;
            {
                int C_kv = 1024, has_dyn = is_swa ? 0 : 1, swa_i = is_swa ? 1 : 0;
                push_sconv(b_k_raw, sc_lbase + InkMetalModel::kSconvKOff, b_kd_taps, sc_hist_k[visit], sc_ring_k[visit], b_k, C_kv, has_dyn, true);
                push_sconv(b_v_raw, sc_lbase + InkMetalModel::kSconvVOff, b_vd_taps, sc_hist_v[visit], sc_ring_v[visit], b_v, C_kv, has_dyn, (n_active_slots == 1 && M >= 16 && !save_ring));

                mtl::Dispatch dqr;
                dqr.barrier = false;
                dqr.p = &p_qr_clip_tau;
                dqr.grid[0] = (uint32_t)M; dqr.grid[1] = (uint32_t)c.n_head; dqr.tg[0] = 128;
                dqr.args = {mtl::Arg::B(b_q), mod->tw(p + ".attn.q_l2_clip.scale"),
                            mtl::Arg::B(b_r), mod->tw(p + ".attn.rel_l2_clip.scale"),
                            mtl::Arg::B(b_pos), mtl::Arg::V(swa_i)};
                ds.push_back(std::move(dqr));

                mtl::Dispatch dkv;
                dkv.barrier = true;
                dkv.p = &p_kv_store_f16;
                dkv.grid[0] = (uint32_t)M; dkv.grid[1] = (uint32_t)c.n_kv; dkv.tg[0] = 128;
                dkv.args = {mtl::Arg::B(b_k), mtl::Arg::B(b_v),
                            mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                            mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                            mtl::Arg::V(swa_i), mtl::Arg::V(visit_kv_stride[visit])};
                ds.push_back(std::move(dkv));
            }

            // Fused RelPos + Sink 24:8 GQA Flash Attention + sigmoid(gate_attn)
            {
                InkAttnArgs aa{};
                aa.M = M;
                aa.is_swa = is_swa ? 1 : 0;
                aa.swa_win = c.swa_win;
                aa.rel_extent = c.rel_extent;
                aa.kv_stride = visit_kv_stride[visit];
                aa.kv_fp8 = 0;
                aa.scale = 1.0f / std::sqrt((float)c.head_dim);
                mtl::Dispatch da;
                da.p = &p_relpos_gqa;
                da.grid[0] = (uint32_t)M; da.grid[1] = (uint32_t)c.n_head; da.tg[0] = 128;
                da.args = {mtl::Arg::B(b_q), mtl::Arg::B(b_r),
                           mtl::Arg::B(mod->rel_proj_pad_buf, (size_t)ip * 16 * 1025 * 4),
                           mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn),
                           mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                           mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                           mtl::Arg::B(b_attn_out), mtl::Arg::V(aa)};
                ds.push_back(std::move(da));
            }

            // Output projection wo_ud + FUSED 4-tap attn_sconv + residual/block update (kills 56+ barriers/tok!)
            push_trunk_linear(ds, p + ".attn.wo_ud.weight", mtl::Arg::B(b_attn_out), mtl::Arg::B(b_wo_raw), M);
            const bool is_blk = (ip % c.block_size == 0);
            if (n_active_slots == 1 && M >= 16 && !save_ring) {
                push_sconv(b_wo_raw, sc_lbase + InkMetalModel::kSconvAttnOff, b_kd_taps, sc_hist_attn[visit], sc_ring_attn[visit], b_wo, D, 0, true);
                if (is_blk) {
                    push_copy(mtl::Arg::B(b_cur), mtl::Arg::B(b_blocks, (size_t)n_blocks * block_stride * 4), M_stride);
                    ++n_blocks;
                    push_copy(mtl::Arg::B(b_wo), mtl::Arg::B(b_cur), M_stride);
                } else {
                    mtl::Dispatch d;
                    d.p = &p_add; d.grid[0] = (uint32_t)((M_stride + 255) / 256); d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_cur), mtl::Arg::B(b_wo), mtl::Arg::B(b_cur), mtl::Arg::V(M_stride)};
                    ds.push_back(std::move(d));
                }
            } else {
                int sb_i = is_blk ? 1 : 0;
                mtl::Dispatch d;
                d.p = &p_sconv4_attn_res;
                d.grid[0] = (uint32_t)n_active_slots; d.grid[1] = (uint32_t)((D + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_wo_raw), mtl::Arg::B(mod->static_sconv_buf, (sc_lbase + InkMetalModel::kSconvAttnOff) * 4),
                          mtl::Arg::B(sc_hist_attn[visit]),
                          mtl::Arg::B(b_active_slots), mtl::Arg::B(b_slot_tok_off), mtl::Arg::B(b_slot_tok_cnt), mtl::Arg::B(b_slot_tok_idx),
                          mtl::Arg::B(b_cur),
                          mtl::Arg::B(b_blocks, (size_t)(is_blk ? n_blocks : 0) * block_stride * 4),
                          mtl::Arg::B(sc_ring_attn[visit]),
                          mtl::Arg::V(sb_i), mtl::Arg::V(save_ring)};
                ds.push_back(std::move(d));
                if (is_blk) ++n_blocks;
            }

            // ---- MoE Sublayer ----
            if (!has_msgs) {
                // Fused attn_res_mix + mlp_norm (kills 24 barriers/tok!)
                mtl::Dispatch d;
                d.p = &p_attn_res_mix_norm; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                int nq = 1;
                mtl::Arg mnw = mod->tw(p + ".mlp_norm.weight");
                d.args = {mtl::Arg::B(b_blocks), mtl::Arg::B(b_cur),
                          mod->tw("model.llm.attn_res.mlp_queries." + std::to_string(ip)),
                          mnw, mnw, mnw, mtl::Arg::B(b_mlp_in),
                          mtl::Arg::V(n_blocks), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride), mtl::Arg::V(nq)};
                ds.push_back(std::move(d));
            } else {
                {
                    mtl::Dispatch d;
                    d.p = &p_attn_res_mix; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    int nq = 1, boff = 0;
                    d.args = {mtl::Arg::B(b_blocks), mtl::Arg::B(b_cur),
                              mod->tw("model.llm.attn_res.mlp_queries." + std::to_string(ip)),
                              mtl::Arg::B(b_mlp_in),
                              mtl::Arg::V(n_blocks), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride),
                              mtl::Arg::V(nq), mtl::Arg::V(boff)};
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_ctrl_cat; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_msgs, (size_t)3 * M_stride * 4), mtl::Arg::B(b_cat)};
                    ds.push_back(std::move(d));
                }
                push_trunk_linear(ds, p + ".controllers.mlp.down_weight", mtl::Arg::B(b_cat), mtl::Arg::B(b_ctrl_gate), M);
                if (M > 8) {
                    push_trunk_linear(ds, p + ".controllers.mlp.gain_weight",
                                      mtl::Arg::B(b_ctrl_gate), mtl::Arg::B(b_wo_raw), M, -1, true);
                    push_trunk_linear(ds, p + ".controllers.mlp.forget_weight",
                                      mtl::Arg::B(b_ctrl_gate), mtl::Arg::B(b_wo), M, -1, false);
                    mtl::Dispatch df;
                    df.p = &p_ctrl_fuse_norm; df.grid[0] = (uint32_t)M; df.tg[0] = 256;
                    df.args = {mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_msgs, (size_t)3 * M_stride * 4),
                               mtl::Arg::B(b_wo_raw), mtl::Arg::B(b_wo),
                               mod->tw(p + ".mlp_norm.weight"), mtl::Arg::B(b_mlp_in)};
                    ds.push_back(std::move(df));
                } else {
                    mtl::Dispatch d;
                    d.p = &p_ctrl_gf_64; d.grid[0] = 144; d.grid[1] = (uint32_t)M; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_msgs, (size_t)3 * M_stride * 4), mtl::Arg::B(b_ctrl_gate),
                              mod->tw(p + ".controllers.mlp.gain_weight"),
                              mod->tw(p + ".controllers.mlp.forget_weight"),
                              mtl::Arg::B(b_mlp_in)};
                    ds.push_back(std::move(d));
                    mtl::Dispatch dn;
                    dn.p = &p_rms_norm; dn.grid[0] = (uint32_t)M; dn.tg[0] = 256;
                    dn.args = {mtl::Arg::B(b_mlp_in), mod->tw(p + ".mlp_norm.weight"), mtl::Arg::B(b_mlp_in), mtl::Arg::V(D)};
                    ds.push_back(std::move(dn));
                }
            }

            // Router (exact FP32!) + Top-8 (+ fused M==1 moe_plan -> kills 48 barriers!) + MoE
            push_trunk_linear(ds, p + ".mlp.gate.weight", mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_r_logits), M, -1, true, true);
            if (M == 1) {
                mtl::Dispatch d;
                d.p = &p_router_topk8_plan_m1; d.grid[0] = 1; d.tg[0] = 128;
                d.args = {mtl::Arg::B(b_r_logits), mod->tw(p + ".mlp.gate.bias"),
                          mtl::Arg::B(b_topk_idx), mtl::Arg::B(b_topk_w),
                          mtl::Arg::B(b_moe_uexp), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                          mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::V(c.route_scale)};
                ds.push_back(std::move(d));
            } else {
                {
                    mtl::Dispatch d;
                    d.p = &p_router_topk8; d.grid[0] = (uint32_t)M; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_r_logits), mod->tw(p + ".mlp.gate.bias"),
                              mtl::Arg::B(b_topk_idx), mtl::Arg::B(b_topk_w), mtl::Arg::V(c.route_scale)};
                    ds.push_back(std::move(d));
                }
                {
                    int P = M * c.top_k;
                    mtl::Dispatch d;
                    d.p = &p_moe_plan; d.grid[0] = 1; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_topk_idx), mtl::Arg::V(P),
                              mtl::Arg::B(b_moe_uexp), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                              mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_tiles)};
                    ds.push_back(std::move(d));
                }
            }
            const size_t ex_off = (size_t)ip * c.n_experts * sizeof(InkExpertDev);
            // Flattened parallel (expert, m_tile) BM=32 BN=64 BK=64 simdgroup_matrix MoE for M >= 48 (zero tail straggler!)
            const bool use_moe_mm = (M >= 48);
            {
                mtl::Dispatch d;
                if (use_moe_mm) {
                    d.p = (layer_homog_gu[ip] == 15) ? &p_moe_gu_mm_sf4
                        : (layer_homog_gu[ip] == 16) ? &p_moe_gu_mm_mx8 : &p_moe_gu_mm_gen;
                    d.ind_off = 8 * sizeof(uint32_t);
                } else {
                    d.p = (layer_homog_gu[ip] == 15) ? &p_moe_gu_sf4
                        : (layer_homog_gu[ip] == 16) ? &p_moe_gu_mx8 : &p_moe_gu;
                    d.ind_off = 2 * sizeof(uint32_t);
                }
                d.tg[0] = 256;
                d.indirect = &b_moe_meta;
                d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_mlp_in),
                          use_moe_mm ? mtl::Arg::B(b_moe_tiles) : mtl::Arg::B(b_moe_uexp),
                          mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                          mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h)};
                ds.push_back(std::move(d));
            }
            {
                mtl::Dispatch d;
                if (use_moe_mm) {
                    d.p = (layer_homog_dn[ip] == 15) ? &p_moe_dn_mm_sf4
                        : (layer_homog_dn[ip] == 16) ? &p_moe_dn_mm_mx8 : &p_moe_dn_mm_gen;
                    d.ind_off = 8 * sizeof(uint32_t);
                } else {
                    d.p = (layer_homog_dn[ip] == 15) ? &p_moe_dn_sf4
                        : (layer_homog_dn[ip] == 16) ? &p_moe_dn_mx8 : &p_moe_dn;
                    d.ind_off = 5 * sizeof(uint32_t);
                }
                d.tg[0] = 256;
                d.indirect = &b_moe_meta;
                d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_topk_w),
                          use_moe_mm ? mtl::Arg::B(b_moe_tiles) : mtl::Arg::B(b_moe_uexp),
                          mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                          mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h), mtl::Arg::B(b_moe_yp)};
                ds.push_back(std::move(d));
            }
            // Fused MoE reduce8 + mlp_sconv + residual add (kills 96 dispatches & 96 barriers/tok!)
            if (n_active_slots == 1 && M >= 16 && !save_ring) {
                {
                    mtl::Dispatch d;
                    d.p = &p_moe_red8; d.grid[0] = (uint32_t)M; d.grid[1] = 9; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_moe_yp), mtl::Arg::B(b_moe_raw)};
                    ds.push_back(std::move(d));
                }
                push_sconv(b_moe_raw, sc_lbase + InkMetalModel::kSconvMlpOff, b_kd_taps, sc_hist_mlp[visit], sc_ring_mlp[visit], b_moe_out, D, 0, true);
                {
                    mtl::Dispatch d;
                    d.p = &p_add; d.grid[0] = (uint32_t)((M_stride + 255) / 256); d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_cur), mtl::Arg::B(b_moe_out), mtl::Arg::B(b_cur), mtl::Arg::V(M_stride)};
                    ds.push_back(std::move(d));
                }
            } else {
                mtl::Dispatch d;
                d.p = &p_moe_red8_sconv_add;
                d.grid[0] = (uint32_t)n_active_slots; d.grid[1] = 9; d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_moe_yp), mtl::Arg::B(mod->static_sconv_buf, (sc_lbase + InkMetalModel::kSconvMlpOff) * 4),
                          mtl::Arg::B(sc_hist_mlp[visit]),
                          mtl::Arg::B(b_active_slots), mtl::Arg::B(b_slot_tok_off), mtl::Arg::B(b_slot_tok_cnt), mtl::Arg::B(b_slot_tok_idx),
                          mtl::Arg::B(b_cur), mtl::Arg::B(sc_ring_mlp[visit]), mtl::Arg::V(save_ring)};
                ds.push_back(std::move(d));
            }
        };

        // 2. Prelude: physical layers 0..5 (visits 0..5)
        int visit = 0;
        for (int ip = 0; ip < c.core_start; ++ip) {
            run_visit(ip, visit++, 0, false);
        }

        // 3. Looped Core: physical layers 6..17 x 3 passes (visits 6..41)
        push_copy(mtl::Arg::B(b_cur), mtl::Arg::B(b_prelude), M_stride);
        prelude_n_blocks = n_blocks;  // 1 (from ip=0)
        for (int p_idx = 0; p_idx < c.num_passes; ++p_idx) {
            int bank_n = 0;
            if (p_idx > 0) {
                // bank = blocks[prelude_n_blocks + 1 .. n_blocks - 1], bank_out = cur
                bank_n = n_blocks - (prelude_n_blocks + 1);
                for (int bi = 0; bi < bank_n; ++bi) {
                    push_copy(mtl::Arg::B(b_blocks, (size_t)(prelude_n_blocks + 1 + bi) * block_stride * 4),
                              mtl::Arg::B(b_bank_blk, (size_t)bi * block_stride * 4), M_stride);
                }
                push_copy(mtl::Arg::B(b_cur), mtl::Arg::B(b_bank_out), M_stride);
                push_copy(mtl::Arg::B(b_prelude), mtl::Arg::B(b_cur), M_stride);
                n_blocks = prelude_n_blocks;
            }
            for (int ip = c.core_start; ip < c.core_end; ++ip) {
                if (p_idx > 0) {
                    mtl::Dispatch dm;
                    dm.p = &p_attn_res_mix;
                    dm.grid[0] = (uint32_t)M; dm.tg[0] = 256;
                    int nq = 4, boff = 0;
                    dm.args = {mtl::Arg::B(b_bank_blk), mtl::Arg::B(b_bank_out),
                               mtl::Arg::B(mod->msg_queries_buf, (size_t)(ip - c.core_start) * 4 * D * 4),
                               mtl::Arg::B(b_msgs),
                               mtl::Arg::V(bank_n), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride),
                               mtl::Arg::V(nq), mtl::Arg::V(boff)};
                    ds.push_back(std::move(dm));
                }
                run_visit(ip, visit++, p_idx, p_idx > 0);
            }
        }

        // 4. Coda: physical layers 18..23 (visits 42..47)
        for (int ip = c.core_end; ip < c.n_phys; ++ip) {
            run_visit(ip, visit++, 0, false);
        }

        // 5. Fused Final Output Attention-Residual mix + model.llm.norm
        {
            mtl::Dispatch d;
            d.p = &p_attn_res_mix_norm; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
            int nq = 1;
            mtl::Arg fnw = mod->tw("model.llm.norm.weight");
            d.args = {mtl::Arg::B(b_blocks), mtl::Arg::B(b_cur),
                      mod->tw("model.llm.attn_res.output_query"), fnw, fnw, fnw, mtl::Arg::B(b_h_norm),
                      mtl::Arg::V(n_blocks), mtl::Arg::V(block_stride), mtl::Arg::V(M_stride), mtl::Arg::V(nq)};
            ds.push_back(std::move(d));
        }

        // 6. Optional DSpark tap[3] (visit 48 = h_norm) -> populate DSpark context KV & f_rnn on GPU!
        if (step.capture_taps && use_dspark) {
            push_copy(mtl::Arg::B(b_h_norm), mtl::Arg::B(b_taps4, (size_t)3 * M_stride * 4), M_stride);
            // Compute all 3 layer mixes into b_in3[0..2] + f_rnn into dsp_frnn_tmp CONCURRENTLY (1 barrier instead of 4!)
            for (int dl = 0; dl < 3; ++dl) {
                std::string dp = "dspark.layers." + std::to_string(dl);
                mtl::Dispatch d;
                d.barrier = (dl == 0);
                d.p = &p_dsp_mix; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_taps4), mod->dw("dspark.fuse_weight"), mod->dw(dp + ".dflare_norm.weight"),
                          mtl::Arg::B(b_in3, (size_t)dl * M_stride * 4), mtl::Arg::V(dl), mtl::Arg::V(M)};
                ds.push_back(std::move(d));
            }
            {
                int row3 = 3;
                mtl::Dispatch d;
                d.barrier = false;
                d.p = &p_dsp_mix; d.grid[0] = (uint32_t)M; d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_taps4), mod->dw("dspark.fuse_weight"),
                          mod->dw("dspark.markov_head.dflare_norm.weight"),
                          mtl::Arg::B(dsp_frnn_tmp), mtl::Arg::V(row3), mtl::Arg::V(M)};
                ds.push_back(std::move(d));
            }
            for (int dl = 0; dl < 3; ++dl) {
                std::string dp = "dspark.layers." + std::to_string(dl);
                mtl::Arg mx = mtl::Arg::B(b_in3, (size_t)dl * M_stride * 4);
                push_dspark_linear(ds, dp + ".self_attn.k_proj_ctx.weight", mx, mtl::Arg::B(b_k_raw), M, true);
                push_dspark_linear(ds, dp + ".self_attn.v_proj_ctx.weight", mx, mtl::Arg::B(b_v_raw), M, false);
                mtl::Dispatch d;
                d.p = &p_dsp_ctx_kv; d.grid[0] = (uint32_t)M; d.grid[1] = 8; d.tg[0] = 128;
                d.args = {mtl::Arg::B(b_k_raw), mtl::Arg::B(b_v_raw), mod->dw(dp + ".self_attn.k_norm_ctx.weight"),
                          mtl::Arg::B(dsp_ctx_k[dl]), mtl::Arg::B(dsp_ctx_v[dl]),
                          mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                          mtl::Arg::V(max_ctx), mtl::Arg::V(c.dspark_rope_theta)};
                ds.push_back(std::move(d));
            }
        }

        // 7. Unembed + softcap(30.0) + argmax (fused unembed+argmax when logits_out == nullptr & n_lg == 1!)
        int n_lg = 0;
        if (step.compute_logits) {
            const auto& uti = mod->man.get("model.llm.unembed.weight");
            if (step.logits_last_only_idx >= 0) {
                n_lg = 1;
                const mtl::Arg h_arg = mtl::Arg::B(b_h_norm, (size_t)step.logits_last_only_idx * D * 4);
                if (!logits_out && argmax_out && (uti.fmt == FMT_Q8ROW || uti.fmt == FMT_BF16 || uti.fmt == FMT_S6)) {
                    const int V = c.vocab_unpadded;
                    const int rpt = (uti.fmt == FMT_S6) ? 16 : 32;
                    const int nparts = (V + rpt - 1) / rpt;
                    mtl::Dispatch du;
                    du.p = (uti.fmt == FMT_Q8ROW) ? &p_unemb_argmax_q8
                         : (uti.fmt == FMT_BF16)  ? &p_unemb_argmax_bf16 : &p_unemb_argmax_sk6;
                    du.grid[0] = (uint32_t)nparts; du.tg[0] = 256;
                    du.args = {mod->tw("model.llm.unembed.weight"), h_arg,
                               mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::V(V), mtl::Arg::V(D)};
                    ds.push_back(std::move(du));
                    mtl::Dispatch dr;
                    dr.p = &p_argmax_part; dr.grid[0] = 1; dr.tg[0] = 256;
                    dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::B(b_argmax), mtl::Arg::V(nparts)};
                    ds.push_back(std::move(dr));
                } else {
                    push_trunk_linear(ds, "model.llm.unembed.weight", h_arg, mtl::Arg::B(b_logits), 1, c.vocab_unpadded);
                    if (logits_out) {
                        int tot = c.vocab_unpadded;
                        mtl::Dispatch d;
                        d.p = &p_softcap; d.grid[0] = (uint32_t)((tot + 255) / 256); d.tg[0] = 256;
                        d.args = {mtl::Arg::B(b_logits), mtl::Arg::V(tot), mtl::Arg::V(c.logit_softcap)};
                        ds.push_back(std::move(d));
                    }
                    if (argmax_out) {
                        mtl::Dispatch d;
                        d.p = &p_argmax_rows; d.grid[0] = 1; d.tg[0] = 256;
                        d.args = {mtl::Arg::B(b_logits), mtl::Arg::B(b_argmax), mtl::Arg::V(c.vocab_unpadded)};
                        ds.push_back(std::move(d));
                    }
                }
            } else {
                n_lg = std::min(M, 64);
                push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(b_h_norm), mtl::Arg::B(b_logits),
                                  n_lg, c.vocab_unpadded);
                if (logits_out) {
                    int tot = n_lg * c.vocab_unpadded;
                    mtl::Dispatch d;
                    d.p = &p_softcap; d.grid[0] = (uint32_t)((tot + 255) / 256); d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_logits), mtl::Arg::V(tot), mtl::Arg::V(c.logit_softcap)};
                    ds.push_back(std::move(d));
                }
                if (argmax_out) {
                    mtl::Dispatch d;
                    d.p = &p_argmax_rows; d.grid[0] = (uint32_t)n_lg; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_logits), mtl::Arg::B(b_argmax), mtl::Arg::V(c.vocab_unpadded)};
                    ds.push_back(std::move(d));
                }
            }
        }

        // Execute ALL 48 visits in ONE Metal command buffer!
        last_gpu_ms = cx->run_graph(ds) * 1e3;

        // Update latest f_rnn per slot if taps were captured
        if (step.capture_taps && use_dspark) {
            const float* src = dsp_frnn_tmp.as<float>();
            float* dst = dsp_frnn_slot.as<float>();
            for (int m = 0; m < M; ++m) {
                std::memcpy(dst + (size_t)step.slot_ids[m] * D, src + (size_t)m * D, (size_t)D * 4);
            }
        }

        if (step.compute_logits && n_lg > 0) {
            if (logits_out) {
                std::memcpy(logits_out, b_logits.ptr, (size_t)n_lg * c.vocab_unpadded * sizeof(float));
            }
            if (argmax_out) {
                std::memcpy(argmax_out, b_argmax.ptr, (size_t)n_lg * sizeof(int32_t));
            }
        }
        return true;
    }

    // Persistent 100% On-GPU DSpark Block-8 Recurrent Markov Drafting (ONE command buffer, ZERO host sync!)
    // Generates 7 draft tokens in out_draft[0..6] conditioned on `anchor_tok` at `anchor_pos` (using committed ctx_kv < anchor_pos & f_rnn[slot])!
    int dspark_draft_block8(int slot, int32_t anchor_tok, int anchor_pos, int32_t* out_draft) {
        if (!use_dspark || anchor_pos <= 0 || anchor_pos + 8 >= max_ctx) return 0;
        const auto& c = mod->man.c;
        const int D = c.hidden;

        // Seed tok_chain[0] = anchor_tok, zero Markov RNN state [256]
        bd_tok_chain.as<int32_t>()[0] = anchor_tok;
        std::memset(bd_state.ptr, 0, 256 * sizeof(float));

        std::vector<mtl::Dispatch> ds;
        ds.reserve(256);

        auto push_copy = [&](const mtl::Arg& src, const mtl::Arg& dst, int n_floats) {
            mtl::Dispatch d;
            d.p = &p_copy_f32;
            d.grid[0] = (uint32_t)((n_floats + 255) / 256); d.tg[0] = 256;
            d.args = {src, dst, mtl::Arg::V(n_floats)};
            ds.push_back(std::move(d));
        };

        // 1. Init Block-8: [rms(embed(anchor_tok)), mask_embed x 7]
        {
            const auto& eti = mod->man.get("model.llm.embed.weight");
            mtl::Dispatch d;
            d.grid[0] = 8; d.tg[0] = 256;
            if (eti.fmt == FMT_BF16) {
                d.p = &p_dsp_init_bf16;
                d.args = {mod->tw("model.llm.embed.weight"), mod->tw("model.llm.embed_norm.weight"),
                          mod->dw("dspark.mask_embed"), mtl::Arg::B(bd_cur), mtl::Arg::V(anchor_tok)};
            } else {
                d.p = &p_dsp_init_q8;
                d.args = {mod->tw("model.llm.embed.weight"), mod->tw("model.llm.embed_norm.weight"),
                          mod->dw("dspark.mask_embed"), mtl::Arg::B(bd_cur), mtl::Arg::V(anchor_tok),
                          mtl::Arg::V(c.vocab_padded)};
            }
            ds.push_back(std::move(d));
        }

        // 2. 3 non-causal block-attention layers
        for (int il = 0; il < 3; ++il) {
            std::string lp = "dspark.layers." + std::to_string(il);
            push_copy(mtl::Arg::B(bd_cur), mtl::Arg::B(bd_res), 8 * D);
            {
                mtl::Dispatch d;
                d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".input_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, lp + ".self_attn.q_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_q), 8, true);
            push_dspark_linear(ds, lp + ".self_attn.k_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_k), 8, false);
            push_dspark_linear(ds, lp + ".self_attn.v_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_v), 8, false);
            {
                int nh = 24, nkv = 8;
                mtl::Dispatch dq; dq.p = &p_dsp_qk_rope; dq.grid[0] = 8; dq.grid[1] = 24; dq.tg[0] = 128;
                dq.args = {mtl::Arg::B(bd_q), mod->dw(lp + ".self_attn.q_norm.weight"),
                           mtl::Arg::V(nh), mtl::Arg::V(anchor_pos), mtl::Arg::V(c.dspark_rope_theta)};
                ds.push_back(std::move(dq));
                mtl::Dispatch dk; dk.barrier = false; dk.p = &p_dsp_qk_rope; dk.grid[0] = 8; dk.grid[1] = 8; dk.tg[0] = 128;
                dk.args = {mtl::Arg::B(bd_k), mod->dw(lp + ".self_attn.k_norm.weight"),
                           mtl::Arg::V(nkv), mtl::Arg::V(anchor_pos), mtl::Arg::V(c.dspark_rope_theta)};
                ds.push_back(std::move(dk));
            }
            {
                int is_swa = c.dspark_is_swa[il];
                mtl::Dispatch da; da.p = &p_dsp_attn; da.grid[0] = 8; da.grid[1] = 24; da.tg[0] = 128;
                da.args = {mtl::Arg::B(bd_q), mtl::Arg::B(bd_k), mtl::Arg::B(bd_v),
                           mtl::Arg::B(dsp_ctx_k[il]), mtl::Arg::B(dsp_ctx_v[il]), mtl::Arg::B(bd_ao),
                           mtl::Arg::V(slot), mtl::Arg::V(anchor_pos), mtl::Arg::V(is_swa),
                           mtl::Arg::V(c.dspark_swa_win), mtl::Arg::V(max_ctx)};
                ds.push_back(std::move(da));
            }
            push_dspark_linear(ds, lp + ".self_attn.o_proj.weight", mtl::Arg::B(bd_ao), mtl::Arg::B(bd_proj), 8);
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_res), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
            // MLP
            push_copy(mtl::Arg::B(bd_cur), mtl::Arg::B(bd_res), 8 * D);
            {
                mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".post_attention_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, lp + ".mlp.gate_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mg), 8, true);
            push_dspark_linear(ds, lp + ".mlp.up_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mu), 8, false);
            {
                int n8i = 8 * 9216;
                mtl::Dispatch d; d.p = &p_dsp_swiglu; d.grid[0] = (uint32_t)((n8i + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_mg), mtl::Arg::B(bd_mu), mtl::Arg::B(bd_mh), mtl::Arg::V(n8i)};
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, lp + ".mlp.down_proj.weight", mtl::Arg::B(bd_mh), mtl::Arg::B(bd_proj), 8);
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_res), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
        }

        // 3. Final norm (linearity fusion: unembed(cur[s] + corr) == base_logits[s] + bias_logits -> 7 unembeds vs 14!)
        {
            mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
            d.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.norm.weight"), mtl::Arg::B(bd_cur), mtl::Arg::V(D)};
            ds.push_back(std::move(d));
        }

        // 4. 7-step Recurrent Markov RNN chain — 100% on GPU inside the SAME command buffer!
        const auto& mw1_ti = mod->man.get_d("dspark.markov_head.markov_w1.weight");
        for (int s = 1; s < 8; ++s) {
            {
                mtl::Dispatch d;
                d.grid[0] = (5120 + 255) / 256; d.tg[0] = 256;
                if (mw1_ti.fmt == FMT_BF16) {
                    d.p = &p_dsp_mjoint_bf16;
                    d.args = {mtl::Arg::B(bd_state), mod->dw("dspark.markov_head.markov_w1.weight"),
                              mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                              mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(s)};
                } else {
                    d.p = &p_dsp_mjoint_q8;
                    d.args = {mtl::Arg::B(bd_state), mod->dw("dspark.markov_head.markov_w1.weight"),
                              mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                              mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(s),
                              mtl::Arg::V(c.vocab_padded)};
                }
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, "dspark.markov_head.rnn_proj.weight", mtl::Arg::B(bd_joint), mtl::Arg::B(bd_rproj), 1);
            {
                mtl::Dispatch d; d.p = &p_dsp_mrnn; d.grid[0] = 1; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_rproj), mod->dw("dspark.markov_head.rnn_proj.bias"),
                          mtl::Arg::B(bd_state), mtl::Arg::B(bd_latent)};
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, "dspark.markov_head.hidden_out.weight", mtl::Arg::B(bd_latent), mtl::Arg::B(bd_corr), 1);
            {
                mtl::Dispatch d; d.p = &p_dsp_add_bias; d.grid[0] = (D + 255) / 256; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_corr), mod->dw("dspark.markov_head.hidden_out.bias"), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            {
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (D + 255) / 256; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur, (size_t)s * D * 4), mtl::Arg::B(bd_corr), mtl::Arg::B(bd_comb), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            // Fused Unembed + Argmax directly on un-softcapped logits (ZERO 800 KB bd_logits DRAM write/read!)
            const auto& uti = mod->man.get("model.llm.unembed.weight");
            if (uti.fmt == FMT_Q8ROW || uti.fmt == FMT_BF16 || uti.fmt == FMT_S6) {
                const int V = c.vocab_unpadded;
                const int rpt = (uti.fmt == FMT_S6) ? 16 : 32;
                const int nparts = (V + rpt - 1) / rpt;
                mtl::Dispatch du;
                du.p = (uti.fmt == FMT_Q8ROW) ? &p_unemb_argmax_q8
                     : (uti.fmt == FMT_BF16)  ? &p_unemb_argmax_bf16 : &p_unemb_argmax_sk6;
                du.grid[0] = (uint32_t)nparts; du.tg[0] = 256;
                du.args = {mod->tw("model.llm.unembed.weight"), mtl::Arg::B(bd_comb),
                           mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::V(V), mtl::Arg::V(D)};
                ds.push_back(std::move(du));
                mtl::Dispatch dr;
                dr.p = &p_argmax_part; dr.grid[0] = 1; dr.tg[0] = 256;
                dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::B(bd_tok_chain, (size_t)s * 4), mtl::Arg::V(nparts)};
                ds.push_back(std::move(dr));
            } else {
                push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(bd_comb), mtl::Arg::B(bd_logits),
                                  1, c.vocab_unpadded);
                mtl::Dispatch da; da.p = &p_argmax_rows; da.grid[0] = 1; da.tg[0] = 256;
                da.args = {mtl::Arg::B(bd_logits), mtl::Arg::B(bd_tok_chain, (size_t)s * 4), mtl::Arg::V(c.vocab_unpadded)};
                ds.push_back(std::move(da));
            }
        }

        cx->run_graph(ds);
        const int32_t* chain = bd_tok_chain.as<int32_t>();
        for (int i = 0; i < 7; ++i) out_draft[i] = chain[i + 1];
        return 7;
    }

private:
    mtl::Pipeline p_copy_f32;

    bool compile_shaders(const std::string& kernel_dir, std::string& err);
};

}  // namespace ink_metal
