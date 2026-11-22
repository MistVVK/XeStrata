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
    bool is_tree = false;               // exact Multi-Branch Tree verify (M <= 32, non-destructive KV & ShortConv!)
    int tree_anchor_pos = 0;            // anchor_pos for tree root node 0
    std::vector<int32_t> tree_parent;   // [M], parent[0] = -1, parent[m] < m
    std::vector<uint32_t> tree_anc_mask;// [M], bit j set iff tree node j is an ancestor of m (inclusive)
};

struct InkDraftTree {
    int M = 0;
    bool is_linear = false;
    float expected_gain = 1.0f;  // expected_tps / plain_tps
    std::vector<int32_t> tok_ids;
    std::vector<int32_t> parents;
    std::vector<int32_t> depths;
    std::vector<uint32_t> anc_masks;
};

struct DspPoolNodeHost {
    int32_t tok;
    int32_t parent_pool;
    int32_t depth;
    float logp;
};

struct InkMetalRunner {
    mtl::Context* cx = nullptr;
    InkMetalModel* mod = nullptr;
    InkMetalVision vis;

    int lib_codecs = -1, lib_ops = -1, lib_dspark = -1, lib_vis = -1, lib_mpp = -1;
    bool is_apple10 = false;
    bool use_mpp = false, use_mpp_trunk_f16 = false, use_dspark_f16 = false;
    int mpp_decomp_cap = 0;

    // Metal 4 MPP Neural Accelerator pipelines (active when lib_mpp >= 0)
    mtl::Pipeline p_mpp_f32_to_f16, p_mpp_bf16_to_f16, p_mpp_q8_to_f16, p_mpp_sk_to_f16;
    mtl::Pipeline p_mpp_mm_16_64, p_mpp_mm_32_64, p_mpp_mm_64_64, p_mpp_mm_64_128, p_mpp_mm_128_64;
    mtl::Pipeline p_mpp_split_qrsg, p_mpp_split_kv, p_mpp_ctrl_fuse_gf;
    mtl::Pipeline p_mpp_moe_plan, p_mpp_moe_gather;
    // [codec: 0=sf4, 1=mx8, 2=gen][bucket: 0=16, 1=32, 2=64, 3=128]
    mtl::Pipeline p_mpp_moe_gu_dir[3][4], p_mpp_moe_dn_dir[3][4];
    mtl::Pipeline p_mpp_moe_combine, p_mpp_prep_qr, p_mpp_relpos_gqa_pf;

    // Pipelines
    mtl::Pipeline p_lin_bf16_mv1, p_lin_bf16_mv, p_lin_bf16_mm8, p_lin_bf16_mm;
    mtl::Pipeline p_lin_q8_mv1, p_lin_q8_mv, p_lin_q8_mm8, p_lin_q8_mm;
    mtl::Pipeline p_lin_f32, p_lin_f32_mm, p_lin_f32_exact_mm;
    mtl::Pipeline p_lin_sk[9], p_lin_sk5_mm8, p_lin_sk6_mm8;
    mtl::Pipeline p_unemb_argmax_q8, p_unemb_argmax_bf16, p_unemb_argmax_sk6, p_argmax_part;
    mtl::Pipeline p_unemb_argmax_m_q8, p_unemb_argmax_m_bf16, p_unemb_argmax_m_sk6, p_argmax_part_m;
    mtl::Pipeline p_moe_plan, p_moe_gu, p_moe_dn, p_moe_gu_sk23, p_moe_dn_sk23, p_moe_red8;
    mtl::Pipeline p_moe_gu_sf4, p_moe_dn_sf4, p_moe_gu_mx8, p_moe_dn_mx8;
    mtl::Pipeline p_moe_gu_sk2, p_moe_dn_sk2, p_moe_gu_sk3, p_moe_dn_sk3, p_moe_gu_sk4, p_moe_dn_sk4, p_moe_gu_sk5, p_moe_dn_sk5;
    mtl::Pipeline p_moe_gu_mm_sf4, p_moe_dn_mm_sf4, p_moe_gu_mm_mx8, p_moe_dn_mm_mx8, p_moe_gu_mm_gen, p_moe_dn_mm_gen;
    int layer_homog_gu[24]{};
    int layer_homog_dn[24]{};
    uint32_t layer_gu_mask[24]{};
    uint32_t layer_dn_mask[24]{};

    mtl::Pipeline p_emb_bf16, p_emb_q8;
    mtl::Pipeline p_attn_res_mix, p_attn_res_mix_norm, p_ctrl_cat, p_ctrl_apply, p_ctrl_gf_64, p_ctrl_fuse_norm, p_rms_norm, p_add;
    mtl::Pipeline p_sconv4_multi, p_sconv4_attn_res, p_moe_red8_sconv_add, p_sconv4_pf, p_sconv4_pf_commit;
    mtl::Pipeline p_sconv4_tree, p_sconv4_attn_res_tree, p_sconv4_mlp_add_tree, p_sconv4_tree_commit;
    mtl::Pipeline p_qr_clip_tau, p_kv_store_f16, p_kv_store_tree_f16, p_kv_tree_commit_f16;
    mtl::Pipeline p_relpos_gqa, p_relpos_gqa_splitk, p_relpos_gqa_splitk_red, p_relpos_tree_gqa_splitk;
    mtl::Pipeline p_router_topk8, p_router_topk8_plan_m1, p_softcap, p_argmax_rows;

    mtl::Pipeline p_dsp_mix, p_dsp_ctx_kv, p_dsp_ctx_kv_tree, p_dsp_ctx_kv_tree_commit, p_dsp_init_bf16, p_dsp_init_q8;
    mtl::Pipeline p_dsp_qk_rope, p_dsp_attn, p_dsp_swiglu;
    mtl::Pipeline p_dsp_mjoint_bf16, p_dsp_mjoint_q8, p_dsp_mjoint_tree_bf16, p_dsp_mjoint_tree_q8, p_dsp_mrnn, p_dsp_mrnn_tree, p_dsp_add_bias;
    mtl::Pipeline p_dsp_comb_s1, p_dsp_comb_s2;
    mtl::Pipeline p_dsp_add_bias7, p_dsp_top512, p_dsp_score512_bf16, p_dsp_score512_q8, p_dsp_score512_sk6, p_dsp_score512_red;
    mtl::Pipeline p_dsp_build_meff_q8, p_dsp_meff_argmax_q8, p_dsp_meff_top3_q8, p_dsp_top3_red;
    mtl::Pipeline p_dsp_meff_top4_q8, p_dsp_beam4_init_s1, p_dsp_mjoint_beam4_bf16, p_dsp_mjoint_beam4_q8;
    mtl::Pipeline p_dsp_mrnn_beam4, p_dsp_meff_beam4_top2_q8, p_dsp_beam4_select;
    bool use_dspark_meff = false;

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

    // Per-step ShortConv verify ring [48] x [32, C] (serves BOTH linear 8x3 verify ring AND Tree M<=32 raw save!)
    std::vector<mtl::Buffer> sc_ring_k;
    std::vector<mtl::Buffer> sc_ring_v;
    std::vector<mtl::Buffer> sc_ring_attn;
    std::vector<mtl::Buffer> sc_ring_mlp;

    // Tree verify scratch KV [48] x [32, 8, 128] FP16 (6 MiB total) + DSpark tree ctx KV [3] (384 KiB)
    std::vector<mtl::Buffer> tree_k;
    std::vector<mtl::Buffer> tree_v;
    mtl::Buffer dsp_tree_k[3], dsp_tree_v[3];
    mtl::Buffer b_tree_parent, b_tree_anc_mask, b_tree_path;
    mtl::Buffer bd_tree_states, bd_tree_top3_i, bd_tree_top3_v, bd_tree_btoks;
    mtl::Buffer bd_beam_st_in, bd_beam_st_out, bd_beam_lat, bd_beam_joint, bd_beam_rproj;
    mtl::Buffer bd_beam_toks, bd_beam_logp, bd_beam_pidx, bd_beam_pool;

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
    mtl::Buffer b_sinks, b_gate_attn, b_attn_out, b_attn_part_o, b_attn_part_ml, b_wo_raw, b_wo;
    mtl::Buffer b_mlp_in, b_r_logits, b_topk_idx, b_topk_w;
    mtl::Buffer b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_meta, b_moe_tiles;
    mtl::Buffer b_moe_h, b_moe_yp, b_moe_raw, b_moe_out;
    mtl::Buffer b_h_out, b_h_norm, b_logits, b_argmax;
    mtl::Buffer b_taps4;                   // [4, max_batch, 2304] for DSpark taps [5, 23, 41, 48]

    // DSpark Block-8 scratch buffers (M=8)
    mtl::Buffer bd_cur, bd_res, bd_h, bd_q, bd_k, bd_v, bd_ao, bd_proj;
    mtl::Buffer bd_mg, bd_mu, bd_mh, bd_logits, bd_comb;
    mtl::Buffer bd_state, bd_joint, bd_rproj, bd_latent, bd_corr;
    mtl::Buffer bd_tok_chain, bd_part_val, bd_part_idx, bd_cand_val, bd_cand_idx;
    mtl::Buffer bd_meff_sc, bd_meff_q;

    // Metal 4 MPP workspace & optional concatenated FP16 trunk cache (on >= 24 GiB working-set Macs)
    mtl::Buffer b_mpp_x_f16, b_mpp_tmp_out, b_mpp_qh, b_mpp_rh;
    mtl::Buffer b_mpp_srt_pair, b_mpp_pair_pos, b_mpp_tiles_128, b_mpp_tiles_64, b_mpp_tiles_32, b_mpp_tiles_16, b_mpp_meta;
    mtl::Buffer b_mpp_xs, b_mpp_hs, b_mpp_ye;
    mtl::Buffer w_qrsg_f16[24], w_k_f16[24], w_v_f16[24], w_wo_f16[24];
    mtl::Buffer w_ctrl_dn_f16[12][4], w_ctrl_gf_f16[12][4];
    mtl::Buffer w_dsp_q_f16[3], w_dsp_k_f16[3], w_dsp_v_f16[3], w_dsp_o_f16[3];
    mtl::Buffer w_dsp_g_f16[3], w_dsp_u_f16[3], w_dsp_d_f16[3];

    double last_gpu_ms = 0.0;

    bool init(mtl::Context& ctx, InkMetalModel& m, int slots, int ctx_len, int batch_cap,
              bool enable_dspark, const std::string& vision_path,
              const std::string& kernel_dir, std::string& err) {
        cx = &ctx;
        mod = &m;
        is_apple10 = cx->has_family_apple(10);
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
        p_lin_f32_exact_mm = cx->pipeline(lib_codecs, "ink_linear_f32_exact_mm");
        p_unemb_argmax_q8   = cx->pipeline(lib_codecs, "ink_unembed_argmax_q8row");
        p_unemb_argmax_bf16 = cx->pipeline(lib_codecs, "ink_unembed_argmax_bf16");
        p_unemb_argmax_sk6  = cx->pipeline(lib_codecs, "ink_unembed_argmax_sk6");
        p_argmax_part       = cx->pipeline(lib_codecs, "ink_argmax_part_reduce");
        p_unemb_argmax_m_q8   = cx->pipeline(lib_codecs, "ink_unembed_argmax_m_q8row");
        p_unemb_argmax_m_bf16 = cx->pipeline(lib_codecs, "ink_unembed_argmax_m_bf16");
        p_unemb_argmax_m_sk6  = cx->pipeline(lib_codecs, "ink_unembed_argmax_m_sk6");
        p_argmax_part_m       = cx->pipeline(lib_codecs, "ink_argmax_part_reduce_m");
        for (int f : {2, 3, 4, 5, 6, 8}) {
            p_lin_sk[f] = cx->pipeline(lib_codecs, "ink_linear_sk_mv_" + std::to_string(f));
        }
        p_lin_sk5_mm8   = cx->pipeline(lib_codecs, "ink_linear_sk5_mm8");
        p_lin_sk6_mm8   = cx->pipeline(lib_codecs, "ink_linear_sk6_mm8");
        p_moe_plan      = cx->pipeline(lib_codecs, "ink_moe_plan");
        p_moe_gu        = cx->pipeline(lib_codecs, "ink_moe_gu_situ");
        p_moe_dn        = cx->pipeline(lib_codecs, "ink_moe_dn");
        p_moe_gu_sk23   = cx->pipeline(lib_codecs, "ink_moe_gu_sk23");
        p_moe_dn_sk23   = cx->pipeline(lib_codecs, "ink_moe_dn_sk23");
        p_moe_gu_sf4    = cx->pipeline(lib_codecs, "ink_moe_gu_sf4");
        p_moe_dn_sf4    = cx->pipeline(lib_codecs, "ink_moe_dn_sf4");
        p_moe_gu_mx8    = cx->pipeline(lib_codecs, "ink_moe_gu_mx8");
        p_moe_dn_mx8    = cx->pipeline(lib_codecs, "ink_moe_dn_mx8");
        p_moe_gu_sk2    = cx->pipeline(lib_codecs, "ink_moe_gu_sk2");
        p_moe_dn_sk2    = cx->pipeline(lib_codecs, "ink_moe_dn_sk2");
        p_moe_gu_sk3    = cx->pipeline(lib_codecs, "ink_moe_gu_sk3");
        p_moe_dn_sk3    = cx->pipeline(lib_codecs, "ink_moe_dn_sk3");
        p_moe_gu_sk4    = cx->pipeline(lib_codecs, "ink_moe_gu_sk4");
        p_moe_dn_sk4    = cx->pipeline(lib_codecs, "ink_moe_dn_sk4");
        p_moe_gu_sk5    = cx->pipeline(lib_codecs, "ink_moe_gu_sk5");
        p_moe_dn_sk5    = cx->pipeline(lib_codecs, "ink_moe_dn_sk5");
        p_moe_gu_mm_sf4 = cx->pipeline(lib_codecs, "ink_moe_gu_mm_sf4");
        p_moe_dn_mm_sf4 = cx->pipeline(lib_codecs, "ink_moe_dn_mm_sf4");
        p_moe_gu_mm_mx8 = cx->pipeline(lib_codecs, "ink_moe_gu_mm_mx8");
        p_moe_dn_mm_mx8 = cx->pipeline(lib_codecs, "ink_moe_dn_mm_mx8");
        p_moe_gu_mm_gen = cx->pipeline(lib_codecs, "ink_moe_gu_mm_gen");
        p_moe_dn_mm_gen = cx->pipeline(lib_codecs, "ink_moe_dn_mm_gen");
        p_moe_red8      = cx->pipeline(lib_codecs, "ink_moe_reduce8");

        // Detect homogeneous layers & per-layer codec bitmasks for zero-spill concurrent dispatch!
        const auto* ed_host = mod->expert_dev_buf.as<InkExpertDev>();
        for (int il = 0; il < mod->man.c.n_phys; ++il) {
            int g0 = ed_host[il * mod->man.c.n_experts].gu_code;
            int d0 = ed_host[il * mod->man.c.n_experts].dn_code;
            bool hg = true, hd = true;
            uint32_t gm = 0, dm = 0;
            for (int e = 0; e < mod->man.c.n_experts; ++e) {
                const auto& ed = ed_host[il * mod->man.c.n_experts + e];
                if (ed.gu_code != g0) hg = false;
                if (ed.dn_code != d0) hd = false;
                if (ed.gu_code < 32) gm |= (1u << ed.gu_code);
                if (ed.dn_code < 32) dm |= (1u << ed.dn_code);
            }
            layer_homog_gu[il] = hg ? g0 : -1;
            layer_homog_dn[il] = hd ? d0 : -1;
            layer_gu_mask[il] = gm;
            layer_dn_mask[il] = dm;
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
        p_sconv4_tree         = cx->pipeline(lib_ops, "ink_sconv4_tree");
        p_sconv4_attn_res_tree = cx->pipeline(lib_ops, "ink_sconv4_attn_res_tree");
        p_sconv4_mlp_add_tree = cx->pipeline(lib_ops, "ink_sconv4_mlp_add_tree");
        p_sconv4_tree_commit  = cx->pipeline(lib_ops, "ink_sconv4_tree_commit");
        p_qr_clip_tau         = cx->pipeline(lib_ops, "ink_qr_clip_tau");
        p_kv_store_f16        = cx->pipeline(lib_ops, "ink_kv_norm_store_f16");
        p_kv_store_tree_f16   = cx->pipeline(lib_ops, "ink_kv_store_tree_f16");
        p_kv_tree_commit_f16  = cx->pipeline(lib_ops, "ink_kv_tree_commit_f16");
        p_relpos_gqa          = cx->pipeline(lib_ops, "ink_relpos_sink_gqa");
        p_relpos_gqa_splitk   = cx->pipeline(lib_ops, "ink_relpos_gqa_splitk");
        p_relpos_gqa_splitk_red = cx->pipeline(lib_ops, "ink_relpos_gqa_splitk_reduce");
        p_relpos_tree_gqa_splitk = cx->pipeline(lib_ops, "ink_relpos_tree_gqa_splitk");
        p_router_topk8        = cx->pipeline(lib_ops, "ink_router_topk8");
        p_router_topk8_plan_m1 = cx->pipeline(lib_ops, "ink_router_topk8_plan_m1");
        p_softcap             = cx->pipeline(lib_ops, "ink_softcap_f32");
        p_argmax_rows         = cx->pipeline(lib_ops, "ink_argmax_rows");

        p_dsp_mix         = cx->pipeline(lib_dspark, "dspark_mix_taps");
        p_dsp_ctx_kv      = cx->pipeline(lib_dspark, "dspark_ctx_kv_store");
        p_dsp_ctx_kv_tree = cx->pipeline(lib_dspark, "dspark_ctx_kv_store_tree");
        p_dsp_ctx_kv_tree_commit = cx->pipeline(lib_dspark, "dspark_ctx_kv_tree_commit");
        p_dsp_init_bf16   = cx->pipeline(lib_dspark, "dspark_init_block_bf16");
        p_dsp_init_q8     = cx->pipeline(lib_dspark, "dspark_init_block_q8row");
        p_dsp_qk_rope     = cx->pipeline(lib_dspark, "dspark_qk_norm_rope");
        p_dsp_attn        = cx->pipeline(lib_dspark, "dspark_block_attn");
        p_dsp_swiglu      = cx->pipeline(lib_dspark, "dspark_swiglu");
        p_dsp_mjoint_bf16 = cx->pipeline(lib_dspark, "dspark_markov_joint_bf16");
        p_dsp_mjoint_q8   = cx->pipeline(lib_dspark, "dspark_markov_joint_q8row");
        p_dsp_mjoint_tree_bf16 = cx->pipeline(lib_dspark, "dspark_markov_joint_tree_bf16");
        p_dsp_mjoint_tree_q8   = cx->pipeline(lib_dspark, "dspark_markov_joint_tree_q8row");
        p_dsp_mrnn        = cx->pipeline(lib_dspark, "dspark_markov_rnn_update");
        p_dsp_mrnn_tree   = cx->pipeline(lib_dspark, "dspark_markov_rnn_update_tree");
        p_dsp_add_bias    = cx->pipeline(lib_dspark, "dspark_add_bias");
        p_dsp_comb_s1     = cx->pipeline(lib_dspark, "dspark_combine_argmax_s1");
        p_dsp_comb_s2     = cx->pipeline(lib_dspark, "dspark_combine_argmax_s2");
        p_dsp_add_bias7     = cx->pipeline(lib_dspark, "dspark_add_bias_7");
        p_dsp_top512        = cx->pipeline(lib_dspark, "dspark_top512_shortlist");
        p_dsp_score512_bf16 = cx->pipeline(lib_dspark, "dspark_score_512_bf16");
        p_dsp_score512_q8   = cx->pipeline(lib_dspark, "dspark_score_512_q8row");
        p_dsp_score512_sk6  = cx->pipeline(lib_dspark, "dspark_score_512_sk6");
        p_dsp_score512_red  = cx->pipeline(lib_dspark, "dspark_score_512_reduce");
        p_dsp_build_meff_q8 = cx->pipeline(lib_dspark, "dspark_build_meff_q8");
        p_dsp_meff_argmax_q8 = cx->pipeline(lib_dspark, "dspark_meff_argmax_q8");
        p_dsp_meff_top3_q8  = cx->pipeline(lib_dspark, "dspark_meff_top3_q8");
        p_dsp_top3_red      = cx->pipeline(lib_dspark, "dspark_top3_reduce");
        p_dsp_meff_top4_q8  = cx->pipeline(lib_dspark, "dspark_meff_top4_q8");
        p_dsp_beam4_init_s1 = cx->pipeline(lib_dspark, "dspark_beam4_init_s1");
        p_dsp_mjoint_beam4_bf16 = cx->pipeline(lib_dspark, "dspark_markov_joint_beam4_bf16");
        p_dsp_mjoint_beam4_q8   = cx->pipeline(lib_dspark, "dspark_markov_joint_beam4_q8row");
        p_dsp_mrnn_beam4        = cx->pipeline(lib_dspark, "dspark_markov_rnn_update_beam4");
        p_dsp_meff_beam4_top2_q8 = cx->pipeline(lib_dspark, "dspark_meff_beam4_top2_q8");
        p_dsp_beam4_select       = cx->pipeline(lib_dspark, "dspark_beam4_select");

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
        tree_k.resize(c.n_visits);
        tree_v.resize(c.n_visits);

        size_t kv_bytes_tot = 0, sc_bytes_tot = 0;
        for (int v = 0; v < c.n_visits; ++v) {
            const int ip = c.phys_for_visit(v);
            const int stride = c.is_swa[ip] ? c.swa_win : max_ctx;
            visit_kv_stride[v] = stride;
            const size_t kvb = (size_t)max_slots * stride * c.n_kv * c.head_dim * sizeof(uint16_t);
            kv_k[v] = cx->alloc(kvb);
            kv_v[v] = cx->alloc(kvb);
            kv_bytes_tot += 2 * kvb;
            const size_t tkvb = 32ul * c.n_kv * c.head_dim * sizeof(uint16_t);
            tree_k[v] = cx->alloc(tkvb);
            tree_v[v] = cx->alloc(tkvb);

            const size_t sc_kv = (size_t)max_slots * 3 * (c.n_kv * c.head_dim) * sizeof(float);
            const size_t sc_d  = (size_t)max_slots * 3 * c.hidden * sizeof(float);
            sc_hist_k[v]    = cx->alloc(sc_kv);
            sc_hist_v[v]    = cx->alloc(sc_kv);
            sc_hist_attn[v] = cx->alloc(sc_d);
            sc_hist_mlp[v]  = cx->alloc(sc_d);
            sc_ring_k[v]    = cx->alloc(32ul * (c.n_kv * c.head_dim) * sizeof(float));
            sc_ring_v[v]    = cx->alloc(32ul * (c.n_kv * c.head_dim) * sizeof(float));
            sc_ring_attn[v] = cx->alloc(32ul * c.hidden * sizeof(float));
            sc_ring_mlp[v]  = cx->alloc(32ul * c.hidden * sizeof(float));
            sc_bytes_tot += 2 * sc_kv + 2 * sc_d;
        }
        b_tree_parent   = cx->alloc(32 * sizeof(int32_t));
        b_tree_anc_mask = cx->alloc(32 * sizeof(uint32_t));
        b_tree_path     = cx->alloc(32 * sizeof(int32_t));

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
        b_sinks        = cx->alloc(MB * c.n_head * 4);
        b_gate_attn    = cx->alloc(MB * (c.n_head * c.head_dim) * 4);
        b_attn_out     = cx->alloc(MB * (c.n_head * c.head_dim) * 4);
        b_attn_part_o  = cx->alloc(32ul * c.n_head * 8ul * c.head_dim * 4);
        b_attn_part_ml = cx->alloc(32ul * c.n_head * 8ul * 2ul * 4);
        b_wo_raw       = cx->alloc(MB * D * 4);
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
        bd_part_val = cx->alloc(16ul * 16384ul * 4);
        bd_part_idx = cx->alloc(16ul * 16384ul * 4);
        b_taps4     = cx->alloc(4 * MB * D * 4);

        if (use_dspark) {
            const size_t dkv = (size_t)max_slots * max_ctx * c.n_kv * c.head_dim * sizeof(uint16_t);
            const size_t dtkv = 32ul * c.n_kv * c.head_dim * sizeof(uint16_t);
            for (int l = 0; l < 3; ++l) {
                dsp_ctx_k[l] = cx->alloc(dkv);
                dsp_ctx_v[l] = cx->alloc(dkv);
                dsp_tree_k[l] = cx->alloc(dtkv);
                dsp_tree_v[l] = cx->alloc(dtkv);
            }
            bd_tree_states = cx->alloc(9ul * 256ul * sizeof(float));  // [0..7] main states + [8] branch scratch state
            bd_tree_top3_i = cx->alloc(7ul * 3ul * sizeof(int32_t));
            bd_tree_top3_v = cx->alloc(7ul * 3ul * sizeof(float));
            bd_tree_btoks  = cx->alloc(8ul * 8ul * sizeof(int32_t));  // up to 8 branches x 8 tokens
            bd_beam_st_in  = cx->alloc(4ul * 256ul * sizeof(float));
            bd_beam_st_out = cx->alloc(4ul * 256ul * sizeof(float));
            bd_beam_lat    = cx->alloc(4ul * 256ul * sizeof(float));
            bd_beam_joint  = cx->alloc(4ul * 5120ul * sizeof(float));
            bd_beam_rproj  = cx->alloc(4ul * 768ul * sizeof(float));
            bd_beam_toks   = cx->alloc(4ul * sizeof(int32_t));
            bd_beam_logp   = cx->alloc(4ul * sizeof(float));
            bd_beam_pidx   = cx->alloc(4ul * sizeof(int32_t));
            bd_beam_pool   = cx->alloc(52ul * sizeof(DspPoolNodeHost));
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
            bd_cand_val    = cx->alloc(7ul * 512ul * sizeof(float));
            bd_cand_idx    = cx->alloc(7ul * 512ul * sizeof(int32_t));

            // Precompute EXACT low-rank M_eff = unembed @ hidden_out.weight [V, 256] as Q8_ROW (~52 MiB, ~12 ms at init)!
            const auto& uti = mod->man.get("model.llm.unembed.weight");
            const auto& hti = mod->man.get_d("dspark.markov_head.hidden_out.weight");
            if ((uti.fmt == FMT_BF16 || uti.fmt == FMT_Q8ROW || uti.fmt == FMT_S6) &&
                (hti.fmt == FMT_F32 || hti.fmt == FMT_BF16 || hti.fmt == FMT_Q8ROW) &&
                hti.rows() == (uint64_t)D && hti.cols() == 256ul) {
                bd_meff_sc = cx->alloc((size_t)c.vocab_unpadded * sizeof(float));
                bd_meff_q  = cx->alloc((size_t)c.vocab_unpadded * 256ul);
                mtl::Buffer w_hout_t = cx->alloc(256ul * D * sizeof(uint16_t));
                uint16_t* dst_t = w_hout_t.as<uint16_t>();
                auto f32_to_f16 = [](float f) -> uint16_t {
                    __fp16 h = (__fp16)f;
                    uint16_t u;
                    std::memcpy(&u, &h, 2);
                    return u;
                };
                const uint8_t* hraw = mod->base + mod->sec1_off + hti.off;
                for (int d = 0; d < (int)D; ++d) {
                    if (hti.fmt == FMT_F32) {
                        const float* rp = reinterpret_cast<const float*>(hraw) + (size_t)d * 256;
                        for (int r = 0; r < 256; ++r) dst_t[(size_t)r * D + d] = f32_to_f16(rp[r]);
                    } else if (hti.fmt == FMT_BF16) {
                        const uint16_t* rp = reinterpret_cast<const uint16_t*>(hraw) + (size_t)d * 256;
                        for (int r = 0; r < 256; ++r) {
                            uint32_t bits = (uint32_t)rp[r] << 16;
                            float fv; std::memcpy(&fv, &bits, 4);
                            dst_t[(size_t)r * D + d] = f32_to_f16(fv);
                        }
                    } else {
                        const float sc = reinterpret_cast<const float*>(hraw)[d];
                        const int8_t* rp = reinterpret_cast<const int8_t*>(hraw + (size_t)D * 4) + (size_t)d * 256;
                        for (int r = 0; r < 256; ++r) dst_t[(size_t)r * D + d] = f32_to_f16((float)rp[r] * sc);
                    }
                }
                int ufmt = (uti.fmt == FMT_BF16) ? 0 : (uti.fmt == FMT_Q8ROW) ? 1 : 2;
                mtl::Dispatch dm;
                dm.p = &p_dsp_build_meff_q8;
                dm.grid[0] = (uint32_t)((c.vocab_unpadded + 15) / 16);
                dm.tg[0] = 256;
                dm.args = {mod->tw("model.llm.unembed.weight"), mtl::Arg::B(w_hout_t),
                           mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                           mtl::Arg::V(c.vocab_unpadded), mtl::Arg::V(c.vocab_padded), mtl::Arg::V(ufmt)};
                std::vector<mtl::Dispatch> one; one.push_back(std::move(dm));
                cx->run_graph(one);
                use_dspark_meff = true;
            }
        }

        if (!vision_path.empty()) {
            if (!vis.load(*cx, lib_codecs, lib_vis, vision_path, &InkMetalModel::read_section0_json, err)) {
                return false;
            }
        }

        // Optional Metal 4 Neural Accelerator (mpp::tensor_ops::matmul2d) pipelines & FP16 cache!
        if (lib_mpp >= 0) {
            use_mpp = true;
            p_mpp_f32_to_f16    = cx->pipeline(lib_mpp, "ink_mpp_f32_to_f16");
            p_mpp_bf16_to_f16   = cx->pipeline(lib_mpp, "ink_mpp_bf16_to_f16");
            p_mpp_q8_to_f16     = cx->pipeline(lib_mpp, "ink_mpp_q8row_to_f16");
            p_mpp_sk_to_f16     = cx->pipeline(lib_mpp, "ink_mpp_sk_to_f16");
            p_mpp_mm_16_64      = cx->pipeline(lib_mpp, "ink_mpp_mm_f16_16_64_4");
            p_mpp_mm_32_64      = cx->pipeline(lib_mpp, "ink_mpp_mm_f16_32_64_4");
            p_mpp_mm_64_64      = cx->pipeline(lib_mpp, "ink_mpp_mm_f16_64_64_4");
            p_mpp_mm_64_128     = cx->pipeline(lib_mpp, "ink_mpp_mm_f16_64_128_4");
            p_mpp_mm_128_64     = cx->pipeline(lib_mpp, "ink_mpp_mm_f16_128_64_4");
            p_mpp_split_qrsg    = cx->pipeline(lib_mpp, "ink_mpp_split_qrsg");
            p_mpp_split_kv      = cx->pipeline(lib_mpp, "ink_mpp_split_kv_dyn");
            p_mpp_ctrl_fuse_gf  = cx->pipeline(lib_mpp, "ink_mpp_ctrl_fuse_gf_norm");
            p_mpp_moe_plan       = cx->pipeline(lib_mpp, "ink_mpp_moe_plan");
            p_mpp_moe_gather     = cx->pipeline(lib_mpp, "ink_mpp_moe_gather");
            const char* ctags[3] = {"sf4", "mx8", "gen"};
            const char* btags[4] = {"16", "32", "64", "128"};
            for (int ci = 0; ci < 3; ++ci) {
                for (int bi = 0; bi < 4; ++bi) {
                    p_mpp_moe_gu_dir[ci][bi] = cx->pipeline(lib_mpp, std::string("ink_mpp_moe_gu_") + ctags[ci] + "_" + btags[bi]);
                    p_mpp_moe_dn_dir[ci][bi] = cx->pipeline(lib_mpp, std::string("ink_mpp_moe_dn_") + ctags[ci] + "_" + btags[bi]);
                }
            }
            p_mpp_moe_combine    = cx->pipeline(lib_mpp, "ink_mpp_moe_combine");
            p_mpp_prep_qr        = cx->pipeline(lib_mpp, "ink_mpp_prep_qr");
            p_mpp_relpos_gqa_pf  = cx->pipeline(lib_mpp, "ink_mpp_relpos_gqa_pf");

            const size_t mb_pad = (MB + 31ul) & ~31ul;
            const size_t max_tot_rows = MB * (size_t)c.top_k + (size_t)c.n_experts * 128ul;
            b_mpp_x_f16     = cx->alloc(std::max((size_t)8 * 9216, 3ul * MB * 4608ul) * sizeof(uint16_t));
            b_mpp_tmp_out   = cx->alloc(MB * (6552ul + 2112ul) * sizeof(float));
            b_mpp_qh        = cx->alloc(24ul * mb_pad * 128ul * sizeof(uint16_t));
            b_mpp_rh        = cx->alloc(24ul * mb_pad * 16ul * sizeof(uint16_t));
            b_mpp_srt_pair  = cx->alloc(max_tot_rows * sizeof(int32_t));
            b_mpp_pair_pos  = cx->alloc(MB * (size_t)c.top_k * sizeof(int32_t));
            b_mpp_tiles_128 = cx->alloc((max_tot_rows / 64ul + 256ul) * 16ul);
            b_mpp_tiles_64  = cx->alloc((max_tot_rows / 32ul + 256ul) * 16ul);
            b_mpp_tiles_32  = cx->alloc((max_tot_rows / 16ul + 256ul) * 16ul);
            b_mpp_tiles_16  = cx->alloc((max_tot_rows / 16ul + 256ul) * 16ul);
            b_mpp_meta      = cx->alloc(24ul * sizeof(uint32_t));
            b_mpp_xs        = cx->alloc(max_tot_rows * D * sizeof(uint16_t));
            b_mpp_hs        = cx->alloc(max_tot_rows * (size_t)c.inter * sizeof(uint16_t));
            b_mpp_ye        = cx->alloc(max_tot_rows * D * sizeof(float));

            const uint64_t ws = cx->recommended_working_set();
            mpp_decomp_cap = 0;
            use_mpp_trunk_f16 = (ws >= (24ull << 30));
            if (use_mpp_trunk_f16) {
                std::vector<mtl::Dispatch> init_ds;
                init_ds.reserve(512);
                auto cvt_tensor = [&](const InkTensorInfo& ti, const mtl::Arg& src_w, const mtl::Buffer& dst_b, size_t dst_row_off) {
                    const int N = (int)ti.rows(), K = (int)ti.cols();
                    const size_t byte_off = dst_row_off * (size_t)K * sizeof(uint16_t);
                    mtl::Dispatch d;
                    d.barrier = false;
                    if (ti.fmt == FMT_BF16) {
                        int tot = N * K;
                        d.p = &p_mpp_bf16_to_f16; d.grid[0] = (uint32_t)((tot + 1023) / 1024); d.tg[0] = 256;
                        d.args = {src_w, mtl::Arg::B(dst_b, byte_off), mtl::Arg::V(tot)};
                    } else if (ti.fmt == FMT_F32) {
                        int tot = N * K;
                        d.p = &p_mpp_f32_to_f16; d.grid[0] = (uint32_t)((tot + 1023) / 1024); d.tg[0] = 256;
                        d.args = {src_w, mtl::Arg::B(dst_b, byte_off), mtl::Arg::V(tot)};
                    } else if (ti.fmt == FMT_Q8ROW) {
                        d.p = &p_mpp_q8_to_f16; d.grid[0] = (uint32_t)((K + 255) / 256); d.grid[1] = (uint32_t)N; d.tg[0] = 256;
                        d.args = {src_w, mtl::Arg::B(dst_b, byte_off), mtl::Arg::V(N), mtl::Arg::V(K)};
                    } else if (ti.fmt >= FMT_S2 && ti.fmt <= FMT_S8) {
                        int fmt_i = (int)ti.fmt;
                        d.p = &p_mpp_sk_to_f16; d.grid[0] = (uint32_t)((K + 255) / 256); d.grid[1] = (uint32_t)N; d.tg[0] = 32;
                        d.args = {src_w, mtl::Arg::B(dst_b, byte_off), mtl::Arg::V(N), mtl::Arg::V(K), mtl::Arg::V(fmt_i)};
                    }
                    init_ds.push_back(std::move(d));
                };
                auto cvt_trunk = [&](const std::string& name, const mtl::Buffer& dst_b, size_t dst_row_off) {
                    cvt_tensor(mod->man.get(name), mod->tw(name), dst_b, dst_row_off);
                };
                for (int il = 0; il < c.n_phys; ++il) {
                    std::string p = "model.llm.layers." + std::to_string(il);
                    const bool is_swa = (c.is_swa[il] != 0);
                    w_qrsg_f16[il] = cx->alloc(6552ul * D * sizeof(uint16_t));
                    cvt_trunk(p + ".attn.wq_du.weight",           w_qrsg_f16[il], 0);
                    cvt_trunk(p + ".attn.wr_du.weight",           w_qrsg_f16[il], 3072);
                    cvt_trunk(p + ".attn.attn_sink_proj.weight",  w_qrsg_f16[il], 3456);
                    cvt_trunk(p + ".attn.gate_attn.weight",       w_qrsg_f16[il], 3480);

                    const size_t krows = is_swa ? 1024ul : 1056ul;
                    w_k_f16[il] = cx->alloc(krows * D * sizeof(uint16_t));
                    w_v_f16[il] = cx->alloc(krows * D * sizeof(uint16_t));
                    cvt_trunk(p + ".attn.wk_dv.weight", w_k_f16[il], 0);
                    cvt_trunk(p + ".attn.wv_dv.weight", w_v_f16[il], 0);
                    if (!is_swa) {
                        cvt_trunk(p + ".attn.k_sconv.weight_proj", w_k_f16[il], 1024);
                        cvt_trunk(p + ".attn.v_sconv.weight_proj", w_v_f16[il], 1024);
                    }
                    w_wo_f16[il] = cx->alloc(D * 3072ul * sizeof(uint16_t));
                    cvt_trunk(p + ".attn.wo_ud.weight", w_wo_f16[il], 0);
                }
                const char* cnames[4] = {"attn", "k", "v", "mlp"};
                for (int il = c.core_start; il < c.core_end; ++il) {
                    std::string p = "model.llm.layers." + std::to_string(il);
                    const int ci = il - c.core_start;
                    for (int j = 0; j < 4; ++j) {
                        w_ctrl_dn_f16[ci][j] = cx->alloc((size_t)c.gate_rank * (2ul * D) * sizeof(uint16_t));
                        w_ctrl_gf_f16[ci][j] = cx->alloc((2ul * D) * (size_t)c.gate_rank * sizeof(uint16_t));
                        cvt_trunk(p + ".controllers." + cnames[j] + ".down_weight",   w_ctrl_dn_f16[ci][j], 0);
                        cvt_trunk(p + ".controllers." + cnames[j] + ".gain_weight",   w_ctrl_gf_f16[ci][j], 0);
                        cvt_trunk(p + ".controllers." + cnames[j] + ".forget_weight", w_ctrl_gf_f16[ci][j], D);
                    }
                }
                // Keep S5_K DSpark layers compressed (161 MB vs 496 MB FP16!) and run native LUT ink_linear_sk5_mm8 -> 3x less DRAM & -496 MiB RAM!
                use_dspark_f16 = false;
                if (use_dspark) {
                    const auto& gti = mod->man.get_d("dspark.layers.0.mlp.gate_proj.weight");
                    if (gti.fmt != FMT_S5) {
                        use_dspark_f16 = true;
                        auto cvt_dsp = [&](const std::string& name, mtl::Buffer& dst_b) {
                            const auto& ti = mod->man.get_d(name);
                            dst_b = cx->alloc((size_t)ti.rows() * (size_t)ti.cols() * sizeof(uint16_t));
                            cvt_tensor(ti, mod->dw(name), dst_b, 0);
                        };
                        for (int dl = 0; dl < 3; ++dl) {
                            std::string lp = "dspark.layers." + std::to_string(dl);
                            cvt_dsp(lp + ".self_attn.q_proj.weight", w_dsp_q_f16[dl]);
                            cvt_dsp(lp + ".self_attn.k_proj.weight", w_dsp_k_f16[dl]);
                            cvt_dsp(lp + ".self_attn.v_proj.weight", w_dsp_v_f16[dl]);
                            cvt_dsp(lp + ".self_attn.o_proj.weight", w_dsp_o_f16[dl]);
                            cvt_dsp(lp + ".mlp.gate_proj.weight",    w_dsp_g_f16[dl]);
                            cvt_dsp(lp + ".mlp.up_proj.weight",      w_dsp_u_f16[dl]);
                            cvt_dsp(lp + ".mlp.down_proj.weight",    w_dsp_d_f16[dl]);
                        }
                    }
                }
                cx->run_graph(init_ds);
            }
        }

        fprintf(stderr,
                "[ink_metal] runner ready: slots=%d ctx=%d batch=%d dspark=%d vision=%d mpp=%d(trunk=%d,hot=%d) | KV=%.1f MiB SC=%.1f MiB TotalAlloc=%.2f GiB\n",
                max_slots, max_ctx, max_batch, (int)use_dspark, (int)vis.loaded,
                (int)use_mpp, (int)use_mpp_trunk_f16, mpp_decomp_cap,
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
            } else if (M >= 2 && M <= 8 && N >= 32768) {
                // On huge vocab unembed (N=200k), simdgroup_half8x8 mm8 beats register mv at M=2..8 (cuts DSpark k=2..4 by -0.7 ms!)
                d.p = &p_lin_bf16_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else if (M <= 8) {
                // Single-pass barrier-free register GEMV (exact FP32 activations, ~1.8x faster than mm8 at M=2..4!)
                d.p = &p_lin_bf16_mv;
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_bf16_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 63) / 64);
                d.tg[0] = 256;
            }
        } else if (ti.fmt == FMT_Q8ROW) {
            if (M == 1 && (K & 15) == 0) {
                d.p = &p_lin_q8_mv1;
                d.grid[0] = (uint32_t)((N + 31) / 32);
                d.tg[0] = 256;
            } else if (M >= 2 && M <= 8 && N >= 32768) {
                d.p = &p_lin_q8_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else if (M <= 8) {
                // Single-pass barrier-free 1B-weight register GEMV (2x less DRAM than FP16 MPP, exact FP32 activations!)
                d.p = &p_lin_q8_mv;
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            } else {
                d.p = &p_lin_q8_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 31) / 32);
                d.tg[0] = 256;
            }
        } else if (ti.fmt == FMT_F32) {
            if (M > 8 && !exact_f32) {
                d.p = &p_lin_f32_mm;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 31) / 32);
                d.tg[0] = 256;
            } else if (M >= 64 && exact_f32) {
                d.p = &p_lin_f32_exact_mm;
                d.grid[0] = (uint32_t)((N + 31) / 32);
                d.grid[1] = (uint32_t)((M + 31) / 32);
                d.tg[0] = 128;
            } else {
                d.p = &p_lin_f32;
                d.grid[0] = (uint32_t)((N + 3) / 4);
                d.grid[1] = (uint32_t)((M + 3) / 4);
                d.tg[0] = 128;
            }
        } else if (ti.fmt >= FMT_S2 && ti.fmt <= FMT_S8) {
            if (ti.fmt == FMT_S6 && M >= 2 && M <= 8 && is_apple10) {
                d.p = &p_lin_sk6_mm8;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.tg[0] = 256;
            } else if (ti.fmt == FMT_S5 && M >= 2 && M <= 8) {
                d.p = &p_lin_sk5_mm8;
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

    struct InkMppMmArgs {
        int M, N, K, out_stride;
    };
    struct InkMppAttnArgs {
        int M, M_pad, slot, pos0, is_swa, swa_win, rel_extent, kv_stride;
        float scale;
        int pad;
    };
    struct InkTreeAttnArgs {
        int M, slot, anchor_pos, is_swa, swa_win, rel_extent, kv_stride, pad;
        float scale;
        int pad2;
    };

    void push_f32_to_f16(std::vector<mtl::Dispatch>& ds, const mtl::Arg& src, const mtl::Arg& dst, int n_floats, bool barrier = true) {
        mtl::Dispatch d;
        d.barrier = barrier;
        d.p = &p_mpp_f32_to_f16;
        d.grid[0] = (uint32_t)((n_floats + 1023) / 1024);
        d.tg[0] = 256;
        d.args = {src, dst, mtl::Arg::V(n_floats)};
        ds.push_back(std::move(d));
    }

    void push_mpp_gemm(std::vector<mtl::Dispatch>& ds, const mtl::Arg& w_f16, const mtl::Arg& x_f16, const mtl::Arg& y_f32,
                       int M, int N, int K, int out_stride = -1, bool barrier = true) {
        if (out_stride <= 0) out_stride = N;
        InkMppMmArgs a{M, N, K, out_stride};
        mtl::Dispatch d;
        d.barrier = barrier;
        d.tg[0] = 128;
        if (M <= 16) {
            d.p = &p_mpp_mm_16_64;
            d.grid[0] = (uint32_t)((N + 63) / 64);
            d.grid[1] = (uint32_t)((M + 15) / 16);
        } else if (M <= 32) {
            d.p = &p_mpp_mm_32_64;
            d.grid[0] = (uint32_t)((N + 63) / 64);
            d.grid[1] = (uint32_t)((M + 31) / 32);
        } else if (M >= 128 && N >= 2304) {
            d.p = &p_mpp_mm_64_128;
            d.grid[0] = (uint32_t)((N + 127) / 128);
            d.grid[1] = (uint32_t)((M + 63) / 64);
        } else {
            d.p = &p_mpp_mm_64_64;
            d.grid[0] = (uint32_t)((N + 63) / 64);
            d.grid[1] = (uint32_t)((M + 63) / 64);
        }
        d.args = {w_f16, x_f16, y_f32, mtl::Arg::V(a)};
        ds.push_back(std::move(d));
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
        const bool is_tree = step.is_tree;
        if (is_tree) {
            if (M > 32 || (int)step.tree_parent.size() != M || (int)step.tree_anc_mask.size() != M) {
                err = "invalid tree batch M=" + std::to_string(M);
                return false;
            }
            std::memcpy(b_tree_parent.ptr, step.tree_parent.data(), (size_t)M * sizeof(int32_t));
            std::memcpy(b_tree_anc_mask.ptr, step.tree_anc_mask.data(), (size_t)M * sizeof(uint32_t));
        }
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
                if (M > 8) {
                    if (use_mpp && use_mpp_trunk_f16) {
                        const int ci = ip - c.core_start;
                        push_f32_to_f16(ds, mtl::Arg::B(b_cat), mtl::Arg::B(b_mpp_x_f16), 3 * M * 2 * D, true);
                        for (int j = 0; j < 3; ++j) {
                            const size_t xoff = (size_t)j * (size_t)M * 2 * D * 2;
                            const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                            push_mpp_gemm(ds, mtl::Arg::B(w_ctrl_dn_f16[ci][j]), mtl::Arg::B(b_mpp_x_f16, xoff),
                                          mtl::Arg::B(b_ctrl_gate, goff), M, c.gate_rank, 2 * D, c.gate_rank, (j == 0));
                        }
                        push_f32_to_f16(ds, mtl::Arg::B(b_ctrl_gate), mtl::Arg::B(b_mpp_x_f16), 3 * M * c.gate_rank, true);
                        for (int j = 0; j < 3; ++j) {
                            const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 2;
                            const size_t gf_off = (size_t)j * (size_t)M * 2 * D * 4;
                            push_mpp_gemm(ds, mtl::Arg::B(w_ctrl_gf_f16[ci][j]), mtl::Arg::B(b_mpp_x_f16, goff),
                                          mtl::Arg::B(b_cat, gf_off), M, 2 * D, c.gate_rank, 2 * D, (j == 0));
                        }
                        const int order[3] = {1, 2, 0};
                        for (int oi = 0; oi < 3; ++oi) {
                            const int j = is_swa ? order[oi] : oi;
                            const size_t src_off = is_swa ? 0 : ((size_t)j * M_stride * 4);
                            const size_t dst_off = (size_t)j * M_stride * 4;
                            const size_t moff = (size_t)j * M_stride * 4;
                            const size_t gf_off = (size_t)j * (size_t)M * 2 * D * 4;
                            std::string nkey = is_swa ? (p + ".attn_norm.weight") : (p + "." + nnames[j]);
                            mtl::Dispatch df;
                            df.barrier = (oi == 0) || (is_swa && oi == 2);
                            df.p = &p_mpp_ctrl_fuse_gf; df.grid[0] = (uint32_t)M; df.tg[0] = 256;
                            df.args = {mtl::Arg::B(b_in3, src_off), mtl::Arg::B(b_msgs, moff),
                                       mtl::Arg::B(b_cat, gf_off), mod->tw(nkey), mtl::Arg::B(b_in3, dst_off)};
                            ds.push_back(std::move(df));
                        }
                    } else {
                        for (int j = 0; j < 3; ++j) {
                            const size_t coff = (size_t)j * (size_t)M * 2 * D * 4;
                            const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                            push_trunk_linear(ds, p + ".controllers." + cnames[j] + ".down_weight",
                                              mtl::Arg::B(b_cat, coff), mtl::Arg::B(b_ctrl_gate, goff), M, -1, (j == 0));
                        }
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
                    }
                } else {
                    for (int j = 0; j < 3; ++j) {
                        const size_t coff = (size_t)j * (size_t)M * 2 * D * 4;
                        const size_t goff = (size_t)j * (size_t)M * c.gate_rank * 4;
                        push_trunk_linear(ds, p + ".controllers." + cnames[j] + ".down_weight",
                                          mtl::Arg::B(b_cat, coff), mtl::Arg::B(b_ctrl_gate, goff), M, -1, (j == 0));
                    }
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
            // Measured exact Apple10 crossover: BF16 mv beats MPP at M<=3, MPP wins at M>=4 (-6.7 ms at M=8!);
            // Q8_ROW 1B mv beats MPP at M<=4 (-4.4 ms at M=1!), MPP wins at M>=5 (-5.8 ms at M=8!).
            const bool trunk_is_q8 = (mod->man.get(p + ".attn.wq_du.weight").fmt == FMT_Q8ROW);
            const int mpp_trunk_min_m = is_apple10 ? (trunk_is_q8 ? 5 : 4) : 16;
            if (use_mpp && use_mpp_trunk_f16 && M >= mpp_trunk_min_m) {
                const bool shared_qkv = (is_swa && !has_msgs);
                const int n_in_floats = (shared_qkv ? 1 : 3) * M_stride;
                push_f32_to_f16(ds, mtl::Arg::B(b_in3, 0), mtl::Arg::B(b_mpp_x_f16, 0), n_in_floats, true);
                const mtl::Arg xq = mtl::Arg::B(b_mpp_x_f16, 0);
                const mtl::Arg xk = shared_qkv ? xq : mtl::Arg::B(b_mpp_x_f16, (size_t)M_stride * 2);
                const mtl::Arg xv = shared_qkv ? xq : mtl::Arg::B(b_mpp_x_f16, (size_t)2 * M_stride * 2);

                const int kv_rows = is_swa ? 1024 : 1056;
                const size_t tmp_k_off = (size_t)M * 6552 * 4;
                const size_t tmp_v_off = (size_t)M * (6552 + 1056) * 4;
                // 3 CONCURRENT Metal 4 Neural Accelerator GEMMs (replaces 6-8 separate dispatches!)
                push_mpp_gemm(ds, mtl::Arg::B(w_qrsg_f16[ip]), xq, mtl::Arg::B(b_mpp_tmp_out, 0), M, 6552, D, 6552, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_k_f16[ip]), xk,
                              is_swa ? mtl::Arg::B(b_k_raw) : mtl::Arg::B(b_mpp_tmp_out, tmp_k_off), M, kv_rows, D, kv_rows, false);
                push_mpp_gemm(ds, mtl::Arg::B(w_v_f16[ip]), xv,
                              is_swa ? mtl::Arg::B(b_v_raw) : mtl::Arg::B(b_mpp_tmp_out, tmp_v_off), M, kv_rows, D, kv_rows, false);
                {
                    int st_q = 6552;
                    mtl::Dispatch dq;
                    dq.barrier = true;
                    dq.p = &p_mpp_split_qrsg; dq.grid[0] = (uint32_t)M; dq.grid[1] = 26; dq.tg[0] = 256;
                    dq.args = {mtl::Arg::B(b_mpp_tmp_out, 0), mtl::Arg::B(b_q), mtl::Arg::B(b_r),
                               mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn), mtl::Arg::V(st_q)};
                    ds.push_back(std::move(dq));
                }
                if (!is_swa) {
                    int hd = 1, st_kv = 1056;
                    mtl::Dispatch dk;
                    dk.barrier = false;
                    dk.p = &p_mpp_split_kv; dk.grid[0] = (uint32_t)M; dk.grid[1] = 5; dk.tg[0] = 256;
                    dk.args = {mtl::Arg::B(b_mpp_tmp_out, tmp_k_off), mtl::Arg::B(b_k_raw), mtl::Arg::B(b_kd_taps),
                               mtl::Arg::V(hd), mtl::Arg::V(st_kv)};
                    ds.push_back(std::move(dk));
                    mtl::Dispatch dv;
                    dv.barrier = false;
                    dv.p = &p_mpp_split_kv; dv.grid[0] = (uint32_t)M; dv.grid[1] = 5; dv.tg[0] = 256;
                    dv.args = {mtl::Arg::B(b_mpp_tmp_out, tmp_v_off), mtl::Arg::B(b_v_raw), mtl::Arg::B(b_vd_taps),
                               mtl::Arg::V(hd), mtl::Arg::V(st_kv)};
                    ds.push_back(std::move(dv));
                }
            } else {
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
            }

            // Helper: 4-tap causal ShortConv (parallel across M*C for single-slot prefill M >= 16 AND tree mode!)
            auto push_sconv = [&](const mtl::Buffer& xin, size_t sc_off, const mtl::Buffer& dyn,
                                  const mtl::Buffer& hist, const mtl::Buffer& ring, const mtl::Buffer& yout,
                                  int C_ch, int has_dyn, bool first_barrier) {
                if (is_tree) {
                    int s0 = as_p[0];
                    mtl::Dispatch d;
                    d.barrier = first_barrier;
                    d.p = &p_sconv4_tree;
                    d.grid[0] = (uint32_t)M; d.grid[1] = (uint32_t)((C_ch + 255) / 256); d.tg[0] = 256;
                    d.args = {mtl::Arg::B(xin), mtl::Arg::B(mod->static_sconv_buf, sc_off * 4),
                              mtl::Arg::B(dyn), mtl::Arg::B(hist), mtl::Arg::B(b_tree_parent),
                              mtl::Arg::B(ring), mtl::Arg::B(yout),
                              mtl::Arg::V(s0), mtl::Arg::V(C_ch), mtl::Arg::V(has_dyn)};
                    ds.push_back(std::move(d));
                } else if (n_active_slots == 1 && M >= 16 && !save_ring) {
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
                push_sconv(b_v_raw, sc_lbase + InkMetalModel::kSconvVOff, b_vd_taps, sc_hist_v[visit], sc_ring_v[visit], b_v, C_kv, has_dyn, (!is_tree && n_active_slots == 1 && M >= 16 && !save_ring));

                mtl::Dispatch dqr;
                dqr.barrier = false;
                dqr.p = &p_qr_clip_tau;
                dqr.grid[0] = (uint32_t)M; dqr.grid[1] = (uint32_t)c.n_head; dqr.tg[0] = 128;
                dqr.args = {mtl::Arg::B(b_q), mod->tw(p + ".attn.q_l2_clip.scale"),
                            mtl::Arg::B(b_r), mod->tw(p + ".attn.rel_l2_clip.scale"),
                            mtl::Arg::B(b_pos), mtl::Arg::V(swa_i)};
                ds.push_back(std::move(dqr));

                if (is_tree) {
                    mtl::Dispatch dkv;
                    dkv.barrier = true;
                    dkv.p = &p_kv_store_tree_f16;
                    dkv.grid[0] = (uint32_t)M; dkv.grid[1] = (uint32_t)c.n_kv; dkv.tg[0] = 128;
                    dkv.args = {mtl::Arg::B(b_k), mtl::Arg::B(b_v),
                                mtl::Arg::B(tree_k[visit]), mtl::Arg::B(tree_v[visit])};
                    ds.push_back(std::move(dkv));
                } else {
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
            }

            // Fused RelPos + Sink 24:8 GQA Flash Attention + sigmoid(gate_attn)
            {
                if (is_tree) {
                    InkTreeAttnArgs ta{};
                    ta.M = M;
                    ta.slot = as_p[0];
                    ta.anchor_pos = step.tree_anchor_pos;
                    ta.is_swa = is_swa ? 1 : 0;
                    ta.swa_win = c.swa_win;
                    ta.rel_extent = c.rel_extent;
                    ta.kv_stride = visit_kv_stride[visit];
                    ta.scale = 1.0f / std::sqrt((float)c.head_dim);
                    mtl::Dispatch d1;
                    d1.p = &p_relpos_tree_gqa_splitk;
                    d1.grid[0] = (uint32_t)M; d1.grid[1] = (uint32_t)c.n_kv; d1.grid[2] = 8u; d1.tg[0] = 128;
                    d1.args = {mtl::Arg::B(b_q), mtl::Arg::B(b_r),
                               mtl::Arg::B(mod->rel_proj_t_f16_buf, (size_t)ip * 1025 * 16 * 2),
                               mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                               mtl::Arg::B(tree_k[visit]), mtl::Arg::B(tree_v[visit]),
                               mtl::Arg::B(b_pos), mtl::Arg::B(b_tree_anc_mask),
                               mtl::Arg::B(b_attn_part_o), mtl::Arg::B(b_attn_part_ml), mtl::Arg::V(ta)};
                    ds.push_back(std::move(d1));
                    mtl::Dispatch d2;
                    d2.p = &p_relpos_gqa_splitk_red;
                    d2.grid[0] = (uint32_t)M; d2.grid[1] = (uint32_t)c.n_head; d2.tg[0] = 128;
                    d2.args = {mtl::Arg::B(b_attn_part_o), mtl::Arg::B(b_attn_part_ml),
                               mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn), mtl::Arg::B(b_attn_out)};
                    ds.push_back(std::move(d2));
                } else {
                    bool use_mpp_attn = false;
                    if (use_mpp && n_active_slots == 1 && M >= 32) {
                        use_mpp_attn = true;
                        for (int m = 1; m < M; ++m) if (step.pos[m] != step.pos[0] + m) { use_mpp_attn = false; break; }
                    }
                    if (use_mpp_attn) {
                        const int M_pad = (M + 31) & ~31;
                        const float sc = 1.0f / std::sqrt((float)c.head_dim);
                        {
                            mtl::Dispatch dp;
                            dp.p = &p_mpp_prep_qr; dp.grid[0] = (uint32_t)M_pad; dp.grid[1] = 24; dp.tg[0] = 128;
                            dp.args = {mtl::Arg::B(b_q), mtl::Arg::B(b_r), mtl::Arg::B(b_mpp_qh), mtl::Arg::B(b_mpp_rh),
                                       mtl::Arg::V(M), mtl::Arg::V(M_pad), mtl::Arg::V(sc)};
                            ds.push_back(std::move(dp));
                        }
                        InkMppAttnArgs ma{};
                        ma.M = M;
                        ma.M_pad = M_pad;
                        ma.slot = as_p[0];
                        ma.pos0 = step.pos[0];
                        ma.is_swa = is_swa ? 1 : 0;
                        ma.swa_win = c.swa_win;
                        ma.rel_extent = c.rel_extent;
                        ma.kv_stride = visit_kv_stride[visit];
                        ma.scale = sc;
                        mtl::Dispatch da;
                        da.p = &p_mpp_relpos_gqa_pf;
                        da.grid[0] = (uint32_t)(M_pad / 32); da.grid[1] = 24; da.tg[0] = 128;
                        da.args = {mtl::Arg::B(b_mpp_qh), mtl::Arg::B(b_mpp_rh),
                                   mtl::Arg::B(mod->rel_proj_t_f16_buf, (size_t)ip * 1025 * 16 * 2),
                                   mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn),
                                   mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                                   mtl::Arg::B(b_attn_out), mtl::Arg::V(ma)};
                        ds.push_back(std::move(da));
                    } else {
                        InkAttnArgs aa{};
                        aa.M = M;
                        aa.is_swa = is_swa ? 1 : 0;
                        aa.swa_win = c.swa_win;
                        aa.rel_extent = c.rel_extent;
                        aa.kv_stride = visit_kv_stride[visit];
                        aa.kv_fp8 = 0;
                        aa.scale = 1.0f / std::sqrt((float)c.head_dim);
                        int max_p = 0;
                        for (int m = 0; m < M; ++m) if (step.pos[m] > max_p) max_p = step.pos[m];
                        const int splitk_min_p = is_apple10 ? 64 : 256;
                        if (M <= 32 && max_p >= splitk_min_p) {
                            mtl::Dispatch d1;
                            d1.p = &p_relpos_gqa_splitk;
                            d1.grid[0] = (uint32_t)M; d1.grid[1] = (uint32_t)c.n_kv; d1.grid[2] = 8u; d1.tg[0] = 128;
                            d1.args = {mtl::Arg::B(b_q), mtl::Arg::B(b_r),
                                       mtl::Arg::B(mod->rel_proj_t_f16_buf, (size_t)ip * 1025 * 16 * 2),
                                       mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                                       mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                                       mtl::Arg::B(b_attn_part_o), mtl::Arg::B(b_attn_part_ml), mtl::Arg::V(aa)};
                            ds.push_back(std::move(d1));
                            mtl::Dispatch d2;
                            d2.p = &p_relpos_gqa_splitk_red;
                            d2.grid[0] = (uint32_t)M; d2.grid[1] = (uint32_t)c.n_head; d2.tg[0] = 128;
                            d2.args = {mtl::Arg::B(b_attn_part_o), mtl::Arg::B(b_attn_part_ml),
                                       mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn), mtl::Arg::B(b_attn_out)};
                            ds.push_back(std::move(d2));
                        } else {
                            mtl::Dispatch da;
                            da.p = &p_relpos_gqa;
                            da.grid[0] = (uint32_t)M; da.grid[1] = (uint32_t)c.n_head; da.tg[0] = 128;
                            da.args = {mtl::Arg::B(b_q), mtl::Arg::B(b_r),
                                       mtl::Arg::B(mod->rel_proj_t_f16_buf, (size_t)ip * 1025 * 16 * 2),
                                       mtl::Arg::B(b_sinks), mtl::Arg::B(b_gate_attn),
                                       mtl::Arg::B(kv_k[visit]), mtl::Arg::B(kv_v[visit]),
                                       mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                                       mtl::Arg::B(b_attn_out), mtl::Arg::V(aa)};
                            ds.push_back(std::move(da));
                        }
                    }
                }
            }

            // Output projection wo_ud + FUSED 4-tap attn_sconv + residual/block update (kills 56+ barriers/tok!)
            if (use_mpp && use_mpp_trunk_f16 && M >= mpp_trunk_min_m) {
                push_f32_to_f16(ds, mtl::Arg::B(b_attn_out), mtl::Arg::B(b_mpp_x_f16), M * 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_wo_f16[ip]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(b_wo_raw), M, D, 3072, D, true);
            } else {
                push_trunk_linear(ds, p + ".attn.wo_ud.weight", mtl::Arg::B(b_attn_out), mtl::Arg::B(b_wo_raw), M);
            }
            const bool is_blk = (ip % c.block_size == 0);
            if (is_tree) {
                int s0 = as_p[0], sb_i = is_blk ? 1 : 0;
                mtl::Dispatch d;
                d.p = &p_sconv4_attn_res_tree;
                d.grid[0] = (uint32_t)M; d.grid[1] = (uint32_t)((D + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_wo_raw), mtl::Arg::B(mod->static_sconv_buf, (sc_lbase + InkMetalModel::kSconvAttnOff) * 4),
                          mtl::Arg::B(sc_hist_attn[visit]), mtl::Arg::B(b_tree_parent),
                          mtl::Arg::B(sc_ring_attn[visit]), mtl::Arg::B(b_cur),
                          mtl::Arg::B(b_blocks, (size_t)(is_blk ? n_blocks : 0) * block_stride * 4),
                          mtl::Arg::V(s0), mtl::Arg::V(sb_i)};
                ds.push_back(std::move(d));
                if (is_blk) ++n_blocks;
            } else if (n_active_slots == 1 && M >= 16 && !save_ring) {
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
                if (M > 8) {
                    if (use_mpp && use_mpp_trunk_f16) {
                        const int ci = ip - c.core_start;
                        push_f32_to_f16(ds, mtl::Arg::B(b_cat), mtl::Arg::B(b_mpp_x_f16), M * 2 * D, true);
                        push_mpp_gemm(ds, mtl::Arg::B(w_ctrl_dn_f16[ci][3]), mtl::Arg::B(b_mpp_x_f16),
                                      mtl::Arg::B(b_ctrl_gate), M, c.gate_rank, 2 * D, c.gate_rank, true);
                        push_f32_to_f16(ds, mtl::Arg::B(b_ctrl_gate), mtl::Arg::B(b_mpp_x_f16), M * c.gate_rank, true);
                        push_mpp_gemm(ds, mtl::Arg::B(w_ctrl_gf_f16[ci][3]), mtl::Arg::B(b_mpp_x_f16),
                                      mtl::Arg::B(b_cat), M, 2 * D, c.gate_rank, 2 * D, true);
                        mtl::Dispatch df;
                        df.p = &p_mpp_ctrl_fuse_gf; df.grid[0] = (uint32_t)M; df.tg[0] = 256;
                        df.args = {mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_msgs, (size_t)3 * M_stride * 4),
                                   mtl::Arg::B(b_cat), mod->tw(p + ".mlp_norm.weight"), mtl::Arg::B(b_mlp_in)};
                        ds.push_back(std::move(df));
                    } else {
                        push_trunk_linear(ds, p + ".controllers.mlp.down_weight", mtl::Arg::B(b_cat), mtl::Arg::B(b_ctrl_gate), M);
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
                    }
                } else {
                    push_trunk_linear(ds, p + ".controllers.mlp.down_weight", mtl::Arg::B(b_cat), mtl::Arg::B(b_ctrl_gate), M);
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
            const size_t ex_off = (size_t)ip * c.n_experts * sizeof(InkExpertDev);
            const int mpp_moe_min_m = is_apple10 ? 10 : 16;
            if (use_mpp && M >= mpp_moe_min_m) {
                {
                    mtl::Dispatch d;
                    d.p = &p_router_topk8; d.grid[0] = (uint32_t)M; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_r_logits), mod->tw(p + ".mlp.gate.bias"),
                              mtl::Arg::B(b_topk_idx), mtl::Arg::B(b_topk_w), mtl::Arg::V(c.route_scale)};
                    ds.push_back(std::move(d));
                }
                const int P = M * c.top_k;
                const int allow_128 = (M >= 128) ? 1 : 0;
                const int gu_ci = (layer_homog_gu[ip] == 15) ? 0 : (layer_homog_gu[ip] == 16) ? 1 : 2;
                const int dn_ci = (layer_homog_dn[ip] == 15) ? 0 : (layer_homog_dn[ip] == 16) ? 1 : 2;
                {
                    mtl::Dispatch d;
                    d.p = &p_mpp_moe_plan; d.grid[0] = 1; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_topk_idx), mtl::Arg::V(P), mtl::Arg::V(allow_128),
                              mtl::Arg::B(b_mpp_srt_pair), mtl::Arg::B(b_mpp_pair_pos),
                              mtl::Arg::B(b_mpp_tiles_128), mtl::Arg::B(b_mpp_tiles_64),
                              mtl::Arg::B(b_mpp_tiles_32), mtl::Arg::B(b_mpp_tiles_16),
                              mtl::Arg::B(b_mpp_meta)};
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch dg;
                    dg.p = &p_mpp_moe_gather; dg.ind_off = 4 * sizeof(uint32_t); dg.tg[0] = 256; dg.indirect = &b_mpp_meta;
                    dg.args = {mtl::Arg::B(b_mlp_in), mtl::Arg::B(b_mpp_srt_pair), mtl::Arg::B(b_mpp_xs)};
                    ds.push_back(std::move(dg));
                }
                // All 4 exact-binary tile buckets (128, 64, 32, 16) write disjoint rows of HS -> run 100% CONCURRENTLY!
                if (allow_128) {
                    mtl::Dispatch d128;
                    d128.p = &p_mpp_moe_gu_dir[gu_ci][3]; d128.ind_off = 7 * sizeof(uint32_t); d128.tg[0] = 128; d128.indirect = &b_mpp_meta;
                    d128.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                 mtl::Arg::B(b_mpp_xs), mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_tiles_128)};
                    ds.push_back(std::move(d128));
                }
                {
                    mtl::Dispatch d64;
                    d64.barrier = !allow_128;
                    d64.p = &p_mpp_moe_gu_dir[gu_ci][2]; d64.ind_off = 10 * sizeof(uint32_t); d64.tg[0] = 128; d64.indirect = &b_mpp_meta;
                    d64.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_xs), mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_tiles_64)};
                    ds.push_back(std::move(d64));
                }
                {
                    mtl::Dispatch d32;
                    d32.barrier = false;
                    d32.p = &p_mpp_moe_gu_dir[gu_ci][1]; d32.ind_off = 13 * sizeof(uint32_t); d32.tg[0] = 128; d32.indirect = &b_mpp_meta;
                    d32.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_xs), mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_tiles_32)};
                    ds.push_back(std::move(d32));
                }
                {
                    mtl::Dispatch d16;
                    d16.barrier = false;
                    d16.p = &p_mpp_moe_gu_dir[gu_ci][0]; d16.ind_off = 16 * sizeof(uint32_t); d16.tg[0] = 128; d16.indirect = &b_mpp_meta;
                    d16.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_xs), mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_tiles_16)};
                    ds.push_back(std::move(d16));
                }
                // All 4 w2 buckets (128, 64, 32, 16) write disjoint rows of YE -> run 100% CONCURRENTLY!
                if (allow_128) {
                    mtl::Dispatch d128;
                    d128.p = &p_mpp_moe_dn_dir[dn_ci][3]; d128.ind_off = 7 * sizeof(uint32_t); d128.tg[0] = 128; d128.indirect = &b_mpp_meta;
                    d128.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                 mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_ye), mtl::Arg::B(b_mpp_tiles_128)};
                    ds.push_back(std::move(d128));
                }
                {
                    mtl::Dispatch d64;
                    d64.barrier = !allow_128;
                    d64.p = &p_mpp_moe_dn_dir[dn_ci][2]; d64.ind_off = 10 * sizeof(uint32_t); d64.tg[0] = 128; d64.indirect = &b_mpp_meta;
                    d64.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_ye), mtl::Arg::B(b_mpp_tiles_64)};
                    ds.push_back(std::move(d64));
                }
                {
                    mtl::Dispatch d32;
                    d32.barrier = false;
                    d32.p = &p_mpp_moe_dn_dir[dn_ci][1]; d32.ind_off = 13 * sizeof(uint32_t); d32.tg[0] = 128; d32.indirect = &b_mpp_meta;
                    d32.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_ye), mtl::Arg::B(b_mpp_tiles_32)};
                    ds.push_back(std::move(d32));
                }
                {
                    mtl::Dispatch d16;
                    d16.barrier = false;
                    d16.p = &p_mpp_moe_dn_dir[dn_ci][0]; d16.ind_off = 16 * sizeof(uint32_t); d16.tg[0] = 128; d16.indirect = &b_mpp_meta;
                    d16.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off),
                                mtl::Arg::B(b_mpp_hs), mtl::Arg::B(b_mpp_ye), mtl::Arg::B(b_mpp_tiles_16)};
                    ds.push_back(std::move(d16));
                }
                {
                    mtl::Dispatch dc;
                    dc.p = &p_mpp_moe_combine; dc.grid[0] = (uint32_t)M; dc.grid[1] = 9; dc.tg[0] = 256;
                    dc.args = {mtl::Arg::B(b_mpp_ye), mtl::Arg::B(b_mpp_pair_pos),
                               mtl::Arg::B(b_topk_w), mtl::Arg::B(b_moe_raw)};
                    ds.push_back(std::move(dc));
                }
                if (is_tree) {
                    int s0 = as_p[0];
                    mtl::Dispatch d;
                    d.p = &p_sconv4_mlp_add_tree;
                    d.grid[0] = (uint32_t)M; d.grid[1] = 9; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_moe_raw), mtl::Arg::B(mod->static_sconv_buf, (sc_lbase + InkMetalModel::kSconvMlpOff) * 4),
                              mtl::Arg::B(sc_hist_mlp[visit]), mtl::Arg::B(b_tree_parent),
                              mtl::Arg::B(sc_ring_mlp[visit]), mtl::Arg::B(b_cur), mtl::Arg::V(s0)};
                    ds.push_back(std::move(d));
                } else {
                    push_sconv(b_moe_raw, sc_lbase + InkMetalModel::kSconvMlpOff, b_kd_taps, sc_hist_mlp[visit], sc_ring_mlp[visit], b_moe_out, D, 0, true);
                    {
                        mtl::Dispatch d;
                        d.p = &p_add; d.grid[0] = (uint32_t)((M_stride + 255) / 256); d.tg[0] = 256;
                        d.args = {mtl::Arg::B(b_cur), mtl::Arg::B(b_moe_out), mtl::Arg::B(b_cur), mtl::Arg::V(M_stride)};
                        ds.push_back(std::move(d));
                    }
                }
            } else {
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
                const bool use_moe_mm = (M >= 48);
                if (use_moe_mm) {
                    {
                        mtl::Dispatch d;
                        d.p = (layer_homog_gu[ip] == 15) ? &p_moe_gu_mm_sf4
                            : (layer_homog_gu[ip] == 16) ? &p_moe_gu_mm_mx8 : &p_moe_gu_mm_gen;
                        d.ind_off = 8 * sizeof(uint32_t);
                        d.tg[0] = 256; d.indirect = &b_moe_meta;
                        d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_mlp_in),
                                  mtl::Arg::B(b_moe_tiles), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                                  mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h)};
                        ds.push_back(std::move(d));
                    }
                    {
                        mtl::Dispatch d;
                        d.p = (layer_homog_dn[ip] == 15) ? &p_moe_dn_mm_sf4
                            : (layer_homog_dn[ip] == 16) ? &p_moe_dn_mm_mx8 : &p_moe_dn_mm_gen;
                        d.ind_off = 8 * sizeof(uint32_t);
                        d.tg[0] = 256; d.indirect = &b_moe_meta;
                        d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_topk_w),
                                  mtl::Arg::B(b_moe_tiles), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                                  mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h), mtl::Arg::B(b_moe_yp)};
                        ds.push_back(std::move(d));
                    }
                } else {
                    // Homogeneous layers use lean specialized kernels; SQ-10 mixed {S2,S3,SF4} uses ultra-lean sk23 (branchless 16-lane w2 + 3.25KB LUTs)!
                    constexpr uint32_t kSk23Mask = (1u << 2) | (1u << 3) | (1u << 15);
                    {
                        mtl::Dispatch d;
                        if (layer_homog_gu[ip] == 15)      { d.p = &p_moe_gu_sf4;  d.ind_off = 5 * sizeof(uint32_t); }
                        else if (layer_homog_gu[ip] == 16) { d.p = &p_moe_gu_mx8;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_gu[ip] == 2)  { d.p = &p_moe_gu_sk2;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_gu[ip] == 3)  { d.p = &p_moe_gu_sk3;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_gu[ip] == 4)  { d.p = &p_moe_gu_sk4;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_gu[ip] == 5)  { d.p = &p_moe_gu_sk5;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if ((layer_gu_mask[ip] & ~kSk23Mask) == 0) { d.p = &p_moe_gu_sk23; d.ind_off = 2 * sizeof(uint32_t); }
                        else                               { d.p = &p_moe_gu;      d.ind_off = 2 * sizeof(uint32_t); }
                        d.tg[0] = 256; d.indirect = &b_moe_meta;
                        d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_mlp_in),
                                  mtl::Arg::B(b_moe_uexp), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                                  mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h)};
                        ds.push_back(std::move(d));
                    }
                    {
                        mtl::Dispatch d;
                        if (layer_homog_dn[ip] == 15)      { d.p = &p_moe_dn_sf4;  d.ind_off = 11 * sizeof(uint32_t); }
                        else if (layer_homog_dn[ip] == 16) { d.p = &p_moe_dn_mx8;  d.ind_off = 5 * sizeof(uint32_t); }
                        else if (layer_homog_dn[ip] == 2)  { d.p = &p_moe_dn_sk2;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_dn[ip] == 3)  { d.p = &p_moe_dn_sk3;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_dn[ip] == 4)  { d.p = &p_moe_dn_sk4;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if (layer_homog_dn[ip] == 5)  { d.p = &p_moe_dn_sk5;  d.ind_off = 2 * sizeof(uint32_t); }
                        else if ((layer_dn_mask[ip] & ~kSk23Mask) == 0) { d.p = &p_moe_dn_sk23; d.ind_off = 5 * sizeof(uint32_t); }
                        else                               { d.p = &p_moe_dn;      d.ind_off = 2 * sizeof(uint32_t); }
                        d.tg[0] = 256; d.indirect = &b_moe_meta;
                        d.args = {mod->sec4_arg(), mtl::Arg::B(mod->expert_dev_buf, ex_off), mtl::Arg::B(b_topk_w),
                                  mtl::Arg::B(b_moe_uexp), mtl::Arg::B(b_moe_uoff), mtl::Arg::B(b_moe_ucnt),
                                  mtl::Arg::B(b_moe_pairs), mtl::Arg::B(b_moe_meta), mtl::Arg::B(b_moe_h), mtl::Arg::B(b_moe_yp)};
                        ds.push_back(std::move(d));
                    }
                }
                if (is_tree) {
                    {
                        mtl::Dispatch d;
                        d.p = &p_moe_red8; d.grid[0] = (uint32_t)M; d.grid[1] = 9; d.tg[0] = 256;
                        d.args = {mtl::Arg::B(b_moe_yp), mtl::Arg::B(b_moe_raw)};
                        ds.push_back(std::move(d));
                    }
                    int s0 = as_p[0];
                    mtl::Dispatch d;
                    d.p = &p_sconv4_mlp_add_tree;
                    d.grid[0] = (uint32_t)M; d.grid[1] = 9; d.tg[0] = 256;
                    d.args = {mtl::Arg::B(b_moe_raw), mtl::Arg::B(mod->static_sconv_buf, (sc_lbase + InkMetalModel::kSconvMlpOff) * 4),
                              mtl::Arg::B(sc_hist_mlp[visit]), mtl::Arg::B(b_tree_parent),
                              mtl::Arg::B(sc_ring_mlp[visit]), mtl::Arg::B(b_cur), mtl::Arg::V(s0)};
                    ds.push_back(std::move(d));
                } else if (n_active_slots == 1 && M >= 16 && !save_ring) {
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
                if (is_tree) {
                    mtl::Dispatch d;
                    d.p = &p_dsp_ctx_kv_tree; d.grid[0] = (uint32_t)M; d.grid[1] = 8; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_k_raw), mtl::Arg::B(b_v_raw), mod->dw(dp + ".self_attn.k_norm_ctx.weight"),
                              mtl::Arg::B(dsp_tree_k[dl]), mtl::Arg::B(dsp_tree_v[dl]),
                              mtl::Arg::B(b_pos), mtl::Arg::V(c.dspark_rope_theta)};
                    ds.push_back(std::move(d));
                } else {
                    mtl::Dispatch d;
                    d.p = &p_dsp_ctx_kv; d.grid[0] = (uint32_t)M; d.grid[1] = 8; d.tg[0] = 128;
                    d.args = {mtl::Arg::B(b_k_raw), mtl::Arg::B(b_v_raw), mod->dw(dp + ".self_attn.k_norm_ctx.weight"),
                              mtl::Arg::B(dsp_ctx_k[dl]), mtl::Arg::B(dsp_ctx_v[dl]),
                              mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos),
                              mtl::Arg::V(max_ctx), mtl::Arg::V(c.dspark_rope_theta)};
                    ds.push_back(std::move(d));
                }
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
                if (!logits_out && argmax_out && n_lg == 1 && (uti.fmt == FMT_Q8ROW || uti.fmt == FMT_BF16 || uti.fmt == FMT_S6)) {
                    const int V = c.vocab_unpadded;
                    const int rpt = (uti.fmt == FMT_S6) ? 16 : 32;
                    const int nparts = (V + rpt - 1) / rpt;
                    mtl::Dispatch du;
                    du.p = (uti.fmt == FMT_Q8ROW) ? &p_unemb_argmax_q8
                         : (uti.fmt == FMT_BF16)  ? &p_unemb_argmax_bf16 : &p_unemb_argmax_sk6;
                    du.grid[0] = (uint32_t)nparts; du.tg[0] = 256;
                    du.args = {mod->tw("model.llm.unembed.weight"), mtl::Arg::B(b_h_norm),
                               mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::V(V), mtl::Arg::V(D)};
                    ds.push_back(std::move(du));
                    mtl::Dispatch dr;
                    dr.p = &p_argmax_part; dr.grid[0] = 1; dr.tg[0] = 256;
                    dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::B(b_argmax), mtl::Arg::V(nparts)};
                    ds.push_back(std::move(dr));
                } else if (!logits_out && argmax_out && n_lg <= 4 && (uti.fmt == FMT_Q8ROW || uti.fmt == FMT_BF16 || uti.fmt == FMT_S6)) {
                    const int V = c.vocab_unpadded;
                    const int nparts = (V + 15) / 16;
                    mtl::Dispatch du;
                    du.p = (uti.fmt == FMT_Q8ROW) ? &p_unemb_argmax_m_q8
                         : (uti.fmt == FMT_BF16)  ? &p_unemb_argmax_m_bf16 : &p_unemb_argmax_m_sk6;
                    du.grid[0] = (uint32_t)nparts; du.tg[0] = 256;
                    du.args = {mod->tw("model.llm.unembed.weight"), mtl::Arg::B(b_h_norm),
                               mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::V(V), mtl::Arg::V(D), mtl::Arg::V(n_lg), mtl::Arg::V(nparts)};
                    ds.push_back(std::move(du));
                    mtl::Dispatch dr;
                    dr.p = &p_argmax_part_m; dr.grid[0] = (uint32_t)n_lg; dr.tg[0] = 256;
                    dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                               mtl::Arg::B(b_argmax), mtl::Arg::V(nparts)};
                    ds.push_back(std::move(dr));
                } else {
                    // For M=5..8 S6_K unembed, use mm8 (1.56 ms vs 2.60 ms mv!); for M=9..16 S6_K, split into two mm8 calls (3.1 ms vs 5.25 ms!)
                    if (uti.fmt == FMT_S6 && n_lg >= 5 && n_lg <= 16) {
                        for (int m0 = 0; m0 < n_lg; m0 += 8) {
                            const int mm = std::min(8, n_lg - m0);
                            mtl::Dispatch du;
                            du.p = &p_lin_sk6_mm8;
                            du.grid[0] = (uint32_t)((c.vocab_unpadded + 63) / 64);
                            du.tg[0] = 256;
                            du.args = {mod->tw("model.llm.unembed.weight"),
                                       mtl::Arg::B(b_h_norm, (size_t)m0 * D * 4),
                                       mtl::Arg::B(b_logits, (size_t)m0 * c.vocab_unpadded * 4),
                                       mtl::Arg::V(c.vocab_unpadded), mtl::Arg::V(D), mtl::Arg::V(mm)};
                            ds.push_back(std::move(du));
                        }
                    } else {
                        push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(b_h_norm), mtl::Arg::B(b_logits),
                                          n_lg, c.vocab_unpadded);
                    }
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
        }

        // Execute ALL 48 visits in ONE Metal command buffer!
        last_gpu_ms = cx->run_graph(ds) * 1e3;

        // Update latest f_rnn per slot if taps were captured (tree mode commits via commit_tree_path!)
        if (step.capture_taps && use_dspark && !is_tree) {
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
    // Generates `n_steps` (1..7) draft tokens in out_draft[0..n_steps-1] conditioned on `anchor_tok` at `anchor_pos`!
    int dspark_draft_block8(int slot, int32_t anchor_tok, int anchor_pos, int32_t* out_draft, int n_steps = 7) {
        n_steps = std::max(1, std::min(7, n_steps));
        if (!use_dspark || anchor_pos <= 0 || anchor_pos + n_steps + 1 >= max_ctx) return 0;
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

        // 2. 3 non-causal block-attention layers (in-place residual add eliminates 6 redundant copies & 6 barriers!)
        for (int il = 0; il < 3; ++il) {
            std::string lp = "dspark.layers." + std::to_string(il);
            {
                mtl::Dispatch d;
                d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".input_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_q_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_q), 8, 3072, D, 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_k_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_k), 8, 1024, D, 1024, false);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_v_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_v), 8, 1024, D, 1024, false);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.q_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_q), 8, true);
                push_dspark_linear(ds, lp + ".self_attn.k_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_k), 8, false);
                push_dspark_linear(ds, lp + ".self_attn.v_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_v), 8, false);
            }
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
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_ao), mtl::Arg::B(b_mpp_x_f16), 8 * 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_o_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 3072, D, true);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.o_proj.weight", mtl::Arg::B(bd_ao), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
            // MLP
            {
                mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".post_attention_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_g_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mg), 8, 9216, D, 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_u_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mu), 8, 9216, D, 9216, false);
            } else {
                push_dspark_linear(ds, lp + ".mlp.gate_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mg), 8, true);
                push_dspark_linear(ds, lp + ".mlp.up_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mu), 8, false);
            }
            {
                int n8i = 8 * 9216;
                mtl::Dispatch d; d.p = &p_dsp_swiglu; d.grid[0] = (uint32_t)((n8i + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_mg), mtl::Arg::B(bd_mu), mtl::Arg::B(bd_mh), mtl::Arg::V(n8i)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_mh), mtl::Arg::B(b_mpp_x_f16), 8 * 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_d_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 9216, D, true);
            } else {
                push_dspark_linear(ds, lp + ".mlp.down_proj.weight", mtl::Arg::B(bd_mh), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
        }

        // 3. Final norm (linearity fusion: unembed(cur[s] + corr) == base_logits[s] + bias_logits -> 7 unembeds vs 14!)
        {
            mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
            d.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.norm.weight"), mtl::Arg::B(bd_cur), mtl::Arg::V(D)};
            ds.push_back(std::move(d));
        }

        // 4. EXACT Full-Vocab Low-Rank Markov Chain (M_eff = unembed @ hidden_out.weight, 51 MiB Q8_ROW!)
        //    - ONE batched mm8 unembed for all 7 base rows (~1.5 ms)
        //    - 7 exact full-vocab (200,058 tokens!) Markov steps via M_eff (~0.10 ms/step -> 0.7 ms total, 0% alpha loss!)
        const auto& mw1_ti = mod->man.get_d("dspark.markov_head.markov_w1.weight");
        if (use_dspark_meff) {
            {
                mtl::Dispatch db;
                db.p = &p_dsp_add_bias7; db.grid[0] = (uint32_t)n_steps; db.tg[0] = 256;
                db.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.markov_head.hidden_out.bias"), mtl::Arg::B(bd_h)};
                ds.push_back(std::move(db));
            }
            push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(bd_h), mtl::Arg::B(b_logits), n_steps, c.vocab_unpadded);
        }
        const int V = c.vocab_unpadded;
        const int nparts_meff = (V + 31) / 32;
        for (int s = 1; s <= n_steps; ++s) {
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
            if (use_dspark_meff) {
                const size_t boff = (size_t)(s - 1) * (size_t)V * sizeof(float);
                mtl::Dispatch dm;
                dm.p = &p_dsp_meff_argmax_q8;
                dm.grid[0] = (uint32_t)nparts_meff; dm.tg[0] = 256;
                dm.args = {mtl::Arg::B(b_logits, boff), mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                           mtl::Arg::B(bd_latent), mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx), mtl::Arg::V(V)};
                ds.push_back(std::move(dm));
                mtl::Dispatch dr;
                dr.p = &p_argmax_part; dr.grid[0] = 1; dr.tg[0] = 256;
                dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::B(bd_tok_chain, (size_t)s * 4), mtl::Arg::V(nparts_meff)};
                ds.push_back(std::move(dr));
            } else {
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
                push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(bd_comb), mtl::Arg::B(bd_logits),
                                  1, c.vocab_unpadded);
                mtl::Dispatch da; da.p = &p_argmax_rows; da.grid[0] = 1; da.tg[0] = 256;
                da.args = {mtl::Arg::B(bd_logits), mtl::Arg::B(bd_tok_chain, (size_t)s * 4), mtl::Arg::V(c.vocab_unpadded)};
                ds.push_back(std::move(da));
            }
        }

        cx->run_graph(ds);
        const int32_t* chain = bd_tok_chain.as<int32_t>();
        for (int i = 0; i < n_steps; ++i) out_draft[i] = chain[i + 1];
        return n_steps;
    }

    // O(1) Concurrent GPU Commit of winning tree path[0..L-1] into KV, ShortConv, DSpark ctx KV, and f_rnn!
    void commit_tree_path(int slot, int anchor_pos, const std::vector<int32_t>& path_nodes) {
        const int L = (int)path_nodes.size();
        if (L <= 0 || L > 32) return;
        const auto& c = mod->man.c;
        const int D = c.hidden;
        std::memcpy(b_tree_path.ptr, path_nodes.data(), (size_t)L * sizeof(int32_t));

        std::vector<mtl::Dispatch> ds;
        ds.reserve(c.n_visits * 5 + 4);
        int C_kv = c.n_kv * c.head_dim;
        int C_d = D;
        for (int v = 0; v < c.n_visits; ++v) {
            const int ip = c.phys_for_visit(v);
            int swa_i = c.is_swa[ip] ? 1 : 0;
            {
                mtl::Dispatch d;
                d.barrier = false;
                d.p = &p_kv_tree_commit_f16;
                d.grid[0] = (uint32_t)L; d.grid[1] = (uint32_t)c.n_kv; d.tg[0] = 128;
                d.args = {mtl::Arg::B(tree_k[v]), mtl::Arg::B(tree_v[v]),
                          mtl::Arg::B(kv_k[v]), mtl::Arg::B(kv_v[v]),
                          mtl::Arg::B(b_tree_path), mtl::Arg::V(slot), mtl::Arg::V(anchor_pos),
                          mtl::Arg::V(swa_i), mtl::Arg::V(visit_kv_stride[v])};
                ds.push_back(std::move(d));
            }
            auto push_sc_commit = [&](const mtl::Buffer& raw_b, const mtl::Buffer& hist_b, int C_ch) {
                mtl::Dispatch d;
                d.barrier = false;
                d.p = &p_sconv4_tree_commit;
                d.grid[0] = (uint32_t)((C_ch + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(raw_b), mtl::Arg::B(hist_b), mtl::Arg::B(b_tree_path),
                          mtl::Arg::V(slot), mtl::Arg::V(L), mtl::Arg::V(C_ch)};
                ds.push_back(std::move(d));
            };
            push_sc_commit(sc_ring_k[v],    sc_hist_k[v],    C_kv);
            push_sc_commit(sc_ring_v[v],    sc_hist_v[v],    C_kv);
            push_sc_commit(sc_ring_attn[v], sc_hist_attn[v], C_d);
            push_sc_commit(sc_ring_mlp[v],  sc_hist_mlp[v],  C_d);
        }
        if (use_dspark) {
            for (int dl = 0; dl < 3; ++dl) {
                mtl::Dispatch d;
                d.barrier = false;
                d.p = &p_dsp_ctx_kv_tree_commit;
                d.grid[0] = (uint32_t)L; d.grid[1] = 8; d.tg[0] = 128;
                d.args = {mtl::Arg::B(dsp_tree_k[dl]), mtl::Arg::B(dsp_tree_v[dl]),
                          mtl::Arg::B(dsp_ctx_k[dl]), mtl::Arg::B(dsp_ctx_v[dl]),
                          mtl::Arg::B(b_tree_path), mtl::Arg::V(slot), mtl::Arg::V(anchor_pos),
                          mtl::Arg::V(max_ctx)};
                ds.push_back(std::move(d));
            }
        }
        cx->run_graph(ds);
        if (use_dspark) {
            const int last_node = path_nodes.back();
            std::memcpy(dsp_frnn_slot.as<float>() + (size_t)slot * D,
                        dsp_frnn_tmp.as<float>() + (size_t)last_node * D,
                        (size_t)D * sizeof(float));
        }
    }

    // Multi-Branch Tree DSpark Drafter (Exact 0.00000 KLD, zero extra 3-layer DSpark or unembed compute!)
    // Builds a probability-optimal draft tree of up to `tree_budget` nodes (e.g. 12, 16, 24; <= 32) rooted at anchor_tok.
    int dspark_draft_tree(int slot, int32_t anchor_tok, int anchor_pos, int tree_budget, InkDraftTree& out) {
        out.M = 0;
        out.tok_ids.clear(); out.parents.clear(); out.depths.clear(); out.anc_masks.clear();
        tree_budget = std::max(2, std::min(32, tree_budget));
        if (!use_dspark || !use_dspark_meff || anchor_pos <= 0) return 0;
        const auto& c = mod->man.c;
        const int main_len = std::min(7, std::min(tree_budget - 1, max_ctx - anchor_pos - 2));
        if (main_len <= 0) return 0;
        const int D = c.hidden;
        const int V = c.vocab_unpadded;
        const int nparts_meff = (V + 31) / 32;

        // Seed tok_chain[0] = anchor_tok and zero initial Markov state bd_tree_states[0]
        bd_tok_chain.as<int32_t>()[0] = anchor_tok;
        std::memset(bd_tree_states.ptr, 0, 256 * sizeof(float));

        std::vector<mtl::Dispatch> ds;
        ds.reserve(256);
        auto push_copy = [&](const mtl::Arg& src, const mtl::Arg& dst, int n_floats) {
            mtl::Dispatch d;
            d.p = &p_copy_f32;
            d.grid[0] = (uint32_t)((n_floats + 255) / 256); d.tg[0] = 256;
            d.args = {src, dst, mtl::Arg::V(n_floats)};
            ds.push_back(std::move(d));
        };

        // Phase A: 1) Init Block-8 + 3 non-causal DSpark layers + Final norm + batched unembed(main_len)
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
        for (int il = 0; il < 3; ++il) {
            std::string lp = "dspark.layers." + std::to_string(il);
            {
                mtl::Dispatch d;
                d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".input_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_q_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_q), 8, 3072, D, 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_k_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_k), 8, 1024, D, 1024, false);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_v_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_v), 8, 1024, D, 1024, false);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.q_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_q), 8, true);
                push_dspark_linear(ds, lp + ".self_attn.k_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_k), 8, false);
                push_dspark_linear(ds, lp + ".self_attn.v_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_v), 8, false);
            }
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
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_ao), mtl::Arg::B(b_mpp_x_f16), 8 * 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_o_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 3072, D, true);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.o_proj.weight", mtl::Arg::B(bd_ao), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
            {
                mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".post_attention_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_g_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mg), 8, 9216, D, 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_u_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mu), 8, 9216, D, 9216, false);
            } else {
                push_dspark_linear(ds, lp + ".mlp.gate_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mg), 8, true);
                push_dspark_linear(ds, lp + ".mlp.up_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mu), 8, false);
            }
            {
                int n8i = 8 * 9216;
                mtl::Dispatch d; d.p = &p_dsp_swiglu; d.grid[0] = (uint32_t)((n8i + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_mg), mtl::Arg::B(bd_mu), mtl::Arg::B(bd_mh), mtl::Arg::V(n8i)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_mh), mtl::Arg::B(b_mpp_x_f16), 8 * 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_d_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 9216, D, true);
            } else {
                push_dspark_linear(ds, lp + ".mlp.down_proj.weight", mtl::Arg::B(bd_mh), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
        }
        {
            mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
            d.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.norm.weight"), mtl::Arg::B(bd_cur), mtl::Arg::V(D)};
            ds.push_back(std::move(d));
        }
        {
            mtl::Dispatch db;
            db.p = &p_dsp_add_bias7; db.grid[0] = (uint32_t)main_len; db.tg[0] = 256;
            db.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.markov_head.hidden_out.bias"), mtl::Arg::B(bd_h)};
            ds.push_back(std::move(db));
        }
        push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(bd_h), mtl::Arg::B(b_logits), main_len, V);

        // Phase A: 2) Main chain 1..main_len, saving bd_tree_states[1..main_len] AND full-vocab Top-3!
        const auto& mw1_ti = mod->man.get_d("dspark.markov_head.markov_w1.weight");
        for (int s = 1; s <= main_len; ++s) {
            const size_t sin_off  = (size_t)(s - 1) * 256ul * sizeof(float);
            const size_t sout_off = (size_t)s * 256ul * sizeof(float);
            int tidx = s - 1;
            {
                mtl::Dispatch d;
                d.grid[0] = (5120 + 255) / 256; d.tg[0] = 256;
                if (mw1_ti.fmt == FMT_BF16) {
                    d.p = &p_dsp_mjoint_tree_bf16;
                    d.args = {mtl::Arg::B(bd_tree_states, sin_off), mod->dw("dspark.markov_head.markov_w1.weight"),
                              mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                              mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx), mtl::Arg::V(s)};
                } else {
                    d.p = &p_dsp_mjoint_tree_q8;
                    d.args = {mtl::Arg::B(bd_tree_states, sin_off), mod->dw("dspark.markov_head.markov_w1.weight"),
                              mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                              mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx), mtl::Arg::V(s),
                              mtl::Arg::V(c.vocab_padded)};
                }
                ds.push_back(std::move(d));
            }
            push_dspark_linear(ds, "dspark.markov_head.rnn_proj.weight", mtl::Arg::B(bd_joint), mtl::Arg::B(bd_rproj), 1);
            {
                mtl::Dispatch d; d.p = &p_dsp_mrnn_tree; d.grid[0] = 1; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_rproj), mod->dw("dspark.markov_head.rnn_proj.bias"),
                          mtl::Arg::B(bd_tree_states, sin_off), mtl::Arg::B(bd_tree_states, sout_off), mtl::Arg::B(bd_latent)};
                ds.push_back(std::move(d));
            }
            {
                const size_t boff = (size_t)(s - 1) * (size_t)V * sizeof(float);
                mtl::Dispatch dm;
                dm.p = &p_dsp_meff_top3_q8; dm.grid[0] = (uint32_t)nparts_meff; dm.tg[0] = 256;
                dm.args = {mtl::Arg::B(b_logits, boff), mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                           mtl::Arg::B(bd_latent), mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx), mtl::Arg::V(V)};
                ds.push_back(std::move(dm));
                mtl::Dispatch dr;
                dr.p = &p_dsp_top3_red; dr.grid[0] = 1; dr.tg[0] = 256;
                dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::B(bd_tok_chain, (size_t)s * 4),
                           mtl::Arg::B(bd_tree_top3_i, (size_t)(s - 1) * 3ul * 4),
                           mtl::Arg::B(bd_tree_top3_v, (size_t)(s - 1) * 3ul * 4),
                           mtl::Arg::V(nparts_meff)};
                ds.push_back(std::move(dr));
            }
        }
        cx->run_graph(ds);

        // Unified Probability-Optimal Tree Planner:
        // Competes BOTH main-chain depths (s=2..main_len) AND branches (top2/top3 + continuations) in ONE
        // expected-yield priority queue! Automatically produces pure 7-deep chains on confident code (0 Phase-B!)
        // and compact branching trees on uncertain chat/math (same M=6..8 verify cost, higher alpha!).
        const int32_t* t3i = bd_tree_top3_i.as<int32_t>();
        const float*   t3v = bd_tree_top3_v.as<float>();
        out.tok_ids.push_back(anchor_tok);
        out.parents.push_back(-1);
        out.depths.push_back(0);
        out.anc_masks.push_back(1u);

        // Always include depth-1 top-1 (node 1)
        out.tok_ids.push_back(t3i[0]);
        out.parents.push_back(0);
        out.depths.push_back(1);
        out.anc_masks.push_back(3u);

        const int rem_budget = tree_budget - 2;
        if (rem_budget <= 0) { out.M = (int)out.tok_ids.size(); return out.M; }

        struct BrCand { int s; int rank; int32_t root_tok; double root_p; int count = 0; };
        std::vector<BrCand> cands;
        cands.reserve(main_len * 2);
        struct NodeChoice { double score; int is_main; int idx; int off; };
        std::vector<NodeChoice> choices;
        choices.reserve(80);

        double surv = 1.0;
        for (int s = 1; s <= main_len; ++s) {
            const float v0 = t3v[(s - 1) * 3 + 0];
            const float v1 = t3v[(s - 1) * 3 + 1];
            const float v2 = t3v[(s - 1) * 3 + 2];
            const int32_t i0 = t3i[(s - 1) * 3 + 0];
            const int32_t i1 = t3i[(s - 1) * 3 + 1];
            const int32_t i2 = t3i[(s - 1) * 3 + 2];
            const double e1 = std::exp(std::max(-20.0, (double)(v1 - v0) / 1.25));
            const double e2 = std::exp(std::max(-20.0, (double)(v2 - v0) / 1.25));
            const double z = 1.0 + e1 + e2 + 0.15;
            const double p0 = 1.0 / z, p1 = e1 / z, p2 = e2 / z;
            if (s >= 2) choices.push_back({surv * p0, 1, s, 0});
            if (i1 != i0) {
                const int ci = (int)cands.size();
                cands.push_back({s, 1, i1, surv * p1, 0});
                double w = surv * p1;
                for (int off = 0; off <= main_len - s; ++off) {
                    choices.push_back({w, 0, ci, off});
                    w *= 0.65;
                }
            }
            if (i2 != i0 && i2 != i1) {
                const int ci = (int)cands.size();
                cands.push_back({s, 2, i2, surv * p2, 0});
                double w = surv * p2;
                for (int off = 0; off <= main_len - s; ++off) {
                    choices.push_back({w, 0, ci, off});
                    w *= 0.65;
                }
            }
            surv *= p0;
        }
        std::sort(choices.begin(), choices.end(), [](const NodeChoice& a, const NodeChoice& b) {
            return a.score > b.score;
        });

        int chosen_main = 1, used_nodes = 0, active_branches = 0;
        for (const auto& ch : choices) {
            if (used_nodes >= rem_budget) break;
            if (ch.is_main) {
                if (ch.idx <= chosen_main) continue;
                chosen_main = ch.idx;
                ++used_nodes;
            } else {
                const int ci = ch.idx;
                if (cands[ci].s - 1 > chosen_main) continue;
                if (cands[ci].count == 0) {
                    if (active_branches >= 8) continue;
                    ++active_branches;
                }
                if (ch.off + 1 <= cands[ci].count) continue;
                cands[ci].count = ch.off + 1;
                ++used_nodes;
            }
        }

        // Append selected main-chain depths 2..chosen_main (nodes 2..chosen_main)
        for (int s = 2; s <= chosen_main; ++s) {
            out.tok_ids.push_back(t3i[(s - 1) * 3 + 0]);
            out.parents.push_back(s - 1);
            out.depths.push_back(s);
            out.anc_masks.push_back(out.anc_masks[s - 1] | (1u << s));
        }

        // Collect selected branches; run Phase B ONLY for continuations (count > 1)!
        std::vector<BrCand> sel;
        int32_t* btoks = bd_tree_btoks.as<int32_t>();
        bool need_phase_b = false;
        for (const auto& bc : cands) {
            if (bc.count > 0) {
                const int bi = (int)sel.size();
                btoks[bi * 8 + 0] = bc.root_tok;
                if (bc.count > 1) need_phase_b = true;
                sel.push_back(bc);
            }
        }

        if (need_phase_b) {
            std::vector<mtl::Dispatch> dsb;
            dsb.reserve(128);
            const size_t scratch_st_off = 8ul * 256ul * sizeof(float);
            for (int bi = 0; bi < (int)sel.size(); ++bi) {
                const auto& bc = sel[bi];
                const int n_cont = bc.count - 1;
                if (n_cont <= 0) continue;
                for (int k = 1; k <= n_cont; ++k) {
                    const int step_s = bc.s + k;
                    const size_t sin_off = (k == 1) ? ((size_t)bc.s * 256ul * sizeof(float)) : scratch_st_off;
                    int tidx = bi * 8 + (k - 1);
                    {
                        mtl::Dispatch d;
                        d.grid[0] = (5120 + 255) / 256; d.tg[0] = 256;
                        if (mw1_ti.fmt == FMT_BF16) {
                            d.p = &p_dsp_mjoint_tree_bf16;
                            d.args = {mtl::Arg::B(bd_tree_states, sin_off), mod->dw("dspark.markov_head.markov_w1.weight"),
                                      mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                                      mtl::Arg::B(bd_tree_btoks), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx), mtl::Arg::V(step_s)};
                        } else {
                            d.p = &p_dsp_mjoint_tree_q8;
                            d.args = {mtl::Arg::B(bd_tree_states, sin_off), mod->dw("dspark.markov_head.markov_w1.weight"),
                                      mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                                      mtl::Arg::B(bd_tree_btoks), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx), mtl::Arg::V(step_s),
                                      mtl::Arg::V(c.vocab_padded)};
                        }
                        dsb.push_back(std::move(d));
                    }
                    push_dspark_linear(dsb, "dspark.markov_head.rnn_proj.weight", mtl::Arg::B(bd_joint), mtl::Arg::B(bd_rproj), 1);
                    {
                        mtl::Dispatch d; d.p = &p_dsp_mrnn_tree; d.grid[0] = 1; d.tg[0] = 256;
                        d.args = {mtl::Arg::B(bd_rproj), mod->dw("dspark.markov_head.rnn_proj.bias"),
                                  mtl::Arg::B(bd_tree_states, sin_off), mtl::Arg::B(bd_tree_states, scratch_st_off), mtl::Arg::B(bd_latent)};
                        dsb.push_back(std::move(d));
                    }
                    {
                        const size_t boff = (size_t)(step_s - 1) * (size_t)V * sizeof(float);
                        mtl::Dispatch dm;
                        dm.p = &p_dsp_meff_argmax_q8; dm.grid[0] = (uint32_t)nparts_meff; dm.tg[0] = 256;
                        dm.args = {mtl::Arg::B(b_logits, boff), mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                                   mtl::Arg::B(bd_latent), mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx), mtl::Arg::V(V)};
                        dsb.push_back(std::move(dm));
                        mtl::Dispatch dr;
                        dr.p = &p_argmax_part; dr.grid[0] = 1; dr.tg[0] = 256;
                        dr.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                                   mtl::Arg::B(bd_tree_btoks, (size_t)(bi * 8 + k) * sizeof(int32_t)), mtl::Arg::V(nparts_meff)};
                        dsb.push_back(std::move(dr));
                    }
                }
            }
            cx->run_graph(dsb);
        }

        // Assemble final topologically-sorted tree (unique children per parent guaranteed!)
        for (int bi = 0; bi < (int)sel.size(); ++bi) {
            const auto& bc = sel[bi];
            int par = bc.s - 1;
            for (int k = 0; k < bc.count; ++k) {
                if ((int)out.tok_ids.size() >= tree_budget) break;
                const int32_t tok = btoks[bi * 8 + k];
                int dup = -1;
                for (int idx = par + 1; idx < (int)out.tok_ids.size(); ++idx) {
                    if (out.parents[idx] == par && out.tok_ids[idx] == tok) { dup = idx; break; }
                }
                if (dup >= 0) {
                    par = dup;
                    continue;
                }
                const int nid = (int)out.tok_ids.size();
                out.tok_ids.push_back(tok);
                out.parents.push_back(par);
                out.depths.push_back(bc.s + k);
                out.anc_masks.push_back(out.anc_masks[par] | (1u << nid));
                par = nid;
            }
        }
        out.M = (int)out.tok_ids.size();
        out.is_linear = true;
        for (int i = 1; i < out.M; ++i) if (out.parents[i] != i - 1) { out.is_linear = false; break; }
        return out.M;
    }

    // True W=4 Parallel On-GPU Markov Beam Tree + Sparse-MoE Bayes-Optimal Dynamic Tree Sizing!
    // - Runs 3-layer DSpark + batched unembed + W=4 7-step Parallel Beam Search in ONE SINGLE Metal CB (0 host sync!)
    // - Reads 51 MiB M_eff ONCE across all 4 beams (~0.14 ms/depth) -> produces 52-node exact beam pool
    // - Selects Bayes-optimal M* maximizing (1 + sum p_k) / (T_draft + T_verify(M)) on Apple Silicon Sparse MoE!
    int dspark_draft_beam_tree(int slot, int32_t anchor_tok, int anchor_pos, int max_budget,
                               bool dynamic_bayes, InkDraftTree& out, int max_depth_cap = 7) {
        out.M = 0; out.is_linear = false; out.expected_gain = 1.0f;
        out.tok_ids.clear(); out.parents.clear(); out.depths.clear(); out.anc_masks.clear();
        max_budget = std::max(2, std::min(32, max_budget));
        if (!use_dspark || !use_dspark_meff || anchor_pos <= 0) return 0;
        const auto& c = mod->man.c;
        const int max_depth = std::min(max_depth_cap, std::min(7, std::min(max_budget - 1, max_ctx - anchor_pos - 2)));
        if (max_depth <= 0) return 0;
        const int D = c.hidden;
        const int V = c.vocab_unpadded;
        const int nparts = (V + 31) / 32;

        bd_tok_chain.as<int32_t>()[0] = anchor_tok;
        std::memset(bd_tree_states.ptr, 0, 256 * sizeof(float));

        std::vector<mtl::Dispatch> ds;
        ds.reserve(256);

        // 1. Init Block-8 + 3 non-causal DSpark layers + Final norm + batched unembed(max_depth)
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
        for (int il = 0; il < 3; ++il) {
            std::string lp = "dspark.layers." + std::to_string(il);
            {
                mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".input_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_q_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_q), 8, 3072, D, 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_k_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_k), 8, 1024, D, 1024, false);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_v_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_v), 8, 1024, D, 1024, false);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.q_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_q), 8, true);
                push_dspark_linear(ds, lp + ".self_attn.k_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_k), 8, false);
                push_dspark_linear(ds, lp + ".self_attn.v_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_v), 8, false);
            }
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
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_ao), mtl::Arg::B(b_mpp_x_f16), 8 * 3072, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_o_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 3072, D, true);
            } else {
                push_dspark_linear(ds, lp + ".self_attn.o_proj.weight", mtl::Arg::B(bd_ao), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
            {
                mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mod->dw(lp + ".post_attention_layernorm.weight"), mtl::Arg::B(bd_h), mtl::Arg::V(D)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_h), mtl::Arg::B(b_mpp_x_f16), 8 * D, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_g_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mg), 8, 9216, D, 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_u_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_mu), 8, 9216, D, 9216, false);
            } else {
                push_dspark_linear(ds, lp + ".mlp.gate_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mg), 8, true);
                push_dspark_linear(ds, lp + ".mlp.up_proj.weight", mtl::Arg::B(bd_h), mtl::Arg::B(bd_mu), 8, false);
            }
            {
                int n8i = 8 * 9216;
                mtl::Dispatch d; d.p = &p_dsp_swiglu; d.grid[0] = (uint32_t)((n8i + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_mg), mtl::Arg::B(bd_mu), mtl::Arg::B(bd_mh), mtl::Arg::V(n8i)};
                ds.push_back(std::move(d));
            }
            if (use_mpp && use_dspark_f16) {
                push_f32_to_f16(ds, mtl::Arg::B(bd_mh), mtl::Arg::B(b_mpp_x_f16), 8 * 9216, true);
                push_mpp_gemm(ds, mtl::Arg::B(w_dsp_d_f16[il]), mtl::Arg::B(b_mpp_x_f16), mtl::Arg::B(bd_proj), 8, D, 9216, D, true);
            } else {
                push_dspark_linear(ds, lp + ".mlp.down_proj.weight", mtl::Arg::B(bd_mh), mtl::Arg::B(bd_proj), 8);
            }
            {
                int n8d = 8 * D;
                mtl::Dispatch d; d.p = &p_add; d.grid[0] = (uint32_t)((n8d + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(bd_cur), mtl::Arg::B(bd_proj), mtl::Arg::B(bd_cur), mtl::Arg::V(n8d)};
                ds.push_back(std::move(d));
            }
        }
        {
            mtl::Dispatch d; d.p = &p_rms_norm; d.grid[0] = 8; d.tg[0] = 256;
            d.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.norm.weight"), mtl::Arg::B(bd_cur), mtl::Arg::V(D)};
            ds.push_back(std::move(d));
        }
        {
            mtl::Dispatch db;
            db.p = &p_dsp_add_bias7; db.grid[0] = (uint32_t)max_depth; db.tg[0] = 256;
            db.args = {mtl::Arg::B(bd_cur), mod->dw("dspark.markov_head.hidden_out.bias"), mtl::Arg::B(bd_h)};
            ds.push_back(std::move(db));
        }
        push_trunk_linear(ds, "model.llm.unembed.weight", mtl::Arg::B(bd_h), mtl::Arg::B(b_logits), max_depth, V);

        // 2. Depth s = 1: single beam -> Top-4 -> seeds 4 parallel beams + pool[0..3]!
        const auto& mw1_ti = mod->man.get_d("dspark.markov_head.markov_w1.weight");
        {
            int tidx0 = 0, s1 = 1;
            mtl::Dispatch d;
            d.grid[0] = (5120 + 255) / 256; d.tg[0] = 256;
            if (mw1_ti.fmt == FMT_BF16) {
                d.p = &p_dsp_mjoint_tree_bf16;
                d.args = {mtl::Arg::B(bd_tree_states, 0), mod->dw("dspark.markov_head.markov_w1.weight"),
                          mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                          mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx0), mtl::Arg::V(s1)};
            } else {
                d.p = &p_dsp_mjoint_tree_q8;
                d.args = {mtl::Arg::B(bd_tree_states, 0), mod->dw("dspark.markov_head.markov_w1.weight"),
                          mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                          mtl::Arg::B(bd_tok_chain), mtl::Arg::B(bd_joint), mtl::Arg::V(tidx0), mtl::Arg::V(s1),
                          mtl::Arg::V(c.vocab_padded)};
            }
            ds.push_back(std::move(d));
            push_dspark_linear(ds, "dspark.markov_head.rnn_proj.weight", mtl::Arg::B(bd_joint), mtl::Arg::B(bd_rproj), 1);
            {
                mtl::Dispatch drnn; drnn.p = &p_dsp_mrnn_tree; drnn.grid[0] = 1; drnn.tg[0] = 256;
                drnn.args = {mtl::Arg::B(bd_rproj), mod->dw("dspark.markov_head.rnn_proj.bias"),
                             mtl::Arg::B(bd_tree_states, 0), mtl::Arg::B(bd_tree_states, 256ul * 4), mtl::Arg::B(bd_latent)};
                ds.push_back(std::move(drnn));
            }
            {
                mtl::Dispatch dm; dm.p = &p_dsp_meff_top4_q8; dm.grid[0] = (uint32_t)nparts; dm.tg[0] = 256;
                dm.args = {mtl::Arg::B(b_logits, 0), mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                           mtl::Arg::B(bd_latent), mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx), mtl::Arg::V(V)};
                ds.push_back(std::move(dm));
            }
            {
                mtl::Dispatch di; di.p = &p_dsp_beam4_init_s1; di.grid[0] = 1; di.tg[0] = 256;
                di.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::B(bd_tree_states, 256ul * 4), mtl::Arg::B(bd_beam_st_in),
                           mtl::Arg::B(bd_beam_toks), mtl::Arg::B(bd_beam_logp), mtl::Arg::B(bd_beam_pidx),
                           mtl::Arg::B(bd_beam_pool), mtl::Arg::V(nparts)};
                ds.push_back(std::move(di));
            }
        }

        // 3. Depths s = 2..max_depth: W=4 parallel beams in ONE shared-DRAM M_eff pass per depth!
        for (int s = 2; s <= max_depth; ++s) {
            {
                mtl::Dispatch dj;
                dj.grid[0] = (5120 + 255) / 256; dj.grid[1] = 4; dj.tg[0] = 256;
                if (mw1_ti.fmt == FMT_BF16) {
                    dj.p = &p_dsp_mjoint_beam4_bf16;
                    dj.args = {mtl::Arg::B(bd_beam_st_in), mod->dw("dspark.markov_head.markov_w1.weight"),
                               mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                               mtl::Arg::B(bd_beam_toks), mtl::Arg::B(bd_beam_joint), mtl::Arg::V(s)};
                } else {
                    dj.p = &p_dsp_mjoint_beam4_q8;
                    dj.args = {mtl::Arg::B(bd_beam_st_in), mod->dw("dspark.markov_head.markov_w1.weight"),
                               mtl::Arg::B(bd_cur), mtl::Arg::B(dsp_frnn_slot, (size_t)slot * D * 4),
                               mtl::Arg::B(bd_beam_toks), mtl::Arg::B(bd_beam_joint), mtl::Arg::V(s),
                               mtl::Arg::V(c.vocab_padded)};
                }
                ds.push_back(std::move(dj));
            }
            push_dspark_linear(ds, "dspark.markov_head.rnn_proj.weight", mtl::Arg::B(bd_beam_joint), mtl::Arg::B(bd_beam_rproj), 4);
            {
                mtl::Dispatch dr; dr.p = &p_dsp_mrnn_beam4; dr.grid[0] = 4; dr.tg[0] = 256;
                dr.args = {mtl::Arg::B(bd_beam_rproj), mod->dw("dspark.markov_head.rnn_proj.bias"),
                           mtl::Arg::B(bd_beam_st_in), mtl::Arg::B(bd_beam_st_out), mtl::Arg::B(bd_beam_lat)};
                ds.push_back(std::move(dr));
            }
            {
                const size_t boff = (size_t)(s - 1) * (size_t)V * sizeof(float);
                mtl::Dispatch dm; dm.p = &p_dsp_meff_beam4_top2_q8; dm.grid[0] = (uint32_t)nparts; dm.tg[0] = 256;
                dm.args = {mtl::Arg::B(b_logits, boff), mtl::Arg::B(bd_meff_sc), mtl::Arg::B(bd_meff_q),
                           mtl::Arg::B(bd_beam_lat), mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                           mtl::Arg::V(V), mtl::Arg::V(nparts)};
                ds.push_back(std::move(dm));
            }
            {
                mtl::Dispatch dsel; dsel.p = &p_dsp_beam4_select; dsel.grid[0] = 1; dsel.tg[0] = 256;
                dsel.args = {mtl::Arg::B(bd_part_val), mtl::Arg::B(bd_part_idx),
                             mtl::Arg::B(bd_beam_st_out), mtl::Arg::B(bd_beam_st_in),
                             mtl::Arg::B(bd_beam_toks), mtl::Arg::B(bd_beam_logp), mtl::Arg::B(bd_beam_pidx),
                             mtl::Arg::B(bd_beam_pool), mtl::Arg::V(s), mtl::Arg::V(nparts)};
                ds.push_back(std::move(dsel));
            }
        }

        // Execute ENTIRE W=4 Beam Search in ONE single Metal command buffer!
        cx->run_graph(ds);

        // 4. CPU Sparse-MoE Bayes-Optimal Connected Tree Extraction (< 3 us!)
        const int n_pool = 4 + (max_depth - 1) * 8;
        const auto* pool = bd_beam_pool.as<DspPoolNodeHost>();
        std::vector<int> order(n_pool);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return pool[a].logp > pool[b].logp;
        });

        // Build maximal valid connected tree up to max_budget, tracking exact per-node expected probability!
        std::vector<int> pool_to_node(n_pool, -1);
        std::vector<int32_t> cand_tok = {anchor_tok};
        std::vector<int32_t> cand_par = {-1};
        std::vector<int32_t> cand_dep = {0};
        std::vector<double>  cand_p   = {1.0};
        for (int pidx : order) {
            if ((int)cand_tok.size() >= max_budget) break;
            const auto& pn = pool[pidx];
            if (pn.logp < -12.0f) break;
            const int par_node = (pn.parent_pool < 0) ? 0 : pool_to_node[pn.parent_pool];
            if (par_node < 0) continue;  // parent not yet added (should never happen by strict logp monotonicity)
            bool dup = false;
            for (int k = par_node + 1; k < (int)cand_tok.size(); ++k) {
                if (cand_par[k] == par_node && cand_tok[k] == pn.tok) {
                    pool_to_node[pidx] = k;
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            const int nid = (int)cand_tok.size();
            pool_to_node[pidx] = nid;
            cand_tok.push_back(pn.tok);
            cand_par.push_back(par_node);
            cand_dep.push_back(pn.depth);
            cand_p.push_back(std::exp((double)pn.logp));
        }

        int best_M = (int)cand_tok.size();
        const bool is_bf16_trunk = (mod->man.get("model.llm.layers.0.attn.wq_du.weight").fmt == FMT_BF16);
        const bool is_sq10 = (mod->man.build.find("10") != std::string::npos);
        // Hardware-calibrated verify latency T_ver(M) in ms (Apple M5 Max measured on real prompts!)
        auto t_ver_ms = [&](int M) -> double {
            if (is_apple10) {
                const double t1 = is_bf16_trunk ? 17.0 : (is_sq10 ? 13.0 : 14.4);
                const double uexp = 128.0 * (1.0 - std::pow(15.0 / 16.0, (double)M));
                const double base = is_bf16_trunk ? 12.6 : (is_sq10 ? 8.8 : 10.6);
                const double moe_per_e = is_bf16_trunk ? 0.38 : (is_sq10 ? 0.44 : 0.43);
                const double trunk_extra = (M >= 5) ? (is_bf16_trunk ? 2.0 : 4.2)
                                         : (M >= 4 && is_bf16_trunk) ? 2.0
                                         : ((M - 1) * (is_bf16_trunk ? 0.75 : 1.35));
                return std::max(t1, base + trunk_extra + moe_per_e * uexp + (M > 8 ? 2.2 : 0.0));
            } else {
                // Apple M4 (120 GB/s): steeper MoE bandwidth scaling
                const double uexp = 128.0 * (1.0 - std::pow(15.0 / 16.0, (double)M));
                return 34.0 + 1.85 * uexp;
            }
        };
        const double t_draft = is_apple10 ? (is_bf16_trunk ? 4.5 : 4.0) : 14.0;
        const double plain_tps = 1.0 / t_ver_ms(1);

        if (dynamic_bayes && (int)cand_tok.size() >= 2) {
            double best_tps = -1.0;
            double cum_e = 1.0;
            for (int m = 2; m <= (int)cand_tok.size(); ++m) {
                cum_e += cand_p[m - 1];
                // Check if prefix [0..m-1] is linear (saves 0.4 ms commit overhead!)
                bool lin_m = true;
                for (int i = 1; i < m; ++i) if (cand_par[i] != i - 1) { lin_m = false; break; }
                const double c_ovh = lin_m ? 0.08 : 0.45;
                const double tps = cum_e / (t_draft + t_ver_ms(m) + c_ovh);
                if (tps > best_tps) {
                    best_tps = tps;
                    best_M = m;
                }
            }
            out.expected_gain = (float)(best_tps / plain_tps);
        }

        out.M = best_M;
        out.tok_ids.assign(cand_tok.begin(), cand_tok.begin() + best_M);
        out.parents.assign(cand_par.begin(), cand_par.begin() + best_M);
        out.depths.assign(cand_dep.begin(), cand_dep.begin() + best_M);
        out.anc_masks.resize(best_M);
        out.anc_masks[0] = 1u;
        out.is_linear = true;
        for (int i = 1; i < best_M; ++i) {
            out.anc_masks[i] = out.anc_masks[out.parents[i]] | (1u << i);
            if (out.parents[i] != i - 1) out.is_linear = false;
        }
        return out.M;
    }

private:
    mtl::Pipeline p_copy_f32;

    bool compile_shaders(const std::string& kernel_dir, std::string& err);
};

}  // namespace ink_metal
