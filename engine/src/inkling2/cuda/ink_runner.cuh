// engine/src/inkling2/cuda/ink_runner.cuh - Complete 48-Visit Inkling-2-Mini CUDA Runner (B=1..64, Tree-16, Full CUDA Graph, sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"
#include "ink_hybrid.cuh"
#include "ink_kernels.cuh"
#include "ink_manifest.hpp"
#include "ink_moe_cuda.cuh"

#include "strata/artifact/strata_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// ---- Tree-32 Speculative Verification Descriptor (M <= 32 nodes rooted at anchor 0, topological order parent[i] < i) ----
struct InkTreeSpec {
    int M = 0;
    int32_t tok[kInkMaxTree] = {};
    int8_t  parent[kInkMaxTree] = {};
    uint8_t depth[kInkMaxTree] = {};
    uint8_t anc[kInkMaxTree][kInkMaxTreeDepth] = {};
    int8_t  sconv_lag[kInkMaxTree * 3] = {};  // for each node m: lag-1, lag-2, lag-3 indices (>=0 node, -1/-2/-3 history[2/1/0])

    void build() {
        for (int i = 0; i < M; ++i) {
            int cur = i;
            for (int d = (int)depth[i]; d >= 0; --d) {
                anc[i][d] = (uint8_t)cur;
                cur = (cur > 0) ? (int)parent[cur] : -1;
            }
            // ShortConv 3 lags: t-1, t-2, t-3
            int p = (i > 0) ? (int)parent[i] : -1;
            for (int l = 0; l < 3; ++l) {
                if (p >= 0) {
                    sconv_lag[i * 3 + l] = (int8_t)p;
                    p = (p > 0) ? (int)parent[p] : -1;
                } else {
                    // Remaining lags come from committed history: first miss = -1 (hist[2]), next = -2 (hist[1]), next = -3 (hist[0])
                    int miss = l - (int)depth[i];  // 0 -> -1, 1 -> -2, 2 -> -3
                    sconv_lag[i * 3 + l] = (int8_t)(-(miss + 1));
                }
            }
        }
    }
};

// (Exact 2-stage screened argmax kernels live in ink_kernels.cuh)

// ============================================================================
// InkCudaModel: GPU-resident .strata v1.1 Trunk + Experts + Pre-processed Tensors
// ============================================================================
struct InkCudaModel {
    InkManifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    int primary_gpu = 0;
    int sm_cc = 0;  // major * 10 + minor (e.g. 60, 61, 70, 75, 86, 89, 120, 121)
    bool trunk_attn_fp16 = false;  // true on sm_70/75: .attn.* BF16 weights converted in-place to FP16 at load!

    uint8_t* d_trunk = nullptr;
    uint8_t* d_sec4  = nullptr;
    uint8_t* h_sec4_pinned = nullptr;  // optional pinned overflow for 8/12/16 GB GPUs
    size_t trunk_bytes = 0, sec4_bytes = 0;
    size_t n_vram_experts = 0, n_host_experts = 0;

    // Preprocessed per-layer F32 buffers (tiny: < 40 MB total)
    float* d_msg_queries[12] = {};      // [4, 2304] F32 for physical layers 6..17
    float* d_sconv_k[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_v[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_attn[24] = {};       // [2304, 4] F32
    float* d_sconv_mlp[24] = {};        // [2304, 4] F32

    // Per-physical-layer expert directory on GPU + Strata Hyper-Inference Hybrid Engine (for 8/12/16 GB GPUs!)
    InkExpertDev* d_experts[24] = {};
    InkLayerMoeInfo moe_info[24]{};
    bool layer_has_host[24] = {};
    InkHybridEngine hybrid;

    // Exact 2-stage unembed screening index (for BF16 unembed in SQ-20)
    UnembedScreenIndex unembed_screen;

    const uint8_t* trunk_ptr(const std::string& name) const {
        return d_trunk + man.get(name).off;
    }
    const TensorInfo& tinfo(const std::string& name) const {
        return man.get(name);
    }

    bool load(const std::string& strata_path, int gpu_id, bool build_screen, std::string& err, size_t extra_reserve_bytes = 0) {
        path = strata_path;
        primary_gpu = gpu_id;
        INK_CK(cudaSetDevice(primary_gpu));
        {
            cudaDeviceProp prop{};
            INK_CK(cudaGetDeviceProperties(&prop, primary_gpu));
            sm_cc = prop.major * 10 + prop.minor;
        }
        if (!rd.open(strata_path, err)) return false;
        parse_ink_manifest(rd.manifest_json(), man, true);

        const auto& hdr = rd.header();
        trunk_bytes = (size_t)hdr.sections[1].byte_length;
        sec4_bytes  = (size_t)hdr.sections[4].byte_length;

        double t0 = now_s();
        // 1. Load Section 1 (Contiguous GPU Trunk + DSpark) in one DMA transfer
        INK_CK(cudaMalloc(&d_trunk, trunk_bytes));
        {
            const size_t CH = 64ull << 20;
            uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, trunk_bytes));
            size_t done = 0;
            while (done < trunk_bytes) {
                size_t want = std::min(CH, trunk_bytes - done);
                if (!rd.read_at(hdr.sections[1].file_offset + done, hbuf, want, err)) {
                    cudaFreeHost(hbuf);
                    return false;
                }
                INK_CK(cudaMemcpy(d_trunk + done, hbuf, want, cudaMemcpyHostToDevice));
                done += want;
            }
            INK_CK(cudaFreeHost(hbuf));
        }
        double t_trunk = now_s() - t0;

        // 1B. On Volta sm_70 (V100) & Turing sm_75 (2080 Ti), convert .attn.* BF16 weights IN-PLACE to IEEE FP16 once!
        //     Eliminates 336+ per-visit conversion launches & unlocks 125 TFLOPS direct FP16 Tensor Core GEMMs!
        trunk_attn_fp16 = (sm_cc >= 70 && sm_cc < 80);
        if (trunk_attn_fp16) {
            for (const auto& kv : man.trunk) {
                if (kv.second.fmt == FMT_BF16 && kv.first.find(".attn.") != std::string::npos) {
                    const size_t nw = (size_t)kv.second.rows() * kv.second.cols();
                    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(d_trunk + kv.second.off);
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                }
            }
            INK_CK(cudaDeviceSynchronize());
        }

        // 2. Check free VRAM -> load Section 4 experts (100% VRAM when fits, else 3x loop-weighted vram_load_order!)
        size_t vfree = 0, vtot = 0;
        INK_CK(ink_cuda_mem_info(&vfree, &vtot));
        const auto& dir = rd.expert_directory();
        const int n_total = (int)dir.size();  // 24 * 128 = 3072
        std::vector<InkExpertDev> h_exp(n_total);
        for (int il = 0; il < man.c.n_phys; ++il) {
            moe_info[il] = {};
            for (int r = 0; r < man.c.n_experts; ++r) {
                const auto& e = dir[(size_t)il * man.c.n_experts + r];
                InkExpertDev dev{};
                dev.body_off = e.body_offset;
                dev.dn_off   = e.dn_offset;
                dev.gu_code  = e.gu_code_id;
                dev.dn_code  = e.dn_code_id;
                dev.tier     = 0;
                h_exp[(size_t)il * man.c.n_experts + e.expert_id] = dev;
                if (e.gu_code_id == FMT_SF4) moe_info[il].has_sf4 = true;
                else if (e.gu_code_id == FMT_MX8) moe_info[il].has_mx8 = true;
                else moe_info[il].has_sk = true;
            }
        }

        double t1 = now_s();
        // Tight on-demand VRAM reserve: lets SQ-10 fit 100% in 12 GB RTX 5070 (even under Win11 WDDM!) and SQ-15 in 16 GB RTX 5070 Ti,
        // while keeping full headroom for FP8 KV + workspaces + DSpark + CUDA Graph (spilling cleanly to Hybrid on 11 GB GPUs)!
        const size_t base_reserve = (vtot >= (20ull << 30)) ? (1600ull << 20)
                                  : (vtot <= (13ull << 30)) ? (780ull << 20)
                                                            : (1280ull << 20);
        const size_t reserve = std::max(base_reserve, extra_reserve_bytes);
        for (int il = 0; il < 24; ++il) layer_has_host[il] = false;
        if (vfree > sec4_bytes + reserve) {
            // 100% VRAM resident! Single sequential streaming DMA!
            INK_CK(cudaMalloc(&d_sec4, sec4_bytes));
            const size_t CH = 64ull << 20;
            uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, sec4_bytes));
            size_t done = 0;
            while (done < sec4_bytes) {
                size_t want = std::min(CH, sec4_bytes - done);
                if (!rd.read_at(hdr.sections[4].file_offset + done, hbuf, want, err)) {
                    cudaFreeHost(hbuf);
                    return false;
                }
                INK_CK(cudaMemcpy(d_sec4 + done, hbuf, want, cudaMemcpyHostToDevice));
                done += want;
            }
            INK_CK(cudaFreeHost(hbuf));
            n_vram_experts = n_total;
            n_host_experts = 0;
        } else {
            // 3x Loop-Weighted Tiered Residency + Strata Predictive CPU/RAM Hybrid Engine (Doorbell + Ping-Pong DMA)!
            const auto& vorder = rd.vram_load_order();
            std::vector<uint16_t> order;
            if ((int)vorder.size() == n_total) {
                order = vorder;
            } else {
                order.resize(n_total);
                for (int i = 0; i < n_total; ++i) order[i] = (uint16_t)i;
            }
            // Reserve full headroom (1.66 GiB for runner+DSpark+cuBLAS+Graph + 256 MiB for ping-pong DMA + promo ring)!
            const size_t hyb_post_res = std::max(reserve, (size_t)(1664ull << 20));
            const size_t hyb_reserve = hyb_post_res + (256ull << 20);
            const size_t vram_budget = (vfree > hyb_reserve) ? (vfree - hyb_reserve) : 0;
            size_t vram_used = 0, host_used = 0, max_exp_bytes = 0;
            std::vector<uint8_t> in_vram(n_total, 0);
            for (uint16_t di : order) {
                const auto& e = dir[di];
                max_exp_bytes = std::max<size_t>(max_exp_bytes, e.body_bytes);
                if (vram_used + e.body_bytes <= vram_budget) {
                    in_vram[di] = 1;
                    vram_used += e.body_bytes;
                    ++n_vram_experts;
                } else {
                    host_used += (size_t)((e.body_bytes + 63ull) & ~63ull);
                    ++n_host_experts;
                }
            }
            INK_CK(cudaMalloc(&d_sec4, std::max<size_t>(256, vram_used)));
            if (host_used > 0) {
                INK_CK(cudaHostAlloc(&hybrid.h_sec4_pinned, host_used, cudaHostAllocDefault));
                hybrid.h_sec4_bytes = host_used;
                hybrid.host_meta.assign((size_t)man.c.n_phys * man.c.n_experts, {});
                hybrid.orig_host_off.assign((size_t)man.c.n_phys * man.c.n_experts, 0);
            }

            size_t v_cur = 0, h_cur = 0;
            std::vector<uint8_t> tmp_body;
            for (int di = 0; di < n_total; ++di) {
                const auto& e = dir[di];
                const int il = di / man.c.n_experts;
                const int key = il * man.c.n_experts + e.expert_id;
                tmp_body.resize(e.body_bytes);
                if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, tmp_body.data(), e.body_bytes, err)) return false;
                auto& dev = h_exp[key];
                if (in_vram[di]) {
                    INK_CK(cudaMemcpy(d_sec4 + v_cur, tmp_body.data(), e.body_bytes, cudaMemcpyHostToDevice));
                    dev.body_off = v_cur;
                    dev.tier = 0;
                    v_cur += e.body_bytes;
                } else {
                    uint8_t* hdst = hybrid.h_sec4_pinned + h_cur;
                    std::memcpy(hdst, tmp_body.data(), e.body_bytes);
                    hybrid.host_meta[key].body       = hdst;
                    hybrid.host_meta[key].body_bytes = e.body_bytes;
                    hybrid.host_meta[key].dn_off     = e.dn_offset;
                    hybrid.host_meta[key].gu_code    = e.gu_code_id;
                    hybrid.host_meta[key].dn_code    = e.dn_code_id;
                    hybrid.orig_host_off[key]        = h_cur;
                    dev.body_off = h_cur;
                    dev.tier = 2;  // CPU Doorbell (M=1) + Ping-Pong DMA Streaming (M>=2)!
                    layer_has_host[il] = true;
                    h_cur += (size_t)((e.body_bytes + 63ull) & ~63ull);
                }
            }
            if (n_host_experts > 0) {
                hybrid.start_workers();
                size_t vf_now = 0, vt_now = 0;
                INK_CK(ink_cuda_mem_info(&vf_now, &vt_now));
                const size_t avail = (vf_now > hyb_post_res) ? (vf_now - hyb_post_res) : 0;
                hybrid.alloc_staging(max_exp_bytes, avail);
            }
        }
        double t_exp = now_s() - t1;

        for (int il = 0; il < man.c.n_phys; ++il) {
            d_experts[il] = dalloc<InkExpertDev>(man.c.n_experts);
            INK_CK(cudaMemcpy(d_experts[il], h_exp.data() + (size_t)il * man.c.n_experts,
                              (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
        }

        // 3. Preprocess msg_queries[6..17] and static_sconv[0..23] into exact contiguous F32 GPU arrays
        {
            const auto& ti_q  = man.get("model.llm.middle_cycle_cross_pass_queries");
            const auto& ti_kv = man.get("model.llm.middle_cycle_cross_pass_kv_queries");
            std::vector<float> h_q(24 * kInkHidden), h_kv(24 * kInkHidden);
            INK_CK(cudaMemcpy(h_q.data(), d_trunk + ti_q.off, h_q.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(h_kv.data(), d_trunk + ti_kv.off, h_kv.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 12; ++i) {
                std::vector<float> mq(4 * kInkHidden);
                std::memcpy(mq.data() + 0 * kInkHidden, h_q.data()  + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 1 * kInkHidden, h_kv.data() + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 2 * kInkHidden, h_kv.data() + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 3 * kInkHidden, h_q.data()  + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                d_msg_queries[i] = dalloc<float>(4 * kInkHidden);
                INK_CK(cudaMemcpy(d_msg_queries[i], mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        for (int il = 0; il < man.c.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            d_sconv_attn[il] = (float*)(d_trunk + man.get(p + ".attn_sconv.weight").off);
            d_sconv_mlp[il]  = (float*)(d_trunk + man.get(p + ".mlp_sconv.weight").off);
            if (man.c.is_swa[il]) {
                d_sconv_k[il] = (float*)(d_trunk + man.get(p + ".attn.k_sconv.weight").off);
                d_sconv_v[il] = (float*)(d_trunk + man.get(p + ".attn.v_sconv.weight").off);
            } else {
                // Global static_w is [4, 1024] with lag 0 first -> flip(0).T -> [1024, 4] with lag 3..0!
                auto flip_t = [&](const std::string& tname) -> float* {
                    std::vector<float> src(4 * kInkKVDim), dst(kInkKVDim * 4);
                    INK_CK(cudaMemcpy(src.data(), d_trunk + man.get(tname).off, src.size() * 4, cudaMemcpyDeviceToHost));
                    for (int c = 0; c < kInkKVDim; ++c) {
                        for (int l = 0; l < 4; ++l) dst[c * 4 + l] = src[(3 - l) * kInkKVDim + c];
                    }
                    float* dptr = dalloc<float>(dst.size());
                    INK_CK(cudaMemcpy(dptr, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
                    return dptr;
                };
                d_sconv_k[il] = flip_t(p + ".attn.k_sconv.static_w");
                d_sconv_v[il] = flip_t(p + ".attn.v_sconv.static_w");
            }
        }

        // 4. Build exact 2-stage Cauchy-Schwarz screening index for BF16 or S6_K unembed (SQ-20, SQ-15, SQ-10)!
        const auto& ui = man.get("model.llm.unembed.weight");
        if (build_screen && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K)) {
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            if (vfree > (2048ull << 20)) {
                unembed_screen.V = man.c.vocab_unpadded;
                unembed_screen.q8  = dalloc<int8_t>((size_t)unembed_screen.V * kInkHidden);
                unembed_screen.sc  = dalloc<float>(unembed_screen.V);
                unembed_screen.eps = dalloc<float>(unembed_screen.V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_build_unembed_screen_bf16<<<(unembed_screen.V + 7) / 8, 256>>>(
                        reinterpret_cast<const uint16_t*>(d_trunk + ui.off),
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                } else {
                    k_ink_build_unembed_screen_s6k<<<(unembed_screen.V + 7) / 8, 256>>>(
                        d_trunk + ui.off,
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                }
                INK_CK(cudaDeviceSynchronize());
                unembed_screen.ready = true;
            }
        }

        std::fprintf(stderr, "[InkCUDA] Loaded %s (%s): trunk=%.2f GB (%.2fs), experts=%zu VRAM / %zu host (%.2fs, hyb=%d, promo=%d, pf_buf=%.0fMB), sm120_tc=%d, screen=%d\n",
                     strata_path.c_str(), man.build.c_str(), (double)trunk_bytes / 1e9, t_trunk,
                     n_vram_experts, n_host_experts, t_exp, (int)hybrid.active, hybrid.n_promo_slots,
                     (double)hybrid.pf_stage_bytes / (1024.0 * 1024.0), (int)sm120::available(), (int)unembed_screen.ready);
        return true;
    }

    void free_all() {
        hybrid.free_all();
        unembed_screen.free_all();
        for (int i = 0; i < 12; ++i) if (d_msg_queries[i]) { cudaFree(d_msg_queries[i]); d_msg_queries[i] = nullptr; }
        for (int il = 0; il < 24; ++il) {
            if (d_experts[il]) { cudaFree(d_experts[il]); d_experts[il] = nullptr; }
            if (!man.c.is_swa.empty() && !man.c.is_swa[il]) {
                if (d_sconv_k[il]) { cudaFree(d_sconv_k[il]); d_sconv_k[il] = nullptr; }
                if (d_sconv_v[il]) { cudaFree(d_sconv_v[il]); d_sconv_v[il] = nullptr; }
            }
        }
        if (d_sec4) { cudaFree(d_sec4); d_sec4 = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; }
        if (d_trunk) { cudaFree(d_trunk); d_trunk = nullptr; }
    }
};

// ============================================================================
// InkCudaRunner: Full 48-Visit Forward, Continuous Batching (B=1..64), Tree-16, CUDA Graphs!
// ============================================================================
struct InkCudaRunner {
    InkCudaModel& m;
    cudaStream_t st = nullptr;
    cublasHandle_t cublas = nullptr;

    int max_slots = 1;
    int max_ctx = 4096;
    int max_batch = 512;
    int swa_ring = 1024 + 512;
    int tree_cap = 16;
    int max_passes_override = 0;  // 0 = full 3 passes (48 visits); 1 or 2 = early-pass exit probe!
    InkKvDtype kv_dtype = InkKvDtype::FP16;

    // KV Cache (48 visits: 40 SWA, 8 Global)
    InkKvVisitDesc kv_desc[kInkVisits]{};
    // ShortConv state pool: [48 visits] x 4 states (k [1024], v [1024], attn [2304], mlp [2304]) -> 1 contiguous buffer!
    // Layout: visit -> [max_slots, 3, 6656] F32 (or 256 MB for 64 slots)
    float* d_sconv_pool[kInkVisits] = {};

    // Activation workspaces
    float *d_cur = nullptr, *d_res = nullptr, *d_prelude = nullptr;
    float *d_blocks[8] = {};          // attention-residual block bank (max 6 needed)
    float *d_prelude_blocks[2] = {};  // saved prelude block snapshots
    float *d_bank[4] = {};            // saved previous-pass core blocks
    float *d_bank_out = nullptr;
    float *d_mixed = nullptr;         // [4, max_batch, 2304]
    float *d_msgs = nullptr;          // [4, max_batch, 2304]
    float *d_inp0 = nullptr, *d_inp1 = nullptr, *d_inp2 = nullptr;
    float *d_q = nullptr, *d_k_raw = nullptr, *d_v_raw = nullptr, *d_k = nullptr, *d_v = nullptr, *d_r = nullptr;
    float *d_k_dyn = nullptr, *d_v_dyn = nullptr;
    float *d_sinks = nullptr, *d_gate_attn = nullptr, *d_rel_table = nullptr;
    float *d_attn_out = nullptr, *d_wo_raw = nullptr, *d_wo_conv = nullptr;
    float *d_moe_in = nullptr, *d_moe_raw = nullptr, *d_moe_conv = nullptr;
    float *d_h_norm = nullptr;        // [max_batch, 2304] final output_norm hidden
    float *d_taps[4] = {};            // [max_batch, 2304] taps at visits [5, 23, 41, 48]
    float *d_taps_compact[4] = {};    // [16, 2304] compacted accepted-path taps for DSpark
    float *d_ctrl_cat = nullptr, *d_ctrl_gate = nullptr, *d_ctrl_gain = nullptr, *d_ctrl_forget = nullptr;
    float *d_attn_pm = nullptr, *d_attn_pl = nullptr, *d_attn_po = nullptr;  // Split-K FlashDecoding buffers
    __nv_bfloat16 *d_pf_q = nullptr, *d_pf_k = nullptr, *d_pf_v = nullptr, *d_pf_p = nullptr;  // TC prefill attention
    float *d_pf_s = nullptr, *d_pf_o = nullptr;
    int pf_kv_cap = 0;
    float *d_stage_k = nullptr, *d_stage_v = nullptr, *d_stage_sconv = nullptr;  // Tree-16 verify staging
    InkKvVisitDesc* d_kv_desc_arr = nullptr;
    float** d_sconv_pool_arr = nullptr;
    int32_t* d_acc_path = nullptr;
    __nv_bfloat16 *d_x_bf16 = nullptr;
    __nv_bfloat16 *d_w_scratch = nullptr;  // [max_N, max_K] BF16 scratch for Q8_ROW / S_K cuBLAS prefill

    // Logits & screened argmax buffers
    float   *d_logits = nullptr;      // [logits_cap, 200058] (allocated lazily if full logits needed)
    int      logits_cap = 0;
    float   *d_dot_hat = nullptr;     // [16, 200058] for screened argmax
    float   *d_h_norm2 = nullptr;     // [16]
    int     *d_max_lb = nullptr;      // [16]
    int32_t *d_cand_ids = nullptr;    // [16, 4096]
    int32_t *d_cand_cnt = nullptr;    // [16]
    int32_t *d_argmax = nullptr;      // [64]

    // Step metadata in device memory (CUDA-Graph capturable!) + 64-slot pinned host ring (zero async H2D overwrite!)
    int32_t *d_tok_ids = nullptr;     // [max_batch]
    int32_t *d_slots = nullptr;       // [max_batch]
    int32_t *d_pos = nullptr;         // [max_batch]
    uint8_t *d_is_img = nullptr;      // [max_batch] optional vision mask
    float   *d_img_emb = nullptr;     // [max_batch, 2304] optional vision embeddings
    bool     has_img_step = false;
    int8_t  *d_sconv_lag = nullptr;   // [16 * 3]
    uint8_t *d_tree_anc = nullptr;    // [16 * 16]
    uint8_t *d_tree_dep = nullptr;    // [16]
    int32_t *h_meta_pin = nullptr;    // 64-slot ring buffer
    size_t   meta_ring_stride = 0;
    size_t   meta_ring_idx = 0;

    InkMoeWorkspace moe_ws;

    // 48-Visit CUDA Graph cache (keyed by M, is_tree, capture_taps)
    struct GraphKey {
        int M;
        int is_tree;
        int want_taps;
        int kv_fp8;
        bool operator==(const GraphKey& o) const {
            return M == o.M && is_tree == o.is_tree && want_taps == o.want_taps && kv_fp8 == o.kv_fp8;
        }
    };
    struct GraphEntry {
        GraphKey key;
        cudaGraphExec_t exec = nullptr;
    };
    std::vector<GraphEntry> graph_cache;
    bool use_cuda_graph = true;

    // Optional per-stage profiler (active only when prof_enabled=true and graph disabled)
    bool prof_enabled = false;
    double prof_ms[8] = {};  // 0=res/ctrl/norm, 1=dense_qkv, 2=sconv_clip_kv, 3=attn, 4=dense_wo, 5=moe, 6=other, 7=unembed
    cudaEvent_t ev_p0 = nullptr, ev_p1 = nullptr;
    void prof_mark_start() { if (prof_enabled) INK_CK(cudaEventRecord(ev_p0, st)); }
    void prof_mark_end(int idx) {
        if (!prof_enabled) return;
        INK_CK(cudaEventRecord(ev_p1, st));
        INK_CK(cudaEventSynchronize(ev_p1));
        float ms = 0.0f;
        INK_CK(cudaEventElapsedTime(&ms, ev_p0, ev_p1));
        prof_ms[idx] += ms;
    }
    void prof_reset() { for (double& x : prof_ms) x = 0.0; }
    void prof_print(const char* tag, int iters) const {
        double tot = 0; for (double x : prof_ms) tot += x;
        std::printf("[PROF %s / %d iters] total=%.2f ms/step | res_ctrl=%.2f dense_qkv=%.2f sconv_kv=%.2f attn=%.2f dense_wo=%.2f moe=%.2f unembed=%.2f\n",
                    tag, iters, tot / iters,
                    prof_ms[0] / iters, prof_ms[1] / iters, prof_ms[2] / iters, prof_ms[3] / iters,
                    prof_ms[4] / iters, prof_ms[5] / iters, prof_ms[7] / iters);
    }

    InkCudaRunner(InkCudaModel& model) : m(model) {}

    void init(int slots, int ctx, int chunk, InkKvDtype kvd = InkKvDtype::FP16) {
        max_slots = std::max(1, slots);
        max_ctx = std::max(64, ctx);
        max_batch = std::max(chunk, std::max(16, max_slots));
        // Compact SWA ring for high concurrency (c >= 16) saves ~2 GB VRAM at c = 64!
        swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + std::max(max_batch, 64));
        kv_dtype = kvd;

        INK_CK(cudaSetDevice(m.primary_gpu));
        INK_CK(cudaStreamCreate(&st));
        INK_CK(cudaEventCreate(&ev_p0));
        INK_CK(cudaEventCreate(&ev_p1));
        INK_CUBLAS_CK(cublasCreate(&cublas));
        INK_CUBLAS_CK(cublasSetStream(cublas, st));
        INK_CUBLAS_CK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));

        // Auto-fit max_ctx to free VRAM so c = 32 / 64 never OOMs on 32 GiB RTX 5090!
        {
            size_t vfree = 0, vtot = 0;
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            tree_cap = (max_slots <= 4 && vfree > (700ull << 20)) ? kInkMaxTree : 16;
            max_batch = std::max(max_batch, tree_cap);
            // On tight 12 GB GPUs (especially Win11 WDDM), cap max_batch to 512 so workspaces never trigger host paging!
            if (vfree < (2200ull << 20) && max_batch > 512) {
                max_batch = 512;
                swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + max_batch);
            }
            const size_t bpe = (kv_dtype == InkKvDtype::FP8) ? 2 : 4;  // K+V bytes per element
            const size_t sc_per_pos = (kv_dtype == InkKvDtype::FP8) ? (size_t)(2 * kInkKVHeads * 4) : 0;
            const size_t swa_kv_bytes = (size_t)40 * max_slots * swa_ring * ((size_t)kInkKVDim * bpe + sc_per_pos);
            const size_t sconv_bytes  = (size_t)kInkVisits * max_slots * 3 * kInkSconvCh * 4;
            const size_t ws_reserve   = (max_slots >= 8) ? (1400ull << 20) : (480ull << 20);  // activation workspaces + DSpark + cuBLAS headroom
            if (vfree > swa_kv_bytes + sconv_bytes + ws_reserve) {
                const size_t avail_glob = vfree - swa_kv_bytes - sconv_bytes - ws_reserve;
                const size_t per_pos_glob = (size_t)8 * max_slots * ((size_t)kInkKVDim * bpe + sc_per_pos);
                const int fit_ctx = (int)std::min<size_t>((size_t)max_ctx, (avail_glob / std::max<size_t>(1, per_pos_glob)) & ~(size_t)63);
                if (fit_ctx < max_ctx) {
                    const int clamped = std::max(256, fit_ctx);
                    std::fprintf(stderr, "[InkCUDA] Auto-sizing max_ctx %d -> %d for parallel=%d (%.2f GiB free VRAM)\n",
                                 max_ctx, clamped, max_slots, (double)vfree / (1024.0 * 1024.0 * 1024.0));
                    max_ctx = clamped;
                }
            } else if (max_ctx > 256) {
                std::fprintf(stderr, "[InkCUDA] Tight VRAM (%.2f GiB free) for parallel=%d: clamping max_ctx %d -> 256\n",
                             (double)vfree / (1024.0 * 1024.0 * 1024.0), max_slots, max_ctx);
                max_ctx = 256;
            }
        }

        // Allocate KV caches for all 48 visits
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const bool swa = m.man.c.is_swa[ip] != 0;
            const int cap = swa ? swa_ring : max_ctx;
            kv_desc[v].cap_pos = cap;
            kv_desc[v].is_swa = swa;
            kv_desc[v].is_fp8 = (kv_dtype == InkKvDtype::FP8);
            const size_t elems = (size_t)max_slots * cap * kInkKVHeads * kInkHeadDim;
            if (!kv_desc[v].is_fp8) {
                kv_desc[v].k_f16 = dalloc<__half>(elems);
                kv_desc[v].v_f16 = dalloc<__half>(elems);
            } else {
                kv_desc[v].k_f8 = dalloc<uint8_t>(elems);
                kv_desc[v].v_f8 = dalloc<uint8_t>(elems);
                kv_desc[v].k_sc = dalloc<float>((size_t)max_slots * cap * kInkKVHeads);
                kv_desc[v].v_sc = dalloc<float>((size_t)max_slots * cap * kInkKVHeads);
            }
            d_sconv_pool[v] = dalloc<float>((size_t)max_slots * 3 * kInkSconvCh);
            INK_CK(cudaMemset(d_sconv_pool[v], 0, (size_t)max_slots * 3 * kInkSconvCh * sizeof(float)));
        }

        const size_t MB = (size_t)max_batch;
        d_cur       = dalloc<float>(MB * kInkHidden);
        d_res       = dalloc<float>(MB * kInkHidden);
        d_prelude   = dalloc<float>(MB * kInkHidden);
        d_bank_out  = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 8; ++i) d_blocks[i] = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 2; ++i) d_prelude_blocks[i] = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 4; ++i) d_bank[i] = dalloc<float>(MB * kInkHidden);
        d_mixed     = dalloc<float>(4 * MB * kInkHidden);
        d_msgs      = dalloc<float>(4 * MB * kInkHidden);
        d_inp0      = dalloc<float>(MB * kInkHidden);
        d_inp1      = dalloc<float>(MB * kInkHidden);
        d_inp2      = dalloc<float>(MB * kInkHidden);
        d_q         = dalloc<float>(MB * kInkQDim);
        d_k_raw     = dalloc<float>(MB * kInkKVDim);
        d_v_raw     = dalloc<float>(MB * kInkKVDim);
        d_k         = dalloc<float>(MB * kInkKVDim);
        d_v         = dalloc<float>(MB * kInkKVDim);
        d_r         = dalloc<float>(MB * kInkHeads * kInkDRel);
        d_k_dyn     = dalloc<float>(MB * kInkKVHeads * 4);
        d_v_dyn     = dalloc<float>(MB * kInkKVHeads * 4);
        d_sinks     = dalloc<float>(MB * kInkHeads);
        d_gate_attn = dalloc<float>(MB * kInkQDim);
        d_rel_table = dalloc<float>(MB * kInkHeads * (kInkRelExtent + 1));
        d_attn_out  = dalloc<float>(MB * kInkQDim);
        d_wo_raw    = dalloc<float>(MB * kInkHidden);
        d_wo_conv   = dalloc<float>(MB * kInkHidden);
        d_moe_in    = dalloc<float>(MB * kInkHidden);
        d_moe_raw   = dalloc<float>(MB * kInkHidden);
        d_moe_conv  = dalloc<float>(MB * kInkHidden);
        d_h_norm    = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 4; ++i) {
            d_taps[i]         = dalloc<float>(MB * kInkHidden);
            d_taps_compact[i] = dalloc<float>((size_t)tree_cap * kInkHidden);
        }
        d_ctrl_cat    = dalloc<float>(3ull * MB * 2 * kInkHidden);
        d_ctrl_gate   = dalloc<float>(3ull * MB * kInkGateRank);
        d_ctrl_gain   = dalloc<float>(MB * kInkHidden);
        d_ctrl_forget = dalloc<float>(MB * kInkHidden);
        d_attn_pm     = dalloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits);
        d_attn_pl     = dalloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits);
        d_attn_po     = dalloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits * kInkHeadDim);
        d_stage_k     = dalloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        d_stage_v     = dalloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        d_stage_sconv = dalloc<float>((size_t)kInkVisits * tree_cap * kInkSconvCh);
        d_kv_desc_arr = dalloc<InkKvVisitDesc>(kInkVisits);
        INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
        d_sconv_pool_arr = dalloc<float*>(kInkVisits);
        INK_CK(cudaMemcpy(d_sconv_pool_arr, d_sconv_pool, sizeof(d_sconv_pool), cudaMemcpyHostToDevice));
        d_acc_path    = dalloc<int32_t>(kInkMaxTree);

        d_x_bf16    = dalloc<__nv_bfloat16>(MB * std::max(kInkDsparkInter, std::max(kInkQDim, kInkHidden)));
        d_w_scratch = dalloc<__nv_bfloat16>((size_t)kInkDsparkInter * kInkHidden);

        // Warmup cuBLAS Tensor Cores once at init (eliminates 61 ms first-call JIT tax from prompt #1!)
        {
            INK_CK(cudaMemsetAsync(d_w_scratch, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_x_bf16, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_ctrl_cat, 0, 64 * 64 * sizeof(float), st));
            const float a = 1.0f, b = 0.0f;
            if (m.sm_cc >= 80) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16BF, 64, d_x_bf16, CUDA_R_16BF, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else if (m.sm_cc >= 70) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16F, 64, d_x_bf16, CUDA_R_16F, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                         d_ctrl_cat, CUDA_R_32F, 64, d_ctrl_cat, CUDA_R_32F, 64, &b,
                         d_ctrl_gain, CUDA_R_32F, 64,
                         m.sm_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            INK_CK(cudaStreamSynchronize(st));
        }

        d_dot_hat   = dalloc<float>(8ull * kInkVocabUnpad);
        d_h_norm2   = dalloc<float>(8);
        d_max_lb    = dalloc<int>(8);
        d_cand_ids  = dalloc<int32_t>(8ull * 4096);
        d_cand_cnt  = dalloc<int32_t>(8);
        d_argmax    = dalloc<int32_t>(std::max(64, max_batch));

        d_tok_ids   = dalloc<int32_t>(MB);
        d_slots     = dalloc<int32_t>(MB);
        d_pos       = dalloc<int32_t>(MB);
        d_is_img    = dalloc<uint8_t>(MB);
        d_img_emb   = dalloc<float>(MB * kInkHidden);
        d_sconv_lag = dalloc<int8_t>(kInkMaxTree * 3);
        d_tree_anc  = dalloc<uint8_t>(kInkMaxTree * kInkMaxTreeDepth);
        d_tree_dep  = dalloc<uint8_t>(kInkMaxTree);
        meta_ring_stride = std::max<size_t>(MB * 3 + 256, 512);
        meta_ring_idx = 0;
        h_meta_pin  = halloc_pinned<int32_t>(64ull * meta_ring_stride);

        moe_ws.alloc(max_batch);
        // Pre-allocate Tensor Core prefill attention workspace outside timed prefill loop when VRAM allows (~140 MB)!
        if (m.sm_cc >= 60 && max_slots <= 4 && max_batch >= 128) {
            size_t vf = 0, vt = 0;
            INK_CK(cudaMemGetInfo(&vf, &vt));
            if (vf > (640ull << 20)) ensure_pf_attn(std::min(max_ctx, std::max(max_batch, 2048)));
        }
    }

    void reset_slot(int slot) {
        constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
        constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
        for (int v = 0; v < kInkVisits; ++v) {
            for (int s = 0; s < 4; ++s) {
                float* p = d_sconv_pool[v] + (size_t)c_off[s] * 3 * max_slots + (size_t)slot * 3 * c_len[s];
                INK_CK(cudaMemsetAsync(p, 0, (size_t)3 * c_len[s] * sizeof(float), st));
            }
        }
    }

    void ensure_logits(int M) {
        if (M <= logits_cap) return;
        INK_CK(cudaStreamSynchronize(st));
        if (d_logits) INK_CK(cudaFree(d_logits));
        logits_cap = M;
        d_logits = dalloc<float>((size_t)logits_cap * m.man.c.vocab_unpadded);
    }

    // ---- High-Speed Rank-64 Loop Controller + RMSNorm Dispatcher (100x faster than single-CTA!) ----
    void run_loop_control_norm(
        const float* cur, const float* msg,
        const float* w_down, const float* w_gain, const float* w_forget,
        const float* norm_w, float* out, int M,
        bool cat_ready = false, bool exact_fp32 = false) {
        constexpr int D = kInkHidden;
        constexpr int D2 = 2 * D;
        if (!cat_ready) k_ink_ctrl_cat_rms<<<M, 256, 0, st>>>(cur, msg, d_ctrl_cat, M, D);
        if (M <= 16) {
            if (M == 1) {
                k_ink_ctrl_down_smallM<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<1><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 4) {
                k_ink_ctrl_down_smallM<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<4><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 8) {
                k_ink_ctrl_down_smallM<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<8><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else {
                k_ink_ctrl_down_smallM<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<16><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            }
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            if (!exact_fp32 && m.sm_cc >= 70 && m.sm_cc < 80) {
                // Volta sm_70 (V100) / Turing sm_75 (2080 Ti): 125 TFLOPS FP16 TC for loop controller GEMMs (4x faster than FP32!)
                __half* x_h = reinterpret_cast<__half*>(d_x_bf16);
                __half* w_h = reinterpret_cast<__half*>(d_w_scratch);
                const size_t nx_d = (size_t)M * D2;
                const size_t nw_d = (size_t)kInkGateRank * D2;
                k_ink_f32_to_fp16<<<(nx_d + 255) / 256, 256, 0, st>>>(d_ctrl_cat, x_h, nx_d);
                k_ink_f32_to_fp16<<<(nw_d + 255) / 256, 256, 0, st>>>(w_down, w_h, nw_d);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_h, CUDA_R_16F, D2, x_h, CUDA_R_16F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const size_t nx_g = (size_t)M * kInkGateRank;
                const size_t nw_g = (size_t)D * kInkGateRank;
                k_ink_f32_to_fp16<<<(nx_g + 255) / 256, 256, 0, st>>>(d_ctrl_gate, x_h, nx_g);
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_gain, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_forget, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                const cublasComputeType_t ctype = (!exact_fp32 && m.sm_cc >= 80) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_down, CUDA_R_32F, D2, d_ctrl_cat, CUDA_R_32F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_gain, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_forget, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            k_ink_ctrl_blend_norm<<<M, 256, 0, st>>>(cur, msg, d_ctrl_gain, d_ctrl_forget, norm_w, out, M, D);
        }
    }

    // ---- Cross-Architecture cuBLAS GEMM for BF16 weights: sm_80+ native BF16 TC, sm_70/75 FP16 TC, sm_60/61 FP32 ----
    // When w_is_scratch==true, caller placed BF16 weights at d_w_scratch (or at d_w_scratch + 2*(cap/3) when sm_cc < 70).
    void gemm_bf16_compat(const __nv_bfloat16* W_bf16, const float* X, float* Y, int M, int N, int K,
                          int ldY, bool reuse_x, bool w_is_scratch) {
        const float alpha = 1.0f, beta = 0.0f;
        if (m.sm_cc >= 80) {
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_bf16<<<((nx + 3) / 4 + 255) / 256, 256, 0, st>>>(X, d_x_bf16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha,
                W_bf16, CUDA_R_16BF, K,
                d_x_bf16, CUDA_R_16BF, K,
                &beta, Y, CUDA_R_32F, ldY,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (m.sm_cc >= 70) {
            // Volta (sm_70) / Turing (sm_75): IEEE FP16 Tensor Cores (110-125 TFLOPS!)
            __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
            }
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_fp16, nw);
                cublasStatus_t st_fp16 = cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_fp16, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st_fp16 != CUBLAS_STATUS_SUCCESS) {
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            } else {
                const int tile_n = std::min(N, (int)(cap / (size_t)K));
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_fp16, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        } else {
            // Pascal (sm_60 / sm_61): exact IEEE FP32 cuBLAS SGEMM (full 11 TFLOPS on GTX 1080 Ti!)
            // Non-overlapping staging inside d_w_scratch: float[0..cap/3) vs bf16[2*(cap/3)..cap)
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_elems = cap / 3;
            float* w_f32 = reinterpret_cast<float*>(d_w_scratch);
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_f32, nw);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_f32, CUDA_R_32F, K,
                    X, CUDA_R_32F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                const int tile_n = std::min(N, (int)(max_elems / (size_t)K));
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_f32, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_f32, CUDA_R_32F, K,
                        X, CUDA_R_32F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            }
        }
    }

    // ---- Dense linear projection helper: Y[M, N] = X[M, K] @ W[N, K]^T ----
    void linear_multi(const std::string* names, float* const* Ys, int count, const float* X, int M, bool exact_fp32_x = false) {
        if (count <= 0) return;
        const auto& ti0 = m.tinfo(names[0]);
        const int K = (int)ti0.cols();
        const bool is_fp16 = m.trunk_attn_fp16 && (names[0].find(".attn.") != std::string::npos);
        // BF16/FP16: sm_70/75 direct FP16 TC at M>=4; sm_80+ exact GEMV up to M<=16 during spec verify & cuBLAS TC at M>=8!
        const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || m.sm_cc < 80) ? 16 : 7);
        bool can_fuse = (count <= 5) && ((ti0.fmt == FMT_BF16 && M <= bf16_gemv_max) || (ti0.fmt == FMT_Q8ROW && M <= 16));
        InkMultiProj mp{};
        mp.count = count;
        mp.cumN[0] = 0;
        for (int i = 0; i < count; ++i) {
            const auto& ti = m.tinfo(names[i]);
            if (ti.fmt != ti0.fmt || (int)ti.cols() != K) { can_fuse = false; break; }
            if (m.trunk_attn_fp16 && ((names[i].find(".attn.") != std::string::npos) != is_fp16)) { can_fuse = false; break; }
            mp.W[i] = m.d_trunk + ti.off;
            mp.Y[i] = Ys[i];
            mp.N[i] = (int)ti.rows();
            mp.cumN[i + 1] = mp.cumN[i] + mp.N[i];
        }
        if (can_fuse) {
            const int grid = (mp.cumN[count] + 7) / 8;
            if (ti0.fmt == FMT_BF16) {
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
            } else if ((K % 64) == 0 && K <= kInkQDim && (m.sm_cc == 61 || m.sm_cc >= 75)) {
                if (M == 1) {
                    k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                } else {
                    const int nblk = M * (K / 64);
                    k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, moe_ws.xq_sk, M, K);
                    if (M <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                    else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                    else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                }
            } else {
                const int g4 = (mp.cumN[count] + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_q8row<1><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_q8row<4><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_q8row<8><<<g4, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_q8row<16><<<g4, 128, 0, st>>>(mp, X, M, K);
            }
            return;
        }
        for (int i = 0; i < count; ++i) linear(names[i], X, Ys[i], M, 0, i > 0, exact_fp32_x);
    }

    void linear(const std::string& tname, const float* X, float* Y, int M, int N_override = 0, bool reuse_x_bf16 = false, bool exact_fp32_x = false) {
        const auto& ti = m.tinfo(tname);
        const int N = N_override > 0 ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        const uint8_t* W = m.d_trunk + ti.off;

        if (ti.fmt == FMT_BF16) {
            const bool is_fp16 = m.trunk_attn_fp16 && (tname.find(".attn.") != std::string::npos);
            const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || m.sm_cc < 80) ? 16 : 7);
            if (M <= bf16_gemv_max) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                const int grid = (N + 7) / 8;
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
            } else if (is_fp16) {
                // Volta sm_70 / Turing sm_75: direct FP16 Tensor Core GEMM (ZERO weight conversion, 125 TFLOPS!)
                __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
                if (!reuse_x_bf16) {
                    const size_t nx = (size_t)M * K;
                    k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
                }
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    W, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                gemm_bf16_compat(reinterpret_cast<const __nv_bfloat16*>(W), X, Y, M, N, K, N, reuse_x_bf16, false);
            }
        } else if (ti.fmt == FMT_F32) {
            if (M <= 16) {
                const int grid = (N + 7) / 8;
                if (M == 1)      k_ink_gemv_f32<1><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 4) k_ink_gemv_f32<4><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 8) k_ink_gemv_f32<8><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else             k_ink_gemv_f32<16><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
            } else {
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                    W, CUDA_R_32F, K, X, CUDA_R_32F, K, &beta, Y, CUDA_R_32F, N,
                    m.sm_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        } else if (ti.fmt == FMT_Q8ROW) {
            // Hardware INT8 GEMM (45 TOPS dp4a on GTX 1080 Ti sm_61!)
            if (M > 16 && m.sm_cc == 61 && (K % 4 == 0) && (N % 4 == 0) && N_override == 0) {
                int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
                float*  x_sc = d_ctrl_gain;  // [max_batch * 2304] -> plenty of room for M floats!
                if (!reuse_x_bf16) {
                    k_ink_quant_x_row_i8<<<M, 256, 0, st>>>(X, x_i8, x_sc, M, K);
                }
                const float*  w_sc = reinterpret_cast<const float*>(W);
                const int8_t* w_i8 = reinterpret_cast<const int8_t*>(W + (size_t)ti.rows() * 4);
                const int32_t alpha_i = 1, beta_i = 0;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha_i,
                    w_i8, CUDA_R_8I, K,
                    x_i8, CUDA_R_8I, K,
                    &beta_i, Y, CUDA_R_32I, N,
                    CUBLAS_COMPUTE_32I,
                    (m.sm_cc >= 75) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
                const size_t tot = (size_t)M * N;
                k_ink_i32_to_f32_scale<<<(tot + 255) / 256, 256, 0, st>>>(
                    reinterpret_cast<int32_t*>(Y), Y, w_sc, x_sc, M, N);
                return;
            }
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_tile = (m.sm_cc < 70) ? (cap / 3) : cap;
            if (M <= 16 || (size_t)N * K > max_tile) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                if ((K % 64) == 0 && K <= kInkQDim && M <= 16 && (m.sm_cc == 61 || m.sm_cc >= 75)) {
                    const int grid = (N + 7) / 8;
                    if (M == 1) {
                        k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                        return;
                    }
                    const int nblk = M * (K / 64);
                    if ((size_t)nblk <= (size_t)moe_ws.cap_M * (kInkHidden / 64)) {
                        k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, moe_ws.xq_sk, M, K);
                        if (M <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                        else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                        else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, moe_ws.xq_sk, M, K);
                        return;
                    }
                }
                const int grid = (N + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_q8row<1><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_q8row<4><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_q8row<8><<<grid, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_q8row<16><<<grid, 128, 0, st>>>(mp, X, M, K);
            } else {
                __nv_bfloat16* w_dst = (m.sm_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
                dim3 dg((K + 255) / 256, N);
                k_ink_q8row_to_bf16<<<dg, 256, 0, st>>>(W, w_dst, N, K);
                gemm_bf16_compat(w_dst, X, Y, M, N, K, N, reuse_x_bf16, true);
            }
        } else {
            // S5_K / S6_K / S8_K dense trunk tensor (dequant tile -> cuBLAS BF16/FP16/FP32 GEMM!)
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_tile = (m.sm_cc < 70) ? (cap / 3) : cap;
            const int tile_n = std::min(N, (int)(max_tile / (size_t)K));
            const size_t row_b = (size_t)(K / 256) * ink_sb_bytes(ti.fmt);
            __nv_bfloat16* w_dst = (m.sm_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
            for (int n0 = 0; n0 < N; n0 += tile_n) {
                const int nt = std::min(tile_n, N - n0);
                dim3 dg((K + 255) / 256, nt);
                const uint8_t* Wsub = W + (size_t)n0 * row_b;
                if (ti.fmt == FMT_S5_K)      k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else if (ti.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else                         k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                gemm_bf16_compat(w_dst, X, Y + n0, M, nt, K, N, n0 > 0, true);
            }
        }
    }

    void ensure_pf_attn(int K_len) {
        constexpr int Q_CH = 512;
        if (K_len <= pf_kv_cap && d_pf_q) return;
        INK_CK(cudaStreamSynchronize(st));
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p); fr(d_pf_s); fr(d_pf_o);
        pf_kv_cap = std::max(K_len, 2048);
        d_pf_q = dalloc<__nv_bfloat16>((size_t)kInkHeads * max_batch * kInkHeadDim);
        d_pf_k = dalloc<__nv_bfloat16>((size_t)kInkHeads * pf_kv_cap * kInkHeadDim);
        d_pf_v = dalloc<__nv_bfloat16>((size_t)kInkHeads * pf_kv_cap * kInkHeadDim);
        d_pf_s = dalloc<float>((size_t)kInkHeads * Q_CH * pf_kv_cap);
        d_pf_p = dalloc<__nv_bfloat16>((size_t)kInkHeads * Q_CH * pf_kv_cap);
        d_pf_o = dalloc<float>((size_t)kInkHeads * Q_CH * kInkHeadDim);
    }

    // ---- Execute 1 virtual visit (visit in 0..47, phys layer ip in 0..23) ----
    // mode: 0 = single-slot sequential (prefill or M=1 decode), 1 = batched multi-slot (B=1..64, 1 tok/slot), 2 = Tree-32 verify
    void run_visit(int visit, int ip, const DepthStatePtrs* bank_stp, int M, int mode, int slot0, int p0_val,
                   int& n_blocks, const float* (&block_ptrs)[8]) {
        const std::string p = "model.llm.layers." + std::to_string(ip);
        const bool is_swa = m.man.c.is_swa[ip] != 0;
        const bool has_msg = (bank_stp != nullptr);
        const bool exact_x = (mode == 2);
        const size_t n_hid = (size_t)M * kInkHidden;

        // 1. Attention-Residual Depth Mix (+ Fused Looped Pre-Attn Msg + CatRMS -> eliminates 48+72 launches!)
        prof_mark_start();
        const float* nw_q = (const float*)m.trunk_ptr(is_swa ? (p + ".attn_norm.weight") : (p + ".q_input_norm.weight"));
        const float* nw_k = (const float*)m.trunk_ptr(is_swa ? (p + ".attn_norm.weight") : (p + ".k_input_norm.weight"));
        const float* nw_v = (const float*)m.trunk_ptr(is_swa ? (p + ".attn_norm.weight") : (p + ".v_input_norm.weight"));

        const float* in_q = d_inp0;
        const float* in_k = d_inp0;
        const float* in_v = d_inp0;
        if (ip > 0) {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_res = (const float*)m.trunk_ptr("model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
            const int nq = is_swa ? 1 : 3;
            if (!has_msg) {
                k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_res, d_mixed, nq, M, kInkHidden, nw_q, is_swa ? nullptr : nw_k, is_swa ? nullptr : nw_v);
                in_q = d_mixed;
                in_k = is_swa ? d_mixed : (d_mixed + 1 * n_hid);
                in_v = is_swa ? d_mixed : (d_mixed + 2 * n_hid);
            } else {
                // FUSED 1-Launch: bank msg_0..3 + depth mixed_0..2 + cat3_rms (when M <= 16)!
                k_ink_attn_res_3x_cat_rms<<<dim3(M, 4), 256, 0, st>>>(
                    stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ip - m.man.c.core_start],
                    d_mixed, d_msgs, (M <= 16) ? d_ctrl_cat : nullptr, M, kInkHidden);
                const float* src0 = d_mixed;
                const float* src1 = d_mixed + 1 * n_hid;
                const float* src2 = d_mixed + 2 * n_hid;
                const float* wd0 = (const float*)m.trunk_ptr(p + ".controllers.attn.down_weight");
                const float* wg0 = (const float*)m.trunk_ptr(p + ".controllers.attn.gain_weight");
                const float* wf0 = (const float*)m.trunk_ptr(p + ".controllers.attn.forget_weight");
                const float* wd1 = (const float*)m.trunk_ptr(p + ".controllers.k.down_weight");
                const float* wg1 = (const float*)m.trunk_ptr(p + ".controllers.k.gain_weight");
                const float* wf1 = (const float*)m.trunk_ptr(p + ".controllers.k.forget_weight");
                const float* wd2 = (const float*)m.trunk_ptr(p + ".controllers.v.down_weight");
                const float* wg2 = (const float*)m.trunk_ptr(p + ".controllers.v.gain_weight");
                const float* wf2 = (const float*)m.trunk_ptr(p + ".controllers.v.forget_weight");
                if (M <= 16) {
                    constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                    if (M == 1) {
                        k_ink_ctrl_3x_down_smallM<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 4) {
                        k_ink_ctrl_3x_down_smallM<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 8) {
                        k_ink_ctrl_3x_down_smallM<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else {
                        k_ink_ctrl_3x_down_smallM<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    }
                    k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                } else {
                    run_loop_control_norm(src0, d_msgs + 0 * n_hid, wd0, wg0, wf0, nw_q, d_inp0, M, false, exact_x);
                    run_loop_control_norm(src1, d_msgs + 1 * n_hid, wd1, wg1, wf1, nw_k, d_inp1, M, false, exact_x);
                    run_loop_control_norm(src2, d_msgs + 2 * n_hid, wd2, wg2, wf2, nw_v, d_inp2, M, false, exact_x);
                }
                in_q = d_inp0; in_k = d_inp1; in_v = d_inp2;
            }
        } else {
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_cur, nw_q, d_inp0, nullptr, M, kInkHidden);
            in_q = in_k = in_v = d_inp0;
        }

        // 3. Fused Q, K, V, R, Gate_Attn, Sinks projections (1 launch on SWA-no-msg at M<=16 -> 1,500+ GB/s!)
        prof_mark_end(0); prof_mark_start();
        const bool share_qkv = (in_q == in_k && in_q == in_v);
        if (M <= 16 && share_qkv) {
            const std::string names[5] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight",
                p + ".attn.wk_dv.weight", p + ".attn.wv_dv.weight"
            };
            float* ys[5] = { d_q, d_r, d_gate_attn, d_k_raw, d_v_raw };
            linear_multi(names, ys, 5, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
        } else if (M <= 16) {
            const std::string names[3] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight"
            };
            float* ys[3] = { d_q, d_r, d_gate_attn };
            linear_multi(names, ys, 3, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
            linear(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, 0, false, exact_x);
            linear(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, 0, false, exact_x);
        } else {
            linear(p + ".attn.wq_du.weight", in_q, d_q, M, 0, false);
            linear(p + ".attn.wr_du.weight", in_q, d_r, M, 0, true);
            linear(p + ".attn.gate_attn.weight", in_q, d_gate_attn, M, 0, true);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false);
            linear(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, 0, share_qkv);
            linear(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, 0, share_qkv);
        }

        // 4. Dynamic K/V ShortConv tap projections on global layers
        float* k_dyn_ptr = nullptr;
        float* v_dyn_ptr = nullptr;
        if (!is_swa) {
            linear(p + ".attn.k_sconv.weight_proj", in_k, d_k_dyn, M);
            linear(p + ".attn.v_sconv.weight_proj", in_v, d_v_dyn, M);
            k_dyn_ptr = d_k_dyn;
            v_dyn_ptr = d_v_dyn;
        }
        prof_mark_end(1); prof_mark_start();
        auto run_sconv_uniform = [&](const float* src, const float* sw, const float* dyn, float* dst, int C, int ch_off, int nh, bool add_to_out = false) {
            float* hbase = d_sconv_pool[visit] + (size_t)ch_off * 3 * max_slots;
            if (mode == 0) {
                float* hslot = hbase + (size_t)slot0 * 3 * C;
                if (M > 16) {
                    dim3 g((C + 255) / 256, (M + 15) / 16);
                    k_ink_sconv4_seq_par<<<g, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, add_to_out);
                    k_ink_sconv4_save_hist<<<(C + 255) / 256, 256, 0, st>>>(src, hslot, M, C);
                } else {
                    k_ink_sconv4_seq<<<(C + 255) / 256, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, true, add_to_out);
                }
            } else if (mode == 1) {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_batched<<<g, 256, 0, st>>>(src, sw, dyn, hbase, d_slots, dst, M, C, nh, add_to_out);
            } else {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_tree<<<g, 256, 0, st>>>(
                    src, sw, dyn, hbase + (size_t)slot0 * 3 * C, d_sconv_lag, dst,
                    d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh, ch_off, M, C, nh, add_to_out);
            }
        };

        // 5-6. FUSED Post-QKV (K/V ShortConv4 + Q/K/V/R Clip + V RMSNorm + Global Tau + KV Write/Stage) in 1 launch for decode/batched/tree!
        const auto& ti_qc = m.tinfo(p + ".attn.q_l2_clip.scale");
        const auto& ti_rc = m.tinfo(p + ".attn.rel_l2_clip.scale");
        const int qc_elems = (int)(ti_qc.bytes / 4);
        const int rc_elems = (int)(ti_rc.bytes / 4);
        if ((mode == 0 && M == 1) || mode == 1 || mode == 2) {
            float* hk_pool = d_sconv_pool[visit];
            float* hv_pool = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots;
            k_ink_post_qkv_fused<<<dim3(64, M), 128, 0, st>>>(
                d_q, (const float*)(m.d_trunk + ti_qc.off), qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_pool, hv_pool,
                d_k, d_v, d_r, (const float*)(m.d_trunk + ti_rc.off), rc_elems,
                (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                (mode == 2) ? d_sconv_lag : nullptr,
                (mode == 2) ? (d_stage_k + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_v + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh) : nullptr,
                kv_desc[visit], !is_swa, mode, M);
        } else {
            run_sconv_uniform(d_k_raw, m.d_sconv_k[ip], k_dyn_ptr, d_k, kInkKVDim, 0, kInkKVHeads);
            run_sconv_uniform(d_v_raw, m.d_sconv_v[ip], v_dyn_ptr, d_v, kInkKVDim, kInkKVDim, kInkKVHeads);
            k_ink_qkvr_clip_tau<<<dim3(64, M), 128, 0, st>>>(
                d_q, (const float*)(m.d_trunk + ti_qc.off), qc_elems,
                d_k, d_v, d_r, (const float*)(m.d_trunk + ti_rc.off), rc_elems,
                d_pos, !is_swa, M);
            k_ink_kv_write<<<dim3(kInkKVHeads, M), 128, 0, st>>>(d_k, d_v, nullptr, slot0, d_pos, kv_desc[visit], M);
        }

        // 7. RelPos Table + Fused Sink Flash-GQA (Split-K for M<=16 or Tree-32, BF16 TC Batched GEMM for prefill M>=128!)
        prof_mark_end(2); prof_mark_start();
        {
            dim3 gr(kInkHeads, M);
            k_ink_rel_table<<<gr, 256, 0, st>>>(d_r, (const float*)m.trunk_ptr(p + ".attn.rel_logits_proj.proj"), d_rel_table, M);
            if (M <= 16 || mode == 2) {
                dim3 gsp(kInkKVHeads, kInkAttnSplits, M);
                k_ink_flash_gqa_splitk<<<gsp, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks,
                    kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                    (mode == 2) ? d_k : nullptr, (mode == 2) ? d_v : nullptr,
                    (mode == 2) ? d_tree_anc : nullptr, (mode == 2) ? d_tree_dep : nullptr,
                    p0_val, d_attn_pm, d_attn_pl, d_attn_po, M);
                k_ink_flash_gqa_merge<<<gr, 128, 0, st>>>(d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out, M);
            } else if (mode == 0 && M >= 128 && (m.sm_cc >= 70 || m.sm_cc == 60) && (d_pf_q != nullptr || [&](){ size_t vf=0,vt=0; cudaMemGetInfo(&vf,&vt); return vf > (480ull<<20); }())) {
                // High-speed Causal/SWA-Sliced Tensor Core Prefill Attention (BF16 on sm_80+, FP16 on sm_60..75!)
                const bool use_fp16_pf = (m.sm_cc < 80);
                const cudaDataType_t pf_dt = use_fp16_pf ? CUDA_R_16F : CUDA_R_16BF;
                const cublasGemmAlgo_t pf_algo = (m.sm_cc >= 70) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
                const int k_start = is_swa ? std::max(0, p0_val - kInkSwaWin + 1) : 0;
                const int K_len = (p0_val + M) - k_start;
                ensure_pf_attn(K_len);
                if (use_fp16_pf) {
                    k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);
                    k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                } else {
                    k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);
                    k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                }
                constexpr int Q_CH = 256;
                const float sc_qk = 0.08838834764831845f;
                const float one = 1.0f, zero = 0.0f;
                for (int q0 = 0; q0 < M; q0 += Q_CH) {
                    const int C = std::min(Q_CH, M - q0);
                    const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
                    const int k_sub_end = p0_val + q0 + C;
                    const int k_off = k_sub_beg - k_start;
                    const int K_sub = std::max(1, k_sub_end - k_sub_beg);
                    // S[h, C, K_sub] = (1/sqrt(128)) * Q[h, q0..q0+C, 128] @ K[h, k_off..k_off+K_sub, 128]^T
                    INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        K_sub, C, kInkHeadDim, &sc_qk,
                        d_pf_k + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                        d_pf_q + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                        &zero, d_pf_s, CUDA_R_32F, K_sub, (long long)C * K_sub,
                        kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                    if (use_fp16_pf) {
                        k_ink_pf_softmax_rel_sink<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                            d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                    } else {
                        k_ink_pf_softmax_rel_sink<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                            d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                    }
                    // O[h, C, 128] = P[h, C, K_sub] @ V[h, k_off..k_off+K_sub, 128]
                    INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        kInkHeadDim, C, K_sub, &one,
                        d_pf_v + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                        d_pf_p, pf_dt, K_sub, (long long)C * K_sub,
                        &zero, d_pf_o, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                        kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                    k_ink_pf_unpack_o_gate<<<dim3(kInkHeads, C), 128, 0, st>>>(d_pf_o, d_gate_attn, d_attn_out, q0, C);
                }
            } else {
                dim3 g3(kInkKVHeads, M);
                k_ink_flash_gqa_gqa3<<<g3, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks, d_gate_attn,
                    kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                    d_attn_out, M);
            }
        }
        prof_mark_end(3); prof_mark_start();

        // 8. Output projection wo + FUSED post-attn ShortConv + residual add / block boundary!
        linear(p + ".attn.wo_ud.weight", d_attn_out, d_wo_raw, M, 0, false, (mode == 2));
        prof_mark_end(4); prof_mark_start();
        if (ip % m.man.c.block_size == 0) {
            // Append residual (d_cur) to blocks, then overwrite d_cur directly with ShortConv(d_wo_raw)!
            INK_CK(cudaMemcpyAsync(d_blocks[n_blocks], d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            block_ptrs[n_blocks] = d_blocks[n_blocks];
            ++n_blocks;
            run_sconv_uniform(d_wo_raw, m.d_sconv_attn[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim, 0, false);
        } else {
            run_sconv_uniform(d_wo_raw, m.d_sconv_attn[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim, 0, true);
        }

        // 9. MoE Sublayer: Fused Pre-MLP Mix+CatRMS -> SITU MoE -> Fused Reduce8 + Post-MLP ShortConv4 + Add!
        const float* mlp_nw = (const float*)m.trunk_ptr(p + ".mlp_norm.weight");
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_mlp = (const float*)m.trunk_ptr("model.llm.attn_res.mlp_queries." + std::to_string(ip));
            if (has_msg) {
                k_ink_mlp_res_cat_rms<<<M, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, d_ctrl_cat, M, kInkHidden);
                run_loop_control_norm(
                    d_moe_in, d_msgs + 3 * n_hid,
                    (const float*)m.trunk_ptr(p + ".controllers.mlp.down_weight"),
                    (const float*)m.trunk_ptr(p + ".controllers.mlp.gain_weight"),
                    (const float*)m.trunk_ptr(p + ".controllers.mlp.forget_weight"),
                    mlp_nw, d_moe_in, M, /*cat_ready=*/true, exact_x);
            } else {
                k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden, mlp_nw);
            }
        }
        prof_mark_end(0); prof_mark_start();

        if (m.layer_has_host[ip] && m.hybrid.active) {
            launch_ink_moe_hybrid(
                st, cublas, m.d_sec4, m.d_experts[ip], m.moe_info[ip], ip, m.hybrid,
                (const float*)m.trunk_ptr(p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr(p + ".mlp.gate.bias"),
                d_moe_in, d_moe_raw, M, moe_ws, m.man.c.route_scale, (mode == 2));
            prof_mark_end(5); prof_mark_start();
            run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
        } else {
            const bool fuse_moe_sconv = ((mode == 0 && M == 1) || mode == 1 || mode == 2);
            launch_ink_moe(
                st, cublas, m.d_sec4, m.d_experts[ip], m.moe_info[ip],
                (const float*)m.trunk_ptr(p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr(p + ".mlp.gate.bias"),
                d_moe_in, d_moe_raw, M, moe_ws, m.man.c.route_scale, (mode == 2), !fuse_moe_sconv);
            prof_mark_end(5); prof_mark_start();
            if (fuse_moe_sconv) {
                constexpr int mlp_ch_off = 2 * kInkKVDim + kInkHidden;
                float* hbase = d_sconv_pool[visit] + (size_t)mlp_ch_off * 3 * max_slots;
                if (mode == 0 && M == 1) {
                    k_ink_moe_reduce8_sconv_m1<<<(kInkHidden + 255) / 256, 256, 0, st>>>(
                        moe_ws.y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden, d_cur);
                } else if (mode == 1) {
                    k_ink_moe_reduce8_sconv_batched<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(
                        moe_ws.y_pair, m.d_sconv_mlp[ip], hbase, d_slots, d_cur, M);
                } else {
                    k_ink_moe_reduce8_sconv_tree<<<(kInkHidden + 127) / 128, 128, 0, st>>>(
                        moe_ws.y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden,
                        d_sconv_lag, d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh,
                        mlp_ch_off, d_cur, M);
                }
            } else {
                run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
            }
        }
        prof_mark_end(2);
    }

    // ---- Full 48-Visit Body (Capturable inside a SINGLE CUDA Graph!) ----
    void emit_48_visits(int M, int mode, int slot0, int tree_p0, bool want_taps) {
        const size_t n_hid = (size_t)M * kInkHidden;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<M, 256, 0, st>>>(
            m.d_trunk + ei.off, ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_tok_ids, has_img_step ? d_is_img : nullptr, has_img_step ? d_img_emb : nullptr, d_cur, M, kInkHidden);

        const float* block_ptrs[8] = {};
        int n_blocks = 0;
        int visit = 0;

        auto maybe_tap = [&](int v, const float* src) {
            if (!want_taps) return;
            if (v == 5)       INK_CK(cudaMemcpyAsync(d_taps[0], src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            else if (v == 23) INK_CK(cudaMemcpyAsync(d_taps[1], src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            else if (v == 41) INK_CK(cudaMemcpyAsync(d_taps[2], src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        };

        // 1. Prelude: physical layers 0..5 (visits 0..5)
        for (int ip = 0; ip < m.man.c.core_start; ++ip) {
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }

        // Save prelude state & prelude_blocks (n_blocks == 1 here!)
        INK_CK(cudaMemcpyAsync(d_prelude, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        const int n_pre_blk = n_blocks;
        for (int b = 0; b < n_pre_blk; ++b) {
            INK_CK(cudaMemcpyAsync(d_prelude_blocks[b], block_ptrs[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
        }

        // 2. Looped Core: physical layers 6..17 x 3 passes (visits 6..41)
        const int n_passes = (max_passes_override > 0 && max_passes_override < m.man.c.num_passes)
                                 ? max_passes_override
                                 : m.man.c.num_passes;
        int n_bank = 0;
        for (int pass = 0; pass < n_passes; ++pass) {
            if (pass > 0) {
                // Bank = previous pass's blocks after the first core block + bank_out (cur from Layer 17)
                n_bank = n_blocks - (n_pre_blk + 1);
                for (int b = 0; b < n_bank; ++b) {
                    INK_CK(cudaMemcpyAsync(d_bank[b], block_ptrs[n_pre_blk + 1 + b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                }
                INK_CK(cudaMemcpyAsync(d_bank_out, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                // Reset cur = prelude, blocks = prelude_blocks
                INK_CK(cudaMemcpyAsync(d_cur, d_prelude, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                n_blocks = n_pre_blk;
                for (int b = 0; b < n_pre_blk; ++b) {
                    INK_CK(cudaMemcpyAsync(d_blocks[b], d_prelude_blocks[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                    block_ptrs[b] = d_blocks[b];
                }
            }
            DepthStatePtrs bst{};
            if (pass > 0) {
                for (int b = 0; b < n_bank; ++b) bst.p[b] = d_bank[b];
                bst.p[n_bank] = d_bank_out;
                bst.count = n_bank + 1;
            }
            for (int ip = m.man.c.core_start; ip < m.man.c.core_end; ++ip) {
                maybe_tap(visit, d_cur);
                run_visit(visit, ip, (pass > 0) ? &bst : nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
                ++visit;
            }
        }

        // 3. Coda: physical layers 18..23 (visits 42..47)
        visit = 42;
        for (int ip = m.man.c.core_end; ip < m.man.c.n_phys; ++ip) {
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }

        // 4. Output Depth Mix + Output RMSNorm -> d_h_norm (and tap 48!)
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, M, kInkHidden);
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, M, kInkHidden);
            if (want_taps) INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        }
    }

    // ---- Launch Full 48-Visit Forward (with Automatic CUDA Graph Capture for Decode/Batched/Tree!) ----
    void forward_seq(const int32_t* tok_ids, int p0, int M, int slot0, bool want_taps) {
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)M * 4);
        int32_t* h_pos = slot_buf + M;
        for (int i = 0; i < M; ++i) h_pos[i] = p0 + i;
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, h_pos, (size_t)M * 4, cudaMemcpyHostToDevice, st));

        // Use CUDA Graph for single-token / small-M decode!
        if (use_cuda_graph && !has_img_step && max_passes_override == 0 && M == 1 && slot0 == 0) {
            GraphKey key{M, 0, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                INK_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                emit_48_visits(M, 0, slot0, p0, want_taps);
                INK_CK(cudaStreamEndCapture(st, &gr));
                INK_CK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));
                INK_CK(cudaGraphDestroy(gr));
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_promote(m.d_sec4, m.d_experts, st);
            return;
        }
        emit_48_visits(M, 0, slot0, p0, want_taps);
        if (M == 1 && m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_promote(m.d_sec4, m.d_experts, st);
    }

    // Speculative Tree-32 / Block-8 verify (100% CUDA-Graph capturable across ALL p0 when 100% VRAM!)
    void forward_tree_verify(const InkTreeSpec& tree, int p0, int slot0, bool want_taps) {
        const int M = tree.M;
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        int32_t* h_tok = slot_buf;
        int32_t* h_pos = slot_buf + kInkMaxTree;
        uint8_t* h_bytes = reinterpret_cast<uint8_t*>(slot_buf + 2 * kInkMaxTree);
        for (int i = 0; i < M; ++i) {
            h_tok[i] = tree.tok[i];
            h_pos[i] = p0 + (int)tree.depth[i];
        }
        std::memcpy(h_bytes,       tree.sconv_lag, kInkMaxTree * 3);
        std::memcpy(h_bytes + 128, tree.anc,       kInkMaxTree * kInkMaxTreeDepth);
        std::memcpy(h_bytes + 384, tree.depth,     kInkMaxTree);

        INK_CK(cudaMemcpyAsync(d_tok_ids,   h_tok,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_pos,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_sconv_lag, h_bytes,       kInkMaxTree * 3, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_anc,  h_bytes + 128, kInkMaxTree * kInkMaxTreeDepth, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_dep,  h_bytes + 384, kInkMaxTree, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && max_passes_override == 0 && slot0 == 0 && M <= tree_cap && m.n_host_experts == 0) {
            GraphKey key{M, 2, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                INK_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                emit_48_visits(M, 2, slot0, 0, want_taps);
                INK_CK(cudaStreamEndCapture(st, &gr));
                INK_CK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));
                INK_CK(cudaGraphDestroy(gr));
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            return;
        }
        emit_48_visits(M, 2, slot0, p0, want_taps);
    }

    // Batched multi-slot 1-token-per-slot step (B = 1..64 concurrent requests, captured in CUDA Graph when 100% VRAM!)
    void forward_batched(const int32_t* tok_ids, const int32_t* slot_ids, const int32_t* pos_arr, int B, bool want_taps = false) {
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)B * 4);
        std::memcpy(slot_buf + B, slot_ids, (size_t)B * 4);
        std::memcpy(slot_buf + 2 * B, pos_arr, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_slots, slot_buf + B, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, slot_buf + 2 * B, (size_t)B * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && B <= 64 && (m.n_host_experts == 0 || B == 1)) {
            GraphKey key{B, 10, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                INK_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                emit_48_visits(B, 1, 0, 0, want_taps);
                INK_CK(cudaStreamEndCapture(st, &gr));
                INK_CK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));
                INK_CK(cudaGraphDestroy(gr));
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            return;
        }
        emit_48_visits(B, 1, 0, 0, want_taps);
    }

    // ---- Unembed + Softcap(30.0) + Argmax ----
    // Fast exact 2-stage screened argmax (0.14 ms on RTX 5090, 100.000% mathematically bit-exact up to M <= 32!)
    void head_argmax_exact(int M, int32_t* h_out_ids) {
        prof_mark_start();
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (m.unembed_screen.ready && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K) && M <= kInkMaxTree) {
            const int V = m.man.c.vocab_unpadded;
            const int g1 = (V + 7) / 8;
            for (int m0 = 0; m0 < M; m0 += 8) {
                const int mc = std::min(8, M - m0);
                const float* hm = d_h_norm + (size_t)m0 * kInkHidden;
                k_ink_row_l2norm<<<mc, 256, 0, st>>>(hm, d_h_norm2, d_max_lb, d_cand_cnt, mc);
                if (mc == 1) {
                    k_ink_screen_pass1_multiM<1><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 4) {
                    k_ink_screen_pass1_multiM<4><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else {
                    k_ink_screen_pass1_multiM<8><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                }
                dim3 g2((V + 255) / 256, mc);
                k_ink_screen_pass2_bf16<<<g2, 256, 0, st>>>(
                    nullptr, m.unembed_screen.eps,
                    hm, d_h_norm2, d_dot_hat, d_max_lb, d_cand_ids, d_cand_cnt, 4096, mc, V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_screen_eval_bf16<<<mc, 256, 0, st>>>(
                        (const uint16_t*)(m.d_trunk + ui.off), hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                } else {
                    k_ink_screen_eval_s6k<<<mc, 256, 0, st>>>(
                        m.d_trunk + ui.off, hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                }
            }
            prof_mark_end(7);
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return;
        }
        head_logits_full(M);
        k_ink_argmax_rows<<<M, 1024, 0, st>>>(d_logits, m.man.c.vocab_unpadded, d_argmax);
        prof_mark_end(7);
        INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
    }

    // Full exact softcapped logits [n_want, 200058] for selected hidden rows in d_h_src -> d_logits
    void head_logits_from(const float* d_h_src, int n_want) {
        ensure_logits(n_want);
        const int V = m.man.c.vocab_unpadded;
        linear("model.llm.unembed.weight", d_h_src, d_logits, n_want, V);
        const size_t n = (size_t)n_want * V;
        k_ink_softcap<<<(n + 255) / 256, 256, 0, st>>>(d_logits, n, m.man.c.logit_softcap);
    }

    // Full exact softcapped logits [M, 200058] into d_logits
    void head_logits_full(int M) {
        head_logits_from(d_h_norm, M);
    }

    void free_all() {
        for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        graph_cache.clear();
        moe_ws.free_all();
        for (int v = 0; v < kInkVisits; ++v) {
            if (kv_desc[v].k_f16) cudaFree(kv_desc[v].k_f16);
            if (kv_desc[v].v_f16) cudaFree(kv_desc[v].v_f16);
            if (kv_desc[v].k_f8)  cudaFree(kv_desc[v].k_f8);
            if (kv_desc[v].v_f8)  cudaFree(kv_desc[v].v_f8);
            if (kv_desc[v].k_sc)  cudaFree(kv_desc[v].k_sc);
            if (kv_desc[v].v_sc)  cudaFree(kv_desc[v].v_sc);
            if (d_sconv_pool[v])  cudaFree(d_sconv_pool[v]);
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_cur); fr(d_res); fr(d_prelude); fr(d_bank_out);
        for (int i = 0; i < 8; ++i) fr(d_blocks[i]);
        for (int i = 0; i < 2; ++i) fr(d_prelude_blocks[i]);
        for (int i = 0; i < 4; ++i) fr(d_bank[i]);
        fr(d_mixed); fr(d_msgs); fr(d_inp0); fr(d_inp1); fr(d_inp2);
        fr(d_q); fr(d_k_raw); fr(d_v_raw); fr(d_k); fr(d_v); fr(d_r);
        fr(d_k_dyn); fr(d_v_dyn); fr(d_sinks); fr(d_gate_attn); fr(d_rel_table);
        fr(d_attn_out); fr(d_wo_raw); fr(d_wo_conv); fr(d_moe_in); fr(d_moe_raw); fr(d_moe_conv);
        fr(d_h_norm);
        for (int i = 0; i < 4; ++i) { fr(d_taps[i]); fr(d_taps_compact[i]); }
        fr(d_ctrl_cat); fr(d_ctrl_gate); fr(d_ctrl_gain); fr(d_ctrl_forget);
        fr(d_attn_pm); fr(d_attn_pl); fr(d_attn_po);
        fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p); fr(d_pf_s); fr(d_pf_o);
        fr(d_stage_k); fr(d_stage_v); fr(d_stage_sconv);
        fr(d_kv_desc_arr); fr(d_sconv_pool_arr); fr(d_acc_path);
        fr(d_x_bf16); fr(d_w_scratch); fr(d_logits);
        fr(d_dot_hat); fr(d_h_norm2); fr(d_max_lb); fr(d_cand_ids); fr(d_cand_cnt); fr(d_argmax);
        fr(d_tok_ids); fr(d_slots); fr(d_pos); fr(d_is_img); fr(d_img_emb); fr(d_sconv_lag); fr(d_tree_anc); fr(d_tree_dep);
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (ev_p0) { cudaEventDestroy(ev_p0); ev_p0 = nullptr; }
        if (ev_p1) { cudaEventDestroy(ev_p1); ev_p1 = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (st) { cudaStreamDestroy(st); st = nullptr; }
    }
};

}  // namespace strata::ink
