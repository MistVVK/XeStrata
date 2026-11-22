// engine/src/inkling2/cuda/ink_cuda_backend.cu - InkBackend CUDA Plugin for `strata-inkling` (sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"

#include "ink_cuda.cuh"
#include "ink_dspark.cuh"
#include "ink_runner.cuh"
#include "ink_spec.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// Copy 1 slot's 48-visit ShortConv state [48 x 3 x 6656] <-> snapshot buffer (2.5 us on GDDR7!)
__global__ void __launch_bounds__(256) k_ink_sconv_slot_copy(
    float* const* __restrict__ sconv_pools,  // [48]
    float* __restrict__ snap_buf,            // [48, 3 * 6656]
    int slot,
    int max_slots,
    bool to_snap) {
    const int v = blockIdx.x;
    const int strm = blockIdx.y;
    const int tid = threadIdx.x;
    if (v >= kInkVisits || strm >= 4) return;
    constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
    constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
    const int ch_off = c_off[strm];
    const int C = c_len[strm];

    float* slot_h = sconv_pools[v] + (size_t)ch_off * 3 * max_slots + (size_t)slot * 3 * C;
    float* snap_h = snap_buf + (size_t)v * 3 * kInkSconvCh + (size_t)ch_off * 3;
    const int n = 3 * C;
    if (to_snap) {
        for (int i = tid; i < n; i += 256) snap_h[i] = slot_h[i];
    } else {
        for (int i = tid; i < n; i += 256) slot_h[i] = snap_h[i];
    }
}

// Gather selected hidden rows d_h_norm[idx[k], 2304] -> d_h_sel[k, 2304]
__global__ void __launch_bounds__(256) k_ink_gather_rows(
    const float* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int n_sel) {
    const int k = blockIdx.x;
    if (k >= n_sel) return;
    const int row = idx[k];
    const float4* s4 = reinterpret_cast<const float4*>(src + (size_t)row * kInkHidden);
    float4* d4 = reinterpret_cast<float4*>(dst + (size_t)k * kInkHidden);
    for (int i = threadIdx.x; i < (kInkHidden / 4); i += 256) d4[i] = s4[i];
}

}  // namespace strata::ink

namespace ink2 {

class InkCudaBackend final : public InkBackend {
public:
    ~InkCudaBackend() override { cleanup(); }

    const char* backend_name() const noexcept override { return "cuda"; }

    std::string device_info() const override { return dev_info_; }

    BackendStats stats() const override {
        BackendStats st{};
        if (initialized_) {
            st.vram_experts_loaded = model_.n_vram_experts;
            st.ram_experts_loaded  = model_.n_host_experts;
            st.total_experts       = model_.n_vram_experts + model_.n_host_experts;
            size_t vf = 0, vt = 0;
            if (cudaMemGetInfo(&vf, &vt) == cudaSuccess) {
                st.vram_total_bytes = vt;
                st.vram_used_bytes  = vt - vf;
            }
            st.max_slots = max_slots_;
            int act = 0;
            for (bool a : slot_alloc_) if (a) ++act;
            st.active_slots = act;
            st.kv_dtype = kv_dtype_;
        }
        return st;
    }

    bool init(const InkModelFile& mf, const BackendOpts& opts, std::string& err) override {
        cleanup();
        gpu_id_ = opts.gpu_index;
        cudaError_t ce = cudaSetDevice(gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("cudaSetDevice failed: ") + cudaGetErrorString(ce);
            return false;
        }
        cudaDeviceProp prop{};
        ce = cudaGetDeviceProperties(&prop, gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(ce);
            return false;
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%.200s (sm_%d%d, %d SMs, %.1f GiB VRAM, sm120_tc=%d)",
                      prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                      (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                      (int)strata::ink::sm120::available());
        dev_info_ = buf;

        // Compute exact minimum KV + ShortConv + workspace headroom for parallel slots before pinning experts!
        max_slots_ = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        kv_dtype_ = opts.kv_dtype;
        if (opts.kv_dtype_auto && (max_slots_ >= 16 || prop.totalGlobalMem <= (13ull << 30)) && prop.totalGlobalMem < (40ull << 30)) {
            kv_dtype_ = KvDtype::FP8_E4M3;
        }
        const int eff_chunk = std::max(64, opts.chunk);
        const size_t swa_r = (max_slots_ >= 16) ? (size_t)(strata::ink::kInkSwaWin + 32)
                                                : (size_t)(strata::ink::kInkSwaWin + std::max(eff_chunk, 64));
        const size_t bpe = (kv_dtype_ == KvDtype::FP8_E4M3) ? 2ull : 4ull;
        const size_t sc_per_pos = (kv_dtype_ == KvDtype::FP8_E4M3) ? (size_t)(2 * strata::ink::kInkKVHeads * 4) : 0ull;
        const size_t min_kv_bytes = (size_t)40 * max_slots_ * swa_r * ((size_t)strata::ink::kInkKVDim * bpe + sc_per_pos)
                                  + (size_t)strata::ink::kInkVisits * max_slots_ * 3 * strata::ink::kInkSconvCh * 4
                                  + (size_t)8 * max_slots_ * 256 * ((size_t)strata::ink::kInkKVDim * bpe + sc_per_pos);
        const size_t ws_pad = (max_slots_ >= 8) ? (1536ull << 20)
                            : ((prop.totalGlobalMem >= (20ull << 30)) ? (1200ull << 20)
                             : (prop.totalGlobalMem <= (13ull << 30)) ? (260ull << 20)
                                                                      : (320ull << 20));
        const size_t min_reserve_bytes = min_kv_bytes + ws_pad;

        // Load .strata v1.1 trunk + experts (100% VRAM or 3x loop-weighted vram_load_order!)
        const bool build_screen = opts.enable_dspark && mf.m.dspark.present;
        if (!model_.load(mf.path, gpu_id_, build_screen, err, min_reserve_bytes)) return false;

        auto cuda_kvd = (kv_dtype_ == KvDtype::FP8_E4M3) ? strata::ink::InkKvDtype::FP8
                                                          : strata::ink::InkKvDtype::FP16;

        runner_ = std::make_unique<strata::ink::InkCudaRunner>(model_);
        runner_->use_cuda_graph = true;
        runner_->init(max_slots_, std::max(256, opts.max_ctx), std::max(64, opts.chunk), cuda_kvd);
        eff_max_ctx_ = runner_->max_ctx;

        if (opts.enable_dspark && model_.man.has_dspark) {
            dspark_ = std::make_unique<strata::ink::InkDSpark>(*runner_);
            dspark_->init(eff_max_ctx_, 512);
        }

        // Allocate snapshot buffer (3.83 MB) + selected hidden gather buffers + pinned D2H logits staging
        d_sconv_snap_ = strata::ink::dalloc<float>((size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh);
        d_h_sel_      = strata::ink::dalloc<float>((size_t)runner_->max_batch * strata::ink::kInkHidden);
        d_sel_idx_    = strata::ink::dalloc<int32_t>(runner_->max_batch);
        slot_alloc_.assign(max_slots_, false);
        snap_active_.assign(max_slots_, false);
        snap_p0_.assign(max_slots_, 0);
        snap_M_.assign(max_slots_, 0);
        initialized_ = true;
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) {
            err = "invalid slot " + std::to_string(slot);
            return false;
        }
        slot_alloc_[slot] = true;
        reset_slot(slot);
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < max_slots_) {
            slot_alloc_[slot] = false;
            snap_active_[slot] = false;
            if (slot == 0) tree_snap_ready_ = false;
        }
    }

    void reset_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) return;
        runner_->reset_slot(slot);
        snap_active_[slot] = false;
        if (slot == 0) { dspark_slot_ = -1; tree_snap_ready_ = false; }
    }

    void snapshot_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) return;
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, true);
        snap_active_[slot] = true;
        snap_M_[slot] = 0;
        if (slot == 0) tree_snap_ready_ = false;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) return;
        if (snap_active_[slot]) {
            const int n_keep = std::max(0, std::min(snap_M_[slot], keep_len - snap_p0_[slot]));
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, false);
            if (n_keep > 0) {
                int32_t path[16];
                for (int i = 0; i < n_keep; ++i) path[i] = i;
                INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)n_keep * 4, cudaMemcpyHostToDevice, runner_->st));
                strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                    runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_keep, slot, runner_->max_slots, runner_->tree_cap);
            }
            snap_active_[slot] = false;
            return;
        }
        if (slot == 0 && tree_snap_ready_) {
            // Exact multi-branch tree rewind on early EOS / stop_tok / anti-loop!
            const int n_keep = std::max(0, std::min(tree_snap_M_, keep_len - tree_snap_p0_));
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, 0, runner_->max_slots, false);
            if (n_keep > 0) {
                INK_CK(cudaMemcpyAsync(runner_->d_acc_path, tree_snap_path_, (size_t)n_keep * 4, cudaMemcpyHostToDevice, runner_->st));
                strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                    runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_keep, 0, runner_->max_slots, runner_->tree_cap);
            }
            tree_snap_ready_ = false;
        }
    }

    bool forward_batch(
        const StepBatch& batch,
        float* logits_out,
        TapCapture* taps_out,
        std::string& err) override {
        if (!initialized_) {
            err = "CUDA backend not initialized";
            return false;
        }
        const int N = batch.n_tokens;
        if (N <= 0) return true;
        tree_snap_ready_ = false;
        const int V = model_.man.c.vocab_unpadded;

        // Pre-compute packed output row index for every token with want_logits[i] != 0
        std::vector<int> out_row(N, -1);
        int total_want = 0;
        for (int i = 0; i < N; ++i) {
            if (batch.want_logits[i]) out_row[i] = total_want++;
        }

        if (taps_out) {
            for (int t = 0; t < 4; ++t) taps_out->taps[t].assign((size_t)N * strata::ink::kInkHidden, 0.0f);
        }

        // Helper: extract wanted logits rows from runner_->d_h_norm[0..M-1] into logits_out
        auto emit_selected_logits = [&](const std::vector<int>& local_idx, const std::vector<int>& dst_rows) {
            const int nw = (int)local_idx.size();
            if (nw == 0 || !logits_out) return;
            const float* h_src = runner_->d_h_norm;
            if (nw == 1) {
                h_src = runner_->d_h_norm + (size_t)local_idx[0] * strata::ink::kInkHidden;
            } else {
                bool contiguous_0 = true;
                for (int k = 0; k < nw; ++k) if (local_idx[k] != k) { contiguous_0 = false; break; }
                if (!contiguous_0) {
                    INK_CK(cudaMemcpyAsync(d_sel_idx_, local_idx.data(), (size_t)nw * 4, cudaMemcpyHostToDevice, runner_->st));
                    strata::ink::k_ink_gather_rows<<<nw, 256, 0, runner_->st>>>(runner_->d_h_norm, d_sel_idx_, d_h_sel_, nw);
                    h_src = d_h_sel_;
                }
            }
            runner_->head_logits_from(h_src, nw);
            // Use pinned host staging buffer for full PCIe 5.0 DMA bandwidth (~50 GB/s vs ~12 GB/s pageable)!
            ensure_pinned_logits(nw, V);
            INK_CK(cudaMemcpyAsync(h_logits_pin_, runner_->d_logits,
                                   (size_t)nw * V * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
            INK_CK(cudaStreamSynchronize(runner_->st));
            bool contiguous_dst = true;
            for (int k = 1; k < nw; ++k) if (dst_rows[k] != dst_rows[k - 1] + 1) { contiguous_dst = false; break; }
            if (contiguous_dst) {
                std::memcpy(logits_out + (size_t)dst_rows[0] * V, h_logits_pin_, (size_t)nw * V * sizeof(float));
            } else {
                for (int k = 0; k < nw; ++k) {
                    std::memcpy(logits_out + (size_t)dst_rows[k] * V,
                                h_logits_pin_ + (size_t)k * V, (size_t)V * sizeof(float));
                }
            }
        };

        auto copy_host_taps = [&](const std::vector<int>& orig_indices) {
            if (!taps_out) return;
            const int M = (int)orig_indices.size();
            for (int t = 0; t < 4; ++t) {
                std::vector<float> tmp((size_t)M * strata::ink::kInkHidden);
                INK_CK(cudaMemcpyAsync(tmp.data(), runner_->d_taps[t], tmp.size() * 4, cudaMemcpyDeviceToHost, runner_->st));
                INK_CK(cudaStreamSynchronize(runner_->st));
                for (int k = 0; k < M; ++k) {
                    std::memcpy(taps_out->taps[t].data() + (size_t)orig_indices[k] * strata::ink::kInkHidden,
                                tmp.data() + (size_t)k * strata::ink::kInkHidden,
                                strata::ink::kInkHidden * sizeof(float));
                }
            }
        };

        // Helper: run a contiguous single-slot sequence (chunked to runner_->max_batch if needed!)
        auto run_single_slot_seq = [&](int slot, const std::vector<int>& idxs) {
            const int total_m = (int)idxs.size();
            int off = 0;
            while (off < total_m) {
                const int M = std::min(total_m - off, runner_->max_batch);
                const int p0 = batch.pos[idxs[off]];
                std::vector<int32_t> toks(M);
                std::vector<uint8_t> is_img(M, 0);
                bool any_img = false;
                for (int k = 0; k < M; ++k) {
                    const int oi = idxs[off + k];
                    toks[k] = batch.token_ids[oi];
                    if (batch.img_embeds && oi < (int)batch.img_embed_idx.size() && batch.img_embed_idx[oi] >= 0) {
                        is_img[k] = 1;
                        any_img = true;
                    }
                }
                if (any_img) {
                    std::vector<float> img_flat((size_t)M * strata::ink::kInkHidden, 0.0f);
                    for (int k = 0; k < M; ++k) {
                        if (is_img[k]) {
                            const int img_r = batch.img_embed_idx[idxs[off + k]];
                            std::memcpy(img_flat.data() + (size_t)k * strata::ink::kInkHidden,
                                        batch.img_embeds + (size_t)img_r * strata::ink::kInkHidden,
                                        strata::ink::kInkHidden * sizeof(float));
                        }
                    }
                    INK_CK(cudaMemcpyAsync(runner_->d_is_img, is_img.data(), M, cudaMemcpyHostToDevice, runner_->st));
                    INK_CK(cudaMemcpyAsync(runner_->d_img_emb, img_flat.data(), img_flat.size() * 4, cudaMemcpyHostToDevice, runner_->st));
                    runner_->has_img_step = true;
                } else {
                    runner_->has_img_step = false;
                }

                const bool want_taps = (taps_out != nullptr) || (batch.capture_taps && dspark_ && dspark_->ready && slot == 0);

                // If snapshot_slot(slot) is active for DSpark verify (M <= 16, no images):
                // run forward_tree_verify (stages ShortConv + KV) & commit all M tokens so rollback_slot can replay exact n_keep!
                if (snap_active_[slot] && !any_img && M <= 16 && total_m <= 16) {
                    strata::ink::InkTreeSpec tr{};
                    tr.M = M;
                    for (int k = 0; k < M; ++k) {
                        tr.tok[k] = toks[k];
                        tr.parent[k] = (int8_t)(k - 1);
                        tr.depth[k] = (uint8_t)k;
                    }
                    tr.build();
                    runner_->forward_tree_verify(tr, p0, slot, want_taps);
                    int32_t path[16];
                    for (int k = 0; k < M; ++k) path[k] = k;
                    INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)M * 4, cudaMemcpyHostToDevice, runner_->st));
                    strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                        runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, M, slot, runner_->max_slots, runner_->tree_cap);
                    strata::ink::k_ink_commit_kv_all48<<<dim3(strata::ink::kInkVisits, strata::ink::kInkKVHeads), 128, 0, runner_->st>>>(
                        runner_->d_kv_desc_arr, runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, p0, M, slot, runner_->tree_cap);
                    snap_p0_[slot] = p0;
                    snap_M_[slot] = M;
                } else {
                    runner_->forward_seq(toks.data(), p0, M, slot, want_taps);
                }
                runner_->has_img_step = false;

                if (batch.capture_taps && dspark_ && dspark_->ready && slot == 0 && p0 + M <= dspark_->max_ctx) {
                    dspark_->ingest_taps(p0, M);
                    dspark_slot_ = 0;
                }
                if (taps_out) {
                    std::vector<int> slice(idxs.begin() + off, idxs.begin() + off + M);
                    copy_host_taps(slice);
                }

                std::vector<int> loc_idx, dst_rows;
                for (int k = 0; k < M; ++k) {
                    const int oi = idxs[off + k];
                    if (out_row[oi] >= 0) { loc_idx.push_back(k); dst_rows.push_back(out_row[oi]); }
                }
                emit_selected_logits(loc_idx, dst_rows);
                off += M;
            }
        };

        // Partition tokens by slot count:
        // - Distinct 1-token-per-slot items with no images -> ONE batched CUDA-Graph forward_batched(B) call!
        // - Multi-token or image-bearing slot runs -> run_single_slot_seq(slot, idxs)!
        std::unordered_map<int, std::vector<int>> by_slot;
        std::vector<int> slot_order;
        for (int i = 0; i < N; ++i) {
            int s = batch.slot_ids[i];
            if (by_slot.find(s) == by_slot.end()) slot_order.push_back(s);
            by_slot[s].push_back(i);
        }

        // Fast-path A: Single slot (prefill, M=1 decode, or DSpark verify)
        if (slot_order.size() == 1) {
            run_single_slot_seq(slot_order[0], by_slot[slot_order[0]]);
            INK_CK(cudaStreamSynchronize(runner_->st));
            return true;
        }

        // Fast-path B: Separate 1-token-per-slot decode items (coalesced into 1 forward_batched!) and multi-token prefill runs!
        std::vector<int> dec_orig;
        std::vector<int32_t> dec_toks, dec_slots, dec_pos;
        std::vector<int> multi_slots;
        for (int s : slot_order) {
            const auto& idxs = by_slot[s];
            const bool is_img = (batch.img_embeds && idxs[0] < (int)batch.img_embed_idx.size() && batch.img_embed_idx[idxs[0]] >= 0);
            if (idxs.size() == 1 && !is_img && !snap_active_[s]) {
                dec_orig.push_back(idxs[0]);
                dec_toks.push_back(batch.token_ids[idxs[0]]);
                dec_slots.push_back(s);
                dec_pos.push_back(batch.pos[idxs[0]]);
            } else {
                multi_slots.push_back(s);
            }
        }

        if (!dec_orig.empty()) {
            const int B = (int)dec_orig.size();
            const bool want_taps = (taps_out != nullptr);
            runner_->has_img_step = false;
            runner_->forward_batched(dec_toks.data(), dec_slots.data(), dec_pos.data(), B, want_taps);
            if (taps_out) copy_host_taps(dec_orig);
            std::vector<int> loc_idx, dst_rows;
            for (int k = 0; k < B; ++k) {
                const int oi = dec_orig[k];
                if (out_row[oi] >= 0) { loc_idx.push_back(k); dst_rows.push_back(out_row[oi]); }
            }
            emit_selected_logits(loc_idx, dst_rows);
        }

        for (int s : multi_slots) {
            run_single_slot_seq(s, by_slot[s]);
        }

        INK_CK(cudaStreamSynchronize(runner_->st));
        return true;
    }

    int dspark_draft(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_draft,
        int32_t* out_draft_ids,
        std::string& /*err*/) override {
        if (!initialized_ || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 8 >= dspark_->max_ctx) return 0;
        int32_t ids7[8] = {};
        float conf7[8] = {};
        dspark_->draft_block8(anchor_tok, anchor_pos, ids7, conf7);
        const int cap = std::max(0, std::min(7, max_draft));
        if (cap <= 0) return 0;
        // Sparse-MoE Bayes-Optimal Adaptive Draft Length (128 experts, Top-8):
        // Pick nd in [2..cap] maximizing expected emitted tokens / (1 + 4.2 * (1 - (15/16)^(nd+1)))!
        double best_rate = -1.0;
        int best_nd = std::min(2, cap);
        double cum_p = 1.0;
        double e_tok = 1.0;
        for (int k = 1; k <= cap; ++k) {
            cum_p *= std::max(0.0, std::min(1.0, (double)conf7[k - 1]));
            e_tok += cum_p;
            if (k >= 2 || k == cap) {
                const double exp_frac = 1.0 - std::pow(15.0 / 16.0, (double)(k + 1));
                const double rate = e_tok / (1.0 + 4.2 * exp_frac);
                if (rate > best_rate) { best_rate = rate; best_nd = k; }
            }
        }
        for (int i = 0; i < best_nd; ++i) out_draft_ids[i] = ids7[i];
        return best_nd;
    }

    int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& /*err*/) override {
        if (!initialized_ || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 12 >= dspark_->max_ctx || max_emit < 1 || !out_emitted) return 0;

        // 1. Snapshot slot-0 ShortConv pre-verify state (2.5 us) for exact mid-tree EOS rollback if needed!
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, 0, runner_->max_slots, true);

        // 2. True W=4 Multi-Branch Sparse-MoE Bayes-Optimal Adaptive Beam Tree (M* in {3,4,5,6,8,10,12})!
        strata::ink::InkTreeSpec tr{};
        const int max_nodes = std::min(12, runner_->tree_cap);
        dspark_->draft_beam_tree(anchor_tok, anchor_pos, /*beam_w=*/4, max_nodes, tr, /*adaptive_moe_ratio=*/4.2f);

        // 3. Single CUDA-Graph 48-visit tree verify + exact GPU screened argmax (ZERO 6.4 MB logits D2H!)
        runner_->forward_tree_verify(tr, anchor_pos, 0, true);
        int32_t target_next[strata::ink::kInkMaxTree] = {};
        runner_->head_argmax_exact(tr.M, target_next);

        // 4. Exact greedy tree walk (clamped to min(max_emit, 8))
        const int emit_cap = std::min(max_emit, 8);
        int32_t path[strata::ink::kInkMaxTree] = {};
        int n_emit = 0;
        int cur_node = 0;
        while (n_emit < emit_cap) {
            path[n_emit] = cur_node;
            const int32_t tok_next = target_next[cur_node];
            out_emitted[n_emit++] = tok_next;
            int next_child = -1;
            for (int j = cur_node + 1; j < tr.M; ++j) {
                if (tr.parent[j] == cur_node && tr.tok[j] == tok_next) {
                    next_child = j;
                    break;
                }
            }
            if (next_child >= 0) cur_node = next_child;
            else break;
        }

        // 5. Commit ONLY the accepted tree path across all 48 visits (2 launches) + compact & ingest DSpark taps!
        INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)n_emit * 4, cudaMemcpyHostToDevice, runner_->st));
        strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_emit, 0, runner_->max_slots, runner_->tree_cap);
        strata::ink::k_ink_commit_kv_all48<<<dim3(strata::ink::kInkVisits, strata::ink::kInkKVHeads), 128, 0, runner_->st>>>(
            runner_->d_kv_desc_arr, runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, anchor_pos, n_emit, 0, runner_->tree_cap);
        for (int t = 0; t < 4; ++t) {
            strata::ink::k_ink_compact_taps<<<n_emit, 256, 0, runner_->st>>>(
                runner_->d_taps[t], runner_->d_taps_compact[t], runner_->d_acc_path, n_emit);
        }
        dspark_->ingest_taps_from(runner_->d_taps_compact, anchor_pos, n_emit);

        tree_snap_ready_ = true;
        tree_snap_p0_ = anchor_pos;
        tree_snap_M_ = n_emit;
        for (int i = 0; i < n_emit; ++i) tree_snap_path_[i] = path[i];
        if (out_drafted) *out_drafted = tr.M - 1;
        return n_emit;
    }

    bool selfcheck(std::string& err) override {
        cudaDeviceProp prop{};
        cudaError_t ce = cudaGetDeviceProperties(&prop, gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("CUDA device query failed: ") + cudaGetErrorString(ce);
            return false;
        }
        // Verify SITU + softcap + basic device execution
        float* d_tmp = strata::ink::dalloc<float>(64);
        std::vector<float> h_in(64), h_out(64);
        for (int i = 0; i < 64; ++i) h_in[i] = (float)(i - 32) * 1.5f;
        INK_CK(cudaMemcpy(d_tmp, h_in.data(), 64 * 4, cudaMemcpyHostToDevice));
        strata::ink::k_ink_softcap<<<1, 256>>>(d_tmp, 64, 30.0f);
        INK_CK(cudaMemcpy(h_out.data(), d_tmp, 64 * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaFree(d_tmp));
        for (int i = 0; i < 64; ++i) {
            float ref = 30.0f * std::tanh(h_in[i] / 30.0f);
            if (std::fabs(h_out[i] - ref) > 1e-4f) {
                err = "CUDA softcap selfcheck mismatch";
                return false;
            }
        }
        return true;
    }

private:
    void ensure_pinned_logits(int nw, int V) {
        if (nw <= h_logits_cap_ && h_logits_pin_) return;
        if (h_logits_pin_) { cudaFreeHost(h_logits_pin_); h_logits_pin_ = nullptr; }
        h_logits_cap_ = std::max(nw, 8);
        h_logits_pin_ = strata::ink::halloc_pinned<float>((size_t)h_logits_cap_ * V);
    }

    void cleanup() {
        if (dspark_) { dspark_->free_all(); dspark_.reset(); }
        if (runner_) { runner_->free_all(); runner_.reset(); }
        if (d_sconv_snap_) { cudaFree(d_sconv_snap_); d_sconv_snap_ = nullptr; }
        if (d_h_sel_)      { cudaFree(d_h_sel_); d_h_sel_ = nullptr; }
        if (d_sel_idx_)    { cudaFree(d_sel_idx_); d_sel_idx_ = nullptr; }
        if (h_logits_pin_) { cudaFreeHost(h_logits_pin_); h_logits_pin_ = nullptr; h_logits_cap_ = 0; }
        model_.free_all();
        initialized_ = false;
    }

    bool initialized_ = false;
    int gpu_id_ = 0;
    int max_slots_ = 1;
    int eff_max_ctx_ = 4096;
    int dspark_slot_ = -1;
    KvDtype kv_dtype_ = KvDtype::FP16;
    std::string dev_info_;

    strata::ink::InkCudaModel model_;
    std::unique_ptr<strata::ink::InkCudaRunner> runner_;
    std::unique_ptr<strata::ink::InkDSpark> dspark_;
    float* d_sconv_snap_ = nullptr;
    float* d_h_sel_ = nullptr;
    int32_t* d_sel_idx_ = nullptr;
    float* h_logits_pin_ = nullptr;
    int h_logits_cap_ = 0;
    std::vector<bool> slot_alloc_;
    std::vector<bool> snap_active_;
    std::vector<int> snap_p0_;
    std::vector<int> snap_M_;
    bool tree_snap_ready_ = false;
    int tree_snap_p0_ = 0;
    int tree_snap_M_ = 0;
    int32_t tree_snap_path_[strata::ink::kInkMaxTree] = {};
};

std::unique_ptr<InkBackend> make_cuda_backend() {
    return std::make_unique<InkCudaBackend>();
}

}  // namespace ink2
