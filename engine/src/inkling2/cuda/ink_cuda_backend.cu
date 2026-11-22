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
#include "strata/core/volta_forever.hpp"

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

    const char* backend_name() const noexcept override {
        return model_.ls_active ? "cuda-ls" : (model_.ep2_active ? "cuda-ep2" : "cuda");
    }

    int effective_max_ctx() const override { return initialized_ ? eff_max_ctx_ : 0; }

    bool dspark_ready() const override { return initialized_ && dspark_ && dspark_->ready; }

    std::string device_info() const override { return dev_info_; }

    BackendStats stats() const override {
        BackendStats st{};
        if (initialized_) {
            st.vram_experts_loaded = model_.n_vram_experts + model_.n_peer_experts;
            st.ram_experts_loaded  = model_.n_host_experts;
            st.total_experts       = model_.n_vram_experts + model_.n_peer_experts + model_.n_host_experts;
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
        int n_dev = 0;
        cudaError_t ce = cudaGetDeviceCount(&n_dev);
        if (ce != cudaSuccess || n_dev <= 0) {
            err = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(ce);
            return false;
        }
        // Resolve primary + optional multi-GPU list (up to 4 GPUs!) from opts.gpu_ids / opts.gpu_index / opts.gpus
        gpu_id_ = (opts.gpu_index >= 0 && opts.gpu_index < n_dev) ? opts.gpu_index : 0;
        std::vector<int> all_gpus;
        if (!opts.gpu_ids.empty()) {
            for (int g : opts.gpu_ids) {
                if (g >= 0 && g < n_dev && std::find(all_gpus.begin(), all_gpus.end(), g) == all_gpus.end() && all_gpus.size() < 4) {
                    all_gpus.push_back(g);
                }
            }
            if (!all_gpus.empty()) gpu_id_ = all_gpus[0];
        } else if (opts.gpus >= 2 && n_dev >= 2) {
            const int want = std::min(std::min(opts.gpus, n_dev), 4);
            all_gpus.push_back(gpu_id_);
            for (int g = 0; g < n_dev && (int)all_gpus.size() < want; ++g) {
                if (g != gpu_id_) all_gpus.push_back(g);
            }
        } else {
            all_gpus.push_back(gpu_id_);
        }
        const int peer_gpu_id = (all_gpus.size() >= 2) ? all_gpus[1] : -1;

        ce = cudaSetDevice(gpu_id_);
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
        const int sm_cc = prop.major * 10 + prop.minor;

        // Compute exact minimum KV + ShortConv + workspace headroom BEFORE pinning experts!
        // Keep lossless FP16 KV as default when Host KV Offload handles capacity (#552)!
        max_slots_ = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        const int kv_offload_mode = (opts.kv_offload == "on") ? 1 : ((opts.kv_offload == "off") ? -1 : 0);
        kv_dtype_ = opts.kv_dtype;
        if (opts.kv_dtype_auto) {
            if (max_slots_ >= 16) {
                kv_dtype_ = KvDtype::FP8_E4M3;
            } else if (kv_offload_mode < 0) {
                if (prop.totalGlobalMem <= (13ull << 30)) {
                    kv_dtype_ = KvDtype::FP8_E4M3;
                } else if (sm_cc >= 89 && opts.max_ctx > 16384 && prop.totalGlobalMem < (40ull << 30)) {
                    kv_dtype_ = KvDtype::FP8_E4M3;
                }
            }
        }
        auto map_kvd = [](KvDtype d) -> strata::ink::InkKvDtype {
            if (d == KvDtype::FP8_E4M3) return strata::ink::InkKvDtype::FP8;
            if (d == KvDtype::K8V8)     return strata::ink::InkKvDtype::K8V8;
            if (d == KvDtype::K8V4)     return strata::ink::InkKvDtype::K8V4;
            return strata::ink::InkKvDtype::FP16;
        };
        const auto cuda_kvd = map_kvd(kv_dtype_);

        const int eff_chunk = std::max(64, opts.chunk);
        const size_t swa_r = (max_slots_ >= 16) ? (size_t)(strata::ink::kInkSwaWin + 32)
                                                : (size_t)(strata::ink::kInkSwaWin + std::max(eff_chunk, 64));
        const size_t per_visit_pos = strata::ink::InkCudaRunner::kv_bytes_per_visit_pos(cuda_kvd);
        const int init_glob = (kv_offload_mode == 1)
            ? 0
            : ((kv_offload_mode == 0 && prop.totalGlobalMem <= (13ull << 30))
                ? std::min(std::max(256, opts.max_ctx), 2048)
                : std::max(256, std::min(std::max(256, opts.max_ctx), (max_slots_ == 1) ? 8192 : 4096)));
        const size_t min_kv_bytes = (size_t)40 * max_slots_ * swa_r * per_visit_pos
                                  + (size_t)strata::ink::kInkVisits * max_slots_ * 3 * strata::ink::kInkSconvCh * 4
                                  + (size_t)8 * max_slots_ * init_glob * per_visit_pos;
        // Include O(1) prefill ws (~114 MiB) + compact DSpark (~160 MiB) + activation ws (WDDM floor added inside loader!)
        const size_t ws_pad = (max_slots_ >= 8) ? (1280ull << 20)
                            : ((prop.totalGlobalMem >= (20ull << 30)) ? (960ull << 20)
                             : (prop.totalGlobalMem <= (13ull << 30)) ? (512ull << 20)
                                                                      : (640ull << 20));
        size_t min_reserve_bytes = min_kv_bytes + ws_pad;
        if (opts.vram_reserve_mib >= 0) {
            min_reserve_bytes = std::max(min_reserve_bytes, (size_t)opts.vram_reserve_mib << 20);
        }

        // Load .strata v1.1 / .gguf (Unsloth + ggml-org) / HF safetensors trunk + experts (Single-GPU, EP2, Layer-Split, or 4-tier GPU+RAM+SSD!)
        const bool build_screen = opts.enable_dspark && mf.m.dspark.present;
        if (!model_.load_from_model_file(mf, gpu_id_, peer_gpu_id, (float)opts.ep_split, build_screen, err,
                                         min_reserve_bytes, gguf_detail::gguf_dequant_row_f32, gguf_detail::gguf_row_bytes,
                                         opts.vram_experts_gb, opts.ram_experts_gb, opts.ssd_experts,
                                         all_gpus, opts.layer_split, opts.threads)) {
            return false;
        }

        {
            char buf[512];
            if (model_.ls_active) {
                std::string s = "LayerSplit[";
                for (int i = 0; i < model_.n_stages; ++i) {
                    cudaDeviceProp sp{};
                    cudaGetDeviceProperties(&sp, model_.stage_gpu[i]);
                    if (i > 0) s += " + ";
                    char part[128];
                    std::snprintf(part, sizeof(part), "GPU%d:%.48s(%.0fG,L%d-%d)",
                                  model_.stage_gpu[i], sp.name, (double)sp.totalGlobalMem / 1073741824.0,
                                  model_.stage_l0[i], model_.stage_l1[i] - 1);
                    s += part;
                }
                s += "]";
                dev_info_ = s;
            } else if (peer_gpu_id >= 0 && model_.ep2_active) {
                cudaDeviceProp prop1{};
                cudaGetDeviceProperties(&prop1, peer_gpu_id);
                std::snprintf(buf, sizeof(buf), "GPU%d:%.120s + GPU%d:%.120s (sm_%d%d, %.1f+%.1f GiB VRAM, ep2=1)",
                              gpu_id_, prop.name, peer_gpu_id, prop1.name, prop.major, prop.minor,
                              (double)prop.totalGlobalMem / 1073741824.0, (double)prop1.totalGlobalMem / 1073741824.0);
                dev_info_ = buf;
            } else {
                std::snprintf(buf, sizeof(buf), "%.200s (sm_%d%d, %d SMs, %.1f GiB VRAM, sm120_tc=%d)",
                              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                              (double)prop.totalGlobalMem / 1073741824.0, (int)strata::ink::sm120::available());
                dev_info_ = buf;
            }
        }

        runner_ = std::make_unique<strata::ink::InkCudaRunner>(model_);
        const bool vf_graph_ok = strata::core::allow_1024_thread_cuda_graphs(
            prop.major, prop.minor, strata::core::volta_forever_active());
        runner_->use_cuda_graph = vf_graph_ok && !model_.ep2_active && !model_.ls_active;
        if (!vf_graph_ok) {
            std::fprintf(stderr, "[inkling-cuda] sm_70 under volta-forever: disabling 1024-thread CUDA Graph capture by default\n");
        }
        runner_->init(max_slots_, std::max(256, opts.max_ctx), std::max(64, opts.chunk), cuda_kvd, kv_offload_mode);
        eff_max_ctx_ = runner_->max_ctx;

        if (opts.enable_dspark && model_.man.has_dspark && !model_.ls_active) {
            dspark_ = std::make_unique<strata::ink::InkDSpark>(*runner_);
            dspark_->init(eff_max_ctx_, 512);
        }

        // Allocate snapshot buffer (3.83 MB) + selected hidden gather buffers
        if (!model_.ls_active) {
            d_sconv_snap_ = strata::ink::dalloc<float>((size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh);
        }
        d_h_sel_      = strata::ink::dalloc<float>((size_t)runner_->max_batch * strata::ink::kInkHidden);
        d_sel_idx_    = strata::ink::dalloc<int32_t>(runner_->max_batch);
        slot_alloc_.assign(max_slots_, false);
        snap_active_.assign(max_slots_, false);
        snap_p0_.assign(max_slots_, 0);
        snap_M_.assign(max_slots_, 0);
        prompt_ckpts_.assign(max_slots_, PromptCkpt{});
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
            if (slot < (int)prompt_ckpts_.size()) prompt_ckpts_[slot] = PromptCkpt{};
            if (slot == 0) tree_snap_ready_ = false;
        }
    }

    void reset_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) return;
        runner_->reset_slot(slot);
        snap_active_[slot] = false;
        if (slot < (int)prompt_ckpts_.size()) prompt_ckpts_[slot].valid = false;
        if (slot == 0) { dspark_slot_ = -1; tree_snap_ready_ = false; }
    }

    // #6627: Save/restore slot's ShortConv (3.83 MB) + 40 SWA KV ring windows (<=160 MiB in host RAM) + DSpark SWA KV
    // so multi-turn chats and exact repeats reuse the full prior prompt even if hidden thinking or trailing tokens diverged!
    bool save_prompt_checkpoint(int slot, int seq_len) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return false;
        if (seq_len <= 0 || max_slots_ > 8) return false;
        auto& ck = prompt_ckpts_[slot];
        ck.valid = false;
        ck.seq_len = seq_len;
        // 1. Copy 48-visit ShortConv state via d_sconv_snap_ -> host RAM (3.83 MB)
        const size_t sc_elems = (size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh;
        ck.sconv.resize(sc_elems);
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, true);
        INK_CK(cudaMemcpyAsync(ck.sconv.data(), d_sconv_snap_, sc_elems * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
        // 2. Copy active SWA KV ring window [start_pos .. seq_len-1] for all 40 SWA visits
        const int start_pos = std::max(0, seq_len - strata::ink::kInkSwaWin);
        const int n_pos = seq_len - start_pos;
        ck.swa_start = start_pos;
        ck.swa_len = n_pos;
        ck.swa_k.clear(); ck.swa_v.clear(); ck.swa_ksc.clear(); ck.swa_vsc.clear();
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            const auto& kv = runner_->kv_desc[v];
            if (!kv.is_swa) continue;
            const int cap = kv.cap_pos;
            const int p0 = start_pos % cap;
            const int seg1 = std::min(n_pos, cap - p0);
            const int seg2 = n_pos - seg1;
            const size_t slot_base = (size_t)slot * cap;
            auto copy_d2h = [&](const void* base, size_t elem_b, size_t stride_elems, std::vector<uint8_t>& dst) {
                const size_t off = dst.size();
                dst.resize(off + (size_t)n_pos * stride_elems * elem_b);
                const uint8_t* bp = reinterpret_cast<const uint8_t*>(base);
                INK_CK(cudaMemcpyAsync(dst.data() + off, bp + (slot_base + p0) * stride_elems * elem_b,
                                       (size_t)seg1 * stride_elems * elem_b, cudaMemcpyDeviceToHost, runner_->st));
                if (seg2 > 0) {
                    INK_CK(cudaMemcpyAsync(dst.data() + off + (size_t)seg1 * stride_elems * elem_b,
                                           bp + slot_base * stride_elems * elem_b,
                                           (size_t)seg2 * stride_elems * elem_b, cudaMemcpyDeviceToHost, runner_->st));
                }
            };
            if (kv.kv_fmt == 0) {
                copy_d2h(kv.k_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_k);
                copy_d2h(kv.v_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_v);
            } else {
                const size_t v_stride = (kv.kv_fmt == 3) ? (strata::ink::kInkKVDim / 2) : strata::ink::kInkKVDim;
                copy_d2h(kv.k_f8, 1, strata::ink::kInkKVDim, ck.swa_k);
                copy_d2h(kv.v_f8, 1, v_stride, ck.swa_v);
                copy_d2h(kv.k_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_ksc);
                copy_d2h(kv.v_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_vsc);
            }
        }
        // 3. Save DSpark SWA-1025 context KV (layers 0 & 2) + f_rnn row at seq_len-1 for slot 0
        ck.has_dspark = false;
        if (slot == 0 && dspark_ && dspark_->ready && dspark_slot_ == 0) {
            ck.dspark_k.clear(); ck.dspark_v.clear();
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds = [&](const __half* src, std::vector<uint8_t>& dst) {
                    const size_t row_b = (size_t)strata::ink::kInkKVDim * sizeof(__half);
                    const size_t off = dst.size();
                    dst.resize(off + (size_t)np * row_b);
                    const uint8_t* bp = reinterpret_cast<const uint8_t*>(src);
                    INK_CK(cudaMemcpyAsync(dst.data() + off, bp + (size_t)p0 * row_b, (size_t)seg1 * row_b, cudaMemcpyDeviceToHost, runner_->st));
                    if (seg2 > 0) {
                        INK_CK(cudaMemcpyAsync(dst.data() + off + (size_t)seg1 * row_b, bp, (size_t)seg2 * row_b, cudaMemcpyDeviceToHost, runner_->st));
                    }
                };
                copy_ds(dspark_->d_ctx_k[il], ck.dspark_k);
                copy_ds(dspark_->d_ctx_v[il], ck.dspark_v);
            }
            ck.dspark_frnn.resize(strata::ink::kInkHidden);
            const int fr_slot = (seq_len - 1) % dspark_->frnn_cap;
            INK_CK(cudaMemcpyAsync(ck.dspark_frnn.data(), dspark_->d_f_rnn + (size_t)fr_slot * strata::ink::kInkHidden,
                                   (size_t)strata::ink::kInkHidden * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
            ck.has_dspark = true;
        }
        INK_CK(cudaStreamSynchronize(runner_->st));
        ck.valid = true;
        return true;
    }

    bool restore_prompt_checkpoint(int slot, int seq_len) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return false;
        if (slot >= (int)prompt_ckpts_.size()) return false;
        const auto& ck = prompt_ckpts_[slot];
        if (!ck.valid || ck.seq_len != seq_len) return false;
        snap_active_[slot] = false;
        if (slot == 0) tree_snap_ready_ = false;
        // 1. Restore ShortConv state
        const size_t sc_elems = (size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh;
        INK_CK(cudaMemcpyAsync(d_sconv_snap_, ck.sconv.data(), sc_elems * sizeof(float), cudaMemcpyHostToDevice, runner_->st));
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, false);
        // 2. Restore 40 SWA KV ring windows (Global KV is append-only and already intact at [0..seq_len-1]!)
        size_t ok = 0, ov = 0, oks = 0, ovs = 0;
        const int start_pos = ck.swa_start;
        const int n_pos = ck.swa_len;
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            const auto& kv = runner_->kv_desc[v];
            if (!kv.is_swa) continue;
            const int cap = kv.cap_pos;
            const int p0 = start_pos % cap;
            const int seg1 = std::min(n_pos, cap - p0);
            const int seg2 = n_pos - seg1;
            const size_t slot_base = (size_t)slot * cap;
            auto copy_h2d = [&](void* base, size_t elem_b, size_t stride_elems, const std::vector<uint8_t>& src, size_t& off) {
                uint8_t* bp = reinterpret_cast<uint8_t*>(base);
                INK_CK(cudaMemcpyAsync(bp + (slot_base + p0) * stride_elems * elem_b, src.data() + off,
                                       (size_t)seg1 * stride_elems * elem_b, cudaMemcpyHostToDevice, runner_->st));
                if (seg2 > 0) {
                    INK_CK(cudaMemcpyAsync(bp + slot_base * stride_elems * elem_b,
                                           src.data() + off + (size_t)seg1 * stride_elems * elem_b,
                                           (size_t)seg2 * stride_elems * elem_b, cudaMemcpyHostToDevice, runner_->st));
                }
                off += (size_t)n_pos * stride_elems * elem_b;
            };
            if (kv.kv_fmt == 0) {
                copy_h2d(kv.k_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_k, ok);
                copy_h2d(kv.v_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_v, ov);
            } else {
                const size_t v_stride = (kv.kv_fmt == 3) ? (strata::ink::kInkKVDim / 2) : strata::ink::kInkKVDim;
                copy_h2d(kv.k_f8, 1, strata::ink::kInkKVDim, ck.swa_k, ok);
                copy_h2d(kv.v_f8, 1, v_stride, ck.swa_v, ov);
                copy_h2d(kv.k_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_ksc, oks);
                copy_h2d(kv.v_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_vsc, ovs);
            }
        }
        // 3. Restore DSpark SWA-1025 context KV + f_rnn row for slot 0
        if (slot == 0 && ck.has_dspark && dspark_ && dspark_->ready) {
            size_t dk_o = 0, dv_o = 0;
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds_h2d = [&](__half* dst, const std::vector<uint8_t>& src, size_t& off) {
                    const size_t row_b = (size_t)strata::ink::kInkKVDim * sizeof(__half);
                    uint8_t* bp = reinterpret_cast<uint8_t*>(dst);
                    INK_CK(cudaMemcpyAsync(bp + (size_t)p0 * row_b, src.data() + off, (size_t)seg1 * row_b, cudaMemcpyHostToDevice, runner_->st));
                    if (seg2 > 0) {
                        INK_CK(cudaMemcpyAsync(bp, src.data() + off + (size_t)seg1 * row_b, (size_t)seg2 * row_b, cudaMemcpyHostToDevice, runner_->st));
                    }
                    off += (size_t)np * row_b;
                };
                copy_ds_h2d(dspark_->d_ctx_k[il], ck.dspark_k, dk_o);
                copy_ds_h2d(dspark_->d_ctx_v[il], ck.dspark_v, dv_o);
            }
            const int fr_slot = (seq_len - 1) % dspark_->frnn_cap;
            INK_CK(cudaMemcpyAsync(dspark_->d_f_rnn + (size_t)fr_slot * strata::ink::kInkHidden, ck.dspark_frnn.data(),
                                   (size_t)strata::ink::kInkHidden * sizeof(float), cudaMemcpyHostToDevice, runner_->st));
            dspark_slot_ = 0;
        }
        INK_CK(cudaStreamSynchronize(runner_->st));
        return true;
    }

    void snapshot_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return;
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, true);
        snap_active_[slot] = true;
        snap_M_[slot] = 0;
        if (slot == 0) tree_snap_ready_ = false;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return;
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
        // Defensive KV position bounds check + elastic global KV growth (fixes #6590B CUDA error 700 & CA1JEeAx!)
        int max_pos_req = 0;
        for (int i = 0; i < N; ++i) {
            if (batch.pos[i] < 0 || batch.pos[i] >= eff_max_ctx_) {
                err = "CUDA KV position " + std::to_string(batch.pos[i]) + " out of bounds [0, " + std::to_string(eff_max_ctx_) + ")";
                return false;
            }
            if (batch.pos[i] > max_pos_req) max_pos_req = batch.pos[i];
        }
        if (!runner_->ensure_global_kv_cap(max_pos_req + 1)) {
            err = "VRAM exhausted growing elastic global KV cache to " + std::to_string(max_pos_req + 1) + " tokens";
            return false;
        }
        if (batch.capture_taps && dspark_ && dspark_->ready) {
            if (!dspark_->ensure_ctx_cap(max_pos_req + 1)) {
                dspark_->ready = false;
            }
        }
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
        if (!initialized_ || model_.ls_active || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 8 >= eff_max_ctx_ || anchor_pos + 8 >= dspark_->max_ctx) return 0;
        if (!runner_->ensure_global_kv_cap(anchor_pos + 8) || !dspark_->ensure_ctx_cap(anchor_pos + 8)) return 0;
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
        if (!initialized_ || model_.ls_active || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 12 >= eff_max_ctx_ || anchor_pos + 12 >= dspark_->max_ctx || max_emit < 1 || !out_emitted) return 0;
        if (!runner_->ensure_global_kv_cap(anchor_pos + 12) || !dspark_->ensure_ctx_cap(anchor_pos + 12)) return 0;

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
        prompt_ckpts_.clear();
        initialized_ = false;
    }

    struct PromptCkpt {
        bool valid = false;
        bool has_dspark = false;
        int seq_len = 0;
        int swa_start = 0;
        int swa_len = 0;
        std::vector<float> sconv;
        std::vector<uint8_t> swa_k, swa_v, swa_ksc, swa_vsc;
        std::vector<uint8_t> dspark_k, dspark_v;
        std::vector<float> dspark_frnn;
    };

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
    std::vector<PromptCkpt> prompt_ckpts_;
    bool tree_snap_ready_ = false;
    int tree_snap_p0_ = 0;
    int tree_snap_M_ = 0;
    int32_t tree_snap_path_[strata::ink::kInkMaxTree] = {};
};

std::unique_ptr<InkBackend> make_cuda_backend() {
    return std::make_unique<InkCudaBackend>();
}

}  // namespace ink2
