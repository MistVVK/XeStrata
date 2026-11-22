// engine/src/inkling2/ink_cpu_backend.cpp - Exact Multi-Slot (B=1..64) CPU Backend for strata-inkling
// Implements 24->48 looped visits, AttnRes, 4x Rank-64 controllers, 4-tap static+dynamic ShortConv,
// Paged SWA-1024 + Global KV (FP16/FP8), coalesced multi-slot MoE (SF4/MX8/paired-2-row S_K + SITU),
// and embedded DSpark Block-8 Recurrent Markov drafter.
// MIT License - Copyright (c) 2026 Strata contributors
#define INKLING_DEFINE_STBI 1
#include "ink_backend.h"
#include "ink_vision.h"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ink2 {
namespace {

using namespace strata::inkling2;
using namespace strata::quant::sk;

// Decode any Section 1 trunk/DSpark tensor format (F32, BF16, F16, Q8_ROW, S2_K..S8_K) into FP32
std::vector<float> decode_section1_tensor(const uint8_t* sec1, const TInfo& ti) {
    const size_t n = static_cast<size_t>(ti.numel());
    std::vector<float> out(n, 0.0f);
    const uint8_t* src = sec1 + ti.off;
    if (ti.fmt == F_F32) {
        std::memcpy(out.data(), src, n * sizeof(float));
    } else if (ti.fmt == F_BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
    } else if (ti.fmt == F_F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(p[i]);
    } else if (ti.fmt == F_Q8ROW) {
        const size_t rows = static_cast<size_t>(ti.rows());
        const size_t cols = static_cast<size_t>(ti.cols());
        // Q8_ROW layout: [rows] float32 scales followed by [rows, cols] int8_t weights
        const float* scales = reinterpret_cast<const float*>(src);
        const int8_t* qs = reinterpret_cast<const int8_t*>(src + rows * sizeof(float));
        for (size_t r = 0; r < rows; ++r) {
            const float sc = scales[r];
            for (size_t c = 0; c < cols; ++c) out[r * cols + c] = static_cast<float>(qs[r * cols + c]) * sc;
        }
    } else {
        // S_K super-block format (e.g., S6_K unembed or S5_K embed)
        dequant_sk_row(src, static_cast<SkCode>(ti.fmt), out.data(), n);
    }
    return out;
}

inline float dot_f32_fast(const float* a, const float* b, int n) noexcept {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

void matmul_f32(const float* x, const float* w, float* y, int M, int N, int K, const float* bias = nullptr) {
    #pragma omp parallel for schedule(static) if (M * N >= 4096)
    for (int m = 0; m < M; ++m) {
        const float* xm = x + static_cast<size_t>(m) * K;
        float* ym = y + static_cast<size_t>(m) * N;
        for (int n = 0; n < N; ++n) {
            ym[n] = dot_f32_fast(xm, w + static_cast<size_t>(n) * K, K) + (bias ? bias[n] : 0.0f);
        }
    }
}

void apply_half_rope_vec(float* x, int n_heads, int head_dim, int pos, float theta) noexcept {
    const int half = head_dim / 2;
    for (int j = 0; j < half; ++j) {
        const float freq = 1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        const float c = std::cos(ang), s = std::sin(ang);
        for (int h = 0; h < n_heads; ++h) {
            float* xh = x + h * head_dim;
            float a = xh[j], b = xh[j + half];
            xh[j]        = a * c - b * s;
            xh[j + half] = a * s + b * c;
        }
    }
}

struct CpuSlot {
    bool allocated = false;
    int max_ctx = 0;
    SlotKvPages kv;
    SlotShortConvState sconv;
    std::vector<SlotShortConvState> sconv_snaps;  // [0..8] exact post-step ShortConv snapshots for DSpark partial-accept rollback!
    int snap_seq_len = -1;
    bool recording_snaps = false;

    // DSpark per-slot context KV (3 layers) + f_rnn ring [pos, D]
    std::vector<std::vector<uint16_t>> dspark_k;  // [3][max_ctx * kv_dim] BF16 (compact on demand)
    std::vector<std::vector<uint16_t>> dspark_v;
    std::vector<uint16_t> dspark_frnn;            // [pos * hidden] BF16
    int dspark_len = 0;
};

class CpuBackend final : public InkBackend {
public:
    const char* backend_name() const noexcept override { return "cpu"; }

    std::string device_info() const override {
        int nth = 1;
#ifdef _OPENMP
        nth = omp_get_max_threads();
#endif
        return std::string("CPU (") + active_sk_isa() + ", " + std::to_string(nth) + " threads)";
    }

    BackendStats stats() const override {
        BackendStats st;
        st.ram_experts_loaded = expert_bodies_.size();
        st.total_experts = model_ ? model_->expert_directory().size() : 0;
        st.max_slots = static_cast<int>(slots_.size());
        for (const auto& s : slots_) if (s.allocated) ++st.active_slots;
        st.kv_dtype = default_kv_dtype_;
        return st;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        model_ = &model;
        opts_ = opts;
        c_ = model.m.c;
        dspark_cfg_ = model.m.dspark;
        default_kv_dtype_ = opts.kv_dtype;

#ifdef _OPENMP
        if (opts.threads > 0) omp_set_num_threads(opts.threads);
#endif

        // 1. Load Section 1 trunk + DSpark arena (works identically for .strata, official safetensors, and .gguf!)
        std::vector<uint8_t> sec1;
        if (!model.read_section1(sec1, err)) return false;

        // Decode trunk tensors (keep embed/unembed in compact raw/BF16 if huge? On CPU, keep BF16 for embed/unembed & F32 for layers)
        const std::string embed_k = "model.llm.embed.weight";
        const std::string unemb_k = "model.llm.unembed.weight";

        for (const auto& kv : model.m.trunk) {
            const auto& ti = kv.second;
            if (kv.first == embed_k || kv.first == unemb_k) {
                // Keep embed/unembed in their packed Section 1 format to save RAM, decode rows on the fly!
                packed_trunk_[kv.first] = {ti, std::vector<uint8_t>(sec1.begin() + ti.off, sec1.begin() + ti.off + ti.bytes)};
            } else {
                trunk_f32_[kv.first] = decode_section1_tensor(sec1.data(), ti);
            }
        }

        // Precompute stacked cross-pass msg_queries [4, D] per core layer
        const auto& q_cp = trunk_f32_.at("model.llm.middle_cycle_cross_pass_queries");
        const auto& kv_cp = trunk_f32_.at("model.llm.middle_cycle_cross_pass_kv_queries");
        const int core_len = c_.core_end - c_.core_start;
        msg_queries_.resize(c_.n_phys);
        for (int i = 0; i < core_len; ++i) {
            auto& mq = msg_queries_[c_.core_start + i];
            mq.resize(4u * c_.hidden);
            std::memcpy(mq.data() + 0 * c_.hidden, q_cp.data()  + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
            std::memcpy(mq.data() + 1 * c_.hidden, kv_cp.data() + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
            std::memcpy(mq.data() + 2 * c_.hidden, kv_cp.data() + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
            std::memcpy(mq.data() + 3 * c_.hidden, q_cp.data()  + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
        }

        // Pre-normalize static ShortConv weights to [C, 4] (lag 3..0)
        for (int il = 0; il < c_.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            static_sconv_[p + ".attn_sconv"] = trunk_f32_.at(p + ".attn_sconv.weight");
            static_sconv_[p + ".mlp_sconv"]  = trunk_f32_.at(p + ".mlp_sconv.weight");
            if (c_.is_swa[il]) {
                static_sconv_[p + ".k_sconv"] = trunk_f32_.at(p + ".attn.k_sconv.weight");
                static_sconv_[p + ".v_sconv"] = trunk_f32_.at(p + ".attn.v_sconv.weight");
            } else {
                // Global static_w is [4, kv_dim] lag 0..3 -> transpose & flip to [kv_dim, 4] lag 3..0
                const int kv_dim = c_.n_kv * c_.head_dim;
                const auto& kw = trunk_f32_.at(p + ".attn.k_sconv.static_w");
                const auto& vw = trunk_f32_.at(p + ".attn.v_sconv.static_w");
                std::vector<float> kt(kv_dim * 4), vt(kv_dim * 4);
                for (int c = 0; c < kv_dim; ++c) {
                    for (int lag = 0; lag < 4; ++lag) {
                        kt[c * 4 + (3 - lag)] = kw[lag * kv_dim + c];
                        vt[c * 4 + (3 - lag)] = vw[lag * kv_dim + c];
                    }
                }
                static_sconv_[p + ".k_sconv"] = std::move(kt);
                static_sconv_[p + ".v_sconv"] = std::move(vt);
            }
        }

        // 2. Decode embedded DSpark tensors if present and enabled
        has_dspark_ = opts.enable_dspark && dspark_cfg_.present;
        if (has_dspark_) {
            for (const auto& kv : model.m.dspark_tensors) {
                std::string k = kv.first;
                if (k.rfind("dspark.", 0) == 0) k = k.substr(7);
                dspark_f32_[k] = decode_section1_tensor(sec1.data(), kv.second);
            }
        }

        // 3. Load Section 4 experts into RAM in 3x loop-weighted vram_load_order priority (respecting --ram-experts-gb)
        const auto& dir = model.expert_directory();
        expert_bodies_.resize(dir.size());
        const size_t ram_cap_bytes = (opts.ram_experts_gb > 0.0)
            ? static_cast<size_t>(opts.ram_experts_gb * 1073741824.0)
            : static_cast<size_t>(-1);
        size_t loaded_bytes = 0;
        for (uint16_t idx : model.vram_load_order) {
            if (idx >= dir.size()) continue;
            if (loaded_bytes + dir[idx].body_bytes > ram_cap_bytes) continue;
            if (!model.read_expert_body(idx, expert_bodies_[idx], err)) return false;
            loaded_bytes += dir[idx].body_bytes;
        }

        // 4. Initialize slot pool (1..64)
        const int n_slots = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        slots_.resize(n_slots);
        return true;
    }

    bool alloc_slot(int slot, int max_ctx, KvDtype kv_dtype, std::string& err) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) {
            err = "slot index out of range";
            return false;
        }
        auto& sl = slots_[slot];
        sl.allocated = true;
        sl.max_ctx = max_ctx;
        sl.kv.init(c_.n_swa_visits, c_.n_global_visits, c_.n_kv * c_.head_dim, c_.swa_win, kv_dtype);
        sl.sconv.init(c_.n_visits, c_.n_kv * c_.head_dim, c_.hidden);
        sl.sconv_snaps.clear();
        sl.snap_seq_len = -1;
        sl.recording_snaps = false;
        sl.dspark_k.assign(dspark_cfg_.n_layers, {});
        sl.dspark_v.assign(dspark_cfg_.n_layers, {});
        sl.dspark_frnn.clear();
        sl.dspark_len = 0;
        return true;
    }

    void free_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) return;
        slots_[slot] = CpuSlot{};
    }

    void reset_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        slots_[slot].kv.reset();
        slots_[slot].sconv.reset();
        slots_[slot].recording_snaps = false;
        slots_[slot].snap_seq_len = -1;
        slots_[slot].dspark_len = 0;
    }

    void snapshot_slot(int slot) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        if (sl.sconv_snaps.empty()) {
            sl.sconv_snaps.resize(9);
            for (auto& sn : sl.sconv_snaps) sn.init(c_.n_visits, c_.n_kv * c_.head_dim, c_.hidden);
        }
        sl.sconv_snaps[0] = sl.sconv;
        sl.snap_seq_len = sl.kv.seq_len;
        sl.recording_snaps = true;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        sl.kv.seq_len = keep_len;
        sl.dspark_len = std::min(sl.dspark_len, keep_len);
        if (sl.snap_seq_len >= 0 && keep_len >= sl.snap_seq_len) {
            const int step = keep_len - sl.snap_seq_len;
            if (step >= 0 && step < static_cast<int>(sl.sconv_snaps.size())) {
                sl.sconv = sl.sconv_snaps[step];
            }
        }
        sl.recording_snaps = false;
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        const int B = batch.n_tokens;
        if (B <= 0) return true;
        const int D = c_.hidden;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const int q_dim = c_.n_head * c_.head_dim;

        // Validate slots
        for (int b = 0; b < B; ++b) {
            int s = batch.slot_ids[b];
            if (s < 0 || s >= static_cast<int>(slots_.size()) || !slots_[s].allocated) {
                err = "unallocated slot " + std::to_string(s);
                return false;
            }
        }

        // 1. Embeddings (image tokens bypass embed_norm!)
        std::vector<float> cur(static_cast<size_t>(B) * D);
        const float* emb_nw = trunk_f32_.at("model.llm.embed_norm.weight").data();
        const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
        std::vector<float> raw_row(D);
        for (int b = 0; b < B; ++b) {
            if (batch.img_embed_idx[b] >= 0 && batch.img_embeds != nullptr) {
                std::memcpy(cur.data() + static_cast<size_t>(b) * D,
                            batch.img_embeds + static_cast<size_t>(batch.img_embed_idx[b]) * D,
                            D * sizeof(float));
            } else {
                decode_packed_row(emb_pk, batch.token_ids[b], raw_row.data());
                rms_norm_f32(raw_row.data(), emb_nw, cur.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
            }
        }

        // Per-token log-position scaling tau
        std::vector<float> tau(B);
        for (int b = 0; b < B; ++b) tau[b] = log_pos_tau(batch.pos[b], c_.log_scale_floor, c_.log_scale_alpha);

        // Attention-Residual block bank (shared depth schedule across all B tokens!)
        std::vector<std::vector<float>> blocks;
        std::vector<float> prelude;
        std::vector<std::vector<float>> prelude_blocks;
        std::vector<std::vector<float>> bank;
        std::vector<float> bank_out;

        // DSpark tap buffers [4][B * D] for target_visits [5, 23, 41, 48]
        const bool need_taps = (taps_out != nullptr) || (has_dspark_ && batch.capture_taps);
        std::vector<float> taps[4];
        if (need_taps) {
            for (int k = 0; k < 4; ++k) taps[k].resize(static_cast<size_t>(B) * D);
        }
        auto maybe_tap = [&](int visit_id, const std::vector<float>& vec) {
            if (!need_taps) return;
            for (int k = 0; k < 4 && k < static_cast<int>(dspark_cfg_.target_visits.size()); ++k) {
                if (dspark_cfg_.target_visits[k] == visit_id) {
                    std::memcpy(taps[k].data(), vec.data(), static_cast<size_t>(B) * D * sizeof(float));
                }
            }
        };

        // 2. Run all 48 virtual visits (L0..5 x1, L6..17 x3 passes, L18..23 x1)
        int visit = 0;
        auto run_visit = [&](int ip, int pass_idx, const std::vector<float>* msgs) -> bool {
            maybe_tap(visit, cur);
            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = c_.is_swa[ip] != 0;
            const int kv_slot = c_.visit_kv_slot[visit];

            std::vector<float> residual = cur;
            std::vector<float> in0(B * D), in1(B * D), in2(B * D);

            // k1: Attention Residuals depth mix + optional Rank-64 loop controller + input RMSNorm
            if (ip > 0) {
                const auto& q_res = trunk_f32_.at("model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
                const int nq = is_swa ? 1 : 3;
                #pragma omp parallel for schedule(static) if (B > 1)
                for (int b = 0; b < B; ++b) {
                    std::vector<const float*> blk_ptrs(blocks.size());
                    for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
                    std::vector<float> mx(nq * D);
                    attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, q_res.data(), nq, D, mx.data());
                    if (is_swa) {
                        std::memcpy(in0.data() + static_cast<size_t>(b) * D, mx.data(), D * sizeof(float));
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, mx.data(), D * sizeof(float));
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, mx.data(), D * sizeof(float));
                    } else {
                        std::memcpy(in0.data() + static_cast<size_t>(b) * D, mx.data() + 0 * D, D * sizeof(float));
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, mx.data() + 1 * D, D * sizeof(float));
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, mx.data() + 2 * D, D * sizeof(float));
                    }
                }
            } else {
                in0 = cur; in1 = cur; in2 = cur;
            }

            if (msgs != nullptr) {
                const char* cnames[3] = {"attn", "k", "v"};
                std::vector<float>* ins[3] = {&in0, &in1, &in2};
                for (int j = 0; j < 3; ++j) {
                    const std::string cp = p + ".controllers." + cnames[j];
                    const float* wd = trunk_f32_.at(cp + ".down_weight").data();
                    const float* wg = trunk_f32_.at(cp + ".gain_weight").data();
                    const float* wf = trunk_f32_.at(cp + ".forget_weight").data();
                    #pragma omp parallel for schedule(static) if (B > 1)
                    for (int b = 0; b < B; ++b) {
                        std::vector<float> out_c(D);
                        loop_control_f32(ins[j]->data() + static_cast<size_t>(b) * D,
                                         msgs->data() + (static_cast<size_t>(b) * 4 + j) * D,
                                         wd, wg, wf, D, c_.gate_rank, out_c.data());
                        std::memcpy(ins[j]->data() + static_cast<size_t>(b) * D, out_c.data(), D * sizeof(float));
                    }
                }
            }

            if (is_swa) {
                const float* an = trunk_f32_.at(p + ".attn_norm.weight").data();
                for (int b = 0; b < B; ++b) {
                    rms_norm_f32(in0.data() + static_cast<size_t>(b) * D, an, in0.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    if (msgs != nullptr) {
                        rms_norm_f32(in1.data() + static_cast<size_t>(b) * D, an, in1.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                        rms_norm_f32(in2.data() + static_cast<size_t>(b) * D, an, in2.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    } else {
                        std::memcpy(in1.data() + static_cast<size_t>(b) * D, in0.data() + static_cast<size_t>(b) * D, D * sizeof(float));
                        std::memcpy(in2.data() + static_cast<size_t>(b) * D, in0.data() + static_cast<size_t>(b) * D, D * sizeof(float));
                    }
                }
            } else {
                const float* qn = trunk_f32_.at(p + ".q_input_norm.weight").data();
                const float* kn = trunk_f32_.at(p + ".k_input_norm.weight").data();
                const float* vn = trunk_f32_.at(p + ".v_input_norm.weight").data();
                for (int b = 0; b < B; ++b) {
                    rms_norm_f32(in0.data() + static_cast<size_t>(b) * D, qn, in0.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    rms_norm_f32(in1.data() + static_cast<size_t>(b) * D, kn, in1.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                    rms_norm_f32(in2.data() + static_cast<size_t>(b) * D, vn, in2.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
                }
            }

            // Batched Q, K, V, R, Sink, Gate projections across all B active tokens!
            std::vector<float> q(B * q_dim), k_raw(B * kv_dim), v_raw(B * kv_dim), r_raw(B * c_.n_head * c_.d_rel);
            std::vector<float> sinks(B * c_.n_head), gate_a(B * q_dim);
            matmul_f32(in0.data(), trunk_f32_.at(p + ".attn.wq_du.weight").data(), q.data(), B, q_dim, D);
            matmul_f32(in1.data(), trunk_f32_.at(p + ".attn.wk_dv.weight").data(), k_raw.data(), B, kv_dim, D);
            matmul_f32(in2.data(), trunk_f32_.at(p + ".attn.wv_dv.weight").data(), v_raw.data(), B, kv_dim, D);
            matmul_f32(in0.data(), trunk_f32_.at(p + ".attn.wr_du.weight").data(), r_raw.data(), B, c_.n_head * c_.d_rel, D);
            matmul_f32(in0.data(), trunk_f32_.at(p + ".attn.attn_sink_proj.weight").data(), sinks.data(), B, c_.n_head, D);
            matmul_f32(in0.data(), trunk_f32_.at(p + ".attn.gate_attn.weight").data(), gate_a.data(), B, q_dim, D);

            std::vector<float> k_dyn, v_dyn;
            if (!is_swa) {
                k_dyn.resize(B * c_.n_kv * 4);
                v_dyn.resize(B * c_.n_kv * 4);
                matmul_f32(in1.data(), trunk_f32_.at(p + ".attn.k_sconv.weight_proj").data(), k_dyn.data(), B, c_.n_kv * 4, D);
                matmul_f32(in2.data(), trunk_f32_.at(p + ".attn.v_sconv.weight_proj").data(), v_dyn.data(), B, c_.n_kv * 4, D);
            }

            // k2 + k3: Per-token causal ShortConv (K/V) + L2 clip + V RMSNorm + tau + KV store + RelPos Sink GQA!
            // Note: tokens belonging to the SAME slot in batch MUST be processed in ascending position order for causal ShortConv & KV visibility.
            const float* k_sw = static_sconv_.at(p + ".k_sconv").data();
            const float* v_sw = static_sconv_.at(p + ".v_sconv").data();
            const auto& q_clip_vec = trunk_f32_.at(p + ".attn.q_l2_clip.scale");
            const auto& r_clip_vec = trunk_f32_.at(p + ".attn.rel_l2_clip.scale");
            const float* q_clip_sc = q_clip_vec.data();
            const float* r_clip_sc = r_clip_vec.data();
            const bool q_clip_per_head = q_clip_vec.size() >= static_cast<size_t>(c_.n_head * c_.head_dim);
            const bool r_clip_per_head = r_clip_vec.size() >= static_cast<size_t>(c_.n_head * c_.d_rel);
            const float* rel_proj = trunk_f32_.at(p + ".attn.rel_logits_proj.proj").data();  // [16, 1024]

            std::vector<float> attn_out(B * q_dim, 0.0f);
            std::vector<float> k_conv(kv_dim), v_conv(kv_dim), h_tmp(3 * kv_dim);
            std::vector<float> k_mem(kv_dim), v_mem(kv_dim);
            std::vector<float> rel_tab(c_.n_head * (c_.rel_extent + 1));
            const int n_rep = c_.n_head / c_.n_kv;
            const float attn_scale = 1.0f / std::sqrt(static_cast<float>(c_.head_dim));

            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                const int pos_b = batch.pos[b];

                // ShortConv on K & V
                uint16_t* kh_ptr = sl.sconv.k_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim;
                uint16_t* vh_ptr = sl.sconv.v_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim;
                SlotShortConvState::load_3lag(kh_ptr, h_tmp.data(), kv_dim);
                shortconv4_step_f32(k_raw.data() + static_cast<size_t>(b) * kv_dim, k_sw,
                                    is_swa ? nullptr : (k_dyn.data() + static_cast<size_t>(b) * c_.n_kv * 4),
                                    kv_dim, c_.n_kv, h_tmp.data(), k_conv.data());
                SlotShortConvState::store_3lag(h_tmp.data(), kh_ptr, kv_dim);
                if (sl.recording_snaps) {
                    const int st = pos_b - sl.snap_seq_len + 1;
                    if (st >= 1 && st < static_cast<int>(sl.sconv_snaps.size())) {
                        std::memcpy(sl.sconv_snaps[st].k_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim,
                                    kh_ptr, static_cast<size_t>(3 * kv_dim) * sizeof(uint16_t));
                    }
                }

                SlotShortConvState::load_3lag(vh_ptr, h_tmp.data(), kv_dim);
                shortconv4_step_f32(v_raw.data() + static_cast<size_t>(b) * kv_dim, v_sw,
                                    is_swa ? nullptr : (v_dyn.data() + static_cast<size_t>(b) * c_.n_kv * 4),
                                    kv_dim, c_.n_kv, h_tmp.data(), v_conv.data());
                SlotShortConvState::store_3lag(h_tmp.data(), vh_ptr, kv_dim);
                if (sl.recording_snaps) {
                    const int st = pos_b - sl.snap_seq_len + 1;
                    if (st >= 1 && st < static_cast<int>(sl.sconv_snaps.size())) {
                        std::memcpy(sl.sconv_snaps[st].v_hist.data() + static_cast<size_t>(visit) * 3 * kv_dim,
                                    vh_ptr, static_cast<size_t>(3 * kv_dim) * sizeof(uint16_t));
                    }
                }

                // L2 clip Q/K/R, V RMSNorm, tau scaling on global layers
                float* qb = q.data() + static_cast<size_t>(b) * q_dim;
                float* rb = r_raw.data() + static_cast<size_t>(b) * c_.n_head * c_.d_rel;
                for (int h = 0; h < c_.n_head; ++h) {
                    const float* qsc = q_clip_per_head ? (q_clip_sc + h * c_.head_dim) : q_clip_sc;
                    const float* rsc = r_clip_per_head ? (r_clip_sc + h * c_.d_rel) : r_clip_sc;
                    l2_clip_f32(qb + h * c_.head_dim, qsc, qb + h * c_.head_dim, c_.head_dim, c_.qk_l2_clip);
                    l2_clip_f32(rb + h * c_.d_rel, rsc, rb + h * c_.d_rel, c_.d_rel, c_.rel_l2_clip);
                    if (!is_swa) {
                        for (int j = 0; j < c_.head_dim; ++j) qb[h * c_.head_dim + j] *= tau[b];
                        for (int j = 0; j < c_.d_rel; ++j) rb[h * c_.d_rel + j] *= tau[b];
                    }
                }
                for (int kh = 0; kh < c_.n_kv; ++kh) {
                    l2_clip_f32(k_conv.data() + kh * c_.head_dim, nullptr, k_conv.data() + kh * c_.head_dim, c_.head_dim, c_.qk_l2_clip);
                    rms_norm_f32(v_conv.data() + kh * c_.head_dim, nullptr, v_conv.data() + kh * c_.head_dim, c_.head_dim, c_.rms_eps);
                }

                // Store into paged SWA / Global KV
                if (is_swa) sl.kv.store_swa(kv_slot, pos_b, k_conv.data(), v_conv.data());
                else        sl.kv.store_global(kv_slot, pos_b, k_conv.data(), v_conv.data());

                // Compute relative-position bias table[h, d] for d in 0..min(pos_b, rel_extent)
                const int k_start = is_swa ? std::max(0, pos_b - c_.swa_win + 1) : 0;
                const int k_len = pos_b - k_start + 1;
                const int max_d = std::min(pos_b, c_.rel_extent - 1);
                for (int h = 0; h < c_.n_head; ++h) {
                    const float* rh = rb + h * c_.d_rel;
                    float* th = rel_tab.data() + h * (c_.rel_extent + 1);
                    for (int d = 0; d <= max_d; ++d) {
                        float s = 0.0f;
                        for (int rc = 0; rc < c_.d_rel; ++rc) s += rh[rc] * rel_proj[rc * c_.rel_extent + d];
                        th[d] = s;
                    }
                    th[c_.rel_extent] = 0.0f;
                }

                // GQA attention with online softmax + per-query sink!
                std::vector<float> h_max(c_.n_head), h_sum(c_.n_head, 1.0f);
                for (int h = 0; h < c_.n_head; ++h) {
                    // Initialize softmax accumulator with the sink score!
                    h_max[h] = sinks[b * c_.n_head + h];
                    h_sum[h] = 1.0f;
                }
                float* ao = attn_out.data() + static_cast<size_t>(b) * q_dim;
                for (int kp = k_start; kp <= pos_b; ++kp) {
                    if (is_swa) sl.kv.load_swa(kv_slot, kp, k_mem.data(), v_mem.data());
                    else        sl.kv.load_global(kv_slot, kp, k_mem.data(), v_mem.data());
                    const int dist = std::min(pos_b - kp, c_.rel_extent);
                    for (int h = 0; h < c_.n_head; ++h) {
                        const int kvh = h / n_rep;
                        const float* qh = qb + h * c_.head_dim;
                        const float* kh = k_mem.data() + kvh * c_.head_dim;
                        const float* vh = v_mem.data() + kvh * c_.head_dim;
                        const float sc = dot_f32_fast(qh, kh, c_.head_dim) * attn_scale +
                                         rel_tab[h * (c_.rel_extent + 1) + dist];
                        float* oh = ao + h * c_.head_dim;
                        if (sc > h_max[h]) {
                            const float exp_old = std::exp(h_max[h] - sc);
                            h_sum[h] = h_sum[h] * exp_old + 1.0f;
                            for (int j = 0; j < c_.head_dim; ++j) oh[j] = oh[j] * exp_old + vh[j];
                            h_max[h] = sc;
                        } else {
                            const float w = std::exp(sc - h_max[h]);
                            h_sum[h] += w;
                            for (int j = 0; j < c_.head_dim; ++j) oh[j] += w * vh[j];
                        }
                    }
                }
                for (int h = 0; h < c_.n_head; ++h) {
                    const float inv = 1.0f / h_sum[h];
                    float* oh = ao + h * c_.head_dim;
                    const float* gh = gate_a.data() + (static_cast<size_t>(b) * q_dim + h * c_.head_dim);
                    for (int j = 0; j < c_.head_dim; ++j) {
                        const float sig = 1.0f / (1.0f + std::exp(-gh[j]));
                        oh[j] = (oh[j] * inv) * sig;
                    }
                }
            }

            // O-proj + post-attention ShortConv
            std::vector<float> o_proj(B * D);
            matmul_f32(attn_out.data(), trunk_f32_.at(p + ".attn.wo_ud.weight").data(), o_proj.data(), B, D, q_dim);
            const float* attn_sw = static_sconv_.at(p + ".attn_sconv").data();
            std::vector<float> a_sconv(B * D), h_d(3 * D);
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                uint16_t* ah_ptr = sl.sconv.attn_hist.data() + static_cast<size_t>(visit) * 3 * D;
                SlotShortConvState::load_3lag(ah_ptr, h_d.data(), D);
                shortconv4_step_f32(o_proj.data() + static_cast<size_t>(b) * D, attn_sw, nullptr, D, 0, h_d.data(),
                                    a_sconv.data() + static_cast<size_t>(b) * D);
                SlotShortConvState::store_3lag(h_d.data(), ah_ptr, D);
                if (sl.recording_snaps) {
                    const int st = batch.pos[b] - sl.snap_seq_len + 1;
                    if (st >= 1 && st < static_cast<int>(sl.sconv_snaps.size())) {
                        std::memcpy(sl.sconv_snaps[st].attn_hist.data() + static_cast<size_t>(visit) * 3 * D,
                                    ah_ptr, static_cast<size_t>(3 * D) * sizeof(uint16_t));
                    }
                }
            }

            // Attention-Residual block boundary rule
            if (ip % c_.block_size == 0) {
                blocks.push_back(residual);
                cur = std::move(a_sconv);
            } else {
                for (size_t i = 0; i < cur.size(); ++i) cur[i] = residual[i] + a_sconv[i];
            }

            // MoE sublayer: AttnRes mix + optional FFN loop controller + mlp_norm
            residual = cur;
            const auto& mlp_q = trunk_f32_.at("model.llm.attn_res.mlp_queries." + std::to_string(ip));
            std::vector<float> mlp_in(B * D);
            #pragma omp parallel for schedule(static) if (B > 1)
            for (int b = 0; b < B; ++b) {
                std::vector<const float*> blk_ptrs(blocks.size());
                for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
                attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, mlp_q.data(), 1, D,
                                 mlp_in.data() + static_cast<size_t>(b) * D);
            }
            if (msgs != nullptr) {
                const std::string cp = p + ".controllers.mlp";
                const float* wd = trunk_f32_.at(cp + ".down_weight").data();
                const float* wg = trunk_f32_.at(cp + ".gain_weight").data();
                const float* wf = trunk_f32_.at(cp + ".forget_weight").data();
                #pragma omp parallel for schedule(static) if (B > 1)
                for (int b = 0; b < B; ++b) {
                    std::vector<float> out_c(D);
                    loop_control_f32(mlp_in.data() + static_cast<size_t>(b) * D,
                                     msgs->data() + (static_cast<size_t>(b) * 4 + 3) * D,
                                     wd, wg, wf, D, c_.gate_rank, out_c.data());
                    std::memcpy(mlp_in.data() + static_cast<size_t>(b) * D, out_c.data(), D * sizeof(float));
                }
            }
            const float* mn = trunk_f32_.at(p + ".mlp_norm.weight").data();
            for (int b = 0; b < B; ++b) {
                rms_norm_f32(mlp_in.data() + static_cast<size_t>(b) * D, mn, mlp_in.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
            }

            // Top-8 / 128 Sigmoid Router with F32 bias + COALESCED BATCHED MoE DISPATCH across all B tokens!
            std::vector<float> r_logits(B * c_.n_experts);
            matmul_f32(mlp_in.data(), trunk_f32_.at(p + ".mlp.gate.weight").data(), r_logits.data(), B, c_.n_experts, D);
            const float* r_bias = trunk_f32_.at(p + ".mlp.gate.bias").data();

            // Group (token_idx, weight) by expert_id so every expert body is read ONCE across all B=1..64 slots!
            std::vector<std::vector<std::pair<int, float>>> exp_tokens(c_.n_experts);
            for (int b = 0; b < B; ++b) {
                std::vector<std::pair<float, int>> sc(c_.n_experts);
                std::vector<float> probs(c_.n_experts);
                for (int e = 0; e < c_.n_experts; ++e) {
                    float pr = 1.0f / (1.0f + std::exp(-r_logits[b * c_.n_experts + e]));
                    probs[e] = pr;
                    sc[e] = {pr + r_bias[e], e};
                }
                std::partial_sort(sc.begin(), sc.begin() + c_.top_k, sc.end(),
                                  [](const auto& a, const auto& c) { return a.first > c.first || (a.first == c.first && a.second < c.second); });
                float wsum = 0.0f;
                for (int k = 0; k < c_.top_k; ++k) wsum += probs[sc[k].second];
                const float inv_w = c_.route_scale / (wsum + 1e-20f);
                for (int k = 0; k < c_.top_k; ++k) {
                    int eid = sc[k].second;
                    exp_tokens[eid].push_back({b, probs[eid] * inv_w});
                }
            }

            // Collect active experts and execute batched GEMM per expert!
            std::vector<int> active_exps;
            for (int e = 0; e < c_.n_experts; ++e) if (!exp_tokens[e].empty()) active_exps.push_back(e);

            std::vector<float> moe_out(static_cast<size_t>(B) * D, 0.0f);
            std::vector<std::vector<float>> exp_y(active_exps.size());
            const auto& dir = model_->expert_directory();
            std::string io_err;
            bool io_ok = true;

            #pragma omp parallel for schedule(dynamic)
            for (int ai = 0; ai < static_cast<int>(active_exps.size()); ++ai) {
                const int eid = active_exps[ai];
                const int dir_idx = model_->dir_index_of[ip * c_.n_experts + eid];
                if (dir_idx < 0) continue;
                const auto& ent = dir[dir_idx];
                const auto& tok_list = exp_tokens[eid];
                const size_t m = tok_list.size();
                std::vector<float> xe(m * D);
                for (size_t t = 0; t < m; ++t) {
                    std::memcpy(xe.data() + t * D, mlp_in.data() + static_cast<size_t>(tok_list[t].first) * D, D * sizeof(float));
                }
                const uint8_t* body_ptr = nullptr;
                std::vector<uint8_t> disk_buf;
                if (!expert_bodies_[dir_idx].empty()) {
                    body_ptr = expert_bodies_[dir_idx].data();
                } else {
                    #pragma omp critical
                    {
                        std::string e2;
                        if (!model_->read_expert_body(dir_idx, disk_buf, e2)) {
                            io_ok = false;
                            io_err = e2;
                        }
                    }
                    if (!disk_buf.empty()) body_ptr = disk_buf.data();
                }
                if (body_ptr) {
                    exp_y[ai].resize(m * D);
                    eval_inkling2_expert_batch(body_ptr,
                                               static_cast<SkCode>(ent.gu_code_id),
                                               static_cast<SkCode>(ent.dn_code_id),
                                               ent.dn_offset, xe.data(), m, D, c_.inter, exp_y[ai].data());
                }
            }
            if (!io_ok) { err = io_err; return false; }

            // Deterministic scatter-add of active expert outputs
            for (size_t ai = 0; ai < active_exps.size(); ++ai) {
                const auto& tok_list = exp_tokens[active_exps[ai]];
                const auto& ye = exp_y[ai];
                if (ye.empty()) continue;
                for (size_t t = 0; t < tok_list.size(); ++t) {
                    const int b = tok_list[t].first;
                    const float w = tok_list[t].second;
                    float* dst = moe_out.data() + static_cast<size_t>(b) * D;
                    const float* src = ye.data() + t * D;
                    for (int d = 0; d < D; ++d) dst[d] += src[d] * w;
                }
            }

            // Post-MLP ShortConv + residual add
            const float* mlp_sw = static_sconv_.at(p + ".mlp_sconv").data();
            std::vector<float> m_sconv(B * D);
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                uint16_t* mh_ptr = sl.sconv.mlp_hist.data() + static_cast<size_t>(visit) * 3 * D;
                SlotShortConvState::load_3lag(mh_ptr, h_d.data(), D);
                shortconv4_step_f32(moe_out.data() + static_cast<size_t>(b) * D, mlp_sw, nullptr, D, 0, h_d.data(),
                                    m_sconv.data() + static_cast<size_t>(b) * D);
                SlotShortConvState::store_3lag(h_d.data(), mh_ptr, D);
                if (sl.recording_snaps) {
                    const int st = batch.pos[b] - sl.snap_seq_len + 1;
                    if (st >= 1 && st < static_cast<int>(sl.sconv_snaps.size())) {
                        std::memcpy(sl.sconv_snaps[st].mlp_hist.data() + static_cast<size_t>(visit) * 3 * D,
                                    mh_ptr, static_cast<size_t>(3 * D) * sizeof(uint16_t));
                    }
                }
            }
            for (size_t i = 0; i < cur.size(); ++i) cur[i] = residual[i] + m_sconv[i];
            ++visit;
            return true;
        };

        // Prelude: physical layers 0..5
        for (int ip = 0; ip < c_.core_start; ++ip) {
            if (!run_visit(ip, 0, nullptr)) return false;
        }

        // Looped Core: physical layers 6..17 x 3 passes
        prelude = cur;
        prelude_blocks = blocks;
        for (int p_idx = 0; p_idx < c_.num_passes; ++p_idx) {
            if (p_idx > 0) {
                bank.assign(blocks.begin() + static_cast<ptrdiff_t>(prelude_blocks.size() + 1), blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = c_.core_start; ip < c_.core_end; ++ip) {
                std::vector<float> msgs;
                const std::vector<float>* msgs_ptr = nullptr;
                if (p_idx > 0) {
                    msgs.resize(static_cast<size_t>(B) * 4 * D);
                    const float* mq = msg_queries_[ip].data();
                    #pragma omp parallel for schedule(static) if (B > 1)
                    for (int b = 0; b < B; ++b) {
                        std::vector<const float*> bank_ptrs(bank.size());
                        for (size_t k = 0; k < bank.size(); ++k) bank_ptrs[k] = bank[k].data() + static_cast<size_t>(b) * D;
                        attn_res_mix_f32(bank_ptrs, bank_out.data() + static_cast<size_t>(b) * D,
                                         mq, 4, D, msgs.data() + static_cast<size_t>(b) * 4 * D);
                    }
                    msgs_ptr = &msgs;
                }
                if (!run_visit(ip, p_idx, msgs_ptr)) return false;
            }
        }

        // Coda: physical layers 18..23
        for (int ip = c_.core_end; ip < c_.n_phys; ++ip) {
            if (!run_visit(ip, 0, nullptr)) return false;
        }

        // 3. Output depth mix + output_norm
        const auto& out_q = trunk_f32_.at("model.llm.attn_res.output_query");
        const auto& out_nw = trunk_f32_.at("model.llm.norm.weight");
        std::vector<float> h_norm(static_cast<size_t>(B) * D);
        #pragma omp parallel for schedule(static) if (B > 1)
        for (int b = 0; b < B; ++b) {
            std::vector<const float*> blk_ptrs(blocks.size());
            for (size_t k = 0; k < blocks.size(); ++k) blk_ptrs[k] = blocks[k].data() + static_cast<size_t>(b) * D;
            std::vector<float> h_out(D);
            attn_res_mix_f32(blk_ptrs, cur.data() + static_cast<size_t>(b) * D, out_q.data(), 1, D, h_out.data());
            rms_norm_f32(h_out.data(), out_nw.data(), h_norm.data() + static_cast<size_t>(b) * D, D, c_.rms_eps);
        }
        maybe_tap(c_.n_visits, h_norm);

        // Update slot seq_len and DSpark context cache
        for (int b = 0; b < B; ++b) {
            auto& sl = slots_[batch.slot_ids[b]];
            sl.kv.seq_len = std::max(sl.kv.seq_len, batch.pos[b] + 1);
            if (has_dspark_ && batch.capture_taps) {
                update_dspark_ctx_for_token(sl, batch.pos[b], taps, b);
            }
        }
        for (int b = 0; b < B; ++b) slots_[batch.slot_ids[b]].recording_snaps = false;
        if (taps_out) {
            for (int k = 0; k < 4; ++k) taps_out->taps[k] = std::move(taps[k]);
        }

        // 4. Unembed + logit softcap(30.0) + slice[:vocab_unpadded] ONLY for tokens with want_logits[b] == 1
        if (logits_out != nullptr) {
            const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
            const int V = c_.vocab_unpadded;
            const float cap = c_.logit_softcap;
            const float inv_cap = 1.0f / cap;
            std::vector<int> want_idx;
            for (int b = 0; b < B; ++b) if (batch.want_logits[b]) want_idx.push_back(b);
            const int n_want = static_cast<int>(want_idx.size());
            if (n_want > 0) {
                #pragma omp parallel for schedule(static)
                for (int v = 0; v < V; ++v) {
                    float wrow[4096];  // D=2304 <= 4096
                    decode_packed_row(unemb_pk, v, wrow);
                    for (int wi = 0; wi < n_want; ++wi) {
                        const float* hb = h_norm.data() + static_cast<size_t>(want_idx[wi]) * D;
                        const float z = dot_f32_fast(hb, wrow, D);
                        logits_out[static_cast<size_t>(wi) * V + v] = cap * std::tanh(z * inv_cap);
                    }
                }
            }
        }
        return true;
    }

    // Full exact 3-Layer Block-8 Recurrent Markov DSpark drafter!
    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft, int32_t* out_ids, std::string& err) override {
        if (!has_dspark_ || max_draft <= 0) return 0;
        if (slot < 0 || slot >= static_cast<int>(slots_.size()) || !slots_[slot].allocated) {
            err = "dspark: invalid slot";
            return 0;
        }
        auto& sl = slots_[slot];
        if (anchor_pos < 1 || sl.dspark_len < anchor_pos) return 0;

        const int B = dspark_cfg_.block_size;  // 8
        const int n_draft = std::min(max_draft, B - 1);
        const int D = dspark_cfg_.hidden;
        const int H = dspark_cfg_.n_head;
        const int KV = dspark_cfg_.n_kv;
        const int hd = dspark_cfg_.head_dim;
        const int kv_dim = KV * hd;
        const int q_dim = H * hd;
        const int inter = dspark_cfg_.inter;
        const int rank = dspark_cfg_.markov_rank;

        // Build draft block [B, D]: slot 0 = rms(embed(anchor_tok)), slots 1..B-1 = mask_embed
        std::vector<float> cur(B * D);
        std::vector<float> raw_emb(D);
        decode_packed_row(packed_trunk_.at("model.llm.embed.weight"), anchor_tok, raw_emb.data());
        rms_norm_f32(raw_emb.data(), trunk_f32_.at("model.llm.embed_norm.weight").data(), cur.data(), D, c_.rms_eps);
        const float* mask_e = dspark_f32_.at("mask_embed").data();
        for (int i = 1; i < B; ++i) std::memcpy(cur.data() + i * D, mask_e, D * sizeof(float));

        const int n_rep = H / KV;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

        // 3 non-causal block-attention layers (attending to ctx < anchor_pos + its own B draft tokens)
        for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
            const std::string p = "layers." + std::to_string(il);
            std::vector<float> res = cur, h(B * D);
            const float* in_ln = dspark_f32_.at(p + ".input_layernorm.weight").data();
            for (int i = 0; i < B; ++i) rms_norm_f32(cur.data() + i * D, in_ln, h.data() + i * D, D, dspark_cfg_.rms_eps);

            std::vector<float> q(B * q_dim), kd(B * kv_dim), vd(B * kv_dim);
            matmul_f32(h.data(), dspark_f32_.at(p + ".self_attn.q_proj.weight").data(), q.data(), B, q_dim, D);
            matmul_f32(h.data(), dspark_f32_.at(p + ".self_attn.k_proj.weight").data(), kd.data(), B, kv_dim, D);
            matmul_f32(h.data(), dspark_f32_.at(p + ".self_attn.v_proj.weight").data(), vd.data(), B, kv_dim, D);

            const float* qnw = dspark_f32_.at(p + ".self_attn.q_norm.weight").data();
            const float* knw = dspark_f32_.at(p + ".self_attn.k_norm.weight").data();
            for (int i = 0; i < B; ++i) {
                for (int hh = 0; hh < H; ++hh) rms_norm_f32(q.data() + (i * H + hh) * hd, qnw, q.data() + (i * H + hh) * hd, hd, dspark_cfg_.rms_eps);
                for (int kh = 0; kh < KV; ++kh) rms_norm_f32(kd.data() + (i * KV + kh) * hd, knw, kd.data() + (i * KV + kh) * hd, hd, dspark_cfg_.rms_eps);
                apply_half_rope_vec(q.data() + i * q_dim, H, hd, anchor_pos + i, dspark_cfg_.rope_theta);
                apply_half_rope_vec(kd.data() + i * kv_dim, KV, hd, anchor_pos + i, dspark_cfg_.rope_theta);
            }

            const int c0 = dspark_cfg_.is_swa[il] ? std::max(0, anchor_pos - dspark_cfg_.swa_win + 1) : 0;
            const int c_len = anchor_pos - c0;
            const int tot_k = c_len + B;
            std::vector<float> attn_out(B * q_dim, 0.0f);

            for (int qi = 0; qi < B; ++qi) {
                std::vector<float> sc(tot_k);
                for (int hh = 0; hh < H; ++hh) {
                    const int kvh = hh / n_rep;
                    const float* qv = q.data() + (qi * H + hh) * hd;
                    float mx = -1e30f;
                    for (int ki = 0; ki < c_len; ++ki) {
                        const uint16_t* kp = sl.dspark_k[il].data() + (static_cast<size_t>(c0 + ki) * KV + kvh) * hd;
                        float dot = 0.0f;
                        for (int j = 0; j < hd; ++j) dot += qv[j] * bf16_to_f32(kp[j]);
                        sc[ki] = dot * attn_scale;
                        if (sc[ki] > mx) mx = sc[ki];
                    }
                    for (int di = 0; di < B; ++di) {
                        const float* kp = kd.data() + (di * KV + kvh) * hd;
                        float dot = dot_f32_fast(qv, kp, hd) * attn_scale;
                        sc[c_len + di] = dot;
                        if (dot > mx) mx = dot;
                    }
                    float sum = 0.0f;
                    for (int t = 0; t < tot_k; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                    const float inv = 1.0f / sum;
                    float* ov = attn_out.data() + (qi * H + hh) * hd;
                    for (int ki = 0; ki < c_len; ++ki) {
                        const float pr = sc[ki] * inv;
                        const uint16_t* vp = sl.dspark_v[il].data() + (static_cast<size_t>(c0 + ki) * KV + kvh) * hd;
                        for (int j = 0; j < hd; ++j) ov[j] += pr * bf16_to_f32(vp[j]);
                    }
                    for (int di = 0; di < B; ++di) {
                        const float pr = sc[c_len + di] * inv;
                        const float* vp = vd.data() + (di * KV + kvh) * hd;
                        for (int j = 0; j < hd; ++j) ov[j] += pr * vp[j];
                    }
                }
            }

            std::vector<float> o_out(B * D);
            matmul_f32(attn_out.data(), dspark_f32_.at(p + ".self_attn.o_proj.weight").data(), o_out.data(), B, D, q_dim);
            for (int i = 0; i < B * D; ++i) cur[i] = res[i] + o_out[i];

            res = cur;
            const float* post_ln = dspark_f32_.at(p + ".post_attention_layernorm.weight").data();
            for (int i = 0; i < B; ++i) rms_norm_f32(cur.data() + i * D, post_ln, h.data() + i * D, D, dspark_cfg_.rms_eps);
            std::vector<float> mg(B * inter), mu(B * inter);
            matmul_f32(h.data(), dspark_f32_.at(p + ".mlp.gate_proj.weight").data(), mg.data(), B, inter, D);
            matmul_f32(h.data(), dspark_f32_.at(p + ".mlp.up_proj.weight").data(), mu.data(), B, inter, D);
            for (int i = 0; i < B * inter; ++i) {
                const float g = mg[i];
                mg[i] = (g / (1.0f + std::exp(-g))) * mu[i];
            }
            matmul_f32(mg.data(), dspark_f32_.at(p + ".mlp.down_proj.weight").data(), o_out.data(), B, D, inter);
            for (int i = 0; i < B * D; ++i) cur[i] = res[i] + o_out[i];
        }

        // Final norm -> unembed base logits for slots 1..n_draft + 7-step Recurrent Markov head!
        const float* fnw = dspark_f32_.at("norm.weight").data();
        for (int i = 0; i < B; ++i) rms_norm_f32(cur.data() + i * D, fnw, cur.data() + i * D, D, dspark_cfg_.rms_eps);

        std::vector<float> context(D);
        const uint16_t* fr_src = sl.dspark_frnn.data() + static_cast<size_t>(anchor_pos - 1) * D;
        for (int d = 0; d < D; ++d) context[d] = bf16_to_f32(fr_src[d]);

        const float* mw1 = dspark_f32_.at("markov_head.markov_w1.weight").data();
        const float* rpw = dspark_f32_.at("markov_head.rnn_proj.weight").data();
        const float* rpb = dspark_f32_.at("markov_head.rnn_proj.bias").data();
        const float* how = dspark_f32_.at("markov_head.hidden_out.weight").data();
        const float* hob = dspark_f32_.at("markov_head.hidden_out.bias").data();
        const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
        const int V = c_.vocab_unpadded;

        std::vector<float> state(rank, 0.0f);
        std::vector<float> joint(2 * rank + 2 * D), proj(3 * rank), latent(rank), corr(D), combined(D);
        int32_t prev_tok = anchor_tok;
        for (int step = 1; step <= n_draft; ++step) {
            std::memcpy(joint.data(), state.data(), rank * sizeof(float));
            std::memcpy(joint.data() + rank, mw1 + static_cast<size_t>(prev_tok) * rank, rank * sizeof(float));
            std::memcpy(joint.data() + 2 * rank, cur.data() + step * D, D * sizeof(float));
            std::memcpy(joint.data() + 2 * rank + D, context.data(), D * sizeof(float));
            for (int r = 0; r < 3 * rank; ++r) {
                proj[r] = dot_f32_fast(rpw + static_cast<size_t>(r) * (2 * rank + 2 * D), joint.data(), 2 * rank + 2 * D) + rpb[r];
            }
            for (int r = 0; r < rank; ++r) {
                const float g = 1.0f / (1.0f + std::exp(-proj[r]));
                const float cand = std::tanh(proj[rank + r]);
                state[r] = cand + g * (state[r] - cand);
                latent[r] = std::tanh(proj[2 * rank + r]);
            }
            for (int d = 0; d < D; ++d) {
                corr[d] = dot_f32_fast(how + static_cast<size_t>(d) * rank, latent.data(), rank) + hob[d];
                // Notice: base_logits + unembed @ corr == unembed @ (cur[step] + corr)! Exact linear identity saves 2x unembed GEMV!
                combined[d] = cur[step * D + d] + corr[d];
            }
            // Deterministic first-max argmax over unpadded vocab (softcap is strictly monotone!)
            int32_t best_id = 0;
            float best_z = -1e30f;
            #pragma omp parallel
            {
                int32_t loc_id = 0;
                float loc_z = -1e30f;
                float wrow[4096];
                #pragma omp for schedule(static)
                for (int v = 0; v < V; ++v) {
                    decode_packed_row(unemb_pk, v, wrow);
                    float z = dot_f32_fast(combined.data(), wrow, D);
                    if (z > loc_z || (z == loc_z && v < loc_id)) {
                        loc_z = z;
                        loc_id = v;
                    }
                }
                #pragma omp critical
                {
                    if (loc_z > best_z || (loc_z == best_z && loc_id < best_id)) {
                        best_z = loc_z;
                        best_id = loc_id;
                    }
                }
            }
            out_ids[step - 1] = best_id;
            prev_tok = best_id;
        }
        return n_draft;
    }

    bool selfcheck(std::string& err) override {
        // 1. Verify SF4 1-instruction SIMD LUT dot product against scalar reference
        const size_t ncols = 2304;
        std::vector<uint8_t> sf4_p(ncols / 2), sf4_s(ncols / 16);
        std::vector<float> x(ncols), w_dec(ncols);
        for (size_t i = 0; i < sf4_p.size(); ++i) sf4_p[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFFu);
        for (size_t i = 0; i < sf4_s.size(); ++i) sf4_s[i] = static_cast<uint8_t>(0x38u + (i % 16));
        for (size_t i = 0; i < ncols; ++i) x[i] = std::sin(static_cast<float>(i) * 0.013f);
        dequant_sf4_row(sf4_p.data(), sf4_s.data(), 0.25f, w_dec.data(), ncols);
        float ref = 0.0f;
        for (size_t i = 0; i < ncols; ++i) ref += w_dec[i] * x[i];
        float got = dot_sf4_f32_row(sf4_p.data(), sf4_s.data(), 0.25f, x.data(), ncols);
        if (std::fabs(ref - got) > 1e-4f * std::max(1.0f, std::fabs(ref))) {
            err = "SF4 SIMD LUT selfcheck mismatch: ref=" + std::to_string(ref) + " got=" + std::to_string(got);
            return false;
        }
        // 2. Verify MX8 dot product
        std::vector<uint8_t> mx8_w(ncols), mx8_s(ncols / 32);
        for (size_t i = 0; i < ncols; ++i) mx8_w[i] = static_cast<uint8_t>((i * 29 + 7) & 0x7Eu);
        for (size_t i = 0; i < mx8_s.size(); ++i) mx8_s[i] = static_cast<uint8_t>(125 + (i % 5));
        dequant_mx8_row(mx8_w.data(), mx8_s.data(), w_dec.data(), ncols);
        ref = 0.0f;
        for (size_t i = 0; i < ncols; ++i) ref += w_dec[i] * x[i];
        got = dot_mx8_f32_row(mx8_w.data(), mx8_s.data(), x.data(), ncols);
        if (std::fabs(ref - got) > 1e-4f * std::max(1.0f, std::fabs(ref))) {
            err = "MX8 selfcheck mismatch: ref=" + std::to_string(ref) + " got=" + std::to_string(got);
            return false;
        }
        // 3. Verify FP8 E4M3 KV round-trip accuracy
        for (float v : {-12.5f, -1.0f, -0.125f, 0.0f, 0.25f, 3.5f, 48.0f}) {
            float rt = fp8_e4m3_to_f32(f32_to_fp8_e4m3(v));
            if (std::fabs(rt - v) > std::max(0.05f, std::fabs(v) * 0.1f)) {
                err = "FP8 E4M3 KV round-trip error";
                return false;
            }
        }
        // 4. Verify K8V8 & K8V4 KV round-trip accuracy (#6520 / @merbanan)
        {
            std::vector<float> kv_in(1024), kv_k(1024), kv_v(1024);
            for (int i = 0; i < 1024; ++i) kv_in[i] = std::sin(static_cast<float>(i) * 0.031f) * 2.5f;
            SlotKvPages p8, p4;
            p8.init(1, 1, 1024, 64, KvDtype::K8V8);
            p8.store_swa(0, 0, kv_in.data(), kv_in.data());
            p8.load_swa(0, 0, kv_k.data(), kv_v.data());
            for (int i = 0; i < 1024; ++i) {
                if (std::fabs(kv_k[i] - kv_in[i]) > 0.05f || std::fabs(kv_v[i] - kv_in[i]) > 0.05f) {
                    err = "K8V8 KV round-trip error";
                    return false;
                }
            }
            p4.init(1, 1, 1024, 64, KvDtype::K8V4);
            p4.store_swa(0, 0, kv_in.data(), kv_in.data());
            p4.load_swa(0, 0, kv_k.data(), kv_v.data());
            for (int i = 0; i < 1024; ++i) {
                if (std::fabs(kv_k[i] - kv_in[i]) > 0.05f || std::fabs(kv_v[i] - kv_in[i]) > 0.45f) {
                    err = "K8V4 KV round-trip error";
                    return false;
                }
            }
        }
        // 5. Verify Section 5 payload if .strata model is loaded
        if (model_ && model_->kind == ModelSourceKind::Strata) {
            std::vector<uint8_t> s5;
            if (!model_->rd.read_section(5, s5, err)) return false;
            if (s5.empty()) {
                err = "Section 5 sq.selfcheck is empty";
                return false;
            }
        }
        return true;
    }

private:
    struct PackedTensor {
        TInfo info;
        std::vector<uint8_t> bytes;
    };

    const InkModelFile* model_ = nullptr;
    BackendOpts opts_;
    InkCfg c_;
    DsparkCfg dspark_cfg_;
    KvDtype default_kv_dtype_ = KvDtype::FP16;
    bool has_dspark_ = false;

    std::unordered_map<std::string, std::vector<float>> trunk_f32_;
    std::unordered_map<std::string, PackedTensor> packed_trunk_;
    std::unordered_map<std::string, std::vector<float>> static_sconv_;
    std::vector<std::vector<float>> msg_queries_;
    std::unordered_map<std::string, std::vector<float>> dspark_f32_;
    std::vector<std::vector<uint8_t>> expert_bodies_;
    std::vector<CpuSlot> slots_;

    static void decode_packed_row(const PackedTensor& pk, int row, float* dst) noexcept {
        const size_t cols = static_cast<size_t>(pk.info.cols());
        const uint8_t* base = pk.bytes.data();
        if (pk.info.fmt == F_BF16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + static_cast<size_t>(row) * cols;
            for (size_t c = 0; c < cols; ++c) dst[c] = bf16_to_f32(p[c]);
        } else if (pk.info.fmt == F_F32) {
            std::memcpy(dst, base + static_cast<size_t>(row) * cols * 4, cols * 4);
        } else if (pk.info.fmt == F_Q8ROW) {
            const size_t rows = static_cast<size_t>(pk.info.rows());
            const float sc = reinterpret_cast<const float*>(base)[row];
            const int8_t* q = reinterpret_cast<const int8_t*>(base + rows * 4) + static_cast<size_t>(row) * cols;
            for (size_t c = 0; c < cols; ++c) dst[c] = static_cast<float>(q[c]) * sc;
        } else {
            const size_t sb_b = superblock_bytes(static_cast<SkCode>(pk.info.fmt));
            const size_t row_b = (cols / QK_S) * sb_b;
            dequant_sk_row(base + static_cast<size_t>(row) * row_b, static_cast<SkCode>(pk.info.fmt), dst, cols);
        }
    }

    void update_dspark_ctx_for_token(CpuSlot& sl, int pos, const std::vector<float> (&taps)[4], int b) {
        const int D = dspark_cfg_.hidden;
        const int KV = dspark_cfg_.n_kv;
        const int hd = dspark_cfg_.head_dim;
        const int kv_dim = KV * hd;

        // Locate fuse_weight [4, 4]
        const float* fw = nullptr;
        auto it = dspark_f32_.find("fuse_weight");
        if (it != dspark_f32_.end()) fw = it->second.data();
        else {
            for (const auto& kv : dspark_f32_) if (kv.first.find("fuse") != std::string::npos) { fw = kv.second.data(); break; }
        }
        if (!fw) return;

        auto mix_row = [&](int row, const std::string& norm_k, std::vector<float>& out) {
            float w[4], mx = -1e30f, sum = 0.0f;
            for (int i = 0; i < 4; ++i) mx = std::max(mx, fw[row * 4 + i]);
            for (int i = 0; i < 4; ++i) { w[i] = std::exp(fw[row * 4 + i] - mx); sum += w[i]; }
            for (int i = 0; i < 4; ++i) w[i] /= sum;
            out.assign(D, 0.0f);
            for (int i = 0; i < 4; ++i) {
                const float* tp = taps[i].data() + static_cast<size_t>(b) * D;
                for (int d = 0; d < D; ++d) out[d] += w[i] * tp[d];
            }
            rms_norm_f32(out.data(), dspark_f32_.at(norm_k).data(), out.data(), D, dspark_cfg_.rms_eps);
        };

        std::vector<float> ctx(D), kc(kv_dim), vc(kv_dim);
        for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
            const std::string p = "layers." + std::to_string(il);
            mix_row(il, p + ".dflare_norm.weight", ctx);
            matmul_f32(ctx.data(), dspark_f32_.at(p + ".self_attn.k_proj_ctx.weight").data(), kc.data(), 1, kv_dim, D);
            matmul_f32(ctx.data(), dspark_f32_.at(p + ".self_attn.v_proj_ctx.weight").data(), vc.data(), 1, kv_dim, D);
            const float* knw = dspark_f32_.at(p + ".self_attn.k_norm_ctx.weight").data();
            for (int kh = 0; kh < KV; ++kh) rms_norm_f32(kc.data() + kh * hd, knw, kc.data() + kh * hd, hd, dspark_cfg_.rms_eps);
            apply_half_rope_vec(kc.data(), KV, hd, pos, dspark_cfg_.rope_theta);

            const size_t need = static_cast<size_t>(pos + 1) * kv_dim;
            if (sl.dspark_k[il].size() < need) {
                sl.dspark_k[il].resize(need);
                sl.dspark_v[il].resize(need);
            }
            for (int i = 0; i < kv_dim; ++i) {
                sl.dspark_k[il][static_cast<size_t>(pos) * kv_dim + i] = f32_to_bf16(kc[i]);
                sl.dspark_v[il][static_cast<size_t>(pos) * kv_dim + i] = f32_to_bf16(vc[i]);
            }
        }
        mix_row(dspark_cfg_.n_layers, "markov_head.dflare_norm.weight", ctx);
        const size_t need_fr = static_cast<size_t>(pos + 1) * D;
        if (sl.dspark_frnn.size() < need_fr) sl.dspark_frnn.resize(need_fr);
        for (int d = 0; d < D; ++d) sl.dspark_frnn[static_cast<size_t>(pos) * D + d] = f32_to_bf16(ctx[d]);
        sl.dspark_len = std::max(sl.dspark_len, pos + 1);
    }
};

}  // namespace

std::unique_ptr<InkBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace ink2
