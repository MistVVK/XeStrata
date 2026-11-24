// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sycl_backend.cpp - Inkling-2-Mini's GPU backend: the runner of ink_sy_runner.h on the GPU the
// engine selected (Intel's, NVIDIA's and AMD's alike: ink_sy_common.h).
// stb_image a static copy in each of the engine's two files that read pictures (this one and ink_main.cpp)
#define STB_IMAGE_STATIC
#define INKLING_DEFINE_STBI   // ink_vision.h, included first, compiles it
#include "../ink_backend.h"
#include "../ink_vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ink_sy_runner.h"

namespace ink2 {

class SyclInkBackend final : public InkBackend {
public:
    const char* backend_name() const noexcept override { return "sycl"; }

    std::string device_info() const override {
        const auto& d = ink_sycl::dev_caps();
        return d.name + " (" + d.platform + ", " +
               std::to_string((int)(d.global_mem / (1024ull * 1024ull * 1024ull))) + " GiB, sub-group " +
               std::to_string(d.sg) + (d.matrix_f16 ? ", FP16 matrix engines)" : ")");
    }

    BackendStats stats() const override {
        BackendStats s;
        s.total_experts = (size_t)mod_.man.c.n_phys * (size_t)mod_.man.c.n_experts;
        s.vram_experts_loaded = (size_t)mod_.n_vram_experts;
        s.ram_experts_loaded = (size_t)mod_.n_host_experts;
        s.vram_used_bytes = ink_sycl::dev_bytes_allocated();
        s.vram_total_bytes = ink_sycl::dev_caps().global_mem;
        s.active_slots = active_slots_;
        s.max_slots = runner_.max_slots;
        s.kv_dtype = kv_dtype_;
        return s;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        kv_dtype_ = opts.kv_dtype;
        const int par = std::max(1, std::min(64, opts.parallel));
        const int ctx = std::max(256, opts.max_ctx);
        const int batch_cap = std::max(256, std::max(opts.chunk, par));
        // What the runner allocates after the load (KV, DSpark, ShortConv, activations, vision, the hybrid ring and the
        // GEMM scratch): the experts' budget leaves it free
        size_t kv_est = 0;
        const auto& mc = model.m.c;
        const int core_len = mc.core_end - mc.core_start;
        for (int v = 0; v < mc.n_visits; ++v) {
            const int ip = (v < mc.core_start) ? v
                         : (v < mc.core_start + core_len * mc.num_passes)
                             ? (mc.core_start + (v - mc.core_start) % core_len)
                             : (mc.core_end + (v - mc.core_start - core_len * mc.num_passes));
            const size_t stride = (size_t)(mc.is_swa[ip] ? mc.swa_win : ctx);
            kv_est += 2ull * (size_t)par * stride * (size_t)mc.n_kv * (size_t)mc.head_dim * 2ull;
        }
        const size_t dsp_kv = (opts.enable_dspark && model.m.dspark.present)
            ? (3ull * 2ull * (size_t)par * (size_t)ctx * 1024ull * 2ull) : 0ull;
        const size_t sc_est = (size_t)mc.n_visits * (size_t)par * 3ull * (2048ull + 2ull * (size_t)mc.hidden) * 8ull;
        const size_t act_est = (size_t)batch_cap * 512ull * 1024ull + (128ull << 20);
        const size_t vis_est = !opts.vision_path.empty() ? (700ull << 20) : 0ull;
        const size_t hyb_xmx_est = (320ull << 20);  // Elastic VRAM cache ring (<=256 MiB) + XMX scratch (~32 MiB) + ping-pong
        // the QSA indexer (ink_sy_qsa.h): its weights, the slots' block keys and tails, the batch's scores and ids
        const size_t qsa_blocks = (size_t)ctx / 4 + 1;
        const size_t qsa_est = opts.qsa_path.empty() ? 0ull
            : (24ull << 20) + (size_t)par * 8ull * (qsa_blocks * 128ull + 4ull * (size_t)mc.hidden) * 4ull +
              (size_t)batch_cap * (qsa_blocks + (size_t)opts.qsa_budget + 4ull + 3ull * (size_t)mc.hidden) * 4ull;
        const size_t extra_reserve = kv_est + dsp_kv + sc_est + act_est + vis_est + hyb_xmx_est + qsa_est;
        if (!mod_.load(model, opts.vram_experts_gb, err, extra_reserve, opts.vram_reserve_mib)) return false;
        if (!runner_.init(mod_, par, ctx, batch_cap, opts.enable_dspark, err)) {
            return false;
        }
        if (!opts.vision_path.empty()) {
            if (!runner_.vis.load(opts.vision_path, err)) return false;
        }
        if (!opts.qsa_path.empty() &&
            !runner_.qsa.load(opts.qsa_path, mc.n_visits, mc.hidden, opts.qsa_budget, opts.qsa_local, opts.qsa_sink, par,
                              ctx, runner_.max_batch, err))
            return false;
        for (int v = 0; v < mc.n_visits && runner_.qsa.loaded; ++v)
            if (runner_.qsa.fidx_of_visit[(size_t)v] >= 0 && mc.is_swa[(size_t)mc.visit_to_phys[(size_t)v]]) {
                err = opts.qsa_path + ": its visit " + std::to_string(v) + " is a sliding-window visit of this model";
                return false;
            }
        slot_used_.assign((size_t)par, 0);
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) { err = "sycl slot out of range"; return false; }
        if (!slot_used_[slot]) { slot_used_[slot] = 1; ++active_slots_; }
        runner_.reset_slot(slot);
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < runner_.max_slots && slot_used_[slot]) {
            slot_used_[slot] = 0;
            --active_slots_;
            runner_.reset_slot(slot);
        }
    }

    void reset_slot(int slot) override { runner_.reset_slot(slot); }
    void set_slot_qsa(int slot, bool on) override {
        if (runner_.qsa.loaded && slot >= 0 && slot < runner_.max_slots) runner_.qsa.slot_on[(size_t)slot] = on ? 1 : 0;
    }
    void snapshot_slot(int slot) override { runner_.snapshot_slot(slot); }
    void rollback_slot(int slot, int keep_len) override { runner_.rollback_slot(slot, keep_len); }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        const int M_tot = batch.n_tokens;
        const int D = mod_.man.c.hidden;
        const int V = mod_.man.c.vocab_unpadded;
        const int cap = runner_.max_batch;

        int dst_logit_row = 0;
        for (int off = 0; off < M_tot; off += cap) {
            const int M = std::min(cap, M_tot - off);
            std::vector<uint8_t> is_img;
            std::vector<float> img_emb;
            bool any_img = false;
            for (int m = 0; m < M; ++m) if (batch.img_embed_idx[off + m] >= 0) { any_img = true; break; }
            if (any_img && batch.img_embeds) {
                is_img.assign((size_t)M, 0);
                img_emb.assign((size_t)M * D, 0.0f);
                for (int m = 0; m < M; ++m) {
                    const int idx = batch.img_embed_idx[off + m];
                    if (idx >= 0) {
                        is_img[m] = 1;
                        std::memcpy(img_emb.data() + (size_t)m * D,
                                    batch.img_embeds + (size_t)idx * D,
                                    (size_t)D * sizeof(float));
                    }
                }
            }

            std::vector<int> want_idx;
            if (logits_out != nullptr) {
                for (int m = 0; m < M; ++m) if (batch.want_logits[off + m] != 0) want_idx.push_back(m);
            }
            const int n_want = (int)want_idx.size();
            float* chunk_lg = (n_want > 0 && logits_out != nullptr) ? (logits_out + (size_t)dst_logit_row * V) : nullptr;
            int mode = 0;
            if (n_want == 1 && want_idx[0] == M - 1) mode = 1;
            else if (n_want > 0) mode = 2;

            if (!runner_.forward_batch(batch.token_ids.data() + off, batch.slot_ids.data() + off, batch.pos.data() + off, M,
                                       any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                       mode, chunk_lg, nullptr, batch.capture_taps,
                                       (mode == 2) ? want_idx.data() : nullptr, n_want)) {
                err = "sycl forward_batch failed"; return false;
            }
            dst_logit_row += n_want;
        }
        (void)taps_out;
        return true;
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
                     int32_t* out_draft_ids, std::string& /*err*/) override {
        if (!runner_.use_dspark || max_draft <= 0) return 0;
        int tmp[7]{};
        int n = runner_.dspark_draft_block8(slot, anchor_tok, anchor_pos, tmp);
        int ret = std::min(n, max_draft);
        for (int i = 0; i < ret; ++i) out_draft_ids[i] = tmp[i];
        return ret;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (!runner_.vis.loaded || img.file.empty()) return false;
        int ntok = 0;
        return runner_.vis.encode_image_file(img.file, out_embeds, ntok, err);
    }

    bool selfcheck(std::string& err) override {
        StepBatch b;
        b.push(0, 0, 1, true);
        std::vector<float> lg((size_t)mod_.man.c.vocab_unpadded);
        runner_.reset_slot(0);
        if (!forward_batch(b, lg.data(), nullptr, err)) return false;
        for (float v : lg) {
            if (!std::isfinite(v)) { err = "sycl selfcheck produced non-finite logits"; return false; }
        }
        runner_.reset_slot(0);
        return true;
    }

private:
    ink_sycl::InkSyModel mod_;
    ink_sycl::InkSyRunner runner_;
    std::vector<uint8_t> slot_used_;
    int active_slots_ = 0;
    KvDtype kv_dtype_ = KvDtype::FP16;
};

std::unique_ptr<InkBackend> make_sycl_backend() {
    return std::make_unique<SyclInkBackend>();
}

}  // namespace ink2
