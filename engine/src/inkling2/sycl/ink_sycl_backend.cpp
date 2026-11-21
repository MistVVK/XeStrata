// engine/src/inkling2/sycl/ink_sycl_backend.cpp - InkBackend plugin for Intel Arc SYCL (dg2 + bmg_g31 B70)
// CRITICAL BATTLEMAGE B70 RULE: Zero reqd_sub_group_size(8); strictly SG16!
// MIT License - Copyright (c) 2026 Strata contributors
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
               std::to_string((int)(d.global_mem / (1024ull * 1024ull * 1024ull))) + " GiB, SG16)";
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
        if (!mod_.load(model.path, opts.vram_experts_gb, err)) return false;
        const int par = std::max(1, std::min(64, opts.parallel));
        const int batch_cap = std::max(256, std::max(opts.chunk, par));
        if (!runner_.init(mod_, par, std::max(256, opts.max_ctx), batch_cap, opts.enable_dspark, err)) {
            return false;
        }
        if (!opts.vision_path.empty()) {
            if (!runner_.vis.load(opts.vision_path, err)) return false;
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
    void snapshot_slot(int slot) override { runner_.snapshot_slot(slot); }
    void rollback_slot(int slot, int keep_len) override { runner_.rollback_slot(slot, keep_len); }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        const int M = batch.n_tokens;
        const int D = mod_.man.c.hidden;

        std::vector<uint8_t> is_img;
        std::vector<float> img_emb;
        bool any_img = false;
        for (int idx : batch.img_embed_idx) if (idx >= 0) { any_img = true; break; }
        if (any_img && batch.img_embeds) {
            is_img.assign((size_t)M, 0);
            img_emb.assign((size_t)M * D, 0.0f);
            for (int m = 0; m < M; ++m) {
                if (batch.img_embed_idx[m] >= 0) {
                    is_img[m] = 1;
                    std::memcpy(img_emb.data() + (size_t)m * D,
                                batch.img_embeds + (size_t)batch.img_embed_idx[m] * D,
                                (size_t)D * sizeof(float));
                }
            }
        }

        const int n_want = batch.count_logits();
        const int V = mod_.man.c.vocab_unpadded;
        if (n_want > 0 && logits_out != nullptr) {
            if (n_want == 1 && batch.want_logits[M - 1] != 0) {
                if (!runner_.forward_batch(batch.token_ids.data(), batch.slot_ids.data(), batch.pos.data(), M,
                                           any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                           1, logits_out, nullptr, batch.capture_taps)) {
                    err = "sycl forward_batch failed"; return false;
                }
            } else {
                std::vector<float> all_lg((size_t)M * V);
                if (!runner_.forward_batch(batch.token_ids.data(), batch.slot_ids.data(), batch.pos.data(), M,
                                           any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                           2, all_lg.data(), nullptr, batch.capture_taps)) {
                    err = "sycl forward_batch failed"; return false;
                }
                int dst_row = 0;
                for (int m = 0; m < M; ++m) {
                    if (batch.want_logits[m]) {
                        std::memcpy(logits_out + (size_t)dst_row * V, all_lg.data() + (size_t)m * V, (size_t)V * 4);
                        ++dst_row;
                    }
                }
            }
        } else {
            if (!runner_.forward_batch(batch.token_ids.data(), batch.slot_ids.data(), batch.pos.data(), M,
                                       any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                       0, nullptr, nullptr, batch.capture_taps)) {
                err = "sycl forward_batch failed"; return false;
            }
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
