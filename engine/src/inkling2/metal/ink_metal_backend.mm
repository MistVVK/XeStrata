// engine/src/inkling2/metal/ink_metal_backend.mm - InkBackend plugin for Apple Silicon Metal (strata-inkling)
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

#include "ink_metal_runner.h"

#ifdef INK_METAL_EMBEDDED_SHADERS
#include "ink_metal_embedded_shaders.h"
#endif

namespace ink_metal {
inline bool InkMetalRunner::compile_shaders(const std::string& kernel_dir, std::string& err) {
#ifdef INK_METAL_EMBEDDED_SHADERS
    (void)kernel_dir;
    std::string why;
    lib_codecs = cx->compile_try({}, kInkMetalCodecsSrc, false, &why);
    if (lib_codecs < 0) { err = "codecs MSL: " + why; return false; }
    lib_ops = cx->compile_try({}, kInkMetalOpsSrc, false, &why);
    if (lib_ops < 0) { err = "ops MSL: " + why; return false; }
    lib_dspark = cx->compile_try({}, kInkMetalDsparkSrc, false, &why);
    if (lib_dspark < 0) { err = "dspark MSL: " + why; return false; }
    lib_vis = cx->compile_try({}, kInkMetalVisionSrc, false, &why);
    if (lib_vis < 0) { err = "vision MSL: " + why; return false; }
    lib_mpp = -1;
#ifdef INK_METAL_HAS_MPP_SRC
    if (cx->supports_metal4()) {
        std::string mwhy;
        lib_mpp = cx->compile_try({}, kInkMetalMppSrc, false, &mwhy);
        if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
    }
#endif
    return true;
#else
    std::string why;
    lib_codecs = cx->compile_try({kernel_dir + "/ink_metal_codecs.metal"}, "", false, &why);
    if (lib_codecs < 0) { err = "codecs MSL: " + why; return false; }
    lib_ops = cx->compile_try({kernel_dir + "/ink_metal_ops.metal"}, "", false, &why);
    if (lib_ops < 0) { err = "ops MSL: " + why; return false; }
    lib_dspark = cx->compile_try({kernel_dir + "/ink_metal_dspark.metal"}, "", false, &why);
    if (lib_dspark < 0) { err = "dspark MSL: " + why; return false; }
    lib_vis = cx->compile_try({kernel_dir + "/ink_metal_vision.metal"}, "", false, &why);
    if (lib_vis < 0) { err = "vision MSL: " + why; return false; }
    lib_mpp = -1;
    if (cx->supports_metal4()) {
        std::string mwhy;
        lib_mpp = cx->compile_try({kernel_dir + "/ink_metal_mpp.metal"}, "", false, &mwhy);
        if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
    }
    return true;
#endif
}
}  // namespace ink_metal

namespace ink2 {

class MetalInkBackend final : public InkBackend {
public:
    const char* backend_name() const noexcept override { return "metal"; }

    std::string device_info() const override {
        return cx_.device_name() + " (workSet=" +
               std::to_string((int)(cx_.recommended_working_set() / (1024ull * 1024ull * 1024ull))) + " GiB)";
    }

    BackendStats stats() const override {
        BackendStats s;
        s.total_experts = (size_t)mod_.man.c.n_phys * (size_t)mod_.man.c.n_experts;
        s.vram_experts_loaded = s.total_experts;  // Unified zero-copy Apple Silicon memory!
        s.ram_experts_loaded = 0;
        s.vram_used_bytes = cx_.allocated();
        s.vram_total_bytes = cx_.recommended_working_set();
        s.active_slots = active_slots_;
        s.max_slots = runner_.max_slots;
        s.kv_dtype = kv_dtype_;
        return s;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        kv_dtype_ = opts.kv_dtype;
        if (!mod_.load(cx_, model.path, err)) return false;
        const int par = std::max(1, std::min(64, opts.parallel));
        const int batch_cap = std::max(256, std::max(opts.chunk, par));
        if (!runner_.init(cx_, mod_, par, std::max(256, opts.max_ctx), batch_cap,
                          opts.enable_dspark, opts.vision_path, "", err)) {
            return false;
        }
        snaps_.resize((size_t)par);
        frnn_snaps_.resize((size_t)par);
        slot_used_.assign((size_t)par, 0);
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) {
            err = "metal slot out of range";
            return false;
        }
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

    void snapshot_slot(int slot) override {
        if (slot >= 0 && slot < (int)snaps_.size()) {
            runner_.snapshot_slot(slot, snaps_[slot]);
            if (runner_.use_dspark) {
                const size_t D = (size_t)mod_.man.c.hidden;
                frnn_snaps_[slot].resize(D);
                std::memcpy(frnn_snaps_[slot].data(), runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, D * 4);
            }
        }
    }

    void rollback_slot(int slot, int keep_len) override {
        if (slot < 0 || slot >= (int)snaps_.size()) return;
        // O(1) ZERO-REPLAY partial-accept rollback via ShortConv verify ring (matches ink_serve.h:690!)
        if (slot == last_ver_slot_ && keep_len > last_ver_pos0_ && keep_len <= last_ver_pos0_ + last_ver_m_) {
            runner_.restore_verify_step(slot, keep_len - last_ver_pos0_ - 1);
            return;
        }
        runner_.restore_slot(slot, snaps_[slot]);
        if (runner_.use_dspark && !frnn_snaps_[slot].empty()) {
            const size_t D = (size_t)mod_.man.c.hidden;
            std::memcpy(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, frnn_snaps_[slot].data(), D * 4);
        }
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        ink_metal::InkBatchStep st;
        st.M = batch.n_tokens;
        st.slot_ids = batch.slot_ids;
        st.pos = batch.pos;
        st.tok_ids = batch.token_ids;
        st.capture_taps = batch.capture_taps;
        // Enable O(1) verify ring when running a single-slot multi-token batch (e.g. DSpark verify M <= 8)
        if (st.M > 1 && st.M <= 8) {
            bool same_slot = true;
            for (int m = 1; m < st.M; ++m) if (st.slot_ids[m] != st.slot_ids[0]) { same_slot = false; break; }
            if (same_slot) {
                st.record_verify_ring = true;
                last_ver_slot_ = st.slot_ids[0];
                last_ver_pos0_ = st.pos[0];
                last_ver_m_ = st.M;
            }
        }
        const int D = mod_.man.c.hidden;
        bool any_img = false;
        for (int idx : batch.img_embed_idx) if (idx >= 0) { any_img = true; break; }
        if (any_img && batch.img_embeds) {
            st.is_img.assign((size_t)st.M, 0);
            st.img_embeds.assign((size_t)st.M * D, 0.0f);
            for (int m = 0; m < st.M; ++m) {
                if (batch.img_embed_idx[m] >= 0) {
                    st.is_img[m] = 1;
                    std::memcpy(st.img_embeds.data() + (size_t)m * D,
                                batch.img_embeds + (size_t)batch.img_embed_idx[m] * D,
                                (size_t)D * sizeof(float));
                }
            }
        }

        const int n_want = batch.count_logits();
        st.compute_logits = (n_want > 0 && logits_out != nullptr);
        if (st.compute_logits) {
            if (n_want == 1) {
                for (int m = 0; m < st.M; ++m) {
                    if (batch.want_logits[m]) { st.logits_last_only_idx = m; break; }
                }
                if (!runner_.forward_batch(st, logits_out, nullptr, err)) return false;
            } else {
                const int V = mod_.man.c.vocab_unpadded;
                std::vector<float> all_lg((size_t)st.M * V);
                st.logits_last_only_idx = -1;
                if (!runner_.forward_batch(st, all_lg.data(), nullptr, err)) return false;
                int dst_row = 0;
                for (int m = 0; m < st.M; ++m) {
                    if (batch.want_logits[m]) {
                        std::memcpy(logits_out + (size_t)dst_row * V,
                                    all_lg.data() + (size_t)m * V,
                                    (size_t)V * sizeof(float));
                        ++dst_row;
                    }
                }
            }
        } else {
            if (!runner_.forward_batch(st, nullptr, nullptr, err)) return false;
        }
        (void)taps_out;
        return true;
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
                     int32_t* out_draft_ids, std::string& /*err*/) override {
        if (!runner_.use_dspark || max_draft <= 0) return 0;
        const int req = std::min(7, max_draft);
        int32_t tmp[7]{};
        int n = runner_.dspark_draft_block8(slot, anchor_tok, anchor_pos, tmp, req);
        for (int i = 0; i < n; ++i) out_draft_ids[i] = tmp[i];
        return n;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (!runner_.vis.loaded || img.file.empty()) return false;
        int ntok = 0;
        return runner_.vis.encode_image_file(cx_, img.file, out_embeds, ntok, err);
    }

    bool selfcheck(std::string& err) override {
        StepBatch b;
        b.push(0, 0, 1, true);
        std::vector<float> lg((size_t)mod_.man.c.vocab_unpadded);
        runner_.reset_slot(0);
        if (!forward_batch(b, lg.data(), nullptr, err)) return false;
        for (float v : lg) {
            if (!std::isfinite(v)) { err = "metal selfcheck produced non-finite logits"; return false; }
        }
        runner_.reset_slot(0);
        return true;
    }

private:
    mtl::Context cx_;
    ink_metal::InkMetalModel mod_;
    ink_metal::InkMetalRunner runner_;
    std::vector<ink_metal::InkMetalRunner::SlotSconvSnap> snaps_;
    std::vector<std::vector<float>> frnn_snaps_;
    std::vector<uint8_t> slot_used_;
    int last_ver_slot_ = -1, last_ver_pos0_ = -1, last_ver_m_ = 0;
    int active_slots_ = 0;
    KvDtype kv_dtype_ = KvDtype::FP16;
};

std::unique_ptr<InkBackend> make_metal_backend() {
    return std::make_unique<MetalInkBackend>();
}

}  // namespace ink2
