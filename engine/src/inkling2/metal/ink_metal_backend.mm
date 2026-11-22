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
        slot_beam_depth_.assign((size_t)par, runner_.is_apple10 ? 5 : 4);
        slot_ema_gain_.assign((size_t)par, 1.15);
        slot_cooldown_.assign((size_t)par, 0);
        slot_cd_len_.assign((size_t)par, 8);
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) {
            err = "metal slot out of range";
            return false;
        }
        if (!slot_used_[slot]) { slot_used_[slot] = 1; ++active_slots_; }
        runner_.reset_slot(slot);
        slot_beam_depth_[slot] = runner_.is_apple10 ? 5 : 4;
        slot_ema_gain_[slot] = 1.15;
        slot_cooldown_[slot] = 0;
        slot_cd_len_[slot] = 8;
        if (last_ver_slot_ == slot) last_ver_slot_ = -1;
        if (last_tree_slot_ == slot) last_tree_slot_ = -1;
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < runner_.max_slots && slot_used_[slot]) {
            slot_used_[slot] = 0;
            --active_slots_;
            runner_.reset_slot(slot);
            if (last_ver_slot_ == slot) last_ver_slot_ = -1;
            if (last_tree_slot_ == slot) last_tree_slot_ = -1;
        }
    }

    void reset_slot(int slot) override {
        runner_.reset_slot(slot);
        if (slot >= 0 && slot < (int)slot_beam_depth_.size()) {
            slot_beam_depth_[slot] = runner_.is_apple10 ? 5 : 4;
            slot_ema_gain_[slot] = 1.15;
            slot_cooldown_[slot] = 0;
            slot_cd_len_[slot] = 8;
        }
        if (last_ver_slot_ == slot) last_ver_slot_ = -1;
        if (last_tree_slot_ == slot) last_tree_slot_ = -1;
    }

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
        // O(1) ZERO-REPLAY partial-accept rollback via ShortConv verify ring (linear verify!)
        if (slot == last_ver_slot_ && keep_len > last_ver_pos0_ && keep_len <= last_ver_pos0_ + last_ver_m_) {
            runner_.restore_verify_step(slot, keep_len - last_ver_pos0_ - 1);
            return;
        }
        // O(1) ZERO-REPLAY early-stop rollback for multi-branch tree commit!
        if (slot == last_tree_slot_ && keep_len >= last_tree_anchor_ &&
            keep_len < last_tree_anchor_ + (int)last_tree_path_.size()) {
            runner_.restore_slot(slot, snaps_[slot]);
            const int kept = keep_len - last_tree_anchor_;
            if (kept > 0) {
                std::vector<int32_t> sub(last_tree_path_.begin(), last_tree_path_.begin() + kept);
                runner_.commit_tree_path(slot, last_tree_anchor_, sub);
            } else if (runner_.use_dspark && !frnn_snaps_[slot].empty()) {
                const size_t D = (size_t)mod_.man.c.hidden;
                std::memcpy(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, frnn_snaps_[slot].data(), D * 4);
            }
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
        // Auto-slice oversized multi-slot prefill batches (e.g. --bench --parallel 64 --prompt-len 64 -> M=4096)
        if (batch.n_tokens > runner_.max_batch) {
            const int V = mod_.man.c.vocab_unpadded;
            int out_row = 0;
            for (int off = 0; off < batch.n_tokens; off += runner_.max_batch) {
                const int take = std::min(runner_.max_batch, batch.n_tokens - off);
                StepBatch sub;
                sub.capture_taps = batch.capture_taps;
                sub.img_embeds = batch.img_embeds;
                int sub_want = 0;
                for (int i = 0; i < take; ++i) {
                    const int k = off + i;
                    const bool wl = batch.want_logits[k] != 0;
                    if (wl) ++sub_want;
                    sub.push(batch.slot_ids[k], batch.pos[k], batch.token_ids[k], wl, batch.img_embed_idx[k]);
                }
                float* sub_lg = (logits_out && sub_want > 0) ? (logits_out + (size_t)out_row * V) : nullptr;
                if (!forward_batch(sub, sub_lg, taps_out, err)) return false;
                out_row += sub_want;
            }
            return true;
        }
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

    int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) override {
        if (!runner_.use_dspark || !runner_.use_dspark_meff || slot < 0 || slot >= runner_.max_slots ||
            max_emit < 2 || anchor_pos <= 0 || anchor_pos + 12 >= runner_.max_ctx) {
            return 0;
        }

        // Realized-gain adaptive cooldown: on high-entropy prose where DSpark cannot beat plain M=1,
        // execute zero-logits-D2H fused GPU argmax M=1 directly (zero regression!), then probe periodically!
        if (slot_cooldown_[slot] > 0) {
            --slot_cooldown_[slot];
            ink_metal::InkBatchStep st;
            st.M = 1;
            st.slot_ids = {slot};
            st.pos = {anchor_pos};
            st.tok_ids = {anchor_tok};
            st.capture_taps = true;
            st.compute_logits = true;
            int32_t tok = 0;
            if (!runner_.forward_batch(st, nullptr, &tok, err)) return -1;
            out_emitted[0] = tok;
            if (out_drafted) *out_drafted = 0;
            last_ver_slot_ = -1;
            last_tree_slot_ = -1;
            return 1;
        }

        const double t_step0 = mtl::now_s();
        const bool is_bf16_trunk = (mod_.man.get("model.llm.layers.0.attn.wq_du.weight").fmt == ink_metal::FMT_BF16);
        const bool is_sq10 = (mod_.man.build.find("10") != std::string::npos);
        const double t_plain_ms = runner_.is_apple10 ? (is_bf16_trunk ? 17.2 : (is_sq10 ? 13.5 : 14.8)) : 41.0;

        auto update_gain = [&](int n_emit) {
            const double step_ms = std::max(1.0, (mtl::now_s() - t_step0) * 1e3);
            const double step_gain = (double)n_emit * t_plain_ms / step_ms;
            slot_ema_gain_[slot] = 0.72 * slot_ema_gain_[slot] + 0.28 * step_gain;
            if (step_gain >= 1.05) {
                slot_ema_gain_[slot] = std::max(slot_ema_gain_[slot], 1.14);
                slot_cd_len_[slot] = 8;
            } else if (slot_ema_gain_[slot] < 0.90) {
                slot_cooldown_[slot] = slot_cd_len_[slot];
                slot_cd_len_[slot] = std::min(16, slot_cd_len_[slot] + 4);
                slot_ema_gain_[slot] = 0.96;
            }
        };

        ink_metal::InkDraftTree tree;
        const int max_b = (max_emit <= 4) ? std::min(10, max_emit + 2) : 10;
        const int dcap = std::max(4, std::min(slot_beam_depth_[slot], std::min(7, max_emit - 1)));
        const int tm = runner_.dspark_draft_beam_tree(slot, anchor_tok, anchor_pos, max_b, /*dynamic_bayes=*/true, tree, dcap);
        if (tm <= 1) return 0;

        if (tree.is_linear) {
            ink_metal::InkBatchStep st;
            st.M = tm;
            st.slot_ids.assign((size_t)tm, slot);
            st.pos.resize((size_t)tm);
            for (int i = 0; i < tm; ++i) st.pos[i] = anchor_pos + i;
            st.tok_ids = tree.tok_ids;
            st.capture_taps = true;
            st.compute_logits = true;
            st.record_verify_ring = true;
            int32_t ver_ids[32]{};
            if (!runner_.forward_batch(st, nullptr, ver_ids, err)) return -1;

            const int rem = std::min(tm - 1, max_emit - 1);
            int acc = 0;
            for (int i = 0; i < rem; ++i) {
                if (ver_ids[i] == tree.tok_ids[i + 1]) {
                    out_emitted[acc++] = ver_ids[i];
                } else {
                    break;
                }
            }
            out_emitted[acc] = ver_ids[acc];
            if (acc < tm - 1) runner_.restore_verify_step(slot, acc);

            last_ver_slot_ = slot;
            last_ver_pos0_ = anchor_pos;
            last_ver_m_ = tm;
            last_tree_slot_ = -1;
            slot_beam_depth_[slot] = std::max(4, std::min(7, acc + 3));
            if (out_drafted) *out_drafted = rem;
            update_gain(1 + acc);
            return 1 + acc;
        }

        // Multi-branch W=4 Beam Tree verify + O(1) GPU commit (exact 0.00000 KLD, zero logits D2H!)
        snapshot_slot(slot);
        ink_metal::InkBatchStep st;
        st.M = tm;
        st.slot_ids.assign((size_t)tm, slot);
        st.pos.resize((size_t)tm);
        int max_d = 1;
        for (int i = 0; i < tm; ++i) {
            st.pos[i] = anchor_pos + tree.depths[i];
            if (tree.depths[i] > max_d) max_d = tree.depths[i];
        }
        st.tok_ids = tree.tok_ids;
        st.capture_taps = true;
        st.compute_logits = true;
        st.is_tree = true;
        st.tree_anchor_pos = anchor_pos;
        st.tree_parent = tree.parents;
        st.tree_anc_mask = tree.anc_masks;
        int32_t ver_ids[32]{};
        if (!runner_.forward_batch(st, nullptr, ver_ids, err)) return -1;

        std::vector<int32_t> path;
        path.reserve(8);
        path.push_back(0);
        int cur_node = 0;
        int acc = 0;
        while (acc + 1 < max_emit) {
            const int32_t tn = ver_ids[cur_node];
            int mc = -1;
            for (int ci = cur_node + 1; ci < tm; ++ci) {
                if (tree.parents[ci] == cur_node && tree.tok_ids[ci] == tn) { mc = ci; break; }
            }
            if (mc < 0) break;
            out_emitted[acc++] = tn;
            path.push_back(mc);
            cur_node = mc;
        }
        out_emitted[acc] = ver_ids[cur_node];
        runner_.commit_tree_path(slot, anchor_pos, path);

        last_tree_slot_ = slot;
        last_tree_anchor_ = anchor_pos;
        last_tree_path_ = std::move(path);
        last_ver_slot_ = -1;
        slot_beam_depth_[slot] = std::max(4, std::min(7, acc + 3));
        if (out_drafted) *out_drafted = max_d;
        update_gain(1 + acc);
        return 1 + acc;
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
    std::vector<int> slot_beam_depth_;
    std::vector<double> slot_ema_gain_;
    std::vector<int> slot_cooldown_;
    std::vector<int> slot_cd_len_;
    int last_ver_slot_ = -1, last_ver_pos0_ = -1, last_ver_m_ = 0;
    int last_tree_slot_ = -1, last_tree_anchor_ = -1;
    std::vector<int32_t> last_tree_path_;
    int active_slots_ = 0;
    KvDtype kv_dtype_ = KvDtype::FP16;
};

std::unique_ptr<InkBackend> make_metal_backend() {
    return std::make_unique<MetalInkBackend>();
}

}  // namespace ink2
