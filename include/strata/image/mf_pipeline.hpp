// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_pipeline.hpp - Mage-Flow's text to image on the GPU: the text encoder's states (Qwen3-VL-4B,
// after its final norm) of the prompt's ids, the DiT's flow sampling from the seed's noise, and Mage-VAE's image,
// within a VRAM budget.
//
// The budget is split per request as QiPipeline splits it: the weights that always stay (the DiT's parts outside its
// blocks, the VAE's) and the products' scratch first, then the DiT's activations (the MLP in bands of rows when the
// whole would take too much), then the DiT's blocks (used at every step) past the text encoder's two slots, then the
// encoder's layers.  The VAE decodes in what is left, the DiT's activations freed when it needs them (it is not tiled:
// its group norms take the whole latent, a few MiB at 2048 x 2048).  The same prompt's text states are kept.
//
// Edits (with the edit DiT and the vision encoder): up to three reference images (the edit models' training).  Each
// goes to the VAE's encoder at the image's size (its posterior sampled with the seed's noise) and, its long edge at
// most 384 pixels and its sides multiples of 32 as Qwen3-VL's processor takes it, to the vision encoder, whose tokens
// widen the reference's pad in the prompt (their positions plain, as the reference's text encoder takes them); the
// edit DiT takes their latents after the target's.  The DiT not in use gives its VRAM back; with keep its Q8_0 stays
// in RAM for the next switch, without it the DiT is read from its file again.
//
// img2img and a mask (img_start.hpp) take the base image through the VAE's encoder at the image's size (Mage-VAE's,
// or the Flux2 VAE's for MageTrail), the posterior's mean; with the edit DiT their base image is the first reference.
//
// The DiT's weights (q8 auto, the default): FP16 on the GPU, the matrix units taking them as they are, when its blocks
// all fit the budget beside the request's working memory; else Q8_0 (half the VRAM, and half the bytes of a streamed
// block), dequantized at each product.  The choice is made as the DiT is opened, and again when a request finds the
// FP16 blocks no longer fit (the DiT opened again as Q8_0).
#pragma once

#include "strata/image/img_start.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/core/vision.hpp"
#include "strata/image/text_encoder.hpp"
#include "strata/image/w8_choice.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// Mage-Flow's static-shift schedule (shift s) of n steps: sigma = s t / (1 + (s - 1) t) for t from 1 down to 1 / n,
/// and the terminal 0 (the reference's FlowMatchEulerDiscreteScheduler, shift 6).

class MfPipeline : public ImagePipeline {
public:
    bool takes_loras() const override { return true; }
    bool takes_guidance() const override { return true; }

    /// Reads the parts' files for their sizes; the parts go on the GPU with the first request.  `q8`: the DiT's blocks
    /// as Q8_0 (1), FP16 (0) or by the budget (-1, the default).  `edit_dit` and `vision` (the mmproj): edits (none
    /// without them); `keep`: the DiT not in use kept in RAM.
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
              int q8 = -1, const std::string& edit_dit = "", const std::string& vision = "", bool keep = true);

    void set_budget(uint64_t bytes) override;
    void release(bool all = false) override;
    uint64_t held() const override;
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) override;
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;

private:
    struct Edit;   // an edit's references as the parts take them
    QiPlan plan(int64_t hw, int n, int64_t max_text, int64_t n_ids);
    void ensure_open();
    MfDit& use_dit(bool edit, bool q8);
    /// Binds the request's adapters to the DiT's products (MfDit::set_loras); after use_dit, which may open the DiT
    /// again (a set that does not bind leaves the products as they were).
    void use_loras(const QiRequest& r);
    bool want_q8(int64_t rows, int n, int64_t max_text, int64_t n_ids) const;
    MfDit* active() const { return edit_active_ ? edit_dit_.get() : dit_.get(); }
    MfDit& dit() { return *active(); }
    void edit_refs(const QiRequest& r, Edit& e);
    void encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls);

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_, edit_path_, vision_path_;
    int q8_ = -1;
    W8Choice w8_;   // --dit-q8 auto with both forms in the budget: the faster here (MfDit::time_forms)
    bool keep_ = true, edit_active_ = false;
    bool base_encoder_ = false;   // the VAE's file has the encoder a base image takes (img2img, a mask)
    /// The text encoder's states of a prompt (its pictures' rows among the tokens), the engine's stream's work done
    /// first (the encoder runs on its own); a failure throws.
    std::vector<float> text(const std::vector<int32_t>& ids, const std::vector<TextEncoder::Picture>& pictures);
    std::unique_ptr<TextEncoder> te_;
    std::unique_ptr<MfDit> dit_, edit_dit_;
    // the closed DiTs' blocks in 8 bits (release(true)), for their next open
    std::vector<std::vector<uint8_t>> dit_host_, edit_host_;
    std::unique_ptr<core::Vision> vision_;        // the edits' vision encoder, while the budget has room for it
    std::unique_ptr<core::Vision> vision_size_;
    std::unique_ptr<MfVae> vae_;
    std::unique_ptr<TextEncoder> te_size_;
    std::unique_ptr<MfDit> dit_size_, dit16_size_;   // the DiT's sizes as Q8_0 and as FP16
    std::unique_ptr<MfVae> vae_size_;
    uint64_t te_set_ = 0;
    uint64_t te_want_ = 0;   // the encoder's layers the plan left room for, set where it encodes (te_layers)
    void te_layers();
    std::vector<float> last_states_, last_neg_states_;
};

}  // namespace strata::image
