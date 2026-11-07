// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_pipeline.hpp - Qwen-Image-2.1's text to image on the GPU: the text encoder's states of the
// prompt's ids, the DiT's flow sampling from the seed's noise, and the VAE's image, within a VRAM budget.
//
// The budget is split per request (plan): the weights that always stay (the DiT's small ones, the VAE's), the
// products' scratch and the DiT's activations come first (the MLP in bands of rows when the whole would take too
// much); then the DiT's prefix cache (the rest of it in pinned RAM: an edit's many reference images); then the DiT's
// blocks get what remains, all of them when they fit (they are used 8 times a request), past
// the text encoder's two slots; then the text encoder's layers.  The VAE decodes in what is left, the DiT's
// activations freed when it needs them, in tiles when even that is too little.  What a request leaves in VRAM (the
// layers' resident prefixes, the activations) stays for the next one: the cache.  The same prompt's text states are
// kept too.
#pragma once

#include "strata/image/img_start.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/image/qi_vae.hpp"
#include "strata/image/qi_vision.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// The size an edit's reference image is worked at: about 1024 x 1024 pixels in the reference's shape, sides rounded to
/// multiples of 32, whatever the image's size (diffusers' QwenImage21Pipeline, its output_resolution 1024).
void qi_ref_size(int64_t ref_w, int64_t ref_h, int64_t& w, int64_t& h);

/// The token id of <|image_pad|> in Qwen3-VL's vocabulary.
constexpr int32_t kImagePad = 151655;

class QiPipeline : public ImagePipeline {
public:
    /// Reads the parts' files for their sizes; the parts go on the GPU with the first request (ensure_open).
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
              const std::string& vision = "");

    /// The VRAM the engine may take (0: the free VRAM at open, less a margin, within STRATA_VRAM_LIMIT_MIB).
    void set_budget(uint64_t bytes) override;
    /// Frees what the engine holds on the GPU beyond its fixed weights (the cache): the next request loads again.
    /// `all`: the fixed weights and the products' scratch too (the parts are opened again on the next request).
    void release(bool all = false) override;
    ~QiPipeline() override;
    /// The budget set to what the engine holds and the VRAM free now, less `reserve` (another program's room): taken
    /// before a request when the GPU's memory is shared.
    /// What the engine holds on the GPU now (an estimate of its own buffers).
    uint64_t held() const override;

    /// What an image of n_text prompt ids (after the system turn) at width x height needs (from the parts' sizes:
    /// nothing put on the GPU); refs: an edit's reference images' sizes (width, height as read); mask: the request has
    /// a base image (img_start.hpp: img2img's or a mask's), encoded at the image's size, its latent and a mask's noise
    /// and positions kept through the sampling;
    /// n_neg: guidance's negative prompt's ids after the system turn (0: no guidance), its own prefix beside the first.
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) override;

    /// RGBA8 [height][width][4] of the request.  `step`: the DiT's progress (QiDit::sample), false to stop.
    /// `stage`: each stage as it starts ("refs", the references through the vision encoder and the VAE's encoder;
    /// "text", the prompt through the text encoder; "sampling"; "decode", the VAE's decoder).
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;

private:
    QiPlan plan(int64_t T, int64_t h, int64_t w, int64_t n_ids = 0, int64_t n_neg_pre = 0);

    /// The adapters r asks for, bound to the DiT's products (set_loras): a failure throws err.  A set that changed
    /// takes the context with it, so the request's set_context comes after this (the text states themselves are the
    /// encoder's, the adapters' or not, and are kept).
    void use_loras(const QiRequest& r);

    void ensure_open();
    void te_layers();
    void edit_refs(const QiRequest& r, std::vector<int32_t>& ids, std::vector<QiVisionOut>& vouts,
                   std::vector<QiText::Image>& te_images, std::vector<QiDit::Ref>& dit_refs, std::vector<float*>& ref_z,
                   std::vector<int32_t>& neg_ids, std::vector<QiText::Image>& neg_te,
                   std::vector<QiDit::Ref>& neg_refs, bool vision = true);
    /// The base image (img_start.hpp) at the image's size through the VAE's encoder into `ls`'s base latent.
    void encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls);

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_, vision_path_;
    std::unique_ptr<QiVision> vision_;           // an edit's, opened for it and let go after it
    std::unique_ptr<QiVision> vision_size_;
    std::unique_ptr<QiText> te_;
    std::unique_ptr<QiDit> dit_;
    std::unique_ptr<QiVae> vae_;
    // the parts read for their sizes only (open, dry): what an image needs is known with nothing on the GPU
    std::unique_ptr<QiText> te_size_;
    std::unique_ptr<QiDit> dit_size_;
    std::unique_ptr<QiVae> vae_size_;
    uint64_t dit_set_ = 0, te_set_ = 0;
    uint64_t te_want_ = 0;   // the encoder's layers the plan gives them: taken when it encodes (te_layers)
    int64_t mlp_set_ = -1;
    std::vector<int32_t> last_ids_;
    std::string te_lora_;   // the text encoder's adapters the kept states were made with (lora_key_of)
    std::vector<float> last_neg_states_;   // the last negative prompt's states (guidance), for the same ids again
    std::vector<int32_t> last_neg_ids_;   // the last prompt, and its text states (host)
    std::vector<float> last_states_;
    /// The references the last states read (img_start.hpp's refs_key; empty, none): an edit's states are kept for the
    /// same prompt and the same references, as a generation's for the same prompt
    std::string last_refs_, last_neg_refs_;
    /// The last edit's request (its ids and negative ids as asked, before the pads are widened) and each reference's
    /// vision output's shape: an edit whose states are kept (the same request and references) widens its pads from
    /// these, without the vision encoder, whose output only the text encoder reads
    std::vector<int32_t> last_req_ids_, last_req_neg_;
    std::vector<QiVisionOut> last_vision_;
    /// The VAE's latents kept for the same images: the references' (the DiT's input; refs_key of them, empty none) and
    /// a mask's or img2img's base image's at the picture's size (its key and the size): an edit of the same images
    /// again encodes neither
    struct Kept {
        std::string refs, base;
        std::vector<float*> z;
        float* base_z = nullptr;
    } kept_;
    void drop_kept(bool base_too = true);
};

}  // namespace strata::image
