// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/pipeline.hpp - what strata-image asks of an image model's pipeline (Qwen-Image-2.1's QiPipeline,
// Mage-Flow's MfPipeline): its VRAM budget, what an image needs, the image itself, and letting its cache go.
#pragma once

#include "strata/image/image_lora.hpp"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

/// A request stopped on its caller's word (the step callback said so).
struct QiStopped : std::runtime_error {
    QiStopped() : std::runtime_error("image engine: stopped") {}
};

/// An edit's reference image: RGBA8 [height][width][4] as read (qi_image_io).
struct QiRef {
    std::vector<uint8_t> rgba;
    int64_t width = 0, height = 0;
};

struct QiRequest {
    std::vector<int32_t> ids;   // the prompt in the model's chat form, tokenized; an edit's one <|image_pad|> per ref
    int64_t drop = 0;           // the leading ids whose states the DiT does not take (the system turn)
    int64_t width = 1024, height = 1024;   // multiples of 16
    uint64_t seed = 0;
    std::vector<float> sigmas;  // empty: the model's default (its Turbo's steps)
    int64_t steps = 0;          // with no sigmas: the sampler's steps (0: the model's own).  HunyuanImage 3.0's
                                // schedule is the shift of its own sampler; a fixed-schedule model leaves this.
    std::vector<QiRef> refs;    // an edit's reference images, in the order of their pads
    QiRef mask;                 // a mask over the base image (img_start.hpp: the init image, else refs[0]; OpenAI's:
                                // alpha 0 where the image changes); empty: none
    std::vector<float> mask_share;   // with a mask, its edit share by pixel at width x height (Crop and Stitch's,
                                     // img_crop.hpp): the sampler's mask is its mean over each latent position rather
                                     // than the mask's rule (request_latent_mask); empty: the mask's rule
    QiRef init;                 // img2img's base image for a model without edits (img_start.hpp); empty: none
    double strength = 1.0;      // how much of the base image's schedule runs (diffusers' strength, 0 .. 1]: below 1,
                                // the sampler starts from the base image's latent (img_start.hpp); 1 from noise.
                                // A double, as the server's: the steps it skips are the server's count
    std::vector<int32_t> neg_ids;   // classifier-free guidance's negative prompt, as ids (its pads as the prompt's)
    float cfg = 1.0f;               // its scale (diffusers' true_cfg_scale): above 1 with neg_ids, guidance
    float guidance = 0.0f;          // HunyuanImage 3.0's: the scale its guidance token embeds (the model's own 2.5
                                    // when 0, the number the model sees being 1000 times it).  The other models leave it.
    // partial images (OpenAI's partial_images): after these steps (counted from 1), the step's prediction of the
    // finished image, decoded in the room the VAE has beside the sampling, goes to `on_preview` (its index from 0,
    // the step, RGBA8 [height][width][4]).  A step whose decode does not fit there is left out; the final image is
    // the same either way.  HunyuanImage 3.0 makes none (its VAE needs the room of the sampling's sequence).
    std::vector<int> preview;
    std::function<void(int, int, const std::vector<uint8_t>&)> on_preview;
    // MOSS-Image's (the other models leave them): its schedule's shift (0: the model's own); the blocks guidance's
    // negative pass skips (uncond_skip_set false: the model's own, block 22); skip-layer guidance, a third pass of the
    // prompt with slg_layers skipped, slg_scale times the difference added; both active in the steps from slg_start to
    // slg_stop (shares of the schedule's steps, as the reference counts them)
    float shift = 0.0f;
    std::vector<int> uncond_skip;
    bool uncond_skip_set = false;
    std::vector<int> slg_layers;
    float slg_scale = 0.0f, slg_start = 0.0f, slg_stop = 1.0f;
    // the text encoder's states made by the chat's engine (MOSS-Image's: [rows][width], every row of the prompt's ids
    // and of the negative prompt's); empty: the image engine encodes the ids itself
    std::vector<float> te, neg_te;
    int64_t te_rows = 0, neg_te_rows = 0;
    // Anima's: the prompt's T5 ids and their weights (the LLM adapter's queries and the prompt's emphasis), and the
    // negative prompt's; the other models leave them
    std::vector<int32_t> t5_ids, neg_t5_ids;
    std::vector<float> t5_weights, neg_t5_weights;
    // the sampler: "euler" or "er_sde" (Anima's and Mage-Flow's; empty: the model's own)
    std::string sampler;
    // the adapters the DiT's products add (image_lora.hpp), each file read once with the strength its delta is scaled
    // by, applied in this order: a set that changes starts the context again (the prefix's K and V were the other
    // set's).  A model that applies none refuses the request rather than make another image.
    std::vector<LoraUse> loras;
};

struct QiTimes {
    double text = 0, sampling = 0, decode = 0;
};

/// How a request's budget was split (bytes; the report).
struct QiPlan {
    uint64_t budget = 0, fixed = 0, activations = 0, blocks = 0, layers = 0, vae = 0, prefix_kv = 0;
    int resident_blocks = 0, resident_layers = 0, resident_kv = 0;   // resident_kv: the blocks whose prefix cache is in VRAM
    int64_t mlp_rows = 0;   // 0: all
    uint64_t streamed = 0;  // HunyuanImage 3.0's: the expert banks' bytes one pass reads past what is resident
    uint64_t lora = 0;      // the adapters bound, their A and B on the GPU (part of fixed)
    int lora_n = 0;
    bool vae_tiled = false, text_cached = false;
};

/// The VRAM an image needs (bytes): `least`, the budget it runs in at all (the layers streamed, the MLP in bands, the
/// VAE in tiles); `full`, the budget with the DiT's blocks resident and the VAE in one piece (the encoder, run once a
/// prompt, streamed); `held`, what the engine holds now; `free`, the device's free VRAM; `cap`, the most the engine
/// could have (the device's memory less a margin, within STRATA_VRAM_LIMIT_MIB); `host`, what goes to pinned RAM with
/// `cap` (Qwen-Image's prefix cache past the blocks it leaves room for).
struct QiNeed {
    uint64_t least = 0, full = 0, held = 0, free = 0, cap = 0, host = 0;
};

class ImagePipeline {
public:
    virtual ~ImagePipeline() = default;
    /// The DiT's plain LoRAs folded into its blocks' weights (the default) or added to each product (false: the
    /// weights as the file has them), from the next request's adapters on.
    void set_lora_bake(bool on) { lora_bake_ = on; }
    /// The VRAM the engine may take (0: the free VRAM at open, less a margin, within STRATA_VRAM_LIMIT_MIB).
    virtual void set_budget(uint64_t bytes) = 0;
    /// The budget in force: the one set, else the free VRAM's at open.
    virtual uint64_t budget() const = 0;
    /// Frees what the engine holds on the GPU beyond its fixed weights (the cache): the next request loads again.
    /// `all`: the fixed weights and the products' scratch too (the parts are opened again on the next request).
    virtual void release(bool all = false) = 0;
    /// The budget set to what the engine holds and the VRAM free now, less `reserve` (another program's room).
    virtual void fit(uint64_t reserve) = 0;
    /// What the engine holds on the GPU now (an estimate of its own buffers).
    virtual uint64_t held() const = 0;
    /// What an image of n_text prompt ids (after the system turn) at width x height needs (from the parts' sizes:
    /// nothing put on the GPU); refs: an edit's reference images' sizes (width, height as read); mask: the edit has a
    /// mask; n_neg: guidance's negative prompt's ids after the system turn (0: no guidance).
    /// `loras`: the adapters the request applies, whose A and B stay on the GPU (the fixed weights); a model that
    /// applies none counts none.
    virtual QiNeed need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                        int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) = 0;
    /// The request `serve` warms the engine up with.  The default is the generic one (a plain token-id ramp at
    /// 256 x 256, one step), which the models whose DiT reads any ids take; a model whose sequence must carry its
    /// image block's markers (HunyuanImage 3.0) fills `r` with a sequence of its own and returns true.
    virtual bool warm_up_request(QiRequest& r) const { (void) r; return false; }

    /// The size a request leaves to the model: i of the `<img_ratio_i>` the model writes after `r.ids` (its
    /// resolution bucket), as HunyuanImage 3.0's image_size "auto" chooses it; `seconds`: what it took.  A model
    /// that does not choose its own size throws.
    virtual int choose_ratio(const QiRequest& r, double* seconds = nullptr) {
        (void) r;
        (void) seconds;
        throw std::invalid_argument("image engine: this model does not choose its own size");
    }

    /// RGBA8 [height][width][4] of the request.  `step`: the sampling's progress, false to stop (QiStopped).
    /// `stage`: each stage as it starts ("refs", "text", "sampling", "decode").
    virtual std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                          const std::function<bool(int, int)>& step = {},
                                          const std::function<void(const char*)>& stage = {}) = 0;

protected:
    bool lora_bake_ = true;
};

/// The most the engine could have: a device's memory less the driver's margin, within STRATA_VRAM_LIMIT_MIB.
uint64_t vram_cap(uint64_t total);

}  // namespace strata::image
