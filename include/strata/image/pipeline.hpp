// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/pipeline.hpp - what strata-image asks of an image model's pipeline (Qwen-Image-2.1's QiPipeline,
// Mage-Flow's MfPipeline): its VRAM budget, what an image needs, the image itself, and letting its cache go.
#pragma once

#include "strata/image/guidance.hpp"
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

// MOSS-Image's edits: where an edit adapter's references sit beside the image, as the adapter was trained
// (MossDit::set_refs; ai-toolkit's training code and each adapter's model card), named after the trainers.  MossEdit:
// chinoll's Moss-Image-Edit ("moss-image-edit": the references at timestep 0, seeing only each other); AitkKv:
// ai-toolkit's with its kv_cache on ("ai-toolkit": the same, the image's rows moved past them); AitkNoKv: ai-toolkit's
// with it off ("ai-toolkit-no-kv": the references in the sequence, at timestep 0); Appended: thearkorbit's
// moss-image-edit-lora-poc ("appended": the references appended to the sequence at the image's timestep)
enum class QiRefLayout { MossEdit, AitkKv, AitkNoKv, Appended };

struct QiRequest {
    std::vector<int32_t> ids;   // the prompt in the model's chat form, tokenized; an edit's one <|image_pad|> per ref
    int64_t drop = 0;           // the leading ids whose states the DiT does not take (the system turn)
    int64_t width = 1024, height = 1024;   // multiples of 16
    uint64_t seed = 0;
    std::vector<float> sigmas;  // empty: the model's default (its Turbo's steps)
    bool base_schedule = false; // SCHED:base: Qwen-Image's base schedule (qi_base_sigmas), not the Turbo's
    int64_t steps = 0;          // with no sigmas: the sampler's steps (0: the model's own).  HunyuanImage 3.0's
                                // schedule is the shift of its own sampler; a fixed-schedule model leaves this.
    std::vector<QiRef> refs;    // an edit's reference images, in the order of their pads
    // MOSS-Image's edits: the references' layout (an ai-toolkit adapter's also has pictures: the text encoder sees
    // each at a <|image_pad|> of ids (one a picture, which the engine widens to its tokens: its sides, multiples of
    // 32, over 32 each), their tokens' rows from the vision encoder)
    QiRefLayout refs_layout = QiRefLayout::MossEdit;
    // the color match of a cut's drawing as it goes back (CropStitch::stitch): 0 none, 1 the part kept
    // around the paint (the default's: the drawing's drift only), 2 the whole cut (a Reinhard transfer)
    int crop_color = 0;
    // an upscale: the base image resized to the request's size (Lanczos) before it is drawn again (strata_image's
    // UPSCALE; the pipelines take it at that size)
    bool upscale = false;
    // its first pass by an upscaler (strata_image's UPS:<name>[:<path>], strata/upscale/upscale.hpp) rather than Lanczos
    // alone, and that pass's color correction (UPSCOLOR:, upscale.hpp's names; empty: the upscaler's own); empty: Lanczos
    std::string upscaler, upscaler_color;
    // an upscale past the model's size, drawn again a tile at a time (strata_image's TILES:): the
    // tile's side, the context around it and the blend at its edges, in pixels; 0: one drawing
    int64_t tile = 0, tile_pad = 0, tile_blur = 0;
    std::vector<QiRef> pictures;
    QiRef mask;                 // a mask over the base image (img_start.hpp: the init image, else refs[0]; OpenAI's:
                                // alpha 0 where the image changes); empty: none
    std::vector<float> mask_share;   // with a mask, its edit share by pixel at width x height (the cut's,
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
    // the sampler: "euler" or "er_sde"; empty selects the model's own
    std::string sampler;
    // the guidance added to the sampling (guidance.hpp: CFG norm, CFGZero*, APG, skip layer guidance,
    // Detail Daemon; strata_image's GUIDE words): Qwen-Image-2.1's and Mage-Flow's; the other models leave it
    GuidanceSpec guide;
    // Qwen-Image-2.1's ControlNet (qi_control.hpp; strata_image's CNET: and CTRL words): the branch's file (empty:
    // none), the control image (a preprocessed map at any size, resized to the request's; empty: the branch's control
    // channels zero, as an inpaint through the branch alone), the skips' strength and the sigmas the branch is on from
    // and to (the server's of the start and end shares of the schedule); with a mask, the kept mask and the masked base
    // image go to the branch too unless control_inpaint is false.  The other models refuse a request with one.
    std::string control_net;
    QiRef control;
    float control_strength = 1.0f, control_sigma_hi = 1.0f, control_sigma_lo = 0.0f;
    bool control_inpaint = true;
    // MOSS-Image's color and depth control (darask0's adapter: moss_dit.hpp's set_control; strata_image's MCTL:,
    // MCOLOR:, MDEPTH:, MCTLENDS: words): the adapter's file (its LoRA the request's LORA: applies), the color map and
    // the depth map (each resized to the request's size; empty: that one off), and the share of the steps each is on
    // for.  The other models refuse a request with one.
    std::string mctl;
    QiRef mctl_color, mctl_depth;
    float mctl_color_end = 1.0f, mctl_depth_end = 1.0f;
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

/// The room a plan leaves the VAE beside the DiT's activations (a partial image's, a decode's first).
inline uint64_t vae_beside(const QiPlan& p) { return p.vae > p.activations ? p.vae - p.activations : 0; }

/// Partial images (QiRequest::preview): the sampler's peek, a step's x0 decoded in the room the VAE has beside the
/// DiT's activations now (vae_beside), nothing given up for it; a step whose decode does not fit there is
/// left out.  `decode(x0, room)` decodes the latent x0 (`values` floats) into the pixels `image()` then gives as
/// RGBA8, and throws where it does not fit; `drop` lets the VAE's scratch go after it.
class PreviewDecode {
public:
    PreviewDecode(const QiRequest& r, const QiPlan& p, int64_t values, std::function<void(float*, uint64_t)> decode,
                  std::function<void()> drop, std::function<std::vector<uint8_t>()> image)
        : r_(r), p_(p), values_(values), decode_(std::move(decode)), drop_(std::move(drop)), image_(std::move(image)) {}
    PreviewDecode(const PreviewDecode&) = delete;
    PreviewDecode& operator=(const PreviewDecode&) = delete;
    ~PreviewDecode() { release(); }
    /// The sampler's peek: `done` steps, `make` writes their x0
    void operator()(int done, const std::function<void(float*)>& make);
    /// x0's buffer let go (the decode's room after the sampling)
    void release();

private:
    const QiRequest& r_;
    const QiPlan& p_;
    int64_t values_;
    std::function<void(float*, uint64_t)> decode_;
    std::function<void()> drop_;
    std::function<std::vector<uint8_t>()> image_;
    float* x0_ = nullptr;
    int shown_ = 0;
};

class ImagePipeline {
public:
    virtual ~ImagePipeline() = default;
    /// The DiT's plain LoRAs folded into its blocks' weights (the default) or added to each product (false: the
    /// weights as the file has them), from the next request's adapters on.
    void set_lora_bake(bool on) { lora_bake_ = on; }
    /// The VRAM the engine may take (0: the free VRAM at open, less a margin, within STRATA_VRAM_LIMIT_MIB).
    virtual void set_budget(uint64_t bytes) = 0;
    /// The budget in force: the one set, else the free VRAM's at open, less what the device's use past this engine's
    /// allocations has grown since then (vram_other).
    uint64_t budget() const { return less_others(budget_ > 0 ? budget_ : auto_budget_); }
    /// Frees what the engine holds on the GPU beyond its fixed weights (the cache): the next request loads again.
    /// `all`: the fixed weights and the products' scratch too (the parts are opened again on the next request).
    virtual void release(bool all = false) = 0;
    /// The budget set to what the engine holds and the VRAM free now, less `reserve` (another program's room).
    virtual void fit(uint64_t reserve);
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
    /// Request features supported by this pipeline.
    virtual bool takes_loras() const { return false; }
    virtual bool takes_guidance() const { return false; }
    virtual bool takes_control_net() const { return false; }
    virtual bool takes_moss_options() const { return false; }
    virtual const char* no_loras_message() const { return "image engine: this image model's DiT takes no adapters"; }
    /// Extra inputs used when estimating a request, reset by the caller after need().
    virtual void set_need_context(bool, const std::vector<std::pair<int64_t, int64_t>>&, QiRefLayout) {}
    /// Number of MOSS-Image color/depth maps in the request (reset after need).
    virtual void set_need_control(int) {}
    virtual std::pair<uint64_t, uint64_t> control_need(const std::string&, int64_t, int64_t) {
        throw std::invalid_argument("image engine: ControlNet is Qwen-Image-2.1's only");
    }
    /// Logits of the last size choice, if the model reports them.
    virtual const std::vector<double>& ratio_logits() const {
        static const std::vector<double> empty;
        return empty;
    }

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
    /// n's held, free and cap: what the engine holds, the device's free VRAM and vram_cap of its memory (the end of
    /// every model's need)
    void vram_into(QiNeed& n) const;
    /// The device's use past the engine's allocations taken as the budget's base (at open, and as a budget is set).
    void note_others();
    uint64_t less_others(uint64_t b) const;

    /// The budget past the plan's fixed weights; throws where the budget is below them.
    static uint64_t plan_avail(const QiPlan& p);
    /// Throws: the budget is too small for this image.
    [[noreturn]] static void plan_too_small(const QiPlan& p);
    /// The DiT's blocks given `blocks` of the budget (set where it is not what the last plan set).
    template<class Dit> void plan_blocks(Dit& d, uint64_t blocks) {
        if (blocks == dit_set_) return;
        d.set_block_budget(blocks);
        dit_set_ = blocks;
    }
    /// The DiT's MLP in the plan's bands `mlp` (a plan with image rows: `rows`), its activations let go without them
    /// (a budget change between requests: they come back on use), and the plan's resident blocks.
    template<class Dit> void plan_rows(QiPlan& p, Dit& d, bool rows, int64_t mlp) {
        p.mlp_rows = mlp;
        if (rows && mlp != mlp_set_) {
            d.set_mlp_rows(mlp);
            mlp_set_ = mlp;
        }
        if (!rows) d.drop_activations();
        p.resident_blocks = d.resident_blocks();
    }
    /// The VAE's room for a decode of `need` bytes: beside the DiT's activations where it fits there, else theirs too
    /// (they let go).  The model gives it more past that (the encoder's layers, the DiT's blocks) its own way.
    template<class Dit> static uint64_t decode_room(const QiPlan& p, Dit& d, uint64_t need) {
        const uint64_t room = vae_beside(p);
        if (need <= room) return room;
        d.drop_activations();
        return p.vae;
    }
    /// The text encoder's adapters now `key` (lora_key_of): the states kept for the last prompts are let go where
    /// they were made with others.
    void te_loras_now(const std::string& key) {
        if (key == te_lora_) return;
        last_ids_.clear();
        last_neg_ids_.clear();
        te_lora_ = key;
    }

    bool lora_bake_ = true;
    uint64_t budget_ = 0, auto_budget_ = 0;   // the budget set (0: none) and the one from the free VRAM at open
    int64_t other_base_ = 0;                  // vram_other() as the budget was set
    uint64_t dit_set_ = 0;                    // the DiT's blocks' budget the last plan set (plan_blocks)
    int64_t mlp_set_ = -1;                    // the DiT's MLP's bands the last plan set (plan_rows)
    // the last prompt and negative prompt, whose text states the model keeps, and the text encoder's adapters they
    // were made with (lora_key_of)
    std::vector<int32_t> last_ids_, last_neg_ids_;
    std::string te_lora_;
};

/// The most the engine could have: a device's memory less the driver's margin, within STRATA_VRAM_LIMIT_MIB.
uint64_t vram_cap(uint64_t total);
/// The device's free VRAM now (gpu::mem_info).
uint64_t free_vram();
/// The room this device's driver and runtime keep for themselves and the copies' staging takes: core::vram_room of
/// the memory the engine sees for the device (gpu::mem_info's total, STRATA_VRAM_LIMIT_MIB applied).  `parts`: the
/// part of it a stage keeps beside a plan that already keeps the whole room (MOSS-Image's and SeedVR2's slack).
uint64_t vram_room(uint64_t parts = 1);
/// The device's memory in use past this process's own allocations (gpu::device_held).
int64_t vram_other();

/// A plan's DiT MLP in bands of rows (4096 .. 512) while its activations take more than a third of `avail` ->
/// the rows (0: the whole MLP at once), `activations` the bytes with them; `act(rows)` the activations' bytes with
/// bands of `rows` (0: none), `any` false (no image rows): no activations.
template <class F>
int64_t mlp_bands(uint64_t avail, bool any, uint64_t& activations, const F& act) {
    activations = any ? act(0) : 0;
    int64_t mlp = 0;
    for (int64_t rows : {4096, 2048, 1024, 512}) {
        if (!any || activations * 3 <= avail) break;
        mlp = rows;
        activations = act(rows);
    }
    return mlp;
}

}  // namespace strata::image
