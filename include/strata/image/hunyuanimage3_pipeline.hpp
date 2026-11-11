// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_pipeline.hpp - HunyuanImage 3.0's text to image on the GPU: the DiT's own
// reading of the prompt's ids, its flow sampling from the seed's noise, and the VAE's image, within a VRAM budget.
//
// The model has no text encoder, so this pipeline takes two files where Qwen-Image's and Mage-Flow's take three:
// the DiT embeds the ids itself (`wte`, with the timestep, the guidance and the meanflow's next timestep written
// over three of those rows), reads them beside the image block through its MoE decoder layers, and the DCAE VAE
// decodes the latent to the picture.  `drop` is 0 (the system turn is part of what the layers read, not a prefix
// they skip), and the guidance is a meta token's value rather than a second pass: Instruct-Distil samples once.
//
// The budget is split per request the way the other pipelines split theirs: the weights that always stay (the
// DiT's dense products, the VAE's) come first, then the sequence's activations, then the experts' banks - a
// resident prefix, the rest streamed from the file's pages through two slots, since 43 GB of quantized bytes is
// no card's memory - and the VAE decodes in what the DiT leaves, its buffers given back first.  The VAE is not
// tiled: its group norms take the whole image, so an image whose decode will not fit is refused by name rather
// than degraded.  What a request leaves on the card (the resident banks, the buffers) stays for the next one.
//
// The pieces around the layers: `wte` and the five embedders still run on the host in FP32, the numbers the CPU
// golden is held to; they cost a step a few hundred milliseconds (a few hundred rows out of a BF16 table, five
// 4096-wide products over one row each).  `patch_embed` and `final_layer` are the card's (HunyuanImage3EntryExit):
// measured on the B70, the host's FP32 convolutions were 30 s of a 33 s step, the layers and the VAE the rest, and
// the card does them in 0.07 s.  `STRATA_HUNYUANIMAGE3_HOST_ENDS=patch|final|both` puts the host FP32 pieces back
// for a check (the card's products take each operand as two FP16 halves, about 2e-6 of the host's: with the
// golden's latent start and 8 steps the B70's picture holds PSNR 35.67 dB, the host's 35.78).
//
// An edit (reference images) is the model's own: each reference stands in the sequence before the prompt as the
// reference's `cond_joint_image` - `<boi> <img_size_N> <img_ratio_i> <timestep> <img> * (its VAE grid)
// <joint_img_sep> <img> * 1024 <eoi>` - whose rows are its VAE latent through `patch_embed` at timestep 0 (the
// VAE's encoder, hunyuanimage3_vae), `timestep_emb(0)`, and SigLIP2's 1024 features through the aligner
// (hunyuanimage3_vision).  The reference is read as the model's processor reads it (hunyuanimage3_refs: the VAE
// copy resized and center-cropped to the bucket the ids name, the ViT copy at its naflex size), its span attends
// to all of itself (hunyuanimage3_positions_of's key limits) and each of its two grids has 2D rope positions of its
// own.  Those rows are the sequence's constant part, computed once a request with the text rows.  The vision
// model's file (`--vision`, the model's bf16 shard with `vision_model.*`) opens the edits; without it a request
// with reference images is refused by name.
//
// A size left to the model (the reference's image_size "auto") is its own choice too: choose_ratio runs the
// sequence up to the generated block's `<boi> <img_size_N>` through the layers as text, and takes the
// `<img_ratio_i>` whose logit is the largest, through the final norm and those 37 rows of the output projection
// (`--ratio-head`, the file setup takes them into out of the bf16 shards) - the reference's img_ratio task, greedy.
// The caller then builds the sequence at that bucket; the rows it shares with the choice's are kept.
#pragma once

#include "strata/image/hunyuanimage3_dit.hpp"
#include "strata/image/hunyuanimage3_entry_exit.hpp"
#include "strata/image/hunyuanimage3_host.hpp"
#include "strata/image/hunyuanimage3_vae.hpp"
#include "strata/image/hunyuanimage3_vision.hpp"
#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// HunyuanImage 3.0 Instruct-Distil on the GPU, from its quantized safetensors and the VAE's fp16 ones: no GGUF,
/// no conversion, no text encoder.
class HunyuanImage3Pipeline : public ImagePipeline {
public:
    const char* no_loras_message() const override { return "image engine: HunyuanImage 3.0 takes no adapters"; }

    /// Reads the two files for their sizes; the parts go on the GPU with the first request.  `max_layers`: fewer
    /// than the file holds - a smaller run for a check on a small card, and not a comparison against the golden.
    /// `vision`: the file with the model's `vision_model.*` tensors (SigLIP2), which opens the edits - with it the
    /// VAE's encoder is read too; empty, reference images are refused.  False with `err` on a file that is not a
    /// quantized HunyuanImage 3.0 save, or whose shapes are not this walk's.
    /// `ratio_head`: the final norm and the ratio tokens' output rows (choose_ratio); empty, a size choice is
    /// refused.
    bool open(const std::string& dit, const std::string& vae, std::string& err, int64_t max_layers = 0,
              const std::string& vision = std::string(), const std::string& ratio_head = std::string());
    /// Whether the pipeline takes reference images (a vision file was given).
    bool edits() const { return !vision_path_.empty(); }

    void set_budget(uint64_t bytes) override;
    /// Frees the sequence's buffers and the VAE's, and lets every streamed bank's region go (the dense weights
    /// stay: the next request loads banks, not the model).  `all`: the parts themselves - the next request opens
    /// them again, which for this model means re-dequantizing its dense products.
    void release(bool all = false) override;
    uint64_t held() const override;

    /// The model's own warm-up: a 64 x 64 image of one step, the sequence its trim and positions take - the
    /// generic token ramp has no image block, so nothing of this model's graph would run.
    bool warm_up_request(QiRequest& r) const override;

    /// What an image needs, from the files' sizes with nothing on the GPU.  `n_text` is the request's ids after
    /// the system turn *including* the image block (and an edit's reference blocks), which this model's sequence
    /// carries as `<img>` tokens: the walk's sequence is exactly `n_text` tokens, so unlike the other pipelines
    /// nothing is added to it.  The image is (width x height)/16 on a side in tokens.  `refs` (their sizes): an
    /// edit's reference images, whose encoding before the walk takes the VAE encoder's and SigLIP2's buffers;
    /// `mask`: a base image (img_start.hpp: img2img's or a mask's) encoded at the image's size before the walk;
    /// `n_neg`: what this model does not take - ignored; `loras`: so too (HunyuanImage 3.0 takes no adapters, and
    /// strata_image refuses a request that asks for one).
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) override;

    /// `r.ids`: the request's sequence up to the generated block's `<boi> <img_size_N>` (the system turn and the
    /// reference images' blocks before it, as generate takes them), `r.refs` the images; width, height and the
    /// sampling's fields are not read.  Returns the bucket's index; the text and reference rows stay cached for the
    /// generate that follows.
    int choose_ratio(const QiRequest& r, double* seconds = nullptr) override;
    /// The last choice's 37 logits (a check's: `strata-image --ratio` prints them against the CPU golden's).
    const std::vector<double>& ratio_logits() const override { return ratio_logits_; }

    /// RGBA8 [height][width][4] of the request (every pixel opaque: the VAE decodes three planes).  `step`: the
    /// sampling's progress over the sigmas' steps, false to stop (QiStopped).  `stage`: "text" as the sequence is
    /// read, "sampling" for the walk, "decode" for the VAE.
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;

    /// A check's: start the sampling on this [32][token_h][token_w] FP32 latent instead of the seed's Gaussian, so
    /// the shipped path can be run from where a reference run started (`probe pipeline`, against the CPU golden).
    /// `strata-image` and serve never set it; setting it empty returns to the seed.
    void set_init_latent(const float* z, int64_t n) { init_latent_.assign(z, z + (size_t) n); }
    /// The entry and the exit, once the parts are open (`probe around` opens them on their own instead).
    const HunyuanImage3EntryExit* around() const { return around_.get(); }

private:
    /// How the budget is split for a `T`-token sequence at a `th` x `tw` latent grid.
    QiPlan plan(int64_t T, int64_t th, int64_t tw);
    /// plan's sizes alone, nothing applied: throws when the least the image takes (the weights that always stay -
    /// the vision model's and the VAE encoder's with `refs` - the two slots, the sequence's buffers, the VAE's in
    /// tiles where the whole does not fit, or `encode`, the largest reference's encode) is over the budget.  A
    /// request asks it before it reads its reference images.
    QiPlan size_plan(int64_t T, int64_t th, int64_t tw, bool refs, uint64_t encode) const;
    /// The GPU memory the largest of the first `n` blocks' reference encodes works in (their grids set).
    uint64_t refs_encode_bytes(const std::vector<HunyuanImage3Block>& blocks, size_t n) const;
    /// What a base image's encode at a tw x th token grid works in, and its latent [32][th][tw] (host) through the
    /// VAE's encoder (the posterior's mean), placed as hunyuanimage3_vae_copy places a reference.
    uint64_t base_encode_bytes(int64_t tw, int64_t th) const;
    std::vector<float> base_latent(const QiRef& base, int64_t tw, int64_t th);
    void ensure_open();
    /// The request's sequence: trimmed, its image blocks read (the reference images' grids from the buckets the ids
    /// name and the images' naflex sizes, the generated block's from the size), positioned, with the attention's
    /// key limits.  False with `err`.
    bool read_sequence(const QiRequest& r, std::vector<int32_t>& ids, int64_t& T, int64_t& start,
                       std::vector<int32_t>& y, std::vector<int32_t>& x, std::vector<int32_t>& key_lim,
                       std::vector<HunyuanImage3Block>& blocks, std::string& err) const;
    /// The reference blocks' grids (the first `r.refs.size()` of `blocks`): the VAE grid from the bucket the ids
    /// name, the ViT grid from the image's naflex size.  False with `err`.
    bool reference_grids(const QiRequest& r, std::vector<HunyuanImage3Block>& blocks, std::string& err) const;
    /// The text and reference rows of `ids[0 .. n)` into `hid`: the cached ones as far as the last request's ids
    /// agree (and its reference images are these), `wte` and reference_rows for the rest; then cached for the next.
    void text_rows(const QiRequest& r, const std::vector<int32_t>& ids, int64_t n,
                   const std::vector<HunyuanImage3Block>& blocks, int64_t hidden, float* hid, bool& cached);
    /// The reference images' rows into `hid` ([T][hidden], host): each block's VAE latent through patch_embed at
    /// timestep 0, its `<timestep>` row and its SigLIP2 rows.  Opens the vision part on first use; the encoders'
    /// buffers are given back before the walk.
    void reference_rows(const QiRequest& r, const std::vector<HunyuanImage3Block>& blocks, int64_t hidden,
                        float* hid);
    void ensure_vision();

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, vision_path_;
    std::vector<float> ratio_norm_, ratio_rows_;   // the ratio head: ln_f [hidden], the 37 rows [37][hidden]
    std::vector<double> ratio_logits_;
    std::unique_ptr<HunyuanImage3Vision> vision_, vision_size_;
    int64_t max_layers_ = 0;
    std::unique_ptr<HunyuanImage3Dit> dit_;
    std::unique_ptr<HunyuanImage3Vae> vae_;
    std::unique_ptr<HunyuanImage3EntryExit> around_;
    // the parts read for their sizes only (open, dry): what an image needs with nothing on the GPU
    std::unique_ptr<HunyuanImage3Dit> dit_size_;
    std::unique_ptr<HunyuanImage3Vae> vae_size_;
    std::unique_ptr<HunyuanImage3EntryExit> around_size_;
    uint64_t banks_set_ = 0;
    // the text rows of the last request's sequence: `wte` does not change between a request's steps, so they are
    // gathered once and copied back in, and the same ids coming again do not pay for it twice
    std::vector<int32_t> text_ids_;
    std::string refs_key_;   // the reference images the cached rows were made with (their sizes and bytes)
    std::vector<float> text_rows_;
    std::vector<float> init_latent_;
};

}  // namespace strata::image
