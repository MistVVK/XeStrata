// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_pipeline.cpp - see include/strata/image/hunyuanimage3_pipeline.hpp.
//
// The walk of one image, in the order the reference runs it
// (records/2026-10-23-hunyuanimage3-plan/CONFORM-2026-10-24.md, and `golden-cpu/` for the numbers every step here
// is held to):
//
//   ids -> trim at the last `<img>` -> the block's 2D rope positions
//   wte(text rows), and each step: the timestep/guidance/timestep_r rows over them, and patch_embed(latent) for
//   the block
//   32 MoE decoder layers over the whole sequence                (HunyuanImage3Dit, on the card)
//   final_layer(the block's rows, time_embed_2(t)) -> the velocity, and x += (sigma_next - sigma) * v
//   the VAE's decode of the last latent                          (HunyuanImage3Vae, on the card)
//   [-1,1] -> [0,1] -> RGBA8
#include "strata/image/hunyuanimage3_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_host.hpp"
#include "strata/image/hunyuanimage3_refs.hpp"
#include "strata/image/img_start.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

// The Instruct-Distil's own sampling: 8 steps at shift 3, its guidance 2.5 (the golden's manifest agrees, and
// `probe sigmas` holds the schedule to the reference's FlowMatchDiscreteScheduler).  A request that gives sigmas
// or a guidance of its own overrides these; the steps the progress reports are the sigmas' less the terminal one.
constexpr int64_t kDistilSteps = 8;
constexpr float kDistilShift = 3.0f, kDistilGuidance = 2.5f;

// The latent's channels and the VAE's spatial downsample: an image of W x H is W/16 x H/16 tokens of 32 channels.
constexpr int64_t kLatentChannels = 32, kDownsample = 16;

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

// RGBA8 [th][tw][4] of `f` as hunyuanimage3_vae_copy places it (the aspect kept, the center cropped), nearest: the
// base image's kept pixels and a mask over it, where the VAE's copy is
std::vector<uint8_t> crop_rgba8(const QiRef& f, int64_t tw, int64_t th) {
    const double tr = (double) th / (double) tw, r = (double) f.height / (double) f.width;
    int64_t rw = 0, rh = 0;
    if (r < tr) {
        rh = th;
        rw = (int64_t) std::nearbyint((double) th / (double) f.height * (double) f.width);
    } else {
        rw = tw;
        rh = (int64_t) std::nearbyint((double) tw / (double) f.width * (double) f.height);
    }
    rw = std::max(rw, tw);
    rh = std::max(rh, th);
    const int64_t top = (int64_t) std::nearbyint((double) (rh - th) / 2.0);
    const int64_t left = (int64_t) std::nearbyint((double) (rw - tw) / 2.0);
    std::vector<uint8_t> out((size_t) (tw * th * 4));
    for (int64_t y = 0; y < th; ++y)
        for (int64_t x = 0; x < tw; ++x)
            std::copy_n(f.rgba.data() + (((y + top) * f.height / rh) * f.width + (x + left) * f.width / rw) * 4, 4,
                        out.data() + (y * tw + x) * 4);
    return out;
}

}  // namespace

bool HunyuanImage3Pipeline::open(const std::string& dit, const std::string& vae, std::string& err,
                                 int64_t max_layers, const std::string& vision, const std::string& ratio_head) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    vision_path_ = vision;
    max_layers_ = max_layers;
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    dit_size_ = std::make_unique<HunyuanImage3Dit>();
    vae_size_ = std::make_unique<HunyuanImage3Vae>();
    around_size_ = std::make_unique<HunyuanImage3EntryExit>();
    if (!(dit_size_->open(dit, s_, err, max_layers, true) && vae_size_->open(vae, s_, err, true, edits()) &&
          around_size_->open(dit, s_, err, true)))
        return false;
    if (edits()) {
        vision_size_ = std::make_unique<HunyuanImage3Vision>();
        if (!vision_size_->open(vision, dit, s_, err, true)) return false;
    }
    if (!ratio_head.empty()) {   // 300 KB: read whole now, so a wrong file is told at the start
        try {
            SafeTensors st(ratio_head);
            const int64_t hidden = dit_size_->dims().hidden;
            ratio_norm_ = st.f32("model.ln_f.weight");
            ratio_rows_ = st.f32("lm_head.img_ratio.weight");
            if ((int64_t) ratio_norm_.size() != hidden ||
                (int64_t) ratio_rows_.size() != (int64_t) kHunyuanImage3RatioBuckets * hidden) {
                err = ratio_head + ": not the final norm and the " + std::to_string(kHunyuanImage3RatioBuckets) +
                      " ratio rows of a " + std::to_string(hidden) + "-wide model";
                return false;
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    auto_budget_ = vram_cap(free);
    note_others();
    return true;
}

void HunyuanImage3Pipeline::ensure_open() {
    if (dit_ && vae_ && around_) return;
    std::string err;
    dit_ = std::make_unique<HunyuanImage3Dit>();
    vae_ = std::make_unique<HunyuanImage3Vae>();
    around_ = std::make_unique<HunyuanImage3EntryExit>();
    if (!dit_->open(dit_path_, s_, err, max_layers_)) {
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err);
    }
    if (!vae_->open(vae_path_, s_, err, false)) {   // the encoder comes with an edit's first reference (ensure_vision)
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err);
    }
    if (!around_->open(dit_path_, s_, err) || around_->hidden() != dit_->dims().hidden) {
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err.empty() ? "image engine: the entry and exit's hidden width is not the decoder's"
                                             : err);
    }
    dit_->set_bank_budget(dit_->bank_min());   // nothing resident until a request plans its budget
    banks_set_ = dit_->bank_min();
    text_ids_.clear();
    text_rows_.clear();
}

void HunyuanImage3Pipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    note_others();
    if (dit_ == nullptr) return;   // released: the next request plans with it
    const QiPlan p = plan(dit_->tokens(), 0, 0);
    (void) p;   // a shrink takes effect at once; a grown budget is taken at the next request
}

void HunyuanImage3Pipeline::release(bool all) {
    if (dit_ == nullptr && vae_ == nullptr) return;
    if (all) {   // the parts go with it: the next request opens them again, dense products re-dequantized
        if (dit_) dit_->drop_sequence();
        if (vae_) vae_->drop_scratch();
        if (around_) around_->drop_grid();
        if (s_ != nullptr) gpu::stream_sync(s_);
        dit_.reset();
        vae_.reset();
        around_.reset();
        vision_.reset();
        banks_set_ = 0;
        text_ids_.clear();
        refs_key_.clear();
        text_rows_.clear();
        return;
    }
    dit_->drop_sequence();
    vae_->drop_scratch();
    if (around_) around_->drop_grid();
    if (vision_) vision_->drop_scratch();
    dit_->set_bank_budget(dit_->bank_min());   // the streamed regions stay, the resident prefix gives its VRAM back
    dit_->drop_bank_homes();                   // and the streamed banks' copies their RAM
    banks_set_ = dit_->bank_min();
}

uint64_t HunyuanImage3Pipeline::held() const {
    if (dit_ == nullptr || vae_ == nullptr) return 0;
    return dit_->held() + vae_->weight_bytes() + vae_->scratch_held() + (around_ ? around_->held() : 0) +
           (vision_ ? vision_->held() : 0);
}

void HunyuanImage3Pipeline::ensure_vision() {
    if (vision_) return;
    if (!edits()) throw std::invalid_argument("image engine: HunyuanImage 3.0's edits need the vision model's file "
                                              "(--vision): run setup with --images again");
    std::string err;
    auto v = std::make_unique<HunyuanImage3Vision>();
    if (!v->open(vision_path_, dit_path_, s_, err)) throw std::runtime_error("image engine: " + err);
    vision_ = std::move(v);
    if (!vae_->open_encoder(vae_path_, err)) throw std::runtime_error("image engine: " + err);
}

// The reference images' rows, in the sequence's constant part: for each reference block, the image's VAE copy
// (the bucket the block names) through the encoder and `patch_embed` at timestep 0 into its VAE tokens, its
// `<timestep>` row as timestep_emb(0), and its ViT copy through SigLIP2 and the aligner into its 1024 slots - the
// reference's `instantiate_vae_image_tokens` and `instantiate_vit_image_tokens`.  The encoders' buffers are given
// back here: the walk's banks and the VAE's decode take the card after.
void HunyuanImage3Pipeline::reference_rows(const QiRequest& r, const std::vector<HunyuanImage3Block>& blocks,
                                           int64_t hidden, float* hid) {
    ensure_vision();
    std::string err;
    for (size_t k = 0; k < r.refs.size(); ++k) {
        const HunyuanImage3Block& b = blocks[k];
        const QiRef& f = r.refs[k];
        const int64_t H = b.vae_h * kDownsample, W = b.vae_w * kDownsample;
        const std::vector<float> img = hunyuanimage3_vae_copy(f, W, H);
        // the posterior sampled as the reference's generator does, with noise of the seed's own: each reference its
        // stream, apart from the target's (Mage-Flow's references take the same)
        std::vector<float> z((size_t) (kLatentChannels * b.vae_n));
        const auto n = philox_randn(r.seed ^ (0x9E3779B97F4A7C15ull * (k + 1)), (uint64_t) z.size());
        // the resident banks give way to the encode where it does not fit beside them (size_plan made sure the
        // card has room for it)
        const uint64_t have = held();
        if (refs_encode_bytes(blocks, r.refs.size()) > (budget() > have ? budget() - have : 0) &&
            dit_->bank_vram() > dit_->bank_min()) {
            dit_->set_bank_budget(dit_->bank_min());
            banks_set_ = dit_->bank_min();
        }
        {
            float *d_img = nullptr, *d_z = nullptr, *d_eps = nullptr;
            if (!gpu::alloc_device(&d_img, img.size() * sizeof(float)) ||
                !gpu::alloc_device(&d_z, z.size() * sizeof(float)) ||
                !gpu::alloc_device(&d_eps, z.size() * sizeof(float))) {
                if (d_img != nullptr) gpu::free(d_img);
                if (d_z != nullptr) gpu::free(d_z);
                throw std::runtime_error(std::string("image engine: GPU memory for a reference image: ") +
                                         gpu::last_error());
            }
            try {
                gpu::copy(d_img, img.data(), img.size() * sizeof(float));
                gpu::copy(d_eps, n.data(), n.size() * sizeof(float));
                vae_->encode(d_img, H, W, d_z, d_eps);
                gpu::copy(z.data(), d_z, z.size() * sizeof(float));
            } catch (...) {
                gpu::free(d_img);
                gpu::free(d_z);
                gpu::free(d_eps);
                vae_->drop_scratch();
                throw;
            }
            gpu::free(d_img);
            gpu::free(d_z);
            gpu::free(d_eps);
            vae_->drop_scratch();
        }
        if (!around_->set_grid(b.vae_h, b.vae_w, err) ||
            !around_->patch_embed(z.data(), b.vae_h, b.vae_w, 0.0f, hid + b.vae_start * hidden, err))
            throw std::runtime_error("image engine: a reference image's entry: " + err);
        around_->drop_grid();
        if (!hunyuanimage3_time_host(*dit_->file(), "timestep_emb", 0.0f, hid + b.timestep * hidden, err))
            throw std::runtime_error("image engine: " + err);
        int64_t ph = 0, pw = 0;
        const std::vector<float> patches =
            hunyuanimage3_vit_patches(f, vision_->patch(), vision_->max_patches(), ph, pw);
        if (ph != b.vit_h || pw != b.vit_w || b.vit_n != vision_->max_patches())
            throw std::logic_error("image engine: a reference image's ViT grid is not the sequence's");
        if (!vision_->rows(patches.data(), ph, pw, hid + b.vit_start * hidden, err))
            throw std::runtime_error("image engine: " + err);
    }
    vision_->drop_scratch();
}

bool HunyuanImage3Pipeline::warm_up_request(QiRequest& r) const {
    // the smallest sequence the model's own markers make: `<boi>`, the size and ratio tokens, the three the step
    // writes over, and a 4 x 4 image block (64 x 64 pixels).  Nothing here is a picture; it is the graph's build.
    r.ids.clear();
    r.ids.push_back(kHunyuanImage3Boi);
    r.ids.push_back(kHunyuanImage3ImgSize256);   // the block is 4 x 4; the size token is the smallest the model has
    r.ids.push_back(kHunyuanImage3ImgRatio0);
    r.ids.push_back(kHunyuanImage3Timestep);
    r.ids.push_back(kHunyuanImage3Guidance);
    r.ids.push_back(kHunyuanImage3TimestepR);
    for (int64_t i = 0; i < (int64_t) (4 * 4); ++i) r.ids.push_back(kHunyuanImage3Image);
    r.drop = 0;
    r.width = r.height = 64;
    r.seed = 0;
    r.sigmas = {1.0f, 0.0f};   // one step
    r.guidance = 0.0f;
    return true;
}

bool HunyuanImage3Pipeline::read_sequence(const QiRequest& r, std::vector<int32_t>& ids, int64_t& T, int64_t& start,
                                          std::vector<int32_t>& y, std::vector<int32_t>& x,
                                          std::vector<int32_t>& key_lim, std::vector<HunyuanImage3Block>& blocks,
                                          std::string& err) const {
    if (r.width % kDownsample != 0 || r.height % kDownsample != 0 || r.width <= 0 || r.height <= 0) {
        err = "image engine: the size must be a positive multiple of 16";
        return false;
    }
    if (r.ids.empty()) {
        err = "image engine: the prompt is empty";
        return false;
    }
    ids = r.ids;
    const int64_t keep = hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) return false;
    ids.resize((size_t) keep);
    T = (int64_t) ids.size();
    if (!hunyuanimage3_blocks(ids.data(), T, blocks, err)) return false;
    const size_t n_ref = blocks.size() - 1;
    if (n_ref != r.refs.size()) {
        err = "image engine: the ids hold " + std::to_string(n_ref) + " reference image block(s) and the request " +
              std::to_string(r.refs.size()) + " reference image(s)";
        return false;
    }
    if (n_ref > 0 && !edits()) {
        err = "image engine: HunyuanImage 3.0's edits need the vision model's file (--vision): run setup with "
              "--images again";
        return false;
    }
    HunyuanImage3Block& g = blocks.back();
    g.vae_h = r.height / kDownsample;
    g.vae_w = r.width / kDownsample;
    if (g.vae_n != g.vae_h * g.vae_w) {
        err = "image engine: the prompt's image block is not the " + std::to_string(g.vae_h * g.vae_w) +
              " tokens of a " + std::to_string(r.width) + " x " + std::to_string(r.height) + " image";
        return false;
    }
    if (!reference_grids(r, blocks, err)) return false;
    start = g.vae_start;
    if (!hunyuanimage3_positions_of(T, blocks, y, x, key_lim, err)) return false;
    if (start <= 0) {
        err = "image engine: the sequence holds no text before the image block";
        return false;
    }
    return true;
}

bool HunyuanImage3Pipeline::reference_grids(const QiRequest& r, std::vector<HunyuanImage3Block>& blocks,
                                            std::string& err) const {
    for (size_t k = 0; k < r.refs.size(); ++k) {
        HunyuanImage3Block& b = blocks[k];
        const QiRef& f = r.refs[k];
        if (f.width <= 0 || f.height <= 0 || f.rgba.size() != (size_t) (f.width * f.height * 4)) {
            err = "image engine: reference image " + std::to_string(k + 1) + " is empty";
            return false;
        }
        const std::vector<HunyuanImage3Bucket> table = hunyuanimage3_buckets(b.base_size);
        if (b.ratio_index < 0 || (size_t) b.ratio_index >= table.size()) {
            err = "image engine: reference image " + std::to_string(k + 1) + "'s block names ratio " +
                  std::to_string(b.ratio_index) + ", which the " + std::to_string(b.base_size) + " base has not";
            return false;
        }
        b.vae_h = table[(size_t) b.ratio_index].height / kDownsample;
        b.vae_w = table[(size_t) b.ratio_index].width / kDownsample;
        if (b.vae_n != b.vae_h * b.vae_w) {
            err = "image engine: reference image " + std::to_string(k + 1) + "'s block has " +
                  std::to_string(b.vae_n) + " VAE tokens, not the bucket's " + std::to_string(b.vae_h * b.vae_w);
            return false;
        }
        const int64_t patch = vision_size_->patch(), slots = vision_size_->max_patches();
        int64_t rw = 0, rh = 0;
        hunyuanimage3_naflex_size(f.width, f.height, patch, slots, rw, rh);
        b.vit_h = rh / patch;
        b.vit_w = rw / patch;
        if (b.vit_n != slots) {
            err = "image engine: reference image " + std::to_string(k + 1) + "'s block has " +
                  std::to_string(b.vit_n) + " ViT tokens, not the model's " + std::to_string(slots);
            return false;
        }
    }
    return true;
}

QiNeed HunyuanImage3Pipeline::need(int64_t n_text, int64_t width, int64_t height,
                                   const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t,
                                   const std::vector<LoraUse>&) {
    const int64_t th = height / kDownsample, tw = width / kDownsample;
    if (th <= 0 || tw <= 0 || n_text < th * tw + 1) {
        throw std::invalid_argument("image engine: the ids do not hold the image block and some text");
    }
    // the VAE's decoder alone for a picture; its encoder (and the vision model) with an edit
    uint64_t fixed = dit_size_->dense_bytes() + vae_size_->weight_bytes() - vae_size_->encoder_bytes() +
                     around_size_->weight_bytes();
    const uint64_t act = dit_size_->activation_bytes(n_text) + around_size_->scratch_bytes(th, tw);
    // the VAE's decode (and an edit's encode) in the reference's tiles where the whole does not fit: the least is
    // the tiled one's, the full the whole one's
    const uint64_t vae_tiled = vae_size_->tiled_scratch_bytes();
    uint64_t vae = vae_size_->scratch_bytes(th, tw);
    uint64_t vae_least = std::min(vae, vae_tiled);
    if (!refs.empty() && vision_size_) {
        // an edit: the vision model stays with the weights; each reference's encoding (the VAE encoder's buffers
        // at the bucket nearest its shape, SigLIP2's) comes and goes before the walk, so it counts as the decode
        // does - the larger of the two workers
        fixed += vision_size_->weight_bytes() + vae_size_->encoder_bytes();
        const std::vector<HunyuanImage3Bucket> table = hunyuanimage3_buckets(1024);
        for (const auto& wh : refs) {
            const HunyuanImage3Bucket& b = hunyuanimage3_nearest_bucket(table, std::max<int64_t>(wh.first, 1),
                                                                        std::max<int64_t>(wh.second, 1));
            const uint64_t ref = around_size_->scratch_bytes(b.height / kDownsample, b.width / kDownsample) +
                                 vision_size_->scratch_bytes();
            const uint64_t enc = vae_size_->encode_scratch_bytes(b.height, b.width);
            vae = std::max(vae, enc + ref);
            vae_least = std::max(vae_least, enc + ref);   // the encode is never tiled (hunyuanimage3_vae.hpp)
        }
    }
    if (mask) {   // a base image (img_start.hpp): its encode at the image's size, before the walk as a reference's
        if (refs.empty() || !vision_size_) fixed += vae_size_->encoder_bytes();
        vae = std::max(vae, base_encode_bytes(tw, th));
        vae_least = std::max(vae_least, base_encode_bytes(tw, th));
    }
    QiNeed n;
    // least: every bank streamed (two slots of the largest), the DiT's sequence buffers and the VAE's taken in
    // turn - the decode runs after the walk gives them back.  full: every bank resident as well, both buffers kept.
    n.least = fixed + dit_size_->bank_min() + std::max(act, vae_least);
    n.full = fixed + dit_size_->bank_bytes() + act + vae;
    vram_into(n);
    const uint64_t room = n.cap > fixed + std::max(act, vae) ? n.cap - fixed - std::max(act, vae) : 0;
    n.host = dit_size_->bank_bytes() - std::min(dit_size_->bank_bytes(), room);
    return n;
}

QiPlan HunyuanImage3Pipeline::plan(int64_t T, int64_t th, int64_t tw) {
    QiPlan p = size_plan(T, th, tw, vision_ != nullptr, 0);
    p.resident_blocks = 0;   // this model's streamed units are its layers' banks
    p.layers = dit_->bank_bytes();
    p.resident_layers = 0;   // set when the budget is applied
    p.streamed = 0;          // as above: the bytes a pass reads for the banks that are not resident
    if (p.blocks != banks_set_) {
        dit_->set_bank_budget(p.blocks);
        banks_set_ = p.blocks;
    }
    p.resident_layers = dit_->resident_banks();
    // the banks past the resident prefix, by their sizes: bank_vram() counts a resident bank only once it is
    // loaded, which the walk does, not this plan
    for (int64_t l = p.resident_layers; l < dit_->layers(); ++l) p.streamed += dit_->bank_bytes(l);
    return p;
}

uint64_t HunyuanImage3Pipeline::base_encode_bytes(int64_t tw, int64_t th) const {
    const int64_t H = th * kDownsample, W = tw * kDownsample;
    return vae_size_->encode_scratch_bytes(H, W) + (uint64_t) (3 * H * W + kLatentChannels * th * tw) * sizeof(float);
}

std::vector<float> HunyuanImage3Pipeline::base_latent(const QiRef& base, int64_t tw, int64_t th) {
    std::string err;
    if (!vae_->has_encoder() && !vae_->open_encoder(vae_path_, err)) throw std::runtime_error("image engine: " + err);
    const int64_t H = th * kDownsample, W = tw * kDownsample;
    const std::vector<float> img = hunyuanimage3_vae_copy(base, W, H);
    std::vector<float> z((size_t) (kLatentChannels * th * tw));
    // the resident banks give way to the encode where it does not fit beside them (as a reference's)
    const uint64_t have = held();
    if (base_encode_bytes(tw, th) > (budget() > have ? budget() - have : 0) && dit_->bank_vram() > dit_->bank_min()) {
        dit_->set_bank_budget(dit_->bank_min());
        banks_set_ = dit_->bank_min();
    }
    float *d_img = nullptr, *d_z = nullptr;
    if (!gpu::alloc_device(&d_img, img.size() * sizeof(float)) || !gpu::alloc_device(&d_z, z.size() * sizeof(float))) {
        if (d_img != nullptr) gpu::free(d_img);
        throw std::runtime_error(std::string("image engine: GPU memory for the base image: ") + gpu::last_error());
    }
    try {
        gpu::copy(d_img, img.data(), img.size() * sizeof(float));
        vae_->encode(d_img, H, W, d_z, nullptr);   // the posterior's mean
        gpu::copy(z.data(), d_z, z.size() * sizeof(float));
    } catch (...) {
        gpu::free(d_img);
        gpu::free(d_z);
        vae_->drop_scratch();
        throw;
    }
    gpu::free(d_img);
    gpu::free(d_z);
    vae_->drop_scratch();
    return z;
}

uint64_t HunyuanImage3Pipeline::refs_encode_bytes(const std::vector<HunyuanImage3Block>& blocks, size_t n) const {
    uint64_t b = 0;
    for (size_t k = 0; k < n; ++k) {
        const int64_t H = blocks[k].vae_h * kDownsample, W = blocks[k].vae_w * kDownsample;
        b = std::max(b, vae_size_->encode_scratch_bytes(H, W) +
                            (uint64_t) (3 * H * W + 2 * kLatentChannels * blocks[k].vae_n) * sizeof(float));
    }
    return b;
}

QiPlan HunyuanImage3Pipeline::size_plan(int64_t T, int64_t th, int64_t tw, bool refs, uint64_t encode) const {
    QiPlan p;
    p.budget = budget();
    const HunyuanImage3Vision* vision = vision_ ? vision_.get() : vision_size_.get();
    // the VAE's encoder and the vision model are an edit's: put on the card with its first reference image
    p.fixed = dit_->dense_bytes() + vae_->weight_bytes() + around_->weight_bytes() +
              (refs && vision != nullptr ? vision->weight_bytes() : 0) +
              (refs && !vae_->has_encoder() ? vae_size_->encoder_bytes() : 0);
    if (p.budget <= p.fixed)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is below the weights that always stay (" +
                                 std::to_string(p.fixed >> 20) + " MiB)");
    const uint64_t avail = p.budget - p.fixed;
    // the DiT's buffers for the sequence, and the VAE's for the picture: one at a time (generate frees the
    // sequence's before the decode), so the plan keeps the larger of the two out of the banks' way
    p.activations = T > 0 ? dit_->activation_bytes(T) : 0;
    p.activations += around_->scratch_bytes(th, tw);   // the entry and the exit's grid, alive over the same walk
    p.vae = th > 0 ? vae_->scratch_bytes(th, tw) : 0;
    // the decode in the reference's tiles where the whole one does not fit beside the slots
    p.vae_tiled = th > 0 && p.vae + dit_->bank_min() > avail;
    const uint64_t work = std::max(p.activations, p.vae_tiled ? std::min(p.vae, vae_->tiled_scratch_bytes()) : p.vae);
    // an edit's reference images are encoded before the walk, beside the two slots (the banks give back the rest):
    // the larger of the two
    const uint64_t least = std::max(work, encode) + dit_->bank_min();
    if (avail < least)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image: its walk needs " + std::to_string(least >> 20) +
                                 " MiB beside the weights");
    p.blocks = clampu(avail - work, dit_->bank_min(), dit_->bank_bytes());
    return p;
}

void HunyuanImage3Pipeline::text_rows(const QiRequest& r, const std::vector<int32_t>& ids, int64_t n,
                                      const std::vector<HunyuanImage3Block>& blocks, int64_t hidden, float* hid,
                                      bool& cached) {
    // the cached rows as far as the ids agree: a size choice's sequence is the generate's up to its block's
    // `<img_size_N>`, so the generate that follows it reads its reference images' rows from the cache
    const std::string key = refs_key(r.refs);
    int64_t have = 0;
    if (key == refs_key_ && (int64_t) text_ids_.size() <= n &&
        std::equal(text_ids_.begin(), text_ids_.end(), ids.begin()))
        have = (int64_t) text_ids_.size();
    // the references' rows come whole or not at all (reference_rows writes every block's)
    const int64_t refs_end = r.refs.empty() ? 0 : blocks[r.refs.size() - 1].end;
    if (have < refs_end) have = 0;
    std::copy(text_rows_.begin(), text_rows_.begin() + have * hidden, hid);
    cached = have == n;
    if (have < n) {
        std::string err;
        if (!hunyuanimage3_wte_host(*dit_->file(), ids.data() + have, n - have, hid + have * hidden, err))
            throw std::runtime_error("image engine: " + err);
        if (have == 0 && !r.refs.empty()) reference_rows(r, blocks, hidden, hid);
    }
    text_ids_.assign(ids.begin(), ids.begin() + n);
    refs_key_ = key;
    text_rows_.assign(hid, hid + n * hidden);
}

int HunyuanImage3Pipeline::choose_ratio(const QiRequest& r, double* seconds) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    if (ratio_rows_.empty())
        throw std::invalid_argument("image engine: HunyuanImage 3.0 chooses its size with the ratio head's file "
                                    "(--ratio-head): run setup with --images again");
    if (r.drop != 0)
        throw std::invalid_argument("image engine: HunyuanImage 3.0 reads its whole sequence: drop must be 0");
    const std::vector<int32_t>& ids = r.ids;
    const int64_t T = (int64_t) ids.size();
    if (T < 3 || ids[(size_t) (T - 2)] != kHunyuanImage3Boi || ids[(size_t) (T - 1)] < kHunyuanImage3ImgSize256 ||
        ids[(size_t) (T - 1)] >= kHunyuanImage3ImgRatio0)
        throw std::invalid_argument("image engine: a size choice's ids end with the image's <boi> <img_size_N>");
    std::string err;
    std::vector<HunyuanImage3Block> blocks;
    if (!hunyuanimage3_blocks(ids.data(), T - 2, blocks, err, false)) throw std::invalid_argument(err);
    for (const HunyuanImage3Block& b : blocks)
        if (!b.reference) throw std::invalid_argument("image engine: a size choice's ids hold an image block");
    if (blocks.size() != r.refs.size())
        throw std::invalid_argument("image engine: the ids hold " + std::to_string(blocks.size()) +
                                    " reference image block(s) and the request " + std::to_string(r.refs.size()) +
                                    " reference image(s)");
    if (!blocks.empty() && !edits())
        throw std::invalid_argument("image engine: HunyuanImage 3.0's edits need the vision model's file "
                                    "(--vision): run setup with --images again");
    std::vector<int32_t> y, x, key_lim;
    if (!reference_grids(r, blocks, err) || !hunyuanimage3_positions_of(T, blocks, y, x, key_lim, err))
        throw std::invalid_argument(err);
    ensure_open();
    size_plan(T, 0, 0, !blocks.empty(), refs_encode_bytes(blocks, blocks.size()));   // refused before the refs' reading
    const int64_t hidden = dit_->dims().hidden;
    std::vector<float> hid((size_t) (T * hidden));
    bool cached = false;
    text_rows(r, ids, T, blocks, hidden, hid.data(), cached);
    // the sequence as text: no image block (every row before it is causal, a reference's span sees itself)
    plan(T, 0, 0);
    if (!dit_->set_sequence(T, 0, y.data(), x.data(), err, blocks.empty() ? nullptr : key_lim.data()))
        throw std::runtime_error("image engine: " + err);
    if (!dit_->forward(hid.data(), hid.data(), dit_->layers(), {}, err))
        throw std::runtime_error("image engine: the DiT: " + err);
    dit_->drop_sequence();
    // the last row through ln_f (an RMS norm, the layers' eps) and the 37 rows of lm_head: the largest logit
    const float* h = hid.data() + (size_t) ((T - 1) * hidden);
    double ss = 0.0;
    for (int64_t c = 0; c < hidden; ++c) ss += (double) h[c] * h[c];
    const double inv = 1.0 / std::sqrt(ss / (double) hidden + (double) dit_->dims().eps);
    int best = 0;
    ratio_logits_.assign((size_t) kHunyuanImage3RatioBuckets, 0.0);
    for (int i = 0; i < kHunyuanImage3RatioBuckets; ++i) {
        const float* w = ratio_rows_.data() + (size_t) i * (size_t) hidden;
        double logit = 0.0;
        for (int64_t c = 0; c < hidden; ++c) logit += (double) h[c] * inv * ratio_norm_[(size_t) c] * w[c];
        ratio_logits_[(size_t) i] = logit;
        if (logit > ratio_logits_[(size_t) best]) best = i;
    }
    if (seconds != nullptr) *seconds = std::chrono::duration<double>(clock::now() - t0).count();
    return best;
}

std::vector<uint8_t> HunyuanImage3Pipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                                     const std::function<bool(int, int)>& step,
                                                     const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs a base image (an init image or "
                                    "a reference)");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    if (!r.neg_ids.empty() || r.cfg > 1.0f)
        throw std::invalid_argument("image engine: HunyuanImage 3.0 samples once: no negative prompt, no cfg");
    if (r.drop != 0)
        throw std::invalid_argument("image engine: HunyuanImage 3.0 reads its whole sequence: drop must be 0");
    const auto t0 = clock::now();
    std::string err;
    std::vector<int32_t> ids;
    int64_t T = 0, start = 0;
    std::vector<int32_t> y, x, key_lim;
    std::vector<HunyuanImage3Block> blocks;
    if (!read_sequence(r, ids, T, start, y, x, key_lim, blocks, err)) throw std::invalid_argument(err);
    ensure_open();
    const int64_t hidden = dit_->dims().hidden;
    const HunyuanImage3Block& gen = blocks.back();

    // ---- the text rows: `wte` over the ids before the block, gathered once for the sequence and kept for the same
    // ids (and the same reference images) coming again, the reference images' rows with them.  The walk rewrites
    // every row of the sequence, text included, so the cached rows go back into the buffer at each step - the host
    // reference re-gathers them per step (`hidden_in_host`), which is why the two paths agree at one step and only
    // there when this copy is missed.  The references are encoded before the plan takes the card for the banks.
    if (stage) stage(r.refs.empty() ? "text" : "refs");
    const int64_t th = r.height / kDownsample, tw = r.width / kDownsample;
    size_plan(T, th, tw, !r.refs.empty() || base != nullptr,
              std::max(refs_encode_bytes(blocks, r.refs.size()), base != nullptr ? base_encode_bytes(tw, th) : 0));
    std::vector<float> hid((size_t) (T * hidden));
    bool cached = false;
    text_rows(r, ids, start, blocks, hidden, hid.data(), cached);
    // a base image (img_start.hpp): its latent at the generated block's size, where the references were read
    std::vector<float> z0;
    if (base != nullptr) z0 = base_latent(*base, tw, th);
    QiPlan p = plan(T, r.height / kDownsample, r.width / kDownsample);
    p.text_cached = cached;
    if (!dit_->set_sequence(T, (T - start), y.data(), x.data(), err, blocks.size() > 1 ? key_lim.data() : nullptr))
        throw std::runtime_error("image engine: " + err);
    auto restore_text_rows = [&] {
        std::copy(text_rows_.begin(), text_rows_.end(), hid.begin());
    };

    // ---- the schedule: the request's sigmas if it gives them, else its step count through the model's shift,
    // else the model's own 8 steps (`--steps`, or `STEPS:` on an IMG line, or neither: the Distil's default)
    std::vector<float> sigmas = r.sigmas;
    if (sigmas.empty()) {
        const int64_t wanted = r.steps > 0 ? r.steps : kDistilSteps;
        sigmas.resize((size_t) (wanted + 1));
        hunyuanimage3_sigmas(wanted, kDistilShift, sigmas.data());
    }
    sigmas = strength_sigmas(sigmas, r.strength);
    const int64_t steps = (int64_t) sigmas.size() - 1;
    const float guidance_scale = r.guidance > 0.0f ? r.guidance : kDistilGuidance;
    const float guidance = 1000.0f * guidance_scale;   // the number the guidance token embeds
    const int64_t n_img = th * tw, n_lat = kLatentChannels * n_img;
    float* img = hid.data() + (size_t) (start * hidden);   // the block's rows, where patch_embed writes
    if (!around_->set_grid(th, tw, err)) throw std::runtime_error("image engine: " + err);

    std::vector<float> lat((size_t) n_lat), vel((size_t) n_lat);
    if (!init_latent_.empty()) {
        if ((int64_t) init_latent_.size() != n_lat)
            throw std::invalid_argument("image engine: the starting latent is not this size's [32][" +
                                        std::to_string(th) + "][" + std::to_string(tw) + "]");
        std::copy(init_latent_.begin(), init_latent_.end(), lat.begin());
    } else {
        const auto noise = philox_randn(r.seed, (uint64_t) n_lat);
        std::copy(noise.begin(), noise.end(), lat.begin());
    }
    // img2img: from the base image's latent on the flow's path at the first sigma; a mask: the positions it keeps
    // follow that path after each step (img_start.hpp, here on the host latent [32][th][tw])
    const std::vector<float> noise = base != nullptr ? lat : std::vector<float>();
    if (base != nullptr && r.strength < 1.0)
        for (size_t i = 0; i < (size_t) n_lat; ++i) lat[i] = (1.f - sigmas[0]) * z0[i] + sigmas[0] * noise[i];
    std::vector<uint8_t> mask_px;
    std::vector<float> m;
    if (mask) {
        mask_px = crop_rgba8(r.mask, r.width, r.height);
        m = request_latent_mask(r, mask_px, r.width, r.height, kDownsample);
    }

    const auto t1 = clock::now();
    if (stage) stage("sampling");
    float *dz = nullptr, *drgb = nullptr;
    try {
        for (int64_t s = 0; s < steps; ++s) {
            const float sigma = sigmas[(size_t) s], sigma_next = sigmas[(size_t) (s + 1)];
            const float t = sigma * 1000.0f, t_r = sigma_next * 1000.0f;
            restore_text_rows();
            // STRATA_HUNYUANIMAGE3_HOST_ENDS=patch|final|both: a check's - the host FP32 pieces in place of the
            // card's, so one run at a time can tell either piece's cost in the picture (see the pipeline header)
            static const std::string kHostEnds = [] {
                const char* v = strata::env_get("HUNYUANIMAGE3_HOST_ENDS");
                return std::string(v != nullptr ? v : "");
            }();
            const bool host_patch = kHostEnds == "patch" || kHostEnds == "both";
            const bool host_final = kHostEnds == "final" || kHostEnds == "both";
            // the generated block's three meta rows (a reference block's `<timestep>` keeps timestep_emb(0),
            // which the cached rows carry), then its patch_embed
            if (!hunyuanimage3_meta_rows_host(*dit_->file(), ids.data() + gen.boi, start - gen.boi, t, t_r,
                                              guidance, hid.data() + gen.boi * hidden, err) ||
                (host_patch ? !hunyuanimage3_patch_embed_host(*dit_->file(), lat.data(), th, tw, t, img, err)
                            : !around_->patch_embed(lat.data(), th, tw, t, img, err)))
                throw std::runtime_error("image engine: " + err);
            if (!dit_->forward(hid.data(), hid.data(), dit_->layers(), {}, err))
                throw std::runtime_error("image engine: the DiT: " + err);
            if (host_final ? !hunyuanimage3_final_layer_host(*dit_->file(), img, th, tw, t, vel.data(), err)
                           : !around_->final_layer(img, th, tw, t, vel.data(), err))
                throw std::runtime_error("image engine: " + err);
            hunyuanimage3_euler_step(lat.data(), vel.data(), n_lat, sigma, sigma_next);
            if (mask)
                for (size_t i = 0; i < (size_t) n_lat; ++i) {
                    const float k = m[i % (size_t) n_img];
                    lat[i] = k * lat[i] + (1.f - k) * ((1.f - sigma_next) * z0[i] + sigma_next * noise[i]);
                }
            if (step && !step((int) (s + 1), (int) steps)) throw QiStopped();
        }
        if (times != nullptr) times->sampling = std::chrono::duration<double>(clock::now() - t1).count();

        // ---- the VAE.  The sequence's buffers are the DiT's last hold on the card, and the decode is the bigger
        // worker at a full image's size: give them back first, then what the banks leave is the VAE's.  The banks'
        // resident prefix is this model's cache between requests, but they are 43 GB and the card is not, so a
        // decode that needs the room takes it from them and the next image's walk streams the banks in again (as
        // QiPipeline frees the blocks and the encoder's layers the VAE lacks room beside).
        dit_->drop_sequence();
        around_->drop_grid();   // the entry and the exit's buffers too: the decode is the bigger worker
        const uint64_t vae_need = vae_->scratch_bytes(th, tw);
        auto room = [&] {
            const uint64_t used = p.fixed + dit_->bank_vram();
            return p.budget > used ? p.budget - used : 0;
        };
        uint64_t vae_room = room();
        // the resident banks give way to a whole decode that then fits; one that cannot fit anyway goes in tiles
        // and leaves them
        if (vae_need > vae_room && vae_need + dit_->bank_min() <= p.budget - p.fixed &&
            dit_->bank_vram() > dit_->bank_min()) {
            dit_->set_bank_budget(dit_->bank_min());
            banks_set_ = dit_->bank_min();
            vae_room = room();
        }
        // the whole decode where it fits, else the reference's tiles
        const bool tiled = vae_need > vae_room;
        if (tiled && vae_->tiled_scratch_bytes() > vae_room)
            throw std::runtime_error("image engine: the VAE's decode of a " + std::to_string(r.width) + " x " +
                                     std::to_string(r.height) + " image needs " +
                                     std::to_string(vae_->tiled_scratch_bytes() >> 20) +
                                     " MiB in tiles beside the weights and the streamed slots, and the budget leaves " +
                                     std::to_string(vae_room >> 20));
        p.vae_tiled = tiled;
        if (stage) stage("decode");
        const auto t2 = clock::now();
        const int64_t HW = r.width * r.height;
        if (!gpu::alloc_device(&dz, (size_t) n_lat * sizeof(float)) ||
            !gpu::alloc_device(&drgb, (size_t) (3 * HW) * sizeof(float))) {
            if (dz != nullptr) gpu::free(dz);
            if (drgb != nullptr) gpu::free(drgb);
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        }
        gpu::copy(dz, lat.data(), (size_t) n_lat * sizeof(float));
        if (tiled) vae_->decode_tiled(dz, th, tw, drgb);
        else vae_->decode(dz, th, tw, drgb);
        std::vector<float> rgb((size_t) (3 * HW));
        gpu::copy(rgb.data(), drgb, rgb.size() * sizeof(float));
        gpu::free(dz);
        gpu::free(drgb);
        dz = drgb = nullptr;
        // the next request's plan counts the decode's buffers beside the banks only while it decodes: kept, they
        // took the walk's room on a 12 GB card (RTX 4070: the second image's sequence refused)
        vae_->drop_scratch();
        if (times != nullptr) times->decode = std::chrono::duration<double>(clock::now() - t2).count();

        // the reference pipeline's postprocess: [-1,1] -> [0,1], the three planes into RGBA8 rows
        std::vector<uint8_t> out((size_t) (4 * HW));
        for (int64_t q = 0; q < HW; ++q) {
            for (int c = 0; c < 3; ++c) {
                const float v = std::clamp(rgb[(size_t) (c * HW + q)] / 2.0f + 0.5f, 0.0f, 1.0f);
                out[(size_t) (q * 4 + c)] = (uint8_t) std::lround(v * 255.0f);
            }
            out[(size_t) (q * 4 + 3)] = 255;
        }
        // a mask's kept part is the base image's own pixels (not through the VAE), blended by the mask's alpha
        if (mask) keep_pixels(out, crop_rgba8(*base, r.width, r.height), mask_px);
        if (times != nullptr) times->text = std::chrono::duration<double>(t1 - t0).count();
        if (plan_out != nullptr) {
            p.budget = budget();   // what the next request plans with (the banks may have given room to the VAE)
            *plan_out = p;
        }
        return out;
    } catch (...) {
        if (dz != nullptr) gpu::free(dz);
        if (drgb != nullptr) gpu::free(drgb);
        dit_->drop_sequence();   // a stopped or failed request gives its buffers back
        around_->drop_grid();
        vae_->drop_scratch();
        throw;
    }
}

}  // namespace strata::image
