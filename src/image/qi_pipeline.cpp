// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_pipeline.cpp - see include/strata/image/qi_pipeline.hpp.

#include "sigmas.hpp"
#include "strata/image/qi_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/cnet_prep.hpp"
#include "strata/image/img_start.hpp"
#include "strata/image/qi_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
// the text encoder's longest prompt, references' rows among it (its K/V cells, shared by its layers: 32 MiB for
// Qwen3-VL-8B)
constexpr int64_t kTextTokens = 16384;
constexpr uint64_t kScratch = 384 * kMiB;     // the products' dequantization scratch (the DiT's and the encoder's)

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

void qi_ref_size(int64_t ref_w, int64_t ref_h, int64_t& w, int64_t& h) {
    // diffusers' calculate_dimensions(1024 * 1024, ratio): the width the root of the area times the ratio, the height the
    // width over it, each rounded to a multiple of 32 as Python's round does (half to even: nearbyint)
    const double ratio = (double) ref_w / (double) ref_h;
    const double vw = std::sqrt(1024.0 * 1024.0 * ratio), vh = vw / ratio;
    w = std::max<int64_t>(32, (int64_t) std::nearbyint(vw / 32) * 32);
    h = std::max<int64_t>(32, (int64_t) std::nearbyint(vh / 32) * 32);
}

uint64_t free_vram() {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    return free;
}

// The room this device's driver and runtime keep for themselves, and the copies' staging: the one rule every
// engine and stage sizes by (core::vram_room), of the memory the engine sees for this device.  `parts`: the part
// of it a stage keeps beside a plan that already keeps the whole room (MOSS-Image's slack beside its estimates).
uint64_t vram_room(uint64_t parts) {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    return core::vram_room(total, parts);
}

// The device's memory in use past this process's own allocations (gpu::device_held): the runtime's and the libraries'
// (their code, their workspaces), other programs', and what the allocations take past their sizes.  The HIP runtime
// takes a block of 1 to 2 MiB as 2 MiB and a larger one in steps of 2 MiB (the RX 9060 XT, ROCm 7.1 and 10.1,
// 2026-11-05: 200 blocks of 1 MiB + 4 KiB took 400 MiB), so it grows with the engine's blocks: an edit after a refined
// edit on ROCm 10.1 held 15176 MiB in 1151 blocks of a 15358 MiB budget with 46 MiB free, 1082 MiB of the 16304 past
// them against 434 when the budget was set.
int64_t vram_other() {
    size_t free = 0, total = 0;
    if (!gpu::mem_info(&free, &total)) return 0;
    return (int64_t) total - (int64_t) free - (int64_t) gpu::device_held();
}

void ImagePipeline::note_others() { other_base_ = vram_other(); }

uint64_t ImagePipeline::less_others(uint64_t b) const {
    const int64_t grown = vram_other() - other_base_;
    return grown > 0 ? b - std::min(b, (uint64_t) grown) : b;
}

void ImagePipeline::fit(uint64_t reserve) {
    const uint64_t have = held() + free_vram();
    set_budget(have > reserve ? have - reserve : 0);
}

uint64_t ImagePipeline::plan_avail(const QiPlan& p) {
    if (p.budget <= p.fixed)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is below the weights that always stay");
    return p.budget - p.fixed;
}

void ImagePipeline::plan_too_small(const QiPlan& p) {
    throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                             " MiB) is too small for this image");
}

void PreviewDecode::operator()(int done, const std::function<void(float*)>& make) {
    if (!r_.on_preview || std::find(r_.preview.begin(), r_.preview.end(), done) == r_.preview.end()) return;
    const uint64_t room = vae_beside(p_);
    if (x0_ == nullptr && !gpu::alloc_device(&x0_, (size_t) values_ * 4)) return;
    make(x0_);
    try {
        decode_(x0_, room);
    } catch (const std::exception&) {
        drop_();
        return;
    }
    drop_();
    r_.on_preview(shown_++, done, image_());
}

void PreviewDecode::release() {
    gpu::free(x0_);
    x0_ = nullptr;
}

void ImagePipeline::vram_into(QiNeed& n) const {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    n.held = held();
    n.free = free;
    n.cap = vram_cap(total);
}

uint64_t vram_cap(uint64_t total) {
    const uint64_t margin = vram_room();
    uint64_t cap = total > margin ? total - margin : 0;
    if (const uint64_t l = core::vram_limit_bytes(); l > 0) cap = std::min(cap, l > margin ? l - margin : 0);
    return cap;
}

bool QiPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                      const std::string& vision) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    vision_path_ = vision;
    if (!vision.empty()) {
        vision_size_ = std::make_unique<core::Vision>();
        if (!vision_size_->open(vision, s_, err, true)) return false;
    }
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    te_size_ = std::make_unique<TextEncoder>();
    dit_size_ = std::make_unique<QiDit>();
    vae_size_ = std::make_unique<QiVae>();
    if (!(te_size_->open(text, kTextTokens, err, UINT64_MAX, true) && dit_size_->open(dit, s_, err, true) &&
          vae_size_->open(vae, s_, err, true)))
        return false;
    // the budget a request plans with: what was free before the parts, less a margin, within the small-card limit the
    // engine takes too
    auto_budget_ = vram_cap(free);
    note_others();
    return true;
}

void QiPipeline::ensure_open() {
    if (dit_ && vae_ && te_) return;
    std::string err;
    te_ = std::make_unique<TextEncoder>();
    dit_ = std::make_unique<QiDit>();
    vae_ = std::make_unique<QiVae>();
    gpu::stream_sync(s_);   // the encoder's own stream: what the engine freed on its stream first
    if (!(te_->open(text_path_, kTextTokens, err) && dit_->open(dit_path_, s_, err) && vae_->open(vae_path_, s_, err))) {
        te_.reset();
        dit_.reset();
        vae_.reset();
        throw std::runtime_error(err);
    }
    te_->set_final_norm(false);   // Qwen-Image's context: the last block's output before the final norm
    dit_->set_block_budget(dit_->block_min());   // nothing resident until a request plans its budget
    te_->set_layer_budget(te_->layer_min());
    dit_set_ = dit_->block_min();
    te_set_ = te_->layer_min();
    mlp_set_ = -1;
}

uint64_t QiPipeline::held() const {
    if (!(dit_ && vae_ && te_)) return 0;
    const uint64_t ctl = control_ ? control_->weight_bytes() + control_->blocks_vram() + control_->activations_held() : 0;
    return dit_->weight_bytes() + dit_->lora_bytes() + vae_->weight_bytes() + kScratch + dit_->blocks_vram() +
           te_->layers_vram() + dit_->activations_held() + vae_->scratch_held() + ctl;
}

QiNeed QiPipeline::need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                        const std::vector<LoraUse>& loras) {
    const QiDit& dit = *dit_size_;
    const TextEncoder& te = *te_size_;
    const QiVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16;
    int64_t T = n_text + h * w, te_T = n_text + 16;   // the encoder's tokens: the system turn's too
    uint64_t edit = 0;   // an edit's reference work before the sampling: the vision encoder and the VAE's encoder
    for (const auto& [rw0, rh0] : refs) {
        int64_t rw = 0, rh = 0;
        qi_ref_size(rw0, rh0, rw, rh);
        T += (rh / 16) * (rw / 16) - 1;   // the pad's place: the reference's latent tokens
        te_T += (rh / 32) * (rw / 32) - 1;   // and in the encoder its image tokens
        const uint64_t v = vision_size_ ? vision_size_->weight_bytes() + vision_size_->scratch_bytes((rh / 16) * (rw / 16)) : 0;
        edit = std::max(edit, v + vae.encode_scratch_bytes(rh, rw) + (uint64_t) (4 * rh * rw) * 4);
    }
    // a base image (img2img's or a mask's): encoded at the image's size; its latent, and a mask's noise and positions,
    // kept through the sampling
    if (mask) edit = std::max(edit, vae.encode_scratch_bytes(height, width) + (uint64_t) (4 * height * width) * 4);
    const uint64_t inpaint = mask ? LatentStart::bytes(dit.channels() * h * w, h * w, true) : 0;
    // guidance: the negative prompt's prefix beside the first (the same references), and its sequence
    // its activations: both sequences side by side after the first step (forward_pair)
    const int64_t neg_pre = n_neg > 0 ? n_neg + (T - h * w - n_text) : 0,
                  T_act = n_neg > 0 ? qi_attn_rows(T) + qi_attn_rows(neg_pre + h * w) : T;
    // the adapters: their A and B (in VRAM for the request, the blocks streamed or not), and the products' scratch of
    // a step's rows; nothing bound yet, so the DiT's activations here hold none of it
    const uint64_t lora = lora_gpu_bytes(loras, T_act);
    const uint64_t fixed = dit.weight_bytes() + vae.weight_bytes() + kScratch + lora +
                           dit.prefix_row_bytes(T - h * w) + dit.prefix_row_bytes(neg_pre) + inpaint;
    // the prefix cache: in VRAM in the plan's room, the rest in pinned RAM (least: all of it there)
    const uint64_t kv = dit.prefix_kv_bytes(T - h * w) + dit.prefix_kv_bytes(neg_pre), kv_all = kv * (uint64_t) dit.blocks();
    QiNeed n;
    if (n_neg > 0) te_T = std::max(te_T, te_T - n_text + n_neg);
    T = T_act;
    edit = std::max(edit, te.activation_bytes(te_T));   // the encoder's own buffers, before the DiT's
    n.least = fixed + std::max({dit.activation_bytes(T, 512), vae.scratch_min(), edit}) + dit.block_min() +
              te.layer_min();
    // the DiT's blocks resident (used 8 times an image), the encoder streamed (once a prompt), the VAE in one piece
    n.full = fixed + kv_all + dit.block_bytes() + te.layer_min() +
             std::max({dit.activation_bytes(T), vae.scratch_bytes(h, w), edit});
    vram_into(n);
    const uint64_t room = n.cap > n.least ? n.cap - n.least : 0;
    n.host = kv == 0 ? 0 : kv_all - std::min<uint64_t>(room / kv, (uint64_t) dit.blocks()) * kv;
    return n;
}

void QiPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    note_others();
    if (!(dit_ && vae_ && te_)) return;   // released: the next request plans with it
    const QiPlan p = plan(0, 0, 0);   // shrink at once what no longer fits (a grown budget is taken at the next request)
    (void) p;
}

void QiPipeline::release(bool all) {
    if (!(dit_ && vae_ && te_)) return;
    if (all) {   // every buffer the parts hold goes with them; the next request opens them again
        if (dit_) dit_->set_control(nullptr);
        control_.reset();
        control_set_ = 0;
        drop_kept();
        dit_->drop_loras();   // the adapters' A and B too (they go back on the next request's, from its files)
        te_.reset();
        dit_.reset();
        vae_.reset();
        return;
    }
    dit_->drop_blocks();
    te_->drop_layers();
    dit_->drop_block_homes();   // the RAM the streamed layers' copies hold, back for the other engines
    te_->drop_layer_homes();
    dit_->drop_activations();
    vae_->drop_scratch();
    if (control_) {
        control_->drop_blocks();
        control_->drop_block_homes();
        control_->drop_activations();
        control_->clear();
        control_set_ = 0;
    }
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

void QiPipeline::use_loras(const QiRequest& r) {
    std::string err;
    std::vector<LoraUse> dit, te;
    lora_split(r.loras, dit, te);
    dit_->set_lora_bake(lora_bake_);
    if (!dit_->set_loras(dit, err) || !te_->set_loras(te, err)) throw LoraRejected(err);
    te_loras_now(te_->lora_key());   // the states kept were the encoder's with the other adapters
}

std::vector<float> QiPipeline::text(const std::vector<int32_t>& ids, const std::vector<TextEncoder::Picture>& pictures) {
    if ((int64_t) ids.size() > kTextTokens)
        throw std::invalid_argument("image engine: the prompt and its references are longer than " +
                                    std::to_string(kTextTokens) + " tokens");
    gpu::stream_sync(s_);
    std::string err;
    std::vector<float> out = te_->encode(ids, err, pictures);
    if (out.empty()) throw std::runtime_error(err);
    return out;
}

void QiPipeline::te_layers() {
    if (te_set_ == te_want_) return;
    te_->set_layer_budget(te_want_);
    te_set_ = te_want_;
}

QiPlan QiPipeline::plan(int64_t T, int64_t h, int64_t w, int64_t n_ids, int64_t n_neg_pre) {
    QiPlan p;
    p.budget = budget();
    // the prefix's rows stay through the sampling and the decode (an edit's: its references' tokens too)
    const int64_t n_pre = T > 0 ? T - h * w : 0;
    // the adapters' A and B are the fixed weights' (in VRAM whatever the plan does with the blocks; their scratch of a
    // step's rows is among the activations, activation_bytes).  Bound already: what this plan puts beside them
    p.lora = dit_->lora_bytes();
    p.lora_n = dit_->lora_count();
    p.fixed = dit_->weight_bytes() + p.lora + vae_->weight_bytes() + kScratch + dit_->prefix_row_bytes(n_pre) +
              dit_->prefix_row_bytes(n_neg_pre);
    // the ControlNet branch: its weights outside the blocks and its mapped conditioning stay; its stream is among the
    // activations; its blocks share the DiT's room
    QiControl* ctl = control_on_ && control_ ? control_.get() : nullptr;
    if (ctl != nullptr) p.fixed += ctl->weight_bytes() + (uint64_t) (h * w * dit_->hidden()) * 4;
    // the activations: guidance's both sequences side by side after the first step (forward_pair)
    if (T > 0 && n_neg_pre > 0) T = qi_attn_rows(T) + qi_attn_rows(n_neg_pre + h * w);
    const uint64_t avail = plan_avail(p), te_min = te_->layer_min(),
                   dit_min = dit_->block_min() + (ctl != nullptr ? ctl->block_min() : 0);
    // the DiT's activations, the MLP in bands of rows when the whole takes more than a third of what is there
    const uint64_t stream = ctl != nullptr ? (uint64_t) (T * dit_->hidden()) * 4 : 0;
    const int64_t mlp = mlp_bands(avail, T > 0, p.activations,
                                  [&](int64_t rows) { return dit_->activation_bytes(T, rows) + stream; });
    // the room after the layers: the DiT's activations, and later the VAE's least (they take it in turn)
    // the encoder's buffers too: they come and go before the DiT's (generate frees them after the encode)
    const uint64_t work = std::max({p.activations, vae_->scratch_min(), te_->activation_bytes(n_ids)});
    if (avail < work + dit_min + te_min) plan_too_small(p);
    // the prefix cache first: a block's that is not in VRAM is copied in at every step with the compute waiting (the
    // blocks' weights stream beside it); then the DiT's blocks (8 or more uses an image), then the encoder's layers
    const uint64_t kv = dit_->prefix_kv_bytes(n_pre) + dit_->prefix_kv_bytes(n_neg_pre),
                   spare = avail - work - dit_min - te_min;
    p.resident_kv = kv > 0 ? (int) std::min<uint64_t>(spare / kv, (uint64_t) dit_->blocks()) : dit_->blocks();
    p.prefix_kv = (uint64_t) p.resident_kv * kv;
    dit_->set_prefix_vram_blocks(p.resident_kv);
    const uint64_t ctl_bytes = ctl != nullptr ? ctl->block_bytes() : 0;
    p.blocks = clampu(avail - work - te_min - p.prefix_kv, dit_min, dit_->block_bytes() + ctl_bytes);
    p.layers = clampu(avail - work - p.blocks - p.prefix_kv, te_min, te_->layer_bytes());
    uint64_t dit_blocks = p.blocks;
    if (ctl != nullptr) {   // the blocks' room shared by their bytes, each at least its two slots
        const uint64_t cb = clampu((uint64_t) ((double) p.blocks * (double) ctl_bytes /
                                               (double) (dit_->block_bytes() + ctl_bytes)),
                                   ctl->block_min(), ctl_bytes);
        dit_blocks = std::max(p.blocks - std::min(p.blocks, cb), dit_->block_min());
        if (cb != control_set_) {
            ctl->set_block_budget(cb);
            control_set_ = cb;
        }
    }
    plan_blocks(*dit_, dit_blocks);
    // the encoder's layers: fewer at once (their room is the DiT's now), more only when it encodes (te_layers): the
    // decode gives them to the VAE, and a prompt whose states are kept does not run it (RX 9060 XT: 163 ms an image
    // to take 6300 MiB back for nothing)
    te_want_ = p.layers;
    if (p.layers < te_set_) {
        te_->set_layer_budget(p.layers);
        te_set_ = p.layers;
    }
    plan_rows(p, *dit_, T > 0, mlp);
    p.resident_layers = te_->resident_layers();
    // the VAE: what the layers leave, the DiT's activations' share included (freed if it needs it), in tiles when even
    // that is short
    const uint64_t left = avail - p.blocks - p.layers - p.prefix_kv;
    p.vae = left;
    p.vae_tiled = h > 0 && vae_->scratch_bytes(h, w) > left;
    return p;
}

void QiPipeline::edit_refs(const QiRequest& r, std::vector<int32_t>& ids, std::vector<core::VisionOut>& vouts,
                           std::vector<TextEncoder::Picture>& te_images, std::vector<QiDit::Ref>& dit_refs,
                           std::vector<float*>& ref_z, std::vector<int32_t>& neg_ids,
                           std::vector<TextEncoder::Picture>& neg_te, std::vector<QiDit::Ref>& neg_refs, bool vision) {
    if (vision_path_.empty()) throw std::invalid_argument("image engine: an edit needs the vision encoder (--vision)");
    const int32_t pad = te_size_->image_pad();   // the <|image_pad|> token, from the text encoder's vocabulary
    if (std::count(r.ids.begin(), r.ids.end(), pad) != (int64_t) r.refs.size())
        throw std::invalid_argument("image engine: the prompt's image pads and the reference images differ in number");
    // room for the vision encoder and the VAE's encoder: the cache gives way when the budget is short of them
    std::vector<std::pair<int64_t, int64_t>> sizes;
    sizes.reserve(r.refs.size());
    for (const QiRef& f : r.refs) sizes.emplace_back(f.width, f.height);
    const bool base = base_image(r) != nullptr;
    const QiNeed nd = need(0, r.width, r.height, sizes, base);
    const uint64_t edit = nd.least - need(0, r.width, r.height).least;
    if (held() + edit > budget()) release(false);
    ensure_open();
    std::string err;
    if (vision) {
        vision_ = std::make_unique<core::Vision>();
        if (!vision_->open(vision_path_, s_, err)) {
            vision_.reset();
            throw std::runtime_error(err);
        }
    }
    std::vector<std::pair<int64_t, int64_t>> grids;   // each reference's worked size
    // their latents (owned by kept_): the kept ones for the same references, else encoded again
    const std::string key = refs_key(r.refs);
    const bool same = key == kept_.refs && kept_.z.size() == r.refs.size();
    if (!same) {
        drop_kept(false);
        kept_.refs.clear();
    }
    for (const QiRef& f : r.refs) {
        if (f.width <= 0 || f.height <= 0 || f.rgba.size() != (size_t) (f.width * f.height * 4))
            throw std::invalid_argument("image engine: a reference image is empty");
        int64_t rw = 0, rh = 0;
        qi_ref_size(f.width, f.height, rw, rh);
        // resized as stable-diffusion.cpp resizes it (nearest): the VAE's input its 0 .. 1 as the file has it, the
        // vision encoder's input over white in -1 .. 1 (neither needed for kept states and a kept latent)
        std::vector<float> rgba, rgb;
        if (vision || !same) {
            const std::vector<uint8_t> px = resize_nearest(f, rw, rh);
            rgba = image_planes(px, rw, rh, 4, PlaneAlpha::Keep, PlaneRange::Unit);
            rgb = image_planes(px, rw, rh, 3, PlaneAlpha::WhiteUnit, PlaneRange::Signed);
        }
        if (vision) {
            vouts.push_back(vision_->encode(rgb.data(), rw, rh));
        } else {   // the kept states' references: their vision outputs' shapes only (no rows: the text is not encoded)
            core::VisionOut v;
            const core::VisionOut& was = last_vision_[vouts.size()];
            v.rows = was.rows;
            v.dim = was.dim;
            v.grid_h = was.grid_h;
            v.grid_w = was.grid_w;
            vouts.push_back(std::move(v));
        }
        grids.emplace_back(rw, rh);
        if (same) {
            ref_z.push_back(kept_.z[ref_z.size()]);
            continue;
        }
        float *d = nullptr, *z = nullptr;
        if (!gpu::alloc_device(&z, (size_t) (64 * (rh / 16) * (rw / 16)) * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        kept_.z.push_back(z);
        ref_z.push_back(z);
        if (!gpu::alloc_device(&d, rgba.size() * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        gpu::copy(d, rgba.data(), rgba.size() * 4);
        vae_->encode(d, rh, rw, z);
        gpu::free(d);
    }
    vision_.reset();   // its weights go: an edit opens it again
    vae_->drop_scratch();
    kept_.refs = key;   // every reference's latent is in kept_ now
    if (vision) {   // their shapes, for the next edit of the same request and references
        last_vision_.clear();
        for (const core::VisionOut& v : vouts) {
            core::VisionOut s;
            s.rows = v.rows;
            s.dim = v.dim;
            s.grid_h = v.grid_h;
            s.grid_w = v.grid_w;
            last_vision_.push_back(std::move(s));
        }
    }
    // the ids with each pad widened to its reference's tokens: the prompt's, and guidance's negative prompt's
    auto widen = [&](const std::vector<int32_t>& src, std::vector<int32_t>& out, std::vector<TextEncoder::Picture>& te,
                     std::vector<QiDit::Ref>& refs) {
        size_t k = 0;
        for (const int32_t id : src) {
            if (id != pad) {
                out.push_back(id);
                continue;
            }
            if (k >= vouts.size())
                throw std::invalid_argument("image engine: the negative prompt's image pads and the references differ");
            const core::VisionOut& v = vouts[k];
            TextEncoder::Picture im;
            im.index = (int64_t) out.size();
            im.nx = v.grid_w;
            im.ny = v.grid_h;
            im.rows = v.main.data();
            for (const auto& d : v.deepstack) im.deepstack.push_back(d.data());
            te.push_back(im);
            QiDit::Ref f;
            f.slot = im.index;
            f.slots = v.rows;
            f.w = grids[k].first / 16;
            f.h = grids[k].second / 16;
            f.latent = ref_z[k];
            refs.push_back(f);
            out.insert(out.end(), (size_t) v.rows, pad);
            ++k;
        }
    };
    widen(r.ids, ids, te_images, dit_refs);
    if (r.cfg > 1.0f && !r.neg_ids.empty()) widen(r.neg_ids, neg_ids, neg_te, neg_refs);
}

std::vector<uint8_t> QiPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                          const std::function<bool(int, int)>& step,
                                          const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    if (!r.sampler.empty() && r.sampler != "euler" && r.sampler != "er_sde")
        throw std::invalid_argument("image engine: Qwen-Image's sampler is euler or er_sde");
    const int64_t h = r.height / 16, w = r.width / 16;
    const auto t0 = clock::now();
    ensure_open();
    if (r.control_net.empty() && control_) {
        // A request without ControlNet does not budget its weights or blocks.  Give them back before processing
        // references too: their vision and VAE encoders run before use_control and before the sampling plan.
        dit_->set_control(nullptr);
        control_.reset();
        control_set_ = 0;
        control_on_ = false;
    }
    use_loras(r);   // before the plan (their bytes are fixed) and the context (a set that changed drops it)
    // an edit: its references through the vision encoder and the VAE's encoder, their pads widened to their tokens
    std::vector<int32_t> ids;
    std::vector<core::VisionOut> vouts;
    std::vector<TextEncoder::Picture> te_images;
    std::vector<QiDit::Ref> dit_refs;
    std::vector<float*> ref_z;   // the references' latents (kept_'s)
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs a base image (an init image or "
                                    "a reference)");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    const std::vector<float> sigmas = strength_sigmas(
        r.sigmas.empty() ? (r.base_schedule && r.steps > 0 ? qi_base_sigmas((int) r.steps, r.width, r.height)
                                                           : turbo_sigmas())
                         : r.sigmas, r.strength);
    LatentStart ls;
    const bool guided = r.cfg > 1.0f && !r.neg_ids.empty();
    std::vector<int32_t> neg_ids;   // an edit's negative prompt, its pads widened
    std::vector<TextEncoder::Picture> neg_te;
    std::vector<QiDit::Ref> neg_refs;
    // an edit whose text states are kept (the same request and references, the guidance's negative prompt too): the
    // vision encoder, which only the text encoder reads, is not run
    const std::string refs = refs_key(r.refs);
    const bool kept = !r.refs.empty() && refs == last_refs_ && r.ids == last_req_ids_ &&
                      last_vision_.size() == r.refs.size() &&
                      (!guided || (refs == last_neg_refs_ && r.neg_ids == last_req_neg_));
    if (!r.refs.empty()) {
        if (stage) stage("refs");
        edit_refs(r, ids, vouts, te_images, dit_refs, ref_z, neg_ids, neg_te, neg_refs, !kept);
    }
    if (base != nullptr) {
        if (stage && r.refs.empty()) stage("refs");
        encode_base(r, *base, ls);
    }
    use_control(r);
    if (stage) stage("text");
    const std::vector<int32_t>& all_ids = r.refs.empty() ? r.ids : ids;
    const int64_t n_ctx = (int64_t) all_ids.size() - r.drop;
    int64_t n_pre = n_ctx;
    for (const QiDit::Ref& f : dit_refs) n_pre += f.h * f.w - f.slots;
    const std::vector<int32_t>& neg_all = r.refs.empty() ? r.neg_ids : neg_ids;
    const int64_t n_neg_ctx = guided ? (int64_t) neg_all.size() - r.drop : 0;
    if (guided && n_neg_ctx <= 0) throw std::invalid_argument("image engine: no negative prompt after the system turn");
    const int64_t n_neg_pre = guided ? n_neg_ctx + (n_pre - n_ctx) : 0;
    QiPlan p = plan(n_pre + h * w, h, w, (int64_t) std::max(all_ids.size(), neg_all.size()), n_neg_pre);
    // the text states, kept for the same prompt and the same references (an edit again with another seed)
    if (all_ids != last_ids_ || refs != last_refs_) {
        if (kept) throw std::logic_error("image engine: an edit's kept states do not match its request");
        te_layers();
        last_states_ = text(r.refs.empty() ? r.ids : all_ids, r.refs.empty() ? std::vector<TextEncoder::Picture>{} : te_images);
        te_->drop_activations();
        last_ids_ = all_ids;
        last_refs_ = refs;
        last_req_ids_ = r.ids;
    } else {
        p.text_cached = true;
    }
    const int64_t C = te_->hidden();
    for (QiDit::Ref& f : dit_refs) f.slot -= r.drop;
    dit_->set_context(last_states_.data() + r.drop * C, n_ctx, dit_refs);
    if (guided) {   // the negative prompt's states (kept as the prompt's are; its references are the prompt's)
        if (neg_all != last_neg_ids_ || refs != last_neg_refs_) {
            te_layers();
            last_neg_states_ = text(r.refs.empty() ? r.neg_ids : neg_all,
                                    r.refs.empty() ? std::vector<TextEncoder::Picture>{} : neg_te);
            if (kept) throw std::logic_error("image engine: an edit's kept states do not match its request");
            last_neg_ids_ = neg_all;
            last_neg_refs_ = refs;
            last_req_neg_ = r.neg_ids;
        }
        te_->drop_activations();
        for (QiDit::Ref& f : neg_refs) f.slot -= r.drop;
        dit_->set_context(last_neg_states_.data() + r.drop * C, n_neg_ctx, neg_refs, true);
    } else {
        dit_->drop_second_context();
    }
    const auto t1 = clock::now();
    const int64_t n = dit_->channels() * h * w, HW = r.width * r.height;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    std::vector<float> out((size_t) (4 * HW));
    // a mask: the latent tokens with a pixel of the mask less than opaque change, the rest follow the base image's
    // latent from the seed's noise (img_start.hpp)
    std::vector<uint8_t> mask_px;
    if (mask) {
        mask_px = resize_nearest(r.mask, r.width, r.height);
        ls.set_mask(request_latent_mask(r, mask_px, r.width, r.height, 16), noise, dit_->channels(), h * w, false);
    }
    try {
        gpu::copy(x, noise.data(), (size_t) n * 4);
        if (r.strength < 1.0) ls.start(x, sigmas[0], s_);   // img2img: from the base image's latent at that time
        if (stage) stage("sampling");
        // partial images (r.preview)
        PreviewDecode preview(
            r, p, n, [&](float* x0, uint64_t room) { vae_->decode(x0, h, w, rgba, room); },
            [&] { vae_->drop_scratch(); },
            [&] {
                gpu::copy(out.data(), rgba, out.size() * 4);
                std::vector<uint8_t> img = rgba8(out, HW);
                if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
                return img;
            });
        auto peek = [&](int done, const std::function<void(float*)>& make) { preview(done, make); };
        // The first block can load library workspaces after the initial plan. It also reveals the device's actual
        // per-allocation cost. Reconcile the remaining resident allocations with free memory before loading them;
        // the text encoder has finished, so its cached layers can give up room without changing this image.
        dit_->set_first_block_hook([&] {
            uint64_t pending = dit_->blocks_pending_physical(), free = free_vram();
            if (pending <= free) return;
            const uint64_t shortfall = pending - free;
            const uint64_t layers = std::min(p.layers, te_->layers_vram());
            if (layers > te_->layer_min()) {
                const uint64_t give = std::min(shortfall, layers - te_->layer_min());
                p.layers = layers - give;
                te_->set_layer_budget(p.layers);
                te_set_ = p.layers;
                p.resident_layers = te_->resident_layers();
            }
            pending = dit_->blocks_pending_physical();
            free = free_vram();
            if (pending > free) {
                // The remaining DiT blocks can stream too. The stream accounts for its two slots when it chooses
                // the resident prefix, and synchronizes before giving up any block already used by this step.
                const uint64_t give = pending - free;
                if (control_on_ || p.blocks <= dit_->block_min() + give) plan_too_small(p);
                p.blocks -= give;
                plan_blocks(*dit_, p.blocks);
                p.resident_blocks = dit_->resident_blocks();
            }
        });
        dit_->sample(x, h, w, sigmas, step, ls.after(s_), guided ? r.cfg : 1.0f, peek, &r.guide,
                     r.sampler == "er_sde", r.shift > 0 ? r.shift : 1.0f, r.seed);
        dit_->drop_host_prefix();
        if (control_) {   // the conditioning and the stream are this request's
            control_->clear();
            control_->drop_activations();
        }
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it
        uint64_t vae_room = decode_room(p, *dit_, vae_->scratch_bytes(h, w));
        if (vae_->scratch_bytes(h, w) > vae_room && p.layers > 0) {
            // the encoder's layers too, or their room if they have not taken it (the next prompt that encodes takes it
            // back; the same prompt's states are kept)
            te_->drop_layers();
            te_set_ = 0;
            vae_room += p.layers;
            p.vae_tiled = vae_->scratch_bytes(h, w) > vae_room;
        }
        if (vae_->scratch_bytes(h, w) > vae_room && dit_set_ > dit_->block_min()) {
            // and as many of the DiT's resident blocks as the VAE still lacks: this image's sampling is done with
            // them, and the next one streams them back in (a budget a little short after a shrink tiled the VAE)
            const uint64_t give = std::min<uint64_t>(vae_->scratch_bytes(h, w) - vae_room, dit_set_ - dit_->block_min());
            dit_set_ -= give;
            dit_->set_block_budget(dit_set_);
            vae_room += give;
            p.vae_tiled = vae_->scratch_bytes(h, w) > vae_room;
        }
        vae_->decode(x, h, w, rgba, vae_room);
        // a tile's buffers are not worth keeping beside the layers, and buffers past the plan's share for the VAE (the
        // room the encoder's layers or the DiT's blocks gave it) are that room: the next request takes it back for
        // them (RTX 4070, 1920x1088: the second image's blocks were refused beside the first one's VAE buffers)
        if (p.vae_tiled || vae_->scratch_bytes(h, w) > p.vae) vae_->drop_scratch();
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        dit_->set_first_block_hook({});
        dit_->drop_host_prefix();
        gpu::free(x);
        gpu::free(rgba);
        throw;
    }
    gpu::free(x);
    gpu::free(rgba);
    if (plan_out != nullptr) *plan_out = p;
    std::vector<uint8_t> img = rgba8(out, HW);
    // a mask's kept part is the base image's own pixels (not through the VAE), blended by the mask's alpha
    if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
    return img;
}

void QiPipeline::encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls) {
    if (r.refs.empty()) {   // the VAE encoder's room: the cache gives way when the budget is short of it (an edit's
        const uint64_t more = need(0, r.width, r.height, {}, true).least - need(0, r.width, r.height).least;
        if (held() + more > budget()) release(false);   // references made it in edit_refs already)
    }
    ensure_open();
    if (!vae_->has_encoder()) throw std::invalid_argument("image engine: the VAE file has no encoder for a base image");
    const int64_t n = dit_->channels() * (r.height / 16) * (r.width / 16);
    float* z0 = ls.base(n);
    const std::string key = refs_key({base}) + "@" + std::to_string(r.width) + "x" + std::to_string(r.height);
    if (kept_.base_z != nullptr && key == kept_.base) {   // the same image at the same size: its kept latent
        gpu::copy_async(z0, kept_.base_z, (size_t) n * 4, s_);
        return;
    }
    const std::vector<uint8_t> px = resize_nearest(base, r.width, r.height);
    encode_px(px, r.width, r.height, z0, "the base image");
    // kept for the next request of the same image (1 MiB at 1024 x 1024); without the room, not kept
    if (kept_.base_z != nullptr) gpu::free(kept_.base_z);
    kept_.base_z = nullptr;
    kept_.base.clear();
    if (gpu::alloc_device(&kept_.base_z, (size_t) n * 4)) {
        gpu::copy_async(kept_.base_z, z0, (size_t) n * 4, s_);
        kept_.base = key;
    }
}

void QiPipeline::encode_px(const std::vector<uint8_t>& px, int64_t w, int64_t h, float* z, const char* what) {
    const std::vector<float> rgba = image_planes(px, w, h, 4, PlaneAlpha::Keep, PlaneRange::Unit);
    float* d = nullptr;
    if (!gpu::alloc_device(&d, rgba.size() * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    gpu::copy(d, rgba.data(), rgba.size() * 4);
    const bool ok = vae_->encode(d, h, w, z);
    gpu::free(d);
    vae_->drop_scratch();
    if (!ok) throw std::runtime_error(std::string("image engine: the VAE's encoder failed on ") + what);
}

void QiPipeline::use_control(const QiRequest& r) {
    control_on_ = !r.control_net.empty();
    if (!control_on_) {
        if (control_) control_->clear();
        dit_->set_control(nullptr);
        return;
    }
    const bool mask = !r.mask.rgba.empty() && r.control_inpaint;
    if (r.control.rgba.empty() && !mask)
        throw std::invalid_argument("image engine: ControlNet needs a control image, or a mask to draw through it");
    if (!vae_->has_encoder()) throw std::invalid_argument("image engine: the VAE file has no encoder for ControlNet");
    if (!control_ || control_->path() != r.control_net) {
        dit_->set_control(nullptr);
        control_.reset();
        control_set_ = 0;
        auto c = std::make_unique<QiControl>();
        std::string err;
        if (!c->open(r.control_net, *dit_, err)) throw std::runtime_error("image engine (ControlNet): " + err);
        c->set_block_budget(c->block_min());   // nothing resident until the plan
        control_set_ = c->block_min();
        control_ = std::move(c);
    }
    // control_img_in's 129 inputs (its weight's width) as VideoX-Fun's QwenImage21ControlPipeline (Apache-2.0) builds
    // them: the VAE latent of the control picture at the image's size (64), one channel of 1 - the binarised mask taken
    // to the latent grid by its nearest pixel (1 where the base stays), and the latent of the base picture whose pixels
    // to be redrawn are 0 in -1 .. 1, that is mid grey (64); a part the request leaves out stays 0, as the pipeline
    // feeds it
    const int64_t h = r.height / 16, w = r.width / 16, hw = h * w, ch = dit_->channels();
    std::vector<float> keep((size_t) hw, 0.f);
    float* cond = nullptr;
    if (!gpu::alloc_device(&cond, (size_t) ((2 * ch + 1) * hw) * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    struct Free {
        float* p;
        void* s;
        ~Free() {
            gpu::stream_sync(s);
            gpu::free(p);
        }
    } free_cond{cond, s_};
    gpu::memset(cond, 0, (size_t) ((2 * ch + 1) * hw) * 4);
    if (!r.control.rgba.empty())
        encode_px(control_resize(r.control, r.width, r.height), r.width, r.height, cond, "a ControlNet image");
    if (mask) {
        const QiRef* base = base_image(r);
        if (base == nullptr) throw std::invalid_argument("image engine: a mask through ControlNet needs a base image");
        // redrawn where the mask (its edit share, 1 - alpha) is at least a half, the pipeline's binarisation
        const std::vector<uint8_t> m = resize_nearest(r.mask, r.width, r.height);
        std::vector<uint8_t> px = resize_nearest(*base, r.width, r.height);
        for (int64_t q = 0; q < r.width * r.height; ++q)
            if (255 - (int) m[(size_t) (q * 4 + 3)] >= 128)
                for (int c = 0; c < 3; ++c) px[(size_t) (q * 4 + c)] = 128;
        for (int64_t y = 0; y < h; ++y)
            for (int64_t x = 0; x < w; ++x) {   // nearest: the pixel at the latent cell's place
                const int64_t py = std::min<int64_t>(r.height - 1, (y * r.height) / h),
                              pxx = std::min<int64_t>(r.width - 1, (x * r.width) / w);
                keep[(size_t) (y * w + x)] = 255 - (int) m[(size_t) ((py * r.width + pxx) * 4 + 3)] >= 128 ? 0.f : 1.f;
            }
        encode_px(px, r.width, r.height, cond + (ch + 1) * hw, "a ControlNet image");
    }
    gpu::copy(cond + ch * hw, keep.data(), keep.size() * 4);
    control_->set_input(cond, h, w, r.control_strength, r.control_sigma_lo, r.control_sigma_hi);
    dit_->set_control(control_.get());
}

std::pair<uint64_t, uint64_t> QiPipeline::control_need(const std::string& path, int64_t width, int64_t height) {
    if (!control_size_ || control_size_->path() != path) {
        auto c = std::make_unique<QiControl>();
        std::string err;
        if (!c->open(path, *dit_size_, err, true)) throw std::runtime_error("image engine (ControlNet): " + err);
        control_size_ = std::move(c);
    }
    const QiControl& c = *control_size_;
    const int64_t hw = (width / 16) * (height / 16), T = hw + 512;   // the stream: the image and a prompt's rows
    const uint64_t fixed = c.weight_bytes() + (uint64_t) (hw * dit_size_->hidden()) * 4 + c.activation_bytes(T, hw);
    return {fixed + c.block_min(), fixed + c.block_bytes()};
}

void QiPipeline::drop_kept(bool base_too) {
    gpu::stream_sync(s_);
    for (float* p : kept_.z) gpu::free(p);
    kept_.z.clear();
    kept_.refs.clear();
    if (base_too) {
        if (kept_.base_z != nullptr) gpu::free(kept_.base_z);
        kept_.base_z = nullptr;
        kept_.base.clear();
    }
}

QiPipeline::~QiPipeline() { drop_kept(); }

}  // namespace strata::image
