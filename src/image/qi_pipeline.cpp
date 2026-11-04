// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_pipeline.cpp - see include/strata/image/qi_pipeline.hpp.

#include "strata/image/qi_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
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
constexpr uint64_t kMargin = 512 * kMiB;      // the driver's own and the copies' room, kept free
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

uint64_t vram_cap(uint64_t total) {
    uint64_t cap = total > kMargin ? total - kMargin : 0;
    if (const char* lim = std::getenv("STRATA_VRAM_LIMIT_MIB")) {
        const uint64_t l = std::strtoull(lim, nullptr, 10) * kMiB;
        if (l > 0) cap = std::min(cap, l > kMargin ? l - kMargin : 0);
    }
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
        vision_size_ = std::make_unique<QiVision>();
        if (!vision_size_->open(vision, s_, err, true)) return false;
    }
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    te_size_ = std::make_unique<QiText>();
    dit_size_ = std::make_unique<QiDit>();
    vae_size_ = std::make_unique<QiVae>();
    if (!(te_size_->open(text, s_, err, true) && dit_size_->open(dit, s_, err, true) &&
          vae_size_->open(vae, s_, err, true)))
        return false;
    // the budget a request plans with: what was free before the parts, less a margin, within the small-card limit the
    // engine takes too
    auto_budget_ = vram_cap(free);
    return true;
}

void QiPipeline::ensure_open() {
    if (dit_ && vae_ && te_) return;
    std::string err;
    te_ = std::make_unique<QiText>();
    dit_ = std::make_unique<QiDit>();
    vae_ = std::make_unique<QiVae>();
    if (!(te_->open(text_path_, s_, err) && dit_->open(dit_path_, s_, err) && vae_->open(vae_path_, s_, err))) {
        te_.reset();
        dit_.reset();
        vae_.reset();
        throw std::runtime_error(err);
    }
    dit_->set_block_budget(dit_->block_min());   // nothing resident until a request plans its budget
    te_->set_layer_budget(te_->layer_min());
    dit_set_ = dit_->block_min();
    te_set_ = te_->layer_min();
    mlp_set_ = -1;
}

uint64_t QiPipeline::held() const {
    if (!(dit_ && vae_ && te_)) return 0;
    return dit_->weight_bytes() + dit_->lora_bytes() + vae_->weight_bytes() + kScratch + dit_->blocks_vram() +
           te_->layers_vram() + dit_->activations_held() + vae_->scratch_held();
}

void QiPipeline::fit(uint64_t reserve) {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    const uint64_t have = held() + (uint64_t) free;
    set_budget(have > reserve ? have - reserve : 0);
}

QiNeed QiPipeline::need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                        const std::vector<LoraUse>& loras) {
    const QiDit& dit = *dit_size_;
    const QiText& te = *te_size_;
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
    n.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    n.free = free;
    n.cap = vram_cap(total);
    const uint64_t room = n.cap > n.least ? n.cap - n.least : 0;
    n.host = kv == 0 ? 0 : kv_all - std::min<uint64_t>(room / kv, (uint64_t) dit.blocks()) * kv;
    return n;
}

void QiPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    if (!(dit_ && vae_ && te_)) return;   // released: the next request plans with it
    const QiPlan p = plan(0, 0, 0);   // shrink at once what no longer fits (a grown budget is taken at the next request)
    (void) p;
}

void QiPipeline::release(bool all) {
    if (!(dit_ && vae_ && te_)) return;
    if (all) {   // every buffer the parts hold goes with them; the next request opens them again
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
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

void QiPipeline::use_loras(const QiRequest& r) {
    std::string err;
    std::vector<LoraUse> dit, te;
    lora_split(r.loras, dit, te);
    dit_->set_lora_bake(lora_bake_);
    if (!dit_->set_loras(dit, err) || !te_->set_loras(te, err)) throw LoraRejected(err);
    if (te_->lora_key() != te_lora_) {   // the states kept were the encoder's with the other adapters
        last_ids_.clear();
        last_neg_ids_.clear();
        te_lora_ = te_->lora_key();
    }
}

void QiPipeline::te_layers() {
    if (te_set_ == te_want_) return;
    te_->set_layer_budget(te_want_);
    te_set_ = te_want_;
}

QiPlan QiPipeline::plan(int64_t T, int64_t h, int64_t w, int64_t n_ids, int64_t n_neg_pre) {
    QiPlan p;
    p.budget = budget_ > 0 ? budget_ : auto_budget_;
    // the prefix's rows stay through the sampling and the decode (an edit's: its references' tokens too)
    const int64_t n_pre = T > 0 ? T - h * w : 0;
    // the adapters' A and B are the fixed weights' (in VRAM whatever the plan does with the blocks; their scratch of a
    // step's rows is among the activations, activation_bytes).  Bound already: what this plan puts beside them
    p.lora = dit_->lora_bytes();
    p.lora_n = dit_->lora_count();
    p.fixed = dit_->weight_bytes() + p.lora + vae_->weight_bytes() + kScratch + dit_->prefix_row_bytes(n_pre) +
              dit_->prefix_row_bytes(n_neg_pre);
    // the activations: guidance's both sequences side by side after the first step (forward_pair)
    if (T > 0 && n_neg_pre > 0) T = qi_attn_rows(T) + qi_attn_rows(n_neg_pre + h * w);
    if (p.budget <= p.fixed) throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                                      " MiB) is below the weights that always stay");
    const uint64_t avail = p.budget - p.fixed, dit_min = dit_->block_min(), te_min = te_->layer_min();
    // the DiT's activations, the MLP in bands of rows when the whole takes more than a third of what is there
    int64_t mlp = 0;
    p.activations = T > 0 ? dit_->activation_bytes(T) : 0;
    for (int64_t rows : {4096, 2048, 1024, 512}) {
        if (T == 0 || p.activations * 3 <= avail) break;
        mlp = rows;
        p.activations = dit_->activation_bytes(T, rows);
    }
    // the room after the layers: the DiT's activations, and later the VAE's least (they take it in turn)
    // the encoder's buffers too: they come and go before the DiT's (generate frees them after the encode)
    const uint64_t work = std::max({p.activations, vae_->scratch_min(), te_->activation_bytes(n_ids)});
    if (avail < work + dit_min + te_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image");
    // the prefix cache first: a block's that is not in VRAM is copied in at every step with the compute waiting (the
    // blocks' weights stream beside it); then the DiT's blocks (8 or more uses an image), then the encoder's layers
    const uint64_t kv = dit_->prefix_kv_bytes(n_pre) + dit_->prefix_kv_bytes(n_neg_pre),
                   spare = avail - work - dit_min - te_min;
    p.resident_kv = kv > 0 ? (int) std::min<uint64_t>(spare / kv, (uint64_t) dit_->blocks()) : dit_->blocks();
    p.prefix_kv = (uint64_t) p.resident_kv * kv;
    dit_->set_prefix_vram_blocks(p.resident_kv);
    p.blocks = clampu(avail - work - te_min - p.prefix_kv, dit_min, dit_->block_bytes());
    p.layers = clampu(avail - work - p.blocks - p.prefix_kv, te_min, te_->layer_bytes());
    p.mlp_rows = mlp;
    if (p.blocks != dit_set_) {
        dit_->set_block_budget(p.blocks);
        dit_set_ = p.blocks;
    }
    // the encoder's layers: fewer at once (their room is the DiT's now), more only when it encodes (te_layers): the
    // decode gives them to the VAE, and a prompt whose states are kept does not run it (RX 9060 XT: 163 ms an image
    // to take 6300 MiB back for nothing)
    te_want_ = p.layers;
    if (p.layers < te_set_) {
        te_->set_layer_budget(p.layers);
        te_set_ = p.layers;
    }
    if (T > 0 && mlp != mlp_set_) {
        dit_->set_mlp_rows(mlp);
        mlp_set_ = mlp;
    }
    if (T == 0) dit_->drop_activations();   // a budget change between requests: the activations come back on use
    p.resident_blocks = dit_->resident_blocks();
    p.resident_layers = te_->resident_layers();
    // the VAE: what the layers leave, the DiT's activations' share included (freed if it needs it), in tiles when even
    // that is short
    const uint64_t left = avail - p.blocks - p.layers - p.prefix_kv;
    p.vae = left;
    p.vae_tiled = h > 0 && vae_->scratch_bytes(h, w) > left;
    return p;
}

void QiPipeline::edit_refs(const QiRequest& r, std::vector<int32_t>& ids, std::vector<QiVisionOut>& vouts,
                           std::vector<QiText::Image>& te_images, std::vector<QiDit::Ref>& dit_refs,
                           std::vector<float*>& ref_z, std::vector<int32_t>& neg_ids,
                           std::vector<QiText::Image>& neg_te, std::vector<QiDit::Ref>& neg_refs, bool vision) {
    if (vision_path_.empty()) throw std::invalid_argument("image engine: an edit needs the vision encoder (--vision)");
    if (std::count(r.ids.begin(), r.ids.end(), kImagePad) != (int64_t) r.refs.size())
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
        vision_ = std::make_unique<QiVision>();
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
        // resized as stable-diffusion.cpp resizes it (nearest), 0 .. 1; the vision encoder's input over white (neither
        // needed for kept states and a kept latent)
        std::vector<float> rgba, rgb;
        if (vision || !same) {
            rgba.resize((size_t) (4 * rh * rw));
            rgb.resize((size_t) (3 * rh * rw));
        }
        for (int64_t y = 0; y < (vision || !same ? rh : 0); ++y)
            for (int64_t x = 0; x < rw; ++x) {
                const uint8_t* px = f.rgba.data() + ((y * f.height / rh) * f.width + x * f.width / rw) * 4;
                const float a = (float) px[3] / 255.f;
                for (int c = 0; c < 4; ++c) rgba[(size_t) ((c * rh + y) * rw + x)] = (float) px[c] / 255.f;
                for (int c = 0; c < 3; ++c)
                    rgb[(size_t) ((c * rh + y) * rw + x)] = 2.f * ((float) px[c] / 255.f * a + 1.f - a) - 1.f;
            }
        if (vision) {
            vouts.push_back(vision_->encode(rgb.data(), rw, rh));
        } else {   // the kept states' references: their vision outputs' shapes only (no rows: the text is not encoded)
            QiVisionOut v;
            const QiVisionOut& was = last_vision_[vouts.size()];
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
        for (const QiVisionOut& v : vouts) {
            QiVisionOut s;
            s.rows = v.rows;
            s.dim = v.dim;
            s.grid_h = v.grid_h;
            s.grid_w = v.grid_w;
            last_vision_.push_back(std::move(s));
        }
    }
    // the ids with each pad widened to its reference's tokens: the prompt's, and guidance's negative prompt's
    auto widen = [&](const std::vector<int32_t>& src, std::vector<int32_t>& out, std::vector<QiText::Image>& te,
                     std::vector<QiDit::Ref>& refs) {
        size_t k = 0;
        for (const int32_t id : src) {
            if (id != kImagePad) {
                out.push_back(id);
                continue;
            }
            if (k >= vouts.size())
                throw std::invalid_argument("image engine: the negative prompt's image pads and the references differ");
            const QiVisionOut& v = vouts[k];
            QiText::Image im;
            im.index = (int64_t) out.size();
            im.rows = v.rows;
            im.grid_h = v.grid_h;
            im.grid_w = v.grid_w;
            im.main = v.main.data();
            for (const auto& d : v.deepstack) im.deepstack.push_back(d.data());
            te.push_back(im);
            QiDit::Ref f;
            f.slot = im.index;
            f.slots = v.rows;
            f.w = grids[k].first / 16;
            f.h = grids[k].second / 16;
            f.latent = ref_z[k];
            refs.push_back(f);
            out.insert(out.end(), (size_t) v.rows, kImagePad);
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
    const int64_t h = r.height / 16, w = r.width / 16;
    const auto t0 = clock::now();
    ensure_open();
    use_loras(r);   // before the plan (their bytes are fixed) and the context (a set that changed drops it)
    // an edit: its references through the vision encoder and the VAE's encoder, their pads widened to their tokens
    std::vector<int32_t> ids;
    std::vector<QiVisionOut> vouts;
    std::vector<QiText::Image> te_images;
    std::vector<QiDit::Ref> dit_refs;
    std::vector<float*> ref_z;   // the references' latents (kept_'s)
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs a base image (an init image or "
                                    "a reference)");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    const std::vector<float> sigmas = strength_sigmas(r.sigmas.empty() ? turbo_sigmas() : r.sigmas, r.strength);
    LatentStart ls;
    const bool guided = r.cfg > 1.0f && !r.neg_ids.empty();
    std::vector<int32_t> neg_ids;   // an edit's negative prompt, its pads widened
    std::vector<QiText::Image> neg_te;
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
        last_states_ = r.refs.empty() ? te_->encode(r.ids) : te_->encode(all_ids, te_images);
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
            last_neg_states_ = r.refs.empty() ? te_->encode(r.neg_ids) : te_->encode(neg_all, neg_te);
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
    float *x = nullptr, *rgba = nullptr, *x0 = nullptr;
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
        // partial images (r.preview): the step's x0 decoded in the room the VAE has beside the DiT now, nothing given
        // up for it; a step whose decode does not fit there is left out
        int shown = 0;
        auto peek = [&](int done, const std::function<void(float*)>& make) {
            if (!r.on_preview || std::find(r.preview.begin(), r.preview.end(), done) == r.preview.end()) return;
            const uint64_t room = p.vae > p.activations ? p.vae - p.activations : 0;
            if (x0 == nullptr && !gpu::alloc_device(&x0, (size_t) n * 4)) return;
            make(x0);
            try {
                vae_->decode(x0, h, w, rgba, room);
            } catch (const std::exception&) {
                vae_->drop_scratch();
                return;
            }
            vae_->drop_scratch();
            gpu::copy(out.data(), rgba, out.size() * 4);
            std::vector<uint8_t> img = rgba8(out, HW);
            if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
            r.on_preview(shown++, done, img);
        };
        dit_->sample(x, h, w, sigmas, step, ls.after(s_), guided ? r.cfg : 1.0f, peek, &r.guide);
        dit_->drop_host_prefix();
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it
        uint64_t vae_room = p.vae > p.activations ? p.vae - p.activations : 0;
        if (vae_->scratch_bytes(h, w) > vae_room) {
            dit_->drop_activations();
            vae_room = p.vae;
        }
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
        dit_->drop_host_prefix();
        gpu::free(x);
        gpu::free(x0);
        gpu::free(rgba);
        throw;
    }
    gpu::free(x);
    gpu::free(x0);
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
    const int64_t HW = r.width * r.height;
    std::vector<float> rgba((size_t) (4 * HW));
    for (int64_t q = 0; q < HW; ++q)
        for (int c = 0; c < 4; ++c) rgba[(size_t) (c * HW + q)] = (float) px[(size_t) (q * 4 + c)] / 255.f;
    float* d = nullptr;
    if (!gpu::alloc_device(&d, rgba.size() * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    gpu::copy(d, rgba.data(), rgba.size() * 4);
    const bool ok = vae_->encode(d, r.height, r.width, z0);
    gpu::free(d);
    vae_->drop_scratch();
    if (!ok) throw std::runtime_error("image engine: the VAE's encoder failed on the base image");
    // kept for the next request of the same image (1 MiB at 1024 x 1024); without the room, not kept
    if (kept_.base_z != nullptr) gpu::free(kept_.base_z);
    kept_.base_z = nullptr;
    kept_.base.clear();
    if (gpu::alloc_device(&kept_.base_z, (size_t) n * 4)) {
        gpu::copy_async(kept_.base_z, z0, (size_t) n * 4, s_);
        kept_.base = key;
    }
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
