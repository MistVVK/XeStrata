// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_pipeline.cpp - see include/strata/image/anima_pipeline.hpp.

#include "sigmas.hpp"
#include "strata/image/anima_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/flow_sampler.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/platform/env.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace anima_pipeline_detail {

constexpr int kSteps = 30;          // the model card's 30 to 50 for the base and aesthetic models
constexpr float kShift = 3.0f;      // the model's sampling shift
constexpr int64_t kMaxTokens = 4096;
constexpr int32_t kPad = 151643;    // Qwen3's <|endoftext|>: the one token of an empty prompt
constexpr int32_t kT5End = 1;       // T5's </s>

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

// z [L][2h][2w] (the VAE's latent) from the tokens [h w][4 L] (channel c of the patch's pixel (m, n) at c * 4 + m * 2
// + n), and back
void tokens_to_latent(const float* x, int64_t h, int64_t w, int64_t L, float* z, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (h * w * 4 * L)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (4 * L), j = i - t * 4 * L, c = j / 4, m = (j / 2) % 2, n = j % 2;
        const int64_t y = t / w, xx = t - y * w;
        z[(c * 2 * h + 2 * y + m) * 2 * w + 2 * xx + n] = x[i];
    });
}

void latent_to_tokens(const float* z, int64_t h, int64_t w, int64_t L, float* x, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (h * w * 4 * L)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (4 * L), j = i - t * 4 * L, c = j / 4, m = (j / 2) % 2, n = j % 2;
        const int64_t y = t / w, xx = t - y * w;
        x[i] = z[(c * 2 * h + 2 * y + m) * 2 * w + 2 * xx + n];
    });
}

}  // namespace anima_pipeline_detail

using namespace anima_pipeline_detail;

std::vector<float> anima_sigmas(int n, float shift) {
    constexpr int kTrain = 1000;
    std::vector<float> out;
    for (int i = 0; i < n; ++i) {
        const int k = (int) ((double) i * ((double) kTrain / (double) n));
        const float t = (float) (kTrain - k) / (float) kTrain;
        out.push_back(sigma_shift(t, shift));
    }
    out.push_back(0.f);
    return out;
}

bool AnimaPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                         int q8) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    q8_ = q8;
    dit_size_ = std::make_unique<AnimaDit>();
    dit16_size_ = std::make_unique<AnimaDit>();
    vae_size_ = std::make_unique<QiVae>();
    te_size_ = std::make_unique<TextEncoder>();
    if (!(dit_size_->open(dit, s_, err, true, true) && dit16_size_->open(dit, s_, err, true, false) &&
          vae_size_->open(vae, s_, err, true) && te_size_->open(text, kMaxTokens, err, UINT64_MAX, true)))
        return false;
    if (vae_size_->latent_channels() != dit_size_->latent_channels()) {
        err = "image engine: Anima's VAE is Wan 2.1's of " + std::to_string(dit_size_->latent_channels()) +
              " channels (" + vae + " is not)";
        return false;
    }
    if (te_size_->hidden() != dit_size_->text_dim()) {
        err = "image engine: Anima's text encoder is Qwen3-0.6B-Base, " + std::to_string(dit_size_->text_dim()) +
              " wide (" + text + " is " + std::to_string(te_size_->hidden()) + ")";
        return false;
    }
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    auto_budget_ = vram_cap(free);
    note_others();
    return true;
}

bool AnimaPipeline::warm_up_request(QiRequest& r) const {
    r.ids = {kPad};
    r.t5_ids = {kT5End};
    r.t5_weights = {1.f};
    r.width = r.height = 256;
    r.steps = 1;
    r.sampler = "euler";
    return true;
}

void AnimaPipeline::use_loras(const QiRequest& r) {
    std::string err;
    const std::string before = dit_->lora_key();
    std::vector<LoraUse> dit;
    lora_split(r.loras, dit, te_loras_);
    dit_->set_lora_bake(lora_bake_);
    if (!dit_->set_loras(dit, err)) {
        ctx_key_[0] = {};
        ctx_key_[1] = {};
        throw LoraRejected(err);
    }
    if (te_ && !te_->set_loras(te_loras_, err)) {
        ctx_key_[0] = {};
        ctx_key_[1] = {};
        te_loras_now(te_->lora_key());
        throw LoraRejected(err);
    }
    if (dit_->lora_key() != before) {   // the contexts were the adapter's with the other adapters' weights
        ctx_key_[0] = {};
        ctx_key_[1] = {};
    }
    te_loras_now(lora_key_of(te_loras_));   // the states kept were the encoder's with the other adapters
}

void AnimaPipeline::ensure_open(bool q8) {
    if (dit_ && dit_requested_q8_ != q8) {
        gpu::stream_sync(s_);
        dit_.reset();
    }
    if (!q8) dit_host_.clear();
    std::string err;
    if (!vae_) {
        auto v = std::make_unique<QiVae>();
        if (!v->open(vae_path_, s_, err)) throw std::runtime_error(err);
        vae_ = std::move(v);
    }
    if (!dit_) {
        auto d = std::make_unique<AnimaDit>();
        const bool ok = d->open(dit_path_, s_, err, false, q8, &dit_host_);
        dit_host_.clear();
        if (!ok) throw std::runtime_error(err);
        d->set_block_budget(d->block_min());
        d->set_lora_block_map(block_map_);
        dit_ = std::move(d);
        dit_requested_q8_ = q8;
        dit_set_ = dit_->block_min();
        mlp_set_ = -1;
        ctx_key_[0] = {};
        ctx_key_[1] = {};
    }
}

// W8Choice::want_q8 on the FP16 blocks all resident beside the rest.  The encoder's share is all its layers and its
// activations at 512 ids: what plan() keeps it open with
bool AnimaPipeline::want_q8(int64_t rows, int n) {
    const AnimaDit& d16 = *dit16_size_;
    const uint64_t fixed = d16.weight_bytes() + d16.text_bytes() + vae_size_->weight_bytes() +
                           te_size_->layer_bytes() + te_size_->activation_bytes(512);
    return w8_.want_q8(q8_, fixed + d16.block_bytes() + d16.activation_bytes(rows, n), budget(), rows, "Anima's DiT",
                       s_, [&](int64_t r) { return d16.time_forms(r); });
}

uint64_t AnimaPipeline::held() const {
    uint64_t b = 0;
    if (vae_) b += vae_->weight_bytes() + vae_->scratch_held();
    if (dit_) {
        b += dit_->weight_bytes() + dit_->lora_bytes() + dit_->scratch_bytes() + dit_->blocks_vram() +
             dit_->activations_held();
        if (dit_->text_loaded()) b += dit_->text_bytes();
    }
    if (te_) b += te_->layers_vram() + te_->activation_bytes(512);
    return b;
}

QiNeed AnimaPipeline::need(int64_t n_text, int64_t width, int64_t height,
                           const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                           const std::vector<LoraUse>& loras) {
    (void) refs;   // Anima has no edits
    const AnimaDit& dit = q8_ == 0 ? *dit16_size_ : *dit_size_;
    const AnimaDit& full = q8_ == 1 ? *dit_size_ : *dit16_size_;
    const QiVae& vae = *vae_size_;
    const int64_t s = vae.scale(), h = height / s, w = width / s, hw = (h / 2) * (w / 2);
    const int n = n_neg > 0 ? 2 : 1;
    const uint64_t lora = lora_gpu_bytes(loras, n * hw);
    uint64_t base = 0, fixed = vae.weight_bytes() + dit.text_bytes() + lora;
    if (mask) {
        fixed += LatentStart::bytes(dit.token_values() * hw, hw, true);
        base = vae.encode_scratch_bytes(height, width) + (uint64_t) (4 * height * width) * 4;
    }
    const int64_t T = std::max<int64_t>({n_text, n_neg, 1});
    const uint64_t te = te_size_->activation_bytes(T) + te_size_->layer_min();
    QiNeed nd;
    nd.least = fixed + dit.weight_bytes() + dit.scratch_bytes() +
               std::max({dit.activation_bytes(hw, n, 512) + dit.block_min(), te, vae.scratch_min(), base});
    nd.full = fixed + full.weight_bytes() + full.scratch_bytes() + full.block_bytes() + te_size_->layer_bytes() +
              te_size_->activation_bytes(T) + std::max({full.activation_bytes(hw, n), vae.scratch_bytes(h, w), base});
    vram_into(nd);
    return nd;
}

void AnimaPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    note_others();
    if (held() == 0 || !dit_) return;
    (void) plan(0, 1);
}

void AnimaPipeline::release(bool all) {
    if (held() == 0) return;
    gpu::stream_sync(s_);
    if (all) {
        te_.reset();
        if (dit_) {
            dit_->drop_loras();
            if (dit_->q8()) dit_host_ = dit_->take_host();
        }
        dit_.reset();
        vae_.reset();
        ctx_key_[0] = {};
        ctx_key_[1] = {};
        return;
    }
    if (dit_) {
        dit_->drop_blocks();
        dit_->drop_block_homes();
        dit_->drop_activations();
    }
    te_.reset();
    if (vae_) vae_->drop_scratch();
    dit_set_ = 0;
    mlp_set_ = -1;
}

QiPlan AnimaPipeline::plan(int64_t hw, int n, bool text) {
    QiPlan p;
    p.budget = budget();
    AnimaDit& d = *dit_;
    p.fixed = d.weight_bytes() + d.lora_bytes() + d.scratch_bytes() + (text || d.text_loaded() ? d.text_bytes() : 0) +
              vae_->weight_bytes();
    p.lora = d.lora_bytes();
    p.lora_n = d.lora_count();
    const uint64_t avail = plan_avail(p), dit_min = d.block_min();
    const int64_t mlp = mlp_bands(avail, hw > 0, p.activations,
                                  [&](int64_t rows) { return d.activation_bytes(hw, n, rows); });
    if (avail < p.activations + dit_min) plan_too_small(p);
    uint64_t te_need = te_ ? te_->layer_bytes() + te_->activation_bytes(512) : 0;
    if (te_ && avail < p.activations + d.block_bytes() + te_need) {   // the encoder goes: its states are made
        gpu::stream_sync(s_);
        te_.reset();
        te_need = 0;
    }
    p.blocks = clampu(avail - p.activations - te_need, dit_min, d.block_bytes());
    plan_blocks(d, p.blocks);
    if (te_) {
        p.layers = te_->layer_bytes();
        p.resident_layers = te_->resident_layers();
    }
    plan_rows(p, d, hw > 0, mlp);
    p.vae = avail - p.blocks - te_need;
    return p;
}

// the text encoder opened whole (Qwen3-0.6B: 1.2 GiB in FP16) in what the budget leaves, the DiT's blocks and
// activations giving way where it does not fit
void AnimaPipeline::open_text() {
    if (te_) return;
    auto room = [&] { const uint64_t h = held(); return budget() > h ? budget() - h : 0; };
    const uint64_t want = te_size_->layer_bytes() + te_size_->activation_bytes(kMaxTokens);
    if (room() < want && dit_) {
        dit_->drop_blocks();
        dit_->drop_activations();
        dit_set_ = 0;
        mlp_set_ = -1;
    }
    auto t = std::make_unique<TextEncoder>();
    std::string err;
    gpu::stream_sync(s_);   // the encoder's own stream: what the engine freed on its stream first
    if (!t->open(text_path_, kMaxTokens, err)) throw std::runtime_error(err);
    if (!t->set_loras(te_loras_, err)) throw LoraRejected(err);
    const uint64_t r = room();
    t->set_layer_budget(std::max(t->layer_min(), std::min(t->layer_bytes(), r > want ? r : t->layer_min())));
    te_ = std::move(t);
}

void AnimaPipeline::encode(const std::vector<int32_t>& ids, std::vector<float>& out) {
    if ((int64_t) ids.size() > kMaxTokens)
        throw std::invalid_argument("image engine: Anima's prompt is longer than 4096 tokens");
    open_text();
    gpu::stream_sync(s_);
    std::string err;
    auto states = te_->encode(ids, err);
    te_->drop_activations();
    if (states.empty()) throw std::runtime_error("image engine: Anima's text encoder: " + err);
    out = std::move(states);
}

void AnimaPipeline::encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls) {
    const int64_t s = vae_->scale(), h = r.height / s / 2, w = r.width / s / 2;
    const uint64_t more = need(0, r.width, r.height, {}, true).least - need(0, r.width, r.height).least;
    if (held() + more > budget()) release(false);
    // RGBA [4][H][W] in 0 .. 1 over white, the alpha's plane the opaque one (img_start.hpp's image_planes);
    // Wan 2.1's VAE encoder reads its first three channels
    const std::vector<float> px =
        image_planes(base, r.width, r.height, 4, PlaneAlpha::White, PlaneRange::Unit);
    float *rgba = nullptr, *z = nullptr;
    const int64_t L = dit_->latent_channels();
    if (!gpu::alloc_device(&rgba, px.size() * 4) || !gpu::alloc_device(&z, (size_t) (L * 4 * h * w) * 4)) {
        gpu::free(rgba);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    try {
        float* z0 = ls.base(dit_->token_values() * h * w);
        gpu::copy(rgba, px.data(), px.size() * 4);
        if (!vae_->encode(rgba, r.height, r.width, z)) throw std::runtime_error("image engine: the VAE has no encoder");
        latent_to_tokens(z, h, w, L, z0, s_);
        gpu::stream_sync(s_);
    } catch (...) {
        gpu::free(rgba);
        gpu::free(z);
        vae_->drop_scratch();
        throw;
    }
    gpu::free(rgba);
    gpu::free(z);
    vae_->drop_scratch();
}

void AnimaPipeline::sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, int n, float cfg,
                           bool er, float shift, uint64_t seed, const std::function<bool(int, int)>& step,
                           const std::function<void(float*, float)>& after,
                           const std::function<void(int, const std::function<void(float*)>&)>& peek,
                           const GuidanceSpec& spec) {
    AnimaDit& d = *dit_;
    for (const int b : spec.slg_layers)
        if (b < 0 || b >= d.blocks())
            throw std::invalid_argument("image engine: skip layer guidance's block " + std::to_string(b) +
                                        " is not one of the model's " + std::to_string(d.blocks()));
    Guidance g(spec, s_, h * w * d.token_values(), d.token_values(), true, cfg, sigmas);
    flow_sample(s_, [&](const float* xx, float sigma, int nn, float* out) { d.forward(xx, sigma, h, w, nn, out); }, x,
                h * w * d.token_values(), n, sigmas, cfg, er, shift, seed, step, after, peek, &g,
                [&](const float* xx, float sigma, float* out) {
                    d.forward(xx, sigma, h, w, 1, out, &g.skip_layers());
                });
}

std::vector<uint8_t> AnimaPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                             const std::function<bool(int, int)>& step,
                                             const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if (!r.refs.empty()) throw std::invalid_argument("image engine: Anima takes no reference images");
    if (r.t5_ids.empty() || r.t5_ids.size() != r.t5_weights.size())
        throw std::invalid_argument("image engine: Anima's prompt needs its T5 ids and a weight for each");
    const bool guided = r.cfg != 1.0f && !r.neg_t5_ids.empty();
    if (guided && r.neg_t5_ids.size() != r.neg_t5_weights.size())
        throw std::invalid_argument("image engine: a weight for each of the negative prompt's T5 ids");
    if (!r.sampler.empty() && r.sampler != "euler" && r.sampler != "er_sde")
        throw std::invalid_argument("image engine: Anima's sampler is euler or er_sde");
    const bool er = r.sampler != "euler";
    const int64_t s = vae_size_->scale(), h = r.height / s / 2, w = r.width / s / 2, HW = r.width * r.height;
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs an init image");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    const std::vector<float> full = !r.sigmas.empty() ? r.sigmas
                                                      : anima_sigmas(r.steps > 0 ? (int) r.steps : kSteps,
                                                                     r.shift > 0 ? r.shift : kShift);
    const std::vector<float> sigmas = strength_sigmas(full, r.strength);
    const auto t0 = clock::now();
    const int n = guided ? 2 : 1;
    ensure_open(want_q8(h * w, n));
    AnimaDit& d = *dit_;
    use_loras(r);
    LatentStart ls;
    if (base != nullptr) {
        if (stage) stage("refs");
        encode_base(r, *base, ls);
    }
    if (stage) stage("text");
    // STRATA_IMAGE_PARTS: each part's time on stderr (to compare the parts with another program's), the prompt's
    // states and contexts made again every image
    const bool parts = strata::env_get("IMAGE_PARTS") != nullptr;
    if (parts) {
        last_ids_.clear();
        last_neg_ids_.clear();
        ctx_key_[0] = {};
        ctx_key_[1] = {};
    }
    const auto te0 = clock::now();
    bool cached = true;
    const std::vector<int32_t> ids = r.ids.empty() ? std::vector<int32_t>{kPad} : r.ids;
    if (ids != last_ids_) {
        encode(ids, last_states_);
        last_ids_ = ids;
        cached = false;
    }
    std::vector<int32_t> neg = r.neg_ids.empty() ? std::vector<int32_t>{kPad} : r.neg_ids;
    if (guided && neg != last_neg_ids_) {
        encode(neg, last_neg_states_);
        last_neg_ids_ = neg;
        cached = false;
    }
    const auto te1 = clock::now();
    const CtxKey k0{last_states_, r.t5_weights, r.t5_ids};
    const CtxKey k1 = guided ? CtxKey{last_neg_states_, r.neg_t5_weights, r.neg_t5_ids} : CtxKey{};
    const bool set0 = !(k0 == ctx_key_[0]), set1 = guided && !(k1 == ctx_key_[1]);
    const int64_t C = d.text_dim();
    QiPlan p = plan(h * w, n, set0 || set1);
    if (vae_->scratch_held() > p.vae - p.activations) vae_->drop_scratch();
    std::vector<AnimaDit::ContextIn> sets;
    if (set0) sets.push_back({0, last_states_.data(), (int64_t) last_states_.size() / C, &r.t5_ids, &r.t5_weights});
    if (set1)
        sets.push_back({1, last_neg_states_.data(), (int64_t) last_neg_states_.size() / C, &r.neg_t5_ids,
                        &r.neg_t5_weights});
    if (!sets.empty()) d.set_contexts(sets);
    if (set0) ctx_key_[0] = k0;
    if (set1) ctx_key_[1] = k1;
    if (!guided && d.context_rows(1) > 0) {
        d.set_context(1, nullptr, 0, {}, {});
        ctx_key_[1] = {};
    }
    const auto te2 = clock::now();
    p = plan(h * w, n);
    if (d.text_loaded() && p.blocks < d.block_bytes()) {
        d.drop_text();
        p = plan(h * w, n);
    }
    p.text_cached = cached;
    const auto t1 = clock::now();
    // the seed's noise as PyTorch's CPU randn makes it (of the latent [16][2h][2w]), in the tokens' order
    const int64_t L = d.latent_channels(), TV = d.token_values(), N = TV * h * w;
    const auto noise = torch_cpu_randn(r.seed, (uint64_t) N);
    std::vector<float> tokens((size_t) N);
    noise_to_tokens(noise.data(), L, 2 * h, 2 * w, 2, tokens.data());
    float *x = nullptr, *rgba = nullptr, *z = nullptr;
    if (!gpu::alloc_device(&x, (size_t) N * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4) ||
        !gpu::alloc_device(&z, (size_t) N * 4)) {
        gpu::free(x);
        gpu::free(rgba);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    std::vector<float> out((size_t) (4 * HW));
    std::vector<uint8_t> mask_px;
    try {
        if (mask) {
            mask_px = resize_nearest(r.mask, r.width, r.height);
            ls.set_mask(request_latent_mask(r, mask_px, r.width, r.height, 2 * s), tokens, TV, h * w, true);
        }
        gpu::copy(x, tokens.data(), (size_t) N * 4);
        if (r.strength < 1.0) ls.start(x, sigmas[0], s_);
        if (stage) stage("sampling");
        // partial images (r.preview)
        PreviewDecode preview(
            r, p, N,
            [&](float* x0, uint64_t room) {
                tokens_to_latent(x0, h, w, L, z, s_);
                vae_->decode(z, 2 * h, 2 * w, rgba, room);
            },
            [&] { vae_->drop_scratch(); },
            [&] {
                gpu::copy(out.data(), rgba, out.size() * 4);
                std::vector<uint8_t> img = rgba8(out, HW);
                if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
                return img;
            });
        auto peek = [&](int done, const std::function<void(float*)>& make) { preview(done, make); };
        sample(x, h, w, sigmas, n, guided ? r.cfg : 1.0f, er, r.shift > 0 ? r.shift : kShift, r.seed, step, ls.after(s_), peek, r.guide);
        preview.release();
        const auto t2 = clock::now();
        if (stage) stage("decode");
        uint64_t room = decode_room(p, d, vae_->scratch_bytes(2 * h, 2 * w));
        if (vae_->scratch_bytes(2 * h, 2 * w) > room && te_) {
            room += te_->layers_vram();
            te_.reset();
        }
        tokens_to_latent(x, h, w, L, z, s_);
        vae_->decode(z, 2 * h, 2 * w, rgba, room);
        p.vae_tiled = vae_->scratch_bytes(2 * h, 2 * w) > room;
        if (vae_->scratch_bytes(2 * h, 2 * w) > p.vae) vae_->drop_scratch();
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (parts) {
            const auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
            std::fprintf(stderr, "image engine parts: text_encoder_s=%.3f adapter_s=%.3f step_s=%.4f steps=%d "
                                 "contexts=%d decode_s=%.3f\n", sec(te0, te1), sec(te1, te2),
                         sec(t1, t2) / (double) std::max<size_t>(1, sigmas.size() - 1), (int) sigmas.size() - 1, n,
                         sec(t2, t3));
        }
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        gpu::free(x);
        gpu::free(rgba);
        gpu::free(z);
        throw;
    }
    gpu::free(x);
    gpu::free(rgba);
    gpu::free(z);
    if (plan_out != nullptr) *plan_out = p;
    std::vector<uint8_t> img = rgba8(out, HW);
    if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
    return img;
}

}  // namespace strata::image
