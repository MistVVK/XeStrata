// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_pipeline.cpp - see include/strata/image/mf_pipeline.hpp.

#include "sigmas.hpp"
#include "strata/image/mf_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/flow_sampler.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/qi_dit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kScratch = 384 * kMiB;   // the products' dequantization scratch (the DiT's and the encoder's)
constexpr int kTurboSteps = 4;              // Mage-Flow-Turbo's steps (its model card)
// the text encoder's longest prompt, the references' rows among it (its K/V cells, shared by its layers)
constexpr int64_t kTextTokens = 16384;
constexpr size_t kMaxRefs = 3;              // the edit models' training takes up to three references
constexpr int64_t kVlEdge = 384;            // the vision encoder's input: its long edge at most this (training's)
constexpr int64_t kVlFactor = 32, kVlMinPixels = 65536, kVlMaxPixels = 16777216;   // Qwen3-VL's processor

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

// the vision encoder's input size of a w x h reference: its long edge capped at kVlEdge (Python's round), then
// Qwen3-VL's smart_resize to multiples of kVlFactor within its pixel bounds; cw, ch the capped size
void vl_size(int64_t w, int64_t h, int64_t& cw, int64_t& ch, int64_t& vw, int64_t& vh) {
    cw = w;
    ch = h;
    if (std::max(w, h) > kVlEdge) {
        const double sc = (double) kVlEdge / (double) std::max(w, h);
        cw = std::max<int64_t>(1, (int64_t) std::nearbyint((double) w * sc));
        ch = std::max<int64_t>(1, (int64_t) std::nearbyint((double) h * sc));
    }
    const auto f = (double) kVlFactor;
    double bh = std::max(f, std::nearbyint((double) ch / f) * f), bw = std::max(f, std::nearbyint((double) cw / f) * f);
    if (bh * bw > (double) kVlMaxPixels) {
        const double beta = std::sqrt((double) ch * (double) cw / (double) kVlMaxPixels);
        bh = std::max(f, std::floor((double) ch / beta / f) * f);
        bw = std::max(f, std::floor((double) cw / beta / f) * f);
    } else if (bh * bw < (double) kVlMinPixels) {
        const double beta = std::sqrt((double) kVlMinPixels / ((double) ch * (double) cw));
        bh = std::ceil((double) ch * beta / f) * f;
        bw = std::ceil((double) cw * beta / f) * f;
    }
    vw = (int64_t) bw;
    vh = (int64_t) bh;
}

}  // namespace

// an edit's references as the parts take them: the widened prompts and their images for the text encoder, the
// references' latents for the DiT (device, [rows][channels])
struct MfPipeline::Edit {
    std::vector<core::VisionOut> vouts;
    std::vector<int32_t> ids, neg_ids;
    std::vector<TextEncoder::Picture> te, neg_te;
    std::vector<std::pair<int64_t, int64_t>> sizes;
    float* z = nullptr;
    int64_t rows = 0;
    ~Edit() { gpu::free(z); }
};

bool MfPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                      int q8, const std::string& edit_dit, const std::string& vision, bool keep) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    edit_path_ = edit_dit;
    vision_path_ = vision;
    q8_ = q8;
    keep_ = keep;
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    te_size_ = std::make_unique<TextEncoder>();
    dit_size_ = std::make_unique<MfDit>();
    dit16_size_ = std::make_unique<MfDit>();
    vae_size_ = std::make_unique<MfVae>();
    if (!(te_size_->open(text, kTextTokens, err, UINT64_MAX, true) && dit_size_->open(dit, s_, err, true, true) &&
          dit16_size_->open(dit, s_, err, true, false) && vae_size_->open(vae, s_, err, true)))
        return false;
    {   // the encoder's sizes, for a base image: a VAE file without one makes images but takes no base image
        std::string no;
        base_encoder_ = vae_size_->open_encoder(no, true);
    }
    if (!edit_path_.empty() || !vision_path_.empty()) {
        if (edit_path_.empty() || vision_path_.empty()) {
            err = "image engine: Mage-Flow's edits need both the edit DiT and the vision encoder";
            return false;
        }
        vision_size_ = std::make_unique<core::Vision>();
        if (!vision_size_->open(vision_path_, s_, err, true) || !vae_size_->open_encoder(err, true)) return false;
    }
    auto_budget_ = vram_cap(free);
    note_others();
    return true;
}

void MfPipeline::ensure_open() {
    if (vae_ && te_) return;
    std::string err;
    te_ = std::make_unique<TextEncoder>();
    vae_ = std::make_unique<MfVae>();
    gpu::stream_sync(s_);   // the encoder's own stream: what the engine freed on its stream first
    if (!(te_->open(text_path_, kTextTokens, err) && vae_->open(vae_path_, s_, err))) {
        te_.reset();
        vae_.reset();
        throw std::runtime_error(err);
    }
    te_->set_final_norm(true);
    te_->set_flat_positions(true);   // the reference's encoder: every token's position its index, an image's too
    te_->set_layer_budget(te_->layer_min());
    te_set_ = te_->layer_min();
    edit_active_ = false;   // the DiT is opened by the request, which knows its size
}

// 8 bits or FP16 for a request of `rows` image rows (an edit's references' included): FP16 when its blocks all fit
// the budget beside the weights that stay, the larger of the DiT's activations and the encoder's, and the encoder's
// layers, all of them: FP16 blocks that leave the encoder a few slots cost more than they give (RTX 4070, an edit:
// its text and image read 0.5 s slower, a second in all against the 8-bit DiT's 3.2)
bool MfPipeline::want_q8(int64_t rows, int n, int64_t max_text, int64_t n_ids) const {
    const MfDit& d16 = *dit16_size_;
    const uint64_t fixed = d16.weight_bytes() + vae_size_->weight_bytes() + kScratch +
                           (vision_size_ ? vae_size_->encoder_bytes() : 0);
    const uint64_t work = std::max(d16.activation_bytes(rows, n, max_text), te_size_->activation_bytes(n_ids));
    return w8_.want_q8(q8_, fixed + d16.block_bytes() + work + te_size_->layer_bytes(), budget(), rows,
                       "Mage-Flow's DiT", s_, [&](int64_t r) { return d16.time_forms(r); });
}

// the DiT for a generation or an edit: the other gives its VRAM back (its Q8_0 kept in RAM with keep_, else let go),
// this one opened when it is not, nothing of it resident until a request plans its budget
MfDit& MfPipeline::use_dit(bool edit, bool q8) {
    std::unique_ptr<MfDit>& want = edit ? edit_dit_ : dit_;
    std::unique_ptr<MfDit>& other = edit ? dit_ : edit_dit_;
    if (want && want->q8() != q8) {   // the other form for this request's budget: opened again
        gpu::stream_sync(s_);
        want.reset();
    }
    if (edit == edit_active_ && want) return *want;
    if (other) {
        other->drop_blocks();
        other->drop_activations();
        if (!keep_) other.reset();
    }
    edit_active_ = edit;
    if (!want) {
        std::string err;
        std::vector<std::vector<uint8_t>>& host = edit ? edit_host_ : dit_host_;
        if (!q8) host.clear();   // FP16 reads the file in place
        auto d = std::make_unique<MfDit>();
        const bool ok = d->open(edit ? edit_path_ : dit_path_, s_, err, false, q8, &host);
        host.clear();
        if (!ok) throw std::runtime_error(err);
        want = std::move(d);
    }
    want->set_block_budget(want->block_min());
    dit_set_ = want->block_min();
    mlp_set_ = -1;
    return *want;
}

uint64_t MfPipeline::held() const {
    if (!(vae_ && te_)) return 0;
    uint64_t b = vae_->weight_bytes() + kScratch + te_->layers_vram() + vae_->scratch_held() +
                 (vision_ ? vision_->weight_bytes() : 0);
    for (const MfDit* d : {dit_.get(), edit_dit_.get()})
        if (d != nullptr) b += d->weight_bytes() + d->lora_bytes();
    if (const MfDit* d = active()) b += d->blocks_vram() + d->activations_held();
    return b;
}

void MfPipeline::use_loras(const QiRequest& r) {
    std::string err;
    std::vector<LoraUse> dit_part, te;
    lora_split(r.loras, dit_part, te);
    dit().set_lora_bake(lora_bake_);
    if (!dit().set_loras(dit_part, err) || !te_->set_loras(te, err)) throw LoraRejected(err);
    te_loras_now(te_->lora_key());   // the states kept were the encoder's with the other adapters
}

QiNeed MfPipeline::need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                        const std::vector<LoraUse>& loras) {
    const MfDit& dit = q8_ == 0 ? *dit16_size_ : *dit_size_;   // the least: Q8_0's, unless FP16 is asked for
    const MfDit& full = q8_ == 1 ? *dit_size_ : *dit16_size_;  // all resident: FP16's, unless Q8_0 is
    const TextEncoder& te = *te_size_;
    const MfVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16;
    int64_t hw = h * w;
    uint64_t edit = 0, fixed = dit.weight_bytes() + vae.weight_bytes() + kScratch;
    if (!refs.empty() && vision_size_) {   // the references: each a target's worth of latent rows, its image tokens in
        fixed += vae.encoder_bytes();      // the prompts; the vision encoder and the VAE's encoder while they run
        for (const auto& [rw, rh] : refs) {
            int64_t cw = 0, ch = 0, vw = 0, vh = 0;
            vl_size(rw, rh, cw, ch, vw, vh);
            const int64_t tokens = (vh / kVlFactor) * (vw / kVlFactor) - 1;
            n_text += tokens;
            if (n_neg > 0) n_neg += tokens;
            edit = std::max(edit, vision_size_->weight_bytes() + vision_size_->scratch_bytes((vh / 16) * (vw / 16)));
        }
        edit = std::max(edit, vae.encode_scratch_bytes(h, w) + (uint64_t) (3 * height * width) * 4);
        hw += (int64_t) refs.size() * h * w;
    }
    if (mask && base_encoder_) {   // a base image (img_start.hpp): its encode, and its latent (and a mask's) kept
        if (refs.empty() || !vision_size_) fixed += vae.encoder_bytes();
        edit = std::max(edit, vae.encode_scratch_bytes(h, w) + (uint64_t) (3 * height * width) * 4);
        fixed += LatentStart::bytes(dit.latent_channels() * h * w, h * w, true);
    }
    const int64_t max_text = std::max(n_text, n_neg);
    const int n = n_neg > 0 ? 2 : 1;
    const uint64_t te_act = te.activation_bytes(std::max(n_text, n_neg) + 64);   // the system turn's ids too
    // the adapters' A and B and their scratch (the most rows one product has: a context's image rows plus its text)
    const uint64_t lora = lora_gpu_bytes(loras, n * (hw + max_text));
    QiNeed nd;
    nd.least = fixed + lora +
               std::max({dit.activation_bytes(hw, n, max_text, 512), vae.least_scratch_bytes(h, w), te_act, edit}) +
               dit.block_min() + te.layer_min();
    nd.full = fixed - dit.weight_bytes() + full.weight_bytes() + full.block_bytes() + te.layer_min() + lora +
              std::max({full.activation_bytes(hw, n, max_text), vae.scratch_bytes(h, w), te_act, edit});
    vram_into(nd);
    return nd;
}

std::vector<float> MfPipeline::text(const std::vector<int32_t>& ids, const std::vector<TextEncoder::Picture>& pictures) {
    if ((int64_t) ids.size() > kTextTokens)
        throw std::invalid_argument("image engine: the prompt and its references are longer than " +
                                    std::to_string(kTextTokens) + " tokens");
    gpu::stream_sync(s_);
    std::string err;
    std::vector<float> out = te_->encode(ids, err, pictures);
    if (out.empty()) throw std::runtime_error(err);
    return out;
}

void MfPipeline::te_layers() {
    if (te_want_ == 0 || te_set_ == te_want_) return;   // (no plan yet: the encoder as it was opened)
    te_->set_layer_budget(te_want_);
    te_set_ = te_want_;
}

void MfPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    note_others();
    if (held() == 0 || active() == nullptr) return;
    (void) plan(0, 1, 0, 0);   // shrink at once what no longer fits
}

void MfPipeline::release(bool all) {
    if (held() == 0) return;
    gpu::stream_sync(s_);
    vision_.reset();
    if (all) {
        te_.reset();
        // the RAM setup counts for them: their blocks kept quantized for the next open (the other DiT's only with keep_,
        // which is when it is still open)
        if (dit_) dit_->drop_loras();   // the adapters' A and B too (they go back on the next request's, from its files)
        if (edit_dit_) edit_dit_->drop_loras();
        if (dit_ && dit_->q8()) dit_host_ = dit_->take_host();
        if (edit_dit_ && edit_dit_->q8()) edit_host_ = edit_dit_->take_host();
        dit_.reset();
        edit_dit_.reset();
        vae_.reset();
        edit_active_ = false;
        return;
    }
    if (MfDit* d = active()) {
        d->drop_blocks();
        d->drop_block_homes();   // the RAM the streamed layers' copies hold, back for the other engines
        d->drop_activations();
    }
    te_->drop_layers();
    te_->drop_layer_homes();
    vae_->drop_scratch();
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

QiPlan MfPipeline::plan(int64_t hw, int n, int64_t max_text, int64_t n_ids) {
    QiPlan p;
    p.budget = budget();
    MfDit& d = dit();
    const MfDit* o = edit_active_ ? dit_.get() : edit_dit_.get();
    p.fixed = d.weight_bytes() + d.lora_bytes() + (o ? o->weight_bytes() : 0) + vae_->weight_bytes() + kScratch;
    p.lora = d.lora_bytes();
    p.lora_n = d.lora_count();
    if (vision_) {
        // the vision encoder an edit opened stays only where the DiT's blocks and the encoder's layers all fit
        // beside it: opening it again takes 0.2 s (B70, its 836 MB), a streamed block more each step
        const uint64_t vw = vision_->weight_bytes(), act = hw > 0 ? d.activation_bytes(hw, n, max_text) : 0;
        if (p.budget < p.fixed + vw + std::max(act, te_->activation_bytes(n_ids)) + d.block_bytes() +
                           te_->layer_bytes()) {
            gpu::stream_sync(s_);
            vision_.reset();
        } else {
            p.fixed += vw;
        }
    }
    const uint64_t avail = plan_avail(p), dit_min = d.block_min(), te_min = te_->layer_min();
    // the DiT's activations, the MLP in bands of rows when the whole takes more than a third of what is there
    const int64_t mlp = mlp_bands(avail, hw > 0, p.activations,
                                  [&](int64_t rows) { return d.activation_bytes(hw, n, max_text, rows); });
    const uint64_t work = std::max(p.activations, te_->activation_bytes(n_ids));
    if (avail < work + dit_min + te_min) plan_too_small(p);
    p.blocks = clampu(avail - work - te_min, dit_min, d.block_bytes());
    p.layers = clampu(avail - work - p.blocks, te_min, te_->layer_bytes());
    plan_blocks(d, p.blocks);
    // the encoder's layers: fewer at once (their room is the DiT's now), more only when it encodes (te_layers), as
    // QiPipeline's: a prompt whose states are kept does not run it
    te_want_ = p.layers;
    if (p.layers < te_set_) {
        te_->set_layer_budget(p.layers);
        te_set_ = p.layers;
    }
    plan_rows(p, d, hw > 0, mlp);
    p.resident_layers = te_->resident_layers();
    p.vae = avail - p.blocks - p.layers;
    return p;
}

void MfPipeline::edit_refs(const QiRequest& r, Edit& e) {
    if (edit_path_.empty() || vision_path_.empty())
        throw std::invalid_argument("image engine: Mage-Flow's edits need the edit DiT and the vision encoder");
    if (r.refs.size() > kMaxRefs)
        throw std::invalid_argument("image engine: Mage-Flow's edits take up to 3 reference images");
    const int32_t pad = te_size_->image_pad();   // the <|image_pad|> token, from the text encoder's vocabulary
    if (std::count(r.ids.begin(), r.ids.end(), pad) != (int64_t) r.refs.size())
        throw std::invalid_argument("image engine: the prompt's image pads and the reference images differ in number");
    const int64_t h = r.height / 16, w = r.width / 16, HW = r.width * r.height;
    std::string err;
    if (!vae_->open_encoder(err)) throw std::runtime_error(err);
    if (!vision_) {   // kept for the next edit while the budget has room for it (plan)
        auto v = std::make_unique<core::Vision>();
        if (!v->open(vision_path_, s_, err)) throw std::runtime_error(err);
        vision_ = std::move(v);
    }
    core::Vision& vision = *vision_;
    e.rows = (int64_t) r.refs.size() * h * w;
    const int64_t Z = dit_size_->latent_channels();
    float* rgb = nullptr;
    float* eps = nullptr;
    if (!gpu::alloc_device(&e.z, (size_t) (e.rows * Z) * 4) || !gpu::alloc_device(&rgb, (size_t) (3 * HW) * 4) ||
        !gpu::alloc_device(&eps, (size_t) (h * w * Z) * 4)) {
        gpu::free(rgb);
        gpu::free(eps);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    try {
        for (size_t j = 0; j < r.refs.size(); ++j) {
            const QiRef& f = r.refs[j];
            if (f.width <= 0 || f.height <= 0 || f.rgba.size() != (size_t) (f.width * f.height * 4))
                throw std::invalid_argument("image engine: a reference image is empty");
            // the reference's RGB [3][H][W] (0 .. 255) over white (img_start.hpp), resized below by Pillow's
            // bicubic on the host (img_crop.hpp): the encoder reads these floats, and its resize is the reference's
            const std::vector<float> src =
                image_planes(f, f.width, f.height, 3, PlaneAlpha::White, PlaneRange::Byte);
            // the vision encoder's input: the long edge capped, then the processor's size (two resizes, as the
            // reference's)
            int64_t cw = 0, ch = 0, vw = 0, vh = 0;
            vl_size(f.width, f.height, cw, ch, vw, vh);
            std::vector<float> cap = pillow_resize_planes(src, 3, f.width, f.height, cw, ch, Filter::bicubic);
            std::vector<float> vl = pillow_resize_planes(cap, 3, cw, ch, vw, vh, Filter::bicubic);
            for (float& v : vl) v = v / 127.5f - 1.f;
            e.vouts.push_back(vision.encode(vl.data(), vw, vh));
            // the VAE's input at the image's size; its posterior sampled with noise of the seed's own (each reference
            // its stream, apart from the target's)
            std::vector<float> px = pillow_resize_planes(src, 3, f.width, f.height, r.width, r.height,
                                                          Filter::bicubic);
            for (float& v : px) v = v / 127.5f - 1.f;
            gpu::copy(rgb, px.data(), px.size() * 4);
            const auto n = philox_randn(r.seed ^ (0x9E3779B97F4A7C15ull * (j + 1)), (uint64_t) (h * w * Z));
            gpu::copy(eps, n.data(), n.size() * 4);
            vae_->encode(rgb, h, w, eps, e.z + (int64_t) j * h * w * Z);
            e.sizes.emplace_back(h, w);
        }
    } catch (...) {
        gpu::free(rgb);
        gpu::free(eps);
        throw;
    }
    gpu::free(rgb);
    gpu::free(eps);
    // the prompts with each pad widened to its reference's image tokens
    auto widen = [&](const std::vector<int32_t>& src, std::vector<int32_t>& out, std::vector<TextEncoder::Picture>& te) {
        size_t k = 0;
        for (const int32_t id : src) {
            if (id != pad) {
                out.push_back(id);
                continue;
            }
            if (k >= e.vouts.size())
                throw std::invalid_argument("image engine: the negative prompt's image pads and the references differ");
            const core::VisionOut& v = e.vouts[k++];
            TextEncoder::Picture im;
            im.index = (int64_t) out.size();
            im.nx = v.grid_w;
            im.ny = v.grid_h;
            im.rows = v.main.data();
            for (const auto& d : v.deepstack) im.deepstack.push_back(d.data());
            te.push_back(im);
            out.insert(out.end(), (size_t) v.rows, pad);
        }
    };
    widen(r.ids, e.ids, e.te);
    if (r.cfg > 1.0f && !r.neg_ids.empty()) widen(r.neg_ids, e.neg_ids, e.neg_te);
}

std::vector<uint8_t> MfPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                          const std::function<bool(int, int)>& step,
                                          const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    const bool guided = r.cfg > 1.0f && !r.neg_ids.empty(), editing = !r.refs.empty();
    if (guided && (int64_t) r.neg_ids.size() <= r.drop)
        throw std::invalid_argument("image engine: no negative prompt after the system turn");
    const int64_t h = r.height / 16, w = r.width / 16, HW = r.width * r.height;
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs a base image (an init image or "
                                    "a reference)");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    if (base != nullptr && !base_encoder_)
        throw std::invalid_argument("image engine: the VAE file has no encoder for a base image");
    const std::vector<float> sigmas =
        strength_sigmas(r.sigmas.empty() ? mage_sigmas(r.steps > 0 ? (int) r.steps : kTurboSteps)
                                          : r.sigmas, r.strength);
    LatentStart ls;
    const auto t0 = clock::now();
    ensure_open();
    // the references first (the vision encoder and the VAE's encoder in VRAM the DiT's cache gives way to)
    Edit e;
    if (editing) {
        if (stage) stage("refs");
        std::vector<std::pair<int64_t, int64_t>> sizes;
        sizes.reserve(r.refs.size());
        for (const QiRef& f : r.refs) sizes.emplace_back(f.width, f.height);
        uint64_t more = need(0, r.width, r.height, sizes).least - need(0, r.width, r.height).least;
        if (vision_) more -= std::min(more, vision_->weight_bytes());   // held already
        if (held() + more > budget()) release(false);
        edit_refs(r, e);
        vae_->drop_scratch();
    }
    if (base != nullptr) {
        if (stage && !editing) stage("refs");
        encode_base(r, *base, ls);
    }
    const int64_t max_ids = (int64_t) std::max(editing ? e.ids.size() : r.ids.size(),
                                               editing ? e.neg_ids.size() : r.neg_ids.size());
    MfDit& d = use_dit(editing, want_q8(h * w + e.rows, guided ? 2 : 1, max_ids - r.drop, max_ids));
    use_loras(r);   // before the plan (their bytes are fixed) and the contexts (they are made from the weights)
    if (stage) stage("text");
    const std::vector<int32_t>& ids = editing ? e.ids : r.ids;
    const std::vector<int32_t>& neg_ids = editing ? e.neg_ids : r.neg_ids;
    const int64_t n_ctx = (int64_t) ids.size() - r.drop, n_neg = guided ? (int64_t) neg_ids.size() - r.drop : 0;
    QiPlan p = plan(h * w + e.rows, guided ? 2 : 1, std::max(n_ctx, n_neg),
                    (int64_t) std::max(ids.size(), neg_ids.size()));
    {   // the VAE's scratch the last image kept stays only beside this one's work: the Flux2 VAE's, GiBs, does not
        const int64_t n_ids = (int64_t) std::max(ids.size(), neg_ids.size());
        const uint64_t work = std::max(p.activations, te_->activation_bytes(n_ids));
        if (vae_->scratch_held() > (p.vae > work ? p.vae - work : 0)) vae_->drop_scratch();
    }
    // the states: an edit's afresh (its images are not in the ids), a generation's kept for the same prompt
    if (editing) {
        te_layers();
        last_states_ = text(ids, e.te);
        last_ids_.clear();
    } else if (r.ids != last_ids_) {
        te_layers();
        last_states_ = text(r.ids, {});
        last_ids_ = r.ids;
    } else {
        p.text_cached = true;
    }
    const int64_t C = te_->hidden();
    d.set_context(0, last_states_.data() + r.drop * C, n_ctx);
    if (guided) {
        if (editing) {
            te_layers();
            last_neg_states_ = text(neg_ids, e.neg_te);
            last_neg_ids_.clear();
        } else if (r.neg_ids != last_neg_ids_) {
            te_layers();
            last_neg_states_ = text(r.neg_ids, {});
            last_neg_ids_ = r.neg_ids;
        }
        d.set_context(1, last_neg_states_.data() + r.drop * C, n_neg);
    } else {
        d.set_context(1, nullptr, 0);
    }
    te_->drop_activations();
    d.set_refs(e.z, e.sizes);
    const auto t1 = clock::now();
    // the seed's noise as Qwen-Image's ([channels][h][w], PyTorch's CUDA randn), as the latent's tokens [h w][channels]
    const int64_t Cl = d.latent_channels(), n = Cl * h * w;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    noise_to_tokens(noise.data(), Cl, h, w, 1, tokens.data());
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4)) {
        gpu::free(x);
        d.set_refs(nullptr, {});
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    std::vector<float> out((size_t) (4 * HW));
    // a mask: the tokens with a pixel of the mask less than opaque change, the rest follow the base image's latent
    // from the seed's noise (img_start.hpp)
    std::vector<uint8_t> mask_px;
    try {
        if (mask) {
            mask_px = resize_nearest(r.mask, r.width, r.height);
            ls.set_mask(request_latent_mask(r, mask_px, r.width, r.height, 16), tokens, Cl, h * w, true);
        }
        gpu::copy(x, tokens.data(), (size_t) n * 4);
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
        d.sample(x, h, w, sigmas, guided ? r.cfg : 1.0f, step, ls.after(s_), peek, &r.guide,
                 r.sampler == "er_sde", r.shift > 0 ? r.shift : 6.0f, r.seed);
        d.set_refs(nullptr, {});
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it, then the encoder's
        // layers (the next prompt streams them back); what still does not fit, the Flux2 VAE decodes in tiles
        uint64_t room = decode_room(p, d, vae_->scratch_bytes(h, w));
        if (vae_->scratch_bytes(h, w) > room && te_->layers_vram() > 0) {
            room += te_->layers_vram();
            te_->drop_layers();
            te_set_ = 0;
        }
        vae_->decode(x, h, w, rgba, room);
        p.vae_tiled = vae_->last_tiled();
        if (vae_->scratch_bytes(h, w) > p.vae) vae_->drop_scratch();
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        d.set_refs(nullptr, {});
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

void MfPipeline::encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls) {
    const int64_t h = r.height / 16, w = r.width / 16;
    if (r.refs.empty()) {   // the encoder's room: the cache gives way when the budget is short of it (an edit's
        const uint64_t more = need(0, r.width, r.height, {}, true).least - need(0, r.width, r.height).least;
        if (held() + more > budget()) release(false);   // references made it already)
    }
    std::string err;
    if (!vae_->open_encoder(err)) throw std::runtime_error(err);
    // the VAE's input at the image's size, over white, resized as the references are (Pillow's bicubic)
    const std::vector<float> src =
        image_planes(base, base.width, base.height, 3, PlaneAlpha::White, PlaneRange::Byte);
    std::vector<float> px =
        pillow_resize_planes(src, 3, base.width, base.height, r.width, r.height, Filter::bicubic);
    for (float& v : px) v = v / 127.5f - 1.f;
    float* rgb = nullptr;
    if (!gpu::alloc_device(&rgb, px.size() * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    try {
        float* z0 = ls.base(dit_size_->latent_channels() * h * w);
        gpu::copy(rgb, px.data(), px.size() * 4);
        vae_->encode(rgb, h, w, nullptr, z0);   // the posterior's mean
    } catch (...) {
        gpu::free(rgb);
        vae_->drop_scratch();
        throw;
    }
    gpu::free(rgb);
    vae_->drop_scratch();
}

}  // namespace strata::image
