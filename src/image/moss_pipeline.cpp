// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_pipeline.cpp - see include/strata/image/moss_pipeline.hpp.

#include "sigmas.hpp"
#include "strata/image/moss_pipeline.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/cnet_prep.hpp"
#include "strata/image/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr int kSteps = 50;            // the reference pipeline's
constexpr float kShift = 5.0f;        // its scheduler's
constexpr int kUncondSkip = 22;       // the block its negative pass skips
constexpr int64_t kMaxTokens = 4096;  // its text encoder's longest prompt
// The text encoder's VRAM besides its blocks before it was first measured (te_fixed_): its other tensors but the head
// (an encoder has none) and this for the K/V cells and the buffers of a 4096-token chunk.  Measured on the B70
// (Qwen3.5-9B Q4_K_M, 2026-10-28): 1805 MiB in all, 1260 past token_embd; the first open measures it on this GPU and
// that replaces it.
constexpr uint64_t kTextWork = 1280 * kMiB;
// the room a stage keeps beside the plan it was given, for the device's own use past what held() counts: an
// eighth of the device's room (core::vram_room; the 64 MiB measured here on the 16 GiB card, where the
// room is 512).
uint64_t stage_slack() { return vram_room(/*parts*/ 8); }
// ... and a quarter for a whole frame's VAE work: the 128 MiB the RTX 4070's refused 64 MiB buffer measured
uint64_t frame_slack() { return vram_room(/*parts*/ 4); }

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

// an edit's references in the stream at every step (MossDit::set_refs), rather than kept as keys and values
bool in_stream(QiRefLayout l) { return l == QiRefLayout::AitkNoKv || l == QiRefLayout::Appended; }

}  // namespace

std::vector<float> moss_sigmas(int n, float shift) {
    if (n == 1) return {1.f, 0.f};
    constexpr int kTrain = 1000;
    // torch.linspace(1, end, count) in float32 on an AVX2 CPU: from the start below the middle, from the end past it,
    // each value one fused multiply-add (the reference's grid, 2.11)
    auto linspace = [](float start, float end, int count) {
        std::vector<float> v((size_t) count);
        const float step = count > 1 ? (end - start) / (float) (count - 1) : 0.f;
        const int half = count / 2;
        for (int i = 0; i < count; ++i)
            v[(size_t) i] = i < half ? std::fma(step, (float) i, start) : std::fma(-step, (float) (count - 1 - i), end);
        return v;
    };
    auto grid = [&](int count, float s) {
        std::vector<float> v = linspace(1.f, (float) ((double) s / (double) (kTrain + s - 1)), count);
        for (float& x : v) x = sigma_shift(x, s);
        return v;
    };
    const std::vector<float> want = grid(n, shift);
    std::vector<float> train = grid(kTrain, 1.f);
    std::reverse(train.begin(), train.end());   // rising
    std::vector<float> out;
    for (const float s : want) {
        const int64_t up = std::min<int64_t>(std::lower_bound(train.begin(), train.end(), s) - train.begin(),
                                             kTrain - 1);
        const int64_t lo = std::max<int64_t>(up - 1, 0);
        const float v = std::fabs(s - train[(size_t) lo]) <= std::fabs(train[(size_t) up] - s) ? train[(size_t) lo]
                                                                                               : train[(size_t) up];
        if (out.empty() || out.back() != v) out.push_back(v);
    }
    out.push_back(0.f);
    return out;
}

std::vector<float> torch_cpu_randn(uint64_t seed, uint64_t n) {
    std::mt19937 g((uint32_t) seed);
    std::vector<float> d((size_t) n);
    // uniform_real_distribution<float>: the draw's low 24 bits over 2^24
    auto uniform = [&g] { return (float) (g() & ((1u << 24) - 1)) * (1.f / (float) (1u << 24)); };
    // normal_fill_16: eight pairs, the first eight values' radius with the last eight's angle
    auto fill16 = [](float* p) {
        for (int j = 0; j < 8; ++j) {
            const float u1 = 1.f - p[j], u2 = p[j + 8];
            const float radius = std::sqrt(-2.f * std::log(u1));
            // pi written out: MSVC's <cmath> has no M_PI without a define
            const float theta = 2.f * (float) 3.14159265358979323846 * u2;
            p[j] = radius * std::cos(theta);
            p[j + 8] = radius * std::sin(theta);
        }
    };
    if (n < 16) {   // normal_distribution's own way for fewer than 16 (the pipelines' noise never is)
        std::normal_distribution<float> nd;
        for (float& v : d) v = nd(g);
        return d;
    }
    for (float& v : d) v = uniform();
    for (uint64_t i = 0; i + 16 <= n; i += 16) fill16(d.data() + i);
    if (n % 16 != 0) {   // the last sixteen again, from fresh draws
        float* p = d.data() + n - 16;
        for (int j = 0; j < 16; ++j) p[j] = uniform();
        fill16(p);
    }
    return d;
}

bool moss_is_dit(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        std::ifstream f(std::filesystem::path(path) / "config.json");
        const std::string cfg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return cfg.find("\"MOSSImageTransformer\"") != std::string::npos;
    }
    if (path.size() < 12 || path.compare(path.size() - 12, 12, ".safetensors") != 0) return false;
    try {
        SafeTensors st(path);
        return st.find("shared_block_modulation.weight") != nullptr && st.find("text_embedder.proj.weight") != nullptr;
    } catch (const std::exception&) {
        return false;
    }
}

bool MossPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                        int q8, const std::string& vision) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    q8_ = q8;
    dit_size_ = std::make_unique<MossDit>();
    dit16_size_ = std::make_unique<MossDit>();
    vae_size_ = std::make_unique<MfVae>();
    if (!(dit_size_->open(dit, s_, err, true, true) && dit16_size_->open(dit, s_, err, true, false) &&
          vae_size_->open(vae, s_, err, true)))
        return false;
    if (!vae_size_->flux2()) {
        err = "image engine: MOSS-Image's VAE is the Flux2 VAE (" + vae + " is not)";
        return false;
    }
    vision_path_ = vision;
    if (!vision.empty()) {   // an ai-toolkit edit adapter's pictures: Qwen3.5-9B's ViT
        vision_size_ = std::make_unique<core::Vision>();
        if (!vision_size_->open(vision, s_, err, true)) return false;
        image_pad_ = TextEncoder::pad_id(text);   // the shared read of the vocabulary
    }
    {
        std::string no;
        base_encoder_ = vae_size_->open_encoder(no, true);
    }
    // Not TextEncoder's dry open (the other models' way): it lists only the layers that stream, so 4.0 MiB less over
    // the blocks, 0.3 MiB less over the two slots, none for the rest but the head (94 MiB), and its blocks' resident
    // weights take 6 MiB of VRAM at open, which moved MOSS-Image's picture (RX 9060 XT, 2026-11-10).  The safer figure.
    try {   // the text encoder's blocks (blk.N.*) and the rest but the head, from the file's tensors
        GgufFile g(text);
        std::vector<uint64_t> blk;
        for (const TensorInfo& t : g.tensors()) {
            const uint64_t b = tensor_payload_bytes(t);
            if (t.name.rfind("blk.", 0) == 0) {
                const size_t l = (size_t) std::stoul(t.name.substr(4));
                if (blk.size() <= l) blk.resize(l + 1, 0);
                blk[l] += b;
            } else if (t.name != "output.weight") {
                te_other_ += b;
            }
        }
        for (const uint64_t b : blk) {
            te_layers_ += b;
            te_layer_max_ = std::max(te_layer_max_, b);
        }
    } catch (const std::exception& e) {
        err = "image engine: MOSS-Image's text encoder: " + std::string(e.what());
        return false;
    }
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    auto_budget_ = vram_cap(free);
    note_others();
    return true;
}

void MossPipeline::use_loras(const QiRequest& r) {
    std::string err;
    const std::string before = dit_->lora_key();
    std::vector<LoraUse> dit;
    lora_split(r.loras, dit, te_loras_);
    dit_->set_lora_bake(lora_bake_);
    if (!dit_->set_loras(dit, err)) {
        ctx_states_[0].clear();
        ctx_states_[1].clear();
        throw LoraRejected(err);
    }
    if (te_ && !te_->set_loras(te_loras_, err)) {
        ctx_states_[0].clear();
        ctx_states_[1].clear();
        te_loras_now(te_->lora_key());
        throw LoraRejected(err);
    }
    te_loras_now(lora_key_of(te_loras_));   // the states kept were the encoder's with the other adapters
    if (dit_->lora_key() != before) {   // the contexts' text rows were made with the other adapters' weights
        ctx_states_[0].clear();
        ctx_states_[1].clear();
    }
}

uint64_t MossPipeline::text_fixed() const { return te_fixed_ > 0 ? te_fixed_ : te_other_ + kTextWork; }

void MossPipeline::ensure_open(bool q8) {
    if (dit_ && dit_->q8() != q8) {   // the other form for this request's budget: opened again
        gpu::stream_sync(s_);
        dit_.reset();
    }
    if (!q8) dit_host_.clear();   // FP16 reads the file in place
    std::string err;
    if (!vae_) {
        auto v = std::make_unique<MfVae>();
        if (!v->open(vae_path_, s_, err)) throw std::runtime_error(err);
        vae_ = std::move(v);
    }
    if (!dit_) {
        auto d = std::make_unique<MossDit>();
        const bool ok = d->open(dit_path_, s_, err, false, q8, &dit_host_);
        dit_host_.clear();
        if (!ok) throw std::runtime_error(err);
        d->set_block_budget(d->block_min());   // nothing resident until the request plans its budget
        d->set_skip(1, {kUncondSkip});
        dit_ = std::move(d);
        dit_set_ = dit_->block_min();
        mlp_set_ = -1;
        ctx_states_[0].clear();
        ctx_states_[1].clear();
    }
}

// W8Choice::want_q8 on the FP16 blocks all resident beside the rest.  The encoder's share is its fixed part and two
// layer slots: what plan() keeps it open with (need()'s least)
bool MossPipeline::want_q8(int64_t rows, int n, int64_t max_text) {
    const MossDit& d16 = *dit16_size_;
    const uint64_t fixed = d16.weight_bytes() + d16.text_bytes() + vae_size_->weight_bytes() +
                           (states_given_ ? 0 : text_fixed() + 2 * te_layer_max_);
    return w8_.want_q8(q8_, fixed + d16.block_bytes() + d16.activation_bytes(rows, n, max_text), budget(), rows,
                       "MOSS-Image's DiT", s_, [&](int64_t r) { return d16.time_forms(r); });
}

uint64_t MossPipeline::held() const {
    uint64_t b = 0;
    if (vae_) b += vae_->weight_bytes() + vae_->scratch_held() + (vae_->has_encoder() ? vae_->encoder_bytes() : 0);
    if (dit_) {
        b += dit_->weight_bytes() + dit_->lora_bytes() + dit_->scratch_bytes() + dit_->blocks_vram() +
             dit_->activations_held() + dit_->ref_bytes(dit_->ref_rows()) + dit_->stream_bytes(dit_->stream_rows());
        if (dit_->text_loaded()) b += dit_->text_bytes();
    }
    if (te_) b += text_fixed() + te_->layers_vram();
    return b;
}

QiNeed MossPipeline::need(int64_t n_text, int64_t width, int64_t height,
                          const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                          const std::vector<LoraUse>& loras) {
    const MossDit& dit = q8_ == 0 ? *dit16_size_ : *dit_size_;   // the least: 8 bits, unless FP16 is asked for
    const MossDit& full = q8_ == 1 ? *dit_size_ : *dit16_size_;  // all resident: FP16's, unless 8 bits are
    const MfVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16;
    // an edit's references (Moss-Image-Edit's): their latent rows' kept keys and values, the attention's rows before
    // each context's, and each one's encode
    int64_t R = 0;
    uint64_t ref_enc = 0;
    for (const auto& [rw, rh] : refs) {
        R += (rw / 16) * (rh / 16);
        ref_enc = std::max(ref_enc, vae.encode_scratch_bytes(rh / 16, rw / 16) + (uint64_t) (3 * rw * rh) * 4 +
                                        (uint64_t) ((rw / 16) * (rh / 16) * dit.latent_channels()) * 4);
    }
    // the references in the stream (ai-toolkit's with its kv_cache off; MossDit::set_refs) are rows after each
    // context's text: no pass of their own, no keys and values kept
    const bool stream = in_stream(ref_layout_);
    const int64_t hw = stream ? h * w : std::max(h * w, R);   // the activations' rows: the image's, or the
                                                              // references' at their pass
    const int64_t max_text = std::max(n_text, n_neg) + (stream ? R : 0);
    const int64_t kept = stream ? 0 : R;
    const int n = n_neg > 0 ? 2 : 1;
    // the adapters: their A and B (in VRAM for the request, the blocks streamed or not), and the products' scratch of a
    // step's rows; nothing bound yet, so the DiT's activations here hold none of it
    const uint64_t lora = lora_gpu_bytes(loras, n * (hw + max_text));
    uint64_t base = ref_enc,
             fixed = vae.weight_bytes() + dit.text_bytes() + lora + (stream ? dit.stream_bytes(R) : dit.ref_bytes(R));
    if ((R > 0 || mask || need_control_maps_ > 0) && base_encoder_) fixed += vae.encoder_bytes();
    if (mask && base_encoder_) {   // a base image: its encode, and its latent (and a mask's) kept
        fixed += LatentStart::bytes(dit.latent_channels() * h * w, h * w, true);
        base = std::max(base, vae.encode_scratch_bytes(h, w) + (uint64_t) (3 * height * width) * 4);
    }
    if (need_control_maps_ > 0) {
        fixed += dit.control_weight_bytes() + dit.control_work_bytes(h * w, need_control_maps_);
        base = std::max(base, vae.encode_scratch_bytes(h, w) + (uint64_t) (3 * height * width) * 4);
    }
    // an ai-toolkit edit's pictures: the vision encoder, open beside the text encoder while they are read
    uint64_t vis = 0;
    if (vision_size_)
        for (const auto& [pw, ph] : pictures_)
            vis = std::max(vis, vision_size_->weight_bytes() + vision_size_->scratch_bytes((pw / 16) * (ph / 16)));
    const uint64_t te = (states_given_ ? 0 : text_fixed() + 2 * te_layer_max_) + vis;
    QiNeed nd;
    nd.least = fixed + dit.weight_bytes() + dit.scratch_bytes() +
               std::max({dit.activation_bytes(hw, n, max_text, 512, kept) + dit.block_min(), te,
                         vae.least_scratch_bytes(h, w), base});
    nd.full = fixed + full.weight_bytes() + full.scratch_bytes() + full.block_bytes() +
              (states_given_ ? 0 : text_fixed() + te_layers_) + vis +
              std::max({full.activation_bytes(hw, n, max_text, 0, kept), vae.scratch_bytes(h, w), base});
    vram_into(nd);
    return nd;
}

void MossPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    note_others();
    if (held() == 0 || !dit_) return;
    (void) plan(0, 1, 0);   // shrink at once what no longer fits
}

void MossPipeline::release(bool all) {
    if (held() == 0) return;
    gpu::stream_sync(s_);
    if (all) {
        te_.reset();
        if (dit_) {
            dit_->drop_loras();   // the adapters' A and B too (they go back on the next request's, from its files)
            if (dit_->q8()) dit_host_ = dit_->take_host();   // the RAM setup counts for it: kept for the next open
        }
        dit_.reset();
        vae_.reset();
        ctx_states_[0].clear();
        ctx_states_[1].clear();
        return;
    }
    if (dit_) {
        dit_->drop_blocks();
        dit_->drop_block_homes();   // the RAM the streamed blocks' copies hold, back for the other engines
        dit_->drop_activations();
        dit_->set_refs({});   // an edit's references' keys and values (made again for the next edit)
    }
    te_.reset();
    if (vae_) vae_->drop_scratch();
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

// The split: the DiT's activations and blocks first (used every step), then the text encoder, kept open only where its
// fixed part and two slots fit beside all of them (else closed: its states are made).  text: the text embedder's weights
// counted though not loaded (set_context loads them).  hw 0: only what no longer fits goes.
QiPlan MossPipeline::plan(int64_t hw, int n, int64_t max_text, bool text) {
    QiPlan p;
    p.budget = budget();
    MossDit& d = *dit_;
    p.fixed = d.weight_bytes() + d.lora_bytes() + d.scratch_bytes() + control_work_ +
              (text || d.text_loaded() ? d.text_bytes() : 0) + vae_->weight_bytes() +
              (vae_->has_encoder() ? vae_->encoder_bytes() : control_work_ ? vae_size_->encoder_bytes() : 0) +
              (control_work_ && d.control_path().empty() ? d.control_weight_bytes() : 0) +
              (in_stream(ref_layout_) ? d.stream_bytes(ref_rows_) : d.ref_bytes(ref_rows_));
    p.lora = d.lora_bytes();
    p.lora_n = d.lora_count();
    const uint64_t avail = plan_avail(p), dit_min = d.block_min();
    // an edit's references: their pass over the blocks takes the activations' rows too (set_refs; activation_bytes);
    // in the stream, they are rows after each context's text instead
    const bool stream = in_stream(ref_layout_);
    const int64_t kept = stream ? 0 : ref_rows_;
    if (stream && hw > 0) max_text += ref_rows_;
    const int64_t mlp = mlp_bands(avail, hw > 0, p.activations,
                                  [&](int64_t rows) { return d.activation_bytes(hw, n, max_text, rows, kept); });
    if (avail < p.activations + dit_min) plan_too_small(p);
    uint64_t te_fixed = te_ ? text_fixed() : 0;
    uint64_t te_min = te_ ? te_->layer_min() : 0;
    // the encoder stays only beside every block of the DiT: a streamed block costs every step (the RTX 4070's link at
    // x4: 67 s for 50 steps resident, 73 streamed past 15 blocks), opening the encoder again a second or two a prompt
    if (te_ && avail < p.activations + d.block_bytes() + te_fixed + te_min) {   // it goes: its states are made
        gpu::stream_sync(s_);
        te_.reset();
        te_set_ = 0;
        te_fixed = 0;
        te_min = 0;
    }
    p.blocks = clampu(avail - p.activations - te_fixed - te_min, dit_min, d.block_bytes());
    plan_blocks(d, p.blocks);
    if (te_) {
        p.layers = clampu(avail - p.activations - p.blocks - te_fixed, te_min, te_->layer_bytes());
        if (p.layers != te_set_) {
            te_->set_layer_budget(p.layers);
            te_set_ = p.layers;
        }
        p.resident_layers = te_->resident_layers();
    }
    plan_rows(p, d, hw > 0, mlp);
    p.vae = avail - p.blocks - p.layers - te_fixed;
    return p;
}

// The text encoder opened in what the budget leaves beside what the engine holds (the DiT's blocks and activations
// giving way when its fixed part and two slots do not fit there), its fixed VRAM measured
bool MossPipeline::open_text() {
    if (te_) return true;
    const uint64_t te_min = 2 * te_layer_max_;
    auto room = [&] { const uint64_t h = held(); return budget() > h ? budget() - h : 0; };
    if (room() < text_fixed() + te_min && dit_) {
        dit_->drop_blocks();
        dit_->drop_activations();
        dit_set_ = 0;
        mlp_set_ = -1;
    }
    const uint64_t r = room();
    if (r < text_fixed() + te_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(budget() >> 20) +
                                 " MiB) is too small for MOSS-Image's text encoder");
    gpu::stream_sync(s_);
    const uint64_t before = free_vram();
    auto t = std::make_unique<TextEncoder>();
    std::string err;
    if (!t->open(text_path_, kMaxTokens, err, r - text_fixed())) throw std::runtime_error(err);
    if (te_measure_ == 0) {   // the first open alone: later ones overlap the DiT's frees, which the driver reports late
        const uint64_t after = free_vram();
        if (before > after && before - after > t->layers_vram()) {
            te_fixed_ = before - after - t->layers_vram();
            te_measure_ = 1;
        }
    }
    if (!t->set_loras(te_loras_, err)) throw LoraRejected(err);
    te_ = std::move(t);
    te_set_ = te_->layers_vram();
    std::fprintf(stderr, "image engine: MOSS-Image's text encoder open: %d blocks resident (%llu MiB), %llu MiB besides\n",
                 te_->resident_layers(), (unsigned long long) (te_->layers_vram() >> 20),
                 (unsigned long long) (text_fixed() >> 20));
    return true;
}

// The encoder's layers' budget, which a plan set beside that request's work, against what is held and free now: the
// last request's leftovers (its decode's VAE scratch, its edit's references) count in held() but were not in that plan
// (an edit after a refined edit on ROCm 10.1: a budget of 3266 MiB for 442 MiB of layers with 2498 MiB free, and a
// layer refused).  The VAE's scratch gives way first, then the DiT's activations and blocks.
void MossPipeline::fit_text() {
    if (!te_) return;
    auto room = [&]() -> uint64_t {
        const uint64_t others = held() - te_->layers_vram(), on_device = free_vram() + te_->layers_vram(),
                       slack = stage_slack();
        return std::min<uint64_t>(budget() > others ? budget() - others : 0, on_device > slack ? on_device - slack : 0);
    };
    if (room() < te_set_ && vae_) vae_->drop_scratch();
    if (room() < te_->layer_min() && dit_) {
        dit_->drop_activations();
        dit_->drop_blocks();
        dit_set_ = 0;
        mlp_set_ = -1;
    }
    const uint64_t r = clampu(room(), te_->layer_min(), te_->layer_bytes());
    if (r < te_set_) {
        te_->set_layer_budget(r);
        te_set_ = r;
    }
}

void MossPipeline::encode(const std::vector<int32_t>& ids, std::vector<float>& out,
                          const std::vector<TextEncoder::Picture>& pictures) {
    if ((int64_t) ids.size() > kMaxTokens)
        throw std::invalid_argument("image engine: MOSS-Image's prompt is longer than 4096 tokens");
    open_text();
    fit_text();
    const uint64_t before = free_vram() + te_->layers_vram();
    std::string err;
    auto states = te_->encode(ids, err, pictures);
    if (states.empty()) throw std::runtime_error(err);
    out = std::move(states);
    // the prompt's buffers, made by the first open's first encode, are the encoder's fixed part too
    if (te_measure_ == 1) {
        const uint64_t after = free_vram() + te_->layers_vram();
        if (before > after) te_fixed_ += before - after;
        te_measure_ = 2;
    }
}

void MossPipeline::encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls) {
    const int64_t h = r.height / 16, w = r.width / 16;
    const uint64_t more = need(0, r.width, r.height, {}, true).least - need(0, r.width, r.height).least;
    // the encoder's room by the device's own count as well: the last request's parts may hold more than held() counts
    // (an edit after an edit on the RTX 4070's 12 GB held 11525 MiB of an 11215 MiB budget, and the encode was refused)
    const uint64_t enc = vae_->encode_scratch_bytes(h, w) + (uint64_t) (3 * r.width * r.height) * 4 +
                         (vae_->has_encoder() ? 0 : vae_->encoder_bytes()) + stage_slack();
    if (held() + more > budget() || free_vram() < enc) release(false);
    std::string err;
    if (!vae_->open_encoder(err)) throw std::runtime_error(err);
    // RGB [3][H][W] in -1 .. 1 over white, at the image's size (nearest): img_start.hpp's image_planes
    const std::vector<float> px = image_planes(base, r.width, r.height, 3, PlaneAlpha::White, PlaneRange::Signed);
    float* rgb = nullptr;
    if (!gpu::alloc_device(&rgb, px.size() * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    try {
        float* z0 = ls.base(dit_->latent_channels() * h * w);
        gpu::copy(rgb, px.data(), px.size() * 4);
        vae_->encode(rgb, h, w, nullptr, z0);   // the posterior's mean
    } catch (...) {
        gpu::free(rgb);
        vae_->drop_scratch();
        throw;
    }
    gpu::free(rgb);   // the VAE's scratch stays for the decode where the plan has room for it (generate)
}

std::vector<core::VisionOut> MossPipeline::encode_pictures(const std::vector<QiRef>& pictures) {
    if (vision_path_.empty() || image_pad_ < 0)
        throw std::invalid_argument("image engine: an ai-toolkit edit's pictures need the vision encoder (--vision, "
                                    "Qwen3.5-9B's mmproj) and a text encoder with <|image_pad|>");
    std::string err;
    // its weights and the largest picture's scratch beside what is held: the last request's leftovers give way, the
    // VAE's scratch first, then the DiT's activations and blocks (an ai-toolkit edit after one on the RTX 4070's 12 GB:
    // 7 MiB free, and the encoder's first buffer refused)
    {
        core::Vision size;
        if (!size.open(vision_path_, s_, err, true)) throw std::runtime_error(err);
        int64_t most = 0;
        for (const QiRef& f : pictures) most = std::max<int64_t>(most, (f.height / 16) * (f.width / 16));
        const uint64_t need = size.weight_bytes() + size.scratch_bytes(most) + stage_slack();
        auto room = [&]() -> uint64_t {
            const uint64_t h = held();
            return std::min(budget() > h ? budget() - h : 0, free_vram());
        };
        if (room() < need && vae_) vae_->drop_scratch();
        if (room() < need && dit_) {
            dit_->drop_activations();
            dit_->drop_blocks();
            dit_set_ = 0;
            mlp_set_ = -1;
        }
        if (room() < need) release(false);
    }
    core::Vision v;
    if (!v.open(vision_path_, s_, err)) throw std::runtime_error(err);
    std::vector<core::VisionOut> out;
    for (const QiRef& f : pictures) {
        const std::vector<float> rgb = image_planes(f, f.width, f.height, 3, PlaneAlpha::White, PlaneRange::Signed);
        out.push_back(v.encode(rgb.data(), f.width, f.height));
    }
    v.drop_scratch();
    return out;
}

float* MossPipeline::encode_ref(const QiRef& f) {
    const int64_t h = f.height / 16, w = f.width / 16;
    std::string err;
    if (!vae_->open_encoder(err)) throw std::runtime_error(err);
    const std::vector<float> px = image_planes(f, f.width, f.height, 3, PlaneAlpha::White, PlaneRange::Signed);
    float *rgb = nullptr, *z = nullptr;
    if (!gpu::alloc_device(&rgb, px.size() * 4) || !gpu::alloc_device(&z, (size_t) (dit_->latent_channels() * h * w) * 4)) {
        gpu::free(rgb);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    try {
        gpu::copy(rgb, px.data(), px.size() * 4);
        vae_->encode(rgb, h, w, nullptr, z);   // the posterior's mean
    } catch (...) {
        gpu::free(rgb);
        gpu::free(z);
        vae_->drop_scratch();
        throw;
    }
    gpu::free(rgb);   // the VAE's scratch stays for the decode where the plan has room for it (generate)
    return z;
}

std::vector<uint8_t> MossPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                            const std::function<bool(int, int)>& step,
                                            const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    // an edit's references (Moss-Image-Edit's adapter, which the request names): each at its own size
    ref_rows_ = 0;
    for (const QiRef& f : r.refs) {
        if (f.width <= 0 || f.height <= 0 || f.width % 16 != 0 || f.height % 16 != 0 ||
            f.rgba.size() != (size_t) (f.width * f.height * 4))
            throw std::invalid_argument("image engine: a reference image's sides must be multiples of 16");
        ref_rows_ += (f.width / 16) * (f.height / 16);
    }
    ref_layout_ = r.refs_layout;
    // an ai-toolkit edit's pictures: each <|image_pad|> of the prompt (and of the negative prompt) widened to the
    // picture's tokens, one per 32 x 32 pixels
    for (const QiRef& f : r.pictures)
        if (f.width <= 0 || f.height <= 0 || f.width % 32 != 0 || f.height % 32 != 0 ||
            f.rgba.size() != (size_t) (f.width * f.height * 4))
            throw std::invalid_argument("image engine: a picture's sides must be multiples of 32");
    auto widen = [&](const std::vector<int32_t>& in) {
        if (r.pictures.empty()) return in;
        std::vector<int32_t> out;
        size_t k = 0;
        for (const int32_t t : in) {
            if (t == image_pad_ && k < r.pictures.size()) {
                out.insert(out.end(), (size_t) ((r.pictures[k].width / 32) * (r.pictures[k].height / 32)), t);
                ++k;
            } else {
                out.push_back(t);
            }
        }
        if (k != 0 && k != r.pictures.size())
            throw std::invalid_argument("image engine: the prompt's <|image_pad|> tokens are not one a picture");
        return out;
    };
    const std::vector<int32_t> ids0 = widen(r.ids), ids1 = widen(r.neg_ids);
    const std::string pics = r.pictures.empty() ? std::string() : refs_key(r.pictures);
    if (!r.pictures.empty() && (r.te_rows > 0 || r.neg_te_rows > 0))
        throw std::invalid_argument("image engine: the text encoder's pictures need its own states (no TE:)");
    const int64_t width = dit_size_->text_dim();
    if (r.te_rows > 0 && (int64_t) r.te.size() != r.te_rows * width)
        throw std::invalid_argument("image engine: the prompt's states are not " + std::to_string(width) + " wide");
    if (r.neg_te_rows > 0 && (int64_t) r.neg_te.size() != r.neg_te_rows * width)
        throw std::invalid_argument("image engine: the negative prompt's states are not " + std::to_string(width) +
                                    " wide");
    if (r.te_rows == 0 && r.ids.empty()) throw std::invalid_argument("image engine: no prompt");
    const bool guided = r.cfg != 1.0f && (r.neg_te_rows > 0 || !r.neg_ids.empty());
    const int64_t h = r.height / 16, w = r.width / 16, HW = r.width * r.height;
    const int control_maps = r.mctl.empty() ? 0 : (int) !r.mctl_color.rgba.empty() + (int) !r.mctl_depth.rgba.empty();
    control_work_ = control_maps > 0 ? dit_size_->control_work_bytes(h * w, control_maps) : 0;
    struct ControlWorkReset { uint64_t& bytes; ~ControlWorkReset() { bytes = 0; } } reset_control_work{control_work_};
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs an init image");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    if ((base != nullptr || !r.refs.empty()) && !base_encoder_)
        throw std::invalid_argument("image engine: the VAE file has no encoder for a base image");
    if (!r.sampler.empty() && r.sampler != "euler" && r.sampler != "er_sde")
        throw std::invalid_argument("image engine: MOSS-Image's sampler is euler or er_sde");
    MossDit::Guide g;
    g.cfg = guided ? r.cfg : 1.0f;
    g.uncond_skip = r.uncond_skip_set ? r.uncond_skip : std::vector<int>{kUncondSkip};
    g.slg = r.slg_layers;
    g.slg_scale = r.slg_scale;
    g.start = r.slg_start;
    g.stop = r.slg_stop;
    if (!(0.f <= g.start && g.start <= g.stop && g.stop <= 1.f) || !(g.slg_scale >= 0.f))
        throw std::invalid_argument("image engine: skip-layer guidance needs 0 <= slg_start <= slg_stop <= 1 and a "
                                    "scale of 0 or more");
    for (const auto* v : {&g.uncond_skip, &g.slg})
        for (const int b : *v)
            if (b < 0 || b >= dit_size_->blocks())
                throw std::invalid_argument("image engine: no block " + std::to_string(b) + " to skip (0 to " +
                                            std::to_string(dit_size_->blocks() - 1) + ")");
    const std::vector<float> full =
        !r.sigmas.empty() ? r.sigmas : moss_sigmas(r.steps > 0 ? (int) r.steps : kSteps, r.shift > 0 ? r.shift : kShift);
    const int total = (int) full.size() - 1;
    const int first = r.strength < 1.0 ? (int) strength_skip(total, r.strength) : 0;
    const std::vector<float> sigmas = strength_sigmas(full, r.strength);
    const auto t0 = clock::now();
    const int n = guided ? 2 : 1;
    // the chat engine's states have no text encoder adapters: with some, the image engine encodes the ids itself
    std::vector<LoraUse> dit_part, te_part;
    lora_split(r.loras, dit_part, te_part);
    const bool own = !te_part.empty();
    const int64_t rows0 = r.te_rows > 0 && !own ? r.te_rows : (int64_t) ids0.size();
    const int64_t rows1 = guided ? (r.neg_te_rows > 0 && !own ? r.neg_te_rows : (int64_t) ids1.size()) : 0;
    ensure_open(want_q8(h * w, n, std::max(rows0, rows1) + (in_stream(ref_layout_) ? ref_rows_ : 0)));
    MossDit& d = *dit_;
    if (d.ref_rows() > 0) d.set_refs({});   // the last edit's references: this request makes its own (or has none)
    use_loras(r);   // before the plan (their bytes are fixed) and the contexts (a set that changed makes them again)
    LatentStart ls;
    if (base != nullptr) {
        if (stage) stage("refs");
        encode_base(r, *base, ls);
    }
    // the states: the chat's engine's, else the encoder's (kept for the same ids)
    if (stage) stage("text");
    bool cached = true;
    const std::vector<float>* st[2] = {&r.te, &r.neg_te};
    // an ai-toolkit edit's pictures through the vision encoder, where a prompt with them is encoded
    std::vector<core::VisionOut> vouts;
    std::vector<TextEncoder::Picture> pv;
    auto pictures = [&](const std::vector<int32_t>& ids) {
        if (r.pictures.empty() || std::find(ids.begin(), ids.end(), image_pad_) == ids.end())
            return std::vector<TextEncoder::Picture>{};
        if (vouts.empty()) {
            vouts = encode_pictures(r.pictures);
            for (const core::VisionOut& v : vouts) {
                TextEncoder::Picture pic;
                pic.rows = v.main.data();
                pic.nx = v.grid_w;
                pic.ny = v.grid_h;
                pv.push_back(pic);
            }
        }
        return pv;
    };
    if (r.te_rows == 0 || own) {
        if (ids0 != last_ids_ || pics != last_pics_) {
            encode(ids0, last_states_, pictures(ids0));
            last_ids_ = ids0;
            last_pics_ = pics;
            cached = false;
        }
        st[0] = &last_states_;
    }
    if (guided && (r.neg_te_rows == 0 || own)) {
        if (ids1 != last_neg_ids_ || pics != last_neg_pics_) {
            encode(ids1, last_neg_states_, pictures(ids1));
            last_neg_ids_ = ids1;
            last_neg_pics_ = pics;
            cached = false;
        }
        st[1] = &last_neg_states_;
    }
    // the contexts set where their states differ from what they were set from: the text embedder in VRAM for it
    const bool set0 = *st[0] != ctx_states_[0], set1 = guided && *st[1] != ctx_states_[1];
    QiPlan p = plan(h * w, n, std::max(rows0, rows1), set0 || set1);
    // the VAE's scratch the last image kept stays only beside this one's work (the plan leaves it out)
    if (vae_->scratch_held() > p.vae - p.activations) vae_->drop_scratch();
    if (set0) {
        d.set_context(0, st[0]->data(), rows0);
        ctx_states_[0] = *st[0];
    }
    if (set1) {
        d.set_context(1, st[1]->data(), rows1);
        ctx_states_[1] = *st[1];
    }
    if (!guided && d.context_rows(1) > 0) {
        d.set_context(1, nullptr, 0);
        ctx_states_[1].clear();
    }
    p = plan(h * w, n, std::max(rows0, rows1));
    if (d.text_loaded() && p.blocks < d.block_bytes()) {   // the blocks take the text embedder's VRAM
        d.drop_text();
        p = plan(h * w, n, std::max(rows0, rows1));
    }
    p.text_cached = cached;
    if (!r.mctl.empty()) {
        if (control_maps == 0)
            throw std::invalid_argument("image engine: MCTL: needs a color map (MCOLOR:) or a depth map (MDEPTH:)");
        // The encoder and control projection open after planning.  The previous request's VAE scratch, activations,
        // blocks and text layers give way before these allocations, including the maps kept for sampling.
        const uint64_t enc = vae_->encode_scratch_bytes(h, w) + (uint64_t) (3 * HW) * 4 + control_work_ +
                             (vae_->has_encoder() ? 0 : vae_size_->encoder_bytes()) +
                             (d.control_path() == r.mctl ? 0 : d.control_weight_bytes()) + stage_slack();
        const auto room = [&] {
            const uint64_t b = budget(), used = held();
            return std::min(free_vram(), b > used ? b - used : 0);
        };
        if (room() < enc) vae_->drop_scratch();
        if (room() < enc) d.drop_activations();
        if (room() < enc) { d.drop_blocks(); dit_set_ = 0; mlp_set_ = -1; }
        if (room() < enc && te_) { te_.reset(); te_set_ = 0; }
        if (room() < enc)
            throw std::runtime_error("image engine: not enough GPU memory to encode the MOSS-Image control maps");
    }
    d.set_control(r.mctl);
    // the references' keys and values through the adapted blocks, once a request
    if (!r.refs.empty()) {
        std::vector<float*> z;
        try {
            std::vector<MossDit::RefLatent> lat;
            // their encoding's room: the plan leaves it out, and the last request's activations and the text
            // encoder's layers may hold it (a second edit on the RTX 4070's 12 GB was refused the VAE's scratch)
            uint64_t enc = 0;
            for (const QiRef& f : r.refs)
                enc = std::max(enc, vae_->encode_scratch_bytes(f.height / 16, f.width / 16) +
                                        (uint64_t) (3 * f.width * f.height) * 4 +
                                        (uint64_t) (d.latent_channels() * (f.height / 16) * (f.width / 16)) * 4);
            enc += (vae_->has_encoder() ? 0 : vae_->encoder_bytes()) + stage_slack();
            if (free_vram() < enc) d.drop_activations();
            if (free_vram() < enc && te_ && te_->layers_vram() > te_->layer_min()) {
                te_->set_layer_budget(te_->layer_min());
                te_set_ = te_->layer_min();
            }
            for (const QiRef& f : r.refs) {
                z.push_back(encode_ref(f));
                lat.push_back({z.back(), f.height / 16, f.width / 16});
            }
            // the encoders' scratch (the references', the base image's) kept for the decode only where every block
            // is resident and it fits beside this image's work, and let go before the blocks run over the references:
            // allocating it again at the decode cost a 1 MP edit's decode a fifth (B70, 2026-11-04), but a card that
            // streams the blocks (the RTX 4070's 12 GB) needs the room for their layers
            if (d.resident_blocks() < d.blocks() || vae_->scratch_held() > p.vae - p.activations) vae_->drop_scratch();
            d.set_refs(lat, r.refs_layout);
        } catch (...) {
            for (float* p0 : z) gpu::free(p0);
            throw;
        }
        for (float* p0 : z) gpu::free(p0);
    } else if (d.ref_rows() > 0 || d.stream_rows() > 0) {
        d.set_refs({});
    }
    // the color and depth control: the adapter's input projection, and each map through the VAE's encoder at the
    // image's size (the posterior's mean, packed as the image's latent), alive through the sampling
    struct ControlMaps {
        MossDit& d;
        float *color = nullptr, *depth = nullptr;
        ~ControlMaps() {
            d.set_control_input(nullptr, nullptr, 1.f, 1.f);
            gpu::free(color);
            gpu::free(depth);
        }
    } maps{d};
    if (!r.mctl.empty()) {
        const auto encode_map = [&](const QiRef& m) -> float* {
            if (m.rgba.empty()) return nullptr;
            QiRef f;
            f.width = r.width;
            f.height = r.height;
            f.rgba = control_resize(m, r.width, r.height);
            return encode_ref(f);
        };
        maps.color = encode_map(r.mctl_color);
        maps.depth = encode_map(r.mctl_depth);
        if (d.resident_blocks() < d.blocks() || vae_->scratch_held() > p.vae - p.activations) vae_->drop_scratch();
        d.set_control_input(maps.color, maps.depth, r.mctl_color_end, r.mctl_depth_end);
        p = plan(h * w, n, std::max(rows0, rows1));
        p.text_cached = cached;
    }
    const auto t1 = clock::now();
    // the seed's noise as the reference's ([channels][h][w], PyTorch's CPU randn), as the latent's tokens [h w][channels]
    const int64_t Cl = d.latent_channels(), N = Cl * h * w;
    const auto noise = torch_cpu_randn(r.seed, (uint64_t) N);
    std::vector<float> tokens((size_t) N);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < h * w; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * h * w + q)];
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) N * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4)) {
        gpu::free(x);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    std::vector<float> out((size_t) (4 * HW));
    std::vector<uint8_t> mask_px;
    try {
        if (mask) {
            mask_px = resize_nearest(r.mask, r.width, r.height);
            ls.set_mask(request_latent_mask(r, mask_px, r.width, r.height, 16), tokens, Cl, h * w, true);
        }
        gpu::copy(x, tokens.data(), (size_t) N * 4);
        if (r.strength < 1.0) ls.start(x, sigmas[0], s_);   // img2img: from the base image's latent at that time
        if (stage) stage("sampling");
        // partial images (r.preview)
        PreviewDecode preview(
            r, p, N, [&](float* x0, uint64_t room) { vae_->decode(x0, h, w, rgba, room); },
            [&] { vae_->drop_scratch(); },
            [&] {
                gpu::copy(out.data(), rgba, out.size() * 4);
                std::vector<uint8_t> img = rgba8(out, HW);
                if (mask) keep_pixels(img, resize_nearest(*base, r.width, r.height), mask_px);
                return img;
            });
        auto peek = [&](int done, const std::function<void(float*)>& make) { preview(done, make); };
        d.sample(x, h, w, sigmas, g, first, total, step, ls.after(s_), peek, &r.guide,
                 r.sampler == "er_sde", r.shift > 0 ? r.shift : kShift, r.seed);
        preview.release();
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it, then the encoder's layers
        uint64_t room = decode_room(p, d, vae_->scratch_bytes(h, w));
        if (vae_->scratch_bytes(h, w) > room && te_ && te_->layers_vram() > te_->layer_min()) {
            room += te_->layers_vram() - te_->layer_min();
            te_->set_layer_budget(te_->layer_min());
            te_set_ = te_->layer_min();
        }
        // the room by the device's own count too, less a margin: the scratch's estimate runs a little short (an
        // ai-toolkit edit on the RTX 4070: 3346 MiB estimated in 3394 of room, and a 64 MiB buffer refused) - where it
        // does not fit, the VAE decodes in tiles
        // the budget beside what is held now, where the plan's share is short of it (the encoder's layers it counted but
        // did not take back since a decode before gave them up: the chinoll edit's second request, 832 x 1248, tiled
        // in 1536 MiB of 3346 with 5221 free on the RX 9060 XT)
        if (vae_->scratch_bytes(h, w) > room) {
            const uint64_t others = held() - vae_->scratch_held();
            if (budget() > others) room = std::max(room, budget() - others);
        }
        room = std::min(room, free_vram() + vae_->scratch_held());
        const uint64_t slack = frame_slack();
        room -= std::min(room, slack);
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

}  // namespace strata::image
