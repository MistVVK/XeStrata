// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_pipeline.cpp - see include/strata/image/moss_pipeline.hpp.

#include "strata/image/moss_pipeline.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
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

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

uint64_t free_vram() {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    return free;
}

// RGB [3][H][W] in -1 .. 1 of an RGBA8 image at W x H (nearest), over white where it is not opaque
std::vector<float> base_rgb(const QiRef& f, int64_t W, int64_t H) {
    const std::vector<uint8_t> px = resize_nearest(f, W, H);
    const int64_t n = W * H;
    std::vector<float> rgb((size_t) (3 * n));
    for (int64_t q = 0; q < n; ++q) {
        const float a = (float) px[(size_t) (q * 4 + 3)] / 255.f;
        for (int c = 0; c < 3; ++c) {
            const float v = std::nearbyint((float) px[(size_t) (q * 4 + c)] * a + 255.f * (1.f - a));
            rgb[(size_t) (c * n + q)] = v / 127.5f - 1.f;
        }
    }
    return rgb;
}

}  // namespace

std::vector<float> moss_sigmas(int n, float shift) {
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
        for (float& x : v) x = s * x / (1.f + (s - 1.f) * x);
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
                        int q8) {
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
    {
        std::string no;
        base_encoder_ = vae_size_->open_encoder(no, true);
    }
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
    return true;
}

void MossPipeline::use_loras(const QiRequest& r) {
    std::string err;
    const std::string before = dit_->lora_key();
    std::vector<LoraUse> dit;
    lora_split(r.loras, dit, te_loras_);
    if (!dit_->set_loras(dit, err)) throw LoraRejected(err);
    if (te_ && !te_->set_loras(te_loras_, err)) throw LoraRejected(err);
    if (lora_key_of(te_loras_) != te_lora_) {   // the states kept were the encoder's with the other adapters
        last_ids_.clear();
        last_neg_ids_.clear();
        te_lora_ = lora_key_of(te_loras_);
    }
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

// as MfPipeline::want_q8: FP16 when its blocks all fit beside the rest, all of it, and FP16 is the faster here
bool MossPipeline::want_q8(int64_t rows, int n, int64_t max_text) {
    if (q8_ >= 0) return q8_ == 1;
    const MossDit& d16 = *dit16_size_;
    const uint64_t fixed = d16.weight_bytes() + d16.text_bytes() + vae_size_->weight_bytes() +
                           (te_ ? text_fixed() + te_layers_ : 0);
    if (fixed + d16.block_bytes() + d16.activation_bytes(rows, n, max_text) > budget()) return true;
    if ((w8_faster_ < 0 || rows > 2 * w8_rows_) && prefill::Gemm::i8_usable(s_)) {
        const auto [f16, w8] = d16.time_forms(rows);
        if (f16 > 0.0 && w8 > 0.0) {
            w8_faster_ = w8 < f16 * 1.05 ? 1 : 0;   // 8 bits unless clearly the slower: a near tie stays put
            w8_rows_ = rows;
            std::fprintf(stderr, "image engine: MOSS-Image's DiT in %s (its MLP's product at %lld rows: %.2f ms in "
                                 "FP16, %.2f in W8A8)\n", w8_faster_ ? "8 bits" : "16 bits", (long long) rows, 1e3 * f16,
                         1e3 * w8);
        }
    }
    return w8_faster_ == 1;
}

uint64_t MossPipeline::held() const {
    uint64_t b = 0;
    if (vae_) b += vae_->weight_bytes() + vae_->scratch_held() + (vae_->has_encoder() ? vae_->encoder_bytes() : 0);
    if (dit_) {
        b += dit_->weight_bytes() + dit_->lora_bytes() + dit_->scratch_bytes() + dit_->blocks_vram() +
             dit_->activations_held();
        if (dit_->text_loaded()) b += dit_->text_bytes();
    }
    if (te_) b += text_fixed() + te_->layers_vram();
    return b;
}

void MossPipeline::fit(uint64_t reserve) {
    const uint64_t have = held() + free_vram();
    set_budget(have > reserve ? have - reserve : 0);
}

QiNeed MossPipeline::need(int64_t n_text, int64_t width, int64_t height,
                          const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg,
                          const std::vector<LoraUse>& loras) {
    (void) refs;   // MOSS-Image has no edits
    const MossDit& dit = q8_ == 0 ? *dit16_size_ : *dit_size_;   // the least: 8 bits, unless FP16 is asked for
    const MossDit& full = q8_ == 1 ? *dit_size_ : *dit16_size_;  // all resident: FP16's, unless 8 bits are
    const MfVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16, hw = h * w;
    const int64_t max_text = std::max(n_text, n_neg);
    const int n = n_neg > 0 ? 2 : 1;
    // the adapters: their A and B (in VRAM for the request, the blocks streamed or not), and the products' scratch of a
    // step's rows; nothing bound yet, so the DiT's activations here hold none of it
    const uint64_t lora = lora_gpu_bytes(loras, n * (hw + max_text));
    uint64_t base = 0, fixed = vae.weight_bytes() + dit.text_bytes() + lora;
    if (mask && base_encoder_) {   // a base image: its encode, and its latent (and a mask's) kept
        fixed += vae.encoder_bytes() + LatentStart::bytes(dit.latent_channels() * hw, hw, true);
        base = vae.encode_scratch_bytes(h, w) + (uint64_t) (3 * height * width) * 4;
    }
    const uint64_t te = states_given_ ? 0 : text_fixed() + 2 * te_layer_max_;
    QiNeed nd;
    nd.least = fixed + dit.weight_bytes() + dit.scratch_bytes() +
               std::max({dit.activation_bytes(hw, n, max_text, 512) + dit.block_min(), te,
                         vae.least_scratch_bytes(h, w), base});
    nd.full = fixed + full.weight_bytes() + full.scratch_bytes() + full.block_bytes() +
              (states_given_ ? 0 : text_fixed() + te_layers_) +
              std::max({full.activation_bytes(hw, n, max_text), vae.scratch_bytes(h, w), base});
    nd.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    nd.free = free;
    nd.cap = vram_cap(total);
    return nd;
}

void MossPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
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
    p.fixed = d.weight_bytes() + d.lora_bytes() + d.scratch_bytes() +
              (text || d.text_loaded() ? d.text_bytes() : 0) + vae_->weight_bytes() +
              (vae_->has_encoder() ? vae_->encoder_bytes() : 0);
    p.lora = d.lora_bytes();
    p.lora_n = d.lora_count();
    if (p.budget <= p.fixed)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is below the weights that always stay");
    const uint64_t avail = p.budget - p.fixed, dit_min = d.block_min();
    int64_t mlp = 0;
    p.activations = hw > 0 ? d.activation_bytes(hw, n, max_text) : 0;
    for (int64_t rows : {4096, 2048, 1024, 512}) {
        if (hw == 0 || p.activations * 3 <= avail) break;
        mlp = rows;
        p.activations = d.activation_bytes(hw, n, max_text, rows);
    }
    if (avail < p.activations + dit_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image");
    uint64_t te_fixed = te_ ? text_fixed() : 0;
    const uint64_t te_min = te_ ? te_->layer_min() : 0;
    // the encoder stays only beside every block of the DiT: a streamed block costs every step (the RTX 4070's link at
    // x4: 67 s for 50 steps resident, 73 streamed past 15 blocks), opening the encoder again a second or two a prompt
    if (te_ && avail < p.activations + d.block_bytes() + te_fixed + te_min) {   // it goes: its states are made
        gpu::stream_sync(s_);
        te_.reset();
        te_set_ = 0;
        te_fixed = 0;
    }
    p.blocks = clampu(avail - p.activations - te_fixed - te_min, dit_min, d.block_bytes());
    if (p.blocks != dit_set_) {
        d.set_block_budget(p.blocks);
        dit_set_ = p.blocks;
    }
    if (te_) {
        p.layers = clampu(avail - p.activations - p.blocks - te_fixed, te_min, te_->layer_bytes());
        if (p.layers != te_set_) {
            te_->set_layer_budget(p.layers);
            te_set_ = p.layers;
        }
        p.resident_layers = te_->resident_layers();
    }
    p.mlp_rows = mlp;
    if (hw > 0 && mlp != mlp_set_) {
        d.set_mlp_rows(mlp);
        mlp_set_ = mlp;
    }
    if (hw == 0) d.drop_activations();
    p.resident_blocks = d.resident_blocks();
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
    auto t = std::make_unique<MossText>();
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

void MossPipeline::encode(const std::vector<int32_t>& ids, std::vector<float>& out) {
    if ((int64_t) ids.size() > kMaxTokens)
        throw std::invalid_argument("image engine: MOSS-Image's prompt is longer than 4096 tokens");
    open_text();
    const uint64_t before = free_vram() + te_->layers_vram();
    std::string err;
    out = te_->encode(ids, err);
    if (out.empty()) throw std::runtime_error(err);
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
    if (held() + more > budget()) release(false);
    std::string err;
    if (!vae_->open_encoder(err)) throw std::runtime_error(err);
    const std::vector<float> px = base_rgb(base, r.width, r.height);
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
    gpu::free(rgb);
    vae_->drop_scratch();
}

std::vector<uint8_t> MossPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                            const std::function<bool(int, int)>& step,
                                            const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if (!r.refs.empty()) throw std::invalid_argument("image engine: MOSS-Image takes no reference images");
    const int64_t width = dit_size_->text_dim();
    if (r.te_rows > 0 && (int64_t) r.te.size() != r.te_rows * width)
        throw std::invalid_argument("image engine: the prompt's states are not " + std::to_string(width) + " wide");
    if (r.neg_te_rows > 0 && (int64_t) r.neg_te.size() != r.neg_te_rows * width)
        throw std::invalid_argument("image engine: the negative prompt's states are not " + std::to_string(width) +
                                    " wide");
    if (r.te_rows == 0 && r.ids.empty()) throw std::invalid_argument("image engine: no prompt");
    const bool guided = r.cfg != 1.0f && (r.neg_te_rows > 0 || !r.neg_ids.empty());
    const int64_t h = r.height / 16, w = r.width / 16, HW = r.width * r.height;
    const bool mask = !r.mask.rgba.empty();
    const QiRef* base = base_image(r);
    if ((mask || r.strength < 1.0) && base == nullptr)
        throw std::invalid_argument("image engine: a mask or a strength below 1 needs an init image");
    if (mask && r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4))
        throw std::invalid_argument("image engine: the mask is empty");
    if (base != nullptr && !base_encoder_)
        throw std::invalid_argument("image engine: the VAE file has no encoder for a base image");
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
    const int64_t rows0 = r.te_rows > 0 && !own ? r.te_rows : (int64_t) r.ids.size();
    const int64_t rows1 = guided ? (r.neg_te_rows > 0 && !own ? r.neg_te_rows : (int64_t) r.neg_ids.size()) : 0;
    ensure_open(want_q8(h * w, n, std::max(rows0, rows1)));
    MossDit& d = *dit_;
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
    if (r.te_rows == 0 || own) {
        if (r.ids != last_ids_) {
            encode(r.ids, last_states_);
            last_ids_ = r.ids;
            cached = false;
        }
        st[0] = &last_states_;
    }
    if (guided && (r.neg_te_rows == 0 || own)) {
        if (r.neg_ids != last_neg_ids_) {
            encode(r.neg_ids, last_neg_states_);
            last_neg_ids_ = r.neg_ids;
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
    const auto t1 = clock::now();
    // the seed's noise as the reference's ([channels][h][w], PyTorch's CPU randn), as the latent's tokens [h w][channels]
    const int64_t Cl = d.latent_channels(), N = Cl * h * w;
    const auto noise = torch_cpu_randn(r.seed, (uint64_t) N);
    std::vector<float> tokens((size_t) N);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < h * w; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * h * w + q)];
    float *x = nullptr, *rgba = nullptr, *x0 = nullptr;
    if (!gpu::alloc_device(&x, (size_t) N * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4)) {
        gpu::free(x);
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    }
    std::vector<float> out((size_t) (4 * HW));
    std::vector<uint8_t> mask_px;
    try {
        if (mask) {
            mask_px = resize_nearest(r.mask, r.width, r.height);
            ls.set_mask(latent_mask(mask_px, r.width, r.height, 16), tokens, Cl, h * w, true);
        }
        gpu::copy(x, tokens.data(), (size_t) N * 4);
        if (r.strength < 1.0) ls.start(x, sigmas[0], s_);   // img2img: from the base image's latent at that time
        if (stage) stage("sampling");
        // partial images (r.preview): the step's x0 decoded in the room the VAE has beside the DiT now, nothing given
        // up for it; a step whose decode does not fit there is left out
        int shown = 0;
        auto peek = [&](int done, const std::function<void(float*)>& make) {
            if (!r.on_preview || std::find(r.preview.begin(), r.preview.end(), done) == r.preview.end()) return;
            const uint64_t room = p.vae > p.activations ? p.vae - p.activations : 0;
            if (x0 == nullptr && !gpu::alloc_device(&x0, (size_t) N * 4)) return;
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
        d.sample(x, h, w, sigmas, g, first, total, step, ls.after(s_), peek);
        gpu::free(x0);
        x0 = nullptr;
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it, then the encoder's layers
        uint64_t room = p.vae > p.activations ? p.vae - p.activations : 0;
        if (vae_->scratch_bytes(h, w) > room) {
            d.drop_activations();
            room = p.vae;
        }
        if (vae_->scratch_bytes(h, w) > room && te_ && te_->layers_vram() > te_->layer_min()) {
            room += te_->layers_vram() - te_->layer_min();
            te_->set_layer_budget(te_->layer_min());
            te_set_ = te_->layer_min();
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
        gpu::free(x);
        gpu::free(x0);
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
