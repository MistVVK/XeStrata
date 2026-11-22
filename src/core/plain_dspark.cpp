// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "plain_dspark.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include <cstdio>
#include <string_view>
#include "strata/platform/env.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {
namespace kc = strata::kernels::cpu;
constexpr int width = 2304, rank = 256, vocab = 200058, padded_vocab = 201024;
constexpr int layers = 3, ff = 9216, hd = 128, heads = 24, kv_heads = 8, window = 1024;
constexpr int kv_width = kv_heads * hd;
constexpr std::array<bool, layers> sliding{true, false, true};
constexpr float eps = 1e-6f, rope_base = 1e6f, cap = 30.0f;

using Vec = std::vector<float>;
using Row = std::array<float, width>;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void norm(float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[(size_t) i] * x[(size_t) i];
    const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) x[(size_t) i] *= scale * w[i];
}

float dot(const float* a, const float* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += (double) a[i] * b[i];
    return (float) sum;
}

void softmax(float* x, size_t n) {
    const float peak = *std::max_element(x, x + n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) { x[(size_t) i] = std::exp(x[(size_t) i] - peak); sum += x[(size_t) i]; }
    for (size_t i = 0; i < n; ++i) x[(size_t) i] /= (float) sum;
}

void rope(float* x, int64_t pos) {
    for (int i = 0; i < hd / 2; ++i) {
        const float theta = (float) pos * std::pow(rope_base, -2.0f * (float) i / hd);
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = x[(size_t) i], b = x[i + hd / 2];
        x[(size_t) i] = a*c - b*s;
        x[i + hd / 2] = a*s + b*c;
    }
}

struct DraftDevice {
    struct Free { void operator()(void* p) const { if (p) gpu::free(p); } };
    template<class T> using Buffer = std::unique_ptr<T, Free>;
    template<class T> static Buffer<T> alloc(size_t n) {
        Buffer<T> p(static_cast<T*>(gpu::alloc_device(n * sizeof(T))));
        if (!p) throw std::runtime_error("DSpark: resident buffer allocation failed");
        return p;
    }
    struct Queue {
        gpu::Stream s = gpu::stream_create();
        ~Queue() { if (s) gpu::stream_destroy(s); }
    } queue;
    Buffer<float> input, output;
    bool batch = false;
    struct Cache { Buffer<float> k, v; size_t capacity = 0; };
    std::array<Cache, layers> history;
    Buffer<float> q, noise_k, noise_v, scores, attention;
    // The layers on the device (propose): the rows, their norms, the projections, the FFN and the RoPE table
    bool layered = false;
    Buffer<float> x, normed, proj, up, gate, cs;
    std::array<std::array<Buffer<float>, 4>, layers> norms;   // attn, q, k, ffn
    void copy(void* dst, const void* from, size_t n) const {
        if (!gpu::copy_async(dst, from, n, queue.s)) throw std::runtime_error("DSpark: resident copy failed");
    }
    void wait() const {
        if (!gpu::stream_sync(queue.s)) throw std::runtime_error("DSpark: resident stream failed");
    }
};

struct Weight {
    const uint8_t* data = nullptr;
    int type = 0, in = 0, out = 0;
    size_t stride = 0;
    DraftDevice* device = nullptr;
    DraftDevice::Buffer<float> resident;

    void upload(DraftDevice& dev, uint8_t* raw) {
        resident = DraftDevice::alloc<float>((size_t) in * (size_t) out);
        dev.copy(raw, data, stride * (size_t) out);
        kernels::iq_dequant_f32(type, raw, (int64_t) in * out, resident.get(), dev.queue.s);
        dev.wait();
        device = &dev;
    }

    void row(int index, float* dst) const {
        if (resident) {
            device->copy(dst, resident.get() + (size_t) index * in, (size_t) in * sizeof(float));
            device->wait();
            return;
        }
        if (!kc::native_row_to_float(type, data + (size_t) index * stride, dst, in))
            throw std::runtime_error("DSpark: host tensor decoding failed");
    }
    void mul(const float* x, float* y) const {
        if (resident) {
            device->copy(device->input.get(), x, (size_t) in * sizeof(float));
            if (!kernels::gemv_f32_ordered(device->input.get(), resident.get(), device->output.get(), out, in,
                                            device->queue.s)) throw std::runtime_error("DSpark: ordered GPU GEMV unavailable");
            device->copy(y, device->output.get(), (size_t) out * sizeof(float));
            device->wait();
            return;
        }
        Vec w((size_t) in);
        for (int i = 0; i < out; ++i) {
            row(i, w.data());
            y[i] = dot(w.data(), x, in);
        }
    }
    void mul_block(const float* x, float* y) const {
        if (!resident || !device->batch) {
            for (size_t t = 0; t < PlainDspark::block; ++t) mul(x + t * in, y + t * out);
            return;
        }
        const auto rows = PlainDspark::block;
        device->copy(device->input.get(), x, (size_t) rows * in * sizeof(float));
        if (!kernels::gemm_f32_ordered(device->input.get(), resident.get(), device->output.get(), rows, out, in,
                                        device->queue.s)) throw std::runtime_error("DSpark: ordered GPU product unavailable");
        device->copy(y, device->output.get(), (size_t) rows * out * sizeof(float));
        device->wait();
    }
    void mul_dev(const float* x, float* y, int rows) const {
        if (!kernels::gemm_f32_ordered(x, resident.get(), y, rows, out, in, device->queue.s))
            throw std::runtime_error("DSpark: ordered GPU product unavailable");
    }
    Vec vec() const {
        Vec x((size_t) in);
        row(0, x.data());
        return x;
    }
};

Weight read_weight(const GgufModel& model, const std::string& name, int in, int out = 0) {
    size_t shard = 0;
    const auto* t = model.find(name, &shard);
    if (t == nullptr || t->shape.size() != (out ? 2u : 1u) || t->shape[0] != (uint64_t) in ||
        (out && t->shape[1] != (uint64_t) out) || !model.in_bounds(*t, shard) ||
        (t->type != 0 && t->type != 8 && t->type != 30))
        throw std::runtime_error("DSpark: missing or incompatible tensor " + name);
    const auto stride = kc::native_row_bytes((int) t->type, in);
    if (!stride) throw std::runtime_error("DSpark: invalid row size for " + name);
    return {model.shard(shard).tensor_data(*t), (int) t->type, in, std::max(1, out), stride, nullptr, {}};
}

void check_metadata(const GgufFile& g) {
    const auto eq = [&](const char* key, double expected) {
        const auto* v = g.get(key);
        if (!v || !v->is_num() || v->num() != expected)
            throw std::runtime_error(std::string("DSpark: missing or incompatible metadata ") + key);
    };
    const auto* arch = g.get("general.architecture");
    if (!arch || arch->s != "dflash") throw std::runtime_error("DSpark: expected dflash GGUF");
    eq("dflash.embedding_length", width);
    eq("dflash.block_count", layers);
    eq("dflash.feed_forward_length", ff);
    eq("dflash.vocab_size", padded_vocab);
    eq("dflash.vocab_size_unpadded", vocab);
    eq("dflash.block_size", PlainDspark::block);
    eq("dflash.dspark.markov_rank", rank);
    eq("dflash.attention.head_count", heads);
    eq("dflash.attention.head_count_kv", kv_heads);
    eq("dflash.attention.key_length", hd);
    eq("dflash.attention.value_length", hd);
    eq("dflash.attention.sliding_window", window);
    eq("dflash.attention.causal", 0);
    eq("dflash.rope.dimension_count", hd);
    eq("dflash.rope.freq_base", rope_base);
    eq("dflash.attention.layer_norm_rms_epsilon", eps);
    eq("dflash.final_logit_softcapping", cap);
    eq("dflash.sample_from_anchor", 0);
    eq("dflash.has_confidence_head", 0);
    const auto array_eq = [&](const char* key, const auto& expected) {
        const auto* v = g.get(key);
        if (v && v->type == MetaType::ARRAY && v->items.size() == expected.size() &&
            std::equal(v->items.begin(), v->items.end(), expected.begin(),
                       [](const MetaValue& a, auto b) { return a.is_num() && a.num() == (double) b; })) return;
        throw std::runtime_error(std::string("DSpark: missing or incompatible array ") + key);
    };
    array_eq("dflash.target_layers", PlainDspark::taps);
    array_eq("dflash.attention.sliding_window_pattern", sliding);
}

struct Kv {
    std::array<float, kv_width> k, v;
};

struct Layer {
    Weight q, k, v, o, ctx_k, ctx_v, up, gate, down;
    Vec attn_norm, q_norm, k_norm, ffn_norm, ctx_norm, ctx_k_norm;
    std::deque<Kv> history;
};
}  // namespace

struct PlainDspark::Impl {
    explicit Impl(const std::string& path) : model(GgufModel::open(path)) {}
    GgufModel model;
    std::unique_ptr<DraftDevice> device;
    std::array<Layer, layers> layer;
    std::array<std::array<float, taps.size()>, layers + 1> fuse{};
    Vec mask, last_norm, context_norm, joint_bias, correction_bias;
    Weight markov, joint, correction;
    Row context{};
    int64_t committed = 0;

    Row mix(const float* features, int which, const Vec& w) const {
        Row out{};
        for (size_t t = 0; t < taps.size(); ++t)
            for (int i = 0; i < width; ++i) out[i] += fuse[which][t] * features[t * width + i];
        norm(out.data(), w.data(), width);
        return out;
    }
};

PlainDspark::PlainDspark() = default;
PlainDspark::~PlainDspark() = default;

bool PlainDspark::load(const std::string& path, const GgufModel& target, std::string& err) {
    try {
        auto p = std::make_unique<Impl>(path);
        check_metadata(p->model.meta());
        const auto tokens = p->model.meta().string_array("tokenizer.ggml.tokens");
        if (tokens.size() < (size_t) vocab || tokens != target.meta().string_array("tokenizer.ggml.tokens"))
            throw std::runtime_error("DSpark: target and draft vocabulary tokens differ");
        const auto weight = [&](const std::string& n, int a, int b = 0) { return read_weight(p->model, n, a, b); };
        const auto fuse = weight("dspark_fuse.weight", (int) taps.size(), layers + 1);
        for (int i = 0; i <= layers; ++i) {
            fuse.row(i, p->fuse[i].data());
            softmax(p->fuse[i].data(), taps.size());
        }
        p->mask = weight("mask_embd.weight", width).vec();
        p->last_norm = weight("output_norm.weight", width).vec();
        p->context_norm = weight("dspark_norm.weight", width).vec();
        p->joint = weight("joint_proj.weight", 2 * (width + rank), 3 * rank);
        p->joint_bias = weight("joint_proj.bias", 3 * rank).vec();
        p->correction = weight("dspark_hidden_out.weight", rank, width);
        p->correction_bias = weight("dspark_hidden_out.bias", width).vec();
        p->markov = weight("markov_w1.weight", rank, padded_vocab);
        for (int i = 0; i < layers; ++i) {
            auto& l = p->layer[i];
            const auto w = [&](const char* n, int a, int b = 0) {
                return weight("blk." + std::to_string(i) + "." + n + ".weight", a, b);
            };
            l.q = w("attn_q", width, hd * heads);
            l.k = w("attn_k", width, kv_width);
            l.v = w("attn_v", width, kv_width);
            l.o = w("attn_output", hd * heads, width);
            l.ctx_k = w("attn_k_ctx", width, kv_width);
            l.ctx_v = w("attn_v_ctx", width, kv_width);
            l.up = w("ffn_up", width, ff);
            l.gate = w("ffn_gate", width, ff);
            l.down = w("ffn_down", ff, width);
            l.attn_norm = w("attn_norm", width).vec();
            l.q_norm = w("attn_q_norm", hd).vec();
            l.k_norm = w("attn_k_norm", hd).vec();
            l.ffn_norm = w("ffn_norm", width).vec();
            l.ctx_norm = w("dflare_norm", width).vec();
            l.ctx_k_norm = w("attn_k_ctx_norm", hd).vec();
        }
        p_ = std::move(p);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool PlainDspark::enable_gpu(std::string& err) {
    try {
        size_t free = 0, total = 0;
        if (!gpu::mem_info(&free, &total)) throw std::runtime_error("DSpark: cannot query VRAM budget");
        const uint64_t budget = free - std::min<uint64_t>(free, vram_room(total));
        const uint64_t largest = max_alloc_bytes();
        std::vector<Weight*> matrices{&p_->markov, &p_->joint, &p_->correction};
        for (auto& l : p_->layer)
            matrices.insert(matrices.end(), {&l.q, &l.k, &l.v, &l.o, &l.ctx_k, &l.ctx_v, &l.up, &l.gate, &l.down});
        size_t bytes = 0, raw_bytes = 0, x = 0, y = 0;
        for (const auto* w : matrices) {
            const size_t resident = (size_t) w->in * (size_t) w->out * sizeof(float);
            if (resident > largest || !kernels::embed_type_supported(w->type) || w->in % 256 != 0) return false;
            bytes += resident;
            raw_bytes = std::max(raw_bytes, w->stride * (size_t) w->out);
            x = std::max(x, (size_t) w->in);
            if (w != &p_->markov) y = std::max(y, (size_t) w->out);
        }
        const char* batch_setting = strata::env_get("INKLING_HOSTSTATE");
        const bool batch = !(batch_setting && std::string_view(batch_setting) == "0") &&
                           bytes + raw_bytes + block * (x + y) * sizeof(float) <= budget &&
                           block * std::max(x, y) * sizeof(float) <= largest;
        if (bytes + raw_bytes + (x + y) * sizeof(float) > budget || raw_bytes > largest ||
            !Runtime::get().device().has(sycl::aspect::fp64)) return false;
        auto dev = std::make_unique<DraftDevice>();
        if (!dev->queue.s) throw std::runtime_error("DSpark: cannot create resident stream");
        auto raw = DraftDevice::alloc<uint8_t>(raw_bytes);
        dev->batch = batch;
        dev->input = DraftDevice::alloc<float>(x * (batch ? block : 1));
        dev->output = DraftDevice::alloc<float>(y * (batch ? block : 1));
        std::fprintf(stderr, "xestrata: DSpark ordered product staging: %zu bytes, %d rows\n",
                     (x + y) * (batch ? block : 1) * sizeof(float), batch ? block : 1);
        for (auto* w : matrices) w->upload(*dev, raw.get());
        p_->device = std::move(dev);
        std::fprintf(stderr, "xestrata: DSpark resident: %zu matrices, %zu bytes, staging %zu bytes, budget %llu bytes\n",
                     matrices.size(), bytes, raw_bytes, (unsigned long long) budget);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool PlainDspark::place_attention(int64_t context, std::string& err) {
    auto* dev = p_->device.get();
    if (!dev || context < 1) return true;
    try {
        size_t free = 0, total = 0;
        if (!gpu::mem_info(&free, &total)) throw std::runtime_error("DSpark: cannot query attention budget");
        const uint64_t budget = free - std::min<uint64_t>(free, vram_room(total));
        const uint64_t limit = max_alloc_bytes();
        constexpr size_t q_elements = (size_t) block * heads * hd, noise_elements = (size_t) block * kv_width;
        constexpr size_t layer_elements = 3 * block * width + 2 * block * ff + block * hd +
                                          layers * (2 * width + 2 * hd);
        const bool layered = dev->batch;
        const uint64_t fixed = (2 * (q_elements + noise_elements) + (layered ? layer_elements : 0)) * sizeof(float);
        if (budget <= fixed || limit < q_elements * sizeof(float)) return true;
        const auto capacity = [&](int64_t n, int layer) {
            return (uint64_t) (sliding[layer] ? std::min<int64_t>(n, window) : n);
        };
        const auto pool_bytes = [&](int64_t n) {
            uint64_t bytes = 0;
            for (int i = 0; i < layers; ++i) bytes += 2 * capacity(n, i) * kv_width * sizeof(float);
            return bytes;
        };
        const auto score_bytes = [&](int64_t n) { return (uint64_t) (n + block) * block * heads * sizeof(float); };
        int64_t n = (int64_t) std::min<uint64_t>((uint64_t) context, limit / (kv_width * sizeof(float)));
        while (n > 0 && (score_bytes(n) > limit || pool_bytes(n) + score_bytes(n) > budget - fixed)) n /= 2;
        if (n == 0) return true;
        dev->q = DraftDevice::alloc<float>(q_elements);
        dev->attention = DraftDevice::alloc<float>(q_elements);
        dev->noise_k = DraftDevice::alloc<float>(noise_elements);
        dev->noise_v = DraftDevice::alloc<float>(noise_elements);
        dev->scores = DraftDevice::alloc<float>(score_bytes(n) / sizeof(float));
        for (int i = 0; i < layers; ++i) {
            auto& cache = dev->history[i];
            cache.capacity = (size_t) capacity(n, i);
            cache.k = DraftDevice::alloc<float>(cache.capacity * kv_width);
            cache.v = DraftDevice::alloc<float>(cache.capacity * kv_width);
        }
        if (layered) {
            dev->x = DraftDevice::alloc<float>((size_t) block * width);
            dev->normed = DraftDevice::alloc<float>((size_t) block * width);
            dev->proj = DraftDevice::alloc<float>((size_t) block * width);
            dev->up = DraftDevice::alloc<float>((size_t) block * ff);
            dev->gate = DraftDevice::alloc<float>((size_t) block * ff);
            dev->cs = DraftDevice::alloc<float>((size_t) block * hd);
            for (int i = 0; i < layers; ++i) {
                const auto& l = p_->layer[i];
                const Vec* w[] = {&l.attn_norm, &l.q_norm, &l.k_norm, &l.ffn_norm};
                for (int j = 0; j < 4; ++j) {
                    dev->norms[i][j] = DraftDevice::alloc<float>(w[j]->size());
                    dev->copy(dev->norms[i][j].get(), w[j]->data(), w[j]->size() * sizeof(float));
                }
            }
            dev->wait();
            dev->layered = true;
        }
        std::fprintf(stderr, "xestrata: DSpark attention: %lld/%lld context rows, %llu bytes KV, "
                             "%llu bytes scratch, %llu bytes budget, layers on %s\n", (long long) n, (long long) context,
                     (unsigned long long) pool_bytes(n), (unsigned long long) fixed + score_bytes(n),
                     (unsigned long long) budget, dev->layered ? "the device" : "the host");
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

void PlainDspark::reset() {
    for (auto& l : p_->layer) l.history.clear();
    p_->committed = 0;
}

void PlainDspark::commit(const float* features) {
    for (int i = 0; i < layers; ++i) {
        auto& l = p_->layer[i];
        const Row x = p_->mix(features, i, l.ctx_norm);
        auto& kv = l.history.emplace_back();
        l.ctx_k.mul(x.data(), kv.k.data());
        l.ctx_v.mul(x.data(), kv.v.data());
        for (int h = 0; h < kv_heads; ++h) {
            norm(kv.k.data() + (size_t) h * hd, l.ctx_k_norm.data(), hd);
            rope(kv.k.data() + (size_t) h * hd, p_->committed);
        }
        if (auto* dev = p_->device.get(); dev && dev->history[i].capacity) {
            auto& cache = dev->history[i];
            const size_t cell = (size_t) p_->committed % cache.capacity;
            dev->copy(cache.k.get() + cell * kv_width, kv.k.data(), kv_width * sizeof(float));
            dev->copy(cache.v.get() + cell * kv_width, kv.v.data(), kv_width * sizeof(float));
            dev->wait();
        }
        if (sliding[i] && l.history.size() >= window) l.history.pop_front();
    }
    p_->context = p_->mix(features, layers, p_->context_norm);
    ++p_->committed;
}

// Based on llama.cpp PR #32218's Inkling graph: all eight noise rows participate even if only one proposal
// is wanted. Context KV contains only committed target rows; noise KV lives for this call only.
bool PlainDspark::propose(int32_t anchor, int64_t pos, int count, const Embedding& embedding, const Head& head,
                          std::vector<int32_t>& tokens, std::string& err) const {
    tokens.clear();
    if (!p_ || count < 1 || count >= block || anchor < 0 || anchor >= vocab || pos < 1 || pos != p_->committed) {
        err = "DSpark: invalid proposal or missing committed context";
        return false;
    }
    std::vector<Row> x(block);
    if (!embedding(anchor, x[0].data())) {
        err = "DSpark: could not read the target embedding";
        return false;
    }
    for (int t = 1; t < block; ++t) std::copy(p_->mask.begin(), p_->mask.end(), x[(size_t) t].begin());
    // The layers stay on the device when every context fits its ring: one upload of the rows and one read back.
    // Each step keeps the host's order and rounding (SwiGLU's exp runs on the host). Else the host loop below.
    auto* gdev = p_->device.get();
    bool on_device = gdev && gdev->layered;
    for (int i = 0; i < layers && on_device; ++i)
        on_device = gdev->history[i].capacity > 0 && p_->layer[i].history.size() <= gdev->history[i].capacity;
    if (on_device) {
        std::array<float, (size_t) block * hd> cs;
        for (int t = 0; t < block; ++t)
            for (int i = 0; i < hd / 2; ++i) {
                const float theta = (float) (pos + t) * std::pow(rope_base, -2.0f * (float) i / hd);
                cs[(size_t) (t * hd / 2 + i) * 2] = std::cos(theta);
                cs[(size_t) (t * hd / 2 + i) * 2 + 1] = std::sin(theta);
            }
        Vec rows((size_t) block * width);
        for (size_t t = 0; t < block; ++t) std::copy(x[t].begin(), x[t].end(), rows.data() + t * width);
        gdev->copy(gdev->x.get(), rows.data(), rows.size() * sizeof(float));
        gdev->copy(gdev->cs.get(), cs.data(), sizeof(cs));
        Vec up_host((size_t) block * ff), gate_host((size_t) block * ff);
        const auto norm_rows = [&](const float* in, const DraftDevice::Buffer<float>& w, float* out, int64_t n, int64_t dim) {
            if (!kernels::rows_norm_ordered(in, w.get(), out, n, dim, eps, gdev->queue.s))
                throw std::runtime_error("DSpark: ordered GPU norm unavailable");
        };
        for (int layer = 0; layer < layers; ++layer) {
            const auto& l = p_->layer[layer];
            const auto& w = gdev->norms[layer];
            const auto& cache = gdev->history[layer];
            const int64_t n_past = (int64_t) l.history.size();
            norm_rows(gdev->x.get(), w[0], gdev->normed.get(), block, width);
            l.q.mul_dev(gdev->normed.get(), gdev->q.get(), block);
            l.k.mul_dev(gdev->normed.get(), gdev->noise_k.get(), block);
            l.v.mul_dev(gdev->normed.get(), gdev->noise_v.get(), block);
            norm_rows(gdev->q.get(), w[1], gdev->q.get(), (int64_t) block * heads, hd);
            kernels::rope_rows_table(gdev->q.get(), gdev->cs.get(), block, heads, hd, gdev->queue.s);
            norm_rows(gdev->noise_k.get(), w[2], gdev->noise_k.get(), (int64_t) block * kv_heads, hd);
            kernels::rope_rows_table(gdev->noise_k.get(), gdev->cs.get(), block, kv_heads, hd, gdev->queue.s);
            if (!kernels::ordered_block_attn(gdev->q.get(), cache.k.get(), cache.v.get(), gdev->noise_k.get(),
                                             gdev->noise_v.get(), pos - n_past, n_past, (int64_t) cache.capacity, block,
                                             heads, kv_heads, hd, gdev->scores.get(), gdev->attention.get(), gdev->queue.s))
                throw std::runtime_error("DSpark: GPU attention unavailable");
            l.o.mul_dev(gdev->attention.get(), gdev->proj.get(), block);
            kernels::add_rows_f32(gdev->x.get(), gdev->proj.get(), (int64_t) block * width, gdev->queue.s);
            norm_rows(gdev->x.get(), w[3], gdev->normed.get(), block, width);
            l.up.mul_dev(gdev->normed.get(), gdev->up.get(), block);
            l.gate.mul_dev(gdev->normed.get(), gdev->gate.get(), block);
            // SwiGLU's exp on the host: the device's exp is not the host libm's, and the rows must match the host path
            gdev->copy(up_host.data(), gdev->up.get(), up_host.size() * sizeof(float));
            gdev->copy(gate_host.data(), gdev->gate.get(), gate_host.size() * sizeof(float));
            gdev->wait();
            for (size_t i = 0; i < up_host.size(); ++i) up_host[i] *= gate_host[i] * sigmoid(gate_host[i]);
            gdev->copy(gdev->up.get(), up_host.data(), up_host.size() * sizeof(float));
            l.down.mul_dev(gdev->up.get(), gdev->proj.get(), block);
            kernels::add_rows_f32(gdev->x.get(), gdev->proj.get(), (int64_t) block * width, gdev->queue.s);
        }
        gdev->copy(rows.data(), gdev->x.get(), rows.size() * sizeof(float));
        gdev->wait();
        for (size_t t = 0; t < block; ++t) std::copy_n(rows.data() + t * width, width, x[t].begin());
    }
    for (int layer = 0; layer < layers && !on_device; ++layer) {
        const auto& l = p_->layer[layer];
        std::vector<Kv> kv(block);
        std::vector<std::array<float, (size_t) hd * heads>> q(block);
        Vec normed((size_t) block * width), qb((size_t) block * heads * hd);
        Vec kb((size_t) block * kv_width), vb((size_t) block * kv_width);
        for (size_t t = 0; t < block; ++t) {
            float* n = normed.data() + t * width;
            std::copy(x[t].begin(), x[t].end(), n);
            norm(n, l.attn_norm.data(), width);
        }
        l.q.mul_block(normed.data(), qb.data());
        l.k.mul_block(normed.data(), kb.data());
        l.v.mul_block(normed.data(), vb.data());
        for (int t = 0; t < block; ++t) {
            const size_t at = (size_t) t;
            std::copy_n(qb.data() + at * heads * hd, heads * hd, q[at].data());
            std::copy_n(kb.data() + at * kv_width, kv_width, kv[at].k.data());
            std::copy_n(vb.data() + at * kv_width, kv_width, kv[at].v.data());
            for (int h = 0; h < heads; ++h) {
                norm(q[(size_t) t].data() + (size_t) h * hd, l.q_norm.data(), hd);
                rope(q[(size_t) t].data() + (size_t) h * hd, pos + t);
            }
            for (int h = 0; h < kv_heads; ++h) {
                norm(kv[(size_t) t].k.data() + (size_t) h * hd, l.k_norm.data(), hd);
                rope(kv[(size_t) t].k.data() + (size_t) h * hd, pos + t);
            }
        }
        const size_t n_past = l.history.size();
        Vec scores(n_past + block), up((size_t) block * ff), gate((size_t) block * ff);
        const float scale = 1.0f / std::sqrt((float) hd);
        const auto cell = [&](size_t j) -> const Kv& { return j < n_past ? l.history[j] : kv[j - n_past]; };
        using AttnRow = std::array<float, (size_t) hd * heads>;
        std::vector<AttnRow> gpu_att;
        if (auto* dev = p_->device.get(); dev && dev->history[layer].capacity > 0 &&
            n_past <= dev->history[layer].capacity) {
            const auto& cache = dev->history[layer];
            std::array<float, (size_t) block * kv_width> nk, nv;
            for (size_t t = 0; t < block; ++t) {
                std::copy(kv[t].k.begin(), kv[t].k.end(), nk.data() + t * kv_width);
                std::copy(kv[t].v.begin(), kv[t].v.end(), nv.data() + t * kv_width);
            }
            dev->copy(dev->q.get(), q.data(), block * sizeof(AttnRow));
            dev->copy(dev->noise_k.get(), nk.data(), sizeof(nk));
            dev->copy(dev->noise_v.get(), nv.data(), sizeof(nv));
            if (!kernels::ordered_block_attn(dev->q.get(), cache.k.get(), cache.v.get(), dev->noise_k.get(),
                                             dev->noise_v.get(), pos - (int64_t) n_past, (int64_t) n_past,
                                             (int64_t) cache.capacity, block, heads, kv_heads, hd, dev->scores.get(),
                                             dev->attention.get(), dev->queue.s)) throw std::runtime_error("DSpark: GPU attention unavailable");
            gpu_att.resize(block);
            dev->copy(gpu_att.data(), dev->attention.get(), block * sizeof(AttnRow));
            dev->wait();
        }
        Vec all_att((size_t) block * heads * hd), projected((size_t) block * width);
        for (int t = 0; t < block; ++t) {
            AttnRow att{};
            if (!gpu_att.empty()) att = gpu_att[(size_t) t];
            else for (int h = 0; h < heads; ++h) {
                const int hkv = h / (heads / kv_heads);
                for (size_t j = 0; j < scores.size(); ++j)
                    scores[j] = scale * dot(q[(size_t) t].data() + (size_t) h * hd, cell(j).k.data() + (size_t) hkv * hd, hd);
                softmax(scores.data(), scores.size());
                for (size_t j = 0; j < scores.size(); ++j)
                    for (int d = 0; d < hd; ++d) att[h * hd + d] += scores[j] * cell(j).v[hkv * hd + d];
            }
            std::copy(att.begin(), att.end(), all_att.data() + (size_t) t * heads * hd);
        }
        l.o.mul_block(all_att.data(), projected.data());
        for (size_t t = 0; t < block; ++t) {
            float* n = normed.data() + t * width;
            for (int i = 0; i < width; ++i) { x[t][i] += projected[t * width + i]; n[i] = x[t][i]; }
            norm(n, l.ffn_norm.data(), width);
        }
        l.up.mul_block(normed.data(), up.data());
        l.gate.mul_block(normed.data(), gate.data());
        for (size_t i = 0; i < up.size(); ++i) up[i] *= gate[i] * sigmoid(gate[i]);
        l.down.mul_block(up.data(), projected.data());
        for (size_t t = 0; t < block; ++t)
            for (int i = 0; i < width; ++i) x[t][i] += projected[t * width + i];
    }
    std::array<float, (size_t) 2 * (rank + width)> input{};
    std::copy(p_->context.begin(), p_->context.end(), input.begin() + 2L * rank + width);
    Vec logits(vocab);
    int32_t prev = anchor;
    for (int i = 1; i <= count; ++i) {
        Row hidden = x[(size_t) i];
        norm(hidden.data(), p_->last_norm.data(), width);
        p_->markov.row(prev, input.data() + rank);
        std::copy(hidden.begin(), hidden.end(), input.begin() + 2L * rank);
        std::array<float, (size_t) 3 * rank> joint;
        p_->joint.mul(input.data(), joint.data());
        for (int j = 0; j < 3 * rank; ++j) joint[j] += p_->joint_bias[j];
        for (int j = 0; j < rank; ++j) {
            const float candidate = std::tanh(joint[j + rank]);
            input[j] = candidate + sigmoid(joint[j]) * (input[j] - candidate);
            joint[j + 2L * rank] = std::tanh(joint[j + 2L * rank]);
        }
        Row correction;
        p_->correction.mul(joint.data() + 2L * rank, correction.data());
        for (int j = 0; j < width; ++j) hidden[j] += correction[j] + p_->correction_bias[j];
        // Linearity lets the shared head project base + correction once. The cap follows both terms.
        if (!head(hidden.data(), logits.data())) {
            err = "DSpark: could not run the target LM head";
            return false;
        }
        for (float& z : logits) z = cap * std::tanh(z / cap);
        prev = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        tokens.push_back(prev);
    }
    return true;
}

}  // namespace strata::core
