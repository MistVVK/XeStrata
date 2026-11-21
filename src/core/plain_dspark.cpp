// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "plain_dspark.hpp"

#include "strata/artifact/gguf_reader.hpp"
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

struct Weight {
    const uint8_t* data = nullptr;
    int type = 0, in = 0, out = 0;
    size_t stride = 0;

    void row(int index, float* dst) const {
        if (!kc::native_row_to_float(type, data + (size_t) index * stride, dst, in))
            throw std::runtime_error("DSpark: host tensor decoding failed");
    }
    void mul(const float* x, float* y) const {
        Vec w((size_t) in);
        for (int i = 0; i < out; ++i) {
            row(i, w.data());
            y[i] = dot(w.data(), x, in);
        }
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
    return {model.shard(shard).tensor_data(*t), (int) t->type, in, std::max(1, out), stride};
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
    for (const auto& l : p_->layer) {
        std::vector<Kv> kv(block);
        std::vector<std::array<float, (size_t) hd * heads>> q(block);
        for (int t = 0; t < block; ++t) {
            Row n = x[(size_t) t];
            norm(n.data(), l.attn_norm.data(), width);
            l.q.mul(n.data(), q[(size_t) t].data());
            l.k.mul(n.data(), kv[(size_t) t].k.data());
            l.v.mul(n.data(), kv[(size_t) t].v.data());
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
        Vec scores(n_past + block), up(ff), gate(ff);
        const float scale = 1.0f / std::sqrt((float) hd);
        const auto cell = [&](size_t j) -> const Kv& { return j < n_past ? l.history[j] : kv[j - n_past]; };
        for (int t = 0; t < block; ++t) {
            std::array<float, (size_t) hd * heads> att{};
            for (int h = 0; h < heads; ++h) {
                const int hkv = h / (heads / kv_heads);
                for (size_t j = 0; j < scores.size(); ++j)
                    scores[j] = scale * dot(q[(size_t) t].data() + (size_t) h * hd, cell(j).k.data() + (size_t) hkv * hd, hd);
                softmax(scores.data(), scores.size());
                for (size_t j = 0; j < scores.size(); ++j)
                    for (int d = 0; d < hd; ++d) att[h * hd + d] += scores[j] * cell(j).v[hkv * hd + d];
            }
            Row o, n;
            l.o.mul(att.data(), o.data());
            for (int i = 0; i < width; ++i) { x[(size_t) t][i] += o[i]; n[i] = x[(size_t) t][i]; }
            norm(n.data(), l.ffn_norm.data(), width);
            l.up.mul(n.data(), up.data());
            l.gate.mul(n.data(), gate.data());
            for (int i = 0; i < ff; ++i) up[i] *= gate[i] * sigmoid(gate[i]);
            l.down.mul(up.data(), o.data());
            for (int i = 0; i < width; ++i) x[(size_t) t][i] += o[i];
        }
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
