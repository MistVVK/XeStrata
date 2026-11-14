// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// The YASAKANI block drafter's weights (engine35.cpp's load_bdraft, written anew).
#include "strata/core/bdraft2.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {
using strata::yskn::Container;
using strata::yskn::TensorDesc;
using strata::yskn::VectorDesc;
using strata::kernels::YsknDev;

const TensorDesc& tensor(const Container& c, const std::string& name, int rows, int cols) {
    const TensorDesc* t = c.find_tensor(name);
    if (!t || t->role != "draft" || t->rows != rows || t->cols != cols || t->has_rot_out)
        throw std::runtime_error("bdraft2: missing or incompatible matrix " + name);
    return *t;
}
const VectorDesc& vector(const Container& c, const std::string& name, int64_t n) {
    const VectorDesc* v = c.find_vector(name);
    if (!v || v->n != n || v->dtype != "bf16")
        throw std::runtime_error("bdraft2: missing or incompatible vector " + name);
    return *v;
}
}  // namespace

Bdraft2Weights::~Bdraft2Weights() { release(); }

void Bdraft2Weights::release() {
    head_.release();
    if (device_ >= 0) {
        const OnDevice on(device_);
        for (void* p : allocations_) gpu::free(p);
    }
    allocations_.clear(); matrices_.clear(); vectors_.clear(); container_.reset();
    device_ = -1; bytes_ = 0; scratch_bytes_ = 0;
    full_head_ = {}; head_mode_ = HeadMode::Auto; probe_shead_ = false;
}

bool Bdraft2Weights::load(const std::string& path, const std::string& head, int device, int target_hidden,
                          int target_layers, int target_vocab, std::string& err) {
    release();
    try {
        auto c = std::make_unique<Container>(path);
        const auto& m = c->model();
        if (device < 0 || m.arch != "bdraft2" || m.layers != 5 || m.hidden != target_hidden || m.hidden != 5120 ||
            m.ffn != 17408 || m.heads != 32 || m.kv_heads != 8 || m.head_dim != 128 || m.vocab != target_vocab ||
            m.block != 8 || m.window != 2048 || m.top_k != 16 || m.rank != 256 || m.mask_token >= m.vocab ||
            m.target_layers.size() != 5 || !std::is_sorted(m.target_layers.begin(), m.target_layers.end()) ||
            std::adjacent_find(m.target_layers.begin(), m.target_layers.end()) != m.target_layers.end() ||
            m.target_layers.back() >= target_layers)
            throw std::runtime_error("bdraft2: incompatible target or drafter geometry");
        std::vector<const TensorDesc*> ts;
        std::vector<const VectorDesc*> vs;
        auto add_t = [&](const std::string& name, int r, int k) { ts.push_back(&tensor(*c, name, r, k)); };
        auto add_v = [&](const std::string& name, int64_t n) { vs.push_back(&vector(*c, name, n)); };
        add_t("bdraft2.fc", m.hidden, 5 * m.hidden);
        add_v("bdraft2.hidden_norm", m.hidden); add_v("bdraft2.norm", m.hidden);
        for (int i = 0; i < m.layers; ++i) {
            const std::string p = "bdraft2.layers." + std::to_string(i) + ".";
            add_t(p + "q", m.heads * m.head_dim, m.hidden);
            add_t(p + "k", m.kv_heads * m.head_dim, m.hidden);
            add_t(p + "v", m.kv_heads * m.head_dim, m.hidden);
            add_t(p + "o", m.hidden, m.heads * m.head_dim);
            add_t(p + "gate", m.ffn, m.hidden); add_t(p + "up", m.ffn, m.hidden); add_t(p + "down", m.hidden, m.ffn);
            add_t(p + "akp", m.hidden / 4, m.hidden); add_t(p + "mkp", m.hidden / 4, m.hidden);
            add_v(p + "ln1", m.hidden); add_v(p + "ln2", m.hidden);
            add_v(p + "q_norm", m.head_dim); add_v(p + "k_norm", m.head_dim);
            add_v(p + "abase", 4 * m.hidden); add_v(p + "mbase", 4 * m.hidden);
        }
        const auto& rotation = tensor(*c, "bdraft2.layers.0.k", m.kv_heads * m.head_dim, m.hidden).rot_in;
        for (int i = 0; i < m.layers; ++i) {
            const std::string p = "bdraft2.layers." + std::to_string(i) + ".";
            for (const char* name : {"k", "v"}) {
                const auto& r = tensor(*c, p + name, m.kv_heads * m.head_dim, m.hidden).rot_in;
                if (r.rule != rotation.rule || r.sign_seed != rotation.sign_seed || r.sign_offset != rotation.sign_offset)
                    throw std::runtime_error("bdraft2: context projections have different input rotations");
            }
        }
        add_v("bdraft2.sel.hp", (int64_t) m.rank * m.hidden);
        add_v("bdraft2.sel.pred", (int64_t) m.vocab * m.rank);
        add_v("bdraft2.sel.succ", (int64_t) m.vocab * m.rank);
        device_ = device;
        const OnDevice on(device_);
        auto put = [&](const void* data, size_t n) -> void* {
            void* p = gpu::alloc_device(n);
            if (!p) throw std::runtime_error("bdraft2: device allocation failed");
            allocations_.push_back(p); bytes_ += n;
            if (!gpu::copy(p, data, n)) throw std::runtime_error("bdraft2: device copy failed");
            return p;
        };
        for (const TensorDesc* t : ts) {
            YsknDev d;
            d.rows = t->physical_rows; d.rows_used = t->rows; d.cols = t->cols; d.bits = t->bits;
            std::vector<uint16_t> lut8;
            std::vector<float> tau;
            strata::yskn::lut8_tau(*c, *t, lut8, tau);
            const auto words = strata::yskn::device_order(*c, *t);
            const auto signs = strata::yskn::sign_stream(t->rot_in.rule, t->rot_in.sign_seed, t->rot_in.sign_offset, (size_t) d.cols);
            d.words = static_cast<const uint32_t*>(put(words.data(), words.size() * sizeof(uint32_t)));
            d.lut8 = static_cast<const uint16_t*>(put(lut8.data(), lut8.size() * sizeof(uint16_t)));
            d.tau = static_cast<const float*>(put(tau.data(), tau.size() * sizeof(float)));
            d.sign_in = static_cast<const int8_t*>(put(signs.data(), signs.size()));
            scratch_bytes_ = std::max(scratch_bytes_, strata::kernels::yskn_scratch_bytes(d, m.block));
            matrices_.emplace(t->name, d);
        }
        for (const VectorDesc* v : vs) {
            const auto* data = reinterpret_cast<const uint16_t*>(c->at(v->off, (uint64_t) v->n * 2));
            if (v->name.find("bdraft2.sel.") == 0) {
                vectors_.emplace(v->name, put(data, (size_t) v->n * 2));
            } else {
                std::vector<float> f((size_t) v->n);
                for (size_t i = 0; i < f.size(); ++i) f[i] = strata::kernels::f32_from_bf16(data[i]);
                vectors_.emplace(v->name, put(f.data(), f.size() * sizeof(float)));
            }
        }
        if (!head.empty() && !head_.load(head, device, m.vocab, m.hidden, m.block - 1, err))
            throw std::runtime_error(err);
        container_ = std::move(c);
        return true;
    } catch (const std::exception& e) {
        err = e.what(); release(); return false;
    }
}

void Bdraft2Weights::project(const std::string& name, const float* x, int rows, float* y, void* scratch, void* stream) const {
    if (!loaded() || rows < 1 || rows > model().block) throw std::invalid_argument("bdraft2: invalid projection rows");
    const OnDevice on(device_);
    const auto& w = matrix(name);
    strata::kernels::yskn_proj(w, x, w.cols, rows, y, w.rows_used, scratch, stream);
}

void Bdraft2Weights::context(const float* taps, int rows, const int32_t* positions, const float* rope, float* const* keys,
                             float* const* values, int slots, int64_t* tag, const int32_t* epoch,
                             const ContextWorkspace& work, void* stream) const {
    if (!loaded() || rows < 1 || rows > model().block || !taps || !work.fc || !work.normalized || !work.quantized ||
        !work.scales || !work.projection)
        throw std::invalid_argument("bdraft2: invalid context arguments");
    const OnDevice on(device_);
    project("bdraft2.fc", taps, rows, work.fc, work.projection, stream);
    strata::kernels::rms_norm_rows(work.fc, static_cast<const float*>(vector("bdraft2.hidden_norm")), 1, work.normalized,
                                  nullptr, model().hidden, rows, model().eps, stream);
    const auto& first = matrix("bdraft2.layers.0.k");
    strata::kernels::yskn_act(first, work.normalized, model().hidden, rows, work.quantized, work.scales, stream);
    context_replay(work.quantized, work.scales, rows, positions, rope, keys, values, slots, tag, epoch, work, stream);
}

void Bdraft2Weights::context_replay(const int8_t* quantized, const float* scales, int rows, const int32_t* positions,
                                    const float* rope, float* const* keys, float* const* values, int slots, int64_t* tag,
                                    const int32_t* epoch, const ContextWorkspace& work, void* stream) const {
    if (!loaded() || rows < 1 || rows > model().block || !quantized || !scales || !positions || !rope || !keys || !values ||
        slots < model().window || !work.key || !work.value || !work.projection || ((tag == nullptr) != (epoch == nullptr)))
        throw std::invalid_argument("bdraft2: invalid context replay arguments");
    for (int i = 0; i < model().layers; ++i)
        if (!keys[i] || !values[i]) throw std::invalid_argument("bdraft2: missing context ring");
    const OnDevice on(device_);
    for (int i = 0; i < model().layers; ++i) {
        const std::string p = "bdraft2.layers." + std::to_string(i) + ".";
        auto proj = [&](const char* name, float* out) {
            const auto& w = matrix(p + name);
            auto* partial = static_cast<int64_t*>(work.projection);
            strata::kernels::yskn_gemv(w, quantized, scales, rows, partial, stream);
            strata::kernels::yskn_finish(w, partial, scales, rows, out, w.rows_used, stream);
        };
        proj("k", work.key); proj("v", work.value);
        strata::kernels::df2_kv_ring(work.key, work.value, static_cast<const float*>(vector(p + "k_norm")), positions,
                                     rope, keys[i], values[i], slots, rows, model().kv_heads, model().eps, stream, tag, epoch);
    }
}

void Bdraft2Weights::block(const float* anchor_embedding, const float* mask_embedding, const int32_t* anchor,
                           const int32_t* anchor_pos, const float* rope, float* const* keys, float* const* values,
                           int slots, const int64_t* tag, const int32_t* epoch, const BlockWorkspace& w, int32_t* out,
                           float* probabilities, void* stream) {
    if (!loaded() || !anchor_embedding || !mask_embedding || !anchor || !anchor_pos || !rope || !keys || !values ||
        slots < model().window || ((tag == nullptr) != (epoch == nullptr)) || !w.h || !w.normalized || !w.conv ||
        !w.dynamic || !w.output || !w.q || !w.qnorm || !w.k || !w.knorm || !w.v || !w.attention || !w.gate || !w.up ||
        !w.glu || !w.final || !w.hp || !w.part || !w.scores || !w.ids || !w.projection || !out || !probabilities)
        throw std::invalid_argument("bdraft2: invalid block arguments");
    for (int i = 0; i < model().layers; ++i)
        if (!keys[i] || !values[i]) throw std::invalid_argument("bdraft2: missing block ring");
    const OnDevice on(device_);
    const int B = model().block, H = model().hidden, group = 16, dyn_stride = 4 * H / group;
    strata::kernels::df2_embed_rows(anchor_embedding, mask_embedding, w.h, H, B, stream);
    for (int i = 0; i < model().layers; ++i) {
        const std::string p = "bdraft2.layers." + std::to_string(i) + ".";
        const auto* abase = static_cast<const float*>(vector(p + "abase"));
        const auto* mbase = static_cast<const float*>(vector(p + "mbase"));
        auto norm = [&](const char* name) {
            strata::kernels::rms_norm_rows(w.h, static_cast<const float*>(vector(p + name)), 1, w.normalized, nullptr,
                                          H, B, model().eps, stream);
        };
        auto proj = [&](const char* name, const float* x, float* y) { project(p + name, x, B, y, w.projection, stream); };
        norm("ln1");
        proj("akp", w.normalized, w.dynamic);
        strata::kernels::df2_conv(w.normalized, w.dynamic, dyn_stride, abase, nullptr, w.conv, B, H, group, stream);
        proj("q", w.conv, w.q); proj("k", w.conv, w.k); proj("v", w.conv, w.v);
        strata::kernels::df2_norm_rope_qk(w.q, w.k, static_cast<const float*>(vector(p + "q_norm")),
                                         static_cast<const float*>(vector(p + "k_norm")), rope, w.qnorm, w.knorm,
                                         B, model().heads, model().kv_heads, model().eps, stream);
        strata::kernels::df2_attention_f32(w.qnorm, keys[i], values[i], w.knorm, w.v, anchor_pos, slots, model().window,
                                           w.part, w.attention, stream, tag, epoch);
        proj("o", w.attention, w.output);
        strata::kernels::df2_conv(w.output, w.dynamic + 2 * H / group, dyn_stride, abase + 2 * H, w.h, w.h,
                                  B, H, group, stream);
        norm("ln2");
        proj("mkp", w.normalized, w.dynamic);
        strata::kernels::df2_conv(w.normalized, w.dynamic, dyn_stride, mbase, nullptr, w.conv, B, H, group, stream);
        proj("gate", w.conv, w.gate); proj("up", w.conv, w.up);
        strata::kernels::native_swiglu_f32(w.gate, w.up, w.glu, model().ffn, B, stream);
        proj("down", w.glu, w.output);
        strata::kernels::df2_conv(w.output, w.dynamic + 2 * H / group, dyn_stride, mbase + 2 * H, w.h, w.h,
                                  B, H, group, stream);
    }
    strata::kernels::rms_norm_rows(w.h + H, static_cast<const float*>(vector("bdraft2.norm")), 1, w.final, nullptr,
                                  H, B - 1, model().eps, stream);
    select(w.final, B - 1, anchor, w.hp, w.ids, w.scores, out, probabilities, stream);
}

bool Bdraft2Weights::release_shead(std::string& err) {
    if (uses_shead()) { err = "bdraft2: cannot release the selected SHEAD"; return false; }
    head_.release();
    return true;
}

void Bdraft2Weights::top512(const float* h, int rows, int32_t* ids, float* values, void* stream) {
    head_.top512(h, rows, ids, values, stream);
}

void Bdraft2Weights::select(const float* h, int rows, const int32_t* anchor, float* hp, int32_t* ids, float* values,
                            int32_t* out, float* probabilities, void* stream) {
    if (!loaded() || rows < 1 || rows >= model().block || !h || !anchor || !hp || !ids || !values || !out || !probabilities)
        throw std::invalid_argument("bdraft2: invalid selector arguments");
    const OnDevice on(device_);
    const auto* w = static_cast<const uint16_t*>(vector("bdraft2.sel.hp"));
    const auto* pred = static_cast<const uint16_t*>(vector("bdraft2.sel.pred"));
    const auto* succ = static_cast<const uint16_t*>(vector("bdraft2.sel.succ"));
    strata::kernels::bf16_gemv_fp32_mmvf_multi(h, model().hidden, w, hp, model().rank, model().hidden, model().rank,
                                              rows, stream);
    if (uses_shead()) {
        if (!head_.loaded()) throw std::runtime_error("bdraft2: SHEAD was selected but not loaded");
        head_.top512(h, rows, ids, values, stream);
    } else {
        const auto& full = full_head_;
        if (!full.weights || !full.projection || !full.logits || !full.top_scratch ||
            full.weights->cols != model().hidden || full.weights->rows_used != model().vocab)
            throw std::runtime_error("bdraft2: the full target head and its scratch are required");
        strata::kernels::yskn_proj(*full.weights, h, model().hidden, rows, full.logits, model().vocab,
                                   full.projection, stream);
        strata::kernels::df2_topk(full.logits, model().vocab, nullptr, rows, full.top_scratch, values, ids, stream);
    }
    strata::kernels::df2_select(hp, values, ids, pred, succ, model().rank, rows, anchor, out, probabilities, stream);
}

}  // namespace strata::core
