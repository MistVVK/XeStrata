// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_text.cpp - see include/strata/image/qi_text.hpp.

#include "strata/image/qi_text.hpp"
#include "strata/image/qi_gguf.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

template<class T> T* alloc(std::vector<void*>& owned, int64_t n) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (text): GPU memory: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

}  // namespace

QiText::QiText() = default;

QiText::~QiText() {
    for (void* p : allocs_) gpu::free(p);
}

bool QiText::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        g_ = std::make_unique<GgufFile>(path);
        const MetaValue* arch = g_->get("general.architecture");
        const std::string a = arch ? arch->s : "";
        auto num = [&](const std::string& key, double def) {
            const MetaValue* v = g_->get(a + "." + key);
            return v ? v->num() : def;
        };
        const auto layers = (int64_t) num("block_count", 0);
        C_ = (int64_t) num("embedding_length", 4096);
        heads_ = (int64_t) num("attention.head_count", 32);
        kv_heads_ = (int64_t) num("attention.head_count_kv", 8);
        head_dim_ = (int64_t) num("attention.key_length", 128);
        inter_ = (int64_t) num("feed_forward_length", 12288);
        theta_ = (float) num("rope.freq_base", 5e6);
        eps_ = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
        if (const MetaValue* sec = g_->get(a + ".rope.dimension_sections"); sec != nullptr && sec->items.size() >= 3) {
            sec_h_ = (int) sec->items[1].num();
            sec_w_ = (int) sec->items[2].num();
        }
        if (layers <= 0) throw std::runtime_error("image engine (text): " + path + " has no blocks");
        const TensorInfo* e = find_checked(*g_, "token_embd.weight");
        if (e == nullptr || e->shape.size() < 2) throw std::runtime_error("image engine (text): no token_embd.weight");
        embd_type_ = (int) e->type;
        embd_row_bytes_ = tensor_payload_bytes(*e) / e->shape[1];
        embd_ = g_->tensor_data(*e);
        int64_t scratch = 0;
        for (int64_t i = 0; i < layers; ++i) {   // the layers: spans of the mapped file, through layers_stream_
            const std::string p = "blk." + std::to_string(i) + ".";
            Layer l;
            std::vector<LayerStream::Span> spans;
            auto span = [&](const std::string& name, W* w) {
                const std::string full = std::string(p).append(name);
                const TensorInfo* t = find_checked(*g_, full);
                if (t == nullptr) throw std::runtime_error(std::string("image engine (text): the encoder has no ").append(full));
                if (w != nullptr) {
                    w->type = (int) t->type;
                    w->k = (int64_t) t->shape[0];
                    w->n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
                } else if (t->type != 0) {
                    throw std::runtime_error(std::string("image engine (text): ").append(full).append(" is not FP32"));
                }
                spans.push_back({g_->tensor_data(*t), tensor_payload_bytes(*t), 0});
                return (int) spans.size() - 1;
            };
            const int iq = span("attn_q.weight", &l.q), ik = span("attn_k.weight", &l.k),
                      iv = span("attn_v.weight", &l.v), io = span("attn_output.weight", &l.o),
                      ig = span("ffn_gate.weight", &l.gate), iu = span("ffn_up.weight", &l.up),
                      id = span("ffn_down.weight", &l.down), ian = span("attn_norm.weight", nullptr),
                      ifn = span("ffn_norm.weight", nullptr), iqn = span("attn_q_norm.weight", nullptr),
                      ikn = span("attn_k_norm.weight", nullptr);
            const int layer = layers_stream_.add(spans);
            auto off = [&](int k) { return layers_stream_.offset(layer, k); };
            l.q.d = reinterpret_cast<const void*>(off(iq));
            l.k.d = reinterpret_cast<const void*>(off(ik));
            l.v.d = reinterpret_cast<const void*>(off(iv));
            l.o.d = reinterpret_cast<const void*>(off(io));
            l.gate.d = reinterpret_cast<const void*>(off(ig));
            l.up.d = reinterpret_cast<const void*>(off(iu));
            l.down.d = reinterpret_cast<const void*>(off(id));
            l.attn_norm = off(ian);
            l.ffn_norm = off(ifn);
            l.qn = off(iqn);
            l.kn = off(ikn);
            for (const W* w : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) scratch = std::max(scratch, w->n * w->k);
            layers_.push_back(l);
        }
        if (dry) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        layers_stream_.configure(layers_stream_.total(), s_);   // all resident until a budget is set
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

void QiText::ensure(int64_t T) {
    if (T <= cap_T_) return;
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) h_, (void*) gu_, (void*) h16_,
                    (void*) q16_, (void*) k16_, (void*) v16_, (void*) rows_})
        release(allocs_, p);
    const int64_t kv = kv_heads_ * head_dim_, qd = heads_ * head_dim_;
    x_ = alloc<float>(allocs_, T * C_);
    q_ = alloc<float>(allocs_, T * qd);
    k_ = alloc<float>(allocs_, T * kv);
    v_ = alloc<float>(allocs_, T * kv);
    a_ = alloc<float>(allocs_, T * qd);
    h_ = alloc<float>(allocs_, T * C_);
    gu_ = alloc<float>(allocs_, T * 2 * inter_);
    h16_ = alloc<uint16_t>(allocs_, T * std::max({C_, inter_, qd}));
    // Q, K and V in FP16 hold their rows rounded up as the shared attention reads them (qi_attn_rows), zero at first
    // (the rows past a prompt then hold an earlier prompt's values, finite, which the mask leaves out)
    const int64_t TP = qi_attn_rows(T);
    q16_ = alloc<uint16_t>(allocs_, TP * qd);
    k16_ = alloc<uint16_t>(allocs_, TP * kv);
    v16_ = alloc<uint16_t>(allocs_, TP * kv);
    for (uint16_t* p : {q16_, k16_, v16_}) gpu::memset(p, 0, (size_t) (TP * (p == q16_ ? qd : kv)) * 2);
    rows_ = alloc<uint8_t>(allocs_, (int64_t) (T * (int64_t) embd_row_bytes_));
    cap_T_ = T;
}

uint64_t QiText::activation_bytes(int64_t T) const {
    const int64_t kv = kv_heads_ * head_dim_, qd = heads_ * head_dim_;
    return (uint64_t) T * ((uint64_t) (2 * C_ + 2 * qd + 2 * kv + 2 * inter_) * 4 +
                           (uint64_t) std::max({C_, inter_, qd}) * 2 + embd_row_bytes_) +
           (uint64_t) qi_attn_rows(T) * (uint64_t) (qd + 2 * kv) * 2;
}

void QiText::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) h_, (void*) gu_, (void*) h16_,
                    (void*) q16_, (void*) k16_, (void*) v16_, (void*) rows_})
        release(allocs_, p);
    x_ = q_ = k_ = v_ = a_ = h_ = gu_ = nullptr;
    h16_ = q16_ = k16_ = v16_ = nullptr;
    rows_ = nullptr;
    cap_T_ = 0;
}

void QiText::set_final_norm(bool on) {
    if (on && out_norm_.empty()) {
        const TensorInfo* t = find_checked(*g_, "output_norm.weight");
        if (t == nullptr || t->type != 0 || (int64_t) t->elements() != C_)
            throw std::runtime_error("image engine (text): the encoder has no FP32 output_norm.weight");
        out_norm_.resize((size_t) C_);
        std::memcpy(out_norm_.data(), g_->tensor_data(*t), (size_t) C_ * 4);
    }
    final_norm_ = on;
}

void QiText::lora_add(const W& w, const uint16_t* x16, int64_t T, float* y, int64_t ldy) {
    if (w.lora == nullptr) return;
    if (lora_f32_ == nullptr || T > lora_t_) {
        gpu::stream_sync(s_);
        release(allocs_, lora_f32_);
        release(allocs_, lora16_);
        lora_f32_ = alloc<float>(allocs_, T * lora_cols_);
        lora16_ = alloc<uint16_t>(allocs_, T * lora_cols_);
        lora_t_ = T;
    }
    w.lora->add(x16, T, y, ldy, 0, w.n, nullptr, gemm_, lora_f32_, lora16_, s_);
}

uint64_t QiText::lora_bytes() const {
    uint64_t n = 0;
    for (const auto& l : lora_layers_) n += l->bytes();
    return n + (uint64_t) lora_t_ * (uint64_t) lora_cols_ * 6;
}

bool QiText::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    const std::string key = lora_key_of(loras);
    if (key == lora_key_) return true;
    auto unbind = [this] {   // the set bound before (its layers and scratch)
        for (Layer& l : layers_)
            for (W* w : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) w->lora = nullptr;
        gpu::stream_sync(s_);
        lora_layers_.clear();
        lora_files_.clear();
        lora_key_.clear();
        release(allocs_, lora_f32_);
        release(allocs_, lora16_);
        lora_f32_ = nullptr;
        lora16_ = nullptr;
        lora_cols_ = lora_t_ = 0;
    };
    unbind();
    if (key.empty()) return true;
    // the encoder's weights by their names, dotted as lora_text_encoder_name writes a key
    std::vector<std::pair<std::string, W*>> slots;
    for (size_t i = 0; i < layers_.size(); ++i) {
        Layer& l = layers_[i];
        const std::string p = "layers." + std::to_string(i) + ".";
        for (auto [name, w] : std::vector<std::pair<const char*, W*>>{
                 {"self.attn.q.proj", &l.q}, {"self.attn.k.proj", &l.k}, {"self.attn.v.proj", &l.v},
                 {"self.attn.o.proj", &l.o}, {"mlp.gate.proj", &l.gate}, {"mlp.up.proj", &l.up},
                 {"mlp.down.proj", &l.down}})
            slots.emplace_back(p + name, w);
    }
    auto fail = [&](const std::string& why) {
        err = why;
        unbind();
        return false;
    };
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        for (const LoraEntry& e : u.file->entries()) {
            const std::string name = lora_text_encoder_name(e.key);
            auto it = std::find_if(slots.begin(), slots.end(), [&](const auto& s) { return s.first == name; });
            if (it == slots.end())
                return fail("image engine: " + u.file->path() + ": " + e.key + ": the text encoder has no such weight");
            W& w = *it->second;
            if (e.kind == LoraKind::oft || e.kind == LoraKind::glora || e.kind == LoraKind::bias || !e.dora.empty() ||
                !e.bias.empty() || e.replaces)
                return fail("image engine: " + u.file->path() + ": " + e.key +
                            ": the text encoder takes a plain LoRA, a LoKr, a LoHa or a whole delta");
            if (e.rows != w.n || e.cols != w.k)
                return fail("image engine: " + u.file->path() + ": " + e.key + " is " + std::to_string(e.cols) + " x " +
                            std::to_string(e.rows) + ", not the text encoder's " + std::to_string(w.k) + " x " +
                            std::to_string(w.n));
            const float scale = u.scale * e.scale();
            if (scale == 0.0f) continue;
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, s_, err)) return fail(err);
            if (w.lora == nullptr) {
                lora_layers_.push_back(std::make_unique<LoraLayer>());
                w.lora = lora_layers_.back().get();
            }
            LoraPart part;
            part.entry = dev;
            part.n = w.n;
            part.scale = scale;
            w.lora->parts.push_back(std::move(part));
            lora_cols_ = std::max(lora_cols_, lora_scratch_cols(e));
        }
        lora_files_.push_back(u.file);
    }
    lora_key_ = key;
    return true;
}

std::vector<float> QiText::encode(const std::vector<int32_t>& ids, const std::vector<Image>& images) {
    const auto T = (int64_t) ids.size();
    if (T == 0) return {};
    ensure(T);
    // the positions: each token its index, the images' tokens (time, row, column) from their first index, the tokens
    // after an image moved on by its longer side less its count (Qwen3-VL's M-RoPE, as stable-diffusion.cpp builds it);
    // with flat positions every token its index
    std::vector<int32_t> pos3;
    int32_t* dpos = nullptr;
    std::vector<float*> ds;   // the images' DeepStack blocks on the GPU, image by image, layer by layer
    if (!images.empty()) {
        pos3.resize((size_t) (3 * T));
        for (int64_t t = 0; t < T; ++t) pos3[(size_t) t] = pos3[(size_t) (T + t)] = pos3[(size_t) (2 * T + t)] = (int32_t) t;
        int64_t offset = 0;
        for (const Image& im : images) {
            if (im.index < 0 || im.index + im.rows > T || im.rows != im.grid_h * im.grid_w)
                throw std::invalid_argument("image engine (text): an image's tokens do not fit the prompt");
            if (!flat_positions_) {
                const int64_t end = im.index + im.rows, next = im.index + std::max(im.grid_h, im.grid_w) + offset;
                for (int64_t t = end; t < T; ++t)
                    for (int a = 0; a < 3; ++a) pos3[(size_t) (a * T + t)] = (int32_t) (next + t - end);
                for (int64_t r = 0; r < im.rows; ++r) {
                    const int64_t base = im.index + offset;
                    pos3[(size_t) (im.index + r)] = (int32_t) base;
                    pos3[(size_t) (T + im.index + r)] = (int32_t) (base + r / im.grid_w);
                    pos3[(size_t) (2 * T + im.index + r)] = (int32_t) (base + r % im.grid_w);
                }
                offset += std::max(im.grid_h, im.grid_w) - im.rows;
            }
            for (const float* d : im.deepstack) {
                float* g = alloc<float>(allocs_, im.rows * C_);
                gpu::copy(g, d, (size_t) (im.rows * C_) * 4);
                ds.push_back(g);
            }
        }
        dpos = alloc<int32_t>(allocs_, 3 * T);
        gpu::copy(dpos, pos3.data(), pos3.size() * 4);
    }
    // the prompt's embedding rows, gathered from the mapped file, expanded on the GPU
    std::vector<uint8_t> host((size_t) (T * (int64_t) embd_row_bytes_));
    for (int64_t t = 0; t < T; ++t)
        std::memcpy(&host[(size_t) t * embd_row_bytes_], embd_ + (uint64_t) ids[(size_t) t] * embd_row_bytes_,
                    embd_row_bytes_);
    gpu::copy(rows_, host.data(), host.size());
    kernels::dequant_f32(embd_type_, rows_, 0, T, C_, x_, s_);
    for (const Image& im : images)   // the image pads' rows: the vision encoder's embeddings
        gpu::copy_async(x_ + im.index * C_, im.main, (size_t) (im.rows * C_) * 4, s_);
    const bool sdpa_causal = qi_sdpa_causal(s_);
    for (int li = 0; li < (int) layers_.size(); ++li) {
        const Layer& l = layers_[(size_t) li];
        const uint8_t* base = layers_stream_.begin(li);
        auto w = [base](const W& x) { return base + reinterpret_cast<uintptr_t>(x.d); };
        auto f = [base](uint64_t off) { return reinterpret_cast<const float*>(base + off); };
        kernels::rms_norm_rows(x_, f(l.attn_norm), 1, nullptr, h16_, (int) (C_), (int) (T), 1e-6f, s_);
        gemm_.native(h16_, l.q.type, w(l.q), q_, T, l.q.n, l.q.k);
        gemm_.native(h16_, l.k.type, w(l.k), k_, T, l.k.n, l.k.k);
        gemm_.native(h16_, l.v.type, w(l.v), v_, T, l.v.n, l.v.k);
        lora_add(l.q, h16_, T, q_, l.q.n);
        lora_add(l.k, h16_, T, k_, l.k.n);
        lora_add(l.v, h16_, T, v_, l.v.n);
        qi_text_qk_rope16(q_, k_, v_, f(l.qn), f(l.kn), T, heads_, kv_heads_, head_dim_, theta_, q16_, k16_, v16_, s_,
                          dpos, sec_h_, sec_w_);
        // the attention through a GPU maker's kernel where one takes it causal (K and V repeated to a query head each,
        // in a_'s room, which then is not needed), straight to FP16; else the engine's own kernel
        bool done = false;
        if (sdpa_causal) {
            auto* kx = reinterpret_cast<uint16_t*>(a_);
            uint16_t* vx = kx + T * heads_ * head_dim_;
            qi_repeat_kv16(k16_, T, heads_, kv_heads_, head_dim_, kx, s_);
            qi_repeat_kv16(v16_, T, heads_, kv_heads_, head_dim_, vx, s_);
            done = qi_sdpa(q16_, kx, vx, 0, T, T, heads_, head_dim_, h16_, s_, true);
        }
        if (!done) {   // the attention the plain models' prompt shares (flash_attn.hpp), on the matrix engines
            kernels::FlashAttnArgs fa;
            fa.q = q16_;
            fa.q_ld = heads_ * head_dim_;
            fa.k = k16_;
            fa.v = v16_;
            fa.kv_ld = kv_heads_ * head_dim_;
            fa.n_head = heads_;
            fa.n_kv = kv_heads_;
            fa.head_dim = head_dim_;
            fa.rows = fa.keys = fa.n_causal = fa.row_end = T;
            fa.out = h16_;
            fa.out_ld = heads_ * head_dim_;
            fa.scale = 1.0f / std::sqrt((float) head_dim_);
            done = kernels::flash_attn(fa, s_);
        }
        if (!done) {
            qi_attention_causal(q16_, k16_, v16_, T, heads_, kv_heads_, head_dim_, a_, s_);
            kernels::f32_to_f16_bulk(a_, h16_, T * heads_ * head_dim_, s_);
        }
        gemm_.native(h16_, l.o.type, w(l.o), h_, T, l.o.n, l.o.k);
        lora_add(l.o, h16_, T, h_, l.o.n);
        qi_axpy(x_, h_, 1.0f, T * C_, s_);
        kernels::rms_norm_rows(x_, f(l.ffn_norm), 1, nullptr, h16_, (int) (C_), (int) (T), 1e-6f, s_);
        gemm_.native(h16_, l.gate.type, w(l.gate), gu_, T, l.gate.n, l.gate.k, 2 * inter_);
        gemm_.native(h16_, l.up.type, w(l.up), gu_ + inter_, T, l.up.n, l.up.k, 2 * inter_);
        lora_add(l.gate, h16_, T, gu_, 2 * inter_);
        lora_add(l.up, h16_, T, gu_ + inter_, 2 * inter_);
        prefill::swiglu_pair(gu_, gu_ + inter_, h16_, T, s_, inter_, 2 * inter_);
        gemm_.native(h16_, l.down.type, w(l.down), h_, T, l.down.n, l.down.k);
        lora_add(l.down, h16_, T, h_, l.down.n);
        qi_axpy(x_, h_, 1.0f, T * C_, s_);
        layers_stream_.end(li);
        size_t k = 0;   // DeepStack: after layer li, the images' tokens get their block li
        for (const Image& im : images)
            for (size_t d = 0; d < im.deepstack.size(); ++d, ++k)
                if ((int) d == li) qi_axpy(x_ + im.index * C_, ds[k], 1.0f, im.rows * C_, s_);
    }
    std::vector<float> out((size_t) (T * C_));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), x_, out.size() * 4);
    for (float* g : ds) release(allocs_, g);
    release(allocs_, dpos);
    if (final_norm_)   // the final RMS norm, on the host: a prompt's few rows
        for (int64_t t = 0; t < T; ++t) {
            float* r = out.data() + t * C_;
            double ss = 0;
            for (int64_t c = 0; c < C_; ++c) ss += (double) r[c] * r[c];
            const auto inv = (float) (1.0 / std::sqrt(ss / (double) C_ + (double) eps_));
            for (int64_t c = 0; c < C_; ++c) r[c] = r[c] * inv * out_norm_[(size_t) c];
        }
    return out;
}

}  // namespace strata::image
