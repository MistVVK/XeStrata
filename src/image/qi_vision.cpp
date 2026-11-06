// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_vision.cpp - see include/strata/image/qi_vision.hpp.
//
// Qwen3-VL's ViT as stable-diffusion.cpp runs it for Qwen-Image-2.1's edits: 16 x 16 patches (the two temporal
// kernels of the conv3d summed: an image is its own second frame) in the order of the 2 x 2 merge (merged row, merged
// column, row in the block, column in it), the learned 48 x 48 position embeddings resized bilinearly (corners
// aligned), 2D rotary embeddings (half the pairs by row, half by column), blocks of LayerNorm, attention over every
// patch and a GELU (tanh) MLP; after the DeepStack layers a merger of 4 patches (LayerNorm over the 4, GELU (erf) MLP),
// and at the end the final LayerNorm per patch and the same merger.

#include "strata/image/qi_vision.hpp"
#include "strata/image/qi_gguf.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr int kTypeF16 = 1, kTypeF32 = 0;

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (vision): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (vision): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void put(void* d, const void* h, size_t bytes) {
    if (!gpu::copy(d, h, bytes))
        throw std::runtime_error(std::string("image engine (vision): copy: ") + gpu::last_error());
}

}  // namespace

QiVision::QiVision() = default;

QiVision::~QiVision() {
    drop_scratch();
    for (void* p : allocs_) gpu::free(p);
}

QiVision::W QiVision::linear(const std::string& name) {
    const TensorInfo* t = find_checked(*g_, name + ".weight");
    if (t == nullptr || t->shape.size() != 2) throw std::runtime_error("image engine (vision): no " + name);
    W w;
    w.k = (int64_t) t->shape[0];
    w.n = (int64_t) t->shape[1];
    const TensorInfo* b = find_checked(*g_, name + ".bias");
    if (b != nullptr && ((int) b->type != kTypeF32 || (int64_t) b->elements() != w.n))   // #29: read as w.n floats
        throw std::runtime_error("image engine (vision): " + name + ".bias does not match its weight");
    weight_bytes_ += (uint64_t) (w.n * w.k) * 2 + (b ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    if ((int) t->type == kTypeF16) std::memcpy(h.data(), g_->tensor_data(*t), h.size() * 2);
    else if ((int) t->type == kTypeF32) {
        const auto* f = reinterpret_cast<const float*>(g_->tensor_data(*t));
        for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(f[i]);
    } else {
        throw std::runtime_error("image engine (vision): " + name + ": type " + std::to_string(t->type));
    }
    uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
    put(d, h.data(), h.size() * 2);
    w.w = d;
    if (b != nullptr) {
        float* db = alloc<float>(allocs_, w.n);
        put(db, g_->tensor_data(*b), (size_t) w.n * 4);
        w.b = db;
    }
    return w;
}

QiVision::Norm QiVision::norm(const std::string& name) {
    Norm n;
    for (const char* part : {".weight", ".bias"}) {
        const TensorInfo* t = find_checked(*g_, name + part);
        if (t == nullptr || (int) t->type != kTypeF32) throw std::runtime_error("image engine (vision): no " + name);
        const auto c = (int64_t) t->elements();
        weight_bytes_ += (uint64_t) c * 4;
        if (dry_) continue;
        float* d = alloc<float>(allocs_, c);
        put(d, g_->tensor_data(*t), (size_t) c * 4);
        (part[1] == 'w' ? n.w : n.b) = d;
    }
    return n;
}

bool QiVision::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        g_ = std::make_unique<GgufFile>(path);
        auto num = [&](const std::string& key, double def) {
            const MetaValue* v = g_->get(key);
            return v ? v->num() : def;
        };
        C_ = (int64_t) num("clip.vision.embedding_length", 1152);
        heads_ = (int64_t) num("clip.vision.attention.head_count", 16);
        ffn_ = (int64_t) num("clip.vision.feed_forward_length", 4304);
        patch_ = (int64_t) num("clip.vision.patch_size", 16);
        merge_ = (int64_t) num("clip.vision.spatial_merge_size", 2);
        out_dim_ = (int64_t) num("clip.vision.projection_dim", 4096);
        eps_ = (float) num("clip.vision.attention.layer_norm_epsilon", 1e-6);
        if (C_ <= 0 || heads_ <= 0 || patch_ <= 0) throw std::runtime_error("image engine (vision): bad shape in " + path);
        hd_ = C_ / heads_;
        if (hd_ > 128 || hd_ % 4 != 0) throw std::runtime_error("image engine (vision): head size " + std::to_string(hd_));
        const auto layers = (int64_t) num("clip.vision.block_count", 27);
        // the patch embedding: the two temporal kernels summed (the image is its own second frame)
        const TensorInfo* p0 = find_checked(*g_, "v.patch_embd.weight");
        const TensorInfo* p1 = find_checked(*g_, "v.patch_embd.weight.1");
        const TensorInfo* pb = find_checked(*g_, "v.patch_embd.bias");
        if (p0 == nullptr || pb == nullptr || (int) p0->type != kTypeF16)
            throw std::runtime_error("image engine (vision): no FP16 patch embedding");
        const int64_t pk = 3 * patch_ * patch_;
        // #29: the reads below take their lengths from the header, so the tensors must be that long
        if ((int64_t) p0->elements() != C_ * pk || (p1 != nullptr && (p1->type != p0->type || p1->elements() != p0->elements()))
            || (int) pb->type != kTypeF32 || (int64_t) pb->elements() != C_)
            throw std::runtime_error("image engine (vision): the patch embedding does not match the header");
        weight_bytes_ += (uint64_t) (C_ * pk) * 2 + (uint64_t) C_ * 4;
        if (!dry_) {
            std::vector<uint16_t> h((size_t) (C_ * pk));
            const auto* a = reinterpret_cast<const uint16_t*>(g_->tensor_data(*p0));
            const auto* b = p1 ? reinterpret_cast<const uint16_t*>(g_->tensor_data(*p1)) : nullptr;
            for (size_t i = 0; i < h.size(); ++i)
                h[i] = kernels::f16_from_f32(kernels::f32_from_f16(a[i]) + (b ? kernels::f32_from_f16(b[i]) : 0.f));
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
            put(d, h.data(), h.size() * 2);
            patch_w_ = d;
            float* db = alloc<float>(allocs_, C_);
            put(db, g_->tensor_data(*pb), (size_t) C_ * 4);
            patch_b_ = db;
        }
        const TensorInfo* pe = find_checked(*g_, "v.position_embd.weight");
        if (pe == nullptr || (int) pe->type != kTypeF32 || pe->shape.size() < 2)
            throw std::runtime_error("image engine (vision): no position embeddings");
        grid_ = (int64_t) std::lround(std::sqrt((double) pe->shape[1]));
        if (grid_ * grid_ * C_ > (int64_t) pe->elements())
            throw std::runtime_error("image engine (vision): the position embeddings are shorter than the header says");
        pos_.resize((size_t) (grid_ * grid_ * C_));
        std::memcpy(pos_.data(), g_->tensor_data(*pe), pos_.size() * 4);
        for (int64_t i = 0; i < layers; ++i) {
            const std::string p = "v.blk." + std::to_string(i) + ".";
            Layer l;
            l.ln1 = norm(p + "ln1");
            l.ln2 = norm(p + "ln2");
            l.qkv = linear(p + "attn_qkv");
            l.o = linear(p + "attn_out");
            l.up = linear(p + "ffn_up");
            l.down = linear(p + "ffn_down");
            const std::string ds = "v.deepstack." + std::to_string(i) + ".";
            if (find_checked(*g_, ds + "fc1.weight") != nullptr) {
                l.deepstack = (int) deepstack_.size();
                deepstack_.push_back({norm(ds + "norm"), linear(ds + "fc1"), linear(ds + "fc2")});
            }
            layers_.push_back(l);
        }
        final_ = {norm("v.post_ln"), linear("mm.0"), linear("mm.2")};
        if (final_.fc2.n != out_dim_) throw std::runtime_error("image engine (vision): the merger's output size");
        if (dry_) return true;
        return gemm_.init(s_, 16, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

uint64_t QiVision::scratch_bytes(int64_t patches) const {
    const int64_t T = patches, TP = qi_attn_rows(T), wide = std::max(3 * C_, ffn_);
    return (uint64_t) T * ((uint64_t) C_ * 4 + (uint64_t) wide * 4 + (uint64_t) heads_ * 128 * 4 +
                           (uint64_t) std::max(C_, ffn_) * 2 + (uint64_t) heads_ * 128 * 2 + (uint64_t) C_ * 2 + 8) +
           (uint64_t) TP * (uint64_t) heads_ * 128 * 2 * 3;
}

void QiVision::drop_scratch() {
    for (void* p : work_) gpu::free(p);
    work_.clear();
    cap_ = 0;
}

void QiVision::grow(int64_t patches) {
    if (patches <= cap_) return;
    drop_scratch();
    const int64_t T = patches, TP = qi_attn_rows(T);
    x_ = alloc<float>(work_, T * C_);
    y_ = alloc<float>(work_, T * std::max(3 * C_, ffn_));
    big_ = alloc<float>(work_, T * heads_ * 128);
    h16_ = alloc<uint16_t>(work_, T * std::max({C_, ffn_, 3 * patch_ * patch_}));
    q16_ = alloc<uint16_t>(work_, TP * heads_ * 128, true);   // the rows past T stay zero
    k16_ = alloc<uint16_t>(work_, TP * heads_ * 128, true);
    v16_ = alloc<uint16_t>(work_, TP * heads_ * 128, true);
    a16_ = alloc<uint16_t>(work_, T * heads_ * 128);
    o16_ = alloc<uint16_t>(work_, T * C_);
    yx_ = alloc<int32_t>(work_, 2 * T);
    cap_ = patches;
}

void QiVision::product(const uint16_t* x16, const W& w, float* y, int64_t T) {
    gemm_.f16(x16, w.w, y, T, w.n, w.k);
    if (w.b != nullptr) qi_bias_add(y, w.b, T, w.n, s_);
}

// rows tokens of in_dim (x, FP32, the merged tokens' patches side by side) -> out [rows][out_dim] (host); norm_merged:
// the LayerNorm over the merged token (DeepStack), else over each patch (the final merger)
void QiVision::merger(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged,
                      std::vector<float>& out) {
    const int64_t mm = merge_ * merge_;
    if (norm_merged) qi_layer_norm16(x, m.norm.w, m.norm.b, rows, in_dim * mm, eps_, h16_, s_);
    else qi_layer_norm16(x, m.norm.w, m.norm.b, rows * mm, in_dim, eps_, h16_, s_);
    product(h16_, m.fc1, big_, rows);
    qi_gelu_erf16(big_, rows * m.fc1.n, h16_, s_);
    product(h16_, m.fc2, y_, rows);
    out.resize((size_t) (rows * m.fc2.n));
    gpu::stream_sync(s_);
    if (!gpu::copy(out.data(), y_, out.size() * 4))
        throw std::runtime_error(std::string("image engine (vision): copy: ") + gpu::last_error());
}

QiVisionOut QiVision::encode(const float* rgb, int64_t width, int64_t height) {
    const int64_t unit = patch_ * merge_;
    if (width % unit != 0 || height % unit != 0 || width <= 0 || height <= 0)
        throw std::invalid_argument("image engine (vision): the sides must be multiples of " + std::to_string(unit));
    const int64_t ph = height / patch_, pw = width / patch_, T = ph * pw, pk = 3 * patch_ * patch_;
    grow(T);
    // the patches, their positions and their position embeddings, in the merge's order
    std::vector<uint16_t> patches((size_t) (T * pk));
    std::vector<int32_t> yx((size_t) (2 * T));
    std::vector<float> pos((size_t) (T * C_));
    auto coord = [&](int64_t i, int64_t n, int64_t& lo, int64_t& hi, float& f) {   // bilinear, corners aligned
        const double g = n > 1 ? (double) i * (double) (grid_ - 1) / (double) (n - 1) : 0.0;
        lo = (int64_t) std::floor(g);
        hi = std::min(lo + 1, grid_ - 1);
        f = (float) (g - (double) lo);
    };
    int64_t t = 0;
    for (int64_t y2 = 0; y2 < ph / merge_; ++y2)
        for (int64_t x2 = 0; x2 < pw / merge_; ++x2)
            for (int64_t dy = 0; dy < merge_; ++dy)
                for (int64_t dx = 0; dx < merge_; ++dx, ++t) {
                    const int64_t py = y2 * merge_ + dy, px = x2 * merge_ + dx;
                    uint16_t* row = patches.data() + t * pk;
                    for (int64_t c = 0; c < 3; ++c)
                        for (int64_t ky = 0; ky < patch_; ++ky)
                            for (int64_t kx = 0; kx < patch_; ++kx)
                                row[(c * patch_ + ky) * patch_ + kx] = kernels::f16_from_f32(
                                    rgb[(c * height + py * patch_ + ky) * width + px * patch_ + kx]);
                    yx[(size_t) (2 * t)] = (int32_t) py;
                    yx[(size_t) (2 * t + 1)] = (int32_t) px;
                    int64_t y0, y1, x0, x1;
                    float fy, fx;
                    coord(py, ph, y0, y1, fy);
                    coord(px, pw, x0, x1, fx);
                    const float* a = pos_.data() + (y0 * grid_ + x0) * C_;
                    const float* b = pos_.data() + (y0 * grid_ + x1) * C_;
                    const float* c0 = pos_.data() + (y1 * grid_ + x0) * C_;
                    const float* d = pos_.data() + (y1 * grid_ + x1) * C_;
                    float* o = pos.data() + t * C_;
                    for (int64_t k = 0; k < C_; ++k)
                        o[k] = (1 - fy) * ((1 - fx) * a[k] + fx * b[k]) + fy * ((1 - fx) * c0[k] + fx * d[k]);
                }
    put(h16_, patches.data(), patches.size() * 2);
    put(yx_, yx.data(), yx.size() * 4);
    put(y_, pos.data(), pos.size() * 4);
    gemm_.f16(h16_, patch_w_, x_, T, C_, pk);
    qi_bias_add(x_, patch_b_, T, C_, s_);
    qi_axpy(x_, y_, 1.f, T * C_, s_);
    QiVisionOut out;
    out.dim = out_dim_;
    out.rows = T / (merge_ * merge_);
    out.grid_h = ph / merge_;
    out.grid_w = pw / merge_;
    out.deepstack.resize(deepstack_.size());
    const float q_scale = std::sqrt(128.f / (float) hd_);   // the attention divides by sqrt(128), the head is hd_
    for (const Layer& l : layers_) {
        qi_layer_norm16(x_, l.ln1.w, l.ln1.b, T, C_, eps_, h16_, s_);
        product(h16_, l.qkv, y_, T);
        qi_vit_qkv_rope16(y_, yx_, T, heads_, hd_, 10000.f, q_scale, q16_, k16_, v16_, s_);
        qi_attention(q16_, k16_, v16_, T, heads_, 128, 0, a16_, s_, nullptr, 0, true);
        qi_unpad_heads16(a16_, T, heads_, hd_, o16_, s_);
        product(o16_, l.o, y_, T);
        qi_axpy(x_, y_, 1.f, T * C_, s_);
        qi_layer_norm16(x_, l.ln2.w, l.ln2.b, T, C_, eps_, h16_, s_);
        product(h16_, l.up, y_, T);
        qi_gelu16(y_, T * ffn_, h16_, s_);
        product(h16_, l.down, y_, T);
        qi_axpy(x_, y_, 1.f, T * C_, s_);
        if (l.deepstack >= 0)
            merger(deepstack_[(size_t) l.deepstack], x_, out.rows, C_, true, out.deepstack[(size_t) l.deepstack]);
    }
    merger(final_, x_, out.rows, C_, false, out.main);
    return out;
}

}  // namespace strata::image
