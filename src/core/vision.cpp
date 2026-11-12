// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/vision.cpp - see include/strata/core/vision.hpp.
//
// Qwen3-VL's ViT as stable-diffusion.cpp runs it for Qwen-Image-2.1's edits (the same ViT the chat models' mmproj
// files carry, without DeepStack there): 16 x 16 patches (the two temporal
// kernels of the conv3d summed: an image is its own second frame) in the order of the 2 x 2 merge (merged row, merged
// column, row in the block, column in it), the learned 48 x 48 position embeddings resized bilinearly (corners
// aligned), 2D rotary embeddings (half the pairs by row, half by column), blocks of LayerNorm, attention over every
// patch and a GELU (tanh) MLP; after the DeepStack layers a merger of 4 patches (LayerNorm over the 4, GELU (erf) MLP),
// and at the end the final LayerNorm per patch and the same merger.
//
// On the CPU (open_cpu) the same steps run in FP32 rows, every product's weight packed as int8 when the file is read
// and the products W8A8 (the weights int8, a scale per kCpuI8Block inputs of an output; the activations quantized a row
// at a time: i8_linear); with XESTRATA_VISION_CPU_I8=0 the weights are packed in BF16 instead and the products are
// bf16_linear (kernels/cpu/cpu_ops.hpp), the attention's two products the same way (kernels/cpu/vision_cpu.hpp), the
// closer form.

#include "strata/core/vision.hpp"
#include "strata/image/qi_gguf.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/image_attn.hpp"
#include "strata/image/gpu_buf.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::core {

namespace gpu = strata::gpu;
namespace image = strata::image;   // the rows and the attention it shares with the image engine

namespace {

constexpr int kTypeF16 = 1, kTypeF32 = 0, kTypeBF16 = 30, kTypeQ8_0 = 8;

// The inputs one int8 scale covers in the opt-in int8 products, the weights' and the activations' alike.
constexpr int kCpuI8Block = 32;

// A weight's element i as FP32.  The chat models' mmproj files hold BF16 and an F32 patch embedding; Qwen-Image's hold
// F16 (2026-11-12, the files XeStrata downloads); a user's own may be Q8_0 (llama-quantize's, which the documents
// describe: blocks of 32 int8 values after an FP16 scale).  The GPU's products take FP16 weights and the CPU's BF16,
// so every type is converted once, here.
float weight_at(const void* data, int type, size_t i) {
    switch (type) {
        case kTypeF16: return kernels::f32_from_f16(reinterpret_cast<const uint16_t*>(data)[i]);
        case kTypeF32: return reinterpret_cast<const float*>(data)[i];
        case kTypeBF16: return kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(data)[i]);
        case kTypeQ8_0: {
            const auto* block = reinterpret_cast<const uint8_t*>(data) + (i / 32) * 34;
            uint16_t d;
            std::memcpy(&d, block, 2);
            return kernels::f32_from_f16(d) * (float) reinterpret_cast<const int8_t*>(block + 2)[i % 32];
        }
        default: throw std::runtime_error("vision: a weight of type " + std::to_string(type));
    }
}

bool weight_type_ok(int type) {
    return type == kTypeF16 || type == kTypeF32 || type == kTypeBF16 || type == kTypeQ8_0;
}

void put(void* d, const void* h, size_t bytes) {
    if (!gpu::copy(d, h, bytes))
        throw std::runtime_error(std::string("vision: copy: ") + gpu::last_error());
}

}  // namespace

Vision::Vision() = default;

Vision::~Vision() {
    drop_scratch();
    for (void* p : allocs_) gpu::free(p);
}

Vision::W Vision::linear(const std::string& name) {
    const TensorInfo* t = image::find_checked(*g_, name + ".weight");
    if (t == nullptr || t->shape.size() != 2) throw std::runtime_error("vision: no " + name);
    W w;
    w.k = (int64_t) t->shape[0];
    w.n = (int64_t) t->shape[1];
    const TensorInfo* b = image::find_checked(*g_, name + ".bias");
    if (b != nullptr && ((int) b->type != kTypeF32 || (int64_t) b->elements() != w.n))   // #29: read as w.n floats
        throw std::runtime_error("vision: " + name + ".bias does not match its weight");
    const int type = (int) t->type;
    if (pool_) {
        if (!weight_type_ok(type))
            throw std::runtime_error("vision: " + name + ": type " + std::to_string(t->type));
        const void* data = g_->tensor_data(*t);
        const int64_t k = w.k;
        auto row = [&](int o, float* dst) {
            for (int64_t i = 0; i < k; ++i) dst[i] = weight_at(data, type, (size_t) (o * k + i));
            return true;
        };
        const bool ok = i8_ ? kernels::cpu::i8_pack_rows(*pool_, (int) w.k, (int) w.n, kCpuI8Block,
                                                         kernels::cpu::i8_gemm_weight_levels(isa_), row, w.q8)
                            : kernels::cpu::bf16_pack_rows(*pool_, (int) w.k, (int) w.n, row, w.b16);
        if (!ok) throw std::runtime_error("vision: " + name + ": cannot pack it");
        w.b = b != nullptr ? reinterpret_cast<const float*>(g_->tensor_data(*b)) : nullptr;
        weight_bytes_ += w.b16.bytes() + w.q8.bytes() + (b ? (uint64_t) w.n * 4 : 0);
        return w;
    }
    weight_bytes_ += (uint64_t) (w.n * w.k) * 2 + (b ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    if (!weight_type_ok(type))
        throw std::runtime_error("vision: " + name + ": type " + std::to_string(t->type));
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    if (type == kTypeF16) {
        std::memcpy(h.data(), g_->tensor_data(*t), h.size() * 2);
    } else {
        for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(weight_at(g_->tensor_data(*t), type, i));
    }
    uint16_t* d = image::alloc<uint16_t>(allocs_, (int64_t) h.size(), "vision");
    put(d, h.data(), h.size() * 2);
    w.w = d;
    if (b != nullptr) {
        float* db = image::alloc<float>(allocs_, w.n, "vision");
        put(db, g_->tensor_data(*b), (size_t) w.n * 4);
        w.b = db;
    }
    return w;
}

Vision::Norm Vision::norm(const std::string& name) {
    Norm n;
    for (const char* part : {".weight", ".bias"}) {
        const TensorInfo* t = image::find_checked(*g_, name + part);
        if (t == nullptr || (int) t->type != kTypeF32) throw std::runtime_error("vision: no " + name);
        const auto c = (int64_t) t->elements();
        weight_bytes_ += (uint64_t) c * 4;
        if (pool_) {
            (part[1] == 'w' ? n.w : n.b) = reinterpret_cast<const float*>(g_->tensor_data(*t));
            continue;
        }
        if (dry_) continue;
        float* d = image::alloc<float>(allocs_, c, "vision");
        put(d, g_->tensor_data(*t), (size_t) c * 4);
        (part[1] == 'w' ? n.w : n.b) = d;
    }
    return n;
}

bool Vision::open(const std::string& path, void* stream, std::string& err, bool dry) {
    s_ = stream;
    dry_ = dry;
    if (!read(path, err)) return false;
    if (dry_) return true;
    return gemm_.init(s_, 16, err);
}

bool Vision::open_cpu(const std::string& path, int threads, std::string& err) {
    if (!kernels::cpu::cpu_avx2_ok()) {
        err = "the CPU vision encoder needs AVX2";
        return false;
    }
    pool_ = std::make_unique<kernels::cpu::ExpertPool>(std::max(1, threads), true, true, 0, 0, true);
    const std::vector<int> cores = kernels::cpu::physical_cores(false);
    if (pool_->workers() > 0 && !cores.empty()) host_cpu_ = cores[0];
    const char* e = strata::env_get("VISION_CPU_I8");
    i8_ = e == nullptr || std::strcmp(e, "0") != 0;   // the default: the int8 products; =0 the BF16 ones
    isa_ = kernels::cpu::i8_gemm_isa();
    return read(path, err);
}

// The file's shape and weights, each placed where the encoder runs (open: the GPU, open_cpu: the host, int8)
bool Vision::read(const std::string& path, std::string& err) {
    try {
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
        if (C_ <= 0 || heads_ <= 0 || patch_ <= 0) throw std::runtime_error("vision: bad shape in " + path);
        hd_ = C_ / heads_;
        if (hd_ > 128 || hd_ % 4 != 0) throw std::runtime_error("vision: head size " + std::to_string(hd_));
        const auto layers = (int64_t) num("clip.vision.block_count", 27);
        // the patch embedding: the two temporal kernels summed (the image is its own second frame)
        const TensorInfo* p0 = image::find_checked(*g_, "v.patch_embd.weight");
        const TensorInfo* p1 = image::find_checked(*g_, "v.patch_embd.weight.1");
        const TensorInfo* pb = image::find_checked(*g_, "v.patch_embd.bias");
        const int ptype = p0 != nullptr ? (int) p0->type : -1;
        if (p0 == nullptr || pb == nullptr ||
            (ptype != kTypeF16 && ptype != kTypeF32 && ptype != kTypeBF16))
            throw std::runtime_error("vision: no patch embedding (F16, F32 or BF16)");
        const int64_t pk = 3 * patch_ * patch_;
        // #29: the reads below take their lengths from the header, so the tensors must be that long
        if ((int64_t) p0->elements() != C_ * pk || (p1 != nullptr && (p1->type != p0->type || p1->elements() != p0->elements()))
            || (int) pb->type != kTypeF32 || (int64_t) pb->elements() != C_)
            throw std::runtime_error("vision: the patch embedding does not match the header");
        auto patch_at = [&, ptype, a = g_->tensor_data(*p0), b = p1 != nullptr ? g_->tensor_data(*p1) : nullptr](size_t i) {
            return weight_at(a, ptype, i) + (b != nullptr ? weight_at(b, ptype, i) : 0.f);
        };
        if (pool_) {
            auto row = [&](int o, float* dst) {
                for (int64_t i = 0; i < pk; ++i) dst[i] = patch_at((size_t) (o * pk + i));
                return true;
            };
            if (!(i8_ ? kernels::cpu::i8_pack_rows(*pool_, (int) pk, (int) C_, kCpuI8Block,
                                                   kernels::cpu::i8_gemm_weight_levels(isa_), row, patch16_.q8)
                      : kernels::cpu::bf16_pack_rows(*pool_, (int) pk, (int) C_, row, patch16_.b16)))
                throw std::runtime_error("vision: cannot pack the patch embedding");
            patch16_.n = C_;
            patch16_.k = pk;
            patch16_.b = reinterpret_cast<const float*>(g_->tensor_data(*pb));
            weight_bytes_ += patch16_.b16.bytes() + patch16_.q8.bytes() + (uint64_t) C_ * 4;
        } else {
            weight_bytes_ += (uint64_t) (C_ * pk) * 2 + (uint64_t) C_ * 4;
        }
        if (!dry_ && !pool_) {
            std::vector<uint16_t> h((size_t) (C_ * pk));
            for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(patch_at(i));
            uint16_t* d = image::alloc<uint16_t>(allocs_, (int64_t) h.size(), "vision");
            put(d, h.data(), h.size() * 2);
            patch_w_ = d;
            float* db = image::alloc<float>(allocs_, C_, "vision");
            put(db, g_->tensor_data(*pb), (size_t) C_ * 4);
            patch_b_ = db;
        }
        const TensorInfo* pe = image::find_checked(*g_, "v.position_embd.weight");
        if (pe == nullptr || (int) pe->type != kTypeF32 || pe->shape.size() < 2)
            throw std::runtime_error("vision: no position embeddings");
        grid_ = (int64_t) std::lround(std::sqrt((double) pe->shape[1]));
        if (grid_ * grid_ * C_ > (int64_t) pe->elements())
            throw std::runtime_error("vision: the position embeddings are shorter than the header says");
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
            if (image::find_checked(*g_, ds + "fc1.weight") != nullptr) {
                l.deepstack = (int) deepstack_.size();
                deepstack_.push_back({norm(ds + "norm"), linear(ds + "fc1"), linear(ds + "fc2")});
            }
            layers_.push_back(l);
        }
        final_ = {norm("v.post_ln"), linear("mm.0"), linear("mm.2")};
        if (final_.fc2.n != out_dim_) throw std::runtime_error("vision: the merger's output size");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

uint64_t Vision::scratch_bytes(int64_t patches) const {
    const int64_t T = patches, TP = image::qi_attn_rows(T), wide = std::max(3 * C_, ffn_);
    return (uint64_t) T * ((uint64_t) C_ * 4 + (uint64_t) wide * 4 + (uint64_t) heads_ * 128 * 4 +
                           (uint64_t) std::max(C_, ffn_) * 2 + (uint64_t) heads_ * 128 * 2 + (uint64_t) C_ * 2 + 8) +
           (uint64_t) TP * (uint64_t) heads_ * 128 * 2 * 3;
}

void Vision::drop_scratch() {
    for (void* p : work_) gpu::free(p);
    work_.clear();
    cap_ = 0;
}

void Vision::grow(int64_t patches) {
    if (patches <= cap_) return;
    drop_scratch();
    const int64_t T = patches, TP = image::qi_attn_rows(T);
    x_ = image::alloc<float>(work_, T * C_, "vision");
    y_ = image::alloc<float>(work_, T * std::max(3 * C_, ffn_), "vision");
    big_ = image::alloc<float>(work_, T * heads_ * 128, "vision");
    h16_ = image::alloc<uint16_t>(work_, T * std::max({C_, ffn_, 3 * patch_ * patch_}), "vision");
    q16_ = image::alloc<uint16_t>(work_, TP * heads_ * 128, "vision", true);   // the rows past T stay zero
    k16_ = image::alloc<uint16_t>(work_, TP * heads_ * 128, "vision", true);
    v16_ = image::alloc<uint16_t>(work_, TP * heads_ * 128, "vision", true);
    a16_ = image::alloc<uint16_t>(work_, T * heads_ * 128, "vision");
    o16_ = image::alloc<uint16_t>(work_, T * C_, "vision");
    yx_ = image::alloc<int32_t>(work_, 2 * T, "vision");
    cap_ = patches;
}

void Vision::product(const uint16_t* x16, const W& w, float* y, int64_t T) {
    gemm_.f16(x16, w.w, y, T, w.n, w.k);
    if (w.b != nullptr) image::qi_bias_add(y, w.b, T, w.n, s_);
}

// rows tokens of in_dim (x, FP32, the merged tokens' patches side by side) -> out [rows][out_dim] (host); norm_merged:
// the LayerNorm over the merged token (DeepStack), else over each patch (the final merger)
void Vision::merger(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged,
                      std::vector<float>& out) {
    const int64_t mm = merge_ * merge_;
    if (norm_merged) image::qi_layer_norm16(x, m.norm.w, m.norm.b, rows, in_dim * mm, eps_, h16_, s_);
    else image::qi_layer_norm16(x, m.norm.w, m.norm.b, rows * mm, in_dim, eps_, h16_, s_);
    product(h16_, m.fc1, big_, rows);
    image::qi_gelu_erf16(big_, rows * m.fc1.n, h16_, s_);
    product(h16_, m.fc2, y_, rows);
    out.resize((size_t) (rows * m.fc2.n));
    gpu::stream_sync(s_);
    if (!gpu::copy(out.data(), y_, out.size() * 4))
        throw std::runtime_error(std::string("vision: copy: ") + gpu::last_error());
}

// The patches [T][3 * patch * patch], their rows and columns [T][2] and their position embeddings [T][C], in the
// merge's order: what the GPU and the CPU start from
void Vision::layout(const float* rgb, int64_t width, int64_t height, std::vector<float>& patches,
                    std::vector<int32_t>& yx, std::vector<float>& pos) const {
    const int64_t unit = patch_ * merge_;
    if (width % unit != 0 || height % unit != 0 || width <= 0 || height <= 0)
        throw std::invalid_argument("vision: the sides must be multiples of " + std::to_string(unit));
    const int64_t ph = height / patch_, pw = width / patch_, T = ph * pw, pk = 3 * patch_ * patch_;
    patches.resize((size_t) (T * pk));
    yx.resize((size_t) (2 * T));
    pos.resize((size_t) (T * C_));
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
                    float* row = patches.data() + t * pk;
                    for (int64_t c = 0; c < 3; ++c)
                        for (int64_t ky = 0; ky < patch_; ++ky)
                            for (int64_t kx = 0; kx < patch_; ++kx)
                                row[(c * patch_ + ky) * patch_ + kx] =
                                    rgb[(c * height + py * patch_ + ky) * width + px * patch_ + kx];
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
}

VisionOut Vision::encode(const float* rgb, int64_t width, int64_t height) {
    if (pool_) return encode_cpu(rgb, width, height);
    std::vector<float> pf;
    std::vector<int32_t> yx;
    std::vector<float> pos;
    layout(rgb, width, height, pf, yx, pos);
    const int64_t ph = height / patch_, pw = width / patch_, T = ph * pw, pk = 3 * patch_ * patch_;
    grow(T);
    std::vector<uint16_t> patches(pf.size());
    for (size_t i = 0; i < pf.size(); ++i) patches[i] = kernels::f16_from_f32(pf[i]);
    put(h16_, patches.data(), patches.size() * 2);
    put(yx_, yx.data(), yx.size() * 4);
    put(y_, pos.data(), pos.size() * 4);
    gemm_.f16(h16_, patch_w_, x_, T, C_, pk);
    image::qi_bias_add(x_, patch_b_, T, C_, s_);
    image::qi_axpy(x_, y_, 1.f, T * C_, s_);
    VisionOut out;
    out.dim = out_dim_;
    out.rows = T / (merge_ * merge_);
    out.grid_h = ph / merge_;
    out.grid_w = pw / merge_;
    out.deepstack.resize(deepstack_.size());
    const float q_scale = std::sqrt(128.f / (float) hd_);   // the attention divides by sqrt(128), the head is hd_
    for (const Layer& l : layers_) {
        image::qi_layer_norm16(x_, l.ln1.w, l.ln1.b, T, C_, eps_, h16_, s_);
        product(h16_, l.qkv, y_, T);
        image::qi_vit_qkv_rope16(y_, yx_, T, heads_, hd_, 10000.f, q_scale, q16_, k16_, v16_, s_);
        image::qi_attention(q16_, k16_, v16_, T, heads_, 128, 0, a16_, s_, nullptr, 0, true);
        image::qi_unpad_heads16(a16_, T, heads_, hd_, o16_, s_);
        product(o16_, l.o, y_, T);
        image::qi_axpy(x_, y_, 1.f, T * C_, s_);
        image::qi_layer_norm16(x_, l.ln2.w, l.ln2.b, T, C_, eps_, h16_, s_);
        product(h16_, l.up, y_, T);
        image::qi_gelu16(y_, T * ffn_, h16_, s_);
        product(h16_, l.down, y_, T);
        image::qi_axpy(x_, y_, 1.f, T * C_, s_);
        if (l.deepstack >= 0)
            merger(deepstack_[(size_t) l.deepstack], x_, out.rows, C_, true, out.deepstack[(size_t) l.deepstack]);
    }
    merger(final_, x_, out.rows, C_, false, out.main);
    return out;
}

// y [T][w.n] = x [T][w.k] . w^T + b on the CPU, the product on the pool (the int8 one: the rows quantized first)
void Vision::product_cpu(const float* x, const W& w, float* y, int64_t T) {
    if (!i8_) {
        kernels::cpu::bf16_linear(*pool_, w.b16, x, (size_t) w.k, (int) T, y, (size_t) w.n, w.b);
        return;
    }
    kernels::cpu::i8_quantize_rows(*pool_, isa_, x, (size_t) w.k, (int) T, (int) w.k, kCpuI8Block, rows8_);
    kernels::cpu::i8_linear(*pool_, isa_, w.q8, rows8_, y, (size_t) w.n, w.b);
}

void Vision::merger_cpu(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged,
                        std::vector<float>& out) {
    namespace kc = kernels::cpu;
    const int64_t mm = merge_ * merge_;
    if (norm_merged) kc::vision_layer_norm(*pool_, x, rows, in_dim * mm, m.norm.w, m.norm.b, eps_, ch_.data());
    else kc::vision_layer_norm(*pool_, x, rows * mm, in_dim, m.norm.w, m.norm.b, eps_, ch_.data());
    cbig_.resize((size_t) (rows * std::max(m.fc1.n, ffn_)));
    product_cpu(ch_.data(), m.fc1, cbig_.data(), rows);
    kc::vision_gelu_erf(*pool_, cbig_.data(), rows * m.fc1.n);
    out.resize((size_t) (rows * m.fc2.n));
    product_cpu(cbig_.data(), m.fc2, out.data(), rows);
}

VisionOut Vision::encode_cpu(const float* rgb, int64_t width, int64_t height) {
    namespace kc = kernels::cpu;
    const kc::ScopedThreadPin pin(host_cpu_);
    std::vector<float> patches, pos;
    std::vector<int32_t> yx;
    layout(rgb, width, height, patches, yx, pos);
    const int64_t ph = height / patch_, pw = width / patch_, T = ph * pw;
    cx_.resize((size_t) (T * C_));
    ch_.resize((size_t) (T * std::max(C_, ffn_)));
    cy_.resize((size_t) (T * std::max(3 * C_, ffn_)));
    cbig_.resize((size_t) (T * ffn_));
    product_cpu(patches.data(), patch16_, cx_.data(), T);
    kc::vision_add(*pool_, cx_.data(), pos.data(), T * C_);
    std::vector<float> cos, sin;
    kc::vision_rope_table(yx.data(), T, hd_, 10000.f, cos, sin);
    VisionOut out;
    out.dim = out_dim_;
    out.rows = T / (merge_ * merge_);
    out.grid_h = ph / merge_;
    out.grid_w = pw / merge_;
    out.deepstack.resize(deepstack_.size());
    for (const Layer& l : layers_) {
        kc::vision_layer_norm(*pool_, cx_.data(), T, C_, l.ln1.w, l.ln1.b, eps_, ch_.data());
        product_cpu(ch_.data(), l.qkv, cy_.data(), T);
        attn_.run(*pool_, cy_.data(), T, heads_, hd_, cos, sin, ch_.data(), i8_ ? &isa_ : nullptr);
        product_cpu(ch_.data(), l.o, cy_.data(), T);
        kc::vision_add(*pool_, cx_.data(), cy_.data(), T * C_);
        kc::vision_layer_norm(*pool_, cx_.data(), T, C_, l.ln2.w, l.ln2.b, eps_, ch_.data());
        product_cpu(ch_.data(), l.up, cbig_.data(), T);
        kc::vision_gelu_tanh(*pool_, cbig_.data(), T * ffn_);
        product_cpu(cbig_.data(), l.down, cy_.data(), T);
        kc::vision_add(*pool_, cx_.data(), cy_.data(), T * C_);
        if (l.deepstack >= 0)
            merger_cpu(deepstack_[(size_t) l.deepstack], cx_.data(), out.rows, C_, true,
                       out.deepstack[(size_t) l.deepstack]);
    }
    merger_cpu(final_, cx_.data(), out.rows, C_, false, out.main);
    return out;
}

}  // namespace strata::core
