// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_vision.cpp - see include/strata/image/hunyuanimage3_vision.hpp.
#include "strata/image/hunyuanimage3_vision.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/hunyuanimage3_refs.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

using strata::kernels::f16_from_f32;

// qkv [P][3 C] (FP32: q, k, v side by side; the three biases are read here, qkv is read nowhere else) -> the
// shared attention's rows q16, k16, v16 [P][H][128] in FP16, the head's D channels and the rest zero.  q carries
// sqrt(128 / D): qi_attention divides by sqrt(128) over the 128-wide head, and the model wants 1 / sqrt(D) (the
// same trick qi_vision's reader uses).  k's and v's rows past n are zero: the padded patches attend only to the
// real keys (the pixel mask), and the keys past `n` are read in whole chunks of 64.
void heads_to_shared(const float* qkv, const float* bq, const float* bk, const float* bv, int64_t P, int64_t n,
                     int64_t C, int64_t H, int64_t D, float q_scale, uint16_t* q16, uint16_t* k16, uint16_t* v16,
                     void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (H * P * 128)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t d = i % 128, h = (i / 128) % H, p = i / (128 * H);
        q16[i] = d < D ? f16_from_f32((qkv[p * 3 * C + h * D + d] + bq[h * D + d]) * q_scale) : (uint16_t) 0;
    });
    q_of(s).parallel_for(sycl::range<1>((size_t) (2 * H * P * 128)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const bool side = i >= H * P * 128;   // the values, after the keys
        const auto j = side ? i - H * P * 128 : i;
        const int64_t d = j % 128, h = (j / 128) % H, p = j / (128 * H);
        const float* row = qkv + p * 3 * C + (side ? 2 * C : C);
        const float* b = side ? bv : bk;
        (side ? v16 : k16)[j] = p < n && d < D ? f16_from_f32(row[h * D + d] + b[h * D + d]) : (uint16_t) 0;
    });
}

}  // namespace

HunyuanImage3Vision::HunyuanImage3Vision() = default;

HunyuanImage3Vision::~HunyuanImage3Vision() {
    if (s_ != nullptr) gpu::stream_sync(s_);
    for (void* p : scratch_) gpu::free(p);
    for (void* p : weights_) gpu::free(p);
}

const float* HunyuanImage3Vision::vec(SafeTensors& st, const std::string& name, std::string& err) {
    const SafeTensor* t = st.find(name);
    if (t == nullptr) {
        err = "HunyuanImage 3.0 (SigLIP2): no tensor " + name;
        return nullptr;
    }
    weight_bytes_ += (uint64_t) t->elements() * 4;
    if (dry_) return nullptr;
    const auto v = st.f32(name);
    float* d = alloc<float>(weights_, (int64_t) v.size(), "image engine (HunyuanImage 3.0 SigLIP2)");
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

HunyuanImage3Vision::Norm HunyuanImage3Vision::norm(SafeTensors& st, const std::string& name, std::string& err) {
    Norm n;
    n.w = vec(st, name + ".weight", err);
    if (err.empty()) n.b = vec(st, name + ".bias", err);
    return n;
}

// a Linear's weight [out][in] (BF16 in the file) as the product's FP16 W[N][K], and its bias FP32
HunyuanImage3Vision::Lin HunyuanImage3Vision::lin(SafeTensors& st, const std::string& name, std::string& err) {
    Lin l;
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || t->shape.size() != 2) {
        err = "HunyuanImage 3.0 (SigLIP2): no matrix " + name;
        return l;
    }
    l.out = t->shape[0];
    l.in = t->shape[1];
    weight_bytes_ += (uint64_t) l.out * (uint64_t) l.in * 2;
    if (!dry_) {
        const auto w = st.f32(name + ".weight");
        std::vector<uint16_t> h(w.size());
        for (size_t i = 0; i < w.size(); ++i) h[i] = f16_from_f32(w[i]);
        uint16_t* d = alloc<uint16_t>(weights_, (int64_t) h.size(), "image engine (HunyuanImage 3.0 SigLIP2)");
        gpu::copy(d, h.data(), h.size() * 2);
        l.w = d;
    }
    l.b = vec(st, name + ".bias", err);
    return l;
}

bool HunyuanImage3Vision::open(const std::string& vision, const std::string& dit, void* stream, std::string& err,
                               bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        SafeTensors vs(vision);
        const std::string pre = "vision_model.";
        patch_ = lin(vs, pre + "embeddings.patch_embedding", err);
        if (!err.empty()) return false;
        hidden_ = patch_.out;
        if (patch_.in != kPatch * kPatch * 3) {
            err = "HunyuanImage 3.0 (SigLIP2): the patch embedding reads " + std::to_string(patch_.in) +
                  " values, not a " + std::to_string(kPatch) + " x " + std::to_string(kPatch) + " RGB patch";
            return false;
        }
        if (hidden_ % kHeads != 0 || hidden_ / kHeads > kHeadPad) {
            err = "HunyuanImage 3.0 (SigLIP2): a hidden width of " + std::to_string(hidden_) + " is not " +
                  std::to_string(kHeads) + " heads of at most " + std::to_string(kHeadPad);
            return false;
        }
        const SafeTensor* pe = vs.find(pre + "embeddings.position_embedding.weight");
        if (pe == nullptr || pe->shape.size() != 2 || pe->shape[1] != hidden_) {
            err = "HunyuanImage 3.0 (SigLIP2): no positional grid";
            return false;
        }
        pos_side_ = (int64_t) std::llround(std::sqrt((double) pe->shape[0]));
        if (pos_side_ * pos_side_ != pe->shape[0]) {
            err = "HunyuanImage 3.0 (SigLIP2): the positional grid is not square";
            return false;
        }
        weight_bytes_ += (uint64_t) pe->elements() * 4;   // the grid stays on the host; the resized copy is a buffer
        if (!dry_) pos_ = vs.f32(pre + "embeddings.position_embedding.weight");
        n_layers_ = 0;
        while (vs.find(pre + "encoder.layers." + std::to_string(n_layers_) + ".layer_norm1.weight") != nullptr)
            ++n_layers_;
        if (n_layers_ == 0) {
            err = "HunyuanImage 3.0 (SigLIP2): no encoder layers";
            return false;
        }
        layers_.resize((size_t) n_layers_);
        for (int64_t i = 0; i < n_layers_; ++i) {
            const std::string p = pre + "encoder.layers." + std::to_string(i) + ".";
            Layer& l = layers_[(size_t) i];
            l.ln1 = norm(vs, p + "layer_norm1", err);
            l.ln2 = norm(vs, p + "layer_norm2", err);
            l.q = lin(vs, p + "self_attn.q_proj", err);
            l.k = lin(vs, p + "self_attn.k_proj", err);
            l.v = lin(vs, p + "self_attn.v_proj", err);
            l.o = lin(vs, p + "self_attn.out_proj", err);
            l.fc1 = lin(vs, p + "mlp.fc1", err);
            l.fc2 = lin(vs, p + "mlp.fc2", err);
            if (!err.empty()) return false;
            if (l.q.out != hidden_ || l.fc2.out != hidden_ || l.fc1.out != l.fc2.in) {
                err = "HunyuanImage 3.0 (SigLIP2): layer " + std::to_string(i) + "'s shapes are not the model's";
                return false;
            }
            inter_ = l.fc1.out;
        }
        post_ = norm(vs, pre + "post_layernorm", err);
        if (!err.empty()) return false;
        SafeTensors ds(dit);
        al0_ = lin(ds, "vision_aligner.layers.0", err);
        if (!err.empty()) return false;
        al2_ = lin(ds, "vision_aligner.layers.2", err);
        if (!err.empty()) return false;
        if (al0_.in != hidden_ || al2_.in != al0_.out) {
            err = "HunyuanImage 3.0 (SigLIP2): the aligner does not read the vision model's width";
            return false;
        }
        out_ = al2_.out;
        if (dry_) return true;
        return gemm_.init(s_, 16, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

uint64_t HunyuanImage3Vision::scratch_bytes() const {
    const int64_t P = kMaxPatches, C = hidden_, Dp = kHeadPad, H = kHeads;
    uint64_t b = 0;
    b += (uint64_t) P * patch_.in * 2;                      // patches16
    b += (uint64_t) P * C * 2 * 1;                          // a16
    b += (uint64_t) 4 * P * H * Dp * 2;                     // q16, k16, v16, out16 (the shared attention's rows)
    b += (uint64_t) P * inter_ * (4 + 2);                   // mlp32, mlp16
    b += (uint64_t) P * al0_.out * (4 + 2);                 // al32, al16
    b += (uint64_t) P * out_ * 4;                           // out32
    b += (uint64_t) P * C * 4 * 4;                          // x32, tmp32, pos32 and qkv32's extra
    b += (uint64_t) P * 3 * C * 4;                          // qkv32
    return b;
}

void HunyuanImage3Vision::grow() {
    if (x32_ != nullptr) return;
    const int64_t P = kMaxPatches, C = hidden_, Dp = kHeadPad, H = kHeads;
    patches16_ = alloc<uint16_t>(scratch_, P * patch_.in, "image engine (HunyuanImage 3.0 SigLIP2)");
    a16_ = alloc<uint16_t>(scratch_, P * C, "image engine (HunyuanImage 3.0 SigLIP2)");
    q16_ = alloc<uint16_t>(scratch_, P * H * Dp, "image engine (HunyuanImage 3.0 SigLIP2)");
    k16_ = alloc<uint16_t>(scratch_, P * H * Dp, "image engine (HunyuanImage 3.0 SigLIP2)");
    v16_ = alloc<uint16_t>(scratch_, P * H * Dp, "image engine (HunyuanImage 3.0 SigLIP2)");
    out16_ = alloc<uint16_t>(scratch_, P * H * Dp, "image engine (HunyuanImage 3.0 SigLIP2)");
    mlp16_ = alloc<uint16_t>(scratch_, P * inter_, "image engine (HunyuanImage 3.0 SigLIP2)");
    al16_ = alloc<uint16_t>(scratch_, P * al0_.out, "image engine (HunyuanImage 3.0 SigLIP2)");
    x32_ = alloc<float>(scratch_, P * C, "image engine (HunyuanImage 3.0 SigLIP2)");
    qkv32_ = alloc<float>(scratch_, P * 3 * C, "image engine (HunyuanImage 3.0 SigLIP2)");
    tmp32_ = alloc<float>(scratch_, P * C, "image engine (HunyuanImage 3.0 SigLIP2)");
    mlp32_ = alloc<float>(scratch_, P * inter_, "image engine (HunyuanImage 3.0 SigLIP2)");
    al32_ = alloc<float>(scratch_, P * al0_.out, "image engine (HunyuanImage 3.0 SigLIP2)");
    out32_ = alloc<float>(scratch_, P * out_, "image engine (HunyuanImage 3.0 SigLIP2)");
    pos32_ = alloc<float>(scratch_, P * C, "image engine (HunyuanImage 3.0 SigLIP2)");
    held_bytes_ = scratch_bytes();
}

void HunyuanImage3Vision::drop_scratch() {
    if (s_ != nullptr) gpu::stream_sync(s_);
    for (void* p : scratch_) gpu::free(p);
    scratch_.clear();
    patches16_ = a16_ = q16_ = k16_ = v16_ = out16_ = mlp16_ = al16_ = nullptr;
    x32_ = qkv32_ = tmp32_ = mlp32_ = al32_ = out32_ = pos32_ = nullptr;
    pos_ph_ = pos_pw_ = 0;
    held_bytes_ = 0;
}

// one layer's attention over the P rows through the shared attention (qi_attention_cross, the reader qi_vision
// takes): every row's query against the n real keys (the pixel mask), the heads padded to 128, the outputs
// unpadded into a16_ for the out projection
void HunyuanImage3Vision::attention(const Layer& l, int64_t n) {
    const int64_t P = kMaxPatches, C = hidden_, H = kHeads, D = C / H;
    gemm_.f16(a16_, l.q.w, qkv32_, P, C, C, 3 * C);
    gemm_.f16(a16_, l.k.w, qkv32_ + C, P, C, C, 3 * C);
    gemm_.f16(a16_, l.v.w, qkv32_ + 2 * C, P, C, C, 3 * C);
    const float q_scale = std::sqrt(128.f / (float) D);
    heads_to_shared(qkv32_, l.q.b, l.k.b, l.v.b, P, n, C, H, D, q_scale, q16_, k16_, v16_, s_);
    qi_attention_cross(q16_, k16_, v16_, P, n, H, kHeadPad, out16_, s_);
    qi_unpad_heads16(out16_, P, H, D, a16_, s_);
    gemm_.f16(a16_, l.o.w, tmp32_, P, C, C);
    hunyuanimage3_bias_add32(tmp32_, l.o.b, P, C, s_);
    hunyuanimage3_add(x32_, tmp32_, P * C, s_);
}

bool HunyuanImage3Vision::rows(const float* patches, int64_t ph, int64_t pw, float* out, std::string& err) {
    if (dry_) {
        err = "HunyuanImage 3.0 (SigLIP2): the part was opened for its sizes only";
        return false;
    }
    const int64_t P = kMaxPatches, C = hidden_, n = ph * pw;
    if (n <= 0 || n > P) {
        err = "HunyuanImage 3.0 (SigLIP2): a " + std::to_string(ph) + " x " + std::to_string(pw) + " grid";
        return false;
    }
    try {
        grow();
        // the patches, FP16
        std::vector<uint16_t> h16((size_t) (P * patch_.in));
        for (size_t i = 0; i < h16.size(); ++i) h16[i] = f16_from_f32(patches[i]);
        if (!gpu::copy(patches16_, h16.data(), h16.size() * 2)) throw std::runtime_error(gpu::last_error());
        // the positional grid at this grid's size: the reference's F.interpolate(bilinear, antialias) of the
        // [side][side] grid to [ph][pw], the padded rows the first entry
        if (ph != pos_ph_ || pw != pos_pw_) {
            std::vector<float> grid((size_t) (C * pos_side_ * pos_side_));   // [C][side][side]
            for (int64_t q = 0; q < pos_side_ * pos_side_; ++q)
                for (int64_t c = 0; c < C; ++c) grid[(size_t) (c * pos_side_ * pos_side_ + q)] = pos_[(size_t) (q * C + c)];
            const std::vector<float> rs = hunyuanimage3_resize(grid, C, pos_side_, pos_side_, pw, ph, false, false);
            std::vector<float> rows_pos((size_t) (P * C));
            for (int64_t p = 0; p < P; ++p) {
                const int64_t src = p < n ? p : 0;
                for (int64_t c = 0; c < C; ++c) rows_pos[(size_t) (p * C + c)] = rs[(size_t) (c * n + src)];
            }
            if (!gpu::copy(pos32_, rows_pos.data(), rows_pos.size() * 4)) throw std::runtime_error(gpu::last_error());
            pos_ph_ = ph;
            pos_pw_ = pw;
        }
        gemm_.f16(patches16_, patch_.w, x32_, P, C, patch_.in);
        hunyuanimage3_bias_add32(x32_, patch_.b, P, C, s_);
        hunyuanimage3_add(x32_, pos32_, P * C, s_);
        for (const Layer& l : layers_) {
            qi_layer_norm16(x32_, l.ln1.w, l.ln1.b, P, C, kEps, a16_, s_);
            attention(l, n);
            qi_layer_norm16(x32_, l.ln2.w, l.ln2.b, P, C, kEps, a16_, s_);
            gemm_.f16(a16_, l.fc1.w, mlp32_, P, inter_, C);
            hunyuanimage3_bias_add32(mlp32_, l.fc1.b, P, inter_, s_);
            qi_gelu16(mlp32_, P * inter_, mlp16_, s_);
            gemm_.f16(mlp16_, l.fc2.w, tmp32_, P, C, inter_);
            hunyuanimage3_bias_add32(tmp32_, l.fc2.b, P, C, s_);
            hunyuanimage3_add(x32_, tmp32_, P * C, s_);
        }
        qi_layer_norm16(x32_, post_.w, post_.b, P, C, kEps, a16_, s_);
        // the post-norm features, kept for the probe (FP16 as the aligner reads them)
        {
            std::vector<uint16_t> bits((size_t) (P * C));
            gpu::stream_sync(s_);
            gpu::copy(bits.data(), a16_, bits.size() * 2);
            last_hidden_.resize(bits.size());
            for (size_t i = 0; i < bits.size(); ++i) last_hidden_[i] = kernels::f32_from_f16(bits[i]);
        }
        gemm_.f16(a16_, al0_.w, al32_, P, al0_.out, C);
        hunyuanimage3_bias_add32(al32_, al0_.b, P, al0_.out, s_);
        qi_gelu_erf16(al32_, P * al0_.out, al16_, s_);
        gemm_.f16(al16_, al2_.w, out32_, P, out_, al2_.in);
        hunyuanimage3_bias_add32(out32_, al2_.b, P, out_, s_);
        gpu::stream_sync(s_);
        if (!gpu::copy(out, out32_, (size_t) (P * out_) * 4)) throw std::runtime_error(gpu::last_error());
        return true;
    } catch (const std::exception& e) {
        err = std::string("HunyuanImage 3.0 (SigLIP2): ") + e.what();
        return false;
    }
}

}  // namespace strata::image
