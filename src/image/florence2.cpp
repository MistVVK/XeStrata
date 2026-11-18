// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/image/florence2.cpp - see include/strata/image/florence2.hpp.  The beam search (decode) follows transformers'
// generation (src/transformers/generation/utils.py, _beam_search; Apache-2.0).

#include "strata/image/florence2.hpp"

#include "qi_conv.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/aux_attn.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr float kEps = 1e-5f;               // the LayerNorms' (torch's default; BART's layer_norm_eps)
constexpr int32_t kBos = 0, kEos = 2;   // <s>, </s>; the decoder starts from </s>
constexpr int64_t kPosOffset = 2;           // BART's learned positions start at 2
constexpr int kNgram = 3;                   // the generation config's no_repeat_ngram_size
constexpr float kNeg = -1.0e9f;             // transformers' beam search's "never" score

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<uint16_t> f16_of(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = kernels::f16_from_f32(v[i]);
    return h;
}

}  // namespace

Florence2::Florence2(const std::string& path) : AuxNet(path, "Florence-2") {
    // the shapes the code takes (Florence-2-base and -large differ in their widths and depths, read from the file)
    const SafeTensor* sh = st_->find("language_model.model.shared.weight");
    const SafeTensor* ip = st_->find("image_projection");
    if (sh == nullptr || sh->shape.size() != 2 || ip == nullptr || ip->shape.size() != 2)
        throw std::runtime_error(path + ": not Florence-2's weights (no shared embeddings or image projection)");
    vocab_ = sh->shape[0];
    d_ = sh->shape[1];
    image_dim_ = ip->shape[0];
    if (ip->shape[1] != d_) throw std::runtime_error(path + ": the image projection's width is not the model's");
    const SafeTensor* fc1 = st_->find("language_model.model.encoder.layers.0.fc1.weight");
    if (fc1 == nullptr) throw std::runtime_error(path + ": no encoder layers");
    ffn_ = fc1->shape[0];
    heads_ = d_ / 64;   // BART's heads: 64 channels each in Florence-2-base (12) and -large (16)
    const SafeTensor* ep = st_->find("language_model.model.encoder.embed_positions.weight");
    if (ep == nullptr) throw std::runtime_error(path + ": no positions");
    max_pos_ = ep->shape[0] - kPosOffset;
}

uint64_t Florence2::work_bytes(int beams, int max_new) const {
    // the first stage's buffers (its tokens 4 wide in the MLP), and the decoder's caches
    const int64_t n0 = (image_size_ / 4) * (image_size_ / 4), c0 = 128;
    const uint64_t vision = (uint64_t) n0 * c0 * (4 + 4 + 16 + 8 + 2 * 6) + (64ull << 20);
    size_t layers = dec_.size();
    if (!loaded_)
        while (st_->find("language_model.model.decoder.layers." + std::to_string(layers) + ".fc1.weight") != nullptr)
            ++layers;
    const uint64_t cache = (uint64_t) layers * 4 * beams * (max_new + 1) * d_ * 2;
    return vision + cache + (uint64_t) beams * vocab_ * 4;
}

uint64_t Florence2::need_bytes(int beams, int max_new) const {
    const uint64_t want = work_bytes(beams, max_new), have = held() - weight_bytes_;
    // the FP32 norms and biases and the products' own scratch: a tenth over the file's FP16 tensors
    return (loaded_ ? 0 : file_bytes_ + file_bytes_ / 10) + (want > have ? want - have : 0);
}

Florence2::Dw Florence2::dw(const std::string& name) {
    Dw d;
    const auto w = st_->f32(name + ".weight"), b = st_->f32(name + ".bias");
    if (w.size() != b.size() * 9) throw std::runtime_error(path_ + ": " + name + " is not a depthwise 3 x 3");
    auto* dwp = static_cast<float*>(alloc(w.size() * 4));
    gpu::copy(dwp, w.data(), w.size() * 4);
    auto* dbp = static_cast<float*>(alloc(b.size() * 4));
    gpu::copy(dbp, b.data(), b.size() * 4);
    d.w = dwp;
    d.b = dbp;
    weight_bytes_ += (w.size() + b.size()) * 4;
    return d;
}

void Florence2::load_weights() {
    stages_.clear();
    enc_.clear();
    dec_.clear();
    feats_ = nullptr;
    {
        // DaViT: the stages' widths from their embeddings, the blocks counted in the file
        static const int kStride[] = {4, 2, 2, 2}, kPad[] = {3, 1, 1, 1};
        for (int s = 0; s < 4; ++s) {
            const std::string p = "vision_tower.convs." + std::to_string(s);
            Stage st;
            st.embed = conv(p + ".proj", kStride[s], kPad[s]);
            st.embed_norm = norm(p + ".norm");
            st.pre_norm = s > 0;
            st.dim = st.embed.o;
            st.heads = st.dim / 32;    // DaViT's heads and channel groups: 32 channels each
            st.groups = st.dim / 32;
            for (int j = 0;; ++j) {
                const std::string b = "vision_tower.blocks." + std::to_string(s) + "." + std::to_string(j) + ".";
                if (st_->find(b + "spatial_block.conv1.fn.dw.weight") == nullptr) break;
                for (int which = 0; which < 2; ++which) {
                    const std::string q = b + (which == 0 ? "spatial_block." : "channel_block.");
                    const std::string a = q + (which == 0 ? "window_attn" : "channel_attn");
                    VBlock vb;
                    vb.conv1 = dw(q + "conv1.fn.dw");
                    vb.conv2 = dw(q + "conv2.fn.dw");
                    vb.n1 = norm(a + ".norm");
                    vb.qkv = lin(a + ".fn.qkv");
                    const int64_t third = vb.qkv.n / 3;
                    Lin* views[3] = {&vb.q, &vb.k, &vb.v};
                    for (int64_t i = 0; i < 3; ++i)
                        *views[i] = Lin{vb.qkv.w + i * third * vb.qkv.k,
                                        vb.qkv.b != nullptr ? vb.qkv.b + i * third : nullptr, third, vb.qkv.k};
                    vb.proj = lin(a + ".fn.proj");
                    vb.n2 = norm(q + "ffn.norm");
                    vb.fc1 = lin(q + "ffn.fn.net.fc1");
                    vb.fc2 = lin(q + "ffn.fn.net.fc2");
                    (which == 0 ? st.spatial : st.channel).push_back(vb);
                }
            }
            if (st.spatial.empty()) throw std::runtime_error(path_ + ": stage " + std::to_string(s) + " has no blocks");
            stages_.push_back(std::move(st));
        }
        if (stages_.back().dim != image_dim_) throw std::runtime_error(path_ + ": the last stage is not the projection's");
        // the 2D positions (column then row halves) and the first frame's cosine, on the host
        const auto row = st_->f32("image_pos_embed.row_embeddings.weight");
        const auto col = st_->f32("image_pos_embed.column_embeddings.weight");
        const auto tmp = st_->f32("visual_temporal_embed.pos_idx_to_embed");
        const int64_t g = image_size_ / 32, half = image_dim_ / 2;
        if ((int64_t) row.size() < g * half || (int64_t) col.size() < g * (image_dim_ - half) ||
            (int64_t) tmp.size() < image_dim_)
            throw std::runtime_error(path_ + ": the image position tables are short");
        pos_table_.assign((size_t) (g * g * image_dim_), 0.f);
        for (int64_t y = 0; y < g; ++y)
            for (int64_t x = 0; x < g; ++x) {
                float* o = pos_table_.data() + (y * g + x) * image_dim_;
                for (int64_t c = 0; c < image_dim_ - half; ++c) o[c] = col[(size_t) (x * (image_dim_ - half) + c)] + tmp[(size_t) c];
                for (int64_t c = 0; c < half; ++c) o[image_dim_ - half + c] = row[(size_t) (y * half + c)] + tmp[(size_t) (image_dim_ - half + c)];
            }
        {
            auto* d = static_cast<float*>(alloc(pos_table_.size() * 4));
            gpu::copy(d, pos_table_.data(), pos_table_.size() * 4);
            weight_bytes_ += pos_table_.size() * 4;
            pos_dev_ = d;
        }
        {
            // image_projection [image_dim][d] -> the product's [d][image_dim]
            const auto w = st_->f32("image_projection");
            std::vector<float> t((size_t) (d_ * image_dim_));
            for (int64_t i = 0; i < image_dim_; ++i)
                for (int64_t j = 0; j < d_; ++j) t[(size_t) (j * image_dim_ + i)] = w[(size_t) (i * d_ + j)];
            const auto h = f16_of(t);
            auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
            gpu::copy(dw, h.data(), h.size() * 2);
            proj_.w = dw;
            proj_.n = d_;
            proj_.k = image_dim_;
            weight_bytes_ += h.size() * 2;
        }
        proj_norm_ = norm("image_proj_norm");
        shared_ = f16("language_model.model.shared.weight");
        enc_pos_ = f16("language_model.model.encoder.embed_positions.weight");
        dec_pos_ = f16("language_model.model.decoder.embed_positions.weight");
        {
            std::vector<float> lb((size_t) vocab_, 0.f);
            if (st_->find("language_model.final_logits_bias") != nullptr) {
                const auto v = st_->f32("language_model.final_logits_bias");
                std::copy(v.begin(), v.begin() + std::min<int64_t>((int64_t) v.size(), vocab_), lb.begin());
            }
            auto* d = static_cast<float*>(alloc(lb.size() * 4));
            gpu::copy(d, lb.data(), lb.size() * 4);
            logits_bias_ = d;
            weight_bytes_ += lb.size() * 4;
        }
        enc_ln_ = norm("language_model.model.encoder.layernorm_embedding");
        dec_ln_ = norm("language_model.model.decoder.layernorm_embedding");
        for (int i = 0;; ++i) {
            const std::string p = "language_model.model.encoder.layers." + std::to_string(i) + ".";
            if (st_->find(p + "fc1.weight") == nullptr) break;
            ELayer l;
            l.qkv = lin({p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"});
            l.o = lin(p + "self_attn.out_proj");
            l.n_attn = norm(p + "self_attn_layer_norm");
            l.fc1 = lin(p + "fc1");
            l.fc2 = lin(p + "fc2");
            l.n_ffn = norm(p + "final_layer_norm");
            enc_.push_back(l);
        }
        for (int i = 0;; ++i) {
            const std::string p = "language_model.model.decoder.layers." + std::to_string(i) + ".";
            if (st_->find(p + "fc1.weight") == nullptr) break;
            DLayer l;
            l.qkv = lin({p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"});
            l.o = lin(p + "self_attn.out_proj");
            l.n_self = norm(p + "self_attn_layer_norm");
            l.cq = lin(p + "encoder_attn.q_proj");
            l.ckv = lin({p + "encoder_attn.k_proj", p + "encoder_attn.v_proj"});
            l.co = lin(p + "encoder_attn.out_proj");
            l.n_cross = norm(p + "encoder_attn_layer_norm");
            l.fc1 = lin(p + "fc1");
            l.fc2 = lin(p + "fc2");
            l.n_ffn = norm(p + "final_layer_norm");
            dec_.push_back(l);
        }
        if (enc_.empty() || dec_.empty()) throw std::runtime_error(path_ + ": no encoder or decoder layers");
    }
}

// slots of the scratch: the run's buffers, grown to the largest stage and kept
enum Slot { kX, kY, kBig, kH16, kG16, kW16, kA16, kQT, kKT, kV16, kS, kCol, kImg, kRgba, kFeats, kEnc16, kDecFirst };

void Florence2::vision(const QiRef& image) {
    const int64_t S = image_size_;
    // the picture at S x S (Pillow's bicubic, as the processor's CLIPImageProcessor resizes), normalized
    const QiRef r = pillow_resize(image, S, S, Filter::bicubic);
    if (dump) dump("rgb768", std::vector<float>(r.rgba.begin(), r.rgba.end()));
    auto* rgba = reinterpret_cast<uint8_t*>(scratch(kRgba, r.rgba.size()));
    gpu::copy(rgba, r.rgba.data(), r.rgba.size());
    auto* img16 = reinterpret_cast<uint16_t*>(scratch(kImg, (size_t) (S * S * 3 * 2)));
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    aux_rgb_norm16(rgba, S, S, mean, stdv, img16, s_);
    // the buffers at the largest stage's size
    size_t xb = 0, bigb = 0;
    {
        int64_t H = S / 4;
        for (const Stage& st : stages_) {
            xb = std::max(xb, (size_t) (H * H * st.dim * 4));
            bigb = std::max(bigb, (size_t) (H * H * st.dim * 4 * 4));
            H /= 2;
        }
    }
    float* x = scratch(kX, xb);
    float* y = scratch(kY, xb);
    float* big = scratch(kBig, bigb);
    auto* h16 = reinterpret_cast<uint16_t*>(scratch(kH16, xb / 2));
    auto* g16 = reinterpret_cast<uint16_t*>(scratch(kG16, bigb / 2));
    auto* w16 = reinterpret_cast<uint16_t*>(scratch(kW16, xb / 2));
    auto* a16 = reinterpret_cast<uint16_t*>(scratch(kA16, xb / 2));
    auto* qT = reinterpret_cast<uint16_t*>(scratch(kQT, xb / 2));
    auto* kT = reinterpret_cast<uint16_t*>(scratch(kKT, xb / 2));
    auto* v16 = reinterpret_cast<uint16_t*>(scratch(kV16, xb / 2));
    int64_t H = S, W = S, C = 3;
    for (size_t si = 0; si < stages_.size(); ++si) {
        const Stage& st = stages_[si];
        if (st.pre_norm) {
            aux_layer_norm(x, nullptr, st.embed_norm.w, st.embed_norm.b, H * W, C, kEps, nullptr, h16, s_);
            convolve(st.embed, h16, H, W, x, kCol);
        } else {
            convolve(st.embed, img16, H, W, y, kCol);
        }
        const int64_t ek = st.embed.k, ep = st.embed.pad, es = st.embed.stride;
        H = (H + 2 * ep - ek) / es + 1;
        W = (W + 2 * ep - ek) / es + 1;
        C = st.dim;
        const int64_t N = H * W;
        if (!st.pre_norm) aux_layer_norm(y, nullptr, st.embed_norm.w, st.embed_norm.b, N, C, kEps, x, nullptr, s_);
        if (si == 0) host("conv0", x, (size_t) (N * C));
        const int64_t ws = window_;
        if (H % ws != 0 || W % ws != 0)
            throw std::runtime_error("image engine (Florence-2): the stage's size is not a multiple of the window");
        const int64_t win = ws * ws, nw = N / win, heads = st.heads, hd = C / heads, G = st.groups, Cg = C / G;
        // the products' bias and activation with them (product16) and their residual's add with its bias in one pass
        // (aux_residual_norm16, no norm: a depthwise convolution comes next)
        auto ffn = [&](const VBlock& b) {
            aux_dwconv3_norm16(x, b.conv2.w, b.conv2.b, b.n2.w, b.n2.b, H, W, C, 0, kEps, y, h16, s_);
            std::swap(x, y);
            product16(h16, b.fc1, g16, N, true, ImageAct::gelu);
            product(g16, b.fc2, y, N, 0, false);
            aux_residual_norm16(x, y, b.fc2.b, nullptr, nullptr, H, W, C, 0, 0, kEps, nullptr, s_);
        };
        for (size_t j = 0; j < st.spatial.size(); ++j) {
            // the spatial block: windows of ws x ws, heads of 32 channels
            const VBlock& sb = st.spatial[j];
            aux_dwconv3_norm16(x, sb.conv1.w, sb.conv1.b, sb.n1.w, sb.n1.b, H, W, C, ws, kEps, y, w16, s_);
            std::swap(x, y);
            // q, k and v each its own rows (the packed weight's thirds), so the windows go to a library in one call
            const Lin* parts[3] = {&sb.q, &sb.k, &sb.v};
            uint16_t* out3[3] = {qT, kT, v16};
            for (int i = 0; i < 3; ++i) product16(w16, *parts[i], out3[i], N, true);
            QiSdpa a;
            a.q = qT;
            a.k = kT;
            a.v = v16;
            a.out = a16;
            a.batch = nw;
            a.heads = heads;
            a.lq = a.lk = win;
            a.hd = hd;
            a.q_row = a.k_row = a.v_row = C;
            a.q_batch = a.k_batch = a.v_batch = win * C;
            a.o_row = C;
            a.o_batch = win * C;
            a.scale = 1.f / std::sqrt((float) hd);
            qi_sdpa_ex(a, s_);
            product(a16, sb.proj, y, N, 0, false);
            aux_residual_norm16(x, y, sb.proj.b, nullptr, nullptr, H, W, C, ws, 0, kEps, nullptr, s_);
            ffn(sb);
            if (si == 0 && j == 0) host("s0spatial", x, (size_t) (N * C));
            // the channel block: groups of Cg channels attend to each other over the N tokens
            const VBlock& cb = st.channel[j];
            aux_dwconv3_norm16(x, cb.conv1.w, cb.conv1.b, cb.n1.w, cb.n1.b, H, W, C, 0, kEps, y, h16, s_);
            std::swap(x, y);
            // the groups' products as one batch each (G of them a block: 750 small products in the large's vision)
            product16(h16, cb.qkv, g16, N, true);
            if (C % 32 == 0) {
                aux_channel_split16(g16, N, C, G, 1.f / std::sqrt((float) N), qT, kT, v16, s_);
            } else {
                product(h16, cb.qkv, big, N);
                aux_channel_split(big, N, C, G, 1.f / std::sqrt((float) N), qT, kT, v16, s_);
            }
            float* sc = scratch(kS, (size_t) (G * Cg * Cg * 4));
            auto* p16 = reinterpret_cast<uint16_t*>(big);   // big is free again
            // the channels attend to each other over the tokens (v . p^T), the shared matrix-engine attention; the
            // scale 1/sqrt(N) is folded into q by the split.  Each group's out [N][Cg] merged into the tokens' rows
            aux_attn_matrix(gemm_, qT, kT, v16, G, 1, Cg, Cg, N, N, Cg, true, 1.f, nullptr, nullptr, 1, nullptr, 0, 0,
                            sc, p16, y, s_);
            aux_channel_merge16(y, N, C, G, a16, s_);
            product(a16, cb.proj, y, N, 0, false);
            aux_residual_norm16(x, y, cb.proj.b, nullptr, nullptr, H, W, C, 0, 0, kEps, nullptr, s_);
            ffn(cb);
        }
        host(("stage" + std::to_string(si)).c_str(), x, (size_t) (N * C));
    }
    // the projector: the 2D and temporal positions, the pooled token and the tokens, the projection and its norm
    const int64_t N = H * W;
    if (N * image_dim_ != (int64_t) pos_table_.size())
        throw std::runtime_error("image engine (Florence-2): the last stage's size is not the position table's");
    host("vision_out", x, (size_t) (N * image_dim_));
    qi_axpy(x, pos_dev_, 1.f, N * image_dim_, s_);
    {
        std::vector<float> t((size_t) (N * image_dim_)), m((size_t) image_dim_, 0.f);
        gpu::stream_sync(s_);
        gpu::copy(t.data(), x, t.size() * 4);
        std::vector<double> acc((size_t) image_dim_, 0.0);
        for (int64_t i = 0; i < N; ++i)
            for (int64_t c = 0; c < image_dim_; ++c) acc[(size_t) c] += t[(size_t) (i * image_dim_ + c)];
        for (int64_t c = 0; c < image_dim_; ++c) m[(size_t) c] = (float) (acc[(size_t) c] / (double) N);
        // [pooled; tokens] as FP16 for the projection
        std::vector<float> all((size_t) ((N + 1) * image_dim_));
        std::copy(m.begin(), m.end(), all.begin());
        std::copy(t.begin(), t.end(), all.begin() + image_dim_);
        const auto h = f16_of(all);
        gpu::copy(h16, h.data(), h.size() * 2);
    }
    img_tokens_ = N + 1;
    feats_ = scratch(kFeats, (size_t) ((img_tokens_ + max_pos_) * d_ * 4));
    gemm_.f16(h16, proj_.w, y, img_tokens_, d_, image_dim_);
    aux_layer_norm(y, nullptr, proj_norm_.w, proj_norm_.b, img_tokens_, d_, kEps, feats_, nullptr, s_);
    host("image_features", feats_, (size_t) (img_tokens_ * d_));
}

void Florence2::encoder(const std::vector<int32_t>& text_ids) {
    const int64_t n = (int64_t) text_ids.size(), T = img_tokens_ + n;
    if (T > max_pos_) throw std::invalid_argument("image engine (Florence-2): the words are too long");
    for (int32_t id : text_ids)
        if (id < 0 || id >= vocab_) throw std::invalid_argument("image engine (Florence-2): a token outside the vocabulary");
    enc_T_ = T;
    std::vector<int32_t> ids((size_t) T, 0), pos((size_t) T);
    for (int64_t i = 0; i < n; ++i) ids[(size_t) (img_tokens_ + i)] = text_ids[(size_t) i];
    std::iota(pos.begin(), pos.end(), (int32_t) kPosOffset);
    auto* dids = reinterpret_cast<int32_t*>(scratch(kS, (size_t) (2 * T * 4)));
    gpu::copy(dids, ids.data(), (size_t) T * 4);
    gpu::copy(dids + T, pos.data(), (size_t) T * 4);
    float* x = feats_;   // the picture's tokens, then the words' (scale_embedding false: times 1)
    aux_embed(dids, dids + T, nullptr, enc_pos_, img_tokens_, d_, 1.f, x, s_);
    aux_embed(dids + img_tokens_, dids + T + img_tokens_, shared_, enc_pos_, n, d_, 1.f, x + img_tokens_ * d_, s_);
    float* y = scratch(kY, (size_t) (T * std::max(ffn_, 3 * d_) * 4));
    float* big = scratch(kBig, (size_t) (T * std::max(ffn_, 3 * d_) * 4));
    auto* x16 = reinterpret_cast<uint16_t*>(scratch(kEnc16, (size_t) (T * d_ * 2)));
    auto* g16 = reinterpret_cast<uint16_t*>(scratch(kG16, (size_t) (T * std::max(ffn_, 3 * d_) * 2)));
    auto* a16 = reinterpret_cast<uint16_t*>(scratch(kA16, (size_t) (T * d_ * 2)));
    aux_layer_norm(x, nullptr, enc_ln_.w, enc_ln_.b, T, d_, kEps, x, x16, s_);
    const int64_t hd = d_ / heads_;
    for (const ELayer& l : enc_) {
        product(x16, l.qkv, big, T);
        aux_split3_16(big, T, d_, g16, g16 + T * d_, g16 + 2 * T * d_, s_);
        QiSdpa a;
        a.q = g16;
        a.k = g16 + T * d_;
        a.v = g16 + 2 * T * d_;
        a.out = a16;
        a.heads = heads_;
        a.lq = a.lk = T;
        a.hd = hd;
        a.q_row = a.k_row = a.v_row = a.o_row = d_;
        a.q_batch = a.k_batch = a.v_batch = a.o_batch = T * d_;
        a.scale = 1.f / std::sqrt((float) hd);
        qi_sdpa_ex(a, s_);
        product(a16, l.o, y, T);
        aux_layer_norm(x, y, l.n_attn.w, l.n_attn.b, T, d_, kEps, x, x16, s_);
        product(x16, l.fc1, big, T);
        qi_gelu_erf16(big, T * ffn_, g16, s_);
        product(g16, l.fc2, y, T);
        aux_layer_norm(x, y, l.n_ffn.w, l.n_ffn.b, T, d_, kEps, x, x16, s_);
    }
    host("encoder", x, (size_t) (T * d_));
}

// A decoder step of B beams, onto the stream: what changes from step to step is data in `dids` (decode()), so a
// recording of it replays any step
void Florence2::step(int64_t B, int64_t T, int64_t cap, uint16_t* cache, const uint16_t* ckv, const int32_t* dids,
                     float* h, float* y, uint16_t* h16, uint16_t* g16, uint16_t* a16, uint16_t* q16, float* logits) {
    const int64_t d = d_, L = (int64_t) dec_.size(), hd = d_ / heads_, V = vocab_;
    aux_embed(dids, dids + B, shared_, dec_pos_, B, d, 1.f, h, s_);
    aux_layer_norm(h, nullptr, dec_ln_.w, dec_ln_.b, B, d, kEps, h, h16, s_);
    for (int64_t li = 0; li < L; ++li) {
        const DLayer& l = dec_[(size_t) li];
        uint16_t* kc = cache + (li * 2) * B * cap * d;
        uint16_t* vc = cache + (li * 2 + 1) * B * cap * d;
        // the step's few rows: the post-norms with the residual and the bias, the caches' writes in one
        product(h16, l.qkv, y, B);
        kernels::f32_to_f16_bulk(y, g16, B * 3 * d, s_);
        aux_cache_put_kv(g16, B, 3 * d, d, 2 * d, d, kc, vc, cap, dids + 2 * B, s_);
        QiSdpa a;
        a.q = g16;
        a.k = kc;
        a.v = vc;
        a.out = a16;
        a.batch = B;
        a.heads = heads_;
        a.lq = 1;
        a.lk = cap;
        a.key_len = dids + 2 * B + 1;
        a.hd = hd;
        a.q_row = a.q_batch = 3 * d;
        a.k_row = a.v_row = d;
        a.k_batch = a.v_batch = cap * d;
        a.o_row = a.o_batch = d;
        a.scale = 1.f / std::sqrt((float) hd);
        qi_sdpa_ex(a, s_);
        product(a16, l.o, y, B, 0, false);
        aux_add_norm(h, y, l.o.b, l.n_self.w, l.n_self.b, B, d, kEps, h, h16, s_);
        // the cross-attention over the encoder's states, every beam the same keys
        product(h16, l.cq, y, B);
        kernels::f32_to_f16_bulk(y, q16, B * d, s_);
        QiSdpa c;
        c.q = q16;
        c.k = ckv + li * 2 * T * d;
        c.v = ckv + (li * 2 + 1) * T * d;
        c.out = a16;
        c.batch = B;
        c.kv_div = B;
        c.heads = heads_;
        c.lq = 1;
        c.lk = T;
        c.hd = hd;
        c.q_row = c.q_batch = c.o_row = c.o_batch = d;
        c.k_row = c.v_row = d;
        c.k_batch = c.v_batch = T * d;
        c.scale = 1.f / std::sqrt((float) hd);
        qi_sdpa_ex(c, s_);
        product(a16, l.co, y, B, 0, false);
        aux_add_norm(h, y, l.co.b, l.n_cross.w, l.n_cross.b, B, d, kEps, h, h16, s_);
        product(h16, l.fc1, y, B);
        qi_gelu_erf16(y, B * ffn_, g16, s_);
        product(g16, l.fc2, y, B, 0, false);
        aux_add_norm(h, y, l.fc2.b, l.n_ffn.w, l.n_ffn.b, B, d, kEps, h, h16, s_);
    }
    gemm_.f16(h16, shared_, logits, B, V, d);
    qi_bias_add(logits, logits_bias_, B, V, s_);
}

std::vector<int32_t> Florence2::decode(int max_new, int beams, int& steps, const std::function<void(int)>& each) {
    const int64_t B = beams, T = enc_T_, d = d_, L = (int64_t) dec_.size(), V = vocab_;
    const int64_t max_length = 1 + max_new, cap = max_length;
    if (max_length > max_pos_) throw std::invalid_argument("image engine (Florence-2): max_new past the positions");
    const uint16_t* enc16 = reinterpret_cast<const uint16_t*>(scratch(kEnc16, (size_t) (T * d * 2)));
    // the cross-attention's keys and values, once a request: [layer][2][T][d]
    auto* ckv = reinterpret_cast<uint16_t*>(scratch(kDecFirst, (size_t) (L * 2 * T * d * 2)));
    float* big = scratch(kBig, (size_t) (T * std::max(ffn_, 3 * d) * 4));
    auto* kv = reinterpret_cast<uint16_t*>(scratch(kG16, (size_t) (T * 2 * d * 2)));
    for (int64_t i = 0; i < L; ++i) {
        product(enc16, dec_[(size_t) i].ckv, big, T);
        kernels::f32_to_f16_bulk(big, kv, T * 2 * d, s_);
        // [T][2d] -> keys [T][d], values [T][d] (aux_cache_put's T "beams" of one row each)
        aux_cache_put(kv, T, 2 * d, 0, d, ckv + i * 2 * T * d, 1, 0, s_);
        aux_cache_put(kv, T, 2 * d, d, d, ckv + (i * 2 + 1) * T * d, 1, 0, s_);
    }
    // the self-attention's caches: [layer][2][B][cap][d], and a second set for the beams' reorder
    const size_t cache_elems = (size_t) (L * 2 * B * cap * d);
    auto* cache = reinterpret_cast<uint16_t*>(scratch(kDecFirst + 1, cache_elems * 2));
    auto* cache2 = reinterpret_cast<uint16_t*>(scratch(kDecFirst + 2, cache_elems * 2));
    float* h = scratch(kX, (size_t) (B * d * 4));
    float* y = scratch(kY, (size_t) (B * std::max(ffn_, 3 * d) * 4));
    auto* h16 = reinterpret_cast<uint16_t*>(scratch(kH16, (size_t) (B * d * 2)));
    auto* g16 = reinterpret_cast<uint16_t*>(scratch(kW16, (size_t) (B * std::max(ffn_, 3 * d) * 2)));
    auto* a16 = reinterpret_cast<uint16_t*>(scratch(kA16, (size_t) (B * d * 2)));
    auto* q16 = reinterpret_cast<uint16_t*>(scratch(kQT, (size_t) (B * d * 2)));
    float* logits = scratch(kCol, (size_t) (B * V * 4));
    // the step's data on the device (a recorded step reads it at each replay): the beams' tokens [B], their positions
    // [B], the caches' row, the self-attention's keys [B]; then the reorder's sources [B], which the copy of the next
    // step's leaves alone (the copy does not wait for the stream's gather)
    auto* dids = reinterpret_cast<int32_t*>(scratch(kS, (size_t) ((4 * B + 1) * 4)));
    std::vector<float> lg((size_t) (B * V));

    // the beam search's state (transformers' _beam_search, one batch)
    std::vector<std::vector<int32_t>> run_seq((size_t) B, std::vector<int32_t>{kEos});
    std::vector<float> run_score((size_t) B, kNeg);
    run_score[0] = 0.f;
    std::vector<std::vector<int32_t>> fin_seq((size_t) B, std::vector<int32_t>{kEos});
    std::vector<float> fin_score((size_t) B, kNeg);
    std::vector<char> fin_done((size_t) B, 0);
    bool heur = true;
    int64_t cur_len = 1;
    std::vector<int32_t> toks((size_t) B, kEos);
    const int64_t K2 = 2 * B;
    std::vector<char> banned((size_t) V, 0);   // the step's banned tokens of a beam, cleared after it
    steps = 0;
    for (;;) {
        // the step: each beam's last token at position cur_len - 1
        const int64_t at = cur_len - 1;
        std::vector<int32_t> in((size_t) (3 * B + 1));
        for (int64_t b = 0; b < B; ++b) {
            in[(size_t) b] = toks[(size_t) b];
            in[(size_t) (B + b)] = (int32_t) (at + kPosOffset);
            in[(size_t) (2 * B + 1 + b)] = (int32_t) (at + 1);
        }
        in[(size_t) (2 * B)] = (int32_t) at;
        gpu::copy(dids, in.data(), in.size() * 4);
        // what the step bakes in: the buffers (the caches' current set among them) and the shapes
        uint64_t key = 1469598103934665603ull;
        for (const uint64_t v : {(uint64_t) (uintptr_t) cache, (uint64_t) (uintptr_t) h, (uint64_t) (uintptr_t) y,
                                 (uint64_t) (uintptr_t) logits, (uint64_t) (uintptr_t) dids,
                                 (uint64_t) (uintptr_t) ckv, (uint64_t) (uintptr_t) g16, (uint64_t) (uintptr_t) h16,
                                 (uint64_t) (uintptr_t) a16, (uint64_t) (uintptr_t) q16, (uint64_t) B,
                                 (uint64_t) cap, (uint64_t) T})
            key = (key ^ v) * 1099511628211ull;
        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        // a recording checked once against the step itself on the logits, cleared before its replay (the step writes
        // its caches' row again)
        const auto out = [&] {
            std::vector<uint8_t> v((size_t) (B * V * 4));
            gpu::stream_sync(s_);
            gpu::copy(v.data(), logits, v.size());
            return v;
        };
        if (!steps_.run(key, s_, [&] { step(B, T, cap, cache, ckv, dids, h, y, h16, g16, a16, q16, logits); }, err,
                        out, [&] { gpu::memset_async(logits, 0, (size_t) (B * V * 4), s_); }))
            throw std::runtime_error("image engine (Florence-2): " + err);
        gpu::stream_sync(s_);
        if (steps_.form() == -1)
            steps_.timed(since(t0) * 1e3, "xestrata image (Florence-2)", "the decoder's steps");
        gpu::copy(lg.data(), logits, lg.size() * 4);
        if (steps < 4 && dump)
            dump(("logits" + std::to_string(steps)).c_str(), std::vector<float>(lg.begin(), lg.begin() + V));
        ++steps;
        if (each) each(steps);

        // log-softmax, the processors (no repeated 3-gram, <s> first, </s> at the last place), the running scores
        // a beam's values: the log-softmax, the banned ones and all but the forced token -inf, plus its running score.
        // The step's top 2 x beams (the higher first, the lower index on a tie) are among each beam's own top 2 x beams,
        // each picked in one pass over its row (adding a beam's running score keeps its order)
        constexpr float ninf = -std::numeric_limits<float>::infinity();
        std::vector<std::pair<float, int64_t>> pool;
        pool.reserve((size_t) (B * K2));
        const bool forced = cur_len == 1 || cur_len == max_length - 1;
        const int32_t keep = cur_len == 1 ? kBos : kEos;
        for (int64_t b = 0; b < B; ++b) {
            const float* r = lg.data() + b * V;
            const float mx = *std::max_element(r, r + V);
            double z = 0.0;
            for (int64_t v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
            const float lse = mx + (float) std::log(std::max(z, 1.0));   // the largest's own term is 1
            const auto& sq = run_seq[(size_t) b];
            std::vector<int32_t> ban;
            if (cur_len + 1 >= kNgram) {
                const int32_t p0 = sq[(size_t) (cur_len - 2)], p1 = sq[(size_t) (cur_len - 1)];
                for (int64_t i = 0; i + 2 < cur_len; ++i)
                    if (sq[(size_t) i] == p0 && sq[(size_t) (i + 1)] == p1) ban.push_back(sq[(size_t) (i + 2)]);
            }
            for (int32_t t : ban) banned[(size_t) t] = 1;
            std::vector<std::pair<float, int64_t>> best;   // the beam's top K2, the higher first
            for (int64_t v = 0; v < V; ++v) {
                const float val = forced ? (v == keep ? 0.f : ninf) : banned[(size_t) v] ? ninf : r[v] - lse;
                if ((int64_t) best.size() == K2 && !(val > best.back().first)) continue;
                auto at = std::find_if(best.begin(), best.end(), [&](const auto& e) { return val > e.first; });
                best.insert(at, {val, v});
                if ((int64_t) best.size() > K2) best.pop_back();
            }
            for (int32_t t : ban) banned[(size_t) t] = 0;
            for (const auto& e : best) pool.push_back({e.first + run_score[(size_t) b], b * V + e.second});
        }
        std::stable_sort(pool.begin(), pool.end(), [](const auto& a, const auto& c) {
            return a.first > c.first || (a.first == c.first && a.second < c.second);
        });
        std::vector<int64_t> top((size_t) K2);
        std::vector<float> top_val((size_t) K2);
        for (int64_t k = 0; k < K2; ++k) top[(size_t) k] = pool[(size_t) k].second, top_val[(size_t) k] = pool[(size_t) k].first;
        std::vector<std::vector<int32_t>> cand_seq((size_t) K2);
        std::vector<float> cand_score((size_t) K2);
        std::vector<int32_t> cand_src((size_t) K2);
        std::vector<char> hits((size_t) K2);
        bool all_hit = true;
        for (int64_t k = 0; k < K2; ++k) {
            const int64_t f = top[(size_t) k], src = f / V;
            const auto tok = (int32_t) (f % V);
            cand_src[(size_t) k] = (int32_t) src;
            cand_seq[(size_t) k] = run_seq[(size_t) src];
            cand_seq[(size_t) k].push_back(tok);
            cand_score[(size_t) k] = top_val[(size_t) k];
            hits[(size_t) k] = tok == kEos || cur_len + 1 >= max_length ? 1 : 0;
            all_hit = all_hit && hits[(size_t) k];
        }
        // the running beams of the next step: the best that did not finish
        std::vector<float> rs((size_t) K2);
        for (int64_t k = 0; k < K2; ++k) rs[(size_t) k] = cand_score[(size_t) k] + (hits[(size_t) k] ? kNeg : 0.f);
        std::vector<int64_t> order((size_t) K2);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) { return rs[(size_t) i] > rs[(size_t) j]; });
        // the finished ones: the best beams so far and the top beams' new </s> (length penalty 1)
        const bool full = std::all_of(fin_done.begin(), fin_done.end(), [](char c) { return c != 0; });
        std::vector<float> ms(fin_score);
        std::vector<std::vector<int32_t>> mseq(fin_seq);
        std::vector<char> mdone(fin_done);
        for (int64_t k = 0; k < K2; ++k) {
            const bool just = hits[(size_t) k] && k < B;
            float sc = cand_score[(size_t) k] / (float) cur_len;
            if (full) sc += kNeg;
            if (!heur) sc += kNeg;
            if (!just) sc += kNeg;
            ms.push_back(sc);
            mseq.push_back(cand_seq[(size_t) k]);
            mdone.push_back(just ? 1 : 0);
        }
        std::vector<int64_t> mo(ms.size());
        std::iota(mo.begin(), mo.end(), 0);
        std::stable_sort(mo.begin(), mo.end(), [&](int64_t i, int64_t j) { return ms[(size_t) i] > ms[(size_t) j]; });
        for (int64_t b = 0; b < B; ++b) {
            const auto m = (size_t) mo[(size_t) b];
            fin_score[(size_t) b] = ms[m];
            fin_seq[(size_t) b] = mseq[m];
            fin_done[(size_t) b] = mdone[m];
        }
        // the caches follow the running beams' sources
        std::vector<int32_t> from((size_t) B);
        bool same = true;
        for (int64_t b = 0; b < B; ++b) {
            const auto k = (size_t) order[(size_t) b];
            from[(size_t) b] = cand_src[k];
            same = same && cand_src[k] == (int32_t) b;
            run_seq[(size_t) b] = cand_seq[k];
            run_score[(size_t) b] = rs[k];
            toks[(size_t) b] = cand_seq[k].back();
        }
        if (!same) {
            gpu::copy(dids + 3 * B + 1, from.data(), (size_t) B * 4);
            for (int64_t li = 0; li < 2 * L; ++li)
                aux_cache_gather(cache + li * B * cap * d, dids + 3 * B + 1, B, cap, cur_len, d, cache2 + li * B * cap * d, s_);
            std::swap(cache, cache2);
        }
        ++cur_len;
        {
            const float best = run_score[0] / (float) (cur_len - 1);
            const float lo = *std::min_element(fin_score.begin(), fin_score.end());
            bool any = false;
            for (int64_t b = 0; b < B; ++b) any = any || best > (fin_done[(size_t) b] ? lo : kNeg);
            heur = heur && any;
        }
        const bool all_done = std::all_of(fin_done.begin(), fin_done.end(), [](char c) { return c != 0; });
        if (!heur || all_done || all_hit) break;
    }
    gpu::stream_sync(s_);
    return fin_seq[0];
}

std::vector<int32_t> Florence2::run(const QiRef& image, const std::vector<int32_t>& text_ids, int max_new, int beams,
                                    Times* t) {
    if (image.width <= 0 || image.height <= 0 || image.rgba.size() != (size_t) (image.width * image.height * 4))
        throw std::invalid_argument("image engine (Florence-2): no picture");
    if (beams < 1 || beams > 8 || max_new < 1) throw std::invalid_argument("image engine (Florence-2): beams 1 to 8");
    load();
    // the share done (progress): the vision tower, the encoder and the decoder weighed by the last run's seconds of
    // each (before there is one: 3, 1 and 6 tenths), the decoder's steps against the last run's (32 before one, its
    // share held under its end until the answer ends: an answer's length is not known ahead)
    const double last = last_.vision + last_.encoder + last_.decoder;
    const double wv = last > 0 ? last_.vision / last : 0.3, we = last > 0 ? last_.encoder / last : 0.1;
    const double expect = last_.steps > 0 ? last_.steps : 32;
    if (progress) progress(0.0);
    Times now;
    auto t0 = std::chrono::steady_clock::now();
    vision(image);
    gpu::stream_sync(s_);
    now.vision = since(t0);
    if (progress) progress(wv);
    t0 = std::chrono::steady_clock::now();
    encoder(text_ids);
    gpu::stream_sync(s_);
    now.encoder = since(t0);
    if (progress) progress(wv + we);
    t0 = std::chrono::steady_clock::now();
    int steps = 0;
    auto ids = decode(std::min<int>(max_new, (int) max_pos_ - 1), beams, steps, progress ? std::function<void(int)>(
        [&](int done) { progress(wv + we + (1.0 - wv - we) * std::min(0.95, done / expect)); }) : nullptr);
    now.decoder = since(t0);
    now.steps = steps;
    last_ = now;
    if (t) *t = now;
    if (progress) progress(1.0);
    return ids;
}

QiRef Florence2::polygon_mask(const std::vector<std::vector<float>>& polygons, int64_t W, int64_t H) {
    if (W <= 0 || H <= 0) throw std::invalid_argument("image engine: a mask of no size");
    std::vector<float> xy;
    std::vector<int32_t> starts{0};
    for (const auto& p : polygons) {
        if (p.size() % 2 != 0) throw std::invalid_argument("image engine: a polygon's coordinates come in pairs");
        for (size_t i = 0; i < p.size(); i += 2) {   // clamped to the picture, as the reference's drawing is
            xy.push_back(std::clamp(p[i], 0.f, (float) (W - 1)));
            xy.push_back(std::clamp(p[i + 1], 0.f, (float) (H - 1)));
        }
        starts.push_back((int32_t) (xy.size() / 2));
    }
    void* s = nullptr;   // the engine's default queue
    float* dxy = nullptr;
    int32_t* dst = nullptr;
    uint8_t* dm = nullptr;
    if (!gpu::alloc_device(&dxy, std::max<size_t>(xy.size(), 2) * 4) || !gpu::alloc_device(&dst, starts.size() * 4) ||
        !gpu::alloc_device(&dm, (size_t) (W * H))) {
        gpu::free(dxy), gpu::free(dst), gpu::free(dm);
        throw std::runtime_error(std::string("image engine: GPU memory (mask): ") + gpu::last_error());
    }
    if (!xy.empty()) gpu::copy(dxy, xy.data(), xy.size() * 4);
    gpu::copy(dst, starts.data(), starts.size() * 4);
    aux_polygon_fill(dxy, dst, (int64_t) polygons.size(), W, H, dm, s);
    std::vector<uint8_t> m((size_t) (W * H));
    gpu::copy(m.data(), dm, m.size());
    gpu::free(dxy), gpu::free(dst), gpu::free(dm);
    QiRef r;
    r.width = W;
    r.height = H;
    r.rgba.assign((size_t) (W * H * 4), 255);
    for (size_t i = 0; i < m.size(); ++i) {
        r.rgba[i * 4] = r.rgba[i * 4 + 1] = r.rgba[i * 4 + 2] = 0;
        r.rgba[i * 4 + 3] = (uint8_t) (255 - m[i]);   // OpenAI's alpha: 0 where the picture changes
    }
    return r;
}

}  // namespace strata::image
