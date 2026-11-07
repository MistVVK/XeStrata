// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/image/sam3.cpp - see include/strata/image/sam3.hpp.  The order of the operations, the layers' wiring and the
// constants follow vLLM-Omni's sam3_model.py (15bb15d, Apache-2.0); the tensor names are the native checkpoint's.

#include "strata/image/sam3.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr const char* kV = "detector.backbone.vision_backbone.";
constexpr const char* kT = "detector.backbone.language_backbone.";
constexpr const char* kE = "detector.transformer.encoder.layers.";
constexpr const char* kD = "detector.transformer.decoder.";
constexpr const char* kS = "detector.dot_prod_scoring.";
constexpr const char* kM = "detector.segmentation_head.";

constexpr float kVitEps = 1e-6f, kEps = 1e-5f;
constexpr float kTheta = 10000.f;
constexpr float kPi = 3.14159265358979323846f;
constexpr int64_t kMaskChunk = 16;   // the kept queries' masks made at once

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

float inverse_sigmoid(float x) {
    x = std::clamp(x, 0.f, 1.f);
    return std::log(std::max(x, 1e-3f) / std::max(1.f - x, 1e-3f));
}

// The sine embedding of one value (the DETR decoder's boxes): F channels, sin and cos interleaved at 10000^(2i/F)
void sine(float v, int64_t F, float* out) {
    for (int64_t i = 0; i < F; ++i) {
        const int64_t pair = i / 2;   // sin and cos share a frequency
        const float dt = std::pow(10000.f, (float) (2 * pair) / (float) F);
        const float p = v * 2.f * kPi / dt;
        out[i] = i % 2 == 0 ? std::sin(p) : std::cos(p);
    }
}

std::vector<uint16_t> f16_of(const float* v, size_t n) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = kernels::f16_from_f32(v[i]);
    return h;
}

// scratch slots
enum Slot {
    kPix, kRgba, kX, kH16, kW16, kBig, kG16, kQ16, kK16, kV16, kA16, kY, kCol,
    kF0, kF1, kF2, kPos, kTxt, kTxt16, kEnc, kEnc16, kEncPos16, kDec, kDec16, kQp, kBias, kTmp, kPix0, kPix1,
    kInst16, kMasks, kMasks8, kPart, kSmall, kSmall16, kTxtOut, kTmp2, kQo16, kNeck,
};

}  // namespace

Sam3::Sam3(const std::string& path) : AuxNet(path, "SAM 3") {
    const SafeTensor* tok = st_->find(std::string(kT) + "encoder.token_embedding.weight");
    const SafeTensor* pe = st_->find(std::string(kV) + "trunk.patch_embed.proj.weight");
    const SafeTensor* qe = st_->find(std::string(kD) + "query_embed.weight");
    if (tok == nullptr || pe == nullptr || qe == nullptr || tok->shape.size() != 2 || pe->shape.size() != 4)
        throw std::runtime_error(path + ": not SAM 3's detector (Meta's checkpoint layout)");
    vocab_ = tok->shape[0];
    td_ = tok->shape[1];
    wd_ = pe->shape[0];
    patch_ = pe->shape[2];
    queries_ = qe->shape[0];
    d_ = qe->shape[1];
    grid_ = image_ / patch_;
    // the detector's tensors only: the tracker's stay in the file
    file_bytes_ = 0;
    for (const auto& t : st_->tensors())
        if (t.first.rfind("detector.", 0) == 0) file_bytes_ += t.second.end - t.second.begin;
}

Sam3::Attn Sam3::attn(const std::string& name) {
    Attn a;
    lin3(name + ".in_proj_weight", name + ".in_proj_bias", a.q, a.k, a.v);
    a.o = lin(name + ".out_proj");
    return a;
}

void Sam3::load_weights() {
    // again after a release(): what the last load built is gone with the GPU's buffers
    vit_.clear();
    fpn_.clear();
    txt_.clear();
    enc_.clear();
    dec_.clear();
    patch_conv_ = conv(std::string(kV) + "trunk.patch_embed.proj", (int) patch_, 0);
    {
        // the pretraining's table without its class token's row: [24 * 24][wd]
        const auto p = host_f32(std::string(kV) + "trunk.pos_embed");
        const int64_t rows = (int64_t) p.size() / wd_;
        const int64_t side = (int64_t) std::lround(std::sqrt((double) (rows - 1)));
        if (side * side != rows - 1 || side != window_) throw std::runtime_error(path_ + ": the position table's size");
        auto* d = static_cast<float*>(alloc((size_t) (side * side * wd_) * 4));
        gpu::copy(d, p.data() + wd_, (size_t) (side * side * wd_) * 4);
        weight_bytes_ += (size_t) (side * side * wd_) * 4;
        pos_ = d;
    }
    ln_pre_ = norm(std::string(kV) + "trunk.ln_pre");
    for (int i = 0;; ++i) {
        const std::string b = std::string(kV) + "trunk.blocks." + std::to_string(i) + ".";
        if (st_->find(b + "attn.qkv.weight") == nullptr) break;
        VitLayer l;
        l.n1 = norm(b + "norm1");
        l.n2 = norm(b + "norm2");
        l.qkv = lin(b + "attn.qkv");
        l.o = lin(b + "attn.proj");
        l.fc1 = lin(b + "mlp.fc1");
        l.fc2 = lin(b + "mlp.fc2");
        l.global = i % 8 == 7;   // the checkpoint's global_attn_indexes: 7, 15, 23, 31
        vit_.push_back(l);
    }
    if (vit_.size() != 32) throw std::runtime_error(path_ + ": the ViT has " + std::to_string(vit_.size()) + " layers");
    for (int i = 0; i < 3; ++i) {
        const std::string b = std::string(kV) + "convs." + std::to_string(i) + ".";
        Fpn f;
        if (st_->find(b + "dconv_2x2_0.weight") != nullptr) {
            f.up1 = deconv2(b + "dconv_2x2_0");
            f.up2 = deconv2(b + "dconv_2x2_1");
            f.ups = 2;
        } else if (st_->find(b + "dconv_2x2.weight") != nullptr) {
            f.up1 = deconv2(b + "dconv_2x2");
            f.ups = 1;
        }
        f.c1 = conv(b + "conv_1x1", 1, 0);
        f.c3 = conv(b + "conv_3x3", 1, 1);
        fpn_.push_back(f);
    }
    if (fpn_[0].ups != 2 || fpn_[1].ups != 1 || fpn_[2].ups != 0)
        throw std::runtime_error(path_ + ": the neck is not 4x, 2x and 1x");
    // the words
    tok_ = f16(std::string(kT) + "encoder.token_embedding.weight");
    tpos_ = f16(std::string(kT) + "encoder.positional_embedding");
    for (int i = 0;; ++i) {
        const std::string b = std::string(kT) + "encoder.transformer.resblocks." + std::to_string(i) + ".";
        if (st_->find(b + "attn.in_proj_weight") == nullptr) break;
        TextLayer l;
        l.n1 = norm(b + "ln_1");
        l.n2 = norm(b + "ln_2");
        l.qkv = lin_wb(b + "attn.in_proj_weight", b + "attn.in_proj_bias");   // q, k, v in a row
        l.o = lin(b + "attn.out_proj");
        l.fc1 = lin(b + "mlp.c_fc");
        l.fc2 = lin(b + "mlp.c_proj");
        txt_.push_back(l);
    }
    ln_final_ = norm(std::string(kT) + "encoder.ln_final");
    resizer_ = lin(std::string(kT) + "resizer");
    // the DETR encoder
    for (int i = 0;; ++i) {
        const std::string b = std::string(kE) + std::to_string(i) + ".";
        if (st_->find(b + "linear1.weight") == nullptr) break;
        EncLayer l;
        l.n1 = norm(b + "norm1");
        l.n2 = norm(b + "norm2");
        l.n3 = norm(b + "norm3");
        l.self = attn(b + "self_attn");
        l.cross = attn(b + "cross_attn_image");
        l.fc1 = lin(b + "linear1");
        l.fc2 = lin(b + "linear2");
        enc_.push_back(l);
    }
    // the DETR decoder
    for (int i = 0;; ++i) {
        const std::string b = std::string(kD) + "layers." + std::to_string(i) + ".";
        if (st_->find(b + "linear1.weight") == nullptr) break;
        DecLayer l;
        l.self = attn(b + "self_attn");
        l.text = attn(b + "ca_text");
        l.vis = attn(b + "cross_attn");
        l.n_self = norm(b + "norm2");
        l.n_text = norm(b + "catext_norm");
        l.n_vis = norm(b + "norm1");
        l.n_mlp = norm(b + "norm3");
        l.fc1 = lin(b + "linear1");
        l.fc2 = lin(b + "linear2");
        dec_.push_back(l);
    }
    if (enc_.empty() || dec_.empty()) throw std::runtime_error(path_ + ": no DETR encoder or decoder layers");
    dec_out_ = norm(std::string(kD) + "norm");
    pres_norm_ = norm(std::string(kD) + "presence_token_out_norm");
    box1_ = lin(std::string(kD) + "bbox_embed.layers.0");
    box2_ = lin(std::string(kD) + "bbox_embed.layers.1");
    box3_ = lin(std::string(kD) + "bbox_embed.layers.2");
    pres1_ = lin(std::string(kD) + "presence_token_head.layers.0");
    pres2_ = lin(std::string(kD) + "presence_token_head.layers.1");
    pres3_ = lin(std::string(kD) + "presence_token_head.layers.2");
    ref1_ = lin(std::string(kD) + "ref_point_head.layers.0");
    ref2_ = lin(std::string(kD) + "ref_point_head.layers.1");
    query_embed_ = host_f32(std::string(kD) + "query_embed.weight");
    ref_points_ = host_f32(std::string(kD) + "reference_points.weight");
    presence_tok_ = host_f32(std::string(kD) + "presence_token.weight");
    {
        // the box bias's MLPs, the rows' (y) then the columns' (x), as aux_box_bias reads them
        std::vector<float> all;
        for (const char* axis : {"y", "x"})
            for (const char* part : {"layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"}) {
                const auto v = host_f32(std::string(kD) + "boxRPB_embed_" + axis + "." + part);
                all.insert(all.end(), v.begin(), v.end());
            }
        const SafeTensor* t = st_->find(std::string(kD) + "boxRPB_embed_x.layers.0.weight");
        rpb_hidden_ = t->shape[0];
        if ((int64_t) all.size() != 2 * (rpb_hidden_ * 3 + heads_ * rpb_hidden_ + heads_))
            throw std::runtime_error(path_ + ": the box bias's MLPs");
        auto* d = static_cast<float*>(alloc(all.size() * 4));
        gpu::copy(d, all.data(), all.size() * 4);
        weight_bytes_ += all.size() * 4;
        rpb_ = d;
    }
    // the scores
    sc_fc1_ = lin(std::string(kS) + "prompt_mlp.layers.0");
    sc_fc2_ = lin(std::string(kS) + "prompt_mlp.layers.1");
    sc_norm_ = norm(std::string(kS) + "prompt_mlp.out_norm");
    sc_proj_ = lin(std::string(kS) + "prompt_proj");
    sc_qproj_ = lin(std::string(kS) + "hs_proj");
    // the masks
    m_cross_ = attn(std::string(kM) + "cross_attend_prompt");
    m_norm_ = norm(std::string(kM) + "cross_attn_norm");
    for (int i = 0; i < 2; ++i) {
        pix_conv_[i] = conv(std::string(kM) + "pixel_decoder.conv_layers." + std::to_string(i), 1, 1);
        pix_gn_[i] = norm(std::string(kM) + "pixel_decoder.norms." + std::to_string(i));
    }
    {
        const Conv c = conv(std::string(kM) + "instance_seg_head", 1, 0);
        inst_.w = c.w;
        inst_.b = c.b;
        inst_.n = c.o;
        inst_.k = c.c;
    }
    emb1_ = lin(std::string(kM) + "mask_predictor.mask_embed.layers.0");
    emb2_ = lin(std::string(kM) + "mask_predictor.mask_embed.layers.1");
    emb3_ = lin(std::string(kM) + "mask_predictor.mask_embed.layers.2");
    {
        // the 1x level's sine positions (normalized: (i + 1) / (grid + 1e-6) * 2 pi), the rows' d / 2 channels then
        // the columns'
        const int64_t G = grid_, F = d_ / 2;
        std::vector<float> pos((size_t) (G * G * d_));
        std::vector<float> ty((size_t) F), tx((size_t) F);
        for (int64_t py = 0; py < G; ++py)
            for (int64_t px = 0; px < G; ++px) {
                const float vy = (float) (py + 1) / ((float) G + 1e-6f), vx = (float) (px + 1) / ((float) G + 1e-6f);
                sine(vy, F, ty.data());
                sine(vx, F, tx.data());
                float* o = pos.data() + (py * G + px) * d_;
                std::copy(ty.begin(), ty.end(), o);
                std::copy(tx.begin(), tx.end(), o + F);
            }
        auto* dp = static_cast<float*>(alloc(pos.size() * 4));
        gpu::copy(dp, pos.data(), pos.size() * 4);
        weight_bytes_ += pos.size() * 4;
        sine_pos_ = dp;
    }
}

uint64_t Sam3::need_bytes(int64_t width, int64_t height) const {
    const int64_t N = grid_ * grid_, P0 = 16 * N, P1 = 4 * N;
    // the ViT's buffers (its MLP's hidden rows the largest), the neck's levels, the pixel decoder's, the masks'
    const int64_t mlp = vit_.empty() ? 4736 : vit_[0].fc1.n;   // the ViT's MLP (4736 in SAM 3)
    uint64_t want = (uint64_t) N * (wd_ * 4 * 3 + mlp * 6 + wd_ * 2 * 6);
    want += (uint64_t) (P0 + P1 + N) * d_ * 4 + (uint64_t) P0 * (wd_ / 4) * 6 + (uint64_t) P1 * (wd_ / 2) * 6;
    want += (uint64_t) P0 * d_ * (4 + 2 + 2) + (uint64_t) kMaskChunk * (P0 * 4 + width * height);
    want += (uint64_t) heads_ * (queries_ + 1) * N * 4 + (uint64_t) image_ * image_ * 8 + (64ull << 20);
    const uint64_t have = held() - weight_bytes_;
    return (loaded_ ? 0 : file_bytes_ + file_bytes_ / 8) + (want > have ? want - have : 0);
}

void Sam3::attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int64_t lq, int64_t lk, int64_t heads,
                     int64_t hd, int64_t q_row, int64_t k_row, int64_t v_row, uint16_t* out, int64_t batch,
                     const float* bias, bool causal) {
    QiSdpa a;
    a.q = q;
    a.k = k;
    a.v = v;
    a.out = out;
    a.batch = batch;
    a.heads = heads;
    a.lq = lq;
    a.lk = lk;
    a.hd = hd;
    a.q_row = q_row;
    a.k_row = k_row;
    a.v_row = v_row;
    a.o_row = heads * hd;
    a.q_batch = lq * q_row;
    a.k_batch = lk * k_row;
    a.v_batch = lk * v_row;
    a.o_batch = lq * heads * hd;
    a.scale = 1.f / std::sqrt((float) hd);
    a.bias = bias;
    a.causal = causal;
    qi_sdpa_ex(a, s_);
}

void Sam3::vision(const QiRef& image) {
    const int64_t S = image_, G = grid_, N = G * G, C = wd_, hd = C / vit_heads_;
    // the picture at S x S (torchvision's antialiased bilinear: Pillow's triangle filter), normalized to [-1, 1]
    const QiRef r = pillow_resize(image, S, S, Filter::triangle);
    auto* rgba = scratch_as<uint8_t>(kRgba, r.rgba.size());
    gpu::copy(rgba, r.rgba.data(), r.rgba.size());
    auto* pix16 = scratch_as<uint16_t>(kPix, (size_t) (S * S * 3));
    static const float half[3] = {0.5f, 0.5f, 0.5f};
    aux_rgb_norm16(rgba, S, S, half, half, pix16, s_);
    float* x = scratch(kX, (size_t) (N * C * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (N * C));
    auto* w16 = scratch_as<uint16_t>(kW16, (size_t) (N * C));
    const int64_t mlp = vit_[0].fc1.n;
    float* big = scratch(kBig, (size_t) (N * std::max(mlp, 3 * C) * 4));
    auto* g16 = scratch_as<uint16_t>(kG16, (size_t) (N * mlp));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (N * C));
    auto* k16 = scratch_as<uint16_t>(kK16, (size_t) (N * C));
    auto* v16 = scratch_as<uint16_t>(kV16, (size_t) (N * C));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (N * C));
    float* y = scratch(kY, (size_t) (N * C * 4));
    // the patches, their tiled positions, the norm
    convolve(patch_conv_, pix16, S, S, y, kCol);
    aux_tile_add(y, pos_, G, G, C, window_, window_, s_);
    aux_layer_norm(y, nullptr, ln_pre_.w, ln_pre_.b, N, C, kVitEps, x, nullptr, s_);
    host("embed", x, (size_t) (N * C));
    const int64_t ws = window_, nwin = (G / ws) * (G / ws);
    if (G % ws != 0) throw std::runtime_error("image engine (SAM 3): the patches are not whole windows");
    for (size_t li = 0; li < vit_.size(); ++li) {
        const VitLayer& l = vit_[li];
        aux_layer_norm(x, nullptr, l.n1.w, l.n1.b, N, C, kVitEps, nullptr, h16, s_);
        const uint16_t* in16 = h16;
        if (!l.global) {
            aux_window16(h16, G, G, C, ws, 0, G, G, w16, s_);
            in16 = w16;
        }
        product(in16, l.qkv, big, N);
        // the window's positions, or every patch's scaled onto the window's grid
        if (l.global) aux_rope2d_split16(big, N, vit_heads_, hd, G, (float) ws / (float) G, kTheta, q16, k16, v16, s_);
        else aux_rope2d_split16(big, N, vit_heads_, hd, ws, 1.f, kTheta, q16, k16, v16, s_);
        if (l.global) attention(q16, k16, v16, N, N, vit_heads_, hd, C, C, C, a16);
        else attention(q16, k16, v16, ws * ws, ws * ws, vit_heads_, hd, C, C, C, a16, nwin);
        product(a16, l.o, y, N);
        if (l.global) qi_axpy(x, y, 1.f, N * C, s_);
        else aux_unwindow_add(x, y, G, G, C, ws, 0, G, G, s_);
        aux_layer_norm(x, nullptr, l.n2.w, l.n2.b, N, C, kVitEps, nullptr, h16, s_);
        product(h16, l.fc1, big, N);
        qi_gelu_erf16(big, N * l.fc1.n, g16, s_);
        product(g16, l.fc2, y, N);
        qi_axpy(x, y, 1.f, N * C, s_);
        if (li == 0 || li == 7 || li + 1 == vit_.size()) host(("vit" + std::to_string(li)).c_str(), x, (size_t) (N * C));
    }
    // the neck: 4x, 2x and 1x levels of d_ channels
    kernels::f32_to_f16_bulk(x, h16, N * C, s_);
    float* lv[3] = {scratch(kF0, (size_t) (16 * N * d_ * 4)), scratch(kF1, (size_t) (4 * N * d_ * 4)),
                    scratch(kF2, (size_t) (N * d_ * 4))};
    for (int i = 0; i < 3; ++i) {
        const Fpn& f = fpn_[(size_t) i];
        const uint16_t* in16 = h16;
        int64_t side = G;
        if (f.ups >= 1) {
            product(h16, Lin{f.up1.w, nullptr, f.up1.n, f.up1.k}, big, N);
            aux_deconv2_scatter(big, side, side, f.up1.n / 4, f.up1.b, f.ups == 2, nullptr, g16, s_);
            side *= 2;
            in16 = g16;
        }
        if (f.ups == 2) {
            auto* up16 = scratch_as<uint16_t>(kInst16, (size_t) (4 * side * side * (f.up2.n / 4)));
            product(g16, Lin{f.up2.w, nullptr, f.up2.n, f.up2.k}, big, side * side);
            aux_deconv2_scatter(big, side, side, f.up2.n / 4, f.up2.b, false, nullptr, up16, s_);
            side *= 2;
            in16 = up16;
        }
        const int64_t P = side * side;
        float* c = scratch(kNeck, (size_t) (P * d_ * 4));
        convolve(f.c1, in16, side, side, c, kCol);
        auto* c16 = scratch_as<uint16_t>(kA16, (size_t) (P * d_));
        kernels::f32_to_f16_bulk(c, c16, P * d_, s_);
        convolve(f.c3, c16, side, side, lv[i], kCol);
        host(("fpn" + std::to_string(i)).c_str(), lv[i], (size_t) (P * d_));
    }
}

void Sam3::text(const std::vector<int32_t>& ids) {
    const int64_t T = (int64_t) ids.size(), C = td_, hd = C / txt_heads_;
    if (T < 2 || T > 32) throw std::invalid_argument("image engine (SAM 3): the words are 2 to 32 tokens");
    for (int32_t id : ids)
        if (id < 0 || id >= vocab_) throw std::invalid_argument("image engine (SAM 3): a token outside the vocabulary");
    T_ = T;
    std::vector<int32_t> both((size_t) (2 * T));
    std::copy(ids.begin(), ids.end(), both.begin());
    std::iota(both.begin() + T, both.end(), 0);
    auto* dids = scratch_as<int32_t>(kSmall, (size_t) (2 * T));
    gpu::copy(dids, both.data(), both.size() * 4);
    float* x = scratch(kTxt, (size_t) (T * C * 4));
    aux_embed(dids, dids + T, tok_, tpos_, T, C, 1.f, x, s_);
    float* big = scratch(kBig, (size_t) (T * 4 * C * 4));
    float* y = scratch(kY, (size_t) (T * C * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (T * 4 * C));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (3 * T * C));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (T * C));
    for (const TextLayer& l : txt_) {
        aux_layer_norm(x, nullptr, l.n1.w, l.n1.b, T, C, kEps, nullptr, h16, s_);
        product(h16, l.qkv, big, T);
        aux_split3_16(big, T, C, q16, q16 + T * C, q16 + 2 * T * C, s_);
        attention(q16, q16 + T * C, q16 + 2 * T * C, T, T, txt_heads_, hd, C, C, C, a16, 1, nullptr, true);
        product(a16, l.o, y, T);
        qi_axpy(x, y, 1.f, T * C, s_);
        aux_layer_norm(x, nullptr, l.n2.w, l.n2.b, T, C, kEps, nullptr, h16, s_);
        product(h16, l.fc1, big, T);
        qi_gelu_erf16(big, T * l.fc1.n, h16, s_);
        product(h16, l.fc2, y, T);
        qi_axpy(x, y, 1.f, T * C, s_);
    }
    aux_layer_norm(x, nullptr, ln_final_.w, ln_final_.b, T, C, kEps, nullptr, h16, s_);
    float* t = scratch(kTxtOut, (size_t) (T * d_ * 4));
    product(h16, resizer_, t, T);
    auto* t16 = scratch_as<uint16_t>(kTxt16, (size_t) (T * d_));
    kernels::f32_to_f16_bulk(t, t16, T * d_, s_);
    host("text", t, (size_t) (T * d_));
}

std::vector<Segment> Sam3::run(const QiRef& image, const std::vector<int32_t>& ids, float score_threshold,
                               float mask_threshold, Times* tm) {
    Times local;
    Times& t = tm != nullptr ? *tm : local;
    load();
    auto t0 = std::chrono::steady_clock::now();
    vision(image);
    gpu::stream_sync(s_);
    t.vision = since(t0);
    t0 = std::chrono::steady_clock::now();
    text(ids);
    gpu::stream_sync(s_);
    t.text = since(t0);
    t0 = std::chrono::steady_clock::now();
    const int64_t G = grid_, N = G * G, d = d_, T = T_, hd = d / heads_, Q = queries_, Q1 = Q + 1;
    const int64_t mlp = enc_[0].fc1.n;
    const auto* txt16 = scratch_as<uint16_t>(kTxt16, (size_t) (T * d));
    float* x = scratch(kF2, (size_t) (N * d * 4));   // the 1x level, the encoder's stream in place
    float* h = scratch(kY, (size_t) (N * d * 4));
    float* big = scratch(kBig, (size_t) (N * std::max<int64_t>(mlp, 4 * d) * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (N * d));
    auto* hp16 = scratch_as<uint16_t>(kW16, (size_t) (N * d));
    auto* g16 = scratch_as<uint16_t>(kG16, (size_t) (N * mlp));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (N * d));
    auto* k16 = scratch_as<uint16_t>(kK16, (size_t) (N * d));
    auto* v16 = scratch_as<uint16_t>(kV16, (size_t) (N * d));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (N * d));
    auto* kt16 = scratch_as<uint16_t>(kSmall16, (size_t) (2 * T * d));
    auto* vt16 = kt16 + T * d;
    float* small = scratch(kTmp2, (size_t) (std::max<int64_t>(Q1, T) * std::max<int64_t>(mlp, d) * 4));
    // the words' keys and values of an attention, at T rows
    const auto word_kv = [&](const Attn& a) {
        product(txt16, a.k, small, T);
        kernels::f32_to_f16_bulk(small, kt16, T * d, s_);
        product(txt16, a.v, small, T);
        kernels::f32_to_f16_bulk(small, vt16, T * d, s_);
    };
    const auto to16 = [&](const uint16_t* in16, const Lin& l, int64_t rows, uint16_t* out16) {
        product(in16, l, big, rows);
        kernels::f32_to_f16_bulk(big, out16, rows * l.n, s_);
    };
    // the DETR encoder over the 1x level
    for (const EncLayer& l : enc_) {
        aux_layer_norm(x, nullptr, l.n1.w, l.n1.b, N, d, kEps, h, h16, s_);
        aux_add16(h, sine_pos_, N * d, hp16, s_);
        to16(hp16, l.self.q, N, q16);
        to16(hp16, l.self.k, N, k16);
        to16(h16, l.self.v, N, v16);
        attention(q16, k16, v16, N, N, heads_, hd, d, d, d, a16);
        product(a16, l.self.o, h, N);
        qi_axpy(x, h, 1.f, N * d, s_);
        aux_layer_norm(x, nullptr, l.n2.w, l.n2.b, N, d, kEps, nullptr, h16, s_);
        to16(h16, l.cross.q, N, q16);
        word_kv(l.cross);
        attention(q16, kt16, vt16, N, T, heads_, hd, d, d, d, a16);
        product(a16, l.cross.o, h, N);
        qi_axpy(x, h, 1.f, N * d, s_);
        aux_layer_norm(x, nullptr, l.n3.w, l.n3.b, N, d, kEps, nullptr, h16, s_);
        product(h16, l.fc1, big, N);
        aux_relu(big, N * l.fc1.n, nullptr, g16, s_);
        product(g16, l.fc2, h, N);
        qi_axpy(x, h, 1.f, N * d, s_);
    }
    host("encoder", x, (size_t) (N * d));
    // the decoder's keys' inputs: the patches with and without their positions
    auto* enc16 = scratch_as<uint16_t>(kEnc16, (size_t) (N * d));
    auto* encpos16 = scratch_as<uint16_t>(kEncPos16, (size_t) (N * d));
    kernels::f32_to_f16_bulk(x, enc16, N * d, s_);
    aux_add16(x, sine_pos_, N * d, encpos16, s_);
    // the DETR decoder: the presence token, then the queries
    float* hq = scratch(kDec, (size_t) (Q1 * d * 4));
    auto* hq16 = scratch_as<uint16_t>(kDec16, (size_t) (Q1 * d));
    float* qp = scratch(kQp, (size_t) (Q1 * d * 4));
    float* qo = scratch(kPix1, (size_t) (Q * d * 4));   // the queries after the output norm
    auto* qo16 = scratch_as<uint16_t>(kQo16, (size_t) (Q * d));
    float* bias = scratch(kBias, (size_t) (heads_ * Q1 * N * 4));
    float* tmp = scratch(kTmp, (size_t) (Q * 2 * G * heads_ * 4));
    auto* sin16 = scratch_as<uint16_t>(kSmall, (size_t) (Q * 2 * d + Q * 4 * 2));
    auto* boxes_dev = reinterpret_cast<float*>(sin16 + Q * 2 * d);
    {
        std::vector<float> init((size_t) (Q1 * d));
        std::copy(presence_tok_.begin(), presence_tok_.begin() + d, init.begin());
        std::copy(query_embed_.begin(), query_embed_.begin() + Q * d, init.begin() + d);
        gpu::copy(hq, init.data(), init.size() * 4);
        gpu::memset(qp, 0, (size_t) (d * 4));
    }
    std::vector<float> ref((size_t) (Q * 4));
    for (int64_t i = 0; i < Q * 4; ++i) ref[(size_t) i] = sigmoid(ref_points_[(size_t) i]);
    std::vector<float> delta((size_t) (Q * 4));
    float presence = 0.f;
    // the words' keys and values for the scoring's and the masks' attention are computed where used; the decoder's
    // per layer (each its own projections)
    for (size_t li = 0; li < dec_.size(); ++li) {
        const DecLayer& l = dec_[li];
        {
            // the query positions from the boxes (their sines: y, x, w, h) and the box bias
            std::vector<float> sn((size_t) (Q * 2 * d)), xyxy((size_t) (Q * 4));
            for (int64_t q = 0; q < Q; ++q) {
                const float* b = ref.data() + q * 4;
                float* o = sn.data() + q * 2 * d;
                sine(b[1], d / 2, o);
                sine(b[0], d / 2, o + d / 2);
                sine(b[2], d / 2, o + d);
                sine(b[3], d / 2, o + 3 * d / 2);
                xyxy[(size_t) (q * 4)] = b[0] - 0.5f * b[2];
                xyxy[(size_t) (q * 4 + 1)] = b[1] - 0.5f * b[3];
                xyxy[(size_t) (q * 4 + 2)] = b[0] + 0.5f * b[2];
                xyxy[(size_t) (q * 4 + 3)] = b[1] + 0.5f * b[3];
            }
            const auto s16 = f16_of(sn.data(), sn.size());
            gpu::stream_sync(s_);
            gpu::copy(sin16, s16.data(), s16.size() * 2);
            gpu::copy(boxes_dev, xyxy.data(), xyxy.size() * 4);
            product(sin16, ref1_, small, Q);
            aux_relu(small, Q * d, nullptr, a16, s_);
            product(a16, ref2_, qp + d, Q);
            aux_box_bias(boxes_dev, Q, G, G, rpb_, rpb_hidden_, heads_, tmp, bias, s_);
        }
        // self-attention (q and k with the positions), post-norm
        kernels::f32_to_f16_bulk(hq, hq16, Q1 * d, s_);
        aux_add16(hq, qp, Q1 * d, hp16, s_);
        to16(hp16, l.self.q, Q1, q16);
        to16(hp16, l.self.k, Q1, k16);
        to16(hq16, l.self.v, Q1, v16);
        attention(q16, k16, v16, Q1, Q1, heads_, hd, d, d, d, a16);
        product(a16, l.self.o, small, Q1);
        aux_layer_norm(hq, small, l.n_self.w, l.n_self.b, Q1, d, kEps, hq, nullptr, s_);
        // the words
        aux_add16(hq, qp, Q1 * d, hp16, s_);
        to16(hp16, l.text.q, Q1, q16);
        word_kv(l.text);
        attention(q16, kt16, vt16, Q1, T, heads_, hd, d, d, d, a16);
        product(a16, l.text.o, small, Q1);
        aux_layer_norm(hq, small, l.n_text.w, l.n_text.b, Q1, d, kEps, hq, nullptr, s_);
        // the patches, with the box bias
        aux_add16(hq, qp, Q1 * d, hp16, s_);
        to16(hp16, l.vis.q, Q1, q16);
        to16(encpos16, l.vis.k, N, k16);
        to16(enc16, l.vis.v, N, v16);
        attention(q16, k16, v16, Q1, N, heads_, hd, d, d, d, a16, 1, bias);
        product(a16, l.vis.o, small, Q1);
        aux_layer_norm(hq, small, l.n_vis.w, l.n_vis.b, Q1, d, kEps, hq, hq16, s_);
        // the MLP
        product(hq16, l.fc1, big, Q1);
        aux_relu(big, Q1 * l.fc1.n, nullptr, g16, s_);
        product(g16, l.fc2, small, Q1);
        aux_layer_norm(hq, small, l.n_mlp.w, l.n_mlp.b, Q1, d, kEps, hq, nullptr, s_);
        // the queries' boxes refined
        aux_layer_norm(hq + d, nullptr, dec_out_.w, dec_out_.b, Q, d, kEps, qo, qo16, s_);
        product(qo16, box1_, small, Q);
        aux_relu(small, Q * d, nullptr, a16, s_);
        product(a16, box2_, small, Q);
        aux_relu(small, Q * d, nullptr, a16, s_);
        product(a16, box3_, small, Q);
        gpu::stream_sync(s_);
        gpu::copy(delta.data(), small, delta.size() * 4);
        for (int64_t i = 0; i < Q * 4; ++i)
            ref[(size_t) i] = sigmoid(delta[(size_t) i] + inverse_sigmoid(ref[(size_t) i]));
        if (li + 1 == dec_.size()) {
            aux_layer_norm(hq, nullptr, pres_norm_.w, pres_norm_.b, 1, d, kEps, nullptr, a16, s_);
            product(a16, pres1_, small, 1);
            aux_relu(small, d, nullptr, a16, s_);
            product(a16, pres2_, small, 1);
            aux_relu(small, d, nullptr, a16, s_);
            product(a16, pres3_, small, 1);
            gpu::stream_sync(s_);
            gpu::copy(&presence, small, 4);
            presence = std::clamp(presence, -10.f, 10.f);
        }
    }
    host("queries", qo, (size_t) (Q * d));
    // the scores: each query against the words' pooled projection
    std::vector<float> score((size_t) Q);
    {
        const float* txt = scratch(kTxtOut, (size_t) (T * d * 4));
        product(txt16, sc_fc1_, small, T);
        aux_relu(small, T * sc_fc1_.n, nullptr, g16, s_);
        product(g16, sc_fc2_, h, T);
        aux_layer_norm(txt, h, sc_norm_.w, sc_norm_.b, T, d, kEps, h, nullptr, s_);
        std::vector<float> tt((size_t) (T * d)), pooled((size_t) d, 0.f);
        gpu::stream_sync(s_);
        gpu::copy(tt.data(), h, tt.size() * 4);
        for (int64_t i = 0; i < T; ++i)
            for (int64_t c = 0; c < d; ++c) pooled[(size_t) c] += tt[(size_t) (i * d + c)] / (float) T;
        const auto p16 = f16_of(pooled.data(), pooled.size());
        gpu::copy(a16, p16.data(), p16.size() * 2);
        product(a16, sc_proj_, small, 1);
        product(qo16, sc_qproj_, h, Q);
        std::vector<float> tp((size_t) d), qq((size_t) (Q * d));
        gpu::stream_sync(s_);
        gpu::copy(tp.data(), small, tp.size() * 4);
        gpu::copy(qq.data(), h, qq.size() * 4);
        const float scale = 1.f / std::sqrt((float) d);
        for (int64_t q = 0; q < Q; ++q) {
            float s = 0.f;
            for (int64_t c = 0; c < d; ++c) s += qq[(size_t) (q * d + c)] * tp[(size_t) c];
            score[(size_t) q] = std::clamp(s * scale, -12.f, 12.f);
        }
    }
    std::vector<int64_t> keep;
    std::vector<float> prob((size_t) Q);
    for (int64_t q = 0; q < Q; ++q) {
        prob[(size_t) q] = sigmoid(score[(size_t) q]) * sigmoid(presence);
        if (prob[(size_t) q] > score_threshold) keep.push_back(q);
    }
    std::stable_sort(keep.begin(), keep.end(), [&](int64_t a, int64_t b) { return prob[(size_t) a] > prob[(size_t) b]; });
    if (dump) {
        std::vector<float> v(score.begin(), score.end());
        v.push_back(presence);
        dump("scores", v);
        std::vector<float> b(ref.begin(), ref.end());
        dump("boxes", b);
    }
    t.detect = since(t0);
    t0 = std::chrono::steady_clock::now();
    std::vector<Segment> out;
    const int64_t W = image.width, H = image.height;
    for (int64_t q : keep) {
        const float* b = ref.data() + q * 4;
        Segment sg;
        sg.x0 = (b[0] - 0.5f * b[2]) * (float) W;
        sg.y0 = (b[1] - 0.5f * b[3]) * (float) H;
        sg.x1 = (b[0] + 0.5f * b[2]) * (float) W;
        sg.y1 = (b[1] + 0.5f * b[3]) * (float) H;
        sg.score = prob[(size_t) q];
        out.push_back(std::move(sg));
    }
    if (!keep.empty()) {
        // the encoder's patches after the words' cross-attention
        aux_layer_norm(x, nullptr, m_norm_.w, m_norm_.b, N, d, kEps, nullptr, h16, s_);
        to16(h16, m_cross_.q, N, q16);
        word_kv(m_cross_);
        attention(q16, kt16, vt16, N, T, heads_, hd, d, d, d, a16);
        product(a16, m_cross_.o, h, N);
        qi_axpy(x, h, 1.f, N * d, s_);
        // the pixel decoder up to the 4x level
        float* f1 = scratch(kF1, (size_t) (4 * N * d * 4));
        float* f0 = scratch(kF0, (size_t) (16 * N * d * 4));
        float* c1 = scratch(kPix0, (size_t) (4 * N * d * 4));
        auto* t16 = scratch_as<uint16_t>(kInst16, (size_t) (16 * N * d));
        float* part = scratch(kPart, (size_t) (3 * 8 * 64 * 4));
        aux_up_add16(x, G, G, f1, 2 * G, 2 * G, d, t16, s_);
        convolve(pix_conv_[0], t16, 2 * G, 2 * G, c1, kCol);
        aux_group_norm(c1, 4 * N, d, 8, pix_gn_[0].w, pix_gn_[0].b, kEps, true, c1, nullptr, part, s_);
        aux_up_add16(c1, 2 * G, 2 * G, f0, 4 * G, 4 * G, d, t16, s_);
        float* c0 = scratch(kMasks, (size_t) (16 * N * d * 4));
        convolve(pix_conv_[1], t16, 4 * G, 4 * G, c0, kCol);
        aux_group_norm(c0, 16 * N, d, 8, pix_gn_[1].w, pix_gn_[1].b, kEps, true, nullptr, t16, part, s_);
        // the instance projection (1 x 1), FP16 for the masks' product
        product(t16, inst_, c0, 16 * N);
        kernels::f32_to_f16_bulk(c0, t16, 16 * N * d, s_);
        // the kept queries' mask embeddings by the projection, a chunk at a time
        std::vector<float> qh((size_t) (Q * d));
        gpu::stream_sync(s_);
        gpu::copy(qh.data(), qo, qh.size() * 4);
        float* logits = c0;   // free again: [chunk][16N]
        auto* m8 = scratch_as<uint8_t>(kMasks8, (size_t) (kMaskChunk * W * H));
        for (size_t k0 = 0; k0 < keep.size(); k0 += (size_t) kMaskChunk) {
            const int64_t k = std::min<int64_t>(kMaskChunk, (int64_t) (keep.size() - k0));
            std::vector<float> rows((size_t) (k * d));
            for (int64_t i = 0; i < k; ++i)
                std::copy(qh.begin() + keep[k0 + (size_t) i] * d, qh.begin() + (keep[k0 + (size_t) i] + 1) * d,
                          rows.begin() + i * d);
            const auto r16 = f16_of(rows.data(), rows.size());
            gpu::stream_sync(s_);
            gpu::copy(a16, r16.data(), r16.size() * 2);
            product(a16, emb1_, small, k);
            aux_relu(small, k * d, nullptr, a16, s_);
            product(a16, emb2_, small, k);
            aux_relu(small, k * d, nullptr, a16, s_);
            product(a16, emb3_, small, k);
            kernels::f32_to_f16_bulk(small, a16, k * d, s_);
            gemm_.f16(a16, t16, logits, k, 16 * N, d);
            if (k0 == 0) host("mask_logits", logits, (size_t) (16 * N));
            aux_mask_resize(logits, k, 4 * G, 4 * G, H, W, mask_threshold, m8, s_);
            gpu::stream_sync(s_);
            for (int64_t i = 0; i < k; ++i) {
                auto& m = out[k0 + (size_t) i].mask;
                m.resize((size_t) (W * H));
                gpu::copy(m.data(), m8 + i * W * H, m.size());
            }
        }
    }
    t.masks = since(t0);
    return out;
}

}  // namespace strata::image
