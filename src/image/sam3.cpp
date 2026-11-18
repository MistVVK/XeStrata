// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/image/sam3.cpp - see include/strata/image/sam3.hpp.  The order of the operations, the layers' wiring and the
// constants follow vLLM-Omni's sam3_model.py (15bb15d, Apache-2.0); the tensor names are the native checkpoint's.

#include "strata/image/sam3.hpp"

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
    kF0, kF1, kF2, kPos, kTxt, kTxt16, kEnc, kEnc16, kEncPos16, kDec, kDec16, kQp, kTmp, kGq, kGk, kGv, kGs, kGp, kGo,
    kPix0, kPix1,
    kInst16, kMasks, kMasks8, kPart, kSmall, kSmall16, kTxtOut, kTmp2, kQo16, kNeck, kProj16, kRef, kTrunk16, kF2Keep,
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
    track_loaded_ = false;   // a load after a release: the tracker's weights went with the rest
    text_ids_.clear();
    text_cache_.clear();
    fpn_ = neck_of(std::string(kV) + "convs.");
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

std::vector<Sam3::Fpn> Sam3::neck_of(const std::string& prefix) {
    std::vector<Fpn> set;
    for (int i = 0; i < 3; ++i) {
        const std::string b = prefix + std::to_string(i) + ".";
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
        set.push_back(f);
    }
    if (set[0].ups != 2 || set[1].ups != 1 || set[2].ups != 0)
        throw std::runtime_error(path_ + ": the neck " + prefix + " is not 4x, 2x and 1x");
    return set;
}

uint64_t Sam3::need_bytes(int64_t width, int64_t height) const {
    const int64_t N = grid_ * grid_, P0 = 16 * N, P1 = 4 * N;
    // the ViT's buffers (its MLP's hidden rows the largest), the neck's levels, the pixel decoder's, the masks'
    const int64_t mlp = vit_.empty() ? 4736 : vit_[0].fc1.n;   // the ViT's MLP (4736 in SAM 3)
    uint64_t want = (uint64_t) N * (wd_ * 4 * 3 + mlp * 6 + wd_ * 2 * 6);
    want += (uint64_t) (P0 + P1 + N) * d_ * 4 + (uint64_t) P0 * (wd_ / 4) * 6 + (uint64_t) P1 * (wd_ / 2) * 6;
    want += (uint64_t) P0 * d_ * (4 + 2 + 2) + (uint64_t) kMaskChunk * (P0 * 4 + width * height);
    // the decoder's attention over the patches: the heads' keys and values apart, their scores and weights
    want += (uint64_t) N * d_ * 4 + (uint64_t) heads_ * (queries_ + 1) * N * 6;
    want += (uint64_t) image_ * image_ * 8 + (64ull << 20);
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

// The decoder's queries over the patches with the box bias (a row's term and a column's of the grid, the presence
// token's row none), through the shared matrix-engine attention (aux_attn_matrix): each head's scores through the
// Gemm, their softmax with the bias, by the values.  The scores of 201 rows over 5,184 keys are few enough to hold
// (the B70: 8 ms a frame in qi_sdpa_ex's old FMA kernels for a biased attention, 2026-11-07)
void Sam3::grid_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t lq, const float* grid,
                          uint16_t* out) {
    const int64_t G = grid_, N = G * G, d = d_, H = heads_, hd = d / H;
    auto* qh = scratch_as<uint16_t>(kGq, (size_t) (lq * d));
    auto* kh = scratch_as<uint16_t>(kGk, (size_t) (N * d));
    auto* vt = scratch_as<uint16_t>(kGv, (size_t) (N * d));
    float* sc = scratch(kGs, (size_t) (H * lq * N * 4));
    auto* p16 = scratch_as<uint16_t>(kGp, (size_t) (H * lq * N));
    float* o = scratch(kGo, (size_t) (lq * d * 4));
    aux_heads16(q16, lq, H, hd, d, false, qh, s_);
    aux_heads16(k16, N, H, hd, d, false, kh, s_);
    aux_heads16(v16, N, H, hd, d, true, vt, s_);
    aux_attn_matrix(gemm_, qh, kh, vt, 1, H, lq, N, hd, lq, hd, false, 1.f / std::sqrt((float) hd), nullptr, nullptr,
                    1, grid, G, 1, sc, p16, o, s_);
    aux_heads_merge16(o, lq, H, hd, out, s_);
}

void Sam3::vision(const QiRef& image) {
    trunk(image);
    detect_neck();
}

void Sam3::detect_neck(bool keep) {
    const size_t b2 = (size_t) (grid_ * grid_ * d_ * 4);
    float* lv[3] = {scratch(kF0, (size_t) (16 * grid_ * grid_ * d_ * 4)),
                    scratch(kF1, (size_t) (4 * grid_ * grid_ * d_ * 4)), scratch(kF2, b2)};
    neck(fpn_, lv);
    if (keep) gpu::copy_async(scratch(kF2Keep, b2), lv[2], b2, s_);
}

void Sam3::detect_restore() {
    const size_t b2 = (size_t) (grid_ * grid_ * d_ * 4);
    gpu::copy_async(scratch(kF2, b2), scratch(kF2Keep, b2), b2, s_);
}

void Sam3::trunk(const QiRef& image, Filter filter) {
    const int64_t S = image_;
    // the picture at S x S (a picture's: torchvision's antialiased bilinear, Pillow's triangle filter; a video's
    // frames: OpenCV's cubic), normalized to [-1, 1]
    const QiRef r = pillow_resize(image, S, S, filter);
    auto* rgba = scratch_as<uint8_t>(kRgba, r.rgba.size());
    gpu::copy(rgba, r.rgba.data(), r.rgba.size());
    auto* pix16 = scratch_as<uint16_t>(kPix, (size_t) (S * S * 3));
    static const float half[3] = {0.5f, 0.5f, 0.5f};
    aux_rgb_norm16(rgba, S, S, half, half, pix16, s_);
    trunk_from(pix16);
}

void Sam3::trunk_pixels(const float* chw) {
    const int64_t S = image_;
    std::vector<uint16_t> hwc((size_t) (S * S * 3));
    for (int64_t c = 0; c < 3; ++c)
        for (int64_t p = 0; p < S * S; ++p) hwc[(size_t) (p * 3 + c)] = kernels::f16_from_f32(chw[c * S * S + p]);
    auto* pix16 = scratch_as<uint16_t>(kPix, hwc.size());
    gpu::copy(pix16, hwc.data(), hwc.size() * 2);
    trunk_from(pix16);
}

void Sam3::trunk_from(const uint16_t* pix16) {
    const int64_t S = image_, G = grid_, N = G * G, C = wd_, hd = C / vit_heads_;
    float* x = scratch(kX, (size_t) (N * C * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (N * C));
    auto* w16 = scratch_as<uint16_t>(kW16, (size_t) (N * C));
    const int64_t mlp = vit_[0].fc1.n;
    float* big = scratch(kBig, (size_t) (N * std::max(mlp, 3 * C) * 4));
    auto* big16 = reinterpret_cast<uint16_t*>(big);   // product16's rows (qkv, the MLP's first): half big's bytes
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
    // layer 0's first norm here; each layer's residual adds then take the next norm with them (aux_residual_norm16):
    // the projections' bias, the residual and the norm in one pass, the windows' order where the layer has windows
    const auto win = [&](const VitLayer& l) { return l.global ? (int64_t) 0 : ws; };
    if (vit_[0].global) aux_layer_norm(x, nullptr, vit_[0].n1.w, vit_[0].n1.b, N, C, kVitEps, nullptr, h16, s_);
    else aux_layer_norm_window16(x, vit_[0].n1.w, vit_[0].n1.b, G, G, C, ws, kVitEps, w16, s_);
    for (size_t li = 0; li < vit_.size(); ++li) {
        const VitLayer& l = vit_[li];
        product16(l.global ? h16 : w16, l.qkv, big16, N);
        // the window's positions, or every patch's scaled onto the window's grid; the bias added as they are read
        if (l.global)
            aux_rope2d_split16h(big16, l.qkv.b, N, vit_heads_, hd, G, (float) ws / (float) G, kTheta, q16, k16, v16, s_);
        else aux_rope2d_split16h(big16, l.qkv.b, N, vit_heads_, hd, ws, 1.f, kTheta, q16, k16, v16, s_);
        if (l.global) attention(q16, k16, v16, N, N, vit_heads_, hd, C, C, C, a16);
        else attention(q16, k16, v16, ws * ws, ws * ws, vit_heads_, hd, C, C, C, a16, nwin);
        product(a16, l.o, y, N, 0, false);
        aux_residual_norm16(x, y, l.o.b, l.n2.w, l.n2.b, G, G, C, win(l), 0, kVitEps, h16, s_);
        product16(h16, l.fc1, g16, N, true, ImageAct::gelu);
        product(g16, l.fc2, y, N, 0, false);
        const VitLayer* next = li + 1 < vit_.size() ? &vit_[li + 1] : nullptr;
        aux_residual_norm16(x, y, l.fc2.b, next ? next->n1.w : nullptr, next ? next->n1.b : nullptr, G, G, C, 0,
                            next ? win(*next) : 0, kVitEps, next && !next->global ? w16 : h16, s_);
        if (li == 0 || li == 7 || li + 1 == vit_.size()) host(("vit" + std::to_string(li)).c_str(), x, (size_t) (N * C));
    }
    // the trunk as FP16, which every neck reads
    auto* t16 = scratch_as<uint16_t>(kTrunk16, (size_t) (N * C));
    kernels::f32_to_f16_bulk(x, t16, N * C, s_);
}

void Sam3::neck(const std::vector<Fpn>& set, float* const lv[3]) {
    const int64_t G = grid_, N = G * G, C = wd_;
    const auto* h16 = scratch_as<uint16_t>(kTrunk16, (size_t) (N * C));
    float* big = scratch(kBig, (size_t) (N * C * 4 * 4));
    auto* g16 = scratch_as<uint16_t>(kG16, (size_t) (4 * N * C / 2));
    for (int i = 0; i < 3; ++i) {
        if (lv[i] == nullptr) continue;
        const Fpn& f = set[(size_t) i];
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

void Sam3::drop_scratch() {
    AuxNet::drop_scratch();
    text_ids_.clear();
}

void Sam3::text(const std::vector<int32_t>& ids) {
    if (ids == text_ids_) return;   // the same words as the last call's: their states are still there (a clip's)
    const auto hit = text_cache_.find(ids);
    if (hit != text_cache_.end()) {   // a clip's other words: their states as they were made
        T_ = (int64_t) ids.size();
        gpu::copy(scratch(kTxtOut, (size_t) (T_ * d_ * 4)), hit->second.data(), (size_t) (T_ * d_) * 4);
        gpu::copy(scratch_as<uint16_t>(kTxt16, (size_t) (T_ * d_)), hit->second.data() + T_ * d_ * 4,
                  (size_t) (T_ * d_) * 2);
        text_ids_ = ids;
        return;
    }
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
    text_ids_ = ids;
    if (text_cache_.size() >= kTextCache) text_cache_.clear();
    std::vector<uint8_t>& keep = text_cache_[ids];
    keep.resize((size_t) (T * d_) * 6);
    gpu::stream_sync(s_);
    gpu::copy(keep.data(), t, (size_t) (T * d_) * 4);
    gpu::copy(keep.data() + T * d_ * 4, t16, (size_t) (T * d_) * 2);
}

Sam3::Detection Sam3::detect(float score_threshold) {
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
        product16(in16, l, out16, rows, true);
    };
    // the DETR encoder over the 1x level
    for (const EncLayer& l : enc_) {
        aux_layer_norm(x, nullptr, l.n1.w, l.n1.b, N, d, kEps, h, h16, s_);
        aux_add16(h, sine_pos_, N * d, hp16, s_);
        to16(hp16, l.self.q, N, q16);
        to16(hp16, l.self.k, N, k16);
        to16(h16, l.self.v, N, v16);
        attention(q16, k16, v16, N, N, heads_, hd, d, d, d, a16);
        product(a16, l.self.o, h, N, 0, false);
        aux_residual_norm16(x, h, l.self.o.b, l.n2.w, l.n2.b, G, G, d, 0, 0, kEps, h16, s_);
        to16(h16, l.cross.q, N, q16);
        word_kv(l.cross);
        attention(q16, kt16, vt16, N, T, heads_, hd, d, d, d, a16);
        product(a16, l.cross.o, h, N, 0, false);
        aux_residual_norm16(x, h, l.cross.o.b, l.n3.w, l.n3.b, G, G, d, 0, 0, kEps, h16, s_);
        product16(h16, l.fc1, g16, N, true, ImageAct::relu);
        product(g16, l.fc2, h, N, 0, false);
        aux_residual_norm16(x, h, l.fc2.b, nullptr, nullptr, G, G, d, 0, 0, kEps, nullptr, s_);
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
    float presence = 0.f;
    // the words' keys and values for the scoring's and the masks' attention are computed where used; the decoder's
    // per layer (each its own projections)
    // the boxes stay on the GPU through the layers (their sines and corners, and their refinement, its kernels): the
    // host reads them once after the last
    float* ref_dev = scratch(kRef, (size_t) (Q * 4 * 4));
    gpu::copy(ref_dev, ref.data(), ref.size() * 4);
    for (size_t li = 0; li < dec_.size(); ++li) {
        const DecLayer& l = dec_[li];
        {
            // the query positions from the boxes (their sines: y, x, w, h) and the box bias
            aux_box_sine16(ref_dev, Q, d / 2, sin16, boxes_dev, s_);
            product(sin16, ref1_, small, Q);
            aux_relu(small, Q * d, nullptr, a16, s_);
            product(a16, ref2_, qp + d, Q);
            aux_box_bias(boxes_dev, Q, G, G, rpb_, rpb_hidden_, heads_, tmp, nullptr, s_);
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
        grid_attention(q16, k16, v16, Q1, tmp, a16);
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
        aux_box_refine(ref_dev, small, Q * 4, s_);
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
    gpu::stream_sync(s_);
    gpu::copy(ref.data(), ref_dev, ref.size() * 4);
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
    Detection r;
    r.keep = std::move(keep);
    r.prob = std::move(prob);
    r.box = std::move(ref);
    return r;
}

void Sam3::mask_prep() {
    const int64_t G = grid_, N = G * G, d = d_, T = T_, hd = d / heads_, Q1 = queries_ + 1;
    const int64_t mlp = enc_[0].fc1.n;
    const auto* txt16 = scratch_as<uint16_t>(kTxt16, (size_t) (T * d));
    float* x = scratch(kF2, (size_t) (N * d * 4));   // the 1x level, the encoder's stream in place
    float* h = scratch(kY, (size_t) (N * d * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (N * d));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (N * d));
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
        product16(in16, l, out16, rows, true);
    };
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
    float* part = scratch(kPart, (size_t) qi_group_norm_ws_floats(16 * N, 8) * sizeof(float));
    aux_up_add16(x, G, G, f1, 2 * G, 2 * G, d, t16, s_);
    convolve(pix_conv_[0], t16, 2 * G, 2 * G, c1, kCol);
    image_group_norm(c1, nullptr, 4 * N, d, 8, pix_gn_[0].w, pix_gn_[0].b, kEps, ImageAct::relu, c1, nullptr, part, s_);
    aux_up_add16(c1, 2 * G, 2 * G, f0, 4 * G, 4 * G, d, t16, s_);
    float* c0 = scratch(kMasks, (size_t) (16 * N * d * 4));
    convolve(pix_conv_[1], t16, 4 * G, 4 * G, c0, kCol);
    image_group_norm(c0, nullptr, 16 * N, d, 8, pix_gn_[1].w, pix_gn_[1].b, kEps, ImageAct::relu, nullptr, t16, part, s_);
    // the instance projection (1 x 1), FP16 for the masks' product
    auto* p16 = scratch_as<uint16_t>(kProj16, (size_t) (16 * N * d));
    product16(t16, inst_, p16, 16 * N, true);
}

void Sam3::mask_logits(const int64_t* qs, int64_t k, float* logits) {
    const int64_t G = grid_, N = G * G, d = d_, Q = queries_, mlp = enc_[0].fc1.n;
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (N * d));
    float* small = scratch(kTmp2, (size_t) (std::max<int64_t>(Q + 1, T_) * std::max<int64_t>(mlp, d) * 4));
    const float* qo = scratch(kPix1, (size_t) (Q * d * 4));
    const auto* p16 = scratch_as<uint16_t>(kProj16, (size_t) (16 * N * d));
    std::vector<float> qh((size_t) (Q * d));
    gpu::stream_sync(s_);
    gpu::copy(qh.data(), qo, qh.size() * 4);
    for (int64_t k0 = 0; k0 < k; k0 += kMaskChunk) {
        const int64_t n = std::min<int64_t>(kMaskChunk, k - k0);
        std::vector<float> rows((size_t) (n * d));
        for (int64_t i = 0; i < n; ++i)
            std::copy(qh.begin() + qs[k0 + i] * d, qh.begin() + (qs[k0 + i] + 1) * d, rows.begin() + i * d);
        const auto r16 = f16_of(rows.data(), rows.size());
        gpu::stream_sync(s_);
        gpu::copy(a16, r16.data(), r16.size() * 2);
        product(a16, emb1_, small, n);
        aux_relu(small, n * d, nullptr, a16, s_);
        product(a16, emb2_, small, n);
        aux_relu(small, n * d, nullptr, a16, s_);
        product(a16, emb3_, small, n);
        kernels::f32_to_f16_bulk(small, a16, n * d, s_);
        gemm_.f16(a16, p16, logits + k0 * 16 * N, n, 16 * N, d);
    }
}

std::vector<Segment> Sam3::segments(const Detection& det, int64_t W, int64_t H, float mask_threshold) {
    std::vector<Segment> out;
    const int64_t G = grid_, N = G * G;
    for (int64_t q : det.keep) {
        const float* b = det.box.data() + q * 4;
        Segment sg;
        sg.x0 = (b[0] - 0.5f * b[2]) * (float) W;
        sg.y0 = (b[1] - 0.5f * b[3]) * (float) H;
        sg.x1 = (b[0] + 0.5f * b[2]) * (float) W;
        sg.y1 = (b[1] + 0.5f * b[3]) * (float) H;
        sg.score = det.prob[(size_t) q];
        out.push_back(std::move(sg));
    }
    if (det.keep.empty()) return out;
    mask_prep();
    // the kept queries' masks, a chunk at a time
    float* logits = scratch(kMasks, (size_t) (16 * N * d_ * 4));   // free again: [chunk][16N]
    auto* m8 = scratch_as<uint8_t>(kMasks8, (size_t) (kMaskChunk * W * H));
    for (size_t k0 = 0; k0 < det.keep.size(); k0 += (size_t) kMaskChunk) {
        const int64_t k = std::min<int64_t>(kMaskChunk, (int64_t) (det.keep.size() - k0));
        mask_logits(det.keep.data() + k0, k, logits);
        if (k0 == 0) host("mask_logits", logits, (size_t) (16 * N));
        aux_mask_resize(logits, k, 4 * G, 4 * G, H, W, false, mask_threshold, m8, s_);
        gpu::stream_sync(s_);
        for (int64_t i = 0; i < k; ++i) {
            auto& m = out[k0 + (size_t) i].mask;
            m.resize((size_t) (W * H));
            gpu::copy(m.data(), m8 + i * W * H, m.size());
        }
    }
    return out;
}

std::vector<std::vector<Segment>> Sam3::run_many(const QiRef& image, const std::vector<std::vector<int32_t>>& ids,
                                                 float score_threshold, float mask_threshold) {
    load();
    trunk(image);
    std::vector<std::vector<Segment>> out;
    for (size_t i = 0; i < ids.size(); ++i) {
        // the detector's encoder runs in place on the 1x level: kept from the first words for the next ones
        if (i == 0) detect_neck(ids.size() > 1);
        else detect_restore();
        text(ids[i]);
        out.push_back(segments(detect(score_threshold), image.width, image.height, mask_threshold));
    }
    return out;
}

std::vector<Segment> Sam3::run(const QiRef& image, const std::vector<int32_t>& ids, float score_threshold,
                               float mask_threshold, Times* tm) {
    Times local;
    Times& t = tm != nullptr ? *tm : local;
    load();
    // the share done (progress) at each part's end, the parts weighed by the last run's seconds of each (before
    // there is one: 6, 1, 2 and 1 tenths)
    const double last = last_.vision + last_.text + last_.detect + last_.masks;
    const double wv = last > 0 ? last_.vision / last : 0.6, wt = last > 0 ? last_.text / last : 0.1;
    const double wd = last > 0 ? last_.detect / last : 0.2;
    if (progress) progress(0.0);
    auto t0 = std::chrono::steady_clock::now();
    vision(image);
    gpu::stream_sync(s_);
    t.vision = since(t0);
    if (progress) progress(wv);
    t0 = std::chrono::steady_clock::now();
    text(ids);
    gpu::stream_sync(s_);
    t.text = since(t0);
    if (progress) progress(wv + wt);
    t0 = std::chrono::steady_clock::now();
    const Detection det = detect(score_threshold);
    t.detect = since(t0);
    if (progress) progress(wv + wt + wd);
    t0 = std::chrono::steady_clock::now();
    std::vector<Segment> out = segments(det, image.width, image.height, mask_threshold);
    t.masks = since(t0);
    last_ = t;
    if (progress) progress(1.0);
    return out;
}

}  // namespace strata::image
