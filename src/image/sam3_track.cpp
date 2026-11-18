// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/sam3_track.cpp - SAM 3.1's multiplex tracker's parts (sam3.hpp's video section), written for the engine:
// the images normalized to [-1, 1], the sine positions at (i + 1) / (n + 1e-6), the object pointer from the mask
// token of the best mask, the CXBlocks' norm at 1e-6.
//
//   * the necks: the trunk's patches through the propagation's (and the interactive's) 4x, 2x and 1x convolutions;
//   * the memory attention: 4 layers over the frame's 1x level (+ 0.1 its sine positions): a self-attention whose q
//     and k turn by the 2D axial RoPE of the 72 x 72 grid; a decoupled cross-attention whose query is the frame's
//     image and its stream projected and summed, whose keys are the memories' frames' images and memories projected
//     and summed (plus those frames' sine and time positions, turned by the RoPE; the object pointers' rows not),
//     whose values are the memories'; a GELU MLP; 8 heads, a final norm;
//   * the decoders: SAM's two-way transformer (2 layers: tokens' self-attention, tokens to image, an MLP, image to
//     tokens; a last tokens to image) over the image's 1x level at a random Fourier position table, then the masks
//     from the 1x level upscaled twice (transposed 2 x 2 convolutions, the 2x and 4x levels added, LayerNorm2d,
//     GELU) by each mask token's hypernetwork MLP; the multiplex decoder holds 16 objects of 3 masks a bucket, the
//     interactive one an object of 4 (its prompt: a mask through a 4 x 4 downsample and the prompt encoder, or clicks:
//     each point's random Fourier position with its label's embedding, a box as its two corners);
//   * the memory encoder: a bucket's masks (2 sigmoid - 1) and their cond channels at 1152 x 1152 down four 3 x 3
//     stride 2 convolutions (LayerNorm2d, GELU) and a 1 x 1, added to the frame's 1x level's 1 x 1 projection, two
//     ConvNeXt blocks (a depthwise 7 x 7, a norm, an MLP of 4 d, a scale), the empty or vanished slots' no-object
//     embeddings added.

#include "strata/image/sam3.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr const char* kTr = "tracker.model.";
constexpr const char* kVb = "detector.backbone.vision_backbone.";
constexpr float kEps = 1e-5f, kEps2d = 1e-6f, kTheta = 10000.f;
constexpr float kPi = 3.14159265358979323846f;
constexpr int64_t kMemHeads = 8, kDecHeads = 8;

// the tracker's scratch slots, past the detector's
enum Slot {
    kL0 = 100, kL1, kL2, kHr016, kHr116, kStream, kNormed16, kImg16, kQa, kQb, kQ16, kK16, kV16, kA16, kY, kKey, kHid16,
    kQkv16, kTok, kTokPe, kKeys, kKeyPe, kUp, kUp16, kUp2, kHyper, kHyper16, kMasks, kSmall, kSmall16, kMux16,
    kConvA, kConvB, kConv16, kFeat, kFeat16, kDw, kPix16, kImds, kPrompt, kPrompt16, kInterL2, kCol,
    kInterL0, kInterL1, kIHr0, kIHr1, kLive,
};

// upload_async's stages: the memory encoder's cond and valid values, its live slots, its no-object sum
enum Stage { kStageCond = 1, kStageValid, kStageLive, kStageAdd };

// the random Fourier positions of a w x h grid ((x + 0.5) / w, (y + 0.5) / h, as SAM's PositionEmbeddingRandom): sin
// and cos of 2 pi (2 c - 1) . G for G [2][F] -> [h w][2F]
std::vector<float> fourier_grid(const std::vector<float>& g, int64_t F, int64_t h, int64_t w) {
    std::vector<float> out((size_t) (h * w * 2 * F));
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x) {
            const float cx = 2.f * (((float) x + 0.5f) / (float) w) - 1.f, cy = 2.f * (((float) y + 0.5f) / (float) h) - 1.f;
            float* o = out.data() + (y * w + x) * 2 * F;
            for (int64_t f = 0; f < F; ++f) {
                const float p = 2.f * kPi * (cx * g[(size_t) f] + cy * g[(size_t) (F + f)]);
                o[f] = std::sin(p);
                o[F + f] = std::cos(p);
            }
        }
    return out;
}

}  // namespace

float* Sam3::track_scratch(int slot, size_t bytes) { return scratch(slot, bytes); }

Sam3::Mlp Sam3::mlp(const std::string& p) {
    Mlp m;
    for (int i = 0; st_->find(p + ".layers." + std::to_string(i) + ".weight") != nullptr; ++i)
        m.l.push_back(lin(p + ".layers." + std::to_string(i)));
    if (m.l.empty()) throw std::runtime_error(path_ + ": no MLP " + p);
    return m;
}

void Sam3::mlp_run(const Mlp& m, const uint16_t* x16, int64_t rows, float* y) {
    // the hidden rows FP16 (ReLU), the last layer's out FP32
    auto* h16 = scratch_as<uint16_t>(kHid16, (size_t) (rows * 2 * d_ * 2));
    const uint16_t* in = x16;
    for (size_t i = 0; i < m.l.size(); ++i) {
        if (i + 1 == m.l.size()) {
            product(in, m.l[i], y, rows);
        } else {
            uint16_t* out = h16 + (i % 2) * rows * d_;
            product16(in, m.l[i], out, rows, true, ImageAct::relu);
            in = out;
        }
    }
}

Sam3::MaskDecoder Sam3::decoder(const std::string& p, int64_t M, int64_t T) {
    MaskDecoder m;
    m.M = M;
    m.T = T;
    for (int i = 0;; ++i) {
        const std::string b = p + ".transformer.layers." + std::to_string(i) + ".";
        if (st_->find(b + "self_attn.q_proj.weight") == nullptr) break;
        TwoWayLayer l;
        l.sq = lin(b + "self_attn.q_proj");
        l.sk = lin(b + "self_attn.k_proj");
        l.sv = lin(b + "self_attn.v_proj");
        l.so = lin(b + "self_attn.out_proj");
        l.tq = lin(b + "cross_attn_token_to_image.q_proj");
        l.tk = lin(b + "cross_attn_token_to_image.k_proj");
        l.tv = lin(b + "cross_attn_token_to_image.v_proj");
        l.to = lin(b + "cross_attn_token_to_image.out_proj");
        l.iq = lin(b + "cross_attn_image_to_token.q_proj");
        l.ik = lin(b + "cross_attn_image_to_token.k_proj");
        l.iv = lin(b + "cross_attn_image_to_token.v_proj");
        l.io = lin(b + "cross_attn_image_to_token.out_proj");
        l.fc1 = lin(b + "mlp.lin1");
        l.fc2 = lin(b + "mlp.lin2");
        l.n1 = norm(b + "norm1");
        l.n2 = norm(b + "norm2");
        l.n3 = norm(b + "norm3");
        l.n4 = norm(b + "norm4");
        m.layers.push_back(l);
    }
    if (m.layers.empty()) throw std::runtime_error(path_ + ": no two-way transformer in " + p);
    const std::string f = p + ".transformer.final_attn_token_to_image.";
    m.fq = lin(f + "q_proj");
    m.fk = lin(f + "k_proj");
    m.fv = lin(f + "v_proj");
    m.fo = lin(f + "out_proj");
    m.nf = norm(p + ".transformer.norm_final_attn");
    m.obj = host_f32(p + ".obj_score_token.weight");
    m.iou = host_f32(p + ".iou_token.weight");
    m.mask = host_f32(p + ".mask_tokens.weight");
    if ((int64_t) m.obj.size() != M * d_ || (int64_t) m.iou.size() != M * d_ || (int64_t) m.mask.size() != M * T * d_)
        throw std::runtime_error(path_ + ": " + p + "'s tokens are not " + std::to_string(M) + " x " +
                                 std::to_string(T));
    m.up1 = deconv2(p + ".output_upscaling.0");
    m.ln1 = norm(p + ".output_upscaling.1");
    m.up2 = deconv2(p + ".output_upscaling.3");
    {
        const Conv c0 = conv(p + ".conv_s0", 1, 0), c1 = conv(p + ".conv_s1", 1, 0);
        m.s0 = Lin{c0.w, c0.b, c0.o, c0.c};
        m.s1 = Lin{c1.w, c1.b, c1.o, c1.c};
    }
    for (int i = 0; st_->find(p + ".output_hypernetworks_mlps." + std::to_string(i) + ".layers.0.weight") != nullptr; ++i)
        m.hyper.push_back(mlp(p + ".output_hypernetworks_mlps." + std::to_string(i)));
    m.iou_head = mlp(p + ".iou_prediction_head");
    m.score_head = mlp(p + ".pred_obj_score_head");
    return m;
}

void Sam3::track_load() {
    load();
    if (track_loaded_) return;
    const std::string t = kTr;
    prop_ = neck_of(std::string(kVb) + "propagation_convs.");
    inter_ = neck_of(std::string(kVb) + "interactive_convs.");
    mdec_ = decoder(t + "sam_mask_decoder", kMux, 3);
    idec_ = decoder(t + "interactive_sam_mask_decoder", 1, 4);
    if ((int64_t) mdec_.hyper.size() != mdec_.T || (int64_t) idec_.hyper.size() != idec_.T)
        throw std::runtime_error(path_ + ": the decoders' hypernetworks");
    // the memory attention: the self-attention's q, k and v stacked; the decoupled cross-attention's query and key
    // as one product of [image | stream] (their projections side by side, their biases summed); its value the
    // memory's alone (zeros over the image's half)
    mem_.clear();
    for (int i = 0;; ++i) {
        const std::string b = t + "transformer.encoder.layers." + std::to_string(i) + ".";
        if (st_->find(b + "self_attn_q_proj.weight") == nullptr) break;
        MemLayer l;
        l.qkv = lin({b + "self_attn_q_proj", b + "self_attn_k_proj", b + "self_attn_v_proj"});
        l.so = lin(b + "self_attn_out_proj");
        const auto side = [&](const std::string& img, const std::string& own, bool img_zero) {
            const auto wi = host_f32(img + ".weight"), wo = host_f32(own + ".weight");
            const auto bi = host_f32(img + ".bias"), bo = host_f32(own + ".bias");
            const int64_t d = d_;
            std::vector<float> w((size_t) (d * 2 * d)), bias((size_t) d);
            for (int64_t r = 0; r < d; ++r) {
                for (int64_t c = 0; c < d; ++c) {
                    w[(size_t) (r * 2 * d + c)] = img_zero ? 0.f : wi[(size_t) (r * d + c)];
                    w[(size_t) (r * 2 * d + d + c)] = wo[(size_t) (r * d + c)];
                }
                bias[(size_t) r] = (img_zero ? 0.f : bi[(size_t) r]) + bo[(size_t) r];
            }
            return lin_host(w, bias, d, 2 * d);
        };
        l.cq_img = lin(b + "image_cross_attn_q_proj");
        l.cq = lin(b + "cross_attn_q_proj");
        l.ck = side(b + "image_cross_attn_k_proj", b + "cross_attn_k_proj", false);
        l.cv = side(b + "cross_attn_v_proj", b + "cross_attn_v_proj", true);
        l.co = lin(b + "cross_attn_out_proj");
        l.fc1 = lin(b + "linear1");
        l.fc2 = lin(b + "linear2");
        l.n1 = norm(b + "norm1");
        l.n2 = norm(b + "norm2");
        l.n3 = norm(b + "norm3");
        mem_.push_back(l);
    }
    if (mem_.empty()) throw std::runtime_error(path_ + ": no memory attention layers");
    mem_norm_ = norm(t + "transformer.encoder.norm");
    // the memory encoder
    mds_.clear();
    mds_ln_.clear();
    mds0_live_.clear();
    for (int i = 0;; i += 3) {
        const std::string b = t + "maskmem_backbone.mask_downsampler.encoder.";
        if (st_->find(b + std::to_string(i + 1) + ".weight") == nullptr) {
            const Conv c = conv(b + std::to_string(i), 1, 0);
            mds_out_ = Lin{c.w, c.b, c.o, c.c};
            break;
        }
        mds_.push_back(conv(b + std::to_string(i), 2, 1));
        mds_ln_.push_back(norm(b + std::to_string(i + 1)));
        if (i == 0) mds0_w_ = host_f32(b + "0.weight");
    }
    {
        const Conv c = conv(t + "maskmem_backbone.pix_feat_proj", 1, 0);
        pix_proj_ = Lin{c.w, c.b, c.o, c.c};
    }
    fuser_.clear();
    for (int i = 0;; ++i) {
        const std::string b = t + "maskmem_backbone.fuser.layers." + std::to_string(i) + ".";
        if (st_->find(b + "dwconv.weight") == nullptr) break;
        CxBlock x;
        x.dw = f32(b + "dwconv.weight");
        x.dwb = f32(b + "dwconv.bias");
        x.gamma = f32(b + "gamma");
        x.n = norm(b + "norm");
        x.pw1 = lin(b + "pwconv1");
        x.pw2 = lin(b + "pwconv2");
        fuser_.push_back(x);
    }
    // the pointers
    ptr_proj_ = mlp(t + "obj_ptr_proj");
    iptr_proj_ = mlp(t + "interactive_obj_ptr_proj");
    no_obj_ptr_ = lin(t + "no_obj_ptr_linear");
    tpos_proj_ = lin(t + "obj_ptr_tpos_proj");
    {
        const auto mt = host_f32(t + "maskmem_tpos_enc");
        mem_frames_ = (int64_t) mt.size() / d_;
        mem_tpos_ = upload(mt);
    }
    // the interactive prompt: the mask's 4 x 4 downsample, the prompt encoder's mask downscaling
    imds_ = conv(t + "interactive_mask_downsample", 4, 0);
    pe_c0_ = conv(t + "interactive_sam_prompt_encoder.mask_downscaling.0", 2, 0);
    pe_ln0_ = norm(t + "interactive_sam_prompt_encoder.mask_downscaling.1");
    pe_c1_ = conv(t + "interactive_sam_prompt_encoder.mask_downscaling.3", 2, 0);
    pe_ln1_ = norm(t + "interactive_sam_prompt_encoder.mask_downscaling.4");
    {
        const Conv c = conv(t + "interactive_sam_prompt_encoder.mask_downscaling.6", 1, 0);
        pe_c2_ = Lin{c.w, c.b, c.o, c.c};
    }
    not_point_ = host_f32(t + "interactive_sam_prompt_encoder.not_a_point_embed.weight");
    no_mask_ = host_f32(t + "interactive_sam_prompt_encoder.no_mask_embed.weight");
    for (int i = 0; i < 4; ++i)
        point_emb_[(size_t) i] =
            host_f32(t + "interactive_sam_prompt_encoder.point_embeddings." + std::to_string(i) + ".weight");
    prompt_gauss_ = host_f32(t + "interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix");
    no_mem_ = host_f32(t + "interactivity_no_mem_embed");
    valid_emb_ = host_f32(t + "output_valid_embed");
    invalid_emb_ = host_f32(t + "output_invalid_embed");
    no_obj_spatial_ = host_f32(t + "no_obj_embed_spatial");
    const int64_t F = d_ / 2;
    dense_pe_ = upload(fourier_grid(host_f32(t + "image_pe_layer.positional_encoding_gaussian_matrix"), F, grid_, grid_));
    prompt_pe_ = upload(fourier_grid(prompt_gauss_, F, grid_, grid_));
    track_loaded_ = true;
}

void Sam3::track_features() {
    const int64_t N = grid_ * grid_;
    float* lv[3] = {scratch(kL0, (size_t) (16 * N * d_ * 4)), scratch(kL1, (size_t) (4 * N * d_ * 4)),
                    scratch(kL2, (size_t) (N * d_ * 4))};
    neck(prop_, lv);
}

const float* Sam3::track_level(int i) const {
    const int slot = i == 0 ? kL0 : i == 1 ? kL1 : kL2;
    return const_cast<Sam3*>(this)->scratch(slot, 0);
}

void Sam3::two_way(const MaskDecoder& m, int64_t B, int64_t Nt, float* q, const float* qpe, float* k,
                   const float* kpe) {
    const int64_t d = d_, P = grid_ * grid_, BT = B * Nt, BP = B * P;
    const int64_t ci = m.layers[0].tq.n;   // the cross-attentions' width (d / 2)
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (BP * d));
    auto* k16 = scratch_as<uint16_t>(kK16, (size_t) (BP * d));
    auto* v16 = scratch_as<uint16_t>(kV16, (size_t) (BP * d));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (BP * d));
    auto* x16 = scratch_as<uint16_t>(kNormed16, (size_t) (BP * d));
    auto* xp16 = scratch_as<uint16_t>(kImg16, (size_t) (BP * d));
    float* y = scratch(kY, (size_t) (BP * d * 4));
    float* kpe_all = scratch(kKeyPe, (size_t) (BP * d * 4));
    for (int64_t b = 0; b < B; ++b) gpu::copy_async(kpe_all + b * P * d, kpe, (size_t) (P * d) * 4, s_);
    const auto attend = [&](const Lin& lq, const Lin& lk, const Lin& lv, const Lin& lo, const uint16_t* qin,
                            const uint16_t* kin, const uint16_t* vin, int64_t nq, int64_t nk, float* out) {
        product16(qin, lq, q16, B * nq, true);
        product16(kin, lk, k16, B * nk, true);
        product16(vin, lv, v16, B * nk, true);
        const int64_t w = lq.n;
        attention(q16, k16, v16, nq, nk, kDecHeads, w / kDecHeads, w, w, w, a16, B);
        product(a16, lo, out, B * nq);
    };
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const TwoWayLayer& l = m.layers[li];
        // the tokens' self-attention (the first layer's without the positions, and without the residual)
        kernels::f32_to_f16_bulk(q, x16, BT * d, s_);
        if (li == 0) {
            attend(l.sq, l.sk, l.sv, l.so, x16, x16, x16, Nt, Nt, y);
            aux_layer_norm(y, nullptr, l.n1.w, l.n1.b, BT, d, kEps, q, nullptr, s_);
        } else {
            aux_add16(q, qpe, BT * d, xp16, s_);
            attend(l.sq, l.sk, l.sv, l.so, xp16, xp16, x16, Nt, Nt, y);
            aux_layer_norm(q, y, l.n1.w, l.n1.b, BT, d, kEps, q, nullptr, s_);
        }
        // the tokens to the image
        aux_add16(q, qpe, BT * d, xp16, s_);
        auto* kp16 = scratch_as<uint16_t>(kHid16, (size_t) (BP * d * 2));
        auto* kk16 = kp16 + BP * d;
        aux_add16(k, kpe_all, BP * d, kp16, s_);
        kernels::f32_to_f16_bulk(k, kk16, BP * d, s_);
        attend(l.tq, l.tk, l.tv, l.to, xp16, kp16, kk16, Nt, P, y);
        aux_layer_norm(q, y, l.n2.w, l.n2.b, BT, d, kEps, q, x16, s_);
        // the MLP
        {
            auto* h16 = scratch_as<uint16_t>(kSmall16, (size_t) (BT * l.fc1.n));
            product16(x16, l.fc1, h16, BT, true, ImageAct::relu);
            product(h16, l.fc2, y, BT);
            aux_layer_norm(q, y, l.n3.w, l.n3.b, BT, d, kEps, q, nullptr, s_);
        }
        // the image to the tokens
        aux_add16(q, qpe, BT * d, xp16, s_);
        kernels::f32_to_f16_bulk(q, x16, BT * d, s_);
        aux_add16(k, kpe_all, BP * d, kp16, s_);
        {
            product16(kp16, l.iq, q16, BP, true);
            product16(xp16, l.ik, k16, BT, true);
            product16(x16, l.iv, v16, BT, true);
            attention(q16, k16, v16, P, Nt, kDecHeads, ci / kDecHeads, ci, ci, ci, a16, B);
            product(a16, l.io, y, BP);
        }
        aux_layer_norm(k, y, l.n4.w, l.n4.b, BP, d, kEps, k, nullptr, s_);
    }
    // the last tokens to the image
    aux_add16(q, qpe, BT * d, xp16, s_);
    auto* kp16 = scratch_as<uint16_t>(kHid16, (size_t) (BP * d * 2));
    auto* kk16 = kp16 + BP * d;
    aux_add16(k, kpe_all, BP * d, kp16, s_);
    kernels::f32_to_f16_bulk(k, kk16, BP * d, s_);
    attend(m.fq, m.fk, m.fv, m.fo, xp16, kp16, kk16, Nt, P, y);
    aux_layer_norm(q, y, m.nf.w, m.nf.b, BT, d, kEps, q, nullptr, s_);
}

void Sam3::mem_attention(const float* x, const float* pos, const uint16_t* mem16, int64_t B, int64_t N, int64_t nsp,
                         const float* sine, const float* tpos, const float* ptr_pos, int64_t per, float* out) {
    const int64_t d = d_, G = grid_, P = G * G, BP = B * P, hd = d / kMemHeads;
    auto* img16 = scratch_as<uint16_t>(kImg16, (size_t) (P * d));
    kernels::f32_to_f16_bulk(x, img16, P * d, s_);
    // the stream: x + 0.1 pos, a bucket after another
    {
        for (int64_t b = 0; b < B; ++b) gpu::copy_async(out + b * P * d, x, (size_t) (P * d) * 4, s_);
        float* p = scratch(kY, (size_t) (BP * d * 4));
        for (int64_t b = 0; b < B; ++b) gpu::copy_async(p + b * P * d, pos, (size_t) (P * d) * 4, s_);
        qi_axpy(out, p, 0.1f, BP * d, s_);
    }
    auto* n16 = scratch_as<uint16_t>(kNormed16, (size_t) (BP * d));
    auto* qkv16 = scratch_as<uint16_t>(kQkv16, (size_t) (BP * 3 * d));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (BP * d));
    auto* k16 = scratch_as<uint16_t>(kK16, (size_t) (std::max(BP, B * N) * d));
    auto* v16 = scratch_as<uint16_t>(kV16, (size_t) (std::max(BP, B * N) * d));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (BP * d));
    float* y = scratch(kY, (size_t) (BP * d * 4));
    float* qa = scratch(kQa, (size_t) (P * d * 4));
    float* key = scratch(kKey, (size_t) (B * N * d * 4));
    host("mem_in", out, (size_t) (BP * d));
    for (const MemLayer& l : mem_) {
        // the self-attention, q and k turned on the grid
        aux_layer_norm(out, nullptr, l.n1.w, l.n1.b, BP, d, kEps, nullptr, n16, s_);
        product16(n16, l.qkv, qkv16, BP);
        aux_rope2d_split16h(qkv16, l.qkv.b, BP, kMemHeads, hd, G, 1.f, kTheta, q16, k16, v16, s_);
        attention(q16, k16, v16, P, P, kMemHeads, hd, d, d, d, a16, B);
        product(a16, l.so, y, BP);
        qi_axpy(out, y, 1.f, BP * d, s_);
        host("mem_self", out, (size_t) (BP * d));
        // the decoupled cross-attention to the memory
        aux_layer_norm(out, nullptr, l.n2.w, l.n2.b, BP, d, kEps, nullptr, n16, s_);
        product(img16, l.cq_img, qa, P);   // the image's share, the same for every bucket
        product(n16, l.cq, y, BP);
        aux_rope2d16(y, qa, P, BP, kMemHeads, hd, G, kTheta, q16, s_);
        product(mem16, l.ck, key, B * N);
        aux_mem_key16(key, B, N, nsp, P, sine, tpos, ptr_pos, per, kMemHeads, hd, G, kTheta, k16, s_);
        product16(mem16, l.cv, v16, B * N, true);
        attention(q16, k16, v16, P, N, kMemHeads, hd, d, d, d, a16, B);
        product(a16, l.co, y, BP);
        qi_axpy(out, y, 1.f, BP * d, s_);
        host("mem_cross", out, (size_t) (BP * d));
        // the MLP
        aux_layer_norm(out, nullptr, l.n3.w, l.n3.b, BP, d, kEps, nullptr, n16, s_);
        auto* h16 = scratch_as<uint16_t>(kHid16, (size_t) (BP * l.fc1.n));
        product16(n16, l.fc1, h16, BP, true, ImageAct::gelu);
        product(h16, l.fc2, y, BP);
        qi_axpy(out, y, 1.f, BP * d, s_);
        host("mem_layer", out, (size_t) (BP * d));
    }
    aux_layer_norm(out, nullptr, mem_norm_.w, mem_norm_.b, BP, d, kEps, out, nullptr, s_);
}

// A decoder's heads on one batch: its keys kb [P][d] upscaled to the 4x level (2x with the 2x level's projection s1,
// LayerNorm2d, GELU; 4x with the 4x level's s0, GELU), each of the M objects' T mask tokens (qb's rows from 2M; the
// scores' from 0, the IoUs' from M) through its hypernetwork by those features -> masks [M T][P0], iou [M][T], score [M]
void Sam3::mask_heads(const MaskDecoder& m, const float* qb, int64_t Nt, const float* kb, const float* s0,
                      const float* s1, float* masks, float* iou, float* score) {
    const int64_t d = d_, G = grid_, P = G * G, M = m.M, T = m.T;
    const int64_t P1 = 4 * P, P0 = 16 * P, c1 = m.up1.n / 4, c0 = m.up2.n / 4;
    auto* src16 = scratch_as<uint16_t>(kNormed16, (size_t) (P * d));
    float* up = scratch(kUp, (size_t) (P1 * std::max(4 * c1, 4 * c0) * 4));
    float* u1 = scratch(kUp2, (size_t) (P1 * c1 * 4));
    auto* u116 = scratch_as<uint16_t>(kUp16, (size_t) (P1 * c1));
    auto* u016 = scratch_as<uint16_t>(kConv16, (size_t) (P0 * c0));
    float* u0 = scratch(kFeat, (size_t) (P0 * c0 * 4));
    float* hyp = scratch(kHyper, (size_t) (T * M * T * c0 * 4));
    auto* hyp16 = scratch_as<uint16_t>(kHyper16, (size_t) (M * T * c0));
    auto* tok16 = scratch_as<uint16_t>(kSmall16, (size_t) (Nt * d));
    kernels::f32_to_f16_bulk(kb, src16, P * d, s_);
    product(src16, Lin{m.up1.w, nullptr, m.up1.n, m.up1.k}, up, P);
    aux_deconv2_scatter(up, G, G, c1, m.up1.b, false, u1, nullptr, s_);
    qi_axpy(u1, s1, 1.f, P1 * c1, s_);
    aux_layer_norm(u1, nullptr, m.ln1.w, m.ln1.b, P1, c1, kEps2d, u1, nullptr, s_);
    kernels::f32_to_f16_bulk(u1, u116, P1 * c1, s_);
    aux_bias_act16(u116, nullptr, P1, c1, ImageAct::gelu, u116, s_);
    product(u116, Lin{m.up2.w, nullptr, m.up2.n, m.up2.k}, up, P1);
    aux_deconv2_scatter(up, 2 * G, 2 * G, c0, m.up2.b, false, u0, nullptr, s_);
    qi_axpy(u0, s0, 1.f, P0 * c0, s_);
    kernels::f32_to_f16_bulk(u0, u016, P0 * c0, s_);
    aux_bias_act16(u016, nullptr, P0, c0, ImageAct::gelu, u016, s_);
    // each mask token through its mask's hypernetwork (the masks' t-th MLP for the t-th of an object's), then by the
    // upscaled features
    kernels::f32_to_f16_bulk(qb, tok16, Nt * d, s_);
    const uint16_t* mt16 = tok16 + 2 * M * d;
    for (int64_t t = 0; t < T; ++t) mlp_run(m.hyper[(size_t) t], mt16, M * T, hyp + t * M * T * c0);
    aux_mod_select16(hyp, T, M * T, c0, hyp16, s_);
    gemm_.f16(hyp16, u016, masks, M * T, P0, c0);
    mlp_run(m.iou_head, tok16 + M * d, M, iou);
    mlp_run(m.score_head, tok16, M, score);
}

Sam3::MuxOut Sam3::mux_decode(const float* img, const float* hr0, const float* hr1, const std::vector<uint8_t>& valid,
                              int64_t B) {
    const MaskDecoder& m = mdec_;
    const int64_t d = d_, G = grid_, P = G * G, M = m.M, T = m.T, Nt = 2 * M + M * T, BT = B * Nt;
    const int64_t P1 = 4 * P, P0 = 16 * P, c1 = m.up1.n / 4, c0 = m.up2.n / 4;
    if ((int64_t) valid.size() != B * M) throw std::invalid_argument("SAM 3 tracker: the slots' validity");
    // the tokens: the objects' scores', their IoUs', then each object's masks' (with its slot's validity embedding)
    std::vector<float> tok((size_t) (BT * d));
    for (int64_t b = 0; b < B; ++b) {
        float* o = tok.data() + b * Nt * d;
        std::copy(m.obj.begin(), m.obj.end(), o);
        std::copy(m.iou.begin(), m.iou.end(), o + M * d);
        for (int64_t s = 0; s < M; ++s) {
            const std::vector<float>& e = valid[(size_t) (b * M + s)] ? valid_emb_ : invalid_emb_;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t c = 0; c < d; ++c)
                    o[(2 * M + s * T + t) * d + c] = m.mask[(size_t) ((s * T + t) * d + c)] + e[(size_t) (s * d + c)];
        }
    }
    float* q = scratch(kTok, (size_t) (BT * d * 4));
    float* qpe = scratch(kTokPe, (size_t) (BT * d * 4));
    gpu::copy_async(q, tok.data(), tok.size() * 4, s_);
    gpu::copy_async(qpe, tok.data(), tok.size() * 4, s_);
    float* k = scratch(kKeys, (size_t) (B * P * d * 4));
    gpu::copy_async(k, img, (size_t) (B * P * d) * 4, s_);
    two_way(m, B, Nt, q, qpe, k, dense_pe_);
    gpu::stream_sync(s_);   // tok's host rows outlive their copies
    // the 4x and 2x levels' projections, once for the buckets
    auto* h016 = scratch_as<uint16_t>(kHr016, (size_t) (P0 * d));
    auto* h116 = scratch_as<uint16_t>(kHr116, (size_t) (P1 * d));
    kernels::f32_to_f16_bulk(hr0, h016, P0 * d, s_);
    kernels::f32_to_f16_bulk(hr1, h116, P1 * d, s_);
    float* s1 = scratch(kConvA, (size_t) (P1 * c1 * 4));
    float* s0 = scratch(kConvB, (size_t) (P0 * c0 * 4));
    product(h116, m.s1, s1, P1);
    product(h016, m.s0, s0, P0);
    MuxOut r;
    r.T = T;
    r.P0 = P0;
    r.masks = scratch(kMasks, (size_t) (B * M * T * P0 * 4));
    r.iou = scratch(kSmall, (size_t) (B * M * (T + 1 + T * d) * 4));
    r.score = r.iou + B * M * T;
    r.tokens = r.score + B * M;
    for (int64_t b = 0; b < B; ++b) {
        mask_heads(m, q + b * Nt * d, Nt, k + b * P * d, s0, s1, r.masks + b * M * T * P0, r.iou + b * M * T,
                   r.score + b * M);
        gpu::copy_async(r.tokens + b * M * T * d, q + (b * Nt + 2 * M) * d, (size_t) (M * T * d) * 4, s_);
    }
    return r;
}

void Sam3::pointers(const float* tokens, int64_t n, bool interactive, const std::vector<float>& score, float* ptr) {
    const int64_t d = d_;
    auto* t16 = scratch_as<uint16_t>(kPrompt16, (size_t) (n * d));
    kernels::f32_to_f16_bulk(tokens, t16, n * d, s_);
    mlp_run(interactive ? iptr_proj_ : ptr_proj_, t16, n, ptr);
    bool any_gone = false;
    for (float s : score) any_gone = any_gone || !(s > 0.f);
    if (!any_gone) return;
    // the rows not seen as an object: the no-object linear of the pointer
    float* alt = scratch(kPrompt, (size_t) (n * d * 4));
    kernels::f32_to_f16_bulk(ptr, t16, n * d, s_);
    product(t16, no_obj_ptr_, alt, n);
    for (int64_t i = 0; i < n; ++i)
        if (!(score[(size_t) i] > 0.f)) gpu::copy_async(ptr + i * d, alt + i * d, (size_t) d * 4, s_);
}

void Sam3::mem_encode(const float* pix, const float* masks, int64_t N, int64_t h, int64_t w, bool sigmoid_first,
                      const std::vector<float>& cond, const std::vector<float>& is_obj, int64_t B, float* mem,
                      uint16_t* mem16, int64_t ld, int64_t col, const std::vector<float>* valid) {
    const int64_t M = kMux, S = 16 * grid_;   // the masks' side: 1152
    if ((int64_t) cond.size() != N || (int64_t) is_obj.size() != N || (valid != nullptr && (int64_t) valid->size() != N))
        throw std::invalid_argument("SAM 3 tracker: cond");
    float* dcond = scratch(kSmall, (size_t) std::max<int64_t>(2 * N, 1) * 4);
    upload_async(dcond, cond.data(), (size_t) N * 4, kStageCond);
    if (valid != nullptr) upload_async(dcond + N, valid->data(), (size_t) N * 4, kStageValid);
    auto* mux16 = scratch_as<uint16_t>(kMux16, (size_t) (S * S * 2 * M));
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> slot((size_t) M, 0.f);
        std::vector<int32_t> live;
        for (int64_t s = 0; s < M && b * M + s < N; ++s) {
            slot[(size_t) s] = is_obj[(size_t) (b * M + s)];
            if (valid == nullptr || (*valid)[(size_t) (b * M + s)] > 0.f) live.push_back((int32_t) s);
        }
        // a bucket of few objects: the input of its live slots alone, the empty ones' zero channels left out of the
        // first convolution (the same sums; the B70's input 1152 x 1152 x 32 0.9 ms an encode, 2026-11-07)
        const Conv* first = nullptr;
        if (!live.empty() && (int64_t) live.size() < M) {
            first = &mds_first(live);
            auto* dl = reinterpret_cast<int32_t*>(scratch(kLive, live.size() * 4));
            std::vector<int32_t> abs(live);
            for (int32_t& v : abs) v += (int32_t) (b * M);
            upload_async(dl, abs.data(), abs.size() * 4, kStageLive);
            aux_mux_masks16(masks, N, h, w, S, S, sigmoid_first, dcond, nullptr, M, 1, mux16, s_, dl,
                            (int64_t) live.size());
        } else {
            aux_mux_masks16(masks + b * M * h * w, std::min<int64_t>(M, N - b * M), h, w, S, S, sigmoid_first,
                            dcond + b * M, valid != nullptr ? dcond + N + b * M : nullptr, M, 1, mux16, s_);
        }
        encode_bucket(pix, mux16, slot, mem != nullptr ? mem + b * P_() * d_ : nullptr,
                      mem16 != nullptr ? mem16 + b * P_() * ld : nullptr, ld, col, first);
    }
}

void Sam3::mem_encode_muxed(const float* pix, const uint16_t* mux16, const std::vector<float>& slot_is_obj,
                            float* mem) {
    encode_bucket(pix, mux16, slot_is_obj, mem, nullptr, 0, 0);
}

const AuxNet::Conv& Sam3::mds_first(const std::vector<int32_t>& live) {
    const auto it = mds0_live_.find(live);
    if (it != mds0_live_.end()) return it->second;
    const Conv& c = mds_[0];
    const int64_t M = kMux, L = (int64_t) live.size(), kk = (int64_t) c.k * c.k;
    std::vector<float> w((size_t) (c.o * 2 * L * kk));
    for (int64_t o = 0; o < c.o; ++o)
        for (int64_t j = 0; j < 2 * L; ++j) {
            const int64_t src = j < L ? live[(size_t) j] : M + live[(size_t) (j - L)];
            std::copy_n(mds0_w_.begin() + (o * 2 * M + src) * kk, kk, w.begin() + (o * 2 * L + j) * kk);
        }
    std::vector<float> b((size_t) c.o);
    gpu::copy(b.data(), c.b, b.size() * 4);
    if (mds0_live_.size() >= 64) mds0_live_.clear();   // the sets a clip's buckets went through: few
    return mds0_live_[live] = conv_host(w, b, c.o, 2 * L, c.k, c.stride, c.pad);
}

void Sam3::encode_bucket(const float* pix, const uint16_t* mux16, const std::vector<float>& slot_is_obj, float* mem,
                         uint16_t* mem16, int64_t ld, int64_t col, const Conv* first) {
    const int64_t d = d_, G = grid_, P = G * G, M = kMux, S = 16 * G;
    auto* pix16 = scratch_as<uint16_t>(kPix16, (size_t) (P * d));
    kernels::f32_to_f16_bulk(pix, pix16, P * d, s_);
    float* feat = scratch(kFeat, (size_t) (P * d * 4));
    // the bucket's masks and cond channels down the convolutions (LayerNorm2d and GELU after each)
    const uint16_t* in16 = mux16;
    int64_t side = S;
    for (size_t i = 0; i < mds_.size(); ++i) {
        const Conv& c = i == 0 && first != nullptr ? *first : mds_[i];
        const int64_t out = (side + int64_t{2} * c.pad - c.k) / c.stride + 1;
        float* y = scratch(kConvA, (size_t) (out * out * c.o * 4));
        convolve(c, in16, side, side, y, kCol);
        auto* y16 = scratch_as<uint16_t>(i % 2 == 0 ? kConv16 : kFeat16, (size_t) (out * out * c.o));
        aux_layer_norm_gelu16(y, mds_ln_[i].w, mds_ln_[i].b, out * out, c.o, kEps2d, y16, s_);
        in16 = y16;
        side = out;
    }
    if (side != G) throw std::runtime_error("SAM 3 tracker: the mask downsampler's size");
    product(in16, mds_out_, feat, P);
    // + the frame's projection, then the fuser's ConvNeXt blocks
    float* pp = scratch(kConvB, (size_t) (P * d * 4));
    product(pix16, pix_proj_, pp, P);
    qi_axpy(feat, pp, 1.f, P * d, s_);
    float* dw = scratch(kDw, (size_t) (P * d * 4));
    auto* n16 = scratch_as<uint16_t>(kNormed16, (size_t) (P * d));
    for (const CxBlock& x : fuser_) {
        aux_dwconv(feat, x.dw, x.dwb, G, G, d, 7, dw, s_);
        aux_layer_norm(dw, nullptr, x.n.w, x.n.b, P, d, kEps2d, nullptr, n16, s_);
        auto* h16 = scratch_as<uint16_t>(kHid16, (size_t) (P * x.pw1.n));
        product16(n16, x.pw1, h16, P, true, ImageAct::gelu);
        product(h16, x.pw2, dw, P);
        aux_scale_add(feat, dw, x.gamma, P, d, s_);
    }
    // the slots without an object (empty, or the object not seen): their no-object embeddings, summed
    std::vector<float> add((size_t) d, 0.f);
    for (int64_t s = 0; s < M; ++s) {
        if (slot_is_obj[(size_t) s] > 0.f) continue;
        for (int64_t c = 0; c < d; ++c) add[(size_t) c] += no_obj_spatial_[(size_t) (s * d + c)];
    }
    float* da = scratch(kSmall16, (size_t) d * 4);
    upload_async(da, add.data(), add.size() * 4, kStageAdd);
    aux_bias_act(feat, P, d, d, da, false, s_);
    if (mem != nullptr) gpu::copy_async(mem, feat, (size_t) (P * d) * 4, s_);
    if (mem16 != nullptr) {
        auto* f16 = scratch_as<uint16_t>(kFeat16, (size_t) (P * d));
        kernels::f32_to_f16_bulk(feat, f16, P * d, s_);
        gpu::copy2d_async(mem16 + col, (size_t) ld * 2, f16, (size_t) d * 2, (size_t) d * 2, (size_t) P, s_);
    }
    gpu::stream_sync(s_);   // add's host row
}

void Sam3::ptr_positions(const std::vector<int64_t>& t, int64_t max_abs, float* out) {
    // the 1D sine of t / (max_abs - 1): sin of the d / 2 frequencies 10000^(2 (i / 2) / (d / 2)), then their cos
    const int64_t d = d_, F = d / 2, n = (int64_t) t.size();
    if (n == 0) return;
    std::vector<float> pe((size_t) (n * d));
    for (int64_t r = 0; r < n; ++r) {
        const float pos = (float) t[(size_t) r] / (float) std::max<int64_t>(max_abs - 1, 1);
        for (int64_t i = 0; i < F; ++i) {
            // Each pair of channels uses the same integer frequency index.
            const float dt = std::pow(kTheta, (float) (2 * (i / 2)) / (float) F); // NOLINT(bugprone-integer-division)
            pe[(size_t) (r * d + i)] = std::sin(pos / dt);
            pe[(size_t) (r * d + F + i)] = std::cos(pos / dt);
        }
    }
    auto* p16 = scratch_as<uint16_t>(kPrompt16, (size_t) (n * d));
    std::vector<uint16_t> h((size_t) (n * d));
    for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(pe[i]);
    gpu::copy(p16, h.data(), h.size() * 2);
    product(p16, tpos_proj_, out, n);
}

// the prompt encoder's dense embedding of a mask's logits at the 4x level [4G][4G]: its downscaling (2 x 2
// convolutions with LayerNorm2d and GELU, a 1 x 1) -> out [P][d]
void Sam3::mask_dense(const float* low, float* out) {
    const int64_t G = grid_, P = G * G, P0 = 16 * P;
    auto* up16 = scratch_as<uint16_t>(kUp16, (size_t) P0);
    float* c0 = scratch(kConvA, (size_t) (4 * P * pe_c0_.o * 4));
    auto* c016 = scratch_as<uint16_t>(kFeat16, (size_t) (4 * P * pe_c0_.o));
    float* c1 = scratch(kConvB, (size_t) (P * pe_c1_.o * 4));
    auto* c116 = scratch_as<uint16_t>(kPix16, (size_t) (P * pe_c1_.o));
    kernels::f32_to_f16_bulk(low, up16, P0, s_);
    convolve(pe_c0_, up16, 4 * G, 4 * G, c0, kCol);
    aux_layer_norm(c0, nullptr, pe_ln0_.w, pe_ln0_.b, 4 * P, pe_c0_.o, kEps2d, nullptr, c016, s_);
    aux_bias_act16(c016, nullptr, 4 * P, pe_c0_.o, ImageAct::gelu, c016, s_);
    convolve(pe_c1_, c016, 2 * G, 2 * G, c1, kCol);
    aux_layer_norm(c1, nullptr, pe_ln1_.w, pe_ln1_.b, P, pe_c1_.o, kEps2d, nullptr, c116, s_);
    aux_bias_act16(c116, nullptr, P, pe_c1_.o, ImageAct::gelu, c116, s_);
    product(c116, pe_c2_, out, P);
}

void Sam3::interactive(const float* masks, int64_t n, float* tokens, std::vector<float>& score) {
    const MaskDecoder& m = idec_;
    const int64_t d = d_, G = grid_, P = G * G, S = image_, Nt = 2 + m.T + 2;
    score.assign((size_t) n, 0.f);
    // the frame's interactive 1x level, + the no-memory embedding, as every object's keys
    float* lv[3] = {nullptr, nullptr, scratch(kInterL2, (size_t) (P * d * 4))};
    neck(inter_, lv);
    float* nm = scratch(kSmall, (size_t) d * 4);
    gpu::copy(nm, no_mem_.data(), (size_t) d * 4);
    aux_bias_act(lv[2], P, d, d, nm, false, s_);
    float* k = scratch(kKeys, (size_t) (n * P * d * 4));
    // each mask: the 4 x 4 downsample (S / 4), up to 4 G bilinearly, the prompt encoder's downscaling to G -> + keys
    const int64_t s4 = S / 4, P0 = 16 * P;
    auto* m16 = scratch_as<uint16_t>(kConv16, (size_t) (S * S));
    float* ds = scratch(kImds, (size_t) (s4 * s4 * 4));
    float* up = scratch(kUp, (size_t) (P0 * 4));
    for (int64_t i = 0; i < n; ++i) {
        kernels::f32_to_f16_bulk(masks + i * S * S, m16, S * S, s_);
        convolve(imds_, m16, S, S, ds, kCol);
        host("imds", ds, (size_t) (s4 * s4));
        aux_resize_into(ds, s4, s4, 1, up, 4 * G, 4 * G, 1, 0, false, false, s_);
        mask_dense(up, k + i * P * d);
        host("prompt_dense", k + i * P * d, (size_t) (P * d));
        qi_axpy(k + i * P * d, lv[2], 1.f, P * d, s_);
    }
    // the tokens: the score's, the IoU's, the 4 masks', two not-a-point embeddings (the empty point prompt, padded)
    std::vector<float> tok((size_t) (n * Nt * d));
    for (int64_t i = 0; i < n; ++i) {
        float* o = tok.data() + i * Nt * d;
        std::copy(m.obj.begin(), m.obj.end(), o);
        std::copy(m.iou.begin(), m.iou.end(), o + d);
        std::copy(m.mask.begin(), m.mask.end(), o + 2 * d);
        std::copy(not_point_.begin(), not_point_.end(), o + (2 + m.T) * d);
        std::copy(not_point_.begin(), not_point_.end(), o + (3 + m.T) * d);
    }
    float* q = scratch(kTok, (size_t) (n * Nt * d * 4));
    float* qpe = scratch(kTokPe, (size_t) (n * Nt * d * 4));
    gpu::copy_async(q, tok.data(), tok.size() * 4, s_);
    gpu::copy_async(qpe, tok.data(), tok.size() * 4, s_);
    two_way(m, n, Nt, q, qpe, k, prompt_pe_);
    // the first mask token (the single mask's) and the score
    auto* t16 = scratch_as<uint16_t>(kSmall16, (size_t) (n * Nt * d));
    kernels::f32_to_f16_bulk(q, t16, n * Nt * d, s_);
    float* sc = scratch(kHyper, (size_t) (n * 4));
    for (int64_t i = 0; i < n; ++i) {
        gpu::copy_async(tokens + i * d, q + (i * Nt + 2) * d, (size_t) d * 4, s_);
        mlp_run(m.score_head, t16 + i * Nt * d, 1, sc + i);
    }
    gpu::stream_sync(s_);
    gpu::copy(score.data(), sc, (size_t) n * 4);
}

float Sam3::point_prompt(const std::vector<std::array<float, 3>>& pts, const float* prev, std::vector<float>& low,
                         float* token) {
    const MaskDecoder& m = idec_;
    const int64_t d = d_, G = grid_, P = G * G, P0 = 16 * P, P1 = 4 * P, T = m.T;
    const int64_t np = (int64_t) pts.size(), Nt = 2 + T + np + 1, F = d / 2;
    const int64_t c1 = m.up1.n / 4, c0 = m.up2.n / 4;
    if (np == 0) throw std::invalid_argument("SAM 3 tracker: a point prompt without points");
    // the frame's interactive levels; the 1x one with the no-memory embedding, + the dense prompt: the mask before
    // (its logits through the downscaling) or the no-mask embedding
    float* lv[3] = {scratch(kInterL0, (size_t) (16 * P * d * 4)), scratch(kInterL1, (size_t) (4 * P * d * 4)),
                    scratch(kInterL2, (size_t) (P * d * 4))};
    neck(inter_, lv);
    std::vector<float> add(no_mem_);
    if (prev == nullptr)
        for (int64_t c = 0; c < d; ++c) add[(size_t) c] += no_mask_[(size_t) c];
    float* nm = scratch(kSmall, (size_t) d * 4);
    gpu::copy(nm, add.data(), (size_t) d * 4);
    aux_bias_act(lv[2], P, d, d, nm, false, s_);
    float* k = scratch(kKeys, (size_t) (P * d * 4));
    if (prev != nullptr) {
        mask_dense(prev, k);
        qi_axpy(k, lv[2], 1.f, P * d, s_);
    } else {
        gpu::copy_async(k, lv[2], (size_t) (P * d) * 4, s_);
    }
    // the tokens: the score's, the IoU's, the 4 masks', then the points' (each its random Fourier position, of its
    // center in the 1008 picture, + its label's embedding) and a padding one (the not-a-point embedding)
    std::vector<float> tok((size_t) (Nt * d), 0.f);
    std::copy(m.obj.begin(), m.obj.end(), tok.begin());
    std::copy(m.iou.begin(), m.iou.end(), tok.begin() + d);
    std::copy(m.mask.begin(), m.mask.end(), tok.begin() + 2 * d);
    constexpr double kTwoPi = 6.283185307179586;
    for (int64_t i = 0; i < np; ++i) {
        float* o = tok.data() + (2 + T + i) * d;
        const int label = (int) pts[(size_t) i][2];
        if (label < 0 || label > 3) throw std::invalid_argument("SAM 3 tracker: a point's label is 0 to 3");
        const double x = 2.0 * ((pts[(size_t) i][0] * (double) image_ + 0.5) / (double) image_) - 1.0;
        const double y = 2.0 * ((pts[(size_t) i][1] * (double) image_ + 0.5) / (double) image_) - 1.0;
        for (int64_t f = 0; f < F; ++f) {
            const double a = kTwoPi * (x * prompt_gauss_[(size_t) f] + y * prompt_gauss_[(size_t) (F + f)]);
            o[f] = (float) std::sin(a) + point_emb_[(size_t) label][(size_t) f];
            o[F + f] = (float) std::cos(a) + point_emb_[(size_t) label][(size_t) (F + f)];
        }
    }
    std::copy(not_point_.begin(), not_point_.end(), tok.begin() + (Nt - 1) * d);
    float* q = scratch(kTok, (size_t) (Nt * d * 4));
    float* qpe = scratch(kTokPe, (size_t) (Nt * d * 4));
    gpu::copy_async(q, tok.data(), tok.size() * 4, s_);
    gpu::copy_async(qpe, tok.data(), tok.size() * 4, s_);
    two_way(m, 1, Nt, q, qpe, k, prompt_pe_);
    gpu::stream_sync(s_);   // tok's host rows outlive their copies
    // the 4x and 2x levels through the interactive decoder's projections, then its heads
    auto* h016 = scratch_as<uint16_t>(kHr016, (size_t) (P0 * d));
    auto* h116 = scratch_as<uint16_t>(kHr116, (size_t) (P1 * d));
    kernels::f32_to_f16_bulk(lv[0], h016, P0 * d, s_);
    kernels::f32_to_f16_bulk(lv[1], h116, P1 * d, s_);
    float* s1 = scratch(kIHr1, (size_t) (P1 * c1 * 4));
    float* s0 = scratch(kIHr0, (size_t) (P0 * c0 * 4));
    product(h116, m.s1, s1, P1);
    product(h016, m.s0, s0, P0);
    float* masks = scratch(kMasks, (size_t) (T * P0 * 4));
    float* small = scratch(kSmall, (size_t) ((T + 1) * 4));
    mask_heads(m, q, Nt, k, s0, s1, masks, small, small + T);
    gpu::stream_sync(s_);
    std::vector<float> iou((size_t) T + 1);
    gpu::copy(iou.data(), small, iou.size() * 4);
    const float score = iou[(size_t) T];
    // one point: the best of the 3 masks by their IoU; more (a box's corners among them): the single mask, unless it
    // is unstable (its area at logits past +-0.05 differing by 2% or more), then the best of the 3 (a dynamic
    // multimask choice)
    const auto best_of_three = [&] {
        int64_t b = 1;
        for (int64_t t = 2; t < T; ++t)
            if (iou[(size_t) t] > iou[(size_t) b]) b = t;
        return b;
    };
    int64_t pick = 0;
    low.resize((size_t) P0);
    if (np <= 1) {
        pick = best_of_three();
    } else {
        gpu::copy(low.data(), masks, (size_t) P0 * 4);
        int64_t in = 0, un = 0;
        for (float v : low) {
            in += v > 0.05f;
            un += v > -0.05f;
        }
        if (un > 0 && (float) in / (float) un < 0.98f) pick = best_of_three();
    }
    gpu::copy(low.data(), masks + pick * P0, (size_t) P0 * 4);
    if (!(score > 0.f)) std::fill(low.begin(), low.end(), -1024.f);
    if (token != nullptr) gpu::copy_async(token, q + (2 + pick) * d, (size_t) d * 4, s_);
    return score;
}

uint64_t Sam3::video_need_bytes(int64_t width, int64_t height, int64_t buckets) const {
    const uint64_t G = (uint64_t) grid_, P = G * G, P0 = 16 * P, P1 = 4 * P, d = (uint64_t) d_, M = kMux, T = 4;
    const uint64_t S = 16 * G, B = (uint64_t) std::max<int64_t>(buckets, 1);
    uint64_t want = 2 * (P0 + P1 + P) * d * 4;          // the propagation's and the clicks' levels
    want += 3 * S * S * 2 * M * 2;                      // the memory encoder's input and its first convolutions
    want += M * T * P0 * 4 + (P0 + P1) * d * 2 + P0 * d * 4;   // the decoder's masks, levels, upscaled features
    const uint64_t rows = (uint64_t) (mem_frames_ + 4) * P + 16 * M;   // a frame's memory attention's keys
    want += rows * 2 * d * 2 + rows * d * 2 * 4;        // their rows, and their keys' and values' projections
    // each bucket's frames kept (the pointers' reach and the memories', its first frame): rows and pointers
    want += B * (uint64_t) (16 + 2) * (P * 2 * d * 2 + M * d * 4);
    return need_bytes(width, height) + want;
}

}  // namespace strata::image
