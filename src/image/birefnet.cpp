// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/birefnet.cpp - see include/strata/image/birefnet.hpp.

#include "strata/image/birefnet.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr float kEps = 1e-5f;   // the layer and batch norms' (torch's defaults)
constexpr int64_t kDecInter = 64, kAspp = 256;   // BasicDecBlk's and ASPPDeformable's widths (the model's config)

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

enum Slot {
    kRgba, kImg, kImg16, kHalf, kHalf16, kX, kY, kH16, kW16, kBig, kG16, kQ16, kK16, kV16, kA16, kCol, kMerge,
    kF0, kF1, kF2, kF3,            // a pass's stage outputs (the full picture's, then the half's)
    kH0, kH1, kH2, kH3,
    kC1, kC2, kC3, kC4,            // the stages of both passes, concatenated
    kCat, kCat16, kSq, kT, kT16, kAspp5, kAspp16, kOm, kGap, kGap16, kP, kP16, kGdt, kGdt16, kAttn, kPatch16,
    kIpt, kIpt2, kLogit, kLogitUp, kAlpha,
};

}  // namespace

Birefnet::Birefnet(const std::string& path) : AuxNet(path, "BiRefNet") {
    const SafeTensor* pe = st_->find("bb.patch_embed.proj.weight");
    const SafeTensor* tb = st_->find("bb.layers.0.blocks.0.attn.relative_position_bias_table");
    if (pe == nullptr || tb == nullptr || pe->shape.size() != 4 || tb->shape.size() != 2)
        throw std::runtime_error(path + ": not BiRefNet's weights (no Swin backbone)");
    embed_ = pe->shape[0];
    window_ = ((int64_t) std::lround(std::sqrt((double) tb->shape[0])) + 1) / 2;
    if ((2 * window_ - 1) * (2 * window_ - 1) != tb->shape[0])
        throw std::runtime_error(path + ": the relative position table is not a square window's");
}

AuxNet::Conv Birefnet::conv_bn(const std::string& conv, const std::string& bn, int stride, int pad) {
    const SafeTensor* t = st_->find(conv + ".weight");
    if (t == nullptr || t->shape.size() != 4) throw std::runtime_error(path_ + ": no convolution " + conv);
    const int64_t o = t->shape[0], c = t->shape[1], k = t->shape[2];
    auto w = host_f32(conv + ".weight");
    std::vector<float> b = st_->find(conv + ".bias") != nullptr ? host_f32(conv + ".bias") : std::vector<float>((size_t) o, 0.f);
    if (!bn.empty() && st_->find(bn + ".running_mean") != nullptr) {
        const auto mean = host_f32(bn + ".running_mean"), var = host_f32(bn + ".running_var");
        const auto gamma = st_->find(bn + ".weight") != nullptr ? host_f32(bn + ".weight") : std::vector<float>((size_t) o, 1.f);
        const auto beta = st_->find(bn + ".bias") != nullptr ? host_f32(bn + ".bias") : std::vector<float>((size_t) o, 0.f);
        const int64_t per = c * k * k;
        for (int64_t i = 0; i < o; ++i) {
            const float s = gamma[(size_t) i] / std::sqrt(var[(size_t) i] + kEps);
            for (int64_t j = 0; j < per; ++j) w[(size_t) (i * per + j)] *= s;
            b[(size_t) i] = (b[(size_t) i] - mean[(size_t) i]) * s + beta[(size_t) i];
        }
    }
    return conv_host(w, b, o, c, (int) k, stride, pad);
}

void Birefnet::load_weights() {
    patch_ = conv("bb.patch_embed.proj", 4, 0);
    patch_norm_ = norm("bb.patch_embed.norm");
    const int64_t ws = window_, n = ws * ws, side = 2 * ws - 1;
    for (int i = 0;; ++i) {
        const std::string p = "bb.layers." + std::to_string(i) + ".";
        if (st_->find(p + "blocks.0.norm1.weight") == nullptr) break;
        Stage s;
        for (int j = 0;; ++j) {
            const std::string b = p + "blocks." + std::to_string(j) + ".";
            if (st_->find(b + "norm1.weight") == nullptr) break;
            Block k;
            k.n1 = norm(b + "norm1");
            k.n2 = norm(b + "norm2");
            k.qkv = lin(b + "attn.qkv");
            k.proj = lin(b + "attn.proj");
            k.fc1 = lin(b + "mlp.fc1");
            k.fc2 = lin(b + "mlp.fc2");
            k.shift = j % 2 == 1;
            // the relative position bias, [heads][query][key] of the window's positions (row-major)
            const auto table = host_f32(b + "attn.relative_position_bias_table");
            k.heads = (int64_t) table.size() / (side * side);
            std::vector<float> bias((size_t) (k.heads * n * n));
            for (int64_t a = 0; a < n; ++a)
                for (int64_t c = 0; c < n; ++c) {
                    const int64_t idx = (a / ws - c / ws + ws - 1) * side + (a % ws - c % ws + ws - 1);
                    for (int64_t h = 0; h < k.heads; ++h)
                        bias[(size_t) ((h * n + a) * n + c)] = table[(size_t) (idx * k.heads + h)];
                }
            auto* d = static_cast<float*>(alloc(bias.size() * 4));
            gpu::copy(d, bias.data(), bias.size() * 4);
            weight_bytes_ += bias.size() * 4;
            k.bias = d;
            s.blocks.push_back(k);
        }
        s.dim = s.blocks[0].n1.w != nullptr ? st_->find(p + "blocks.0.norm1.weight")->shape[0] : 0;
        if (st_->find(p + "downsample.reduction.weight") != nullptr) {
            s.merge_norm = norm(p + "downsample.norm");
            s.merge = lin(p + "downsample.reduction");
        }
        s.out_norm = norm("bb.norm" + std::to_string(i));
        stages_.push_back(std::move(s));
    }
    if (stages_.size() != 4) throw std::runtime_error(path_ + ": the backbone has " + std::to_string(stages_.size()) + " stages");
    const auto blk = [&](const std::string& name) {
        DecBlk b;
        b.in = conv_bn(name + ".conv_in", name + ".bn_in", 1, 1);
        b.out = conv_bn(name + ".conv_out", name + ".bn_out", 1, 1);
        const std::string a = name + ".dec_att.";
        for (int j = 0; j < 4; ++j) {
            const std::string d = a + (j == 0 ? std::string("aspp1") : "aspp_deforms." + std::to_string(j - 1));
            const std::string c = d + ".atrous_conv.";
            Deform& f = b.aspp[j];
            f.conv = conv_bn(c + "regular_conv", d + ".bn", 1, 0);
            f.k = f.conv.k;
            f.conv.pad = f.k / 2;
            // the offsets' and the modulator's convolutions as one: [2k^2 offsets (dy, dx per tap); k^2 modulators]
            auto wo = host_f32(c + "offset_conv.weight"), bo = host_f32(c + "offset_conv.bias");
            const auto wm = host_f32(c + "modulator_conv.weight"), bm = host_f32(c + "modulator_conv.bias");
            wo.insert(wo.end(), wm.begin(), wm.end());
            bo.insert(bo.end(), bm.begin(), bm.end());
            f.offmask = conv_host(wo, bo, 3 * (int64_t) f.k * f.k, f.conv.c, f.k, 1, f.k / 2);
        }
        b.gap = conv_bn(a + "global_avg_pool.1", a + "global_avg_pool.2", 1, 0);
        b.squeeze = conv_bn(a + "conv1", a + "bn1", 1, 0);
        return b;
    };
    squeeze_ = blk("squeeze_module.0");
    for (int k = 0; k < 4; ++k) dec_[k] = blk("decoder.decoder_block" + std::to_string(k + 1));
    for (int k = 0; k < 3; ++k) {   // the blocks of the levels 2, 3 and 4 (x1, x2, x3)
        const std::string n = std::to_string(k + 2);
        lateral_[k] = conv("decoder.lateral_block" + n + ".conv", 1, 0);
        gdt_[k] = conv_bn("decoder.gdt_convs_" + n + ".0", "decoder.gdt_convs_" + n + ".1", 1, 1);
        gdt_attn_[k] = conv("decoder.gdt_convs_attn_" + n + ".0", 1, 0);
    }
    for (int k = 0; k < 5; ++k) {
        const std::string n = "decoder.ipt_blk" + std::to_string(k + 1);
        ipt_[k].c1 = conv(n + ".conv1", 1, 1);
        ipt_[k].c2 = conv(n + ".conv_out", 1, 1);
    }
    out_ = conv("decoder.conv_out1.0", 1, 0);
}

const float* Birefnet::shift_mask(int64_t Hp, int64_t Wp) {
    for (const auto& m : masks_)
        if (m.first == Hp * 100000 + Wp) return m.second;
    // the regions of the rolled grid (the last window's rows and columns split at the shift), -100 between regions
    const int64_t ws = window_, sh = ws / 2, nwx = Wp / ws, nW = (Hp / ws) * nwx, n = ws * ws;
    const auto region = [&](int64_t v, int64_t len) { return v < len - ws ? 0 : v < len - sh ? 1 : 2; };
    std::vector<float> m((size_t) (nW * n * n));
    for (int64_t w = 0; w < nW; ++w)
        for (int64_t a = 0; a < n; ++a)
            for (int64_t c = 0; c < n; ++c) {
                const int64_t ya = (w / nwx) * ws + a / ws, xa = (w % nwx) * ws + a % ws;
                const int64_t yc = (w / nwx) * ws + c / ws, xc = (w % nwx) * ws + c % ws;
                const int ra = region(ya, Hp) * 3 + region(xa, Wp), rc = region(yc, Hp) * 3 + region(xc, Wp);
                m[(size_t) ((w * n + a) * n + c)] = ra == rc ? 0.f : -100.f;
            }
    auto* d = static_cast<float*>(alloc(m.size() * 4));
    gpu::copy(d, m.data(), m.size() * 4);
    masks_.emplace_back(Hp * 100000 + Wp, d);
    return d;
}

uint64_t Birefnet::need_bytes(int64_t size, int64_t width, int64_t height) const {
    const int64_t P1 = (size / 4) * (size / 4), C0 = embed_;
    // the first stage's attention and MLP buffers, the concatenated stages, the largest decoder block's branches,
    // the picture's planes and the alpha
    uint64_t want = (uint64_t) P1 * C0 * (4 * 2 + 2 * 4 + 4 * 4 * 2 + 4 * 2);
    want += (uint64_t) P1 * (2 * C0) * 4 * 2 + (uint64_t) P1 * (5 * kAspp) * 6 + (uint64_t) P1 * kDecInter * 4 * 8;
    want += (uint64_t) size * size * (4 * 3 + 2 * 3 + 64 * 6 + 8) + (uint64_t) width * height * 6 + (64ull << 20);
    const uint64_t have = held() - weight_bytes_;
    return (loaded_ ? 0 : file_bytes_ + file_bytes_ / 8) + (want > have ? want - have : 0);
}

void Birefnet::backbone(const uint16_t* img16, int64_t S, int slot0, float* feat[4], int64_t side[4]) {
    const int64_t ws = window_, n = ws * ws;
    int64_t H = S / 4, W = S / 4, C = embed_;
    const int64_t P0 = H * W, mlp = stages_[0].blocks[0].fc1.n / embed_;   // the MLPs' ratio (4)
    // the buffers at the first stage's size, the largest (its windows padded)
    const int64_t Hp0 = (H + ws - 1) / ws * ws, T0 = Hp0 * Hp0;
    float* x = scratch(kX, (size_t) (P0 * C * 4));
    float* y = scratch(kY, (size_t) (std::max(T0, P0) * C * 4));
    float* big = scratch(kBig, (size_t) (std::max(T0 * 3, P0 * mlp) * C * 4));
    auto* h16 = scratch_as<uint16_t>(kH16, (size_t) (P0 * C));
    auto* w16 = scratch_as<uint16_t>(kW16, (size_t) (T0 * C));
    auto* g16 = scratch_as<uint16_t>(kG16, (size_t) (P0 * mlp * C));
    auto* q16 = scratch_as<uint16_t>(kQ16, (size_t) (T0 * C));
    auto* k16 = scratch_as<uint16_t>(kK16, (size_t) (T0 * C));
    auto* v16 = scratch_as<uint16_t>(kV16, (size_t) (T0 * C));
    auto* a16 = scratch_as<uint16_t>(kA16, (size_t) (T0 * C));
    // the patches and their norm
    convolve(patch_, img16, S, S, x, kCol);
    aux_layer_norm(x, nullptr, patch_norm_.w, patch_norm_.b, P0, C, kEps, x, nullptr, s_);
    for (size_t si = 0; si < stages_.size(); ++si) {
        const Stage& st = stages_[si];
        const int64_t P = H * W, Hp = (H + ws - 1) / ws * ws, Wp = (W + ws - 1) / ws * ws, T = Hp * Wp;
        const int64_t nW = T / n;
        const float* mask = shift_mask(Hp, Wp);
        for (const Block& b : st.blocks) {
            const int64_t hd = C / b.heads, shift = b.shift ? ws / 2 : 0;
            aux_layer_norm(x, nullptr, b.n1.w, b.n1.b, P, C, kEps, nullptr, h16, s_);
            aux_window16(h16, H, W, C, ws, shift, Hp, Wp, w16, s_);   // the padding zero, as Swin pads after the norm
            product(w16, b.qkv, big, T);
            aux_split3_16(big, T, C, q16, k16, v16, s_);
            QiSdpa a;
            a.q = q16, a.k = k16, a.v = v16, a.out = a16;
            a.batch = nW, a.heads = b.heads, a.lq = a.lk = n, a.hd = hd;
            a.q_row = a.k_row = a.v_row = a.o_row = C;
            a.q_batch = a.k_batch = a.v_batch = a.o_batch = n * C;
            a.scale = 1.f / std::sqrt((float) hd);
            a.bias = b.bias;
            if (shift) a.mask = mask, a.mask_n = nW;
            qi_sdpa_ex(a, s_);
            product(a16, b.proj, y, T);
            aux_unwindow_add(x, y, H, W, C, ws, shift, Hp, Wp, s_);
            aux_layer_norm(x, nullptr, b.n2.w, b.n2.b, P, C, kEps, nullptr, h16, s_);
            product(h16, b.fc1, big, P);
            qi_gelu_erf16(big, P * b.fc1.n, g16, s_);
            product(g16, b.fc2, y, P);
            qi_axpy(x, y, 1.f, P * C, s_);
        }
        feat[si] = scratch(slot0 + (int) si, (size_t) (P * C * 4));
        side[si] = H;
        aux_layer_norm(x, nullptr, st.out_norm.w, st.out_norm.b, P, C, kEps, feat[si], nullptr, s_);
        if (st.merge.w == nullptr) break;
        // patch merging: each 2 x 2 of the grid as one row of 4C, normed, to 2C
        float* m = scratch(kMerge, (size_t) (P * C * 4));
        aux_merge2(x, H, W, C, m, s_);
        aux_layer_norm(m, nullptr, st.merge_norm.w, st.merge_norm.b, P / 4, 4 * C, kEps, nullptr, h16, s_);
        H /= 2, W /= 2, C *= 2;
        product(h16, st.merge, x, H * W);
    }
}

void Birefnet::deform(const Deform& d, const uint16_t* x16, int64_t H, int64_t W, float* y, int64_t ldy) {
    const int64_t P = H * W, k2 = (int64_t) d.k * d.k, C = d.conv.c, K = k2 * C, O = d.conv.o;
    float* om = scratch(kOm, (size_t) (P * 3 * k2 * 4));
    convolve(d.offmask, x16, H, W, om, kCol);
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(P, (64ll << 20) / (K * 2)));
    auto* col = scratch_as<uint16_t>(kCol, (size_t) (rows * K));
    for (int64_t p0 = 0; p0 < P; p0 += rows) {
        const int64_t nr = std::min(rows, P - p0);
        aux_deform_col16(x16, H, W, C, d.k, d.k / 2, om, p0, nr, col, s_);
        gemm_.f16(col, d.conv.w, y + p0 * ldy, nr, O, K, ldy);
    }
    aux_bias_act(y, P, O, ldy, d.conv.b, true, s_);
}

void Birefnet::dec_block(const DecBlk& b, const uint16_t* x16, int64_t H, int64_t W, float* y) {
    const int64_t P = H * W, inter = b.in.o, C5 = 5 * kAspp;
    float* t = scratch(kT, (size_t) (P * inter * 4));
    auto* t16 = scratch_as<uint16_t>(kT16, (size_t) (P * inter));
    convolve(b.in, x16, H, W, t, kCol);
    aux_relu(t, P * inter, t, t16, s_);
    float* cat = scratch(kAspp5, (size_t) (P * C5 * 4));
    for (int j = 0; j < 4; ++j) deform(b.aspp[j], t16, H, W, cat + j * kAspp, C5);
    // the pooled branch: the pixels' mean through a 1 x 1 convolution, spread back over the grid
    float* gap = scratch(kGap, (size_t) (std::max(inter, kAspp) * 4 * 2));
    auto* gap16 = scratch_as<uint16_t>(kGap16, (size_t) inter);
    aux_col_mean(t, P, inter, gap, s_);
    kernels::f32_to_f16_bulk(gap, gap16, inter, s_);
    float* g = gap + std::max(inter, kAspp);
    product(gap16, Lin{b.gap.w, b.gap.b, b.gap.o, b.gap.c}, g, 1);
    aux_relu(g, kAspp, g, nullptr, s_);
    aux_resize_into(g, 1, 1, kAspp, cat, H, W, C5, 4 * kAspp, true, false, s_);
    auto* cat16 = scratch_as<uint16_t>(kAspp16, (size_t) (P * C5));
    kernels::f32_to_f16_bulk(cat, cat16, P * C5, s_);
    convolve(b.squeeze, cat16, H, W, t, kCol);
    aux_relu(t, P * b.squeeze.o, nullptr, t16, s_);
    convolve(b.out, t16, H, W, y, kCol);
}

void Birefnet::ipt(const Ipt& p, const float* img, int64_t S, int64_t side, float* y, int64_t ldy, int64_t c0) {
    const int64_t hg = S / side, P = side * side, cin = 3 * hg * hg;
    auto* pt16 = scratch_as<uint16_t>(kPatch16, (size_t) (P * cin));
    aux_image_patches16(img, S, hg, pt16, s_);
    float* t = scratch(kIpt, (size_t) (P * p.c1.o * 4));
    auto* t16 = scratch_as<uint16_t>(kT16, (size_t) (P * std::max(p.c1.o, kDecInter)));
    convolve(p.c1, pt16, side, side, t, kCol);
    kernels::f32_to_f16_bulk(t, t16, P * p.c1.o, s_);
    float* u = scratch(kIpt2, (size_t) (P * p.c2.o * 4));
    convolve(p.c2, t16, side, side, u, kCol);
    aux_resize_into(u, side, side, p.c2.o, y, side, side, ldy, c0, true, false, s_);
}

std::vector<uint8_t> Birefnet::run(const QiRef& image, int64_t S, Times* tm) {
    Times local;
    Times& tt = tm != nullptr ? *tm : local;
    if (S % 64 != 0) throw std::invalid_argument("image engine (BiRefNet): the size is not a multiple of 64");
    load();
    auto t0 = std::chrono::steady_clock::now();
    // the picture at S x S (Pillow's bilinear), ImageNet's normalization; at S / 2 too (bilinear, corners aligned)
    const QiRef r = pillow_resize(image, S, S, Filter::triangle);
    auto* rgba = scratch_as<uint8_t>(kRgba, r.rgba.size());
    gpu::copy(rgba, r.rgba.data(), r.rgba.size());
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    float* img = scratch(kImg, (size_t) (S * S * 3 * 4));
    aux_rgb_norm(rgba, S, S, mean, stdv, img, s_);
    auto* img16 = scratch_as<uint16_t>(kImg16, (size_t) (S * S * 3));
    kernels::f32_to_f16_bulk(img, img16, S * S * 3, s_);
    const int64_t S2 = S / 2;
    float* half = scratch(kHalf, (size_t) (S2 * S2 * 3 * 4));
    aux_resize_into(img, S, S, 3, half, S2, S2, 3, 0, true, false, s_);
    auto* half16 = scratch_as<uint16_t>(kHalf16, (size_t) (S2 * S2 * 3));
    kernels::f32_to_f16_bulk(half, half16, S2 * S2 * 3, s_);
    // both passes, each stage of the half's resized onto the full's and the two concatenated
    float* full[4];
    float* hf[4];
    int64_t side[4], hside[4], dim[4];
    backbone(img16, S, kF0, full, side);
    backbone(half16, S2, kH0, hf, hside);
    float* x[4];
    for (int i = 0; i < 4; ++i) {
        dim[i] = stages_[(size_t) i].dim;
        const int64_t P = side[i] * side[i];
        x[i] = scratch(kC1 + i, (size_t) (P * 2 * dim[i] * 4));
        aux_resize_into(full[i], side[i], side[i], dim[i], x[i], side[i], side[i], 2 * dim[i], 0, true, false, s_);
        aux_resize_into(hf[i], hside[i], hside[i], dim[i], x[i], side[i], side[i], 2 * dim[i], dim[i], true, false, s_);
        host(("x" + std::to_string(i + 1)).c_str(), x[i], (size_t) (P * 2 * dim[i]));
    }
    gpu::stream_sync(s_);
    tt.backbone = since(t0);
    t0 = std::chrono::steady_clock::now();
    // the coarsest level with the three others resized onto it, squeezed
    const int64_t s4 = side[3], P4 = s4 * s4;
    int64_t ctot = 0;
    for (int i = 0; i < 4; ++i) ctot += 2 * dim[i];
    float* cat = scratch(kCat, (size_t) (P4 * ctot * 4));
    for (int i = 0, c0 = 0; i < 4; c0 += 2 * (int) dim[i], ++i)
        aux_resize_into(x[i], side[i], side[i], 2 * dim[i], cat, s4, s4, ctot, c0, true, false, s_);
    auto* cat16 = scratch_as<uint16_t>(kCat16, (size_t) (P4 * ctot));
    kernels::f32_to_f16_bulk(cat, cat16, P4 * ctot, s_);
    float* x4 = scratch(kSq, (size_t) (P4 * squeeze_.out.o * 4));
    dec_block(squeeze_, cat16, s4, s4, x4);
    host("squeeze", x4, (size_t) (P4 * squeeze_.out.o));
    // the decoder, coarse to fine: level 4 (x4 and the picture's pixels), then 3, 2, 1, each from the coarser one's
    // output gated by its gradient branch, plus its stage's lateral convolution and the pixels
    float* p = nullptr;
    int64_t ps = 0, pc = 0;
    for (int lv = 3; lv >= 0; --lv) {
        const int64_t sd = side[lv], P = sd * sd;
        const DecBlk& b = dec_[lv];
        const int64_t cin = b.in.c, ci = ipt_[lv + 1].c2.o, cmain = cin - ci;
        float* in = scratch(kCat, (size_t) (P * cin * 4));
        if (lv == 3) {
            aux_resize_into(x4, sd, sd, cmain, in, sd, sd, cin, 0, true, false, s_);
        } else {
            // the lateral 1 x 1 convolution of the stage, then the coarser output resized onto it and added
            const Conv& l = lateral_[lv];
            auto* xs16 = scratch_as<uint16_t>(kCat16, (size_t) (P * 2 * dim[lv]));
            kernels::f32_to_f16_bulk(x[lv], xs16, P * 2 * dim[lv], s_);
            gemm_.f16(xs16, l.w, in, P, l.o, l.c, cin);
            aux_bias_act(in, P, l.o, cin, l.b, false, s_);
            aux_resize_into(p, ps, ps, pc, in, sd, sd, cin, 0, true, true, s_);
        }
        ipt(ipt_[lv + 1], img, S, sd, in, cin, cmain);
        auto* in16 = scratch_as<uint16_t>(kCat16, (size_t) (P * cin));
        kernels::f32_to_f16_bulk(in, in16, P * cin, s_);
        float* out = scratch(lv % 2 == 0 ? kP : kGdt, (size_t) (P * b.out.o * 4));
        dec_block(b, in16, sd, sd, out);
        if (lv > 0) {
            // the gradient branch: a 3 x 3 convolution to 16 (its norm folded), ReLU, a 1 x 1 to one logit per pixel,
            // whose sigmoid scales the block's output
            auto* o16 = scratch_as<uint16_t>(kP16, (size_t) (P * b.out.o));
            kernels::f32_to_f16_bulk(out, o16, P * b.out.o, s_);
            const Conv& g = gdt_[lv - 1];
            float* gt = scratch(kAttn, (size_t) (P * (g.o + 1) * 4));
            auto* gt16 = scratch_as<uint16_t>(kGdt16, (size_t) (P * g.o));
            convolve(g, o16, sd, sd, gt, kCol);
            aux_relu(gt, P * g.o, nullptr, gt16, s_);
            float* logit = gt + P * g.o;
            convolve(gdt_attn_[lv - 1], gt16, sd, sd, logit, kCol);
            aux_scale_by_sigmoid(out, P, b.out.o, logit, s_);
        }
        host(("p" + std::to_string(lv + 1)).c_str(), out, (size_t) (P * b.out.o));
        p = out, ps = sd, pc = b.out.o;
    }
    // the last 1 x 1 convolution: its weights on the decoder's channels taken at the finest level and resized (both
    // linear, so in either order), then the picture's own pixels' branch at S x S with the rest of its weights
    {
        const int64_t P1 = ps * ps, ci = ipt_[0].c2.o;
        if (out_.c != pc + ci) throw std::runtime_error("image engine (BiRefNet): the last convolution's width");
        auto* p16 = scratch_as<uint16_t>(kP16, (size_t) (P1 * pc));
        kernels::f32_to_f16_bulk(p, p16, P1 * pc, s_);
        float* lo = scratch(kLogit, (size_t) (P1 * 4 + S * S * (ci + 1) * 4));
        gemm_.f16(p16, out_.w, lo, P1, 1, pc, 0);   // the first pc of each weight row: row stride out_.c
        float* up = scratch(kLogitUp, (size_t) (S * S * 4));
        aux_resize_into(lo, ps, ps, 1, up, S, S, 1, 0, true, false, s_);
        float* pix = lo + P1;
        ipt(ipt_[0], img, S, S, pix, ci, 0);
        auto* pix16 = scratch_as<uint16_t>(kPatch16, (size_t) (S * S * ci));
        kernels::f32_to_f16_bulk(pix, pix16, S * S * ci, s_);
        gemm_.f16(pix16, out_.w + pc, up, S * S, 1, ci, 0, false, true);
        aux_bias_act(up, S * S, 1, 1, out_.b, false, s_);
        host("logits", up, (size_t) (S * S));
    }
    gpu::stream_sync(s_);
    tt.decoder = since(t0);
    t0 = std::chrono::steady_clock::now();
    const int64_t W = image.width, H = image.height;
    auto* alpha = scratch_as<uint8_t>(kAlpha, (size_t) (W * H));
    aux_alpha_bicubic(scratch(kLogitUp, (size_t) (S * S * 4)), S, S, W, H, alpha, s_);
    std::vector<uint8_t> outv((size_t) (W * H));
    gpu::stream_sync(s_);
    gpu::copy(outv.data(), alpha, outv.size());
    tt.mask = since(t0);
    return outv;
}

}  // namespace strata::image
