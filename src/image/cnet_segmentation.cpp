// SPDX-FileCopyrightText: 2022 SenseTime X-Lab
// SPDX-FileCopyrightText: 2020 OpenMMLab
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/image/cnet_segmentation.cpp - UniFormer Small with UPerNet, on the ADE20K checkpoint's 150 classes.  The
// backbone's local blocks are convolutions; its global blocks use the shared attention and products.  UPerNet's
// pyramid pooling and top-down feature maps end in the ADE20K palette (OpenMMLab's ADE20KDataset).

#include "strata/image/cnet_models.hpp"

#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace cnet_segmentation_k {
using kernels::f16_from_f32;
using kernels::f32_from_f16;

void input(sycl::queue& q, const uint8_t* rgb, int64_t n, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i % 8;
        const float mean = c == 0 ? 123.675f : c == 1 ? 116.28f : 103.53f;
        const float sd = c == 0 ? 58.395f : c == 1 ? 57.12f : 57.375f;
        out[i] = c < 3 ? f16_from_f32(((float) rgb[p * 3 + c] - mean) / sd) : (uint16_t) 0;
    });
}

void widen(sycl::queue& q, const uint16_t* in, int64_t n, float* out) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id[0]] = f32_from_f16(in[id[0]]); });
}

void batch(sycl::queue& q, const uint16_t* in, int64_t n, int64_t C, const float* scale, const float* bias,
           uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        out[i] = f16_from_f32(f32_from_f16(in[i]) * scale[i % C] + bias[i % C]);
    });
}

// Adaptive average pooling's bin boundaries are floor(i * H / OH), ceil((i+1) * H / OH).
void pool(sycl::queue& q, const uint16_t* in, int64_t H, int64_t W, int64_t C, int64_t OH, int64_t OW,
          uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, oy = p / OW, ox = p % OW;
        const int64_t y0 = oy * H / OH, y1 = ((oy + 1) * H + OH - 1) / OH;
        const int64_t x0 = ox * W / OW, x1 = ((ox + 1) * W + OW - 1) / OW;
        float sum = 0.f;
        for (int64_t y = y0; y < y1; ++y)
            for (int64_t x = x0; x < x1; ++x) sum += f32_from_f16(in[(y * W + x) * C + c]);
        out[i] = f16_from_f32(sum / (float) ((y1 - y0) * (x1 - x0)));
    });
}

void colors(sycl::queue& q, const uint16_t* logits, int64_t n, int64_t C, const float* palette, uint8_t* rgb) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto p = static_cast<int64_t>(id[0]);
        int winner = 0;
        float best = f32_from_f16(logits[p * C]);
        for (int c = 1; c < 150; ++c) {
            const float v = f32_from_f16(logits[p * C + c]);
            if (v > best) best = v, winner = c;
        }
        for (int c = 0; c < 3; ++c) rgb[p * 3 + c] = (uint8_t) palette[winner * 3 + c];
    });
}
}  // namespace cnet_segmentation_k

void UniformerSeg::load_weights() {
    if (st_->find("backbone.patch_embed1.proj.weight") == nullptr || st_->find("decode_head.conv_seg.weight") == nullptr)
        throw std::runtime_error(path_ + ": not the UniFormer UPerNet segmentation model");
    const auto batch = [&](const std::string& n) {
        const auto g = host_f32(n + ".weight"), b = host_f32(n + ".bias");
        const auto mean = host_f32(n + ".running_mean"), var = host_f32(n + ".running_var");
        if (g.size() != b.size() || g.size() != mean.size() || g.size() != var.size())
            throw std::runtime_error(path_ + ": invalid batch norm " + n);
        std::vector<float> scale(g.size()), bias(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            scale[i] = g[i] / std::sqrt(var[i] + 1e-5f);
            bias[i] = b[i] - mean[i] * scale[i];
        }
        return Batch{upload(scale), upload(bias)};
    };
    const int counts[] = {3, 4, 8, 3};
    for (int i = 0; i < 4; ++i) {
        Stage& s = stages_[i];
        const std::string p = "backbone.patch_embed" + std::to_string(i + 1) + ".";
        s.patch = mconv(p + "proj.weight", p + "proj.bias", i == 0 ? 4 : 2, 0);
        s.patch_norm = norm(p + "norm");
        s.output_norm = norm("backbone.norm" + std::to_string(i + 1));
        s.blocks.clear();
        for (int j = 0; j < counts[i]; ++j) {
            const std::string b = "backbone.blocks" + std::to_string(i + 1) + "." + std::to_string(j) + ".";
            Block k;
            k.pos = mconv(b + "pos_embed.weight", b + "pos_embed.bias", 1, 1, "", 1e-5f, 0, true);
            if (i < 2) {
                k.bn1 = batch(b + "norm1");
                k.bn2 = batch(b + "norm2");
                k.conv1 = mconv(b + "conv1.weight", b + "conv1.bias", 1, 0);
                k.depthwise = mconv(b + "attn.weight", b + "attn.bias", 1, 2, "", 1e-5f, 0, true);
                k.conv2 = mconv(b + "conv2.weight", b + "conv2.bias", 1, 0);
            } else {
                k.norm1 = norm(b + "norm1");
                k.norm2 = norm(b + "norm2");
                k.qkv = lin(b + "attn.qkv");
                k.conv2 = mconv(b + "attn.proj.weight", b + "attn.proj.bias", 1, 0);
            }
            k.fc1 = mconv(b + "mlp.fc1.weight", b + "mlp.fc1.bias", 1, 0);
            k.fc2 = mconv(b + "mlp.fc2.weight", b + "mlp.fc2.bias", 1, 0);
            s.blocks.push_back(k);
        }
    }
    const auto cm = [&](const std::string& p, int pad) {
        return mconv(p + ".conv.weight", "", 1, pad, p + ".bn");
    };
    for (int i = 0; i < 4; ++i) psp_[i] = cm("decode_head.psp_modules." + std::to_string(i) + ".1", 0);
    bottleneck_ = cm("decode_head.bottleneck", 1);
    for (int i = 0; i < 3; ++i) {
        lateral_[i] = cm("decode_head.lateral_convs." + std::to_string(i), 0);
        fpn_[i] = cm("decode_head.fpn_convs." + std::to_string(i), 1);
    }
    fpn_bottleneck_ = cm("decode_head.fpn_bottleneck", 1);
    classes_ = mconv("decode_head.conv_seg.weight", "decode_head.conv_seg.bias", 1, 0);
    if (classes_.out != 150) throw std::runtime_error(path_ + ": expected 150 ADE20K classes");
    // OpenMMLab's ADE20KDataset palette, in the class order of this checkpoint.
    static const float palette[][3] = {
    {120, 120, 120},
    {180, 120, 120},
    {6, 230, 230},
    {80, 50, 50},
    {4, 200, 3},
    {120, 120, 80},
    {140, 140, 140},
    {204, 5, 255},
    {230, 230, 230},
    {4, 250, 7},
    {224, 5, 255},
    {235, 255, 7},
    {150, 5, 61},
    {120, 120, 70},
    {8, 255, 51},
    {255, 6, 82},
    {143, 255, 140},
    {204, 255, 4},
    {255, 51, 7},
    {204, 70, 3},
    {0, 102, 200},
    {61, 230, 250},
    {255, 6, 51},
    {11, 102, 255},
    {255, 7, 71},
    {255, 9, 224},
    {9, 7, 230},
    {220, 220, 220},
    {255, 9, 92},
    {112, 9, 255},
    {8, 255, 214},
    {7, 255, 224},
    {255, 184, 6},
    {10, 255, 71},
    {255, 41, 10},
    {7, 255, 255},
    {224, 255, 8},
    {102, 8, 255},
    {255, 61, 6},
    {255, 194, 7},
    {255, 122, 8},
    {0, 255, 20},
    {255, 8, 41},
    {255, 5, 153},
    {6, 51, 255},
    {235, 12, 255},
    {160, 150, 20},
    {0, 163, 255},
    {140, 140, 140},
    {250, 10, 15},
    {20, 255, 0},
    {31, 255, 0},
    {255, 31, 0},
    {255, 224, 0},
    {153, 255, 0},
    {0, 0, 255},
    {255, 71, 0},
    {0, 235, 255},
    {0, 173, 255},
    {31, 0, 255},
    {11, 200, 200},
    {255, 82, 0},
    {0, 255, 245},
    {0, 61, 255},
    {0, 255, 112},
    {0, 255, 133},
    {255, 0, 0},
    {255, 163, 0},
    {255, 102, 0},
    {194, 255, 0},
    {0, 143, 255},
    {51, 255, 0},
    {0, 82, 255},
    {0, 255, 41},
    {0, 255, 173},
    {10, 0, 255},
    {173, 255, 0},
    {0, 255, 153},
    {255, 92, 0},
    {255, 0, 255},
    {255, 0, 245},
    {255, 0, 102},
    {255, 173, 0},
    {255, 0, 20},
    {255, 184, 184},
    {0, 31, 255},
    {0, 255, 61},
    {0, 71, 255},
    {255, 0, 204},
    {0, 255, 194},
    {0, 255, 82},
    {0, 10, 255},
    {0, 112, 255},
    {51, 0, 255},
    {0, 194, 255},
    {0, 122, 255},
    {0, 255, 163},
    {255, 153, 0},
    {0, 255, 10},
    {255, 112, 0},
    {143, 255, 0},
    {82, 0, 255},
    {163, 255, 0},
    {255, 235, 0},
    {8, 184, 170},
    {133, 0, 255},
    {0, 255, 92},
    {184, 0, 255},
    {255, 0, 31},
    {0, 184, 255},
    {0, 214, 255},
    {255, 0, 112},
    {92, 255, 0},
    {0, 224, 255},
    {112, 224, 255},
    {70, 184, 160},
    {163, 0, 255},
    {153, 0, 255},
    {71, 255, 0},
    {255, 0, 163},
    {255, 204, 0},
    {255, 0, 143},
    {0, 255, 235},
    {133, 255, 0},
    {255, 0, 235},
    {245, 0, 255},
    {255, 0, 122},
    {255, 245, 0},
    {10, 190, 212},
    {214, 255, 0},
    {0, 204, 255},
    {20, 0, 255},
    {255, 255, 0},
    {0, 153, 255},
    {0, 41, 255},
    {0, 255, 204},
    {41, 0, 255},
    {41, 255, 0},
    {173, 0, 255},
    {0, 245, 255},
    {71, 0, 255},
    {122, 0, 255},
    {0, 255, 184},
    {0, 92, 255},
    {184, 255, 0},
    {0, 133, 255},
    {255, 214, 0},
    {25, 194, 194},
    {102, 255, 0},
    {92, 0, 255}
    };
    std::vector<float> colors;
    for (const auto& color : palette) colors.insert(colors.end(), color, color + 3);
    palette_ = upload(colors);
}

uint64_t UniformerSeg::need_bytes(int64_t W, int64_t H) const {
    // The retained backbone maps and UPerNet's four 512-channel feature maps, with convolution workspaces.
    return (uint64_t) (W * H) * 16000 + (256ull << 20);
}

std::vector<uint8_t> UniformerSeg::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H) {
    if (W <= 0 || H <= 0 || W % 32 || H % 32 || rgb.size() != (size_t) (W * H * 3))
        throw std::invalid_argument("UniFormer segmentation: an RGB picture with sides divisible by 32");
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(H, W, 8);
    cnet_segmentation_k::input(q, src, H * W, x.p);
    const auto normal = [&](const Map& in, const Norm& n, float eps) {
        Map out = map(in.h, in.w, in.c);
        auto* wide = buf<float>((size_t) (in.h * in.w * in.c));
        cnet_segmentation_k::widen(q, in.p, in.h * in.w * in.c, wide);
        aux_layer_norm(wide, nullptr, n.w, n.b, in.h * in.w, in.c, eps, nullptr, out.p, s_);
        return out;
    };
    const auto bn = [&](const Map& in, const Batch& b) {
        Map out = map(in.h, in.w, in.c);
        cnet_segmentation_k::batch(q, in.p, in.h * in.w, in.c, b.scale, b.bias, out.p);
        return out;
    };
    Map taps[4];
    for (int i = 0; i < 4; ++i) {
        const Stage& st = stages_[i];
        x = normal(AuxMapNet::run(st.patch, x), st.patch_norm, 1e-5f);
        for (const Block& b : st.blocks) {
            x = AuxMapNet::run(b.pos, x, ImageAct::none, &x);
            Map y;
            if (i < 2) {
                y = AuxMapNet::run(b.conv1, bn(x, b.bn1));
                y = AuxMapNet::run(b.depthwise, y);
            } else {
                const Map n = normal(x, b.norm1, 1e-6f);
                const int64_t T = n.h * n.w, C = n.c;
                auto* qkv = buf<float>((size_t) (T * 3 * C));
                auto* qq = buf<uint16_t>((size_t) (T * C));
                auto* kk = buf<uint16_t>((size_t) (T * C));
                auto* vv = buf<uint16_t>((size_t) (T * C));
                y = map(n.h, n.w, C);
                product(n.p, b.qkv, qkv, T);
                aux_split3_16(qkv, T, C, qq, kk, vv, s_);
                QiSdpa att;
                att.q = qq, att.k = kk, att.v = vv, att.out = y.p;
                att.heads = C / 64, att.lq = att.lk = T, att.hd = 64;
                att.q_row = att.k_row = att.v_row = att.o_row = C;
                att.q_batch = att.k_batch = att.v_batch = att.o_batch = T * C;
                att.scale = 1.f / 8.f;
                qi_sdpa_ex(att, s_);
            }
            x = AuxMapNet::run(b.conv2, y, ImageAct::none, &x);
            y = i < 2 ? bn(x, b.bn2) : normal(x, b.norm2, 1e-6f);
            y = AuxMapNet::run(b.fc1, y, ImageAct::gelu);
            x = AuxMapNet::run(b.fc2, y, ImageAct::none, &x);
        }
        taps[i] = normal(x, st.output_norm, 1e-6f);
    }
    std::vector<Map> pyramid{taps[3]};
    pyramid.reserve(5);
    const int scales[] = {1, 2, 3, 6};
    for (int i = 0; i < 4; ++i) {
        Map pooled = map(scales[i], scales[i], taps[3].c);
        cnet_segmentation_k::pool(q, taps[3].p, taps[3].h, taps[3].w, taps[3].c, scales[i], scales[i], pooled.p);
        pyramid.push_back(resize(AuxMapNet::run(psp_[i], pooled, ImageAct::relu), taps[3].h, taps[3].w, false));
    }
    Map laterals[4];
    for (int i = 0; i < 3; ++i) laterals[i] = AuxMapNet::run(lateral_[i], taps[i], ImageAct::relu);
    laterals[3] = AuxMapNet::run(bottleneck_, concat(pyramid), ImageAct::relu);
    for (int i = 3; i > 0; --i)
        laterals[i - 1] = add(laterals[i - 1], resize(laterals[i], laterals[i - 1].h, laterals[i - 1].w, false));
    std::vector<Map> features;
    features.reserve(4);
    for (int i = 0; i < 3; ++i) features.push_back(AuxMapNet::run(fpn_[i], laterals[i], ImageAct::relu));
    features.push_back(laterals[3]);
    for (int i = 1; i < 4; ++i) features[(size_t) i] = resize(features[(size_t) i], features[0].h, features[0].w, false);
    x = AuxMapNet::run(fpn_bottleneck_, concat(features), ImageAct::relu);
    x = resize(AuxMapNet::run(classes_, x), H, W, false);
    auto* result = buf<uint8_t>((size_t) (W * H * 3));
    cnet_segmentation_k::colors(q, x.p, W * H, x.c, palette_, result);
    std::vector<uint8_t> out((size_t) (W * H * 3));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), result, out.size());
    return out;
}

}  // namespace strata::image
