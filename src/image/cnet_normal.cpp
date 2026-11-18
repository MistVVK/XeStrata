// SPDX-FileCopyrightText: 2021 Gwangbin Bae
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/image/cnet_normal.cpp - BAE's surface normals: the EfficientNet-B5 encoder and the BN decoder in
// baegwangbin/surface_normal_uncertainty, using the ScanNet weights' tensor names. The encoder's inverted blocks use
// TensorFlow SAME padding, squeeze-and-excitation and SiLU. The decoder upsamples with aligned corners and refines
// the normalized prediction at each scale. Products and convolutions go through AuxMapNet.

#include "strata/image/cnet_models.hpp"

#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace cnet_normal_k {

using kernels::f16_from_f32;
using kernels::f32_from_f16;

void input(sycl::queue& q, const uint8_t* rgb, int64_t n, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i % 8;
        const float mean = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f;
        const float sd = c == 0 ? 0.229f : c == 1 ? 0.224f : 0.225f;
        out[i] = c < 3 ? f16_from_f32(((float) rgb[p * 3 + c] / 255.f - mean) / sd) : (uint16_t) 0;
    });
}

void narrow(sycl::queue& q, const float* in, int64_t n, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id[0]] = f16_from_f32(in[id[0]]); });
}

void widen(sycl::queue& q, const uint16_t* in, int64_t n, float* out) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id[0]] = f32_from_f16(in[id[0]]); });
}

void leaky(sycl::queue& q, uint16_t* x, int64_t n) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = f32_from_f16(x[id[0]]);
        x[id[0]] = f16_from_f32(v >= 0.f ? v : 0.01f * v);
    });
}

void normalize(sycl::queue& q, uint16_t* x, int64_t n, int64_t C) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto p = static_cast<int64_t>(id[0]);
        const float a = f32_from_f16(x[p * C]), b = f32_from_f16(x[p * C + 1]), c = f32_from_f16(x[p * C + 2]);
        const float k = f32_from_f16(x[p * C + 3]);
        const float d = sycl::sqrt(a * a + b * b + c * c) + 1e-10f;
        x[p * C] = f16_from_f32(a / d);
        x[p * C + 1] = f16_from_f32(b / d);
        x[p * C + 2] = f16_from_f32(c / d);
        x[p * C + 3] = f16_from_f32((k > 0.f ? k : sycl::exp(k) - 1.f) + 1.01f);
        for (int64_t j = 4; j < C; ++j) x[p * C + j] = 0;
    });
}

void output(sycl::queue& q, const uint16_t* x, int64_t n, int64_t C, uint8_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * 3)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = (f32_from_f16(x[(i / 3) * C + i % 3]) + 1.f) * 127.5f;
        out[i] = (uint8_t) sycl::clamp(v, 0.f, 255.f);
    });
}

}  // namespace cnet_normal_k

void BaeNormal::load_weights() {
    const std::string e = "encoder.original_model.";
    if (st_->find(e + "conv_stem.weight") == nullptr || st_->find("decoder.out_conv_res1.6.weight") == nullptr)
        throw std::runtime_error(path_ + ": not the BAE ScanNet normal model");
    constexpr float tf_eps = 1e-3f;
    stem_ = mconv(e + "conv_stem.weight", "", 2, 0, e + "bn1", tf_eps);
    // The decoder takes encoder feature 11, immediately after conv_head and before bn2 / act2.
    head_ = mconv(e + "conv_head.weight", "", 1, 0);
    for (int stage = 0; stage < 7; ++stage) {
        blocks_[stage].clear();
        for (int i = 0;; ++i) {
            const std::string p = e + "blocks." + std::to_string(stage) + "." + std::to_string(i) + ".";
            if (st_->find(p + "conv_dw.weight") == nullptr) break;
            Block b;
            b.has_expand = st_->find(p + "conv_pwl.weight") != nullptr;
            b.same_pad = i == 0 && (stage == 1 || stage == 2 || stage == 3 || stage == 5);
            const SafeTensor* dw = st_->find(p + "conv_dw.weight");
            const int k = (int) dw->shape.at(2), stride = b.same_pad ? 2 : 1;
            if (b.has_expand) b.expand = mconv(p + "conv_pw.weight", "", 1, 0, p + "bn1", tf_eps);
            b.depthwise = mconv(p + "conv_dw.weight", "", stride, b.same_pad ? 0 : k / 2,
                                p + (b.has_expand ? "bn2" : "bn1"), tf_eps, 0, true);
            b.reduce = mconv(p + "se.conv_reduce.weight", p + "se.conv_reduce.bias", 1, 0);
            b.gate = mconv(p + "se.conv_expand.weight", p + "se.conv_expand.bias", 1, 0);
            const std::string proj = b.has_expand ? "conv_pwl" : "conv_pw";
            b.project = mconv(p + proj + ".weight", "", 1, 0, p + (b.has_expand ? "bn3" : "bn2"), tf_eps);
            const int64_t in = b.has_expand ? b.expand.in : b.depthwise.in;
            b.residual = !b.same_pad && in == b.project.out;
            blocks_[stage].push_back(b);
        }
    }
    const int counts[] = {3, 5, 5, 7, 7, 9, 3};
    for (int i = 0; i < 7; ++i)
        if (blocks_[i].size() != (size_t) counts[i]) throw std::runtime_error(path_ + ": not EfficientNet-B5");
    bottleneck_ = mconv("decoder.conv2.weight", "decoder.conv2.bias", 1, 0);
    initial_ = mconv("decoder.out_conv_res8.weight", "decoder.out_conv_res8.bias", 1, 1);
    for (int i = 0; i < 4; ++i) {
        const std::string p = "decoder.up" + std::to_string(i + 1) + "._net.";
        up_[i].first = mconv(p + "0.weight", p + "0.bias", 1, 1, p + "1");
        up_[i].second = mconv(p + "3.weight", p + "3.bias", 1, 1, p + "4");
    }
    const int resolutions[] = {4, 2, 1};
    for (int i = 0; i < 3; ++i) {
        const std::string p = "decoder.out_conv_res" + std::to_string(resolutions[i]) + ".";
        for (int j = 0; j < 4; ++j) {
            const std::string name = p + std::to_string(2 * j);
            const SafeTensor* t = st_->find(name + ".weight");
            if (t == nullptr || t->shape.size() != 3 || t->shape[2] != 1)
                throw std::runtime_error(path_ + ": not a BAE refinement layer: " + name);
            refine_[i][j] = mconv_host(host_f32(name + ".weight"), host_f32(name + ".bias"),
                                        t->shape[0], t->shape[1], 1, 1, 0);
        }
    }
}

uint64_t BaeNormal::need_bytes(int64_t W, int64_t H) const {
    // The maps retained by AuxMapNet through the encoder and decoder, including the refinement at full size.
    return (uint64_t) (W * H) * 16000 + (256ull << 20);
}

std::vector<uint8_t> BaeNormal::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H) {
    if (W <= 0 || H <= 0 || W % 32 || H % 32 || rgb.size() != (size_t) (W * H * 3))
        throw std::invalid_argument("BAE normal: an RGB picture with sides divisible by 32");
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(H, W, 8);
    cnet_normal_k::input(q, src, W * H, x.p);
    x = AuxMapNet::run(stem_, pad(x, 0, 0, 1, 1), ImageAct::silu);
    Map taps[7];
    for (int stage = 0; stage < 7; ++stage) {
        for (const Block& b : blocks_[stage]) {
            const Map residual = x;
            if (b.has_expand) x = AuxMapNet::run(b.expand, x, ImageAct::silu);
            if (b.same_pad) {
                const int64_t left = b.depthwise.k / 2 - 1, right = b.depthwise.k / 2;
                x = pad(x, left, left, right, right);
            }
            x = AuxMapNet::run(b.depthwise, x, ImageAct::silu);
            float* mean = buf<float>((size_t) x.c);
            aux_col_mean16(x.p, x.h * x.w, x.c, mean, s_);
            Map pooled = map(1, 1, x.c);
            cnet_normal_k::narrow(q, mean, x.c, pooled.p);
            Map gate = AuxMapNet::run(b.reduce, pooled, ImageAct::silu);
            gate = AuxMapNet::run(b.gate, gate, ImageAct::sigmoid);
            float* scale = buf<float>((size_t) gate.c);
            cnet_normal_k::widen(q, gate.p, gate.c, scale);
            aux_scale_channels16(x.p, x.h * x.w, x.c, scale, s_);
            x = AuxMapNet::run(b.project, x, ImageAct::none, b.residual ? &residual : nullptr);
        }
        taps[stage] = x;
    }
    x = AuxMapNet::run(head_, x);
    x = AuxMapNet::run(bottleneck_, x);
    const int skip[] = {4, 2, 1, 0};
    Map dec[4];
    for (int i = 0; i < 4; ++i) {
        const Map& t = taps[skip[i]];
        x = concat({resize(x, t.h, t.w, true), t});
        x = AuxMapNet::run(up_[i].first, x);
        cnet_normal_k::leaky(q, x.p, x.h * x.w * x.c);
        x = AuxMapNet::run(up_[i].second, x);
        cnet_normal_k::leaky(q, x.p, x.h * x.w * x.c);
        dec[i] = x;
    }
    Map pred = AuxMapNet::run(initial_, dec[1]);
    cnet_normal_k::normalize(q, pred.p, pred.h * pred.w, pred.c);
    for (int i = 0; i < 3; ++i) {
        const Map& feat = dec[i + 1];
        const int64_t oh = pred.h * 2, ow = pred.w * 2;
        x = concat({resize(feat, oh, ow, true), resize(pred, oh, ow, true)});
        for (int j = 0; j < 4; ++j) x = AuxMapNet::run(refine_[i][j], x, j < 3 ? ImageAct::relu : ImageAct::none);
        pred = x;
        cnet_normal_k::normalize(q, pred.p, pred.h * pred.w, pred.c);
    }
    auto* result = buf<uint8_t>((size_t) (W * H * 3));
    cnet_normal_k::output(q, pred.p, W * H, pred.c, result);
    std::vector<uint8_t> out((size_t) (W * H * 3));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), result, out.size());
    return out;
}

}  // namespace strata::image
