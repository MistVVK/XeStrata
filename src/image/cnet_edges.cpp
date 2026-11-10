// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/cnet_edges.cpp - the line maps' networks (cnet_models.hpp): HED's soft edges (Xie and Tu 2015, as
// ControlNet 1.1's annotator weights hold it: the picture less a learnt colour, five blocks of 3 x 3 convolutions and
// ReLU between 2 x 2 max pools, a 1 x 1 side output each, their mean through a sigmoid), and line art (Chan, Durand and
// Isola 2022: a ResNet generator, reflection padding and instance norms, its output a sigmoid), and M-LSD's straight
// lines (Gu et al. 2022: a MobileNetV2 trunk, a decoder of skip merges and residual 3 x 3 blocks, its tri-points map of a
// line's centre and its two ends' displacements, read as segments on the host as its authors' navervision/mlsd
// (Apache-2.0) reads them: the centres' 3 x 3 peaks, the 200 strongest, a score and a length to keep a segment).
// The input's fourth channel of ones is the weights' (their first convolution takes four).  The convolutions go
// through AuxMapNet; the side outputs' fusion, the strokes and the paddings are the kernels here and in aux_kernels.

#include "strata/image/cnet_models.hpp"

#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::image {

namespace cnet_edges_k {   // the kernels' names (an anonymous namespace mangles the same in every file)

using kernels::f16_from_f32;
using kernels::f32_from_f16;

// the network's input: rgb [n][3] (8 bits) times `scale`, less sub [3] where given, as a map of 8 channels
void input(sycl::queue& q, const uint8_t* rgb, int64_t n, float scale, const float* sub, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i % 8;
        out[i] = c < 3 ? f16_from_f32((float) rgb[p * 3 + c] * scale - (sub != nullptr ? sub[c] : 0.f)) : (uint16_t) 0;
    });
}

// channel 0 of a side output [h][w][C] at pixel (y, x) of an H x W picture, bilinearly (pixel centres, the border
// clamped)
inline float side_at(const uint16_t* m, int64_t h, int64_t w, int64_t C, int64_t H, int64_t W, int64_t y, int64_t x) {
    const float fy = sycl::clamp(((float) y + 0.5f) * (float) h / (float) H - 0.5f, 0.f, (float) (h - 1));
    const float fx = sycl::clamp(((float) x + 0.5f) * (float) w / (float) W - 0.5f, 0.f, (float) (w - 1));
    const int64_t y0 = (int64_t) fy, x0 = (int64_t) fx, y1 = sycl::min(y0 + 1, h - 1), x1 = sycl::min(x0 + 1, w - 1);
    const float ty = fy - (float) y0, tx = fx - (float) x0;
    const auto at = [&](int64_t yy, int64_t xx) { return f32_from_f16(m[(yy * w + xx) * C]); };
    return (at(y0, x0) * (1.f - tx) + at(y0, x1) * tx) * (1.f - ty) + (at(y1, x0) * (1.f - tx) + at(y1, x1) * tx) * ty;
}

struct Side {
    const uint16_t* p;
    int64_t h, w, c;
};

// the edges [H][W] (8 bits): the sigmoid of the side outputs' mean at the picture's size (the annotator's weights
// carry no fusion layer); `safe` rounds them to the four levels 0, 1/3, 2/3 and 1, the faint responses dropped
void fuse(sycl::queue& q, const Side* sides, int64_t H, int64_t W, bool safe, uint8_t* out) {
    const Side s0 = sides[0], s1 = sides[1], s2 = sides[2], s3 = sides[3], s4 = sides[4];
    q.parallel_for(sycl::range<1>((size_t) (H * W)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t y = i / W, x = i % W;
        float m = 0.f;
        for (const Side& s : {s0, s1, s2, s3, s4}) m += side_at(s.p, s.h, s.w, s.c, H, W, y, x);
        float e = 1.f / (1.f + sycl::exp(-m / 5.f));
        if (safe) e = sycl::rint(e * 3.f) / 3.f;
        out[i] = (uint8_t) sycl::clamp(e * 255.f, 0.f, 255.f);
    });
}

// the strokes of a scribble: 255 within `r` pixels (a disc) of an edge whose strength passes a half, else 0
void strokes(sycl::queue& q, const uint8_t* e, int64_t W, int64_t H, int r, uint8_t* out) {
    q.parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        bool on = false;
        for (int dy = -r; dy <= r && !on; ++dy)
            for (int dx = -r; dx <= r && !on; ++dx) {
                const int64_t yy = y + dy, xx = x + dx;
                on = dx * dx + dy * dy <= r * r && yy >= 0 && yy < H && xx >= 0 && xx < W && e[yy * W + xx] >= 128;
            }
        out[y * W + x] = on ? 255 : 0;
    });
}

// line art's map: the generator's output (channel 0, 1 white paper) turned into white lines on black, 8 bits
void lines(sycl::queue& q, const uint16_t* m, int64_t n, int64_t C, uint8_t* out) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const auto v = (int) (uint8_t) sycl::clamp(f32_from_f16(m[i * C]) * 255.f, 0.f, 255.f);
        out[i] = (uint8_t) (255 - v);
    });
}

// M-LSD's input: rgb [n][3] and a fourth channel of ones, as 2 v / 255 - 1, a map of 8 channels
void mlsd_input(sycl::queue& q, const uint8_t* rgb, int64_t n, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i % 8;
        const float v = c < 3 ? (float) rgb[p * 3 + c] : 1.f;
        out[i] = c < 4 ? f16_from_f32(v / 127.5f - 1.f) : (uint16_t) 0;
    });
}

}  // namespace cnet_edges_k

namespace {

// a scribble's pen: the radius its strokes are drawn with, in the processed picture's pixels (512 on the short side)
constexpr int kStrokeRadius = 2;

// a segment drawn one pixel wide, its pixels along its longer axis (8-connected), clipped to the picture
void draw_segment(std::vector<uint8_t>& m, int64_t W, int64_t H, int64_t x0, int64_t y0, int64_t x1, int64_t y1) {
    const int64_t dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int64_t err = dx + dy;
    for (int64_t n = 0; n <= 4 * (W + H); ++n) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) m[(size_t) (y0 * W + x0)] = 255;
        if (x0 == x1 && y0 == y1) break;
        const int64_t e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

}  // namespace

void HedNet::load_weights() {
    if (st_->find("norm") == nullptr || st_->find("block1.convs.0.weight") == nullptr)
        throw std::runtime_error(path_ + ": not ControlNet 1.1's HED (ControlNetHED)");
    norm_ = f32("norm");
    for (int b = 0; b < 5; ++b) {
        const std::string p = "block" + std::to_string(b + 1) + ".";
        convs_[b].clear();
        for (int i = 0; st_->find(p + "convs." + std::to_string(i) + ".weight") != nullptr; ++i) {
            const std::string c = p + "convs." + std::to_string(i);
            convs_[b].push_back(mconv(c + ".weight", c + ".bias", 1, 1));
        }
        proj_[b] = mconv(p + "projection.weight", p + "projection.bias", 1, 0);
    }
}

uint64_t HedNet::need_bytes(int64_t W, int64_t H) const {
    // the first block's maps at full size (64 channels, two at once and the library's FP32 sums), the rest smaller
    return (uint64_t) (W * H) * (64 * (2 * 3 + 4) + 32) + (128ull << 20);
}

std::vector<uint8_t> HedNet::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H, bool safe, bool scribble) {
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(H, W, 8);
    cnet_edges_k::input(q, src, H * W, 1.f, norm_, x.p);
    cnet_edges_k::Side sides[5];
    for (int b = 0; b < 5; ++b) {
        if (b > 0) x = maxpool(x, 2, 2, 0);
        for (const MConv& c : convs_[b]) x = AuxMapNet::run(c, x, ImageAct::relu);
        const Map s = AuxMapNet::run(proj_[b], x);
        sides[b] = {s.p, s.h, s.w, s.c};
    }
    auto* edge = buf<uint8_t>((size_t) (H * W));
    cnet_edges_k::fuse(q, sides, H, W, safe, edge);
    if (scribble) {   // the strokes: the strong edges thickened to a pen's width
        auto* st = buf<uint8_t>((size_t) (H * W));
        cnet_edges_k::strokes(q, edge, W, H, kStrokeRadius, st);
        edge = st;
    }
    std::vector<uint8_t> out((size_t) (H * W));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), edge, out.size());
    return out;
}

void LineartNet::load_weights() {
    if (st_->find("model0.1.weight") == nullptr || st_->find("model4.1.weight") == nullptr)
        throw std::runtime_error(path_ + ": not the line art generator (sk_model)");
    in_ = mconv("model0.1.weight", "model0.1.bias", 1, 0);
    down_[0] = mconv("model1.0.weight", "model1.0.bias", 2, 1);
    down_[1] = mconv("model1.3.weight", "model1.3.bias", 2, 1);
    res_.clear();
    for (int b = 0; st_->find("model2." + std::to_string(b) + ".conv_block.1.weight") != nullptr; ++b) {
        const std::string p = "model2." + std::to_string(b) + ".conv_block.";
        res_.emplace_back(mconv(p + "1.weight", p + "1.bias", 1, 0), mconv(p + "5.weight", p + "5.bias", 1, 0));
    }
    up_[0] = mdeconv("model3.0.weight", "model3.0.bias", true);
    up_[1] = mdeconv("model3.3.weight", "model3.3.bias", true);
    out_ = mconv("model4.1.weight", "model4.1.bias", 1, 0);
}

uint64_t LineartNet::need_bytes(int64_t W, int64_t H) const {
    // the full-size maps of 64 channels (padded by 3, three at once, the unfolded 7 x 7 slice) and the FP32 sums
    return (uint64_t) ((W + 6) * (H + 6)) * (64 * (2 * 4 + 4) + 32) + (192ull << 20);
}

std::vector<uint8_t> LineartNet::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H) {
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(H, W, 8);
    cnet_edges_k::input(q, src, H * W, 1.f / 255.f, nullptr, x.p);
    x = AuxMapNet::run(in_, reflect(x, 3));
    instance_norm(x, 1e-5f, ImageAct::relu);
    for (const MConv& c : down_) {
        x = AuxMapNet::run(c, x);
        instance_norm(x, 1e-5f, ImageAct::relu);
    }
    for (const auto& [c1, c2] : res_) {
        Map h = AuxMapNet::run(c1, reflect(x, 1));
        instance_norm(h, 1e-5f, ImageAct::relu);
        h = AuxMapNet::run(c2, reflect(h, 1));
        instance_norm(h, 1e-5f, ImageAct::none);
        x = add(x, h);
    }
    for (const MConv& c : up_) {   // transposed 3 x 3 of stride 2, padding 1, output padding 1: twice the size
        x = run_deconv_general(c, x, 2, 1, 1);
        instance_norm(x, 1e-5f, ImageAct::relu);
    }
    x = AuxMapNet::run(out_, reflect(x, 3), ImageAct::sigmoid);
    auto* line = buf<uint8_t>((size_t) (H * W));
    cnet_edges_k::lines(q, x.p, H * W, x.c, line);
    std::vector<uint8_t> out((size_t) (H * W));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), line, out.size());
    return out;
}

void MlsdNet::load_weights() {
    const std::string F = "backbone.features.";
    if (st_->find(F + "0.0.weight") == nullptr || st_->find("block23.conv3.weight") == nullptr)
        throw std::runtime_error(path_ + ": not M-LSD's large model (mlsd_large_512_fp32)");
    first_ = mconv(F + "0.0.weight", "", 2, 0, F + "0.1");
    blocks_.clear();
    int64_t in = first_.out;
    for (int i = 1; st_->find(F + std::to_string(i) + ".conv.0.0.weight") != nullptr; ++i) {
        const std::string p = F + std::to_string(i) + ".conv.";
        Inverted b;
        // with an expansion: conv.0 1 x 1, conv.1 the depthwise 3 x 3, conv.2 1 x 1 and conv.3 its norm; without
        // (the first block): conv.0 the depthwise one, conv.1 1 x 1 and conv.2 its norm
        b.has_expand = st_->find(p + "3.weight") != nullptr;
        const std::string dw = b.has_expand ? p + "1." : p + "0.";
        const SafeTensor* t = st_->find(dw + "0.weight");
        const int64_t hidden = t->shape[0];
        if (b.has_expand) b.expand = mconv(p + "0.0.weight", "", 1, 0, p + "0.1");
        const std::string pw = b.has_expand ? p + "2" : p + "1", pn = b.has_expand ? p + "3" : p + "2";
        const int64_t out = st_->find(pw + ".weight")->shape[0];
        // the stride by the MobileNetV2 schedule: the first block of each width 24, 32 and 64 halves the size
        const int stride = (i == 2 || i == 4 || i == 7) ? 2 : 1;
        b.tf_pad = stride == 2;
        b.dw = mconv(dw + "0.weight", "", stride, stride == 2 ? 0 : 1, dw + "1", 1e-5f, 0, true);
        b.project = mconv(pw + ".weight", "", 1, 0, pn);
        b.residual = stride == 1 && in == out;
        (void) hidden;
        blocks_.push_back(b);
        in = out;
    }
    taps_ = {1, 3, 6, 10, 13};
    for (int j = 0; j < 4; ++j) {   // blocks 15, 17, 19, 21: the merges; 16, 18, 20, 22: the refinements
        const std::string a = "block" + std::to_string(15 + 2 * j) + ".", b = "block" + std::to_string(16 + 2 * j) + ".";
        merge_[j].deep = mconv(a + "conv1.0.weight", a + "conv1.0.bias", 1, 0, a + "conv1.1");
        merge_[j].skip = mconv(a + "conv2.0.weight", a + "conv2.0.bias", 1, 0, a + "conv2.1");
        merge_[j].up = j > 0;
        refine_[j].c1 = mconv(b + "conv1.0.weight", b + "conv1.0.bias", 1, 1, b + "conv1.1");
        refine_[j].c2 = mconv(b + "conv2.0.weight", b + "conv2.0.bias", 1, 1, b + "conv2.1");
    }
    dil_ = mconv("block23.conv1.0.weight", "block23.conv1.0.bias", 1, 5, "block23.conv1.1");
    dil_.dil = 5;
    c3_ = mconv("block23.conv2.0.weight", "block23.conv2.0.bias", 1, 1, "block23.conv2.1");
    out_ = mconv("block23.conv3.weight", "block23.conv3.bias", 1, 0);
}

uint64_t MlsdNet::need_bytes(int64_t W, int64_t H) const {
    // the half-size maps: the expanded 96 channels at most there, the decoder's 128, the dilated slice
    return (uint64_t) (W * H / 4) * (128 * 2 * 4 + 64 * 4) + (192ull << 20);
}

std::vector<uint8_t> MlsdNet::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H, float score, float dist) {
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    if (W % 32 != 0 || H % 32 != 0) throw std::invalid_argument("M-LSD: the picture's sides are multiples of 32");
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(H, W, 8);
    cnet_edges_k::mlsd_input(q, src, H * W, x.p);
    // the stride-2 layers pad their right and bottom by one, none on the left and top (as the model was trained)
    x = AuxMapNet::run(first_, pad(x, 0, 0, 1, 1), ImageAct::relu6);
    std::vector<Map> taps;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const Inverted& b = blocks_[i];
        Map h = b.has_expand ? AuxMapNet::run(b.expand, x, ImageAct::relu6) : x;
        h = AuxMapNet::run(b.dw, b.tf_pad ? pad(h, 0, 0, 1, 1) : h, ImageAct::relu6);
        x = AuxMapNet::run(b.project, h, ImageAct::none, b.residual ? &x : nullptr);
        if (std::find(taps_.begin(), taps_.end(), (int) i + 1) != taps_.end()) taps.push_back(x);
        if ((int) i + 1 == taps_.back()) break;
    }
    // the decoder from the deepest tap: merge (the deeper map's 1 x 1, upsampled 2x with the corners aligned, beside
    // the skip's 1 x 1), then refine (a residual 3 x 3, a 3 x 3 to 64)
    Map d = taps[4];
    for (int j = 0; j < 4; ++j) {
        const Map& skip = taps[(size_t) (3 - j)];
        Map deep = AuxMapNet::run(merge_[j].deep, d, ImageAct::relu);
        if (merge_[j].up) deep = resize(deep, deep.h * 2, deep.w * 2, true);
        const Map s2 = AuxMapNet::run(merge_[j].skip, skip, ImageAct::relu);
        d = concat({s2, deep});
        d = AuxMapNet::run(refine_[j].c1, d, ImageAct::relu, &d);
        d = AuxMapNet::run(refine_[j].c2, d, ImageAct::relu);
    }
    d = AuxMapNet::run(dil_, d, ImageAct::relu);
    d = AuxMapNet::run(c3_, d, ImageAct::relu);
    d = AuxMapNet::run(out_, d);
    // the tri-points map: channel 7 the centres' logits, 8 .. 11 the displacements of the two ends (x, y, x, y)
    const int64_t h = d.h, w = d.w, C = d.c, n = h * w;
    std::vector<uint16_t> m16((size_t) (n * C));
    gpu::stream_sync(s_);
    gpu::copy(m16.data(), d.p, m16.size() * 2);
    const auto at = [&](int64_t p, int c) { return kernels::f32_from_f16(m16[(size_t) (p * C + c)]); };
    std::vector<float> heat((size_t) n);
    for (int64_t p = 0; p < n; ++p) heat[(size_t) p] = 1.f / (1.f + std::exp(-at(p, 7)));
    std::vector<float> peak((size_t) n, 0.f);   // a 3 x 3 local maximum keeps its score, the rest 0
    for (int64_t y = 0; y < h; ++y)
        for (int64_t xx = 0; xx < w; ++xx) {
            const float v = heat[(size_t) (y * w + xx)];
            bool top = true;
            for (int64_t dy = -1; dy <= 1 && top; ++dy)
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    const int64_t yy = y + dy, x2 = xx + dx;
                    if (yy >= 0 && yy < h && x2 >= 0 && x2 < w && heat[(size_t) (yy * w + x2)] > v) {
                        top = false;
                        break;
                    }
                }
            if (top) peak[(size_t) (y * w + xx)] = v;
        }
    std::vector<int64_t> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    const size_t k = std::min<size_t>(200, order.size());   // the 200 best centres
    std::partial_sort(order.begin(), order.begin() + (std::ptrdiff_t) k, order.end(),
                      [&](int64_t a, int64_t b) { return peak[(size_t) a] > peak[(size_t) b]; });
    std::vector<uint8_t> out((size_t) (W * H), 0);
    const double sx = (double) W / (double) w, sy = (double) H / (double) h;
    for (size_t i = 0; i < k; ++i) {
        const int64_t p = order[i], cy = p / w, cx = p % w;
        const float x0 = at(p, 8), y0 = at(p, 9), x1 = at(p, 10), y1 = at(p, 11);
        if (peak[(size_t) p] <= score || std::hypot(x0 - x1, y0 - y1) <= dist) continue;
        const auto fx = (double) cx, fy = (double) cy;
        draw_segment(out, W, H, (int64_t) ((fx + x0) * sx), (int64_t) ((fy + y0) * sy), (int64_t) ((fx + x1) * sx),
                     (int64_t) ((fy + y1) * sy));
    }
    return out;
}

}  // namespace strata::image
