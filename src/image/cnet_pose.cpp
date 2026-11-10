// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/cnet_pose.cpp - the pose map (cnet_models.hpp): YOLOX-L finds the people (Ge et al. 2021, "YOLOX:
// Exceeding YOLO Series in 2021"), DWPose finds each one's 133 whole-body keypoints (Yang et al. 2023, "Effective
// Whole-body Pose Estimation with Two-stages Distillation", on RTMPose's CSPNeXt and RTMCC head: Jiang et al. 2023),
// and the skeletons are drawn with the limbs, colours, widths and the hands' and face's points of the pictures
// ControlNet's pose models were trained on (IDEA-Research/DWPose's ControlNet-v1-1-nightly/annotator/dwpose/util.py,
// Apache-2.0, a convention those models expect).  Written from the papers and the weights' names and shapes; the
// convolutions and the heads go through AuxMapNet and kernels here, the drawing runs on the host.

#include "strata/image/cnet_models.hpp"

#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::image {

namespace cnet_pose_k {   // the kernels' names (an anonymous namespace mangles the same in every file)

using kernels::f16_from_f32;

// DWPose's crop: out [384][288][8] FP16 of the picture under the affine map a (the crop's pixel (x, y) at the
// picture's (a0 x + a1 y + a2, a3 x + a4 y + a5)), bilinear, 0 outside, its colours BGR less mean over std; the
// channels past 3 zero
void crop(sycl::queue& q, const uint8_t* rgb, int64_t W, int64_t H, sycl::float8 a, int64_t ow, int64_t oh,
          uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (ow * oh * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t ch = i % 8, p = i / 8, y = p / ow, x = p % ow;
        if (ch >= 3) {
            out[i] = 0;
            return;
        }
        const int64_t c = 2 - ch;
        const float fx = a[0] * (float) x + a[1] * (float) y + a[2], fy = a[3] * (float) x + a[4] * (float) y + a[5];
        const float x0f = sycl::floor(fx), y0f = sycl::floor(fy), tx = fx - x0f, ty = fy - y0f;
        const auto x0 = (int64_t) x0f, y0 = (int64_t) y0f;
        const auto at = [&](int64_t yy, int64_t xx) {
            return yy >= 0 && yy < H && xx >= 0 && xx < W ? (float) rgb[(yy * W + xx) * 3 + c] : 0.f;
        };
        const float v = (at(y0, x0) * (1.f - tx) + at(y0, x0 + 1) * tx) * (1.f - ty) +
                        (at(y0 + 1, x0) * (1.f - tx) + at(y0 + 1, x0 + 1) * tx) * ty;
        const float mean = ch == 0 ? 123.675f : ch == 1 ? 116.28f : 103.53f;
        const float sd = ch == 0 ? 58.395f : ch == 1 ? 57.12f : 57.375f;
        out[i] = f16_from_f32((v - mean) / sd);
    });
}

// The channel attention's gate: g[o] = the hard sigmoid of b[o] + w[o] . m (w [C][C], m the channels' means)
void channel_gate(sycl::queue& q, const float* m, const float* w, const float* b, int64_t C, float* g) {
    q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const auto o = static_cast<int64_t>(id[0]);
        float v = b[o];
        for (int64_t i = 0; i < C; ++i) v += w[o * C + i] * m[i];
        g[o] = sycl::clamp(v / 6.f + 0.5f, 0.f, 1.f);
    });
}

// The head's input: row k of out [K][HW] (FP16) channel k of f [HW][C] over its positions, scale-normed (each row over
// its root mean square, times g)
void head_rows(sycl::queue& q, const uint16_t* f, int64_t HW, int64_t C, int64_t K, float g, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) K), [=](sycl::id<1> id) {
        const auto k = static_cast<int64_t>(id[0]);
        float ss = 0.f;
        for (int64_t p = 0; p < HW; ++p) {
            const float v = kernels::f32_from_f16(f[p * C + k]);
            ss += v * v;
        }
        const float s = g / sycl::fmax(sycl::sqrt(ss / (float) HW), 1e-5f);
        for (int64_t p = 0; p < HW; ++p) out[k * HW + p] = f16_from_f32(kernels::f32_from_f16(f[p * C + k]) * s);
    });
}

// out [n][k] (FP16) = x [n][k] scale-normed, times g
void norm_rows(sycl::queue& q, const float* x, int64_t n, int64_t k, float g, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto r = static_cast<int64_t>(id[0]);
        const float* v = x + r * k;
        float ss = 0.f;
        for (int64_t j = 0; j < k; ++j) ss += v[j] * v[j];
        const float s = g / sycl::fmax(sycl::sqrt(ss / (float) k), 1e-5f);
        for (int64_t j = 0; j < k; ++j) out[r * k + j] = f16_from_f32(v[j] * s);
    });
}

// The gated attention unit's weights: att [K][K] = relu(q_r . k_c / sqrt(S))^2, q and k the rows' last S values of uv
// [K][2E + S] each scaled and shifted by its own gamma and beta
void gau_scores(sycl::queue& q, const float* uv, const float* gamma, const float* beta, int64_t K, int64_t E,
                int64_t S, float* att) {
    q.parallel_for(sycl::range<2>((size_t) K, (size_t) K), [=](sycl::id<2> id) {
        const auto r = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        const float* br = uv + r * (2 * E + S) + 2 * E;
        const float* bc = uv + c * (2 * E + S) + 2 * E;
        float v = 0.f;
        for (int64_t j = 0; j < S; ++j) v += (br[j] * gamma[j] + beta[j]) * (bc[j] * gamma[S + j] + beta[S + j]);
        const float z = sycl::fmax(v / sycl::sqrt((float) S), 0.f);
        att[r * K + c] = z * z;
    });
}

// out [K][E] (FP16) = u * (att . v), u and v the rows' first and second E values of uv
void gau_mix(sycl::queue& q, const float* uv, const float* att, int64_t K, int64_t E, int64_t S, uint16_t* out) {
    q.parallel_for(sycl::range<2>((size_t) K, (size_t) E), [=](sycl::id<2> id) {
        const auto r = static_cast<int64_t>(id[0]), j = static_cast<int64_t>(id[1]);
        float v = 0.f;
        for (int64_t c = 0; c < K; ++c) v += att[r * K + c] * uv[c * (2 * E + S) + E + j];
        out[r * E + j] = f16_from_f32(uv[r * (2 * E + S) + j] * v);
    });
}

// out [n][d] (FP16) = o + x * scale (the unit's scaled shortcut)
void shortcut(sycl::queue& q, const float* o, const float* x, const float* scale, int64_t n, int64_t d, uint16_t* out) {
    q.parallel_for(sycl::range<1>((size_t) (n * d)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        out[i] = f16_from_f32(o[i] + x[i] * scale[i % d]);
    });
}

// SimCC's reading: out [K][3] the place of each row's largest x and y logits (in the doubled crop) and the smaller
// of the two maxima
void simcc_argmax(sycl::queue& q, const float* sx, const float* sy, int64_t K, int64_t nx, int64_t ny, float* out) {
    q.parallel_for(sycl::range<1>((size_t) K), [=](sycl::id<1> id) {
        const auto k = static_cast<int64_t>(id[0]);
        int64_t ix = 0, iy = 0;
        for (int64_t i = 1; i < nx; ++i) if (sx[k * nx + i] > sx[k * nx + ix]) ix = i;
        for (int64_t i = 1; i < ny; ++i) if (sy[k * ny + i] > sy[k * ny + iy]) iy = i;
        out[k * 3] = (float) ix;
        out[k * 3 + 1] = (float) iy;
        out[k * 3 + 2] = sycl::fmin(sx[k * nx + ix], sy[k * ny + iy]);
    });
}

}  // namespace cnet_pose_k

namespace {

constexpr float kYoloxEps = 1e-3f, kDwEps = 1e-5f;
// a person found where YOLOX's objectness times its class score passes kPerson; boxes overlapping past kOverlap are
// one person (the IoU YOLOX's evaluation suppresses at)
constexpr float kPerson = 0.3f, kOverlap = 0.45f;

float iou(const ControlBox& a, const ControlBox& b) {   // the boxes' intersection over their union
    const float w = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float h = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float ia = w * h, aa = (a.x1 - a.x0) * (a.y1 - a.y0), ab = (b.x1 - b.x0) * (b.y1 - b.y0);
    return ia > 0.f ? ia / (aa + ab - ia) : 0.f;
}

std::vector<float> f32_of(const std::vector<uint16_t>& h) {
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = kernels::f32_from_f16(h[i]);
    return v;
}

}  // namespace

// ---- YOLOX-L

AuxMapNet::MConv YoloxNet::conv(const std::string& p, int stride) {
    const SafeTensor* t = st_->find(p + ".conv.weight");
    if (t == nullptr) throw std::runtime_error(path_ + ": no " + p);
    const int k = (int) t->shape[2];
    return mconv(p + ".conv.weight", "", stride, k / 2, p + ".bn", kYoloxEps);
}

YoloxNet::Csp YoloxNet::load_csp(const std::string& p, bool shortcut) {
    Csp c;
    c.c1 = conv(p + ".conv1", 1);
    c.c2 = conv(p + ".conv2", 1);
    c.c3 = conv(p + ".conv3", 1);
    for (int i = 0; st_->find(p + ".m." + std::to_string(i) + ".conv1.conv.weight") != nullptr; ++i) {
        const std::string m = p + ".m." + std::to_string(i);
        c.m.emplace_back(conv(m + ".conv1", 1), conv(m + ".conv2", 1));
    }
    c.shortcut = shortcut;
    return c;
}

void YoloxNet::load_weights() {
    const std::string B = "backbone.backbone.";
    if (st_->find(B + "stem.conv.conv.weight") == nullptr || st_->find("head.obj_preds.0.weight") == nullptr)
        throw std::runtime_error(path_ + ": not YOLOX (Megvii's names)");
    // the Focus stem's 12 channels: tl, bl, tr, br of each colour, padded to 16
    stem_ = mconv(B + "stem.conv.conv.weight", "", 1, 1, B + "stem.conv.bn", kYoloxEps, 16);
    for (int i = 0; i < 4; ++i) {
        const std::string d = B + "dark" + std::to_string(i + 2);
        down_[i] = conv(d + ".0", 2);
        if (i < 3) dark_[i] = load_csp(d + ".1", true);
    }
    spp1_ = conv(B + "dark5.1.conv1", 1);
    spp2_ = conv(B + "dark5.1.conv2", 1);
    dark_[3] = load_csp(B + "dark5.2", false);
    lateral0_ = conv("backbone.lateral_conv0", 1);
    reduce1_ = conv("backbone.reduce_conv1", 1);
    bu2_ = conv("backbone.bu_conv2", 2);
    bu1_ = conv("backbone.bu_conv1", 2);
    p4_ = load_csp("backbone.C3_p4", false);
    p3_ = load_csp("backbone.C3_p3", false);
    n3_ = load_csp("backbone.C3_n3", false);
    n4_ = load_csp("backbone.C3_n4", false);
    for (int k = 0; k < 3; ++k) {
        const std::string s = std::to_string(k);
        stems_[k] = conv("head.stems." + s, 1);
        for (int j = 0; j < 2; ++j) {
            cls_[k][j] = conv("head.cls_convs." + s + "." + std::to_string(j), 1);
            reg_[k][j] = conv("head.reg_convs." + s + "." + std::to_string(j), 1);
        }
        cls_pred_[k] = mconv("head.cls_preds." + s + ".weight", "head.cls_preds." + s + ".bias", 1, 0);
        reg_pred_[k] = mconv("head.reg_preds." + s + ".weight", "head.reg_preds." + s + ".bias", 1, 0);
        obj_pred_[k] = mconv("head.obj_preds." + s + ".weight", "head.obj_preds." + s + ".bias", 1, 0);
    }
}

uint64_t YoloxNet::need_bytes() const {
    // the 320 x 320 maps of 64 channels and the 160 x 160 of 128 (several at once), the library's FP32 sums
    return (uint64_t) (320 * 320) * 64 * (2 * 4 + 4) + (uint64_t) (160 * 160) * 128 * 2 * 8 + (256ull << 20);
}

AuxMapNet::Map YoloxNet::up2(const Map& x) {
    Map out = map(2 * x.h, 2 * x.w, x.c);
    qi_up2x16(x.p, x.h, x.w, x.c, out.p, s_);
    return out;
}

AuxMapNet::Map YoloxNet::csp(const Csp& c, const Map& x) {
    Map a = AuxMapNet::run(c.c1, x, ImageAct::silu);
    const Map b = AuxMapNet::run(c.c2, x, ImageAct::silu);
    for (const auto& [m1, m2] : c.m) {
        const Map h = AuxMapNet::run(m1, a, ImageAct::silu);
        a = AuxMapNet::run(m2, h, ImageAct::silu, c.shortcut ? &a : nullptr);
    }
    return AuxMapNet::run(c.c3, concat({a, b}), ImageAct::silu);
}

std::vector<ControlBox> YoloxNet::people(const std::vector<uint8_t>& rgb, int64_t W, int64_t H) {
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    const float r = std::min(640.f / (float) H, 640.f / (float) W);
    const auto nw = (int64_t) ((float) W * r), nh = (int64_t) ((float) H * r);
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    Map x = map(320, 320, 16);
    aux_letterbox16(src, W, H, nw, nh, 0, 0, 640, 640, r, true, x.p, &q);
    x = AuxMapNet::run(stem_, x, ImageAct::silu);
    Map feat[3];
    for (int i = 0; i < 4; ++i) {
        x = AuxMapNet::run(down_[i], x, ImageAct::silu);
        if (i == 3) {   // the spatial pyramid pooling: the map and its 5, 9 and 13 max pools side by side
            const Map s = AuxMapNet::run(spp1_, x, ImageAct::silu);
            x = AuxMapNet::run(spp2_, concat({s, maxpool(s, 5, 1, 2), maxpool(s, 9, 1, 4), maxpool(s, 13, 1, 6)}),
                               ImageAct::silu);
        }
        x = csp(dark_[i], x);
        if (i >= 1) feat[i - 1] = x;   // dark3, dark4, dark5
    }
    // PAFPN: top-down, then bottom-up
    const Map f0 = AuxMapNet::run(lateral0_, feat[2], ImageAct::silu);
    Map t = csp(p4_, concat({up2(f0), feat[1]}));
    const Map f1 = AuxMapNet::run(reduce1_, t, ImageAct::silu);
    const Map pan2 = csp(p3_, concat({up2(f1), feat[0]}));
    const Map pan1 = csp(n3_, concat({AuxMapNet::run(bu2_, pan2, ImageAct::silu), f1}));
    const Map pan0 = csp(n4_, concat({AuxMapNet::run(bu1_, pan1, ImageAct::silu), f0}));
    // the decoupled head of each level: the boxes (centre's offset in cells and log sizes), objectness, class 0
    std::vector<ControlBox> cand;
    const Map levels[3] = {pan2, pan1, pan0};
    const int strides[3] = {8, 16, 32};
    std::vector<std::vector<uint16_t>> outs;
    for (int k = 0; k < 3; ++k) {
        const Map s = AuxMapNet::run(stems_[k], levels[k], ImageAct::silu);
        const Map c = AuxMapNet::run(cls_pred_[k], AuxMapNet::run(cls_[k][1], AuxMapNet::run(cls_[k][0], s, ImageAct::silu),
                                                                  ImageAct::silu));
        const Map rf = AuxMapNet::run(reg_[k][1], AuxMapNet::run(reg_[k][0], s, ImageAct::silu), ImageAct::silu);
        const Map rg = AuxMapNet::run(reg_pred_[k], rf), ob = AuxMapNet::run(obj_pred_[k], rf);
        for (const Map* m : {&c, &rg, &ob}) {
            std::vector<uint16_t> h((size_t) (m->h * m->w * m->c));
            gpu::stream_sync(s_);
            gpu::copy(h.data(), m->p, h.size() * 2);
            outs.push_back(std::move(h));
        }
        const auto cv = f32_of(outs[outs.size() - 3]), rv = f32_of(outs[outs.size() - 2]), ov = f32_of(outs.back());
        const int64_t gw = levels[k].w, gh = levels[k].h;
        for (int64_t p = 0; p < gw * gh; ++p) {
            const float obj = 1.f / (1.f + std::exp(-ov[(size_t) (p * ob.c)]));
            const float cl = 1.f / (1.f + std::exp(-cv[(size_t) (p * c.c)]));
            const float sc = obj * cl;
            if (sc <= kPerson) continue;
            const float* b = &rv[(size_t) (p * rg.c)];
            const int64_t gy = p / gw, gx = p - gy * gw;
            const float cx = ((float) gx + b[0]) * (float) strides[k], cy = ((float) gy + b[1]) * (float) strides[k];
            const float bw = std::exp(b[2]) * (float) strides[k], bh = std::exp(b[3]) * (float) strides[k];
            cand.push_back({(cx - bw / 2) / r, (cy - bh / 2) / r, (cx + bw / 2) / r, (cy + bh / 2) / r, sc});
        }
        outs.clear();
    }
    // the greedy non-maximum suppression: the strongest box of each overlapping group (IoU past kOverlap) stays
    std::sort(cand.begin(), cand.end(), [](const ControlBox& a, const ControlBox& b) { return a.score > b.score; });
    std::vector<ControlBox> keep;
    for (const ControlBox& b : cand) {
        bool ok = true;
        for (const ControlBox& k : keep)
            if (iou(b, k) > kOverlap) {
                ok = false;
                break;
            }
        if (ok) keep.push_back(b);
    }
    return keep;
}

// ---- DWPose

AuxMapNet::MConv DwPoseNet::conv(const std::string& p, int stride, bool depthwise) {
    const SafeTensor* t = st_->find(p + ".conv.weight");
    if (t == nullptr) throw std::runtime_error(path_ + ": no " + p);
    const int k = (int) t->shape[2];
    return mconv(p + ".conv.weight", "", stride, k / 2, p + ".bn", kDwEps, 0, depthwise);
}

void DwPoseNet::load_weights() {
    const std::string B = "backbone.";
    if (st_->find(B + "stem.0.conv.weight") == nullptr || st_->find("head.gau.uv.weight") == nullptr)
        throw std::runtime_error(path_ + ": not DWPose (RTMPose's names)");
    for (int i = 0; i < 3; ++i) stem_[i] = conv(B + "stem." + std::to_string(i), i == 0 ? 2 : 1);
    for (int s = 0; s < 4; ++s) {
        const std::string p = B + "stage" + std::to_string(s + 1) + ".";
        Stage& st = stages_[s];
        st.down = conv(p + "0", 2);
        st.spp = st_->find(p + "1.conv1.conv.weight") != nullptr && st_->find(p + "2.main_conv.conv.weight") != nullptr;
        const std::string c = p + (st.spp ? "2." : "1.");
        if (st.spp) {
            st.spp1 = conv(p + "1.conv1", 1);
            st.spp2 = conv(p + "1.conv2", 1);
        }
        st.identity = !st.spp;   // the last stage's blocks add no shortcut
        st.main = conv(c + "main_conv", 1);
        st.shortc = conv(c + "short_conv", 1);
        st.fin = conv(c + "final_conv", 1);
        st.blocks.clear();
        for (int b = 0; st_->find(c + "blocks." + std::to_string(b) + ".conv1.conv.weight") != nullptr; ++b) {
            const std::string k = c + "blocks." + std::to_string(b) + ".";
            st.blocks.push_back({conv(k + "conv1", 1), conv(k + "conv2.depthwise_conv", 1, true),
                                 conv(k + "conv2.pointwise_conv", 1)});
        }
        st.att_w = upload(host_f32(c + "attention.fc.weight"));
        st.att_b = upload(host_f32(c + "attention.fc.bias"));
    }
    final_ = mconv("head.final_layer.weight", "head.final_layer.bias", 1, 3);
    K_ = st_->find("head.final_layer.weight")->shape[0];
    mlp_g_ = host_f32("head.mlp.0.g")[0];
    gau_g_ = host_f32("head.gau.ln.g")[0];
    const std::vector<float> gamma = host_f32("head.gau.gamma");
    S_ = (int64_t) gamma.size() / 2;
    gamma_ = upload(gamma);
    beta_ = upload(host_f32("head.gau.beta"));
    res_scale_ = upload(host_f32("head.gau.res_scale.scale"));
    D_ = st_->find("head.gau.o.weight")->shape[0];
    E_ = st_->find("head.gau.o.weight")->shape[1];
    in_w_ = st_->find("head.cls_x.weight")->shape[0] / 2;
    in_h_ = st_->find("head.cls_y.weight")->shape[0] / 2;
    wm_ = lin("head.mlp.1");
    wuv_ = lin("head.gau.uv");
    wo_ = lin("head.gau.o");
    wx_ = lin("head.cls_x");
    wy_ = lin("head.cls_y");
}

uint64_t DwPoseNet::need_bytes() const {
    // the crop's 192 x 144 maps of 64 channels (several), the library's sums
    return (uint64_t) (192 * 144) * 64 * 2 * 10 + (128ull << 20);
}

std::vector<WholeBody> DwPoseNet::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H,
                                      const std::vector<ControlBox>& boxes) {
    load();
    sycl::queue& q = core::Runtime::get().stream(s_);
    std::vector<ControlBox> bx = boxes;
    if (bx.empty()) bx.push_back({0.f, 0.f, (float) W, (float) H, 1.f});
    const int64_t K = K_, P = (int64_t) bx.size();
    // each person's keypoints, read once after the last
    auto* res = scratch_as<float>(kResultSlot, (size_t) (P * K * 3));
    auto* src = scratch_as<uint8_t>(kPictureSlot, rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    std::vector<sycl::float4> crops;   // each crop's centre and size in the picture
    for (int64_t n = 0; n < P; ++n) {
        const ControlBox& b = bx[(size_t) n];
        begin_run();
        // the box's centre and size (1.25 times, the margin RTMPose's top-down configurations give a person's box; to
        // the crop's aspect), the affine map from the crop to the picture
        const float cx = (b.x0 + b.x1) / 2.f, cy = (b.y0 + b.y1) / 2.f, aspect = (float) in_w_ / (float) in_h_;
        float sw = (b.x1 - b.x0) * 1.25f, sh = (b.y1 - b.y0) * 1.25f;
        if (sw > sh * aspect) sh = sw / aspect;
        else sw = sh * aspect;
        crops.emplace_back(cx, cy, sw, sh);
        const float k = sw / (float) in_w_;   // picture pixels per crop pixel (the same both ways)
        // the crop's pixel (x, y) at the picture's (left + x k, top + y k), the box's centre at the crop's
        const sycl::float8 a(k, 0.f, cx - sw / 2.f, 0.f, k, cy - sh / 2.f, 0.f, 0.f);
        Map x = map(in_h_, in_w_, 8);
        cnet_pose_k::crop(q, src, W, H, a, in_w_, in_h_, x.p);
        for (const MConv& c : stem_) x = AuxMapNet::run(c, x, ImageAct::silu);
        for (const Stage& st : stages_) {
            x = AuxMapNet::run(st.down, x, ImageAct::silu);
            if (st.spp) {
                const Map s = AuxMapNet::run(st.spp1, x, ImageAct::silu);
                x = AuxMapNet::run(st.spp2, concat({s, maxpool(s, 5, 1, 2), maxpool(s, 9, 1, 4), maxpool(s, 13, 1, 6)}),
                                   ImageAct::silu);
            }
            Map m = AuxMapNet::run(st.main, x, ImageAct::silu);
            const Map sc = AuxMapNet::run(st.shortc, x, ImageAct::silu);
            for (const Block& bl : st.blocks) {
                const Map h = AuxMapNet::run(bl.dw, AuxMapNet::run(bl.c1, m, ImageAct::silu), ImageAct::silu);
                m = AuxMapNet::run(bl.pw, h, ImageAct::silu, st.identity ? &m : nullptr);
            }
            Map cat = concat({m, sc});
            // the channel attention: the channels' means through a 1 x 1 convolution and a hard sigmoid
            const int64_t C = cat.c;
            auto* mean = buf<float>((size_t) C);
            auto* gate = buf<float>((size_t) C);
            aux_col_mean16(cat.p, cat.h * cat.w, C, mean, s_);
            cnet_pose_k::channel_gate(q, mean, st.att_w, st.att_b, C, gate);
            aux_scale_channels16(cat.p, cat.h * cat.w, C, gate, s_);
            x = AuxMapNet::run(st.fin, cat, ImageAct::silu);
        }
        const Map f = AuxMapNet::run(final_, x);   // [h][w][K] (K padded)
        // the head: each keypoint's map flattened and scale-normed, to D, a gated attention unit, SimCC's two
        // classifiers
        const int64_t HW = f.h * f.w, U = 2 * E_ + S_;
        auto* t = buf<uint16_t>((size_t) (K * HW));
        cnet_pose_k::head_rows(q, f.p, HW, f.c, K, mlp_g_, t);
        auto* xk = buf<float>((size_t) (K * D_));
        product(t, wm_, xk, K);
        auto* xn = buf<uint16_t>((size_t) (K * D_));
        cnet_pose_k::norm_rows(q, xk, K, D_, gau_g_, xn);
        auto* uv = buf<float>((size_t) (K * U));
        product(xn, wuv_, uv, K);
        ProductStore p;
        p.in = uv; p.out = uv; p.rows = K; p.cols = U; p.act = ImageAct::silu; p.add_zero = false;
        product_store(p, s_);
        auto* att = buf<float>((size_t) (K * K));
        cnet_pose_k::gau_scores(q, uv, gamma_, beta_, K, E_, S_, att);
        auto* g = buf<uint16_t>((size_t) (K * E_));
        cnet_pose_k::gau_mix(q, uv, att, K, E_, S_, g);
        auto* o = buf<float>((size_t) (K * D_));
        product(g, wo_, o, K);
        auto* o16 = buf<uint16_t>((size_t) (K * D_));
        cnet_pose_k::shortcut(q, o, xk, res_scale_, K, D_, o16);
        auto* sx = buf<float>((size_t) (K * 2 * in_w_));
        auto* sy = buf<float>((size_t) (K * 2 * in_h_));
        product(o16, wx_, sx, K);
        product(o16, wy_, sy, K);
        cnet_pose_k::simcc_argmax(q, sx, sy, K, 2 * in_w_, 2 * in_h_, res + n * K * 3);
    }
    std::vector<float> r((size_t) (P * K * 3));
    gpu::stream_sync(s_);
    gpu::copy(r.data(), res, r.size() * 4);
    std::vector<WholeBody> out;
    for (int64_t n = 0; n < P; ++n) {
        const sycl::float4 c = crops[(size_t) n];
        WholeBody body{};
        for (int64_t kk = 0; kk < K && kk < 133; ++kk) {
            const float* v = &r[(size_t) ((n * K + kk) * 3)];
            float kx = v[0] / 2.f, ky = v[1] / 2.f;
            if (v[2] <= 0.f) kx = ky = -1.f;
            body[(size_t) kk] = {kx / (float) in_w_ * c.z() + c.x() - c.z() / 2.f,
                                 ky / (float) in_h_ * c.w() + c.y() - c.w() / 2.f, v[2]};
        }
        out.push_back(body);
    }
    return out;
}

// ---- the drawing

namespace {

struct Canvas {
    std::vector<uint8_t>& px;
    int64_t W, H;
    void put(int64_t x, int64_t y, const uint8_t* c) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        uint8_t* p = &px[(size_t) ((y * W + x) * 3)];
        p[0] = c[0], p[1] = c[1], p[2] = c[2];
    }
    void disc(float cx, float cy, float r, const uint8_t* c) {
        for (auto y = (int64_t) std::floor(cy - r); y <= (int64_t) std::ceil(cy + r); ++y)
            for (auto x = (int64_t) std::floor(cx - r); x <= (int64_t) std::ceil(cx + r); ++x)
                if (((float) x - cx) * ((float) x - cx) + ((float) y - cy) * ((float) y - cy) <= r * r) put(x, y, c);
    }
    // a filled ellipse of half axes a (along the angle) and b about (cx, cy)
    void ellipse(float cx, float cy, float a, float b, float ang, const uint8_t* c) {
        const float cs = std::cos(ang), sn = std::sin(ang), r = std::max(a, b) + 1.f;
        for (auto y = (int64_t) std::floor(cy - r); y <= (int64_t) std::ceil(cy + r); ++y)
            for (auto x = (int64_t) std::floor(cx - r); x <= (int64_t) std::ceil(cx + r); ++x) {
                const float dx = (float) x - cx, dy = (float) y - cy, u = dx * cs + dy * sn, v = -dx * sn + dy * cs;
                if ((u * u) / std::max(a * a, 1e-6f) + (v * v) / std::max(b * b, 1e-6f) <= 1.f) put(x, y, c);
            }
    }
    // a segment of the given width
    void line(float x0, float y0, float x1, float y1, float w, const uint8_t* c) {
        const float len = std::hypot(x1 - x0, y1 - y0), r = w / 2.f;
        const auto lo_x = (int64_t) std::floor(std::min(x0, x1) - r), hi_x = (int64_t) std::ceil(std::max(x0, x1) + r);
        const auto lo_y = (int64_t) std::floor(std::min(y0, y1) - r), hi_y = (int64_t) std::ceil(std::max(y0, y1) + r);
        for (int64_t y = lo_y; y <= hi_y; ++y)
            for (int64_t x = lo_x; x <= hi_x; ++x) {
                float t = len > 0.f ? (((float) x - x0) * (x1 - x0) + ((float) y - y0) * (y1 - y0)) / (len * len) : 0.f;
                t = std::clamp(t, 0.f, 1.f);
                const float dx = (float) x - (x0 + t * (x1 - x0)), dy = (float) y - (y0 + t * (y1 - y0));
                if (dx * dx + dy * dy <= r * r) put(x, y, c);
            }
    }
};

void hsv_rgb(float h, uint8_t* c) {   // a hue in [0, 1) at full saturation and value
    const float k6 = h * 6.f;
    const int i = (int) std::floor(k6) % 6;
    const float f = k6 - std::floor(k6);
    float r = 0, g = 0, b = 0;
    switch (i) {
    case 0: r = 1, g = f, b = 0; break;
    case 1: r = 1 - f, g = 1, b = 0; break;
    case 2: r = 0, g = 1, b = f; break;
    case 3: r = 0, g = 1 - f, b = 1; break;
    case 4: r = f, g = 0, b = 1; break;
    default: r = 1, g = 0, b = 1 - f; break;
    }
    c[0] = (uint8_t) (r * 255.f), c[1] = (uint8_t) (g * 255.f), c[2] = (uint8_t) (b * 255.f);
}

}  // namespace

std::vector<uint8_t> draw_pose(const std::vector<WholeBody>& people, int64_t W, int64_t H, bool body, bool hands,
                               bool face) {
    std::vector<uint8_t> px((size_t) (W * H * 3), 0);
    Canvas cv{px, W, H};
    constexpr float kMin = 0.3f;
    // the drawing's 18 body points from COCO's 17 (the neck between the shoulders), their limbs and colours
    static const int kFromCoco[18] = {0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3};
    static const int kLimbs[17][2] = {{1, 2},  {1, 5},   {2, 3},   {3, 4},  {5, 6},  {6, 7},   {1, 8},   {8, 9}, {9, 10},
                                      {1, 11}, {11, 12}, {12, 13}, {1, 0},  {0, 14}, {14, 16}, {0, 15}, {15, 17}};
    static const uint8_t kColors[18][3] = {{255, 0, 0},   {255, 85, 0},  {255, 170, 0}, {255, 255, 0}, {170, 255, 0},
                                           {85, 255, 0},  {0, 255, 0},   {0, 255, 85},  {0, 255, 170}, {0, 255, 255},
                                           {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},  {170, 0, 255},
                                           {255, 0, 255}, {255, 0, 170}, {255, 0, 85}};
    for (const WholeBody& p : people) {
        if (body) {
            float pt[18][3];
            for (int i = 0; i < 18; ++i) {
                if (kFromCoco[i] >= 0) {
                    const auto& k = p[(size_t) kFromCoco[i]];
                    pt[i][0] = k[0], pt[i][1] = k[1], pt[i][2] = k[2];
                } else {   // the neck: where both shoulders are seen
                    const auto &l = p[5], &r = p[6];
                    pt[i][0] = (l[0] + r[0]) / 2.f, pt[i][1] = (l[1] + r[1]) / 2.f;
                    pt[i][2] = l[2] > kMin && r[2] > kMin ? 1.f : 0.f;
                }
            }
            for (int i = 0; i < 17; ++i) {
                const float* a = pt[kLimbs[i][0]];
                const float* b = pt[kLimbs[i][1]];
                if (a[2] < kMin || b[2] < kMin) continue;
                const uint8_t c[3] = {(uint8_t) ((float) kColors[i][0] * 0.6f), (uint8_t) ((float) kColors[i][1] * 0.6f),
                                      (uint8_t) ((float) kColors[i][2] * 0.6f)};
                const float len = std::hypot(a[0] - b[0], a[1] - b[1]);
                cv.ellipse((a[0] + b[0]) / 2.f, (a[1] + b[1]) / 2.f, len / 2.f, 4.f, std::atan2(b[1] - a[1], b[0] - a[0]), c);
            }
            for (int i = 0; i < 18; ++i)
                if (pt[i][2] >= kMin) cv.disc(pt[i][0], pt[i][1], 4.f, kColors[i]);
        }
        if (hands) {
            for (int h = 0; h < 2; ++h) {   // the left hand's 21 points from 91, the right's from 112
                const int b0 = 91 + 21 * h;
                for (int e = 0; e < 20; ++e) {   // the bones: from the wrist out along each finger
                    const int i0 = e % 4 == 0 ? 0 : e, i1 = e + 1;
                    const auto &a = p[(size_t) b0 + (size_t) i0], &b = p[(size_t) b0 + (size_t) i1];
                    if (a[2] < kMin || b[2] < kMin) continue;
                    uint8_t c[3];
                    hsv_rgb((float) e / 20.f, c);
                    cv.line(a[0], a[1], b[0], b[1], 2.f, c);
                }
                static const uint8_t kBlue[3] = {0, 0, 255};
                for (int i = 0; i < 21; ++i) {
                    const auto& k = p[(size_t) b0 + (size_t) i];
                    if (k[2] >= kMin) cv.disc(k[0], k[1], 4.f, kBlue);
                }
            }
        }
        if (face) {
            static const uint8_t kWhite[3] = {255, 255, 255};
            for (int i = 23; i < 91; ++i) {
                const auto& k = p[(size_t) i];
                if (k[2] >= kMin) cv.disc(k[0], k[1], 3.f, kWhite);
            }
        }
    }
    return px;
}

}  // namespace strata::image
