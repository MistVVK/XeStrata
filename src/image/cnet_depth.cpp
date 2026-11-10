// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/cnet_depth.cpp - Depth Anything V2 Small (cnet_models.hpp), from its paper (Yang et al. 2024) and DPT's
// (Ranftl et al. 2021), on the weights as transformers names them.  The ViT's linear layers go through AuxNet's
// products (prefill::Gemm), its attention through qi_sdpa_ex, the DPT head's convolutions through AuxMapNet.

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

namespace cnet_depth_k {   // the kernels' names (an anonymous namespace mangles the same in every file)

inline void cubic_weights(float t, float* w) {   // Keys' cubic of a = -0.75 at the offsets -1 .. 2 from t's floor
    constexpr float A = -0.75f;
    const float t1 = t + 1.f, t2 = 1.f - t, t3 = 2.f - t;
    w[0] = ((A * t1 - 5.f * A) * t1 + 8.f * A) * t1 - 4.f * A;
    w[1] = ((A + 2.f) * t - (A + 3.f)) * t * t + 1.f;
    w[2] = ((A + 2.f) * t2 - (A + 3.f)) * t2 * t2 + 1.f;
    w[3] = ((A * t3 - 5.f * A) * t3 + 8.f * A) * t3 - 4.f * A;
}

// The patches [P][kp] FP16 of the picture rgb [H][W][3] resized bicubically to nw x nh (pixel centres, the edges
// repeated) and normalized by ImageNet's mean and deviation: patch q's column c p p + y p + x (the patch embedding's
// order), the columns past 3 p p zero.
void patches(sycl::queue& q, const uint8_t* rgb, int64_t W, int64_t H, int64_t nw, int64_t nh, int64_t p, int64_t kp,
             uint16_t* out) {
    const int64_t pw = nw / p, P = (nh / p) * pw;
    const float sx = (float) W / (float) nw, sy = (float) H / (float) nh;
    q.parallel_for(sycl::range<1>((size_t) (P * kp)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t pq = i / kp, col = i - pq * kp;
        if (col >= 3 * p * p) {
            out[i] = 0;
            return;
        }
        const int64_t c = col / (p * p), r = col - c * p * p;
        const int64_t y = (pq / pw) * p + r / p, x = (pq % pw) * p + r % p;
        const float fx = ((float) x + 0.5f) * sx - 0.5f, fy = ((float) y + 0.5f) * sy - 0.5f;
        const float x0 = sycl::floor(fx), y0 = sycl::floor(fy);
        float wx[4], wy[4];
        cubic_weights(fx - x0, wx);
        cubic_weights(fy - y0, wy);
        float v = 0.f;
        for (int b = 0; b < 4; ++b) {
            const int64_t yy = sycl::clamp((int64_t) y0 - 1 + b, (int64_t) 0, H - 1);
            for (int a = 0; a < 4; ++a) {
                const int64_t xx = sycl::clamp((int64_t) x0 - 1 + a, (int64_t) 0, W - 1);
                v += wy[b] * wx[a] * (float) rgb[(yy * W + xx) * 3 + c];
            }
        }
        const float mean = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f, sd = c == 0 ? 0.229f : c == 1 ? 0.224f : 0.225f;
        out[i] = kernels::f16_from_f32((v / 255.f - mean) / sd);
    });
}

// x [1 + P][C]: row 0 the class token, the others the patches' embeddings, each plus its position
void tokens(sycl::queue& q, const float* emb, const float* cls, const float* pos, int64_t P, int64_t C, float* x) {
    q.parallel_for(sycl::range<1>((size_t) ((P + 1) * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        x[i] = (t == 0 ? cls[c] : emb[(t - 1) * C + c]) + pos[i];
    });
}

// a tap of the ViT: the normed rows past the class token as a map [P][C] FP16
void tap(sycl::queue& q, const uint16_t* n16, int64_t P, int64_t C, uint16_t* map16) {
    q.parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) { map16[id[0]] = n16[id[0] + C]; });
}

// out [OH][OW] = channel 0 of x16 [H][W][C] resized bilinearly, the corners aligned
void depth_out(sycl::queue& q, const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t OH, int64_t OW,
               float* out) {
    const float sy = OH > 1 ? (float) (H - 1) / (float) (OH - 1) : 0.f;
    const float sx = OW > 1 ? (float) (W - 1) / (float) (OW - 1) : 0.f;
    q.parallel_for(sycl::range<1>((size_t) (OH * OW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t oy = i / OW, ox = i - oy * OW;
        const float fy = (float) oy * sy, fx = (float) ox * sx;
        const int64_t y0 = sycl::min((int64_t) fy, H - 1), x0 = sycl::min((int64_t) fx, W - 1);
        const int64_t y1 = sycl::min(y0 + 1, H - 1), x1 = sycl::min(x0 + 1, W - 1);
        const float ty = fy - (float) y0, tx = fx - (float) x0;
        const auto at = [&](int64_t yy, int64_t xx) { return kernels::f32_from_f16(x16[(yy * W + xx) * C]); };
        const float top = at(y0, x0) * (1.f - tx) + at(y0, x1) * tx, bot = at(y1, x0) * (1.f - tx) + at(y1, x1) * tx;
        out[i] = top * (1.f - ty) + bot * ty;
    });
}

}  // namespace cnet_depth_k

namespace {

// The position table [n][n][C] resized bicubically (align_corners false, the edges repeated) by the factors (sy, sx)
// to oh x ow, on the host: the positions at the picture's grid (DINOv2's interpolation, its factors (n' + 0.1) / n)
std::vector<float> resize_positions(const float* g, int64_t n, int64_t C, double sy, double sx, int64_t oh,
                                    int64_t ow) {
    std::vector<float> out((size_t) (oh * ow * C), 0.f);
    for (int64_t y = 0; y < oh; ++y) {
        const double fy = ((double) y + 0.5) / sy - 0.5, y0 = std::floor(fy);
        float wy[4];
        cnet_depth_k::cubic_weights((float) (fy - y0), wy);
        for (int64_t x = 0; x < ow; ++x) {
            const double fx = ((double) x + 0.5) / sx - 0.5, x0 = std::floor(fx);
            float wx[4];
            cnet_depth_k::cubic_weights((float) (fx - x0), wx);
            float* o = &out[(size_t) ((y * ow + x) * C)];
            for (int b = 0; b < 4; ++b) {
                const int64_t j = std::clamp<int64_t>((int64_t) y0 - 1 + b, 0, n - 1);
                for (int a = 0; a < 4; ++a) {
                    const int64_t i = std::clamp<int64_t>((int64_t) x0 - 1 + a, 0, n - 1);
                    const float wt = wy[b] * wx[a];
                    const float* src = g + (j * n + i) * C;
                    for (int64_t c = 0; c < C; ++c) o[c] += wt * src[c];
                }
            }
        }
    }
    return out;
}

constexpr float kEps = 1e-6f;
constexpr int64_t kSide = 518;   // the shorter side the model takes the picture at

}  // namespace

void DepthAnything2::load_weights() {
    const std::string e = "backbone.embeddings.", L = "backbone.encoder.layer.";
    const SafeTensor* pe = st_->find(e + "patch_embeddings.projection.weight");
    if (pe == nullptr || pe->shape.size() != 4 || pe->shape[1] != 3)
        throw std::runtime_error(path_ + ": not Depth Anything V2 (transformers' names)");
    C_ = pe->shape[0];
    patch_ = pe->shape[2];
    heads_ = C_ / 64;
    kp_ = up8(3 * patch_ * patch_);
    {   // the patch embedding as a product of the patches in (c, y, x) order, the weight's own, padded to kp
        const auto w = host_f32(e + "patch_embeddings.projection.weight");
        const int64_t K = 3 * patch_ * patch_;
        std::vector<float> wp((size_t) (C_ * kp_), 0.f);
        for (int64_t o = 0; o < C_; ++o) std::copy_n(w.begin() + o * K, K, wp.begin() + o * kp_);
        patch_w_ = lin_host(wp, host_f32(e + "patch_embeddings.projection.bias"), C_, kp_);
    }
    pos_ = host_f32(e + "position_embeddings");
    cls_ = host_f32(e + "cls_token");
    const size_t rows = pos_.size() / (size_t) C_;   // the class token's row and the grid's
    grid_ = (int64_t) std::lround(std::sqrt((double) rows - 1.0));
    layers_.clear();
    for (int i = 0; st_->find(L + std::to_string(i) + ".norm1.weight") != nullptr; ++i) {
        const std::string p = L + std::to_string(i) + ".";
        Layer l;
        l.n1 = norm(p + "norm1");
        l.n2 = norm(p + "norm2");
        l.ls1 = f32(p + "layer_scale1.lambda1");
        l.ls2 = f32(p + "layer_scale2.lambda1");
        const std::string a = p + "attention.attention.";
        l.qkv = lin({a + "query", a + "key", a + "value"});
        l.o = lin(p + "attention.output.dense");
        l.fc1 = lin(p + "mlp.fc1");
        l.fc2 = lin(p + "mlp.fc2");
        layers_.push_back(l);
    }
    if (layers_.size() != 12) throw std::runtime_error(path_ + ": not the 12 layers of Depth Anything V2 Small");
    final_ = norm("backbone.layernorm");
    const std::string R = "neck.reassemble_stage.layers.", F = "neck.fusion_stage.layers.";
    for (int i = 0; i < 4; ++i) {
        const std::string r = R + std::to_string(i) + ".";
        proj_[i] = mconv(r + "projection.weight", r + "projection.bias", 1, 0);
        const SafeTensor* rz = st_->find(r + "resize.weight");
        if (rz == nullptr) {
            resize_kind_[i] = 0;
        } else if (rz->shape[2] == 3) {   // a 3 x 3 convolution of stride 2 (the deepest tap's)
            resize_kind_[i] = -1;
            resize_[i] = mconv(r + "resize.weight", r + "resize.bias", 2, 1);
        } else {                          // a transposed convolution whose kernel is its stride (4, 2)
            resize_kind_[i] = 1;
            resize_[i] = mdeconv(r + "resize.weight", r + "resize.bias", false);
        }
        neck_[i] = mconv("neck.convs." + std::to_string(i) + ".weight", "", 1, 1);
        const std::string f = F + std::to_string(i) + ".";
        fuse_[i] = mconv(f + "projection.weight", f + "projection.bias", 1, 0);
        int j = 0;
        for (const char* res : {"residual_layer1.", "residual_layer2."})
            for (const char* cv : {"convolution1", "convolution2"})
                rcu_[i][j++] = mconv(f + res + cv + ".weight", f + res + cv + ".bias", 1, 1);
    }
    head1_ = mconv("head.conv1.weight", "head.conv1.bias", 1, 1);
    head2_ = mconv("head.conv2.weight", "head.conv2.bias", 1, 1);
    head3_ = mconv("head.conv3.weight", "head.conv3.bias", 1, 0);
}

void DepthAnything2::sides(int64_t W, int64_t H, int64_t& nw, int64_t& nh) const {
    // kSide the lower bound of the shorter side, the ratio kept, each side the nearest multiple of the patch (up where
    // the nearest falls under the bound)
    const double scale = std::max((double) kSide / (double) W, (double) kSide / (double) H);
    const auto fit = [&](double x) {
        int64_t y = (int64_t) std::lround(x / (double) patch_) * patch_;
        if (y < kSide) y = (int64_t) std::ceil(x / (double) patch_) * patch_;
        return y;
    };
    nw = fit(scale * (double) W);
    nh = fit(scale * (double) H);
}

uint64_t DepthAnything2::need_bytes(int64_t W, int64_t H) const {
    const int64_t p = 14, C = C_ > 0 ? C_ : 384;
    const double scale = std::max((double) kSide / (double) W, (double) kSide / (double) H);
    const int64_t nw = (int64_t) std::ceil(scale * (double) W / (double) p) * p + p;
    const int64_t nh = (int64_t) std::ceil(scale * (double) H / (double) p) * p + p;
    const auto T = (uint64_t) ((nw / p) * (nh / p) + 1);
    // the ViT's rows (the stream, its products, q k v, the MLP's), the head's maps at the model's size, the picture
    // and the depth, the libraries' room
    const uint64_t vit = T * (uint64_t) C * (4 + 12 + 4 + 8 + 8);
    const uint64_t head = (uint64_t) (nw * nh) * 64 * (2 * 10 + 4);
    return vit + head + (uint64_t) (W * H) * 8 + (256ull << 20);
}

std::vector<float> DepthAnything2::run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H) {
    load();
    begin_run();
    sycl::queue& q = core::Runtime::get().stream(s_);
    int64_t nw = 0, nh = 0;
    sides(W, H, nw, nh);
    const int64_t ph = nh / patch_, pw = nw / patch_, P = ph * pw, T = P + 1, C = C_;
    auto* src = buf<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    auto* pt = buf<uint16_t>((size_t) (P * kp_));
    cnet_depth_k::patches(q, src, W, H, nw, nh, patch_, kp_, pt);
    float* emb = buf<float>((size_t) (P * C));
    product(pt, patch_w_, emb, P);
    std::vector<float> pos((size_t) (T * C));
    std::copy_n(pos_.begin(), C, pos.begin());
    const auto g = resize_positions(pos_.data() + C, grid_, C, ((double) ph + 0.1) / (double) grid_,
                                    ((double) pw + 0.1) / (double) grid_, ph, pw);
    std::copy(g.begin(), g.end(), pos.begin() + C);
    auto* posd = buf<float>(pos.size());
    gpu::copy(posd, pos.data(), pos.size() * 4);
    auto* clsd = buf<float>((size_t) C);
    gpu::copy(clsd, cls_.data(), (size_t) C * 4);
    auto* x = buf<float>((size_t) (T * C));
    cnet_depth_k::tokens(q, emb, clsd, posd, P, C, x);
    auto* h16 = buf<uint16_t>((size_t) (T * C));
    auto* big = buf<float>((size_t) (T * 3 * C));
    auto* q16 = buf<uint16_t>((size_t) (T * C));
    auto* k16 = buf<uint16_t>((size_t) (T * C));
    auto* v16 = buf<uint16_t>((size_t) (T * C));
    auto* a16 = buf<uint16_t>((size_t) (T * C));
    auto* g16 = buf<uint16_t>((size_t) (T * layers_[0].fc1.n));
    auto* y = buf<float>((size_t) (T * C));
    std::vector<Map> feats;
    for (size_t li = 0; li < layers_.size(); ++li) {
        const Layer& l = layers_[li];
        aux_layer_norm(x, nullptr, l.n1.w, l.n1.b, T, C, kEps, nullptr, h16, s_);
        product(h16, l.qkv, big, T);
        aux_split3_16(big, T, C, q16, k16, v16, s_);
        QiSdpa a;
        a.q = q16, a.k = k16, a.v = v16, a.out = a16;
        a.heads = heads_, a.lq = a.lk = T, a.hd = C / heads_;
        a.q_row = a.k_row = a.v_row = a.o_row = C;
        a.q_batch = a.k_batch = a.v_batch = a.o_batch = T * C;
        a.scale = 1.f / std::sqrt((float) a.hd);
        qi_sdpa_ex(a, s_);
        product(a16, l.o, y, T);
        aux_scale_add(x, y, l.ls1, T, C, s_);
        aux_layer_norm(x, nullptr, l.n2.w, l.n2.b, T, C, kEps, nullptr, h16, s_);
        product16(h16, l.fc1, g16, T, true, ImageAct::gelu);
        product(g16, l.fc2, y, T);
        aux_scale_add(x, y, l.ls2, T, C, s_);
        if (li == 2 || li == 5 || li == 8 || li == 11) {   // the four taps: after layers 3, 6, 9 and 12, normed
            aux_layer_norm(x, nullptr, final_.w, final_.b, T, C, kEps, nullptr, h16, s_);
            Map f = map(ph, pw, C);
            cnet_depth_k::tap(q, h16, P, C, f.p);
            feats.push_back(f);
        }
    }
    // the reassembly: each tap projected and resized (4x, 2x, 1x, 1/2x), then to the fusion's width
    Map lay[4];
    for (int i = 0; i < 4; ++i) {
        Map m = AuxMapNet::run(proj_[i], feats[(size_t) i]);
        if (resize_kind_[i] == 1) m = run_deconv(resize_[i], m);
        else if (resize_kind_[i] == -1) m = AuxMapNet::run(resize_[i], m);
        lay[i] = AuxMapNet::run(neck_[i], m);
    }
    // the fusion from the deepest: the running map plus the pre-activation residual unit of the tap (resized to it
    // where it differs), a second unit, up to the next tap's size (the last: twice) with the corners aligned, and the
    // projection
    const auto unit = [&](const Map& in, const MConv& c1, const MConv& c2) {
        const Map h = AuxMapNet::run(c1, act(in, ImageAct::relu), ImageAct::relu);
        return AuxMapNet::run(c2, h, ImageAct::none, &in);
    };
    Map fused;
    for (int i = 0; i < 4; ++i) {
        const Map& f = lay[3 - i];
        Map h = f;
        if (i > 0) {
            Map r = f;
            if (r.h != fused.h || r.w != fused.w) r = resize(r, fused.h, fused.w, false);
            h = add(fused, unit(r, rcu_[i][0], rcu_[i][1]));
        }
        h = unit(h, rcu_[i][2], rcu_[i][3]);
        const int64_t oh = i < 3 ? lay[2 - i].h : h.h * 2, ow = i < 3 ? lay[2 - i].w : h.w * 2;
        fused = AuxMapNet::run(fuse_[i], resize(h, oh, ow, true));
    }
    Map d = AuxMapNet::run(head1_, fused);
    d = resize(d, nh, nw, true);
    d = AuxMapNet::run(head2_, d, ImageAct::relu);
    d = AuxMapNet::run(head3_, d, ImageAct::relu);
    auto* out = buf<float>((size_t) (W * H));
    cnet_depth_k::depth_out(q, d.p, d.h, d.w, d.c, H, W, out);
    std::vector<float> host((size_t) (W * H));
    gpu::stream_sync(s_);
    gpu::copy(host.data(), out, host.size() * 4);
    return host;
}

}  // namespace strata::image
