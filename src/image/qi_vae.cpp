// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_vae.cpp - see include/strata/image/qi_vae.hpp.

#include "strata/image/qi_vae.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/elementwise.hpp"

#include "tile_sides.hpp"
#include "img_out.hpp"
#include "qi_conv.hpp"
#include "vae_attention.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

using kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using kernels::f16_gpu::f32_from_f16;

// Qwen/Qwen-Image-2.1's vae/config.json: latents_mean and latents_std (the DiT's latent is (z - mean) / std)
const float kMean[64] = {0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f, 1.8873f,
                         -1.7177f, -0.651f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f, 2.021f, 0.0472f, 0.932f,
                         2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f, -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f,
                         -1.4518f, -1.9036f, -1.9955f, -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f,
                         -0.1076f, 1.7376f, -1.0914f, -1.9435f, -0.2784f, -1.368f, 0.4809f, -0.4433f, 0.3764f, 0.5729f,
                         -2.0595f, 1.096f, -1.326f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f, 0.3026f, 1.9373f,
                         1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f};
const float kStd[64] = {3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.791f, 3.0785f, 3.65f,
                        3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.732f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f,
                        2.9249f, 3.9454f, 3.0988f, 4.2489f, 3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.283f,
                        3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.077f, 3.5075f, 3.27f, 3.4767f, 2.8063f, 5.1125f,
                        3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f, 3.9624f, 4.0168f,
                        3.5643f, 4.055f, 5.5614f, 4.2963f, 4.408f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.149f,
                        3.7662f, 3.6746f, 3.4563f, 3.8161f};
// Wan 2.1's (its VAE's latent means and deviations): Qwen-Image's first VAE, Anima's
const float kWanMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                            0.4134f,  -0.0715f, 0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float kWanStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                           3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
const bool kTemporalUp[5] = {true, true, true, false, false};   // the up blocks whose shortcut also spans time
constexpr int64_t kScoreValues = 16ll << 20;                     // the attention's scores a band of query rows at most

// the attention's band of query rows over hw keys: its scores within kScoreValues
int64_t attn_rows(int64_t hw) { return std::max<int64_t>(1, std::min(hw, kScoreValues / std::max<int64_t>(1, hw))); }

// out16[p][c] = (z[c][p] * std[c] + mean[c]), the DiT's latent [C][HW] as the decoder's [HW][C]
void unnorm_tokens16(const float* z, const float* mean, const float* sd, int64_t C, int64_t HW, uint16_t* out16,
                     void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        out16[i] = f16_from_f32(z[c * HW + p] * sd[c] + mean[c]);
    });
}

// y [2H][2W][cout] += the up block's input xin [H][W][cin] spread: channel c of pixel (2h + a, 2w + b) is the input
// channel ((c * ft + ft - 1) * 4 + 2a + b) / rep of pixel (h, w), rep = cout * ft * 4 / cin (one frame, the first
// chunk's last of its ft)
void dupup_add(float* y, const float* xin, int64_t H, int64_t W, int64_t cin, int64_t cout, int64_t ft, void* s) {
    const int64_t rep = cout * ft * 4 / cin;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * H * W * cout)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % cout, P = i / cout, X = P % (2 * W), Y = P / (2 * W);
        const int64_t a = Y % 2, bb = X % 2;
        const int64_t src = ((c * ft + ft - 1) * 4 + 2 * a + bb) / rep;
        y[i] += xin[((Y / 2) * W + X / 2) * cin + src];
    });
}

// out16 [HW][Cp] = rgba [4][HW]'s first n channels scaled to -1 .. 1, the rest zero (the convolution reads 8 at a time)
void rgba_tokens16(const float* rgba, int64_t HW, int64_t Cp, int64_t n, uint16_t* out16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * Cp)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / Cp, c = i - p * Cp;
        out16[i] = f16_from_f32(c < n ? rgba[c * HW + p] * 2.f - 1.f : 0.f);
    });
}

// y [Ho][Wo][cout] += the down block's input xin [H][W][cin] averaged (AvgDown3D on one frame): output channel o is
// the mean of the group of `g` = cin * ft * fs * fs / cout entries of index ((c * ft + t) * fs + a) * fs + b, frame t
// 0 the zero padding in front when ft = 2, (a, b) the pixel's place in its fs x fs block
void avgdown_add(float* y, const float* xin, int64_t H, int64_t W, int64_t cin, int64_t cout, int64_t ft, int64_t fs,
                 void* s) {
    const int64_t Ho = H / fs, Wo = W / fs, g = cin * ft * fs * fs / cout;
    q_of(s).parallel_for(sycl::range<1>((size_t) (Ho * Wo * cout)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % cout, P = i / cout, xo = P % Wo, yo = P / Wo;
        float sum = 0.f;
        for (int64_t k = 0; k < g; ++k) {
            const int64_t idx = o * g + k, b = idx % fs, a = (idx / fs) % fs, t = (idx / (fs * fs)) % ft,
                          c = idx / (fs * fs * ft);
            if (ft == 2 && t == 0) continue;   // the padded frame
            sum += xin[((yo * fs + a) * W + xo * fs + b) * cin + c];
        }
        y[i] += sum / (float) g;
    });
}

// z [C][HW] = (h [HW][ch] channel c < C - mean[c]) / std[c]: the encoder's mean, as the DiT's latent
void norm_latent(const float* h, int64_t HW, int64_t ch, int64_t C, const float* mean, const float* sd, float* z,
                 void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        z[i] = (h[p * ch + c] - mean[c]) / sd[c];
    });
}

}  // namespace

QiVae::QiVae() = default;

QiVae::~QiVae() {
    for (void* p : allocs_) gpu::free(p);
}

QiVae::Conv QiVae::conv_w(SafeTensors& st, const std::string& name) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || (t->shape.size() != 4 && t->shape.size() != 5))
        throw std::runtime_error("image engine (VAE): no " + name);
    Conv c;
    c.out = t->shape[0];
    const int64_t in = t->shape[1];
    c.in = (in + 7) / 8 * 8;   // the input channels padded with zeros to 8s (the unfolding reads 8 at a time)
    // a causal 3-D kernel [out][in][kt][k][k] on one frame after kt - 1 zero frames: its last time slice
    const int64_t kt = t->shape.size() == 5 ? t->shape[2] : 1;
    c.k = t->shape[t->shape.size() - 1];
    if (dry_) {
        weight_bytes_ += (uint64_t) (c.out * c.k * c.k * c.in) * 2;
        c.b = vec(st, name + ".bias");
        return c;
    }
    const auto w = st.f32(name + ".weight");   // [out][in][kt][k][k]
    std::vector<uint16_t> h((size_t) (c.out * c.k * c.k * c.in));
    for (int64_t o = 0; o < c.out; ++o)
        for (int64_t i = 0; i < in; ++i)
            for (int64_t dy = 0; dy < c.k; ++dy)
                for (int64_t dx = 0; dx < c.k; ++dx)
                    h[(size_t) (((o * c.k + dy) * c.k + dx) * c.in + i)] = kernels::f16_from_f32(
                        w[(size_t) ((((o * in + i) * kt + kt - 1) * c.k + dy) * c.k + dx)]);
    uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (VAE)");
    gpu::copy(d, h.data(), h.size() * 2);
    c.w = d;
    c.b = vec(st, name + ".bias");
    weight_bytes_ += h.size() * 2;
    return c;
}

const float* QiVae::vec(SafeTensors& st, const std::string& name) {
    if (dry_) {
        const SafeTensor* t = st.find(name);
        if (t == nullptr) throw std::runtime_error("image engine (VAE): no " + name);
        weight_bytes_ += (uint64_t) t->elements() * 4;
        return nullptr;
    }
    const auto v = st.f32(name);
    float* d = alloc<float>(allocs_, (int64_t) v.size(), "image engine (VAE)");
    gpu::copy(d, v.data(), v.size() * 4);
    weight_bytes_ += v.size() * 4;
    return d;
}

QiVae::Res QiVae::res_w(SafeTensors& st, const std::string& p) {
    Res r;
    r.g1 = vec(st, p + "norm1.gamma");
    r.g2 = vec(st, p + "norm2.gamma");
    r.c1 = conv_w(st, p + "conv1");
    r.c2 = conv_w(st, p + "conv2");
    if (st.find(p + "conv_shortcut.weight") != nullptr) r.sc = conv_w(st, p + "conv_shortcut");
    return r;
}

// Wan 2.1's VAE by its original checkpoint's names: conv2 the post-quantization 1 x 1, decoder.conv1, the
// middle's residual block, attention and residual block, the upsamples (residual blocks, each run's resample after
// it), the head's norm and convolution to RGB; the encoder the other way round, conv1 its quantization 1 x 1.  A
// residual block's norms and convolutions are residual.0, .2, .3 and .6, its 1 x 1 shortcut `shortcut`.
void QiVae::open_wan21(SafeTensors& st) {
    wan21_ = true;
    rgb_in_ = 3;
    if (!dry_) {
        mean_ = alloc<float>(allocs_, 16, "image engine (VAE)");
        std_ = alloc<float>(allocs_, 16, "image engine (VAE)");
        gpu::copy((void*) mean_, kWanMean, sizeof(kWanMean));
        gpu::copy((void*) std_, kWanStd, sizeof(kWanStd));
    }
    auto res = [&](const std::string& p) {
        Res r;
        r.g1 = vec(st, p + "residual.0.gamma");
        r.g2 = vec(st, p + "residual.3.gamma");
        r.c1 = conv_w(st, p + "residual.2");
        r.c2 = conv_w(st, p + "residual.6");
        if (st.find(p + "shortcut.weight") != nullptr) r.sc = conv_w(st, p + "shortcut");
        return r;
    };
    pq_ = conv_w(st, "conv2");
    cin_ = conv_w(st, "decoder.conv1");
    mid0_ = res("decoder.middle.0.");
    mid1_ = res("decoder.middle.2.");
    ag_ = vec(st, "decoder.middle.1.norm.gamma");
    aqkv_ = conv_w(st, "decoder.middle.1.to_qkv");
    aproj_ = conv_w(st, "decoder.middle.1.proj");
    up_.assign(1, Up{});
    scale_ = 1;
    for (int i = 0;; ++i) {
        const std::string p = "decoder.upsamples." + std::to_string(i) + ".";
        if (st.find(p + "residual.2.weight") != nullptr) {
            up_.back().r.push_back(res(p));
        } else if (st.find(p + "resample.1.weight") != nullptr) {
            up_.back().up = conv_w(st, p + "resample.1");
            up_.back().ft = 0;
            up_.push_back(Up{});
            scale_ *= 2;
        } else {
            break;
        }
    }
    up_.back().ft = 0;
    if (up_.back().r.empty()) up_.pop_back();
    gout_ = vec(st, "decoder.head.0.gamma");
    cout_ = conv_w(st, "decoder.head.2");
    if (st.find("encoder.conv1.weight") == nullptr) return;
    ein_ = conv_w(st, "encoder.conv1");
    down_.assign(1, Down{});
    for (int i = 0;; ++i) {
        const std::string p = "encoder.downsamples." + std::to_string(i) + ".";
        if (st.find(p + "residual.2.weight") != nullptr) {
            down_.back().r.push_back(res(p));
        } else if (st.find(p + "resample.1.weight") != nullptr) {
            down_.back().down = conv_w(st, p + "resample.1");
            down_.back().ft = 0;
            down_.push_back(Down{});
        } else {
            break;
        }
    }
    down_.back().ft = 0;
    if (down_.back().r.empty()) down_.pop_back();
    emid0_ = res("encoder.middle.0.");
    emid1_ = res("encoder.middle.2.");
    eag_ = vec(st, "encoder.middle.1.norm.gamma");
    eaqkv_ = conv_w(st, "encoder.middle.1.to_qkv");
    eaproj_ = conv_w(st, "encoder.middle.1.proj");
    egout_ = vec(st, "encoder.head.0.gamma");
    eout_ = conv_w(st, "encoder.head.2");
    quant_ = conv_w(st, "conv1");
    has_encoder_ = true;
}

bool QiVae::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        SafeTensors st(path);
        if (st.find("decoder.conv1.weight") != nullptr) {   // Wan 2.1's, its keys as decoder.conv1 and so on
            open_wan21(st);
            if (dry_) return true;
            return gemm_.init(s_, 16, err);
        }
        if (!dry_) {
            mean_ = alloc<float>(allocs_, 64, "image engine (VAE)");
            std_ = alloc<float>(allocs_, 64, "image engine (VAE)");
            gpu::copy((void*) mean_, kMean, sizeof(kMean));
            gpu::copy((void*) std_, kStd, sizeof(kStd));
        }
        pq_ = conv_w(st, "post_quant_conv");
        cin_ = conv_w(st, "decoder.conv_in");
        mid0_ = res_w(st, "decoder.mid_block.resnets.0.");
        mid1_ = res_w(st, "decoder.mid_block.resnets.1.");
        ag_ = vec(st, "decoder.mid_block.attentions.0.norm.gamma");
        aqkv_ = conv_w(st, "decoder.mid_block.attentions.0.to_qkv");
        aproj_ = conv_w(st, "decoder.mid_block.attentions.0.proj");
        up_.assign(5, Up{});
        for (int i = 0; i < 5; ++i) {
            const std::string p = "decoder.up_blocks." + std::to_string(i) + ".";
            for (int j = 0; j < 3; ++j) up_[i].r.push_back(res_w(st, p + "resnets." + std::to_string(j) + "."));
            if (st.find(p + "upsampler.resample.1.weight") != nullptr) up_[i].up = conv_w(st, p + "upsampler.resample.1");
            up_[i].ft = kTemporalUp[i] ? 2 : 1;
        }
        gout_ = vec(st, "decoder.norm_out.gamma");
        cout_ = conv_w(st, "decoder.conv_out");
        if (st.find("encoder.conv_in.weight") != nullptr) {   // the encoder, for the edits' reference images
            ein_ = conv_w(st, "encoder.conv_in");
            down_.assign(5, Down{});
            for (int i = 0; i < 5; ++i) {
                const std::string p = "encoder.down_blocks." + std::to_string(i) + ".";
                for (int j = 0; j < 2; ++j) down_[i].r.push_back(res_w(st, p + "resnets." + std::to_string(j) + "."));
                if (st.find(p + "downsampler.resample.1.weight") != nullptr)
                    down_[i].down = conv_w(st, p + "downsampler.resample.1");
                down_[i].ft = st.find(p + "downsampler.time_conv.weight") != nullptr ? 2 : 1;
            }
            emid0_ = res_w(st, "encoder.mid_block.resnets.0.");
            emid1_ = res_w(st, "encoder.mid_block.resnets.1.");
            eag_ = vec(st, "encoder.mid_block.attentions.0.norm.gamma");
            eaqkv_ = conv_w(st, "encoder.mid_block.attentions.0.to_qkv");
            eaproj_ = conv_w(st, "encoder.mid_block.attentions.0.proj");
            egout_ = vec(st, "encoder.norm_out.gamma");
            eout_ = conv_w(st, "encoder.conv_out");
            quant_ = conv_w(st, "quant_conv");
            has_encoder_ = true;
        }
        if (dry_) return true;
        if (!gemm_.init(s_, 16, err)) return false;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void QiVae::grow(int64_t elems16, int64_t elems32, int64_t col16) {
    if (elems16 > cap16_) {
        release(allocs_, a16_);
        a16_ = alloc<uint16_t>(allocs_, elems16, "image engine (VAE)");
        cap16_ = elems16;
    }
    if (elems32 > cap32_) {
        for (float** p : {&t0_, &t1_, &t2_}) {
            release(allocs_, *p);
            *p = alloc<float>(allocs_, elems32, "image engine (VAE)");
        }
        cap32_ = elems32;
    }
    if (col16 > col_cap_) {
        release(allocs_, col16_);
        col16_ = alloc<uint16_t>(allocs_, col16, "image engine (VAE)");
        col_cap_ = col16;
    }
}

void QiVae::conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add) {
    const int64_t HW = H * W, K = c.k * c.k * c.in;
    const int p = (int) c.k / 2;
    conv::Request r{in16, H, W, c.in, c.w, c.out, (int) c.k, 1, p, p, p, p};
    r.bias = c.b;
    r.add = add;
    r.out = out;
    conv::drive(q_of(s_), r, HW, K, col_cap_, c.k == 1,
        [&](int64_t p0, int64_t n) { qi_im2col16(in16, 1, H, W, c.in, 1, (int) c.k, 1, p, H, W, p0, n, col16_, s_); },
        [&](int64_t p0, int64_t n, bool point) {
            gemm_.f16(point ? in16 : col16_, c.w, out + p0 * c.out, n, c.out, point ? c.in : K, 0, false, add);
        },
        [&] { if (c.b != nullptr) qi_bias_add(out, c.b, HW, c.out, s_); });
}

void QiVae::resblock(const Res& r, float*& x, int64_t H, int64_t W, int64_t cin) {
    const int64_t HW = H * W, cout = r.c1.out;
    float* tmp = t2_;
    vae_mid::rms16(x, r.g1, HW, cin, true, a16_, s_);
    conv(r.c1, a16_, H, W, tmp);
    vae_mid::rms16(tmp, r.g2, HW, cout, true, a16_, s_);
    if (r.sc.w != nullptr) {
        float* y = x == t0_ ? t1_ : t0_;   // the block's output: the shortcut's channels are not x's
        conv(r.c2, a16_, H, W, y);
        kernels::f32_to_f16_bulk(x, a16_, HW * cin, s_);
        conv(r.sc, a16_, H, W, y, true);
        x = y;
    } else {
        conv(r.c2, a16_, H, W, x, true);   // x itself takes the sum
    }
}

void QiVae::attention(float* x, int64_t H, int64_t W, int64_t C, const float* g, const Conv& qkv_w, const Conv& proj) {
    const int64_t HW = H * W;
    float* qkv = x == t0_ ? t1_ : t0_;
    vae_mid::Attention a{H, W, C, attn_rows(HW)};
    a.norm = vae_mid::Norm::rms; a.x = x; a.weight = g; a.normalized = a16_;
    a.qkv = qkv;
    a.q = a16_; a.k = a16_ + HW * C; a.vt = a16_ + 2 * HW * C;
    a.p = col16_; a.scores = t2_; a.out = qkv;
    vae_mid::run(a, gemm_, s_,
        [&] { conv(qkv_w, a16_, H, W, qkv); },
        [&] {
            conv(proj, a16_, H, W, x, true);
        });
}

QiVae::Need QiVae::need(int64_t h, int64_t w) const {
    // the largest activation: each stage's pixels times its widest channel count; the attention's scores [hw][hw]
    Need n;
    n.big32 = h * w * std::max<int64_t>(3 * cin_.out, attn_rows(h * w));
    n.big16 = 3 * h * w * cin_.out;
    int64_t H = h, W = w, c = cin_.out, unfold = 0;   // the largest stage's unfolded neighbourhoods
    for (const Up& u : up_) {
        const int64_t co = u.r[0].c1.out;
        n.big32 = std::max(n.big32, H * W * std::max(c, co));
        n.big16 = std::max(n.big16, H * W * std::max(c, co));
        unfold = std::max(unfold, H * W * 9 * std::max(c, co));
        if (u.up.out > 0) {
            n.keep = std::max(n.keep, H * W * c);
            H *= 2;
            W *= 2;
            n.big32 = std::max(n.big32, H * W * co);
            n.big16 = std::max(n.big16, H * W * co);
            unfold = std::max(unfold, H * W * 9 * co);
            c = u.up.out;
            continue;
        }
        c = co;
    }
    // The convolutions and attention share the allocation; each keeps its own band.
    n.col = std::max(attn_rows(h * w) * h * w, std::min(unfold, conv::col_values(q_of(s_))));
    return n;
}

uint64_t QiVae::scratch_bytes(int64_t h, int64_t w) const {
    const Need n = need(h, w);
    return (uint64_t) n.big16 * 2 + (uint64_t) n.big32 * 4 * 3 + (uint64_t) n.keep * 4 + (uint64_t) n.col * 2;
}

void QiVae::drop_scratch() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) a16_, (void*) col16_, (void*) t0_, (void*) t1_, (void*) t2_}) release(allocs_, p);
    a16_ = col16_ = nullptr;
    t0_ = t1_ = t2_ = nullptr;
    cap16_ = cap32_ = col_cap_ = 0;
}

void QiVae::decode(const float* z, int64_t h, int64_t w, float* rgba, uint64_t max_scratch) {
    if (scratch_bytes(h, w) <= max_scratch) {
        decode_piece(z, h, w, rgba);
        return;
    }
    // tiles that overlap their neighbours by a quarter of the largest square whose decode fits, as even as that
    // leaves and of the least work (tile_sides.hpp), each at least the two overlaps its blend ramps over
    int64_t t = std::min(h, w);
    while (t > 8 && scratch_bytes(t, t) > max_scratch) t -= 4;
    if (scratch_bytes(t, t) > max_scratch)
        throw std::runtime_error("image engine (VAE): not enough GPU memory for a tile of the image");
    const int64_t S = scale_, ov = std::max<int64_t>(2, t / 4), H = S * h, W = S * w, C = pq_.in;
    auto [th, tw] = tile_sides(h, w, ov, 2 * ov,
                               [&](int64_t a, int64_t b) { return scratch_bytes(a, b) <= max_scratch; });
    if (th == 0) {
        th = std::min(t, h);
        tw = std::min(t, w);
    }
    float *sub = nullptr, *piece = nullptr, *wsum = nullptr;
    if (!gpu::alloc_device(&sub, (size_t) (C * th * tw) * 4) ||
        !gpu::alloc_device(&piece, (size_t) (4 * S * S * th * tw) * 4) ||
        !gpu::alloc_device(&wsum, (size_t) (H * W) * 4))
        throw std::runtime_error(std::string("image engine (VAE): GPU memory: ") + gpu::last_error());
    gpu::memset_async(rgba, 0, (size_t) (4 * H * W) * 4, s_);
    gpu::memset_async(wsum, 0, (size_t) (H * W) * 4, s_);
    const auto ys = tile_positions(h, th, ov), xs = tile_positions(w, tw, ov);
    for (const int64_t yy : ys)
        for (const int64_t xx : xs) {
            tile_crop(z, C, TilePlanes::Planes, h, w, yy, xx, th, tw, sub, s_);
            decode_piece(sub, th, tw, piece);
            tile_blend(piece, 4, TilePlanes::Planes, S * th, S * tw, S * yy, S * xx, H, W, S * ov, TileRamp::Step,
                       yy > 0, yy + th < h, xx > 0, xx + tw < w, rgba, wsum, s_);
        }
    tile_finish(rgba, 4, TilePlanes::Planes, H * W, wsum, rgba, s_);
    gpu::stream_sync(s_);
    gpu::free(sub);
    gpu::free(piece);
    gpu::free(wsum);
}

void QiVae::decode_piece(const float* z, int64_t h, int64_t w, float* rgba) {
    const Need nd = need(h, w);
    grow(nd.big16, nd.big32, nd.col);
    int64_t H = h, W = w, c = cin_.out;
    float* x = t0_;
    unnorm_tokens16(z, mean_, std_, pq_.in, H * W, a16_, s_);
    conv(pq_, a16_, H, W, t1_);
    kernels::f32_to_f16_bulk(t1_, a16_, H * W * pq_.out, s_);
    conv(cin_, a16_, H, W, x);
    resblock(mid0_, x, H, W, c);
    attention(x, H, W, c, ag_, aqkv_, aproj_);
    resblock(mid1_, x, H, W, c);
    for (const Up& u : up_) {
        const int64_t cin = c, co = u.r[0].c1.out;
        // the block's input, for its shortcut: in the buffer neither the residual blocks nor the upsampling write
        float* keep = nullptr;
        if (u.up.out > 0 && u.ft > 0) {
            keep = alloc<float>(allocs_, H * W * cin, "image engine (VAE)");
            gpu::copy_async(keep, x, (size_t) (H * W * cin) * 4, s_);
        }
        for (const Res& r : u.r) {
            resblock(r, x, H, W, c);
            c = r.c1.out;
        }
        if (u.up.out > 0) {
            qi_up2x16(x, H, W, co, a16_, s_);
            float* y = x == t0_ ? t1_ : t0_;
            conv(u.up, a16_, 2 * H, 2 * W, y);
            if (keep != nullptr) {
                dupup_add(y, keep, H, W, cin, co, u.ft, s_);
                gpu::stream_sync(s_);
                release(allocs_, keep);
            }
            x = y;
            H *= 2;
            W *= 2;
            c = u.up.out;   // Wan 2.1's upsampler halves the channels
        }
    }
    vae_mid::rms16(x, gout_, H * W, c, true, a16_, s_);
    float* out = x == t0_ ? t1_ : t0_;
    conv(cout_, a16_, H, W, out);
    to_rgba(out, H * W, cout_.out, TilePlanes::Interleaved, rgba, s_);
    gpu::stream_sync(s_);
}

QiVae::Need QiVae::encode_need(int64_t H, int64_t W) const {
    Need n;
    int64_t h = H, w = W, c = ein_.in, unfold = H * W * 9 * ein_.in;
    n.big32 = n.big16 = H * W * std::max(c, ein_.out);
    c = ein_.out;
    for (const Down& d : down_) {
        const int64_t co = d.r[0].c1.out;
        n.big32 = std::max(n.big32, h * w * std::max(c, co));
        n.big16 = std::max(n.big16, h * w * std::max(c, co));
        unfold = std::max(unfold, h * w * 9 * std::max(c, co));
        n.keep = std::max(n.keep, h * w * c);
        if (d.down.out > 0) {
            h /= 2;
            w /= 2;
            unfold = std::max(unfold, h * w * 9 * co);
        }
        c = co;
    }
    // the middle's attention: q, k, v [hw][c] and a band's scores
    n.big32 = std::max(n.big32, h * w * std::max(3 * c, attn_rows(h * w)));
    n.big16 = std::max(n.big16, 3 * h * w * c);
    n.big32 = std::max(n.big32, h * w * std::max(eout_.out, quant_.out));
    n.col = std::max(attn_rows(h * w) * h * w, std::min(unfold, conv::col_values(q_of(s_))));
    return n;
}

uint64_t QiVae::encode_scratch_bytes(int64_t H, int64_t W) const {
    const Need n = encode_need(H, W);
    return (uint64_t) n.big16 * 2 + (uint64_t) n.big32 * 4 * 3 + (uint64_t) n.keep * 4 + (uint64_t) n.col * 2;
}

bool QiVae::encode(const float* rgba, int64_t H, int64_t W, float* z, float* zstd) {
    if (!has_encoder_) return false;
    if (H % 16 != 0 || W % 16 != 0 || H <= 0 || W <= 0)
        throw std::invalid_argument("image engine (VAE): the sides must be multiples of 16");
    const Need nd = encode_need(H, W);
    grow(nd.big16, nd.big32, nd.col);
    rgba_tokens16(rgba, H * W, ein_.in, rgb_in_, a16_, s_);
    float* x = t0_;
    conv(ein_, a16_, H, W, x);
    int64_t c = ein_.out;
    for (const Down& d : down_) {
        const int64_t cin = c;
        float* keep = nullptr;   // the block's input, for its shortcut
        if (d.ft > 0) {
            keep = alloc<float>(allocs_, H * W * cin, "image engine (VAE)");
            gpu::copy_async(keep, x, (size_t) (H * W * cin) * 4, s_);
        }
        for (const Res& r : d.r) {
            resblock(r, x, H, W, c);
            c = r.c1.out;
        }
        int64_t fs = 1;
        if (d.down.out > 0) {   // zero row and column after the edge, then the 3 x 3 convolution of stride 2
            kernels::f32_to_f16_bulk(x, a16_, H * W * c, s_);
            float* y = x == t0_ ? t1_ : t0_;
            const int64_t Ho = H / 2, Wo = W / 2, P = Ho * Wo, K = 9 * c;
            conv::Request r{a16_, H, W, c, d.down.w, d.down.out, 3, 2, 0, 0, 1, 1};
            r.bias = d.down.b;
            r.out = y;
            conv::drive(q_of(s_), r, P, K, col_cap_, false,
                [&](int64_t p0, int64_t n) { qi_im2col16(a16_, 1, H, W, c, 1, 3, 2, 0, Ho, Wo, p0, n, col16_, s_); },
                [&](int64_t p0, int64_t n, bool) { gemm_.f16(col16_, d.down.w, y + p0 * d.down.out, n, d.down.out, K); },
                [&] { qi_bias_add(y, d.down.b, P, d.down.out, s_); });
            x = y;
            fs = 2;
        }
        if (keep != nullptr) {
            avgdown_add(x, keep, H, W, cin, c, d.ft, fs, s_);
            gpu::stream_sync(s_);
            release(allocs_, keep);
        }
        H /= fs;
        W /= fs;
    }
    resblock(emid0_, x, H, W, c);
    attention(x, H, W, c, eag_, eaqkv_, eaproj_);
    resblock(emid1_, x, H, W, c);
    vae_mid::rms16(x, egout_, H * W, c, true, a16_, s_);
    float* y = x == t0_ ? t1_ : t0_;
    conv(eout_, a16_, H, W, y);
    kernels::f32_to_f16_bulk(y, a16_, H * W * eout_.out, s_);
    conv(quant_, a16_, H, W, t2_);
    norm_latent(t2_, H * W, quant_.out, pq_.in, mean_, std_, z, s_);
    if (zstd != nullptr) {   // the posterior's log-variance: the quant conv's second half of channels
        const float* h = t2_;
        const int64_t HW = H * W, ch = quant_.out, C = pq_.in;
        const float* sd = std_;
        q_of(s_).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
            const auto i = static_cast<int64_t>(id[0]);
            const int64_t c = i / HW, p = i - c * HW;
            zstd[i] = sycl::exp(0.5f * sycl::clamp(h[p * ch + C + c], -30.f, 20.f)) / sd[c];
        });
    }
    gpu::stream_sync(s_);
    return true;
}

}  // namespace strata::image
