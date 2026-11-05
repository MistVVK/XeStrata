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

#if STRATA_ONEDNN || STRATA_CUDNN
#include "qi_conv.hpp"
#endif

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;
namespace syclex = sycl::ext::oneapi::experimental;

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
// Wan 2.1's (ComfyUI's latent_formats.Wan21): Qwen-Image's first VAE, Anima's
const float kWanMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                            0.4134f,  -0.0715f, 0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float kWanStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                           3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
const bool kTemporalUp[5] = {true, true, true, false, false};   // the up blocks whose shortcut also spans time
constexpr int64_t kColBytes = 256ll << 20;                      // the unfolded neighbourhoods a product at most

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

template<class T> T* alloc(std::vector<void*>& owned, int64_t n) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (VAE): GPU memory: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

// out16[p][c] = (z[c][p] * std[c] + mean[c]), the DiT's latent [C][HW] as the decoder's [HW][C]
void unnorm_tokens16(const float* z, const float* mean, const float* sd, int64_t C, int64_t HW, uint16_t* out16,
                     void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        out16[i] = f16_from_f32(z[c * HW + p] * sd[c] + mean[c]);
    });
}

// rms16 a sub-group of 32 a pixel, PIX pixels a work-group
struct Rms16 {
    static constexpr int64_t SG = 32, PIX = 8, LANES = SG * PIX;
    const float *x, *g;
    int64_t HW, C;
    bool silu;
    uint16_t* out16;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t p = static_cast<int64_t>(it.get_group_linear_id()) * PIX + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (p >= HW) return;   // the whole sub-group
        const float* r = x + p * C;
        float ss = 0.f;
        for (int64_t c = lane; c < C; c += SG) ss += r[c] * r[c];
        const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) C + 1e-12f);
        for (int64_t c = lane; c < C; c += SG) {
            float v = r[c] * inv * g[c];
            if (silu) v = v / (1.f + sycl::exp(-v));
            out16[p * C + c] = f16_from_f32(v);
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

// out16[p] = RMS-normed x[p] over the channels times g (SiLU after it when `silu`): a sub-group of 32 a pixel, the
// sum within it (a work-group a pixel and its sum through local memory and barriers took the RX 9060 XT 14.8 ms a
// call on average in a 1024 x 1024 decode)
void rms16(const float* x, const float* g, int64_t HW, int64_t C, bool silu, uint16_t* out16, void* s) {
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((HW + Rms16::PIX - 1) / Rms16::PIX * Rms16::LANES)),
                                           sycl::range<1>((size_t) Rms16::LANES)),
                         Rms16{x, g, HW, C, silu, out16});
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

// qkv [HW][3C] -> q16, k16 [HW][C] and vt16 [C][HW]
void split_qkv16(const float* qkv, int64_t HW, int64_t C, uint16_t* q16, uint16_t* k16, uint16_t* vt16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        q16[i] = f16_from_f32(qkv[p * 3 * C + c]);
        k16[i] = f16_from_f32(qkv[p * 3 * C + C + c]);
        vt16[c * HW + p] = f16_from_f32(qkv[p * 3 * C + 2 * C + c]);
    });
}

// rgba [4][HW] = clamp((x [HW][C] + 1) / 2, 0, 1); C 3: alpha 1
void to_rgba(const float* x, int64_t HW, int64_t C, float* rgba, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        rgba[i] = c < C ? sycl::clamp((x[p * C + c] + 1.f) * 0.5f, 0.f, 1.f) : 1.f;
    });
}

// sub [C][th][tw] = z [C][h][w] rows y0.., columns x0..
void crop_latent(const float* z, int64_t C, int64_t h, int64_t w, int64_t y0, int64_t x0, int64_t th, int64_t tw,
                 float* sub, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (C * th * tw)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / (th * tw), r = i % (th * tw), y = r / tw, x = r % tw;
        sub[i] = z[(c * h + y0 + y) * w + x0 + x];
    });
}

// acc [4][H][W] += piece [4][ph][pw] * weight, wsum [H][W] += weight, the piece at image row Y0, column X0; the weight
// ramps from 0 to 1 over `ramp` pixels at each edge that meets another tile, 1 elsewhere
void blend_add(const float* piece, int64_t ph, int64_t pw, int64_t Y0, int64_t X0, int64_t H, int64_t W, int64_t ramp,
               bool top, bool bottom, bool left, bool right, float* acc, float* wsum, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (ph * pw)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t y = i / pw, x = i % pw;
        auto edge = [ramp](int64_t d) { return sycl::fmin(1.0f, (float) (d + 1) / (float) (ramp + 1)); };
        float wgt = 1.0f;
        if (top) wgt *= edge(y);
        if (bottom) wgt *= edge(ph - 1 - y);
        if (left) wgt *= edge(x);
        if (right) wgt *= edge(pw - 1 - x);
        const int64_t p = (Y0 + y) * W + X0 + x;
        for (int c = 0; c < 4; ++c) acc[c * H * W + p] += piece[c * ph * pw + i] * wgt;
        wsum[p] += wgt;
    });
}

void normalize(float* acc, const float* wsum, int64_t HW, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        acc[i] /= wsum[i % HW];
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
    uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
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
    float* d = alloc<float>(allocs_, (int64_t) v.size());
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

// Wan 2.1's VAE as ComfyUI names it (comfy/ldm/wan/vae.py): conv2 the post-quantization 1 x 1, decoder.conv1, the
// middle's residual block, attention and residual block, the upsamples (residual blocks, each run's resample after
// it), the head's norm and convolution to RGB; the encoder the other way round, conv1 its quantization 1 x 1.  A
// residual block's norms and convolutions are residual.0, .2, .3 and .6, its 1 x 1 shortcut `shortcut`.
void QiVae::open_wan21(SafeTensors& st) {
    wan21_ = true;
    rgb_in_ = 3;
    if (!dry_) {
        mean_ = alloc<float>(allocs_, 16);
        std_ = alloc<float>(allocs_, 16);
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
        if (st.find("decoder.conv1.weight") != nullptr) {   // Wan 2.1's, as ComfyUI has it
            open_wan21(st);
            if (dry_) return true;
            return gemm_.init(s_, 16, err);
        }
        if (!dry_) {
            mean_ = alloc<float>(allocs_, 64);
            std_ = alloc<float>(allocs_, 64);
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
        a16_ = alloc<uint16_t>(allocs_, elems16);
        cap16_ = elems16;
    }
    if (elems32 > cap32_) {
        for (float** p : {&t0_, &t1_, &t2_}) {
            release(allocs_, *p);
            *p = alloc<float>(allocs_, elems32);
        }
        cap32_ = elems32;
    }
    if (col16 > col_cap_) {
        release(allocs_, col16_);
        col16_ = alloc<uint16_t>(allocs_, col16);
        col_cap_ = col16;
    }
}

void QiVae::conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add) {
    const int64_t HW = H * W;
#if STRATA_ONEDNN
    // oneDNN where it takes them (Intel GPUs), the bias and the residual within its pass: the 3 x 3 ones, and the 1 x 1
    // ones that add to a residual (the others gained nothing over the library's product on the B70, 2026-11-02)
    const int p = (int) c.k / 2;
    if ((c.k == 3 || add) &&
        conv::onednn(q_of(s_), in16, H, W, c.in, c.w, c.out, (int) c.k, 1, p, p, p, p, c.b, add, out))
        return;
#endif
#if STRATA_CUDNN
    if ((c.k == 3 || add) && conv::cudnn(q_of(s_), in16, H, W, c.in, c.w, c.out, (int) c.k, c.b, add, out)) return;
#endif
    if (c.k == 1) {
        gemm_.f16(in16, c.w, out, HW, c.out, c.in, 0, false, add);
    } else {
        const int64_t K = 9 * c.in, rows = std::max<int64_t>(1, col_cap_ / K);
        for (int64_t p0 = 0; p0 < HW; p0 += rows) {
            const int64_t n = std::min(rows, HW - p0);
            qi_im2col16(in16, 1, H, W, c.in, 1, 3, 1, 1, H, W, p0, n, col16_, s_);
            gemm_.f16(col16_, c.w, out + p0 * c.out, n, c.out, K, 0, false, add);
        }
    }
    qi_bias_add(out, c.b, HW, c.out, s_);
}

void QiVae::resblock(const Res& r, float*& x, int64_t H, int64_t W, int64_t cin) {
    const int64_t HW = H * W, cout = r.c1.out;
    float* tmp = t2_;
    rms16(x, r.g1, HW, cin, true, a16_, s_);
    conv(r.c1, a16_, H, W, tmp);
    rms16(tmp, r.g2, HW, cout, true, a16_, s_);
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
    rms16(x, g, HW, C, false, a16_, s_);
    conv(qkv_w, a16_, H, W, qkv);
    uint16_t *q16 = a16_, *k16 = a16_ + HW * C, *vt16 = a16_ + 2 * HW * C;
    split_qkv16(qkv, HW, C, q16, k16, vt16, s_);
    float* sc = t2_;
    gemm_.f16(q16, k16, sc, HW, HW, C);
    uint16_t* p16 = col16_;
    qi_softmax16(sc, HW, HW, 1.0f / std::sqrt((float) C), p16, s_);
    gemm_.f16(p16, vt16, qkv, HW, C, HW);   // O = P . V (V^T's rows are V's columns)
    kernels::f32_to_f16_bulk(qkv, a16_, HW * C, s_);
    conv(proj, a16_, H, W, x, true);
}

QiVae::Need QiVae::need(int64_t h, int64_t w) const {
    // the largest activation: each stage's pixels times its widest channel count; the attention's scores [hw][hw]
    Need n;
    n.big32 = h * w * std::max<int64_t>(3 * cin_.out, h * w);
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
    // the convolutions unfold a band of rows at a time into at most kColBytes; the attention's probabilities [hw][hw]
    n.col = std::max(h * w * h * w, std::min(unfold, kColBytes / 2));
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
    // square tiles of t latent pixels, the largest whose decode fits, a quarter of a tile overlapping its neighbours
    int64_t t = std::min(h, w);
    while (t > 8 && scratch_bytes(t, t) > max_scratch) t -= 4;
    if (scratch_bytes(t, t) > max_scratch)
        throw std::runtime_error("image engine (VAE): not enough GPU memory for a tile of the image");
    const int64_t S = scale_, ov = std::max<int64_t>(2, t / 4), step = t - ov, H = S * h, W = S * w, C = pq_.in;
    const int64_t th = std::min(t, h), tw = std::min(t, w);
    float *sub = nullptr, *piece = nullptr, *wsum = nullptr;
    if (!gpu::alloc_device(&sub, (size_t) (C * th * tw) * 4) ||
        !gpu::alloc_device(&piece, (size_t) (4 * S * S * th * tw) * 4) ||
        !gpu::alloc_device(&wsum, (size_t) (H * W) * 4))
        throw std::runtime_error(std::string("image engine (VAE): GPU memory: ") + gpu::last_error());
    gpu::memset_async(rgba, 0, (size_t) (4 * H * W) * 4, s_);
    gpu::memset_async(wsum, 0, (size_t) (H * W) * 4, s_);
    for (int64_t y0 = 0;; y0 += step) {
        const int64_t yy = std::min(y0, h - th);   // the last row of tiles ends at the edge
        for (int64_t x0 = 0;; x0 += step) {
            const int64_t xx = std::min(x0, w - tw);
            crop_latent(z, C, h, w, yy, xx, th, tw, sub, s_);
            decode_piece(sub, th, tw, piece);
            blend_add(piece, S * th, S * tw, S * yy, S * xx, H, W, S * ov, yy > 0, yy + th < h, xx > 0, xx + tw < w,
                      rgba, wsum, s_);
            if (xx + tw >= w) break;
        }
        if (yy + th >= h) break;
    }
    normalize(rgba, wsum, H * W, s_);
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
            keep = alloc<float>(allocs_, H * W * cin);
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
    rms16(x, gout_, H * W, c, true, a16_, s_);
    float* out = x == t0_ ? t1_ : t0_;
    conv(cout_, a16_, H, W, out);
    to_rgba(out, H * W, cout_.out, rgba, s_);
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
    // the middle's attention: q, k, v [hw][c] and the scores [hw][hw]
    n.big32 = std::max(n.big32, h * w * std::max(3 * c, h * w));
    n.big16 = std::max(n.big16, 3 * h * w * c);
    n.big32 = std::max(n.big32, h * w * std::max(eout_.out, quant_.out));
    n.col = std::max(h * w * h * w, std::min(unfold, kColBytes / 2));
    return n;
}

uint64_t QiVae::encode_scratch_bytes(int64_t H, int64_t W) const {
    const Need n = encode_need(H, W);
    return (uint64_t) n.big16 * 2 + (uint64_t) n.big32 * 4 * 3 + (uint64_t) n.keep * 4 + (uint64_t) n.col * 2;
}

bool QiVae::encode(const float* rgba, int64_t H, int64_t W, float* z) {
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
            keep = alloc<float>(allocs_, H * W * cin);
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
            const int64_t Ho = H / 2, Wo = W / 2, K = 9 * c, rows = std::max<int64_t>(1, col_cap_ / K);
            for (int64_t p0 = 0; p0 < Ho * Wo; p0 += rows) {
                const int64_t n = std::min(rows, Ho * Wo - p0);
                qi_im2col16(a16_, 1, H, W, c, 1, 3, 2, 0, H / 2, W / 2, p0, n, col16_, s_);
                gemm_.f16(col16_, d.down.w, y + p0 * d.down.out, n, d.down.out, K);
            }
            qi_bias_add(y, d.down.b, Ho * Wo, d.down.out, s_);
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
    rms16(x, egout_, H * W, c, true, a16_, s_);
    float* y = x == t0_ ? t1_ : t0_;
    conv(eout_, a16_, H, W, y);
    kernels::f32_to_f16_bulk(y, a16_, H * W * eout_.out, s_);
    conv(quant_, a16_, H, W, t2_);
    norm_latent(t2_, H * W, quant_.out, pq_.in, mean_, std_, z, s_);
    gpu::stream_sync(s_);
    return true;
}

}  // namespace strata::image
