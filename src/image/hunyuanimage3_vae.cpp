// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_vae.cpp - see include/strata/image/hunyuanimage3_vae.hpp.
//
// The graph holds to the file's 280 tensors and Tencent's config (VAE.md's decode table, checked against
// the record's hy3_vae_ref.py stage by stage at rel <= 3.2e-6); the arithmetic follows the reference's
// autocast(fp16) as
// VAE.md measured it: the convolutions read FP16 and their products come out FP32 (prefill::Gemm's f16 path,
// the one the DiT uses), the residual stream, the group norms and the attention's softmax are FP32.  The DCAE
// channel split is einops "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)" with a shortcut of the input
// repeat_interleave'd and split the same way - neither ever materialized wider than a convolution's band.

#include "strata/image/hunyuanimage3_vae.hpp"

#include "qi_conv.hpp"
#include "vae_attention.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

using strata::kernels::f16_from_f32;

constexpr int64_t kGroups = 32;             // every GroupNorm here: nn.GroupNorm(32, C, eps = 1e-6, affine)
constexpr float kEps = 1e-6f;

// z [C][HW] (the DiT's latent, model space, FP32) -> the decoder's input: in16 [HW][C] = FP16(z[c][p] / sf) for
// the convolutions and x32 [HW][C] = the same value in FP32 for conv_in's repeat_interleave shortcut.  The
// reference's pipeline divides the latents by scaling_factor before it calls decode(); this does the same, in
// FP32 as torch's `latents / scalar` on an FP32 tensor does - no FP64 in any kernel here, so a device without
// the fp64 aspect (the processor's own graphics) runs the decoder too.
void latent_tokens(const float* z, float sf, int64_t C, int64_t HW, uint16_t* in16, float* x32, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        const float v = z[i] / sf;
        in16[p * C + c] = f16_from_f32(v);
        x32[p * C + c] = v;
    });
}

// x [HW][O] += tok [HW][I] with every input channel repeated O / I (conv_in's shortcut: latent channel c opens
// decoder channels rep*c .. rep*c+rep-1 - the reference's z.repeat_interleave(32, dim = 1), no weights)
void rep_add(float* x, const float* tok, int64_t HW, int64_t O, int64_t I, void* s) {
    const int64_t rep = O / I;
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * O)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % O, p = i / O;
        x[i] += tok[p * I + o / rep];
    });
}

// The group norm's statistics are qi_group_norm16's: deterministic as the golden's
// comparisons need one.  The workspace they take is sized for the most positions one GroupNorm here can span.
constexpr int64_t kMaxGnPositions = (int64_t) 256 * 256 * 256;   // 16.7 M positions
constexpr int64_t kWsFloats = 2 * kGroups + (kMaxGnPositions / 256 + kMaxGnPositions / 256 / 256) * kGroups;

// The UpsampleDCAE's split: out [(f r1 + i1, h 2 + i2, w 2 + i3)][c] = conv[(f, h, w)][m C + c] +
// x[(f, h, w)][(m C + c) / rep] with m = (i1 2 + i2) 2 + i3 - einops "b (r1 r2 r3 c) f h w -> b c (f r1)
// (h r2) (w r3)" on the conv's FP32 output [P][4 r1 C] plus the weightless shortcut (the input
// repeat_interleave'd rep = 4 r1 C / I times, split the same way).  Nothing wider than the conv's own output is
// materialized - the torch reference's OOM at 1024 x 1024 is its repeat_interleave and split of the FULL
// tensor; a gather over a convolution band here costs no tensor at all.
void dcae_split(const float* conv, const float* x, int64_t T, int64_t H, int64_t W, int64_t I, int64_t C,
                int64_t r1, float* out, void* s) {
    const int64_t factor = 4 * r1, rep = factor * C / I;
    const int64_t TP = T * r1, HP = 2 * H, WP = 2 * W;
    q_of(s).parallel_for(sycl::range<1>((size_t) (TP * HP * WP * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, op = i / C;
        const int64_t xp = op % WP, yp = (op / WP) % HP, tp = op / (WP * HP);
        const int64_t i1 = tp % r1, f = tp / r1, i2 = yp % 2, hh = yp / 2, i3 = xp % 2, ww = xp / 2;
        const int64_t m = (i1 * 2 + i2) * 2 + i3, p = (f * H + hh) * W + ww;
        out[i] = conv[p * factor * C + m * C + c] + x[p * I + (m * C + c) / rep];
    });
}

// The encoder's input: rgb [3][HW] (-1 .. 1, FP32) as the T replicated frames' FP16 rows in16 [T HW][Cp], the
// three channels in the first of the Cp (8: the unfold's vec8) and zeros in the rest.  The reference's encode
// expands a one-frame image to ffactor_temporal frames before conv_in.
void rgb_frames16(const float* rgb, int64_t HW, int64_t T, int64_t Cp, uint16_t* in16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * HW * Cp)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % Cp, p = (i / Cp) % HW;
        in16[i] = c < 3 ? f16_from_f32(rgb[c * HW + p]) : (uint16_t) 0;
    });
}

// The DownsampleDCAE's fold: out [(f, hh, ww)][m Cc + c] = conv[(f r1 + i1, 2 hh + i2, 2 ww + i3)][c] with
// m = (i1 2 + i2) 2 + i3, plus the shortcut: the input x [P][I] folded the same way into factor * I channels
// and averaged in groups of `group` = factor I / (factor Cc) consecutive folded channels - einops "b c (f r1)
// (h r2) (w r3) -> b (r1 r2 r3 c) f h w" on both, then view(out, group).mean(2) on the input's
void dcae_fold_down(const float* conv, const float* x, int64_t T, int64_t H, int64_t W, int64_t Cc, int64_t I,
                    int64_t r1, int64_t group, float* out, void* s) {
    const int64_t TP = T / r1, HP = H / 2, WP = W / 2, O = 4 * r1 * Cc;
    q_of(s).parallel_for(sycl::range<1>((size_t) (TP * HP * WP * O)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % O, op = i / O;
        const int64_t ww = op % WP, hh = (op / WP) % HP, f = op / (WP * HP);
        const int64_t m = o / Cc, c = o % Cc;
        const int64_t i1 = m / 4, i2 = (m / 2) % 2, i3 = m % 2;
        const int64_t p = ((f * r1 + i1) * H + 2 * hh + i2) * W + 2 * ww + i3;
        float sum = 0.f;
        for (int64_t g = 0; g < group; ++g) {
            const int64_t fc = o * group + g, mm = fc / I, ci = fc % I;
            const int64_t j1 = mm / 4, j2 = (mm / 2) % 2, j3 = mm % 2;
            sum += x[(((f * r1 + j1) * H + 2 * hh + j2) * W + 2 * ww + j3) * I + ci];
        }
        out[i] = conv[p * Cc + c] + sum / (float) group;
    });
}

// The encoder's head: moment m = conv_out[p][m] + mean_{r < group} h[p][m group + r] - the reference's shortcut
// "b (c r) f h w -> b c r f h w".mean(2) added to conv_out's 2 zc channels, of which the first zc are the
// posterior's mean and the next zc its log-variance - and z[c][p] = sf * (mean + exp(logvar / 2) eps[c][p])
// (logvar clamped to -30 .. 20, the reference's DiagonalGaussianDistribution; eps null: the mean); channels-first,
// as the DiT's latent is
void encoder_head(const float* conv, const float* h, const float* eps, int64_t P, int64_t Cin, int64_t Cout,
                  int64_t group, int64_t zc, float sf, float* z, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (P * zc)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / P, p = i % P;
        auto moment = [&](int64_t m) {
            float sum = 0.f;
            for (int64_t r = 0; r < group; ++r) sum += h[p * Cin + m * group + r];
            return conv[p * Cout + m] + sum / (float) group;
        };
        float v = moment(c);
        if (eps != nullptr) v += sycl::exp(0.5f * sycl::clamp(moment(zc + c), -30.f, 20.f)) * eps[i];
        z[i] = sf * v;
    });
}

// rgb [3][H W] = the LAST of the T frames: the reference's decode keeps decoded[:, :, -1:] (the four frames are
// different pictures 13 - 22 dB apart, VAE.md's measurement - the wrong one is a wrong image, not a detail)
void last_frame(const float* x, int64_t T, int64_t H, int64_t W, float* rgb, void* s) {
    const int64_t HW = H * W;
    q_of(s).parallel_for(sycl::range<1>((size_t) (3 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, r = i - c * HW;
        rgb[i] = x[((T - 1) * HW + r) * 3 + c];
    });
}

}  // namespace

HunyuanImage3Vae::HunyuanImage3Vae() = default;

HunyuanImage3Vae::~HunyuanImage3Vae() {
    for (void* p : allocs_) gpu::free(p);
}

// A 3 x 3 x 3 convolution's weight [O][I][3][3][3] (F16 in the file) as the product's W[O][kt * 9 I] with the
// taps in (dt, dy, dx) order, matching im2col3d.  kt 1 keeps the center time tap alone.
HunyuanImage3Vae::Conv HunyuanImage3Vae::conv_w(SafeTensors& st, const std::string& name, int64_t kt,
                                                int64_t pad_in) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || t->shape.size() != 5 || t->shape[2] != 3 || t->shape[3] != 3 || t->shape[4] != 3)
        throw std::runtime_error("image engine (HunyuanImage 3.0 VAE): no 3x3x3 " + name);
    Conv c;
    c.out = t->shape[0];
    const int64_t in_file = t->shape[1];
    c.in = std::max(in_file, pad_in);   // pad_in: the product reads this many channels, the extra ones zero
    c.kt = kt;
    const int64_t K = kt * 9 * c.in;
    if (dry_) {
        weight_bytes_ += (uint64_t) c.out * K * 2;
        c.b = vec(st, name + ".bias");
        return c;
    }
    const auto w = st.f32(name + ".weight");   // [O][I][3][3][3]
    std::vector<uint16_t> h((size_t) (c.out * K), 0);
    for (int64_t o = 0; o < c.out; ++o)
        for (int64_t i = 0; i < in_file; ++i)
            for (int64_t tap = 0; tap < kt * 9; ++tap) {
                const int64_t dt = kt == 3 ? tap / 9 : 1;   // kt 1 takes the center tap
                const int64_t dy = (tap % 9) / 3, dx = tap % 3;
                h[(size_t) (o * K + tap * c.in + i)] =
                    kernels::f16_from_f32(w[((size_t) (o * in_file + i) * 3 + dt) * 9 + dy * 3 + dx]);
            }
    uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (HunyuanImage 3.0 VAE)");
    gpu::copy(d, h.data(), h.size() * 2);
    c.w = d;
    weight_bytes_ += h.size() * 2;
    c.b = vec(st, name + ".bias");
    return c;
}

// A 1 x 1 x 1 convolution (the attention's projections) as the plain product W[O][I]
HunyuanImage3Vae::Conv HunyuanImage3Vae::point_w(SafeTensors& st, const std::string& name) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || t->shape.size() != 5 || t->shape[2] != 1)
        throw std::runtime_error("image engine (HunyuanImage 3.0 VAE): no 1x1x1 " + name);
    Conv c;
    c.out = t->shape[0];
    c.in = t->shape[1];
    c.kt = 0;
    if (dry_) {
        weight_bytes_ += (uint64_t) c.out * c.in * 2;
        c.b = vec(st, name + ".bias");
        return c;
    }
    const auto w = st.f32(name + ".weight");
    std::vector<uint16_t> h((size_t) (c.out * c.in));
    for (size_t j = 0; j < h.size(); ++j) h[j] = kernels::f16_from_f32(w[j]);
    uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (HunyuanImage 3.0 VAE)");
    gpu::copy(d, h.data(), h.size() * 2);
    c.w = d;
    weight_bytes_ += h.size() * 2;
    c.b = vec(st, name + ".bias");
    return c;
}

// A 1-D tensor (a GroupNorm's or a convolution's weight or bias) in FP32, the file's F16 widened
const float* HunyuanImage3Vae::vec(SafeTensors& st, const std::string& name) {
    if (dry_) {
        const SafeTensor* t = st.find(name);
        if (t == nullptr) throw std::runtime_error("image engine (HunyuanImage 3.0 VAE): no " + name);
        weight_bytes_ += (uint64_t) t->elements() * 4;
        return nullptr;
    }
    const auto v = st.f32(name);
    float* d = alloc<float>(allocs_, (int64_t) v.size(), "image engine (HunyuanImage 3.0 VAE)");
    gpu::copy(d, v.data(), v.size() * 4);
    weight_bytes_ += v.size() * 4;
    return d;
}

HunyuanImage3Vae::Res HunyuanImage3Vae::res_w(SafeTensors& st, const std::string& p, int64_t kt) {
    Res r;
    r.g1 = vec(st, p + "norm1.weight");
    r.b1 = vec(st, p + "norm1.bias");
    r.c1 = conv_w(st, p + "conv1", kt);
    r.g2 = vec(st, p + "norm2.weight");
    r.b2 = vec(st, p + "norm2.bias");
    r.c2 = conv_w(st, p + "conv2", kt);
    return r;
}

bool HunyuanImage3Vae::open(const std::string& path, void* stream, std::string& err, bool dry, bool encoder) {
    try {
        s_ = stream;
        dry_ = dry;
        SafeTensors st(path);
        // The grid's time depth per stage (VAE.md's decode table): T = 1 through conv_in, the mid blocks and the
        // up.0 level; 2 at up.1; 4 from up.2 to the head - the two temporal DCAE upsamples (ffactor_temporal 4)
        // sit after up.0 and up.1, the spatial ones (x2 each, ffactor_spatial 16) after up.2 and up.3.  So the
        // twelve convolutions of the T = 1 region keep their center time tap alone (the file's other 340 M
        // numbers are read by no image), and every tensor from up.1's blocks on is loaded whole.
        cin_ = conv_w(st, "decoder.conv_in", 1);
        mid0_ = res_w(st, "decoder.mid.block_1.", 1);
        mid2_ = res_w(st, "decoder.mid.block_2.", 1);
        ag_ = vec(st, "decoder.mid.attn_1.norm.weight");
        ab_ = vec(st, "decoder.mid.attn_1.norm.bias");
        aq_ = point_w(st, "decoder.mid.attn_1.q");
        ak_ = point_w(st, "decoder.mid.attn_1.k");
        av_ = point_w(st, "decoder.mid.attn_1.v");
        aproj_ = point_w(st, "decoder.mid.attn_1.proj_out");
        for (int64_t i = 0; i < 5; ++i) {
            const std::string p = "decoder.up." + std::to_string(i) + ".";
            const int64_t kt = i == 0 ? 1 : 3;   // the level's blocks and its upsample conv read the same T
            for (int64_t j = 0; j < 3; ++j)
                up_[i].r[j] = res_w(st, p + "block." + std::to_string(j) + ".", kt);
            if (i < 4) {
                up_[i].up = conv_w(st, p + "upsample.conv", kt);
                up_[i].temporal = i < 2 ? 1 : 0;
            }
        }
        gout_ = vec(st, "decoder.norm_out.weight");
        bout_ = vec(st, "decoder.norm_out.bias");
        cout_ = conv_w(st, "decoder.conv_out", 3);
        if (!dry_ && !gemm_.init(s_, 16, err)) return false;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return !encoder || open_encoder(path, err);
}

bool HunyuanImage3Vae::open_encoder(const std::string& path, std::string& err) {
    if (enc_open_) return true;
    try {
        SafeTensors st(path);
        const uint64_t before = weight_bytes_;
        // The encoder's time depth (VAE.md's section 4): four frames through conv_in, the down.0 - down.2 levels
        // and down.2's downsample (temporal: 4 -> 2), two through down.3 and its downsample (2 -> 1), one from
        // down.4 to the head.  conv_in reads the image's 3 channels padded to the unfold's 8.
        ecin_ = conv_w(st, "encoder.conv_in", 3, 8);
        for (int64_t i = 0; i < 5; ++i) {
            const std::string p = "encoder.down." + std::to_string(i) + ".";
            const int64_t kt = i <= 3 ? 3 : 1;
            for (int64_t j = 0; j < 2; ++j)
                down_[i].r[j] = res_w(st, p + "block." + std::to_string(j) + ".", kt);
            if (i < 4) {
                down_[i].down = conv_w(st, p + "downsample.conv", kt);
                down_[i].temporal = i >= 2 ? 1 : 0;
                down_[i].out = down_[i].down.out * 4 * (down_[i].temporal ? 2 : 1);
            }
        }
        emid0_ = res_w(st, "encoder.mid.block_1.", 1);
        emid2_ = res_w(st, "encoder.mid.block_2.", 1);
        eag_ = vec(st, "encoder.mid.attn_1.norm.weight");
        eab_ = vec(st, "encoder.mid.attn_1.norm.bias");
        eaq_ = point_w(st, "encoder.mid.attn_1.q");
        eak_ = point_w(st, "encoder.mid.attn_1.k");
        eav_ = point_w(st, "encoder.mid.attn_1.v");
        eaproj_ = point_w(st, "encoder.mid.attn_1.proj_out");
        egout_ = vec(st, "encoder.norm_out.weight");
        ebout_ = vec(st, "encoder.norm_out.bias");
        ecout_ = conv_w(st, "encoder.conv_out", 1);
        enc_open_ = true;
        enc_bytes_ = weight_bytes_ - before;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

void HunyuanImage3Vae::grow(int64_t elems32, int64_t elems16, int64_t col16, int64_t scores) {
    if (elems32 > cap32_) {
        for (float** p : {&t0_, &t1_, &tmp32_}) release(allocs_, *p);
        for (float** p : {&t0_, &t1_, &tmp32_}) *p = alloc<float>(allocs_, elems32, "image engine (HunyuanImage 3.0 VAE)");
        cap32_ = elems32;
    }
    if (elems16 > cap16_) {
        release(allocs_, a16_);
        a16_ = alloc<uint16_t>(allocs_, elems16, "image engine (HunyuanImage 3.0 VAE)");
        cap16_ = elems16;
    }
    if (col16 > col_cap_) {
        release(allocs_, col16_);
        col16_ = alloc<uint16_t>(allocs_, col16, "image engine (HunyuanImage 3.0 VAE)");
        col_cap_ = col16;
    }
    if (scores > score_cap_) {
        release(allocs_, scores_);
        scores_ = alloc<float>(allocs_, scores, "image engine (HunyuanImage 3.0 VAE)");
        score_cap_ = scores;
    }
    if (gn_ws_ == nullptr) gn_ws_ = alloc<float>(allocs_, kWsFloats, "image engine (HunyuanImage 3.0 VAE)");
    held_bytes_ = (uint64_t) cap32_ * 4 * 3 + (uint64_t) cap16_ * 2 + (uint64_t) col_cap_ * 2 +
                  (uint64_t) score_cap_ * 4 + (uint64_t) kWsFloats * 4;
}

// out [P][c.out] (FP32) = c(in16 [P][c.in]) + bias, the pixels the linear (t, y, x) index; kt 0 products
// straight, kt 1 / 3 through the unfold's bands
void HunyuanImage3Vae::conv(const Conv& c, const uint16_t* in16, int64_t T, int64_t H, int64_t W, float* out) {
    const int64_t frames = c.kt == 1 ? T : 1;
    const int64_t times = c.kt == 1 ? 1 : T;
    const int k = c.kt == 0 ? 1 : 3, kt = c.kt == 0 ? 1 : (int) c.kt, pad = k / 2;
    const int64_t P = times * H * W, K = (int64_t) kt * k * k * c.in;
    sycl::queue& q = core::Runtime::get().stream(s_);
    for (int64_t t = 0; t < frames; ++t) {
        const uint16_t* x = in16 + t * H * W * c.in;
        float* y = out + t * H * W * c.out;
        conv::Request r{x, H, W, c.in, c.w, c.out, k, 1, pad, pad, pad, pad};
        r.temporal = kt;
        r.bias = c.b;
        r.out = y;
        conv::drive(q, r, P, K, col_cap_, k == 1,
            [&](int64_t p0, int64_t n) { qi_im2col16(x, times, H, W, c.in, kt, k, 1, pad, H, W, p0, n, col16_, s_); },
            [&](int64_t p0, int64_t n, bool point) {
                gemm_.f16(point ? x : col16_, c.w, y + p0 * c.out, n, c.out, point ? c.in : K);
            },
            [&] { qi_bias_add(y, c.b, P, c.out, s_); });
    }
}

// one GroupNorm(32, eps 1e-6) over P pixels of C channels and the swish after it (when `silu`), FP16 out
void HunyuanImage3Vae::groupnorm(const float* x, const float* g, const float* b, int64_t P, int64_t C, bool silu,
                                 uint16_t* y16, const std::string& tag) {
    qi_group_norm16(x, P, C, g, b, kEps, silu, y16, gn_ws_, s_);
    if (!tag.empty()) dump_stage(0, tag + "_stats2", gn_ws_, 2 * kGroups);   // the means, then the inverse deviations
}

// GN -> swish -> conv1 -> GN -> swish -> conv2 -> + x.  The channels are equal in and out in every block of
// this VAE (the DCAE form puts the channel changes in the up/down samples), so there is no nin_shortcut - the
// file has none.  `tag` (the dump path): the block's two convolutions and the swish'd operands are dumped too.
void HunyuanImage3Vae::resblock(const Res& r, const float* x, int64_t T, int64_t H, int64_t W, int64_t C,
                                float* y, const std::string& tag) {
    const int64_t N = T * H * W;
    groupnorm(x, r.g1, r.b1, N, C, true, a16_, tag);
    if (!tag.empty()) dump_stage16("X" + tag + "_gn1", a16_, N * C);
    conv(r.c1, a16_, T, H, W, tmp32_);
    if (!tag.empty()) dump_stage(0, tag + "_c1", tmp32_, N * r.c1.out);
    groupnorm(tmp32_, r.g2, r.b2, N, C, true, a16_, tag + "_2");
    if (!tag.empty()) dump_stage16("X" + tag + "_gn2", a16_, N * C);
    conv(r.c2, a16_, T, H, W, y);
    if (!tag.empty()) dump_stage(0, tag + "_c2", y, N * r.c2.out);
    hunyuanimage3_add(y, x, N * C, s_);
}

// The mid attention over the whole grid (single-head, scale 1 / sqrt(C), no mask): GN -> q, k, v -> softmax
// (FP32) over the P positions -> V -> proj -> residual into x.  qkv and the attention's output take tmp32_ in
// turn, q16 / k16 / vt16 a16_ past the normed tokens (a16_ is read by the three products only until the split
// overwrites its head again), the probabilities the head of col16_ (no unfold band is live here).
void HunyuanImage3Vae::attention(float* x, int64_t P, int64_t C, const std::string& tag) {
    attention_w(ag_, ab_, aq_, ak_, av_, aproj_, x, P, C, tag);
}

void HunyuanImage3Vae::attention_w(const float* g, const float* b, const Conv& aq, const Conv& ak, const Conv& av,
                                   const Conv& aproj, float* x, int64_t P, int64_t C, const std::string& tag) {
    float* qkv = tmp32_;
    vae_mid::Attention mid{1, P, C, P};
    mid.x = x; mid.weight = g; mid.bias = b; mid.normalized = a16_; mid.norm_workspace = gn_ws_;
    mid.qkv = qkv;
    mid.q = a16_ + P * C; mid.k = a16_ + 2 * P * C; mid.vt = a16_ + 3 * P * C;
    mid.p = col16_; mid.scores = scores_; mid.out = tmp32_;
    vae_mid::run(mid, gemm_, s_,
        [&] {
            if (!tag.empty()) dump_stage(0, tag + "_stats2", gn_ws_, 2 * kGroups);
            gemm_.f16(a16_, aq.w, qkv, P, C, C, 3 * C);
            gemm_.f16(a16_, ak.w, qkv + C, P, C, C, 3 * C);
            gemm_.f16(a16_, av.w, qkv + 2 * C, P, C, C, 3 * C);
            qi_bias_add(qkv, aq.b, P, C, s_, nullptr, 3 * C);
            qi_bias_add(qkv + C, ak.b, P, C, s_, nullptr, 3 * C);
            qi_bias_add(qkv + 2 * C, av.b, P, C, s_, nullptr, 3 * C);
        },
        [&] {
            gemm_.f16(a16_, aproj.w, tmp32_, P, C, C);
            qi_bias_add(tmp32_, aproj.b, P, C, s_);
            hunyuanimage3_add(x, tmp32_, P * C, s_);
        });
}

// The DCAE downsample: the conv (its input's T decides the time taps), then the fold and the averaged shortcut in
// one gather over the conv's output - the reference's rearrange of both and the view(...).mean(2)
void HunyuanImage3Vae::downsample(const Down& d, const float* x, int64_t T, int64_t H, int64_t W, int64_t C,
                                  float* y) {
    const int64_t P = T * H * W;
    kernels::f32_to_f16_bulk(x, a16_, P * C, s_);
    conv(d.down, a16_, T, H, W, tmp32_);
    const int64_t r1 = d.temporal ? 2 : 1, factor = 4 * r1;
    dcae_fold_down(tmp32_, x, T, H, W, d.down.out, C, r1, factor * C / d.out, y, s_);
}

// The DCAE upsample: the conv (its input's T decides the time taps), then the split and the shortcut in one
// gather over the conv's band-sized output - nothing factor-times-wide is ever materialized whole
void HunyuanImage3Vae::upsample(const Up& u, const float* x, int64_t T, int64_t H, int64_t W, int64_t C,
                                float* y) {
    const int64_t P = T * H * W;
    // the conv reads the block's output FP16, as the reference's conv3d does
    kernels::f32_to_f16_bulk(x, a16_, P * C, s_);
    conv(u.up, a16_, T, H, W, tmp32_);
    const int64_t r1 = u.temporal ? 2 : 1;
    dcae_split(tmp32_, x, T, H, W, C, u.up.out / (4 * r1), r1, y, s_);
}

// A stage boundary for the probe's stage-by-stage look: the device buffer to a host file, name.bin (FP32, the
// pixels of the stage's grid in row order).  Only the dump path asks for it (index sorts the files; the
// resblock's internals pass 0 and carry the stage in the name).
void HunyuanImage3Vae::dump_stage(int index, const std::string& name, const float* p, int64_t n) {
    if (dump_.empty()) return;
    std::vector<float> host((size_t) n);
    gpu::stream_sync(s_);
    gpu::copy(host.data(), p, (size_t) n * sizeof(float));
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%02d_%s.bin", dump_.c_str(), index, name.c_str());
    std::FILE* fp = std::fopen(path, "wb");
    if (fp != nullptr) {
        std::fwrite(host.data(), sizeof(float), host.size(), fp);
        std::fclose(fp);
    }
}

// the same for an FP16 buffer (the convolutions' operands), widened on the way out
void HunyuanImage3Vae::dump_stage16(const std::string& name, const uint16_t* p, int64_t n) {
    if (dump_.empty()) return;
    std::vector<uint16_t> bits((size_t) n);
    gpu::stream_sync(s_);
    gpu::copy(bits.data(), p, (size_t) n * sizeof(uint16_t));
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s.bin", dump_.c_str(), name.c_str());
    std::FILE* fp = std::fopen(path, "wb");
    if (fp != nullptr) {
        std::vector<float> host((size_t) n);
        for (int64_t i = 0; i < n; ++i) host[(size_t) i] = kernels::f32_from_f16(bits[(size_t) i]);
        std::fwrite(host.data(), sizeof(float), host.size(), fp);
        std::fclose(fp);
    }
}

void HunyuanImage3Vae::decode(const float* z, int64_t h, int64_t w, float* rgb, const std::string& dump) {
    dump_ = dump;
    const int64_t HW = h * w;
    // the widest stage in elements: the head's grid T = 4, 16 h x 16 w of 128 channels = 131072 HW (the same
    // count at every DCAE upsample's conv output); the attention is the T = 1 grid: P = HW positions, C = 1024.
    const int64_t big32 = 131072 * HW, p_attn = HW;
    grow(big32, big32, std::max(conv::col_values(q_of(s_)), p_attn * p_attn), p_attn * p_attn);
    latent_tokens(z, (float) kScalingFactor, cin_.in, HW, a16_, t0_, s_);   // t0_ the FP32 tokens [HW][32]
    dump_stage(0, "tokens", t0_, HW * cin_.in);
    conv(cin_, a16_, 1, h, w, t1_);
    rep_add(t1_, t0_, HW, cin_.out, cin_.in, s_);                   // conv_in's repeat_interleave(32) shortcut
    float* x = t1_;
    dump_stage(1, "conv_in", x, HW * cin_.out);
    int64_t T = 1, H = h, W = w, C = cin_.out;
    resblock(mid0_, x, T, H, W, C, t0_, "mid0");
    x = t0_;
    dump_stage(2, "mid0", x, T * H * W * C);
    attention(x, T * H * W, C, "mid_attn");
    dump_stage(3, "mid_attn", x, T * H * W * C);
    resblock(mid2_, x, T, H, W, C, t1_, "mid2");
    x = t1_;
    dump_stage(4, "mid2", x, T * H * W * C);
    for (int64_t i = 0; i < 5; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            float* y = x == t0_ ? t1_ : t0_;
            const std::string tag = "up" + std::to_string(i) + "_b" + std::to_string(j);
            resblock(up_[i].r[j], x, T, H, W, C, y, tag);
            x = y;
            dump_stage(10 + (int) (i * 4 + j), tag, x, T * H * W * C);
        }
        if (up_[i].up.out > 0) {
            float* y = x == t0_ ? t1_ : t0_;
            upsample(up_[i], x, T, H, W, C, y);
            x = y;
            C = up_[i].up.out / static_cast<int64_t>(4 * (up_[i].temporal ? 2 : 1));
            if (up_[i].temporal) T *= 2;
            H *= 2;
            W *= 2;
            dump_stage(30 + (int) i, "up" + std::to_string(i) + "_up", x, T * H * W * C);
        }
    }
    groupnorm(x, gout_, bout_, T * H * W, C, true, a16_, "head");
    conv(cout_, a16_, T, H, W, tmp32_);
    dump_stage(50, "head4", tmp32_, T * H * W * cout_.out);
    last_frame(tmp32_, T, H, W, rgb, s_);
    gpu::stream_sync(s_);
    dump_.clear();
}

void HunyuanImage3Vae::encode(const float* rgb, int64_t H, int64_t W, float* z, const float* eps,
                              const std::string& dump) {
    if (!enc_open_) throw std::runtime_error("image engine (HunyuanImage 3.0 VAE): the encoder is not open");
    if (H % 16 != 0 || W % 16 != 0 || H <= 0 || W <= 0)
        throw std::invalid_argument("image engine (HunyuanImage 3.0 VAE): the image's sides must be multiples of 16");
    dump_ = dump;
    const int64_t HW = H * W, hw = HW / 256;
    // the widest stage: conv_in's four frames of 128 channels over the image = 512 HW (the decoder's widest is
    // the same count at that latent); the attention is the T = 1 grid of h x w positions
    const int64_t big32 = 512 * HW, p_attn = hw;
    grow(big32, big32, std::max(conv::col_values(q_of(s_)), p_attn * p_attn), p_attn * p_attn);
    int64_t T = 4;
    rgb_frames16(rgb, HW, T, ecin_.in, a16_, s_);
    conv(ecin_, a16_, T, H, W, t0_);
    float* x = t0_;
    int64_t C = ecin_.out, Hc = H, Wc = W;
    dump_stage(1, "enc_conv_in", x, T * Hc * Wc * C);
    for (int64_t i = 0; i < 5; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            float* y = x == t0_ ? t1_ : t0_;
            const std::string tag = "down" + std::to_string(i) + "_b" + std::to_string(j);
            resblock(down_[i].r[j], x, T, Hc, Wc, C, y, tag);
            x = y;
            dump_stage(10 + (int) (i * 3 + j), tag, x, T * Hc * Wc * C);
        }
        if (down_[i].out > 0) {
            float* y = x == t0_ ? t1_ : t0_;
            downsample(down_[i], x, T, Hc, Wc, C, y);
            x = y;
            C = down_[i].out;
            if (down_[i].temporal) T /= 2;
            Hc /= 2;
            Wc /= 2;
            dump_stage(30 + (int) i, "down" + std::to_string(i) + "_down", x, T * Hc * Wc * C);
        }
    }
    {
        float* y = x == t0_ ? t1_ : t0_;
        resblock(emid0_, x, T, Hc, Wc, C, y, "enc_mid0");
        x = y;
    }
    attention_w(eag_, eab_, eaq_, eak_, eav_, eaproj_, x, T * Hc * Wc, C, "enc_mid_attn");
    {
        float* y = x == t0_ ? t1_ : t0_;
        resblock(emid2_, x, T, Hc, Wc, C, y, "enc_mid2");
        x = y;
    }
    dump_stage(40, "enc_mid2", x, T * Hc * Wc * C);
    groupnorm(x, egout_, ebout_, T * Hc * Wc, C, true, a16_, "enc_head");
    conv(ecout_, a16_, T, Hc, Wc, tmp32_);
    dump_stage(50, "enc_conv_out", tmp32_, T * Hc * Wc * ecout_.out);
    encoder_head(tmp32_, x, eps, T * Hc * Wc, C, ecout_.out, C / ecout_.out, ecout_.out / 2, (float) kScalingFactor,
                 z, s_);
    gpu::stream_sync(s_);
    dump_.clear();
}

namespace {

// A tile of planes on the host: C planes of h x w.
struct HostTile {
    int64_t h = 0, w = 0;
    std::vector<float> v;
};

// The reference's blend_v / blend_h (autoencoder_kl_3d.py), in place on b: its first `extent` rows (columns) take
// a's last ones at weight 1 - y / extent, its own at y / extent
void blend_v(const HostTile& a, HostTile& b, int64_t C, int64_t extent) {
    extent = std::min({a.h, b.h, extent});
    for (int64_t y = 0; y < extent; ++y) {
        const double t = (double) y / (double) extent;
        const float wa = (float) (1.0 - t), wb = (float) t;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t x = 0; x < b.w; ++x) {
                float& d = b.v[(size_t) ((c * b.h + y) * b.w + x)];
                d = a.v[(size_t) ((c * a.h + a.h - extent + y) * a.w + x)] * wa + d * wb;
            }
    }
}

void blend_h(const HostTile& a, HostTile& b, int64_t C, int64_t extent) {
    extent = std::min({a.w, b.w, extent});
    for (int64_t x = 0; x < extent; ++x) {
        const double t = (double) x / (double) extent;
        const float wa = (float) (1.0 - t), wb = (float) t;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t y = 0; y < b.h; ++y) {
                float& d = b.v[(size_t) ((c * b.h + y) * b.w + x)];
                d = a.v[(size_t) ((c * a.h + y) * a.w + a.w - extent + x)] * wa + d * wb;
            }
    }
}

// The reference's stitch: each tile blended in place into the one above and the one to its left over `extent`
// (in this order, so a blended tile is the next one's neighbour), cut to `limit` and set at its place in out [C][OH][OW]
void stitch(std::vector<std::vector<HostTile>>& rows, int64_t C, int64_t OH, int64_t OW, int64_t extent, int64_t limit,
            std::vector<float>& out) {
    out.assign((size_t) (C * OH * OW), 0.f);
    int64_t oy = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        int64_t ox = 0, rh = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            HostTile& t = rows[i][j];
            if (i > 0) blend_v(rows[i - 1][j], t, C, extent);
            if (j > 0) blend_h(rows[i][j - 1], t, C, extent);
            const int64_t ch = std::min(t.h, limit), cw = std::min(t.w, limit);
            for (int64_t c = 0; c < C; ++c)
                for (int64_t y = 0; y < ch; ++y)
                    std::copy_n(t.v.begin() + (c * t.h + y) * t.w, cw, out.begin() + ((c * OH + oy + y) * OW + ox));
            ox += cw;
            rh = ch;
        }
        oy += rh;
    }
}

// The input's tiles of `tile` every `step` (C planes of H x W), each through `run` into its output tile
std::vector<std::vector<HostTile>> tiles(const std::vector<float>& in, int64_t C, int64_t H, int64_t W, int64_t tile,
                                         int64_t step, const std::function<void(const HostTile&, HostTile&)>& run) {
    std::vector<std::vector<HostTile>> rows;
    for (int64_t i = 0; i < H; i += step) {
        std::vector<HostTile> row;
        for (int64_t j = 0; j < W; j += step) {
            HostTile t;
            t.h = std::min(tile, H - i);
            t.w = std::min(tile, W - j);
            t.v.resize((size_t) (C * t.h * t.w));
            for (int64_t c = 0; c < C; ++c)
                for (int64_t y = 0; y < t.h; ++y)
                    std::copy_n(in.begin() + ((c * H + i + y) * W + j), t.w, t.v.begin() + (c * t.h + y) * t.w);
            HostTile o;
            run(t, o);
            row.push_back(std::move(o));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

void HunyuanImage3Vae::decode_tiled(const float* z, int64_t h, int64_t w, float* rgb) {
    const int64_t zc = cin_.in;
    std::vector<float> zh((size_t) (zc * h * w));
    gpu::copy(zh.data(), z, zh.size() * sizeof(float));
    const int64_t step = (int64_t) ((double) kTileTokens * (1.0 - kTileOverlap));         // 21
    const int64_t extent = (int64_t) ((double) kTilePixels * kTileOverlap);              // 48
    float *dz = nullptr, *drgb = nullptr;
    if (!gpu::alloc_device(&dz, (size_t) (zc * kTileTokens * kTileTokens) * sizeof(float)) ||
        !gpu::alloc_device(&drgb, (size_t) (3 * kTilePixels * kTilePixels) * sizeof(float))) {
        if (dz != nullptr) gpu::free(dz);
        throw std::runtime_error(std::string("image engine (HunyuanImage 3.0 VAE): a tile's buffers: ") +
                                 gpu::last_error());
    }
    std::vector<float> out;
    try {
        auto rows = tiles(zh, zc, h, w, kTileTokens, step, [&](const HostTile& t, HostTile& o) {
            gpu::copy(dz, t.v.data(), t.v.size() * sizeof(float));
            decode(dz, t.h, t.w, drgb);
            o.h = t.h * 16;
            o.w = t.w * 16;
            o.v.resize((size_t) (3 * o.h * o.w));
            gpu::copy(o.v.data(), drgb, o.v.size() * sizeof(float));
        });
        stitch(rows, 3, h * 16, w * 16, extent, kTilePixels - extent, out);
    } catch (...) {
        gpu::free(dz);
        gpu::free(drgb);
        throw;
    }
    gpu::free(dz);
    gpu::free(drgb);
    gpu::copy(rgb, out.data(), out.size() * sizeof(float));
}

uint64_t HunyuanImage3Vae::tiled_scratch_bytes() const {
    return scratch_bytes(kTileTokens, kTileTokens) + (uint64_t) (3 * kTilePixels * kTilePixels) * 4 +
           (uint64_t) (cin_.in * kTileTokens * kTileTokens) * 4;
}

uint64_t HunyuanImage3Vae::encode_scratch_bytes(int64_t H, int64_t W) const {
    return scratch_bytes(H / 16, W / 16);   // the widest stage and the attention's grid are the decode's at that latent
}

uint64_t HunyuanImage3Vae::scratch_bytes(int64_t h, int64_t w) const {
    const int64_t HW = h * w, big32 = 131072 * HW, col = std::max(conv::col_values(q_of(s_)), HW * HW);
    return (uint64_t) big32 * 4 * 3 + (uint64_t) big32 * 2 + (uint64_t) col * 2 + (uint64_t) HW * HW * 4 +
           (uint64_t) kWsFloats * 4;
}

void HunyuanImage3Vae::drop_scratch() {
    gpu::stream_sync(s_);
    for (float** p : {&t0_, &t1_, &tmp32_, &scores_, &gn_ws_}) release(allocs_, *p);
    for (uint16_t** p : {&a16_, &col16_}) release(allocs_, (void*) *p);
    t0_ = t1_ = tmp32_ = scores_ = gn_ws_ = nullptr;
    a16_ = col16_ = nullptr;
    cap32_ = cap16_ = col_cap_ = score_cap_ = 0;
    held_bytes_ = 0;
}

}  // namespace strata::image
