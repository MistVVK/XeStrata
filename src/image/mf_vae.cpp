// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_vae.cpp - see include/strata/image/mf_vae.hpp.

#include "strata/image/mf_vae.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/image/gelu.hpp"
#include "strata/kernels/bf16_to_f16.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/elementwise.hpp"

#include "dit_rows.hpp"
#include "tile_sides.hpp"
#include "img_out.hpp"
#include "qi_conv.hpp"
#include "vae_attention.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

// The kernels' helpers in a namespace of this file's own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1), so a helper of the same
// name and signature in another image file (qi_vae.cpp's to_rgba) gave its kernel to both.
namespace mf_vae_detail {

using kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using kernels::f16_gpu::f32_from_f16;

constexpr int64_t kPatch = 16, kPix = kPatch * kPatch;   // a latent position's pixels
constexpr int64_t kWin = 32, kWinTok = kWin * kWin;       // the CoD decoder's attention windows
constexpr int64_t kGroups = 32;                           // its group norms
constexpr int64_t kF2Overlap = 4, kF2TileMin = 16;        // the Flux2 VAE's tiles: tokens shared, the least side
// a video's tiles' tokens shared, blended over them all: 128 pixels (4 took a tile of 160 pixels
// 2.75% off the whole frame's decode against 1.7% with 8, 2026-11-08)
constexpr int64_t kVOverlap = 8;
constexpr int64_t kX = 32;                                // the per-pixel MLP's width
constexpr int kResBlocks = 3;
constexpr float kEps = 1e-6f;

// the per-pixel MLP's weights in one FP32 buffer (offsets in floats)
constexpr int64_t kEmbW = 0, kEmbB = kEmbW + kX * 99, kInW = kEmbB + kX, kInB = kInW + kX * kX, kRes0 = kInB + kX;
constexpr int64_t kAdaW = 0, kAdaB = kAdaW + 3 * kX * kX, kLnW = kAdaB + 3 * kX, kLnB = kLnW + kX, kM1W = kLnB + kX,
                  kM1B = kM1W + kX * kX, kM2W = kM1B + kX, kM2B = kM2W + kX * kX, kResSize = kM2B + kX;
constexpr int64_t kFnW = kRes0 + kResBlocks * kResSize, kFlW = kFnW + kX, kFlB = kFlW + 3 * kX, kPixW = kFlB + 3;

struct TempBuffers : std::vector<void*> {
    explicit TempBuffers(void* stream) : stream_(stream) {}
    ~TempBuffers() {
        gpu::stream_sync(stream_);
        for (void* p : *this) gpu::free(p);
    }
    void* stream_;
};

template<class T> T* alloc_pinned(std::vector<void*>& owned, int64_t n) {
    void* p = gpu::alloc_host((size_t) std::max<int64_t>(n, 1) * sizeof(T));
    if (p == nullptr)
        throw std::runtime_error(std::string("image engine (Mage-VAE): pinned host memory: ") + gpu::last_error());
    owned.push_back(p);
    return static_cast<T*>(p);
}

float silu(float x) { return x / (1.f + std::exp(-x)); }

// y16[p] = LayerNorm(x[p]) over the channels (no affine) * (1 + scale) + shift: qi_ln_mod16's
void ln_mod16(const float* x, const float* shift, const float* scale, int64_t HW, int64_t C, uint16_t* y16, void* s) {
    qi_ln_mod16(x, scale, 0, HW, C, y16, s, shift);
}

// out = GELU(depthwise 3 x 3 convolution of t (zero outside) + bias), the exact (erf) GELU
void dw3x3_gelu(const float* t, const float* w, const float* b, int64_t H, int64_t W, int64_t C, float* out, void* s) {
    Depthwise d; d.x = t; d.w = w; d.bias = b; d.out = out;
    d.h = d.oh = H; d.wdim = d.ow = W; d.c = C; d.act = ImageAct::gelu;
    image_depthwise(d, s);
}

// ca[o] = sigmoid(b[o] + w[o] . mean of x's channels over the pixels): the channel attention's gate
void channel_gate(const float* x, int64_t HW, int64_t C, const float* w, const float* b, float* mean, float* ca,
                  void* s) {
    sycl::queue& q = q_of(s);
    // the means: a work-group 32 channels, PL lanes over the pixels for each (a work-item a channel walking every pixel
    // ran a few hundred items at once)
    constexpr int64_t CB = 32, PL = 16;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (CB * PL)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((C + CB - 1) / CB * CB * PL)),
                                         sycl::range<1>((size_t) (CB * PL))),
                       [=](sycl::nd_item<1> it) {
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t pl = lane / CB, cb = lane - pl * CB, c = (int64_t) it.get_group_linear_id() * CB + cb;
            float a = 0.f;
            if (c < C)
                for (int64_t p = pl; p < HW; p += PL) a += x[p * C + c];
            part[lane] = a;
            sycl::group_barrier(it.get_group());
            if (pl == 0 && c < C) {
                float t = 0.f;
                for (int64_t j = 0; j < PL; ++j) t += part[j * CB + cb];
                mean[c] = t / (float) HW;
            }
        });
    });
    q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const auto o = static_cast<int64_t>(id[0]);
        float a = b[o];
        for (int64_t i = 0; i < C; ++i) a += w[o * C + i] * mean[i];
        ca[o] = 1.f / (1.f + sycl::exp(-a));
    });
}

// y16 = x * ca (per channel)
void scale16(const float* x, const float* ca, int64_t HW, int64_t C, uint16_t* y16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[i] = f16_from_f32(x[i] * ca[i % C]);
    });
}

// x += gate * u (per channel): dit_rows' gate_add
void gate_add(float* x, const float* u, const float* gate, int64_t HW, int64_t C, void* s) {
    DitOut h;
    h.f = u;
    h.C = C;
    dit_rows::gate_add(x, h, gate, HW, C, s);
}

// The per-pixel MLP of patches p0 .. p0 + np: pixel j of patch p takes the 32 values yx[p][c * 256 + j] (with the
// noise's three zeros and the patch's DCT features) through the NerfEmbedder, the input projection, three residual
// blocks modulated by ce[p][j * 32 ..] and the final norm and projection, into rgb [3][H][W].
void pixel_mlp(const float* yx, const float* ce, const float* pw, const float* dct, int64_t p0, int64_t np, int64_t w,
               int64_t H, int64_t W, float* rgb, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (np * kPix)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t pl = i / kPix, j = i - pl * kPix, p = p0 + pl;
        const float* y = yx + pl * kX * kPix;
        const float* d = dct + j * 64;
        float h[kX], e[kX];
        for (int64_t o = 0; o < kX; ++o) {   // the embedder: [3 zeros, 32 condition values, 64 DCT features]
            const float* wr = pw + kEmbW + o * 99;
            float a = pw[kEmbB + o];
            for (int64_t c = 0; c < kX; ++c) a += wr[3 + c] * y[c * kPix + j];
            for (int64_t c = 0; c < 64; ++c) a += wr[35 + c] * d[c];
            e[o] = a;
        }
        for (int64_t o = 0; o < kX; ++o) {
            float a = pw[kInB + o];
            for (int64_t c = 0; c < kX; ++c) a += pw[kInW + o * kX + c] * e[c];
            h[o] = a;
        }
        const float* cp = ce + pl * kPix * kX + j * kX;
        float sc[kX];
        for (int64_t c = 0; c < kX; ++c) sc[c] = cp[c] / (1.f + sycl::exp(-cp[c]));
        for (int r = 0; r < kResBlocks; ++r) {
            const float* rb = pw + kRes0 + r * kResSize;
            float m = 0.f, v = 0.f;
            for (int64_t c = 0; c < kX; ++c) m += h[c];
            m /= (float) kX;
            for (int64_t c = 0; c < kX; ++c) v += (h[c] - m) * (h[c] - m);
            const float inv = 1.f / sycl::sqrt(v / (float) kX + kEps);
            float hn[kX], t[kX];
            for (int64_t o = 0; o < kX; ++o) {   // adaLN: shift, scale; then the gate below
                float sh = rb[kAdaB + o], sl = rb[kAdaB + kX + o];
                for (int64_t c = 0; c < kX; ++c) {
                    sh += rb[kAdaW + o * kX + c] * sc[c];
                    sl += rb[kAdaW + (kX + o) * kX + c] * sc[c];
                }
                hn[o] = ((h[o] - m) * inv * rb[kLnW + o] + rb[kLnB + o]) * (1.f + sl) + sh;
            }
            for (int64_t o = 0; o < kX; ++o) {
                float a = rb[kM1B + o];
                for (int64_t c = 0; c < kX; ++c) a += rb[kM1W + o * kX + c] * hn[c];
                t[o] = a / (1.f + sycl::exp(-a));
            }
            for (int64_t o = 0; o < kX; ++o) {
                float a = rb[kM2B + o], g = rb[kAdaB + 2 * kX + o];
                for (int64_t c = 0; c < kX; ++c) {
                    a += rb[kM2W + o * kX + c] * t[c];
                    g += rb[kAdaW + (2 * kX + o) * kX + c] * sc[c];
                }
                e[o] = g * a;
            }
            for (int64_t o = 0; o < kX; ++o) h[o] += e[o];
        }
        float ss = 0.f;
        for (int64_t c = 0; c < kX; ++c) ss += h[c] * h[c];
        const float inv = 1.f / sycl::sqrt(ss / (float) kX + kEps);
        const int64_t py = p / w, px = p % w, Y = py * kPatch + j / kPatch, X = px * kPatch + j % kPatch;
        for (int64_t o = 0; o < 3; ++o) {
            float a = pw[kFlB + o];
            for (int64_t c = 0; c < kX; ++c) a += pw[kFlW + o * kX + c] * h[c] * inv * pw[kFnW + c];
            rgb[(o * H + Y) * W + X] = a;
        }
    });
}

// y16[p][(c * 16 + ky) * 16 + kx] = rgb[c][py * 16 + ky][px * 16 + kx]: the encoder's 16 x 16 patches as rows, in its
// patch convolution's (channel, row, column) order
void patchify16(const float* rgb, int64_t h, int64_t w, uint16_t* y16, void* s) {
    const int64_t H = h * kPatch, W = w * kPatch;
    q_of(s).parallel_for(sycl::range<1>((size_t) (h * w * 3 * kPix)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / (3 * kPix), r = i - p * 3 * kPix, c = r / kPix, k = r - c * kPix;
        const int64_t y = (p / w) * kPatch + k / kPatch, x = (p % w) * kPatch + k % kPatch;
        y16[i] = f16_from_f32(rgb[(c * H + y) * W + x]);
    });
}

// z[p][c] = mean + exp(logvar / 2) eps (eps null: the mean), from the moments [p][2 Z] (logvar clamped to -20 .. 10)
void posterior(const float* mom, const float* eps, int64_t n, int64_t Z, float* z, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (n * Z)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / Z, c = i - p * Z;
        float v = mom[p * 2 * Z + c];
        if (eps != nullptr) v += sycl::exp(0.5f * sycl::clamp(mom[p * 2 * Z + Z + c], -20.f, 10.f)) * eps[i];
        z[i] = v;
    });
}

// the Flux2 VAE's input: out16[Y][X][c] = z[(Y / 2) w + X / 2][4 c + 2 (Y % 2) + X % 2] std + mean (the same index):
// the batch norm undone (none without mean: SeedVR2's) and each token's 4 Lc values spread over 2 x 2 positions of Lc
// channels
void f2_unpatch16(const float* z, const float* mean, const float* sd, int64_t h, int64_t w, int64_t Lc,
                  uint16_t* out16, void* s) {
    const int64_t W = 2 * w;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * h * w * Lc)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / Lc, c = i - p * Lc, Y = p / W, X = p - Y * W;
        const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
        const float v = z[((Y / 2) * w + X / 2) * 4 * Lc + k];
        out16[i] = f16_from_f32(mean != nullptr ? v * sd[k] + mean[k] : v);
    });
}

// SeedVR2's upsampling: out16[2 Y + x][2 X + y][c] = in[Y][X][(2 x + y) C + c], in [H][W][4 C] (its 1 x 1
// convolution's output, the reference's "(x y z c)" channels with the time's z = 0 kept)
void f2_shuffle16(const float* in, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* s) {
    const int64_t W2 = 2 * W;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C, Y2 = p / W2, X2 = p - Y2 * W2;
        out16[i] = f16_from_f32(in[((Y2 / 2) * W + X2 / 2) * 4 * C + (2 * (Y2 % 2) + X2 % 2) * C + c]);
    });
}

// SeedVR2's video upsampling: out16[z][2 Y + x][2 X + y][c] = in[Y][X][((2 x + y) Z + z) C + c], in [H][W][4 Z C]
// (its 1 x 1 convolution's output, the reference's "(x y z c)" channels): Z frames of 2 H x 2 W
void f2_shuffle_t16(const float* in, int64_t H, int64_t W, int64_t C, int64_t Z, uint16_t* out16, void* s) {
    const int64_t W2 = 2 * W, F = 4 * H * W * C;
    q_of(s).parallel_for(sycl::range<1>((size_t) (Z * F)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t z = i / F, r = i - z * F, p = r / C, c = r - p * C, Y2 = p / W2, X2 = p - Y2 * W2;
        out16[i] = f16_from_f32(in[((Y2 / 2) * W + X2 / 2) * 4 * Z * C + ((2 * (Y2 % 2) + X2 % 2) * Z + z) * C + c]);
    });
}

// The video form's 16-bit activations (FP16 bits, BF16's with bf) and FP32: o = 16 bits of t + bias[c] (+ o with
// `add`), t [n / O][O]
void act_store(const float* t, const float* bias, int64_t n, int64_t O, bool bf, bool add, uint16_t* o, void* s) {
    ProductStore p;
    p.in = t; p.bias = bias; p.rows = n / O; p.cols = O; p.bf = bf; p.add_zero = false;
    p.residual = add ? o : nullptr; p.out16 = o;
    product_store(p, s);
}

// o = FP16 of y + bias[c] (+ o with `add`), y [n / O][O] FP16 (o itself where nothing is added)
void act_bias16(const uint16_t* y, const float* bias, int64_t n, int64_t O, bool add, uint16_t* o, void* s) {
    ProductStore p;
    p.in16 = y; p.bias = bias; p.rows = n / O; p.cols = O;
    p.residual = add ? o : nullptr; p.out16 = o;
    product_store(p, s);
}

// a video convolution's taps [O][kt][per] as [kt][O][per]
void taps_apart(const uint16_t* taps, int64_t O, int64_t kt, int64_t per, uint16_t* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (O * kt * per)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t z = i / (O * per), r = i - z * O * per, o = r / per, j = r - o * per;
        out[i] = taps[(o * kt + z) * per + j];
    });
}

// y = FP16 of BF16 x (a convolution's input)
void bf16_to_f16(const uint16_t* x, int64_t n, uint16_t* y, void* s) {
    kernels::bf16_to_f16<kernels::F16Range::native>(q_of(s), x, y, n);
}

// the Flux2 VAE encoder's input: out16[p][c] = rgb[c][p] for the 3 channels, 0 for the other 5 (its first
// convolution's input widened to 8 channels)
void f2_rgb8_16(const float* rgb, int64_t HW, uint16_t* out16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (8 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i - p * 8;
        out16[i] = f16_from_f32(c < 3 ? rgb[c * HW + p] : 0.f);
    });
}

// f2_unpatch16's inverse: z[(Y / 2) w + X / 2][4 c + 2 (Y % 2) + X % 2] = (m[Y][X][c] - mean) / std (the same
// index; as it is without mean) over the posterior's mean, the first Lc of the moments m [2 h][2 w][2 Lc]
void f2_patch(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, int64_t Lc, float* z,
              void* s) {
    const int64_t W = 2 * w;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * h * w * Lc)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / Lc, c = i - p * Lc, Y = p / W, X = p - Y * W;
        const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
        const float v = m[p * 2 * Lc + c];
        z[((Y / 2) * w + X / 2) * 4 * Lc + k] = mean != nullptr ? (v - mean[k]) / sd[k] : v;
    });
}

// rgb [3][HW] from x [HW][3]
void hwc3_to_chw(const float* x, int64_t HW, float* rgb, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (3 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        rgb[i] = x[p * 3 + c];
    });
}

}  // namespace mf_vae_detail

using namespace mf_vae_detail;

MfVae::MfVae() = default;

MfVae::~MfVae() {
    for (void* p : allocs_) gpu::free(p);
}

const float* MfVae::upload(const std::vector<float>& v) {
    weight_bytes_ += v.size() * 4;
    if (dry_) return nullptr;
    float* d = alloc<float>(allocs_, (int64_t) v.size(), "image engine (Mage-VAE)");
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

const float* MfVae::vec(SafeTensors& st, const std::string& name) {
    if (dry_) {
        const SafeTensor* t = st.find(name);
        if (t == nullptr) throw std::runtime_error("image engine (Mage-VAE): no " + name);
        weight_bytes_ += (uint64_t) t->elements() * 4;
        return nullptr;
    }
    return upload(st.f32(name));
}

// a convolution's (or linear layer's) weights as FP16 [out][k][k][in], input columns col0 .. col0 + cols of them; a
// video convolution's [out][in][kt][k][k] (SeedVR2's VAE) as the 2D one it is on an image's one frame, its time taps
// summed (the frame repeated before it, as its causal padding has it); `rows` (null: all) the output rows taken, in
// that order
MfVae::Conv MfVae::conv_w(SafeTensors& st, const std::string& name, bool bias, int64_t col0, int64_t cols,
                          const std::vector<int64_t>* rows) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || (t->shape.size() != 5 && t->shape.size() != 4 && t->shape.size() != 2))
        throw std::runtime_error("image engine (Mage-VAE): no " + name);
    Conv c;
    const int64_t all = t->shape[0], in = t->shape[1], kt = t->shape.size() == 5 ? t->shape[2] : 1;
    c.out = rows != nullptr ? (int64_t) rows->size() : all;
    c.in = cols < 0 ? in - col0 : cols;
    c.k = t->shape.size() >= 4 ? t->shape[t->shape.size() - 1] : 1;
    if (c.in % 8 != 0) throw std::runtime_error("image engine (Mage-VAE): " + name + ": input channels not 8s");
    const bool taps = video_ && kt > 1;   // the video form: each time tap, not their sum
    if (taps) c.kt = kt;
    weight_bytes_ += (uint64_t) ((taps ? kt : 1) * c.out * c.k * c.k * c.in) * 2;
    if (!dry_) {
        const auto w = st.f32(name + ".weight");   // [out][in][kt][k][k]
        std::vector<uint16_t> h((size_t) (taps ? 0 : c.out * c.k * c.k * c.in));
        for (int64_t o = 0; o < (taps ? 0 : c.out); ++o) {
            const int64_t src = rows != nullptr ? (*rows)[(size_t) o] : o;
            for (int64_t i = 0; i < c.in; ++i)
                for (int64_t dy = 0; dy < c.k; ++dy)
                    for (int64_t dx = 0; dx < c.k; ++dx) {
                        float v = 0.f;
                        for (int64_t z = 0; z < kt; ++z)
                            v += w[(size_t) ((((src * in + col0 + i) * kt + z) * c.k + dy) * c.k + dx)];
                        h[(size_t) (((o * c.k + dy) * c.k + dx) * c.in + i)] = kernels::f16_from_f32(v);
                    }
        }
        if (!taps) {
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (Mage-VAE)");
            gpu::copy(d, h.data(), h.size() * 2);
            c.w = d;
        }
        if (taps) {
            std::vector<uint16_t> tw((size_t) (kt * c.out * c.k * c.k * c.in));
            for (int64_t o = 0; o < c.out; ++o) {
                const int64_t src = rows != nullptr ? (*rows)[(size_t) o] : o;
                for (int64_t z = 0; z < kt; ++z)
                    for (int64_t i = 0; i < c.in; ++i)
                        for (int64_t dy = 0; dy < c.k; ++dy)
                            for (int64_t dx = 0; dx < c.k; ++dx)
                                tw[(size_t) ((((o * kt + z) * c.k + dy) * c.k + dx) * c.in + i)] =
                                    kernels::f16_from_f32(
                                        w[(size_t) ((((src * in + col0 + i) * kt + z) * c.k + dy) * c.k + dx)]);
            }
            uint16_t* td = alloc<uint16_t>(allocs_, (int64_t) tw.size(), "image engine (Mage-VAE)");
            gpu::copy(td, tw.data(), tw.size() * 2);
            c.taps = td;
        }
    }
    if (bias) {
        if (rows == nullptr) {
            c.b = vec(st, name + ".bias");
        } else {
            const auto b = st.f32(name + ".bias");
            std::vector<float> v;
            for (const int64_t r : *rows) v.push_back(b[(size_t) r]);
            c.b = upload(v);
        }
    }
    return c;
}

// a self-attention block's weights (its norm, q, k, v and output projection, by name), q, k and v as one 1 x 1
// convolution
MfVae::Attn MfVae::attn_w(SafeTensors& st, const std::string& norm, const std::string (&qkv)[3],
                          const std::string& proj, int64_t C) {
    Attn x;
    x.nw = vec(st, norm + ".weight");
    x.nb = vec(st, norm + ".bias");
    if (!dry_) {   // q, k and v as one 1 x 1 convolution of 3 C outputs
        std::vector<uint16_t> h((size_t) (3 * C * C));
        std::vector<float> b((size_t) (3 * C));
        for (int j = 0; j < 3; ++j) {
            const auto w = st.f32(qkv[j] + ".weight");
            const auto bb = st.f32(qkv[j] + ".bias");
            if ((int64_t) w.size() != C * C) throw std::runtime_error("image engine (VAE): " + qkv[j]);
            for (int64_t k = 0; k < C * C; ++k) h[(size_t) (j * C * C + k)] = kernels::f16_from_f32(w[(size_t) k]);
            std::copy(bb.begin(), bb.end(), b.begin() + j * C);
        }
        uint16_t* dw = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (Mage-VAE)");
        gpu::copy(dw, h.data(), h.size() * 2);
        x.qkv.w = dw;
        x.qkv.b = upload(b);
    } else {
        weight_bytes_ += (uint64_t) (3 * C * C) * 2 + (uint64_t) (3 * C) * 4;
    }
    x.qkv.out = 3 * C;
    x.qkv.in = C;
    x.proj = conv_w(st, proj);
    return x;
}

bool MfVae::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        path_ = path;
        SafeTensors st(path);
        if ((st.find("bn.running_mean") != nullptr && st.find("decoder.conv_in.weight") != nullptr) ||
            st.find("decoder.up_blocks.2.upsamplers.0.upscale_conv.weight") != nullptr) {   // Flux2's, SeedVR2's
            open_flux2(st);
            if (!dry_ && !gemm_.init(s_, 16, err)) return false;
            return true;
        }
        const std::string d = "pipeline.y_embedder.decoder.";
        cin_ = conv_w(st, d + "conv_in");
        C_ = cin_.out;
        if (C_ % kGroups != 0) throw std::runtime_error("image engine (Mage-VAE): channels not a multiple of 32");
        for (int i = 0, r = 0, a = 0; i < 5; ++i) {   // ResBlock, attention, ResBlock, attention, ResBlock
            const std::string p = d + "block." + std::to_string(i) + ".";
            if (i % 2 == 0) {
                Res& x = res_[r++];
                x.n1w = vec(st, p + "norm1.weight");
                x.n1b = vec(st, p + "norm1.bias");
                x.c1 = conv_w(st, p + "conv1");
                x.n2w = vec(st, p + "norm2.weight");
                x.n2b = vec(st, p + "norm2.bias");
                x.c2 = conv_w(st, p + "conv2");
            } else {
                const std::string qkv[3] = {p + "q", p + "k", p + "v"};
                attn_[a++] = attn_w(st, p + "norm", qkv, p + "proj_out", C_);
            }
        }
        gnw_ = vec(st, d + "norm_out.weight");
        gnb_ = vec(st, d + "norm_out.bias");
        cout_ = conv_w(st, d + "conv_out");
        // the denoiser at t = 0: its timestep embedding, then each block's constant modulation
        auto host = [&](const std::string& n) { return st.f32(n); };
        auto linear = [](const std::vector<float>& w, const std::vector<float>& b, const std::vector<float>& x) {
            const size_t n = b.size(), k = x.size();
            std::vector<float> y(n);
            for (size_t o = 0; o < n; ++o) {
                double a = b[o];
                for (size_t i = 0; i < k; ++i) a += (double) w[o * k + i] * x[i];
                y[o] = (float) a;
            }
            return y;
        };
        std::vector<float> emb(256, 0.f);
        for (int i = 0; i < 128; ++i) emb[(size_t) i] = 1.f;   // cos(0) for the 128 frequencies, then sin(0)
        auto c = linear(host("pipeline.t_embedder.mlp.0.weight"), host("pipeline.t_embedder.mlp.0.bias"), emb);
        for (float& v : c) v = silu(v);
        c = linear(host("pipeline.t_embedder.mlp.2.weight"), host("pipeline.t_embedder.mlp.2.bias"), c);
        std::vector<float> sc(c.size());
        for (size_t i = 0; i < c.size(); ++i) sc[i] = silu(c[i]);
        s_in_ = conv_w(st, "pipeline.s_embedder.proj2", true, 128, C_);
        for (int i = 0;; ++i) {
            const std::string p = "pipeline.blocks." + std::to_string(i) + ".";
            if (st.find(p + "conv1.weight") == nullptr) break;
            blocks_.push_back(dico_w(st, p, linear(host(p + "adaLN_modulation.1.weight"),
                                                   host(p + "adaLN_modulation.1.bias"), sc)));
            F_ = blocks_.back().c4.out;
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (Mage-VAE): no denoiser blocks");
        yx_ = conv_w(st, "pipeline.y_embedder_x");
        ce_ = conv_w(st, "pipeline.dec_net.cond_embed");
        if (yx_.out != kX * kPix || ce_.out != kX * kPix)
            throw std::runtime_error("image engine (Mage-VAE): the per-pixel MLP is not 32 wide on 16 x 16 patches");
        // the per-pixel MLP's weights, in kPixW's layout
        std::vector<float> pw((size_t) kPixW, 0.f);
        auto put = [&](int64_t off, const std::string& n, size_t want) {
            const auto v = host(n);
            if (v.size() != want) throw std::runtime_error("image engine (Mage-VAE): " + n + ": unexpected size");
            std::copy(v.begin(), v.end(), pw.begin() + off);
        };
        put(kEmbW, "pipeline.x_embedder.embedder.0.weight", kX * 99);
        put(kEmbB, "pipeline.x_embedder.embedder.0.bias", kX);
        put(kInW, "pipeline.dec_net.input_proj.weight", kX * kX);
        put(kInB, "pipeline.dec_net.input_proj.bias", kX);
        for (int r = 0; r < kResBlocks; ++r) {
            const std::string p = "pipeline.dec_net.res_blocks." + std::to_string(r) + ".";
            const int64_t o = kRes0 + r * kResSize;
            put(o + kAdaW, p + "adaLN_modulation.1.weight", 3 * kX * kX);
            put(o + kAdaB, p + "adaLN_modulation.1.bias", 3 * kX);
            put(o + kLnW, p + "in_ln.weight", kX);
            put(o + kLnB, p + "in_ln.bias", kX);
            put(o + kM1W, p + "mlp.0.weight", kX * kX);
            put(o + kM1B, p + "mlp.0.bias", kX);
            put(o + kM2W, p + "mlp.2.weight", kX * kX);
            put(o + kM2B, p + "mlp.2.bias", kX);
        }
        if (st.find("pipeline.dec_net.res_blocks." + std::to_string(kResBlocks) + ".mlp.0.weight") != nullptr)
            throw std::runtime_error("image engine (Mage-VAE): more than 3 per-pixel blocks");
        put(kFnW, "pipeline.final_layer.norm.weight", kX);
        put(kFlW, "pipeline.final_layer.linear.weight", 3 * kX);
        put(kFlB, "pipeline.final_layer.linear.bias", 3);
        px_ = upload(pw);
        // the NerfEmbedder's features of a 16 x 16 patch: pixel (iy, ix) at (iy / 15, ix / 15), frequencies 8 k / 7
        std::vector<float> dct((size_t) (kPix * 64));
        const double pi = 3.14159265358979323846;
        for (int64_t j = 0; j < kPix; ++j) {
            const int64_t iy = j / kPatch, ix = j % kPatch;
            const double py = (double) iy / 15.0, px = (double) ix / 15.0;
            for (int a = 0; a < 8; ++a)
                for (int b = 0; b < 8; ++b) {
                    const double fa = 8.0 * a / 7.0, fb = 8.0 * b / 7.0;
                    dct[(size_t) (j * 64 + a * 8L + b)] =
                        (float) (std::cos(px * fa * pi) * std::cos(py * fb * pi) / (1.0 + fa * fb));
                }
        }
        dct_ = upload(dct);
        if (dry_) return true;
        if (!gemm_.init(s_, 16, err)) return false;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

uint64_t MfVae::least_scratch_bytes(int64_t h, int64_t w) const {
    return f2_ ? f2_least_bytes(h, w) : scratch_bytes(h, w);
}

uint64_t MfVae::scratch_bytes(int64_t h, int64_t w) const {
    if (f2_) return f2_scratch_bytes(h, w);
    const int64_t n = h * w, band = std::min<int64_t>(n, 1024);
    return (uint64_t) n * (uint64_t) (C_ * 3 + 3 * F_ + kGroups * 2) * 4 + (uint64_t) kWinTok * C_ * 4 +
           (uint64_t) n * (uint64_t) std::max(F_, 3 * C_) * 2 +             // a16
           (uint64_t) std::min(n * 9 * C_, conv::col_values(q_of(s_))) * 2 + // the unfolded neighbourhoods
           (uint64_t) (3 * kWinTok * C_ + kWinTok * kWinTok) * 2 + (uint64_t) kWinTok * kWinTok * 4 +
           (uint64_t) band * 2 * kX * kPix * 4;
}

void MfVae::grow(int64_t h, int64_t w) {
    const int64_t n = h * w;
    if (n <= cap_hw_) return;
    drop_scratch();
    x_ = alloc<float>(allocs_, n * C_, "image engine (Mage-VAE)");
    cond_ = alloc<float>(allocs_, n * C_, "image engine (Mage-VAE)");
    s_state_ = alloc<float>(allocs_, n * C_, "image engine (Mage-VAE)");
    t0_ = alloc<float>(allocs_, n * F_, "image engine (Mage-VAE)");
    t1_ = alloc<float>(allocs_, n * F_, "image engine (Mage-VAE)");
    t2_ = alloc<float>(allocs_, std::max(n * F_, n * C_ + kWinTok * C_), "image engine (Mage-VAE)");   // the attention's output and a window's
    gn_part_ = alloc<float>(allocs_, n * kGroups * 2, "image engine (Mage-VAE)");
    vec_ = alloc<float>(allocs_, 4 * F_, "image engine (Mage-VAE)");
    a16_ = alloc<uint16_t>(allocs_, n * std::max(F_, 3 * C_), "image engine (Mage-VAE)");
    col_cap_ = std::min(n * 9 * C_, conv::col_values(q_of(s_)));
    col16_ = alloc<uint16_t>(allocs_, col_cap_, "image engine (Mage-VAE)");
    w16_ = alloc<uint16_t>(allocs_, 3 * kWinTok * C_ + kWinTok * kWinTok, "image engine (Mage-VAE)");
    scores_ = alloc<float>(allocs_, kWinTok * kWinTok, "image engine (Mage-VAE)");
    cap_band_ = std::min<int64_t>(n, 1024);
    yxb_ = alloc<float>(allocs_, cap_band_ * kX * kPix, "image engine (Mage-VAE)");
    ceb_ = alloc<float>(allocs_, cap_band_ * kX * kPix, "image engine (Mage-VAE)");
    cap_hw_ = n;
}

void MfVae::drop_scratch() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) cond_, (void*) s_state_, (void*) t0_, (void*) t1_, (void*) t2_,
                    (void*) gn_part_, (void*) vec_, (void*) a16_, (void*) col16_, (void*) w16_, (void*) scores_,
                    (void*) yxb_, (void*) ceb_})
        release(allocs_, p);
    x_ = cond_ = s_state_ = t0_ = t1_ = t2_ = gn_part_ = vec_ = scores_ = yxb_ = ceb_ = nullptr;
    a16_ = col16_ = w16_ = nullptr;
    cap_hw_ = cap_band_ = col_cap_ = 0;
    for (void* p : {(void*) f2_x_, (void*) f2_y_, (void*) f2_t_, (void*) f2_q_, (void*) f2_qk16_, (void*) f2_p16_})
        release(allocs_, p);
    f2_x_ = f2_y_ = f2_t_ = nullptr;
    f2_q_ = nullptr;
    f2_qk16_ = f2_p16_ = nullptr;
    f2_cap_ = F2Sizes();
    f2_held_ = 0;
    // a video's slice buffers (not the frames its convolutions carry)
    for (void* p : {(void*) vx_, (void*) vy_, (void*) vt_, (void*) va16_base_, (void*) vtmp_, (void*) vtapw_})
        release(allocs_, p);
    vx_ = vy_ = vt_ = vtapw_ = nullptr;
    va16_ = va16_base_ = nullptr;
    vtmp_ = nullptr;
    vcap_ = vhead_ = vtmp_cap_ = vtapw_cap_ = 0;
}

void MfVae::conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add) {
    const int64_t HW = H * W, K = c.k * c.k * c.in;
    const int p = (int) c.k / 2;
    conv::Request r{in16, H, W, c.in, c.w, c.out, (int) c.k, 1, p, p, p, p};
    r.bias = c.b;
    r.add = add;
    r.out = out;
    conv::drive(core::Runtime::get().stream(s_), r, HW, K, (f2_ ? f2_band_ : col_cap_), c.k == 1,
        [&](int64_t p0, int64_t n) { qi_im2col16(in16, 1, H, W, c.in, 1, (int) c.k, 1, p, H, W, p0, n, col16_, s_); },
        [&](int64_t p0, int64_t n, bool point) {
            gemm_.f16(point ? in16 : col16_, c.w, out + p0 * c.out, n, c.out, point ? c.in : K, 0, false, add);
        },
        [&] { if (c.b != nullptr) qi_bias_add(out, c.b, HW, c.out, s_); });
}

void MfVae::conv_s2(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add) {
    const int64_t OH = H / 2, OW = W / 2, P = OH * OW, K = 9 * c.in;
    conv::Request r{in16, H, W, c.in, c.w, c.out, 3, 2, 0, 0, 1, 1};
    r.bias = c.b;
    r.add = add;
    r.out = out;
    conv::drive(core::Runtime::get().stream(s_), r, P, K, f2_ ? f2_band_ : col_cap_, false,
        [&](int64_t p0, int64_t n) { qi_im2col16(in16, 1, H, W, c.in, 1, 3, 2, 0, OH, OW, p0, n, col16_, s_); },
        [&](int64_t p0, int64_t n, bool) { gemm_.f16(col16_, c.w, out + p0 * c.out, n, c.out, K, 0, false, add); },
        [&] { if (c.b != nullptr) qi_bias_add(out, c.b, P, c.out, s_); });
}

bool MfVae::conv16(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, uint16_t* out16, bool add,
                   uint16_t* side) {
    const int p = (int) c.k / 2;
    conv::Request r{in16, H, W, c.in, c.w, c.out, (int) c.k, 1, p, p, p, p};
    r.bias = c.b;
    r.add = add;
    r.out = out16;
    r.form.y16 = true;
    // cuDNN without bias or residual uses precompiled kernels.  Fusing them cost 0.6 s more for a tiled 1024 x 1024
    // decode on the RTX 4070 (2026-11-08); add them in the engine after that convolution.
    uint16_t* y = add ? side : out16;
    const auto chosen = conv::library(core::Runtime::get().stream(s_), r, y, true);
    if (chosen == conv::Library::cudnn) act_bias16(y, c.b, H * W * c.out, c.out, add, out16, s_);
    return chosen != conv::Library::none;
}

void MfVae::gn_silu16(const float* x, const float* w, const float* b, int64_t HW, int64_t C, bool silu_on,
                      uint16_t* y16) {
    qi_group_norm16(x, HW, C, w, b, kEps, silu_on, y16, gn_part_, s_);
}

void MfVae::resblock(const Res& r, float*& x, int64_t H, int64_t W) {
    const int64_t HW = H * W;
    gn_silu16(x, r.n1w, r.n1b, HW, C_, true, a16_);
    conv(r.c1, a16_, H, W, t2_);
    gn_silu16(t2_, r.n2w, r.n2b, HW, C_, true, a16_);
    conv(r.c2, a16_, H, W, x, true);
}

void MfVae::attention(const Attn& a, float* x, int64_t H, int64_t W) {
    const int64_t HW = H * W;
    float* qkv = x == t0_ ? t1_ : t0_;
    vae_mid::Attention mid{H, W, C_, kWinTok, kWin};
    mid.x = x; mid.weight = a.nw; mid.bias = a.nb; mid.normalized = a16_; mid.norm_workspace = gn_part_;
    mid.qkv = qkv;
    mid.q = w16_; mid.k = w16_ + kWinTok * C_; mid.vt = w16_ + 2 * kWinTok * C_;
    mid.p = w16_ + 3 * kWinTok * C_; mid.scores = scores_;
    mid.out = t2_; mid.window_out = t2_ + HW * C_;
    vae_mid::run(mid, gemm_, s_,
        [&] { conv(a.qkv, a16_, H, W, qkv); },
        [&] {
            conv(a.proj, a16_, H, W, x, true);
        });
}

// a DiCo block's weights under prefix p, with its modulation `mod` [6][C] (host)
MfVae::DiCo MfVae::dico_w(SafeTensors& st, const std::string& p, const std::vector<float>& mod) {
    DiCo b;
    b.c1 = conv_w(st, p + "conv1");
    b.c3 = conv_w(st, p + "conv3");
    b.c4 = conv_w(st, p + "conv4");
    b.c5 = conv_w(st, p + "conv5");
    b.dw = vec(st, p + "conv2.weight");
    b.db = vec(st, p + "conv2.bias");
    b.caw = vec(st, p + "ca.1.weight");
    b.cab = vec(st, p + "ca.1.bias");
    if ((int64_t) mod.size() != 6 * b.c1.out) throw std::runtime_error("image engine (Mage-VAE): " + p + ": modulation");
    b.mod = upload(mod);
    return b;
}

void MfVae::dico(const DiCo& d, float* s, int64_t H, int64_t W, const DiCoBufs& b) {
    const int64_t HW = H * W, C = d.c1.out, F = d.c4.out;
    const float* m = d.mod;
    ln_mod16(s, m, m + C, HW, C, b.a16, s_);
    conv(d.c1, b.a16, H, W, b.t0);
    dw3x3_gelu(b.t0, d.dw, d.db, H, W, C, b.t1, s_);
    channel_gate(b.t1, HW, C, d.caw, d.cab, b.vec, b.vec + C, s_);
    scale16(b.t1, b.vec + C, HW, C, b.a16, s_);
    conv(d.c3, b.a16, H, W, b.t0);
    gate_add(s, b.t0, m + 2 * C, HW, C, s_);
    ln_mod16(s, m + 3 * C, m + 4 * C, HW, C, b.a16, s_);
    conv(d.c4, b.a16, H, W, b.t0);
    qi_gelu_erf16(b.t0, HW * F, b.a16, s_);
    conv(d.c5, b.a16, H, W, b.t1);
    gate_add(s, b.t1, m + 5 * C, HW, C, s_);
}

bool MfVae::open_encoder(std::string& err, bool dry) {
    if (has_encoder()) return true;
    if (f2_) {
        try {
            const bool was_dry = dry_;
            const uint64_t before = weight_bytes_;
            dry_ = dry;
            SafeTensors st(path_);
            open_flux2_encoder(st);
            enc_bytes_ = weight_bytes_ - before;
            if (dry) weight_bytes_ = before;   // sizes only (kept for encode_scratch_bytes): not weights held
            dry_ = was_dry;
            return true;
        } catch (const std::exception& ex) {
            err = ex.what();
            return false;
        }
    }
    try {
        const bool was_dry = dry_;
        const uint64_t before = weight_bytes_;
        dry_ = dry;
        SafeTensors st(path_);
        const std::string e = "student.dconv_encoder.";
        // the patch convolution as a product of the patch rows [h w][3 16 16] (its weight's own order)
        const SafeTensor* pt = st.find(e + "patch_cond_embed.weight");
        if (pt == nullptr || pt->shape.size() != 4 || pt->shape[1] != 3 || pt->shape[2] != kPatch ||
            pt->shape[3] != kPatch)
            throw std::runtime_error("image engine (Mage-VAE): no 16 x 16 patch convolution in the encoder");
        e_patch_.out = pt->shape[0];
        e_patch_.in = 3 * kPix;
        weight_bytes_ += (uint64_t) (e_patch_.out * e_patch_.in) * 2;
        if (!dry_) {
            const auto w = st.f32(e + "patch_cond_embed.weight");
            std::vector<uint16_t> h(w.size());
            for (size_t i = 0; i < w.size(); ++i) h[i] = kernels::f16_from_f32(w[i]);
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (Mage-VAE)");
            gpu::copy(d, h.data(), h.size() * 2);
            e_patch_.w = d;
        }
        e_patch_.b = vec(st, e + "patch_cond_embed.bias");
        auto host = [&](const std::string& n) { return st.f32(n); };
        // the head blocks: their affine layer norms as a modulation (shift = bias, scale = weight - 1), gates 1
        for (int i = 0;; ++i) {
            const std::string p = e + "head_blocks." + std::to_string(i) + ".";
            if (st.find(p + "conv1.weight") == nullptr) break;
            const auto n1w = host(p + "norm1.weight"), n1b = host(p + "norm1.bias");
            const auto n2w = host(p + "norm2.weight"), n2b = host(p + "norm2.bias");
            const size_t C = n1w.size();
            std::vector<float> mod(6 * C, 1.f);
            for (size_t c = 0; c < C; ++c) {
                mod[c] = n1b[c];
                mod[C + c] = n1w[c] - 1.f;
                mod[3 * C + c] = n2b[c];
                mod[4 * C + c] = n2w[c] - 1.f;
            }
            enc_head_.push_back(dico_w(st, p, mod));
        }
        e_down_ = conv_w(st, e + "proj_down");
        // the fuse of [the patches' features, the zero latent's projection (its bias)]: the second half's product
        // is constant, folded into the bias
        e_fuse_ = conv_w(st, e + "fuse_proj", false, 0, e_down_.out);
        {
            const SafeTensor* ft = st.find(e + "fuse_proj.weight");
            const int64_t n = ft->shape[0], k = ft->shape[1];
            const auto fw = host(e + "fuse_proj.weight"), fb = host(e + "fuse_proj.bias"), zb = host(e + "z_proj.bias");
            std::vector<float> b(fb);
            for (int64_t o = 0; o < n; ++o)
                for (int64_t i = e_down_.out; i < k; ++i)
                    b[(size_t) o] += fw[(size_t) (o * k + i)] * zb[(size_t) (i - e_down_.out)];
            e_fuse_.b = upload(b);
        }
        // the blocks at t = 0: the encoder's own timestep embedding, then each block's constant modulation
        std::vector<float> emb(256, 0.f);
        for (int i = 0; i < 128; ++i) emb[(size_t) i] = 1.f;
        auto linear = [](const std::vector<float>& w, const std::vector<float>& b, const std::vector<float>& x) {
            const size_t n = b.size(), k = x.size();
            std::vector<float> y(n);
            for (size_t o = 0; o < n; ++o) {
                double a = b[o];
                for (size_t i = 0; i < k; ++i) a += (double) w[o * k + i] * x[i];
                y[o] = (float) a;
            }
            return y;
        };
        auto c = linear(host(e + "t_embedder.mlp.0.weight"), host(e + "t_embedder.mlp.0.bias"), emb);
        for (float& v : c) v = silu(v);
        c = linear(host(e + "t_embedder.mlp.2.weight"), host(e + "t_embedder.mlp.2.bias"), c);
        for (float& v : c) v = silu(v);
        for (int i = 0;; ++i) {
            const std::string p = e + "blocks." + std::to_string(i) + ".";
            if (st.find(p + "conv1.weight") == nullptr) break;
            enc_blocks_.push_back(dico_w(st, p, linear(host(p + "adaLN_modulation.1.weight"),
                                                       host(p + "adaLN_modulation.1.bias"), c)));
        }
        if (enc_head_.empty() || enc_blocks_.empty()) throw std::runtime_error("image engine (Mage-VAE): no encoder");
        {
            const auto nw = host(e + "norm_out.weight"), nb = host(e + "norm_out.bias");
            std::vector<float> m(2 * nw.size());
            for (size_t i = 0; i < nw.size(); ++i) {
                m[i] = nb[i];
                m[nw.size() + i] = nw[i] - 1.f;
            }
            e_norm_ = upload(m);
        }
        e_out_ = conv_w(st, e + "proj_out");
        enc_bytes_ = weight_bytes_ - before;
        if (dry) {   // sizes only: what was counted stays out of the weights held
            weight_bytes_ = before;
            enc_head_.clear();
            enc_blocks_.clear();
        }
        dry_ = was_dry;
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

uint64_t MfVae::encode_scratch_bytes(int64_t h, int64_t w) const {
    if (f2_) return f2_bytes(f2_enc_walk(h, w));
    const int64_t n = h * w, Ch = enc_head_.empty() ? 768 : enc_head_[0].c1.out,
                  Fh = enc_head_.empty() ? 3072 : enc_head_[0].c4.out;
    return (uint64_t) n * (uint64_t) (Ch + C_ + 2 * Fh) * 4 + (uint64_t) n * (uint64_t) std::max(Fh, 3 * kPix) * 2 +
           (uint64_t) 2 * Ch * 4;
}

void MfVae::encode_moments(const float* rgb, int64_t h, int64_t w, float* moments) {
    if (f2_) throw std::runtime_error("image engine (Flux2 VAE): its encoder gives the latent only (encode)");
    if (!has_encoder()) throw std::runtime_error("image engine (Mage-VAE): the encoder is not loaded");
    const int64_t n = h * w, Ch = enc_head_[0].c1.out, Fh = enc_head_[0].c4.out;
    TempBuffers tmp(s_);
    float* hs = alloc<float>(tmp, n * Ch, "image engine (Mage-VAE)");
    float* ls = alloc<float>(tmp, n * C_, "image engine (Mage-VAE)");
    DiCoBufs b;
    b.t0 = alloc<float>(tmp, n * Fh, "image engine (Mage-VAE)");
    b.t1 = alloc<float>(tmp, n * Fh, "image engine (Mage-VAE)");
    b.a16 = alloc<uint16_t>(tmp, n * std::max(Fh, 3 * kPix), "image engine (Mage-VAE)");
    b.vec = alloc<float>(tmp, 2 * Ch, "image engine (Mage-VAE)");
    patchify16(rgb, h, w, b.a16, s_);
    conv(e_patch_, b.a16, h, w, hs);
    for (const DiCo& d : enc_head_) dico(d, hs, h, w, b);
    kernels::f32_to_f16_bulk(hs, b.a16, n * Ch, s_);
    conv(e_down_, b.a16, h, w, ls);
    kernels::f32_to_f16_bulk(ls, b.a16, n * C_, s_);
    conv(e_fuse_, b.a16, h, w, hs);   // the state [n][C] (in hs's room)
    for (const DiCo& d : enc_blocks_) dico(d, hs, h, w, b);
    ln_mod16(hs, e_norm_, e_norm_ + C_, n, C_, b.a16, s_);
    conv(e_out_, b.a16, h, w, moments);
    gpu::stream_sync(s_);
}

void MfVae::encode(const float* rgb, int64_t h, int64_t w, const float* eps, float* z, uint64_t limit) {
    if (f2_) {
        if (f2_bytes(f2_enc_walk(h, w)) <= limit) {
            f2_encode(rgb, h, w, z);
            if (overflowed(z, h * w * 4 * f2_lc_)) f2_encode(rgb, h, w, z);
            return;
        }
        // in tiles: the least work whose tiles' encode, their image and the blend fit the limit
        const auto bytes = [&](int64_t th, int64_t tw) {
            return f2_bytes(f2_enc_walk(th, tw)) + (uint64_t) (3 * th * tw * kPix + th * tw * 4 * f2_lc_) * 4 +
                   (uint64_t) ((4 * f2_lc_ + 1) * h * w) * 4;
        };
        auto [th, tw] =
            tile_sides(h, w, kF2Overlap, kF2TileMin, [&](int64_t a, int64_t b) { return bytes(a, b) <= limit; });
        if (th == 0) {   // nothing fits: the least, which may still find the room
            th = std::min(h, kF2TileMin);
            tw = std::min(w, kF2TileMin);
        }
        f2_encode_tiled(rgb, h, w, z, th, tw);
        if (overflowed(z, h * w * 4 * f2_lc_)) f2_encode_tiled(rgb, h, w, z, th, tw);
        return;
    }
    const int64_t n = h * w, Z = e_out_.out / 2;
    float* mom = nullptr;
    if (!gpu::alloc_device(&mom, (size_t) (n * 2 * Z) * 4))
        throw std::runtime_error(std::string("image engine (Mage-VAE): GPU memory: ") + gpu::last_error());
    encode_moments(rgb, h, w, mom);
    posterior(mom, eps, n, Z, z, s_);
    gpu::stream_sync(s_);
    gpu::free(mom);
}

void MfVae::decode_raw(const float* z, int64_t h, int64_t w, float* rgb) {
    if (f2_) {
        f2_decode_raw(z, h, w, rgb);
        if (overflowed(rgb, 3 * h * w * kPix)) f2_decode_raw(z, h, w, rgb);
        return;
    }
    grow(h, w);
    const int64_t HW = h * w, H = h * kPatch, W = w * kPatch;
    // the CoD decoder: the condition [HW][C]
    kernels::f32_to_f16_bulk(z, a16_, HW * cin_.in, s_);
    float* x = x_;
    conv(cin_, a16_, h, w, x);
    resblock(res_[0], x, h, w);
    attention(attn_[0], x, h, w);
    resblock(res_[1], x, h, w);
    attention(attn_[1], x, h, w);
    resblock(res_[2], x, h, w);
    gn_silu16(x, gnw_, gnb_, HW, C_, true, a16_);
    conv(cout_, a16_, h, w, cond_);
    // the denoiser's blocks on the per-patch state
    kernels::f32_to_f16_bulk(cond_, a16_, HW * C_, s_);
    conv(s_in_, a16_, h, w, s_state_);
    DiCoBufs b;
    b.t0 = t0_;
    b.t1 = t1_;
    b.a16 = a16_;
    b.vec = vec_;
    for (const DiCo& d : blocks_) dico(d, s_state_, h, w, b);
    // the per-pixel MLP, a band of patches at a time
    for (int64_t p0 = 0; p0 < HW; p0 += cap_band_) {
        const int64_t np = std::min(cap_band_, HW - p0);
        kernels::f32_to_f16_bulk(cond_ + p0 * C_, a16_, np * C_, s_);
        gemm_.f16(a16_, yx_.w, yxb_, np, yx_.out, yx_.in);
        qi_bias_add(yxb_, yx_.b, np, yx_.out, s_);
        kernels::f32_to_f16_bulk(s_state_ + p0 * C_, a16_, np * C_, s_);
        gemm_.f16(a16_, ce_.w, ceb_, np, ce_.out, ce_.in);
        qi_bias_add(ceb_, ce_.b, np, ce_.out, s_);
        pixel_mlp(yxb_, ceb_, px_, dct_, p0, np, w, H, W, rgb, s_);
    }
}

void MfVae::decode(const float* z, int64_t h, int64_t w, float* rgba, uint64_t limit) {
    const int64_t HW = h * w * kPix;   // the Flux2 VAE's 8 x 8 a position of 2 x 2 positions a token: 16 x 16 too
    float* rgb = nullptr;
    if (!gpu::alloc_device(&rgb, (size_t) (3 * HW) * 4))
        throw std::runtime_error(std::string("image engine (Mage-VAE): GPU memory: ") + gpu::last_error());
    tiled_ = f2_ && f2_scratch_bytes(h, w) > limit;
    if (tiled_) {   // the Flux2 VAE where the whole does not fit: the deepest front and the tiles of least work that do
        F2Tiling g = f2_tiling(h, w, limit);
        if (g.k < 0) {   // nothing fits: the least, which may still find the room
            g.k = 0;
            g.th = std::min(h, kF2TileMin);
            g.tw = std::min(w, kF2TileMin);
        }
        f2_decode_tiled(z, h, w, rgb, g.k, g.th, g.tw);
        if (overflowed(rgb, 3 * HW)) f2_decode_tiled(z, h, w, rgb, g.k, g.th, g.tw);
    } else {
        decode_raw(z, h, w, rgb);
    }
    to_rgba(rgb, HW, 3, TilePlanes::Planes, rgba, s_);
    gpu::stream_sync(s_);
    gpu::free(rgb);
}

// ---- the Flux2 VAE

void MfVae::open_flux2(SafeTensors& st) {
    f2_ = true;
    if (const SafeTensor* bm = st.find("bn.running_mean")) {   // SeedVR2's VAE has none
        if (bm->elements() != 128)
            throw std::runtime_error("image engine (Flux2 VAE): the batch norm is not of 128 values");
        const auto m = st.f32("bn.running_mean"), v = st.f32("bn.running_var");
        std::vector<float> sd(v.size());
        for (size_t i = 0; i < v.size(); ++i) sd[i] = std::sqrt(v[i] + 1e-4f);   // the batch norm's eps
        f2_mean_ = upload(m);
        f2_std_ = upload(sd);
    }
    // the names as the LDM checkpoint has them (BFL's ae.safetensors, Comfy-Org's flux2-vae), or as diffusers' (the
    // vae/ folder of FLUX.2-klein): nm takes the first to the file's
    const bool df = st.find("decoder.mid_block.resnets.0.conv1.weight") != nullptr;
    int dl = 0;
    while (df && st.find("decoder.up_blocks." + std::to_string(dl) + ".resnets.0.conv1.weight") != nullptr) ++dl;
    auto nm = [df, dl](const std::string& k) -> std::string {
        if (!df) return k;
        static const std::pair<const char*, const char*> fixed[] = {
            {"decoder.post_quant_conv", "post_quant_conv"},
            {"decoder.mid.block_1.", "decoder.mid_block.resnets.0."},
            {"decoder.mid.block_2.", "decoder.mid_block.resnets.1."},
            {"decoder.mid.attn_1.norm", "decoder.mid_block.attentions.0.group_norm"},
            {"decoder.mid.attn_1.q", "decoder.mid_block.attentions.0.to_q"},
            {"decoder.mid.attn_1.k", "decoder.mid_block.attentions.0.to_k"},
            {"decoder.mid.attn_1.v", "decoder.mid_block.attentions.0.to_v"},
            {"decoder.mid.attn_1.proj_out", "decoder.mid_block.attentions.0.to_out.0"},
            {"decoder.norm_out", "decoder.conv_norm_out"}};
        for (const auto& [from, to] : fixed) {   // a whole name, or a prefix of the names under it
            const std::string f(from);
            if (k.rfind(f, 0) == 0 && (k.size() == f.size() || f.back() == '.' || k[f.size()] == '.'))
                return to + k.substr(f.size());
        }
        if (k.rfind("decoder.up.", 0) == 0) {   // diffusers counts the levels the other way, from the largest
            const size_t e = k.find('.', 11);
            std::string rest = k.substr(e + 1);
            if (rest.rfind("block.", 0) == 0) {
                rest = "resnets." + rest.substr(6);
                const size_t n = rest.find("nin_shortcut");
                if (n != std::string::npos) rest.replace(n, 12, "conv_shortcut");
            } else if (rest.rfind("upsample.", 0) == 0) {
                rest = "upsamplers.0." + rest.substr(9);
            }
            return "decoder.up_blocks." + std::to_string(dl - 1 - std::stoi(k.substr(11, e - 11))) + "." + rest;
        }
        return k;
    };
    const std::string d = "decoder.";
    f2_pq_ = Conv();
    if (st.find(nm(d + "post_quant_conv") + ".weight") != nullptr) f2_pq_ = conv_w(st, nm(d + "post_quant_conv"));
    cin_ = conv_w(st, nm(d + "conv_in"));
    f2_lc_ = f2_pq_.out > 0 ? f2_pq_.in : cin_.in;
    if ((f2_pq_.out > 0 && f2_pq_.out != cin_.in) || (f2_mean_ != nullptr && f2_lc_ != 32) || f2_lc_ % 8 != 0)
        throw std::runtime_error("image engine (Flux2 VAE): the latent's width is not the batch norm's or conv_in's");
    auto res = [&](const std::string& p) {
        F2Res r;
        r.n1w = vec(st, nm(p + "norm1.weight"));
        r.n1b = vec(st, nm(p + "norm1.bias"));
        r.c1 = conv_w(st, nm(p + "conv1"));
        r.n2w = vec(st, nm(p + "norm2.weight"));
        r.n2b = vec(st, nm(p + "norm2.bias"));
        r.c2 = conv_w(st, nm(p + "conv2"));
        if (st.find(nm(p + "nin_shortcut.weight")) != nullptr) r.nin = conv_w(st, nm(p + "nin_shortcut"));
        if (r.c1.in % kGroups != 0 || r.c1.out % kGroups != 0)
            throw std::runtime_error("image engine (Flux2 VAE): " + p + ": channels not a multiple of 32");
        return r;
    };
    f2_mid_[0] = res(d + "mid.block_1.");
    const std::string a = d + "mid.attn_1.", qkv[3] = {nm(a + "q"), nm(a + "k"), nm(a + "v")};
    f2_attn_ = attn_w(st, nm(a + "norm"), qkv, nm(a + "proj_out"), cin_.out);
    f2_mid_[1] = res(d + "mid.block_2.");
    int levels = 0;
    while (st.find(nm(d + "up." + std::to_string(levels) + ".block.0.conv1.weight")) != nullptr) ++levels;
    if (levels == 0) throw std::runtime_error("image engine (Flux2 VAE): no up levels");
    for (int i = levels - 1; i >= 0; --i) {   // the decoder runs them from the last (the smallest size)
        const std::string p = d + "up." + std::to_string(i) + ".";
        F2Level l;
        for (int j = 0; st.find(nm(p + "block." + std::to_string(j) + ".conv1.weight")) != nullptr; ++j)
            l.blocks.push_back(res(p + "block." + std::to_string(j) + "."));
        if (st.find(nm(p + "upsample.conv.weight")) != nullptr) l.up = conv_w(st, nm(p + "upsample.conv"));
        if (const SafeTensor* u = st.find(nm(p + "upsample.upscale_conv.weight"))) {
            // SeedVR2's: "(x y z c)" output channels, z the frames in time (2 where it doubles them); the first
            // frame's (z = 0) 4 C kept
            const int64_t C = u->shape[1], Z = u->shape[0] / (4 * C);
            if (Z < 1 || u->shape[0] != 4 * Z * C) throw std::runtime_error("image engine (VAE): " + p + "upscale_conv");
            std::vector<int64_t> rows;
            for (int64_t xy = 0; xy < 4; ++xy)
                for (int64_t c = 0; c < C; ++c) rows.push_back(xy * Z * C + c);
            l.shuffle = conv_w(st, nm(p + "upsample.upscale_conv"), true, 0, -1, &rows);
            if (video_) {   // all its outputs: "(x y z c)", 2 x 2 positions of z frames
                l.shuffle_t = conv_w(st, nm(p + "upsample.upscale_conv"));
                l.z = Z;
            }
        }
        f2_levels_.push_back(std::move(l));
    }
    int ups = 0;
    for (const F2Level& l : f2_levels_) ups += l.up.out > 0;
    if (ups != 3) throw std::runtime_error("image engine (Flux2 VAE): not three upsamplings (8 x 8 pixels a position)");
    gnw_ = vec(st, nm(d + "norm_out.weight"));
    gnb_ = vec(st, nm(d + "norm_out.bias"));
    cout_ = conv_w(st, nm(d + "conv_out"));
    C_ = cin_.out;
    f2_ac_ = C_;
}

// The sizes of the stages from level k0 up to k1 (front: from the latent, its 1 x 1 convolution, conv_in and the
// middle first; tail: norm_out and conv_out after) of an h x w latent's region: the largest tensor (most, values), the
// largest number of positions (pix), the middle's positions (mid, its attention's; 0 without the front), and its band
// of query rows whose scores take 64 MiB (16M FP32)
MfVae::F2Sizes MfVae::f2_walk(int64_t h, int64_t w, int k0, int k1, bool front, bool tail) const {
    F2Sizes z;
    int64_t pix = 4 * h * w, c = C_;
    for (int i = 0; i < k0; ++i) {
        const F2Level& l = f2_levels_[(size_t) i];
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out > 0) {
            pix *= 4;
            c = l.up.out;
        }
    }
    if (front) {
        z.mid = pix;
        z.most = std::max(pix * f2_lc_, pix * 3 * C_);
        z.rows = std::max<int64_t>(1, std::min<int64_t>(z.mid, (int64_t) (16 << 20) / z.mid));
    }
    z.most = std::max(z.most, pix * c);
    for (int i = k0; i < k1; ++i) {
        const F2Level& l = f2_levels_[(size_t) i];
        for (const F2Res& r : l.blocks) z.most = std::max(z.most, pix * std::max(r.c1.in, r.c2.out));
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out > 0) {
            pix *= 4;
            z.most = std::max(z.most, pix * c);   // the upsampled input (SeedVR2's: its 1 x 1 convolution's 4 c)
            c = l.up.out;
            z.most = std::max(z.most, pix * c);
        }
    }
    if (tail) z.most = std::max(z.most, pix * 3);
    z.pix = pix;
    z.c = c;
    return z;
}

uint64_t MfVae::f2_bytes(const F2Sizes& z) const {
    int64_t cmax = C_;
    for (const F2Level& l : f2_levels_)
        for (const F2Res& r : l.blocks) cmax = std::max(cmax, r.c1.in);
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) cmax = std::max(cmax, r.c1.in);
        cmax = std::max(cmax, l.down.in);
    }
    const int64_t col = std::min(z.pix * 9 * cmax, conv::col_values(q_of(s_)));
    return (uint64_t) z.most * (3 * 2 + 2 + 4) +                       // x, y, t (16 bits); a16; the FP32 sums
           (uint64_t) col * 2 +                                          // the unfolded neighbourhoods, a band
           (uint64_t) qi_group_norm_ws_floats(z.pix) * 4 + 4 * (uint64_t) cmax * 4 +   // the group norms' sums, stats
           (uint64_t) z.mid * f2_ac_ * 3 * 2 +                         // q, k, v^T
           (uint64_t) z.rows * z.mid * (4 + 2);                        // a band's scores, probabilities
}

uint64_t MfVae::f2_scratch_bytes(int64_t h, int64_t w) const {
    return f2_bytes(f2_walk(h, w, 0, (int) f2_levels_.size(), true, true));
}

// the larger of two stages' sizes, which share the buffers
static MfVae::F2Sizes f2_max(const MfVae::F2Sizes& a, const MfVae::F2Sizes& b) {
    MfVae::F2Sizes z;
    z.most = std::max(a.most, b.most);
    z.pix = std::max(a.pix, b.pix);
    z.mid = std::max(a.mid, b.mid);
    z.rows = std::max(a.rows, b.rows);
    return z;
}

void MfVae::f2_grow(const F2Sizes& z) {
    if (z.most <= f2_cap_.most && z.pix <= f2_cap_.pix && z.mid <= f2_cap_.mid && z.rows <= f2_cap_.rows) return;
    const F2Sizes n = f2_max(z, f2_cap_);
    drop_scratch();
    int64_t cmax = C_;
    for (const F2Level& l : f2_levels_)
        for (const F2Res& r : l.blocks) cmax = std::max(cmax, r.c1.in);
    // the convolutions' bands by the decoder's widths alone, so that a decode is the same whether the encoder (wider
    // in MOSS-Image's VAE) is loaded or not
    f2_band_ = std::min(n.pix * 9 * cmax, conv::col_values(q_of(s_)));
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) cmax = std::max(cmax, r.c1.in);
        cmax = std::max(cmax, l.down.in);
    }
    f2_x_ = alloc<uint16_t>(allocs_, n.most, "image engine (Mage-VAE)");
    f2_y_ = alloc<uint16_t>(allocs_, n.most, "image engine (Mage-VAE)");
    f2_t_ = alloc<uint16_t>(allocs_, n.most, "image engine (Mage-VAE)");
    vtmp_ = alloc<float>(allocs_, n.most, "image engine (Mage-VAE)");
    vtmp_cap_ = n.most;
    a16_ = alloc<uint16_t>(allocs_, n.most, "image engine (Mage-VAE)");
    col_cap_ = std::min(n.pix * 9 * cmax, conv::col_values(q_of(s_)));
    col16_ = alloc<uint16_t>(allocs_, col_cap_, "image engine (Mage-VAE)");
    gn_part_ = alloc<float>(allocs_, qi_group_norm_ws_floats(n.pix), "image engine (Mage-VAE)");
    vec_ = alloc<float>(allocs_, 4 * cmax, "image engine (Mage-VAE)");
    f2_qk16_ = alloc<uint16_t>(allocs_, n.mid * f2_ac_ * 3, "image engine (Mage-VAE)");
    f2_q_ = alloc<float>(allocs_, n.rows * n.mid, "image engine (Mage-VAE)");
    f2_p16_ = alloc<uint16_t>(allocs_, n.rows * n.mid, "image engine (Mage-VAE)");
    f2_cap_ = n;
    f2_held_ = f2_bytes(n);
}

// the two of x, y, t that are not `x`
void MfVae::f2_others(const uint16_t* x, uint16_t*& a, uint16_t*& b) const {
    uint16_t* all[3] = {f2_x_, f2_y_, f2_t_};
    int k = 0;
    for (uint16_t* p : all)
        if (p != x) (k++ == 0 ? a : b) = p;
}

// n activations as a convolution's FP16 input: themselves, or (BF16) made FP16 in `room`
const uint16_t* MfVae::f2_in16(const uint16_t* x, int64_t n, uint16_t* room) {
    if (!vbf_) return x;
    bf16_to_f16(x, n, room, s_);
    return room;
}

// x = x (or its 1 x 1 shortcut) + conv2(SiLU(GN(conv1(SiLU(GN(x)))))): the result in another of x, y, t
void MfVae::f2_resblock(const F2Res& r, uint16_t*& x, int64_t H, int64_t W) {
    const int64_t HW = H * W;
    uint16_t *t = nullptr, *y = nullptr;
    f2_others(x, t, y);
    qi_group_norm16(x, vbf_, HW, r.c1.in, r.n1w, r.n1b, kEps, true, a16_, gn_part_, s_);
    v_frame(r.c1, a16_, H, W, false, t, true, false);
    qi_group_norm16(t, vbf_, HW, r.c1.out, r.n2w, r.n2b, kEps, true, a16_, gn_part_, s_);
    if (r.nin.out > 0) {
        v_frame(r.c2, a16_, H, W, false, y, true, false);
        v_frame(r.nin, f2_in16(x, HW * r.nin.in, a16_), H, W, false, y, true, true, t);
        x = y;
    } else {
        v_frame(r.c2, a16_, H, W, false, x, true, true, y);   // x itself takes the sum
    }
}

// x += proj(softmax(q k^T / sqrt(C)) v), q, k, v of GN(x): one head over every position, a band of query rows at a time
void MfVae::f2_attention(const Attn& a, uint16_t* x, int64_t H, int64_t W) {
    const int64_t T = H * W, C = a.qkv.in, rows = f2_cap_.rows;
    uint16_t *qkv = nullptr, *unused = nullptr;
    f2_others(x, qkv, unused);
    vae_mid::Attention mid{H, W, C, rows};
    mid.x16 = x; mid.weight = a.nw; mid.bias = a.nb; mid.normalized = a16_; mid.norm_workspace = gn_part_;
    mid.qkv16 = qkv; mid.bf = vbf_; mid.group_softmax = true;
    mid.q = f2_qk16_; mid.k = f2_qk16_ + T * C; mid.vt = f2_qk16_ + 2 * T * C;
    mid.p = f2_p16_; mid.scores = f2_q_; mid.out = reinterpret_cast<float*>(qkv);
    vae_mid::run(mid, gemm_, s_,
        [&] { v_frame(a.qkv, a16_, H, W, false, qkv, true, false); },
        [&] {
            v_frame(a.proj, a16_, H, W, false, x, true, true, unused);
        });
}

// the stages up to level k of the latent z (h x w): x [H][W][c] in one of x, y, t
void MfVae::f2_front(const float* z, int64_t h, int64_t w, int k, uint16_t*& x, int64_t& H, int64_t& W,
                     int64_t& c) {
    H = 2 * h;
    W = 2 * w;
    f2_unpatch16(z, f2_mean_, f2_std_, h, w, f2_lc_, a16_, s_);
    if (f2_pq_.out > 0) {
        conv(f2_pq_, a16_, H, W, vtmp_);
        kernels::f32_to_f16_bulk(vtmp_, a16_, H * W * f2_pq_.out, s_);
    }
    x = f2_x_;
    v_frame(cin_, a16_, H, W, false, x, true, false);
    f2_resblock(f2_mid_[0], x, H, W);
    f2_attention(f2_attn_, x, H, W);
    f2_resblock(f2_mid_[1], x, H, W);
    c = C_;
    f2_levels(0, k, x, H, W, c);
}

// levels k0 .. k1 - 1 on x [H][W][c]
void MfVae::f2_levels(int k0, int k1, uint16_t*& x, int64_t& H, int64_t& W, int64_t& c) {
    for (int i = k0; i < k1; ++i) {
        const F2Level& l = f2_levels_[(size_t) i];
        for (const F2Res& r : l.blocks) f2_resblock(r, x, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out > 0) {
            uint16_t *y = nullptr, *t = nullptr;
            f2_others(x, y, t);
            if (l.shuffle.out > 0) {   // SeedVR2's: its 1 x 1 convolution's 4 c spread over 2 x 2 positions
                conv(l.shuffle, f2_in16(x, H * W * c, a16_), H, W, vtmp_);
                f2_shuffle16(vtmp_, H, W, c, a16_, s_);
            } else {
                qi_up2x16(f2_in16(x, H * W * c, t), H, W, c, a16_, s_);
            }
            H *= 2;
            W *= 2;
            v_frame(l.up, a16_, H, W, false, y, true, false);
            x = y;
            c = l.up.out;
        }
    }
}

// norm_out and conv_out of x [H][W][c]: rgb [3][H][W]
void MfVae::f2_tail(const uint16_t* x, int64_t H, int64_t W, int64_t c, float* rgb) {
    qi_group_norm16(x, vbf_, H * W, c, gnw_, gnb_, kEps, true, a16_, gn_part_, s_);
    uint16_t *y16 = nullptr, *unused = nullptr;
    f2_others(x, y16, unused);
    float* y = reinterpret_cast<float*>(y16);   // the RGB in FP32 (3 channels: in room)
    conv(cout_, a16_, H, W, y);
    hwc3_to_chw(y, H * W, rgb, s_);
}

void MfVae::f2_decode_raw(const float* z, int64_t h, int64_t w, float* rgb) {
    const int n = (int) f2_levels_.size();
    f2_grow(f2_walk(h, w, 0, n, true, true));
    uint16_t* x = nullptr;
    int64_t H = 0, W = 0, c = 0;
    f2_front(z, h, w, n, x, H, W, c);
    f2_tail(x, H, W, c, rgb);
}

// A tiled decode: the stages up to level k on the whole latent (its statistics and the middle's attention over all of
// it), their output kept, and the levels from k on in tiles of th x tw tokens.  Its scratch: the buffers the two
// share, the output kept, a tile's input and image, and the blend's planes
uint64_t MfVae::f2_tiled_bytes(int64_t h, int64_t w, int k, int64_t th, int64_t tw) const {
    const int n = (int) f2_levels_.size();
    th = std::min(h, th);
    tw = std::min(w, tw);
    const F2Sizes f = f2_walk(h, w, 0, k, true, false), b = f2_walk(th, tw, k, n, false, true);
    const F2Sizes bi = f2_walk(th, tw, k, k, false, false);   // a tile's input
    return f2_bytes(f2_max(f, b)) + (uint64_t) (f.pix * f.c) * 2 + (uint64_t) (bi.pix * bi.c) * 2 +
           (uint64_t) (3 * th * tw * kPix) * 4 + (uint64_t) (4 * h * w * kPix) * 4;
}

// the deepest front whose least tiles fit `limit`, and its tiles of the least work within it; k -1 when nothing fits
MfVae::F2Tiling MfVae::f2_tiling(int64_t h, int64_t w, uint64_t limit) const {
    const int n = (int) f2_levels_.size();
    const int64_t lh = std::min(h, kF2TileMin), lw = std::min(w, kF2TileMin);
    for (int k = n; k >= 0; --k) {
        if (f2_tiled_bytes(h, w, k, lh, lw) > limit) continue;
        const auto [th, tw] =
            tile_sides(h, w, kF2Overlap, kF2TileMin,
                       [&](int64_t a, int64_t b) { return f2_tiled_bytes(h, w, k, a, b) <= limit; });
        if (th > 0) return {k, th, tw};
    }
    return {};
}

uint64_t MfVae::f2_least_bytes(int64_t h, int64_t w) const {
    const int n = (int) f2_levels_.size();
    const int64_t lh = std::min(h, kF2TileMin), lw = std::min(w, kF2TileMin);
    uint64_t b = f2_scratch_bytes(h, w);
    for (int k = 0; k <= n; ++k) b = std::min(b, f2_tiled_bytes(h, w, k, lh, lw));
    return b;
}

void MfVae::f2_decode_tiled(const float* z, int64_t h, int64_t w, float* rgb, int k, int64_t th, int64_t tw) {
    const int n = (int) f2_levels_.size();
    th = std::min(h, th);
    tw = std::min(w, tw);
    const int64_t H = h * kPatch, W = w * kPatch;
    const F2Sizes f = f2_walk(h, w, 0, k, true, false), b = f2_walk(th, tw, k, n, false, true);
    f2_grow(f2_max(f, b));
    TempBuffers tmp(s_);
    float* acc = alloc<float>(tmp, 4 * H * W, "image engine (Mage-VAE)");
    float* trgb = alloc<float>(tmp, 3 * th * tw * kPix, "image engine (Mage-VAE)");
    gpu::memset_async(acc, 0, (size_t) (4 * H * W) * 4, s_);
    // the front on the whole, its output kept [FH][FW][c]: s positions a token
    uint16_t* x = nullptr;
    int64_t FH = 0, FW = 0, c = 0;
    f2_front(z, h, w, k, x, FH, FW, c);
    const int64_t sp = FH / h;
    uint16_t* kept = alloc<uint16_t>(tmp, FH * FW * c, "image engine (Mage-VAE)");
    gpu::copy_async(kept, x, (size_t) (FH * FW * c) * 2, s_);
    const auto ys = tile_positions(h, th, kF2Overlap), xs = tile_positions(w, tw, kF2Overlap);
    for (const int64_t y0 : ys)
        for (const int64_t x0 : xs) {
            int64_t TH = th * sp, TW = tw * sp, tc = c;
            uint16_t* tx = f2_x_;
            tile_crop(kept, c, TilePlanes::Interleaved, FH, FW, y0 * sp, x0 * sp, TH, TW, tx, s_);
            f2_levels(k, n, tx, TH, TW, tc);
            f2_tail(tx, TH, TW, tc, trgb);
            tile_blend(trgb, 3, TilePlanes::Planes, th * kPatch, tw * kPatch, y0 * kPatch, x0 * kPatch, H, W,
                       kF2Overlap * kPatch, TileRamp::Centre, y0 > 0, y0 + th < h, x0 > 0, x0 + tw < w, acc,
                       acc + 3 * H * W, s_);
        }
    tile_finish(acc, 3, TilePlanes::Planes, H * W, acc + 3 * H * W, rgb, s_);
    gpu::stream_sync(s_);
}

// ---- the Flux2 VAE's encoder

// The encoder in tiles of t x t tokens (a tile's image 16 t pixels a side, the last ones flush with the edges), each
// encoded whole (its group norms and its middle's attention over the tile) and the
// tiles' latents blended linearly across the kF2Overlap tokens two of them share
void MfVae::f2_encode_tiled(const float* rgb, int64_t h, int64_t w, float* z, int64_t th, int64_t tw) {
    th = std::min(h, th);
    tw = std::min(w, tw);
    const int64_t H = h * kPatch, W = w * kPatch, V = 4 * f2_lc_;
    TempBuffers tmp(s_);
    float* acc = alloc<float>(tmp, (V + 1) * h * w, "image engine (Mage-VAE)");
    float* trgb = alloc<float>(tmp, 3 * th * tw * kPix, "image engine (Mage-VAE)");
    float* tz = alloc<float>(tmp, th * tw * V, "image engine (Mage-VAE)");
    gpu::memset_async(acc, 0, (size_t) ((V + 1) * h * w) * 4, s_);
    const auto ys = tile_positions(h, th, kF2Overlap), xs = tile_positions(w, tw, kF2Overlap);
    for (const int64_t y0 : ys)
        for (const int64_t x0 : xs) {
            for (int64_t c = 0; c < 3; ++c)   // the tile's image, a plane at a time
                tile_crop(rgb + c * H * W, 1, TilePlanes::Planes, H, W, y0 * kPatch, x0 * kPatch, th * kPatch,
                          tw * kPatch, trgb + c * th * tw * kPix, s_);
            f2_encode(trgb, th, tw, tz);
            tile_blend(tz, V, TilePlanes::Interleaved, th, tw, y0, x0, h, w, kF2Overlap, TileRamp::Centre, y0 > 0,
                       y0 + th < h, x0 > 0, x0 + tw < w, acc, acc + V * h * w, s_);
        }
    tile_finish(acc, V, TilePlanes::Interleaved, h * w, acc + V * h * w, z, s_);
    gpu::stream_sync(s_);
}

void MfVae::open_flux2_encoder(SafeTensors& st) {
    // the LDM checkpoint's names, or diffusers' (FLUX.2-klein's vae/: down_blocks counted the same way, from the
    // largest; quant_conv outside the encoder)
    const bool df = st.find("encoder.mid_block.resnets.0.conv1.weight") != nullptr;
    auto nm = [df](const std::string& k) -> std::string {
        if (!df) return k;
        static const std::pair<const char*, const char*> fixed[] = {
            {"encoder.quant_conv", "quant_conv"},
            {"encoder.mid.block_1.", "encoder.mid_block.resnets.0."},
            {"encoder.mid.block_2.", "encoder.mid_block.resnets.1."},
            {"encoder.mid.attn_1.norm", "encoder.mid_block.attentions.0.group_norm"},
            {"encoder.mid.attn_1.q", "encoder.mid_block.attentions.0.to_q"},
            {"encoder.mid.attn_1.k", "encoder.mid_block.attentions.0.to_k"},
            {"encoder.mid.attn_1.v", "encoder.mid_block.attentions.0.to_v"},
            {"encoder.mid.attn_1.proj_out", "encoder.mid_block.attentions.0.to_out.0"},
            {"encoder.norm_out", "encoder.conv_norm_out"}};
        for (const auto& [from, to] : fixed) {   // a whole name, or a prefix of the names under it
            const std::string f(from);
            if (k.rfind(f, 0) == 0 && (k.size() == f.size() || f.back() == '.' || k[f.size()] == '.'))
                return to + k.substr(f.size());
        }
        if (k.rfind("encoder.down.", 0) == 0) {
            const size_t e = k.find('.', 13);
            std::string rest = k.substr(e + 1);
            if (rest.rfind("block.", 0) == 0) {
                rest = "resnets." + rest.substr(6);
                const size_t n = rest.find("nin_shortcut");
                if (n != std::string::npos) rest.replace(n, 12, "conv_shortcut");
            } else if (rest.rfind("downsample.", 0) == 0) {
                rest = "downsamplers.0." + rest.substr(11);
            }
            return "encoder.down_blocks." + k.substr(13, e - 13) + "." + rest;
        }
        return k;
    };
    const std::string e = "encoder.";
    if (st.find(nm(e + "conv_in.weight")) == nullptr)
        throw std::runtime_error("image engine (Flux2 VAE): the file has no encoder");
    {   // conv_in's 3 input channels widened to 8 (zero weights), as the products take them
        const std::string n = nm(e + "conv_in");
        const SafeTensor* t = st.find(n + ".weight");
        const size_t r = t->shape.size();   // [out][3][3][3], or [out][3][kt][3][3] (SeedVR2's: its time taps summed)
        if ((r != 4 && r != 5) || t->shape[1] != 3 || t->shape[r - 2] != 3 || t->shape[r - 1] != 3)
            throw std::runtime_error("image engine (Flux2 VAE): the encoder's conv_in is not 3 x 3 over RGB");
        const int64_t kt = r == 5 ? t->shape[2] : 1;
        f2_ein_.out = t->shape[0];
        f2_ein_.in = 8;
        f2_ein_.k = 3;
        weight_bytes_ += (uint64_t) (f2_ein_.out * 9 * 8) * 2;
        const bool taps = video_ && kt > 1;
        if (taps) {
            f2_ein_.kt = kt;
            weight_bytes_ += (uint64_t) (kt * f2_ein_.out * 9 * 8) * 2;
        }
        if (!dry_) {
            const auto w = st.f32(n + ".weight");   // [out][3][kt][3][3]
            std::vector<uint16_t> h((size_t) (f2_ein_.out * 9 * 8), 0);
            for (int64_t o = 0; o < f2_ein_.out; ++o)
                for (int64_t i = 0; i < 3; ++i)
                    for (int64_t k = 0; k < 9; ++k) {
                        float v = 0.f;
                        for (int64_t z = 0; z < kt; ++z) v += w[(size_t) (((o * 3 + i) * kt + z) * 9 + k)];
                        h[(size_t) ((o * 9 + k) * 8 + i)] = kernels::f16_from_f32(v);
                    }
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (Mage-VAE)");
            gpu::copy(d, h.data(), h.size() * 2);
            f2_ein_.w = d;
            if (taps) {   // [out][kt][9][8], the input widened to 8 alike
                std::vector<uint16_t> tw((size_t) (kt * f2_ein_.out * 9 * 8), 0);
                for (int64_t z = 0; z < kt; ++z)
                    for (int64_t o = 0; o < f2_ein_.out; ++o)
                        for (int64_t i = 0; i < 3; ++i)
                            for (int64_t k = 0; k < 9; ++k)
                                tw[(size_t) (((o * kt + z) * 9 + k) * 8 + i)] =
                                    kernels::f16_from_f32(w[(size_t) (((o * 3 + i) * kt + z) * 9 + k)]);
                uint16_t* td = alloc<uint16_t>(allocs_, (int64_t) tw.size(), "image engine (Mage-VAE)");
                gpu::copy(td, tw.data(), tw.size() * 2);
                f2_ein_.taps = td;
            }
        }
        f2_ein_.b = vec(st, n + ".bias");
    }
    auto res = [&](const std::string& p) {
        F2Res r;
        r.n1w = vec(st, nm(p + "norm1.weight"));
        r.n1b = vec(st, nm(p + "norm1.bias"));
        r.c1 = conv_w(st, nm(p + "conv1"));
        r.n2w = vec(st, nm(p + "norm2.weight"));
        r.n2b = vec(st, nm(p + "norm2.bias"));
        r.c2 = conv_w(st, nm(p + "conv2"));
        if (st.find(nm(p + "nin_shortcut.weight")) != nullptr) r.nin = conv_w(st, nm(p + "nin_shortcut"));
        if (r.c1.in % kGroups != 0 || r.c1.out % kGroups != 0)
            throw std::runtime_error("image engine (Flux2 VAE): " + p + ": channels not a multiple of 32");
        return r;
    };
    std::vector<F2Down> levels;
    for (int i = 0; st.find(nm(e + "down." + std::to_string(i) + ".block.0.conv1.weight")) != nullptr; ++i) {
        const std::string p = e + "down." + std::to_string(i) + ".";
        F2Down l;
        for (int j = 0; st.find(nm(p + "block." + std::to_string(j) + ".conv1.weight")) != nullptr; ++j)
            l.blocks.push_back(res(p + "block." + std::to_string(j) + "."));
        if (st.find(nm(p + "downsample.conv.weight")) != nullptr) l.down = conv_w(st, nm(p + "downsample.conv"));
        levels.push_back(std::move(l));
    }
    int downs = 0;
    for (const F2Down& l : levels) downs += l.down.out > 0;
    if (downs != 3) throw std::runtime_error("image engine (Flux2 VAE): the encoder does not halve the image 3 times");
    f2_emid_[0] = res(e + "mid.block_1.");
    const std::string a = e + "mid.attn_1.", qkv[3] = {nm(a + "q"), nm(a + "k"), nm(a + "v")};
    f2_eattn_ = attn_w(st, nm(a + "norm"), qkv, nm(a + "proj_out"), f2_emid_[0].c2.out);
    f2_ac_ = std::max(C_, f2_eattn_.qkv.in);   // MOSS-Image's decoder is narrower than its encoder
    f2_emid_[1] = res(e + "mid.block_2.");
    f2_enw_ = vec(st, nm(e + "norm_out.weight"));
    f2_enb_ = vec(st, nm(e + "norm_out.bias"));
    f2_eout_ = conv_w(st, nm(e + "conv_out"));
    if (f2_eout_.out != 2 * f2_lc_)
        throw std::runtime_error("image engine (Flux2 VAE): the encoder's moments are not twice the latent's width");
    f2_qc_ = Conv();
    if (st.find(nm(e + "quant_conv.weight")) != nullptr) f2_qc_ = conv_w(st, nm(e + "quant_conv"));
    f2_down_ = std::move(levels);
    // scratch a decode left is sized by the decoder's widths (its attention's channels, the norms' stats), which
    // MOSS-Image's encoder exceeds, and f2_grow keeps scratch whose sizes still fit: an img2img after a generation in
    // the same engine read past them, its image black (B70, 2026-10-29).  Taken again at the next use.
    if (f2_held_ > 0) drop_scratch();
}

// The encoder's stage sizes for an h x w latent's image (16 h x 16 w pixels), as f2_walk's
MfVae::F2Sizes MfVae::f2_enc_walk(int64_t h, int64_t w) const {
    F2Sizes z;
    int64_t pix = 256 * h * w, c = f2_ein_.out;
    z.pix = pix;
    z.most = pix * std::max<int64_t>(8, c);
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) z.most = std::max(z.most, pix * std::max(r.c1.in, r.c2.out));
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.down.out > 0) {
            pix /= 4;
            c = l.down.out;
            z.most = std::max(z.most, pix * c);
        }
    }
    z.mid = pix;
    z.most = std::max({z.most, pix * 3 * f2_eattn_.qkv.in, pix * 2 * f2_lc_});
    z.rows = std::max<int64_t>(1, std::min<int64_t>(z.mid, (int64_t) (16 << 20) / z.mid));
    z.c = c;
    return z;
}

void MfVae::f2_encode(const float* rgb, int64_t h, int64_t w, float* z) {
    if (!has_encoder()) throw std::runtime_error("image engine (Flux2 VAE): the encoder is not loaded");
    int64_t H = 16 * h, W = 16 * w;
    f2_grow(f2_enc_walk(h, w));
    f2_rgb8_16(rgb, H * W, a16_, s_);
    uint16_t* x = f2_x_;
    v_frame(f2_ein_, a16_, H, W, false, x, true, false);
    int64_t c = f2_ein_.out;
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) f2_resblock(r, x, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.down.out > 0) {
            uint16_t *y = nullptr, *unused = nullptr;
            f2_others(x, y, unused);
            v_frame(l.down, f2_in16(x, H * W * c, a16_), H, W, true, y, true, false);
            H /= 2;
            W /= 2;
            x = y;
            c = l.down.out;
        }
    }
    f2_resblock(f2_emid_[0], x, H, W);
    f2_attention(f2_eattn_, x, H, W);
    f2_resblock(f2_emid_[1], x, H, W);
    qi_group_norm16(x, vbf_, H * W, c, f2_enw_, f2_enb_, kEps, true, a16_, gn_part_, s_);
    uint16_t *m16 = nullptr, *t16 = nullptr;
    f2_others(x, m16, t16);
    float *m = reinterpret_cast<float*>(m16), *t = reinterpret_cast<float*>(t16);   // the moments in FP32 (in room)
    conv(f2_eout_, a16_, H, W, m);
    if (f2_qc_.out > 0) {
        kernels::f32_to_f16_bulk(m, a16_, H * W * f2_eout_.out, s_);
        conv(f2_qc_, a16_, H, W, t);
        m = t;
    }
    f2_patch(m, f2_mean_, f2_std_, h, w, f2_lc_, z, s_);
    gpu::stream_sync(s_);
}

// ---- the video form (SeedVR2's VAE) ----

// a slice's frame buffers, `values` floats each
void MfVae::v_grow(int64_t values, int64_t head, int64_t frame) {
    if (frame > vtmp_cap_) {
        release(allocs_, vtmp_);
        vtmp_ = alloc<float>(allocs_, frame, "image engine (Mage-VAE)");
        vtmp_cap_ = frame;
    }
    if (values <= vcap_ && head <= vhead_) return;
    values = std::max(values, vcap_);
    head = std::max(head, vhead_);
    for (void* p : {(void*) vx_, (void*) vy_, (void*) vt_, (void*) va16_base_}) release(allocs_, p);
    vx_ = alloc<uint16_t>(allocs_, values, "image engine (Mage-VAE)");
    vy_ = alloc<uint16_t>(allocs_, values, "image engine (Mage-VAE)");
    vt_ = alloc<uint16_t>(allocs_, values, "image engine (Mage-VAE)");
    va16_base_ = alloc<uint16_t>(allocs_, head + values, "image engine (Mage-VAE)");
    va16_ = va16_base_ + head;
    vcap_ = values;
    vhead_ = head;
}

// the two of vx, vy, vt that are not `x`
void MfVae::v_others(const uint16_t* x, uint16_t*& a, uint16_t*& b) const {
    uint16_t* all[3] = {vx_, vy_, vt_};
    int k = 0;
    for (uint16_t* p : all)
        if (p != x) (k++ == 0 ? a : b) = p;
}

// n activations as a convolution's FP16 input: themselves, or (BF16) made FP16 in va16_
const uint16_t* MfVae::v_in16(const uint16_t* x, int64_t n) {
    if (!vbf_) return x;
    bf16_to_f16(x, n, va16_, s_);
    return va16_;
}

// one frame's convolution of one tap into out: FP32, or the activations' 16 bits (o16) through a library that writes
// them where it takes the convolution, else through vtmp_
void MfVae::v_frame(const Conv& k, const uint16_t* in16, int64_t H, int64_t W, bool s2, void* out, bool o16,
                    bool add, uint16_t* side) {
    if (!o16) {
        if (s2) conv_s2(k, in16, H, W, static_cast<float*>(out), add);
        else conv(k, in16, H, W, static_cast<float*>(out), add);
        return;
    }
    if (!s2 && !vbf_ && conv16(k, in16, H, W, static_cast<uint16_t*>(out), add, side)) return;
    Conv kb = k;   // the bias in the pass that narrows the sums (the engine's own products add it in one more)
    kb.b = nullptr;
    if (s2) conv_s2(kb, in16, H, W, vtmp_, false);
    else conv(kb, in16, H, W, vtmp_, false);
    act_store(vtmp_, k.b, (s2 ? (H / 2) * (W / 2) : H * W) * k.out, k.out, vbf_, add, static_cast<uint16_t*>(out),
              s_);
}

// whether x's n values are all finite (one flag on the GPU, read back)
bool MfVae::finite(const float* x, int64_t n) {
    if (vbad_ == nullptr) vbad_ = alloc<int>(allocs_, 1, "image engine (Mage-VAE)");
    int* bad = vbad_;
    gpu::memset_async(bad, 0, sizeof(int), s_);
    q_of(s_).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        if (!sycl::isfinite(x[id[0]]))
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*bad).store(1);
    });
    int h = 0;
    gpu::stream_sync(s_);
    gpu::copy(&h, bad, sizeof(int));
    return h == 0;
}

// an output of FP16 activations past FP16's range: BF16's from now on (true: the caller runs the call again)
bool MfVae::overflowed(const float* x, int64_t n) {
    if (vbf_ || finite(x, n)) return false;
    vbf_ = true;
    std::fprintf(stderr, "xestrata-image: the VAE's activations passed FP16's range: BF16 from now on\n");
    return true;
}

void MfVae::video_reset() {
    for (auto& side : vcache_) {
        for (auto& [c, v] : side) release(allocs_, v.buf);
        side.clear();
    }
    for (auto& side : vdev_) {
        for (auto& [c, v] : side) release(allocs_, v.buf);
        side.clear();
    }
}

// with the host cache, a side's carried frames on the GPU leave it (back to their tiles' host copies, unless its last
// call said none reads them)
void MfVae::v_leave(int side) {
    auto& dev = vdev_[side];
    if (!vhost_ || dev.empty()) return;
    for (auto& [c, d] : dev)
        if (d.owner >= 0) {
            VCache& o = vcache_[side][VKey{c, d.owner}];
            if (vkept_[side]) gpu::copy_async(o.buf, d.buf, (size_t) (o.m * o.frame) * 2, s_);
            else o.n = 0;
        }
    gpu::stream_sync(s_);
    for (auto& [c, d] : dev) release(allocs_, d.buf);
    dev.clear();
}

// the encoder's (0) or the decoder's (1) turn: the other side's carried frames leave the GPU
void MfVae::v_enter(int side) {
    vside_ = side;
    vkept_[side] = vkeep_;
    v_leave(1 - side);
}

void MfVae::video_park() {
    v_leave(0);
    v_leave(1);
}

void MfVae::set_video_host_cache(bool on) {
    if (on == vhost_) return;
    gpu::stream_sync(s_);   // a host buffer may still be a copy's source or destination
    video_reset();
    vhost_ = on;
}

// A causal convolution over T frames in16 [T][H W in] (FP16), stride ts in time: its time taps over the frames the
// last slice left (kt - ts of them) and these, the first slice's first frame repeated kt - 1 times before them (the
// reference's extend_head); s2: 3 x 3 of stride 2 in space (conv_s2's).  out [To][H' W' out] (+= with `add`); what
// the next slice sees kept.  A convolution of one tap: each frame alone.
void MfVae::vconv(const Conv& c, const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t ts, bool s2,
                  void* out, bool o16, bool add, int64_t& To, uint16_t* side) {
    const int64_t fin = H * W * c.in, fout = (s2 ? (H / 2) * (W / 2) : H * W) * c.out;
    const auto at = [&](int64_t j) -> void* {
        return o16 ? (void*) (static_cast<uint16_t*>(out) + j * fout) : (void*) (static_cast<float*>(out) + j * fout);
    };
    if (c.kt <= 1 || c.taps == nullptr) {
        if (ts != 1) throw std::logic_error("image engine (VAE): a stride in time without time taps");
        for (int64_t t = 0; t < T; ++t)
            v_frame(c, in16 + t * fin, H, W, s2, at(t), o16, add, side != nullptr ? side + t * fout : nullptr);
        To = T;
        return;
    }
    VCache& vc = vcache_[vside_][VKey{&c, vtile_}];
    const int64_t m = c.kt - ts;
    VCache* d = vhost_ && m > 0 ? &vdev_[vside_][&c] : nullptr;   // the GPU's copy, shared by the tiles in turn
    if (vc.frame != fin || vc.m != m) {   // a new size (another tile's): nothing carried
        release(allocs_, vc.buf);
        vc = VCache();
        vc.buf = m <= 0 ? nullptr
                 : vhost_ ? alloc_pinned<uint16_t>(allocs_, m * fin) : alloc<uint16_t>(allocs_, m * fin, "image engine (Mage-VAE)");
        vc.frame = fin;
        vc.m = m;
        if (d != nullptr && d->owner == vtile_) d->owner = -1;
    }
    uint16_t* buf = vc.buf;   // where the carried frames are read and kept on the GPU
    if (d != nullptr) {       // this tile's frames in for its turn, the last owner's back first (unless none reads)
        if (d->owner != vtile_) {
            if (d->owner >= 0) {
                VCache& o = vcache_[vside_][VKey{&c, d->owner}];
                if (vkeep_) gpu::copy_async(o.buf, d->buf, (size_t) (o.m * o.frame) * 2, s_);
                else o.n = 0;
            }
            d->owner = -1;
            if (d->frame * d->m < m * fin) {
                release(allocs_, d->buf);
                d->buf = alloc<uint16_t>(allocs_, m * fin, "image engine (Mage-VAE)");
                d->frame = fin;
                d->m = m;
            }
            if (vc.n == m) gpu::copy_async(d->buf, vc.buf, (size_t) (m * fin) * 2, s_);
            d->owner = vtile_;
        }
        buf = d->buf;
    }
#if STRATA_CUDNN
    // the taps in one pass (cuDNN's 3D convolution): the m frames before the input, in the room before va16_, the
    // carried ones or the first frame repeated; the last m of them all then carried
    if (!s2 && ts == 1 && m > 0 && in16 == va16_ && m * fin <= vhead_ && !(o16 && vbf_)) {
        uint16_t* head = va16_ - m * fin;
        if (vc.n == m) gpu::copy_async(head, buf, (size_t) (m * fin) * 2, s_);
        else
            for (int64_t i = 0; i < m; ++i) gpu::copy_async(head + i * fin, in16, (size_t) fin * 2, s_);
        // FP16 out: the convolution alone (cuDNN's precompiled kernels: with the bias or the residual in its pass it
        // takes a kernel it compiles, a third slower on the RTX 4070, 2026-11-08), then the bias and the residual
        // in a pass of the engine's; where no room for the sum, all in cuDNN's pass
        uint16_t* y16 = !o16 ? nullptr : !add ? static_cast<uint16_t*>(out) : side;
        if (y16 != nullptr ? conv::cudnn3(core::Runtime::get().stream(s_), head, m + T, H, W, c.in, c.taps, c.out,
                                          (int) c.kt, (int) c.k, nullptr, false, y16, true)
                           : conv::cudnn3(core::Runtime::get().stream(s_), head, m + T, H, W, c.in, c.taps, c.out,
                                          (int) c.kt, (int) c.k, c.b, add, out, o16)) {
            if (y16 != nullptr) act_bias16(y16, c.b, T * fout, c.out, add, static_cast<uint16_t*>(out), s_);
            To = T;
            gpu::copy_async(buf, head + T * fin, (size_t) (m * fin) * 2, s_);
            vc.n = m;
            return;
        }
    }
#else
    (void) side;
#endif
    std::vector<const uint16_t*> ext;
    if (vc.n == m && m > 0) {
        for (int64_t i = 0; i < m; ++i) ext.push_back(buf + i * fin);
    } else {
        for (int64_t i = 0; i + 1 < c.kt; ++i) ext.push_back(in16);
    }
    for (int64_t t = 0; t < T; ++t) ext.push_back(in16 + t * fin);
    const auto L = (int64_t) ext.size();
    To = L >= c.kt ? (L - c.kt) / ts + 1 : 0;
    const int64_t per = c.out * c.k * c.k * c.in;
    if (vtapw_cap_ < c.kt * per) {   // each tap's weights apart, as a 2D convolution's
        release(allocs_, vtapw_);
        vtapw_ = alloc<uint16_t>(allocs_, c.kt * per, "image engine (Mage-VAE)");
        vtapw_cap_ = c.kt * per;
    }
    taps_apart(c.taps, c.out, c.kt, c.k * c.k * c.in, vtapw_, s_);
    for (int64_t j = 0; j < To; ++j) {   // the taps summed in FP32 (into vtmp_ for 16-bit activations)
        for (int64_t i = 0; i < c.kt; ++i) {
            Conv k = c;
            k.w = vtapw_ + i * per;
            k.b = i == 0 && !o16 ? c.b : nullptr;   // the bias once (with 16-bit activations as they are narrowed)
            v_frame(k, ext[(size_t) (j * ts + i)], H, W, s2, o16 ? (void*) vtmp_ : at(j), false,
                    o16 ? i > 0 : add || i > 0);
        }
        if (o16) act_store(vtmp_, c.b, fout, c.out, vbf_, add, static_cast<uint16_t*>(at(j)), s_);
    }
    if (m > 0) {   // the last m frames, each to a lower or the same place of the cache: in order, no overlap read
        for (int64_t i = 0; i < m; ++i) {
            const uint16_t* src = ext[(size_t) (L - m + i)];
            if (src != buf + i * fin) gpu::copy_async(buf + i * fin, src, (size_t) fin * 2, s_);
        }
        vc.n = m;   // (the host cache's copy back when another tile's turn comes)
    }
}

// a ResNet block over T frames x [T][H W c] (16-bit): as f2_resblock's, its convolutions causal in time
void MfVae::v_resblock(const F2Res& r, uint16_t*& x, int64_t T, int64_t H, int64_t W) {
    const int64_t HW = H * W;
    uint16_t *t = nullptr, *y = nullptr;
    v_others(x, t, y);
    for (int64_t f = 0; f < T; ++f)
        qi_group_norm16(x + f * HW * r.c1.in, vbf_, HW, r.c1.in, r.n1w, r.n1b, kEps, true, va16_ + f * HW * r.c1.in,
                        gn_part_, s_);
    int64_t To = 0;
    vconv(r.c1, va16_, T, H, W, 1, false, t, true, false, To);
    for (int64_t f = 0; f < T; ++f)
        qi_group_norm16(t + f * HW * r.c1.out, vbf_, HW, r.c1.out, r.n2w, r.n2b, kEps, true,
                        va16_ + f * HW * r.c1.out, gn_part_, s_);
    if (r.nin.out > 0) {
        vconv(r.c2, va16_, T, H, W, 1, false, y, true, false, To);
        vconv(r.nin, v_in16(x, T * HW * r.nin.in), T, H, W, 1, false, y, true, true, To, t);
        x = y;
    } else {
        vconv(r.c2, va16_, T, H, W, 1, false, x, true, true, To, y);
    }
}

// the middle's attention, each frame's own (the reference's "(b f) c h w")
void MfVae::v_attention(const Attn& a, uint16_t* x, int64_t T, int64_t H, int64_t W) {
    const int64_t f = H * W * a.qkv.in;
    for (int64_t t = 0; t < T; ++t) {
        gpu::copy_async(f2_x_, x + t * f, (size_t) f * 2, s_);
        f2_attention(a, f2_x_, H, W);
        gpu::copy_async(x + t * f, f2_x_, (size_t) f * 2, s_);
    }
}

uint64_t MfVae::video_scratch_bytes(int64_t h, int64_t w, int64_t frames) const {
    const int n = (int) f2_levels_.size();
    F2Sizes z = f2_walk(h, w, 0, n, true, true);
    if (!f2_down_.empty()) z = f2_max(z, f2_enc_walk(h, w));
    // the frame buffers (3 of the activations' 16 bits, 1 FP16 after 2 frames' room) of the largest stage's frames,
    // and one frame's own buffers (its FP32 sums among them)
    return (uint64_t) (frames * z.most) * (3 * 2 + 2) + (uint64_t) (2 * z.most) * 2 + f2_bytes(z);
}

int64_t MfVae::encode_video(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z) {
    if (!video_ || f2_ein_.taps == nullptr) throw std::logic_error("image engine (VAE): not opened as a video's");
    if (!has_encoder()) throw std::runtime_error("image engine (VAE): the encoder is not loaded");
    if (T < 1 || (first ? (T - 1) % 4 : T % 4) != 0)
        throw std::invalid_argument("image engine (VAE): a video slice of 1 + 4 k frames first, then 4 k");
    v_enter(0);
    if (first)   // the tile's own (another tile's carry on)
        for (auto& [k, v] : vcache_[0])
            if (k.tile == vtile_) v.n = 0;
    int64_t H = 16 * h, W = 16 * w;
    const F2Sizes fz = f2_enc_walk(h, w);
    f2_grow(fz);
    v_grow(T * fz.most, 2 * fz.most, fz.most);
    for (int64_t f = 0; f < T; ++f) f2_rgb8_16(rgb + f * 3 * H * W, H * W, va16_ + f * H * W * 8, s_);
    uint16_t* x = vx_;
    int64_t Tc = 0;
    vconv(f2_ein_, va16_, T, H, W, 1, false, x, true, false, Tc);
    int64_t c = f2_ein_.out;
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) v_resblock(r, x, Tc, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.down.out > 0) {
            uint16_t *y = nullptr, *unused = nullptr;
            v_others(x, y, unused);
            int64_t To = 0;
            vconv(l.down, v_in16(x, Tc * H * W * c), Tc, H, W, l.down.kt > 1 ? 2 : 1, true, y, true, false, To);
            H /= 2;
            W /= 2;
            x = y;
            Tc = To;
            c = l.down.out;
        }
    }
    v_resblock(f2_emid_[0], x, Tc, H, W);
    v_attention(f2_eattn_, x, Tc, H, W);
    v_resblock(f2_emid_[1], x, Tc, H, W);
    const int64_t HW = H * W;
    for (int64_t f = 0; f < Tc; ++f)
        qi_group_norm16(x + f * HW * c, vbf_, HW, c, f2_enw_, f2_enb_, kEps, true, va16_ + f * HW * c, gn_part_, s_);
    uint16_t *m16 = nullptr, *unused = nullptr;
    v_others(x, m16, unused);
    float* m = reinterpret_cast<float*>(m16);   // the moments in FP32 (2 Lc channels at the latents' size: in room)
    int64_t To = 0;
    vconv(f2_eout_, va16_, Tc, H, W, 1, false, m, false, false, To);
    if (f2_qc_.out > 0) throw std::runtime_error("image engine (VAE): a video's encoder with a quant_conv");
    for (int64_t f = 0; f < To; ++f)
        f2_patch(m + f * HW * f2_eout_.out, f2_mean_, f2_std_, h, w, f2_lc_, z + f * h * w * 4 * f2_lc_, s_);
    gpu::stream_sync(s_);
    return To;
}

int64_t MfVae::decode_video(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb) {
    if (!video_ || cin_.taps == nullptr) throw std::logic_error("image engine (VAE): not opened as a video's");
    if (n < 1) throw std::invalid_argument("image engine (VAE): no latents to decode");
    v_enter(1);
    if (first)
        for (auto& [k, v] : vcache_[1])
            if (k.tile == vtile_) v.n = 0;
    const int nl = (int) f2_levels_.size();
    const F2Sizes fz = f2_walk(h, w, 0, nl, true, true);
    f2_grow(fz);
    const int64_t frames = first ? 1 + 4 * (n - 1) : 4 * n;
    v_grow(std::max<int64_t>(frames, 2 * n) * fz.most, 2 * fz.most, fz.most);
    int64_t H = 2 * h, W = 2 * w;
    for (int64_t f = 0; f < n; ++f)
        f2_unpatch16(z + f * h * w * 4 * f2_lc_, f2_mean_, f2_std_, h, w, f2_lc_, va16_ + f * H * W * f2_lc_, s_);
    if (f2_pq_.out > 0) throw std::runtime_error("image engine (VAE): a video's decoder with a post_quant_conv");
    uint16_t* x = vx_;
    int64_t Tc = 0;
    vconv(cin_, va16_, n, H, W, 1, false, x, true, false, Tc);
    v_resblock(f2_mid_[0], x, Tc, H, W);
    v_attention(f2_attn_, x, Tc, H, W);
    v_resblock(f2_mid_[1], x, Tc, H, W);
    int64_t c = C_;
    for (const F2Level& l : f2_levels_) {
        for (const F2Res& r : l.blocks) v_resblock(r, x, Tc, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out == 0) continue;
        if (l.shuffle_t.out == 0)
            throw std::runtime_error("image engine (VAE): a video's upsampling without its 1 x 1 convolution");
        const int64_t Z = l.z, F = 4 * H * W * c;
        for (int64_t f = 0; f < Tc; ++f) {   // each frame's 1 x 1 convolution to Z frames of 2 H x 2 W
            const uint16_t* in16 = x + f * H * W * c;
            if (vbf_) {
                bf16_to_f16(in16, H * W * c, a16_, s_);
                in16 = a16_;
            }
            conv(l.shuffle_t, in16, H, W, vtmp_);
            f2_shuffle_t16(vtmp_, H, W, c, Z, va16_ + f * Z * F, s_);
        }
        int64_t Tz = Tc * Z;
        if (first && Z > 1) {   // the first slice's duplicated first frame (the reference's remove_head)
            for (int64_t f = 2; f < Tz; ++f) gpu::copy_async(va16_ + (f - 1) * F, va16_ + f * F, (size_t) F * 2, s_);
            --Tz;
        }
        H *= 2;
        W *= 2;
        uint16_t *y = nullptr, *unused = nullptr;
        v_others(x, y, unused);
        vconv(l.up, va16_, Tz, H, W, 1, false, y, true, false, Tc);
        x = y;
        c = l.up.out;
    }
    const int64_t HW = H * W;
    for (int64_t f = 0; f < Tc; ++f)
        qi_group_norm16(x + f * HW * c, vbf_, HW, c, gnw_, gnb_, kEps, true, va16_ + f * HW * c, gn_part_, s_);
    uint16_t *y16 = nullptr, *unused = nullptr;
    v_others(x, y16, unused);
    float* y = reinterpret_cast<float*>(y16);   // the RGB in FP32 (3 channels: in room)
    int64_t To = 0;
    vconv(cout_, va16_, Tc, H, W, 1, false, y, false, false, To);
    for (int64_t f = 0; f < To; ++f) hwc3_to_chw(y + f * HW * 3, HW, rgb + f * 3 * HW, s_);
    gpu::stream_sync(s_);
    return To;
}

namespace {
// The side of a video's tiles along n of at most `side`: as many tiles as that side takes, each as small as they then
// cover n with (the last no larger than the rest: a tile flush with the edge at the full side computed a fifth more of
// a 720p frame, 2026-11-08).  The side and the corners are the VAE tiles' shared ones (tile_sides.hpp) at the video's
// overlap; only the "of at most this" step is a video's own.
int64_t video_side(int64_t n, int64_t side) {
    if (n <= side) return n;
    return tile_side(n, (int64_t) tile_positions(n, side, kVOverlap).size(), kVOverlap);
}
}  // namespace

int64_t MfVae::video_tile_side(int64_t n, int64_t tt) { return tt <= 0 ? n : video_side(n, tt); }

int64_t MfVae::video_tiles(int64_t n, int64_t tt) {
    return tt <= 0 || n <= tt ? 1 : (int64_t) tile_positions(n, video_side(n, tt), kVOverlap).size();
}

uint64_t MfVae::video_carried_bytes(int64_t h, int64_t w, int side) const {
    // a causal convolution carries kt - stride frames of its input: the encoder's at its level's size, the decoder's
    // alike; a bound: every one of 2 frames at the largest of its input's widths a level
    uint64_t b = 0;
    int64_t pix = 256 * h * w;
    const auto carry = [&](const Conv& c) {
        if (c.kt > 1) b += (uint64_t) (2 * pix * c.in) * 2;
    };
    carry(f2_ein_);
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) carry(r.c1), carry(r.c2), carry(r.nin);
        carry(l.down);
        if (l.down.out > 0) pix /= 4;
    }
    for (const F2Res& r : f2_emid_) carry(r.c1), carry(r.c2);
    carry(f2_eout_);
    if (side == 0) return b;
    if (side == 1) b = 0;
    pix = 4 * h * w;
    carry(cin_);
    for (const F2Res& r : f2_mid_) carry(r.c1), carry(r.c2);
    for (const F2Level& l : f2_levels_) {
        for (const F2Res& r : l.blocks) carry(r.c1), carry(r.c2), carry(r.nin);
        if (l.up.out > 0) pix *= 4;
        carry(l.up);
    }
    carry(cout_);
    return b;
}

// the clip's first call again in BF16 where its FP16 activations overflowed; a later one cannot be (its slices
// before it carried in FP16), so the clip is to be run again
int64_t MfVae::encode_video_tiled(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z, int64_t tt) {
    const int64_t k = v_encode_tiled(rgb, T, h, w, first, z, tt);
    if (!overflowed(z, k * h * w * 4 * f2_lc_)) return k;
    if (!first) throw std::runtime_error("image engine (VAE): the activations passed FP16's range: run the clip again");
    return v_encode_tiled(rgb, T, h, w, first, z, tt);
}

int64_t MfVae::decode_video_tiled(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb, int64_t tt) {
    const int64_t T = v_decode_tiled(z, n, h, w, first, rgb, tt);
    if (!overflowed(rgb, T * 3 * h * w * kPix)) return T;
    if (!first) throw std::runtime_error("image engine (VAE): the activations passed FP16's range: run the clip again");
    return v_decode_tiled(z, n, h, w, first, rgb, tt);
}

int64_t MfVae::v_encode_tiled(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z, int64_t tt) {
    // a slice at a time: 1 + 4 (vslice_ - 1) frames first, 4 vslice_ after (the work video_scratch_bytes(..,
    // 4 vslice_) bounds)
    const auto sliced = [&](const float* x, int64_t hh, int64_t ww, float* zz) {
        const int64_t plane = 3 * hh * ww * kPix, lat = hh * ww * 4 * f2_lc_;
        int64_t got = 0;
        for (int64_t f = 0; f < T;) {
            const bool head = first && f == 0;
            const int64_t n = std::min(T - f, head ? 1 + 4 * (vslice_ - 1) : 4 * vslice_);
            got += encode_video(x + f * plane, n, hh, ww, head, zz + got * lat);
            f += n;
        }
        return got;
    };
    if (tt <= 0 || (tt >= h && tt >= w)) {
        vtile_ = 0;
        return sliced(rgb, h, w, z);
    }
    const int64_t th = video_side(h, tt), tw = video_side(w, tt), H = h * kPatch, W = w * kPatch, V = 4 * f2_lc_;
    const int64_t k = first ? (T - 1) / 4 + 1 : T / 4, plane = th * tw * kPix;
    TempBuffers tmp(s_);
    float* acc = alloc<float>(tmp, k * (V + 1) * h * w, "image engine (Mage-VAE)");
    float* trgb = alloc<float>(tmp, T * 3 * plane, "image engine (Mage-VAE)");
    float* tz = alloc<float>(tmp, k * th * tw * V, "image engine (Mage-VAE)");
    gpu::memset_async(acc, 0, (size_t) (k * (V + 1) * h * w) * 4, s_);
    const auto ys = tile_positions(h, th, kVOverlap), xs = tile_positions(w, tw, kVOverlap);
    int tile = 0;
    for (const int64_t y0 : ys)
        for (const int64_t x0 : xs) {
            vtile_ = tile++;
            for (int64_t f = 0; f < T; ++f)
                for (int64_t c = 0; c < 3; ++c)
                    tile_crop(rgb + (f * 3 + c) * H * W, 1, TilePlanes::Planes, H, W, y0 * kPatch, x0 * kPatch,
                              th * kPatch, tw * kPatch, trgb + (f * 3 + c) * plane, s_);
            const int64_t got = sliced(trgb, th, tw, tz);
            for (int64_t j = 0; j < got; ++j) {
                float* aj = acc + j * (V + 1) * h * w;
                tile_blend(tz + j * th * tw * V, V, TilePlanes::Interleaved, th, tw, y0, x0, h, w, kVOverlap,
                           TileRamp::Centre, y0 > 0, y0 + th < h, x0 > 0, x0 + tw < w, aj, aj + V * h * w, s_);
            }
        }
    vtile_ = 0;
    for (int64_t j = 0; j < k; ++j) {
        const float* aj = acc + j * (V + 1) * h * w;
        tile_finish(aj, V, TilePlanes::Interleaved, h * w, aj + V * h * w, z + j * h * w * V, s_);
    }
    gpu::stream_sync(s_);
    return k;
}

int64_t MfVae::v_decode_tiled(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb, int64_t tt) {
    // vslice_ latents at a time (the work video_scratch_bytes(.., 4 vslice_) bounds)
    const auto sliced = [&](const float* zz, int64_t hh, int64_t ww, float* out) {
        const int64_t lat = hh * ww * 4 * f2_lc_, frame = 3 * hh * ww * kPix;
        int64_t got = 0;
        for (int64_t j = 0; j < n; j += vslice_)
            got += decode_video(zz + j * lat, std::min(vslice_, n - j), hh, ww, first && j == 0, out + got * frame);
        return got;
    };
    if (tt <= 0 || (tt >= h && tt >= w)) {
        vtile_ = 0;
        return sliced(z, h, w, rgb);
    }
    const int64_t th = video_side(h, tt), tw = video_side(w, tt), H = h * kPatch, W = w * kPatch, V = 4 * f2_lc_;
    const int64_t frames = first ? 1 + 4 * (n - 1) : 4 * n, TH = th * kPatch, TW = tw * kPatch;
    TempBuffers tmp(s_);
    float* acc = alloc<float>(tmp, frames * 4 * H * W, "image engine (Mage-VAE)");
    float* tz = alloc<float>(tmp, n * th * tw * V, "image engine (Mage-VAE)");
    float* trgb = alloc<float>(tmp, frames * 3 * TH * TW, "image engine (Mage-VAE)");
    gpu::memset_async(acc, 0, (size_t) (frames * 4 * H * W) * 4, s_);
    const auto ys = tile_positions(h, th, kVOverlap), xs = tile_positions(w, tw, kVOverlap);
    int tile = 0;
    for (const int64_t y0 : ys)
        for (const int64_t x0 : xs) {
            vtile_ = tile++;
            for (int64_t j = 0; j < n; ++j)
                tile_crop(z + j * h * w * V, V, TilePlanes::Interleaved, h, w, y0, x0, th, tw, tz + j * th * tw * V,
                          s_);
            const int64_t got = sliced(tz, th, tw, trgb);
            for (int64_t f = 0; f < got; ++f) {
                float* af = acc + f * 4 * H * W;
                tile_blend(trgb + f * 3 * TH * TW, 3, TilePlanes::Planes, TH, TW, y0 * kPatch, x0 * kPatch, H, W,
                           kVOverlap * kPatch, TileRamp::Centre, y0 > 0, y0 + th < h, x0 > 0, x0 + tw < w, af,
                           af + 3 * H * W, s_);
            }
        }
    vtile_ = 0;
    for (int64_t f = 0; f < frames; ++f)
        tile_finish(acc + f * 4 * H * W, 3, TilePlanes::Planes, H * W, acc + f * 4 * H * W + 3 * H * W,
                    rgb + f * 3 * H * W, s_);
    gpu::stream_sync(s_);
    return frames;
}

}  // namespace strata::image
