// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_vae.cpp - see include/strata/image/mf_vae.hpp.

#include "strata/image/mf_vae.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "f16_gpu.hpp"

#if STRATA_ONEDNN || STRATA_CUDNN
#include "qi_conv.hpp"
#endif

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

// The kernels' helpers in a namespace of this file's own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1), so a helper of the same
// name and signature in another image file (qi_vae.cpp's to_rgba) gave its kernel to both.
namespace mf_vae_detail {

using f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using f16_gpu::f32_from_f16;

constexpr int64_t kPatch = 16, kPix = kPatch * kPatch;   // a latent position's pixels
constexpr int64_t kWin = 32, kWinTok = kWin * kWin;       // the CoD decoder's attention windows
constexpr int64_t kGroups = 32;                           // its group norms
constexpr int64_t kF2Overlap = 4, kF2TileMin = 16;        // the Flux2 VAE's tiles: tokens shared, the least side
constexpr int64_t kX = 32;                                // the per-pixel MLP's width
constexpr int kResBlocks = 3;
constexpr float kEps = 1e-6f;

// the per-pixel MLP's weights in one FP32 buffer (offsets in floats)
constexpr int64_t kEmbW = 0, kEmbB = kEmbW + kX * 99, kInW = kEmbB + kX, kInB = kInW + kX * kX, kRes0 = kInB + kX;
constexpr int64_t kAdaW = 0, kAdaB = kAdaW + 3 * kX * kX, kLnW = kAdaB + 3 * kX, kLnB = kLnW + kX, kM1W = kLnB + kX,
                  kM1B = kM1W + kX * kX, kM2W = kM1B + kX, kM2B = kM2W + kX * kX, kResSize = kM2B + kX;
constexpr int64_t kFnW = kRes0 + kResBlocks * kResSize, kFlW = kFnW + kX, kFlB = kFlW + 3 * kX, kPixW = kFlB + 3;

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

template<class T> T* alloc(std::vector<void*>& owned, int64_t n) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Mage-VAE): GPU memory: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

float silu(float x) { return x / (1.f + std::exp(-x)); }

// col16[r][(dy * 3 + dx) * C + c] = in16[y + dy - 1][x + dx - 1][c] (0 outside), pixel p0 + r = y * W + x; C a
// multiple of 8
void im2col3(const uint16_t* in16, int64_t H, int64_t W, int64_t C, int64_t p0, int64_t rows, uint16_t* col16,
             void* s) {
    const int64_t C8 = C / 8;
    q_of(s).parallel_for(sycl::range<1>((size_t) (rows * 9 * C8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t rk = i / C8, r = rk / 9, kk = rk - r * 9, c8 = (i - rk * C8) * 8;
        const int64_t p = p0 + r, y = p / W + kk / 3 - 1, x = p % W + kk % 3 - 1;
        using V = sycl::vec<uint16_t, 8>;
        V v(0);
        if (y >= 0 && y < H && x >= 0 && x < W) v = *reinterpret_cast<const V*>(in16 + (y * W + x) * C + c8);
        *reinterpret_cast<V*>(col16 + rk * C + c8) = v;
    });
}

// the Flux2 VAE encoder's downsampling: col16[r][(dy * 3 + dx) * C + c] = in16[2 y + dy][2 x + dx][c] (0 past the
// bottom and right edges: the 0, 1, 0, 1 padding before the 3 x 3 convolution of stride 2), output pixel p0 + r =
// y * (W / 2) + x
void im2col3_s2(const uint16_t* in16, int64_t H, int64_t W, int64_t C, int64_t p0, int64_t rows, uint16_t* col16,
                void* s) {
    const int64_t C8 = C / 8, Wo = W / 2;
    q_of(s).parallel_for(sycl::range<1>((size_t) (rows * 9 * C8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t rk = i / C8, r = rk / 9, kk = rk - r * 9, c8 = (i - rk * C8) * 8;
        const int64_t p = p0 + r, y = 2 * (p / Wo) + kk / 3, x = 2 * (p % Wo) + kk % 3;
        using V = sycl::vec<uint16_t, 8>;
        V v(0);
        if (y < H && x < W) v = *reinterpret_cast<const V*>(in16 + (y * W + x) * C + c8);
        *reinterpret_cast<V*>(col16 + rk * C + c8) = v;
    });
}

// The group norm's statistics in one pass over x: chunks of kGnChunk pixels, a work-group a chunk whose lanes read
// whole rows of channels (a lane one channel, R rows at a time: C R lanes), each lane its values' mean and squared
// deviations, merged within each group (Chan's): part[(k kGroups + g) 3 + 0, 1, 2] = the count, the mean
// and the squared deviations of chunk k's group g.  (A work-group a chunk and group, reading its C / 32 channels of
// each pixel, and a second pass for the squares took MOSS-Image's VAE 64 ms of its 170 ms decode on the B70.)
constexpr int64_t kGnChunk = 256;
int64_t gn_chunks(int64_t HW) { return (HW + kGnChunk - 1) / kGnChunk; }
int64_t gn_rows(int64_t C) { return sycl::max<int64_t>(1, 256 / C); }

void gn_partial(const float* x, int64_t HW, int64_t C, float* part, void* s) {
    const int64_t cg = C / kGroups, nk = gn_chunks(HW), R = gn_rows(C), local = C * R;
    q_of(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ln(sycl::range<1>((size_t) local), h), lm(sycl::range<1>((size_t) local), h),
            lq(sycl::range<1>((size_t) local), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nk * local)), sycl::range<1>((size_t) local)),
                       [=](sycl::nd_item<1> it) {
            const auto k = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t r = lane / C, c = lane - r * C, p0 = k * kGnChunk, p1 = sycl::min(p0 + kGnChunk, HW);
            // sums about the lane's first value (no division a value; the shift keeps them accurate in FP32)
            const float k0 = p0 + r < p1 ? x[(p0 + r) * C + c] : 0.f;
            float n = 0.f, s1 = 0.f, s2 = 0.f;
            for (int64_t p = p0 + r; p < p1; p += R) {
                const float d = x[p * C + c] - k0;
                n += 1.f;
                s1 += d;
                s2 += d * d;
            }
            ln[lane] = n;
            lm[lane] = n > 0.f ? k0 + s1 / n : 0.f;
            lq[lane] = n > 0.f ? s2 - s1 * s1 / n : 0.f;
            sycl::group_barrier(it.get_group());
            if (lane < kGroups) {
                float N = 0.f, M = 0.f, Q = 0.f;
                for (int64_t rr = 0; rr < R; ++rr)
                    for (int64_t j = 0; j < cg; ++j) {
                        const int64_t e = rr * C + lane * cg + j;
                        const float nb = ln[e];
                        if (nb == 0.f) continue;
                        const float t = N + nb, d = lm[e] - M;
                        M += d * nb / t;
                        Q += lq[e] + d * d * N * nb / t;
                        N = t;
                    }
                float* o = part + (k * kGroups + lane) * 3;
                o[0] = N;
                o[1] = M;
                o[2] = Q;
            }
        });
    });
}

// stats[2 g] = group g's mean over every pixel, stats[2 g + 1] = 1 / its std, gn_partial's chunks merged (Chan's): a
// work-group a group, each lane a stride of the chunks, then the lanes in a tree
void gn_stats(const float* part, int64_t HW, int64_t C, float* stats, void* s) {
    (void) C;
    constexpr int64_t local = 256;
    const int64_t nk = gn_chunks(HW);
    q_of(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ln(sycl::range<1>((size_t) local), h), lm(sycl::range<1>((size_t) local), h),
            lq(sycl::range<1>((size_t) local), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (kGroups * local)), sycl::range<1>((size_t) local)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const auto merge = [](float& N, float& M, float& Q, float nb, float mb, float qb) {
                if (nb == 0.f) return;
                const float t = N + nb, d = mb - M;
                M += d * nb / t;
                Q += qb + d * d * N * nb / t;
                N = t;
            };
            float N = 0.f, M = 0.f, Q = 0.f;
            for (int64_t k = lane; k < nk; k += local) {
                const float* e = part + (k * kGroups + g) * 3;
                merge(N, M, Q, e[0], e[1], e[2]);
            }
            ln[lane] = N;
            lm[lane] = M;
            lq[lane] = Q;
            for (int64_t half = local / 2; half > 0; half /= 2) {
                sycl::group_barrier(it.get_group());
                if (lane < half) {
                    merge(N, M, Q, ln[lane + half], lm[lane + half], lq[lane + half]);
                    ln[lane] = N;
                    lm[lane] = M;
                    lq[lane] = Q;
                }
            }
            if (lane == 0) {
                stats[2 * g] = M;
                stats[2 * g + 1] = 1.f / sycl::sqrt(Q / N + kEps);
            }
        });
    });
}

void gn_apply16(const float* x, const float* stats, const float* w, const float* b, int64_t HW, int64_t C, bool silu_on,
                uint16_t* y16, void* s) {
    // 8 values a work-item, in one load and one store (C is a multiple of 32): a value each took 30 ms of MOSS-Image's
    // decode on the B70
    const int64_t cg = C / kGroups;
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * C / 8)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]) * 8, c0 = i % C;
        const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8>*>(x + i);
        sycl::vec<uint16_t, 8> o;
        for (int j = 0; j < 8; ++j) {
            const int64_t c = c0 + j, g = c / cg;
            float u = (v[j] - stats[2 * g]) * stats[2 * g + 1] * w[c] + b[c];
            if (silu_on) u = u / (1.f + sycl::exp(-u));
            o[j] = f16_from_f32(u);
        }
        *reinterpret_cast<sycl::vec<uint16_t, 8>*>(y16 + i) = o;
    });
}

// a window's q, k (rows: its 32 x 32 positions, the latent's edge replicated past it) and v transposed ([C][1024]),
// from qkv [HW][3 C]
void win_gather16(const float* qkv, int64_t H, int64_t W, int64_t C, int64_t wy, int64_t wx, uint16_t* q16,
                  uint16_t* k16, uint16_t* vt16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (kWinTok * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        const int64_t y = sycl::min(wy * kWin + t / kWin, H - 1), x = sycl::min(wx * kWin + t % kWin, W - 1);
        const float* r = qkv + (y * W + x) * 3 * C;
        q16[i] = f16_from_f32(r[c]);
        k16[i] = f16_from_f32(r[C + c]);
        vt16[c * kWinTok + t] = f16_from_f32(r[2 * C + c]);
    });
}

// p16[r] = softmax(scale sc[r]) over a row of n
void softmax16(const float* sc, int64_t rows, int64_t n, float scale, uint16_t* p16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> id) {
        const auto r = static_cast<int64_t>(id[0]);
        const float* row = sc + r * n;
        float m = -INFINITY;
        for (int64_t j = 0; j < n; ++j) m = sycl::fmax(m, row[j] * scale);
        float sum = 0.f;
        for (int64_t j = 0; j < n; ++j) sum += sycl::exp(row[j] * scale - m);
        const float inv = 1.f / sum;
        for (int64_t j = 0; j < n; ++j) p16[r * n + j] = f16_from_f32(sycl::exp(row[j] * scale - m) * inv);
    });
}

// out[y][x] = o's row for (y, x) of a window, for the window's positions inside the latent
void win_scatter(const float* o, int64_t H, int64_t W, int64_t C, int64_t wy, int64_t wx, float* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (kWinTok * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        const int64_t y = wy * kWin + t / kWin, x = wx * kWin + t % kWin;
        if (y < H && x < W) out[(y * W + x) * C + c] = o[i];
    });
}

// y16[p] = LayerNorm(x[p]) over the channels (no affine) * (1 + scale) + shift: a work-group a pixel, its lanes over
// the channels (a work-item a pixel, walking its channels three times, left the 4096 pixels of a 1024 x 1024 image's
// latent too little to run at once)
void ln_mod16(const float* x, const float* shift, const float* scale, int64_t HW, int64_t C, uint16_t* y16, void* s) {
    constexpr size_t local = 128;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) HW * local), sycl::range<1>(local)),
                         [=](sycl::nd_item<1> it) {
        const auto p = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const float* r = x + p * C;
        float m = 0.f;
        for (int64_t c = lane; c < C; c += (int64_t) local) m += r[c];
        m = sycl::reduce_over_group(it.get_group(), m, sycl::plus<float>()) / (float) C;
        float v = 0.f;
        for (int64_t c = lane; c < C; c += (int64_t) local) v += (r[c] - m) * (r[c] - m);
        const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / (float) C +
                                           kEps);
        for (int64_t c = lane; c < C; c += (int64_t) local)
            y16[p * C + c] = f16_from_f32((r[c] - m) * inv * (1.f + scale[c]) + shift[c]);
    });
}

// out = GELU(depthwise 3 x 3 convolution of t (zero outside) + bias), the exact (erf) GELU
void dw3x3_gelu(const float* t, const float* w, const float* b, int64_t H, int64_t W, int64_t C, float* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C, y = p / W, x = p % W;
        float a = b[c];
        for (int64_t dy = -1; dy <= 1; ++dy)
            for (int64_t dx = -1; dx <= 1; ++dx) {
                const int64_t yy = y + dy, xx = x + dx;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                    a += w[c * 9 + (dy + 1) * 3 + dx + 1] * t[(yy * W + xx) * C + c];
            }
        out[i] = 0.5f * a * (1.f + sycl::erf(a * 0.70710678118654752f));
    });
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

// x += gate * u (per channel)
void gate_add(float* x, const float* u, const float* gate, int64_t HW, int64_t C, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] += gate[i % C] * u[i];
    });
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
// the batch norm undone and each token's 128 values spread over 2 x 2 positions of 32 channels
void f2_unpatch16(const float* z, const float* mean, const float* sd, int64_t h, int64_t w, uint16_t* out16, void* s) {
    const int64_t W = 2 * w;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * h * w * 32)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 32, c = i - p * 32, Y = p / W, X = p - Y * W;
        const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
        out16[i] = f16_from_f32(z[((Y / 2) * w + X / 2) * 128 + k] * sd[k] + mean[k]);
    });
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
// index) over the posterior's mean, the first 32 of the moments m [2 h][2 w][64]
void f2_patch(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, float* z, void* s) {
    const int64_t W = 2 * w;
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * h * w * 32)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 32, c = i - p * 32, Y = p / W, X = p - Y * W;
        const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
        z[((Y / 2) * w + X / 2) * 128 + k] = (m[p * 64 + c] - mean[k]) / sd[k];
    });
}

// out16 [2 H][2 W][C] = x [H][W][C] repeated (nearest), FP16
void up2_16(const float* x, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C, Y = p / (2 * W), X = p - Y * 2 * W;
        out16[i] = f16_from_f32(x[((Y / 2) * W + X / 2) * C + c]);
    });
}

// q16 [T][C], k16 [T][C] and v transposed vt16 [C][T] from qkv [T][3 C]
void qkv_split16(const float* qkv, int64_t T, int64_t C, uint16_t* q16, uint16_t* k16, uint16_t* vt16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        q16[i] = f16_from_f32(qkv[t * 3 * C + c]);
        k16[i] = f16_from_f32(qkv[t * 3 * C + C + c]);
        vt16[c * T + t] = f16_from_f32(qkv[t * 3 * C + 2 * C + c]);
    });
}

// p16[r] = softmax(scale sc[r]) over a row of n: a work-group a row (the attention's rows are long, and few at once)
void softmax_wg16(const float* sc, int64_t rows, int64_t n, float scale, uint16_t* p16, void* s) {
    constexpr int64_t local = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * local)), sycl::range<1>((size_t) local)),
                         [=](sycl::nd_item<1> it) {
        const auto r = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const float* row = sc + r * n;
        float m = -INFINITY;
        for (int64_t j = lane; j < n; j += local) m = sycl::fmax(m, row[j] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.f;
        for (int64_t j = lane; j < n; j += local) sum += sycl::exp(row[j] * scale - m);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.f / sum;
        for (int64_t j = lane; j < n; j += local) p16[r * n + j] = f16_from_f32(sycl::exp(row[j] * scale - m) * inv);
    });
}

// a tile of x [..][w][C]: xt [th][tw][C] = x's rows y0 .. y0 + th, columns x0 .. x0 + tw
void tile_gather(const float* x, int64_t w, int64_t C, int64_t y0, int64_t x0, int64_t th, int64_t tw, float* xt,
                 void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (th * tw * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C, y = p / tw, xx = p - y * tw;
        xt[i] = x[((y0 + y) * w + x0 + xx) * C + c];
    });
}

// a tile's image trgb [3][TH][TW] (at Y0, X0 of the image [3][H][W]) into acc (3 planes, then the weights' plane),
// weighted by a linear ramp over `ramp` pixels on each side that another tile shares (top, bottom, left, right)
void tile_blend(const float* trgb, int64_t TH, int64_t TW, int64_t Y0, int64_t X0, int64_t H, int64_t W, int64_t ramp,
                bool top, bool bottom, bool left, bool right, float* acc, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (TH * TW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t y = i / TW, x = i - y * TW;
        const float r = (float) ramp;
        float wy = 1.f, wx = 1.f;
        if (top) wy = sycl::fmin(wy, ((float) y + 0.5f) / r);
        if (bottom) wy = sycl::fmin(wy, ((float) (TH - y) - 0.5f) / r);
        if (left) wx = sycl::fmin(wx, ((float) x + 0.5f) / r);
        if (right) wx = sycl::fmin(wx, ((float) (TW - x) - 0.5f) / r);
        const float wt = wy * wx;
        const int64_t o = (Y0 + y) * W + X0 + x, HW = H * W;
        for (int64_t c = 0; c < 3; ++c) acc[c * HW + o] += wt * trgb[c * TH * TW + i];
        acc[3 * HW + o] += wt;
    });
}

// rgb [3][HW] = acc's planes over its weights
void tile_finish(const float* acc, int64_t HW, float* rgb, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (3 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        rgb[i] = acc[i] / acc[3 * HW + i % HW];
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

void to_rgba(const float* rgb, int64_t HW, float* rgba, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW;
        rgba[i] = c == 3 ? 1.f : sycl::clamp((rgb[i] + 1.f) * 0.5f, 0.f, 1.f);
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
    float* d = alloc<float>(allocs_, (int64_t) v.size());
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

// a convolution's (or linear layer's) weights as FP16 [out][k][k][in], input columns col0 .. col0 + cols of them
MfVae::Conv MfVae::conv_w(SafeTensors& st, const std::string& name, bool bias, int64_t col0, int64_t cols) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || (t->shape.size() != 4 && t->shape.size() != 2))
        throw std::runtime_error("image engine (Mage-VAE): no " + name);
    Conv c;
    c.out = t->shape[0];
    const int64_t in = t->shape[1];
    c.in = cols < 0 ? in - col0 : cols;
    c.k = t->shape.size() == 4 ? t->shape[2] : 1;
    if (c.in % 8 != 0) throw std::runtime_error("image engine (Mage-VAE): " + name + ": input channels not 8s");
    weight_bytes_ += (uint64_t) (c.out * c.k * c.k * c.in) * 2;
    if (!dry_) {
        const auto w = st.f32(name + ".weight");   // [out][in][k][k]
        std::vector<uint16_t> h((size_t) (c.out * c.k * c.k * c.in));
        for (int64_t o = 0; o < c.out; ++o)
            for (int64_t i = 0; i < c.in; ++i)
                for (int64_t dy = 0; dy < c.k; ++dy)
                    for (int64_t dx = 0; dx < c.k; ++dx)
                        h[(size_t) (((o * c.k + dy) * c.k + dx) * c.in + i)] =
                            kernels::f16_from_f32(w[(size_t) (((o * in + col0 + i) * c.k + dy) * c.k + dx)]);
        uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
        gpu::copy(d, h.data(), h.size() * 2);
        c.w = d;
    }
    if (bias) c.b = vec(st, name + ".bias");
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
        uint16_t* dw = alloc<uint16_t>(allocs_, (int64_t) h.size());
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
        if (st.find("bn.running_mean") != nullptr && st.find("decoder.conv_in.weight") != nullptr) {
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
           (uint64_t) n * 9 * (uint64_t) C_ * 2 +                            // the unfolded neighbourhoods
           (uint64_t) (3 * kWinTok * C_ + kWinTok * kWinTok) * 2 + (uint64_t) kWinTok * kWinTok * 4 +
           (uint64_t) band * 2 * kX * kPix * 4;
}

void MfVae::grow(int64_t h, int64_t w) {
    const int64_t n = h * w;
    if (n <= cap_hw_) return;
    drop_scratch();
    x_ = alloc<float>(allocs_, n * C_);
    cond_ = alloc<float>(allocs_, n * C_);
    s_state_ = alloc<float>(allocs_, n * C_);
    t0_ = alloc<float>(allocs_, n * F_);
    t1_ = alloc<float>(allocs_, n * F_);
    t2_ = alloc<float>(allocs_, std::max(n * F_, n * C_ + kWinTok * C_));   // the attention's output and a window's
    gn_part_ = alloc<float>(allocs_, n * kGroups * 2);
    vec_ = alloc<float>(allocs_, 4 * F_);
    a16_ = alloc<uint16_t>(allocs_, n * std::max(F_, 3 * C_));
    col_cap_ = n * 9 * C_;
    col16_ = alloc<uint16_t>(allocs_, col_cap_);
    w16_ = alloc<uint16_t>(allocs_, 3 * kWinTok * C_ + kWinTok * kWinTok);
    scores_ = alloc<float>(allocs_, kWinTok * kWinTok);
    cap_band_ = std::min<int64_t>(n, 1024);
    yxb_ = alloc<float>(allocs_, cap_band_ * kX * kPix);
    ceb_ = alloc<float>(allocs_, cap_band_ * kX * kPix);
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
    f2_x_ = f2_y_ = f2_t_ = f2_q_ = nullptr;
    f2_qk16_ = f2_p16_ = nullptr;
    f2_cap_ = F2Sizes();
    f2_held_ = 0;
}

void MfVae::conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add) {
    const int64_t HW = H * W;
#if STRATA_ONEDNN
    // oneDNN where it takes them (Intel GPUs), the bias and the residual within its pass: the 3 x 3 ones, and the 1 x 1
    // ones that add to a residual (the others gained nothing over the library's product on the B70, 2026-11-02)
    const int p = (int) c.k / 2;
    if ((c.k == 3 || add) &&
        conv::onednn(core::Runtime::get().stream(s_), in16, H, W, c.in, c.w, c.out, (int) c.k, 1, p, p, p, p, c.b, add,
                     out))
        return;
#endif
#if STRATA_CUDNN
    if ((c.k == 3 || add) && conv::cudnn(core::Runtime::get().stream(s_), in16, H, W, c.in, c.w, c.out, (int) c.k, c.b, add, out)) return;
#endif
    if (c.k == 1) {
        gemm_.f16(in16, c.w, out, HW, c.out, c.in, 0, false, add);
    } else {
        const int64_t K = 9 * c.in, rows = std::max<int64_t>(1, (f2_ ? f2_band_ : col_cap_) / K);
        for (int64_t p0 = 0; p0 < HW; p0 += rows) {
            const int64_t n = std::min(rows, HW - p0);
            im2col3(in16, H, W, c.in, p0, n, col16_, s_);
            gemm_.f16(col16_, c.w, out + p0 * c.out, n, c.out, K, 0, false, add);
        }
    }
    if (c.b != nullptr) qi_bias_add(out, c.b, HW, c.out, s_);
}

void MfVae::conv_s2(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out) {
    const int64_t HWo = (H / 2) * (W / 2), K = 9 * c.in, rows = std::max<int64_t>(1, (f2_ ? f2_band_ : col_cap_) / K);
    for (int64_t p0 = 0; p0 < HWo; p0 += rows) {
        const int64_t n = std::min(rows, HWo - p0);
        im2col3_s2(in16, H, W, c.in, p0, n, col16_, s_);
        gemm_.f16(col16_, c.w, out + p0 * c.out, n, c.out, K);
    }
    if (c.b != nullptr) qi_bias_add(out, c.b, HWo, c.out, s_);
}

void MfVae::gn_silu16(const float* x, const float* w, const float* b, int64_t HW, int64_t C, bool silu_on,
                      uint16_t* y16) {
    gn_partial(x, HW, C, gn_part_, s_);
    gn_stats(gn_part_, HW, C, vec_, s_);
    gn_apply16(x, vec_, w, b, HW, C, silu_on, y16, s_);
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
    gn_silu16(x, a.nw, a.nb, HW, C_, false, a16_);
    conv(a.qkv, a16_, H, W, qkv);
    uint16_t *q16 = w16_, *k16 = w16_ + kWinTok * C_, *vt16 = w16_ + 2 * kWinTok * C_,
             *p16 = w16_ + 3 * kWinTok * C_;
    float* o = t2_;   // the attention's output [HW][C]
    float* ow = t2_ + HW * C_;   // a window's [1024][C]
    for (int64_t wy = 0; wy * kWin < H; ++wy)
        for (int64_t wx = 0; wx * kWin < W; ++wx) {
            win_gather16(qkv, H, W, C_, wy, wx, q16, k16, vt16, s_);
            gemm_.f16(q16, k16, scores_, kWinTok, kWinTok, C_);
            softmax16(scores_, kWinTok, kWinTok, 1.0f / std::sqrt((float) C_), p16, s_);
            gemm_.f16(p16, vt16, ow, kWinTok, C_, kWinTok);
            win_scatter(ow, H, W, C_, wy, wx, o, s_);
        }
    qi_to_f16(o, HW * C_, a16_, s_);
    conv(a.proj, a16_, H, W, x, true);
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
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
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
    std::vector<void*> tmp;
    float* hs = alloc<float>(tmp, n * Ch);
    float* ls = alloc<float>(tmp, n * C_);
    DiCoBufs b;
    b.t0 = alloc<float>(tmp, n * Fh);
    b.t1 = alloc<float>(tmp, n * Fh);
    b.a16 = alloc<uint16_t>(tmp, n * std::max(Fh, 3 * kPix));
    b.vec = alloc<float>(tmp, 2 * Ch);
    patchify16(rgb, h, w, b.a16, s_);
    conv(e_patch_, b.a16, h, w, hs);
    for (const DiCo& d : enc_head_) dico(d, hs, h, w, b);
    qi_to_f16(hs, n * Ch, b.a16, s_);
    conv(e_down_, b.a16, h, w, ls);
    qi_to_f16(ls, n * C_, b.a16, s_);
    conv(e_fuse_, b.a16, h, w, hs);   // the state [n][C] (in hs's room)
    for (const DiCo& d : enc_blocks_) dico(d, hs, h, w, b);
    ln_mod16(hs, e_norm_, e_norm_ + C_, n, C_, b.a16, s_);
    conv(e_out_, b.a16, h, w, moments);
    gpu::stream_sync(s_);
    for (void* p : tmp) gpu::free(p);
}

void MfVae::encode(const float* rgb, int64_t h, int64_t w, const float* eps, float* z) {
    if (f2_) {
        f2_encode(rgb, h, w, z);
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
        return;
    }
    grow(h, w);
    const int64_t HW = h * w, H = h * kPatch, W = w * kPatch;
    // the CoD decoder: the condition [HW][C]
    qi_to_f16(z, HW * cin_.in, a16_, s_);
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
    qi_to_f16(cond_, HW * C_, a16_, s_);
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
        qi_to_f16(cond_ + p0 * C_, np * C_, a16_, s_);
        gemm_.f16(a16_, yx_.w, yxb_, np, yx_.out, yx_.in);
        qi_bias_add(yxb_, yx_.b, np, yx_.out, s_);
        qi_to_f16(s_state_ + p0 * C_, np * C_, a16_, s_);
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
    if (tiled_) {   // the Flux2 VAE where the whole does not fit: the deepest front and largest tiles that do
        auto [k, t] = f2_tiling(h, w, limit);
        if (k < 0) {   // nothing fits: the least, which may still find the room
            k = 0;
            t = std::min({h, w, kF2TileMin});
        }
        f2_decode_tiled(z, h, w, rgb, k, t);
    } else {
        decode_raw(z, h, w, rgb);
    }
    to_rgba(rgb, HW, rgba, s_);
    gpu::stream_sync(s_);
    gpu::free(rgb);
}

// ---- the Flux2 VAE

void MfVae::open_flux2(SafeTensors& st) {
    f2_ = true;
    const SafeTensor* bm = st.find("bn.running_mean");
    if (bm->elements() != 128)
        throw std::runtime_error("image engine (Flux2 VAE): the batch norm is not of 128 values");
    {
        const auto m = st.f32("bn.running_mean"), v = st.f32("bn.running_var");
        std::vector<float> sd(v.size());
        for (size_t i = 0; i < v.size(); ++i) sd[i] = std::sqrt(v[i] + 1e-4f);   // ComfyUI's bn_eps
        f2_mean_ = upload(m);
        f2_std_ = upload(sd);
    }
    // the names as the LDM checkpoint has them (BFL's ae.safetensors, ComfyUI's flux2-vae), or as diffusers' (the vae/
    // folder of FLUX.2-klein): nm takes the first to the file's
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
    f2_pq_ = conv_w(st, nm(d + "post_quant_conv"));
    cin_ = conv_w(st, nm(d + "conv_in"));
    if (f2_pq_.in != 32 || cin_.in != 32)
        throw std::runtime_error("image engine (Flux2 VAE): the latent is not 32 wide");
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
        z.most = std::max(pix * 32, pix * 3 * C_);
        z.rows = std::max<int64_t>(1, std::min<int64_t>(z.mid, (int64_t) (16 << 20) / z.mid));
    }
    z.most = std::max(z.most, pix * c);
    for (int i = k0; i < k1; ++i) {
        const F2Level& l = f2_levels_[(size_t) i];
        for (const F2Res& r : l.blocks) z.most = std::max(z.most, pix * std::max(r.c1.in, r.c2.out));
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out > 0) {
            pix *= 4;
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
    const int64_t col_rows = std::min<int64_t>(z.pix, 32768);
    return (uint64_t) z.most * (3 * 4 + 2) +                           // x, y, t; a16
           (uint64_t) col_rows * 9 * cmax * 2 +                        // the unfolded neighbourhoods, a band
           (uint64_t) z.pix * kGroups * 2 * 4 + 4 * (uint64_t) cmax * 4 +   // the group norms' sums, stats
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
    f2_band_ = std::min<int64_t>(n.pix, 32768) * 9 * cmax;
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) cmax = std::max(cmax, r.c1.in);
        cmax = std::max(cmax, l.down.in);
    }
    f2_x_ = alloc<float>(allocs_, n.most);
    f2_y_ = alloc<float>(allocs_, n.most);
    f2_t_ = alloc<float>(allocs_, n.most);
    a16_ = alloc<uint16_t>(allocs_, n.most);
    col_cap_ = std::min<int64_t>(n.pix, 32768) * 9 * cmax;
    col16_ = alloc<uint16_t>(allocs_, col_cap_);
    gn_part_ = alloc<float>(allocs_, n.pix * kGroups * 2);
    vec_ = alloc<float>(allocs_, 4 * cmax);
    f2_qk16_ = alloc<uint16_t>(allocs_, n.mid * f2_ac_ * 3);
    f2_q_ = alloc<float>(allocs_, n.rows * n.mid);
    f2_p16_ = alloc<uint16_t>(allocs_, n.rows * n.mid);
    f2_cap_ = n;
    f2_held_ = f2_bytes(n);
}

// the two of x, y, t that are not `x`
void MfVae::f2_others(const float* x, float*& a, float*& b) const {
    float* all[3] = {f2_x_, f2_y_, f2_t_};
    int k = 0;
    for (float* p : all)
        if (p != x) (k++ == 0 ? a : b) = p;
}

// x = x (or its 1 x 1 shortcut) + conv2(SiLU(GN(conv1(SiLU(GN(x)))))): the result in another of x, y, t
void MfVae::f2_resblock(const F2Res& r, float*& x, int64_t H, int64_t W) {
    const int64_t HW = H * W;
    float *t = nullptr, *y = nullptr;
    f2_others(x, t, y);
    gn_silu16(x, r.n1w, r.n1b, HW, r.c1.in, true, a16_);
    conv(r.c1, a16_, H, W, t);
    gn_silu16(t, r.n2w, r.n2b, HW, r.c1.out, true, a16_);
    if (r.nin.out > 0) {
        conv(r.c2, a16_, H, W, y);
        qi_to_f16(x, HW * r.nin.in, a16_, s_);
        conv(r.nin, a16_, H, W, y, true);
        x = y;
    } else {
        conv(r.c2, a16_, H, W, x, true);   // x itself takes the sum
    }
}

// x += proj(softmax(q k^T / sqrt(C)) v), q, k, v of GN(x): one head over every position, a band of query rows at a time
void MfVae::f2_attention(const Attn& a, float* x, int64_t H, int64_t W) {
    const int64_t T = H * W, C = a.qkv.in, rows = f2_cap_.rows;
    float *qkv = nullptr, *o = nullptr;
    f2_others(x, qkv, o);
    gn_silu16(x, a.nw, a.nb, T, C, false, a16_);
    conv(a.qkv, a16_, H, W, qkv);
    uint16_t *q16 = f2_qk16_, *k16 = f2_qk16_ + T * C, *vt16 = f2_qk16_ + 2 * T * C;
    qkv_split16(qkv, T, C, q16, k16, vt16, s_);
    for (int64_t r0 = 0; r0 < T; r0 += rows) {
        const int64_t n = std::min(rows, T - r0);
        gemm_.f16(q16 + r0 * C, k16, f2_q_, n, T, C);
        softmax_wg16(f2_q_, n, T, 1.0f / std::sqrt((float) C), f2_p16_, s_);
        gemm_.f16(f2_p16_, vt16, o + r0 * C, n, C, T);
    }
    qi_to_f16(o, T * C, a16_, s_);
    conv(a.proj, a16_, H, W, x, true);
}

// the stages up to level k of the latent z (h x w): x [H][W][c] in one of x, y, t
void MfVae::f2_front(const float* z, int64_t h, int64_t w, int k, float*& x, int64_t& H, int64_t& W, int64_t& c) {
    H = 2 * h;
    W = 2 * w;
    f2_unpatch16(z, f2_mean_, f2_std_, h, w, a16_, s_);
    conv(f2_pq_, a16_, H, W, f2_t_);
    qi_to_f16(f2_t_, H * W * f2_pq_.out, a16_, s_);
    x = f2_x_;
    conv(cin_, a16_, H, W, x);
    f2_resblock(f2_mid_[0], x, H, W);
    f2_attention(f2_attn_, x, H, W);
    f2_resblock(f2_mid_[1], x, H, W);
    c = C_;
    f2_levels(0, k, x, H, W, c);
}

// levels k0 .. k1 - 1 on x [H][W][c]
void MfVae::f2_levels(int k0, int k1, float*& x, int64_t& H, int64_t& W, int64_t& c) {
    for (int i = k0; i < k1; ++i) {
        const F2Level& l = f2_levels_[(size_t) i];
        for (const F2Res& r : l.blocks) f2_resblock(r, x, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.up.out > 0) {
            up2_16(x, H, W, c, a16_, s_);
            H *= 2;
            W *= 2;
            float *y = nullptr, *unused = nullptr;
            f2_others(x, y, unused);
            conv(l.up, a16_, H, W, y);
            x = y;
            c = l.up.out;
        }
    }
}

// norm_out and conv_out of x [H][W][c]: rgb [3][H][W]
void MfVae::f2_tail(const float* x, int64_t H, int64_t W, int64_t c, float* rgb) {
    gn_silu16(x, gnw_, gnb_, H * W, c, true, a16_);
    float *y = nullptr, *unused = nullptr;
    f2_others(x, y, unused);
    conv(cout_, a16_, H, W, y);
    hwc3_to_chw(y, H * W, rgb, s_);
}

void MfVae::f2_decode_raw(const float* z, int64_t h, int64_t w, float* rgb) {
    const int n = (int) f2_levels_.size();
    f2_grow(f2_walk(h, w, 0, n, true, true));
    float* x = nullptr;
    int64_t H = 0, W = 0, c = 0;
    f2_front(z, h, w, n, x, H, W, c);
    f2_tail(x, H, W, c, rgb);
}

// A tiled decode: the stages up to level k on the whole latent (its statistics and the middle's attention over all of
// it), their output kept, and the levels from k on in tiles of t x t tokens.  Its scratch: the buffers the two share,
// the output kept, a tile's input and image, and the blend's planes
uint64_t MfVae::f2_tiled_bytes(int64_t h, int64_t w, int k, int64_t t) const {
    const int n = (int) f2_levels_.size();
    const int64_t th = std::min(h, t), tw = std::min(w, t);
    const F2Sizes f = f2_walk(h, w, 0, k, true, false), b = f2_walk(th, tw, k, n, false, true);
    const F2Sizes bi = f2_walk(th, tw, k, k, false, false);   // a tile's input
    return f2_bytes(f2_max(f, b)) + (uint64_t) (f.pix * f.c) * 4 + (uint64_t) (bi.pix * bi.c) * 4 +
           (uint64_t) (3 * th * tw * kPix) * 4 + (uint64_t) (4 * h * w * kPix) * 4;
}

// the deepest front and the largest tiles within `limit` -> (k, t); (-1, 0) when nothing fits
std::pair<int, int64_t> MfVae::f2_tiling(int64_t h, int64_t w, uint64_t limit) const {
    const int n = (int) f2_levels_.size();
    const int64_t least = std::min({h, w, kF2TileMin});
    for (int k = n; k >= 0; --k) {
        if (f2_tiled_bytes(h, w, k, least) > limit) continue;
        int64_t t = std::max(h, w);
        while (t > least && f2_tiled_bytes(h, w, k, t) > limit) --t;
        return {k, t};
    }
    return {-1, 0};
}

uint64_t MfVae::f2_least_bytes(int64_t h, int64_t w) const {
    const int n = (int) f2_levels_.size();
    const int64_t least = std::min({h, w, kF2TileMin});
    uint64_t b = f2_scratch_bytes(h, w);
    for (int k = 0; k <= n; ++k) b = std::min(b, f2_tiled_bytes(h, w, k, least));
    return b;
}

void MfVae::f2_decode_tiled(const float* z, int64_t h, int64_t w, float* rgb, int k, int64_t t) {
    const int n = (int) f2_levels_.size();
    const int64_t th = std::min(h, t), tw = std::min(w, t), H = h * kPatch, W = w * kPatch;
    const F2Sizes f = f2_walk(h, w, 0, k, true, false), b = f2_walk(th, tw, k, n, false, true);
    f2_grow(f2_max(f, b));
    std::vector<void*> tmp;
    float* acc = alloc<float>(tmp, 4 * H * W);
    float* trgb = alloc<float>(tmp, 3 * th * tw * kPix);
    gpu::memset_async(acc, 0, (size_t) (4 * H * W) * 4, s_);
    // the front on the whole, its output kept [FH][FW][c]: s positions a token
    float* x = nullptr;
    int64_t FH = 0, FW = 0, c = 0;
    f2_front(z, h, w, k, x, FH, FW, c);
    const int64_t sp = FH / h;
    float* kept = alloc<float>(tmp, FH * FW * c);
    gpu::copy_async(kept, x, (size_t) (FH * FW * c) * 4, s_);
    // the tiles' corners: t apart less the overlap, the last flush with the edge
    auto starts = [](int64_t n, int64_t side) {
        std::vector<int64_t> v;
        for (int64_t a = 0;; a += side - kF2Overlap) {
            if (a + side >= n) {
                v.push_back(n - side);
                break;
            }
            v.push_back(a);
        }
        return v;
    };
    const auto ys = starts(h, th), xs = starts(w, tw);
    for (const int64_t y0 : ys)
        for (const int64_t x0 : xs) {
            int64_t TH = th * sp, TW = tw * sp, tc = c;
            float* tx = f2_x_;
            tile_gather(kept, FW, c, y0 * sp, x0 * sp, TH, TW, tx, s_);
            f2_levels(k, n, tx, TH, TW, tc);
            f2_tail(tx, TH, TW, tc, trgb);
            tile_blend(trgb, th * kPatch, tw * kPatch, y0 * kPatch, x0 * kPatch, H, W, kF2Overlap * kPatch, y0 > 0,
                       y0 + th < h, x0 > 0, x0 + tw < w, acc, s_);
        }
    tile_finish(acc, H * W, rgb, s_);
    gpu::stream_sync(s_);
    for (void* p : tmp) gpu::free(p);
}

// ---- the Flux2 VAE's encoder

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
        if (t->shape.size() != 4 || t->shape[1] != 3 || t->shape[2] != 3 || t->shape[3] != 3)
            throw std::runtime_error("image engine (Flux2 VAE): the encoder's conv_in is not 3 x 3 over RGB");
        f2_ein_.out = t->shape[0];
        f2_ein_.in = 8;
        f2_ein_.k = 3;
        weight_bytes_ += (uint64_t) (f2_ein_.out * 9 * 8) * 2;
        if (!dry_) {
            const auto w = st.f32(n + ".weight");   // [out][3][3][3]
            std::vector<uint16_t> h((size_t) (f2_ein_.out * 9 * 8), 0);
            for (int64_t o = 0; o < f2_ein_.out; ++o)
                for (int64_t i = 0; i < 3; ++i)
                    for (int64_t k = 0; k < 9; ++k)
                        h[(size_t) ((o * 9 + k) * 8 + i)] = kernels::f16_from_f32(w[(size_t) ((o * 3 + i) * 9 + k)]);
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
            gpu::copy(d, h.data(), h.size() * 2);
            f2_ein_.w = d;
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
    if (f2_eout_.out != 64) throw std::runtime_error("image engine (Flux2 VAE): the encoder's moments are not 2 x 32");
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
    z.most = std::max({z.most, pix * 3 * f2_eattn_.qkv.in, pix * 64});
    z.rows = std::max<int64_t>(1, std::min<int64_t>(z.mid, (int64_t) (16 << 20) / z.mid));
    z.c = c;
    return z;
}

void MfVae::f2_encode(const float* rgb, int64_t h, int64_t w, float* z) {
    if (!has_encoder()) throw std::runtime_error("image engine (Flux2 VAE): the encoder is not loaded");
    int64_t H = 16 * h, W = 16 * w;
    f2_grow(f2_enc_walk(h, w));
    f2_rgb8_16(rgb, H * W, a16_, s_);
    float* x = f2_x_;
    conv(f2_ein_, a16_, H, W, x);
    int64_t c = f2_ein_.out;
    for (const F2Down& l : f2_down_) {
        for (const F2Res& r : l.blocks) f2_resblock(r, x, H, W);
        if (!l.blocks.empty()) c = l.blocks.back().c2.out;
        if (l.down.out > 0) {
            qi_to_f16(x, H * W * c, a16_, s_);
            float *y = nullptr, *unused = nullptr;
            f2_others(x, y, unused);
            conv_s2(l.down, a16_, H, W, y);
            H /= 2;
            W /= 2;
            x = y;
            c = l.down.out;
        }
    }
    f2_resblock(f2_emid_[0], x, H, W);
    f2_attention(f2_eattn_, x, H, W);
    f2_resblock(f2_emid_[1], x, H, W);
    gn_silu16(x, f2_enw_, f2_enb_, H * W, c, true, a16_);
    float *m = nullptr, *t = nullptr;
    f2_others(x, m, t);
    conv(f2_eout_, a16_, H, W, m);
    if (f2_qc_.out > 0) {
        qi_to_f16(m, H * W * f2_eout_.out, a16_, s_);
        conv(f2_qc_, a16_, H, W, t);
        m = t;
    }
    f2_patch(m, f2_mean_, f2_std_, h, w, z, s_);
    gpu::stream_sync(s_);
}

}  // namespace strata::image
