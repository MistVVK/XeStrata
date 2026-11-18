// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2024 ZhengPeng
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/image/aux_kernels.cpp - see include/strata/image/aux_kernels.hpp.

#include "strata/image/aux_kernels.hpp"

#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/qk_norm_rope.hpp"

#include "row_norm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace strata::image {

using rows::kRowVals;
using rows::row_items;
using rows::row_stats;
using rows::row_vals;

namespace {

// FP16 by the GPU's own conversion (round to nearest even, as kernels::f16_from_f32): the bits' arithmetic of
// f16_bits.hpp, a few dozen instructions a value, held the elementwise passes of SAM 3's ViT (its MLP's bias and GELU,
// its norms) to about half the memory's rate on the RTX 4070 (2026-11-07)
inline uint16_t f16_from_f32(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }
inline float f32_from_f16(uint16_t b) { return static_cast<float>(sycl::bit_cast<sycl::half>(b)); }

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }


void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

void aux_rgb_norm16(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, uint16_t* y16,
                    void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float m0 = mean[0], m1 = mean[1], m2 = mean[2], s0 = 1.f / std[0], s1 = 1.f / std[1], s2 = 1.f / std[2];
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[3 * i] = f16_from_f32(((float) rgba[4 * i] / 255.f - m0) * s0);
        y16[3 * i + 1] = f16_from_f32(((float) rgba[4 * i + 1] / 255.f - m1) * s1);
        y16[3 * i + 2] = f16_from_f32(((float) rgba[4 * i + 2] / 255.f - m2) * s2);
    });
    finish(stream, e, "aux_rgb_norm16");
}

void aux_conv_few16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, const uint16_t* w16, int64_t O,
                    int64_t real, int k, int stride, int pad, int64_t OH, int64_t OW, float* y, void* stream) {
    if (OH <= 0 || OW <= 0 || O <= 0) return;
    if (O > kConvFewOut || real > O) throw std::invalid_argument("aux_conv_few16: more than kConvFewOut outputs");
    // a pixel's channels shared by kLanes work-items (each read of a tap's channels one run of memory), their sums
    // added in local memory
    constexpr int64_t kLanes = 16, kPixels = 16;
    const int64_t K = (int64_t) k * k * C, P = OH * OW, groups = (P + kPixels - 1) / kPixels;
    const auto e = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (kLanes * kPixels * kConvFewOut)), hd);
        hd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * kLanes * kPixels)),
                                          sycl::range<1>((size_t) (kLanes * kPixels))),
                        [=](sycl::nd_item<1> it) {
            const auto l = static_cast<int64_t>(it.get_local_id(0));
            const int64_t lane = l % kLanes, slot = l / kLanes;
            const int64_t p = static_cast<int64_t>(it.get_group(0)) * kPixels + slot;
            float acc[kConvFewOut] = {};
            if (p < P) {
                const int64_t oy = p / OW, ox = p - oy * OW;
                for (int ty = 0; ty < k; ++ty) {
                    const int64_t yy = oy * stride - pad + ty;
                    if (yy < 0 || yy >= H) continue;
                    for (int tx = 0; tx < k; ++tx) {
                        const int64_t xx = ox * stride - pad + tx;
                        if (xx < 0 || xx >= W) continue;
                        const uint16_t* px = x16 + (yy * W + xx) * C;
                        const int64_t t0 = ((int64_t) ty * k + tx) * C;
                        for (int64_t c = lane; c < C; c += kLanes) {
                            const float v = f32_from_f16(px[c]);
                            for (int64_t o = 0; o < real; ++o) acc[o] += v * f32_from_f16(w16[o * K + t0 + c]);
                        }
                    }
                }
            }
            for (int64_t o = 0; o < kConvFewOut; ++o) part[(size_t) (l * kConvFewOut + o)] = acc[o];
            sycl::group_barrier(it.get_group());
            if (lane < O && p < P) {   // lane o sums output o over the pixel's lanes
                float s = 0.f;
                if (lane < real)
                    for (int64_t j = 0; j < kLanes; ++j) s += part[(size_t) ((slot * kLanes + j) * kConvFewOut + lane)];
                y[p * O + lane] = s;
            }
        });
    });
    finish(stream, e, "aux_conv_few16");
}

void aux_dwconv3_norm16(const float* x, const float* w, const float* b, const float* nw, const float* nb, int64_t H,
                        int64_t W, int64_t C, int64_t o_ws, float eps, float* y, uint16_t* y16, void* stream) {
    Depthwise d; d.x = x; d.w = w; d.bias = b; d.norm_w = nw; d.norm_b = nb;
    d.out = y; d.out16 = y16; d.h = d.oh = H; d.wdim = d.ow = W; d.c = C;
    d.window = o_ws; d.eps = eps; d.residual = true;
    image_depthwise(d, stream);
}

namespace {
// aux_layer_norm's rows, row t's output at row to(t)
template <class To>
void layer_norm_rows(const float* x, const float* r, const float* w, const float* b, int64_t T, int64_t C, float eps,
                     float* y, uint16_t* y16, To to, void* stream, const char* what, const float* rb = nullptr,
                     bool gelu = false) {
    if (T <= 0) return;
    const bool regs = row_vals(C);
    const int64_t L = regs ? row_items(C) : 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * L)), sycl::range<1>((size_t) L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t o = to(t);
            const float* xr = x + t * C;
            const float* rr = r != nullptr ? r + t * C : nullptr;
            if (regs) {
                float v[kRowVals];
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    v[k] = c < C ? xr[c] + (rr ? rr[c] + (rb != nullptr ? rb[c] : 0.f) : 0.f) : 0.f;
                }
                const sycl::float2 st = row_stats(it, v, l, L, C, eps);
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    if (c >= C) continue;
                    float u = (v[k] - st[0]) * st[1] * w[c] + b[c];
                    if (gelu) u = 0.5f * u * (1.f + sycl::erf(u * 0.70710678118654752f));
                    if (y16 != nullptr) y16[o * C + c] = f16_from_f32(u);
                    if (y != nullptr) y[o * C + c] = u;
                }
                return;
            }
            float s = 0.f;
            for (int64_t c = l; c < C; c += L) s += xr[c] + (rr ? rr[c] + (rb != nullptr ? rb[c] : 0.f) : 0.f);
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float s2 = 0.f;   // two passes: the variance about the mean (the residual streams' large values)
            for (int64_t c = l; c < C; c += L) {
                const float d = xr[c] + (rr ? rr[c] + (rb != nullptr ? rb[c] : 0.f) : 0.f) - mean;
                s2 += d * d;
            }
            const float inv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
            for (int64_t c = l; c < C; c += L) {
                float v = (xr[c] + (rr ? rr[c] + (rb != nullptr ? rb[c] : 0.f) : 0.f) - mean) * inv * w[c] + b[c];
                if (gelu) v = 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f));
                if (y16 != nullptr) y16[o * C + c] = f16_from_f32(v);
                if (y != nullptr) y[o * C + c] = v;
            }
        });
    finish(stream, e, what);
}
}  // namespace

void aux_layer_norm(const float* x, const float* r, const float* w, const float* b, int64_t T, int64_t C, float eps,
                    float* y, uint16_t* y16, void* stream) {
    layer_norm_rows(x, r, w, b, T, C, eps, y, y16, [](int64_t t) { return t; }, stream, "aux_layer_norm");
}

void aux_layer_norm_gelu16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps,
                           uint16_t* y16, void* stream) {
    if (T <= 0) return;
    if (C > 64) {
        layer_norm_rows(x, nullptr, w, b, T, C, eps, nullptr, y16, [](int64_t t) { return t; }, stream,
                        "aux_layer_norm_gelu16", nullptr, true);
        return;
    }
    // few channels (a mask downsampler's first layers, 16 a row over 331,776 rows): a work-item a value, 256 / C rows a
    // work-group, each row's sums by halving in local memory - the reads and writes in rows' order (a work-group a row
    // left most of its items idle; a work-item a row read its values C apart from its neighbours')
    if (256 % C == 0) {
        constexpr int L = 256;
        const int ch = (int) C;
        const int64_t groups = (T * C + L - 1) / L;
        const auto e = queue_for(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sh(L, h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * L)), sycl::range<1>(L)),
                           [=](sycl::nd_item<1> it) {
                const int l = (int) it.get_local_linear_id(), c = l % ch, base = l - c;
                const int64_t i = (int64_t) it.get_group_linear_id() * L + l;
                const bool live = i < T * C;
                const float v = live ? x[i] : 0.f;
                const auto row_sum = [&](float a) {
                    sh[l] = a;
                    for (int st = ch / 2; st > 0; st /= 2) {
                        sycl::group_barrier(it.get_group());
                        if (c < st) sh[l] += sh[l + st];
                    }
                    sycl::group_barrier(it.get_group());
                    const float r = sh[base];
                    sycl::group_barrier(it.get_group());
                    return r;
                };
                const float mean = row_sum(v) / (float) ch;
                const float dv = v - mean;
                const float inv = 1.f / sycl::sqrt(row_sum(dv * dv) / (float) ch + eps);
                if (!live) return;
                const float u = dv * inv * w[c] + b[c];
                y16[i] = f16_from_f32(0.5f * u * (1.f + sycl::erf(u * 0.70710678118654752f)));
            });
        });
        finish(stream, e, "aux_layer_norm_gelu16");
        return;
    }
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> id) {
        const auto t = static_cast<int64_t>(id[0]);
        const float* xr = x + t * C;
        float s = 0.f;
        for (int64_t c = 0; c < C; ++c) s += xr[c];
        const float mean = s / (float) C;
        float s2 = 0.f;
        for (int64_t c = 0; c < C; ++c) {
            const float d = xr[c] - mean;
            s2 += d * d;
        }
        const float inv = 1.f / sycl::sqrt(s2 / (float) C + eps);
        for (int64_t c = 0; c < C; ++c) {
            const float u = (xr[c] - mean) * inv * w[c] + b[c];
            y16[t * C + c] = f16_from_f32(0.5f * u * (1.f + sycl::erf(u * 0.70710678118654752f)));
        }
    });
    finish(stream, e, "aux_layer_norm_gelu16");
}

void aux_residual_norm16(float* x, const float* r, const float* rb, const float* w, const float* b, int64_t H, int64_t W,
                         int64_t C, int64_t r_ws, int64_t o_ws, float eps, uint16_t* y16, void* stream) {
    const int64_t T = H * W;
    if (T <= 0) return;
    const bool regs = row_vals(C);
    const int64_t L = regs ? row_items(C) : 256;
    // row t of the grid at its window's place (ws > 0: aux_window16's order, no shift, whole windows), else as it is
    const auto place = [=](int64_t t, int64_t ws) {
        if (ws <= 0) return t;
        const int64_t py = t / W, px = t - py * W;
        return ((py / ws) * (W / ws) + px / ws) * ws * ws + (py % ws) * ws + px % ws;
    };
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * L)), sycl::range<1>((size_t) L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            float* xr = x + t * C;
            const float* rr = r + place(t, r_ws) * C;
            if (regs) {
                float v[kRowVals];
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    v[k] = c < C ? xr[c] + rr[c] + (rb != nullptr ? rb[c] : 0.f) : 0.f;
                    if (c < C) xr[c] = v[k];
                }
                if (w == nullptr) return;
                const sycl::float2 st = row_stats(it, v, l, L, C, eps);
                uint16_t* yr = y16 + place(t, o_ws) * C;
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    if (c < C) yr[c] = f16_from_f32((v[k] - st[0]) * st[1] * w[c] + b[c]);
                }
                return;
            }
            float s = 0.f;
            for (int64_t c = l; c < C; c += L) {   // each item its own columns: written, then read back
                const float v = xr[c] + rr[c] + (rb != nullptr ? rb[c] : 0.f);
                xr[c] = v;
                s += v;
            }
            if (w == nullptr) return;
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float s2 = 0.f;
            for (int64_t c = l; c < C; c += L) {
                const float d = xr[c] - mean;
                s2 += d * d;
            }
            const float inv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
            uint16_t* yr = y16 + place(t, o_ws) * C;
            for (int64_t c = l; c < C; c += L) yr[c] = f16_from_f32((xr[c] - mean) * inv * w[c] + b[c]);
        });
    finish(stream, e, "aux_residual_norm16");
}

void aux_add_norm(const float* x, const float* r, const float* rb, const float* w, const float* b, int64_t T, int64_t C,
                  float eps, float* y, uint16_t* y16, void* stream) {
    layer_norm_rows(x, r, w, b, T, C, eps, y, y16, [](int64_t t) { return t; }, stream, "aux_add_norm", rb);
}

void aux_cache_put_kv(const uint16_t* src16, int64_t B, int64_t ld, int64_t koff, int64_t voff, int64_t C,
                      uint16_t* kc, uint16_t* vc, int64_t cap, const int32_t* at, void* stream) {
    if (B <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) B, (size_t) C), [=](sycl::id<2> id) {
        const auto b = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        const int64_t r = (b * cap + at[0]) * C + c;
        kc[r] = src16[b * ld + koff + c];
        vc[r] = src16[b * ld + voff + c];
    });
    finish(stream, e, "aux_cache_put_kv");
}

void aux_layer_norm_window16(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C,
                             int64_t ws, float eps, uint16_t* y16, void* stream) {
    const int64_t nwx = W / ws;
    layer_norm_rows(x, nullptr, w, b, H * W, C, eps, nullptr, y16, [=](int64_t t) {
        const int64_t py = t / W, px = t - py * W;
        return ((py / ws) * nwx + px / ws) * ws * ws + (py % ws) * ws + px % ws;
    }, stream, "aux_layer_norm_window16");
}

void aux_window16(const uint16_t* src16, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                  int64_t Wp, uint16_t* dst16, void* stream) {
    const int64_t n = Hp * Wp * C;
    if (n <= 0) return;
    const int64_t nwx = Wp / ws;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = i / C, win = t / (ws * ws), r = t - win * ws * ws;
        const int64_t y = ((win / nwx) * ws + r / ws + shift) % Hp, x = ((win % nwx) * ws + r % ws + shift) % Wp;
        dst16[i] = y < H && x < W ? src16[(y * W + x) * C + c] : (uint16_t) 0;
    });
    finish(stream, e, "aux_window16");
}

void aux_unwindow_add(float* x, const float* y, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                      int64_t Wp, void* stream) {
    const int64_t n = Hp * Wp * C;
    if (n <= 0) return;
    const int64_t nwx = Wp / ws;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = i / C, win = t / (ws * ws), r = t - win * ws * ws;
        const int64_t py = ((win / nwx) * ws + r / ws + shift) % Hp, px = ((win % nwx) * ws + r % ws + shift) % Wp;
        if (py < H && px < W) x[(py * W + px) * C + c] += y[i];
    });
    finish(stream, e, "aux_unwindow_add");
}

void aux_channel_split(const float* qkv, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16, uint16_t* kT16,
                       uint16_t* v16, void* stream) {
    const int64_t n = N * C;
    if (n <= 0) return;
    const int64_t Cg = C / G;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        // i over [G][Cg][N]: the transposed operands' writes in order
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i % N, gc = i / N, g = gc / Cg, c = gc - g * Cg;
        qT16[i] = f16_from_f32(qkv[(t * 3) * C + g * Cg + c] * scale);
        kT16[i] = f16_from_f32(qkv[(t * 3 + 1) * C + g * Cg + c]);
        v16[(g * N + t) * Cg + c] = f16_from_f32(qkv[(t * 3 + 2) * C + g * Cg + c]);
    });
    finish(stream, e, "aux_channel_split");
}

void aux_channel_split16(const uint16_t* qkv16, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16,
                         uint16_t* kT16, uint16_t* v16, void* stream) {
    if (N <= 0 || C <= 0) return;
    constexpr int64_t TS = 32, RW = 8;   // a tile of 32 tokens by 32 channels, 8 rows of items
    const int64_t Cg = C / G, nt = (N + TS - 1) / TS;
    sycl::queue& q = queue_for(stream);
    // q and k (2C channels) transposed through local memory: read along the channels, written along the tokens
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 2> tile(sycl::range<2>(TS, TS + 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (nt * RW), (size_t) (2 * C)),
                                         sycl::range<2>((size_t) RW, (size_t) TS)),
                       [=](sycl::nd_item<2> it) {
            const auto ly = static_cast<int64_t>(it.get_local_id(0)), lx = static_cast<int64_t>(it.get_local_id(1));
            const int64_t t0 = static_cast<int64_t>(it.get_group(0)) * TS, c0 = static_cast<int64_t>(it.get_group(1)) * TS;
            for (int64_t r = ly; r < TS; r += RW) {
                const int64_t t = t0 + r;
                tile[r][lx] = t < N ? qkv16[t * 3 * C + c0 + lx] : (uint16_t) 0;
            }
            sycl::group_barrier(it.get_group());
            const int64_t t = t0 + lx;
            if (t < N)
                for (int64_t r = ly; r < TS; r += RW) {
                    const int64_t c = c0 + r;
                    const uint16_t v = tile[lx][r];
                    if (c < C) qT16[c * N + t] = f16_from_f32(f32_from_f16(v) * scale);
                    else kT16[(c - C) * N + t] = v;
                }
        });
    });
    finish(stream, e, "aux_channel_split16 (q, k)");
    e = q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const auto t = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        const int64_t g = c / Cg;
        v16[(g * N + t) * Cg + c - g * Cg] = qkv16[t * 3 * C + 2 * C + c];
    });
    finish(stream, e, "aux_channel_split16 (v)");
}

void aux_channel_merge16(const float* yg, int64_t N, int64_t C, int64_t G, uint16_t* y16, void* stream) {
    if (N <= 0 || C <= 0) return;
    const int64_t Cg = C / G;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const auto t = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        const int64_t g = c / Cg;
        y16[t * C + c] = f16_from_f32(yg[(g * N + t) * Cg + c - g * Cg]);
    });
    finish(stream, e, "aux_channel_merge16");
}

void aux_softmax_attn16(const float* s, int64_t batch, int64_t heads, int64_t rows, int64_t cols, float scale,
                        const float* bias, const float* mask, int64_t mask_n, const float* grid, int64_t G,
                        int64_t skip, uint16_t* p16, void* stream) {
    if (batch * heads * rows <= 0) return;
    constexpr size_t L = 256;
    const int64_t hr = heads * rows;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (batch * hr) * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t b = g / hr, h = (g / rows) % heads, i = g % rows;
            const float* x = s + g * cols;
            const float* bb = bias != nullptr ? bias + (h * rows + i) * cols : nullptr;
            const float* mb = mask != nullptr ? mask + ((b % mask_n) * rows + i) * cols : nullptr;
            const float* gb = grid != nullptr && i >= skip ? grid + (i - skip) * 2 * G * heads : nullptr;
            const auto at = [&](int64_t c) {
                float v = x[c] * scale;
                if (bb != nullptr) v += bb[c];
                if (mb != nullptr) v += mb[c];
                if (gb != nullptr) {
                    const int64_t y = c / G;
                    v += gb[y * heads + h] + gb[(G + c - y * G) * heads + h];
                }
                return v;
            };
            float m = -INFINITY;
            for (int64_t c = l; c < cols; c += (int64_t) L) m = sycl::fmax(m, at(c));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float z = 0.f;
            for (int64_t c = l; c < cols; c += (int64_t) L) z += sycl::exp(at(c) - m);
            z = sycl::reduce_over_group(it.get_group(), z, sycl::plus<float>());
            const float inv = 1.f / z;
            for (int64_t c = l; c < cols; c += (int64_t) L) p16[g * cols + c] = f16_from_f32(sycl::exp(at(c) - m) * inv);
        });
    finish(stream, e, "aux_softmax_attn16");
}

void aux_heads16(const uint16_t* x, int64_t rows, int64_t heads, int64_t hd, int64_t ld, bool transpose, uint16_t* y,
                 void* stream) {
    const int64_t n = heads * rows * hd;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);   // the source's order: rows, then its channels
        const int64_t r = i / (heads * hd), c = i - r * heads * hd, h = c / hd, d = c - h * hd;
        y[transpose ? (h * hd + d) * rows + r : (h * rows + r) * hd + d] = x[r * ld + c];
    });
    finish(stream, e, "aux_heads16");
}

void aux_heads_batch16(const uint16_t* x, int64_t batch, int64_t rows, int64_t heads, int64_t hd, int64_t ld,
                       int64_t xbatch, bool transpose, uint16_t* y, void* stream) {
    const int64_t n = batch * heads * rows * hd;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);   // the destination's order: batch, head, row, channel
        const int64_t d = i % hd, r = (i / hd) % rows, h = (i / (hd * rows)) % heads, b = i / (hd * rows * heads);
        const uint16_t v = x[b * xbatch + r * ld + h * hd + d];
        y[transpose ? ((b * heads + h) * hd + d) * rows + r : i] = v;
    });
    finish(stream, e, "aux_heads_batch16");
}

void aux_heads_merge16(const float* o, int64_t rows, int64_t heads, int64_t hd, uint16_t* y, void* stream) {
    const int64_t n = heads * rows * hd;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / (heads * hd), c = i - r * heads * hd, h = c / hd, d = c - h * hd;
        y[i] = f16_from_f32(o[(h * rows + r) * hd + d]);
    });
    finish(stream, e, "aux_heads_merge16");
}

void aux_heads_merge_batch16(const float* o, int64_t batch, int64_t rows, int64_t heads, int64_t hd, int64_t ld,
                             int64_t ybatch, uint16_t* y, void* stream) {
    const int64_t n = batch * heads * rows * hd;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);   // the source's order: batch, head, row, channel
        const int64_t d = i % hd, r = (i / hd) % rows, h = (i / (hd * rows)) % heads, b = i / (hd * rows * heads);
        y[b * ybatch + r * ld + h * hd + d] = f16_from_f32(o[i]);
    });
    finish(stream, e, "aux_heads_merge_batch16");
}

void aux_embed(const int32_t* ids, const int32_t* pos_ids, const uint16_t* tok, const uint16_t* pos, int64_t n,
               int64_t C, float scale, float* y, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        const float p = f32_from_f16(pos[(int64_t) pos_ids[t] * C + c]);
        y[i] = tok != nullptr ? f32_from_f16(tok[(int64_t) ids[t] * C + c]) * scale + p : y[i] + p;
    });
    finish(stream, e, "aux_embed");
}

void aux_split3_16(const float* x, int64_t T, int64_t C, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        q16[i] = f16_from_f32(x[t * 3 * C + c]);
        k16[i] = f16_from_f32(x[t * 3 * C + C + c]);
        v16[i] = f16_from_f32(x[t * 3 * C + 2 * C + c]);
    });
    finish(stream, e, "aux_split3_16");
}

void aux_cache_put(const uint16_t* src16, int64_t B, int64_t ld, int64_t off, int64_t C, uint16_t* cache, int64_t cap,
                   int64_t at, void* stream) {
    if (B <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (B * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t b = i / C, c = i - b * C;
        cache[(b * cap + at) * C + c] = src16[b * ld + off + c];
    });
    finish(stream, e, "aux_cache_put");
}

void aux_cache_gather(const uint16_t* src, const int32_t* from, int64_t B, int64_t cap, int64_t len, int64_t C,
                      uint16_t* dst, void* stream) {
    if (B <= 0 || len <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (B * len * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = (i / C) % len, b = i / (C * len);
        dst[(b * cap + t) * C + c] = src[((int64_t) from[b] * cap + t) * C + c];
    });
    finish(stream, e, "aux_cache_gather");
}

void aux_polygon_fill(const float* xy, const int32_t* starts, int64_t n, int64_t W, int64_t H, uint8_t* mask,
                      void* stream) {
    if (W <= 0 || H <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (W * H)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t row = i / W;
        const float px = (float) (i - row * W), py = (float) row;
        bool on = false;
        for (int64_t p = 0; p < n && !on; ++p) {
            const int64_t a = starts[p], z = starts[p + 1];
            if (z - a < 3) continue;
            bool in = false;
            for (int64_t v = a, u = z - 1; v < z; u = v++) {
                const float x0 = xy[2 * u], y0 = xy[2 * u + 1], x1 = xy[2 * v], y1 = xy[2 * v + 1];
                // even-odd crossing of the row through the pixel's center
                if ((y1 > py) != (y0 > py) && px < (x0 - x1) * (py - y1) / (y0 - y1) + x1) in = !in;
                // the outline: within half a pixel of the edge
                const float ex = x0 - x1, ey = y0 - y1, l2 = ex * ex + ey * ey;
                float tt = l2 > 0.f ? ((px - x1) * ex + (py - y1) * ey) / l2 : 0.f;
                tt = sycl::clamp(tt, 0.f, 1.f);
                const float dx = x1 + tt * ex - px, dy = y1 + tt * ey - py;
                if (dx * dx + dy * dy <= 0.25f) in = true, on = true;
            }
            on = on || in;
        }
        mask[i] = on ? (uint8_t) 255 : (uint8_t) 0;
    });
    finish(stream, e, "aux_polygon_fill");
}

void aux_relu(const float* x, int64_t n, float* y, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = sycl::fmax(x[i], 0.f);
        if (y != nullptr) y[i] = v;
        if (y16 != nullptr) y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_relu");
}

void aux_add16(const float* a, const float* b, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[i] = f16_from_f32(a[i] + b[i]);
    });
    finish(stream, e, "aux_add16");
}

void aux_tile_add(float* x, const float* table, int64_t H, int64_t W, int64_t C, int64_t th, int64_t tw, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        x[i] += table[((py % th) * tw + px % tw) * C + c];
    });
    finish(stream, e, "aux_tile_add");
}

namespace {
inline float value_of(float v) { return v; }
inline float value_of(uint16_t v) { return f32_from_f16(v); }

// The 2D axial RoPE's cos and sin [grid^2][hd / 2][2] (a position's channel pairs: x for the first quarter, y for the
// second), made on the host once a GPU and shape and kept: their exp, cos and sin a pair of every row of every layer
// were most of SAM 3's ViT's split on the B70 (6.4 ms a frame, 2026-11-07).  The tables are the queue's own belonging
// (Runtime::queue_item), freed when its stream is destroyed - keyed by the bare queue address they would outlive the
// queue (an auxiliary model's, made again for the next tool's run) and meet the next one there.
struct RopeTables {
    explicit RopeTables(sycl::queue& q) : q(q) {}
    ~RopeTables() {
        for (auto& e : tables)
            if (e.second != nullptr) {
                sycl::free(e.second, q);
                core::graph_epoch_bump();   // a recorded step may hold the old one
            }
    }
    sycl::queue q;
    std::mutex mu;
    std::map<std::tuple<int64_t, float, int64_t, float>, float*> tables;
};
const float* rope_table(void* stream, int64_t grid, float pscale, int64_t hd, float theta) {
    sycl::queue& q = queue_for(stream);
    RopeTables& rt = core::Runtime::of_queue(q).queue_item<RopeTables>(q);
    std::lock_guard<std::mutex> lock(rt.mu);
    float*& t = rt.tables[{grid, pscale, hd, theta}];
    if (t != nullptr) return t;
    const int64_t half = hd / 2, quarter = hd / 4, gg = grid * grid;
    std::vector<float> h((size_t) (gg * half * 2));
    for (int64_t li = 0; li < gg; ++li)
        for (int64_t p = 0; p < half; ++p) {
            const int64_t f = kernels::qk::axial_frequency(p, quarter);
            const double pos = (double) (p < quarter ? li % grid : li / grid) * pscale;
            const double ang = kernels::qk::theta_angle(pos, f, (double) hd, 4,
                                                        [=](double e) { return std::pow((double) theta, e); });
            h[(size_t) ((li * half + p) * 2)] = (float) std::cos(ang);
            h[(size_t) ((li * half + p) * 2 + 1)] = (float) std::sin(ang);
        }
    t = sycl::malloc_device<float>(h.size(), q);
    if (t == nullptr) throw std::runtime_error("aux_rope2d: no GPU memory for its table");
    q.memcpy(t, h.data(), h.size() * 4).wait();
    return t;
}

// aux_rope2d_split16 over qkv rows of floats or of FP16 bits, `bias` [3C] added where not null
template <class In>
void rope2d_split16(const In* qkv, const float* bias, int64_t N, int64_t heads, int64_t hd, int64_t grid, float pscale,
                    float theta, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    const int64_t C = heads * hd, half = hd / 2;
    if (N <= 0 || half <= 0) return;
    const float* tab = rope_table(stream, grid, pscale, hd, theta);
    // a 2D range (rows, a row's pairs) and 32-bit indices
    const int hf = (int) half, gg = (int) (grid * grid);
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) N, (size_t) (heads * half)), [=](sycl::id<2> id) {
        const int64_t t = (int64_t) id[0];
        const int hp = (int) id[1], h = hp / hf, p = hp - h * hf;
        const int li = (int) (t % gg);
        const int ti = (li * hf + p) * 2;   // the table: grid^2 x half pairs, far below 2^31
        const float cs = tab[ti], sn = tab[ti + 1];
        const int p2 = 2 * p;   // the pair's first value: 32-bit, as the device counts
        const int64_t o = t * C + h * hd + p2, c0 = h * hd + p2;
        auto at = [&](int64_t part, int64_t j) {
            return value_of(qkv[(t * 3 + part) * C + c0 + j]) +
                   (bias != nullptr ? bias[part * C + c0 + j] : 0.f);
        };
        const float q0 = at(0, 0), q1 = at(0, 1), k0 = at(1, 0), k1 = at(1, 1);
        q16[o] = f16_from_f32(q0 * cs - q1 * sn);
        q16[o + 1] = f16_from_f32(q1 * cs + q0 * sn);
        k16[o] = f16_from_f32(k0 * cs - k1 * sn);
        k16[o + 1] = f16_from_f32(k1 * cs + k0 * sn);
        v16[o] = f16_from_f32(at(2, 0));
        v16[o + 1] = f16_from_f32(at(2, 1));
    });
    finish(stream, e, "aux_rope2d_split16");
}
}  // namespace

void aux_rope2d_split16(const float* qkv, int64_t N, int64_t heads, int64_t hd, int64_t grid, float pscale, float theta,
                        uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    rope2d_split16(qkv, nullptr, N, heads, hd, grid, pscale, theta, q16, k16, v16, stream);
}

void aux_rope2d_split16h(const uint16_t* qkv16, const float* bias, int64_t N, int64_t heads, int64_t hd, int64_t grid,
                         float pscale, float theta, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    rope2d_split16(qkv16, bias, N, heads, hd, grid, pscale, theta, q16, k16, v16, stream);
}

void aux_bias_act16(const uint16_t* x16, const float* bias, int64_t T, int64_t n, ImageAct act, uint16_t* y16,
                    void* stream) {
    ProductStore p;
    p.in16 = x16; p.bias = bias; p.rows = T; p.cols = n; p.act = act; p.out16 = y16;
    product_store(p, stream);
}

void aux_narrow_bias16(const float* y, const float* bias, int64_t T, int64_t n, uint16_t* y16, void* stream) {
    ProductStore p;
    p.in = y; p.bias = bias; p.rows = T; p.cols = n; p.out16 = y16;
    product_store(p, stream);
}

void aux_deconv2_scatter(const float* y, int64_t H, int64_t W, int64_t O, const float* bias, bool gelu, float* out,
                         uint16_t* out16, void* stream) {
    const int64_t n = 4 * H * W * O;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        // i over out's [2H][2W][O]
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % O, p = i / O, oy = p / (2 * W), ox = p - oy * (2 * W);
        const int64_t src = ((oy / 2) * W + ox / 2) * 4 * O + ((oy % 2) * 2 + ox % 2) * O + o;
        float v = y[src] + (bias != nullptr ? bias[o] : 0.f);
        if (gelu) v = 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f));
        if (out != nullptr) out[i] = v;
        if (out16 != nullptr) out16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_deconv2_scatter");
}

void aux_up_add16(const float* prev, int64_t h, int64_t w, const float* feat, int64_t H, int64_t W, int64_t C,
                  uint16_t* y16, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        const int64_t sy = sycl::min(py * h / H, h - 1), sx = sycl::min(px * w / W, w - 1);
        y16[i] = f16_from_f32(feat[i] + prev[(sy * w + sx) * C + c]);
    });
    finish(stream, e, "aux_up_add16");
}

void aux_mask_resize(const float* logits, int64_t K, int64_t h, int64_t w, int64_t H, int64_t W, bool as_logits,
                     float threshold, uint8_t* masks, void* stream) {
    const int64_t n = K * H * W;
    if (n <= 0) return;
    const float sy = (float) h / (float) H, sx = (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t k = i / (H * W), p = i - k * H * W, py = p / W, px = p - py * W;
        const float fy = sycl::fmax(((float) py + 0.5f) * sy - 0.5f, 0.f);
        const float fx = sycl::fmax(((float) px + 0.5f) * sx - 0.5f, 0.f);
        const int64_t y0 = sycl::min((int64_t) fy, h - 1), x0 = sycl::min((int64_t) fx, w - 1);
        const int64_t y1 = sycl::min(y0 + 1, h - 1), x1 = sycl::min(x0 + 1, w - 1);
        const float ay = fy - (float) y0, ax = fx - (float) x0;
        const float* l = logits + k * h * w;
        auto sg = [as_logits](float v) { return as_logits ? v : 1.f / (1.f + sycl::exp(-v)); };
        const float v = (1.f - ay) * ((1.f - ax) * sg(l[y0 * w + x0]) + ax * sg(l[y0 * w + x1])) +
                        ay * ((1.f - ax) * sg(l[y1 * w + x0]) + ax * sg(l[y1 * w + x1]));
        masks[i] = v > threshold ? (uint8_t) 255 : (uint8_t) 0;
    });
    finish(stream, e, "aux_mask_resize");
}

void aux_box_bias(const float* boxes, int64_t Q, int64_t H, int64_t W, const float* mlp, int64_t hidden, int64_t heads,
                  float* tmp, float* bias, void* stream) {
    if (Q <= 0) return;
    auto& q = queue_for(stream);
    const int64_t R = H + W, per = hidden * 2 + hidden + heads * hidden + heads;
    // tmp [Q][R][heads]: the rows' MLP (i < H) and the columns' (H + j)
    auto e = q.parallel_for(sycl::range<1>((size_t) (Q * R)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t qq = i / R, r = i - qq * R;
        const bool rows = r < H;
        const float c = rows ? (float) r / (float) H : (float) (r - H) / (float) W;
        const float* bx = boxes + qq * 4;
        float d[2] = {c - (rows ? bx[1] : bx[0]), c - (rows ? bx[3] : bx[2])};
        for (float& v : d) {
            v *= 8.f;
            v = (v > 0.f ? 1.f : v < 0.f ? -1.f : 0.f) * sycl::log2(sycl::fabs(v) + 1.f) / 3.f;
        }
        const float* m = mlp + (rows ? 0 : per);
        const float *w1 = m, *b1 = m + hidden * 2, *w2 = b1 + hidden, *b2 = w2 + heads * hidden;
        float out[16];
        for (int64_t o = 0; o < heads && o < 16; ++o) out[o] = b2[o];
        for (int64_t k = 0; k < hidden; ++k) {
            const float a = sycl::fmax(w1[2 * k] * d[0] + w1[2 * k + 1] * d[1] + b1[k], 0.f);
            for (int64_t o = 0; o < heads && o < 16; ++o) out[o] += w2[o * hidden + k] * a;
        }
        for (int64_t o = 0; o < heads && o < 16; ++o) tmp[i * heads + o] = out[o];
    });
    if (bias == nullptr) {
        finish(stream, e, "aux_box_bias");
        return;
    }
    if (!stream) core::Runtime::get().wait(e, "aux_box_bias");
    const int64_t HW = H * W, n = heads * (Q + 1) * HW;
    e = q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i % HW, hq = i / HW, row = hq % (Q + 1), h = hq / (Q + 1);
        if (row == 0) {
            bias[i] = 0.f;
            return;
        }
        const int64_t qq = row - 1, py = p / W, px = p - py * W;
        bias[i] = tmp[(qq * R + py) * heads + h] + tmp[(qq * R + H + px) * heads + h];
    });
    finish(stream, e, "aux_box_bias");
}

void aux_merge2(const float* x, int64_t H, int64_t W, int64_t C, float* y, void* stream) {
    const int64_t h = (H + 1) / 2, w = (W + 1) / 2, n = h * w * 4 * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c4 = i % (4 * C), p = i / (4 * C), part = c4 / C, c = c4 - part * C;
        const int64_t py = p / w, px = p - py * w;
        const int64_t sy = 2 * py + part % 2, sx = 2 * px + part / 2;   // x0 (0, 0), x1 (1, 0), x2 (0, 1), x3 (1, 1)
        y[i] = sy < H && sx < W ? x[(sy * W + sx) * C + c] : 0.f;
    });
    finish(stream, e, "aux_merge2");
}

void aux_resize_into(const float* src, int64_t h, int64_t w, int64_t C, float* dst, int64_t H, int64_t W, int64_t ld,
                     int64_t c0, bool align_corners, bool add, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const bool same = h == H && w == W;
    // the source coordinate of a destination pixel: align_corners maps the corners' centers onto each other
    const float ry = align_corners ? (H > 1 ? (float) (h - 1) / (float) (H - 1) : 0.f) : (float) h / (float) H;
    const float rx = align_corners ? (W > 1 ? (float) (w - 1) / (float) (W - 1) : 0.f) : (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        float v;
        if (same) {
            v = src[i];
        } else {
            const float fy = align_corners ? (float) py * ry : sycl::fmax(((float) py + 0.5f) * ry - 0.5f, 0.f);
            const float fx = align_corners ? (float) px * rx : sycl::fmax(((float) px + 0.5f) * rx - 0.5f, 0.f);
            const int64_t y0 = sycl::min((int64_t) fy, h - 1), x0 = sycl::min((int64_t) fx, w - 1);
            const int64_t y1 = sycl::min(y0 + 1, h - 1), x1 = sycl::min(x0 + 1, w - 1);
            const float ay = fy - (float) y0, ax = fx - (float) x0;
            v = (1.f - ay) * ((1.f - ax) * src[(y0 * w + x0) * C + c] + ax * src[(y0 * w + x1) * C + c]) +
                ay * ((1.f - ax) * src[(y1 * w + x0) * C + c] + ax * src[(y1 * w + x1) * C + c]);
        }
        float& d = dst[p * ld + c0 + c];
        d = add ? d + v : v;
    });
    finish(stream, e, "aux_resize_into");
}

void aux_deform_col16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int pad, const float* om,
                      int64_t p0, int64_t n, uint16_t* col16, void* stream) {
    const int64_t k2 = (int64_t) k * k, K = k2 * C;
    if (n <= 0 || K <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * K)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / K, q = i - r * K, tap = q / C, c = q - tap * C;
        const int64_t p = p0 + r, oy = p / W, ox = p - oy * W;
        const float* o = om + p * 3 * k2;
        const int64_t ty = tap / k, tx = tap - ty * k;   // the tap's row and column in the kernel
        const float y = (float) (oy - pad + ty) + o[2 * tap], x = (float) (ox - pad + tx) + o[2 * tap + 1];
        const float m = 2.f / (1.f + sycl::exp(-o[2 * k2 + tap]));
        float v = 0.f;
        // torchvision's bilinear_interpolate: nothing outside (-1, H) x (-1, W), the corners outside it zero
        if (y > -1.f && y < (float) H && x > -1.f && x < (float) W) {
            const float fy = sycl::floor(y), fx = sycl::floor(x);
            const int64_t yl = (int64_t) fy, xl = (int64_t) fx, yh = yl + 1, xh = xl + 1;
            const float ly = y - fy, lx = x - fx, hy = 1.f - ly, hx = 1.f - lx;
            auto at = [&](int64_t yy, int64_t xx) {
                return yy >= 0 && yy < H && xx >= 0 && xx < W ? f32_from_f16(x16[(yy * W + xx) * C + c]) : 0.f;
            };
            v = hy * hx * at(yl, xl) + hy * lx * at(yl, xh) + ly * hx * at(yh, xl) + ly * lx * at(yh, xh);
        }
        col16[i] = f16_from_f32(v * m);
    });
    finish(stream, e, "aux_deform_col16");
}

void aux_bias_act(float* y, int64_t rows, int64_t n, int64_t ld, const float* bias, bool relu, void* stream) {
    if (rows <= 0 || n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * n)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / n, c = i - r * n;
        float& v = y[r * ld + c];
        float t = v + (bias != nullptr ? bias[c] : 0.f);
        if (relu) t = sycl::fmax(t, 0.f);
        v = t;
    });
    finish(stream, e, "aux_bias_act");
}

void aux_col_mean(const float* x, int64_t P, int64_t C, float* mean, void* stream) {
    if (P <= 0 || C <= 0) return;
    constexpr size_t L = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) C * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto c = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            float s = 0.f;
            for (int64_t p = l; p < P; p += (int64_t) L) s += x[p * C + c];
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (l == 0) mean[c] = s / (float) P;
        });
    finish(stream, e, "aux_col_mean");
}

void aux_scale_by_sigmoid(float* x, int64_t P, int64_t C, const float* logit, void* stream) {
    if (P <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] *= 1.f / (1.f + sycl::exp(-logit[i / C]));
    });
    finish(stream, e, "aux_scale_by_sigmoid");
}

void aux_image_patches16(const float* img, int64_t SH, int64_t SW, int64_t g, uint16_t* y16, void* stream) {
    const int64_t sh = SH / g, sw = SW / g, ch = 3 * g * g, n = sh * sw * ch;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i % ch, p = i / ch, py = p / sw, px = p - py * sw;
        const int64_t c = q / (g * g), gy = (q / g) % g, gx = q % g;
        y16[i] = f16_from_f32(img[((gy * sh + py) * SW + gx * sw + px) * 3 + c]);
    });
    finish(stream, e, "aux_image_patches16");
}

void aux_rgb_norm(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, float* y,
                  void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float m0 = mean[0], m1 = mean[1], m2 = mean[2], s0 = 1.f / std[0], s1 = 1.f / std[1], s2 = 1.f / std[2];
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y[3 * i] = ((float) rgba[4 * i] / 255.f - m0) * s0;
        y[3 * i + 1] = ((float) rgba[4 * i + 1] / 255.f - m1) * s1;
        y[3 * i + 2] = ((float) rgba[4 * i + 2] / 255.f - m2) * s2;
    });
    finish(stream, e, "aux_rgb_norm");
}

void aux_alpha_bicubic(const float* logits, int64_t h, int64_t w, int64_t W, int64_t H, uint8_t* alpha, void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float sy = (float) h / (float) H, sx = (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t py = i / W, px = i - py * W;
        // torch's upsample_bicubic2d: the source coordinate unclamped, the four taps' indices clamped to the edge
        const float fy = ((float) py + 0.5f) * sy - 0.5f, fx = ((float) px + 0.5f) * sx - 0.5f;
        const float by = sycl::floor(fy), bx = sycl::floor(fx), ty = fy - by, tx = fx - bx;
        auto cubic = [](float t, float* wt) {
            constexpr float A = -0.75f;
            auto c1 = [](float x) { return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f; };
            auto c2 = [](float x) { return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A; };
            wt[0] = c2(t + 1.f), wt[1] = c1(t), wt[2] = c1(1.f - t), wt[3] = c2(2.f - t);
        };
        float wy[4], wx[4];
        cubic(ty, wy);
        cubic(tx, wx);
        float v = 0.f;
        for (int a = 0; a < 4; ++a) {
            const int64_t yy = sycl::clamp((int64_t) by - 1 + a, (int64_t) 0, h - 1);
            float r = 0.f;
            for (int b = 0; b < 4; ++b) {
                const int64_t xx = sycl::clamp((int64_t) bx - 1 + b, (int64_t) 0, w - 1);
                r += wx[b] * logits[yy * w + xx];
            }
            v += wy[a] * r;
        }
        alpha[i] = (uint8_t) sycl::round(255.f / (1.f + sycl::exp(-v)));
    });
    finish(stream, e, "aux_alpha_bicubic");
}

void aux_fg_stack(const uint8_t* rgba, const uint8_t* alpha, const float* fb, int64_t n, float* x, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float a = (float) alpha[i] / 255.f;
        float* o = x + i * 7;
        o[0] = a;
        for (int c = 0; c < 3; ++c) {
            const float img = (float) rgba[4 * i + c] / 255.f;
            const float f = fb != nullptr ? fb[i * 6 + c] : img, b = fb != nullptr ? fb[i * 6 + 3 + c] : img;
            o[1 + c] = f * a;
            o[4 + c] = b * (1.f - a);
        }
    });
    finish(stream, e, "aux_fg_stack");
}

void aux_box_mean(const float* x, int64_t H, int64_t W, int64_t C, int64_t k, float* tmp, float* y, void* stream) {
    if (H <= 0 || W <= 0 || C <= 0 || k <= 0) return;
    const int64_t lo = k % 2 == 0 ? k / 2 - 1 : k / 2, hi = k / 2;
    const float inv = 1.f / (float) k;
    sycl::queue& q = queue_for(stream);
    // a running sum along each row (a work-item a row and channel), then along each column (one a column and channel:
    // neighbours read neighbouring bytes).  A window of zeros gives 0 exactly, not the sum's rounding left over from
    // the values that left it: the foreground divides the means by the alpha's mean plus 1e-5, which would make that
    // residue a color where the picture is transparent
    auto e = q.parallel_for(sycl::range<1>((size_t) (H * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / C, c = i - r * C;
        const float* in = x + r * W * C + c;
        float* out = tmp + r * W * C + c;
        auto at = [&](int64_t j) { return in[sycl::clamp(j, (int64_t) 0, W - 1) * C]; };
        float sum = 0.f;
        int64_t nz = 0;
        for (int64_t j = -lo; j <= hi; ++j) sum += at(j), nz += at(j) != 0.f;
        for (int64_t j = 0; j < W; ++j) {
            out[j * C] = nz != 0 ? sum * inv : 0.f;
            const float in1 = at(j + hi + 1), out1 = at(j - lo);
            sum += in1 - out1;
            nz += (in1 != 0.f) - (out1 != 0.f);
        }
    });
    finish(stream, e, "aux_box_mean rows");
    e = q.parallel_for(sycl::range<1>((size_t) (W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float* in = tmp + i;
        float* out = y + i;
        const int64_t row = W * C;
        auto at = [&](int64_t j) { return in[sycl::clamp(j, (int64_t) 0, H - 1) * row]; };
        float sum = 0.f;
        int64_t nz = 0;
        for (int64_t j = -lo; j <= hi; ++j) sum += at(j), nz += at(j) != 0.f;
        for (int64_t j = 0; j < H; ++j) {
            out[j * row] = nz != 0 ? sum * inv : 0.f;
            const float in1 = at(j + hi + 1), out1 = at(j - lo);
            sum += in1 - out1;
            nz += (in1 != 0.f) - (out1 != 0.f);
        }
    });
    finish(stream, e, "aux_box_mean columns");
}

void aux_fg_combine(const uint8_t* rgba, const uint8_t* alpha, const float* y, int64_t n, float* fb, uint8_t* rgb,
                    void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float a = (float) alpha[i] / 255.f;
        const float* m = y + i * 7;
        const float da = m[0] + 1e-5f, db = (1.f - m[0]) + 1e-5f;
        for (int c = 0; c < 3; ++c) {
            const float img = (float) rgba[4 * i + c] / 255.f, mf = m[1 + c] / da, mb = m[4 + c] / db;
            const float f = sycl::clamp(mf + a * (img - a * mf - (1.f - a) * mb), 0.f, 1.f);
            if (fb != nullptr) {
                fb[i * 6 + c] = f;
                fb[i * 6 + 3 + c] = mb;
            }
            if (rgb != nullptr) rgb[i * 3 + c] = (uint8_t) (f * 255.f);
        }
    });
    finish(stream, e, "aux_fg_combine");
}

void aux_box_sine16(const float* boxes, int64_t Q, int64_t F, uint16_t* sin16, float* xyxy, void* stream) {
    if (Q <= 0 || F <= 0) return;
    const float two_pi = 6.28318530717958647692f;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) Q, (size_t) (4 * F)), [=](sycl::id<2> id) {
        const auto q = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        const float* b = boxes + q * 4;
        const int64_t part = c / F, i = c - part * F, pair = i / 2;
        const float v = part == 0 ? b[1] : part == 1 ? b[0] : part == 2 ? b[2] : b[3];   // y, x, w, h
        const float p = v * two_pi / sycl::pow(10000.f, (float) (2 * pair) / (float) F);
        sin16[q * 4 * F + c] = f16_from_f32(i % 2 == 0 ? sycl::sin(p) : sycl::cos(p));
        if (c < 4) {
            const float h = 0.5f * (c % 2 == 0 ? b[2] : b[3]);
            xyxy[q * 4 + c] = (c % 2 == 0 ? b[0] : b[1]) + (c < 2 ? -h : h);
        }
    });
    finish(stream, e, "aux_box_sine16");
}

void aux_box_refine(float* boxes, const float* delta, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float x = sycl::clamp(boxes[i], 0.f, 1.f);
        const float inv = sycl::log(sycl::fmax(x, 1e-3f) / sycl::fmax(1.f - x, 1e-3f));
        boxes[i] = 1.f / (1.f + sycl::exp(-(delta[i] + inv)));
    });
    finish(stream, e, "aux_box_refine");
}

void aux_dwconv(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C, int k, float* y,
                void* stream) {
    Depthwise d; d.x = x; d.w = w; d.bias = b; d.out = y;
    d.h = d.oh = H; d.wdim = d.ow = W; d.c = C; d.k = k; d.pad = k / 2;
    image_depthwise(d, stream);
}

void aux_scale_add(float* x, const float* y, const float* g, int64_t n, int64_t C, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] += g[i % C] * y[i];
    });
    finish(stream, e, "aux_scale_add");
}

void aux_fill(float* x, int64_t n, float v, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { x[id[0]] = v; });
    finish(stream, e, "aux_fill");
}

namespace {
// the 2D axial RoPE's turn of a pair (a, b) of channel 2 p of a head of hd at grid position li (aux_rope2d_split16's)
inline void rope_pair(float& a, float& b, int64_t p, int64_t hd, int64_t li, int64_t grid, float lt) {
    const int64_t quarter = hd / 4, f = kernels::qk::axial_frequency(p, quarter);
    const float pos = (float) (p < quarter ? li % grid : li / grid);
    const float ang = kernels::qk::theta_angle(pos, f, (float) hd, 4,
                                               [=](float e) { return sycl::exp(e * lt); });
    const float cs = sycl::cos(ang), sn = sycl::sin(ang), a0 = a;
    a = a0 * cs - b * sn;
    b = b * cs + a0 * sn;
}
}  // namespace

void aux_rope2d16(const float* y, const float* add, int64_t P, int64_t N, int64_t heads, int64_t hd, int64_t grid,
                  float theta, uint16_t* y16, void* stream) {
    const int64_t C = heads * hd, half = hd / 2;
    if (N <= 0) return;
    const float lt = sycl::log(theta);
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (N * heads * half)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i % half, th = i / half, r = th / heads, h = th - r * heads, c = h * hd + 2 * p;
        float a = y[r * C + c], b = y[r * C + c + 1];
        if (add != nullptr) {
            a += add[(r % P) * C + c];
            b += add[(r % P) * C + c + 1];
        }
        rope_pair(a, b, p, hd, r % (grid * grid), grid, lt);
        y16[r * C + c] = f16_from_f32(a);
        y16[r * C + c + 1] = f16_from_f32(b);
    });
    finish(stream, e, "aux_rope2d16");
}

void aux_mem_key16(const float* y, int64_t B, int64_t N, int64_t nsp, int64_t P, const float* sine, const float* tpos,
                   const float* ptr_pos, int64_t per, int64_t heads, int64_t hd, int64_t grid, float theta,
                   uint16_t* k16, void* stream) {
    const int64_t C = heads * hd, half = hd / 2;
    if (B * N <= 0) return;
    const float lt = sycl::log(theta);
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (B * N * heads * half)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i % half, th = i / half, row = th / heads, h = th - row * heads, c = h * hd + 2 * p;
        const int64_t r = row % N;
        float a = y[row * C + c], b = y[row * C + c + 1];
        if (r < nsp) {
            const int64_t li = r % P, f = r / P, sr = tpos != nullptr ? li : r;
            a += sine[sr * C + c] + (tpos != nullptr ? tpos[f * C + c] : 0.f);
            b += sine[sr * C + c + 1] + (tpos != nullptr ? tpos[f * C + c + 1] : 0.f);
            rope_pair(a, b, p, hd, li % (grid * grid), grid, lt);
        } else {
            const int64_t j = (r - nsp) / per;
            a += ptr_pos[j * C + c];
            b += ptr_pos[j * C + c + 1];
        }
        k16[row * C + c] = f16_from_f32(a);
        k16[row * C + c + 1] = f16_from_f32(b);
    });
    finish(stream, e, "aux_mem_key16");
}

void aux_mux_masks16(const float* masks, int64_t N, int64_t h, int64_t w, int64_t H, int64_t W, bool sigmoid_first,
                     const float* cond, const float* valid, int64_t M, int64_t B, uint16_t* y16, void* stream,
                     const int32_t* live, int64_t L) {
    if (B * H * W <= 0) return;
    if (live != nullptr) {   // one bucket's live slots alone
        M = L;
        B = 1;
    }
    const float sy = (float) h / (float) H, sx = (float) w / (float) W;
    const int hh = (int) H, ww = (int) W, mh = (int) h, mw = (int) w, m2 = (int) (2 * M), mm = (int) M, n = (int) N;
    // a work-item a pixel's channel, the channels innermost so neighbouring items write neighbouring halves; a 3D
    // range and 32-bit indices (the B70 1.7 ms a 1152 x 1152 x 32 in the 1D form with 64-bit ones, 1.1 ms with a
    // pixel's channels an item, 2026-11-07)
    const auto e = queue_for(stream).parallel_for(sycl::range<3>((size_t) (B * H), (size_t) W, (size_t) (2 * M)),
                                                  [=](sycl::id<3> id) {
        const int row = (int) id[0], ox = (int) id[1], ch = (int) id[2], b = row / hh, oy = row - b * hh;
        const int s = ch < mm ? ch : ch - mm, o = live != nullptr ? live[s] : b * mm + s;
        float v = 0.f;
        if (live != nullptr || (o < n && (valid == nullptr || valid[o] > 0.f))) {
            if (ch >= mm) {
                v = cond[o];
            } else {
                // torch's bilinear, align_corners false: the source position clamped at 0
                const float fy = sycl::fmax(((float) oy + 0.5f) * sy - 0.5f, 0.f);
                const float fx = sycl::fmax(((float) ox + 0.5f) * sx - 0.5f, 0.f);
                const int y0 = sycl::min((int) fy, mh - 1), x0 = sycl::min((int) fx, mw - 1);
                const int y1 = sycl::min(y0 + 1, mh - 1), x1 = sycl::min(x0 + 1, mw - 1);
                const float wy = fy - (float) y0, wx = fx - (float) x0;
                const float* m = masks + (size_t) o * mh * mw;
                const auto at = [&](int yy, int xx) {
                    const float l = m[yy * mw + xx];
                    return sigmoid_first ? 1.f / (1.f + sycl::exp(-l)) : l;
                };
                float t = (1.f - wy) * ((1.f - wx) * at(y0, x0) + wx * at(y0, x1)) +
                          wy * ((1.f - wx) * at(y1, x0) + wx * at(y1, x1));
                if (!sigmoid_first) t = 1.f / (1.f + sycl::exp(-t));
                v = 2.f * t - 1.f;
            }
        }
        y16[((size_t) row * ww + ox) * m2 + ch] = f16_from_f32(v);
    });
    finish(stream, e, "aux_mux_masks16");
}

void aux_binarize(const float* x, int64_t n, float* y, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        y[id[0]] = x[id[0]] > 0.f ? 1.f : 0.f;
    });
    finish(stream, e, "aux_binarize");
}

void aux_mod_select16(const float* src, int64_t T, int64_t R, int64_t C, uint16_t* y16, void* stream) {
    if (R * C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / C;
        y16[i] = f16_from_f32(src[((r % T) * R) * C + i]);
    });
    finish(stream, e, "aux_mod_select16");
}

void aux_map_store16(const float* y, int64_t n, int64_t O, const float* bias, ImageAct act, const uint16_t* res16,
                     uint16_t* dst16, int64_t ld, int64_t off, void* stream) {
    ProductStore p;
    p.in = y; p.rows = n; p.cols = O; p.bias = bias; p.act = act; p.residual = res16;
    p.out16 = dst16; p.out_stride = ld; p.offset = off;
    product_store(p, stream);
}

void aux_dwconv16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, const uint16_t* w16, const float* bias, int k,
                  int s, int p, int64_t OH, int64_t OW, ImageAct act, uint16_t* y16, void* stream) {
    Depthwise d; d.x16 = x16; d.w16 = w16; d.bias = bias; d.out16 = y16;
    d.h = H; d.wdim = W; d.c = C; d.oh = OH; d.ow = OW;
    d.k = k; d.stride = s; d.pad = p; d.act = act;
    image_depthwise(d, stream);
}

void aux_maxpool16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int s, int p, int64_t OH, int64_t OW,
                   uint16_t* y16, void* stream, int64_t ld, int64_t off, uint16_t* copy) {
    if (ld == 0) ld = C;
    if (OH <= 0 || OW <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, q = i / C, oy = q / OW, ox = q - oy * OW;
        float m = -INFINITY;
        for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx) {
                const int64_t yy = oy * s + dy - p, xx = ox * s + dx - p;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W) m = sycl::fmax(m, f32_from_f16(x16[(yy * W + xx) * C + c]));
            }
        const uint16_t v = f16_from_f32(m);
        y16[q * ld + off + c] = v;
        if (copy != nullptr) copy[i] = v;
    });
    finish(stream, e, "aux_maxpool16");
}


void aux_letterbox16(const uint8_t* src, int64_t w, int64_t h, int64_t nw, int64_t nh, int64_t left, int64_t top,
                     int64_t iw, int64_t ih, float ratio, bool focus, uint16_t* out, void* stream) {
    const int64_t ow = focus ? iw / 2 : iw, oh = focus ? ih / 2 : ih, channels = focus ? 16 : 8;
    const float sx = (float) w / (float) nw, sy = (float) h / (float) nh;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (ow * oh * channels)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t ch = i % channels, p = i / channels, y = p / ow, x = p % ow;
        if (ch >= (focus ? 12 : 3)) { out[i] = 0; return; }
        const int64_t g = ch / 3, color = focus ? 2 - ch % 3 : ch;
        const int64_t u = (focus ? 2 * x + (g >= 2 ? 1 : 0) : x) - left;
        const int64_t v = (focus ? 2 * y + (g == 1 || g == 3 ? 1 : 0) : y) - top;
        float val = 114.f;
        if (u >= 0 && u < nw && v >= 0 && v < nh) {
            float fx = focus ? sycl::clamp(((float) u + 0.5f) / ratio - 0.5f, 0.f, (float) (w - 1))
                             : ((float) u + 0.5f) * sx - 0.5f;
            float fy = focus ? sycl::clamp(((float) v + 0.5f) / ratio - 0.5f, 0.f, (float) (h - 1))
                             : ((float) v + 0.5f) * sy - 0.5f;
            int64_t x0 = (int64_t) sycl::floor(fx), y0 = (int64_t) sycl::floor(fy);
            fx -= (float) x0; fy -= (float) y0;
            if (x0 < 0) { x0 = 0; fx = 0.f; }
            if (y0 < 0) { y0 = 0; fy = 0.f; }
            if (x0 >= w - 1) { x0 = w - 1; fx = 0.f; }
            if (y0 >= h - 1) { y0 = h - 1; fy = 0.f; }
            const int64_t x1 = sycl::min(x0 + 1, w - 1), y1 = sycl::min(y0 + 1, h - 1);
            const int64_t stride = focus ? 3 : 4;
            const float a = src[(y0 * w + x0) * stride + color], b = src[(y0 * w + x1) * stride + color];
            const float c = src[(y1 * w + x0) * stride + color], d = src[(y1 * w + x1) * stride + color];
            if (focus) val = sycl::rint((a * (1.f - fx) + b * fx) * (1.f - fy) + (c * (1.f - fx) + d * fx) * fy);
            else {
                const float t = a + (b - a) * fx, bt = c + (d - c) * fx;
                val = sycl::round(t + (bt - t) * fy);
            }
        }
        out[i] = f16_from_f32(focus ? val : val / 255.f);
    });
    finish(stream, e, "aux_letterbox16");
}

void aux_concat2_16(const uint16_t* a, int64_t ca, bool up, const uint16_t* b, int64_t cb, int64_t h, int64_t w,
                    uint16_t* dst, void* stream) {
    const int64_t c = ca + cb;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (h * w * c)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / c, ch = i - q * c;
        if (ch < ca) {
            const int64_t src = up ? ((q / w) / 2) * (w / 2) + (q % w) / 2 : q;
            dst[i] = a[src * ca + ch];
        } else {
            dst[i] = b[q * cb + ch - ca];
        }
    });
    finish(stream, e, "aux_concat2_16");
}

void aux_bilinear16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t OH, int64_t OW, bool align_corners,
                    bool add, uint16_t* y16, void* stream) {
    if (OH <= 0 || OW <= 0 || C <= 0) return;
    const float sy = align_corners ? (OH > 1 ? (float) (H - 1) / (float) (OH - 1) : 0.f) : (float) H / (float) OH;
    const float sx = align_corners ? (OW > 1 ? (float) (W - 1) / (float) (OW - 1) : 0.f) : (float) W / (float) OW;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, q = i / C, oy = q / OW, ox = q - oy * OW;
        float fy = align_corners ? (float) oy * sy : sycl::fmax(((float) oy + 0.5f) * sy - 0.5f, 0.f);
        float fx = align_corners ? (float) ox * sx : sycl::fmax(((float) ox + 0.5f) * sx - 0.5f, 0.f);
        const int64_t y0 = sycl::min((int64_t) fy, H - 1), x0 = sycl::min((int64_t) fx, W - 1);
        const int64_t y1 = sycl::min(y0 + 1, H - 1), x1 = sycl::min(x0 + 1, W - 1);
        fy -= (float) y0;
        fx -= (float) x0;
        const auto at = [&](int64_t yy, int64_t xx) { return f32_from_f16(x16[(yy * W + xx) * C + c]); };
        const float top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
        const float bot = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
        float v = top + (bot - top) * fy;
        if (add) v += f32_from_f16(y16[i]);
        y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_bilinear16");
}

void aux_deconv_scatter16(const float* y, int64_t H, int64_t W, int64_t k, int64_t O, const float* bias, ImageAct act,
                          uint16_t* y16, void* stream) {
    if (H <= 0 || W <= 0 || O <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (k * k * H * W * O)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % O, q = i / O, Y = q / (k * W), X = q - Y * (k * W);
        const int64_t src = ((Y / k) * W + X / k) * k * k + (Y % k) * k + (X % k);
        y16[i] = f16_from_f32(image_activate(act, y[src * O + o] + (bias != nullptr ? bias[o] : 0.f)));
    });
    finish(stream, e, "aux_deconv_scatter16");
}

void aux_copy_channels16(const uint16_t* src16, int64_t n, int64_t C, uint16_t* dst16, int64_t ld, int64_t off,
                         void* stream) {
    if (n <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / C;
        dst16[q * ld + off + (i - q * C)] = src16[i];
    });
    finish(stream, e, "aux_copy_channels16");
}

void aux_add16h(const uint16_t* a16, const uint16_t* b16, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[i] = f16_from_f32(f32_from_f16(a16[i]) + f32_from_f16(b16[i]));
    });
    finish(stream, e, "aux_add16h");
}

void aux_pad_reflect16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t p, uint16_t* y16, void* stream) {
    const int64_t OH = H + 2 * p, OW = W + 2 * p;
    if (H <= 0 || W <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, q = i / C;
        int64_t y = q / OW - p, x = q % OW - p;
        y = y < 0 ? -y : y >= H ? 2 * (H - 1) - y : y;
        x = x < 0 ? -x : x >= W ? 2 * (W - 1) - x : x;
        y16[i] = x16[(y * W + x) * C + c];
    });
    finish(stream, e, "aux_pad_reflect16");
}

void aux_zero_insert16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t s, int64_t lo, int64_t hi,
                       uint16_t* y16, void* stream) {
    const int64_t OH = (H - 1) * s + 1 + lo + hi, OW = (W - 1) * s + 1 + lo + hi;
    if (H <= 0 || W <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, q = i / C, y = q / OW - lo, x = q % OW - lo;
        const bool on = y >= 0 && x >= 0 && y % s == 0 && x % s == 0 && y / s < H && x / s < W;
        y16[i] = on ? x16[((y / s) * W + x / s) * C + c] : (uint16_t) 0;
    });
    finish(stream, e, "aux_zero_insert16");
}

void aux_scale_channels16(uint16_t* x16, int64_t n, int64_t C, const float* gate, void* stream) {
    if (n <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x16[i] = f16_from_f32(f32_from_f16(x16[i]) * gate[i % C]);
    });
    finish(stream, e, "aux_scale_channels16");
}

void aux_col_mean16(const uint16_t* x16, int64_t n, int64_t C, float* mean, void* stream) {
    if (n <= 0 || C <= 0) return;
    sycl::queue& q = queue_for(stream);
    const size_t wg = std::min<size_t>(256, q.get_device().get_info<sycl::info::device::max_work_group_size>());
    const auto e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) C * wg), sycl::range<1>(wg)),
                                  [=](sycl::nd_item<1> it) {
        const auto c = static_cast<int64_t>(it.get_group(0));
        float s = 0.f;
        for (auto p = static_cast<int64_t>(it.get_local_id(0)); p < n; p += (int64_t) wg) s += f32_from_f16(x16[p * C + c]);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) mean[c] = s / (float) n;
    });
    finish(stream, e, "aux_col_mean16");
}

void aux_pad16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t top, int64_t left, int64_t bottom,
               int64_t right, uint16_t* y16, void* stream) {
    const int64_t OH = H + top + bottom, OW = W + left + right;
    if (OH <= 0 || OW <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (OH * OW * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, q = i / C, y = q / OW - top, x = q % OW - left;
        y16[i] = y >= 0 && y < H && x >= 0 && x < W ? x16[(y * W + x) * C + c] : (uint16_t) 0;
    });
    finish(stream, e, "aux_pad16");
}

}  // namespace strata::image
