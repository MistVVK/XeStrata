// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/image_depthwise.cpp - image_depthwise (include/strata/image/aux_kernels.hpp).  A source of its own, so its
// kernels are a device image of their own: a VAE whose decode is the first user (Mage-Flow's) loads these alone.  In
// aux_kernels.cpp the first call loaded that file's whole image, 684 KB on the RX 9060 XT, and took Mage-Flow's first
// 512-square decode from 30 to 38 ms (2026-11-11).

#include "strata/image/aux_kernels.hpp"

#include "strata/core/runtime.hpp"

#include "row_norm.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::image {

namespace {

using rows::kRowVals;
using rows::row_items;
using rows::row_stats;
using rows::row_vals;

inline uint16_t f16_from_f32(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }
inline float f32_from_f16(uint16_t b) { return static_cast<float>(sycl::bit_cast<sycl::half>(b)); }

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

namespace {
// One tap loop for all depthwise users, including the row-reduction fusion below.
template <typename I, bool Half>
inline float depthwise_value(const Depthwise& d, I oy, I ox, I c) {
    float acc = d.bias ? d.bias[c] : 0.f;
    for (I ky = 0; ky < d.k; ++ky) {
        const I yy = oy * d.stride + ky * d.dilation - d.pad;
        if (yy < 0 || yy >= d.h) continue;
        for (I kx = 0; kx < d.k; ++kx) {
            const I xx = ox * d.stride + kx * d.dilation - d.pad;
            if (xx < 0 || xx >= d.wdim) continue;
            const I xi = (yy * (I) d.wdim + xx) * (I) d.c + c;
            const I wi = (c * d.k + ky) * d.k + kx;
            if constexpr (Half) acc += f32_from_f16(d.x16[xi]) * f32_from_f16(d.w16[wi]);
            else acc += d.w[wi] * d.x[xi];
        }
    }
    return acc;
}

template <typename I>
void depthwise_norm_launch(const Depthwise& d, void* stream);

template <typename I, bool Half>
void depthwise_launch(const Depthwise& d, void* stream) {
    const auto e = queue_for(stream).parallel_for(
        sycl::range<3>((size_t) d.oh, (size_t) d.ow, (size_t) d.c), [=](sycl::id<3> id) {
            const I oy = (I) id[0], ox = (I) id[1], c = (I) id[2];
            const I i = (oy * (I) d.ow + ox) * (I) d.c + c;
            float v = image_activate(d.act, depthwise_value<I, Half>(d, oy, ox, c));
            if (d.residual) v = d.x[i] + v;
            if constexpr (Half) d.out16[i] = f16_from_f32(v);
            else d.out[i] = v;
        });
    finish(stream, e, "image_depthwise");
}
}  // namespace

void image_depthwise(const Depthwise& d, void* stream) {
    if (d.oh <= 0 || d.ow <= 0 || d.c <= 0) return;
    if (d.h <= 0 || d.wdim <= 0 || d.k <= 0 || d.stride <= 0 || d.dilation <= 0 || d.pad < 0)
        throw std::invalid_argument("image_depthwise: invalid dimensions");
    // The 3D launch removes per-output division. The former 1D form took 0.7 ms on B70 at 72 x 72 x 256.
    // Keep wide indices for tensors and padded coordinates that do not fit in a signed 32-bit index.
    const int64_t limit = std::numeric_limits<int>::max();
    const bool narrow = d.h <= limit / d.wdim / d.c && d.oh <= limit / d.ow / d.c &&
                        d.c <= limit / d.k / d.k &&
                        d.oh <= (limit - (int64_t) d.k * d.dilation - d.pad) / d.stride &&
                        d.ow <= (limit - (int64_t) d.k * d.dilation - d.pad) / d.stride;
    if (d.norm_w) {
        depthwise_norm_launch<int64_t>(d, stream);
    } else if (d.x16) {
        if (narrow) depthwise_launch<int, true>(d, stream);
        else depthwise_launch<int64_t, true>(d, stream);
    } else {
        if (narrow) depthwise_launch<int, false>(d, stream);
        else depthwise_launch<int64_t, false>(d, stream);
    }
}


namespace {
// A channel reduction requires one work-group per pixel. Keep its 1D mapping: the 3D mapping changed Florence-2
// outputs on RTX 4070 (stage0 mean absolute difference 4.35e-5, max 0.00175) and took 45.6 vs 44.2 ms in vision.
template <typename I>
void depthwise_norm_launch(const Depthwise& d, void* stream) {
    const auto x = d.x; const auto w = d.w; const auto b = d.bias;
    const auto nw = d.norm_w; const auto nb = d.norm_b;
    const auto H = d.h, W = d.wdim, C = d.c, o_ws = d.window;
    const auto eps = d.eps; auto y = d.out; auto y16 = d.out16;
    const int64_t T = H * W;
    if (T <= 0) return;
    const bool regs = row_vals(C);
    const int64_t L = regs ? row_items(C) : 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * L)), sycl::range<1>((size_t) L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const I py = t / W, px = t - py * W;
            float* yr = y + t * C;
            const auto conv = [&](int64_t c) {
                Depthwise conv; conv.x = x; conv.w = w; conv.bias = b; conv.h = H; conv.wdim = W; conv.c = C;
                return depthwise_value<I, false>(conv, py, px, (I) c) + x[t * C + c];
            };
            const int64_t o = o_ws > 0 ? ((py / o_ws) * (W / o_ws) + px / o_ws) * o_ws * o_ws + (py % o_ws) * o_ws +
                                             px % o_ws
                                       : t;
            if (regs) {
                float v[kRowVals];
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    v[k] = c < C ? conv(c) : 0.f;
                    if (c < C) yr[c] = v[k];
                }
                const sycl::float2 st = row_stats(it, v, l, L, C, eps);
#pragma unroll
                for (int64_t k = 0; k < kRowVals; ++k) {
                    const int64_t c = l + k * L;
                    if (c < C) y16[o * C + c] = f16_from_f32((v[k] - st[0]) * st[1] * nw[c] + nb[c]);
                }
                return;
            }
            float s = 0.f;
            for (int64_t c = l; c < C; c += L) {   // each item its own channels: written, then read back
                const float v = conv(c);
                yr[c] = v;
                s += v;
            }
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float s2 = 0.f;
            for (int64_t c = l; c < C; c += L) {
                const float d = yr[c] - mean;
                s2 += d * d;
            }
            const float inv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
            for (int64_t c = l; c < C; c += L) y16[o * C + c] = f16_from_f32((yr[c] - mean) * inv * nw[c] + nb[c]);
        });
    finish(stream, e, "aux_dwconv3_norm16");
}
}  // namespace

}  // namespace strata::image
