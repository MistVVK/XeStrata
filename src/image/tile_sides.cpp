// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/tile_sides.cpp - the device's parts of a VAE's tiles: a tile cut out of the picture it comes from or goes
// into, its weighted add into the whole, and the divide that ends the blend (tile_sides.hpp).

#include "tile_sides.hpp"

#include "strata/core/runtime.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

namespace strata::image {

template<class T>
void tile_crop(const T* src, int64_t C, TilePlanes planes, int64_t h, int64_t w, int64_t y0, int64_t x0, int64_t th,
               int64_t tw, T* dst, void* stream) {
    q_of(stream).parallel_for(sycl::range<1>((size_t) (C * th * tw)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        if (planes == TilePlanes::Planes) {   // a plane per channel, the whole's rows h x w
            const int64_t c = i / (th * tw), r = i - c * (th * tw), y = r / tw, x = r - y * tw;
            dst[i] = src[(c * h + y0 + y) * w + x0 + x];
        } else {                              // the C channels together at every position
            const int64_t p = i / C, c = i - p * C, y = p / tw, x = p - y * tw;
            dst[i] = src[((y0 + y) * w + x0 + x) * C + c];
        }
    });
}

template void tile_crop(const float* src, int64_t C, TilePlanes planes, int64_t h, int64_t w, int64_t y0, int64_t x0,
                        int64_t th, int64_t tw, float* dst, void* stream);
template void tile_crop(const uint16_t* src, int64_t C, TilePlanes planes, int64_t h, int64_t w, int64_t y0,
                        int64_t x0, int64_t th, int64_t tw, uint16_t* dst, void* stream);

void tile_blend(const float* tile, int64_t C, TilePlanes planes, int64_t TH, int64_t TW, int64_t Y0, int64_t X0,
                int64_t H, int64_t W, int64_t ramp, TileRamp shape, bool top, bool bottom, bool left, bool right,
                float* acc, float* wsum, void* stream) {
    q_of(stream).parallel_for(sycl::range<1>((size_t) (TH * TW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t y = i / TW, x = i - y * TW;
        const float wgt = tile_weight(shape, y, x, TH, TW, ramp, top, bottom, left, right);
        const int64_t HW = H * W, p = (Y0 + y) * W + X0 + x;
        for (int64_t c = 0; c < C; ++c)
            acc[c * HW + p] += wgt * (planes == TilePlanes::Planes ? tile[c * TH * TW + i] : tile[i * C + c]);
        wsum[p] += wgt;
    });
}

void tile_finish(const float* acc, int64_t C, TilePlanes planes, int64_t HW, const float* wsum, float* out,
                 void* stream) {
    q_of(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = planes == TilePlanes::Planes ? i / HW : i - (i / C) * C;
        const int64_t p = planes == TilePlanes::Planes ? i - c * HW : i / C;
        out[i] = acc[c * HW + p] / wsum[p];
    });
}

}  // namespace strata::image
