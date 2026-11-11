// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_crop_test.cpp - img_crop's host resize of a picture's planar floats (pillow_resize_planes).  The
// check is against the formula Mage-Flow read with before its private bicubic went into these parts, so the merge
// cannot move a byte of its references' and base image's pixels; the flat picture and the whole bytes are what the
// shared taps owe every filter's user.  No GPU, no model.

#include "strata/image/img_crop.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using strata::image::Filter;
using strata::image::pillow_resize_planes;

namespace {

int failed = 0;

#define CHECK(x)                                                              \
    do {                                                                      \
        if (!(x)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                         \
        }                                                                     \
    } while (0)

// ---- the formula before the merge (src/image/mf_pipeline.cpp at bdc55265a): PIL's bicubic (a = -0.5), its taps
// totalled in doubles over the row and divided by their sum when the row is done, each pass rounded and clipped to a
// whole byte.

double bicubic_w(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

void resample_axis(const std::vector<float>& src, int64_t lines, int64_t n_in, int64_t stride_line, int64_t stride_in,
                   int64_t n_out, std::vector<float>& dst, int64_t dst_line, int64_t dst_out) {
    const double scale = (double) n_in / (double) n_out, fs = std::max(scale, 1.0), support = 2.0 * fs;
    for (int64_t o = 0; o < n_out; ++o) {
        const double center = ((double) o + 0.5) * scale;
        const auto lo = std::max<int64_t>((int64_t) std::floor(center - support + 0.5), 0);
        const auto hi = std::min<int64_t>((int64_t) std::floor(center + support + 0.5), n_in);
        std::vector<double> w((size_t) (hi - lo));
        double sum = 0.0;
        for (int64_t i = lo; i < hi; ++i) sum += w[(size_t) (i - lo)] = bicubic_w(((double) i - center + 0.5) / fs);
        for (int64_t l = 0; l < lines; ++l) {
            double v = 0.0;
            for (int64_t i = lo; i < hi; ++i)
                v += w[(size_t) (i - lo)] * src[(size_t) (l * stride_line + i * stride_in)];
            dst[(size_t) (l * dst_line + o * dst_out)] = (float) std::clamp(std::nearbyint(v / sum), 0.0, 255.0);
        }
    }
}

std::vector<float> resize_bicubic(const std::vector<float>& rgb, int64_t W, int64_t H, int64_t W2, int64_t H2) {
    if (W == W2 && H == H2) return rgb;
    std::vector<float> t((size_t) (3 * H * W2)), out((size_t) (3 * H2 * W2));
    resample_axis(rgb, 3 * H, W, W, 1, W2, t, W2, 1);
    for (int64_t c = 0; c < 3; ++c) {
        std::vector<float> plane(t.begin() + c * H * W2, t.begin() + (c + 1) * H * W2), o((size_t) (H2 * W2));
        resample_axis(plane, W2, H, 1, W2, H2, o, 1, W2);
        std::copy(o.begin(), o.end(), out.begin() + c * H2 * W2);
    }
    return out;
}

// ---- the pictures: the whole bytes an encoder reads.  Noise for the rounding, a hard edge for the cubic's negative
// lobes and their clipping.
std::vector<float> noise(int64_t C, int64_t W, int64_t H, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::vector<float> p((size_t) (C * W * H));
    for (float& v : p) v = (float) (g() % 256);
    return p;
}

std::vector<float> edges(int64_t C, int64_t W, int64_t H) {
    std::vector<float> p((size_t) (C * W * H));
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < H; ++y)
            for (int64_t x = 0; x < W; ++x) {
                const bool on = ((x / 7 + y / 11) % 2) == (c % 2) || (x > W / 2 && y < H / 3);
                p[(size_t) (c * W * H + y * W + x)] = on ? 255.f : 0.f;
            }
    return p;
}

std::vector<float> flat(int64_t C, int64_t W, int64_t H, float v) {
    return std::vector<float>((size_t) (C * W * H), v);
}

struct Case {
    int64_t w, h, w2, h2;
};

// Mage-Flow's resize, before and after the merge: the same bytes
void same_as_before() {
    const Case cases[] = {{64, 64, 64, 64}, {101, 67, 51, 33},   {32, 32, 100, 76}, {512, 512, 384, 288},
                          {384, 288, 352, 288}, {97, 41, 41, 97}, {16, 16, 3, 5},   {7, 5, 64, 64}};
    for (const Case& c : cases) {
        const std::vector<float> noisy = noise(3, c.w, c.h, 7);
        const std::vector<float> edged = edges(3, c.w, c.h);
        for (const std::vector<float>* p : {&noisy, &edged}) {
            const std::vector<float> was = resize_bicubic(*p, c.w, c.h, c.w2, c.h2);
            const std::vector<float> now = pillow_resize_planes(*p, 3, c.w, c.h, c.w2, c.h2, Filter::bicubic);
            CHECK(was.size() == now.size());
            int64_t differ = 0;
            float worst = 0.f;
            for (size_t i = 0; i < std::min(was.size(), now.size()); ++i) {
                if (was[i] != now[i]) {
                    ++differ;
                    worst = std::max(worst, std::fabs(was[i] - now[i]));
                }
            }
            if (differ != 0)
                std::fprintf(stderr, "  %lld x %lld to %lld x %lld: %lld of %lld values differ, by %g\n",
                             (long long) c.w, (long long) c.h, (long long) c.w2, (long long) c.h2, (long long) differ,
                             (long long) was.size(), worst);
            CHECK(differ == 0);
        }
    }
}

// a flat picture is that flat level whatever the filter and the size: each output's taps total to their own sum
void flat_picture() {
    const Case cases[] = {{40, 24, 17, 61}, {17, 61, 40, 24}, {9, 9, 9, 9}, {1, 3, 64, 64}};
    for (const Case& c : cases)
        for (const Filter f : {Filter::bicubic, Filter::lanczos, Filter::triangle})
            for (const float v : {0.f, 1.f, 137.f, 254.f, 255.f}) {
                const std::vector<float> p = flat(3, c.w, c.h, v);
                const std::vector<float> r = pillow_resize_planes(p, 3, c.w, c.h, c.w2, c.h2, f);
                CHECK(r.size() == (size_t) (3 * c.w2 * c.h2));
                for (const float q : r)
                    if (q != v) {
                        CHECK(false);
                        break;
                    }
            }
}

// the outputs are whole bytes within 0 .. 255, and a resize to the size the picture is leaves it alone
void bytes_and_identity() {
    const std::vector<float> p = edges(3, 51, 37);
    const Case cases[] = {{51, 37, 23, 91}, {51, 37, 51, 37}};
    for (const Case& c : cases) {
        const std::vector<float> r = pillow_resize_planes(p, 3, c.w, c.h, c.w2, c.h2, Filter::bicubic);
        for (const float q : r) CHECK(q == std::floor(q) && q >= 0.f && q <= 255.f);
        if (c.w == c.w2 && c.h == c.h2) CHECK(r == p);
    }
}

}  // namespace

int main() {
    same_as_before();
    flat_picture();
    bytes_and_identity();
    if (failed != 0) {
        std::fprintf(stderr, "img_crop_test: %d checks failed\n", failed);
        return 1;
    }
    std::fprintf(stderr, "img_crop_test: the host resize of a picture's floats, passed\n");
    return 0;
}
