// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_start_test.cpp - img_start's two shared parts: a base or reference picture as the planar floats its
// model's encoder reads, and the seed's noise as its DiT's tokens take it.  The picture's part is where the models
// differ, so every one of their ways is checked against the formula that model read with before the parts were merged
// here (Qwen-Image's VAE and vision encoder, Mage-Flow's reference, MOSS-Image's, Anima's).  No GPU, no model.

#include "strata/image/img_start.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using strata::image::image_planes;
using strata::image::noise_to_tokens;
using strata::image::PlaneAlpha;
using strata::image::PlaneRange;

namespace {

int failed = 0;

#define CHECK(x)                                                                \
    do {                                                                        \
        if (!(x)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);   \
            ++failed;                                                           \
        }                                                                       \
    } while (0)

constexpr int64_t W = 2, H = 2, n = W * H;

// a 2 x 2 RGBA8 picture: one opaque pixel, one clear, one of a quarter's alpha, one of half
const std::vector<uint8_t>& px() {
    static const std::vector<uint8_t> v = {10, 20, 30, 255, 100, 150, 200, 0, 0, 128, 255, 64, 7, 8, 9, 128};
    return v;
}

float alpha(int64_t q) { return (float) px()[(size_t) (q * 4 + 3)] / 255.f; }

// Qwen-Image's VAE input: each channel as the file has it, 0 .. 1, the alpha one of them
void keep_unit() {
    const std::vector<float> p = image_planes(px(), W, H, 4, PlaneAlpha::Keep, PlaneRange::Unit);
    CHECK(p.size() == (size_t) (4 * n));
    for (int64_t q = 0; q < n; ++q)
        for (int64_t c = 0; c < 4; ++c)
            CHECK(p[(size_t) (c * n + q)] == (float) px()[(size_t) (q * 4 + c)] / 255.f);
}

// Qwen-Image's vision encoder's input: over white in the 0 .. 1 floats, not rounded to a byte, then -1 .. 1
void white_unit_vision() {
    const std::vector<float> p = image_planes(px(), W, H, 3, PlaneAlpha::WhiteUnit, PlaneRange::Signed);
    for (int64_t q = 0; q < n; ++q) {
        const float a = alpha(q);
        for (int64_t c = 0; c < 3; ++c)
            CHECK(p[(size_t) (c * n + q)] == 2.f * ((float) px()[(size_t) (q * 4 + c)] / 255.f * a + 1.f - a) - 1.f);
    }
}

// the over-white composite at the whole byte, as the three other models read it: Mage-Flow keeps 0 .. 255 (it resizes
// and maps itself), MOSS-Image -1 .. 1, Anima 0 .. 1 with an opaque alpha plane
void white_byte() {
    const std::vector<float> p = image_planes(px(), W, H, 3, PlaneAlpha::White, PlaneRange::Byte);
    for (int64_t q = 0; q < n; ++q) {
        const float a = alpha(q);
        for (int64_t c = 0; c < 3; ++c)
            CHECK(p[(size_t) (c * n + q)] ==
                  std::nearbyint((float) px()[(size_t) (q * 4 + c)] * a + 255.f * (1.f - a)));
    }
}

void white_signed() {
    const std::vector<float> p = image_planes(px(), W, H, 3, PlaneAlpha::White, PlaneRange::Signed);
    for (int64_t q = 0; q < n; ++q) {
        const float a = alpha(q);
        for (int64_t c = 0; c < 3; ++c) {
            const float v = std::nearbyint((float) px()[(size_t) (q * 4 + c)] * a + 255.f * (1.f - a));
            CHECK(p[(size_t) (c * n + q)] == v / 127.5f - 1.f);
        }
    }
}

void white_rgba_unit() {
    const std::vector<float> p = image_planes(px(), W, H, 4, PlaneAlpha::White, PlaneRange::Unit);
    for (int64_t q = 0; q < n; ++q) {
        const float a = alpha(q);
        for (int64_t c = 0; c < 3; ++c)
            CHECK(p[(size_t) (c * n + q)] ==
                  std::nearbyint((float) px()[(size_t) (q * 4 + c)] * a + 255.f * (1.f - a)) / 255.f);
        CHECK(p[(size_t) (3 * n + q)] == 1.f);   // Anima's fourth channel: the opaque white
    }
}

// a latent of C channels of lh x lw, each value its own place in the [C][H][W] order, its tokens of one latent pixel
// (Mage-Flow, MOSS-Image) and of a 2 x 2 patch (Anima, SeedVR2)
void token_orders() {
    constexpr int64_t C = 3, lh = 2, lw = 4, N = C * lh * lw;
    std::vector<float> src((size_t) N), out((size_t) N);
    for (int64_t i = 0; i < N; ++i) src[(size_t) i] = (float) i;

    noise_to_tokens(src.data(), C, lh, lw, 1, out.data());
    for (int64_t c = 0; c < C; ++c)
        for (int64_t q = 0; q < lh * lw; ++q)
            CHECK(out[(size_t) (q * C + c)] == src[(size_t) (c * lh * lw + q)]);

    noise_to_tokens(src.data(), C, lh, lw, 2, out.data());
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < lh; ++y)
            for (int64_t x = 0; x < lw; ++x)
                CHECK(out[(size_t) (((y / 2) * (lw / 2) + x / 2) * 4 * C + c * 4 + (y % 2) * 2 + x % 2)] ==
                      src[(size_t) ((c * lh + y) * lw + x)]);

    // a video's [channel][frame][H][W]: two frames of the same shape, their channels two frames' worth apart
    std::vector<float> vid((size_t) (2 * N)), both((size_t) (2 * N));
    for (int64_t i = 0; i < 2 * N; ++i) vid[(size_t) i] = (float) i;
    for (int64_t f = 0; f < 2; ++f)   // the frame's first value, and a channel two frames of lh x lw apart
        noise_to_tokens(vid.data() + f * lh * lw, C, lh, lw, 2, both.data() + f * N, 2 * lh * lw);
    for (int64_t f = 0; f < 2; ++f)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t y = 0; y < lh; ++y)
                for (int64_t x = 0; x < lw; ++x)
                    CHECK(both[(size_t) (f * N + ((y / 2) * (lw / 2) + x / 2) * 4 * C + c * 4 + (y % 2) * 2 + x % 2)] ==
                          vid[(size_t) ((c * 2 + f) * lh * lw + y * lw + x)]);
}

}  // namespace

int main() {
    keep_unit();
    white_unit_vision();
    white_byte();
    white_signed();
    white_rgba_unit();
    token_orders();
    if (failed != 0) {
        std::fprintf(stderr, "img_start_test: %d checks failed\n", failed);
        return 1;
    }
    std::fprintf(stderr, "img_start_test: the models' ways of reading a picture and its noise, passed\n");
    return 0;
}
