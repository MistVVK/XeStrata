// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_crop_test.cpp - img_crop's host resize of a picture's planar floats (pillow_resize_planes).  The
// check is against PIL's own Image.resize (Pillow 12.3.0) of the same pictures, byte for byte, held here as each
// output's hash; the flat picture and the whole bytes are what every filter's user is owed.  No GPU, no model, no
// Python.

#include "strata/image/img_crop.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

// ---- the pictures: the whole bytes an encoder reads.  Noise for the rounding, a hard edge for the cubic's negative
// lobes and their clipping.
// (i * 2654435761 mod 2^32) >> 24 over the planar index i: noise any language makes alike
std::vector<float> hashed(int64_t C, int64_t W, int64_t H) {
    std::vector<float> p((size_t) (C * W * H));
    for (size_t i = 0; i < p.size(); ++i) p[i] = (float) ((uint32_t) ((uint64_t) i * 2654435761u) >> 24);
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

// FNV-1a 64 of the outputs as bytes, [C][H][W]
uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const float x : v) h = (h ^ (uint64_t) (uint8_t) x) * 0x100000001b3ull;
    return h;
}

// PIL's own resize of the pictures (Pillow 12.3.0 hashed by a script: Image.fromarray of the planes as L or RGB,
// Image.resize with LANCZOS, BILINEAR or BICUBIC): the same bytes.  The models' references were made by PIL, which
// this matches exactly (the doubles' totals before were a level apart in 0.01 to 1.2% of the values).  The last three
// are a picture over a hundred times taller than wide losing height, which Image.resize resizes columns first.
void same_as_pil() {
    struct Pil {
        const char* picture;
        int64_t c, w, h, w2, h2;
        Filter f;
        uint64_t hash;
    };
    const Pil cases[] = {
        {"hash", 3, 101, 67, 51, 33, Filter::bicubic, 0x81cc405d1054b608ull},
        {"edges", 3, 101, 67, 51, 33, Filter::bicubic, 0xaf4af526c55d8910ull},
        {"hash", 3, 32, 32, 100, 76, Filter::bicubic, 0x80f076b795464fc7ull},
        {"edges", 3, 32, 32, 100, 76, Filter::bicubic, 0xe7bdcbda22343018ull},
        {"hash", 3, 512, 512, 384, 288, Filter::bicubic, 0xe9e7034a90614fc9ull},
        {"edges", 3, 512, 512, 384, 288, Filter::bicubic, 0x3c7e9e5c5b44aa57ull},
        {"hash", 3, 384, 288, 352, 288, Filter::bicubic, 0x52cd7aa82ddcdeabull},
        {"edges", 3, 384, 288, 352, 288, Filter::bicubic, 0x2e3f71e8ef379a96ull},
        {"hash", 3, 97, 41, 41, 97, Filter::bicubic, 0x62f18eb1ee570627ull},
        {"edges", 3, 97, 41, 41, 97, Filter::bicubic, 0x928efa918c29efe0ull},
        {"hash", 3, 16, 16, 3, 5, Filter::bicubic, 0x1de5451f92f9d5afull},
        {"edges", 3, 16, 16, 3, 5, Filter::bicubic, 0x1a41815db5221ef8ull},
        {"hash", 3, 7, 5, 64, 64, Filter::bicubic, 0x62299abe83e3450aull},
        {"edges", 3, 7, 5, 64, 64, Filter::bicubic, 0x90c267ab9bc4c862ull},
        {"hash", 3, 256, 256, 512, 512, Filter::bicubic, 0x0560fc11442803e9ull},
        {"edges", 3, 256, 256, 512, 512, Filter::bicubic, 0xf587c91dabadfe9dull},
        {"hash", 3, 101, 67, 51, 33, Filter::lanczos, 0x8b8156decd48cc1aull},
        {"hash", 3, 32, 32, 100, 76, Filter::lanczos, 0x7704ca844abb35faull},
        {"hash", 3, 97, 41, 41, 97, Filter::lanczos, 0x65d0aec4a9c8a2a8ull},
        {"hash", 3, 300, 40, 900, 30, Filter::lanczos, 0x5f02c10e62306816ull},
        {"hash", 3, 101, 67, 51, 33, Filter::triangle, 0xe08d77e1b5f89d70ull},
        {"hash", 3, 32, 32, 100, 76, Filter::triangle, 0xf27da35aee17a2e0ull},
        {"hash", 3, 97, 41, 41, 97, Filter::triangle, 0x9ff5c98d1bbb5df0ull},
        {"hash", 3, 300, 40, 900, 30, Filter::triangle, 0xcdb8e18c3f58da10ull},
        {"hash", 1, 2, 318, 338, 84, Filter::lanczos, 0x3312d24dd65dee79ull},
        {"edges", 1, 3, 400, 5, 100, Filter::bicubic, 0xe2b3fbf576dea204ull},
        {"hash", 1, 3, 400, 3, 100, Filter::triangle, 0xa69cd742dc34f9b1ull},
    };
    for (const Pil& p : cases) {
        const std::vector<float> in = p.picture[0] == 'h' ? hashed(p.c, p.w, p.h) : edges(p.c, p.w, p.h);
        const std::vector<float> r = pillow_resize_planes(in, p.c, p.w, p.h, p.w2, p.h2, p.f);
        CHECK(r.size() == (size_t) (p.c * p.w2 * p.h2));
        if (fnv(r) != p.hash) {
            std::fprintf(stderr, "  %s %lld x %lld to %lld x %lld (filter %d): not PIL's bytes\n", p.picture,
                         (long long) p.w, (long long) p.h, (long long) p.w2, (long long) p.h2, (int) p.f);
            CHECK(false);
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
    same_as_pil();
    flat_picture();
    bytes_and_identity();
    if (failed != 0) {
        std::fprintf(stderr, "img_crop_test: %d checks failed\n", failed);
        return 1;
    }
    std::fprintf(stderr, "img_crop_test: the host resize of a picture's floats, passed\n");
    return 0;
}
