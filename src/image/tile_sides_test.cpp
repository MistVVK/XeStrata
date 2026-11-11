// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/tile_sides_test.cpp - the VAE tiles' shared geometry and blend weight, every one against the formula its
// own file read with before the parts were merged here: Qwen-Image's VAE walked its tile corners with a clamped loop
// and weighted a position by its four edges' (d + 1) / (ramp + 1) multiplied, the Flux2 VAE built its corners a list
// at a time and weighted a position by (d + 1/2) / ramp, the lesser of an axis's two edges, and a video's tile side
// came out of the number of tiles its cap takes.  No GPU, no model.

#include "tile_sides.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::image;

namespace {

int failed = 0;

#define CHECK(x)                                                              \
    do {                                                                      \
        if (!(x)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                         \
        }                                                                     \
    } while (0)

// Qwen-Image's VAE (qi_vae.cpp's decode): each row of tiles clamped to end on the edge
std::vector<int64_t> clamped_corners(int64_t n, int64_t side, int64_t ov) {
    std::vector<int64_t> v;
    for (int64_t y0 = 0;; y0 += side - ov) {
        const int64_t yy = std::min(y0, n - side);
        v.push_back(yy);
        if (yy + side >= n) break;
    }
    return v;
}

// the Flux2 VAE and its video (mf_vae.cpp's starts, its tile_starts): the list built in one pass
std::vector<int64_t> listed_corners(int64_t n, int64_t side, int64_t ov) {
    std::vector<int64_t> v;
    for (int64_t a = 0;; a += side - ov) {
        if (a + side >= n) {
            v.push_back(n - side);
            break;
        }
        v.push_back(a);
    }
    return v;
}

// a video's tile side along n of at most `cap`, as mf_vae.cpp computed it before tile_side() was shared
int64_t video_side_before(int64_t n, int64_t cap, int64_t ov) {
    if (n <= cap) return n;
    const int64_t k = (int64_t) listed_corners(n, cap, ov).size();
    return (n + (k - 1) * ov + k - 1) / k;
}

// Qwen-Image's VAE's weight: every edge it shares, (d + 1) / (ramp + 1) capped at 1, the four multiplied
float weight_step_before(int64_t y, int64_t x, int64_t th, int64_t tw, int64_t ramp, bool top, bool bottom, bool left,
                         bool right) {
    const auto edge = [ramp](int64_t d) { return std::fmin(1.0f, (float) (d + 1) / (float) (ramp + 1)); };
    float w = 1.0f;
    if (top) w *= edge(y);
    if (bottom) w *= edge(th - 1 - y);
    if (left) w *= edge(x);
    if (right) w *= edge(tw - 1 - x);
    return w;
}

// the Flux2 VAE's weight: (d + 0.5) / ramp, the lesser of an axis's two edges, the two axes multiplied
float weight_centre_before(int64_t y, int64_t x, int64_t th, int64_t tw, int64_t ramp, bool top, bool bottom, bool left,
                           bool right) {
    const float r = (float) ramp;
    float wy = 1.f, wx = 1.f;
    if (top) wy = std::fmin(wy, ((float) y + 0.5f) / r);
    if (bottom) wy = std::fmin(wy, ((float) (th - y) - 0.5f) / r);
    if (left) wx = std::fmin(wx, ((float) x + 0.5f) / r);
    if (right) wx = std::fmin(wx, ((float) (tw - x) - 0.5f) / r);
    return wy * wx;
}

void corners() {
    for (const int64_t ov : {2, 4, 8})
        for (int64_t n = 1; n <= 300; ++n)
            for (int64_t side = ov + 1; side <= n; ++side) {
                const std::vector<int64_t> now = tile_positions(n, side, ov);
                CHECK(now == clamped_corners(n, side, ov));
                CHECK(now == listed_corners(n, side, ov));
                CHECK(now.front() == 0);        // the first tile starts on the edge
                CHECK(now.back() + side == n);  // and the last ends on it
                for (size_t i = 1; i < now.size(); ++i) {
                    CHECK(now[i] > now[i - 1]);                 // no corner twice
                    CHECK(now[i] - now[i - 1] <= side - ov);    // never further than a tile less the band
                }
            }
}

void sides() {
    for (const int64_t ov : {2, 4, 8})
        for (int64_t n = 1; n <= 300; ++n) {
            for (int64_t k = 1; k <= 8; ++k) {
                const int64_t side = tile_side(n, k, ov);
                CHECK(side >= 1);
                if (k == 1) CHECK(side == n);
            }
            for (int64_t cap = ov + 1; cap <= n; ++cap)   // a video: the side its cap's tiles take
                CHECK(tile_side(n, (int64_t) tile_positions(n, cap, ov).size(), ov) == video_side_before(n, cap, ov));
        }
}

void weights() {
    for (const int64_t ramp : {1, 2, 8, 16, 64})
        for (const int64_t th : {4, 8, 16, 32, 128})
            for (const int64_t tw : {4, 16, 32})
                for (int64_t y = 0; y < th; y += (th + 7) / 8)
                    for (int64_t x = 0; x < tw; x += (tw + 7) / 8)
                        for (int f = 0; f < 16; ++f) {   // every mix of shared edges
                            const bool top = (f & 1) != 0, bottom = (f & 2) != 0, left = (f & 4) != 0,
                                     right = (f & 8) != 0;
                            CHECK(tile_weight(TileRamp::Step, y, x, th, tw, ramp, top, bottom, left, right) ==
                                  weight_step_before(y, x, th, tw, ramp, top, bottom, left, right));
                            CHECK(tile_weight(TileRamp::Centre, y, x, th, tw, ramp, top, bottom, left, right) ==
                                  weight_centre_before(y, x, th, tw, ramp, top, bottom, left, right));
                        }
    // one edge's factor: 1 clear of the band, and never above it
    for (const int64_t ramp : {2, 8, 32})
        for (int64_t d = 0; d <= ramp + 3; ++d) {
            CHECK(tile_ramp(TileRamp::Step, d, ramp) <= 1.0f);
            CHECK(tile_ramp(TileRamp::Centre, d, ramp) <= 1.0f);
            CHECK(tile_ramp(TileRamp::Step, ramp, ramp) == 1.0f);
            CHECK(tile_ramp(TileRamp::Centre, ramp, ramp) == 1.0f);
            CHECK(tile_ramp(TileRamp::Step, d, ramp) > tile_ramp(TileRamp::Step, d - 1, ramp) || d > ramp);
        }
    // the two shapes are not the same weight, which is why the shape stays an argument
    CHECK(tile_weight(TileRamp::Step, 0, 0, 32, 32, 8, true, false, false, false) !=
          tile_weight(TileRamp::Centre, 0, 0, 32, 32, 8, true, false, false, false));
}

}  // namespace

int main() {
    corners();
    sides();
    weights();
    if (failed != 0) {
        std::fprintf(stderr, "tile_sides_test: %d checks failed\n", failed);
        return 1;
    }
    std::fprintf(stderr, "tile_sides_test: the tiles' corners, sides and blend weights, passed\n");
    return 0;
}
