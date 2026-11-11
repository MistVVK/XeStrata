// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/tile_sides.hpp - what the image models' VAEs share when a latent does not fit the GPU: how its tiles are
// cut (as many rows and columns as cover the latent with the least work, each side as even as the overlap leaves),
// the tiles' positions, a tile's weight in the blend, and the device parts that cut a tile, add it weighted and
// divide the whole back (tile_sides.cpp).
// QiVae (qi_vae.cpp) and MfVae (mf_vae.cpp), its pictures and its video alike, cut the same way; they only ramp the
// weight over the shared band differently, and either shape read over the other's tiles moves its pixels, so the
// shape is an argument and both stand (TileRamp).
// SeedVR2's upscale (upscale.cpp) takes none of these: its windows are a core plus the context around it, and every
// output pixel comes from one window - nothing overlaps, so there is no band to ramp, no weights to keep and no
// divide to end, and its side comes from halving until the run fits rather than from tile_sides().
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::image {

/// A tile's channels' place in its picture.
enum class TilePlanes {
    Planes,        // a plane apiece: [C][H][W] (Qwen-Image's latent, a Flux2 tile's RGB)
    Interleaved,   // the channels together at every position: [H][W][C] (a Flux2 tile's tokens, a video latent's)
};

/// A blend's ramp over the `ramp` positions a tile shares with the neighbour at an edge, `d` of them in from that
/// edge, and the weight one of the tile's positions carries.  The two VAEs differ in both the step and the way the
/// edges are joined, and either read over the other's tiles moves its pixels, so both stand as they were.
enum class TileRamp {
    Step,    // Qwen-Image's VAE: every shared edge's (d + 1) / (ramp + 1), the four multiplied
    Centre,  // the Flux2 VAE, its pictures and its video alike: (d + 1/2) / ramp, the lesser of an axis's two edges,
             // the two axes multiplied
};

/// One edge's factor: `d` positions in from an edge a neighbour shares over `ramp` of them, in `shape`'s step; 1 clear
/// of the band.
inline float tile_ramp(TileRamp shape, int64_t d, int64_t ramp) {
    const float w = shape == TileRamp::Step ? (float) (d + 1) / (float) (ramp + 1)
                                            : ((float) d + 0.5f) / (float) ramp;
    return w < 1.0f ? w : 1.0f;
}

/// The weight of a blended tile's position (y, x) of a TH x TW tile whose edges another tile shares (top, bottom,
/// left, right), each of them ramping over `ramp` positions `shape`'s way.
inline float tile_weight(TileRamp shape, int64_t y, int64_t x, int64_t TH, int64_t TW, int64_t ramp, bool top,
                         bool bottom, bool left, bool right) {
    if (shape == TileRamp::Centre) {
        float wy = 1.0f, wx = 1.0f;
        if (top) wy = tile_ramp(shape, y, ramp);
        if (bottom) {
            const float b = tile_ramp(shape, TH - 1 - y, ramp);
            if (b < wy) wy = b;
        }
        if (left) wx = tile_ramp(shape, x, ramp);
        if (right) {
            const float r = tile_ramp(shape, TW - 1 - x, ramp);
            if (r < wx) wx = r;
        }
        return wy * wx;
    }
    float w = 1.0f;
    if (top) w *= tile_ramp(shape, y, ramp);
    if (bottom) w *= tile_ramp(shape, TH - 1 - y, ramp);
    if (left) w *= tile_ramp(shape, x, ramp);
    if (right) w *= tile_ramp(shape, TW - 1 - x, ramp);
    return w;
}

/// The side of `k` tiles that cover n with `ov` of every two shared: as even as that leaves them.  The search below
/// takes it; so does a video, whose k is the number of tiles a given side takes (mf_vae.cpp).
inline int64_t tile_side(int64_t n, int64_t k, int64_t ov) {
    return k == 1 ? n : (n + (k - 1) * ov + k - 1) / k;
}

/// The first positions of the tiles of `side` that cover a side of n, `side - ov` apart, the last flush with the edge
/// (`side` at most n, above `ov`).
inline std::vector<int64_t> tile_positions(int64_t n, int64_t side, int64_t ov) {
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

/// The tiles' rows and columns (th, tw) of an h x w latent that `fits` (a tile's rows and columns) takes: the counts of
/// rows and columns of tiles whose positions summed are the fewest (then the fewer tiles), each tile's side
/// tile_side() for that count, none below min(n, least); (0, 0) when not even the least tile fits.
///  (Squares as large as fit took a 52 x 78 latent in four 47 x 47 tiles, twice its positions, and a 117 x 78 one in
///  two 78 x 78, a third more.)
template<class Fits>
std::pair<int64_t, int64_t> tile_sides(int64_t h, int64_t w, int64_t ov, int64_t least, const Fits& fits) {
    const int64_t least_h = std::min(h, least), least_w = std::min(w, least);
    std::pair<int64_t, int64_t> best{0, 0};
    int64_t best_work = INT64_MAX, best_tiles = INT64_MAX;
    for (int64_t ny = 1;; ++ny) {
        const int64_t th = tile_side(h, ny, ov);
        if (th < least_h || (ny > 1 && th <= ov)) break;
        for (int64_t nx = 1;; ++nx) {
            const int64_t tw = tile_side(w, nx, ov);
            if (tw < least_w || (nx > 1 && tw <= ov)) break;
            if (!fits(th, tw)) continue;
            const int64_t work = ny * nx * th * tw, tiles = ny * nx;
            if (work < best_work || (work == best_work && tiles < best_tiles)) {
                best = {th, tw};
                best_work = work;
                best_tiles = tiles;
            }
            break;   // more columns only add work
        }
        if (th == least_h) break;
    }
    return best;
}

/// `dst` = the th x tw tile at row y0, column x0 of a picture of h x w positions and C channels in `planes`' order.
template<class T>
void tile_crop(const T* src, int64_t C, TilePlanes planes, int64_t h, int64_t w, int64_t y0, int64_t x0, int64_t th,
               int64_t tw, T* dst, void* stream);

/// acc, C planes of H x W, += the tile (TH x TW positions, `planes`' order) at row Y0, column X0 times its weight, and
/// wsum, H x W, += that weight: the weight 1 clear of the band, `shape`'s ramp over `ramp` positions at every edge
/// another tile shares (top, bottom, left, right).  The Flux2 VAE keeps its weights as acc's plane C; Qwen-Image's VAE
/// has a buffer of its own.
void tile_blend(const float* tile, int64_t C, TilePlanes planes, int64_t TH, int64_t TW, int64_t Y0, int64_t X0,
                int64_t H, int64_t W, int64_t ramp, TileRamp shape, bool top, bool bottom, bool left, bool right,
                float* acc, float* wsum, void* stream);

/// out, C channels in `planes`' order, = acc's planes over wsum.  out may be acc itself: with Planes every position is
/// read and written once, so the whole divides in place (Qwen-Image's VAE decodes straight into its output buffer).
void tile_finish(const float* acc, int64_t C, TilePlanes planes, int64_t HW, const float* wsum, float* out,
                 void* stream);

}  // namespace strata::image
