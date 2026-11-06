// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/tile_sides.hpp - how a VAE cuts an h x w latent into overlapping tiles where the whole does not fit: as
// many rows and columns of tiles as cover it with the least work, each side as even as the overlap leaves.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace strata::image {

/// The tiles' rows and columns (th, tw) of an h x w latent that `fits` (a tile's rows and columns) takes: the counts of
/// rows and columns of tiles whose positions summed are the fewest (then the fewer tiles), each tile's side
/// ceil((n + (k - 1) ov) / k) for k tiles over n with `ov` shared between two, none below min(n, least); (0, 0) when
/// not even the least tile fits.  (Squares as large as fit took a 52 x 78 latent in four 47 x 47 tiles, twice its
/// positions, and a 117 x 78 one in two 78 x 78, a third more.)
template<class Fits>
std::pair<int64_t, int64_t> tile_sides(int64_t h, int64_t w, int64_t ov, int64_t least, const Fits& fits) {
    const auto side = [ov](int64_t n, int64_t k) { return k == 1 ? n : (n + (k - 1) * ov + k - 1) / k; };
    const int64_t least_h = std::min(h, least), least_w = std::min(w, least);
    std::pair<int64_t, int64_t> best{0, 0};
    int64_t best_work = INT64_MAX, best_tiles = INT64_MAX;
    for (int64_t ny = 1;; ++ny) {
        const int64_t th = side(h, ny);
        if (th < least_h || (ny > 1 && th <= ov)) break;
        for (int64_t nx = 1;; ++nx) {
            const int64_t tw = side(w, nx);
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

}  // namespace strata::image
