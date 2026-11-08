// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/image/img_color.hpp - the image engine's color corrections on the GPU, shared by the cut's color
// match (img_crop.hpp) and the upscalers' correction (strata/upscale/upscale.hpp).
//
// The images are RGBA8 on the GPU, on the image engine's queue (the runtime's default stream).  Statistics are each
// channel's mean and spread over the opaque pixels (alpha 255), summed a work-group at a time on the GPU and the
// groups' sums on the host in doubles.  A transfer moves an image's channels from its own mean and spread to another's
// (Reinhard et al., Color Transfer between Images, 2001): in CIE Lab (D65, as kornia's rgb_to_lab) or in RGB (an AdaIN
// of the pixels).  The wavelet correction keeps an image's high frequencies and takes another's low ones (five dilated
// 3 x 3 blurs, the edges repeated).
#pragma once

#include <cstdint>

namespace strata::image::color {

enum class Space { rgb, lab };

/// Each channel's sum and sum of squares and the pixels counted.
struct Stats {
    double sum[3] = {0.0, 0.0, 0.0}, sq[3] = {0.0, 0.0, 0.0}, n = 0.0;
};

/// The statistics of rgba [n][4]'s opaque pixels in `space`; with `mask` [n], only those where it is below `below`;
/// with `partner` [n][4], only those opaque there too.
Stats stats(const uint8_t* rgba, int64_t n, Space space, const float* mask = nullptr, float below = 0.0f,
            const uint8_t* partner = nullptr);

/// rgba's channels (its alpha left) moved from `from`'s mean and spread to `to`'s in `space` -> true; false, the image
/// as it was, where fewer than `least` pixels were counted on either side or a channel is flat (no spread to scale).
bool transfer(uint8_t* rgba, int64_t n, Space space, const Stats& from, const Stats& to, double least);

/// rgba (w x h) with its high frequencies kept and `ref`'s (w x h too) low ones: `levels` blurs of the kernel
/// [1 2 1] x [1 2 1] / 16, dilated 1, 2, 4 ... pixels (at most an eighth of the shorter side), the edges repeated; the
/// image less its low part plus ref's.
void wavelet(uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, int levels = 5);

/// The VRAM wavelet() takes for w x h pixels.
uint64_t wavelet_bytes(int64_t w, int64_t h);

/// SeedVR2's lab correction (vLLM-Omni's color_correction_method "lab"): the wavelet correction toward
/// ref, then in CIE Lab (the reference's matrices) each channel given ref's distribution, rank for rank (a histogram
/// of 65536 bins of the channel's range: a bin's pixels take ref's value at their middle rank), the lightness kept
/// `lum_weight` the corrected image's own.
void lab_match(uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, float lum_weight = 0.8f);
uint64_t lab_match_bytes(int64_t w, int64_t h);

}  // namespace strata::image::color
