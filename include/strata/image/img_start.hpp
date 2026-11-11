// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/img_start.hpp - what img2img and a mask share across the image models' pipelines: the
// request's base image, the step its strength starts the sampler at, the mask's latent positions and kept pixels, the
// base and reference pictures as the encoders' planar floats, the seed's noise as its DiT's tokens take it, and the
// device buffers the sampler's start and its steps' blend read.
//
// img2img (a strength below 1) is the flow-matching img2img of diffusers' pipelines: of a schedule's n steps, the
// first floor(n (1 - strength)) are skipped, and the sampler starts at the next one's sigma s from the base image's
// latent on the flow's path, (1 - s) z0 + s noise, the noise being the seed's that a picture from nothing starts
// with: at strength 1 nothing is skipped and the picture is that one.  A mask keeps the base image where it is
// opaque: after each step the latent there goes back to the base's at the step's time, and the picture's kept pixels
// are the base image's own, blended by the mask's alpha (OpenAI's mask: alpha 0 where the image changes).  An edit
// model's base image is its first reference; a model without edits takes the request's init image.
#pragma once

#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::image {

/// The image the request's strength and mask work on: its init image, else (with a strength below 1 or a mask) its
/// first reference; null when it has none of them.
const QiRef* base_image(const QiRequest& r);

/// The steps of an n-step schedule that a strength skips (diffusers' get_timesteps: n less n strength, floored).
/// Throws std::invalid_argument when the strength is outside (0, 1] or would leave no step to run.
int64_t strength_skip(int64_t steps, double strength);

/// The schedule `sigmas` (n steps and the terminal 0) as the strength runs it: the sigmas from the first step it runs.
std::vector<float> strength_sigmas(const std::vector<float>& sigmas, double strength);

/// The reference images as a key for the states made from them (a pipeline's cache): their sizes and a hash of their
/// pixels; empty for none.
std::string refs_key(const std::vector<QiRef>& refs);

/// RGBA8 [H][W][4] of `f` at W x H, nearest (the base image and the mask as the pipelines take them).
std::vector<uint8_t> resize_nearest(const QiRef& f, int64_t W, int64_t H);

/// The transparency's part in the planar floats an image model's encoder reads, its base or reference picture turned
/// into them.  The models differ here, and only here: where the alpha goes, and where an 8-bit value lands in the
/// floats.  The picture's own order (channel first, a plane per channel) is the same in every model.
enum class PlaneAlpha {
    Keep,       // none of it: each channel is taken as the file has it, the alpha one of them (Qwen-Image's VAE input)
    White,      // over white, at the whole byte a saved picture would hold (Mage-Flow, MOSS-Image, Anima)
    WhiteUnit,  // over white in the 0 .. 1 floats, not rounded to a byte (Qwen-Image's vision encoder's input)
};

/// An 8-bit value's place in those floats.
enum class PlaneRange {
    Byte,    // 0 .. 255 (what Mage-Flow resizes before its own map to -1 .. 1)
    Unit,    // 0 .. 1
    Signed,  // -1 .. 1
};

/// The RGBA8 picture `px` at W x H as planar floats [channels][HW] (channels 3 for RGB, 4 for RGBA), the input of a
/// VAE's or a vision encoder's.  Where the alpha goes over white, the fourth plane of a four-channel take is the
/// opaque white.
std::vector<float> image_planes(const std::vector<uint8_t>& px, int64_t W, int64_t H, int64_t channels,
                                PlaneAlpha alpha, PlaneRange range);

/// The same of `f` at W x H, nearest (resize_nearest).
std::vector<float> image_planes(const QiRef& f, int64_t W, int64_t H, int64_t channels, PlaneAlpha alpha,
                                PlaneRange range);

/// The mask over a W x H picture by latent position (`patch` pixels a side): 1 where a pixel of the position is less
/// than opaque (the part that changes), 0 elsewhere, [H / patch][W / patch].  `px`: the mask at W x H (nearest).
std::vector<float> latent_mask(const std::vector<uint8_t>& px, int64_t W, int64_t H, int64_t patch);

/// The sampler's mask of a request: latent_mask's, or with r.mask_share (the cut's) that share's mean over
/// each position, the part a position changes (the step's blend, qi_inpaint_blend, takes a share as it takes 0 and 1:
/// a position at the mask's blurred edge follows the base image as much as it is kept).
std::vector<float> request_latent_mask(const QiRequest& r, const std::vector<uint8_t>& px, int64_t W, int64_t H,
                                       int64_t patch);

/// The VAE's RGBA, planar floats [4][HW] in 0..1, as RGBA8 [HW][4] (the final image's and a partial image's).
std::vector<uint8_t> rgba8(const std::vector<float>& planar, int64_t HW);

/// The picture's kept part: img = (a base + (255 - a) img) / 255 by the mask's alpha a, all RGBA8 [H][W][4].
void keep_pixels(std::vector<uint8_t>& img, const std::vector<uint8_t>& base, const std::vector<uint8_t>& mask);

/// The seed's noise as the latent's [C][H][W] (the order a randn of the latent draws it in) in the order its DiT's
/// tokens take them: H / patch x W / patch tokens of C * patch * patch values, the latent's channel c and the patch's
/// part (dy, dx) at c * patch * patch + dy * patch + dx.  patch 1 is a token of one latent pixel (Mage-Flow,
/// MOSS-Image); 2 the 2 x 2 patch a token of Mage-VAE's and Wan 2.1's latents (Anima, SeedVR2).  `src` and `out` take
/// C * H * W values; `stride`, the values one channel is apart, is H * W of a single latent, and a video's
/// [channel][frame][H][W] passes its frames' worth, called once a frame with `src` and `out` at that frame's start.
void noise_to_tokens(const float* src, int64_t C, int64_t H, int64_t W, int64_t patch, float* out,
                     int64_t stride = 0);

/// The device buffers of a request's start and blend: the base image's latent, and for a mask its positions and the
/// seed's noise.  Freed with the object.
class LatentStart {
public:
    LatentStart() = default;
    ~LatentStart();
    LatentStart(const LatentStart&) = delete;
    LatentStart& operator=(const LatentStart&) = delete;

    /// The base latent's buffer, n values (device FP32, x's layout), for the caller's VAE to write.
    float* base(int64_t n);
    /// The mask by position (latent_mask's; empty: none) and the seed's noise (host, x's layout), on the device.
    /// `tokens`: x is [HW][C] (Mage-Flow's tokens) rather than [C][HW].
    void set_mask(const std::vector<float>& mask, const std::vector<float>& noise, int64_t C, int64_t HW, bool tokens);
    /// x (holding the seed's noise) = (1 - s) z0 + s x: the sampler's start at sigma s.
    void start(float* x, float s, void* stream) const;
    /// The sampler's after-step hook with a mask (qi_inpaint_blend); empty without one.
    std::function<void(float*, float)> after(void* stream) const;
    bool masked() const { return mask_ != nullptr; }

    /// The VRAM the buffers take: n latent values, HW positions, with or without a mask.
    static uint64_t bytes(int64_t n, int64_t HW, bool mask) { return (uint64_t) ((mask ? 2 * n + HW : n) * 4); }

private:
    float *z0_ = nullptr, *noise_ = nullptr, *mask_ = nullptr;
    int64_t n_ = 0, C_ = 0, HW_ = 0;
    bool tokens_ = false;
};

}  // namespace strata::image
