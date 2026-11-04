// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/img_crop.hpp - Crop and Stitch: a mask's edit drawn at the size the model draws well and put
// back into the picture, with the steps and defaults of lquesada's ComfyUI-Inpaint-CropAndStitch (InpaintCropImproved
// and InpaintStitchImproved; GPL-3.0, read, not copied).
//
// The server (serve/images.py, plan_crop) chooses the box and the size the model draws; the engine does the rest.  The
// mask, as its edit share (1 where the picture changes), has its faint values taken out and its holes filled on the
// host (mask_hipass_filter, fillholes_iterative_hipass_fill_m), then on the GPU is grown by a square window and blurred
// (mask_blend_pixels: a window of ceil(blend / 4 * 1.5 + 1) pixels, a Gaussian of sigma blend / 8 cut at 4 sigma,
// scipy's reflected edges).  The box is cut out of the picture, its parts past the picture's edges the edges' copies
// (the mask 1 there), and resized to the drawing's size (Lanczos; the mask bilinear), and the mask blurred again there:
// the blend mask.  The model redraws where the blend mask is
// above 0, its latent positions following the base image as much as the resized mask keeps them (share), and its
// picture, resized back to the box (Lanczos), goes into the picture by the blend mask (bilinear): where that is 0 the
// picture's own pixels stay as they are, byte for byte.  The resizes are Pillow's (its filters, its supports widened by the scale when it shrinks), in floats.
#pragma once

#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

struct CropBox {
    int64_t x = 0, y = 0, w = 0, h = 0;   // the box on the picture, which may reach past its edges
    int64_t blend = 32;                    // mask_blend_pixels: the mask's growth and blur, in the picture's pixels
    int64_t width = 0, height = 0;         // the picture's size, as the server read it
    // the first reference beside the drawing's base, the box at this size (an ai-toolkit MOSS-Image edit's, whose
    // references are smaller than the drawing); 0: the base is the reference
    int64_t ref_w = 0, ref_h = 0;
};

/// The IMG and NEED lines' CROP:<x>,<y>,<w>,<h>,<blend>,<width>,<height>[,<ref_w>,<ref_h>] -> the box;
/// std::invalid_argument when the word is not that.
CropBox parse_crop(const std::string& word);

class CropStitch {
public:
    /// Cuts `box` out of `base` (RGBA8, box.width x box.height) for a drawing of gw x gh, by `mask` (OpenAI's alpha over
    /// base: 0 where the picture changes).  Throws std::invalid_argument when the sizes do not agree, and
    /// std::runtime_error when the GPU has no room for the steps.
    void prepare(const QiRef& base, const QiRef& mask, const CropBox& box, int64_t gw, int64_t gh);
    /// The drawing's base image (the box at gw x gh) and its mask (alpha 0 where the model redraws, 255 elsewhere).
    const QiRef& crop() const { return crop_; }
    const QiRef& regen() const { return regen_; }
    /// The box at the box's ref_w x ref_h (Lanczos, from the picture as the base is), when it asks for one.
    const QiRef& ref() const { return ref_; }
    /// The drawing's edit share (the grown and blurred mask at gw x gh, before the blend mask's blur): the sampler's
    /// mask (QiRequest::mask_share), as ComfyUI's noise mask is the cropped mask, so that the blurred edge follows the
    /// picture rather than being drawn anew and blended over it.
    const std::vector<float>& share() const { return share_; }
    /// The drawing (RGBA8, gw x gh) put back into the picture -> the picture (RGBA8, at its own size).  `color`
    /// (QiRequest::crop_color): the drawing's colors matched first, as ComfyUI's ColorTransfer does (reinhard_lab:
    /// each Lab channel's mean and standard deviation): 1 by the pixels the blend mask keeps (the drawing's drift from
    /// the picture there, which a change asked inside the paint does not move), 2 by the whole cut; none where too
    /// few pixels or a flat channel give no statistics.
    std::vector<uint8_t> stitch(const std::vector<uint8_t>& made, int color = 0) const;
    /// The VRAM the steps take at most, for a picture of width x height, a box of w x h, a drawing of gw x gh and a
    /// reference of rw x rh (none of it is held while the model draws).
    static uint64_t bytes(int64_t width, int64_t height, int64_t w, int64_t h, int64_t gw, int64_t gh, int64_t rw = 0,
                          int64_t rh = 0);

private:
    std::vector<uint8_t> base_;   // the picture, RGBA8
    std::vector<float> blend_;    // the blend mask at gw x gh
    std::vector<float> share_;    // the edit share at gw x gh
    CropBox box_;
    int64_t gw_ = 0, gh_ = 0;
    QiRef crop_, regen_, ref_;
};

}  // namespace strata::image
