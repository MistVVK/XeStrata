// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/img_crop.hpp - a mask's edit cut out, drawn at the size the model draws well and put back into
// the picture.
//
// The server (serve/images.py, plan_crop) chooses the box and the size the model draws; the engine does the rest.  The
// mask, as its edit share (1 where the picture changes), loses its faintest strokes and has its holes filled with the
// level that encloses them on the host (edit_share), then on the GPU grows by a disc of an eighth of the blend width
// and is softened by three box means of an eighth of it each: the blend mask.  The box is cut out of the picture, its
// parts past the picture's edges the edges' copies (the mask 1 there), and resized to the drawing's size (Lanczos; the
// mask bilinear).  The model redraws where the blend mask is above 0, its latent positions following the base image
// as much as the mask keeps them (share), and its picture, resized back to the box (Lanczos), goes into the picture
// by the blend mask (bilinear): where that is 0 the picture's own pixels stay as they are, byte for byte.  The
// resizes are Pillow's (its filters, its supports widened by the scale when it shrinks), in floats on the GPU
// (pillow_resize); a picture a model's encoder reads as floats before they go up is resized on the host by Pillow's
// own 8-bit arithmetic (pillow_resize_planes), the same bytes as PIL's.
#pragma once

#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

struct CropBox {
    int64_t x = 0, y = 0, w = 0, h = 0;   // the box on the picture, which may reach past its edges
    int64_t blend = 32;                    // the mask's growth and softening, in the picture's pixels
    int64_t width = 0, height = 0;         // the picture's size, as the server read it
    // the first reference beside the drawing's base, the box at this size (an ai-toolkit MOSS-Image edit's, whose
    // references are smaller than the drawing); 0: the base is the reference
    int64_t ref_w = 0, ref_h = 0;
};

/// Pillow's resampling filters (its LANCZOS, BILINEAR and BICUBIC).
/// Pillow's filters, and OpenCV's INTER_CUBIC (cv_cubic: a = -0.75, 4 taps whatever the scale, the edges' samples
/// repeated past them: SAM 3's video frames').
enum class Filter { lanczos, triangle, bicubic, cv_cubic };

/// `src` (RGBA8) at w x h by Pillow's filter, its support widened by the scale when it shrinks, the rows' pass rounded
/// and clipped to 8 bits before the columns' as Pillow's is (its fixed-point weights aside: an output within 1 of its).
QiRef pillow_resize(const QiRef& src, int64_t w, int64_t h, Filter filter);

/// The resize of `planes`, planar floats `[C][H][W]` whose values are the whole bytes of a picture (0 .. 255), read by
/// a model's encoder on the host before they go up, as PIL's Image.resize gives it for an 8-bit picture (L or RGB),
/// byte for byte: Pillow's fixed-point weights and totals (libImaging/Resample.c), each pass rounded and clipped to 8
/// bits, the rows first (the columns first where Image.resize takes them first: a picture over a hundred times
/// taller than wide losing height).  The GPU's pillow_resize reads float weights and totals in floats: up to one level
/// apart from it.  Lanczos, the triangle (bilinear) or the cubic (bicubic); std::invalid_argument for cv_cubic (not
/// Pillow's), an empty image or no size; `planes` itself for a resize to the size it is.
std::vector<float> pillow_resize_planes(const std::vector<float>& planes, int64_t c, int64_t w, int64_t h, int64_t w2,
                                        int64_t h2, Filter filter);

/// `src` (RGBA8) at w x h by Pillow's Lanczos, its support widened by the scale when it shrinks (an upscale's base
/// image: the IMG line's UPSCALE).
QiRef lanczos_resize(const QiRef& src, int64_t w, int64_t h);
/// The same on the GPU: `src` (RGBA8, sw x sh, a device pointer) into `dst` (RGBA8, w x h), on the image engine's
/// queue (the runtime's default stream), done when it returns; std::runtime_error when the GPU has no room for its
/// scratch (lanczos_resize_bytes).
void lanczos_resize_device(const uint8_t* src, int64_t sw, int64_t sh, uint8_t* dst, int64_t w, int64_t h);
uint64_t lanczos_resize_bytes(int64_t sw, int64_t sh, int64_t w, int64_t h);

/// The IMG and NEED lines' CROP:<x>,<y>,<w>,<h>,<blend>,<width>,<height>[,<ref_w>,<ref_h>] -> the box;
/// std::invalid_argument when the word is not that.
CropBox parse_crop(const std::string& word);

class CropStitch {
public:
    /// Cuts `box` out of `base` (RGBA8, box.width x box.height) for a drawing of gw x gh, by `mask` (OpenAI's alpha over
    /// base: 0 where the picture changes).  Throws std::invalid_argument when the sizes do not agree, and
    /// std::runtime_error when the GPU has no room for the steps.
    /// `keep`: the cut kept on the GPU too, for stitch's color match (not copied up again: 2.8 ms of a 1 MP cut over
    /// the RTX 4070's PCIe x4).  `scaled`: the box already at gw x gh (an upscaler's, the request's crop_upscale), taken
    /// for the drawing's base instead of the box resized here (Lanczos); the mask is resized here either way.
    void prepare(const QiRef& base, const QiRef& mask, const CropBox& box, int64_t gw, int64_t gh, bool keep = false,
                 const QiRef* scaled = nullptr);
    /// The box of `base` at its own size (RGBA8 rows, the picture's edge repeated where the box passes it): what an
    /// upscaler enlarges for prepare's `scaled`.
    static QiRef cut(const QiRef& base, const CropBox& box);
    CropStitch() = default;
    CropStitch(const CropStitch&) = delete;
    CropStitch& operator=(const CropStitch&) = delete;
    ~CropStitch();
    /// The drawing's base image (the box at gw x gh) and its mask (alpha 0 where the model redraws, 255 elsewhere).
    const QiRef& crop() const { return crop_; }
    const QiRef& regen() const { return regen_; }
    /// The box at the box's ref_w x ref_h (Lanczos, from the picture as the base is), when it asks for one.
    const QiRef& ref() const { return ref_; }
    /// The drawing's edit share (the blend mask at gw x gh): the sampler's mask (QiRequest::mask_share), so that the
    /// soft edge follows the picture rather than being drawn anew and blended over it.
    const std::vector<float>& share() const { return share_; }
    /// The drawing (RGBA8, gw x gh) put back into the picture -> the picture (RGBA8, at its own size).  `color`
    /// (QiRequest::crop_color): the drawing's colors matched first (Reinhard's transfer in CIE Lab: each channel's mean
    /// and standard deviation): 1 by the pixels the blend mask keeps (the drawing's drift from
    /// the picture there, which a change asked inside the paint does not move), 2 by the whole cut; none where too
    /// few pixels or a flat channel give no statistics.
    std::vector<uint8_t> stitch(const std::vector<uint8_t>& made, int color = 0) const;
    /// The VRAM the steps take at most, for a picture of width x height, a box of w x h, a drawing of gw x gh and a
    /// reference of rw x rh (none of it is held while the model draws but prepare's kept cut, 4 bytes a pixel of
    /// gw x gh).
    static uint64_t bytes(int64_t width, int64_t height, int64_t w, int64_t h, int64_t gw, int64_t gh, int64_t rw = 0,
                          int64_t rh = 0);

private:
    std::vector<uint8_t> base_;   // the picture, RGBA8
    std::vector<float> blend_;    // the blend mask at gw x gh
    std::vector<float> share_;    // the edit share at gw x gh
    CropBox box_;
    int64_t gw_ = 0, gh_ = 0;
    QiRef crop_, regen_, ref_;
    uint8_t* crop_dev_ = nullptr;   // the cut on the GPU (prepare's keep)
};

}  // namespace strata::image
