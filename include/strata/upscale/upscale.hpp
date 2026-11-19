// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/upscale/upscale.hpp - the upscalers' common frame: a model (UpscaleModel) says what it is and makes
// a clip of frames larger, nothing else; the driver (UpscaleRunner) does what does not depend on the model - the
// resize to the size asked for, the padding to the model's multiples, the tiles, the color correction - so that a
// model added later (SeedVR2, a diffusion model that takes the frames at their target size) takes the same path.
//
// A model is a `clip` model: each clip of frames on its own (an image is a clip of one frame).  Its native scale says
// how it is fed: RealESRGAN x4plus (scale 4) takes the frames at their own size and gives them four times as large, the
// driver then resizing that to the size asked for (Lanczos); a model of scale 0 takes the frames at the size asked for
// (the driver resizes them first, by Lanczos, and pads them to the model's multiple by repeating the edges, then cuts
// the padding off again).  A model with a context (a convolutional one: what a pixel sees is near it) runs on tiles of
// the frame, each with that many pixels of the frame around it, as large as the VRAM allows (the whole frame where it
// fits); the tiles' outputs are put together from their cores.
//
// Lanczos is a model of its own here (scale 0, nothing to run): the plain resize, chosen the same way as the others.
//
// The color correction (color::, img_color.hpp, whose statistics the cut's color match shares) moves the
// output's colors back toward the input's (Lanczos-resized to the output's size): wavelet (the output's high
// frequencies over the input's low ones), lab (the wavelet's, then each CIE Lab channel matched to the input's
// distribution, the lightness mostly kept: SeedVR2's), adain (each RGB channel's mean and spread) or none.  These are
// SeedVR2's names for its corrections (vLLM-Omni's color_correction_method); each
// model has its default.
//
// A video (run_video) goes the same way frame by frame: read a few frames at a time from its file (video_io.hpp), each
// at the size asked for through the model, the colors corrected against the frame it came from, written to a file with
// the source's audio.  A model whose frames depend on the others (temporal(): SeedVR2's) takes the frames in time
// (video_begin, video_push, its frames ready later, video_end); the others each frame on its own (run's way).
#pragma once

#include "strata/image/pipeline.hpp"
#include "strata/video/video_io.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace strata::upscale {

using image::QiRef;

enum class ColorFix { none, lab, adain, wavelet };

/// "none", "lab", "adain" or "wavelet" -> the correction; std::invalid_argument for another word.
ColorFix parse_color(const std::string& word);
const char* color_name(ColorFix c);

/// What a model is, as the driver needs to know it.
struct ModelInfo {
    std::string name;              // the request's word for it ("lanczos", "realesrgan")
    int scale = 0;                 // a native factor (the frames go in at their size); 0: at the size asked for
    int64_t multiple = 1;          // the sides of what it takes, a multiple of this (padded to it)
    int64_t frame_min = 1;         // a clip's frames: frame_min + k * frame_step (the last frame repeated up to it)
    int64_t frame_step = 1;
    int64_t context = 0;           // 0: the frame whole; else tiles, this many pixels of the frame around each core
    int64_t min_tile = 64;         // the smallest core a tile shrinks to before the run fails for want of VRAM
    ColorFix color = ColorFix::none;   // the correction it takes when the request names none
    bool identity = false;         // nothing to run (Lanczos): the driver's resize is all of it
};

class UpscaleModel {
public:
    virtual ~UpscaleModel() = default;
    /// Told a run_clip's share done so far (0..1) as it goes, by a model whose run has parts worth telling (SeedVR2:
    /// its encoder, each DiT block, its decoder); none: nothing told.  UpscaleRunner::run sets it for its run.
    std::function<void(double)> progress;
    virtual const ModelInfo& info() const = 0;
    /// The weights' bytes on the GPU (held between runs until release(true)).
    virtual uint64_t weight_bytes() const = 0;
    /// The least of the GPU's memory the weights run in (a model that streams its layers: those that stay and the
    /// slots the others pass through); weight_bytes where they are all held.
    virtual uint64_t weight_least() const { return weight_bytes(); }
    /// The VRAM the next run may take for its weights and its scratch (a model that streams its layers keeps as many
    /// of them as that leaves room for).
    virtual void set_room(uint64_t bytes) { (void) bytes; }
    /// The bytes a run of a clip of `frames` frames of w x h input pixels takes besides the weights (its scratch), and
    /// its largest single buffer (against the device's largest allocation).
    virtual uint64_t clip_bytes(int64_t w, int64_t h, int64_t frames) const = 0;
    virtual uint64_t clip_largest(int64_t w, int64_t h, int64_t frames) const = 0;
    /// The least scratch such a run fits in: clip_bytes where the model's parts cannot be cut smaller (a model that
    /// cuts its own parts into tiles in the room it is given: their smallest).
    virtual uint64_t clip_least(int64_t w, int64_t h, int64_t frames) const { return clip_bytes(w, h, frames); }
    /// A clip: `in` [frames][h][w][4] RGBA8 on the GPU -> `out` [frames][h * s][w * s][4] (s the native scale, 1 for a
    /// model of scale 0).  The weights go to the GPU first when they are not there; done when it returns.  Throws
    /// std::runtime_error when the GPU has no room (the driver tries a smaller tile).
    virtual void run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) = 0;
    /// Lets the scratch go; `all`: the weights too.
    virtual void release(bool all) = 0;
    /// What it holds on the GPU now.
    virtual uint64_t held() const = 0;
    /// The last run's parts' seconds as key=value words (a model of several parts: SeedVR2's encode, DiT, decode);
    /// empty where it has none to tell.
    virtual std::string parts() const { return {}; }
    /// A model whose frames depend on the others in a video (a video's frames through video_begin .. video_end; the
    /// others' each through run_clip).
    virtual bool temporal() const { return false; }
    /// The temporal models' video, frames of w x h (the model's multiples) RGBA8 on the GPU: begun (about `frames`
    /// of them; chunks of `chunk` frames overlapping by `overlap` where it cuts the video so, 0 its own choice), the
    /// frames pushed in order (read before video_push returns: the caller may write the next into the same place),
    /// those made taken as they are ready, then the end (the rest made ready).
    virtual void video_begin(int64_t w, int64_t h, int64_t frames, uint64_t seed, int64_t chunk, int64_t overlap) {
        (void) w, (void) h, (void) frames, (void) seed, (void) chunk, (void) overlap;
        throw std::logic_error("upscale: not a temporal model");
    }
    virtual void video_push(const uint8_t* in, int64_t n) { (void) in, (void) n; }
    virtual int64_t video_ready() const { return 0; }
    virtual void video_take(uint8_t* out, int64_t n) { (void) out, (void) n; }
    virtual void video_end() {}
    /// A temporal model's VRAM for a video's chunk of `chunk` frames of w x h besides its weights, at its least
    /// (`least`: within the RAM free now for what it keeps there, UINT64_MAX where that is too little) or whole (0
    /// for the others).
    virtual uint64_t video_bytes(int64_t w, int64_t h, int64_t chunk, bool least) const {
        (void) w, (void) h, (void) chunk, (void) least;
        return 0;
    }
};

/// The model `name` ("lanczos", "realesrgan") from its weights' file `path` (none for Lanczos) -> the model;
/// std::invalid_argument for a name it does not know, std::runtime_error for a file that is not the model's.
std::unique_ptr<UpscaleModel> open_model(const std::string& name, const std::string& path);

/// One upscale's request: the size asked for, the correction (`color_set` false: the model's own), a seed (a
/// diffusion model's noise), the largest tile's core (0: as large as the VRAM allows).
struct Request {
    int64_t width = 0, height = 0;
    ColorFix color = ColorFix::none;
    bool color_set = false;
    uint64_t seed = 0;
    int64_t max_tile = 0;
};

/// What a run did: its parts' seconds, the tiles' core and count (0: the frame whole), the correction applied.
struct Report {
    double model_s = 0.0, resize_s = 0.0, color_s = 0.0;
    int64_t tile = 0, tiles = 0;
    ColorFix color = ColorFix::none;
    bool color_applied = false;   // false where the statistics gave nothing to move (a flat image)
};

/// A video's upscale: its frames as Request's, the temporal models' chunks (0: the model's choice) and the file's
/// codec.
struct VideoRequest {
    Request frame;
    int64_t chunk = 0, overlap = 0;
    video::Codec codec = video::Codec::automatic;
    /// Called where the run can wait with nothing in flight (a temporal model's chunk out, a few frames of the
    /// others'): the server lets other requests use the GPU there.
    std::function<void()> pause;
};

/// What a video's run made: the file, its frames and their rate (the source's average: a VFR source was read at it),
/// the parts' seconds.
struct VideoReport {
    std::string path, mime;
    int64_t frames = 0, width = 0, height = 0;
    double fps = 0.0;
    bool vfr = false;
    double read_s = 0.0, model_s = 0.0, resize_s = 0.0, color_s = 0.0, write_s = 0.0;
    ColorFix color = ColorFix::none;
};

class UpscaleRunner {
public:
    /// `src` (RGBA8) at r.width x r.height through `m`, in `avail` bytes of VRAM (the model's weights included): the
    /// tiles from it (halved while an allocation fails, down to the model's min_tile, then std::runtime_error).  The
    /// model's scratch is let go at the end; its weights stay.
    /// `step` is told the share done (0..1): the tiles done, and within a tile what the model tells (progress).
    static QiRef run(UpscaleModel& m, const QiRef& src, const Request& r, uint64_t avail, Report* rep = nullptr,
                     const std::function<void(double)>& step = nullptr);
    /// The VRAM a run of an sw x sh image to w x h takes: `least` with the smallest tiles, `full` the frame whole.
    static uint64_t least_bytes(const UpscaleModel& m, int64_t sw, int64_t sh, int64_t w, int64_t h,
                                ColorFix color);
    static uint64_t full_bytes(const UpscaleModel& m, int64_t sw, int64_t sh, int64_t w, int64_t h, ColorFix color);
    /// The video `in` upscaled to r.frame's size through `m` into `stem` + its container's extension, in `avail`
    /// bytes of VRAM; `step` told each frame written (the frames written, the source's count, an estimate where its
    /// stream gives none), false from it stops the run (std::runtime_error, the file not left).  video::VideoError for
    /// a file it does not read or write.
    static VideoReport run_video(UpscaleModel& m, const std::string& in, const std::string& stem,
                                 const VideoRequest& r, uint64_t avail,
                                 const std::function<bool(int64_t, int64_t)>& step = nullptr);
};

}  // namespace strata::upscale
