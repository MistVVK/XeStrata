// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/cnet_prep.hpp - the control images ControlNet and the edit models' references take (the
// engine's CONTROL line, serve/images.py's control), made on the GPU from a picture: Canny's edges, the grey picture,
// scribbles, and the networks' maps (cnet_models.hpp: HED's soft edges and their strokes, line art, M-LSD's straight
// lines, Depth Anything V2's depth, YOLOX's and DWPose's skeletons).
//
// The picture (its alpha over white) is scaled for its shorter side to be `resolution`, each side taken to the nearest
// multiple of 64 (shrunk by area, grown by cubic convolution), and processed there: the map is RGB at that size, the
// caller resizes it to the image it conditions (the depth alone runs at the picture's own size first).
//
// canny: Canny's edges (1986) of the luma: Sobel's gradient, the non-maximum suppression across the edge in four
// directions, `low` and `high` on the gradient's length, the hysteresis over the 8 neighbours.  gray: luma by ITU-R
// BT.601.  scribble: the picture's own dark lines (its luma below the middle grey) white on black.  scribble_xdog: the
// extended difference of Gaussians (Winnemoeller et al. 2012) of the luma, thresholded.
#pragma once

#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

class DepthAnything2;
class BaeNormal;
class UniformerSeg;
class HedNet;
class LineartNet;
class MlsdNet;
class YoloxNet;
class DwPoseNet;

struct ControlParams {
    int resolution = 512;                          // the shorter side the picture is processed at
    float low = 100.f, high = 200.f;               // canny's thresholds (of 0 .. 255 values)
    float threshold = 12.f;                        // scribble_xdog's (of 0 .. 255 values)
    bool safe = true;                              // HED's edges in steps
    float score = 0.1f, distance = 0.1f;           // M-LSD's: a centre's score and a segment's length to keep it
    bool body = true, hands = true, face = true;   // the pose map's parts
};

/// A kind of control image and the network it needs ("": none; "pose": YOLOX and DWPose).
struct ControlKind {
    const char* name;
    const char* net;
};

/// The networks a run may use, opened by the caller (strata_image's auxiliary models), null where not given.
struct ControlNets {
    DepthAnything2* depth = nullptr;
    BaeNormal* normal = nullptr;
    UniformerSeg* segmentation = nullptr;
    HedNet* hed = nullptr;
    LineartNet* lineart = nullptr;
    MlsdNet* mlsd = nullptr;
    YoloxNet* yolox = nullptr;
    DwPoseNet* dwpose = nullptr;
};

/// A control map (RGBA8, its alpha over white) resized to w x h (Keys's cubic convolution, pixel centres, the edges
/// repeated), RGBA8 with alpha 255, on the GPU.
std::vector<uint8_t> control_resize(const QiRef& img, int64_t w, int64_t h);

class ControlPrep {
public:
    ControlPrep();
    ~ControlPrep();
    ControlPrep(const ControlPrep&) = delete;
    ControlPrep& operator=(const ControlPrep&) = delete;

    static const std::vector<ControlKind>& kinds();
    static const ControlKind* find(const std::string& kind);

    /// The control map of `img` (RGBA8; its alpha over white): RGB as RGBA8 (alpha 255) at the processed size.
    /// Throws std::invalid_argument for a kind it does not make, a parameter out of range or a network not given.
    QiRef run(const std::string& kind, const QiRef& img, const ControlParams& p, const ControlNets& nets);

    /// The VRAM the steps outside the networks take for a picture of width x height.
    static uint64_t work_bytes(int64_t width, int64_t height, const ControlParams& p);
    /// Frees its buffers.
    void release();

    struct State;

private:
    std::unique_ptr<State> st_;
};

}  // namespace strata::image
