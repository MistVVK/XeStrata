// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/image/sam3_video.hpp - SAM 3.1's video segmentation: every frame the detector finds the words'
// instances, the multiplex tracker (sam3.hpp's video section) carries the objects found before, and the two are
// matched.  The rules follow vLLM-Omni's sam3_video.py (2b610b3, Apache-2.0: its association, hot start, occlusion
// suppression and delayed outputs), and XeStrata's own: masks compared by their intersection over the smaller one
// (IoM), detections whose centers are near the frame's edges left out, a masklet shown only after a run of matched
// frames, reconditioning only of one-to-one pairs, up to 16 objects in buckets of 16 slots.
//
// A clip goes in a frame at a time (push); each frame's objects come out confirm_after - 1 frames later (a masklet
// shown once it is confirmed: the outputs need not wait the whole hot start, only the unconfirmed ones are hidden,
// which that many frames tell), the rest at finish().  A masklet the hot start removes stops being tracked.
//
// Several words at once are vLLM-Omni's: each frame the detector runs for each, a detection matches only its own
// words' masklets, and the occlusion, the memory's overlaps and the outputs' are settled among one words' objects.
// XeStrata adds words that start on a later frame, and clicks (points and boxes) on any frame: the tracker's
// prompts (its interactive decoder on the frame without memory), which add an object or redo one's mask there; such
// an object is shown at once, kept whatever the detector finds, and tracked from that frame on.
#pragma once

#include "strata/image/pipeline.hpp"
#include "strata/image/sam3.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace strata::image {

/// The video segmentation's settings (the thresholds the multiplex checkpoint was tuned to work with).
struct Sam3VideoConfig {
    float score_threshold = 0.4f;    // a detection kept
    float nms_iom = 0.1f;            // detections overlapping a better one past this (IoM) dropped
    float assoc = 0.1f;              // a detection matches a masklet (IoM)
    float trk_assoc = 0.5f;          // a masklet counts as matched (IoM)
    float new_det = 0.65f;           // an unmatched detection starts an object
    int hot_start = 15;              // frames an object may still be removed in
    int hot_unmatched = 8, hot_duplicate = 8;
    float occlusion = 0.7f;          // overlapping masklets (IoU): the more recently occluded one hidden
    float edge = 0.025f;             // detections whose center is within this share of the frame of an edge left out
    int recondition_every = 16;
    float recondition_iom = 0.5f, high_conf = 0.8f;
    int confirm_after = 3;           // frames in a row a masklet must be matched in before it is shown
    int max_objects = 16;
    int max_cond_frames = 4, max_pointers = 16;
};

/// Words (CLIP's tokens) looked for from frame `from` on.
struct Sam3VideoWords {
    std::vector<int32_t> ids;
    int64_t from = 0;
};

/// Clicks on frame `frame`: points (x, y in the clip's pixels; label 1 in, 0 out) and a box's corners (labels 2 and
/// 3, first), for a new object (object -1) or to redo the mask of the object of that id.
struct Sam3VideoClick {
    int64_t frame = 0;
    int64_t object = -1;
    std::vector<std::array<float, 3>> points;
};

class Sam3Video {
public:
    using Config = Sam3VideoConfig;
    /// A frame's result: its index, each shown object's id, what made it (its words' index, or its click's), its first
    /// detection's score (1 a click's), its mask (width x height, 255 inside) and box (pixels, x1 y1 past the last).
    struct Object {
        int64_t id = 0;
        int64_t words = -1, click = -1;
        float score = 0;
        std::vector<uint8_t> mask;
        int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    };
    struct Frame {
        int64_t index = 0;
        std::vector<Object> objects;
    };

    /// A clip of `frames` frames (an estimate where the stream does not say) of width x height, the words and the
    /// clicks looked for (at most 16 words).
    Sam3Video(Sam3& sam, int64_t width, int64_t height, int64_t frames, std::vector<Sam3VideoWords> words,
              std::vector<Sam3VideoClick> clicks = {}, Config c = Config());
    ~Sam3Video();
    Sam3Video(const Sam3Video&) = delete;
    Sam3Video& operator=(const Sam3Video&) = delete;

    /// The next frame -> the frames now final (each hot_start frames behind).
    std::vector<Frame> push(const QiRef& frame);
    /// The clip's end -> the frames still held.
    std::vector<Frame> finish();

    /// Each part's seconds so far: the trunk, the detector, the tracker (the memory attention and the decoder), the
    /// memory (the memory encoder and the masks given as clean ones), the rules and the outputs.
    struct Times {
        double trunk = 0, detect = 0, track = 0, memory = 0, rules = 0, output = 0;
    };
    const Times& times() const { return t_; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
    Times t_;
};

}  // namespace strata::image
