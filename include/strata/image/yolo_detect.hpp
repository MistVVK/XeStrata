// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/yolo_detect.hpp - a YOLOv8 detector (the boxes of faces, hands or people, and a person's mask)
// for the parts' fix (the IMG line's DETAIL:, serve/images.py's fix_image), on the GPU.
//
// The weights are a checkpoint converted by tools/image/yolo_pt.py (Bingsu/adetailer's face_yolov8m, hand_yolov8s,
// person_yolov8m-seg): each convolution's weight and batch norm under its module path, and the layer list in the
// header's metadata (`graph`).  The network is built from that list and the tensors' shapes, after the published
// descriptions of YOLOv8: convolutions with their batch norm folded in and SiLU; C2f blocks (a 1 x 1 convolution split
// in two halves, bottlenecks of two 3 x 3 convolutions on the last half, every output concatenated into a 1 x 1
// convolution); SPPF (three 5 x 5 max pools in a row); nearest 2x upsampling and concatenation in the neck; and the
// anchor-free head: per level, the box's four distances as distribution focal loss bins (a softmax over 16 bins, its
// expectation, times the level's stride) and the classes' sigmoids, and for the segmentation head 32 mask
// coefficients and a prototype branch (two 3 x 3 convolutions around a 2 x 2 transposed one, then a 1 x 1).
//
// The picture is letterboxed to 640 on its longer side (bilinear, kept ratio, the short side padded to a multiple of
// 32 with grey 114, the padding split in two) on the GPU; the k x k convolutions go through oneDNN (qi_conv) where
// the build and the GPU take it, else, and the 1 x 1 ones, through prefill::Gemm (k x k unfolded); the activations,
// pools, upsampling, concatenations, the head's decoding and the masks are the engine's kernels.  The candidates above
// the threshold go through non-maximum suppression on the host (IoU 0.7, every class together, 300 at most), and the
// boxes are mapped back to the picture and clipped to it.
#pragma once

#include "strata/image/pipeline.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

struct Detection {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // the box on the picture, its pixels' edges (x1, y1 past the last)
    float score = 0;
    int cls = 0;
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    /// Reads the converted checkpoint and puts its weights on the GPU.  False, with `err`, when the file is not one
    /// (no layer list, a module or shape it does not take) or the GPU has no room.
    bool open(const std::string& path, std::string& err);
    const std::string& path() const { return path_; }
    /// Whether the head gives masks (the segmentation head).
    bool masks() const { return seg_; }
    int classes() const { return nc_; }
    /// The class names from the checkpoint ("face", "hand", "person"), by index.
    const std::vector<std::string>& names() const { return names_; }

    /// The detections of `img` (RGBA8) with a score of `conf` or more, highest first.  With `masks` (and a
    /// segmentation head): each detection's mask at the picture's size, 1 inside the person and the box, 0 elsewhere.
    std::vector<Detection> detect(const QiRef& img, float conf, std::vector<std::vector<uint8_t>>* masks = nullptr,
                                  float iou = 0.7f, int max_det = 300, int imgsz = 640);

    /// The VRAM the weights hold, and the most a detection of a picture at most `imgsz` on its longer side adds.
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t work_bytes(int imgsz = 640) const;
    /// Frees the buffers of the last detection (the weights stay).
    void drop_scratch();

    struct Net;

private:
    std::string path_;
    std::unique_ptr<Net> net_;
    prefill::Gemm gemm_;
    bool seg_ = false;
    int nc_ = 0;
    std::vector<std::string> names_;
    uint64_t weight_bytes_ = 0;
};

}  // namespace strata::image
