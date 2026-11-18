// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/cnet_models.hpp - the networks behind the control images (cnet_prep.cpp runs them, the engine's CONTROL
// line): each an AuxMapNet (aux_net.hpp) on a safetensors file, its weights on the GPU at its first run, and a run from
// an RGB8 picture on the host to its map.  Written from the papers and the weights' names and shapes.
#pragma once

#include "strata/image/aux_net.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

/// Depth Anything V2 Small (Yang et al. 2024: DINOv2's ViT-S/14 and a DPT head; the weights by transformers' names):
/// the relative depth of a picture (near large) at its own size.  The picture goes in at 518 on its shorter side, each
/// side a multiple of 14, ImageNet's normalization; the depth comes out bilinearly at the picture's size.
class DepthAnything2 : public AuxMapNet {
public:
    explicit DepthAnything2(const std::string& path) : AuxMapNet(path, "Depth Anything V2") {}
    std::vector<float> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H);
    /// The VRAM a run on a picture of W x H takes beyond the weights.
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    struct Layer {
        Norm n1, n2;
        const float *ls1 = nullptr, *ls2 = nullptr;
        Lin qkv, o, fc1, fc2;
    };
    void sides(int64_t W, int64_t H, int64_t& nw, int64_t& nh) const;
    int64_t C_ = 0, heads_ = 0, patch_ = 14, grid_ = 0, kp_ = 0;
    Lin patch_w_;
    std::vector<float> pos_, cls_;   // the position embeddings [1 + n n][C] and the class token, on the host
    std::vector<Layer> layers_;
    Norm final_;
    MConv proj_[4], resize_[4], neck_[4], fuse_[4], rcu_[4][4], head1_, head2_, head3_;
    int resize_kind_[4] = {0, 0, 0, 0};   // 1: transposed (kernel = stride), -1: a convolution of stride 2, 0: none
};

/// BAE surface normals (Bae et al. 2021): an EfficientNet-B5 encoder and a progressively refined decoder.  The
/// output is a camera-space normal map, RGB8, at the picture's size.
class BaeNormal : public AuxMapNet {
public:
    explicit BaeNormal(const std::string& path) : AuxMapNet(path, "BAE normal") {}
    std::vector<uint8_t> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H);
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    struct Block {
        MConv expand, depthwise, reduce, gate, project;
        bool has_expand = false, residual = false, same_pad = false;
    };
    struct Up { MConv first, second; };
    MConv stem_, head_, bottleneck_, initial_;
    std::vector<Block> blocks_[7];
    Up up_[4];
    MConv refine_[3][4];
};

/// UniFormer Small with UPerNet (Li et al. 2022): ADE20K's 150 classes as an RGB8 color map.
class UniformerSeg : public AuxMapNet {
public:
    explicit UniformerSeg(const std::string& path) : AuxMapNet(path, "UniFormer segmentation") {}
    std::vector<uint8_t> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H);
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    struct Batch { const float* scale = nullptr; const float* bias = nullptr; };
    struct Block {
        MConv pos, conv1, depthwise, conv2, fc1, fc2;
        Batch bn1, bn2;
        Norm norm1, norm2;
        Lin qkv;
    };
    struct Stage {
        MConv patch;
        Norm patch_norm, output_norm;
        std::vector<Block> blocks;
    };
    Stage stages_[4];
    MConv psp_[4], bottleneck_, lateral_[3], fpn_[3], fpn_bottleneck_, classes_;
    const float* palette_ = nullptr;
};

/// HED's soft edges (Xie and Tu 2015; the ControlNet 1.1 annotator's ControlNetHED weights: five VGG-like blocks,
/// each a side output): an RGB8 picture [H][W][3] to its edges [H][W], 8 bits.  `safe` quantizes them to four steps;
/// `scribble` thins them into strokes.
class HedNet : public AuxMapNet {
public:
    explicit HedNet(const std::string& path) : AuxMapNet(path, "HED") {}
    std::vector<uint8_t> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H, bool safe, bool scribble);
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    const float* norm_ = nullptr;
    std::vector<MConv> convs_[5];
    MConv proj_[5];
};

/// Line art (Chan, Durand and Isola 2022, Informative Drawings: a ResNet generator with instance norms; sk_model, or
/// sk_model2 for the coarse lines): an RGB8 picture to its lines, white on black, 8 bits.
class LineartNet : public AuxMapNet {
public:
    explicit LineartNet(const std::string& path) : AuxMapNet(path, "line art") {}
    std::vector<uint8_t> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H);
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    MConv in_, down_[2], up_[2], out_;
    std::vector<std::pair<MConv, MConv>> res_;
};

/// M-LSD (Gu et al. 2022, the large MobileNetV2 model): a picture's straight line segments drawn one pixel wide, white
/// on black, at its size.  A segment is kept where its centre's score passes `score` and its length `dist` (in the
/// model's half-size map's pixels).
class MlsdNet : public AuxMapNet {
public:
    explicit MlsdNet(const std::string& path) : AuxMapNet(path, "M-LSD") {}
    std::vector<uint8_t> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H, float score, float dist);
    uint64_t need_bytes(int64_t W, int64_t H) const;

protected:
    void load_weights() override;

private:
    struct Inverted {
        MConv expand, dw, project;
        bool has_expand = false, residual = false, tf_pad = false;
    };
    MConv first_;
    std::vector<Inverted> blocks_;
    std::vector<int> taps_;   // the backbone's blocks whose outputs the decoder takes
    struct Merge {            // a skip's and the deeper map's 1 x 1 convolutions, the deeper one upsampled 2x first
        MConv skip, deep;
        bool up = true;
    } merge_[4];
    struct Refine {           // a residual 3 x 3, then a 3 x 3 to the next width
        MConv c1, c2;
    } refine_[4];
    MConv dil_, c3_, out_;
};

/// A detected box on the picture (pixels), with its score.
struct ControlBox {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0, score = 0;
};

/// YOLOX-L (Ge et al. 2021: CSPDarknet with a Focus stem, PAFPN, decoupled anchor-free heads; Megvii's COCO weights):
/// the people in a picture (COCO's class 0), for DWPose.  The picture is letterboxed into 640 x 640 in the colour order
/// the weights were trained on (BGR, values 0 .. 255; the rest 114).
class YoloxNet : public AuxMapNet {
public:
    explicit YoloxNet(const std::string& path) : AuxMapNet(path, "YOLOX") {}
    std::vector<ControlBox> people(const std::vector<uint8_t>& rgb, int64_t W, int64_t H);
    uint64_t need_bytes() const;

protected:
    void load_weights() override;

private:
    struct Csp {
        MConv c1, c2, c3;
        std::vector<std::pair<MConv, MConv>> m;
        bool shortcut = true;
    };
    Csp load_csp(const std::string& p, bool shortcut);
    Map csp(const Csp& c, const Map& x);
    Map up2(const Map& x);
    MConv conv(const std::string& p, int stride);
    MConv stem_, down_[4], spp1_, spp2_, lateral0_, reduce1_, bu2_, bu1_;
    Csp dark_[4], p4_, p3_, n3_, n4_;
    MConv stems_[3], cls_[3][2], reg_[3][2], cls_pred_[3], reg_pred_[3], obj_pred_[3];
};

/// One person's keypoints (COCO-WholeBody's 133: 17 of the body, 6 of the feet, 68 of the face, 21 of each hand),
/// each (x, y, score) on the picture.
using WholeBody = std::array<std::array<float, 3>, 133>;

/// DWPose (Yang et al. 2023: RTMPose-l's CSPNeXt backbone and RTMCC head with SimCC outputs, distilled for
/// COCO-WholeBody at 288 x 384): the keypoints of each box's person (the whole picture where there is no box).
class DwPoseNet : public AuxMapNet {
public:
    explicit DwPoseNet(const std::string& path) : AuxMapNet(path, "DWPose") {}
    std::vector<WholeBody> run(const std::vector<uint8_t>& rgb, int64_t W, int64_t H,
                               const std::vector<ControlBox>& boxes);
    uint64_t need_bytes() const;

protected:
    void load_weights() override;

private:
    struct Block {
        MConv c1, dw, pw;
    };
    struct Stage {
        MConv down, spp1, spp2, main, shortc, fin;
        bool spp = false, identity = true;
        std::vector<Block> blocks;
        const float *att_w = nullptr, *att_b = nullptr;   // the channel attention's 1 x 1 convolution (FP32)
    };
    MConv conv(const std::string& p, int stride, bool depthwise = false);
    MConv stem_[3];
    Stage stages_[4];
    MConv final_;
    float mlp_g_ = 1.f, gau_g_ = 1.f;
    Lin wm_, wuv_, wo_, wx_, wy_;   // the head's products
    const float *gamma_ = nullptr, *beta_ = nullptr, *res_scale_ = nullptr;
    static constexpr int kResultSlot = 250, kPictureSlot = 251;   // the keypoints read back, the picture
    int64_t in_w_ = 288, in_h_ = 384, K_ = 133, E_ = 512, S_ = 128, D_ = 256;
};

/// The pose map: each person's body, hands and face drawn as DWPose draws its skeletons (the limbs as coloured
/// ellipses, the hands' bones in a hue wheel, the face's points white) on black, W x H, RGB8.  A keypoint is drawn
/// where its score passes 0.3.
std::vector<uint8_t> draw_pose(const std::vector<WholeBody>& people, int64_t W, int64_t H, bool body, bool hands,
                               bool face);

}  // namespace strata::image
