// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/image/sam3.hpp - SAM 3's detector (Meta's Segment Anything 3; the weights under the SAM License,
// downloaded when first needed after the user agreed to it): the words-to-masks model of the image tools
// (find_region, edit_region, /v1/images/segmentations).  Its parts follow vLLM-Omni's port
// (vllm_omni/diffusion/models/sam3/sam3_model.py and pipeline_sam3.py at 15bb15d, Apache-2.0, itself after
// transformers' modeling_sam3.py), written again for the engine:
//   * the picture resized to 1008 x 1008 (bilinear, its support widened as it shrinks: torchvision's antialiased
//     resize), normalized to [-1, 1];
//   * the ViT: a 14 x 14 patch embedding, the pretraining's 24 x 24 position table tiled over the 72 x 72 patches, a
//     layer norm, then 32 layers (pre-norm; windows of 24 x 24 but in layers 7, 15, 23 and 31, which see every patch)
//     whose attention turns q and k by a 2D axial RoPE (the window's positions; the global layers' scaled onto the
//     window's grid) and whose MLP is a GELU's;
//   * the neck: the patches at 4x (two 2 x 2 transposed convolutions around a GELU), 2x (one) and 1x, each through a
//     1 x 1 and a 3 x 3 convolution to 256 channels;
//   * the words: CLIP's text tower (24 causal layers of 1024, GELU), its last states through a linear layer to 256;
//   * the DETR encoder: 6 layers over the 1x level's patches with its sine positions (self-attention, the words by
//     cross-attention, a ReLU MLP);
//   * the DETR decoder: 200 queries and a presence token over 6 post-norm layers (self-attention, the words, the
//     patches with a relative position bias from each query's box), each refining the queries' boxes;
//   * the scores: each query's dot product with the words' pooled projection, times the presence token's sigmoid;
//   * the masks: the encoder's patches after a cross-attention to the words, a pixel decoder up to the 4x level (nearest
//     upsampling, 3 x 3 convolutions, group norms, ReLU), its 1 x 1 projection by each kept query's mask embedding,
//     the logits' sigmoid resized to the picture (bilinear) and thresholded.
// The products go through prefill::Gemm, the attention through qi_sdpa_ex, the convolutions through qi_conv's
// libraries or unfolded products; the rest is the engine's kernels (aux_kernels.hpp), the few boxes on the host.
#pragma once

#include "strata/image/aux_net.hpp"
#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

struct Segment {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // the box on the picture (pixels, x1 and y1 past the last)
    float score = 0;
    std::vector<uint8_t> mask;              // width x height, 255 inside
};

class Sam3 : public AuxNet {
public:
    explicit Sam3(const std::string& path);

    struct Times {
        double vision = 0, text = 0, detect = 0, masks = 0;
    };
    /// The instances of the words `ids` (CLIP's tokens, <|startoftext|> .. <|endoftext|>, at most 32) in `image`
    /// whose score passes `score_threshold`, highest first, each with its mask (`mask_threshold` on the
    /// probability) at the picture's size.
    std::vector<Segment> run(const QiRef& image, const std::vector<int32_t>& ids, float score_threshold,
                             float mask_threshold, Times* t = nullptr);

    /// What a run needs on the GPU beyond what it holds now (the weights when not loaded, the scratch not grown).
    uint64_t need_bytes(int64_t width, int64_t height) const;

private:
    void load_weights() override;
    void vision(const QiRef& image);
    void text(const std::vector<int32_t>& ids);
    void attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int64_t lq, int64_t lk, int64_t heads,
                   int64_t hd, int64_t q_row, int64_t k_row, int64_t v_row, uint16_t* out, int64_t batch = 1,
                   const float* bias = nullptr, bool causal = false);

    struct VitLayer {
        Norm n1, n2;
        Lin qkv, o, fc1, fc2;
        bool global = false;
    };
    struct TextLayer {
        Norm n1, n2;
        Lin qkv, o, fc1, fc2;
    };
    struct Attn {   // nn.MultiheadAttention's packed projections, split
        Lin q, k, v, o;
    };
    struct EncLayer {
        Norm n1, n2, n3;
        Attn self, cross;
        Lin fc1, fc2;
    };
    struct DecLayer {
        Attn self, text, vis;
        Norm n_self, n_text, n_vis, n_mlp;
        Lin fc1, fc2;
    };
    struct Fpn {
        Lin up1, up2;   // the transposed convolutions (none, one or two)
        int ups = 0;
        Conv c1, c3;
    };
    Attn attn(const std::string& name);

    // the ViT
    Conv patch_conv_;
    const float* pos_ = nullptr;   // the 24 x 24 table, [576][1024]
    Norm ln_pre_;
    std::vector<VitLayer> vit_;
    std::vector<Fpn> fpn_;
    // the words
    const uint16_t* tok_ = nullptr;     // [vocab][1024] FP16
    const uint16_t* tpos_ = nullptr;    // [32][1024] FP16
    std::vector<TextLayer> txt_;
    Norm ln_final_;
    Lin resizer_;
    // the detector
    std::vector<EncLayer> enc_;
    std::vector<DecLayer> dec_;
    Norm dec_out_, pres_norm_;
    Lin box1_, box2_, box3_, pres1_, pres2_, pres3_, ref1_, ref2_;
    const float* rpb_ = nullptr;        // the box bias's MLPs (aux_box_bias's layout)
    int64_t rpb_hidden_ = 0;
    std::vector<float> query_embed_, ref_points_, presence_tok_;
    Lin sc_fc1_, sc_fc2_, sc_proj_, sc_qproj_;
    Norm sc_norm_;
    const float* sine_pos_ = nullptr;   // the 1x level's sine positions [grid^2][d]
    // the masks
    Attn m_cross_;
    Norm m_norm_;
    Conv pix_conv_[2];
    Norm pix_gn_[2];
    Lin inst_, emb1_, emb2_, emb3_;

    // the model's shape (the checkpoint's, checked as it loads)
    int64_t vocab_ = 49408, wd_ = 1024, td_ = 1024, d_ = 256, grid_ = 72, image_ = 1008, patch_ = 14;
    int64_t vit_heads_ = 16, txt_heads_ = 16, heads_ = 8, queries_ = 200, window_ = 24;
    // a request's state
    int64_t T_ = 0;   // the words' tokens
};

}  // namespace strata::image
