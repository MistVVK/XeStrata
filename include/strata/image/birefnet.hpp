// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/birefnet.hpp - BiRefNet (ZhengPeng7, MIT): the background remover of the image tools
// (remove_background, /v1/images/remove-background), its parts as the model's own birefnet.py runs them (written
// again for the engine):
//   * the picture resized to 1024 x 1024 (Pillow's bilinear, as the model's handler resizes) and normalized with
//     ImageNet's mean and deviation;
//   * the Swin backbone (Swin-L in the general, portrait, matting, HR and dynamic versions, Swin-T in lite): a 4 x 4
//     patch embedding, four stages of window attention (windows of 12 (Swin-L) or 7, every other block's shifted by
//     half a window, the stage's size padded to whole windows; a relative position bias per head) with GELU MLPs and
//     patch merging between them, each stage's output normed; run on the picture and on it at half the size (bilinear,
//     corners aligned), the two concatenated per stage;
//   * the decoder: the coarsest stage with the three others resized onto it, squeezed by a decoder block; four decoder
//     blocks (a 3 x 3 convolution, an ASPP of modulated deformable convolutions of 1, 1, 3 and 7 and a global pool, a
//     3 x 3 convolution, their batch norms folded in), each fed the stage's lateral 1 x 1 convolution, the coarser
//     block's output (resized, corners aligned) gated by its gradient branch's sigmoid, and the picture's pixels
//     rearranged into the block's grid through two 3 x 3 convolutions; a 1 x 1 convolution to the mask's logit.
// The products go through prefill::Gemm, the window attention through qi_sdpa_ex (its bias and the shifted windows'
// masks), the convolutions through AuxNet::convolve (qi_conv's libraries or unfolded products), the deformable ones
// as products of their sampled patches; the rest is the engine's kernels (aux_kernels.hpp).
#pragma once

#include "strata/image/aux_net.hpp"
#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

class Birefnet : public AuxNet {
public:
    explicit Birefnet(const std::string& path);

    struct Times {
        double backbone = 0, decoder = 0, mask = 0;
    };
    /// The alpha of `image`'s foreground at its size (255 the subject, 0 the background), from the logits at
    /// sw x sh (the model's resolution, multiples of 32: 1024 x 1024; 2048 x 2048 for the HR versions, 2560 x 1440
    /// for lite-2K, the picture's own for dynamic) resized to the picture (bicubic) before their sigmoid.
    std::vector<uint8_t> run(const QiRef& image, int64_t sw = 1024, int64_t sh = 1024, Times* t = nullptr);

    /// The foreground's colors of `image` under its `alpha` (run()'s), RGB8 at its size: refine_foreground's blur
    /// fusion (aux_fg_stack), for a cut-out laid on another background.
    std::vector<uint8_t> foreground(const QiRef& image, const std::vector<uint8_t>& alpha);

    /// What a run at sw x sh needs on the GPU beyond what it holds now; with `refine`, foreground() after it too.
    uint64_t need_bytes(int64_t sw, int64_t sh, int64_t width, int64_t height, bool refine = false) const;

private:
    void load_weights() override;

    struct Block {   // a Swin block
        Norm n1, n2;
        Lin qkv, proj, fc1, fc2;
        const float* bias = nullptr;   // the relative position bias [heads][ws^2][ws^2]
        int64_t heads = 0;
        bool shift = false;
    };
    struct Stage {
        std::vector<Block> blocks;
        Norm merge_norm, out_norm;
        Lin merge;   // patch merging's reduction (4C -> 2C), none in the last stage
        int64_t dim = 0;
    };
    struct Deform {   // a modulated deformable convolution and its folded batch norm
        Conv offmask;   // the offsets' and the modulator's convolutions, stacked: 2k^2 + k^2 outputs
        Conv conv;      // the regular convolution, the norm folded in
        int k = 1;
    };
    struct DecBlk {   // BasicDecBlk with ASPPDeformable
        Conv in, out;           // 3 x 3, their norms folded
        Deform aspp[4];         // 1 x 1, then 1, 3 and 7
        Conv gap;               // 1 x 1 of the pooled pixels, its norm folded
        Conv squeeze;           // 1 x 1 of the five branches, its norm folded
    };
    struct Ipt {   // SimpleConvs: two 3 x 3 convolutions
        Conv c1, c2;
    };

    /// The Swin's four stage outputs (normed, FP32 [sh[i]][sw[i]][dim]) of img16 [SH][SW][3] in scratch slots slot0 ..
    /// slot0 + 3; sh[i] = SH / 4 >> i, sw[i] = SW / 4 >> i.
    void backbone(const uint16_t* img16, int64_t SH, int64_t SW, int slot0, float* feat[4], int64_t sh[4],
                  int64_t sw[4]);
    void dec_block(const DecBlk& b, const uint16_t* x16, int64_t H, int64_t W, float* y);
    void deform(const Deform& d, const uint16_t* x16, int64_t H, int64_t W, float* y, int64_t ldy);
    void ipt(const Ipt& p, const float* img, int64_t SH, int64_t SW, int64_t sh, int64_t sw, float* y, int64_t ldy,
             int64_t c0);
    Conv conv_bn(const std::string& conv, const std::string& bn, int stride, int pad);

    Conv patch_;
    Norm patch_norm_;
    std::vector<Stage> stages_;
    int64_t window_ = 12, embed_ = 192;
    DecBlk squeeze_, dec_[4];
    Conv lateral_[3];
    Conv gdt_[3], gdt_attn_[3];
    Ipt ipt_[5];
    Conv out_;
    // the shifted windows' masks, by the padded stage's side
    std::vector<std::pair<int64_t, const float*>> masks_;
    const float* shift_mask(int64_t Hp, int64_t Wp);
};

}  // namespace strata::image
