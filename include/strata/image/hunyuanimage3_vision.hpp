// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_vision.hpp - HunyuanImage 3.0's reading of a reference image: SigLIP2 (the
// model's `vision_model`, so400m, patch 16, naflex) and the `vision_aligner` that puts its features into the
// decoder's width, on the GPU.
//
// An edit's reference image reaches the model twice: as its VAE latent (hunyuanimage3_vae's encode) and as SigLIP2's
// features, which this part makes.  The processor resizes the image to at most 1024 patches of 16 x 16 pixels
// (hunyuanimage3_refs), pads the patches to 1024 rows, and the ViT runs over all 1024 - the padded rows attend only
// to the real keys (the pixel mask), but every row comes out, and the reference scatters all 1024 into the
// sequence's vit slots (its `instantiate_vit_image_tokens`), so this does too.
//
//   patches [1024][768] -> Linear(768 -> 1152) + the 16 x 16 positional grid resized to the real grid (bilinear,
//   antialiased; the padded rows get the grid's first entry) -> 27 x (LayerNorm, 16-head attention of head_dim 72,
//   residual; LayerNorm, MLP 1152 -> 4304 -> 1152 with GELU's tanh form, residual) -> post LayerNorm -> the aligner:
//   Linear(1152 -> 4096), GELU (erf), Linear(4096 -> 4096) -> rows [1024][4096]
//
// The attention is the shared one qi_vision's reader takes (qi_attention_cross: the heads padded to 128, the GPU
// maker's kernel and the engine's own timed once a GPU and a shape), q scaled by sqrt(128 / head_dim) so the
// shared scale 1 / sqrt(128) is the model's 1 / sqrt(head_dim).  The products are prefill::Gemm's FP16 ones, the
// norms, the softmax and the residual stream FP32 (the reference runs it in BF16; `probe refs` holds this to the
// golden's FP32 features).  The weights come from the model's own bf16 shard (the `vision_model.*` tensors) and the
// quantized file (`vision_aligner.*`, BF16), as FP16 on the device: 0.86 GB.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

class HunyuanImage3Vision {
public:
    HunyuanImage3Vision();
    ~HunyuanImage3Vision();
    HunyuanImage3Vision(const HunyuanImage3Vision&) = delete;
    HunyuanImage3Vision& operator=(const HunyuanImage3Vision&) = delete;

    /// `vision`: a safetensors file with the model's `vision_model.*` tensors (its bf16 shard, or a file of those
    /// tensors alone); `dit`: the quantized file, for `vision_aligner.*`.  `dry`: the sizes only, nothing on the GPU.
    bool open(const std::string& vision, const std::string& dit, void* stream, std::string& err, bool dry = false);

    int64_t patch() const { return kPatch; }
    int64_t max_patches() const { return kMaxPatches; }
    /// The aligner's output width (the decoder's hidden, 4096).
    int64_t out_width() const { return out_; }
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// What the part holds on the GPU now: the weights and, after a run, its buffers.
    uint64_t held() const { return weight_bytes_ + held_bytes_; }
    /// The buffers a run takes (device), for the plan.
    uint64_t scratch_bytes() const;
    void drop_scratch();

    /// The 1024 rows of a reference whose patches (hunyuanimage3_vit_patches: [max_patches][768], the real `ph` x
    /// `pw` first, zeros after) are `patches`: `rows` the host buffer of max_patches * out_width() floats.  False
    /// with `err`.
    bool rows(const float* patches, int64_t ph, int64_t pw, float* rows, std::string& err);

    /// The last run's post-LayerNorm features ([max_patches][hidden], host FP32), for the probe's stage check
    /// against the golden's `cond_vit_hidden`.
    const std::vector<float>& last_hidden() const { return last_hidden_; }

private:
    struct Lin {
        const uint16_t* w = nullptr;   // [out][in] FP16
        const float* b = nullptr;      // [out] FP32
        int64_t in = 0, out = 0;
    };
    struct Norm {
        const float* w = nullptr;
        const float* b = nullptr;
    };
    struct Layer {
        Norm ln1, ln2;
        Lin q, k, v, o, fc1, fc2;
    };
    Lin lin(class SafeTensors& st, const std::string& name, std::string& err);
    Norm norm(class SafeTensors& st, const std::string& name, std::string& err);
    const float* vec(class SafeTensors& st, const std::string& name, std::string& err);
    void grow();
    void attention(const Layer& l, int64_t n);

    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> weights_, scratch_;
    uint64_t weight_bytes_ = 0, held_bytes_ = 0;
    bool dry_ = false;
    int64_t hidden_ = 0, inter_ = 0, out_ = 0, n_layers_ = 0;
    Lin patch_;
    std::vector<float> pos_;   // the 16 x 16 positional grid, host FP32 [256][hidden]
    int64_t pos_side_ = 0;
    std::vector<Layer> layers_;
    Norm post_;
    Lin al0_, al2_;
    // the buffers
    uint16_t *patches16_ = nullptr, *a16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr,
             *out16_ = nullptr, *mlp16_ = nullptr, *al16_ = nullptr;
    float *x32_ = nullptr, *qkv32_ = nullptr, *tmp32_ = nullptr, *mlp32_ = nullptr, *al32_ = nullptr,
          *out32_ = nullptr, *pos32_ = nullptr;
    int64_t pos_ph_ = 0, pos_pw_ = 0;   // the grid pos32_ holds
    std::vector<float> last_hidden_;

    static constexpr int64_t kPatch = 16, kMaxPatches = 1024;   // the model's vit_processor (config.json)
    static constexpr int64_t kHeads = 16;                       // its vit.num_attention_heads
    static constexpr int64_t kHeadPad = 128;                    // the shared attention's head_dim (the head's 72
                                                                // channels, the rest zero); kMaxPatches key rows
                                                                // cover any n rounded up to a multiple of 64
    static constexpr float kEps = 1e-6f;                        // its layer_norm_eps
};

}  // namespace strata::image
