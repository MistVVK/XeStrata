// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/florence2.hpp - Florence-2 (Microsoft, MIT): the words-to-mask model of the image tools
// (edit_region), its parts as transformers' Florence2ForConditionalGeneration runs them: the picture resized to 768 x
// 768 (Pillow's bicubic) and normalized, DaViT's four stages (a convolution's embedding, then blocks of a depthwise
// position convolution, window attention, a channel attention and GELU MLPs), the 2D and the temporal position
// embeddings, the pooled token and the projection; BART's encoder over the picture's tokens and the task's words, and
// its decoder in a beam search (beams, no repeated 3-grams, the first token <s>, the last </s>, length penalty 1 and
// stopping when the beams are done, the generation config's) with a cache of the keys and values.  The tokens out are
// the server's to read (serve/image_regions.py: <loc_*> into polygons); aux_polygon_fill makes a mask of the polygons.
//
// The weights are Microsoft's model.safetensors (FP16), downloaded by the server when first needed; they stay on the
// GPU until release().
#pragma once

#include "strata/image/aux_net.hpp"
#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::image {

class Florence2 : public AuxNet {
public:
    /// Opens the weights (the file's header, checked against the model's shapes); nothing on the GPU before run().
    explicit Florence2(const std::string& path);

    struct Times {
        double vision = 0, encoder = 0, decoder = 0;
        int steps = 0;
    };
    /// The tokens the decoder chose for `image` and the task's words `text_ids` (<s> ... </s>, the server's
    /// tokenizer's): the decoder's start (</s>) first, as transformers' generate() returns them.
    std::vector<int32_t> run(const QiRef& image, const std::vector<int32_t>& text_ids, int max_new = 1024,
                             int beams = 3, Times* t = nullptr);

    // dump (AuxNet's): vision_out [576][1024], image_features [577][768], encoder [T][768], logits0 [1][vocab] of the
    // start token alone

    /// Fills `mask` (RGBA8 W x H, alpha 0 where a polygon is, 255 elsewhere: the server's edit mask) from polygons
    /// (x, y pairs in the picture's pixels), on the GPU.
    static QiRef polygon_mask(const std::vector<std::vector<float>>& polygons, int64_t W, int64_t H);

    /// What a run needs on top of the weights for `beams` beams.
    uint64_t work_bytes(int beams, int max_new) const;
    /// What a run needs on the GPU beyond what it holds now: the weights (from the file's size) when they are not
    /// loaded, and the buffers it has not grown yet.
    uint64_t need_bytes(int beams, int max_new) const;

private:
    struct Dw {
        const float *w = nullptr, *b = nullptr;   // [c][9], [c]
    };
    struct VBlock {   // a spatial or a channel block
        Dw conv1, conv2;
        Norm n1, n2;
        Lin qkv, proj, fc1, fc2;
        Lin q, k, v;   // qkv's thirds, views of its rows (the spatial blocks' attention reads each on its own)
    };
    struct Stage {
        Conv embed;
        Norm embed_norm;
        bool pre_norm = false;
        int64_t dim = 0, heads = 0, groups = 0;
        std::vector<VBlock> spatial, channel;
    };
    struct ELayer {
        Lin qkv, o, fc1, fc2;
        Norm n_attn, n_ffn;
    };
    struct DLayer {
        Lin qkv, o, cq, ckv, co, fc1, fc2;
        Norm n_self, n_cross, n_ffn;
    };

    void load_weights() override;
    Dw dw(const std::string& name);
    void vision(const QiRef& image);
    void step(int64_t B, int64_t T, int64_t cap, uint16_t* cache, const uint16_t* ckv, const int32_t* dids, float* h,
              float* y, uint16_t* h16, uint16_t* g16, uint16_t* a16, uint16_t* q16, float* logits);
    void encoder(const std::vector<int32_t>& text_ids);
    /// `each`, where given, is told the steps done after each.
    std::vector<int32_t> decode(int max_new, int beams, int& steps, const std::function<void(int)>& each = nullptr);

    // the model's shape (config.json's, checked against the file)
    int64_t d_ = 768, heads_ = 12, ffn_ = 3072, vocab_ = 51289, max_pos_ = 1024, image_dim_ = 1024;
    int64_t window_ = 12, image_size_ = 768;
    std::vector<Stage> stages_;
    std::vector<float> pos_table_;        // the 2D positions plus the first frame's cosine, [24 x 24][image_dim]
    const float* pos_dev_ = nullptr;      // the same on the GPU
    Lin proj_;                            // image_projection, transposed to [768][1024]
    Norm proj_norm_;
    const uint16_t* shared_ = nullptr;    // [vocab][d] FP16: the embeddings and the LM head
    const float* logits_bias_ = nullptr;
    const uint16_t* enc_pos_ = nullptr;
    const uint16_t* dec_pos_ = nullptr;
    Norm enc_ln_, dec_ln_;
    std::vector<ELayer> enc_;
    std::vector<DLayer> dec_;
    // the run's state
    float* feats_ = nullptr;   // [577][d]: the picture's tokens, then the encoder's states [T][d]
    int64_t img_tokens_ = 0, enc_T_ = 0;
    Times last_;               // the last run's parts and steps: the shares progress weighs them by
};

}  // namespace strata::image
