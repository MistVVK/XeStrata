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
#include "strata/image/img_crop.hpp"
#include "strata/image/pipeline.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::image {

struct Segment {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // the box on the picture (pixels, x1 and y1 past the last)
    float score = 0;
    std::vector<uint8_t> mask;              // width x height, 255 inside
};

class Sam3Video;

class Sam3 : public AuxNet {
    friend class Sam3Video;   // a clip's run: the tracker's parts and the frame's features in its scratch

public:
    explicit Sam3(const std::string& path);

    /// The last words live in scratch; keep the host cache but invalidate the GPU shortcut.
    void drop_scratch() override;

    struct Times {
        double vision = 0, text = 0, detect = 0, masks = 0;
    };
    /// The instances of the words `ids` (CLIP's tokens, <|startoftext|> .. <|endoftext|>, at most 32) in `image`
    /// whose score passes `score_threshold`, highest first, each with its mask (`mask_threshold` on the
    /// probability) at the picture's size.
    std::vector<Segment> run(const QiRef& image, const std::vector<int32_t>& ids, float score_threshold,
                             float mask_threshold, Times* t = nullptr);

    /// run() of each of several words on one picture: the trunk made once, each words' detector on it in turn (as a
    /// clip's frame takes its prompts) -> the instances of each, in the words' order.
    std::vector<std::vector<Segment>> run_many(const QiRef& image, const std::vector<std::vector<int32_t>>& ids,
                                               float score_threshold, float mask_threshold);

    /// What a run needs on the GPU beyond what it holds now (the weights when not loaded, the scratch not grown).
    uint64_t need_bytes(int64_t width, int64_t height) const;

    /// run()'s parts, for a clip's run (sam3_video.cpp): the words' states (text; kept while the words are the same),
    /// the detector on the trunk's frame (detect_neck, then detect: every query's probability, its box cx, cy, w, h
    /// on [0, 1], the queries past `score_threshold` best first), then the masks' features (mask_prep) and the
    /// queries' mask logits at the 4x level (mask_logits: logits [k][16 grid^2], FP32 on the GPU).
    struct Detection {
        std::vector<int64_t> keep;
        std::vector<float> prob, box;
    };
    void text(const std::vector<int32_t>& ids);
    /// keep: the 1x level kept, for detect_restore to put back before the next words' detect (which runs its
    /// encoder in place there).
    void detect_neck(bool keep = false);
    void detect_restore();
    Detection detect(float score_threshold);
    void mask_prep();
    void mask_logits(const int64_t* qs, int64_t k, float* logits);
    /// A detection's instances (their boxes and scores, and their masks at W x H: mask_prep and mask_logits).
    std::vector<Segment> segments(const Detection& det, int64_t W, int64_t H, float mask_threshold);

    // ---------------------------------------------------------------------------------------------- the video
    // SAM 3.1's multiplex tracker (sam3_track.cpp): its parts, which a clip's run (sam3_video.cpp) puts together.  A
    // frame's trunk is the detector's; the tracker reads it through necks of its own.  Up to M objects share a
    // bucket (their masks, scores and pointers decoded at once); a clip of more has more buckets, B of them.
    static constexpr int64_t kMux = 16;   // M: the objects a bucket holds (the checkpoint's multiplex count)

    /// The tracker's weights onto the GPU (the detector's with them).
    void track_load();
    /// What a clip's run needs on the GPU beyond what it holds now: need_bytes' and the tracker's, `buckets` buckets.
    uint64_t video_need_bytes(int64_t width, int64_t height, int64_t buckets) const;
    /// A frame's trunk (the ViT's patches, kept for its necks) from a picture, resized and normalized as run()'s; or
    /// from pixels already [3][1008][1008] on the host, taken as they are (the probe's).
    void trunk(const QiRef& image, Filter filter = Filter::triangle);
    void trunk_pixels(const float* chw);
    /// The trunk's propagation features (the neck's 4x, 2x and 1x levels, d channels, [H][W][d]), kept for the frame;
    /// their device pointers by level.
    void track_features();
    const float* track_level(int i) const;
    /// The memory attention of B buckets: x [P][d] (the frame's 1x level) + 0.1 pos [P][d], attending to mem16
    /// [B][N][2d] (a row: the frame's image features | its memory; an object pointer's: zeros | the pointer), its
    /// first nsp rows the frames' (P a frame) at sine [P][d] + tpos [frame][d] (tpos null: sine is [nsp][d]), the
    /// rest pointers at ptr_pos [(r - nsp) / per][d] -> out [B][P][d].
    void mem_attention(const float* x, const float* pos, const uint16_t* mem16, int64_t B, int64_t N, int64_t nsp,
                       const float* sine, const float* tpos, const float* ptr_pos, int64_t per, float* out);
    /// The multiplex decoder over B buckets: img [B][P][d] (the frame's features after the memory attention), the
    /// 4x and 2x levels hr0, hr1, each slot's validity valid [B M] -> masks [B][M T][P0] (logits at 4x), iou
    /// [B][M][T], scores [B][M], the mask tokens [B][M][T][d] (device scratch, valid until the next call).
    struct MuxOut {
        float *masks = nullptr, *iou = nullptr, *score = nullptr, *tokens = nullptr;
        int64_t T = 0, P0 = 0;
    };
    MuxOut mux_decode(const float* img, const float* hr0, const float* hr1, const std::vector<uint8_t>& valid,
                      int64_t B);
    /// The object pointers of tokens [n][d] (the propagation's projection, or the interactive one's), each taken to
    /// the no-object pointer where its score [n] (host) is not above 0 -> ptr [n][d].
    void pointers(const float* tokens, int64_t n, bool interactive, const std::vector<float>& score, float* ptr);
    /// The memory encoder of B buckets: pix [P][d] (the frame's 1x level), masks [N][h][w] (logits; sigmoid_first:
    /// the cond frame's own masks, resized after their sigmoid, else before), cond [N] (1: a clean mask), is_obj [N]
    /// (host) -> mem [B][P][d] (FP32) and the same as FP16 into mem16 rows (its stride ld, at column `col`).
    void mem_encode(const float* pix, const float* masks, int64_t N, int64_t h, int64_t w, bool sigmoid_first,
                    const std::vector<float>& cond, const std::vector<float>& is_obj, int64_t B, float* mem,
                    uint16_t* mem16, int64_t ld, int64_t col, const std::vector<float>* valid = nullptr);
    /// The object pointers' time positions: the pointer projection of the 1D sine of t / (max_abs - 1), each t ->
    /// out [n][d].
    void ptr_positions(const std::vector<int64_t>& t, int64_t max_abs, float* out);
    /// The memories' time positions [mem_frames()][d] (the device's), the memory frames a frame attends to at most.
    const float* mem_tpos() const { return mem_tpos_; }
    int64_t mem_frames() const { return mem_frames_; }
    /// The sine positions of the 1x level [grid^2][d] (the device's), the memories' and the frame's.
    const float* sine_positions() const { return sine_pos_; }
    /// The memory encoder of a bucket from its input already made: mux16 [1152][1152][2M] (the masks' 2 sigmoid - 1,
    /// then the cond channels), each slot's being an object [M] (the probe's) -> mem [P][d].
    void mem_encode_muxed(const float* pix, const uint16_t* mux16, const std::vector<float>& slot_is_obj, float* mem);
    /// The interactive decoder on masks [n][S][S] (0 or 1, S the image size): the frame's interactive neck's 1x level
    /// with the no-memory embedding, each mask through the mask downsample and the prompt encoder -> their tokens [n]
    /// [d] (the first mask token's) and scores [n] (host).
    void interactive(const float* masks, int64_t n, float* tokens, std::vector<float>& score);
    /// The interactive decoder on clicks: pts [n] (x, y on the 1008 picture's [0, 1]; label 0 out, 1 in, 2 and 3 a
    /// box's top left and bottom right), on the frame's interactive features (no memory) with the object's mask
    /// before (prev: its logits [4G][4G] on the GPU) or none -> its mask logits [16 grid^2] (host; none where its
    /// score is not above 0), its token [d] (the GPU's; token null: none) and score (the logit).
    float point_prompt(const std::vector<std::array<float, 3>>& pts, const float* prev, std::vector<float>& low,
                       float* token);

private:
    Times last_;   // the last run's parts: the shares progress weighs them by
    void load_weights() override;
    void vision(const QiRef& image);
    void trunk_from(const uint16_t* pix16);
    std::vector<int32_t> text_ids_;   // the words whose states text() holds
    // the states of the words text() made (FP32 then FP16 [T][d], the host's): a clip's several words each frame
    static constexpr size_t kTextCache = 32;
    std::map<std::vector<int32_t>, std::vector<uint8_t>> text_cache_;
    void attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int64_t lq, int64_t lk, int64_t heads,
                   int64_t hd, int64_t q_row, int64_t k_row, int64_t v_row, uint16_t* out, int64_t batch = 1,
                   const float* bias = nullptr, bool causal = false);
    void grid_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t lq, const float* grid,
                        uint16_t* out);

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
    /// The neck `set` on the trunk into out[3] (4x, 2x, 1x; [H][W][d] FP32).
    void neck(const std::vector<Fpn>& set, float* const out[3]);
    std::vector<Fpn> neck_of(const std::string& prefix);

    // the tracker's parts
    struct Mlp {   // ReLU between its layers
        std::vector<Lin> l;
    };
    struct TwoWayLayer {
        Lin sq, sk, sv, so, tq, tk, tv, to, iq, ik, iv, io, fc1, fc2;
        Norm n1, n2, n3, n4;
    };
    struct MaskDecoder {
        std::vector<TwoWayLayer> layers;
        Lin fq, fk, fv, fo;
        Norm nf;
        std::vector<float> obj, iou, mask;   // the tokens (host), rows of d
        Lin up1, up2, s0, s1;
        Norm ln1;
        std::vector<Mlp> hyper;
        Mlp iou_head, score_head;
        int64_t M = 1, T = 4;   // the objects at once, the masks an object
    };
    struct MemLayer {
        Lin qkv, so, cq_img, cq, ck, cv, co, fc1, fc2;
        Norm n1, n2, n3;
    };
    struct CxBlock {
        const float *dw = nullptr, *dwb = nullptr, *gamma = nullptr;
        Norm n;
        Lin pw1, pw2;
    };
    MaskDecoder decoder(const std::string& p, int64_t M, int64_t T);
    Mlp mlp(const std::string& p);
    void mlp_run(const Mlp& m, const uint16_t* x16, int64_t rows, float* y);
    /// The two-way transformer of a decoder over B batches: queries q [B][Nt][d] (tokens qpe [B][Nt][d] their
    /// positions), keys k [B][P][d] at kpe [P][d]; both updated in place.
    void two_way(const MaskDecoder& m, int64_t B, int64_t Nt, float* q, const float* qpe, float* k, const float* kpe);
    /// A decoder's heads on one batch (sam3_track.cpp), and the prompt encoder's dense embedding of a mask's logits.
    void mask_heads(const MaskDecoder& m, const float* qb, int64_t Nt, const float* kb, const float* s0,
                    const float* s1, float* masks, float* iou, float* score);
    void mask_dense(const float* low, float* out);
    float* track_scratch(int slot, size_t bytes);
    void encode_bucket(const float* pix, const uint16_t* mux16, const std::vector<float>& slot_is_obj, float* mem,
                       uint16_t* mem16, int64_t ld, int64_t col, const Conv* first = nullptr);
    /// The memory encoder's first convolution over the live slots' channels alone (their masks, then their conds):
    /// its weights' columns for them, made once a set of slots.
    const Conv& mds_first(const std::vector<int32_t>& live);
    int64_t P_() const { return grid_ * grid_; }

    std::vector<Fpn> prop_, inter_;
    MaskDecoder mdec_, idec_;
    std::vector<MemLayer> mem_;
    Norm mem_norm_;
    std::vector<Conv> mds_;
    std::vector<float> mds0_w_;                          // the first convolution's weights [o][2M][k][k] (host)
    std::map<std::vector<int32_t>, Conv> mds0_live_;     // mds_first's
    std::vector<Norm> mds_ln_;
    Lin mds_out_, pix_proj_;
    std::vector<CxBlock> fuser_;
    Mlp ptr_proj_, iptr_proj_;
    Lin no_obj_ptr_, tpos_proj_;
    const float* mem_tpos_ = nullptr;
    int64_t mem_frames_ = 7;
    Conv imds_, pe_c0_, pe_c1_;
    Norm pe_ln0_, pe_ln1_;
    Lin pe_c2_;
    std::vector<float> no_mask_, prompt_gauss_;
    std::array<std::vector<float>, 4> point_emb_;   // the clicks' labels: out, in, a box's two corners
    std::vector<float> not_point_, no_mem_, valid_emb_, invalid_emb_, no_obj_spatial_;
    const float* dense_pe_ = nullptr;    // the multiplex decoder's image positions [P][d]
    const float* prompt_pe_ = nullptr;   // the prompt encoder's [P][d]
    bool track_loaded_ = false;

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
