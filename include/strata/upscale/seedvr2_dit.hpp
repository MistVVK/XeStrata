// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/upscale/seedvr2_dit.hpp - SeedVR2's DiT on the GPU (ByteDance-Seed/SeedVR, Apache-2.0: its 7B
// models.dit NaDiT, written from its structure and Comfy-Org's weights' names): the velocity of its one restoration
// step, from the noise and the condition latent of an image (a clip of one frame).
//
// The 7B (7B sharp alike): 36 blocks of 3072 (24 heads of 128), video and text each with weights of their own in every
// block (MMModule, not shared).  The text is the release's positive conditioning ([58][5120], in the file) through
// txt_in; the time embedding, of the step's one timestep (1000), is a constant worked out when the file is read, and so
// is each block's modulation (the embedding's shift, scale and gate plus the block's own, AdaSingle's sum): a block
// RMS-norms (no affine, eps 1e-5) each stream, scales and shifts it, attends, adds the attention's output through its
// gate, then the same around a GELU (tanh) MLP.
//
// The attention is window attention, as the reference's NaSwinAttention: the token grid (the latent's 2 x 2 patches) is
// cut into windows scaled to a 720p grid of 3 x 3 (4 in time), the even blocks' regular and the odd blocks' shifted by
// half a window, and each window's video rows attend jointly with every text row (q and k RMS-normed a head, eps 1e-5;
// the video's q and k rotated by the window-local pixel RoPE: positions -1 .. 1 over the window's extent on each axis,
// 10 frequencies an axis, the head's first 60 channels in pairs (2j, 2j + 1)).  A text row's output is the mean of its
// windows' outputs.  The last block's text is not used after its attention, which is not computed.
//
// The 3B (dit_v2) differs: 32 blocks of 2560 (20 heads), the first 10 with video and text weights of their own and
// the rest shared by the two; a SwiGLU MLP without biases; the RoPE (mmrope3d) of 21 lang frequencies an axis over the
// first 126 channels, the video's positions the window-local frame (after the text's rows), row and column, the
// text's its row on all three axes, both rotated; and the output RMS-normed (with its weight) and modulated (the
// embedding's attn shift and scale plus vid_out_ada's) before the projection.
//
// The tokens are the VAE's (MfVae's SeedVR2 form): [h][w][64], value 4 c + 2 dy + dx of latent channel c at the
// token's position (dy, dx); the patch embedder takes a token's 2 x 2 positions of the 33 channels (the noise, the
// condition, a mask of ones) in its own order, and the output's 64 come back to the tokens' order.
//
// The blocks' weights are the file's, through a LayerStream (the first ones that fit the budget stay in VRAM, the
// others are copied while the block before them runs): FP16 (seedvr2_7b_fp16), or 8 bits (seedvr2_7b_int8_convrot:
// int8 rows with a scale each, in the basis of the regular Hadamard rotation of 256-value groups, hunyuanimage3_file.hpp)
// taken as they are where the GPU's library has int8 products (W8A8: the activations rotated alike and made int8 rows,
// dit_rows.hpp's RowsI8Of), else as Q8_0 blocks of those rows (exact: the row's scale the block's) against the
// activations rotated in FP16.
#pragma once

#include "strata/image/dit_product.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::upscale {

class Seedvr2Dit {
public:
    Seedvr2Dit();
    ~Seedvr2Dit();
    Seedvr2Dit(const Seedvr2Dit&) = delete;
    Seedvr2Dit& operator=(const Seedvr2Dit&) = delete;

    /// Reads the DiT's safetensors (Comfy-Org's names) and puts the parts outside the blocks on the GPU.  `dry`: read
    /// for its sizes only.  std::runtime_error inside: false and `err` for a file that is not SeedVR2 7B's.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// The velocity [t h w][64] (device FP32, the tokens' order) of the noise and the condition latent ([t h w][64]
    /// each, device FP32): t latent frames (a video's; 1 an image's) of h x w tokens, the windows and the RoPE over
    /// all three axes.  `block`, where given, is told each block done (its index from 1, the blocks' count) once the
    /// GPU has run it.
    void forward(const float* noise, const float* cond, int64_t h, int64_t w, float* out, int64_t t = 1,
                 const std::function<void(int, int)>& block = nullptr);

    uint64_t weight_bytes() const { return weight_bytes_; }   // the parts outside the blocks
    uint64_t block_bytes() const { return blocks_.total(); }
    uint64_t block_min() const { return blocks_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_.resident(); }
    uint64_t blocks_vram() const { return blocks_.vram(); }
    void drop_blocks() { blocks_.drop(); }
    /// The working buffers t frames of h x w tokens take, and what they take now.
    uint64_t activation_bytes(int64_t h, int64_t w, int64_t t = 1) const;
    uint64_t activations_held() const { return held_; }
    void drop_activations();

private:
    using W = image::DitW;   // type 1 F16, 8 Q8_0, -8 W8A8's int8 rows (and their scales at `sw`)
    struct Stream {   // one of a block's two streams (video, text)
        W qkv, out, in, fc;                       // proj_qkv, proj_out, mlp.proj_in, mlp.proj_out
        uint64_t b_out = 0, b_in = 0, b_fc = 0;   // their biases (FP32), offsets in the region
        uint64_t nq = 0, nk = 0;                  // the head norms' weights (FP32)
        uint64_t mod = 0;                         // [6][C] FP32: attn shift, scale, gate; mlp shift, scale, gate
    };
    struct Block {
        Stream v, t;
    };
    // a window layout's geometry (layout 0 regular, 1 shifted) for the current grid
    struct Layout {
        int64_t windows = 0, jrows = 0;   // the joint rows: each window's text and video, padded to qi_attn_rows
        std::vector<int64_t> at, len;     // a window's first joint row and its rows (text and video)
        int32_t* src = nullptr;           // [jrows]: a joint row's video token, -1 - r for text row r, kPad for none
        int32_t* starts = nullptr;        // [windows]: `at` on the GPU
        float *cos = nullptr, *sin = nullptr;   // the video tokens' RoPE [h w][hd / 2] (window-local)
    };
    struct File;
    void ensure(int64_t t, int64_t h, int64_t w);
    void layout(int which, int64_t t, int64_t h, int64_t w, std::vector<void*>& owned);
    // y [T][n] = x16 [T][k] W^T, times the rows' scales rs (null: 1), plus bias (null: none); an 8-bit weight's input
    // is the one `input` made (xq8_ and xs_in_, or rot16_)
    void product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                 const float* bias);
    // the same product as its reader takes it, the rows' scales and the bias not applied (y FP32, or W8A8's int32 sums
    // with their input's scales xs_in_ and the weights', or W8A8's values as BF16 where the GPU makes them)
    image::DitOut product_out(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y,
                              int64_t T, const float* bias);
    // the shared product's operands (image::DitProducts::run)
    image::DitProduct operands(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y,
                               int64_t T, const float* bias) const;
    float* next_xs();   // W8A8: the other of xs_ and xs2_, made xs_in_ (the next int8 rows' scales)
    // a product's input rows by the weights' form: FP16 rows (op16, into y16 with their scales rs), W8A8's int8 rows
    // (opi8), or the FP16 rows rotated for Q8_0 -> the FP16 rows' pointer (null for W8A8)
    template<class Op16, class OpI8>
    const uint16_t* input(const Op16& op16, const OpI8& opi8, int64_t T, int64_t K, uint16_t* y16, float* rs);
    image::DitOut stream_attn_in(const Stream& st, const uint8_t* base, const float* x, int64_t T, float* qkv);
    void stream_out(const Stream& st, const uint8_t* base, const uint16_t* o16, float* x, int64_t T);
    void stream_mlp(const Stream& st, const uint8_t* base, float* x, int64_t T);

    std::unique_ptr<File> file_;
    std::vector<std::vector<uint8_t>> host_;   // the blocks' FP32 vectors (biases, norms, modulations)
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_, act_;
    uint64_t weight_bytes_ = 0, held_ = 0;
    // q8_: the file's blocks in 8 bits (int8_convrot), taken as Q8_0 unless w8_; w8_: the blocks as W8A8 (the int8
    // file's rows, or the FP16 file's made so); v2_: the 3B
    bool dry_ = false, q8_ = false, w8_ = false, v2_ = false;
    int64_t scratch_ = 16;                         // the library's dequantizing scratch (values)
    int64_t C_ = 3072, heads_ = 24, hd_ = 128, F_ = 12288, L_ = 58, lat_ = 16, kin_ = 136, n_rot_ = 60;
    std::vector<float> freqs_;   // the pixel RoPE's frequencies (an axis's)
    W vin_, vout_;
    const float *b_vin_ = nullptr, *b_vout_ = nullptr;
    float* txt0_ = nullptr;      // the text through txt_in [L][C], the blocks' text stream's start
    const float* outmod_ = nullptr;   // the 3B's output norm weight, scale and shift [3][C]
    float *tcos_ = nullptr, *tsin_ = nullptr;   // the 3B's text RoPE [L][hd / 2]
    std::vector<Block> blocks_v_;
    image::LayerStream blocks_;
    // activations for a t x h x w grid
    int64_t cap_t_ = 0, cap_h_ = 0, cap_w_ = 0;
    Layout lay_[2];
    float *x_ = nullptr, *t_ = nullptr, *qkv_ = nullptr, *qkt_ = nullptr, *y_ = nullptr, *hid_ = nullptr,
          *tsum_ = nullptr, *rs_ = nullptr, *rs2_ = nullptr;
    uint16_t *a16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr, *jq_ = nullptr, *jk_ = nullptr,
             *jv_ = nullptr, *jo_ = nullptr, *o16_ = nullptr, *g16_ = nullptr, *in16_ = nullptr, *rot16_ = nullptr;
    int8_t* xq8_ = nullptr;   // W8A8: a product's int8 rows [T][K] and their scales xs_in_, xs_ or xs2_ by turns
    float *xs_ = nullptr, *xs2_ = nullptr, *xs_in_ = nullptr;   // (a product's sums read with the scales they came from)
    image::DitProducts prod_{&gemm_, &allocs_, &s_, "image engine (SeedVR2 DiT)"};
};

}  // namespace strata::upscale
