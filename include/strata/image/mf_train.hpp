// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_train.hpp - Mage-Flow's DiT trained: its own part of a LoRA, a LoKr or a DoRA learning step
// on the GPU (one block, forward with a tape and backward; its latent's packing), put together out of the shared parts
// (train_ops.hpp's products, adapters and loss, dit_rows.hpp's row kernels and their backwards, kernels::qk_norm_rope
// and its backward, qi_attention_train and qi_attention_bwd), beside the inference side (src/image/mf_dit.cpp) and
// never in its path.
//
// A training step runs the inference block's own math with a tape: every value a gradient is read from is kept (the
// LayerNorm's mean and inverse, the products' FP16 input rows and their power-of-two scales, the attention's softmax
// statistics, the GELU's input).  A product is prefill::Gemm's, with the same FP16 rows and FP32 sums the inference
// takes, so a step starts from the values the engine already reads.
//
// Checked per op against NumPy and one block against torch (tools/image/mf_train_ref.py,
// tools/image/mf_train_torch_ref.py).
#pragma once

#include "strata/image/train_ops.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>

namespace strata::image {

/// The 2 x 2 packing of a latent to tokens with the VAE's batch norm (a token's 128 values are four positions' 32
/// channels at k = 4 c + 2 (Y % 2) + X % 2) and z = (m - mean) / sd, backwards: dm is the scatter of dz / sd, and
/// dmean and dsd the tokens' sums added into (null: not taken).  m is [2 h][2 w][64]'s first 32 channels, dz is
/// [h w][128].
void mf_patch_bwd(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, const float* dz, float* dm,
                  float* dmean, float* dsd, void* stream);

struct MfBlockW {
    int64_t N[6] = {}, K[6] = {};        // q, k, v and o are C x C, mlp1 is F x C, mlp2 is C x F (set: the adapter
                                         // path reads them, and zero widths enqueue no work and fail)
    const uint16_t* w[6] = {};           // [N][K] FP16, as the file writes them
    const uint16_t* wt[6] = {};          // [K][N] FP16, the same values transposed (dgrad reads them)
    const uint16_t* wr[6] = {};          // [N][K] FP16 the dgrad reads as they are where wt is null (Product::wr)
    const int8_t* w8[6] = {};            // a QLoRA's int8 base instead of w (train::Product's w8, sw; null: w)
    const float* sw[6] = {};
    const float* b[6] = {};              // [N] FP32
    const train::Adapter* ad[6] = {};         // null: no adapter on the product
    const float *nq = nullptr, *nk = nullptr;      // [head_dim] FP32 each: the file's attn.norm_q / norm_k weight
    const float* mod = nullptr;          // [6 C]: shift, scale, gate of the attention, then of the MLP
    float* dw[6] = {};                   // [N][K] FP32, added into
    float* db[6] = {};                   // [N]
    float *dnq = nullptr, *dnk = nullptr;   // [head_dim] each, added into
    float* dmod = nullptr;               // [6 C] the modulation's gradient, added into (null: not taken)
};

struct MfBlockRun {
    int64_t C = 0, heads = 0, hd = 0, F = 0, rows_img = 0, rows_txt = 0;
    const float *img = nullptr, *txt = nullptr;    // [rows][C] FP32
    const float *cos = nullptr, *sin = nullptr;    // [rows_img][hd / 2] (null: no rotation)
    MfBlockW img_w, txt_w;
    float *out_img = nullptr, *out_txt = nullptr;         // the forward's outputs
    const float *d_out_img = nullptr, *d_out_txt = nullptr;   // the gradients reaching them
    float *dimg = nullptr, *dtxt = nullptr;         // the block's inputs' gradients, added into
    float* tape = nullptr;    // the forward's stages out to the parity harness (null: not taken, see MfTapeSlice)
    /// Where the forward's values the backward reads go (null: the call's scratch): a forward alone with it keeps
    /// them there, and a later call with kept_replay and the outputs' gradients runs the backward on them without the
    /// forward again.
    train::Arena* kept = nullptr;
    bool kept_replay = false;
    bool lazy_dgrad = false;  // make one product's dgrad weight at a time, in the block scratch
};

/// The block's forward tape, widened to FP32 and copied out for tools/image/mf_train_torch_ref.py to compare stage by
/// stage: `MfBlockRun::tape` (null: not taken) is a device array of `mf_block_tape_elems`'s count, laid out as the
/// slices below per stream (the image's rows first, then the text's) and then the joint attention's.  A product's FP16
/// input rows land here widened but NOT multiplied back by their row's scale, so a reference that rounds the same way
/// reads the same numbers.  Every copy is off the inference path: a real training step passes null and pays nothing.
enum MfTapeSlice : int {
    kTapeLn1 = 0,   // [rows][C]   the first norm's rows, as the q, k and v products read them
    kTapeQ,         // [rows][C]
    kTapeK,         // [rows][C]
    kTapeV,         // [rows][C]
    kTapeO,         // [rows][C]   the output projection
    kTapeH1,        // [rows][C]   the attention's gated residual, the second norm's input
    kTapeLn2,       // [rows][C]   the second norm's rows, as mlp1 reads them
    kTapeM1,        // [rows][F]
    kTapeGelu,      // [rows][F]   mlp2's input rows
    kTapeM2,        // [rows][C]
    kTapeOut,       // [rows][C]   the block's output
    // then the same tensors' gradients, in the order the backward reaches them (what each one's producer's output
    // gradient is, before it moves on): the backward is the longer half to get wrong and the harder to see
    kTapeDLn1,      // [rows][C]   into the first norm's FP32 output, the three products' dgrads summed
    kTapeDQ,        // [rows][C]
    kTapeDK,        // [rows][C]
    kTapeDV,        // [rows][C]
    kTapeDYo,       // [rows][C]   into the output projection
    kTapeDH1,       // [rows][C]   into the first residual's sum
    kTapeDLn2,      // [rows][C]   into the second norm's FP32 output
    kTapeDM1,       // [rows][F]   into mlp1
    kTapeDGelu,     // [rows][F]   into the GELU's output, which is mlp2's dgrad
    kTapeDM2,       // [rows][C]   into mlp2
    kTapeStreamN,   // slices a stream: the tape holds the image's rows first, then the text's
};
/// the joint rows, after both streams: the attention's own inputs and output, its statistics, and their gradients
enum MfTapeJoint : int { kTapeJQ = 0, kTapeJK, kTapeJV, kTapeJO, kTapeJDO, kTapeJDQ, kTapeJDK, kTapeJDV,
                         kTapeLse,   // the last slot, and the only one `heads` wide instead of `C`
                         kTapeJointN };
/// after the joint block, the rows' power-of-two scales, the image's rows then the text's, one pair per stored row set:
/// they are the other half of the contract a product reads (its rows and the scale that puts them back), and a scale
/// that came out zero turns every product into its own bias - the shape of the failure a dump of the products cannot
/// tell apart from a zero weight.
enum MfTapeScale : int { kTapeScLn1I = 0, kTapeScLn1T, kTapeScLn2I, kTapeScLn2T, kTapeScGeluI, kTapeScGeluT,
                         kTapeScaleN };

int64_t mf_block_tape_elems(const MfBlockRun& run);

/// One Mage-Flow block, run forwards with a tape and backwards through it, out of the shared parts: what
/// tools/image/mf_train_torch_ref.py is checked against.  The block is the inference block's own math
/// (src/image/mf_dit.cpp): each stream's LayerNorm (no affine, eps 1e-6) with the time's shift and scale, q, k and v,
/// each head RMS-normed with its weight and the image's rows rotated pairwise, one attention over [text, image] where
/// every row sees every key, the output projection and the gated residual, then the second norm and the tanh-GELU MLP
/// with its gated residual.  A product's input rows are FP16 divided by a power of two and its sums are FP32, exactly
/// as the inference reads them, so an FP32 torch reference meets these gradients at FP16's rounding.
///
/// The time's modulation is a value, not a trained tensor: v1 leaves the modulations out of the target modules
/// (docs/LORA_TRAIN.md, "The four models"), so nothing goes back into shift, scale or gate's own weights.  An adapter
/// (MfBlockW::ad, null: none) adds its delta to the product's rows in the forward and its gradients in the backward;
/// a null dw/db skips the base weight's own wgrad and bias sum (a frozen weight takes no gradient, and a LoKr-folded
/// one reads only dDelta through dw - the block knows no LoKr).
///
/// The forward and backward in one call, or the backward alone on a kept forward (MfBlockRun::kept).
/// `arena` (null: the call's own memory) keeps the buffers from one call to the next.
void mf_block(const MfBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena = nullptr);
/// The device memory a block's kept forward takes (MfBlockRun::kept): its shape and adapters' alone.
size_t mf_block_tape_bytes(const MfBlockRun& run, prefill::Gemm& gemm);

}  // namespace strata::image
