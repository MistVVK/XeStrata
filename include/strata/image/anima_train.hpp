// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/anima_train.hpp - Anima's DiT trained: one block's forward and its backward
// (src/image/anima_train_block.cpp), out of the shared parts (train_ops.hpp's products and adapters, dit_rows.hpp's
// row kernels and their backwards, qi_attention_train and qi_attention_bwd), beside the inference side
// (src/image/anima_dit.cpp) and never in its path.
//
// The block is the inference block's math (AnimaDit::forward) on one image: self-attention among the image's rows
// (q and k RMS-normed a head and rotated by the 3-D tables, pairs (j, j + hd / 2)), cross-attention to the context's
// rows (q and k RMS-normed a head, not rotated; the context's keys and values through the block's own products), and
// a GELU (erf) MLP, each on a LayerNorm (no affine, eps 1e-6) times 1 + its scale plus its shift, added back through
// its gate.  Both attentions run the shared attention (qi_attention_train and qi_attention_bwd), the cross-attention's
// keys the context's rows: its probabilities are made again from the scores and the log-sum-exp, not kept.
#pragma once
#include "strata/image/train_ops.hpp"
#include "strata/prefill/gemm.hpp"
#include <cstdint>

namespace strata::image {

/// The block's products in their order: self_attn q, k, v, output; cross_attn q, k, v, output; mlp layer1, layer2.
enum AnimaTrainProduct : int { kAnSq = 0, kAnSk, kAnSv, kAnSo, kAnCq, kAnCk, kAnCv, kAnCo, kAnL1, kAnL2, kAnProducts };

struct AnimaBlockRun {
    int64_t C = 0, heads = 0, hd = 0, F = 0, rows = 0;   // the image's rows
    int64_t L = 0, Cc = 0;                               // the context's rows and width
    const float* x = nullptr;                            // [rows][C] FP32, the block's input
    const float *cos = nullptr, *sin = nullptr;          // [rows][hd / 2]: the self-attention's rotation
    /// [3][C] each: shift, scale, gate of the self-attention, the cross-attention, the MLP
    const float *mod_sa = nullptr, *mod_ca = nullptr, *mod_ml = nullptr;
    /// the context's rows [L][Cc] FP16 (each times ctx_rs), transposed [Cc][L] (an adapter's gradient reads it)
    const uint16_t *ctx16 = nullptr, *ctx16t = nullptr;
    const float* ctx_rs = nullptr;
    const float *sqn = nullptr, *skn = nullptr, *cqn = nullptr, *ckn = nullptr;   // [hd] the q and k norms
    train::Product p[kAnProducts];                       // the frozen weights (w; wt the dgrad's)
    const train::Adapter* ad[kAnProducts] = {};          // null: no adapter on the product
    float* out = nullptr;                                // [rows][C] the block's output
    const float* dout = nullptr;                         // [rows][C] its gradient (null: the forward alone)
    float* dx = nullptr;                                 // [rows][C] the input's gradient, added into (null: not taken)
    /// [3][C] each, added into (null: not taken): the three modulations' gradients (their adapters')
    float *dmod_sa = nullptr, *dmod_ca = nullptr, *dmod_ml = nullptr;
    /// Where the forward's values the backward reads go (null: the call's scratch): a forward alone with it keeps
    /// them there, and a later call with tape_replay and dout runs the backward on them without the forward again.
    train::Arena* tape = nullptr;
    bool tape_replay = false;
};

/// One block, forwards (to out) and, given dout, forwards again and backwards: the adapters' gradients added into
/// their dA and dB, the input's into dx, the modulations' into dmod_*.
/// `arena` (null: the call's own memory) keeps the buffers from one call to the next.
void anima_block(const AnimaBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena = nullptr);
/// The device memory a block's tape takes (AnimaBlockRun::tape): its shape and adapters' alone.
size_t anima_block_tape_bytes(const AnimaBlockRun& run, prefill::Gemm& gemm);

}  // namespace strata::image
