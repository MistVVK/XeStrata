// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_train.hpp - Qwen-Image-2.1's DiT trained: one block's forward with a tape and its backward
// (src/image/qi_train_block.cpp), put together out of the shared parts (train_ops.hpp's products and adapters,
// dit_rows.hpp's row kernels and their backwards, qi_attention_train and qi_attention_bwd), beside the inference side
// (src/image/qi_dit.cpp) and never in its path.
//
// The block is the inference block's math (QiDit::forward) on one sequence, the text's rows first and the image's
// after them: each row's LayerNorm (no affine, eps 1e-6) times 1 + its scale, q, k and v, each head RMS-normed with
// its weight (eps 1e-6) and rotated in pairs (2j, 2j + 1) by the row's 3-axis tables, one attention in which a text
// row sees the text up to itself and an image row every row, the output projection added through tanh of its gate;
// then the second norm and the SwiGLU MLP (gate_up's first half the gate) added through tanh of its gate.  The text's
// rows take the modulation of t = 0, the image's that of the step's t (ai-toolkit's QwenImage21Transformer2DModel with
// causal_condition).  A product's input rows are FP16 divided by a power of two and its sums FP32, as the inference
// reads them.
#pragma once

#include "strata/image/train_ops.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>

namespace strata::image {

/// The block's products in their order: to_q, to_k, to_v, to_out.0, img_mlp.gate_up, img_mlp.out.
enum QiTrainProduct : int { kQiQ = 0, kQiK, kQiV, kQiO, kQiGateUp, kQiDown, kQiProducts };

struct QiBlockRun {
    int64_t C = 0, heads = 0, hd = 0, F = 0;   // F: the MLP's inner width (gate_up is 2 F x C, out C x F)
    int64_t rows = 0, n_text = 0;               // the sequence: n_text text rows, then rows - n_text image rows
    const float* x = nullptr;                  // [rows][C] FP32, the block's input
    const float *cos = nullptr, *sin = nullptr;   // [rows][hd / 2]: every row's rotation
    /// [4][2][C]: the first norm's scale, tanh of the attention's gate, the second norm's scale, tanh of the MLP's
    /// gate; row 0 of each the image's (the step's t), row 1 the text's (t = 0)
    const float* mod = nullptr;
    train::Product p[kQiProducts];             // the frozen weights: w [N][K] FP16 and wt [K][N]; dw, db null
    const train::Adapter* ad[kQiProducts] = {};   // null: no adapter on the product
    const float *nq = nullptr, *nk = nullptr;     // [hd] FP32: attn.norm_q / norm_k
    /// An int8 base's q, k and v as FP16 rows (w8_dequant_rows) for an input that needs no gradient: the first block
    /// of a step, whose q, k and v read rows no LoRA has touched, where ai-toolkit's convrot8 leaves the input
    /// unquantized (null: the W8A8 products)
    const uint16_t* w_plain[3] = {};
    float* out = nullptr;                      // [rows][C] the block's output
    const float* dout = nullptr;               // [rows][C] its gradient (null: the forward alone)
    float* dx = nullptr;                       // [rows][C] the input's gradient, written (null: not taken)
    /// Where the forward's tape goes (null: the call's own memory): a forward alone with it keeps the tape there, and
    /// a later call with tape_replay and dout runs the backward on it without the forward again.
    train::Arena* tape = nullptr;
    bool tape_replay = false;
    /// The MLP's rows a pass (0: every row): its products, SwiGLU and their backwards a block of this many rows at a
    /// time, its widest scratch (2F a row) that many rows (taken where its adapters are plain LoRAs or none and the
    /// GPU takes the transposed products)
    int64_t mlp_rows = 0;
};

/// One block, forwards (to out) and, given dout, backwards through its own tape: the adapters' gradients added into
/// their dA and dB (train::Adapter) and the input's into dx.  A training step's checkpointed pass runs it once
/// forwards alone and again with dout, so the second forward is the first one's, bit for bit.  The tape and every
/// scratch buffer belong to the call.
/// `arena` (null: the call's own memory, freed as it returns) keeps the tape and the scratch between calls.
void qi_block(const QiBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena = nullptr);

/// The device memory a block's tape takes (QiBlockRun::tape): its shape and adapters' alone.
size_t qi_block_tape_bytes(const QiBlockRun& run);

/// XESTRATA_TRAIN_PROFILE=1: prints the blocks' stage times summed since the last report (nothing without it).
void qi_block_profile_report();

}  // namespace strata::image
