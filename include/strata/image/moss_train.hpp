// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_train.hpp - MOSS-Image's DiT trained: one block's forward and its backward
// (src/image/moss_train_block.cpp), put together out of the shared parts (train_ops.hpp's products and adapters,
// dit_rows.hpp's row kernels and their backwards, qi_attention_train and qi_attention_bwd), beside the inference side
// (src/image/moss_dit.cpp) and never in its path.
//
// The block is the inference block's math (MossDit::run_block) on one sequence, the image's rows first and the text's
// after them, every row taking the step's modulation: each row RMS-normed with its weight (eps 1e-6), times 1 + its
// scale plus its shift; one qkv product, q and k RMS-normed a head with their weights and rotated in pairs
// (j, j + hd / 2) by the row's 4-axis tables, one attention over every row (no mask), the output projection added
// through its gate; then the second norm and the MLP (ffn_up_gate's first half the hidden values, the second the
// gates, SiTU between) added through its gate.  A product's input rows are FP16 divided by a power of two and its sums
// FP32, as the inference reads them; the residual rows are FP32 (the inference keeps them in bf16).
#pragma once
#include "strata/image/train_ops.hpp"
#include "strata/prefill/gemm.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::image {

/// The block's products in their order: proj.qkv (3 C x C, rows q, k, v), proj.o, proj.ffn_up_gate (2 F x C, the
/// hidden values' rows then the gates'), proj.ffn_down.
enum MossTrainProduct : int { kMossQkv = 0, kMossO, kMossUpGate, kMossDown, kMossProducts };

struct MossBlockRun {
    int64_t C = 0, heads = 0, hd = 0, F = 0, rows = 0;
    const float* x = nullptr;                      // [rows][C] FP32, the block's input
    const float *cos = nullptr, *sin = nullptr;    // [rows][hd / 2]: every row's rotation
    const float* mod = nullptr;                    // [6][C]: shift, scale, gate of the attention, then of the MLP
    const float *n1 = nullptr, *n2 = nullptr;      // [C] the two norms' weights
    const float *nq = nullptr, *nk = nullptr;      // [hd] q's and k's
    train::Product p[kMossProducts];               // the frozen weights (w, or an int8 base's w8 and sw; wt the dgrad's)
    const train::Adapter* ad[kMossProducts] = {};  // null: no adapter on the product
    /// An int8 base's qkv as FP16 rows (w8_dequant_rows) for an input that needs no gradient: the first block, whose
    /// qkv reads rows no LoRA has touched, where ai-toolkit's convrot8 leaves the input unquantized (null: W8A8)
    const uint16_t* w_plain_qkv = nullptr;
    float* out = nullptr;                          // [rows][C] the block's output
    const float* dout = nullptr;                   // [rows][C] its gradient (null: the forward alone)
    float* dx = nullptr;                           // [rows][C] the input's gradient, added into (null: not taken)
    /// Where the forward's values the backward reads go (null: the call's scratch): a forward alone with it keeps
    /// them there, and a later call with tape_replay and dout runs the backward on them without the forward again.
    train::Arena* tape = nullptr;
    bool tape_replay = false;
};

/// One block, forwards (to out) and, given dout, forwards again and backwards: the adapters' gradients added into
/// their dA and dB (train::Adapter) and the input's into dx.
/// `arena` (null: the call's own memory) keeps the buffers from one call to the next.
void moss_block(const MossBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena = nullptr);
/// The device memory a block's tape takes (MossBlockRun::tape): its shape and adapters' alone.
size_t moss_block_tape_bytes(const MossBlockRun& run, prefill::Gemm& gemm);
/// XESTRATA_TRAIN_PROFILE=1: prints the blocks' stage times summed since the last report (nothing without it).
void moss_block_profile_report();

}  // namespace strata::image
