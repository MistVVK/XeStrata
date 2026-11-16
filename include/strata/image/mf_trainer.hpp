// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_trainer.hpp - Mage-Flow's smallest training loop: the model a step runs, the step's own
// forward and backward, and the safetensors file it writes (docs/LORA_TRAIN.md, card 5).
//
// A step trains adapters (train::Adapter, one per block product the SPEC names) with the base weights frozen: the block
// is mf_block run forwards and, on the way back, run again from its input's boundary rows (the tape of a whole
// model does not fit, so every block pays one forward twice).  What the trainer reaches through prefill::Gemm is
// the same math the inference reads (src/image/mf_dit.cpp): the sinusoidal time embedding, the modulations, the
// joint attention without a mask, AdaLayerNormContinuous and the projection to the latent's channels.
#pragma once

#include "strata/image/train_trainer.hpp"

#include "strata/image/mf_train.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_sample.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

/// One frozen product: the FP16 rows the products read and their transposed view, and the FP32 bias.
struct MfTrainProduct {
    int64_t N = 0, K = 0;
    uint16_t* w = nullptr;    // [N][K] FP16
    uint16_t* wt = nullptr;   // [K][N] FP16
    float* b = nullptr;       // [N] FP32
    int8_t* w8 = nullptr;     // a QLoRA's int8 base instead of w (train::Product's w8, sw; null: w)
    float* sw = nullptr;
};

/// One block's trainable side: an adapter per stream and product (null: none), with its own FP32 gradients.
struct MfTrainAdapter {
    train::Adapter ad[2][6];
    int64_t rank = 0;
    train::AdapterMethod method = train::AdapterMethod::lora;
    float scale = 1.f;   // s * alpha / rank, folded (alpha free, default candidate rank)
    float alpha = 16.f;
    float *A[2][6] = {};    // [rank][K] FP32 masters, trained
    float *B[2][6] = {};    // [N][rank] FP32 masters, trained
    float *M[2][6] = {};    // [N] FP32 DoRA magnitudes, trained (null: plain LoRA)
    float *W1[2][6] = {};   // [ol][im] FP32 LoKr masters, trained (null: not LoKr)
    float *W2A[2][6] = {};  // [ok][rank] FP32 LoKr masters, trained
    float *W2B[2][6] = {};  // [rank][inn] FP32 LoKr masters, trained
    // the modulation linears' LoRA (img_mod.1 and txt_mod.1, [6 C][C], ai-toolkit's targets): FP32 masters the
    // one-row product reads as they are and their gradients (null: the modulations not trained)
    float *modA[2] = {}, *modB[2] = {}, *dmodA[2] = {}, *dmodB[2] = {};
    bool off[2][6] = {};   // a product the run does not train (train::Targets): no adapter bound, none saved
};

/// The DiT as the trainer reads it: every block's products and head norms, the weights outside the blocks, and one
/// adapter set.  All device buffers; freed with the model.
struct MfTrainModel {
    void* stream = nullptr;
    int64_t C = 0, F = 0, heads = 0, hd = 0, nblk = 0, txt_dim = 0, in_ch = 0;
    std::vector<int64_t> axes;   // the RoPE's axes (16, 56, 56)
    MfTrainProduct img_in, txt_in, t1, t2, nout, pout;
    int64_t mrank = 0;                 // MageTrail's shared low-rank modulation's width (0: Mage-Flow's own)
    float *mdown_w = nullptr, *mdown_b = nullptr;   // [mrank][C], [mrank] FP32
    float* txt_norm = nullptr;   // [txt_dim] FP32
    struct Block {
        MfTrainProduct prod[2][6];   // [stream][q, k, v, o, mlp1, mlp2]
        MfTrainProduct mod[2];       // [stream] the img_mod.1 / txt_mod.1 linears ([6C][C]; MageTrail's: the bias)
        float* modlow[2] = {};       // [stream] MageTrail's low-rank form [6C][mrank] FP32 (null: mod's FP16 w)
        float* nq[2] = {};           // [hd] FP32 per stream
        float* nk[2] = {};
        MfTrainAdapter* adap = nullptr;   // null: the block runs plain
    };
    std::vector<Block> blk;
    std::string targets;   // the products trained, as the metadata says them (train::Targets::text; "": all)
};

/// Open the bf16 DiT (every weight as FP16, as the engine reads it) and an adapter set of the given rank (0: none).
/// Throws when a shape the trainer reads is not the file's (the file's own block count and widths win).
/// int8 makes the blocks' products a QLoRA's base (ai-toolkit's convrot8, W8A8; the modulations stay FP16).
/// The shared trainer under the model (the VRAM budget, pause/resume, the state file: train::Control).
train::Trainer& mf_trainer_base(MfTrainModel* m);

MfTrainModel* mf_trainer_open(const std::string& dit_path, int64_t rank, void* stream, std::string& err,
                              bool int8 = false);
void mf_trainer_close(MfTrainModel* m);
/// The adapters' starting point (once, before training): A kaiming-uniform (bound 1/sqrt(fan_in)), B = 0, m = norms
/// (method 0 plain LoRA, 1 DoRA; alpha free, default candidate rank).  The FP16 working copies sync from the masters
/// here and after every optimizer step.  mod_targets also trains the modulation linears (img_mod.1, txt_mod.1), as
/// ai-toolkit's Mage-Flow job does: a plain LoRA there whatever the method.  `targets`: the products and modulations
/// trained, by their names ("transformer_blocks.N.attn.to_q", "transformer_blocks.N.img_mod.1"); every one by default.
void mf_trainer_init_adapters(MfTrainModel* m, int method, float alpha, uint64_t seed, bool mod_targets = false,
                              const train::Targets& targets = {});

/// The step's constants from its sigma: the time embedding, the modulations of every block and stream, and the
/// norm_out pair, on the device (the caller keeps them for the backward).  Computed through the products, the same
/// math the inference reads.
struct MfTrainCond {
    float* temb = nullptr;                 // [C] FP32 (and its SiLU image below)
    std::vector<float*> mod;               // [block][stream][6 C] FP32, kept
    float* nmod = nullptr;                 // [2 C] FP32
    float* tsil = nullptr;                 // [C] FP32: the modulations' input, as the base product read it
    float* cos = nullptr, * sin = nullptr;   // [tokens][hd / 2] FP32
};
MfTrainCond mf_trainer_cond(MfTrainModel* m, float sigma, int64_t h, int64_t w, prefill::Gemm& gemm);
void mf_trainer_free_cond(MfTrainModel* m, MfTrainCond& c);

/// One step's kept tensors (device buffers, freed with the state): the noised input and its target, the
/// velocity and its gradient, and every block's input boundary rows (13 x (tokens + rows) x C).
struct MfTrainState {
    float *x0 = nullptr, *xe = nullptr, *v_pred = nullptr, *dv_pred = nullptr;
    float* bounds = nullptr;   // [13][tokens + rows][C]: block i reads bounds[i], writes bounds[i + 1]
    /// Where the device's memory is short: the blocks' inputs as BF16 instead ([13][tokens + rows][C], bounds null),
    /// and the last block's image rows in FP32 for the head
    uint16_t* bounds16 = nullptr;
    float* last = nullptr;
    float *mean = nullptr, *inv = nullptr;   // [tokens] the final norm's statistics, its backward's
    int64_t R = 0, Rt = 0;
    int64_t first_kept = INT64_MAX;          // the blocks from here on kept their forwards for the backward
};
void mf_trainer_free_state(MfTrainModel* m, MfTrainState& s);

/// The forward with its state kept for the backward (bounds, x0/xe, v_pred; dv_pred filled by the backward).
float mf_trainer_forward(MfTrainModel* m, const MfTrainSample& s, const MfTrainCond& c, const float* eps,
                         float sigma, float* v_pred, std::vector<float>* bounds, prefill::Gemm& gemm);
float mf_trainer_forward_keep(MfTrainModel* m, const MfTrainSample& s, const MfTrainCond& c, const float* eps,
                              float sigma, MfTrainState& st, prefill::Gemm& gemm);
/// The backward from the kept state: the norm and output projection, then every block from its boundary input.
/// The adapters' gradients are added into (accumulate over micro-steps, then step the optimizer).
void mf_trainer_backward(MfTrainModel* m, const MfTrainSample& s, const MfTrainCond& c, MfTrainState& st,
                         float sigma, prefill::Gemm& gemm);
/// One AdamW step over every adapter (gradients averaged over accum_n, then zeroed) and the FP16 working copies
/// synced back from the masters.
void mf_trainer_adamw(MfTrainModel* m, float lr, float beta1, float beta2, float eps, float wd, int64_t accum_n);
/// The safetensors file (SPEC D1): dot keys, per-weight .alpha, the metadata v1 writes.
void mf_trainer_save(MfTrainModel* m, const std::string& path, const std::string& title,
                     const std::string& trigger, uint64_t seed, float lr, int64_t steps, int64_t max_steps,
                     const std::string& kind = "");

}  // namespace strata::image
