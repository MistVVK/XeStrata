// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_trainer.hpp - a LoRA on Qwen-Image-2.1's DiT trained on the GPU (src/image/qi_trainer.cpp),
// one step as ai-toolkit's Qwen-Image-2.1 takes it: the text encoder's states (taken before the trainer runs) through
// the DiT's text projection, the noised latent's tokens through img_in, the 32 blocks (qi_block, checkpointed: each
// block's input kept, the block run again in the backward), the final norm and proj_out, and the flow loss against
// noise - latent; the gradients reach the LoRAs on each block's six products (to_q, to_k, to_v, to_out.0,
// img_mlp.gate_up, img_mlp.out) and nothing else.
//
// The base is the DiT's GGUF with 16-bit weights (Unsloth's qwen-image-2.1-F16.gguf: F16 that holds the bf16 values
// exactly, its text and image projections bf16), every weight resident on the GPU.
#pragma once

#include "strata/image/train_loop.hpp"
#include "strata/image/train_trainer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

class QiTrainer {
public:
    QiTrainer();
    ~QiTrainer();
    QiTrainer(const QiTrainer&) = delete;
    QiTrainer& operator=(const QiTrainer&) = delete;

    /// The DiT's file and a plain LoRA of `rank` (alpha = rank: scale 1, ai-toolkit's) on every block's six products,
    /// A drawn from `seed` (train::lora_init) and B zero.
    /// `int8`: a QLoRA, the blocks' products on ai-toolkit's convrot8 base (train::Product::w8) made from the file's
    /// 16-bit weights, as ai-toolkit's default (quantize, qtype convrot8) trains; else the 16-bit weights themselves.
    /// `max_blocks` (0: all of them) keeps the first blocks only: a check that cuts the DiT short.
    /// `targets`: the products trained (train::Targets; every one by default), the rest without an adapter.
    bool open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err,
              int64_t max_blocks = 0, bool int8 = false, const train::Targets& targets = {});

    /// The shared trainer under this one (the VRAM budget, pause/resume, the state file: train::Control).
    train::Trainer& base();

    /// A caption's text encoder states [n][4096] (host FP32, the system turn already dropped) through the DiT's text
    /// projection, kept on the GPU; returns its id for step().
    int add_text(const std::vector<float>& states, int64_t n);
    /// Plan step buffers for the largest image and text row count in the cache, before the first step.
    void plan_rows(int64_t rows);

    /// One sample's forward and backward: x0 and noise [h w][64] (host FP32, the latent's tokens: x0 the posterior's
    /// draw, normalized), `timestep` the schedule's (sigma * 1000).  The LoRAs' gradients are added into; the loss
    /// comes back.
    float step(int text, const float* x0, const float* noise, int64_t h, int64_t w, float timestep);

    /// The clip (ai-toolkit's max_grad_norm 1: the gradients scaled by 1 / (norm + 1e-6) when the global norm is past
    /// 1) and one AdamW step over every LoRA factor, the gradients zeroed; returns the norm before the clip.
    double update(const train::AdamW& o);

    /// The LoRA as ai-toolkit saves Qwen-Image-2.1's: diffusion_model.transformer_blocks.N.<product>.lora_A.weight
    /// [rank][K] and .lora_B.weight [N][rank], bf16, no alpha (alpha = rank).
    void save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const;

    /// The LoRA factors' gradients as they stand (host FP32, the order save writes them in: per block, per product, A
    /// then B), for a check against a reference.
    std::vector<train::Tensor> grads() const;
    /// The last step's prediction [h w][64] (host FP32).
    std::vector<float> last_pred() const;
    /// Sets a LoRA's factors (host FP32, the same order as grads), for a check against a reference.
    void set_factors(const std::vector<train::Tensor>& ts);

    int64_t blocks() const;
    int64_t width() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

/// Qwen-Image-2.1's training schedule for an h x w latent (its scheduler config: dynamic shifting from the latent's
/// tokens, base 256 / 0.5, max 8192 / 0.9, the exponential shift, stretched to end at 0.02).
train::Schedule qi_train_schedule(int64_t h, int64_t w);

}  // namespace strata::image
