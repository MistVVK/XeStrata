// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/anima_trainer.hpp - a LoRA on Anima's DiT as ai-toolkit's anima job trains one: every linear
// layer of the blocks (the self- and cross-attention's q, k, v and output, the MLP's two, and the three modulations'
// two low-rank layers each), alpha = rank, on the 16-bit base (ai-toolkit does not quantize Anima's).
//
// A step reads an export dir's sample (mf_trainer_sample: the latent tokens [h w][64] with the posterior's deviation,
// and the context's rows already through the DiT's LLM adapter, tools/image/mf_pre.py --cache an), noises it, runs
// every block (anima_block) keeping their inputs, takes the flow loss on the velocity and goes back through the
// blocks, the modulations' adapters taking their gradients from every block's.
#pragma once
#include "strata/image/mf_trainer.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_trainer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

class AnimaTrainer {
public:
    AnimaTrainer();
    ~AnimaTrainer();
    AnimaTrainer(const AnimaTrainer&) = delete;
    AnimaTrainer& operator=(const AnimaTrainer&) = delete;

    /// The DiT's safetensors file (its names with or without `net.` or `model.diffusion_model.`), a LoRA of `rank` on
    /// every product and modulation row `targets` takes (by its name: "blocks.N.self_attn.q_proj",
    /// "blocks.N.adaln_modulation_mlp.1").
    bool open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err,
              const train::Targets& targets = {});
    /// The cached text rows' width (the context's: the LLM adapter's).
    /// The shared trainer under this one (the VRAM budget, pause/resume, the state file: train::Control).
    train::Trainer& base();

    int64_t text_cols() const;
    /// One micro-step at sigma: x0 the latent tokens [tokens][64] (drawn from the posterior), eps the noise; the
    /// loss times `weight` (ai-toolkit's weighted timesteps), and with `backward` the adapters' gradients added.
    float step(const MfTrainSample& smp, const std::vector<float>& x0, const std::vector<float>& eps, float sigma,
               float weight, bool backward);
    /// The optimizer's step over the summed gradients (torch's clip_grad_norm_ at 1 first): the norm before the clip.
    double adamw(const train::AdamW& o);
    /// The LoRA in ai-toolkit's form for Anima (convert_lora_weights_before_save: ComfyUI's names under
    /// diffusion_model.blocks.N., bf16).
    void save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

}  // namespace strata::image
