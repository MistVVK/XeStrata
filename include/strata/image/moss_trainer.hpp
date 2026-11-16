// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_trainer.hpp - a LoRA on MOSS-Image's DiT as ai-toolkit's moss_image job trains one: the
// blocks' four products (proj.qkv fused, as ai-toolkit fuses the vendor's q, k and v on load; proj.o,
// proj.ffn_up_gate, proj.ffn_down), alpha = rank, every block but those it skips (ai-toolkit's UI skips layers.22,
// the block the negative branch of guidance leaves out), on an int8 base (convrot8) or the 16-bit one.
//
// A step reads an export dir's sample (mf_trainer_sample: the latent, its posterior's deviation where the VAE gives
// one, and the text rows already through the DiT's text embedder, tools/image/mf_pre.py --cache ms), noises it, runs
// every block (moss_block) keeping their inputs, takes the flow loss on the image's velocity and goes back through
// the blocks for the adapters' gradients.
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

class MossTrainer {
public:
    MossTrainer();
    ~MossTrainer();
    MossTrainer(const MossTrainer&) = delete;
    MossTrainer& operator=(const MossTrainer&) = delete;

    /// The DiT folder's shards (or one safetensors file), a LoRA of `rank` on every product `targets` takes (its
    /// skip: the blocks left out; a product by its name "layers.N.proj.qkv", lora::kMossBlock's).
    bool open(const std::string& dit, int64_t rank, bool int8, const train::Targets& targets, uint64_t seed,
              void* stream, std::string& err);
    /// The cached text rows' width (the DiT's hidden size).
    /// The shared trainer under this one (the VRAM budget, pause/resume, the state file: train::Control).
    train::Trainer& base();

    int64_t text_cols() const;
    /// One micro-step at sigma: x0 the sample's latent [tokens][128] (already drawn from the posterior), eps the
    /// noise; the loss times `weight` (a weighted timestep's), and with `backward` the adapters' gradients added.
    float step(const MfTrainSample& smp, const std::vector<float>& x0, const std::vector<float>& eps, float sigma,
               float weight, bool backward);
    /// The optimizer's step over the summed gradients (torch's clip_grad_norm_ at 1 first): the norm before the clip.
    double adamw(const train::AdamW& o);
    /// The LoRA in ai-toolkit's form: diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}.lora_{A,B}.weight,
    /// bf16, q, k and v sharing the fused product's A and splitting its B (convert_lora_weights_before_save).
    void save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

}  // namespace strata::image
