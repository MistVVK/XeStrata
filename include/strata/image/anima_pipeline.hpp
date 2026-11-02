// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/anima_pipeline.hpp - Anima's text to image on the GPU: Qwen3-0.6B-Base's states of the prompt
// (QiText, every token after the final norm), the LLM adapter and the DiT's flow sampling (AnimaDit), and the image
// through Wan 2.1's VAE (QiVae), within a VRAM budget.
//
// The sampler is ComfyUI's for the model (ModelSamplingDiscreteFlow, shift 3, the timestep sigma itself): the
// "simple" schedule, then er_sde (the model card's default, ComfyUI's sample_er_sde: the extended reverse-time SDE
// solver of three stages, 200 points of its integrals, the noise scaler x (e^(x^0.3) + 10)) or Euler.  er_sde's noise at
// each step is the CPU generator's (torch_cpu_randn, a seed a step), not ComfyUI's GPU generator's, so the same seed makes
// another picture than ComfyUI's; the latent's start is PyTorch's CPU noise of the seed, as ComfyUI makes it.
// Guidance: u + cfg (c - u) with the negative prompt's context (CFG 1, Turbo's, runs the prompt alone).
//
// The budget is split as MossPipeline's: the weights that always stay (the DiT's parts outside its blocks, the VAE's),
// the DiT's activations and blocks, then the text encoder (a small one: kept beside the DiT where it fits, else
// closed once the states are made); the VAE decodes in what is left, in tiles past that.
#pragma once

#include "strata/image/anima_dit.hpp"
#include "strata/image/img_start.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/image/qi_vae.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// ComfyUI's "simple" schedule of a flow model at `shift` over 1000 training steps: n sigmas from the top, then 0.
std::vector<float> anima_sigmas(int n, float shift = 3.0f);

class AnimaPipeline : public ImagePipeline {
public:
    /// `dit`: the DiT's safetensors, `vae`: Wan 2.1's VAE (ComfyUI's qwen_image_vae.safetensors), `text`: Qwen3-0.6B-Base's
    /// GGUF.  `q8`: the DiT's blocks in 8 bits (1), FP16 (0) or by the budget (-1).
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err, int q8 = -1);
    /// The release's blocks in an expanded DiT (AnimaDit::set_lora_block_map), for the adapters made for the release.
    void set_lora_block_map(std::vector<int> map) { block_map_ = std::move(map); }
    /// The blocks' plain LoRAs folded into their weights (the default) or added to each product (set_lora_bake).
    void set_lora_bake(bool on) { lora_bake_ = on; }
    void set_budget(uint64_t bytes) override;
    uint64_t budget() const override { return budget_ > 0 ? budget_ : auto_budget_; }
    void release(bool all = false) override;
    void fit(uint64_t reserve) override;
    uint64_t held() const override;
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) override;
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;
    bool warm_up_request(QiRequest& r) const override;

private:
    QiPlan plan(int64_t hw, int n, bool text = false);
    void ensure_open(bool q8);
    void use_loras(const QiRequest& r);
    bool want_q8(int64_t rows, int n);
    void open_text();
    void encode(const std::vector<int32_t>& ids, std::vector<float>& out);
    void encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls);
    // the sampling: n contexts, er_sde or Euler over sigmas from x (tokens), the mask's blend after each step
    void sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, int n, float cfg, bool er_sde,
                uint64_t seed, const std::function<bool(int, int)>& step, const std::function<void(float*, float)>& after,
                const std::function<void(int, const std::function<void(float*)>&)>& peek);
    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_;
    int q8_ = -1;
    mutable int w8_faster_ = -1;
    mutable int64_t w8_rows_ = 0;
    std::unique_ptr<QiText> te_, te_size_;
    std::unique_ptr<AnimaDit> dit_, dit_size_, dit16_size_;
    std::vector<std::vector<uint8_t>> dit_host_;
    std::vector<int> block_map_;
    bool lora_bake_ = true;
    std::unique_ptr<QiVae> vae_, vae_size_;
    uint64_t budget_ = 0, auto_budget_ = 0, dit_set_ = 0;
    int64_t mlp_set_ = -1;
    std::vector<int32_t> last_ids_, last_neg_ids_;
    std::vector<LoraUse> te_loras_;   // the request's text encoder adapters, bound as the encoder opens
    std::string te_lora_;             // those the kept states were made with (lora_key_of)
    std::vector<float> last_states_, last_neg_states_;
    // what the DiT's contexts were set from: the states and the T5 ids and weights
    struct CtxKey {
        std::vector<float> states, t5w;
        std::vector<int32_t> t5;
        bool operator==(const CtxKey&) const = default;
    } ctx_key_[2];
};

}  // namespace strata::image
