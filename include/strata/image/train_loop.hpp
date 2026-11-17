// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_loop.hpp - the parts of an image model's LoRA training loop that no model owns
// (src/image/train_loop.cpp), written after ai-toolkit's (its BaseSDTrainProcess, CustomFlowMatchEulerDiscreteScheduler
// and DataLoader): the schedule a step's sigma is read from and the way its index is drawn, the LoRA factors' first
// values, the global gradient norm and AdamW, and the file a LoRA is saved to.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace strata::image::train {

// The same defaults the browser and preprocessing read; one table for every training entry.
#define TRAIN_DEFAULT(name, value) inline constexpr float name = value;
#include "../../../serve/train_defaults.inc"
#undef TRAIN_DEFAULT

/// A training schedule: 1000 timesteps (sigma * 1000, FP32, from 1000 down), ai-toolkit's set_train_timesteps.
struct Schedule {
    std::vector<float> t;   // [1000] timesteps; sigma = t / 1000
};

/// timestep_type "linear": linspace(1000, 1, 1000) (Mage-Flow's).
Schedule schedule_linear();
/// timestep_type "shift" with a fixed shift s: sigma' = s sigma / (1 + (s - 1) sigma) over linspace(1, 0.001, 1000)
/// (MOSS-Image's, s 5).
Schedule schedule_shift(double shift);
/// timestep_type "shift" with dynamic shifting (Qwen-Image-2.1's scheduler config): mu from the latent's tokens
/// (calculate_shift: base_seq 256, max_seq 8192, base_shift 0.5, max_shift 0.9 for Qwen-Image-2.1), the exponential
/// time shift exp(mu) / (exp(mu) + (1 / t - 1)), then stretched to end at `terminal` (0.02; 0: not stretched).
Schedule schedule_dynamic(int64_t image_seq_len, double base_seq, double max_seq, double base_shift, double max_shift,
                          double terminal);

/// How a step's index into the schedule is drawn (ai-toolkit's content_or_style): balanced uniform in [0, 998];
/// content floor(1000 u^3) (the high noise); style floor(1000 (1 - u^3)) (the low noise), clamped to [0, 999].
enum class IndexDraw { balanced, content, style };
bool parse_index_draw(const std::string& s, IndexDraw& out);
int64_t draw_index(std::mt19937_64& rng, IndexDraw how);

/// A LoRA's first values (peft's: A kaiming-uniform with a = sqrt(5), bound 1 / sqrt(K); B zero), host FP32.
void lora_init(std::mt19937_64& rng, int64_t rank, int64_t K, std::vector<float>& A);

/// ai-toolkit's "weighted" timesteps (Anima's default): the loss weight of index i (0 .. 999) of the linear timesteps
/// (src/image/flow_weighting.cpp, ai-toolkit's default_weighing_scheme).
float flow_weight(int64_t index);

/// The sum of squares of every device FP32 array of `parts` (pointer, count: the gradients of a global norm), in
/// double on the host, read back once.
double sq_sum(const std::vector<std::pair<const float*, int64_t>>& parts, void* stream);

/// One AdamW step (torch's, decoupled weight decay) over n device FP32 values: g read times gscale (the clip) and
/// zeroed after, m and v its moments; bc1 = 1 - beta1^step, bc2 = 1 - beta2^step.
struct AdamW {
    float lr = 1e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-6f, wd = 1e-4f;
};
void adamw(float* w, float* g, float* m, float* v, int64_t n, const AdamW& o, int64_t step, float gscale,
           void* stream);

/// A LoRA file: each entry's FP32 host values written as bf16 (ai-toolkit's save dtype), keys sorted, with the
/// metadata strings.
struct Tensor {
    std::string key;
    std::vector<int64_t> shape;   // outer dim first
    std::vector<float> v;
};
void save_safetensors_bf16(const std::string& path, std::vector<Tensor> ts,
                           const std::vector<std::pair<std::string, std::string>>& meta);

/// The weights a run trains, of those a model adapts (every one by default): ai-toolkit's network_kwargs
/// only_if_contains and ignore_if_contains (a part of the weight's name, as lora_names.hpp's tables and the saved keys
/// write it after "diffusion_model.": "attn", "img_mlp", "transformer_blocks.3."), and sd-scripts' block selection
/// (train_double_block_indices: "0-5,10", "all") with the blocks a model leaves out besides (MOSS-Image's --skip).
/// One reading for every trainer: --train-blocks, --only-if-contains, --ignore-if-contains.
struct Targets {
    std::vector<std::pair<int, int>> blocks;   // the blocks trained, ranges [first, last]; empty: all of them
    std::vector<int> skip;                     // blocks left out whatever `blocks` says
    std::vector<std::string> only, ignore;     // parts of a name: one of `only` (where given) and none of `ignore`
    /// Whether the weight `name` (its block `block`, -1 outside the blocks) is trained.
    bool take(int block, const std::string& name) const;
    /// The selection as the saved metadata's ss_xestrata_targets says it ("" for every weight).
    std::string text() const;
};
/// --train-blocks's ("all", or ranges and numbers with commas: "0-5,10"), --only-if-contains's and
/// --ignore-if-contains's (parts with commas) values into `out`; false with err where one does not read.
bool parse_targets(const std::string& blocks, const std::string& only, const std::string& ignore, Targets& out,
                   std::string& err);

/// --kind's values (LoraMeta::kind): "" (not said), "character", "style", "other".
inline bool lora_kind_ok(const std::string& k) {
    return k.empty() || k == "character" || k == "style" || k == "other";
}

/// What a trainer writes into a saved LoRA's metadata, the same keys from every model's (docs/LORA_TRAIN.md, "The
/// saved file"): ai-toolkit's (name, version, software, format), modelspec.title and modelspec.trigger_phrase (where
/// there is one), and sd-scripts' (ss_network_module, ss_network_dim, ss_network_alpha, ss_model: the run's name,
/// ss_base_model_version, ss_steps: the steps so far, ss_max_steps, ss_seed, ss_learning_rate, ss_optimizer_type),
/// ss_xestrata_revision, ss_xestrata_kind (what it was trained to add: "character", "style", "other"; serve's listing
/// reads it) and ss_xestrata_targets (Targets::text) where given, then `extra`.
struct LoraMeta {
    std::string name;                        // the run's (the output's file name without its folder by default)
    std::string base;                        // the base model: "qwen-image-2.1", "moss_image", "anima", "mage-flow"
    std::string module = "networks.lora";    // or networks.lokr
    std::string lr;                          // the learning rate as it was given
    std::string trigger;
    std::string kind;                        // "character", "style" or "other" ("": not said)
    std::string targets;                     // Targets::text() of the run's selection
    int64_t rank = 0, steps = 0, max_steps = 0;
    double alpha = 0.0;
    uint64_t seed = 0;
    std::vector<std::pair<std::string, std::string>> extra;
};
std::vector<std::pair<std::string, std::string>> lora_meta(const LoraMeta& m);

}  // namespace strata::image::train
