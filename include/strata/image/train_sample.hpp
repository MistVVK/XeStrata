// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_sample.hpp - a training step's sample read back from the disk caches the preprocessing
// writes (tools/image/mf_pre.py): one reader a cache form, apart from the trainers that take the values.  The three
// models that train from an export dir read `mf_trainer_sample` (Mage-Flow's trainer, MOSS-Image's and Anima's), and
// Qwen-Image-2.1's trainer reads its own caches (`qi_train_latent`, `qi_train_states`).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

/// One sample's disk form (tools/image/mf_pre.py's export): the latent [tokens][ch] F32, the text [rows][txt_dim]
/// F32 (the bf16 widened), the empty prompt's rows beside them (caption dropout reads those), and the bucket's
/// own rows and columns (the RoPE tables read them).
struct MfTrainSample {
    std::vector<float> lat;      // [tokens][ch], the posterior's mean
    std::vector<float> lat_sd;   // [tokens][ch], its standard deviation (empty: an export without it)
    std::vector<float> te;       // [rows][txt_dim]
    std::vector<float> uncond;   // [uncond_rows][txt_dim], the empty prompt
    int64_t tokens = 0, rows = 0, uncond_rows = 0, h = 0, w = 0;
    int64_t ch = 128;            // a token's latent values (Mage-Flow's and MOSS-Image's 128, Anima's 64)
};

/// Read one sample of an exported run dir by its manifest stem.
MfTrainSample mf_trainer_sample(const std::string& dir, const std::string& stem, int64_t txt_dim);
/// Largest image token count plus largest text-row count in an export's manifest, before the first step.
int64_t mf_train_max_tokens(const std::string& dir);

/// Qwen-Image-2.1's latent cache: the width and height, then the posterior's mean and standard deviation [64][h][w].
struct QiTrainLatent {
    int64_t h = 0, w = 0;
    std::vector<float> mean, sd;
};
QiTrainLatent qi_train_latent(const std::string& path);
/// Qwen-Image-2.1's states cache: the rows, then that many rows of 4096 values.
std::vector<float> qi_train_states(const std::string& path, int64_t& rows);

}  // namespace strata::image
