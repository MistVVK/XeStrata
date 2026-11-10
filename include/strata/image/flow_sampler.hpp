// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/flow_sampler.hpp - The samplers of a flow model (x_t = sigma noise + (1 - sigma) x0, the
// model's output a velocity, the denoised x - sigma v) over a DiT's forward: Euler and er_sde (the extended
// reverse-time SDE solver of three stages, its integrals over 200 points, the noise scaler x (e^(x^0.3) + 10)).
// er_sde's first sigma is moved below 1, to the model's schedule at 1e-4 of the way from the top (a sigma of 1 has no
// signal for its log-SNR), and its noise at each step is PyTorch's CPU generator's of a seed a step (torch_cpu_randn),
// not a GPU generator's.  Guidance: u + cfg (c - u) of the contexts' velocities, or guidance.hpp's.
#pragma once

#include "strata/image/guidance.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace strata::image {

/// The DiT's velocities [n][N] (device FP32) of the latent x [N] at sigma, for its n contexts.
using FlowForward = std::function<void(const float* x, float sigma, int n, float* out)>;

// MOSS-Image's original step-window SLG combines the skipped velocity differently without a negative pass.
struct FlowStepSkip {
    std::function<bool(int)> active;
    std::function<void(const float*, float, float*)> forward;
    float scale;
};

/// The sampling of x [N] (device FP32, in place) over sigmas (the steps and the terminal 0) on `stream`.  `shift`: the
/// model's schedule's (er_sde's first sigma); `seed`: er_sde's noise; step, after and peek as the pipelines' (progress,
/// a mask's blend at each step's next sigma, a partial image's latent: the step's denoised values).
/// `guide` (if set and on): guidance.hpp's guidance, made over these sigmas, and `skip_forward` its skip layer
/// guidance's pass (the prompt's velocity with its blocks skipped).
void flow_sample(void* stream, const FlowForward& forward, float* x, int64_t N, int n, const std::vector<float>& sigmas,
                 float cfg, bool er_sde, float shift, uint64_t seed, const std::function<bool(int, int)>& step,
                 const std::function<void(float*, float)>& after,
                 const std::function<void(int, const std::function<void(float*)>&)>& peek, Guidance* guide = nullptr,
                 const std::function<void(const float* x, float sigma, float* out)>& skip_forward = {},
                 bool velocity_euler = false, const FlowStepSkip* step_skip = nullptr, bool unfused_euler = false);

}  // namespace strata::image
