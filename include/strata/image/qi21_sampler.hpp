// include/strata/image/qi21_sampler.hpp - Qwen-Image-2.1 Turbo's sampling loop pieces in Strata's own code (step 5):
// the fixed sigma schedule, the timestep the DiT gets, and one Euler step with CFG 1 for the flow model, written to
// give stable-diffusion.cpp 2988060's bytes (src/runtime/denoiser.hpp sample_euler, DiscreteFlowDenoiser,
// src/pipeline/diffusion_engine.cpp's model callback).  CPU, plain float; no ggml.
//
// sd.cpp's step, per value, each line a separate pass over the tensor (so nothing is fused):
//   c_skip = 1, c_out = -sigma, c_in = 1 (DiscreteFlowDenoiser::get_scalings); the DiT gets x * c_in = x and
//   t = sigma * 1000 (sigma_to_t); with CFG 1 the guidance returns the DiT output v unchanged;
//   a = v * c_out;  b = x * c_skip;  den = a + b;  d = (x - den) / sigma;  dt = sigma_next - sigma;  e = d * dt;  x += e
//   (a step whose sigma equals the next one is skipped, as sample_euler does).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace strata::image {

// the schedule serve/images.py passes to sd-cli as --sigmas for Turbo's 8 steps (TURBO_SIGMAS)
extern const char* const QI21_TURBO_SIGMAS;

// a comma-separated sigma list parsed as sd-cli does (each item trimmed, std::stof = strtof); empty on an error
std::vector<float> qi21_parse_sigmas(const std::string& text);

// the DiT's timestep for a sigma: sigma * 1000 in float (DiscreteFlowDenoiser::sigma_to_t)
float qi21_timestep(float sigma);

// x (n values) <- one Euler step from sigma to sigma_next with the DiT output v, in sd.cpp's order (above)
void qi21_euler_step(float* x, const float* v, size_t n, float sigma, float sigma_next);

}  // namespace strata::image
