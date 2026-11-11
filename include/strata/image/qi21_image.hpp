// include/strata/image/qi21_image.hpp - the small steps of stable-diffusion.cpp 2988060's Qwen-Image-2.1 picture
// pipeline that sit between Strata's tested parts (step 5, src/image/pipeline/qi21_glue.cpp), each written in sd.cpp's
// own float order so that, given the same inputs, it gives sd.cpp's bytes.  CPU, plain float; no ggml.
//
//   the starting latent  src/runtime/denoiser.hpp DiscreteFlowDenoiser::noise_scaling on a zero init latent:
//                        x = latent * (1 - sigma) + noise * sigma, each product and the sum rounded on its own
//   the condition        src/conditioning/conditioner.hpp apply_token_weights + LLMEmbedder::encode_prompt's slice:
//                        nothing when every weight is 1; else mean (double sum, then float) of the whole [n][hidden]
//                        block, each row times its weight, the mean again, every value times (old mean / new mean)
//                        in float when both are finite and the new one is not 0; then the first n_system rows dropped
//   the 0..1 picture     src/model/vae/vae.hpp scale_tensor_to_0_1: v = (x + 1) * 0.5, then max(0, min(1, v))
//
// Used by the test program strata-image (src/image/qi21_image_main.cpp); nothing shipped calls it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

// noise (n values, the latent's order) -> the sampler's first x at sigma0, as sd.cpp computes it from a zero latent
std::vector<float> qi21_initial_latent(const std::vector<float>& noise, float sigma0);

// hidden: n x hidden floats (the text encoder's rows for every id, template included), in place; returns the rows
// from n_system on (what the DiT gets)
std::vector<float> qi21_condition(std::vector<float> hidden, int n, int hidden_dim, const std::vector<float>& weights,
                                  int n_system);

// the decoder's output (about -1..1) to 0..1 in place, n values
void qi21_picture_to_unit(float* x, size_t n);

// the PNG "parameters" text strata-image writes (the prompt, then one line of settings)
std::string qi21_parameters_text(const std::string& prompt, int steps, uint64_t seed, int width, int height,
                                 const std::string& model_name);

}  // namespace strata::image
