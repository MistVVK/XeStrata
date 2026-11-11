// include/strata/image/qi21_noise.hpp - the initial noise of a Qwen-Image-2.1 picture in Strata's own code (step 5):
// stable-diffusion.cpp 2988060's default noise (--rng cuda: PhiloxRNG, manual_seed(seed), one randn over the latent),
// from Strata's integer-only Philox generator (tools/image/sdcpp_noise/strata_noise.hpp, the same file sd-cli's
// strata_noise=1 patch uses; it reproduces sd.cpp's x86 GCC stream on every OS and CPU).  CPU; no ggml.
//
// Build: the generator's compiled-in sin table comes from `python tools/image/sdcpp_noise/apply.py --prepare DIR`
// (strata_sin_exc_data.inc); compile src/image/noise/qi21_noise.cpp with -I tools/image/sdcpp_noise -I DIR.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::image {

// n values in the latent tensor's own order ([64][H][W] as sd.cpp's noise tensor holds them)
std::vector<float> qi21_noise(uint64_t seed, uint32_t n);

}  // namespace strata::image
