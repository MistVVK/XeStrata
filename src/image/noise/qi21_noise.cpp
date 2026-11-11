// src/image/noise/qi21_noise.cpp - see include/strata/image/qi21_noise.hpp.
#include "strata/image/qi21_noise.hpp"

#include "strata_noise.hpp"   // tools/image/sdcpp_noise (one source for sd-cli's patch and for this)

namespace strata::image {

std::vector<float> qi21_noise(uint64_t seed, uint32_t n) { return strata_noise::randn(seed, n, 0); }

}  // namespace strata::image
