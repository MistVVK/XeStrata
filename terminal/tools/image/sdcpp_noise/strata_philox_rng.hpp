// strata_philox_rng.hpp - the RNG sd-cli uses with --model-args strata_noise=1: stable-diffusion.cpp's PhiloxRNG
// (--rng cuda) stream as its Linux x86 build makes it, computed by strata_noise.hpp on every operating system and CPU.
// Same interface and the same counter handling as PhiloxRNG in src/core/rng_philox.hpp: the k-th randn() call after
// manual_seed uses Philox offset k.  tools/image/sdcpp_noise/apply.py puts this file and strata_noise.hpp into
// src/core/ of a stable-diffusion.cpp checkout and hooks it into StableDiffusionGGML::init.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/strata_noise.hpp"
#include "core/util.h"

namespace strata_noise {
// true when the model args ("a=1,b=2" or "a=1;b=2", sd.cpp's key=value list) hold strata_noise=1
static inline bool requested(const char* model_args) {
    for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
        if (key == "strata_noise" && value == "1") {
            return true;
        }
    }
    return false;
}
}  // namespace strata_noise

class StrataPhiloxRNG : public RNG {
private:
    uint64_t seed   = 0;
    uint32_t offset = 0;

public:
    StrataPhiloxRNG(uint64_t seed = 0) : seed(seed) {}

    std::shared_ptr<RNG> clone() const override {
        return std::make_shared<StrataPhiloxRNG>(*this);
    }

    void manual_seed(uint64_t s) override {
        seed   = s;
        offset = 0;
    }

    std::vector<float> randn(uint32_t n) override {
        std::vector<float> r = strata_noise::randn(seed, n, offset);
        offset += 1;
        return r;
    }
};
