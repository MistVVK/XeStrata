// src/quant/arm/tier_common.hpp - what the Arm tier files of the layer need before they include ggml's ported NEON code (arm_dots.inc, arm_act.inc).
//
// dot_neon.cpp and act_neon.cpp are compiled with the flavour ggml-cpu ships on arm64 (cmake/backends/quant.cmake): armv8.2-a + dotprod + fp16 + i8mm, no SVE
// (the DGX Spark's STRATA_PORTABLE build; the Mac's native build has the same features and no SVE).  Never include this header from a file compiled without them.
#pragma once

#define STRATA_QUANT_TABLES
#include "../internal.hpp"

#include <arm_neon.h>

#include <cassert>
#include <cstdlib>

#if !defined(__ARM_NEON) || !defined(__ARM_FEATURE_DOTPROD)
#error "the Arm tier needs NEON and the dot product extension (armv8.2-a+dotprod)"
#endif
#if defined(__ARM_FEATURE_SVE)
#error "the Arm tier is the no-SVE flavour of ggml-cpu: compile it without SVE (-march=armv8.2-a+dotprod+fp16+i8mm)"
#endif

#define UNUSED(x) (void) (x)

// ggml's own choice on arm64: the hardware conversions through __fp16 (the same values as the software ones for every finite input).
namespace strata::quant {
inline float h2f(uint16_t h) {
    __fp16 t;
    std::memcpy(&t, &h, sizeof t);
    return (float) t;
}
inline uint16_t f2h(float f) {
    const __fp16 t = f;
    uint16_t r;
    std::memcpy(&r, &t, sizeof r);
    return r;
}
}  // namespace strata::quant
