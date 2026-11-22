// engine/src/inkling2/ink_host_avx2.h - Runtime-dispatched AVX2+FMA host dot kernels for InkHybridEngine
// Compiled in a separate TU (ink_host_avx2.cpp) with -mavx2 -mfma -mf16c (/arch:AVX2) and ONLY called
// after strata::quant::caps().avx2 is verified at runtime (fixes #6618 SIGILL on non-AVX2 CPUs).
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>

namespace strata::ink::host_avx2 {

// Returns true if this binary includes the AVX2+FMA host TU AND the running CPU supports AVX2+FMA+F16C.
bool available() noexcept;

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept;
float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept;
float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept;

}  // namespace strata::ink::host_avx2
