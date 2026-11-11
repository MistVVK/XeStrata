// src/image/vae/qi21_pixels.cpp - see include/strata/image/qi21_pixels.hpp.  Contraction off for the file: v * 255 + 0.5
// must round twice, as sd.cpp's x86 build computes it, on every compiler and CPU.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "strata/image/qi21_pixels.hpp"

#include <cstddef>

namespace strata::image {

uint8_t qi21_float_to_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    volatile float m = v * 255.0f;   // its own rounding (volatile: no fused multiply-add on any compiler)
    return static_cast<uint8_t>(m + 0.5f);
}

void qi21_picture_to_u8(const float* planar, int w, int h, int c, uint8_t* out) {
    const size_t px = (size_t) w * (size_t) h;
    for (size_t i = 0; i < px; ++i)
        for (int k = 0; k < c; ++k) out[i * c + k] = qi21_float_to_u8(planar[(size_t) k * px + i]);
}

}  // namespace strata::image
