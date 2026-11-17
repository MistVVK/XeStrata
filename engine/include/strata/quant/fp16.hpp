// include/strata/quant/fp16.hpp - IEEE half <-> float, scalar and exact (own-stack phase 2).
//
// The same values as F16C's VCVTPH2PS / VCVTPS2PH and ggml's table for every finite input (fp16 -> fp32 is exact; fp32 -> fp16 rounds to nearest even; a NaN
// comes out as 0x7E00).  quant_parity checks both over every fp16 value and 4M random floats.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::quant {

inline float f32_from_bits(uint32_t w) {
    float f;
    std::memcpy(&f, &w, sizeof f);
    return f;
}
inline uint32_t f32_to_bits(float f) {
    uint32_t w;
    std::memcpy(&w, &f, sizeof w);
    return w;
}

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const float exp_scale = f32_from_bits(0x7800000u);   // 2^-112
    const float normalized = f32_from_bits((two_w >> 4) + (0xE0u << 23)) * exp_scale;
    const float denormalized = f32_from_bits((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t result = sign | (two_w < (1u << 27) ? f32_to_bits(denormalized) : f32_to_bits(normalized));
    return f32_from_bits(result);
}

inline uint16_t fp32_to_fp16(float f) {
    const float scale_to_inf = f32_from_bits(0x77800000u);    // 2^112
    const float scale_to_zero = f32_from_bits(0x08800000u);   // 2^-110
    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w = f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = f32_from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t) ((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

}  // namespace strata::quant
