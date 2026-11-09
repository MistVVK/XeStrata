// src/quant/internal.hpp - block layouts and helpers shared by the layer's own sources (never included by a public header).
//
// Block structs and codebooks come from third_party/strata-ggml/strata-quant-common.h (ours since 1B; llama.cpp's ggml-common.h at the pin).
// It is included here in its C declaration flavour: the anonymous unions of the K-quant blocks (`d`/`dmin`/`dm`) are then plain members in C++ too
// (GCC, Clang and MSVC accept them).  A source that needs the codebook tables (the i-quant grids, kvalues_iq4nl, ...) defines STRATA_QUANT_TABLES
// before this header; they are `static const` tables per translation unit.
#pragma once

#define GGML_COMMON_DECL_C
#ifdef STRATA_QUANT_TABLES
#define GGML_COMMON_IMPL_C
#endif
#include "strata-quant-common.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "strata/quant/types.hpp"

#if defined(_MSC_VER)
#define STRATA_RESTRICT __restrict
#else
#define STRATA_RESTRICT __restrict__
#endif

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

/// fp16 -> fp32, exact (the same values as F16C's VCVTPH2PS and ggml's table, for every input).
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

/// fp32 -> fp16, round to nearest even (the same bits as F16C's VCVTPS2PH for every finite input; NaN comes out as 0x7E00).
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

/// ggml's ue4m3 (NVFP4 sub-block scales) and e8m0 (MXFP4) decoders, halved as ggml does.
inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = exp == 0 ? std::ldexp((float) man, -9) : std::ldexp(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}
inline float e8m0_to_fp32_half(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : (uint32_t) (x - 1) << 23;
    return f32_from_bits(bits);
}

/// ggml's nearest_int: round half to even through the 2^23 + 2^22 magic number (|v| <= 4194303).
inline int nearest_int(float v) {
    const float val = v + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

}  // namespace strata::quant
