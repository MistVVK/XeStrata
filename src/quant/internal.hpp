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

#include "strata/quant/fp16.hpp"
#include "strata/quant/types.hpp"

#if defined(_MSC_VER)
#define STRATA_RESTRICT __restrict
#else
#define STRATA_RESTRICT __restrict__
#endif

namespace strata::quant {


/// ggml's ue4m3 (NVFP4 sub-block scales) decoder, halved as ggml does: a 256-entry table (the NVFP4 dots decode four scales a block, and the ldexp form made them
/// five times slower than ggml's table lookup).  The values are those of the formula in ue4m3_formula.
inline float ue4m3_formula(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = exp == 0 ? std::ldexp((float) man, -9) : std::ldexp(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}
struct Ue4m3Table {
    float v[256];
    Ue4m3Table() {
        for (int i = 0; i < 256; ++i) v[i] = ue4m3_formula((uint8_t) i);
    }
};
inline const Ue4m3Table g_ue4m3_table;   // built before main (ggml's ggml_table_f32_ue4m3 is a plain global table too); nothing decodes a scale during static initialization
inline float ue4m3_to_fp32(uint8_t x) { return g_ue4m3_table.v[x]; }
/// ggml's e8m0 (MXFP4) decoder, halved as ggml does.
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
