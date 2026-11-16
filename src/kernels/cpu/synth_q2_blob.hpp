// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// A Q2_0 expert blob in Strata's form (strata/kernels/cpu/expert.hpp) made from a seed: for expert_parity and
// pool_test where pack/full/experts.bin, a Q2_0 pack's, is not there (no machine of the project keeps one: the packs
// read their experts from the GGUF).  The codes are uniform random bytes, the fp16 scales +-[0.005, 0.05] (Q2_0's
// magnitudes); one seed an index, so the blob of an index is the same on every run and the next index's differs.
// The kernels' checks need a blob's form, not a model's weights: the VNNI kernel against the scalar oracle on it.

#include "strata/kernels/cpu/expert.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace strata::kernels::cpu::test {

/// float -> IEEE half, round to nearest even, for normal halves (the scales here are)
inline uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t) ((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    uint32_t h = sign | ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rest = man & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

inline void synth_q2_blob(long long index, std::vector<uint8_t>& out) {
    out.assign(BLOB, 0);
    std::mt19937_64 rng(0x51A7A0ull + (unsigned long long) index);
    for (size_t i = 0; i < O_GU_SCALES; ++i) out[i] = (uint8_t) (rng() & 0xFFu);
    std::uniform_real_distribution<float> mag(0.005f, 0.05f);
    for (size_t o = O_GU_SCALES; o + 2 <= BLOB; o += 2) {
        const float s = (rng() & 1u) ? mag(rng) : -mag(rng);
        const uint16_t h = half_bits(s);
        std::memcpy(&out[o], &h, 2);
    }
}

}  // namespace strata::kernels::cpu::test
