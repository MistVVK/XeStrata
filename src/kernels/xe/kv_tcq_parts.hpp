// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/kv_tcq_parts.hpp - a trellis-coded K/V row's values (kv_tcq.hpp) for the attention kernels that read
// the cache (qsa_decode_attn.cpp, prompt_attn.cpp), and the rotation's signs; after YASAKANI's kvx_codec.cuh and
// kvq_attn.cuh (TbLaneDec), rewritten for SYCL.
#pragma once

#include "strata/kernels/kv_tcq.hpp"
#include "strata/kernels/kv_tcq_tables.hpp"
#include "strata/kernels/trellis.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::tcq {

// The rotation's sign of dimension d (kvx_codec.cuh's kvx_sign: a 32-bit integer hash of d + 1, its lowest bit)
inline float sign_of(int d) {
    uint32_t x = (uint32_t) d + 1u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) ? -1.0f : 1.0f;
}

// State st's value in 1/256 of the row's scale
inline int code_of(int bits, int st) { return bits == 4 ? kCode4[st] : kCode5[st]; }

inline int row_bytes(int head_dim, int bits) { return head_dim * bits / 8 + 4; }

// The row's scale: a code over 256 times it is the value
inline float row_scale(const uint8_t* r, int head_dim, int bits) {
    return *reinterpret_cast<const float*>(r + head_dim * bits / 8);
}

// 64 bits of a row's code from bit `at` on, the first the most significant: the shared trellis window
// (kernels/trellis.hpp, Order::Msb).  The words read stay inside the row: the code's last word is followed by the scale's.
inline uint64_t bits64(const uint32_t* w, int at) { return trellis::window_msb(w, at); }

// A code's value over 256 from the codebook in local memory (tab, code_of(bits, i) / 256), else from the constant one
inline float code_val(int bits, int st, const float* tab) {
    return tab != nullptr ? tab[st] : (float) code_of(bits, st) * (1.0f / 256.0f);
}

// Value d of a row (r: its first byte): the shared trellis state (kernels/trellis.hpp, Order::Msb)
inline float value(const uint8_t* r, int head_dim, int bits, int d, const float* tab = nullptr) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(r);
    const int end = bits * (d + 1), first = bits * (d & ~(kTcqChain - 1));
    const int st = (int) trellis::state_msb(w, end, first);
    return code_val(bits, st, tab) * row_scale(r, head_dim, bits);
}

// Values dd .. dd + N - 1 of a row (dd a multiple of N, N of 4 or 8, inside one chain): one 64-bit window of code
// holds them and the bits before the first (none at a chain's start)
template<int N>
inline void values(const uint8_t* r, int head_dim, int bits, int dd, float* out, const float* tab = nullptr) {
    static_assert(N == 4 || N == 8, "a 64-bit window holds 8 values of 5 bits and the 3 before");
    const uint32_t* w = reinterpret_cast<const uint32_t*>(r);
    const float sc = row_scale(r, head_dim, bits);
    const int pre = (dd & (kTcqChain - 1)) == 0 ? 0 : 8 - bits;
    const uint64_t z = bits64(w, bits * dd - pre);
#pragma unroll
    for (int e = 0; e < N; ++e) {
        const int end = pre + bits * (e + 1);
        out[e] = code_val(bits, (int) (z >> (64 - end)) & 0xFF, tab) * sc;
    }
}

// The ring of deferred coding (kv_tcq_ring_bytes): a plane's row (cell mod kTcqRing, KV head) is plane * kTcqRing *
// n_kv + slot * n_kv + h
inline int ring_row(int plane, int slot, int h, int n_kv) { return (plane * kTcqRing + slot) * n_kv + h; }
inline const int8_t* ring_codes(const uint8_t* ring, int row, int head_dim) {
    return reinterpret_cast<const int8_t*>(ring) + (int64_t) row * head_dim;
}
inline const uint16_t* ring_scales(const uint8_t* ring, int row, int head_dim, int n_kv) {
    return reinterpret_cast<const uint16_t*>(ring + 2ll * kTcqRing * n_kv * head_dim) + (int64_t) row * (head_dim / 64);
}
// the cell below which every cell is coded (the rest are read from the ring)
inline int* ring_coded(uint8_t* ring, int head_dim, int n_kv) {
    return reinterpret_cast<int*>(ring + 2ll * kTcqRing * n_kv * (head_dim + head_dim / 64 * 2));
}
inline int ring_coded(const uint8_t* ring, int head_dim, int n_kv) {
    return *reinterpret_cast<const int*>(ring + 2ll * kTcqRing * n_kv * (head_dim + head_dim / 64 * 2));
}

}  // namespace strata::kernels::tcq
