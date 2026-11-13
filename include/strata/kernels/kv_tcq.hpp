// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/kv_tcq.hpp - the trellis-coded K/V cache (--kv tcq4, tcq5), after YASAKANI's KV5
// (kvx_codec.cuh, kvq_fast.cuh, kvq_vit2.cuh; rewritten for SYCL).
//
// K, V and the queries are rotated by the signed 256-point Hadamard transform (fwht256_signed: <Rq, Rk> = <q, k>, and
// the attention's output, a mix of rotated values, is rotated back).  Each rotated row of 256 is then coded on its
// own: four chains of 64 values, each a path through a trellis of 256 states (a state the last 8 bits of the chain's
// code, B bits shifted in a value), the path of least squared error found by an integer Viterbi search, at the scale of
// least error among a few multipliers of the row's root mean square.  A value is read back from the 8 bits of code
// that end at it (fewer at a chain's start, the bits before it zero) through a codebook of 256 (kv_tcq_tables.hpp):
// no state is kept, so the attention decodes any value on its own.
//
// A row is B x 32 bytes of code, 32-bit words with the first bit the most significant, then its scale (FP32, rounded
// to FP16 first): 132 bytes for B = 4 (Q4_0's 144), 164 for B = 5 (INT8's 264).  A row's bits are a function of the
// row alone: one token or a prompt's chunk at a time, the same bits.
//
// The codebooks: each a trellis quantizer trained on normal values (tools/kv_tcq_table.py), XeStrata's own: on held-out
// normal rows the 5-bit one's error was 28.02 dB below the signal, YASAKANI's 27.97 (kvq_u5.bin), and the two gave the
// same KL on Qwen3.5-9B (2026-11-14, RX 9060 XT).
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int kTcqChain = 64;   ///< values a chain: a row of 256 is four

/// Bytes per cell and KV head: the code and the scale.  bits 4 or 5.
inline uint64_t kv_tcq_bytes_per_head(int head_dim, int bits) { return (uint64_t) head_dim * bits / 8 + 4; }

/// The signed rotation of rows of 256 floats (in place): forward x -> H D x / 16, inverse y -> D H y / 16, with D the
/// signs of kvx_codec.cuh's hash of the dimension.
void fwht256_signed(float* data, int64_t n_rows, bool inverse, void* stream);

}  // namespace strata::kernels
