// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/rope.hpp - NEOX partial RoPE, host-callable (P2.S2).
//
// Pairing is NEOX: (i, i + n_rot/2), NOT the adjacent pair (2i, 2i+1).  The rotation is PARTIAL - only the
// first n_rot of head_dim are touched, 64 of 256 in this artifact - and the tail passes through.
//
// The cos/sin table is built on the HOST in float64 (see Strata's src/kernels/cuda/rope.cu for why) and the kernel is
// a pure rotation over it.  The table depends on (n_rot, theta, position) and not on the token, so it is worth
// caching; `build_rope_table` fills `max_pos` positions of `n_rot/2` pairs each.
//
// ROPE SCALING RIDES IN THE TABLE, NOT IN THE KERNEL.  The kernels below read cos/sin values and
// cannot tell a scaled table from an unscaled one - with scaling on, the scaled angles (and YaRN's
// mscale magnitude correction, folded into the same values) simply ARE the table: the config enters
// once, here, at build time (`rope_scaling.hpp`).
#pragma once

#include "strata/kernels/rope_scaling.hpp"

#include <cstdint>

#include "strata/kernels/rope_scaling.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_ROPE_HD __host__ __device__
#else
#define STRATA_ROPE_HD
#endif

namespace strata::kernels {

/// ONE NEOX PAIR: `(a, b) -> (a*c - b*s, a*s + b*c)`.
///
/// This exists so the pairing and the sign convention live in ONE place: the kernel (qk_norm_rope.hpp) and the
/// host references in the tests rotate with the same convention, because a mismatched rotation produces a
/// well-formed key pointing the wrong way.
///
/// It is `__host__ __device__` (the `f16_bits.hpp` pattern) rather than device-only so a host-side reference in
/// a test can call THE SAME function instead of transcribing it.  A transcription is exactly how the pairing
/// convention gets written down wrong.
///
/// In-place is safe: the caller writes both outputs from the two inputs it read, and under NEOX no two pairs
/// share an element (`i` pairs with `i + n_rot/2`, so the halves are disjoint).
STRATA_ROPE_HD inline void rope_neox_pair(float a, float b, float c, float s, float& oa, float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab);

/// The same table under the process's rope scaling (rope_scaling.hpp).  For RopeScalingType::None this is the
/// five-argument builder above (called as it is); for linear and YaRN the scaled angles, YaRN's magnitude included.
void build_rope_table(int n_rot, const RopeScaling& scaling, int max_pos, float* cos_tab, float* sin_tab);

/// The LLMs' one RoPE (qk_norm_rope.hpp's kernel, the heads rotated only): T rows of `heads` heads of head_dim, a row
/// ld_in floats apart in `x` and ld_out in `out` (in place where out is x), the first n_rot of each head rotated in
/// pairs (i, i + n_rot / 2) at positions[t] (null: pos0 + t), Qwen3.8-Flash-Next's IMRoPE sections and the session's
/// table where n_rot is 64, else the angle computed under `scaling`.  native_rope_apply and the prompt's rope are
/// this.
void rope_rows(const float* x, float* out, int64_t T, int64_t heads, int head_dim, int64_t ld_in, int64_t ld_out,
               int n_rot, const RopeScaling& scaling, const int* positions, int64_t pos0, void* stream);

}  // namespace strata::kernels
