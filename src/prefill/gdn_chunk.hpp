// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gdn_chunk.hpp - the prompt path's DeltaNet recurrence as a chunked scan on the matrix engines
// (gdn_chunk.cpp); kernels.cpp's Gdn<HK, HV>::recurrence chooses between it and its token-by-token kernels.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace strata::prefill {

/// The output normalization gdn_recurrence applies (rmsnorm(o) * gamma * sigmoid(z), or silu(z)), for the scan to
/// write y16 itself
struct GdnChunkNorm {
    const float* z;
    const float* gamma;
    float eps;
    bool silu;
    uint16_t* y16;
};

/// The forms the scan takes on q's device for h_k key and h_v value heads: 0, 1, 2 for a work-group over all of a
/// value head's 128 columns, half of them, a quarter.  Empty where it cannot run: no FP16 matrix engines it is written
/// for (or XESTRATA_NO_XMX=1), too little local memory, kernels the device does not build, no room for its scratch;
/// the reason is said once.
const std::vector<int>& gdn_chunk_forms(sycl::queue& q, int64_t h_k, int64_t h_v);

/// T tokens of one layer's recurrence in form `form` (one of gdn_chunk_forms'): the state advanced, the raw outputs
/// (o / sqrt(128)) in y.  With `norm` and form 0, y16 is written normalized instead and y is not; returns whether it
/// was (otherwise the caller normalizes y).  Not the bits of the token-by-token kernels: FP16 operands.
bool gdn_chunk_run(sycl::queue& q, int form, int64_t h_k, int64_t h_v, float* state, const float* h,
                   const float* gate, const float* beta, float* y, const GdnChunkNorm* norm, int64_t T);

}  // namespace strata::prefill
