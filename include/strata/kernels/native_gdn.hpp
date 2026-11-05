// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include "strata/kernels/gdn.hpp"

namespace strata::kernels {

// Pinned CUDA gated_delta_net recurrence for one token, S=128 and h_v % h_k=0.
// q/k are normalized F32 inputs; q MUST NOT be pre-scaled by 1/sqrt(S).
// Scaling is applied to the completed readout, as in the pinned fused operator.
// State retains Strata's layout: state[(i*h_v+h)*S+j]. The caller owns it and
// output; both must be disjoint from each other and from every read-only input.
// All spans must be aligned to four bytes. A nonnull ordered stream is required.
// No allocation, wait, or change to normalization/gate preprocessing occurs.
void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shapes, void* stream);

// The same recurrence over T tokens in one launch, the state in registers between them: token t reads q at
// h + t * h_stride (k at + h_k * S, v at + 2 * h_k * S), its gate and beta at + t * h_v, and writes its readout at
// output + t * out_stride; after token t < n_snaps the state goes to snaps[t] as well (a speculative window's rows to
// go back to; at most kNativeGdnMaxSnaps).  One read and one write of the state instead of T of each, each work-item
// a column.
inline constexpr int kNativeGdnMaxSnaps = 8;
void native_gdn_step_rows(float* state, const float* h, int64_t h_stride, const float* gate, const float* beta,
                          float* output, int64_t out_stride, int T, float* const* snaps, int n_snaps,
                          const GdnShapes& shapes, void* stream);

} // namespace strata::kernels
