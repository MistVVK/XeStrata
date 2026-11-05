// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include <cstdint>

namespace strata::kernels {
// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);
/// n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each as the single call.
/// `shared_gate` (n_tok raw gates): the shared rows are scaled by their gate's sigmoid here, as shared_expert_multi's
/// last step would have (upstream b08cf3e1: the same bits, one launch fewer).
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream, const float* shared_gate = nullptr,
                              const int32_t* slot = nullptr);
/// native_moe_combine_multi with moe_hit_add folded in (upstream a9917a55's STRATA_FUSE_HITS): the entries listed in
/// dst[0, *count) (entry = token * k + expert, at most n_tok * k of them) read parts + hits, as the add moe_hit_add makes
/// in place would leave them; the same bits, one launch fewer.  STRATA_FUSE_HITS=0: the two launches.
bool native_moe_fuse_hits();
void native_moe_combine_multi_hits(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                   const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                   int n_tok, void* stream, const float* shared_gate = nullptr);
}
