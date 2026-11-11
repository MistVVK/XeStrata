// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include <cstdint>

namespace strata::kernels {
// Whether the native router takes this shape: 512 or 256 experts (a pruned model's), top 10.
bool native_router_takes(int n_expert, int k);
// Pinned CUDA topk-moe contract for ONE token, `n_expert` experts (native_router_takes), top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads n_expert finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream, int n_expert = 512);
/// n_tok rows at once (logits [n,n_expert], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream,
                               int n_expert = 512);
/// router_top10 (router_top10.hpp) and the doorbell (elementwise.hpp's doorbell_publish) in one launch: the rows routed
/// as there (n_expert <= 1,024, k <= 32 and below n_expert), the ids and weights written to the device buffers and to
/// the mapped `ids_out`/`weights_out`, `x` (n_x floats) copied to `x_out`, then `*d_seq` incremented for the host.
/// n_tok sub-groups of one work-group.  False, nothing launched, where the shape or n_tok x 32 work-items past the
/// device's work-group do not fit: the caller routes and rings apart.
bool router_publish(const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert, int k, const float* x,
                    int64_t n_x, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream);
/// STRATA_ROUTE_RESIDENT (upstream 5df35dcb): swap non-resident tail-rank experts for resident ones within `margin`
/// logits (see router.cpp).  `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident
/// before, non-resident after.
/// STRATA_ROUTE_PRIOR=lambda (upstream 2de9ae06, opt-in, changes the output): edits the router's top 10 per token
/// (`ids`/`weights`, n_tok x 10, over `logits` n_tok x n_expert) towards the experts in VRAM (`res_layer[e] >= 0`): a
/// non-resident expert from the lowest rank up gives way to the best resident one whose logit + lambda beats it (the
/// set upstream's biased top 10 picks); weights renormalised over the new 10.  `lambda_d` is a device float read by the kernel (0: no change, the router's bits).  `stats` is nullptr or 4
/// device uint64: tokens, tokens with a swap, swaps, dropped probability x 1e6 (router.cpp).
void route_prior_top10(const float* logits, const int32_t* res_layer, const float* lambda_d, int32_t* ids,
                       float* weights, int n_tok, unsigned long long* stats, void* stream, int n_expert = 512);
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           float margin, int rank_lo, int rank_hi, unsigned long long* stats, void* stream,
                           int n_expert = 512);
}
