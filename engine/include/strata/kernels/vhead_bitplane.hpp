// include/strata/kernels/vhead_bitplane.hpp - Clean-room MIT exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Accelerates large-vocabulary (e.g., 248,320 x 2,560) greedy argmax projection for T = 1..16 tokens on sm_80+
// (Ampere RTX 3090 / A4000, Ada, Blackwell RTX 5090 / PRO 5000):
//   - Load-time lossless split of Q8_0 (8-bit -> hi4 + lo4 + per-row residual L2 norm rho) and Q6_K (6-bit -> hi4 +
//     lo2 + per-row residual L2 norm rho).
//   - Stage 1: streams ONLY the compact 4-bit hi_plane + scales using INT8 Tensor Cores (mma.m16n8k32 for Q8_0,
//     mma.m16n8k16 for Q6_K) across all T tokens in one pass, maintaining a monotone global lower bound
//     g_max_lo[t] = max_r (approx(r, t) - rho[r] * ||x_t||_2). Any row with approx(r, t) + rho[r] * ||x_t||_2 >=
//     g_max_lo[t] is appended to a compact candidate list (mathematically guaranteed to contain the exact argmax).
//   - Stage 2: filters candidates against the final g_max_lo[t] and evaluates the exact dot product ONLY on the
//     surviving rows (typically 1..16 rows per token), returning the bitwise-exact greedy argmax token ids.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct BitplaneHeadHandle;

/// True when the current CUDA device supports the exact bit-plane LM head (sm_80+, non-HIP).
bool vhead_bitplane_supported();

/// True when STRATA_VHEAD_BITPLANE is enabled (default ON on sm_80+ when supported; =0 disables).
bool vhead_bitplane_enabled();

/// Build a lossless bit-plane view from a device Q8_0 (ggml_type == 8) or Q6_K (ggml_type == 14) output.weight
/// matrix of shape [n_out, n_in] (n_in must be a multiple of 256). Returns nullptr if unsupported or OOM.
BitplaneHeadHandle* vhead_bitplane_create(int ggml_type, const void* d_weights, int n_in, int n_out);

/// Free all device buffers owned by `h`. Safe on nullptr.
void vhead_bitplane_destroy(BitplaneHeadHandle* h);

/// Approximate VRAM bytes held by `h` (for diagnostics / accounting).
size_t vhead_bitplane_bytes(const BitplaneHeadHandle* h);

/// Exact greedy argmax over all `n_out` vocabulary rows for `T` tokens (`1 <= T <= 16`).
/// `x_q8_1` holds `T` contiguous rows of `n_in / 32` `block_q8_1` structs (matching `native_quantize_q8_1`).
/// Writes exact argmax token ids to `out_tokens[0..T-1]` (lowest index wins exact ties, matching `argmax_rows`).
/// When `out_nonfinite != nullptr` (`nonfinite_stride > 0`), populates `out_nonfinite[t * nonfinite_stride + 0..stride-1]`
/// with non-zero if token `t` produced a non-finite norm or winning logit (0 otherwise), matching `logits_nonfinite_rows`.
/// Returns true if launched cleanly; false if fallback to full `native_mmvq` + `argmax_rows` is needed.
bool vhead_bitplane_argmax(BitplaneHeadHandle* h, const void* x_q8_1, int T, int32_t* out_tokens, void* stream,
                           uint32_t* out_nonfinite = nullptr, int nonfinite_stride = 0);

}  // namespace strata::kernels
