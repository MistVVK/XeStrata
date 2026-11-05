// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/gdn.hpp - the gated delta-net (GDN) layer's non-projection parts, P2.S2.
//
// Transcribed from `ref/gdn.py`, which is itself transcribed from `build_layer_attn_linear` and
// `build_delta_net_autoregressive`.  The kernels are native_gdn.hpp's recurrence and native_gdn_preprocess.hpp's
// conv, norms and gates (fused in fused_gdn.hpp); this header holds the geometry and the gate setting they share.
//
// Geometry of the real artifact (48 layers are GDN, 12 are QSA):
//
//     S = state_size = 128          head_k_dim == head_v_dim
//     h_k = 16  k/q heads           h_v = 48  v heads        (idx[h] = h % h_k)
//     C   = 2*128*16 + 128*48 = 10240 channels               d_conv = 4
//
// STATE LAYOUT.  This is the one place the kernels deliberately differ from the reference, and the reason is
// coalescing.  `ref/gdn.py` carries the state as (S, S, H_v) = (i, j, h); here it is **(S, H_v, S)** =
// (i, h, j), so that for a fixed i a warp walking consecutive j reads CONTIGUOUS memory.  The state is a
// runtime buffer, not a weight, so this costs nothing to choose - but it must be said, because the two
// layouts index the same three numbers and a mismatch is silent.
//
// WHAT MAKES THE RECURRENCE PARALLEL, and it is worth stating because it is not obvious:
//
//     dec      = exp(gate)                          (h_v,)
//     st       = state * dec
//     sk[h,j]  = sum_i st[i,j,h] * k[idx[h]][i]
//     d[h,j]   = (v[h,j] - sk[h,j]) * beta[h]
//     st[i,j,h] += k[idx[h]][i] * d[h,j]
//     o[h,j]   = sum_i st[i,j,h] * q[idx[h]][i]
//
// Every line touches ONLY the (j, h) column.  Nothing couples one column to another, so one thread can own a
// column for the whole step and no barrier is needed anywhere in the recurrence.  (The decay must still be
// applied BEFORE the rank-1 update - `ref/gdn.py` PROPERTY 5 pins that, and the test re-pins it here.)
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Geometry of one GDN layer.  The real artifact: S = 128, h_k = 16, h_v = 48.
struct GdnShapes {
    int64_t S = 0;    ///< head dim, and the state's contracted axis
    int64_t h_k = 0;  ///< key/query heads (q and k are (h_k, S))
    int64_t h_v = 0;  ///< value heads (v, z, o are (h_v, S)); h_v must be a multiple of h_k
};

/// The closing norm's gate for every GDN kernel that applies it (`native_gdn_out_norm`, `fused_gdn_step_norm` and
/// the verify-window kernels): false (the default) `sigmoid(z)`, qwen4exp's; true
/// `silu(z)`, qwen35moe's.  Set before the session's graphs are captured; captured graphs keep the gate they saw.
void gdn_set_silu_gate(bool silu);
bool gdn_silu_gate();

}  // namespace strata::kernels
