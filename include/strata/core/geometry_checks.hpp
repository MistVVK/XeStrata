// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/geometry_checks.hpp - what every compiled shape must satisfy.
//
// The Qwen3.8 engine's kernels are written for a family of shapes, not for one number each: some values are the same
// in every Qwen model of this layer design (the GDN state width, the attention head width, the conv width) and are
// held fixed here; the others have to be divisible or within a range the tiles, the local-memory plans and the
// register arrays were written for.  A geometry header that breaks one of these fails the build with the line below
// that names the field, instead of compiling a kernel that mis-runs.  Included by layout.hpp, after the selected
// header.
#pragma once

#include "strata/core/layout.hpp"

namespace strata::core::geometry_checks {

inline constexpr const ModelGeometry& g = kCompiledGeometry;

// ---- held fixed: the same in every published Qwen3-Next / 3.5 / 3.6 config and in Flash-Next; the kernels unroll them
static_assert(g.ssm_state_size == 128,
              "geometry: ssm.state_size must be 128 (the GDN kernels: 4 rows a lane, 128-thread blocks, 32x32 state tiles)");
static_assert(g.ssm_k_heads == 16, "geometry: ssm.group_count must be 16 (the GDN kernels' key heads)");
static_assert(g.ssm_d_conv == 4, "geometry: ssm.conv_kernel must be 4 (3 history taps and 4 weights are unrolled)");
static_assert(g.head_dim == 256,
              "geometry: attention.key_length must be 256 (the QSA decode: one work-item a dimension, 4 warps x 64)");
static_assert(g.qsa_interval == 4, "geometry: full_attention_interval must be 4");
static_assert(g.hc == 4, "geometry: hyper_connection.count must be 4 (the register and local arrays hold 4 streams)");
static_assert(g.idx_q_heads == 4 && g.idx_key_dim == 128,
              "geometry: the indexer must be 4 heads x 128 (one float4 a lane, one warp a head)");

// ---- ranges and divisibility the tiles were written for
static_assert(g.n_embd % 256 == 0 && g.n_embd >= 1024 && g.n_embd <= 4096,
              "geometry: embedding_length must be a multiple of 256 within 1024..4096 (K-quant rows, 8x512 partial sums)");
static_assert(g.n_ff % 64 == 0 && g.n_ff >= 256 && g.n_ff <= 1100,
              "geometry: expert_feed_forward_length must be a multiple of 64 within 256..1100 (the Q2 blobs, 64-wide act groups)");
static_assert(g.hc_lr % 64 == 0 && g.hc_lr / 8 >= 32 && g.hc_lr / 8 <= 64 && g.hc_lr % 8 == 0,
              "geometry: hyper_connection.low_rank must be a multiple of 64 with 32 <= low_rank/8 <= 64 (the fused_gr lane rule)");
static_assert(g.n_head_kv >= 1 && g.n_head % g.n_head_kv == 0 && g.n_head / g.n_head_kv <= 16,
              "geometry: attention.head_count / head_count_kv must be a whole number <= 16 (one M16 tile of query heads)");
static_assert(g.ssm_v_heads % g.ssm_k_heads == 0 && g.ssm_v_heads / g.ssm_k_heads >= 1 && g.ssm_v_heads / g.ssm_k_heads <= 4,
              "geometry: ssm.time_step_rank must be 1..4 times ssm.group_count");
static_assert(g.n_expert % 32 == 0 && g.n_expert >= 32 && g.n_expert <= 512,
              "geometry: expert_count must be a multiple of 32 up to 512 (the router: one lane per 32 experts)");
static_assert(g.n_layers % g.qsa_interval == 0 && g.n_layers >= g.qsa_interval,
              "geometry: block_count must be a multiple of full_attention_interval");

// ---- derived fields written out in the header must agree
static_assert(g.ssm_conv_channels == 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads,
              "geometry: ssm_conv_channels must be 2*state*k_heads + state*v_heads");
static_assert(g.ssm_value_dim == g.ssm_state_size * g.ssm_v_heads, "geometry: ssm_value_dim must be state*v_heads");
static_assert(g.hc_dim() == g.hc * g.n_embd, "geometry: hc_dim must be hc * n_embd");

#if defined(STRATA_GEOMETRY_CUSTOM)
// A shape other than Flash-Next needs every kernel that still writes a dimension as a literal converted first.  Those
// files hold Flash-Next's numbers today: src/prefill/prefill.cpp (N, HC, LR, K, C, ZV, HV and the head widths in its
// calls), src/prefill/kernels.cpp (blob_dequant's 2560x640 Q2 blob layout), src/kernels/xe/fused_gr.cpp (N, HC, LR and
// the tile plan built from them), src/kernels/xe/s2_expert_grouped.cpp (H, FF), src/kernels/xe/ple.cpp (the native PLE
// key's 2560x10240).  A build for another shape is refused here rather than compiled into kernels that compute
// Flash-Next's rows; delete this block with that work.
static_assert(g.n_embd == 2560 && g.n_layers == 48 && g.n_ff == 640 && g.n_expert == 512 && g.hc_lr == 320 &&
                  g.n_head == 24 && g.n_head_kv == 2 && g.ssm_v_heads == 48 && g.ssm_conv_channels == 10240 &&
                  g.ssm_value_dim == 6144,
              "geometry: a shape other than Flash-Next needs the kernels' compile-time constants converted first (see "
              "cmake/StrataGeometry.cmake and the files listed in geometry_checks.hpp)");
#endif

}  // namespace strata::core::geometry_checks
