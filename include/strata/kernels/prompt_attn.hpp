// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/prompt_attn.hpp - the plain models' prompt attention: causal, over every cell of the paged
// FP16 K/V pools, many queries a launch.
//
// The decode kernel the prompt used before (qsa_dense_attn_batch) reads every cell's K and V again for each query:
// at 16K tokens llm-jp-4.1's attention took 62 s of a 73 s prompt.  Here a work-group takes BQ queries of one KV head
// (BQ x the heads that share it: GQA's rows), walks the cells up to its last query 16 at a time through local memory,
// and keeps an online softmax per row, so a cell's K and V are read once per work-group.  Q.K^T and P.V run on the
// FP16 matrix engines (Q and the probabilities rounded to FP16): Intel's XMX, NVIDIA's tensor cores and RDNA3's and
// RDNA4's WMMA instructions through flash_attn.hpp's kernel; a GPU without them takes a kernel in FP32 FMAs (K and V
// enter from FP16).  STRATA_PROMPT_ATTN_MMA=0 takes the FMA kernel everywhere.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// attn[t, h, :] = softmax(q[t, h, :] . K[c, h / G, :] / sqrt(head_dim), c <= pos0 + t) . V for the T queries at
/// positions pos0 .. pos0 + T - 1, whose cells are already in the pools: q and attn [T, n_head, head_dim] FP32, the
/// pools `[page][kv_head][page_size][head_dim]` FP16 through `page_table`.  False, nothing launched, for a shape it is
/// not built for (head_dim 128 with 10 query heads a KV head, 256 with 4, 8 or 12): the caller keeps the decode kernel.
bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream, int64_t window = 0);

/// The same over `pools` (qsa_decode_attn.hpp): FP16 ones, INT8 codes with an FP16 scale per 64 values (--kv int8),
/// or Q4_0 (--kv q4_0) or INT8 K with Q4_0 V (--kv k8v4), rotated as kv_q4.hpp has them (the caller rotates the queries
/// and the output).  On the matrix engines where the device has them (Intel's XMX, NVIDIA's tensor cores from sm_70,
/// RDNA3's and RDNA4's WMMA) through the attention the image engine shares (flash_attn.hpp: head_dim 128 or 256, the
/// Q4_0 forms 256; any number of query heads a KV head); Q and the probabilities in FP16.  Else the FMA kernel above
/// (FP16 pools only).  False, nothing launched, when none takes the pools or the shape.  `window` > 0: a sliding
/// window, query t sees the cells from pos0 + t + 1 - window on only (both overloads).
bool prompt_attn(const float* q, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s, float* attn,
                 void* stream, int64_t window = 0);

/// Times the kernels the pools overload can take for shape `s` and pool format `pool_format` (0 FP16, 1 INT8, 2 Q4_0,
/// 3 K8V4, 4 trellis-coded: the matrix engines', the FP16 kernel, and the caller's decode kernel,
/// qsa_dense_attn_batch) on the stream's GPU, once a GPU, shape and format, over dummy chunks after 8K and up to 64K of
/// `max_cells` cells, and keeps the fastest of each for prompt_attn, which takes the one nearer a chunk's length; a
/// line to stderr says what it chose.  It allocates and synchronizes: call it before the cache takes its VRAM.
/// Untimed, prompt_attn takes the matrix engines' kernel where the GPU has one.  `timed` false: no timing, the maker's
/// kernel kept (the matrix engines' where it does not take a shape), its buffers taken now as the timing takes them.
void prompt_attn_tune(const QsaShapes& s, int pool_format, int64_t max_cells, void* stream, bool timed = true);

}  // namespace strata::kernels
