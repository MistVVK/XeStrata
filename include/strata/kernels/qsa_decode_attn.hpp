// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const uint8_t* k_tc = nullptr;      ///< trellis-coded rows (kv_tcq.hpp), same layout, head_dim * bits / 8 + 4 B
    const uint8_t* v_tc = nullptr;
    int tc_bits = 0;                    ///< 4 or 5 with k_tc
    const int32_t* page_table = nullptr;
};

/// A batch window's queries, each with its own slot's pools (upstream 7a9245b6c): query z reads rows.p[z].
/// `n == 0`: `pools` for every query.  At most kQsaRowsMax rows (a verify window's slots).
struct QsaAttnPoolsRows {
    QsaAttnPools p[kQsaRowsMax];
    int n = 0;
};

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);
/// The same with a batch window's own pools per query (upstream 7a9245b6c): row z of the window attends over the
/// cells of its slot's K/V, in one launch.  Every row's pools must be of the same format.
void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

/// The same kernels over EVERY cell of the cache, cells [0, step[kStepNKv]), with no selection: the full attention of
/// qwen35moe (Ornith-1.5: 16/2 heads of 256), qwen35 (Qwen3.5-9B: 16/4 heads of 256) and qwen3moe (llm-jp-4.1: 40/4
/// heads of 128).  `max_cells` is the cache's capacity and sizes the grid (`qsa_decode_attn_scratch_floats(max_cells,
/// s)` scratch floats per query); a grid that followed the context would be baked into a captured graph.  The shapes
/// the kernels are built for are (head_dim 256, 12, 8 or 4 query heads per KV head) and (128, 10, 6, 16 or 8); any other
/// throws, naming the shape.  `window` > 0 (a sliding-window layer, PLaMo 3's): the last `window` cells only, cells
/// [max(0, n_kv - window), n_kv).
void qsa_dense_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* step, int64_t max_cells,
                         const QsaShapes& s, float* scratch, float* attn, void* stream, int64_t window = 0);
void qsa_dense_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* steps, int64_t max_cells,
                          const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                          int64_t window = 0);
/// Times the fp16 score loop's forms (the cells a step takes) for shape `s` on the stream's GPU, once a GPU and shape,
/// over a dummy cache of up to 16K of `max_cells` cells, and keeps the fastest for qsa_dense_attn_step / _batch with an
/// fp16 cache.  Call before a graph that holds them is recorded (it allocates and synchronizes); without it they take
/// two cells a step.
void qsa_dense_attn_tune(const QsaShapes& s, int64_t max_cells, void* stream);

}  // namespace strata::kernels
