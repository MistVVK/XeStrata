// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/flash_attn.hpp - the attention on the GPU's FP16 matrix engines that the image engine's DiT
// (qi_attention), the plain models' prompt (prompt_attn) and Qwen3.8-Flash-Next's prompt (flash_attn_select) share:
// one entry, the keys and values read where they are (a dense [key][row] layout, or the K/V cache's pages) or gathered
// from the pools, the output in FP16 or FP32.
//
// A sub-group takes TM query rows of one head and walks that head's keys 64 at a time on its own: S = Q . K^T with
// Q's tiles in registers and K's read from global memory (the caches share them between the sub-groups of a
// work-group, which take neighbouring rows of the same head), the online softmax in local memory with a lazily moved
// reference (FlashAttention-3's), and O += P . V in the matrix engines' accumulators.  Intel's XMX (8 x 16 x 16 on 16
// lanes, 8 x 8 x 16 on 8), NVIDIA's tensor cores from sm_70 on (16 x 16 x 16 on 32) through joint_matrix, RDNA3's and
// RDNA4's WMMA instructions (amd_wmma.hpp); a GPU without them, or STRATA_NO_XMX=1, has none (flash_attn_path -1).
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

struct FlashAttnArgs {
    const uint16_t* q = nullptr;     ///< FP16 queries: row t, head h at q + t * q_ld + h * head_dim (rows past `rows`
    const float* q32 = nullptr;      ///< up to the sub-group's tile there, zero), or FP32 ones (q32, none read past)
    int64_t q_ld = 0;
    const uint16_t* k = nullptr;     ///< FP16 keys and values, dense (page_table null): key j of KV head g at
    const uint16_t* v = nullptr;     ///< k + j * kv_ld + g * head_dim; or paged: [page][n_kv][page_size][head_dim]
    int64_t kv_ld = 0;
    const int32_t* page_table = nullptr;   ///< a page's slot (page_size a multiple of 64 for k and v read in place)
    int64_t page_size = 0;
    /// The K/V cache's pools in another format (INT8 with an FP16 scale per 64 values, Q4_0, INT8 K with Q4_0 V;
    /// pools.page_table its table): a work-group gathers each chunk of cells into its local memory as FP16 for the
    /// rows (query, head) of one KV head it takes, any page_size.  pools_format -1: k and v (FP16) as above.
    QsaAttnPools pools;
    int pools_format = -1;
    /// A selection (Qwen3.8-Flash-Next's QSA, flash_attn_select): query t reads the cells sel_ids[t * sel_cap + j],
    /// j < sel_steps[t * kStepCount + kStepWidth], none masked but those whose page is -1; the pools in any format
    /// (pools_format unused), FP32 queries and output, rows 0 .. rows - 1.
    const int32_t* sel_ids = nullptr;
    const int32_t* sel_steps = nullptr;
    int64_t sel_cap = 0;
    int64_t n_head = 0, n_kv = 0, head_dim = 0;   ///< query heads, KV heads (query head h reads h / (n_head / n_kv))
    int64_t rows = 0;      ///< query rows; the keys are read in whole chunks of 64 up to `keys`, which must be there
    int64_t keys = 0;      ///< the keys the rows from n_causal on see (all of them)
    int64_t n_causal = 0;  ///< rows t below it see keys below key_lim[t], or below pos0 + t + 1 without key_lim
    int64_t pos0 = 0;
    const int32_t* key_lim = nullptr;   ///< ascending
    /// A sliding window (PLaMo 3's): rows t below n_causal see keys from pos0 + t + 1 - window on as well; 0: from
    /// key 0.  Not with key_lim.
    int64_t window = 0;
    int64_t first = 0, row_end = 0;     ///< the rows asked: first .. row_end - 1, at the output's row t - first
    void* out = nullptr;                ///< FP16 or (out_f32) FP32: row t, head h at
                                        ///< out + (t - first) * out_ld + h * head_dim
    int64_t out_ld = 0;
    bool out_f32 = false;
    float* lse = nullptr;               ///< non-null: each row's log-sum-exp of its scaled scores, row t, head h at
                                        ///< lse[(t - first) * n_head + h] (a training step's backward reads it)
    float scale = 0.0f;                 ///< the scores' scale (1 / sqrt(head_dim))
    bool fma = false;                   ///< the FMA form (FP32 FMAs) on a GPU with matrix engines too (timing it)
};

/// The matrix engines' form the device takes (0 Xe2's 8 x 16 x 16 on 16 lanes, 1 Alchemist's 8 x 8 x 16 on 8, 2
/// NVIDIA's 16 x 16 x 16 on 32, 3 RDNA3's and RDNA4's WMMA), or -1: none (no FP16 matrix engines, STRATA_NO_XMX=1).
/// The device is a `sycl::device` (void: the header stays free of SYCL's).
int flash_attn_path(const void* sycl_device);

/// The most sub-groups a work-group the device takes for `head_dim` (128 or 256), up to 8; 0 for none.
int flash_attn_nsg_most(const void* sycl_device, int64_t head_dim);

/// Launches the attention of `a` on `stream` (the runtime's queue) with `nsg` sub-groups a work-group (0: the most),
/// not waiting for it.  False, nothing launched, where no form takes the shape (head_dim 128 or 256; a paged layout's
/// page_size a multiple of 64 on Intel's and NVIDIA's matrix engines; without matrix engines FP16 K and V and 1, 2,
/// 4, 8 and 10 or 12 query heads a KV head).
bool flash_attn(const FlashAttnArgs& a, void* stream, int nsg = 0);

/// Choose the sub-group count by device and shape on substantial matrix-engine work.
bool flash_attn_tuned(const FlashAttnArgs& a, void* stream);

// The int8 path (flash_attn_i8.cpp, SageAttention's scheme): rows that see every key, head_dim 128, Q, K and V FP16
// [rows][heads * 128] (the rows past T up to a multiple of flash_attn_i8_chunk zero).  Q . K^T and P . V on the int8
// matrix instructions: RDNA3's and RDNA4's WMMA (comfy-kitchen's kernels where the build carries them, with their own
// statistics), NVIDIA's mma.sync from sm_80 on.

/// The int8 kernel `stream`'s GPU takes: 1 the WMMA one, 2 the mma.sync one, 0 none (the caller takes flash_attn).
int flash_attn_i8_kind(void* stream);
/// The keys of the int8 path's chunks: P in int8 at 127 steps of a row's largest in each chunk (from key 0).
int flash_attn_i8_chunk();
/// The statistics of the keys and values a caller may give the int8 path instead of their pass over K and V: V's
/// scales [heads * 128] (a column's largest |v| / 127, never 0), and K's mean over the keys [heads * 128], read where
/// kmean_ready and else made into it (any mean leaves the softmax as it is - the scores of a row all move by q . mean
/// - so one made at an earlier step serves: only the keys' int8 rounding asks it be near).
struct FlashAttnI8Stats {
    const float* sv = nullptr;
    float* kmean = nullptr;
    bool kmean_ready = false;
};
/// Query rows r0 .. T - 1 of T keys into out16 (its row 0 query row `first`); false where the GPU has no int8 kernel
/// or its buffers do not fit (nothing written).  A null stream waits.
bool flash_attn_i8(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                   int64_t r0, int64_t first, uint16_t* out16, void* stream, const FlashAttnI8Stats* st = nullptr);
/// The causal form (the LLMs' prompt): Tq query rows of `heads` heads, row t at position pos0 + t, against the Tk
/// cells of `pools` (pages of page_size cells, FP16, int8 or Q4_0; key j at position j) of kv_heads heads (query head h
/// reads KV head h / (heads / kv_heads)), each row seeing the keys up to its position; q16 and out16
/// [Tq][heads head_dim], FP16.  Its buffers are the stream's, kept for the next call.  False (nothing launched) where
/// no int8 kernel takes it here (the GPU, head_dim) or its buffers would cut into the GPU's reserve: the caller's
/// 16-bit path.
bool flash_attn_i8_causal(const uint16_t* q16, const QsaAttnPools& pools, int64_t page_size, int64_t Tq, int64_t Tk,
                          int64_t heads, int64_t kv_heads, int64_t head_dim, int64_t pos0, uint16_t* out16,
                          void* stream);
/// Takes the stream's buffers of flash_attn_i8_causal for Tq rows over Tk keys now (before an engine plans its memory
/// around them; false: not on this GPU, or past a 32nd of its memory), or gives them back (a caller that will not
/// take it).
bool flash_attn_i8_causal_prepare(int64_t Tq, int64_t Tk, int64_t heads, int64_t kv_heads, int64_t head_dim,
                                  void* stream);
void flash_attn_i8_causal_release(void* stream);
/// A prefix of P keys made once (flash_attn_i8_prefix_bytes of device memory, from their k16 and v16) that
/// flash_attn_i8_prefixed's T rows attend to after their own keys, its values at their scales times v_from / v_to (a
/// head's).
uint64_t flash_attn_i8_prefix_bytes(int64_t P, int64_t heads);
bool flash_attn_i8_prefix(const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* prefix,
                          void* stream);
bool flash_attn_i8_prefixed(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                            const void* prefix, int64_t P, const float* v_from, const float* v_to, uint16_t* out16,
                            void* stream);

// The selection form (formerly qsa_prompt_attn.hpp, perf-review D-1: the prompt path's QSA attention on the matrix
// engines):
//
// `qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel: FP32 dot products with a warp
// reduction per head and cell, 64-cell chunks whose partial sums go through global scratch to a second (merge)
// kernel. At 32K that is 21% of the prompt (5.6 s of 26 s on a 5070, Q2_0, int8 KV), and it is limited by
// instruction issue, not memory (~150 GB/s of logical reads, 3.4 TFLOP/s).
//
// This kernel keeps the same per-query selection (no masking, no union) and reads the same pools, but:
//   * one block per (query, KV head) walks all its selected cells in chunks (32 on CUDA, 16 on Xe) with an online
//     softmax: no split-K scratch and no merge kernel;
//   * q.k and p.v are FP16 matrix products with FP32 accumulation (m16n8k16 on CUDA, joint_matrix 8x16x16 on Xe)
//     over the 12 query heads of the KV head (+4 pad rows);
//   * the stored values enter exactly: int8 codes are exact in FP16 and their per-64 scales are applied in FP32
//     (to the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is; Q4_0's codes
//     minus 8 are exact in FP16 too, and its signed per-32 scales are applied the same way (upstream 778e1f6);
//   * q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: the result differs from
//     the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. K8V4 (INT8 K, Q4_0 V) is handled by the portable
// kernel only (NVIDIA's tensor cores; on Intel's XMX it ran slower than the FP32 kernel), Q4_0 KV by Intel's kernel
// only, and nothing on a device without the matrix engines.
//
/// flash_attn over a selection, with `qsa_decode_attn_batch`'s arguments minus the scratch.  Returns false (nothing
/// launched) when the device has no kernel for the pools (above) or the geometry is not 24 heads / 2 KV heads / 256:
/// the caller then uses the decode kernel.
///
/// 16 heads / 4 KV heads / 256 (SenseNova-Loop's QSA) take a form of their own: 4 heads a KV head leave 12 of the
/// tile's 16 rows empty, so a work-group takes 4 neighbouring queries, walks the union of their selections once and
/// masks each row to its own query's cells (FP16 pools, the portable kernel: RDNA3's and RDNA4's WMMA, NVIDIA's tensor
/// cores; not Intel's XMX).  It needs `scratch_bytes` of `scratch` for the unions: 5 * 4 * cap bytes and 4 more a 4
/// queries.  The same cells and weights as the decode kernel, in another summation order.
bool flash_attn_select(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                       const QsaShapes& s, float* attn, int64_t n_q, void* stream, void* scratch = nullptr,
                       size_t scratch_bytes = 0);

}  // namespace strata::kernels
