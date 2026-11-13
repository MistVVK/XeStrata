// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/iq_mmq.hpp - the prompt path's expert products on the int8 matrix engines, the weights read
// in their GGUF blocks (llama.cpp's MMQ, written anew on joint_matrix): the activations as int8 with a scale for each
// 32 values, the weights decoded in local memory to int8 with the scale of their 32 (or 16) values, and each such
// block's integer sum scaled by both into FP32.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Whether `q`'s device runs these products on its matrix engines: it reports a tile shape they are written for (int8
/// inputs, int32 accumulators: NVIDIA's 16 x 16 x 16 on 32 lanes, Intel's XMX 8 x 16 x 32 on 16 or 8 x 8 x 32 on 8) at
/// its sub-group size, or it is an RDNA3 or RDNA4 GPU (their WMMA); and its accumulators hold their elements where the
/// products read them (checked on the device once).  STRATA_NO_XMX=1 says no.
bool iq_mmq_usable(sycl::queue& q);
/// Whether the same products run on `q`'s device without matrix engines (dp4a; local memory and work-group size
/// permitting): iq_mmq_grouped takes that form where iq_mmq_usable says no.
bool iq_mmq_dp4a_usable(sycl::queue& q);
/// What runs these products on `q`'s device, for the start's report: "WMMA" (RDNA3, RDNA4), "joint_matrix" (a shape
/// the device reports: NVIDIA's tensor cores), "XMX" (Intel's shapes), "DP4a" (no matrix engines), or null (neither).
const char* iq_mmq_path(sycl::queue& q);
/// The fixed-point form of a type's products (canon_math.hpp's fx_*): kIqMmqFx + the ggml type, on canonical
/// activations (iq_mmq_quantize's canon); the same bits as the CPU's canonical rows (canon_expert.hpp) and the decode's
/// grouped experts, in any tile shape and on every GPU.
inline constexpr int kIqMmqFx = 1000;
/// The type iq_mmq_grouped takes for the fixed-point products of ggml_type (the gate/up's IQ2_XXS, IQ2_XS, IQ3_XXS,
/// IQ3_S and IQ2_S, the down's IQ4_NL), or 0 where it has none.
int iq_mmq_fx(int ggml_type);
/// The ggml types whose weights these products decode (the i-quants but IQ1_S, Q2_0, Q8_0, Q5_0, Q2_K to Q6_K), and
/// the fixed-point forms (iq_mmq_fx).
bool iq_mmq_type_ok(int ggml_type);
/// Bytes of a weight row of `cols` values in ggml_type (iq_mmq_type_ok).
size_t iq_mmq_row_bytes(int ggml_type, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values (a multiple of 32): the int8 values, then a scale per 32, then the
/// sum per 32 (the scale times the int8 values' sum).
size_t iq_mmq_act_bytes(int64_t rows, int64_t cols);
/// Whether products of ggml_type read the activations' sums (the K-quants with a min term).
bool iq_mmq_type_sums(int ggml_type);
/// Whether products of ggml_type read rows of twice the weights' columns, every activation block twice (Q2_K: its
/// per-16 mins as a second block of ones): iq_mmq_quantize's `dup`, and iq_mmq_grouped's w_cols doubled.
bool iq_mmq_type_dup(int ggml_type);
/// The activations: row i of xq from row ids[i] of x (row i when ids is null), `ld` floats a row of x; `sums`: their
/// block sums too (iq_mmq_type_sums), else that part is left unwritten; `dup`: each block twice (rows of 2 cols).
/// `canon` (the fixed-point products', iq_mmq_fx; no sums or dup): canon_math.hpp's int8 values with one scale for each
/// `canon` values (256: a gate/up's, cols a multiple of 256; 32: a down's), its fp16 value in each of its blocks - the
/// CPU's canonical rows' activations; 0: MMQ's own.
sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows, bool sums, bool dup = false, int canon = 0);
/// The rows iq_mmq_quantize(x, src, ...) writes for `tokens` x `k_used` rows, written token by token: token t's row of
/// x is quantized once and its blocks stored at rows slot[t * k_used + k] (`slot` the inverse of `src`): the same bytes,
/// a k_used-th of the quantizations (upstream c675f5b7, llama.cpp's quantize_scatter_mmq_q8_1).
sycl::event iq_mmq_quantize_scatter(sycl::queue& q, const float* x, const int32_t* slot, void* xq, int64_t cols,
                                    int64_t ld, int64_t tokens, int k_used, bool sums, bool dup = false,
                                    int canon = 0);

/// n experts, one launch: expert e's weights ([w_rows, w_cols] in ggml_type's blocks) at w + e * expert_bytes, its
/// activation rows bounds[e] .. bounds[e + 1] of xq (bounds on the device; xq holds total_rows rows), into dst rows
/// ids[row] (ld_dst floats apart).  max_rows: the most rows an expert has.  w_rows a multiple of 128, w_cols of 32
/// (and of the type's block).
/// wtab (on the device, n entries): expert e's weights at wtab[e] + woff instead, wherever each lies.
/// rows: the n experts' rows together (0: total_rows), which with max_rows picks the tile's shape.
sycl::event iq_mmq_grouped(sycl::queue& q, int ggml_type, const void* w, size_t expert_bytes, int64_t w_rows,
                           int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                           int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst,
                           const uint8_t* const* wtab = nullptr, size_t woff = 0, int64_t rows = 0);

/// Times the two forms Intel's XMX take these products in for products of ggml_type, w_rows x w_cols (the weights'
/// own columns, as iq_mmq_row_bytes takes them), n experts (1: a dense product) of 8 to 256 rows each, and keeps the
/// faster for each class of rows (iq_mmq_grouped then takes it; a product not timed takes the work-group form).  For
/// the engine's warming, before its first prompt: a few milliseconds a class, once a GPU and shape.  Nothing where
/// the GPU takes one form (not Intel's XMX).
void iq_mmq_choose_forms(sycl::queue& q, int ggml_type, int64_t w_rows, int64_t w_cols, int n = 16);

/// One expert's product of few rows (a dense model's short prompt) takes K in parts, a work-group each, the parts'
/// sums added in order after: the same sums from start to start.  They go into a buffer the queue reserves beforehand
/// (iq_mmq_split_reserve: `floats` at least, kept): a product it does not hold takes K whole.
bool iq_mmq_split_reserve(sycl::queue& q, size_t floats);
/// The parts' floats a product of `rows` rows (at most kIqMmqSplitRows) and `n_out` outputs may take.
constexpr int64_t kIqMmqSplitRows = 64, kIqMmqSplitMax = 2;
constexpr size_t iq_mmq_split_floats(int64_t rows, int64_t n_out) { return (size_t) (kIqMmqSplitMax * rows * n_out); }

}  // namespace strata::kernels
