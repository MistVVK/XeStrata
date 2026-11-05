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
/// The ggml types whose weights these products decode (the i-quants but IQ1_S, Q2_0, Q8_0, Q5_0, Q2_K to Q6_K).
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
sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows, bool sums, bool dup = false);
/// The rows iq_mmq_quantize(x, src, ...) writes for `tokens` x `k_used` rows, written token by token: token t's row of
/// x is quantized once and its blocks stored at rows slot[t * k_used + k] (`slot` the inverse of `src`): the same bytes,
/// a k_used-th of the quantizations (upstream c675f5b7, llama.cpp's quantize_scatter_mmq_q8_1).
sycl::event iq_mmq_quantize_scatter(sycl::queue& q, const float* x, const int32_t* slot, void* xq, int64_t cols,
                                    int64_t ld, int64_t tokens, int k_used, bool sums, bool dup = false);

/// Whether iq_mmq_q8_rows runs on `q`'s device: an NVIDIA GPU whose device code here was compiled for sm_80 or later
/// (asked of the device once), with the local memory and work-group size its form takes.  STRATA_NO_XMX=1 says no.
bool iq_mmq_q8_rows_usable(sycl::queue& q);
/// The dense product of Q8_0 weights in kNativeQ8Rows' layout (native_q8_0_to_rows: each row's K int8 values, then its
/// K / 32 fp16 scales; 16-byte aligned) on NVIDIA's int8 tensor cores' mma.sync: Y[T, N] (row stride ldy) = X[T, K] .
/// W[N, K]^T, xq iq_mmq_quantize's T rows (without the sums).  N a multiple of 128, K of 256 (every row 16-byte
/// aligned).  Each block's integer sum scaled by both scales into FP32, as iq_mmq_grouped's products.
sycl::event iq_mmq_q8_rows(sycl::queue& q, const void* xq, const void* w, float* y, int64_t T, int64_t N, int64_t K,
                           int64_t ldy);

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

}  // namespace strata::kernels
