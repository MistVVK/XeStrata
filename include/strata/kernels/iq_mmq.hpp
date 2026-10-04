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

/// Whether `q`'s device runs these products: it reports a tile shape they are written for (int8 inputs, int32
/// accumulators) at its sub-group size, and its accumulators of int32 and of FP32 hold their elements in the same
/// places (checked on the device once).  STRATA_NO_XMX=1 says no.
bool iq_mmq_usable(sycl::queue& q);
/// The ggml types whose weights these products decode (the i-quants but IQ1_M, and Q2_0).
bool iq_mmq_type_ok(int ggml_type);
/// Bytes of a weight row of `cols` values in ggml_type (iq_mmq_type_ok).
size_t iq_mmq_row_bytes(int ggml_type, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values (a multiple of 128): the int8 values, then a float per 32.
size_t iq_mmq_act_bytes(int64_t rows, int64_t cols);
/// The activations: row i of xq from row ids[i] of x (row i when ids is null), `ld` floats a row of x.
sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows);

/// n experts, one launch: expert e's weights ([w_rows, w_cols] in ggml_type's blocks) at w + e * expert_bytes, its
/// activation rows bounds[e] .. bounds[e + 1] of xq (bounds on the device; xq holds total_rows rows), into dst rows
/// ids[row] (ld_dst floats apart).  max_rows: the most rows an expert has.  w_rows a multiple of 128, w_cols of 128.
sycl::event iq_mmq_grouped(sycl::queue& q, int ggml_type, const void* w, size_t expert_bytes, int64_t w_rows,
                           int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                           int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst);

}  // namespace strata::kernels
