// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include <cstdint>

namespace strata::kernels {

// Set before session construction/capture; existing CUDA graphs retain their
// selected kernels. Disabled by default. Integration chooses the call sites.
void native_qsa_set_enabled(bool enabled);
bool native_qsa_enabled();

// Pinned single-token CUDA F32 arithmetic from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. No allocation or synchronization;
// both functions require a nonnull caller-owned stream and four-byte alignment.
// Positive dimensions with n_cols*n_rows (or head_dim*n_head) <= INT_MAX.
// Finite inputs and finite intermediate sums/products are caller preconditions.

// Contiguous input/output [n_cols,n_rows], gamma[n_cols] broadcast over rows.
// Each row has independent RMS statistics. Exact output==input is supported;
// otherwise all three spans must be disjoint. Epsilon must be finite and >= 0;
// epsilon==0 additionally requires a nonzero sum of squares in every row.
// Pinned 256-thread (width<1024) or 1024-thread XOR reduction, (scale*x)*gamma.
void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream);

// The engines' one RMSNorm (the image engine's, the plain models' and Qwen3.8-Flash-Next's, prompt and decode): rows of
// n_cols, row r scaled by gamma's row r % groups (gamma null: by 1), into `output` (FP32; it may be `input`) or `out16`
// (FP16), one of them.  `pinned`: native_qsa_rms_norm_weighted's arithmetic to the bit; else the form the device runs
// fastest (its own work-group reduction where it has 16-wide sub-groups), the same for any number of rows.  A null
// stream waits.
void rms_norm_rows(const float* input, const float* gamma, int groups, float* output, uint16_t* out16, int n_cols,
                   int n_rows, float epsilon, void* stream, bool pinned = false);

// The same, writing the output also as q8_1 blocks (n_cols a multiple of 32) into x_q8_1, bit for bit
// native_quantize_q8_1 of the output: one launch for the norm and the quantization of the next products.
void native_qsa_rms_norm_weighted_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
                                       int n_cols, int n_rows, float epsilon, void* stream);

// The same per row with gamma[n_cols * groups]: row r is normalized on its own and scaled by gamma's slice
// r % groups (the MTP's hidden input, one row per hyper-connection stream: --mtp-hnorm stream).
void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream);

// attn/output [head_dim,n_head]; q_full [2*head_dim,n_head], each row is
// [query channels, gate channels]. Output is attn*sigmoid(second-half gate).
// Exact output==attn is supported; all other spans must be disjoint.  out16: the output as FP16 there instead
// (output unused; Qwen3.8-Flash-Next's prompt, whose o-projection reads FP16).
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream, uint16_t* out16 = nullptr);

} // namespace strata::kernels
