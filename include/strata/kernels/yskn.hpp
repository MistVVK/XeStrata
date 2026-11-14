// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/yskn.hpp - the products of a .yskn matrix on the GPU (src/kernels/xe/yskn.cpp): YASAKANI's
// trellis-coded weights (strata/artifact/yskn.hpp) read where they are multiplied, written anew in SYCL from its
// lnord.cu, a8_prologue.cuh and q35_kernels.cu.
//
// The decode (a window of 1..kYsknMaxRows rows): the rows' activations rotated (their signs, the 512-point Hadamard
// transform), rounded to FP16 and quantized to int8 with a scale a 512-group (yskn_act); each output row's codes
// decoded through the 4,096-pair int8 table in local memory and multiplied with dp4a, a 512-group's int32 dot added
// in int64 through its fixed-point scale (yskn_sum.hpp), the columns divided among `n_workers` work-items whose int64
// partial sums go to a scratch (yskn_gemv); the sums added, scaled, rotated back over the rows and written in the
// engine's row order (yskn_finish).  Integer sums: a row's result is the same bits whatever the window's rows and the
// work's division.  No sub-group operation: a lane is a row, and the device order (artifact::device_order) makes a
// sub-group of any width read consecutive 16 bytes.
//
// The prompt: the matrix decoded to FP16 rows in the rotated basis (yskn_dequant_f16) for prefill::Gemm's FP16 product
// (which picks the GPU's matrix engines or dp4a by what the device reports), the input rotated in FP16
// (yskn_rotate_in_f16) and the output rotated back (yskn_rotate_out).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kYsknMaxRows = 8;   ///< the decode kernel's rows a launch at most (a longer window: several launches)

/// One .yskn matrix on the device
struct YsknDev {
    const uint32_t* words = nullptr;    ///< artifact::yskn::device_order's words
    const uint16_t* lut8 = nullptr;     ///< 4,096 int8 pairs
    const uint16_t* lut16 = nullptr;    ///< 4,096 FP16 pairs (the prompt's dequantization)
    const float* tau = nullptr;         ///< [rows]: the row's scale times the table's int8 step
    const uint16_t* scale16 = nullptr;  ///< [rows]: the row's FP16 scale (the prompt's dequantization)
    const int8_t* sign_in = nullptr;    ///< [cols]
    const int8_t* sign_out = nullptr;   ///< [rows], null: no output rotation (the head)
    /// [cols]: the input column c reads the engine's column perm_in[c] (null: c); [rows]: the matrix's row r is the
    /// engine's row perm_out[r] (null: r) - the GDN's V heads, grouped in the file and tiled in the engine
    const int32_t* perm_in = nullptr;
    const int32_t* perm_out = nullptr;
    int rows = 0;          ///< physical rows (a multiple of 64; of 512 with an output rotation)
    int rows_used = 0;     ///< the matrix's rows (the rest is padding, never written)
    int cols = 0;          ///< a multiple of 512
    int bits = 6;          ///< K: bits a step, 4..8
    int n_workers = 1;     ///< the work-items a row's columns are divided among (a divisor of cols / 512)
};

/// The int8 activations of T rows of x (row stride ldx, cols values read through perm_in): q [T][cols], alpha
/// [T][cols / 512]
void yskn_act(const YsknDev& w, const float* x, int64_t ldx, int T, int8_t* q, float* alpha, void* stream);
/// The int64 partial sums of T <= kYsknMaxRows rows: part [n_workers][T][rows]
void yskn_gemv(const YsknDev& w, const int8_t* q, const float* alpha, int T, int64_t* part, void* stream);
/// y [T] (row stride ldy) at the engine's rows: the partial sums added, times max alpha * 2^-24 and tau, the output
/// rotation; the rows past rows_used are not written
void yskn_finish(const YsknDev& w, const int64_t* part, const float* alpha, int T, float* y, int64_t ldy, void* stream);
/// The three above, T rows in launches of kYsknMaxRows; the scratch: yskn_scratch_bytes(w, T)
void yskn_proj(const YsknDev& w, const float* x, int64_t ldx, int T, float* y, int64_t ldy, void* scratch,
               void* stream);
size_t yskn_scratch_bytes(const YsknDev& w, int T);

/// The matrix's FP16 rows [rows][cols] in the rotated basis (FP16 of lut16 times the row's scale)
void yskn_dequant_f16(const YsknDev& w, uint16_t* out, void* stream);
/// x16 [T] (row stride ldx, through perm_in) rotated into FP16 rows [T][cols]
void yskn_rotate_in_f16(const YsknDev& w, const uint16_t* x16, int64_t ldx, int T, uint16_t* out, void* stream);
/// The product's rows y_rot [T][rows] (FP32, in the rotated basis) rotated back into y [T] (row stride ldy) at the
/// engine's rows
void yskn_rotate_out(const YsknDev& w, const float* y_rot, int T, float* y, int64_t ldy, void* stream);

}  // namespace strata::kernels
