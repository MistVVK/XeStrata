// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/yskn.hpp - the products of a .yskn matrix on the GPU (src/kernels/xe/yskn.cpp): YASAKANI's
// trellis-coded weights (strata/artifact/yskn.hpp) read where they are multiplied, written anew in SYCL from its
// lnord.cu, a8_prologue.cuh and q35_kernels.cu.
//
// The decode (a window of 1..kWindowMaxT rows): the rows' activations rotated (their signs, the 512-point Hadamard
// transform), rounded to FP16 and quantized to int8 with a scale a 512-group (yskn_act); each output row's codes
// decoded through the 4,096-pair int8 table in local memory and multiplied, a 512-group's int32 dot added in int64
// through its fixed-point scale (yskn_sum.hpp), the columns divided among `n_workers` whose int64 partial sums go to
// a scratch; the sums added, scaled, rotated back over the rows and written in the engine's row order (yskn_finish).
// Two forms of the multiply, the same integer sums (the same bits): DP4a, a lane a row, up to kYsknMaxRows rows a
// launch (yskn_gemv); and the device's int8 matrix units, up to kWindowMaxT rows a launch (yskn_mma_proj,
// src/kernels/xe/yskn_mma.cpp: Intel's XMX, NVIDIA's tensor cores, RDNA's WMMA, the shape from the device's report).
// Which one: XESTRATA_YSKN_MMA (0 DP4a, 1 matrix), else the row boundary timed at the load (yskn_mma_tune); a device
// without int8 matrix units, or XESTRATA_NO_XMX=1, takes DP4a.  Integer sums: a row's result is the same bits whatever
// the window's rows, the work's division and the form.  The DP4a form has no sub-group operation: a lane is a row, and
// the device order (artifact::device_order) makes a sub-group of any width read consecutive 16 bytes.
//
// The prompt: the matrix decoded to FP16 rows in the rotated basis (yskn_dequant_f16) for prefill::Gemm's FP16 product
// (which picks the GPU's matrix engines or dp4a by what the device reports), the input rotated in FP16
// (yskn_rotate_in_f16) and the output rotated back (yskn_rotate_out).
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kYsknMaxRows = 8;   ///< the DP4a decode kernel's rows a launch at most (a longer window: several)
constexpr int kYsknMmaRows = kWindowMaxT;   ///< the matrix form's rows a launch (the widest window)

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
/// The DP4a form's int64 partial sums of T <= kYsknMaxRows rows: part [n_workers][T][rows]
void yskn_gemv(const YsknDev& w, const int8_t* q, const float* alpha, int T, int64_t* part, void* stream);
/// y [T] (row stride ldy, T <= kWindowMaxT) at the engine's rows: the partial sums added, times max alpha * 2^-24 and
/// tau, the output rotation; the rows past rows_used are not written
void yskn_finish(const YsknDev& w, const int64_t* part, const float* alpha, int T, float* y, int64_t ldy, void* stream);
/// The product of T rows: the activations, the multiply (the matrix form where yskn_mma_take says so, in launches of
/// kYsknMmaRows; else DP4a in launches of kYsknMaxRows), the finish; the scratch: yskn_scratch_bytes(w, T)
void yskn_proj(const YsknDev& w, const float* x, int64_t ldx, int T, float* y, int64_t ldy, void* scratch,
               void* stream);
/// The scratch's bytes: the activations and partial sums of up to kYsknMmaRows rows (either form reads them)
size_t yskn_scratch_bytes(const YsknDev& w, int T);

/// The scratch's parts (yskn_scratch_bytes): the int8 activations [kYsknMmaRows][cols], their scales
/// [kYsknMmaRows][cols / 512], the int64 partial sums [n_workers][T][rows] (T the launch's rows, as yskn_gemv writes)
struct YsknScratch {
    int8_t* xq;
    float* alpha;
    int64_t* part;
};
inline YsknScratch yskn_scratch_parts(const YsknDev& w, void* scratch) {
    const size_t rows = (size_t) kYsknMmaRows;
    const size_t qb = (rows * (size_t) w.cols + 255) / 256 * 256;
    const size_t ab = (rows * (size_t) (w.cols / 512) * 4 + 255) / 256 * 256;
    auto* b = static_cast<uint8_t*>(scratch);
    return {static_cast<int8_t*>(scratch), reinterpret_cast<float*>(b + qb), reinterpret_cast<int64_t*>(b + qb + ab)};
}

/// Whether the device runs the decode's matrix form (its int8 matrix units, the shape from its report; RDNA's WMMA):
/// XESTRATA_NO_XMX=1 says no
bool yskn_mma_usable(void* stream);
/// The matrix form's product of T <= kYsknMmaRows rows in one launch (yskn_mma_usable): the same scratch and the same
/// integer sums as yskn_proj's (the same bits as the DP4a form)
void yskn_mma_proj(const YsknDev& w, const float* x, int64_t ldx, int T, float* y, int64_t ldy, void* scratch,
                   void* stream);
/// Whether the product of T rows takes the matrix form: XESTRATA_YSKN_MMA (0 DP4a, 1 matrix) where the device runs it,
/// else T at or past the boundary yskn_mma_tune timed (before the tune: DP4a)
bool yskn_mma_take(int T, void* stream);
/// Times the row boundary between the two forms on `w` (the largest matrix the engine loads) and keeps it for the
/// device: the smallest window rows where the matrix form is the faster (5% or more), else none.  The load's warming,
/// before any graph: the scratch's activations zeroed (the sums are zero, the times are the forms').  Nothing where
/// the device does not run the matrix form, or XESTRATA_YSKN_MMA forces one
void yskn_mma_tune(const YsknDev& w, void* scratch, void* stream);

/// The matrix's FP16 rows [rows][cols] in the rotated basis (FP16 of lut16 times the row's scale)
void yskn_dequant_f16(const YsknDev& w, uint16_t* out, void* stream);
/// x16 [T] (row stride ldx, through perm_in) rotated into FP16 rows [T][cols]
void yskn_rotate_in_f16(const YsknDev& w, const uint16_t* x16, int64_t ldx, int T, uint16_t* out, void* stream);
/// The product's rows y_rot [T][rows] (FP32, in the rotated basis) rotated back into y [T] (row stride ldy) at the
/// engine's rows
void yskn_rotate_out(const YsknDev& w, const float* y_rot, int T, float* y, int64_t ldy, void* stream);

}  // namespace strata::kernels
