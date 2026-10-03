// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/main/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);

/// Whether iq_gemm_grouped_f16 takes weights of `ggml_type` in [N, K] matrices: N and K multiples of 128.  Q3_K is
/// not taken.
bool iq_gemm_grouped_ok(int ggml_type, int64_t N, int64_t K) noexcept;
/// The most experts one iq_gemm_grouped_f16 takes.
constexpr int kIqGemmGroup = 32;
/// The rows of X one work-group of iq_gemm_grouped_f16 takes: each one dequantizes its slice of the weights again, so
/// an expert with more rows than this has its weights decoded more than once.
constexpr int kIqGemmTileRows = 256;
/// xmx_gemm_grouped with the weights still in GGUF blocks: for expert e < G, rows [bounds[e], bounds[e+1]) of the
/// FP16 X (K values a row) times its [N, K] matrix at device address W[e] (W itself a host array), into the same
/// rows of Y (`ldy` floats apart).  With `interleave` (N / 2) the matrix is gate rows then up rows and Y's columns
/// alternate gate, up (iq_dequant_gu_f16's layout).  The weights are dequantized in local memory to the FP16 values
/// iq_dequant_f16 writes, so Y is that of dequantizing and multiplying with xmx_gemm_grouped (the same bits), without
/// the dequantized weights in memory.  `max_rows`: the most rows an expert has.  Needs XMX for FP16
/// (xmx_available).
void iq_gemm_grouped_f16(int ggml_type, const uint16_t* X, const void* const* W, int64_t interleave, float* Y,
                         int64_t ldy, const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K,
                         void* stream);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
};
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.  `grid_groups` rows of groups
/// are launched, each striding over the groups (0: cap_groups, a row per possible group); the results do not depend
/// on it.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0);

}  // namespace strata::kernels
