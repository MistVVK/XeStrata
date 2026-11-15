// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
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
/// The GGUF token-embedding types embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
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
/// A token-embedding table as the GPU reads it, through a device pointer (VRAM, or mapped pinned host memory): a
/// GGUF table (`type` one of embed_type_supported; row r at `rows` + r * `row_bytes`, `n_embd` a multiple of 256),
/// or (`type` kPacked) the packed planes of the canonical S2/S4/S8 tensor: `rows` the codes (`row_bytes` a row, low
/// bits first), `scales` and the optional `offsets` FP32 per group of `group_elems`, a value
/// `(code + code_bias) * scale + offset` with the product and the sum rounded separately (s_gemv_q8k_split's
/// decode).  `n_rows`: the vocabulary, for the caller's checks of host ids.
struct EmbedTable {
    static constexpr int kPacked = -1;
    int type = kPacked;
    const void* rows = nullptr;
    size_t row_bytes = 0;
    int64_t n_rows = 0, n_embd = 0;
    const float* scales = nullptr;
    const float* offsets = nullptr;
    int code_bits = 0, code_bias = 0, group_elems = 0;
};
/// The FP32 rows of `n_tok` tokens, out[t * n_embd + i]: the ids in device memory (`ids`, which a graph can
/// capture), or with `ids` null the one row `token`.  The ids are not checked on the device.
void embed_rows(const EmbedTable& table, const int32_t* ids, int64_t token, int64_t n_tok, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);
/// iq_dequant_gu_f16 and iq_dequant_f16 of the down matrix (n_embd x n_ff values of `d_type`) in one launch (upstream
/// 75109596, in its sycl/ port: the same values, one launch an expert fewer).
void iq_dequant_expert_f16(int gu_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* gu_dst,
                           int d_type, const void* down, uint16_t* d_dst, void* stream);

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
    int up_type = -1;                   // the up rows' type: gu_type, or another (a mixed layer, native_mixed_gu)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_row = 0;                  // bytes per up row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
    // APR (prism.hadamard): the down weights are Hadamard-folded, so native_expert_grouped rotates h before
    // quantizing it (hadamard_rows).  had_block 0 = none; d_signs is a DEVICE vector of n_ff signs or nullptr.
    int had_block = 0;
    const float* d_signs = nullptr;
};
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
/// `up_type` (-1: the gate's): the up rows' type where it differs.
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int up_type = -1) noexcept;
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int up_type = -1);

/// Time the two gate/up row layouts for this format and shape before a verify graph records them.
void native_expert_gu_pair_tune(const NativeExpertLayout& L, int type, int part, void* stream);

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
                           int64_t grid_groups = 0, int mode8_override = -1);
/// APR (prism.hadamard): y = H_block (signs * x) per block of `block` values of each of `rows` rows of `width`
/// (x and y may be the same rows); `signs` a DEVICE vector of `width` signs or nullptr, the float values those of
/// hadamard_rotate.  The f16 form rotates FP16 rows in place, computing in float.
void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream);
void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream);
/// The current engine device's copy of a host sign vector (hadamard_signs), made once and kept; nullptr for nullptr.
const float* hadamard_device_signs(const float* host, int width);
/// The same for BF16 experts (gate/up and down both BF16: qwen35moe's and qwen3moe's BF16 files), against the tokens'
/// FP32 rows `x` (n_embd floats a token) as llama.cpp's float products take a BF16 weight; the same scratch.
void native_expert_grouped_bf16(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const float* x,
                                void* scratch, float* out, void* stream, int64_t grid_groups = 0);

}  // namespace strata::kernels
