// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/dflash2.hpp - the kernels of the DFlash2 drafter (core/dflash2.hpp), for every engine.
//
// The drafter's matrices run as tiled Q4_0 / Q8_0 through df2_gemm on planar int8 activations; what is here is
// everything between them: the grouped dynamic 2-tap convolution, the per-head RMS norm + rotary rotation, the
// windowed non-causal attention over a K/V ring, the top-16 over the head's logits, the selector's left-to-right
// walk, and the verify window's / prompt's tap (the layer's completed output, the mean of the hyper-connection
// streams).  All launches are asynchronous on the given stream (null: the runtime's compute queue), allocate
// nothing and are graph-capturable.  Layouts are row-major f32 unless noted.  No device FP64: the rope tables are
// computed on the host (df2_rope_table_cpu) - a device FP64 kernel slows every later kernel on an Arc Alchemist.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kDf2Rows = 8;            ///< block rows (the anchor + 7 masks) and the most context rows one update takes
constexpr int kDf2Kv = 8;              ///< K/V heads
constexpr int kDf2Gqa = 4;             ///< query heads per K/V head
constexpr int kDf2Hd = 128;            ///< head dim
constexpr int kDf2Chunk = 64;          ///< keys per attention work-group
constexpr int kDf2PartChunks = 33;     ///< the attention partials hold ceil((2047 + 8) / 64) chunks
constexpr int kDf2TopK = 16;
constexpr int kDf2TopBlocks = 64;      ///< partial top-k blocks per row (a block holds at most 4096 logits)

/// out[b][c] = sum_{o < 2, b - o >= 0} (base[o][c] + dyn[b * dyn_stride + o * groups + c / group]) * x[(b - o) * H + c]
/// (+ resid[b][c] when resid != null; out may equal resid).  x is [rows][H]; base is [2][H]; `dyn` points at this
/// convolution's [2][groups] slice of row 0.
void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out,
              int rows, int H, int group, void* stream);

/// table[r][0][i], table[r][1][i] = cos, sin of pos[r] * theta^(-2 i / 128) for i < 64, the angle reduced in double
/// first so a large position keeps its precision.  Host: table [rows][2][64].  Bitwise the device table up to sincosf.
void df2_rope_table_cpu(const int32_t* pos, int rows, float theta, float* table);

/// The block's q (q_heads x 128) and k (k_heads x 128) per row: RMS norm and rope with the table of the block's
/// positions -> qo, ko (row-major [row][head][128]).
void df2_norm_rope_qk(const float* q, const float* k, const float* gq, const float* gk, const float* table, float* qo,
                      float* ko, int rows, int q_heads, int k_heads, float eps, void* stream);

/// The context update's ring write: K (norm + rope) and V of `rows` rows into the head-major rings ([head][slot][128])
/// at slot pos[r] % slots.  With `tag` (int64 per slot) the slot is stamped (*epoch << 32 | pos); the attention (same
/// tag / epoch) ignores a slot whose stamp is not the position it expects.
void df2_kv_ring(const float* kin, const float* vin, const float* gk, const int32_t* pos, const float* table,
                 float* kring, float* vring, int slots, int rows, int heads, float eps, void* stream,
                 int64_t* tag = nullptr, const int32_t* epoch = nullptr);

/// Attention of the block's 8 query rows (q: [8][32][128], already normed and rotated) over the context ring
/// (kring, vring: head-major [8][ring_slots][128]) and the block's own keys (kblk, vblk: [8][8][128]).  Row i at
/// position s + i sees context position j when s + i - j < window (j in [max(0, s - window + 1), s - 1], slot
/// j % ring_slots) and every block key.  `s` is read from *anchor_pos on the device.  Partials in `part`
/// ([kDf2PartChunks][8][32][130] floats: max, sum, 128 outputs) are combined and the 8 output rows (4096 wide)
/// written planar-quantized to xq / xd.
void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk,
                   const int32_t* anchor_pos, int ring_slots, int window, float* part, void* xq, float* xd,
                   void* stream, const int64_t* tag = nullptr, const int32_t* epoch = nullptr);
/// The same attention, returning eight FP32 rows [8][4096] instead of quantizing them. The output can feed
/// a weight format with its own activation quantizer. The rings, block keys and partials have the same layout.
void df2_attention_f32(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk,
                       const int32_t* anchor_pos, int ring_slots, int window, float* part, float* out, void* stream,
                       const int64_t* tag = nullptr, const int32_t* epoch = nullptr);
/// 0: XESTRATA_DF2_ATTN_256 (auto: timed on the first window), 1: the 8-query-group form, 2: the 256-work-item
/// form (the tests run both; the engine leaves 0).
void df2_attention_variant(int v);

/// The drafter's GEMM: up to 3 tiled weight matrices that share the input, times the planar activations.  A matrix
/// is TILED (df2_q4_tile / df2_q8_tile make it from ggml blocks): the int8 / nibble plane then the fp16 scale plane.
/// Activations: xq int8 [col][n_in], xd float [col][n_in / 32].  y (matrix i) is [ncols][n_out].  n_in a multiple of
/// 512, n_out of 16, ncols 1..8.
struct Df2Mat { const void* w; float* y; int n_out; bool q4 = false; };
void df2_gemm(const Df2Mat* mats, int n, const void* xq, const float* xd, int n_in, int ncols, void* stream);

/// Planar quantization of rows x[ncols][K]: per 32-value block d = amax / 127, q = round(x / d).
void df2_quant_rows(const float* x, int K, int ncols, void* xq, float* xd, void* stream);
/// RMS norm (weights gamma) of `rows` rows of N, written to xn (optional) and quantized.
void df2_rms_quant(const float* h, const float* gamma, float* xn, int rows, int N, float eps, void* xq, float* xd,
                   void* stream);
/// df2_conv without residual, then quantized.
void df2_conv_quant(const float* x, const float* dyn, int dyn_stride, const float* base, int rows, int H, int group,
                    void* xq, float* xd, void* stream);
/// silu(gate) * up of `rows` rows of I, quantized.
void df2_swiglu_quant(const float* gate, const float* up, int rows, int I, void* xq, float* xd, void* stream);
/// h += conv(x) (the finish convolution with the residual), then the RMS norm of the new h (weights gamma) -> xn
/// (optional) and the planar quantization.  rows <= 8, H <= 4096 (H a multiple of 32).
void df2_conv_rms_quant(const float* x, const float* dyn, int dyn_stride, const float* base, float* h,
                        const float* gamma, float* xn, int rows, int H, int group, float eps, void* xq, float* xd,
                        void* stream);

/// ggml Q8_0 blocks (34 bytes) of a matrix with rows of n_in -> the tiled int8 plane and the tiled fp16 scale plane
/// (count = rows * n_in / 32 blocks).
void df2_q8_tile(const void* blocks, void* qs, void* dpl, int n_in, int64_t count, void* stream);
/// ggml Q4_0 blocks (18 bytes) -> the tiled nibble plane (n_out * n_in / 2 bytes) and the tiled fp16 scale plane.
void df2_q4_tile(const void* blocks, void* qs, void* dpl, int n_in, int64_t count, void* stream);

/// Top-16 per row of logits [rows][n] (value-descending; ties to the lower index).  idx_map (optional) maps a column
/// to a token id.  scratch: rows * kDf2TopBlocks * 16 * 2 words.  Outputs val[rows][16] and tok[rows][16].
/// n <= 4096 * kDf2TopBlocks.
void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok,
              void* stream);

/// The candidate selector's walk: for row i = 0..rows-1 the 16 candidates' scores are unary[i][j] + dot(succ[tok[i][j]],
/// pred[prev] * hp[i]) over `rank` values, the pick is the argmax, prev = the pick.  Writes out_tok[i] and out_prob[i]
/// (the softmax probability of the pick among the 16).  `prev0` is read from *anchor_tok on the device.  The codebooks
/// are bf16 rows of `rank`, or with `cb_q4` ggml Q4_0 rows in block order (rank / 32 blocks of 18 bytes a row, NOT
/// tiled like the matrices), dequantized to fp32 ((q - 8) * d) before the same dot.
void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb,
                int rank, int rows, const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream,
                bool cb_q4 = false);

/// The block's input rows from two embedding rows (a host-embedded target, the plain engine's): h[0] = anchor,
/// h[1..rows-1] = mask, each of N floats.
void df2_embed_rows(const float* anchor, const float* mask, float* h, int N, int rows, void* stream);

/// The verify window's tap: out[r][d] = mean over the hc streams c of (R[r][c][d] + bo[r][d] * 2 * sigmoid(inject[r][c]
/// / hc)) - the layer output with its pending FFN write applied, averaged over the streams.  R rows are R_stride floats
/// apart, bo rows bo_stride, inject rows inj_stride, out rows out_stride.
void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject,
                int64_t inj_stride, float* out, int64_t out_stride, int rows, int64_t n_embd, int hc, void* stream);

/// out[r][:n_embd] = the mean of the hc streams of row r of R ([rows][hc][n_embd]) written to out + r * out_stride
/// (a tap slot; the prompt path, whose GR write has already landed).
void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream);

}  // namespace strata::kernels