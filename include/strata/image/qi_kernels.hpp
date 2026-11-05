// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_kernels.hpp - the image engine's element kernels for Qwen-Image-2.1's DiT.
//
// The DiT runs on one sequence of T tokens: the n_text text tokens, then the image tokens.  Each block scales its
// two LayerNorms and gates its two residual additions by the time modulation, one row for the image tokens (the
// step's time) and one for the text tokens (time 0): `mod` is [2][C], row 0 the image's, row 1 the text's.
// Activations are FP32 [T, C] row-major; the products' inputs FP16 bits.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::image {

/// y16[t] = LayerNorm(x[t]) * (1 + mod[t < n_text][...]) (no affine, eps 1e-6), FP16.
void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream);

/// A W8A8 product's output left as its int32 sums (their bits in the FP32 array): value [t][n] = sum * sx[t] * sw[n].
struct QiI8 {
    const float *sx = nullptr, *sw = nullptr;
};

/// x[t] += h[t] * tanh(mod[t < n_text][...]); with i8, h holds a W8A8 product's sums.
void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream,
                 const QiI8* i8 = nullptr);

/// For each token and head: q and k RMS-normed over the head (weights wq, wk; eps 1e-6), then rotated pairwise
/// (2j, 2j + 1) by cos[t][j], sin[t][j] (rope tables [T][head_dim / 2]); written as FP16 q16, k16, and v as v16, all
/// [T][heads * head_dim].  With i8 (three: q's, k's and v's), q, k and v hold W8A8 products' sums.
void qi_qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                       const float* cos, const float* sin, int64_t T, int64_t heads, int64_t head_dim,
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream, const QiI8* i8 = nullptr);

/// out[t, h, :] = softmax(q . k^T / sqrt(head_dim)) . v over the rows token t sees: a prefix token (t < n_text) the
/// tokens up to itself, or with key_lim the first key_lim[t] (never falling from row to row: an edit's reference
/// image sees all of itself), an image token every token.  FP32 accumulation; the plain form, every GPU.
void qi_attention_plain(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, int64_t n_text, float* out, void* stream, const int32_t* key_lim = nullptr);

/// The rows q16, k16 and v16 must hold for qi_attention: T rounded up to a multiple of 128, the rows past T zero.
constexpr int64_t qi_attn_rows(int64_t T) { return (T + 127) / 128 * 128; }

/// The same attention on the GPU's matrix engines where it has them (src/image/qi_attn.cpp), FP16 out16
/// [T - first][heads * head_dim]: the query rows from `first` on (the rows before it, a cached prefix's, are not
/// asked); else qi_attention_plain through `scratch` (FP32 [T][heads * head_dim]).  `int8`: the rows that see every key
/// may take the int8 kernel where it is the faster (qi_attention_i8: a model whose images it keeps, Qwen-Image's; not
/// Mage-Flow's, whose image stream's outlier channels int8's row scales lose).  `spans`: prefix rows [a, b) that see
/// keys 0 .. b - 1 and no others (an edit's references, as key_lim says on the GPU): they take the image rows' kernels,
/// as rows past a context of b keys.
void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, float* scratch, void* stream,
                  const int32_t* key_lim = nullptr, int64_t first = 0, bool int8 = false,
                  const std::vector<std::pair<int64_t, int64_t>>* spans = nullptr);

/// The path qi_attention takes on the current GPU, for the report.
const char* qi_attention_path();

/// The attention of query rows q0 .. q1 - 1 over all T keys (no mask), FP16 out16 [q1 - q0][heads * head_dim] (the
/// layout of qi_attention's), through a GPU maker's kernel where the build has one for the GPU (src/image/qi_sdpa.cpp:
/// oneDNN's SDPA on Intel GPUs); false where none takes it.  `causal`: each query row t reads keys 0 .. t only (q0 0,
/// q1 T: the text encoder's prompt), so far through aotriton on AMD GPUs alone.
bool qi_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1, int64_t T,
             int64_t heads, int64_t head_dim, uint16_t* out16, void* stream, bool causal = false);
/// qi_sdpa can take a causal attention on the stream's GPU (a kernel for it in the build and the GPU's runtime).
bool qi_sdpa_causal(void* stream);

/// qi_attention's int8 kernel (src/image/qi_attn.cpp) alone: query rows r0 .. T - 1, which see all T keys (no mask), to
/// out16 [rows from `first`][heads * 128]; false where it is not built for the GPU or its buffers do not fit.
/// qi_attention takes it, where it is the faster, for Qwen-Image's image rows (STRATA_IMAGE_ATTN_I8).
bool qi_attention_i8(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                     int64_t r0, int64_t first, uint16_t* out16, void* stream);
/// The keys of qi_attention_i8's chunks: P in int8 at 127 steps of a row's largest in each chunk (from key 0).
int qi_attention_i8_chunk();
/// The int8 kernel over a prefix of keys made once and read by many attentions (an edit's references, each block's):
/// qi_attention_i8_prefix_bytes of device memory a prefix of P keys takes; qi_attention_i8_prefix makes it from their
/// k16 and v16 ([P][heads * 128]); qi_attention_i8_prefixed is the attention of T rows (q16, k16, v16 [T][heads * 128])
/// over the prefix's P keys and their own T, to out16 [T][heads * 128], the prefix's V taken at v_from[h] / v_to[h]
/// (device [heads]) of its values.  false where the kernel is not built for the GPU or its buffers do not fit.
uint64_t qi_attention_i8_prefix_bytes(int64_t P, int64_t heads);
bool qi_attention_i8_prefix(const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* prefix,
                            void* stream);
bool qi_attention_i8_prefixed(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                              int64_t heads, const void* prefix, int64_t P, const float* v_from, const float* v_to,
                              uint16_t* out16, void* stream);
/// qi_attention takes the int8 kernel for `rows` rows of `keys` keys on the stream's GPU (as it times it, once a shape
/// in steps of 1024; STRATA_IMAGE_ATTN_I8 forces it): false where the GPU has no int8 kernel or matrix engines.
bool qi_attention_i8_preferred(int64_t rows, int64_t keys, int64_t heads, void* stream);

/// The text encoder's q and k ([T][q_heads * head_dim], [T][kv_heads * head_dim]): each head RMS-normed (wq, wk; eps
/// 1e-6), then rotated as NEOX does (pair (j, j + head_dim / 2) by a position times theta^(-2j / head_dim)); FP16 out,
/// v as v16.  The position: t, or with pos3 (three [T] rows: time, row, column) Qwen3-VL's interleaved M-RoPE, pair j
/// by the row when j % 3 == 1 and j < 3 * sec_h, by the column when j % 3 == 2 and j < 3 * sec_w, else by the time.
void qi_text_qk_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk, int64_t T,
                       int64_t q_heads, int64_t kv_heads, int64_t head_dim, float theta, uint16_t* q16, uint16_t* k16,
                       uint16_t* v16, void* stream, const int32_t* pos3 = nullptr, int sec_h = 0, int sec_w = 0);

/// Causal attention with grouped K/V heads (query head h reads K/V head h / (q_heads / kv_heads)): out [T][q_heads *
/// head_dim] FP32, the plain form (the text encoder's few tokens).
void qi_attention_causal(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t q_heads,
                         int64_t kv_heads, int64_t head_dim, float* out, void* stream);
/// Grouped K/V heads [T][kv_heads * head_dim] (FP16) repeated to one a query head, out [T][q_heads * head_dim] (query
/// head h's the K/V head h / (q_heads / kv_heads)): a grouped attention as qi_sdpa's plain one.
void qi_repeat_kv16(const uint16_t* src, int64_t T, int64_t q_heads, int64_t kv_heads, int64_t head_dim, uint16_t* out,
                    void* stream);


/// The activations of an int8 product (kernels::iq_mmq_quantize's layout, without the sums): rows x cols (a multiple of
/// 32) of x16 as int8 with a scale for each 32.
void qi_quantize_q8(const uint16_t* x16, int64_t rows, int64_t cols, void* xq, void* stream);

/// The group a W8A8 product's operands are rotated in (qi_rows_i8): K must be a multiple of it.
constexpr int64_t kQiRotate = 256;

/// The operands of a W8A8 product (prefill::Gemm::i8): rows of K FP16 (or with bf16 BF16) values (K up to 32768 on a
/// GPU of 1024-item work-groups), each group of kQiRotate times the normalized Hadamard matrix (orthogonal, so the
/// product of two rotated operands is that of the originals; it spreads a channel's outliers over the group), then as
/// int8 with a scale for the row: q [rows][K], s [rows], times rs [rows] where given (rows the caller scaled down).
/// The weights' rows and the activations' tokens alike.
void qi_rows_i8(const uint16_t* src, bool bf16, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs,
                void* stream);

/// qi_rows_i8 of a matrix of GGUF blocks (`type`: Q4_K, Q5_K or Q6_K; K a multiple of 256) straight from the blocks,
/// the int8 rows qi_rows_i8 makes of kernels::dequant_f16's output; false for another type (nothing is written).
bool qi_rows_i8_blocks(int type, const void* blocks, int64_t rows, int64_t K, int8_t* q, float* s, void* stream);

/// qi_rows_i8 of the SwiGLU's values (prefill::swiglu_pair's), from the gate_up product's int32 sums gu [T][2 inter]
/// (scaled by g): the down product's int8 input in one pass.
void qi_swiglu_i8(const int32_t* gu, const QiI8& g, int64_t T, int64_t inter, int8_t* q, float* s, void* stream);

/// y [T][N] (row stride ldy) holds a W8A8 product's int32 sums (their bits): y = sum * sx[t] * sw[n] + bias[n], in
/// place (bias null: none).  col[n] (null: 1) is a DoRA's factor of the weight's row n: the product's part of y only,
/// the bias not scaled.
void qi_i8_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* sx, const float* sw, const float* bias,
                 const float* col, void* stream);

/// y [T][N] (row stride ldy) *= col[n], in place: a dense product's rows, scaled where a DoRA's c is not 1 (col is
/// the weight's rows' factors, y's rows the weight's row0 on).
void qi_rows_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* col, void* stream);


/// y16 = FP16(GELU(x)), GELU's tanh form, n elements.
void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// x = SiLU(x) in place, and y16 = FP16 of it (y16 may be null), n elements.
void qi_silu(float* x, int64_t n, uint16_t* y16, void* stream);

/// tokens16[p][c] = FP16(latent[c][p]), p < HW, c < C: a latent [C][H][W] as the image's tokens.
void qi_latent_tokens16(const float* latent, int64_t C, int64_t HW, uint16_t* tokens16, void* stream);

/// latent[c][p] = tokens[p][c]: the tokens' outputs [HW][C] back as a latent [C][H][W].
void qi_tokens_latent(const float* tokens, int64_t C, int64_t HW, float* latent, void* stream);

/// x += a * v, n elements.
void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream);

/// An edit with a mask, after a sampler's step to time s: x [C][HW] where m [HW] is 0 (the part kept) becomes the
/// base image's latent z0 at that time, (1 - s) z0 + s noise (the flow's path from the same noise); where m is 1 it
/// stays the DiT's.  `tokens`: x, z0 and noise are [HW][C] instead (Mage-Flow's latent tokens).
void qi_inpaint_blend(float* x, const float* z0, const float* noise, const float* m, float s, int64_t C, int64_t HW,
                      void* stream, bool tokens = false);
/// x = (1 - s) z0 + s noise over n values: the base image's latent at time s on the flow's path (img2img's start).
void qi_noise_mix(float* x, const float* z0, const float* noise, float s, int64_t n, void* stream);

// --- the vision encoder (Qwen3-VL's ViT, src/image/qi_vision.cpp)

/// y[t, :] += b, T rows of N.
void qi_bias_add(float* y, const float* b, int64_t T, int64_t N, void* stream);

/// y16 = LayerNorm(x) * w + b per row of C (eps), FP16 out.
void qi_layer_norm16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps, uint16_t* y16,
                     void* stream);

/// y16 = GELU(x), the exact (erf) form, n elements.
void qi_gelu_erf16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// qkv [T][3 * heads * hd] (biased) -> q16, k16, v16 [rows][heads][128]: each head's hd channels, the rest zero (the
/// attention takes 128); q and k rotated in 2D (channel j paired with j + hd / 2; the first hd / 4 pairs by the
/// token's row yx[2t], the next by its column yx[2t + 1], each at theta^(-4j / hd)); q times q_scale.
void qi_vit_qkv_rope16(const float* qkv, const int32_t* yx, int64_t T, int64_t heads, int64_t hd, float theta,
                       float q_scale, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);

/// out16 [T][heads * hd] = the first hd channels of each head of in16 [T][heads][128].
void qi_unpad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, uint16_t* out16, void* stream);

}  // namespace strata::image
