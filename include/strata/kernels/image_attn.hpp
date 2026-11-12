// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/image_attn.hpp - the attention the image engine's models share and the row kernels its vision
// reader shares (definitions: src/kernels/xe/image_attn.cpp, src/kernels/xe/image_sdpa.cpp,
// src/kernels/xe/vision_rows.cpp): the DiTs' and the text encoders' attention, and the vision encoder's
// (src/core/vision.cpp) rows and attention.
//
// The names keep the image engine's (strata::image, qi_*): its twenty-odd call sites stay as they are, and a move of
// the code alone leaves every model's output bit for bit what it was.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::kernels {
struct FlashAttnI8Stats;
}

namespace strata::image {

/// The rows q16, k16 and v16 must hold for qi_attention: T rounded up to a multiple of 128, the rows past T zero.
constexpr int64_t qi_attn_rows(int64_t T) { return (T + 127) / 128 * 128; }

/// out16[t, h, :] = softmax(q . k^T / sqrt(head_dim)) . v over the rows token t sees: a prefix token (t < n_text) the
/// tokens up to itself, or with key_lim the first key_lim[t] (never falling from row to row: an edit's reference
/// image sees all of itself), an image token every token.  On the GPU's matrix engines where it has them
/// (src/kernels/xe/image_attn.cpp, flash_attn.hpp), FP16 out16 [T - first][heads * head_dim]: the query rows from
/// `first` on (the rows before it, a cached prefix's, are not asked); else the same kernel's FMA form.  `keys`: the
/// keys the rows read, where they are not the rows' own (a cross-attention, qi_attention_cross; 0: T).  `int8`: the
/// rows that see every key may take flash_attn's int8 path where it is the faster on this GPU (qi_attention_i8; every
/// image model passes it: on the RTX 4070 Anima's, Mage-Flow Turbo's and Qwen-Image's images stay as good, 2026-11-06).
/// `spans`: prefix rows [a, b) that see keys 0 .. b - 1 and no others (an edit's references, as key_lim says on the
/// GPU): they take the image rows' kernels, as rows past a context of b keys.  `i8_stats`: the keys' and values'
/// statistics the int8 path takes instead of its own pass over them (kernels::FlashAttnI8Stats).
void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, void* stream,
                  const int32_t* key_lim = nullptr, int64_t first = 0, bool int8 = false,
                  const std::vector<std::pair<int64_t, int64_t>>* spans = nullptr,
                  const kernels::FlashAttnI8Stats* i8_stats = nullptr, int64_t keys = 0);

/// A cross-attention: `Tq` query rows over `Tk` keys, every row seeing every key (no mask, no int8), FP16 out16
/// [Tq][heads * head_dim], through qi_attention's own choice of the GPU maker's kernel and the engine's own (timed
/// once a GPU and shape, as its image rows are).  q16's rows rounded up to qi_attn_rows(Tq) and zero past Tq, k16 and
/// v16 [Tk][heads * head_dim] rounded up to a multiple of 64 and zero past Tk: the engine's own kernel reads its keys
/// in whole chunks of 64.  head_dim 128, as qi_attention's.  Anima's cross-attention, generation and training alike.
void qi_attention_cross(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t Tq, int64_t Tk,
                        int64_t heads, int64_t head_dim, uint16_t* out16, void* stream);

/// The path qi_attention takes on the current GPU, for the report.
const char* qi_attention_path();

/// The attention of query rows q0 .. q1 - 1 over all T keys (no mask), FP16 out16 [q1 - q0][heads * head_dim] (the
/// layout of qi_attention's), through a GPU maker's kernel where the build has one for the GPU
/// (src/kernels/xe/image_sdpa.cpp: oneDNN's SDPA on Intel GPUs); false where none takes it.  `causal`: each query row t
/// reads keys 0 .. t only (q0 0, q1 T: the text encoder's prompt), so far through aotriton on AMD GPUs alone.  `batch`
/// attentions of the same shape one after another in each tensor (a ViT's windows; q0 0 and not causal where there are
/// more than one): one call of the library where it takes a batch (oneDNN, cuDNN), else one a batch.
bool qi_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1, int64_t T,
             int64_t heads, int64_t head_dim, uint16_t* out16, void* stream, bool causal = false, int64_t batch = 1);
/// qi_sdpa can take a causal attention on the stream's GPU (a kernel for it in the build and the GPU's runtime).
bool qi_sdpa_causal(void* stream);

/// qi_attention's int8 kernel (src/kernels/xe/image_attn.cpp) alone: query rows r0 .. T - 1, which see all T keys (no
/// mask), to out16 [rows from `first`][heads * 128]; false where it is not built for the GPU or its buffers do not fit.
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

/// y16 = FP16(GELU(x)), GELU's tanh form, n elements.
void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// x += a * v, n elements.
void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream);

/// y[t, :N] += b (+ add[t, :], null: none), T rows of N, a row ldy apart (0: N); b null: nothing.  The image engine's
/// one bias add (the DiTs', the VAEs', the vision readers').
void qi_bias_add(float* y, const float* b, int64_t T, int64_t N, void* stream, const float* add = nullptr,
                 int64_t ldy = 0);

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
/// out16 [T][heads][128] = in16 [T][heads * hd] widened to 128 a head, the channels past hd zero (the shared
/// attention's head dim).  `ld` the input row's stride (0: heads * hd).
void qi_pad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, int64_t ld, uint16_t* out16,
                    void* stream);

}  // namespace strata::image
