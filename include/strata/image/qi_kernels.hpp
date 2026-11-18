// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_kernels.hpp - the image engine's element kernels for Qwen-Image-2.1's DiT.
//
// The DiT runs on one sequence of T tokens: the n_text text tokens, then the image tokens.  Each block scales its
// two LayerNorms and gates its two residual additions by the time modulation, one row for the image tokens (the
// step's time) and one for the text tokens (time 0): `mod` is [2][C], row 0 the image's, row 1 the text's.
// Activations are FP32 [T, C] row-major; the products' inputs FP16 bits.
#pragma once

#include "strata/image/product_store.hpp"
#include "strata/kernels/image_attn.hpp"   // the attention and the
                                                             // vision reader's row kernels
#include "strata/kernels/rows_i8.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::kernels {
struct FlashAttnI8Stats;
}

namespace strata::prefill {
class Gemm;
}

namespace strata::image {

/// y16[t] = LayerNorm(x[t]) * (1 + mod[t < n_text][...]) (+ shift: the VAEs' modulation; no affine, eps 1e-6), FP16:
/// the image engine's one modulated LayerNorm.
void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream,
                 const float* shift = nullptr);

struct DitOut;

/// x[t] += h[t] * tanh(mod[t < n_text][...]); with i8 (a W8A8 product's DitOut), h holds its int32 sums (their bits in
/// the FP32 array: value [t][n] = sum * i8->sx[t] * i8->sw[n]), or its values as BF16 (i8->bf set).
void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream,
                 const DitOut* i8 = nullptr);

/// For each token and head: q and k RMS-normed over the head (weights wq, wk; eps 1e-6), then rotated pairwise
/// (2j, 2j + 1) by cos[t][j], sin[t][j] (rope tables [T][head_dim / 2]); written as FP16 q16, k16, and v as v16, all
/// [T][heads * head_dim].  With i8 (three: q's, k's and v's), q, k and v hold W8A8 products' sums.
void qi_qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                       const float* cos, const float* sin, int64_t T, int64_t heads, int64_t head_dim,
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream, const DitOut* i8 = nullptr);

/// A training step's attention (src/image/qi_attn_train.cpp), every row from n_causal on seeing every key and a row
/// below it the keys up to itself (Qwen-Image-2.1's text rows, the inference's prefix rule): q16, k16 and v16 are
/// [T][heads * head_dim] FP16, their rows rounded up to qi_attn_rows(T) and zero past T, as qi_attention's.  The
/// forward runs the shared attention kernel (kernels::flash_attn, the one inference takes) to out32 [T][heads *
/// head_dim] FP32 and lse [T][heads], each row's log-sum-exp of its scaled scores.  `keys`: the keys the rows read,
/// where they are not the rows' own (a cross-attention: k16 and v16 [keys][heads * head_dim], their rows rounded up
/// to qi_attn_rows(keys) and zero past keys, n_causal 0; 0: T).
void qi_attention_train(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, float* out32, float* lse, void* stream, int64_t n_causal = 0,
                        int64_t keys = 0);
/// Its backward: dQ, dK and dV [T][heads * head_dim] FP32 from dO (the same shape, FP32), the forward's out32 and lse,
/// through the shared products (prefill::Gemm) a head and a tile of query rows at a time, P made again from the
/// scores and lse.  dQ is written; dK and dV are written, or added into with `accumulate`.  Null dQ, dK or dV: not
/// taken.  `keys` as qi_attention_train's: dK and dV are [keys][heads * head_dim] then.  The fused backward
/// (kernels::flash_attn_bwd) walks the rows as their own keys, so a cross-attention takes the products or the maker's
/// library.
void qi_attention_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32,
                      const float* lse, const float* dO, int64_t T, int64_t heads, int64_t head_dim, float* dQ,
                      float* dK, float* dV, bool accumulate, prefill::Gemm& gemm, void* stream,
                      int64_t n_causal = 0, int64_t keys = 0);
/// The workspaces the training attention's forms keep for `stream` (the fused one's, the products' two, the
/// library's forward and backward): given back after the stream is synced, so a trainer's pause returns their VRAM
/// too.  The next call allocates its own again.
void qi_attention_train_release(void* stream);

/// The group a W8A8 product's operands are rotated in (kernels::rows_i8): K must be a multiple of it.
constexpr int64_t kQiRotate = kernels::kI8Rotate;

/// kernels::rows_i8 of the SwiGLU's values (prefill::swiglu_pair's), from the gate_up product's int32 sums gu [T][2 inter]
/// (scaled by g): the down product's int8 input in one pass.
void qi_swiglu_i8(const int32_t* gu, const DitOut& g, int64_t T, int64_t inter, int8_t* q, float* s, void* stream);

/// y [T][N] (row stride ldy) *= col[n], in place: a dense product's rows, scaled where a DoRA's c is not 1 (col is
/// the weight's rows' factors, y's rows the weight's row0 on).
void qi_rows_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* col, void* stream);


/// x = SiLU(x) in place, and y16 = FP16 of it (y16 may be null), n elements.
void qi_silu(float* x, int64_t n, uint16_t* y16, void* stream);

/// tokens16[p][c] = FP16(latent[c][p]), p < HW, c < C: a latent [C][H][W] as the image's tokens.
void qi_latent_tokens16(const float* latent, int64_t C, int64_t HW, uint16_t* tokens16, void* stream);

/// latent[c][p] = tokens[p][c]: the tokens' outputs [HW][C] back as a latent [C][H][W].
void qi_tokens_latent(const float* tokens, int64_t C, int64_t HW, float* latent, void* stream);

/// An edit with a mask, after a sampler's step to time s: x [C][HW] where m [HW] is 0 (the part kept) becomes the
/// base image's latent z0 at that time, (1 - s) z0 + s noise (the flow's path from the same noise); where m is 1 it
/// stays the DiT's.  `tokens`: x, z0 and noise are [HW][C] instead (Mage-Flow's latent tokens).
void qi_inpaint_blend(float* x, const float* z0, const float* noise, const float* m, float s, int64_t C, int64_t HW,
                      void* stream, bool tokens = false);
/// x = (1 - s) z0 + s noise over n values: the base image's latent at time s on the flow's path (img2img's start).
void qi_noise_mix(float* x, const float* z0, const float* noise, float s, int64_t n, void* stream);

// --- the vision encoder (Qwen3-VL's ViT, src/core/vision.cpp; its rows are in strata/kernels/image_attn.hpp)

/// The image engine's one unfold for its convolutions: col16[r][(dt k k + dy k + dx) C + c] = in16[t + dt dil -
/// 1][oy s + dy dil - pad][ox s + dx dil - pad][c] (0 outside), output pixel p0 + r = (t OH + oy) OW + ox, in16
/// [T][H][W][C], kt 1 or 3 frames (3: the frames before and after, the 3D VAE's), k x k taps of stride s spaced
/// `dil` apart (1: next to one another, a dilated convolution's spacing the line art uses).  Eight channels a work
/// item where C and `ldx` are both a multiple of 8 (every map the aux nets unfold and the engine's own tensors are),
/// one value a work item where they are not (a picture's 3-channel patch embedding).  `ldx`: in16's elements between
/// two pixels where it is wider than C (the first C channels read: RealESRGAN's dense blocks; 0: C).
void qi_im2col16(const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt, int k, int s, int pad,
                 int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16, void* stream, int64_t ldx = 0,
                 int dil = 1);

/// The image engine's one convolution output: y [T][ldy] (FP32, an im2col product's or a library's) to FP16, + bias
/// [O] (null: none), the activation, + res [T][O] (FP16, null: none), into dst [T][ld] from channel `off`, and its
/// channels from `from2` on also into dst2 [T][O - from2] (null: none).  The detector's (SiLU) and RealESRGAN's
/// (LeakyReLU of `slope`).
void qi_conv_store16(const float* y, int64_t T, int64_t O, int64_t ldy, const float* bias, ImageAct act, float slope,
                     const uint16_t* res, uint16_t* dst, int64_t ld, int64_t off, void* stream, uint16_t* dst2 = nullptr,
                     int64_t from2 = 0);

/// out16 [2H][2W][C] = x [H][W][C] repeated (nearest), FP16: the image engine's one 2x nearest upsampling (the VAEs',
/// RealESRGAN's); x FP16 in the second form.
void qi_up2x16(const float* x, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* stream);
void qi_up2x16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* stream);

/// GroupNorm: y16[p][c] = FP16(v), v = (x - mean[g]) * inv[g] * w[c] + b[c] over P
/// positions of C channels (g = c / (C / groups), the statistics over every position and the group's channels: one
/// pass of Chan's merges where a pixel's channels fit a work-group, two Kahan-folded passes past that), then
/// v * (1 + scale[c]) + shift[c] with `scale` (a ResBlock's embedding), then silu;
/// `lo16`: FP16(v - y16) too. `ws`: qi_group_norm_ws_floats(P, groups) floats, the means then inverse deviations.
int64_t qi_group_norm_ws_floats(int64_t P, int64_t groups = 32);
void qi_group_norm16(const float* x, int64_t P, int64_t C, const float* w, const float* b, float eps, bool silu,
                     uint16_t* y16, float* ws, void* stream, const float* scale = nullptr, const float* shift = nullptr,
                     uint16_t* lo16 = nullptr, int64_t groups = 32);
/// The same of 16-bit x16 (FP16 bits, BF16's with `bf16`), its statistics in FP32 as the FP32 form's.
void qi_group_norm16(const uint16_t* x16, bool bf16, int64_t P, int64_t C, const float* w, const float* b, float eps,
                     bool silu, uint16_t* y16, float* ws, void* stream, int64_t groups = 32);

/// GroupNorm over [P][C], one input and one or both outputs, optional affine and activation. FP16 may be in-place.
/// C must be divisible by groups; workspace has qi_group_norm_ws_floats(P, groups) floats.
void image_group_norm(const float* x, const uint16_t* x16, int64_t P, int64_t C, int64_t groups, const float* w,
                      const float* b, float eps, ImageAct act, float* y, uint16_t* y16, float* ws, void* stream);

/// p16[r][:ld] = FP16(softmax(scale sc[r][:n])) and zero past n, rows ld apart (0: n), FP32 statistics: a sub-group
/// of 32 a row up to 1,024 columns (8 rows a work-group), a work-group of 256 a row past them.  The image engine's one
/// softmax (the VAEs' attention, Anima's).
void qi_softmax16(const float* sc, int64_t rows, int64_t n, float scale, uint16_t* p16, void* stream, int64_t ld = 0);

}  // namespace strata::image
