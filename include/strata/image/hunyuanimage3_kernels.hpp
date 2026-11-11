// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_kernels.hpp - the image engine's element kernels for HunyuanImage 3.0.
//
// A layer of the model is: an RMSNorm, the save's rotation of the activations (the quantized files hold their
// weights in the rotated basis), a product, the per-KV-head qkv split with the 2D RoPE and the qk-norm, the
// attention (prefix causal, the image block full), the router and the experts' grouped products, SwiGLU with the
// rotation again.  The products are prefill::Gemm's; what is here is everything around them.
//
// The formulas are the ones the record's `hy3_conform.py` holds to Tencent's own modules
// (records/2026-10-23-hunyuanimage3-plan/CONFORM-2026-10-24.md): the rotation is the REGULAR Hadamard
// kron(H4,H4,H4,H4)/16, not the Sylvester one kernels::rows_i8 uses; the RoPE turns column j and j + head_dim/2 by the
// SAME angle; SwiGLU silu's the second half; the qk-norm comes after the rope.
#pragma once

#include <cstdint>

namespace strata::image {

// The layer's RMSNorm is the engines' one (kernels::rms_norm_rows, native_qsa.hpp) at the model's eps 1e-5
// (config.rms_norm_eps), FP16 out, or FP32 where the router multiplies by it: its logits decide which of the 64
// experts a token takes, and a FP16 activation there moves a logit by ~1e-3 - enough to swap a near-tie's expert and
// change that token's output by a few percent (measured on the B70 with the w4a8 file and the CPU golden, 2026-10-24).

/// The save's rotation of an FP32 input: the same groups, FP32 in and FP16 out.  The layer's activations have a
/// few enormous channels (the model's "massive activations": absmax 464 where the rms is 7), and the rotation
/// spreads them over its group - so rounding the *pre-rotation* row to FP16 leaves that row's few percent in
/// every rotated value, while rounding the rotated one costs 5e-4.  The norm therefore writes FP32 and this
/// kernel takes it (measured on the B70 with the w4a8 file and the CPU golden, 2026-10-24).
void hunyuanimage3_rotate16_from32(const float* x32, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream);

/// The save's rotation: every group of `block` (256) values of every row times the normalized regular Hadamard
/// (kron(H4,H4,H4,H4)/16 with H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]), applied as four butterflies on
/// the group's base-4 digits.  H is orthogonal and symmetric, so a rotated activation against the file's rotated
/// weights gives the unrotated product.  K is a multiple of `block`; [rows][K] in and out.
void hunyuanimage3_rotate16(const uint16_t* x16, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream);

/// A w4a8 matrix's rows [r0, r1) (of one expert of a bank: the caller passes that expert's slice): two 4-bit
/// codes a byte with the EVEN column in the low nibble, a codebook[16] value a code, an FP8 e4m3 scale `s_rel`
/// every `group` (16) columns and a scale a row `s_channel`: value = codebook[code] * e4m3(s_rel) * s_channel,
/// in the file's rotated basis.  FP16 out [r1-r0][K].
void hunyuanimage3_w4a8_dequant16(const int8_t* codes, const uint8_t* s_rel, const float* s_channel, const float* codebook,
                        int64_t r0, int64_t r1, int64_t N, int64_t K, int64_t group, uint16_t* out16, void* stream);

/// An int8_convrot matrix's rows [r0, r1): int8 values with one FP32 scale a row, in the rotated basis.  FP16 out.
void hunyuanimage3_int8_dequant16(const int8_t* w, const float* row_scale, int64_t r0, int64_t r1, int64_t N, int64_t K,
                        uint16_t* out16, void* stream);

/// ang[t][j] = (j % 2 ? pos_x[t] : pos_y[t]) * theta^(-2 j / head_dim), j < head_dim/2: the 2D rope's angles, one
/// set a token (the reference's `build_2d_rope`; the caller keeps the positions it built from the ids).
void hunyuanimage3_rope_angles(const int32_t* pos_y, const int32_t* pos_x, int64_t T, int64_t head_dim, float theta,
                     float* ang, void* stream);

/// The qkv product's FP32 output [T][kv_heads * (groups + 2) * head_dim] (each KV head's rows q0..q(groups-1), k,
/// v, the file's order): rotate q and k by `ang` (column j and j + head_dim/2 by the SAME angle, rotate_half),
/// RMS-norm them over the head (wq, wk; eps), and write q, k and v FP16 as [T][q_heads * head_dim] with each KV
/// head repeated `groups` times - so the plain attention, which takes one head count, can read them.
void hunyuanimage3_qk_rope_norm16(const float* qkv, const float* wq, const float* wk, const float* ang, int64_t T,
                        int64_t q_heads, int64_t kv_heads, int64_t head_dim, float eps, uint16_t* q16,
                        uint16_t* k16, uint16_t* v16, void* stream);

/// out16[t][i] = FP16(gu[t][i] * silu(gu[t][inter + i])) rotated by the save's Hadamard in groups of 256: the
/// fused gate_up product's two halves (the SECOND one silu'd, as the reference chunks them) and the down
/// product's rotated input in one pass.  gu [T][2 inter], out16 [T][inter]; inter is a multiple of 256.
void hunyuanimage3_swiglu16_rotated(const float* gu, int64_t T, int64_t inter, uint16_t* out16, void* stream);


/// x += v, n FP32 values (the residual).
void hunyuanimage3_add(float* x, const float* v, int64_t n, void* stream);

/// y16[p][c] = FP16(x[c][p]): the FP32 [C][P] a host piece writes (the latent, a convolution in channels-first)
/// as the FP16 [P][C] every product here takes.  P a multiple of 1 and C of 8 (the vec8 inner).
/// `lo16` (here and in to16 and group_apply16), when given: FP16(v - y16), the rounding FP16 left, so that a
/// product of y16 plus one of lo16 has the operand to about 22 bits.
void hunyuanimage3_grid_in16(const float* x, int64_t C, int64_t P, uint16_t* out16, void* stream,
                             uint16_t* lo16 = nullptr);

/// y16[i] = FP16(x[i]), n values: an FP32 tensor as the next product's row.
void hunyuanimage3_to16(const float* x, int64_t n, uint16_t* y16, void* stream, uint16_t* lo16 = nullptr);

/// col16[r][tap * C + c] = in16[(y + dy, x + dx)][c] (0 outside the grid), output pixel p0 + r with
/// p = y * W + x, taps in (dy, dx) order matching the convolution weights' packing: the unfold of a 3x3
/// convolution at a grid of H x W pixels, which `prefill::Gemm::f16` then multiplies as [rows][9 C] by [O][9 C].
/// C a multiple of 8 (the vec8 unfold).
void hunyuanimage3_im2col16(const uint16_t* in16, int64_t H, int64_t W, int64_t C, int64_t p0, int64_t rows,
                            uint16_t* col16, void* stream);

/// y[r][:] += b[:], rows of N (a convolution's or a linear's bias over its product's FP32 output).
void hunyuanimage3_bias_add32(float* y, const float* b, int64_t rows, int64_t N, void* stream);

/// dst16[:] = src16[:], n values: a meta token's row overwritten by its embedder's output.
void hunyuanimage3_copy16(const uint16_t* src16, uint16_t* dst16, int64_t n, void* stream);



/// The routed rows grouped by expert: bounds[e] .. bounds[e + 1] (E + 1, from 0) of rows and w are expert e's tokens
/// (in the order the device's atomics give, which changes no row's product) and their weights; `cursor` (E) is the
/// fill's.  On the device: the caller reads `bounds` back for the experts' products.  The engines' one device-side
/// list: the LLMs' prompts list their rows on the host, with where each expert runs (prefill.cpp, plain_prompt.cpp).
void hunyuanimage3_route_lists(const int32_t* idx, const float* wt, int64_t T, int64_t K, int64_t E, int32_t* bounds,
                               int32_t* rows, float* w, int32_t* cursor, void* stream);

/// y[t][:] = FP16(x[t][:] / rs[t]) - a row-scaled read of FP32 products into the FP16 the next product takes.
void hunyuanimage3_scale_rows16(const float* x, const float* rs, int64_t n_rows, int64_t C, uint16_t* y16, void* stream);

}  // namespace strata::image
