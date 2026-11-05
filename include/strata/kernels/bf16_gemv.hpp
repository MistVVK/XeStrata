// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/bf16_gemv.hpp - the GEMV for a BF16 weight, P2.S5.
//
// WHY THIS EXISTS WHEN `s_gemv` ALREADY DOES.  `s_gemv` decodes the S-family canonical forms (S2/S4/S8) from
// `codes` + `scales`, and a BF16 weight is not one of them - it has no codes and no scales, it is a plain
// 16-bit matrix.  Six tensors per layer are in that state:
//
//     ssm_alpha.weight, ssm_beta.weight                 GDN,  [n_embd, 48]
//     indexer.q_proj.weight, indexer.k_proj.weight      QSA,  [n_embd, 512] and [n_embd, 128]
//     ple_value.weight                                  layer 1, [n_embd, n_embd]
//     ffn_gate_inp.weight                               handled inside `router_top10`
//
// The entry points follow the pinned CUDA single-token MMVF contract: BF16 weights, FP32 activations.  See
// docs/activation-contract.md for why activation precision depends on backend and batch geometry rather than weight
// type alone.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Single-token, contiguous 2D BF16-weight/F32-activation MMVF from the pinned llama.cpp CUDA path.
/// Preserves pair accumulation, adaptive block size and warp-XOR reduction order.
/// `w[o*n_in+i]` is BF16 bits; x/y are F32. n_in must be positive and even, n_out positive, both <= INT_MAX.
/// x must be 8-byte aligned, w 4-byte aligned, and y 4-byte aligned. Throws on invalid geometry/pointers.
/// The device buffers must not overlap; allocation bounds remain the caller's responsibility.
/// No allocations or synchronization, including when stream is null (the CUDA default stream).
void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream);
/// The same single-token MMVF for an F32 weight (ggml-cuda's mmvf with float weights): the router and the shared
/// expert's scalar gate of qwen35moe / qwen3moe, which their GGUFs keep in F32.  w must be 8-byte aligned.
void f32_gemv_fp32_mmvf(const float* x, const float* w, float* y, int64_t n_in, int64_t n_out, void* stream);
/// f32_gemv_fp32_mmvf for n_tok (1..8) activation rows x[t * ldx], outputs y[t * ldy + j]; one launch, the weight
/// read once, every output bit-identical to its own single-row call.  ldx even.
void f32_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const float* w, float* y, int64_t ldy, int64_t n_in,
                              int64_t n_out, int n_tok, void* stream);
/// y[t * ldy + j] = x[t * ldx ..] . w[j * n_in ..] for n_tok rows at once, F32 throughout: the prompt's router and
/// shared-expert gate, where n_tok is hundreds and the MMVF would read the activations again for every output.  Sums
/// in another order than the MMVF (the same values to rounding).
void f32_gemm_rows(const float* x, int64_t ldx, const float* w, float* y, int64_t ldy, int64_t n_in, int64_t n_out,
                   int64_t n_tok, void* stream);
/// bf16_gemv_fp32_mmvf for n_tok (1..8) activation rows x[t * ldx], outputs y[t * ldy + j]; one launch,
/// the weight read once, every output bit-identical to its own single-row call.
void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream);
/// Times bf16_gemv_fp32_mmvf_multi's two forms (a row a work-group, 4 rows a work-group: the same bits) on the stream's
/// device for an n_in x n_out matrix and 2..8 rows, and keeps the faster for each; call it before any graph is
/// recorded (once a device and shape).  Untuned shapes take a row a work-group.  STRATA_MMVF_ROWS=0|1 fixes the form.
void bf16_gemv_fp32_mmvf_multi_tune(int64_t n_in, int64_t n_out, void* stream);
/// bf16_gemv_fp32_mmvf_multi(x, ldx, w, y, ldy, n_in, n_out, n_tok) and the 1-row
/// bf16_gemv_fp32_mmvf_multi(x, ldx, w_aux, y_aux, ldy_aux, n_in, 1, n_tok) in one launch, every output the two calls'
/// bits (upstream b08cf3e1: the router and the shared expert's scalar gate).  2..8 rows.  Returns false, launching
/// nothing, when the main product takes the 4-rows form here.
bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream);

}  // namespace strata::kernels
