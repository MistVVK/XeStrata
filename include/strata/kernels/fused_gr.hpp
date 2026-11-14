// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    /// STRATA_HC_Q8=1: the GGUF's Q8_0 down and up projections in kNativeQ8Rows' layout (each row's int8 values,
    /// then its fp16 scales), both or neither; fused_gr_read_multi reads them in place of w_down / w_up
    const uint8_t* q8_down = nullptr;  ///< [hc_lr][hc*n_embd]
    const uint8_t* q8_up = nullptr;    ///< [hc*n_embd][hc_lr]
    /// with the Q8_0 read: this token's mixed is also written as q8_1 blocks here (n_embd / 32 of 36 bytes, bitwise
    /// native_quantize_q8_1's) by a 32-column up (upstream 0a379a1f); every token's or none
    void* q8_mixed = nullptr;
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
/// STRATA_HC_Q8 on a pack whose hyper-connection projections are BF16 (upstream 0a379a1f): `nb` Q8_0 blocks (34 bytes:
/// an fp16 scale, then 32 int8 values) of the BF16 values `src`, as ggml's quantize_row_q8_0 (scale = max |x| / 127,
/// each value rounded to the nearest).  Ordered on `stream`.
void fused_gr_quantize_q8_0(const uint16_t* src, uint8_t* dst, int64_t nb, void* stream);
void fused_gr_read(const FusedGrArgs& a, void* stream);
/// STRATA_HC_Q8 with the BF16 copies dropped: a projection's Q8_0 values in kNativeQ8Rows' layout (`ne1` rows of
/// `ne0`) as BF16 [ne1][ne0], for the readers that take BF16 (the prompt path, the single-token read).  On `stream`.
void fused_gr_q8_rows_to_bf16(const uint8_t* q8, uint16_t* dst, int64_t ne0, int64_t ne1, void* stream);
/// STRATA_HC_Q8 unset: one layer's read (`w`'s weights; its BF16 and its Q8_0 pointers) timed on a window of `n_tok`
/// tokens, `reps` times after a warm-up, on zero inputs (the time does not depend on the values) -> the mean
/// microseconds of the BF16 read and of the Q8_0 read, each with the launch that quantizes mixed to q8_1 after it,
/// and of the Q8_0 read that writes the q8_1 itself (`q8_mixed`).
void fused_gr_time_q8(const FusedGrArgs& w, int n_tok, int reps, void* stream, double& bf16_us, double& q8_us,
                      double& q8f_us);
/// The Q8_0 read writes mixed's q8_1 itself (the verify window's first half then skips its quantize launch), where
/// fused_gr_time_q8 found it faster at load.  Process-wide.
void fused_gr_set_q8_fuse(bool on);
bool fused_gr_q8_fuse();

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

}  // namespace strata::kernels
