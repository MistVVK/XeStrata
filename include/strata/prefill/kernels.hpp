// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/prefill/kernels.hpp - plan v0.3 P5: the element-wise and stateful kernels of a prompt chunk.
//
// T tokens at a time, all row-major with the token as the slow index.  The arithmetic mirrors the per-token native
// kernels (the decode path), batched; where a step is a recurrence in time (the GDN conv and state, the PLE conv)
// the kernel walks the chunk's tokens in order inside one launch.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cstdint>

namespace strata::prefill {

// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// rs[t*4 + c] = rsqrt(mean_d R[t,c,:]^2 + eps) and xn16[t, c*2560 + d] = bf16(R[t,c,d] * rs * w_norm[c*2560 + d]); the
/// FP32 normalized rows are not written, gr_mix_r recomputes them (F-1, upstream 882bb6d).
/// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x
/// (STRATA_PREFILL_BF16X2); the other `*_lo` arguments below the same for their BF16 images.
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo = nullptr);
/// mixed[t, d] = mean_c xn[t,c,d] * sigmoid(gated[t,c,d]), xn = R * rs * w_norm as gr_norm_rs computes it (the same
/// bits); FP32, BF16 and FP16 (either image may be null).
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);
/// F-2 (upstream b046845): gr_write, then gr_norm_rs of the next half (its norm weights) over the rows just written -
/// the same bits as the two calls, without reading R back.
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr);
/// lo16[t, k] = bf16(silu(lo[t, k] / hc))
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr);
/// R[t, c, d] += bo[t, d] * 2 sigmoid(inj[t, c] / hc)   (inj has row stride inj_ld)
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream);
/// gr_write, cvec_apply(R, layer, ...) of the loaded control vector and gr_norm_rs of the next half in one pass over R
/// (upstream 6448f0f7): gr_write's expression, cvec_apply's dot order and two-level sum, gr_norm_rs's sum order and
/// block_sum - the same bits as the three launches.
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t layer,
                           const float* w_norm_next, float eps, float* rs_out, uint16_t* xn16, int64_t T, void* stream,
                           uint16_t* xn16_lo = nullptr);
/// R[t, c, :] = e[t, :] for all four streams (the embedding broadcast).
void gr_broadcast(const float* e, float* R, int64_t T, void* stream);

// ---- GDN (state 128, 4 taps; h_k key heads and h_v value heads, C = 2 h_k 128 + h_v 128 conv channels: Qwen3.8's
// 16 and 48, the defaults, and qwen35moe's 16 and 32 - any other shape throws)
/// gate[t,h] = softplus(ab[t,h] + dt[h]) * ssm_a[h];  beta[t,h] = sigmoid(ab[t, h_v + h])  (ab: [T, 2 h_v])
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream,
               int64_t h_k = 16, int64_t h_v = 48);
/// The 4-tap causal conv + SiLU over the chunk (history [C][3] in, updated to the chunk's last three inputs), then
/// the L2 norm of the q and k heads of every token.  h: [T, C].
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              int64_t h_k = 16, int64_t h_v = 48);
/// The recurrence over the chunk, state in registers; y16[t] = rmsnorm(o) * gamma * sigmoid(z) in FP16 (what the
/// out projection reads; `silu`: * silu(z), qwen35moe's gate); y is FP32 scratch.
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t h_k = 16,
                    int64_t h_v = 48, bool silu = false);
/// KVA (kva.hpp): the same recurrence kernels as gdn_recurrence, the same state bits, without the output
/// normalization: state, h, gate, beta and the scratch `y` only.  A state-only late layer has no reader for the output
/// projection or the gated residual write, nor for their input (z, the gate projection).
void gdn_recurrence_state_only(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                               void* stream, int64_t h_k = 16, int64_t h_v = 48);
// ---- MoE
/// softmax over 512, top-10 (ties to the lower id), weights renormalised over the ten (the native router).
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream);
/// Expert blob (Strata pack layout, Q2_0) -> BF16 matrices: gate/up interleaved [1280, 2560], down [2560, 640].
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);
/// h16[n, r] = fp16(silu(gu[n, 2r]) * gu[n, 2r + 1])   (the interleaved expert gate/up)
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream);
/// h16[n, r] = fp16(silu(g[n, r]) * u[n, r])   (the shared expert, gate and up separate, n rows of `width`: Qwen3.8's
/// 640, qwen35moe's 512; the image engine's blocks), FP16's range saturated; g's and u's rows `ld` apart (0: width),
/// h16's width apart
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream, int64_t width = 640,
                 int64_t ld = 0);
/// Gather rows: dst16[i, :] = x16[src[i], :] (n rows of `width` BF16).
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream);
/// Gather rows: dst[i, :] = x[src[i], :] (n rows of `width` floats).
void gather_rows(const float* x, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream);
/// Scatter rows: dst[dst_row[i], :] = src[i, :] (n rows of `width` floats).
void scatter_rows(const float* src, const int32_t* dst_row, float* dst, int64_t n, int64_t width, void* stream);
/// bo[t, :] = shared[t, :] * sigmoid(sg[t]) + sum_k w[t, k] * D[slot[t, k], :]
void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream);
/// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): acc[t, :] = shared[t, :] * sigmoid(sg[t]) - the shared expert's
/// part of moe_combine, written before the routed experts are known.
void moe_shared_init(const float* shared, const float* sg, float* acc, int64_t T, void* stream);
/// Layer-major prefill: one expert group's rows added into the super-chunk's accumulator, per token in a fixed order
/// (deterministic, no atomics): for each u < n_u, acc[u_tok[u], :] += sum_{j in [u_off[u], u_off[u+1])} u_w[j] *
/// Dm[u_rows[j], :].  The sum runs in the group's row order, not moe_combine's k order (FP32 rounding only).  Rows of
/// `width` (0: Qwen3.8-Flash-Next's 2560); u_off and u_rows null: token u's one row is row u (an expert's own rows,
/// one a token: HunyuanImage 3.0's experts one after another).
void moe_scatter_add(const float* Dm, const int32_t* u_tok, const int32_t* u_off, const int32_t* u_rows,
                     const float* u_w, int64_t n_u, float* acc, void* stream, int64_t width = 0);

// ---- QSA helpers
/// NEOX rotary on the first 64 dims of each head: x [T, heads, dim] at positions pos0 + t.  The rope
/// scaling (rope_scaling.hpp) rides in as the process config - none is today's arithmetic exactly.
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream);
/// q_full [T, 24, 512] (q | gate per head) -> q [T, 24, 256]
void split_q(const float* q_full, float* q, int64_t T, void* stream);

/// K and V of T consecutive cells (positions pos0..pos0+T-1; K normed and rotated) into the paged pools: FP16
/// (`k_pool`/`v_pool`) or INT8 codes + FP16 scale per 64 (`k_q`...), the decode append's arithmetic.
/// K, V: [T, 2, 256].
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host = nullptr,
               const strata::kernels::KvHostPools* stage = nullptr);

/// fp32 -> bf16, n elements (the prompt GEMMs' BF16 activations; their FP16 ones: kernels::f32_to_f16_bulk).
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
/// bf16 -> fp16 bits, n elements, finite values past FP16's range saturated at +-65504 (Inf and NaN stay): exact for
/// every value in FP16's normal range (Gemm::bf16 through FP16)
void bf16_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream);
/// y = fp32(fp16(x)): what an FP16 store of x would read back (the FP16 indexer-key experiment)
void round_f16(const float* x, float* y, int64_t n, void* stream);
/// Expert blob -> FP16 (Q2_0 values are exact in FP16).
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);

}  // namespace strata::prefill
