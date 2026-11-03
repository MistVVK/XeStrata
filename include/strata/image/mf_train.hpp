// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_train.hpp - Mage-Flow's DiT trained: the gradients a LoRA, a LoKr or a DoRA learning step
// needs, on the GPU, beside the inference side (src/image/mf_dit.cpp) and never in its path.
//
// A training step runs the inference block's own math with a tape: every value a gradient is read from is kept (the
// LayerNorm's mean and inverse, the products' FP16 input rows and their power-of-two scales, the attention's softmax
// statistics, the GELU's input).  A product is prefill::Gemm's, with the same FP16 rows and FP32 sums the inference
// takes, so a step starts from the values the engine already reads.
//
// The gradients themselves are FP32.  One product Y[T,N] = X[T,K] . W[N,K]^T + b has
//
//   dX[T,K] = dY[T,N] . W[N,K]        (mf_dgrad, W read along its rows: the caller's transposed view wt[K,N])
//   dW[N,K] = dY[T,N]^T . X[T,K]      (mf_wgrad)
//   db[N]   = sum_t dY[t,N]           (mf_bias_bwd)
//
// and prefill::Gemm knows only Y = X . W^T with W row-major, so the training side keeps a transposed FP16 view of every
// weight it takes gradients through (built when a block is opened for training) and reads a gradient's rows transposed
// too (mf_transpose_f16, mf_scaled_transpose_f16).  A row's or head's power of two cancels exactly, except in dW,
// which sums over the rows: there the row scales are folded into the FP32 gradient first and one global power of two
// (G, from mf_scaled_transpose_f16) is put back on the sum.
//
// An adapter's gradients, one weight, with s the request's strength and k = alpha / rank:
//
//   plain LoRA   h = x A^T, delta = s k (h B^T)   dB = s k dY^T h   dA = s k (dY B)^T x
//   LoKr         delta = s k kron(w1, w2_a w2_b)   each factor by the partial trace of the delta's gradient
//   DoRA         c_r = m_r / (|W_r| + eps); the rows' factors a_r = 1 + s (c_r - 1) and b_r = s c_r, and
//                dm_r = s sum_t dY[t,r] (p[t,r] + d[t,r]) / (|W_r| + eps)   (p the base product, d the delta's)
//
// |W_r| is the row norm of the weight the engine reads (its own dora_norms), eps FP32's.
//
// The step is sd-scripts' rectified flow (docs/LORA_TRAIN.md, "Timestep sampling and loss"): x_sigma =
// (1 - sigma) x0 + sigma eps, the target v = eps - x0, the loss the mean of (v_pred - v)^2, so
// dL/dv_pred = 2 (v_pred - v) / n.
//
// Checked per op against NumPy and one block against torch, on the AMD GPU (src/image/mf_train_parity.cpp,
// tools/image/mf_train_ref.py, tools/image/mf_train_torch_ref.py).
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>

namespace strata::image {

/// The LayerNorm(no affine, eps 1e-6) * (1 + scale) + shift a block's norm and modulation does, backwards: dx is the
/// gradient of the norm's input, dshift and dscale the columns' sums added into (null: not taken - a frozen modulation
/// does not ask).  mean and inv are what the forward kept, one per row.
void mf_ln_mod_bwd(const float* x, const float* mean, const float* inv, const float* scale, const float* dy,
                   int64_t rows, int64_t cols, float* dx, float* dshift, float* dscale, void* stream);

/// dL/dv of the GELU (tanh form) the MLPs use: g(v) = 0.5 v (1 + tanh(0.7978845608 (v + 0.044715 v^3))).
void mf_gelu_bwd(const float* v, const float* dy, int64_t n, float* dv, void* stream);

/// The residual x += gate * h, backwards: dh[c] = gate[c] * dx[c], and dgate[c] = sum_t dx[t,c] h[t,c] added into
/// (either null: not taken).  gate null: 1.
void mf_gate_add_bwd(const float* h, const float* gate, const float* dx, int64_t rows, int64_t cols, float* dh,
                     float* dgate, void* stream);

/// SiLU(y) = y sigmoid(y), backwards: dy = d * s (1 + y (1 - s)), s the sigmoid of y.
void mf_silu_bwd(const float* y, const float* d, int64_t n, float* dy, void* stream);

/// The 2 x 2 packing of a latent to tokens with the VAE's batch norm (a token's 128 values are four positions' 32
/// channels at k = 4 c + 2 (Y % 2) + X % 2) and z = (m - mean) / sd, backwards: dm is the scatter of dz / sd, and
/// dmean and dsd the tokens' sums added into (null: not taken).  m is [2 h][2 w][64]'s first 32 channels, dz is
/// [h w][128].
void mf_patch_bwd(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, const float* dz, float* dm,
                  float* dmean, float* dsd, void* stream);

/// One step's flow loss: the mean of (v_pred - (eps - x0))^2 over n values, returned on the host, with dv_pred the
/// gradient 2 (v_pred - v) / n.  dx0 and deps (null: not taken) are that loss's way back to the latent and the noise
/// through the target; the noised input x_sigma = (1 - sigma) x0 + sigma eps is the step's own argument.
float mf_flow_loss(const float* v_pred, const float* x0, const float* eps, int64_t n, float* dv_pred, float* dx0,
                   float* deps, void* stream);

/// [rows][cols] FP16 -> [cols][rows] FP16, both row-major.
void mf_transpose_f16(const uint16_t* in, int64_t rows, int64_t cols, uint16_t* out, void* stream);

/// A gradient's rows read by a dW: out[c * rows + t] = (g[t,c] * rsX[t]) / G as FP16, G one power of two over the
/// whole tensor (the stored X rows are divided by their own rsX, so the factors cancel and only G is left to put
/// back).  The G used comes back as a host float: mf_wgrad takes it by value, and reading it off the device afterwards
/// would be a second pass over the queue for nothing.
float mf_scaled_transpose_f16(const float* g, const float* rsX, int64_t rows, int64_t cols, uint16_t* out,
                              void* stream);

/// A plain LoRA's factors, backwards, with s the request's strength and k = alpha / rank folded into `scale`.  The
/// forward kept h16 [T][rank] (the input through A) and the weight's own product read x16 [T][cols]; dY is the
/// gradient reaching the weight's rows [T][rows] in FP32.  dB [rows][rank] = scale dY^T h and
/// dA [rank][cols] = dh^T x read dh = scale (dY B), the intermediate this same call's scratch starts with (dL/dh,
/// so dA takes no second scale); dh's own rows and scale take the product's own dgrad back to the input.
/// h16t and x16t are the transposed views the forward kept (mf_transpose_f16 of the rows it wrote).
void mf_lora_bwd(const float* hr, const float* xr, const uint16_t* h16t, const uint16_t* x16t,
                 const uint16_t* bt16, const float* dY, int64_t T, int64_t rows, int64_t cols, int64_t rank,
                 float scale, float* dB, float* dA, float* scratch, uint16_t* tmp16, float* rsT, prefill::Gemm& gemm,
                 void* stream);

/// A LoKr's factors, backwards.  The delta is kron(w1, w2_a w2_b) and the step's gradient of the whole weight
/// (dDelta [out][in], out = out_l * out_k, in = in_m * in_n) is what a wgrad of the delta's rows gives; each factor
/// is read through it by a partial trace:
///   dw1[l][m] = sum_{kn} dDelta[l * out_k + k][m * in_n + n] w2[k][n]
///   dw2[k][n] = sum_{lm} dDelta[l * out_k + k][m * in_n + n] w1[l][m]
/// and then w2 = w2_a w2_b splits as a product does.  dw2a [out_k][rank] and dw2b [rank][in_n] are added into.
void mf_lokr_partial_trace(const float* dDelta, const uint16_t* w1, const uint16_t* w2, int64_t out_l, int64_t out_k,
                           int64_t in_m, int64_t in_n, float* dw1, float* dw2, void* stream);
/// dw2 [out_k][in_n] into its two factors: dw2a[o][r] = sum_n dw2[o][n] w2b[r][n] and
/// dw2b[r][n] = sum_o w2a[r][o] dw2[o][n] (the same products the forward reads, in FP32 here - the factors are small).
void mf_lokr_factor_bwd(const float* dw2, const uint16_t* w2a, const uint16_t* w2b, int64_t out_k, int64_t in_n,
                        int64_t rank, float* dw2a, float* dw2b, void* stream);

/// A DoRA's magnitude m [rows], backwards, with the row norm |W_r| it was divided by (the engine's own dora_norms)
/// and eps FP32's: dm[r] = scale * sum_t dY[t,r] (p[t,r] + d[t,r]) / (|W_r| + eps), where p is the base product's
/// rows and d the delta's.  `da` and `db` (null: not taken) get the factors the two paths' own gradients are read by:
/// da[r] = 1 + scale (c_r - 1) and db[r] = scale c_r, so the caller's dgrad and wgrad see dY da and dY db.
void mf_dora_bwd(const float* dY, const float* p, const float* d, const float* m, const float* norms, int64_t T,
                 int64_t rows, float scale, float* dm, float* da, float* db, void* stream);

/// One Mage-Flow block, run forwards with a tape and backwards through it, out of the ops above: what
/// tools/image/mf_train_torch_ref.py is checked against.  The block is the inference block's own math
/// (src/image/mf_dit.cpp): each stream's LayerNorm (no affine, eps 1e-6) with the time's shift and scale, q, k and v,
/// each head RMS-normed with its weight and the image's rows rotated pairwise, one attention over [text, image] where
/// every row sees every key, the output projection and the gated residual, then the second norm and the tanh-GELU MLP
/// with its gated residual.  A product's input rows are FP16 divided by a power of two and its sums are FP32, exactly
/// as the inference reads them, so an FP32 torch reference meets these gradients at FP16's rounding.
///
/// The time's modulation is a value, not a trained tensor: v1 leaves the modulations out of the target modules
/// (docs/LORA_TRAIN.md, "The four models"), so nothing goes back into shift, scale or gate's own weights.  An adapter
/// (MfBlockW::ad, null: none) adds its delta to the product's rows in the forward and its gradients in the backward;
/// a null dw/db skips the base weight's own wgrad and bias sum (a frozen weight takes no gradient, and a LoKr-folded
/// one reads only dDelta through dw - the block knows no LoKr).
/// y[t][c] *= r[t] (in place): a DoRA's row factors (mf_dora_bwd's da/db) on a gradient's rows.
void mf_scale_rows(float* y, const float* r, int64_t rows, int64_t cols, void* stream);

/// One block product's adapter (plain LoRA or DoRA; v1 trains the attention and the MLP projections, all six).
/// null in MfBlockW::ad: the product runs as it always did, at the same cost.
enum class MfAdapterMethod : int { lora = 0, dora = 1 };
struct MfAdapter {
    MfAdapterMethod method = MfAdapterMethod::lora;
    int64_t rank = 0;
    float scale = 0.f;                 // s * alpha / rank, folded (the request's strength times the file's alpha)
    const uint16_t* a = nullptr;       // [rank][K] FP16: h = x A^T reads it
    const uint16_t* at = nullptr;      // [K][rank] the same values transposed (the input gradient's dgrad reads it)
    const uint16_t* b = nullptr;       // [N][rank] FP16: delta = h B^T reads it
    const uint16_t* bt = nullptr;      // [rank][N] the same values transposed (mf_lora_bwd reads it)
    float* dA = nullptr;               // [rank][K], added into
    float* dB = nullptr;               // [N][rank], added into
    const float* m = nullptr;          // [N] the DoRA magnitude (null: a plain LoRA)
    const float* norms = nullptr;      // [N] the base weight's row norms (null: a plain LoRA)
    float* dm = nullptr;               // [N], added into (null: not taken)
};

struct MfBlockW {
    int64_t N[6] = {}, K[6] = {};        // q, k, v and o are C x C, mlp1 is F x C, mlp2 is C x F (set: the adapter
                                         // path reads them, and zero widths enqueue no work and fail)
    const uint16_t* w[6] = {};           // [N][K] FP16, as the file writes them
    const uint16_t* wt[6] = {};          // [K][N] FP16, the same values transposed (dgrad reads them)
    const float* b[6] = {};              // [N] FP32
    const MfAdapter* ad[6] = {};         // null: no adapter on the product
    const float *nq = nullptr, *nk = nullptr;      // [head_dim] FP32 each: the file's attn.norm_q / norm_k weight
    const float* mod = nullptr;          // [6 C]: shift, scale, gate of the attention, then of the MLP
    float* dw[6] = {};                   // [N][K] FP32, added into
    float* db[6] = {};                   // [N]
    float *dnq = nullptr, *dnk = nullptr;   // [head_dim] each, added into
};

struct MfBlockRun {
    int64_t C = 0, heads = 0, hd = 0, F = 0, rows_img = 0, rows_txt = 0;
    const float *img = nullptr, *txt = nullptr;    // [rows][C] FP32
    const float *cos = nullptr, *sin = nullptr;    // [rows_img][hd / 2] (null: no rotation)
    MfBlockW img_w, txt_w;
    float *out_img = nullptr, *out_txt = nullptr;         // the forward's outputs
    const float *d_out_img = nullptr, *d_out_txt = nullptr;   // the gradients reaching them
    float *dimg = nullptr, *dtxt = nullptr;         // the block's inputs' gradients, added into
    float* tape = nullptr;    // the forward's stages out to the parity harness (null: not taken, see MfTapeSlice)
};

/// The block's forward tape, widened to FP32 and copied out for tools/image/mf_train_torch_ref.py to compare stage by
/// stage: `MfBlockRun::tape` (null: not taken) is a device array of `mf_block_tape_elems`'s count, laid out as the
/// slices below per stream (the image's rows first, then the text's) and then the joint attention's.  A product's FP16
/// input rows land here widened but NOT multiplied back by their row's scale, so a reference that rounds the same way
/// reads the same numbers.  Every copy is off the inference path: a real training step passes null and pays nothing.
enum MfTapeSlice : int {
    kTapeLn1 = 0,   // [rows][C]   the first norm's rows, as the q, k and v products read them
    kTapeQ,         // [rows][C]
    kTapeK,         // [rows][C]
    kTapeV,         // [rows][C]
    kTapeO,         // [rows][C]   the output projection
    kTapeH1,        // [rows][C]   the attention's gated residual, the second norm's input
    kTapeLn2,       // [rows][C]   the second norm's rows, as mlp1 reads them
    kTapeM1,        // [rows][F]
    kTapeGelu,      // [rows][F]   mlp2's input rows
    kTapeM2,        // [rows][C]
    kTapeOut,       // [rows][C]   the block's output
    // then the same tensors' gradients, in the order the backward reaches them (what each one's producer's output
    // gradient is, before it moves on): the backward is the longer half to get wrong and the harder to see
    kTapeDLn1,      // [rows][C]   into the first norm's FP32 output, the three products' dgrads summed
    kTapeDQ,        // [rows][C]
    kTapeDK,        // [rows][C]
    kTapeDV,        // [rows][C]
    kTapeDYo,       // [rows][C]   into the output projection
    kTapeDH1,       // [rows][C]   into the first residual's sum
    kTapeDLn2,      // [rows][C]   into the second norm's FP32 output
    kTapeDM1,       // [rows][F]   into mlp1
    kTapeDGelu,     // [rows][F]   into the GELU's output, which is mlp2's dgrad
    kTapeDM2,       // [rows][C]   into mlp2
    kTapeStreamN,   // slices a stream: the tape holds the image's rows first, then the text's
};
/// the joint rows, after both streams: the attention's own inputs and output, its statistics, and their gradients
enum MfTapeJoint : int { kTapeJQ = 0, kTapeJK, kTapeJV, kTapeJO, kTapeJDO, kTapeJDQ, kTapeJDK, kTapeJDV,
                         kTapeLse,   // the last slot, and the only one `heads` wide instead of `C`
                         kTapeJointN };
/// after the joint block, the rows' power-of-two scales, the image's rows then the text's, one pair per stored row set:
/// they are the other half of the contract a product reads (its rows and the scale that puts them back), and a scale
/// that came out zero turns every product into its own bias - the shape of the failure a dump of the products cannot
/// tell apart from a zero weight.
enum MfTapeScale : int { kTapeScLn1I = 0, kTapeScLn1T, kTapeScLn2I, kTapeScLn2T, kTapeScGeluI, kTapeScGeluT,
                         kTapeScaleN };

int64_t mf_block_tape_elems(const MfBlockRun& run);

/// the block's forward and backward in one call.  The tape and every scratch buffer belong to the call: the block is
/// small and reading it twice is cheaper than getting an alias wrong.
void mf_block(const MfBlockRun& run, prefill::Gemm& gemm, void* stream);

/// qk_rope_v16's backward: dgq and dgk are the gradients of the normed and rotated rows (what the attention reads),
/// dq and dk those of the products' FP32 outputs.  A head's rows are RMS-normed over head_dim with the weights wq and
/// wk and rotated pairwise (2j, 2j+1) by cos[t][j] and sin[t][j] (rope false: only the norm).  wqn and wkn (null: not
/// taken) are the norm weights' gradients, one per channel of a head, summed over the rows and the heads.  v is not
/// here: its head scale belongs to the row kernels around the attention, not to this op.
void mf_qk_norm_rope_bwd(const float* q, const float* k, const float* wq, const float* wk, const float* cos,
                         const float* sin, bool rope, const float* dgq, const float* dgk, int64_t rows, int64_t heads,
                         int64_t head_dim, float* dq, float* dk, float* wqn, float* wkn, void* stream);

/// Mage-Flow's joint attention, in the form the training step holds it: q16, k16 and v16 are [rows][heads *
/// head_dim] FP16 with each head's power of two already folded out (dit_rows' Heads puts it back on the way to the
/// output projection), and every row sees every key (the joint attention has no mask).
///
/// The forward and backward are tiled over the heads and read through the products (prefill::Gemm, FP32 sums): one
/// head's rows gathered contiguous, S = Q K^T, the softmax in place, O = P V.  The backward recomputes P from Q and K
/// (the same kernel, so the same bits) instead of keeping rows x rows per head.  A work-group a row was 88% of a
/// block at real bucket sizes (fwd 1.45 s + bwd 4.45 s); the products are milliseconds (card 5).
void mf_attn_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t heads,
                 int64_t head_dim, uint16_t* o16, float* o32, float* lse, prefill::Gemm& gemm, void* stream);
/// dO is [rows][heads * head_dim] FP32 (the head's scale folded out, as the forward's rows are), and dQ, dK, dV are
/// the same shape in FP32.  `accumulate` adds into dK and dV (two contexts' rows of one block share them).
///
/// The row's own output enters as D = dO . O, and it has to be the FP32 O the forward summed, not the FP16 image the
/// next product reads: dS = P (dO V^T - D) is a difference of two quantities of the same size, so the 5e-4 the FP16
/// image costs is not noise here but the whole of the difference - the block's dQ and dK came out 4e-2 off against
/// autograd for that alone.  o32 is the forward's `o32` dump.
void mf_attn_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32,
                 const float* dO, int64_t rows, int64_t heads, int64_t head_dim, float* dQ, float* dK, float* dV,
                 bool accumulate, prefill::Gemm& gemm, void* stream);

/// A product's forward rows: y16[t][c] = y[t][c] / rs[t] with rs a power of two at least 1 that brings each row's
/// largest value under FP16's reach (dit_rows' rows16 rule, so a gradient's rows read the same way), and the
/// transposed copy the next op's wgrad wants.
void mf_rows16(const float* y, int64_t rows, int64_t cols, uint16_t* y16, float* rs, void* stream);


/// dX[T,K] = dY . W read along its rows through wt[K,N] (the transposed FP16 view of W[N,K]), the rows' scales rsY
/// (null: 1) put back, added into dX or written over it.  scratch holds T * K FP32 sums.
void mf_dgrad(const uint16_t* dy16, const float* rsY, const uint16_t* wt, int64_t T, int64_t N, int64_t K, float* dX,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream);

/// dW[N,K] = sum_t dY[t,n] X[t,k], from the two transposed FP16 views (dy16t [N,T], x16t [K,T]) times gscale.
void mf_wgrad(const uint16_t* dy16t, const uint16_t* x16t, int64_t T, int64_t N, int64_t K, float gscale, float* dW,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream);

/// db[N] += sum over the rows of dY (FP32).
void mf_bias_bwd(const float* dy, int64_t T, int64_t N, float* db, void* stream);

}  // namespace strata::image
