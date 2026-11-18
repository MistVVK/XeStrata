// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_ops.hpp - the parts of an image model's training step that every DiT's takes, on the GPU
// (src/image/train_ops.cpp): a product's gradients, an adapter's (a plain LoRA's, a LoKr's, a DoRA's), the flow
// matching loss and the transposes they read.  The forward's own parts are the inference's (dit_rows.hpp's row
// kernels, kernels::qk_norm_rope, qi_attention_train), and their backwards sit beside them (dit_rows.hpp,
// qk_norm_rope.hpp, qi_attention_bwd); a model's own block (mf_train.hpp's, Mage-Flow's) is put together out of these.
//
// The gradients themselves are FP32.  One product Y[T,N] = X[T,K] . W[N,K]^T + b has
//
//   dX[T,K] = dY[T,N] . W[N,K]        (dgrad, W read along its rows: the caller's transposed view wt[K,N])
//   dW[N,K] = dY[T,N]^T . X[T,K]      (wgrad)
//   db[N]   = sum_t dY[t,N]           (bias_bwd)
//
// and prefill::Gemm knows only Y = X . W^T with W row-major, so the training side keeps a transposed FP16 view of every
// weight it takes gradients through (built when a block is opened for training) and reads a gradient's rows transposed
// too (transpose_f16, scaled_transpose_f16).  A row's or head's power of two cancels exactly, except in dW, which sums
// over the rows: there the row scales are folded into the FP32 gradient first and one global power of two (G, from
// scaled_transpose_f16) is put back on the sum.
//
// An adapter's gradients, one weight, with s the request's strength (training takes 1) and k = alpha / rank:
//
//   plain LoRA   h = x A^T, delta = s k (h B^T)   dB = s k dY^T h   dA = s k (dY B)^T x
//   LoKr         delta = s k kron(w1, w2_a w2_b)   each factor by the partial trace of the delta's gradient
//   DoRA         PEFT's: n_r = ||W_r + k (BA)_r||, the composed weight's row norm, recomputed every forward with
//                the factors detached (W frozen, A and B the last step's); c_r = m_r / (n_r + eps); the rows'
//                factors a_r = c_r and b_r = k c_r (s = 1), and
//                dm_r = sum_t dY[t,r] (p[t,r] + k d[t,r]) / (n_r + eps)   (p the base product, d the delta's)
//
// n_r is made without the composed weight: n_r^2 = |W_r|^2 + 2k sum_j B_rj (W A^T)_rj + k^2 B_r (A A^T) B_r^T
// (dora_composed_norms; |W_r| once at open, the two products each forward).  eps FP32's.  Inference reads the same
// definition (image_lora.cpp's lora_dora_rows, dit_dora_norms at the bind).
//
// The step is rectified flow (docs/LORA_TRAIN.md, "Timestep sampling and loss"): x_sigma = (1 - sigma) x0 + sigma eps,
// the target v = eps - x0, the loss the mean of (v_pred - v)^2, so dL/dv_pred = 2 (v_pred - v) / n.
//
// Checked per op against NumPy and one Mage-Flow block against torch (tools/image/mf_train_ref.py,
// tools/image/mf_train_torch_ref.py).
#pragma once

#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::image::train {

/// One step's flow loss: the mean of (v_pred - (eps - x0))^2 over n values, returned on the host, with dv_pred the
/// gradient 2 (v_pred - v) / n.  dx0 and deps (null: not taken) are that loss's way back to the latent and the noise
/// through the target; the noised input x_sigma = (1 - sigma) x0 + sigma eps is the step's own argument.
float flow_loss(const float* v_pred, const float* x0, const float* eps, int64_t n, float* dv_pred, float* dx0,
                float* deps, void* stream);

/// [rows][cols] FP16 -> [cols][rows] FP16, both row-major (32 x 32 tiles through local memory).
void transpose_f16(const uint16_t* in, int64_t rows, int64_t cols, uint16_t* out, void* stream);

/// A gradient's rows read by a dW: out[c * rows + t] = (g[t,c] * rsX[t]) / G as FP16, G one power of two over the
/// whole tensor (dit_rows' pow2_full: the largest value into [2^13, 2^14)); the stored X rows are divided by their own
/// rsX, so the factors cancel and only G is left to put back.  G stays on the device: *gbits (a device word) gets the
/// largest value's bits, which wgrad reads G from, so nothing waits for the host.
void scaled_transpose_f16(const float* g, const float* rsX, int64_t rows, int64_t cols, uint16_t* out, uint32_t* gbits,
                          void* stream);
/// The G a scaled_transpose_f16 used, read back (a check's).
float scale_of(const uint32_t* gbits, void* stream);
/// A device word for scaled_transpose_f16's G, one of two a stream (k 0 or 1: a LoRA's backward holds two at once),
/// allocated once: the stream runs in order, so a word is free again when the next call's write to it runs.
uint32_t* scale_word(int k, void* stream);

/// dX[T,K] = dY . W read along its rows through wt[K,N] (the transposed FP16 view of W[N,K]), the rows' scales rsY
/// (null: 1) put back, added into dX or written over it.  scratch holds T * K FP32 sums.
void dgrad(const uint16_t* dy16, const float* rsY, const uint16_t* wt, int64_t T, int64_t N, int64_t K, float* dX,
           float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream);
/// The same through W's rows [N][K] as they are (Gemm::f16_batch_ex, which the caller checked takes the GPU).
void dgrad_rows(const uint16_t* dy16, const float* rsY, const uint16_t* w, int64_t T, int64_t N, int64_t K, float* dX,
                float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream);

/// dW[N,K] = sum_t dY[t,n] X[t,k], from the two transposed FP16 views (dy16t [N,T], x16t [K,T]) times mul and the G
/// of gbits (scaled_transpose_f16's; null: 1).
void wgrad(const uint16_t* dy16t, const uint16_t* x16t, int64_t T, int64_t N, int64_t K, const uint32_t* gbits,
           float mul, float* dW, float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream);

/// db[N] += sum over the rows of dY (FP32).
void bias_bwd(const float* dy, int64_t T, int64_t N, float* db, void* stream);

/// y[t][c] *= r[c] (in place): a DoRA's factors of the weight's rows (dora_bwd's da/db) on a gradient's columns.
void scale_cols(float* y, const float* r, int64_t rows, int64_t cols, void* stream);

/// A plain LoRA's factors, backwards, with s the request's strength and k = alpha / rank folded into `scale`.  The
/// forward kept h16 [T][rank] (the input through A) and the weight's own product read x16 [T][cols]; dY is the
/// gradient reaching the weight's rows [T][rows] in FP32.  dB [rows][rank] = scale dY^T h and
/// dA [rank][cols] = dh^T x read dh = scale (dY B), the intermediate this same call's scratch starts with (dL/dh,
/// so dA takes no second scale); dh's own rows and scale take the product's own dgrad back to the input.
/// h16t and x16t are the transposed views the forward kept (transpose_f16 of the rows it wrote).  dy16_done and
/// rsY_done (null: made here) are dY's FP16 rows and their scales where the caller has them already (its dgrad's).
/// With h16 and x16, the rows themselves, and a GPU f16_batch_ex takes (Gemm::f16_batch_ex_ok), dB and dA are products
/// that read dY's and dh's rows transposed and nothing is transposed: h16t and x16t may be null then, and bt16 too
/// with B's own rows b16 [rows][rank] (dh's product reads them as they are).
void lora_bwd(const float* hr, const float* xr, const uint16_t* h16t, const uint16_t* x16t, const uint16_t* bt16,
              const float* dY, int64_t T, int64_t rows, int64_t cols, int64_t rank, float scale, float* dB, float* dA,
              float* scratch, uint16_t* tmp16, float* rsT, prefill::Gemm& gemm, void* stream,
              const uint16_t* dy16_done = nullptr, const float* rsY_done = nullptr, const uint16_t* h16 = nullptr,
              const uint16_t* x16 = nullptr, const uint16_t* b16 = nullptr);

/// A LoKr's factors, backwards.  The delta is kron(w1, w2_a w2_b) and the step's gradient of the whole weight
/// (dDelta [out][in], out = out_l * out_k, in = in_m * in_n) is what a wgrad of the delta's rows gives; each factor
/// is read through it by a partial trace:
///   dw1[l][m] = sum_{kn} dDelta[l * out_k + k][m * in_n + n] w2[k][n]
///   dw2[k][n] = sum_{lm} dDelta[l * out_k + k][m * in_n + n] w1[l][m]
/// and then w2 = w2_a w2_b splits as a product does.
void lokr_partial_trace(const float* dDelta, const uint16_t* w1, const uint16_t* w2, int64_t out_l, int64_t out_k,
                        int64_t in_m, int64_t in_n, float* dw1, float* dw2, void* stream);
/// dw2 [out_k][in_n] into its two factors: dw2a[o][r] = sum_n dw2[o][n] w2b[r][n] and
/// dw2b[r][n] = sum_o w2a[r][o] dw2[o][n] (the same products the forward reads, in FP32 here - the factors are small).
void lokr_factor_bwd(const float* dw2, const uint16_t* w2a, const uint16_t* w2b, int64_t out_k, int64_t in_n,
                     int64_t rank, float* dw2a, float* dw2b, void* stream);

/// A DoRA's magnitude m [rows], backwards, with the composed row norms n_r = ||W_r + scale (BA)_r|| the forward
/// divided by (dora_composed_norms) and eps FP32's, at strength 1 (PEFT's): dm[r] = sum_t dY[t,r] (p[t,r] +
/// scale d[t,r]) / (n_r + eps), where p is the base product's rows and d the delta's (the norm detached: no
/// gradient through it).  `da` and `db` (null: not taken) get the factors the two paths' own gradients are read by:
/// da[r] = c_r and db[r] = scale c_r (c_r = m_r / (n_r + eps)), so the caller's dgrad and wgrad see dY da and dY db.
void dora_bwd(const float* dY, const float* p, const float* d, const float* m, const float* norms, int64_t T,
              int64_t rows, float scale, float* dm, float* da, float* db, void* stream);

/// One product's adapter as a training step holds it (plain LoRA, DoRA or LoKr): the FP16 views the products read,
/// written from the FP32 masters after each optimizer step, and the FP32 gradients added into.
enum class AdapterMethod : int { lora = 0, dora = 1, lokr = 2 };
struct Adapter {
    AdapterMethod method = AdapterMethod::lora;
    int64_t rank = 0;
    float scale = 0.f;                 // s * alpha / rank, folded (the request's strength times the file's alpha)
    uint16_t* a = nullptr;            // [rank][K] FP16: h = x A^T reads it (written on sync, read in the block)
    uint16_t* at = nullptr;           // [K][rank] the same values transposed (the input gradient's dgrad reads it; null:
                                      // the GPU's transposed products read a's rows, Gemm::f16_batch_ex_ok)
    uint16_t* b = nullptr;            // [N][rank] FP16: delta = h B^T reads it
    uint16_t* bt = nullptr;           // [rank][N] the same values transposed (lora_bwd reads it; null as at)
    float* dA = nullptr;               // [rank][K], added into
    float* dB = nullptr;               // [N][rank], added into
    const float* m = nullptr;         // [N] the DoRA magnitude (null: a plain LoRA)
    const uint16_t* w = nullptr;      // [N][K] FP16 the base weight's rows (DoRA: dora_composed_norms' W A^T reads
                                      // them; bound with the product's own, an int8 base takes no DoRA)
    float* norms0 = nullptr;          // [N] |W_r| at open (DoRA: the composed norm's base term, written once)
    float* norms = nullptr;           // [N] the composed norms ||W_r + scale (BA)_r|| (DoRA: rewritten each forward,
                                      // the same call's backward reads them)
    float* dm = nullptr;               // [N], added into (null: not taken)
    // LoKr (method lokr): delta = s kron(w1, w2a w2b), N = ol * ok, K = im * inn (SPEC's factorization)
    int64_t ol = 0, ok = 0, im = 0, inn = 0;
    uint16_t* w1 = nullptr;           // [ol][im] FP16, whole
    uint16_t* w1t = nullptr;          // [im][ol] transposed view (written on sync, read in the block)
    uint16_t* w2a = nullptr;          // [ok][rank] FP16
    uint16_t* w2b = nullptr;          // [rank][inn] FP16
    float* dW1 = nullptr;              // [ol][im], added into
    float* dW2a = nullptr;             // [ok][rank], added into
    float* dW2b = nullptr;             // [rank][inn], added into
};

/// A DoRA's composed row norms into ad->norms [N], PEFT's ||W_r + scale (BA)_r|| with the factors as the products
/// read them (FP16) and the norm detached: |W_r|^2 from ad->norms0 (once at open), W A^T and A A^T two products of
/// ad->a with ad->w and itself, the FP16 factors widened and summed in FP32 (a library's half GEMM sums in FP16,
/// whose range and rounding pull the norm at large rows).  `wat` holds rank * N and `aat` rank * rank FP32 (the
/// block's adapter scratch).
void dora_composed_norms(const Adapter* ad, int64_t N, int64_t K, float* wat, float* aat, void* stream);

/// out[i] = a[i] * col[i % C] (col null: 1), n elements: a gate's or a norm's per-column factor on rows.
void mul_cols(const float* a, const float* col, int64_t n, int64_t C, float* out, void* stream);
/// a[i] *= k[0], or /= with `divide` (k a device scalar: v's power of two).
void scale_by(float* a, int64_t n, const float* k, bool divide, void* stream);
/// v's one power of two over its rows (two arrays: a two-stream DiT's streams; n1 0 for one stream), into vs[0], and
/// every one of `rows` rows' scale set to it (vs[1 + r], the output projection's input row scales): the attention
/// mixes the rows head by head, so they share it.  Mage-Flow's v passes FP16's 65504 at a high sigma (75176 in block
/// 0 at sigma 1.0); inference scales it the same way (head_scale).  amax is a zeroed device word the rows' absolute
/// maximum lands in, as bits (non-negative floats order as their bits).
void v_scale(const float* v0, int64_t n0, const float* v1, int64_t n1, uint32_t* amax, float* vs, int64_t rows,
             void* stream);

/// FP32 rows copied, added into (a += b) and narrowed to FP16, n elements: the blocks' gradients moving between roles.
void copy_rows(const float* a, int64_t n, float* out, void* stream);
void add_into(float* a, const float* b, int64_t n, void* stream);
void f16_copy(const float* x, int64_t n, uint16_t* y, void* stream);
/// The same narrowing at one power of two over the whole tensor (pow2_full: the largest value into [2^13, 2^14)),
/// its bits left in *gbits (a device word, scale_word's) for the caller to put back after the next product.  A
/// gradient's rows already sit near 2^14 (rows16's full range), so a product over a factor's rows can pass FP16's
/// 65504: the plain copy overflowed there (a LoKr's backward at Mage-Flow's mlp1, out_k 128).
void f16_copy_full(const float* x, int64_t n, uint16_t* y, uint32_t* gbits, void* stream);
/// What a trainer's arena (its blocks' scratch, `held` bytes now) should still grow by for a step of `rows`: in
/// proportion to the most rows a step has had (`seen`, 0: none yet), 10% over, less what it holds; `guess` before a
/// step has shown it.  The weights' budgets and the inputs' form read it.
inline uint64_t scratch_growth(uint64_t held, int64_t rows, int64_t seen, uint64_t guess) {
    if (seen <= 0) return guess;
    const uint64_t want = (uint64_t) (1.1 * (double) held * (double) (rows > seen ? rows : seen) / (double) seen);
    return want > held ? want - held : 0;
}

/// BF16 rows kept: x rounded to BF16 in place and its bits stored in y (what is read back is what was used), and read
/// back into FP32, n elements: a trainer's block inputs held at half the memory where the device is short of it.
void bf16_keep(float* x, int64_t n, uint16_t* y, void* stream);
void bf16_load(const uint16_t* y, int64_t n, float* x, void* stream);

/// XESTRATA_TRAIN_PROFILE=1: a training block's stages' times, summed over its calls (each mark waits for the GPU, so
/// the sums are the stages' own and the whole runs slower).  begin() at the block's start, mark(stage) after each
/// stage, report() prints the sums and clears them; off, every call returns at once.
class StageTimes {
public:
    explicit StageTimes(const char* name);
    bool on() const { return on_; }
    void begin(void* stream);
    void mark(const char* stage, void* stream);
    void report();

private:
    const char* name_;
    bool on_ = false;
    int64_t calls_ = 0;
    double last_ = 0.0;
    std::vector<std::pair<const char*, double>> t_;
};

/// One-row products in FP32 (a modulation linear's LoRA, a low-rank modulation): h = A x (A [rank][K]); y += scale B
/// (A x) (B [N][rank], h the rank's scratch); and that LoRA's gradients from dy: dB += scale dy h^T, dA += g x^T with
/// g = scale B^T dy (g written, the rank's).
void row_down(const float* A, const float* x, int64_t rank, int64_t K, float* h, void* stream);
void row_lora_fwd(const float* A, const float* B, const float* x, float scale, int64_t rank, int64_t N, int64_t K,
                  float* y, float* h, void* stream);
void row_lora_bwd(const float* A, const float* B, const float* x, const float* dy, float scale, int64_t rank,
                  int64_t N, int64_t K, float* dA, float* dB, float* h, float* g, void* stream);

/// One frozen product a block takes gradients through: Y[T,N] = X[T,K] . W^T + b, its weight as the forward reads it
/// (w [N][K] FP16) and transposed (wt [K][N], dgrad's view), and the base's own gradients (null: frozen, not taken).
struct Product {
    int64_t N = 0, K = 0;
    const uint16_t* w = nullptr;
    const uint16_t* wt = nullptr;
    float* dw = nullptr;
    float* db = nullptr;
    /// A QLoRA's base instead of w (w null): ai-toolkit's convrot8, the weight's rows times the regular Hadamard
    /// rotation a group of 256 and then int8 with a scale a row (value = w8 sw); the product's input rotated and made
    /// int8 a token the same way (W8A8).  The backward reads the straight-through weight (the int8 values, rotated
    /// back) through wt, which w8_dequant_t writes.
    const int8_t* w8 = nullptr;
    const float* sw = nullptr;
    /// W's rows [N][K] FP16 the dgrad reads as they are where wt is null (a GPU Gemm::f16_batch_ex takes): w itself,
    /// or an int8 base's straight-through weight (w8_dequant_rows), with no transposed copy made
    const uint16_t* wr = nullptr;
};

/// A weight's rows [N][K] (FP16) as a QLoRA's base: q [N][K] int8 and s [N] (Product::w8, sw).
void w8_quantize(const uint16_t* w16, int64_t N, int64_t K, int8_t* q, float* s, void* stream);
/// The W8A8 product of x16 [T][K] FP16 rows by P's int8 base into y [T][N] FP32 (the rows as x16 holds them, before
/// their row scales): x8 [T][K] and sx [T] its scratch.  Where the GPU takes no int8 products the same values go
/// through FP16 ones (ai-toolkit's path without them): x16f [T][K] and w16f [N][K] FP16 scratch then, else null.
void w8_product(const uint16_t* x16, int64_t T, const Product& P, float* y, int8_t* x8, float* sx, uint16_t* x16f,
                uint16_t* w16f, prefill::Gemm& gemm, void* stream);
/// Whether w8_product takes the int8 products on `stream`'s GPU (else its FP16 scratch is needed).
bool w8_int8(void* stream);
/// An int8 base's weight as FP16 rows [N][K], rotated back: the product an unquantized input takes with it
/// (ai-toolkit's convrot8 when the input needs no gradient: its weight's int8 values, the input as it is).
void w8_dequant_rows(const int8_t* q, const float* s, int64_t N, int64_t K, uint16_t* out, void* stream);
/// The straight-through weight of an int8 base, rotated back and transposed: wt [K][N] FP16 (a dgrad's view), through
/// tmp [N][K] FP16 (w8_dequant_rows, then transposed as transpose_f16 does).
void w8_dequant_t(const int8_t* q, const float* s, int64_t N, int64_t K, uint16_t* tmp, uint16_t* wt, void* stream);

/// What an adapted product's forward keeps for its backward: h = x A^T as FP16 rows (h16 [T][rank]) with their
/// scales (hr [T], the input's own scale folded in) and transposed (h16t [rank][T]).
struct AdKeep {
    uint16_t* h16 = nullptr;
    float* hr = nullptr;
    uint16_t* h16t = nullptr;
};

/// The adapters' transients a block shares between its products (they run one at a time), sized for the largest:
/// hf [T][rank]; y1 and y2 [T][N] (the delta, a DoRA's base and delta rows, a LoKr's maps); fa and fb [N] (a DoRA's
/// row factors); scratch, lora_bwd's (T rank + N rank + rank K); tmp16, T N + T rank FP16 for LoRA, twice T N for
/// DoRA or LoKr; rsT [T].
struct AdScratch {
    float *hf = nullptr, *y1 = nullptr, *y2 = nullptr, *fa = nullptr, *fb = nullptr, *scratch = nullptr;
    uint16_t* tmp16 = nullptr;
    float* rsT = nullptr;
};

/// The largest of the adapters' transients, over the products i < n whose adapter is not null: the rows product i
/// takes (rows_of_product(i)), its width and input width, and the adapter's rank; `plain` while every one is a plain
/// LoRA (its delta takes y1 alone: no y2).
struct AdSizes {
    int64_t rows = 0, n = 0, rank = 0, scratch = 0, tmp = 0, map = 0;
    bool plain = true;
};

template <class Rows>
AdSizes ad_sizes(const Product* const* p, const Adapter* const* ad, int n, Rows rows_of_product) {
    AdSizes s;
    for (int i = 0; i < n; ++i) {
        const Adapter* a = ad[i];
        if (a == nullptr) continue;
        const int64_t T = rows_of_product(i), N = p[i]->N, K = p[i]->K, rk = a->rank;
        s.rows = std::max(s.rows, T);
        s.n = std::max(s.n, N);
        s.rank = std::max(s.rank, rk);
        s.scratch = std::max(s.scratch, T * rk + N * rk + rk * K);
        // A plain LoRA reuses the dgrad's FP16 dY rows. DoRA needs a second set after its row scaling.
        s.tmp = std::max(s.tmp, (a->method == AdapterMethod::lora ? T * N : 2 * T * N) + T * rk);
        s.map = std::max(s.map, T * N);
        if (a->method == AdapterMethod::lokr) {
            s.map = std::max({s.map, T * a->ol * rk, T * a->ol * a->inn, T * a->im * rk,
                              T * a->im * a->ok, a->ok * a->inn,
                              a->ol * a->im + a->ok * a->inn + a->ok * rk + rk * a->inn});
            s.tmp = std::max(s.tmp, T * N + T * a->ol * rk + T * a->inn * a->ol +
                                        a->ok * a->inn + a->ok * rk + rk * a->inn);
        }
        s.plain = s.plain && a->method == AdapterMethod::lora;
    }
    return s;
}

class Pool;

/// AdSizes taken from a block's pool: every buffer null where no product is adapted. A plain LoRA does not take the
/// second map's transient. Mage-Flow's eval on the RX 7900 XTX differed by 1e-6 between layouts with the runtime's
/// timed product selection (2026-11-11); compare with that selection fixed.
AdScratch take_ad_scratch(Pool& pool, const AdSizes& s);

/// One product's rows with its adapter (null: the product as always), the base product's sums already in y [R][N]
/// (still divided by the input's row scales rsx, no bias yet): the scales and the bias put back and the adapter's
/// delta added, what the forward's next stage reads.  keep receives what the backward reads.
/// Not the inference's adapter path (image_lora.cpp's LoraLayer::add): that one adds a frozen adapter of any kind into
/// a product's finished rows in one accumulating product, x A^T rounded to FP16 with the strength folded in; this one
/// keeps h = x A^T as FP16 rows with their own scales for the backward, adds before the row scales and the bias, and
/// reads a DoRA's magnitude and a LoKr's factors as they are trained.
void ad_product_fwd(const Adapter* ad, const uint16_t* x16, const float* rsx, float* y, const float* bias, int64_t R,
                    int64_t N, int64_t K, const AdKeep& keep, const AdScratch& sc, prefill::Gemm& gemm, void* stream);

/// The same product backwards: dYv [R][N] is the gradient reaching its rows (consumed: a DoRA scales it), dxv [R][K]
/// the input's gradient (written, or added into with `accumulate`), the base weight's own dw and db where P has them,
/// and the adapter's factors' gradients.  x16, rsx and xt are the forward's input rows, their scales and their
/// transposed view; t16, trs, t16t and sums the block's own temporaries (R N, R, N R and N K + R K; a LoKr's sums
/// holds 2 N K + R K: its whole delta's gradient and the wgrad's scratch beside it).
/// Whether ad_product_bwd reads the input's transposed rows (xt): a full weight's gradient and a LoKr do, and a LoRA
/// or DoRA where the GPU has no transposed products (Gemm::f16_batch_ex_ok); elsewhere xt may be null.
bool ad_needs_xt(const Product& P, const Adapter* ad, prefill::Gemm& gemm);
/// XESTRATA_TRAIN_PROFILE=1: prints ad_product_bwd's stage times summed since the last report.
void ad_profile_report();
void ad_product_bwd(const Product& P, const Adapter* ad, const uint16_t* x16, const float* rsx, const uint16_t* xt,
                    float* dYv, float* dxv, bool accumulate, const AdKeep& keep, const AdScratch& sc, uint16_t* t16,
                    float* trs, uint16_t* t16t, float* sums, int64_t R, prefill::Gemm& gemm, void* stream);

/// Device memory a block keeps from one call to the next (a training step runs the same block shapes again and
/// again): one buffer, handed out in order and grown when a call needed more.  The stream runs in order, so a call's
/// buffers are free for the next call's once its kernels are queued.
class Arena {
public:
    Arena() = default;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena();
    /// A new call: what the last one handed out is free again; a call that outgrew the buffer gets a larger one.
    void reset(void* stream);
    /// A second call that reads what the last one wrote: its takes, of the same sizes in the same order, get the same
    /// buffers back, untouched (a block's tape kept from its forward for its backward).
    void rewind();
    bool replaying() const { return replay_; }
    void* take(size_t bytes);
    /// The device memory it holds (its buffer, and what the last call took past it).
    size_t held() const { return cap_ > used_ ? cap_ : used_; }
    /// Everything it holds given back (a pause: the next call takes its buffers again).
    void release(void* stream);
    /// Takes from `cap` bytes at `base` (memory of the caller's, not freed here) from now on, instead of its own.
    void attach(void* base, size_t cap);

private:
    uint8_t* base_ = nullptr;
    size_t cap_ = 0, used_ = 0, need_ = 0;
    std::vector<void*> spill_;   // what a call took past the buffer, freed at the next reset
    bool owned_ = true;          // base_ is its own (not attach's)
    std::vector<std::pair<void*, size_t>> taken_;   // what the call since the last reset took, for rewind
    size_t next_ = 0;
    bool replay_ = false;
};

/// A block's device memory for one call, each buffer zeroed on the block's own stream: freed (after the stream is
/// done) when the pool goes, or, with an arena, kept in it for the next call.  The tape and the gradients of one
/// forward and backward.
class Pool {
public:
    /// `replay`: the arena's last call's buffers again (Arena::rewind), not zeroed.
    explicit Pool(void* stream, Arena* arena = nullptr, bool replay = false) : s_(stream), arena_(arena) {
        if (arena_ != nullptr) {
            if (replay) arena_->rewind();
            else arena_->reset(stream);
        }
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool();
    float* f(int64_t n) { return static_cast<float*>(take(n, 4)); }
    uint16_t* u(int64_t n) { return static_cast<uint16_t*>(take(n, 2)); }
    /// The same, not zeroed: a buffer the call writes whole before it reads it (a block's tape is gigabytes, and
    /// zeroing it every call cost more than some of the products did).
    float* rf(int64_t n) { return static_cast<float*>(take(n, 4, false)); }
    uint16_t* ru(int64_t n) { return static_cast<uint16_t*>(take(n, 2, false)); }

private:
    void* take(int64_t n, int64_t bytes, bool zero = true);
    void* s_;
    Arena* arena_ = nullptr;
    std::vector<void*> owned_;
};

/// The bytes a block's tape asks for: take_tape's takes in the same order, without a pool (an arena's 256-byte steps)
struct Sizer {
    size_t bytes = 0;
    void* add(int64_t n, int64_t w) {
        bytes += ((size_t) (n * w) + 255) / 256 * 256;
        return nullptr;
    }
    float* f(int64_t n) { return static_cast<float*>(add(n, 4)); }
    float* rf(int64_t n) { return f(n); }
    uint16_t* u(int64_t n) { return static_cast<uint16_t*>(add(n, 2)); }
    uint16_t* ru(int64_t n) { return u(n); }
};

}  // namespace strata::image::train
