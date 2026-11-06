// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_dit.hpp - Mage-Flow's DiT on the GPU: a two-stream MMDiT (12 blocks of 3072, 24 heads of
// 128) over the image's latent tokens [h w][128] and the text encoder's states, the velocity of a flow-matching step.
//
// Each block modulates the image and the text stream by the time (its own six shift / scale / gate rows for each),
// projects them to q, k and v with each stream's own weights, runs one attention over [text, image] of a context (no
// mask: the text rows see the image and the other way round; the image rows rotated by the 2D multi-scale RoPE, the
// text rows not), and adds each stream's projection and MLP (GELU, tanh form) back through its gates.  The text stream
// changes with the time, so nothing of it is kept between steps beyond its input projection.
//
// The weights come from the bf16 safetensors (Comfy-Org/Mage-Flow's files as they are).  A file with a shared low-rank
// modulation (MageTrail's, bluvoll's mageflow-lowrank-modulation-v1) projects SiLU(temb) once a step to a few hundred
// values (modulation_down) and each block's modulations from those: its FP32 factors stay FP32, as the format asks.
// With q8 each block is quantized to 8 bits on the host as it is read, once, and the file's pages are dropped from the
// page cache: half the VRAM and the bytes a streamed block moves.  Where the GPU's library takes int8 products (prefill::Gemm::i8_usable) the
// weights are rotated in groups and quantized with a scale a row, and so are the activations a token at each product
// (W8A8, qi_rows_i8: ComfyUI's int8 form, the rotation spreading the outlier channels below); elsewhere as Q8_0, the
// products dequantizing it to FP16 each time.  Without q8 the bf16 bytes
// go from the mapped file to the GPU as they are and are rewritten there to FP16 in place (the same two bytes, the few
// values past FP16's range saturated), which the products take as they are: no dequantization, the matrix units fed
// straight (MfPipeline takes this when the blocks fit in VRAM).  The products take FP16 activations (rows scaled into
// FP16's range, mf_dit.cpp): not int8 activations a scale each 32 (iq_mmq_q8_rows'), which the image
// stream's outlier channels (1e5 among values of 1) cost too much (RTX 4070, Turbo's 4 steps from the same noise: the
// final latent's cosine to the reference 0.955 with them, 0.992 with FP16 on the B70, 0.996 in bf16).  The blocks go to
// the GPU through a LayerStream: those that fit stay, the rest are streamed.
//
// Guidance: up to two contexts (the prompt and the negative prompt) run through each product together, the image's
// rows twice, the weights read once a step; the attention one context at a time.
//
// Edits (the edit models): the references' latent tokens (clean, the same every step) follow the target's rows in
// each context, [target, reference 1, ..., reference N], and take part in every block; each image's rows are rotated
// by its own size's centred positions and by its frame (0 the target, j reference j), and only the target's rows
// give the velocity.
//
// Adapters (image_lora.hpp): a request's files are bound to the weights' products (set_loras) and their delta added to
// the product's output there (y += strength (x A^T) B^T), in the input's own units as the readers take it: the whole
// weight, every block's, and the ones outside the blocks.  A weight with an adapter keeps its sums in FP32 (a delta
// cannot be added to an int32 sum) and its rows' scale is the delta's input's, as the plain product's is.
#pragma once

#include "strata/image/guidance.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

/// W values side by side, aligned to their size: a work-item's load or store of them in one access (vec<T, W> of
/// SYCL, whose header this one does not take)
template<class T, int W> struct alignas(sizeof(T) * W) Packed {
    T v[W];
};

/// The rows of a block's run that take their modulation from another set, alt floats past the step's (0 for every row:
/// none): [r0, e0) and [r1, ...) of the run, read t0 rows on (MossDit's references in the stream: the last rows of
/// each context's, at timestep 0).
struct AltRows {
    int64_t r0 = INT64_MAX, e0 = 0, r1 = INT64_MAX, t0 = 0, alt = 0;
    int64_t at(int64_t t) const {
        t += t0;
        return (t >= r0 && t < e0) || t >= r1 ? alt : 0;
    }
    AltRows from(int64_t rows) const {
        AltRows r = *this;
        r.t0 += rows;
        return r;
    }
};

/// A product's output [T][C] as its readers take it: FP32 sums (f) times the rows' scales (rs, null: 1), or a W8A8
/// product's int32 sums (i) times the activations' scales (sx, a row) and the weights' (sw, a column); plus the bias
/// (b, null: none).
struct MfOut {
    const float* f = nullptr;
    const int32_t* i = nullptr;
    const float *rs = nullptr, *sx = nullptr, *sw = nullptr, *b = nullptr;
    int64_t C = 0;
    float at(int64_t t, int64_t c) const {
        const float v = i != nullptr ? (float) i[t * C + c] * sx[t] * sw[c]
                                     : (rs != nullptr ? f[t * C + c] * rs[t] : f[t * C + c]);
        return b != nullptr ? v + b[c] : v;
    }
    /// at(t, c0 + d) as r(d), the row's offsets taken once (the per-head kernels' reads: qk_norm_rope.hpp)
    auto row(int64_t t, int64_t c0) const {
        const int64_t o = t * C + c0;
        const int32_t* ip = i != nullptr ? i + o : nullptr;
        const float* fp = f != nullptr ? f + o : nullptr;
        const float sxt = i != nullptr ? sx[t] : 0.f, rst = rs != nullptr ? rs[t] : 1.f;
        const float* swp = sw != nullptr ? sw + c0 : nullptr;
        const float* bp = b != nullptr ? b + c0 : nullptr;
        const bool scaled = rs != nullptr;
        return [=](int64_t d) {
            const float v = ip != nullptr ? (float) ip[d] * sxt * swp[d] : (scaled ? fp[d] * rst : fp[d]);
            return bp != nullptr ? v + bp[d] : v;
        };
    }
    /// at(t, c + p) for p below W, read W side by side (c and C multiples of W)
    template<int W> void at_w(int64_t t, int64_t c, float* o) const {
        if (i != nullptr) {
            const Packed<int32_t, W> iv = *reinterpret_cast<const Packed<int32_t, W>*>(i + t * C + c);
            const Packed<float, W> wv = *reinterpret_cast<const Packed<float, W>*>(sw + c);
            const float st = sx[t];
            for (int p = 0; p < W; ++p) o[p] = (float) iv.v[p] * st * wv.v[p];
        } else {
            const Packed<float, W> fv = *reinterpret_cast<const Packed<float, W>*>(f + t * C + c);
            const float r = rs != nullptr ? rs[t] : 1.f;
            for (int p = 0; p < W; ++p) o[p] = rs != nullptr ? fv.v[p] * r : fv.v[p];
        }
        if (b != nullptr) {
            const Packed<float, W> bv = *reinterpret_cast<const Packed<float, W>*>(b + c);
            for (int p = 0; p < W; ++p) o[p] += bv.v[p];
        }
    }
    /// the rows from r on
    MfOut from(int64_t r) const {
        MfOut o = *this;
        if (f != nullptr) o.f += r * C;
        if (i != nullptr) o.i += r * C;
        if (rs != nullptr) o.rs += r;
        if (sx != nullptr) o.sx += r;
        return o;
    }
};

class MappedFile;
class SafeTensors;

class MfDit {
public:
    MfDit();
    ~MfDit();
    MfDit(const MfDit&) = delete;
    MfDit& operator=(const MfDit&) = delete;

    /// `q8`: the blocks as Q8_0 (the default), else FP16 on the GPU from the file's bf16.  `dry`: read for its sizes
    /// only.  `reuse`: the blocks' host bytes an earlier open of the same file in the same form left (take_host),
    /// taken instead of quantizing the file again.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool q8 = true,
              std::vector<std::vector<uint8_t>>* reuse = nullptr);
    /// The blocks' host bytes (8 bits: their quantized weights and FP32 vectors), handed over for a later open's
    /// `reuse`: the part is closed afterwards.
    std::vector<std::vector<uint8_t>> take_host() { return std::move(host_); }

    /// Context i (0: the prompt, 1: the negative prompt) from the text encoder's states [rows][2560] (host FP32, the
    /// rows after the system turn): their norm and input projection, kept on the GPU.
    void set_context(int i, const float* states, int64_t rows);
    int64_t context_rows(int i) const { return ctx_[i].rows; }

    /// An edit's references: their latent tokens z (device FP32, [sum of h w][128], reference by reference) and their
    /// sizes in latent positions (h, w); none (the default) for a generation.
    void set_refs(const float* z, const std::vector<std::pair<int64_t, int64_t>>& sizes);
    /// The rows the references add to each context's image rows.
    int64_t ref_rows() const { return ref_rows_; }

    /// The velocity [n][h w][128] (device FP32) of the latent tokens x [h w][128] (device FP32) at noise level sigma,
    /// for the contexts 0 .. n - 1 (n 1 or 2).
    void forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out);

    /// Sampling: x [h w][128] (device FP32, the initial noise) stepped through sigmas (descending, the last 0) by
    /// Euler; with cfg > 1 and a negative context, the velocity neg + cfg (pos - neg).  step(i, n) after each step:
    /// false stops (QiStopped).  `after` (if set) is given x and the step's new sigma first, on the sampler's stream
    /// (a mask: img_start.hpp's blend).
    /// `peek` (if set) is called after each step with the steps done and a function that writes the step's x0, the
    /// sampler's prediction of the finished latent (x - sigma * v), into a buffer of x's size: a partial image's
    /// latent, on the sampler's stream; the sampling itself is not changed.
    /// `guide` (if set and on): the guidance of guidance.hpp, the steps then going by its denoised values.
    void sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                const std::function<bool(int, int)>& step = {},
                const std::function<void(float*, float)>& after = {},
                const std::function<void(int, const std::function<void(float*)>&)>& peek = {},
                const GuidanceSpec* guide = nullptr);
    /// forward of context 0 alone with `blocks` skipped (their image and text rows past them as they came: ComfyUI's
    /// double_block replaced by its input): skip layer guidance's pass.
    void forward_skip(const float* x, float sigma, int64_t h, int64_t w, const std::vector<int>& blocks, float* out);

    int64_t hidden() const { return C_; }
    /// The time of `rows` image rows through the MLP's first product (the hidden width to the MLP's, a block's largest)
    /// in each of the two forms on this stream's GPU, the best of three after one: {FP16 weights and activations, FP32
    /// sums; W8A8: the activations rotated and quantized, int32 sums scaled} in seconds, 0 for a form the GPU's library
    /// does not take (W8A8: prefill::Gemm::i8_usable) or the memory for it short.  Needs only the file's sizes (dry).
    std::pair<double, double> time_forms(int64_t rows) const;
    int64_t latent_channels() const { return in_ch_; }
    bool q8() const { return q8_; }
    int64_t text_dim() const { return txt_dim_; }
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    int blocks() const { return (int) blocks_.size(); }
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    void drop_blocks() { blocks_stream_.drop(); }
    void drop_block_homes() { blocks_stream_.drop_homes(); }
    /// The adapters folded into the blocks' weights where they are plain LoRAs (the default), else added to each
    /// product: false keeps the weights as the file has them.
    void set_lora_bake(bool on) { lora_bake_ = on; }
    uint64_t weight_bytes() const { return weight_bytes_; }   // what stays on the GPU besides the blocks
    /// The working buffers n contexts of hw image rows (an edit's references' included) take (text rows up to
    /// max_text a context), with the MLP a band of mlp_rows image rows at a time (0: all).
    uint64_t activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows = 0) const;
    void set_mlp_rows(int64_t rows) { mlp_cap_ = rows; }
    uint64_t activations_held() const {
        return cap_hw_ > 0 ? activation_bytes(cap_hw_, (int) cap_n_, cap_text_, mlp_rows_) : 0;
    }
    void drop_activations();

    /// An adapter as a file names it: a weight of the DiT, the rows of the weight a file's entry changes, and the rows
    /// of the entry they are (one name a weight here: no weight is fused or split).
    struct LoraTarget {
        std::string name;
        int64_t row0 = 0, rows = 0, cols = 0, src_row = 0;
    };
    /// The weights adapters may change: what a file's keys are matched against (the probe's map, the server's list).
    std::vector<LoraTarget> lora_targets() const;

    /// The adapters the next steps apply, each with its strength, bound to the weights' products: a key the DiT does
    /// not take (or a shape that is not its weight's, or a form the engine does not apply) is false with err, and
    /// nothing bound.  The contexts' rows are made from the other adapters' weights, so set_context again before the
    /// next forward.  The same set again (its files and strengths) changes nothing.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    /// The set bound, as its files' paths and strengths (empty: none).
    const std::string& lora_key() const { return lora_key_; }
    /// The bound entries' A and B on the GPU.
    uint64_t lora_bytes() const;
    /// The adapter files bound (a set whose strengths are all 0 binds none).
    int lora_count() const { return (int) lora_files_.size(); }
    void drop_loras();

    /// The image and text streams after block 0 of the last forward's context 0 (device FP32 [h w][C], [rows][C]), for
    /// the probe (kept when keep_block0(true)).
    void keep_block0(bool on) { keep0_ = on; }
    const float* block0_img() const { return b0_img_; }
    const float* block0_txt() const { return b0_txt_; }

private:
    void sample_guided(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                       const std::function<bool(int, int)>& step, const std::function<void(float*, float)>& after,
                       const std::function<void(int, const std::function<void(float*)>&)>& peek,
                       const GuidanceSpec& spec);

    bool lora_bake_ = true, baked_any_ = false;
    float* bake_tmp_ = nullptr;      // a weight's delta [n][k], while adapters are baked
    uint16_t* bake_at16_ = nullptr;  // an adapter's A^T
    void install_prepare();
    struct W {
        int type = 0;            // 8 Q8_0, 1 F16, 32 F32 (the low-rank modulations)
        int64_t n = 0, k = 0;    // n rows of k
        const void* d = nullptr; // a block's: its offset in the block's region; else a device pointer
        uint64_t row_bytes = 0;
        LoraLayer* lora = nullptr;   // the adapters' deltas for its products (null: none: the product is unchanged)
        bool baked = false;          // its adapters folded into the block's copy of it (set_loras), not added to products
    };
    struct Block {
        W imod, tmod, q, k, v, tq, tk, tv, o, to, imlp1, imlp2, tmlp1, tmlp2;
        uint64_t b_imod = 0, b_tmod = 0, b_q = 0, b_k = 0, b_v = 0, b_tq = 0, b_tk = 0, b_tv = 0, b_o = 0, b_to = 0,
                 b_imlp1 = 0, b_imlp2 = 0, b_tmlp1 = 0, b_tmlp2 = 0;   // the FP32 biases' offsets
        uint64_t nq = 0, nk = 0, ntq = 0, ntk = 0;                     // the FP32 q / k norms' offsets
    };
    /// A weight and the rows of a file's entry it takes (Mage-Flow's DiT has no fused or split weight: one name is one
    /// weight, all of them with a bias, which an entry's bias's delta is added beside).
    struct LoraSlot {
        std::string name;
        W* w = nullptr;
        int64_t row0 = 0, rows = 0, src_row = 0;
    };
    struct Ctx {
        int64_t rows = 0;
        float* txt0 = nullptr;   // [rows][C]: txt_in(txt_norm(states))
    };
    // y [T][w.n] = (x16 [T][w.k], rows scaled down by rs) . w^T, the rows times rs again, plus bias (rs, bias null:
    // none).  same_input: x16 is the previous product's, which was W8A8 too (its int8 input is taken again).  defer:
    // y is left the product's sums, for the reader to scale and add the bias to as the view says.
    MfOut product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                  const float* bias, bool same_input = false, bool defer = false);
    // a block's modulation y [w.n] = SiLU(temb) . w^T + bias (tsil: its FP16 copy, for a 16-bit w); an F32 w (the
    // low-rank form) reads the shared projection mlow_ instead
    void modulation(const uint16_t* tsil, const W& w, const uint8_t* base, float* y, const float* bias);
    void build_lora_targets();
    void lora_scratch(int64_t T);
    /// A DoRA's row norms: the norms of a slot's weight, one a row of it, from the file's tensor, cached a tensor
    /// (open drops the cache).
    bool dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err);
    /// One of the file's tensors' row norms, FP32, cached (`name`: the file's own name).
    bool tensor_norms(const std::string& name, const std::vector<float>*& out, std::string& err);
    void ensure(int64_t hw, int n, int64_t max_text);
    void rope_tables(int64_t h, int64_t w);
    void drop_refs();
    void time_embedding(float sigma);

    std::unique_ptr<MappedFile> file_;
    std::unique_ptr<SafeTensors> st_;   // the file's header (a DoRA's row norms read its tensors)
    std::vector<std::vector<uint8_t>> host_;   // the blocks' Q8_0 (and every block's FP32 vectors), when not in place
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false, q8_ = true, w8_ = false, keep0_ = false;
    const std::vector<int>* skip_ = nullptr;   // forward_skip's blocks
    int64_t C_ = 3072, F_ = 12288, heads_ = 24, hd_ = 128, in_ch_ = 128, txt_dim_ = 2560;
    std::vector<int64_t> axes_{16, 56, 56};
    // the parts outside the blocks, FP16 on the GPU
    W img_in_, txt_in_, t1_, t2_, nout_, pout_;
    W mdown_;   // the shared low-rank modulation's projection (F32, device; n 0 without one)
    float *b_mdown_ = nullptr, *mlow_ = nullptr;
    const float *b_img_in_ = nullptr, *b_txt_in_ = nullptr, *b_t1_ = nullptr, *b_t2_ = nullptr, *b_nout_ = nullptr,
                *b_pout_ = nullptr, *txt_norm_ = nullptr;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    Ctx ctx_[2];
    // activations, grown on demand
    int64_t cap_hw_ = 0, cap_n_ = 0, cap_text_ = 0, mlp_cap_ = 0, mlp_rows_ = 0, dirty_ = 0;
    float *img_ = nullptr, *txt_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr,
          *gu_ = nullptr, *mod_i_ = nullptr, *mod_t_ = nullptr, *temb_ = nullptr, *nmod_ = nullptr;
    float *tq_ = nullptr, *tk_ = nullptr, *tv_ = nullptr;   // the text rows' q, k, v
    // the FP16 inputs' row scales (rs_ the current product's, rs2_ the text rows' q, k, v's and the MLP's second's
    // while rs_ is still read, rs_io_ / rs_to_ the attention's outputs by stream), the
    // attention's v heads' scales (hs_, a context's heads) and their rows' maxima
    float *rs_ = nullptr, *rs2_ = nullptr, *rs_io_ = nullptr, *rs_to_ = nullptr, *hs_ = nullptr, *vpart_ = nullptr;
    uint16_t *h16_ = nullptr, *jq16_ = nullptr, *jk16_ = nullptr, *jv16_ = nullptr, *jo16_ = nullptr,
             *io16_ = nullptr, *to16_ = nullptr, *t16_ = nullptr;
    float *cos_ = nullptr, *sin_ = nullptr, *cos_id_ = nullptr, *sin_id_ = nullptr;
    // W8A8's input (w8_): xq8_rows_ rows of up to xq8_cols_ int8 values, then two arrays of their scales, taken in
    // turn (xs_next_), so that the deferred sums of one input keep theirs while the next input is quantized
    int8_t* xq8_ = nullptr;
    int64_t xq8_rows_ = 0, xq8_cols_ = 0;
    int xs_next_ = 0;
    float *b0_img_ = nullptr, *b0_txt_ = nullptr;
    int64_t rope_h_ = -1, rope_w_ = -1, rope_refs_ = -1;
    // an edit's references
    float* refs_ = nullptr;   // [ref_rows_][in_ch_]
    int64_t ref_rows_ = 0, refs_gen_ = 0;
    std::vector<std::pair<int64_t, int64_t>> ref_sizes_;
    float* xin_ = nullptr;    // the image rows' input [hw + ref_rows_][in_ch_]
    // the adapters bound
    std::vector<LoraSlot> lora_slots_;
    std::vector<std::unique_ptr<LoraLayer>> lora_layers_;
    std::vector<std::shared_ptr<const LoraFile>> lora_files_;
    std::string lora_key_;
    int64_t lora_cols_ = 0, lora_tmp_t_ = 0;   // the deltas' scratch's columns and rows (the largest lora_scratch_cols)
    std::map<std::string, std::vector<float>> norm_cache_;   // DoRA's row norms, a file's tensor (open clears them)
    float* lora_f32_ = nullptr;
    uint16_t* lora16_ = nullptr;
    // a GLoRA's own path: its input's columns, the weight's rows it is added over, and the buffers a band of its
    // products takes (the rows' table, a W8A8 weight's int8 input and its scales)
    int64_t lora_glora_cols_ = 0, lora_glora_rows_ = 0;
    float* lora_glora_y_ = nullptr;
    int8_t* lora_glora_x8_ = nullptr;
    float* lora_glora_sx_ = nullptr;
    void lora_glora_scratch();   // allocates them, once a set with a GLoRA is bound
    void free_glora_scratch();
    /// A GLoRA's own path: the weight's rows [r0, r0 + n) product of v (nt tokens, the first of them t0), added to
    /// the product's output y at those rows (the weight's own path, as product takes it).
    void glora_weight(const uint16_t* v, const W& w, const uint8_t* d, int64_t r0, int64_t n, int64_t nt, int64_t t0,
                      float* y);
    /// The layer's GLoRAs on the product's T rows of y (none: nothing), through glora_weight.
    void add_glora(const LoraLayer* lora, const uint16_t* x16, const float* rs, const W& w, const uint8_t* d,
                   int64_t T, float* y);
};

}  // namespace strata::image
