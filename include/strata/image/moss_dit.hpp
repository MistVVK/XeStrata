// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_dit.hpp - MOSS-Image's DiT on the GPU: one stream of 42 blocks (3072 wide, 24 heads of
// 128) over a context's rows, the image's latent tokens [h w][128] and then the text's, the velocity of a
// flow-matching step.
//
// The text encoder's states go through the DiT's text embedder once a context: an input projection and two blocks of
// attention among the text rows alone (no time in them), so the rows they give are the same at every step.  Each of
// the 42 blocks then takes all of a context's rows: RMSNorm scaled and shifted by the time (one modulation the blocks
// share: shift, scale and gate of the attention and of the MLP), q and k RMS-normed a head and rotated by a RoPE of
// four axes, one attention over every row (no mask), and an MLP whose activation is SiTU (4 tanh(g / 4) sigmoid(g)
// times 25 tanh(h / 25)), each added back through its gate.  The RoPE gives a token four coordinates, 16 frequencies
// each (10000^(-k / 16)), pairs (j, j + 64) rotated: an image row (0, the text's length, its row and its column, both
// centred), a text row (0, i, 0, 0).  The last norm and projection take the image's rows alone.
//
// The weights are the diffusers folder's bf16 safetensors (its shards, as they are).  With q8 each block is quantized
// on the host as it is read, as MfDit's are (dit_rows.hpp): W8A8 where the GPU's library takes int8 products
// (prefill::Gemm::i8_usable), else Q8_0; without q8 the bf16 bytes go to the GPU and are rewritten there to FP16 in
// place.  The products take FP16 activations, each row scaled into FP16's range.  The blocks go to the GPU through a
// LayerStream: those that fit stay, the rest are streamed.  The parts outside the blocks (the embedders, the time's
// layers, the shared modulation, the head) are FP16 on the GPU.
//
// Guidance: up to two contexts (the prompt and the negative prompt) run through each product together, a context's
// rows after the other's; a context may skip blocks (the reference pipeline's negative branch skips block 22).
// Skip-layer guidance runs the prompt a third time with its blocks skipped (sample).
//
// The text embedder's weights (0.4 GiB) are needed only by set_context: drop_text gives their VRAM back, and the next
// set_context reads them again from the file.
#pragma once

#include "strata/image/dit_loras.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/pipeline.hpp"
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

class MossFile;


/// The RoPE's tables [rows][hd / 2] as the reference makes them (cos and sin, bf16-rounded): row r's four
/// coordinates from coord(r), each axis hd / 8 frequencies 1 / base^(k / (hd / 8)); the trainer's as the DiT's.
void moss_rope_rows(int64_t rows, int64_t hd, double base, const std::function<void(int64_t, float*)>& coord,
                    std::vector<float>& c, std::vector<float>& s);

/// The time's sinusoid [freq] as the reference makes it: cos then sin of t by freq / 2 frequencies 10000^(-i / (freq /
/// 2)) in FP64, rounded to bf16 (its first layer's type), as FP16: the DiT's and the trainer's.
void moss_time_sinusoid(double t, int64_t freq, std::vector<uint16_t>& e);

class MossDit {
public:
    MossDit();
    ~MossDit();
    MossDit(const MossDit&) = delete;
    MossDit& operator=(const MossDit&) = delete;

    /// `path`: the transformer's folder (its *.safetensors) or one file.  `q8`: the blocks in 8 bits (the default),
    /// else FP16 from the file's bf16.  `dry`: read for its sizes only.  `reuse`: the blocks' host bytes an earlier
    /// open of the same file in the same form left (take_host), taken instead of quantizing the file again.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool q8 = true,
              std::vector<std::vector<uint8_t>>* reuse = nullptr);
    /// The blocks' host bytes (8 bits: their quantized weights and FP32 vectors), handed over for a later open's
    /// `reuse`: the part is closed afterwards.
    std::vector<std::vector<uint8_t>> take_host() { return std::move(host_); }

    /// Context i (0: the prompt, 1: the negative prompt) from the text encoder's states [rows][4096] (host FP32, every
    /// row of the prompt): through the text embedder, kept on the GPU.
    void set_context(int i, const float* states, int64_t rows);
    int64_t context_rows(int i) const { return ctx_[i].rows; }
    /// The blocks context i skips (none by default).
    void set_skip(int i, std::vector<int> blocks) { ctx_[i].skip = std::move(blocks); }

    /// An edit's reference images, as Moss-Image-Edit's adapter was trained to take them (its model card): each a
    /// latent [h w][128] (device FP32, the VAE's encode), on the rotary axes at r = its index + 1, f = 0, its rows and
    /// columns centered.  They go through every block once, at timestep 0
    /// and attending to each other alone, and each block's keys and values of them are kept (their v heads' scales
    /// with them); at every step the rows of each context attend to them as well, their keys before the context's
    /// own.  Called after the adapters are bound (the blocks they go through are the adapted ones); empty: none, the
    /// kept keys and values freed.  `layout` AitkKv: an ai-toolkit adapter's (as ai-toolkit trains it with its
    /// kv_cache on), each context's image and text rows on f moved back by the context's rows (the references keep
    /// f = 0; Moss-Image-Edit's, MossEdit, leave them where a picture without references has them).  AitkNoKv
    /// and Appended, the references in the stream (ai-toolkit's with its kv_cache off; Appended at the image's
    /// timestep, thearkorbit's moss-image-edit-lora-poc's): f as AitkKv's, but their rows go after each context's text at every
    /// step, attending to all of its rows and they to them, modulated at timestep 0 (AitkNoKv) or at the step's
    /// (Appended); only their input projection is kept (stream_rows of them a context).
    struct RefLatent {
        const float* tokens;
        int64_t h, w;
    };
    void set_refs(const std::vector<RefLatent>& refs, QiRefLayout layout = QiRefLayout::MossEdit);
    int64_t ref_rows() const { return ref_rows_; }
    int64_t stream_rows() const { return stream_rows_; }
    /// The VRAM the input projection of `rows` references in the stream takes.
    uint64_t stream_bytes(int64_t rows) const { return (uint64_t) rows * C_ * 4; }
    /// The VRAM the kept keys and values of `rows` reference rows take.
    uint64_t ref_bytes(int64_t rows) const;

    /// The sampler's guidance (the reference pipeline's arguments).  cfg: with context 1, u + cfg (c - u);
    /// uncond_skip: the blocks context 1 skips; slg (with slg_scale above 0): a third pass of context 0 with those
    /// blocks skipped, slg_scale (c - k) added.  Both skips act in the steps whose share of the schedule is in
    /// [start, stop).
    struct Guide {
        float cfg = 1.f;
        std::vector<int> uncond_skip, slg;
        float slg_scale = 0.f, start = 0.f, stop = 1.f;
    };
    /// Euler steps over sigmas (n steps and the terminal 0) from x [h w][128] (device FP32, in place), each at the
    /// timestep sigma times 1000.  first, total: the schedule's steps before sigmas' first and in all (the shares the
    /// guide's window is in).  step: the progress, false to stop (QiStopped); after: at each step's next sigma (a mask's
    /// blend); peek (if set) is given each step's count from 1 and a writer of the sampler's prediction of the finished
    /// latent (x - sigma * v) into a buffer of x's size: a partial image's latent, the sampling itself not changed.
    void sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, const Guide& g, int first, int total,
                const std::function<bool(int, int)>& step = {}, const std::function<void(float*, float)>& after = {},
                const std::function<void(int, const std::function<void(float*)>&)>& peek = {},
                const GuidanceSpec* guide = nullptr, bool er = false, float shift = 5.f, uint64_t seed = 0);

    /// The velocity [n][h w][128] (device FP32) of the latent tokens x [h w][128] (device FP32) at `timestep` (sigma
    /// times 1000, as the reference pipeline gives it), for the contexts 0 .. n - 1 (n 1 or 2).
    void forward(const float* x, float timestep, int64_t h, int64_t w, int n, float* out);

    /// The color and depth control (darask0's MOSS-Image color-depth adapter, its LoRA applied as the request's): the
    /// input projection widened to the noisy latent's, the color map's and the depth map's tokens side by side
    /// ([C][3 in]: `video_embedder` of the adapter's file `path`, read once; "" none), in place of the model's own.
    /// The images' rows take its first `in` columns (the adapter trained them with the rest).
    void set_control(const std::string& path);
    const std::string& control_path() const { return ctl_path_; }
    /// The widened and image-only FP16 projections and bias, and a request's maps and FP16 input rows.
    uint64_t control_weight_bytes() const { return (uint64_t) C_ * ((uint64_t) in_ch_ * 8 + 4); }
    uint64_t control_work_bytes(int64_t rows, int maps) const {
        return (uint64_t) rows * (uint64_t) in_ch_ * (uint64_t) (maps * 4 + 6);
    }
    /// The request's control tokens [h w][in] (device FP32, the caller's, alive through sample(); null: that map off)
    /// and the share of the steps each is on for (a map past its end: zero tokens, as the adapter was trained).
    void set_control_input(const float* color, const float* depth, float color_end, float depth_end);

    int64_t hidden() const { return C_; }
    int64_t latent_channels() const { return in_ch_; }
    int64_t text_dim() const { return txt_dim_; }
    int blocks() const { return (int) blocks_.size(); }
    bool q8() const { return q8_; }
    bool w8a8() const { return w8_; }
    /// As MfDit's: the time of `rows` rows through the MLP's first product in FP16 and in W8A8 (0: not taken).
    std::pair<double, double> time_forms(int64_t rows) const;
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    void drop_blocks() { blocks_stream_.drop(); }
    /// Frees the streamed blocks' copies in RAM (LayerStream's homes).
    void drop_block_homes() { blocks_stream_.drop_homes(); }
    /// The adapters folded into the blocks' weights where they are plain LoRAs (the default), else added to each
    /// product: false keeps the weights as the file has them.
    void set_lora_bake(bool on) { loras_.bake = on; }
    uint64_t weight_bytes() const { return weight_bytes_; }   // what stays on the GPU besides the blocks and the text's
    uint64_t text_bytes() const { return text_bytes_; }       // the text embedder's weights
    bool text_loaded() const { return text_loaded_; }
    void drop_text();
    uint64_t scratch_bytes() const { return (uint64_t) scratch_ * 2; }   // the products' dequantization scratch
    /// The working buffers n contexts of hw image rows take (text rows up to max_text a context), with the MLP a band
    /// of mlp_rows rows at a time (0: all), and ref_rows reference rows before each context's in its attention.
    uint64_t activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows = 0, int64_t ref_rows = 0) const;
    void set_mlp_rows(int64_t rows) { mlp_cap_ = rows; }
    /// What the working buffers take now
    uint64_t activations_held() const {
        return cap_hw_ > 0 || cap_ref_ > 0 ? activation_bytes(cap_hw_, (int) cap_n_, cap_text_, mlp_rows_, cap_ref_)
                                           : 0;
    }
    void drop_activations();

    /// The rows of context 0 after block 0 and the text embedder's rows of context 0 (device BF16 [h w + rows][C],
    /// [rows][C]: the residual stream's form), for the probe (kept when keep_block0(true)).
    void keep_block0(bool on) { keep0_ = on; }
    const uint16_t* block0() const { return b0_; }
    const uint16_t* text_rows(int i) const { return ctx_[i].txt0; }

    /// An adapter as a file names it: a weight of the DiT, the rows of the weight a file's entry changes, and the rows
    /// of the entry they are (the text embedder's fused qkv is three weights on the GPU: q takes the entry's rows
    /// [0, C), k [C, 2C), v [2C, 3C); the text embedder's ffn_up and ffn_gate are halves of one [up | gate]).
    using LoraTarget = DitLoraTarget;
    /// The weights adapters may change: what a file's keys are matched against (the probe's map, the server's list).
    std::vector<LoraTarget> lora_targets() const;

    /// The adapters the next steps apply, each with its strength, bound to the weights' products: a key the DiT does
    /// not take (or a shape that is not its weight's, or a form the engine does not apply) is false with err, and
    /// nothing bound.  The contexts' rows were made from the other adapters' weights, so set_context again before the
    /// next forward.  The same set again (its files and strengths) changes nothing.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    /// The set bound, as its files' paths and strengths (empty: none).
    const std::string& lora_key() const { return loras_.key(); }
    /// The bound entries' A and B on the GPU.
    uint64_t lora_bytes() const;
    /// The adapter files bound (a set whose strengths are all 0 binds none).
    int lora_count() const { return loras_.count(); }
    void drop_loras();

private:
    void install_prepare();
    using W = DitW;
    using LoraSlot = DitLoraSlot;
    struct Block {
        W q, k, v, o, up_gate, down;
        uint64_t n1 = 0, n2 = 0, nq = 0, nk = 0;   // the FP32 norms' offsets
    };
    struct TextBlock {
        W q, k, v, o, up_gate, down;               // up_gate: ffn_up's rows, then ffn_gate's
        const float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr, *bug = nullptr, *bdown = nullptr;
        const float *n1 = nullptr, *n2 = nullptr, *nq = nullptr, *nk = nullptr;
    };
    struct Ctx {
        int64_t rows = 0;
        uint16_t* txt0 = nullptr;   // [rows][C]: the text embedder's output (BF16)
        std::vector<int> skip;
        // the RoPE's tables of the context's rows, [h w + rows][64] (for the size they were made for)
        float *cos = nullptr, *sin = nullptr;
        int64_t rope_h = -1, rope_w = -1, rope_rows = -1, rope_off = 0, rope_refs = 0;
    };
    W dense(const std::vector<std::string>& names, bool biased, const float** bias, std::vector<void*>& owned,
            uint64_t& bytes);
    const float* vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes);
    void load_text();
    /// The weights adapters may take, from the blocks, the text embedder and the weights outside them (open, and a
    /// later load_text).
    void build_lora_targets();
    /// A DoRA's composed row norms of the slot's rows (dit_dora_norms over the file's tensors: the text embedder's
    /// up_gate is ffn_up's rows and then ffn_gate's, the blocks' qkv one tensor or the three q, k and v).
    bool dora_norms(const LoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err);
    // the shared product (DitProducts::run)
    DitOut product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                   const float* bias, bool same_input = false, bool defer = false) {
        return prod_.run(x16, rs, w, base, y, T, bias, same_input, defer);
    }
    void ensure(int64_t hw, int n, int64_t max_text);
    void rope_tables(Ctx& c, int64_t h, int64_t w);
    void time_embedding(float timestep);
    // a block over x's rows, its segments' (a context each, its attention among its own rows) one after the other:
    // w q, k, v, o, up_gate, down; bias (null: none) theirs; mod null (the text embedder's blocks): no time
    void run_block(const W* const* w, const float* const* bias, const float* n1, const float* n2, const float* nq,
                   const float* nk, const uint8_t* base, const float* mod, uint16_t* x, const std::vector<int64_t>& seg,
                   const std::vector<const float*>& cos, const std::vector<const float*>& sin);

    std::unique_ptr<MossFile> files_;
    std::vector<std::vector<uint8_t>> host_;   // the blocks' quantized weights and FP32 vectors, when not in place
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_, text_allocs_;
    uint64_t weight_bytes_ = 0, text_bytes_ = 0;
    int64_t scratch_ = 0;
    // the products (W8A8's input, the deltas' buffers)
    DitProducts prod_{&gemm_, &allocs_, &s_, "image engine (MOSS-Image DiT)"};
    // the adapters bound (their table built at open, build_lora_targets)
    DitLoras loras_{&prod_.lora, &allocs_, &s_, "image engine (MOSS-Image DiT)"};
    bool dry_ = false, q8_ = true, w8_ = false, keep0_ = false, text_loaded_ = false;
    int64_t C_ = 3072, F_ = 8192, heads_ = 24, hd_ = 128, in_ch_ = 128, txt_dim_ = 4096, freq_ = 256;
    double rope_base_ = 10000.0;
    // the parts outside the blocks, FP16 on the GPU
    W vin_, tproj_, t1_, t2_, smod_, hmod_, fout_;
    // the color and depth control (set_control): the widened projection, its first columns alone, their bias; the
    // request's tokens and ends; the step's share of the schedule (sample's)
    W ctl_w_, ctl_w0_;
    const float* ctl_b_ = nullptr;
    std::string ctl_path_;
    std::vector<void*> ctl_allocs_;
    const float *ctl_color_ = nullptr, *ctl_depth_ = nullptr;
    float ctl_color_end_ = 1.f, ctl_depth_end_ = 1.f;
    double ctl_at_ = 0.0;
    uint16_t* ctl16_ = nullptr;
    uint64_t ctl_bytes_ = 0;
    int64_t ctl16_rows_ = 0;
    /// The images' rows through the input projection: the control's where one is set (with the maps on at this step:
    /// the widened one), else the model's own.
    void input_projection(const float* x, int64_t rows, float* y);
    const float *b_vin_ = nullptr, *b_tproj_ = nullptr, *b_t1_ = nullptr, *b_t2_ = nullptr, *b_smod_ = nullptr,
                *b_hmod_ = nullptr, *fnorm_ = nullptr;
    std::vector<TextBlock> tblocks_;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    Ctx ctx_[2];
    // activations, grown on demand
    int64_t cap_hw_ = 0, cap_n_ = 0, cap_text_ = 0, cap_ref_ = 0, mlp_cap_ = 0, mlp_rows_ = 0;
    bool cap_ref_i8_ = false;
    struct Rows {
        int64_t R, ctx, TP, J, O;   // all rows, a context's, a context's attention rows, q/k/v's and the output's
    };
    Rows rows_for(int64_t hw, int n, int64_t max_text, int64_t ref_rows, bool ref_i8) const;
    int64_t dirty_[2] = {0, 0};   // the rows of each attention slot (jq16_ ...) that may not be zero
    // x_: the residual stream, BF16 (the RMS norms read and write it: 8 bytes a value in FP32, 4
    // in BF16; the RTX 4070's step 634 -> 620 ms, 2026-11-06)
    uint16_t* x_ = nullptr;
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *gu_ = nullptr,
          *temb_ = nullptr, *mod_ = nullptr, *head_ = nullptr;
    // the FP16 inputs' row scales (rs_ the current product's, rs2_ the MLP's second's while rs_ is still read, rs_o_
    // the attention's output's), the v heads' scales (hs_, a context's heads) and their rows' maxima
    float *rs_ = nullptr, *rs2_ = nullptr, *rs_o_ = nullptr, *hs_ = nullptr, *vpart_ = nullptr;
    // the int8 attention's statistics (kernels::FlashAttnI8Stats): a context's V scales, K's mean a block and context
    // (kmean_ [blocks][2][C], a new image resets it); cur_c0_ the first context run_block runs
    float *vsv_ = nullptr;
    I8Means kmean_;
    int cur_c0_ = 0;
    uint16_t *h16_ = nullptr, *jq16_ = nullptr, *jk16_ = nullptr, *jv16_ = nullptr, *jo16_ = nullptr,
             *o16_ = nullptr, *t16_ = nullptr;
    int64_t jrows_ = 0;   // the rows of jq16_ .. jv16_ a context (qi_attn_rows of its rows at most)
    // the other of the W8A8 input's two arrays of scales, made the current one (what product's same_input reads)
    float* i8_slot() { return prod_.in.slot(); }
    // the last block's MLP output not yet added to x: the next block's first kernel adds it where it runs over the
    // same rows (rows_i8's RmsI8), else flush_pending; its weights' scales copied to pend_sw_
    struct Pending {
        DitOut add;
        const float* gate = nullptr;
        uint16_t* x = nullptr;
        int64_t R = 0;
        AltRows rr;
    } pend_;
    float* pend_sw_ = nullptr;
    void flush_pending();
    uint16_t* b0_ = nullptr;
    // an edit's references (set_refs): every block's keys and values of their rows and its v heads' scales; capture_
    // while set_refs runs the blocks over them; cur_block_ the block run_block runs
    uint16_t *ref_k16_ = nullptr, *ref_v16_ = nullptr;
    float* ref_hs_ = nullptr;
    int64_t ref_rows_ = 0;
    // or, where the attention takes the int8 kernel (refs_in_i8), each block's keys and values made into its prefix
    // once (qi_attention_i8_prefix: ref_i8_stride_ bytes a block), and the rows' own q, k and v without the
    // references' rows before them
    uint8_t* ref_i8_ = nullptr;
    uint64_t ref_i8_stride_ = 0;
    bool ref_i8_on_ = false;
    bool refs_in_i8(int64_t rows) const;
    // the references in the stream (AitkNoKv, Appended): their rows through the input projection [R][C], and the
    // modulation's six rows at timestep 0 after mod_'s (AitkNoKv's)
    uint16_t* ref_x_ = nullptr;
    int64_t stream_rows_ = 0;
    std::vector<std::pair<int64_t, int64_t>> ref_shapes_;   // the references' latent rows and columns
    void ref_coord(int64_t r, float* p) const;
    bool stream_t0_ = false;
    bool ref_text_start_ = true, capture_ = false;
    int cur_block_ = 0;
};

}  // namespace strata::image
