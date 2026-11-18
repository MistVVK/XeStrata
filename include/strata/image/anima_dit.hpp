// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/anima_dit.hpp - Anima's DiT on the GPU (Cosmos-Predict2-2B's MiniTrainDIT): its LLM adapter,
// which turns the text encoder's states into the cross-attention's context once a prompt, and its blocks, the velocity
// of a flow-matching step.
//
// The latent is Wan 2.1's (16 channels at an eighth of the image's sides), in 2 x 2 patches: a token's 64 values are
// channel c of the patch's pixel (m, n) at c * 4 + m * 2 + n (the patch embedder's order), the sampler's tokens
// [h w][64] (h, w the patches' rows and columns).  The patch embedder's 17th channel, the padding mask, is all zeros for
// an image, so its columns are left out.
//
// The time: the sinusoid [cos, sin] of sigma (not sigma times 1000) by 1024 frequencies 10000^(-i / 1024), rounded to
// bf16; its RMS norm (t_embedding_norm) is the blocks' embedding, and the timestep embedder's two layers of the
// sinusoid give the AdaLN-LoRA term each modulation adds.  Each block: self-attention among a context's image tokens
// (q and k RMS-normed a head and rotated by the 3-D RoPE: 22 pairs by time, 21 by row, 21 by column, pairs (j, j + 64)),
// cross-attention to the context's rows (q and k RMS-normed, not rotated), and a GELU MLP, each on a LayerNorm (no
// affine, eps 1e-6) scaled by 1 + scale and shifted, added back through its gate; the three modulations are each
// block's own (SiLU of the embedding, 2048 -> 256 -> 6144, plus the AdaLN-LoRA term).  The final layer is the same
// LayerNorm and modulation (two thirds of it) and a projection to the patch's 64 values, in the unpatchify's order
// (pixel (p1, p2) of channel c at (p1 * 2 + p2) * 16 + c), which forward writes back in the tokens' order.
//
// The LLM adapter (6 blocks of 1024, 16 heads of 64): the T5 tokenizer's ids embedded, each block self-attention (RoPE
// over the ids' positions, pairs (j, j + 32)), cross-attention to the text encoder's states (the queries rotated by
// their positions, the keys by theirs) and an MLP with biases, on RMS norms with weights; then out_proj and an RMS norm.
// Its rows are multiplied by the T5 ids' weights (the prompt's emphasis) and padded with zero rows to 512: the zero
// rows take part in the blocks' cross-attention, unmasked.  The cross-attention's keys and values of a context's
// rows are the same at every step: each block's are made at its first step and kept.
//
// The weights: one safetensors file (BF16 as distributed; F16, F32 and F8 E4M3 with a scale_weight a matrix converted
// as read), its names with or without the prefix `net.` (the release) or `model.diffusion_model.` (single-file saves of
// derivatives).
// The blocks as MossDit's: 8 bits (W8A8 where the GPU's library takes int8 products, else Q8_0) quantized on the host,
// or FP16 from bf16, through a LayerStream; the number of blocks from the file (28; the 2.9B expansion's 40).  The
// parts outside the blocks and the adapter are FP16 on the GPU; the adapter's (0.2 GiB) are needed only by
// set_context, and drop_text gives their VRAM back.
#pragma once

#include "strata/image/dit_loras.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

class AnimaFile;

namespace anima_dit_detail {
/// An F8 E4M3 byte's value (a scaled fp8 save's weight; no inf, 0x7f / 0xff NaN)
float e4m3(uint8_t b);
}  // namespace anima_dit_detail

/// The path is a safetensors file of Anima's DiT (its LLM adapter beside Cosmos-Predict2's blocks), with or without
/// a prefix.
bool anima_is_dit(const std::string& path);

/// The self-attention's 3-D RoPE tables [h w][hd / 2] (cos, sin) of one image (frame 0): the time's pairs, then the
/// rows', then the columns', as the DiT reads them; the trainer's as the inference's.
void anima_rope_tables(int64_t h, int64_t w, int64_t hd, std::vector<float>& c, std::vector<float>& s);

/// The time's two FP16 rows [2][C]: the sinusoid of sigma (its C / 2 frequencies 10000^(-i / (C / 2)), rounded to
/// bf16), then the SiLU of its RMS norm (eps 1e-6, weight `tnorm` [C], rounded to bf16) that every modulation reads:
/// the DiT's and the trainer's.
void anima_time_rows(float sigma, const std::vector<float>& tnorm, std::vector<uint16_t>& h);

class AnimaDit {
public:
    AnimaDit();
    ~AnimaDit();
    AnimaDit(const AnimaDit&) = delete;
    AnimaDit& operator=(const AnimaDit&) = delete;

    /// `q8`: the blocks in 8 bits, else FP16 from the file's bf16 (a file of another type always takes 8 bits).
    /// `dry`: read for its sizes only.  `reuse`: the blocks' host bytes an earlier open left (take_host).
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool q8 = true,
              std::vector<std::vector<uint8_t>>* reuse = nullptr);
    std::vector<std::vector<uint8_t>> take_host() { return std::move(host_); }

    /// Context i (0: the prompt, 1: the negative prompt): the text encoder's states [rows][1024] (host FP32) through
    /// the LLM adapter with the T5 ids and their weights (one each), kept on the GPU.
    void set_context(int i, const float* states, int64_t rows, const std::vector<int32_t>& t5,
                     const std::vector<float>& t5_weights);
    /// Several contexts at once (a guided request's two), through the adapter as one run of rows.
    struct ContextIn {
        int i;
        const float* states;
        int64_t rows;
        const std::vector<int32_t>* t5;
        const std::vector<float>* t5w;
    };
    void set_contexts(const std::vector<ContextIn>& in);
    int64_t context_rows(int i) const { return ctx_[i].rows; }
    /// Context i's rows as the blocks' cross-attention reads them: [context_rows(i)][context_cols()] FP16, each row
    /// times its scale (context_scales), the zero rows after the prompt's included (a trainer's cache of them).
    const uint16_t* context16(int i) const { return ctx_[i].c16; }
    const float* context_scales(int i) const { return ctx_[i].rs; }
    int64_t context_cols() const { return ad_C_; }

    /// The velocity [n][h w][64] (device FP32, the tokens' order) of the tokens x [h w][64] at sigma, for the contexts
    /// 0 .. n - 1 (n 1 or 2).
    void forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out,
                 const std::vector<int>* skip = nullptr);

    int64_t hidden() const { return C_; }
    int64_t token_values() const { return 4 * lat_; }   // a token's values: 16 channels x 2 x 2
    int64_t latent_channels() const { return lat_; }
    int64_t text_dim() const { return src_dim_; }
    int blocks() const { return (int) blocks_.size(); }
    bool q8() const { return q8_; }
    bool w8a8() const { return w8_; }
    std::pair<double, double> time_forms(int64_t rows) const;
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    void drop_blocks() { blocks_stream_.drop(); }
    void drop_block_homes() { blocks_stream_.drop_homes(); }
    uint64_t weight_bytes() const { return weight_bytes_; }   // what stays besides the blocks and the adapter's
    uint64_t text_bytes() const { return text_bytes_; }       // the LLM adapter's weights
    bool text_loaded() const { return text_loaded_; }
    void drop_text();
    uint64_t scratch_bytes() const { return (uint64_t) scratch_ * 2; }
    /// The working buffers n contexts of hw image tokens take, the MLP a band of mlp_rows rows at a time (0: all),
    /// and the contexts' kept keys and values.
    uint64_t activation_bytes(int64_t hw, int n, int64_t mlp_rows = 0) const;
    void set_mlp_rows(int64_t rows) { mlp_cap_ = rows; }
    uint64_t activations_held() const { return cap_hw_ > 0 ? activation_bytes(cap_hw_, (int) cap_n_, mlp_rows_) : 0; }
    void drop_activations();

    /// As MossDit's: the weights adapters may change, the set bound (set_loras), its key, bytes and files.  A set that
    /// changes drops the contexts' kept keys and values.
    using LoraTarget = DitLoraTarget;
    std::vector<LoraTarget> lora_targets() const;
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    /// An expanded DiT's blocks that the release's are (the 2.9B's: its 40 blocks, 12 of them inserted between the
    /// release's 28): an adapter whose blocks are all below the map's size is the release's, its block k bound to
    /// block map[k].  Empty: the blocks as the file names them.
    void set_lora_block_map(std::vector<int> map) { block_map_ = std::move(map); }
    /// The adapters folded into the blocks' weights where they are plain LoRAs (the default), else added to each
    /// product: false keeps the weights as the file has them.
    void set_lora_bake(bool on) { loras_.bake = on; }
    const std::string& lora_key() const { return loras_.key(); }
    uint64_t lora_bytes() const;
    int lora_count() const { return loras_.count(); }
    void drop_loras();

private:
    using W = DitW;
    using LoraSlot = DitLoraSlot;
    struct Block {
        W sq, sk, sv, so, cq, ck, cv, co, l1, l2, msa1, msa2, mca1, mca2, mml1, mml2;
        uint64_t sqn = 0, skn = 0, cqn = 0, ckn = 0;   // the FP32 norms' offsets
    };
    struct AdapterBlock {
        W sq, sk, sv, so, cq, ck, cv, co, m0, m2;
        const float *bm0 = nullptr, *bm2 = nullptr;
        const float *n_sa = nullptr, *n_ca = nullptr, *n_ml = nullptr, *sqn = nullptr, *skn = nullptr,
                    *cqn = nullptr, *ckn = nullptr;
    };
    struct Ctx {
        int64_t rows = 0;            // the context's rows (512, or the adapter's when more)
        uint16_t* c16 = nullptr;     // [rows][1024] FP16: the adapter's output, the blocks' cross-attention's source
        float* rs = nullptr;         // its rows' scales
        std::vector<uint16_t*> kv;   // each block's keys [heads][rows][hd] and values [heads][hd][rows] (null: not made)
        std::vector<int64_t> kv_lp;  // the padded rows each block's was allocated for
        std::vector<char> kv_ok;     // each block's made from these rows (a new prompt keeps the buffers, not this)
    };
    W dense(const std::string& name, bool biased, const float** bias, std::vector<void*>& owned, uint64_t& bytes,
            int64_t cols = 0);
    const float* vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes);
    void load_text();
    void build_lora_targets();
    bool dora_norms(const LoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err);
    // the shared product (DitProducts::run); defer: a W8A8 product's int32 sums left for its reader to scale
    // (DitOut::at), no pass of its own (a 16-bit weight's sums are scaled here: Anima's readers take them as FP32)
    DitOut product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                   const float* bias, bool same_input = false, bool defer = false) {
        return prod_.run(x16, rs, w, base, y, T, bias, same_input, defer && w.type == kDitW8);
    }
    void ensure(int64_t hw, int n);
    // the cross-attention's products form (a GPU without FP16 matrix engines): the keys and values head-major, then a
    // head at a time the scores, their softmax and the weighted values through the shared products, out [Tq][C] FP32
    void attn_kv(const DitOut& k, const DitOut& v, int64_t Tk, int64_t heads, int64_t hd, const float* wk, uint16_t* kv);
    void attn_q(const DitOut& q, int64_t Tq, int64_t heads, int64_t hd, const float* wq, const uint16_t* kv, int64_t Tk,
                float* out);
    void rope_tables(int64_t h, int64_t w);
    void time_embedding(float sigma);
    // a modulation's 6144 values (or the final layer's 4096): SiLU(emb) through w1 and w2, plus the AdaLN-LoRA term
    void modulation(const W& w1, const W& w2, const uint8_t* base, float* out);
    struct AdapterIn {
        Ctx* c;
        const float* states;
        int64_t rows;
        const std::vector<int32_t>* t5;
        const std::vector<float>* t5w;
    };
    void adapter(const std::vector<AdapterIn>& in);
    void drop_kv();

    std::unique_ptr<AnimaFile> file_;
    std::vector<std::vector<uint8_t>> host_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_, text_allocs_;
    uint64_t weight_bytes_ = 0, text_bytes_ = 0;
    int64_t scratch_ = 0;
    std::vector<int> block_map_;
    bool lora_remap_ = false;   // the file being bound names its blocks through block_map_
    // the products (W8A8's input, the deltas' buffers)
    DitProducts prod_{&gemm_, &allocs_, &s_, "image engine (Anima DiT)"};
    // the adapters bound (their table built at open, build_lora_targets)
    DitLoras loras_{&prod_.lora, &allocs_, &s_, "image engine (Anima DiT)"};
    bool dry_ = false, q8_ = true, w8_ = false, text_loaded_ = false;
    int64_t C_ = 2048, F_ = 8192, heads_ = 16, hd_ = 128, lat_ = 16, src_dim_ = 1024, mod_dim_ = 256;
    int64_t ad_C_ = 1024, ad_heads_ = 16, ad_hd_ = 64, ad_F_ = 4096, vocab_ = 32128;
    // the parts outside the blocks, FP16 on the GPU
    W xemb_, t1_, t2_, flin_, fm1_, fm2_;
    std::vector<float> tnorm_host_;   // t_embedding_norm's weight (host)
    // the adapter
    std::vector<AdapterBlock> ablocks_;
    W aout_;
    const float *b_aout_ = nullptr, *anorm_ = nullptr;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    Ctx ctx_[2];
    // activations, grown on demand
    int64_t cap_hw_ = 0, cap_n_ = 0, mlp_cap_ = 0, mlp_rows_ = 0, cap_L_ = 0;
    int64_t rope_h_ = -1, rope_w_ = -1;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *g_ = nullptr, *o32_ = nullptr;
    float *rs_ = nullptr, *rs2_ = nullptr, *rs_o_ = nullptr, *hs_ = nullptr, *vpart_ = nullptr;
    // the int8 attention's statistics (kernels::FlashAttnI8Stats): a context's V scales, K's mean a block and context
    // (kmean_ [blocks][2][C], a sigma above the last starting one resets it)
    float *vsv_ = nullptr;
    I8Means kmean_;
    float last_sigma_ = 0.f;
    float *cos_ = nullptr, *sin_ = nullptr;   // the image tokens' RoPE [h w][64]
    float *sc_ = nullptr;                     // the products form's scores [Tq][Tk]
    uint16_t *h16_ = nullptr, *jq16_ = nullptr, *jk16_ = nullptr, *jv16_ = nullptr, *jo16_ = nullptr,
             *o16_ = nullptr, *qh16_ = nullptr, *p16_ = nullptr;
    int64_t jrows_ = 0;
    int64_t dirty_[2] = {0, 0};
    void install_prepare();
    int cross_form_ = -1;   // the cross-attention: unknown, the shared entry, the products (no matrix engines)
    // the time's parts: the embedding's SiLU (FP16), the AdaLN-LoRA term, a modulation's hidden values and its output
    uint16_t *e16_ = nullptr, *m16_ = nullptr;
    // the adapter's working buffers, one region kept between prompts (grown on use, freed with the adapter's weights)
    uint8_t* ad_scratch_ = nullptr;
    uint64_t ad_bytes_ = 0;
    float *alora_ = nullptr, *mh_ = nullptr, *mod_ = nullptr, *fmod_ = nullptr;
};

}  // namespace strata::image
