// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_dit.hpp - Qwen-Image-2.1's DiT on the GPU: the velocity of a latent at a time, and the
// flow sampler over Turbo's schedule.
//
// The sequence is the prefix (the text encoder's states, projected once by set_context, and in an edit its reference
// images' latent tokens in place of their image pads) and the image's H x W tokens (the latent's 64 channels each).
// A text token sees the prefix up to itself, a reference image's token the prefix up to the image's end, and their
// time is 0, so their hidden states, and their K and V in every block, are the same at every step: the first step
// keeps the blocks' prefix K and V (the prefix cache), and the later ones compute the image's rows alone.  The cache
// of the first blocks is in VRAM, as many as the budget gives it room for (set_prefix_vram_blocks); the rest is in
// pinned RAM and copied in at each step (an edit with many or large reference images).
#pragma once

#include "strata/image/image_lora.hpp"
#include "strata/image/i8_means.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::image {

class QiControl;

/// Philox 4x32-10 normal values as PyTorch's CUDA randn gives them for `seed` (the generator stable-diffusion.cpp
/// uses): value i from counter (offset, 0, i, 0) and key (seed low, seed high), Box-Muller's sine half.
std::vector<float> philox_randn(uint64_t seed, uint64_t n, uint32_t offset = 0);

/// Qwen-Image-2.1-Turbo's own 8-step schedule (its model_index.json's sample_sigmas) and the terminal 0.
const std::vector<float>& turbo_sigmas();

class QiDit {
public:
    QiDit();
    ~QiDit();
    QiDit(const QiDit&) = delete;
    QiDit& operator=(const QiDit&) = delete;

    /// Reads the DiT's GGUF (mapped) and puts its weights on the current GPU.  False with `err` on failure.
    /// `dry`: the file read for its sizes only (block_bytes, weight_bytes, activation_bytes...: what a budget is
    /// planned with), nothing put on the GPU; such a part cannot run.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// An edit's reference image: its `slots` image pads from context row `slot` are its latent's h x w tokens instead
    /// (latent: device FP32 [64][h][w], the VAE encoder's; 4 tokens a pad), at positions [p, row, column] (the grid's
    /// centre 0, as the target's), p moving on by the image's longer side after it.
    struct Ref {
        int64_t slot = 0, slots = 0, h = 0, w = 0;
        const float* latent = nullptr;
    };

    /// The text encoder's states [n_ctx][4096] (host FP32, after the system turn): projected into the sequence's
    /// prefix, with `refs` (in order) in place of their pads; the prefix cache starts again.  `second`: the second
    /// context (classifier-free guidance's negative prompt), which `sample` runs beside the first.
    void set_context(const float* ctx, int64_t n_ctx, const std::vector<Ref>& refs = {}, bool second = false);
    /// Frees the second context (a request without guidance).
    void drop_second_context();

    /// out = the velocity [64][H][W] (device FP32) of the latent x (device FP32, the same shape) at time t (sigma x
    /// 1000).  `prefix`: the text rows' K and V from the cache (a step after the first of one context).
    void forward(const float* x, float t, int64_t H, int64_t W, bool prefix, float* out);

    /// The flow sampler: x (device FP32 [64][H][W], the starting noise) through `sigmas` by Euler's steps.
    /// `step` (if set) is called after each step with (steps done, steps); it returns false to stop (the sampler then
    /// throws QiStopped).  `after` (if set) is given x and the step's new sigma first, on the sampler's stream (a mask:
    /// img_start.hpp's blend).  `cfg` above 1 (with a second context set): classifier-free guidance, the
    /// velocity neg + cfg * (cond - neg) of the two contexts' (diffusers' true_cfg_scale).
    /// `peek` (if set) is called after each step with the steps done and a function that writes the step's x0, the
    /// sampler's prediction of the finished latent (x - sigma * v), into a buffer of x's size: a partial image's
    /// latent, on the sampler's stream; the sampling itself is not changed.  `guide` (if set and on): the guidance of
    /// guidance.hpp, the steps then going by its denoised values (Euler's step from them).
    void sample(float* x, int64_t H, int64_t W, const std::vector<float>& sigmas,
                const std::function<bool(int, int)>& step = {},
                const std::function<void(float*, float)>& after = {}, float cfg = 1.0f,
                const std::function<void(int, const std::function<void(float*)>&)>& peek = {},
                const GuidanceSpec* guide = nullptr, bool er = false, float shift = 1.0f, uint64_t seed = 0);

    int64_t channels() const { return in_ch_; }
    int64_t hidden() const { return C_; }
    /// The bytes the weights outside the blocks take on the GPU (always there), and the blocks' (in the file).
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    /// The two slots a streamed block walk takes at least.
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    /// The VRAM the blocks may take: those that fit stay, the rest are streamed (LayerStream).
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    /// What the blocks and the activations hold on the GPU now.
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    uint64_t activations_held() const { return cap_T_ > 0 ? activation_bytes(cap_T_, mlp_rows_) : 0; }
    /// Frees the blocks' VRAM (they come back from the file on use).
    void drop_blocks() { blocks_stream_.drop(); }
    /// Frees the streamed blocks' copies in RAM (LayerStream's homes).
    void drop_block_homes() { blocks_stream_.drop_homes(); }
    /// The adapters folded into the blocks' weights where they are plain LoRAs (the default), else added to each
    /// product: false keeps the weights as the file has them.
    void set_lora_bake(bool on) { lora_bake_ = on; }
    /// Frees the activations (grown again on use).
    void drop_activations();
    /// The GPU memory the activations of a T-token sequence take, the MLP computed `mlp_rows` rows at a time (0: all).
    uint64_t activation_bytes(int64_t T, int64_t mlp_rows = 0) const;
    /// What a prefix of n rows holds while it is the context: its rows (always in VRAM), and one block's K and V
    /// (the prefix cache, in VRAM or pinned RAM).
    uint64_t prefix_row_bytes(int64_t n) const { return (uint64_t) n * (uint64_t) C_ * 4; }
    uint64_t prefix_kv_bytes(int64_t n) const { return (uint64_t) n * (uint64_t) C_ * 4; }
    int blocks() const { return (int) blocks_.size(); }
    /// The blocks whose prefix cache the next set_context puts in VRAM (the first n); the rest's goes to pinned RAM.
    void set_prefix_vram_blocks(int n) { kv_vram_ = n; }
    /// Frees the prefix cache kept in pinned RAM (the sampling is done with it: the RAM goes back at once).
    void drop_host_prefix();
    /// The rows the MLP takes at a time (0: all of them): its 2 x 12288-wide product is the largest activation.
    void set_mlp_rows(int64_t rows);

    /// An adapter as a file names it: a weight of the DiT, and the rows of the weight a file's entry of that name
    /// changes (a fused weight's halves by their own names: img_mlp.gate_layer is gate_up's rows [0, inter)).
    struct LoraTarget {
        std::string name;
        int64_t row0 = 0, rows = 0, cols = 0;
    };
    /// The weights adapters may change: what a file's keys are matched against (the probe's map, the server's list).
    std::vector<LoraTarget> lora_targets() const;

    /// The adapters the next steps apply, each with its strength, bound to the weights' products: a key the DiT does
    /// not take (or a shape that is not its weight's, or a form the engine does not apply) is false with err, and
    /// nothing bound.  The context starts again (set_context before the next forward): the prefix's K and V were the
    /// other adapters'.  The same set again (its files and strengths) changes nothing.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    /// The set bound, as its files' paths and strengths (empty: none).
    const std::string& lora_key() const { return lora_key_; }
    /// The bound entries' A and B on the GPU.
    uint64_t lora_bytes() const;
    /// The adapter files bound (a set whose strengths are all 0 binds none).
    int lora_count() const { return (int) lora_files_.size(); }
    void drop_loras();

    /// The ControlNet branch the sampling runs beside the blocks (qi_control.hpp; null: none): in a step it is on for,
    /// every row is computed (the prefix cache neither read nor written) and the branch's skips are added.
    void set_control(QiControl* control) { control_ = control; }

private:
    friend class QiControl;
    QiControl* control_ = nullptr;
    bool control_on(float t) const;   // the branch on for the step at time t (sigma x 1000)
    bool lora_bake_ = true, baked_any_ = false;
    float* bake_tmp_ = nullptr;      // a weight's delta [n][k], while adapters are baked
    uint16_t* bake_at16_ = nullptr;  // an adapter's A^T
    void install_prepare();
    struct W {
        int type = 0;
        int64_t n = 0, k = 0;   // n rows of k
        const void* d = nullptr;
        uint64_t row_bytes = 0;
        bool w8 = false;        // rewritten as int8 rows and a scale a row (qi_rows_i8) for prefill::Gemm::i8
        bool w8t = false;       // w8 through w8t_: a block format of fewer bits than int8's rows, kept as it is and
                                // written as int8 rows into the scratch at each product
        LoraLayer* lora = nullptr;   // the adapters' deltas for its products (null: none: the product is unchanged)
        bool baked = false;          // its adapters folded into the block's copy of it (set_loras), not added to products
    };
    struct Block {
        W q, k, v, o, gu, down;   // d: the offset within the block's region (LayerStream)
        uint64_t nq = 0, nk = 0;  // the FP32 norms' offsets
        uint16_t *pk = nullptr, *pv = nullptr;   // the text rows' K and V (the prefix cache), device or pinned host
    };
    W upload(const std::string& name);
    const float* upload_f32(const std::string& name, float add = 0.0f);
    // y [T][rows] (row stride ldy) = x16 . w's rows row0 .. row0 + rows - 1.  A W8A8 one (W::w8): same_input takes
    // the previous one's int8 input again (x16 unread), and defer leaves y its int32 sums, their scales in i8_; true
    // when it did.
    bool product(const uint16_t* x16, const W& w, float* y, int64_t T, int64_t ldy = 0, int64_t row0 = 0,
                 int64_t rows = -1, bool same_input = false, bool defer = false);
    // the int8 input of a W8A8 product of T rows of K, and the next of the two scale arrays (*sx)
    int8_t* xq8(int64_t T, int64_t K, float** sx);
    // the MLP's rows from m0: gate_up, SwiGLU, down, gated into x (n_text of them text rows)
    void mlp(const W& wgu, const W& wdown, float* x, const float* gate, int64_t n_text, int64_t mr);
    void step_modulation(float t);
    /// The rotary tables [n_text + H W][64] of a context's positions into cos and sin (device).
    void rope_tables(const std::vector<float>& pre_pos, int64_t n_text, int64_t target_pos, int64_t H, int64_t W,
                     float* cos, float* sin);
    /// forward for both contexts at once, their prefixes cached (guidance after its first step): the image's rows of
    /// the two side by side through each product, the attention one context at a time; out the first's velocity,
    /// out2 the other's.
    void forward_pair(const float* x, float t, int64_t H, int64_t W, float* out, float* out2);

    void ensure(int64_t T, int64_t HW);
    void swap_context();   // the first context's state <-> other_'s
    /// The weights adapters may take, from the blocks and the weights outside them (open).
    void build_lora_targets();
    /// The products' scratch: T rows of the largest bound rank, FP32 and FP16 (grown on use).
    void lora_scratch(int64_t T);
    /// The context's prefix let go: its text rows, and both contexts' K and V caches (an adapter change).
    void drop_prefix();

    std::unique_ptr<GgufFile> g_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false;
    int64_t C_ = 4096, in_ch_ = 64, heads_ = 32, inter_ = 12288, n_text_ = 0;
    int kv_vram_ = 1 << 30;   // the blocks whose prefix cache is in VRAM (set_prefix_vram_blocks)
    W t1_, t2_, mod_, nout_, pout_, tin_, tout_, iin_;
    const float* tnorm_ = nullptr;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    // the adapters: the weights a file's keys land on, the layers the weights hold, and the products' scratch
    struct LoraSlot {
        std::string name;
        W* w = nullptr;
        int64_t row0 = 0, rows = 0;
    };
    /// A DoRA's row norms: the norms of a slot's weight, one a row of it, from the file's tensor (a fused weight's
    /// gate_layer and proj are its gate_up's rows), cached a tensor (open drops the cache).
    bool dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err);
    /// One of the file's tensors' row norms, FP32, cached (`key`: the GGUF's own name).
    bool tensor_norms(const std::string& key, const std::vector<float>*& out, std::string& err);
    std::vector<LoraSlot> lora_slots_;
    std::vector<std::unique_ptr<LoraLayer>> lora_layers_;
    std::vector<std::shared_ptr<const LoraFile>> lora_files_;
    std::string lora_key_;
    int64_t lora_cols_ = 0;   // the deltas' scratch's columns (the largest lora_scratch_cols of the bound entries)
    float* lora_f32_ = nullptr;
    uint16_t* lora16_ = nullptr;
    int64_t lora_tmp_t_ = 0;   // the scratch's rows
    // a GLoRA's own path: its input's columns, the weight's rows it is added over, and the buffers a band of its
    // products takes (the rows' table, a W8A8 weight's int8 input and its scales)
    int64_t lora_glora_cols_ = 0, lora_glora_rows_ = 0;
    float* lora_glora_y_ = nullptr;
    int8_t* lora_glora_x8_ = nullptr;
    float* lora_glora_sx_ = nullptr;
    void lora_glora_scratch();   // allocates them, once a set with a GLoRA is bound
    void free_glora_scratch();
    /// A GLoRA's own path: the weight's rows [r0, r0 + n) product of v (nt tokens, the first of them t0), added to
    /// the product's output y at the rows' place in it (the weight's own path, as product takes it).
    void glora_weight(const uint16_t* v, const W& w, int64_t r0, int64_t n, int64_t nt, int64_t t0, float* y,
                      int64_t ldy, int64_t row0);
    std::map<std::string, std::vector<float>> norm_cache_;   // DoRA's row norms, a file's tensor (open clears them)
    std::vector<std::vector<float>> norms32_;   // an FP16 GGUF's q/k norms widened to FP32, streamed in their place
    bool prefix_cache_ = false;   // the blocks' pk and pv hold the context's prefix (set_context; set_loras drops it)
    // activations, grown on demand
    int64_t cap_T_ = 0, mlp_cap_ = 0, mlp_rows_ = 0;
    const std::vector<int>* skip_ = nullptr;   // the blocks a forward with the prefix cached skips (skip layer guidance)
    int64_t dirty_T_ = 0;   // the most rows q16_, k16_ and v16_ were written to (the padding past a shorter T is zeroed)
    // the int8 attention's K means kept from the last step (a block and context), reset on a new image (t above the
    // last); ctx_i_ the context forward runs (swap_context turns it, forward_pair uses both slots)
    I8Means kmean_;
    int ctx_i_ = 0;
    float last_t_ = -1.f;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *h_ = nullptr, *gu_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr, *o16_ = nullptr;
    float *cos_ = nullptr, *sin_ = nullptr;
    bool w8_ = false;       // some block's products are W8A8 ones (W::w8): xq_ holds their activations
    void* xq_ = nullptr;
    size_t xq_bytes_ = 0;
    float* xs_ = nullptr;    // W8A8: two arrays of xs_rows_ scales, taken in turn (xs_next_), so that a product's
    int64_t xs_rows_ = 0;    // deferred sums keep theirs while the next input is quantized
    int xs_next_ = 0;
    QiI8 i8_;                // the last deferred product's scales
    // W::w8t: a weight's int8 rows (the largest weight's), and kW8tScales arrays of its row scales taken in turn (a
    // deferred product's scales are read after up to two more products: q, k and v)
    static constexpr int kW8tScales = 4;
    int8_t* w8t_ = nullptr;
    float* w8t_s_ = nullptr;
    int64_t w8t_rows_ = 0;
    int w8t_next_ = 0;
    int64_t rope_H_ = -1, rope_W_ = -1, rope_text_ = -1;
    float *text_ = nullptr;           // the prefix's rows [n_text][C] (the projected text, the references' tokens)
    std::vector<float> pre_pos_;      // the prefix rows' positions [n_text][3]
    int64_t target_pos_ = 0;          // the image's first position
    int32_t* key_lim_ = nullptr;      // the prefix rows' key limits, with references (null: each its own prefix)
    std::vector<std::pair<int64_t, int64_t>> ref_spans_;   // the references' rows [a, b): they see keys 0 .. b - 1
    // the other context (guidance's negative prompt): the same state, swapped in while it runs (swap_context)
    struct Context {
        float* text = nullptr;
        int64_t n_text = 0, target_pos = 0;
        std::vector<float> pre_pos;
        int32_t* key_lim = nullptr;
        std::vector<std::pair<int64_t, int64_t>> ref_spans;
        std::vector<uint16_t*> pk, pv;
        float *cos = nullptr, *sin = nullptr;   // its rotary tables (forward_pair), for this size and prefix
        int64_t rope_H = -1, rope_W = -1, rope_text = -1, rope_cap = 0;
    } other_;
    float *mods_ = nullptr;           // [4][2][C]: the step's modulation, row 0 the image's, row 1 the text's
    float *scale_ = nullptr;          // [2][C]: the final norm's scale (row 0)
    float *temb_ = nullptr;           // [2][C] the time embeddings (t, 0), after their SiLU
    uint16_t *small16_ = nullptr;     // [2][C] FP16 operands of the time path
};

}  // namespace strata::image
