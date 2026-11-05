// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_dit.hpp - HunyuanImage 3.0's decoder on the GPU: the dense weights resident as
// FP16, the routed experts' banks streamed as the file's quantized bytes and dequantized on the device.
//
// One layer's arithmetic is hunyuanimage3_layer_run's, and the direct path (hunyuanimage3_gpu's hunyuanimage3_layer_gpu, which
// dequantizes on the host and uploads what it gets) runs the same: the layer-0 numbers held against the host FP32
// are this file's numbers too.  What is added here is where the weights come from, and how a model whose expert
// banks are 43 GB of quantized bytes walks on a card that holds none of them at once - the dense products stay
// resident, the banks come through a LayerStream (a resident prefix, the rest in two slots) and are turned into
// FP16 one whole expert at a time, on the device, as the router sends tokens to it.
//
// The shapes are the model's (its config.json), not the hardware's; the sizes a budget is planned with come from
// the file.  Nothing here is tuned: the tiles and the local-memory forms come once the values are held to the
// golden.
#pragma once

#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// The model's own shapes and scalars: hidden 4096, 32 query heads over 8 key-value heads of head_dim 128, an
/// intermediate of 3072 (6144 fused), 64 routed experts with the top 8 of them, the rms-norm eps (its config's
/// `rms_norm_eps`) and the rope theta.  A file's matrix shapes say the same of hidden, the intermediate and the
/// qkv rows; the head counts are the config's, which a quantized file does not carry.
struct HunyuanImage3Dims {
    int64_t hidden = 4096, heads = 32, kv_heads = 8, head_dim = 128, inter = 3072;
    int64_t experts = 64, topk = 8, rotate = 256;
    float eps = 1e-5f, theta = 10000.0f;
};

/// One layer's weights on the device, however they got there: the FP16 rows of the four dense products - row-major
/// [n][k], in the save's rotated basis - and the FP32 norms and router.  `egu_*` and `edn_*` are the routed
/// experts' shapes: the two matrices whose FP16 image `HunyuanImage3Experts::fill` writes.
struct HunyuanImage3LayerWeights {
    const uint16_t *qkv = nullptr, *o = nullptr, *sgu = nullptr, *sdn = nullptr;
    const float *n_in = nullptr, *n_post = nullptr, *n_q = nullptr, *n_k = nullptr, *wg = nullptr;
    int64_t qkv_n = 0, qkv_k = 0, o_n = 0, o_k = 0, sgu_n = 0, sgu_k = 0, sdn_n = 0, sdn_k = 0;
    int64_t egu_n = 0, egu_k = 0, edn_n = 0, edn_k = 0;
};

/// The device buffers one layer's walk takes, for `T` tokens of the model `d`.  The attention's q, k and v hold
/// `rows` = qi_attn_rows(T) rows, the rows past T zero.  `egu16` and `edn16` are the FP16 image of ONE routed
/// expert, which `HunyuanImage3Experts::fill` writes as the walk reaches it.
struct HunyuanImage3Scratch {
    int64_t T = 0, rows = 0;
    uint16_t *xr16 = nullptr, *q16 = nullptr, *k16 = nullptr, *v16 = nullptr, *a16 = nullptr, *ar16 = nullptr,
             *xr216 = nullptr, *act16 = nullptr, *gact16 = nullptr, *gx16 = nullptr, *egu16 = nullptr,
             *edn16 = nullptr;
    /// The two norms stay FP32 up to the rotation: the row's few enormous channels would carry their FP16 rounding
    /// through the rotation into every value of the group (see hunyuanimage3_rotate16_from32).
    float *xn32 = nullptr, *xn232 = nullptr;
    float *qkv32 = nullptr, *o32 = nullptr, *gu32 = nullptr, *eg32 = nullptr, *ed32 = nullptr,
          *logits = nullptr, *ang = nullptr, *d_w = nullptr, *d_tw = nullptr;
    /// the routing: each token's experts and weights (d_idx, d_tw), then grouped by expert (d_rows, d_w, d_bounds)
    int32_t *d_rows = nullptr, *d_idx = nullptr, *d_bounds = nullptr, *d_cursor = nullptr;
};

/// One routed expert's FP16 weights for the layer's two expert products: gate_up [egu_n][egu_k] and down
/// [edn_n][edn_k] rows, in the save's rotated basis.  A streamed bank dequantizes them on the device from the
/// layer's region; the direct path dequantizes them on the host and uploads what it gets.
class HunyuanImage3Experts {
public:
    virtual ~HunyuanImage3Experts() = default;
    virtual bool fill(int64_t expert, uint16_t* gate_up16, uint16_t* down16, std::string& err) = 0;
};

/// The scratch's buffers for `T` tokens (device USM, one allocation each), freed together by hunyuanimage3_scratch_free;
/// hunyuanimage3_scratch_bytes is what the same would take.  False with `err` on a refused allocation.
bool hunyuanimage3_scratch_alloc(int64_t T, const HunyuanImage3Dims& d, HunyuanImage3Scratch& sc, void* stream, std::string& err);
void hunyuanimage3_scratch_free(HunyuanImage3Scratch& sc);
uint64_t hunyuanimage3_scratch_bytes(int64_t T, const HunyuanImage3Dims& d);

/// One decoder layer on the device: FP32 `x` ([T][hidden]) to FP32 `out` (the same shape; `out` may alias `x`).
/// The sequence's last `n_img` tokens are the image block - a block row sees all T tokens, a text row its prefix,
/// or with `key_lim` (device [T]) the first key_lim[t] keys: an edit's reference image spans see all of themselves
/// (hunyuanimage3_positions_of builds it) - and `pos_y`, `pos_x` are its device 2D rope positions ([T], or null
/// for y = x = the index).  The products are
/// prefill::Gemm's FP16 ones, the attention `qi_attention`'s, and the router's softmax and top-k on the device (only
/// the experts' row counts come back, for the products' sizes).  False with `err`.
bool hunyuanimage3_layer_run(const HunyuanImage3LayerWeights& w, HunyuanImage3Experts& experts, HunyuanImage3Scratch& sc, const float* x, float* out,
                   int64_t T, const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, const HunyuanImage3Dims& d,
                   prefill::Gemm& gemm, void* stream, std::string& err, const int32_t* key_lim = nullptr);

/// HunyuanImage 3.0's decoder layers on the GPU, read straight from a quantized file (no GGUF, no conversion).
class HunyuanImage3Dit {
public:
    HunyuanImage3Dit();
    ~HunyuanImage3Dit();
    HunyuanImage3Dit(const HunyuanImage3Dit&) = delete;
    HunyuanImage3Dit& operator=(const HunyuanImage3Dit&) = delete;

    /// `path`: the quantized safetensors (w4a8 or int8_convrot; either form, the same walk).  The dense products
    /// are dequantized on the device into FP16 and stay there; the norms and the router stay FP32.  The experts'
    /// banks are put on a LayerStream as the file's bytes, which is what a budget below the whole can do best: a
    /// resident prefix, the rest through two slots.  Until `set_bank_budget` says otherwise nothing is resident and
    /// every bank is streamed (the least VRAM, the most PCIe), which is how a small card runs it.  `max_layers`:
    /// fewer than the file holds (a smaller run for a check).  `dry`: the sizes only, nothing on the GPU - such a
    /// part cannot run.  False with `err`.
    bool open(const std::string& path, void* stream, std::string& err, int64_t max_layers = 0, bool dry = false);

    const HunyuanImage3Dims& dims() const { return dims_; }
    int64_t layers() const { return n_layers_; }
    /// The mapped file, once open (null before).  The DiT reads its bytes where they lie, so it owns it.
    const HunyuanImage3File* file() const { return f_.get(); }

    /// The VRAM the dense weights take, the banks' bytes all together, and the least a streamed walk takes (two
    /// slots of the largest bank).
    uint64_t dense_bytes() const { return dense_bytes_; }
    uint64_t bank_bytes() const { return banks_.total(); }
    uint64_t bank_min() const { return banks_.min_budget(); }
    /// The VRAM the banks may take: that much resident as a prefix, the rest streamed.  Below two banks it throws,
    /// as LayerStream does.  Set it before `set_sequence`.
    void set_bank_budget(uint64_t bytes);
    int resident_banks() const { return banks_.resident(); }
    /// Frees the streamed banks' copies in RAM (LayerStream's homes).
    void drop_bank_homes() { banks_.drop_homes(); }
    uint64_t bank_vram() const { return banks_.vram(); }
    /// What the part holds on the GPU now: the dense weights, the banks and the sequence's buffers.
    uint64_t held() const;
    /// What a `T`-token sequence needs besides the weights: the layer's scratch and the two hidden rows.
    uint64_t activation_bytes(int64_t T) const { return hunyuanimage3_scratch_bytes(T, dims_) + 2 * row_bytes(T); }
    /// The bytes a bank's region takes (the layer's quantized pieces, 256-byte aligned).
    uint64_t bank_bytes(int64_t l) const { return banks_.bytes((int) l); }

    /// The sequence the walk runs on: `T` tokens, the last `n_img` of them the image block, whose 2D rope positions
    /// are the host arrays `pos_y` and `pos_x` ([T]); `key_lim` (host [T], or null): the rows' key limits before the
    /// block (an edit's reference spans).  Grows the scratch and the hidden rows for `T`.
    bool set_sequence(int64_t T, int64_t n_img, const int32_t* pos_y, const int32_t* pos_x, std::string& err,
                      const int32_t* key_lim = nullptr);
    int64_t tokens() const { return T_; }

    /// Layers [0, n) over that sequence: the device FP32 [T][hidden] `x` in, the last layer's output in the device
    /// FP32 `out` (which may be `x`).  `layer_done`, if set, is called after layer l with that layer's output on
    /// the device, before the next layer overwrites it.  False with `err`.
    bool forward(const float* x, float* out, int64_t n,
                 const std::function<void(int64_t, const float*)>& layer_done, std::string& err);

    /// One layer, `l`, on the current sequence: device FP32 [T][hidden] `x` in, the device FP32 [T][hidden] `out`
    /// (which may be `x`).  This is what `forward` walks; run on its own it is how one layer's own error is told
    /// from the error a whole walk accumulates - the reference's hidden states go in, not this walk's.
    bool layer(int64_t l, const float* x, float* out, std::string& err);

    /// The sequence's scratch, with the intermediates the last `layer`/`forward` left in it: `probe stages` reads
    /// them to hold one layer's walk to the golden stage by stage (the walk itself never does).  Every buffer but
    /// `gu32`/`act16`/`eg32`/`gact16`/`ed32`/`gx16` is the whole sequence's, and the products' rows past `T` are the
    /// attention's padding.  Empty (`T == 0`) with no sequence set.
    const HunyuanImage3Scratch& scratch() const { return sc_; }

    /// Frees the sequence's buffers (the weights stay).
    void drop_sequence();

private:
    struct Dense {
        uint16_t *qkv = nullptr, *o = nullptr, *sgu = nullptr, *sdn = nullptr;
        float *n_in = nullptr, *n_post = nullptr, *n_q = nullptr, *n_k = nullptr, *wg = nullptr;
        int64_t qkv_n = 0, qkv_k = 0, o_n = 0, o_k = 0, sgu_n = 0, sgu_k = 0, sdn_n = 0, sdn_k = 0;
    };
    /// One layer's expert bank: its two matrices' shapes and quantization, and the index of each piece's span
    /// within the layer's device region (-1: the form does not have that piece).
    struct Bank {
        static constexpr int kEguW = 0, kEguSrel = 1, kEguSchan = 2, kEguCb = 3, kEdnW = 4, kEdnSrel = 5,
                             kEdnSchan = 6, kEdnCb = 7, kSpans = 8;
        HunyuanImage3Matrix egu, edn;
        int spans[kSpans] = {-1, -1, -1, -1, -1, -1, -1, -1};
    };
    /// The `HunyuanImage3Experts` of a bank that sits in a LayerStream region: the file's bytes, on the device.
    struct RegionExperts : HunyuanImage3Experts {
        const HunyuanImage3Dit* dit = nullptr;
        int layer = 0;
        const Bank* bank = nullptr;
        const uint8_t* region = nullptr;
        void* s = nullptr;
        bool fill(int64_t expert, uint16_t* gate_up16, uint16_t* down16, std::string& err) override;
    };

    uint64_t row_bytes(int64_t T) const { return (uint64_t) T * (uint64_t) dims_.hidden * sizeof(float); }
    /// One layer with its bank begun and ended around it (the walk and the single-layer check both go through it).
    bool run_layer(int64_t l, const float* x, float* out, std::string& err);
    /// The start of one of a bank's pieces within its region (throws for a piece the form does not have).
    uint64_t bank_offset(int layer, int span) const;
    /// The layer's dense products: the file's bytes up, dequantized on the device into the resident FP16, then the
    /// norms and the router as FP32.
    bool upload_dense(int64_t l, Dense& d, std::string& err);
    /// The same products' shapes out of the header and nothing else: what `dense_bytes` reports for a dry part.
    bool dense_shapes(int64_t l, Dense& d, std::string& err);
    /// The layer's bank: its shapes, and its pieces' spans appended to `spans` (the caller's list, which becomes
    /// the layer's device region).
    bool add_bank(int64_t l, Bank& b, std::vector<LayerStream::Span>& spans, std::string& err);

    std::unique_ptr<HunyuanImage3File> f_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    HunyuanImage3Dims dims_;
    int64_t n_layers_ = 0;
    bool dry_ = false;
    uint64_t dense_bytes_ = 0;
    std::vector<Dense> dense_;
    std::vector<Bank> bank_refs_;
    LayerStream banks_;
    uint64_t bank_budget_ = 0;
    bool budget_set_ = false;
    // the sequence
    int64_t T_ = 0, n_img_ = 0;
    HunyuanImage3Scratch sc_;
    int32_t *d_pos_y_ = nullptr, *d_pos_x_ = nullptr, *d_key_lim_ = nullptr;
    bool has_key_lim_ = false;
    float *h_ = nullptr, *h2_ = nullptr;
    std::vector<void*> owned_;       // the dense weights, freed with the part
    std::vector<void*> seq_owned_;   // the sequence's buffers, freed by drop_sequence()
};

}  // namespace strata::image
