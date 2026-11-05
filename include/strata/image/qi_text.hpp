// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_text.hpp - the image model's text encoder (Qwen3-VL's language part, its GGUF: 8B for
// Qwen-Image, 4B for Mage-Flow, whose heads are wider than its hidden states) on the GPU:
// the token ids of the prompt (the server tokenizes it in the image model's chat form) to the last block's output
// states, before the final norm (Qwen-Image) or after it (Mage-Flow), the DiT's context (the rows after the system turn).
#pragma once

#include "strata/image/image_lora.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::image {

class QiText {
public:
    QiText();
    ~QiText();
    QiText(const QiText&) = delete;
    QiText& operator=(const QiText&) = delete;

    /// `dry`: read for its sizes only (layer_bytes, layer_min), nothing put on the GPU; such a part cannot run.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// A reference image in the prompt: its `rows` image pads from ids[index] (one per merged patch, grid_h x grid_w in
    /// rows) take the vision encoder's embeddings `main` [rows][hidden], and `deepstack[i]` [rows][hidden] is added to
    /// them after layer i (host FP32).
    struct Image {
        int64_t index = 0, rows = 0, grid_h = 0, grid_w = 0;
        const float* main = nullptr;
        std::vector<const float*> deepstack;
    };

    /// The last block's output [ids.size()][hidden] (host FP32) for the prompt's token ids (and its images, in order),
    /// through the final norm when set_final_norm(true).
    std::vector<float> encode(const std::vector<int32_t>& ids, const std::vector<Image>& images = {});
    /// Whether encode applies the model's final RMS norm (output_norm.weight): Mage-Flow takes the states after it.
    void set_final_norm(bool on);
    /// Whether an image's tokens take plain positions, each its index like the text's (Mage-Flow's encoder passes
    /// them so), not Qwen3-VL's M-RoPE ones by the image's rows and columns (Qwen-Image's).
    void set_flat_positions(bool on) { flat_positions_ = on; }

    int64_t hidden() const { return C_; }
    /// The layers' bytes (in the file) and the two slots a streamed walk takes at least.
    uint64_t layer_bytes() const { return layers_stream_.total(); }
    uint64_t layer_min() const { return layers_stream_.min_budget(); }
    /// The VRAM the layers may take (LayerStream: those that fit stay, the rest are streamed).
    void set_layer_budget(uint64_t bytes) { layers_stream_.configure(bytes, s_); }
    int resident_layers() const { return layers_stream_.resident(); }
    uint64_t layers_vram() const { return layers_stream_.vram(); }
    void drop_layers() { layers_stream_.drop(); }
    void drop_layer_homes() { layers_stream_.drop_homes(); }
    /// The adapters' text encoder entries (lora_split's `te`: sd-scripts' lora_te_layers_N_..., ComfyUI's and
    /// diffusers' names of the layers' q, k, v, o, gate, up and down projections) added to the layers' products from
    /// the next encode: a plain LoRA, a LoKr, a LoHa or a whole delta (a DoRA, an OFT, a GLoRA or a bias's delta is
    /// refused, as a key the encoder has no weight for), false with err and nothing bound.  The same set again changes
    /// nothing.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    const std::string& lora_key() const { return lora_key_; }
    uint64_t lora_bytes() const;
    /// The working buffers an encode of T tokens takes; drop_activations frees them (grown again on use).
    uint64_t activation_bytes(int64_t T) const;
    void drop_activations();

private:
    struct W {
        int type = 0;
        int64_t n = 0, k = 0;
        const void* d = nullptr;
        LoraLayer* lora = nullptr;   // the adapters' deltas for its products (null: none)
    };
    // a product's output y (row stride ldy) plus the weight's adapters' deltas of the input x16
    void lora_add(const W& w, const uint16_t* x16, int64_t T, float* y, int64_t ldy);
    std::vector<std::unique_ptr<LoraLayer>> lora_layers_;
    std::vector<std::shared_ptr<const LoraFile>> lora_files_;
    std::string lora_key_;
    int64_t lora_cols_ = 0, lora_t_ = 0;
    float* lora_f32_ = nullptr;
    uint16_t* lora16_ = nullptr;
    struct Layer {
        W q, k, v, o, gate, up, down;   // d: the offset within the layer's region (LayerStream)
        uint64_t attn_norm = 0, ffn_norm = 0, qn = 0, kn = 0;
    };
    void ensure(int64_t T);
    int sec_h_ = 20, sec_w_ = 20;   // the M-RoPE's pairs by row and by column (rope.dimension_sections)

    std::unique_ptr<GgufFile> g_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    int64_t C_ = 4096, heads_ = 32, kv_heads_ = 8, head_dim_ = 128, inter_ = 12288;
    float theta_ = 5e6f, eps_ = 1e-6f;
    int embd_type_ = 0;
    bool final_norm_ = false, flat_positions_ = false;
    std::vector<float> out_norm_;   // output_norm.weight (host), read when set_final_norm(true)
    uint64_t embd_row_bytes_ = 0;
    int64_t embd_rows_ = 0;
    const uint8_t* embd_ = nullptr;   // the token embeddings in the mapped file (rows dequantized per prompt)
    std::vector<Layer> layers_;
    LayerStream layers_stream_;
    int64_t cap_T_ = 0;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *h_ = nullptr, *gu_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr;
};

}  // namespace strata::image
