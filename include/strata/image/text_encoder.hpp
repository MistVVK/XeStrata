// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/text_encoder.hpp - the image models' text encoders through the chat engine's dense model
// (core::PlainModel, opened as an encoder): MOSS-Image's Qwen3.5-9B, Qwen-Image's Qwen3-VL-8B, Mage-Flow's Qwen3-VL-4B
// and Anima's Qwen3-0.6B-Base (their GGUFs).  The prompt's token ids in the image model's chat form (one prompt at a
// time, no padding) to every token's state - after the final norm, or before it (Qwen-Image's) - the DiT's context.
// Its blocks' matrices go through a LayerStream: those that fit the budget stay in VRAM, the rest are copied from the
// mapped file as the prompt reaches them.  When the chat's engine reads the same GGUF (MOSS-Image), the server asks it
// instead (its ENCODE command), and the states come in a file (read_states).
#pragma once

#include "strata/image/image_lora.hpp"
#include "strata/image/qi_stream.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
class PlainModel;
}

namespace strata::image {

class TextEncoder {
public:
    TextEncoder();
    ~TextEncoder();
    TextEncoder(const TextEncoder&) = delete;
    TextEncoder& operator=(const TextEncoder&) = delete;

    /// Opens the GGUF for prompts of up to `max_tokens` tokens, its blocks within `layer_budget` bytes of VRAM (all of
    /// them by default; two slots at least).  `dry`: the file read for its sizes only (hidden, layer_bytes, layer_min,
    /// activation_bytes), nothing but the blocks' norms on the GPU; such an encoder cannot encode.
    bool open(const std::string& gguf, int64_t max_tokens, std::string& err, uint64_t layer_budget = UINT64_MAX,
              bool dry = false);
    bool is_open() const { return model_ != nullptr && !dry_; }
    /// Frees the weights and the buffers (open again for the next text).
    void close();
    /// Whether encode puts the rows through the model's final RMS norm (the default; Qwen-Image takes them before it).
    void set_final_norm(bool on) { final_norm_ = on; }
    /// Whether a picture's tokens take plain positions, each its index like the text's (Mage-Flow's encoder passes
    /// them so), not Qwen3-VL's M-RoPE ones by the picture's rows and columns.
    void set_flat_positions(bool on) { flat_positions_ = on; }
    /// A picture among the prompt's tokens: its rows [nx ny][hidden()] (host, the vision encoder's, the merged grid's
    /// rows one after the other) take the place of the tokens' embeddings, from token `index` (-1: the next run of
    /// <|image_pad|> tokens, image_pad()), the grid's columns nx and rows ny; `deepstack[i]` [nx ny][hidden()] (host) is
    /// added to the picture's rows after layer i (Qwen3-VL's DeepStack).
    struct Picture {
        const float* rows = nullptr;
        int64_t nx = 0, ny = 0;
        int64_t index = -1;
        std::vector<const float*> deepstack;
    };
    /// [ids.size()][hidden()] floats (host); empty with `err` on a failure.  The pictures' M-RoPE positions are the
    /// chat engine's (mtmd's: t = p, h = p + y, w = p + x; the text after at p + max(nx, ny)).
    std::vector<float> encode(const std::vector<int32_t>& ids, std::string& err,
                              const std::vector<Picture>& pictures = {});
    /// The <|image_pad|> token's id (-1: the vocabulary has none).
    int32_t image_pad() const { return image_pad_; }
    /// The same id read straight from a GGUF's vocabulary, for a model that needs the id and no encoder yet
    /// (MOSS-Image: opening an encoder, even for its sizes, takes the blocks' resident weights onto the GPU).
    static int32_t pad_id(const std::string& gguf);
    int64_t hidden() const { return hidden_; }
    /// The adapters' text encoder entries (lora_split's `te`: sd-scripts' lora_te_layers_N_... or the model's own
    /// dotted names) added after the blocks' products from the next encode: a plain LoRA, a LoKr, a LoHa or a whole
    /// delta (Qwen3.5's GDN weights: a plain LoRA or a whole delta, their value heads in the GGUF's order, the order
    /// llama.cpp's converter wrote them in, the entries' rows or columns put in it as they are bound); another form, or
    /// a key the encoder has no weight for, false with err and nothing bound.  The same set again changes nothing.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    const std::string& lora_key() const;
    /// The bound adapters' device bytes.
    uint64_t lora_bytes() const;
    /// The blocks' bytes (in the file) and the two slots a streamed walk takes at least.
    uint64_t layer_bytes() const { return layers_ != nullptr ? layers_->total() : 0; }
    uint64_t layer_min() const { return layers_ != nullptr ? layers_->min_budget() : 0; }
    /// The VRAM the blocks may take (LayerStream: those that fit stay, the rest are streamed; two slots at least).
    void set_layer_budget(uint64_t bytes);
    int resident_layers() const { return layers_ != nullptr ? layers_->resident() : 0; }
    uint64_t layers_vram() const { return layers_ != nullptr ? layers_->vram() : 0; }
    /// The resident layers given back (streamed again on the next encode), and their homes in pinned RAM freed.
    void drop_layers() {
        if (layers_ != nullptr && !dry_) layers_->drop();
    }
    void drop_layer_homes() {
        if (layers_ != nullptr && !dry_) layers_->drop_homes();
    }
    /// The working buffers an encode of T tokens makes (kept for the next prompt as long); drop_activations frees them
    /// (made again on use).
    uint64_t activation_bytes(int64_t T) const;
    void drop_activations();
    /// The chat engine's ENCODE file (plain_generate.cpp): int32 'STE1', the rows n and their width, then n rows of
    /// floats.  False with the reason when it is not one.
    static bool read_states(const std::string& path, std::vector<float>& rows, int64_t& n, int64_t& width,
                            std::string& err);

private:
    struct Source;
    struct Hook;
    struct Deep;
    std::unique_ptr<Hook> hook_;
    std::unique_ptr<Deep> deep_;
    std::unique_ptr<LayerStream> layers_;
    std::unique_ptr<Source> source_;
    std::unique_ptr<core::PlainModel> model_;
    bool dry_ = false;   // a dry open: sizes only
    int32_t image_pad_ = -1;
    int64_t hidden_ = 0;
    int64_t pos_set_ = 0;   // the cells whose M-RoPE positions an earlier prompt's pictures set (not the identity)
    bool final_norm_ = true, flat_positions_ = false;
};

}  // namespace strata::image
