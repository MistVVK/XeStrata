// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_text.hpp - MOSS-Image's text encoder: Qwen3.5-9B (its GGUF: the official one by Unsloth or
// HauhauCS's) through the chat engine's dense model (core::PlainModel, opened as an encoder), the prompt's token ids in
// MOSS-Image's chat form (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`, one prompt at a time, no
// padding) to every token's state after the final norm, the DiT's context.  Its blocks' matrices go through a
// LayerStream: those that fit the budget stay in VRAM, the rest are copied from the mapped file as the prompt reaches
// them.  When the chat's engine reads the same GGUF, the server asks it instead (its ENCODE command), and the states
// come in a file (read_states).
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

class MossText {
public:
    MossText();
    ~MossText();
    MossText(const MossText&) = delete;
    MossText& operator=(const MossText&) = delete;

    /// Opens the GGUF for prompts of up to `max_tokens` tokens (MOSS-Image's 4096 at most), its blocks within
    /// `layer_budget` bytes of VRAM (all of them by default; two slots at least).
    bool open(const std::string& gguf, int64_t max_tokens, std::string& err, uint64_t layer_budget = UINT64_MAX);
    bool is_open() const { return model_ != nullptr; }
    /// Frees the weights and the buffers (open again for the next text).
    void close();
    /// A picture among the prompt's tokens (an ai-toolkit edit adapter's: the text encoder sees the references): its
    /// rows [nx ny][hidden()] (host, the vision encoder's, the merged grid's rows one after the other), the grid's
    /// columns nx and rows ny.
    struct Picture {
        const float* rows;
        int64_t nx, ny;
    };
    /// [ids.size()][hidden()] floats (host); empty with `err` on a failure.  `pictures` take the runs of <|image_pad|>
    /// tokens (image_pad()) in order, nx ny each, their rows in place of the tokens' embeddings and their M-RoPE
    /// positions the chat engine's (mtmd's: t = p, h = p + y, w = p + x; the text after at p + max(nx, ny)).
    std::vector<float> encode(const std::vector<int32_t>& ids, std::string& err,
                              const std::vector<Picture>& pictures = {});
    /// The <|image_pad|> token's id (-1: the vocabulary has none).
    int32_t image_pad() const { return image_pad_; }
    int64_t hidden() const;
    /// The adapters' text encoder entries (lora_split's `te`: Qwen3.5's names, sd-scripts' lora_te_layers_N_... or
    /// the model's own) added after the blocks' products from the next encode: a plain LoRA or a whole delta (the GDN
    /// weights' value heads in the GGUF's order, the order llama.cpp's converter wrote them in, the entries' rows or
    /// columns put in it as they are bound); another form, or a key the encoder has no weight for, false with err.
    bool set_loras(const std::vector<LoraUse>& loras, std::string& err);
    const std::string& lora_key() const;

    /// The blocks' bytes (in the file) and the two slots a streamed walk takes at least.
    uint64_t layer_bytes() const { return layers_ != nullptr ? layers_->total() : 0; }
    uint64_t layer_min() const { return layers_ != nullptr ? layers_->min_budget() : 0; }
    /// The VRAM the blocks may take (LayerStream: those that fit stay, the rest are streamed; two slots at least).
    void set_layer_budget(uint64_t bytes);
    int resident_layers() const { return layers_ != nullptr ? layers_->resident() : 0; }
    uint64_t layers_vram() const { return layers_ != nullptr ? layers_->vram() : 0; }

    /// The chat engine's ENCODE file (plain_generate.cpp): int32 'STE1', the rows n and their width, then n rows of
    /// floats.  False with the reason when it is not one.
    static bool read_states(const std::string& path, std::vector<float>& rows, int64_t& n, int64_t& width,
                            std::string& err);

private:
    struct Source;
    struct Hook;
    std::unique_ptr<Hook> hook_;
    std::unique_ptr<LayerStream> layers_;
    std::unique_ptr<Source> source_;
    std::unique_ptr<core::PlainModel> model_;
    int32_t image_pad_ = -1;
    int64_t pos_set_ = 0;   // the cells whose M-RoPE positions an earlier prompt's pictures set (not the identity)
};

}  // namespace strata::image
