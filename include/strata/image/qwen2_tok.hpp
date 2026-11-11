// include/strata/image/qwen2_tok.hpp - the token ids of Qwen-Image-2.1's text encoder, as stable-diffusion.cpp 2988060
// builds them (src/image/te/qwen2_tok.cpp): its Qwen2 byte-level BPE (tokenizers/qwen2_tokenizer.cpp, bpe_tokenizer.cpp),
// its prompt-weight parser (core/util.cpp parse_prompt_attention, the A1111 syntax) and LLMEmbedder's chat template
// (conditioning/conditioner.hpp, VERSION_QWEN_IMAGE_2_1, no reference pictures).
//
// Gate (tools/test_te_tokenizer.py): the same ids, weights and system-token count as sd.cpp's own on the cases of
// tools/image/gate/te_tok_cases.py (gate and long prompts, every special literal, prompt-weight syntax, unicode,
// empty / whitespace prompts, long words, invalid UTF-8, 300 random prompts), recorded from sd.cpp through the stage
// hook (tools/image/gate/data/te_tok_expected.json).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

class Qwen2Tok {
public:
    // The merges of a Qwen-Image text-encoder GGUF (tokenizer.ggml.merges; for Qwen3-VL-8B the same 151,387 merges
    // sd.cpp compiles in, checked 11-11).  nullptr and *err on failure.
    static std::unique_ptr<Qwen2Tok> from_gguf(const std::string& path, std::string* err);
    // Merges given with their ranks (tests: the merges a set of prompts uses), n_merges = the full list's length (the
    // special ids follow it).  A merge listed twice keeps its last rank, as sd.cpp's map does.
    static std::unique_ptr<Qwen2Tok> from_ranked(const std::vector<std::pair<int, std::string>>& ranked_merges,
                                                 int n_merges, std::string* err);

    // sd.cpp's BPETokenizer::encode: special literals split out, the Qwen2 pre-tokenizer, byte-level BPE.  false and
    // *err for text that is not valid UTF-8 (sd.cpp's regex refuses it the same way).
    bool encode(const std::string& text, std::vector<int32_t>& ids, std::string* err) const;

    // Test hook: every merge rank encode() looked up and found, and every token string it emitted with its id.
    void set_trace(std::vector<int>* ranks_used, std::vector<std::pair<std::string, int>>* tokens_emitted) const;

    struct Impl;
    ~Qwen2Tok();

private:
    explicit Qwen2Tok(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// core/util.cpp parse_prompt_attention, as its code runs (not as its comment says: "\]" stays "\]", only "\(" loses
// its backslash).
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);

// sd-cli's extract_and_remove_lora (examples/common/common.cpp): with its default --lora-model-dir "." every
// "<lora:NAME:WEIGHT>" tag leaves the prompt before the text encoder sees it, whether or not a LoRA file is found.
std::string strip_lora_tags(const std::string& prompt);

struct QwenImageTokens {
    std::vector<int32_t> ids;      // the whole prompt, template included
    std::vector<float> weights;    // per id, the prompt weight (1 for plain text, -1 for a BREAK)
    int n_system = 0;              // prompt_template_encode_start_idx: the leading ids the condition drops
};

// strip_lora_tags, then LLMEmbedder::get_learned_condition + tokenize for VERSION_QWEN_IMAGE_2_1: the system / user / assistant template,
// the user text through parse_prompt_attention, each piece encoded on its own.  "" on success.
std::string qwen_image_tokens(const Qwen2Tok& tok, const std::string& prompt, QwenImageTokens* out);

}  // namespace strata::image
