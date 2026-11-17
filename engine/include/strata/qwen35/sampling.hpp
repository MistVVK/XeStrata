#pragma once
// The qwen35 engine's host sampler settings and the repeat / frequency / presence penalties, header-only so a unit test
// (src/qwen35/sampling_test.cpp) can run them without a GPU.
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "strata/kernels/sampler.hpp"   // penalty_counts_prompt (declarations only; no GPU needed)

namespace strata::qwen35 {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The penalties on the logit row `lg` for the tokens of the last `s.last_n` (all when < 0) history tokens.  A row can be
/// shorter than the vocabulary (the MTP draft head picks among the first N token ids only, --draft-vocab N): a history
/// token that is not in the row (id < 0 or >= lg.size()) is not in it and takes no penalty - it used to index past the
/// row (#3911).
inline void apply_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

/// #4590: the history the penalties start from for one request.  By default only the current reply counts: the history starts
/// at the last prompt token (the one the first decode step feeds in) and grows with the tokens written, so the prompt's own
/// numbers and a quoted earlier answer are not penalised.  `counts_prompt` (STRATA_PENALTY_PROMPT=1) is the old window, the
/// whole prompt.  Same window as src/program/generate.cpp.
inline std::vector<int32_t> penalty_history_start(const std::vector<int32_t>& prompt, bool counts_prompt) {
    if (counts_prompt || prompt.empty()) return prompt;
    return std::vector<int32_t>(1, prompt.back());
}

}  // namespace strata::qwen35
