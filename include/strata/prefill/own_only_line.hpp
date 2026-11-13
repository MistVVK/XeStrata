// include/strata/prefill/own_only_line.hpp - the start-up line of an engine built without llama.cpp's MMQ prompt kernels: which kernels a prompt
// chunk runs on, per layer count, as the rule functions that pick the path answer for this machine (prefill.cpp makes the counts, this header words them).
// A pure function: no GPU, no model; own_only_line_test builds it with the four cases of the wording.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::prefill {

/// Layers per prompt path for one class of chunk (long = at or above the long-prompt floor, short = below it).  The four counts add up to the layer count.
struct OwnOnlySplit {
    int64_t fused = 0;    ///< the fused kernels (at the floor and above)
    int64_t staged = 0;   ///< the staged group walk's fused entry (chunks below the floor)
    int64_t i8 = 0;       ///< the int8 expert kernels
    int64_t fp16 = 0;     ///< the dequantize + FP16 GEMM path
    bool operator==(const OwnOnlySplit& o) const { return fused == o.fused && staged == o.staged && i8 == o.i8 && fp16 == o.fp16; }
};

/// One layer into the two splits (prefill.cpp's start-up counts).  `fo`: the fused kernels take the layer at the floor and above.  `i8`: the int8 kernels take it.  `i8_gated`: the int8 default is for chunks under
/// the floor only.  `entry_on`: the staged group entry is on for fused-only layers.  `unfolded`: the layer is not Hadamard-folded and has no tail (the staged entry and the int8 kernels read neither, so a folded layer
/// below the floor runs each routed expert on the FP16 path).
inline void own_only_count_layer(OwnOnlySplit& big, OwnOnlySplit& small, bool fo, bool i8, bool i8_gated, bool entry_on, bool unfolded) {
    if (fo) ++big.fused; else if (i8 && !i8_gated) ++big.i8; else ++big.fp16;   // (a gated class: no int8 in the long class)
    if (fo && entry_on && unfolded) ++small.staged; else if (i8) ++small.i8; else ++small.fp16;
}

namespace own_only_detail {
inline std::string parts(const OwnOnlySplit& s, int64_t layers, bool staged_name, bool with_total) {
    std::vector<std::string> v;
    const int64_t a = staged_name ? s.staged : s.fused;
    if (a > 0) v.push_back(std::to_string(a) + " on the " + (staged_name ? "staged fused entry" : "fused kernels"));
    if (s.i8 > 0) v.push_back(std::to_string(s.i8) + " on the int8 kernels");
    if (s.fp16 > 0) v.push_back(std::to_string(s.fp16) + " on the FP16 path");
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) out += (i == 0 ? "" : (i + 1 == v.size() ? " and " : ", ")) + v[i];
    // the first part of the sentence's first clause carries "N of L layers"
    if (with_total && !out.empty()) {
        const size_t at = out.find(" on ");
        out = out.substr(0, at) + " of " + std::to_string(layers) + " layers" + out.substr(at);
    }
    return out;
}
}  // namespace own_only_detail

/// One plain sentence.  `big` runs at `floor_tokens` tokens or more, `small` below it; equal splits (no staged entry, same counts) say "every chunk".
/// `peer_off`: --peer-device is on, so the fused kernels are off and the counts are the ones that result.
inline std::string own_only_line(int64_t layers, int64_t floor_tokens, const OwnOnlySplit& big, const OwnOnlySplit& small, bool peer_off) {
    std::string s = "strata: prompt kernels: ";
    const bool equal = small.staged == 0 && big == small;
    if (equal) {
        s += "every chunk runs " + own_only_detail::parts(big, layers, false, true);
    } else {
        s += "chunks of " + std::to_string(floor_tokens) + " tokens or more run " + own_only_detail::parts(big, layers, false, true) +
             "; shorter chunks run " + own_only_detail::parts(small, layers, small.staged > 0, false);
        // a shorter chunk with both a staged entry and a fused count is not a state the rule produces; the staged part is the one worded
    }
    if (peer_off) s += " (--peer-device: the fused kernels are off)";
    s += ".";
    return s;
}

}  // namespace strata::prefill
