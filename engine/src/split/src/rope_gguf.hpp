// A model's rotary scaling as llama.cpp at a37cc46 reads it from the GGUF (llama-model.cpp's rope keys); header-only
// beside gguf.hpp so tools/gguf_test.cpp checks it on small written files and on a real model's header
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "gguf.hpp"

namespace fnx {

struct ModelRope {
    double yarn_factor = 0;   // the model's own YaRN factor, 0 for none
    int yarn_orig = 0;        // its original_context_length, 0 when it names none
};

// llama.cpp reads rope.scaling.type, "linear" when the key is absent, and rope.scaling.factor (else the older
// rope.scale_linear), and scales the rotary by that factor unless the type is "none". sFNX's rotary is the plain one
// or YaRN (Engine, choose_rope), so: a type of "none", or a factor of 0 or 1 (which scales nothing), is unscaled; type
// "yarn" with a factor is run from its original_context_length. Anything else is refused rather than run on angles
// llama.cpp would not use: another type with a factor that scales, a compressing factor, llama.cpp's other rotary keys
// at values that are not its defaults, a section order other than its tyxz, a rope_freqs table
inline ModelRope model_rope(const Gguf& g, const std::string& p) {
    const std::string sk = p + "rope.scaling.";
    const std::string type = g.has_key(sk + "type") ? g.s(sk + "type") : "linear";
    if (type != "none" && type != "linear" && type != "yarn")
        throw std::runtime_error("fnx: the model's rope scaling is " + type + "; sFNX follows none, yarn, and linear only unscaled");
    const std::string fk = g.has_key(sk + "factor") ? sk + "factor" : p + "rope.scale_linear";
    const double factor = type == "none" || !g.has_key(fk) ? 0.0 : g.num(fk);
    if (factor != 0.0 && factor < 1.0)
        throw std::runtime_error("fnx: " + fk + " " + std::to_string(factor) + " would compress the rotary, which sFNX does not follow");
    ModelRope r;
    if (factor > 1.0) {
        if (type != "yarn")
            throw std::runtime_error("fnx: the model scales its rotary " + type + " x" + std::to_string(factor) + " (" + fk +
                                     (g.has_key(sk + "type") ? "" : ", no rope.scaling.type: llama.cpp takes that as linear") +
                                     "); sFNX follows yarn only");
        r.yarn_factor = factor;
        if (g.has_key(sk + "original_context_length")) {
            const int64_t n = g.i(sk + "original_context_length");
            if (n <= 0 || n > INT32_MAX) throw std::runtime_error("fnx: " + sk + "original_context_length " + std::to_string(n) + " is not a context length");
            r.yarn_orig = (int) n;
        }
    }
    const std::pair<const char*, double> fixed[] = {
        {"rope.scaling.attn_factor", 1.0},     {"rope.scaling.yarn_log_multiplier", 0.0}, {"rope.scaling.yarn_ext_factor", 1.0},
        {"rope.scaling.yarn_attn_factor", 1.0}, {"rope.scaling.yarn_beta_fast", 32.0},     {"rope.scaling.yarn_beta_slow", 1.0},
        {"rope.scaling.alpha", 0.0}};
    for (const auto& [k, want] : fixed)
        if (g.has_key(p + k) && g.num(p + k) != want)
            throw std::runtime_error("fnx: the model sets " + p + k + " " + std::to_string(g.num(p + k)) + "; sFNX's rotary follows only " +
                                     std::to_string(want));
    if (g.has_key(p + "rope.section_order") && g.s(p + "rope.section_order") != "tyxz")
        throw std::runtime_error("fnx: the model's rope.section_order is " + g.s(p + "rope.section_order") + "; sFNX's M-RoPE follows tyxz");
    if (g.find("rope_freqs.weight")) throw std::runtime_error("fnx: the model has rope_freqs.weight (per-pair rotary factors), which sFNX does not follow");
    return r;
}

}  // namespace fnx
