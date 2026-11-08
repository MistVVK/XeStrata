// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/shared_options.hpp - the engine options that Qwen3.8-Flash-Next's engine (generate.cpp) and
// the other models' (plain_generate.cpp) take with one meaning, parsed in one place: each engine points the fields at
// its own options (a null field: that engine does not take the option, and it falls to the engine's own parsing).
// One parser, so an option the two take cannot come to mean two things (PlainModel had --adapt-min-gain's 1.5 fixed
// in its code while the other engine took the option, 943abb056).
#pragma once

#include "strata/prefill/prefill.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::program {

struct SharedOptionFields {
    const char* who = "xestrata";   ///< the engine's name in its messages
    // the adaptive tier
    int* adapt_every = nullptr;
    bool* adapt_every_given = nullptr;
    int* adapt_swaps = nullptr;
    float* adapt_min_gain = nullptr;
    float* adapt_decay = nullptr;
    std::string* expert_profile = nullptr;
    std::string* expert_profile_save = nullptr;
    double* expert_profile_save_min = nullptr;
    // the context, the output and the sampling
    int64_t* max_context = nullptr;
    int64_t* max_new = nullptr;
    float* temperature = nullptr;
    int* top_k = nullptr;
    float* top_p = nullptr;
    uint64_t* seed = nullptr;
    bool* greedy = nullptr;   ///< --seed turns it off
    // speculative decoding
    int* spec = nullptr;
    std::string* spec_window = nullptr;
    bool* spec_window_given = nullptr;
    bool spec_window_all = false;   ///< --spec-window all is taken (PlainModel's every-draft window)
    double* spec_min_p = nullptr;
    // memory
    bool* mmap_experts = nullptr;
    int64_t* vram_free_mib = nullptr;
    bool* vram_elastic = nullptr;
    int64_t* vram_segment_mib = nullptr;
    int* pool_workers = nullptr;
    bool prefill_cpu_share = false;   ///< --prefill-cpu-share (prefill::set_cpu_share) is taken
    // the rotation
    std::string* rope_scaling = nullptr;
    double* rope_scale = nullptr;
    double* rope_freq_base = nullptr;
    double* rope_freq_scale = nullptr;
    double* yarn_orig_ctx = nullptr;
};

/// 1: `a` was one of them (its value taken through `next(name)`), 0: it is not (or this engine does not take it),
/// -1: a value it refuses (said on stderr; the engine stops with 2).
template <class Next>
int parse_shared_option(const std::string& a, Next&& next, const SharedOptionFields& f) {
    auto i64 = [&](int64_t* p) { *p = std::strtoll(next(a.c_str()), nullptr, 10); return 1; };
    auto f64 = [&](double* p) { *p = std::strtod(next(a.c_str()), nullptr); return 1; };
    if (a == "--adapt-every" && f.adapt_every) {
        *f.adapt_every = (int) std::strtol(next(a.c_str()), nullptr, 10);
        if (f.adapt_every_given) *f.adapt_every_given = true;
        return 1;
    }
    if (a == "--adapt-swaps" && f.adapt_swaps) { *f.adapt_swaps = (int) std::strtol(next(a.c_str()), nullptr, 10); return 1; }
    if (a == "--adapt-min-gain" && f.adapt_min_gain) { *f.adapt_min_gain = std::strtof(next(a.c_str()), nullptr); return 1; }
    if (a == "--adapt-decay" && f.adapt_decay) {
        // inside 0..1 it forgets old routing; above 1 the counts grow without end and the swap gains turn NaN, which
        // breaks the sort that orders the swaps: an unusable value runs with the default, said (upstream 4a7e373da)
        const double v = std::strtod(next(a.c_str()), nullptr);
        *f.adapt_decay = std::isfinite(v) && v >= 0.0 && v <= 1.0 ? (float) v : 0.7f;
        if (*f.adapt_decay != (float) v)
            std::fprintf(stderr, "xestrata: --adapt-decay %g is not between 0 and 1; using 0.7 (counts above 1 grow without "
                                 "end and the swap gains turn NaN)\n", v);
        return 1;
    }
    if (a == "--expert-profile" && f.expert_profile) { *f.expert_profile = next(a.c_str()); return 1; }
    if (a == "--expert-profile-save" && f.expert_profile_save) { *f.expert_profile_save = next(a.c_str()); return 1; }
    if (a == "--expert-profile-save-every" && f.expert_profile_save_min) return f64(f.expert_profile_save_min);
    if (a == "--max-context" && f.max_context) return i64(f.max_context);
    if (a == "--max-new" && f.max_new) return i64(f.max_new);
    if (a == "--temperature" && f.temperature) { *f.temperature = std::strtof(next(a.c_str()), nullptr); return 1; }
    if (a == "--top-k" && f.top_k) { *f.top_k = (int) std::strtol(next(a.c_str()), nullptr, 10); return 1; }
    if (a == "--top-p" && f.top_p) { *f.top_p = std::strtof(next(a.c_str()), nullptr); return 1; }
    if (a == "--seed" && f.seed) {
        *f.seed = std::strtoull(next(a.c_str()), nullptr, 10);
        if (f.greedy) *f.greedy = false;
        return 1;
    }
    if (a == "--spec" && f.spec) { *f.spec = (int) std::strtol(next(a.c_str()), nullptr, 10); return 1; }
    if (a == "--spec-window" && f.spec_window) {
        *f.spec_window = next(a.c_str());
        if (f.spec_window_given) *f.spec_window_given = true;
        const std::string& v = *f.spec_window;
        if (v != "auto" && v != "min-p" && !(f.spec_window_all && v == "all")) {
            std::fprintf(stderr, "%s: --spec-window takes auto%s or min-p, not %s\n", f.who,
                         f.spec_window_all ? ", all" : "", v.c_str());
            return -1;
        }
        return 1;
    }
    if (a == "--spec-min-p" && f.spec_min_p) {
        *f.spec_min_p = std::clamp(std::strtod(next(a.c_str()), nullptr), 0.0, 1.0);
        return 1;
    }
    if (a == "--mmap-experts" && f.mmap_experts) { *f.mmap_experts = true; return 1; }
    if (a == "--vram-free-mib" && f.vram_free_mib) {
        *f.vram_free_mib = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10));
        return 1;
    }
    if (a == "--vram-elastic" && f.vram_elastic) { *f.vram_elastic = true; return 1; }
    if (a == "--vram-segment-mib" && f.vram_segment_mib) return i64(f.vram_segment_mib);
    if (a == "--pool-workers" && f.pool_workers) { *f.pool_workers = (int) std::strtol(next(a.c_str()), nullptr, 10); return 1; }
    if (a == "--prefill-cpu-share" && f.prefill_cpu_share) {
        const std::string v = next(a.c_str());
        strata::prefill::set_cpu_share(v == "auto" ? -1.0 : std::max(0.0, std::strtod(v.c_str(), nullptr)));
        return 1;
    }
    if (a == "--rope-scaling" && f.rope_scaling) { *f.rope_scaling = next(a.c_str()); return 1; }
    if (a == "--rope-scale" && f.rope_scale) return f64(f.rope_scale);
    if (a == "--rope-freq-base" && f.rope_freq_base) return f64(f.rope_freq_base);
    if (a == "--rope-freq-scale" && f.rope_freq_scale) return f64(f.rope_freq_scale);
    if (a == "--yarn-orig-ctx" && f.yarn_orig_ctx) return f64(f.yarn_orig_ctx);
    return 0;
}

}  // namespace strata::program
