// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/shared_options.hpp - the engine options that Qwen3.8-Flash-Next's engine (generate.cpp) and
// the other models' (plain_generate.cpp) take with one meaning, parsed in one place: each engine points the fields at
// its own options (a null field: that engine does not take the option, and it falls to the engine's own parsing).
// One parser, so an option the two take cannot come to mean two things (PlainModel had --adapt-min-gain's 1.5 fixed
// in its code while the other engine took the option, 943abb056).
#pragma once

#include "strata/core/window_rows.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::program {

/// --expert-cache means the same thing in both engines: `auto` asks for a capacity-led tier (-1), an explicit
/// 0 is off (a profile must not turn it back on), and a positive number caps the slots.  The engines' defaults
/// are auto; the tier is filled from this run's routing unless a profile is given as a head start.
inline int64_t expert_cache_request(const std::string& value) {
    return value == "auto" || value.empty() ? -1 : (int64_t) std::strtoll(value.c_str(), nullptr, 10);
}

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
    /// --spec auto: the window is one of the candidates the serve engine measures at its start (strata/spec/
    /// draft_probe.hpp), from `spec_auto_default` (its largest) down; a number is kept as given
    bool* spec_auto = nullptr;
    int spec_auto_default = 4;
    std::string* spec_window = nullptr;
    bool* spec_window_given = nullptr;
    bool spec_window_all = false;   ///< --spec-window all is taken (PlainModel's every-draft window)
    double* spec_min_p = nullptr;
    std::string* window_hashes = nullptr;   ///< --window-hashes PATH: a fingerprint line per verify window
    /// --serve, --draft-costs FILE / --draft-costs-key TEXT: the window costs' table (strata/spec/draft_policy.hpp,
    /// DraftCosts) and the server's part of what it is valid for
    std::string* draft_costs = nullptr;
    std::string* draft_costs_key = nullptr;
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
    // the rest of what the two engines take with one meaning
    bool* serve = nullptr;
    bool* vision = nullptr;
    bool* stop_eos = nullptr;
    bool* session_save_reclaim = nullptr;
    bool* prompt_cache_tail = nullptr;
    bool* prompt_cache_spread = nullptr;
    bool* conversation_save_compress = nullptr;
    bool* no_host_worker = nullptr;      ///< generate: true
    bool* pool_host_works = nullptr;     ///< plain: false
    int* batch_mtp_int = nullptr;        ///< generate's int
    bool* batch_mtp_bool = nullptr;      ///< plain's bool
    int* adapt_async_int = nullptr;      ///< generate's int
    bool* adapt_async_bool = nullptr;    ///< plain's bool
    bool* coupled_draft = nullptr;
    int* kv_grow = nullptr;
    /// --kv-resident N|auto|off (the plain engine): -1 auto, 0 off, >0 the resident cells a layer. generate.cpp keeps
    /// its own parse (N, 0 off; setup decides its auto), so it leaves this null
    int64_t* kv_resident = nullptr;
    std::string* conversation_save = nullptr;
    std::string* dump_logits = nullptr;
    std::string* dump_routing = nullptr;
    std::string* pool_cores = nullptr;
    std::string* shared_expert_arena = nullptr;
    std::string* expert_cache_remote_placement = nullptr;
    int* expert_cache = nullptr;         ///< auto -> -1
    bool gpu_consume = false;            ///< --gpu: applied at startup, before the runtime
    std::vector<std::pair<std::string, float>>* cvec_files = nullptr;
    int* cvec_first = nullptr;
    int* cvec_last = nullptr;
    int* cvec_mode = nullptr;
    int* cvec_single = nullptr;
    int* prompt_cache = nullptr;         ///< max 0
    int* suffix_draft = nullptr;         ///< max 0
    int* lookup_chain = nullptr;
    int lookup_chain_lo = 0, lookup_chain_hi = -1;   ///< generate 0..7; plain 0.. (hi < lo: no upper bound)
    int* lookup_chain_min = nullptr;
    int lookup_chain_min_lo = 1;         ///< generate 3; plain 1
    int* pipeline_windows = nullptr;
    int pipeline_windows_hi = -1;        ///< generate 2; plain no upper bound
    bool wide_window = false;            ///< --wide-window auto|off|on (core::set_wide_window, window_rows.hpp)
    int* adapt_mib_per_token = nullptr;
    bool adapt_mib_clamp0 = false;       ///< generate: max 0; plain as it is
    int* batch = nullptr;                ///< --batch / --slots
    bool batch_clamp0 = false;           ///< plain: max 0; generate as it is
    int* expert_cache_remote = nullptr;  ///< --expert-cache-device1/2/3, an array of 3
    bool expert_cache_remote_clamp0 = false;   ///< plain: max 0
    bool* split_roles_helper = nullptr;  ///< --split-roles any|helper (helper: true), --layer-split auto's second GPU
    bool* split_skip_if_fits = nullptr;  ///< --split-skip-if-fits: the first GPU alone when it holds every profiled pair
    int* pool_tasks = nullptr;
    const char* pool_tasks_msg = nullptr;      ///< the engine's own error line
    bool pool_tasks_strict = false;            ///< generate: the whole value an integer
    int64_t* prompt_cache_every = nullptr;
    int64_t* prompt_cache_root = nullptr;
    int64_t* reply_checkpoint_every = nullptr;
    int64_t* short_read = nullptr;
    bool* short_read_given = nullptr;
    int64_t* turn_token = nullptr;             ///< generate: as it is
    int64_t* tail_role_token = nullptr;
    int32_t* turn_token_i32 = nullptr;         ///< plain: max(-1, v)
    int32_t* tail_role_token_i32 = nullptr;
    int* vram_reserve_mib = nullptr;           ///< generate: int, no clamp
    bool* vram_reserve_given = nullptr;
    int64_t* vram_reserve_mib_i64 = nullptr;   ///< plain: max 0
    int* vram_reserve_later_mib = nullptr;
    int64_t* vram_reserve_later_mib_i64 = nullptr;
    double* pcie_frac = nullptr;
    bool pcie_frac_validate = false;           ///< plain: 0..1, else the error
    double* yarn_ext_factor = nullptr;
    double* yarn_attn_factor = nullptr;
    double* yarn_beta_fast = nullptr;
    double* yarn_beta_slow = nullptr;
    bool yarn_validate = false;                ///< plain: finite, ext >= -1, the rest > 0
    std::string* aux_cpus = nullptr;           ///< parse_spec checked, who in the error
    int* batch_groups = nullptr;
    bool* batch_groups_set = nullptr;          ///< generate
    bool* batch_groups_auto = nullptr;         ///< generate
    bool batch_groups_plain = false;           ///< plain: auto -> -1, else max 1
    // the conversation cache / save / memory group (parse_conversation_option): generate checks the value strictly
    // (from_chars, a per-option limit, its error); plain takes strtoll's value clamped to 0 and converts it itself
    bool conv_strict = false;
    int64_t* conversation_cache_mib = nullptr;
    int* conversation_cache_slots = nullptr;
    int64_t* conversation_cache_min_free_mib = nullptr;
    int64_t* conversation_cache_min_tokens = nullptr;
    int64_t* conversation_cache_park_gain = nullptr;
    int* conversation_save_checkpoints = nullptr;
    int64_t* conversation_save_mib = nullptr;
    int64_t* conversation_save_hours = nullptr;
    int64_t* session_min_free_mib = nullptr;
    int64_t* memory_limit_mib = nullptr;
    int64_t* ram_reserve_mib = nullptr;
};

/// 1: `a` was one of them (its value taken through `next(name)`), 0: it is not (or this engine does not take it),
/// -1: a value it refuses (said on stderr; the engine stops with 2).
template <class Next>
int parse_shared_option(const std::string& a, Next&& next, const SharedOptionFields& f) {
    auto i64 = [&](int64_t* p) { *p = std::strtoll(next(a.c_str()), nullptr, 10); return 1; };
    auto f64 = [&](double* p) { *p = std::strtod(next(a.c_str()), nullptr); return 1; };
    if (a == "--wide-window" && f.wide_window) {
        const char* v = next(a.c_str());
        strata::core::WideWindow w;
        if (!strata::core::parse_wide_window(v, w)) {
            std::fprintf(stderr, "%s: --wide-window takes auto, off or on, not %s\n", f.who, v);
            return -1;
        }
        strata::core::set_wide_window(w);
        return 1;
    }
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
    if (a == "--spec" && f.spec) {
        const std::string v = next(a.c_str());
        if (v == "auto" && f.spec_auto) {
            *f.spec_auto = true;
            *f.spec = f.spec_auto_default;
        } else {
            if (f.spec_auto) *f.spec_auto = false;
            *f.spec = (int) std::strtol(v.c_str(), nullptr, 10);
        }
        return 1;
    }
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
    if (a == "--window-hashes" && f.window_hashes) { *f.window_hashes = next(a.c_str()); return 1; }
    if (a == "--draft-costs" && f.draft_costs) { *f.draft_costs = next(a.c_str()); return 1; }
    if (a == "--draft-costs-key" && f.draft_costs_key) { *f.draft_costs_key = next(a.c_str()); return 1; }
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
    // the rest of the two engines' common options
    if (a == "--serve" && f.serve) { *f.serve = true; return 1; }
    if (a == "--vision" && f.vision) { *f.vision = true; return 1; }
    if (a == "--greedy" && f.greedy) { *f.greedy = true; return 1; }
    if (a == "--stop-eos" && f.stop_eos) { *f.stop_eos = true; return 1; }
    if (a == "--session-save-reclaim" && f.session_save_reclaim) { *f.session_save_reclaim = true; return 1; }
    if (a == "--prompt-cache-tail" && f.prompt_cache_tail) { *f.prompt_cache_tail = true; return 1; }
    if (a == "--prompt-cache-spread" && f.prompt_cache_spread) { *f.prompt_cache_spread = true; return 1; }
    if (a == "--conversation-save-compress" && f.conversation_save_compress) { *f.conversation_save_compress = true; return 1; }
    if (a == "--no-host-worker") {
        if (f.no_host_worker) { *f.no_host_worker = true; return 1; }
        if (f.pool_host_works) { *f.pool_host_works = false; return 1; }
    }
    if (a == "--batch-mtp" || a == "--no-batch-mtp") {
        const bool on = a == "--batch-mtp";
        if (f.batch_mtp_int) { *f.batch_mtp_int = on ? 1 : 0; return 1; }
        if (f.batch_mtp_bool) { *f.batch_mtp_bool = on; return 1; }
    }
    if (a == "--adapt-async") {
        const bool on = std::strtol(next(a.c_str()), nullptr, 10) != 0;
        if (f.adapt_async_int) { *f.adapt_async_int = on ? 1 : 0; return 1; }
        if (f.adapt_async_bool) { *f.adapt_async_bool = on; return 1; }
    }
    if (a == "--coupled-draft" && f.coupled_draft) { *f.coupled_draft = true; return 1; }
    if (a == "--no-coupled-draft" && f.coupled_draft) { *f.coupled_draft = false; return 1; }
    if (a == "--kv-grow" && f.kv_grow) { *f.kv_grow = 1; return 1; }
    if (a == "--no-kv-grow" && f.kv_grow) { *f.kv_grow = 0; return 1; }
    if (a == "--kv-resident" && f.kv_resident) {
        const std::string v = next(a.c_str());
        if (v == "auto") *f.kv_resident = -1;
        else if (v == "off") *f.kv_resident = 0;
        else {
            char* end = nullptr;
            const long long n = std::strtoll(v.c_str(), &end, 10);
            if (end == v.c_str() || *end != '\0' || n < 0) {
                std::fprintf(stderr, "%s: --kv-resident takes N, auto or off, not \"%s\"\n", f.who, v.c_str());
                return -1;
            }
            *f.kv_resident = n;
        }
        return 1;
    }
    if (a == "--conversation-save" && f.conversation_save) { *f.conversation_save = next(a.c_str()); return 1; }
    if (a == "--dump-logits" && f.dump_logits) { *f.dump_logits = next(a.c_str()); return 1; }
    if (a == "--dump-routing" && f.dump_routing) { *f.dump_routing = next(a.c_str()); return 1; }
    if (a == "--pool-cores" && f.pool_cores) { *f.pool_cores = next(a.c_str()); return 1; }
    if (a == "--shared-expert-arena" && f.shared_expert_arena) { *f.shared_expert_arena = next(a.c_str()); return 1; }
    if (a == "--expert-cache-remote-placement" && f.expert_cache_remote_placement) {
        *f.expert_cache_remote_placement = next(a.c_str()); return 1;
    }
    if (a == "--expert-cache" && f.expert_cache) {
        const std::string v = next(a.c_str());
        *f.expert_cache = (int) expert_cache_request(v);
        return 1;
    }
    if (a == "--gpu" && f.gpu_consume) { (void) next(a.c_str()); return 1; }   // applied at startup, before the runtime
    if (a == "--control-vector" && f.cvec_files) { f.cvec_files->push_back({next(a.c_str()), 1.0f}); return 1; }
    if (a == "--control-vector-scaled" && f.cvec_files) {   // FILE:SCALE, comma-separated; the last colon before the scale
        std::stringstream list(next(a.c_str()));
        std::string item;
        while (std::getline(list, item, ',')) {
            const size_t colon = item.rfind(':');
            char* end = nullptr;
            const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
            if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                return -1;
            }
            f.cvec_files->push_back({item.substr(0, colon), sc});
        }
        return 1;
    }
    if (a == "--control-vector-layer-range" && f.cvec_first && f.cvec_last) {
        *f.cvec_first = (int) std::strtol(next(a.c_str()), nullptr, 10);
        *f.cvec_last = (int) std::strtol(next(a.c_str()), nullptr, 10);
        return 1;
    }
    if (a == "--cvec-mode" && f.cvec_mode) {
        const std::string m = next(a.c_str());
        if (m != "project" && m != "add") {
            std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str());
            return -1;
        }
        *f.cvec_mode = m == "project" ? 0 : 1;
        return 1;
    }
    if (a == "--cvec-dir" && f.cvec_single) {
        const std::string d = next(a.c_str());
        if (d == "per-layer") *f.cvec_single = -1;
        else if (d.rfind("single:", 0) == 0) *f.cvec_single = (int) std::strtol(d.c_str() + 7, nullptr, 10);
        else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return -1; }
        return 1;
    }
    if (a == "--prompt-cache" && f.prompt_cache) { *f.prompt_cache = std::max(0, (int) std::strtol(next(a.c_str()), nullptr, 10)); return 1; }
    if (a == "--suffix-draft" && f.suffix_draft) { *f.suffix_draft = std::max(0, (int) std::strtol(next(a.c_str()), nullptr, 10)); return 1; }
    if (a == "--lookup-chain" && f.lookup_chain) {
        const int v = (int) std::strtol(next(a.c_str()), nullptr, 10);
        *f.lookup_chain = f.lookup_chain_hi < f.lookup_chain_lo ? std::max(f.lookup_chain_lo, v)
                                                               : std::clamp(v, f.lookup_chain_lo, f.lookup_chain_hi);
        return 1;
    }
    if (a == "--lookup-chain-min" && f.lookup_chain_min) {
        *f.lookup_chain_min = std::max(f.lookup_chain_min_lo, (int) std::strtol(next(a.c_str()), nullptr, 10));
        return 1;
    }
    if (a == "--pipeline-windows" && f.pipeline_windows) {
        const int v = std::max(0, (int) std::strtol(next(a.c_str()), nullptr, 10));
        *f.pipeline_windows = f.pipeline_windows_hi < 0 ? v : std::min(f.pipeline_windows_hi, v);
        return 1;
    }
    if (a == "--adapt-mib-per-token" && f.adapt_mib_per_token) {
        const int v = (int) std::strtol(next(a.c_str()), nullptr, 10);
        *f.adapt_mib_per_token = f.adapt_mib_clamp0 ? std::max(0, v) : v;
        return 1;
    }
    if ((a == "--batch" || a == "--slots") && f.batch) {
        const int v = (int) std::strtol(next(a.c_str()), nullptr, 10);
        *f.batch = f.batch_clamp0 ? std::max(0, v) : v;
        return 1;
    }
    if ((a == "--expert-cache-device1" || a == "--expert-cache-device2" || a == "--expert-cache-device3") &&
        f.expert_cache_remote) {
        const int v = (int) std::strtol(next(a.c_str()), nullptr, 10);
        f.expert_cache_remote[(size_t) (a.back() - '1')] = f.expert_cache_remote_clamp0 ? std::max(0, v) : v;
        return 1;
    }
    if (a == "--split-skip-if-fits" && f.split_skip_if_fits) { *f.split_skip_if_fits = true; return 1; }
    if (a == "--split-roles" && f.split_roles_helper) {
        *f.split_roles_helper = std::string(next(a.c_str())) == "helper";
        return 1;
    }
    if (a == "--pool-tasks" && f.pool_tasks) {
        const char* v = next(a.c_str());
        char* end = nullptr;
        const long tasks = std::strtol(v, &end, 10);
        if ((f.pool_tasks_strict && (end == v || *end != '\0')) || tasks < 0 || tasks > 4096) {
            std::fprintf(stderr, "%s\n", f.pool_tasks_msg != nullptr ? f.pool_tasks_msg : "--pool-tasks takes 0..4096");
            return -1;
        }
        *f.pool_tasks = (int) tasks;
        return 1;
    }
    if (a == "--prompt-cache-every" && f.prompt_cache_every) { *f.prompt_cache_every = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10)); return 1; }
    if (a == "--prompt-cache-root" && f.prompt_cache_root) { *f.prompt_cache_root = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10)); return 1; }
    if (a == "--reply-checkpoint-every" && f.reply_checkpoint_every) { *f.reply_checkpoint_every = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10)); return 1; }
    if (a == "--short-read" && f.short_read) {
        *f.short_read = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10));
        if (f.short_read_given) *f.short_read_given = true;
        return 1;
    }
    if (a == "--turn-token") {
        const long long v = std::strtoll(next(a.c_str()), nullptr, 10);
        if (f.turn_token) { *f.turn_token = v; return 1; }
        if (f.turn_token_i32) { *f.turn_token_i32 = (int32_t) std::max(-1LL, v); return 1; }
    }
    if (a == "--tail-role-token") {
        const long long v = std::strtoll(next(a.c_str()), nullptr, 10);
        if (f.tail_role_token) { *f.tail_role_token = v; return 1; }
        if (f.tail_role_token_i32) { *f.tail_role_token_i32 = (int32_t) std::max(-1LL, v); return 1; }
    }
    if (a == "--vram-reserve-mib") {
        const long long v = std::strtoll(next(a.c_str()), nullptr, 10);
        if (f.vram_reserve_mib) {
            *f.vram_reserve_mib = (int) v;
            if (f.vram_reserve_given) *f.vram_reserve_given = true;
            return 1;
        }
        if (f.vram_reserve_mib_i64) { *f.vram_reserve_mib_i64 = std::max<int64_t>(0, v); return 1; }
    }
    if (a == "--vram-reserve-later-mib") {
        const long long v = std::strtoll(next(a.c_str()), nullptr, 10);
        if (f.vram_reserve_later_mib) { *f.vram_reserve_later_mib = (int) v; return 1; }
        if (f.vram_reserve_later_mib_i64) { *f.vram_reserve_later_mib_i64 = std::max<int64_t>(0, v); return 1; }
    }
    if (a == "--pcie-frac" && f.pcie_frac) {
        const double v = std::strtod(next(a.c_str()), nullptr);
        if (f.pcie_frac_validate && (!std::isfinite(v) || v < 0.0 || v > 1.0)) {
            std::fprintf(stderr, "%s: --pcie-frac takes 0 .. 1\n", f.who);
            return -1;
        }
        *f.pcie_frac = v;
        return 1;
    }
    if ((a == "--yarn-ext-factor" || a == "--yarn-attn-factor" || a == "--yarn-beta-fast" || a == "--yarn-beta-slow") &&
        (f.yarn_ext_factor || f.yarn_attn_factor || f.yarn_beta_fast || f.yarn_beta_slow)) {
        const double v = std::strtod(next(a.c_str()), nullptr);
        // plain checks them: finite, the ext factor >= -1 (< 0: auto), the rest positive
        if (f.yarn_validate && (!std::isfinite(v) || (a == "--yarn-ext-factor" ? v < -1.0 : v <= 0.0))) {
            std::fprintf(stderr, "%s: %s %g: invalid rope scaling knob\n", f.who, a.c_str(), v);
            return -1;
        }
        double* dst = a == "--yarn-ext-factor"    ? f.yarn_ext_factor
                    : a == "--yarn-attn-factor" ? f.yarn_attn_factor
                    : a == "--yarn-beta-fast"   ? f.yarn_beta_fast
                                                : f.yarn_beta_slow;
        if (dst != nullptr) *dst = v;
        return 1;
    }
    if (a == "--aux-cpus" && f.aux_cpus) {
        *f.aux_cpus = next(a.c_str());
        std::string aerr;
        strata::aux_cpus::Config acfg;
        if (!strata::aux_cpus::parse_spec(*f.aux_cpus, acfg, aerr)) {
            std::fprintf(stderr, "%s: %s\n", f.who, aerr.c_str());
            return -1;
        }
        return 1;
    }
    if (a == "--batch-groups" && f.batch_groups) {
        const std::string v = next(a.c_str());
        if (f.batch_groups_set) *f.batch_groups_set = true;
        if (v == "auto") {
            if (f.batch_groups_auto) *f.batch_groups_auto = true;
            if (f.batch_groups_plain) *f.batch_groups = -1;
        } else {
            const int n = (int) std::strtol(v.c_str(), nullptr, 10);
            *f.batch_groups = f.batch_groups_plain ? std::max(1, n) : n;
        }
        return 1;
    }
    return 0;
}


/// The conversation cache / save / memory group (--conversation-cache-*, --conversation-save-*, --session-min-free-mib,
/// --memory-limit-mib, --ram-reserve-mib).  generate checks the value strictly (from_chars, a per-option limit, its
/// error line); plain takes strtoll's value clamped to 0 and converts it to its own units itself.  1: taken,
/// -1: refused (strict only), 0: not one of them.
template <class Next>
int parse_conversation_option(const std::string& a, Next&& next, const SharedOptionFields& f) {
    if (a != "--conversation-cache-mib" && a != "--conversation-cache-slots" &&
        a != "--conversation-cache-min-free-mib" && a != "--conversation-cache-min-tokens" &&
        a != "--conversation-save-checkpoints" && a != "--conversation-save-mib" &&
        a != "--conversation-save-hours" && a != "--session-min-free-mib" &&
        a != "--conversation-cache-park-gain" && a != "--memory-limit-mib" && a != "--ram-reserve-mib")
        return 0;
    int64_t parsed = 0;
    if (f.conv_strict) {
        const std::string value = next(a.c_str());
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        const int64_t limit = a == "--conversation-cache-slots" || a == "--conversation-save-checkpoints" ||
                                      a == "--conversation-cache-min-tokens" || a == "--conversation-cache-park-gain"
                                  ? INT32_MAX
                              : a == "--conversation-save-hours" ? INT64_MAX / 3600
                                                                 : INT64_MAX / (int64_t{1024} * 1024);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0 || parsed > limit) {
            std::fprintf(stderr, "%s needs a nonnegative integer within range\n", a.c_str());
            return -1;
        }
    } else {
        parsed = std::max<int64_t>(0, std::strtoll(next(a.c_str()), nullptr, 10));
    }
    if (a == "--conversation-cache-mib" && f.conversation_cache_mib) *f.conversation_cache_mib = parsed;
    else if (a == "--conversation-cache-min-free-mib" && f.conversation_cache_min_free_mib) *f.conversation_cache_min_free_mib = parsed;
    else if (a == "--conversation-cache-min-tokens" && f.conversation_cache_min_tokens) *f.conversation_cache_min_tokens = parsed;
    else if (a == "--conversation-cache-park-gain" && f.conversation_cache_park_gain) *f.conversation_cache_park_gain = parsed;
    else if (a == "--conversation-save-checkpoints" && f.conversation_save_checkpoints) *f.conversation_save_checkpoints = (int) parsed;
    else if (a == "--conversation-save-mib" && f.conversation_save_mib) *f.conversation_save_mib = parsed;
    else if (a == "--conversation-save-hours" && f.conversation_save_hours) *f.conversation_save_hours = parsed;
    else if (a == "--session-min-free-mib" && f.session_min_free_mib) *f.session_min_free_mib = parsed;
    else if (a == "--memory-limit-mib" && f.memory_limit_mib) *f.memory_limit_mib = parsed;
    else if (a == "--ram-reserve-mib" && f.ram_reserve_mib) *f.ram_reserve_mib = parsed;
    else if (a == "--conversation-cache-slots" && f.conversation_cache_slots) *f.conversation_cache_slots = (int) parsed;
    return 1;
}

}  // namespace strata::program
