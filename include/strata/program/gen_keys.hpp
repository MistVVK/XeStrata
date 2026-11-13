// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/gen_keys.hpp - the GEN/GENI line's keys between max_new and the ids, which the two engines
// parsed a copy of each.  One reader; each engine points the fields at its own request state (a null
// field: that engine does not take the key, and it is skipped as unknown).  The differences the engines keep are
// arguments: pin and penalty_last_n clamped to 0 or not, pcie_frac/spec_min_p read through float (generate) or
// double (plain), aux_cpus applied at once (plain) or kept for the request (generate).  Unknown keys are skipped:
// the ids start at the first token without '='.
#pragma once

#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::program {

struct GenKeyFields {
    // the sampling
    float* temperature = nullptr;
    float* top_p = nullptr;
    int* top_k = nullptr;
    float* min_p = nullptr;
    uint64_t* seed = nullptr;
    float* penalty_repeat = nullptr;
    float* penalty_freq = nullptr;
    float* penalty_present = nullptr;
    int* penalty_last_n = nullptr;
    bool penalty_last_n_clamp0 = false;   ///< plain: max(0, v); generate takes v as it is
    int64_t* pen_from = nullptr;          ///< the request continues one whose reply began at token N (its penalty
                                          ///< window starts there; a solo -> batch slot or slot -> solo leg)
    // the request
    int* ckpt_int = nullptr;              ///< generate's int
    bool* ckpt_bool = nullptr;            ///< plain's bool
    bool* prefill = nullptr;
    int64_t* pin = nullptr;
    bool pin_clamp0 = false;              ///< generate: max(0, v); plain takes v as it is
    int* logprobs = nullptr;
    std::string* grammar = nullptr;
    std::string* logit_bias = nullptr;
    bool* logit_bias_present = nullptr;
    std::string* script_table = nullptr;
    std::string* script_allow = nullptr;
    std::string* script_pending = nullptr;
    int* cvec_int = nullptr;              ///< generate's int
    bool* cvec_bool = nullptr;            ///< plain's bool
    int* aux_cpus_deferred = nullptr;     ///< generate: 0|1 for the request (-1 = the start-up value)
    bool aux_cpus_immediate = false;      ///< plain: aux_cpus::set_enabled at once
    double* pcie_frac = nullptr;
    bool* pcie_set = nullptr;             ///< generate: the request set it
    bool pcie_float = false;              ///< generate: (double) strtof; plain: strtod
    double* spec_min_p = nullptr;
    bool* minp_rule = nullptr;            ///< generate: the window rule turns to min-p
    bool spec_float = false;              ///< generate: (double) strtof; plain: strtod
    bool* route_prior = nullptr;          ///< generate only
    std::string* draft = nullptr;         ///< draft=mtp|dflash2|auto|off: this request's drafter (--draft auto)
    int* spec_cap = nullptr;              ///< spec=N: the largest window this request checks (N >= 1)
    int* probe = nullptr;                 ///< probe=K: the start's measurement's run K (its own lines only)
    std::vector<int32_t>* allowed = nullptr;
    std::vector<uint64_t>* img_keys = nullptr;
};

/// Reads the keys from `endp` (the text after max_new), leaving it at the ids' start (the first token without '=').
/// With `geni`, the embeddings file path (the first word after the keys) goes to `emb_path` and endp moves past it.
inline void parse_gen_keys(char*& endp, const GenKeyFields& f, bool geni = false, std::string* emb_path = nullptr) {
    if (endp != nullptr) {
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
            const char* v = val.c_str();
            if (key == "temperature" && f.temperature) *f.temperature = std::strtof(v, nullptr);
            else if (key == "top_p" && f.top_p) *f.top_p = std::strtof(v, nullptr);
            else if (key == "top_k" && f.top_k) *f.top_k = (int) std::strtol(v, nullptr, 10);
            else if (key == "min_p" && f.min_p) *f.min_p = std::strtof(v, nullptr);
            else if (key == "seed" && f.seed) *f.seed = std::strtoull(v, nullptr, 10);
            else if (key == "penalty_repeat" && f.penalty_repeat) *f.penalty_repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq" && f.penalty_freq) *f.penalty_freq = std::strtof(v, nullptr);
            else if (key == "penalty_present" && f.penalty_present) *f.penalty_present = std::strtof(v, nullptr);
            else if (key == "pen_from" && f.pen_from) *f.pen_from = std::strtoll(v, nullptr, 10);
            else if (key == "penalty_last_n" && f.penalty_last_n)
                *f.penalty_last_n = f.penalty_last_n_clamp0 ? std::max(0, (int) std::strtol(v, nullptr, 10))
                                                            : (int) std::strtol(v, nullptr, 10);
            else if (key == "ckpt") {
                const bool on = std::strtol(v, nullptr, 10) != 0;
                if (f.ckpt_int) *f.ckpt_int = on ? 1 : 0;
                if (f.ckpt_bool) *f.ckpt_bool = on;
            }
            else if (key == "prefill" && f.prefill) *f.prefill = std::strtol(v, nullptr, 10) != 0;
            else if (key == "pin" && f.pin)
                *f.pin = f.pin_clamp0 ? std::max<int64_t>(0, std::strtoll(v, nullptr, 10))
                                      : std::strtoll(v, nullptr, 10);
            else if (key == "logprobs" && f.logprobs)
                *f.logprobs = (int) std::clamp(std::strtol(v, nullptr, 10), -1L, 20L);
            else if (key == "grammar" && f.grammar) *f.grammar = val;
            else if (key == "logit_bias" && f.logit_bias) { *f.logit_bias = val; if (f.logit_bias_present) *f.logit_bias_present = true; }
            else if (key == "script_table" && f.script_table) *f.script_table = val;
            else if (key == "script_allow" && f.script_allow) *f.script_allow = val;
            else if (key == "script_pending" && f.script_pending) *f.script_pending = val;
            else if (key == "cvec") {
                const long on = std::strtol(v, nullptr, 10);
                if (f.cvec_int) *f.cvec_int = (int) on;
                if (f.cvec_bool) *f.cvec_bool = on != 0;
            }
            else if (key == "aux_cpus") {
                const bool on = std::strtol(v, nullptr, 10) != 0;
                if (f.aux_cpus_deferred) *f.aux_cpus_deferred = on ? 1 : 0;
                if (f.aux_cpus_immediate) strata::aux_cpus::set_enabled(on);
            }
            else if (key == "pcie_frac" && f.pcie_frac) {
                *f.pcie_frac = std::clamp(f.pcie_float ? (double) std::strtof(v, nullptr) : std::strtod(v, nullptr),
                                          0.0, 1.0);
                if (f.pcie_set) *f.pcie_set = true;
            }
            else if (key == "spec_min_p" && f.spec_min_p) {
                *f.spec_min_p = std::clamp(f.spec_float ? (double) std::strtof(v, nullptr) : std::strtod(v, nullptr),
                                           0.0, 1.0);
                if (f.minp_rule) *f.minp_rule = true;
            }
            else if (key == "route_prior" && f.route_prior) *f.route_prior = std::strtol(v, nullptr, 10) != 0;
            else if (key == "draft" && f.draft) *f.draft = val;
            else if (key == "spec" && f.spec_cap) *f.spec_cap = std::max(1, (int) std::strtol(v, nullptr, 10));
            else if (key == "probe" && f.probe) *f.probe = (int) std::strtol(v, nullptr, 10);
            else if (key == "allowed" && f.allowed) {
                f.allowed->clear();
                for (const char* q = v; *q != '\0';) {
                    char* qe = nullptr;
                    const long id = std::strtol(q, &qe, 10);
                    if (qe == q) break;
                    f.allowed->push_back((int32_t) id);
                    q = *qe == ',' ? qe + 1 : qe;
                }
            }
            else if (key == "img_keys" && f.img_keys) {
                f.img_keys->clear();
                for (const char* q = v; *q != '\0';) {
                    char* qe = nullptr;
                    const unsigned long long id = std::strtoull(q, &qe, 16);
                    if (qe == q || (*qe != ',' && *qe != '\0')) { f.img_keys->clear(); break; }
                    f.img_keys->push_back((uint64_t) id);
                    q = *qe == ',' ? qe + 1 : qe;
                }
            }
            // unknown keys are skipped: the ids start at the first token without '='
        }
    }
    if (geni && emb_path != nullptr && endp != nullptr) {
        while (*endp == ' ') ++endp;
        if (char* gap = std::strchr(endp, ' '); gap != nullptr) {
            emb_path->assign(endp, (size_t) (gap - endp));
            endp = gap;
        }
    }
}

}  // namespace strata::program
