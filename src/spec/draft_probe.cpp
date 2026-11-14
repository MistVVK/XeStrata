// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/draft_probe.cpp - see include/strata/spec/draft_probe.hpp.
#include "strata/spec/draft_probe.hpp"

#include "strata/platform/env.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace strata::spec {

const char* drafter_name(Drafter d) {
    switch (d) {
        case Drafter::None: return "none";
        case Drafter::Mtp: return "mtp";
        case Drafter::Df2: return "dflash2";
    }
    return "?";
}

std::vector<ProbeArm> DraftProbe::arms(bool mtp, bool df2, bool none_ok, int spec, bool spec_auto, int df2_spec,
                                       bool static_available, int wide_mode, bool static_forced) {
    std::vector<ProbeArm> a;
    if (mtp) {
        a.push_back({Drafter::Mtp, spec});
        // the smallest window with a draft: on a machine where a verify row costs much it can beat the larger one
        if (spec_auto && spec > 2) a.push_back({Drafter::Mtp, 2});
    }
    if (df2) {
        const int first = wide_mode == 1 && df2_spec >= kernels::kWindowMaxT ? 1 : 0;
        const int last = wide_mode == 2 && df2_spec >= kernels::kWindowMaxT ? 2 : first;
        for (int block = first; block <= last; ++block) {
            const int cap = block == 0 ? std::min(df2_spec, kernels::kVerifyNarrowT) : df2_spec;
            if (!static_forced) a.push_back({Drafter::Df2, cap, false, block});
            if (static_available || static_forced) a.push_back({Drafter::Df2, cap, true, block});
        }
    }
    if (none_ok && (mtp || df2)) a.push_back({Drafter::None, 1});
    if (a.size() < 2) a.clear();
    return a;
}

std::vector<int> DraftProbe::plan(int n_arms, int rounds) {
    std::vector<int> p;
    if (n_arms < 2) return p;
    p.push_back(-1);   // the warm-up: the first arm, not counted (the first requests pay the kernels' first launches)
    // a b c, then c b a: a drift over the measurement (the GPU's clock coming up) falls on every arm alike
    for (int r = 0; r < rounds; ++r)
        for (int i = 0; i < n_arms; ++i) p.push_back(r % 2 == 0 ? i : n_arms - 1 - i);
    return p;
}

ProbeVerdict DraftProbe::decide(const std::vector<ProbeArm>& arms, const std::vector<ProbeRun>& runs,
                                int spec_default) const {
    ProbeVerdict v;
    const size_t n = arms.size();
    std::vector<int64_t> tok(n, 0), drafted(n, 0), accepted(n, 0);
    std::vector<double> ms(n, 0.0), lo(n, 1e300), hi(n, 0.0);
    for (const ProbeRun& r : runs) {
        if (r.arm < 0 || (size_t) r.arm >= n || r.ms <= 0.0 || r.tokens <= 0) continue;
        const size_t i = (size_t) r.arm;
        tok[i] += r.tokens;
        ms[i] += r.ms;
        drafted[i] += r.drafted;
        accepted[i] += r.accepted;
        const double t = 1000.0 * (double) r.tokens / r.ms;
        lo[i] = std::min(lo[i], t);
        hi[i] = std::max(hi[i], t);
    }
    v.tps.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) v.tps[i] = ms[i] > 0.0 ? 1000.0 * (double) tok[i] / ms[i] : 0.0;
    v.measured = n >= 2 && std::all_of(v.tps.begin(), v.tps.end(), [](double t) { return t > 0.0; });
    char buf[160];
    v.line = "drafter measured at the start:";
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof buf, " %s%s%.1f tok/s", drafter_name(arms[i].drafter),
                      arms[i].drafter == Drafter::None ? " " : (" window " + std::to_string(arms[i].spec) + " ").c_str(),
                      v.tps[i]);
        v.line += buf;
        if (arms[i].drafter == Drafter::Df2 && arms[i].block == 2) v.line += " adaptive block";
        if (arms[i].static_table) v.line += " + static table";
        if (drafted[i] > 0) {
            std::snprintf(buf, sizeof buf, " (drafts kept %.2f)", (double) accepted[i] / (double) drafted[i]);
            v.line += buf;
        }
        v.line += i + 1 < n ? "," : "";
    }
    if (!v.measured) {
        v.line += " - not every arm ran: the drafters stay as loaded";
        return v;
    }
    // The default form (the first arm: the MTP at its configured window) stays unless another beats it clearly, and
    // in every run: each of the other's runs faster than each of the default's, and its speed `switch_` (10%) and its
    // own and the default's spread between runs above.  A single shallow measurement must not leave a start at a
    // worse form for good (RTX 4070, Qwen3.8: one start in 8 kept "none" at 50.8 against the MTP's 49.2 tok/s, and
    // that start decoded 17% slower).  Of the arms that beat it so, the fastest.
    const size_t def = 0;
    auto spread = [&](size_t i) { return hi[i] > 0.0 && lo[i] < 1e299 ? (hi[i] - lo[i]) / hi[i] : 0.0; };
    auto beats_default = [&](size_t i) {
        const double bar = std::max({switch_, margin_, spread(i), spread(def)});
        return lo[i] > hi[def] && v.tps[i] >= v.tps[def] * (1.0 + bar);
    };
    size_t pick = def;
    for (size_t i = 1; i < n; ++i)
        if (beats_default(i) && (pick == def || v.tps[i] > v.tps[pick])) pick = i;
    v.keep = arms[pick].drafter;
    v.spec = arms[pick].drafter == Drafter::None ? 1 : arms[pick].spec;
    (void) spec_default;
    // the kept arm's form (Y6): DFlash2's block mode and static table
    v.df2_static = v.keep == Drafter::Df2 && arms[pick].static_table;
    v.df2_block = v.keep == Drafter::Df2 ? arms[pick].block : 0;
    v.line += std::string(" - keeps ") + drafter_name(v.keep) +
              (v.keep == Drafter::None ? std::string() : " window " + std::to_string(v.spec));
    if (v.keep == Drafter::Df2) {
        v.line += std::string(" (DFlash2 block ") + (v.df2_block == 0 ? "8" : v.df2_block == 1 ? "16" : "adaptive") +
                  (v.df2_static ? ", static table on)" : ", static table off)");
    }
    return v;
}

bool DraftProbe::settled(const std::vector<ProbeArm>& arms, const std::vector<ProbeRun>& runs) {
    // every other arm has a run no faster than one of the default's: none can beat it in every run any more
    const size_t n = arms.size();
    if (n < 2) return true;
    std::vector<double> lo(n, 1e300), hi(n, 0.0);
    for (const ProbeRun& r : runs) {
        if (r.arm < 0 || (size_t) r.arm >= n || r.ms <= 0.0 || r.tokens <= 0) continue;
        const double t = 1000.0 * (double) r.tokens / r.ms;
        lo[(size_t) r.arm] = std::min(lo[(size_t) r.arm], t);
        hi[(size_t) r.arm] = std::max(hi[(size_t) r.arm], t);
    }
    if (hi[0] <= 0.0) return false;
    for (size_t i = 1; i < n; ++i)
        if (hi[i] <= 0.0 || lo[i] > hi[0]) return false;
    return true;
}

bool probe_enabled() {
    const char* v = strata::env_get("DRAFT_PROBE");
    return !(v != nullptr && (v[0] == '0' || std::strcmp(v, "off") == 0 || std::strcmp(v, "no") == 0));
}

int probe_tokens() {
    const char* v = strata::env_get("DRAFT_PROBE_TOKENS");
    long n = 48;   // a value that is not a whole number keeps the default
    if (v != nullptr && v[0] != '\0') {
        char* end = nullptr;
        errno = 0;
        const long r = std::strtol(v, &end, 10);
        if (*end == '\0' && errno == 0 && r <= INT_MAX) n = r;
    }
    return std::max(16, (int) std::max(n, 0L));
}

int probe_run_tokens(double warm_tps) {
    const int most = probe_tokens();
    if (!(warm_tps > 0.0)) return most;
    return std::clamp((int) std::lround(warm_tps * 1.0), 16, std::max(16, most));
}

double probe_budget_s() {
    const char* v = strata::env_get("DRAFT_PROBE_S");
    double s = 20.0;   // a value that is not a number keeps the default
    if (v != nullptr && v[0] != '\0') {
        char* end = nullptr;
        errno = 0;
        const double r = std::strtod(v, &end);
        if (*end == '\0' && errno == 0) s = r;
    }
    return s > 0.0 ? s : 20.0;
}

std::string probe_line_tokens(const std::string& line, int n) {
    if (line.rfind("GEN ", 0) != 0) return line;
    const size_t sp = line.find(' ', 4);
    if (sp == std::string::npos) return line;
    return "GEN " + std::to_string(n) + line.substr(sp);
}

std::vector<int32_t> probe_prompt(const std::vector<std::string>& vocab) {
    // an explanation the model continues in kind: long enough that the drafters see a context, plain words
    static const char* const kText =
        "The following is a short guide to how a computer program reads a file from the disk and what happens at each "
        "step. First the program asks the operating system to open the file by its name and the system checks that the "
        "file exists and that the program may read it. Then the program reads the data in small blocks into memory and "
        "works on each block in turn until it reaches the end of the file. At the end it closes the file so that other "
        "programs can use it. In this guide we will look at each of these steps in more detail and explain why they "
        "matter. The first step is";
    std::unordered_map<std::string, int32_t> id;
    id.reserve(vocab.size());
    for (size_t i = 0; i < vocab.size(); ++i) id.emplace(vocab[i], (int32_t) i);
    std::vector<int32_t> out;
    const std::string gpt_space = "\xC4\xA0", sp_space = "\xE2\x96\x81";   // U+0120, U+2581
    const char* p = kText;
    bool first = true;
    while (*p != '\0') {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p != '\0' && *p != ' ' && *p != '.') ++p;
        std::string w(s, (size_t) (p - s));
        if (!w.empty()) {
            const std::string forms[3] = {(first ? "" : gpt_space) + w, sp_space + w, w};
            for (const std::string& f : forms) {
                auto it = id.find(f);
                if (it != id.end()) { out.push_back(it->second); break; }
            }
            first = false;
        }
        if (*p == '.') {
            auto it = id.find(".");
            if (it != id.end()) out.push_back(it->second);
            ++p;
        }
    }
    if (out.size() < 16) out.clear();
    return out;
}

}  // namespace strata::spec
