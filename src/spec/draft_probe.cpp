// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/draft_probe.cpp - see include/strata/spec/draft_probe.hpp.
#include "strata/spec/draft_probe.hpp"

#include "strata/platform/env.hpp"

#include <algorithm>
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

std::vector<ProbeArm> DraftProbe::arms(bool mtp, bool df2, bool none_ok, int spec, bool spec_auto, int df2_spec) {
    std::vector<ProbeArm> a;
    if (mtp) {
        a.push_back({Drafter::Mtp, spec});
        // the smallest window with a draft: on a machine where a verify row costs much it can beat the larger one
        if (spec_auto && spec > 2) a.push_back({Drafter::Mtp, 2});
    }
    if (df2) a.push_back({Drafter::Df2, df2_spec});
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

ProbeVerdict DraftProbe::decide(const std::vector<ProbeArm>& arms, const std::vector<ProbeRun>& runs, bool both_ok,
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
    // a clear win: `margin` over the arm, and over that arm's own spread between its runs
    auto beats = [&](size_t a, size_t b) {
        const double spread = hi[b] > 0.0 && lo[b] < 1e299 ? (hi[b] - lo[b]) / hi[b] : 0.0;
        return v.tps[a] >= v.tps[b] * (1.0 + std::max(margin_, spread));
    };
    // the cheaper form first among ties: the MTP, then DFlash2, then none; the configured window before another
    auto rank = [&](size_t i) {
        const ProbeArm& a = arms[i];
        const int d = a.drafter == Drafter::Mtp ? 0 : a.drafter == Drafter::Df2 ? 1 : 2;
        return d * 2 + (a.spec == spec_default || a.drafter != Drafter::Mtp ? 0 : 1);
    };
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rank(a) < rank(b); });
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (v.tps[i] > v.tps[best]) best = i;
    // among the arms the fastest does not clearly beat (itself included), the cheapest form
    size_t pick = best;
    for (size_t i : order)
        if (i == best || !beats(best, i)) { pick = i; break; }
    v.keep = arms[pick].drafter;
    v.spec = arms[pick].drafter == Drafter::None ? 1 : arms[pick].spec;
    // the MTP and DFlash2 both within the margin of the fastest: both stay and DrafterPolicy picks per request
    if (both_ok && (v.keep == Drafter::Mtp || v.keep == Drafter::Df2)) {
        const Drafter other = v.keep == Drafter::Mtp ? Drafter::Df2 : Drafter::Mtp;
        for (size_t i = 0; i < n; ++i)
            if (arms[i].drafter == other && (i == best || !beats(best, i))) v.keep_both = true;
    }
    v.line += v.keep_both ? " - both drafters stay (within the margin), picked per request"
                          : std::string(" - keeps ") + drafter_name(v.keep) +
                                (v.keep == Drafter::None ? std::string() : " window " + std::to_string(v.spec));
    return v;
}

bool probe_enabled() {
    const char* v = strata::env_get("DRAFT_PROBE");
    return !(v != nullptr && (v[0] == '0' || std::strcmp(v, "off") == 0 || std::strcmp(v, "no") == 0));
}

int probe_tokens() {
    const char* v = strata::env_get("DRAFT_PROBE_TOKENS");
    const int n = v != nullptr && v[0] != '\0' ? std::atoi(v) : 64;
    return std::max(16, n);
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
