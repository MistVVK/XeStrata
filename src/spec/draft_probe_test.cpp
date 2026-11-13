// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/draft_probe_test.cpp - DraftProbe: which drafter an engine keeps after its start's measurement.
//
//   1. the arms of a start (one drafter alone: none to measure);
//   2. a clear win keeps that drafter alone;
//   3. within the margin the cheaper form stays (the MTP over DFlash2 over none);
//   4. the MTP and DFlash2 within the margin are both kept where the engine can hold both;
//   5. no drafter wins where a window costs more than its drafts bring;
//   6. a noisy arm's spread raises the bar to beat it;
//   7. the prompt from a vocabulary (the byte-level BPE's and SentencePiece's word marks).
#include "strata/spec/draft_probe.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::spec::DraftProbe;
using strata::spec::Drafter;
using strata::spec::ProbeArm;
using strata::spec::ProbeRun;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
// two runs an arm at the given speeds (64 tokens each)
std::vector<ProbeRun> runs(const std::vector<std::pair<double, double>>& tps) {
    std::vector<ProbeRun> r;
    for (size_t i = 0; i < tps.size(); ++i) {
        r.push_back({(int) i, 64, 64000.0 / tps[i].first, 0, 0});
        r.push_back({(int) i, 64, 64000.0 / tps[i].second, 0, 0});
    }
    return r;
}
}  // namespace

int main() {
    std::printf("draft_probe_test\n");
    const DraftProbe probe(0.03);

    check(DraftProbe::arms(true, false, false, 4, false, 6).empty(), "the MTP alone: nothing to measure");
    const auto a3 = DraftProbe::arms(true, true, true, 4, false, 6);
    check(a3.size() == 3 && a3[0].drafter == Drafter::Mtp && a3[1].drafter == Drafter::Df2 &&
              a3[2].drafter == Drafter::None && a3[2].spec == 1,
          "MTP, DFlash2 and none");
    const auto a4 = DraftProbe::arms(true, false, true, 4, true, 6);
    check(a4.size() == 3 && a4[1].drafter == Drafter::Mtp && a4[1].spec == 2, "--spec auto adds the MTP's window 2");
    check(DraftProbe::plan(3).size() == 7 && DraftProbe::plan(3)[0] == -1 && DraftProbe::plan(3)[3] == 2 &&
              DraftProbe::plan(3)[4] == 2,
          "a warm-up, then a b c c b a");

    auto v = probe.decide(a3, runs({{100, 100}, {75, 75}, {60, 60}}), true, 4);
    check(v.measured && v.keep == Drafter::Mtp && !v.keep_both && v.spec == 4, "the MTP 25% ahead: the MTP alone");
    v = probe.decide(a3, runs({{80, 80}, {100, 100}, {60, 60}}), true, 4);
    check(v.keep == Drafter::Df2 && !v.keep_both, "DFlash2 25% ahead: DFlash2 alone");
    v = probe.decide(a3, runs({{100, 100}, {101.5, 101.5}, {60, 60}}), false, 4);
    check(v.keep == Drafter::Mtp && !v.keep_both, "within the margin, both not possible: the MTP");
    v = probe.decide(a3, runs({{100, 100}, {101.5, 101.5}, {60, 60}}), true, 4);
    check(v.keep == Drafter::Mtp && v.keep_both, "within the margin, both possible: both stay");
    v = probe.decide(a3, runs({{50, 50}, {45, 45}, {70, 70}}), true, 4);
    check(v.keep == Drafter::None && v.spec == 1, "no drafter 40% ahead: none");
    v = probe.decide(a3, runs({{70, 70}, {45, 45}, {71, 71}}), true, 4);
    check(v.keep == Drafter::Mtp, "none within the margin of the MTP: the MTP");
    v = probe.decide(a4, runs({{100, 100}, {110, 110}, {60, 60}}), true, 4);
    check(v.keep == Drafter::Mtp && v.spec == 2, "the window 2 10% ahead: the MTP with window 2");
    v = probe.decide(a3, runs({{100, 80}, {95, 95}, {60, 60}}), true, 4);
    check(v.keep == Drafter::Mtp && v.keep_both, "a noisy MTP: DFlash2 does not clear its spread");
    std::vector<ProbeRun> part = runs({{100, 100}, {75, 75}});
    v = probe.decide(a3, part, true, 4);
    check(!v.measured, "an arm that did not run: nothing changes");

    std::vector<std::string> bpe(300), spm(300);
    for (size_t i = 0; i < bpe.size(); ++i) bpe[i] = spm[i] = "<unused" + std::to_string(i) + ">";
    const char* words[] = {"The", "following", "is", "a", "short", "guide", "to", "how", "computer", "program", "reads",
                           "file", "from", "the", "disk", "and", "what", "happens", "at", "each", "step", "."};
    size_t k = 10;
    for (const char* w : words) {
        const std::string s = w;
        bpe[k] = s == "The" || s == "." ? s : "\xC4\xA0" + s;
        spm[k] = s == "." ? s : "\xE2\x96\x81" + s;
        ++k;
    }
    const auto pb = strata::spec::probe_prompt(bpe), ps = strata::spec::probe_prompt(spm);
    check(pb.size() >= 16 && pb[0] == 10 && pb[1] == 11, "the byte-level BPE's words");
    check(ps.size() >= 16 && ps[0] == 10, "SentencePiece's words");
    check(strata::spec::probe_prompt(std::vector<std::string>(50, "x")).empty(), "a vocabulary without the words: none");

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
