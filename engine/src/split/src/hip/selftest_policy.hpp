// When the split engine checks a card's kernels before it loads the model, and the plain lines it prints (AMD only). CPU only, so
// tools/selftest_policy_test.cpp holds the rules without a card.
//
// Strata has run the HIP kernels on gfx1201 (Radeon AI PRO R9700) only. The same sources are built for the other RDNA cards (gfx1200, gfx1100..gfx1103), whose WMMA operand
// layouts differ from gfx12's (hip/mma.h), and nobody has run them. On a card that is not gfx1201 the engine therefore (1) says once that the split is experimental there and
// asks the person to report whether it works, and (2) runs hip/selftest.cu on every card before anything is loaded: the expert tile, the attention tile's WMMA shapes, the
// dense product, the dp4a and the recurrent step against a scalar reference on random data. Any mismatch or error: the engine refuses to start with one line (the server then
// starts the normal layer split and prints that line). FNX_SELFTEST=1 runs the self-test on gfx1201 too (to see it pass); FNX_SELFTEST_FAIL=1 forces a failure (tests).
#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace fnx {

constexpr const char* SPLIT_REPORT_URL = "https://github.com/Niko1221/Strata/issues/new?template=amd_split_report.yml";

// the architectures Strata has run the split on (gcnArchName starts with one of these)
inline bool arch_tested(const std::string& gcn_arch) { return gcn_arch.rfind("gfx1201", 0) == 0; }

struct SelftestDecision {
    bool run = false;                // the self-test was run (or forced to fail)
    bool ok = true;                  // the engine may go on
    std::vector<std::string> lines;  // what to print, in order: stderr, one plain line each
};

// run(what): the self-test on the card(s); returns "" when it passed (what: what it covered, for the pass line) or the failure in a few words
inline SelftestDecision selftest_decide(const std::string& gcn_arch, const char* env_selftest, const char* env_fail, const std::function<std::string(std::string&)>& run) {
    SelftestDecision d;
    const bool tested = arch_tested(gcn_arch);
    const bool forced = env_selftest && std::strcmp(env_selftest, "1") == 0;
    if (tested && !forced) return d;
    const std::string arch = gcn_arch.substr(0, gcn_arch.find(':'));
    if (!tested)
        d.lines.push_back("The split engine is experimental on this card (" + arch + "): Strata has not run it on this architecture. Please report whether it works: " +
                          std::string(SPLIT_REPORT_URL));
    d.run = true;
    std::string covered, bad;
    if (env_fail && std::strcmp(env_fail, "1") == 0) bad = "forced failure (FNX_SELFTEST_FAIL=1)";
    else bad = run(covered);
    if (!bad.empty()) {
        d.ok = false;
        d.lines.push_back("The split engine's self-test failed on " + arch + " (" + bad + "), so it does not run on this card: the normal layer split starts instead.");
        return d;
    }
    d.lines.push_back("The split engine's self-test passed on " + arch + " (" + covered + ").");
    return d;
}

}  // namespace fnx
