// keeping the KFD quiet on an integrated gfx11 GPU under Linux (include/strata/core/kfd_quiet.hpp).  No GPU: the switch rule and the
// load order's rule.  (The device rule itself is igpu_kfd_settle_default, checked in src/core/arch_defaults_test.cpp.)
#include "strata/core/kfd_quiet.hpp"

#include <cstdio>
#include <string>
#include <utility>

using namespace strata::core::kfd_quiet;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

void switches() {
    CHECK(switch_on(nullptr, nullptr));
    CHECK(switch_on("1", nullptr));
    CHECK(switch_on("", ""));            // empty = unset
    CHECK(!switch_on("0", nullptr));     // the kill switch turns everything off
    CHECK(!switch_on("0", "1"));         // ... also when the fix's own switch says 1
    CHECK(!switch_on(nullptr, "0"));     // one fix off
    CHECK(!switch_on("1", "0"));
    CHECK(switch_on("yes", "on"));
    CHECK(std::string(fix_env(Fix::Load)) == "STRATA_KFD_QUIET_LOAD");
    CHECK(std::string(fix_env(Fix::Exit)) == "STRATA_KFD_QUIET_EXIT");
    CHECK(std::string(fix_env(Fix::Xfer)) == "STRATA_KFD_QUIET_XFER");
}

void load_order() {
    // not the class: none of the three moves, whatever the switches say (every other card keeps the old order)
    for (const char* all : {(const char*) nullptr, "1", "0"})
        for (const char* one : {(const char*) nullptr, "1", "0"}) {
            const LoadMoves m = load_moves(false, all, one);
            CHECK(!m.hc_check_early && !m.settle_after_cache && !m.first_prompt_buffers);
        }
    // the class, nothing set: all three
    {
        const LoadMoves m = load_moves(true, nullptr, nullptr);
        CHECK(m.hc_check_early && m.settle_after_cache && m.first_prompt_buffers);
    }
    // the class, STRATA_KFD_QUIET=0 or STRATA_KFD_QUIET_LOAD=0: the old order (the A/B arm of the same binary)
    for (const auto& p : {std::pair<const char*, const char*>{"0", nullptr}, {nullptr, "0"}, {"0", "0"}, {"0", "1"}}) {
        const LoadMoves m = load_moves(true, p.first, p.second);
        CHECK(!m.hc_check_early && !m.settle_after_cache && !m.first_prompt_buffers);
    }
    // the flag as the engine sets it: off until the card is named
    CHECK(!class_flag().load());
    CHECK(!on(Fix::Load) && !on(Fix::Exit));
}
}  // namespace

int main() {
    switches();
    load_order();
    if (failed) {
        std::fprintf(stderr, "kfd_quiet_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("kfd_quiet_test: ok\n");
    return 0;
}
