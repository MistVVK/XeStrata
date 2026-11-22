// src/core/adapt_fetch_default_test.cpp - STRATA_ADAPT_FETCH=2 as the default of a compute capability 12.x card with at most 16 GiB of memory
// (0.2.17, owner yes 11-19; BW-9 in phaseA-tests/bw/RESULTS-BW-F.md): which cards, the off switches, the user's own value, the start-up flag.
// No GPU needed.  Its own program (not nv_defaults_test) because apply_nv_defaults reads strata::qwen_defaults_026(), which caches
// STRATA_DEFAULTS at its first call; this test never sets it.
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static const char* val(const char* k) { return std::getenv(k); }
static bool has(const std::vector<std::pair<std::string, std::string>>& t, const char* k, const char* v) {
    for (const auto& kv : t)
        if (kv.first == k) return kv.second == v;
    return false;
}
static bool lacks(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return false;
    return true;
}
static bool in(const std::vector<std::string>& v, const char* k) {
    for (const auto& s : v)
        if (s == k) return true;
    return false;
}

int main() {
    using namespace strata::core;
    const uint64_t GiB = 1ull << 30;
    unset("STRATA_SM120_PERF"); unset("STRATA_SM121_PERF"); unset("STRATA_SM86_PERF"); unset("STRATA_SM70_PERF"); unset("STRATA_SM60_PERF");
    unset("STRATA_DEFAULTS"); unset("STRATA_EMULATE_CC"); unset("STRATA_ADAPT_FETCH");

    // the card rule: compute capability 12.x, 0 < memory <= 16 GiB
    CHECK(kAdaptFetchMaxVramBytes == 16 * GiB);
    CHECK(adapt_fetch_default_card(12, 0, 12 * GiB));                    // RTX 5070 12 GB (the owner PC)
    CHECK(adapt_fetch_default_card(12, 0, 16 * GiB - (1ull << 20)));     // RTX 5060 Ti / 5070 Ti / 5080 16 GB: about 15.9 GiB reported
    CHECK(adapt_fetch_default_card(12, 0, 16 * GiB));                    // exactly 16 GiB
    CHECK(adapt_fetch_default_card(12, 0, 8 * GiB));                     // RTX 5060 8 GB, laptop parts
    CHECK(adapt_fetch_default_card(12, 0, 5 * GiB));
    CHECK(adapt_fetch_default_card(12, 1, 16 * GiB));                    // 12.x, the rule does not look at the minor
    CHECK(!adapt_fetch_default_card(12, 0, 16 * GiB + 1));               // over 16 GiB
    CHECK(!adapt_fetch_default_card(12, 0, 24 * GiB));                   // RTX PRO 4000 24 GB
    CHECK(!adapt_fetch_default_card(12, 0, 32 * GiB));                   // RTX 5090
    CHECK(!adapt_fetch_default_card(12, 0, 72 * GiB));                   // RTX PRO 5000 72 GB
    CHECK(!adapt_fetch_default_card(12, 1, 121 * GiB));                  // GB10, unified memory
    CHECK(!adapt_fetch_default_card(12, 0, 0));                          // memory unknown: no
    CHECK(adapt_fetch_default_card(8, 6, 8 * GiB));                      // RTX 3070 8 GB (Track 6A)
    CHECK(adapt_fetch_default_card(8, 6, 12 * GiB));                     // RTX 3060 12 GB (Track 6A)
    CHECK(!adapt_fetch_default_card(8, 6, 16 * GiB));                    // RTX A4000 16 GB (> 13 GiB)
    CHECK(adapt_fetch_default_card(8, 9, 8 * GiB));                      // RTX 4060 8 GB (Track 6A)
    CHECK(adapt_fetch_default_card(8, 9, 12 * GiB));                     // RTX 4070 / 4070 Ti 12 GB (Track 6A)
    CHECK(!adapt_fetch_default_card(8, 9, 16 * GiB));                    // RTX 4080 16 GB (> 13 GiB)
    CHECK(!adapt_fetch_default_card(9, 0, 16 * GiB));
    CHECK(!adapt_fetch_default_card(10, 0, 16 * GiB));
    CHECK(!adapt_fetch_default_card(6, 0, 16 * GiB));                    // P100 16 GB

    auto unset_sm120 = [] {
        for (const char* k : {
                 "STRATA_ADAPT_FETCH", "STRATA_VERIFY_DEVICE_HITS", "STRATA_L2_PF", "STRATA_HC_UPMIX",
                 "STRATA_Q80_ROWS1", "STRATA_EXP_MODE", "STRATA_SIDE_MIXER", "STRATA_DEQUANT_SIDE",
                 "STRATA_QFUSE", "STRATA_LFUSE", "STRATA_PLE_COMPACT", "STRATA_PLE_COPY_STREAM",
                 "STRATA_PF_FUSED_KQ", "STRATA_HC_Q8", "STRATA_PREFILL_CPU_SHARE_NT", "STRATA_PREFILL_NVFP4"}) {
            unset(k);
        }
    };

    // the table: the sm_120 entries (12 base, #6431: L2_PF opt-in), + ADAPT_FETCH for <= 16 GiB or + VERIFY_DEVICE_HITS for > 16 GiB
    {
        const auto t = nv_default_env(12, 0, false, 12 * GiB);
        CHECK(t.size() == 13 && has(t, "STRATA_ADAPT_FETCH", "2") && has(t, "STRATA_HC_UPMIX", "1") && has(t, "STRATA_EXP_MODE", "8"));
        CHECK(nv_default_env(12, 0, false, 16 * GiB).size() == 13);
        CHECK(nv_default_env(12, 0, false, 32 * GiB).size() == 13 &&
              has(nv_default_env(12, 0, false, 32 * GiB), "STRATA_VERIFY_DEVICE_HITS", "1") &&
              lacks(nv_default_env(12, 0, false, 32 * GiB), "STRATA_ADAPT_FETCH"));
        CHECK(nv_default_env(12, 0, false, 0).size() == 12);              // no memory given: the base sm_120 table
        CHECK(nv_default_env(12, 0, false).size() == 12);
        CHECK(nv_default_env(12, 0, true, 12 * GiB).empty());           // STRATA_DEFAULTS=026
        put("STRATA_SM120_PERF", "0");
        CHECK(nv_default_env(12, 0, false, 12 * GiB).empty());          // STRATA_SM120_PERF=0 leaves the whole table out, the switch with it
        unset("STRATA_SM120_PERF");
        CHECK(nv_default_env(12, 1, false, 12 * GiB).size() == 12);     // a 12.1 part: 11 base + ADAPT_FETCH when <= 16 GiB
        CHECK(nv_default_env(12, 1, false, 121 * GiB).size() == 12 &&
              has(nv_default_env(12, 1, false, 121 * GiB), "STRATA_VERIFY_DEVICE_HITS", "1") &&
              lacks(nv_default_env(12, 1, false, 121 * GiB), "STRATA_ADAPT_FETCH"));
    }
    // sm_86 & sm_89 <= 13 GiB get ADAPT_FETCH=2; > 13 GiB and older architectures do not
    CHECK(has(nv_default_env(8, 6, false, 12 * GiB), "STRATA_ADAPT_FETCH", "2"));
    CHECK(has(nv_default_env(8, 9, false, 12 * GiB), "STRATA_ADAPT_FETCH", "2"));
    CHECK(lacks(nv_default_env(8, 6, false, 16 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(lacks(nv_default_env(8, 9, false, 16 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(lacks(nv_default_env(7, 0, false, 16 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(lacks(nv_default_env(6, 0, false, 16 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(nv_default_env(10, 0, false, 16 * GiB).empty());

    // apply_nv_defaults: sets the environment where the user has not
    {
        unset_sm120();
        const auto set = apply_nv_defaults(12, 0, 12 * GiB);
        CHECK(in(set, "STRATA_ADAPT_FETCH") && set.size() == 13);
        const char* v = val("STRATA_ADAPT_FETCH");
        CHECK(v != nullptr && std::string(v) == "2");
        CHECK(adapt_fetch_is_default());
    }
    {   // a card over 16 GiB: the 12 base entries + VERIFY_DEVICE_HITS, ADAPT_FETCH stays unset, the flag is false
        unset_sm120();
        const auto set = apply_nv_defaults(12, 0, 32 * GiB);
        CHECK(!in(set, "STRATA_ADAPT_FETCH") && in(set, "STRATA_VERIFY_DEVICE_HITS") && set.size() == 13);
        CHECK(val("STRATA_ADAPT_FETCH") == nullptr);
        CHECK(!adapt_fetch_is_default());
    }
    {   // the user's own value wins, 0 included (the way back), and then the flag is false: the value is not the default's
        for (const char* mine : {"0", "1", "2"}) {
            unset_sm120();
            put("STRATA_ADAPT_FETCH", mine);
            const auto set = apply_nv_defaults(12, 0, 12 * GiB);
            CHECK(!in(set, "STRATA_ADAPT_FETCH") && set.size() == 12);
            const char* v = val("STRATA_ADAPT_FETCH");
            CHECK(v != nullptr && std::string(v) == mine);
            CHECK(!adapt_fetch_is_default());
        }
        unset_sm120();
    }
    {   // STRATA_SM120_PERF=0: nothing of the table is set, so no switch either
        unset_sm120();
        put("STRATA_SM120_PERF", "0");
        const auto set = apply_nv_defaults(12, 0, 12 * GiB);
        CHECK(set.empty() && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
        unset("STRATA_SM120_PERF");
    }
    {   // sm_86 with 12 GiB gets ADAPT_FETCH=2 (Track 6A); 16 GiB sm_86 does not
        unset_sm120();
        const auto set12 = apply_nv_defaults(8, 6, 12 * GiB);
        CHECK(in(set12, "STRATA_ADAPT_FETCH") && val("STRATA_ADAPT_FETCH") != nullptr &&
              std::string(val("STRATA_ADAPT_FETCH")) == "2" && adapt_fetch_is_default());
        unset_sm120();
        const auto set16 = apply_nv_defaults(8, 6, 16 * GiB);
        CHECK(!in(set16, "STRATA_ADAPT_FETCH") && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
    }
    {   // no memory given (the two-argument call of the other callers): as before (12 base entries)
        unset_sm120();
        const auto set = apply_nv_defaults(12, 0);
        CHECK(set.size() == 12 && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
    }
    unset_sm120();
    if (fails == 0) std::printf("adapt_fetch_default_test: ok\n");
    return fails == 0 ? 0 : 1;
}
