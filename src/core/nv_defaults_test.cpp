// src/core/nv_defaults_test.cpp - the NVIDIA per-architecture default tables (no GPU needed): the sm_121 (GB10, DGX Spark) table, its
// two off switches, and that no other card's table changed.
#include "strata/core/nv_chunk.hpp"
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

static const std::string* find(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return &kv.second;
    return nullptr;
}

int main() {
    using namespace strata::core;
    unset("STRATA_SM121_PERF");
    unset("STRATA_SM86_PERF");
    unset("STRATA_SM60_PERF");
    unset("STRATA_EMULATE_CC");
    // GB10 (12.1): the five measured switches, and nothing else
    {
        const auto t = nv_default_env(12, 1, false);
        CHECK(t.size() == 5);
        CHECK(find(t, "STRATA_PF_FUSED") && *find(t, "STRATA_PF_FUSED") == "1");
        CHECK(find(t, "STRATA_PREFILL_NVFP4") && *find(t, "STRATA_PREFILL_NVFP4") == "w4a4");
        CHECK(find(t, "STRATA_HC_UPMIX") && *find(t, "STRATA_HC_UPMIX") == "1");
        CHECK(find(t, "STRATA_SIDE_MIXER") && *find(t, "STRATA_SIDE_MIXER") == "2");
        CHECK(find(t, "STRATA_SPEC_COST") && *find(t, "STRATA_SPEC_COST") == "1");
    }
    // STRATA_DEFAULTS=026 (the 0.2.7 master switch; the sm_121 table is newer than 0.2.6) empties it
    CHECK(nv_default_env(12, 1, true).empty());
    // STRATA_SM121_PERF=0 empties it; any other value keeps it
    put("STRATA_SM121_PERF", "0");
    CHECK(nv_default_env(12, 1, false).empty());
    put("STRATA_SM121_PERF", "1");
    CHECK(nv_default_env(12, 1, false).size() == 5);
    unset("STRATA_SM121_PERF");
    // the other cards: sm_120 (RTX 50) and sm_100 get nothing from this file; the sm_86 / sm_60 tables are unchanged by the flag
    CHECK(nv_default_env(12, 0, false).empty());
    CHECK(nv_default_env(10, 0, false).empty());
    CHECK(nv_default_env(8, 9, false).empty());
    CHECK(nv_default_env(7, 5, false).empty());
    {
        const auto a = nv_default_env(8, 6, false), b = nv_default_env(8, 6, true);
        CHECK(!a.empty() && a == b);               // STRATA_DEFAULTS=026 does not touch the 2026-10 sm_86 table
        const auto p = nv_default_env(6, 0, false), q = nv_default_env(6, 0, true);
        CHECK(!p.empty() && p == q);
        CHECK(!find(a, "STRATA_PF_FUSED") && !find(p, "STRATA_PF_FUSED"));
    }
    // round 0212: STRATA_MMQ_KQ=1 is a default for compute capability 8.0+ (owner yes 11-03), with its own off switches
    unset("STRATA_NV_KQ_DEFAULT");
    CHECK(kNvKqDefaultOn);
    for (int major : {8, 9, 10, 12}) CHECK(nv_kq_default_env(major, 0, false).size() == 1);   // on without any switch
    put("STRATA_NV_KQ_DEFAULT", "1");
    for (int major : {8, 9, 10, 12}) {
        const auto k = nv_kq_default_env(major, 0, false);
        CHECK(k.size() == 1 && find(k, "STRATA_MMQ_KQ") && *find(k, "STRATA_MMQ_KQ") == "1");
    }
    CHECK(nv_kq_default_env(7, 5, false).empty());     // Turing / Volta / Pascal keep the FP16 prompt path
    CHECK(nv_kq_default_env(6, 0, false).empty());
    CHECK(nv_kq_default_env(12, 0, true).empty());     // STRATA_DEFAULTS=026
    put("STRATA_NV_KQ_DEFAULT", "0");
    CHECK(nv_kq_default_env(12, 0, false).empty());    // its own off switch
    unset("STRATA_NV_KQ_DEFAULT");
    // the other tables are unchanged by it
    CHECK(nv_default_env(12, 0, false).empty());
    // round 0212: the NVIDIA prompt-chunk ceiling of a bare `--prefill auto` (nv_chunk.hpp)
    CHECK(nv_chunk_ceiling(nullptr, false) == 32768);
    CHECK(nv_chunk_ceiling("1", false) == 32768);
    CHECK(nv_chunk_ceiling("16", false) == 16384);
    CHECK(nv_chunk_ceiling("0", false) == 0);          // STRATA_NV_CHUNK=0 keeps 8192
    CHECK(nv_chunk_ceiling(nullptr, true) == 0);       // STRATA_DEFAULTS=026 keeps 8192
    CHECK(nv_chunk_ceiling("16", true) == 0);
    CHECK(nv_chunk_card_ok(7, 0));                                                    // Volta (V100): measured, takes the raise
    CHECK(!nv_chunk_card_ok(7, 5));                                                   // Turing: not measured, keeps 8192
    CHECK(!nv_chunk_card_ok(6, 0) && !nv_chunk_card_ok(6, 1) && !nv_chunk_card_ok(6, 2));   // Pascal: not measured, keeps 8192
    CHECK(!nv_chunk_card_ok(5, 2) && !nv_chunk_card_ok(0, 0));
    CHECK(nv_chunk_card_ok(8, 0) && nv_chunk_card_ok(8, 6) && nv_chunk_card_ok(9, 0) && nv_chunk_card_ok(12, 0));
    CHECK(nv_chunk_step_down(32768) == 16384 && nv_chunk_step_down(16384) == 8192 && nv_chunk_step_down(8192) == 8192);
    // the one-argument form reads STRATA_DEFAULTS
    put("STRATA_DEFAULTS", "026");
    CHECK(nv_default_env(12, 1).empty());
    unset("STRATA_DEFAULTS");
    if (fails == 0) std::printf("nv_defaults_test: ok\n");
    return fails == 0 ? 0 : 1;
}
