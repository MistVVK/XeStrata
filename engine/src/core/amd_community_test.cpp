// src/core/amd_community_test.cpp - STRATA_AMD_COMMUNITY_TUNING turns each community-made AMD change off (no GPU needed).
//
// Built twice by CMake: once as the default build (EXPECT_DOT_BUILT=1) and once with STRATA_AMD_COMMUNITY_DOT_OFF defined
// (EXPECT_DOT_BUILT=0), which is what cmake -DSTRATA_AMD_COMMUNITY_TUNING=OFF does.  Run from the source directory.
#include "strata/core/amd_community.hpp"
#include "../prefill/rocblas_tuning.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

int main() {
    namespace ac = strata::amd_community;
    const char* kSwitch = "STRATA_AMD_COMMUNITY_TUNING";

    // the switch: on by default, off for 0 / off / false / no, on for anything else
    unset(kSwitch);
    CHECK(ac::tuning_on());
    put(kSwitch, "1");
    CHECK(ac::tuning_on());
    for (const char* off : {"0", "off", "false", "no"}) {
        put(kSwitch, off);
        CHECK(!ac::tuning_on());
    }

    // which tables are the community's: exactly the two named, only for gfx1030 (feature suffix allowed)
    CHECK(ac::is_community_rocblas_table("gfx1030", "5.4.0.ad84daa81c"));
    CHECK(ac::is_community_rocblas_table("gfx1030:xnack-", "5.2.0.dabb6df2b98-dirty"));
    CHECK(!ac::is_community_rocblas_table("gfx1030", "5.6.0.8d1ae90e"));      // shipped before, not under the switch
    CHECK(!ac::is_community_rocblas_table("gfx1031", "5.4.0.ad84daa81c"));    // another card
    CHECK(!ac::is_community_rocblas_table("gfx1030", "5.2.0.dabb6df2b9"));    // AMD's Ubuntu build: not this table
    CHECK(!ac::is_community_rocblas_table("gfx1201", "5.4.0.ad84daa81c"));

    // rocBLAS tables: the switch off refuses every community table and nothing else
    unset(kSwitch);
    for (const auto& t : ac::kRocblasTables) CHECK(ac::rocblas_table_allowed(t.arch, t.version));
    put(kSwitch, "0");
    for (const auto& t : ac::kRocblasTables) CHECK(!ac::rocblas_table_allowed(t.arch, t.version));
    CHECK(ac::rocblas_table_allowed("gfx1030", "5.6.0.8d1ae90e"));
    CHECK(ac::rocblas_table_allowed("gfx1201", "anything"));

    // the shipped files are well-formed, name the arch and version of the list, and the loader takes each only for that pair
    for (const auto& t : ac::kRocblasTables) {
        std::string tag(t.version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        const std::string path = std::string("../terminal/tools/hip/") + t.arch + "-rocblas-" + tag + ".txt";
        std::string err;
        strata::prefill::rocblas_tuning::Table ok;
        CHECK(ok.load(path, t.arch, t.version, err));
        CHECK(ok.rows().size() == 112);
        strata::prefill::rocblas_tuning::Table other_version, other_arch;
        CHECK(!other_version.load(path, t.arch, "5.6.0.8d1ae90e", err));   // a non-matching rocBLAS build just does not load
        CHECK(!other_arch.load(path, "gfx1031", t.version, err));
    }

    // the dot product (device code): the build option decides; the intrinsics header carries the guard
    CHECK(ac::dot4_gfx1010_built() == (EXPECT_DOT_BUILT != 0));
    {
        std::ifstream in("include/strata/hip_compat/intrinsics.hpp");
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        CHECK(text.find("defined(__gfx1010__) && !defined(STRATA_AMD_COMMUNITY_DOT_OFF)") != std::string::npos);
        CHECK(text.find("!defined(STRATA_GFX1012_PORTABLE_DOT)") != std::string::npos);
    }

    // #3073: the simt2 selector is the default of gfx103x only, and only with the switch on
    unset(kSwitch);
    CHECK(ac::select_simt2_default("gfx1030") && ac::select_simt2_default("gfx1031:xnack-") && ac::select_simt2_default("gfx1032"));
    CHECK(!ac::select_simt2_default("gfx1201") && !ac::select_simt2_default("gfx1100") && !ac::select_simt2_default("gfx906") &&
          !ac::select_simt2_default("gfx1010") && !ac::select_simt2_default("") && !ac::select_simt2_default(nullptr));
    put(kSwitch, "0");
    CHECK(!ac::select_simt2_default("gfx1030"));
    unset(kSwitch);

    // the log line: only gfx1010 / gfx103x, and it says what the switch does
    unset(kSwitch);
    CHECK(ac::startup_line("gfx1201").empty());
    CHECK(ac::startup_line("gfx1100").empty());
    CHECK(ac::startup_line("gfx1012").empty());
    CHECK(ac::startup_line("gfx1031").find("simt2") != std::string::npos);   // gfx1031: the selector default only
    CHECK(ac::startup_line("gfx1030").find("simt2") != std::string::npos);
    CHECK(ac::startup_line("").empty());
    CHECK(ac::startup_line(nullptr).empty());
    CHECK(ac::startup_line("gfx1030:xnack-").find("on:") != std::string::npos);
    CHECK(ac::startup_line("gfx1010").find("STRATA_AMD_COMMUNITY_TUNING=0") != std::string::npos);
    put(kSwitch, "0");
    CHECK(ac::startup_line("gfx1030").find("off:") != std::string::npos);
    CHECK(ac::startup_line("gfx1010").find("rebuild") != std::string::npos || !ac::dot4_gfx1010_built());

    if (fails == 0) std::printf("amd_community_test: OK\n");
    return fails == 0 ? 0 : 1;
}
