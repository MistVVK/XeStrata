// src/prefill/hipblaslt_tuning_test.cpp - pure CPU unit tests for hipblaslt_tuning.hpp (#6574).
#include "hipblaslt_tuning.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fails; } } while (0)

int main() {
    using namespace strata::prefill::hipblaslt;

    // parse_cwsr_enable_value
    CHECK(parse_cwsr_enable_value("0\n") == 0);
    CHECK(parse_cwsr_enable_value("  0  ") == 0);
    CHECK(parse_cwsr_enable_value("1\n") == 1);
    CHECK(parse_cwsr_enable_value("-1\n") == 1);
    CHECK(parse_cwsr_enable_value("") == -1);
    CHECK(parse_cwsr_enable_value("abc") == -1);
    CHECK(parse_cwsr_enable_value("0 extra") == -1);

    // cmdline_has_cwsr_disabled
    CHECK(cmdline_has_cwsr_disabled("BOOT_IMAGE=/vmlinuz ro quiet amdgpu.cwsr_enable=0 splash"));
    CHECK(cmdline_has_cwsr_disabled("amdgpu.cwsr_enable=0"));
    CHECK(!cmdline_has_cwsr_disabled("ro quiet amdgpu.cwsr_enable=1"));
    CHECK(!cmdline_has_cwsr_disabled("ro quiet amdgpu.cwsr_enable=01"));
    CHECK(!cmdline_has_cwsr_disabled(""));

    // gfx1103_hipblaslt_cwsr_safe
    CHECK(!gfx1103_hipblaslt_cwsr_safe("gfx1103", true, 1, false));    // Linux gfx1103 with CWSR active -> unsafe
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1103", true, 0, false));     // Linux gfx1103 with sysfs cwsr_enable=0 -> safe
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1103", true, 1, true));      // Linux gfx1103 with cmdline disable -> safe
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1103", true, -1, false));    // unreadable sysfs -> do not falsely block
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1103", false, 1, false));    // Windows -> safe (no KFD CWSR)
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1151", true, 1, false));     // Strix Halo -> untouched
    CHECK(gfx1103_hipblaslt_cwsr_safe("gfx1100", true, 1, false));     // discrete RDNA3 -> untouched

    // hipblaslt_prelaunch_error
    CHECK(hipblaslt_prelaunch_error(1));   // NOT_INITIALIZED
    CHECK(hipblaslt_prelaunch_error(2));   // ALLOC_FAILED
    CHECK(hipblaslt_prelaunch_error(3));   // INVALID_VALUE
    CHECK(hipblaslt_prelaunch_error(8));   // ARCH_MISMATCH
    CHECK(hipblaslt_prelaunch_error(10));  // NOT_SUPPORTED
    CHECK(!hipblaslt_prelaunch_error(0));  // SUCCESS
    CHECK(!hipblaslt_prelaunch_error(5));  // EXECUTION_FAILED (may have written Y)
    CHECK(!hipblaslt_prelaunch_error(6));  // INTERNAL_ERROR (may have written Y)

    // TuningTable load & closest lookup
    {
        const auto tmp = std::filesystem::temp_directory_path() / "strata_hipblaslt_tuning_test.txt";
        {
            std::ofstream out(tmp);
            out << "# test table\n"
                << "STRATA_HIPBLASLT_TUNING_V1 gfx1103 100200\n"
                << "f16 2560 2560 2560 1024 42\n"
                << "f16 2560 2560 2560 4096 99\n";
        }
        TuningTable tbl;
        std::string err;
        CHECK(tbl.load(tmp.string(), "gfx1103", 100200, err));
        CHECK(tbl.rows().size() == 2);
        const TuningRow* r1 = tbl.closest(InputType::f16, 2560, 2560, 2560, 2000);
        CHECK(r1 != nullptr && r1->solution_id == 42);
        const TuningRow* r2 = tbl.closest(InputType::f16, 2560, 2560, 2560, 3500);
        CHECK(r2 != nullptr && r2->solution_id == 99);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    if (g_fails == 0) std::printf("hipblaslt_tuning_test: ok\n");
    return g_fails == 0 ? 0 : 1;
}
