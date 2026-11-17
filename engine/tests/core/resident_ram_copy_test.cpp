// STRATA_RESIDENT_PIN and the automatic RAM copy (the engine's fallback when the expert arena does not fit RAM).
// (a) The variable was read only while parsing --resident-experts / --resident-budget-gib; the automatic RAM copy set the pin
//     unconditionally (generate.cpp:7776 at dbb2d168c), so STRATA_RESIDENT_PIN=0 left it page-locked (the #5718 cap test's
//     arm Z: "page-locked and mapped" with the variable set).  Here the fallback's mode must read it like the flags do.
// (b) The line printed when that copy's budget is cut to the available RAM said "--resident-budget-gib 47.07 exceeds
//     available RAM" although no such flag was given.  The flag's own line stays byte for byte; the automatic one names no flag.
// No GPU, no model: the rules are header-only.
#include "strata/core/resident_ram_copy.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// the cap test's numbers (r1-Z-engine.log line 46): budget 47.07 GiB, 49.49 GiB available, 4 GiB headroom, cut to 45.24
const uint64_t budget = (uint64_t) (47.07 * 1073741824.0);
const uint64_t available = (uint64_t) (49.49 * 1073741824.0);
const uint64_t headroom = 4ull << 30;
const uint64_t clamped = (uint64_t) (45.24 * 1073741824.0);
}  // namespace

int main() {
    // (a) the variable's reading, shared by the flags and the fallback
    CHECK(resident_pin_from_env(nullptr));           // unset: page-locked, as before
    CHECK(!resident_pin_from_env("0"));              // "0": pageable
    CHECK(resident_pin_from_env("1"));
    CHECK(resident_pin_from_env(""));
    CHECK(resident_pin_from_env("00"));              // only "0" turns it off, as the old string compare did
    CHECK(resident_pin_from_env("0 "));

    // (a) the automatic RAM copy's mode
    const AutoRamCopyMode unset = auto_ram_copy_mode(nullptr);
    CHECK(unset.resident_cpu_experts && unset.pin && unset.soft);       // unchanged default
    const AutoRamCopyMode off = auto_ram_copy_mode("0");
    if (off.pin)
        std::fprintf(stderr, "the automatic RAM copy with STRATA_RESIDENT_PIN=0: page-locked (the variable was not read)\n");
    CHECK(off.resident_cpu_experts && !off.pin && off.soft);             // the variable acts on the fallback too
    const AutoRamCopyMode one = auto_ram_copy_mode("1");
    CHECK(one.pin);

    // (b) the clamp line: the flag's text byte for byte as before
    const std::string flag = resident_clamp_line(ResidentBudgetFrom::Flag, budget, "RAM", available, headroom, clamped);
    CHECK(flag == "FileExpertSource: --resident-budget-gib 47.07 exceeds available RAM (49.49 GiB minus 4 GiB headroom "
                  "and a 0.25 GiB margin): 45.24 GiB\n");
    const std::string fits = resident_clamp_line(ResidentBudgetFrom::WhatFits, budget, "commit", available, headroom, clamped);
    CHECK(fits == "FileExpertSource: room for the complement: 45.24 GiB (49.49 GiB available commit minus 4 GiB headroom "
                  "and a 0.25 GiB margin)\n");
    // (b) the automatic copy: no flag named, the copy and its numbers are
    const std::string automatic = resident_clamp_line(ResidentBudgetFrom::Automatic, budget, "RAM", available, headroom, clamped);
    std::fprintf(stderr, "automatic: %s", automatic.c_str());
    CHECK(automatic.find("--resident-budget-gib") == std::string::npos);
    CHECK(automatic.find("--") == std::string::npos);
    CHECK(automatic.find("automatic RAM copy of 47.07 GiB") != std::string::npos);
    CHECK(automatic.find("49.49 GiB minus 4 GiB headroom and a 0.25 GiB margin") != std::string::npos);
    CHECK(automatic.find("using 45.24 GiB") != std::string::npos);
    CHECK(!automatic.empty() && automatic.back() == '\n');
    CHECK(automatic == "strata: the automatic RAM copy of 47.07 GiB does not fit the available RAM (49.49 GiB minus 4 GiB "
                       "headroom and a 0.25 GiB margin); using 45.24 GiB\n");

    if (failed) { std::fprintf(stderr, "resident_ram_copy_test: %d FAILED\n", failed); return 1; }
    std::printf("resident_ram_copy_test: ok\n");
    return 0;
}
