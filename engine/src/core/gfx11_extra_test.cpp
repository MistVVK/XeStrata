// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/core/gfx11_extra_test.cpp - discrete RDNA3 (gfx1100 / gfx1101 / gfx1102) defaults and off switches (no GPU needed).
#include "strata/core/arch_defaults.hpp"
#include "strata/core/gfx11_extra.hpp"
#include "strata/kernels/rdna3_wddm_guard.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset_var(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}

static void put_var(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool has_kv(const std::vector<std::pair<std::string, std::string>>& t, const char* k, const char* v) {
    for (const auto& kv : t) if (kv.first == k && kv.second == v) return true;
    return false;
}

static void clear_all_gfx11_vars() {
    for (const auto& kv : strata::core::gfx11_discrete_exact_env("gfx1100", false, true)) unset_var(kv.first.c_str());
    for (const auto& kv : strata::core::gfx11_discrete_wmma_env("gfx1100", false)) unset_var(kv.first.c_str());
    for (const auto& kv : strata::core::gfx11_discrete_parallel_env("gfx1100", 4, false)) unset_var(kv.first.c_str());
    unset_var("STRATA_GFX11_DEFAULTS");
    unset_var("STRATA_GFX11_WMMA_DEFAULTS");
    unset_var("STRATA_DEFAULTS");
}

int main() {
    using namespace strata::core;
    clear_all_gfx11_vars();

    // 1. Target matching: discrete RDNA3 (gfx1100, gfx1101, gfx1102, feature suffixes) only
    for (const char* g : {"gfx1100", "gfx1101", "gfx1102", "gfx1100:sramecc-:xnack-", "gfx1102:xnack-"}) {
        const auto ex_lin = gfx11_discrete_exact_env(g, false, false);
        const auto ex_win = gfx11_discrete_exact_env(g, false, true);
        const auto wm     = gfx11_discrete_wmma_env(g, false);
        CHECK(ex_lin.size() == 24);
        CHECK(ex_win.size() == 26);
        CHECK(wm.size() == 12);
        CHECK(has_kv(ex_lin, "STRATA_GDN_HEAD", "1"));
        CHECK(has_kv(ex_lin, "STRATA_GDN_PP", "2"));
        CHECK(has_kv(ex_lin, "STRATA_EXPERT_V2", "1"));
        CHECK(has_kv(ex_lin, "STRATA_EXPERT_V2K", "1"));
        CHECK(has_kv(ex_lin, "STRATA_HEAD_MIX_MULTI", "1"));
        CHECK(has_kv(ex_lin, "STRATA_SMALL_UNR", "3"));
        CHECK(!has_kv(ex_lin, "STRATA_SH_STREAM", "1"));
        CHECK(!has_kv(ex_lin, "STRATA_VERIFY_ASYNC_LAUNCH", "1"));
        CHECK(has_kv(ex_win, "STRATA_VERIFY_ASYNC_LAUNCH", "1"));
        CHECK(has_kv(ex_win, "GPU_ENABLE_PAL_SDMA", "1"));
        CHECK(has_kv(wm, "STRATA_HIP_WMMA", "1"));
        CHECK(has_kv(wm, "STRATA_SELECT_WMMA", "1"));
        CHECK(!has_kv(wm, "STRATA_GDN_CHUNKED", "1"));
        CHECK(has_kv(wm, "STRATA_PF_F16", "2"));
        CHECK(has_kv(wm, "STRATA_PF_P1A", "1"));
        CHECK(has_kv(wm, "STRATA_PF_P1A_L2", "1"));
        CHECK(has_kv(wm, "STRATA_DFLASH_WMMA", "1"));
        CHECK(has_kv(wm, "STRATA_HIP_FAST_DEC", "1"));
        CHECK(!has_kv(ex_lin, "STRATA_VERIFY_DEVICE_HITS", "1"));
        CHECK(gfx11_prefill_share_default(g, false));
    }

    // Non-discrete-RDNA3 targets must get nothing
    for (const char* other : {"gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201", "gfx1030", "gfx906", "gfx11000", "", "sm_120"}) {
        CHECK(gfx11_discrete_exact_env(other, false, false).empty());
        CHECK(gfx11_discrete_exact_env(other, false, true).empty());
        CHECK(gfx11_discrete_wmma_env(other, false).empty());
        CHECK(gfx11_discrete_parallel_env(other, 4, false).empty());
        CHECK(!gfx11_prefill_share_default(other, false));
    }
    CHECK(gfx11_discrete_exact_env(nullptr, false, false).empty());
    CHECK(gfx11_discrete_wmma_env(nullptr, false).empty());

    // 2. Off switches: STRATA_DEFAULTS=026, STRATA_GFX11_DEFAULTS=0, STRATA_GFX11_WMMA_DEFAULTS=0
    CHECK(gfx11_discrete_exact_env("gfx1100", true, false).empty());
    CHECK(gfx11_discrete_wmma_env("gfx1100", true).empty());
    CHECK(!gfx11_prefill_share_default("gfx1100", true));

    put_var("STRATA_GFX11_WMMA_DEFAULTS", "0");
    CHECK(!gfx11_discrete_exact_env("gfx1100", false, false).empty());   // exact stays on
    CHECK(gfx11_discrete_wmma_env("gfx1100", false).empty());            // WMMA turns off
    CHECK(!gfx11_prefill_share_default("gfx1100", false));
    unset_var("STRATA_GFX11_WMMA_DEFAULTS");

    put_var("STRATA_GFX11_DEFAULTS", "0");
    CHECK(gfx11_discrete_exact_env("gfx1100", false, true).empty());
    CHECK(gfx11_discrete_wmma_env("gfx1100", false).empty());
    CHECK(gfx11_discrete_parallel_env("gfx1100", 4, false).empty());
    CHECK(!gfx11_prefill_share_default("gfx1100", false));
    unset_var("STRATA_GFX11_DEFAULTS");

    // 3. apply_arch_defaults integration: discrete vs integrated, user override preservation, batch slots
    clear_all_gfx11_vars();
    put_var("STRATA_PF_P1A", "0");           // explicit user =0 must win
    put_var("STRATA_HEAD_MIX_MULTI", "0");   // explicit user =0 must win
    CHECK(apply_arch_defaults("gfx1100", false, 4).empty());   // kept apart from main returned list, like gfx12_extra
    CHECK(std::string(std::getenv("STRATA_PF_P1A")) == "0");
    CHECK(std::string(std::getenv("STRATA_HEAD_MIX_MULTI")) == "0");
    CHECK(std::getenv("STRATA_HIP_WMMA") != nullptr && std::string(std::getenv("STRATA_HIP_WMMA")) == "1");
    CHECK(std::getenv("STRATA_PF_F16") != nullptr && std::string(std::getenv("STRATA_PF_F16")) == "2");
    CHECK(std::getenv("STRATA_EXPERT_V2") != nullptr && std::string(std::getenv("STRATA_EXPERT_V2")) == "1");
    CHECK(std::getenv("STRATA_CKPT_POOL") != nullptr && std::string(std::getenv("STRATA_CKPT_POOL")) == "1");

    // Integrated == true on gfx1100 (hypothetical/mixed) must NOT apply discrete RDNA3 table
    clear_all_gfx11_vars();
    apply_arch_defaults("gfx1100", true, 4);
    CHECK(std::getenv("STRATA_PF_P1A_L2") == nullptr);
    CHECK(std::getenv("STRATA_EXPERT_V2") == nullptr);

    // 4. Chunk ceiling helper
    CHECK(gfx11_chunk_ceiling(nullptr, false) == 32768);
    CHECK(gfx11_chunk_ceiling("", false) == 32768);
    CHECK(gfx11_chunk_ceiling(nullptr, true) == 0);
    CHECK(gfx11_chunk_ceiling("16", false) == 16384);
    CHECK(gfx11_chunk_ceiling("32", false) == 32768);
    CHECK(gfx11_chunk_ceiling("1", false) == 32768);
    CHECK(gfx11_chunk_ceiling("0", false) == 0);

    // 5. CU-count-safe persistent grid sizing & WDDM spin budget (rdna3_wddm_guard.hpp)
    {
        using namespace strata::kernels;
        // 12-CU 780M (gfx1103), 16-CU 890M (gfx1150), 40-CU Strix Halo (gfx1151), 96-CU 7900 XTX (gfx1100)
        const auto p780_win = compute_safe_persistent_grid(12, 2, 168, true);
        CHECK(p780_win.safe_blocks == 16 && !p780_win.can_fuse_fixed);
        const auto p890_win = compute_safe_persistent_grid(16, 2, 168, true);
        CHECK(p890_win.safe_blocks == 24 && !p890_win.can_fuse_fixed);
        const auto phalo_win = compute_safe_persistent_grid(40, 2, 168, true);
        CHECK(phalo_win.safe_blocks == 72 && !phalo_win.can_fuse_fixed);
        const auto pxtx_win = compute_safe_persistent_grid(96, 2, 168, true);
        CHECK(pxtx_win.safe_blocks == 184 && !pxtx_win.can_fuse_fixed);   // Windows WDDM never fuses fixed 168-CTA barrier
        const auto pxtx_lin = compute_safe_persistent_grid(96, 2, 168, false, true);
        CHECK(pxtx_lin.safe_blocks == 188 && pxtx_lin.can_fuse_fixed);    // Linux 96-CU can safely fuse 168 CTAs
        const auto p780_lin = compute_safe_persistent_grid(12, 2, 168, false, true);
        CHECK(p780_lin.safe_blocks == 20 && !p780_lin.can_fuse_fixed);
        CHECK(rdna3_spin_budget_ticks(true) == 15'000'000ULL);            // 150 ms @ 100 MHz
        CHECK(rdna3_spin_budget_ticks(false) == 150'000'000ULL);          // 1,500 ms @ 100 MHz
    }

    clear_all_gfx11_vars();
    if (fails == 0) std::printf("gfx11_extra_test: OK\n");
    return fails == 0 ? 0 : 1;
}
