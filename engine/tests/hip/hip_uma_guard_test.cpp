// tests/hip/hip_uma_guard_test.cpp - Unit test for first-class AMD iGPU (gfx1103 780M, gfx1150 Strix Point 890M/880M,
// gfx1151 Strix Halo) UMA memory pool & <= 1.0s TDR/MES dispatch guards (runs on CPU, no GPU required).
#include "strata/hip_uma_guard.hpp"

#include <cassert>
#include <cstdio>

int main() {
    using namespace strata::hip_uma;

    // 1. AMD generation classification
    assert(classify_amd_gen("gfx1200") == AmdGen::Rdna4);
    assert(classify_amd_gen("gfx1201:sramecc-:xnack-") == AmdGen::Rdna4);
    assert(classify_amd_gen("gfx1150") == AmdGen::Rdna35);
    assert(classify_amd_gen("gfx1151:sramecc-:xnack-") == AmdGen::Rdna35);
    assert(classify_amd_gen("gfx1100") == AmdGen::Rdna3);
    assert(classify_amd_gen("gfx1103") == AmdGen::Rdna3);
    assert(classify_amd_gen("gfx1030") == AmdGen::Rdna2);
    assert(classify_amd_gen("gfx1031") == AmdGen::Rdna2);

    // 2. iGPU identification
    assert(is_known_amd_igpu_arch("gfx1103"));
    assert(is_known_amd_igpu_arch("gfx1103:xnack-"));
    assert(is_known_amd_igpu_arch("gfx1150"));
    assert(is_known_amd_igpu_arch("gfx1151"));
    assert(!is_known_amd_igpu_arch("gfx1100"));
    assert(!is_known_amd_igpu_arch("gfx1201"));

    // 3. <= 1.0s dispatch guard checks across all AMD tiers
    const auto g780_win = compute_dispatch_guard("gfx1103", 12, true, true);
    assert(g780_win.max_prefill_chunk <= 256);
    assert(g780_win.visit_flush_stride <= 8);
    assert(g780_win.max_persistent_blocks <= 16);
    assert(g780_win.max_spin_ms <= 150);

    const auto g780_lin = compute_dispatch_guard("gfx1103", 12, true, false);
    assert(g780_lin.max_prefill_chunk <= 512);
    assert(g780_lin.visit_flush_stride <= 8);

    const auto g890_win = compute_dispatch_guard("gfx1150", 16, true, true);
    assert(g890_win.max_prefill_chunk <= 512);
    assert(g890_win.visit_flush_stride <= 12);
    assert(g890_win.max_persistent_blocks <= 24);

    const auto ghalo_win = compute_dispatch_guard("gfx1151", 40, true, true);
    assert(ghalo_win.max_prefill_chunk == 2048);
    assert(ghalo_win.visit_flush_stride == 16);
    // RDNA3.5 HIP reports multiProcessorCount in WGPs (20 WGPs = 40 CUs on Strix Halo, 6 WGPs = 12 CUs on 780M)
    const auto ghalo_wgp = compute_dispatch_guard("gfx1151", 20, true, true);
    assert(ghalo_wgp.max_prefill_chunk == 2048);
    assert(ghalo_wgp.visit_flush_stride == 16);
    const auto g780_wgp = compute_dispatch_guard("gfx1103", 6, true, true);
    assert(g780_wgp.max_prefill_chunk == 256);
    assert(g780_wgp.visit_flush_stride == 8);

    const auto grdna2 = compute_dispatch_guard("gfx1031", 40, false, false);
    assert(grdna2.max_prefill_chunk <= 1024);

    const auto gr9700 = compute_dispatch_guard("gfx1201", 64, false, false);
    assert(gr9700.max_prefill_chunk == 8192);
    assert(gr9700.visit_flush_stride == 48);

    // 4. Effective UMA memory expands tiny 512 MiB BIOS carveout using host RAM
    uint64_t eff_fr = 0, eff_tot = 0;
    const uint64_t carve512 = 512ull << 20;
    effective_uma_memory(true, carve512, carve512, eff_fr, eff_tot);
    assert(eff_tot >= carve512);
    assert(eff_fr >= carve512);

    uint64_t d_fr = 0, d_tot = 0;
    effective_uma_memory(false, carve512, carve512, d_fr, d_tot);
    assert(d_fr == carve512 && d_tot == carve512);

    std::printf("hip_uma_guard_test PASS (780M/890M/StrixHalo/RDNA2/RDNA4 guards verified)\n");
    return 0;
}
