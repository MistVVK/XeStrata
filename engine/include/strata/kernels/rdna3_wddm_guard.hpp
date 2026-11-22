// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// include/strata/kernels/rdna3_wddm_guard.hpp - Shared RDNA3 (gfx1100..gfx1151) & Windows WDDM safety primitives:
//   1. CU-count-aware co-resident persistent grid sizing (never hardcodes 96 CUs / 168 CTAs; safe on 12-CU 780M gfx1103,
//      16-CU 890M gfx1150, 40-CU Strix Halo gfx1151, 32-CU 7600 XT gfx1102, 54-60 CU 7700/7800 XT gfx1101, 80-96 CU gfx1100).
//   2. 100 MHz shader-clock WDDM TDR-safe spin-wait guard (150 ms Windows budget vs 2,000 ms TdrDelay; 1,500 ms Linux).
//   3. Reusable across Qwen3.8, Qwen3.6, Hy4-Flash, and Inkling-2-Mini HIP backends.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::kernels {

/// RDNA3 / RDNA3.5 (gfx1100..gfx1151) shader reference clock runs at 100 MHz (10 ns/tick).
constexpr uint64_t kRdna3ClockHz = 100'000'000ULL;

/// Per-kernel GPU spin budget in 100 MHz ticks: 150 ms on Windows WDDM (13x margin below 2,000 ms TdrDelay), 1,500 ms on Linux.
constexpr uint64_t rdna3_spin_budget_ticks(bool windows_wddm) noexcept {
    return windows_wddm ? (150ULL * (kRdna3ClockHz / 1000ULL)) : (1500ULL * (kRdna3ClockHz / 1000ULL));
}

struct PersistentGridPlan {
    int safe_blocks = 0;         ///< Maximum deadlock-free co-resident CTA count for this GPU + OS
    bool can_fuse_fixed = false; ///< True only when fixed `required_fixed_blocks` safely co-reside without WDDM preemption
};

/// Computes a deadlock-free co-resident CTA count for persistent grid-barrier kernels across 12..96 CU RDNA3 parts.
constexpr PersistentGridPlan compute_safe_persistent_grid(
    int cu_count,
    int max_blocks_per_cu,
    int required_fixed_blocks,
    bool windows_wddm,
    bool display_attached = true) noexcept {
    if (cu_count <= 0 || max_blocks_per_cu <= 0) return {0, false};
    const int total_slots = cu_count * max_blocks_per_cu;
    // Reserve headroom for Windows DWM / desktop compositor so grid barriers never deadlock under CU preemption.
    const int reserve_cus = windows_wddm ? 4 : (display_attached ? 2 : 0);
    const int reserve_slots = std::min(total_slots / 2, reserve_cus * max_blocks_per_cu);
    const int safe = std::max(1, total_slots - reserve_slots);
    const bool fixed_ok = !windows_wddm && (required_fixed_blocks > 0) && (safe >= required_fixed_blocks);
    return {safe, fixed_ok};
}

#if defined(__HIPCC__)
/// Bounded device-side spin-wait check for RDNA3 persistent/doorbell kernels.
/// Samples wall_clock64() once every 256 iterations (~16 us) so spin loops add zero overhead while remaining TDR-proof.
__device__ __forceinline__ bool rdna3_spin_expired(
    uint64_t t0_ticks, uint32_t& iter, unsigned* mapped_stall = nullptr, unsigned code = 1u) {
#if defined(__HIP_DEVICE_COMPILE__)
    if ((++iter & 255u) != 0u) return false;
#if defined(_WIN32)
    constexpr uint64_t kBudget = rdna3_spin_budget_ticks(true);
#else
    constexpr uint64_t kBudget = rdna3_spin_budget_ticks(false);
#endif
    const uint64_t now = (uint64_t) wall_clock64();
    if (now - t0_ticks < kBudget) return false;
    if (mapped_stall != nullptr) {
        __hip_atomic_store(mapped_stall, code, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    return true;
#else
    (void) t0_ticks; (void) iter; (void) mapped_stall; (void) code;
    return false;
#endif
}
#endif

}  // namespace strata::kernels
