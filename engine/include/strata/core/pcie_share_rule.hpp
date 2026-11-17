#pragma once
// The decode miss split between the CPU pool and the PCIe link (PcieAuto): the share of a layer's missed experts the link carries is
// link / (link + k R), R the CPU pool's rate in GB/s of expert bytes, rounded to 0.05 and kept between 0.10 and a ceiling.
// 0.2.10.1 (HYBRID, round 0211): the ceiling is 0.95 and k falls with R, on a discrete card; STRATA_PCIE_SHARE_MAX and STRATA_PCIE_K_RULE
// set either by hand, STRATA_PCIE_AUTO_SPLIT=0/1 decides whether a CUDA layer split refines its shares too.  #4007: on by default on discrete
// gfx103x (pcie_auto_hip_default).  Pure functions, no GPU.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata {

/// The shipped ceiling and the 0.95 of the rule that follows the pool.
inline constexpr double kPcieCeilingShipped = 0.60;
inline constexpr double kPcieCeilingFollow  = 0.95;

/// The factor k.  1.85 (shipped; fits a strong pool whose reads compete with the uploads for RAM bandwidth) at R >= 30 GB/s of expert bytes,
/// falling linearly to 0.8 at R <= 12 (a compute-bound weak pool: 4 workers settled best at 0.85 where 1.85 gave 0.70).  k_rule false: 1.85.
inline double pcie_pool_k(double r, bool k_rule) {
    if (!k_rule || r >= 30.0) return 1.85;
    if (r <= 12.0) return 0.8;
    return 0.8 + (r - 12.0) / 18.0 * (1.85 - 0.8);
}

/// The share for a link of `link` GB/s against a pool of `r` GB/s, rounded to 0.05, between 0.10 and `ceiling`.
inline double pcie_share_for(double link, double r, double ceiling, bool k_rule) {
    const double raw = link / (link + pcie_pool_k(r, k_rule) * r);
    return std::min(ceiling, std::max(0.10, std::round(raw * 20.0) / 20.0));
}

/// Does PcieAuto also refine a layer split's shares?  `env` is STRATA_PCIE_AUTO_SPLIT (nullptr when unset): an explicit value wins either way.
/// Unset: on for a discrete CUDA card (2 x RTX PRO 6000, 12 GB per card: +12% decode), off on HIP (opt-in; measured on 2 x R9700 with mixed
/// results), off on an integrated GPU and under the master switch 026.
inline bool pcie_auto_split_default(bool hip_build, bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::atoi(env) != 0;
    return !hip_build && discrete && !v026;
}

/// The pool-rate factor on?  STRATA_PCIE_K_RULE (nullptr unset) wins; unset: on for a discrete card, off for an integrated GPU (its "link" is the
/// shared memory, not PCIe) and under the master switch 026.
inline bool pcie_k_rule_default(bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::atoi(env) != 0;
    return discrete && !v026;
}

/// PcieAuto on a HIP part outside gfx12 (gfx12 has it through the decode defaults).  `env` is STRATA_PCIE_FRAC_DEFAULT (nullptr unset): "auto"
/// turns it on for any HIP part; any other value (for example "old", which the caller also reads to keep the start-up rule) leaves it off.
/// Unset: on for a discrete gfx103x card (RX 6000), off for every other part, for an integrated GPU (gfx1035 / gfx1036 APUs share the host's RAM)
/// and under the master switch 026.  Measured on 0.2.13, unset against auto, ten interleaved pairs each (#4007): RX 6900 XT on Windows decode
/// +13.3% at 32K / +15.1% at 4K (10/10), on Linux +1.0% / +4.9% (10/10); RX 6700 XT (gfx1031, Linux) +3.4% / +4.9% (10/10, 9/10); prompts unchanged.
inline bool pcie_auto_hip_default(const char* gcn_arch, bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::strcmp(env, "auto") == 0;
    return discrete && !v026 && gcn_arch != nullptr && std::strncmp(gcn_arch, "gfx103", 6) == 0;
}

/// The ceiling.  STRATA_PCIE_SHARE_MAX (nullptr unset) wins (clamped 0.10..1.0); unset: 0.95 on a discrete card, 0.60 otherwise.
inline double pcie_share_ceiling(bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::min(1.0, std::max(0.10, std::atof(env)));
    return discrete && !v026 ? kPcieCeilingFollow : kPcieCeilingShipped;
}

}  // namespace strata
