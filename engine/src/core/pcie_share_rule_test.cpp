// src/core/pcie_share_rule_test.cpp - the decode miss share's rule, its ceiling and the 0.2.10.1 defaults (no GPU needed).
#include "strata/core/pcie_share_rule.hpp"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    using namespace strata;
    // k: the shipped 1.85 from 30 GB/s up and whenever the rule is off; 0.8 at 12 and below; linear between
    CHECK(near(pcie_pool_k(30.0, true), 1.85));
    CHECK(near(pcie_pool_k(65.0, true), 1.85));
    CHECK(near(pcie_pool_k(10.3, true), 0.8));
    CHECK(near(pcie_pool_k(12.0, true), 0.8));
    CHECK(near(pcie_pool_k(21.0, true), 0.8 + 9.0 / 18.0 * 1.05));
    CHECK(near(pcie_pool_k(5.0, false), 1.85));
    // measured on the RTX PRO 6000 box (link 56 GB/s): a strong pool settles at 0.35 with either rule
    CHECK(near(pcie_share_for(56.1, 61.2, kPcieCeilingShipped, false), 0.35));
    CHECK(near(pcie_share_for(56.1, 61.2, kPcieCeilingFollow, true), 0.35));
    // a strong pool is bit-for-bit the old share over every link up to PCIe 5 x16 and every rate from 30 GB/s
    for (double link = 4.0; link <= 64.0; link += 0.5)
        for (double r = 30.0; r <= 120.0; r += 0.5)
            CHECK(near(pcie_share_for(link, r, kPcieCeilingShipped, false), pcie_share_for(link, r, kPcieCeilingFollow, true)));
    // a weak pool: 4 workers (R about 10.3-13 GB/s) 0.60 -> 0.85, 2 workers (R about 8) -> 0.90 and more
    CHECK(near(pcie_share_for(56.3, 10.3, kPcieCeilingShipped, false), 0.60));
    CHECK(near(pcie_share_for(56.3, 10.3, kPcieCeilingFollow, true), 0.85));
    CHECK(near(pcie_share_for(56.2, 8.0, kPcieCeilingFollow, true), 0.90));
    CHECK(near(pcie_share_for(56.0, 22.3, kPcieCeilingFollow, true), 0.65));   // 8 workers: 0.60 -> 0.65
    // the ceiling holds and the floor holds
    CHECK(near(pcie_share_for(56.0, 0.5, kPcieCeilingFollow, true), 0.95));
    CHECK(near(pcie_share_for(1.0, 200.0, kPcieCeilingFollow, true), 0.10));
    // never above the ceiling, never below 0.10, steps of 0.05
    for (double r = 1.0; r <= 100.0; r += 1.0) {
        const double s = pcie_share_for(50.0, r, kPcieCeilingFollow, true);
        CHECK(s >= 0.10 - 1e-9 && s <= 0.95 + 1e-9 && near(std::round(s * 20.0) / 20.0, s));
    }
    // defaults: a discrete card follows the pool, an integrated GPU and the master switch keep the shipped rule, an explicit value wins
    CHECK(near(pcie_share_ceiling(true, false, nullptr), 0.95));
    CHECK(near(pcie_share_ceiling(false, false, nullptr), 0.60));
    CHECK(near(pcie_share_ceiling(true, true, nullptr), 0.60));
    CHECK(near(pcie_share_ceiling(false, false, "0.8"), 0.8));
    CHECK(near(pcie_share_ceiling(true, false, "2"), 1.0));
    CHECK(near(pcie_share_ceiling(true, false, "0.6"), 0.6));
    CHECK(pcie_k_rule_default(true, false, nullptr));
    CHECK(!pcie_k_rule_default(false, false, nullptr));
    CHECK(!pcie_k_rule_default(true, true, nullptr));
    CHECK(!pcie_k_rule_default(true, false, "0"));
    CHECK(pcie_k_rule_default(false, false, "1"));
    // the layer split: on for a discrete CUDA card, off for HIP (opt-in), for an integrated GPU and under 026; =0 / =1 win
    CHECK(pcie_auto_split_default(false, true, false, nullptr));
    CHECK(!pcie_auto_split_default(true, true, false, nullptr));
    CHECK(!pcie_auto_split_default(false, false, false, nullptr));
    CHECK(!pcie_auto_split_default(false, true, true, nullptr));
    CHECK(!pcie_auto_split_default(false, true, false, "0"));
    CHECK(pcie_auto_split_default(true, true, false, "1"));
    CHECK(pcie_auto_split_default(false, false, false, "1"));
    CHECK(pcie_auto_split_default(false, true, false, ""));   // empty = unset
    // PcieAuto on HIP outside gfx12: on for a discrete gfx103x card, off for other parts, an integrated GPU and 026; =auto turns it on
    // anywhere, any other value (old) leaves it off, empty = unset
    CHECK(pcie_auto_hip_default("gfx1030", true, false, nullptr));
    CHECK(pcie_auto_hip_default("gfx1031", true, false, nullptr));
    CHECK(pcie_auto_hip_default("gfx1030:xnack-", true, false, nullptr));
    CHECK(pcie_auto_hip_default("gfx1030", true, false, ""));
    CHECK(!pcie_auto_hip_default("gfx1036", false, false, nullptr));   // Raphael iGPU
    CHECK(!pcie_auto_hip_default("gfx1030", true, true, nullptr));
    CHECK(!pcie_auto_hip_default("gfx1030", true, false, "old"));
    CHECK(!pcie_auto_hip_default("gfx1030", true, false, "0"));
    CHECK(!pcie_auto_hip_default("gfx1100", true, false, nullptr));
    CHECK(!pcie_auto_hip_default("gfx1010", true, false, nullptr));
    CHECK(!pcie_auto_hip_default(nullptr, true, false, nullptr));
    CHECK(pcie_auto_hip_default("gfx1100", true, false, "auto"));
    CHECK(pcie_auto_hip_default("gfx1036", false, true, "auto"));
    if (fails == 0) std::printf("pcie_share_rule_test: ok\n");
    return fails == 0 ? 0 : 1;
}
