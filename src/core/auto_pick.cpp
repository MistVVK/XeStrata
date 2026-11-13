// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/auto_pick.cpp - the switches the first decode windows choose (auto_pick.hpp): the adaptive tier's swap wait
// and a helper GPU's experts out of the primary's candidates.  Both show only where the experts do not all fit VRAM.
#include "strata/core/auto_pick.hpp"

#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <atomic>
#include <string>

namespace strata::core {
namespace {

// A round of the adaptive tier's swaps is due every `adapt_period` tokens, and its copies land over the windows that
// follow: only a window with one of them waiting can tell the two forms apart, and only those are counted.
WindowChoice& lag_choice() {
    static WindowChoice c("xestrata adaptive tier", "ADAPT_LAG",
                          "the previous round's swaps taken by the next window", 8, 2, 1.03, "a round");
    return c;
}
WindowChoice& disjoint_choice() {
    static WindowChoice c("xestrata adaptive tier", "DISJOINT_ADAPT",
                          "a helper GPU's experts left out of the primary's candidates", 8, 3, 1.03, "a token");
    return c;
}
// The windows run on the decode loop's thread; the adaptive round that reads these may run on its own (--adapt-async,
// PlainModel's), so the decision reaches it as one word.
std::atomic<int> lag_level{0};       // 0: the engine's default; 1: the next window waits for the copies
std::atomic<bool> disjoint_on{false};
Pick disjoint_mode() {
    static const Pick p = pick_from_env("DISJOINT_ADAPT");
    return p;
}

}  // namespace

bool adapt_layers_round_alike() {
    static const bool v = [] {
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (!lay.native || lay.fmt.empty()) return false;
        for (const auto& f : lay.fmt)
            if ((f.had_gu || !strata::kernels::cpu::canon_layer(f)) && !strata::kernels::cpu::tq_supported(f.gu_type))
                return false;
        return true;
    }();
    return v;
}

int adapt_lag_forced() {
    static const int v = [] {
        const char* e = strata::env_get("ADAPT_LAG");
        if (e == nullptr || e[0] == 0 || std::strcmp(e, "auto") == 0) return 0;
        char* end = nullptr;
        const long n = std::strtol(e, &end, 10);
        if (end == e || *end != 0 || n < 0) {
            std::fprintf(stderr, "xestrata: XESTRATA_ADAPT_LAG=%s is not auto or a window count: auto\n", e);
            return 0;
        }
        return n == 0 ? 1 : (int) n;   // 0: off, the next window waits for the copies
    }();
    return v;
}

int adapt_lag_chosen() {
    if (!adapt_layers_round_alike()) return 0;   // 0: the engine's own default stays
    return lag_level.load(std::memory_order_relaxed);
}

int adapt_lag_forming() {
    if (adapt_lag_forced() > 0 || !adapt_layers_round_alike() || !lag_choice().choosing()) return 0;
    return lag_choice().on() ? 1 : 0;
}

void adapt_lag_window(double cost_ms) {
    if (adapt_lag_forced() > 0 || !adapt_layers_round_alike() || !lag_choice().choosing()) return;
    lag_choice().took(cost_ms, 1);
    lag_level.store(lag_choice().choosing() ? 0 : (lag_choice().on() ? 1 : 0), std::memory_order_relaxed);
}

bool disjoint_adapt_on() {
    const Pick p = disjoint_mode();
    if (p == Pick::Off) return false;
    if (p == Pick::On) return true;
    return disjoint_on.load(std::memory_order_relaxed);
}

void disjoint_adapt_window(double ms, int64_t tokens, bool helpers) {
    if (!helpers || disjoint_mode() != Pick::Auto || !adapt_layers_round_alike()) return;
    if (!disjoint_choice().choosing()) return;
    disjoint_choice().took(ms, tokens);
    disjoint_on.store(disjoint_choice().on(), std::memory_order_relaxed);
}

}  // namespace strata::core
