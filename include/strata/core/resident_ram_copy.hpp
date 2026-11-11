#pragma once
// The resident RAM copy of the experts: how STRATA_RESIDENT_PIN is read, the mode the automatic RAM copy takes when the
// expert arena does not fit, and the line printed when a RAM budget is cut to the available memory.  Header-only, no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace strata::core {

/// STRATA_RESIDENT_PIN: "0" keeps the RAM copy pageable (ordinary memory, not locked); unset or any other value keeps it
/// page-locked, the default.  One reading for --resident-experts, --resident-budget-gib and the automatic RAM copy.
inline bool resident_pin_from_env(const char* value) {
    return !(value != nullptr && std::strcmp(value, "0") == 0);
}

/// The automatic RAM copy (the engine's own fallback when the expert arena does not fit RAM): the --resident-experts form
/// with a budget (the hottest experts in RAM, the rest read from the files), page-locked unless STRATA_RESIDENT_PIN=0,
/// read the same way as for the flags.  `pin_env`: the variable's value, nullptr when unset.
struct AutoRamCopyMode { bool resident_cpu_experts; bool pin; bool soft; };
inline AutoRamCopyMode auto_ram_copy_mode(const char* pin_env) {
    return AutoRamCopyMode{true, resident_pin_from_env(pin_env), true};
}

/// Where a RAM budget came from, for the line printed when it is cut to the available memory.
enum class ResidentBudgetFrom {
    Flag,        ///< --resident-budget-gib N
    WhatFits,    ///< the soft mode's second try (#467): sized from the available memory itself
    Automatic,   ///< the automatic RAM copy: no flag was given
};

/// The line printed when a budget is cut (`budget` > `clamped`).  `limit` is "RAM" or "commit" (Windows commit capacity),
/// `available` the memory read for it, `headroom` what stays free; 0.25 GiB more is kept as the #403 margin.
inline std::string resident_clamp_line(ResidentBudgetFrom from, uint64_t budget, const char* limit, uint64_t available,
                                       uint64_t headroom, uint64_t clamped) {
    const double gib = 1073741824.0;
    char line[512];
    switch (from) {
        case ResidentBudgetFrom::WhatFits:
            std::snprintf(line, sizeof line,
                          "FileExpertSource: room for the complement: %.2f GiB (%.2f GiB available %s "
                          "minus %.0f GiB headroom and a 0.25 GiB margin)\n",
                          (double) clamped / gib, (double) available / gib, limit, (double) headroom / gib);
            break;
        case ResidentBudgetFrom::Automatic:
            // the owner's wording (11-11).  It must not name --resident-budget-gib: on this path no such flag was given.
            std::snprintf(line, sizeof line,
                          "strata: the automatic RAM copy of %.2f GiB does not fit the available %s "
                          "(%.2f GiB minus %.0f GiB headroom and a 0.25 GiB margin); using %.2f GiB\n",
                          (double) budget / gib, limit, (double) available / gib, (double) headroom / gib,
                          (double) clamped / gib);
            break;
        case ResidentBudgetFrom::Flag:
        default:
            std::snprintf(line, sizeof line,
                          "FileExpertSource: --resident-budget-gib %.2f exceeds available %s "
                          "(%.2f GiB minus %.0f GiB headroom and a 0.25 GiB margin): %.2f GiB\n",
                          (double) budget / gib, limit, (double) available / gib, (double) headroom / gib,
                          (double) clamped / gib);
            break;
    }
    return line;
}

} // namespace strata::core
