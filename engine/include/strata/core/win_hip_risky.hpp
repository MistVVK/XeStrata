// include/strata/core/win_hip_risky.hpp - two opt-in settings that are refused on Windows with an AMD card (HIP) unless STRATA_WIN_HIP_RISKY=1 is set.
//
//   --adapt-async 1          the asynchronous adaptive tier: on an RX 7900 XTX (Windows, HIP) long generations ended with a verify timeout and a display-driver
//                            reset (Windows event 4101) every 1 to 3 long answers; without the option the same engine and config ran clean at the same speed.
//   STRATA_ADAPT_OVERLAP=1   the adaptive tier's copies land while the next window runs: on a gfx1030 card (Windows, HIP) the first request ended with a GPU
//                            fault and a display-driver timeout; it was confirmed on CUDA only.
// On every other platform (Linux HIP, CUDA, SYCL, Metal) nothing changes. A pure function: no GPU, no OS call.  The caller says what it is running on.
#pragma once

#include <cstring>
#include <string>

namespace strata::core {

enum class WinHipRisky { Allow, Refuse, AllowWithWarning };

/// STRATA_WIN_HIP_RISKY as the process sees it: on when it is set to anything but "" and "0".
inline bool win_hip_risky_override(const char* value) { return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0; }

/// What to do with a risky setting the user asked for.  `asked`: the setting was asked for (--adapt-async 1, STRATA_ADAPT_OVERLAP=1).  `windows`, `hip`: the engine is a
/// Windows build with the HIP backend.  `override_value`: STRATA_WIN_HIP_RISKY.
inline WinHipRisky win_hip_risky_decide(bool asked, bool windows, bool hip, const char* override_value) {
    if (!asked || !windows || !hip) return WinHipRisky::Allow;
    return win_hip_risky_override(override_value) ? WinHipRisky::AllowWithWarning : WinHipRisky::Refuse;
}

/// The start-up line for a refused setting (`setting`: "--adapt-async 1" or "STRATA_ADAPT_OVERLAP=1").
inline std::string win_hip_refused_line(const char* setting) {
    const bool async_tier = std::strcmp(setting, "--adapt-async 1") == 0;
    return std::string("strata: ") + setting + " is off on Windows with an AMD card: " +
           (async_tier ? "it reset the display driver there during long answers."
                       : "it ended the first request with a GPU fault there, and it was only checked on NVIDIA cards.") +
           " STRATA_WIN_HIP_RISKY=1 turns it on anyway.";
}

/// The start-up line when the override is set and the setting is on as asked.
inline std::string win_hip_warning_line(const char* setting) {
    return std::string("strata: STRATA_WIN_HIP_RISKY=1: ") + setting +
           " is on as asked, on Windows with an AMD card; it has reset the display driver or ended requests with a GPU fault there.";
}

}  // namespace strata::core
