// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/window_rows.hpp - whether this run's verify and batch windows may be wider than 8 rows.
//
// A window of 9..16 rows (kernels/window_limits.hpp) is a speed choice: it reads the weights once for more rows, but
// its own kernels are other instances and a draft past the eighth is accepted less often, so it is used only where it
// pays.  The run's setting (--wide-window, the serve configuration's "wide_window", XESTRATA_WIDE_WINDOW) says how:
//   auto (the default): the limit is 16, and each user of a window decides whether to go past 8 (a draft window by
//        timing its arms, a batch group only when its slots do not fit in 8 rows);
//   off: no window holds more than 8 rows, as before;
//   on:  the users take 16 rows wherever they can.
#pragma once

#include "strata/kernels/window_limits.hpp"
#include "strata/platform/env.hpp"

#include <atomic>
#include <cstring>

namespace strata::core {

enum class WideWindow : int { Auto = -1, Off = 0, On = 1 };

namespace detail {
inline std::atomic<int>& wide_window_value() {
    static std::atomic<int> v{strata::env_auto("WIDE_WINDOW")};   // XESTRATA_WIDE_WINDOW=auto|0|1
    return v;
}
}  // namespace detail

/// The run's setting (XESTRATA_WIDE_WINDOW until set_wide_window).
inline WideWindow wide_window() { return (WideWindow) detail::wide_window_value().load(std::memory_order_relaxed); }
/// Sets it (the command line's --wide-window, before any window is sized).
inline void set_wide_window(WideWindow w) { detail::wide_window_value().store((int) w, std::memory_order_relaxed); }
/// "auto", "off" / "0", "on" / "1" -> the setting; false for anything else.
inline bool parse_wide_window(const char* s, WideWindow& out) {
    if (s == nullptr) return false;
    if (std::strcmp(s, "auto") == 0) { out = WideWindow::Auto; return true; }
    if (std::strcmp(s, "off") == 0 || std::strcmp(s, "0") == 0) { out = WideWindow::Off; return true; }
    if (std::strcmp(s, "on") == 0 || std::strcmp(s, "1") == 0) { out = WideWindow::On; return true; }
    return false;
}
/// The most rows a window may hold in this run: 8 when the wide windows are off, else 16.
inline int window_rows_cap() {
    return wide_window() == WideWindow::Off ? strata::kernels::kVerifyNarrowT : strata::kernels::kWindowMaxT;
}
/// The rows an engine sizes its windows for (its buffers and graphs, before it loads): 16 when the wide windows are on,
/// or under auto when a user of the windows will take more than 8 rows (`wants_wide`: a drafter that measures wide
/// windows, a batch group of more than 8 slots); else the 8 a window always had.
inline int window_rows_for(bool wants_wide) {
    const WideWindow w = wide_window();
    return w == WideWindow::On || (w == WideWindow::Auto && wants_wide) ? window_rows_cap()
                                                                        : strata::kernels::kVerifyNarrowT;
}

}  // namespace strata::core
