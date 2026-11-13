// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/window_limits.hpp - how many rows one verify / batch window can hold.
//
// A window held at most 8 rows; it now holds up to 16 (a draft window of 15 drafts, a pipelined group of more than 8
// slots), so every compile-time maximum that follows the window (the verifier's per-row arrays, the expert pool's
// per-token tables) is derived from these numbers.  Upstream chooses one of them at build time (STRATA_VERIFY_MAX_T);
// XeStrata's one binary runs every GPU, so it holds both, and whether a run uses the wide windows is chosen at run time
// (window_rows.hpp).
//
// The rule every user of the numbers follows: a window of up to kVerifyNarrowT rows runs EXACTLY the code it ran
// before (the same kernel instantiations, the same argument structures, the same buffer sizes); the wider code is
// reached only by a window of 9..16 rows.
#pragma once

namespace strata::kernels {

/// The window the 8-row kernels were built for: windows of up to this many rows keep their instantiations.
inline constexpr int kVerifyNarrowT = 8;
/// The most rows one window can hold (the capacity of every per-row array).
inline constexpr int kWindowMaxT = 16;

static_assert(kWindowMaxT == 2 * kVerifyNarrowT, "the wide kernels are built for twice the narrow window");

}  // namespace strata::kernels
