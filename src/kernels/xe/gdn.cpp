// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/gdn.cpp - the gated delta-net's closing-norm gate setting (gdn.hpp).  The recurrence, its causal
// convolution and the norms around it are the native kernels (native_gdn.hpp, native_gdn_preprocess.hpp, fused_gdn.hpp).
#include "strata/kernels/gdn.hpp"

#include <atomic>

namespace strata::kernels {
namespace {

std::atomic<bool> silu_gate{false};

}  // namespace

void gdn_set_silu_gate(bool silu) { silu_gate.store(silu); }
bool gdn_silu_gate() { return silu_gate.load(); }

}  // namespace strata::kernels
