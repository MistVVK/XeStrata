// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/exact.hpp - which arithmetic the experts' canonical rows take.
//
// The canonical rows (canon_expert.hpp, iq_kernels.cpp's seg_dot_canon) give an expert the same bits on the CPU and
// the GPU's decode either way.  The fixed-point sums (canon_math.hpp's fx_*) do it in any order, so the prompt's
// int8 products can take them too (iq_mmq.hpp's iq_mmq_fx) and a prompt chunk the CPU shares keeps its answer however
// the share it measures moves; they cost speed (strix, 2026-11-14: the 1K prompt -7%, decode -3%).  So they are the
// way to a fixed answer, not the default where they are slower.
#pragma once

#include "strata/platform/env.hpp"

#include <cstring>

namespace strata::kernels {

/// XESTRATA_EXACT=1: the experts' canonical rows in the fixed-point sums, on the CPU, in the decode and in a prompt
/// chunk the CPU may share; 0: the float sums (the GPU's own products in the prompt); auto (unset): the float sums on
/// every GPU until the fixed-point ones are measured as fast somewhere.  Read once a process.
inline bool exact_sums() {
    static const bool on = [] {
        const char* v = strata::env_get("EXACT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

}  // namespace strata::kernels
