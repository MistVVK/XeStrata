// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// include/strata/kernels/onednn.hpp - oneDNN's verbose, silenced once before the first use unless the user asked
// for it.  oneDNN enables its error level by default (no ONEDNN_VERBOSE set), and an error prints its header too:
// on a machine whose current GPU is another maker's, its SYCL device enumeration errors once ("gpu devices
// queried but not found", oneDNN 3.14 src/xpu/sycl/utils.cpp, get_device_index) and the lines land on the
// engine's stdout, which is serve's line channel.  Every oneDNN entry (the int8 products, the SDPA, the image
// convolutions) calls this before it makes its engine.  dnnl_set_verbose marks the setting initialized, so
// oneDNN's own lazy reading of the environment does not override it (src/common/verbose.cpp, setting_t::set);
// a user's ONEDNN_VERBOSE (or the legacy DNNL_VERBOSE) is left to oneDNN.

#include <oneapi/dnnl/dnnl.h>

#include <cstdlib>

namespace strata::kernels {

inline void onednn_quiet() {
    static const bool done = [] {
        if (std::getenv("ONEDNN_VERBOSE") == nullptr && std::getenv("DNNL_VERBOSE") == nullptr)
            dnnl_set_verbose(0);
        return true;
    }();
    (void) done;
}

}  // namespace strata::kernels
