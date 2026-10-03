// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// Multi-GPU: make `device` current for a scope.  The Xe runtime drives exactly one device (docs/XE.md), so device 0 and
// a negative device are the current one and do nothing; any other device is refused where the object that owns it is
// created (a layer split across GPUs is not supported).

#include <stdexcept>

namespace strata::core {

struct OnDevice {
    explicit OnDevice(int device) {
        if (device > 0) throw std::runtime_error("OnDevice: the Xe engine drives one device; device " +
                                                 std::to_string(device) + " is not available");
    }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core
