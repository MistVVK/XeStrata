// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/core/latent_share.hpp - the session's latent D/U pair on the device, per device: the driver
// uploads it once (generate.cpp's LatentTab) and every window and the prompt path on that device pull it in their
// own init, so the several verifiers of a serve session share one copy.  A layer split gives each stage's device
// only its own layers' pair (layer0 the first of them, the readers index from it); a single-GPU run holds every
// layer's on device 0.  A non-latent pack never stores a pair: the accessors stay null.
// Written after edge0-strata's latent_share.hpp.
#pragma once

#include "strata/core/on_device.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {

struct LatentShared {
    const uint8_t* d = nullptr;
    const uint8_t* u = nullptr;
    int fmt = 8;
    int64_t layer0 = 0;   ///< the first layer this device's pair holds (a stage's first; 0 for a whole-model pair)
    uint64_t bytes = 0;   ///< the pair's device bytes (D and U together, over this device's layers)
};

inline std::vector<LatentShared>& latent_shared_table() { static std::vector<LatentShared> t; return t; }

/// The pair uploaded on `device` (null d when the pack is not latent or this device holds none).
inline const LatentShared& latent_shared_on(int device) {
    static const LatentShared none;
    const auto& t = latent_shared_table();
    return device >= 0 && (size_t) device < t.size() ? t[(size_t) device] : none;
}

/// The current device's pair: the window and the prompt path call this in their init, inside the device's scope.
inline const LatentShared& latent_shared() { return latent_shared_on(current_device()); }

/// The pair's device bytes on `device` (0 where there is none): the cache sizing leaves room for it.
inline uint64_t latent_device_bytes(int device) { return latent_shared_on(device).bytes; }

}  // namespace strata::core
