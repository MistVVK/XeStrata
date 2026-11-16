// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/core/latent_share.hpp - the session's latent D/U pair on the device, per device: the driver
// uploads it once (generate.cpp's LatentTab) and every window and the prompt path on that device pull it in their
// own init, so the several verifiers of a serve session share one copy.  A layer split gives each stage's device
// only its own layers' pair (layer0 the first of them, the readers index from it); a single-GPU run holds every
// layer's on device 0, or, on a card whose expert cache the whole pair would leave too small, some layers' in pinned
// host memory, which each reader copies into a ring of its own ahead of the layer.  A non-latent pack never stores a
// pair: the accessors stay null.
// Written after edge0-strata's latent_share.hpp.
#pragma once

#include "strata/core/on_device.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {

struct LatentShared {
    const uint8_t* d = nullptr;   ///< the resident layers' D, one after another (null: none resident)
    const uint8_t* u = nullptr;
    int fmt = 8;
    int64_t layer0 = 0;   ///< the first layer this device's pair holds (a stage's first; 0 for a whole-model pair)
    uint64_t bytes = 0;   ///< the pair's device bytes (the resident layers' D and U)
    uint64_t d_bytes = 0, u_bytes = 0;   ///< one layer's D and U
    // Streamed layers (a card whose expert cache the whole pair would leave short of what it needs): their D then U
    // in pinned host memory, copied into a reader's ring a few layers ahead of their use.  `at[l - layer0]` is the
    // layer's place: >= 0 among the resident layers, -1 - i the i-th streamed one; empty, every layer resident.
    const uint8_t* host = nullptr;
    std::vector<int32_t> at;
    int ring = 0;   ///< the slots a reader's ring holds (2; 0: nothing streamed)

    bool on() const { return d != nullptr || host != nullptr; }
    int32_t place(int64_t layer) const {
        return at.empty() ? (int32_t) (layer - layer0) : at[(size_t) (layer - layer0)];
    }
    bool streamed(int64_t layer) const { return place(layer) < 0; }
    /// the streamed layer's index among the streamed ones (its ring slot is this modulo the ring)
    int32_t stream_index(int64_t layer) const { return -1 - place(layer); }
    const uint8_t* res_d(int64_t layer) const { return d + (size_t) place(layer) * d_bytes; }
    const uint8_t* res_u(int64_t layer) const { return u + (size_t) place(layer) * u_bytes; }
    /// a streamed layer's D then U in pinned host memory
    const uint8_t* host_du(int64_t layer) const { return host + (size_t) stream_index(layer) * (d_bytes + u_bytes); }
    /// the next streamed layer after `layer` (-1: none), and the one `ahead` places after it among the streamed
    int64_t streamed_after(int64_t layer, int ahead = 1) const {
        for (int64_t l = layer + 1; l - layer0 < (int64_t) at.size(); ++l)
            if (streamed(l) && --ahead == 0) return l;
        return -1;
    }
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

/// The prompt path's ring for the streamed layers on the current device: two layers' D and U (one read while the
/// next lands), 0 where nothing is streamed.  In Prefill::bytes_needed and its carve.
inline uint64_t latent_prefill_ring_bytes() {
    const LatentShared& sh = latent_shared();
    return sh.ring > 0 ? 2 * (sh.d_bytes + sh.u_bytes) : 0;
}

}  // namespace strata::core
