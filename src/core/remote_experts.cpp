// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/remote_experts.cpp - the helper-GPU expert tier (--expert-cache-device1..3), on the Xe runtime.
//
// The CUDA engine could place a static expert tier on CUDA1..3 next to the main GPU.  The tier is not carried: its
// per-layer round trip made decode slower on a card behind a slow link (upstream bench/results/2026-09-29-layer-split),
// and a B70 and a 4070 have no peer path.  open refuses with the reason, and nothing else is reached.  preflight is
// kept, as a layer split's later stages call it to open their GPU first.  The CUDA implementation is
// src/core/remote_experts.cpp at a4f0edb.
#include "strata/core/remote_experts.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

namespace strata::core {
namespace {

std::string unavailable(int device) {
    return "helper-GPU expert cache on device " + std::to_string(device) + ": not supported by the Xe engine";
}

}  // namespace

RemoteExperts::~RemoteExperts() { close(); }

// Opens the device's runtime (its context and queues) and reads its free memory.
bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    free_gib = 0;
    try {
        const OnDevice on(device);
        (void) Runtime::get();
        size_t free_bytes = 0, total = 0;
        if (!gpu::mem_info(&free_bytes, &total)) {
            err = "GPU " + std::to_string(device) + ": " + gpu::last_error();
            return false;
        }
        free_gib = (double) free_bytes / 1073741824.0;
        return true;
    } catch (const std::exception& e) {
        err = "GPU " + std::to_string(device) + ": " + e.what();
        return false;
    }
}

void RemoteExperts::close() { device_ = -1; }

bool RemoteExperts::open(int device, int, int64_t, int64_t, const std::vector<std::pair<int32_t, int32_t>>&,
                         const ExpertCache&, ExpertSource&, std::vector<uint8_t>&, std::string& err) {
    err = unavailable(device);
    return false;
}

bool RemoteExperts::begin(int64_t, const float*, const int32_t*, int64_t, int64_t, const int32_t*, const int32_t*,
                          std::string& err) {
    err = "helper-GPU expert cache: not open";
    return false;
}

bool RemoteExperts::finish(float*, std::string& err) {
    err = "helper-GPU expert cache: not open";
    return false;
}

}  // namespace strata::core
