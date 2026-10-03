// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/remote_experts.cpp - the helper-GPU expert tier (--expert-cache-device1..3), on the Xe runtime.
//
// The CUDA engine could place a static expert tier on CUDA1..3 next to the main GPU.  The Xe runtime drives exactly
// one GPU (docs/XE.md, Runtime contract), so there is no second device to hold a tier: preflight and open refuse with
// the reason, and nothing else is reached.  The CUDA implementation is src/core/remote_experts.cpp at a4f0edb.
#include "strata/core/remote_experts.hpp"

namespace strata::core {
namespace {

std::string unavailable(int device) {
    return "helper-GPU expert cache on device " + std::to_string(device) +
           ": the Xe engine drives one GPU; a second device is not supported";
}

}  // namespace

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    free_gib = 0;
    err = unavailable(device);
    return false;
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
