// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <string>

namespace strata::core {
// GB/s, or negative on failure. The current device must be selected by the caller.
double best_copy_gbps(void* dst, const void* src, size_t bytes, std::string* samples = nullptr);
double probe_host_to_device_gbps(std::string* samples = nullptr);
}
