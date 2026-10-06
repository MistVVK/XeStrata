// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// A value worked out once per GPU: a kernel's choice from what the device reports, or a buffer of its own.  With a
// layer split the engine drives GPUs of different makers in one process, so a choice cached for the first one would
// be wrong for the next.

#include <sycl/sycl.hpp>

#include <mutex>
#include <unordered_map>

namespace strata::core {

template <class T>
class PerDevice {
public:
    /// The value for `device`, made by `make()` the first time it is asked for.
    template <class F>
    const T& get(const sycl::device& device, F&& make) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = values_.find(device);
        if (it == values_.end()) it = values_.emplace(device, make()).first;
        return it->second;
    }

private:
    std::mutex mutex_;
    std::unordered_map<sycl::device, T> values_;
};

}  // namespace strata::core
