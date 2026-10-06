// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// Multi-GPU: make `device` current for a scope, as cudaSetDevice does.  The current device is per thread; the
// engine's GPU calls (strata::gpu, Runtime::get) act on it.  A negative device keeps the current one.

namespace strata::core {

/// This thread's current engine device (0 until a scope chooses another).
int current_device();
/// Makes `device` (0 .. Runtime::count() - 1) this thread's current device; refuses any other.
void set_current_device(int device);
/// Puts back a device that was current before (unchecked: it was valid then).
void restore_current_device(int device) noexcept;

struct OnDevice {
    explicit OnDevice(int device) : previous_(current_device()) {
        if (device >= 0) set_current_device(device);
    }
    ~OnDevice() { restore_current_device(previous_); }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;

private:
    int previous_;
};

}  // namespace strata::core
