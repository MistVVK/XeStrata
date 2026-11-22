// sycl/include/strata/core/device_lost.hpp - #6445: detect and diagnose Level Zero UR_RESULT_ERROR_DEVICE_LOST
// (typically a Linux xe CCS preemption timeout when root GPU monitors such as `sudo xpu-smi` submit render jobs
// against the 640 ms default preempt_timeout_us).  Host-only, no SYCL headers required (tested by device_lost_test).
#pragma once

#include <atomic>
#include <string>

namespace strata {

inline std::atomic<bool>& device_lost_flag() {
    static std::atomic<bool> lost{false};
    return lost;
}

inline bool device_is_lost() {
    return device_lost_flag().load(std::memory_order_acquire);
}

inline void mark_device_lost() {
    device_lost_flag().store(true, std::memory_order_release);
}

/// True when a SYCL / Level Zero exception text indicates a lost GPU context (UR_RESULT_ERROR_DEVICE_LOST / error 20).
inline bool is_device_lost(const std::string& what) {
    if (what.find("DEVICE_LOST") != std::string::npos) return true;
    if (what.find("level_zero") != std::string::npos && what.find("error: 20") != std::string::npos) return true;
    return false;
}

/// Actionable hint for Linux xe CCS preemption resets (#6445).
inline const char* device_lost_hint() {
    return "On Linux with the xe driver, UR_RESULT_ERROR_DEVICE_LOST is commonly caused by a CCS compute preemption "
           "timeout (preempt_timeout_us, default 640 ms) when a root GPU monitor such as 'sudo xpu-smi' submits a "
           "render job while Strata is running. Run xpu-smi unprivileged (or disable root GPU polling during "
           "generation), and optionally raise "
           "/sys/class/drm/cardN/device/tile0/gt0/engines/ccs/preempt_timeout_us (see docs/INTEL.md).";
}

}  // namespace strata
