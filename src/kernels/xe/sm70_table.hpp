// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sm70_table.hpp - upstream's switch for its Volta decode kernels (c85b7c87, PR 1401).
//
// Upstream measured expert mode 8 (iq_kernels.cpp) and the latency-hidden norm/up (fused_gr.cpp) on a V100 and keeps
// them opt-in on sm_70 with STRATA_SM70_TABLE=1 until its contributor confirms them there with the final code.
// Their own switches (STRATA_EXP_MODE, STRATA_GR_FAST) take them on any GPU.
#pragma once

#include "strata/core/per_device.hpp"

#include <sycl/sycl.hpp>

#include <cstdlib>

namespace strata::kernels::xe {

/// STRATA_SM70_TABLE=1 is set and the queue's device reports Volta (sm_70).
inline bool sm70_table(sycl::queue& q) {
    namespace syclex = sycl::ext::oneapi::experimental;
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SM70_TABLE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (!on) return false;
    static core::PerDevice<bool> volta;
    const sycl::device d = q.get_device();
    return volta.get(d, [&d] {
        return d.get_backend() == sycl::backend::ext_oneapi_cuda &&
               d.get_info<syclex::info::device::architecture>() == syclex::architecture::nvidia_gpu_sm_70;
    });
}

}  // namespace strata::kernels::xe
