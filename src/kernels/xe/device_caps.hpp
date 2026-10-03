// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/device_caps.hpp - what a kernel may ask of the device it runs on.
//
// Kernels written for the B70 launch another instantiation where the device reports less:
// - work-groups of 1024 work-items (the B70 takes 1024, the UHD 770 512);
// - FP64 (the B70 has it; the UHD 770 and the Arc A series do not): a kernel that names `double` anywhere is refused
//   whole on such a device, so the kernels that sum in double take their accumulator type as a parameter, FP32 there.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace strata::kernels::xe {

/// The device's largest work-group, in work-items.
inline size_t max_work_group(const sycl::queue& q) {
    return q.get_device().get_info<sycl::info::device::max_work_group_size>();
}

/// The largest of 1024, 512 and 256 work-items that `q`'s device takes (256 below that: every Intel GPU takes it).
inline int work_group_upto_1024(const sycl::queue& q) {
    const size_t m = max_work_group(q);
    return m >= 1024 ? 1024 : m >= 512 ? 512 : 256;
}

/// Whether `q`'s device runs FP64 kernels.
inline bool has_fp64(const sycl::queue& q) { return q.get_device().has(sycl::aspect::fp64); }

}  // namespace strata::kernels::xe
