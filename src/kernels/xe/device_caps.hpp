// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/device_caps.hpp - what a kernel may ask of the device it runs on.
//
// Kernels written for the B70 launch another instantiation where the device reports less:
// - work-groups of 1024 work-items (the B70 takes 1024, the UHD 770 512, a kernel with many registers on CUDA less);
// - FP64 (the B70 has it; the UHD 770 and the Arc A series do not): a kernel that names `double` anywhere is refused
//   whole on such a device, so the kernels that sum in double take their accumulator type as a parameter, FP32 there.
#pragma once

#include "strata/core/per_device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

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

/// The largest work-group the kernel named `Name` takes on `q`'s device, asked of the kernel itself
/// (kernel_device_specific::work_group_size) once a device: on CUDA its registers bound it below the device's (#35:
/// sm_90's top-k at 67 registers a work-item, #39: a P40's at 86; 1024 work-items of them pass a block's 65,536).
/// The device's own largest where the kernel cannot be asked.
template<class Name>
size_t kernel_work_group(const sycl::queue& q) {
    static core::PerDevice<size_t> per_device;
    return per_device.get(q.get_device(), [&q] {
        try {
            const sycl::kernel_id id = sycl::get_kernel_id<Name>();
            const auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()},
                                                                                   {id});
            return kb.get_kernel(id).template get_info<sycl::info::kernel_device_specific::work_group_size>(
                q.get_device());
        } catch (const sycl::exception&) {
            return max_work_group(q);
        }
    });
}

/// work_group_upto_1024 for the kernel `K<that many work-items>` too (kernel_work_group): 512 where the device takes
/// 1024 and its 1024-item kernel does not, and so on.
template<template<int> class K>
int work_group_upto_1024(const sycl::queue& q) {
    const int wg = work_group_upto_1024(q);
    if (wg == 1024 && kernel_work_group<K<1024>>(q) >= 1024) return 1024;
    if (wg >= 512 && kernel_work_group<K<512>>(q) >= 512) return 512;
    return 256;
}

/// work_group_upto_1024 for the one kernel `Name`, which takes its work-group's size at run time.
template<class Name>
int work_group_upto_1024_of(const sycl::queue& q) {
    const size_t m = std::min(max_work_group(q), kernel_work_group<Name>(q));
    return m >= 1024 ? 1024 : m >= 512 ? 512 : 256;
}

/// Whether the device runs sub-groups of `width` work-items (its sub_group_sizes, asked once a device).
inline bool has_sub_group(const sycl::device& d, size_t width) {
    static core::PerDevice<std::vector<size_t>> per_device;
    const auto& s = per_device.get(d, [&d] { return d.get_info<sycl::info::device::sub_group_sizes>(); });
    return std::find(s.begin(), s.end(), width) != s.end();
}

/// Whether `q`'s device runs FP64 kernels.
inline bool has_fp64(const sycl::queue& q) { return q.get_device().has(sycl::aspect::fp64); }

}  // namespace strata::kernels::xe
