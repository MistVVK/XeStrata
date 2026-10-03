// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tools/xmx_probe.cpp - what setup asks a SYCL compiler and its runtime before building the engine: does each Level
// Zero GPU report the matrix combinations src/kernels/xe/xmx_gemm.cpp needs (FP16 and BF16 8 x 16 x 16 with FP32
// accumulators, 16-wide sub-groups)?  The same test as the engine's at start (device_has_xmx), from what the device
// and the runtime report, never from the device's name or ID.
//
//     <sycl compiler> -fsycl tools/xmx_probe.cpp -o xmx_probe && ./xmx_probe
//
// One line per GPU: "pci=<domain:bus:device.function> fp16=<0|1> bf16=<0|1> name=<name>".
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <string>

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

static bool has(const sycl::device& d, mx::matrix_type in) {
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), size_t(16)) == sg.end()) return false;
    for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
        const bool m = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
        const bool n = c.nsize == 16 || (c.nsize == 0 && c.max_nsize >= 16);
        const bool k = c.ksize == 16 || (c.ksize == 0 && c.max_ksize >= 16);
        if (c.atype == in && c.btype == in && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
            m && n && k)
            return true;
    }
    return false;
}

int main() {
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) {
            std::string pci = "?";
            if (d.has(sycl::aspect::ext_intel_pci_address))
                pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
            std::printf("pci=%s fp16=%d bf16=%d name=%s\n", pci.c_str(), (int) has(d, mx::matrix_type::fp16),
                        (int) has(d, mx::matrix_type::bf16), d.get_info<sycl::info::device::name>().c_str());
        }
    }
    return 0;
}
