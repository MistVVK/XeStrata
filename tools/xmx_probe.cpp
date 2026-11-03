// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tools/xmx_probe.cpp - what setup asks a SYCL compiler and its runtime before building the engine: does each Level
// Zero GPU report a matrix combination XeStrata's joint_matrix kernels run on (FP16 and BF16 with FP32 accumulators):
// 8 x 16 x 16 in 16-wide sub-groups (src/kernels/xe/xmx_gemm.cpp, Xe2 and later), or 8 x 8 x 16 in 8-wide ones
// (src/kernels/xe/mma_gemm.cpp's second shape: the Arc A-series, which the first test alone called "no XMX", #51)?
// The engine's tests at start (device_has_xmx, mma_gemm's `reported`), from what the device and the runtime report,
// never from the device's name or ID.
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

static bool has(const sycl::device& d, mx::matrix_type in, size_t tn, size_t sg_size) {
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), sg_size) == sg.end()) return false;
    for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
        const bool m = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
        const bool n = c.nsize == tn || (c.nsize == 0 && c.max_nsize >= tn);
        const bool k = c.ksize == 16 || (c.ksize == 0 && c.max_ksize >= 16);
        if (c.atype == in && c.btype == in && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
            m && n && k)
            return true;
    }
    return false;
}

static bool has(const sycl::device& d, mx::matrix_type in) { return has(d, in, 16, 16) || has(d, in, 8, 8); }

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
