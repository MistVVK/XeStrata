// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/matrix_report.cpp - see include/strata/kernels/matrix_report.hpp.
#include "strata/kernels/matrix_report.hpp"
#include "mma_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/platform/env.hpp"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
namespace syclex = sycl::ext::oneapi::experimental;
namespace mx = syclex::matrix;

bool env_on(const char* name) {
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}
}  // namespace

bool no_bf16_mma() {
    static const bool v = env_on("NO_BF16_MMA");
    return v;
}

bool no_xmx() {
    static const bool v = env_on("NO_XMX");
    return v;
}

std::vector<mx::combination> matrix_combinations(const sycl::device& d) {
    static const bool no_int8 = env_on("NO_INT8_MMA");
    std::vector<mx::combination> out;
    for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
        const bool bf16 = c.atype == mx::matrix_type::bf16 || c.btype == mx::matrix_type::bf16;
        const bool int8 = c.atype == mx::matrix_type::sint8 || c.atype == mx::matrix_type::uint8 ||
                          c.btype == mx::matrix_type::sint8 || c.btype == mx::matrix_type::uint8;
        if ((bf16 && no_bf16_mma()) || (int8 && no_int8)) continue;
        out.push_back(c);
    }
    return out;
}

int wmma_gfx(const sycl::device& d) {
    if (d.get_backend() != sycl::backend::ext_oneapi_hip) return 0;
    // the architecture's values carry the version: 0x0200000000110000 is gfx1100
    const auto a = static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>());
    if (a < static_cast<uint64_t>(syclex::architecture::amd_gpu_gfx1100) || a >= 0x0200000000130000ull) return 0;
    return a >= 0x0200000000120000ull ? 12 : 11;
}

bool has_form(const sycl::device& d, MatrixForm f) {
    if (no_xmx() && !xe::mma_forced()) return false;
    switch (f) {
        case MatrixForm::wmma:
        case MatrixForm::f16_from_int8:
        case MatrixForm::f16_kn: return wmma_gfx(d) != 0;
        case MatrixForm::mma_s8: {
            if (d.get_backend() != sycl::backend::ext_oneapi_cuda) return false;
            const auto a = static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>());
            return a >= static_cast<uint64_t>(syclex::architecture::nvidia_gpu_sm_80) && a < 0x0200000000000000ull;
        }
        default: break;
    }
    if (d != core::Runtime::get().compute().get_device())
        throw std::logic_error(std::string("has_form: ") + form_name(f) + " asked of a device other than the compute one");
    switch (f) {
        case MatrixForm::xmx_f16:
        case MatrixForm::f16_from_blocks: return xmx_available(XmxType::f16);
        case MatrixForm::f16_grouped: return f16_matrix_engines();
        default: break;
    }
    return false;
}

bool has_form(MatrixForm f) { return has_form(core::Runtime::get().compute().get_device(), f); }

const char* form_name(MatrixForm f) {
    switch (f) {
        case MatrixForm::xmx_f16: return "xmx_f16";
        case MatrixForm::f16_grouped: return "f16_grouped";
        case MatrixForm::f16_from_blocks: return "f16_from_blocks";
        case MatrixForm::f16_from_int8: return "f16_from_int8";
        case MatrixForm::f16_kn: return "f16_kn";
        case MatrixForm::wmma: return "wmma";
        case MatrixForm::mma_s8: return "mma_s8";
    }
    return "?";
}

}  // namespace strata::kernels
