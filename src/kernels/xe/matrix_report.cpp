// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/matrix_report.cpp - see include/strata/kernels/matrix_report.hpp.
#include "strata/kernels/matrix_report.hpp"

#include <cstdlib>

namespace strata::kernels {
namespace {
namespace syclex = sycl::ext::oneapi::experimental;
namespace mx = syclex::matrix;

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}
}  // namespace

bool no_bf16_mma() {
    static const bool v = env_on("STRATA_NO_BF16_MMA");
    return v;
}

std::vector<mx::combination> matrix_combinations(const sycl::device& d) {
    static const bool no_int8 = env_on("STRATA_NO_INT8_MMA");
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

}  // namespace strata::kernels
