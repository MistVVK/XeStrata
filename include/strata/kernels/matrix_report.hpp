// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/matrix_report.hpp - the joint_matrix combinations the kernels choose their paths from.
//
// The device's matrix_combinations, less the input types a switch takes away, so that a GPU whose matrix engines lack
// them (NVIDIA's before sm_80 have no BF16, before sm_72 no int8) can be imitated on one that has them:
// STRATA_NO_BF16_MMA=1 drops the BF16 combinations, STRATA_NO_INT8_MMA=1 the int8 ones.  STRATA_NO_XMX=1 (no matrix
// engines at all) is read where each path is chosen.
#pragma once

#include <sycl/sycl.hpp>

#include <vector>

namespace strata::kernels {

std::vector<sycl::ext::oneapi::experimental::matrix::combination> matrix_combinations(const sycl::device& d);

/// Whether STRATA_NO_BF16_MMA=1 is set (the products' BF16 library path takes it as well).
bool no_bf16_mma();

}  // namespace strata::kernels
