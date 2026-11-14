// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/yskn_mma.hpp - the decode's .yskn product on the int8 matrix units (yskn_mma.cpp), the parts
// strata/kernels/yskn.hpp's yskn_proj and yskn_mma_proj call.  The form (the device's report, XESTRATA_NO_XMX) and the
// row boundary (timed at the load) are chosen once a device; the sums are yskn_sum.hpp's, the DP4a form's own.
#pragma once

#include "strata/kernels/yskn.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::xe {

/// The int8 matrix form the device's report gives: 0 none (DP4a), 1 Intel Xe2's 8 x 16 x 32 on 16 lanes, 2 Intel
/// Xe-HPG's 8 x 8 x 32 on 8, 3 the 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores from sm_72, RDNA3's and RDNA4's
/// WMMA).  XESTRATA_NO_XMX=1: 0.  Once a device
int yskn_mma_shape(sycl::queue& q);

/// The matrix form's int64 partial sums of T <= kYsknMmaRows rows: part [n_workers][T][rows], as yskn_gemv's.  `shape`
/// yskn_mma_shape's (the caller has it); the activations and the finish are yskn_proj's
sycl::event yskn_mma_gemv(sycl::queue& q, const YsknDev& w, const int8_t* xq, const float* alpha, int T, int64_t* part,
                          int shape);

}  // namespace strata::kernels::xe