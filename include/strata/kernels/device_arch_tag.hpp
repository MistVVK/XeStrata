// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// include/strata/kernels/device_arch_tag.hpp - every source with device code includes this: its device code for each
// architecture then carries "xestrata-device-arch:<the architecture it was compiled for>:<the source>" (gfx1201,
// sm_890, spir64), which tools/package/check_device_code.py compares with the name the image is filed under.  The name comes from the
// link and stays right when the link is handed another architecture's device code (issue #7: clang-offload-bundler
// gave each architecture's link the first unclaimed entry of the triple), so only a mark made at compile time shows
// that.  The kernel is never launched; it keeps the string in the device code.
#pragma once

#include <sycl/sycl.hpp>

// The source names itself before the include (#define STRATA_DEVICE_ARCH_TAG_TU xe_sampler): the device link joins
// every source's code before it splits it into images, and a mark with the same name in two sources is kept once.
#ifndef STRATA_DEVICE_ARCH_TAG_TU
#error "define XESTRATA_DEVICE_ARCH_TAG_TU (the source's name) before including strata/kernels/device_arch_tag.hpp"
#endif

#define STRATA_DEVICE_ARCH_TAG_STR2(x) #x
#define STRATA_DEVICE_ARCH_TAG_STR(x) STRATA_DEVICE_ARCH_TAG_STR2(x)
#define STRATA_DEVICE_ARCH_TAG_CAT2(a, b) a##b
#define STRATA_DEVICE_ARCH_TAG_CAT(a, b) STRATA_DEVICE_ARCH_TAG_CAT2(a, b)

#if defined(__SYCL_DEVICE_ONLY__) && defined(__amdgcn_processor__)
#define STRATA_DEVICE_ARCH_TAG_ARCH __amdgcn_processor__
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__)
#define STRATA_DEVICE_ARCH_TAG_ARCH "sm_" STRATA_DEVICE_ARCH_TAG_STR(__SYCL_CUDA_ARCH__)
#else
#define STRATA_DEVICE_ARCH_TAG_ARCH "spir64"
#endif

namespace strata::device_arch_tag {

static constexpr char STRATA_DEVICE_ARCH_TAG_CAT(tag_, STRATA_DEVICE_ARCH_TAG_TU)[] =
    "xestrata-device-arch:" STRATA_DEVICE_ARCH_TAG_ARCH ":" STRATA_DEVICE_ARCH_TAG_STR(STRATA_DEVICE_ARCH_TAG_TU);

[[maybe_unused]] static void STRATA_DEVICE_ARCH_TAG_CAT(kernel_, STRATA_DEVICE_ARCH_TAG_TU)(sycl::queue& q, char* out,
                                                                                         int n) {
    // indexed by the work-item, so the string stays an array in the image instead of becoming immediate stores
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        constexpr auto& tag = STRATA_DEVICE_ARCH_TAG_CAT(tag_, STRATA_DEVICE_ARCH_TAG_TU);
        out[i[0]] = tag[i[0] % sizeof(tag)];
    });
}

}  // namespace strata::device_arch_tag
