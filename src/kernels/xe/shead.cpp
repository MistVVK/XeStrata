// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/shead.cpp - see include/strata/kernels/shead.hpp.
#define STRATA_DEVICE_ARCH_TAG_TU xe_shead
#include "strata/kernels/shead.hpp"
#include "strata/kernels/draft_head.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace strata::kernels {

void shead_top512(const float* h, const float* bt, const int8_t* a, const float* as, const int8_t* w,
                  const float* ws, int rows, int hidden, int rank, int vocab, float* q, float* score,
                  int32_t* candidates, float* exact, int32_t* sort_tmp, int32_t* top_id, float* top_value, void* stream) {
    if (!h || !bt || !a || !as || !w || !ws || !q || !score || !candidates || !exact || !top_id || !top_value ||
        !sort_tmp || !stream || rows < 1 || hidden < 1 || rank < 1 || vocab < 512)
        throw std::invalid_argument("shead: invalid buffers or shape");
    sycl::queue& qu = core::Runtime::get().stream(stream);
    constexpr int M = 512, TOP = 16;
    const auto device = qu.get_device();
    const size_t limit = std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                  device.get_info<sycl::info::device::max_work_item_sizes<2>>()[1]);
    int wg = 1;
    while ((size_t) wg * 2 <= limit && wg < 256) wg *= 2;
    // A work-group owns one dot product. Every launch is in the caller's in-order stream and is graph-capturable.
    qu.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) rank * wg}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
        const int r = (int) it.get_group(0), k = (int) it.get_group(1), t = (int) it.get_local_id(1);
        float v = 0;
        for (int j = t; j < hidden; j += wg) v += h[(size_t) r * hidden + j] * bt[(size_t) k * hidden + j];
        v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
        if (t == 0) q[(size_t) r * rank + k] = v;
    });
    qu.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) vocab * wg}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
        const int r = (int) it.get_group(0), token = (int) it.get_group(1), t = (int) it.get_local_id(1);
        float v = 0;
        for (int j = t; j < rank; j += wg) v += q[(size_t) r * rank + j] * a[(size_t) token * rank + j];
        v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
        if (t == 0) {
            const float value = v * as[token];
            score[(size_t) r * vocab + token] = sycl::isnan(value) ? -INFINITY : value;
        }
    });
    // Share the exact vocabulary top-k with the draft head, including its ID tie rule.
    const size_t scratch_bytes = topk_scratch_bytes(vocab, M);
    for (int r = 0; r < rows; ++r)
        topk(score + (size_t) r * vocab, vocab, M, candidates + (size_t) r * M,
             reinterpret_cast<uint8_t*>(sort_tmp) + (size_t) r * scratch_bytes, stream);
    qu.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) M * wg}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
        const int r = (int) it.get_group(0), k = (int) it.get_group(1), t = (int) it.get_local_id(1);
        const int id = candidates[(size_t) r * M + k];
        float v = 0;
        for (int j = t; j < hidden; j += wg) v += h[(size_t) r * hidden + j] * w[(size_t) id * hidden + j];
        v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
        if (t == 0) {
            const float value = v * ws[id];
            exact[(size_t) r * M + k] = sycl::isnan(value) ? -INFINITY : value;
        }
    });
    for (int r = 0; r < rows; ++r)
        topk_mapped(exact + (size_t) r * M, candidates + (size_t) r * M, M, TOP, top_id + (size_t) r * TOP,
                    top_value + (size_t) r * TOP, reinterpret_cast<uint8_t*>(sort_tmp) + (size_t) r * scratch_bytes,
                    stream);
}

}  // namespace strata::kernels
