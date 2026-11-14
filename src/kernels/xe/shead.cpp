// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/shead.cpp - see include/strata/kernels/shead.hpp.
#define STRATA_DEVICE_ARCH_TAG_TU xe_shead
#include "strata/kernels/shead.hpp"

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
    if (device.get_info<sycl::info::device::local_mem_size>() < 1024 * (sizeof(float) + sizeof(int32_t)))
        throw std::runtime_error("shead: the device's local memory is too small for a sort tile");
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
    // Sort tiles, retain their best 512, then merge pairs of sorted lists. Exact top-k: every discarded token has
    // 512 tokens ahead of it in its own tile, so it cannot enter the global shortlist.
    constexpr int TILE = 1024;
    const int chunks = (vocab - 1) / TILE + 1;
    int32_t* src = sort_tmp;
    int32_t* dst = sort_tmp + (size_t) rows * chunks * M;
    qu.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> values(TILE, hd);
        sycl::local_accessor<int32_t, 1> ids(TILE, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) chunks * wg}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
            const int row = (int) it.get_group(0), chunk = (int) it.get_group(1), t = (int) it.get_local_id(1);
            auto group = it.get_group();
            for (int i = t; i < TILE; i += wg) {
                const int id = chunk * TILE + i;
                ids[i] = id < vocab ? id : INT_MAX;
                values[i] = id < vocab ? score[(size_t) row * vocab + id] : -INFINITY;
            }
            sycl::group_barrier(group);
            for (int k = 2; k <= TILE; k *= 2) {
                for (int stride = k / 2; stride > 0; stride /= 2) {
                    for (int i = t; i < TILE; i += wg) {
                        const int j = i ^ stride;
                        if (j <= i) continue;
                        const bool better_j = values[j] > values[i] || (values[j] == values[i] && ids[j] < ids[i]);
                        const bool better_i = values[i] > values[j] || (values[i] == values[j] && ids[i] < ids[j]);
                        if (((i & k) == 0 && better_j) || ((i & k) != 0 && better_i)) {
                            const float v = values[i]; values[i] = values[j]; values[j] = v;
                            const int32_t id = ids[i]; ids[i] = ids[j]; ids[j] = id;
                        }
                    }
                    sycl::group_barrier(group);
                }
            }
            for (int i = t; i < M; i += wg) src[((size_t) row * chunks + chunk) * M + i] = ids[i];
        });
    });
    int lists = chunks;
    while (lists > 1) {
        const int next = (lists + 1) / 2;
        qu.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) next * 2 * M}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
            const int row = (int) it.get_group(0), j = (int) it.get_global_id(1);
            const int pair = j / (2 * M), item = j % (2 * M), side = item / M, index = item % M;
            const int left = pair * 2, right = left + 1;
            const int32_t* in = src + (size_t) row * lists * M;
            int32_t* out = dst + ((size_t) row * next + pair) * M;
            if (right >= lists) { if (side == 0) out[index] = in[(size_t) left * M + index]; return; }
            const int32_t id = in[(size_t) (left + side) * M + index];
            if (id == INT_MAX) return;
            const float value = score[(size_t) row * vocab + id];
            const int32_t* other = in + (size_t) (left + 1 - side) * M;
            int lo = 0, hi = M;
            while (lo < hi) {
                const int mid = (lo + hi) / 2, oid = other[mid];
                const float ov = oid == INT_MAX ? -INFINITY : score[(size_t) row * vocab + oid];
                if (ov > value || (ov == value && oid < id)) lo = mid + 1; else hi = mid;
            }
            const int pos = index + lo;
            if (pos < M) out[pos] = id;
        });
        lists = next;
        int32_t* swap = src; src = dst; dst = swap;
    }
    qu.parallel_for(sycl::range<2>((size_t) rows, M), [=](sycl::id<2> i) {
        candidates[i[0] * M + i[1]] = src[i[0] * M + i[1]];
    });
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
    qu.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) wg}, {1, (size_t) wg}), [=](sycl::nd_item<2> it) {
        const int r = (int) it.get_group(0), t = (int) it.get_local_id(1);
        auto group = it.get_group();
        float* row = exact + (size_t) r * M;
        const int32_t* ids = candidates + (size_t) r * M;
        for (int k = 0; k < TOP; ++k) {
            float best = -INFINITY;
            for (int j = t; j < M; j += wg) best = sycl::fmax(best, row[j]);
            best = sycl::reduce_over_group(group, best, sycl::maximum<float>());
            int id = INT_MAX;
            for (int j = t; j < M; j += wg) if (!sycl::isnan(row[j]) && row[j] == best) id = sycl::min(id, (int) ids[j]);
            id = sycl::reduce_over_group(group, id, sycl::minimum<int>());
            if (t == 0) { top_id[(size_t) r * TOP + k] = id; top_value[(size_t) r * TOP + k] = best; }
            for (int j = t; j < M; j += wg) if (ids[j] == id) row[j] = sycl::nan(0u);
            sycl::group_barrier(group);
        }
    });
}

}  // namespace strata::kernels
