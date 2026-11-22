// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Four-query selected attention: compare the union's IDs and masks exactly, and every output against FP64.
// The portable FP16 matrix path needs a 16x16x16 / subgroup-32 device; Intel's XMX path does not take it.
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "xe/mma_gemm.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace par = strata::parity;

int main() {
    auto& queue = strata::core::Runtime::get().compute();
    // batch_gqa4's local allocation: mma_local_bytes() plus the 16 masks (see flash_attn.cpp).
    constexpr size_t local_bytes = 34432;
    if (k::xmx_available(k::XmxType::f16) || !k::xe::mma_shape_is(queue, 16, 16, 16, 32) ||
        queue.get_device().get_info<sycl::info::device::local_mem_size>() < local_bytes) {
        std::printf("SKIP: four-query attention needs the portable 16x16x16/subgroup-32 FP16 path\n");
        return 77;
    }
    constexpr int64_t nq = 7, cap = 300, cells = 512, groups = (nq + 3) / 4, ucap = 4 * cap;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = 16; s.n_head_kv = 4; s.head_dim = 256; s.page_size = 4;
    const int64_t nh = s.n_head, nkv = s.n_head_kv, hd = s.head_dim, ps = s.page_size;
    const int64_t pages = cells / ps, rows = pages * nkv * ps;
    std::mt19937 rng(917);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the parity inputs reproducible
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kh((size_t) (rows * hd)), vh(kh.size());
    for (auto& x : kh) x = k::f16_from_f32(nd(rng) * 1.5f);
    for (auto& x : vh) x = k::f16_from_f32(nd(rng));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<float> q((size_t) (nq * nh * hd));
    for (auto& x : q) x = nd(rng) * 2.f;

    // Sorted, overlapping and distinct selections, including a 300-entry list (scan crosses 256 lanes),
    // one-cell selection, partial chunks and a final group with only three queries.
    std::vector<std::vector<int32_t>> lists((size_t) nq);
    for (int c = 0; c < cells; ++c) {
        if (c < 300) lists[0].push_back(c);
        if (c % 2 == 0) lists[1].push_back(c);
        if (c % 3 == 0) lists[2].push_back(c);
        if (c >= 150 && c < 450) lists[3].push_back(c);
        if (c == 511) lists[4].push_back(c);
        if (c >= 80 && c < 111) lists[5].push_back(c);
        if (c >= 96 && c < 353) lists[6].push_back(c);
    }
    std::vector<int32_t> ids((size_t) (nq * cap), -1), steps((size_t) (nq * k::kStepCount), 0);
    for (int64_t i = 0; i < nq; ++i) {
        std::copy(lists[(size_t) i].begin(), lists[(size_t) i].end(), ids.begin() + i * cap);
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) lists[(size_t) i].size();
    }
    par::Device dev;
    k::QsaAttnPools pools;
    pools.k_pool = dev.upload(kh); pools.v_pool = dev.upload(vh); pools.page_table = dev.upload(table);
    const float* dq = dev.upload(q);
    const int32_t* di = dev.upload(ids);
    const int32_t* ds = dev.upload(steps);
    float* out = dev.alloc<float>(q.size());
    const size_t scratch_bytes = (size_t) (groups * ucap * 5 + groups * 4);
    uint8_t* scratch = dev.alloc<uint8_t>(scratch_bytes);
    if (k::flash_attn_select(dq, pools, di, ds, cap, s, out, nq, nullptr, scratch, scratch_bytes - 1) ||
        !k::flash_attn_select(dq, pools, di, ds, cap, s, out, nq, nullptr, scratch, scratch_bytes)) {
        std::fprintf(stderr, "FAIL: scratch size or supported device rejected\n");
        return 1;
    }
    std::vector<int32_t> uids((size_t) (groups * ucap)), counts((size_t) groups);
    std::vector<uint8_t> masks((size_t) (groups * ucap));
    par::copy_out(uids.data(), reinterpret_cast<int32_t*>(scratch), uids.size());
    par::copy_out(counts.data(), reinterpret_cast<int32_t*>(scratch) + groups * ucap, counts.size());
    par::copy_out(masks.data(), scratch + groups * ucap * 4 + groups * 4, masks.size());
    int bad = 0;
    for (int64_t g = 0; g < groups; ++g) {
        int64_t at = 0;
        for (int c = 0; c < cells; ++c) {
            uint8_t bits = 0;
            for (int j = 0; j < 4 && g * 4 + j < nq; ++j) {
                const auto& list = lists[(size_t) (g * 4 + j)];
                if (std::binary_search(list.begin(), list.end(), c)) bits |= (uint8_t) (1u << j);
            }
            if (!bits) continue;
            const size_t index = (size_t) (g * ucap + at++);
            if (uids[index] != c || masks[index] != bits) ++bad;
        }
        if (counts[(size_t) g] != at) ++bad;
    }
    const int union_bad = bad;
    std::vector<float> got(q.size());
    par::copy_out(got.data(), out, got.size());
    double err = 0, scale = 0;
    for (int64_t i = 0; i < nq; ++i) for (int64_t h = 0; h < nh; ++h) {
        const auto& list = lists[(size_t) i];
        std::vector<double> scores(list.size());
        double mx = -std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < list.size(); ++c) {
            const int64_t cell = list[c], row = ((int64_t) table[(size_t) (cell / ps)] * nkv + h / 4) * ps + cell % ps;
            double dot = 0;
            for (int64_t d = 0; d < hd; ++d)
                dot += (double) q[(size_t) ((i * nh + h) * hd + d)] * k::f32_from_f16(kh[(size_t) (row * hd + d)]);
            scores[c] = dot / std::sqrt((double) hd);
            mx = std::max(mx, scores[c]);
        }
        double sum = 0;
        for (double& v : scores) { v = std::exp(v - mx); sum += v; }
        for (int64_t d = 0; d < hd; ++d) {
            double ref = 0;
            for (size_t c = 0; c < list.size(); ++c) {
                const int64_t cell = list[c], row = ((int64_t) table[(size_t) (cell / ps)] * nkv + h / 4) * ps + cell % ps;
                ref += scores[c] * k::f32_from_f16(vh[(size_t) (row * hd + d)]);
            }
            ref /= sum;
            const float actual = got[(size_t) ((i * nh + h) * hd + d)];
            if (!std::isfinite(actual)) ++bad;
            else err = std::max(err, std::fabs((double) actual - ref));
            scale = std::max(scale, std::fabs(ref));
        }
    }
    if (err > 1e-4 * scale) ++bad;
    std::printf("qsa_gqa4_attn: union failures %d; FP64 error %.3e, scale %.3e; total failures %d (%s)\n",
                union_bad, err, scale, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
