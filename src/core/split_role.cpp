// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/split_role.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace strata::core {

double probe_helper_layer_ms(double& submit_ms) {
    constexpr size_t kMeta = (size_t) 2 << 10, kBytes = (size_t) 64 << 10;
    submit_ms = -1.0;
    void* h = nullptr;
    void* d = nullptr;
    strata::gpu::Stream s = strata::gpu::stream_create();
    if (s == nullptr) return -1.0;
    if (!strata::gpu::alloc_host(&h, kMeta) || !strata::gpu::alloc_device(&d, kBytes)) {
        if (h != nullptr) strata::gpu::free(h);
        strata::gpu::stream_destroy(s);
        return -1.0;
    }
    std::memset(h, 0, kMeta);
    std::vector<double> sub, all;
    for (int i = 0; i < 17; ++i) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        bool ok = strata::gpu::copy_async(d, h, kMeta, s);
        for (int c = 0; c < 3 && ok; ++c) ok = strata::gpu::memset_async((char*) d + kMeta, c, kBytes - kMeta, s);
        if (!ok) break;
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        strata::gpu::Status st = strata::gpu::Status::Pending;
        while ((st = strata::gpu::stream_status(s)) == strata::gpu::Status::Pending) std::this_thread::yield();
        if (st == strata::gpu::Status::Failed) break;
        if (i > 0) {
            sub.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            all.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    strata::gpu::stream_sync(s);
    strata::gpu::free(d);
    strata::gpu::free(h);
    strata::gpu::stream_destroy(s);
    if (all.empty()) return -1.0;
    std::sort(sub.begin(), sub.end());
    std::sort(all.begin(), all.end());
    submit_ms = sub[sub.size() / 2];
    return all[all.size() / 2];
}

HelperCachePrediction predict_helper_cache(const HelperCacheInputs& in) {
    HelperCachePrediction out;
    const int64_t L = in.layers;
    std::vector<double> held0_l((size_t) L, 0.0), heldh_l((size_t) L, 0.0), layer_mass((size_t) L, 0.0);
    for (size_t r = 0; r < in.profile.size(); ++r)
        layer_mass[(size_t) in.profile[r].first] += in.mass.empty() ? 1.0 : in.mass[r];
    // the first GPU's cache takes the hottest pairs until it is full, then the helper's until it is
    int64_t u0 = 0, uh = 0;
    bool full0 = false;
    for (size_t r = 0; r < in.profile.size(); ++r) {
        const int64_t l = in.profile[r].first;
        const double m = in.mass.empty() ? 1.0 : in.mass[r];
        const int64_t c = in.cost(l);
        if (!full0) {
            if (u0 + c <= in.cap0) {
                u0 += c;
                held0_l[(size_t) l] += m;
                continue;
            }
            full0 = true;
        }
        if (uh + c > in.caph) break;
        uh += c;
        heldh_l[(size_t) l] += m;
        ++out.held;
    }
    double ms = 0, kept = 0, all = 0;
    for (int64_t l = 0; l < L; ++l) {
        const double lm = layer_mass[(size_t) l];
        const double hit0 = lm > 0 ? held0_l[(size_t) l] / lm : 1.0;
        const double hith = lm > 0 ? heldh_l[(size_t) l] / lm : 0.0;
        const double experts = in.experts_per_row * in.rows * in.blob_bytes(l);
        const double gpu = experts * hit0 / in.bw0;
        const double cpu = experts * std::max(0.0, 1.0 - hit0 - hith) * in.tail_keep / in.cpu_bw * in.link_scale0;
        const double help = hith > 0 ? experts * hith / in.bwh + in.rt : 0.0;
        ms += in.dense_layer / in.bw0 + (hith > 0 ? in.submit_ms : 0.0) + std::max({gpu, cpu, help});
        kept += held0_l[(size_t) l] + heldh_l[(size_t) l];
        all += lm;
    }
    ms += (in.head_bytes + (in.drafts ? (in.rows - 1.0) * in.dense_layer : 0.0)) / in.bw0;
    out.ms = ms;
    out.kept_mass = all > 0 ? kept / all : 0.0;
    return out;
}

SplitRoleChoice choose_split_role(const SplitRoleInputs& in) {
    SplitRoleChoice out;
    double chosen_ms = in.roles_helper ? 1e30 : in.split_ms;
    SplitRole chosen = SplitRole::Split;   // the helper takes it below; --split-roles helper leaves no split to keep
    if (in.skip_if_fits && in.alone_holds_all && in.role.empty()) {
        out.role = SplitRole::SkipIfFits;
        out.ms = -1;
        return out;
    }
    if (in.helper_eligible && in.role != "split") {
        // the helper only when its window is predicted faster by more than the model's error: the model prices decode
        // only, and the split reads a prompt on both GPUs, the helper every layer on the first GPU
        const bool faster = in.roles_helper || in.helper_ms < in.split_ms * (1.0 - in.helper_gain);
        const bool helper = in.helper_held > 0 && (in.role == "helper" || (in.role.empty() && faster));
        out.helper_chosen = helper;
        if (helper) {
            chosen_ms = in.helper_ms;
            chosen = SplitRole::Helper;
        }
    }
    // the first GPU alone, last: the second GPU only when its role is predicted shorter by the margin
    if (chosen_ms >= 0 && in.role.empty() && !(chosen_ms < in.alone_ms * (1.0 - in.gain))) {
        out.role = SplitRole::Alone;
        out.ms = chosen_ms;
        return out;
    }
    out.role = chosen;
    out.ms = chosen_ms;
    return out;
}

}  // namespace strata::core
