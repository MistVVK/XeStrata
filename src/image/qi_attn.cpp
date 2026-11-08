// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn.cpp - the DiT's attention on the GPU's matrix engines (qi_kernels.hpp: qi_attention).
//
// The 16-bit attention is the kernel the plain models' prompt shares (strata/kernels/flash_attn.hpp, on the
// matrix engines); Q, K and V hold their rows rounded up to a multiple of 128 (qi_attn_rows), the rows past T zero.
// A GPU without matrix engines, or STRATA_NO_XMX=1, takes its FMA form.  The int8 attention is flash_attn's too
// (flash_attn_i8.cpp).
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/env.hpp"
#include "../kernels/xe/attn_parts.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace strata::image {

namespace {

using kernels::xe::attn::built;   // the attention kernels' shared parts (attn_parts.hpp)

constexpr int D = 128;      // head_dim

}  // namespace

const char* qi_attention_path() {
    static const char* names[] = {"matrix engines 8x16x16", "matrix engines 8x8x16", "matrix engines 16x16x16",
                                  "WMMA 16x16x16"};
    const int p = kernels::flash_attn_path(&core::Runtime::get().device());
    return p < 0 ? "plain (FP32)" : names[p];
}

namespace {
thread_local bool g_no_i8 = false;   // the 16-bit path's run while the int8 one is timed against it
thread_local bool g_no_sdpa = false;   // the engine's own kernel's run while the maker's is timed against it

// qi_attention's choices of the int8 kernel, a GPU's: (image rows, keys) / 1024 -> int8
struct Choices {
    std::mutex mu;
    std::map<std::pair<int64_t, int64_t>, bool> of;
};
Choices& choices_of(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<Choices>> of;
    return *of.get(q.get_device(), [] { return std::make_shared<Choices>(); });
}
Choices& sdpa_choices_of(sycl::queue& q) {   // the maker's kernel (true) or the engine's own, as choices_of
    static core::PerDevice<std::shared_ptr<Choices>> of;
    return *of.get(q.get_device(), [] { return std::make_shared<Choices>(); });
}
int forced_i8() {   // STRATA_IMAGE_ATTN_I8: 1 always, 0 never, else -1 (timed)
    static const int forced = [] {
        const char* v = strata::env_get("IMAGE_ATTN_I8");
        return v == nullptr || *v == 0 ? -1 : (int) (std::strtol(v, nullptr, 10) != 0);
    }();
    return forced;
}
}  // namespace

// the int8 path is flash_attn's (flash_attn_i8.cpp), every model's
bool qi_attention_i8(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                     int64_t r0, int64_t first, uint16_t* out16, void* stream) {
    return kernels::flash_attn_i8(q16, k16, v16, T, heads, r0, first, out16, stream);
}

int qi_attention_i8_chunk() { return kernels::flash_attn_i8_chunk(); }

uint64_t qi_attention_i8_prefix_bytes(int64_t P, int64_t heads) { return kernels::flash_attn_i8_prefix_bytes(P, heads); }

bool qi_attention_i8_prefix(const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* prefix,
                            void* stream) {
    return kernels::flash_attn_i8_prefix(k16, v16, P, heads, prefix, stream);
}

bool qi_attention_i8_prefixed(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                              int64_t heads, const void* prefix, int64_t P, const float* v_from, const float* v_to,
                              uint16_t* out16, void* stream) {
    return kernels::flash_attn_i8_prefixed(q16, k16, v16, T, heads, prefix, P, v_from, v_to, out16, stream);
}

bool qi_attention_i8_preferred(int64_t rows, int64_t keys, int64_t heads, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    const sycl::device dev = q.get_device();
    if (kernels::flash_attn_i8_kind(stream) == 0 || kernels::flash_attn_path(&dev) < 0) return false;
    if (forced_i8() >= 0) return forced_i8() == 1;
    rows = std::max<int64_t>(rows, 2048);   // qi_attention times from 2048 rows on
    keys = std::max(keys, rows);
    Choices& ch = choices_of(q);
    const auto key = std::make_pair((rows + 1023) / 1024, (keys + 1023) / 1024);
    {
        std::lock_guard<std::mutex> lock(ch.mu);
        if (const auto it = ch.of.find(key); it != ch.of.end()) return it->second;
    }
    // not timed yet: qi_attention times it, on rows of zeros (the kernels' speed does not ask what the values are)
    const int64_t C = heads * D, TP = qi_attn_rows(keys);
    uint16_t *qkv = nullptr, *out = nullptr;
    if (!strata::gpu::alloc_device(&qkv, (size_t) (3 * TP * C) * 2) ||
        !strata::gpu::alloc_device(&out, (size_t) (rows * C) * 2)) {
        if (qkv != nullptr) strata::gpu::free(qkv);
        return false;
    }
    bool use = false;
    try {
        strata::gpu::memset_async(qkv, 0, (size_t) (3 * TP * C) * 2, stream);
        qi_attention(qkv, qkv + TP * C, qkv + 2 * TP * C, keys, heads, D, 0, out, stream, nullptr,
                     keys - rows, true);
        q.wait();
        std::lock_guard<std::mutex> lock(ch.mu);
        const auto it = ch.of.find(key);
        use = it != ch.of.end() && it->second;
    } catch (...) {
        strata::gpu::free(qkv);
        strata::gpu::free(out);
        throw;
    }
    strata::gpu::free(qkv);
    strata::gpu::free(out);
    return use;
}

void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, void* stream,
                  const int32_t* key_lim, int64_t first, bool int8,
                  const std::vector<std::pair<int64_t, int64_t>>* spans, const kernels::FlashAttnI8Stats* i8_stats) {
    if (T <= first) return;
    if (head_dim != D) throw std::invalid_argument("qi_attention: head_dim 128");
    // the image rows (they see every key) through the int8 kernel where it is the faster, else the GPU maker's kernel
    // where there is one; the prefix's own rows, a few dozen, by the kernel below
    int64_t row_end = T;
    const int64_t r0 = std::max(first, n_text);
    // the int8 kernel: STRATA_IMAGE_ATTN_I8=1 always, 0 never; else the image rows timed in it and in the 16-bit kernel
    // that would take them (the GPU maker's, else the engine's own), once a GPU and a shape (the rows and the keys, in
    // steps of 1024: a generation's and an edit's differ), from 2048 image rows on (fewer: not taken)
    bool i8_done = false;
    if (int8 && T - r0 >= 256) {
        sycl::queue& qq = core::Runtime::get().stream(stream);
        const int forced = forced_i8();
        Choices& ch = choices_of(qq);
        const auto key = std::make_pair((T - r0 + 1023) / 1024, (T + 1023) / 1024);
        auto run_i8 = [&] {
            return kernels::flash_attn_i8(q16, k16, v16, T, heads, r0, first, out16, stream, i8_stats);
        };
        int use = forced;
        if (use < 0) {
            std::lock_guard<std::mutex> lock(ch.mu);
            const auto it = ch.of.find(key);
            if (it != ch.of.end()) use = it->second;
            else if (T - r0 >= 2048 && kernels::flash_attn_i8_kind(stream) != 0) {
                auto timed = [&](const std::function<bool()>& f) {
                    if (!f()) return -1.0;
                    qq.wait();
                    const auto t0 = std::chrono::steady_clock::now();
                    f();
                    qq.wait();
                    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                };
                const double i8ms = timed(run_i8);
                double f16ms = timed([&] {   // the maker's kernel on the image rows, else the engine's on all asked
                    return qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim, out16 + (r0 - first) * heads * head_dim,
                                   stream);
                });
                if (f16ms < 0.0) {
                    g_no_i8 = true;
                    f16ms = timed([&] {
                        qi_attention(q16, k16, v16, T, heads, head_dim, n_text, out16, stream, key_lim, first);
                        return true;
                    });
                    g_no_i8 = false;
                }
                // int8 unless it is clearly the slower: a near tie (the RX 9060 XT, Qwen-Image's 4096 rows, 2026-10-29:
                // 6.58 against 6.66 ms) picked one or the other from start to start, and with it the image
                use = i8ms > 0.0 && i8ms < f16ms * 1.05;
                ch.of[key] = use;
                std::fprintf(stderr, "xestrata-image: the attention's image rows %s at %lld rows of %lld keys (int8 "
                                     "%.2f ms, 16-bit %.2f ms)\n", use ? "in int8" : "in 16 bits", (long long) (T - r0),
                             (long long) T, i8ms, f16ms);
            } else {
                use = 0;
            }
        }
        if (use == 1 && !g_no_i8) i8_done = run_i8();
    }
    // the maker's 16-bit kernel (qi_sdpa) or the engine's own, once a GPU and a shape (as the int8 kernel's choice) on
    // the first attention of 2048 image rows or more, timed both ways: the maker's unless the engine's own is clearly
    // the faster (5%: a near tie taken one way or the other from start to start would change the image with it)
    auto maker = [&] {
        if (g_no_sdpa) return false;
        if (T - r0 < 2048) return true;
        sycl::queue& qq = core::Runtime::get().stream(stream);
        Choices& ch = sdpa_choices_of(qq);
        const auto key = std::make_pair((T - r0 + 1023) / 1024, (T + 1023) / 1024);
        {
            std::lock_guard<std::mutex> lock(ch.mu);
            const auto it = ch.of.find(key);
            if (it != ch.of.end()) return it->second;
        }
        auto timed = [&](const std::function<bool()>& f) {
            if (!f()) return -1.0;
            qq.wait();
            const auto t0 = std::chrono::steady_clock::now();
            f();
            qq.wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        const double mk = timed([&] {
            return qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim, out16 + (r0 - first) * heads * head_dim, stream);
        });
        double own = -1.0;
        if (mk >= 0.0) {
            g_no_sdpa = true;
            own = timed([&] {
                qi_attention(q16, k16, v16, T, heads, head_dim, n_text, out16, stream, key_lim, first);
                return true;
            });
            g_no_sdpa = false;
        }
        const bool use = mk >= 0.0 && !(own >= 0.0 && own < mk * 0.95);
        std::lock_guard<std::mutex> lock(ch.mu);
        ch.of[key] = use;
        if (mk >= 0.0)
            std::fprintf(stderr,
                         "xestrata-image: the attention's image rows by %s at %lld rows of %lld keys (the maker's %.2f "
                         "ms, the engine's own %.2f ms)\n",
                         use ? "the maker's kernel" : "the engine's own", (long long) (T - r0), (long long) T, mk, own);
        return use;
    };
    // the image rows went through the int8 kernel or else the maker's 16-bit one; the text rows are left
    if (i8_done || (T - r0 >= 256 && maker() && qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim,
                                                        out16 + (r0 - first) * heads * head_dim, stream))) {
        if (first >= n_text) {
            if (!stream) core::Runtime::get().stream(stream).wait();
            return;
        }
        row_end = n_text;
    }
    sycl::queue& q = core::Runtime::get().stream(stream);
    const sycl::device dev = q.get_device();
    const int p = kernels::flash_attn_path(&dev);
    // The sub-groups a work-group: more share the keys' tiles in the caches, fewer leave the GPU freer to overlap
    // work-groups.  The fastest of the most that fit, a half and a quarter of it, timed once a GPU on the first
    // attention of 2048 rows or more (the images' own inputs; a small first image would not show it): 1024 x 1024
    // block, the B70 4 (11.88 -> 11.44 ms), the RTX 4070 2 (11.55 -> 10.53 ms).  Until then, the most.
    static core::PerDevice<std::shared_ptr<std::atomic<int>>> tuned_of;
    std::atomic<int>& tuned = *tuned_of.get(q.get_device(), [] { return std::make_shared<std::atomic<int>>(0); });
    // rows f .. e - 1 by the shared kernel (flash_attn.hpp), the dense layout; `fma`: its FMA form (no matrix engines)
    auto run = [&](int nsg, int64_t f, int64_t e, bool fma = false) {
        kernels::FlashAttnArgs a;
        a.q = q16;
        a.q_ld = heads * D;
        a.k = k16;
        a.v = v16;
        a.kv_ld = heads * D;
        a.n_head = a.n_kv = heads;
        a.head_dim = D;
        a.rows = a.keys = T;
        a.n_causal = n_text;
        a.key_lim = key_lim;
        a.first = f;
        a.row_end = e;
        a.out = out16 + (f - first) * heads * head_dim;
        a.out_ld = heads * D;
        a.scale = 1.0f / std::sqrt((float) D);
        a.fma = fma;
        return kernels::flash_attn(a, stream, nsg);
    };
    // The references' rows (spans) as image rows past a context of b keys, through the image rows' kernels (the int8
    // one or the GPU maker's, timed as the image rows' are; an edit's 4096 rows of a reference, 4104 keys: 22.8 ms a
    // block in the kernel below, 6.3 in AMD's on the RX 9060 XT); the rows between them, the text's, by the kernel below
    std::vector<std::pair<int64_t, int64_t>> segs;   // the rows left to the kernel below
    {
        int64_t f = first;
        if (p >= 0 && spans != nullptr)
            for (const auto& [a, b] : *spans) {
                if (a < f || b > std::min(row_end, n_text)) continue;   // a span asked only in part: the kernel below
                qi_attention(q16, k16, v16, b, heads, head_dim, a, out16 + (a - first) * heads * head_dim,
                             stream, nullptr, a, int8);
                if (a > f) segs.emplace_back(f, a);
                f = b;
            }
        if (row_end > f) segs.emplace_back(f, row_end);
    }
    const int most = p >= 0 ? kernels::flash_attn_nsg_most(&dev, D) : 0;
    int nsg = tuned.load();
    const auto big = std::max_element(segs.begin(), segs.end(), [](const auto& x, const auto& y) {
        return x.second - x.first < y.second - y.first;
    });
    if (most > 0 && nsg == 0 && big != segs.end() && big->second - big->first >= 2048) {
        q.wait();
        double best = 1e30;
        std::string said;
        for (int n = most; n >= std::max(1, most / 4); n /= 2) {
            run(n, big->first, big->second);   // its kernel built, its caches warm
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            run(n, big->first, big->second);
            q.wait();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            said += (said.empty() ? "" : ", ") + std::to_string(n) + ": " + std::to_string(ms).substr(0, 5) + " ms";
            if (ms < best) { best = ms; nsg = n; }
        }
        tuned.store(nsg);
        std::fprintf(stderr, "xestrata-image: the attention takes %d sub-groups a work-group (timed at %lld rows of %lld "
                             "keys: %s)\n", nsg, (long long) (big->second - big->first), (long long) T, said.c_str());
    }
    if (nsg == 0) nsg = most;
    bool done = most > 0;
    for (const auto& [f, end] : segs)
        if (done) done = run(nsg, f, end);
    if (!done && !run(0, first, row_end, true))
        throw std::invalid_argument("qi_attention: no form of the attention takes the shape");
    if (!stream) q.wait();
}

}  // namespace strata::image
