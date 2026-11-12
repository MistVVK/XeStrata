// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/image_attn.cpp - the image engine's attention on the GPU's matrix engines
// (include/strata/kernels/image_attn.hpp: qi_attention).
//
// The 16-bit attention is the kernel the plain models' prompt shares (strata/kernels/flash_attn.hpp, on the
// matrix engines); Q, K and V hold their rows rounded up to a multiple of 128 (qi_attn_rows), the rows past T zero.
// A GPU without matrix engines, or STRATA_NO_XMX=1, takes its FMA form.  The int8 attention is flash_attn's too
// (flash_attn_i8.cpp).
#include "strata/kernels/image_attn.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/env.hpp"
#include "attn_parts.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
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

using ChoiceShape = std::tuple<int64_t, int64_t, int64_t>;
using Choices = core::TimedCache<sycl::device, ChoiceShape, bool>;
Choices& choices_of() { static Choices choices; return choices; }
Choices& sdpa_choices_of() { static Choices choices; return choices; }
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
    Choices& ch = choices_of();
    const auto key = std::make_tuple(rows, keys, heads);
    if (const auto found = ch.get(dev, key)) return *found;
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
        use = ch.get(dev, key).value_or(false);
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
                  const std::vector<std::pair<int64_t, int64_t>>* spans, const kernels::FlashAttnI8Stats* i8_stats,
                  int64_t keys) {
    if (T <= first) return;
    if (head_dim != D) throw std::invalid_argument("qi_attention: head_dim 128");
    if (keys <= 0) keys = T;   // the rows' own keys (a self-attention), else a cross-attention's
    // the image rows (they see every key) through the int8 kernel where it is the faster, else the GPU maker's kernel
    // where there is one; the prefix's own rows, a few dozen, by the kernel below
    int64_t row_end = T;
    const int64_t r0 = std::max(first, n_text);
    // the int8 kernel: STRATA_IMAGE_ATTN_I8=1 always, 0 never; else the image rows timed in it and in the 16-bit kernel
    // that would take them (the GPU maker's, else the engine's own), once a GPU and a shape (rows, keys and heads),
    // from 2048 image rows on (fewer: not taken). A generation's and an edit's shapes differ.  The int8 kernel reads
    // the rows as their own keys, so a cross-attention's rows never take it.
    bool i8_done = false;
    if (int8 && keys == T && T - r0 >= 256) {
        sycl::queue& qq = core::Runtime::get().stream(stream);
        const int forced = forced_i8();
        Choices& ch = choices_of();
        const auto key = std::make_tuple(T - r0, T, heads);
        auto run_i8 = [&] {
            return kernels::flash_attn_i8(q16, k16, v16, T, heads, r0, first, out16, stream, i8_stats);
        };
        int use = forced;
        if (use < 0) {
            const auto found = ch.get(qq.get_device(), key);
            if (found) use = *found;
            else if (T - r0 >= 2048 && kernels::flash_attn_i8_kind(stream) != 0 &&
                     qq.ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::recording) {
                auto run_f16 = [&] {
                    if (qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim,
                                out16 + (r0 - first) * heads * head_dim, stream)) return true;
                    struct Guard { Guard() { g_no_i8 = true; } ~Guard() { g_no_i8 = false; } } guard;
                    qi_attention(q16, k16, v16, T, heads, head_dim, n_text, out16, stream, key_lim, first);
                    return true;
                };
                const auto measured = core::timed_choice({{run_i8}, {run_f16}}, [&] { qq.wait_and_throw(); });
                const double i8ms = measured.seconds[0] * 1000, f16ms = measured.seconds[1] * 1000;
                use = measured.pick == 0;
                use = ch.put(qq.get_device(), key, use != 0);
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
    // the first attention of a shape, timed both ways: the maker's unless the engine's own is clearly
    // the faster (5%: a near tie taken one way or the other from start to start would change the image with it)
    auto maker = [&] {
        if (g_no_sdpa) return false;
        sycl::queue& qq = core::Runtime::get().stream(stream);
        Choices& ch = sdpa_choices_of();
        const auto key = std::make_tuple(T - r0, keys, heads);
        if (const auto found = ch.get(qq.get_device(), key)) return *found;
        if (qq.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) return true;
        auto run_maker = [&] {
            return qi_sdpa(q16, k16, v16, r0, T, keys, heads, head_dim,
                           out16 + (r0 - first) * heads * head_dim, stream);
        };
        auto run_own = [&] {
            struct Guard { Guard() { g_no_sdpa = true; } ~Guard() { g_no_sdpa = false; } } guard;
            qi_attention(q16, k16, v16, T, heads, head_dim, n_text, out16, stream, key_lim, first, false, nullptr,
                         nullptr, keys);
            return true;
        };
        const auto measured = core::timed_choice({{run_maker}, {run_own}}, [&] { qq.wait_and_throw(); });
        const double mk = measured.seconds[0] * 1000, own = measured.seconds[1] * 1000;
        const bool use = ch.put(qq.get_device(), key, measured.pick == 0);
        if (mk >= 0.0)
            std::fprintf(stderr,
                         "xestrata-image: the attention's image rows by %s at %lld rows of %lld keys (the maker's %.2f "
                         "ms, the engine's own %.2f ms)\n",
                         use ? "the maker's kernel" : "the engine's own", (long long) (T - r0), (long long) keys, mk, own);
        return use;
    };
    // the image rows went through the int8 kernel or else the maker's 16-bit one; the text rows are left
    if (i8_done || (T - r0 >= 256 && maker() && qi_sdpa(q16, k16, v16, r0, T, keys, heads, head_dim,
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
    // rows f .. e - 1 by the shared kernel (flash_attn.hpp), the dense layout; `fma`: its FMA form (no matrix engines)
    auto run = [&](int64_t f, int64_t e, bool fma = false) {
        kernels::FlashAttnArgs a;
        a.q = q16;
        a.q_ld = heads * D;
        a.k = k16;
        a.v = v16;
        a.kv_ld = heads * D;
        a.n_head = a.n_kv = heads;
        a.head_dim = D;
        a.rows = T;
        a.keys = keys;
        a.n_causal = n_text;
        a.key_lim = key_lim;
        a.first = f;
        a.row_end = e;
        a.out = out16 + (f - first) * heads * head_dim;
        a.out_ld = heads * D;
        a.scale = 1.0f / std::sqrt((float) D);
        a.fma = fma;
        return kernels::flash_attn_tuned(a, stream);
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
    bool done = most > 0;
    for (const auto& [f, end] : segs)
        if (done) done = run(f, end);
    if (!done && !run(first, row_end, true))
        throw std::invalid_argument("qi_attention: no form of the attention takes the shape");
    if (!stream) q.wait();
}

// A cross-attention (Anima's blocks): the rows over another count of keys, every row seeing every key.  The same
// entry as the image rows', so the maker's kernel and the engine's own are chosen and timed as they are; no int8
// (its kernel reads the rows as their own keys) and no prefix rows.
void qi_attention_cross(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t Tq, int64_t Tk,
                        int64_t heads, int64_t head_dim, uint16_t* out16, void* stream) {
    qi_attention(q16, k16, v16, Tq, heads, head_dim, 0, out16, stream, nullptr, 0, false, nullptr, nullptr, Tk);
}

}  // namespace strata::image
