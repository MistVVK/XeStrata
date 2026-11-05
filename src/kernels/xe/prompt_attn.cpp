// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "attn_parts.hpp"
#include "strata/core/per_device.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace strata::kernels {
namespace {

// the pools' formats (prompt_attn_tune's pool_format), the attention kernels' (attn_parts.hpp)
using xe::attn::kFmtF16;
using xe::attn::kFmtI8;
using xe::attn::kFmtQ4;
using xe::attn::kFmtK8V4;
using xe::attn::kFmtNames;
using xe::attn::pool_format;

// the shared attention for the pools (below)
bool flash_pools(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
                 const QsaShapes& s, float* attn);

}  // namespace

bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0) return false;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    static const bool no_mma = [] {   // STRATA_PROMPT_ATTN_MMA=0: the FMA kernel on every GPU
        const char* v = std::getenv("STRATA_PROMPT_ATTN_MMA");
        return v != nullptr && std::strtol(v, nullptr, 10) == 0;
    }();
    // RDNA3 and RDNA4: the shared attention's WMMA form (flash_attn.hpp), which these GPUs' FMA form is far behind;
    // elsewhere its FMA form (FP32 FMAs, K and V from FP16)
    QsaAttnPools pools;
    pools.k_pool = k_pool;
    pools.v_pool = v_pool;
    pools.page_table = page_table;
    if (!no_mma && xe::amd_wmma(queue.get_device())) {
        if (!flash_pools(queue, q, pools, kFmtF16, pos0, T, s, attn)) return false;
    } else {
        FlashAttnArgs a;
        a.q32 = q;
        a.q_ld = a.out_ld = s.n_head * s.head_dim;
        a.k = k_pool;
        a.v = v_pool;
        a.page_table = page_table;
        a.page_size = s.page_size;
        a.n_head = s.n_head;
        a.n_kv = s.n_head_kv;
        a.head_dim = s.head_dim;
        a.rows = a.n_causal = a.row_end = T;
        a.keys = pos0 + T;
        a.pos0 = pos0;
        a.out = attn;
        a.out_f32 = true;
        a.scale = 1.0f / std::sqrt((float) s.head_dim);
        a.fma = true;
        if (!flash_attn(a, &queue)) return false;
    }
    if (stream == nullptr) queue.wait_and_throw();
    return true;
}

namespace {

// The kernel a prompt's attention takes, for a GPU, a shape and a pool format (prompt_attn_tune): the matrix
// engines' (JmKernel, or WmmaKernel's INT8 form on RDNA3 and RDNA4), the FP16 pools' FMA or WMMA kernel, or none: the
// caller's decode kernel (qsa_dense_attn_batch).  Their order changes with the GPU, the pools and the context: over a
// 2,048-query chunk after 28K cells of Qwen3.6-35B-A3B's 16/2 heads of 256 they ran 222, 256 and 216 ms on a B70 with
// FP16 pools (237, -, 272 with INT8 ones), 217, 166 and 397 ms on an RTX 4070 (222, -, 391); llm-jp-4.1's 40/4 heads
// of 128 187 and 1,078 ms on the B70 (226 and 194 on the RTX 4070).  The decode kernel reads every cell again for
// each query: Qwen3.5-9B's 16/4 heads of 256 took it at 16K cells on the B70 and its 64K prompt then ran 86.1 s against
// the FP16 kernel's 83.5.  So the choice is timed at a short and a long context and taken by the request's length.
enum Pick : int { kPickUntimed = -1, kPickNone = 0, kPickMma = 1, kPickFp16 = 2 };
const char* const kPickNames[3] = {"the decode kernel", "the matrix engines'", "the FP16 kernel"};
constexpr double kDecodeMargin = 0.9;   // the decode kernel's dummy time must be under this share of the others' (below)
struct Picks {
    int short_pick = kPickUntimed, long_pick = kPickUntimed;
    int64_t split = 0;   // a chunk reaching this many cells takes long_pick
};
using PickKey = std::tuple<int64_t, int64_t, int64_t, int>;   // head_dim, query heads, KV heads, the pools' format
std::mutex picks_mu;
std::map<PickKey, std::unique_ptr<core::PerDevice<Picks>>> picks;

core::PerDevice<Picks>& picks_for(const QsaShapes& s, int fmt) {
    std::lock_guard<std::mutex> lock(picks_mu);
    auto& p = picks[PickKey{s.head_dim, s.n_head, s.n_head_kv, fmt}];
    if (!p) p = std::make_unique<core::PerDevice<Picks>>();
    return *p;
}

bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}

// The attention the image engine shares (flash_attn.hpp): FP16 pools in pages of 64 cells (or a multiple) read where
// they are, the other formats and page sizes gathered into local memory as FP16 a chunk at a time; the output kept in
// the matrix engines' accumulators.  STRATA_PROMPT_ATTN_FLASH=0: the kernels above (A/B)
bool flash_pools(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
                 const QsaShapes& s, float* attn) {
    static const bool off = env_off("STRATA_PROMPT_ATTN_FLASH");
    if (off) return false;
    FlashAttnArgs a;
    a.q32 = q;
    a.q_ld = s.n_head * s.head_dim;
    if (fmt == kFmtF16 && s.page_size % 64 == 0) {
        a.k = pools.k_pool;
        a.v = pools.v_pool;
        a.page_table = pools.page_table;
    } else {
        a.pools = pools;
        a.pools_format = fmt;
    }
    a.page_size = s.page_size;
    a.n_head = s.n_head;
    a.n_kv = s.n_head_kv;
    a.head_dim = s.head_dim;
    a.rows = a.n_causal = T;
    a.keys = pos0 + T;
    a.pos0 = pos0;
    a.first = 0;
    a.row_end = T;
    a.out = attn;
    a.out_ld = s.n_head * s.head_dim;
    a.out_f32 = true;
    a.scale = 1.0f / std::sqrt((float) s.head_dim);
    return flash_attn(a, &queue);
}

// the matrix engines' kernel for the pools, launched (false: none takes them on this GPU)
bool run_mma(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
             const QsaShapes& s, float* attn) {
    return flash_pools(queue, q, pools, fmt, pos0, T, s, attn);
}

// The fastest of the kernels over a dummy chunk: up to 2,048 queries after the rest of `cells` cells, zero pools
// behind an identity table; one line to stderr.  kPickUntimed when the dummy could not be made (its VRAM).  As many
// queries as a prompt's chunk keeps the GPU as busy as the prompt does: with 512 the FMA kernel's blocks of a few
// queries left it idle and the decode kernel (which also splits the cells) came out 29% ahead at 64K cells on a B70,
// where the prompt ran it 3.5% behind
int time_pick(sycl::queue& queue, const QsaShapes& s, int fmt, int64_t cells, void* stream) {
    const int64_t PS = s.page_size, nkv = s.n_head_kv, nh = s.n_head, hd = s.head_dim;
    const int64_t T = std::min<int64_t>(2048, std::max<int64_t>(1, cells / 2));
    const int64_t pos0 = cells - T, pages = (cells + PS - 1) / PS, rows = pages * nkv * PS;
    std::vector<void*> owned;
    auto dev = [&](size_t bytes) {
        void* p = sycl::malloc_device(std::max<size_t>(bytes, 64), queue);
        if (p == nullptr) throw std::bad_alloc();
        queue.memset(p, 0, std::max<size_t>(bytes, 64));
        owned.push_back(p);
        return p;
    };
    int best = kPickNone;
    double ms[3] = {-1.0, -1.0, -1.0};   // the decode kernel, the matrix engines', the FP16 kernel
    try {
        QsaAttnPools pools;
        std::vector<int32_t> table((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
        auto* tbl = static_cast<int32_t*>(dev(table.size() * 4));
        queue.memcpy(tbl, table.data(), table.size() * 4);
        pools.page_table = tbl;
        const size_t q4 = (size_t) rows * (size_t) kv_q4_bytes_per_head((int) hd);
        if (fmt == kFmtI8 || fmt == kFmtK8V4) {
            pools.k_q = static_cast<int8_t*>(dev((size_t) (rows * hd)));
            pools.k_scale = static_cast<uint16_t*>(dev((size_t) (rows * hd / 64) * 2));
        }
        if (fmt == kFmtI8) {
            pools.v_q = static_cast<int8_t*>(dev((size_t) (rows * hd)));
            pools.v_scale = static_cast<uint16_t*>(dev((size_t) (rows * hd / 64) * 2));
        }
        if (fmt == kFmtQ4) pools.k_q4 = static_cast<uint8_t*>(dev(q4));
        if (fmt == kFmtQ4 || fmt == kFmtK8V4) pools.v_q4 = static_cast<uint8_t*>(dev(q4));
        if (fmt == kFmtF16) {
            pools.k_pool = static_cast<uint16_t*>(dev((size_t) (rows * hd) * 2));
            pools.v_pool = static_cast<uint16_t*>(dev((size_t) (rows * hd) * 2));
        }
        auto* qv = static_cast<float*>(dev((size_t) (T * nh * hd) * 4));
        auto* out = static_cast<float*>(dev((size_t) (T * nh * hd) * 4));
        queue.wait_and_throw();
        // a warm run of a few queries (the kernel's compile), then the timed chunk; -1: not taken
        auto timed = [&](auto&& run) -> double {
            if (!run(std::min<int64_t>(T, 16))) return -1.0;
            queue.wait_and_throw();
            const auto t0 = std::chrono::steady_clock::now();
            run(T);
            queue.wait_and_throw();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        ms[kPickMma] = timed([&](int64_t n) { return run_mma(queue, qv, pools, fmt, pos0 + T - n, n, s, out); });
        if (fmt == kFmtF16)
            ms[kPickFp16] = timed([&](int64_t n) {
                return prompt_attn(qv, pools.k_pool, pools.v_pool, pools.page_table, pos0 + T - n, n, s, out, stream);
            });
        // the decode kernel over a slice of the queries its scratch fits in (it takes each query alike)
        const uint64_t per_q = qsa_decode_attn_scratch_floats(cells, s) * 4;
        const int64_t nq = std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) ((256ull << 20) / std::max<uint64_t>(per_q, 1))));
        std::vector<int32_t> hs((size_t) (nq * kStepCount));
        for (int64_t t = 0; t < nq; ++t) qsa_step_fill(hs.data() + t * kStepCount, pos0 + T - nq + t, s);
        auto* steps = static_cast<int32_t*>(dev(hs.size() * 4));
        queue.memcpy(steps, hs.data(), hs.size() * 4);
        auto* scratch = static_cast<float*>(dev((size_t) (per_q * (uint64_t) nq)));
        queue.wait_and_throw();
        try {
            const double d = timed([&](int64_t n) {
                if (n == T) qsa_dense_attn_batch(qv, pools, steps, cells, s, scratch, out, nq, stream);
                else qsa_dense_attn_batch(qv, pools, steps, cells, s, scratch, out, std::min<int64_t>(nq, n), stream);
                return true;
            });
            ms[kPickNone] = d * (double) T / (double) nq;
        } catch (const std::exception&) {   // a shape the decode kernel does not take
            ms[kPickNone] = -1.0;
        }
        // the kernels that read a cell once for many queries first; the decode kernel only when it is clearly ahead:
        // timed alone it came out 1.3% ahead of the matrix engines' at 64K cells of Qwen3.5-9B's 16/4 heads of 256 on
        // a B70, and in the prompt (its per-query scratch beside the other layers' work) it ran 15% behind them
        // (attention 75.1 s against 65.2 of a 63K prompt)
        double t_best = -1.0;
        for (int k : {(int) kPickMma, (int) kPickFp16})
            if (ms[k] >= 0.0 && (t_best < 0.0 || ms[k] < t_best)) {
                t_best = ms[k];
                best = k;
            }
        if (ms[kPickNone] >= 0.0 && (t_best < 0.0 || ms[kPickNone] < kDecodeMargin * t_best)) best = kPickNone;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: timing the prompt attention at %lld cells: %s\n", (long long) cells, e.what());
        best = kPickUntimed;
    }
    for (void* p : owned) sycl::free(p, queue);
    if (best >= 0)
        std::fprintf(stderr, "strata: prompt attention (%lld/%lld heads of %lld, %s pools, %lld queries after %lld cells): "
                             "the decode kernel %.1f ms, the matrix engines' %.1f, the FP16 kernel %.1f: %s\n",
                     (long long) s.n_head, (long long) s.n_head_kv, (long long) s.head_dim, kFmtNames[fmt],
                     (long long) T, (long long) pos0, ms[kPickNone], ms[kPickMma], ms[kPickFp16], kPickNames[best]);
    return best;
}

// STRATA_PROMPT_ATTN_PICK=decode|mma|fp16: that kernel whatever the timing would choose (A/B); kPickUntimed: none
int forced_pick() {
    const char* v = std::getenv("STRATA_PROMPT_ATTN_PICK");
    if (v == nullptr) return kPickUntimed;
    const std::string w = v;
    return w == "decode" ? kPickNone : w == "mma" ? kPickMma : w == "fp16" ? kPickFp16 : kPickUntimed;
}

}  // namespace

void prompt_attn_tune(const QsaShapes& s, int fmt, int64_t max_cells, void* stream) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0 || max_cells <= 0 || fmt < kFmtF16 ||
        fmt > kFmtK8V4)
        return;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    picks_for(s, fmt).get(queue.get_device(), [&] {
        Picks p;
        if (env_off("STRATA_PROMPT_ATTN_MMA")) return p;   // the fixed choice (prompt_attn)
        if (const int f = forced_pick(); f != kPickUntimed) {   // prompt_attn takes it untimed
            std::fprintf(stderr, "strata: prompt attention: %s (STRATA_PROMPT_ATTN_PICK)\n", kPickNames[f]);
            return p;
        }
        // 8K and up to 64K cells (1 to 2 s on a B70); a cache shorter than the long point takes the short one's choice
        const int64_t a = std::min<int64_t>(max_cells, 8192), b = std::min<int64_t>(max_cells, 65536);
        p.short_pick = time_pick(queue, s, fmt, a, stream);
        p.long_pick = b > a ? time_pick(queue, s, fmt, b, stream) : p.short_pick;
        p.split = (a + b) / 2;
        return p;
    });
}

bool prompt_attn(const float* q, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s, float* attn,
                 void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0 || pools.page_table == nullptr) return false;
    const int fmt = pool_format(pools);
    if (fmt < 0) return false;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    // STRATA_PROMPT_ATTN_MMA=0: the FP16 kernel for FP16 pools and the decode kernel for INT8 ones, as before; untimed
    // (no prompt_attn_tune for this shape): the matrix engines' kernel where there is one
    static const bool no_mma = env_off("STRATA_PROMPT_ATTN_MMA");
    static const int forced = forced_pick();
    int pick = kPickUntimed;
    if (no_mma) {
        pick = fmt != kFmtF16 ? kPickNone : kPickFp16;
    } else if (forced != kPickUntimed) {
        pick = forced;
    } else {
        const Picks& p = picks_for(s, fmt).get(queue.get_device(), [] { return Picks{}; });
        pick = pos0 + T >= p.split ? p.long_pick : p.short_pick;
    }
    if (pick == kPickNone) return false;
    if (pick != kPickFp16 && run_mma(queue, q, pools, fmt, pos0, T, s, attn)) {
        if (stream == nullptr) queue.wait_and_throw();
        return true;
    }
    if (fmt != kFmtF16) return false;   // the FP32 and WMMA kernels read FP16 pools: the caller keeps the decode kernel
    return prompt_attn(q, pools.k_pool, pools.v_pool, pools.page_table, pos0, T, s, attn, stream);
}

}  // namespace strata::kernels
