// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "attn_parts.hpp"
#include "strata/core/per_device.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/sdpa_aotriton.hpp"
#include "strata/kernels/sdpa_onednn.hpp"
#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

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
#include <unordered_map>
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
using xe::attn::h8;
using xe::attn::i8_row8;
using xe::attn::pool_format;
using xe::attn::q4_row8;

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
        const char* v = strata::env_get("PROMPT_ATTN_MMA");
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
enum Pick : int { kPickUntimed = -1, kPickNone = 0, kPickMma = 1, kPickFp16 = 2, kPickI8 = 3, kPickMaker = 4 };
const char* const kPickNames[5] = {"the decode kernel", "the matrix engines'", "the FP16 kernel", "the int8 kernel",
                                   "the maker's kernel"};
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
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}

// The attention the image engine shares (flash_attn.hpp): FP16 pools in pages of 64 cells (or a multiple) read where
// they are, the other formats and page sizes gathered into local memory as FP16 a chunk at a time; the output kept in
// the matrix engines' accumulators.  STRATA_PROMPT_ATTN_FLASH=0: the kernels above (A/B)
bool flash_pools(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
                 const QsaShapes& s, float* attn) {
    static const bool off = env_off("PROMPT_ATTN_FLASH");
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

// The int8 attention (flash_attn_i8_causal: SageAttention's scheme, the image engine's kernel) over the pools, read
// through the page table in any of their formats: the queries made FP16, its FP16 output widened into attn (the
// stream's scratch, 4 bytes a query row's value); false where it does not take the shape here (the GPU, the head size,
// the memory left)
// a stream's scratch (slot 0 run_i8's, 1 run_maker's), kept for the next call: at most a 32nd of the GPU's memory
// (past that the next kernel, not a later allocation of the engine's failing)
struct Scratch {
    uint16_t* p = nullptr;
    size_t n = 0;
};
std::mutex scratch_mu;
std::unordered_map<const sycl::queue*, Scratch[2]>& scratch_map() {
    static std::unordered_map<const sycl::queue*, Scratch[2]> of;
    return of;
}
uint16_t* scratch(sycl::queue& queue, int slot, size_t n) {
    std::lock_guard<std::mutex> g(scratch_mu);
    Scratch& sc = scratch_map()[&queue][slot];
    if (sc.n >= n) return sc.p;
    size_t free_b = 0, total_b = 0;
    if (!gpu::mem_info(&free_b, &total_b) || n * 2 > total_b / 32 || free_b + sc.n * 2 < n * 2) return nullptr;
    queue.wait();
    if (sc.p != nullptr) gpu::free(sc.p);
    sc = Scratch{};
    if (gpu::alloc_device(&sc.p, n * sizeof(uint16_t))) sc.n = n;
    else sc.p = nullptr;
    return sc.p;
}
void release_scratch(sycl::queue& queue, int slot) {
    std::lock_guard<std::mutex> g(scratch_mu);
    auto& of = scratch_map();
    if (auto it = of.find(&queue); it != of.end() && it->second[slot].p != nullptr) {
        queue.wait();
        gpu::free(it->second[slot].p);
        it->second[slot] = Scratch{};
    }
}

bool run_i8(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int64_t pos0, int64_t T,
            const QsaShapes& s, float* attn, void* stream) {
    if (pools.page_table == nullptr || flash_attn_i8_kind(stream) == 0) return false;
    const int64_t CQ = s.n_head * s.head_dim;
    uint16_t* q16 = scratch(queue, 0, (size_t) (2 * T * CQ));
    if (q16 == nullptr) return false;
    uint16_t* o16 = q16 + T * CQ;
    queue.parallel_for(sycl::range<1>((size_t) (T * CQ)), [=](sycl::id<1> i) { q16[i] = f16_gpu::f16_from_f32(q[i]); });
    if (!flash_attn_i8_causal(q16, pools, s.page_size, T, pos0 + T, s.n_head, s.n_head_kv, s.head_dim, pos0, o16,
                              stream))
        return false;
    queue.parallel_for(sycl::range<1>((size_t) (T * CQ)),
                       [=](sycl::id<1> i) { attn[i] = f16_gpu::f32_from_f16(o16[i]); });
    return true;
}

// The maker's kernel (onednn_sdpa: oneDNN's fused SDPA on Intel GPUs; aotriton_sdpa: aotriton's flash attention on AMD
// ones) over the pools gathered into dense FP16 rows
// (INT8 and Q4_0 decoded), the queries made FP16, its output widened into attn; false where it does not take it.  Its
// shapes are the chunk's own, not padded: oneDNN's fused kernel takes the causal mask's offset from the shapes, not
// from the lengths given with them (padded, the rows saw the padding's keys: the B70, oneDNN 3.12, 2026-11-07), and a
// prompt's new shape a chunk costs nothing seen (Qwen3.5-9B's 16K prompt in 2048-token chunks).
// run_maker's scratch for T rows over Tk keys (FP16 values): K and V, Q and the output
size_t maker_scratch(const QsaShapes& s, int64_t T, int64_t Tk) {
    return (size_t) (2 * Tk * s.n_head_kv * s.head_dim + 2 * T * s.n_head * s.head_dim);
}

template<int HD>
void gather16(sycl::queue& queue, const QsaAttnPools& pools, int64_t page_size, int64_t KH, int64_t Tk, uint16_t* k16,
              uint16_t* v16) {
    const int fmt = pool_format(pools);
    const uint16_t *kf = pools.k_pool, *vf = pools.v_pool, *ksc = pools.k_scale, *vsc = pools.v_scale;
    const int8_t *ki = pools.k_q, *vi = pools.v_q;
    const uint8_t *k4 = pools.k_q4, *v4 = pools.v_q4;
    const int32_t* tbl = pools.page_table;
    const int64_t PS = page_size, CK = KH * HD;
    queue.parallel_for(sycl::range<1>((size_t) (Tk * CK / 8)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0] * 8, j = i / CK, g = (i - j * CK) / HD;
        const int dd = (int) (i - j * CK - g * HD);
        const int64_t row = ((int64_t) tbl[j / PS] * KH + g) * PS + j % PS;
        h8 kv, vv;
        if (fmt == kFmtF16) {
            kv = *reinterpret_cast<const h8*>(kf + row * HD + dd);
            vv = *reinterpret_cast<const h8*>(vf + row * HD + dd);
        } else {
            kv = fmt == kFmtQ4 ? q4_row8<HD>(k4, row, dd) : i8_row8<HD>(ki, ksc, row, dd);
            vv = fmt == kFmtI8 ? i8_row8<HD>(vi, vsc, row, dd) : q4_row8<HD>(v4, row, dd);
        }
        *reinterpret_cast<h8*>(k16 + i) = kv;
        *reinterpret_cast<h8*>(v16 + i) = vv;
    });
}

bool run_maker(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int64_t pos0, int64_t T,
               const QsaShapes& s, float* attn, void* stream) {
    const bool aot = aotriton_sdpa_ready(stream);
#if !STRATA_ONEDNN
    if (!aot) return false;
#endif
    const int64_t hd = s.head_dim, KH = s.n_head_kv, NH = s.n_head, Tk = pos0 + T;
    if (pools.page_table == nullptr || pool_format(pools) < 0 || (hd != 128 && hd != 256)) return false;
    const int64_t CK = KH * hd, CQ = NH * hd;
    uint16_t* k16 = scratch(queue, 1, maker_scratch(s, T, Tk));
    if (k16 == nullptr) return false;
    uint16_t *v16 = k16 + Tk * CK, *q16 = v16 + Tk * CK, *o16 = q16 + T * CQ;
    if (hd == 128) gather16<128>(queue, pools, s.page_size, KH, Tk, k16, v16);
    else gather16<256>(queue, pools, s.page_size, KH, Tk, k16, v16);
    queue.parallel_for(sycl::range<1>((size_t) (T * CQ)), [=](sycl::id<1> i) { q16[i] = f16_gpu::f16_from_f32(q[i]); });
    bool ok = aot && aotriton_sdpa(q16, k16, v16, T, Tk, NH, KH, hd, o16, stream, true);
#if STRATA_ONEDNN
    ok = ok || (!aot && onednn_sdpa(q16, k16, v16, T, Tk, NH, KH, hd, o16, stream, true));
#endif
    if (!ok) return false;
    queue.parallel_for(sycl::range<1>((size_t) (T * CQ)),
                       [=](sycl::id<1> i) { attn[i] = f16_gpu::f32_from_f16(o16[i]); });
    return true;
}

// The buffers of run_i8 and run_maker are taken before the timing at the size of its longer shape (or of the shorter,
// past a 32nd of the GPU's memory), so that the engine, which plans its memory after the timing, plans around them:
// kept and freed (and grown) between the timing's shapes, the B70's driver kept what was freed, and on a card of
// 8 GB (the B70 under STRATA_VRAM_LIMIT_MIB=8192, Qwen3.5-9B at 20K cells) 289 MiB less were free to the plan
// (2026-11-07).  A stream that does not take them gives them back after the timing.
void prepare_buffers(sycl::queue& queue, const QsaShapes& s, int64_t a, int64_t b, void* stream) {
    const int64_t T = std::min<int64_t>(2048, std::max<int64_t>(1, a / 2));   // time_pick's rows
    for (const int64_t cells : {b, a})
        if (flash_attn_i8_causal_prepare(T, cells, s.n_head, s.n_head_kv, s.head_dim, stream)) break;
    // a maker's kernel here: aotriton (an AMD GPU with its module) or oneDNN (an Intel GPU's Level Zero or OpenCL)
    const sycl::backend be = queue.get_device().get_backend();
    bool maker = aotriton_sdpa_ready(stream);
#if STRATA_ONEDNN
    maker = maker || be == sycl::backend::ext_oneapi_level_zero || be == sycl::backend::opencl;
#endif
    (void) be;
    if (maker)
        for (const int64_t cells : {b, a})
            if (scratch(queue, 1, maker_scratch(s, std::min<int64_t>(2048, cells / 2), cells)) != nullptr) break;
}
void release_unpicked(sycl::queue& queue, void* stream, const Picks& p) {
    if (p.short_pick != kPickI8 && p.long_pick != kPickI8) {
        flash_attn_i8_causal_release(stream);
        release_scratch(queue, 0);
    }
    if (p.short_pick != kPickMaker && p.long_pick != kPickMaker) {
        release_scratch(queue, 1);
#if STRATA_ONEDNN
        onednn_sdpa_drop_causal(stream);
#endif
    }
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
    double ms[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};   // as kPickNames
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
        // a warm run of the timed shape itself, then the timed chunk; -1: not taken.  The warm run is the whole chunk:
        // a library compiles a kernel per shape (oneDNN's SDPA), and a cold kernel cache (a first start, a new
        // driver) put that compile into a shorter warm-up's timed chunk: 419.7 ms for oneDNN's 8.2 on the B70
        // (2026-11-07), the slower kernel kept for the run
        auto timed = [&](auto&& run) -> double {
            if (!run(T)) return -1.0;
            queue.wait_and_throw();
            const auto t0 = std::chrono::steady_clock::now();
            run(T);
            queue.wait_and_throw();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        ms[kPickMma] = timed([&](int64_t n) { return run_mma(queue, qv, pools, fmt, pos0 + T - n, n, s, out); });
        ms[kPickI8] = timed([&](int64_t n) { return run_i8(queue, qv, pools, pos0 + T - n, n, s, out, stream); });
        ms[kPickMaker] =
            timed([&](int64_t n) { return run_maker(queue, qv, pools, pos0 + T - n, n, s, out, stream); });
        if (fmt == kFmtF16)
            ms[kPickFp16] = timed([&](int64_t n) {
                return prompt_attn(qv, pools.k_pool, pools.v_pool, pools.page_table, pos0 + T - n, n, s, out, stream);
            });
        // the decode kernel over a slice of the queries its scratch fits in (it takes each query alike), 32 MiB: the
        // timing's buffers are freed after it, and a GPU's driver may keep them from the free memory the engine plans
        // with (the B70's kept 256 MiB of it at 256, 2026-11-07)
        const uint64_t per_q = qsa_decode_attn_scratch_floats(cells, s) * 4;
        const int64_t nq =
            std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) ((32ull << 20) / std::max<uint64_t>(per_q, 1))));
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
        for (int k : {(int) kPickMma, (int) kPickFp16, (int) kPickI8, (int) kPickMaker})
            if (ms[k] >= 0.0 && (t_best < 0.0 || ms[k] < t_best)) {
                t_best = ms[k];
                best = k;
            }
        if (ms[kPickNone] >= 0.0 && (t_best < 0.0 || ms[kPickNone] < kDecodeMargin * t_best)) best = kPickNone;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the prompt attention at %lld cells: %s\n", (long long) cells, e.what());
        best = kPickUntimed;
    }
    for (void* p : owned) sycl::free(p, queue);
    if (best >= 0)
        std::fprintf(stderr, "xestrata: prompt attention (%lld/%lld heads of %lld, %s pools, %lld queries after %lld cells): "
                             "the decode kernel %.1f ms, the matrix engines' %.1f, the FP16 kernel %.1f, the int8 "
                             "kernel %.1f, the maker's %.1f: %s\n",
                     (long long) s.n_head, (long long) s.n_head_kv, (long long) s.head_dim, kFmtNames[fmt],
                     (long long) T, (long long) pos0, ms[kPickNone], ms[kPickMma], ms[kPickFp16], ms[kPickI8],
                     ms[kPickMaker], kPickNames[best]);
    return best;
}

// STRATA_PROMPT_ATTN_PICK=decode|mma|fp16|int8|maker: that kernel whatever the timing would choose (A/B);
// kPickUntimed: none
int forced_pick() {
    const char* v = strata::env_get("PROMPT_ATTN_PICK");
    if (v == nullptr) return kPickUntimed;
    const std::string w = v;
    return w == "decode" ? kPickNone
           : w == "mma"  ? kPickMma
           : w == "fp16" ? kPickFp16
           : w == "int8" ? kPickI8
           : w == "maker" ? kPickMaker
                          : kPickUntimed;
}

}  // namespace

void prompt_attn_tune(const QsaShapes& s, int fmt, int64_t max_cells, void* stream) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0 || max_cells <= 0 || fmt < kFmtF16 ||
        fmt > kFmtK8V4)
        return;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    picks_for(s, fmt).get(queue.get_device(), [&] {
        Picks p;
        if (env_off("PROMPT_ATTN_MMA")) return p;   // the fixed choice (prompt_attn)
        if (const int f = forced_pick(); f != kPickUntimed) {   // prompt_attn takes it untimed
            std::fprintf(stderr, "xestrata: prompt attention: %s (XESTRATA_PROMPT_ATTN_PICK)\n", kPickNames[f]);
            return p;
        }
        // 8K and up to 64K cells (1 to 2 s on a B70); a cache shorter than the long point takes the short one's choice
        const int64_t a = std::min<int64_t>(max_cells, 8192), b = std::min<int64_t>(max_cells, 65536);
        prepare_buffers(queue, s, a, b, stream);
        p.short_pick = time_pick(queue, s, fmt, a, stream);
        p.long_pick = b > a ? time_pick(queue, s, fmt, b, stream) : p.short_pick;
        p.split = (a + b) / 2;
        release_unpicked(queue, stream, p);
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
    static const bool no_mma = env_off("PROMPT_ATTN_MMA");
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
    if ((pick == kPickI8 && run_i8(queue, q, pools, pos0, T, s, attn, stream)) ||
        (pick == kPickMaker && run_maker(queue, q, pools, pos0, T, s, attn, stream))) {
        if (stream == nullptr) queue.wait_and_throw();
        return true;
    }
    if (pick != kPickFp16 && run_mma(queue, q, pools, fmt, pos0, T, s, attn)) {
        if (stream == nullptr) queue.wait_and_throw();
        return true;
    }
    if (fmt != kFmtF16) return false;   // the FP32 and WMMA kernels read FP16 pools: the caller keeps the decode kernel
    return prompt_attn(q, pools.k_pool, pools.v_pool, pools.page_table, pos0, T, s, attn, stream);
}

}  // namespace strata::kernels
