// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/moe_mmq.cpp - see include/strata/prefill/moe_mmq.hpp: the prompt path's MMQ steps over
// strata::kernels' int8 products (iq_mmq.hpp).
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/swiglu_clamp.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU prefill_moe_mmq
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace strata::prefill::mmq {
namespace {
sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
}  // namespace

bool built() { return kernels::iq_mmq_usable(core::Runtime::get().compute()); }
bool built_dp4a() { return kernels::iq_mmq_dp4a_usable(core::Runtime::get().compute()); }
bool chosen() { return built() || (built_dp4a() && !kernels::f16_matrix_engines()); }
const char* path() { return chosen() ? kernels::iq_mmq_path(core::Runtime::get().compute()) : nullptr; }
bool supported(int t) { return kernels::iq_mmq_type_ok(t); }
size_t matrix_bytes(int t, int64_t rows, int64_t cols) { return (size_t) rows * kernels::iq_mmq_row_bytes(t, cols); }
size_t q8_bytes(int64_t rows, int64_t cols) { return kernels::iq_mmq_act_bytes(rows, cols); }

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows > 0)
        kernels::iq_mmq_quantize(Q(stream), x, ids, xq, cols, ld, rows, kernels::iq_mmq_type_sums(t),
                                 kernels::iq_mmq_type_dup(t));
}

void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int t, int64_t cols,
                      int64_t ld, int64_t tokens, int k_used, void* stream) {
    static const bool gather = [] {
        const char* e = strata::env_get("QUANT_GATHER");
        return e != nullptr && e[0] == '1';
    }();
    if (gather) return quantize(x, src, xq, t, cols, ld, tokens * k_used, stream);
    if (tokens > 0)
        kernels::iq_mmq_quantize_scatter(Q(stream), x, slot, xq, cols, ld, tokens, k_used, kernels::iq_mmq_type_sums(t),
                                         kernels::iq_mmq_type_dup(t));
}

void choose_forms(int t, int64_t w_rows, int64_t w_cols, int n, void* stream) {
    kernels::iq_mmq_choose_forms(Q(stream), t, w_rows, w_cols, n);
}
void split_set(int quarters, void* stream) { kernels::iq_mmq_split_set(Q(stream), quarters); }
int split_get(void* stream) { return kernels::iq_mmq_split_get(Q(stream)); }
bool split_reserve(int64_t rows, int64_t n_out, void* stream) {
    return kernels::iq_mmq_split_reserve(Q(stream), kernels::iq_mmq_split_floats(rows, n_out));
}


Context::Context() {}
Context::~Context() {}
void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    // a type that reads every activation block twice (Q2_K) reads rows of twice the columns
    const int64_t cols = kernels::iq_mmq_type_dup(p.type) ? 2 * p.w_cols : p.w_cols;
    kernels::iq_mmq_grouped(Q(stream), p.type, p.w, p.expert_bytes, p.w_rows, cols, p.n, p.xq, p.total_rows,
                            p.bounds, p.max_rows, p.ids, p.dst, p.ld_dst, p.wtab, p.woff,
                            p.rows > 0 ? p.rows : p.total_rows);
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    sycl::queue& q = Q(stream);
    // One kernel of 16-byte copies rather than three memcpys: a device-to-device memcpy goes to the copy engine, behind
    // the streamed experts' host-to-device copies (RTX 4070, IQ2_XS, a 32K prompt: 1228 -> 1244 tok/s)
    const auto a16 = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (!a16(gate) || !a16(up) || !a16(down) || !a16(gu_dst) || !a16(d_dst) || gu_half_bytes % 16 != 0 ||
        d_bytes % 16 != 0) {
        q.memcpy(gu_dst, gate, gu_half_bytes);
        q.memcpy(static_cast<uint8_t*>(gu_dst) + gu_half_bytes, up, gu_half_bytes);
        q.memcpy(d_dst, down, d_bytes);
        return;
    }
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)
    const auto* g = static_cast<const u32x4*>(gate);
    const auto* u = static_cast<const u32x4*>(up);
    const auto* dn = static_cast<const u32x4*>(down);
    auto* gu = static_cast<u32x4*>(gu_dst);
    auto* dd = static_cast<u32x4*>(d_dst);
    const int64_t h = (int64_t) (gu_half_bytes / 16), nd = (int64_t) (d_bytes / 16), total = 2 * h + nd;
    const size_t groups = (size_t) std::min<int64_t>((total + 255) / 256, 1024);
    q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t step = (int64_t) it.get_global_range(0);
        for (int64_t i = (int64_t) it.get_global_id(0); i < total; i += step) {
            if (i < h) gu[i] = g[i];
            else if (i < 2 * h) gu[i] = u[i - h];
            else dd[i - 2 * h] = dn[i - 2 * h];
        }
    });
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride |
                  d_stride;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    struct Blobs { const u32x4* p[kGatherGroupMax]; } blobs{};
    for (int q = g.first; q < g.n; ++q) blobs.p[q] = reinterpret_cast<const u32x4*>(g.blob[q]);
    const int64_t h = (int64_t) (gu_half_bytes / 16), nd = (int64_t) (d_bytes / 16), total = 2 * h + nd;
    const int64_t uo = (int64_t) (up_off / 16), dof = (int64_t) (down_off / 16);
    const int64_t gs = (int64_t) (gu_stride / 16), ds = (int64_t) (d_stride / 16);
    auto* gu = static_cast<u32x4*>(gu_dst);
    auto* dd = static_cast<u32x4*>(d_dst);
    const int first = g.first;
    const size_t ne = (size_t) (g.n - g.first), groups = (size_t) std::min<int64_t>((total + 255) / 256, 256);
    Q(stream).parallel_for(sycl::nd_range<2>({ne, groups * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const int q = first + (int) it.get_group(0);
        // the expert's blob by constant indices: indexed by q, the NVPTX compile copied the 16 pointers to each
        // work-item's stack (the kernel ran at a quarter of the speed of upstream's copy16_group_kernel)
        const u32x4* src = nullptr;
        for (int k = 0; k < kGatherGroupMax; ++k)
            if (k == q) src = blobs.p[k];
        if (src == nullptr) return;   // an expert the products read where it is (no copy)
        u32x4* ab = gu + q * gs;
        u32x4* cd = dd + q * ds;
        const int64_t step = (int64_t) it.get_global_range(1);
        for (int64_t i = (int64_t) it.get_global_id(1); i < total; i += step) {
            if (i < h) ab[i] = src[i];
            else if (i < 2 * h) ab[i] = src[uo + (i - h)];
            else cd[i - 2 * h] = src[dof + (i - 2 * h)];
        }
    });
    return true;
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    const float lim = kernels::expert_swiglu_limit();   // GLM-5.3-Flash's clamp (0: none)
    Q(stream).parallel_for(sycl::range<1>((size_t) (rows * n_ff)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / n_ff, k = i % n_ff;
        const float* row = gu + r * 2 * n_ff;
        float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
        kernels::swiglu_clamp(g, u, lim);
        h[i] = g / (1.0f + sycl::exp(-g)) * u;
    });
}

// swiglu()'s value of row r's element k, its gate and up side by side (interleaved) or in the row's halves, after
// the clamp (swiglu_quantize's source)
struct SwigluRows {
    static constexpr bool kRows = true;
    const float* gu;
    int64_t n_ff;
    float lim;
    bool interleaved;
    float operator()(int64_t r, int64_t c, int j) const {
        const float* row = gu + r * 2 * n_ff;
        const int64_t k = c * 32 + j;
        float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
        kernels::swiglu_clamp(g, u, lim);
        return g / (1.0f + sycl::exp(-g)) * u;
    }
};

void swiglu_quantize(const float* gu, void* xq, int t, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    // swiglu()'s value for each element, then kernels::iq_mmq_quantize's blocks (ids null, ld n_ff): the same bytes
    // without H's round trip through memory
    const bool dup = kernels::iq_mmq_type_dup(t);
    const int64_t oc = dup ? 2 * n_ff : n_ff, nb = n_ff / 32;
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * oc);
    float* xu = xs + rows * (oc / 32);
    const SwigluRows src{gu, n_ff, kernels::expert_swiglu_limit(), interleaved};
    if (kernels::iq_mmq_type_sums(t)) {
        const kernels::q8act::Rows<true> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1};
        kernels::q8act::quantize<8, kernels::q8act::Mul>(Q(stream), rows * nb, nb, src, out);
    } else {
        const kernels::q8act::Rows<false> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1};
        kernels::q8act::quantize<8, kernels::q8act::Mul>(Q(stream), rows * nb, nb, src, out);
    }
}

// From how many rows an expert the FP16 path beats the int8 products on this GPU, for a layer of format `f` (its types
// and shape; synthetic blocks: any bits time alike), timed as a chunk's layer runs them: the experts a chunk of 8192
// routed rows spreads over at 16 to 1024 rows an expert (all of the model's at the fewest rows; a quarter of the rows where the
// device cannot hold the larger buffers), the int8 products `launches.int8_group` experts a launch from their
// activations' int8 rows (each expert's blob first gathered into a group buffer where the caller does), against the
// experts dequantized to FP16 `launches.f16_group` at a time and their grouped products, each with its SwiGLU and its
// rows' conversion.  Timed 8 experts a launch on either side before (2026-11-12), the int8 side paid 32 launches for
// what PlainModel's prompt takes in one, and an Ornith-1.5-35B-A3B GSQ-RCO 3bit layer chose FP16 from 23 or 45 rows an
// expert (B70, from start to start): its 1K prompt (32 rows) read in 0.64 s against 0.40 in int8, its 4K one (111 rows)
// 1.05 against 1.26.  The two sides measured in turn, each the least of its timings after both ran a while (the GPU
// clocked up), and the crossover put between the last row count int8 won at and the first it lost at, where the two
// lines cross on a log scale: a timing a few percent off moves it a few percent, not to the next power of two (a
// chunk's mean of 32 rows went int8 or FP16 from start to start).  FP16 wins a count where it is 5% faster.  INT64_MAX:
// int8 at every count; 0: FP16 from the fewest.  Where the timing cannot run, `fallback`.
static int64_t time_int8_rows_below(const kernels::cpu::NativeFmt& f, size_t blob, int64_t fallback,
                                    const ExpertLaunches& launches) {
    namespace k = strata::kernels;
    using Clock = std::chrono::steady_clock;
    constexpr int64_t RS[] = {16, 32, 64, 128, 256, 512, 1024};
    const int64_t FF = f.n_ff, NE = f.n_embd;
    const int G8 = std::max(1, launches.int8_group), G16 = std::max(1, launches.f16_group);
    const int64_t n_expert = std::max<int64_t>(launches.n_expert, G16);
    // a layer whose gate and up differ in type (k::cpu::native_mixed_gu): gate and up a product each, as PlainModel's
    // prompt takes them
    const bool mixed = k::cpu::native_mixed_gu(f);
    // the int8 products' forms as the prompt finds them chosen (PlainModel's prompt_warm and Qwen3.8-Flash-Next's
    // mmq_plan choose them for every layer; timed before them, every product took the work-group form)
    if (mixed) {
        choose_forms(f.gu_type, FF, NE, 16, nullptr);
        choose_forms(f.up_type, FF, NE, 16, nullptr);
    } else {
        choose_forms(f.gu_type, 2 * FF, NE, 16, nullptr);
    }
    choose_forms(f.d_type, NE, FF, 16, nullptr);
    std::vector<void*> owned;
    struct FreeAll {
        std::vector<void*>& o;
        ~FreeAll() { for (void* p : o) strata::gpu::free(p); }
    } free_all{owned};
    auto dev = [&](size_t bytes) {
        void* p = strata::gpu::alloc_device(bytes);
        if (p != nullptr) owned.push_back(p);
        return p;
    };
    // the rows of a timing: a chunk's 8192 routed rows, else 2048 (a device short of memory)
    int64_t RMAX = 8192;
    const auto row_bytes = [&](int64_t rows) {
        return (size_t) rows * (size_t) (NE * 12 + FF * 18 + 4) + q8_bytes(rows, NE) * 2 + q8_bytes(rows, FF) * 2;
    };
    if (void* probe = strata::gpu::alloc_device(row_bytes(RMAX))) strata::gpu::free(probe);
    else RMAX = 2048;
    const auto experts_at = [&](int64_t r) { return std::max<int64_t>(G16, std::min<int64_t>(n_expert, RMAX / r)); };
    const int64_t GE = experts_at(RS[0]);
    // the experts' weights: up to 64 different blobs (more than a GPU's cache holds), the others read them again
    int64_t GW = std::min<int64_t>(GE, 64);
    uint8_t* w = static_cast<uint8_t*>(dev((size_t) GW * blob));
    while (w == nullptr && GW > 8) {
        GW /= 2;
        w = static_cast<uint8_t*>(dev((size_t) GW * blob));
    }
    const int GG = launches.int8_gathers ? G8 : 0;
    auto* gath = GG > 0 ? static_cast<uint8_t*>(dev((size_t) GG * blob)) : nullptr;
    auto* dq_gu = static_cast<uint16_t*>(dev((size_t) G16 * 2 * FF * NE * 2));
    auto* dq_d = static_cast<uint16_t*>(dev((size_t) G16 * NE * FF * 2));
    auto* xs16 = static_cast<uint16_t*>(dev((size_t) RMAX * NE * 2));
    auto* x16 = static_cast<uint16_t*>(dev((size_t) RMAX * NE * 2));
    auto* h16 = static_cast<uint16_t*>(dev((size_t) RMAX * FF * 2));
    auto* gu = static_cast<float*>(dev((size_t) RMAX * 2 * FF * 4));
    auto* hh = static_cast<float*>(dev((size_t) RMAX * FF * 4));
    auto* dn = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    auto* zx = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    void* xq = dev(q8_bytes(RMAX, NE) * 2);
    void* hq = dev(q8_bytes(RMAX, FF) * 2);
    auto* bounds = static_cast<int32_t*>(dev((size_t) (GE + 1) * 4));
    auto* rbounds = static_cast<int32_t*>(dev((size_t) (G8 + 1) * 4));
    auto* ident = static_cast<int32_t*>(dev((size_t) RMAX * 4));
    auto* wtab = static_cast<const uint8_t**>(dev((size_t) GE * sizeof(void*)));
    auto* gtab = static_cast<const uint8_t**>(dev((size_t) std::max(GG, 1) * sizeof(void*)));
    auto give_up = [&](const char* why) {
        std::fprintf(stderr, "xestrata: timing the experts' prompt products: %s (%s); %s\n", why, strata::gpu::last_error(),
                     fallback == 0 ? "the FP16 path" : "the int8 products");
        return fallback;
    };
    for (const void* p : {(const void*) w, (const void*) dq_gu, (const void*) dq_d, (const void*) xs16,
                          (const void*) x16, (const void*) h16, (const void*) gu, (const void*) hh, (const void*) dn,
                          (const void*) zx, (const void*) xq, (const void*) hq, (const void*) bounds,
                          (const void*) rbounds, (const void*) ident, (const void*) wtab, (const void*) gtab})
        if (p == nullptr) return give_up("no device memory for it");
    if (GG > 0 && gath == nullptr) return give_up("no device memory for it");
    try {
        void* s = nullptr;   // the compute queue
        std::vector<const uint8_t*> tab((size_t) GE), gt((size_t) std::max(GG, 1));
        for (int64_t e = 0; e < GE; ++e) tab[(size_t) e] = w + (size_t) (e % GW) * blob;
        for (int q = 0; q < GG; ++q) gt[(size_t) q] = gath + (size_t) q * blob;
        if (!strata::gpu::memset_async(w, 0x11, (size_t) GW * blob, s) ||
            !strata::gpu::memset_async(zx, 0, (size_t) RMAX * NE * 4, s) ||
            !strata::gpu::memset_async(xs16, 0, (size_t) RMAX * NE * 2, s) ||
            !strata::gpu::copy(static_cast<void*>(wtab), static_cast<const void*>(tab.data()),
                               tab.size() * sizeof(void*)) ||
            !strata::gpu::copy(static_cast<void*>(gtab), static_cast<const void*>(gt.data()), gt.size() * sizeof(void*)))
            return give_up("its buffers could not be set");
        iota(ident, RMAX, s);
        Context ctx;
        // the int8 side at r rows an expert over n experts: the activations' rows to int8, then a launch at a time
        // gate/up, the SwiGLU and its rows to int8, down (PlainModel's prompt_routed, Qwen3.8-Flash-Next's groups)
        auto run8 = [&](int64_t r, int64_t n) {
            const int64_t R = n * r;
            quantize(zx, ident, xq, f.gu_type, NE, NE, R, s);
            for (int64_t e0 = 0; e0 < n; e0 += G8) {
                const int g = (int) std::min<int64_t>(G8, n - e0);
                const int64_t r0 = e0 * r, nr = g * r;
                const uint8_t* const* wt = wtab + e0;
                if (GG > 0) {   // the group's blobs into its buffer first
                    for (int q = 0; q < g; ++q)
                        gather_native(tab[(size_t) (e0 + q)], tab[(size_t) (e0 + q)] + f.up_off, f.up_off,
                                      tab[(size_t) (e0 + q)] + f.down_off, blob - f.down_off, gath + (size_t) q * blob,
                                      gath + (size_t) q * blob + f.down_off, s);
                    wt = gtab;
                }
                Product a;
                a.type = f.gu_type; a.w_rows = 2 * FF; a.w_cols = NE; a.n = g; a.xq = xq; a.bounds = bounds + e0;
                a.ids = ident; a.total_rows = R; a.max_rows = r; a.rows = nr; a.dst = gu; a.ld_dst = 2 * FF;
                a.wtab = wt; a.woff = 0;
                if (mixed) {
                    a.w_rows = FF;
                    ctx.run(a, s);
                    a.type = f.up_type; a.dst = gu + FF; a.woff = f.up_off;
                }
                ctx.run(a, s);
                swiglu(gu + r0 * 2 * FF, hh + r0 * FF, nr, FF, false, s);
                quantize(hh + r0 * FF, nullptr, hq, f.d_type, FF, FF, nr, s);
                Product b;
                b.type = f.d_type; b.w_rows = NE; b.w_cols = FF; b.n = g; b.xq = hq; b.bounds = rbounds;
                b.ids = ident + r0; b.total_rows = nr; b.max_rows = r; b.rows = nr; b.dst = dn; b.ld_dst = NE;
                b.wtab = wt; b.woff = f.down_off;
                ctx.run(b, s);
            }
        };
        // the FP16 side: the rows to FP16, then f16_group experts at a time dequantized and their grouped products
        std::vector<int32_t> hb;
        auto run16 = [&](int64_t r, int64_t n) {
            const int64_t R = n * r;
            gather_rows16(xs16, ident, x16, R, NE, s);
            for (int64_t e0 = 0; e0 < n; e0 += G16) {
                const int g = (int) std::min<int64_t>(G16, n - e0);
                const int64_t r0 = e0 * r, nr = g * r;
                for (int q = 0; q < g; ++q) {
                    const uint8_t* t = tab[(size_t) (e0 + q)];
                    uint16_t* gd = dq_gu + (size_t) q * 2 * FF * NE;
                    uint16_t* dd = dq_d + (size_t) q * NE * FF;
                    if (mixed) {
                        k::dequant_f16(f.gu_type, t, 0, FF, NE, gd, s);
                        k::dequant_f16(f.up_type, t + f.up_off, 0, FF, NE, gd + FF * NE, s);
                        k::dequant_f16(f.d_type, t + f.down_off, 0, NE, FF, dd, s);
                    } else {
                        k::iq_dequant_expert_f16(f.gu_type, t, t + f.up_off, FF, NE, gd, f.d_type, t + f.down_off, dd,
                                                 s);
                    }
                }
                if (!Gemm::f16_groups(x16, dq_gu, 2 * FF * NE, gu, hb.data() + e0, g, 2 * FF, NE, s))
                    k::xmx_gemm_grouped(x16, dq_gu, 2 * FF * NE, gu, bounds + e0, g, r, 2 * FF, NE, s);
                swiglu(gu + r0 * 2 * FF, hh + r0 * FF, nr, FF, !mixed, s);
                k::f32_to_f16_bulk(hh + r0 * FF, h16 + r0 * FF, nr * FF, s);
                if (!Gemm::f16_groups(h16, dq_d, NE * FF, dn, hb.data() + e0, g, NE, FF, s))
                    k::xmx_gemm_grouped(h16, dq_d, NE * FF, dn, bounds + e0, g, r, NE, FF, s);
            }
        };
        auto timed = [&](const std::function<void()>& run) {
            const auto t0 = Clock::now();
            run();
            strata::gpu::stream_sync(s);
            return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        };
        // the GPU clocked up before the first timing: both sides at the fewest rows for 50 ms at least
        {
            const auto t0 = Clock::now();
            do {
                run8(RS[0], GE);
                run16(RS[0], GE);
                strata::gpu::stream_sync(s);
            } while (Clock::now() - t0 < std::chrono::milliseconds(50));
        }
        int64_t won = 0;
        double won_ratio = 0;
        std::string said;
        for (const int64_t r : RS) {
            if (r * G16 > RMAX) break;
            const int64_t n = experts_at(r);
            hb.assign((size_t) n + 1, 0);
            for (int64_t e = 0; e <= n; ++e) hb[(size_t) e] = (int32_t) (e * r);
            std::vector<int32_t> rb((size_t) G8 + 1);
            for (int e = 0; e <= G8; ++e) rb[(size_t) e] = (int32_t) (e * r);
            if (!strata::gpu::copy(bounds, hb.data(), hb.size() * 4) || !strata::gpu::copy(rbounds, rb.data(), rb.size() * 4))
                return give_up("its row bounds could not be set");
            // each side once (a new launch shape), then five in turn, the least of each
            timed([&] { run8(r, n); });
            timed([&] { run16(r, n); });
            double us8 = 1e30, us16 = 1e30;
            for (int t = 0; t < 5; ++t) {
                us8 = std::min(us8, timed([&] { run8(r, n); }));
                us16 = std::min(us16, timed([&] { run16(r, n); }));
            }
            us8 /= (double) n;   // an expert's
            us16 /= (double) n;
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%lld: %.1f/%.1f", said.empty() ? "" : ", ", (long long) r, us8, us16);
            said += buf;
            const double ratio = us8 / (us16 * 1.05);   // above 1: FP16 wins this count
            if (ratio > 1.0) {
                int64_t below = 0;
                if (won > 0) {   // where the two lines cross between won and r (log of the rows, log of the ratio)
                    const double t = -std::log(won_ratio) / (std::log(ratio) - std::log(won_ratio));
                    below = (int64_t) std::llround((double) won * std::pow((double) r / (double) won, t));
                }
                std::fprintf(stderr, "xestrata: the experts' prompt products: int8 below %lld rows an expert, FP16 from "
                                     "there (us an expert int8/FP16 at rows %s)\n", (long long) below, said.c_str());
                return below;
            }
            won = r;
            won_ratio = ratio;
        }
        std::fprintf(stderr, "xestrata: the experts' prompt products: int8 at every size (us an expert int8/FP16 at rows %s)\n",
                     said.c_str());
        return INT64_MAX;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the experts' prompt products: %s; %s\n", e.what(),
                     fallback == 0 ? "the FP16 path" : "the int8 products");
        return fallback;
    }
}

int64_t int8_rows_below(const kernels::cpu::NativeFmt& f, size_t blob, int64_t fallback, const ExpertLaunches& launches) {
    // kept a GPU, format and launch shape: a model opened again in the same process (an image's text encoder, after
    // each image) takes what the first open measured
    struct Kept {
        std::mutex m;
        std::map<std::tuple<int, int, int, int64_t, int64_t, size_t, size_t, size_t, int64_t, int, int, bool>, int64_t> of;
    };
    static core::PerDevice<std::shared_ptr<Kept>> per_device;
    Kept& kept = *per_device.get(core::Runtime::get().device(), [] { return std::make_shared<Kept>(); });
    const auto key = std::make_tuple(f.gu_type, f.up_type, f.d_type, f.n_ff, f.n_embd, f.up_off, f.down_off, blob,
                                     launches.n_expert, launches.int8_group, launches.f16_group, launches.int8_gathers);
    std::lock_guard<std::mutex> lock(kept.m);
    if (const auto it = kept.of.find(key); it != kept.of.end()) return it->second;
    const int64_t below = time_int8_rows_below(f, blob, fallback, launches);
    kept.of.emplace(key, below);
    return below;
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n > 0) Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] = (int32_t) i[0]; });
}

}  // namespace strata::prefill::mmq
