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
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
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

// From how many rows an expert the FP16 path beats the int8 products on this GPU, timed on 8 experts of a layer's
// format (its types and shape, synthetic blocks: any bits time alike) at 16 to 1024 rows an expert (256 where the
// device cannot hold the larger buffers), in the calls a chunk makes: the int8 products from the blocks, against the
// experts dequantized to FP16 and their grouped products.  The geometric middle of the last row count int8 won and the
// first it lost; INT64_MAX where it won at every count, 0 where it lost at 16.  The int8 products were a fixed choice
// before (FP16 where Intel's XMX take it): B70, Qwen3.8 IQ3_XXS, the int8 form 1K 923 -> 1074 tok/s, 4K 1594 -> 1604,
// 32K 2526 -> 2075.  Where the timing cannot run, `fallback`.  The int8 time grows with the rows and the FP16 one
// hardly does, so a type whose int8 products are cheap crosses later: B70, IQ3_S's IQ4_XS and UD-Q4_K_XL's K-quants
// still won at 256 rows (487/609, 592/669 us), and a 32K chunk, about 600 rows an expert, ran its int8 products: 32K
// prompt 2,318 -> 1,896 and 2,125 -> 1,721 tok/s.
static int64_t time_int8_rows_below(const kernels::cpu::NativeFmt& f, size_t blob, int64_t fallback) {
    namespace k = strata::kernels;
    using Clock = std::chrono::steady_clock;
    constexpr int G = 8;
    constexpr int64_t RS[] = {16, 32, 64, 128, 256, 512, 1024};
    const int64_t FF = f.n_ff, NE = f.n_embd;
    // a layer whose gate and up differ in type (k::cpu::native_mixed_gu): gate and up a product each, as PlainModel's
    // prompt takes them
    const bool mixed = k::cpu::native_mixed_gu(f);
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
    // the row buffers for the most rows timed: 1024 an expert, else 256 (a device short of memory)
    int64_t r_top = 1024;
    if (void* probe = strata::gpu::alloc_device((size_t) r_top * G * (NE * 10 + FF * 13 + 4) + q8_bytes(r_top * G, NE) * 2))
        strata::gpu::free(probe);
    else
        r_top = 256;
    const int64_t RMAX = r_top * G;
    auto* w = static_cast<uint8_t*>(dev((size_t) G * blob));
    auto* dq_gu = static_cast<uint16_t*>(dev((size_t) G * 2 * FF * NE * 2));
    auto* dq_d = static_cast<uint16_t*>(dev((size_t) G * NE * FF * 2));
    auto* x16 = static_cast<uint16_t*>(dev((size_t) RMAX * NE * 2));
    auto* h16 = static_cast<uint16_t*>(dev((size_t) RMAX * FF * 2));
    auto* gu = static_cast<float*>(dev((size_t) RMAX * 2 * FF * 4));
    auto* dn = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    auto* zx = static_cast<float*>(dev((size_t) RMAX * NE * 4));
    void* xq = dev(q8_bytes(RMAX, NE) * 2);
    void* hq = dev(q8_bytes(RMAX, FF) * 2);
    auto* bounds = static_cast<int32_t*>(dev((size_t) (G + 1) * 4));
    auto* ident = static_cast<int32_t*>(dev((size_t) RMAX * 4));
    auto* wtab = static_cast<const uint8_t**>(dev((size_t) G * sizeof(void*)));
    auto give_up = [&](const char* why) {
        std::fprintf(stderr, "xestrata: timing the experts' prompt products: %s (%s); %s\n", why, strata::gpu::last_error(),
                     fallback == 0 ? "the FP16 path" : "the int8 products");
        return fallback;
    };
    for (const void* p : {(const void*) w, (const void*) dq_gu, (const void*) dq_d, (const void*) x16, (const void*) h16,
                          (const void*) gu, (const void*) dn, (const void*) zx, (const void*) xq, (const void*) hq, (const void*) bounds,
                          (const void*) ident, (const void*) wtab})
        if (p == nullptr) return give_up("no device memory for it");
    try {
        void* s = nullptr;   // the compute queue
        const uint8_t* tab[G];
        for (int e = 0; e < G; ++e) tab[e] = w + (size_t) e * blob;
        if (!strata::gpu::memset_async(w, 0x11, (size_t) G * blob, s) || !strata::gpu::memset_async(zx, 0, (size_t) RMAX * NE * 4, s) ||
            !strata::gpu::memset_async(x16, 0, (size_t) RMAX * NE * 2, s) || !strata::gpu::memset_async(h16, 0, (size_t) RMAX * FF * 2, s) ||
            !strata::gpu::copy(static_cast<void*>(wtab), static_cast<const void*>(tab), sizeof tab))
            return give_up("its buffers could not be set");
        quantize(zx, nullptr, xq, f.gu_type, NE, NE, RMAX, s);
        quantize(zx, nullptr, hq, f.d_type, FF, FF, RMAX, s);
        iota(ident, RMAX, s);
        Context ctx;
        auto time_us = [&](const std::function<void()>& run) {
            run();
            strata::gpu::stream_sync(s);
            double best = 1e30;
            for (int t = 0; t < 3; ++t) {
                const auto t0 = Clock::now();
                for (int i = 0; i < 5; ++i) run();
                strata::gpu::stream_sync(s);
                best = std::min(best, std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / 5);
            }
            return best;
        };
        int64_t won = 0;
        std::string said;
        for (const int64_t r : RS) {
            if (r > r_top) break;
            std::vector<int32_t> hb(G + 1);
            for (int e = 0; e <= G; ++e) hb[(size_t) e] = (int32_t) (e * r);
            if (!strata::gpu::copy(bounds, hb.data(), hb.size() * 4)) return give_up("its row bounds could not be set");
            const int64_t R = G * r;
            const double us8 = time_us([&] {
                Product a;
                a.type = f.gu_type; a.w_rows = 2 * FF; a.w_cols = NE; a.n = G; a.xq = xq; a.bounds = bounds; a.ids = ident;
                a.total_rows = R; a.max_rows = r; a.rows = R; a.dst = gu; a.ld_dst = 2 * FF; a.wtab = wtab; a.woff = 0;
                if (mixed) {
                    a.w_rows = FF;
                    ctx.run(a, s);
                    a.type = f.up_type; a.dst = gu + FF; a.woff = f.up_off;
                }
                ctx.run(a, s);
                Product b;
                b.type = f.d_type; b.w_rows = NE; b.w_cols = FF; b.n = G; b.xq = hq; b.bounds = bounds; b.ids = ident;
                b.total_rows = R; b.max_rows = r; b.rows = R; b.dst = dn; b.ld_dst = NE; b.wtab = wtab; b.woff = f.down_off;
                ctx.run(b, s);
            });
            const double us16 = time_us([&] {
                for (int e = 0; e < G; ++e) {
                    uint16_t* gd = dq_gu + (size_t) e * 2 * FF * NE;
                    uint16_t* dd = dq_d + (size_t) e * NE * FF;
                    if (mixed) {
                        k::dequant_f16(f.gu_type, tab[e], 0, FF, NE, gd, s);
                        k::dequant_f16(f.up_type, tab[e] + f.up_off, 0, FF, NE, gd + FF * NE, s);
                        k::dequant_f16(f.d_type, tab[e] + f.down_off, 0, NE, FF, dd, s);
                    } else {
                        k::iq_dequant_expert_f16(f.gu_type, tab[e], tab[e] + f.up_off, FF, NE, gd, f.d_type,
                                                 tab[e] + f.down_off, dd, s);
                    }
                }
                if (!Gemm::f16_groups(x16, dq_gu, 2 * FF * NE, gu, hb.data(), G, 2 * FF, NE, s))
                    k::xmx_gemm_grouped(x16, dq_gu, 2 * FF * NE, gu, bounds, G, r, 2 * FF, NE, s);
                if (!Gemm::f16_groups(h16, dq_d, NE * FF, dn, hb.data(), G, NE, FF, s))
                    k::xmx_gemm_grouped(h16, dq_d, NE * FF, dn, bounds, G, r, NE, FF, s);
            });
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%lld: %.0f/%.0f", said.empty() ? "" : ", ", (long long) r, us8, us16);
            said += buf;
            // FP16 only when it is 3% faster: above the crossover the int8 group scratch no longer bounds the chunk, so
            // --prefill auto takes a smaller one.  The RTX 4070's 1024 rows are within 1% either way (int8 1972-1982 us,
            // FP16 1957-1976), and a run that took FP16 there read a 32K serve prompt in 8192-token chunks: 1,687 -> 1,162 tok/s.
            if (us16 * 1.03 < us8) {
                const int64_t below = won == 0 ? 0 : (int64_t) std::llround(std::sqrt((double) won * (double) r));
                std::fprintf(stderr, "xestrata: the experts' prompt products: int8 below %lld rows an expert, FP16 from "
                                     "there (us int8/FP16 at rows %s)\n", (long long) below, said.c_str());
                return below;
            }
            won = r;
        }
        std::fprintf(stderr, "xestrata: the experts' prompt products: int8 at every size (us int8/FP16 at rows %s)\n",
                     said.c_str());
        return INT64_MAX;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the experts' prompt products: %s; %s\n", e.what(),
                     fallback == 0 ? "the FP16 path" : "the int8 products");
        return fallback;
    }
}

int64_t int8_rows_below(const kernels::cpu::NativeFmt& f, size_t blob, int64_t fallback) {
    // kept a GPU and format: a model opened again in the same process (an image's text encoder, after each image)
    // takes what the first open measured
    struct Kept {
        std::mutex m;
        std::map<std::tuple<int, int, int, int64_t, int64_t, size_t, size_t, size_t>, int64_t> of;
    };
    static core::PerDevice<std::shared_ptr<Kept>> per_device;
    Kept& kept = *per_device.get(core::Runtime::get().device(), [] { return std::make_shared<Kept>(); });
    const auto key = std::make_tuple(f.gu_type, f.up_type, f.d_type, f.n_ff, f.n_embd, f.up_off, f.down_off, blob);
    std::lock_guard<std::mutex> lock(kept.m);
    if (const auto it = kept.of.find(key); it != kept.of.end()) return it->second;
    const int64_t below = time_int8_rows_below(f, blob, fallback);
    kept.of.emplace(key, below);
    return below;
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n > 0) Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] = (int32_t) i[0]; });
}

}  // namespace strata::prefill::mmq
