// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/moe_mmq.cpp - see include/strata/prefill/moe_mmq.hpp: the prompt path's MMQ steps over
// strata::kernels' int8 products (iq_mmq.hpp).
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/swiglu_clamp.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU prefill_moe_mmq
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>

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

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n > 0) Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] = (int32_t) i[0]; });
}

}  // namespace strata::prefill::mmq
