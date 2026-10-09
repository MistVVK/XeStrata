// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/moe_mmq.cpp - see include/strata/prefill/moe_mmq.hpp: the prompt path's MMQ steps over
// strata::kernels' int8 products (iq_mmq.hpp).
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_mmq.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace strata::prefill::mmq {
namespace {
sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
}  // namespace

bool built() { return kernels::iq_mmq_usable(core::Runtime::get().compute()); }
bool supported(int t) { return kernels::iq_mmq_type_ok(t); }
size_t matrix_bytes(int t, int64_t rows, int64_t cols) { return (size_t) rows * kernels::iq_mmq_row_bytes(t, cols); }
size_t q8_bytes(int64_t rows, int64_t cols) { return kernels::iq_mmq_act_bytes(rows, cols); }

void quantize(const float* x, const int32_t* ids, void* xq, int, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows > 0) kernels::iq_mmq_quantize(Q(stream), x, ids, xq, cols, ld, rows);
}

Context::Context() {}
Context::~Context() {}
void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    kernels::iq_mmq_grouped(Q(stream), p.type, p.w, p.expert_bytes, p.w_rows, p.w_cols, p.n, p.xq, p.total_rows,
                            p.bounds, p.max_rows, p.ids, p.dst, p.ld_dst);
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

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (rows * n_ff)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / n_ff, k = i % n_ff;
        const float* row = gu + r * 2 * n_ff;
        const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
        h[i] = g / (1.0f + sycl::exp(-g)) * u;
    });
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n > 0) Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] = (int32_t) i[0]; });
}

}  // namespace strata::prefill::mmq
