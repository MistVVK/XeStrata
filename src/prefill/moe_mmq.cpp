// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/moe_mmq.cpp - see include/strata/prefill/moe_mmq.hpp: the prompt path's MMQ steps over
// strata::kernels' int8 products (iq_mmq.hpp).
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_mmq.hpp"

#include <sycl/sycl.hpp>

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
    q.memcpy(gu_dst, gate, gu_half_bytes);
    q.memcpy(static_cast<uint8_t*>(gu_dst) + gu_half_bytes, up, gu_half_bytes);
    q.memcpy(d_dst, down, d_bytes);
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
