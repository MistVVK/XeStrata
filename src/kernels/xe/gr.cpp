// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/gr.cpp - the Xe ports of Strata's src/kernels/cuda/gr.cu (the gated residual / hyper-connection, `gr_read`
// and `gr_write`), Strata's src/kernels/cuda/native_gr_norm.cu and Strata's src/kernels/cuda/native_gr_postops.cu.
//
// See the CUDA files for the shape of each projection and the history behind it.  A warp is a sub-group of 32; CUDA's
// shuffle-down sum followed by a lane-0 broadcast is an xor butterfly whose lane-0 value is broadcast, which adds in
// the same order.  The native kernels follow ggml-cuda at llama.cpp 3cf03257 (MIT License, Copyright (c) 2023-2026
// The ggml authors, see third_party/main/ggml/LICENSE); CUDA compiled them with --use_fast_math, here the math is precise.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/native_qsa.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_gr
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

/// sigmoid exactly as `ref/gr.py` writes it - in FLOAT, not double.
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// ---- native GR norm and postops
inline float scale_zero_bias(float x, float scale) { return sycl::fma(scale, x, 0.0f); }

void check_pointer(const void* p, const char* what) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0) throw std::invalid_argument(what);
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}

}  // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    const char* msg = "native GR RMSNorm requires non-null four-byte aligned pointers";
    check_pointer(input, msg);
    check_pointer(gamma, msg);
    check_pointer(output, msg);
    // a gamma row each row: the engines' one RMSNorm with as many groups as rows (native_qsa.hpp), pinned: the sums
    // of native_qsa_rms_norm_weighted, as before
    rms_norm_rows(input, gamma, n_rows, output, nullptr, n_cols, n_rows, epsilon, stream, true);
}

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo, "native GR postops require non-null four-byte aligned pointers");
    const float scale = 1.0f / float(hc);
    auto& q = queue_for(stream);
    q.parallel_for(sycl::range<1>((size_t) hc_lr), [=](sycl::id<1> i) {
        const float x = scale_zero_bias(lo[i], scale);
        lo[i] = x / (1.0f + sycl::exp(-x));
    });
    if (!stream) core::Runtime::get().finish(q);
}

void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream) {
    check_shape(n_embd, hc);
    const char* msg = "native GR postops require non-null four-byte aligned pointers";
    check_pointer(xn, msg); check_pointer(gate, msg); check_pointer(mixed, msg);
    const float scale = 1.0f / float(hc);
    auto& q = queue_for(stream);
    q.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> id) {
        const size_t d = id[0];
        float sum = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const size_t i = size_t(c) * n_embd + d;
            const float x = xn[i], w = sigmoid_f(gate[i]);
            const float product = x * w;
            gate[i] = product;
            if (fused_layer) sum = sycl::fma(x, w, sum);
            else sum = c == 0 ? product : sum + product;
        }
        mixed[d] = fused_layer ? scale * sum : scale_zero_bias(sum, scale);
    });
    if (!stream) core::Runtime::get().finish(q);
}

void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    check_shape(n_embd, hc);
    const char* msg = "native GR postops require non-null four-byte aligned pointers";
    check_pointer(residual, msg); check_pointer(block_out, msg); check_pointer(inject, msg); check_pointer(output, msg);
    const float scale = 1.0f / float(hc);
    auto& q = queue_for(stream);
    q.parallel_for(sycl::range<1>((size_t) n_embd * hc), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int c = int(i / n_embd), d = int(i % n_embd);
        const float weight = scale_zero_bias(sigmoid_f(scale_zero_bias(inject[c], scale)), 2.0f);
        // Exact residual/output alias is supported; no other work-item reads residual[i].
        output[i] = sycl::fma(block_out[d], weight, residual[i]);
    });
    if (!stream) core::Runtime::get().finish(q);
}

void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream) {
    check_shape(n_embd, hc);
    const char* msg = "native GR postops require non-null four-byte aligned pointers";
    check_pointer(residual, msg); check_pointer(block_out, msg); check_pointer(inject, msg); check_pointer(output, msg);
    const float scale = 1.0f / float(hc);
    const size_t hc_dim = (size_t) n_embd * hc;
    auto& q = queue_for(stream);
    q.parallel_for(sycl::range<2>((size_t) n_tok, hc_dim), [=](sycl::id<2> id) {
        const size_t t = id[0], i = id[1];
        const int c = int(i / n_embd), d = int(i % n_embd);
        const float weight = scale_zero_bias(sigmoid_f(scale_zero_bias(inject[t * hc + c], scale)), 2.0f);
        output[t * hc_dim + i] = sycl::fma(block_out[t * n_embd + d], weight, residual[t * hc_dim + i]);
    });
    if (!stream) core::Runtime::get().finish(q);
}

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // One table, indexed by the same `k` that assigns the pointers below (see the CUDA file for why).
    const size_t sz[3] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(float),                // 1: gated
        (size_t) s.hc_lr * sizeof(float),      // 2: lo
    };
    size_t al[3], bytes = 0;
    for (int k = 0; k < 3; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[3];
        for (int k = 0; k < 3; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.gated = (float*) ptr[1];
        out.lo = (float*) ptr[2];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.gated == nullptr || ws.lo == nullptr)
        throw core::DeviceError("gr_read: GrWorkspace is not initialised (see gr_workspace_init)");
    if (ws.bytes < gr_workspace_bytes(s))
        throw core::DeviceError("gr_read: GrWorkspace is " + std::to_string(ws.bytes) + " bytes but this geometry needs " +
                                std::to_string(gr_workspace_bytes(s)));
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);
    auto& q = queue_for(stream);
    void* st = &q;   // the kernels below all run on the caller's queue, in order

    if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
        throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
    native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, st);
    bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, st);
    native_gr_down_silu(ws.lo, hc_lr, hc, st);
    bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, st);
    // The existing null-injection contract identifies the final mixer.
    native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, st);
    // Absent for the final mixer, and then nothing is written.
    if (w_inject != nullptr) bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, st);
    if (!stream) core::Runtime::get().finish(q);
}

void gr_write_multi(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
                    int n_tok, void* stream) {
    if (n_tok <= 0 || s.n_embd <= 0 || s.hc <= 0) return;
    static const bool per_token = [] {
        const char* v = strata::env_get("NO_MULTI_GR");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    if (n_tok > 1 && !per_token) {
        native_gr_post_multi(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, n_tok, stream);
        return;
    }
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    for (int t = 0; t < n_tok; ++t)
        gr_write(R + t * hc_dim, block_out + (size_t) t * s.n_embd, inject + (size_t) t * s.hc, s, R_out + t * hc_dim,
                 stream);
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    auto& q = queue_for(stream);
    native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, &q);
    if (!stream) core::Runtime::get().finish(q);
}

}  // namespace strata::kernels
