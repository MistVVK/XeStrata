// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/gr.cpp - the Xe ports of Strata's src/kernels/cuda/gr.cu (the gated residual / hyper-connection, `gr_read`
// and `gr_write`), Strata's src/kernels/cuda/native_gr_norm.cu and Strata's src/kernels/cuda/native_gr_postops.cu.
//
// See the CUDA files for the shape of each projection and the history behind it.  A warp is a sub-group of 32; CUDA's
// shuffle-down sum followed by a lane-0 broadcast is an xor butterfly whose lane-0 value is broadcast, which adds in
// the same order.  The native kernels follow ggml-cuda at llama.cpp 3cf03257 (MIT License, Copyright (c) 2023-2026
// The ggml authors, see third_party/main/ggml/LICENSE); CUDA compiled them with --use_fast_math, here the math is precise.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WARP = 32;
bool fp32_activations = false;
bool native_mmvf = false;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

inline float activation_f32(float x) { return x; }
inline float activation_f32(uint16_t x) { return f32_from_bf16(x); }
inline void store_activation(float* dst, int i, float x) { dst[i] = x; }
inline void store_activation(uint16_t* dst, int i, float x) { dst[i] = bf16_from_f32(x); }

/// silu and sigmoid exactly as `ref/gr.py` writes them - in FLOAT, not double.
inline float silu_f(float x) { return x / (1.0f + sycl::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

/// CUDA's shuffle-down tree then __shfl(v, 0): lane 0's butterfly value, broadcast.
inline float warp_sumf(const sycl::sub_group& sg, float v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return sycl::group_broadcast(sg, v, 0);
}

/// The FP32 work-group sum, broadcast to every work-item (see block_sumf in the CUDA file for the leading barrier).
inline float block_sumf(const sycl::nd_item<1>& it, float v, const sycl::local_accessor<float, 1>& scratch) {
    const auto g = it.get_group();
    const sycl::sub_group sg = it.get_sub_group();
    sycl::group_barrier(g);
    const int tid = (int) it.get_local_id(0);
    const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
    v = warp_sumf(sg, v);
    if (lane == 0) scratch[warp] = v;
    sycl::group_barrier(g);
    const int nw = ((int) it.get_local_range(0) + 31) >> 5;
    v = (tid < nw) ? scratch[tid] : 0.0f;
    if (warp == 0) v = warp_sumf(sg, v);
    if (tid == 0) scratch[0] = v;
    sycl::group_barrier(g);
    return scratch[0];
}

template <bool FP32_ACT>
void launch_norm(sycl::queue& q, const float* R, const float* w_norm, float eps, int n_embd, int hc, float* xn,
                 uint16_t* xq) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const size_t c = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* Rc = R + c * n_embd;
            float* xnc = xn + c * n_embd;
            uint16_t* xqc = xq + c * n_embd;
            float ss = 0.0f;
            for (int d = tid; d < n_embd; d += THREADS) {
                const float v = Rc[d];
                ss += v * v;
            }
            const float ms = block_sumf(it, ss, scratch) / (float) n_embd;
            const float rs = sycl::rsqrt(ms + eps);
            for (int d = tid; d < n_embd; d += THREADS) {
                const float x = Rc[d] * rs * w_norm[c * n_embd + d];
                xnc[d] = x;
                if constexpr (!FP32_ACT) xqc[d] = bf16_from_f32(x);
            }
        });
    });
}

/// `lo = silu((bf16(xn) @ w_down.T) / hc)`, one work-group per output row with its sub-groups splitting the reduction.
template <typename Activation>
void launch_down(sycl::queue& q, const Activation* xq, const uint16_t* w_down, int hc_dim, int hc_lr, int hc,
                 Activation* lq) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(WARPS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc_lr * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int k = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int nw = WARPS;
            const uint16_t* row = w_down + (size_t) k * hc_dim;
            float acc = 0.0f;
            for (int i = warp * 32 + lane; i < hc_dim; i += nw * 32)
                acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
            acc = warp_sumf(sg, acc);
            if (lane == 0) part[warp] = acc;
            sycl::group_barrier(it.get_group());
            if (warp == 0) {
                float t = (lane < nw) ? part[lane] : 0.0f;
                t = warp_sumf(sg, t);
                if (lane == 0) store_activation(lq, k, silu_f(t / (float) hc));
            }
        });
    });
}

/// `gated[i] = xn[i] * sigmoid(bf16(lo) @ w_up.T)`, one sub-group per output row, lanes striding hc_lr.
template <typename Activation>
void launch_gate(sycl::queue& q, const Activation* lq, const uint16_t* w_up, const float* xn, int hc_dim, int hc_lr,
                 float* gated) {
    const size_t groups = (size_t) ((hc_dim + WARPS - 1) / WARPS);
    q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int i = (int) it.get_group(0) * WARPS + (int) sg.get_group_linear_id();
        if (i >= hc_dim) return;
        const int lane = (int) sg.get_local_linear_id();
        const uint16_t* row = w_up + (size_t) i * hc_lr;
        float acc = 0.0f;
        for (int k = lane; k < hc_lr; k += 32) acc += activation_f32(lq[k]) * f32_from_bf16(row[k]);
        acc = warp_sumf(sg, acc);
        if (lane == 0) gated[i] = xn[i] * sigmoid_f(acc);
    });
}

/// `inject[c] = bf16(xn) @ w_inject[c]`, one sub-group per stream, lanes striding hc_dim.
template <typename Activation>
void launch_inject(sycl::queue& q, const Activation* xq, const uint16_t* w_inject, int hc_dim, int hc, float* inject) {
    q.parallel_for(sycl::nd_range<1>((size_t) 32 * hc, (size_t) 32 * hc), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int c = (int) sg.get_group_linear_id();
        if (c >= hc) return;
        const int lane = (int) sg.get_local_linear_id();
        const uint16_t* row = w_inject + (size_t) c * hc_dim;
        float acc = 0.0f;
        for (int i = lane; i < hc_dim; i += 32) acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
        acc = warp_sumf(sg, acc);
        if (lane == 0) inject[c] = acc;
    });
}

// ---- native GR norm and postops
inline float scale_zero_bias(float x, float scale) { return sycl::fma(scale, x, 0.0f); }

void check_pointer(const void* p, const char* what) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0) throw std::invalid_argument(what);
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
size_t rounded(size_t n) { return (n + THREADS - 1) / THREADS * THREADS; }

template <int BlockSize>
void launch_weighted_rms_norm(sycl::queue& q, const float* input, const float* gamma, float* output, int n_cols,
                              int n_rows, float epsilon) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_rows * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const size_t row_offset = it.get_group(0) * (size_t) n_cols;
            const float* in = input + row_offset;
            const float* ga = gamma + row_offset;
            float* out = output + row_offset;
            float partial = 0.0f;
            for (int col = tid; col < n_cols; col += BlockSize) {
                const float value = in[col];
                partial += value * value;
            }
            // Pinned block_reduce<SUM,BlockSize>: every sub-group repeats the final xor reduction.
            for (int o = 16; o > 0; o >>= 1) partial += sycl::permute_group_by_xor(sg, partial, o);
            const int lane = tid % 32;
            if (lane == 0) sums[tid / 32] = partial;
            sycl::group_barrier(it.get_group());
            partial = 0.0f;
            if (lane < BlockSize / 32) partial = sums[lane];
            for (int o = 16; o > 0; o >>= 1) partial += sycl::permute_group_by_xor(sg, partial, o);
            const float mean = partial / n_cols;
            const float scale = sycl::rsqrt(mean + epsilon);
            for (int col = tid; col < n_cols; col += BlockSize) out[col] = scale * in[col] * ga[col];
        });
    });
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
    auto& q = queue_for(stream);
    // a device that takes no 1024-item work-group sums the long rows in another order
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024(q);
    if (wg == 1024) launch_weighted_rms_norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else if (wg == 512) launch_weighted_rms_norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else launch_weighted_rms_norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon);
    if (!stream) core::Runtime::get().finish(q);
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

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // One table, indexed by the same `k` that assigns the pointers below (see the CUDA file for why).
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),      // 4: lo (FP32 activation experiment)
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr)
        throw core::DeviceError("gr_read: GrWorkspace is not initialised (see gr_workspace_init)");
    if (ws.bytes < gr_workspace_bytes(s))
        throw core::DeviceError("gr_read: GrWorkspace is " + std::to_string(ws.bytes) + " bytes but this geometry needs " +
                                std::to_string(gr_workspace_bytes(s)));
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);
    auto& q = queue_for(stream);
    void* st = &q;   // the kernels below all run on the caller's queue, in order

    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, st);
        bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, st);
        native_gr_down_silu(ws.lo, hc_lr, hc, st);
        bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, st);
        // The existing null-injection contract identifies the final mixer.
        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, st);
    } else if (use_fp32) {
        launch_norm<true>(q, R, w_norm, eps, n_embd, hc, ws.xn, ws.xq);
        launch_down<float>(q, ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo);
        launch_gate<float>(q, ws.lo, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
    } else {
        launch_norm<false>(q, R, w_norm, eps, n_embd, hc, ws.xn, ws.xq);
        launch_down<uint16_t>(q, ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq);
        launch_gate<uint16_t>(q, ws.lq, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
    }
    if (!use_native) {
        const float* gated = ws.gated;
        q.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> id) {
            const int d = (int) id[0];
            float m = 0.0f;
            for (int c = 0; c < hc; ++c) m += gated[(size_t) c * n_embd + d];
            mixed[d] = m / (float) hc;
        });
    }
    // Absent for the final mixer, and then nothing is written.
    if (w_inject != nullptr) {
        if (use_native) bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, st);
        else if (use_fp32) launch_inject<float>(q, ws.xn, w_inject, hc_dim, hc, inject);
        else launch_inject<uint16_t>(q, ws.xq, w_inject, hc_dim, hc, inject);
    }
    if (!stream) core::Runtime::get().finish(q);
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    auto& q = queue_for(stream);
    if (native_mmvf) {
        native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, &q);
    } else {
        const int n_embd = (int) s.n_embd, hc = (int) s.hc;
        const long long n = (long long) hc * n_embd;
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> w(sycl::range<1>((size_t) hc), h);
            h.parallel_for(sycl::nd_range<1>(rounded((size_t) n), THREADS), [=](sycl::nd_item<1> it) {
                // w = 2*sigmoid(inject/hc), computed once per stream in local memory rather than per element.
                const int tid = (int) it.get_local_id(0);
                if (tid < hc) w[tid] = 2.0f * sigmoid_f(inject[tid] / (float) hc);
                sycl::group_barrier(it.get_group());
                const long long i = (long long) it.get_global_id(0);
                if (i >= n) return;
                const int c = (int) (i / n_embd), d = (int) (i % n_embd);
                // every stream adds the SAME block output; only the weight differs per stream
                R_out[i] = R[i] + block_out[d] * w[c];
            });
        });
    }
    if (!stream) core::Runtime::get().finish(q);
}

}  // namespace strata::kernels
