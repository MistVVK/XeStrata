// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/shared_expert.cpp - the Xe port of Strata's src/kernels/cuda/shared_expert.cu: the shared expert
// (h = silu(x W_gate) * (x W_up), then W_down, scaled by the per-token scalar gate sigmoid(x . w)) and the MoE
// block's final combination.  See the CUDA source for the reference and the readings it pins.
//
// The native paths' SwiGLU and sigmoid were CUDA fast-math intrinsics (__fdividef, __expf); here they are the
// precise division and exp.  The scalar gate's warp sums are xor butterflies whose lane 0 adds in the order of
// CUDA's shuffle-down tree and is then broadcast, as CUDA broadcast lane 0.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
bool native_bf16 = false;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

size_t round_up(size_t n, size_t b) { return (n + b - 1) / b * b; }

// silu(gate) * up with the silu in DOUBLE, as ref/moe.py computes it (D: double where the device has FP64, float
// elsewhere; device_caps.hpp)
template <typename D>
void swiglu_t(sycl::queue& q, const float* gate, const float* up, float* out, int n) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const D x = (D) gate[i];
        out[i] = (float) (x / ((D) 1 + sycl::exp(-x))) * up[i];
    });
}
void swiglu(sycl::queue& q, const float* gate, const float* up, float* out, int n) {
    if (xe::has_fp64(q)) swiglu_t<double>(q, gate, up, out, n);
    else swiglu_t<float>(q, gate, up, out, n);
}

// ggml unary op_silu then the gated multiply, in FP32
void native_swiglu(sycl::queue& q, const float* gate, const float* up, float* out, int n) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        out[i] = gate[i] / (1.0f + sycl::exp(-gate[i])) * up[i];
    });
}

template <typename D>
inline D warp_sum_d(const sycl::sub_group& sg, D v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return sycl::group_broadcast(sg, v, 0);
}

// sigmoid(dot(x, w)) over BF16 operands, accumulated in double (float without FP64): one work-group of 256, then a
// two-level warp sum
template <typename D>
void scalar_gate_t(sycl::queue& q, const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd) {
    constexpr int GATE_THREADS = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> scratch(sycl::range<1>(GATE_THREADS / WARP), h);
        h.parallel_for(sycl::nd_range<1>(GATE_THREADS, GATE_THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            D acc = 0;
            for (int i = tid; i < n_embd; i += GATE_THREADS)
                acc += (D) f32_from_bf16(x_bf16[i]) * (D) f32_from_bf16(w_bf16[i]);
            sycl::group_barrier(it.get_group());
            const int lane = tid & 31, warp = tid >> 5;
            acc = warp_sum_d(sg, acc);
            if (lane == 0) scratch[warp] = acc;
            sycl::group_barrier(it.get_group());
            constexpr int nw = GATE_THREADS / WARP;
            if (warp == 0) {
                acc = tid < nw ? scratch[tid] : (D) 0;
                acc = warp_sum_d(sg, acc);
                if (tid == 0) out[0] = (float) ((D) 1 / ((D) 1 + sycl::exp(-acc)));
            }
        });
    });
}
void scalar_gate(sycl::queue& q, const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd) {
    if (xe::has_fp64(q)) scalar_gate_t<double>(q, x_bf16, w_bf16, out, n_embd);
    else scalar_gate_t<float>(q, x_bf16, w_bf16, out, n_embd);
}

void scale_rows(sycl::queue& q, float* out, const float* g, int n, int n_tok) {
    q.parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        out[id[0] * n + id[1]] *= g[id[0]];
    });
}

// the native gate's sigmoid and the row scale in one launch: each work-item takes its row's sigmoid of the raw gate
// and scales four columns where the rows are 16-byte aligned (upstream 822251be)
void sigmoid_scale_rows(sycl::queue& q, float* out, const float* g, int n, int n_tok) {
    if (n % 4 == 0 && (reinterpret_cast<uintptr_t>(out) & 15u) == 0) {
        const int n4 = n / 4;
        auto* out4 = reinterpret_cast<sycl::float4*>(out);
        q.parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n4), [=](sycl::id<2> id) {
            const float gt = 1.0f / (1.0f + sycl::exp(-g[id[0]]));
            out4[id[0] * n4 + id[1]] *= gt;
        });
        return;
    }
    q.parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        const float gt = 1.0f / (1.0f + sycl::exp(-g[id[0]]));
        out[id[0] * n + id[1]] *= gt;
    });
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool shared_expert_native_bf16() { return native_bf16; }

bool shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream, bool gate_later, bool g_ready) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    auto& q = queue_for(stream);
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    // gate and up in one launch where the layout allows (upstream b08cf3e1; STRATA_LFUSE_PAIR=0: two)
    static const bool pair_on = [] {
        const char* v = std::getenv("STRATA_LFUSE_PAIR");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (!(pair_on && nw.gate_type == nw.up_type &&
          native_mmvq_pair(nw.gate_type, nw.gate_data, nw.up_data, nw.q8_1, gate, up, (int) n_embd, (int) n_ff, n_tok,
                           stream))) {
        native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
        native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    }
    native_swiglu(q, gate, up, gate, (int) (n_ff * n_tok));
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    static const bool batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && g_ready) {
        // the router's launch computed the raw gates
    } else if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), the sigmoid in the scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16) bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
            else scalar_gate(q, x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd);
        }
    }
    if (native_bf16 && gate_later) return true;                             // the combine scales the rows
    if (native_bf16) sigmoid_scale_rows(q, out, g, (int) n_embd, n_tok);   // g holds the raw gates
    else scale_rows(q, out, g, (int) n_embd, n_tok);                       // scalar_gate wrote the sigmoids
    return false;
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = round_up((uint64_t) n_ff * 4, 16);
    const uint64_t q0 = round_up((uint64_t) (n_ff / 32) * 34, 16);
    const uint64_t qk = round_up((uint64_t) (n_ff / 256) * 292, 16);
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr)
        throw core::DeviceError("shared_expert: scratch is null; the caller owns it (see shared_expert_scratch_bytes)");
    if (!native_down && down_form.act_kind == 1 && n_ff % 256 != 0)
        throw core::DeviceError("shared_expert: the down weight wants Q8_K but n_ff " + std::to_string(n_ff) +
                                " is not a multiple of 256; Q8_K is structurally impossible here");
    auto& q = queue_for(stream);
    // Every step below is enqueued on the caller's queue without waiting; a null stream waits once at the end.
    void* s = &q;

    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = round_up((uint64_t) n_ff * 4, 16);
    const uint64_t q0 = round_up((uint64_t) (n_ff / 32) * 34, 16);
    const uint64_t qk = round_up((uint64_t) (n_ff / 256) * 292, 16);
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // which activation this projection wants, read from its own form (SForm::act_kind)
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, s);
        else if (f.act_kind == 1) s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, s);
        else s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, s);
    };

    if (native_gate || native_up) native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, s);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, s);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, s);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection) native_swiglu(q, gate, up, gate, (int) n_ff);
    else swiglu(q, gate, up, gate, (int) n_ff);

    // down, with the intermediate quantized to the down weight's own activation contract
    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, s);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, s);
    } else if (down_form.act_kind == 1) {
        quantize_q8_K(gate, h_q8k, n_ff, s);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, s);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    // the per-token scalar gate, from the ORIGINAL hidden state, then the multiply
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, s);
        sigmoid_scale_rows(q, out, g, (int) n_embd, 1);
    } else {
        scalar_gate(q, x_bf16, gate_inp_bf16, g, (int) n_embd);
        scale_rows(q, out, g, (int) n_embd, 1);
    }
    if (!stream) core::Runtime::get().finish(q);
}

namespace {
// double, in the reference's order (float without FP64); the shared output is added plain
template <typename D>
sycl::event moe_combine_t(sycl::queue& q, const float* parts, const float* weights, const float* shared, float* y, int n,
                          int kk) {
    return q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int j = (int) id[0];
        D acc = 0;
        for (int i = 0; i < kk; ++i) acc += (D) weights[i] * (D) parts[(size_t) i * n + j];
        if (shared) acc += (D) shared[j];
        y[j] = (float) acc;
    });
}
}  // namespace

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated
    if (k > 64) throw core::DeviceError("moe_combine: k = " + std::to_string(k) + " exceeds 64");
    const int n = (int) n_embd, kk = (int) k;
    auto& q = queue_for(stream);
    const auto e = xe::has_fp64(q) ? moe_combine_t<double>(q, parts, weights, shared, y, n, kk)
                                   : moe_combine_t<float>(q, parts, weights, shared, y, n, kk);
    if (!stream) core::Runtime::get().wait(e, "moe_combine");
}

}  // namespace strata::kernels
