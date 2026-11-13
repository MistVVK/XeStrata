// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
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
#include "strata/kernels/window_limits.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_shared_expert
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {


sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

size_t round_up(size_t n, size_t b) { return (n + b - 1) / b * b; }

// ggml unary op_silu then the gated multiply, in FP32
void native_swiglu(sycl::queue& q, const float* gate, const float* up, float* out, int n) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        out[i] = gate[i] / (1.0f + sycl::exp(-gate[i])) * up[i];
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


bool shared_expert_multi(int n_tok, const float* x, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream, bool gate_later, bool g_ready, const void* x_q8_1) {
    if (n_tok < 1 || n_tok > kWindowMaxT || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..16 tokens, native weights, scratch and a stream");
    auto& q = queue_for(stream);
    const void* xq = x_q8_1;
    if (xq == nullptr) {
        native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
        xq = nw.q8_1;
    }
    // gate and up in one launch where the layout allows (upstream b08cf3e1; STRATA_LFUSE_PAIR=0: two)
    static const bool pair_on = [] {
        const char* v = strata::env_get("LFUSE_PAIR");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (!(pair_on && nw.gate_type == nw.up_type &&
          native_mmvq_pair(nw.gate_type, nw.gate_data, nw.up_data, xq, gate, up, (int) n_embd, (int) n_ff, n_tok,
                           stream))) {
        native_mmvq(nw.gate_type, nw.gate_data, xq, gate, (int) n_embd, (int) n_ff, n_tok, stream);
        native_mmvq(nw.up_type, nw.up_data, xq, up, (int) n_embd, (int) n_ff, n_tok, stream);
    }
    // the SwiGLU and its q8_1 image in one launch (upstream be7889ed); STRATA_VERIFY_QDEDUP=0: two
    static const bool fused_q8 = [] {
        const char* v = strata::env_get("VERIFY_QDEDUP");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (fused_q8) {
        native_swiglu_q8_1(gate, up, nw.q8_1, (int) n_ff, n_tok, stream);
    } else {
        native_swiglu(q, gate, up, gate, (int) (n_ff * n_tok));
        native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    }
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    static const bool batch = [] { const char* v = strata::env_get("DEC_BATCH"); return v == nullptr || std::strtol(v, nullptr, 10) != 0; }();
    if (g_ready) {
        // the router's launch computed the raw gates
    } else if (batch && n_tok > 1) {   // one gemv for all rows (outputs identical), the sigmoid in the scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
    } else {
        for (int t = 0; t < n_tok; ++t) bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
    }
    if (gate_later) return true;                             // the combine scales the rows
    sigmoid_scale_rows(q, out, g, (int) n_embd, n_tok);      // g holds the raw gates
    return false;
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = round_up((uint64_t) n_ff * 4, 16);
    const uint64_t q0 = round_up((uint64_t) (n_ff / 32) * 34, 16);
    const uint64_t qk = round_up((uint64_t) (n_ff / 256) * 292, 16);
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if (!x_f32) throw std::invalid_argument("shared_expert requires the unrounded x_f32 (its scalar gate reads it)");
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

    uint8_t* p = reinterpret_cast<uint8_t*>(scratch);
    const uint64_t a = round_up((uint64_t) n_ff * 4, 16);
    const uint64_t q0 = round_up((uint64_t) (n_ff / 32) * 34, 16);
    const uint64_t qk = round_up((uint64_t) (n_ff / 256) * 292, 16);
    float* gate = reinterpret_cast<float*>(p);
    float* up = reinterpret_cast<float*>(p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = reinterpret_cast<float*>(p + a * 2 + q0 + qk);

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
    native_swiglu(q, gate, up, gate, (int) n_ff);

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
    bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, s);
    sigmoid_scale_rows(q, out, g, (int) n_embd, 1);
    if (!stream) core::Runtime::get().finish(q);
}

}  // namespace strata::kernels
