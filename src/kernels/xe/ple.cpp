// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/ple.cpp - the Xe port of Strata's src/kernels/cuda/ple.cu: the PLE block's GPU half (see ple.hpp).
//
// The legacy arithmetic follows the captured CPU ggml graph, as in CUDA: `rms_norm` accumulates (double)(x*x) with
// the product rounded in f32 first (ggml_compute_forward_rms_norm_f32), `silu` is x / (1 + exp(-x)) in f32, the
// gate's sqrt and sigmoid are f32.  Work shapes are the CUDA kernels'; the double block sum's shuffle-down tree is an
// xor butterfly whose lane 0 adds in the same order.  The opt-in native paths are unchanged in structure.
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARP = 32;
bool native_bf16 = false;
bool native_postops = false;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

inline uint16_t bf16_bits(float f) {
    uint32_t i = sycl::bit_cast<uint32_t>(f);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}
inline float bf16_float(uint16_t h) { return sycl::bit_cast<float>((uint32_t) h << 16); }
inline float silu_f(float x) { return x / (1.0f + sycl::exp(-x)); }

// D in the kernels below: double where the device has FP64 (the captured graph's sums), float elsewhere
// (device_caps.hpp)
template <typename D>
inline D warp_sum_d(const sycl::sub_group& sg, D v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}
template <typename D>
inline D block_sum(const sycl::nd_item<1>& it, D v, D* scratch) {
    const sycl::sub_group sg = it.get_sub_group();
    sycl::group_barrier(it.get_group());
    const int tid = (int) it.get_local_id(0), lane = tid & 31, warp = tid >> 5;
    v = warp_sum_d(sg, v);
    if (lane == 0) scratch[warp] = v;
    sycl::group_barrier(it.get_group());
    const int nw = (THREADS + 31) >> 5;
    v = (tid < nw) ? scratch[tid] : (D) 0;
    if (warp == 0) v = warp_sum_d(sg, v);
    if (tid == 0) scratch[0] = v;
    sycl::group_barrier(it.get_group());
    return scratch[0];
}

template <typename D>
void gnorm_t(sycl::queue& q, const float* x, const float* w, float* y, int n_embd, float eps, int hc) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> scratch(sycl::range<1>(8), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int c = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const float* xc = x + (size_t) c * n_embd;
            const float* wc = w + (size_t) c * n_embd;
            float* yc = y + (size_t) c * n_embd;
            D acc = 0;
            for (int d = tid; d < n_embd; d += THREADS) {
                const float sq = xc[d] * xc[d];   // the product rounded in f32 before it is widened
                acc += (D) sq;
            }
            const float mean = (float) (block_sum(it, acc, &scratch[0]) / (D) n_embd);
            const float scale = 1.0f / sycl::sqrt(mean + eps);
            for (int d = tid; d < n_embd; d += THREADS) yc[d] = xc[d] * scale * wc[d];
        });
    });
}
void gnorm(sycl::queue& q, const float* x, const float* w, float* y, int n_embd, float eps, int hc) {
    if (xe::has_fp64(q)) gnorm_t<double>(q, x, w, y, n_embd, eps, hc);
    else gnorm_t<float>(q, x, w, y, n_embd, eps, hc);
}

// the value projection over BF16 operands, summed in D
template <typename D>
void value_t(sycl::queue& q, const uint16_t* emb16, const uint16_t* wv, float* value, int n_embd) {
    q.parallel_for(sycl::range<1>(n_embd), [=](sycl::id<1> id) {
        const int o = (int) id[0];
        const uint16_t* row = wv + (size_t) o * n_embd;
        D acc = 0;
        for (int i = 0; i < n_embd; ++i) acc += (D) bf16_float(emb16[i]) * (D) bf16_float(row[i]);
        value[o] = (float) acc;
    });
}

// the gate per stream: sigmoid of the signed square root of key . query / sqrt(n), the dot summed in D
template <typename D>
void gate_t(sycl::queue& q, const float* key, const float* query, float* gate, int n_embd, int hc) {
    const float inv_sqrt_n = 1.0f / std::sqrt((float) n_embd);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> scr(sycl::range<1>(8), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int c = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const float* kc = key + (size_t) c * n_embd;
            const float* qc = query + (size_t) c * n_embd;
            D acc = 0;
            for (int d = tid; d < n_embd; d += THREADS) acc += (D) (kc[d] * qc[d]);
            const float s = (float) (block_sum(it, acc, &scr[0]) / (D) 1) * inv_sqrt_n;
            const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
            const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
            if (tid == 0) gate[c] = 1.0f / (1.0f + sycl::exp(-(sgn * mag)));
        });
    });
}

}  // namespace

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    auto& q = core::Runtime::get().stream(stream);
    const auto e = q.parallel_for(sycl::range<1>(NG_HC_DIM), [=](sycl::id<1> id) {
        const int channel = (int) id[0];
        float* column = hist + (size_t) channel * NG_HIST;
        for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
        column[NG_HIST - 1] = normalized[channel];
    });
    if (!stream) core::Runtime::get().wait(e, "ple_history_advance");
}

bool ple_block_available() {
    try { (void) core::Runtime::get(); return true; }
    catch (const std::exception&) { return false; }
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, then the BF16 embedding copy, each 16-byte
    // aligned because the scratch is cast to float* and uint8_t* at those offsets
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 || (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 && w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires supported native weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    auto& q = core::Runtime::get().stream(stream);
    void* st = &q;

    // the workspace is carved from the caller's (a caller-owned workspace, no token-path allocation)
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) throw core::DeviceError("ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)");
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}
    };
    for (const auto& item : exports) {
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    }
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) ||
            (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}
        };
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb): the optional native projection follows pinned CUDA Q8_1 MMVQ; the
    // default keeps its canonical Q8_0 path
    if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, st);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, st);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1, d_key, n_embd, hc_dim, 1, st);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    if (!native_postops) {
        gnorm(q, d_key, w.norm_key, d_key, n_embd, NG_RMS_EPS, hc);
        gnorm(q, hidden, w.norm_query, d_query, n_embd, NG_RMS_EPS, hc);
    }

    // the value projection's independent option leaves the nonlinear PLE operations unchanged
    if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, st);
    } else {
        q.parallel_for(sycl::range<1>(n_embd), [=](sycl::id<1> i) { d_emb16[i] = bf16_bits(emb[i]); });
        if (xe::has_fp64(q)) value_t<double>(q, d_emb16, w.value_bf16, d_value, n_embd);
        else value_t<float>(q, d_emb16, w.value_bf16, d_value, n_embd);
    }

    const float* normalized_key = d_key;
    if (native_postops) {
        // the key norm has a distinct destination; the query storage is reused for the normalized gated values
        // after the gate consumes it
        NativePlePostopsBuffers buffers{d_query, d_norm, d_gate, d_gated, d_norm, d_conv, out.result};
        native_ple_postops(d_key, hidden, d_value, hist_rows, w, buffers, st);
        normalized_key = d_query;
    } else {
        if (xe::has_fp64(q)) gate_t<double>(q, d_key, d_query, d_gate, n_embd, hc);
        else gate_t<float>(q, d_key, d_query, d_gate, n_embd, hc);
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> id) {
            const int i = (int) id[0];
            d_gated[i] = d_value[i % n_embd] * d_gate[i / n_embd];
        });
        gnorm(q, d_gated, w.norm_conv, d_norm, n_embd, NG_RMS_EPS, hc);
        const uint16_t* kW = w.conv1d_f16;
        const int kern = PLE_CONV_KERNEL, dil = NGRAM_SIZE, nhist = NG_HIST;
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> id) {
            const int c = (int) id[0];
            float acc = 0.0f;
            for (int k = 0; k < kern; ++k) {
                const int row = nhist - (kern - 1 - k) * dil;   // tap 0 reads the FURTHEST back
                // row-fastest: hist[row + nhist*c] (ggml_reshape_3d(state, d_conv-1, conv_channels, n_seqs))
                const float v = (row == nhist) ? d_norm[c] : hist_rows[(size_t) row + (size_t) nhist * c];
                acc += f32_from_f16(kW[k + kern * c]) * v;
            }
            d_conv[c] = silu_f(acc);
        });
        float* result = out.result;
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> i) { result[i] = hidden[i] + d_gated[i] + d_conv[i]; });
    }

    // the intermediates the oracle comparison needs: `key` is the NORMALISED key, `value` the projection before
    // the gate
    if (out.key) q.memcpy(out.key, normalized_key, hc_dim * sizeof(float));
    if (out.value) q.memcpy(out.value, d_value, n_embd * sizeof(float));
    if (out.gate) q.memcpy(out.gate, d_gate, hc * sizeof(float));
    if (out.gated) q.memcpy(out.gated, d_gated, hc_dim * sizeof(float));
    if (out.normalized) q.memcpy(out.normalized, d_norm, hc_dim * sizeof(float));
    if (out.conv) q.memcpy(out.conv, d_conv, hc_dim * sizeof(float));
    // no synchronisation here: inside a capture it is an error, and the engine's caller does not want it
}

}  // namespace strata::kernels
