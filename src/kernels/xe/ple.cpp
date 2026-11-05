// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/ple.cpp - the Xe port of Strata's src/kernels/cuda/ple.cu: the PLE block's GPU half (see ple.hpp).
//
// The projections are the pinned CUDA ones (BF16 MMVF, Q8_1 MMVQ) and the arithmetic after them native_ple_postops'.
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_ple
#include "strata/kernels/device_arch_tag.hpp"

#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

}  // namespace

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

namespace {
constexpr int8_t kPleIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

void ple_iq4nl_dequant(const uint8_t* raw, int64_t rows, float* out, void* stream) {
    const int64_t n = rows * PLE_HEAD_DIM;
    if (n <= 0) return;
    auto& q = core::Runtime::get().stream(stream);
    // one work-item a value: row r, block b (18 bytes: fp16 d, 16 bytes of codes), element j of 32 (j < 16: the low
    // nibble of qs[j], else the high nibble of qs[j - 16] - the split halves of iq4nl_dequant_row)
    const int64_t groups = (n + THREADS - 1) / THREADS;
    const auto e = q.parallel_for(sycl::nd_range<1>((size_t) groups * THREADS, THREADS), [=](sycl::nd_item<1> it) {
        const int64_t i = (int64_t) it.get_global_id(0);
        if (i >= n) return;
        const int64_t r = i / PLE_HEAD_DIM;
        const int el = (int) (i - r * PLE_HEAD_DIM), b = el >> 5, j = el & 31;
        const uint8_t* blk = raw + r * PLE_ROW_BYTES + (int64_t) (b * 18);
        const float d = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint8_t c = blk[2 + (j & 15)];
        out[i] = d * (float) kPleIq4Nl[j < 16 ? (c & 15) : (c >> 4)];
    });
    if (!stream) core::Runtime::get().wait(e, "ple_iq4nl_dequant");
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, each 16-byte aligned because the scratch is
    // cast to float* and uint8_t* at those offsets
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 || (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 && w.key_native_type != 8 &&
                        w.key_native_type != kNativeQ8Rows)))
        throw std::invalid_argument("ple_block: native key requires supported native weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    auto& q = core::Runtime::get().stream(stream);
    void* st = &q;

    // the workspace is carved from the caller's (a caller-owned workspace, no token-path allocation)
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
    float* d_scratch = reinterpret_cast<float*>(base);
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = ple_key @ emb: the GGUF key (native_mmvq, pinned CUDA Q8_1) or the model files' BF16 one; a canonical
    // pack's S2 key, which no GGUF form of the weight came with, its Q8_0 product
    if (w.pre_key != nullptr) {
        q.memcpy(d_key, w.pre_key, (size_t) hc_dim * sizeof(float));
    } else if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, st);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, st);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1, d_key, n_embd, hc_dim, 1, st);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    // ---- value = ple_value @ emb (BF16 weights, FP32 activations: pinned CUDA MMVF)
    if (w.pre_value != nullptr) q.memcpy(d_value, w.pre_value, (size_t) n_embd * sizeof(float));
    else bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, st);

    // ---- the norms, the gate, the convolution and the residual: the pinned CUDA postprojection arithmetic.  The key
    // norm has a distinct destination; the query storage is reused for the normalized gated values after the gate
    // consumes it
    NativePlePostopsBuffers buffers{d_query, d_norm, d_gate, d_gated, d_norm, d_conv, out.result};
    native_ple_postops(d_key, hidden, d_value, hist_rows, w, buffers, st);
    const float* normalized_key = d_query;

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
