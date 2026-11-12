// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/dit_product.cpp - the DiTs' one product entry (dit_product.hpp): Qwen-Image's, Mage-Flow's, MOSS-Image's,
// Anima's and SeedVR2's products, which each DiT had its own copy of.
#include "strata/image/dit_product.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/gpu_buf.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/rows_i8.hpp"
#include "strata/prefill/gemm.hpp"

namespace strata::image {

void DitLoraScratch::ensure(int64_t T, std::vector<void*>& owned, void* stream, const char* what) {
    if (cols <= 0 || T <= 0) return;
    // a GLoRA's own path's buffers are a band's, not the tokens': the activations' drop frees them, this gives them back
    if (glora_cols > 0 && glora_y == nullptr) glora_alloc(owned, what);
    if (f32 != nullptr && T <= rows) return;
    gpu::stream_sync(stream);   // the old buffers may still be read
    for (void** p : {reinterpret_cast<void**>(&f32), reinterpret_cast<void**>(&f16)})
        if (*p != nullptr) {
            release(owned, *p);
            *p = nullptr;
        }
    f32 = alloc<float>(owned, T * cols, what);
    f16 = alloc<uint16_t>(owned, T * cols, what);
    rows = T;
}

// A GLoRA's own path's buffers: the table the weight's product of its input is added from (a band of tokens, the
// weight's rows), and that input as a W8A8 product takes it (int8 values and a scale a row).  The deltas' scratch
// (ensure) holds the input's own two products.
void DitLoraScratch::glora_alloc(std::vector<void*>& owned, const char* what) {
    if (glora_cols <= 0 || glora_rows <= 0) return;
    const size_t band = (size_t) kLoraGloraRows;
    glora_y = alloc<float>(owned, (int64_t) (band * (size_t) glora_rows), what);
    glora_x8 = alloc<int8_t>(owned, (int64_t) (band * (size_t) glora_cols), what);
    glora_sx = alloc<float>(owned, (int64_t) band, what);
}

void DitLoraScratch::glora_free(std::vector<void*>& owned, void* stream) {
    for (void* p : {(void*) glora_y, (void*) glora_x8, (void*) glora_sx})
        if (p != nullptr) {
            gpu::stream_sync(stream);
            release(owned, p);
        }
    glora_y = nullptr;
    glora_x8 = nullptr;
    glora_sx = nullptr;
}

void DitLoraScratch::drop(std::vector<void*>& owned, void* stream) {
    cols = 0;
    glora_cols = glora_rows = 0;
    glora_free(owned, stream);
    if (f32 != nullptr || f16 != nullptr) {
        gpu::stream_sync(stream);
        for (void** p : {reinterpret_cast<void**>(&f32), reinterpret_cast<void**>(&f16)})
            if (*p != nullptr) {
                release(owned, *p);
                *p = nullptr;
            }
        rows = 0;
    }
}

void DitLoraScratch::drop_buffers(std::vector<void*>& owned) {
    for (void* p : {(void*) f32, (void*) f16, (void*) glora_y, (void*) glora_x8, (void*) glora_sx}) release(owned, p);
    f32 = nullptr;
    f16 = nullptr;
    rows = 0;
    glora_y = nullptr;
    glora_x8 = nullptr;
    glora_sx = nullptr;
}

uint64_t DitLoraScratch::glora_bytes() const {
    if (glora_y == nullptr) return 0;
    return (uint64_t) kLoraGloraRows * ((uint64_t) glora_rows * 4 + (uint64_t) glora_cols + 4);
}

uint64_t DitLoraScratch::bytes(int64_t T) const {
    return cols > 0 ? (uint64_t) std::max(T, rows) * (uint64_t) cols * 6 : 0;
}

float* DitI8Input::take(const uint16_t* x16, const float* rs, int64_t T, int64_t K, bool same_input, void* stream) {
    // the activations rotated and quantized as the weights (once for the products of one input), their scales times
    // the rows' (rs)
    if (same_input) return reinterpret_cast<float*>(q + rows * cols) + next * rows;
    float* sx = slot();
    kernels::rows_i8(x16, false, T, K, q, sx, rs, stream);
    return sx;
}

namespace {

// A GLoRA's own path: the weight's rows [r0, r0 + n) product of v (nt tokens, the first of them t0), added to the
// product's output y at those rows (y holds the weight's rows from row0 on, ldy apart), the weight's own path
void glora_weight(const DitW& w, const uint8_t* d, const uint16_t* v, int64_t r0, int64_t n, int64_t nt, int64_t t0,
                  float* y, int64_t ldy, int64_t row0, prefill::Gemm& gemm, const DitLoraScratch& ls, void* s) {
    if (w.int8_rows()) {   // (Qwen-Image's w8t weights keep their blocks: the native path below)
        // the input's own int8 quantization (its buffer, not the product's: that one's is the other products' input)
        kernels::rows_i8(v, false, nt, w.k, ls.glora_x8, ls.glora_sx, nullptr, s);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k) + r0;
        prefill::Gemm::w8a8(ls.glora_x8, ls.glora_sx, reinterpret_cast<const int8_t*>(d) + r0 * w.k, sw, nullptr,
                            ls.glora_y, nt, n, w.k, n, false, s);
        kernels::i8_scale(ls.glora_y, nt, n, n, ls.glora_sx, sw, nullptr, nullptr, s);
    } else if (w.type == kDitF16) {
        gemm.f16(v, reinterpret_cast<const uint16_t*>(d + (uint64_t) r0 * w.row_bytes), ls.glora_y, nt, n, w.k, n);
    } else {
        gemm.native(v, w.type, d + (uint64_t) r0 * w.row_bytes, ls.glora_y, nt, n, w.k, n);
    }
    lora_output_add(y + t0 * ldy, nt, ldy, r0 - row0, ls.glora_y, n, s);
}

}  // namespace

DitOut DitProducts::run(const DitProduct& p) {
    prefill::Gemm& gemm = *gemm_;
    DitLoraScratch& ls = lora;
    void* s = *s_;
    const DitW& w = *p.w;
    const int64_t rows = p.rows < 0 ? w.n : p.rows, ldy = p.ldy > 0 ? p.ldy : rows, T = p.T;
    float* y = p.y;
    DitOut out;
    out.f = y;
    out.C = ldy;
    if (T <= 0) return out;
    LoraLayer* lora = w.baked ? nullptr : w.lora;
    const bool defer = p.defer && lora == nullptr;   // a delta is FP32: the int32 sums are scaled into y first
    // a DoRA's factor of the weight's rows: the base product's, then the delta's own
    const float* col = lora != nullptr && lora->has_base() ? lora->base() + p.row0 : nullptr;
    // an OFT turns the weight's rows: the weight's own bias goes after it (the delta is the input's own, x16 = x / rs,
    // and the turn is of x16's units too)
    const bool oft = lora != nullptr && lora->has_oft();
    if (p.x8 != nullptr) {
        // W8A8: the int32 sums into y, scaled there (sx carries the rows' scales: y is in the input's units), or by
        // the reader (defer)
        const int8_t* w8 = p.w8 != nullptr ? p.w8 : reinterpret_cast<const int8_t*>(p.d) + p.row0 * w.k;
        const float* sw = p.sw != nullptr ? p.sw : reinterpret_cast<const float*>(p.d + w.n * w.k) + p.row0;
        if (prefill::Gemm::w8a8(p.x8, p.sx, w8, sw, p.bias, y, T, rows, w.k, ldy, defer, s)) {
            out.f = nullptr;
            out.bf = reinterpret_cast<const uint16_t*>(y);
            return out;
        }
        if (defer) {
            out.f = nullptr;
            out.i = reinterpret_cast<const int32_t*>(y);
            out.sx = p.sx;
            out.sw = sw;
            out.b = p.bias;
            return out;
        }
        kernels::i8_scale(y, T, rows, ldy, p.sx, sw, oft ? nullptr : p.bias, col, s);
    } else {
        const uint8_t* base = p.d + (uint64_t) p.row0 * w.row_bytes;
        if (w.type == kDitF16) gemm.f16(p.x16, reinterpret_cast<const uint16_t*>(base), y, T, rows, w.k, ldy);
        else gemm.native(p.x16, w.type, base, y, T, rows, w.k, ldy);
        if (defer) {
            out.rs = p.rs;
            out.b = p.bias;
            return out;
        }
        const float* b = oft ? nullptr : p.bias;
        if (p.rs != nullptr || b != nullptr) dit_rows::rows_back(y, p.rs, b, T, rows, s, col);
        else if (col != nullptr) qi_rows_scale(y, T, rows, ldy, col, s);   // (ldy: a part of a fused weight's rows)
    }
    if (lora != nullptr) {
        // the delta is the input's own (x16 = x / rs): its rows' scales put it back in y's units, as rows_back does
        ls.ensure(T, *owned_, s, what_);
        lora->add(p.x16, T, y, ldy, p.row0, rows, p.rs, gemm, ls.f32, ls.f16, s);
        if (lora->has_glora())   // a GLoRA's own path: the weight's own product of its input, a band of tokens at a time
            lora->add_glora(p.x16, T, p.row0, rows, p.rs, gemm, ls.f32, ls.f16, s,
                            [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                                glora_weight(w, p.d, v, r0, n2, nt, t0, y, ldy, p.row0, gemm, ls, s);
                            });
        if (oft && p.bias != nullptr) dit_rows::bias_add(y, p.bias, T, rows, s);
    }
    return out;
}

DitOut DitProducts::run(const uint16_t* x16, const float* rs, const DitW& w, const uint8_t* base, float* y, int64_t T,
                        const float* bias, bool same_input, bool defer) {
    DitProduct p;
    p.w = &w;
    p.d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    p.x16 = x16;
    p.rs = rs;
    p.bias = bias;
    p.y = y;
    p.T = T;
    p.defer = defer;
    if (w.type == kDitW8 && T > 0) {
        p.x8 = in.q;
        p.sx = in.take(x16, rs, T, w.k, same_input, *s_);
    }
    return run(p);
}

}  // namespace strata::image
