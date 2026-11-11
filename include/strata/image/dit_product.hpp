// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/dit_product.hpp - the DiTs' products, one entry for every image model's DiT (Qwen-Image,
// Mage-Flow, MOSS-Image, Anima) and SeedVR2's: a weight as the products read it (DitW), a product's output as its
// readers take it (DitOut: FP32 sums, W8A8's int32 sums left for the reader to scale, or BF16 values), and the product
// itself (dit_product): the dense product through prefill::Gemm (W8A8 through prefill::Gemm::w8a8), the rows' scales,
// the bias and a DoRA's factors, and the adapters' deltas (image_lora.hpp) on the output.
#pragma once

#include "strata/image/image_lora.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cstdint>
#include <vector>

namespace strata::prefill {
class Gemm;
}

namespace strata::image {

/// A DiT weight's forms: FP16 rows, Q8_0 blocks, W8A8's int8 rows (their FP32 scales after them, or at `sw`), FP32
/// (Mage-Flow's low-rank modulation); Qwen-Image's weights keep the file's GGUF type (w8 marks the int8 rewrite).
constexpr int kDitF16 = 1, kDitQ8_0 = 8, kDitW8 = -8, kDitF32 = 32;

/// W values side by side, aligned to their size: a work-item's load or store of them in one access (vec<T, W> of
/// SYCL, whose header this one does not take)
template<class T, int W> struct alignas(sizeof(T) * W) Packed {
    T v[W];
};

/// A product's output [T][C] as its readers take it: FP32 sums (f) times the rows' scales (rs, null: 1), or a W8A8
/// product's int32 sums (i) times the activations' scales (sx, a row) and the weights' (sw, a column); plus the bias
/// (b, null: none).  Or a W8A8 product's values as they are (bf, BF16: kernels::i8_gemm_bf16 scaled them and added
/// the bias), half the bytes of the sums.
struct DitOut {
    const float* f = nullptr;
    const int32_t* i = nullptr;
    const float *rs = nullptr, *sx = nullptr, *sw = nullptr, *b = nullptr;
    int64_t C = 0;
    const uint16_t* bf = nullptr;
    float at(int64_t t, int64_t c) const {
        if (bf != nullptr) return kernels::f32_from_bf16(bf[t * C + c]);
        const float v = i != nullptr ? (float) i[t * C + c] * sx[t] * sw[c]
                                     : (rs != nullptr ? f[t * C + c] * rs[t] : f[t * C + c]);
        return b != nullptr ? v + b[c] : v;
    }
    /// at(t, c0 + d) as r(d), the row's offsets taken once (the per-head kernels' reads: qk_norm_rope.hpp); r.r2(d, a,
    /// b) reads d and d + 1 in one access (d even, c0 and C too)
    struct Row {
        const uint16_t* bp16;
        const int32_t* ip;
        const float *fp, *swp, *bp;
        float sxt, rst;
        bool scaled;
        float operator()(int64_t d) const {
            if (bp16 != nullptr) return kernels::f32_from_bf16(bp16[d]);
            const float v = ip != nullptr ? (float) ip[d] * sxt * swp[d] : (scaled ? fp[d] * rst : fp[d]);
            return bp != nullptr ? v + bp[d] : v;
        }
        void r2(int64_t d, float& v0, float& v1) const {
            if (bp16 != nullptr) {
                const Packed<uint16_t, 2> h = *reinterpret_cast<const Packed<uint16_t, 2>*>(bp16 + d);
                v0 = kernels::f32_from_bf16(h.v[0]);
                v1 = kernels::f32_from_bf16(h.v[1]);
                return;
            }
            if (ip != nullptr) {
                const Packed<int32_t, 2> iv = *reinterpret_cast<const Packed<int32_t, 2>*>(ip + d);
                const Packed<float, 2> wv = *reinterpret_cast<const Packed<float, 2>*>(swp + d);
                v0 = (float) iv.v[0] * sxt * wv.v[0];
                v1 = (float) iv.v[1] * sxt * wv.v[1];
            } else {
                const Packed<float, 2> fv = *reinterpret_cast<const Packed<float, 2>*>(fp + d);
                v0 = scaled ? fv.v[0] * rst : fv.v[0];
                v1 = scaled ? fv.v[1] * rst : fv.v[1];
            }
            if (bp != nullptr) {
                const Packed<float, 2> bv = *reinterpret_cast<const Packed<float, 2>*>(bp + d);
                v0 += bv.v[0];
                v1 += bv.v[1];
            }
        }
    };
    Row row(int64_t t, int64_t c0) const {
        const int64_t o = t * C + c0;
        return Row{bf != nullptr ? bf + o : nullptr, i != nullptr ? i + o : nullptr, f != nullptr ? f + o : nullptr,
                   sw != nullptr ? sw + c0 : nullptr, b != nullptr ? b + c0 : nullptr, i != nullptr ? sx[t] : 0.f,
                   rs != nullptr ? rs[t] : 1.f, rs != nullptr};
    }
    /// at(t, c + p) for p below W, read W side by side (c and C multiples of W)
    template<int W> void at_w(int64_t t, int64_t c, float* o) const {
        if (bf != nullptr) {
            const Packed<uint16_t, W> hv = *reinterpret_cast<const Packed<uint16_t, W>*>(bf + t * C + c);
            for (int p = 0; p < W; ++p) o[p] = kernels::f32_from_bf16(hv.v[p]);
            return;
        }
        if (i != nullptr) {
            const Packed<int32_t, W> iv = *reinterpret_cast<const Packed<int32_t, W>*>(i + t * C + c);
            const Packed<float, W> wv = *reinterpret_cast<const Packed<float, W>*>(sw + c);
            const float st = sx[t];
            for (int p = 0; p < W; ++p) o[p] = (float) iv.v[p] * st * wv.v[p];
        } else {
            const Packed<float, W> fv = *reinterpret_cast<const Packed<float, W>*>(f + t * C + c);
            const float r = rs != nullptr ? rs[t] : 1.f;
            for (int p = 0; p < W; ++p) o[p] = rs != nullptr ? fv.v[p] * r : fv.v[p];
        }
        if (b != nullptr) {
            const Packed<float, W> bv = *reinterpret_cast<const Packed<float, W>*>(b + c);
            for (int p = 0; p < W; ++p) o[p] += bv.v[p];
        }
    }
    /// the rows from r on
    DitOut from(int64_t r) const {
        DitOut o = *this;
        if (f != nullptr) o.f += r * C;
        if (i != nullptr) o.i += r * C;
        if (bf != nullptr) o.bf += r * C;
        if (rs != nullptr) o.rs += r;
        if (sx != nullptr) o.sx += r;
        return o;
    }
};

/// FP32 rows [T][C] as the shared row kernels read a product's output (a block's tape or scratch)
inline DitOut rows_of(const float* p, int64_t C) {
    DitOut m;
    m.f = p;
    m.C = C;
    return m;
}


/// A DiT's weight as its products read it.
struct DitW {
    int type = 0;              // kDitF16, kDitQ8_0, kDitW8, kDitF32, or Qwen-Image's GGUF type
    int64_t n = 0, k = 0;      // n rows of k
    const void* d = nullptr;   // a block's: its offset in the block's region; else a device pointer
    uint64_t row_bytes = 0;
    uint64_t sw = 0;           // SeedVR2's W8A8: its rows' scales' offset in the block's region (0: after the rows)
    bool w8 = false;           // Qwen-Image: rewritten as int8 rows and a scale a row (kernels::rows_i8)
    bool w8t = false;          // Qwen-Image: w8 through a scratch: a block format of fewer bits than int8's rows, kept
                               // as it is and written as int8 rows at each product
    LoraLayer* lora = nullptr;   // the adapters' deltas for its products (null: none: the product is unchanged)
    bool baked = false;          // its adapters folded into the block's copy of it (set_loras), not added to products
    /// The weight's bytes are int8 rows and their scales (a GLoRA's own path takes them as W8A8)
    bool int8_rows() const { return type == kDitW8 || (w8 && !w8t); }
};

/// The buffers the adapters' deltas take in a DiT's products (LoraLayer::add and add_glora), grown on demand in the
/// DiT's allocations: `cols` the deltas' largest (lora_scratch_cols), `glora_cols` and `glora_rows` a GLoRA's input's
/// columns and the weight's rows its own path adds over (0: none bound).
struct DitLoraScratch {
    float* f32 = nullptr;
    uint16_t* f16 = nullptr;
    int64_t cols = 0, rows = 0;   // rows: the tokens f32 and f16 hold
    int64_t glora_cols = 0, glora_rows = 0;
    float* glora_y = nullptr;     // a band of the weight's product of a GLoRA's input [kLoraGloraRows][glora_rows]
    int8_t* glora_x8 = nullptr;   // that input as W8A8's int8 rows, and their scales
    float* glora_sx = nullptr;

    /// f32 and f16 for T tokens (and the GLoRA's buffers, which the activations' drop frees), in `owned`
    void ensure(int64_t T, std::vector<void*>& owned, void* stream, const char* what);
    void glora_alloc(std::vector<void*>& owned, const char* what);
    void glora_free(std::vector<void*>& owned, void* stream);
    /// every buffer freed (drop_loras) and the sizes forgotten
    void drop(std::vector<void*>& owned, void* stream);
    /// the buffers freed with the activations: the GLoRA's are made again by ensure
    void drop_buffers(std::vector<void*>& owned);
    /// The GPU memory the GLoRA's buffers hold (none: 0)
    uint64_t glora_bytes() const;
    /// The deltas' buffers for T tokens (0 without an adapter bound)
    uint64_t bytes(int64_t T) const;
};

/// One product's operands: y [T][rows] (row stride ldy, 0: rows) = x [T][w.k] . the weight's rows [row0, row0 + rows)
/// (rows -1: all of them), times the input rows' scales rs (null: 1; x16 is x / rs), plus bias (null: none), plus the
/// weight's adapters (w.lora, not w.baked).
struct DitProduct {
    const DitW* w = nullptr;
    const uint8_t* d = nullptr;           // the weight's bytes (a block's, in its region)
    const uint16_t* x16 = nullptr;        // the FP16 input (an adapter's delta reads it: W8A8's products too)
    const int8_t* x8 = nullptr;           // W8A8: the input's int8 rows and their scales (rs folded in)
    const float* sx = nullptr;
    const int8_t* w8 = nullptr;           // W8A8: the weight's int8 rows from row0 on and their scales, where they are
    const float* sw = nullptr;            // not d's own (null: d's)
    const float* rs = nullptr;
    const float* bias = nullptr;
    float* y = nullptr;
    int64_t T = 0, ldy = 0, row0 = 0, rows = -1;
    /// y left as the product made it, for the reader to scale (W8A8's int32 sums or BF16 values, FP32 sums and rs):
    /// not with an adapter (a delta is FP32, added to scaled values)
    bool defer = false;
};

/// mf/moss/anima's W8A8 input: rows of up to `cols` int8 values, then two arrays of their scales taken in turn, so
/// that the deferred sums of one input keep theirs while the next input is quantized
struct DitI8Input {
    int8_t* q = nullptr;
    int64_t rows = 0, cols = 0;
    int next = 0;
    /// The int8 rows of x16 (rows scaled down by rs: their scales carry rs) for a product of K columns, and their
    /// scales; same_input: the previous product's (its int8 rows taken again)
    float* take(const uint16_t* x16, const float* rs, int64_t T, int64_t K, bool same_input, void* stream);
    /// The other of the two arrays of scales, made the current one (what a product's same_input reads): for int8 rows
    /// a DiT's own kernel writes into q
    float* slot() {
        next ^= 1;
        return reinterpret_cast<float*>(q + rows * cols) + next * rows;
    }
};

/// A DiT's products: the DiT's library handle, its allocations and its stream (the members' addresses: the stream is
/// set when the DiT opens), and the buffers its products take (the adapters' deltas', the W8A8 input).
class DitProducts {
public:
    DitProducts(prefill::Gemm* gemm, std::vector<void*>* owned, void* const* stream, const char* what)
        : gemm_(gemm), owned_(owned), s_(stream), what_(what) {}
    DitProducts(const DitProducts&) = delete;
    DitProducts& operator=(const DitProducts&) = delete;

    /// The product on the stream's GPU, as its reader takes y.  A W8A8 product's values come as BF16 where the fused
    /// kernel is the faster (prefill::Gemm::w8a8, defer only): half the bytes the reader reads.
    DitOut run(const DitProduct& p);
    /// y [T][w.n] = (x16 [T][w.k], rows scaled down by rs) . w^T, the rows times rs again, plus bias (rs, bias null:
    /// none), the weight's bytes at base + w.d (base null: w.d).  A W8A8 weight's input is x16 made int8 rows here
    /// (`in`); same_input: x16 is the previous product's, which was W8A8 too (its int8 rows taken again).  defer: as
    /// DitProduct's.
    DitOut run(const uint16_t* x16, const float* rs, const DitW& w, const uint8_t* base, float* y, int64_t T,
               const float* bias, bool same_input = false, bool defer = false);

    DitI8Input in;
    DitLoraScratch lora;

private:
    prefill::Gemm* gemm_;
    std::vector<void*>* owned_;
    void* const* s_;
    const char* what_;
};

}  // namespace strata::image
