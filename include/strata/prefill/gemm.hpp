// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are either BF16 on the device already (the pack's BF16 tensors) or dequantized from their
// native GGUF blocks to FP16 into a reusable scratch right before the product; the activations are in the same type.
// The products run through oneMath in the contrib modes, otherwise on XeStrata's own kernels
// (strata/kernels/xmx_gemm.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::gpu { struct Event; }

namespace strata::prefill {

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: 16-bit elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with a caller-owned device buffer (the prompt path borrowing expert-cache slots).
    void init_external(void* stream, uint16_t* scratch, int64_t scratch_elems);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major); `accumulate`: Y +=.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              bool accumulate = false);

    /// Y = X . W^T with both in FP16 (bits).  `half_sums`: the sums in FP16 where oneMath takes the product (its
    /// half GEMM, then widened into Y), for products whose sums stay in FP16's range (inputs after a norm); else, and
    /// in XeStrata's own kernels, in FP32.  `accumulate`: Y += (read before the product, in FP32; with `half_sums`
    /// the sums stay FP32, the half GEMM taking no beta).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             bool half_sums = false, bool accumulate = false);

    /// `batch` products of the same shape at strides (elements): Y + b sy (row stride ldy, 0: N) = (X + b sx) [T][K] .
    /// (W + b sw)^T [N][K], FP16 in, FP32 out; one call of oneMath's strided batch where it has one for the GPU, else
    /// one product each.
    void f16_batch(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t batch,
                   int64_t sx, int64_t sw, int64_t sy, int64_t ldy = 0);
    /// The experts' groups (FP16), one product each through oneMath: rows bounds[e] .. bounds[e + 1] (host values) of
    /// X and Y (row stride N) by W + e * w_stride, e < G.  False when oneMath does not take FP16 products on `stream`'s
    /// GPU: the caller runs kernels::xmx_gemm_grouped then.
    static bool f16_groups(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds,
                           int G, int64_t N, int64_t K, void* stream);

    /// Whether `stream`'s GPU takes i8: oneMath's backend for it has int8 products (asked of it once, by a trial), or
    /// oneDNN takes the GPU (kernels::i8_gemm_s32: Intel's GPUs, the free mode's int8 products).
    static bool i8_usable(void* stream);
    /// Y[T, N] (int32, row stride ldy) = X[T, K] . W[N, K]^T, both int8 row-major, int32 sums, through oneMath, else
    /// oneDNN; only where i8_usable says yes.
    static void i8(const int8_t* X, const int8_t* W, int32_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                   void* stream);
    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16; `half_sums` as
    /// f16's.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, bool half_sums = false);
    /// native with FP16 sums into FP16 rows Y16 (row stride ldy), not widened into FP32: the reader takes them as they
    /// are (prefill::add_h, swiglu_pair_h), a pass over the sums fewer.  Where half_ok says no, the FP32 product
    /// narrowed into Y16 (the same rows, a pass more).
    void native_half(const uint16_t* X, int ggml_type, const void* W_blocks, uint16_t* Y16, int64_t T, int64_t N,
                     int64_t K, int64_t ldy = 0);
    /// Whether `stream`'s GPU takes native_half (oneMath's half GEMM).
    static bool half_ok(void* stream);
    /// The products' warm-up, shared by the models: native() once a shape (type, N x K, ldy) a GPU in the process -
    /// a library's first product of a shape loads its kernels for it (rocBLAS on the RX 9060 XT: 0.5 s for the
    /// first product, 0.9 s more for the first of 32 rows) - on X's rows of zeros into a buffer the caller
    /// overwrites.  A shape already warm on this Gemm's GPU is skipped: native_warm says whether it ran, native_cold
    /// whether it would.
    bool native_cold(int ggml_type, int64_t N, int64_t K, int64_t ldy = 0) const;
    bool native_warm(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                     int64_t ldy = 0);

    /// Loads the products' library ahead of the first prompt (oneMath's backend for the current GPU), on a thread of
    /// its own: call once for each GPU at startup.
    static void prepare();
    /// Waits for prepare's trial on the current GPU (the library and its first kernels loaded).
    static void settle();
    /// Waits for every GPU's trial still running: before the process ends, so no trial is still loading its library
    /// while the exit handlers take the GPU runtime down (main's guard; a run that ended in an error before its first
    /// prompt crashed or hung there on the RX 9060 XT, 2026-11-08).
    static void finish();
    /// The path the products take on the current GPU, for the startup report: "oneMath", or the own kernels'
    /// (kernels::gemm_path).
    static const char* path();

    /// Caller-owned buffer only: the scratch moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems);

    /// STRATA_DEQUANT_SIDE=1 (gemm_dequant_side): a second dequant scratch of scratch_elems() elements, caller-owned
    /// like the first.  native() then dequantizes on a side stream into the two in turn, so a weight's dequant runs
    /// while the kernels queued ahead of its product do; events order every write and read of each scratch, so the
    /// products are the serial form's, bit for bit.  nullptr: the serial form.  Called after init_external and after
    /// every rebind, on the device of the Gemm's stream.
    bool set_dequant_side(uint16_t* scratch2, std::string& err);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    template<class F> void with_f16(int ggml_type, const void* W_blocks, int64_t N, int64_t K, F prod);
    bool bf16_through_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy);

    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    bool external_ = false;
    uint16_t* tc_w_ = nullptr;   // bf16_through_f16's FP16 images of W and of a slice of X, grown on demand
    uint16_t* tc_x_ = nullptr;
    int64_t tc_w_elems_ = 0, tc_x_elems_ = 0;
    // STRATA_DEQUANT_SIDE (set_dequant_side): the second scratch, the side stream and, per scratch, "the dequant
    // landed" (recorded on the side stream) and "the product read it" (recorded on the compute stream); side_cur_ is
    // the scratch the next native() dequantizes into (0: scratch_, 1: side_scratch_)
    uint16_t* side_scratch_ = nullptr;
    void* side_stream_ = nullptr;
    strata::gpu::Event* side_ready_[2] = {};
    strata::gpu::Event* side_free_[2] = {};
    int side_cur_ = 0;
};

/// STRATA_DEQUANT_SIDE=1: the prompt path's dense-weight dequant on a side stream with a second GEMM scratch
/// (Gemm::set_dequant_side; the prompt path takes the scratch from its region when this is on).  Off by default.
bool gemm_dequant_side();

}  // namespace strata::prefill
