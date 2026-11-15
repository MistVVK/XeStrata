// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts multiplied in their
// quantized blocks (iq_mmq.hpp: llama.cpp's MMQ, written anew on joint_matrix): the activations are rounded to int8,
// the products run on the int8 matrix engines.  The dequantize-to-FP16 path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_native` one expert at a time, `gather_native_group` a group's at once) and multiplied in one launch per
// product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {
struct NativeFmt;
}

namespace strata::prefill::mmq {

/// The GPU runs these products (iq_mmq_usable).
bool built();
/// The GPU runs them without matrix engines (iq_mmq_dp4a_usable), where built() says no.
bool built_dp4a();
/// Whether the prompt's products take their int8 form: on the matrix engines where built(); else by dp4a only where the
/// FP16 products have no matrix engines either (built_dp4a, and not kernels::f16_matrix_engines: Volta, whose tensor
/// cores take FP16 and not int8, keeps the FP16 form).  PlainModel's prompt takes it as it is; Qwen3.8's (prefill.cpp)
/// keeps the FP16 form where Intel's XMX take it, faster there than the int8 products.
bool chosen();
/// What runs them where chosen() (kernels::iq_mmq_path), else null.
const char* path();
/// They cover this ggml type (the i-quants, Q2_0, Q8_0, Q5_0, Q2_K to Q6_K).
bool supported(int ggml_type);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (twice that for a type that repeats every block,
/// Q2_K: kernels::iq_mmq_type_dup).
size_t q8_bytes(int64_t rows, int64_t cols);

/// int8 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.  `canon` (the fixed-point products, kernels::iq_mmq_fx): the canonical
/// activations with one scale for each `canon` values (256 a gate/up's, 32 a down's), the CPU's canonical rows'.
/// `col_scale`: each column multiplied by it as it is read (null: none).
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream, int canon = 0, const float* col_scale = nullptr);
/// The rows quantize(x, src, ...) writes for tokens x k_used rows in expert order, written token by token: token t's
/// row of x is quantized once and stored at rows slot[t * k_used + k] (`slot` the inverse of `src`) - the same bytes,
/// a k_used-th of the quantizations (upstream c675f5b7).  STRATA_QUANT_GATHER=1: quantize() over `src`, as before.
void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int ggml_type, int64_t cols,
                      int64_t ld, int64_t tokens, int k_used, void* stream, int canon = 0);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    int64_t rows = 0;   ///< the n experts' rows together (0: total_rows), for the tile's shape
    float* dst = nullptr;
    int64_t ld_dst = 0;
    const uint8_t* const* wtab = nullptr;   ///< on the device: expert e's weights at wtab[e] + woff instead of w's
    size_t woff = 0;
};

/// The faster of the products' forms timed for products of ggml_type, w_rows x w_cols, n experts (1: a dense product)
/// at a few row counts (kernels::iq_mmq_choose_forms): once a GPU and shape, before the first prompt.
void choose_forms(int ggml_type, int64_t w_rows, int64_t w_cols, int n, void* stream);

/// The parts' buffer the stream reserves for the one-expert products of few rows that take K in parts
/// (kernels::iq_mmq_split_reserve), of up to `rows` rows and `n_out` outputs.
bool split_reserve(int64_t rows, int64_t n_out, void* stream);

/// The launch context.  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// Experts [first, n) of a group, each a GGUF-native blob (gate rows at its start, up at `up_off`, down at
/// `down_off`), into their slots of the group buffers (slot q at gu_dst + q * gu_stride and d_dst + q * d_stride), in
/// one launch.  False, and nothing done, when an address or a size is not a multiple of 16 bytes (gather_native then).
/// A null blob is skipped (an expert the products read in place).
constexpr int kGatherGroupMax = 16;
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);
/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).  `canon`: canon_math.hpp's
/// SwiGLU (the CPU's canonical rows' and the decode's, bit for bit).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream, bool canon = false);

/// How a prompt's layer launches its experts' products, for int8_rows_below to time them so: the model's experts, how
/// many the int8 products take a launch, how many the FP16 path dequantizes and multiplies a launch, and whether each
/// expert's blob is first gathered into a group buffer for the int8 products (Qwen3.8-Flash-Next's prompt).
struct ExpertLaunches {
    int64_t n_expert = 0;
    int int8_group = 0, f16_group = 0;
    bool int8_gathers = false;
};
/// From how many rows an expert (a layer's rows over the experts it routes to) the FP16 path - the experts dequantized
/// to FP16 and their grouped products - beats the int8 products on the current GPU, for experts of format `f` (`blob`
/// bytes each) launched as `launches` says: timed the first time a GPU, format and launch shape are asked, and kept.
/// INT64_MAX: int8 at every size; 0: FP16 at every size; `fallback` where the timing cannot run.  Both engines' prompts
/// choose a chunk's layer by it.
int64_t int8_rows_below(const kernels::cpu::NativeFmt& f, size_t blob, int64_t fallback,
                        const ExpertLaunches& launches);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace strata::prefill::mmq
