// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses (2560 values as Q8_K: 10 x 292; a BF16 layer's
/// activation is BF16, llm-jp-4.1's 2560 values 5120 bytes).
inline constexpr size_t kNativeActBytes = 8192;
/// One token's gate/up activation, as native_quant_act writes it for the layer.  Aligned for the Q2_0 rows' ActQ.
struct alignas(64) NativeAct {
    uint8_t bytes[kNativeActBytes];
};
/// Bytes of the largest quantized down activation (640 values as Q8_0: 20 x 34, or Q8_K 3 x 292; llm-jp-4.1's 960
/// as Q8_0: 30 x 34, as BF16 1920; GLM-5.3-Flash's 2048 as Q8_0: 64 x 34, as Q8_K 8 x 292).
inline constexpr size_t kNativeHBytes = 4096;
/// The widest expert intermediate a native layer may have (GLM-5.3-Flash: 2048); the pool's row buffers are this wide.
inline constexpr int64_t kNativeMaxFF = 2048;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    /// The up rows' type: gu_type, or another one with the same activation format (Ornith-1.5's GSQ-RCO files mix
    /// Q2_K, Q3_K and Q4_K between gate and up in some layers).  gu_row and up_off are the gate rows'.
    int up_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_row = 0;                  ///< bytes per up row (gu_row when up_type is gu_type)
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, or the ActQ; n_ff of d_act)
    /// NVFP4 (ggml type 40) only: the blob ends with {s_gate, s_up, s_down, 0} at this offset, the per-expert
    /// weight_scale_2 the block format has no room for (tools/iq_pack.py). 0 = no tail. Each scale goes on its own
    /// projection's FP32 output: gate * s_gate, up * s_up, down * s_down. Do not fold s_down into up: the hidden
    /// drops to ~1e-5 and its q8 block scale becomes an FP16 subnormal (expert error 2-12% instead of 1.1%).
    size_t tail_off = 0;
    /// prism.hadamard (hadamard.hpp): the experts of this layer are stored in a rotated basis, and
    /// native_quant_act / native_quant_h apply y = H_block (signs * x) before quantizing.  had_block 0 = none.
    int had_block = 0;
    bool had_gu = false, had_d = false;
    const float* gu_signs = nullptr;    ///< n_embd signs for x (nullptr: identity)
    const float* d_signs = nullptr;     ///< n_ff signs for h (nullptr: identity)
    /// The latent MoE's Q3_K shared column scales (native_experts.txt v6, scales/l*.f32): the gate/up rows are
    /// A*diag(s), so the D stage's rows take 1/s before any consumer reads them (the vector is the reciprocals,
    /// n_embd floats, layer-major in the layout's storage).  nullptr: the pack has no column scales.
    const float* gu_colscale = nullptr;
};

/// Whether the layer's gate and up rows are of different types (a NativeFmt filled by native_fmt; one built another
/// way leaves up_type at -1: the gate's).
inline bool native_mixed_gu(const NativeFmt& f) { return f.up_type >= 0 && f.up_type != f.gu_type; }

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.  `up_type` (-1: the gate's)
/// is the up rows' when they differ, which must quantize the activation the same way.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                int up_type = -1);

/// One row of `n` values of a GGUF tensor of ggml type `type` to floats, by ggml's own dequantizer (F32 copied): the
/// one host lookup of an embedding row, for every table that stays in host memory (the plain models' and the image
/// engine's text readers').  False when ggml has none for the type or `n` is not whole blocks.
bool native_row_to_float(int type, const void* src, float* dst, int64_t n);
/// Bytes of one row of `n` values of ggml type `type` (0 when `n` is not whole blocks).
size_t native_row_bytes(int type, int64_t n);

/// A dense matrix's rows of ggml type `type` read where they are (a GGUF tensor), as ggml-cpu multiplies them: the
/// activation quantized to the type's vec_dot format, each row's dot by ggml-cpu's own vec_dot (the K-quants' AVX2 and
/// AVX-VNNI kernels).  native_vec_act_bytes: the bytes of one activation of `n` values in that format (0: ggml-cpu
/// has no vec_dot for the type).
size_t native_vec_act_bytes(int type, int64_t n);
/// x (n floats) to the activation native_vec_rows reads for `type`, in `dst` (native_vec_act_bytes bytes).
void native_vec_quant(int type, const float* x, void* dst, int64_t n);
/// y[r] = row r . act for the rows [r0, r1) of `rows` (row_bytes apart, n values each).
void native_vec_rows(int type, const uint8_t* rows, size_t row_bytes, int64_t n, int64_t r0, int64_t r1,
                     const void* act, float* y);

/// x (n_embd floats) -> the gate/up activation the layer's rows read (act_bytes), rotated first when the layer is
/// Hadamard-folded: a Q2_0 layer's is Strata's ActQ, any other layer's its vec_dot_type.  Every caller gets the right one from here, so none chooses by type.  `dst` is
/// a NativeAct's bytes, or kNativeActBytes as aligned.
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes), rotated first when the layer is Hadamard-folded.
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
