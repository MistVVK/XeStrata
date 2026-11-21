// include/strata/prefill/moe_fused_iq.hpp - #136: the fused int8 prompt experts (moe_fused.hpp) for the native GGUF
// packs (tools/iq_pack.py: an expert is its raw GGUF slices [gate rows | up rows | down rows], formats per layer in
// expert_layout().fmt), opt-in with the same STRATA_PF_FUSED=1 (NVIDIA sm_80 and newer; sm_75 and HIP: MMQ).
//
// The grouping on the GPU (fused::group), the per-expert pointers (fused::Batch: a cache slot or a ring slot, no
// gather) and the launch shape are the Q2_0 path's.  What differs is the load stage: the i-quant blocks are decoded to
// int8 in shared memory before the mma (codebook and sign lookups, as llama.cpp's MMQ load_tiles does), with the
// block's scale per 32 (or per 16: IQ2_XS, IQ2_S, multiplied at m16n8k16) beside them.  The activations and H are
// int8 per 32 values in their natural order here (quantize_act_native), not the Q2_0 path's permuted one.
//
// Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL - every layer of the IQ2_XS,
// IQ3_XXS and IQ3_S packs but the IQ1_M ones (which MMQ does not cover either).  Other layers keep MMQ.
#pragma once

#include "strata/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// One layer's native expert geometry (cpu::NativeFmt's fields the kernels read; n_embd and n_ff are the compiled geometry's: 2560 and 640 for Flash-Next).
struct NativeGeom {
    int gu_type = -1, d_type = -1;    ///< ggml types
    size_t gu_row = 0, d_row = 0;     ///< bytes per gate/up row and per down row
    size_t up_off = 0, down_off = 0;  ///< inside the blob
};

/// enabled() (STRATA_PF_FUSED=1 on sm_80+), and the kernels cover this gate/up and down pair on this device.
bool native_supported(int gu_type, int d_type);

/// AMD-Q: true when the gfx12 build also takes the IQ2_S and IQ4_XS gate/up layers on the fused kernels (Linux; OPT-IN: STRATA_PF_FUSED_IQ2S=1; STRATA_PF_FUSED=0 turns all fused kernels off).
/// prefill.cpp picks the CPU share's limit from it (share_rules::gfx12_share_limit).
bool gfx12_iq2s_default();

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)): int8 per 32 values in natural order, the native kernels' form.
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// experts() for native blobs: b.blob[e - e0] is the native blob of expert e (gate at 0, up at g.up_off, down at
/// g.down_off; 2-byte aligned).  `xa` from quantize_act_native; H in `ha` (act_bytes(n, n_ff)); down into `dm` at the
/// rows of `group`.  `dm` 16-byte aligned (gfx12 stores float4; checked there - the prompt path's buffers are 256-aligned
/// through Alloc's 256-byte takes from a 256-aligned cache slot or cudaMalloc).
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);
/// gfx11 F16 kernels: the next experts_native calls may write the down rows as FP16 (2,560 halves a row, saturated at +-65504) instead of FP32 when the
/// down runs on an F16 kernel; dm_f16_used() says which form the last call wrote (false on every other path), for moe_combine16.
void want_dm_f16(bool on);
bool dm_f16_used();
/// STRATA_PF_P1A=1 on gfx1151 (HIP; a default on Windows gfx1151, false everywhere else): the K-quant layers may take moe_p1a.cu's kernels.
bool p1a_wanted();
/// For the next experts_native calls (when p1a_wanted()): the layer input as FP16 rows [token][n_embd] (nullptr: none) and the
/// bytes of `ha`.  A call takes the P1A kernels only with an input, want_dm_f16(true) and room for FP16 H; p1a_used() says whether
/// the last call did (it wrote FP16 down rows then, dm_f16_used() is true).
void p1a_input(const void* x16, size_t h_bytes);
bool p1a_used();
/// p1a_wanted() and a layer of this gate/up + down type pair would take the P1A kernels (the buffers grow only for such a pack).
bool p1a_takes_pair(int gu_type, int d_type);
/// STRATA_PF_F16=2 on gfx11 WMMA targets (gfx1100..gfx1151): 128-token pair FP16 WMMA kernel (native_h11p_kernel).
bool h11p_wanted();
bool h11p_takes_pair(int gu_type, int d_type);
/// (CUDA, STRATA_PF_FUSED_TQ=1) a Hadamard-folded down for the next experts_native calls of a trellis pair: the gate/up writes H as FP32
/// into the first n_ff floats of each row's own down-output row in `dm` (no other buffer), which is rotated there (y = H_128 (signs * h) /
/// sqrt(128) per 128 features; signs nullptr = identity, else n_ff values on the device) and quantized into `ha` before the down product
/// rewrites the row.  Off by default; set_down_rotation(false, nullptr) after the layer.
void set_down_rotation(bool on, const float* signs);
/// (CUDA) the same rotation in place on `rows` FP32 rows of `cols` values (a multiple of 128), on `stream`: a folded gate/up's input before
/// quantize_act_native, or (tests) MMQ's H.
void hadamard128_rows(float* x, int64_t rows, int64_t cols, const float* signs, void* stream);
/// The NT = 256 kernels (STRATA_PF_FUSED_NT=256 or _F16=1) or the warp-specialized kernel (STRATA_PF_FUSED_WS=1) are
/// set up on the current device: experts_native takes a WideBatch (STRATA_PF_BATCH up to kBatchRoom experts a launch).
bool native_wide();
/// Launches of experts_native that took the unstaged 64-row kernel because the weight-staging check failed (an expert blob not in device memory, or an odd address / row / offset):
/// 0 in every default configuration; 0 in a HIP build (no staging kernels).  The first such launch logs one line to stderr; the process does not exit.  For the GPU test.
int staged_blob_fallbacks();
/// experts_native() of a WideBatch, on the NT = 256 kernels or the warp-specialized kernel; only when native_wide().
void experts_native(const WideBatch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);

}  // namespace strata::prefill::fused
