// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The Xe prompt path keeps GGUF weights quantized and multiplies q8_1 activation rows with
// the integer dot products used by decode. IQ1_M layers keep the FP16 path.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the Xe quantized prompt products.
bool built();
/// Supported GGUF weight types; IQ1_M is not covered.
bool supported(int ggml_type);
/// Supported type and a positive row count; these kernels need no weight tile in local memory.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

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
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// One launch context per prompt path.
class Context {
public:
    Context() = default;
    ~Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// gather_native for an MMQ group's experts [first, n) in ONE launch: expert q's blob (`blob[q]`; gate at +0, up at
/// +up_off, down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride - the same bytes as one gather_native
/// each.  Every pointer, offset and size 16-byte aligned (false otherwise: nothing launched, gather one at a time).
constexpr int kGatherGroupMax = 16;
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace strata::prefill::mmq
