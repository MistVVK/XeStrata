// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/elementwise.hpp - the small ops a LAYER needs between its GEMVs, P2.S5.
//
// These are not interesting kernels and that is why they are in one file with one parity test: they are the
// glue in the per-layer graph, and glue that is written inline at each call site is glue with several
// slightly different float conventions.
//
//   * `f32_to_f16_bulk` is what lets the f32 arithmetic of the mixer feed the fp16-activation GEMVs.
//     `docs/activation-contract.md` is explicit that this is NOT the contract for a quantized weight - it is
//     the bridge between two stages that both already exist, and the contract decision is a separate one.
#pragma once

#include <cstdint>

namespace strata::kernels {

// Xe entry points take &core::Runtime::get().compute() as the opaque stream.
// A null stream uses that queue and waits for completion. Unported entry points
// are not linked into strata_kernels yet.

/// `x[i] *= s`, in place.
void scale_inplace(float* x, int64_t n, float s, void* stream);
/// `dst[i] += a * src[i]` (a 1: the plain sum, its bits; the one residual add of the LLMs and the image engine).
/// **ADDED FOR R4's HIT/MISS SPLIT, WHERE THE TWO HALVES COME FROM DIFFERENT
/// ENGINES.**  The CPU writes the misses into `parts` and the GPU writes the resident experts into a
/// buffer of their own, so that the GPU's work can run CONCURRENTLY with the CPU's instead of after it -
/// and the two are summed here, on the stream, immediately before `moe_combine` reads the result.
///
/// The alternative - having the hit kernel write straight into `parts` - costs the whole overlap, because
/// the copy of the misses into `parts` cannot happen until the pool has produced them, and the kernel is
/// then ordered behind that copy.  Measured: the drain fell 18.2 -> 10.2 ms and the token did not move.
void add_inplace(float* dst, const float* src, int64_t n, void* stream, float a = 1.0f);

/// `y[i] = f16(x[i])`, round-to-nearest-even, using the shared conversion in `f16_bits.hpp`.
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);

/// *seq = v in MAPPED PINNED memory - THE DOORBELL.  A KERNEL and not cudaEventRecord, because round 199
/// found that an event record inside a capture is SILENTLY DROPPED while a kernel is captured normally, and
/// round 209 measured the host seeing this one's result 0.050 ms into a 39.8 ms graph.
///
/// It lives here because the caller that needs it - moe_layer - is a HOST translation unit, where
/// __global__ and <<<>>> do not exist.
/// *seq += 1 - AN INCREMENT, NOT A STORE OF A CALLER-SUPPLIED VALUE.  A value passed in is evaluated on the
/// HOST at CAPTURE time, so a captured graph would ring the same number on every replay.  Reading and
/// incrementing in memory is what makes the count advance under cudaGraphLaunch.
void doorbell_ring(uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: one thread spins until the host-written `*d_flag` equals the ring value `*d_seq`
/// (both mapped pinned memory), then fences so everything the host wrote before the flag is visible to the
/// nodes that follow.  Captured once per layer between `pre[l]` and the parts copy; it is what lets one graph
/// hold the whole token (bench/micro/device_wait.cu: 5.2 us per handoff, no driver call).
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: copy `n` floats from MAPPED pinned host memory (`src` is its device pointer) into
/// device memory with a kernel, so the handoff stays on the compute queue.
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream);
/// `rows` rows of `width` floats from mapped memory, except the rows listed in hit_rows[0, *count) (device),
/// which are written +0.0 instead (a verify window's GPU-computed entries: the pool leaves zeros there).
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream);

/// Plan v0.3 P3: the doorbell's payload and its ring in ONE kernel.  Copies `x` (n floats), `ids` and `weights`
/// (k each) into the mapped host regions, fences, and increments the mapped sequence number - replacing three
/// device-to-host memcpy nodes (copy-engine operations in the middle of the layer chain) and the ring kernel.
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream);

/// Plan v0.3 P3: copy `n` int32 from mapped pinned host memory into device memory with a kernel (the QSA
/// per-token step and positions), instead of a host-to-device memcpy node in the middle of a layer.
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream);
/// `*flag = value` from the device, behind the stream's earlier work (a copy's landing told through mapped host
/// memory: `flag` is its device alias), with a system-scope fence before the store.
void mapped_flag_store(uint32_t* flag, uint32_t value, void* stream);
/// Up to 8 copies from mapped host memory in one launch: `words` 4-byte words each (16 bytes a load where both ends
/// are 16-byte aligned and the count a multiple of 4).  Graph-capturable.
struct MappedCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    int64_t words = 0;
};
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream);
/// `rows` rows of 2n floats: row r of `a` (n floats), then row r of `b` (the MTP layer's input, [embedding | hidden]).
/// n a multiple of 4, every pointer 16-byte aligned.  Graph-capturable.
void concat_rows(float* dst, const float* a, const float* b, int64_t n, int64_t rows, void* stream);

}  // namespace strata::kernels
