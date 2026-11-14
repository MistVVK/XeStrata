// src/prefill/moe_helpers.cu - see include/strata/prefill/moe_mmq.hpp.  The prompt experts' helpers that need nothing of llama.cpp's
// ggml-cuda: the gathers of expert blobs into group buffers, the SwiGLU passes, the row table and the buffer sizes.  P4-SPLIT: cut
// unchanged from moe_mmq.cu into Strata's own prompt library (strata_pf_own), so that a build without llama.cpp's MMQ keeps them.
#include "strata/prefill/moe_mmq.hpp"

#include "strata/prefill/moe_fused.hpp"   // kMoeEmbd, kMoeFF: the compiled geometry's widths

#include "strata/quant/types.hpp"   // the block geometry of a type id (Strata's own type table, no ggml)

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// MMQ's q8_1 activation rows, for the buffer sizes: block_q8_1_mmq is 4 scales and 128 int8 values (144 bytes) for 4 x QK8_1 = 128
// values (ggml-cuda/mmq.cuh; moe_mmq.cu checks both numbers against the real struct where it is compiled)
constexpr size_t kQ8_1MmqBytes = 144;
constexpr int kQK8_1 = 32;

// ... plus, in the same launch, the expert's 16-byte NVFP4 tail (tail -> tail_dst) and the zeroed MMQ tails after
// the group's last slot (nz uint4 after ab_dst's and c_dst's copies): one launch per expert instead of a kernel,
// a 16-byte copy and, per group, two memsets.
__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst,
                              const uint4* __restrict__ tail, uint4* __restrict__ tail_dst, int64_t nz) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = na + nb + nc;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < n) c_dst[i - na - nb] = c[i - na - nb];
    else if (i < n + nz) ab_dst[na + nb + (i - n)] = make_uint4(0, 0, 0, 0);
    else if (i < n + 2 * nz) c_dst[nc + (i - n - nz)] = make_uint4(0, 0, 0, 0);
    else if (i == n + 2 * nz && tail != nullptr) *tail_dst = *tail;
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y); the last one (n - 1) also zeroes nz uint4
// after its gu and d slots, and each copies its NVFP4 tail (tail_off, in uint4; tail_dst null = none)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride, tail_off;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int n, int64_t na, int64_t nc, int64_t nz,
                                    uint4* __restrict__ gu_dst, uint4* __restrict__ d_dst, uint4* __restrict__ tail_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = 2 * na + nc;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < m) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
    else if (i < m + nz) { if (q == n - 1) ab[ga.gu_stride + (i - m)] = make_uint4(0, 0, 0, 0); }
    else if (i < m + 2 * nz) { if (q == n - 1) cd[ga.d_stride + (i - m - nz)] = make_uint4(0, 0, 0, 0); }
    else if (i == m + 2 * nz && tail_dst != nullptr) tail_dst[q] = src[ga.tail_off];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [2 F][E / 4 B], down codes [E][F / 4 B], gate/up scales [2 F][E / 64] f16, down scales
// [E][F / 64] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel; E, F the compiled geometry's: Flash-Next
// [1280][640 B], [2560][160 B], [1280][40], [2560][10]).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
constexpr int Q2_GU_ROWS = fused::kMoeGuRows, Q2_ROW_GU = fused::kMoeEmbd / 4, Q2_ROW_D = fused::kMoeFF / 4, Q2_SC_GU = fused::kMoeEmbd / 64, Q2_SC_D = fused::kMoeFF / 64;
static_assert(fused::kMoeEmbd % 64 == 0 && fused::kMoeFF % 64 == 0, "the Strata Q2 blob holds whole 64-weight blocks");
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) Q2_GU_ROWS * Q2_ROW_GU, O_GU_SC = O_D_CODES + (size_t) fused::kMoeEmbd * Q2_ROW_D,
                     O_D_SC = O_GU_SC + (size_t) Q2_GU_ROWS * Q2_SC_GU * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = (int64_t) Q2_GU_ROWS * Q2_SC_GU, n_d = (int64_t) fused::kMoeEmbd * Q2_SC_D;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / Q2_SC_GU, b = i % Q2_SC_GU;
        codes = blob + row * Q2_ROW_GU + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * Q2_SC_GU + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / Q2_SC_D, b = j % Q2_SC_D;
        codes = blob + O_D_CODES + row * Q2_ROW_D + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * Q2_SC_D + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

// swiglu_kernel on the MMQ gate/up output with each expert's gate and up scales applied as it is read: the products
// scale_gu_rows stored, without the pass that stored them
__global__ void swiglu_scaled_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                                     const int32_t* __restrict__ bounds, int n, const float* __restrict__ tails,
                                     int64_t row0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff, ra = row0 + r;
    int q = 0;
    while (q + 1 < n && ra >= bounds[q + 1]) ++q;
    const float* row = gu + r * 2 * n_ff;
    const float g = row[k] * tails[4 * q], u = row[n_ff + k] * tails[4 * q + 1];
    h[i] = g / (1.0f + __expf(-g)) * u;
}
__global__ void down_row_scales_kernel(float* __restrict__ sd, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails, int64_t nrows) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    sd[r] = tails[4 * q + 2];
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

// The block geometry comes from Strata's type table (src/quant/types.cpp): the GGUF format's ids with ggml's sizes, the trellis types
// 144-149 included (128 weights; 34 / 26 / 30 / 42 / 50 / 58 bytes); tests/quant/type_geometry_oracle.cpp checks it against ggml's
// own table.  An id the table does not know has no size here (0): no prompt path reads such a matrix.
size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    const ::strata::quant::TypeInfo* ti = ::strata::quant::type_info(t);
    if (ti == nullptr) return 0;
    return (size_t) rows * (size_t) (cols / ti->blck) * ti->bytes;
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * kQ8_1MmqBytes / (4 * kQK8_1) + 128 * kQ8_1MmqBytes;
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail, void* tail_dst, size_t zero_bytes) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      (uintptr_t) tail | (uintptr_t) tail_dst | gu_half_bytes | d_bytes | zero_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc + 2 * nz + 1), 256, 0, s>>>(
            (const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst, (const uint4*) down, nc, (uint4*) d_dst,
            (const uint4*) tail, (uint4*) tail_dst, nz);
    } else {
        if (tail) cudaMemcpyAsync(tail_dst, tail, 16, cudaMemcpyDeviceToDevice, s);
        if (zero_bytes) {
            cudaMemsetAsync((uint8_t*) gu_dst + 2 * gu_half_bytes, 0, zero_bytes, s);
            cudaMemsetAsync((uint8_t*) d_dst + d_bytes, 0, zero_bytes, s);
        }
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream, size_t tail_off,
                         void* tail_dst, size_t zero_bytes) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    if (tail_off == 0) tail_dst = nullptr;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride |
                  tail_off | (uintptr_t) tail_dst | zero_bytes;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    ga.tail_off = (int64_t) tail_off / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
    const int64_t work = 2 * na + nc + 2 * nz + (tail_dst != nullptr ? 1 : 0);
    copy16_group_kernel<<<dim3(blocks(work), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, g.n, na, nc, nz, (uint4*) gu_dst, (uint4*) d_dst, (uint4*) tail_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks((int64_t) Q2_GU_ROWS * Q2_SC_GU + (int64_t) fused::kMoeEmbd * Q2_SC_D), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "swiglu_scaled");
}
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    down_row_scales_kernel<<<blocks(nrows), 256, 0, (cudaStream_t) stream>>>(sd, bounds, n, tails, nrows);
    ck(cudaGetLastError(), "down_row_scales");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace strata::prefill::mmq
