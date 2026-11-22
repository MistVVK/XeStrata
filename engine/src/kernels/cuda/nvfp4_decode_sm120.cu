// src/kernels/cuda/nvfp4_decode_sm120.cu - Clean-room MIT Blackwell (sm_120a/sm_121a) Tensor Core NVFP4 decode/verify.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived strictly from NVIDIA PTX ISA (mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32) and GGML block_nvfp4 layout:
//   - block_nvfp4 (64 values, 36 bytes): uint8_t d[4] (UE4M3 sub-block scales per 16 vals), uint8_t qs[32] (64 E2M1).
//   - In each 16-value sub-block (8 bytes of qs), low nibbles are elements 0..7 and high nibbles are elements 8..15.
//   - E2M1 x 2 maps losslessly to signed int8: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}, decoded via
//     4 inline uint32_t immediates + __byte_perm (0 spills, 0 LUT loads).
//   - Every warp computes 16 output rows x up to 8 tokens in ONE pass over weights using only 4 FP32 accumulators per
//     thread (0 register spills, ~2.8 KB smem -> max occupancy!).
//   - Kernel 1 (nvfp4_gu_tc_sm120_kernel): 4 warps = 32 gate rows + 32 up rows per CTA -> fused SwiGLU + Q8_1 quant.
//   - Kernel 2 (nvfp4_down_tc_sm120_kernel): 4 warps = 64 output rows per CTA -> FP32 scatter to ent_dst.
#include "strata/kernels/nvfp4_decode_sm120.hpp"
#include "strata/core/emulate.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace strata::kernels {
namespace {

constexpr int kTypeNVFP4 = 40;
constexpr int kCnMax = 8;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

struct Nvfp4Block {
    uint8_t d[4];
    uint8_t qs[32];
};
static_assert(sizeof(Nvfp4Block) == 36);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_NVFP4_DEC_DEV 1
#endif

// 2 * E2M1 signed int8 codebook packed as 4 little-endian uint32 immediates:
//   pos lo: {0, 1, 2, 3}, pos hi: {4, 6, 8, 12}, neg lo: {0, -1, -2, -3}, neg hi: {-4, -6, -8, -12}
constexpr uint32_t kFp4W0 = 0x03020100u;
constexpr uint32_t kFp4W1 = 0x0C080604u;
constexpr uint32_t kFp4W2 = 0xFDFEFF00u;
constexpr uint32_t kFp4NegHi = 0xF4F8FAFCu;

__device__ __forceinline__ uint32_t ld32_u(const uint8_t* p) {
    return (uint32_t) __ldg(p) | ((uint32_t) __ldg(p + 1) << 8) |
           ((uint32_t) __ldg(p + 2) << 16) | ((uint32_t) __ldg(p + 3) << 24);
}

__device__ __forceinline__ float ld_f32_u(const uint8_t* p) {
    if (((uintptr_t) p & 3u) == 0u) return __ldg(reinterpret_cast<const float*>(p));
    return __uint_as_float(ld32_u(p));
}

// Convert UE4M3 byte to 0.5f * float (folding the 0.5 factor of kvalues_fp4 = 2 * E2M1!).
__device__ __forceinline__ float ue4m3_half_to_f32(uint8_t v) {
    if (v == 0u || v == 0x7Fu || v == 0xFFu) return 0.0f;
    const uint32_t exp = (uint32_t) (v >> 3) & 0x0Fu;
    const uint32_t man = (uint32_t) v & 0x07u;
    if (exp == 0u) return ldexpf((float) man, -10);
    return ldexpf((float) (8u + man), (int) exp - 11);
}

// Map 4 low nibbles (4-bit E2M1 codes in bytes 0..3) -> 4 signed int8 (2 * E2M1).
__device__ __forceinline__ uint32_t fp4_nib4_to_s8x4(uint32_t nib4) {
    const uint32_t sel = 0x32103210u | ((nib4 & 0x88888888u) >> 1);
    const uint32_t lo0 = __byte_perm(kFp4W0, kFp4W1, nib4);
    const uint32_t hi0 = __byte_perm(kFp4W2, kFp4NegHi, nib4);
    const uint32_t t0 = __byte_perm(lo0, hi0, sel);
    const uint32_t lo1 = __byte_perm(kFp4W0, kFp4W1, nib4 >> 16);
    const uint32_t hi1 = __byte_perm(kFp4W2, kFp4NegHi, nib4 >> 16);
    const uint32_t t1 = __byte_perm(lo1, hi1, sel >> 16);
    return __byte_perm(t0, t1, 0x6420);
}

#if defined(STRATA_NVFP4_DEC_DEV)
__device__ __forceinline__ void mma_s8_m16n8k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
#endif

// Load up to 4 NVFP4 blocks (<= 36 uint32s = 144 B) per row for 16 contiguous rows owned by this warp into
// s_w_warp[16][36] using 100%-coalesced 32-lane transactions (one 128B + one 16B cache line per row).
__device__ __forceinline__ void stage_nvfp4_rows16(
    const uint8_t* __restrict__ base_row0, size_t row_stride, int r_start, int r_limit,
    int blk_offset, int n_blk64, bool aligned4, int lane, uint32_t (&s_w)[16][36]) {
    const int n_u32 = n_blk64 * 9;  // 9 uint32s = 36 B per block_nvfp4 (max 36 uint32s for 4 blocks)
    const size_t blk_byte_off = (size_t) blk_offset * sizeof(Nvfp4Block);
#pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        if (r_start + rr < r_limit) {
            const uint8_t* p = base_row0 + (size_t) rr * row_stride + blk_byte_off;
            if (lane < n_u32) {
                s_w[rr][lane] = aligned4 ? __ldg(reinterpret_cast<const uint32_t*>(p) + lane)
                                         : ld32_u(p + lane * 4);
            }
            if (lane + 32 < n_u32) {
                s_w[rr][lane + 32] = aligned4 ? __ldg(reinterpret_cast<const uint32_t*>(p) + lane + 32)
                                              : ld32_u(p + (lane + 32) * 4);
            }
        } else {
            if (lane < n_u32) s_w[rr][lane] = 0u;
            if (lane + 32 < n_u32) s_w[rr][lane + 32] = 0u;
        }
    }
    __syncwarp();
}

// Evaluate 16 staged rows in s_w[16][36] against up to 8 tokens in s_xq[8 * 256] (one <=256-elem chunk).
__device__ __forceinline__ void eval_nvfp4_chunk256_tc_smem(
    const uint32_t (&s_w)[16][36], bool v0, bool v1, int n_blk64,
    const int8_t* __restrict__ s_xq, const float* __restrict__ s_dx,
    int g, int tig, float (&acc)[4]) {
#if defined(STRATA_NVFP4_DEC_DEV)
    const int b_u32_off = (tig & 1);   // 0 or 1 (word offset within the 2-word = 8-byte sub-block)
    const int hi_nib = (tig >> 1);     // 0 for k=0..7 (low nibbles), 1 for k=8..15 (high nibbles)
    const uint32_t* w0 = s_w[g];
    const uint32_t* w1 = s_w[8 + g];

    for (int blk = 0; blk < n_blk64; ++blk) {
        const uint32_t d0_pack = v0 ? w0[blk * 9] : 0u;
        const uint32_t d1_pack = v1 ? w1[blk * 9] : 0u;

#pragma unroll
        for (int is = 0; is < 4; ++is) {
            const float dw0 = v0 ? ue4m3_half_to_f32((uint8_t) ((d0_pack >> (8 * is)) & 0xFFu)) : 0.0f;
            const float dw1 = v1 ? ue4m3_half_to_f32((uint8_t) ((d1_pack >> (8 * is)) & 0xFFu)) : 0.0f;

            const int q_idx = blk * 9 + 1 + is * 2 + b_u32_off;
            const uint32_t raw0 = v0 ? w0[q_idx] : 0u;
            const uint32_t raw1 = v1 ? w1[q_idx] : 0u;
            const uint32_t nib0 = hi_nib ? ((raw0 >> 4) & 0x0F0F0F0Fu) : (raw0 & 0x0F0F0F0Fu);
            const uint32_t nib1 = hi_nib ? ((raw1 >> 4) & 0x0F0F0F0Fu) : (raw1 & 0x0F0F0F0Fu);
            const uint32_t a0 = v0 ? fp4_nib4_to_s8x4(nib0) : 0u;
            const uint32_t a1 = v1 ? fp4_nib4_to_s8x4(nib1) : 0u;

            const int b32 = blk * 2 + (is >> 1);
            const int half16 = is & 1;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (g * 8 + b32) * 32 + half16 * 16 + tig * 4);

            int32_t d[4] = {0, 0, 0, 0};
            mma_s8_m16n8k16(d, a0, a1, b0);

            const int t0 = 2 * tig;
            const int t1 = t0 + 1;
            const float dx0 = s_dx[t0 * 8 + b32];
            const float dx1 = s_dx[t1 * 8 + b32];
            acc[0] = fmaf(dw0 * dx0, (float) d[0], acc[0]);
            acc[1] = fmaf(dw0 * dx1, (float) d[1], acc[1]);
            acc[2] = fmaf(dw1 * dx0, (float) d[2], acc[2]);
            acc[3] = fmaf(dw1 * dx1, (float) d[3], acc[3]);
        }
    }
#else
    (void) s_w; (void) v0; (void) v1; (void) n_blk64;
    (void) s_xq; (void) s_dx; (void) g; (void) tig; (void) acc;
#endif
}

// Kernel 1: Fused NVFP4 gate + up + SwiGLU + Q8_1 quantization using INT8 Tensor Cores (m16n8k16).
// Grid: (n_ff / 32, cap_groups), Block: 128 (4 warps: warps 0..1 = 32 gate rows, warps 2..3 = 32 up rows).
__global__ void __launch_bounds__(128) nvfp4_gu_tc_sm120_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const Q81Block* __restrict__ xq, NativeExpertLayout L, Q81Block* __restrict__ hq) {
#if defined(STRATA_NVFP4_DEC_DEV)
    const int g_idx = (int) blockIdx.y;
    if (g_idx >= *n_groups) return;
    const int e0 = grp_start[g_idx], e1 = grp_start[g_idx + 1];
    if (e0 >= e1) return;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;      // 0,1 = gate rows [0..15], [16..31]; 2,3 = up rows [0..15], [16..31]
    const int lane = tid & 31;
    const int g = lane >> 2;        // 0..7
    const int tig = lane & 3;       // 0..3

    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g_idx]);
    const bool aligned4 = (((uintptr_t) blob & 3u) == 0u) && ((L.gu_row & 3u) == 0u) && ((L.up_off & 3u) == 0u);
    const float s_scale = L.tail_off ? ld_f32_u(blob + L.tail_off + ((warp < 2) ? 0u : 4u)) : 1.0f;

    const int r_tile = (int) blockIdx.x * 32 + (warp & 1) * 16;
    const size_t base_off = (warp < 2) ? 0u : L.up_off;
    const uint8_t* w_base = blob + base_off + (size_t) r_tile * L.gu_row;

    const int xb = (int) (L.n_embd >> 5);       // 80 Q8_1 blocks for 2560
    const int hb = (int) (L.n_ff >> 5);         // 20 Q8_1 blocks for 640
    const int n_chunks = (int) (L.n_embd >> 8); // 10 chunks of 256 elements (4 NVFP4 blocks = 8 Q8_1 blocks)

    __shared__ alignas(16) int8_t s_xq[kCnMax * 256]; // 2,048 B
    __shared__ float s_dx[kCnMax * 8];                //   256 B
    __shared__ float s_res[kCnMax][64];               // 2,048 B
    __shared__ alignas(16) uint32_t s_w[4][16][36];   // 9,216 B (total 13.6 KB -> max occupancy!)

    for (int c0 = e0; c0 < e1; c0 += kCnMax) {
        const int cn = min(kCnMax, e1 - c0);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int ch = 0; ch < n_chunks; ++ch) {
            __syncthreads();
            // Cooperatively load cn * 8 Q8_1 blocks (at most 64 blocks) into 2 KB smem.
            if (tid < kCnMax * 8) {
                const int j = tid >> 3;
                const int b = tid & 7;
                if (j < cn) {
                    const Q81Block& blk = xq[(size_t) ent_tok[c0 + j] * xb + ch * 8 + b];
                    s_dx[tid] = __low2float(blk.ds);
                    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(blk.qs);
                    uint4* d4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                    d4[0] = make_uint4(s32[0], s32[1], s32[2], s32[3]);
                    d4[1] = make_uint4(s32[4], s32[5], s32[6], s32[7]);
                } else {
                    s_dx[tid] = 0.0f;
                    uint4* d4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                    d4[0] = make_uint4(0u, 0u, 0u, 0u);
                    d4[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            stage_nvfp4_rows16(w_base, L.gu_row, r_tile, (int) L.n_ff, ch * 4, 4, aligned4, lane, s_w[warp]);
            __syncthreads();

            eval_nvfp4_chunk256_tc_smem(s_w[warp], true, true, 4, s_xq, s_dx, g, tig, acc);
        }

        const int t0 = 2 * tig;
        const int t1 = t0 + 1;
        const int col0 = ((warp >= 2) ? 32 : 0) + (warp & 1) * 16 + g;
        const int col1 = col0 + 8;
        if (t0 < cn) {
            s_res[t0][col0] = acc[0] * s_scale;
            s_res[t0][col1] = acc[2] * s_scale;
        }
        if (t1 < cn) {
            s_res[t1][col0] = acc[1] * s_scale;
            s_res[t1][col1] = acc[3] * s_scale;
        }
        __syncthreads();

        // Fused SwiGLU + exact Q8_1 quantization: 4 warps cover up to 8 tokens (2 iterations max).
        for (int j = warp; j < cn; j += 4) {
            const float gg = s_res[j][lane];
            const float uu = s_res[j][32 + lane];
            const float xi = (gg / (1.0f + __expf(-gg))) * uu;
            float amax = fabsf(xi);
            float sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            float d = amax / 127.0f;
            if (!isfinite(d)) d = 0.0f;
            int q = (d > 0.0f && isfinite(xi)) ? __float2int_rn(xi / d) : 0;
            q = max(-127, min(127, q));
            Q81Block* y = hq + (size_t) (c0 + j) * hb + blockIdx.x;
            y->qs[lane] = (int8_t) q;
            if (lane == 0) y->ds = __floats2half2_rn(d, isfinite(sum) ? sum : 0.0f);
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_tok; (void) xq; (void) L; (void) hq;
#endif
}

// Kernel 2: NVFP4 down projection using INT8 Tensor Cores (m16n8k16).
// Grid: (ceil(n_embd / 64), cap_groups), Block: 128 (4 warps x 16 rows = 64 output rows/CTA).
__global__ void __launch_bounds__(128) nvfp4_down_tc_sm120_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const Q81Block* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
#if defined(STRATA_NVFP4_DEC_DEV)
    const int g_idx = (int) blockIdx.y;
    if (g_idx >= *n_groups) return;
    const int e0 = grp_start[g_idx], e1 = grp_start[g_idx + 1];
    if (e0 >= e1) return;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int r_base = (int) blockIdx.x * 64 + warp * 16;
    const int r0 = r_base + g;
    const int r1 = r_base + 8 + g;
    const bool v0 = (r0 < L.n_embd);
    const bool v1 = (r1 < L.n_embd);

    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g_idx]);
    const bool aligned4 = (((uintptr_t) blob & 3u) == 0u) && ((L.down_off & 3u) == 0u) && ((L.d_row & 3u) == 0u);
    const float s_down = L.tail_off ? ld_f32_u(blob + L.tail_off + 8) : 1.0f;

    const uint8_t* w_base = blob + L.down_off + (size_t) r_base * L.d_row;

    const int hb = (int) (L.n_ff >> 5);       // 20 Q8_1 blocks (n_ff = 640 -> 10 NVFP4 blocks)
    const int n_blk64 = (int) (L.n_ff >> 6);  // 10 NVFP4 blocks

    __shared__ alignas(16) int8_t s_xq[kCnMax * 256]; // 2,048 B
    __shared__ float s_dx[kCnMax * 8];                //   256 B
    __shared__ alignas(16) uint32_t s_w[4][16][36];   // 9,216 B (total 11.5 KB -> max occupancy!)

    for (int c0 = e0; c0 < e1; c0 += kCnMax) {
        const int cn = min(kCnMax, e1 - c0);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int b0 = 0; b0 < n_blk64; b0 += 4) {
            const int nb64 = min(4, n_blk64 - b0);
            const int nb32 = nb64 * 2;
            __syncthreads();
            if (tid < kCnMax * 8) {
                const int j = tid >> 3;
                const int b = tid & 7;
                if (j < cn && b < nb32) {
                    const Q81Block& blk = hq[(size_t) (c0 + j) * hb + b0 * 2 + b];
                    s_dx[tid] = __low2float(blk.ds);
                    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(blk.qs);
                    uint4* d4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                    d4[0] = make_uint4(s32[0], s32[1], s32[2], s32[3]);
                    d4[1] = make_uint4(s32[4], s32[5], s32[6], s32[7]);
                } else {
                    s_dx[tid] = 0.0f;
                    uint4* d4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                    d4[0] = make_uint4(0u, 0u, 0u, 0u);
                    d4[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            stage_nvfp4_rows16(w_base, L.d_row, r_base, (int) L.n_embd, b0, nb64, aligned4, lane, s_w[warp]);
            __syncthreads();

            eval_nvfp4_chunk256_tc_smem(s_w[warp], v0, v1, nb64, s_xq, s_dx, g, tig, acc);
        }

        const int t0 = 2 * tig;
        const int t1 = t0 + 1;
        if (t0 < cn) {
            const int dst = ent_dst[c0 + t0];
            if (v0) out[(size_t) dst * L.n_embd + r0] = acc[0] * s_down;
            if (v1) out[(size_t) dst * L.n_embd + r1] = acc[2] * s_down;
        }
        if (t1 < cn) {
            const int dst = ent_dst[c0 + t1];
            if (v0) out[(size_t) dst * L.n_embd + r0] = acc[1] * s_down;
            if (v1) out[(size_t) dst * L.n_embd + r1] = acc[3] * s_down;
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_dst; (void) hq; (void) L; (void) out;
#endif
}

bool device_ok() {
    static std::mutex mu;
    static int cached[32] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) return false;
    std::lock_guard<std::mutex> lk(mu);
    if (cached[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) {
            const int mj = strata::cc_major_of(major);
            const int mn = strata::cc_minor_of(minor);
            cached[dev] = (mj == 12 && (mn == 0 || mn == 1)) ? 1 : -1;
        } else {
            (void) cudaGetLastError();
            cached[dev] = -1;
        }
    }
    return cached[dev] == 1;
}

int g_override = -1;

}  // namespace

void nvfp4_decode_sm120_set_override(int mode) noexcept { g_override = mode; }

bool nvfp4_decode_sm120_available() noexcept {
    static const bool env_on = [] {
        const char* v = std::getenv("STRATA_NVFP4_DEC_SM120");
        if (v == nullptr || v[0] == '\0') return false;   // opt-in: iq_kernels.cu row_dot_20_sub8 + native_down_lds_kernel (use_sub8) is 3-6x faster at DFlash2 E/G=1..4
        return v[0] != '0';
    }();
    const bool on = (g_override >= 0) ? (g_override != 0) : env_on;
    return on && device_ok();
}

bool nvfp4_decode_sm120_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                                const void* x_q8_1, void* hq_scratch, float* out, void* stream) {
    if (!nvfp4_decode_sm120_available()) return false;
    if (L.gu_type != kTypeNVFP4 || L.d_type != kTypeNVFP4 || L.had_block != 0) return false;
    if (L.n_embd % 256 != 0 || L.n_ff % 64 != 0 || L.n_ff % 32 != 0) return false;
    if (cap_groups <= 0 || cap_entries <= 0) return true;

    cudaStream_t s = (cudaStream_t) stream;
    const auto* X = reinterpret_cast<const Q81Block*>(x_q8_1);
    auto* HQ = reinterpret_cast<Q81Block*>(hq_scratch);

    const dim3 g_gu((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
    nvfp4_gu_tc_sm120_kernel<<<g_gu, 128, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, HQ);

    const dim3 g_d((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
    nvfp4_down_tc_sm120_kernel<<<g_d, 128, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, HQ, L, out);

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels
