// src/prefill/moe_fused_iq.cu - see include/strata/prefill/moe_fused_iq.hpp (#136: the native packs' prompt experts on
// the fused int8 kernels of moe_fused.cu).
//
// The arithmetic.  An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural
// order (per 64 values: 64 codes, then {d_0, d_1} and 8 unused bytes - 80 bytes, the Q2_0 path's size).  A weight
// block of 32 (or 16) values is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied
// (IQ2_XXS / IQ2_XS / IQ2_S: 0..43, IQ3_XXS: 0..62, IQ3_S: 0..15, IQ4_XS / IQ4_NL: -127..113), Q2_0's code - 1.  So
//     sum w x = d_w * d_x * sum q a,     |sum q a| <= 32 * 127 * 127 < 2^22,
// and the int32 dot becomes a float exactly with one float add: the mma starts from C = 0x4B400000 (the bits of
// 1.5 * 2^23), so as_float(D) - 1.5 * 2^23 is the dot.  Per output and 32 values: that add, d_w * d_x, one fma (the
// formats with a scale per 16 values: two dots, each times its d_w, then d_x).  d_w is llama.cpp's own: IQ2_XXS
// d (2s + 1) / 8, IQ2_XS / IQ2_S d (2s + 1) / 8 per 16, IQ3_XXS d (2s + 1) / 4, IQ3_S d (2s + 1), IQ4_XS d (s - 32),
// IQ4_NL and Q2_0 d (dequantize_row_* in ggml-quants.c, mmq-load-tiles.cuh).
//
// The load stage.  Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's
// weight rows: its block bytes are read from the blob with 16-bit loads (the GGUF blocks are 2-byte aligned: 66 / 74 /
// 82 / 98 / 110 / 136 / 18 bytes) one stage ahead into registers (and two 256-value blocks ahead into L2), then turned
// into 32 int8 and the sub-block's scales in shared memory, double-buffered.  The codebooks live in shared memory (up
// to 8 KB, IQ2_S).  The activations arrive by cp.async, four stages deep, as in moe_fused.cu; the fragments come by
// ldmatrix.  The products are mma.sync m16n8k32 (m16n8k16 for the formats with a scale per 16 values); the epilogues
// (SwiGLU, H to int8 per 32 features; down into the per-slot rows) are moe_fused.cu's.
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/pf_fused_override.hpp"
#include "strata/prefill/fused_tile_rule.hpp"
#include "strata/prefill/fused_tq_default.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/emulate.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"

#if defined(__HIPCC__)
#include "moe_f16_common.cuh"
#include "moe_f16_kq.hpp"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
// (STRATA_PF_FUSED_TQ=1: CUDA sm_80+, HIP gfx12) the trellis types' int8 codebook, the same table the decode kernels and the trellis MMQ tiles
// read (tq.cuh's tq_lane_to_q8): the fused kernels take MMQ's int8 weight values exactly
#include "../kernels/cuda/tq_lut_i8.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts (native): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww, int nt = THREADS) { return nt / 32 / ww * 16; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
// Turing: 2 (its copies are plain loads, so depth buys nothing, and 4 does not fit its 64 KB of shared memory)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr int ASTAGES_DEV = 2;
constexpr bool T75_DEV = true;    // the Q2_0 stage epilogue in native_kernel: fewer instructions, its sums in another order
#else
constexpr int ASTAGES_DEV = ASTAGES;
constexpr bool T75_DEV = false;
#endif
constexpr int GU_ROWS_K = kMoeEmbd, D_ROWS_K = kMoeFF;   // K of gate/up (n_embd) and of down (n_ff): the compiled geometry's (2560 and 640 for Flash-Next)
// the weight rows of a product: gate/up 2 n_ff (1280), down n_embd (2560)
constexpr int GU_ROWS_N = kMoeGuRows, D_ROWS_N = kMoeEmbd;
constexpr int H_ROW_BYTES = D_ROWS_K / 64 * AB;   // bytes of a row of H in the int8 activation layout (800 for Flash-Next)
static_assert(GU_ROWS_K % 256 == 0 && D_ROWS_K % 64 == 0, "the native expert kernels: K of gate/up is whole 256-value super-blocks, K of down whole 64-value stages");

// the formats (ggml type ids)
static_assert(GGML_TYPE_IQ2_XXS == strata::prefill::fused_tile::kTypeIq2Xxs && GGML_TYPE_IQ2_XS == strata::prefill::fused_tile::kTypeIq2Xs &&
              GGML_TYPE_IQ2_S == strata::prefill::fused_tile::kTypeIq2S && GGML_TYPE_IQ3_XXS == strata::prefill::fused_tile::kTypeIq3Xxs &&
              GGML_TYPE_IQ3_S == strata::prefill::fused_tile::kTypeIq3S && GGML_TYPE_IQ4_XS == strata::prefill::fused_tile::kTypeIq4Xs,
              "fused_tile_rule.hpp's default_gate_format uses ggml's type ids");
static_assert(GGML_TYPE_Q5_1 == strata::prefill::fused_tile::kTypeQ5_1 && GGML_TYPE_Q8_0 == strata::prefill::fused_tile::kTypeQ8_0 && GGML_TYPE_Q4_K == strata::prefill::fused_tile::kTypeQ4K &&
              GGML_TYPE_Q5_K == strata::prefill::fused_tile::kTypeQ5K && GGML_TYPE_Q6_K == strata::prefill::fused_tile::kTypeQ6K,
              "fused_tile_rule.hpp's wide_pair and kq_flash_kind use ggml's type ids");
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0,
              // Aurora (opt-in, HIP gfx11 only): Unsloth UD-Q4_K_XL's experts - gate/up Q4_K (one layer Q5_K), down Q5_1
              // (five layers Q8_0).  Their weights are w = d_w q + m_w (a minimum), so a sub-block's dot is
              //   sum w x = d_x (d_w sum q a + m_w sum a):
              // the activation blocks carry sum a per 32 values in their eight spare bytes (quant_act_nat_kernel and
              // the H epilogue write it for every path; no other format reads it)
              T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K, T_Q5_0 = GGML_TYPE_Q5_0, T_Q5_1 = GGML_TYPE_Q5_1, T_Q8_0 = GGML_TYPE_Q8_0,
              // CUDA opt-in (STRATA_PF_FUSED_Q6K=1): UD-Q6_K_XL's gate/up (47 layers; UD-Q5_K_XL's one Q6_K layer): w = d sc (q - 32), an
              // int8 scale per 16 values, no minimum
              T_Q6_K = GGML_TYPE_Q6_K,
              // CUDA opt-in (STRATA_PF_FUSED_TQ=1): the Gyro trellis types (agentionai's ids, cmake/strata_ggml_tq_overlay.cmake; the pinned
              // ggml.h has no enumerators for them): 128 weights a block, {fp16 d; qs}; Gyro-M TQ2_T, Gyro-S TQK6 gate/up + TQK7 down
              T_TQ2_T = 144, T_TQK6 = 145, T_TQK7 = 146;
// the trellis types: 128-value blocks (32 steps of 4 weights), two 32-value sub-blocks a 64-value stage
__host__ __device__ constexpr bool is_tq(int t) { return t == T_TQ2_T || t == T_TQK6 || t == T_TQK7; }
__host__ __device__ constexpr int tq_k(int t) { return t == T_TQK6 ? 6 : t == T_TQK7 ? 7 : 8; }
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18 && sizeof(block_q4_K) == 144 &&
              sizeof(block_q5_K) == 176 && sizeof(block_q5_0) == 22 && sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34 &&
              sizeof(block_q6_K) == 210,
              "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : t == T_Q4_K ? 144 : t == T_Q5_K ? 176 : t == T_Q5_0 ? 22 : t == T_Q5_1 ? 24 : t == T_Q8_0 ? 34
         : t == T_Q6_K ? 210 : t == T_TQ2_T ? 34 : t == T_TQK6 ? 26 : t == T_TQK7 ? 30 : 18;
}
// the formats with a minimum (w = d q + m), and the raw words a sub-block's load stage holds
__host__ __device__ constexpr bool has_min(int t) { return t == T_Q4_K || t == T_Q5_K || t == T_Q5_1; }
__host__ __device__ constexpr int raw_words(int t) {
    return t == T_Q4_K ? 10 : t == T_Q5_K ? 18 : t == T_Q5_0 ? 6 : t == T_Q5_1 ? 6 : t == T_Q8_0 ? 9 : t == T_Q6_K ? 18 : t == T_TQ2_T ? 4 : 5;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S || t == T_Q6_K; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : is_tq(t) ? 2048 * 2 : 0;   // (the trellis types: the int8 codebook tq_lut_i8, 2,048 points of two int8)
}
// ---- weight staging (CUDA sm_80+): the 512-thread 128-row instances of the formats below copy each weight row's slot into shared memory with 16-byte
// cp.async (a ring of two slots, issued two slots ahead) instead of every decode thread reading it from global memory in 2- and 4-byte loads; the decode
// then runs the same load_unit on the shared-memory address, so the bits are the same.  It is chosen per instance at compile time, only where it was measured
// faster (RTX A4000, one layer: +10..+16% for these pairs); every other instance keeps its code and its shared memory unchanged.
// A slot is the bytes of SPB stages (64 values of K each) of a row: a 256-value block for the gate/up i-quants, two stages of 32-value blocks (four blocks) for IQ4_NL / Q5_0 / Q8_0 on either side,
// and for the Q6_K gate/up half a super-block gathered from three windows (ql, qh, scales + d).  The GGUF blocks are only 2-byte aligned, so a slot is copied as the 16-byte chunks of the aligned
// window around it (SW_CHUNKS of them).
constexpr int kSwAst = 3;               // the staged instances' activation stages (the ring needs SPB >= AST - 1; 4 stages would not fit the ring beside it)
constexpr size_t kSmemOptIn = 101376;   // 99 KB: the opt-in shared memory a block gets on sm_86 / sm_89 / sm_120 / sm_121
__host__ __device__ constexpr bool sw_on(int t, bool gu) {
#if defined(__HIPCC__)
    (void) t; (void) gu;
    return false;
#else
    // (the K-quant downs: Q5_K / Q6_K rows of 256-value super-blocks, i.e. a K that is a multiple of 256 - Qwen3.6's n_ff 512; the Flash-Next geometry (640) has none; the K-quant gate/up Q4_K / Q5_K:
    // their K is the model width, 2,560 / 2,048 = 10 / 8 super-blocks)
    return gu ? (t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS || t == T_Q8_0 || t == T_Q6_K || t == T_Q4_K || t == T_Q5_K)
              : (t == T_IQ4_NL || t == T_Q8_0 || t == T_Q5_0 || ((t == T_Q5_K || t == T_Q6_K) && D_ROWS_K % 256 == 0));
#endif
}
__host__ __device__ constexpr bool sw_inst(int t, bool gu, int ww, int nt, bool f16) { return sw_on(t, gu) && ww == 2 && nt == THREADS && !f16; }
// the 32-value-block formats (two blocks a 64-value stage), gate/up (Q8_0) or down
__host__ __device__ constexpr bool sw_b32(int t) { return t == T_IQ4_NL || t == T_Q5_0 || t == T_Q8_0; }
// the gate/up formats whose block spans 256 values (the kernel's GU256: one block per four stages)
// (a K-quant down with K a multiple of 256 walks its super-blocks the same way: Q5_K / Q6_K)
__host__ __device__ constexpr bool sw_gu256(int t, bool gu) {
    return (gu || ((t == T_Q5_K || t == T_Q6_K) && D_ROWS_K % 256 == 0)) && t != T_Q2_0 && t != T_Q8_0 && !is_tq(t);
}
// stages a slot: a whole 256-value block (gate/up i-quants), else two (four 32-value blocks; the Q6_K gate/up and the Q5_K / Q6_K downs: half a super-block)
__host__ __device__ constexpr int sw_spb(int t, bool gu) { return sw_gu256(t, gu) && t != T_Q6_K && t != T_Q5_K ? 4 : 2; }
__host__ __device__ constexpr int sw_slot_bytes(int t, bool gu) { return sw_gu256(t, gu) ? block_bytes(t) : sw_spb(t, gu) * 2 * block_bytes(t); }   // (Q6_K: unused, its slot is three windows)
// 16-byte chunks of a row's slot: the aligned window is at most slot + 14 bytes; Q6_K: ql 64 B (5) + qh 32 B (3) + scales and d, at most 18 B (2);
// Q5_K (down only): qs 64 B (5) + qh 32 B (3) + d, dmin and the 12 scale bytes 16 B (2)
__host__ __device__ constexpr int sw_chunks(int t, bool gu) { return (t == T_Q6_K || t == T_Q5_K) ? 10 : (sw_slot_bytes(t, gu) + 14 + 15) / 16; }
constexpr int WLDH = 144;               // F16: bytes per decoded weight row of a stage: 64 halves + 16 (ldmatrix rows apart)
// shared memory of an instance.  `astages` is the device's depth (host_astages()); a staged instance takes kSwAst instead and adds its ring, sized for ITS side
// (gu) only: a failed shared-memory set-up makes dev_info() drop every fused kernel
__host__ __device__ constexpr size_t smem_bytes(int t, int ww, int astages, int nt, bool f16, bool gu) {
    const bool sw = sw_inst(t, gu, ww, nt, f16) && astages == ASTAGES;
    return (size_t) 2 * weight_rows(ww) * (f16 ? WLDH : WLD + 16) + (size_t) (sw ? kSwAst : astages) * tile_rows(ww, nt) * AB +
           (size_t) tile_rows(ww, nt) * 4 + grid_bytes(t) + (sw ? (size_t) 2 * weight_rows(ww) * 16 * sw_chunks(t, gu) : 0);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}

// (STRATA_PF_FUSED_TQ=1: CUDA sm_80+, HIP gfx12, wave32) the block-Hadamard rotation of a Hadamard-folded expert's input (include/strata/kernels/cpu/hadamard.hpp):
// y = H_128 (s * x) / sqrt(128) per block of 128 values, H the Sylvester-Walsh-Hadamard matrix.  One warp a block: lane l holds values
// 32 k + l (k = 0..3); the in-place fast transform's butterflies over strides 1-16 by shuffles, 32 and 64 between the registers.
__device__ __forceinline__ void fwht128_warp(float (&v)[4], int lane) {
#pragma unroll
    for (int st = 1; st < 32; st <<= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float o = __shfl_xor_sync(0xffffffffu, v[k], st);
            v[k] = (lane & st) ? o - v[k] : v[k] + o;
        }
    }
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = (a0 + a2) * 0.08838834764831845f; v[1] = (a1 + a3) * 0.08838834764831845f;   // (1 / sqrt(128))
    v[2] = (a0 - a2) * 0.08838834764831845f; v[3] = (a1 - a3) * 0.08838834764831845f;
}
// in place, rows x cols (cols a multiple of 128): every row's blocks rotated (signs: nullptr = identity)
__global__ void hadamard128_kernel(float* __restrict__ x, int64_t nblk, int cols, const float* __restrict__ signs) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    float* p = x + w * 128;
    const int c0 = (int) ((w * 128) % cols);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = p[32 * k + lane] * (signs ? signs[c0 + 32 * k + lane] : 1.0f);
    fwht128_warp(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) p[32 * k + lane] = v[k];
}
// the FP32 H rows the trellis gate/up wrote (a folded down; n_ff values at the start of each row's n_embd-float down-output row): rotated,
// then int8 per 32 values into the down product's activation layout (quant_act_nat_kernel's: 64 codes, the two scales, the two code sums
// per 80 bytes), for the rows of experts e0..e1 only (tb.off on the device)
__global__ void rot_quant_h_kernel(const float* __restrict__ hf, const int32_t* __restrict__ off, int e0, int e1, int64_t n,
                                   const float* __restrict__ signs, uint8_t* __restrict__ ha) {
    constexpr int BLK = D_ROWS_K / 128;   // Hadamard blocks a row (5 for n_ff 640)
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= n * BLK) return;
    const int64_t row = w / BLK;
    const int b = (int) (w % BLK);
    if (row < off[e0] || row >= off[e1]) return;
    const float* p = hf + row * D_ROWS_N + 128 * b;
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = p[32 * k + lane] * (signs ? signs[128 * b + 32 * k + lane] : 1.0f);
    fwht128_warp(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) {   // value 128 b + 32 k + lane: AB block (128 b + 32 k) / 64, half k & 1
        float am = fabsf(v[k]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const int code = am > 0.0f ? __float2int_rn(v[k] * (127.0f / am)) : 0;
        int cs = code;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) cs += __shfl_xor_sync(0xffffffffu, cs, o);
        uint8_t* o = ha + row * H_ROW_BYTES + ((128 * b + 32 * k) / 64) * AB;
        const int hh = k & 1;
        o[32 * hh + lane] = (uint8_t) (int8_t) code;
        if (lane == 0) { *(float*) (o + 64 + 4 * hh) = am / 127.0f; *(float*) (o + 72 + 4 * hh) = (float) cs; }
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// F16 (STRATA_PF_FUSED_F16): d += A (16 x 16 f16, row) * B (16 x 8 f16, col), FP32 accumulators in place
__device__ __forceinline__ void mma16h(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// four int8 (bytes 0-3) times a scale as two half2: lo = bytes 0, 1, hi = bytes 2, 3.  The flipped sign bit makes
// x + 128 the mantissa of 1024 (exact), minus 1152 is x, then one packed multiply.
__device__ __forceinline__ void i8x4_h2(uint32_t w, __half2 sc, uint32_t& lo, uint32_t& hi) {
    const uint32_t c = w ^ 0x80808080u;
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    const uint32_t l = __byte_perm(c, 0x64646464u, 0x4140), h = __byte_perm(c, 0x64646464u, 0x4342);
    const __half2 lv = __hmul2(__hsub2(*(const __half2*) &l, m), sc), hv = __hmul2(__hsub2(*(const __half2*) &h, m), sc);
    lo = *(const uint32_t*) &lv;
    hi = *(const uint32_t*) &hv;
}
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// Turing (STRATA_PF_FUSED_SM75): plain copies instead of cp.async (the same bytes; the barriers around them stay), and
// the s8 MMAs as m8n8k16 - m16n8k32 is four (rows g / g + 8 by k 0-15 / 16-31), m16n8k16 two (rows g / g + 8), each
// pair starting from MAGIC.  The int32 sums are exact in any order: the same d as the sm_80 instructions.
__device__ __forceinline__ void cp16(void* dst, const void* src) { *(uint4*) dst = __ldg((const uint4*) src); }
__device__ __forceinline__ void cp_commit() {}
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void mma8_s8(int& c0, int& c1, uint32_t a, uint32_t b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma8_s8_c(int& d0, int& d1, uint32_t a, uint32_t b, int c) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%4};\n"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c));
}
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    mma8_s8_c(d[0], d[1], a[0], b0, MAGIC);
    mma8_s8(d[0], d[1], a[2], b1);
    mma8_s8_c(d[2], d[3], a[1], b0, MAGIC);
    mma8_s8(d[2], d[3], a[3], b1);
}
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    mma8_s8_c(d[0], d[1], a0, b0, MAGIC);
    mma8_s8_c(d[2], d[3], a1, b0, MAGIC);
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// F16 (STRATA_PF_FUSED_F16) is sm_80+ only (dev_info stops before the NT = 256 setup on Turing): stubs so that its
// native_kernel instantiations compile here
__device__ __forceinline__ void mma16h(float (&)[4], const uint32_t (&)[4], uint32_t, uint32_t) { __trap(); }
__device__ __forceinline__ void i8x4_h2(uint32_t, __half2, uint32_t&, uint32_t&) { __trap(); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}
#endif
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750) || defined(__HIPCC__)
#if defined(__HIPCC__)
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
#endif
// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
#if defined(__HIPCC__)
// gfx11 (Aurora): the signs from a shared-memory table instead of the compare / subtract emulation.  An entry (one per
// sign byte b; for the 7-bit sign formats per 7 bits, the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the
// bytes whose sign bit is set (bits 0-3 for .x, 4-7 for .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign
// is set - the two's complement, byte-wise - and g elsewhere; no byte carries into the next, because no codebook entry
// has a zero byte (256 - g with g in 1..127; the grids' bytes are 1..62), so it is the bytes signed8 gives.
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
#endif
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// W (X5): direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half;
// exhaustively bit-exact over all 2^32 inputs (X5). Disable with -DSTRATA_W_NO_T16.
#if defined(__HIPCC__) && !defined(STRATA_W_NO_T16)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}
#else
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}
#endif

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
// 32 bits at a 2-byte aligned address (one load when it is 4-byte aligned)
__device__ __forceinline__ uint32_t ld32a(const uint8_t* p) {
    return ((uintptr_t) p & 3) == 0 ? *(const uint32_t*) p : ld32(p);
}
template <int T, int NW> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else if constexpr (T == T_Q4_K || T == T_Q5_K) {
        // qs: 32 bytes of the sub-block pair (ib / 2), the nibble chosen in convert; w[8] = d | dmin << 16; w[9]: the
        // scale/min bytes (get_scale_min_k4's: ib < 4: sc[ib], sc[ib + 4]; else sc[ib + 4], sc[ib - 4], sc[ib]), ib << 24
        constexpr int QS = T == T_Q4_K ? 16 : 48;
        const uint8_t* q = bp + QS + 32 * (ib >> 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32a(q + 4 * k);
        w[8] = ld32a(bp);
        const uint8_t* sc = bp + 4;
        w[9] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                      : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
        w[9] |= (uint32_t) ib << 24;
        if constexpr (T == T_Q5_K) {
#pragma unroll
            for (int k = 0; k < 8; ++k) w[10 + k] = ld32a(bp + 16 + 4 * k);   // qh
        }
    } else if constexpr (T == T_Q5_0) {   // d, qh, qs[16] (22 bytes: 2-byte aligned only)
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 6 + 4 * k);
        w[4] = ld32(bp + 2);
        w[5] = ld16(bp);
    } else if constexpr (T == T_Q5_1) {   // d, m, qh, qs[16]
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32a(bp + 8 + 4 * k);
        w[4] = ld32a(bp + 4);
        w[5] = ld32a(bp);
    } else if constexpr (T == T_Q8_0) {   // d, qs[32]
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[8] = ld16(bp);
    } else if constexpr (T == T_Q6_K) {
        // ql[128], qh[64], scales[16] (int8), d (210 bytes: 2-byte aligned only).  Sub-block ib = values 32 ib.. of the 256: half
        // n = ib / 4, quarter r = ib % 4 (ggml's dequantize_row_q6_K): low nibbles from ql[64 n + 32 (r & 1) ..] (bits 4 (r >> 1)..),
        // two high bits from qh[32 n ..] (bits 2 r..), scales[8 n + 2 r] for values 0-15 and [.. + 1] for 16-31.  w[0..7] ql, w[8..15]
        // qh, w[16] = d | the two scales << 16, w[17] = r
        const int n = ib >> 2, r = ib & 3;
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 64 * n + 32 * (r & 1) + 4 * k);
#pragma unroll
        for (int k = 0; k < 8; ++k) w[8 + k] = ld32(bp + 128 + 32 * n + 4 * k);
        w[16] = ld16(bp + 208) | (ld16(bp + 192 + 8 * n + 2 * r) << 16);
        w[17] = (uint32_t) r;
    } else if constexpr (T == T_TQ2_T) {
        // (STRATA_PF_FUSED_TQ=1) sub-block ib = steps 8 ib .. 8 ib + 7 of the block {fp16 d; qs[32]}: step t's state is
        // qs[(t + 31) % 32] << 8 | qs[t] (tq.cuh tq_states4), so the 8 bytes qs[8 ib ..] and the byte before them (circular)
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = bp[2 + ((8 * ib + 31) & 31)]; w[3] = ld16(bp);
    } else if constexpr (T == T_TQK6 || T == T_TQK7) {
        // (STRATA_PF_FUSED_TQ=1) {fp16 d; qs[4 K]}, qs a circular stream of 32 K bits (2 K u16 words, LSB first); step t's state is the 16
        // bits from stream bit (31 - t) K (tq.cuh tq_states4).  Steps 8 ib .. 8 ib + 7 lie in the 8 K + 15 bits from bit (24 - 8 ib) K: six u16
        // words from word (24 - 8 ib) K / 16, wrapped modulo 2 K, into w[0..2]; w[3] = d, w[4] = the start bit inside the first word
        constexpr int K = tq_k(T), NWD = 2 * K;   // (the u16 words of qs; NWD: NW is the template parameter)
        const int s0 = (24 - 8 * ib) * K, wb = s0 >> 4;
        uint32_t h[6];
#pragma unroll
        for (int k = 0; k < 6; ++k) h[k] = ld16(bp + 2 + 2 * ((wb + k) % NWD));
        w[0] = h[0] | (h[1] << 16); w[1] = h[2] | (h[3] << 16); w[2] = h[4] | (h[5] << 16);
        w[3] = ld16(bp);
        w[4] = (uint32_t) (s0 & 15);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// load_unit's bytes in two passes (native_kernel's opt-in NT = 256 instances only, CUDA): `issue` loads the 16-bit and
// 8-bit pieces into registers and `take` assembles the words from them a stage later, so no instruction waits on the
// loads in the stage that issues them (in one pass the shifts and ORs right after the loads stalled every stage on a
// memory round trip).  The same bytes in the same words as load_unit.
constexpr int NP = 12;   // pieces: at most IQ4_XS's 10 halves and a byte
struct LdIssue {
    uint32_t* pc; int k;
    __device__ __forceinline__ uint32_t h(const uint8_t* p) { pc[k++] = *(const uint16_t*) p; return 0; }
    __device__ __forceinline__ uint32_t b(const uint8_t* p) { pc[k++] = *p; return 0; }
};
struct LdTake {
    const uint32_t* pc; int k;
    __device__ __forceinline__ uint32_t h(const uint8_t*) { return pc[k++]; }
    __device__ __forceinline__ uint32_t b(const uint8_t*) { return pc[k++]; }
};
template <class L> __device__ __forceinline__ uint32_t lw(L& l, const uint8_t* p) {
    const uint32_t lo = l.h(p);
    const uint32_t hi = l.h(p + 2);
    return lo | (hi << 16);
}
template <int T, class L, int NW> __device__ __forceinline__ void load_unit_l(L& l, const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS || T == T_Q2_0) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = l.h(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib);
        const uint32_t d = l.h(bp); const uint32_t s = l.b(bp + 66 + ib);
        w[2] = d | (s << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = lw(l, bp + 2 + 4 * ib); w[1] = lw(l, bp + 34 + 4 * ib);
        const uint32_t d = l.h(bp); const uint32_t qh = l.b(bp + 66 + ib); const uint32_t sc = l.b(bp + 74 + ib);
        w[2] = d | (qh << 16) | (sc << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = lw(l, bp + 66 + 4 * ib); w[3] = l.h(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = lw(l, bp + 74 + 4 * ib);
        const uint32_t d = l.h(bp); const uint32_t qh = l.b(bp + 66 + ib); const uint32_t sc = l.b(bp + 106 + ib / 2);
        w[3] = d | (qh << 16) | (((sc >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = lw(l, bp + 8 + 16 * ib + 4 * k);
        const uint32_t lo = l.b(bp + 4 + ib / 2); const uint32_t hi = l.h(bp + 2); const uint32_t d = l.h(bp);
        const uint32_t ls = ((lo >> (4 * (ib & 1))) & 15) | (((hi >> (2 * ib)) & 3) << 4);
        w[4] = d | (ls << 16);
    } else {   // IQ4_NL
        static_assert(T == T_IQ4_NL, "the CUDA native formats");
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = lw(l, bp + 2 + 4 * k);
        w[4] = l.h(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
__device__ __forceinline__ void convert(const uint32_t (&w)[NW], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1, const uint4* sgn = nullptr) {
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        const int ib = (int) (w[9] >> 24), sh = 4 * (ib & 1);
        uint32_t sc, mn;
        if (ib < 4) { sc = w[9] & 63; mn = (w[9] >> 8) & 63; }
        else {
            sc = ((w[9] & 0xF) | (((w[9] >> 8) & 0xC0) >> 2));   // sc[ib + 4] & 15 | (sc[ib - 4] >> 6) << 4
            mn = (((w[9]) >> 4) & 0xF) | (((w[9] >> 16) & 0xC0) >> 2);   // sc[ib + 4] >> 4 | (sc[ib] >> 6) << 4
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q[k] = (w[k] >> sh) & 0x0F0F0F0Fu;
            if constexpr (T == T_Q5_K) q[k] |= ((w[10 + k] >> ib) & 0x01010101u) << 4;
        }
        s0 = half_at(w[8]) * (float) sc;
        s1 = -(half_at(w[8] >> 16) * (float) mn);
    } else if constexpr (T == T_Q5_0) {   // w = d (q - 16): the signed codes (like Q8_0's), no minimum
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            const uint32_t a = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            const uint32_t b = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
            q[k] = ((a | 0x80808080u) - 0x10101010u) ^ 0x80808080u;       // a byte a in 0..31 -> the int8 a - 16
            q[4 + k] = ((b | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        s0 = s1 = half_at(w[5]);
    } else if constexpr (T == T_Q5_1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            q[k] = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            q[4 + k] = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
        }
        s0 = half_at(w[5]);
        s1 = half_at(w[5] >> 16);
    } else if constexpr (T == T_Q8_0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = w[k];
        s0 = s1 = half_at(w[8]);
    } else if constexpr (T == T_Q6_K) {   // w = d sc (q - 32): the signed codes, an int8 scale per 16 values, no minimum
        const int r = (int) w[17], sl = 4 * (r >> 1), sh = 2 * r;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t a = ((w[k] >> sl) & 0x0F0F0F0Fu) | (((w[8 + k] >> sh) & 0x03030303u) << 4);   // 0..63 a byte
            q[k] = ((a | 0x80808080u) - 0x20202020u) ^ 0x80808080u;                                   // the int8 a - 32
        }
        const float d = half_at(w[16]);
        s0 = d * (float) (int8_t) (w[16] >> 16);
        s1 = d * (float) (int8_t) (w[16] >> 24);
    } else if constexpr (is_tq(T)) {
        // (STRATA_PF_FUSED_TQ=1) the 8 steps' 15-bit states, each x = s * 0x9e3779b1 -> codebook points x >> 21 and (x >> 10) & 2047, two
        // int8 each (tq_lut_i8 in shared memory): 4 weights a step, in weight order, as tq.cuh's tq_lane_to_q8 packs them for MMQ; one scale
        // d * TQ_LUT_I8_SCALE for the block
        const uint16_t* lut = (const uint16_t*) grid;
        uint32_t st[8];
        if constexpr (T == T_TQ2_T) {
            uint32_t prev = w[2];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t b = (w[j >> 2] >> (8 * (j & 3))) & 0xFFu;
                st[j] = (prev << 8) | b;
                prev = b;
            }
        } else {
            constexpr int K = tq_k(T);
            const uint64_t lo = (uint64_t) w[0] | ((uint64_t) w[1] << 32), hi = (uint64_t) w[1] | ((uint64_t) w[2] << 32);
            const int r = (int) w[4];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int rel = r + (7 - j) * K;   // <= 15 + 7 K: inside the 96 bits
                st[j] = (uint32_t) ((rel <= 48 ? lo >> rel : hi >> (rel - 32)) & 0xFFFFu);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t x = (st[j] & 0x7FFFu) * 0x9e3779b1u;   // (TQ_STATE_MASK: 15 state bits)
            q[j] = (uint32_t) lut[x >> 21] | ((uint32_t) lut[(x >> 10) & 2047u] << 16);
        }
        s0 = s1 = half_at(w[3]) * TQ_LUT_I8_SCALE;
    } else if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else if constexpr (is_tq(T)) return tq_lut_i8;   // (STRATA_PF_FUSED_TQ=1)
    else return nullptr;
}
#endif
#if !defined(STRATA_FUSED_DECODE_ONLY)   // (tests/cuda/prefill_decode_golden.cu: the decode above only)
// One work item = (TR routed rows of expert e - one or two of group()'s 64-row tiles -, a block of WR weight rows);
// a persistent grid walks the batch's items.  GU: gate/up rows of WR / 2 features (format WT) against the rows'
// tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm`.  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and
// up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
// NT (STRATA_PF_FUSED_NT=256, sm_80+ opt-in): 256 threads, two blocks a multiprocessor - one block's products run while
// the other waits at a barrier.  WW 2 only (its 2 WR = 256 decode units are the block's threads); a work item is 128
// weight rows x one 64-row tile.  Every output is the same thread's sum in the same order: the same bits.  The NT = 256
// instances take a WideBatch (STRATA_PF_BATCH up to kBatchRoom experts a launch), the default ones Batch.
//
// F16 (STRATA_PF_FUSED_F16=1, NT 256 only; changes the output at rounding level): the decoded weights times their
// scales as F16 and the activations times theirs as F16, m16n8k16 F16 products into FP32 accumulators - no FP32 work
// per 32 values.  The k order inside a 16-value group is permuted the same way for both operands (stored half2 word t
// holds values 4t, 4t + 1 and word 4 + t values 4t + 2, 4t + 3), so the activations' int8 ldmatrix fragments become
// the F16 B fragments in registers; the accumulator layout, and so the epilogues, are the int8 kernel's.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// (CUDA, STRATA_PF_FUSED_KQ=1, A1b) a register-lean load of the K-quant gate/up sub-blocks.  The row's two decode threads (uj 0
// and 1, adjacent lanes) read the same 32 qs bytes (Q5_K also the same 32 qh bytes) and each keeps one nibble (one bit) of every
// byte: here each loads only its half (16 bytes: bytes 16 uj ..) and the two halves are swapped by a shuffle when the stage is
// decoded (kq_lean_expand), so a thread holds 6 (Q4_K) / 10 (Q5_K) raw words across a stage instead of load_unit's 10 / 18.
// The words kq_lean_expand gives are load_unit's: convert<WT> and so the arithmetic are unchanged.
template <int T> __host__ __device__ constexpr int kq_lean_words() { return T == T_Q4_K ? 6 : 10; }
template <int T> __device__ __forceinline__ void kq_lean_load(const uint8_t* bp, int ib, int uj, uint32_t (&w)[kq_lean_words<T>()]) {
    constexpr int QS = T == T_Q4_K ? 16 : 48;
    const uint8_t* q = bp + QS + 32 * (ib >> 1) + 16 * uj;
#pragma unroll
    for (int k = 0; k < 4; ++k) w[k] = ld32a(q + 4 * k);
    w[4] = ld32a(bp);
    const uint8_t* sc = bp + 4;
    w[5] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                  : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
    w[5] |= (uint32_t) ib << 24;
    if constexpr (T == T_Q5_K) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[6 + k] = ld32a(bp + 16 + 16 * uj + 4 * k);   // qh
    }
}
// load_unit's words from the lean ones and the partner thread's (lane ^ 1: every lane of the warp calls this)
template <int T> __device__ __forceinline__ void kq_lean_expand(const uint32_t (&w)[kq_lean_words<T>()], int uj, uint32_t (&f)[raw_words(T)]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t o = __shfl_xor_sync(0xffffffffu, w[k], 1);
        f[k] = uj ? o : w[k];
        f[4 + k] = uj ? w[k] : o;
    }
    f[8] = w[4];
    f[9] = w[5];
    if constexpr (T == T_Q5_K) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t o = __shfl_xor_sync(0xffffffffu, w[6 + k], 1);
            f[10 + k] = uj ? o : w[6 + k];
            f[14 + k] = uj ? w[6 + k] : o;
        }
    }
}
// (CUDA, STRATA_PF_FUSED_Q6K=1) the same for the Q6_K gate/up: the row's two decode threads take sub-blocks 2 k and 2 k + 1 of a stage,
// whose low nibbles are different ql bytes (each loads its own 32) but whose two high bits come from the same 32 qh bytes: each loads
// half of them (bytes 16 uj ..) and the halves are swapped when the stage is decoded.  14 raw words across a stage instead of 18; the
// words kq_lean_expand gives are load_unit<T_Q6_K>'s, so convert and the arithmetic are unchanged.  (Explicit specializations: the
// Q4_K / Q5_K forms above are not touched.)
template <> __host__ __device__ constexpr int kq_lean_words<T_Q6_K>() { return 14; }
template <> __device__ __forceinline__ void kq_lean_load<T_Q6_K>(const uint8_t* bp, int ib, int uj, uint32_t (&w)[14]) {
    const int n = ib >> 2, r = ib & 3;
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 64 * n + 32 * (r & 1) + 4 * k);
#pragma unroll
    for (int k = 0; k < 4; ++k) w[8 + k] = ld32(bp + 128 + 32 * n + 16 * uj + 4 * k);
    w[12] = ld16(bp + 208) | (ld16(bp + 192 + 8 * n + 2 * r) << 16);
    w[13] = (uint32_t) r;
}
template <> __device__ __forceinline__ void kq_lean_expand<T_Q6_K>(const uint32_t (&w)[14], int uj, uint32_t (&f)[raw_words(T_Q6_K)]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = w[k];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t o = __shfl_xor_sync(0xffffffffu, w[8 + k], 1);
        f[8 + k] = uj ? o : w[8 + k];
        f[12 + k] = uj ? w[8 + k] : o;
    }
    f[16] = w[12];
    f[17] = w[13];
}
#endif

template <int NT> using BatchOf = std::conditional_t<NT == THREADS, Batch, WideBatch>;
template <int WT, bool GU, int WW, int NT = THREADS, bool F16 = false>
__global__ void __launch_bounds__(NT, NT == THREADS ? 1 : 2)
native_kernel(const BatchOf<NT> b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr int TR = tile_rows(WW, NT), WR = weight_rows(WW);
    static_assert(2 * WR <= NT, "every decode unit is a thread of the block");
    constexpr int WLDX = F16 ? WLDH : WLD;
    constexpr int WT_BYTES = WR * WLDX, WS_FLOATS = F16 ? 0 : WR * 4, ACT_STAGE = TR * AB;
    constexpr bool TWO_PASS = NT != THREADS;   // the two-pass loads (load_unit_l) only in the opt-in NT = 256 kernels
    static_assert(!(TWO_PASS && WT == T_Q5_0), "the NT = 256 kernels have no Q5_0 form");
    static_assert(!(TWO_PASS && (has_min(WT) || WT == T_Q8_0 || WT == T_Q6_K || is_tq(WT))), "the NT = 256 kernels have no K-quant / Q5_1 / Q8_0 / Q6_K / trellis form");
    constexpr bool SWT = sw_inst(WT, GU, WW, NT, F16) && ASTAGES_DEV == ASTAGES;   // weight staging (see sw_on)
    // (A1b) the K-quant gate/up's lean loads (kq_lean_load); a staged K-quant DOWN (Q5_K / Q6_K, K = 512) gathers the same lean words from its slot's windows (sw_q5 / sw_q6)
    constexpr bool KLEAN = (GU && (WT == T_Q4_K || WT == T_Q5_K || WT == T_Q6_K)) || (SWT && !GU && (WT == T_Q5_K || WT == T_Q6_K));
    constexpr int RAW = KLEAN ? kq_lean_words<WT>() : raw_words(WT) > 5 ? raw_words(WT) : 5;   // the raw words of a sub-block: 5, Q5_0's 6
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? GU_ROWS_N : D_ROWS_N) / WR;          // weight-row blocks per tile
    static_assert(GU_ROWS_N % WR == 0 && D_ROWS_N % WR == 0, "the weight rows of a product are whole work items (2 n_ff and n_embd are multiples of 64 WW)");
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    // weight staging (see sw_on): this instance's slots, its activation depth, and a ring at the end of shared memory
    constexpr int AST = SWT ? kSwAst : ASTAGES_DEV;
    constexpr int SW_SPB = sw_spb(WT, GU), SW_SB = sw_slot_bytes(WT, GU), SW_NCH = sw_chunks(WT, GU), SW_ROW = 16 * SW_NCH;
    static_assert(!SWT || (NS % SW_SPB == 0 && SW_SPB >= AST - 1), "staging: whole slots a row, and deep enough for the ring (a slot is read until two stages after the next copy is issued)");
    static_assert(!SWT || smem_bytes(WT, WW, ASTAGES, NT, F16, GU) <= kSmemOptIn, "staging: the ring does not fit the shared memory a block can get");
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [AST][TR][AB] activations
    int* srow = (int*) (stages + AST * ACT_STAGE);                    // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook
    uint8_t* sraw = sgrid + grid_bytes(WT);                           // staged: [2][WR][SW_ROW] the slots' raw bytes (16-byte aligned: every size before it is a multiple of 16)

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NT) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLDX + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        // gate/up in Q2_0 (64-value blocks): a block per stage, as the down rows read theirs; in Q8_0 (32-value blocks, CUDA
        // STRATA_PF_FUSED_Q8GU=1): two blocks a stage, as the Q8_0 down rows read theirs
        constexpr bool GU256 = (GU || ((WT == T_Q4_K || WT == T_Q5_K || WT == T_Q6_K) && D_ROWS_K % 256 == 0)) && WT != T_Q2_0 && WT != T_Q8_0 && !is_tq(WT);   // (a K-quant DOWN walks its 256-value super-blocks too, when n_ff is a multiple of 256: Qwen3.6's 512)
        constexpr bool B128 = is_tq(WT);   // (CUDA, STRATA_PF_FUSED_TQ=1) 128-value blocks, gate/up and down: a block per two stages, sub-blocks 0-3
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (B128) return wrow + (s >> 1) * BS;
            if (GU256) return wrow + (s >> 2) * BS;
            return (WT == T_IQ4_NL || WT == T_Q5_0 || WT == T_Q5_1 || WT == T_Q8_0) ? wrow + (2 * s + uj) * BS : wrow + s * BS;   // 32-value blocks: two a stage
        };
        auto sub = [&](int s) { return B128 ? 2 * (s & 1) + uj : GU256 ? 2 * (s & 3) + uj : uj; };
        // (staging) slot j of every weight row of the item into ring slot j & 1: each row's slot is the 16-byte chunks of the aligned window around its bytes (a
        // chunk that starts past the slot's end is not copied; the others stay inside the aligned 16 bytes of a valid byte)
        auto sw_copy = [&](int j) {
            if constexpr (SWT) {
                if (j >= NS / SW_SPB) return;
                uint8_t* d = sraw + (j & 1) * (WR * SW_ROW);
                for (int c = tid; c < WR * SW_NCH; c += NT) {
                    const int rr = c / SW_NCH, k = c - SW_NCH * rr;       // weight row rr of the item (the numbering of ur above)
                    const uint8_t* rp = GU ? blob + ((rr & 8) ? geo.up_off : 0) + (size_t) (fb * (WR / 2) + 8 * (rr >> 4) + (rr & 7)) * geo.gu_row
                                           : blob + geo.down_off + (size_t) (rbase + rr) * geo.d_row;
                    if constexpr (WT == T_Q6_K) {   // slot j = half n of super-block j / 2: three windows, chunks 0-4 ql (64 B), 5-7 qh (32 B), 8-9 scales + d (the bytes up to the end of d)
                        const uint8_t* bb = rp + (size_t) (j >> 1) * BS;
                        const int n = j & 1;
                        const uint8_t *a, *e;
                        int dk;
                        if (k < 5) { a = bb + 64 * n; e = a + 64; dk = k; }
                        else if (k < 8) { a = bb + 128 + 32 * n; e = a + 32; dk = k - 5; }
                        else { a = bb + 192 + 8 * n; e = bb + 210; dk = k - 8; }
                        const uint8_t* cs = (const uint8_t*) ((uintptr_t) a & ~(uintptr_t) 15) + 16 * dk;
                        if (cs < e) cp16(d + rr * SW_ROW + 16 * k, cs);
                    } else if constexpr (WT == T_Q5_K) {   // (down) slot j = half n of super-block j / 2: chunks 0-4 qs of the two stages (64 B at 48 + 64 n), 5-7 qh (32 B at 16), 8-9 d, dmin, scales (16 B at 0)
                        const uint8_t* bb = rp + (size_t) (j >> 1) * BS;
                        const int n = j & 1;
                        const uint8_t *a, *e;
                        int dk;
                        if (k < 5) { a = bb + 48 + 64 * n; e = a + 64; dk = k; }
                        else if (k < 8) { a = bb + 16; e = a + 32; dk = k - 5; }
                        else { a = bb; e = a + 16; dk = k - 8; }
                        const uint8_t* cs = (const uint8_t*) ((uintptr_t) a & ~(uintptr_t) 15) + 16 * dk;
                        if (cs < e) cp16(d + rr * SW_ROW + 16 * k, cs);
                    } else {   // (the S1 form, unchanged)
                        const uint8_t* a = rp + (size_t) j * SW_SB;
                        const uint8_t* cs = (const uint8_t*) ((uintptr_t) a & ~(uintptr_t) 15) + 16 * k;
                        if (cs < a + SW_SB) cp16(d + rr * SW_ROW + 16 * k, cs);
                    }
                }
            }
        };
        // (staging) the shared-memory address that stands for unit(s): the same bytes at the same offset inside the window
        auto sw_unit = [&](int s) -> const uint8_t* {
            static_assert(!SWT || GU256 || sw_b32(WT), "staging: the gate/up i-quants' 256-value blocks and the 32-value-block formats (IQ4_NL, Q5_0, Q8_0) only");
            const int j = s / SW_SPB, si = s - SW_SPB * j;
            const uint8_t* p0 = sraw + (j & 1) * (WR * SW_ROW) + ur * SW_ROW + ((uintptr_t) (wrow + (size_t) j * SW_SB) & 15);
            if constexpr (GU256) return p0;
            else return p0 + (2 * si + uj) * BS;   // 32-value blocks (IQ4_NL / Q5_0 / Q8_0, gate/up or down): two a stage
        };
        // (staging, Q6_K gate/up) kq_lean_load<T_Q6_K>'s 14 words from the three gathered windows: the same bytes at the same offsets inside each window (ql: bb + 64 n + 32 (r & 1) + ..., qh:
        // bb + 128 + 32 n + 16 uj + ..., scales / d: bb + 192 + 8 n + 2 r and bb + 208), bb = the slot's super-block
        auto sw_q6 = [&](int s, uint32_t (&w)[RAW]) {
            if constexpr (SWT && WT == T_Q6_K) {
                const int j = s >> 1, n = j & 1, r = sub(s) & 3;
                const uint8_t* slot = sraw + (j & 1) * (WR * SW_ROW) + ur * SW_ROW;
                const uint8_t* bb = wrow + (size_t) (j >> 1) * BS;
                const uint8_t* pa = slot + ((uintptr_t) (bb + 64 * n) & 15);
                const uint8_t* pb = slot + 80 + ((uintptr_t) (bb + 128 + 32 * n) & 15);
                const uint8_t* pq = slot + 128 + ((uintptr_t) (bb + 192 + 8 * n) & 15);
#pragma unroll
                for (int k = 0; k < 8; ++k) w[k] = ld32(pa + 32 * (r & 1) + 4 * k);
#pragma unroll
                for (int k = 0; k < 4; ++k) w[8 + k] = ld32(pb + 16 * uj + 4 * k);
                w[12] = ld16(pq + 16 - 8 * n) | (ld16(pq + 2 * r) << 16);
                w[13] = (uint32_t) r;
            }
        };
        // (staging, Q5_K down) kq_lean_load<T_Q5_K>'s 10 words from the three windows: the same bytes at the same offsets inside each (qs: bb + 48 + 32 (s & 3) + 16 uj, in the slot's
        // window from bb + 48 + 64 n that is 32 (s & 1) + 16 uj; qh: bb + 16 + 16 uj; d / dmin / scales: bb, bb + 4), bb = the slot's super-block
        auto sw_q5 = [&](int s, uint32_t (&w)[RAW]) {
            if constexpr (SWT && WT == T_Q5_K) {
                const int j = s >> 1, n = j & 1, ib = sub(s);
                const uint8_t* slot = sraw + (j & 1) * (WR * SW_ROW) + ur * SW_ROW;
                const uint8_t* bb = wrow + (size_t) (j >> 1) * BS;
                const uint8_t* pq = slot + ((uintptr_t) (bb + 48 + 64 * n) & 15) + 32 * (s & 1) + 16 * uj;
                const uint8_t* ph = slot + 80 + ((uintptr_t) (bb + 16) & 15) + 16 * uj;
                const uint8_t* pd = slot + 128 + ((uintptr_t) bb & 15);
#pragma unroll
                for (int k = 0; k < 4; ++k) w[k] = ld32a(pq + 4 * k);
                w[4] = ld32a(pd);
                const uint8_t* sc = pd + 4;
                w[5] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                              : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
                w[5] |= (uint32_t) ib << 24;
#pragma unroll
                for (int k = 0; k < 4; ++k) w[6 + k] = ld32a(ph + 4 * k);
            }
        };
        auto load_sw = [&](int s, uint32_t (&w)[RAW]) {
            if constexpr (SWT) {
                if constexpr (WT == T_Q5_K) sw_q5(s, w);
                else if constexpr (WT == T_Q6_K) sw_q6(s, w);
                else if constexpr (WT == T_Q4_K && GU) kq_lean_load<WT>(sw_unit(s), sub(s), uj, w);   // (K-quant gate/up: the lean words from the slot, as the unstaged path reads them from the block)
                else load_unit<WT>(sw_unit(s), sub(s), w);
            }
        };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = 2;
        constexpr int SBB = WT == T_Q2_0 ? 4 * BS : WT == T_Q8_0 ? 8 * BS : is_tq(WT) ? 2 * BS : BS;   // bytes per 256 values of a row
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * SBB + (uj ? SBB - 1 : 0));
        };
        if constexpr (SWT) {   // staged: the weight bytes come by cp.async below, no L2 prefetch
        } else if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < geo.d_row + 127; o += 256) pf_l2(wrow + min(o, geo.d_row - 1));
        }
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();                                          // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % AST) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += NT) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[RAW], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            if constexpr (KLEAN) {   // load_unit's words back from the lean ones (a shuffle with the row's other thread)
                uint32_t f[raw_words(WT)];
                kq_lean_expand<WT>(raw, uj, f);
                convert<WT>(f, sgrid, kv, q, s0, s1);
            } else
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            if constexpr (F16) {   // two 16-value groups of 8 half2 words (the permuted order above)
                uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLDX + 64 * uj);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const __half2 sc = __float2half2_rn(k ? s1 : s0);
                    uint32_t lo[4], hi[4];
#pragma unroll
                    for (int w = 0; w < 4; ++w) i8x4_h2(q[4 * k + w], sc, lo[w], hi[w]);
                    d[2 * k] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
                    d[2 * k + 1] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                }
            } else {
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
            }
        };
#pragma unroll
        for (int s = 0; s < AST - 1; ++s) {
            if (s < NS) load_act(s);
            if constexpr (SWT) { if (s < 2) sw_copy(s); }   // slots 0 and 1 go with the first two activation groups
            cp_commit();
        }
        if constexpr (SWT) {   // slot 0 in; with two-stage slots slot 1 too (its first stage, 2, is read at the end of the first iteration, before any later wait)
            if constexpr (SW_SPB == 2) cp_wait<0>(); else cp_wait<AST - 2>();
            __syncthreads();
        }
        uint32_t raw[RAW] = {};
        uint32_t pc[NP];   // stage s + 2's pieces in flight (load_unit_l)
        // (the two-pass loads exist for the NT = 256 instances' formats only: no Q5_0 instance takes them)
        auto issue = [&](int s) { if constexpr (TWO_PASS) { LdIssue l{pc, 0}; uint32_t w[5]; load_unit_l<WT>(l, unit(s), sub(s), w); } };
        auto take = [&](int s) { if constexpr (TWO_PASS) { LdTake l{pc, 0}; load_unit_l<WT>(l, unit(s), sub(s), raw); } };
        if constexpr (SWT) { if (dec) load_sw(0, raw); } else
        if constexpr (KLEAN) { if (dec) kq_lean_load<WT>(unit(0), sub(0), uj, raw); } else
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        if (dec && TWO_PASS && 2 < NS) issue(2);
        put(raw, 0);
        if constexpr (SWT) { if (dec) load_sw(1, raw); } else
        if constexpr (KLEAN) { if (dec) kq_lean_load<WT>(unit(1), sub(1), uj, raw); } else
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<AST - 2>();
            __syncthreads();
            if (s + AST - 1 < NS) load_act(s + AST - 1);
            if constexpr (SWT) { if (s % SW_SPB == SW_SPB - 2) sw_copy(s / SW_SPB + 2); }   // the ring slot's last reader is behind this stage's barrier; the copy is waited for two slots on
            cp_commit();
            if (!SWT && GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % AST) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = F16 ? float2{} : *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
                // (the formats with a minimum, CUDA K-quant opt-in) the activation codes' sums per 32 values, bytes 72-79
                float2 sx[2][2];
                if constexpr (has_min(WT)) {
#pragma unroll
                    for (int n = 0; n < 2; ++n)
#pragma unroll
                        for (int cc = 0; cc < 2; ++cc) {   // (A1b) d_x times the sum, once a stage: the minimum term is then one FMA
                            const float2 sm = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 72);
                            sx[n][cc] = make_float2(dx[n][cc].x * sm.x, dx[n][cc].y * sm.y);
                        }
                }
                if constexpr (F16) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t bq[4];                           // {n0 k 0-15, n0 16-31, n1 0-15, n1 16-31}: token nb0 + 8 n + g
                        ldsm4(bq, sa + b_off + 32 * h);
                        uint32_t bh[2][2][2];                     // [n][16-value group][b0, b1]
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            const __half2 dxh = __float2half2_rn(*(const float*) (sa + (nb0 + 8 * n + g) * AB + 64 + 4 * h));
#pragma unroll
                            for (int k = 0; k < 2; ++k) i8x4_h2(bq[2 * n + k], dxh, bh[n][k][0], bh[n][k][1]);
                        }
#pragma unroll
                        for (int i = 0; i < 4; ++i)
#pragma unroll
                            for (int k = 0; k < 2; ++k) {
                                uint32_t a[4];
                                ldsm4(a, W + a_off + 16 * i * WLDX + 64 * h + 32 * k);
                                mma16h(acc[i][0], a, bh[0][k][0], bh[0][k][1]);
                                if (on1) mma16h(acc[i][1], a, bh[1][k][0], bh[1][k][1]);
                            }
                    }
                } else
                if constexpr (WT == T_Q2_0 && T75_DEV) {
                    // Turing, Q2_0: one d per 64 values, so the stage's two halves share their weight scale - each row's dot
                    // is d (dx_0 dot_0 + dx_1 dot_1): the activation scales per half, then d once
                    uint32_t bq0[4], bq1[4];                      // the two halves' {n0 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq0, sa + b_off);
                    ldsm4(bq1, sa + b_off + 32);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a0[4], a1[4];
                        ldsm4(a0, W + a_off + 16 * i * WLD);
                        ldsm4(a1, W + a_off + 16 * i * WLD + 32);
                        const float wa = S[(64 * wf + 16 * i + g) * 4], wb = S[(64 * wf + 16 * i + 8 + g) * 4];
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            int d0[4], d1[4];
                            mma32(d0, a0, bq0[2 * n], bq0[2 * n + 1]);
                            mma32(d1, a1, bq1[2 * n], bq1[2 * n + 1]);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float t = fmaf(dx[n][q & 1].y, dotf(d1[q]), dx[n][q & 1].x * dotf(d0[q]));
                                acc[i][n][q] = fmaf(q < 2 ? wa : wb, t, acc[i][n][q]);
                            }
                        }
                    }
                } else
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                    if constexpr (has_min(WT))   // w = s0 q + s1: + s1 d_x (sum of the codes), as native_w11_kernel
                                        acc[i][n][q] = fmaf(q < 2 ? wa1 : wb1, h ? sx[n][q & 1].y : sx[n][q & 1].x, acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower on the RTX 5070: a layer of 2048 / 3584 / 8192 tokens, IQ2_S
            // 5.2 / 9.5 / 17.5 ms -> 6.0 / 10.2 / 18.3.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) {
                    if constexpr (TWO_PASS) {
                        take(s + 2);
                        if (s + 3 < NS) issue(s + 3);
                    } else {
                        if constexpr (SWT) load_sw(s + 2, raw); else
                        if constexpr (KLEAN) kq_lean_load<WT>(unit(s + 2), sub(s + 2), uj, raw); else
                        load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                    }
                }
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if constexpr (is_tq(WT)) {
                        // (CUDA, STRATA_PF_FUSED_TQ=1, a Hadamard-folded down: set_down_rotation) H as FP32, n_ff values at the start of the row's
                        // own down-output row in `dm` (n_embd floats a row; the down product rewrites it only after rot_quant_h_kernel has read
                        // it): rotated and quantized there before the down product (the rotation needs the whole 128-feature block of the row)
                        if (dm != nullptr) {
                            if (r < nrows) {
#pragma unroll
                                for (int i = 0; i < 4; ++i) dm[(size_t) (row0 + r) * D_ROWS_N + (WR / 2) * fb + 32 * wf + 8 * i + g] = h[i];
                            }
                            continue;
                        }
                    }
                    if constexpr (has_min(WT)) {
                        // (K-quant gate/up, STRATA_PF_FUSED_KQ=1, CUDA) the down formats with a minimum (Q5_1) read the codes' sum
                        // per 32 values from the H block's spare bytes 72-79, as quant_act_nat_kernel writes them for the
                        // activations and HIP's native_w11_kernel for H: the same codes, summed over the warp's 32 features
                        int code[4], cs = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) { code[i] = __float2int_rn(h[i] * inv); cs += code[i]; }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) cs += __shfl_xor_sync(0xffffffffu, cs, o);
                        if (r < nrows) {
                            uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (blk >> 1) * AB;
                            const int hh = blk & 1;
#pragma unroll
                            for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) code[i];
                            if (g == 0) {
                                *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                                *(float*) (o + 72 + 4 * hh) = (float) cs;
                            }
                        }
                    } else
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * kMoeEmbd + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#if !defined(__HIPCC__)
// STRATA_PF_FUSED_WS=1 (sm_80+, opt-in; the F16 products of native_kernel's F16 path, so the same rounding as
// STRATA_PF_FUSED_F16): warp-specialized.  16 warps at the defaults: warps 8-15 (the producers, STRATA_WS_PROD threads)
// decode the weights to F16 and copy the activations into a ring of WS_SLOTS stages; warps 0-7 (the consumers, WS_CONS
// threads, native_kernel's NT = 256 tiling: 128 weight rows x 64 routed rows) run the products.  Named barriers per
// slot (full: producers arrive, consumers wait; empty: the reverse) replace the block-wide barrier per stage, and the
// ring runs on across work items, so the decode of the next stages and of the next item overlaps the products and the
// epilogue.
#ifndef STRATA_WS_PROD
#define STRATA_WS_PROD 256
#endif
#ifndef STRATA_WS_SLOTS
#define STRATA_WS_SLOTS 2
#endif
// STRATA_WS_NB: n8 tiles a consumer warp (2: 8 consumer warps of 16 routed rows; 4: 4 warps of 32 rows, each weight
// fragment read from shared memory once for twice the products)
#ifndef STRATA_WS_NB
#define STRATA_WS_NB 2
#endif
constexpr int WS_NB = STRATA_WS_NB;
constexpr int WS_CONS = 32 * 2 * (64 / (8 * WS_NB)), WS_PROD = STRATA_WS_PROD, WS_THREADS = WS_CONS + WS_PROD, WS_SLOTS = STRATA_WS_SLOTS;
// STRATA_WS_KS: 64-value stages per ring slot (one handshake per slot: fewer barriers per product)
#ifndef STRATA_WS_KS
#define STRATA_WS_KS 2
#endif
constexpr int WS_KS = STRATA_WS_KS;
// STRATA_WS_HTAB (default 0, off; 1 or 2 on): IQ3_XXS / IQ3_S producers decode through one table of (codebook entry,
// 4 sign bits) -> the four signed values as F16 (32 / 64 KB of shared memory): one 8-byte load and two packed multiplies
// a quad instead of the codebook load, the sign masks, the int8 -> F16 conversion.  Exact small integers either way: the
// same bits.
#ifndef STRATA_WS_HTAB
#define STRATA_WS_HTAB 0
#endif
__host__ __device__ constexpr bool ws_htab(int t) { return STRATA_WS_HTAB && (t == T_IQ3_XXS || t == T_IQ3_S); }
__host__ __device__ constexpr size_t ws_htab_bytes(int t) {
    return !ws_htab(t) ? 0 : STRATA_WS_HTAB == 2 ? (size_t) (t == T_IQ3_S ? 512 : 256) * 8 + 16 * 8
                                                : (size_t) (t == T_IQ3_S ? 512 : 256) * 16 * 8;
}
__host__ __device__ constexpr size_t ws_smem_bytes(int t) {
    return (size_t) WS_SLOTS * WS_KS * (128 * WLDH + 64 * AB) + 2 * 64 * 4 + grid_bytes(t) + ws_htab_bytes(t);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void nbar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
#endif
template <int WT, bool GU>
__global__ void __launch_bounds__(WS_THREADS, 1)
native_ws_kernel(const WideBatch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                 const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int WS_UPT = 256 / WS_PROD;   // decode units (sub-blocks) a producer thread takes per stage
    constexpr int TR = 64, WR = 128, WW = 2;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;
    constexpr int NFB = (GU ? GU_ROWS_N : D_ROWS_N) / WR;
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr int WT_BYTES = WR * WLDH, ACT_STAGE = TR * AB;
    constexpr bool GU256 = GU && WT != T_Q2_0;
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                         // [SLOTS][WR][WLDH] F16 weights
    uint8_t* stages = smem + WS_SLOTS * WS_KS * WT_BYTES;       // [SLOTS * KS][TR][AB] activations
    int* srow = (int*) (stages + WS_SLOTS * WS_KS * ACT_STAGE); // [2][TR] (producers only)
    static_assert(NS % WS_KS == 0, "whole slots");
    uint8_t* sgrid = (uint8_t*) (srow + 2 * TR);
    uint2* htab = (uint2*) (sgrid + grid_bytes(WT));            // ws_htab: [entry][4 sign bits] four F16
    const int tid = threadIdx.x;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += WS_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (ws_htab(WT) && STRATA_WS_HTAB == 2) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        constexpr int NE = WT == T_IQ3_S ? 512 : 256;
        for (int i = tid; i < NE + 16; i += WS_THREADS) {
            if (i < NE) {   // the codebook entry's four values as F16 (exact small integers)
                const uint32_t gw = gs[i];
                const __half2 lo = __floats2half2_rn((float) (gw & 255), (float) ((gw >> 8) & 255)),
                              hi = __floats2half2_rn((float) ((gw >> 16) & 255), (float) (gw >> 24));
                htab[i] = make_uint2(*(const uint32_t*) &lo, *(const uint32_t*) &hi);
            } else {        // four sign bits as the F16 sign-bit masks of the two half2
                const uint32_t sg = (uint32_t) (i - NE);
                htab[i] = make_uint2(((sg & 1) << 15) | ((sg & 2) << 30), ((sg & 4) << 13) | ((sg & 8) << 28));
            }
        }
    } else if constexpr (ws_htab(WT)) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < (int) (ws_htab_bytes(WT) / 8); i += WS_THREADS) {
            const uint32_t gw = gs[i >> 4], sg = (uint32_t) i & 15;
            __half h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v = (float) ((gw >> (8 * j)) & 255);
                h[j] = __float2half_rn((sg >> j) & 1 ? -v : v);
            }
            const __half2 lo = __halves2half2(h[0], h[1]), hi = __halves2half2(h[2], h[3]);
            htab[i] = make_uint2(*(const uint32_t*) &lo, *(const uint32_t*) &hi);
        }
    }
    __syncthreads();
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    const int nmine = (int) blockIdx.x < nwork ? (nwork - 1 - (int) blockIdx.x) / (int) gridDim.x + 1 : 0;
    const int qtotal = nmine * NS;
    auto item = [&](int k, int& e, int& row0, int& nrows, int& fb) {
        const int w = (int) blockIdx.x + k * (int) gridDim.x;
        const int2 tl = tb.tiles[t0 + w / NFB];
        fb = w % NFB; e = tl.x; row0 = tl.y;
        nrows = min(TR, tb.off[e + 1] - row0);
    };
    if (tid >= WS_CONS) {
        // ---- producers: weight row ur (both 32-value sub-blocks of a stage), and a share of the activation copies
        const int p = tid - WS_CONS, ur = WS_UPT == 2 ? p : p >> 1, uj0 = WS_UPT == 2 ? 0 : p & 1;
        uint32_t kv[4] = {0, 0, 0, 0};
        if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
            for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
        }
        int q = 0;
        for (int k = 0; k < nmine; ++k) {
            int e, row0, nrows, fb;
            item(k, e, row0, nrows, fb);
            const uint8_t* blob = b.blob[e - b.e0];
            const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                           (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                     : blob + geo.down_off + (size_t) (fb * WR + ur) * geo.d_row;
            auto unit = [&](int s, int uj) -> const uint8_t* {
                if (GU256) return wrow + (s >> 2) * BS;
                return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
            };
            auto sub = [&](int s, int uj) { return GU256 ? 2 * (s & 3) + uj : uj; };
            int* sr = srow + (k & 1) * TR;
            if (p < TR) sr[p] = p < nrows ? (GU ? src[row0 + p] : row0 + p) : -1;
            if (GU) {
                for (int sb = 0; sb < 2; ++sb) pf_l2(wrow + sb * BS + (uj0 ? BS - 1 : 0));
            } else {
                for (size_t o = 128 * (size_t) uj0; o < geo.d_row; o += 128 * (3 - WS_UPT)) pf_l2(wrow + o);
            }
            nbar_sync(1 + 2 * WS_SLOTS, WS_PROD);   // sr
            uint32_t raw[WS_UPT][5], pc[WS_UPT][NP];
            auto issue = [&](int s) {
#pragma unroll
                for (int u = 0; u < WS_UPT; ++u) { LdIssue l{pc[u], 0}; uint32_t w5[5]; load_unit_l<WT>(l, unit(s, uj0 + u), sub(s, uj0 + u), w5); }
            };
            auto take = [&](int s) {
#pragma unroll
                for (int u = 0; u < WS_UPT; ++u) { LdTake l{pc[u], 0}; load_unit_l<WT>(l, unit(s, uj0 + u), sub(s, uj0 + u), raw[u]); }
            };
#pragma unroll
            for (int u = 0; u < WS_UPT; ++u) load_unit<WT>(unit(0, uj0 + u), sub(0, uj0 + u), raw[u]);
            if (1 < NS) issue(1);
#pragma unroll WS_KS
            for (int s = 0; s < NS; ++s, ++q) {
                const int qs = q / WS_KS, slot = qs % WS_SLOTS, ks = q % WS_KS, buf = slot * WS_KS + ks;
                if (ks == 0 && qs >= WS_SLOTS) nbar_sync(1 + WS_SLOTS + slot, WS_THREADS);   // the consumers are done with the slot
                uint8_t* st = stages + buf * ACT_STAGE;
                for (int c = p; c < TR * 5; c += WS_PROD) {
                    const int r = c / 5, qq = c % 5;
                    if (r < nrows) cp16(st + r * AB + qq * 16, act + (size_t) sr[r] * ACT_LD + s * AB + qq * 16);
                }
                cp_commit();
                if (GU && (s & 3) == 0 && (s >> 2) + 2 < GU_ROWS_K / 256) pf_l2(wrow + ((s >> 2) + 2) * BS + (uj0 ? BS - 1 : 0));
#pragma unroll
                for (int u = 0; u < WS_UPT; ++u) {
                    if constexpr (ws_htab(WT)) {
                        const uint32_t* w = raw[u];
                        const float sf = WT == T_IQ3_XXS ? half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f
                                                         : half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
                        const __half2 sc = __float2half2_rn(sf);
                        uint32_t lo[8], hi[8];
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255, s8;
                            if constexpr (WT == T_IQ3_XXS) {
                                const uint32_t v = (w[2] >> (7 * l)) & 127;
                                s8 = v | ((__popc(v) & 1) << 7);
                            } else {
                                const uint32_t qh = (w[3] >> 16) & 255;
                                i0 |= (qh << (8 - 2 * l)) & 256;
                                i1 |= (qh << (7 - 2 * l)) & 256;
                                s8 = (w[2] >> (8 * l)) & 255;
                            }
                            uint2 e0, e1;
                            if constexpr (STRATA_WS_HTAB == 2) {
                                constexpr int NE = WT == T_IQ3_S ? 512 : 256;
                                const uint2 g0 = htab[i0], g1 = htab[i1], m0 = htab[NE + (s8 & 15)], m1 = htab[NE + (s8 >> 4)];
                                e0 = make_uint2(g0.x ^ m0.x, g0.y ^ m0.y);
                                e1 = make_uint2(g1.x ^ m1.x, g1.y ^ m1.y);
                            } else {
                                e0 = htab[i0 * 16 + (s8 & 15)];
                                e1 = htab[i1 * 16 + (s8 >> 4)];
                            }
                            const __half2 a0 = __hmul2(*(const __half2*) &e0.x, sc), b0 = __hmul2(*(const __half2*) &e0.y, sc),
                                          a1 = __hmul2(*(const __half2*) &e1.x, sc), b1 = __hmul2(*(const __half2*) &e1.y, sc);
                            lo[2 * l] = *(const uint32_t*) &a0; hi[2 * l] = *(const uint32_t*) &b0;
                            lo[2 * l + 1] = *(const uint32_t*) &a1; hi[2 * l + 1] = *(const uint32_t*) &b1;
                        }
                        uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLDH + 64 * (uj0 + u));
#pragma unroll
                        for (int k = 0; k < 2; ++k) {
                            d[2 * k] = make_uint4(lo[4 * k], lo[4 * k + 1], lo[4 * k + 2], lo[4 * k + 3]);
                            d[2 * k + 1] = make_uint4(hi[4 * k], hi[4 * k + 1], hi[4 * k + 2], hi[4 * k + 3]);
                        }
                        continue;
                    }
                    uint32_t qv[8];
                    float s0, s1;
                    convert<WT>(raw[u], sgrid, kv, qv, s0, s1);
                    uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLDH + 64 * (uj0 + u));
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const __half2 sc = __float2half2_rn(h ? s1 : s0);
                        uint32_t lo[4], hi[4];
#pragma unroll
                        for (int w = 0; w < 4; ++w) i8x4_h2(qv[4 * h + w], sc, lo[w], hi[w]);
                        d[2 * h] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
                        d[2 * h + 1] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                    }
                }
                if (s + 1 < NS) {
                    take(s + 1);
                    if (s + 2 < NS) issue(s + 2);
                }
                if (ks == WS_KS - 1) {
                    cp_wait<0>();
                    nbar_arrive(1 + slot, WS_THREADS);   // the slot is full
                }
            }
        }
        return;
    }
    // ---- consumers: WS_NB n8 tiles (8 WS_NB routed rows) by 64 weight rows a warp, F16 products
    constexpr int NB = WS_NB;
    const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 8 * NB * (warp / WW);
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLDH + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    int q = 0;
    for (int k = 0; k < nmine; ++k) {
        int e, row0, nrows, fb;
        item(k, e, row0, nrows, fb);
        const bool on0 = nb0 < nrows;
        float acc[4][NB][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < NB; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][n][c] = 0.0f;
        for (int s = 0; s < NS; ++s, ++q) {
            const int qs = q / WS_KS, slot = qs % WS_SLOTS, ks = q % WS_KS, buf = slot * WS_KS + ks;
            if (ks == 0) nbar_sync(1 + slot, WS_THREADS);   // the slot is full
            if (on0) {
                const uint8_t* W = wt + buf * WT_BYTES;
                const uint8_t* sa = stages + buf * ACT_STAGE;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bh[NB][2][2];
#pragma unroll
                    for (int pr = 0; pr < NB / 2; ++pr) {
                        uint32_t bq[4];
                        ldsm4(bq, sa + b_off + 16 * pr * AB + 32 * h);
#pragma unroll
                        for (int nn = 0; nn < 2; ++nn) {
                            const int n = 2 * pr + nn;
                            const __half2 dxh = __float2half2_rn(*(const float*) (sa + (nb0 + 8 * n + g) * AB + 64 + 4 * h));
#pragma unroll
                            for (int kk = 0; kk < 2; ++kk) i8x4_h2(bq[2 * nn + kk], dxh, bh[n][kk][0], bh[n][kk][1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            uint32_t a[4];
                            ldsm4(a, W + a_off + 16 * i * WLDH + 64 * h + 32 * kk);
#pragma unroll
                            for (int n = 0; n < NB; ++n)
                                if (n == 0 || nb0 + 8 * n < nrows) mma16h(acc[i][n], a, bh[n][kk][0], bh[n][kk][1]);
                        }
                }
            }
            if (ks == WS_KS - 1 && (qs + WS_SLOTS) * WS_KS < qtotal) nbar_arrive(1 + WS_SLOTS + slot, WS_THREADS);   // the slot is free
        }
        if (!on0) continue;
        if (GU) {
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < NB; ++n) {
                if (nb0 + 8 * n >= nrows) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
            const int rbase = fb * WR;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < NB; ++n) {
                    if (nb0 + 8 * n >= nrows) break;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + (c & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * kMoeEmbd + rbase + 64 * wf + 16 * i + 8 * (c >> 1) + g] = acc[i][n][c];
                    }
                }
        }
    }
#endif
}
#endif  // !__HIPCC__

#if defined(__HIPCC__)
// ---- Aurora (S23): the native packs' fused experts on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8
// (wave32).  The CUDA kernel's arithmetic: a 32-value sub-block decoded to int8 (load_unit / convert above) and its
// scales; per 32 values the int32 dot starts from MAGIC (the C operand), so as_float(d) - 1.5 * 2^23 is the dot; the
// formats with a scale per 16 values take each 16-value k-step's dot alone.  Fragments (gfx11): A lane l holds the 16
// k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of column l % 16, C/D lane l holds
// D[2i + l / 16][l % 16].  Both operands are in natural order: the decoded weights (LDS, double-buffered) and the
// activations (quant_act_nat_kernel; read by each lane from global/L2, a stage ahead).
// A work item = NW_ROWS weight rows x a 64-row tile; 8 waves: 4 along the weight rows (32 each) x 2 along the tile
// (32 each), each 2 x 2 WMMA tiles.  Gate/up: local row r is feature r / 2, its gate (r even) or up (r odd) row, so a
// lane pair l, l + 16 holds gate and up of one feature.
// gfx12 (RDNA4: gfx1200 / gfx1201, wave32) has the same instruction with another layout (checked on gfx1201, R9700):
// A lane l holds A[l % 16][8 (l / 16) + j], B lane l B[8 (l / 16) + j][l % 16] (j = 0..7: 8 int8, two VGPRs, no
// replication), C/D lane l holds D[8 (l / 16) + i][l % 16].  A WMMA still contracts 16 k; lane half hi supplies k
// 8 hi .. 8 hi + 7 of them, so a 16-value k-step is still one WMMA (the K16 scales stay per WMMA).  NW_DROW is the
// D row of a lane's element i; on gfx12 a lane's 8 rows are consecutive, so a feature's gate (even local row) and up
// (odd) sit in one lane (elements 2j, 2j + 1) and the SwiGLU needs no cross-lane exchange.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_NAT_W11 1
#else
#define STRATA_NAT_W11 0
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_NAT_W12 1
#else
#define STRATA_NAT_W12 0
#endif
#define STRATA_NAT_WMMA (STRATA_NAT_W11 || STRATA_NAT_W12)
#if STRATA_NAT_W12
#define NW_DROW(i, hi) (8 * (hi) + (i))
#else
#define NW_DROW(i, hi) (2 * (i) + (hi))
#endif
// The VGPR cap of the occupancy variant (amdgpu_waves_per_eu).  8 was measured against a clang before 22; clang 22
// (ROCm 7.10) computes wrong results with it (#1180: gfx1100, every run), so from clang 22 the cap is off - except for
// gfx1151 on clang 23, where the capped kernels give the same bits and take less time (measured with ROCm 7.14.1's
// clang 23, hip_prefill_fused_iq, one layer, 2048-8192 tokens: gate/up IQ2_XXS / IQ3_XXS / IQ3_S / IQ4_XS with down
// Q2_0 / IQ4_NL 8-15%; IQ2_S / Q2_0 2-4%, from its Q2_0 down: IQ2_S's LDS leaves room for 3 blocks per WGP, so the
// compiler keeps its 238 VGPRs).  Two formats stay uncapped there: IQ2_XS gate/up spills 104 VGPRs under the cap and
// takes 61-66% more time, and Q8_0 down (18 spilled) took -1.9% to +1.9% time.  The test is the clang major version:
// other clang 23 builds are not measured, and clang 24 and newer keep the clang 22 rule.  NW_LB is the variant's
// template argument, the same in the host and every device pass (the launch names that instance); whether its cap
// applies is decided per architecture and format in the device pass (nw_capped).  -DSTRATA_W_LB=N sets the cap on
// every architecture and format.
#if defined(STRATA_W_LB)
constexpr int NW_LB = STRATA_W_LB;
#else
constexpr int NW_LB = 8;
#endif
constexpr bool nw_capped(int wt) {
#if defined(STRATA_W_LB) || !defined(__clang_major__) || __clang_major__ < 22
    (void) wt;
    return true;
#elif defined(__gfx1151__) && (__clang_major__ == 23 || (__clang_major__ >= 24 && !defined(STRATA_W_NOCAP24)))
    return wt != T_IQ2_XS && wt != T_Q8_0;
#else
    (void) wt;
    return false;
#endif
}
typedef int nw_i4 __attribute__((ext_vector_type(4)));
typedef int nw_i2 __attribute__((ext_vector_type(2)));
typedef int nw_i8 __attribute__((ext_vector_type(8)));
constexpr int NW_ROWS = 128;
constexpr int NW_THREADS = 256;

// one A / B fragment: gfx11 16 int8 per lane (uint4), gfx12 8 (uint2)
#if STRATA_NAT_W12
typedef uint2 nw_frag;
typedef nw_i2 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint2 v) { return nw_ab{(int) v.x, (int) v.y}; }
#else
typedef uint4 nw_frag;
typedef nw_i4 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint4 v) { return nw_ab{(int) v.x, (int) v.y, (int) v.z, (int) v.w}; }
#endif
// Workgroup barrier with the LDS wait written out (idea and wording of libstrix's barrier(), asf0/libstrix, MIT): in WGP mode
// the compiler may drop the s_waitcnt before __syncthreads()'s s_barrier (llvm/llvm-project#220357), so a read after the
// barrier can overtake another wave's LDS store.  Release/acquire fences keep LDS accesses from moving across it.
__device__ __forceinline__ void nw_barrier() {
#if STRATA_NAT_W11
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_waitcnt(0x7 | (0x3f << 10));   // gfx11 simm16: expcnt [2:0], lgkmcnt [9:4], vmcnt [15:10]: lgkmcnt(0) only
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}
__device__ __forceinline__ nw_i8 nw_wmma(nw_ab a, nw_ab b, nw_i8 c) {
#if STRATA_NAT_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#elif STRATA_NAT_W12
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

// W (X1): ALIAS = the H tile lives on the weight buffers (one extra barrier); LB > 0 = amdgpu_waves_per_eu(LB) (VGPR cap)
// where nw_capped(WT) says the cap applies.
// Disable both with -DSTRATA_W_NO_OCC (the launch sites then use <WT, GU, false, 0> and grid factor wgp_blocks).
template <int WT, bool GU, bool ALIAS = false, int LB = 0>
__global__ void __launch_bounds__(NW_THREADS) __attribute__((amdgpu_waves_per_eu(LB > 0 && nw_capped(WT) ? LB : 1)))
native_w11_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                  const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_NAT_WMMA
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = (GU ? GU_ROWS_N : D_ROWS_N) / NW_ROWS;     // work items per tile
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    __shared__ __align__(16) uint8_t wtbuf[2 * NW_ROWS * WLD];
    uint8_t(*wt)[NW_ROWS][WLD] = reinterpret_cast<uint8_t(*)[NW_ROWS][WLD]>(wtbuf);
    __shared__ __align__(16) float ws[2][NW_ROWS][4];           // their scales per 16 values
    __shared__ __align__(16) uint8_t sgrid[GB];
    constexpr int SGN = (WT == T_IQ2_XXS || WT == T_IQ2_XS || WT == T_IQ2_S || WT == T_IQ3_XXS || WT == T_IQ3_S)
                            ? sign_entries(WT) : 1;
    __shared__ __align__(16) uint4 ssign[SGN];                   // the signs of a sign byte, as byte masks
    __shared__ int srow[kTileRows];
    __shared__ float hs_sep[(GU && !ALIAS) ? kTileRows : 1][65];
    static_assert(!ALIAS || (size_t) kTileRows * 65 * 4 <= (size_t) 2 * NW_ROWS * WLD, "H tile must fit the weight buffers");
    float(*hs)[65] = ALIAS ? reinterpret_cast<float(*)[65]>(wtbuf) : hs_sep;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NW_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += NW_THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int ur = tid >> 1, uj = tid & 1;                       // this thread's decode unit: row ur, sub-block uj
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * NW_ROWS;
        const uint8_t* wrow = GU ? blob + ((ur & 1) ? geo.up_off : 0) + (size_t) (fb * (NW_ROWS / 2) + (ur >> 1)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        constexpr bool B128 = is_tq(WT);   // (STRATA_PF_FUSED_TQ=1, gfx12) 128-value blocks, gate/up and down: a block per two stages, sub-blocks 0-3
        // (STRATA_PF_FUSED_Q8GU=1, gfx12) a Q8_0 gate/up has 32-value blocks, two a stage, like its down: not a 256-value superblock
        constexpr bool GU256 = GU && WT != T_Q2_0 && WT != T_Q8_0;
        auto unit = [&](int s) -> const uint8_t* {
            if constexpr (B128) return wrow + (s >> 1) * BS;
            if (GU256) return wrow + (s >> 2) * BS;   // (Q2_0 gate/up: 64-value blocks, one a stage, like its down)
            return (WT == T_IQ4_NL || WT == T_Q5_0 || WT == T_Q5_1 || WT == T_Q8_0) ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return B128 ? 2 * (s & 1) + uj : GU256 ? 2 * (s & 3) + uj : uj; };
        auto put = [&](const uint32_t (&raw)[raw_words(WT)], int buf) {
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1, ssign);
            uint4* d = (uint4*) &wt[buf][ur][32 * uj];
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) &ws[buf][ur][2 * uj] = make_float2(s0, s1);
        };
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        uint32_t raw[raw_words(WT)] = {};
        load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (NS > 1) load_unit<WT>(unit(1), sub(1), raw);
        __syncthreads();                                          // srow
        const bool on = 32 * wn < nrows;
        const uint8_t* brow[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        nw_frag bq[2][4];        // B per 16-value k-step: bq[nt][2 h + kk] = k 32 h + 16 kk .. (gfx12: this lane's 8 of them)
        float2 bx[2], bsum[2];   // the activation scales, and (the formats with a minimum) the codes' sums per 32 values
        auto fetch_b = [&](int s, nw_frag (&q)[2][4], float2 (&x)[2], float2 (&sm)[2]) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#if STRATA_NAT_W12
                const uint2* p = reinterpret_cast<const uint2*>(brow[nt] + s * AB + 8 * hi);   // k 16 j + 8 hi .. + 7
                q[nt][0] = p[0]; q[nt][1] = p[2]; q[nt][2] = p[4]; q[nt][3] = p[6];
#else
                const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + s * AB);
                q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
#endif
                x[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 64);
                if constexpr (has_min(WT)) sm[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 72);
                else sm[nt] = make_float2(0.0f, 0.0f);
            }
        };
        if (on) fetch_b(0, bq, bx, bsum);
        for (int s = 0; s < NS; ++s) {
#ifndef STRATA_Q_TWOBAR
            nw_barrier();                                         // stage s's weights are in buffer s & 1
#else
            __syncthreads();                                      // stage s's weights are in buffer s & 1
#endif
            nw_frag nbq[2][4];
            float2 nbx[2], nbsum[2];
            if (on && s + 1 < NS) fetch_b(s + 1, nbq, nbx, nbsum);
            if (on) {
                const int bf = s & 1;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int rb = 32 * wm + 16 * mt;
#if STRATA_NAT_W12
                        // k 32 h + 8 hi .. + 7 (A0) and 32 h + 16 + 8 hi .. + 7 (A1) of row rb + l16
                        const uint2* ap = reinterpret_cast<const uint2*>(&wt[bf][rb + l16][32 * h + 8 * hi]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[2]);
#else
                        const uint4* ap = reinterpret_cast<const uint4*>(&wt[bf][rb + l16][32 * h]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[1]);
#endif
                        float w0[8], w1[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float2 sw = *reinterpret_cast<const float2*>(&ws[bf][rb + NW_DROW(i, hi)][2 * h]);
                            w0[i] = sw.x; w1[i] = sw.y;
                        }
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const float dx = h ? bx[nt].y : bx[nt].x;
                            const nw_ab B0 = nw_v(bq[nt][2 * h]), B1 = nw_v(bq[nt][2 * h + 1]);
                            const nw_i8 m = nw_i8{MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC};
                            if constexpr (K16) {
                                const nw_i8 d0 = nw_wmma(A0, B0, m), d1 = nw_wmma(A1, B1, m);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const float v = fmaf(w1[i], dotf(d1[i]), w0[i] * dotf(d0[i]));
                                    acc[mt][nt][i] = fmaf(dx, v, acc[mt][nt][i]);
                                }
                            } else {
                                const nw_i8 d = nw_wmma(A1, B1, nw_wmma(A0, B0, m));
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    acc[mt][nt][i] = fmaf(w0[i] * dx, dotf(d[i]), acc[mt][nt][i]);
                                    if constexpr (has_min(WT)) acc[mt][nt][i] = fmaf(w1[i] * dx, h ? bsum[nt].y : bsum[nt].x, acc[mt][nt][i]);
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);                            // the other buffer: its readers passed this barrier
#ifndef STRATA_Q_TWOBAR
                // (the next stage's first barrier orders these writes: nw_barrier waits for them explicitly)
#else
                __syncthreads();                                      // the writes of every thread are in LDS before the next stage reads them (#1180)
#endif
                if (s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                if (on) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bx[nt] = nbx[nt];
                        bsum[nt] = nbsum[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            if constexpr (ALIAS) __syncthreads();                 // every wave is done reading the weight buffers
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
#if STRATA_NAT_W12
                        // lane l: local rows rb + 8 hi + i = features 16 wm + 8 mt + 4 hi + j, gate (i = 2j) and up (2j + 1)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float gt = acc[mt][nt][2 * j], up = acc[mt][nt][2 * j + 1];
                            hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + 4 * hi + j] = gt / (1.0f + __expf(-gt)) * up;
                        }
#else
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float up = __shfl_xor_sync(0xffffffffu, acc[mt][nt][i], 16);
                            const float gt = acc[mt][nt][i];
                            if (hi == 0) hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + i] = gt / (1.0f + __expf(-gt)) * up;
                        }
#endif
                    }
            }
            __syncthreads();
            if constexpr (is_tq(WT)) {
                // (STRATA_PF_FUSED_TQ=1, gfx12, a Hadamard-folded down: set_down_rotation) H as FP32, n_ff values at the start of the row's own
                // down-output row in `dm` (n_embd floats a row; the down product rewrites it only after rot_quant_h_kernel has read it), as
                // CUDA's native_kernel does (the dm-row invariant: RESULTS-OWN-4X section 8): rotated and quantized there before the down product
                if (dm != nullptr) {
                    if (tid < 2 * kTileRows) {
                        const int r = tid >> 1, hh = tid & 1;
                        if (r < nrows) {
                            float* o = dm + (size_t) (row0 + r) * kMoeEmbd + (NW_ROWS / 2) * fb + 32 * hh;
#pragma unroll 8
                            for (int j = 0; j < 32; ++j) o[j] = hs[r][32 * hh + j];
                        }
                    }
                    continue;   // (the next work item: its first barrier orders these reads of hs before the buffers are reused)
                }
            }
            // H block fb (features 64 fb ..) to int8 per 32, natural order: a thread per (tile row, half)
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (size_t) fb * AB;
                    uint32_t wd[8] = {};
                    int csum = 0;
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int c = __float2int_rn(hs[r][32 * hh + j] * inv);
                        csum += c;
                        wd[j >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (j & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;   // (the formats with a minimum)
                }
            }
        } else {
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int r = 32 * wn + 16 * nt + l16;
                        if (r < nrows) {
#if STRATA_NAT_W12
                            // columns rbase + 32 wm + 16 mt + 8 hi .. + 7: contiguous, 32-byte aligned (dm: 16, checked in experts_native)
                            float4* d = reinterpret_cast<float4*>(dm + (size_t) (row0 + r) * kMoeEmbd + rbase + 32 * wm + 16 * mt + 8 * hi);
                            d[0] = make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
                            d[1] = make_float4(acc[mt][nt][4], acc[mt][nt][5], acc[mt][nt][6], acc[mt][nt][7]);
#else
                            float* d = dm + (size_t) (row0 + r) * kMoeEmbd + rbase + 32 * wm + 16 * mt + hi;
#pragma unroll
                            for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
#endif
                        }
                    }
            }
        }
    }
#else
    __builtin_trap();
#endif
}

// STRATA_PF_F16=1 (gfx11): the other native formats on the F16 matrix cores - moe_f16_common.cuh's work item,
// staged activations and epilogues.  load_unit / convert decode each sub-block once into the weight tile in LDS (int8
// codes, the scales as half2), and each wave turns its own 16 rows into F16 before its products.  Gate/up rows 0..63
// are the item's gate rows, 64..127 its up rows; down rows are output rows.  The output differs from
// native_w11_kernel's (F16 products instead of int8 dots with float scales).
// gfx1151, IQ3_S / IQ4_NL, 16384 tokens x top 10 over 512 experts, one 128-expert launch (ms, native_w11_kernel ->
// this): gate/up 11.65 -> 9.87, down 6.11 -> 4.71; weights decoded to F16 in LDS (18 KB a block) were 9.83 and 5.71.
template <int WT, bool GU, int BN, bool D16 = false>
__global__ void __launch_bounds__(f16::THREADS)
native_f16_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                  const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_F16_DEVICE
    // (moe_fused_iq.cu has its own THREADS and AB: the f16 names are qualified where they collide)
    using namespace f16;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = GU ? GU_NFB : D_NFB;
    constexpr int RW = raw_words(WT);
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    constexpr int SGN = (WT == T_IQ2_XXS || WT == T_IQ2_XS || WT == T_IQ2_S || WT == T_IQ3_XXS || WT == T_IQ3_S)
                            ? sign_entries(WT) : 1;
    constexpr int WLD8 = 80;   // bytes per weight row: 64 int8 codes + 16 (rows 80 bytes apart spread a fragment read)
    constexpr int W_BYTES = ROWS * WLD8 + 2 * ROWS * 4;   // the codes, then the (s0, s1) half2 per row and sub-block
    constexpr int STAGE_BYTES = W_BYTES + ActStage<BN>::BYTES;
    static_assert(GU_EPI_BYTES <= STAGE_BYTES && D_EPI_BYTES <= STAGE_BYTES, "the epilogues fit the stage");
    __shared__ __align__(16) uint8_t lds[STAGE_BYTES];
    __shared__ __align__(16) uint8_t sgrid[GB];
    __shared__ __align__(16) uint4 ssign[SGN];
    __shared__ int srow[BN];
    uint8_t* s_w = lds;
    auto* s_sc = reinterpret_cast<uint32_t*>(lds + ROWS * WLD8);
    auto* s_act = reinterpret_cast<uint4*>(lds + W_BYTES);
    const int tid = threadIdx.x, wave = tid >> 5, sub_lane = tid & 15;

    int fb, e, row0, nrows;
    if (!work_item<BN, NFB>(b, tb.ts, tb.tiles, tb.off, blockIdx.x, fb, e, row0, nrows)) return;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += f16::THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += f16::THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int live_tiles = (nrows + 15) / 16;
    const uint8_t* blob = b.blob[e - b.e0];
    if (tid < BN) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;   // tokens; H by routed row
    __syncthreads();

    const int ur = tid >> 1, uj = tid & 1;   // this thread's decode unit: weight-tile row ur, sub-block uj of a stage
    const uint8_t* wrow = GU ? blob + (ur >= 64 ? geo.up_off : 0) + (size_t) (fb * 64 + (ur & 63)) * geo.gu_row
                             : blob + geo.down_off + (size_t) (fb * ROWS + ur) * geo.d_row;
    const auto unit = [&](int s) -> const uint8_t* {
        if (GU) return wrow + (s >> 2) * BS;
        return (WT == T_IQ4_NL || WT == T_Q5_0 || WT == T_Q8_0) ? wrow + (2 * s + uj) * BS : wrow + s * BS;
    };
    const auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
    uint32_t raw[RW];
    ActStage<BN> as;
    as.init(act, GU ? GU_ACT_LD : H_LD, srow, tid);
    const auto fetch_stage = [&](int s) {
        load_unit<WT>(unit(s), sub(s), raw);
        as.fetch(s);
    };
    // the sub-block as int8 codes and its scales into the weight tile
    const auto commit_stage = [&]() {
        uint32_t q[8];
        float s0, s1;
        convert<WT, RW>(raw, sgrid, kv, q, s0, s1, ssign);
        uint4* d = reinterpret_cast<uint4*>(s_w + ur * WLD8 + 32 * uj);
        d[0] = make_uint4(q[0], q[1], q[2], q[3]);
        d[1] = make_uint4(q[4], q[5], q[6], q[7]);
        s_sc[uj * ROWS + ur] = __builtin_bit_cast(uint32_t, __floats2half2_rn(s0, K16 ? s1 : s0));
        as.commit(s_act);
    };
    v8f acc[BN / 16];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) acc[j] = v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // int8 codes to F16 per wave (gufo's Q8_0 decode): the sign bit flipped carries q + 128, a byte permute makes it
    // the mantissa of 1024 (exact), minus 1152 is q, then one packed multiply by the sub-block's scale
    const __half2 magic = __floats2half2_rn(-1152.0f, -1152.0f), zero2 = __floats2half2_rn(0.0f, 0.0f);
    const auto compute_stage = [&]() {
        const int row = wave * 16 + sub_lane;
        const uint4* wp = reinterpret_cast<const uint4*>(s_w + row * WLD8);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const uint4 c0 = wp[2 * kb], c1 = wp[2 * kb + 1];
            const __half2 sc = __builtin_bit_cast(__half2, s_sc[kb * ROWS + row]);
            const __half2 sc0 = __low2half2(sc), sc1 = __high2half2(sc);
            const uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            uint32_t hw[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {   // values 4i .. 4i + 3 (s0: values 0-15, s1: 16-31)
                const uint32_t c = words[i] ^ 0x80808080u;
                const uint32_t p0 = __builtin_amdgcn_perm(c, 0x64646464u, 0x01050004u);
                const uint32_t p1 = __builtin_amdgcn_perm(c, 0x64646464u, 0x03070206u);
                const __half2 s2 = i < 4 ? sc0 : sc1;
                hw[2 * i] = __builtin_bit_cast(uint32_t, __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), s2, zero2));
                hw[2 * i + 1] = __builtin_bit_cast(uint32_t, __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), s2, zero2));
            }
            v16h a_lo, a_hi;
            __builtin_memcpy(&a_lo, &hw[0], 32);
            __builtin_memcpy(&a_hi, &hw[8], 32);
            mma_tiles<BN>(acc, a_lo, a_hi, s_act, kb, sub_lane, live_tiles);
        }
    };
    fetch_stage(0);
    for (int s = 0; s < NS; ++s) {
        commit_stage();
        __syncthreads();
        if (s + 1 < NS) fetch_stage(s + 1);
        compute_stage();
        __syncthreads();
    }
    if constexpr (GU) gu_epilogue<BN>(acc, lds, tid, live_tiles, nrows, row0, fb, out);
    else down_epilogue<BN, D16>(acc, lds, tid, live_tiles, nrows, row0, fb, dm);
#else
    (void) b; (void) geo; (void) tb; (void) act; (void) src; (void) out; (void) dm;
    __builtin_trap();
#endif
}
#endif  // __HIPCC__

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

// per device: whether every kernel here runs (sm_80+, device code in this build, fits), and their occupancy
struct DevInfo {
    bool done = false, ok = false;
    bool g1151 = false; // gfx1151 (Strix Halo): Q8_0 down layers on the F16 kernel by default (q8d_f16_default)
    bool w12 = false;   // gfx12 (RDNA4): the native kernels only - the Q2_0 pack's (moe_fused.cu) are not ported there
    bool t75 = false;   // Turing (STRATA_PF_FUSED_SM75): the Q2_0 / Q2_0 layers only (what was set up and checked there)
    bool q2n = false;   // sm_80+ with STRATA_PF_FUSED_Q2N=1: the native Q2_0 pack's Q2_0 gate/up is set up as well (opt-in)
    bool s120 = false;  // compute capability 12.0 (sm120_series.hpp): the default for the formats measured there
    bool s121 = false;  // GB10 (sm_121, DGX Spark): 128-row tiles win from 65,536 routed rows a launch up (pick_ww)
    bool s86 = false;   // the sm_86 rule class, compute capability 8.0 / 8.6 / 8.9 (RTX A6000 / A40 / RTX 3090 class; A100 and RTX 40 since 11-12): the same default as s120 with 80+ SMs (STRATA_PF_FUSED_SM86=0 off)
    int sms = 0, occ = 1;
    int occ_nt = 0;     // blocks per multiprocessor of the NT = 256 kernels (0: not set up)
    bool ws = false;    // STRATA_PF_FUSED_WS: the warp-specialized kernels are set up
#if !defined(__HIPCC__)
    bool q50 = false;   // sm_80+ with STRATA_PF_FUSED_Q50=1: the Q5_0 down kernels are set up (opt-in, phase 4 step 1)
    int occ_q50 = 1;    // their blocks per multiprocessor (kept apart, so the other formats' grids do not change)
    bool kq = false;    // the K-quant kernels (Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down) are set up: by default on the cards of fused_tile::kq_takes, STRATA_PF_FUSED_KQ=1 on any sm_80+ card, =0 off
    int occ_kq = 1;     // their blocks per multiprocessor (their own, as occ_q50)
    bool kq8 = false;   // with kq: the Q8_0 down kernels are set up too (A2; UD-Q4_K_XL's five Q8_0-down layers)
    int occ_kq8 = 1;
    bool q8d = false;   // sm_80+ with STRATA_PF_FUSED_Q8D=1: the same Q8_0 down kernels behind an i-quant gate/up (opt-in, phase 4; their occupancy is occ_kq8)
    bool q8gu = false;  // sm_80+ with STRATA_PF_FUSED_Q8GU=1: the Q8_0 gate/up kernels, and the Q8_0 down ones behind them (opt-in, phase 4)
    int occ_q8gu = 1;   // the Q8_0 gate/up kernels' blocks per multiprocessor (the down keeps occ_kq8)
    bool q6k = false;   // sm_80+ with STRATA_PF_FUSED_Q6K=1: the Q6_K gate/up kernels, and the Q8_0 down ones behind them (opt-in, phase 4)
    int occ_q6k = 1;    // the Q6_K gate/up kernels' blocks per multiprocessor (the down keeps occ_kq8)
    bool tq = false;    // the trellis kernels (TQK6 / TQ2_T gate/up, TQK7 / TQ2_T down) are set up: by default on sm_86 with 80+ SMs, STRATA_PF_FUSED_TQ=1 on any sm_80+ (fused_tq_default.hpp)
    int occ_tq_gu = 1, occ_tq_d = 1;   // their blocks per multiprocessor (their own)
#endif
};
std::mutex g_mu;
DevInfo g_dev[32];

#if !defined(__HIPCC__)
// the activation stages of native_kernel on the current device: its ASTAGES_DEV (2 below sm_80)
int host_astages() {
    static int per[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return ASTAGES; }
    if (!per[dev]) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        per[dev] = strata::cc_major_of(major) < 8 ? 2 : ASTAGES;
    }
    return per[dev];
}
// STRATA_PF_FUSED_Q50 on CUDA: "1" sets the Q5_0 down kernels up (dev_info) and lets native_supported take the Q5_0-down layers on
// any sm_80+ card; anything else (unset, "0") keeps those layers on the dequantize + FP16 GEMM path (the default)
bool q50_cuda_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q50"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_Q8D on CUDA: "1" sets the Q8_0 down kernels up (dev_info) and lets native_supported take a Q8_0 down behind an i-quant
// gate/up (the pinned UD-IQ4_XS: IQ3_S / Q8_0 in 4 layers, IQ4_XS / Q8_0 in 1) on any sm_80+ card; anything else keeps those layers on MMQ
bool q8d_cuda_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q8D"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_Q8GU on CUDA: "1" sets the Q8_0 gate/up kernels up (dev_info) and lets native_supported take a Q8_0 gate/up with a Q8_0
// down (the Q8_0 set's 48 layers, UD-Q6_K_XL's one Q8_0 layer) on any sm_80+ card; anything else keeps those layers on MMQ
bool q8gu_cuda_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q8GU"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_Q6K on CUDA: "1" sets the Q6_K gate/up kernels up (dev_info) and lets native_supported take a Q6_K gate/up with a Q8_0
// down (UD-Q6_K_XL's 47 layers, UD-Q5_K_XL's one Q6_K layer) on any sm_80+ card; anything else keeps those layers on MMQ
bool q6k_cuda_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q6K"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_TQ on CUDA (the rule is include/strata/prefill/fused_tq_default.hpp, pinned by tests/prefill/fused_tq_default_test.cpp): unset = the trellis kernels are
// set up (dev_info) and native_supported takes Gyro-S's TQK6 / TQK7 layers on a card of the sm_86 class (compute capability 8.6, and 8.0 / 8.9 since 11-12; fused_tile::s86_class) with 80+ SMs (RTX A6000 class; the A6000 gate series is
// Gyro-S); "1" takes them on any sm_80+ card, and Gyro-M's TQ2_T / TQ2_T layers too (no gate series: they are never taken by default); any other value ("0") and every other card: MMQ.  The Hadamard rotations of a folded layer run inside
// experts_native (set_down_rotation), the gate/up input arrives rotated.
int tq_env() {
    static const int v = strata::prefill::fused_tq::env_value();
    return v;
}
[[maybe_unused]] bool kq_on();   // STRATA_PF_FUSED_KQ (below): the HIP builds' reading of the switch (the CUDA builds decide in dev_info with fused_tile::kq_takes)
template <int T, bool GU, int WW, int NT = THREADS, bool F16 = false> bool setup_ww(int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, native_kernel<T, GU, WW, NT, F16>) != cudaSuccess || fa.ptxVersion < 75) return false;
    const size_t sm = smem_bytes(T, WW, host_astages(), NT, F16, GU);
    if (cudaFuncSetAttribute(native_kernel<T, GU, WW, NT, F16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_kernel<T, GU, WW, NT, F16>, NT, sm) != cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
// STRATA_PF_FUSED_NT=256 (sm_80+, opt-in): the 256-thread kernels (native_kernel's NT)
// STRATA_PF_FUSED_F16=1 (sm_80+, opt-in, changes the output at rounding level): the NT = 256 kernels' F16 products
bool f16c_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_F16"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool nt256_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_NT"); return v != nullptr && std::atoi(v) == 256; }();
    return on || f16c_requested();
}
bool ws_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_WS"); return v != nullptr && v[0] == '1'; }();
    return on;
}
template <int T, bool GU> bool setup_ws() {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, native_ws_kernel<T, GU>) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(native_ws_kernel<T, GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ws_smem_bytes(T)) !=
        cudaSuccess) return false;
    int o = 0;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_ws_kernel<T, GU>, WS_THREADS, ws_smem_bytes(T)) == cudaSuccess && o >= 1;
}
template <int T, bool GU> bool setup_nt(int& occ) {
    return f16c_requested() ? setup_ww<T, GU, 2, 256, true>(occ) : setup_ww<T, GU, 2, 256>(occ);
}
#endif

#if defined(__HIPCC__)
// Resident blocks per multiprocessor for the gfx11 WMMA prompt-expert kernels' persistent grids.  HIP counts a gfx11
// WGP (two CUs, which these kernels use as one in the default WGP mode) as one multiprocessor, and the occupancy query
// answers 1 block for these kernels where 2 run side by side - so the grid was half the GPU's room.  gfx1151 (20
// WGPs), 8192 tokens x top 10, IQ3_S / IQ4_NL, one layer: 1 block per WGP 60.7 ms, 2 44.7, 3 49.9, 4 45.2 (the same
// work items and arithmetic: the results do not change).  STRATA_PF_OCC=N sets the blocks per WGP (an experiment knob).
static int wgp_blocks(int occ) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? env : 2 * std::max(occ, 1);
}
// The occupancy variants' (W, X1) blocks per WGP: as many as are resident at once, so the persistent grid has no second
// wave.  gfx12 (a WGP: 128 KB of LDS, 1536 VGPRs per SIMD; the compiler's occupancy, -Rpass-analysis=kernel-resource-usage,
// clang 23): gate/up IQ2_S (37,120 B of LDS) and IQ2_XS (203 VGPRs) 3, every other one 4.  R9700, IQ2_S / IQ4_NL, one
// layer, 4 runs each: 4K 8.80-8.93 ms at 4, 8.18-8.33 at 3; 16K -3%; IQ3_S / IQ4_NL 2-3% slower at 3.  gfx11 keeps 4
// (measured on gfx1151).  Not from the occupancy API: it counts 64 KB of LDS per multiprocessor (37,120 B -> 1 block).
// STRATA_PF_OCC_W=N sets it for every occupancy variant (an experiment knob).
static int occ_blocks(bool w12, int type, bool gu) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC_W");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    if (env > 0) return env;
    // gfx11 too (gfx1151, IQ2_S gate/up: 235 VGPRs and 37,120 B of LDS leave room for 3 blocks of 8 waves per WGP; a grid of 4 per WGP
    // ran a second, lopsided wave: 16K tokens, one layer, IQ2_S / IQ4_NL 81.7 -> 76.5 ms)
    // The K-quant kernels (Q4_K / Q5_K gate/up, Q5_1 down) reach this launch only with the trial switch STRATA_PF_KQ_OCC=1 (experts_native):
    // 3 blocks per WGP, measured on gfx1201 (R9700, hip_prefill_fused_iq, one layer, 8,192 tokens x top 10 over 512 experts): Q4_K / Q5_1 16.57 ms
    // on the default launch, 15.96 at 4 blocks, 15.54 at 3, 16.52 at 2; Q5_K / Q5_1 17.25 / 16.69 / 16.20 / 17.18; parity and outputs unchanged.
    if (type == T_Q4_K || type == T_Q5_K || type == T_Q5_1) return 3;
    return gu && (type == T_IQ2_S || type == T_IQ2_XS) ? 3 : 4;
}
#endif
const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#elif defined(__HIPCC__)
    {   // gfx11 / gfx12 with this build's code: the WMMA kernels (one occupancy for all: the same shape and LDS budget)
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceProp prop;
        hipFuncAttributes fa{};
        const bool props = cudaGetDeviceProperties(&prop, dev) == cudaSuccess;
        d.w12 = props && strata::kernels::gfx_arch_is_gfx12_wmma(prop.gcnArchName);
        d.g1151 = props && strata::kernels::gfx_arch_is_gfx1151(prop.gcnArchName);
        // the arch lists of STRATA_NAT_W11 / STRATA_NAT_W12 (gfx_arch.hpp: never a prefix - another gfx11 / gfx12 part
        // has the trapping body, which hipFuncGetAttributes and the occupancy query still accept).  And the body this
        // build loaded for the device is the real one: the __builtin_trap() stub (a gfx11-generic / gfx12-generic or
        // SPIR-V build on a listed card) declares no LDS, the real kernel at least its weight buffers.
        if (!props || !(strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName) || d.w12) ||
            hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(native_w11_kernel<T_IQ3_XXS, true>)) != hipSuccess ||
            fa.sharedSizeBytes < (size_t) 2 * NW_ROWS * WLD) {
            cudaGetLastError();
            return d;
        }
        int o = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_w11_kernel<T_IQ2_S, true>, NW_THREADS, 0) != hipSuccess)
            o = 0;
        d.ok = o >= 1;
        d.occ = d.ok ? o : 1;
        cudaGetLastError();
        return d;
    }
#else
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    d.s120 = strata::sm120_default(major, minor);   // (takes the raw attributes, as is_sm120 does)
    d.s121 = major == 12 && minor == 1;             // GB10 (DGX Spark): its own tile threshold, see pick_ww
    // the sm_86 rule class (raw attributes, as s120; fused_tile::s86_class: compute capability 8.6, and since 11-12 8.0 and 8.9): STRATA_PF_FUSED_SM86=0 is a diagnostic for bug reports (MMQ on these cards again); it goes with MMQ
    static const bool sm86_off = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM86"); return v != nullptr && v[0] == '0'; }();
    d.s86 = strata::prefill::fused_tile::s86_class(major, minor, sm86_off);
    major = strata::cc_major_of(major);   // STRATA_EMULATE_CC counts (tests)
    minor = strata::cc_minor_of(minor);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    // Turing (sm_75): the same kernels with plain copies and m8n8k16 MMAs, opt-in (STRATA_PF_FUSED_SM75=1)
    static const bool sm75 = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM75"); return v != nullptr && v[0] == '1'; }();
    if (major < 8 && !(major == 7 && minor == 5 && sm75)) return d;
    int occ = 1 << 20;
    if (major < 8) {   // Turing: the canonical Q2_0 pack's kernels (the codebook formats' do not fit 64 KB at 4 warps a row)
        d.t75 = true;
        d.ok = setup_one<T_Q2_0, true>(occ) && setup_one<T_Q2_0, false>(occ);
        d.occ = d.ok ? occ : 1;
        cudaGetLastError();
        return d;
    }
    d.ok = setup_one<T_IQ2_XXS, true>(occ) && setup_one<T_IQ2_XS, true>(occ) && setup_one<T_IQ2_S, true>(occ) &&
           setup_one<T_IQ3_XXS, true>(occ) && setup_one<T_IQ3_S, true>(occ) && setup_one<T_IQ4_XS, true>(occ) &&
           setup_one<T_Q2_0, false>(occ) && setup_one<T_IQ4_NL, false>(occ);   // (Q2_0 gate/up: the Turing opt-in's, or below)
    // STRATA_PF_FUSED_Q2N=1: the native Q2_0 pack's Q2_0 gate/up on these kernels here too (what #2145 took with
    // STRATA_PF_FUSED=1 on every CUDA card until b6443f3a kept it to Turing).  Its own variable, so STRATA_PF_FUSED=1 keeps
    // its bits: unset, the kernel is not set up and the Q2_0 layers keep MMQ.  A failed setup leaves occ as it was.
    static const bool q2n = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q2N"); return v != nullptr && v[0] == '1'; }();
    if (int o2 = occ; q2n && d.ok && setup_one<T_Q2_0, true>(o2)) { d.q2n = true; occ = o2; }
    d.occ = d.ok ? occ : 1;
    // STRATA_PF_FUSED_Q50=1 (opt-in on CUDA, off by default): the Q5_0 down projections (UD-IQ4_XS v31: 33 of 48 layers, which have
    // no MMQ tile here and otherwise take the dequantize + FP16 GEMM path) on native_kernel - the decode is the HIP kernels'
    // (load_unit / convert<T_Q5_0>: int8 codes q - 16, one scale per 32 values), the products the IQ4_NL down's (int8 mma.sync,
    // FP32 scales and sums).  Their occupancy is their own (occ_q50), so the other formats' launches do not change; a failed
    // setup leaves those layers on the FP16 path.
    if (int o3 = 1 << 20; d.ok && q50_cuda_requested() && setup_one<T_Q5_0, false>(o3)) { d.q50 = true; d.occ_q50 = o3; }
    // The K-quant experts of UD-Q4_K_XL - gate/up Q4_K / Q5_K, down Q5_1 (or Q8_0) - on native_kernel instead of MMQ (libstrata_kq).  On CUDA they are the
    // default on the cards of fused_tile::kq_takes (sm_86-class and sm_120 with 80+ SMs; not the 40-79 SM cards: their one-layer table is below MMQ at 20,000 tokens),
    // STRATA_PF_FUSED_KQ=1 sets them up on every sm_80+ card, STRATA_PF_FUSED_KQ=0 (or STRATA_PF_FUSED=0, native_supported) leaves them on MMQ.
    // The decode is the HIP kernels' (load_unit / convert: int8 codes, a scale and a minimum per 32 values); the minimum enters as s1 d_x (sum of the activation
    // codes), from the sums quant_act_nat_kernel and the K-quant gate/up epilogue store in the spare bytes.  Their own occupancy (occ_kq); a failed setup leaves
    // those layers on MMQ.  (kq_on() below is the HIP builds' reading of the switch; strata-q36 sets STRATA_PF_FUSED_KQ=1 itself, so its cards are unchanged.)
    static const int kq_env = strata::prefill::fused_tile::pf_env(std::getenv("STRATA_PF_FUSED_KQ"));
    const bool kq_want = strata::prefill::fused_tile::kq_takes(kq_env, {d.s120, d.s121, d.sms, d.s86});
    if (int o4 = 1 << 20; d.ok && kq_want && setup_one<T_Q4_K, true>(o4) && setup_one<T_Q5_K, true>(o4) && setup_one<T_Q5_1, false>(o4)) {
        d.kq = true;
        d.occ_kq = o4;
        // (A2) the Q8_0 down behind a K-quant gate/up: int8 codes and one scale per 32 values, no minimum (load_unit / convert<T_Q8_0>);
        // its own occupancy, and a failed setup leaves those layers on MMQ
        if (int o5 = 1 << 20; setup_one<T_Q8_0, false>(o5)) { d.kq8 = true; d.occ_kq8 = o5; }
    }
    // STRATA_PF_FUSED_Q8D=1 (opt-in on CUDA, off by default): the Q8_0 down behind an i-quant gate/up on the same Q8_0 down instances as
    // A2's.  The Q8_0 down reads no code sums (has_min is false), so the i-quant gate/up epilogue's H feeds it unchanged.  The occupancy is
    // the one A2 found for the same instances (occ_kq8); a failed setup leaves those layers on MMQ.
    if (int o6 = 1 << 20; d.ok && q8d_cuda_requested() && setup_one<T_Q8_0, false>(o6)) { d.q8d = true; d.occ_kq8 = o6; }
    // STRATA_PF_FUSED_Q8GU=1 (opt-in on CUDA, off by default): a Q8_0 gate/up (int8 codes, one scale per 32 values: two blocks a 64-value
    // stage, as the Q8_0 down reads them) with a Q8_0 down on A2's down instances.  Both set up or neither; the gate/up's occupancy is its own.
    if (int o7 = 1 << 20, o8 = 1 << 20; d.ok && q8gu_cuda_requested() && setup_one<T_Q8_0, true>(o7) && setup_one<T_Q8_0, false>(o8)) {
        d.q8gu = true;
        d.occ_q8gu = o7;
        d.occ_kq8 = o8;
    }
    // STRATA_PF_FUSED_Q6K=1 (opt-in on CUDA, off by default): a Q6_K gate/up (int8 codes q - 32, an int8 scale times d per 16 values: the
    // per-16-value scale path of IQ2_XS / IQ2_S) with a Q8_0 down on A2's down instances.  Both set up or neither; its own occupancy.
    if (int o9 = 1 << 20, o10 = 1 << 20; d.ok && q6k_cuda_requested() && setup_one<T_Q6_K, true>(o9) && setup_one<T_Q8_0, false>(o10)) {
        d.q6k = true;
        d.occ_q6k = o9;
        d.occ_kq8 = o10;
    }
    // the trellis experts (Gyro-S TQK6 gate/up + TQK7 down, Gyro-M TQ2_T both) on native_kernel, by default on sm_86 with 80+ SMs, STRATA_PF_FUSED_TQ=1 on any sm_80+ card, =0 off:
    // the decode is MMQ's (tq.cuh: 15-bit states, the int8 codebook tq_lut_i8, one scale d * TQ_LUT_I8_SCALE a 128-value block), the products the int8
    // mma.sync path.  All four instance pairs set up or none; their own occupancy.
    if (int og = 1 << 20, od = 1 << 20; d.ok && strata::prefill::fused_tq::on(tq_env(), d.s86, d.sms) && setup_one<T_TQK6, true>(og) && setup_one<T_TQ2_T, true>(og) &&
                                       setup_one<T_TQK7, false>(od) && setup_one<T_TQ2_T, false>(od)) {
        d.tq = true;
        d.occ_tq_gu = og;
        d.occ_tq_d = od;
    }
    if (d.ok && nt256_requested()) {
        int o2 = 1 << 20;
        const bool ok2 = setup_nt<T_IQ2_XXS, true>(o2) && setup_nt<T_IQ2_XS, true>(o2) && setup_nt<T_IQ2_S, true>(o2) &&
                         setup_nt<T_IQ3_XXS, true>(o2) && setup_nt<T_IQ3_S, true>(o2) && setup_nt<T_IQ4_XS, true>(o2) &&
                         setup_nt<T_Q2_0, false>(o2) && setup_nt<T_IQ4_NL, false>(o2) &&
                         (!d.q2n || setup_nt<T_Q2_0, true>(o2));   // (the Q2_0 gate/up of STRATA_PF_FUSED_Q2N=1 too)
        d.occ_nt = ok2 ? o2 : 0;
        if (!ok2) std::fprintf(stderr, "prefill fused experts (native): STRATA_PF_FUSED_NT=256 is not available here\n");
    }
    if (d.ok && ws_requested()) {
        d.ws = setup_ws<T_IQ2_XXS, true>() && setup_ws<T_IQ2_XS, true>() && setup_ws<T_IQ2_S, true>() &&
               setup_ws<T_IQ3_XXS, true>() && setup_ws<T_IQ3_S, true>() && setup_ws<T_IQ4_XS, true>() &&
               setup_ws<T_Q2_0, false>() && setup_ws<T_IQ4_NL, false>() &&
               (!d.q2n || setup_ws<T_Q2_0, true>());   // (the Q2_0 gate/up of STRATA_PF_FUSED_Q2N=1 too)
        if (!d.ws) std::fprintf(stderr, "prefill fused experts (native): STRATA_PF_FUSED_WS=1 is not available here\n");
    }
    cudaGetLastError();
#endif
    return d;
}

// opt-in on top of STRATA_PF_FUSED=1: STRATA_PF_FUSED_KQ=1 takes UD-Q4_K_XL's formats (the output differs from the FP16
// expert path's at rounding level; KL-gated, see aurora_s23.md)
bool kq_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_KQ");
        if (v != nullptr && v[0] != '\0') return v[0] == '1';
#if defined(__HIPCC__) && defined(_WIN32)
        // Round 0215 (STRIX-FIT): on a Strix Halo (gfx1151) under Windows the K-quant experts of UD-Q4_K_XL (Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down)
        // take the fused matrix-core kernels (the F16 ones with STRATA_PF_F16=1, which the gfx1151 prompt set turns on) by default.
        // STRATA_PF_FUSED_KQ=0 or STRATA_GFX1151_DEFAULTS=0 keeps the earlier route.
        const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
        if (off != nullptr && off[0] == '0') return false;
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            return strata::kernels::gfx_arch_is_gfx1151(prop.gcnArchName);
        cudaGetLastError();
#endif
        return false;
    }();
    return on;
}
// STRATA_PF_F16=1 (gfx11): the formats native_f16_kernel takes (the K-quants are moe_f16_kq.cu's)
bool f16_gu(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS;
}
bool f16_d(int t) { return t == T_Q2_0 || t == T_IQ4_NL || t == T_Q8_0 || t == T_Q5_0; }
#if defined(__HIPCC__)
bool tq_hip_requested();   // (STRATA_PF_FUSED_TQ on HIP: below)
bool q8gu_hip_requested();   // (STRATA_PF_FUSED_Q8GU on HIP: below)
bool q6k_hip_requested();    // (STRATA_PF_FUSED_Q6K on HIP: below)
#endif
bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS
#if !defined(__HIPCC__)
           || (t == T_Q2_0 && (dev_info().t75 || dev_info().q2n))   // Q2_0 gate/up: the Turing opt-in, or STRATA_PF_FUSED_Q2N=1 on sm_80+
           || ((t == T_Q4_K || t == T_Q5_K) && dev_info().kq)       // STRATA_PF_FUSED_KQ=1 and its kernels set up
           || (t == T_Q8_0 && dev_info().q8gu)                      // STRATA_PF_FUSED_Q8GU=1 and its kernels set up
           || (t == T_Q6_K && dev_info().q6k)                       // STRATA_PF_FUSED_Q6K=1 and its kernels set up
           || ((t == T_TQK6 || t == T_TQ2_T) && dev_info().tq)      // the trellis kernels set up (default on sm_86 with 80+ SMs, or STRATA_PF_FUSED_TQ=1)
#endif
#if defined(__HIPCC__)
           || (t == T_Q2_0 && dev_info().w12)   // gfx12: the native Q2_0 pack's gate/up on native_w11_kernel (64-value blocks)
           || ((t == T_Q4_K || t == T_Q5_K) && kq_on())   // the gfx11 kernels only
           || ((t == T_TQK6 || t == T_TQ2_T) && dev_info().w12 && tq_hip_requested())   // STRATA_PF_FUSED_TQ=1 on gfx12
           || (t == T_Q8_0 && dev_info().w12 && q8gu_hip_requested())   // STRATA_PF_FUSED_Q8GU=1 on gfx12
           || (t == T_Q6_K && dev_info().w12 && q6k_hip_requested())    // STRATA_PF_FUSED_Q6K=1 on gfx12
#endif
        ;
}
bool d_covered(int t) {
    return t == T_Q2_0 || t == T_IQ4_NL
#if defined(__HIPCC__)
           || ((t == T_Q5_1 || t == T_Q8_0) && kq_on())
#endif
        ;
}
// d_covered on this device: STRATA_PF_F16=1 also takes Q8_0 down on gfx11 without STRATA_PF_FUSED_KQ - the F16 kernel
// has no minimum term, and its output already differs from the int8 kernels' (UD-IQ4_XS: the five Q8_0-down layers
// would otherwise leave the fused path for MMQ).
// gfx1151: a Q8_0 down (UD-IQ4_XS and UD-Q4_K_XL keep five layers of it) takes the F16 matrix-core kernel without STRATA_PF_F16,
// whose output differs from the int8 kernels' only for that layer; every other format keeps the bits it has.  Without it those
// layers leave the fused path for MMQ: UD-IQ4_XS 4K / 8K prompts 1,057 / 1,128 against 1,192 / 1,257 tok/s with it (Windows
// gfx1151, eng034 with STRATA_PF_F16=1, which also covers these layers; A/B in phaseA-tests/strix-u).  STRATA_PF_Q8D_F16=0 turns it off.
bool q8d_f16_default(const DevInfo& d) {
#if defined(__HIPCC__)
    if (d.w12 || !d.g1151) return false;
    static const bool off = [] { const char* v = std::getenv("STRATA_PF_Q8D_F16"); return v != nullptr && v[0] == '0'; }();
    return !off;
#else
    (void) d;
    return false;
#endif
}
// Q5_0 down (UD-IQ4_XS-v31: 33 of 48 layers, with IQ3_XXS or IQ4_XS gate/up).  Round 0217 (Strix Halo): on gfx1151 under Windows these layers take the F16
// matrix-core kernel (native_f16_kernel, the K-quant experts' route: int8 codes q - 16 times the block scale, F16 products, FP16 down rows for the combine) instead
// of leaving the fused path for gather + FP16 dequantize + hipBLASLt (RESULTS-STRIX2.md: 24-29% of the prompt).  It needs the F16 prompt set (STRATA_PF_F16, on by
// default there), so with STRATA_PF_F16=0 the layers keep the earlier route.  STRATA_PF_FUSED_Q50=0 (or STRATA_GFX1151_DEFAULTS=0) turns it off; =1 turns it on
// for any gfx1151 build (Linux: opt-in).  gfx12 (RDNA4) keeps its int8 kernel (native_w11_kernel, int8 codes q - 16: its WMMA layout), reached through the same
// switch.  gfx12 under Linux: ON by default since 0.2.15 (owner 11-10; R9700, UD-IQ4_XS-v31: prompt x1.31 / x1.47 / x1.16 at 4K / 8K / 20K, first-token KL
// against the FP16 path level with the default's, the neutral-writer forced-token gate PASS, first-8K reads PASS in 64 starts); gfx12 under Windows: opt-in
// (not measured there).  STRATA_PF_FUSED_Q50=0 keeps the layers on gather + FP16 dequantize + hipBLASLt (the earlier route, kept two releases).
// The switch means the same everywhere: 1 on, 0 off, unset = the platform's default.
bool q50_on(const DevInfo& d) {
#if defined(__HIPCC__)
    if (!d.w12 && !d.g1151) return false;
    static const int env = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q50"); return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '1' ? 1 : 0); }();
    if (env >= 0) return env == 1;
    if (d.w12) return strata::core::kGfx12PromptDefaultsOn;   // gfx12: on by default under Linux, opt-in under Windows
#if defined(_WIN32)
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    return !(off != nullptr && off[0] == '0');
#else
    return false;
#endif
#else
    (void) d;
    return false;
#endif
}
#if defined(__HIPCC__)
// STRATA_PF_FUSED_Q8D on HIP (phase 4, opt-in, gfx12 only): "1" lets a Q8_0 down behind an i-quant gate/up (the pinned UD-IQ4_XS's five Q8_0-down layers:
// IQ3_S / Q8_0 4, IQ4_XS / Q8_0 1) take native_w11_kernel's Q8_0 down instead of MMQ, without STRATA_PF_FUSED_KQ (whose meaning is the K-quant experts).
// The same switch as CUDA's.  No new kernel: the gfx12 int8 Q8_0 down instances exist for the K-quant route.  gfx11 keeps its own rules (q8d_f16_default,
// STRATA_PF_F16): not measured with this switch.
bool q8d_hip_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q8D"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_TQ on HIP (phase 4, opt-in, gfx12 only): "1" lets Gyro-S's TQK6 / TQK7 and Gyro-M's TQ2_T / TQ2_T layers take native_w11_kernel's
// trellis instances instead of MMQ (the decode is CUDA's: 15-bit states, MMQ's int8 codebook tq_lut_i8, one scale a 128-value block; Hadamard-folded
// downs through the FP32-H epilogue and rot_quant_h_kernel, as on CUDA).  The same switch as CUDA's.  gfx11: not measured, keeps MMQ.
bool tq_hip_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_TQ"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_Q8GU on HIP (phase 4, opt-in, gfx12 only): "1" lets a Q8_0 gate/up with a Q8_0 down (the Q8_0 set's 48 layers, UD-Q6_K_XL's one Q8_0
// layer) take native_w11_kernel's Q8_0 gate/up instance instead of MMQ (32-value blocks, two a stage, like the Q8_0 down; the same decode as CUDA's).  The
// same switch as CUDA's.  gfx11: not measured, keeps MMQ.
bool q8gu_hip_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q8GU"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_PF_FUSED_Q6K on HIP (phase 4, opt-in, gfx12 only): "1" lets a Q6_K gate/up with a Q8_0 down (UD-Q6_K_XL's 47 layers, UD-Q5_K_XL's one) take
// native_w11_kernel's Q6_K gate/up instance instead of MMQ (the K-quant library, STRATA_MMQ_KQ=1, on gfx12).  256-value superblocks, an int8 scale per
// 16 values (native_w11's per-16 path), the same decode as CUDA's.  The same switch as CUDA's.  gfx11: not measured, keeps its route.
bool q6k_hip_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FUSED_Q6K"); return v != nullptr && v[0] == '1'; }();
    return on;
}
#endif
bool d_covered_on(int t, const DevInfo& d) {
#if defined(__HIPCC__)
    if (t == T_Q5_0) return d.w12 ? q50_on(d) : q50_on(d) && f16::requested();   // gfx1151 has no int8 kernel for Q5_0 down (the F16 prompt set must be on); gfx12 has only the int8 one
    if (t == T_Q8_0 && !d.w12 && (f16::requested() || q8d_f16_default(d))) return true;
    if (t == T_Q8_0 && d.w12 && (q8d_hip_requested() || q8gu_hip_requested() || q6k_hip_requested())) return true;   // (behind an i-quant gate/up with Q8D, a Q8_0 one with Q8GU, a Q6_K one with Q6K: native_supported)
    if (t == T_TQK7 || t == T_TQ2_T) return d.w12 && tq_hip_requested();   // STRATA_PF_FUSED_TQ=1 (behind a trellis gate/up only: native_supported)
#else
    if (t == T_Q5_0) return d.q50;   // STRATA_PF_FUSED_Q50=1 and its kernels set up (dev_info)
    if (t == T_Q5_1) return d.kq;    // STRATA_PF_FUSED_KQ=1 (behind a K-quant gate/up only: native_supported)
    if (t == T_TQK7 || t == T_TQ2_T) return d.tq;   // STRATA_PF_FUSED_TQ=1 (behind a trellis gate/up only: native_supported)
    if (t == T_Q8_0) return d.kq8 || d.q8d || d.q8gu || d.q6k;   // STRATA_PF_FUSED_KQ=1, A2 (behind a K-quant gate/up), STRATA_PF_FUSED_Q8D=1 (behind an i-quant one), _Q8GU=1 (behind a Q8_0 one) or _Q6K=1 (behind a Q6_K one): native_supported
#endif
    (void) d;
    return d_covered(t);
}

#if defined(__HIPCC__)
int host_astages() { return ASTAGES; }   // (#2145 defines it for CUDA above; HIP keeps the one depth)
#endif
template <int T, bool GU>
void launch(int ww, unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb, const void* act,
            const int32_t* src, void* out, float* dm, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    if (ww == 4) native_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4, host_astages(), THREADS, false, GU), s>>>(b, g, tb, a, src, o, dm);
    else native_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2, host_astages(), THREADS, false, GU), s>>>(b, g, tb, a, src, o, dm);
}
// the NT = 256 kernels (WW 2), their F16 form (STRATA_PF_FUSED_F16) or the warp-specialized kernel (ws:
// STRATA_PF_FUSED_WS)
template <int T, bool GU>
void launch(bool ws, unsigned grid, const WideBatch& b, const NativeGeom& g, const Tables& tb, const void* act,
            const int32_t* src, void* out, float* dm, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
#if !defined(__HIPCC__)   // (native_ws_kernel and the F16 path are CUDA only; d.ws is never set on HIP)
    if (ws)
        native_ws_kernel<T, GU><<<grid, WS_THREADS, ws_smem_bytes(T), s>>>(b, g, tb, a, src, o, dm);
    else if (f16c_requested())
        native_kernel<T, GU, 2, 256, true><<<grid, 256, smem_bytes(T, 2, host_astages(), 256, true, GU), s>>>(b, g, tb, a, src, o, dm);
    else
#else
    (void) ws;
#endif
    native_kernel<T, GU, 2, 256><<<grid, 256, smem_bytes(T, 2, host_astages(), 256, false, GU), s>>>(b, g, tb, a, src, o, dm);
}

// The work item's shape for a layer of `n` routed rows over `n_expert` experts: 128 routed rows when an expert has
// about one such tile (56 to 112 rows on average), else 64.  Measured on the RTX 5070 (tests/cuda/prefill_fused_iq_test,
// a layer of 2048 / 3584 / 8192 tokens, ms, 64 -> 128): IQ2_S 5.2 -> 6.7, 9.5 -> 9.3, 17.5 -> 19.5; IQ2_XXS 4.2 ->
// 5.8, 7.4 -> 7.6, 13.6 -> 16.5; IQ3_S 5.6 -> 6.6, 10.2 -> 8.8, 18.4 -> 19.0; IQ3_XXS 5.1 -> 6.3, 9.1 -> 8.3, 16.5 ->
// 17.5 (256 rows was slower still).  STRATA_PF_FUSED_TILE=64|128 forces one.
#if !defined(__HIPCC__)
// (staging) what the weight copies assume of a launch: the expert blobs are device memory (the prompt path's expert cache slots and its stage ring), not a host-mapped
// pointer, and every row and offset is 2-byte aligned (the decode reads 16-bit words at the same offsets inside the shared-memory window).  A launch that breaks it
// does not end the process: it takes the unstaged 64-row kernel of the same formats (same output bits), whatever STRATA_PF_FUSED_TILE says, and the first such launch of a
// process logs one line.  Checked on the first 256 launches of a process only (a pointer query is a driver call); once one launch failed, every later launch of the process
// takes the unstaged kernel without asking (fused_tile::StagedGuard: the decision is a pure function, tests/prefill/fused_tile_rule_test.cpp).
std::mutex g_staged_mu;
strata::prefill::fused_tile::StagedGuard g_staged_guard;
int g_staged_fallbacks = 0;   // launches that took the unstaged kernel for this reason (the GPU test reads it)

bool staged_blobs_ok(const Batch& b, const NativeGeom& g) {
    namespace ft = strata::prefill::fused_tile;
    {
        std::lock_guard<std::mutex> lk(g_staged_mu);
        const ft::StagedGuard::Step st = g_staged_guard.next();
        if (st == ft::StagedGuard::Step::kTrust) return true;
        if (st == ft::StagedGuard::Step::kUnstaged) { ++g_staged_fallbacks; return false; }
    }
    char why[256];
    why[0] = 0;
    bool ok = ft::staged_geometry_ok(g.gu_row, g.d_row, g.up_off, g.down_off);
    if (!ok) {
        std::snprintf(why, sizeof why, "rows and offsets must be 2-byte aligned (gu_row %zu d_row %zu up_off %zu down_off %zu)", g.gu_row, g.d_row, g.up_off, g.down_off);
    } else {
        for (int e = b.e0; e < b.e1; ++e) {
            const uint8_t* p = b.blob[e - b.e0];
            cudaPointerAttributes pa{};
            ft::BlobMem m = ft::BlobMem::kUnknown;
            if (cudaPointerGetAttributes(&pa, p) != cudaSuccess) cudaGetLastError();   // (not a pointer the driver knows: the launch's own error says so)
            else m = pa.type == cudaMemoryTypeDevice ? ft::BlobMem::kDevice : pa.type == cudaMemoryTypeManaged ? ft::BlobMem::kManaged : pa.type == cudaMemoryTypeHost ? ft::BlobMem::kHost : ft::BlobMem::kUnregistered;
            if (!ft::staged_blob_ok(m, reinterpret_cast<uintptr_t>(p))) {
                std::snprintf(why, sizeof why, "expert blobs must be in device memory at 2-byte aligned addresses (expert %d: %p, memory type %d)", e, (const void*) p, (int) pa.type);
                ok = false;
                break;
            }
        }
    }
    std::lock_guard<std::mutex> lk(g_staged_mu);
    if (g_staged_guard.report(ok))
        std::fprintf(stderr, "prefill fused experts (native): %s; this launch and every later one in this process take the unstaged kernel (same results, slower)\n", why);
    if (!ok) ++g_staged_fallbacks;
    return ok;
}
#endif
// `staged`: the launch's gate/up kernel is a weight-staging instance (sw_on); the rule then takes 128 rows from every size on the sm_86 / sm_120 cards under 80 SMs.
// `wide`: the launch's gate/up is Q6_K or Q8_0: on sm_86 cards with 40+ SMs the rule takes 128 rows above 112 rows an expert (fused_tile_rule.hpp).
// `kq`: 0, or the launch's pair is one of Flash-Next's K-quant pairs (fused_tile::kq_flash_kind: 1 a Q5_1 down, 2 a Q8_0 down): on sm_86-class cards with 80-99 SMs the rule takes 128 rows from 5.8 rows an expert up.
int pick_ww(int64_t n, int n_expert, bool staged, bool wide, int kq) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_TILE");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 ? 4 : t == 128 ? 2 : 0;
    }();
    if (forced) return forced;
    // the rule itself (the card classes, the thresholds and the measurements behind them) is include/strata/prefill/fused_tile_rule.hpp, a pure function the unit test
    // tests/prefill/fused_tile_rule_test.cpp pins for every measured card class.  STRATA_PF_TILE_DEBUG=1 prints the choice of every launch of a class that has its own rule.
    const DevInfo& d = dev_info();
    const strata::prefill::fused_tile::Pick pk = strata::prefill::fused_tile::pick({d.s120, d.s121, d.sms, d.s86}, n, n_expert, staged, wide, kq);
    using strata::prefill::fused_tile::Why;
    if (pk.why != Why::kBand) {
        static const bool dbg = [] { const char* v = std::getenv("STRATA_PF_TILE_DEBUG"); return v != nullptr && v[0] == '1'; }();
        if (dbg) {
            const double avg = (double) n / std::max(n_expert, 1);
            const char* cls = pk.why == Why::kS121 ? "s121" : pk.why == Why::kS120Small ? "s120 small" : pk.why == Why::kStaged ? "staged" : pk.why == Why::kS86Wide ? "s86 wide" : pk.why == Why::kS86Kq ? "s86 kq" : pk.why == Why::kS120Kq ? "s120 kq" : "s120";
            std::fprintf(stderr, "strata: fused tile: rows %lld experts %d avg %.1f %s yes -> ww %d\n", (long long) n, n_expert, avg, cls, pk.ww);
        }
    }
    return pk.ww;
}

// The work item of one launch of the native kernels: pick_ww's, unless the launch has a weight-staging kernel (`staged`: its gate/up kernel is one; the down's is looked at here) and
// the staging check fails (staged_blobs_ok), in which case the unstaged 64-row instances run (fused_tile::kUnstagedWw), also when STRATA_PF_FUSED_TILE=128 forces the 128-row tile.
// Every call site that launches `launch<...>` with a ww from pick_ww on a batch of blobs takes its ww from here.
int launch_ww(const Batch& b, const NativeGeom& g, int64_t n, int n_expert, bool staged, bool wide, int kq = 0) {
#if !defined(__HIPCC__)
    if ((staged || sw_on(g.d_type, false)) && !staged_blobs_ok(b, g)) return strata::prefill::fused_tile::kUnstagedWw;
#else
    (void) b;
    (void) g;
#endif
    return pick_ww(n, n_expert, staged, wide, kq);
}
#endif
}  // namespace
#if !defined(STRATA_FUSED_DECODE_ONLY)

// Launches that took the unstaged kernel because the staging check failed (see staged_blobs_ok); 0 in a HIP build, which has no staging kernels.
int staged_blob_fallbacks() {
#if !defined(__HIPCC__)
    std::lock_guard<std::mutex> lk(g_staged_mu);
    return g_staged_fallbacks;
#else
    return 0;
#endif
}

bool gfx12_iq2s_default() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_IQ2S");
        const char* all = std::getenv("STRATA_PF_FUSED");
        // ON by default on gfx12 (owner decision 11-01, amd-q RESULTS.md section 12; the quick quality suite lost 2 of 19 tasks with it on, the full suite
        // decided).  STRATA_PF_FUSED_IQ2S=0 turns it off, as does STRATA_PF_FUSED=0.
        return strata::core::kGfx12PromptDefaultsOn && !(v != nullptr && v[0] == '0') && !(all != nullptr && all[0] == '0');
    }();
    return on;
}

#if defined(__HIPCC__)
// IQ2_XXS / IQ2_XS gate/up on gfx12 (the IQ3_XXS pack's 19 layers that kept MMQ, and the IQ2_XS sets): ON by default under Linux since 0.2.15 (owner 11-10;
// R9700, IQ3_XXS: prompt x1.17 / x1.14 / x1.13 at 4K / 8K / 20K, first-token KL against the FP16 path level with the default's, the neutral-writer forced-token
// gate PASS, first-8K reads PASS).  STRATA_PF_FUSED_IQ2X=0 keeps these layers on MMQ (the earlier route, kept two releases), as does STRATA_PF_FUSED=0.
// Windows gfx12: opt-in (STRATA_PF_FUSED=1), as before.  Like gfx12_iq2s_default, native_supported applies it to gfx12 cards only (d.w12).
static bool gfx12_iq2x_default() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_IQ2X");
        const char* all = std::getenv("STRATA_PF_FUSED");
        return strata::core::kGfx12PromptDefaultsOn && !(v != nullptr && v[0] == '0') && !(all != nullptr && all[0] == '0');
    }();
    return on;
}
#endif

bool native_supported(int gu_type, int d_type) {
    static const bool off = [] {   // STRATA_PF_FUSED_NATIVE=0: the native packs keep MMQ under STRATA_PF_FUSED=1 (A/B)
        const char* v = std::getenv("STRATA_PF_FUSED_NATIVE");
        return v != nullptr && v[0] == '0';
    }();
    // gfx12: fused::available() (the Q2_0 pack's expert_w11_kernel, not ported) stays false, so requested() does too; the
    // same opt-in (STRATA_PF_FUSED=1) is read here.  Every other device: requested() as before.
    // unset: on by default on gfx12 for IQ3_S, IQ3_XXS and (148-amd) Q2_0 gate/up (R9700: prompt +13..+20%, first-token KL against the FP16
    // prompt path equal to MMQ's within noise, 96 prompts each; Q2_0: prompt 4K +16.7%, 20K +19.0%, teacher-forced KL 0.0004 / 0.0016,
    // tasks equal to MMQ's); STRATA_PF_FUSED=0 turns it off, =1 keeps the opt-in for every covered format
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
        return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '1' ? 1 : 0);
    }();
    if (off || env == 0 || !gu_covered(gu_type)) return false;
    if (strata::prefill::pf_fused_override().load(std::memory_order_relaxed) == 0) return false;   // the small-card prompt path (pf_fused_override.hpp)
#if !defined(__HIPCC__)
    // CUDA, a K-quant gate/up: a Q4_K / Q5_K gate/up with a Q5_1 (or, with the Q8_0 down kernels set up, Q8_0) down takes the fused
    // kernels on any sm_80+ card whose K-quant kernels were set up: by default on the cards of fused_tile::kq_takes (dev_info), with
    // STRATA_PF_FUSED_KQ=1 on every card, never with STRATA_PF_FUSED_KQ=0; other downs keep MMQ for now.  A Q5_1
    // down needs the code sums only the K-quant gate/up epilogue writes, so it is never taken behind another gate/up.  Not with the
    // NT = 256 / warp-specialized opt-ins (no K-quant form): MMQ.
    // CUDA, a Q8_0 down behind an i-quant gate/up (phase 4, opt-in): with STRATA_PF_FUSED_Q8D=1 such a layer takes the fused kernels on
    // any sm_80+ card whose Q8_0 down kernels were set up, whatever the 80-SM rule says (the pinned UD-IQ4_XS's five Q8_0-down layers;
    // its other 43 layers follow their own rule).  Not behind the Q2_0 gate/up of STRATA_PF_FUSED_Q2N, not with the NT = 256 /
    // warp-specialized opt-ins (no Q8_0 form): MMQ.
    // A Q8_0 gate/up (phase 4, opt-in STRATA_PF_FUSED_Q8GU=1; gu_covered checked q8gu above): with a Q8_0 down only, same conditions.
    if (gu_type == T_Q8_0) {
        const DevInfo& d = dev_info();
        return d_type == T_Q8_0 && d.ok && d.q8gu && !d.t75 && d.occ_nt <= 0 && !d.ws;
    }
    // A Q6_K gate/up (phase 4, opt-in STRATA_PF_FUSED_Q6K=1; gu_covered checked q6k above): with a Q8_0 down only, same conditions.
    if (gu_type == T_Q6_K) {
        const DevInfo& d = dev_info();
        return d_type == T_Q8_0 && d.ok && d.q6k && !d.t75 && d.occ_nt <= 0 && !d.ws;
    }
    // The trellis pairs (phase 4; default on sm_86 with 80+ SMs, STRATA_PF_FUSED_TQ=1 on any sm_80+ card; d.tq): Gyro-S's TQK6 / TQK7 and Gyro-M's TQ2_T / TQ2_T only, same conditions.  (A trellis
    // down behind any other gate/up, or the reverse: MMQ.)
    if (is_tq(gu_type) || is_tq(d_type)) {
        // the compiled geometry must suit the trellis path: whole 128-value blocks along both K, and (a folded down) the FP32 H of a row,
        // n_ff floats, inside the row's own n_embd-float down-output row (set_down_rotation's invariant); the geometry checks allow n_ff up to
        // 1,100 and n_embd down to 1,024, so a geometry that breaks it declines the trellis fused path (MMQ) instead of overrunning the row
        constexpr bool geo_ok = D_ROWS_K <= D_ROWS_N && D_ROWS_K % 128 == 0 && GU_ROWS_K % 128 == 0;
        const DevInfo& d = dev_info();
        // Gyro-S's pair follows the default rule (d.tq); Gyro-M's TQ2_T / TQ2_T has no gate series yet: only with STRATA_PF_FUSED_TQ=1 (fused_tq::gyro_m_on)
        const bool pair = (gu_type == T_TQK6 && d_type == T_TQK7) || (gu_type == T_TQ2_T && d_type == T_TQ2_T && strata::prefill::fused_tq::gyro_m_on(tq_env()));
        return geo_ok && pair && d.ok && d.tq && !d.t75 && d.occ_nt <= 0 && !d.ws;
    }
    if (d_type == T_Q8_0 && gu_type != T_Q4_K && gu_type != T_Q5_K) {
        const DevInfo& d = dev_info();
        return gu_type != T_Q2_0 && d.ok && d.q8d && !d.t75 && d.occ_nt <= 0 && !d.ws;
    }
    if (gu_type == T_Q4_K || gu_type == T_Q5_K || d_type == T_Q5_1 || d_type == T_Q8_0) {
        const DevInfo& d = dev_info();
        return (gu_type == T_Q4_K || gu_type == T_Q5_K) && (d_type == T_Q5_1 || (d_type == T_Q8_0 && d.kq8)) && d.ok && d.kq && !d.t75 &&
               d.occ_nt <= 0 && !d.ws;
    }
    // CUDA, a Q5_0 down (phase 4 step 1, opt-in): with STRATA_PF_FUSED_Q50=1 such a layer takes the fused kernels on any sm_80+ card
    // whose Q5_0 kernels were set up, whatever the 80-SM rule below says (STRATA_PF_FUSED=0 still turns every fused kernel off, above).
    // Not with the NT = 256 / warp-specialized opt-ins (their kernels have no Q5_0 form): those layers keep the FP16 path.
    if (d_type == T_Q5_0) {
        const DevInfo& d = dev_info();
        return d.ok && d.q50 && !d.t75 && d.occ_nt <= 0 && !d.ws;
    }
    // CUDA: unset, on by default on a compute capability 12.0 card with 80+ SMs (sm120_series.hpp, STRATA_SM120_PERF=0 off) for the gate/up
    // formats measured there - the IQ3_S pack's IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS layers with IQ4_NL / Q2_0 down (RTX PRO 5000:
    // prompt +7.0% 4K, +5.2% 20K, +3.4% 98K; teacher-forced KL against MMQ 0.0002 over 5998 positions, 0.0010 over 1760 long
    // ones; first-token KL against the FP16 prompt path 0.01181 vs MMQ's 0.01236 over 48 prompts; HumanEval, reasoning,
    // agent, needles, CJK and tool calls the same).  The same on a compute capability 8.6 card with 80+ SMs (RTX A6000, 84 SMs, IQ3_S pack, 11-11: prompt +6.7% 4K, +3.6% 8K,
    // +1.0% 20K, 5/5 each; first-token KL against the FP16 prompt path at or below MMQ's at 2,048 / 4K / 8K / 20K; neutral-writer
    // forced-token gate PASS over 48 prompts; small chunks, agent, first 8K read, backstop as MMQ; STRATA_PF_FUSED_SM86=0 off).
    // IQ2_XXS / IQ2_XS gate/up (the IQ3_XXS and IQ2_XS packs) joined on 11-12 on the owner's word (RTX A6000, IQ3_XXS pack: prompt time fused / MMQ-mixed 0.874-0.976 from 300 to 8,193
    // tokens on the integration head, 5 rounds; teacher-forced KL vs the FP16 prompt path 0.965 / 1.062 / 1.072 / 1.008 at 512 / 1,000 / 2,048 / 8,192 ids; the rule: fused_tile_rule.hpp).
    // Every other CUDA card (and the 40-79 SM cards for every launch outside the staged class): opt-in as before.
    if (env < 0) {
        // the native Q2_0 pack's Q2_0 / Q2_0 layers: only with STRATA_PF_FUSED_Q2N=1 (d.q2n), on any sm_80+ card
        if (gu_type == T_Q2_0) { const DevInfo& d = dev_info(); return d.ok && d.q2n && d_type == T_Q2_0 && d_covered_on(d_type, d); }
        if (!strata::prefill::fused_tile::default_gate_format(gu_type)) return false;   // IQ2_XXS / IQ2_XS joined 11-12 (owner), see fused_tile_rule.hpp
        const DevInfo& d = dev_info();
        // 12.0 and 8.6 cards with 80+ SMs (RTX PRO 5000 / 6000, RTX 5080 / 5090; RTX A6000 / A40, RTX 3090 / 3090 Ti / 3080 Ti), and since 11-12 8.0 and 8.9 cards (A100; RTX 4090 / 4080 Super, RTX 6000 Ada: the same code, no card run) take every format of the default list;
        // on an RTX 5070 (48 SMs, IQ2_XS pack, experts partly in RAM) the fused kernels were 4.7% slower on 20K prompts (0/6 pairs), so the 40-79 SM cards take only the staged launches below
        // (the unstaged formats keep MMQ there unless STRATA_PF_FUSED=1)
        // The sm_86 / sm_120 cards with 40-79 SMs (RTX A4000, RTX 5070 class) also take the launches with a staged gate/up kernel (IQ3_XXS / IQ3_S / IQ4_XS) and an IQ4_NL / Q2_0 down, by default since
        // the owner's word of 11-12 (RTX A4000, IQ3_XXS set, end to end: prompt time MMQ / fused 1.000 / 1.003 / 1.028 / 1.012 at 2K / 4K / 8K / 20K, forced-token KL against the FP16 prompt path
        // 0.930 [0.881, 0.988] at 2,048 and 0.973 [0.898, 1.057] at 8,192 ids); STRATA_PF_FUSED_SM40=0 takes them back to MMQ (a diagnostic like STRATA_PF_FUSED_SM86=0).  The rule itself is
        // fused_tile::default_takes_env (unit-tested over every card flag, SM count, switch state and launch kind).  This whole branch is CUDA-only (the #if !defined(__HIPCC__) above): a HIP build never reads it.
        static const int sm40_env = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM40"); return v == nullptr ? -1 : (v[0] == '0' ? 0 : 1); }();
        return d.ok && strata::prefill::fused_tile::default_takes_env(env, {d.s120, d.s121, d.sms, d.s86}, sm40_env, sw_on(gu_type, true), d_type == T_IQ4_NL || d_type == T_Q2_0, gu_type) && !d.t75 &&
               d_covered_on(d_type, d);
    }
    if (gu_type == T_Q2_0 && d_type != T_Q2_0) return false;   // the opt-in (STRATA_PF_FUSED=1 as well) takes the Q2_0 / Q2_0 pair only
#endif
#if defined(__HIPCC__)
    // The trellis pairs on gfx12 (opt-in STRATA_PF_FUSED_TQ=1; gu_covered checked it above): Gyro-S's TQK6 / TQK7 and Gyro-M's TQ2_T / TQ2_T only, with
    // CUDA's geometry rule (whole 128-value blocks along both K; a folded down's FP32 H row inside the row's own n_embd-float output row).  Any other
    // pairing with a trellis format: MMQ.
    if (is_tq(gu_type) || is_tq(d_type)) {
        constexpr bool geo_ok = D_ROWS_K <= D_ROWS_N && D_ROWS_K % 128 == 0 && GU_ROWS_K % 128 == 0;
        const DevInfo& d = dev_info();
        const bool pair = (gu_type == T_TQK6 && d_type == T_TQK7) || (gu_type == T_TQ2_T && d_type == T_TQ2_T);
        return geo_ok && pair && d.ok && d.w12 && tq_hip_requested();
    }
    // A Q8_0 gate/up on gfx12 (opt-in STRATA_PF_FUSED_Q8GU=1; gu_covered checked it above): with a Q8_0 down only.
    if (gu_type == T_Q8_0) {
        const DevInfo& d = dev_info();
        return d_type == T_Q8_0 && d.ok && d.w12 && q8gu_hip_requested();
    }
    // A Q6_K gate/up on gfx12 (opt-in STRATA_PF_FUSED_Q6K=1; gu_covered checked it above): with a Q8_0 down only.
    if (gu_type == T_Q6_K) {
        const DevInfo& d = dev_info();
        return d_type == T_Q8_0 && d.ok && d.w12 && q6k_hip_requested();
    }
#endif
    // default: only these three; the Q2_0 one (148-amd) is on Linux only: on Windows it is opt-in (STRATA_PF_FUSED=1) until tested there.
    // AMD-Q (owner decision 10-31): the IQ2_S and IQ4_XS gate/up layers too (the IQ3_S pack's 21 layers that kept MMQ), Linux only like the Q2_0 entry.
    // R9700 (p620, IQ3_S, serve, 3 interleaved pairs): prompt 2K +9.7%, 8K +12.2%, 32K +8.8%; first-token KL against the MMQ default 0.017 (29 prompts of
    // 1,025 tokens, top-1 28 of 29), the answers are not the MMQ bytes.  STRATA_PF_FUSED_IQ2S=0 turns it off; STRATA_PF_FUSED=0 turns all fused kernels off.
    if (env < 0 && gu_type != T_IQ3_S && gu_type != T_IQ3_XXS && !(gu_type == T_Q2_0 && strata::core::kGfx12PromptDefaultsOn) &&
        !(gfx12_iq2s_default() && (gu_type == T_IQ2_S || gu_type == T_IQ4_XS))
#if defined(__HIPCC__)
        && !(gfx12_iq2x_default() && (gu_type == T_IQ2_XXS || gu_type == T_IQ2_XS))   // (0.2.15, Linux gfx12: above)
#endif
        ) return false;
    const DevInfo& d = dev_info();
#if defined(__HIPCC__)
    // STRATA_PF_FUSED_Q8D=1 on gfx12 covers the Q8_0 down behind an i-quant gate/up only (as on CUDA): a Q2_0 gate/up with a Q8_0 down keeps MMQ there
    // unless STRATA_PF_FUSED_KQ=1 (the K-quant route's wider rule, unchanged; gfx11's rules unchanged)
    if (d.w12 && gu_type == T_Q2_0 && d_type == T_Q8_0 && !kq_on()) return false;
    // d_covered_on takes a Q8_0 down on gfx12 for STRATA_PF_FUSED_Q8GU=1 too (behind a Q8_0 gate/up, returned above): an i-quant / K-quant gate/up
    // keeps its own rule (STRATA_PF_FUSED_Q8D=1 or the K-quant route's)
    if (d.w12 && d_type == T_Q8_0 && !kq_on() && !q8d_hip_requested()) return false;
#endif
    if (!d.ok || !d_covered_on(d_type, d)) return false;
    if (d.t75 && (gu_type != T_Q2_0 || d_type != T_Q2_0)) return false;
    if (env < 0) return d.w12;
    return requested() || d.w12;
}

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act_native");
}

// The down rows as FP16 (opt-in per call: the prompt path asks, STRATA_PF_DM16): dm_f16_used() tells which form the last call wrote.
static bool g_dm16_want = false, g_dm16_used = false;
void want_dm_f16(bool on) { g_dm16_want = on; }
bool dm_f16_used() { return g_dm16_used; }

// (CUDA, STRATA_PF_FUSED_TQ=1) a Hadamard-folded down (set_down_rotation): on, and the down input's sign vector
static const float* g_rot_signs = nullptr;
static bool g_rot_on = false;
void set_down_rotation(bool on, const float* signs) { g_rot_on = on; g_rot_signs = on ? signs : nullptr; }
void hadamard128_rows(float* x, int64_t rows, int64_t cols, const float* signs, void* stream) {
    if (rows <= 0) return;
    if (cols % 128 != 0) { std::fprintf(stderr, "hadamard128_rows: %lld columns are not whole 128-value blocks\n", (long long) cols); std::exit(1); }
    const int64_t nblk = rows * (cols / 128);
    hadamard128_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (int) cols, signs);
    ck(cudaGetLastError(), "hadamard128_rows");
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    g_dm16_used = false;
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered_on(g.d_type, d)) {
        std::fprintf(stderr, "prefill fused experts (native): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
#if defined(__HIPCC__)
    {
        const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
        // W (X1): LDS alias of the H tile + waves_per_eu(8) (192 VGPRs) + 4 blocks per WGP, only for the types whose
        // kernels do not spill under the cap (measured: gate/up IQ2_*, IQ3_*, IQ4_XS 0-5 regs; down IQ4_NL 11, Q8_0 / Q2_0 0;
        // Q4_K / Q5_K gate/up and Q5_1 down spill 100+ and keep the old launch).
        // Those counts are from a clang before 22; from clang 22 nw_capped decides which of these kernels get the cap
        // (above NW_LB), and the launch is the same.
#ifdef STRATA_W_NO_OCC
        const bool occ_gu = false, occ_d = false;
#else
        // STRATA_PF_KQ_OCC=1 (a trial switch, off by default, HIP only): the K-quant kernels too (Q4_K / Q5_K gate/up, Q5_1 down) take the
        // occupancy launch above (the H tile on the weight buffers, occ_blocks per WGP; the VGPR cap only where nw_capped says so).
        static const bool kq_occ = [] { const char* v = std::getenv("STRATA_PF_KQ_OCC"); return v != nullptr && v[0] == '1'; }();
        const bool occ_gu = g.gu_type == T_IQ2_XXS || g.gu_type == T_IQ2_XS || g.gu_type == T_IQ2_S ||
                            g.gu_type == T_IQ3_XXS || g.gu_type == T_IQ3_S || g.gu_type == T_IQ4_XS || g.gu_type == T_Q2_0 ||
                            (kq_occ && (g.gu_type == T_Q4_K || g.gu_type == T_Q5_K));
        const bool occ_d = g.d_type == T_Q2_0 || g.d_type == T_Q8_0 || g.d_type == T_IQ4_NL || (kq_occ && g.d_type == T_Q5_1);
#endif
        if (d.w12 && (reinterpret_cast<uintptr_t>(dm) & 15) != 0) {   // gfx12's down store: two float4 per lane
            std::fprintf(stderr, "prefill fused experts (native): the down rows (%p) must be 16-byte aligned\n", (void*) dm);
            std::exit(1);
        }
        const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / NW_ROWS),
                                                           (int64_t) d.sms * (occ_gu ? occ_blocks(d.w12, g.gu_type, true) : wgp_blocks(d.occ)));
        const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / NW_ROWS),
                                                          (int64_t) d.sms * (occ_d ? occ_blocks(d.w12, g.d_type, false) : wgp_blocks(d.occ)));
        const uint8_t* xa8 = (const uint8_t*) xa;
        uint8_t* ha8 = (uint8_t*) ha;
#define STRATA_NW_GU(T) do { if (occ_gu) native_w11_kernel<T, true, true, NW_LB><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); else native_w11_kernel<T, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); } while (0)
        // STRATA_PF_F16=1: gfx11's F16 matrix cores - the K-quants' kernels (moe_f16_kq.cu) and native_f16_kernel for
        // f16_gu / f16_d (gfx12's WMMA layout differs, so it keeps the int8 kernels)
        const bool f16 = !d.w12 && f16::requested();
        const bool f16_q8d = !f16 && g.d_type == T_Q8_0 && q8d_f16_default(d);   // (the Q8_0 down only)
        const bool f16_wide = f16::pick_bn(n, n_expert) == 2 * kTileRows;
        // STRATA_PF_DM16 (the prompt path asks per call): when this layer's down runs on an F16 kernel its rows go to `dm` as FP16 (moe_combine16 reads them)
        const bool d16 = g_dm16_want && ((f16 && (f16kq::covers_down(g.d_type) || f16_d(g.d_type))) || f16_q8d);
        g_dm16_used = d16;
#define STRATA_F16_LAUNCH(T, GU, NFB, ACT, OUT, DM) do { \
    const unsigned grid_f = (unsigned) std::max<int64_t>(1, tiles * (NFB)); \
    if (d16 && !(GU)) { \
        if (f16_wide) native_f16_kernel<T, GU, 2 * kTileRows, !(GU)><<<grid_f, f16::THREADS, 0, s>>>(b, g, tb, ACT, src, OUT, DM); \
        else native_f16_kernel<T, GU, kTileRows, !(GU)><<<grid_f, f16::THREADS, 0, s>>>(b, g, tb, ACT, src, OUT, DM); \
    } else if (f16_wide) native_f16_kernel<T, GU, 2 * kTileRows><<<grid_f, f16::THREADS, 0, s>>>(b, g, tb, ACT, src, OUT, DM); \
    else native_f16_kernel<T, GU, kTileRows><<<grid_f, f16::THREADS, 0, s>>>(b, g, tb, ACT, src, OUT, DM); \
} while (0)
        if (f16 && f16kq::covers(g.gu_type)) f16kq::gate_up(b, g, tb.ts, tb.tiles, tb.off, n, n_expert, tiles, xa8, src, ha8, s);
        else if (f16 && f16_gu(g.gu_type)) {
#define STRATA_F16_GU(T) STRATA_F16_LAUNCH(T, true, f16::GU_NFB, xa8, ha8, nullptr)
            switch (g.gu_type) {
                case T_IQ2_XXS: STRATA_F16_GU(T_IQ2_XXS); break;
                case T_IQ2_XS: STRATA_F16_GU(T_IQ2_XS); break;
                case T_IQ2_S: STRATA_F16_GU(T_IQ2_S); break;
                case T_IQ3_XXS: STRATA_F16_GU(T_IQ3_XXS); break;
                case T_IQ3_S: STRATA_F16_GU(T_IQ3_S); break;
                default: STRATA_F16_GU(T_IQ4_XS); break;
            }
#undef STRATA_F16_GU
        }
        else switch (g.gu_type) {
            case T_IQ2_XXS: STRATA_NW_GU(T_IQ2_XXS); break;
            case T_IQ2_XS: STRATA_NW_GU(T_IQ2_XS); break;
            case T_IQ2_S: STRATA_NW_GU(T_IQ2_S); break;
            case T_IQ3_XXS: STRATA_NW_GU(T_IQ3_XXS); break;
            case T_IQ3_S: STRATA_NW_GU(T_IQ3_S); break;
            case T_Q2_0: STRATA_NW_GU(T_Q2_0); break;
            case T_Q4_K: STRATA_NW_GU(T_Q4_K); break;
            case T_Q5_K: STRATA_NW_GU(T_Q5_K); break;
            // (STRATA_PF_FUSED_TQ=1, gfx12) the trellis gate/up: the plain launch; a folded down (set_down_rotation) takes H as FP32 rows in dm
            // (STRATA_PF_FUSED_Q8GU=1, gfx12) the Q8_0 gate/up (a Q8_0 down behind it): the plain launch
            case T_Q8_0: native_w11_kernel<T_Q8_0, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); break;
            // (STRATA_PF_FUSED_Q6K=1, gfx12) the Q6_K gate/up (a Q8_0 down behind it): the plain launch
            case T_Q6_K: native_w11_kernel<T_Q6_K, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); break;
            case T_TQK6: native_w11_kernel<T_TQK6, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, g_rot_on ? dm : nullptr); break;
            case T_TQ2_T: native_w11_kernel<T_TQ2_T, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, g_rot_on ? dm : nullptr); break;
            default: STRATA_NW_GU(T_IQ4_XS); break;
        }
#undef STRATA_NW_GU
#define STRATA_NW_D(T) do { if (occ_d) native_w11_kernel<T, false, true, NW_LB><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm); else native_w11_kernel<T, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm); } while (0)
        if (f16 && f16kq::covers_down(g.d_type)) f16kq::down(b, g, tb.ts, tb.tiles, tb.off, n, n_expert, tiles, ha8, dm, s, d16);
        else if ((f16 && f16_d(g.d_type)) || f16_q8d) {
            if (g.d_type == T_Q2_0) STRATA_F16_LAUNCH(T_Q2_0, false, f16::D_NFB, ha8, nullptr, dm);
            else if (g.d_type == T_Q8_0) STRATA_F16_LAUNCH(T_Q8_0, false, f16::D_NFB, ha8, nullptr, dm);
            else if (g.d_type == T_Q5_0) STRATA_F16_LAUNCH(T_Q5_0, false, f16::D_NFB, ha8, nullptr, dm);
            else STRATA_F16_LAUNCH(T_IQ4_NL, false, f16::D_NFB, ha8, nullptr, dm);
        }
        else if (g.d_type == T_TQK7 || g.d_type == T_TQ2_T) {   // (STRATA_PF_FUSED_TQ=1, gfx12; d_covered_on checked it above)
            if (g_rot_on) {   // a folded down: the FP32 H rows of this call's experts (in dm), rotated and quantized into ha, as on CUDA
                const int64_t nw = n * (D_ROWS_K / 128);
                rot_quant_h_kernel<<<blocks(nw, 8), 256, 0, s>>>(dm, tb.off, b.e0, b.e1, n, g_rot_signs, ha8);
            }
            if (g.d_type == T_TQK7) native_w11_kernel<T_TQK7, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm);
            else native_w11_kernel<T_TQ2_T, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm);
        }
        else if (g.d_type == T_Q2_0) STRATA_NW_D(T_Q2_0);
        else if (g.d_type == T_Q5_0) STRATA_NW_D(T_Q5_0);
        else if (g.d_type == T_Q5_1) STRATA_NW_D(T_Q5_1);
        else if (g.d_type == T_Q8_0) STRATA_NW_D(T_Q8_0);
        else STRATA_NW_D(T_IQ4_NL);
#undef STRATA_NW_D
#undef STRATA_F16_LAUNCH
        ck(cudaGetLastError(), "experts_native");
        return;
    }
#endif
    if (d.occ_nt > 0 || d.ws) {   // STRATA_PF_FUSED_NT=256 / _F16 / _WS: the kernels that take a WideBatch
        WideBatch w;
        w.e0 = b.e0;
        w.e1 = b.e1;
        std::copy(b.blob, b.blob + (b.e1 - b.e0), w.blob);
        experts_native(w, g, n_expert, n, scratch, xa, src, ha, dm, stream);
        return;
    }
    // the launch's gate/up kernel is a staging instance: the tile rule then takes 128 rows (the gate/up is about two thirds of the layer; a down that is not staged, Q2_0 or Q8_0 or Q5_0,
    // still ran faster in the 128-row tile with it: IQ3_S / Q2_0 on the RTX A4000, RESULTS-P4-SM.md S1).  Either kernel staged: the blob check.
    const bool staged = sw_on(g.gu_type, true), wide = g.gu_type == T_Q6_K || g.gu_type == T_Q8_0;   // wide: the Q6_K / Q8_0 gate/up class of the sm_86 tile line (40+ SMs)
    const int ww = launch_ww(b, g, n, n_expert, staged, wide, strata::prefill::fused_tile::kq_flash_kind(g.gu_type, g.d_type));   // (the staging check inside: a launch that fails it takes the unstaged 64-row kernel; the last argument: Flash-Next's K-quant pairs)
    // the most 64-row tiles the batch can have (every row in it, plus a partial tile per expert) - the items of the
    // ones inside a larger item end at once
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ);
#if !defined(__HIPCC__)   // (a HIP build returns above; no K-quant instance of native_kernel in its code objects)
    if (g.gu_type == T_Q4_K || g.gu_type == T_Q5_K) {   // STRATA_PF_FUSED_KQ=1 (gu_covered / d_covered_on checked d.kq above); their own occupancy
        const unsigned k_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq);
        const unsigned k_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq);
        if (g.d_type != T_Q5_1 && !(g.d_type == T_Q8_0 && d.kq8)) {
            std::fprintf(stderr, "prefill fused experts (native): K-quant gate/up with down type %d is not covered here\n", g.d_type);
            std::exit(1);
        }
        if (g.gu_type == T_Q4_K) launch<T_Q4_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
        else launch<T_Q5_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
        if (g.d_type == T_Q8_0) {   // (A2) its own occupancy
            const unsigned k8 = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq8);
            launch<T_Q8_0, false>(ww, k8, b, g, tb, ha, src, nullptr, dm, s);
        } else launch<T_Q5_1, false>(ww, k_d, b, g, tb, ha, src, nullptr, dm, s);
        ck(cudaGetLastError(), "experts_native");
        return;
    }
#endif
    switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_Q2_0: launch<T_Q2_0, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
#if !defined(__HIPCC__)   // (STRATA_PF_FUSED_Q8GU=1, gu_covered checked d.q8gu above; a HIP build returns above and has no such instance)
        case T_Q8_0:
            launch<T_Q8_0, true>(ww, (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_q8gu), b, g, tb,
                                 xa, src, ha, nullptr, s);
            break;
        case T_Q6_K:   // (STRATA_PF_FUSED_Q6K=1, gu_covered checked d.q6k above)
            launch<T_Q6_K, true>(ww, (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_q6k), b, g, tb,
                                 xa, src, ha, nullptr, s);
            break;
        case T_TQK6:   // (STRATA_PF_FUSED_TQ=1, gu_covered checked d.tq above; native_supported pairs it with a TQK7 down)
            launch<T_TQK6, true>(ww, (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_tq_gu), b, g, tb,
                                 xa, src, ha, g_rot_on ? dm : nullptr, s);   // (a folded down: the FP32 H rows into dm; else H as int8, as every format)
            break;
        case T_TQ2_T:   // (the same; with a TQ2_T down)
            launch<T_TQ2_T, true>(ww, (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_tq_gu), b, g, tb,
                                  xa, src, ha, g_rot_on ? dm : nullptr, s);
            break;
#endif
        default: launch<T_IQ4_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
#if !defined(__HIPCC__)   // (a HIP build returns above; no Q5_0 instance of native_kernel in its code objects)
    else if (g.d_type == T_Q5_0) {   // STRATA_PF_FUSED_Q50=1 (d_covered_on checked d.q50 above); its own occupancy
        const unsigned g_q = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_q50);
        launch<T_Q5_0, false>(ww, g_q, b, g, tb, ha, src, nullptr, dm, s);
    } else if (g.d_type == T_Q8_0) {   // STRATA_PF_FUSED_Q8D=1 (d_covered_on checked d.q8d above); A2's instances and occupancy
        const unsigned g8 = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq8);
        launch<T_Q8_0, false>(ww, g8, b, g, tb, ha, src, nullptr, dm, s);
    } else if (g.d_type == T_TQK7 || g.d_type == T_TQ2_T) {   // STRATA_PF_FUSED_TQ=1 (d_covered_on checked d.tq above); their own occupancy
        if (g_rot_on) {   // a folded down (set_down_rotation): the FP32 H rows of this call's experts (in dm), rotated and quantized into ha
            const int64_t nw = n * (D_ROWS_K / 128);
            rot_quant_h_kernel<<<blocks(nw, 8), 256, 0, s>>>(dm, tb.off, b.e0, b.e1, n, g_rot_signs, (uint8_t*) ha);
        }
        const unsigned gt = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_tq_d);
        if (g.d_type == T_TQK7) launch<T_TQK7, false>(ww, gt, b, g, tb, ha, src, nullptr, dm, s);
        else launch<T_TQ2_T, false>(ww, gt, b, g, tb, ha, src, nullptr, dm, s);
    }
#endif
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native");
}

bool native_wide() { const DevInfo& d = dev_info(); return d.occ_nt > 0 || d.ws; }

void experts_native(const WideBatch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
#if !defined(__HIPCC__)
    // (no Q5_0 / K-quant / Q5_1 form of the NT = 256 kernels: native_supported keeps those layers off this path)
    const bool q50 = g.d_type == T_Q5_0 || g.gu_type == T_Q4_K || g.gu_type == T_Q5_K || g.d_type == T_Q5_1 || g.d_type == T_Q8_0 ||
                     is_tq(g.gu_type) || is_tq(g.d_type);
#else
    constexpr bool q50 = false;
#endif
    if ((d.occ_nt <= 0 && !d.ws) || !gu_covered(g.gu_type) || !d_covered_on(g.d_type, d) || q50) {
        std::fprintf(stderr, "prefill fused experts (native, NT = 256): types %d / %d are not covered here\n", g.gu_type,
                     g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    // WW 2 (128 x 64 items), as many blocks as are resident (the WS kernel: one a multiprocessor); the tiles as in
    // experts_native(const Batch&)
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const bool ws = d.ws;
    const int occ = ws ? 1 : d.occ_nt;
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(2)), (int64_t) d.sms * occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(2)), (int64_t) d.sms * occ);
    switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_Q2_0: launch<T_Q2_0, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: launch<T_IQ4_XS, true>(ws, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ws, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ws, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native (NT = 256)");
}
#endif
}  // namespace strata::prefill::fused
