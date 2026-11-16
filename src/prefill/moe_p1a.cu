// src/prefill/moe_p1a.cu - see moe_p1a.hpp.  gfx11 device code; the engine takes it on gfx1151 only, behind STRATA_PF_P1A=1.
//
// The routed experts' prompt pass on the F16 matrix cores with F16 activations and H, the weight codes in registers
// (gufo's RoutedPairF16Kernel / RoutedDownF16Kernel), on fused::group's tables and the native blobs:
//   gate/up (Q4_K, Q5_K) + SwiGLU -> H as F16 rows [routed row][n_ff]
//   down (Q5_1, Q8_0) of that H -> the routed rows' outputs as F16 [routed row][n_embd] (what moe_combine16 reads)
// What differs from moe_f16_kq.cu (gufo's older RoutedF16GEMMKernel): the activations are the FP16 rows of the layer's input
// (mixed_h), not int8 per 32; H is F16, not int8; each lane decodes only the K16 half of its row that the WMMA reads from it
// and v_permlanex16 hands over the other half; the codes stay in registers and the activations are the only LDS stage;
// a work item is 128 rows of one expert (two of fused::group's 64-row tiles).  The kernels are the harness's
// (tools/hip/strix_bench/k_moe_p1a.cu) line for line except the work-item decode and the blob / geometry arguments;
// MD / WPE / FL are the harness's knobs, the engine instantiates MD 0, WPE 0.
//
// Adapted from gufo 07e1191f36885c144b34566aac8eaa9777d8fa78: src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
// (RoutedPairF16Kernel, RoutedDownF16Kernel, CodesToHalves, SwapHalves, GatherBit, SpreadHighBits).
// MIT License
// Copyright (c) 2026 gufo contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "moe_p1a.hpp"

#include <hip/hip_fp16.h>

#include "strata/quant/types.hpp"   // the type ids (GGUF's numbers, the same as ggml's): Strata's own table, no ggml.h

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fused::p1a {
namespace {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define P1A_DEVICE 1
#else
#define P1A_DEVICE 0
#endif

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using v8u = __attribute__((__vector_size__(32))) uint32_t;

constexpr uint32_t kHalfMagic = 0x64646464U;   // the high bytes of 1024.0 as F16
constexpr int kQ4K = (int) strata::quant::Type::Q4_K, kQ5K = (int) strata::quant::Type::Q5_K, kQ51 = (int) strata::quant::Type::Q5_1, kQ80 = (int) strata::quant::Type::Q8_0;


#if P1A_DEVICE
__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ uint32_t gather_bit(uint32_t w, int s) { return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U; }
__device__ __forceinline__ uint32_t spread_high_bits(uint32_t bits) { return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U; }
// lane i receives lane i ^ 16's value
__device__ __forceinline__ uint32_t swap_halves(uint32_t v) {
    uint32_t old;
    asm("" : "=v"(old));
    return (uint32_t) __builtin_amdgcn_permlanex16((int) old, (int) v, 0x76543210, 0xfedcba98, false, false);
}
__device__ __forceinline__ void codes_to_halves(uint32_t codes, __half2 magic, __half2 scale2, __half2 bias2, __half2& lo, __half2& hi) {
    const uint32_t p0 = __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
    lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
    hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}
#endif

// rows of expert e: [off[e], off[e + 1]); token of row r: src[r]; the outputs go to row r.
// MD (--set p1a_mode=N, timing only, the output is wrong): 0 the kernel; 1 no WMMA (a cheap op consumes A and B); 2 no dequant (A = the raw
// codes); 3 no global activation loads (zeros staged); 5 no global weight loads (codes from registers).
// WPE (--set p1a_wpe=N): the compiler's waves-per-SIMD floor (VGPR cap 1536 / N); 0 = the compiler's choice.
// FL (bit 0 --set p1a_il=1, bit 1 --set p1a_db=1), both kernels, MD 0 only:
//   IL: token tiles in pairs j, j + 1: both B fragments read, then lo j, lo j+1, hi j, hi j+1 (two independent accumulators back to back
//       instead of the dependent lo -> hi pair on one); an odd last live tile also computes its (zero, unused) neighbour.
//   DB: two activation stage buffers in LDS, one barrier per stage instead of two (the stage written next was last read two stages ago,
//       behind the barrier after this stage's commit). No +1 padding: column XOR (row << 2) instead, so 2 x 16 KB = 32 KB a block.
template <int KS, bool DB>
__device__ __forceinline__ int act_slot(int r, int c) { return r * KS + (DB ? (c ^ ((r & 7) << 2)) : c); }
template <int WT, int MD = 0, int WPE = 0, int FL = 0>
__launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : 1))) __global__ void pair_kernel(Batch b, NativeGeom geo, const int32_t* __restrict__ ts, const int2* __restrict__ tiles, const __half* __restrict__ x,
                                                   const int32_t* __restrict__ off, const int32_t* __restrict__ src, __half* __restrict__ out,
                                                   int m, int k) {
#if P1A_DEVICE
    constexpr int BN = 128, BK = 2, kTokTiles = BN / 16, kRows = 64;
    constexpr bool IL = (FL & 1) != 0 && MD == 0, DB = (FL & 2) != 0;
    constexpr bool Q5K = WT == kQ5K;
    constexpr int kBlockBytes = Q5K ? 176 : 144;
    constexpr int kActStride = DB ? BN : BN + 1, kActBuf = BK * 4 * kActStride, kActBytes = (DB ? 2 : 1) * kActBuf * 16;
    constexpr int kPairStride = kRows + 2, kEpilogueBytes = BN * kPairStride * 2;
    constexpr int kLdsBytes = kActBytes > kEpilogueBytes ? kActBytes : kEpilogueBytes;
    __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
    auto* s_act = reinterpret_cast<uint4*>(lds);

    // the work item: fused::group's 64-row tile y of this batch and the next one of the same expert (a 128-row item);
    // the second tile of a pair has no block of its own
    const int t0 = ts[b.e0], w = (int) blockIdx.y;
    if (w >= ts[b.e1] - t0) return;
    const int2 tl = tiles[t0 + w];
    const int expert = tl.x;
    if ((t0 + w - ts[expert]) & 1) return;
    const int t_local = 0;
    const int bucket_begin = tl.y;
    const int bucket_rows = min(BN, off[expert + 1] - bucket_begin);
    const int live_tok_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
    const int num_sb = k / 256;

    const int tid = (int) threadIdx.x, wave_id = tid >> 5, lane_id = tid & 31, sub_lane = lane_id & 15, half_id = lane_id >> 4;
    const int r_block = (int) blockIdx.x * kRows;
    const int row = r_block + wave_id * 8 + (sub_lane % 8);   // tile rows 0-7 gate, 8-15 the matching up rows
    const bool row_live = row < m;
    const int f_c = half_id ^ (sub_lane & 1);
    const uint8_t* w_row = b.blob[expert - b.e0] + (sub_lane < 8 ? 0 : geo.up_off) + (size_t) (row_live ? row : m - 1) * geo.gu_row;

    constexpr int kActFetch = BN * BK * 4 / 256;
    const __half* a_src[kActFetch];
    int a_slot[kActFetch];
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
        const int chunk = tid + i * 256, t = chunk / (BK * 4), sub = chunk % (BK * 4), c_row = t_local + t;
        const int32_t s = c_row < bucket_rows ? src[bucket_begin + c_row] : -1;
        a_src[i] = s >= 0 && MD != 3 ? x + (size_t) s * k + sub * 8 : nullptr;
        a_slot[i] = act_slot<kActStride, DB>(sub, t);
    }
    uint4 raw_header, raw_codes[4], raw_qh[2];
    uint4 header = make_uint4(0u, 0u, 0u, 0u), codes[4], qh[2];
    uint4 a_data[kActFetch];
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) a_data[i] = make_uint4(0u, 0u, 0u, 0u);

    const auto fetch_stage = [&]<int G>(int sb) {
        if constexpr (G == 0 && MD == 5) {
            raw_header = make_uint4(0x3C003C00U ^ (uint32_t) sb, (uint32_t) tid, (uint32_t) sb * 7U, (uint32_t) row);
            if constexpr (Q5K) { raw_qh[0] = make_uint4((uint32_t) sb, 1U, 2U, 3U); raw_qh[1] = make_uint4((uint32_t) tid, 5U, 6U, 7U); }
#pragma unroll
            for (int g = 0; g < 4; ++g) raw_codes[g] = make_uint4((uint32_t) (sb * g), (uint32_t) tid, (uint32_t) g, (uint32_t) (sb ^ tid));
        } else if constexpr (G == 0) {
            const auto* chunks = reinterpret_cast<const uint4*>(w_row + sb * kBlockBytes);
            constexpr int kCodeChunk = Q5K ? 3 : 1;
            raw_header = chunks[0];
            if constexpr (Q5K) { raw_qh[0] = chunks[1]; raw_qh[1] = chunks[2]; }
#pragma unroll
            for (int g = 0; g < 4; ++g) raw_codes[g] = chunks[kCodeChunk + g * 2 + f_c];
        }
        const int kb0 = sb * 8 + G * 2;
#pragma unroll
        for (int i = 0; i < kActFetch; ++i)
            if (a_src[i] != nullptr) a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32);
    };
    uint32_t own_scale = 0, own_high = 0;
    const auto commit_stage = [&]<int G>() {
        if constexpr (G == 0) {
            header = raw_header;
#pragma unroll
            for (int g = 0; g < 4; ++g) codes[g] = raw_codes[g];
            if constexpr (Q5K) { qh[0] = raw_qh[0]; qh[1] = raw_qh[1]; }
        }
        const int sb32 = 2 * G + f_c;
        const unsigned shift = 8U * ((unsigned) sb32 & 3U);
        const uint32_t scale_byte = header.y >> shift, min_byte = header.z >> shift;
        uint32_t sc, mn;
        if (sb32 < 4) {
            sc = scale_byte & 0x3FU;
            mn = min_byte & 0x3FU;
        } else {
            const uint32_t packed = header.w >> shift;
            sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
            mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, header.x);
        const float scale = row_live ? __low2float(dm) * (float) sc : 0.0F;
        const float offset = row_live ? __high2float(dm) * (float) mn : 0.0F;
        own_scale = __builtin_bit_cast(uint32_t, __floats2half2_rn(scale, -offset));
        if constexpr (Q5K) {
            const uint32_t q[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w, qh[1].x, qh[1].y, qh[1].z, qh[1].w};
            uint32_t high = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) high |= gather_bit(q[i], sb32) << (4 * i);
            own_high = high;
        }
        uint4* buf = s_act + (DB ? (G & 1) * kActBuf : 0);
#pragma unroll
        for (int i = 0; i < kActFetch; ++i) buf[a_slot[i]] = a_data[i];
    };
    const auto load_b = [&](const uint4* buf, int kb, int j, v16h& b_lo, v16h& b_hi) {
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = buf[act_slot<kActStride, DB>(kb * 4 + q, j * 16 + sub_lane)];
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
    };
    v8f acc[kTokTiles];
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const __half2 magic = __floats2half2_rn(-1024.0F, -1024.0F);
    const auto compute_stage = [&]<int G>() {
        const uint4 c = codes[G];
        const uint32_t words[4] = {c.x, c.y, c.z, c.w};
        const uint32_t peer_scale = swap_halves(own_scale);
        const uint32_t peer_high = Q5K ? swap_halves(own_high) : 0U;
#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            const __half2 sb = __builtin_bit_cast(__half2, f_c == kb ? own_scale : peer_scale);
            const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
            const uint32_t high = Q5K ? ((f_c == kb ? own_high : peer_high) >> (16U * f_c)) : 0U;
            __half2 h[8];
            v8u own, peer;
            if constexpr (MD == 2) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    own[2 * i] = words[i] ^ (uint32_t) kb ^ __builtin_bit_cast(uint32_t, sb);
                    own[2 * i + 1] = words[i] + high;
                }
                peer = own;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t nib = (words[i] >> (4U * kb)) & 0x0F0F0F0FU;
                    if constexpr (Q5K) nib |= spread_high_bits((high >> (4 * i)) & 0xFU);
                    codes_to_halves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
                }
                __builtin_memcpy(&own, &h[0], 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) peer[i] = swap_halves(own[i]);
            }
            const v16h a_lo = __builtin_bit_cast(v16h, own), a_hi = __builtin_bit_cast(v16h, peer);
            const uint4* buf = s_act + (DB ? (G & 1) * kActBuf : 0);
            if constexpr (IL) {
#pragma unroll
                for (int j = 0; j < kTokTiles; j += 2) {
                    asm volatile("" ::: "memory");   // two token tiles' fragments live at a time
                    if (j >= live_tok_tiles) continue;
                    v16h b0_lo, b0_hi, b1_lo, b1_hi;
                    load_b(buf, kb, j, b0_lo, b0_hi);
                    load_b(buf, kb, j + 1, b1_lo, b1_hi);
                    acc[j] = wmma(a_lo, b0_lo, acc[j]);
                    acc[j + 1] = wmma(a_lo, b1_lo, acc[j + 1]);
                    acc[j] = wmma(a_hi, b0_hi, acc[j]);
                    acc[j + 1] = wmma(a_hi, b1_hi, acc[j + 1]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < kTokTiles; ++j) {
                    asm volatile("" ::: "memory");   // one token tile's fragments live at a time (hoisting all eight spills)
                    if (j >= live_tok_tiles) continue;
                    v16h b_lo, b_hi;
                    load_b(buf, kb, j, b_lo, b_hi);
                    if constexpr (MD == 1) {
                        const v8u bl = __builtin_bit_cast(v8u, b_lo), bh = __builtin_bit_cast(v8u, b_hi);
                        acc[j][0] += __uint_as_float((own[j & 7] ^ peer[(j + 1) & 7] ^ bl[0] ^ bl[5] ^ bh[2] ^ bh[7]) & 0x3F7FFFFFU);
                    } else {
                        acc[j] = wmma(a_lo, b_lo, acc[j]);
                        acc[j] = wmma(a_hi, b_hi, acc[j]);
                    }
                }
            }
        }
    };
    const auto run_stage = [&]<int G>(int sb) {
        commit_stage.template operator()<G>();
        __syncthreads();
        if constexpr (G < 3) fetch_stage.template operator()<G + 1>(sb);
        else if (sb + 1 < num_sb) fetch_stage.template operator()<0>(sb + 1);
        compute_stage.template operator()<G>();
        if constexpr (!DB) __syncthreads();
    };
    fetch_stage.template operator()<0>(0);
    for (int sb = 0; sb < num_sb; ++sb) {
        run_stage.template operator()<0>(sb);
        run_stage.template operator()<1>(sb);
        run_stage.template operator()<2>(sb);
        run_stage.template operator()<3>(sb);
    }
    if constexpr (DB) __syncthreads();   // the epilogue's scratch overlays the stage buffers
    // lane (token n, rows 2l + half) holds gate rows in acc[j][l] and the matching up rows in acc[j][l + 4]
    __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
        if (j >= live_tok_tiles) continue;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            float product = acc[j][l + 4] * acc[j][l];
            asm volatile("" : "+v"(product));
            float value = product * (1.0f / (1.0f + __expf(-acc[j][l])));
            asm volatile("" : "+v"(value));
            scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l + half_id] = __float2half(value);
        }
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < BN * kRows / (2 * 256); ++unit) {
        const int flat = (unit * 256 + tid) * 2, t = t_local + flat / kRows, r = r_block + flat % kRows;
        if (t < bucket_rows && r < m) {
            const size_t o = (size_t) (bucket_begin + t) * m + r;
            const int s = (flat / kRows) * kPairStride + flat % kRows;
            *reinterpret_cast<__half2*>(out + o) = *reinterpret_cast<const __half2*>(scratch + s);
        }
    }
#else
    (void) b; (void) geo; (void) ts; (void) x; (void) tiles; (void) off; (void) src; (void) out; (void) m; (void) k;
    __builtin_trap();
#endif
}

// down: x = H rows [routed row][k]; the outputs to the same rows, F16 [routed row][m]
// MD, WPE, FL as for pair_kernel; BKT (--set p1a_dbk=4): K blocks of 32 a stage
template <int WT, int MD = 0, int WPE = 0, int BKT = 2, int FL = 0>
__launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : 1))) __global__ void down_kernel(Batch b, NativeGeom geo, const int32_t* __restrict__ ts, const int2* __restrict__ tiles, const __half* __restrict__ x,
                                                   const int32_t* __restrict__ off, __half* __restrict__ out, int m, int k) {
#if P1A_DEVICE
    constexpr int BN = 128, BK = BKT, kTokTiles = BN / 16;
    constexpr bool IL = (FL & 1) != 0 && MD == 0, DB = (FL & 2) != 0;
    constexpr bool Q5 = WT == kQ51;
    constexpr int kBlockBytes = Q5 ? 24 : 34;
    constexpr int kActStride = DB ? BN : BN + 1, kActBuf = BK * 4 * kActStride, kActBytes = (DB ? 2 : 1) * kActBuf * 16;
    constexpr int kOutStride = 128 + 8, kEpiTokens = 64, kEpiBytes = kEpiTokens * kOutStride * 2;
    constexpr int kLdsBytes = kActBytes > kEpiBytes ? kActBytes : kEpiBytes;
    __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
    auto* s_act = reinterpret_cast<uint4*>(lds);

    // the work item: fused::group's 64-row tile y of this batch and the next one of the same expert (a 128-row item);
    // the second tile of a pair has no block of its own
    const int t0 = ts[b.e0], w = (int) blockIdx.y;
    if (w >= ts[b.e1] - t0) return;
    const int2 tl = tiles[t0 + w];
    const int expert = tl.x;
    if ((t0 + w - ts[expert]) & 1) return;
    const int t_local = 0;
    const int bucket_begin = tl.y;
    const int bucket_rows = min(BN, off[expert + 1] - bucket_begin);
    const int live_tok_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
    const int num_kb = k / 32;

    const int tid = (int) threadIdx.x, wave_id = tid >> 5, lane_id = tid & 31, sub_lane = lane_id & 15, half_id = lane_id >> 4;
    const int r_block = (int) blockIdx.x * 128;
    const int f_c = half_id ^ (sub_lane & 1);
    const int row = r_block + wave_id * 16 + sub_lane;
    const bool row_live = row < m;
    const uint8_t* w_row = b.blob[expert - b.e0] + geo.down_off + (size_t) (row_live ? row : m - 1) * geo.d_row;

    constexpr int kActFetch = BN * BK * 4 / 256;
    const __half* a_src[kActFetch];
    int a_slot[kActFetch];
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
        const int chunk = tid + i * 256, t = chunk / (BK * 4), sub = chunk % (BK * 4), c_row = t_local + t;
        a_src[i] = c_row < bucket_rows && MD != 3 ? x + (size_t) (bucket_begin + c_row) * k + sub * 8 : nullptr;
        a_slot[i] = act_slot<kActStride, DB>(sub, t);
    }
    const auto load_b = [&](const uint4* buf, int kb, int j, v16h& b_lo, v16h& b_hi) {
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = buf[act_slot<kActStride, DB>(kb * 4 + q, j * 16 + sub_lane)];
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
    };
    uint4 w_codes[2][BK];
    uint32_t w_head[2][BK], w_high[2][BK];
    uint4 a_data[kActFetch];
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) a_data[i] = make_uint4(0u, 0u, 0u, 0u);
    const auto fetch_stage = [&]<int S>(int kb0) {
#pragma unroll
        for (int b = 0; b < BK; ++b) {
            const uint8_t* blk = w_row + (kb0 + b) * kBlockBytes;
            if constexpr (MD == 5) {
                w_head[S][b] = 0x3C003C00U ^ (uint32_t) kb0;
                w_high[S][b] = (uint32_t) (tid + b);
                w_codes[S][b] = make_uint4((uint32_t) kb0, (uint32_t) tid, (uint32_t) b, (uint32_t) row);
            } else if constexpr (Q5) {
                const uint2 head = *reinterpret_cast<const uint2*>(blk);
                w_head[S][b] = head.x;
                w_high[S][b] = head.y;
                w_codes[S][b] = *reinterpret_cast<const uint4*>(blk + 8);
            } else {
                w_head[S][b] = *reinterpret_cast<const uint16_t*>(blk);
                __builtin_memcpy(&w_codes[S][b], blk + 2 + 16 * f_c, 16);
            }
        }
#pragma unroll
        for (int i = 0; i < kActFetch; ++i)
            if (a_src[i] != nullptr) a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32);
    };
    v8f acc[kTokTiles];
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    constexpr float kMagic = Q5 ? -1024.0F : -1152.0F;
    const __half2 magic = __floats2half2_rn(kMagic, kMagic);
    const auto compute_stage = [&]<int S>() {
#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            const uint32_t scale_bias = Q5 ? w_head[S][kb] : (row_live ? w_head[S][kb] : 0U);
            const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
            const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
            const uint4 c = w_codes[S][kb];
            const uint32_t words[4] = {c.x, c.y, c.z, c.w};
            __half2 h[8];
            v8u own, peer;
            if constexpr (MD == 2) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    own[2 * i] = words[i] ^ scale_bias;
                    own[2 * i + 1] = words[i] + w_high[S][kb];
                }
                peer = own;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t nib;
                    if constexpr (Q5) nib = ((words[i] >> (4U * f_c)) & 0x0F0F0F0FU) | spread_high_bits((w_high[S][kb] >> (16 * f_c + 4 * i)) & 0xFU);
                    else nib = words[i] ^ 0x80808080U;
                    codes_to_halves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
                }
                __builtin_memcpy(&own, &h[0], 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) peer[i] = swap_halves(own[i]);
            }
            const v16h a_lo = __builtin_bit_cast(v16h, own), a_hi = __builtin_bit_cast(v16h, peer);
            const uint4* buf = s_act + (DB ? S * kActBuf : 0);
            if constexpr (IL) {
#pragma unroll
                for (int j = 0; j < kTokTiles; j += 2) {
                    asm volatile("" ::: "memory");
                    if (j >= live_tok_tiles) continue;
                    v16h b0_lo, b0_hi, b1_lo, b1_hi;
                    load_b(buf, kb, j, b0_lo, b0_hi);
                    load_b(buf, kb, j + 1, b1_lo, b1_hi);
                    acc[j] = wmma(a_lo, b0_lo, acc[j]);
                    acc[j + 1] = wmma(a_lo, b1_lo, acc[j + 1]);
                    acc[j] = wmma(a_hi, b0_hi, acc[j]);
                    acc[j + 1] = wmma(a_hi, b1_hi, acc[j + 1]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < kTokTiles; ++j) {
                    asm volatile("" ::: "memory");
                    if (j >= live_tok_tiles) continue;
                    v16h b_lo, b_hi;
                    load_b(buf, kb, j, b_lo, b_hi);
                    if constexpr (MD == 1) {
                        const v8u bl = __builtin_bit_cast(v8u, b_lo), bh = __builtin_bit_cast(v8u, b_hi);
                        acc[j][0] += __uint_as_float((own[j & 7] ^ peer[(j + 1) & 7] ^ bl[0] ^ bl[5] ^ bh[2] ^ bh[7]) & 0x3F7FFFFFU);
                    } else {
                        acc[j] = wmma(a_lo, b_lo, acc[j]);
                        acc[j] = wmma(a_hi, b_hi, acc[j]);
                    }
                }
            }
        }
    };
    const auto run_stage = [&]<int S>(int kb0) {
        uint4* wbuf = s_act + (DB ? S * kActBuf : 0);
#pragma unroll
        for (int i = 0; i < kActFetch; ++i) wbuf[a_slot[i]] = a_data[i];
        __syncthreads();
        if (kb0 + BK < num_kb) fetch_stage.template operator()<1 - S>(kb0 + BK);
        compute_stage.template operator()<S>();
        if constexpr (!DB) __syncthreads();
    };
    fetch_stage.template operator()<0>(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * BK) {
        run_stage.template operator()<0>(kb0);
        if (kb0 + BK < num_kb) run_stage.template operator()<1>(kb0 + BK);
    }
    if constexpr (DB) __syncthreads();   // the epilogue's scratch overlays the stage buffers
    __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
    for (int part = 0; part < BN / kEpiTokens; ++part) {
        if (part * (kEpiTokens / 16) >= live_tok_tiles) break;
        if (part > 0) __syncthreads();
#pragma unroll
        for (int jj = 0; jj < kEpiTokens / 16; ++jj) {
            const int j = part * (kEpiTokens / 16) + jj;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                scratch[(jj * 16 + sub_lane) * kOutStride + wave_id * 16 + 2 * i + half_id] =
                    __float2half(fminf(fmaxf(acc[j][i], -65504.0f), 65504.0f));
        }
        __syncthreads();
#pragma unroll
        for (int unit = 0; unit < kEpiTokens * 16 / 256; ++unit) {
            const int flat = unit * 256 + tid, tr = flat / 16, chunk = flat % 16;
            const int t = t_local + part * kEpiTokens + tr, r = r_block + chunk * 8;
            if (t >= bucket_rows || r >= m) continue;
            *reinterpret_cast<uint4*>(out + (size_t) (bucket_begin + t) * m + r) =
                *reinterpret_cast<const uint4*>(scratch + tr * kOutStride + chunk * 8);
        }
    }
#else
    (void) b; (void) geo; (void) ts; (void) x; (void) tiles; (void) off; (void) out; (void) m; (void) k;
    __builtin_trap();
#endif
}

}  // namespace

static_assert(kMoeFF % 64 == 0 && kMoeEmbd % 256 == 0 && kMoeEmbd % 128 == 0 && kMoeFF % 32 == 0, "Path A's tile and stage sizes");

bool takes(int gu_type, int d_type, bool l2) {
    if (gu_type == kQ4K && d_type == kQ51) return true;
    return l2 && (gu_type == kQ4K || gu_type == kQ5K) && (d_type == kQ51 || d_type == kQ80);
}

void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t work_bound,
             const void* x16, const int32_t* src, void* h16, cudaStream_t s) {
    const dim3 grid((unsigned) (kMoeFF / 64), (unsigned) std::max<int64_t>(1, work_bound));
    if (g.gu_type == kQ5K)
        pair_kernel<kQ5K><<<grid, 256, 0, s>>>(b, g, ts, tiles, (const __half*) x16, off, src, (__half*) h16, kMoeFF, kMoeEmbd);
    else
        pair_kernel<kQ4K><<<grid, 256, 0, s>>>(b, g, ts, tiles, (const __half*) x16, off, src, (__half*) h16, kMoeFF, kMoeEmbd);
}

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t work_bound,
          const void* h16, void* d16, cudaStream_t s) {
    const dim3 grid((unsigned) ((kMoeEmbd + 127) / 128), (unsigned) std::max<int64_t>(1, work_bound));
    if (g.d_type == kQ80)
        down_kernel<kQ80><<<grid, 256, 0, s>>>(b, g, ts, tiles, (const __half*) h16, off, (__half*) d16, kMoeEmbd, kMoeFF);
    else
        down_kernel<kQ51><<<grid, 256, 0, s>>>(b, g, ts, tiles, (const __half*) h16, off, (__half*) d16, kMoeEmbd, kMoeFF);
}

}  // namespace strata::prefill::fused::p1a
