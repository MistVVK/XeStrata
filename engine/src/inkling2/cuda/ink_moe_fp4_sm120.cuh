// engine/src/inkling2/cuda/ink_moe_fp4_sm120.cuh - Native Blackwell (sm_120a / sm_121a) mxf4nvf4 & mxf8f6f4 MoE interface
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Compiled in its own translation unit (ink_moe_fp4_sm120.cu) with -gencode arch=compute_120a,code=sm_120a
// (and 121a for DGX Spark GB10) so portable builds (sm_70..sm_121) link cleanly and dispatch here at runtime
// whenever compute capability is 12.0 or 12.1.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ink {

struct InkExpertDev {
    uint64_t body_off;  // offset from Section 4 base (4 KiB aligned)
    uint32_t dn_off;    // byte offset of w2 inside expert body
    uint8_t  gu_code;   // InkFmt code id (15=SF4, 16=MX8, 1..8=S_K)
    uint8_t  dn_code;
    uint16_t tier;      // 0 = GPU VRAM, 1 = pinned host, 2 = CPU hybrid (skip on GPU)
};
static_assert(sizeof(InkExpertDev) == 16, "InkExpertDev must be 16 B");

struct InkMoeTables {
    int32_t* cnt;       // [128]
    int32_t* off;       // [129]
    int32_t* active;    // [128]
    int32_t* nact;      // [1]
    int32_t* pairs;     // [P] packed (tok * 8 + slot)
    int32_t* tok_idx;   // [P] source token index for each sorted pair
    int32_t* n_tiles;   // [1] total 64-row tiles across all active experts
    int2*    tiles;     // [max_tiles] {.x = expert_id, .y = pair_row0}
};

namespace sm120 {

/// True when binary includes sm_120a / sm_121a mxf4nvf4 / mxf8f6f4 kernels and active GPU is sm_120 / sm_121.
bool available();

/// Quantize FP32 [rows, cols] (cols % 64 == 0) -> packed SF4 activation slots (40 B per 64 values: 32B E2M1x2 + 4B UE4M3 + 4B g_blk).
/// Note: uses 48-byte stride per 64 values (16-byte aligned for cp.async!).
constexpr int kActFp4SlotBytes = 48;

void quantize_act_fp4(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream);

/// Quantize FP32 [rows, cols] (cols % 32 == 0) -> packed MX8 activation slots (32B E4M3 + 1B UE8M0, stored as separate planes).
void quantize_act_mx8(const float* x, int rows, int cols, uint8_t* xq, uint8_t* xsc, cudaStream_t stream);

/// Blackwell sm_120a native mxf4nvf4 fused Gate+Up+SITU(4.0, 25.0)+FP4-requant (for SF4 experts, L0..19).
/// tile_rows in {16, 32, 64} matches k_ink_moe_bin tile size (16 for c=16..128, 32 for M=129..768, 64 for M>768).
void moe_sf4_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xa, uint8_t* ha, int max_tiles, int tile_rows, cudaStream_t stream);

/// Blackwell sm_120a native mxf4nvf4 Down projection (for SF4 experts, L0..19).
void moe_sf4_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* ha, const float* topk_w, float* y_pair, int max_tiles, int tile_rows, cudaStream_t stream);

/// Blackwell sm_120a native mxf8f6f4 fused Gate+Up+SITU(4.0, 25.0)+MX8-requant (for MX8 experts, L20..23).
void moe_mx8_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc,
                   int max_tiles, int tile_rows, cudaStream_t stream);

/// Blackwell sm_120a native mxf8f6f4 Down projection (for MX8 experts, L20..23).
void moe_mx8_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair,
                   int max_tiles, int tile_rows, cudaStream_t stream);

/// Standalone single-expert GEMM test entry points (used by ink_kernel_test for FP64 parity & TFLOPS benchmarking).
void test_sf4_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc, float w_outer,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream);

void test_mx8_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream);

}  // namespace sm120
}  // namespace strata::ink
