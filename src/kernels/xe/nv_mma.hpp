// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/nv_mma.hpp - NVIDIA's matrix instructions from sm_80 on, in PTX, for the kernels that take them
// themselves (mma_gemm.cpp, flash_attn.cpp, flash_attn_i8.cpp): joint_matrix's CUDA loads
// read a 16 x 16 tile as 32-bit generic loads (four a lane, four times the instructions of ldmatrix's), and it has
// neither mma.sync's m16n8 tiles nor the asynchronous copies into local memory.
//
// mma.sync's tiles, lane l: A (16 x K, row-major) four 32-bit registers, rows l / 4 and 8 + l / 4; B (K x 8,
// column-major) two, column l / 4; the accumulator four, row l / 4 columns 2 (l % 4) and +1, then 8 rows on.  K is 16
// FP16 or BF16 values or 32 int8 ones.  ldmatrix .x4 loads four 8 x 8 tiles of 16-bit values (the same registers as
// one int8 A of 16 x 32), lanes 8 i .. 8 i + 7 giving the rows' addresses of tile i.
#pragma once

#include "device_target.hpp"

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && STRATA_NV_ARCH >= 800
#define STRATA_NV_MMA 1
#else
#define STRATA_NV_MMA 0
#endif

#if STRATA_NV_MMA
namespace strata::kernels::xe::nv {

/// A pointer into local memory as the 32-bit address the shared-memory instructions take
inline uint32_t smem_addr(const void* p) {
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
/// Four 8 x 8 tiles of 16-bit values out of local memory
inline void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

/// 16 bytes global -> local through L2 only (.cg)
inline void cp_async_cg(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
}
/// the same, zeros in place of the bytes where `valid` is false
inline void cp_async_cg(void* smem, const void* gmem, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(smem)), "l"(gmem),
                 "r"(valid ? 16 : 0));
}
inline void cp_async_commit() { asm volatile("cp.async.commit_group;" ::); }
/// until at most N of the committed groups are in flight
template<int N>
inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

/// d += A (16 x 32 int8) . B (32 x 8 int8), int32 sums
inline void mma_s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// c + the sum of a's four bytes as unsigned (dp4a against ones)
inline uint32_t sum_u8x4(uint32_t a, uint32_t c) {
    uint32_t d;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(0x01010101u), "r"(c));
    return d;
}
/// d += A (16 x 32 uint8) . B (32 x 8 int8), int32 sums
inline void mma_u8s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// d += A (16 x 16) . B (16 x 8), FP16 or (BF) BF16 values, FP32 sums
template<bool BF>
inline void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    if constexpr (BF)   // NOLINT(bugprone-branch-clone): the asm strings differ (bf16, f16), which the check does not compare
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace strata::kernels::xe::nv
#endif
