// hy4_cpu_k512.h - Internal interface between hy4_cpu_experts.cpp (AVX2 / AVX-VNNI / NEON / scalar)
// and hy4_cpu_k512.cpp (AVX-512F+BW+VL+VNNI+VBMI, compiled in its own TU with /arch:AVX512 on MSVC so
// MSVC never emits EVEX 0x62 instructions into AVX2 / AVX-VNNI code paths on Alder/Raptor/Arrow Lake).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once
#include <cstdint>
#include "strata/quant/sk_codecs.hpp"

namespace hy4 {

// Activation row (K = 1536 = 6 super-blocks of 256) in transposed 4-plane layout for fast S_K dot kernels.
// Optional high-precision mode (hi): x = da*qa + db*qb, residual stored in xjb/xdb; xs4 holds 0.25*(s_a + s_b).
struct alignas(64) ActRow {
    int8_t xj[6][4][64];
    float xd[6][16];
    float xs4[6][16];
    int8_t xjb[6][4][64];
    float xdb[6][16];
};
static_assert(sizeof(ActRow) == 4224, "ActRow");

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(HY4_NO_FAST)
namespace k512 {

void store_block(ActRow* a, int b, const strata::quant::sk::BlockQ8_1_G64& blk);
void store_block_hi(ActRow* a, int b, const strata::quant::sk::BlockQ8_1_G64& blk, const strata::quant::sk::BlockQ8_1_G64& blk2);
void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi, int pf_rows);

}  // namespace k512
#endif

}  // namespace hy4
