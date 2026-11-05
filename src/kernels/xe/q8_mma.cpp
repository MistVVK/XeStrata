// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/q8_mma.cpp - see include/strata/kernels/q8_mma.hpp.
//
// A work-group of 8 warps computes 128 rows by 128 outputs, each warp 64 x 32 of them as 4 x 4 tiles of mma.sync's
// m16n8k32 (int8, int32 sums): one product a tile is one block of 32, its sum scaled by the row's and the output's
// scale into an FP32 accumulator.  K advances 128 values (4 blocks) at a time, the next step's tiles copied to local
// memory (cp.async) while the warps multiply out of the other buffer.  RTX 4070, the products below: the weights read
// in their 34-byte Q8_0 blocks (unaligned, the loads uncoalesced: L1-bound) and the sums converted in four
// instructions an element, 7.6 ms; the weights rearranged in place once (q8_mma_repack) and the sums scaled in two
// instructions (kMagic), 5.8.
#include "strata/kernels/q8_mma.hpp"
#include "device_target.hpp"
#include "strata/core/per_device.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_q8_mma
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/kernels/matrix_report.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && STRATA_NV_ARCH >= 800
#define STRATA_Q8_MMA 1
inline uint32_t smem_addr(const void* p) {
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
inline void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// N bytes global -> local without registers, the first `bytes` of them read and the rest zeros
template<int N>
inline void cp_async(void* smem, const void* gmem, int bytes = N) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(smem_addr(smem)), "l"(gmem), "n"(N),
                 "r"(bytes));
}
inline void cp_async_commit() { asm volatile("cp.async.commit_group;" ::); }
inline void cp_async_wait0() { asm volatile("cp.async.wait_group 0;" ::); }
// int8 A (16 x 32, the same registers as an FP16 16 x 16 A: ldmatrix's) by B (32 x 8), added to `c0` in every element
inline void mma16832(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c0) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c0));
}
#else
#define STRATA_Q8_MMA 0
#endif

// A block's int32 sum c (|c| <= 32 x 127 x 128 < 2^22) is started from the bits of 2^23 + 2^22 = 12582912, so the
// sum's bits are the float 12582912 + c, exactly: the conversion instruction runs at an eighth of the FP32 rate on
// NVIDIA's SMs.  Scaled by the weights' scale w less 12582912 w in one fma (-12582912 w a column's for the block), then
// by the row's scale into the accumulator in another: two instructions an element.  12582912 w is rounded once, an
// error of at most w (one in c), the same for every row of the column and block; the int8 activations' rounding is
// larger.
[[maybe_unused]] constexpr int kMagicBits = 0x4B400000;
[[maybe_unused]] constexpr float kMagic = 12582912.0f;

// The instruction's tiles, so NVIDIA's alone.  RTX 4070, Qwen-Image-2.1's products (8,228 x 3,072 x 3,072 and the
// MLP's): K 128 a step in one work-group an SM (196 registers, 78 KiB of local memory) took 5.8 ms where K 64 with two
// work-groups an SM (capped at 128 registers) took 7.2: a step's barrier and copies cost more than the second group
// gained.
[[maybe_unused]] constexpr int WM = 128, WN = 128, NSGN = 4, NSG = 8, SG = 32, LANES = NSG * SG;
[[maybe_unused]] constexpr int SGM = 4, SGN = 4;      // a warp's m16 and n8 tiles: 64 x 32
[[maybe_unused]] constexpr int KC = 128, NB = KC / 32;   // K a step
[[maybe_unused]] constexpr int LD = KC + 16;          // local row stride in bytes (ldmatrix's 8 rows in distinct banks)
// two buffers each: the activations and the weights (int8, row stride LD), the activations' scales (FP32) and the
// weights' (FP16), both [row][block]
[[maybe_unused]] constexpr int A_BUF = WM * LD, B_BUF = WN * LD, XS_BUF = WM * NB * 4, WS_BUF = WN * NB * 2;
[[maybe_unused]] constexpr int A_OFF = 0, B_OFF = A_OFF + 2 * A_BUF, XS_OFF = B_OFF + 2 * B_BUF,
                               WS_OFF = XS_OFF + 2 * XS_BUF, LOCAL = WS_OFF + 2 * WS_BUF;

struct Kernel {
    const int8_t* x8;
    const float* xs;
    const uint8_t* w;
    float* y;
    int64_t T, K, ldy, row_bytes, tiles_m;
    sycl::local_accessor<uint8_t, 1> lm;

    void operator()(sycl::nd_item<1> it) const {
#if STRATA_Q8_MMA
        const int64_t g = (int64_t) it.get_group(0);
        // M first: the work-groups that read a tile of W run side by side and find it in L2 (as mma_gemm)
        const int64_t m0 = g % tiles_m * WM, n0 = g / tiles_m * WN;
        const int lid = (int) it.get_local_id(0), lane = lid % SG, sgid = lid / SG;
        const int ms = sgid / NSGN * (SGM * 16), ns = sgid % NSGN * (SGN * 8);
        uint8_t* base = &lm[0];
        const int64_t kb = K / 32, steps = K / KC;
        // the step's tiles, 16 bytes a copy (rows past T zeros), and the scales a row's NB at once
        auto issue = [&](int64_t s, int buf) {
            const int64_t k0 = s * KC;
            uint8_t* al = base + A_OFF + (size_t) buf * A_BUF;
            uint8_t* bl = base + B_OFF + (size_t) buf * B_BUF;
#pragma unroll
            for (int v = lid; v < WM * (KC / 16); v += LANES) {
                const int r = v / (KC / 16), c = v % (KC / 16) * 16;
                const int64_t row = m0 + r;
                cp_async<16>(al + (size_t) r * LD + c, x8 + (row < T ? row : 0) * K + k0 + c, row < T ? 16 : 0);
            }
#pragma unroll
            for (int v = lid; v < WN * (KC / 16); v += LANES) {
                const int r = v / (KC / 16), c = v % (KC / 16) * 16;
                cp_async<16>(bl + (size_t) r * LD + c, w + (n0 + r) * row_bytes + k0 + c);
            }
            if (lid < WM) {
                const int64_t row = m0 + lid;
                cp_async<NB * 4>(base + XS_OFF + (size_t) buf * XS_BUF + (size_t) lid * NB * 4, xs + (row < T ? row : 0) * kb + k0 / 32,
                                 row < T ? NB * 4 : 0);
            } else if (lid < WM + WN) {
                const int r = lid - WM;
                cp_async<NB * 2>(base + WS_OFF + (size_t) buf * WS_BUF + (size_t) r * NB * 2, w + (n0 + r) * row_bytes + K + k0 / 16);
            }
            cp_async_commit();
        };

        float acc[SGM][SGN][4];
#pragma unroll
        for (int i = 0; i < SGM; ++i)
#pragma unroll
            for (int j = 0; j < SGN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

        const int gr = lane / 4, gc = lane % 4 * 2;
        issue(0, 0);
        for (int64_t s = 0; s < steps; ++s) {
            const int buf = (int) (s & 1);
            // this step's copies landed, and every warp is past the last step, whose buffers the next copies take
            cp_async_wait0();
            sycl::group_barrier(it.get_group());
            if (s + 1 < steps) issue(s + 1, buf ^ 1);
            const uint8_t* al = base + A_OFF + (size_t) buf * A_BUF;
            const uint8_t* bl = base + B_OFF + (size_t) buf * B_BUF;
            const float* xsl = reinterpret_cast<const float*>(base + XS_OFF + (size_t) buf * XS_BUF);
            const sycl::half* wsl = reinterpret_cast<const sycl::half*>(base + WS_OFF + (size_t) buf * WS_BUF);
#pragma unroll 1
            for (int b = 0; b < NB; ++b) {
                uint32_t af[SGM][4], bf[SGN / 2][4];
#pragma unroll
                for (int i = 0; i < SGM; ++i)   // rows lane % 16, bytes 16 (lane / 16) of the block
                    ldsm4(af[i], smem_addr(al + (size_t) ((ms + i * 16 + lane % 16) * LD + b * 32 + lane / 16 * 16)));
#pragma unroll
                for (int j = 0; j < SGN / 2; ++j)   // outputs 8 (lane / 16) + lane % 8, bytes 16 (lane / 8 % 2)
                    ldsm4(bf[j], smem_addr(bl + (size_t) ((ns + j * 16 + lane / 16 * 8 + lane % 8) * LD + b * 32 +
                                                           lane / 8 % 2 * 16)));
                float sx[SGM][2], wf[SGN][2], wm[SGN][2];
#pragma unroll
                for (int i = 0; i < SGM; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) sx[i][h] = xsl[(ms + i * 16 + gr + h * 8) * NB + b];
#pragma unroll
                for (int j = 0; j < SGN; ++j)
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        wf[j][q] = (float) wsl[(ns + j * 8 + gc + q) * NB + b];
                        wm[j][q] = -kMagic * wf[j][q];
                    }
#pragma unroll
                for (int i = 0; i < SGM; ++i)
#pragma unroll
                    for (int j = 0; j < SGN; ++j) {
                        int c[4];
                        mma16832(c, af[i], bf[j / 2][(size_t) (j % 2) * 2], bf[j / 2][(j % 2) * 2 + 1], kMagicBits);
                        // c0, c1: row gr, outputs gc and +1; c2, c3: 8 rows on
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = sycl::fma(sycl::fma(sycl::bit_cast<float>(c[e]), wf[j][e % 2], wm[j][e % 2]),
                                                     sx[i][e / 2], acc[i][j][e]);
                    }
            }
        }
#pragma unroll
        for (int i = 0; i < SGM; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int64_t row = m0 + ms + (int64_t) i * 16 + gr + (int64_t) h * 8;
                if (row >= T) continue;
#pragma unroll
                for (int j = 0; j < SGN; ++j) {
                    float* o = y + row * ldy + n0 + ns + (int64_t) j * 8 + gc;
                    o[0] = acc[i][j][(size_t) 2 * h];
                    o[1] = acc[i][j][2 * h + 1];
                }
            }
#else
        (void) it;
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<32>, syclex::max_work_group_size<LANES>};
    }
};

// a Q8_0 row (K / 32 blocks of an FP16 scale and 32 int8 values) as its K int8 values, then its K / 32 scales: the
// same bytes, one work-group a row through local memory
struct Repack {
    uint8_t* w;
    int64_t K, row_bytes;
    sycl::local_accessor<uint8_t, 1> lm;
    void operator()(sycl::nd_item<1> it) const {
        uint8_t* row = w + (int64_t) it.get_group(0) * row_bytes;
        const int lid = (int) it.get_local_id(0), n = (int) it.get_local_range(0);
        for (int64_t i = lid; i < row_bytes; i += n) lm[i] = row[i];
        sycl::group_barrier(it.get_group());
        for (int64_t i = lid; i < K; i += n) row[i] = lm[i / 32 * 34 + 2 + i % 32];
        for (int64_t i = lid; i < K / 16; i += n) row[K + i] = lm[i / 2 * 34 + i % 2];
    }
};

// the architecture this binary's device code for the GPU was compiled for (STRATA_NV_ARCH: 0 where it is not NVIDIA's)
int device_arch(sycl::queue& q) {
    int* out = sycl::malloc_shared<int>(1, q);
    if (out == nullptr) return 0;
    q.single_task([=] { *out = STRATA_NV_ARCH; }).wait();
    const int a = *out;
    sycl::free(out, q);
    return a;
}

}  // namespace

bool q8_mma_usable(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        if (no_xmx()) return false;
        const sycl::device d = q.get_device();
        if (d.get_backend() != sycl::backend::ext_oneapi_cuda) return false;
        if (d.get_info<sycl::info::device::local_mem_size>() < (size_t) LOCAL ||
            d.get_info<sycl::info::device::max_work_group_size>() < (size_t) LANES)
            return false;
        return device_arch(q) >= 800;
    });
}

void q8_mma_repack(sycl::queue& q, void* w, int64_t N, int64_t K) {
    const int64_t row_bytes = K / 32 * 34;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) row_bytes), h);
        h.parallel_for(sycl::nd_range<1>((size_t) N * 256, 256), Repack{static_cast<uint8_t*>(w), K, row_bytes, lm});
    });
}

sycl::event q8_mma_gemm(sycl::queue& q, const void* xq, const void* w, float* y, int64_t T, int64_t N, int64_t K,
                        int64_t ldy) {
    const int64_t tiles_m = (T + WM - 1) / WM, tiles_n = N / WN;
    const auto* x8 = static_cast<const int8_t*>(xq);
    const auto* xs = reinterpret_cast<const float*>(x8 + T * K);
    const int64_t row_bytes = K / 32 * 34;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) LOCAL), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (tiles_m * tiles_n) * LANES, LANES),
                       Kernel{x8, xs, static_cast<const uint8_t*>(w), y, T, K, ldy, row_bytes, tiles_m, lm});
    });
}

}  // namespace strata::kernels
