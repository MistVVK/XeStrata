// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cuda/i8_gemm_cutlass.cu - i8_gemm.hpp's product on NVIDIA GPUs from sm_80 on, CUTLASS's int8 tensor-op
// GEMM (third_party/main/cutlass, BSD-3-Clause) with an epilogue visitor tree that scales the int32 sums by X's row
// scales and W's, adds the bias and writes BF16.  Compiled twice by intel/llvm's clang as CUDA, never by nvcc: the
// device side alone into each NVIDIA architecture's code (the kernel below, embedded in the program by CMake), the
// host side alone into the parameters' block and the launch, through the CUDA driver (libcuda, NVIDIA's driver, opened
// at the first product) on the SYCL queue's CUDA stream - so no CUDA runtime is linked and nothing registers kernels.
// The two compiles see the same types: the block the host fills is the kernel's argument byte for byte.

#if defined(__clang__) && !defined(__NVCC__) && !defined(__NV_ATOMIC_RELAXED)
// nvcc's atomic builtins under their names, as clang's own: CUTLASS 4.8.1's sub-byte references (which these products
// do not use) call them when told a CUDA of 12.8 or later (__CUDACC_VER_*, given from the toolkit's cuda.h)
#define __NV_ATOMIC_RELAXED __ATOMIC_RELAXED
#define __nv_atomic_load_n(p, order) __atomic_load_n((p), (order))
#endif
#include <cutlass/cutlass.h>
#include <cutlass/detail/helper_macros.hpp>
#if defined(_MSC_VER)
// clang for Windows (MSVC's ABI) takes no MSVC attribute on a lambda, which CUTLASS gives its lambdas there: GNU's
#undef CUTLASS_LAMBDA_FUNC_INLINE
#define CUTLASS_LAMBDA_FUNC_INLINE __attribute__((always_inline))
#endif
#include <cutlass/gemm/kernel/default_gemm_universal.h>   // ahead of the visitors, which use its names
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include "i8_gemm_cutlass.hpp"

namespace {

using namespace cute;
namespace tb = cutlass::epilogue::threadblock;

// The shapes, by the product's rows: 128 x 256 tiles, three stages (72 KiB of local memory a work-group) for most;
// 128 x 128 tiles, four stages (64 KiB) below 2048 rows, where the larger tiles leave more of the GPU idle.  Warps of
// 64 x 64, K 64 a step; sm_80 on give a work-group 99 KiB or more.  On the RTX 4070 (2026-11-06), against cuBLAS's
// int8 product (int32 sums out, twice the bytes for the readers): 4115 x 3072 x 3072 in 0.388 ms (0.41), x 16384 x
// 3072 in 1.92 (1.99), x 3072 x 16384 in 1.80 (1.84), 8230 x 3072 x 16384 in 3.44 (3.70); 1024 x 3072 x 3072 with the
// smaller tiles in 0.109 (0.152; 0.119 with the larger).  256 x 128 tiles were as fast or slower.  The work-groups go
// eight tiles of rows at a time across the columns (the swizzle), so that neighbours share W's tiles in the L2 cache:
// in X's order 8230 x 3072 x 16384 took 6.0 ms against cuBLAS's 3.6.
using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
using Mma = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kAlignC = 8;   // 16 bytes of BF16
constexpr int kSmallRows = 2048;

template <int TM, int TN, int Stages> struct Shape {
    using Tile = cutlass::gemm::GemmShape<TM, TN, 64>;
    using Map = tb::OutputTileThreadLayout<Tile, Warp, cutlass::bfloat16_t, kAlignC, 1>;
    using Accum = tb::VisitorAccFetch;
    using RowScale = tb::VisitorColBroadcast<Map, float, Stride<_1, _0, int32_t>>;   // sx[t]: one a row of Y
    using ColScale = tb::VisitorRowBroadcast<Map, float, Stride<_0, _1, int32_t>>;   // sw[n] and bias[n]: a column
    using Mul = tb::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Fma = tb::VisitorCompute<cutlass::multiply_add, cutlass::bfloat16_t, float,
                                   cutlass::FloatRoundStyle::round_to_nearest>;
    using Scaled = tb::Sm80EVT<Mul, Accum, RowScale>;
    using Out = tb::Sm80EVT<Fma, Scaled, ColScale, ColScale>;
    using Store = tb::VisitorAuxStore<Map, cutlass::bfloat16_t, cutlass::FloatRoundStyle::round_to_nearest,
                                      Stride<int64_t, _1, int64_t>>;
    using Evt = tb::Sm80EVT<Store, Out>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,      // X [T][K]
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,   // W [N][K]: W^T column-major
        cutlass::bfloat16_t, cutlass::layout::RowMajor, kAlignC, int32_t, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, Tile, Warp, Mma, Evt, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        Stages, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
};
using Large = Shape<128, 256, 3>;   // kernel 0
using Small = Shape<128, 128, 4>;   // kernel 1

}  // namespace

#if defined(__CUDA_ARCH__)

namespace {
template <class K> __device__ void run(const typename K::Params& params) {
    extern __shared__ int smem[];
    K op;
    op(params, *reinterpret_cast<typename K::SharedStorage*>(smem));
}
}  // namespace

// The device compile's sizes of each kernel's local memory and parameters, which the host side checks against its own
// before it takes the kernels: an image built from another version of this file (a stale object: a compiler cache once
// served one) gets the wrong parameters, and writes past the local memory it was launched with
extern "C" __global__ void strata_i8_gemm_sizes(unsigned* out) {
    out[0] = (unsigned) sizeof(typename Large::Kernel::SharedStorage);
    out[1] = (unsigned) sizeof(typename Large::Kernel::Params);
    out[2] = (unsigned) sizeof(typename Small::Kernel::SharedStorage);
    out[3] = (unsigned) sizeof(typename Small::Kernel::Params);
}

extern "C" __global__ void __launch_bounds__(Large::Kernel::kThreadCount)
strata_i8_gemm_bf16_0(typename Large::Kernel::Params params) {
    run<typename Large::Kernel>(params);
}

extern "C" __global__ void __launch_bounds__(Small::Kernel::kThreadCount)
strata_i8_gemm_bf16_1(typename Small::Kernel::Params params) {
    run<typename Small::Kernel>(params);
}

#else

namespace strata::kernels::cutlass_i8 {

namespace {
template <class S> bool fill(const Problem& p, Launch& out) {
    using Kernel = typename S::Kernel;
    static_assert(sizeof(typename Kernel::Params) <= sizeof(out.params), "the kernel's parameters fit the block");
    const auto t = (int32_t) p.T, n = (int32_t) p.N;
    typename S::Evt::Arguments evt{
        {
            {
                {},                                                           // Accum
                {p.sx, 1.f, {_1{}, _0{}, t}},                                 // RowScale
                {}                                                            // Mul
            },
            {p.sw, 1.f, {_0{}, _1{}, n}},                                     // ColScale: sw
            {p.bias, 0.f, {_0{}, _1{}, n}},                                   // ColScale: bias (null: 0)
            {}                                                                // Fma
        },
        {reinterpret_cast<cutlass::bfloat16_t*>(p.y), {p.ldy, _1{}, p.ldy * p.T}}   // Store
    };
    const cutlass::gemm::GemmCoord size((int) p.T, (int) p.N, (int) p.K);
    typename Kernel::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, size, 1, evt, p.x, p.w, nullptr, nullptr,
                                    p.T * p.K, p.N * p.K, 0, 0, p.K, p.K, 0, 0);
    if (Kernel::can_implement(args) != cutlass::Status::kSuccess) return false;
    const typename Kernel::Params kp(args, 0, 0);
    const dim3 grid = kp.get_grid_dims();
    out.grid[0] = grid.x;
    out.grid[1] = grid.y;
    out.grid[2] = grid.z;
    out.threads = Kernel::kThreadCount;
    out.smem = (unsigned) sizeof(typename Kernel::SharedStorage);
    out.bytes = sizeof kp;
    __builtin_memcpy(out.params, &kp, sizeof kp);
    return true;
}
}  // namespace

bool params(const Problem& p, Launch& out) {
    // fewer rows than a few tiles' leave much of the GPU idle, where cuBLAS's smaller tiles do better
    if (p.T < kMinRows || p.K % 16 != 0 || p.N % kAlignC != 0 || p.ldy % kAlignC != 0) return false;
    out.kernel = p.T < kSmallRows ? 1 : 0;
    return out.kernel == 1 ? fill<Small>(p, out) : fill<Large>(p, out);
}

void sizes(unsigned* out) {
    out[0] = (unsigned) sizeof(typename Large::Kernel::SharedStorage);
    out[1] = (unsigned) sizeof(typename Large::Kernel::Params);
    out[2] = (unsigned) sizeof(typename Small::Kernel::SharedStorage);
    out[3] = (unsigned) sizeof(typename Small::Kernel::Params);
}

}  // namespace strata::kernels::cutlass_i8

#endif
