// metal/kernels/mm_i8.metal - matmul2d throughput probes for other operand types (tests/mm_i8_bench.cpp):
// C[M][N] = A[M][K] . B[N][K]^T with a typed destination: int8 x int8 -> int32, fp16 x fp16 -> fp16 (and the fp32 reference).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs { int M, N, K; };

template <typename T, typename D, int TM, int TN, int SG>
kernel void mpp_mm_t(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device D* C [[buffer(2)]],
                     constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device D, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mC = tC.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mC);
}

#define MPPT(T, D, M, N, S) template [[host_name("mpp_" #T "_" #D "_" #M "_" #N "_" #S)]] kernel decltype(mpp_mm_t<T, D, M, N, S>) mpp_mm_t<T, D, M, N, S>;
MPPT(char, int, 64, 64, 4) MPPT(char, int, 128, 64, 4) MPPT(char, int, 64, 128, 4)
MPPT(half, half, 64, 64, 4) MPPT(half, half, 128, 64, 4)
MPPT(half, float, 64, 64, 4) MPPT(half, float, 128, 64, 4)
