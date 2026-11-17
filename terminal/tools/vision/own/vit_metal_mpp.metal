// tools/vision/own/vit_metal_mpp.metal - the vision tower's products on the Metal 4 tensor ops (MetalPerformancePrimitives matmul2d): 16-bit operands
// read in place with their row strides, float32 accumulation.  The same contract as vk_gemm in vit_metal.metal (Y[z][r][o] = sum_k X[z][r][k] W[z](o, k)),
// chosen at start only where it compiles and a self-test against vk_gemm passes (vit_metal.mm).  The pattern follows metal/src/engine/msl/ds4_vision_mpp.metal.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct GemmArgs { uint rows, in, out, ldx, ldw, ldy; ulong sx, sw, sy; };

// TI: the operands' type (half or bfloat).  WN: W is [in][out] (o contiguous), else [out][in].  TO: Y's type (float or half).
template <int TM, int TN, int SG, typename TI, bool WN, typename TO>
kernel void vm_gemm(device const TI* X0 [[buffer(0)]], device const TI* W0 [[buffer(1)]], device TO* Y0 [[buffer(2)]], constant GemmArgs& a [[buffer(15)]],
                    uint3 tg [[threadgroup_position_in_grid]]) {
    using E2 = dextents<int32_t, 2>;
    const int M = int(a.rows), N = int(a.out), K = int(a.in);
    device TI* X = (device TI*) (X0 + a.sx * tg.z);
    device TI* W = (device TI*) (W0 + a.sw * tg.z);
    device TO* Y = Y0 + a.sy * tg.z;
    tensor<device TI, E2, tensor_inline> tA(X, E2(K, M), array<int32_t, 2>{1, int(a.ldx)});
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, !WN, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    tensor<device TO, E2, tensor_inline> tY(Y, E2(N, M), array<int32_t, 2>{1, int(a.ldy)});
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    if (WN) {
        tensor<device TI, E2, tensor_inline> tB(W, E2(N, K), array<int32_t, 2>{1, int(a.ldw)});
        auto mB = tB.slice(int(tg.x) * TN, 0);
        op.run(mA, mB, mY);
    } else {
        tensor<device TI, E2, tensor_inline> tB(W, E2(K, N), array<int32_t, 2>{1, int(a.ldw)});
        auto mB = tB.slice(0, int(tg.x) * TN);
        op.run(mA, mB, mY);
    }
}

// weight products (16-bit X and W, float32 Y), 128 or 64 rows a group
template [[host_name("vm_gemm_f16_128")]] kernel decltype(vm_gemm<128, 64, 4, half, false, float>) vm_gemm<128, 64, 4, half, false, float>;
template [[host_name("vm_gemm_f16_64")]] kernel decltype(vm_gemm<64, 64, 4, half, false, float>) vm_gemm<64, 64, 4, half, false, float>;
template [[host_name("vm_gemm_bf16_128")]] kernel decltype(vm_gemm<128, 64, 4, bfloat, false, float>) vm_gemm<128, 64, 4, bfloat, false, float>;
template [[host_name("vm_gemm_bf16_64")]] kernel decltype(vm_gemm<64, 64, 4, bfloat, false, float>) vm_gemm<64, 64, 4, bfloat, false, float>;
// attention: S = Q K^T (F16 out) and context = P V (V o-contiguous, float32 out)
template [[host_name("vm_gemm_s")]] kernel decltype(vm_gemm<64, 64, 4, half, false, half>) vm_gemm<64, 64, 4, half, false, half>;
template [[host_name("vm_gemm_pv")]] kernel decltype(vm_gemm<64, 64, 4, half, true, float>) vm_gemm<64, 64, 4, half, true, float>;
