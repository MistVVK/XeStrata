// hy4_gpu.h - one include for the Hy4-Flash GPU code: HIP (gfx1151, the reference lane) or CUDA (sm_80+: sm_86 / sm_89 / sm_120).
//
// The kernels are written once. This header maps the hipXxx runtime names to cudaXxx and wraps the handful of device primitives that differ:
//   - hipLaunchKernelGGL                  -> kernel<<<>>> through hy4::gpu_launch
//   - __shfl_xor(v, o[, w])               -> __shfl_xor_sync(full mask, ...)      (all kernels call it from converged warps)
//   - packed fp16 pairs as uint32 words   -> h2_sub / h2_splat / h2_dot_f32       (HIP: v_dot2_f32_f16, CUDA: half2 + fp32 FMA)
//   - unaligned dword funnel              -> byte_funnel                           (HIP: v_alignbyte, CUDA: __funnelshift_r)
//   - matrix-core tiles                   -> skmma.h (WMMA 16x16x16 on gfx11, mma.sync m16n8k16 on NVIDIA)
// Build: HIP files are compiled with `-x hip`; CUDA with `nvcc -x cu` (sources keep their .hip.cpp names).
#pragma once

// HY4_MMA: matrix-core kernels (prefill GEMM, grouped expert GEMM, flash attention). 1 = built, 0 = the simple kernels are used instead.
#if defined(__CUDACC__) || defined(HY4_CUDA_HOST)
#ifndef HY4_MMA
#define HY4_MMA 1
#endif
#ifndef HY4_CUDA
#define HY4_CUDA 1
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipStream_t cudaStream_t
#define hipEvent_t cudaEvent_t
#define hipDeviceProp_t cudaDeviceProp
#define hipGetErrorString cudaGetErrorString
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemGetInfo cudaMemGetInfo
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipStreamCreate cudaStreamCreate
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipEventCreate cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostMalloc(p, n, f) cudaHostAlloc((p), (n), (f))
#define hipHostFree cudaFreeHost
#define hipSetDevice cudaSetDevice
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostRegister cudaHostRegister
#define hipGetLastError cudaGetLastError
#define hipHostUnregister cudaHostUnregister
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipHostGetDevicePointer cudaHostGetDevicePointer
#define hipDeviceSynchronize cudaDeviceSynchronize

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace hy4 {
inline const char* gpu_arch(const cudaDeviceProp& p) {
    static char b[16];
    snprintf(b, sizeof b, "sm_%d%d", p.major, p.minor);
    return b;
}
}  // namespace hy4
namespace hy4 {
template <class... KP, class... A>
inline void gpu_launch(void (*k)(KP...), dim3 g, dim3 b, size_t shmem, cudaStream_t s, A... a) {
    static const bool dry = getenv("HY4_DRY_LAUNCH") != nullptr;  // host-overhead experiment: skip the kernel launches
    if (dry) return;
    k<<<g, b, shmem, s>>>(a...);
}
}  // namespace hy4
#define hipLaunchKernelGGL(k, g, b, sh, st, ...) ::hy4::gpu_launch((k), (g), (b), (size_t)(sh), (st), ##__VA_ARGS__)

// full-mask warp shuffles (every call site is converged)
template <class T>
__device__ __forceinline__ T hy4_shfl_xor(T v, int o, int w = 32) { return __shfl_xor_sync(0xffffffffu, v, o, w); }
#define __shfl_xor(...) hy4_shfl_xor(__VA_ARGS__)

namespace hy4 {
// ---- packed fp16 pairs carried as uint32 (low half = element 0)
__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 h2_from(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) { return h2_bits(__hsub2(h2_from(a), h2_from(b))); }
__device__ __forceinline__ uint32_t h2_splat(float f) { return h2_bits(__float2half2_rn(f)); }
// c + a.lo*b.lo + a.hi*b.hi, products and sum in fp32
__device__ __forceinline__ float h2_dot_f32(uint32_t a, uint32_t b, float c) {
    const float2 fa = __half22float2(h2_from(a)), fb = __half22float2(h2_from(b));
    return fmaf(fa.y, fb.y, fmaf(fa.x, fb.x, c));
}
__device__ __forceinline__ uint32_t h2_pack(float a, float b) { return h2_bits(__floats2half2_rn(a, b)); }
__device__ __forceinline__ void h2_unpack(uint32_t w, float& a, float& b) {
    const float2 f = __half22float2(h2_from(w));
    a = f.x;
    b = f.y;
}
// the 32 bits that start `ofs` bytes (0..3) into the 64-bit value {hi, lo}
__device__ __forceinline__ uint32_t byte_funnel(uint32_t hi, uint32_t lo, uint32_t ofs) { return __funnelshift_r(lo, hi, ofs * 8u); }
typedef __half half_t;
__device__ __forceinline__ half_t h_from_float(float f) { return __float2half(f); }
#if HY4_MMA
// NVIDIA: mma.sync.m16n8k16 (row.col, f16 inputs, f32 accumulate). A 16x16 output tile is two n8 tiles: fv8.v[s*4 + e], s = n8 sub-tile, e = 0..3.
// The k axis of a 16-chunk is permuted identically for A and B (lane quad c holds the 4 consecutive k 4c..4c+3, loaded as one 8-byte word), so no shuffling is needed.
struct fv8 {
    float v[8];
    __device__ __forceinline__ float& operator[](int i) { return v[i]; }
    __device__ __forceinline__ const float& operator[](int i) const { return v[i]; }
};
__device__ __forceinline__ fv8 fv8_zero() {
    fv8 r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r.v[i] = 0.f;
    return r;
}
__device__ __forceinline__ int acc_m(int r, int lane) { return (lane >> 2) + 8 * ((r & 3) >> 1); }
__device__ __forceinline__ int acc_n(int r, int lane) { return (r >> 2) * 8 + 2 * (lane & 3) + (r & 1); }
__device__ __forceinline__ void mma_16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ const void* shfl_ptr(const void* p, int src) {
    const uint64_t u = (uint64_t)(uintptr_t)p;
    const uint32_t lo = __shfl_sync(0xffffffffu, (uint32_t)u, src), hi = __shfl_sync(0xffffffffu, (uint32_t)(u >> 32), src);
    return (const void*)(uintptr_t)(((uint64_t)hi << 32) | lo);
}
#endif  // HY4_MMA
}  // namespace hy4

#else  // ---------------------------------------------------------------- HIP
#ifndef HY4_MMA
#define HY4_MMA 1
#endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
namespace hy4 {
inline const char* gpu_arch(const hipDeviceProp_t& p) { return p.gcnArchName; }
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) { return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) - __builtin_bit_cast(hv2, b)); }
__device__ __forceinline__ uint32_t h2_splat(float f) {
    const hv2 h = {(_Float16)f, (_Float16)f};
    return __builtin_bit_cast(uint32_t, h);
}
__device__ __forceinline__ float h2_dot_f32(uint32_t a, uint32_t b, float c) { return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false); }
__device__ __forceinline__ uint32_t h2_pack(float a, float b) {
    const hv2 h = {(_Float16)a, (_Float16)b};
    return __builtin_bit_cast(uint32_t, h);
}
__device__ __forceinline__ void h2_unpack(uint32_t w, float& a, float& b) {
    const hv2 h = __builtin_bit_cast(hv2, w);
    a = (float)h[0];
    b = (float)h[1];
}
__device__ __forceinline__ uint32_t byte_funnel(uint32_t hi, uint32_t lo, uint32_t ofs) { return __builtin_amdgcn_alignbyte(hi, lo, ofs); }
typedef _Float16 half_t;
__device__ __forceinline__ half_t h_from_float(float f) { return (half_t)f; }
}  // namespace hy4
#endif
