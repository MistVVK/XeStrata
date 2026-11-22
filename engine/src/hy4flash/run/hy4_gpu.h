// hy4_gpu.h - one include for the Hy4-Flash GPU code: HIP (gfx1151, the reference lane) or CUDA (sm_70+: sm_70 / sm_75 / sm_86 / sm_89 / sm_120 / sm_121).
//
// The kernels are written once. This header maps the hipXxx runtime names to cudaXxx and wraps the handful of device primitives that differ:
//   - hipLaunchKernelGGL                  -> kernel<<<>>> through hy4::gpu_launch
//   - __shfl_xor(v, o[, w])               -> __shfl_xor_sync(full mask, ...)      (all kernels call it from converged warps)
//   - packed fp16 pairs as uint32 words   -> h2_sub / h2_splat / h2_dot_f32       (HIP: v_dot2_f32_f16, CUDA: half2 + fp32 FMA)
//   - unaligned dword funnel              -> byte_funnel                           (HIP: v_alignbyte, CUDA: __funnelshift_r)
//   - matrix-core tiles                   -> skmma.h (WMMA 16x16x16 on gfx11, mma.sync m16n8k16 on NVIDIA)
// Build: HIP files are compiled with `-x hip`; CUDA with `nvcc -x cu` (sources keep their .hip.cpp names).
#pragma once

// HY4_COLD_API: 1 when hy4_cpu_experts has set_cold_experts (SSD-backed cold experts)
#ifndef HY4_COLD_API
#define HY4_COLD_API 1
#endif
// HY4_MMA: matrix-core kernels (prefill GEMM, grouped expert GEMM, flash attention). 1 = built, 0 = the simple kernels are used instead.
#if defined(__CUDACC__) || defined(HY4_CUDA_HOST)
#ifndef HY4_MMA
#define HY4_MMA 1
#endif
#ifndef HY4_CUDA
#define HY4_CUDA 1
#endif
#ifndef HY4_GFX12
#define HY4_GFX12 0
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
#define hipEventDestroy cudaEventDestroy
#define hipEventQuery cudaEventQuery
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
inline bool hy4_is_rdna2() { return false; }
__device__ __forceinline__ void hy4_sync_lds() { __syncthreads(); }
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
__device__ __forceinline__ T hy4_shfl(T v, int src, int w = 32) { return __shfl_sync(0xffffffffu, v, src, w); }
#define __shfl(...) hy4_shfl(__VA_ARGS__)
template <class T>
__device__ __forceinline__ T hy4_shfl_xor(T v, int o, int w = 32) { return __shfl_xor_sync(0xffffffffu, v, o, w); }
#define __shfl_xor(...) hy4_shfl_xor(__VA_ARGS__)

namespace hy4 {
// ---- packed fp16 pairs carried as uint32 (low half = element 0)
__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 h2_from(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) { return h2_bits(__hsub2(h2_from(a), h2_from(b))); }
__device__ __forceinline__ uint32_t h2_mul(uint32_t a, uint32_t b) { return h2_bits(__hmul2(h2_from(a), h2_from(b))); }
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
__device__ __forceinline__ int hy4_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int s = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int8_t xa = (int8_t)((a >> (8 * i)) & 0xFF);
        int8_t xb = (int8_t)((b >> (8 * i)) & 0xFF);
        s += (int)xa * (int)xb;
    }
    return s;
#endif
}
#if HY4_MMA
// NVIDIA: mma.sync.m16n8k16 (sm_80+), 2x mma.sync.m16n8k8 (sm_75 Turing), or warp-shuffle fallback (sm_70 Volta / < sm_75).
// A 16x16 output tile is two n8 tiles: fv8.v[s*4 + e], s = n8 sub-tile, e = 0..3.
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
static __device__ __noinline__ void mma_16816_shfl(float& c0, float& c1, float& c2, float& c3,
                                            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    // sm_70 Volta / < sm_75 warp-shuffle exact fallback (identical fragment and accumulator layout;
    // __noinline__ prevents IR blowup across multi-format WMMA kernel templates).
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, q = lane & 3;
    const int n0 = 2 * q, n1 = 2 * q + 1;
#pragma unroll 1
    for (int kq = 0; kq < 4; ++kq) {
        const int sa = (g << 2) | kq;
        const int sb0 = (n0 << 2) | kq;
        const int sb1 = (n1 << 2) | kq;
        const uint32_t ar00 = __shfl_sync(0xffffffffu, a0, sa);
        const uint32_t ar10 = __shfl_sync(0xffffffffu, a1, sa);
        const uint32_t ar01 = __shfl_sync(0xffffffffu, a2, sa);
        const uint32_t ar11 = __shfl_sync(0xffffffffu, a3, sa);
        const uint32_t bc00 = __shfl_sync(0xffffffffu, b0, sb0);
        const uint32_t bc01 = __shfl_sync(0xffffffffu, b1, sb0);
        const uint32_t bc10 = __shfl_sync(0xffffffffu, b0, sb1);
        const uint32_t bc11 = __shfl_sync(0xffffffffu, b1, sb1);
        c0 = h2_dot_f32(ar00, bc00, h2_dot_f32(ar01, bc01, c0));
        c1 = h2_dot_f32(ar00, bc10, h2_dot_f32(ar01, bc11, c1));
        c2 = h2_dot_f32(ar10, bc00, h2_dot_f32(ar11, bc01, c2));
        c3 = h2_dot_f32(ar10, bc10, h2_dot_f32(ar11, bc11, c3));
    }
}
// Volta sm_70 native HMMA.884: each quad-pair computes 8x8x4 (4 quad-pairs per warp = 1024 FMAs/inst = full 16x16x4 tile).
__device__ __forceinline__ void mma_884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void)d; (void)a0; (void)a1; (void)b0; (void)b1;
#endif
}
// Convert Volta HMMA.884 16x16 quad-pair accumulator d[8] (qp = (lane>>2)&3: qp_m=qp>>1, qp_n=qp&1) to standard fv8 layout ONCE after the K-loop.
__device__ __forceinline__ void sm70_d8_to_fv8(const float (&d)[8], float (&out)[8]) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, q = lane & 3;
    const bool r2 = (g & 2) != 0, c4 = (q & 2) != 0;
    const int base = ((g >> 2) << 4) | ((q & 1) << 1) | (g & 1);
    const int qps[4] = {0, 2, 1, 3};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int src = base | (qps[k] << 2);
        const float t0 = __shfl_sync(0xffffffffu, d[0], src), t1 = __shfl_sync(0xffffffffu, d[1], src);
        const float t2 = __shfl_sync(0xffffffffu, d[2], src), t3 = __shfl_sync(0xffffffffu, d[3], src);
        const float t4 = __shfl_sync(0xffffffffu, d[4], src), t5 = __shfl_sync(0xffffffffu, d[5], src);
        const float t6 = __shfl_sync(0xffffffffu, d[6], src), t7 = __shfl_sync(0xffffffffu, d[7], src);
        const float a0 = r2 ? t2 : t0, a1 = r2 ? t3 : t1;
        const float b0 = r2 ? t6 : t4, b1 = r2 ? t7 : t5;
        out[2 * k]     = c4 ? b0 : a0;
        out[2 * k + 1] = c4 ? b1 : a1;
    }
}
static __device__ __noinline__ void mma_16816_sm70(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    // Map m16n8k16 fragment layout onto 2x Volta HMMA.884 instructions using all 4 quad-pairs (0 shared memory):
    //   qp = (lane >> 2) & 3: qp 0,1 compute M[0..7] (k_half 0,1); qp 2,3 compute M[8..15] (k_half 0,1).
    //   rl = (lane & 3) + ((lane >> 4) & 1) * 4: row/col index 0..7 within each quad-pair.
    const int lane = threadIdx.x & 31;
    const int qp = (lane >> 2) & 3;
    const int rl = (lane & 3) + ((lane >> 4) & 1) * 4;
    const int s0 = (rl << 2) | ((qp & 1) << 1);
    const int s1 = s0 + 1;
    const uint32_t a0_0 = __shfl_sync(0xffffffffu, a0, s0), a1_0 = __shfl_sync(0xffffffffu, a1, s0);
    const uint32_t a0_1 = __shfl_sync(0xffffffffu, a0, s1), a1_1 = __shfl_sync(0xffffffffu, a1, s1);
    const uint32_t a2_0 = __shfl_sync(0xffffffffu, a2, s0), a3_0 = __shfl_sync(0xffffffffu, a3, s0);
    const uint32_t a2_1 = __shfl_sync(0xffffffffu, a2, s1), a3_1 = __shfl_sync(0xffffffffu, a3, s1);
    const uint32_t ra0 = (qp & 2) ? a1_0 : a0_0;
    const uint32_t ra1 = (qp & 2) ? a1_1 : a0_1;
    const uint32_t ra2 = (qp & 2) ? a3_0 : a2_0;
    const uint32_t ra3 = (qp & 2) ? a3_1 : a2_1;
    const uint32_t rb0 = __shfl_sync(0xffffffffu, b0, s0);
    const uint32_t rb1 = __shfl_sync(0xffffffffu, b0, s1);
    const uint32_t rb2 = __shfl_sync(0xffffffffu, b1, s0);
    const uint32_t rb3 = __shfl_sync(0xffffffffu, b1, s1);
    float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    mma_884(d, ra0, ra1, rb0, rb1);
    mma_884(d, ra2, ra3, rb2, rb3);
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] += __shfl_xor_sync(0xffffffffu, d[i], 4);
    // qp 0,2 publish rows r0 (d[0,1,4,5]); qp 1,3 publish rows r0+2 (d[2,3,6,7])
    const float x0 = (qp & 1) ? d[2] : d[0];
    const float x1 = (qp & 1) ? d[3] : d[1];
    const float y0 = (qp & 1) ? d[6] : d[4];
    const float y1 = (qp & 1) ? d[7] : d[5];
    const int g = lane >> 2, q = lane & 3;
    const int src_top = ((g >> 2) << 4) | (((g >> 1) & 1) << 2) | ((q & 1) << 1) | (g & 1);
    const int src_bot = src_top | 8;
    const float t_x0 = __shfl_sync(0xffffffffu, x0, src_top), t_x1 = __shfl_sync(0xffffffffu, x1, src_top);
    const float t_y0 = __shfl_sync(0xffffffffu, y0, src_top), t_y1 = __shfl_sync(0xffffffffu, y1, src_top);
    const float b_x0 = __shfl_sync(0xffffffffu, x0, src_bot), b_x1 = __shfl_sync(0xffffffffu, x1, src_bot);
    const float b_y0 = __shfl_sync(0xffffffffu, y0, src_bot), b_y1 = __shfl_sync(0xffffffffu, y1, src_bot);
    c[0] += (q & 2) ? t_y0 : t_x0;
    c[1] += (q & 2) ? t_y1 : t_x1;
    c[2] += (q & 2) ? b_y0 : b_x0;
    c[3] += (q & 2) ? b_y1 : b_x1;
}
__device__ __forceinline__ void mma_16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(HY4_FORCE_SM75_MMA) && !defined(HY4_FORCE_MMA_SHFL)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && !defined(HY4_FORCE_MMA_SHFL)
    // Turing sm_75: identical lane/accumulator layout to m16n8k16; k-split into two k=8 Tensor Core ops:
    //   call 1 sums k in {4c, 4c+1} across c=0..3; call 2 sums k in {4c+2, 4c+3} across c=0..3.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a2), "r"(a3), "r"(b1));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(HY4_FORCE_MMA_SHFL)
    mma_16816_sm70(c, a0, a1, a2, a3, b0, b1);
#else
    mma_16816_shfl(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
#endif
}
// Native INT8 Tensor Core m16n8k32 (sm_80+: Ampere/Ada/Hopper/Blackwell). A: s8 [16x32] row-major, B: s8 [8x32] row-major (col-major 32x8), C: s32 [16x8].
// Note: all S2..S6 weights q in [0..63] and S8 weights in [-127..127] fit in signed s8.
__device__ __forceinline__ void mma_16832_s8s8(int32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}
// Portable 4-way int8 dot product: hardware dp4a on sm_61+, scalar fallback on sm_60 (P100)
__device__ __forceinline__ int32_t dp4a_s8(uint32_t a, uint32_t b, int32_t c) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
    return __dp4a((int)a, (int)b, c);
#else
    int32_t r = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) r += (int32_t)(int8_t)((a >> (8 * i)) & 0xFFu) * (int32_t)(int8_t)((b >> (8 * i)) & 0xFFu);
    return r;
#endif
}
__device__ __forceinline__ const void* shfl_ptr(const void* p, int src) {
    const uint64_t u = (uint64_t)(uintptr_t)p;
    const uint32_t lo = __shfl_sync(0xffffffffu, (uint32_t)u, src), hi = __shfl_sync(0xffffffffu, (uint32_t)(u >> 32), src);
    return (const void*)(uintptr_t)(((uint64_t)hi << 32) | lo);
}
#endif  // HY4_MMA
}  // namespace hy4

#else  // ---------------------------------------------------------------- HIP
#ifndef HY4_CUDA
#define HY4_CUDA 0
#endif
// RDNA2 (gfx103x) has no WMMA instructions; guard WMMA cleanly and route to custom Wave32 sdot4/fdot2 kernels.
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
#undef HY4_MMA
#define HY4_MMA 0
#ifndef HY4_RDNA2
#define HY4_RDNA2 1
#endif
#endif
#ifndef HY4_RDNA2
#define HY4_RDNA2 0
#endif
#ifndef HY4_MMA
#if HY4_RDNA2
#define HY4_MMA 0
#else
#define HY4_MMA 1
#endif
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define HY4_GFX12 1
#else
#define HY4_GFX12 0
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx1200__) || defined(__gfx1201__)
#define HY4_DEV_WMMA 1
#else
#define HY4_DEV_WMMA 0
#endif
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
#define HY4_HAVE_SDOT4 1
#else
#define HY4_HAVE_SDOT4 0
#endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(_WIN32)
#include <cstdlib>
namespace hy4 {
inline hipError_t hy4_hip_host_malloc_wddm(void** ptr, size_t size, unsigned int flags) {
    if ((flags & hipHostMallocMapped) != 0u) {
        static const bool coherent_win = [] {
            const char* v = std::getenv("STRATA_HIP_MAPPED_COHERENT");
            return v == nullptr || v[0] != '0';
        }();
        if (coherent_win) flags |= hipHostMallocCoherent;
    }
    return hipHostMalloc(ptr, size, flags);
}
}  // namespace hy4
#define hipHostMalloc(p, n, f) ::hy4::hy4_hip_host_malloc_wddm((p), (n), (f))
#endif

#include <cstdint>
#include <cstring>
namespace hy4 {
inline const char* gpu_arch(const hipDeviceProp_t& p) { return p.gcnArchName; }
__device__ __forceinline__ void hy4_sync_lds() {
#if HY4_GFX12
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#else
    __syncthreads();
#endif
}
inline bool hy4_is_rdna2() {
#if HY4_RDNA2
    return true;
#else
    static int cached = -1;
    if (cached < 0) {
        hipDeviceProp_t dp{};
        if (hipGetDeviceProperties(&dp, 0) == hipSuccess)
            cached = (std::strncmp(dp.gcnArchName, "gfx10", 5) == 0) ? 1 : 0;
        else
            cached = 0;
    }
    return cached != 0;
#endif
}
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) { return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) - __builtin_bit_cast(hv2, b)); }
__device__ __forceinline__ uint32_t h2_mul(uint32_t a, uint32_t b) { return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) * __builtin_bit_cast(hv2, b)); }
__device__ __forceinline__ uint32_t h2_splat(float f) {
    const hv2 h = {(_Float16)f, (_Float16)f};
    return __builtin_bit_cast(uint32_t, h);
}
__device__ __forceinline__ float h2_dot_f32(uint32_t a, uint32_t b, float c) { return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false); }
__device__ __forceinline__ int sdot4_i32(int a, int b, int c) {
#if HY4_HAVE_SDOT4
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    int s = c;
    s += (int)(int8_t)(a & 0xFF) * (int)(int8_t)(b & 0xFF);
    s += (int)(int8_t)((a >> 8) & 0xFF) * (int)(int8_t)((b >> 8) & 0xFF);
    s += (int)(int8_t)((a >> 16) & 0xFF) * (int)(int8_t)((b >> 16) & 0xFF);
    s += (int)(int8_t)((a >> 24) & 0xFF) * (int)(int8_t)((b >> 24) & 0xFF);
    return s;
#endif
}
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
