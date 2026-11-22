// engine/src/inkling2/cuda/ink_cuda.cuh - Portable NVIDIA CUDA primitives for Inkling-2-Mini (sm_70 .. sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Covers Volta (sm_70), Turing (sm_75), Ampere (sm_80/86), Ada (sm_89), and Blackwell (sm_120/120a/121/121a)
// on both Linux (x86_64 + aarch64) and Windows (MSVC/NVCC).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#include <cuda_fp4.h>
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define INK_CK(call)                                                                                                  \
    do {                                                                                                              \
        cudaError_t _e = (call);                                                                                      \
        if (_e != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "strata-inkling CUDA error %d (%s) at %s:%d\n", (int)_e, cudaGetErrorString(_e),     \
                         __FILE__, __LINE__);                                                                         \
            std::exit(3);                                                                                             \
        }                                                                                                             \
    } while (0)

#define INK_CUBLAS_CK(call)                                                                                           \
    do {                                                                                                              \
        cublasStatus_t _s = (call);                                                                                   \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                                            \
            std::fprintf(stderr, "strata-inkling cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__);           \
            std::exit(3);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace strata::ink {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename T>
inline T* dalloc(size_t count) {
    T* p = nullptr;
    if (count > 0) INK_CK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

template <typename T>
inline T* halloc_pinned(size_t count) {
    T* p = nullptr;
    if (count > 0) INK_CK(cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocDefault));
    return p;
}

inline int current_sm_arch(int dev = 0) {
    static int cached_dev = -1;
    static int cached_sm = 0;
    if (cached_dev == dev && cached_sm > 0) return cached_sm;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        cached_dev = dev;
        cached_sm = p.major * 10 + p.minor;
    }
    return cached_sm;
}

// Unified-Memory-safe VRAM query: on Linux integrated GPUs (e.g. DGX Spark GB10 sm_121 128 GB UMA),
// cudaMemGetInfo subtracts reclaimable OS page cache (like the 19 GB .strata file just read!),
// so take max(cuda_free, /proc/meminfo MemAvailable) to prevent false hybrid spill!
inline cudaError_t ink_cuda_mem_info(size_t* vfree, size_t* vtot, int dev = -1) {
    cudaError_t ce = cudaMemGetInfo(vfree, vtot);
    if (ce != cudaSuccess) return ce;
#if defined(__linux__)
    int cur = dev;
    if (cur < 0) cudaGetDevice(&cur);
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, cur) == cudaSuccess && (p.integrated != 0 || (p.major == 12 && p.minor == 1))) {
        FILE* f = std::fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (std::fgets(line, sizeof(line), f)) {
                unsigned long long kb = 0;
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                    size_t avail = (size_t)kb * 1024ull;
                    if (avail > *vfree) *vfree = std::min(avail, *vtot);
                    break;
                }
            }
            std::fclose(f);
        }
    }
#else
    (void)dev;
#endif
    return cudaSuccess;
}

// ---- Exact Inkling-2-Mini model constants ----
constexpr int kInkPhysLayers  = 24;
constexpr int kInkVisits      = 48;
constexpr int kInkCoreStart   = 6;
constexpr int kInkCoreEnd     = 18;
constexpr int kInkCorePasses  = 3;
constexpr int kInkHidden      = 2304;
constexpr int kInkInter       = 1152;
constexpr int kInkHeads       = 24;
constexpr int kInkKVHeads     = 8;
constexpr int kInkHeadDim     = 128;
constexpr int kInkKVDim       = kInkKVHeads * kInkHeadDim;   // 1024
constexpr int kInkQDim        = kInkHeads * kInkHeadDim;     // 3072
constexpr int kInkDRel        = 16;
constexpr int kInkRelExtent   = 1024;
constexpr int kInkSwaWin      = 1024;
constexpr int kInkBlockSize   = 6;
constexpr int kInkExperts     = 128;
constexpr int kInkTopK        = 8;
constexpr int kInkGateRank    = 64;
constexpr int kInkVocabPad    = 201024;
constexpr int kInkVocabUnpad  = 200058;
constexpr int kInkDsparkLayers = 3;
constexpr int kInkDsparkBlock  = 8;
constexpr int kInkMaxTree      = 32;
constexpr int kInkMaxTreeDepth = 8;  // root (depth 0) + up to 7 draft steps = 8 ancestors max!
constexpr int kInkDsparkInter  = 9216;
constexpr int kInkMarkovRank   = 256;
constexpr int kInkSconvCh      = 2 * kInkKVDim + 2 * kInkHidden; // 1024 + 1024 + 2304 + 2304 = 6656

constexpr float kInkRmsEps     = 1e-6f;
constexpr float kInkSituBeta   = 4.0f;
constexpr float kInkSituLb     = 25.0f;
constexpr float kInkQkClip     = 22.627416997969522f;
constexpr float kInkRelClip    = 8.0f;
constexpr float kInkRouteScale = 8.0f;
constexpr float kInkLogAlpha   = 0.1f;
constexpr int   kInkLogFloor   = 8192;
constexpr float kInkSoftcap    = 30.0f;

// Format IDs matching .strata v1.1 Section 0 / Section 3
enum InkFmt : int {
    FMT_S1_25 = 0,
    FMT_S1_5  = 1,
    FMT_S2_K  = 2,
    FMT_S3_K  = 3,
    FMT_S4_K  = 4,
    FMT_S5_K  = 5,
    FMT_S6_K  = 6,
    FMT_S8_K  = 8,
    FMT_SF4   = 15,
    FMT_MX8   = 16,
    FMT_Q8ROW = 100,
    FMT_F32   = 101,
    FMT_BF16  = 102,
    FMT_F16   = 103,
};

__host__ __device__ constexpr int ink_sb_bytes(int fmt) {
    switch (fmt) {
        case FMT_S1_25: return 48;
        case FMT_S1_5:  return 58;
        case FMT_S2_K:  return 70;
        case FMT_S3_K:  return 102;
        case FMT_S4_K:  return 134;
        case FMT_S5_K:  return 166;
        case FMT_S6_K:  return 198;
        case FMT_S8_K:  return 262;
        default:        return 0;
    }
}

__host__ __device__ constexpr float ink_sk_half(int fmt) {
    switch (fmt) {
        case FMT_S2_K: return 1.5f;
        case FMT_S3_K: return 3.5f;
        case FMT_S4_K: return 7.5f;
        case FMT_S5_K: return 15.5f;
        case FMT_S6_K: return 31.5f;
        default:       return 0.0f;
    }
}

__host__ __device__ inline int ink_visit_phys(int visit) {
    if (visit < kInkCoreStart) return visit;
    const int core_len = kInkCoreEnd - kInkCoreStart;
    if (visit < kInkCoreStart + kInkCorePasses * core_len) {
        return kInkCoreStart + ((visit - kInkCoreStart) % core_len);
    }
    return kInkCoreEnd + (visit - kInkCoreStart - kInkCorePasses * core_len);
}

__host__ __device__ inline int ink_visit_pass(int visit) {
    const int core_len = kInkCoreEnd - kInkCoreStart;
    if (visit < kInkCoreStart || visit >= kInkCoreStart + kInkCorePasses * core_len) return 0;
    return (visit - kInkCoreStart) / core_len;
}

__host__ __device__ inline bool ink_is_global_phys(int ip) {
    return ip == 5 || ip == 11 || ip == 17 || ip == 23;
}

// ---- Device math & conversion helpers (sm_70 .. sm_121a) ----
#if defined(__CUDACC__)

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    return *reinterpret_cast<const uint16_t*>(&b);
#else
    uint32_t u = __float_as_uint(x);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding = 0x7FFFu + lsb;
    return (uint16_t)((u + rounding) >> 16);
#endif
}

__device__ __forceinline__ float2 bf16x2_to_f32x2(uint32_t pair) {
    return make_float2(__uint_as_float((pair & 0xFFFFu) << 16), __uint_as_float(pair & 0xFFFF0000u));
}

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
#else
    return (uint32_t)f32_to_bf16(a) | ((uint32_t)f32_to_bf16(b) << 16);
#endif
}

__device__ __forceinline__ float f16_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Fast exact E2M1 nibble decode -> float: {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign bit 3.
// Zero LUT memory traffic: pure register arithmetic (3 ALU ops) or sm_120a hardware cvt!
__device__ __forceinline__ float e2m1_nibble_to_f32(uint32_t nib) {
    const uint32_t mag = nib & 7u;
    // mag: 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
    // Pack magnitudes as scaled half-integers (x2): {0, 1, 2, 3, 4, 6, 8, 12} -> 4 bits per entry in 0xC8643210u!
    const uint32_t x2 = (0xC8643210u >> (mag * 4u)) & 0xFu;
    float v = 0.5f * (float)x2;
    return (nib & 8u) ? -v : v;
}

// Decode 2 consecutive E2M1 nibbles packed in one byte (lo = elem 0, hi = elem 1) to float2.
// On Blackwell (sm_100+/sm_120+), uses 1-instruction hardware cvt via <cuda_fp4.h>!
__device__ __forceinline__ float2 e2m1x2_byte_to_f32x2(uint8_t byte) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
    return __half22float2(__half2(hr));
#else
    return make_float2(e2m1_nibble_to_f32(byte & 0xFu), e2m1_nibble_to_f32((byte >> 4) & 0xFu));
#endif
}

// Exact F8_E4M3FN byte -> float (1-cycle constant-cache LUT on sm_60..sm_86, hardware cvt on sm_89+!).
#if defined(__CUDACC__)
__device__ __forceinline__ float e4m3_to_f32_compute(uint8_t c) {
    const uint32_t sgn = ((uint32_t)c & 0x80u) << 24;
    const uint32_t mag = (uint32_t)c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    if (mag == 0u) return __uint_as_float(sgn);
    const uint32_t exp = mag >> 3;
    const uint32_t man = mag & 7u;
    if (exp == 0u) {
        float v = (float)man * (1.0f / 512.0f);
        return sgn ? -v : v;
    }
    return __uint_as_float(sgn | ((exp + 120u) << 23) | (man << 20));
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
    return __half2float(__half(hr));
#else
    // Branchless bit-math (~4 ALU ops, ZERO divergent branches on Volta/Turing/Ampere/Pascal!)
    const uint32_t sgn = ((uint32_t)c & 0x80u) << 24;
    const uint32_t mag = (uint32_t)c & 0x7Fu;
    const uint32_t exp = mag >> 3;
    const uint32_t man = mag & 7u;
    const float norm = __uint_as_float(sgn | ((exp + 120u) << 23) | (man << 20));
    const float subn = __uint_as_float(sgn | __float_as_uint((float)man * (1.0f / 512.0f)));
    float res = (exp == 0u) ? subn : norm;
    return (mag == 0x7Fu) ? 0.0f : res;
#endif
}
#endif

// Fast 2x E4M3FN bytes packed in low 16 bits -> float2 (hardware cvt on sm_89+ / Blackwell via <cuda_fp8.h>!).
__device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t pair) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)pair, __NV_E4M3);
    return __half22float2(__half2(hr));
#else
    return make_float2(e4m3_to_f32((uint8_t)(pair & 0xFFu)), e4m3_to_f32((uint8_t)(pair >> 8)));
#endif
}

// Exact UE8M0 scale byte -> float (2^(e - 127), e==0 -> 2^-127, e==255 -> 0).
__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    if (e == 255u) return 0.0f;
    if (e == 0u) return 5.877471754111438e-39f;
    return __uint_as_float((uint32_t)e << 23);
}

// Exact SITU(beta=4.0, lb=25.0) double-softcapped MoE activation
__device__ __forceinline__ float situ_f32(float g, float u) {
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float ga = kInkSituBeta * tanhf(g * (1.0f / kInkSituBeta)) * sig;
    const float ua = kInkSituLb * tanhf(u * (1.0f / kInkSituLb));
    return ga * ua;
}

// Hardware dp4a.s32.s32 exists ONLY on sm_61 (Pascal GP102/104/106) and sm_75+ (Turing..Blackwell).
// On sm_60 (P100) and sm_70 (V100), PTX dp4a is software-emulated; use single-instruction PRMT.S4E + IMAD!
__device__ __forceinline__ int32_t dp4a_i8(uint32_t a, uint32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 610) || (__CUDA_ARCH__ >= 750))
    return __dp4a((int)a, (int)b, c);
#else
    const int32_t a0 = (int32_t)(int8_t)(a),        b0 = (int32_t)(int8_t)(b);
    const int32_t a1 = (int32_t)(int8_t)(a >> 8),   b1 = (int32_t)(int8_t)(b >> 8);
    const int32_t a2 = (int32_t)(int8_t)(a >> 16),  b2 = (int32_t)(int8_t)(b >> 16);
    const int32_t a3 = (int32_t)(int8_t)(a >> 24),  b3 = (int32_t)(int8_t)(b >> 24);
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int tid = threadIdx.x;
    v = warp_sum(v);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float s = (tid < nw) ? sh[tid] : 0.0f;
    if (tid < 32) {
        s = warp_sum(s);
        if (tid == 0) sh[0] = s;
    }
    __syncthreads();
    const float res = sh[0];
    __syncthreads();
    return res;
}

// Branchless 2-byte-aligned-safe 16-bit and 32-bit loads (all S_K superblocks are guaranteed 2-byte aligned!)
__device__ __forceinline__ uint16_t ld16_u(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t ld32_u(const uint8_t* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)p16[0] | ((uint32_t)p16[1] << 16);
}

// Native INT8 Tensor Core m16n8k32 (sm_80+: RTX 3090, A100, 4090, 5090, GB10)
__device__ __forceinline__ void mma_m16n8k32_s8(int32_t (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// Native FP16 Tensor Core m16n8k16 with FP32 accumulation (sm_80+: RTX 3090, A100, 4090, 5090, GB10)
__device__ __forceinline__ void mma_m16n8k16_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// Native BF16 Tensor Core m16n8k16 (sm_80+)
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

#endif  // __CUDACC__

}  // namespace strata::ink
