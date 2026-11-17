// Force-included (after the tree's hip_compat/cuda_runtime.h) into every src/split source of the HIP build: the CUDA runtime and
// cuBLAS names src/split uses that the shared shim does not rename.
#pragma once
#include <hip/hip_runtime.h>
#include <cuda_runtime.h>   // the tree's shim (include/strata/hip_compat): its names are redefined below, so it must come first

#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaGetErrorName hipGetErrorName
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaGraphExecUpdateResultInfo fnx_graph_update_info
struct fnx_graph_update_info {
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
};
inline hipError_t cudaGraphExecUpdate(hipGraphExec_t exec, hipGraph_t g, fnx_graph_update_info* info) {
    return hipGraphExecUpdate(exec, g, &info->errorNode, &info->result);
}

// cuBLAS: the f16 products with tensor-op math are hipBLAS's default on a gfx12 card
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_DEFAULT_MATH

// __ldcg: a load that skips the non-coherent caches (data another block of the same launch wrote, behind the hc read's grid barrier):
// a relaxed agent-scope atomic load of 4 / 8 bytes
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "__ldcg: 4 or 8 bytes");
    if constexpr (sizeof(T) == 4) {
        const unsigned u = __hip_atomic_load(reinterpret_cast<const unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 4);
        return r;
    } else {
        const unsigned long long u = __hip_atomic_load(reinterpret_cast<const unsigned long long*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 8);
        return r;
    }
}

// HIP: a plain cudaMemset goes through the legacy stream, which HIP refuses while any stream of the process is capturing (a buffer first
// needed inside a graph capture, as the window graphs' scratch is); the zeroing runs on a stream of its own per card and thread instead
inline hipError_t fnx_zero_dev(void* p, size_t bytes) {
    static thread_local hipStream_t st[64] = {};
    int dev = 0;
    hipGetDevice(&dev);
    if (dev < 0 || dev >= 64) return hipMemset(p, 0, bytes);
    if (!st[dev]) {
        const hipError_t e = hipStreamCreateWithFlags(&st[dev], hipStreamNonBlocking);
        if (e != hipSuccess) return e;
    }
    const hipError_t e = hipMemsetAsync(p, 0, bytes, st[dev]);
    return e != hipSuccess ? e : hipStreamSynchronize(st[dev]);
}
#define FNX_ZERO_DEV(p, bytes) fnx_zero_dev((p), (bytes))

// The synchronous copies and sets (cudaMemcpy, cudaMemset) are the legacy-stream kind, which HIP refuses while ANY stream of the process is
// capturing: the engine captures one card's window graph while the next card's host side still uploads its row tables. They run on a private
// non-blocking stream per card and thread, then wait for it.
inline hipStream_t fnx_side_stream() {
    static thread_local hipStream_t st[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return nullptr;
    if (!st[dev] && hipStreamCreateWithFlags(&st[dev], hipStreamNonBlocking) != hipSuccess) st[dev] = nullptr;
    return st[dev];
}
inline hipError_t fnx_memcpy(void* d, const void* s, size_t n, hipMemcpyKind k) {
    hipStream_t st = fnx_side_stream();
    if (!st) return hipMemcpy(d, s, n, k);
    const hipError_t e = hipMemcpyAsync(d, s, n, k, st);
    return e != hipSuccess ? e : hipStreamSynchronize(st);
}
inline hipError_t fnx_memset(void* p, int v, size_t n) {
    hipStream_t st = fnx_side_stream();
    if (!st) return hipMemset(p, v, n);
    const hipError_t e = hipMemsetAsync(p, v, n, st);
    return e != hipSuccess ? e : hipStreamSynchronize(st);
}
#undef cudaMemcpy
#undef cudaMemset
#define cudaMemcpy fnx_memcpy
#define cudaMemset fnx_memset
