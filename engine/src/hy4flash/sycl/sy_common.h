// sy_common.h - shared SYCL helpers of the Hy4-Flash Intel Arc runner (lane HY4-SYCL, target: Arc A750 8 GB, oneAPI icpx, Level Zero).
// Conventions (every kernel module of the lane follows them, so the pieces fit together):
//   * one global in-order queue hy4s::Q(); all kernels are asynchronous on it, no allocation inside kernel wrappers, no host sync inside them.
//   * device kernels use sub-group size 32 (REQD_SG32) so the reductions match the wave32 HIP code; work-group sizes are multiples of 32.
//   * residual stream x is fp32 [T][4][3072]; GEMM inputs are fp16 (sycl::half), GEMM outputs fp32 (same as the HIP runner).
//   * device buffers larger than 4 GiB are never allocated (Arc: relaxed allocation limits are off by default): big weights live in slabs.
#pragma once
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define REQD_SG32 [[sycl::reqd_sub_group_size(32)]]

namespace hy4s {

using half = sycl::half;

constexpr int HID = 3072;
constexpr int NHC = 4;

inline sycl::queue& Q() {
    // HY4S_DISCARD=1: the queue creates no event objects (cheaper launches; the events returned by submit/memcpy then must not be waited on - use host flags / q.wait())
    static sycl::queue q = [] {
        const char* pf = std::getenv("HY4S_PROF");
        if (pf && pf[0] == '1') return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
        const char* d = std::getenv("HY4S_DISCARD");
        if (d && d[0] == '1')
            return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::ext::oneapi::property::queue::discard_events()}};
        return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
    }();
    return q;
}

inline void wait() { Q().wait_and_throw(); }

inline size_t& dev_bytes_allocated() { static size_t b = 0; return b; }  // running total of dalloc bytes (the free-memory query of the driver reports the whole card)
template <class T> inline T* dalloc(size_t n) {
    T* p = sycl::malloc_device<T>(n ? n : 1, Q());
    dev_bytes_allocated() += (n ? n : 1) * sizeof(T);
    if (!p) {
        fprintf(stderr, "dalloc: out of device memory (%zu bytes)\n", n * sizeof(T));
        exit(3);
    }
    return p;
}
template <class T> inline T* halloc(size_t n) {  // pinned host memory
    T* p = sycl::malloc_host<T>(n ? n : 1, Q());
    if (!p) {
        fprintf(stderr, "halloc: out of host memory (%zu bytes)\n", n * sizeof(T));
        exit(3);
    }
    return p;
}
inline void dfree(void* p) { sycl::free(p, Q()); }
inline void h2d(void* d, const void* h, size_t bytes) { Q().memcpy(d, h, bytes); }
inline void d2h(void* h, const void* d, size_t bytes) { Q().memcpy(h, d, bytes); }
inline void d2d(void* dst, const void* src, size_t bytes) { Q().memcpy(dst, src, bytes); }
inline void dzero(void* d, size_t bytes) { Q().memset(d, 0, bytes); }

// ---- weight formats (same ids as skgemm.hip.h Fmt, and as hy4_manifest.h WFmt)
enum Fmt { FMT_S2 = 2, FMT_S3 = 3, FMT_S4 = 4, FMT_S5 = 5, FMT_S6 = 6, FMT_S8 = 8, FMT_Q8ROW = 100, FMT_F32 = 101 };

constexpr int sk_sb_bytes(int f) {
    return f == FMT_S2 ? 70 : f == FMT_S3 ? 102 : f == FMT_S4 ? 134 : f == FMT_S5 ? 166 : f == FMT_S6 ? 198 : f == FMT_S8 ? 262 : 0;
}
constexpr size_t weight_bytes(int fmt, int N, int K) {
    return fmt == FMT_Q8ROW ? (size_t)N * 4 + (size_t)N * K : fmt == FMT_F32 ? (size_t)N * K * 4 : (size_t)N * (K / 256) * sk_sb_bytes(fmt);
}

// ---- device-side helpers
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }
inline float siluf_(float x) { return x / (1.0f + sycl::native::exp(-x)); }
inline uint16_t ld16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
inline float h2f_bits(uint16_t b) { return (float)sycl::bit_cast<half>(b); }

template <int FMT> constexpr float sk_half() {
    return FMT == FMT_S2 ? 1.5f : FMT == FMT_S3 ? 3.5f : FMT == FMT_S4 ? 7.5f : FMT == FMT_S5 ? 15.5f : FMT == FMT_S6 ? 31.5f : 0.0f;
}
// quantized integer of weight i (0..255) of a super-block payload `pl` (= block + 6)
template <int FMT> inline int sk_q(const uint8_t* pl, int i) {
    if constexpr (FMT == FMT_S2) return (pl[i >> 2] >> ((i & 3) * 2)) & 3;
    else if constexpr (FMT == FMT_S3) return ((pl[i >> 2] >> ((i & 3) * 2)) & 3) | (((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 2);
    else if constexpr (FMT == FMT_S4) return (pl[i >> 1] >> ((i & 1) * 4)) & 15;
    else if constexpr (FMT == FMT_S5) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 3)] >> (i & 7)) & 1) << 4);
    else if constexpr (FMT == FMT_S6) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 2)] >> ((i & 3) * 2)) & 3) << 4);
    else return (int)(int8_t)pl[i];  // S8
}
// weight i (0..255) of super-block blk: (q - H) * D * u[g]/255
template <int FMT> inline float sk_weight(const uint8_t* blk, int i) {
    const float D = h2f_bits(ld16(blk));
    const float eff = D * ((float)blk[2 + (i >> 6)] / 255.0f);
    return ((float)sk_q<FMT>(blk + 6, i) - sk_half<FMT>()) * eff;
}

// sub-group (32 lanes) reductions
template <class It> inline float sg_sum(const It& it, float v) { return sycl::reduce_over_group(it.get_sub_group(), v, sycl::plus<float>()); }
// work-group sum for a 1-D group of any multiple-of-32 size; `sh` is a local array of >= 32 floats
template <class It, class P> inline float wg_sum(const It& it, float v, P sh) {
    auto sg = it.get_sub_group();
    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    const int w = (int)sg.get_group_linear_id(), l = (int)sg.get_local_linear_id();
    sycl::group_barrier(it.get_group());
    if (l == 0) sh[w] = v;
    sycl::group_barrier(it.get_group());
    const int nw = (int)it.get_local_range(0) >> 5;
    float r = (l < nw) ? (float)sh[l] : 0.0f;
    return sycl::reduce_over_group(sg, r, sycl::plus<float>());
}

// work-group local memory declared inside the kernel (like __shared__): T* p = lmem<float, 32>(it);   (uninitialised)
template <class T, int N, class It> inline T* lmem(const It& it) {
    auto p = sycl::ext::oneapi::group_local_memory_for_overwrite<T[N]>(it.get_group());
    return &(*p)[0];
}
// 1-D launch: G groups of B threads
inline sycl::nd_range<1> NR(size_t G, size_t B) { return sycl::nd_range<1>(G * B, B); }
inline int sg_lane(const sycl::nd_item<1>& it) { return (int)it.get_sub_group().get_local_linear_id(); }
// xor-shuffle inside the sub-group
template <class T> inline T shfl_xor(const sycl::nd_item<1>& it, T v, int m) { return sycl::permute_group_by_xor(it.get_sub_group(), v, (unsigned)m); }

}  // namespace hy4s
