// sy_common.h - shared SYCL helpers of the Hy4-Flash Intel Arc runner (lane HY4-SYCL, target: Arc A750 8 GB, oneAPI icpx, Level Zero).
// Conventions (every kernel module of the lane follows them, so the pieces fit together):
//   * one global in-order queue hy4s::Q(); all kernels are asynchronous on it, no allocation inside kernel wrappers, no host sync inside them.
//   * device kernels use sub-group size 32 (REQD_SG32) so the reductions match the wave32 HIP code; work-group sizes are multiples of 32.
//   * residual stream x is fp32 [T][4][3072]; GEMM inputs are fp16 (sycl::half), GEMM outputs fp32 (same as the HIP runner).
//   * device buffers larger than 4 GiB are never allocated (Arc: relaxed allocation limits are off by default): big weights live in slabs.
#pragma once
#include <sycl/sycl.hpp>
#if __has_include(<sycl/ext/oneapi/experimental/graph.hpp>)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#define HY4S_HAS_GRAPH 1
#else
#define HY4S_HAS_GRAPH 0
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#define REQD_SG16 [[sycl::reqd_sub_group_size(16)]]
#define REQD_SG32 [[sycl::reqd_sub_group_size(32)]]

namespace hy4s {

inline void hy4s_setenv(const char* k, const char* v, int overwrite = 1) {
#ifdef _WIN32
    if (!overwrite && std::getenv(k)) return;
    _putenv_s(k, v);
#else
    setenv(k, v, overwrite);
#endif
}

[[noreturn]] inline void hy4s_exit_fast(int code) {
    fflush(stdout);
    fflush(stderr);
    _exit(code);
}

// storage bytes this process has read (/proc/self/io read_bytes: page-cache misses, O_DIRECT reads, readahead), in GB
inline double io_read_gb() {
    double v = 0;
#ifndef _WIN32
    if (FILE* f = fopen("/proc/self/io", "r")) {
        char ln[128];
        while (fgets(ln, sizeof ln, f)) {
            unsigned long long b;
            if (sscanf(ln, "read_bytes: %llu", &b) == 1) v = (double)b / 1e9;
        }
        fclose(f);
    }
#endif
    return v;
}

// Tree-8 speculative verification topology passed to device kernels (command_graph replayable)
struct TreePos {
    int32_t n = 0;               // 0 = standard linear chunk; 1..8 = tree node count
    uint8_t depth[8] = {};       // logical depth of node t (or source node index for KV/ivg compaction)
    uint8_t anc[8][8] = {};      // anc[t][d] = physical node index of node t's ancestor at depth d
};

// Device-resident dynamic step parameters for SYCL command_buffer / command_graph replay
struct StepParams {
    int32_t p0 = 0;
    int32_t cap_row = 0;
    int32_t a = 0;
    int32_t tree_on = 0;
    TreePos tree{};
};

using half = sycl::half;

constexpr int HID = 3072;
constexpr int NHC = 4;

inline int& primary_gpu_index() {
    static int idx = 0;
    return idx;
}

inline void set_primary_gpu_index(int idx) {
    primary_gpu_index() = idx < 0 ? 0 : idx;
}

// Enumerate all GPU devices on the active SYCL backend (Level Zero by default, or OpenCL when HY4S_BACKEND=opencl).
// Filtering to a single backend avoids duplicate OpenCL + Level Zero views of the same physical Arc cards.
inline std::vector<sycl::device> enum_gpu_devices() {
    std::vector<sycl::device> all = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (all.empty()) return all;
    const char* be_env = std::getenv("HY4S_BACKEND");
    const bool want_ocl = (be_env && std::string(be_env) == "opencl");
    const sycl::backend want_be = want_ocl ? sycl::backend::opencl : sycl::backend::ext_oneapi_level_zero;
    std::vector<sycl::device> filtered;
    for (const auto& d : all) {
        if (d.get_backend() == want_be) filtered.push_back(d);
    }
    if (!filtered.empty()) return filtered;
    // Fall back to the backend of device 0 if preferred backend isn't present
    const sycl::backend b0 = all[0].get_backend();
    for (const auto& d : all) {
        if (d.get_backend() == b0) filtered.push_back(d);
    }
    return filtered;
}

// Parse --gpus / --devices / HY4_GPUS spec: accepts either a count ("2") or a CSV ("0,1" / "1,0").
inline void parse_gpu_spec(const std::string& spec, int default_primary, int n_avail, int& out_primary, std::vector<int>& out_peers) {
    out_primary = std::max(0, default_primary);
    out_peers.clear();
    if (spec.empty() || n_avail <= 0) {
        if (n_avail > 0 && out_primary >= n_avail) out_primary = 0;
        return;
    }
    if (spec.find(',') != std::string::npos) {
        std::vector<int> ids;
        for (size_t pos = 0; pos < spec.size();) {
            size_t q = spec.find(',', pos);
            if (q == std::string::npos) q = spec.size();
            if (q > pos) {
                int id = std::atoi(spec.substr(pos, q - pos).c_str());
                if (id >= 0 && id < n_avail) {
                    bool dup = false;
                    for (int x : ids) if (x == id) { dup = true; break; }
                    if (!dup) ids.push_back(id);
                } else {
                    fprintf(stderr, "[gpu] ignoring out-of-range device index %d (visible GPUs: %d)\n", id, n_avail);
                }
            }
            pos = q + 1;
        }
        if (!ids.empty()) {
            out_primary = ids[0];
            for (size_t i = 1; i < ids.size(); ++i) out_peers.push_back(ids[i]);
        }
    } else {
        const int count = std::atoi(spec.c_str());
        if (out_primary >= n_avail) out_primary = 0;
        const int target = std::min(std::max(count, 1), n_avail);
        for (int d = 0; d < n_avail && (int)out_peers.size() + 1 < target; ++d) {
            if (d != out_primary) out_peers.push_back(d);
        }
    }
}

inline sycl::queue& Q() {
    // HY4S_DISCARD=1: the queue creates no event objects (cheaper launches; the events returned by submit/memcpy then must not be waited on - use host flags / q.wait())
    // HY4S_CPU_DEVICE=1: select CPU SYCL device (for CPU-only unit tests when GPU is unavailable/busy)
    static sycl::queue q = [] {
        auto make_props = []() -> sycl::property_list {
            const char* pf = std::getenv("HY4S_PROF");
            if (pf && pf[0] == '1') return sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
            const char* d = std::getenv("HY4S_DISCARD");
            if (d && d[0] == '1') return sycl::property_list{sycl::property::queue::in_order(), sycl::ext::oneapi::property::queue::discard_events()};
            return sycl::property_list{sycl::property::queue::in_order()};
        };
        const char* cpu_e = std::getenv("HY4S_CPU_DEVICE");
        if (cpu_e && cpu_e[0] == '1') {
            return sycl::queue{sycl::cpu_selector_v, make_props()};
        }
        auto devs = enum_gpu_devices();
        if (!devs.empty()) {
            const int idx = std::min(std::max(primary_gpu_index(), 0), (int)devs.size() - 1);
            return sycl::queue{devs[idx], make_props()};
        }
        return sycl::queue{sycl::gpu_selector_v, make_props()};
    }();
    return q;
}

inline void wait() { Q().wait_and_throw(); }

struct DevCaps {
    std::string name;
    std::string platform;
    size_t global_mem = 0;
    size_t max_alloc = 0;
    size_t local_mem = 0;
    size_t max_wg = 0;
    uint32_t eu_count = 0;
    bool has_sg8 = false;
    bool has_sg16 = false;
    bool has_sg32 = false;
    bool is_level_zero = false;
    bool has_graph = false;
    bool has_xmx_sg8 = false;
    bool has_xmx_sg16 = false;
};

inline const DevCaps& dev_caps() {
    static DevCaps c = [] {
        DevCaps d;
        auto dev = Q().get_device();
        d.name = dev.get_info<sycl::info::device::name>();
        d.platform = dev.get_platform().get_info<sycl::info::platform::name>();
        d.global_mem = dev.get_info<sycl::info::device::global_mem_size>();
        d.max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        d.local_mem = dev.get_info<sycl::info::device::local_mem_size>();
        d.max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
        d.eu_count = dev.get_info<sycl::info::device::max_compute_units>();
        for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
            if (s == 8) d.has_sg8 = true;
            if (s == 16) d.has_sg16 = true;
            if (s == 32) d.has_sg32 = true;
        }
        d.is_level_zero = (dev.get_backend() == sycl::backend::ext_oneapi_level_zero);
#if HY4S_HAS_GRAPH
        d.has_graph = dev.has(sycl::aspect::ext_oneapi_limited_graph) || dev.has(sycl::aspect::ext_oneapi_graph);
#endif
        // Alchemist (DG2 / Arc A-series) supports SG=8 XMX (8x8x16 fp16, 8x8x32 int8);
        // Battlemage Xe2 (Arc B570/B580 bmg_g21, Arc Pro B60/B70 bmg_g31, Lunar Lake lnl_m) / PVC supports SG=16 XMX (8x16x16 fp16) and has no SG=8.
        // Older/non-XMX Intel iGPUs fall back to SG=32 SIMT tiled GEMM when HY4S_NO_XMX=1 or when joint_matrix is unsupported.
        const bool no_xmx = std::getenv("HY4S_NO_XMX") && std::getenv("HY4S_NO_XMX")[0] == '1';
        if (!no_xmx) {
            bool is_arc_or_xmx = (d.name.find("Arc") != std::string::npos || d.name.find("Data Center") != std::string::npos ||
                                  d.name.find("A7") != std::string::npos || d.name.find("A5") != std::string::npos ||
                                  d.name.find("A3") != std::string::npos || d.name.find("B5") != std::string::npos ||
                                  d.name.find("B6") != std::string::npos || d.name.find("B7") != std::string::npos ||
                                  d.name.find("Pro B") != std::string::npos ||
                                  d.name.find("140V") != std::string::npos || d.name.find("130V") != std::string::npos ||
                                  d.name.find("Battlemage") != std::string::npos || d.name.find("Alchemist") != std::string::npos ||
                                  d.name.find("DG2") != std::string::npos || d.name.find("BMG") != std::string::npos);
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
            try {
                namespace exp = sycl::ext::oneapi::experimental;
                const auto a = dev.get_info<exp::info::device::architecture>();
                if (a == exp::architecture::intel_gpu_acm_g10 || a == exp::architecture::intel_gpu_acm_g11 ||
                    a == exp::architecture::intel_gpu_acm_g12 || a == exp::architecture::intel_gpu_bmg_g21 ||
                    a == exp::architecture::intel_gpu_bmg_g31 || a == exp::architecture::intel_gpu_lnl_m ||
                    a == exp::architecture::intel_gpu_pvc || a == exp::architecture::intel_gpu_pvc_vg) {
                    is_arc_or_xmx = true;
                }
            } catch (...) {
            }
#endif
            // Linux drivers for Arc Pro B70 / Battlemage may report PCI IDs like "Intel(R) Graphics [0xe223]"
            if (!is_arc_or_xmx) {
                if (const size_t b = d.name.find("[0x"); b != std::string::npos) {
                    const unsigned long id = std::strtoul(d.name.c_str() + b + 1, nullptr, 16);
                    if ((id >= 0xE200 && id <= 0xE2FF) || (id >= 0x5690 && id <= 0x56BF) || id == 0x56C0 || id == 0x56C1)
                        is_arc_or_xmx = true;
                }
            }
            if (is_arc_or_xmx) {
                if (d.has_sg8) d.has_xmx_sg8 = true;
                else if (d.has_sg16) d.has_xmx_sg16 = true;
            }
        }
        return d;
    }();
    return c;
}

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
