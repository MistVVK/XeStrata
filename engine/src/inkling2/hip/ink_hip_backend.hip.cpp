// engine/src/inkling2/hip/ink_hip_backend.hip.cpp
// Complete Multi-Gen AMD ROCm/HIP Backend for strata-inkling (Inkling-2-Mini 24->48 Looped MoE + DSpark + Vision).
// Supports:
//   - RDNA4 (gfx1200/1201): Wave32 half8 + 2x-rate iu8 WMMA
//   - RDNA3 / RDNA3.5 (gfx1100/1101/1102, gfx1103 Radeon 780M iGPU, gfx1150 Strix Point 890M, gfx1151 Strix Halo)
//   - RDNA2 (gfx1030/1031/1032): zero-WMMA Wave32 sdot4 + fdot2 tiled GEMM & SIMT Flash-SWA/GQA
//   - Zero-OOM iGPU UMA coherent GTT fallback (512 MB - 4 GB BIOS carveout -> full 32/60/128 GB UMA!)
//   - Windows WDDM TDR & Linux amdgpu MES <= 1.0s dispatch/chunk safety guard
//   - B = 1..64 multi-request continuous batching, paged 64-tok SWA-1024 & Global KV (FP16 & FP8_E4M3)
//   - Persistent on-GPU DSpark Block-8 Recurrent Markov drafter & GPU 23-layer BF16 Vision ViT
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.

#define INKLING_NO_STBI_IMPL
#include "ink_backend.h"
#include "ink_gguf_transcode.h"
#include "../ink_cold_store.hpp"
#include "ink_vision.h"
#include "ink_hip_ops.hip.h"
#include "ink_hip_tree.hip.h"
#include "ink_hip_blaslt.hpp"
#include "ink_hip_hybrid.hpp"
#include "strata/hip_uma_guard.hpp"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/platform/cgroup_mem.hpp"
#include "strata/quant/sk_codecs.hpp"
#if defined(__linux__)
#include "strata/platform/numa_balancing.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ink2 {
namespace {

using namespace strata::inkling2;
using namespace strata::inkling2::hip;
using namespace strata::hip_uma;
using namespace strata::quant::sk;

struct DeviceMemoryInfo {
    char device_name[256]{};
    char arch_name[128]{};
    AmdGen gen = AmdGen::Unknown;
    int cu_count = 0;
    bool is_uma = false;
    uint64_t carveout_free_bytes = 0;
    uint64_t carveout_total_bytes = 0;
    uint64_t effective_free_bytes = 0;
    uint64_t effective_total_bytes = 0;
    uint64_t reserve_bytes = 0;
};

static DeviceMemoryInfo query_device_memory(int gpu_index) noexcept {
    DeviceMemoryInfo info{};
    hipDeviceProp_t dp{};
    if (hipGetDeviceProperties(&dp, gpu_index) == hipSuccess) {
        std::snprintf(info.device_name, sizeof(info.device_name), "%s", dp.name);
        std::snprintf(info.arch_name, sizeof(info.arch_name), "%s", dp.gcnArchName);
        info.gen = classify_amd_gen(dp.gcnArchName);
        info.cu_count = dp.multiProcessorCount;
        info.is_uma = (dp.integrated != 0) || is_known_amd_igpu_arch(dp.gcnArchName);
    }
    size_t fr = 0, tot = 0;
    if (hipMemGetInfo(&fr, &tot) == hipSuccess) {
        info.carveout_free_bytes = fr;
        info.carveout_total_bytes = tot;
    }
    effective_uma_memory(info.is_uma, info.carveout_free_bytes, info.carveout_total_bytes,
                         info.effective_free_bytes, info.effective_total_bytes);
    info.reserve_bytes = info.is_uma ? (1024ULL << 20) : (512ULL << 20);
    return info;
}

static inline void hip_ign(hipError_t) noexcept {}

__global__ void __launch_bounds__(256) k_bf16_to_f32_vec(const uint16_t* src, float* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = strata::inkling2::hip::bf16_to_f32(src[i]);
}

__global__ void __launch_bounds__(256) k_f32_to_bf16_vec(const float* src, uint16_t* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = strata::inkling2::hip::f32_to_bf16(src[i]);
}

__global__ void __launch_bounds__(256) k_f32_to_bf16_vec4(const float* __restrict__ src, uint16_t* __restrict__ dst, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 v = reinterpret_cast<const float4*>(src)[i];
    reinterpret_cast<uint2*>(dst)[i] = make_uint2(
        static_cast<uint32_t>(strata::inkling2::hip::f32_to_bf16(v.x)) | (static_cast<uint32_t>(strata::inkling2::hip::f32_to_bf16(v.y)) << 16),
        static_cast<uint32_t>(strata::inkling2::hip::f32_to_bf16(v.z)) | (static_cast<uint32_t>(strata::inkling2::hip::f32_to_bf16(v.w)) << 16));
}

__global__ void __launch_bounds__(256) k_add_bias_rows(float* Y, const float* bias, int M, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) Y[idx] += bias[idx % N];
}

__global__ void __launch_bounds__(256) k_fill_dspark_masks(const float* mask_e, float* dst_rows, int n_masks, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_masks * D) dst_rows[idx] = mask_e[idx % D];
}

// Batched DSpark context scatter: copies [B, cols] FP32 -> per-slot BF16 ring at pos[b] in 1 kernel launch!
__global__ void __launch_bounds__(256) k_dspark_ctx_scatter(
    const float* __restrict__ src,         // [B, cols]
    uint16_t* const* __restrict__ dst_tbl, // [max_slots] per-slot base pointers
    const int32_t* __restrict__ slot_ids,  // [B]
    const int32_t* __restrict__ pos,       // [B]
    int B, int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= cols || b >= B) return;
    uint16_t* dst = dst_tbl[slot_ids[b]] + static_cast<size_t>(pos[b]) * cols;
    dst[c] = strata::inkling2::hip::f32_to_bf16(src[static_cast<size_t>(b) * cols + c]);
}

// Single-launch batched ShortConv clear across ALL n_visits for a pair of contiguous pools [n_visits, max_slots, 3, C]!
__global__ void __launch_bounds__(256) k_sconv_clear_pair(
    uint16_t* __restrict__ pool_a,
    uint16_t* __restrict__ pool_b,
    int slot, int max_slots, int n_visits, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;
    const size_t base = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    uint16_t* da = pool_a + base;
    uint16_t* db = pool_b + base;
    da[0 * C + c] = 0; da[1 * C + c] = 0; da[2 * C + c] = 0;
    db[0 * C + c] = 0; db[1 * C + c] = 0; db[2 * C + c] = 0;
}

// Single-launch batched ShortConv snapshot across ALL n_visits (replaces 96 hipMemcpyAsync calls with 1 kernel!)
__global__ void __launch_bounds__(256) k_sconv_snapshot_pair(
    const uint16_t* __restrict__ pool_a,  // [n_visits, max_slots, 3, C]
    const uint16_t* __restrict__ pool_b,  // [n_visits, max_slots, 3, C]
    uint16_t* __restrict__ snap_a,        // [n_visits, 3, C]
    uint16_t* __restrict__ snap_b,        // [n_visits, 3, C]
    int slot, int max_slots, int n_visits, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;
    const size_t sbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t dbase = static_cast<size_t>(v) * 3 * C;
    for (int lag = 0; lag < 3; ++lag) {
        snap_a[dbase + lag * C + c] = pool_a[sbase + lag * C + c];
        snap_b[dbase + lag * C + c] = pool_b[sbase + lag * C + c];
    }
}

// Single-launch batched ShortConv exact rollback across ALL n_visits for keep_steps in [0..8]
// (replaces 96 k_sconv_rollback_exact launches with 1 kernel -> 2 total launches instead of 192 per DSpark round!)
__global__ void __launch_bounds__(256) k_sconv_rollback_pair(
    const uint16_t* __restrict__ snap_a,  // [n_visits, 3, C]
    const uint16_t* __restrict__ rec_a,   // [n_visits, 8, C]
    uint16_t* __restrict__ pool_a,        // [n_visits, max_slots, 3, C]
    const uint16_t* __restrict__ snap_b,  // [n_visits, 3, C]
    const uint16_t* __restrict__ rec_b,   // [n_visits, 8, C]
    uint16_t* __restrict__ pool_b,        // [n_visits, max_slots, 3, C]
    int slot, int max_slots, int n_visits, int keep_steps, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;
    const size_t dbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = static_cast<size_t>(v) * 3 * C;
    const size_t rbase = static_cast<size_t>(v) * 8 * C;
    for (int lag = 0; lag < 3; ++lag) {
        const int t = keep_steps - 3 + lag;
        pool_a[dbase + lag * C + c] = (t < 0) ? snap_a[sbase + (t + 3) * C + c] : rec_a[rbase + t * C + c];
        pool_b[dbase + lag * C + c] = (t < 0) ? snap_b[sbase + (t + 3) * C + c] : rec_b[rbase + t * C + c];
    }
}

// Decode any Section 1 tensor format into FP32 on host (exact match to ink_cpu_backend.cpp)
static std::vector<float> decode_sec1_f32(const uint8_t* sec1, const TInfo& ti) {
    const size_t n = static_cast<size_t>(ti.numel());
    std::vector<float> out(n, 0.0f);
    const uint8_t* src = sec1 + ti.off;
    if (ti.fmt == F_F32) {
        std::memcpy(out.data(), src, n * sizeof(float));
    } else if (ti.fmt == F_BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = ink2::bf16_to_f32(p[i]);
    } else if (ti.fmt == F_F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = ink2::fp16_to_f32(p[i]);
    } else if (ti.fmt == F_Q8ROW) {
        const size_t rows = static_cast<size_t>(ti.rows());
        const size_t cols = static_cast<size_t>(ti.cols());
        const float* scales = reinterpret_cast<const float*>(src);
        const int8_t* qs = reinterpret_cast<const int8_t*>(src + rows * sizeof(float));
        for (size_t r = 0; r < rows; ++r) {
            const float sc = scales[r];
            for (size_t c = 0; c < cols; ++c) out[r * cols + c] = static_cast<float>(qs[r * cols + c]) * sc;
        }
    } else {
        dequant_sk_row(src, static_cast<SkCode>(ti.fmt), out.data(), n);
    }
    return out;
}

static std::vector<uint16_t> f32_vec_to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = ink2::f32_to_bf16(v[i]);
    return out;
}

static bool hip_ok(hipError_t e, const char* ctx, std::string& err) {
    if (e == hipSuccess) return true;
    err = std::string("[ink-hip] ") + ctx + ": " + hipGetErrorString(e);
    return false;
}

// Growable on-demand GPU Paged KV Pool for ONE visit (SWA or Global) across all max_slots!
struct VisitKvPool {
    bool is_swa = true;
    bool is_fp8 = false;
    bool is_uma = false;
    bool host_offload = false;
    int max_slots = 1;
    int max_pages_per_slot = 16;
    int kv_dim = 1024;
    int cap_pages = 0;
    int used_pages = 0;
    uint64_t pool_gen = 0;

    std::vector<int32_t> free_pages;
    std::vector<int32_t> h_pt;  // [max_slots * max_pages_per_slot], -1 = unallocated
    bool pt_dirty = true;

    GpuBuffer d_pt;
    GpuBuffer k_pool;
    GpuBuffer v_pool;
    GpuBuffer k_scales;
    GpuBuffer v_scales;

    void release() {
        free_gpu_buffer(d_pt);
        free_gpu_buffer(k_pool);
        free_gpu_buffer(v_pool);
        free_gpu_buffer(k_scales);
        free_gpu_buffer(v_scales);
        free_pages.clear();
        h_pt.clear();
        cap_pages = used_pages = 0;
        host_offload = false;
    }

    bool init(bool swa, bool fp8, bool uma, int n_slots, int max_pages, int kvd, std::string& err) {
        release();
        is_swa = swa;
        is_fp8 = fp8;
        is_uma = uma;
        host_offload = false;
        max_slots = n_slots;
        max_pages_per_slot = std::max(1, max_pages);
        kv_dim = kvd;
        h_pt.assign(static_cast<size_t>(max_slots) * max_pages_per_slot, -1);
        pt_dirty = true;
        return hip_ok(alloc_gpu_buffer(d_pt, h_pt.size() * sizeof(int32_t), is_uma), "kv d_pt alloc", err);
    }

    void free_slot_pages(int slot) {
        if (slot < 0 || slot >= max_slots) return;
        int32_t* row = h_pt.data() + static_cast<size_t>(slot) * max_pages_per_slot;
        for (int p = 0; p < max_pages_per_slot; ++p) {
            if (row[p] >= 0) {
                free_pages.push_back(row[p]);
                row[p] = -1;
                pt_dirty = true;
            }
        }
    }

    bool try_alloc_pool_buffers(int cap, bool use_host_mapped, GpuBuffer& nk, GpuBuffer& nv, GpuBuffer& nks, GpuBuffer& nvs) {
        const size_t elem_b = is_fp8 ? 1u : 2u;
        const size_t page_b = static_cast<size_t>(kKvPageTokens) * kv_dim * elem_b;
        const size_t kv_b = static_cast<size_t>(cap) * page_b;
        if (alloc_gpu_buffer(nk, kv_b, use_host_mapped, /*prefer_uma_coherent=*/use_host_mapped) != hipSuccess) {
            return false;
        }
        if (alloc_gpu_buffer(nv, kv_b, use_host_mapped, /*prefer_uma_coherent=*/use_host_mapped) != hipSuccess) {
            free_gpu_buffer(nk);
            return false;
        }
        if (is_fp8) {
            const size_t sc_b = static_cast<size_t>(cap) * kKvPageTokens * sizeof(float);
            if (alloc_gpu_buffer(nks, sc_b, use_host_mapped, /*prefer_uma_coherent=*/use_host_mapped) != hipSuccess ||
                alloc_gpu_buffer(nvs, sc_b, use_host_mapped, /*prefer_uma_coherent=*/use_host_mapped) != hipSuccess) {
                free_gpu_buffer(nk); free_gpu_buffer(nv); free_gpu_buffer(nks); free_gpu_buffer(nvs);
                return false;
            }
        }
        return true;
    }

    bool grow_pool(int min_pages, hipStream_t st, std::string& err) {
        const int max_total_pages = std::max(min_pages, max_slots * max_pages_per_slot);
        const size_t elem_b = is_fp8 ? 1u : 2u;
        const size_t page_b = static_cast<size_t>(kKvPageTokens) * kv_dim * elem_b;

        int new_cap = std::min(max_total_pages, std::max(min_pages, std::max(16, cap_pages * 2)));
        size_t vfree = 0, vtot = 0;
        const bool have_vinfo = (!is_uma && !host_offload && hipMemGetInfo(&vfree, &vtot) == hipSuccess);
        if (have_vinfo) {
            // Avoid 3x doubling OOM spike on tight-VRAM 16 GB discrete GPUs (#6613 / #6621):
            // step conservatively when free VRAM < 1.5 GiB or when doubling exceeds 70% of remaining VRAM.
            const size_t need_bytes = static_cast<size_t>(new_cap) * page_b * 2u;
            if (vfree < (1536ULL << 20) || need_bytes > (vfree * 7u) / 10u) {
                const int step = (vfree < (512ULL << 20)) ? 8 : 32;
                new_cap = std::min(max_total_pages, std::max(min_pages, cap_pages + step));
            }
        }

        GpuBuffer nk, nv, nks, nvs;
        bool use_host = (is_uma || host_offload);
        if (!try_alloc_pool_buffers(new_cap, use_host, nk, nv, nks, nvs)) {
            // Retry with exact min_pages before falling back to host-mapped pinned KV.
            if (new_cap > min_pages) {
                new_cap = min_pages;
                if (!try_alloc_pool_buffers(new_cap, use_host, nk, nv, nks, nvs)) {
                    (void)hipGetLastError();
                }
            } else {
                (void)hipGetLastError();
            }
            if (!nk.dev_ptr && !use_host) {
                // VRAM exhausted: seamlessly fall back to host-mapped pinned KV (zero kernel changes, zero OOM!).
                use_host = true;
                new_cap = std::min(max_total_pages, std::max(min_pages, cap_pages + 32));
                if (!try_alloc_pool_buffers(new_cap, true, nk, nv, nks, nvs)) {
                    new_cap = min_pages;
                    if (!try_alloc_pool_buffers(new_cap, true, nk, nv, nks, nvs)) {
                        err = "[ink-hip] kv grow failed in both VRAM and host-mapped pinned memory";
                        return false;
                    }
                }
                host_offload = true;
                std::fprintf(stderr,
                             "[ink-hip] %s KV pool VRAM tight (free=%.0f MiB) -> spilled to host-mapped pinned KV (cap=%d pages, fp8=%d)\n",
                             is_swa ? "SWA" : "Global", have_vinfo ? (vfree / 1048576.0) : 0.0, new_cap, (int)is_fp8);
            } else if (!nk.dev_ptr) {
                err = "[ink-hip] kv grow allocation failed";
                return false;
            }
        }

        if (cap_pages > 0) {
            const size_t old_b = static_cast<size_t>(cap_pages) * page_b;
            if (!hip_ok(hipMemcpyAsync(nk.dev_ptr, k_pool.dev_ptr, old_b, hipMemcpyDefault, st), "kv copy k", err)) {
                free_gpu_buffer(nk); free_gpu_buffer(nv); free_gpu_buffer(nks); free_gpu_buffer(nvs);
                return false;
            }
            if (!hip_ok(hipMemcpyAsync(nv.dev_ptr, v_pool.dev_ptr, old_b, hipMemcpyDefault, st), "kv copy v", err)) {
                free_gpu_buffer(nk); free_gpu_buffer(nv); free_gpu_buffer(nks); free_gpu_buffer(nvs);
                return false;
            }
            if (is_fp8) {
                const size_t old_sc = static_cast<size_t>(cap_pages) * kKvPageTokens * sizeof(float);
                if (!hip_ok(hipMemcpyAsync(nks.dev_ptr, k_scales.dev_ptr, old_sc, hipMemcpyDefault, st), "kv copy ks", err) ||
                    !hip_ok(hipMemcpyAsync(nvs.dev_ptr, v_scales.dev_ptr, old_sc, hipMemcpyDefault, st), "kv copy vs", err)) {
                    free_gpu_buffer(nk); free_gpu_buffer(nv); free_gpu_buffer(nks); free_gpu_buffer(nvs);
                    return false;
                }
            }
            if (!hip_ok(hipStreamSynchronize(st), "kv grow sync", err)) {
                free_gpu_buffer(nk); free_gpu_buffer(nv); free_gpu_buffer(nks); free_gpu_buffer(nvs);
                return false;
            }
        }
        free_gpu_buffer(k_pool); free_gpu_buffer(v_pool);
        free_gpu_buffer(k_scales); free_gpu_buffer(v_scales);
        k_pool = nk; v_pool = nv; k_scales = nks; v_scales = nvs;
        cap_pages = new_cap;
        ++pool_gen;
        return true;
    }

    bool ensure_page(int slot, int logical_page, hipStream_t st, std::string& err) {
        if (logical_page < 0 || logical_page >= max_pages_per_slot) {
            err = "KV logical page out of range (" + std::to_string(logical_page) + " >= " + std::to_string(max_pages_per_slot) + ")";
            return false;
        }
        int32_t& entry = h_pt[static_cast<size_t>(slot) * max_pages_per_slot + logical_page];
        if (entry >= 0) return true;
        int phys = -1;
        if (!free_pages.empty()) {
            phys = free_pages.back();
            free_pages.pop_back();
        } else {
            if (used_pages >= cap_pages) {
                if (!grow_pool(used_pages + 1, st, err)) return false;
            }
            phys = used_pages++;
        }
        entry = phys;
        pt_dirty = true;
        return true;
    }

    bool sync_pt(hipStream_t st, std::string& err) {
        if (!pt_dirty) return true;
        pt_dirty = false;
        return hip_ok(hipMemcpyAsync(d_pt.dev_ptr, h_pt.data(), h_pt.size() * sizeof(int32_t),
                                     hipMemcpyHostToDevice, st),
                      "kv sync d_pt", err);
    }

    size_t allocated_bytes() const noexcept {
        return d_pt.bytes + k_pool.bytes + v_pool.bytes + k_scales.bytes + v_scales.bytes;
    }
};

struct GpuSlotMeta {
    bool allocated = false;
    int max_ctx = 0;
    int seq_len = 0;
    int snap_seq_len = 0;
    bool has_snap = false;
    // Per-slot ShortConv snapshot buffer on GPU (3.83 MiB) + 8-step verify input ring (5.1 MiB) for exact partial rollback
    GpuBuffer sconv_snap_k;
    GpuBuffer sconv_snap_v;
    GpuBuffer sconv_snap_attn;
    GpuBuffer sconv_snap_mlp;
    GpuBuffer sconv_rec_k;
    GpuBuffer sconv_rec_v;
    GpuBuffer sconv_rec_attn;
    GpuBuffer sconv_rec_mlp;
    int sconv_rec_n = 0;
    // Per-slot DSpark context KV (3 layers BF16) + f_rnn (BF16), grown on demand
    GpuBuffer dspark_k[3];
    GpuBuffer dspark_v[3];
    GpuBuffer dspark_frnn;
    int dspark_cap = 0;
    int dspark_len = 0;
    // Adaptive EMA DSpark acceptance tracker (zero regression on low-accept chat prompts!)
    float dspark_ema_acc = 0.50f;
    int dspark_last_draft = 0;
    int dspark_last_anchor = -1;
    int dspark_skip_rem = 0;
    bool dspark_had_rollback = false;
    // Wave-7D: Realized-gain adaptive tree cooldown (zero regression on high-entropy chat!)
    float tree_ema_gain = 1.25f;
    int tree_cool_rem = 0;

    void release() {
        free_gpu_buffer(sconv_snap_k);
        free_gpu_buffer(sconv_snap_v);
        free_gpu_buffer(sconv_snap_attn);
        free_gpu_buffer(sconv_snap_mlp);
        free_gpu_buffer(sconv_rec_k);
        free_gpu_buffer(sconv_rec_v);
        free_gpu_buffer(sconv_rec_attn);
        free_gpu_buffer(sconv_rec_mlp);
        sconv_rec_n = 0;
        for (int i = 0; i < 3; ++i) {
            free_gpu_buffer(dspark_k[i]);
            free_gpu_buffer(dspark_v[i]);
        }
        free_gpu_buffer(dspark_frnn);
        dspark_cap = dspark_len = 0;
        dspark_ema_acc = 0.50f;
        dspark_last_draft = 0;
        dspark_last_anchor = -1;
        dspark_skip_rem = 0;
        dspark_had_rollback = false;
        tree_ema_gain = 1.25f;
        tree_cool_rem = 0;
        allocated = false;
        has_snap = false;
        seq_len = snap_seq_len = 0;
    }
};

struct GpuPackedTensor {
    TInfo info;
    GpuBuffer buf;
};

struct GpuVisionState {
    bool loaded = false;
    VisionShardInfo info;
    std::unordered_map<std::string, GpuBuffer> w_bf16;
    std::unordered_map<std::string, GpuBuffer> w_f32;
    std::vector<float> h_cls;
    std::vector<float> h_cls_pos;
    std::vector<float> h_pos_emb;

    void release() {
        for (auto& kv : w_bf16) free_gpu_buffer(kv.second);
        for (auto& kv : w_f32) free_gpu_buffer(kv.second);
        w_bf16.clear();
        w_f32.clear();
        loaded = false;
    }
};

class HipBackend final : public InkBackend {
public:
    ~HipBackend() override { cleanup(); }

    const char* backend_name() const noexcept override { return ep2_active_ ? "hip-ep2" : "hip"; }

    int effective_max_ctx() const override { return max_ctx_; }

    bool dspark_ready() const override { return has_dspark_; }

    std::string device_info() const override {
        char buf[512];
        if (ep2_active_) {
            std::snprintf(buf, sizeof(buf),
                          "%s (%s, %d CUs) + %s (%s, %d CUs) [EP-2GPU split=%.2f, p2p=%s, eff=%.2f GiB, chunk<=%d, blasLt=%d, hybrid=%d]",
                          mem_info_.device_name, mem_info_.arch_name, mem_info_.cu_count,
                          peer_mem_info_.device_name, peer_mem_info_.arch_name, peer_mem_info_.cu_count,
                          ep_split_, ep2_p2p_ ? "on" : "pinned",
                          static_cast<double>(mem_info_.effective_total_bytes + peer_mem_info_.effective_total_bytes) / 1073741824.0,
                          guard_.max_prefill_chunk,
                          blaslt_.ready() ? 1 : 0,
                          hybrid_.active() ? 1 : 0);
        } else {
            std::snprintf(buf, sizeof(buf), "%s (%s, %d CUs, UMA=%d, eff=%.2f GiB, chunk<=%d, blasLt=%d, hybrid=%d)",
                          mem_info_.device_name, mem_info_.arch_name, mem_info_.cu_count,
                          mem_info_.is_uma ? 1 : 0,
                          static_cast<double>(mem_info_.effective_total_bytes) / 1073741824.0,
                          guard_.max_prefill_chunk,
                          blaslt_.ready() ? 1 : 0,
                          hybrid_.active() ? 1 : 0);
        }
        return std::string(buf);
    }

    BackendStats stats() const override {
        BackendStats st;
        st.vram_experts_loaded = vram_experts_loaded_;
        st.ram_experts_loaded = ram_experts_loaded_;
        st.total_experts = !adj_dir_.empty() ? adj_dir_.size() : (model_ ? model_->expert_directory().size() : 0);
        st.vram_used_bytes = vram_used_bytes() + (ep2_active_ ? peer_expert_gpu_arena_.bytes : 0);
        st.vram_total_bytes = mem_info_.effective_total_bytes + (ep2_active_ ? peer_mem_info_.effective_total_bytes : 0);
        st.max_slots = max_slots_;
        for (const auto& s : slots_) if (s.allocated) ++st.active_slots;
        st.kv_dtype = default_kv_dtype_;
        return st;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        cleanup();
#if defined(__linux__)
        if (const std::string nb = strata::platform::numa_balancing::keep_scanner_off(); !nb.empty()) {
            std::fprintf(stderr, "%s\n", nb.c_str());
        }
#endif
        model_ = &model;
        opts_ = opts;
        c_ = model.m.c;
        dspark_cfg_ = model.m.dspark;

        int dev_count = 0;
        if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count < 1) {
            err = "[ink-hip] no HIP devices visible";
            return false;
        }
        gpu0_id_ = (!opts.gpu_ids.empty()) ? opts.gpu_ids[0] : opts.gpu_index;
        gpu0_id_ = std::max(0, std::min(gpu0_id_, dev_count - 1));
        gpu1_id_ = -1;
        ep2_active_ = false;
        ep2_p2p_ = false;
        ep_split_ = std::clamp(opts.ep_split, 0.1, 0.9);

        if (!hip_ok(hipSetDevice(gpu0_id_), "hipSetDevice", err)) return false;
        if (!hip_ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate", err)) return false;

        mem_info_ = query_device_memory(gpu0_id_);
#if defined(__linux__)
        if (mem_info_.is_uma) {
            if (std::FILE* osf = std::fopen("/proc/sys/kernel/osrelease", "r")) {
                char osrel[256] = {0};
                if (std::fgets(osrel, sizeof(osrel), osf) &&
                    (std::strstr(osrel, "microsoft") || std::strstr(osrel, "Microsoft") || std::strstr(osrel, "WSL"))) {
                    std::fprintf(stderr,
                                 "[ink-hip] WARNING: WSL2 on integrated Ryzen APU (%s) hits ROCm 10.2 #10097 aperture faults; "
                                 "use native Windows HIP or bounded bounce buffers\n",
                                 mem_info_.arch_name);
                }
                std::fclose(osf);
            }
        }
#endif

        // Dual-GPU Expert Parallelism setup (--gpus 2 / --gpus 0,1 / --devices 0,1)
        if ((opts.gpus >= 2 || opts.gpu_ids.size() >= 2) && dev_count >= 2) {
            int g1 = -1;
            if (opts.gpu_ids.size() >= 2) {
                for (size_t k = 1; k < opts.gpu_ids.size(); ++k) {
                    if (opts.gpu_ids[k] >= 0 && opts.gpu_ids[k] < dev_count && opts.gpu_ids[k] != gpu0_id_) {
                        g1 = opts.gpu_ids[k];
                        break;
                    }
                }
            }
            if (g1 < 0) g1 = (gpu0_id_ == 0) ? 1 : 0;
            if (g1 >= 0 && g1 < dev_count && g1 != gpu0_id_) {
                gpu1_id_ = g1;
                if (!hip_ok(hipSetDevice(gpu1_id_), "hipSetDevice peer", err)) return false;
                peer_mem_info_ = query_device_memory(gpu1_id_);
                if (!hip_ok(hipStreamCreateWithFlags(&peer_stream_, hipStreamNonBlocking), "hipStreamCreate peer", err) ||
                    !hip_ok(hipEventCreateWithFlags(&ev_g1_done_, hipEventDisableTiming), "hipEventCreate peer", err)) {
                    (void)hipSetDevice(gpu0_id_);
                    return false;
                }
                if (!hip_ok(hipSetDevice(gpu0_id_), "hipSetDevice restore0", err) ||
                    !hip_ok(hipEventCreateWithFlags(&ev_g0_to_g1_, hipEventDisableTiming), "hipEventCreate g0", err)) {
                    return false;
                }
                // Probe P2P with a 16-byte hardware roundtrip check; fall back to host-pinned bounce if unsupported or broken!
                const char* force_pinned = std::getenv("INK_HIP_PEER_PINNED");
                if (!(force_pinned && force_pinned[0] == '1')) {
                    int p01 = 0, p10 = 0;
                    if (hipDeviceCanAccessPeer(&p01, gpu0_id_, gpu1_id_) == hipSuccess &&
                        hipDeviceCanAccessPeer(&p10, gpu1_id_, gpu0_id_) == hipSuccess &&
                        p01 && p10) {
                        (void)hipSetDevice(gpu0_id_);
                        hipError_t e1 = hipDeviceEnablePeerAccess(gpu1_id_, 0);
                        if (e1 == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); e1 = hipSuccess; }
                        (void)hipSetDevice(gpu1_id_);
                        hipError_t e2 = hipDeviceEnablePeerAccess(gpu0_id_, 0);
                        if (e2 == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); e2 = hipSuccess; }
                        if (e1 == hipSuccess && e2 == hipSuccess) {
                            uint32_t h_tx[4] = {0x12345678u, 0xCAFEBABEu, 0xA5A55A5Au, 0xDEADBEEFu};
                            uint32_t h_rx01[4] = {0, 0, 0, 0};
                            uint32_t h_rx10[4] = {0, 0, 0, 0};
                            uint32_t *d0 = nullptr, *d1 = nullptr;
                            (void)hipSetDevice(gpu0_id_);
                            hipError_t a0 = hipMalloc(&d0, sizeof(h_tx));
                            (void)hipSetDevice(gpu1_id_);
                            hipError_t a1 = hipMalloc(&d1, sizeof(h_tx));
                            bool hw_ok = false;
                            if (a0 == hipSuccess && a1 == hipSuccess) {
                                (void)hipSetDevice(gpu0_id_);
                                (void)hipMemcpy(d0, h_tx, sizeof(h_tx), hipMemcpyHostToDevice);
                                (void)hipSetDevice(gpu1_id_);
                                (void)hipMemset(d1, 0, sizeof(h_tx));
                                hipError_t c01 = hipMemcpyPeer(d1, gpu1_id_, d0, gpu0_id_, sizeof(h_tx));
                                (void)hipMemcpy(h_rx01, d1, sizeof(h_tx), hipMemcpyDeviceToHost);
                                (void)hipSetDevice(gpu0_id_);
                                (void)hipMemset(d0, 0, sizeof(h_tx));
                                hipError_t c10 = hipMemcpyPeer(d0, gpu0_id_, d1, gpu1_id_, sizeof(h_tx));
                                (void)hipMemcpy(h_rx10, d0, sizeof(h_tx), hipMemcpyDeviceToHost);
                                hw_ok = (c01 == hipSuccess && c10 == hipSuccess &&
                                         std::memcmp(h_tx, h_rx01, sizeof(h_tx)) == 0 &&
                                         std::memcmp(h_tx, h_rx10, sizeof(h_tx)) == 0);
                            }
                            if (d0) { (void)hipSetDevice(gpu0_id_); (void)hipFree(d0); }
                            if (d1) { (void)hipSetDevice(gpu1_id_); (void)hipFree(d1); }
                            if (hw_ok) {
                                ep2_p2p_ = true;
                            } else {
                                std::fprintf(stderr,
                                             "[ink-hip] Hardware P2P roundtrip probe FAILED (GPU%d<->GPU%d); using pinned-host DMA fallback\n",
                                             gpu0_id_, gpu1_id_);
                                (void)hipSetDevice(gpu0_id_);
                                if (hipDeviceDisablePeerAccess(gpu1_id_) != hipSuccess) (void)hipGetLastError();
                                (void)hipSetDevice(gpu1_id_);
                                if (hipDeviceDisablePeerAccess(gpu0_id_) != hipSuccess) (void)hipGetLastError();
                            }
                        }
                    }
                }
                (void)hipSetDevice(gpu0_id_);
                ep2_active_ = true;
            }
        }
        guard_ = compute_dispatch_guard(mem_info_.arch_name, mem_info_.cu_count, mem_info_.is_uma,
#if defined(_WIN32)
                                        true
#else
                                        false
#endif
        );

        // Auto-select FP8_E4M3 KV when high concurrency (>= 16) on <= 32 GiB GPU OR on <= 13 GiB discrete GPUs (8/10/12 GB)
        default_kv_dtype_ = opts.kv_dtype;
        if (opts.kv_dtype_auto &&
            ((opts.parallel >= 16 && mem_info_.effective_total_bytes <= 36ULL * 1073741824ULL) ||
             (!mem_info_.is_uma && mem_info_.effective_total_bytes <= 13ULL * 1073741824ULL))) {
            default_kv_dtype_ = KvDtype::FP8_E4M3;
        }
        max_slots_ = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        max_ctx_ = std::max(256, opts.max_ctx);

        // Initialize dynamic hipBLASLt Tensile runner (RDNA3/3.5/4; cleanly falls back on RDNA2 or missing DLL/SO)
        (void)blaslt_.init(mem_info_.is_uma, mem_info_.gen);
        blaslt_.warmup(mem_info_.is_uma, stream_);
        // M >= 8 unlocks hipBLASLt for B=8 DSpark verify, DSpark 3L drafter, and c=8..64 batching!
        blaslt_min_m_ = 8;
        if (const char* ev = std::getenv("STRATA_HIPBLASLT_MIN_M")) {
            const int v = std::atoi(ev);
            if (v >= 1) blaslt_min_m_ = v;
        }
        // Fused zero-DRAM-write 2-launch INT8/FP16 WMMA Grouped-MoE beats unpack-once on both R9700 & Strix Halo
        // (10x less memory traffic, 300x fewer launches!). Keep unpack-once env-tunable via STRATA_HIPBLASLT_MOE_MIN_B.
        blaslt_moe_min_b_ = 999999;
        if (const char* ev = std::getenv("STRATA_HIPBLASLT_MOE_MIN_B")) {
            const int v = std::atoi(ev);
            if (v >= 16) blaslt_moe_min_b_ = v;
        }
        // RDNA4 (gfx120x) has true 2x-rate 16x16x32 iu8 WMMA; RDNA3/3.5 (gfx11xx) wins with double-buffered FP16 WMMA!
        use_i8_moe_wmma_ = (mem_info_.gen == AmdGen::Rdna4);
        if (const char* ev = std::getenv("STRATA_HIP_MOE_I8")) {
            use_i8_moe_wmma_ = (std::atoi(ev) != 0);
        }
        if (blaslt_.ready() && blaslt_moe_min_b_ < 999999) {
            const size_t w13_b = static_cast<size_t>(2 * c_.inter) * c_.hidden * sizeof(half_t);
            const size_t w2_b  = static_cast<size_t>(c_.hidden) * c_.inter * sizeof(half_t);
            (void)alloc_gpu_buffer(d_exp_w13_f16_, w13_b, mem_info_.is_uma);
            (void)alloc_gpu_buffer(d_exp_w2_f16_,  w2_b,  mem_info_.is_uma);
        }

        // 1. Load Section 1 trunk + DSpark arena (supports .strata v1.1, .gguf, and .safetensors!)
        std::vector<uint8_t> sec1;
        if (!model.read_section1(sec1, err)) return false;

        const std::string embed_k = "model.llm.embed.weight";
        const std::string unemb_k = "model.llm.unembed.weight";

        for (const auto& kv : model.m.trunk) {
            const std::string& name = kv.first;
            const TInfo& ti = kv.second;
            if (name == embed_k || name == unemb_k) {
                GpuPackedTensor pt;
                pt.info = ti;
                if (!upload_raw(pt.buf, sec1.data() + ti.off, ti.bytes, err)) return false;
                packed_trunk_[name] = pt;
            } else if (is_large_2d_matmul(name, ti)) {
                auto f32 = decode_sec1_f32(sec1.data(), ti);
                auto bf16 = f32_vec_to_bf16(f32);
                GpuBuffer b;
                if (!upload_raw(b, bf16.data(), bf16.size() * sizeof(uint16_t), err)) return false;
                trunk_bf16_[name] = b;
            } else {
                auto f32 = decode_sec1_f32(sec1.data(), ti);
                // Keep small host copy for cross-pass queries and static ShortConv preprocessing
                if (name == "model.llm.middle_cycle_cross_pass_queries" ||
                    name == "model.llm.middle_cycle_cross_pass_kv_queries" ||
                    name.find("sconv") != std::string::npos) {
                    host_tmp_[name] = f32;
                }
                GpuBuffer b;
                if (!upload_raw(b, f32.data(), f32.size() * sizeof(float), err)) return false;
                trunk_f32_[name] = b;
                if (name.find(".mlp.gate.weight") != std::string::npos) {
                    auto bf16 = f32_vec_to_bf16(f32);
                    GpuBuffer gb;
                    if (!upload_raw(gb, bf16.data(), bf16.size() * sizeof(uint16_t), err)) return false;
                    trunk_bf16_[name] = gb;
                }
            }
        }

        // Precompute stacked cross-pass msg_queries [4, D] per core layer -> upload to GPU
        {
            const auto& q_cp = host_tmp_.at("model.llm.middle_cycle_cross_pass_queries");
            const auto& kv_cp = host_tmp_.at("model.llm.middle_cycle_cross_pass_kv_queries");
            const int core_len = c_.core_end - c_.core_start;
            d_msg_queries_.resize(c_.n_phys);
            for (int i = 0; i < core_len; ++i) {
                std::vector<float> mq(4u * c_.hidden);
                std::memcpy(mq.data() + 0 * c_.hidden, q_cp.data()  + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 1 * c_.hidden, kv_cp.data() + (2 * i + 0) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 2 * c_.hidden, kv_cp.data() + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
                std::memcpy(mq.data() + 3 * c_.hidden, q_cp.data()  + (2 * i + 1) * c_.hidden, c_.hidden * sizeof(float));
                if (!upload_raw(d_msg_queries_[c_.core_start + i], mq.data(), mq.size() * sizeof(float), err)) return false;
            }
        }

        // Pre-normalize static ShortConv weights to [C, 4] (lag 3..0) -> upload to GPU
        for (int il = 0; il < c_.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            auto upload_sc = [&](const std::string& key, const std::vector<float>& vec) -> bool {
                GpuBuffer b;
                if (!upload_raw(b, vec.data(), vec.size() * sizeof(float), err)) return false;
                static_sconv_[key] = b;
                return true;
            };
            if (!upload_sc(p + ".attn_sconv", host_tmp_.at(p + ".attn_sconv.weight"))) return false;
            if (!upload_sc(p + ".mlp_sconv",  host_tmp_.at(p + ".mlp_sconv.weight"))) return false;
            if (c_.is_swa[il]) {
                if (!upload_sc(p + ".k_sconv", host_tmp_.at(p + ".attn.k_sconv.weight"))) return false;
                if (!upload_sc(p + ".v_sconv", host_tmp_.at(p + ".attn.v_sconv.weight"))) return false;
            } else {
                const int kv_dim = c_.n_kv * c_.head_dim;
                const auto& kw = host_tmp_.at(p + ".attn.k_sconv.static_w");
                const auto& vw = host_tmp_.at(p + ".attn.v_sconv.static_w");
                std::vector<float> kt(kv_dim * 4), vt(kv_dim * 4);
                for (int c = 0; c < kv_dim; ++c) {
                    for (int lag = 0; lag < 4; ++lag) {
                        kt[c * 4 + (3 - lag)] = kw[lag * kv_dim + c];
                        vt[c * 4 + (3 - lag)] = vw[lag * kv_dim + c];
                    }
                }
                if (!upload_sc(p + ".k_sconv", kt)) return false;
                if (!upload_sc(p + ".v_sconv", vt)) return false;
            }
        }
        host_tmp_.clear();

        // Concatenate [wq(q_dim); wr(r_dim); gate_a(q_dim); wk(kv_dim); wv(kv_dim)] per layer into 1 contiguous BF16 matrix
        // (0 extra VRAM — frees the 5 individual BF16 buffers immediately -> 1 GEMM instead of 5 per visit!)
        trunk_qkvrg_.resize(c_.n_phys);
        for (int il = 0; il < c_.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il) + ".attn.";
            const std::string ks[5] = {
                p + "wq_du.weight", p + "wr_du.weight", p + "gate_attn.weight",
                p + "wk_dv.weight", p + "wv_dv.weight"
            };
            bool all_bf = true;
            size_t tot_b = 0;
            for (int j = 0; j < 5; ++j) {
                auto it = trunk_bf16_.find(ks[j]);
                if (it == trunk_bf16_.end()) { all_bf = false; break; }
                tot_b += it->second.bytes;
            }
            if (all_bf && tot_b > 0) {
                GpuBuffer cat;
                if (alloc_gpu_buffer(cat, tot_b, mem_info_.is_uma) == hipSuccess) {
                    size_t off = 0;
                    for (int j = 0; j < 5; ++j) {
                        auto& b = trunk_bf16_.at(ks[j]);
                        (void)hipMemcpy(cat.dev_ptr + off, b.dev_ptr, b.bytes, hipMemcpyDeviceToDevice);
                        off += b.bytes;
                    }
                    for (int j = 0; j < 5; ++j) {
                        free_gpu_buffer(trunk_bf16_.at(ks[j]));
                        trunk_bf16_.erase(ks[j]);
                    }
                    trunk_qkvrg_[il] = cat;
                }
            }
        }

        // 2. Load embedded DSpark tensors to GPU if present & enabled
        has_dspark_ = opts.enable_dspark && dspark_cfg_.present;
        if (has_dspark_) {
            for (const auto& kv : model.m.dspark_tensors) {
                std::string k = kv.first;
                if (k.rfind("dspark.", 0) == 0) k = k.substr(7);
                auto f32 = decode_sec1_f32(sec1.data(), kv.second);
                if (k == "fuse_weight" || k.find("fuse") != std::string::npos) {
                    h_dspark_fuse_ = f32;
                }
                const bool store_bf16 = (kv.second.shape.size() == 2) &&
                    (k.find("layers.") == 0 ||
                     k == "markov_head.rnn_proj.weight" ||
                     k == "markov_head.hidden_out.weight");
                if (store_bf16) {
                    auto bf16 = f32_vec_to_bf16(f32);
                    GpuBuffer b;
                    if (!upload_raw(b, bf16.data(), bf16.size() * sizeof(uint16_t), err)) return false;
                    dspark_bf16_[k] = b;
                } else {
                    GpuBuffer b;
                    if (!upload_raw(b, f32.data(), f32.size() * sizeof(float), err)) return false;
                    dspark_f32_[k] = b;
                }
            }
            if (!hip_ok(alloc_gpu_buffer(d_dspark_state_, static_cast<size_t>(dspark_cfg_.markov_rank) * 4, mem_info_.is_uma), "dspark state", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_state2_, static_cast<size_t>(dspark_cfg_.markov_rank) * 4, mem_info_.is_uma), "dspark state2", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_latent_, static_cast<size_t>(dspark_cfg_.markov_rank) * 4, mem_info_.is_uma), "dspark latent", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_corr_, static_cast<size_t>(dspark_cfg_.hidden) * 4, mem_info_.is_uma), "dspark corr", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_toks_, 16 * sizeof(int32_t), mem_info_.is_uma), "dspark toks", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_blk_v_, 1024 * sizeof(float), mem_info_.is_uma), "dspark blkv", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_blk_i_, 1024 * sizeof(int32_t), mem_info_.is_uma), "dspark blki", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_cand_v_, 7u * 1024u * sizeof(float), mem_info_.is_uma), "dspark candv", err) ||
                !hip_ok(alloc_gpu_buffer(d_dspark_cand_i_, 7u * 1024u * sizeof(int32_t), mem_info_.is_uma), "dspark candi", err)) {
                return false;
            }
            // Concatenate DSpark [q_proj; k_proj; v_proj] and [gate_proj; up_proj] per layer (0 extra VRAM -> 1 GEMM instead of 3/2!)
            dspark_qkv_.resize(dspark_cfg_.n_layers);
            dspark_gu_.resize(dspark_cfg_.n_layers);
            for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
                const std::string p = "layers." + std::to_string(il);
                const std::string qk[3] = {
                    p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"
                };
                if (dspark_bf16_.count(qk[0]) && dspark_bf16_.count(qk[1]) && dspark_bf16_.count(qk[2])) {
                    const size_t tb = dspark_bf16_[qk[0]].bytes + dspark_bf16_[qk[1]].bytes + dspark_bf16_[qk[2]].bytes;
                    GpuBuffer cat;
                    if (alloc_gpu_buffer(cat, tb, mem_info_.is_uma) == hipSuccess) {
                        size_t off = 0;
                        for (int j = 0; j < 3; ++j) {
                            auto& b = dspark_bf16_.at(qk[j]);
                            (void)hipMemcpy(cat.dev_ptr + off, b.dev_ptr, b.bytes, hipMemcpyDeviceToDevice);
                            off += b.bytes;
                            free_gpu_buffer(b);
                            dspark_bf16_.erase(qk[j]);
                        }
                        dspark_qkv_[il] = cat;
                    }
                }
                const std::string gk[2] = { p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight" };
                if (dspark_bf16_.count(gk[0]) && dspark_bf16_.count(gk[1])) {
                    const size_t tb = dspark_bf16_[gk[0]].bytes + dspark_bf16_[gk[1]].bytes;
                    GpuBuffer cat;
                    if (alloc_gpu_buffer(cat, tb, mem_info_.is_uma) == hipSuccess) {
                        size_t off = 0;
                        for (int j = 0; j < 2; ++j) {
                            auto& b = dspark_bf16_.at(gk[j]);
                            (void)hipMemcpy(cat.dev_ptr + off, b.dev_ptr, b.bytes, hipMemcpyDeviceToDevice);
                            off += b.bytes;
                            free_gpu_buffer(b);
                            dspark_bf16_.erase(gk[j]);
                        }
                        dspark_gu_[il] = cat;
                    }
                }
            }
        }

        // 3. Load Section 4 experts in 3x loop-weighted vram_load_order priority!
        // Uses strata::hip_uma::alloc_gpu_buffer so 780M (gfx1103) & 890M (gfx1150) with small 512M-4G BIOS carveout
        // seamlessly allocate coherent GTT UMA memory and fit SQ-10 (9.75 GiB) & SQ-15 (13.36 GiB) 100% on GPU!
        // For community .gguf (GGUF_RAW), preserve native lossless GGUF_RAW bytes for all 14 supported GGML types
        // (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ3_XXS/IQ4_NL/Q4_0/Q3_K/Q2_K/MXFP4/BF16/F16/F32) for zero transcode overhead
        // and zero double-quant loss on VRAM, Peer GPU, AND RAM-spilled hybrid experts!
        const bool force_sk_tc = (std::getenv("INK_HIP_FORCE_SK_TRANSCODE") && std::atoi(std::getenv("INK_HIP_FORCE_SK_TRANSCODE")) != 0);
        adj_dir_ = model.expert_directory();
        expert_native_gguf_.assign(adj_dir_.size(), 0);
        size_t n_native_gguf = 0;
        if (model.kind == ModelSourceKind::Gguf && model.gguf_adapter) {
            const size_t hidden_sz = static_cast<size_t>(c_.hidden);
            const size_t inter_sz  = static_cast<size_t>(c_.inter);
            for (size_t idx = 0; idx < adj_dir_.size(); ++idx) {
                auto& e = adj_dir_[idx];
                if (e.gu_code_id != CODE_GGUF_RAW && e.dn_code_id != CODE_GGUF_RAW) continue;
                const int il = static_cast<int>(idx) / c_.n_experts;
                if (il < 0 || il >= static_cast<int>(model.gguf_adapter->layers.size())) continue;
                const auto& lm = model.gguf_adapter->layers[il];
                if (!force_sk_tc && gguf_type_has_hip_kernel(lm.gu_type) && gguf_type_has_hip_kernel(lm.dn_type)) {
                    expert_native_gguf_[idx] = 1;
                    ++n_native_gguf;
                } else {
                    const SkCode gu_c = transcode::choose_sk_code_for_ggml(lm.gu_type);
                    const SkCode dn_c = transcode::choose_sk_code_for_ggml(lm.dn_type);
                    uint32_t dn_off = 0, body_b = 0;
                    transcode::transcoded_sk_expert_layout(gu_c, dn_c, hidden_sz, inter_sz, dn_off, body_b);
                    e.gu_code_id = static_cast<uint8_t>(gu_c);
                    e.dn_code_id = static_cast<uint8_t>(dn_c);
                    e.dn_offset  = dn_off;
                    e.body_bytes = body_b;
                }
            }
            if (n_native_gguf > 0) {
                std::fprintf(stderr, "[ink-hip] native lossless GGUF_RAW GPU kernels active for %zu/%zu experts (zero S_K transcode)\n",
                             n_native_gguf, adj_dir_.size());
            }
        }
        const auto& dir = adj_dir_;
        const size_t n_dir = dir.size();
        expert_gpu_offset_.assign(n_dir, static_cast<size_t>(-1));
        expert_ram_bodies_.resize(n_dir);
        expert_sf4_outer_.assign(n_dir, {1.0f, 1.0f});

        size_t vram_budget = 0;
        if (opts.vram_experts_gb >= 0.0) {
            vram_budget = static_cast<size_t>(opts.vram_experts_gb * 1073741824.0);
        } else {
            const size_t trunk_now = vram_used_bytes();
            // Reserve headroom for KV + activations + OS/desktop (always keep at least OS reserve + 768 MiB)
            const size_t kv_est = std::min<size_t>(
                estimate_slot_bytes(std::min(max_ctx_, 4096), default_kv_dtype_, has_dspark_).total_bytes() * static_cast<size_t>(max_slots_),
                mem_info_.effective_total_bytes / 4);
            const size_t overhead = mem_info_.reserve_bytes + trunk_now + kv_est + (768ULL << 20);
            if (mem_info_.effective_total_bytes > overhead) {
                vram_budget = mem_info_.effective_total_bytes - overhead;
            }
        }

        // Check whether all experts fit in vram_budget; if hybrid (partial VRAM), reserve elastic VRAM cache inside budget!
        size_t total_all_exp_bytes = 0;
        size_t max_exp_bytes = 0;
        for (uint16_t idx : model.vram_load_order) {
            if (idx >= n_dir) continue;
            const size_t aligned = (dir[idx].body_bytes + 63u) & ~size_t(63u);
            total_all_exp_bytes += aligned;
            max_exp_bytes = std::max(max_exp_bytes, aligned);
        }
        size_t hybrid_cache_budget = 0;
        size_t static_vram_budget = vram_budget;
        if (total_all_exp_bytes > vram_budget && max_exp_bytes > 0) {
            const size_t stage2_b = 2u * max_exp_bytes;
            if (static_vram_budget > stage2_b + 16u * max_exp_bytes) {
                hybrid_cache_budget = std::min<size_t>(256ULL << 20, std::max<size_t>(16u * max_exp_bytes, static_vram_budget / 10));
                static_vram_budget -= (hybrid_cache_budget + stage2_b);
            } else if (static_vram_budget > stage2_b) {
                static_vram_budget -= stage2_b;
            }
        }

        // Plan contiguous GPU arena(s) for all experts fitting in GPU 0 (+ GPU 1 when Dual-GPU EP is active)
        peer_expert_gpu_offset_.assign(n_dir, static_cast<size_t>(-1));
        size_t peer_vram_budget = 0;
        if (ep2_active_) {
            const size_t peer_oh = peer_mem_info_.reserve_bytes + (768ULL << 20);
            if (peer_mem_info_.effective_total_bytes > peer_oh) {
                peer_vram_budget = peer_mem_info_.effective_total_bytes - peer_oh;
            }
            if (opts.vram_experts_gb >= 0.0) {
                peer_vram_budget = std::min(peer_vram_budget, static_cast<size_t>(opts.vram_experts_gb * 1073741824.0));
            }
        }
        size_t arena_bytes = 0;
        size_t peer_arena_bytes = 0;
        std::vector<uint16_t> gpu_plan;
        std::vector<uint16_t> peer_gpu_plan;
        std::vector<uint16_t> ram_plan;
        size_t ram_cap = (opts.ssd_experts || opts.ram_experts_gb == 0.0)
            ? 0u
            : ((opts.ram_experts_gb > 0.0)
                ? static_cast<size_t>(strata::platform::cgroup_mem::gib_to_bytes(opts.ram_experts_gb))
                : static_cast<size_t>(-1));
        if (opts.ram_experts_gb < 0.0 && !opts.ssd_experts) {
            // Auto-cap RAM experts to safe available host memory so cold SSD streaming activates automatically on tight-RAM hosts!
            uint64_t host_avail = 0;
#ifdef __linux__
            if (FILE* fp = std::fopen("/proc/meminfo", "r")) {
                char key[64];
                unsigned long long kb = 0;
                char unit[32];
                while (std::fscanf(fp, "%63[^:]: %llu %31s\n", key, &kb, unit) == 3) {
                    if (std::strcmp(key, "MemAvailable") == 0) { host_avail = kb * 1024ull; break; }
                }
                std::fclose(fp);
            }
#endif
            if (host_avail > 0) {
                const uint64_t os_res = std::max<uint64_t>(2ULL << 30, std::min<uint64_t>(host_avail / 4, 4ULL << 30));
                const size_t safe_host = static_cast<size_t>(host_avail > os_res ? (host_avail - os_res) : 0);
                if (ram_cap > safe_host) ram_cap = safe_host;
            }
        }
        {
            const auto cg = strata::platform::cgroup_mem::probe_cgroup_room(strata::platform::cgroup_mem::kUnlimited);
            if (cg.limited && cg.available_bytes != strata::platform::cgroup_mem::kUnlimited && cg.available_bytes > 0) {
                const uint64_t reserve = std::min<uint64_t>(cg.available_bytes / 8, 2ULL << 30);
                const size_t cg_cap = static_cast<size_t>(
                    cg.available_bytes > reserve ? cg.available_bytes - reserve : 0);
                if (ram_cap > cg_cap) ram_cap = cg_cap;
            }
        }
        size_t ram_bytes = 0;

        for (uint16_t idx : model.vram_load_order) {
            if (idx >= n_dir) continue;
            const size_t sz = dir[idx].body_bytes;
            const size_t aligned = (sz + 63u) & ~size_t(63u);
            if (ep2_active_) {
                const int il = static_cast<int>(idx) / c_.n_experts;
                const int eid = static_cast<int>(dir[idx].expert_id);
                const uint32_t hh = (static_cast<uint32_t>(eid) * 2654435761u + static_cast<uint32_t>(il) * 40503u) >> 8;
                const int pref = ((hh % 10000u) < static_cast<uint32_t>(ep_split_ * 10000.0)) ? 1 : 0;
                if (pref == 1 && peer_arena_bytes + aligned <= peer_vram_budget) {
                    peer_expert_gpu_offset_[idx] = peer_arena_bytes;
                    peer_arena_bytes += aligned;
                    peer_gpu_plan.push_back(idx);
                } else if (arena_bytes + aligned <= static_vram_budget) {
                    expert_gpu_offset_[idx] = arena_bytes;
                    arena_bytes += aligned;
                    gpu_plan.push_back(idx);
                } else if (peer_arena_bytes + aligned <= peer_vram_budget) {
                    peer_expert_gpu_offset_[idx] = peer_arena_bytes;
                    peer_arena_bytes += aligned;
                    peer_gpu_plan.push_back(idx);
                } else if (ram_bytes + sz <= ram_cap) {
                    ram_bytes += sz;
                    ram_plan.push_back(idx);
                }
            } else {
                if (arena_bytes + aligned <= static_vram_budget) {
                    expert_gpu_offset_[idx] = arena_bytes;
                    arena_bytes += aligned;
                    gpu_plan.push_back(idx);
                } else if (ram_bytes + sz <= ram_cap) {
                    ram_bytes += sz;
                    ram_plan.push_back(idx);
                }
            }
        }
        if (ep2_active_ && peer_gpu_plan.empty()) {
            ep2_active_ = false;
        }

        const int tc_threads = transcode::default_transcode_threads();
        if (arena_bytes > 0) {
            if (!hip_ok(alloc_gpu_buffer(expert_gpu_arena_, arena_bytes, mem_info_.is_uma), "expert GPU arena alloc", err)) {
                return false;
            }
            if (model.kind == ModelSourceKind::Strata) {
                std::vector<uint8_t> tmp;
                for (uint16_t idx : gpu_plan) {
                    if (!model.rd.read_expert_body(idx, tmp, err)) return false;
                    if ((dir[idx].gu_code_id == CODE_SF4 || dir[idx].dn_code_id == CODE_SF4) && tmp.size() >= 16) {
                        float hdr[4];
                        std::memcpy(hdr, tmp.data(), 16);
                        expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                    }
                    if (!hip_ok(hipMemcpy(expert_gpu_arena_.dev_ptr + expert_gpu_offset_[idx], tmp.data(),
                                          tmp.size(), hipMemcpyHostToDevice),
                                "expert H2D", err)) {
                        return false;
                    }
                }
            } else {
                // Multi-threaded streaming transcode in bounded 16-expert batches (< 128 MiB scratch, 0 temp files)
                const size_t kBatch = 16;
                for (size_t b0 = 0; b0 < gpu_plan.size(); b0 += kBatch) {
                    const size_t bn = std::min(kBatch, gpu_plan.size() - b0);
                    std::vector<std::vector<uint8_t>> bufs(bn);
                    std::vector<uint32_t> dn_offs(bn, 0);
                    std::vector<uint8_t> gu_c(bn, 0), dn_c(bn, 0);
                    bool ok = transcode::parallel_run(bn, tc_threads, [&](size_t j, std::string& lerr) -> bool {
                        return read_expert_for_hip(gpu_plan[b0 + j], bufs[j], dn_offs[j], gu_c[j], dn_c[j], lerr);
                    }, err);
                    if (!ok) return false;
                    for (size_t j = 0; j < bn; ++j) {
                        const uint16_t idx = gpu_plan[b0 + j];
                        if ((gu_c[j] == CODE_SF4 || dn_c[j] == CODE_SF4) && bufs[j].size() >= 16) {
                            float hdr[4];
                            std::memcpy(hdr, bufs[j].data(), 16);
                            expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                        }
                        if (!hip_ok(hipMemcpy(expert_gpu_arena_.dev_ptr + expert_gpu_offset_[idx], bufs[j].data(),
                                              bufs[j].size(), hipMemcpyHostToDevice),
                                    "expert H2D", err)) {
                            return false;
                        }
                    }
                }
            }
        }
        if (ep2_active_ && peer_arena_bytes > 0) {
            if (!hip_ok(hipSetDevice(gpu1_id_), "hipSetDevice peer arena", err)) return false;
            if (!hip_ok(alloc_gpu_buffer(peer_expert_gpu_arena_, peer_arena_bytes, peer_mem_info_.is_uma), "peer expert GPU arena alloc", err)) {
                (void)hipSetDevice(gpu0_id_);
                return false;
            }
            if (model.kind == ModelSourceKind::Strata) {
                std::vector<uint8_t> tmp;
                for (uint16_t idx : peer_gpu_plan) {
                    if (!model.rd.read_expert_body(idx, tmp, err)) {
                        (void)hipSetDevice(gpu0_id_);
                        return false;
                    }
                    if ((dir[idx].gu_code_id == CODE_SF4 || dir[idx].dn_code_id == CODE_SF4) && tmp.size() >= 16) {
                        float hdr[4];
                        std::memcpy(hdr, tmp.data(), 16);
                        expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                    }
                    if (!hip_ok(hipMemcpy(peer_expert_gpu_arena_.dev_ptr + peer_expert_gpu_offset_[idx], tmp.data(),
                                          tmp.size(), hipMemcpyHostToDevice),
                                "peer expert H2D", err)) {
                        (void)hipSetDevice(gpu0_id_);
                        return false;
                    }
                }
            } else {
                const size_t kBatch = 16;
                for (size_t b0 = 0; b0 < peer_gpu_plan.size(); b0 += kBatch) {
                    const size_t bn = std::min(kBatch, peer_gpu_plan.size() - b0);
                    std::vector<std::vector<uint8_t>> bufs(bn);
                    std::vector<uint32_t> dn_offs(bn, 0);
                    std::vector<uint8_t> gu_c(bn, 0), dn_c(bn, 0);
                    bool ok = transcode::parallel_run(bn, tc_threads, [&](size_t j, std::string& lerr) -> bool {
                        return read_expert_for_hip(peer_gpu_plan[b0 + j], bufs[j], dn_offs[j], gu_c[j], dn_c[j], lerr);
                    }, err);
                    if (!ok) { (void)hipSetDevice(gpu0_id_); return false; }
                    for (size_t j = 0; j < bn; ++j) {
                        const uint16_t idx = peer_gpu_plan[b0 + j];
                        if ((gu_c[j] == CODE_SF4 || dn_c[j] == CODE_SF4) && bufs[j].size() >= 16) {
                            float hdr[4];
                            std::memcpy(hdr, bufs[j].data(), 16);
                            expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                        }
                        if (!hip_ok(hipMemcpy(peer_expert_gpu_arena_.dev_ptr + peer_expert_gpu_offset_[idx], bufs[j].data(),
                                              bufs[j].size(), hipMemcpyHostToDevice),
                                    "peer expert H2D", err)) {
                            (void)hipSetDevice(gpu0_id_);
                            return false;
                        }
                    }
                }
            }
            if (!hip_ok(hipSetDevice(gpu0_id_), "hipSetDevice restore0 after peer arena", err)) return false;
        }
        vram_experts_loaded_ = gpu_plan.size() + peer_gpu_plan.size();
        if (!ram_plan.empty()) {
            if (model.kind == ModelSourceKind::Strata) {
                for (uint16_t idx : ram_plan) {
                    if (!model.rd.read_expert_body(idx, expert_ram_bodies_[idx], err)) return false;
                    if ((dir[idx].gu_code_id == CODE_SF4 || dir[idx].dn_code_id == CODE_SF4) && expert_ram_bodies_[idx].size() >= 16) {
                        float hdr[4];
                        std::memcpy(hdr, expert_ram_bodies_[idx].data(), 16);
                        expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                    }
                }
            } else {
                bool ok = transcode::parallel_run(ram_plan.size(), tc_threads, [&](size_t k, std::string& lerr) -> bool {
                    const uint16_t idx = ram_plan[k];
                    uint32_t dn_off = 0;
                    uint8_t gu_c = 0, dn_c = 0;
                    if (!read_expert_for_hip(idx, expert_ram_bodies_[idx], dn_off, gu_c, dn_c, lerr)) {
                        return false;
                    }
                    if ((gu_c == CODE_SF4 || dn_c == CODE_SF4) && expert_ram_bodies_[idx].size() >= 16) {
                        float hdr[4];
                        std::memcpy(hdr, expert_ram_bodies_[idx].data(), 16);
                        expert_sf4_outer_[idx] = {hdr[0] / hdr[1], hdr[2] / hdr[3]};
                    }
                    return true;
                }, err);
                if (!ok) return false;
            }
        }
        ram_experts_loaded_ = ram_plan.size();
        if (model.kind == ModelSourceKind::Strata && vram_experts_loaded_ + ram_experts_loaded_ < n_dir && max_exp_bytes > 0) {
            std::vector<InkColdExpertDesc> cdescs(n_dir);
            const uint64_t sec4_off = model.rd.header().sections[4].file_offset;
            for (size_t i = 0; i < n_dir; ++i) {
                cdescs[i].file_off   = sec4_off + dir[i].body_offset;
                cdescs[i].body_bytes = dir[i].body_bytes;
                cdescs[i].dn_off     = dir[i].dn_offset;
                cdescs[i].gu_code    = dir[i].gu_code_id;
                cdescs[i].dn_code    = dir[i].dn_code_id;
            }
            const size_t ring_b = 512ULL << 20;
            std::string cerr;
            if (cold_store_.open(model.path, std::move(cdescs), max_exp_bytes, ring_b, 4, cerr)) {
                std::fprintf(stderr, "[inkling-hip] GPU+SSD streaming active: %zu VRAM, %zu RAM, %zu SSD-only "
                                     "(%d-slot ColdStore = %.0f MiB, direct=%d)\n",
                             vram_experts_loaded_, ram_experts_loaded_,
                             n_dir - vram_experts_loaded_ - ram_experts_loaded_,
                             cold_store_.num_slots(), (double)cold_store_.pool_bytes() / 1048576.0,
                             (int)cold_store_.is_direct());
            }
        }

        if (max_exp_bytes > 0 && vram_experts_loaded_ < n_dir) {
            if (!hip_ok(alloc_gpu_buffer(d_exp_stage_, max_exp_bytes, mem_info_.is_uma), "expert stage alloc", err)) {
                return false;
            }
            if (!hybrid_.init(n_dir, max_exp_bytes, hybrid_cache_budget, mem_info_.is_uma, opts.threads, err)) {
                return false;
            }
        }

        // Build per-layer GPU expert descriptor tables [128] & per-layer codec class (0=all S_K, 1=all SF4, 2=all MX8, 3=mixed)
        d_layer_exps_.resize(c_.n_phys);
        h_layer_exps_.resize(c_.n_phys);
        peer_d_layer_exps_.resize(ep2_active_ ? c_.n_phys : 0);
        layer_all_gpu_.assign(c_.n_phys, true);
        layer_all_ep2_.assign(c_.n_phys, true);
        layer_gu_class_.assign(c_.n_phys, 3);
        layer_dn_class_.assign(c_.n_phys, 3);
        for (int il = 0; il < c_.n_phys; ++il) {
            std::vector<GpuExpertDesc> tbl(c_.n_experts);
            std::vector<GpuExpertDesc> peer_tbl(ep2_active_ ? c_.n_experts : 0);
            bool all_gu_sk = true, all_gu_sf4 = true, all_gu_mx8 = true;
            bool all_dn_sk = true, all_dn_sf4 = true, all_dn_mx8 = true;
            for (int e = 0; e < c_.n_experts; ++e) {
                const int dir_idx = model.dir_index_of[il * c_.n_experts + e];
                GpuExpertDesc d{};
                GpuExpertDesc pd{};
                if (dir_idx >= 0 && expert_gpu_offset_[dir_idx] != static_cast<size_t>(-1)) {
                    d.body = expert_gpu_arena_.dev_ptr + expert_gpu_offset_[dir_idx];
                    d.dn_offset = dir[dir_idx].dn_offset;
                    d.gu_code = dir[dir_idx].gu_code_id;
                    d.dn_code = dir[dir_idx].dn_code_id;
                    if (d.gu_code != CODE_SF4) all_gu_sf4 = false;
                    if (d.gu_code != CODE_MX8) all_gu_mx8 = false;
                    if (d.gu_code == CODE_SF4 || d.gu_code == CODE_MX8 || d.gu_code == CODE_GGUF_RAW) all_gu_sk = false;
                    if (d.dn_code != CODE_SF4) all_dn_sf4 = false;
                    if (d.dn_code != CODE_MX8) all_dn_mx8 = false;
                    if (d.dn_code == CODE_SF4 || d.dn_code == CODE_MX8 || d.dn_code == CODE_GGUF_RAW) all_dn_sk = false;
                } else if (ep2_active_ && dir_idx >= 0 && peer_expert_gpu_offset_[dir_idx] != static_cast<size_t>(-1)) {
                    pd.body = peer_expert_gpu_arena_.dev_ptr + peer_expert_gpu_offset_[dir_idx];
                    pd.dn_offset = dir[dir_idx].dn_offset;
                    pd.gu_code = dir[dir_idx].gu_code_id;
                    pd.dn_code = dir[dir_idx].dn_code_id;
                    layer_all_gpu_[il] = false;
                    all_gu_sk = all_gu_sf4 = all_gu_mx8 = all_dn_sk = all_dn_sf4 = all_dn_mx8 = false;
                } else {
                    layer_all_gpu_[il] = false;
                    layer_all_ep2_[il] = false;
                    all_gu_sk = all_gu_sf4 = all_gu_mx8 = all_dn_sk = all_dn_sf4 = all_dn_mx8 = false;
                }
                tbl[e] = d;
                if (ep2_active_) peer_tbl[e] = pd;
            }
            layer_gu_class_[il] = all_gu_sk ? 0 : (all_gu_sf4 ? 1 : (all_gu_mx8 ? 2 : 3));
            layer_dn_class_[il] = all_dn_sk ? 0 : (all_dn_sf4 ? 1 : (all_dn_mx8 ? 2 : 3));
            h_layer_exps_[il] = tbl;
            if (!upload_raw(d_layer_exps_[il], tbl.data(), tbl.size() * sizeof(GpuExpertDesc), err)) return false;
            if (ep2_active_) {
                if (!hip_ok(hipSetDevice(gpu1_id_), "hipSetDevice peer tbl", err)) return false;
                if (!hip_ok(alloc_gpu_buffer(peer_d_layer_exps_[il], peer_tbl.size() * sizeof(GpuExpertDesc), peer_mem_info_.is_uma), "peer tbl alloc", err) ||
                    !hip_ok(hipMemcpy(peer_d_layer_exps_[il].dev_ptr, peer_tbl.data(), peer_tbl.size() * sizeof(GpuExpertDesc), hipMemcpyHostToDevice), "peer tbl H2D", err)) {
                    (void)hipSetDevice(gpu0_id_);
                    return false;
                }
                if (!hip_ok(hipSetDevice(gpu0_id_), "hipSetDevice restore0 tbl", err)) return false;
            }
        }
        if (ep2_active_) {
            std::fprintf(stderr,
                         "[ink-hip] Dual-GPU EP2 active: GPU%d (%s, %zu exp, %.2f GiB) + GPU%d (%s, %zu exp, %.2f GiB), RAM=%zu | split=%.2f p2p=%s\n",
                         gpu0_id_, mem_info_.device_name, gpu_plan.size(), arena_bytes / 1073741824.0,
                         gpu1_id_, peer_mem_info_.device_name, peer_gpu_plan.size(), peer_arena_bytes / 1073741824.0,
                         ram_plan.size(), ep_split_, ep2_p2p_ ? "on" : "pinned");
        }

        // 4. Initialize GPU Paged KV pools & 48-Visit ShortConv BF16 pools
        const int kv_dim = c_.n_kv * c_.head_dim;
        const int max_swa_pages = (c_.swa_win + kKvPageTokens - 1) / kKvPageTokens;
        const int max_glob_pages = (max_ctx_ + kKvPageTokens - 1) / kKvPageTokens;

        // Post-load VRAM check (#6613 / #6621): auto-select FP8_E4M3 KV & cap max_prefill_chunk when remaining VRAM
        // is tight (e.g. 16 GB RX 9070 XT running 14.35 GB SQ-15 at 32K or Windows RX 9060 XT 16 GB SQ-10 at 256K).
        if (!mem_info_.is_uma) {
            size_t post_free = 0, post_tot = 0;
            if (hipMemGetInfo(&post_free, &post_tot) == hipSuccess) {
                const size_t fp16_page_kv_b = static_cast<size_t>(kKvPageTokens) * kv_dim * 2u * 2u;  // K + V in FP16
                const size_t fp16_kv_need =
                    (static_cast<size_t>(c_.n_swa_visits) * max_slots_ * max_swa_pages +
                     static_cast<size_t>(c_.n_global_visits) * max_slots_ * max_glob_pages) * fp16_page_kv_b;
                const size_t sconv_need =
                    static_cast<size_t>(c_.n_visits) * max_slots_ * 3u * (2u * kv_dim + 2u * c_.hidden) * sizeof(uint16_t);
#if defined(_WIN32)
                // Reserve 1.5 GiB on Windows WDDM so dedicated VRAM never crosses the ~92% PCIe paging cliff (#6621).
                const size_t kv_headroom = (1536ULL << 20);
#else
                const size_t kv_headroom = (512ULL << 20);
#endif
                if (opts.kv_dtype_auto && default_kv_dtype_ == KvDtype::FP16 &&
                    post_free < fp16_kv_need + sconv_need + kv_headroom) {
                    default_kv_dtype_ = KvDtype::FP8_E4M3;
                    std::fprintf(stderr,
                                 "[ink-hip] Auto-selected FP8_E4M3 KV cache (post-load free=%.0f MiB, FP16 KV need=%.0f MiB for ctx=%d x slots=%d)\n",
                                 post_free / 1048576.0, fp16_kv_need / 1048576.0, max_ctx_, max_slots_);
                }
                const bool tight_vram = (post_free < (2560ULL << 20))
#if defined(_WIN32)
                    || (post_tot <= (17ULL << 30) && post_free < (4096ULL << 20))
#endif
                    ;
                if (tight_vram) {
                    const int tight_chunk = (post_free < (1536ULL << 20)) ? 256 : 512;
                    if (guard_.max_prefill_chunk > tight_chunk) {
                        guard_.max_prefill_chunk = tight_chunk;
                        std::fprintf(stderr,
                                     "[ink-hip] Capped max_prefill_chunk=%d (post-load free=%.0f MiB) to avoid batch scratch VRAM spike\n",
                                     tight_chunk, post_free / 1048576.0);
                    }
                }
            }
        }
        const bool is_fp8 = (default_kv_dtype_ == KvDtype::FP8_E4M3);

        swa_pools_.resize(c_.n_swa_visits);
        for (int i = 0; i < c_.n_swa_visits; ++i) {
            if (!swa_pools_[i].init(true, is_fp8, mem_info_.is_uma, max_slots_, max_swa_pages, kv_dim, err)) return false;
        }
        glob_pools_.resize(c_.n_global_visits);
        for (int i = 0; i < c_.n_global_visits; ++i) {
            if (!glob_pools_[i].init(false, is_fp8, mem_info_.is_uma, max_slots_, max_glob_pages, kv_dim, err)) return false;
        }

        const size_t kv_all_b = static_cast<size_t>(c_.n_visits) * max_slots_ * 3 * kv_dim * sizeof(uint16_t);
        const size_t h_all_b  = static_cast<size_t>(c_.n_visits) * max_slots_ * 3 * c_.hidden * sizeof(uint16_t);
        if (!hip_ok(alloc_gpu_buffer(d_sconv_k_all_, kv_all_b, mem_info_.is_uma), "sconv k all", err) ||
            !hip_ok(alloc_gpu_buffer(d_sconv_v_all_, kv_all_b, mem_info_.is_uma), "sconv v all", err) ||
            !hip_ok(alloc_gpu_buffer(d_sconv_attn_all_, h_all_b, mem_info_.is_uma), "sconv attn all", err) ||
            !hip_ok(alloc_gpu_buffer(d_sconv_mlp_all_, h_all_b, mem_info_.is_uma), "sconv mlp all", err)) {
            return false;
        }
        HIP_ZERO(d_sconv_k_all_); HIP_ZERO(d_sconv_v_all_);
        HIP_ZERO(d_sconv_attn_all_); HIP_ZERO(d_sconv_mlp_all_);

        slots_.resize(max_slots_);
        return true;
    }

    bool alloc_slot(int slot, int max_ctx, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= max_slots_) {
            err = "hip: slot index out of range";
            return false;
        }
        auto& sl = slots_[slot];
        sl.release();
        sl.allocated = true;
        sl.max_ctx = max_ctx;
        sl.seq_len = 0;
        clear_slot_sconv(slot);
        return true;
    }

    void free_slot(int slot) override {
        if (slot < 0 || slot >= max_slots_) return;
        for (auto& p : swa_pools_) p.free_slot_pages(slot);
        for (auto& p : glob_pools_) p.free_slot_pages(slot);
        slots_[slot].release();
    }

    void reset_slot(int slot) override {
        if (slot < 0 || slot >= max_slots_ || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        sl.seq_len = 0;
        sl.dspark_len = 0;
        sl.has_snap = false;
        sl.dspark_ema_acc = 0.50f;
        sl.dspark_last_draft = 0;
        sl.dspark_last_anchor = -1;
        sl.dspark_skip_rem = 0;
        sl.dspark_had_rollback = false;
        sl.tree_ema_gain = 1.25f;
        sl.tree_cool_rem = 0;
        clear_slot_sconv(slot);
    }

    void snapshot_slot(int slot) override {
        if (slot < 0 || slot >= max_slots_ || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        const int kv_dim = c_.n_kv * c_.head_dim;
        const size_t kv_one = 3u * kv_dim * sizeof(uint16_t);
        const size_t h_one  = 3u * c_.hidden * sizeof(uint16_t);
        if (!sl.sconv_snap_k.dev_ptr) {
            hip_ign(alloc_gpu_buffer(sl.sconv_snap_k, static_cast<size_t>(c_.n_visits) * kv_one, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_snap_v, static_cast<size_t>(c_.n_visits) * kv_one, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_snap_attn, static_cast<size_t>(c_.n_visits) * h_one, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_snap_mlp, static_cast<size_t>(c_.n_visits) * h_one, mem_info_.is_uma));
            const size_t kv_rec = 8u * kv_dim * sizeof(uint16_t);
            const size_t h_rec  = 8u * c_.hidden * sizeof(uint16_t);
            hip_ign(alloc_gpu_buffer(sl.sconv_rec_k, static_cast<size_t>(c_.n_visits) * kv_rec, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_rec_v, static_cast<size_t>(c_.n_visits) * kv_rec, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_rec_attn, static_cast<size_t>(c_.n_visits) * h_rec, mem_info_.is_uma));
            hip_ign(alloc_gpu_buffer(sl.sconv_rec_mlp, static_cast<size_t>(c_.n_visits) * h_rec, mem_info_.is_uma));
        }
        // 2 batched launches across all 48 visits instead of 192 hipMemcpyAsync calls!
        k_sconv_snapshot_pair<<<dim3((kv_dim + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<const uint16_t*>(d_sconv_k_all_.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_sconv_v_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_k.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_v.dev_ptr),
            slot, max_slots_, c_.n_visits, kv_dim);
        k_sconv_snapshot_pair<<<dim3((c_.hidden + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<const uint16_t*>(d_sconv_attn_all_.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_sconv_mlp_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_attn.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_mlp.dev_ptr),
            slot, max_slots_, c_.n_visits, c_.hidden);
        sl.snap_seq_len = sl.seq_len;
        sl.sconv_rec_n = 0;
        sl.has_snap = true;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (slot < 0 || slot >= max_slots_ || !slots_[slot].allocated) return;
        auto& sl = slots_[slot];
        sl.seq_len = keep_len;
        sl.dspark_len = std::min(sl.dspark_len, keep_len);
        if (sl.has_snap) {
            const int keep_steps = keep_len - sl.snap_seq_len;
            if (sl.dspark_last_draft > 0 && keep_steps >= 1) {
                const int acc_toks = std::max(0, std::min(sl.dspark_last_draft, keep_steps - 1));
                const float rate = static_cast<float>(acc_toks) / static_cast<float>(sl.dspark_last_draft);
                sl.dspark_ema_acc = 0.70f * sl.dspark_ema_acc + 0.30f * rate;
                sl.dspark_had_rollback = true;
            }
            const int kv_dim = c_.n_kv * c_.head_dim;
            if (keep_steps >= 0 && keep_steps <= sl.sconv_rec_n) {
                // 2 batched launches across all 48 visits instead of 192 kernel/memcpy submissions!
                k_sconv_rollback_pair<<<dim3((kv_dim + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
                    reinterpret_cast<const uint16_t*>(sl.sconv_snap_k.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_rec_k.dev_ptr),
                    reinterpret_cast<uint16_t*>(d_sconv_k_all_.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_snap_v.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_rec_v.dev_ptr),
                    reinterpret_cast<uint16_t*>(d_sconv_v_all_.dev_ptr),
                    slot, max_slots_, c_.n_visits, keep_steps, kv_dim);
                k_sconv_rollback_pair<<<dim3((c_.hidden + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
                    reinterpret_cast<const uint16_t*>(sl.sconv_snap_attn.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_rec_attn.dev_ptr),
                    reinterpret_cast<uint16_t*>(d_sconv_attn_all_.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_snap_mlp.dev_ptr),
                    reinterpret_cast<const uint16_t*>(sl.sconv_rec_mlp.dev_ptr),
                    reinterpret_cast<uint16_t*>(d_sconv_mlp_all_.dev_ptr),
                    slot, max_slots_, c_.n_visits, keep_steps, c_.hidden);
            }
            sl.has_snap = false;
            sl.sconv_rec_n = 0;
        }
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        const int B_tot = batch.n_tokens;
        if (B_tot <= 0) return true;

        // Enforce <= 1.0s TDR / MES chunk cap automatically on iGPUs & Windows WDDM!
        const int chunk_cap = std::max(1, guard_.max_prefill_chunk);
        if (B_tot > chunk_cap) {
            const int V = c_.vocab_unpadded;
            const int D = c_.hidden;
            if (taps_out) {
                for (int k = 0; k < 4; ++k) taps_out->taps[k].resize(static_cast<size_t>(B_tot) * D);
            }
            size_t logit_cursor = 0;
            for (int off = 0; off < B_tot; off += chunk_cap) {
                const int sub_n = std::min(chunk_cap, B_tot - off);
                StepBatch sub;
                sub.n_tokens = sub_n;
                sub.slot_ids.assign(batch.slot_ids.begin() + off, batch.slot_ids.begin() + off + sub_n);
                sub.pos.assign(batch.pos.begin() + off, batch.pos.begin() + off + sub_n);
                sub.token_ids.assign(batch.token_ids.begin() + off, batch.token_ids.begin() + off + sub_n);
                sub.img_embed_idx.assign(batch.img_embed_idx.begin() + off, batch.img_embed_idx.begin() + off + sub_n);
                sub.want_logits.assign(batch.want_logits.begin() + off, batch.want_logits.begin() + off + sub_n);
                sub.img_embeds = batch.img_embeds;
                sub.capture_taps = batch.capture_taps;
                const int sub_want = sub.count_logits();
                float* sub_logits = (logits_out && sub_want > 0) ? (logits_out + logit_cursor * V) : nullptr;
                TapCapture sub_taps;
                if (!forward_batch_chunk(sub, sub_logits, taps_out ? &sub_taps : nullptr, err)) return false;
                if (taps_out) {
                    for (int k = 0; k < 4; ++k) {
                        std::memcpy(taps_out->taps[k].data() + static_cast<size_t>(off) * D,
                                    sub_taps.taps[k].data(), static_cast<size_t>(sub_n) * D * sizeof(float));
                    }
                }
                logit_cursor += static_cast<size_t>(sub_want);
            }
            return true;
        }
        return forward_batch_chunk(batch, logits_out, taps_out, err);
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft, int32_t* out_ids, std::string& err) override {
        if (!has_dspark_ || max_draft <= 0) return 0;
        if (slot < 0 || slot >= max_slots_ || !slots_[slot].allocated) {
            err = "hip dspark: invalid slot";
            return 0;
        }
        auto& sl = slots_[slot];
        if (anchor_pos < 1 || sl.dspark_len < anchor_pos) return 0;

        const int B = dspark_cfg_.block_size;  // 8
        // Adaptive EMA DSpark length control: full K=7 on math/code, graceful backoff K=4->2->0 on low-accept chat!
        if (sl.dspark_last_draft > 0 && !sl.dspark_had_rollback &&
            sl.dspark_last_anchor >= 0 && anchor_pos == sl.dspark_last_anchor + sl.dspark_last_draft + 1) {
            // All drafted tokens were accepted (no rollback called)!
            sl.dspark_ema_acc = 0.70f * sl.dspark_ema_acc + 0.30f * 1.0f;
        }
        sl.dspark_had_rollback = false;
        if (sl.dspark_skip_rem > 0) {
            --sl.dspark_skip_rem;
            sl.dspark_last_draft = 0;
            return 0;
        }
        int adapt_max = B - 1;
        if (sl.dspark_ema_acc < 0.20f) {
            sl.dspark_skip_rem = 6;
            adapt_max = 2;  // periodic probe so chat->code transitions recover immediately with zero chat regression!
        } else if (sl.dspark_ema_acc < 0.25f) {
            adapt_max = 4;
        }
        const int n_draft = std::min(max_draft, adapt_max);
        sl.dspark_last_draft = n_draft;
        sl.dspark_last_anchor = anchor_pos;
        const int D = dspark_cfg_.hidden;
        const int H = dspark_cfg_.n_head;
        const int KV = dspark_cfg_.n_kv;
        const int hd = dspark_cfg_.head_dim;
        const int kv_dim = KV * hd;
        const int q_dim = H * hd;
        const int inter = dspark_cfg_.inter;
        const int rank = dspark_cfg_.markov_rank;

        if (!ensure_dspark_scratch(B, D, q_dim, kv_dim, inter, err)) return 0;

        // Upload 8-byte header {anchor_pos, anchor_tok} into d_dspark_toks_[0..1]
        int32_t* d_toks = reinterpret_cast<int32_t*>(d_dspark_toks_.dev_ptr);
        const int32_t h_in[2] = {anchor_pos, anchor_tok};
        if (!hip_ok(hipMemcpyAsync(d_toks, h_in, 2 * sizeof(int32_t), hipMemcpyHostToDevice, stream_), "dspark hdr", err)) return 0;

        const bool ds_prof = (std::getenv("STRATA_HIP_DSPARK_PROF") && std::atoi(std::getenv("STRATA_HIP_DSPARK_PROF")) > 0);
        const bool use_ds_graph = (!ds_prof && n_draft >= 1 && n_draft <= 7 &&
                                   static_cast<int>(dspark_qkv_.size()) == dspark_cfg_.n_layers &&
                                   dspark_qkv_[0].dev_ptr != nullptr && dspark_gu_[0].dev_ptr != nullptr && hd <= 128);

        if (use_ds_graph) {
            hipGraphExec_t& g_exec = graph_dspark_exec_[n_draft];
            if (g_exec != nullptr && ds_graph_slot_[n_draft] == slot && ds_graph_gen_[n_draft] == dspark_ptr_gen_) {
                if (!hip_ok(hipGraphLaunch(g_exec, stream_), "dspark graph launch", err)) return 0;
                if (!hip_ok(hipMemcpyAsync(out_ids, d_toks + 2, static_cast<size_t>(n_draft) * sizeof(int32_t),
                                           hipMemcpyDeviceToHost, stream_), "dspark D2H", err)) return 0;
                if (!hip_ok(hipStreamSynchronize(stream_), "dspark sync", err)) return 0;
                return n_draft;
            }
            if (g_exec != nullptr) { (void)hipGraphExecDestroy(g_exec); g_exec = nullptr; }
            if (!hip_ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "dspark capture begin", err)) return 0;
        }

        hipEvent_t devt[4]{};
        if (ds_prof) {
            for (int i = 0; i < 4; ++i) (void)hipEventCreate(&devt[i]);
            (void)hipEventRecord(devt[0], stream_);
        }

        // 1. Build draft block [B, D]: row 0 = embed_norm(embed(anchor_tok)), rows 1..B-1 = mask_embed
        const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
        k_embed_lookup<<<1, 256, 0, stream_>>>(
            emb_pk.buf.dev_ptr, emb_pk.info.fmt, static_cast<int>(emb_pk.info.rows()),
            f32_ptr("model.llm.embed_norm.weight"),
            d_toks + 1, nullptr, nullptr,
            reinterpret_cast<float*>(ds_cur_.dev_ptr), 1, D, c_.rms_eps);
        const float* mask_e = reinterpret_cast<const float*>(dspark_f32_.at("mask_embed").dev_ptr);
        k_fill_dspark_masks<<<((B - 1) * D + 255) / 256, 256, 0, stream_>>>(
            mask_e, reinterpret_cast<float*>(ds_cur_.dev_ptr) + D, B - 1, D);

        // 2. Run 3 non-causal block-8 attention + SwiGLU layers on GPU (concatenated QKV & GU + fused Norm+RoPE + ZERO res copies!)
        float* cur = reinterpret_cast<float*>(ds_cur_.dev_ptr);
        float* res = reinterpret_cast<float*>(ds_res_.dev_ptr);
        float* h   = reinterpret_cast<float*>(ds_h_.dev_ptr);
        float* q   = reinterpret_cast<float*>(ds_q_.dev_ptr);
        float* kd  = reinterpret_cast<float*>(ds_k_.dev_ptr);
        float* vd  = reinterpret_cast<float*>(ds_v_.dev_ptr);
        float* ao  = reinterpret_cast<float*>(ds_ao_.dev_ptr);
        float* mg  = reinterpret_cast<float*>(ds_mg_.dev_ptr);
        float* mu  = reinterpret_cast<float*>(ds_mu_.dev_ptr);
        float* qkv_cat = reinterpret_cast<float*>(ds_qkv_.dev_ptr);
        float* gu_cat  = reinterpret_cast<float*>(ds_gu_.dev_ptr);

        for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
            const std::string p = "layers." + std::to_string(il);
            k_rms_norm_rows<<<B, 256, 0, stream_>>>(cur, ds_f32(p + ".input_layernorm.weight"), h, B, D, dspark_cfg_.rms_eps);

            if (dspark_qkv_[il].dev_ptr != nullptr && hd <= 128) {
                launch_bf16_opt(reinterpret_cast<const uint16_t*>(dspark_qkv_[il].dev_ptr), h, nullptr, qkv_cat, B, q_dim + 2 * kv_dim, D);
                k_dspark_split_qkv_norm_rope<<<dim3(H + 2 * KV, B), 128, 0, stream_>>>(
                    qkv_cat, ds_f32(p + ".self_attn.q_norm.weight"), ds_f32(p + ".self_attn.k_norm.weight"),
                    d_toks, q, kd, vd, B, H, KV, hd, dspark_cfg_.rms_eps, dspark_cfg_.rope_theta);
            } else {
                launch_bf16_opt(ds_bf16(p + ".self_attn.q_proj.weight"), h, nullptr, q,  B, q_dim,  D);
                launch_bf16_opt(ds_bf16(p + ".self_attn.k_proj.weight"), h, nullptr, kd, B, kv_dim, D);
                launch_bf16_opt(ds_bf16(p + ".self_attn.v_proj.weight"), h, nullptr, vd, B, kv_dim, D);
                k_rms_norm_rows<<<B * H, 256, 0, stream_>>>(q, ds_f32(p + ".self_attn.q_norm.weight"), q, B * H, hd, dspark_cfg_.rms_eps);
                k_rms_norm_rows<<<B * KV, 256, 0, stream_>>>(kd, ds_f32(p + ".self_attn.k_norm.weight"), kd, B * KV, hd, dspark_cfg_.rms_eps);
                k_half_rope<<<dim3(H, B), hd / 2, 0, stream_>>>(q, nullptr, anchor_pos, B, H, hd, dspark_cfg_.rope_theta);
                k_half_rope<<<dim3(KV, B), hd / 2, 0, stream_>>>(kd, nullptr, anchor_pos, B, KV, hd, dspark_cfg_.rope_theta);
            }

            const int swa_w = dspark_cfg_.is_swa[il] ? dspark_cfg_.swa_win : 0;
            k_dspark_block_attn_dyn<<<dim3((H + 3) / 4, B), 128, 0, stream_>>>(
                q, reinterpret_cast<const uint16_t*>(sl.dspark_k[il].dev_ptr),
                reinterpret_cast<const uint16_t*>(sl.dspark_v[il].dev_ptr),
                kd, vd, d_toks, ao, B, H, KV, swa_w);

            launch_bf16_opt(ds_bf16(p + ".self_attn.o_proj.weight"), ao, nullptr, h, B, D, q_dim);
            k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, h, cur, B * D);

            k_rms_norm_rows<<<B, 256, 0, stream_>>>(cur, ds_f32(p + ".post_attention_layernorm.weight"), h, B, D, dspark_cfg_.rms_eps);
            if (dspark_gu_[il].dev_ptr != nullptr) {
                launch_bf16_opt(reinterpret_cast<const uint16_t*>(dspark_gu_[il].dev_ptr), h, nullptr, gu_cat, B, 2 * inter, D);
                k_swiglu_split_f32<<<(B * inter + 255) / 256, 256, 0, stream_>>>(gu_cat, mg, B, inter);
            } else {
                launch_bf16_opt(ds_bf16(p + ".mlp.gate_proj.weight"), h, nullptr, mg, B, inter, D);
                launch_bf16_opt(ds_bf16(p + ".mlp.up_proj.weight"),   h, nullptr, mu, B, inter, D);
                k_swiglu_f32<<<(B * inter + 255) / 256, 256, 0, stream_>>>(mg, mu, mg, B * inter);
            }
            launch_bf16_opt(ds_bf16(p + ".mlp.down_proj.weight"), mg, nullptr, h, B, D, inter);
            k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, h, cur, B * D);
        }

        // 3. Final norm + dynamic f_rnn load + Single-Pass Base Unembed Top-512 Screen + Recurrent Markov Head!
        k_rms_norm_rows<<<B, 256, 0, stream_>>>(cur, ds_f32("norm.weight"), cur, B, D, dspark_cfg_.rms_eps);
        k_dspark_load_frnn_zero<<<(std::max(D, rank) + 255) / 256, 256, 0, stream_>>>(
            reinterpret_cast<const uint16_t*>(sl.dspark_frnn.dev_ptr), d_toks,
            res, reinterpret_cast<float*>(d_dspark_state_.dev_ptr), D, rank);
        if (ds_prof) (void)hipEventRecord(devt[1], stream_);

        const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
        const int V = c_.vocab_unpadded;
        constexpr int kScreenBlks = 256;
        constexpr int kNumCand = kScreenBlks * 2;
        constexpr int kCandArgBlks = (kNumCand + 7) / 8;

        // Read unembed[V, D] ONCE across all n_draft steps (2 WG/CU union LDS!)
        k5_dspark_base_top2_screen<<<kScreenBlks, 256, 0, stream_>>>(
            unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
            cur + D,
            reinterpret_cast<float*>(d_dspark_cand_v_.dev_ptr),
            reinterpret_cast<int32_t*>(d_dspark_cand_i_.dev_ptr),
            n_draft, D, V);
        if (ds_prof) (void)hipEventRecord(devt[2], stream_);

        float* st_a = reinterpret_cast<float*>(d_dspark_state_.dev_ptr);
        float* st_b = reinterpret_cast<float*>(d_dspark_state2_.dev_ptr);
        float* d_lat = reinterpret_cast<float*>(d_dspark_latent_.dev_ptr);
        float* d_corr = reinterpret_cast<float*>(d_dspark_corr_.dev_ptr);
        for (int step = 1; step <= n_draft; ++step) {
            float* st_in  = (step & 1) ? st_a : st_b;
            float* st_out = (step & 1) ? st_b : st_a;
            k5_dspark_markov_proj_bf16<<<(rank + 7) / 8, 256, 0, stream_>>>(
                d_toks + step,
                st_in, st_out, d_lat,
                cur + static_cast<size_t>(step) * D,
                res,
                ds_f32("markov_head.markov_w1.weight"),
                ds_bf16("markov_head.rnn_proj.weight"),
                ds_f32("markov_head.rnn_proj.bias"),
                D, rank);
            k5_dspark_markov_corr_bf16<<<(D + 7) / 8, 256, 0, stream_>>>(
                d_lat,
                ds_bf16("markov_head.hidden_out.weight"),
                ds_f32("markov_head.hidden_out.bias"),
                d_corr, D, rank);
            k5_dspark_cand_argmax_stage1<<<kCandArgBlks, 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                reinterpret_cast<const float*>(d_dspark_cand_v_.dev_ptr) + static_cast<size_t>(step - 1) * kNumCand,
                reinterpret_cast<const int32_t*>(d_dspark_cand_i_.dev_ptr) + static_cast<size_t>(step - 1) * kNumCand,
                d_corr,
                reinterpret_cast<float*>(d_dspark_blk_v_.dev_ptr),
                reinterpret_cast<int32_t*>(d_dspark_blk_i_.dev_ptr),
                kNumCand, D, V);
            k5_dspark_argmax_stage2<<<1, 256, 0, stream_>>>(
                reinterpret_cast<const float*>(d_dspark_blk_v_.dev_ptr),
                reinterpret_cast<const int32_t*>(d_dspark_blk_i_.dev_ptr),
                kCandArgBlks, d_toks + step + 1);
        }
        if (ds_prof) (void)hipEventRecord(devt[3], stream_);

        if (use_ds_graph) {
            hipGraph_t g = nullptr;
            if (!hip_ok(hipStreamEndCapture(stream_, &g), "dspark capture end", err)) return 0;
            hipGraphExec_t& g_exec = graph_dspark_exec_[n_draft];
            hipError_t ie = hipGraphInstantiate(&g_exec, g, nullptr, nullptr, 0);
            (void)hipGraphDestroy(g);
            if (!hip_ok(ie, "dspark graph instantiate", err)) return 0;
            ds_graph_slot_[n_draft] = slot;
            ds_graph_gen_[n_draft] = dspark_ptr_gen_;
            if (!hip_ok(hipGraphLaunch(g_exec, stream_), "dspark graph launch(first)", err)) return 0;
        }

        // Single tiny <=28-byte D2H copy for all drafted tokens!
        if (!hip_ok(hipMemcpyAsync(out_ids, d_toks + 2, static_cast<size_t>(n_draft) * sizeof(int32_t),
                                   hipMemcpyDeviceToHost, stream_), "dspark D2H", err)) return 0;
        if (!hip_ok(hipStreamSynchronize(stream_), "dspark sync", err)) return 0;
        if (ds_prof) {
            float m0 = 0, m1 = 0, m2 = 0, mt = 0;
            (void)hipEventElapsedTime(&m0, devt[0], devt[1]);
            (void)hipEventElapsedTime(&m1, devt[1], devt[2]);
            (void)hipEventElapsedTime(&m2, devt[2], devt[3]);
            (void)hipEventElapsedTime(&mt, devt[0], devt[3]);
            std::fprintf(stderr, "[DSPARK-PROF] K=%d pos=%d tot=%.2fms | 3L=%.2f screen=%.2f markov=%.2f\n",
                         n_draft, anchor_pos, mt, m0, m1, m2);
            for (int i = 0; i < 4; ++i) (void)hipEventDestroy(devt[i]);
        }
        return n_draft;
    }

    // Wave-7 Full On-GPU W=4 Multi-Branch Adaptive Beam Tree DSpark Speculative Step (greedy, no penalties)!
    // Matches InkBackend::dspark_spec_step_greedy exact ABI (#398) with zero file overlaps outside hip/.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
    int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) {
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        if (!has_dspark_ || max_emit <= 0 || out_emitted == nullptr) return 0;
        if (slot < 0 || slot >= max_slots_ || !slots_[slot].allocated) return 0;
        auto& sl = slots_[slot];
        if (anchor_pos < 1 || sl.dspark_len < anchor_pos || anchor_pos + kTreeMaxM >= sl.max_ctx) return 0;
        if (c_.head_dim > 128 || c_.d_rel > 16 || (c_.rel_extent & 3) != 0) return 0;
        if (static_cast<int>(dspark_qkv_.size()) != dspark_cfg_.n_layers ||
            dspark_qkv_[0].dev_ptr == nullptr || dspark_gu_[0].dev_ptr == nullptr) return 0;
        for (bool ok_l : layer_all_gpu_) if (!ok_l) return 0;

        const int D = dspark_cfg_.hidden;
        const int H = dspark_cfg_.n_head;
        const int KV = dspark_cfg_.n_kv;
        const int hd = dspark_cfg_.head_dim;
        const int kv_dim = KV * hd;
        const int q_dim = H * hd;
        const int inter = dspark_cfg_.inter;
        const int rank = dspark_cfg_.markov_rank;
        const int B_ds = dspark_cfg_.block_size;  // 8

        if (!ensure_dspark_scratch(B_ds, D, q_dim, kv_dim, inter, err)) return 0;
        if (!ensure_batch_scratch(kTreeMaxM, err)) return 0;
        if (!ensure_tree_scratch(err)) return 0;

        // Ensure slot snapshot & record buffers exist so rollback_slot() stays 100% exact on early EOS!
        if (!sl.sconv_snap_k.dev_ptr) {
            const int trunk_kvd = c_.n_kv * c_.head_dim;
            const size_t kv_one = 3u * trunk_kvd * sizeof(uint16_t);
            const size_t h_one  = 3u * c_.hidden * sizeof(uint16_t);
            const size_t kv_rec = 8u * trunk_kvd * sizeof(uint16_t);
            const size_t h_rec  = 8u * c_.hidden * sizeof(uint16_t);
            if (!hip_ok(alloc_gpu_buffer(sl.sconv_snap_k, static_cast<size_t>(c_.n_visits) * kv_one, mem_info_.is_uma), "tree snap k", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_snap_v, static_cast<size_t>(c_.n_visits) * kv_one, mem_info_.is_uma), "tree snap v", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_snap_attn, static_cast<size_t>(c_.n_visits) * h_one, mem_info_.is_uma), "tree snap a", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_snap_mlp, static_cast<size_t>(c_.n_visits) * h_one, mem_info_.is_uma), "tree snap m", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_rec_k, static_cast<size_t>(c_.n_visits) * kv_rec, mem_info_.is_uma), "tree rec k", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_rec_v, static_cast<size_t>(c_.n_visits) * kv_rec, mem_info_.is_uma), "tree rec v", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_rec_attn, static_cast<size_t>(c_.n_visits) * h_rec, mem_info_.is_uma), "tree rec a", err) ||
                !hip_ok(alloc_gpu_buffer(sl.sconv_rec_mlp, static_cast<size_t>(c_.n_visits) * h_rec, mem_info_.is_uma), "tree rec m", err)) {
                return 0;
            }
        }

        // Pre-ensure DSpark context capacity for anchor_pos + 8 before capturing/launching graphs!
        const int old_ds_cap = sl.dspark_cap;
        if (!ensure_dspark_slot_cap(sl, anchor_pos + 8, err)) return 0;
        if (!d_ds_k_ptrs_[0].dev_ptr || sl.dspark_cap != old_ds_cap) {
            std::vector<uint16_t*> pt(max_slots_, nullptr);
            for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
                for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_k[il].dev_ptr);
                if (!d_ds_k_ptrs_[il].dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_k_ptrs_[il], max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds k ptrs", err)) return 0;
                (void)hipMemcpyAsync(d_ds_k_ptrs_[il].dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
                for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_v[il].dev_ptr);
                if (!d_ds_v_ptrs_[il].dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_v_ptrs_[il], max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds v ptrs", err)) return 0;
                (void)hipMemcpyAsync(d_ds_v_ptrs_[il].dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
            }
            for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_frnn.dev_ptr);
            if (!d_ds_frnn_ptrs_.dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_frnn_ptrs_, max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds frnn ptrs", err)) return 0;
            (void)hipMemcpyAsync(d_ds_frnn_ptrs_.dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
            ++dspark_ptr_gen_;
        }

        // ---- STAGE 1 & 2: Realized-Gain Cooldown + On-GPU W=4 Beam Draft + Sparse-MoE Bayes Tree Sizing ----
        const bool tprof = (std::getenv("STRATA_HIP_TREE_PROF") && std::atoi(std::getenv("STRATA_HIP_TREE_PROF")) > 0);
        hipEvent_t tev[4]{};
        if (tprof) {
            for (int i = 0; i < 4; ++i) (void)hipEventCreate(&tev[i]);
            (void)hipEventRecord(tev[0], stream_);
        }

        int32_t t_tok[kTreeMaxM];
        int32_t t_par[kTreeMaxM];
        int32_t t_dep[kTreeMaxM];
        float   t_prob[kTreeMaxM];
        t_tok[0] = anchor_tok;
        t_par[0] = -1;
        t_dep[0] = 0;
        t_prob[0] = 1.0f;

        int M = 1;
        double chosen_cost_ratio = 1.0;

        if (sl.tree_cool_rem > 0) {
            // Fast cooldown -> fall straight into 100% fused B=1 HIP Graph (graph_exec_) with zero draft overhead!
            --sl.tree_cool_rem;
            sl.dspark_skip_rem = std::max(sl.dspark_skip_rem, 1);
            if (tprof) for (int i = 0; i < 4; ++i) (void)hipEventDestroy(tev[i]);
            return 0;
        }
        {
            int32_t* d_hdr = reinterpret_cast<int32_t*>(d_dspark_toks_.dev_ptr);
            const int32_t h_in[2] = {anchor_pos, anchor_tok};
            if (!hip_ok(hipMemcpyAsync(d_hdr, h_in, 2 * sizeof(int32_t), hipMemcpyHostToDevice, stream_), "beam hdr", err)) return 0;

            if (graph_beam_draft_exec_ != nullptr && beam_graph_slot_ == slot && beam_graph_gen_ == dspark_ptr_gen_) {
                if (!hip_ok(hipGraphLaunch(graph_beam_draft_exec_, stream_), "beam graph launch", err)) return 0;
            } else {
                if (graph_beam_draft_exec_ != nullptr) {
                    (void)hipGraphExecDestroy(graph_beam_draft_exec_);
                    graph_beam_draft_exec_ = nullptr;
                }
                if (!hip_ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "beam capture begin", err)) return 0;

                const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
                k_embed_lookup<<<1, 256, 0, stream_>>>(
                    emb_pk.buf.dev_ptr, emb_pk.info.fmt, static_cast<int>(emb_pk.info.rows()),
                    f32_ptr("model.llm.embed_norm.weight"),
                    d_hdr + 1, nullptr, nullptr,
                    reinterpret_cast<float*>(ds_cur_.dev_ptr), 1, D, c_.rms_eps);
                k_fill_dspark_masks<<<((B_ds - 1) * D + 255) / 256, 256, 0, stream_>>>(
                    reinterpret_cast<const float*>(dspark_f32_.at("mask_embed").dev_ptr),
                    reinterpret_cast<float*>(ds_cur_.dev_ptr) + D, B_ds - 1, D);

                float* cur = reinterpret_cast<float*>(ds_cur_.dev_ptr);
                float* res = reinterpret_cast<float*>(ds_res_.dev_ptr);
                float* h   = reinterpret_cast<float*>(ds_h_.dev_ptr);
                float* q   = reinterpret_cast<float*>(ds_q_.dev_ptr);
                float* kd  = reinterpret_cast<float*>(ds_k_.dev_ptr);
                float* vd  = reinterpret_cast<float*>(ds_v_.dev_ptr);
                float* ao  = reinterpret_cast<float*>(ds_ao_.dev_ptr);
                float* mg  = reinterpret_cast<float*>(ds_mg_.dev_ptr);
                float* qkv_cat = reinterpret_cast<float*>(ds_qkv_.dev_ptr);
                float* gu_cat  = reinterpret_cast<float*>(ds_gu_.dev_ptr);

                for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
                    const std::string p = "layers." + std::to_string(il);
                    k_rms_norm_rows<<<B_ds, 256, 0, stream_>>>(cur, ds_f32(p + ".input_layernorm.weight"), h, B_ds, D, dspark_cfg_.rms_eps);
                    launch_bf16_opt(reinterpret_cast<const uint16_t*>(dspark_qkv_[il].dev_ptr), h, nullptr, qkv_cat, B_ds, q_dim + 2 * kv_dim, D);
                    k_dspark_split_qkv_norm_rope<<<dim3(H + 2 * KV, B_ds), 128, 0, stream_>>>(
                        qkv_cat, ds_f32(p + ".self_attn.q_norm.weight"), ds_f32(p + ".self_attn.k_norm.weight"),
                        d_hdr, q, kd, vd, B_ds, H, KV, hd, dspark_cfg_.rms_eps, dspark_cfg_.rope_theta);
                    const int swa_w = dspark_cfg_.is_swa[il] ? dspark_cfg_.swa_win : 0;
                    k_dspark_block_attn_dyn<<<dim3((H + 3) / 4, B_ds), 128, 0, stream_>>>(
                        q, reinterpret_cast<const uint16_t*>(sl.dspark_k[il].dev_ptr),
                        reinterpret_cast<const uint16_t*>(sl.dspark_v[il].dev_ptr),
                        kd, vd, d_hdr, ao, B_ds, H, KV, swa_w);
                    launch_bf16_opt(ds_bf16(p + ".self_attn.o_proj.weight"), ao, nullptr, h, B_ds, D, q_dim);
                    k_vec_add<<<(B_ds * D + 255) / 256, 256, 0, stream_>>>(cur, h, cur, B_ds * D);

                    k_rms_norm_rows<<<B_ds, 256, 0, stream_>>>(cur, ds_f32(p + ".post_attention_layernorm.weight"), h, B_ds, D, dspark_cfg_.rms_eps);
                    launch_bf16_opt(reinterpret_cast<const uint16_t*>(dspark_gu_[il].dev_ptr), h, nullptr, gu_cat, B_ds, 2 * inter, D);
                    k_swiglu_split_f32<<<(B_ds * inter + 255) / 256, 256, 0, stream_>>>(gu_cat, mg, B_ds, inter);
                    launch_bf16_opt(ds_bf16(p + ".mlp.down_proj.weight"), mg, nullptr, h, B_ds, D, inter);
                    k_vec_add<<<(B_ds * D + 255) / 256, 256, 0, stream_>>>(cur, h, cur, B_ds * D);
                }

                k_rms_norm_rows<<<B_ds, 256, 0, stream_>>>(cur, ds_f32("norm.weight"), cur, B_ds, D, dspark_cfg_.rms_eps);
                k_dspark_load_frnn_zero<<<(std::max(D, rank) + 255) / 256, 256, 0, stream_>>>(
                    reinterpret_cast<const uint16_t*>(sl.dspark_frnn.dev_ptr), d_hdr,
                    res, reinterpret_cast<float*>(d_beam_st_a_.dev_ptr), D, rank);

                const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
                const int V = c_.vocab_unpadded;
                constexpr int kScreenBlks = 256;
                constexpr int kNumCand = kScreenBlks * 2;  // 512

                k5_dspark_base_top2_screen<<<kScreenBlks, 256, 0, stream_>>>(
                    unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                    cur + D,
                    reinterpret_cast<float*>(d_dspark_cand_v_.dev_ptr),
                    reinterpret_cast<int32_t*>(d_dspark_cand_i_.dev_ptr),
                    kBeamMaxSteps, D, V);

                BeamTreeNode* d_pool = reinterpret_cast<BeamTreeNode*>(d_beam_pool_.dev_ptr);
                int32_t* d_cnt  = reinterpret_cast<int32_t*>(d_beam_cnt_.dev_ptr);
                int32_t* d_btok = reinterpret_cast<int32_t*>(d_beam_tok_.dev_ptr);
                int32_t* d_bnod = reinterpret_cast<int32_t*>(d_beam_node_.dev_ptr);
                float*   d_blp  = reinterpret_cast<float*>(d_beam_lp_.dev_ptr);
                float*   d_sta  = reinterpret_cast<float*>(d_beam_st_a_.dev_ptr);
                float*   d_stb  = reinterpret_cast<float*>(d_beam_st_b_.dev_ptr);
                float*   d_lat  = reinterpret_cast<float*>(d_beam_lat_.dev_ptr);
                float*   d_cor  = reinterpret_cast<float*>(d_beam_corr_.dev_ptr);
                float*   d_sco  = reinterpret_cast<float*>(d_beam_scores_.dev_ptr);

                k5_beam_init<<<1, 32, 0, stream_>>>(d_hdr, d_pool, d_cnt, d_btok, d_bnod, d_blp);

                for (int step = 1; step <= kBeamMaxSteps; ++step) {
                    const int w_in = (step == 1) ? 1 : kBeamW;
                    k5_beam_markov_proj_bf16<<<dim3((rank + 7) / 8, w_in), 256, 0, stream_>>>(
                        d_btok, d_sta, d_stb, d_lat,
                        cur + static_cast<size_t>(step) * D, res,
                        ds_f32("markov_head.markov_w1.weight"),
                        ds_bf16("markov_head.rnn_proj.weight"),
                        ds_f32("markov_head.rnn_proj.bias"),
                        D, rank);
                    k5_beam_markov_corr_bf16<<<dim3((D + 7) / 8, w_in), 256, 0, stream_>>>(
                        d_lat,
                        ds_bf16("markov_head.hidden_out.weight"),
                        ds_f32("markov_head.hidden_out.bias"),
                        d_cor, D, rank);
                    k5_beam_cand_scores<<<(kNumCand + 7) / 8, 256, 0, stream_>>>(
                        unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                        reinterpret_cast<const float*>(d_dspark_cand_v_.dev_ptr) + static_cast<size_t>(step - 1) * kNumCand,
                        reinterpret_cast<const int32_t*>(d_dspark_cand_i_.dev_ptr) + static_cast<size_t>(step - 1) * kNumCand,
                        d_cor, d_sco, w_in, kNumCand, D, V);
                    k5_beam_select_step<<<1, 256, 0, stream_>>>(
                        d_sco,
                        reinterpret_cast<const int32_t*>(d_dspark_cand_i_.dev_ptr) + static_cast<size_t>(step - 1) * kNumCand,
                        d_stb, d_sta, d_btok, d_bnod, d_blp, d_pool, d_cnt,
                        w_in, kNumCand, step, rank, c_.logit_softcap);
                }

                hipGraph_t g = nullptr;
                if (!hip_ok(hipStreamEndCapture(stream_, &g), "beam capture end", err)) return 0;
                hipError_t ie = hipGraphInstantiate(&graph_beam_draft_exec_, g, nullptr, nullptr, 0);
                (void)hipGraphDestroy(g);
                if (!hip_ok(ie, "beam graph instantiate", err)) return 0;
                beam_graph_slot_ = slot;
                beam_graph_gen_ = dspark_ptr_gen_;
                if (!hip_ok(hipGraphLaunch(graph_beam_draft_exec_, stream_), "beam graph launch(first)", err)) return 0;
            }

            // Download tiny 512-byte candidate pool + 4-byte count
            if (tprof) (void)hipEventRecord(tev[1], stream_);
            BeamTreeNode h_pool[kTreePoolMax];
            int32_t h_cnt = 0;
            if (!hip_ok(hipMemcpyAsync(h_pool, d_beam_pool_.dev_ptr, sizeof(h_pool), hipMemcpyDeviceToHost, stream_), "beam pool D2H", err) ||
                !hip_ok(hipMemcpyAsync(&h_cnt, d_beam_cnt_.dev_ptr, sizeof(int32_t), hipMemcpyDeviceToHost, stream_), "beam cnt D2H", err) ||
                !hip_ok(hipStreamSynchronize(stream_), "beam draft sync", err)) {
                return 0;
            }
            h_cnt = std::max(1, std::min(h_cnt, kTreePoolMax));

            // Build connected prefix-closed tree ordered by descending acceptance probability exp(cum_lp)
            const int max_depth_allowed = std::min(kBeamMaxSteps, max_emit - 1);
            int pool_to_tree[kTreePoolMax];
            for (int i = 0; i < kTreePoolMax; ++i) pool_to_tree[i] = -1;
            pool_to_tree[0] = 0;
            int n_sel = 1;
            constexpr int kMaxTreeSel = 12;

            while (n_sel < kMaxTreeSel) {
                int best_p = -1;
                float best_lp = -1e29f;
                for (int i = 1; i < h_cnt; ++i) {
                    if (pool_to_tree[i] >= 0) continue;
                    if (h_pool[i].depth > max_depth_allowed) continue;
                    const int par_i = h_pool[i].parent;
                    if (par_i < 0 || par_i >= h_cnt || pool_to_tree[par_i] < 0) continue;
                    if (h_pool[i].cum_lp > best_lp) {
                        best_lp = h_pool[i].cum_lp;
                        best_p = i;
                    }
                }
                if (best_p < 0) break;
                pool_to_tree[best_p] = n_sel;
                t_tok[n_sel] = h_pool[best_p].tok;
                t_par[n_sel] = pool_to_tree[h_pool[best_p].parent];
                t_dep[n_sel] = h_pool[best_p].depth;
                t_prob[n_sel] = std::exp(std::max(-40.0f, std::min(0.0f, h_pool[best_p].cum_lp)));
                ++n_sel;
            }

            // Sparse-MoE Bayes-optimal M* normalized to M=1 (1.0x) plain decode!
            static const int kCandMs[] = {3, 4, 5, 6, 8, 10, 12};
            int best_M = 1;
            double best_score = 0.88;  // reject flat high-entropy chat probes while never starving code/math!
            double best_cost = 1.0;
            for (int cm : kCandMs) {
                if (cm > n_sel) continue;
                double exp_acc = 1.0;
                for (int i = 1; i < cm; ++i) exp_acc += static_cast<double>(t_prob[i]);
                const double cost = 1.0 + 4.2 * (1.0 - std::pow(15.0 / 16.0, static_cast<double>(cm)));
                const double score = exp_acc / cost;
                if (score > best_score) {
                    best_score = score;
                    best_M = cm;
                    best_cost = cost;
                }
            }
            M = best_M;
            chosen_cost_ratio = best_cost;
        }
        if (M < 2) {
            // Low-confidence probe -> fall straight into fused B=1 HIP Graph & enter cooldown!
            sl.tree_ema_gain = 0.80f * sl.tree_ema_gain + 0.20f * 0.65f;
            if (sl.tree_ema_gain < 0.82f) sl.tree_cool_rem = 12;
            sl.dspark_skip_rem = std::max(sl.dspark_skip_rem, 1);
            if (tprof) for (int i = 0; i < 4; ++i) (void)hipEventDestroy(tev[i]);
            return 0;
        }

        // Build TreeMetaDev (anc1, anc2, anc3, uint32_t ancestor bitmask)
        TreeMetaDev h_tmeta{};
        h_tmeta.base_pos = anchor_pos;
        h_tmeta.slot = slot;
        int max_dep = 0;
        int32_t h_sids[kTreeMaxM], h_pos[kTreeMaxM], h_img[kTreeMaxM];
        for (int m = 0; m < M; ++m) {
            const int a1 = t_par[m];
            const int a2 = (a1 >= 0) ? t_par[a1] : (a1 - 1);
            const int a3 = (a2 >= 0) ? t_par[a2] : (a2 - 1);
            h_tmeta.anc1[m] = a1;
            h_tmeta.anc2[m] = a2;
            h_tmeta.anc3[m] = a3;
            uint32_t msk = (1u << m);
            int p = a1;
            while (p >= 0) { msk |= (1u << p); p = t_par[p]; }
            h_tmeta.mask[m] = msk;
            h_sids[m] = slot;
            h_pos[m] = anchor_pos + t_dep[m];
            h_img[m] = -1;
            max_dep = std::max(max_dep, t_dep[m]);
        }

        // Pre-ensure KV pages for all possible committed positions anchor_pos .. anchor_pos + max_dep
        for (int d = 0; d <= max_dep; ++d) {
            const int p = anchor_pos + d;
            const int swa_lp = (p % c_.swa_win) >> 6;
            const int glb_lp = p >> 6;
            for (int i = 0; i < c_.n_swa_visits; ++i) {
                if (!swa_pools_[i].ensure_page(slot, swa_lp, stream_, err)) return 0;
            }
            for (int i = 0; i < c_.n_global_visits; ++i) {
                if (!glob_pools_[i].ensure_page(slot, glb_lp, stream_, err)) return 0;
            }
        }
        for (auto& p : swa_pools_) if (!p.sync_pt(stream_, err)) return 0;
        for (auto& p : glob_pools_) if (!p.sync_pt(stream_, err)) return 0;

        uint64_t kv_gen_sum = dspark_ptr_gen_;
        for (const auto& p : swa_pools_) kv_gen_sum += p.pool_gen;
        for (const auto& p : glob_pools_) kv_gen_sum += p.pool_gen;
        if (kv_gen_sum != tree_kv_desc_gen_) {
            std::vector<KvVisitDesc> h_descs(c_.n_visits);
            for (int v = 0; v < c_.n_visits; ++v) {
                const int ip = (v < c_.core_start) ? v
                    : ((v < c_.core_start + c_.num_passes * (c_.core_end - c_.core_start))
                        ? (c_.core_start + (v - c_.core_start) % (c_.core_end - c_.core_start))
                        : (c_.core_end + (v - c_.core_start - c_.num_passes * (c_.core_end - c_.core_start))));
                const bool is_swa = c_.is_swa[ip] != 0;
                const VisitKvPool& kvp = is_swa ? swa_pools_[c_.visit_kv_slot[v]] : glob_pools_[c_.visit_kv_slot[v]];
                h_descs[v] = {
                    kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                    reinterpret_cast<float*>(kvp.k_scales.dev_ptr),
                    reinterpret_cast<float*>(kvp.v_scales.dev_ptr),
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr),
                    kvp.max_pages_per_slot,
                    is_swa ? c_.swa_win : 0
                };
            }
            if (!hip_ok(hipMemcpyAsync(d_tree_kv_descs_.dev_ptr, h_descs.data(),
                                       h_descs.size() * sizeof(KvVisitDesc), hipMemcpyHostToDevice, stream_),
                        "tree kv descs H2D", err)) return 0;
            tree_kv_desc_gen_ = kv_gen_sum;
        }

        // ---- STAGE 3: Exact Batched Tree Verify (M tokens) + Zero-Logits-D2H On-GPU Argmax (1 HIP Graph!) ----
        (void)hipMemcpyAsync(d_tree_meta_.dev_ptr, &h_tmeta, sizeof(TreeMetaDev), hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_slot_ids_.dev_ptr, h_sids, M * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_pos_.dev_ptr, h_pos, M * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_tok_ids_.dev_ptr, t_tok, M * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_img_idx_.dev_ptr, h_img, M * 4, hipMemcpyHostToDevice, stream_);

        if (!forward_tree_verify(M, slot, kv_gen_sum, err)) return 0;
        if (tprof) (void)hipEventRecord(tev[2], stream_);

        int32_t h_argmax[kTreeMaxM];
        if (!hip_ok(hipMemcpyAsync(h_argmax, d_tree_arg_out_.dev_ptr, M * sizeof(int32_t), hipMemcpyDeviceToHost, stream_), "tree argmax D2H", err) ||
            !hip_ok(hipStreamSynchronize(stream_), "tree verify sync", err)) {
            return 0;
        }

        // ---- STAGE 4: Walk Winning Path + 1-Launch Commit (ShortConv + Paged KV + DSpark Context!) ----
        WinPath8 wp{};
        wp.n_acc = 0;
        int cur_node = 0;
        while (wp.n_acc < max_emit && wp.n_acc < 8) {
            const int32_t pred = h_argmax[cur_node];
            out_emitted[wp.n_acc] = pred;
            wp.idx[wp.n_acc] = cur_node;
            ++wp.n_acc;
            if (wp.n_acc >= max_emit || wp.n_acc >= 8) break;
            int next_child = -1;
            for (int j = cur_node + 1; j < M; ++j) {
                if (t_par[j] == cur_node && t_tok[j] == pred) {
                    next_child = j;
                    break;
                }
            }
            if (next_child < 0) break;
            cur_node = next_child;
        }

        const int n_acc = wp.n_acc;
        const int trunk_kvd = c_.n_kv * c_.head_dim;
        const bool is_fp8 = (default_kv_dtype_ == KvDtype::FP8_E4M3);

        // 4a. Commit ShortConv (all 48 visits x 4 pools in 2 launches + populate snap/rec for rollback_slot!)
        k_sconv_commit_tree_pair<<<dim3((trunk_kvd + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<uint16_t*>(d_sconv_k_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_k.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_rec_k.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_tree_rec_k_.dev_ptr),
            reinterpret_cast<uint16_t*>(d_sconv_v_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_v.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_rec_v.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_tree_rec_v_.dev_ptr),
            wp, slot, max_slots_, c_.n_visits, trunk_kvd);
        k_sconv_commit_tree_pair<<<dim3((c_.hidden + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<uint16_t*>(d_sconv_attn_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_attn.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_rec_attn.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_tree_rec_attn_.dev_ptr),
            reinterpret_cast<uint16_t*>(d_sconv_mlp_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_snap_mlp.dev_ptr),
            reinterpret_cast<uint16_t*>(sl.sconv_rec_mlp.dev_ptr),
            reinterpret_cast<const uint16_t*>(d_tree_rec_mlp_.dev_ptr),
            wp, slot, max_slots_, c_.n_visits, c_.hidden);

        // 4b. Commit winning path K/V across ALL 48 visits in 1 launch!
        if (is_fp8) {
            k3_kv_commit_tree_all<true><<<dim3(n_acc, c_.n_visits), 256, 0, stream_>>>(
                reinterpret_cast<const KvVisitDesc*>(d_tree_kv_descs_.dev_ptr),
                reinterpret_cast<const half_t*>(d_tree_k_.dev_ptr),
                reinterpret_cast<const half_t*>(d_tree_v_.dev_ptr),
                wp, slot, anchor_pos, c_.n_visits, trunk_kvd);
        } else {
            k3_kv_commit_tree_all<false><<<dim3(n_acc, c_.n_visits), 256, 0, stream_>>>(
                reinterpret_cast<const KvVisitDesc*>(d_tree_kv_descs_.dev_ptr),
                reinterpret_cast<const half_t*>(d_tree_k_.dev_ptr),
                reinterpret_cast<const half_t*>(d_tree_v_.dev_ptr),
                wp, slot, anchor_pos, c_.n_visits, trunk_kvd);
        }

        // 4c. Commit DSpark context K/V/f_rnn ONLY for the n_acc winning tokens (100% async-safe GPU slot/pos + HIP-graphed!)
        k_gather_tree_taps4<<<dim3((c_.hidden + 255) / 256, n_acc), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(sc_taps_[0].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[1].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[2].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[3].dev_ptr),
            reinterpret_cast<float*>(d_tree_taps_win_[0].dev_ptr),
            reinterpret_cast<float*>(d_tree_taps_win_[1].dev_ptr),
            reinterpret_cast<float*>(d_tree_taps_win_[2].dev_ptr),
            reinterpret_cast<float*>(d_tree_taps_win_[3].dev_ptr),
            wp, c_.hidden);
        k_fill_win_slot_pos<<<1, 32, 0, stream_>>>(
            reinterpret_cast<int32_t*>(sc_slot_ids_.dev_ptr),
            reinterpret_cast<int32_t*>(sc_pos_.dev_ptr),
            slot, anchor_pos, n_acc);
        if (graph_tree_ctx_exec_[n_acc] != nullptr && tctx_graph_gen_[n_acc] == dspark_ptr_gen_ && tctx_graph_B_[n_acc] == scratch_B_) {
            (void)hipGraphLaunch(graph_tree_ctx_exec_[n_acc], stream_);
        } else {
            if (graph_tree_ctx_exec_[n_acc]) {
                (void)hipGraphExecDestroy(graph_tree_ctx_exec_[n_acc]);
                graph_tree_ctx_exec_[n_acc] = nullptr;
            }
            if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) == hipSuccess) {
                launch_dspark_ctx_kernels_from(
                    n_acc,
                    reinterpret_cast<const float*>(d_tree_taps_win_[0].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[1].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[2].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[3].dev_ptr));
                hipGraph_t cg = nullptr;
                if (hipStreamEndCapture(stream_, &cg) == hipSuccess) {
                    if (hipGraphInstantiate(&graph_tree_ctx_exec_[n_acc], cg, nullptr, nullptr, 0) == hipSuccess) {
                        tctx_graph_gen_[n_acc] = dspark_ptr_gen_;
                        tctx_graph_B_[n_acc] = scratch_B_;
                        (void)hipGraphLaunch(graph_tree_ctx_exec_[n_acc], stream_);
                    }
                    (void)hipGraphDestroy(cg);
                }
            } else {
                launch_dspark_ctx_kernels_from(
                    n_acc,
                    reinterpret_cast<const float*>(d_tree_taps_win_[0].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[1].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[2].dev_ptr),
                    reinterpret_cast<const float*>(d_tree_taps_win_[3].dev_ptr));
            }
        }
        if (tprof) {
            (void)hipEventRecord(tev[3], stream_);
            (void)hipEventSynchronize(tev[3]);
            float md = 0, mv = 0, mc = 0;
            (void)hipEventElapsedTime(&md, tev[0], tev[1]);
            (void)hipEventElapsedTime(&mv, tev[1], tev[2]);
            (void)hipEventElapsedTime(&mc, tev[2], tev[3]);
            std::fprintf(stderr, "[TREE-PROF] pos=%d M=%d acc=%d tot=%.2fms | draft=%.2f verify=%.2f commit=%.2f\n",
                         anchor_pos, M, n_acc, md + mv + mc, md, mv, mc);
            for (int i = 0; i < 4; ++i) (void)hipEventDestroy(tev[i]);
        }

        sl.seq_len = anchor_pos + n_acc;
        sl.dspark_len = anchor_pos + n_acc;
        sl.has_snap = true;
        sl.snap_seq_len = anchor_pos;
        sl.sconv_rec_n = n_acc;
        sl.dspark_last_draft = 0;

        // Realized-gain adaptive cooldown: protect high-entropy chat prompts while never starving code/math!
        if (M >= 2) {
            const float realized = static_cast<float>(n_acc) / static_cast<float>(chosen_cost_ratio);
            sl.tree_ema_gain = 0.80f * sl.tree_ema_gain + 0.20f * realized;
            if (sl.tree_ema_gain < 0.82f) {
                sl.tree_cool_rem = 12;
            }
        }

        if (out_drafted) *out_drafted = M - 1;
        return n_acc;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (opts_.vision_path.empty()) return false;
        if (!vision_.loaded) {
            if (!load_vision_gpu(opts_.vision_path, err)) return false;
        }
        const VitCfg& vc = vision_.info.c;
        const int merge = vc.merge;
        const int patch = vc.patch;
        const int cell = patch * merge;
        const int n_tok = img.num_lm_tokens();
        const int ph = img.grid_h * merge;
        const int pw = img.grid_w * merge;
        const int n_patches = ph * pw;
        const int S = n_patches + 1;
        const int D = vc.hidden;
        const int H = vc.heads;
        const int hd = vc.head_dim;
        const int patch_in = 3 * patch * patch;

        // 1. Patchify + 2D pos coords on host (< 1 ms)
        std::vector<float> patch_buf(static_cast<size_t>(n_patches) * patch_in);
        std::vector<int> pos_h(S, 0), pos_w(S, 0);
        int pi = 0;
        for (int gr = 0; gr < img.grid_h; ++gr) {
            for (int gc = 0; gc < img.grid_w; ++gc) {
                const size_t tok = static_cast<size_t>(gr) * img.grid_w + gc;
                const float* cell_px = img.pixel_values.data() + tok * cell * cell * 3;
                for (int mr = 0; mr < merge; ++mr) {
                    for (int mc = 0; mc < merge; ++mc) {
                        float* dst = patch_buf.data() + static_cast<size_t>(pi) * patch_in;
                        int idx = 0;
                        for (int py = 0; py < patch; ++py) {
                            const int ly = mr * patch + py;
                            for (int px = 0; px < patch; ++px) {
                                const int lx = mc * patch + px;
                                const float* rgb = cell_px + (static_cast<size_t>(ly) * cell + lx) * 3;
                                dst[idx++] = rgb[0]; dst[idx++] = rgb[1]; dst[idx++] = rgb[2];
                            }
                        }
                        pos_h[pi + 1] = gr * merge + mr + 1;
                        pos_w[pi + 1] = gc * merge + mc + 1;
                        ++pi;
                    }
                }
            }
        }

        // Precompute bilinearly interpolated pos_embed [S, D] + axial 2D RoPE cos/sin [S, hd] on host
        std::vector<float> h_pos_add(static_cast<size_t>(S) * D, 0.0f);
        for (int d = 0; d < D; ++d) h_pos_add[d] = vision_.h_cls[d] + vision_.h_cls_pos[d];
        const int G = vc.posemb_grid;
        for (int p = 0; p < n_patches; ++p) {
            const float y = (static_cast<float>(pos_h[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / ph) - 0.5f;
            const float x = (static_cast<float>(pos_w[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / pw) - 0.5f;
            const int y0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(y))));
            const int y1 = std::min(G - 1, y0 + 1);
            const int x0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(x))));
            const int x1 = std::min(G - 1, x0 + 1);
            const float wy = std::max(0.0f, std::min(1.0f, y - y0));
            const float wx = std::max(0.0f, std::min(1.0f, x - x0));
            const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx), w11 = wy * wx;
            const float* p00 = vision_.h_pos_emb.data() + (static_cast<size_t>(y0) * G + x0) * D;
            const float* p01 = vision_.h_pos_emb.data() + (static_cast<size_t>(y0) * G + x1) * D;
            const float* p10 = vision_.h_pos_emb.data() + (static_cast<size_t>(y1) * G + x0) * D;
            const float* p11 = vision_.h_pos_emb.data() + (static_cast<size_t>(y1) * G + x1) * D;
            float* dst = h_pos_add.data() + static_cast<size_t>(p + 1) * D;
            for (int d = 0; d < D; ++d) dst[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
        }

        const int sp_dim = hd / 2;
        const int n_freq = sp_dim / 2;
        std::vector<float> inv_freq(n_freq);
        for (int i = 0; i < n_freq; ++i) inv_freq[i] = 1.0f / std::pow(vc.rope_theta, static_cast<float>(2 * i) / sp_dim);
        std::vector<float> h_rc(static_cast<size_t>(S) * hd), h_rs(static_cast<size_t>(S) * hd);
        for (int t = 0; t < S; ++t) {
            for (int i = 0; i < n_freq; ++i) {
                const float aw = static_cast<float>(pos_w[t]) * inv_freq[i];
                const float ah = static_cast<float>(pos_h[t]) * inv_freq[i];
                h_rc[t * hd + 2 * i] = h_rc[t * hd + 2 * i + 1] = std::cos(aw);
                h_rs[t * hd + 2 * i] = h_rs[t * hd + 2 * i + 1] = std::sin(aw);
                h_rc[t * hd + sp_dim + 2 * i] = h_rc[t * hd + sp_dim + 2 * i + 1] = std::cos(ah);
                h_rs[t * hd + sp_dim + 2 * i] = h_rs[t * hd + sp_dim + 2 * i + 1] = std::sin(ah);
            }
        }

        // 2. Allocate temporary GPU buffers for this image & execute 23 ViT layers on GPU!
        GpuBuffer d_patch, d_cur, d_pos, d_norm, d_q, d_k, d_v, d_ao, d_proj, d_mlp, d_rc, d_rs, d_mid, d_out;
        auto cleanup_img = [&]() {
            free_gpu_buffer(d_patch); free_gpu_buffer(d_cur); free_gpu_buffer(d_pos);
            free_gpu_buffer(d_norm); free_gpu_buffer(d_q); free_gpu_buffer(d_k); free_gpu_buffer(d_v);
            free_gpu_buffer(d_ao); free_gpu_buffer(d_proj); free_gpu_buffer(d_mlp);
            free_gpu_buffer(d_rc); free_gpu_buffer(d_rs); free_gpu_buffer(d_mid); free_gpu_buffer(d_out);
        };
        if (!upload_raw(d_patch, patch_buf.data(), patch_buf.size() * 4, err) ||
            !upload_raw(d_pos, h_pos_add.data(), h_pos_add.size() * 4, err) ||
            !upload_raw(d_rc, h_rc.data(), h_rc.size() * 4, err) ||
            !upload_raw(d_rs, h_rs.data(), h_rs.size() * 4, err) ||
            !hip_ok(alloc_gpu_buffer(d_cur, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit cur", err) ||
            !hip_ok(alloc_gpu_buffer(d_norm, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit norm", err) ||
            !hip_ok(alloc_gpu_buffer(d_q, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit q", err) ||
            !hip_ok(alloc_gpu_buffer(d_k, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit k", err) ||
            !hip_ok(alloc_gpu_buffer(d_v, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit v", err) ||
            !hip_ok(alloc_gpu_buffer(d_ao, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit ao", err) ||
            !hip_ok(alloc_gpu_buffer(d_proj, static_cast<size_t>(S) * D * 4, mem_info_.is_uma), "vit proj", err) ||
            !hip_ok(alloc_gpu_buffer(d_mlp, static_cast<size_t>(S) * vc.inter * 4, mem_info_.is_uma), "vit mlp", err) ||
            !hip_ok(alloc_gpu_buffer(d_mid, static_cast<size_t>(n_tok) * vc.proj_hidden * 4, mem_info_.is_uma), "vit mid", err) ||
            !hip_ok(alloc_gpu_buffer(d_out, static_cast<size_t>(n_tok) * vc.out_hidden * 4, mem_info_.is_uma), "vit out", err)) {
            cleanup_img();
            return false;
        }

        float* cur = reinterpret_cast<float*>(d_cur.dev_ptr);
        const int merged_dim = merge * merge * D;
        const size_t vit_xbf_need = static_cast<size_t>(std::max(S, n_tok)) * std::max({D, vc.inter, merged_dim, vc.proj_hidden}) * sizeof(uint16_t);
        if (blaslt_.ready() && sc_x_bf16_.bytes < vit_xbf_need) {
            free_gpu_buffer(sc_x_bf16_);
            (void)alloc_gpu_buffer(sc_x_bf16_, vit_xbf_need, mem_info_.is_uma);
        }
        (void)hipMemsetAsync(cur, 0, static_cast<size_t>(D) * 4, stream_);
        launch_bf16_opt(vit_bf16("model.visual.backbone.embeddings.patch_embedding.weight"),
                        reinterpret_cast<const float*>(d_patch.dev_ptr), nullptr, cur + D,
                        n_patches, D, patch_in);
        k_vec_add<<<(S * D + 255) / 256, 256, 0, stream_>>>(cur, reinterpret_cast<const float*>(d_pos.dev_ptr), cur, S * D);
        k_vit_layernorm_f32<<<S, 256, 0, stream_>>>(cur, vit_f32("model.visual.backbone.ln_pre.weight"),
                                                    vit_f32("model.visual.backbone.ln_pre.bias"), cur, S, D, vc.ln_eps);

        for (int il = 0; il < vc.layers; ++il) {
            const std::string p = "model.visual.backbone.layers." + std::to_string(il);
            k_vit_layernorm_f32<<<S, 256, 0, stream_>>>(cur, vit_f32(p + ".ln_1.weight"), vit_f32(p + ".ln_1.bias"),
                                                        reinterpret_cast<float*>(d_norm.dev_ptr), S, D, vc.ln_eps);
            launch_bf16_opt(vit_bf16(p + ".self_attn.q_proj.weight"), reinterpret_cast<const float*>(d_norm.dev_ptr),
                            vit_f32(p + ".self_attn.q_proj.bias"), reinterpret_cast<float*>(d_q.dev_ptr), S, D, D);
            launch_bf16_opt(vit_bf16(p + ".self_attn.k_proj.weight"), reinterpret_cast<const float*>(d_norm.dev_ptr),
                            vit_f32(p + ".self_attn.k_proj.bias"), reinterpret_cast<float*>(d_k.dev_ptr), S, D, D);
            launch_bf16_opt(vit_bf16(p + ".self_attn.v_proj.weight"), reinterpret_cast<const float*>(d_norm.dev_ptr),
                            vit_f32(p + ".self_attn.v_proj.bias"), reinterpret_cast<float*>(d_v.dev_ptr), S, D, D);
            k_vit_rope_2d<<<dim3(H, S), hd / 2, 0, stream_>>>(
                reinterpret_cast<float*>(d_q.dev_ptr), reinterpret_cast<float*>(d_k.dev_ptr),
                reinterpret_cast<const float*>(d_rc.dev_ptr), reinterpret_cast<const float*>(d_rs.dev_ptr), S, H, hd);
            k_vit_full_attn<<<dim3(H, S), 64, 0, stream_>>>(
                reinterpret_cast<const float*>(d_q.dev_ptr), reinterpret_cast<const float*>(d_k.dev_ptr),
                reinterpret_cast<const float*>(d_v.dev_ptr), reinterpret_cast<float*>(d_ao.dev_ptr), S, H, hd);
            launch_bf16_opt(vit_bf16(p + ".self_attn.out_proj.weight"), reinterpret_cast<const float*>(d_ao.dev_ptr),
                            vit_f32(p + ".self_attn.out_proj.bias"), reinterpret_cast<float*>(d_proj.dev_ptr), S, D, D);
            k_vit_layerscale_res<<<(S * D + 255) / 256, 256, 0, stream_>>>(cur, reinterpret_cast<const float*>(d_proj.dev_ptr),
                                                                           vit_f32(p + ".ls_1.gamma"), S, D);

            k_vit_layernorm_f32<<<S, 256, 0, stream_>>>(cur, vit_f32(p + ".ln_2.weight"), vit_f32(p + ".ln_2.bias"),
                                                        reinterpret_cast<float*>(d_norm.dev_ptr), S, D, vc.ln_eps);
            launch_bf16_opt(vit_bf16(p + ".mlp.c_fc.weight"), reinterpret_cast<const float*>(d_norm.dev_ptr),
                            vit_f32(p + ".mlp.c_fc.bias"), reinterpret_cast<float*>(d_mlp.dev_ptr), S, vc.inter, D);
            k_vit_gelu_exact<<<(S * vc.inter + 255) / 256, 256, 0, stream_>>>(reinterpret_cast<float*>(d_mlp.dev_ptr), S * vc.inter);
            launch_bf16_opt(vit_bf16(p + ".mlp.c_proj.weight"), reinterpret_cast<const float*>(d_mlp.dev_ptr),
                            vit_f32(p + ".mlp.c_proj.bias"), reinterpret_cast<float*>(d_proj.dev_ptr), S, D, vc.inter);
            k_vit_layerscale_res<<<(S * D + 255) / 256, 256, 0, stream_>>>(cur, reinterpret_cast<const float*>(d_proj.dev_ptr),
                                                                           vit_f32(p + ".ls_2.gamma"), S, D);
            if (guard_.visit_flush_stride > 0 && ((il + 1) % guard_.visit_flush_stride == 0)) {
                (void)hipStreamSynchronize(stream_);
            }
        }

        // 2x2 merge projector (drop CLS at index 0)
        launch_bf16_opt(vit_bf16("model.visual.proj_in.weight"), cur + D, nullptr,
                        reinterpret_cast<float*>(d_mid.dev_ptr), n_tok, vc.proj_hidden, merged_dim);
        k_vit_gelu_exact<<<(n_tok * vc.proj_hidden + 255) / 256, 256, 0, stream_>>>(reinterpret_cast<float*>(d_mid.dev_ptr), n_tok * vc.proj_hidden);
        launch_bf16_opt(vit_bf16("model.visual.proj_out.weight"), reinterpret_cast<const float*>(d_mid.dev_ptr), nullptr,
                        reinterpret_cast<float*>(d_out.dev_ptr), n_tok, vc.out_hidden, vc.proj_hidden);

        out_embeds.resize(static_cast<size_t>(n_tok) * vc.out_hidden);
        const bool ok = hip_ok(hipMemcpyAsync(out_embeds.data(), d_out.dev_ptr, out_embeds.size() * 4, hipMemcpyDeviceToHost, stream_), "vit D2H", err)
                     && hip_ok(hipStreamSynchronize(stream_), "vit sync", err);
        cleanup_img();
        return ok;
    }

    bool selfcheck(std::string& err) override {
        // Verify SF4 GPU GEMV & WMMA parity against exact CPU dequant on active GPU!
        const int M = 8, N = 64, K = 2304;
        std::vector<uint8_t> h_p(static_cast<size_t>(N) * (K / 2));
        std::vector<uint8_t> h_s(static_cast<size_t>(N) * (K / 16));
        std::vector<half_t> h_x(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < h_p.size(); ++i) h_p[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFFu);
        for (size_t i = 0; i < h_s.size(); ++i) h_s[i] = static_cast<uint8_t>(0x38u + (i % 16));
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = static_cast<half_t>(std::sin(static_cast<float>(i) * 0.013f));

        GpuBuffer dp, ds, dx, dy;
        auto rel = [&]() { free_gpu_buffer(dp); free_gpu_buffer(ds); free_gpu_buffer(dx); free_gpu_buffer(dy); };
        if (!upload_raw(dp, h_p.data(), h_p.size(), err) ||
            !upload_raw(ds, h_s.data(), h_s.size(), err) ||
            !upload_raw(dx, h_x.data(), h_x.size() * sizeof(half_t), err) ||
            !hip_ok(alloc_gpu_buffer(dy, static_cast<size_t>(M) * N * 4, mem_info_.is_uma), "sc dy", err)) {
            rel();
            return false;
        }
        launch_sf4_linear(dp.dev_ptr, ds.dev_ptr, 0.25f, reinterpret_cast<const half_t*>(dx.dev_ptr),
                          reinterpret_cast<float*>(dy.dev_ptr), M, N, K, mem_info_.gen, stream_, false);
        std::vector<float> got(static_cast<size_t>(M) * N);
        if (!hip_ok(hipMemcpyAsync(got.data(), dy.dev_ptr, got.size() * 4, hipMemcpyDeviceToHost, stream_), "sc D2H", err) ||
            !hip_ok(hipStreamSynchronize(stream_), "sc sync", err)) {
            rel();
            return false;
        }
        rel();

        std::vector<float> w_dec(K);
        double num = 0.0, den = 0.0;
        for (int n = 0; n < N; ++n) {
            dequant_sf4_row(h_p.data() + static_cast<size_t>(n) * (K / 2), h_s.data() + static_cast<size_t>(n) * (K / 16), 0.25f, w_dec.data(), K);
            for (int m = 0; m < M; ++m) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += static_cast<double>(w_dec[k]) * static_cast<double>(static_cast<float>(h_x[static_cast<size_t>(m) * K + k]));
                const double d = acc - static_cast<double>(got[static_cast<size_t>(m) * N + n]);
                num += d * d;
                den += acc * acc;
            }
        }
        const double rel_l2 = std::sqrt(num / (den + 1e-30));
        if (rel_l2 > 1e-2) {
            err = "hip SF4 selfcheck failed: rel_l2=" + std::to_string(rel_l2);
            return false;
        }
        if (model_ && model_->kind == ModelSourceKind::Strata) {
            std::vector<uint8_t> s5;
            if (!model_->rd.read_section(5, s5, err)) return false;
            if (s5.empty()) { err = "Section 5 sq.selfcheck is empty"; return false; }
        }
        return true;
    }

private:
    static bool is_large_2d_matmul(const std::string& name, const TInfo& ti) {
        if (ti.shape.size() != 2) return false;
        if (name.find("attn_res") != std::string::npos ||
            name.find("cross_pass") != std::string::npos ||
            name.find("sconv") != std::string::npos ||
            name.find("rel_logits_proj") != std::string::npos ||
            name.find("mlp.gate") != std::string::npos ||
            name.find("attn_sink_proj") != std::string::npos) {
            return false;
        }
        return ti.numel() >= 32768;
    }

    void HIP_ZERO(GpuBuffer& b) {
        if (b.dev_ptr && b.bytes > 0) (void)hipMemset(b.dev_ptr, 0, b.bytes);
    }

    bool upload_raw(GpuBuffer& dst, const void* src, size_t bytes, std::string& err) {
        if (!hip_ok(alloc_gpu_buffer(dst, bytes, mem_info_.is_uma), "alloc_gpu_buffer", err)) return false;
        if (bytes > 0 && !hip_ok(hipMemcpy(dst.dev_ptr, src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D", err)) return false;
        return true;
    }

    const float* f32_ptr(const std::string& k) const {
        return reinterpret_cast<const float*>(trunk_f32_.at(k).dev_ptr);
    }
    const uint16_t* bf16_ptr(const std::string& k) const {
        return reinterpret_cast<const uint16_t*>(trunk_bf16_.at(k).dev_ptr);
    }
    const float* ds_f32(const std::string& k) const {
        return reinterpret_cast<const float*>(dspark_f32_.at(k).dev_ptr);
    }
    const uint16_t* ds_bf16(const std::string& k) const {
        return reinterpret_cast<const uint16_t*>(dspark_bf16_.at(k).dev_ptr);
    }
    const float* vit_f32(const std::string& k) const {
        auto it = vision_.w_f32.find(k);
        return it == vision_.w_f32.end() ? nullptr : reinterpret_cast<const float*>(it->second.dev_ptr);
    }
    const uint16_t* vit_bf16(const std::string& k) const {
        return reinterpret_cast<const uint16_t*>(vision_.w_bf16.at(k).dev_ptr);
    }

    void convert_x_to_bf16(const float* X, uint16_t* dst, int total) {
        if ((total & 3) == 0) {
            const int n4 = total >> 2;
            k_f32_to_bf16_vec4<<<(n4 + 255) / 256, 256, 0, stream_>>>(X, dst, n4);
        } else {
            k_f32_to_bf16_vec<<<(total + 255) / 256, 256, 0, stream_>>>(X, dst, total);
        }
    }

    void launch_bf16_opt(const uint16_t* W, const float* X, const float* bias, float* Y, int M, int N, int K) {
        if (blaslt_.ready() && M >= blaslt_min_m_ && N >= 64 && K >= 64) {
            const size_t need = static_cast<size_t>(M) * K * sizeof(uint16_t);
            if (sc_x_bf16_.bytes >= need) {
                uint16_t* x_bf = reinterpret_cast<uint16_t*>(sc_x_bf16_.dev_ptr);
                convert_x_to_bf16(X, x_bf, M * K);
                if (blaslt_.gemm_bf16_f32(W, x_bf, Y, M, N, K, stream_, 1.0f)) {
                    if (bias) {
                        k_add_bias_rows<<<(M * N + 255) / 256, 256, 0, stream_>>>(Y, bias, M, N);
                    }
                    return;
                }
            }
        }
        // Wave-5 MALL-Swizzled BM=128, BN=128, BK=32 BF16 WMMA on RDNA3/3.5/4 (pre-converts X once -> 32x less W DRAM!)
        if (M >= 64 && N >= 128 && K >= 64 && (K & 31) == 0 &&
            mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
            const size_t need = static_cast<size_t>(M) * K * sizeof(uint16_t);
            if (sc_x_bf16_.bytes >= need) {
                uint16_t* x_bf = reinterpret_cast<uint16_t*>(sc_x_bf16_.dev_ptr);
                convert_x_to_bf16(X, x_bf, M * K);
                const int tiles = ((M + 127) / 128) * ((N + 127) / 128);
                k_bf16_wmma_bf16_128x128_swiz<<<tiles, 256, 0, stream_>>>(W, x_bf, bias, Y, M, N, K);
                return;
            }
        }
        launch_bf16_linear(W, X, bias, Y, M, N, K, mem_info_.gen, stream_);
    }

    bool read_expert_for_hip(size_t dir_idx, std::vector<uint8_t>& out_body,
                             uint32_t& out_dn_off, uint8_t& out_gu_code, uint8_t& out_dn_code,
                             std::string& err) const {
        if (dir_idx < expert_native_gguf_.size() && expert_native_gguf_[dir_idx]) {
            if (!model_->read_expert_body(dir_idx, out_body, err)) return false;
            out_dn_off  = adj_dir_[dir_idx].dn_offset;
            out_gu_code = adj_dir_[dir_idx].gu_code_id;
            out_dn_code = adj_dir_[dir_idx].dn_code_id;
            return true;
        }
        return transcode::read_or_transcode_expert(*model_, dir_idx, out_body, out_dn_off, out_gu_code, out_dn_code, err);
    }

    // L3-Resident Expert Unpack-Once + hipBLASLt FP16 GEMM helper (used by both full-GPU prefill & double-buffered hybrid prefill!)
    void run_single_expert_blaslt(
        const uint8_t* body_dev, uint8_t gu_code, uint8_t dn_code, uint32_t dn_offset,
        float sf4_w13_outer, float sf4_w2_outer,
        const half_t* x_in_f16, float* gu_buf, half_t* act_buf, float* y_out_f32,
        int m_e, int max_gu_rows, int D, int inter) {
        half_t* w13_f16 = reinterpret_cast<half_t*>(d_exp_w13_f16_.dev_ptr);
        half_t* w2_f16  = reinterpret_cast<half_t*>(d_exp_w2_f16_.dev_ptr);
        const int n13 = 2 * inter;
        const int chunk_cap = std::max(1, max_gu_rows);

        // 1. Unpack w13 [2*inter, D] ONCE -> L3/MALL-resident 11.8 MB FP16 buffer -> chunked hipBLASLt FP16 GEMM + SITU!
        float alpha13 = 1.0f;
        if (gu_code == CODE_SF4) {
            const uint8_t* p = body_dev + 16;
            const uint8_t* s = p + static_cast<size_t>(n13) * (D >> 1);
            k_unpack_sf4_f16<<<(n13 * (D >> 4) + 255) / 256, 256, 0, stream_>>>(p, s, w13_f16, n13, D);
            alpha13 = sf4_w13_outer * 16.0f;
        } else if (gu_code == CODE_MX8) {
            const uint8_t* w = body_dev;
            const uint8_t* s = w + static_cast<size_t>(n13) * D;
            k_unpack_mx8_f16<<<(n13 * (D >> 5) + 255) / 256, 256, 0, stream_>>>(w, s, w13_f16, n13, D);
        } else if (gu_code == CODE_GGUF_RAW) {
            k_unpack_gguf_f16<<<(n13 * (D >> 3) + 255) / 256, 256, 0, stream_>>>(body_dev, 0, 1, w13_f16, n13, D, inter);
        } else {
            const uint8_t* gu_ptr = body_dev + static_cast<size_t>(inter) * 2;
            launch_unpack_sk_f16_dispatch(gu_ptr, gu_code, w13_f16, n13, D, stream_);
        }
        const half_t* sq_dn = (dn_code != CODE_SF4 && dn_code != CODE_MX8 && dn_code != CODE_GGUF_RAW)
            ? reinterpret_cast<const half_t*>(body_dev) : nullptr;
        // Slice m_e by chunk_cap so gu_buf [max_gu_rows, 2*inter] NEVER overflows when m_e > B (#6580)!
        for (int off = 0; off < m_e; off += chunk_cap) {
            const int cur_m = std::min(chunk_cap, m_e - off);
            const half_t* x_sub = x_in_f16 + static_cast<size_t>(off) * D;
            half_t* act_sub = act_buf + static_cast<size_t>(off) * inter;
            if (!blaslt_.gemm_f16_f32(w13_f16, x_sub, gu_buf, cur_m, n13, D, stream_, alpha13)) {
                if (gu_code == CODE_SF4) {
                    const uint8_t* p = body_dev + 16;
                    launch_sf4_linear(p, p + static_cast<size_t>(n13) * (D >> 1), sf4_w13_outer, x_sub, gu_buf, cur_m, n13, D, mem_info_.gen, stream_, false);
                } else if (gu_code == CODE_MX8) {
                    launch_mx8_linear(body_dev, body_dev + static_cast<size_t>(n13) * D, x_sub, gu_buf, cur_m, n13, D, mem_info_.gen, stream_);
                } else if (gu_code == CODE_GGUF_RAW) {
                    launch_gguf_linear(body_dev, 0, 1, x_sub, gu_buf, cur_m, n13, D, inter, stream_);
                } else {
                    launch_sk_linear(body_dev + static_cast<size_t>(inter) * 2, gu_code, x_sub, gu_buf, cur_m, n13, D, mem_info_.gen, stream_);
                }
            }
            // 2. Fused SITU + optional S_K sq_xd_dn divide directly on FP32 gu_buf -> FP16 act_sub!
            k_situ_interleaved_sq<<<(cur_m * inter + 255) / 256, 256, 0, stream_>>>(
                gu_buf, sq_dn, act_sub, cur_m, inter, c_.situ_beta, c_.situ_lb);
        }

        // 3. Unpack w2 [D, inter] ONCE -> L3/MALL-resident 5.9 MB FP16 buffer -> hipBLASLt FP16 GEMM!
        float alpha2 = 1.0f;
        if (dn_code == CODE_SF4) {
            const uint8_t* p = body_dev + dn_offset;
            const uint8_t* s = p + static_cast<size_t>(D) * (inter >> 1);
            k_unpack_sf4_f16<<<(D * (inter >> 4) + 255) / 256, 256, 0, stream_>>>(p, s, w2_f16, D, inter);
            alpha2 = sf4_w2_outer * 16.0f;
        } else if (dn_code == CODE_MX8) {
            const uint8_t* w = body_dev + dn_offset;
            const uint8_t* s = w + static_cast<size_t>(D) * inter;
            k_unpack_mx8_f16<<<(D * (inter >> 5) + 255) / 256, 256, 0, stream_>>>(w, s, w2_f16, D, inter);
        } else if (dn_code == CODE_GGUF_RAW) {
            k_unpack_gguf_f16<<<(D * (inter >> 3) + 255) / 256, 256, 0, stream_>>>(body_dev, dn_offset, 0, w2_f16, D, inter, inter);
        } else {
            launch_unpack_sk_paired2_f16_dispatch(body_dev + dn_offset, dn_code, w2_f16, D, inter, stream_);
        }
        if (!blaslt_.gemm_f16_f32(w2_f16, act_buf, y_out_f32, m_e, D, inter, stream_, alpha2)) {
            if (dn_code == CODE_SF4) {
                const uint8_t* p = body_dev + dn_offset;
                launch_sf4_linear(p, p + static_cast<size_t>(D) * (inter >> 1), sf4_w2_outer, act_buf, y_out_f32, m_e, D, inter, mem_info_.gen, stream_, false);
            } else if (dn_code == CODE_MX8) {
                launch_mx8_linear(body_dev + dn_offset, body_dev + dn_offset + static_cast<size_t>(D) * inter, act_buf, y_out_f32, m_e, D, inter, mem_info_.gen, stream_);
            } else if (dn_code == CODE_GGUF_RAW) {
                launch_gguf_linear(body_dev, dn_offset, 0, act_buf, y_out_f32, m_e, D, inter, inter, stream_);
            } else {
                launch_sk_paired2_linear(body_dev + dn_offset, dn_code, reinterpret_cast<const half_t*>(body_dev), act_buf, y_out_f32, m_e, D, inter, stream_);
            }
        }
    }

    void launch_trunk_linear(const std::string& key, const float* X, float* Y, int M, int N, int K) {
        auto it_bf = trunk_bf16_.find(key);
        if (it_bf != trunk_bf16_.end()) {
            launch_bf16_opt(reinterpret_cast<const uint16_t*>(it_bf->second.dev_ptr), X, nullptr, Y, M, N, K);
        } else {
            launch_f32_linear(reinterpret_cast<const float*>(trunk_f32_.at(key).dev_ptr), X, nullptr, Y, M, N, K, stream_, mem_info_.gen);
        }
    }

    void clear_slot_sconv(int slot) {
        const int kv_dim = c_.n_kv * c_.head_dim;
        k_sconv_clear_pair<<<dim3((kv_dim + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<uint16_t*>(d_sconv_k_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(d_sconv_v_all_.dev_ptr),
            slot, max_slots_, c_.n_visits, kv_dim);
        k_sconv_clear_pair<<<dim3((c_.hidden + 255) / 256, c_.n_visits), 256, 0, stream_>>>(
            reinterpret_cast<uint16_t*>(d_sconv_attn_all_.dev_ptr),
            reinterpret_cast<uint16_t*>(d_sconv_mlp_all_.dev_ptr),
            slot, max_slots_, c_.n_visits, c_.hidden);
    }

    size_t vram_used_bytes() const noexcept {
        size_t tot = expert_gpu_arena_.bytes + d_exp_stage_.bytes + d_exp_w13_f16_.bytes + d_exp_w2_f16_.bytes;
        tot += hybrid_.allocated_bytes();
        for (const auto& b : trunk_qkvrg_) tot += b.bytes;
        for (const auto& kv : trunk_bf16_) tot += kv.second.bytes;
        for (const auto& kv : trunk_f32_) tot += kv.second.bytes;
        for (const auto& kv : packed_trunk_) tot += kv.second.buf.bytes;
        for (const auto& kv : static_sconv_) tot += kv.second.bytes;
        for (const auto& b : d_msg_queries_) tot += b.bytes;
        for (const auto& kv : dspark_bf16_) tot += kv.second.bytes;
        for (const auto& kv : dspark_f32_) tot += kv.second.bytes;
        for (const auto& p : swa_pools_) tot += p.allocated_bytes();
        for (const auto& p : glob_pools_) tot += p.allocated_bytes();
        tot += d_sconv_k_all_.bytes + d_sconv_v_all_.bytes + d_sconv_attn_all_.bytes + d_sconv_mlp_all_.bytes;
        return tot;
    }

    bool ensure_peer_scratch(int B, std::string& err) {
        if (!ep2_active_ || B <= peer_scratch_B_) return true;
        const int D = c_.hidden;
        const int inter = c_.inter;
        const size_t tot_pairs = static_cast<size_t>(B) * c_.top_k;
        auto re0 = [&](GpuBuffer& b, size_t bytes) -> bool {
            free_gpu_buffer(b);
            return hip_ok(alloc_gpu_buffer(b, bytes, mem_info_.is_uma), "ep2 g0 scratch", err);
        };
        if (!re0(sc_peer_moe_out_, static_cast<size_t>(B) * D * sizeof(float))) return false;

        if (!ep2_p2p_) {
            if (h_ep_x_f16_) { (void)hipHostFree(h_ep_x_f16_); h_ep_x_f16_ = nullptr; }
            if (h_ep_ids_)   { (void)hipHostFree(h_ep_ids_);   h_ep_ids_ = nullptr; }
            if (h_ep_ws_)    { (void)hipHostFree(h_ep_ws_);    h_ep_ws_ = nullptr; }
            if (h_ep_y1_)    { (void)hipHostFree(h_ep_y1_);    h_ep_y1_ = nullptr; }
            if (!hip_ok(hipHostMalloc(reinterpret_cast<void**>(&h_ep_x_f16_), static_cast<size_t>(B) * D * sizeof(half_t), hipHostMallocDefault), "ep2 host x", err) ||
                !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&h_ep_ids_),   tot_pairs * sizeof(int32_t), hipHostMallocDefault), "ep2 host ids", err) ||
                !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&h_ep_ws_),    tot_pairs * sizeof(float), hipHostMallocDefault), "ep2 host ws", err) ||
                !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&h_ep_y1_),    static_cast<size_t>(B) * D * sizeof(float), hipHostMallocDefault), "ep2 host y1", err)) {
                return false;
            }
        }

        if (!hip_ok(hipSetDevice(gpu1_id_), "hipSetDevice peer scratch", err)) return false;
        auto re1 = [&](GpuBuffer& b, size_t bytes) -> bool {
            free_gpu_buffer(b);
            return hip_ok(alloc_gpu_buffer(b, bytes, peer_mem_info_.is_uma), "ep2 g1 scratch", err);
        };
        const bool ok =
            re1(peer_sc_mlp_f16_,    static_cast<size_t>(B) * D * sizeof(half_t)) &&
            re1(peer_sc_topk_ids_,   tot_pairs * sizeof(int32_t)) &&
            re1(peer_sc_topk_w_,     tot_pairs * sizeof(float)) &&
            re1(peer_sc_moe_act_,    tot_pairs * inter * sizeof(half_t)) &&
            re1(peer_sc_moe_out_,    static_cast<size_t>(B) * D * sizeof(float)) &&
            re1(peer_sc_exp_off_,    129u * sizeof(int32_t)) &&
            re1(peer_sc_exp_cur_,    129u * sizeof(int32_t)) &&
            re1(peer_sc_bucket_tok_, tot_pairs * sizeof(int32_t)) &&
            re1(peer_sc_slotk_bidx_, tot_pairs * sizeof(int32_t)) &&
            re1(peer_sc_bucket_y_,   tot_pairs * D * sizeof(float));
        (void)hipSetDevice(gpu0_id_);
        if (!ok) return false;
        peer_scratch_B_ = B;
        return true;
    }

    // Launch GPU 1 peer experts asynchronously on peer_stream_ (overlaps with GPU 0 experts on stream_!).
    bool launch_peer_moe_async(int ip, int B, const half_t* mlpf16_g0, const int32_t* tk_i_g0, const float* tk_w_g0, std::string& err) {
        if (!ensure_peer_scratch(B, err)) return false;
        const int D = c_.hidden;
        const int tot_pairs = B * c_.top_k;
        const size_t x_bytes = static_cast<size_t>(B) * D * sizeof(half_t);
        const size_t idx_bytes = static_cast<size_t>(tot_pairs) * sizeof(int32_t);
        const size_t w_bytes = static_cast<size_t>(tot_pairs) * sizeof(float);
        const size_t y_bytes = static_cast<size_t>(B) * D * sizeof(float);

        if (ep2_p2p_) {
            if (!hip_ok(hipEventRecord(ev_g0_to_g1_, stream_), "ep2 rec g0", err)) return false;
            if (!hip_ok(hipSetDevice(gpu1_id_), "ep2 set g1", err)) return false;
            if (!hip_ok(hipStreamWaitEvent(peer_stream_, ev_g0_to_g1_, 0), "ep2 wait g0", err)) {
                (void)hipSetDevice(gpu0_id_); return false;
            }
            if (!hip_ok(hipMemcpyPeerAsync(peer_sc_mlp_f16_.dev_ptr, gpu1_id_, mlpf16_g0, gpu0_id_, x_bytes, peer_stream_), "ep2 p2p x", err) ||
                !hip_ok(hipMemcpyPeerAsync(peer_sc_topk_ids_.dev_ptr, gpu1_id_, tk_i_g0, gpu0_id_, idx_bytes, peer_stream_), "ep2 p2p ids", err) ||
                !hip_ok(hipMemcpyPeerAsync(peer_sc_topk_w_.dev_ptr, gpu1_id_, tk_w_g0, gpu0_id_, w_bytes, peer_stream_), "ep2 p2p ws", err)) {
                (void)hipSetDevice(gpu0_id_); return false;
            }
        } else {
            if (!hip_ok(hipMemcpyAsync(h_ep_x_f16_, mlpf16_g0, x_bytes, hipMemcpyDeviceToHost, stream_), "ep2 d2h x", err) ||
                !hip_ok(hipMemcpyAsync(h_ep_ids_, tk_i_g0, idx_bytes, hipMemcpyDeviceToHost, stream_), "ep2 d2h ids", err) ||
                !hip_ok(hipMemcpyAsync(h_ep_ws_, tk_w_g0, w_bytes, hipMemcpyDeviceToHost, stream_), "ep2 d2h ws", err) ||
                !hip_ok(hipStreamSynchronize(stream_), "ep2 sync g0->h", err)) {
                return false;
            }
            if (!hip_ok(hipSetDevice(gpu1_id_), "ep2 set g1 bounce", err)) return false;
            if (!hip_ok(hipMemcpyAsync(peer_sc_mlp_f16_.dev_ptr, h_ep_x_f16_, x_bytes, hipMemcpyHostToDevice, peer_stream_), "ep2 h2d x", err) ||
                !hip_ok(hipMemcpyAsync(peer_sc_topk_ids_.dev_ptr, h_ep_ids_, idx_bytes, hipMemcpyHostToDevice, peer_stream_), "ep2 h2d ids", err) ||
                !hip_ok(hipMemcpyAsync(peer_sc_topk_w_.dev_ptr, h_ep_ws_, w_bytes, hipMemcpyHostToDevice, peer_stream_), "ep2 h2d ws", err)) {
                (void)hipSetDevice(gpu0_id_); return false;
            }
        }

        const GpuExpertDesc* p_exps = reinterpret_cast<const GpuExpertDesc*>(peer_d_layer_exps_[ip].dev_ptr);
        const half_t*  p_x   = reinterpret_cast<const half_t*>(peer_sc_mlp_f16_.dev_ptr);
        const int32_t* p_tki = reinterpret_cast<const int32_t*>(peer_sc_topk_ids_.dev_ptr);
        const float*   p_tkw = reinterpret_cast<const float*>(peer_sc_topk_w_.dev_ptr);
        half_t*        p_act = reinterpret_cast<half_t*>(peer_sc_moe_act_.dev_ptr);
        float*         p_by  = reinterpret_cast<float*>(peer_sc_bucket_y_.dev_ptr);
        float*         p_out = reinterpret_cast<float*>(peer_sc_moe_out_.dev_ptr);

        if (B <= 4) {
            k_moe_direct_w13_situ<<<dim3((c_.inter + 3) / 4, tot_pairs), 256, 0, peer_stream_>>>(
                p_exps, p_tki, p_x, p_act, B, c_.top_k, D, c_.inter, c_.situ_beta, c_.situ_lb);
            k_moe_direct_w2_par<<<dim3((D + 7) / 8, tot_pairs), 256, 0, peer_stream_>>>(
                p_exps, p_tki, p_act, p_by, B, c_.top_k, D, c_.inter);
            k_moe_direct_reduce<<<dim3((D + 255) / 256, B), 256, 0, peer_stream_>>>(
                p_by, p_tkw, p_out, B, c_.top_k, D);
        } else {
            int32_t* p_off = reinterpret_cast<int32_t*>(peer_sc_exp_off_.dev_ptr);
            int32_t* p_cur = reinterpret_cast<int32_t*>(peer_sc_exp_cur_.dev_ptr);
            int32_t* p_btk = reinterpret_cast<int32_t*>(peer_sc_bucket_tok_.dev_ptr);
            int32_t* p_s2b = reinterpret_cast<int32_t*>(peer_sc_slotk_bidx_.dev_ptr);
            if (B >= 16) {
                (void)hipMemsetAsync(p_off, 0, 129 * sizeof(int32_t), peer_stream_);
                const int cblks = std::min(64, (tot_pairs + 255) / 256);
                k_moe_count_buckets<<<cblks, 256, 0, peer_stream_>>>(p_tki, p_off, tot_pairs);
                k_moe_prefix_buckets<<<1, 128, 0, peer_stream_>>>(p_off, p_off, p_cur);
                k_moe_scatter_buckets<<<(tot_pairs + 255) / 256, 256, 0, peer_stream_>>>(p_tki, p_cur, p_btk, p_s2b, tot_pairs, c_.top_k);
            } else {
                k_moe_build_buckets<<<1, 128, 0, peer_stream_>>>(p_tki, p_off, p_btk, p_s2b, B, c_.top_k);
            }
            if (peer_mem_info_.gen != AmdGen::Rdna2 && peer_mem_info_.gen != AmdGen::Unknown) {
                const int mt = (B >= 1024) ? 8 : ((B >= 256) ? 4 : ((B >= 64) ? 2 : 1));
                const dim3 g13((2 * c_.inter + 127) / 128, c_.n_experts, mt);
                const dim3 g2((D + 127) / 128, c_.n_experts, mt);
                if (B >= 64) {
                    k_moe_grouped_w13_situ_wmma_128<<<g13, 256, 0, peer_stream_>>>(
                        p_exps, p_off, p_btk, p_x, p_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    k_moe_grouped_w2_wmma_128<<<g2, 256, 0, peer_stream_>>>(
                        p_exps, p_off, p_act, p_by, D, c_.inter);
                } else {
                    k_moe_grouped_w13_situ_wmma<<<g13, 256, 0, peer_stream_>>>(
                        p_exps, p_off, p_btk, p_x, p_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    k_moe_grouped_w2_wmma<<<g2, 256, 0, peer_stream_>>>(
                        p_exps, p_off, p_act, p_by, D, c_.inter);
                }
            } else {
                k_moe_grouped_w13_situ<<<dim3((c_.inter + 3) / 4, c_.n_experts), 256, 0, peer_stream_>>>(
                    p_exps, p_off, p_btk, p_x, p_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                k_moe_grouped_w2<<<dim3((D + 7) / 8, c_.n_experts), 256, 0, peer_stream_>>>(
                    p_exps, p_off, p_act, p_by, D, c_.inter);
            }
            k_moe_reduce_buckets_masked<<<dim3((D + 255) / 256, B), 256, 0, peer_stream_>>>(
                p_exps, p_tki, p_by, p_s2b, p_tkw, p_out, B, c_.top_k, D);
        }

        if (ep2_p2p_) {
            if (!hip_ok(hipMemcpyPeerAsync(sc_peer_moe_out_.dev_ptr, gpu0_id_, p_out, gpu1_id_, y_bytes, peer_stream_), "ep2 p2p y", err)) {
                (void)hipSetDevice(gpu0_id_); return false;
            }
        } else {
            if (!hip_ok(hipMemcpyAsync(h_ep_y1_, p_out, y_bytes, hipMemcpyDeviceToHost, peer_stream_), "ep2 d2h y", err)) {
                (void)hipSetDevice(gpu0_id_); return false;
            }
        }
        if (!hip_ok(hipEventRecord(ev_g1_done_, peer_stream_), "ep2 rec g1", err)) {
            (void)hipSetDevice(gpu0_id_); return false;
        }
        return hip_ok(hipSetDevice(gpu0_id_), "ep2 restore g0", err);
    }

    bool finish_peer_moe_add(int B, float* moeout_g0, std::string& err) {
        const int D = c_.hidden;
        if (ep2_p2p_) {
            if (!hip_ok(hipStreamWaitEvent(stream_, ev_g1_done_, 0), "ep2 wait g1", err)) return false;
        } else {
            if (!hip_ok(hipEventSynchronize(ev_g1_done_), "ep2 sync g1", err)) return false;
            if (!hip_ok(hipMemcpyAsync(sc_peer_moe_out_.dev_ptr, h_ep_y1_, static_cast<size_t>(B) * D * sizeof(float),
                                       hipMemcpyHostToDevice, stream_), "ep2 h2d y", err)) {
                return false;
            }
        }
        k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(
            moeout_g0, reinterpret_cast<const float*>(sc_peer_moe_out_.dev_ptr), moeout_g0, B * D);
        return true;
    }

    bool ensure_batch_scratch(int B, std::string& err) {
        if (B <= scratch_B_) return true;
        const int D = c_.hidden;
        const int q_dim = c_.n_head * c_.head_dim;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const int r_dim = c_.n_head * c_.d_rel;
        const int qkvrg_dim = 2 * q_dim + r_dim + 2 * kv_dim;
        const int inter = c_.inter;
        const int n_exp = c_.n_experts;
        // #6580: Per-expert token count m_e can reach B * top_k (not just B). Cap single-expert scratch to
        // moe_gu_cap <= 512 (9 MiB max for sc_moe_gu_) and chunk any m_e > moe_gu_cap safely in callers!
        const int moe_gu_cap = std::max(B, std::min(B * c_.top_k, 512));
        const size_t x_bf16_k = static_cast<size_t>(std::max({2 * D, q_dim, 4096}));
        auto re = [&](GpuBuffer& b, size_t bytes) -> bool {
            free_gpu_buffer(b);
            return hip_ok(alloc_gpu_buffer(b, bytes, mem_info_.is_uma), "scratch alloc", err);
        };
        if (!re(sc_slot_ids_, B * 4) || !re(sc_pos_, B * 4) || !re(sc_tok_ids_, B * 4) || !re(sc_img_idx_, B * 4) ||
            !re(sc_want_idx_, B * 4) || !re(sc_grp_slot_, B * 4) || !re(sc_grp_off_, (B + 1) * 4) || !re(sc_grp_tok_, B * 4) ||
            !re(sc_x_bf16_, static_cast<size_t>(std::max(B, 1025)) * x_bf16_k * sizeof(uint16_t)) ||
            !re(sc_qkvrg_out_, static_cast<size_t>(B) * qkvrg_dim * 4) ||
            !re(sc_cur_, static_cast<size_t>(B) * D * 4) || !re(sc_res_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_in0_, static_cast<size_t>(B) * D * 4) || !re(sc_in1_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_in2_, static_cast<size_t>(B) * D * 4) || !re(sc_mx3_, static_cast<size_t>(B) * 3 * D * 4) ||
            !re(sc_msgs_, static_cast<size_t>(B) * 4 * D * 4) || !re(sc_ctrl_tmp_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_ctrl_in_, static_cast<size_t>(B) * 2 * D * 4) || !re(sc_ctrl_gain_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_ctrl_forget_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_q_, static_cast<size_t>(B) * q_dim * 4) || !re(sc_k_raw_, static_cast<size_t>(B) * kv_dim * 4) ||
            !re(sc_v_raw_, static_cast<size_t>(B) * kv_dim * 4) || !re(sc_k_conv_, static_cast<size_t>(B) * kv_dim * 4) ||
            !re(sc_v_conv_, static_cast<size_t>(B) * kv_dim * 4) || !re(sc_r_raw_, static_cast<size_t>(B) * r_dim * 4) ||
            !re(sc_rel_bias_, static_cast<size_t>(B) * c_.n_head * c_.rel_extent * 4) ||
            !re(sc_sinks_, static_cast<size_t>(B) * c_.n_head * 4) || !re(sc_gate_a_, static_cast<size_t>(B) * q_dim * 4) ||
            !re(sc_k_dyn_, static_cast<size_t>(B) * c_.n_kv * 4 * 4) || !re(sc_v_dyn_, static_cast<size_t>(B) * c_.n_kv * 4 * 4) ||
            !re(sc_attn_out_, static_cast<size_t>(B) * q_dim * 4) ||
            !re(sc_splitkv_part_, static_cast<size_t>(std::max(B, 4)) * c_.n_head * 4 * 130 * 4) ||
            !re(sc_o_proj_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_a_sconv_, static_cast<size_t>(B) * D * 4) || !re(sc_mlp_in_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_mlp_f16_, static_cast<size_t>(moe_gu_cap) * D * sizeof(half_t)) ||
            !re(sc_r_logits_, static_cast<size_t>(B) * n_exp * 4) ||
            !re(sc_topk_ids_, static_cast<size_t>(B) * c_.top_k * 4) || !re(sc_topk_w_, static_cast<size_t>(B) * c_.top_k * 4) ||
            !re(sc_moe_act_, static_cast<size_t>(B) * c_.top_k * inter * sizeof(half_t)) ||
            !re(sc_moe_gu_, static_cast<size_t>(moe_gu_cap) * 2 * inter * 4) ||
            !re(sc_moe_ye_, static_cast<size_t>(moe_gu_cap) * D * 4) ||
            !re(sc_exp_off_, 129 * 4) ||
            !re(sc_exp_cur_, 128 * 4) ||
            !re(sc_x_q8_, static_cast<size_t>(B) * D) ||
            !re(sc_x_d_, static_cast<size_t>(B) * (D >> 4) * 4) ||
            !re(sc_act_q8_, static_cast<size_t>(B) * c_.top_k * inter) ||
            !re(sc_act_d_, static_cast<size_t>(B) * c_.top_k * (inter >> 4) * 4) ||
            !re(sc_bucket_tok_, static_cast<size_t>(B) * c_.top_k * 4) ||
            !re(sc_slotk_bidx_, static_cast<size_t>(B) * c_.top_k * 4) ||
            !re(sc_bucket_x_f16_, static_cast<size_t>(B) * c_.top_k * D * sizeof(half_t)) ||
            !re(sc_bucket_y_, static_cast<size_t>(B) * c_.top_k * D * 4) ||
            !re(sc_moe_out_, static_cast<size_t>(B) * D * 4) || !re(sc_m_sconv_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_exp_tok_idx_, static_cast<size_t>(moe_gu_cap) * 4) || !re(sc_exp_tok_w_, static_cast<size_t>(moe_gu_cap) * 4) ||
            !re(sc_h_norm_, static_cast<size_t>(B) * D * 4) ||
            !re(sc_blk_ptrs_, 16 * sizeof(float*)) ||
            !re(sc_prelude_, static_cast<size_t>(B) * D * 4) || !re(sc_bank_out_, static_cast<size_t>(B) * D * 4)) {
            return false;
        }
        // Depth block pool (max 8 AttnRes blocks across 24 physical layers + 4 bank slots)
        sc_blocks_.resize(8);
        for (int i = 0; i < 8; ++i) if (!re(sc_blocks_[i], static_cast<size_t>(B) * D * 4)) return false;
        sc_bank_.resize(4);
        for (int i = 0; i < 4; ++i) if (!re(sc_bank_[i], static_cast<size_t>(B) * D * 4)) return false;
        for (int k = 0; k < 4; ++k) if (!re(sc_taps_[k], static_cast<size_t>(B) * D * 4)) return false;
        scratch_B_ = B;
        sc_moe_gu_cap_ = moe_gu_cap;
        return true;
    }

    bool ensure_dspark_scratch(int B, int D, int q_dim, int kv_dim, int inter, std::string& err) {
        if (ds_cur_.dev_ptr) return true;
        auto al = [&](GpuBuffer& b, size_t bytes) -> bool {
            return hip_ok(alloc_gpu_buffer(b, bytes, mem_info_.is_uma), "dspark scratch", err);
        };
        return al(ds_cur_, B * D * 4) && al(ds_res_, B * D * 4) && al(ds_h_, B * D * 4) &&
               al(ds_q_, B * q_dim * 4) && al(ds_k_, B * kv_dim * 4) && al(ds_v_, B * kv_dim * 4) &&
               al(ds_ao_, B * q_dim * 4) && al(ds_mg_, B * inter * 4) && al(ds_mu_, B * inter * 4) &&
               al(ds_qkv_, static_cast<size_t>(B) * (q_dim + 2 * kv_dim) * 4) &&
               al(ds_gu_, static_cast<size_t>(B) * 2 * inter * 4);
    }

    bool ensure_dspark_slot_cap(GpuSlotMeta& sl, int need_len, std::string& err) {
        if (need_len <= sl.dspark_cap) return true;
        const int max_cap = std::max(need_len, max_ctx_ + 16);
        const int new_cap = std::min(max_cap, std::max(need_len, std::max(256, sl.dspark_cap * 2)));
        const int kv_dim = dspark_cfg_.n_kv * dspark_cfg_.head_dim;
        const int D = dspark_cfg_.hidden;
        auto alloc_ds_buf = [&](GpuBuffer& dst, size_t bytes, const char* tag) -> bool {
            if (alloc_gpu_buffer(dst, bytes, mem_info_.is_uma) == hipSuccess) return true;
            if (!mem_info_.is_uma) {
                (void)hipGetLastError();
                if (alloc_gpu_buffer(dst, bytes, true, true) == hipSuccess) return true;
            }
            return hip_ok(hipErrorOutOfMemory, tag, err);
        };
        for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
            GpuBuffer nk, nv;
            if (!alloc_ds_buf(nk, static_cast<size_t>(new_cap) * kv_dim * 2, "ds k grow") ||
                !alloc_ds_buf(nv, static_cast<size_t>(new_cap) * kv_dim * 2, "ds v grow")) {
                free_gpu_buffer(nk); free_gpu_buffer(nv);
                return false;
            }
            if (sl.dspark_cap > 0) {
                const size_t old_b = static_cast<size_t>(sl.dspark_cap) * kv_dim * 2;
                (void)hipMemcpyAsync(nk.dev_ptr, sl.dspark_k[il].dev_ptr, old_b, hipMemcpyDefault, stream_);
                (void)hipMemcpyAsync(nv.dev_ptr, sl.dspark_v[il].dev_ptr, old_b, hipMemcpyDefault, stream_);
            }
            (void)hipStreamSynchronize(stream_);
            free_gpu_buffer(sl.dspark_k[il]); free_gpu_buffer(sl.dspark_v[il]);
            sl.dspark_k[il] = nk; sl.dspark_v[il] = nv;
        }
        GpuBuffer nf;
        if (!alloc_ds_buf(nf, static_cast<size_t>(new_cap) * D * 2, "ds frnn grow")) return false;
        if (sl.dspark_cap > 0) {
            (void)hipMemcpyAsync(nf.dev_ptr, sl.dspark_frnn.dev_ptr, static_cast<size_t>(sl.dspark_cap) * D * 2, hipMemcpyDefault, stream_);
            (void)hipStreamSynchronize(stream_);
        }
        free_gpu_buffer(sl.dspark_frnn);
        sl.dspark_frnn = nf;
        sl.dspark_cap = new_cap;
        return true;
    }

    bool load_vision_gpu(const std::string& path, std::string& err) {
        vision_.release();
        if (!vision_probe(path, vision_.info, err)) return false;
        strata::artifact::StrataFileReader rd;
        if (!rd.open(path, err)) return false;
        std::vector<uint8_t> s1;
        if (!rd.read_section(1, s1, err)) return false;

        for (const auto& kv : vision_.info.T) {
            const auto& vt = kv.second;
            size_t n = 1;
            for (int64_t d : vt.shape) n *= static_cast<size_t>(d);
            std::vector<float> f32(n);
            const uint8_t* src = s1.data() + vt.off;
            if (vt.fmt == F_BF16) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                for (size_t i = 0; i < n; ++i) f32[i] = ink2::bf16_to_f32(p[i]);
            } else if (vt.fmt == F_F16) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                for (size_t i = 0; i < n; ++i) f32[i] = ink2::fp16_to_f32(p[i]);
            } else {
                std::memcpy(f32.data(), src, n * 4);
            }

            if (kv.first == "model.visual.backbone.embeddings.class_embedding") vision_.h_cls = f32;
            else if (kv.first == "model.visual.backbone.embeddings.cls_position_embedding") vision_.h_cls_pos = f32;
            else if (kv.first == "model.visual.backbone.embeddings.position_embedding") vision_.h_pos_emb = f32;
            else if (kv.first == "model.visual.backbone.embeddings.patch_embedding.weight" ||
                     (vt.shape.size() >= 2 && vt.shape[0] >= 64 && (n / static_cast<size_t>(vt.shape[0])) >= 64)) {
                auto bf = f32_vec_to_bf16(f32);
                GpuBuffer b;
                if (!upload_raw(b, bf.data(), bf.size() * 2, err)) return false;
                vision_.w_bf16[kv.first] = b;
            } else {
                GpuBuffer b;
                if (!upload_raw(b, f32.data(), f32.size() * 4, err)) return false;
                vision_.w_f32[kv.first] = b;
            }
        }
        vision_.loaded = true;
        return true;
    }

    bool forward_batch_chunk(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) {
        const int B = batch.n_tokens;
        const int D = c_.hidden;
        const int q_dim = c_.n_head * c_.head_dim;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const bool is_fp8 = (default_kv_dtype_ == KvDtype::FP8_E4M3);

        for (int b = 0; b < B; ++b) {
            const int s = batch.slot_ids[b];
            if (s < 0 || s >= max_slots_ || !slots_[s].allocated) {
                err = "hip: unallocated slot " + std::to_string(s);
                return false;
            }
        }
        if (!ensure_batch_scratch(B, err)) return false;

        // Detect whether all B tokens belong to distinct slots (fast parallel ShortConv) or grouped sequential per slot
        bool all_distinct_slots = true;
        std::vector<int> slot_seen(max_slots_, 0);
        for (int b = 0; b < B; ++b) {
            if (++slot_seen[batch.slot_ids[b]] > 1) all_distinct_slots = false;
        }
        int n_groups = 0;
        std::vector<int32_t> h_grp_slot, h_grp_off, h_grp_tok;
        if (!all_distinct_slots) {
            std::vector<std::vector<int32_t>> by_slot(max_slots_);
            for (int b = 0; b < B; ++b) by_slot[batch.slot_ids[b]].push_back(b);
            h_grp_off.push_back(0);
            for (int s = 0; s < max_slots_; ++s) {
                if (by_slot[s].empty()) continue;
                h_grp_slot.push_back(s);
                for (int32_t idx : by_slot[s]) h_grp_tok.push_back(idx);
                h_grp_off.push_back(static_cast<int32_t>(h_grp_tok.size()));
            }
            n_groups = static_cast<int>(h_grp_slot.size());
            (void)hipMemcpyAsync(sc_grp_slot_.dev_ptr, h_grp_slot.data(), h_grp_slot.size() * 4, hipMemcpyHostToDevice, stream_);
            (void)hipMemcpyAsync(sc_grp_off_.dev_ptr, h_grp_off.data(), h_grp_off.size() * 4, hipMemcpyHostToDevice, stream_);
            (void)hipMemcpyAsync(sc_grp_tok_.dev_ptr, h_grp_tok.data(), h_grp_tok.size() * 4, hipMemcpyHostToDevice, stream_);
        }

        // Pre-ensure KV pages exist for all tokens in batch across all 40 SWA & 8 Global pools
        for (int b = 0; b < B; ++b) {
            const int s = batch.slot_ids[b];
            const int p = batch.pos[b];
            const int swa_lp = (p % c_.swa_win) >> 6;
            const int glb_lp = p >> 6;
            for (int i = 0; i < c_.n_swa_visits; ++i) {
                if (!swa_pools_[i].ensure_page(s, swa_lp, stream_, err)) return false;
            }
            for (int i = 0; i < c_.n_global_visits; ++i) {
                if (!glob_pools_[i].ensure_page(s, glb_lp, stream_, err)) return false;
            }
        }
        for (auto& p : swa_pools_) if (!p.sync_pt(stream_, err)) return false;
        for (auto& p : glob_pools_) if (!p.sync_pt(stream_, err)) return false;

        // Upload tiny batch descriptors
        (void)hipMemcpyAsync(sc_slot_ids_.dev_ptr, batch.slot_ids.data(), B * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_pos_.dev_ptr, batch.pos.data(), B * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_tok_ids_.dev_ptr, batch.token_ids.data(), B * 4, hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(sc_img_idx_.dev_ptr, batch.img_embed_idx.data(), B * 4, hipMemcpyHostToDevice, stream_);

        // Upload image embeddings if present in this batch
        int max_img_row = -1;
        for (int b = 0; b < B; ++b) max_img_row = std::max(max_img_row, batch.img_embed_idx[b]);
        GpuBuffer d_img_emb;
        if (max_img_row >= 0 && batch.img_embeds != nullptr) {
            if (!upload_raw(d_img_emb, batch.img_embeds, static_cast<size_t>(max_img_row + 1) * D * 4, err)) return false;
        }

        // Detect single-slot speculative verify batch (recording <= 8 ShortConv inputs for exact partial rollback!)
        int rec_slot = -1;
        if (B >= 2 && B <= 8) {
            const int s0 = batch.slot_ids[0];
            bool same_slot = slots_[s0].has_snap && batch.pos[0] == slots_[s0].snap_seq_len;
            for (int b = 1; b < B && same_slot; ++b) {
                if (batch.slot_ids[b] != s0 || batch.pos[b] != batch.pos[0] + b) same_slot = false;
            }
            if (same_slot) {
                rec_slot = s0;
                slots_[s0].sconv_rec_n = B;
            }
        }

        bool is_contiguous_single_slot = (!all_distinct_slots && n_groups == 1 && B >= 16);
        for (int b = 1; b < B && is_contiguous_single_slot; ++b) {
            if (batch.slot_ids[b] != batch.slot_ids[0] || batch.pos[b] != batch.pos[0] + b) {
                is_contiguous_single_slot = false;
            }
        }

        auto run_sconv = [&](const float* in_ptr, const std::string& sw_key, const float* dyn_ptr,
                             GpuBuffer& hist_pool_all, float* out_ptr, int C, int n_heads,
                             GpuBuffer* rec_pool = nullptr, int rec_visit = 0) {
            if (rec_slot >= 0 && rec_pool && rec_pool->dev_ptr) {
                const size_t one_visit = 8u * static_cast<size_t>(C) * sizeof(uint16_t);
                k_f32_to_bf16_vec<<<(B * C + 255) / 256, 256, 0, stream_>>>(
                    in_ptr, reinterpret_cast<uint16_t*>(rec_pool->dev_ptr + rec_visit * one_visit), B * C);
            }
            const float* sw = reinterpret_cast<const float*>(static_sconv_.at(sw_key).dev_ptr);
            const size_t visit_stride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(C) * sizeof(uint16_t);
            uint16_t* hp = reinterpret_cast<uint16_t*>(hist_pool_all.dev_ptr + rec_visit * visit_stride);
            if (all_distinct_slots) {
                k2_shortconv4_batched<<<dim3((C + 255) / 256, B), 256, 0, stream_>>>(
                    in_ptr, sw, dyn_ptr, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), hp, out_ptr, B, C, n_heads);
            } else if (is_contiguous_single_slot) {
                uint16_t* h_slot = hp + static_cast<size_t>(batch.slot_ids[0]) * 3 * C;
                k2_shortconv4_prefill_par<<<dim3((C + 255) / 256, (B + 15) / 16), 256, 0, stream_>>>(
                    in_ptr, sw, dyn_ptr, h_slot, out_ptr, B, C, n_heads);
                k2_shortconv4_update_hist<<<(C + 255) / 256, 256, 0, stream_>>>(in_ptr, h_slot, B, C);
            } else {
                k2_shortconv4_grouped_seq<<<dim3((C + 255) / 256, n_groups), 256, 0, stream_>>>(
                    in_ptr, sw, dyn_ptr, reinterpret_cast<const int32_t*>(sc_grp_slot_.dev_ptr),
                    reinterpret_cast<const int32_t*>(sc_grp_off_.dev_ptr), reinterpret_cast<const int32_t*>(sc_grp_tok_.dev_ptr),
                    hp, out_ptr, n_groups, C, n_heads);
            }
        };

        auto run_attnres = [&](const std::vector<const float*>& ptrs, const float* cur_p, const float* q_p,
                               int nq, float* out_p) {
            BlockPtrs12 bp{};
            const int nb = static_cast<int>(ptrs.size());
            for (int i = 0; i < nb && i < 12; ++i) bp.p[i] = ptrs[i];
            if (nq == 3) {
                k1_attn_res_mix_multi<3><<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, D);
            } else if (nq == 4) {
                k1_attn_res_mix_multi<4><<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, D);
            } else {
                k1_attn_res_mix<<<dim3(nq, B), 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, nq, D);
            }
        };

        auto run_attnres_norm = [&](const std::vector<const float*>& ptrs, const float* cur_p, const float* q_p,
                                    const float* nw_p, float* out_p) {
            BlockPtrs12 bp{};
            const int nb = static_cast<int>(ptrs.size());
            for (int i = 0; i < nb && i < 12; ++i) bp.p[i] = ptrs[i];
            if ((D & 3) == 0 && D <= 4096) {
                k1_attn_res_mix_norm_fused<<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, nw_p, out_p, B, D, c_.rms_eps);
            } else {
                k1_attn_res_mix<<<dim3(1, B), 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, 1, D);
                k_rms_norm_rows<<<B, 256, 0, stream_>>>(out_p, nw_p, out_p, B, D, c_.rms_eps);
            }
        };

        // Pre-prepare DSpark context buffers/ptrs BEFORE HIP Graph capture so update_dspark_ctx_gpu runs 100% INSIDE the graph!
        const bool do_dspark_ctx = has_dspark_ && batch.capture_taps && h_dspark_fuse_.size() >= 16;
        if (do_dspark_ctx) {
            if (!prepare_dspark_ctx_ptrs(batch, err)) return false;
        }

        // Check eligibility for HIP Graph Capture (all distinct slots, all GPU-resident layers, no images/host-taps, want all logits)
        bool all_gpu_layers = true;
        for (bool ok_l : layer_all_gpu_) if (!ok_l) { all_gpu_layers = false; break; }
        bool all_want_logits = (logits_out != nullptr);
        for (int b = 0; b < B && all_want_logits; ++b) if (!batch.want_logits[b]) all_want_logits = false;
        const bool need_taps = (taps_out != nullptr) || (has_dspark_ && batch.capture_taps);

        uint64_t kv_gen_sum = dspark_ptr_gen_;
        for (const auto& p : swa_pools_) kv_gen_sum += p.pool_gen;
        for (const auto& p : glob_pools_) kv_gen_sum += p.pool_gen;

        const int prof_mode = (std::getenv("STRATA_HIP_PROF") ? std::atoi(std::getenv("STRATA_HIP_PROF")) : 0);
        const bool use_graph = (prof_mode < 2) && all_gpu_layers && all_want_logits && max_img_row < 0 && taps_out == nullptr &&
                               ((all_distinct_slots && rec_slot < 0 && B <= 64) ||
                                (!all_distinct_slots && rec_slot >= 0 && n_groups == 1 && B >= 2 && B <= 8));
        const bool is_verify_graph = use_graph && !all_distinct_slots;

        if (use_graph) {
            const int V = c_.vocab_unpadded;
            const size_t lbytes = static_cast<size_t>(B) * V * sizeof(float);
            if (sc_logits_.bytes < lbytes) {
                invalidate_graph();
                free_gpu_buffer(sc_logits_);
                if (!hip_ok(alloc_gpu_buffer(sc_logits_, lbytes, mem_info_.is_uma), "logits buf", err)) return false;
            }
            std::vector<int32_t> want_seq(B);
            for (int b = 0; b < B; ++b) want_seq[b] = b;
            (void)hipMemcpyAsync(sc_want_idx_.dev_ptr, want_seq.data(), B * 4, hipMemcpyHostToDevice, stream_);

            hipGraphExec_t& g_exec = is_verify_graph ? graph_verify_exec_[B] : graph_exec_;
            const bool g_hit = is_verify_graph
                ? (g_exec != nullptr && vgraph_slot_[B] == rec_slot &&
                   vgraph_need_taps_[B] == need_taps && vgraph_kv_gen_[B] == kv_gen_sum && vgraph_scratch_B_[B] == scratch_B_)
                : (g_exec != nullptr && graph_B_ == B && graph_need_taps_ == need_taps &&
                   graph_kv_gen_ == kv_gen_sum && graph_scratch_B_ == scratch_B_);

            if (g_hit) {
                if (!hip_ok(hipGraphLaunch(g_exec, stream_), "hipGraphLaunch", err)) return false;
                for (int b = 0; b < B; ++b) {
                    auto& sl = slots_[batch.slot_ids[b]];
                    sl.seq_len = std::max(sl.seq_len, batch.pos[b] + 1);
                    if (do_dspark_ctx) sl.dspark_len = std::max(sl.dspark_len, batch.pos[b] + 1);
                }
                if (!hip_ok(hipMemcpyAsync(logits_out, sc_logits_.dev_ptr, lbytes, hipMemcpyDeviceToHost, stream_),
                            "logits D2H", err)) return false;
                return hip_ok(hipStreamSynchronize(stream_), "graph sync", err);
            }
            if (is_verify_graph) {
                if (graph_verify_exec_[B]) { (void)hipGraphExecDestroy(graph_verify_exec_[B]); graph_verify_exec_[B] = nullptr; }
            } else {
                if (graph_exec_) { (void)hipGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
            }
            if (!hip_ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "hipStreamBeginCapture", err)) {
                return false;
            }
        }

        // 1. Embeddings
        float* cur = reinterpret_cast<float*>(sc_cur_.dev_ptr);
        const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
        k_embed_lookup<<<B, 256, 0, stream_>>>(
            emb_pk.buf.dev_ptr, emb_pk.info.fmt, static_cast<int>(emb_pk.info.rows()),
            f32_ptr("model.llm.embed_norm.weight"),
            reinterpret_cast<const int32_t*>(sc_tok_ids_.dev_ptr),
            reinterpret_cast<const int32_t*>(sc_img_idx_.dev_ptr),
            reinterpret_cast<const float*>(d_img_emb.dev_ptr),
            cur, B, D, c_.rms_eps);

        auto maybe_tap = [&](int visit_id, const float* vec) {
            if (!need_taps) return;
            for (int k = 0; k < 4 && k < static_cast<int>(dspark_cfg_.target_visits.size()); ++k) {
                if (dspark_cfg_.target_visits[k] == visit_id) {
                    (void)hipMemcpyAsync(sc_taps_[k].dev_ptr, vec, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                }
            }
        };

        // Depth block tracking (pointers into sc_blocks_[] / sc_bank_[])
        int n_blocks = 0;
        int n_prelude_blocks = 0;
        int n_bank = 0;
        std::vector<const float*> blk_ptrs;
        std::vector<const float*> bank_ptrs;

        float* in0   = reinterpret_cast<float*>(sc_in0_.dev_ptr);
        float* in1   = reinterpret_cast<float*>(sc_in1_.dev_ptr);
        float* in2   = reinterpret_cast<float*>(sc_in2_.dev_ptr);
        float* mx3   = reinterpret_cast<float*>(sc_mx3_.dev_ptr);
        float* msgs  = reinterpret_cast<float*>(sc_msgs_.dev_ptr);
        float* q     = reinterpret_cast<float*>(sc_q_.dev_ptr);
        float* k_raw = reinterpret_cast<float*>(sc_k_raw_.dev_ptr);
        float* v_raw = reinterpret_cast<float*>(sc_v_raw_.dev_ptr);
        float* k_cv  = reinterpret_cast<float*>(sc_k_conv_.dev_ptr);
        float* v_cv  = reinterpret_cast<float*>(sc_v_conv_.dev_ptr);
        float* r_raw = reinterpret_cast<float*>(sc_r_raw_.dev_ptr);
        float* rbias = reinterpret_cast<float*>(sc_rel_bias_.dev_ptr);
        float* sinks = reinterpret_cast<float*>(sc_sinks_.dev_ptr);
        float* gatea = reinterpret_cast<float*>(sc_gate_a_.dev_ptr);
        float* k_dyn = reinterpret_cast<float*>(sc_k_dyn_.dev_ptr);
        float* v_dyn = reinterpret_cast<float*>(sc_v_dyn_.dev_ptr);
        float* aout  = reinterpret_cast<float*>(sc_attn_out_.dev_ptr);
        float* oproj = reinterpret_cast<float*>(sc_o_proj_.dev_ptr);
        float* asc   = reinterpret_cast<float*>(sc_a_sconv_.dev_ptr);
        float* mlpin = reinterpret_cast<float*>(sc_mlp_in_.dev_ptr);
        half_t* mlpf16 = reinterpret_cast<half_t*>(sc_mlp_f16_.dev_ptr);
        float* rlog  = reinterpret_cast<float*>(sc_r_logits_.dev_ptr);
        int32_t* tk_i = reinterpret_cast<int32_t*>(sc_topk_ids_.dev_ptr);
        float*   tk_w = reinterpret_cast<float*>(sc_topk_w_.dev_ptr);
        float* moeout = reinterpret_cast<float*>(sc_moe_out_.dev_ptr);
        float* msc   = reinterpret_cast<float*>(sc_m_sconv_.dev_ptr);

        const bool prof_on = (!use_graph && ((prof_mode == 1 && B >= 64) || prof_mode >= 2));
        hipEvent_t pev[11]{};
        double p_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (prof_on) {
            for (int i = 0; i < 11; ++i) (void)hipEventCreate(&pev[i]);
        }
        auto pmark = [&](int idx) {
            if (prof_on) (void)hipEventRecord(pev[idx], stream_);
        };
        auto pflush = [&]() {
            if (!prof_on) return;
            (void)hipEventSynchronize(pev[8]);
            for (int i = 0; i < 8; ++i) {
                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, pev[i], pev[i + 1]);
                p_acc[i] += ms;
            }
        };

        int visit = 0;
        auto run_visit = [&](int ip, bool has_msgs) -> bool {
            pmark(0);
            maybe_tap(visit, cur);
            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = c_.is_swa[ip] != 0;
            const int kv_slot = c_.visit_kv_slot[visit];

            // k1: AttnRes + optional loop controllers + input RMSNorm (zero unnecessary D2D copies + fused AttnRes+Norm!)
            float* q_in = in0;
            float* k_in = in1;
            float* v_in = in2;
            bool attn_norm_done = false;
            const bool batch3_ctrl = (has_msgs && B <= 12 && (D & 3) == 0);
            if (ip > 0) {
                const float* q_res = f32_ptr("model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
                if (is_swa) {
                    if (!has_msgs) {
                        run_attnres_norm(blk_ptrs, cur, q_res, f32_ptr(p + ".attn_norm.weight"), in0);
                        k_in = in0;
                        v_in = in0;
                        attn_norm_done = true;
                    } else {
                        run_attnres(blk_ptrs, cur, q_res, 1, in0);
                        if (!batch3_ctrl) {
                            (void)hipMemcpyAsync(in1, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                            (void)hipMemcpyAsync(in2, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                        }
                    }
                } else {
                    run_attnres(blk_ptrs, cur, q_res, 3, mx3);
                    k_split_attnres3<<<(B * D + 255) / 256, 256, 0, stream_>>>(mx3, in0, in1, in2, B, D);
                }
            } else {
                q_in = in0;
                if (!is_swa || has_msgs) {
                    (void)hipMemcpyAsync(in0, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    (void)hipMemcpyAsync(in1, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    (void)hipMemcpyAsync(in2, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                }
            }

            float* ctrl_inv_c = reinterpret_cast<float*>(sc_ctrl_tmp_.dev_ptr);
            float* ctrl_inv_m = ctrl_inv_c + B;
            float* ctrl_gate  = ctrl_inv_m + B;

            auto run_loop_ctrl = [&](float* target_in, int msg_ch, const std::string& cp) {
                k1_loop_ctrl_rms<<<B, 256, 0, stream_>>>(target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, B, D);
                const bool is_bf = trunk_bf16_.count(cp + ".down_weight") > 0;
                if (B >= 32) {
                    float* cin = reinterpret_cast<float*>(sc_ctrl_in_.dev_ptr);
                    float* cg  = reinterpret_cast<float*>(sc_ctrl_gain_.dev_ptr);
                    float* cf  = reinterpret_cast<float*>(sc_ctrl_forget_.dev_ptr);
                    if (is_bf && c_.gate_rank == 64 && mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                        if (blaslt_.ready() && sc_x_bf16_.bytes >= static_cast<size_t>(B) * 2 * D * sizeof(uint16_t)) {
                            uint16_t* cin_bf = reinterpret_cast<uint16_t*>(sc_x_bf16_.dev_ptr);
                            k1_loop_ctrl_pack_in_bf16<<<dim3(((D >> 2) + 255) / 256, B), 256, 0, stream_>>>(
                                target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, cin_bf, B, D);
                            if (!blaslt_.gemm_bf16_f32(bf16_ptr(cp + ".down_weight"), cin_bf, ctrl_gate, B, 64, 2 * D, stream_, 1.0f)) {
                                k1_loop_ctrl_down_wmma_fused<<<dim3(1, (B + 63) / 64), 128, 0, stream_>>>(
                                    target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m,
                                    bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D);
                            }
                        } else {
                            // Fused WMMA down (ZERO 37.7 MB cin DRAM!) + fused WMMA up (ZERO 37.7 MB cg/cf DRAM!)
                            k1_loop_ctrl_down_wmma_fused<<<dim3(1, (B + 63) / 64), 128, 0, stream_>>>(
                                target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m,
                                bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D);
                        }
                        k1_loop_ctrl_up_wmma_fused<<<dim3((D + 63) / 64, (B + 63) / 64), 128, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_gate,
                            bf16_ptr(cp + ".gain_weight"), bf16_ptr(cp + ".forget_weight"), B, D);
                    } else if (is_bf) {
                        k1_loop_ctrl_pack_in<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, cin, B, D);
                        launch_bf16_opt(bf16_ptr(cp + ".down_weight"),   cin,       nullptr, ctrl_gate, B, c_.gate_rank, 2 * D);
                        launch_bf16_opt(bf16_ptr(cp + ".gain_weight"),   ctrl_gate, nullptr, cg,        B, D, c_.gate_rank);
                        launch_bf16_opt(bf16_ptr(cp + ".forget_weight"), ctrl_gate, nullptr, cf,        B, D, c_.gate_rank);
                        k1_loop_ctrl_apply_out<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, cg, cf, target_in, B, D);
                    } else {
                        k1_loop_ctrl_pack_in<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, cin, B, D);
                        launch_f32_linear(f32_ptr(cp + ".down_weight"),   cin,       nullptr, ctrl_gate, B, c_.gate_rank, 2 * D, stream_, mem_info_.gen);
                        launch_f32_linear(f32_ptr(cp + ".gain_weight"),   ctrl_gate, nullptr, cg,        B, D, c_.gate_rank,    stream_, mem_info_.gen);
                        launch_f32_linear(f32_ptr(cp + ".forget_weight"), ctrl_gate, nullptr, cf,        B, D, c_.gate_rank,    stream_, mem_info_.gen);
                        k1_loop_ctrl_apply_out<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, cg, cf, target_in, B, D);
                    }
                } else if (is_bf) {
                    if (B <= 4 && (D & 3) == 0) {
                        k1_loop_ctrl_down_hiocc<true><<<dim3(c_.gate_rank, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    } else {
                        k1_loop_ctrl_down<true><<<dim3((c_.gate_rank + 7) / 8, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    }
                    k1_loop_ctrl_up<true><<<dim3((D + 63) / 64, B), 64, 0, stream_>>>(
                        target_in, msgs, 4, msg_ch, ctrl_gate, bf16_ptr(cp + ".gain_weight"), bf16_ptr(cp + ".forget_weight"), target_in, B, D, c_.gate_rank);
                } else {
                    if (B <= 4 && (D & 3) == 0) {
                        k1_loop_ctrl_down_hiocc<false><<<dim3(c_.gate_rank, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, f32_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    } else {
                        k1_loop_ctrl_down<false><<<dim3((c_.gate_rank + 7) / 8, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, f32_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    }
                    k1_loop_ctrl_up<false><<<dim3((D + 63) / 64, B), 64, 0, stream_>>>(
                        target_in, msgs, 4, msg_ch, ctrl_gate, f32_ptr(cp + ".gain_weight"), f32_ptr(cp + ".forget_weight"), target_in, B, D, c_.gate_rank);
                }
            };

            if (has_msgs) {
                const char* cnames[3] = {"attn", "k", "v"};
                const std::string c0 = p + ".controllers." + cnames[0];
                const std::string c1 = p + ".controllers." + cnames[1];
                const std::string c2 = p + ".controllers." + cnames[2];
                const bool all_bf = (trunk_bf16_.count(c0 + ".down_weight") > 0 &&
                                     trunk_bf16_.count(c1 + ".down_weight") > 0 &&
                                     trunk_bf16_.count(c2 + ".down_weight") > 0);
                const bool all_f32 = (trunk_bf16_.count(c0 + ".down_weight") == 0 &&
                                      trunk_bf16_.count(c1 + ".down_weight") == 0 &&
                                      trunk_bf16_.count(c2 + ".down_weight") == 0);
                if (batch3_ctrl && (all_bf || all_f32)) {
                    FPtr3 src_p = is_swa ? FPtr3{{in0, in0, in0}} : FPtr3{{in0, in1, in2}};
                    MutFPtr3 dst_p{{in0, in1, in2}};
                    float* ic3 = reinterpret_cast<float*>(sc_ctrl_tmp_.dev_ptr);
                    float* im3 = ic3 + 3 * B;
                    float* gt3 = im3 + 3 * B;
                    k1_loop_ctrl_rms3<<<dim3(3, B), 256, 0, stream_>>>(src_p, msgs, ic3, im3, B, D);
                    if (all_bf) {
                        Ptr3 wd{{bf16_ptr(c0 + ".down_weight"), bf16_ptr(c1 + ".down_weight"), bf16_ptr(c2 + ".down_weight")}};
                        Ptr3 wg{{bf16_ptr(c0 + ".gain_weight"), bf16_ptr(c1 + ".gain_weight"), bf16_ptr(c2 + ".gain_weight")}};
                        Ptr3 wf{{bf16_ptr(c0 + ".forget_weight"), bf16_ptr(c1 + ".forget_weight"), bf16_ptr(c2 + ".forget_weight")}};
                        k1_loop_ctrl_down_hiocc3<true><<<dim3(c_.gate_rank, B, 3), 256, 0, stream_>>>(
                            src_p, msgs, ic3, im3, wd, gt3, B, D, c_.gate_rank);
                        k1_loop_ctrl_up3<true><<<dim3((D + 63) / 64, B, 3), 64, 0, stream_>>>(
                            src_p, msgs, gt3, wg, wf, dst_p, B, D, c_.gate_rank);
                    } else {
                        Ptr3 wd{{f32_ptr(c0 + ".down_weight"), f32_ptr(c1 + ".down_weight"), f32_ptr(c2 + ".down_weight")}};
                        Ptr3 wg{{f32_ptr(c0 + ".gain_weight"), f32_ptr(c1 + ".gain_weight"), f32_ptr(c2 + ".gain_weight")}};
                        Ptr3 wf{{f32_ptr(c0 + ".forget_weight"), f32_ptr(c1 + ".forget_weight"), f32_ptr(c2 + ".forget_weight")}};
                        k1_loop_ctrl_down_hiocc3<false><<<dim3(c_.gate_rank, B, 3), 256, 0, stream_>>>(
                            src_p, msgs, ic3, im3, wd, gt3, B, D, c_.gate_rank);
                        k1_loop_ctrl_up3<false><<<dim3((D + 63) / 64, B, 3), 64, 0, stream_>>>(
                            src_p, msgs, gt3, wg, wf, dst_p, B, D, c_.gate_rank);
                    }
                } else {
                    if (batch3_ctrl && is_swa) {
                        (void)hipMemcpyAsync(in1, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                        (void)hipMemcpyAsync(in2, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    }
                    float* ins[3] = {in0, in1, in2};
                    for (int j = 0; j < 3; ++j) {
                        run_loop_ctrl(ins[j], j, p + ".controllers." + cnames[j]);
                    }
                }
            }

            if (!attn_norm_done) {
                if (has_msgs && (D & 3) == 0) {
                    MutFPtr3 io_p{{in0, in1, in2}};
                    FPtr3 nw_p = is_swa
                        ? FPtr3{{f32_ptr(p + ".attn_norm.weight"), f32_ptr(p + ".attn_norm.weight"), f32_ptr(p + ".attn_norm.weight")}}
                        : FPtr3{{f32_ptr(p + ".q_input_norm.weight"), f32_ptr(p + ".k_input_norm.weight"), f32_ptr(p + ".v_input_norm.weight")}};
                    k_rms_norm_rows3<<<dim3(3, B), 256, 0, stream_>>>(io_p, nw_p, B, D, c_.rms_eps);
                } else if (is_swa) {
                    const float* an = f32_ptr(p + ".attn_norm.weight");
                    const float* norm_src0 = (ip == 0 && !has_msgs) ? cur : in0;
                    k_rms_norm_rows<<<B, 256, 0, stream_>>>(norm_src0, an, in0, B, D, c_.rms_eps);
                    if (has_msgs) {
                        k_rms_norm_rows<<<B, 256, 0, stream_>>>(in1, an, in1, B, D, c_.rms_eps);
                        k_rms_norm_rows<<<B, 256, 0, stream_>>>(in2, an, in2, B, D, c_.rms_eps);
                    } else {
                        k_in = in0;
                        v_in = in0;
                    }
                } else {
                    if ((D & 3) == 0) {
                        MutFPtr3 io_p{{in0, in1, in2}};
                        FPtr3 nw_p{{f32_ptr(p + ".q_input_norm.weight"), f32_ptr(p + ".k_input_norm.weight"), f32_ptr(p + ".v_input_norm.weight")}};
                        k_rms_norm_rows3<<<dim3(3, B), 256, 0, stream_>>>(io_p, nw_p, B, D, c_.rms_eps);
                    } else {
                        k_rms_norm_rows<<<B, 256, 0, stream_>>>(in0, f32_ptr(p + ".q_input_norm.weight"), in0, B, D, c_.rms_eps);
                        k_rms_norm_rows<<<B, 256, 0, stream_>>>(in1, f32_ptr(p + ".k_input_norm.weight"), in1, B, D, c_.rms_eps);
                        k_rms_norm_rows<<<B, 256, 0, stream_>>>(in2, f32_ptr(p + ".v_input_norm.weight"), in2, B, D, c_.rms_eps);
                    }
                }
            }

            // Batched Q/K/V/R/Sink/Gate projections (fused direct-scatter GEMV + sink at B=1, concatenated GEMM+split for B>1!)
            pmark(1);
            const int r_dim = c_.n_head * c_.d_rel;
            bool sink_done = false;
            const bool sink_is_f32 = (trunk_bf16_.count(p + ".attn.attn_sink_proj.weight") == 0);
            if (trunk_qkvrg_[ip].dev_ptr != nullptr) {
                const uint16_t* w_cat = reinterpret_cast<const uint16_t*>(trunk_qkvrg_[ip].dev_ptr);
                float* cat_out = reinterpret_cast<float*>(sc_qkvrg_out_.dev_ptr);
                if (!has_msgs && is_swa) {
                    const int tot_n = 2 * q_dim + r_dim + 2 * kv_dim;
                    if (B == 1 && D <= 4096 && (D & 7) == 0) {
                        const int ns = sink_is_f32 ? c_.n_head : 0;
                        k_bf16_gemv1_qkvrg<<<(tot_n + ns + 7) / 8, 256, 0, stream_>>>(
                            w_cat, q_in, q, r_raw, gatea, k_raw, v_raw,
                            sink_is_f32 ? f32_ptr(p + ".attn.attn_sink_proj.weight") : nullptr,
                            sink_is_f32 ? sinks : nullptr,
                            q_dim, r_dim, kv_dim, tot_n, ns, D);
                        sink_done = sink_is_f32;
                    } else {
                        launch_bf16_opt(w_cat, q_in, nullptr, cat_out, B, tot_n, D);
                        k_split_qkvrg<<<dim3(((tot_n >> 2) + 255) / 256, B), 256, 0, stream_>>>(
                            cat_out, q, r_raw, gatea, k_raw, v_raw, B, q_dim, r_dim, kv_dim, tot_n);
                    }
                } else {
                    const int qrg_n = 2 * q_dim + r_dim;
                    if (B == 1 && D <= 4096 && (D & 7) == 0) {
                        const int ns = sink_is_f32 ? c_.n_head : 0;
                        k_bf16_gemv1_qkvrg<<<(qrg_n + ns + 7) / 8, 256, 0, stream_>>>(
                            w_cat, q_in, q, r_raw, gatea, nullptr, nullptr,
                            sink_is_f32 ? f32_ptr(p + ".attn.attn_sink_proj.weight") : nullptr,
                            sink_is_f32 ? sinks : nullptr,
                            q_dim, r_dim, kv_dim, qrg_n, ns, D);
                        sink_done = sink_is_f32;
                    } else {
                        launch_bf16_opt(w_cat, q_in, nullptr, cat_out, B, qrg_n, D);
                        k_split_qkvrg<<<dim3(((qrg_n >> 2) + 255) / 256, B), 256, 0, stream_>>>(
                            cat_out, q, r_raw, gatea, nullptr, nullptr, B, q_dim, r_dim, kv_dim, qrg_n);
                    }
                    launch_bf16_opt(w_cat + static_cast<size_t>(qrg_n) * D, k_in, nullptr, k_raw, B, kv_dim, D);
                    launch_bf16_opt(w_cat + static_cast<size_t>(qrg_n + kv_dim) * D, v_in, nullptr, v_raw, B, kv_dim, D);
                }
            } else {
                launch_trunk_linear(p + ".attn.wq_du.weight", q_in, q,     B, q_dim, D);
                launch_trunk_linear(p + ".attn.wk_dv.weight", k_in, k_raw, B, kv_dim, D);
                launch_trunk_linear(p + ".attn.wv_dv.weight", v_in, v_raw, B, kv_dim, D);
                launch_trunk_linear(p + ".attn.wr_du.weight", q_in, r_raw, B, r_dim, D);
                launch_trunk_linear(p + ".attn.gate_attn.weight", q_in, gatea, B, q_dim, D);
            }
            if (!sink_done) {
                launch_trunk_linear(p + ".attn.attn_sink_proj.weight", q_in, sinks, B, c_.n_head, D);
            }
            if (!is_swa) {
                launch_trunk_linear(p + ".attn.k_sconv.weight_proj", k_in, k_dyn, B, c_.n_kv * 4, D);
                launch_trunk_linear(p + ".attn.v_sconv.weight_proj", v_in, v_dyn, B, c_.n_kv * 4, D);
            }

            // k2 + k3 prep: ShortConv K & V + super-fused Q/R/K/V Norm+Clip + FP16 Paged KV Store + RelPos Bias Proj!
            pmark(2);
            GpuSlotMeta* rsl = (rec_slot >= 0) ? &slots_[rec_slot] : nullptr;
            if (all_distinct_slots && rec_slot < 0) {
                const size_t kv_vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(kv_dim) * sizeof(uint16_t);
                k2_shortconv4_kv_pair<<<dim3((kv_dim + 255) / 256, 2 * B), 256, 0, stream_>>>(
                    k_raw, reinterpret_cast<const float*>(static_sconv_.at(p + ".k_sconv").dev_ptr),
                    is_swa ? nullptr : k_dyn, reinterpret_cast<uint16_t*>(d_sconv_k_all_.dev_ptr + visit * kv_vstride), k_cv,
                    v_raw, reinterpret_cast<const float*>(static_sconv_.at(p + ".v_sconv").dev_ptr),
                    is_swa ? nullptr : v_dyn, reinterpret_cast<uint16_t*>(d_sconv_v_all_.dev_ptr + visit * kv_vstride), v_cv,
                    reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), B, kv_dim, c_.n_kv);
            } else {
                run_sconv(k_raw, p + ".k_sconv", is_swa ? nullptr : k_dyn, d_sconv_k_all_, k_cv, kv_dim, c_.n_kv, rsl ? &rsl->sconv_rec_k : nullptr, visit);
                run_sconv(v_raw, p + ".v_sconv", is_swa ? nullptr : v_dyn, d_sconv_v_all_, v_cv, kv_dim, c_.n_kv, rsl ? &rsl->sconv_rec_v : nullptr, visit);
            }

            const int32_t* pos_dev = reinterpret_cast<const int32_t*>(sc_pos_.dev_ptr);
            const std::string q_sc_k = p + ".attn.q_l2_clip.scale";
            const std::string r_sc_k = p + ".attn.rel_l2_clip.scale";
            const int q_sc_stride = (trunk_f32_.at(q_sc_k).bytes >= static_cast<size_t>(c_.n_head) * c_.head_dim * 4) ? c_.head_dim : 0;
            const int r_sc_stride = (trunk_f32_.at(r_sc_k).bytes >= static_cast<size_t>(c_.n_head) * c_.d_rel * 4) ? c_.d_rel : 0;
            VisitKvPool& kvp = is_swa ? swa_pools_[kv_slot] : glob_pools_[kv_slot];
            const int swa_w = is_swa ? c_.swa_win : 0;
            if (!is_fp8 && c_.head_dim <= 128 && c_.d_rel <= 16 && (c_.rel_extent & 3) == 0) {
                k2_qrkv_kvstore_relpos_fused<<<dim3(2 * c_.n_head + 2 * c_.n_kv, B), 128, 0, stream_>>>(
                    q, f32_ptr(q_sc_k), q_sc_stride,
                    r_raw, f32_ptr(r_sc_k), r_sc_stride,
                    f32_ptr(p + ".attn.rel_logits_proj.proj"), rbias,
                    k_cv, v_cv,
                    reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                    reinterpret_cast<half_t*>(kvp.k_pool.dev_ptr),
                    reinterpret_cast<half_t*>(kvp.v_pool.dev_ptr),
                    !is_swa, c_.qk_l2_clip, c_.rel_l2_clip, c_.rms_eps,
                    B, c_.n_head, c_.n_kv, c_.head_dim, c_.d_rel, swa_w, c_.rel_extent);
            } else {
                if (c_.head_dim <= 128 && c_.d_rel <= 32) {
                    k2_qrkv_norm_clip_fused<<<dim3(2 * c_.n_head + 2 * c_.n_kv, B), 128, 0, stream_>>>(
                        q, f32_ptr(q_sc_k), q_sc_stride, r_raw, f32_ptr(r_sc_k), r_sc_stride, k_cv, v_cv,
                        is_swa ? nullptr : pos_dev, c_.qk_l2_clip, c_.rel_l2_clip, c_.rms_eps,
                        B, c_.n_head, c_.n_kv, c_.head_dim, c_.d_rel);
                } else {
                    k2_head_l2_clip<<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                        q, f32_ptr(q_sc_k), q_sc_stride, is_swa ? nullptr : pos_dev, c_.qk_l2_clip, B, c_.n_head, c_.head_dim);
                    k2_head_l2_clip<<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                        r_raw, f32_ptr(r_sc_k), r_sc_stride, is_swa ? nullptr : pos_dev, c_.rel_l2_clip, B, c_.n_head, c_.d_rel);
                    k2_head_l2_clip<<<dim3(c_.n_kv, B), 128, 0, stream_>>>(
                        k_cv, nullptr, 0, nullptr, c_.qk_l2_clip, B, c_.n_kv, c_.head_dim);
                    k_rms_norm_rows<<<B * c_.n_kv, 256, 0, stream_>>>(v_cv, nullptr, v_cv, B * c_.n_kv, c_.head_dim, c_.rms_eps);
                }
                if (is_fp8) {
                    k3_kv_paged_store<true><<<B, 256, 0, stream_>>>(
                        k_cv, v_cv, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                        reinterpret_cast<float*>(kvp.k_scales.dev_ptr), reinterpret_cast<float*>(kvp.v_scales.dev_ptr),
                        B, c_.n_kv, c_.head_dim, swa_w);
                } else {
                    k3_kv_paged_store<false><<<B, 256, 0, stream_>>>(
                        k_cv, v_cv, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                        B, c_.n_kv, c_.head_dim, swa_w);
                }
                k_relpos_bias_proj<<<dim3(c_.n_head, B), 256, 0, stream_>>>(
                    r_raw, f32_ptr(p + ".attn.rel_logits_proj.proj"), pos_dev, rbias, B, c_.n_head, c_.d_rel, c_.rel_extent);
            }

            pmark(3);
            if (is_contiguous_single_slot && B >= 64 && mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                // Wave32 WMMA FlashAttention-2 Prefill (BQ=64, BK=64, 27 KB LDS = 2 WG/CU, uint2 V + hoisted rel_bias skip!)
                const dim3 fgrid(c_.n_head, (B + 63) / 64);
                if (is_fp8) {
                    k3_flash_swa_gqa_prefill_wmma<true><<<fgrid, 128, 0, stream_>>>(
                        q, gatea, rbias, sinks, batch.slot_ids[0], batch.pos[0],
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                        reinterpret_cast<const float*>(kvp.k_scales.dev_ptr), reinterpret_cast<const float*>(kvp.v_scales.dev_ptr),
                        aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                } else {
                    k3_flash_swa_gqa_prefill_wmma<false><<<fgrid, 128, 0, stream_>>>(
                        q, gatea, rbias, sinks, batch.slot_ids[0], batch.pos[0],
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                        aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                }
            } else if (B <= 4) {
                // 2-Stage FlashDecoding Split-KV (96 blocks x 8 waves = 768 waves at B=1 -> saturates ALL CUs!)
                float* sp_part = reinterpret_cast<float*>(sc_splitkv_part_.dev_ptr);
                if (is_fp8) {
                    k3_flash_splitkv_part<true, 4><<<dim3(c_.n_head, B, 4), 256, 0, stream_>>>(
                        q, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                        reinterpret_cast<const float*>(kvp.k_scales.dev_ptr), reinterpret_cast<const float*>(kvp.v_scales.dev_ptr),
                        sp_part, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                } else {
                    k3_flash_splitkv_part<false, 4><<<dim3(c_.n_head, B, 4), 256, 0, stream_>>>(
                        q, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                        sp_part, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                }
                k3_flash_splitkv_reduce<4><<<dim3(c_.n_head, B), 32, 0, stream_>>>(
                    sp_part, gatea, aout, B, c_.n_head);
            } else if (B <= 8) {
                if (is_fp8) {
                    k3_flash_swa_gqa_splitkv<true><<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                        q, gatea, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                        reinterpret_cast<const float*>(kvp.k_scales.dev_ptr), reinterpret_cast<const float*>(kvp.v_scales.dev_ptr),
                        aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                } else {
                    k3_flash_swa_gqa_splitkv<false><<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                        q, gatea, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                        reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                        kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                        aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
                }
            } else if (is_fp8) {
                k3_flash_swa_gqa<true><<<dim3((c_.n_head + 3) / 4, B), 128, 0, stream_>>>(
                    q, gatea, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                    kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                    reinterpret_cast<const float*>(kvp.k_scales.dev_ptr), reinterpret_cast<const float*>(kvp.v_scales.dev_ptr),
                    aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
            } else {
                k3_flash_swa_gqa<false><<<dim3((c_.n_head + 3) / 4, B), 128, 0, stream_>>>(
                    q, gatea, rbias, sinks, reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr), pos_dev,
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                    kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                    aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
            }

            // O-proj + post-attention ShortConv + AttnRes block boundary rule (1-launch super-fused at B=1 distinct-slot!)
            pmark(4);
            const std::string wo_key = p + ".attn.wo_ud.weight";
            if (B == 1 && all_distinct_slots && rec_slot < 0 && q_dim <= 4096 && (q_dim & 7) == 0 && trunk_bf16_.count(wo_key) > 0) {
                const size_t vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(D) * sizeof(uint16_t);
                float* bsave = nullptr;
                if (ip % c_.block_size == 0) {
                    bsave = reinterpret_cast<float*>(sc_blocks_[n_blocks].dev_ptr);
                    blk_ptrs.push_back(bsave);
                    ++n_blocks;
                }
                k_bf16_gemv1_wo_sconv<<<(D + 7) / 8, 256, 0, stream_>>>(
                    bf16_ptr(wo_key), aout,
                    reinterpret_cast<const float*>(static_sconv_.at(p + ".attn_sconv").dev_ptr),
                    reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr),
                    reinterpret_cast<uint16_t*>(d_sconv_attn_all_.dev_ptr + visit * vstride),
                    cur, bsave, D, q_dim);
            } else {
                launch_trunk_linear(wo_key, aout, oproj, B, D, q_dim);
                if (ip % c_.block_size == 0) {
                    run_sconv(oproj, p + ".attn_sconv", nullptr, d_sconv_attn_all_, asc, D, 0, rsl ? &rsl->sconv_rec_attn : nullptr, visit);
                    (void)hipMemcpyAsync(sc_blocks_[n_blocks].dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    blk_ptrs.push_back(reinterpret_cast<const float*>(sc_blocks_[n_blocks].dev_ptr));
                    ++n_blocks;
                    (void)hipMemcpyAsync(cur, asc, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                } else if (all_distinct_slots && rec_slot < 0) {
                    const size_t vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(D) * sizeof(uint16_t);
                    k2_sconv_add_inplace<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                        oproj, reinterpret_cast<const float*>(static_sconv_.at(p + ".attn_sconv").dev_ptr),
                        reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr),
                        reinterpret_cast<uint16_t*>(d_sconv_attn_all_.dev_ptr + visit * vstride),
                        cur, B, D);
                } else {
                    run_sconv(oproj, p + ".attn_sconv", nullptr, d_sconv_attn_all_, asc, D, 0, rsl ? &rsl->sconv_rec_attn : nullptr, visit);
                    k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, asc, cur, B * D);
                }
            }

            // MoE sublayer: fused AttnRes+mlp_norm when !has_msgs (deletes 24 launches/tok!)
            if (!has_msgs) {
                run_attnres_norm(blk_ptrs, cur, f32_ptr("model.llm.attn_res.mlp_queries." + std::to_string(ip)),
                                 f32_ptr(p + ".mlp_norm.weight"), mlpin);
            } else {
                run_attnres(blk_ptrs, cur, f32_ptr("model.llm.attn_res.mlp_queries." + std::to_string(ip)), 1, mlpin);
                run_loop_ctrl(mlpin, 3, p + ".controllers.mlp");
                k_rms_norm_rows<<<B, 256, 0, stream_>>>(mlpin, f32_ptr(p + ".mlp_norm.weight"), mlpin, B, D, c_.rms_eps);
            }

            // k4: Top-8/128 Sigmoid Router (fused with f32->f16 mlpin conversion for B<=4!)
            pmark(5);
            if (B >= 8 && trunk_bf16_.count(p + ".mlp.gate.weight") > 0 &&
                mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                launch_bf16_opt(bf16_ptr(p + ".mlp.gate.weight"), mlpin, nullptr, rlog, B, c_.n_experts, D);
            } else {
                launch_f32_linear(f32_ptr(p + ".mlp.gate.weight"), mlpin, nullptr, rlog, B, c_.n_experts, D, stream_, mem_info_.gen);
            }
            const bool fused_router_f16 = (layer_all_gpu_[ip] && B <= 4 && (D & 3) == 0);
            if (fused_router_f16) {
                k4_moe_router_topk_f16<<<B, 128, 0, stream_>>>(
                    rlog, f32_ptr(p + ".mlp.gate.bias"), mlpin, mlpf16, tk_i, tk_w, B, D, c_.top_k, c_.route_scale);
            } else {
                k4_moe_router_topk<<<B, 128, 0, stream_>>>(rlog, f32_ptr(p + ".mlp.gate.bias"), tk_i, tk_w, B, c_.top_k, c_.route_scale);
            }

            // MoE Expert Execution:
            // Fast Path A1 (B <= 4, all GPU-resident): 2-launch ZERO-SYNC parallel-expert direct MoE (2,304 blocks!)
            // Fast Path A2 (B >= blaslt_moe_min_b_, all GPU-resident, hipBLASLt ready): L3/MALL-resident unpack-once + hipBLASLt FP16 GEMM!
            // Fast Path A3 (B >= 5, all GPU-resident): 4-launch ZERO-SYNC Wave32 WMMA Grouped-MoE
            // Fast Path B1 (Hybrid partial-VRAM, B <= 16): Strata Predictive CPU/RAM Hybrid (GPU VRAM+ElasticCache || Spin-Wait CPU Pool + 3x-Loop Async DMA Prefetch!)
            // Fast Path B2 (Hybrid partial-VRAM, B > 16): Chunked Double-Buffered GPU Matrix-Core Expert Streaming!
            bool fused_moe_post = false;
            if (ep2_active_) {
                // Dual-GPU Expert Parallelism: launch GPU 1 peer experts FIRST, then compute GPU 0 (+ optional hybrid RAM) in parallel!
                if (!fused_router_f16) {
                    k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                }
                if (!launch_peer_moe_async(ip, B, mlpf16, tk_i, tk_w, err)) return false;

                if (layer_all_ep2_[ip]) {
                    const GpuExpertDesc* d_exps = reinterpret_cast<const GpuExpertDesc*>(d_layer_exps_[ip].dev_ptr);
                    if (B <= 4) {
                        k_moe_direct_w13_situ<<<dim3((c_.inter + 3) / 4, B * c_.top_k), 256, 0, stream_>>>(
                            d_exps, tk_i, mlpf16, reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr),
                            B, c_.top_k, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        float* y_part = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                        k_moe_direct_w2_par<<<dim3((D + 7) / 8, B * c_.top_k), 256, 0, stream_>>>(
                            d_exps, tk_i, reinterpret_cast<const half_t*>(sc_moe_act_.dev_ptr),
                            y_part, B, c_.top_k, D, c_.inter);
                        k_moe_direct_reduce<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            y_part, tk_w, moeout, B, c_.top_k, D);
                    } else {
                        int32_t* d_off = reinterpret_cast<int32_t*>(sc_exp_off_.dev_ptr);
                        int32_t* d_cur = reinterpret_cast<int32_t*>(sc_exp_cur_.dev_ptr);
                        int32_t* d_btk = reinterpret_cast<int32_t*>(sc_bucket_tok_.dev_ptr);
                        int32_t* d_s2b = reinterpret_cast<int32_t*>(sc_slotk_bidx_.dev_ptr);
                        half_t*  d_act = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                        float*   d_by  = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                        const int tot_pairs = B * c_.top_k;
                        if (B >= 16) {
                            (void)hipMemsetAsync(d_off, 0, 129 * sizeof(int32_t), stream_);
                            const int cblks = std::min(64, (tot_pairs + 255) / 256);
                            k_moe_count_buckets<<<cblks, 256, 0, stream_>>>(tk_i, d_off, tot_pairs);
                            k_moe_prefix_buckets<<<1, 128, 0, stream_>>>(d_off, d_off, d_cur);
                            k_moe_scatter_buckets<<<(tot_pairs + 255) / 256, 256, 0, stream_>>>(tk_i, d_cur, d_btk, d_s2b, tot_pairs, c_.top_k);
                        } else {
                            k_moe_build_buckets<<<1, 128, 0, stream_>>>(tk_i, d_off, d_btk, d_s2b, B, c_.top_k);
                        }
                        if (mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                            const int mt = (B >= 1024) ? 8 : ((B >= 256) ? 4 : ((B >= 64) ? 2 : 1));
                            const dim3 g13((2 * c_.inter + 127) / 128, c_.n_experts, mt);
                            const dim3 g2((D + 127) / 128, c_.n_experts, mt);
                            if (B >= 64) {
                                k_moe_grouped_w13_situ_wmma_128<<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                                k_moe_grouped_w2_wmma_128<<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            } else {
                                k_moe_grouped_w13_situ_wmma<<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                                k_moe_grouped_w2_wmma<<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            }
                        } else {
                            k_moe_grouped_w13_situ<<<dim3((c_.inter + 3) / 4, c_.n_experts), 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                            k_moe_grouped_w2<<<dim3((D + 7) / 8, c_.n_experts), 256, 0, stream_>>>(
                                d_exps, d_off, d_act, d_by, D, c_.inter);
                        }
                        k_moe_reduce_buckets_masked<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                            d_exps, tk_i, d_by, d_s2b, tk_w, moeout, B, c_.top_k, D);
                    }
                    if (!finish_peer_moe_add(B, moeout, err)) return false;
                }
            }
            if (ep2_active_ && layer_all_ep2_[ip]) {
                // Already completed on GPU 0 || GPU 1 above!
            } else if (layer_all_gpu_[ip] && B <= 4) {
                if (!fused_router_f16) {
                    k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                }
                const GpuExpertDesc* d_exps = reinterpret_cast<const GpuExpertDesc*>(d_layer_exps_[ip].dev_ptr);
                pmark(6);
                k_moe_direct_w13_situ<<<dim3((c_.inter + 3) / 4, B * c_.top_k), 256, 0, stream_>>>(
                    d_exps, tk_i, mlpf16, reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr),
                    B, c_.top_k, D, c_.inter, c_.situ_beta, c_.situ_lb);
                pmark(7);
                float* y_part = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                k_moe_direct_w2_par<<<dim3((D + 7) / 8, B * c_.top_k), 256, 0, stream_>>>(
                    d_exps, tk_i, reinterpret_cast<const half_t*>(sc_moe_act_.dev_ptr),
                    y_part, B, c_.top_k, D, c_.inter);
                if (all_distinct_slots && rec_slot < 0) {
                    const size_t vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(D) * sizeof(uint16_t);
                    k_moe_reduce_sconv_add<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                        y_part, tk_w, reinterpret_cast<const float*>(static_sconv_.at(p + ".mlp_sconv").dev_ptr),
                        reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr),
                        reinterpret_cast<uint16_t*>(d_sconv_mlp_all_.dev_ptr + visit * vstride),
                        cur, B, c_.top_k, D);
                    fused_moe_post = true;
                } else {
                    k_moe_direct_reduce<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                        y_part, tk_w, moeout, B, c_.top_k, D);
                }
            } else if (layer_all_gpu_[ip] && B >= blaslt_moe_min_b_ && blaslt_.ready() &&
                       d_exp_w13_f16_.dev_ptr != nullptr && d_exp_w2_f16_.dev_ptr != nullptr && !use_graph) {
                int32_t* d_off = reinterpret_cast<int32_t*>(sc_exp_off_.dev_ptr);
                int32_t* d_btk = reinterpret_cast<int32_t*>(sc_bucket_tok_.dev_ptr);
                int32_t* d_s2b = reinterpret_cast<int32_t*>(sc_slotk_bidx_.dev_ptr);
                half_t*  d_bx  = reinterpret_cast<half_t*>(sc_bucket_x_f16_.dev_ptr);
                float*   d_gu  = reinterpret_cast<float*>(sc_moe_gu_.dev_ptr);
                half_t*  d_act = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                float*   d_by  = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);

                k_moe_build_buckets<<<1, 128, 0, stream_>>>(tk_i, d_off, d_btk, d_s2b, B, c_.top_k);
                k_gather_rows_f16<<<(B * c_.top_k * D + 255) / 256, 256, 0, stream_>>>(
                    mlpin, d_btk, d_bx, B * c_.top_k, D);

                int32_t h_off[129];
                if (!hip_ok(hipMemcpyAsync(h_off, d_off, 129 * sizeof(int32_t), hipMemcpyDeviceToHost, stream_), "moe off D2H", err) ||
                    !hip_ok(hipStreamSynchronize(stream_), "moe off sync", err)) {
                    return false;
                }
                const auto& dir = adj_dir_;
                for (int eid = 0; eid < c_.n_experts; ++eid) {
                    const int beg = h_off[eid];
                    const int m_e = h_off[eid + 1] - beg;
                    if (m_e <= 0) continue;
                    if (beg < 0 || beg + m_e > B * c_.top_k || sc_moe_gu_cap_ <= 0) {
                        err = "[ink-hip] Fast Path A2 MoE bucket out of bounds (beg=" + std::to_string(beg) +
                              ", m_e=" + std::to_string(m_e) + ", pairs=" + std::to_string(B * c_.top_k) + ")";
                        return false;
                    }
                    const int dir_idx = model_->dir_index_of[ip * c_.n_experts + eid];
                    if (dir_idx < 0) continue;
                    const auto& ent = dir[dir_idx];
                    const uint8_t* body_dev = expert_gpu_arena_.dev_ptr + expert_gpu_offset_[dir_idx];
                    run_single_expert_blaslt(
                        body_dev, ent.gu_code_id, ent.dn_code_id, ent.dn_offset,
                        expert_sf4_outer_[dir_idx].first, expert_sf4_outer_[dir_idx].second,
                        d_bx + static_cast<size_t>(beg) * D,
                        d_gu, d_act, d_by + static_cast<size_t>(beg) * D,
                        m_e, sc_moe_gu_cap_, D, c_.inter);
                }

                k_moe_reduce_buckets<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                    d_by, d_s2b, tk_w, moeout, B, c_.top_k, D);
            } else if (layer_all_gpu_[ip]) {
                const GpuExpertDesc* d_exps = reinterpret_cast<const GpuExpertDesc*>(d_layer_exps_[ip].dev_ptr);
                int32_t* d_off = reinterpret_cast<int32_t*>(sc_exp_off_.dev_ptr);
                int32_t* d_cur = reinterpret_cast<int32_t*>(sc_exp_cur_.dev_ptr);
                int32_t* d_btk = reinterpret_cast<int32_t*>(sc_bucket_tok_.dev_ptr);
                int32_t* d_s2b = reinterpret_cast<int32_t*>(sc_slotk_bidx_.dev_ptr);
                half_t*  d_act = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                float*   d_by  = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                const int tot_pairs = B * c_.top_k;
                if (B >= 16) {
                    (void)hipMemsetAsync(d_off, 0, 129 * sizeof(int32_t), stream_);
                    const int cblks = std::min(64, (tot_pairs + 255) / 256);
                    k_moe_count_buckets<<<cblks, 256, 0, stream_>>>(tk_i, d_off, tot_pairs);
                    k_moe_prefix_buckets<<<1, 128, 0, stream_>>>(d_off, d_off, d_cur);
                    k_moe_scatter_buckets<<<(tot_pairs + 255) / 256, 256, 0, stream_>>>(tk_i, d_cur, d_btk, d_s2b, tot_pairs, c_.top_k);
                } else {
                    k_moe_build_buckets<<<1, 128, 0, stream_>>>(tk_i, d_off, d_btk, d_s2b, B, c_.top_k);
                }
                if (mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                    if (B >= 64) {
                        // Wave-5 BM=128, BN=128 Grouped-MoE with m_tiles=8 hot-expert fan-out!
                        // Cold/avg experts (m_e <= 128) exit z>=1 on instruction #1 -> 1.0x W reads;
                        // Hot Zipfian experts (m_e=500..1000) fan out across 8 parallel CTAs sharing W in L2/MALL (ZERO tail stragglers!)
                        const int m_tiles_128 = (B >= 1024) ? 8 : ((B >= 256) ? 4 : 2);
                        const dim3 g13((2 * c_.inter + 127) / 128, c_.n_experts, m_tiles_128);
                        const dim3 g2((D + 127) / 128, c_.n_experts, m_tiles_128);
                        int8_t* xq = reinterpret_cast<int8_t*>(sc_x_q8_.dev_ptr);
                        float*  xd = reinterpret_cast<float*>(sc_x_d_.dev_ptr);
                        int8_t* aq = reinterpret_cast<int8_t*>(sc_act_q8_.dev_ptr);
                        float*  ad = reinterpret_cast<float*>(sc_act_d_.dev_ptr);

                        pmark(6);
                        if (use_i8_moe_wmma_ && layer_gu_class_[ip] == 0) {
                            const int nb64 = B * (D >> 6);
                            k_quant_f32_to_q8_64<<<(nb64 + 15) / 16, 256, 0, stream_>>>(mlpin, xq, xd, nb64);
                            k_moe_grouped_w13_situ_wmma_sk_i8_128<<<g13, 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, xq, xd, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        } else if (use_i8_moe_wmma_ && layer_gu_class_[ip] == 1) {
                            const int nb16 = B * (D >> 4);
                            k_quant_f32_to_q8_16<<<(nb16 + 63) / 64, 256, 0, stream_>>>(mlpin, xq, xd, nb16);
                            k_moe_grouped_w13_situ_wmma_sf4_i8_128<<<g13, 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, xq, xd, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        } else {
                            k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                            if (layer_gu_class_[ip] == 0) {
                                k_moe_grouped_w13_situ_wmma_128_spec<0><<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                            } else if (layer_gu_class_[ip] == 1) {
                                k_moe_grouped_w13_situ_wmma_128_spec<1><<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                            } else if (layer_gu_class_[ip] == 2) {
                                k_moe_grouped_w13_situ_wmma_128_spec<2><<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                            } else {
                                k_moe_grouped_w13_situ_wmma_128<<<g13, 256, 0, stream_>>>(
                                    d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                            }
                        }
                        pmark(7);
                        if (use_i8_moe_wmma_ && layer_dn_class_[ip] == 0) {
                            const int nb64 = tot_pairs * (c_.inter >> 6);
                            k_quant_f16_to_q8_64<<<(nb64 + 15) / 16, 256, 0, stream_>>>(d_act, aq, ad, nb64);
                            k_moe_grouped_w2_wmma_sk_i8_128<<<g2, 256, 0, stream_>>>(
                                d_exps, d_off, aq, ad, d_by, D, c_.inter);
                        } else if (use_i8_moe_wmma_ && layer_dn_class_[ip] == 1) {
                            const int nb16 = tot_pairs * (c_.inter >> 4);
                            k_quant_f16_to_q8_16<<<(nb16 + 63) / 64, 256, 0, stream_>>>(d_act, aq, ad, nb16);
                            k_moe_grouped_w2_wmma_sf4_i8_128<<<g2, 256, 0, stream_>>>(
                                d_exps, d_off, aq, ad, d_by, D, c_.inter);
                        } else {
                            if (layer_dn_class_[ip] == 0) {
                                k_moe_grouped_w2_wmma_128_spec<0><<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            } else if (layer_dn_class_[ip] == 1) {
                                k_moe_grouped_w2_wmma_128_spec<1><<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            } else if (layer_dn_class_[ip] == 2) {
                                k_moe_grouped_w2_wmma_128_spec<2><<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            } else {
                                k_moe_grouped_w2_wmma_128<<<g2, 256, 0, stream_>>>(
                                    d_exps, d_off, d_act, d_by, D, c_.inter);
                            }
                        }
                    } else {
                        k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                        const dim3 g13((2 * c_.inter + 127) / 128, c_.n_experts, 1);
                        const dim3 g2((D + 127) / 128, c_.n_experts, 1);
                        // Use exact FP16 WMMA for B < 64 (0.09 ms faster at B=8 AND bit-consistent with B=1 decode!)
                        const bool small_i8 = false;
                        pmark(6);
                        if (small_i8 && layer_gu_class_[ip] == 0) {
                            k_moe_grouped_w13_situ_wmma_sk_i8<<<g13, 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        } else if (small_i8 && layer_gu_class_[ip] == 1) {
                            k_moe_grouped_w13_situ_wmma_sf4_i8<<<g13, 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        } else {
                            k_moe_grouped_w13_situ_wmma<<<g13, 256, 0, stream_>>>(
                                d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                        }
                        pmark(7);
                        if (small_i8 && layer_dn_class_[ip] == 0) {
                            k_moe_grouped_w2_wmma_sk_i8<<<g2, 256, 0, stream_>>>(
                                d_exps, d_off, d_act, d_by, D, c_.inter);
                        } else if (small_i8 && layer_dn_class_[ip] == 1) {
                            k_moe_grouped_w2_wmma_sf4_i8<<<g2, 256, 0, stream_>>>(
                                d_exps, d_off, d_act, d_by, D, c_.inter);
                        } else {
                            k_moe_grouped_w2_wmma<<<g2, 256, 0, stream_>>>(
                                d_exps, d_off, d_act, d_by, D, c_.inter);
                        }
                    }
                } else {
                    k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                    k_moe_grouped_w13_situ<<<dim3((c_.inter + 3) / 4, c_.n_experts), 256, 0, stream_>>>(
                        d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    k_moe_grouped_w2<<<dim3((D + 7) / 8, c_.n_experts), 256, 0, stream_>>>(
                        d_exps, d_off, d_act, d_by, D, c_.inter);
                }

                k_moe_reduce_buckets<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                    d_by, d_s2b, tk_w, moeout, B, c_.top_k, D);
            } else if (hybrid_.active() && B <= 16) {
                // Fast Path B1: Strata Predictive CPU/RAM Hybrid Decode!
                // Coherent Host Doorbell + Spin-Wait AVX-512/AVX2 CPU Worker Pool overlapped with GPU VRAM + Elastic Cache + 3x-Loop Predictive Async DMA Prefetch!
                if (!hybrid_.ensure_doorbell_cap(B, D, c_.top_k)) {
                    err = "hybrid doorbell resize failed";
                    return false;
                }
                if (!hip_ok(hipMemcpyAsync(hybrid_.h_ids(), tk_i, static_cast<size_t>(B) * c_.top_k * 4, hipMemcpyDeviceToHost, stream_), "hyb ids D2H", err) ||
                    !hip_ok(hipMemcpyAsync(hybrid_.h_ws(),  tk_w, static_cast<size_t>(B) * c_.top_k * 4, hipMemcpyDeviceToHost, stream_), "hyb ws D2H", err) ||
                    !hip_ok(hipMemcpyAsync(hybrid_.h_x(),   mlpin, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToHost, stream_), "hyb x D2H", err) ||
                    !hip_ok(hipStreamSynchronize(stream_), "hyb route sync", err)) {
                    return false;
                }
                const auto& dir = adj_dir_;
                const bool is_core_repeat_candidate = (ip >= c_.core_start && ip < c_.core_end && visit < (c_.core_start + 2 * (c_.core_end - c_.core_start)));

                std::vector<GpuExpertDesc> step_exps = h_layer_exps_[ip];
                bool any_gpu_exp = false;
                bool any_cache_patch = false;
                std::vector<HybridCpuExpertPool::CpuJob> cpu_jobs;
                std::vector<HybridCpuExpertPool::CpuJob> mx8_stream_jobs;
                std::vector<std::vector<uint8_t>> disk_fallback_bufs;

                int eid_to_job[128];
                int eid_to_mx8[128];
                std::memset(eid_to_job, -1, sizeof(eid_to_job));
                std::memset(eid_to_mx8, -1, sizeof(eid_to_mx8));
                const int32_t* h_ids = hybrid_.h_ids();
                const float*   h_ws  = hybrid_.h_ws();

                for (int b = 0; b < B; ++b) {
                    for (int k = 0; k < c_.top_k; ++k) {
                        const int eid = h_ids[b * c_.top_k + k];
                        const float rw = h_ws[b * c_.top_k + k];
                        const int dir_idx = model_->dir_index_of[ip * c_.n_experts + eid];
                        if (dir_idx < 0) continue;
                        if (ep2_active_ && peer_expert_gpu_offset_[dir_idx] != static_cast<size_t>(-1)) continue;  // handled by GPU 1!
                        if (step_exps[eid].body != nullptr) {
                            any_gpu_exp = true;
                            continue;
                        }
                        // Probe elastic VRAM cache ring first (zero stall!)
                        const uint8_t* cached_dev = hybrid_.probe_ready(dir_idx);
                        if (cached_dev != nullptr) {
                            step_exps[eid].body = cached_dev;
                            step_exps[eid].dn_offset = dir[dir_idx].dn_offset;
                            step_exps[eid].gu_code = dir[dir_idx].gu_code_id;
                            step_exps[eid].dn_code = dir[dir_idx].dn_code_id;
                            any_gpu_exp = true;
                            any_cache_patch = true;
                            continue;
                        }
                        // MX8 RAM experts stream to GPU via double-buffered PCIe DMA (15x faster than scalar CPU MX8!)
                        if (dir[dir_idx].gu_code_id == CODE_MX8 || dir[dir_idx].dn_code_id == CODE_MX8) {
                            int mi = eid_to_mx8[eid];
                            if (mi < 0) {
                                const uint8_t* host_body = nullptr;
                                if (!expert_ram_bodies_[dir_idx].empty()) {
                                    host_body = expert_ram_bodies_[dir_idx].data();
                                    hybrid_.prefetch_async(dir_idx, host_body, dir[dir_idx].body_bytes, false);
                                } else if (cold_store_.is_open()) {
                                    int cslot = -1;
                                    const uint8_t* cp = cold_store_.demand_wait(dir_idx, ip, cold_gen_, cslot);
                                    if (!cp) { err = "hip cold_store mx8 read failed"; return false; }
                                    disk_fallback_bufs.emplace_back(cp, cp + dir[dir_idx].body_bytes);
                                    cold_store_.unpin_slot(cslot);
                                    host_body = disk_fallback_bufs.back().data();
                                } else {
                                    disk_fallback_bufs.emplace_back();
                                    uint32_t tmp_dn = 0; uint8_t tmp_gu = 0, tmp_dnc = 0;
                                    if (!read_expert_for_hip(dir_idx, disk_fallback_bufs.back(), tmp_dn, tmp_gu, tmp_dnc, err)) return false;
                                    host_body = disk_fallback_bufs.back().data();
                                }
                                mi = static_cast<int>(mx8_stream_jobs.size());
                                eid_to_mx8[eid] = mi;
                                HybridCpuExpertPool::CpuJob job;
                                job.dir_idx = dir_idx;
                                job.body = host_body;
                                job.gu_code = dir[dir_idx].gu_code_id;
                                job.dn_code = dir[dir_idx].dn_code_id;
                                job.dn_offset = dir[dir_idx].dn_offset;
                                mx8_stream_jobs.push_back(std::move(job));
                            }
                            mx8_stream_jobs[mi].tok_idx.push_back(b);
                            mx8_stream_jobs[mi].tok_w.push_back(rw);
                            continue;
                        }
                        // S_K / SF4 RAM expert -> dispatch to CPU AVX-512/AVX2 pool + trigger predictive async DMA prefetch!
                        int ji = eid_to_job[eid];
                        if (ji < 0) {
                            const uint8_t* host_body = nullptr;
                            if (!expert_ram_bodies_[dir_idx].empty()) {
                                host_body = expert_ram_bodies_[dir_idx].data();
                                hybrid_.prefetch_async(dir_idx, host_body, dir[dir_idx].body_bytes, is_core_repeat_candidate);
                            } else if (cold_store_.is_open()) {
                                int cslot = -1;
                                const uint8_t* cp = cold_store_.demand_wait(dir_idx, ip, cold_gen_, cslot);
                                if (!cp) { err = "hip cold_store cpu read failed"; return false; }
                                disk_fallback_bufs.emplace_back(cp, cp + dir[dir_idx].body_bytes);
                                cold_store_.unpin_slot(cslot);
                                host_body = disk_fallback_bufs.back().data();
                            } else {
                                disk_fallback_bufs.emplace_back();
                                uint32_t tmp_dn = 0; uint8_t tmp_gu = 0, tmp_dnc = 0;
                                if (!read_expert_for_hip(dir_idx, disk_fallback_bufs.back(), tmp_dn, tmp_gu, tmp_dnc, err)) return false;
                                host_body = disk_fallback_bufs.back().data();
                            }
                            ji = static_cast<int>(cpu_jobs.size());
                            eid_to_job[eid] = ji;
                            HybridCpuExpertPool::CpuJob job;
                            job.dir_idx = dir_idx;
                            job.body = host_body;
                            job.gu_code = dir[dir_idx].gu_code_id;
                            job.dn_code = dir[dir_idx].dn_code_id;
                            job.dn_offset = dir[dir_idx].dn_offset;
                            cpu_jobs.push_back(std::move(job));
                        }
                        cpu_jobs[ji].tok_idx.push_back(b);
                        cpu_jobs[ji].tok_w.push_back(rw);
                    }
                }

                // OVERLAPPED EXECUTION: CPU computes S_K/SF4 RAM experts in DDR5 WHILE GPU computes VRAM + ElasticCache + double-buffered MX8!
                if (!cpu_jobs.empty()) {
                    hybrid_.pool().launch_async(&cpu_jobs, hybrid_.h_x(), hybrid_.h_y(), B, D, c_.inter);
                }
                for (int p_i = 0; p_i < 2 && p_i < static_cast<int>(mx8_stream_jobs.size()); ++p_i) {
                    (void)hybrid_.stage_prefill_async(p_i, mx8_stream_jobs[p_i].body, dir[mx8_stream_jobs[p_i].dir_idx].body_bytes, false);
                }
                if (any_gpu_exp) {
                    const GpuExpertDesc* d_exps = reinterpret_cast<const GpuExpertDesc*>(d_layer_exps_[ip].dev_ptr);
                    if (any_cache_patch) {
                        (void)hipMemcpyAsync(hybrid_.d_step_exps(), step_exps.data(),
                                             step_exps.size() * sizeof(GpuExpertDesc), hipMemcpyHostToDevice, stream_);
                        d_exps = hybrid_.d_step_exps();
                    }
                    k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                    k_moe_direct_w13_situ<<<dim3((c_.inter + 3) / 4, B * c_.top_k), 256, 0, stream_>>>(
                        d_exps, tk_i, mlpf16, reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr),
                        B, c_.top_k, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    k_moe_direct_w2_reduce<<<dim3((D + 7) / 8, B), 256, 0, stream_>>>(
                        d_exps, tk_i, tk_w, reinterpret_cast<const half_t*>(sc_moe_act_.dev_ptr),
                        moeout, B, c_.top_k, D, c_.inter);
                } else {
                    (void)hipMemsetAsync(moeout, 0, static_cast<size_t>(B) * D * 4, stream_);
                }
                for (int mi = 0; mi < static_cast<int>(mx8_stream_jobs.size()); ++mi) {
                    const auto& mj = mx8_stream_jobs[mi];
                    const int m = static_cast<int>(mj.tok_idx.size());
                    const uint8_t* body_dev = hybrid_.wait_prefill_stage(mi, stream_);
                    const int chunk_cap = std::max(1, sc_moe_gu_cap_);
                    float* gu_buf = reinterpret_cast<float*>(sc_moe_gu_.dev_ptr);
                    half_t* act_buf = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                    float* ye_buf = reinterpret_cast<float*>(sc_moe_ye_.dev_ptr);
                    for (int off = 0; off < m; off += chunk_cap) {
                        const int cur_m = std::min(chunk_cap, m - off);
                        (void)hipMemcpyAsync(sc_exp_tok_idx_.dev_ptr, mj.tok_idx.data() + off, cur_m * 4, hipMemcpyHostToDevice, stream_);
                        (void)hipMemcpyAsync(sc_exp_tok_w_.dev_ptr,   mj.tok_w.data() + off,   cur_m * 4, hipMemcpyHostToDevice, stream_);
                        k_gather_rows_f16<<<(cur_m * D + 255) / 256, 256, 0, stream_>>>(
                            mlpin, reinterpret_cast<const int32_t*>(sc_exp_tok_idx_.dev_ptr), mlpf16, cur_m, D);
                        launch_mx8_linear(body_dev, body_dev + static_cast<size_t>(2 * c_.inter) * D, mlpf16, gu_buf, cur_m, 2 * c_.inter, D, mem_info_.gen, stream_);
                        k_situ_interleaved<<<(cur_m * c_.inter + 255) / 256, 256, 0, stream_>>>(gu_buf, act_buf, cur_m, c_.inter, c_.situ_beta, c_.situ_lb);
                        launch_mx8_linear(body_dev + mj.dn_offset, body_dev + mj.dn_offset + static_cast<size_t>(D) * c_.inter, act_buf, ye_buf, cur_m, D, c_.inter, mem_info_.gen, stream_);
                        k_scatter_add_expert<<<(D + 255) / 256, 256, 0, stream_>>>(
                            ye_buf, reinterpret_cast<const int32_t*>(sc_exp_tok_idx_.dev_ptr),
                            reinterpret_cast<const float*>(sc_exp_tok_w_.dev_ptr), moeout, cur_m, D);
                    }
                    hybrid_.mark_prefill_compute_done(mi, stream_);
                    if (mi + 2 < static_cast<int>(mx8_stream_jobs.size())) {
                        (void)hybrid_.stage_prefill_async(mi + 2, mx8_stream_jobs[mi + 2].body, dir[mx8_stream_jobs[mi + 2].dir_idx].body_bytes, true);
                    }
                }
                if (!cpu_jobs.empty()) {
                    hybrid_.pool().wait_and_reduce();
                    // Upload ONLY the tiny 9 KB/tok output vector (NOT 4.5 MB expert weights!) and accumulate!
                    (void)hipMemcpyAsync(hybrid_.d_cpu_y(), hybrid_.h_y(), static_cast<size_t>(B) * D * 4, hipMemcpyHostToDevice, stream_);
                    k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(moeout, hybrid_.d_cpu_y(), moeout, B * D);
                }
                if (!disk_fallback_bufs.empty()) {
                    (void)hipStreamSynchronize(hybrid_.dma_stream());
                    (void)hipStreamSynchronize(stream_);
                }
            } else {
                // Fast Path B2: Chunked Double-Buffered GPU Matrix-Core Expert Streaming for Hybrid Prefill!
                // Overlaps PCIe H2D of RAM expert i+1 on dma_stream_ with GPU hipBLASLt/WMMA compute of expert i!
                int32_t* d_off = reinterpret_cast<int32_t*>(sc_exp_off_.dev_ptr);
                int32_t* d_btk = reinterpret_cast<int32_t*>(sc_bucket_tok_.dev_ptr);
                int32_t* d_s2b = reinterpret_cast<int32_t*>(sc_slotk_bidx_.dev_ptr);
                half_t*  d_bx  = reinterpret_cast<half_t*>(sc_bucket_x_f16_.dev_ptr);
                float*   d_gu  = reinterpret_cast<float*>(sc_moe_gu_.dev_ptr);
                half_t*  d_act = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                float*   d_by  = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);

                k_moe_build_buckets<<<1, 128, 0, stream_>>>(tk_i, d_off, d_btk, d_s2b, B, c_.top_k);
                k_gather_rows_f16<<<(B * c_.top_k * D + 255) / 256, 256, 0, stream_>>>(
                    mlpin, d_btk, d_bx, B * c_.top_k, D);

                int32_t h_off[129];
                if (!hip_ok(hipMemcpyAsync(h_off, d_off, 129 * sizeof(int32_t), hipMemcpyDeviceToHost, stream_), "moe off D2H", err) ||
                    !hip_ok(hipStreamSynchronize(stream_), "moe off sync", err)) {
                    return false;
                }

                const auto& dir = adj_dir_;
                struct ActiveExp {
                    int eid, dir_idx, beg, m_e;
                    const uint8_t* dev_ptr;
                    const uint8_t* host_ptr;
                };
                std::vector<ActiveExp> gpu_exps, ram_exps;
                std::vector<std::vector<uint8_t>> disk_bufs;

                for (int eid = 0; eid < c_.n_experts; ++eid) {
                    const int beg = h_off[eid];
                    const int m_e = h_off[eid + 1] - beg;
                    if (m_e <= 0) continue;
                    const int dir_idx = model_->dir_index_of[ip * c_.n_experts + eid];
                    if (dir_idx < 0) continue;
                    if (ep2_active_ && peer_expert_gpu_offset_[dir_idx] != static_cast<size_t>(-1)) continue;  // handled by GPU 1!
                    const uint8_t* dev_p = nullptr;
                    if (expert_gpu_offset_[dir_idx] != static_cast<size_t>(-1)) {
                        dev_p = expert_gpu_arena_.dev_ptr + expert_gpu_offset_[dir_idx];
                    } else if (hybrid_.active()) {
                        dev_p = hybrid_.probe_ready(dir_idx);
                    }
                    if (dev_p != nullptr) {
                        gpu_exps.push_back({eid, dir_idx, beg, m_e, dev_p, nullptr});
                    } else {
                        const uint8_t* hp = nullptr;
                        if (!expert_ram_bodies_[dir_idx].empty()) {
                            hp = expert_ram_bodies_[dir_idx].data();
                        } else if (cold_store_.is_open()) {
                            int cslot = -1;
                            const uint8_t* cp = cold_store_.demand_wait(dir_idx, ip, cold_gen_, cslot);
                            if (!cp) { err = "hip cold_store prefill read failed"; return false; }
                            disk_bufs.emplace_back(cp, cp + dir[dir_idx].body_bytes);
                            cold_store_.unpin_slot(cslot);
                            hp = disk_bufs.back().data();
                        } else {
                            disk_bufs.emplace_back();
                            uint32_t tmp_dn = 0; uint8_t tmp_gu = 0, tmp_dnc = 0;
                            if (!read_expert_for_hip(dir_idx, disk_bufs.back(), tmp_dn, tmp_gu, tmp_dnc, err)) return false;
                            hp = disk_bufs.back().data();
                        }
                        ram_exps.push_back({eid, dir_idx, beg, m_e, nullptr, hp});
                    }
                }

                auto exec_one = [&](const ActiveExp& ae, const uint8_t* body_dev) -> bool {
                    if (ae.beg < 0 || ae.m_e < 0 || ae.beg + ae.m_e > B * c_.top_k || sc_moe_gu_cap_ <= 0) {
                        err = "[ink-hip] Fast Path B2 MoE bucket out of bounds (beg=" + std::to_string(ae.beg) +
                              ", m_e=" + std::to_string(ae.m_e) + ", pairs=" + std::to_string(B * c_.top_k) + ")";
                        return false;
                    }
                    const auto& ent = dir[ae.dir_idx];
                    float sf4_w13 = expert_sf4_outer_[ae.dir_idx].first;
                    float sf4_w2  = expert_sf4_outer_[ae.dir_idx].second;
                    if (ae.host_ptr && expert_ram_bodies_[ae.dir_idx].empty() &&
                        (ent.gu_code_id == CODE_SF4 || ent.dn_code_id == CODE_SF4)) {
                        float hdr[4];
                        std::memcpy(hdr, ae.host_ptr, 16);
                        sf4_w13 = hdr[0] / hdr[1];
                        sf4_w2  = hdr[2] / hdr[3];
                    }
                    const half_t* x_slice = d_bx + static_cast<size_t>(ae.beg) * D;
                    float* y_slice = d_by + static_cast<size_t>(ae.beg) * D;
                    if (blaslt_.ready() && ae.m_e >= 8 && d_exp_w13_f16_.dev_ptr && d_exp_w2_f16_.dev_ptr) {
                        run_single_expert_blaslt(body_dev, ent.gu_code_id, ent.dn_code_id, ent.dn_offset,
                                                 sf4_w13, sf4_w2, x_slice, d_gu, d_act, y_slice, ae.m_e, sc_moe_gu_cap_, D, c_.inter);
                    } else {
                        // #6580: Chunk m_e by sc_moe_gu_cap_ so d_gu [sc_moe_gu_cap_, 2*inter] NEVER overflows when m_e > B!
                        for (int off = 0; off < ae.m_e; off += sc_moe_gu_cap_) {
                            const int cur_m = std::min(sc_moe_gu_cap_, ae.m_e - off);
                            const half_t* x_sub = x_slice + static_cast<size_t>(off) * D;
                            float* y_sub = y_slice + static_cast<size_t>(off) * D;
                            if (ent.gu_code_id == CODE_SF4) {
                                const uint8_t* p = body_dev + 16;
                                launch_sf4_linear(p, p + static_cast<size_t>(2 * c_.inter) * (D >> 1), sf4_w13, x_sub, d_gu, cur_m, 2 * c_.inter, D, mem_info_.gen, stream_, false);
                            } else if (ent.gu_code_id == CODE_MX8) {
                                launch_mx8_linear(body_dev, body_dev + static_cast<size_t>(2 * c_.inter) * D, x_sub, d_gu, cur_m, 2 * c_.inter, D, mem_info_.gen, stream_);
                            } else if (ent.gu_code_id == CODE_GGUF_RAW) {
                                launch_gguf_linear(body_dev, 0, 1, x_sub, d_gu, cur_m, 2 * c_.inter, D, c_.inter, stream_);
                            } else {
                                launch_sk_linear(body_dev + static_cast<size_t>(c_.inter) * 2, ent.gu_code_id, x_sub, d_gu, cur_m, 2 * c_.inter, D, mem_info_.gen, stream_);
                            }
                            k_situ_interleaved<<<(cur_m * c_.inter + 255) / 256, 256, 0, stream_>>>(
                                d_gu, d_act, cur_m, c_.inter, c_.situ_beta, c_.situ_lb);
                            if (ent.dn_code_id == CODE_SF4) {
                                const uint8_t* p = body_dev + ent.dn_offset;
                                launch_sf4_linear(p, p + static_cast<size_t>(D) * (c_.inter >> 1), sf4_w2, d_act, y_sub, cur_m, D, c_.inter, mem_info_.gen, stream_, false);
                            } else if (ent.dn_code_id == CODE_MX8) {
                                launch_mx8_linear(body_dev + ent.dn_offset, body_dev + ent.dn_offset + static_cast<size_t>(D) * c_.inter, d_act, y_sub, cur_m, D, c_.inter, mem_info_.gen, stream_);
                            } else if (ent.dn_code_id == CODE_GGUF_RAW) {
                                launch_gguf_linear(body_dev, ent.dn_offset, 0, d_act, y_sub, cur_m, D, c_.inter, c_.inter, stream_);
                            } else {
                                launch_sk_paired2_linear(body_dev + ent.dn_offset, ent.dn_code_id,
                                                         reinterpret_cast<const half_t*>(body_dev), d_act, y_sub, cur_m, D, c_.inter, stream_);
                            }
                        }
                    }
                    return true;
                };

                // Kick off initial 2 RAM expert DMAs BEFORE running GPU-resident experts -> full PCIe overlap!
                if (hybrid_.active()) {
                    for (int p_i = 0; p_i < 2 && p_i < static_cast<int>(ram_exps.size()); ++p_i) {
                        (void)hybrid_.stage_prefill_async(p_i, ram_exps[p_i].host_ptr, dir[ram_exps[p_i].dir_idx].body_bytes, false);
                    }
                }
                if (ep2_active_) {
                    (void)hipMemsetAsync(d_by, 0, static_cast<size_t>(B) * c_.top_k * D * sizeof(float), stream_);
                }
                for (const auto& ge : gpu_exps) {
                    if (!exec_one(ge, ge.dev_ptr)) return false;
                }

                for (int ri = 0; ri < static_cast<int>(ram_exps.size()); ++ri) {
                    const auto& re = ram_exps[ri];
                    const uint8_t* body_dev = nullptr;
                    if (hybrid_.active()) {
                        body_dev = hybrid_.wait_prefill_stage(ri, stream_);
                        if (!exec_one(re, body_dev)) return false;
                        hybrid_.mark_prefill_compute_done(ri, stream_);
                        if (ri + 2 < static_cast<int>(ram_exps.size())) {
                            const auto& nxt = ram_exps[ri + 2];
                            (void)hybrid_.stage_prefill_async(ri + 2, nxt.host_ptr, dir[nxt.dir_idx].body_bytes, true);
                        }
                    } else {
                        if (!hip_ok(hipMemcpyAsync(d_exp_stage_.dev_ptr, re.host_ptr, dir[re.dir_idx].body_bytes, hipMemcpyHostToDevice, stream_),
                                    "exp stage H2D", err)) return false;
                        if (!exec_one(re, d_exp_stage_.dev_ptr)) return false;
                    }
                }

                if (!disk_bufs.empty()) {
                    if (hybrid_.active()) (void)hipStreamSynchronize(hybrid_.dma_stream());
                    (void)hipStreamSynchronize(stream_);
                }

                k_moe_reduce_buckets<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                    d_by, d_s2b, tk_w, moeout, B, c_.top_k, D);
            }
            if (ep2_active_ && !layer_all_ep2_[ip]) {
                if (!finish_peer_moe_add(B, moeout, err)) return false;
            }

            // Post-MLP ShortConv + residual add (already fused into k_moe_reduce_sconv_add for B<=4 decode!)
            if (!fused_moe_post) {
                run_sconv(moeout, p + ".mlp_sconv", nullptr, d_sconv_mlp_all_, msc, D, 0, rsl ? &rsl->sconv_rec_mlp : nullptr, visit);
                k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, msc, cur, B * D);
            }
            pmark(8);
            pflush();
            ++visit;

            // Periodic TDR/MES flush on heavy prefill batches (skipped during HIP Graph capture!)
            if (!use_graph && B >= 32 && guard_.visit_flush_stride > 0 && (visit % guard_.visit_flush_stride == 0)) {
                if (!hip_ok(hipStreamSynchronize(stream_), "tdr visit flush", err)) return false;
            }
            return true;
        };

        // 2. Execute all 48 virtual visits: Prelude (0..5) + Core (6..17 x 3 passes) + Coda (18..23)
        for (int ip = 0; ip < c_.core_start; ++ip) {
            if (!run_visit(ip, false)) return false;
        }
        (void)hipMemcpyAsync(sc_prelude_.dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
        n_prelude_blocks = n_blocks;

        for (int p_idx = 0; p_idx < c_.num_passes; ++p_idx) {
            if (p_idx > 0) {
                bank_ptrs.clear();
                n_bank = 0;
                for (int k = n_prelude_blocks + 1; k < n_blocks; ++k) {
                    (void)hipMemcpyAsync(sc_bank_[n_bank].dev_ptr, sc_blocks_[k].dev_ptr, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    bank_ptrs.push_back(reinterpret_cast<const float*>(sc_bank_[n_bank].dev_ptr));
                    ++n_bank;
                }
                (void)hipMemcpyAsync(sc_bank_out_.dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                (void)hipMemcpyAsync(cur, sc_prelude_.dev_ptr, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                n_blocks = n_prelude_blocks;
                blk_ptrs.resize(n_prelude_blocks);
            }
            for (int ip = c_.core_start; ip < c_.core_end; ++ip) {
                if (p_idx > 0) {
                    run_attnres(bank_ptrs, reinterpret_cast<const float*>(sc_bank_out_.dev_ptr),
                                reinterpret_cast<const float*>(d_msg_queries_[ip].dev_ptr), 4, msgs);
                }
                if (!run_visit(ip, p_idx > 0)) return false;
            }
        }
        for (int ip = c_.core_end; ip < c_.n_phys; ++ip) {
            if (!run_visit(ip, false)) return false;
        }

        // 3. Final output depth mix + output_norm (fused!)
        float* h_norm = reinterpret_cast<float*>(sc_h_norm_.dev_ptr);
        run_attnres_norm(blk_ptrs, cur, f32_ptr("model.llm.attn_res.output_query"),
                         f32_ptr("model.llm.norm.weight"), h_norm);
        maybe_tap(c_.n_visits, h_norm);

        if (use_graph) {
            if (do_dspark_ctx) {
                launch_dspark_ctx_kernels(B);
            }
            const int V = c_.vocab_unpadded;
            const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
            k_unembed_softcap<<<dim3((V + 7) / 8, (B + 15) / 16), 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                h_norm, reinterpret_cast<const int32_t*>(sc_want_idx_.dev_ptr),
                reinterpret_cast<float*>(sc_logits_.dev_ptr), B, V, D, c_.logit_softcap);

            hipGraph_t g = nullptr;
            if (!hip_ok(hipStreamEndCapture(stream_, &g), "hipStreamEndCapture", err)) return false;
            hipGraphExec_t& g_exec = is_verify_graph ? graph_verify_exec_[B] : graph_exec_;
            hipError_t ie = hipGraphInstantiate(&g_exec, g, nullptr, nullptr, 0);
            (void)hipGraphDestroy(g);
            if (!hip_ok(ie, "hipGraphInstantiate", err)) return false;
            if (is_verify_graph) {
                vgraph_slot_[B] = rec_slot;
                vgraph_need_taps_[B] = need_taps;
                vgraph_kv_gen_[B] = kv_gen_sum;
                vgraph_scratch_B_[B] = scratch_B_;
            } else {
                graph_B_ = B;
                graph_need_taps_ = need_taps;
                graph_kv_gen_ = kv_gen_sum;
                graph_scratch_B_ = scratch_B_;
            }

            if (!hip_ok(hipGraphLaunch(g_exec, stream_), "hipGraphLaunch(first)", err)) return false;
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                sl.seq_len = std::max(sl.seq_len, batch.pos[b] + 1);
                if (do_dspark_ctx) sl.dspark_len = std::max(sl.dspark_len, batch.pos[b] + 1);
            }
            const size_t lbytes = static_cast<size_t>(B) * V * sizeof(float);
            if (!hip_ok(hipMemcpyAsync(logits_out, sc_logits_.dev_ptr, lbytes, hipMemcpyDeviceToHost, stream_),
                        "logits D2H", err)) return false;
            return hip_ok(hipStreamSynchronize(stream_), "graph sync", err);
        }

        // Update slot seq_len + on-GPU DSpark context KV/f_rnn
        for (int b = 0; b < B; ++b) {
            auto& sl = slots_[batch.slot_ids[b]];
            sl.seq_len = std::max(sl.seq_len, batch.pos[b] + 1);
        }
        if (do_dspark_ctx) {
            launch_dspark_ctx_kernels(B);
            for (int b = 0; b < B; ++b) {
                auto& sl = slots_[batch.slot_ids[b]];
                sl.dspark_len = std::max(sl.dspark_len, batch.pos[b] + 1);
            }
        }
        if (taps_out) {
            for (int k = 0; k < 4; ++k) {
                taps_out->taps[k].resize(static_cast<size_t>(B) * D);
                (void)hipMemcpyAsync(taps_out->taps[k].data(), sc_taps_[k].dev_ptr, static_cast<size_t>(B) * D * 4,
                               hipMemcpyDeviceToHost, stream_);
            }
        }

        // 4. Unembed + softcap(30.0) ONLY for tokens with want_logits[b] == 1
        if (logits_out != nullptr) {
            std::vector<int32_t> want_idx;
            for (int b = 0; b < B; ++b) if (batch.want_logits[b]) want_idx.push_back(b);
            const int n_want = static_cast<int>(want_idx.size());
            if (n_want > 0) {
                const int V = c_.vocab_unpadded;
                const size_t lbytes = static_cast<size_t>(n_want) * V * sizeof(float);
                if (sc_logits_.bytes < lbytes) {
                    free_gpu_buffer(sc_logits_);
                    if (!hip_ok(alloc_gpu_buffer(sc_logits_, lbytes, mem_info_.is_uma), "logits buf", err)) return false;
                }
                (void)hipMemcpyAsync(sc_want_idx_.dev_ptr, want_idx.data(), n_want * 4, hipMemcpyHostToDevice, stream_);
                const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
                if (prof_on) (void)hipEventRecord(pev[9], stream_);
                k_unembed_softcap<<<dim3((V + 7) / 8, (n_want + 15) / 16), 256, 0, stream_>>>(
                    unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                    h_norm, reinterpret_cast<const int32_t*>(sc_want_idx_.dev_ptr),
                    reinterpret_cast<float*>(sc_logits_.dev_ptr), n_want, V, D, c_.logit_softcap);
                if (prof_on) (void)hipEventRecord(pev[10], stream_);
                if (!hip_ok(hipMemcpyAsync(logits_out, sc_logits_.dev_ptr, lbytes, hipMemcpyDeviceToHost, stream_),
                            "logits D2H", err)) return false;
            }
        }

        const bool ok = hip_ok(hipStreamSynchronize(stream_), "forward_batch sync", err);
        if (prof_on) {
            double sum = 0.0;
            for (int i = 0; i < 8; ++i) sum += p_acc[i];
            float unemb_ms = 0.0f;
            if (logits_out != nullptr) (void)hipEventElapsedTime(&unemb_ms, pev[9], pev[10]);
            std::fprintf(stderr,
                "[HIP-PROF] B=%d sum=%.2fms unemb=%.2fms tot=%.2fms (%.1f tok/s) | res_ctrl=%.2f qkvrg=%.2f sconv_kv=%.2f attn=%.2f wo_post=%.2f moe_prep=%.2f moe_w13=%.2f moe_w2_post=%.2f\n",
                B, sum, unemb_ms, sum + unemb_ms, (sum + unemb_ms > 0.0 ? B * 1000.0 / (sum + unemb_ms) : 0.0),
                p_acc[0], p_acc[1], p_acc[2], p_acc[3], p_acc[4], p_acc[5], p_acc[6], p_acc[7]);
            for (int i = 0; i < 11; ++i) (void)hipEventDestroy(pev[i]);
        }
        free_gpu_buffer(d_img_emb);
        return ok;
    }

    bool prepare_dspark_ctx_ptrs(const StepBatch& batch, std::string& err) {
        const int B = batch.n_tokens;
        bool ptrs_changed = false;
        for (int b = 0; b < B; ++b) {
            auto& sl = slots_[batch.slot_ids[b]];
            const int old_cap = sl.dspark_cap;
            if (!ensure_dspark_slot_cap(sl, batch.pos[b] + 1, err)) return false;
            if (sl.dspark_cap != old_cap) ptrs_changed = true;
        }
        if (!d_ds_k_ptrs_[0].dev_ptr || ptrs_changed) {
            std::vector<uint16_t*> pt(max_slots_, nullptr);
            for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
                for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_k[il].dev_ptr);
                if (!d_ds_k_ptrs_[il].dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_k_ptrs_[il], max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds k ptrs", err)) return false;
                (void)hipMemcpyAsync(d_ds_k_ptrs_[il].dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
                for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_v[il].dev_ptr);
                if (!d_ds_v_ptrs_[il].dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_v_ptrs_[il], max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds v ptrs", err)) return false;
                (void)hipMemcpyAsync(d_ds_v_ptrs_[il].dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
            }
            for (int s = 0; s < max_slots_; ++s) pt[s] = reinterpret_cast<uint16_t*>(slots_[s].dspark_frnn.dev_ptr);
            if (!d_ds_frnn_ptrs_.dev_ptr && !hip_ok(alloc_gpu_buffer(d_ds_frnn_ptrs_, max_slots_ * sizeof(uint16_t*), mem_info_.is_uma), "ds frnn ptrs", err)) return false;
            (void)hipMemcpyAsync(d_ds_frnn_ptrs_.dev_ptr, pt.data(), max_slots_ * sizeof(uint16_t*), hipMemcpyHostToDevice, stream_);
            ++dspark_ptr_gen_;
        }
        return true;
    }

    void launch_dspark_ctx_kernels(int B) {
        launch_dspark_ctx_kernels_from(
            B,
            reinterpret_cast<const float*>(sc_taps_[0].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[1].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[2].dev_ptr),
            reinterpret_cast<const float*>(sc_taps_[3].dev_ptr));
    }

    void launch_dspark_ctx_kernels_from(int B, const float* t0, const float* t1, const float* t2, const float* t3) {
        const int D = dspark_cfg_.hidden;
        const int KV = dspark_cfg_.n_kv;
        const int hd = dspark_cfg_.head_dim;
        const int kv_dim = KV * hd;

        auto softmax4 = [&](int row, float w[4]) {
            float mx = -1e30f, sm = 0.0f;
            for (int i = 0; i < 4; ++i) mx = std::max(mx, h_dspark_fuse_[row * 4 + i]);
            for (int i = 0; i < 4; ++i) { w[i] = std::exp(h_dspark_fuse_[row * 4 + i] - mx); sm += w[i]; }
            for (int i = 0; i < 4; ++i) w[i] /= sm;
        };

        float* ctx = reinterpret_cast<float*>(sc_in0_.dev_ptr);
        float* kc  = reinterpret_cast<float*>(sc_k_conv_.dev_ptr);
        float* vc  = reinterpret_cast<float*>(sc_v_conv_.dev_ptr);
        const int32_t* sids_d = reinterpret_cast<const int32_t*>(sc_slot_ids_.dev_ptr);
        const int32_t* pos_d  = reinterpret_cast<const int32_t*>(sc_pos_.dev_ptr);

        for (int il = 0; il < dspark_cfg_.n_layers; ++il) {
            const std::string p = "layers." + std::to_string(il);
            float w[4];
            softmax4(il, w);
            k_dspark_fuse_mix<<<B, 256, 0, stream_>>>(t0, t1, t2, t3, w[0], w[1], w[2], w[3],
                                                      ds_f32(p + ".dflare_norm.weight"), ctx, B, D, dspark_cfg_.rms_eps);
            launch_bf16_opt(ds_bf16(p + ".self_attn.k_proj_ctx.weight"), ctx, nullptr, kc, B, kv_dim, D);
            launch_bf16_opt(ds_bf16(p + ".self_attn.v_proj_ctx.weight"), ctx, nullptr, vc, B, kv_dim, D);
            k_rms_norm_rows<<<B * KV, 256, 0, stream_>>>(kc, ds_f32(p + ".self_attn.k_norm_ctx.weight"), kc, B * KV, hd, dspark_cfg_.rms_eps);
            k_half_rope<<<dim3(KV, B), hd / 2, 0, stream_>>>(kc, pos_d, 0, B, KV, hd, dspark_cfg_.rope_theta);
            k_dspark_ctx_scatter<<<dim3((kv_dim + 255) / 256, B), 256, 0, stream_>>>(
                kc, reinterpret_cast<uint16_t* const*>(d_ds_k_ptrs_[il].dev_ptr), sids_d, pos_d, B, kv_dim);
            k_dspark_ctx_scatter<<<dim3((kv_dim + 255) / 256, B), 256, 0, stream_>>>(
                vc, reinterpret_cast<uint16_t* const*>(d_ds_v_ptrs_[il].dev_ptr), sids_d, pos_d, B, kv_dim);
        }
        float w_rnn[4];
        softmax4(dspark_cfg_.n_layers, w_rnn);
        k_dspark_fuse_mix<<<B, 256, 0, stream_>>>(t0, t1, t2, t3, w_rnn[0], w_rnn[1], w_rnn[2], w_rnn[3],
                                                  ds_f32("markov_head.dflare_norm.weight"), ctx, B, D, dspark_cfg_.rms_eps);
        k_dspark_ctx_scatter<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
            ctx, reinterpret_cast<uint16_t* const*>(d_ds_frnn_ptrs_.dev_ptr), sids_d, pos_d, B, D);
    }

    bool ensure_tree_scratch(std::string& err) {
        if (d_tree_meta_.dev_ptr) return true;
        const int D = c_.hidden;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const int rank = dspark_cfg_.markov_rank;
        const size_t nv = static_cast<size_t>(c_.n_visits);
        auto al = [&](GpuBuffer& b, size_t bytes) -> bool {
            return hip_ok(alloc_gpu_buffer(b, bytes, mem_info_.is_uma), "tree scratch", err);
        };
        if (!al(d_beam_pool_, kTreePoolMax * sizeof(BeamTreeNode)) ||
            !al(d_beam_cnt_, sizeof(int32_t)) ||
            !al(d_beam_tok_, kBeamW * sizeof(int32_t)) ||
            !al(d_beam_node_, kBeamW * sizeof(int32_t)) ||
            !al(d_beam_lp_, kBeamW * sizeof(float)) ||
            !al(d_beam_st_a_, static_cast<size_t>(kBeamW) * rank * sizeof(float)) ||
            !al(d_beam_st_b_, static_cast<size_t>(kBeamW) * rank * sizeof(float)) ||
            !al(d_beam_lat_, static_cast<size_t>(kBeamW) * rank * sizeof(float)) ||
            !al(d_beam_corr_, static_cast<size_t>(kBeamW) * D * sizeof(float)) ||
            !al(d_beam_scores_, static_cast<size_t>(kBeamW) * 512 * sizeof(float)) ||
            !al(d_tree_meta_, sizeof(TreeMetaDev)) ||
            !al(d_tree_k_, nv * kTreeMaxM * kv_dim * sizeof(half_t)) ||
            !al(d_tree_v_, nv * kTreeMaxM * kv_dim * sizeof(half_t)) ||
            !al(d_tree_rec_k_, nv * kTreeMaxM * kv_dim * sizeof(uint16_t)) ||
            !al(d_tree_rec_v_, nv * kTreeMaxM * kv_dim * sizeof(uint16_t)) ||
            !al(d_tree_rec_attn_, nv * kTreeMaxM * D * sizeof(uint16_t)) ||
            !al(d_tree_rec_mlp_, nv * kTreeMaxM * D * sizeof(uint16_t)) ||
            !al(d_tree_h_f16_, static_cast<size_t>(kTreeMaxM) * D * sizeof(half_t)) ||
            !al(d_tree_arg_v_, static_cast<size_t>(kTreeMaxM) * 256 * sizeof(float)) ||
            !al(d_tree_arg_i_, static_cast<size_t>(kTreeMaxM) * 256 * sizeof(int32_t)) ||
            !al(d_tree_arg_out_, static_cast<size_t>(kTreeMaxM) * sizeof(int32_t)) ||
            !al(d_tree_kv_descs_, nv * sizeof(KvVisitDesc))) {
            return false;
        }
        for (int k = 0; k < 4; ++k) {
            if (!al(d_tree_taps_win_[k], 8u * static_cast<size_t>(D) * sizeof(float))) return false;
        }
        return true;
    }

    // Exact 48-Visit Tree Verify + On-GPU Greedy Argmax (100% HIP-Graphable per tree size M=2..12!)
    bool forward_tree_verify(int B, int slot, uint64_t kv_gen_sum, std::string& err) {
        if (B < 1 || B > 12) return false;
        const int tprof_lvl = (std::getenv("STRATA_HIP_TREE_PROF") ? std::atoi(std::getenv("STRATA_HIP_TREE_PROF")) : 0);
        const bool tprof_detail = (tprof_lvl >= 2);
        hipGraphExec_t& g_exec = graph_tree_verify_exec_[B];
        if (!tprof_detail) {
            if (g_exec != nullptr && tgraph_slot_[B] == slot &&
                tgraph_kv_gen_[B] == kv_gen_sum && tgraph_scratch_B_[B] == scratch_B_) {
                return hip_ok(hipGraphLaunch(g_exec, stream_), "tree graph launch", err);
            }
            if (g_exec != nullptr) { (void)hipGraphExecDestroy(g_exec); g_exec = nullptr; }
            if (!hip_ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "tree capture begin", err)) return false;
        }
        hipEvent_t pev[11]{};
        double p_acc[8] = {};
        if (tprof_detail) for (int i = 0; i < 11; ++i) (void)hipEventCreate(&pev[i]);
        auto pm = [&](int idx) { if (tprof_detail) (void)hipEventRecord(pev[idx], stream_); };
        auto pf = [&]() {
            if (!tprof_detail) return;
            (void)hipEventSynchronize(pev[8]);
            for (int i = 0; i < 8; ++i) { float ms = 0; (void)hipEventElapsedTime(&ms, pev[i], pev[i+1]); p_acc[i] += ms; }
        };

        const int D = c_.hidden;
        const int q_dim = c_.n_head * c_.head_dim;
        const int kv_dim = c_.n_kv * c_.head_dim;
        const bool is_fp8 = (default_kv_dtype_ == KvDtype::FP8_E4M3);
        const TreeMetaDev* d_tm = reinterpret_cast<const TreeMetaDev*>(d_tree_meta_.dev_ptr);

        float* cur = reinterpret_cast<float*>(sc_cur_.dev_ptr);
        const auto& emb_pk = packed_trunk_.at("model.llm.embed.weight");
        k_embed_lookup<<<B, 256, 0, stream_>>>(
            emb_pk.buf.dev_ptr, emb_pk.info.fmt, static_cast<int>(emb_pk.info.rows()),
            f32_ptr("model.llm.embed_norm.weight"),
            reinterpret_cast<const int32_t*>(sc_tok_ids_.dev_ptr),
            reinterpret_cast<const int32_t*>(sc_img_idx_.dev_ptr),
            nullptr, cur, B, D, c_.rms_eps);

        auto maybe_tap = [&](int visit_id, const float* vec) {
            for (int k = 0; k < 4 && k < static_cast<int>(dspark_cfg_.target_visits.size()); ++k) {
                if (dspark_cfg_.target_visits[k] == visit_id) {
                    (void)hipMemcpyAsync(sc_taps_[k].dev_ptr, vec, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                }
            }
        };

        auto run_attnres = [&](const std::vector<const float*>& ptrs, const float* cur_p, const float* q_p,
                               int nq, float* out_p) {
            BlockPtrs12 bp{};
            const int nb = static_cast<int>(ptrs.size());
            for (int i = 0; i < nb && i < 12; ++i) bp.p[i] = ptrs[i];
            if (nq == 3) k1_attn_res_mix_multi<3><<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, D);
            else if (nq == 4) k1_attn_res_mix_multi<4><<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, D);
            else k1_attn_res_mix<<<dim3(nq, B), 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, nq, D);
        };
        auto run_attnres_norm = [&](const std::vector<const float*>& ptrs, const float* cur_p, const float* q_p,
                                    const float* nw_p, float* out_p) {
            BlockPtrs12 bp{};
            const int nb = static_cast<int>(ptrs.size());
            for (int i = 0; i < nb && i < 12; ++i) bp.p[i] = ptrs[i];
            if ((D & 3) == 0 && D <= 4096) {
                k1_attn_res_mix_norm_fused<<<B, 256, 0, stream_>>>(bp, nb, cur_p, q_p, nw_p, out_p, B, D, c_.rms_eps);
            } else {
                k1_attn_res_mix<<<dim3(1, B), 256, 0, stream_>>>(bp, nb, cur_p, q_p, out_p, B, 1, D);
                k_rms_norm_rows<<<B, 256, 0, stream_>>>(out_p, nw_p, out_p, B, D, c_.rms_eps);
            }
        };

        int n_blocks = 0, n_prelude_blocks = 0, n_bank = 0;
        std::vector<const float*> blk_ptrs, bank_ptrs;

        float* in0   = reinterpret_cast<float*>(sc_in0_.dev_ptr);
        float* in1   = reinterpret_cast<float*>(sc_in1_.dev_ptr);
        float* in2   = reinterpret_cast<float*>(sc_in2_.dev_ptr);
        float* mx3   = reinterpret_cast<float*>(sc_mx3_.dev_ptr);
        float* msgs  = reinterpret_cast<float*>(sc_msgs_.dev_ptr);
        float* q     = reinterpret_cast<float*>(sc_q_.dev_ptr);
        float* k_raw = reinterpret_cast<float*>(sc_k_raw_.dev_ptr);
        float* v_raw = reinterpret_cast<float*>(sc_v_raw_.dev_ptr);
        float* k_cv  = reinterpret_cast<float*>(sc_k_conv_.dev_ptr);
        float* v_cv  = reinterpret_cast<float*>(sc_v_conv_.dev_ptr);
        float* r_raw = reinterpret_cast<float*>(sc_r_raw_.dev_ptr);
        float* rbias = reinterpret_cast<float*>(sc_rel_bias_.dev_ptr);
        float* sinks = reinterpret_cast<float*>(sc_sinks_.dev_ptr);
        float* gatea = reinterpret_cast<float*>(sc_gate_a_.dev_ptr);
        float* k_dyn = reinterpret_cast<float*>(sc_k_dyn_.dev_ptr);
        float* v_dyn = reinterpret_cast<float*>(sc_v_dyn_.dev_ptr);
        float* aout  = reinterpret_cast<float*>(sc_attn_out_.dev_ptr);
        float* oproj = reinterpret_cast<float*>(sc_o_proj_.dev_ptr);
        float* asc   = reinterpret_cast<float*>(sc_a_sconv_.dev_ptr);
        float* mlpin = reinterpret_cast<float*>(sc_mlp_in_.dev_ptr);
        half_t* mlpf16 = reinterpret_cast<half_t*>(sc_mlp_f16_.dev_ptr);
        float* rlog  = reinterpret_cast<float*>(sc_r_logits_.dev_ptr);
        int32_t* tk_i = reinterpret_cast<int32_t*>(sc_topk_ids_.dev_ptr);
        float*   tk_w = reinterpret_cast<float*>(sc_topk_w_.dev_ptr);
        float* moeout = reinterpret_cast<float*>(sc_moe_out_.dev_ptr);
        float* msc   = reinterpret_cast<float*>(sc_m_sconv_.dev_ptr);

        int visit = 0;
        auto run_tree_visit = [&](int ip, bool has_msgs) {
            pm(0);
            maybe_tap(visit, cur);
            const std::string p = "model.llm.layers." + std::to_string(ip);
            const bool is_swa = c_.is_swa[ip] != 0;
            const int kv_slot = c_.visit_kv_slot[visit];

            float* q_in = in0;
            float* k_in = in1;
            float* v_in = in2;
            bool attn_norm_done = false;
            const bool batch3_ctrl = (has_msgs && B <= 12 && (D & 3) == 0);
            if (ip > 0) {
                const float* q_res = f32_ptr("model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
                if (is_swa) {
                    if (!has_msgs) {
                        run_attnres_norm(blk_ptrs, cur, q_res, f32_ptr(p + ".attn_norm.weight"), in0);
                        k_in = in0; v_in = in0; attn_norm_done = true;
                    } else {
                        run_attnres(blk_ptrs, cur, q_res, 1, in0);
                        if (!batch3_ctrl) {
                            (void)hipMemcpyAsync(in1, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                            (void)hipMemcpyAsync(in2, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                        }
                    }
                } else {
                    run_attnres(blk_ptrs, cur, q_res, 3, mx3);
                    k_split_attnres3<<<(B * D + 255) / 256, 256, 0, stream_>>>(mx3, in0, in1, in2, B, D);
                }
            } else {
                q_in = in0;
                if (!is_swa || has_msgs) {
                    (void)hipMemcpyAsync(in0, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    (void)hipMemcpyAsync(in1, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    (void)hipMemcpyAsync(in2, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                }
            }

            float* ctrl_inv_c = reinterpret_cast<float*>(sc_ctrl_tmp_.dev_ptr);
            float* ctrl_inv_m = ctrl_inv_c + B;
            float* ctrl_gate  = ctrl_inv_m + B;
            auto run_loop_ctrl_one = [&](float* target_in, int msg_ch, const std::string& cp) {
                k1_loop_ctrl_rms<<<B, 256, 0, stream_>>>(target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, B, D);
                const bool is_bf = trunk_bf16_.count(cp + ".down_weight") > 0;
                if (is_bf) {
                    if (B <= 4 && (D & 3) == 0) {
                        k1_loop_ctrl_down_hiocc<true><<<dim3(c_.gate_rank, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    } else {
                        k1_loop_ctrl_down<true><<<dim3((c_.gate_rank + 7) / 8, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, bf16_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    }
                    k1_loop_ctrl_up<true><<<dim3((D + 63) / 64, B), 64, 0, stream_>>>(
                        target_in, msgs, 4, msg_ch, ctrl_gate, bf16_ptr(cp + ".gain_weight"), bf16_ptr(cp + ".forget_weight"), target_in, B, D, c_.gate_rank);
                } else {
                    if (B <= 4 && (D & 3) == 0) {
                        k1_loop_ctrl_down_hiocc<false><<<dim3(c_.gate_rank, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, f32_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    } else {
                        k1_loop_ctrl_down<false><<<dim3((c_.gate_rank + 7) / 8, B), 256, 0, stream_>>>(
                            target_in, msgs, 4, msg_ch, ctrl_inv_c, ctrl_inv_m, f32_ptr(cp + ".down_weight"), ctrl_gate, B, D, c_.gate_rank);
                    }
                    k1_loop_ctrl_up<false><<<dim3((D + 63) / 64, B), 64, 0, stream_>>>(
                        target_in, msgs, 4, msg_ch, ctrl_gate, f32_ptr(cp + ".gain_weight"), f32_ptr(cp + ".forget_weight"), target_in, B, D, c_.gate_rank);
                }
            };

            if (has_msgs) {
                const char* cnames[3] = {"attn", "k", "v"};
                const std::string c0 = p + ".controllers." + cnames[0];
                const std::string c1 = p + ".controllers." + cnames[1];
                const std::string c2 = p + ".controllers." + cnames[2];
                const bool all_bf = (trunk_bf16_.count(c0 + ".down_weight") > 0 &&
                                     trunk_bf16_.count(c1 + ".down_weight") > 0 &&
                                     trunk_bf16_.count(c2 + ".down_weight") > 0);
                const bool all_f32 = (trunk_bf16_.count(c0 + ".down_weight") == 0 &&
                                      trunk_bf16_.count(c1 + ".down_weight") == 0 &&
                                      trunk_bf16_.count(c2 + ".down_weight") == 0);
                if (batch3_ctrl && (all_bf || all_f32)) {
                    FPtr3 src_p = is_swa ? FPtr3{{in0, in0, in0}} : FPtr3{{in0, in1, in2}};
                    MutFPtr3 dst_p{{in0, in1, in2}};
                    float* ic3 = reinterpret_cast<float*>(sc_ctrl_tmp_.dev_ptr);
                    float* im3 = ic3 + 3 * B;
                    float* gt3 = im3 + 3 * B;
                    k1_loop_ctrl_rms3<<<dim3(3, B), 256, 0, stream_>>>(src_p, msgs, ic3, im3, B, D);
                    if (all_bf) {
                        Ptr3 wd{{bf16_ptr(c0 + ".down_weight"), bf16_ptr(c1 + ".down_weight"), bf16_ptr(c2 + ".down_weight")}};
                        Ptr3 wg{{bf16_ptr(c0 + ".gain_weight"), bf16_ptr(c1 + ".gain_weight"), bf16_ptr(c2 + ".gain_weight")}};
                        Ptr3 wf{{bf16_ptr(c0 + ".forget_weight"), bf16_ptr(c1 + ".forget_weight"), bf16_ptr(c2 + ".forget_weight")}};
                        k1_loop_ctrl_down_hiocc3<true><<<dim3(c_.gate_rank, B, 3), 256, 0, stream_>>>(
                            src_p, msgs, ic3, im3, wd, gt3, B, D, c_.gate_rank);
                        k1_loop_ctrl_up3<true><<<dim3((D + 63) / 64, B, 3), 64, 0, stream_>>>(
                            src_p, msgs, gt3, wg, wf, dst_p, B, D, c_.gate_rank);
                    } else {
                        Ptr3 wd{{f32_ptr(c0 + ".down_weight"), f32_ptr(c1 + ".down_weight"), f32_ptr(c2 + ".down_weight")}};
                        Ptr3 wg{{f32_ptr(c0 + ".gain_weight"), f32_ptr(c1 + ".gain_weight"), f32_ptr(c2 + ".gain_weight")}};
                        Ptr3 wf{{f32_ptr(c0 + ".forget_weight"), f32_ptr(c1 + ".forget_weight"), f32_ptr(c2 + ".forget_weight")}};
                        k1_loop_ctrl_down_hiocc3<false><<<dim3(c_.gate_rank, B, 3), 256, 0, stream_>>>(
                            src_p, msgs, ic3, im3, wd, gt3, B, D, c_.gate_rank);
                        k1_loop_ctrl_up3<false><<<dim3((D + 63) / 64, B, 3), 64, 0, stream_>>>(
                            src_p, msgs, gt3, wg, wf, dst_p, B, D, c_.gate_rank);
                    }
                } else {
                    if (batch3_ctrl && is_swa) {
                        (void)hipMemcpyAsync(in1, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                        (void)hipMemcpyAsync(in2, in0, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    }
                    float* ins[3] = {in0, in1, in2};
                    for (int j = 0; j < 3; ++j) run_loop_ctrl_one(ins[j], j, p + ".controllers." + cnames[j]);
                }
            }

            if (!attn_norm_done) {
                if (has_msgs && (D & 3) == 0) {
                    MutFPtr3 io_p{{in0, in1, in2}};
                    FPtr3 nw_p = is_swa
                        ? FPtr3{{f32_ptr(p + ".attn_norm.weight"), f32_ptr(p + ".attn_norm.weight"), f32_ptr(p + ".attn_norm.weight")}}
                        : FPtr3{{f32_ptr(p + ".q_input_norm.weight"), f32_ptr(p + ".k_input_norm.weight"), f32_ptr(p + ".v_input_norm.weight")}};
                    k_rms_norm_rows3<<<dim3(3, B), 256, 0, stream_>>>(io_p, nw_p, B, D, c_.rms_eps);
                } else if (is_swa) {
                    const float* an = f32_ptr(p + ".attn_norm.weight");
                    k_rms_norm_rows<<<B, 256, 0, stream_>>>((ip == 0 && !has_msgs) ? cur : in0, an, in0, B, D, c_.rms_eps);
                    if (!has_msgs) { k_in = in0; v_in = in0; }
                } else {
                    MutFPtr3 io_p{{in0, in1, in2}};
                    FPtr3 nw_p{{f32_ptr(p + ".q_input_norm.weight"), f32_ptr(p + ".k_input_norm.weight"), f32_ptr(p + ".v_input_norm.weight")}};
                    k_rms_norm_rows3<<<dim3(3, B), 256, 0, stream_>>>(io_p, nw_p, B, D, c_.rms_eps);
                }
            }

            // Q/K/V/R/Sink/Gate projections (fused 1-launch GEMV+sink at B=1, concatenated GEMM+split for B>=2!)
            pm(1);
            const int r_dim = c_.n_head * c_.d_rel;
            bool sink_done = false;
            const bool sink_is_f32 = (trunk_bf16_.count(p + ".attn.attn_sink_proj.weight") == 0);
            if (trunk_qkvrg_[ip].dev_ptr != nullptr) {
                const uint16_t* w_cat = reinterpret_cast<const uint16_t*>(trunk_qkvrg_[ip].dev_ptr);
                float* cat_out = reinterpret_cast<float*>(sc_qkvrg_out_.dev_ptr);
                if (!has_msgs && is_swa) {
                    const int tot_n = 2 * q_dim + r_dim + 2 * kv_dim;
                    if (B == 1 && D <= 4096 && (D & 7) == 0) {
                        const int ns = sink_is_f32 ? c_.n_head : 0;
                        k_bf16_gemv1_qkvrg<<<(tot_n + ns + 7) / 8, 256, 0, stream_>>>(
                            w_cat, q_in, q, r_raw, gatea, k_raw, v_raw,
                            sink_is_f32 ? f32_ptr(p + ".attn.attn_sink_proj.weight") : nullptr,
                            sink_is_f32 ? sinks : nullptr,
                            q_dim, r_dim, kv_dim, tot_n, ns, D);
                        sink_done = sink_is_f32;
                    } else {
                        launch_bf16_opt(w_cat, q_in, nullptr, cat_out, B, tot_n, D);
                        k_split_qkvrg<<<dim3(((tot_n >> 2) + 255) / 256, B), 256, 0, stream_>>>(
                            cat_out, q, r_raw, gatea, k_raw, v_raw, B, q_dim, r_dim, kv_dim, tot_n);
                    }
                } else {
                    const int qrg_n = 2 * q_dim + r_dim;
                    if (B == 1 && D <= 4096 && (D & 7) == 0) {
                        const int ns = sink_is_f32 ? c_.n_head : 0;
                        k_bf16_gemv1_qkvrg<<<(qrg_n + ns + 7) / 8, 256, 0, stream_>>>(
                            w_cat, q_in, q, r_raw, gatea, nullptr, nullptr,
                            sink_is_f32 ? f32_ptr(p + ".attn.attn_sink_proj.weight") : nullptr,
                            sink_is_f32 ? sinks : nullptr,
                            q_dim, r_dim, kv_dim, qrg_n, ns, D);
                        sink_done = sink_is_f32;
                    } else {
                        launch_bf16_opt(w_cat, q_in, nullptr, cat_out, B, qrg_n, D);
                        k_split_qkvrg<<<dim3(((qrg_n >> 2) + 255) / 256, B), 256, 0, stream_>>>(
                            cat_out, q, r_raw, gatea, nullptr, nullptr, B, q_dim, r_dim, kv_dim, qrg_n);
                    }
                    launch_bf16_opt(w_cat + static_cast<size_t>(qrg_n) * D, k_in, nullptr, k_raw, B, kv_dim, D);
                    launch_bf16_opt(w_cat + static_cast<size_t>(qrg_n + kv_dim) * D, v_in, nullptr, v_raw, B, kv_dim, D);
                }
            } else {
                launch_trunk_linear(p + ".attn.wq_du.weight", q_in, q, B, q_dim, D);
                launch_trunk_linear(p + ".attn.wk_dv.weight", k_in, k_raw, B, kv_dim, D);
                launch_trunk_linear(p + ".attn.wv_dv.weight", v_in, v_raw, B, kv_dim, D);
                launch_trunk_linear(p + ".attn.wr_du.weight", q_in, r_raw, B, r_dim, D);
                launch_trunk_linear(p + ".attn.gate_attn.weight", q_in, gatea, B, q_dim, D);
            }
            if (!sink_done) {
                launch_trunk_linear(p + ".attn.attn_sink_proj.weight", q_in, sinks, B, c_.n_head, D);
            }
            if (!is_swa) {
                launch_trunk_linear(p + ".attn.k_sconv.weight_proj", k_in, k_dyn, B, c_.n_kv * 4, D);
                launch_trunk_linear(p + ".attn.v_sconv.weight_proj", v_in, v_dyn, B, c_.n_kv * 4, D);
            }

            // Tree ShortConv K & V (records BF16 input into d_tree_rec_k/v WITHOUT mutating hist_pool!)
            pm(2);
            const size_t kv_vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(kv_dim) * sizeof(uint16_t);
            const size_t kv_trec_st = static_cast<size_t>(kTreeMaxM) * static_cast<size_t>(kv_dim);
            k2_shortconv4_tree<<<dim3((kv_dim + 255) / 256, B), 256, 0, stream_>>>(
                k_raw, reinterpret_cast<const float*>(static_sconv_.at(p + ".k_sconv").dev_ptr),
                is_swa ? nullptr : k_dyn, d_tm,
                reinterpret_cast<const uint16_t*>(d_sconv_k_all_.dev_ptr + visit * kv_vstride),
                reinterpret_cast<uint16_t*>(d_tree_rec_k_.dev_ptr) + visit * kv_trec_st,
                k_cv, B, kv_dim, c_.n_kv);
            k2_shortconv4_tree<<<dim3((kv_dim + 255) / 256, B), 256, 0, stream_>>>(
                v_raw, reinterpret_cast<const float*>(static_sconv_.at(p + ".v_sconv").dev_ptr),
                is_swa ? nullptr : v_dyn, d_tm,
                reinterpret_cast<const uint16_t*>(d_sconv_v_all_.dev_ptr + visit * kv_vstride),
                reinterpret_cast<uint16_t*>(d_tree_rec_v_.dev_ptr) + visit * kv_trec_st,
                v_cv, B, kv_dim, c_.n_kv);

            // Super-fused Q/R/K/V Norm+Clip + Direct Tree Scratch KV Store + RelPos Bias Proj
            const int32_t* pos_dev = reinterpret_cast<const int32_t*>(sc_pos_.dev_ptr);
            const std::string q_sc_k = p + ".attn.q_l2_clip.scale";
            const std::string r_sc_k = p + ".attn.rel_l2_clip.scale";
            const int q_sc_stride = (trunk_f32_.at(q_sc_k).bytes >= static_cast<size_t>(c_.n_head) * c_.head_dim * 4) ? c_.head_dim : 0;
            const int r_sc_stride = (trunk_f32_.at(r_sc_k).bytes >= static_cast<size_t>(c_.n_head) * c_.d_rel * 4) ? c_.d_rel : 0;
            half_t* tk_v = reinterpret_cast<half_t*>(d_tree_k_.dev_ptr) + visit * kv_trec_st;
            half_t* tv_v = reinterpret_cast<half_t*>(d_tree_v_.dev_ptr) + visit * kv_trec_st;
            k2_qrkv_treestore_relpos_fused<<<dim3(2 * c_.n_head + 2 * c_.n_kv, B), 128, 0, stream_>>>(
                q, f32_ptr(q_sc_k), q_sc_stride,
                r_raw, f32_ptr(r_sc_k), r_sc_stride,
                f32_ptr(p + ".attn.rel_logits_proj.proj"), rbias,
                k_cv, v_cv, pos_dev, tk_v, tv_v,
                !is_swa, c_.qk_l2_clip, c_.rel_l2_clip, c_.rms_eps,
                B, c_.n_head, c_.n_kv, c_.head_dim, c_.d_rel, c_.rel_extent);

            // Tree Flash-SWA / Flash-GQA Attention (committed prefix + bitmask tree ancestors!)
            pm(3);
            VisitKvPool& kvp = is_swa ? swa_pools_[kv_slot] : glob_pools_[kv_slot];
            const int swa_w = is_swa ? c_.swa_win : 0;
            if (is_fp8) {
                k3_flash_swa_gqa_tree<true><<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                    q, gatea, rbias, sinks, d_tm, pos_dev,
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                    kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr,
                    reinterpret_cast<const float*>(kvp.k_scales.dev_ptr), reinterpret_cast<const float*>(kvp.v_scales.dev_ptr),
                    tk_v, tv_v, aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
            } else {
                k3_flash_swa_gqa_tree<false><<<dim3(c_.n_head, B), 128, 0, stream_>>>(
                    q, gatea, rbias, sinks, d_tm, pos_dev,
                    reinterpret_cast<const int32_t*>(kvp.d_pt.dev_ptr), kvp.max_pages_per_slot,
                    kvp.k_pool.dev_ptr, kvp.v_pool.dev_ptr, nullptr, nullptr,
                    tk_v, tv_v, aout, B, c_.n_head, c_.n_kv, swa_w, c_.rel_extent);
            }

            // O-proj + Tree post-attn ShortConv + AttnRes block boundary
            pm(4);
            launch_trunk_linear(p + ".attn.wo_ud.weight", aout, oproj, B, D, q_dim);
            const size_t h_vstride = static_cast<size_t>(max_slots_) * 3u * static_cast<size_t>(D) * sizeof(uint16_t);
            const size_t h_trec_st = static_cast<size_t>(kTreeMaxM) * static_cast<size_t>(D);
            k2_shortconv4_tree<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                oproj, reinterpret_cast<const float*>(static_sconv_.at(p + ".attn_sconv").dev_ptr),
                nullptr, d_tm,
                reinterpret_cast<const uint16_t*>(d_sconv_attn_all_.dev_ptr + visit * h_vstride),
                reinterpret_cast<uint16_t*>(d_tree_rec_attn_.dev_ptr) + visit * h_trec_st,
                asc, B, D, 0);
            if (ip % c_.block_size == 0) {
                (void)hipMemcpyAsync(sc_blocks_[n_blocks].dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                blk_ptrs.push_back(reinterpret_cast<const float*>(sc_blocks_[n_blocks].dev_ptr));
                ++n_blocks;
                (void)hipMemcpyAsync(cur, asc, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
            } else {
                k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, asc, cur, B * D);
            }

            // MoE sublayer
            if (!has_msgs) {
                run_attnres_norm(blk_ptrs, cur, f32_ptr("model.llm.attn_res.mlp_queries." + std::to_string(ip)),
                                 f32_ptr(p + ".mlp_norm.weight"), mlpin);
            } else {
                run_attnres(blk_ptrs, cur, f32_ptr("model.llm.attn_res.mlp_queries." + std::to_string(ip)), 1, mlpin);
                run_loop_ctrl_one(mlpin, 3, p + ".controllers.mlp");
                k_rms_norm_rows<<<B, 256, 0, stream_>>>(mlpin, f32_ptr(p + ".mlp_norm.weight"), mlpin, B, D, c_.rms_eps);
            }

            pm(5);
            if (B >= 8 && trunk_bf16_.count(p + ".mlp.gate.weight") > 0 &&
                mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                launch_bf16_opt(bf16_ptr(p + ".mlp.gate.weight"), mlpin, nullptr, rlog, B, c_.n_experts, D);
            } else {
                launch_f32_linear(f32_ptr(p + ".mlp.gate.weight"), mlpin, nullptr, rlog, B, c_.n_experts, D, stream_, mem_info_.gen);
            }
            const bool use_direct_moe = (B <= 6);
            const bool fused_router_f16 = ((D & 3) == 0);
            if (fused_router_f16) {
                k4_moe_router_topk_f16<<<B, 128, 0, stream_>>>(
                    rlog, f32_ptr(p + ".mlp.gate.bias"), mlpin, mlpf16, tk_i, tk_w, B, D, c_.top_k, c_.route_scale);
            } else {
                k4_moe_router_topk<<<B, 128, 0, stream_>>>(rlog, f32_ptr(p + ".mlp.gate.bias"), tk_i, tk_w, B, c_.top_k, c_.route_scale);
            }

            const GpuExpertDesc* d_exps = reinterpret_cast<const GpuExpertDesc*>(d_layer_exps_[ip].dev_ptr);
            if (use_direct_moe) {
                if (!fused_router_f16) k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                pm(6);
                k_moe_direct_w13_situ<<<dim3((c_.inter + 3) / 4, B * c_.top_k), 256, 0, stream_>>>(
                    d_exps, tk_i, mlpf16, reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr),
                    B, c_.top_k, D, c_.inter, c_.situ_beta, c_.situ_lb);
                pm(7);
                float* y_part = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                k_moe_direct_w2_par<<<dim3((D + 7) / 8, B * c_.top_k), 256, 0, stream_>>>(
                    d_exps, tk_i, reinterpret_cast<const half_t*>(sc_moe_act_.dev_ptr),
                    y_part, B, c_.top_k, D, c_.inter);
                k_moe_direct_reduce<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                    y_part, tk_w, moeout, B, c_.top_k, D);
            } else {
                // Wave-7E: WMMA Grouped-MoE for B>=7 on RDNA3/3.5/4 (zero-WMMA register grouped MoE on RDNA2!)
                int32_t* d_off = reinterpret_cast<int32_t*>(sc_exp_off_.dev_ptr);
                int32_t* d_btk = reinterpret_cast<int32_t*>(sc_bucket_tok_.dev_ptr);
                int32_t* d_s2b = reinterpret_cast<int32_t*>(sc_slotk_bidx_.dev_ptr);
                half_t*  d_act = reinterpret_cast<half_t*>(sc_moe_act_.dev_ptr);
                float*   d_by  = reinterpret_cast<float*>(sc_bucket_y_.dev_ptr);
                k_moe_build_buckets<<<1, 128, 0, stream_>>>(tk_i, d_off, d_btk, d_s2b, B, c_.top_k);
                if (!fused_router_f16) k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(mlpin, mlpf16, B * D);
                if (mem_info_.gen != AmdGen::Rdna2 && mem_info_.gen != AmdGen::Unknown) {
                    const dim3 g13((2 * c_.inter + 127) / 128, c_.n_experts, 1);
                    const dim3 g2((D + 127) / 128, c_.n_experts, 1);
                    pm(6);
                    k_moe_grouped_w13_situ_wmma<<<g13, 256, 0, stream_>>>(
                        d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    pm(7);
                    k_moe_grouped_w2_wmma<<<g2, 256, 0, stream_>>>(
                        d_exps, d_off, d_act, d_by, D, c_.inter);
                } else {
                    pm(6);
                    k_moe_grouped_w13_situ<<<dim3((c_.inter + 3) / 4, c_.n_experts), 256, 0, stream_>>>(
                        d_exps, d_off, d_btk, mlpf16, d_act, D, c_.inter, c_.situ_beta, c_.situ_lb);
                    pm(7);
                    k_moe_grouped_w2<<<dim3((D + 7) / 8, c_.n_experts), 256, 0, stream_>>>(
                        d_exps, d_off, d_act, d_by, D, c_.inter);
                }

                k_moe_reduce_buckets<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                    d_by, d_s2b, tk_w, moeout, B, c_.top_k, D);
            }

            // Tree post-MLP ShortConv + residual add
            k2_shortconv4_tree<<<dim3((D + 255) / 256, B), 256, 0, stream_>>>(
                moeout, reinterpret_cast<const float*>(static_sconv_.at(p + ".mlp_sconv").dev_ptr),
                nullptr, d_tm,
                reinterpret_cast<const uint16_t*>(d_sconv_mlp_all_.dev_ptr + visit * h_vstride),
                reinterpret_cast<uint16_t*>(d_tree_rec_mlp_.dev_ptr) + visit * h_trec_st,
                msc, B, D, 0);
            k_vec_add<<<(B * D + 255) / 256, 256, 0, stream_>>>(cur, msc, cur, B * D);
            pm(8);
            pf();
            ++visit;
        };

        for (int ip = 0; ip < c_.core_start; ++ip) run_tree_visit(ip, false);
        (void)hipMemcpyAsync(sc_prelude_.dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
        n_prelude_blocks = n_blocks;
        for (int p_idx = 0; p_idx < c_.num_passes; ++p_idx) {
            if (p_idx > 0) {
                bank_ptrs.clear();
                n_bank = 0;
                for (int k = n_prelude_blocks + 1; k < n_blocks; ++k) {
                    (void)hipMemcpyAsync(sc_bank_[n_bank].dev_ptr, sc_blocks_[k].dev_ptr, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                    bank_ptrs.push_back(reinterpret_cast<const float*>(sc_bank_[n_bank].dev_ptr));
                    ++n_bank;
                }
                (void)hipMemcpyAsync(sc_bank_out_.dev_ptr, cur, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                (void)hipMemcpyAsync(cur, sc_prelude_.dev_ptr, static_cast<size_t>(B) * D * 4, hipMemcpyDeviceToDevice, stream_);
                n_blocks = n_prelude_blocks;
                blk_ptrs.resize(n_prelude_blocks);
            }
            for (int ip = c_.core_start; ip < c_.core_end; ++ip) {
                if (p_idx > 0) {
                    run_attnres(bank_ptrs, reinterpret_cast<const float*>(sc_bank_out_.dev_ptr),
                                reinterpret_cast<const float*>(d_msg_queries_[ip].dev_ptr), 4, msgs);
                }
                run_tree_visit(ip, p_idx > 0);
            }
        }
        for (int ip = c_.core_end; ip < c_.n_phys; ++ip) run_tree_visit(ip, false);

        // Final output norm + tap + Zero-Logits-D2H Batched Unembed Argmax!
        float* h_norm = reinterpret_cast<float*>(sc_h_norm_.dev_ptr);
        run_attnres_norm(blk_ptrs, cur, f32_ptr("model.llm.attn_res.output_query"),
                         f32_ptr("model.llm.norm.weight"), h_norm);
        maybe_tap(c_.n_visits, h_norm);

        const auto& unemb_pk = packed_trunk_.at("model.llm.unembed.weight");
        const int V = c_.vocab_unpadded;
        constexpr int kArgBlks = 256;
        half_t* h_f16 = reinterpret_cast<half_t*>(d_tree_h_f16_.dev_ptr);
        pm(9);
        k_f32_to_f16<<<(B * D + 255) / 256, 256, 0, stream_>>>(h_norm, h_f16, B * D);
        if (B <= 4) {
            k_unembed_argmax_batch_stage1<4><<<kArgBlks, 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                h_norm, h_f16,
                reinterpret_cast<float*>(d_tree_arg_v_.dev_ptr),
                reinterpret_cast<int32_t*>(d_tree_arg_i_.dev_ptr),
                B, V, D);
        } else if (B <= 6) {
            k_unembed_argmax_batch_stage1<6><<<kArgBlks, 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                h_norm, h_f16,
                reinterpret_cast<float*>(d_tree_arg_v_.dev_ptr),
                reinterpret_cast<int32_t*>(d_tree_arg_i_.dev_ptr),
                B, V, D);
        } else if (B <= 8) {
            k_unembed_argmax_batch_stage1<8><<<kArgBlks, 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                h_norm, h_f16,
                reinterpret_cast<float*>(d_tree_arg_v_.dev_ptr),
                reinterpret_cast<int32_t*>(d_tree_arg_i_.dev_ptr),
                B, V, D);
        } else {
            k_unembed_argmax_batch_stage1<12><<<kArgBlks, 256, 0, stream_>>>(
                unemb_pk.buf.dev_ptr, unemb_pk.info.fmt, static_cast<int>(unemb_pk.info.rows()),
                h_norm, h_f16,
                reinterpret_cast<float*>(d_tree_arg_v_.dev_ptr),
                reinterpret_cast<int32_t*>(d_tree_arg_i_.dev_ptr),
                B, V, D);
        }
        k_unembed_argmax_batch_stage2<<<B, 256, 0, stream_>>>(
            reinterpret_cast<const float*>(d_tree_arg_v_.dev_ptr),
            reinterpret_cast<const int32_t*>(d_tree_arg_i_.dev_ptr),
            reinterpret_cast<int32_t*>(d_tree_arg_out_.dev_ptr),
            B, kArgBlks);
        pm(10);

        if (tprof_detail) {
            (void)hipEventSynchronize(pev[10]);
            double sum = 0.0;
            for (int i = 0; i < 8; ++i) sum += p_acc[i];
            float unemb_ms = 0.0f;
            (void)hipEventElapsedTime(&unemb_ms, pev[9], pev[10]);
            std::fprintf(stderr,
                "[TREE-DETAIL] B=%d sum=%.2fms unemb=%.2fms tot=%.2fms | res_ctrl=%.2f qkvrg=%.2f sconv_kv=%.2f attn=%.2f wo_post=%.2f moe_prep=%.2f moe_w13=%.2f moe_w2_post=%.2f\n",
                B, sum, unemb_ms, sum + unemb_ms,
                p_acc[0], p_acc[1], p_acc[2], p_acc[3], p_acc[4], p_acc[5], p_acc[6], p_acc[7]);
            for (int i = 0; i < 11; ++i) (void)hipEventDestroy(pev[i]);
            return true;
        }

        hipGraph_t g = nullptr;
        if (!hip_ok(hipStreamEndCapture(stream_, &g), "tree capture end", err)) return false;
        hipError_t ie = hipGraphInstantiate(&g_exec, g, nullptr, nullptr, 0);
        (void)hipGraphDestroy(g);
        if (!hip_ok(ie, "tree graph instantiate", err)) return false;
        tgraph_slot_[B] = slot;
        tgraph_kv_gen_[B] = kv_gen_sum;
        tgraph_scratch_B_[B] = scratch_B_;
        return hip_ok(hipGraphLaunch(g_exec, stream_), "tree graph launch(first)", err);
    }

    void invalidate_graph() {
        if (graph_exec_) {
            (void)hipGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        for (int b = 0; b <= 8; ++b) {
            if (graph_verify_exec_[b]) {
                (void)hipGraphExecDestroy(graph_verify_exec_[b]);
                graph_verify_exec_[b] = nullptr;
            }
            vgraph_slot_[b] = -1;
            vgraph_need_taps_[b] = false;
            vgraph_kv_gen_[b] = 0;
            vgraph_scratch_B_[b] = 0;
        }
        for (int k = 0; k < 8; ++k) {
            if (graph_dspark_exec_[k]) {
                (void)hipGraphExecDestroy(graph_dspark_exec_[k]);
                graph_dspark_exec_[k] = nullptr;
            }
            ds_graph_slot_[k] = -1;
            ds_graph_gen_[k] = 0;
        }
        if (graph_beam_draft_exec_) {
            (void)hipGraphExecDestroy(graph_beam_draft_exec_);
            graph_beam_draft_exec_ = nullptr;
        }
        beam_graph_slot_ = -1;
        beam_graph_gen_ = 0;
        for (int b = 0; b <= 12; ++b) {
            if (graph_tree_verify_exec_[b]) {
                (void)hipGraphExecDestroy(graph_tree_verify_exec_[b]);
                graph_tree_verify_exec_[b] = nullptr;
            }
            tgraph_slot_[b] = -1;
            tgraph_kv_gen_[b] = 0;
            tgraph_scratch_B_[b] = 0;
        }
        for (int k = 0; k <= 8; ++k) {
            if (graph_tree_ctx_exec_[k]) {
                (void)hipGraphExecDestroy(graph_tree_ctx_exec_[k]);
                graph_tree_ctx_exec_[k] = nullptr;
            }
            tctx_graph_gen_[k] = 0;
            tctx_graph_B_[k] = 0;
        }
        tree_kv_desc_gen_ = 0;
        graph_B_ = 0;
        graph_kv_gen_ = 0;
        graph_scratch_B_ = 0;
    }

    void cleanup() {
        invalidate_graph();
        hybrid_.release();
        blaslt_.release();
        if (peer_stream_ || ev_g1_done_ || peer_expert_gpu_arena_.dev_ptr || !peer_d_layer_exps_.empty() || peer_scratch_B_ > 0) {
            if (gpu1_id_ >= 0) (void)hipSetDevice(gpu1_id_);
            if (peer_stream_) {
                (void)hipStreamSynchronize(peer_stream_);
                (void)hipStreamDestroy(peer_stream_);
                peer_stream_ = nullptr;
            }
            if (ev_g1_done_) { (void)hipEventDestroy(ev_g1_done_); ev_g1_done_ = nullptr; }
            for (auto& b : peer_d_layer_exps_) free_gpu_buffer(b);
            peer_d_layer_exps_.clear();
            free_gpu_buffer(peer_expert_gpu_arena_);
            free_gpu_buffer(peer_sc_mlp_f16_); free_gpu_buffer(peer_sc_topk_ids_); free_gpu_buffer(peer_sc_topk_w_);
            free_gpu_buffer(peer_sc_moe_act_); free_gpu_buffer(peer_sc_moe_out_);
            free_gpu_buffer(peer_sc_exp_off_); free_gpu_buffer(peer_sc_exp_cur_);
            free_gpu_buffer(peer_sc_bucket_tok_); free_gpu_buffer(peer_sc_slotk_bidx_); free_gpu_buffer(peer_sc_bucket_y_);
            peer_scratch_B_ = 0;
            if (gpu0_id_ >= 0) (void)hipSetDevice(gpu0_id_);
        }
        if (ev_g0_to_g1_) { (void)hipEventDestroy(ev_g0_to_g1_); ev_g0_to_g1_ = nullptr; }
        free_gpu_buffer(sc_peer_moe_out_);
        if (h_ep_x_f16_) { (void)hipHostFree(h_ep_x_f16_); h_ep_x_f16_ = nullptr; }
        if (h_ep_ids_)   { (void)hipHostFree(h_ep_ids_);   h_ep_ids_ = nullptr; }
        if (h_ep_ws_)    { (void)hipHostFree(h_ep_ws_);    h_ep_ws_ = nullptr; }
        if (h_ep_y1_)    { (void)hipHostFree(h_ep_y1_);    h_ep_y1_ = nullptr; }
        peer_expert_gpu_offset_.clear();
        layer_all_ep2_.clear();
        ep2_active_ = false;
        ep2_p2p_ = false;
        if (stream_) {
            (void)hipStreamSynchronize(stream_);
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
        }
        for (auto& sl : slots_) sl.release();
        slots_.clear();
        for (auto& p : swa_pools_) p.release();
        for (auto& p : glob_pools_) p.release();
        swa_pools_.clear(); glob_pools_.clear();
        free_gpu_buffer(d_sconv_k_all_); free_gpu_buffer(d_sconv_v_all_);
        free_gpu_buffer(d_sconv_attn_all_); free_gpu_buffer(d_sconv_mlp_all_);
        for (auto& b : trunk_qkvrg_) free_gpu_buffer(b);
        trunk_qkvrg_.clear();
        for (auto& kv : trunk_bf16_) free_gpu_buffer(kv.second);
        for (auto& kv : trunk_f32_) free_gpu_buffer(kv.second);
        for (auto& kv : packed_trunk_) free_gpu_buffer(kv.second.buf);
        for (auto& kv : static_sconv_) free_gpu_buffer(kv.second);
        for (auto& b : d_msg_queries_) free_gpu_buffer(b);
        for (auto& b : dspark_qkv_) free_gpu_buffer(b);
        for (auto& b : dspark_gu_) free_gpu_buffer(b);
        dspark_qkv_.clear(); dspark_gu_.clear();
        for (auto& kv : dspark_bf16_) free_gpu_buffer(kv.second);
        for (auto& kv : dspark_f32_) free_gpu_buffer(kv.second);
        for (auto& b : d_layer_exps_) free_gpu_buffer(b);
        trunk_bf16_.clear(); trunk_f32_.clear(); packed_trunk_.clear();
        static_sconv_.clear(); d_msg_queries_.clear(); dspark_bf16_.clear(); dspark_f32_.clear();
        d_layer_exps_.clear(); h_layer_exps_.clear();
        adj_dir_.clear();
        free_gpu_buffer(expert_gpu_arena_); free_gpu_buffer(d_exp_stage_);
        free_gpu_buffer(d_exp_w13_f16_); free_gpu_buffer(d_exp_w2_f16_);
        free_gpu_buffer(d_dspark_state_); free_gpu_buffer(d_dspark_state2_); free_gpu_buffer(d_dspark_latent_);
        free_gpu_buffer(d_dspark_corr_); free_gpu_buffer(d_dspark_toks_);
        free_gpu_buffer(d_dspark_blk_v_); free_gpu_buffer(d_dspark_blk_i_);
        free_gpu_buffer(d_dspark_cand_v_); free_gpu_buffer(d_dspark_cand_i_);
        free_gpu_buffer(ds_cur_); free_gpu_buffer(ds_res_); free_gpu_buffer(ds_h_);
        free_gpu_buffer(ds_q_); free_gpu_buffer(ds_k_); free_gpu_buffer(ds_v_);
        free_gpu_buffer(ds_ao_); free_gpu_buffer(ds_mg_); free_gpu_buffer(ds_mu_);
        free_gpu_buffer(ds_qkv_); free_gpu_buffer(ds_gu_);
        free_gpu_buffer(d_beam_pool_); free_gpu_buffer(d_beam_cnt_);
        free_gpu_buffer(d_beam_tok_); free_gpu_buffer(d_beam_node_); free_gpu_buffer(d_beam_lp_);
        free_gpu_buffer(d_beam_st_a_); free_gpu_buffer(d_beam_st_b_); free_gpu_buffer(d_beam_lat_);
        free_gpu_buffer(d_beam_corr_); free_gpu_buffer(d_beam_scores_);
        free_gpu_buffer(d_tree_meta_); free_gpu_buffer(d_tree_k_); free_gpu_buffer(d_tree_v_);
        free_gpu_buffer(d_tree_rec_k_); free_gpu_buffer(d_tree_rec_v_);
        free_gpu_buffer(d_tree_rec_attn_); free_gpu_buffer(d_tree_rec_mlp_);
        free_gpu_buffer(d_tree_h_f16_); free_gpu_buffer(d_tree_arg_v_);
        free_gpu_buffer(d_tree_arg_i_); free_gpu_buffer(d_tree_arg_out_);
        free_gpu_buffer(d_tree_kv_descs_);
        for (int k = 0; k < 4; ++k) free_gpu_buffer(d_tree_taps_win_[k]);
        for (int i = 0; i < 3; ++i) { free_gpu_buffer(d_ds_k_ptrs_[i]); free_gpu_buffer(d_ds_v_ptrs_[i]); }
        free_gpu_buffer(d_ds_frnn_ptrs_);
        free_gpu_buffer(sc_slot_ids_); free_gpu_buffer(sc_pos_); free_gpu_buffer(sc_tok_ids_); free_gpu_buffer(sc_img_idx_);
        free_gpu_buffer(sc_want_idx_); free_gpu_buffer(sc_grp_slot_); free_gpu_buffer(sc_grp_off_); free_gpu_buffer(sc_grp_tok_);
        free_gpu_buffer(sc_x_bf16_); free_gpu_buffer(sc_qkvrg_out_);
        free_gpu_buffer(sc_cur_); free_gpu_buffer(sc_res_); free_gpu_buffer(sc_in0_); free_gpu_buffer(sc_in1_);
        free_gpu_buffer(sc_in2_); free_gpu_buffer(sc_mx3_); free_gpu_buffer(sc_msgs_); free_gpu_buffer(sc_ctrl_tmp_);
        free_gpu_buffer(sc_ctrl_in_); free_gpu_buffer(sc_ctrl_gain_); free_gpu_buffer(sc_ctrl_forget_);
        free_gpu_buffer(sc_q_); free_gpu_buffer(sc_k_raw_); free_gpu_buffer(sc_v_raw_); free_gpu_buffer(sc_k_conv_);
        free_gpu_buffer(sc_v_conv_); free_gpu_buffer(sc_r_raw_); free_gpu_buffer(sc_rel_bias_);
        free_gpu_buffer(sc_sinks_); free_gpu_buffer(sc_gate_a_); free_gpu_buffer(sc_k_dyn_); free_gpu_buffer(sc_v_dyn_);
        free_gpu_buffer(sc_attn_out_); free_gpu_buffer(sc_splitkv_part_); free_gpu_buffer(sc_o_proj_); free_gpu_buffer(sc_a_sconv_);
        free_gpu_buffer(sc_mlp_in_); free_gpu_buffer(sc_mlp_f16_); free_gpu_buffer(sc_r_logits_);
        free_gpu_buffer(sc_topk_ids_); free_gpu_buffer(sc_topk_w_); free_gpu_buffer(sc_moe_act_);
        free_gpu_buffer(sc_moe_gu_); free_gpu_buffer(sc_moe_ye_); free_gpu_buffer(sc_moe_out_); free_gpu_buffer(sc_m_sconv_);
        free_gpu_buffer(sc_exp_off_); free_gpu_buffer(sc_exp_cur_);
        free_gpu_buffer(sc_x_q8_); free_gpu_buffer(sc_x_d_); free_gpu_buffer(sc_act_q8_); free_gpu_buffer(sc_act_d_);
        free_gpu_buffer(sc_bucket_tok_); free_gpu_buffer(sc_slotk_bidx_);
        free_gpu_buffer(sc_bucket_x_f16_); free_gpu_buffer(sc_bucket_y_);
        free_gpu_buffer(sc_exp_tok_idx_); free_gpu_buffer(sc_exp_tok_w_); free_gpu_buffer(sc_h_norm_);
        free_gpu_buffer(sc_blk_ptrs_); free_gpu_buffer(sc_prelude_); free_gpu_buffer(sc_bank_out_); free_gpu_buffer(sc_logits_);
        for (auto& b : sc_blocks_) free_gpu_buffer(b);
        for (auto& b : sc_bank_) free_gpu_buffer(b);
        for (int k = 0; k < 4; ++k) free_gpu_buffer(sc_taps_[k]);
        sc_blocks_.clear(); sc_bank_.clear();
        scratch_B_ = 0;
        sc_moe_gu_cap_ = 0;
        expert_native_gguf_.clear();
        vision_.release();
    }

    const InkModelFile* model_ = nullptr;
    BackendOpts opts_;
    InkCfg c_;
    DsparkCfg dspark_cfg_;
    DeviceMemoryInfo mem_info_;
    DeviceMemoryInfo peer_mem_info_;
    DispatchGuard guard_;
    hipStream_t stream_ = nullptr;
    int gpu0_id_ = 0;
    int gpu1_id_ = -1;
    bool ep2_active_ = false;
    bool ep2_p2p_ = false;
    double ep_split_ = 0.5;
    hipStream_t peer_stream_ = nullptr;
    hipEvent_t ev_g0_to_g1_ = nullptr;
    hipEvent_t ev_g1_done_ = nullptr;
    GpuBuffer peer_expert_gpu_arena_;
    std::vector<size_t> peer_expert_gpu_offset_;
    std::vector<GpuBuffer> peer_d_layer_exps_;
    std::vector<bool> layer_all_ep2_;
    int peer_scratch_B_ = 0;
    GpuBuffer sc_peer_moe_out_;
    GpuBuffer peer_sc_mlp_f16_, peer_sc_topk_ids_, peer_sc_topk_w_;
    GpuBuffer peer_sc_moe_act_, peer_sc_moe_out_;
    GpuBuffer peer_sc_exp_off_, peer_sc_exp_cur_, peer_sc_bucket_tok_, peer_sc_slotk_bidx_, peer_sc_bucket_y_;
    half_t* h_ep_x_f16_ = nullptr;
    int32_t* h_ep_ids_ = nullptr;
    float* h_ep_ws_ = nullptr;
    float* h_ep_y1_ = nullptr;
    HipBlasLtRunner blaslt_;
    HipHybridController hybrid_;
    int blaslt_min_m_ = 16;
    int blaslt_moe_min_b_ = 999999;
    bool use_i8_moe_wmma_ = true;
    KvDtype default_kv_dtype_ = KvDtype::FP16;
    bool has_dspark_ = false;
    int max_slots_ = 1;
    int max_ctx_ = 32768;

    std::vector<GpuBuffer> trunk_qkvrg_;
    std::unordered_map<std::string, GpuBuffer> trunk_bf16_;
    std::unordered_map<std::string, GpuBuffer> trunk_f32_;
    std::unordered_map<std::string, GpuPackedTensor> packed_trunk_;
    std::unordered_map<std::string, GpuBuffer> static_sconv_;
    std::unordered_map<std::string, std::vector<float>> host_tmp_;
    std::vector<GpuBuffer> d_msg_queries_;

    std::vector<GpuBuffer> dspark_qkv_;
    std::vector<GpuBuffer> dspark_gu_;
    std::unordered_map<std::string, GpuBuffer> dspark_bf16_;
    std::unordered_map<std::string, GpuBuffer> dspark_f32_;
    std::vector<float> h_dspark_fuse_;
    GpuBuffer d_dspark_state_, d_dspark_state2_, d_dspark_latent_, d_dspark_corr_, d_dspark_toks_;
    GpuBuffer d_dspark_blk_v_, d_dspark_blk_i_, d_dspark_cand_v_, d_dspark_cand_i_;
    GpuBuffer ds_cur_, ds_res_, ds_h_, ds_q_, ds_k_, ds_v_, ds_ao_, ds_mg_, ds_mu_, ds_qkv_, ds_gu_;
    GpuBuffer d_ds_k_ptrs_[3], d_ds_v_ptrs_[3], d_ds_frnn_ptrs_;
    uint64_t dspark_ptr_gen_ = 0;

    GpuBuffer expert_gpu_arena_;
    GpuBuffer d_exp_stage_;
    GpuBuffer d_exp_w13_f16_, d_exp_w2_f16_;
    std::vector<size_t> expert_gpu_offset_;
    std::vector<strata::artifact::StrataExpertEntry> adj_dir_;
    std::vector<uint8_t> expert_native_gguf_;
    std::vector<std::vector<uint8_t>> expert_ram_bodies_;
    InkColdStore cold_store_;
    uint64_t cold_gen_ = 1;
    std::vector<std::pair<float, float>> expert_sf4_outer_;
    std::vector<GpuBuffer> d_layer_exps_;
    std::vector<std::vector<GpuExpertDesc>> h_layer_exps_;
    std::vector<bool> layer_all_gpu_;
    std::vector<uint8_t> layer_gu_class_;
    std::vector<uint8_t> layer_dn_class_;
    size_t vram_experts_loaded_ = 0;
    size_t ram_experts_loaded_ = 0;

    std::vector<VisitKvPool> swa_pools_;
    std::vector<VisitKvPool> glob_pools_;
    GpuBuffer d_sconv_k_all_, d_sconv_v_all_, d_sconv_attn_all_, d_sconv_mlp_all_;
    std::vector<GpuSlotMeta> slots_;
    GpuVisionState vision_;

    int scratch_B_ = 0;
    int sc_moe_gu_cap_ = 0;
    GpuBuffer sc_slot_ids_, sc_pos_, sc_tok_ids_, sc_img_idx_, sc_want_idx_;
    GpuBuffer sc_grp_slot_, sc_grp_off_, sc_grp_tok_;
    GpuBuffer sc_x_bf16_, sc_qkvrg_out_;
    GpuBuffer sc_cur_, sc_res_, sc_in0_, sc_in1_, sc_in2_, sc_mx3_, sc_msgs_, sc_ctrl_tmp_, sc_ctrl_in_, sc_ctrl_gain_, sc_ctrl_forget_;
    GpuBuffer sc_q_, sc_k_raw_, sc_v_raw_, sc_k_conv_, sc_v_conv_, sc_r_raw_, sc_rel_bias_;
    GpuBuffer sc_sinks_, sc_gate_a_, sc_k_dyn_, sc_v_dyn_, sc_attn_out_, sc_splitkv_part_, sc_o_proj_, sc_a_sconv_;
    GpuBuffer sc_mlp_in_, sc_mlp_f16_, sc_r_logits_, sc_topk_ids_, sc_topk_w_;
    GpuBuffer sc_moe_act_, sc_moe_gu_, sc_moe_ye_, sc_moe_out_, sc_m_sconv_;
    GpuBuffer sc_exp_off_, sc_exp_cur_, sc_x_q8_, sc_x_d_, sc_act_q8_, sc_act_d_;
    GpuBuffer sc_bucket_tok_, sc_slotk_bidx_, sc_bucket_x_f16_, sc_bucket_y_;
    GpuBuffer sc_exp_tok_idx_, sc_exp_tok_w_, sc_h_norm_, sc_blk_ptrs_, sc_prelude_, sc_bank_out_, sc_logits_;
    std::vector<GpuBuffer> sc_blocks_;
    std::vector<GpuBuffer> sc_bank_;
    GpuBuffer sc_taps_[4];

    hipGraphExec_t graph_exec_ = nullptr;
    int graph_B_ = 0;
    bool graph_need_taps_ = false;
    uint64_t graph_kv_gen_ = 0;
    int graph_scratch_B_ = 0;
    hipGraphExec_t graph_verify_exec_[9]{};
    int vgraph_slot_[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    bool vgraph_need_taps_[9]{};
    uint64_t vgraph_kv_gen_[9]{};
    int vgraph_scratch_B_[9]{};
    hipGraphExec_t graph_dspark_exec_[8]{};
    int ds_graph_slot_[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    uint64_t ds_graph_gen_[8]{};

    GpuBuffer d_beam_pool_, d_beam_cnt_, d_beam_tok_, d_beam_node_, d_beam_lp_;
    GpuBuffer d_beam_st_a_, d_beam_st_b_, d_beam_lat_, d_beam_corr_, d_beam_scores_;
    GpuBuffer d_tree_meta_, d_tree_k_, d_tree_v_;
    GpuBuffer d_tree_rec_k_, d_tree_rec_v_, d_tree_rec_attn_, d_tree_rec_mlp_;
    GpuBuffer d_tree_h_f16_, d_tree_arg_v_, d_tree_arg_i_, d_tree_arg_out_, d_tree_kv_descs_;
    GpuBuffer d_tree_taps_win_[4];
    uint64_t tree_kv_desc_gen_ = 0;
    hipGraphExec_t graph_beam_draft_exec_ = nullptr;
    int beam_graph_slot_ = -1;
    uint64_t beam_graph_gen_ = 0;
    hipGraphExec_t graph_tree_verify_exec_[13]{};
    int tgraph_slot_[13] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    uint64_t tgraph_kv_gen_[13]{};
    int tgraph_scratch_B_[13]{};
    hipGraphExec_t graph_tree_ctx_exec_[9]{};
    uint64_t tctx_graph_gen_[9]{};
    int tctx_graph_B_[9]{};
};

}  // namespace

std::unique_ptr<InkBackend> make_hip_backend() {
    return std::make_unique<HipBackend>();
}

}  // namespace ink2
