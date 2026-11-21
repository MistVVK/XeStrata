// engine/src/inkling2/hybrid_doorbell.cu - CUDA-Graph Mapped-Memory Doorbell, NUMA-Aware AVX-512/AVX2 CPU Expert Pool,
// Decayed-Use Background Expert Promotion & K8V8/K8V4 KV Compression Kernels.
//
// MIT License
// Copyright (c) 2026 Baptiste Rajaut (PR #6520), merbanan, and the Strata contributors
#include "hybrid_doorbell.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strata/inkling2/inkling2_loop.hpp"

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#define INK_LINUX_X86_NUMA 1
#endif

namespace ink2::hybrid {

namespace {

#if defined(INK_LINUX_X86_NUMA)
std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        int a = std::atoi(part.c_str());
        int b = (dash == std::string::npos) ? a : std::atoi(part.c_str() + dash + 1);
        for (int i = a; i <= b; ++i) out.push_back(i);
    }
    return out;
}

std::string read_sysfs_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

bool numa_disabled_env() {
    const char* e = std::getenv("STRATA_NUMA_POOL");
    return e && std::string(e) == "0";
}

std::vector<int> node_physical_cores(int node) {
    std::vector<int> cpus = parse_cpu_list(read_sysfs_line(
        numa_disabled_env()
            ? std::string("/sys/devices/system/cpu/online")
            : ("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist")));
    if (cpus.empty() && node == 0) {
        int n = static_cast<int>(std::thread::hardware_concurrency());
        for (int i = 0; i < n; ++i) cpus.push_back(i);
    }
    std::vector<int> cores;
    for (int c : cpus) {
        auto sib = parse_cpu_list(read_sysfs_line(
            "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
        if (sib.empty() || sib[0] == c) cores.push_back(c);
    }
    return cores;
}

void pin_this_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

inline void cpu_pause_hint() noexcept { _mm_pause(); }
#else
inline void cpu_pause_hint() noexcept { std::this_thread::yield(); }
#endif

struct SpinBarrier {
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
    int n_threads = 1;

    void wait(int& local_sense) noexcept {
        if (n_threads <= 1) return;
        local_sense ^= 1;
        if (count.fetch_add(1, std::memory_order_acq_rel) == n_threads - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(local_sense, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) != local_sense) cpu_pause_hint();
        }
    }
};

struct alignas(64) NumaNodePool {
    int node_id = 0;
    std::vector<int> cpus;
    alignas(64) std::atomic<uint32_t> gen{0};
    SpinBarrier bar;
    int nj = 0;
    int layer = 0;
    int bj[kDoorbellTopK]{};
    int ex[kDoorbellTopK]{};
    float x_local[kDoorbellHidden]{};
    std::vector<std::thread> threads;
};

DoorbellPoolConfig g_cfg{};
std::vector<NumaNodePool*> g_pools;
std::atomic<bool> g_stop{false};
std::atomic<int> g_ready{0};

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void node_compute_jobs(NumaNodePool& pool, int tid, int& sense) {
    const int T = std::max(1, static_cast<int>(pool.threads.size()));
    const int l = pool.layer;
    const int nj = pool.nj;
    // Partition jobs across threads in the NUMA node (for nj >= 2, job-level parallelism avoids barrier overhead;
    // when nj == 1 or fewer jobs than threads, thread 0..nj-1 execute each job with fast SIMD kernels).
    for (int j = tid; j < nj; j += T) {
        const int key = l * g_cfg.n_experts + pool.ex[j];
        float* y_dst = const_cast<float*>(g_cfg.bells[pool.node_id]->y[pool.bj[j]]);
        eval_host_expert_f32(g_cfg.experts[key], pool.x_local, g_cfg.hidden, g_cfg.inter, y_dst);
    }
    pool.bar.wait(sense);
}

void pool_worker_fn(NumaNodePool* pool, int tid) {
#if defined(INK_LINUX_X86_NUMA)
    if (tid < static_cast<int>(pool->cpus.size())) pin_this_thread(pool->cpus[tid]);
#endif
    int sense = 0;
    uint32_t seen = 0;
    g_ready.fetch_add(1, std::memory_order_release);
    double last_work = now_ms();
    while (!g_stop.load(std::memory_order_relaxed)) {
        uint32_t g = pool->gen.load(std::memory_order_acquire);
        if (g == seen) {
            if (now_ms() - last_work > 20.0) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else cpu_pause_hint();
            continue;
        }
        seen = g;
        node_compute_jobs(*pool, tid, sense);
        last_work = now_ms();
    }
}

void pool_master_fn(NumaNodePool* pool) {
#if defined(INK_LINUX_X86_NUMA)
    if (!pool->cpus.empty()) pin_this_thread(pool->cpus[0]);
#endif
    InkDoorbell* bell = g_cfg.bells[pool->node_id];
    int sense = 0;
    uint32_t seen = 0;
    g_ready.fetch_add(1, std::memory_order_release);
    double last_work = now_ms();

    while (!g_stop.load(std::memory_order_relaxed)) {
        const uint32_t seq = bell->ring;
        if (seq == seen) {
            if (now_ms() - last_work > 20.0) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else cpu_pause_hint();
            continue;
        }
        seen = seq;
        const double t0 = now_ms();
        std::atomic_thread_fence(std::memory_order_acquire);

        const int n = std::min(static_cast<int>(bell->n_jobs), kDoorbellTopK);
        const int l = static_cast<int>(bell->layer);
        int nj = 0;
        for (int j = 0; j < n; ++j) {
            const int e = static_cast<int>(bell->expert_id[j]);
            const int key = l * g_cfg.n_experts + e;
            if (g_cfg.experts[key].numa_node != pool->node_id) continue;
            pool->bj[nj] = static_cast<int>(bell->slot_k[j]);
            pool->ex[nj] = e;
            ++nj;
        }
        if (nj > 0) {
            pool->nj = nj;
            pool->layer = l;
            std::memcpy(pool->x_local, const_cast<const float*>(bell->x), static_cast<size_t>(g_cfg.hidden) * sizeof(float));
            pool->gen.store(seq, std::memory_order_release);
            node_compute_jobs(*pool, 0, sense);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        bell->ack[pool->node_id] = seq;
        last_work = now_ms();
        if (nj > 0) {
            float us = static_cast<float>(1000.0 * (last_work - t0) / nj);
            float prev = bell->us_per_job;
            bell->us_per_job = (prev > 0.0f) ? (prev + (us - prev) * 0.0625f) : us;
        }
    }
}

// ============================================================================
// CUDA Device Kernels: Graph-Doorbell Ring/Wait, Decayed-Use Promotion, BG Fetch
// ============================================================================
struct BellPtrs {
    InkDoorbell* b[kMaxNumaNodes];
};

__global__ void k_doorbell_plan_and_ring(
    int layer,
    int n_experts,
    int hidden,
    int n_nodes,
    const int32_t* __restrict__ ids,
    const float* __restrict__ x,
    int32_t* __restrict__ slot_of,
    uint32_t* __restrict__ stamp,
    uint32_t* __restrict__ clock_dev,
    float* __restrict__ use_cnt,
    int32_t* __restrict__ want,
    int32_t* __restrict__ nwant,
    BellPtrs bells,
    uint32_t* __restrict__ seq_dev,
    uint32_t* __restrict__ cur_dev) {
    __shared__ int s_nj;
    if (threadIdx.x == 0) {
        uint32_t clk = *clock_dev;
        int n = 0;
        int nw = *nwant;
        int jk[kDoorbellTopK], je[kDoorbellTopK];
        for (int k = 0; k < kDoorbellTopK; ++k) {
            const int e = ids[k];
            const int key = layer * n_experts + e;
            use_cnt[key] += 1.0f;
            const int sl = slot_of[key];
            if (sl >= 0) {
                stamp[sl] = ++clk;
                continue;
            }
            jk[n] = k;
            je[n] = e;
            ++n;
            bool dup = false;
            for (int i = 0; i < nw; ++i) dup |= (want[i] == key);
            if (!dup && nw < kWantMax) want[nw++] = key;
        }
        *clock_dev = clk;
        *nwant = nw;
        for (int j = 0; j < n; ++j) {
            for (int nd = 0; nd < n_nodes; ++nd) {
                bells.b[nd]->slot_k[j] = jk[j];
                bells.b[nd]->expert_id[j] = je[j];
            }
        }
        s_nj = n;
    }
    __syncthreads();
    if (s_nj == 0) {
        if (threadIdx.x == 0) *cur_dev = 0;
        return;
    }
    for (int nd = 0; nd < n_nodes; ++nd) {
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            bells.b[nd]->x[i] = x[i];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int nd = 0; nd < n_nodes; ++nd) {
            bells.b[nd]->layer = layer;
            bells.b[nd]->n_jobs = s_nj;
        }
        __threadfence_system();
        const uint32_t s = ++(*seq_dev);
        for (int nd = 0; nd < n_nodes; ++nd) bells.b[nd]->ring = s;
        __threadfence_system();
        *cur_dev = s;
    }
}

__global__ void k_doorbell_wait_and_collect(
    BellPtrs bells,
    const uint8_t* __restrict__ home_node,
    const uint32_t* __restrict__ cur_dev,
    int n_nodes,
    int n_experts,
    int hidden,
    float* __restrict__ ey) {
    const uint32_t s = *cur_dev;
    if (s == 0u) return;

    const volatile InkDoorbell* b0 = bells.b[0];
    const volatile InkDoorbell* b1 = bells.b[(n_nodes > 1) ? 1 : 0];
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while (b0->ack[0] != s || (n_nodes > 1 && b1->ack[1] != s)) {
            if (clock64() - t0 > 12000000000ll) __trap();
        }
        ok = (static_cast<int>(blockIdx.x) < b0->n_jobs) ? 1 : 0;
    }
    __syncthreads();
    if (!ok) return;

    const int j = static_cast<int>(blockIdx.x);
    const int k = b0->slot_k[j];
    const int nd = (n_nodes > 1) ? home_node[b0->layer * n_experts + b0->expert_id[j]] : 0;
    const float4* src = reinterpret_cast<const float4*>(bells.b[nd]->y[k]);
    float4* dst = reinterpret_cast<float4*>(ey + static_cast<size_t>(k) * hidden);
    for (int i = threadIdx.x; i < hidden / 4; i += blockDim.x) {
        dst[i] = __ldcv(src + i);
    }
}

__global__ void k_decay_counts(float* __restrict__ cnt, int n, float decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] *= decay;
}

__global__ void k_promote_commit(
    int32_t* __restrict__ slot_of,
    int32_t* __restrict__ owner,
    uint32_t* __restrict__ stamp,
    uint32_t* __restrict__ clock_dev,
    const uint8_t** __restrict__ vram_body,
    uint8_t* __restrict__ vram_pool,
    size_t max_expert_bytes,
    const int32_t* __restrict__ pend_key,
    const int32_t* __restrict__ pend_slot,
    int32_t* __restrict__ npend) {
    if (threadIdx.x != 0) return;
    uint32_t clk = *clock_dev;
    const int n = *npend;
    for (int i = 0; i < n; ++i) {
        const int key = pend_key[i];
        const int sl = pend_slot[i];
        owner[sl] = key;
        slot_of[key] = sl;
        stamp[sl] = ++clk;
        vram_body[key] = vram_pool + static_cast<size_t>(sl) * max_expert_bytes;
    }
    *clock_dev = clk;
    *npend = 0;
}

__global__ void k_promote_plan(
    int32_t* __restrict__ slot_of,
    int32_t* __restrict__ owner,
    uint32_t* __restrict__ stamp,
    int n_slots,
    const uint8_t** __restrict__ vram_body,
    uint8_t* __restrict__ vram_pool,
    size_t max_expert_bytes,
    const uint8_t* const* __restrict__ host_body,
    const uint32_t* __restrict__ body_bytes,
    int32_t* __restrict__ want,
    int32_t* __restrict__ nwant,
    int32_t* __restrict__ pend_key,
    int32_t* __restrict__ pend_slot,
    int32_t* __restrict__ npend,
    const uint8_t** __restrict__ psrc,
    uint8_t** __restrict__ pdst,
    uint32_t* __restrict__ pbytes,
    int max_promote,
    const float* __restrict__ use_cnt,
    int use_freq,
    float admit_gain) {
    __shared__ unsigned long long wmin[32];
    __shared__ int n, np, stop;
    if (threadIdx.x == 0) {
        n = *nwant;
        np = 0;
        stop = (max_promote <= 0 || n_slots <= 0) ? 1 : 0;
        if (use_freq) {
            for (int i = 1; i < n; ++i) {
                int k = want[i], j = i;
                for (; j > 0 && use_cnt[want[j - 1]] < use_cnt[k]; --j) want[j] = want[j - 1];
                want[j] = k;
            }
        } else {
            if (n > max_promote) n = max_promote;
        }
    }
    __syncthreads();
    for (int m = 0; m < n && !stop; ++m) {
        unsigned long long best = ~0ull;
        for (int i = threadIdx.x; i < n_slots; i += blockDim.x) {
            const int o = owner[i];
            const uint32_t key = (stamp[i] == 0xFFFFFFFFu) ? 0xFFFFFFFFu
                                 : !use_freq ? stamp[i]
                                 : (o < 0) ? 0u
                                           : __float_as_uint(use_cnt[o]);
            best = min(best, (static_cast<unsigned long long>(key) << 32) | static_cast<uint32_t>(i));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) best = min(best, __shfl_xor_sync(0xFFFFFFFFu, best, off));
        if ((threadIdx.x & 31) == 0) wmin[threadIdx.x >> 5] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < static_cast<int>(blockDim.x / 32); ++w) best = min(best, wmin[w]);
            const int v = static_cast<int>(best & 0xFFFFFFFFu);
            const int key = want[m];
            const int old = owner[v];
            const bool admit = !use_freq || old < 0 || use_cnt[key] >= use_cnt[old] + admit_gain;
            if (slot_of[key] < 0 && stamp[v] != 0xFFFFFFFFu && admit) {
                if (old >= 0) {
                    slot_of[old] = -1;
                    vram_body[old] = nullptr;
                }
                owner[v] = -1;
                stamp[v] = 0xFFFFFFFFu;
                pend_key[np] = key;
                pend_slot[np] = v;
                psrc[np] = host_body[key];
                pdst[np] = vram_pool + static_cast<size_t>(v) * max_expert_bytes;
                pbytes[np] = body_bytes[key];
                ++np;
                if (np >= max_promote) stop = 1;
            } else if (!admit) {
                stop = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *npend = np;
        *nwant = 0;
    }
}

__global__ void k_fetch_bg(
    const uint8_t* const* __restrict__ psrc,
    uint8_t* const* __restrict__ pdst,
    const uint32_t* __restrict__ pbytes,
    const int32_t* __restrict__ npend) {
    const int n = *npend;
    for (int m = 0; m < n; ++m) {
        const uint4* src = reinterpret_cast<const uint4*>(psrc[m]);
        uint4* dst = reinterpret_cast<uint4*>(pdst[m]);
        const size_t n16 = pbytes[m] / 16u;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n16; i += stride) {
            dst[i] = src[i];
        }
    }
}

// ============================================================================
// CUDA K8V8 & K8V4 KV Quantization/Dequantization Self-Check Kernels (#6520 / @merbanan)
// ============================================================================
__global__ void k_test_k8v4_roundtrip(const float* __restrict__ k_in, const float* __restrict__ v_in,
                                      int8_t* __restrict__ k_q8, __half* __restrict__ k_sc,
                                      uint8_t* __restrict__ v_q4, __half2* __restrict__ v_sm,
                                      float* __restrict__ k_out, float* __restrict__ v_out, int kv_dim) {
    const int g = blockIdx.x;  // 1 block per 64-element group
    const int tid = threadIdx.x; // 64 threads
    const int i = g * 64 + tid;
    if (i >= kv_dim) return;

    // 1. Symmetric INT8 K
    __shared__ float s_buf[64];
    float kv = k_in[i];
    s_buf[tid] = fabsf(kv);
    __syncthreads();
    for (int st = 32; st > 0; st >>= 1) {
        if (tid < st) s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid + st]);
        __syncthreads();
    }
    __half kh = __float2half_rn(s_buf[0] / 127.0f);
    float ks = __half2float(kh);
    int8_t qk = static_cast<int8_t>(max(-127, min(127, (ks > 0.0f) ? __float2int_rn(kv / ks) : 0)));
    k_q8[i] = qk;
    if (tid == 0) k_sc[g] = kh;
    k_out[i] = static_cast<float>(qk) * ks;

    // 2. Asymmetric 4-bit V (packed 2 nibbles/byte + FP16 (scale, min))
    __shared__ float s_min[64], s_max[64];
    float vv = v_in[i];
    s_min[tid] = vv;
    s_max[tid] = vv;
    __syncthreads();
    for (int st = 32; st > 0; st >>= 1) {
        if (tid < st) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + st]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + st]);
        }
        __syncthreads();
    }
    __half vsc_h = __float2half_rn((s_max[0] - s_min[0]) / 15.0f);
    __half vmn_h = __float2half_rn(s_min[0]);
    float vsc = __half2float(vsc_h);
    float vmn = __half2float(vmn_h);
    int qv = max(0, min(15, (vsc > 0.0f) ? __float2int_rn((vv - vmn) / vsc) : 0));
    __shared__ uint8_t sq[64];
    sq[tid] = static_cast<uint8_t>(qv);
    __syncthreads();
    if (tid < 32) {
        v_q4[g * 32 + tid] = static_cast<uint8_t>((sq[2 * tid] & 0x0Fu) | ((sq[2 * tid + 1] & 0x0Fu) << 4));
    }
    if (tid == 0) v_sm[g] = __halves2half2(vsc_h, vmn_h);
    __syncthreads();
    uint8_t packed = v_q4[g * 32 + (tid >> 1)];
    uint8_t nib = (tid & 1) ? ((packed >> 4) & 0x0Fu) : (packed & 0x0Fu);
    v_out[i] = static_cast<float>(nib) * vsc + vmn;
}

}  // namespace

// ============================================================================
// Public Host API
// ============================================================================
int numa_node_count() {
#if defined(INK_LINUX_X86_NUMA)
    if (numa_disabled_env()) return 1;
    int n = 0;
    while (!read_sysfs_line("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist").empty()) ++n;
    if (const char* e = std::getenv("INK2_CPU_NODES")) n = std::min(n, std::max(1, std::atoi(e)));
    return std::min(std::max(n, 1), kMaxNumaNodes);
#else
    return 1;
#endif
}

void* numa_alloc(size_t bytes, int node) {
    bytes = (bytes + 4095u) & ~size_t(4095u);
    if (bytes == 0) bytes = 4096;
#if defined(INK_LINUX_X86_NUMA)
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
    if (!numa_disabled_env()) {
        unsigned long mask[2] = {0, 0};
        mask[node / 64] = 1UL << (node % 64);
        syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, mask, 128, 0);
        auto cores = node_physical_cores(node);
        int nt = std::max(1, std::min<int>(8, static_cast<int>(cores.size())));
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t) {
            th.emplace_back([&, t]() {
                if (!cores.empty()) pin_this_thread(cores[t % cores.size()]);
                size_t a = (bytes / nt) * t;
                size_t b = (t == nt - 1) ? bytes : (bytes / nt) * (t + 1);
                for (size_t i = a & ~size_t(4095); i < b; i += 4096) static_cast<volatile uint8_t*>(p)[i] = 0;
            });
        }
        for (auto& x : th) x.join();
    }
    return p;
#else
    (void)node;
    return std::calloc(1, bytes);
#endif
}

void numa_free(void* ptr, size_t bytes) {
    if (!ptr) return;
#if defined(INK_LINUX_X86_NUMA)
    bytes = (bytes + 4095u) & ~size_t(4095u);
    munmap(ptr, bytes ? bytes : 4096);
#else
    (void)bytes;
    std::free(ptr);
#endif
}

const char* doorbell_cpu_isa() {
    return strata::quant::sk::active_sk_isa();
}

void eval_host_expert_f32(const HostExpertDesc& desc, const float* x, int hidden, int inter, float* y) {
    strata::inkling2::eval_inkling2_expert(
        desc.body,
        static_cast<strata::quant::sk::SkCode>(desc.gu_code),
        static_cast<strata::quant::sk::SkCode>(desc.dn_code),
        desc.dn_offset,
        x,
        static_cast<size_t>(hidden),
        static_cast<size_t>(inter),
        y);
}

bool start_doorbell_pool(const DoorbellPoolConfig& cfg) {
    stop_doorbell_pool();
    g_cfg = cfg;
    const int nn = numa_node_count();
    g_stop.store(false, std::memory_order_release);
    g_ready.store(0, std::memory_order_release);

    int total_threads = 0;
    for (int nd = 0; nd < nn; ++nd) {
        if (!cfg.bells[nd]) continue;
        NumaNodePool* p = new NumaNodePool();
        p->node_id = nd;
#if defined(INK_LINUX_X86_NUMA)
        auto cores = node_physical_cores(nd);
        if (nd == 0 && cores.size() > 1) cores.erase(cores.begin());  // reserve core 0 for GPU launch loop
        int want = (cfg.threads_per_node > 0) ? std::min<int>(cfg.threads_per_node, static_cast<int>(cores.size()))
                                              : std::max(1, static_cast<int>(cores.size()));
        cores.resize(static_cast<size_t>(want));
        p->cpus = std::move(cores);
#else
        int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        int want = (cfg.threads_per_node > 0) ? std::min(cfg.threads_per_node, hw) : std::max(1, hw - 1);
        p->cpus.resize(static_cast<size_t>(want), 0);
#endif
        p->bar.n_threads = static_cast<int>(p->cpus.size());
        g_pools.push_back(p);
    }

    for (NumaNodePool* p : g_pools) {
        p->threads.emplace_back(pool_master_fn, p);
        for (int t = 1; t < static_cast<int>(p->cpus.size()); ++t) {
            p->threads.emplace_back(pool_worker_fn, p, t);
        }
        total_threads += static_cast<int>(p->cpus.size());
    }
    while (g_ready.load(std::memory_order_acquire) < total_threads) std::this_thread::yield();
    return total_threads > 0;
}

void stop_doorbell_pool() {
    if (g_pools.empty()) return;
    g_stop.store(true, std::memory_order_release);
    for (NumaNodePool* p : g_pools) {
        for (auto& t : p->threads) {
            if (t.joinable()) t.join();
        }
        delete p;
    }
    g_pools.clear();
}

// ============================================================================
// CUDA Graph Launchers
// ============================================================================
void launch_doorbell_ring(DoorbellDeviceState& st, int layer, const int32_t* d_ids, const float* d_x, void* stream) {
    BellPtrs bp{};
    for (int i = 0; i < kMaxNumaNodes; ++i) bp.b[i] = st.d_bells[i];
    k_doorbell_plan_and_ring<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        layer, st.n_experts, kDoorbellHidden, st.n_nodes, d_ids, d_x,
        st.d_slot_of, st.d_stamp, st.d_clock, st.d_use_cnt, st.d_want, st.d_nwant,
        bp, st.d_seq, st.d_cur);
}

void launch_doorbell_wait(DoorbellDeviceState& st, float* d_ey, void* stream) {
    BellPtrs bp{};
    for (int i = 0; i < kMaxNumaNodes; ++i) bp.b[i] = st.d_bells[i];
    k_doorbell_wait_and_collect<<<kDoorbellTopK, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        bp, st.d_home_node, st.d_cur, st.n_nodes, st.n_experts, kDoorbellHidden, d_ey);
}

void launch_promote_fork(DoorbellDeviceState& st, void* main_stream) {
    cudaStream_t ms = reinterpret_cast<cudaStream_t>(main_stream);
    cudaStream_t ss = reinterpret_cast<cudaStream_t>(st.side_stream);
    cudaEvent_t ef = reinterpret_cast<cudaEvent_t>(st.ev_fork);
    cudaEvent_t ej = reinterpret_cast<cudaEvent_t>(st.ev_join);
    cudaEventRecord(ef, ms);
    cudaStreamWaitEvent(ss, ef, 0);
    k_fetch_bg<<<st.bg_blocks, 256, 0, ss>>>(st.d_psrc, st.d_pdst, st.d_pbytes, st.d_npend);
    cudaEventRecord(ej, ss);
}

void launch_promote_join(DoorbellDeviceState& st, void* main_stream) {
    cudaStream_t ms = reinterpret_cast<cudaStream_t>(main_stream);
    cudaEvent_t ej = reinterpret_cast<cudaEvent_t>(st.ev_join);
    cudaStreamWaitEvent(ms, ej, 0);
    const int n_tot = st.n_layers * st.n_experts;
    if (st.use_freq_policy) {
        k_decay_counts<<<(n_tot + 255) / 256, 256, 0, ms>>>(st.d_use_cnt, n_tot, st.decay);
    }
    k_promote_commit<<<1, 32, 0, ms>>>(
        st.d_slot_of, st.d_owner, st.d_stamp, st.d_clock,
        st.d_vram_body, st.d_vram_pool, st.max_expert_bytes,
        st.d_pend_key, st.d_pend_slot, st.d_npend);
    k_promote_plan<<<1, 1024, 0, ms>>>(
        st.d_slot_of, st.d_owner, st.d_stamp, st.n_slots_vram,
        st.d_vram_body, st.d_vram_pool, st.max_expert_bytes,
        st.d_host_body, st.d_body_bytes,
        st.d_want, st.d_nwant, st.d_pend_key, st.d_pend_slot, st.d_npend,
        st.d_psrc, st.d_pdst, st.d_pbytes,
        st.promote_per_tok, st.d_use_cnt, st.use_freq_policy ? 1 : 0, st.admit_gain);
}

bool run_doorbell_and_kv_cuda_selfcheck(int gpu_id, char* err_buf, size_t err_cap) {
    auto fail = [&](const char* msg) {
        if (err_buf && err_cap) std::snprintf(err_buf, err_cap, "%s", msg);
        return false;
    };
    if (cudaSetDevice(gpu_id) != cudaSuccess) return fail("cudaSetDevice failed");

    // 1. Verify K8V4 & K8V8 CUDA kernel round-trip
    const int kv_dim = 1024;
    std::vector<float> h_k(kv_dim), h_v(kv_dim), h_ko(kv_dim), h_vo(kv_dim);
    for (int i = 0; i < kv_dim; ++i) {
        h_k[i] = std::sin(static_cast<float>(i) * 0.029f) * 3.0f;
        h_v[i] = std::cos(static_cast<float>(i) * 0.041f) * 1.5f;
    }
    float *dk = nullptr, *dv = nullptr, *dko = nullptr, *dvo = nullptr;
    int8_t* dkq = nullptr;
    __half* dksc = nullptr;
    uint8_t* dvq = nullptr;
    __half2* dvsm = nullptr;
    cudaMalloc(&dk, kv_dim * 4); cudaMalloc(&dv, kv_dim * 4);
    cudaMalloc(&dko, kv_dim * 4); cudaMalloc(&dvo, kv_dim * 4);
    cudaMalloc(&dkq, kv_dim); cudaMalloc(&dksc, (kv_dim / 64) * 2);
    cudaMalloc(&dvq, kv_dim / 2); cudaMalloc(&dvsm, (kv_dim / 64) * 4);
    cudaMemcpy(dk, h_k.data(), kv_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, h_v.data(), kv_dim * 4, cudaMemcpyHostToDevice);
    k_test_k8v4_roundtrip<<<kv_dim / 64, 64>>>(dk, dv, dkq, dksc, dvq, dvsm, dko, dvo, kv_dim);
    cudaDeviceSynchronize();
    cudaMemcpy(h_ko.data(), dko, kv_dim * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vo.data(), dvo, kv_dim * 4, cudaMemcpyDeviceToHost);
    cudaFree(dk); cudaFree(dv); cudaFree(dko); cudaFree(dvo);
    cudaFree(dkq); cudaFree(dksc); cudaFree(dvq); cudaFree(dvsm);

    for (int i = 0; i < kv_dim; ++i) {
        if (std::fabs(h_ko[i] - h_k[i]) > 0.06f) return fail("CUDA K8V4 K-INT8 accuracy mismatch");
        if (std::fabs(h_vo[i] - h_v[i]) > 0.35f) return fail("CUDA K8V4 V-INT4 accuracy mismatch");
    }

    // 2. Verify CUDA-Graph captured Mapped-Memory Doorbell round-trip (zero sync inside graph!)
    const size_t bell_bytes = (sizeof(InkDoorbell) + 4095u) & ~size_t(4095u);
    InkDoorbell* h_bell = nullptr;
    if (cudaHostAlloc(&h_bell, bell_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        return fail("cudaHostAllocMapped for InkDoorbell failed");
    }
    std::memset(h_bell, 0, sizeof(InkDoorbell));
    InkDoorbell* d_bell = nullptr;
    cudaHostGetDevicePointer(&d_bell, h_bell, 0);

    // Build 1 synthetic SF4 expert (all nibbles = 0x22 -> +1.0, scales = 0x38 -> 1.0, outer = 0.01)
    const size_t w13_p = static_cast<size_t>(2 * kDoorbellInter) * (kDoorbellHidden / 2);
    const size_t w13_s = static_cast<size_t>(2 * kDoorbellInter) * (kDoorbellHidden / 16);
    const size_t w2_p  = static_cast<size_t>(kDoorbellHidden) * (kDoorbellInter / 2);
    const size_t w2_s  = static_cast<size_t>(kDoorbellHidden) * (kDoorbellInter / 16);
    std::vector<uint8_t> sf4_body(16 + w13_p + w13_s + w2_p + w2_s, 0);
    float hdr[4] = {0.01f, 1.0f, 0.01f, 1.0f};
    std::memcpy(sf4_body.data(), hdr, 16);
    std::memset(sf4_body.data() + 16, 0x22, w13_p);
    std::memset(sf4_body.data() + 16 + w13_p, 0x38, w13_s);
    const uint32_t dn_off = static_cast<uint32_t>(16 + w13_p + w13_s);
    std::memset(sf4_body.data() + dn_off, 0x22, w2_p);
    std::memset(sf4_body.data() + dn_off + w2_p, 0x38, w2_s);

    std::vector<HostExpertDesc> h_descs(8);
    for (int e = 0; e < 8; ++e) {
        h_descs[e].body = sf4_body.data();
        h_descs[e].body_bytes = static_cast<uint32_t>(sf4_body.size());
        h_descs[e].dn_offset = dn_off;
        h_descs[e].gu_code = static_cast<uint8_t>(strata::quant::sk::SkCode::SF4);
        h_descs[e].dn_code = static_cast<uint8_t>(strata::quant::sk::SkCode::SF4);
        h_descs[e].numa_node = 0;
    }

    DoorbellPoolConfig pcfg{};
    pcfg.n_layers = 1;
    pcfg.n_experts = 8;
    pcfg.experts = h_descs.data();
    pcfg.bells[0] = h_bell;
    pcfg.threads_per_node = 2;
    if (!start_doorbell_pool(pcfg)) {
        cudaFreeHost(h_bell);
        return fail("start_doorbell_pool failed");
    }

    DoorbellDeviceState st{};
    st.n_layers = 1;
    st.n_experts = 8;
    st.n_nodes = 1;
    st.h_bells[0] = h_bell;
    st.d_bells[0] = d_bell;
    cudaMalloc(&st.d_slot_of, 8 * 4);
    std::vector<int32_t> neg1(8, -1);
    cudaMemcpy(st.d_slot_of, neg1.data(), 8 * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&st.d_stamp, 8 * 4); cudaMemset(st.d_stamp, 0, 8 * 4);
    cudaMalloc(&st.d_clock, 4); cudaMemset(st.d_clock, 0, 4);
    cudaMalloc(&st.d_use_cnt, 8 * 4); cudaMemset(st.d_use_cnt, 0, 8 * 4);
    cudaMalloc(&st.d_want, kWantMax * 4);
    cudaMalloc(&st.d_nwant, 4); cudaMemset(st.d_nwant, 0, 4);
    cudaMalloc(&st.d_home_node, 8); cudaMemset(st.d_home_node, 0, 8);
    cudaMalloc(&st.d_seq, 4); cudaMemset(st.d_seq, 0, 4);
    cudaMalloc(&st.d_cur, 4); cudaMemset(st.d_cur, 0, 4);

    int32_t h_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> h_x(kDoorbellHidden, 0.25f), h_ey(8 * kDoorbellHidden, 0.0f), h_ref(kDoorbellHidden, 0.0f);
    eval_host_expert_f32(h_descs[0], h_x.data(), kDoorbellHidden, kDoorbellInter, h_ref.data());

    int32_t* d_ids = nullptr;
    float *d_x = nullptr, *d_ey = nullptr;
    cudaMalloc(&d_ids, 8 * 4); cudaMemcpy(d_ids, h_ids, 8 * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, kDoorbellHidden * 4); cudaMemcpy(d_x, h_x.data(), kDoorbellHidden * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ey, 8 * kDoorbellHidden * 4); cudaMemset(d_ey, 0, 8 * kDoorbellHidden * 4);

    // Capture ring + wait into a CUDA Graph and replay it!
    cudaStream_t cap_stream = nullptr;
    cudaStreamCreate(&cap_stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal);
    launch_doorbell_ring(st, 0, d_ids, d_x, cap_stream);
    launch_doorbell_wait(st, d_ey, cap_stream);
    cudaStreamEndCapture(cap_stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, cap_stream);
    cudaStreamSynchronize(cap_stream);

    cudaMemcpy(h_ey.data(), d_ey, 8 * kDoorbellHidden * 4, cudaMemcpyDeviceToHost);

    stop_doorbell_pool();
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cap_stream);
    cudaFree(d_ids); cudaFree(d_x); cudaFree(d_ey);
    cudaFree(st.d_slot_of); cudaFree(st.d_stamp); cudaFree(st.d_clock);
    cudaFree(st.d_use_cnt); cudaFree(st.d_want); cudaFree(st.d_nwant);
    cudaFree(st.d_home_node); cudaFree(st.d_seq); cudaFree(st.d_cur);
    cudaFreeHost(h_bell);

    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < kDoorbellHidden; ++i) {
            if (std::fabs(h_ey[static_cast<size_t>(k) * kDoorbellHidden + i] - h_ref[i]) > 1e-5f) {
                return fail("CUDA-Graph mapped-memory doorbell output mismatch");
            }
        }
    }
    return true;
}

}  // namespace ink2::hybrid
