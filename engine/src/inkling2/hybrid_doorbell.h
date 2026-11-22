// engine/src/inkling2/hybrid_doorbell.h - CUDA-Graph Mapped-Memory Doorbell + NUMA-Aware CPU Expert Pool
// + Decayed-Use Background Expert Promotion + K8V8/K8V4 KV Compression for strata-inkling.
//
// MIT License
// Copyright (c) 2026 Baptiste Rajaut (PR #6520), merbanan, and the Strata contributors
//
// Architecture:
//   1. Device kernel `k_doorbell_plan_and_ring` runs INSIDE the captured CUDA decode graph (zero host
//      `cudaStreamSynchronize` per layer!): stamps resident VRAM experts, records misses as promotion
//      candidates, and writes missed expert IDs + MoE input vector `x` + sequence number into
//      per-NUMA-node mapped host `InkDoorbell` structs.
//   2. NUMA-bound AVX-512 / AVX2 worker pools spin on their local node's doorbell (`ring`), compute
//      each cold expert (supporting SF4/NVFP4, MX8/MXFP8, S_K, and native GGUF_RAW blocks!) using
//      node-local expert memory (`MPOL_BIND` + first touch), write outputs `y[j]` and `ack[node]`.
//   3. Device kernel `k_doorbell_wait_and_collect` waits on mapped `ack` words and scatters CPU expert
//      outputs into GPU `ey[k]` using non-cached `__ldcv` loads — all still inside the CUDA Graph!
//   4. At token end, `k_decay_counts` + `k_promote_commit` + `k_promote_plan` run on device to promote
//      high-frequency cold experts via a low-occupancy background copy stream (`k_fetch_bg`).
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/quant/sk_codecs.hpp"

namespace ink2::hybrid {

constexpr int kDoorbellHidden = 2304;
constexpr int kDoorbellInter  = 1152;
constexpr int kDoorbellTopK   = 8;
constexpr int kMaxNumaNodes   = 2;
constexpr int kWantMax        = 64;
constexpr int kPromoteMax     = 32;

// One mapped doorbell per NUMA node (allocated on that node & registered cudaHostRegisterMapped).
// GPU writes ring/layer/n/k/e/x; the node's CPU pool writes y/ack/us_job.
struct alignas(64) InkDoorbell {
    volatile uint32_t ring;             // monotonically increasing sequence number written by GPU
    uint32_t pad0[15];
    volatile uint32_t ack[kMaxNumaNodes]; // per-node ack sequence number written by CPU master
    volatile float us_per_job;          // EMA microseconds per expert job (for optional PCIe share)
    uint32_t pad1[13];
    int32_t layer;                      // physical layer (0..23)
    int32_t n_jobs;                     // number of CPU jobs (0..kDoorbellTopK)
    int32_t slot_k[kDoorbellTopK];      // top-k slot index (0..7)
    int32_t expert_id[kDoorbellTopK];   // expert id (0..127)
    uint32_t pad2[14];
    float x[kDoorbellHidden];           // MoE input vector
    float y[kDoorbellTopK][kDoorbellHidden]; // computed expert outputs (without router weight)
};

// Per-expert host descriptor for CPU execution (unified across .strata SF4/MX8/S_K, HF safetensors, and .gguf!)
struct HostExpertDesc {
    const uint8_t* body = nullptr;      // node-local mapped host bytes (exact InkModelFile wire body)
    uint32_t body_bytes = 0;
    uint32_t dn_offset = 0;
    uint8_t gu_code = 0;                // strata::quant::sk::SkCode
    uint8_t dn_code = 0;
    uint8_t numa_node = 0;
    uint8_t reserved = 0;
};

struct DoorbellPoolConfig {
    int n_layers = 24;
    int n_experts = 128;
    int hidden = kDoorbellHidden;
    int inter = kDoorbellInter;
    float gate_cap = 4.0f;
    float up_cap = 25.0f;
    const HostExpertDesc* experts = nullptr; // [n_layers * n_experts]
    InkDoorbell* bells[kMaxNumaNodes] = {};
    int threads_per_node = 0;
};

// NUMA & ISA helpers (Linux sysfs + MPOL_BIND on Linux x86_64; portable fallback on Windows/macOS/ARM)
int numa_node_count();
void* numa_alloc(size_t bytes, int node);
void numa_free(void* ptr, size_t bytes);
const char* doorbell_cpu_isa();

// Start / stop spinning NUMA CPU worker pools
bool start_doorbell_pool(const DoorbellPoolConfig& cfg);
void stop_doorbell_pool();

// Direct host-side single-expert evaluation (for testing / fallback / CPU selfcheck)
void eval_host_expert_f32(const HostExpertDesc& desc, const float* x, int hidden, int inter, float* y);

// ---- CUDA-side Doorbell & Decayed-Use Promotion API (implemented in hybrid_doorbell.cu) ----
#ifdef INKLING_ENABLE_EP_CUDA

struct DoorbellDeviceState {
    int n_layers = 24;
    int n_experts = 128;
    int n_nodes = 1;
    int n_slots_vram = 0;
    size_t max_expert_bytes = 0;

    // Device tables
    const uint8_t** d_vram_body = nullptr;   // [n_layers * n_experts]: VRAM body ptr if resident, else nullptr
    const uint8_t** d_host_body = nullptr;   // [n_layers * n_experts]: mapped host ptr for background promotion
    uint32_t* d_body_bytes = nullptr;        // [n_layers * n_experts]
    uint8_t*  d_home_node = nullptr;         // [n_layers * n_experts]

    // Cache & decayed-use frequency tracking
    int32_t*  d_slot_of = nullptr;           // [n_layers * n_experts] -> VRAM slot index or -1
    int32_t*  d_owner = nullptr;             // [n_slots_vram] -> expert key (l*E + e) or -1
    uint32_t* d_stamp = nullptr;             // [n_slots_vram] -> LRU stamp or 0xFFFFFFFF if in-flight
    uint32_t* d_clock = nullptr;             // [1]
    float*    d_use_cnt = nullptr;           // [n_layers * n_experts] decayed frequency counts
    int32_t*  d_want = nullptr;              // [kWantMax]
    int32_t*  d_nwant = nullptr;             // [1]
    int32_t*  d_pend_key = nullptr;          // [kPromoteMax]
    int32_t*  d_pend_slot = nullptr;         // [kPromoteMax]
    int32_t*  d_npend = nullptr;             // [1]
    const uint8_t** d_psrc = nullptr;        // [kPromoteMax]
    uint8_t** d_pdst = nullptr;              // [kPromoteMax]
    uint32_t* d_pbytes = nullptr;            // [kPromoteMax]

    // Mapped doorbells (host & device pointers)
    InkDoorbell* h_bells[kMaxNumaNodes] = {};
    InkDoorbell* d_bells[kMaxNumaNodes] = {};
    uint32_t* d_seq = nullptr;               // [1] monotonically increasing ring seq
    uint32_t* d_cur = nullptr;               // [1] current layer's seq (0 if all resident)

    // Promotion tuning (#6520 defaults: decay=0.97, gain=1.0, promote_per_tok=12, bg_blocks=4)
    float decay = 0.97f;
    float admit_gain = 1.0f;
    int promote_per_tok = 12;
    int bg_blocks = 4;
    bool use_freq_policy = true;

    // VRAM expert pool & side copy stream
    uint8_t* d_vram_pool = nullptr;
    void* side_stream = nullptr;             // cudaStream_t
    void* ev_fork = nullptr;                 // cudaEvent_t
    void* ev_join = nullptr;                 // cudaEvent_t
};

// Graph-capturable CUDA launchers (zero cudaStreamSynchronize!)
void launch_doorbell_ring(DoorbellDeviceState& st, int layer, const int32_t* d_ids, const float* d_x, void* stream);
void launch_doorbell_wait(DoorbellDeviceState& st, float* d_ey, void* stream);
void launch_promote_fork(DoorbellDeviceState& st, void* main_stream);
void launch_promote_join(DoorbellDeviceState& st, void* main_stream);

// Standalone GPU + CPU Doorbell & K8V4/K8V8 CUDA self-test (verifies zero-sync CUDA Graph doorbell roundtrip!)
bool run_doorbell_and_kv_cuda_selfcheck(int gpu_id, char* err_buf, size_t err_cap);

#endif  // INKLING_ENABLE_EP_CUDA

}  // namespace ink2::hybrid
