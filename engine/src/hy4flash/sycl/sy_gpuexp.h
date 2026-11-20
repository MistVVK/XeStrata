// sy_gpuexp.h - routed experts of a MoE layer on the Arc GPU for PREFILL chunks.
// Upgraded for 5x-10x faster prefill:
//   * All-subgroup active adaptive tiling (zero idle XMX engines for small/medium ntok!)
//   * BK=64 cooperative vectorized dequant (8 KiB SLM -> 4x occupancy) + optional DPAS INT8 XMX (HY4S_GX_I8=1)
//   * Double-buffered planning arrays eliminating cross-layer last_ev stalls
//   * Direct VRAM-slab execution for SyVramExpertCache hits (0 host copy, 0 PCIe H2D!) overlapped with PCIe streaming of host misses
//   * Supports BOTH legacy hy4s::CpuExperts (sy_gpuexp_test) AND unified hy4::CpuExperts (production sy_main)
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "hy4_cpu_experts.h"
#include "hy4s_cpuexp.h"
#include "sy_common.h"
#include "sy_vram_cache.h"

namespace hy4s {

struct GpuExpertStats {
    double seconds = 0;
    double t_plan = 0;
    double t_copy_host = 0;
    double t_wait_host = 0;
    double t_wait_prev = 0;
    uint64_t calls = 0, groups = 0, experts = 0, pairs = 0, bytes = 0, vram_experts = 0;
    double gpu_gemm1_ms = 0, gpu_gemm2_ms = 0, gpu_other_ms = 0, gpu_h2d_ms = 0;
};

class GpuExperts {
public:
    GpuExperts();
    ~GpuExperts();
    GpuExperts(const GpuExperts&) = delete;
    GpuExperts& operator=(const GpuExperts&) = delete;

    // Legacy init (used by sy_gpuexp_test)
    bool init(const CpuExperts& cx, size_t stage_bytes, int max_tokens, std::string& err, int copy_threads = 6);

    // Unified init (used by sy_main / sy_serve with run/hy4_cpu_experts.cpp + optional SyVramExpertCache)
    bool init_unified(const std::string& strata_path, const hy4::CpuExperts* ucx, const SyVramExpertCache* vcache,
                      size_t stage_bytes, int max_tokens, std::string& err, int copy_threads = 6);

    void set_vram_cache(const SyVramExpertCache* vcache);
    void set_profile_queue(sycl::queue* compute);

    void run(int moe_layer, int T, const float* z_dev, const int32_t* idx_host, const float* w_host, float* zout_dev, const uint8_t* only = nullptr);

    void wait_all();
    size_t device_bytes() const;
    const GpuExpertStats& stats() const;
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> im_;
};

}  // namespace hy4s
