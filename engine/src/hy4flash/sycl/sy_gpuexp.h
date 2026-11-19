// sy_gpuexp.h - routed experts of a MoE layer on the Arc GPU for PREFILL chunks (lane HY4-SYCL, fork gpuexp).
// The expert bodies (Section 4, SQ-32 codes S1.25 / S2_K / S3_K) are streamed from the mmap'd page cache (CpuExperts) through a pinned host ring and a device ring
// (two queues, event-chained, so the host copy + PCIe transfer of group g+1 overlap the GPU compute of group g) and evaluated with XMX GEMMs (joint_matrix 8x8x16 fp16 -> fp32):
//   GEMM1  gu = W_gu z (gate = rows 0..1535, up = rows 1536..3071), epilogue h = min(silu(g),10) * clamp(u,-10,10) / sq_xd_dn  -> fp16
//   GEMM2  ye = W_dn h  -> fp32 per (token, slot) pair
//   reduce zout[t] = sum_k w * ye, experts in increasing id order (same order as CpuExperts / the oracle).
// Same math as CpuExperts::run act_mode 0 except fp16 weights/activations on the matrix engine (expected rel L2 ~2e-4).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "hy4s_cpuexp.h"
#include "sy_common.h"

namespace hy4s {

struct GpuExpertStats {
    double seconds = 0;       // host wall time inside run() (the GPU work is asynchronous: wait_all() before reading a time that must include it)
    double t_plan = 0;        // host planning (counting sort, reduce plan, uploads)
    double t_copy_host = 0;   // host memcpy of expert bodies into the pinned ring (summed over groups)
    double t_wait_host = 0;   // host waiting for a pinned slot to become free
    double t_wait_prev = 0;   // host waiting for the previous run() to finish on the GPU (planning arrays are reused); included in t_plan
    uint64_t calls = 0, groups = 0, experts = 0, pairs = 0, bytes = 0;
    // filled by wait_all() when profiling is on (set_profile_queue was used): device time in ms
    double gpu_gemm1_ms = 0, gpu_gemm2_ms = 0, gpu_other_ms = 0, gpu_h2d_ms = 0;
};

class GpuExperts {
public:
    GpuExperts();
    ~GpuExperts();
    GpuExperts(const GpuExperts&) = delete;
    GpuExperts& operator=(const GpuExperts&) = delete;

    // cx: an open CpuExperts (expert directory + mapping of Section 4). stage_bytes: total pinned host ring (3 slots); the device ring is 2 slots of stage_bytes/3.
    // max_tokens: largest T per run() call (larger calls are split). Fails when an expert uses a code other than S1.25 / S2_K / S3_K.
    bool init(const CpuExperts& cx, size_t stage_bytes, int max_tokens, std::string& err, int copy_threads = 6);
    // Optional: an in-order profiling queue of the same context to use instead of Q() for all compute kernels (gives device times in the stats).
    void set_profile_queue(sycl::queue* compute);

    // z_dev: fp32 [T][1536] (device); idx_host int32 [T][16], w_host float [T][16]; zout_dev fp32 [T][1536] (device, assigned).
    // only: optional [512] mask: experts with only[e] == 0 are skipped (their contribution is NOT added).
    // Asynchronous on the compute queue (Q()): later kernels on Q() see zout_dev in order. Host buffers idx/w are not read after return.
    void run(int moe_layer, int T, const float* z_dev, const int32_t* idx_host, const float* w_host, float* zout_dev, const uint8_t* only = nullptr);

    void wait_all();  // drains both queues and collects the profile numbers
    size_t device_bytes() const;  // exact device allocation of this object
    const GpuExpertStats& stats() const;
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> im_;
};

}  // namespace hy4s
