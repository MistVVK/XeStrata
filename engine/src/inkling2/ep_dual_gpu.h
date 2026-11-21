// engine/src/inkling2/ep_dual_gpu.h - Dual-GPU Expert Parallelism (--gpus 2 / --ep-split / IK2_EP_SPLIT)
// + Unified Single-GPU / Hybrid Doorbell CUDA Backend for strata-inkling.
//
// MIT License
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Copyright (c) 2026 Baptiste Rajaut, merbanan, and the Strata contributors
#pragma once

#include <memory>

#include "ink_backend.h"

namespace ink2 {

#ifdef INKLING_ENABLE_EP_CUDA
// Create the unified CUDA backend supporting:
//  - Dual-GPU Expert Parallelism (--gpus 2, --ep-split / IK2_EP_SPLIT, P2P or pinned fallback, 0 cache misses on 2x16 GB!)
//  - Single-GPU Hybrid Graph-Doorbell NUMA CPU Offload (--vram-reserve-mib N --cpu-threads N)
//  - All formats: .strata v1.1 (SF4/MX8/S_K), Official HF safetensors (NVFP4/MXFP8), Community .gguf (NVFP4/Q8_0/Q4_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS/MXFP4)
//  - All KV modes: FP16, FP8_E4M3, K8V8, K8V4
std::unique_ptr<InkBackend> make_ep_cuda_backend();
#endif

}  // namespace ink2
