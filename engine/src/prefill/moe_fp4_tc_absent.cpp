// src/prefill/moe_fp4_tc_absent.cpp - stub for builds without sm_120a / sm_121a CUDA mxf4nvf4 kernels (HIP, older CUDA, non-Blackwell archs).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/prefill/moe_fp4_tc.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fp4_tc {

bool built() { return false; }
bool available() { return false; }
bool wanted() { return false; }

void quantize_act(const float*, int64_t, int64_t, void*, void*) {
    std::fprintf(stderr, "strata: fp4_tc::quantize_act called in a build without sm_120a/sm_121a kernels\n");
    std::exit(1);
}

void experts(const strata::prefill::fused::Batch&, const strata::prefill::fused::NativeGeom&, int, int64_t, const void*,
             const void*, const int32_t*, void*, float*, void*) {
    std::fprintf(stderr, "strata: fp4_tc::experts called in a build without sm_120a/sm_121a kernels\n");
    std::exit(1);
}

}  // namespace strata::prefill::fp4_tc
