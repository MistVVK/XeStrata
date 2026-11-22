// src/prefill/moe_fp8_mxf_tc_absent.cpp - stub for builds without sm_120a / sm_121a mxf8f6f4 kernels.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/prefill/moe_fp8_mxf_tc.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fp8_mxf_tc {

bool built() { return false; }
bool available() { return false; }
bool wanted(int, int) { return false; }

void quantize_act(const float*, int64_t, int64_t, void*, void*) {
    std::fprintf(stderr, "fp8_mxf_tc::quantize_act: sm_120a mxf8f6f4 kernels not built\n");
    std::exit(1);
}

void experts(const strata::prefill::fused::Batch&, const strata::prefill::fused::NativeGeom&, int, int64_t,
             const void*, const void*, const int32_t*, void*, float*, void*) {
    std::fprintf(stderr, "fp8_mxf_tc::experts: sm_120a mxf8f6f4 kernels not built\n");
    std::exit(1);
}

}  // namespace strata::prefill::fp8_mxf_tc
