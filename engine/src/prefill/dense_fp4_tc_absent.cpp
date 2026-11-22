// src/prefill/dense_fp4_tc_absent.cpp - stub when sm_120a / sm_121a mxf4nvf4 kernels are not compiled.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/prefill/dense_fp4_tc.hpp"

namespace strata::prefill::dense_fp4_tc {

bool built() { return false; }
bool available() { return false; }
bool enabled() { return false; }
bool gemm_f16_nvfp4(const uint16_t*, const void*, float*, int64_t, int64_t, int64_t, int64_t, float, void*) {
    return false;
}

}  // namespace strata::prefill::dense_fp4_tc
