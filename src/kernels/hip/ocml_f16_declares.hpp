// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/hip/ocml_f16_declares.hpp - included ahead of HIP's headers where intel/llvm's clang compiles HIP
// sources (comfy-kitchen's, src/kernels/hip/): ROCm's device library functions that HIP's headers (ROCm 7.1 and 7.2)
// call in their FP16 functions, which ROCm's own clang declares and intel/llvm's (clang 23) no longer does.  Those
// inline functions are parsed, not used by the kernels built here: a use would fail at the device link.
#pragma once

#if defined(__HIP__)
typedef _Float16 strata_2f16 __attribute__((ext_vector_type(2)));
typedef short strata_2i16 __attribute__((ext_vector_type(2)));

extern "C" {
__device__ __attribute__((const)) float __ockl_fdot2(strata_2f16, strata_2f16, float, bool);
__device__ __attribute__((const)) _Float16 __ocml_fma_f16(_Float16, _Float16, _Float16);
__device__ __attribute__((const)) strata_2f16 __ocml_fma_2f16(strata_2f16, strata_2f16, strata_2f16);
__device__ __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
__device__ __attribute__((const)) strata_2f16 __ocml_sqrt_2f16(strata_2f16);
__device__ __attribute__((const)) strata_2f16 __ocml_rsqrt_2f16(strata_2f16);
__device__ __attribute__((const)) strata_2i16 __ocml_isinf_2f16(strata_2f16);
__device__ __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
__device__ __attribute__((pure)) strata_2f16 __ocml_exp_2f16(strata_2f16);
__device__ __attribute__((pure)) strata_2f16 __ocml_exp2_2f16(strata_2f16);
__device__ __attribute__((pure)) strata_2f16 __ocml_exp10_2f16(strata_2f16);
__device__ __attribute__((pure)) strata_2f16 __ocml_log_2f16(strata_2f16);
__device__ __attribute__((pure)) strata_2f16 __ocml_log2_2f16(strata_2f16);
__device__ __attribute__((pure)) strata_2f16 __ocml_log10_2f16(strata_2f16);
__device__ strata_2f16 __ocml_cos_2f16(strata_2f16);
__device__ strata_2f16 __ocml_sin_2f16(strata_2f16);
}
#endif
