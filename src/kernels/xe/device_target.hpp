// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/device_target.hpp - which device compile a kernel source is in.
//
// STRATA_DEVICE_NOT_INTEL is 1 in the device compiles for NVIDIA (__NVPTX__) and AMD (__AMDGCN__) GPUs.  The kernels
// written for Intel's matrix engines (joint_matrix with Intel's layouts and checked loads, the large register file)
// compile an empty body there: those compilers have no such code, and the run-time checks that choose them (the
// device's matrix_combinations, its sub-group sizes, the large register file mode) never launch them on such a GPU.
#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && (defined(__NVPTX__) || defined(__AMDGCN__))
#define STRATA_DEVICE_NOT_INTEL 1
#else
#define STRATA_DEVICE_NOT_INTEL 0
#endif

// The NVIDIA architecture a device compile is for (__SYCL_CUDA_ARCH__: 890 for sm_89), 0 in every other compile.  A
// matrix type the architecture's tensor cores lack (int8 before sm_72, BF16 before sm_80) compiles an empty kernel
// body there: the device does not report that combination, so the kernel is never launched on it.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && defined(__SYCL_CUDA_ARCH__)
#define STRATA_NV_ARCH __SYCL_CUDA_ARCH__
#else
#define STRATA_NV_ARCH 0
#endif

// A kernel's required sub-group size SG, as the device compile takes it: the compilers for the other GPUs accept 32
// only, and a kernel built for 16 is never launched on them (narrow_sub_group in dp4a_gemm.hpp picks 32 there).
#define STRATA_SUB_GROUP(SG) (STRATA_DEVICE_NOT_INTEL ? 32 : (SG))
