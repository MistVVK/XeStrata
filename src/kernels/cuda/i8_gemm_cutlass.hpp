// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cuda/i8_gemm_cutlass.hpp - what the SYCL side (i8_gemm.cpp) and CUTLASS's kernel (i8_gemm_cutlass.cu,
// compiled as CUDA) share: the product asked for, and the launch the host side of the CUDA compile works out for it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cutlass_i8 {

struct Problem {
    const int8_t *x, *w;
    const float *sx, *sw, *bias;
    uint16_t* y;
    int64_t T, N, K, ldy;
};

/// The fewest rows a product takes the kernels at (see params): below, cuBLAS (RTX 4070, 2026-11-06: 300 x 3072 x 3072
/// in 0.077 ms against its 0.051 with the larger tiles; 1024 rows in 0.109 against 0.152 with the smaller).
constexpr int64_t kMinRows = 1024;
/// The kernels: strata_i8_gemm_bf16_<k> for k below this, and strata_i8_gemm_sizes, which writes each one's local
/// memory and parameters' sizes as the device compile has them (2 k and 2 k + 1).
constexpr int kKernels = 2;

struct Launch {
    int kernel;                        // which of the kernels
    unsigned grid[3], threads, smem;   // smem: the dynamic local memory, bytes
    size_t bytes;                      // of params
    alignas(16) unsigned char params[1024];
};

/// The launch for `p` (Y = (X . W^T) * sx[t] * sw[n] + bias[n], BF16): the kernel for its shape and its parameters;
/// false where no kernel takes the shape (T at least kMinRows, K a multiple of 16, N and ldy of 8).
bool params(const Problem& p, Launch& out);
/// What strata_i8_gemm_sizes writes, as the host compile has it.
void sizes(unsigned* out);

/// The kernel's code for each NVIDIA architecture it was built for (CMake's embedding): sm[i] the architecture
/// (89 for sm_89), image[i] its code (a cubin, or with ptx[i] PTX for that architecture and newer, NUL-ended),
/// size[i] its bytes; count of them.
extern const int count;
extern const int sm[];
extern const bool ptx[];
extern const unsigned char* const image[];
extern const size_t size[];

}  // namespace strata::kernels::cutlass_i8
