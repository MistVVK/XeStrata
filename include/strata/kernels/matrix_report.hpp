// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/matrix_report.hpp - the matrix forms the kernels choose their paths from.
//
// A path asks for the form its kernels are written in (has_form), never for the GPU's maker, architecture or engine
// name: the forms are worked out here, from what the device reports, and where it reports nothing (RDNA's WMMA) from
// the one place that reads its architecture (wmma_gfx).
//
// The device's matrix_combinations, less the input types a switch takes away, so that a GPU whose matrix engines lack
// them (NVIDIA's before sm_80 have no BF16, before sm_72 no int8) can be imitated on one that has them:
// STRATA_NO_BF16_MMA=1 drops the BF16 combinations, STRATA_NO_INT8_MMA=1 the int8 ones.  STRATA_NO_XMX=1 (no matrix
// engines at all) is no_xmx().
#pragma once

#include <sycl/sycl.hpp>

#include <vector>

namespace strata::kernels {

std::vector<sycl::ext::oneapi::experimental::matrix::combination> matrix_combinations(const sycl::device& d);

/// Whether STRATA_NO_BF16_MMA=1 is set (the products' BF16 library path takes it as well).
bool no_bf16_mma();

/// Whether STRATA_NO_XMX=1 is set: every path that chooses matrix engines takes the one without (a small card's, to
/// check it on a large one).
bool no_xmx();

/// The matrix forms XeStrata's kernels are written in, as the paths ask for them.
enum class MatrixForm {
    xmx_f16,          ///< FP16 8 x 16 x 16 tiles on 16 lanes with the large register file: xmx_gemm's own kernels
    f16_grouped,      ///< FP16 products of experts' groups on matrix engines: xmx_f16, or mma_gemm's tiles
    f16_from_blocks,  ///< f16_grouped with the weights dequantized from GGUF blocks in the tile (iq_gemm_grouped_f16,
                      ///< written on xmx_f16)
    wmma,             ///< FP16 and int8 16 x 16 x 16 tiles on 32 lanes as instructions written out (amd_wmma.hpp)
};

/// Whether device `d` runs kernels of form `f` (STRATA_NO_XMX=1: none).  All but wmma are worked out on the compute
/// queue, so `d` must be its device.
bool has_form(const sycl::device& d, MatrixForm f);
/// has_form on the compute device.
bool has_form(MatrixForm f);

/// The form's name, for reports: never a path's choice.
const char* form_name(MatrixForm f);

/// The WMMA instructions' generation the device has: 11 (GFX11, RDNA3), 12 (GFX12, RDNA4, its own operand layouts) or
/// 0.  The HIP runtime reports no matrix combination for RDNA, so this is the one place its architecture is read.
int wmma_gfx(const sycl::device& d);

}  // namespace strata::kernels
