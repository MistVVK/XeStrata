// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// include/strata/kernels/cpu/resize_rows.hpp - the rows' pass of Pillow's 8-bit resample (the image engine's host
// resize, image/img_crop.cpp's pillow_resize_planes) in vector copies: AVX2, AVX2 with AVX-VNNI, and AVX-512 VNNI.
// Each output's total is Pillow's integer total (its fixed-point weights times the bytes, from half a unit), shifted
// down and clipped to a byte: integer sums, so every copy gives the scalar pass's bytes whatever order it adds in.

#include <cstdint>

namespace strata::kernels::cpu {

enum class ResizeRowsIsa { scalar, avx2, avxvnni, avx512 };

/// The copy this CPU runs: AVX-VNNI where cpu_avxvnni_ok(), else AVX-512 VNNI where cpu_avx512_ok() (its ymm
/// vpdpwssd needs VL as well as VNNI), else AVX2 where cpu_avx2_ok(), else none (scalar: the caller's own pass).
ResizeRowsIsa resize_rows_isa();

/// One axis's taps: output o reads n[o] bytes from lo[o], its weights k[o * stride ..] (`bits` bits below the point)
/// and their int16 halves kh = k >> 15 and kl = k & 0x7fff for the VNNI copies; `stride` a multiple of 16, the weights
/// past n[o] up to it 0; `n_max` the largest n.
struct ResizeRowsTaps {
    const int32_t* lo = nullptr;
    const int32_t* n = nullptr;
    const int32_t* k = nullptr;
    const int16_t* kh = nullptr;
    const int16_t* kl = nullptr;
    int64_t stride = 0;
    int32_t n_max = 0;
};

/// `lines` rows of `n_in` bytes (`src` readable 16 bytes past its end: a copy reads whole vectors, the weights past an
/// output's taps 0) to rows of `n_out` bytes in `dst`.  False where `isa` is scalar: the caller runs its own pass.
/// The VNNI copies hand an axis of over 64 taps (a shrink to under a tenth or so) to the AVX2 copy: their low halves'
/// sums would pass int32 there.
bool resize_rows_u8(ResizeRowsIsa isa, const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out,
                    const ResizeRowsTaps& t, int bits, uint8_t* dst);

}  // namespace strata::kernels::cpu
