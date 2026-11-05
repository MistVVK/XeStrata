// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/f16_gpu.hpp - FP16 to and from FP32 in the image engine's and the LLMs' device code, by the
// GPU's conversion instructions (round to nearest even, as kernels::f16_from_f32 rounds, so the same bits): that
// one's bit operations were most of the instructions of the kernels that convert a value or two each (Qwen-Image
// Q6_K's weights rewritten to int8: 160 a value, compute-bound at a quarter of the memory's bandwidth; RTX 4070,
// 2026-11-05).  The host keeps kernels' functions: sycl::half's conversion there rounded some of Mage-Flow's and
// MOSS-Image's weights differently.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::f16_gpu {

#if defined(__SYCL_DEVICE_ONLY__)
inline uint16_t f16_from_f32(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
inline float f32_from_f16(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
#else
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
#endif

}  // namespace strata::kernels::f16_gpu
