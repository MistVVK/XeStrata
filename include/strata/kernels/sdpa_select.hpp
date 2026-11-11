// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/sdpa_aotriton.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels {

enum class SdpaUse { image, prompt, train };
enum class SdpaMaker { none, aotriton, cudnn, onednn };

// Only FP16 dense rows reach these libraries. Prompt's gather supports 128 and 256 only; image's causal rows
// have always used aotriton only. The prompt's cuDNN path historically ignores NO_XMX, unlike the other paths.
inline SdpaMaker sdpa_maker(sycl::queue& q, void* stream, SdpaUse use, bool causal, int64_t hd) {
    if (use == SdpaUse::prompt && hd != 128 && hd != 256) return SdpaMaker::none;
    const auto be = q.get_device().get_backend();
    if (be == sycl::backend::ext_oneapi_hip)
        return aotriton_sdpa_ready(stream) ? SdpaMaker::aotriton : SdpaMaker::none;
    if (use == SdpaUse::image && causal) return SdpaMaker::none;
    if (no_xmx() && !(use == SdpaUse::prompt && be == sycl::backend::ext_oneapi_cuda)) return SdpaMaker::none;
#if STRATA_CUDNN
    if (be == sycl::backend::ext_oneapi_cuda) return SdpaMaker::cudnn;
#endif
#if STRATA_ONEDNN
    if (be == sycl::backend::ext_oneapi_level_zero || be == sycl::backend::opencl) return SdpaMaker::onednn;
#endif
    return SdpaMaker::none;
}

}  // namespace strata::kernels
