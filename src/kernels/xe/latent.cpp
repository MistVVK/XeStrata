// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/kernels/xe/latent.cpp - the latent MoE's own kernels.  See the header.  Written after edge0-strata's
// src/kernels/cuda/latent.cu (the plan rows and the U product) and iq_kernels.cu's lat_colscale_rows.
//
// The U product's block dots are the integer chains of the engine's other Q8_0 / Q4_0 products (native_mmvq.cpp's
// small_q8_dot, iq_kernels.cpp's vec_dot_q4_0_q8_1): the 32-value integer sum is exact, the scales multiply it as
// (dx * dw) * sumi a block, and Q4_0's codes are centred (q - 8) before the dot, the form ggml-cpu's dot uses.
// The dp4a is the device's own (cuda_intrinsics.hpp: SPIR-V's integer dot product, PTX's dp4a, RDNA's sudot4, a byte
// loop where none exists), never a maker's name.
#include "strata/kernels/latent.hpp"

#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_latent
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/kernels/rows_cols.hpp"
#include "cuda_intrinsics.hpp"
#include "iq_bits.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {

constexpr int kWarp = 32;
constexpr int kSubGroups = 8;   // sub-groups a work-group (the native expert kernels' work-group)

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

using xe::dp4a;
using iq_bits::get_int_b2;
using iq_bits::get_int_b4;

inline float low_half(const ggml_half2& ds) { return (float) ds[0]; }

// Q4_0's centred codes (iq_kernels.cpp's q4_0_centred): per byte 0..15 -> (q - 8) as a signed byte, unsigned
// arithmetic so the sign extension cannot overflow a signed int.
inline int q4_centred(int v) {
    const uint32_t b = (uint32_t) v ^ 0x08080808u;
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));
}

// One 32-value block's dot: the q8_1 activation block (36 B: fp16 d, fp16 sum, 32 int8) against a Q8_0 weight
// block (34 B) or a Q4_0 one (18 B, element j in byte j's low nibble, j+16 in its high).  The weight block is only
// two-byte aligned inside its row (34 / 18 B strides), so its words go through get_int_b2.
template<int FMT>
inline float lat_block_dot(const uint8_t* wb, const block_q8_1* xb) {
    int sumi = 0;
    if constexpr (FMT == 4) {
        for (int j = 0; j < 4; ++j) {
            const int v = get_int_b2(wb + 2, j);
            sumi = dp4a(q4_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(xb->qs, j), sumi);
            sumi = dp4a(q4_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(xb->qs, j + 4), sumi);
        }
        return (float) reinterpret_cast<const block_q4_0*>(wb)->d * low_half(xb->ds) * (float) sumi;
    } else {
        for (int j = 0; j < 8; ++j) sumi = dp4a(get_int_b2(wb + 2, j), get_int_b4(xb->qs, j), sumi);
        return (float) reinterpret_cast<const block_q8_0*>(wb)->d * low_half(xb->ds) * (float) sumi;
    }
}

// One sub-group an output row (entry r, hidden h): the lanes stride the row's ELL/32 blocks, the sub-group sums
// their partial products.  The row's group comes from the entry's expert label; the weights of the group's H rows
// are read where they are (no gather).
template<int FMT>
sycl::event launch_u(sycl::queue& q, const uint8_t* u, const uint8_t* ylq, const int32_t* ids, const int32_t* labels,
                     int64_t E, int64_t cap_entries, int ELL, int H, float* out) {
    const int nb = ELL / 32;
    const size_t wrow = (size_t) nb * (FMT == 4 ? 18 : 34);
    const int xgroups = (H + kSubGroups - 1) / kSubGroups;
    auto body = [=](int64_t r, int hg, const sycl::sub_group& sg, int lane) {
        const int h = hg * kSubGroups + (int) sg.get_group_linear_id();
        if (h >= H) return;                                          // the whole sub-group
        const int32_t e = ids[r];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        const uint8_t* w = u + ((size_t) g * (size_t) H + (size_t) h) * wrow;
        const block_q8_1* x = (const block_q8_1*) (ylq + (size_t) r * (size_t) nb * 36);
        float acc = 0.0f;
        for (int b = lane; b < nb; b += kWarp) acc += lat_block_dot<FMT>(w + (size_t) b * (FMT == 4 ? 18 : 34), x + b);
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) out[(size_t) r * (size_t) H + h] = acc;
    };
    constexpr size_t wg = kSubGroups * kWarp;
    if (cap_entries <= kMaxFirstDim)
        return q.parallel_for(sycl::nd_range<2>({(size_t) cap_entries, (size_t) xgroups * wg}, {1, wg}),
                              [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            body((int64_t) it.get_group(0), (int) it.get_group(1), it.get_sub_group(), (int) it.get_local_linear_id() % kWarp);
        });
    // past the first dimension's limit: one dimension, the entry and the row group out of the group index
    return q.parallel_for(sycl::nd_range<1>((size_t) cap_entries * (size_t) xgroups * wg, wg),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const int64_t gi = (int64_t) it.get_group_linear_id();
        body(gi / xgroups, (int) (gi % xgroups), it.get_sub_group(), (int) it.get_local_linear_id() % kWarp);
    });
}

}  // namespace

void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels, int64_t E, int64_t base_token, int G, int K, void* stream) {
    if (tok == nullptr || dst == nullptr || counts == nullptr || ids == nullptr || labels == nullptr || K <= 0 || G <= 0)
        throw core::DeviceError("latent_plan_rows: a null pointer or a bad geometry");
    auto& q = queue_for(stream);
    // the entry count is the device's (counts[1]): one work-group walks it, as edge0-strata's single block
    constexpr size_t wg = 256;
    auto e = q.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
        const int nj = counts[1];
        for (int j = (int) it.get_global_linear_id(); j < nj; j += (int) wg) {
            const int d = dst[j];
            const int32_t ee = ids[d];
            const int g = (ee >= 0 && ee < E) ? labels[ee] : 0;
            tok[j] = (int32_t) ((base_token + d / K) * G + g);
        }
    });
    sync_if_needed(stream, e, "latent_plan_rows");
}

void latent_colscale_rows(float* x, int64_t rows, int width, const float* inv, void* stream) {
    if (rows <= 0 || inv == nullptr) return;
    if (x == nullptr || width <= 0) throw core::DeviceError("latent_colscale_rows: a null pointer or a bad width");
    auto& q = queue_for(stream);
    auto e = for_rows_cols(q, rows, width, [=](int64_t r, int64_t c) { x[r * width + c] *= inv[c]; });
    sync_if_needed(stream, e, "latent_colscale_rows");
}

void latent_u_grouped(const uint8_t* u, const uint8_t* ylq, const int32_t* ids, const int32_t* labels,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt) {
    if (cap_entries <= 0) return;
    if (u == nullptr || ylq == nullptr || ids == nullptr || labels == nullptr || out == nullptr || E <= 0 || H <= 0 ||
        ELL <= 0 || ELL % 32 != 0 || (fmt != 4 && fmt != 8))
        throw core::DeviceError("latent_u_grouped: a null pointer or a bad geometry");
    auto& q = queue_for(stream);
    const auto e = fmt == 4 ? launch_u<4>(q, u, ylq, ids, labels, E, cap_entries, ELL, H, out)
                            : launch_u<8>(q, u, ylq, ids, labels, E, cap_entries, ELL, H, out);
    sync_if_needed(stream, e, "latent_u_grouped");
}

}  // namespace strata::kernels