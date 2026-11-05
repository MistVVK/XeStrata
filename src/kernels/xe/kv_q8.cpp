// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/kv_q8.cpp - the Xe port of Strata's src/kernels/cuda/kv_q8.cu: INT8 KV storage with an fp16 scale per 64 values.
//
// One work-group of 64 (two sub-groups of 32) per (head, group, K or V), as in the CUDA kernel.  Each value is
// quantized against the STORED (fp16-rounded) scale, and __float2int_rn's round-half-to-even is sycl::rint's.
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_kv_q8
#include "strata/kernels/device_arch_tag.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0)
        throw core::DeviceError(std::string("kv_q8: ") + what + ": head_dim " + std::to_string(s.head_dim) +
                                " must be a multiple of " + std::to_string(KV_Q8_GROUP));
}

}  // namespace

void kv_append_cells(const KvAppend& a, const QsaShapes& s, void* stream) {
    validate(s, "kv_append_cells");
    if (a.n_tok < 1) return;
    const bool f16 = a.k_pool != nullptr;
    if (!f16 && a.k_q == nullptr) throw core::DeviceError("kv_append_cells: no pools");
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int groups = head_dim / KV_Q8_GROUP;
    const int planes = !f16 && a.k_q == a.v_q && a.k == a.v ? 1 : 2;
    const int64_t cur_stride = a.cur_stride > 0 ? a.cur_stride : (int64_t) kv_heads * head_dim;
    const KvHostPools host = a.host ? *a.host : KvHostPools{}, stage = a.stage ? *a.stage : KvHostPools{};
    // upstream 5a0df898 (its sycl/ port): the host copy and the staging pool are written 4 bytes a lane, 64 contiguous
    // bytes a group, instead of one 1- or 2-byte store a lane, which ran at ~30 MB/s into pinned host memory on a B70
    // (2 s of a 4K prompt there with --kv-resident).  STRATA_KV_WIDE_HOST=0: the narrow stores.
    static const bool wide_env = [] {
        const char* v = std::getenv("STRATA_KV_WIDE_HOST");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool wide = wide_env && (f16 ? host.k_pool != nullptr || stage.k_pool != nullptr
                                       : host.k_q != nullptr || stage.k_q != nullptr);
    const KvAppend p = a;
    const auto e = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> warp_max(sycl::range<1>(2), hd);
        sycl::local_accessor<uint32_t, 1> lane(sycl::range<1>(KV_Q8_GROUP), hd);   // the group's values, wide stores
        hd.parallel_for(sycl::nd_range<3>({(size_t) p.n_tok * planes, (size_t) groups, (size_t) kv_heads * KV_Q8_GROUP},
                                          {1, 1, KV_Q8_GROUP}),
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t j = (int64_t) it.get_group(0) / planes;
            const long long pos = p.step ? (long long) p.step[j * p.step_stride + kStepPos] : (long long) (p.pos0 + j);
            const int h = (int) it.get_group(2), g = (int) it.get_group(1), t = (int) it.get_local_id(2);
            const bool is_v = (int) it.get_group(0) % planes == 1;
            const sycl::sub_group sg = it.get_sub_group();
            const int d = g * KV_Q8_GROUP + t;
            const float x = (is_v ? p.v : p.k)[j * cur_stride + (int64_t) h * head_dim + d];
            const long long page = (long long) p.page_table[pos / page_size];
            const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
            const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
            if (f16) {   // uniform
                const uint16_t hv = f16_gpu::f16_from_f32(x);
                if (page >= 0) (is_v ? p.v_pool : p.k_pool)[row * head_dim + d] = hv;
                if (wide) {   // uniform
                    lane[t] = hv;
                    sycl::group_barrier(it.get_group());
                    if (t < KV_Q8_GROUP / 2) {
                        const uint32_t w = lane[2 * (size_t) t] | (lane[2 * (size_t) t + 1] << 16);
                        const long long at = (long long) row_id * head_dim + (long long) g * KV_Q8_GROUP;
                        if (host.k_pool != nullptr) reinterpret_cast<uint32_t*>((is_v ? host.v_pool : host.k_pool) + at)[t] = w;
                        if (stage.k_pool != nullptr) reinterpret_cast<uint32_t*>((is_v ? stage.v_pool : stage.k_pool) + at)[t] = w;
                    }
                    return;
                }
                if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * head_dim + d] = hv;
                if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * head_dim + d] = hv;
                return;
            }
            float am = sycl::fabs(x);
            for (int o = 16; o > 0; o >>= 1) am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, o));
            if ((t & 31) == 0) warp_max[t >> 5] = am;
            sycl::group_barrier(it.get_group());
            const float amax = sycl::fmax(warp_max[0], warp_max[1]);
            const uint16_t sbits = f16_from_f32(amax / 127.0f);
            const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
            int q = 0;
            if (sf > 0.0f) {
                q = (int) sycl::rint(x / sf);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            if (page >= 0) {
                (is_v ? p.v_q : p.k_q)[row * head_dim + d] = (int8_t) q;
                if (t == 0) (is_v ? p.v_scale : p.k_scale)[row * groups + g] = sbits;
            }
            if (wide) {   // uniform
                lane[t] = (uint32_t) (uint8_t) (int8_t) q;
                sycl::group_barrier(it.get_group());
                if (t < KV_Q8_GROUP / 4) {
                    const size_t i = 4 * (size_t) t;
                    const uint32_t w = lane[i] | (lane[i + 1] << 8) | (lane[i + 2] << 16) | (lane[i + 3] << 24);
                    const long long at = (long long) row_id * head_dim + (long long) g * KV_Q8_GROUP;
                    if (host.k_q != nullptr) reinterpret_cast<uint32_t*>((is_v ? host.v_q : host.k_q) + at)[t] = w;
                    if (stage.k_q != nullptr) reinterpret_cast<uint32_t*>((is_v ? stage.v_q : stage.k_q) + at)[t] = w;
                }
                if (t == 0) {
                    if (host.k_q != nullptr) (is_v ? host.v_scale : host.k_scale)[row_id * groups + g] = sbits;
                    if (stage.k_q != nullptr) (is_v ? stage.v_scale : stage.k_scale)[row_id * groups + g] = sbits;
                }
                return;
            }
            for (const KvHostPools* hp : {&host, &stage}) {
                if (hp->k_q == nullptr) continue;
                (is_v ? hp->v_q : hp->k_q)[row_id * head_dim + d] = (int8_t) q;
                if (t == 0) (is_v ? hp->v_scale : hp->k_scale)[row_id * groups + g] = sbits;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "kv_append_cells");
}

void kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride,
                        int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host) {
    KvAppend a;
    a.step = step;
    a.step_stride = step_stride;
    a.k = kcur;
    a.v = vcur;
    a.cur_stride = cur_stride;
    a.n_tok = n_tok;
    a.k_q = k_q;
    a.v_q = v_q;
    a.k_scale = k_scale;
    a.v_scale = v_scale;
    a.page_table = page_table;
    a.host = host;
    kv_append_cells(a, s, stream);
}

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q8_steps(k_q, v_q, k_scale, v_scale, page_table, step, 0, kcur, vcur, 0, 1, s, stream, host);
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const long long cap = max_ids * s.n_head_kv * (s.head_dim / 4);
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int32_t* table = page_table;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) ((cap + 255) / 256) * 256, 256),
                                                  [=](sycl::nd_item<1> it) {
        const long long n_ids = (long long) step[kStepWidth];
        const int per = head_dim / 4;
        const long long total = n_ids * kv_heads * per;
        const long long i = (long long) it.get_global_id(0);
        if (i >= total) return;
        const long long id = i / (kv_heads * (long long) per);
        const int rem = (int) (i % (kv_heads * (long long) per));
        const int h = rem / per, q4 = rem - h * per;
        const int cell = ids[id];
        const long long page = (long long) table[cell / page_size];
        const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
        const int d = q4 * 4;
        const int groups = head_dim / KV_Q8_GROUP;
        const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
        const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
        const int8_t* kc = k_q + row * head_dim + d;
        const int8_t* vc = v_q + row * head_dim + d;
        const long long dst = ((id * kv_heads + h) * (long long) per + q4) * 4;
        for (int j = 0; j < 4; ++j) {
            k_scratch[dst + j] = f16_from_f32((float) kc[j] * ks);
            v_scratch[dst + j] = f16_from_f32((float) vc[j] * vs);
        }
    });
    if (!stream) core::Runtime::get().wait(e, "kv_gather_q8");
}

}  // namespace strata::kernels
