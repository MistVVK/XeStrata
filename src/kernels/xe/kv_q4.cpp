// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/kv_q4.cpp - the Xe port of Strata's src/kernels/cuda/kv_q4.cu: Q4_0 KV storage after a 256-point Walsh-Hadamard
// rotation.  One sub-group of 32 per Q4_0 block or Hadamard row, as in the CUDA kernels.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "kv_q4_parts.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_kv_q4
#include "strata/kernels/device_arch_tag.hpp"

#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    constexpr int rows_per_group = 4;
    const size_t groups = (size_t) ((n_rows + rows_per_group - 1) / rows_per_group);
    const float scale = 1.0f / 16.0f;
    // one dimension: a prompt's rows (tokens x heads) pass the 65,535 work-groups a GPU's first dimension takes
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(groups * rows_per_group * WARP, rows_per_group * WARP),
                                                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        constexpr int N = 256;
        constexpr int el_w = N / WARP;   // 8
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t r = (int64_t) it.get_group(0) * rows_per_group + (int64_t) sg.get_group_linear_id();
        if (r >= n_rows) return;                              // uniform across the row's sub-group
        const float* row_src = src + r * N;
        float* row_dst = dst + r * N;
        float reg[el_w];
        const int lane = (int) sg.get_local_linear_id();
        for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * WARP + lane] * scale;
        for (int h = 1; h < WARP; h *= 2) {
            for (int j = 0; j < el_w; ++j) {
                const float val = reg[j];
                const float val2 = sycl::permute_group_by_xor(sg, val, h);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
        for (int h = WARP; h < N; h *= 2) {
            const int step = h / WARP;
            for (int j = 0; j < el_w; j += 2 * step) {
                for (int k = 0; k < step; ++k) {
                    const float x = reg[j + k];
                    const float y = reg[j + k + step];
                    reg[j + k] = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }
        for (int i = 0; i < el_w; ++i) row_dst[i * WARP + lane] = reg[i];
    });
    finish(stream, e, "fwht256");
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    KvAppend a;
    a.step = step;
    a.step_stride = step_stride;
    a.k = kcur;
    a.v = vcur;
    a.n_tok = n_steps;
    a.k_q4 = k_q4;
    a.v_q4 = v_q4;
    a.page_table = page_table;
    a.host = host;
    kv_append_cells(a, s, stream);
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    KvAppend a;
    a.pos0 = pos0;
    a.k = K;
    a.v = V;
    a.n_tok = T;
    a.k_q4 = k_q4;
    a.v_q4 = v_q4;
    a.page_table = page_table;
    a.host = host;
    a.stage = stage;
    kv_append_cells(a, s, stream);
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int blocks_per_head = head_dim / QK4_0;
    const int64_t cap_blocks = max_ids * kv_heads * blocks_per_head;
    constexpr int rows_per_group = 4;
    const size_t groups = (size_t) ((cap_blocks + rows_per_group - 1) / rows_per_group);
    const int32_t* table = page_table;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<2>({groups * rows_per_group, WARP}, {rows_per_group, WARP}),
                                                  [=](sycl::nd_item<2> it) {
        const long long n_ids = (long long) step[kStepWidth];
        const int bytes_per_head = blocks_per_head * (int) sizeof(block_q4_0);    // 144
        const long long total_blocks = n_ids * kv_heads * blocks_per_head;
        const long long blk_idx = (long long) it.get_global_id(0);
        if (blk_idx >= total_blocks) return;
        const int t = (int) it.get_local_id(1);
        const long long id = blk_idx / (kv_heads * blocks_per_head);
        const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
        const int h = rem / blocks_per_head;
        const int b = rem % blocks_per_head;
        const int cell = ids[id];
        const long long page = (long long) table[cell / page_size];
        const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
        const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
        const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
        const float kd = f32_from_f16(k_blk->d);
        const float vd = f32_from_f16(v_blk->d);
        const int j = t < 16 ? t : (t - 16);
        const uint8_t k_byte = k_blk->qs[j];
        const uint8_t v_byte = v_blk->qs[j];
        const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
        const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
        const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
        k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
        v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
    });
    finish(stream, e, "kv_gather_q4");
}

}  // namespace strata::kernels
