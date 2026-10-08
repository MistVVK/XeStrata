// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/kv_q4.cpp - the Xe port of Strata's src/kernels/cuda/kv_q4.cu: Q4_0 KV storage after a 256-point Walsh-Hadamard
// rotation.  One sub-group of 32 per Q4_0 block or Hadamard row, as in the CUDA kernels.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/core/runtime.hpp"

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

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) throw core::DeviceError(std::string(what) + ": head_dim must be 256 (the Hadamard transform's size)");
}

// One 32-value group: the signed value of largest magnitude (ties to the larger value) sets d = max / -8, and lanes
// 0..15 pack their code with lane + 16's into one byte.
inline uint16_t q4_group(const sycl::sub_group& sg, float x, uint8_t& byte) {
    float amax = sycl::fabs(x), mval = x;
    for (int o = 16; o > 0; o >>= 1) {
        const float a = sycl::permute_group_by_xor(sg, amax, o);
        const float v = sycl::permute_group_by_xor(sg, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = (int) (x * id + 8.5f);                   // __float2int_rz: truncation toward zero
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = sycl::permute_group_by_xor(sg, qc, 16);   // lanes 0..15 read lane + 16, as __shfl_down(16)
    byte = (uint8_t) (qc | (qhi << 4));
    return f16_from_f32(d);
}

inline void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    constexpr int rows_per_group = 4;
    const size_t groups = (size_t) ((n_rows + rows_per_group - 1) / rows_per_group);
    const float scale = 1.0f / 16.0f;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<2>({groups * rows_per_group, WARP}, {rows_per_group, WARP}),
                                                  [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
        constexpr int N = 256;
        constexpr int el_w = N / WARP;   // 8
        const int64_t r = (int64_t) it.get_global_id(0);
        if (r >= n_rows) return;                              // uniform across the row's sub-group
        const sycl::sub_group sg = it.get_sub_group();
        const float* row_src = src + r * N;
        float* row_dst = dst + r * N;
        float reg[el_w];
        const int lane = (int) it.get_local_id(1);
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
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int planes = (k_q4 == v_q4 && kcur == vcur) ? 1 : 2;
    const KvHostPools hp = host ? *host : KvHostPools{};
    const int32_t* table = page_table;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<3>({(size_t) n_steps * planes, (size_t) (head_dim / QK4_0), (size_t) kv_heads * WARP},
                          {1, 1, WARP}),
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int j = (int) it.get_group(0) / planes;
            const long long pos = (long long) step[(long long) j * step_stride + kStepPos];
            const int h = (int) it.get_group(2), b = (int) it.get_group(1), t = (int) it.get_local_id(2);
            const bool is_v = (int) it.get_group(0) % planes == 1;
            const float x = (is_v ? vcur : kcur)[((long long) j * kv_heads + h) * head_dim + (long long) b * QK4_0 + t];
            uint8_t byte;
            const uint16_t d = q4_group(it.get_sub_group(), x, byte);
            const long long page = (long long) table[pos / page_size];
            if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
            if (hp.k_q4 != nullptr)
                q4_store(is_v ? hp.v_q4 : hp.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b,
                         t, d, byte);
        });
    finish(stream, e, "kv_append_q4");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const KvHostPools hh = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    const int32_t* table = page_table;
    auto& q = queue_for(stream);
    sycl::event e;
    for (int is_v_grid = 0; is_v_grid < 2; ++is_v_grid) {
        e = q.parallel_for(
            sycl::nd_range<3>({(size_t) T, (size_t) kv_heads, (size_t) (head_dim / QK4_0) * WARP}, {1, 1, WARP}),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const long long t = (long long) it.get_group(0);
                const long long pos = pos0 + t;
                const int h = (int) it.get_group(1), b = (int) it.get_group(2), th = (int) it.get_local_id(2);
                const bool is_v = is_v_grid != 0;
                const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
                uint8_t byte;
                const uint16_t d = q4_group(it.get_sub_group(), x, byte);
                const long long page = (long long) table[pos / page_size];
                const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
                if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
                if (hh.k_q4 != nullptr) q4_store(is_v ? hh.v_q4 : hh.k_q4, row_id, b, th, d, byte);
                if (st.k_q4 != nullptr) q4_store(is_v ? st.v_q4 : st.k_q4, row_id, b, th, d, byte);
            });
    }
    finish(stream, e, "kv_append_q4 batch");
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
