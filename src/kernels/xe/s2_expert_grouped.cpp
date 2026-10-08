// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/s2_expert_grouped.cpp - the Xe port of Strata's src/kernels/cuda/s2_expert_grouped.cu: the S2 (Strata Q2
// blob) experts computed on the GPU - one hit at a time, per-hit on device counts, grouped by expert for a window,
// and in the CPU pool's summation order - plus the hit selection, grouping and combination.
//
// Work shapes and orders are the CUDA kernels'.  __dp4a is SPIR-V's integer dot product (cuda_intrinsics.hpp); a warp
// is a sub-group of 32 whose xor butterflies reproduce lane 0 of CUDA's shuffle-down trees, including the CPU-order
// reduction over eight lanes (offsets 4, 1, 2).  The ballot/popc compactions are integer group scans, which place
// every hit where CUDA placed it.  Work-items past the last row keep computing (on row 0) and do not write, because
// a sub-group permute needs every lane.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "cuda_intrinsics.hpp"

#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

using xe::dp4a;

constexpr int H = 2560;
constexpr int FF = 640;
static_assert(FF % 32 == 0, "swiglu_q8_launch works in whole 32-value blocks");
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr int THREADS = 256;
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
[[noreturn]] void fail(const std::string& what) { throw core::DeviceError(what); }

inline float f16_at(const uint8_t* p) { return f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }

inline int code_word(unsigned cbyte) {
    return (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) | (((cbyte >> 6) & 3u) << 24));
}

inline float row_dot_s2_q8(const uint8_t* codes, const uint8_t* scales, const uint8_t* x_q8_0, int n_chunks, int lane,
                           const float* x_scales) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x, the weight-independent term, as ones * x
        const int ones = 0x01010101;
        for (int j = 0; j < 8; ++j) {
            const int cw = code_word(cb[j]);
            int xw;
            std::memcpy(&xw, xq + 4 * j, 4);   // a block_q8_0's data is never 4-byte aligned
            s = dp4a(cw, xw, s);
            hx = dp4a(ones, xw, hx);
        }
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);   // one weight scale per two 32-element chunks
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}

// gate/up rows, one sub-group per row; the output is gate-major (every hit's gate rows first, then every hit's up
// rows), so the swiglu and the quantizer read contiguous ranges
sycl::event gu_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                      const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
                      const int32_t* d_count = nullptr, const int32_t* dst_index = nullptr, int tok_div = 0) {
    const long long total = (long long) n_hits * 2LL * FF;
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long slot = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (slot >= total) return;   // uniform over the sub-group
        const long long rows_per_hit = 2LL * FF;
        const int h = (int) (slot / rows_per_hit);
        if (d_count != nullptr && h >= *d_count) return;   // token graph: capacity layout, device count
        const int i = (int) (slot % rows_per_hit);
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xq = x_q8_0;
        const float* xs = x_scales;
        if (tok_div > 0) {   // a verify window: each hit reads its own token's activation
            const int tok = dst_index[h] / tok_div;
            xq += (size_t) tok * (size_t) (H / 32) * 34;
            if (xs != nullptr) xs += (size_t) tok * (size_t) (H / 32);
        }
        const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU, blob + O_GU_SCALES + (size_t) i * SC_GU * 2, xq,
                                        H / 32, lane, xs);
        const float s = warp_sum(sg, acc);
        if (lane != 0) return;
        const int r = i >> 1;
        const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
        gate_up[base + (size_t) r] = s;
    });
}

sycl::event swiglu_launch(sycl::queue& q, float* gate_up, long long n_pairs) {
    return q.parallel_for(sycl::range<1>((size_t) n_pairs), [=](sycl::id<1> id) {
        const long long i = (long long) id[0];
        const float g = gate_up[i];
        const float u = gate_up[n_pairs + i];
        gate_up[i] = (g / (1.0f + sycl::exp(-g))) * u;
    });
}

// swiglu_launch and quantize_q8_0_scaled in one kernel (upstream 3281ac32): one sub-group a 32-value block, each lane
// its value's SwiGLU (written back to gate_up) and the block's scaled Q8_0, the same arithmetic as the two launches
sycl::event swiglu_q8_launch(sycl::queue& q, float* gate_up, long long n_pairs, uint8_t* blocks, float* scales) {
    const long long n_blocks = n_pairs / 32;
    const size_t groups = (size_t) ((n_blocks + 7) / 8);
    return q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long b = (long long) it.get_group(0) * 8 + (long long) sg.get_group_linear_id();
        if (b >= n_blocks) return;   // uniform over the sub-group
        const int lane = (int) sg.get_local_linear_id();
        const long long i = b * 32 + lane;
        const float g = gate_up[i];
        const float u = gate_up[n_pairs + i];
        const float xv = (g / (1.0f + sycl::exp(-g))) * u;
        gate_up[i] = xv;
        uint8_t* out = blocks + b * 34;
        float amax = sycl::fabs(xv);
        for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        const float sc = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = sc > 0.f ? 1.f / sc : 0.f;
        if (lane == 0) {
            scales[b] = sc;
            const uint16_t d16bits = f16_from_f32(sc);
            out[0] = (uint8_t) (d16bits & 0xFF);
            out[1] = (uint8_t) (d16bits >> 8);
        }
        const float t = xv * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + lane] = (uint8_t) (int8_t) v;
    });
}

sycl::event down_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        long long blob_bytes, const uint8_t* h_q8_0, const float* h_scales, float* out, int n_hits,
                        const int32_t* d_count = nullptr) {
    const long long total = (long long) n_hits * H;
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long row = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (row >= total) return;
        const int h = (int) (row / H);
        if (d_count != nullptr && h >= *d_count) return;
        const int r = (int) (row % H);
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
        const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2,
                                        xb, FF / 32, lane, h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
        const float s = warp_sum(sg, acc);
        if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
    });
}

// ---- the CPU pool's summation order (bench/micro/hit_cpu_order_parity in the CUDA tree)
inline int dot4(const uint8_t* codes, const int8_t* q) {
    int xw;
    std::memcpy(&xw, q, sizeof xw);
    return dp4a(code_word(*codes), xw, 0);
}

// eight lanes per row; only lane 0 of each eight is consumed.  _mm_add_ps(low128, high128), then two _mm_hadd_ps:
// the pair order is 4, 1, 2
inline float row_dot_cpu_order(const sycl::sub_group& sg, const uint8_t* codes, const uint8_t* scales,
                               const uint8_t* xq, const float* xs, const float* hx, int blocks, int lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_at(scales + 2 * b);
        const int lo = dot4(codes + b * 16 + lane, (const int8_t*) (xq + (size_t) (2 * b) * 34 + 2) + lane * 4);
        const int hi = dot4(codes + b * 16 + 8 + lane, (const int8_t*) (xq + (size_t) (2 * b + 1) * 34 + 2) + lane * 4);
        acc = sycl::fma(d * xs[2 * b], (float) lo, acc);
        acc = sycl::fma(d * xs[2 * b + 1], (float) hi, acc);
        if (lane == 0) corr = corr + d * (hx[2 * b] + hx[2 * b + 1]);
    }
    acc = acc + sycl::permute_group_by_xor(sg, acc, 4);
    acc = acc + sycl::permute_group_by_xor(sg, acc, 1);
    acc = acc + sycl::permute_group_by_xor(sg, acc, 2);
    return acc - corr;
}

template <bool DOWN>
sycl::event cpu_order_projection(sycl::queue& q, const uint8_t* blob_base, const int32_t* slots,
                                 const int32_t* destinations, long long blob_bytes, const uint8_t* xq, const float* xs,
                                 const float* hx, float* out, int n_hits) {
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const int rows = n_hits * rows_per_hit;
    const int rows_per_group = THREADS / 8;
    const size_t groups = (size_t) ((rows + rows_per_group - 1) / rows_per_group);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int t = (int) it.get_local_id(0);
        const int row_any = (int) it.get_group(0) * rows_per_group + t / 8;
        const bool valid = row_any < rows;
        const int row = valid ? row_any : 0;
        const int h = row / rows_per_hit;
        const int r = row % rows_per_hit;
        const int lane = t & 7;
        const uint8_t* blob = blob_base + (size_t) slots[h] * (size_t) blob_bytes;
        const int chunks_offset = DOWN ? h * (FF / 32) : 0;
        const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
        const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2 : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;
        const float value = row_dot_cpu_order(sg, codes, scales, xq + (size_t) chunks_offset * 34, xs + chunks_offset,
                                              hx + chunks_offset, DOWN ? SC_D : SC_GU, lane);
        if (!valid || lane != 0) return;
        if (DOWN) out[(size_t) destinations[h] * H + r] = value;
        else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) h * FF + (r >> 1)] = value;
    });
}

// ---- grouped by expert (a verify window)
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
// a group holds one entry per token of the window routed to its expert, and the kernels below keep at most GMAX of
// them: a longer window would drop entries without a word (upstream 64cfe3c)
static_assert(GMAX >= kVerifyMaxT, "a verify window's group can exceed GMAX entries");
constexpr int GU_ROWS = 32;                      // gate/up rows per work-group: 4 per sub-group
constexpr int D_ROWS = 64;                       // down rows per work-group: 8 per sub-group

inline float chunk_dot(const uint8_t* cbytes, const int* xw, float dw, float dx) {
    int s = 0, hx = 0;
    const int ones = 0x01010101;
    for (int j = 0; j < 8; ++j) {
        const int cw = code_word(cbytes[j]);
        s = dp4a(cw, xw[j], s);
        hx = dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}

namespace {
void hit_pipeline(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                  const int32_t* d_count, int64_t n, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch,
                  float* out, void* stream, const float* x_scales, int tok_div) {
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) n * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    gu_launch(q, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) n, d_count, dst_index, tok_div);
    void* s = &q;
    if (x_scales != nullptr) {
        swiglu_q8_launch(q, gate_up, n * (long long) FF, h_q8_0, h_scales);
    } else {
        swiglu_launch(q, gate_up, n * (long long) FF);
        quantize_q8_0(gate_up, h_q8_0, n * (int64_t) FF, s);
    }
    const auto e = down_launch(q, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                               x_scales != nullptr ? h_scales : nullptr, out, (int) n, d_count);
    if (!stream) core::Runtime::get().wait(e, "moe_hit_grouped_s2");
}
}  // namespace

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, nullptr, n_hits, blob_bytes, x_q8_0, scratch, out, stream, x_scales, 0);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) fail("moe_hit_select: k must be 1..32");
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        int s = -1;
        if (lane < k) {
            const int ex = ids[lane];
            if (ex >= 0 && ex < n_expert) s = res_row[ex];
        }
        const int flag = s >= 0 ? 1 : 0;
        const int at = sycl::exclusive_scan_over_group(sg, flag, sycl::plus<int>());
        const int total = sycl::reduce_over_group(sg, flag, sycl::plus<int>());
        if (s >= 0) { slot[at] = s; dst[at] = lane; }
        if (lane == 0) *count = total;
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_select");
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales, 0);
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) fail("moe_hit_select_multi: n must be 1..128");
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const int i = (int) it.get_local_id(0);
        int s = -1;
        if (i < n) {
            const int ex = ids[i];
            if (ex >= 0 && ex < n_expert) s = res_row[ex];
        }
        const int flag = s >= 0 ? 1 : 0;
        const int at = sycl::exclusive_scan_over_group(it.get_group(), flag, sycl::plus<int>());
        const int total = sycl::reduce_over_group(it.get_group(), flag, sycl::plus<int>());
        if (s >= 0) { slot[at] = s; dst[at] = i; }
        if (i == 0) *count = total;
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_select_multi");
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales,
                 k_per_token);
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) fail("moe_group_resident: n must be 1..128");
    const long long blob_ll = (long long) blob;
    const auto e = Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> e_s(128, h), first_s(128, h), size_s(128, h), gidx_s(128, h), gstart_s(129, h);
        h.parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            const int ex = i < n ? ids[i] : -1;
            e_s[i] = ex;
            sycl::group_barrier(it.get_group());
            int first = i, rank = 0, size = 0;
            if (i < n) {
                for (int j = 0; j < i; ++j)
                    if (e_s[j] == ex) { if (first == i) first = j; ++rank; }
                if (first == i)
                    for (int j = i; j < n; ++j) size += e_s[j] == ex;
            }
            first_s[i] = first;
            size_s[i] = (i < n && first == i) ? size : 0;
            sycl::group_barrier(it.get_group());
            if (i == 0) {
                int gi = 0, acc = 0;
                for (int j = 0; j < n; ++j)
                    if (first_s[j] == j) {
                        gidx_s[j] = gi;
                        gstart_s[gi] = acc;
                        grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob_ll);
                        grp_start[gi] = acc;
                        acc += size_s[j];
                        ++gi;
                    }
                grp_start[gi] = acc;
                counts[0] = gi;
                counts[1] = acc;
            }
            sycl::group_barrier(it.get_group());
            if (i < n) {
                const int at = gstart_s[gidx_s[first]] + rank;
                ent_dst[at] = i;
                ent_tok[at] = i / k_per_tok;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "moe_group_resident");
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const int cap_e = (int) cap_entries;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> xs_q(sycl::range<1>(GMAX * (H / 4)), hd);   // the entries' int8 activations as words
        sycl::local_accessor<float, 1> xs_d(sycl::range<1>(GMAX * (H / 32)), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) cap_groups, (size_t) (2 * FF / GU_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;   // uniform
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (H / 32); i += 256) {
                const int k = i / (H / 32), c = i - k * (H / 32);
                const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
                xs_d[k * (H / 32) + c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
                const int8_t* qq = (const int8_t*) (xb + 2);
                for (int w = 0; w < 8; ++w) {
                    int v;
                    std::memcpy(&v, qq + 4 * w, 4);
                    xs_q[k * (H / 4) + c * 8 + w] = v;
                }
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * GU_ROWS;
            for (int rr = warp; rr < GU_ROWS; rr += 8) {
                const int i = row0 + rr;
                const uint8_t* codes = blob + (size_t) i * ROW_GU;
                const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
                uint8_t cb[GU_CHUNKS][8];
                float dw[GU_CHUNKS];
                for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                    const int c = lane + 32 * qc;
                    if (c < H / 32) {
                        std::memcpy(cb[qc], codes + (size_t) c * 8, 8);
                        dw[qc] = f16_at(scales + (size_t) (c >> 1) * 2);
                    }
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                        const int c = lane + 32 * qc;
                        if (c >= H / 32) break;
                        int xw[8];
                        for (int j = 0; j < 8; ++j) xw[j] = xs_q[k * (H / 4) + c * 8 + j];
                        acc += chunk_dot(cb[qc], xw, dw[qc], xs_d[k * (H / 32) + c]);
                    }
                    const float sum = warp_sum(sg, acc);
                    if (lane == 0) {
                        const int en = e0 + k, r = i >> 1;
                        const size_t b = (i & 1) ? ((size_t) cap_e * FF + (size_t) en * FF) : ((size_t) en * FF);
                        gate_up[b + (size_t) r] = sum;
                    }
                }
            }
        });
    });
    void* s = &q;
    if (x_scales != nullptr) {
        swiglu_q8_launch(q, gate_up, cap_entries * (long long) FF, h_q8_0, h_scales);
    } else {
        swiglu_launch(q, gate_up, cap_entries * (long long) FF);
        quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, s);
    }
    const float* hs = x_scales != nullptr ? h_scales : nullptr;
    const auto e = q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> hs_q(sycl::range<1>(GMAX * (FF / 4)), hd);
        sycl::local_accessor<float, 1> hs_d(sycl::range<1>(GMAX * (FF / 32)), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) cap_groups, (size_t) (H / D_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (FF / 32); i += 256) {
                const int k = i / (FF / 32), c = i - k * (FF / 32);
                const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
                hs_d[k * (FF / 32) + c] = hs ? hs[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
                const int8_t* qq = (const int8_t*) (xb + 2);
                for (int w = 0; w < 8; ++w) {
                    int v;
                    std::memcpy(&v, qq + 4 * w, 4);
                    hs_q[k * (FF / 4) + c * 8 + w] = v;
                }
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * D_ROWS;
            for (int rr = warp; rr < D_ROWS; rr += 8) {
                const int r = row0 + rr;
                const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
                const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
                const int c = lane;
                uint8_t cb[8] = {};
                float dw = 0.0f;
                if (c < FF / 32) {
                    std::memcpy(cb, codes + (size_t) c * 8, 8);
                    dw = f16_at(scales + (size_t) (c >> 1) * 2);
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    if (c < FF / 32) {
                        int xw[8];
                        for (int j = 0; j < 8; ++j) xw[j] = hs_q[k * (FF / 4) + c * 8 + j];
                        acc += chunk_dot(cb, xw, dw, hs_d[k * (FF / 32) + c]);
                    }
                    const float sum = warp_sum(sg, acc);
                    if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
                }
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "moe_grouped_s2");
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const int ne = (int) n_embd;
    const size_t gx = (size_t) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8);
    const auto e = Q(stream).parallel_for(sycl::nd_range<2>({(size_t) cap, gx * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const int h = (int) it.get_group(0);
        if (h >= *count) return;
        const size_t row = (size_t) dst[h] * (size_t) ne;
        for (int i = (int) it.get_global_id(1); i < ne; i += (int) it.get_global_range(1)) parts[row + i] += hit_out[row + i];
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_add");
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) fail("moe_hit_grouped_s2_cpu_order requires fp32 activation scales");
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    // the input's correction term: scale * sum of its int8 values, per chunk
    q.parallel_for(sycl::range<1>(H / 32), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const int8_t* qv = (const int8_t*) (x_q8_0 + (size_t) c * 34 + 2);
        int sum = 0;
        for (int j = 0; j < 32; ++j) sum += qv[j];
        xh[c] = x_scales[c] * (float) sum;
    });
    cpu_order_projection<false>(q, blob_base, slot_index, dst_index, blob_bytes, x_q8_0, x_scales, xh, gu, (int) n_hits);
    if (gate_up_trace != nullptr) q.memcpy(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float));
    const int pairs = (int) n_hits * FF;
    q.parallel_for(sycl::range<1>((size_t) pairs), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        const float g = gu[i];
        const float eg = sycl::exp(-g);
        gu[i] = (g / (1.0f + eg)) * gu[pairs + i];
    });
    q.parallel_for(sycl::range<1>((size_t) (n_hits * (FF / 32))), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const float* xb = gu + c * 32;
        uint8_t* o = hq + (size_t) c * 34;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        hs[c] = s;
        const uint16_t bits = f16_from_f32(s);
        o[0] = (uint8_t) bits;
        o[1] = (uint8_t) (bits >> 8);
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            const float t = xb[j] * inv;
            int v = (int) (t + (t >= 0.0f ? 0.5f : -0.5f));
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            o[2 + j] = (uint8_t) (int8_t) v;
            sum += v;
        }
        hh[c] = s * (float) sum;
    });
    const auto e = cpu_order_projection<true>(q, blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh, out, (int) n_hits);
    if (!stream) core::Runtime::get().wait(e, "moe_hit_grouped_s2_cpu_order");
}

}  // namespace strata::kernels
