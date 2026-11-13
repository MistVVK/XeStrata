#include "b60_layout_state.hpp"
#include <cstdlib>
// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/kprof.hpp"
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/workload_trace.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
// BEGIN PRIVATE QSA TWO STAGE INCLUDES
#include <climits>
#include <stdexcept>
// END PRIVATE QSA TWO STAGE INCLUDES
#include <cmath>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv;   // query heads per KV head (12 for Flash-Next; 1..16, a multiple of 4)
static_assert(G % 4 == 0 && G <= 16, "qsa_decode_attn: query heads per KV head must be 4, 8, 12 or 16 (float4 probability rows, the 16-lane score reduce)");
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_decode_attn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float warp_max(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__dpct_inline__ void load8_f16(const QsaAttnPools &p, bool value, long long row,
                               int d0, float *out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4 *>(base);
    const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f =
            h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}
__dpct_inline__ void load8_q8(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2 *>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__dpct_inline__ void load8_q4(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__dpct_inline__ void load8(const QsaAttnPools &p, bool value, long long row,
                           int d0, float *out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, bool LANE_CELL = false, int RS = 0>
/*
DPCT1110: The total declared local variable size in device function
attn_chunk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_chunk_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0,
                  QsaAttnPoolsRows prow = {}) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (prow.n > 0) p = prow.p[item_ct1.get_group(0)];   // a batch window's query attends over its own slot's pools
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G * CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    // the scores and probabilities: [head][cell], or - RS - [cell][head] so the value phase reads a cell's 12 probabilities as 3 vectors
    const auto spi = [](int h, int c) { return RS != 0 ? c * G + h : h * CHUNK + c; };
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if constexpr (LANE_CELL) {
        // S25 (STRATA_ATTN_LANECELL=1): thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself - the same
        // 8-dimension partial per former lane (the identical expression), then the former warp_sum's butterfly as
        // lane 0 ran it (a[l] += a[l + o] for o = 16 .. 1), so every score is bit-identical without 12 shuffle
        // reductions per cell
        constexpr int HPT = G * CHUNK / THREADS;   // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[spi(h0 + j, c)] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const sycl::float4 qa =
                        *reinterpret_cast<const sycl::float4 *>(
                            &sq[h0 + j][l * 8]);
                    const sycl::float4 qb =
                        *reinterpret_cast<const sycl::float4 *>(
                            &sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                              k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                              k8[6] * qb.z() + k8[7] * qb.w();
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[spi(h0 + j, c)] = a[j][0] * scale;
            }
        }
    } else
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[spi(lane, c)] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
        if constexpr (RS == 1) {
            // 151-b70: the 12 heads' 32-lane butterfly sums as ONE reduce-scatter - 16 shuffles instead of 60.  Each stage pairs the
            // same lanes in the same order as warp_sum (xor 16, 8, 4, 2, 1) and adds own + partner, so every head's sum is the
            // same tree and the same bits; the heads are padded to 16, halved at each stage (the lane keeps the half its bit
            // selects, sends the other), and after the xor-2 stage lane 2h + {0, 1} holds head h's partial pair; xor 1 finishes it.
            float part[16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 qa =
                    *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
                const sycl::float4 qb =
                    *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
                part[h] = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                          k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                          k8[6] * qb.z() + k8[7] * qb.w();
            }
#pragma unroll
            for (int h = G; h < 16; ++h) part[h] = 0.0f;
            const auto xr = [](float v, int o) {
                return dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
            };
            float c8[8];
            const bool b16 = (lane & 16) != 0, b8 = (lane & 8) != 0, b4 = (lane & 4) != 0, b2 = (lane & 2) != 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float send = b16 ? part[i] : part[8 + i];
                const float keep = b16 ? part[8 + i] : part[i];
                c8[i] = keep + xr(send, 16);
            }
            float c4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float send = b8 ? c8[i] : c8[4 + i];
                const float keep = b8 ? c8[4 + i] : c8[i];
                c4[i] = keep + xr(send, 8);
            }
            float c2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float send = b4 ? c4[i] : c4[2 + i];
                const float keep = b4 ? c4[2 + i] : c4[i];
                c2[i] = keep + xr(send, 4);
            }
            const float send1 = b2 ? c2[0] : c2[1];
            const float keep1 = b2 ? c2[1] : c2[0];
            float s = keep1 + xr(send1, 2);
            s = s + xr(s, 1);
            const int head = (lane >> 1) & 15;
            if ((lane & 1) == 0 && head < G) sp[spi(head, c)] = s * scale;
        } else {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = warp_sum(s);
            if (lane == 0) sp[spi(h, c)] = s * scale;
        }
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[spi(h, lane)], b = sp[spi(h, lane + 32)];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[spi(h, lane)] = ea;
        sp[spi(h, lane + 32)] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[srow[c] * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[srow[c] * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
        if constexpr (RS != 0) {
            // the cell's 12 probabilities as three vector loads (the layout is [cell][head]); the same fma per head in the same order
#pragma unroll
            for (int q4 = 0; q4 < G / 4; ++q4) {
                const sycl::float4 pv = *reinterpret_cast<const sycl::float4 *>(&sp[c * G + 4 * q4]);
                acc[4 * q4 + 0] = sycl::fma(pv.x(), v, acc[4 * q4 + 0]);
                acc[4 * q4 + 1] = sycl::fma(pv.y(), v, acc[4 * q4 + 1]);
                acc[4 * q4 + 2] = sycl::fma(pv.z(), v, acc[4 * q4 + 2]);
                acc[4 * q4 + 3] = sycl::fma(pv.w(), v, acc[4 * q4 + 3]);
            }
        } else {
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[spi(h, c)], v, acc[h]);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so it runs on cards below sm_75 only, and exists only in the experimental
// build (-DSTRATA_EXPERIMENTAL_SM60=ON): the ready-made engine keeps exactly the kernel above.
#if defined(STRATA_EXPERIMENTAL_SM60)
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = 2;   // 4 measured slower on a V100 (the registers cost occupancy)
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's 12 weights (three broadcast float4 loads) into the 12 accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
#endif  // STRATA_EXPERIMENTAL_SM60

__dpct_inline__ void attn_merge_kernel(const float *__restrict__ part_acc,
                                       const float *__restrict__ part_m,
                                       const float *__restrict__ part_l,
                                       int n_chunks, float *__restrict__ attn,
                                       long long scratch_stride = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    part_acc += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    attn += (size_t)item_ct1.get_group(1) *
            (size_t)item_ct1.get_group_range(2) * HD;
    const int h = item_ct1.get_group(2); // global query head
    const int kvh = h / G, hl = h % G;
    const int d = item_ct1.get_local_id(2);
    float M = -FLT_MAX;
#pragma unroll
    for (int c = 0; c < n_chunks; ++c)
        M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma((float)(part_l[slot * G + hl]), (float)w, L);
        acc = sycl::fma((float)(part_acc[((size_t)slot * G + hl) * HD + d]),
                        (float)w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// Chunks beyond the device-side width contain only -FLT_MAX maxima. Keep the
// original fold order while issuing eight independent partial reads together.
__dpct_inline__ void attn_merge_active_kernel(
    const float *__restrict__ part_acc, const float *__restrict__ part_m,
    const float *__restrict__ part_l, int n_chunks, float *__restrict__ attn,
    long long scratch_stride, const int32_t *__restrict__ steps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int z = item_ct1.get_group(1);
    part_acc += (size_t) z * (size_t) scratch_stride;
    part_m += (size_t) z * (size_t) scratch_stride;
    part_l += (size_t) z * (size_t) scratch_stride;
    attn += (size_t) z * (size_t) item_ct1.get_group_range(2) * HD;
    const int width = steps[(size_t) z * kStepCount + kStepWidth];
    const int reach = width <= 0 ? 0 : 1 + (width - 1) / CHUNK;
    const int nc = sycl::min(reach, n_chunks);
    const int h = item_ct1.get_group(2), kvh = h / G, hl = h % G;
    const int d = item_ct1.get_local_id(2);
    const float* pm_h = part_m + (size_t) kvh * n_chunks * G + hl;
    const float* pl_h = part_l + (size_t) kvh * n_chunks * G + hl;
    const float* pa_h = part_acc + ((size_t) kvh * n_chunks * G + hl) * HD + d;
    constexpr int MU = 8;
    float M = -FLT_MAX;
    int c = 0;
    for (; c + MU <= nc; c += MU) {
        float pm[MU];
#pragma unroll
        for (int j = 0; j < MU; ++j) pm[j] = pm_h[(size_t) (c + j) * G];
#pragma unroll
        for (int j = 0; j < MU; ++j) M = sycl::fmax(M, pm[j]);
    }
    for (; c < nc; ++c) M = sycl::fmax(M, pm_h[(size_t) c * G]);
    float L = 0.0f, acc = 0.0f;
    c = 0;
    for (; c + MU <= nc; c += MU) {
        float pm[MU], pl[MU], pa[MU];
#pragma unroll
        for (int j = 0; j < MU; ++j) {
            pm[j] = pm_h[(size_t) (c + j) * G];
            pl[j] = pl_h[(size_t) (c + j) * G];
            pa[j] = pa_h[(size_t) (c + j) * G * HD];
        }
#pragma unroll
        for (int j = 0; j < MU; ++j) {
            if (pm[j] == -FLT_MAX) continue;
            const float w = sycl::native::exp(pm[j] - M);
            L = sycl::fma(pl[j], w, L);
            acc = sycl::fma(pa[j], w, acc);
        }
    }
    for (; c < nc; ++c) {
        const float m = pm_h[(size_t) c * G];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma(pl_h[(size_t) c * G], w, L);
        acc = sycl::fma(pa_h[(size_t) c * G * HD], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

bool merge_active_on(const sycl::device& d) {
    static const bool on = strata::bm_default_on("STRATA_ATTN_MERGE_ACTIVE", d);
    return on;
}

// BEGIN PRIVATE QSA TWO STAGE HELPER

struct QsaMergeGroupPartial { float m, l, acc; };
__dpct_inline__ QsaMergeGroupPartial qsa_merge_group_partial(
    const float* __restrict__ part_acc, const float* __restrict__ part_m,
    const float* __restrict__ part_l, int n_chunks, int kvh, int hl, int d,
    int first, int last) {
    float M = -FLT_MAX;
#pragma unroll
    for (int c = first; c < last; ++c)
        M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = first; c < last; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma((float)(part_l[slot * G + hl]), (float)w, L);
        acc = sycl::fma((float)(part_acc[((size_t)slot * G + hl) * HD + d]),
                        (float)w, acc);
    }
    return {M, L, acc};
}

constexpr int QsaMergeGroupChunks = 32;
class QsaMergeTwoStageOriginal;
class QsaMergeTwoStagePartial;
class QsaMergeTwoStageFinal;

void launch_qsa_merge_two_stage_original(const float* part_acc,
    const float* part_m, const float* part_l, int n_chunks, float* out,
    sycl::queue* queue) {
    auto props = sycl::ext::oneapi::experimental::properties{};
    queue->parallel_for<QsaMergeTwoStageOriginal>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, size_t(24) * THREADS),
                          sycl::range<3>(1, 1, THREADS)), props,
        [=](sycl::nd_item<3>) {
            attn_merge_kernel(part_acc, part_m, part_l, n_chunks, out, 0);
        });
}

void launch_qsa_merge_two_stage_partial(const float* part_acc,
    const float* part_m, const float* part_l, int n_chunks,
    float* group_acc, float* group_m, float* group_l, int n_groups,
    float* out, sycl::queue* queue) {
    static_assert(HD == 256 && G == 12 && THREADS == 256);
    auto props = sycl::ext::oneapi::experimental::properties{};
    queue->parallel_for<QsaMergeTwoStagePartial>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, size_t(24) * n_groups * THREADS),
                          sycl::range<3>(1, 1, THREADS)), props,
        [=](sycl::nd_item<3> item) {
            const size_t index = item.get_group(2);
            const int h = int(index / n_groups), group = int(index % n_groups);
            const int kvh = h / G, hl = h % G;
            const int d = int(item.get_local_id(2));
            const int first = group * QsaMergeGroupChunks;
            const int last = sycl::min(first + QsaMergeGroupChunks, n_chunks);
            const auto partial = qsa_merge_group_partial(
                part_acc, part_m, part_l, n_chunks, kvh, hl, d, first, last);
            const int slot = (kvh * n_groups + group) * G + hl;
            // Keep an unnormalised acc: the original final helper applies L once.
            group_acc[size_t(slot) * HD + d] = partial.acc;
            if (d == 0) { group_m[slot] = partial.m; group_l[slot] = partial.l; }
        });
    // The caller's in-order queue/capture supplies the inter-kernel dependency.
    // No new wait/event/node API, SLM or global barrier is introduced here.
    queue->parallel_for<QsaMergeTwoStageFinal>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, size_t(24) * THREADS),
                          sycl::range<3>(1, 1, THREADS)), props,
        [=](sycl::nd_item<3>) {
            attn_merge_kernel(group_acc, group_m, group_l, n_groups, out, 0);
        });
}
// END PRIVATE QSA TWO STAGE HELPER
#if defined(STRATA_EXPERIMENTAL_SM60)
// the current device is below sm_75 (per device: a layer split can mix cards); STRATA_ATTN_PRE75=0 turns PR #540's
// kernel off (A/B)
bool pre75_attn() {
    static const bool off = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e != nullptr && e[0] == '0';
    }();
    static int cc[64] = {};
    int dev = 0;
    if (off) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
}
#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

// ---- SYCL port: the prompt path's chunk kernel with the scores computed one work-item per cell (attn_bench).
// The warp kernel above splits a cell's 256 dimensions over a sub-group and adds them up with a 5-step shuffle
// tree per query head - 60 shuffles a cell. Here work-item c holds cell c's key row (16-byte loads) and all 12
// heads' dot products, reading the query heads from local memory, where every work-item reads the same element
// at once. 128-cell chunks. The values pass is the warp kernel's. FP32, but another summation order than the warp
// kernel: not bitwise (relative error ~2e-7 in attn_bench); attn_bench: 1.55-1.69x the warp kernel's time.
constexpr int PC_CHUNK = 128;

// 16 consecutive key values of row `row`, dimensions [d0, d0+16)
template <int KV_MODE>
__dpct_inline__ void load16_k(const QsaAttnPools &p, long long row, int d0, float *out) {
    if constexpr (KV_MODE == 0) {
        load8_f16(p, false, row, d0, out);
        load8_f16(p, false, row, d0 + 8, out + 8);
    } else if constexpr (KV_MODE == 1 || KV_MODE == 3) {
        const int8_t* codes = p.k_q + row * HD + d0;
        const float sc = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                             p.k_scale[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]))
                             .convert<float, sycl::rounding_mode::automatic>()[0];
        const sycl::int4 raw = *reinterpret_cast<const sycl::int4 *>(codes);
        const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 16; ++j) out[j] = (float) c[j] * sc;
    } else {   // q4_0 (rotated): element j of a 32-block in qs[j]'s low nibble, j + 16 in its high one
        load8_q4(p, false, row, d0, out);
        load8_q4(p, false, row, d0 + 8, out + 8);
    }
}

template <int KV_MODE>
__dpct_inline__ float load1_v(const QsaAttnPools &p, long long row, int t) {
    if constexpr (KV_MODE == 0) {
        return sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(p.v_pool[row * HD + t]))
            .convert<float, sycl::rounding_mode::automatic>()[0];
    } else if constexpr (KV_MODE == 1) {
        const float sc = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                             p.v_scale[row * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]))
                             .convert<float, sycl::rounding_mode::automatic>()[0];
        return (float) p.v_q[row * HD + t] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + t / QK4_0;
        const float d = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                            .convert<float, sycl::rounding_mode::automatic>()[0];
        const int rem = t % QK4_0;
        const uint8_t byte = blk->qs[rem < 16 ? rem : rem - 16];
        return (float) ((rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8)) * d;
    }
}

template <int KV_MODE>
__dpct_inline__ void attn_chunk_percell_kernel(const float *__restrict__ q, QsaAttnPools p,
                                               const int32_t *__restrict__ ids, const int32_t *__restrict__ step,
                                               int n_kv_heads, int page_size, float scale, float *__restrict__ part_acc,
                                               float *__restrict__ part_m, float *__restrict__ part_l, int n_chunks,
                                               int cap, long long scratch_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item_ct1.get_sub_group();
    q += (size_t) item_ct1.get_group(0) * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) item_ct1.get_group(0) * (size_t) cap;
    step += (size_t) item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t) item_ct1.get_group(0) * (size_t) scratch_stride;
    part_m += (size_t) item_ct1.get_group(0) * (size_t) scratch_stride;
    part_l += (size_t) item_ct1.get_group(0) * (size_t) scratch_stride;
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][PC_CHUNK]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &srow = *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[PC_CHUNK]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2);
    const int lane = (int) sg.get_local_id()[0], warp = (int) sg.get_group_id()[0];
    const int SGS = (int) sg.get_local_range()[0], NW = THREADS / SGS;
    const int c0 = chunk * PC_CHUNK;
    const int n_here = sycl::min(PC_CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    for (int c = t; c < PC_CHUNK; c += THREADS) {
        long long r = -1;
        if (c < n_here) {
            const int cell = ids[c0 + c];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);   // -1: masked (KV streaming)
        }
        srow[c] = r;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int c = t; c < PC_CHUNK; c += THREADS) {
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int h = 0; h < G; ++h) sp[h][c] = -FLT_MAX;
            continue;
        }
        float acc_s[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc_s[h] = 0.0f;
        for (int d = 0; d < HD; d += 16) {
            float kf[16];
            load16_k<KV_MODE>(p, srow[c], d, kf);
#pragma unroll
            for (int h = 0; h < G; ++h) {   // a 16-dimension block in four short chains, then into the cell's sum:
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;   // FP32 error near the warp kernel's tree
#pragma unroll
                for (int j = 0; j < 16; j += 4) {
                    b0 = sycl::fma(kf[j], sq[h][d + j], b0);
                    b1 = sycl::fma(kf[j + 1], sq[h][d + j + 1], b1);
                    b2 = sycl::fma(kf[j + 2], sq[h][d + j + 2], b2);
                    b3 = sycl::fma(kf[j + 3], sq[h][d + j + 3], b3);
                }
                acc_s[h] += (b0 + b1) + (b2 + b3);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) sp[h][c] = acc_s[h] * scale;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int h = warp; h < G; h += NW) {
        float m = -FLT_MAX;
        for (int c = lane; c < PC_CHUNK; c += SGS) m = sycl::fmax(m, sp[h][c]);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float l = 0.0f;
        for (int c = lane; c < PC_CHUNK; c += SGS) {
            const float e = (c < n_here && srow[c] >= 0) ? sycl::native::exp(sp[h][c] - m) : 0.0f;
            sp[h][c] = e;
            l += e;
        }
        l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        const float v = load1_v<KV_MODE == 3 ? 2 : KV_MODE>(p, srow[c], t);
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE>
void launch_percell(dpct::queue_ptr st, const float* q, const QsaAttnPools& pools, const int32_t* ids,
                    const int32_t* steps, int64_t cap, const QsaShapes& s, float* part_acc, float* part_m, float* part_l,
                    int n_chunks, long long stride, int64_t n_q, float scale) {
    const sycl::range<3> grid((size_t) n_q, (size_t) s.n_head_kv, (size_t) n_chunks);
    st->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, THREADS), sycl::range<3>(1, 1, THREADS)),
                     [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                         attn_chunk_percell_kernel<KV_MODE>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                            scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
                     });
}

}  // namespace


static void launch_qsa_chunk_down(int,const float*,const QsaAttnPools&,const int32_t*,const int32_t*,int64_t,
                           const QsaShapes&,float*,float*,int64_t,void*);
bool qsa_chunk_down_supported(int64_t cap, int64_t n_q, const QsaShapes& s, void* stream) {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_SYCL_QSA_CHUNK_DOWN");
        return !v || v[0] == '1';
    }();
    const bool batch = n_q >= 1 && n_q <= 6;
    const bool shape = (cap == 2051 && (batch || n_q == 32)) || (cap == 32768 && batch);
    return enabled && shape && s.n_head == 24 && s.n_head_kv == 2 && s.page_size == 4 &&
        strata::intel_gpu_gen(strata::q_of(stream)->get_device()) == strata::IntelGpuGen::Battlemage;
}
// BEGIN PRIVATE QSA TWO STAGE ENTRY
uint64_t qsa_merge_two_stage_scratch_floats(int n_chunks) {
    if (n_chunks <= 0 || n_chunks > INT_MAX / 24)
        throw std::invalid_argument("QSA two-stage scratch arguments");
    const uint64_t n_groups = (uint64_t(n_chunks) + 31) / 32;
    return n_groups * 24 * (HD + 2) + 64;
}

void qsa_merge_two_stage_probe(int mode, const float* scratch, int n_chunks,
                               float* group_scratch, float* out, void* stream) {
    if (mode < 0 || mode > 1 || !scratch || !out || !stream || n_chunks <= 0 ||
        n_chunks > INT_MAX / 24 || (mode == 1 && !group_scratch))
        throw std::invalid_argument("QSA two-stage diagnostic arguments");
    auto* queue = strata::q_of(stream);
    if (!queue->is_in_order())
        throw std::invalid_argument("QSA two-stage probe requires an in-order queue");
    const float* part_m = scratch + size_t(n_chunks) * 24 * HD;
    const float* part_l = part_m + size_t(n_chunks) * 24;
    if (mode == 0) {
        launch_qsa_merge_two_stage_original(scratch, part_m, part_l, n_chunks, out, queue);
        return;
    }
    const int n_groups = (n_chunks + 31) / 32;
    float* group_m = group_scratch + size_t(n_groups) * 24 * HD;
    float* group_l = group_m + size_t(n_groups) * 24;
    launch_qsa_merge_two_stage_partial(scratch, part_m, part_l, n_chunks,
                                      group_scratch, group_m, group_l, n_groups, out, queue);
}

// END PRIVATE QSA TWO STAGE ENTRY
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    strata::workload_trace::Scope trace("decode.qsa_attention", stream, n_q, s.n_head, s.head_dim,
                                        cap, s.n_head_kv, kv_mode);
    if (qsa_chunk_down_supported(cap, n_q, s, stream) && kv_mode == 1 && pools.v_q) {
        launch_qsa_chunk_down(1, q, pools, ids, steps, cap, s, scratch, attn, n_q, stream);
        return;
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv,
                          (unsigned)n_q);
    dpct::queue_ptr st = strata::q_of(stream);
    // S25: the score phase one cell per thread (bit-identical scores, no shuffle reductions).  Prompt batches take
    // it by default: on 2x Arc Pro B65 it reads a 4,106-token prompt 5.2% faster, 8,191 tokens 5.2% and 16,380
    // tokens 4.8%, with the same tokens; the decode window's few queries per call are 1.4% slower with it (56.6-57.0
    // against 55.9-56.3 tok/s), so the choice follows the query count.  It is a Battlemage (Xe2) default only: on an Arc
    // A750 (Alchemist) the same switch reads a 4,106-token prompt 0.7% slower (three interleaved runs, 4 of 5 pairs
    // faster with it off), so the generation comes from the device's architecture id (strata::intel_gpu_gen: the
    // runtime's architecture enum first, the name only as a fallback), never from the name alone.
    // STRATA_ATTN_LANECELL=1/0 forces either way on any device.
    static const int lane_cell_env = [] {
        const char* v = std::getenv("STRATA_ATTN_LANECELL");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const bool lane_cell = lane_cell_env >= 0
        ? lane_cell_env != 0
        : (n_q >= 16 && strata::intel_gpu_gen(st->get_device()) == strata::IntelGpuGen::Battlemage);
#define STRATA_ATTN_LC(M)                                                      \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});  \
                                                                               \
        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                   \
            auto q_ct0 = q;                                                    \
            QsaAttnPools pools_ct1 = pools;                                    \
            auto ids_ct2 = ids;                                                \
            auto steps_ct3 = steps;                                            \
            auto s_n_head_kv_ct4 = (int)s.n_head_kv;                           \
            auto s_page_size_ct5 = (int)s.page_size;                           \
            auto scale_ct6 = scale;                                            \
            auto part_acc_ct7 = part_acc;                                      \
            auto part_m_ct8 = part_m;                                          \
            auto part_l_ct9 = part_l;                                          \
            auto n_chunks_ct10 = n_chunks;                                     \
            auto cap_ct11 = (int)cap;                                          \
            auto stride_ct12 = stride;                                         \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class attn_chunk_kernel_6e87d5,  \
                                              dpct_kernel_scalar<M>,           \
                                              dpct_kernel_scalar<true>>>(      \
                sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),           \
                                  sycl::range(1, 1, THREADS)),                 \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        attn_chunk_kernel<M, true>(                            \
                            q_ct0, pools_ct1, ids_ct2, steps_ct3,              \
                            s_n_head_kv_ct4, s_page_size_ct5, scale_ct6,       \
                            part_acc_ct7, part_m_ct8, part_l_ct9,              \
                            n_chunks_ct10, cap_ct11, stride_ct12);             \
                    });                                                        \
        });                                                                    \
    }
#define STRATA_ATTN_RSK(M, V)                                                      \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});  \
                                                                               \
        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                   \
            auto q_ct0 = q;                                                    \
            QsaAttnPools pools_ct1 = pools;                                    \
            auto ids_ct2 = ids;                                                \
            auto steps_ct3 = steps;                                            \
            auto s_n_head_kv_ct4 = (int)s.n_head_kv;                           \
            auto s_page_size_ct5 = (int)s.page_size;                           \
            auto scale_ct6 = scale;                                            \
            auto part_acc_ct7 = part_acc;                                      \
            auto part_m_ct8 = part_m;                                          \
            auto part_l_ct9 = part_l;                                          \
            auto n_chunks_ct10 = n_chunks;                                     \
            auto cap_ct11 = (int)cap;                                          \
            auto stride_ct12 = stride;                                         \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class attn_chunk_kernel_rs_7c2a91,  \
                                              dpct_kernel_scalar<M>,           \
                                              dpct_kernel_scalar<V>>>(      \
                sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),           \
                                  sycl::range(1, 1, THREADS)),                 \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        attn_chunk_kernel<M, false, V>(                            \
                            q_ct0, pools_ct1, ids_ct2, steps_ct3,              \
                            s_n_head_kv_ct4, s_page_size_ct5, scale_ct6,       \
                            part_acc_ct7, part_m_ct8, part_l_ct9,              \
                            n_chunks_ct10, cap_ct11, stride_ct12);             \
                    });                                                        \
        });                                                                    \
    }
#undef STRATA_ATTN_RSK_DEFINED
    // 151-b70: STRATA_ATTN_RS=2 (default): the probabilities in [cell][head] order and the value phase reading a cell's 12 as three vectors - the same
    // bits (first-token logits md5 equal at 4K and 16K), +0.8..0.9% prompt tok/s on the B70; =1 adds a 16-shuffle reduce-scatter for the scores
    // (bit-exact too, but -1.5% here: the score phase is not shuffle bound); =0 the layout before
    static const int rs_on = [] { const char* v = std::getenv("STRATA_ATTN_RS"); return v ? std::atoi(v) : 2; }();
    if (lane_cell) {
        if (kv_mode == 3) { STRATA_ATTN_LC(3); }
        else if (kv_mode == 2) { STRATA_ATTN_LC(2); }
        else if (kv_mode == 1) { STRATA_ATTN_LC(1); }
        else { STRATA_ATTN_LC(0); }
    } else if (rs_on == 1) {
        if (kv_mode == 3) { STRATA_ATTN_RSK(3, 1); }
        else if (kv_mode == 2) { STRATA_ATTN_RSK(2, 1); }
        else if (kv_mode == 1) { STRATA_ATTN_RSK(1, 1); }
        else { STRATA_ATTN_RSK(0, 1); }
    } else if (rs_on == 2) {
        if (kv_mode == 3) { STRATA_ATTN_RSK(3, 2); }
        else if (kv_mode == 2) { STRATA_ATTN_RSK(2, 2); }
        else if (kv_mode == 1) { STRATA_ATTN_RSK(1, 2); }
        else { STRATA_ATTN_RSK(0, 2); }
    } else
#undef STRATA_ATTN_LC
#undef STRATA_ATTN_RSK
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_dfacfc,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_b0d448,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_2ad0a4,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_5c04f7,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        if (merge_active_on(st->get_device()) && n_q <= 8) {
            strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_merge_active>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_q, (unsigned)s.n_head) *
                        sycl::range(1, 1, HD),
                    sycl::range(1, 1, HD)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    attn_merge_active_kernel(part_acc, part_m, part_l, n_chunks,
                                             attn, stride, steps);
                });
        } else {
            strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_merge_kernel_338175>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_q, (unsigned)s.n_head) *
                        sycl::range(1, 1, HD),
                    sycl::range(1, 1, HD)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn,
                                      stride);
                });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

namespace {
template <int M>
void attn_rows_chunks(dpct::queue_ptr st, const dpct::dim3& grid, const float* q, const QsaAttnPools& pools,
                      const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size, float scale,
                      float* part_acc, float* part_m, float* part_l, int n_chunks, int cap, long long stride,
                      const QsaAttnPoolsRows& rows) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});
    const QsaAttnPools p0 = pools;
    const QsaAttnPoolsRows rr = rows;
    strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_rows_chunk_kernel_9f2, dpct_kernel_scalar<M>>>(
        sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS), sycl::range(1, 1, THREADS)), exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            attn_chunk_kernel<M, false, 2>(q, p0, ids, steps, n_kv_heads, page_size, scale, part_acc, part_m, part_l,
                                           n_chunks, cap, stride, rr);
        });
}
}  // namespace

void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const QsaAttnPools& pools = rows.p[0];
    if (rows.n != n_q || n_q > kAttnRowsMax || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 ||
        !scratch || !ids || !steps) {
        std::fprintf(stderr, "qsa_decode_attn_batch_rows: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    static const int rs_on = [] { const char* v = std::getenv("STRATA_ATTN_RS"); return v ? std::atoi(v) : 2; }();
    static const int lane_cell_env = [] {
        const char* v = std::getenv("STRATA_ATTN_LANECELL");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    if (rs_on != 2 || lane_cell_env == 1) {   // a non-default per-row kernel is selected: the per-row calls
        for (int64_t z = 0; z < n_q; ++z)
            qsa_decode_attn_batch(q + (size_t) z * s.n_head * HD, rows.p[z], ids + (size_t) z * cap,
                                  steps + (size_t) z * kStepCount, cap, s, scratch + (size_t) z * stride,
                                  attn + (size_t) z * s.n_head * HD, 1, stream);
        return;
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv, (unsigned)n_q);
    dpct::queue_ptr st = strata::q_of(stream);
    if (kv_mode == 3) attn_rows_chunks<3>(st, grid, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, rows);
    else if (kv_mode == 2) attn_rows_chunks<2>(st, grid, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, rows);
    else if (kv_mode == 1) attn_rows_chunks<1>(st, grid, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, rows);
    else attn_rows_chunks<0>(st, grid, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, rows);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_merge_rows_kernel_9f2>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)n_q, (unsigned)s.n_head) * sycl::range(1, 1, HD),
                              sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn, stride);
            });
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    if(qsa_chunk_down_supported(cap, 1, s, stream)&&kv_mode==1){
        launch_qsa_chunk_down(1,q,pools,ids,step,cap,s,scratch,attn,1,stream);return;
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv);
    dpct::queue_ptr st = strata::q_of(stream);
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_f262e6,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_4fd923,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_f6855e,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_chunk_kernel_7c8e09,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::kprof::kp(st, __FILE__, __func__, __LINE__)->template parallel_for<dpct_kernel_name<class attn_merge_kernel_c055e7>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)s.n_head) *
                                  sycl::range(1, 1, HD),
                              sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn, 0);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

// SYCL port, prompt path only (prefill.cpp; decode and the draft layer keep qsa_decode_attn_batch): the chunk
// attention with the per-cell score kernel above, 128-cell chunks in the same scratch as the warp kernel (its stride
// is the 64-cell layout's, so it fits). FP32, another summation order than the warp kernel: not bitwise.
void qsa_prompt_attn_percell(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                             int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    const float scale = 1.0f / sqrtf((float) HD);
    dpct::queue_ptr st = strata::q_of(stream);
    const int nc = (int) ((cap + PC_CHUNK - 1) / PC_CHUNK);
    float* pa = scratch;
    float* pm = scratch + (size_t) nc * s.n_head * HD;
    float* pl = pm + (size_t) nc * s.n_head;
    if (kv_mode == 3) launch_percell<3>(st, q, pools, ids, steps, cap, s, pa, pm, pl, nc, stride, n_q, scale);
    else if (kv_mode == 2) launch_percell<2>(st, q, pools, ids, steps, cap, s, pa, pm, pl, nc, stride, n_q, scale);
    else if (kv_mode == 1) launch_percell<1>(st, q, pools, ids, steps, cap, s, pa, pm, pl, nc, stride, n_q, scale);
    else launch_percell<0>(st, q, pools, ids, steps, cap, s, pa, pm, pl, nc, stride, n_q, scale);
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) n_q, (size_t) s.n_head) * sycl::range<3>(1, 1, HD),
                                       sycl::range<3>(1, 1, HD)),
                     [=](sycl::nd_item<3>) { attn_merge_kernel(pa, pm, pl, nc, attn, stride); });
}

}  // namespace strata::kernels

#include <stdexcept>

namespace strata::kernels {
namespace {
__dpct_inline__ float lane0_sum(float value){
    auto group=sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)value+=sycl::shift_group_left(group,value,offset);
    return value;
}
template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
chunk_down_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
chunk_down_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = lane0_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = lane0_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[srow[c] * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[srow[c] * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}


class QsaChunkDown;
}
namespace {
__dpct_inline__ void private_load8_vector_codes(const QsaAttnPools& p,bool value,long long row,int d0,float* out){
    const int8_t* codes=(value?p.v_q:p.k_q)+row*HD+d0;
    const uint16_t sbits=(value?p.v_scale:p.k_scale)[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP];
    const float sc=sycl::vec<sycl::half,1>(sycl::bit_cast<sycl::half,unsigned short>(sbits)).convert<float,sycl::rounding_mode::automatic>()[0];
    const auto raw=*reinterpret_cast<const sycl::vec<int8_t,8>*>(codes);
    const auto converted=raw.convert<float,sycl::rounding_mode::automatic>();
#pragma unroll
    for(int j=0;j<8;++j)out[j]=converted[j]*sc;
}
template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
chunk_down_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
chunk_vector_codes_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        private_load8_vector_codes(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = lane0_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = lane0_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c0 = 0; c0 < n_here; c0 += 8) {
        long long rows[8]; bool full = c0 + 8 <= n_here;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rows[j] = c0 + j < n_here ? srow[c0 + j] : -1;
            full = full && rows[j] >= 0;
        }
        if (full) {
            float values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
                values[j] = v;
            }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 p0 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0]);
            const sycl::float4 p1 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0 + 4]);
            acc[h] = sycl::fma(p0[0], values[0], acc[h]);
            acc[h] = sycl::fma(p0[1], values[1], acc[h]);
            acc[h] = sycl::fma(p0[2], values[2], acc[h]);
            acc[h] = sycl::fma(p0[3], values[3], acc[h]);
            acc[h] = sycl::fma(p1[0], values[4], acc[h]);
            acc[h] = sycl::fma(p1[1], values[5], acc[h]);
            acc[h] = sycl::fma(p1[2], values[6], acc[h]);
            acc[h] = sycl::fma(p1[3], values[7], acc[h]);
        }

        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
                if (row < 0) continue;
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c0 + j], v, acc[h]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
chunk_down_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
b60_qsa_float4_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G + i / HD) * HD +
            ((i % HD) % 128 / 4) * 8 + ((i % HD) / 128) * 4 + (i % 4)];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        private_load8_vector_codes(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 4]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][128 + lane * 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = lane0_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = lane0_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c0 = 0; c0 < n_here; c0 += 8) {
        long long rows[8]; bool full = c0 + 8 <= n_here;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rows[j] = c0 + j < n_here ? srow[c0 + j] : -1;
            full = full && rows[j] >= 0;
        }
        if (full) {
            float values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
                values[j] = v;
            }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 p0 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0]);
            const sycl::float4 p1 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0 + 4]);
            acc[h] = sycl::fma(p0[0], values[0], acc[h]);
            acc[h] = sycl::fma(p0[1], values[1], acc[h]);
            acc[h] = sycl::fma(p0[2], values[2], acc[h]);
            acc[h] = sycl::fma(p0[3], values[3], acc[h]);
            acc[h] = sycl::fma(p1[0], values[4], acc[h]);
            acc[h] = sycl::fma(p1[1], values[5], acc[h]);
            acc[h] = sycl::fma(p1[2], values[6], acc[h]);
            acc[h] = sycl::fma(p1[3], values[7], acc[h]);
        }

        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
                if (row < 0) continue;
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c0 + j], v, acc[h]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
chunk_down_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
b60_qsa_head6_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int H=6;const int headbase=int(item_ct1.get_group(2)%2)*H;
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[H][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[H][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2)/2, kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < H) { part_m[slot * G + headbase + t] = -FLT_MAX; part_l[slot * G + headbase + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < H * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G + headbase + i / HD) * HD +
            ((i % HD) % 128 / 4) * 8 + ((i % HD) / 128) * 4 + (i % 4)];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < H) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        private_load8_vector_codes(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 4]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][128 + lane * 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = lane0_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < H; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = lane0_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + headbase + h] = m; part_l[slot * G + headbase + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[H];
#pragma unroll
    for (int h = 0; h < H; ++h) acc[h] = 0.0f;
    for (int c0 = 0; c0 < n_here; c0 += 8) {
        long long rows[8]; bool full = c0 + 8 <= n_here;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rows[j] = c0 + j < n_here ? srow[c0 + j] : -1;
            full = full && rows[j] >= 0;
        }
        if (full) {
            float values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
                values[j] = v;
            }
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const sycl::float4 p0 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0]);
            const sycl::float4 p1 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0 + 4]);
            acc[h] = sycl::fma(p0[0], values[0], acc[h]);
            acc[h] = sycl::fma(p0[1], values[1], acc[h]);
            acc[h] = sycl::fma(p0[2], values[2], acc[h]);
            acc[h] = sycl::fma(p0[3], values[3], acc[h]);
            acc[h] = sycl::fma(p1[0], values[4], acc[h]);
            acc[h] = sycl::fma(p1[1], values[5], acc[h]);
            acc[h] = sycl::fma(p1[2], values[6], acc[h]);
            acc[h] = sycl::fma(p1[3], values[7], acc[h]);
        }

        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
                if (row < 0) continue;
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
                for (int h = 0; h < H; ++h) acc[h] = sycl::fma(sp[h][c0 + j], v, acc[h]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < H; ++h) part_acc[((size_t) slot * G + headbase + h) * HD + t] = acc[h];
}

class B60QsaHead6;
template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
chunk_down_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
b60_qsa_local_fence_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G + i / HD) * HD +
            ((i % HD) % 128 / 4) * 8 + ((i % HD) / 128) * 4 + (i % 4)];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        private_load8_vector_codes(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 4]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][128 + lane * 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = lane0_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = lane0_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c0 = 0; c0 < n_here; c0 += 8) {
        long long rows[8]; bool full = c0 + 8 <= n_here;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rows[j] = c0 + j < n_here ? srow[c0 + j] : -1;
            full = full && rows[j] >= 0;
        }
        if (full) {
            float values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
                values[j] = v;
            }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 p0 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0]);
            const sycl::float4 p1 = *reinterpret_cast<const sycl::float4*>(&sp[h][c0 + 4]);
            acc[h] = sycl::fma(p0[0], values[0], acc[h]);
            acc[h] = sycl::fma(p0[1], values[1], acc[h]);
            acc[h] = sycl::fma(p0[2], values[2], acc[h]);
            acc[h] = sycl::fma(p0[3], values[3], acc[h]);
            acc[h] = sycl::fma(p1[0], values[4], acc[h]);
            acc[h] = sycl::fma(p1[1], values[5], acc[h]);
            acc[h] = sycl::fma(p1[2], values[6], acc[h]);
            acc[h] = sycl::fma(p1[3], values[7], acc[h]);
        }

        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const long long row = rows[j];
                if (row < 0) continue;
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[row * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[row * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[row * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c0 + j], v, acc[h]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

class B60QsaLocalFence;
inline bool b60_next_qsa_pool(){static const bool on=[] {const char*p=std::getenv("STRATA_PRIVATE_B60_NEXT_POOL");return p&&p[0]=='1';}();return on;}
class B60QsaFloat4;
thread_local bool b60_qsa_float4_probe_mode=false;

class QsaVectorCodes;
}
static void launch_normal_qsa_vector(int mode,const float* q,const QsaAttnPools& pools,
    const int32_t* ids,const int32_t* steps,int64_t cap,const QsaShapes& sh,
    float* scratch,float* attn,int64_t nq,void* stream,float* secondary = nullptr){
    if(nq<=0)return;
    const int chunks=int((cap+CHUNK-1)/CHUNK);
    const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
    auto* pm=scratch+size_t(chunks)*sh.n_head*HD;
    auto* pl=pm+size_t(chunks)*sh.n_head;
    auto* queue=strata::q_of(stream);const float scale=1.f/sqrtf(float(HD));
    auto props=sycl::ext::oneapi::experimental::properties{};
    if(b60_layout_requested()&&b60_next_qsa_pool()&&((cap==2051&&nq==1&&(!b60_single_actor_requested()&&queue->get_device()==static_cast<const sycl::device&>(dpct::get_device(0)))) || (cap==2051&&nq==1&&(b60_single_actor_requested()||queue->get_device()==static_cast<const sycl::device&>(dpct::get_device(1)))))) {
    queue->parallel_for<B60QsaHead6>(
        sycl::nd_range<3>(sycl::range<3>(nq,sh.n_head_kv,size_t(chunks)*2*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{
            b60_qsa_head6_kernel<1>(q,pools,ids,steps,sh.n_head_kv,sh.page_size,scale,scratch,pm,pl,chunks,cap,stride);
        });
    } else
    if(b60_layout_requested()&&b60_next_qsa_pool()&&((cap==2051&&nq==4&&(!b60_single_actor_requested()&&queue->get_device()==static_cast<const sycl::device&>(dpct::get_device(0)))) || (cap==32768&&nq==1&&(b60_single_actor_requested()||queue->get_device()==static_cast<const sycl::device&>(dpct::get_device(1)))))) {
    queue->parallel_for<B60QsaLocalFence>(
        sycl::nd_range<3>(sycl::range<3>(nq,sh.n_head_kv,size_t(chunks)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{
            b60_qsa_local_fence_kernel<1>(q,pools,ids,steps,sh.n_head_kv,sh.page_size,scale,scratch,pm,pl,chunks,cap,stride);
        });
    } else
    if(b60_qsa_float4_probe_mode || (b60_layout_requested() && ((cap==2051&&nq>=1&&nq<=4)||(cap==32768&&nq==1)))) {
    queue->parallel_for<B60QsaFloat4>(
        sycl::nd_range<3>(sycl::range<3>(nq,sh.n_head_kv,size_t(chunks)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{
            b60_qsa_float4_kernel<1>(q,pools,ids,steps,sh.n_head_kv,sh.page_size,scale,scratch,pm,pl,chunks,cap,stride);
        });
    } else {
    queue->parallel_for<QsaVectorCodes>(
        sycl::nd_range<3>(sycl::range<3>(nq,sh.n_head_kv,size_t(chunks)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{
            chunk_vector_codes_kernel<1>(q,pools,ids,steps,sh.n_head_kv,sh.page_size,scale,scratch,pm,pl,chunks,cap,stride);
        });
    }
    if (secondary) {
        qsa_merge_two_stage_probe(1, scratch, chunks, secondary, attn, stream);
        return;
    }
    queue->parallel_for<dpct_kernel_name<class qsa_vector_codes_merge>>(
        sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(sh.n_head)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>){attn_merge_kernel(scratch,pm,pl,chunks,attn,stride);});
}
static void launch_qsa_chunk_down(int mode,const float* q,const QsaAttnPools& pools,
    const int32_t* ids,const int32_t* steps,int64_t cap,const QsaShapes& s,
    float* scratch,float* attn,int64_t nq,void* stream){
    if(mode<0||mode>1)throw std::invalid_argument("QSA chunk down mode");
    if(mode==0||!pools.k_q||!pools.v_q||pools.k_q4||pools.v_q4){
        qsa_decode_attn_batch(q,pools,ids,steps,cap,s,scratch,attn,nq,stream);return;
    }
    if(nq<=0)return;
    static const bool vector_enabled = [] {
        const char* v = std::getenv("STRATA_SYCL_QSA_VALUE8_K8_VEC");
        return !v || v[0] == '1';
    }();
    const bool vector_shape=(cap==2051&&(nq==1||nq==2||nq==3||nq==4||nq==32))||(cap==32768&&nq==1);
    if(vector_enabled&&vector_shape){launch_normal_qsa_vector(1,q,pools,ids,steps,cap,s,scratch,attn,nq,stream);return;}
    const int chunks=int((cap+CHUNK-1)/CHUNK);
    const long long stride=qsa_decode_attn_scratch_floats(cap,s);
    auto* part_m=scratch+size_t(chunks)*s.n_head*HD;
    auto* part_l=part_m+size_t(chunks)*s.n_head;
    auto* queue=strata::q_of(stream);
    const float scale=1.f/sqrtf(float(HD));
    auto props=sycl::ext::oneapi::experimental::properties{};
    queue->parallel_for<QsaChunkDown>(
        sycl::nd_range<3>(sycl::range<3>(nq,s.n_head_kv,size_t(chunks)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{
            chunk_down_kernel<1>(q,pools,ids,steps,s.n_head_kv,s.page_size,scale,
                scratch,part_m,part_l,chunks,cap,stride);
        });
    queue->parallel_for<dpct_kernel_name<class qsa_chunk_down_merge>>(
        sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(s.n_head)*THREADS),sycl::range<3>(1,1,THREADS)),props,
        [=](sycl::nd_item<3>){attn_merge_kernel(scratch,part_m,part_l,chunks,attn,stride);});
}

bool qsa_decode_attn_batch_try_two_stage(const float* q, const QsaAttnPools& pools,
    const int32_t* ids, const int32_t* steps, int64_t cap, const QsaShapes& s,
    float* scratch, float* attn, int64_t n_q, float* secondary,
    uint64_t secondary_floats, void* stream) {
    // The sole caller owns parts_ and a one-query scratch. Keep every other
    // caller, geometry, storage mode and existing dispatch fallback unchanged.
    if (cap != 32768 || n_q != 1 || s.n_head != 24 || s.n_head_kv != 2 ||
        s.head_dim != 256 || s.page_size != 4 || !q || !ids || !steps ||
        !scratch || !attn || !secondary || !pools.page_table ||
        !pools.k_q || !pools.v_q || !pools.k_scale || !pools.v_scale ||
        pools.k_q4 || pools.v_q4 || !qsa_chunk_down_supported(cap, n_q, s, stream))
        return false;
    static const bool vector_enabled = [] {
        const char* v = std::getenv("STRATA_SYCL_QSA_VALUE8_K8_VEC");
        return !v || v[0] == '1';
    }();
    auto* queue = strata::q_of(stream);
    const int chunks = int((cap + CHUNK - 1) / CHUNK);
    const uint64_t used = qsa_merge_two_stage_scratch_floats(chunks);
    if (!vector_enabled || !queue->is_in_order() || secondary_floats < used)
        return false;
    const auto disjoint = [](const void* a, uint64_t an, const void* b, uint64_t bn) {
        const uintptr_t av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
        if (av > UINTPTR_MAX - an || bv > UINTPTR_MAX - bn) return false;
        return av + an <= bv || bv + bn <= av;
    };
    const uint64_t bytes = used * sizeof(float);
    if (!disjoint(secondary, bytes, scratch, qsa_decode_attn_scratch_floats(cap, s) * sizeof(float)) ||
        !disjoint(secondary, bytes, attn, uint64_t(24 * HD) * sizeof(float)) ||
        !disjoint(secondary, bytes, q, uint64_t(24 * HD) * sizeof(float)))
        return false;
    launch_normal_qsa_vector(1, q, pools, ids, steps, cap, s, scratch, attn,
                             n_q, stream, secondary);
    return true;
}
}

namespace strata::kernels {
void qsa_query_float4_probe(int mode,const float*q,const QsaAttnPools&p,const int32_t*ids,const int32_t*steps,int64_t cap,const QsaShapes&sh,float*scratch,float*attn,int64_t T,void*stream,float*secondary){
 if(mode<0||mode>1)throw std::invalid_argument("B60 query float4 mode");
 struct Restore {bool previous;~Restore(){b60_qsa_float4_probe_mode=previous;}} restore{b60_qsa_float4_probe_mode};
 b60_qsa_float4_probe_mode=mode==1;
 launch_normal_qsa_vector(1,q,p,ids,steps,cap,sh,scratch,attn,T,stream,secondary);
}
}
