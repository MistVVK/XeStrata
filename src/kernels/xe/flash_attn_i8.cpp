// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2024 SageAttention team
// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES (comfy-kitchen)
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/flash_attn_i8.cpp - flash_attn's int8 path (flash_attn.hpp: flash_attn_i8): the attention's
// unmasked rows with Q . K^T and P . V on the int8 matrix instructions (SageAttention's scheme), for every model that
// takes it.
#include "strata/kernels/flash_attn.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "attn_parts.hpp"
#include "amd_wmma.hpp"
#include "cuda_intrinsics.hpp"
#include "nv_mma.hpp"
#include "device_target.hpp"

#if STRATA_HIP_KITCHEN
#include "hip_kitchen.hpp"
#endif
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace strata::kernels {

namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using xe::attn::built;   // the attention kernels' shared parts (attn_parts.hpp)
using f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using f16_gpu::f32_from_f16;

constexpr int D = 128;      // head_dim
// ---- the int8 attention (SageAttention's scheme), the unmasked rows: Q . K^T and P . V both on the int8 matrix
// instructions, with scales.  K has its mean over the keys taken off (a query's scores all move by q . mean, which the
// softmax does not see) and is made int8 with a scale a (key, head), Q a scale a (row, head) with the softmax's scale
// in it; V a scale a (head, dim) over the keys, stored transposed [head][dim][key] for P . V's operand; P, at most 1
// against the row's running maximum (no lazy reference here: P is int8), a scale a (row, chunk) from the chunk's
// largest.  The sums are int32, turned to FP32 with the scales into the output's accumulators.  A work-group takes 128
// query rows of a head (8 waves of 16) and shares each chunk's 64 keys' K and V^T in local memory.
// RDNA3 and RDNA4's WMMA (16 x 16 x 16 iu8 on 32 lanes, the layouts of amd_wmma.hpp and iq_mmq.cpp: Kernel) and
// NVIDIA's mma.sync from sm_80 on (NvKernel); elsewhere not built, and qi_attention keeps the 16-bit path.
namespace i8 {

constexpr int CH = 32;            // keys a chunk
constexpr int QT = 2;             // query tiles of 16 a wave: K's and V^T's operands read once for both
constexpr int NW = 4;             // waves a work-group: NW * 16 * QT query rows
constexpr int KLD = D + 16;       // a key's row of K in local memory (bytes)
constexpr int VLD = CH;           // a dim's row of V^T in local memory (bytes; padded to CH + 16 it was 4% slower on
                                  // the RX 9060 XT, 2026-10-29: a smaller work-group's local memory)
constexpr int SLICES = 32;        // the key stats' partial sums: slices of the keys

// the per-device buffers, grown to the largest T asked: int8 Q and K [T][heads D], V^T [heads][Tp / CH][D][CH], their
// scales; Q and its scales 8 rows past Tp (JmKernel's tiles from a first row r0 leaves anywhere).  A causal attention
// (run_causal) asks for its queries' Tq rows of Cq apart, fewer than its keys', and at most a 32nd of the GPU's
// memory (fit: past that the caller's 16-bit kernel, not a later allocation of the engine's failing)
struct Bufs {
    int8_t *q = nullptr, *k = nullptr, *vt = nullptr;
    float *sq = nullptr, *sk = nullptr, *sv = nullptr, *mean = nullptr, *part = nullptr, *sw = nullptr;
    int64_t cap_t = 0, cap_c = 0, cap_qt = 0, cap_qc = 0, cap_h = 0;
    size_t held = 0;   // the bytes held now (given back before the larger ones are taken)
    void release() {
        if (q != nullptr) core::graph_epoch_bump();
        for (void* p : {(void*) q, (void*) k, (void*) vt, (void*) sq, (void*) sk, (void*) sv, (void*) mean, (void*) part,
                        (void*) sw})
            if (p != nullptr) strata::gpu::free(p);
        *this = Bufs{};
    }
    bool grow(int64_t T, int64_t C, int64_t heads, int64_t Tq = -1, int64_t Cq = -1, int64_t hq = -1,
              bool fit = false) {
        if (Tq < 0) Tq = T, Cq = C, hq = heads;
        const int64_t Tp = (T + CH - 1) / CH * CH, Tqp = (Tq + CH - 1) / CH * CH;
        const int64_t hs = std::max({heads, hq, cap_h});
        if (Tp <= cap_t && C <= cap_c && Tqp <= cap_qt && Cq <= cap_qc && hs <= cap_h) return true;
        const int64_t qt = std::max(Tqp, cap_qt), qc = std::max(Cq, cap_qc), t = std::max(Tp, cap_t),
                      c = std::max(C, cap_c);
        const size_t need = (size_t) ((qt + 8) * qc + 2 * t * c + (qt + 8) * hs * 4 + t * hs * 4 + 3 * c * 4 +
                                      c * 2 * SLICES * 4);
        size_t free_b = 0, total_b = 0;
        if (fit && (!strata::gpu::mem_info(&free_b, &total_b) || need > total_b / 32 || free_b + held < need))
            return false;
        release();
        if (!strata::gpu::alloc_device(&q, (size_t) ((qt + 8) * qc)) ||
            !strata::gpu::alloc_device(&k, (size_t) (t * c)) || !strata::gpu::alloc_device(&vt, (size_t) (t * c)) ||
            !strata::gpu::alloc_device(&sq, (size_t) ((qt + 8) * hs) * 4) ||
            !strata::gpu::alloc_device(&sk, (size_t) (t * hs) * 4) || !strata::gpu::alloc_device(&sv, (size_t) c * 4) ||
            !strata::gpu::alloc_device(&mean, (size_t) c * 4) || !strata::gpu::alloc_device(&sw, (size_t) c * 4) ||
            !strata::gpu::alloc_device(&part, (size_t) 2 * SLICES * c * 4)) {
            release();
            return false;
        }
        cap_t = t, cap_c = c, cap_qt = qt, cap_qc = qc, cap_h = hs;
        held = need;
        return true;
    }
};

Bufs& bufs_of(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<Bufs>> of;
    return *of.get(q.get_device(), [] { return std::make_shared<Bufs>(); });
}

// a causal attention's: a stream's own (two engines on a GPU, an LLM's prompt beside an image's, at once), the
// queue's belonging (Runtime::queue_item): freed when its stream is destroyed - keyed by the bare queue address it
// would outlive the queue and meet the next one there
struct CausalOf {
    explicit CausalOf(sycl::queue& q) : q(q) {}
    ~CausalOf() { b.release(); }
    sycl::queue q;
    std::mutex mu;
    Bufs b;
};
Bufs& causal_bufs_of(sycl::queue& q) {
    return core::Runtime::of_queue(q).queue_item<CausalOf>(q).b;
}

// NVIDIA's mma.sync from sm_80 on (NvKernel): m16n8k32 s8 . s8 into s32, ldmatrix for the B operands out of local
// memory and cp.async for the chunks into it (nv_mma.hpp)
#define STRATA_NV_I8 STRATA_NV_MMA
#if STRATA_NV_I8
namespace nv = kernels::xe::nv;
using nv::mma_s8;
using nv::mma_u8s8;
using nv::ldsm4;
using kernels::xe::byte_perm;
// 2^x flushing a denormal result to 0: sycl::native::exp2 is ex2.approx.f32 without .ftz, which sm_89 runs as a check,
// two multiplies and the instruction (20% of the kernel's instructions, RTX 4070, 2026-11-05); a probability below
// 2^-126 is 0 in int8 either way
inline float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#endif

// Which of the int8 kernels the device takes: 1 Kernel (RDNA3, RDNA4), 2 NvKernel (NVIDIA from sm_80 on, where the
// binary the runtime picked for it carries the kernel: an older target's code JIT-compiled for it does not), 3 and 4
// JmKernel (a GPU whose matrix engines report s8 . s8 into s32 at 8 x 16 x 32 on 16 lanes, or at 8 x 8 x 32 on 8:
// Intel's XMX), else 0
// DotKernel's kind where no other is taken: 0 for now - right on the B70 (qi_attn_parity) but 95 ms against oneDNN's
// 3.3 for 4115 rows of 32 heads (a row's softmax on one work-item, its operands read from local memory a word at a
// time), so that qi_attention's timing of it on a GPU's first shapes would only cost; it waits for its tiles in
// registers
constexpr int kDotKind = 0;

int jm_kind(const sycl::device& d) {
    if (!built(0)) return 0;
    for (const mx::combination& c : matrix_combinations(d)) {
        if (c.atype != mx::matrix_type::sint8 || c.btype != mx::matrix_type::sint8 ||
            c.ctype != mx::matrix_type::sint32 ||
            c.ksize != 32 || !(c.msize == 8 || (c.msize == 0 && c.max_msize >= 8)))
            continue;
        const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
        auto has = [&](size_t n) { return std::find(sgs.begin(), sgs.end(), n) != sgs.end(); };
        if (c.nsize == 16 && has(16)) return 3;
        if (c.nsize == 8 && has(8)) return 4;
    }
    return 0;
}

int kind_of(sycl::queue& q) {
    if (no_xmx()) return kDotKind;
    static core::PerDevice<int> of;
    return of.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        if (has_form(d, MatrixForm::wmma)) return 1;
        if (!has_form(d, MatrixForm::mma_s8)) {
            const int jm = jm_kind(d);
            return jm != 0 ? jm : kDotKind;
        }
        int* f = sycl::malloc_device<int>(1, q);
        if (f == nullptr) return 0;
        int v = 0;
        q.single_task([=] { *f = STRATA_NV_I8; }).wait();
        q.memcpy(&v, f, sizeof v).wait();
        sycl::free(f, q);
        return v == 1 ? 2 : kDotKind;
    });
}

// Where key_stats and prep_i8 read K and V: value (t, c) of the keys' rows [T][C] (c: a head's dims after the heads
// before it).  Dense: FP16 rows (the image engine's).  Paged: the plain models' pools (qsa_decode_attn.hpp), a cell
// t on page table[t / ps] at its KV head c / hd: FP16, int8 codes with an FP16 scale a 64 values, or Q4_0 blocks.
struct Dense {
    const uint16_t* p;
    Dense(const uint16_t* p_) : p(p_) {}
    float at(int64_t t, int64_t c, int64_t C) const { return f32_from_f16(p[t * C + c]); }
    // a lane's U values from column c (zeros where !in)
    template <int U> void load(int64_t t, int64_t c, int64_t C, bool in, float (&x)[U]) const {
#pragma unroll
        for (int w = 0; w < U / 4; ++w) {
            const uint16_t* r = p + t * C + c + (int64_t) 4 * w;
            const sycl::vec<uint16_t, 4> v =
                in ? *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(r) : sycl::vec<uint16_t, 4>(0);
#pragma unroll
            for (int u = 0; u < 4; ++u) x[4 * w + u] = f32_from_f16(v[u]);
        }
    }
};
struct Paged {
    const uint16_t* f16 = nullptr;
    const int8_t* i8 = nullptr;
    const uint16_t* sc = nullptr;   // i8's scales
    const uint8_t* q4 = nullptr;
    const int32_t* table = nullptr;
    int64_t kv_heads = 0, ps = 0, hd = 0;
    float at(int64_t t, int64_t c, int64_t) const {
        const int64_t h = c / hd, d = c - h * hd, cell = ((int64_t) table[t / ps] * kv_heads + h) * ps + t % ps;
        if (f16 != nullptr) return f32_from_f16(f16[cell * hd + d]);
        if (i8 != nullptr) return (float) i8[cell * hd + d] * f32_from_f16(sc[cell * (hd / 64) + d / 64]);
        const uint8_t* blk = q4 + (cell * (hd / 32) + d / 32) * 18;   // block_q4_0 (kv_q4.hpp): its scale, 16 bytes
        const int j = (int) (d % 32);
        const int code = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
        return (float) (code - 8) * f32_from_f16((uint16_t) (blk[0] | blk[1] << 8));
    }
    // a lane's U values (within one head and, for i8, one group of 64): the cell found once
    template <int U> void load(int64_t t, int64_t c, int64_t C, bool in, float (&x)[U]) const {
        if (!in) {
#pragma unroll
            for (int u = 0; u < U; ++u) x[u] = 0.f;
            return;
        }
        const int64_t h = c / hd, d = c - h * hd, cell = ((int64_t) table[t / ps] * kv_heads + h) * ps + t % ps;
        if (f16 != nullptr) {
            Dense(f16 + cell * hd + d).load(0, 0, 0, true, x);
        } else if (i8 != nullptr) {
            const float m = f32_from_f16(sc[cell * (hd / 64) + d / 64]);
#pragma unroll
            for (int w = 0; w < U / 4; ++w) {
                const auto v = *reinterpret_cast<const sycl::vec<int8_t, 4>*>(i8 + cell * hd + d + (int64_t) 4 * w);
#pragma unroll
                for (int u = 0; u < 4; ++u) x[4 * w + u] = (float) v[u] * m;
            }
        } else {
#pragma unroll
            for (int u = 0; u < U; ++u) x[u] = at(t, c + u, C);
        }
    }
};

// the keys' sums and the values' largest magnitudes over slices of the keys: part [2][SLICES][C]
template <class Src = Dense>
void key_stats(sycl::queue& q, std::type_identity_t<const Src&> k16, std::type_identity_t<const Src&> v16, int64_t T,
               int64_t C, float* part, float* mean, float* sv) {
    // mean null: V's alone; sv null: K's alone (the other given: FlashAttnI8Stats)
    const bool with_k = mean != nullptr, with_v = sv != nullptr;
    const Src k = k16, v = v16;
    q.parallel_for(sycl::range<2>((size_t) SLICES, (size_t) C), [=, k16 = k, v16 = v](sycl::id<2> id) {
        const int64_t sl = (int64_t) id[0], c = (int64_t) id[1];
        const int64_t t0 = T * sl / SLICES, t1 = T * (sl + 1) / SLICES;
        float sum = 0.f, mx = 0.f;
        for (int64_t t = t0; t < t1; ++t) {
            if (with_k) sum += k16.at(t, c, C);
            if (with_v) mx = sycl::fmax(mx, sycl::fabs(v16.at(t, c, C)));
        }
        part[sl * C + c] = sum;
        part[(SLICES + sl) * C + c] = mx;
    });
    q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = (int64_t) id[0];
        float sum = 0.f, mx = 0.f;
        for (int sl = 0; sl < SLICES; ++sl) {
            sum += part[sl * C + c];
            mx = sycl::fmax(mx, part[(SLICES + sl) * C + c]);
        }
        if (with_k) mean[c] = sum / (float) T;
        if (with_v) sv[c] = mx > 0.f ? mx / 127.f : 1.f;
    });
}

// Q's and K's rows (a (row, head)'s HD values, less `mean` for K) as int8 with their scales (Q's times `qmul`), and V
// as int8 by its (head, dim) scales, transposed into vt [heads][Tp / CH][D][CH] (a chunk's V^T in one piece, which the
// attention reads at once: a dim's CH bytes Tp apart were 2% slower on the RX 9060 XT): one kernel, a work-group a head
// and RB keys, V^T built in local memory so its rows go out RB bytes at a time; rows and keys past T up to Tp zero.
// q16 null: the keys and values alone (make_prefix).  NvKernel's V^T (kind_of 2) holds each 16 keys in the order its
// P operand takes them: at 4 i + j the key 2 i + (j & 1) + 8 (j >> 1).  JmKernel's (kinds 3, 4) are K and V in
// Intel's VNNI order a chunk at a time, the B operands' packed layout (4 K values of every column in a word): K
// [heads][Tp / CH][D / 4][CH][4] (S = Q . K^T's B: a dim's group of 4, every key of the chunk) and V [heads][Tp / CH]
// [CH / 4][D][4] (O = P . V's B: a key's group of 4, every dim).  Read column-major (K's rows, V^T whole) the B70 ran
// the kernel 9 times slower than oneDNN's 16-bit attention, and V row-major, which int8 takes through a VNNI
// transform, left its threads hanging (IGC 2.41.16, 2026-11-06).
template <int HD = D, class Src = Dense>
void prep_i8(sycl::queue& q, const uint16_t* q16, std::type_identity_t<const Src&> k16,
             std::type_identity_t<const Src&> v16, const float* mean,
             const float* sv, int64_t T, int64_t Tp, int64_t heads, float qmul, int8_t* q8, int8_t* k8, int8_t* vt,
             float* sq, float* sk, bool rot = false) {
    constexpr int U = HD / 32;   // a lane's dims of a row
    static_assert(U % 4 == 0 && (U & (U - 1)) == 0, "whole words of 4 a lane, a power of 2 for the transform");
    const int kind = kind_of(q);
    // q16 null: the keys and values alone; k8 null: the queries alone (their own rows and heads: a causal attention's)
    const bool with_q = q16 != nullptr, with_kv = k8 != nullptr, nv = kind == 2, packed = kind == 3 || kind == 4;
    const int64_t nc = Tp / CH;
    constexpr int RB = 128, WG = 256, TLD = RB + 4;   // V^T's rows in the tile: 4 lanes' dims a bank apart
    const int64_t C = heads * HD;
    const Src ks = k16, vs = v16;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> tile(sycl::range<1>((size_t) (HD * TLD)), h);
        sycl::local_accessor<float, 1> kmx(sycl::range<1>((size_t) RB), h);   // NvKernel's: a key's largest |k|
        // the heads the fast dimension: the groups at work together read whole rows, not one head's slice of
        // rows C * 2 bytes apart
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) ((Tp + RB - 1) / RB), (size_t) (heads * WG)),
                                         sycl::range<2>(1, WG)),
                       [=, k16 = ks, v16 = vs](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t hd = (int64_t) it.get_group(1), t0 = (int64_t) it.get_group(0) * RB;
            const int lid = (int) it.get_local_id(1);
            sycl::sub_group sg = it.get_sub_group();
            const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            // rot: a row's dims (U a lane) through the normalized Walsh-Hadamard transform, Q's and K's alike
            auto hadamard = [&](float (&x)[U]) {
#pragma unroll
                for (int m = 1; m < U; m <<= 1)
#pragma unroll
                    for (int u = 0; u < U; ++u)
                        if ((u & m) == 0) {
                            const float a0 = x[u], a1 = x[u + m];
                            x[u] = a0 + a1;
                            x[u + m] = a0 - a1;
                        }
#pragma unroll
                for (int m = 1; m < 32; m <<= 1) {
                    const bool hi = (lane & m) != 0;
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        const float o = sycl::permute_group_by_xor(sg, x[u], m);
                        x[u] = hi ? o - x[u] : x[u] + o;
                    }
                }
                const float n = 1.f / sycl::sqrt((float) HD);
#pragma unroll
                for (int u = 0; u < U; ++u) x[u] *= n;
            };
            auto store = [&](int8_t* p, const float (&x)[U], float inv) {
#pragma unroll
                for (int w = 0; w < U / 4; ++w) {
                    uint32_t wd = 0;
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        wd |= (uint32_t) (uint8_t) (int8_t) sycl::rint(x[4 * w + u] * inv) << (8 * u);
                    *reinterpret_cast<uint32_t*>(p + (int64_t) 4 * w) = wd;
                }
            };
            // Q and K: a wave a row at a time, U dims a lane; V into the tile by its scales
            float isv[U], mn[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {   // the queries alone: none (sv and mean the keys' heads')
                isv[u] = with_kv ? 1.f / sv[(int64_t) hd * HD + (int64_t) lane * U + u] : 1.f;
                mn[u] = with_kv ? mean[(int64_t) hd * HD + (int64_t) lane * U + u] : 0.f;
            }
            for (int tt = wave; tt < sycl::min<int64_t>(RB, Tp - t0); tt += WG / 32) {
                const int64_t t = t0 + tt, col = (int64_t) hd * HD + (int64_t) lane * U, base = t * C + col;
                const bool in = t < T;
                float vq[U], vk[U], vv[U];
                Dense(q16).load(t, col, C, in && with_q, vq);
                k16.load(t, col, C, in, vk);
                v16.load(t, col, C, in, vv);
                float mq = 0.f, mk = 0.f;
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    if (in) vk[u] -= mn[u];
                    tile[(lane * U + u) * TLD + tt] = (int8_t) sycl::rint(vv[u] * isv[u]);
                }
                if (rot) {
                    hadamard(vq);
                    hadamard(vk);
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    mq = sycl::fmax(mq, sycl::fabs(vq[u]));
                    mk = sycl::fmax(mk, sycl::fabs(vk[u]));
                }
                mq = sycl::reduce_over_group(sg, mq, sycl::maximum<float>());
                mk = sycl::reduce_over_group(sg, mk, sycl::maximum<float>());
                const float cq = mq > 0.f ? mq / 127.f : 1.f, ck = mk > 0.f ? mk / 127.f : 1.f;
                if (with_q) store(q8 + base, vq, 1.f / cq);
                if (with_kv && packed) {   // the lane's words at their dims' groups of 4, the key's place in each
#pragma unroll
                    for (int w = 0; w < U / 4; ++w) {
                        uint32_t wd = 0;
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            wd |= (uint32_t) (uint8_t) (int8_t) sycl::rint(vk[4 * w + u] / ck) << (8 * u);
                        const int64_t g = (int64_t) lane * (U / 4) + w;
                        *reinterpret_cast<uint32_t*>(k8 + (((int64_t) hd * nc + t / CH) * (HD / 4) + g) * (int64_t{CH} * 4) +
                                                     (t % CH) * 4) = wd;
                    }
                } else if (with_kv && !nv) {
                    store(k8 + base, vk, 1.f / ck);
                }
                if (lane == 0) {
                    if (with_q) sq[t * heads + hd] = cq * qmul;
                    if (with_kv && !nv) sk[t * heads + hd] = ck;
                    else kmx[tt] = mk;
                }
            }
            sycl::group_barrier(it.get_group());
            if (!with_kv) return;
            if (nv) {
                // NvKernel's K: a scale for the 8 keys of a chunk a lane holds (columns 8 j + 2 g and + 1 of the
                // chunk's tiles of 8, g its lane % 4): the lane's scores take one scale, their maximum taken before
                // it and their exponents one FMA each (SageAttention's per-thread scales); sk [Tp / CH][4][heads]
                for (int tt = wave; tt < sycl::min<int64_t>(RB, Tp - t0); tt += WG / 32) {
                    const int64_t t = t0 + tt, col = (int64_t) hd * HD + (int64_t) lane * U, base = t * C + col;
                    const int kk = (int) (t % CH), g = kk % 8 / 2, c0 = tt - kk;   // c0: the chunk's first in the tile
                    float gm = 0.f;
#pragma unroll
                    for (int j = 0; j < CH / 8; ++j) {
                        gm = sycl::fmax(gm, kmx[c0 + 8 * j + 2 * g]);
                        gm = sycl::fmax(gm, kmx[c0 + 8 * j + 2 * g + 1]);
                    }
                    const float ck = gm > 0.f ? gm / 127.f : 1.f;
                    const bool in = t < T;
                    float vk[U];
                    k16.load(t, col, C, in, vk);
#pragma unroll
                    for (int u = 0; u < U; ++u)
                        if (in) vk[u] -= mn[u];
                    if (rot) hadamard(vk);
                    store(k8 + base, vk, 1.f / ck);
                    if (lane == 0 && kk < 8 && kk % 2 == 0) sk[((t / CH) * 4 + g) * heads + hd] = ck;
                }
            }
            const int nk = (int) sycl::min<int64_t>(RB, Tp - t0);
            for (int i = lid; i < HD * RB / 4; i += WG) {   // V^T along the keys, 4 a work-item
                const int d = i / (RB / 4), tt = i % (RB / 4) * 4;
                if (tt >= nk) continue;
                uint32_t w;
                if (nv) {
                    const int8_t* r = &tile[d * TLD + (tt & ~15) + (tt & 15) / 2];
                    w = (uint32_t) (uint8_t) r[0] | (uint32_t) (uint8_t) r[1] << 8 | (uint32_t) (uint8_t) r[8] << 16 |
                        (uint32_t) (uint8_t) r[9] << 24;
                } else {
                    w = *reinterpret_cast<const uint32_t*>(&tile[d * TLD + tt]);
                }
                const int64_t key = t0 + tt;
                const int64_t at =
                    packed ? (((int64_t) hd * nc + key / CH) * (CH / 4) + key % CH / 4) * (int64_t{HD} * 4) + (int64_t) d * 4
                           : (((int64_t) hd * nc + key / CH) * HD + d) * CH + key % CH;
                *reinterpret_cast<uint32_t*>(vt + at) = w;
            }
        });
    });
}

// The scores as S^T = K . Q^T, so that a lane holds one query (its column) and that query's keys: the rows' maxima and
// sums are the lane's own (and its other half's, one exchange), the probabilities become P^T's operand for O^T = V^T .
// P^T in the registers they are in, and the output's corrections a query's are the lane's scalars.  K and V^T of the
// chunk in local memory (two buffers: the next chunk's loads go out while this one is worked).  exp2, the scores in
// log2 units (Q's scales carry log2 e).
// A second segment of keys may follow the first's T (Seg: a prefix made once, make_prefix), its own K, V^T and key
// scales, its V at other scales: the output so far is turned to them as its chunks begin (o times sw, a dim's), and sv
// is the second's.
struct Seg {
    const int8_t *k = nullptr, *vt = nullptr;
    const float *sk = nullptr, *sw = nullptr;
    int64_t T = 0, Tp = 0;
};

// CAUSAL, HD and QN as NvKernel's (QN: the query tiles a wave, fewer for the larger head's registers)
template <bool CAUSAL, int HD = D, int QN = QT>
struct Kernel {
    static constexpr int D = HD, KLD = HD + 16, QT = QN;   // as the namespace's for this head size
    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<int8_t, 1> kl, vl;       // two chunks' K [CH][KLD] and V^T [D][VLD]
    sycl::local_accessor<float, 1> skl, svl;      // two chunks' key scales [CH], the head's V scales [D] (and the
                                                  // second segment's turn [D] after them)
    int64_t kv_heads = 0, causal = 0;             // CAUSAL's, as NvKernel's

    // the work-group as an attribute, not a property: intel/llvm bounds an AMD kernel's registers by it from the
    // attribute only (see iq_mmq.cpp), and the 32 queries a wave need more than the 192 VGPRs of a 1024-lane group
    [[sycl::reqd_work_group_size(1, NW * 32), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    __attribute__((always_inline)) void operator()(sycl::nd_item<2> it) const {
#if STRATA_AMD_WMMA
        namespace wm = kernels::xe::wmma;   // amd_wmma.hpp's int8 operands and layouts
        using i8v = wm::iv8;
        using fv8 = wm::fv8;
        using u4 = sycl::vec<uint32_t, 4>;
        using frag = wm::s8f;
        constexpr int KOFF = wm::KOFF;
        auto wmma = [](const frag& a, const frag& b, const i8v& c) { return wm::mad_s8(a, b, c); };
        sycl::sub_group sg = it.get_sub_group();
        const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int lo = lane & 15, hi = lane >> 4, lid = (int) it.get_local_id(1);
        auto row_of = [&](int e) { return wm::row_of(hi, e); };   // the accumulator's row of element e (column lo)
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        const int64_t KH = CAUSAL ? kv_heads : heads, hk = CAUSAL ? h / (heads / KH) : h, CK = KH * D;
        // the wave's QT tiles of 16 queries: my0 + 16 qt + lo
        const int64_t wg0 = r0 + (int64_t) it.get_group(1) * NW * 16 * QT;
        const int64_t my0 = wg0 + (int64_t) wave * 16 * QT;
        const bool live = my0 < row_end;
        int8_t* K0 = kl.get_multi_ptr<sycl::access::decorated::no>().get();
        int8_t* V0 = vl.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = lid; i < D; i += NW * 32) {
            svl[i] = sv[hk * D + i];
            if (b.T > 0) svl[D + i] = b.sw[h * D + i];
        }
        // the queries as B operands of S^T (column lo: a query, K values kk * 16 + hi KOFF ..), their scales
        frag qb[QT][D / 16];
        float sql[QT], m[QT], l[QT];
        fv8 o[QT][D / 16];   // O^T: rows the dims nt * 16 + row_of(e), column the lane's query
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
            // causal: Q's rows those of the queries (its buffer row_end long), else of the keys
            const int64_t qr = sycl::min<int64_t>(my0 + 16 * qt + lo, (CAUSAL ? row_end : Tp) - 1);
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk)
                qb[qt][kk] = *reinterpret_cast<const frag*>(q8 + qr * C + h * D + kk * 16 + hi * KOFF);
            sql[qt] = sq[qr * heads + h];
            m[qt] = -INFINITY;
            l[qt] = 0.f;
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt) o[qt][nt] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
        }
        // a chunk's K, V^T and key scales: read into registers (the next chunk's, before this one is worked), then
        // stored into a buffer once it is: 16 bytes of K and of V^T a work-item and piece
        constexpr int KV16 = CH * D / 16;   // the 16-byte pieces of K (and of V^T)
        constexpr int PIECES = (KV16 + NW * 32 - 1) / (NW * 32);
        static_assert(CH <= NW * 32, "a work-item a key's scale");
        u4 rk[PIECES], rv[PIECES];
        float rs = 0.f;
        // chunk ci: the first segment's ci, else the second's ci - na; causal: the first's up to the work-group's last
        // row's last key
        const int64_t wg_last = sycl::min<int64_t>(row_end, wg0 + (int64_t) NW * 16 * QT) - 1;
        const int64_t na = CAUSAL ? sycl::min<int64_t>(Tp / CH, (wg_last + causal + CH) / CH) : Tp / CH,
                      nch = na + b.Tp / CH;
        auto fetch = [&](int64_t ci) {
            const bool sb = ci >= na;
            const int64_t j0 = (sb ? ci - na : ci) * CH, lim = sb ? b.T : T, nc = sb ? b.Tp / CH : Tp / CH;
            const int8_t *kp = sb ? b.k : k8, *vp = sb ? b.vt : vt;
            const float* sp = sb ? b.sk : sk;
            const int64_t kc = sb ? C : CK, kh = sb ? h : hk, sh = sb ? heads : KH;
#pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                const int i = lid + p * NW * 32;
                if (i >= KV16) break;
                const int kr = i / (D / 16), c = i % (D / 16) * 16;
                rk[p] = *reinterpret_cast<const u4*>(kp + (j0 + kr) * kc + kh * D + c);
                const int d = i / (CH / 16), cv = i % (CH / 16) * 16;
                rv[p] = *reinterpret_cast<const u4*>(vp + ((kh * nc + j0 / CH) * D + d) * CH + cv);
            }
            if (lid < CH) rs = j0 + lid < lim ? sp[(j0 + lid) * sh + kh] : 0.f;
        };
        auto stash = [&](int bf) {
            int8_t* Kb = K0 + bf * CH * KLD;
            int8_t* Vb = V0 + bf * D * VLD;
#pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                const int i = lid + p * NW * 32;
                if (i >= KV16) break;
                *reinterpret_cast<u4*>(Kb + (i / (D / 16)) * KLD + i % (D / 16) * 16) = rk[p];
                *reinterpret_cast<u4*>(Vb + (i / (CH / 16)) * VLD + i % (CH / 16) * 16) = rv[p];
            }
            if (lid < CH) skl[bf * CH + lid] = rs;
        };
        fetch(0);
        stash(0);
        sycl::group_barrier(it.get_group());
        if (nch > 1) fetch(1);
        int bf = 0;
        for (int64_t ci = 0; ci < nch; ++ci, bf ^= 1) {
          const bool sb = ci >= na;
          const int64_t j0 = (sb ? ci - na : ci) * CH, lim = sb ? b.T : T;
          // a wave past the rows (or, causal, whose rows see none of the chunk) shares the loads, nothing else
          if (live && !(CAUSAL && !sb && j0 > my0 + (int64_t) (16 * QT - 1) + causal)) {
            if (sb && ci == na) {   // the second segment's V scales from here on
#pragma unroll
                for (int qt = 0; qt < QT; ++qt)
#pragma unroll
                    for (int nt = 0; nt < D / 16; ++nt)
#pragma unroll
                        for (int e = 0; e < 8; ++e) o[qt][nt][e] *= svl[D + nt * 16 + row_of(e)];
            }
            const int8_t* Kb = K0 + bf * CH * KLD;
            const int8_t* Vb = V0 + bf * D * VLD;
            const float* sk_ = &skl[bf * CH];
            const bool whole = j0 + CH <= lim;   // no key of the chunk past the segment's
            // S^T: a tile's rows the keys nt * 16 + row_of(e), its column a query; K's operand read once for the tiles.
            // The scores times the keys' scales only: the query's (positive) goes into exp2's argument, and the
            // largest is taken before it, a multiply less a score
            float s[QT][CH / 16][8];
            float cmx[QT];
#pragma unroll
            for (int qt = 0; qt < QT; ++qt) cmx[qt] = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < CH / 16; ++nt) {
                i8v c[QT];
#pragma unroll
                for (int qt = 0; qt < QT; ++qt) c[qt] = i8v{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int kk = 0; kk < D / 16; ++kk) {
                    const frag ka = *reinterpret_cast<const frag*>(Kb + (nt * 16 + lo) * KLD + kk * 16 + hi * KOFF);
#pragma unroll
                    for (int qt = 0; qt < QT; ++qt) c[qt] = wmma(ka, qb[qt][kk], c[qt]);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int kr = nt * 16 + row_of(e);
                    const bool in = whole || j0 + kr < lim;
                    const float ks = sk_[kr];
#pragma unroll
                    for (int qt = 0; qt < QT; ++qt) {
                        // causal: the key past the lane's query's last
                        const bool seen = !CAUSAL || sb || j0 + kr <= my0 + (int64_t) (16 * qt + lo) + causal;
                        s[qt][nt][e] = in && seen ? (float) c[qt][e] * ks : -INFINITY;
                        cmx[qt] = sycl::fmax(cmx[qt], s[qt][nt][e]);
                    }
                }
            }
            float ps[QT];
            frag pb[QT][CH / 16];
#pragma unroll
            for (int qt = 0; qt < QT; ++qt) {
                cmx[qt] = sycl::fmax(cmx[qt], sycl::permute_group_by_xor(sg, cmx[qt], 16)) * sql[qt];   // its other keys
                // the output and the sum follow a new reference only where the chunk's largest passes it by 2^8 (or
                // there is none yet): below that, the chunk's scale ps (up to 2^8 / 127) carries it in FP32, and a
                // wave's chunks seldom take this branch, which scales the whole output
                if (cmx[qt] > m[qt] + 8.f) {
                    const float corr = m[qt] == -INFINITY ? 0.f : sycl::native::exp2(m[qt] - cmx[qt]);
#pragma unroll
                    for (int nt = 0; nt < D / 16; ++nt) o[qt][nt] *= corr;
                    l[qt] *= corr;
                    m[qt] = cmx[qt];
                }
                // P^T as int8 against the chunk's largest (its scale ps), straight into O^T's B operands: against the
                // row's m instead, a long row's many small probabilities round to 0 (LPIPS 0.09-0.27 on Qwen-Image).
                // P is taken from the scores less the chunk's largest, not from the row's m times 1 / ps: a chunk far
                // below the row's largest (MOSS-Image's peaked heads, 2^-130 and below) makes ps denormal, 1 / ps
                // infinite and the row's sum NaN, and the head's output 0 (MOSS-Image's fox blurred, LPIPS 0.146 against
                // 0.022 so); such a chunk's ps is 0 and it adds nothing
                ps[qt] = m[qt] == -INFINITY ? 0.f : sycl::native::exp2(cmx[qt] - m[qt]) / 127.f;
                float sum = 0.f;
                // log2 127: the values times 127 in exp2's argument; a query that sees none of the chunk (causal) has
                // a cmx of -inf, and exp2 of -inf + inf would be NaN in its sum
                const float off = cmx[qt] == -INFINITY ? 0.f : 6.98868468677217f - cmx[qt];
#pragma unroll
                for (int nt = 0; nt < CH / 16; ++nt) {
                    // 0 .. 127, rounded and packed a byte at a time by v_cvt_pk_u8_f32 (a score past T: exp2 of -inf,
                    // 0); the sum of the values before their rounding, which is as often up as down
                    uint32_t w0 = 0, w1 = 0;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float pv = sycl::native::exp2(sycl::fma(s[qt][nt][e], sql[qt], off));
                        sum += pv;
                        if (e < 4) w0 = __builtin_amdgcn_cvt_pk_u8_f32(pv, (uint32_t) e, w0);
                        else w1 = __builtin_amdgcn_cvt_pk_u8_f32(pv, (uint32_t) (e - 4), w1);
                    }
#if defined(__GFX12__)
                    // the lane's 8 keys are rows hi * 8 + e: B's K values hi * 8 + e, in order
                    pb[qt][nt] = frag{(int) w0, (int) w1};
#else
                    // GFX11: B takes all 16 K values a lane; the lane has rows 2 e + hi, its other half 2 e + 1 - hi
                    const uint32_t o0 = sycl::permute_group_by_xor(sg, w0, 16), o1 = sycl::permute_group_by_xor(sg, w1, 16);
                    auto byte = [](uint32_t w, int i) { return (w >> (8 * i)) & 0xffu; };
                    const uint32_t a0 = hi ? o0 : w0, b0 = hi ? w0 : o0, a1 = hi ? o1 : w1, b1 = hi ? w1 : o1;
                    // key 2 e + 0 from the half with hi 0 (a), 2 e + 1 from hi 1 (b)
                    pb[qt][nt] = frag{(int) (byte(a0, 0) | byte(b0, 0) << 8 | byte(a0, 1) << 16 | byte(b0, 1) << 24),
                                      (int) (byte(a0, 2) | byte(b0, 2) << 8 | byte(a0, 3) << 16 | byte(b0, 3) << 24),
                                      (int) (byte(a1, 0) | byte(b1, 0) << 8 | byte(a1, 1) << 16 | byte(b1, 1) << 24),
                                      (int) (byte(a1, 2) | byte(b1, 2) << 8 | byte(a1, 3) << 16 | byte(b1, 3) << 24)};
#endif
                }
                sum += sycl::permute_group_by_xor(sg, sum, 16);
                l[qt] += sum * ps[qt];
            }
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt) {   // O^T += V^T . P^T, the dims nt * 16 ..; V^T's operand once
                i8v c[QT];
#pragma unroll
                for (int qt = 0; qt < QT; ++qt) c[qt] = i8v{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int kc = 0; kc < CH / 16; ++kc) {
                    const frag va = *reinterpret_cast<const frag*>(Vb + (nt * 16 + lo) * VLD + kc * 16 + hi * KOFF);
#pragma unroll
                    for (int qt = 0; qt < QT; ++qt) c[qt] = wmma(va, pb[qt][kc], c[qt]);
                }
#pragma unroll
                for (int qt = 0; qt < QT; ++qt)
#pragma unroll
                    for (int e = 0; e < 8; ++e) o[qt][nt][e] += (float) c[qt][e] * ps[qt];   // V's scales at the end
            }
          }
            // the next chunk into the other buffer (read last in the previous step, which the barrier closed), then
            // the one after it into the registers
            if (ci + 1 < nch) stash(bf ^ 1);
            sycl::group_barrier(it.get_group());
            if (ci + 2 < nch) fetch(ci + 2);
        }
        if (!live) return;
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
            const int64_t t = my0 + 16 * qt + lo;
            if (t >= row_end || t < first) continue;
            const float inv = l[qt] > 0.f ? 1.f / l[qt] : 0.f;
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    out16[(t - first) * C + h * D + nt * 16 + row_of(e)] =
                        f16_from_f32(o[qt][nt][e] * inv * svl[nt * 16 + row_of(e)]);
        }
#else
        (void) it;
#endif
    }
};

// NVIDIA's tensor cores from sm_80 on: the same scheme on mma.sync m16n8k32 (s8 . s8 into s32).  S = Q . K^T with the
// queries as A's rows: a warp NQ tiles of 16 queries, Q's operands in registers for the whole row of keys, a lane
// holding rows g and g + 8 (g its lane / 4) and two keys of each 8, a row's maximum and sum over its quad (four lanes,
// two exchanges).  A step takes NK keys (SC of the layout's chunks; P's int8 scale is the step's), copied into local
// memory with cp.async a step ahead, the operands read from there with ldmatrix.  An accumulator's two keys of each
// tile of 8 are not the four consecutive K values a lane holds of P . V's A operand, so P takes them as they are and
// V^T is stored with its keys in that order (prep_i8).  The output's FP32 sums take each step's int32 P . V by the
// step's scale, so the longer step makes that update rarer.  Integers and floats cross by the bits of 1.5 * 2^23 (an
// integer below 2^22 in magnitude is its mantissa's low bits there), on the full-rate units: the conversion
// instructions run at a quarter of their rate on these GPUs, and a step converts each score, probability and sum.  The
// mma's sums start from those bits, so a score is one FMA from them (with the key's scale and its -1.5 * 2^23
// multiple), and P's bytes are packed with prmt.  A step that ends before the segment's last key takes no masks.
// A warp 16 queries and a work-group 8 warps: 3.7 ms on the RTX 4070 for 4400 rows of 32 heads (3.9 before the FMA
// and the packing) against 4.4 with 32 queries and 4 warps (both at 255 registers; bounded to 128 for two work-groups
// an SM, they spilled: 4.9 ms).  P at a fixed scale (127 exp(s - m), m the maximum so far, as SageAttention's) with
// the sums kept in int32 across steps took only 3% off (2.55 to 2.48 ms, 4115 rows of 32 heads, 2026-11-06), and long
// rows' many small probabilities lost too much in int8 (3 times qi_attn_parity's bound at 8192 keys).  P is
// unsigned, 0 .. 255 against the step's largest, into the u8 . s8 mma (SageAttention's): half the step of 0 .. 127 at
// the same speed
constexpr int NQ = 1;             // query tiles of 16 a warp
constexpr int NNW = 8;            // warps a work-group
constexpr int SC = 1;             // the layout's chunks a step
// work-groups an SM, at least: the registers bounded to fit two (128 a thread), the step one chunk so that two take
// the local memory.  On the RTX 4070 (2026-11-06, 32 heads) 4115 rows took 2.53 ms against 2.75 with one work-group
// an SM and two chunks a step (245 registers, 17% occupancy, latency-bound), 5587 rows 4.38 against 4.74; three
// spilled (6.7 ms), two with two chunks a step 2.58
[[maybe_unused]] constexpr int NVMIN = 2;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define STRATA_NV_MINWG(n) [[intel::min_work_groups_per_cu(n)]]
#else
#define STRATA_NV_MINWG(n)
#endif
constexpr int NK = SC * CH;       // keys a step
constexpr int NVLD = NK + 16;     // a dim's row of V^T in local memory (bytes; 8 rows of an ldmatrix in distinct banks)

// CAUSAL: a causal attention's (run_causal), its members below read; else the image engine's, without their tests.
// HD: the head's dims (D the image engine's; 256 a causal attention's too)
template <bool CAUSAL, int HD = D>
struct NvKernel {
    static constexpr int D = HD, KLD = HD + 16;   // as the namespace's for this head size
    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<int8_t, 1> kl, vl;       // two steps' K [NK][KLD] and V^T [D][NVLD]
    sycl::local_accessor<float, 1> skl, svl;      // two steps' key scales [SC][4] (prep_i8's), the
                                                  // head's V scales [D] (and the second
                                                  // segment's turn [D] after them)
    // CAUSAL's: the keys' and values' heads (query head h reads KV head h / (heads / kv_heads)), and query row t
    // sees the keys j <= t + causal only
    int64_t kv_heads = 0, causal = 0;

    [[sycl::reqd_work_group_size(1, NNW * 32), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    STRATA_NV_MINWG(HD == 128 ? NVMIN : 1)
    void operator()(sycl::nd_item<2> it) const {
#if STRATA_NV_I8
        // P as unsigned bytes (offset log2 255, four packed into a word by byte_perm, their row sums by dp4a) after
        // comfy-kitchen's SageAttention kernels (Apache-2.0)
        constexpr int kMagic = 0x4b400000;      // 1.5 * 2^23's bits
        constexpr float kMagicF = 12582912.f;   // 1.5 * 2^23
        sycl::sub_group sg = it.get_sub_group();
        const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int g = lane >> 2, t4 = lane & 3, lid = (int) it.get_local_id(1);
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        const int64_t KH = CAUSAL ? kv_heads : heads, hk = CAUSAL ? h / (heads / KH) : h, CK = KH * D;
        const int64_t wg0 = r0 + (int64_t) it.get_group(1) * NNW * 16 * NQ;
        const int64_t my0 = wg0 + (int64_t) wave * 16 * NQ;
        const bool live = my0 < row_end;
        int8_t* K0 = kl.get_multi_ptr<sycl::access::decorated::no>().get();
        int8_t* V0 = vl.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = lid; i < D; i += NNW * 32) {
            svl[i] = sv[hk * D + i];
            if (b.T > 0) svl[D + i] = b.sw[h * D + i];
        }
        // the members as values (the lambdas below take them so, not through the kernel object)
        const int8_t *const ka = k8, *const kb2 = b.k, *const va = vt, *const vb2 = b.vt;
        const float *const ska = sk, *const skb = b.sk;
        const int64_t Ta = T, Tb = b.T;
        // the steps: the first segment's, then the second's, each from a chunk of CH keys; causal: the first's up to
        // the work-group's last row's last key
        const int64_t wg_last = sycl::min<int64_t>(row_end, wg0 + (int64_t) NNW * 16 * NQ) - 1;
        const int64_t na = CAUSAL ? sycl::min<int64_t>(Tp / CH, (wg_last + causal + CH) / CH) : Tp / CH,
                      nb = b.Tp / CH, sa = (na + SC - 1) / SC, nsteps = sa + (nb + SC - 1) / SC;
        auto step_of = [=](int64_t si, bool& sb, int64_t& c0, int& nc) {
            sb = si >= sa;
            c0 = (sb ? si - sa : si) * SC;
            nc = (int) sycl::min<int64_t>(SC, (sb ? nb : na) - c0);
        };
        // a step's K rows and V^T's chunks into buffer bf (16 bytes a copy), its key scales into a register: piece p
        // of a work-item is K's row kr0 + p KR at column kc0, and V^T's chunk p at dims vd0 + q VD, keys vc0 (offsets
        // from a base each, not arrays: those were held in registers the kernel had none left for)
        constexpr int NT = NNW * 32, KP = NK * D / 16 / NT, KR = NT / (D / 16), VD = NT / (CH / 16), VQ = D / VD;
        static_assert(KP * NT == NK * D / 16 && NT % (D / 16) == 0 && VQ * VD == D, "whole pieces a work-item");
        const int kr0 = lid / (D / 16), vd0 = lid / (CH / 16);
        const uint32_t ks_at = (uint32_t) (kr0 * KLD + lid % (D / 16) * 16),
                       vs_at = (uint32_t) (vd0 * NVLD + lid % (CH / 16) * 16);
        const int64_t kg_at = (int64_t) kr0 * CK + (int64_t) (lid % (D / 16)) * 16,
                      vg_at = (int64_t) vd0 * CH + (int64_t) (lid % (CH / 16)) * 16;
        const uint32_t ks0 = nv::smem_addr(K0), vs0 = nv::smem_addr(V0);
        float rs = 0.f;
        auto issue = [&](int64_t si, int bf) {
            bool sb;
            int64_t c0;
            int nc;
            step_of(si, sb, c0, nc);
            const int8_t* kp = (sb ? kb2 : ka) + c0 * CH * CK + hk * D + kg_at;
            const int8_t* vp = (sb ? vb2 : va) + (hk * (sb ? nb : Tp / CH) + c0) * D * CH + vg_at;
            const uint32_t kb = ks0 + (uint32_t) bf * NK * KLD + ks_at, vb = vs0 + (uint32_t) bf * D * NVLD + vs_at;
#pragma unroll
            for (int p = 0; p < KP; ++p)
                if (kr0 + p * KR < nc * CH) nv::cp_async_cg(kb + (uint32_t) (p * KR * KLD), kp + (int64_t) p * KR * CK);
#pragma unroll
            for (int p = 0; p < SC; ++p)
#pragma unroll
                for (int dq = 0; dq < VQ; ++dq)
                    if (p < nc)
                        nv::cp_async_cg(vb + (uint32_t) (p * CH + dq * VD * NVLD),
                                        vp + ((int64_t) p * D + (int64_t) dq * VD) * CH);
            nv::cp_async_commit();
            // the step's key scales: 4 a chunk (the keys a lane holds, its lane % 4's), [chunk][4][heads]
            if (lid < 4 * SC) rs = lid / 4 < nc ? (sb ? skb : ska)[((c0 + lid / 4) * 4 + lid % 4) * KH + hk] : 0.f;
        };
        // the queries as A operands (rows g and g + 8 of a tile, K values t4 * 4 .. and 16 + t4 * 4 .. of each 32),
        // the two rows' scales, references and sums (a lane's part of its quad's)
        uint32_t qa[NQ][D / 32][4];
        float sql[NQ][2], m[NQ][2], l[NQ][2];
        // the output's sums take P . V's bits as floats (kMagicF + the int32 sum) times the step's scale, and moff the
        // kMagicF times the scales for its row, taken off at the end: one FMA an element and step, not an FADD too.
        // An element's rounding is then of its sum with the kMagicF multiples, about 2^-24 kMagicF (0.75) times a
        // step's scale, against the 0.5 of each of the step's probabilities in int8 times its value
        float moff[NQ][2];
        float o[NQ][D / 8][4];   // tile nt: rows g (0, 1) and g + 8 (2, 3), dims nt * 8 + 2 t4 + (e & 1)
        issue(0, 0);
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int64_t qr = sycl::min<int64_t>(my0 + (16 * qt + g + 8 * r), (CAUSAL ? row_end : Tp) - 1);
#pragma unroll
                for (int ks = 0; ks < D / 32; ++ks) {
                    const int8_t* p = q8 + qr * C + h * D + (ks * 32 + t4 * 4);
                    qa[qt][ks][r] = *reinterpret_cast<const uint32_t*>(p);
                    qa[qt][ks][2 + r] = *reinterpret_cast<const uint32_t*>(p + 16);
                }
                sql[qt][r] = sq[qr * heads + h];
                m[qt][r] = -INFINITY;
                moff[qt][r] = 0.f;
                l[qt][r] = 0.f;
            }
#pragma unroll
            for (int nt = 0; nt < D / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) o[qt][nt][e] = 0.f;
        }
        if (lid < 4 * SC) {
            skl[(size_t) lid] = rs;
        }
        // ldmatrix's row addresses: lane l the row l % 8 of matrix l / 8 (of K: tile j + (l >> 4), dims
        // + 16 (l >> 3 & 1); of V^T: dims tile nt + (l >> 4), keys + 16 (l >> 3 & 1))
        const int lr = lane & 7, lm = lane >> 3;
        const uint32_t koff = (uint32_t) ((8 * (lm >> 1) + lr) * KLD + 16 * (lm & 1)),
                       voff = (uint32_t) ((8 * (lm >> 1) + lr) * NVLD + 16 * (lm & 1));
        for (int64_t si = 0; si < nsteps; ++si) {
            // one barrier a step: past it this step's copies (the only ones out) have landed and every warp has
            // left the last step, so the next step's copies may go into the buffer that step read (two barriers a
            // step, the copy issued before the wait: 3.32 ms on the RTX 4070, 4400 rows of 32 heads)
            const int bf = (int) (si & 1);
            nv::cp_async_wait<0>();
            sycl::group_barrier(it.get_group());
            if (si + 1 < nsteps) issue(si + 1, bf ^ 1);
            bool sb;
            int64_t c0;
            int nc;
            step_of(si, sb, c0, nc);
            if (live) {
                const int64_t j0 = c0 * CH, lim = sb ? Tb : Ta;
                if (sb && c0 == 0) {   // the second segment's V scales from here on (moff taken off first: a dim's)
#pragma unroll
                    for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
                        for (int nt = 0; nt < D / 8; ++nt)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                o[qt][nt][e] = (o[qt][nt][e] - moff[qt][e >> 1]) * svl[D + nt * 8 + 2 * t4 + (e & 1)];
                        moff[qt][0] = moff[qt][1] = 0.f;
                    }
                }
                const uint32_t kb = nv::smem_addr(K0) + (uint32_t) bf * NK * KLD + koff,
                               vb = nv::smem_addr(V0) + (uint32_t) bf * D * NVLD + voff;
                const float* sk_ = &skl[(size_t) bf * 4 * SC];
                const int kmax = (int) sycl::min<int64_t>(NK, lim - j0);   // the step's keys before the segment's end
                // S: tile j's keys j * 8 + 2 t4 + (e & 1), rows g (e < 2) and g + 8; K's operands read once for the
                // tiles
                int acc[NQ][NK / 8][4];
#pragma unroll
                for (int qt = 0; qt < NQ; ++qt)
#pragma unroll
                    for (int j = 0; j < NK / 8; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[qt][j][e] = kMagic;
#pragma unroll
                for (int ks = 0; ks < D / 32; ++ks)
#pragma unroll
                    for (int j = 0; j < NK / 8; j += 2) {
                        uint32_t kf[4];   // tile j's b0, b1, tile j + 1's
                        ldsm4(kf, kb + j * 8 * KLD + ks * 32);
#pragma unroll
                        for (int qt = 0; qt < NQ; ++qt) {
                            mma_s8(acc[qt][j], qa[qt][ks], kf[0], kf[1]);
                            mma_s8(acc[qt][j + 1], qa[qt][ks], kf[2], kf[3]);
                        }
                    }
                uint32_t pa[NQ][NK / 32][4];
                float ps[NQ][2];
                // the scores, P and its sums; a step that ends past the segment's last key or, causal, past a row's
                // last key (edge) masks the keys after them
                auto scores = [&](auto edge, int qt) __attribute__((always_inline)) {
                    int klim[2] = {kmax, kmax};   // rows g and g + 8: the step's keys they see
                    if constexpr (decltype(edge)::value)
                        if (CAUSAL && !sb)
#pragma unroll
                            for (int r = 0; r < 2; ++r)
                                klim[r] = (int) sycl::max<int64_t>(
                                    0, sycl::min<int64_t>(my0 + (int64_t) (16 * qt + g + 8 * r) + causal + 1 - j0,
                                                          (int64_t) kmax));
                    // the lane's keys of a chunk share a scale (sk_[chunk * 4 + t4]): their maximum is taken on the
                    // sums' bits (kMagicF + the sum, as a float: increasing with the sum), then scaled
                    float ksc[SC];   // the lane's keys' scale in each chunk of the step
#pragma unroll
                    for (int c = 0; c < SC; ++c) ksc[c] = sk_[c * 4 + t4];
                    float bmx[SC][2];
#pragma unroll
                    for (int c = 0; c < SC; ++c) bmx[c][0] = bmx[c][1] = -INFINITY;
                    float s[NK / 8][4];
#pragma unroll
                    for (int j = 0; j < NK / 8; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            s[j][e] = sycl::bit_cast<float>(acc[qt][j][e]);
                            if constexpr (decltype(edge)::value)
                                if (j * 8 + 2 * t4 + (e & 1) >= klim[e >> 1]) s[j][e] = -INFINITY;
                            bmx[j / (CH / 8)][e >> 1] = sycl::fmax(bmx[j / (CH / 8)][e >> 1], s[j][e]);
                        }
                    float cmx[2], a[SC][2], b[SC][2];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        cmx[r] = -INFINITY;
#pragma unroll
                        for (int c = 0; c < SC; ++c) {
                            a[c][r] = ksc[c] * sql[qt][r];   // a score's log2 units an int8 unit of the sum
                            cmx[r] = sycl::fmax(cmx[r], (bmx[c][r] - kMagicF) * a[c][r]);
                        }
                        cmx[r] = sycl::fmax(cmx[r], sycl::permute_group_by_xor(sg, cmx[r], 1));
                        cmx[r] = sycl::fmax(cmx[r], sycl::permute_group_by_xor(sg, cmx[r], 2));
                        if (cmx[r] > m[qt][r] + 8.f) {   // as Kernel's: a new reference past 2^8 only
                            const float corr = m[qt][r] == -INFINITY ? 0.f : ex2(m[qt][r] - cmx[r]);
#pragma unroll
                            for (int nt = 0; nt < D / 8; ++nt) {
                                const int e0 = 2 * r;
                                o[qt][nt][e0] *= corr;
                                o[qt][nt][e0 + 1] *= corr;
                            }
                            l[qt][r] *= corr;
                            moff[qt][r] *= corr;
                            m[qt][r] = cmx[r];
                        }
                        ps[qt][r] = m[qt][r] == -INFINITY ? 0.f : ex2(cmx[r] - m[qt][r]) * (1.f / 255.f);
                        // P = 255 * 2^(score - cmx), the score (bits - kMagicF) a: one FMA a score.  b rounds at
                        // kMagicF a's magnitude, up to 2.25 a (ulp(kMagicF a) is at most 1.5 a) off the cmx the
                        // maximum was scaled to: P's top 2.5 a lower, so it never rounds past 255 to a byte of 0
#pragma unroll
                        for (int c = 0; c < SC; ++c)
                            b[c][r] = (7.99435343685886f - 2.5f * a[c][r] - cmx[r]) - kMagicF * a[c][r];
                    }
                    uint32_t w[NK / 8][2];   // tile j's row r's two values, 0 .. 255, in bytes 0 and 1
#pragma unroll
                    for (int j = 0; j < NK / 8; ++j)
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const int e0 = 2 * r;
                            const int c = j / (CH / 8);
                            const float p0 = ex2(sycl::fma(s[j][e0], a[c][r], b[c][r])),
                                        p1 = ex2(sycl::fma(s[j][e0 + 1], a[c][r], b[c][r]));
                            // rounded to the nearest (even) integer by the add, its low byte the value
                            w[j][r] = byte_perm(sycl::bit_cast<uint32_t>(p0 + kMagicF),
                                                sycl::bit_cast<uint32_t>(p1 + kMagicF), 0x0040u);
                        }
                    // A of P . V a 32 keys: row g's keys 4 t4 .. in V^T's order from tiles 4 kc and 4 kc + 1, then
                    // 16 + 4 t4 .. from 4 kc + 2 and 4 kc + 3; row g + 8's the same
#pragma unroll
                    for (int kc = 0; kc < NK / 32; ++kc)
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const int j0 = 4 * kc;
                            pa[qt][kc][r] = byte_perm(w[j0][r], w[j0 + 1][r], 0x5410u);
                            pa[qt][kc][2 + r] = byte_perm(w[j0 + 2][r], w[j0 + 3][r], 0x5410u);
                        }
                    // the row's sum of the rounded P the product takes, its bytes summed (dp4a)
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        uint32_t sum = 0;
#pragma unroll
                        for (int kc = 0; kc < NK / 32; ++kc)
                            sum = nv::sum_u8x4(pa[qt][kc][2 + r], nv::sum_u8x4(pa[qt][kc][r], sum));
                        l[qt][r] += (float) sum * ps[qt][r];
                    }
                };
#pragma unroll
                for (int qt = 0; qt < NQ; ++qt) {
                    // causal: a tile whose first row sees the step's last key takes no masks
                    if (kmax == NK && (!CAUSAL || sb || j0 + NK - 1 <= my0 + (int64_t) (16 * qt) + causal))
                        scores(std::false_type{}, qt);
                    else
                        scores(std::true_type{}, qt);
                }
#pragma unroll
                for (int qt = 0; qt < NQ; ++qt)
#pragma unroll
                    for (int r = 0; r < 2; ++r) moff[qt][r] = sycl::fma(kMagicF, ps[qt][r], moff[qt][r]);
                // O += P . V, dims nt * 8 .. and the next 8; V^T's operands read once
#pragma unroll
                for (int nt = 0; nt < D / 8; nt += 2) {
                    int c[NQ][2][4];
#pragma unroll
                    for (int qt = 0; qt < NQ; ++qt)
#pragma unroll
                        for (int e = 0; e < 4; ++e) c[qt][0][e] = c[qt][1][e] = kMagic;
#pragma unroll
                    for (int kc = 0; kc < NK / 32; ++kc) {
                        uint32_t vf[4];   // dims tile nt's b0, b1, tile nt + 1's
                        ldsm4(vf, vb + nt * 8 * NVLD + kc * 32);
#pragma unroll
                        for (int qt = 0; qt < NQ; ++qt) {
                            mma_u8s8(c[qt][0], pa[qt][kc], vf[0], vf[1]);
                            mma_u8s8(c[qt][1], pa[qt][kc], vf[2], vf[3]);
                        }
                    }
#pragma unroll
                    for (int qt = 0; qt < NQ; ++qt)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float pr = ps[qt][e >> 1];
                            o[qt][nt][e] = sycl::fma(sycl::bit_cast<float>(c[qt][0][e]), pr, o[qt][nt][e]);
                            o[qt][nt + 1][e] = sycl::fma(sycl::bit_cast<float>(c[qt][1][e]), pr, o[qt][nt + 1][e]);
                        }
                }
            }
            if (si + 1 < nsteps && lid < 4 * SC) {   // the next step's key scales (its buffer was read last step)
                skl[(size_t) (bf ^ 1) * 4 * SC + (size_t) lid] = rs;
            }
        }
        if (!live) return;
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float lr2 = l[qt][r];
                lr2 += sycl::permute_group_by_xor(sg, lr2, 1);
                lr2 += sycl::permute_group_by_xor(sg, lr2, 2);
                const int64_t t = my0 + (16 * qt + g + 8 * r);
                if (t >= row_end || t < first) continue;
                const float inv = lr2 > 0.f ? 1.f / lr2 : 0.f;
#pragma unroll
                for (int nt = 0; nt < D / 8; ++nt) {
                    const int d = nt * 8 + 2 * t4;
                    const int e0 = 2 * r;
                    const float mo = moff[qt][r];
                    const uint32_t v = (uint32_t) f16_from_f32((o[qt][nt][e0] - mo) * inv * svl[(size_t) d]) |
                                       (uint32_t) f16_from_f32((o[qt][nt][e0 + 1] - mo) * inv * svl[d + 1]) << 16;
                    *reinterpret_cast<uint32_t*>(out16 + (t - first) * C + h * D + d) = v;
                }
            }
#else
        (void) it;
#endif
    }
};

// Every other GPU: the int8 dot product of four values (dot_acc: DP4A, sdot4, dp4a) without matrix engines - RDNA2,
// Pascal, an Arc or integrated GPU without XMX, a generation not known here.  A work-group takes BQ query rows of a
// head and the chunks of CH keys in turn, Q, the chunk's K and V^T (prep_i8's) in local memory: a work-item a row's
// CH / IR keys' scores, the row's first item its softmax (P in int8 against the chunk's largest, as Kernel's), then a
// work-item a row's D / IR output dims, P . V four keys at a time.
struct DotKernel {
    static constexpr int BQ = 16, WG = 128, IR = WG / BQ, KI = CH / IR, DI = D / IR;
    static constexpr int SLD = CH + 1;
    static_assert(CH % (4 * IR) == 0 || KI * IR == CH, "a row's keys over its items");
    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<int32_t, 1> ql;   // Q [BQ][D / 4], four values a word
    sycl::local_accessor<int32_t, 1> kl;   // the chunk's K [CH][D / 4]
    sycl::local_accessor<int32_t, 1> vl;   // the chunk's V^T [D][CH / 4]
    sycl::local_accessor<float, 1> sl;     // S [BQ][SLD]
    sycl::local_accessor<int32_t, 1> pl;   // P [BQ][CH / 4]
    sycl::local_accessor<float, 1> rl;     // a row's m, l, correction and P scale [4][BQ]

    void operator()(sycl::nd_item<2> it) const {
        const int lid = (int) it.get_local_linear_id();
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        const int64_t q0 = r0 + (int64_t) it.get_group(1) * BQ;
        auto grp = it.get_group();
        // Q's rows (past the rows: the last one's, its output not written)
        for (int i = lid; i < BQ * D / 4; i += WG) {
            const int rr = i / (D / 4), c = i % (D / 4);
            const int64_t row = sycl::min<int64_t>(q0 + rr, row_end - 1);
            ql[i] = *reinterpret_cast<const int32_t*>(q8 + row * C + h * D + (int64_t) c * 4);
        }
        if (lid < BQ) {
            rl[lid] = -INFINITY;
            rl[BQ + lid] = 0.f;
        }
        const int r = lid / IR, part = lid % IR;
        const int64_t row = sycl::min<int64_t>(q0 + r, row_end - 1);
        const float sql = sq[row * heads + h];
        float o[DI];
#pragma unroll
        for (int e = 0; e < DI; ++e) o[e] = 0.f;
        for (int segi = 0; segi < 2; ++segi) {
            const bool sb = segi == 1;
            if (sb && b.T == 0) break;
            const int64_t keys = sb ? b.T : T, nc = (sb ? b.Tp : Tp) / CH;
            const int8_t* kp = sb ? b.k : k8;
            const int8_t* vp = sb ? b.vt : vt;
            const float* skp = sb ? b.sk : sk;
            if (sb) {   // the output so far at the second segment's V scales
#pragma unroll
                for (int e = 0; e < DI; ++e) o[e] *= b.sw[h * D + (int64_t) part * DI + e];
            }
            for (int64_t j0 = 0; j0 < keys; j0 += CH) {
                sycl::group_barrier(grp);   // the last chunk's K, V^T and P read
                for (int i = lid; i < CH * D / 4; i += WG) {
                    const int kr = i / (D / 4), c = i % (D / 4);
                    kl[i] = *reinterpret_cast<const int32_t*>(kp + (j0 + kr) * C + h * D + (int64_t) c * 4);
                }
                const auto* vc = reinterpret_cast<const int32_t*>(vp + (h * nc + j0 / CH) * D * CH);
                for (int i = lid; i < D * CH / 4; i += WG) vl[i] = vc[i];
                sycl::group_barrier(grp);
                // the row's scores, KI keys an item, in log2 units less the row's own scale
#pragma unroll
                for (int u = 0; u < KI; ++u) {
                    const int key = part * KI + u;
                    int32_t acc = 0;
#pragma unroll 8
                    for (int c = 0; c < D / 4; ++c)
                        acc = sycl::ext::oneapi::dot_acc(ql[r * (D / 4) + c], kl[key * (D / 4) + c], acc);
                    sl[r * SLD + key] = j0 + key < keys ? (float) acc * skp[(j0 + key) * heads + h] : -INFINITY;
                }
                sycl::group_barrier(grp);
                if (part == 0) {   // the row's softmax step
                    float cmx = -INFINITY;
                    for (int k = 0; k < CH; ++k) cmx = sycl::fmax(cmx, sl[r * SLD + k]);
                    cmx *= sql;
                    float m = rl[r], l = rl[BQ + r], corr = 1.f;
                    if (cmx > m + 8.f) {   // a new reference only past 2^8 (as Kernel's)
                        corr = m == -INFINITY ? 0.f : sycl::native::exp2(m - cmx);
                        l *= corr;
                        m = cmx;
                    }
                    const float ps = m == -INFINITY ? 0.f : sycl::native::exp2(cmx - m) / 127.f;
                    const float off = 6.98868468677217f - cmx;   // log2 127
                    float sum = 0.f;
                    for (int k4 = 0; k4 < CH / 4; ++k4) {
                        uint32_t w = 0;
                        for (int u = 0; u < 4; ++u) {
                            const float pv = sycl::native::exp2(sycl::fma(sl[r * SLD + k4 * 4 + u], sql, off));
                            sum += pv;
                            w |= (uint32_t) (uint8_t) (int8_t) sycl::rint(pv) << (8 * u);
                        }
                        pl[r * (CH / 4) + k4] = (int32_t) w;
                    }
                    rl[r] = m;
                    rl[BQ + r] = l + sum * ps;
                    rl[2 * BQ + r] = corr;
                    rl[3 * BQ + r] = ps;
                }
                sycl::group_barrier(grp);
                const float corr = rl[2 * BQ + r], ps = rl[3 * BQ + r];
#pragma unroll
                for (int e = 0; e < DI; ++e) {
                    const int d = part * DI + e;
                    int32_t acc = 0;
#pragma unroll
                    for (int k4 = 0; k4 < CH / 4; ++k4)
                        acc = sycl::ext::oneapi::dot_acc(vl[d * (CH / 4) + k4], pl[r * (CH / 4) + k4], acc);
                    o[e] = o[e] * corr + (float) acc * ps;
                }
            }
        }
        const int64_t t = q0 + r;
        if (t < row_end && t >= first) {
            const float l = rl[BQ + r], inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
            for (int e = 0; e < DI; ++e) {
                const int d = part * DI + e;
                out16[(t - first) * C + h * D + d] = f16_from_f32(o[e] * inv * sv[h * D + d]);
            }
        }
    }
};

// joint_matrix's int8 tiles (TM x TN x 32, s8 . s8 into s32) on SG lanes, for a GPU whose matrix engines report
// that combination and that has no kernel above: Intel's XMX (Xe2's 8 x 16 x 32 on 16 lanes, Alchemist's 8 x 8 x 32
// on 8).  A sub-group takes TM query rows of a head and walks its keys a layout chunk (CH) at a time: S = Q . K^T with
// Q's tiles in registers and K's from global memory in VNNI order (prep_i8; the sub-groups of a work-group read the
// same keys, through the caches), the scores into local memory, a row's softmax over its lanes (SG / TM of them), P in
// int8 against the chunk's largest into local memory, O += P . V in FP32 accumulators, V in VNNI order.  A row's
// factors (the chunk's P scale, a moved reference's correction) reach the accumulators through local memory: on
// Intel's GPUs a lane holds its column of a TM x SG accumulator, its elements the rows in order (as iq_mmq.cpp reads
// them); the second segment's V scales are a column's, the lane's own.
template<int TN_, int SG_>
struct JmKernel {
    static constexpr int TN = TN_, SG = SG_;
    static constexpr int TM = 8, TK = 32, NSG = 64 / TM;   // NSG sub-groups a work-group: 64 query rows
    static constexpr int L = SG / TM, CW = CH / L;          // a row's lanes, a lane's columns of the chunk
    static constexpr int SLD = CH + 4, PLD = CH + 16, OLD = D + 4;
    static constexpr int SREG = TM * OLD > TM * SLD ? TM * OLD : TM * SLD;   // the scores, then the output
    static_assert(SG % TM == 0 && CW % 8 == 0 && TN <= CH && CH % TN == 0 && D % TN == 0 && TN == SG && TK == CH,
                  "the tiles' shapes (a lane a column; a chunk of keys P . V's K)");
    using A8 = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK, mx::layout::row_major>;
    using B8 = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, TN, mx::layout::ext_intel_packed>;
    using C32 = mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN>;

    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<float, 1> st;    // per sub-group: S [TM][SLD] (int32), then the output [TM][OLD]
    sycl::local_accessor<int8_t, 1> pl;   // per sub-group: P [TM][PLD]
    sycl::local_accessor<float, 1> fl;    // per sub-group: a factor a row [TM]
    sycl::local_accessor<uint32_t, 1> vl;  // the work-group's: the chunk's V [CH / 4][D][4]

    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }

    template<typename T_> static auto gp(const T_* p) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            const_cast<T_*>(p));
    }
    template<typename T_> static auto lp(T_* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    }

    void operator()(sycl::nd_item<2> it) const {
        // SPIR-V's only (built(0)): the other targets' joint_matrix has no int8 tiles of these shapes
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
        {
        sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        // a sub-group past the rows works on the work-group's first rows (the chunks' V through local memory take
        // every sub-group at the work-group's barriers) and writes nothing
        const int64_t mine = r0 + (int64_t) it.get_group(1) * NSG * TM + (int64_t) g * TM;
        const bool active = mine < row_end;
        const int64_t my0 = active ? mine : r0 + (int64_t) it.get_group(1) * NSG * TM;
        const int wid = (int) it.get_local_id(1);
        float* S = &st[(size_t) g * SREG];
        int32_t* Si = reinterpret_cast<int32_t*>(S);
        int8_t* P8 = &pl[(size_t) g * TM * PLD];
        float* F = &fl[(size_t) g * TM];
        const int r = lane / L, part = lane % L;   // the lane's row of the tile and its share of the row
        // a row's factor into F[row] (its lanes write the same value), for every lane to read by its elements' rows
        auto factors = [&](float v) {
            sycl::group_barrier(sg);
            if (part == 0) F[r] = v;
            sycl::group_barrier(sg);
        };
        A8 qa[D / TK];
#pragma unroll
        for (int kk = 0; kk < D / TK; ++kk) mx::joint_matrix_load(sg, qa[kk], gp(q8 + my0 * C + h * D + kk * TK), C);
        // the output's FP32 sums in the lane's registers: its column of each TN dims, the rows in order
        float o[D / TN][TM];
#pragma unroll
        for (int nt = 0; nt < D / TN; ++nt)
#pragma unroll
            for (int i = 0; i < TM; ++i) o[nt][i] = 0.f;
        const float sql = sq[(my0 + r) * heads + h];
        float m = -INFINITY, l = 0.f;
        for (int segi = 0; segi < 2; ++segi) {
            const bool sb = segi == 1;
            if (sb && b.T == 0) break;
            const int64_t keys = sb ? b.T : T, tp = sb ? b.Tp : Tp;
            const int8_t* kp = sb ? b.k : k8;
            const int8_t* vp = sb ? b.vt : vt;
            const float* skp = sb ? b.sk : sk;
            const int64_t nc = tp / CH;
            if (sb) {   // the output so far at the second segment's V scales: a dim's, the lane's column
#pragma unroll
                for (int nt = 0; nt < D / TN; ++nt) {
                    const float f = b.sw[h * D + nt * TN + lane];
#pragma unroll
                    for (int i = 0; i < TM; ++i) o[nt][i] *= f;
                }
            }
            for (int64_t j0 = 0; j0 < keys; j0 += CH) {
                // the chunk's V ([CH / 4][D][4], VNNI) into local memory, the work-group's, each TN dims' columns a
                // block of their own ([D / TN][CH / 4][TN][4]: a B tile's rows TN words apart).  Its tiles read
                // packed with rows D words apart, from global or local memory, ended the B70's threads there (IGC
                // 2.41, 2026-11-10); and the sub-groups share it.
                {
                    const auto* src = reinterpret_cast<const uint32_t*>(vp + (h * nc + j0 / CH) * (CH * D));
                    for (int w = wid; w < CH * D / 4; w += NSG * SG) {
                        const int kr = w / D, d = w % D;
                        vl[(d / TN) * (CH / 4 * TN) + kr * TN + d % TN] = src[w];
                    }
                }
                sycl::group_barrier(it.get_group());
                // S = Q . K^T, int32, the chunk's keys TN at a time
#pragma unroll
                for (int nt = 0; nt < CH / TN; ++nt) {
                    C32 s;
                    mx::joint_matrix_fill(sg, s, 0);
#pragma unroll
                    for (int kk = 0; kk < D / TK; ++kk) {
                        B8 kb;
                        mx::joint_matrix_load(
                            sg, kb, gp(kp + ((h * nc + j0 / CH) * (D / 4) + kk * (TK / 4)) * (CH * 4) + nt * TN * 4),
                            CH * 4);
                        mx::joint_matrix_mad(sg, s, qa[kk], kb, s);
                    }
                    mx::joint_matrix_store(sg, s, lp(Si + nt * TN), SLD, mx::layout::row_major);
                }
                sycl::group_barrier(sg);
                // the row's scores in log2 units (its keys' scales, then its own), its largest over its lanes
                float sc[CW];
                float cmx = -INFINITY;
#pragma unroll
                for (int c = 0; c < CW; ++c) {
                    const int col = part * CW + c;
                    const int64_t key = j0 + col;
                    sc[c] = key < keys ? (float) Si[r * SLD + col] * skp[key * heads + h] : -INFINITY;
                    cmx = sycl::fmax(cmx, sc[c]);
                }
                for (int x = 1; x < L; x <<= 1) cmx = sycl::fmax(cmx, sycl::permute_group_by_xor(sg, cmx, x));
                cmx *= sql;
                // a new reference only past 2^8 (as Kernel's): the output's rows and the sum follow it
                float corr = 1.f;
                if (cmx > m + 8.f) {
                    corr = m == -INFINITY ? 0.f : sycl::native::exp2(m - cmx);
                    l *= corr;
                    m = cmx;
                }
                if (sycl::any_of_group(sg, corr != 1.f)) {
                    factors(corr);
                    float cf[TM];
#pragma unroll
                    for (int i = 0; i < TM; ++i) cf[i] = F[i];
#pragma unroll
                    for (int nt = 0; nt < D / TN; ++nt)
#pragma unroll
                        for (int i = 0; i < TM; ++i) o[nt][i] *= cf[i];
                }
                // P against the chunk's largest, 0 .. 127, its scale ps (as Kernel's)
                const float ps = m == -INFINITY ? 0.f : sycl::native::exp2(cmx - m) / 127.f;
                const float off = 6.98868468677217f - cmx;   // log2 127
                float sum = 0.f;
#pragma unroll
                for (int c = 0; c < CW; ++c) {
                    const float pv = sycl::native::exp2(sycl::fma(sc[c], sql, off));
                    sum += pv;
                    P8[r * PLD + part * CW + c] = (int8_t) sycl::rint(pv);
                }
                for (int x = 1; x < L; x <<= 1) sum += sycl::permute_group_by_xor(sg, sum, x);
                l += sum * ps;
                factors(ps);
                // O += P . V, V's B operand in VNNI order: the chunk's [CH / 4][D][4]
                A8 pa;
                mx::joint_matrix_load(sg, pa, lp(P8), PLD);
                float pf[TM];
#pragma unroll
                for (int i = 0; i < TM; ++i) pf[i] = F[i];
#pragma unroll
                for (int nt = 0; nt < D / TN; ++nt) {
                    C32 c;
                    mx::joint_matrix_fill(sg, c, 0);
                    B8 vb;
                    mx::joint_matrix_load(sg, vb, lp(reinterpret_cast<int8_t*>(&vl[0]) + nt * (CH * TN)), TN * 4);
                    mx::joint_matrix_mad(sg, c, pa, vb, c);
                    // the int32 sums at the row's P scale into the lane's sums: one matrix a call (an apply over an
                    // int32 and an FP32 matrix together left the B70's threads hanging, IGC 2.41, 2026-11-10)
                    int i = 0;
                    mx::joint_matrix_apply(sg, c, [&](int32_t& v) {
                        o[nt][i] = sycl::fma((float) v, pf[i], o[nt][i]);
                        ++i;
                    });
                }
                sycl::group_barrier(it.get_group());   // S, P and V are read before the next chunk writes them
            }
        }
        // the output: O / l at V's scales, the lanes' sums into local memory by row
#pragma unroll
        for (int nt = 0; nt < D / TN; ++nt)
#pragma unroll
            for (int i = 0; i < TM; ++i) S[i * OLD + nt * TN + lane] = o[nt][i];
        sycl::group_barrier(sg);
        const int64_t t = my0 + r;
        if (active && t < row_end && t >= first) {
            const float inv = l > 0.f ? 1.f / l : 0.f;
            for (int d = part; d < D; d += L)
                out16[(t - first) * C + h * D + d] = f16_from_f32(S[r * OLD + d] * inv * sv[h * D + d]);
        }
        }
#else
        (void) it;
#endif
    }
};

void launch(sycl::queue& q, const Bufs& b, const float* sv, int64_t T, int64_t Tp, int64_t heads, int64_t r0,
            int64_t first, int64_t row_end, uint16_t* out16, const Seg& seg = {});

// The rows r0 .. row_end - 1 (which see every key) into out16 (its row 0 query row `first`) through the int8 kernel;
// false where it is not built for the device or its buffers do not fit (the caller takes the 16-bit path).
bool run(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
         int64_t r0, int64_t first, int64_t row_end, uint16_t* out16, const FlashAttnI8Stats* st = nullptr) {
    if (kind_of(q) == 0) return false;
    Bufs& b = bufs_of(q);
    const int64_t C = heads * D, Tp = (T + CH - 1) / CH * CH;
    if (!b.grow(T, C, heads)) return false;
    // the statistics the caller gives: V's scales, K's mean (one made before: any mean leaves the softmax as it is)
    const float* sv = st != nullptr && st->sv != nullptr ? st->sv : b.sv;
    float* mean = st != nullptr && st->kmean != nullptr ? st->kmean : b.mean;
    const bool need_k = st == nullptr || st->kmean == nullptr || !st->kmean_ready;
    const bool need_v = sv == b.sv;
    if (need_k || need_v) key_stats(q, k16, v16, T, C, b.part, need_k ? mean : nullptr, need_v ? b.sv : nullptr);
    prep_i8(q, q16, k16, v16, mean, sv, T, Tp, heads, 1.4426950408889634f / std::sqrt((float) D),   // log2 e / sqrt D
            b.q, b.k, b.vt, b.sq, b.sk);
    launch(q, b, sv, T, Tp, heads, r0, first, row_end, out16);
    return true;
}

// the kernel on the buffers prepared: T keys (then seg's), query rows r0 .. row_end - 1, the output at V's scales sv
void launch(sycl::queue& q, const Bufs& b, const float* sv, int64_t T, int64_t Tp, int64_t heads, int64_t r0,
            int64_t first, int64_t row_end, uint16_t* out16, const Seg& seg) {
    if (kind_of(q) == 2) {
        const int64_t rows = (int64_t) NNW * 16 * NQ;
        const size_t groups = (size_t) ((row_end - r0 + rows - 1) / rows);
        q.submit([&](sycl::handler& h) {
            NvKernel<false> k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
                       sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * NK * KLD)), h),
                       sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * D * NVLD)), h),
                       sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (4 * NK)), h),
                       sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * D)), h)};
            h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * NNW * 32}, {1, (size_t) NNW * 32}), k);
        });
        return;
    }
    if (const int kind = kind_of(q); kind == 3 || kind == 4) {
        auto go = [&](auto k0) {
            using K = decltype(k0);
            constexpr int64_t rows = (int64_t) K::NSG * K::TM;
            const size_t groups = (size_t) ((row_end - r0 + rows - 1) / rows);
            q.submit([&](sycl::handler& h) {
                K k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
                    sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (K::NSG * K::SREG)), h),
                    sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (K::NSG * K::TM * K::PLD)), h),
                    sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (K::NSG * K::TM)), h),
                    sycl::local_accessor<uint32_t, 1>(sycl::range<1>((size_t) (CH * D / 4)), h)};
                h.parallel_for(
                    sycl::nd_range<2>({(size_t) heads, groups * K::NSG * K::SG}, {1, (size_t) K::NSG * K::SG}), k);
            });
        };
        if (kind == 3) go(JmKernel<16, 16>{});
        else go(JmKernel<8, 8>{});
        return;
    }
    if (kind_of(q) == 5) {
        using K = DotKernel;
        const size_t groups = (size_t) ((row_end - r0 + K::BQ - 1) / K::BQ);
        q.submit([&](sycl::handler& h) {
            K k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
                sycl::local_accessor<int32_t, 1>(sycl::range<1>((size_t) (K::BQ * D / 4)), h),
                sycl::local_accessor<int32_t, 1>(sycl::range<1>((size_t) (CH * D / 4)), h),
                sycl::local_accessor<int32_t, 1>(sycl::range<1>((size_t) (D * CH / 4)), h),
                sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (K::BQ * K::SLD)), h),
                sycl::local_accessor<int32_t, 1>(sycl::range<1>((size_t) (K::BQ * CH / 4)), h),
                sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (4 * K::BQ)), h)};
            h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * K::WG}, {1, (size_t) K::WG}), k);
        });
        return;
    }
    const size_t groups = (size_t) ((row_end - r0 + (int64_t) NW * 16 * QT - 1) / ((int64_t) NW * 16 * QT));
    q.submit([&](sycl::handler& h) {
        Kernel<false> k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
                 sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * CH * KLD)), h),
                 sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * D * VLD)), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * CH)), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * D)), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * NW * 32}, {1, (size_t) NW * 32}), k);
    });
}

// A prefix of keys made once (make_prefix) and read by many attentions (run_prefixed): K [Pp][C] less its mean, V^T
// [heads][Pp / CH][D][CH], the keys' scales [Pp][heads], V's [C] and K's mean [C], one piece
struct Prefix {
    int8_t *k, *vt;
    float *sk, *sv, *mean;
    int64_t Pp;
    Prefix(void* p, int64_t P, int64_t heads) : Pp((P + CH - 1) / CH * CH) {
        const int64_t C = heads * D;
        k = static_cast<int8_t*>(p);
        vt = k + Pp * C;
        sk = reinterpret_cast<float*>(vt + Pp * C);
        sv = sk + Pp * heads;
        mean = sv + C;
    }
    static uint64_t bytes(int64_t P, int64_t heads) {
        const int64_t Pp = (P + CH - 1) / CH * CH, C = heads * D;
        return (uint64_t) Pp * C * 2 + (uint64_t) Pp * heads * 4 + (uint64_t) C * 8;
    }
};

bool make_prefix(sycl::queue& q, const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* to) {
    if (kind_of(q) == 0) return false;
    Bufs& b = bufs_of(q);   // its key stats' partial sums
    const int64_t C = heads * D;
    if (!b.grow(1, C, heads)) return false;
    const Prefix f(to, P, heads);
    key_stats(q, k16, v16, P, C, b.part, f.mean, f.sv);
    prep_i8(q, nullptr, k16, v16, f.mean, f.sv, P, f.Pp, heads, 0.f, nullptr, f.k, f.vt, nullptr, f.sk);
    return true;
}

// The T rows' attention over their own T keys and then the prefix's P, read where they are (the kernel's second
// segment): the rows' K less the prefix's mean (which leaves a row's softmax as it is), their V at their own scales;
// the prefix's V at its scales times pnum / pden (a head's), which the output takes as the prefix's chunks begin
bool run_prefixed(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                  int64_t heads, const void* prefix, int64_t P, const float* pnum, const float* pden, uint16_t* out16) {
    if (kind_of(q) == 0) return false;
    Bufs& b = bufs_of(q);
    const int64_t C = heads * D, Tp = (T + CH - 1) / CH * CH;
    if (!b.grow(T, C, heads)) return false;
    const Prefix f(const_cast<void*>(prefix), P, heads);
    key_stats(q, k16, v16, T, C, b.part, b.mean, b.sv);   // the rows' V scales (their K's mean is not taken)
    const float *sv = b.sv, *psv = f.sv;
    float *fin = b.mean, *sw = b.sw;
    q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = (int64_t) id[0], h = c / D;
        fin[c] = psv[c] * (pnum[h] / pden[h]);   // key_stats' scales are never 0
        sw[c] = sv[c] / fin[c];
    });
    prep_i8(q, q16, k16, v16, f.mean, b.sv, T, Tp, heads, 1.4426950408889634f / std::sqrt((float) D),
            b.q, b.k, b.vt, b.sq, b.sk);
    launch(q, b, fin, T, Tp, heads, 0, 0, T, out16, Seg{f.k, f.vt, f.sk, sw, P, f.Pp});
    return true;
}

// A causal attention (the LLMs' prompt, flash_attn_i8_causal): Tq query rows of `heads` heads (row t at position
// pos0 + t) against Tk keys of kv_heads heads (key j at position j), the query head h reading KV head
// h / (heads / kv_heads), all FP16 rows [rows][heads HD] and [keys][kv_heads HD]; out16 [Tq][heads HD].  NvKernel
// and Kernel (the other kinds: false, the caller's 16-bit kernel).  Q and K go through the Hadamard transform before
// their int8 (prep_i8's rot): an LLM's few large dims of Q and K took the others' precision, and the first token's KL
// from the matrix engines' kernel was 0.137 on a 4K prompt of random tokens to llm-jp-4.1, 0.017 rotated (text: 0.0119
// and 0.0097 the mean of six, the FP16 kernel's 0.0091; RTX 4070, 2026-11-07).
template <int HD>
bool run_causal(sycl::queue& q, const uint16_t* q16, const Paged& k16, const Paged& v16, int64_t Tq, int64_t Tk,
                int64_t heads, int64_t kv_heads, int64_t pos0, uint16_t* out16) {
    const int kind = kind_of(q);
    if ((kind != 1 && kind != 2) || kv_heads <= 0 || heads % kv_heads != 0 || Tq > Tk || pos0 + Tq > Tk) return false;
    Bufs& b = causal_bufs_of(q);
    const int64_t C = heads * HD, CK = kv_heads * HD, Tkp = (Tk + CH - 1) / CH * CH, Tqp = (Tq + CH - 1) / CH * CH;
    if (!b.grow(Tk, CK, kv_heads, Tq, C, heads, true)) return false;   // the keys' rows, the queries'
    key_stats<Paged>(q, k16, v16, Tk, CK, b.part, b.mean, b.sv);
    prep_i8<HD, Paged>(q, nullptr, k16, v16, b.mean, b.sv, Tk, Tkp, kv_heads, 0.f, nullptr, b.k, b.vt, nullptr, b.sk,
                       true);
    prep_i8<HD>(q, q16, q16, q16, b.mean, b.sv, Tq, Tqp, heads, 1.4426950408889634f / std::sqrt((float) HD), b.q,
                nullptr, nullptr, b.sq, nullptr, true);
    const float* sv = b.sv;
    if (kind == 1) {
        using K = Kernel<true, HD, HD == 128 ? QT : 1>;
        const int64_t rows = (int64_t) NW * 16 * K::QT;
        const size_t groups = (size_t) ((Tq + rows - 1) / rows);
        q.submit([&](sycl::handler& h) {
            K k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, Tk, Tkp, heads, 0, 0, Tq, Seg{},
                sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * CH * K::KLD)), h),
                sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * HD * VLD)), h),
                sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * CH)), h),
                sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * HD)), h), kv_heads, pos0};
            h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * NW * 32}, {1, (size_t) NW * 32}), k);
        });
        return true;
    }
    const int64_t rows = (int64_t) NNW * 16 * NQ;
    const size_t groups = (size_t) ((Tq + rows - 1) / rows);
    q.submit([&](sycl::handler& h) {
        NvKernel<true, HD> k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, Tk, Tkp, heads, 0, 0, Tq, Seg{},
                             sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * NK * (HD + 16))), h),
                             sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * HD * NVLD)), h),
                             sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (4 * NK)), h),
                             sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * HD)), h), kv_heads, pos0};
        h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * NNW * 32}, {1, (size_t) NNW * 32}), k);
    });
    return true;
}

}  // namespace i8
}  // namespace

#if STRATA_HIP_KITCHEN
// Whether comfy-kitchen's attention runs instead of XeStrata's own on q's GPU, where both run: the faster, timed once
// a GPU at an image DiT's shape (4096 rows of 24 heads), the least of three after a warm-up of each.
// XESTRATA_I8_ATTN_KITCHEN=0 or 1 fixes it.
static bool kitchen_attn_first(sycl::queue& q) {
    static core::PerDevice<int> of;
    return of.get(q.get_device(), [&q] {
        if (const char* e = strata::env_get("I8_ATTN_KITCHEN")) return std::atoi(e) != 0 ? 1 : 0;
        if (i8::kind_of(q) == 0) return 1;
        constexpr int64_t T = 4096, H = 24, C = H * D;
        auto* x = sycl::malloc_device<uint16_t>((size_t) (4 * T * C), q);
        if (x == nullptr) return 1;
        // values of a few units, varied (equal ones would make every score the same)
        q.parallel_for(sycl::range<1>((size_t) (4 * T * C)), [=](sycl::id<1> id) {
             const uint32_t h = (uint32_t) id[0] * 2654435761u;
             x[id[0]] = f16_gpu::f16_from_f32((float) ((h >> 20) & 255) / 64.f - 2.f);
         }).wait();
        const uint16_t *q16 = x, *k16 = x + T * C, *v16 = x + 2 * T * C;
        uint16_t* out = x + 3 * T * C;
        auto best = [&](const std::function<bool()>& f) {
            double t = 1e30;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!f()) return 1e30;
                q.wait();
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (i > 0) t = std::min(t, dt);
            }
            return t;
        };
        const double kit = best([&] { return hip_kitchen::attn(q, q16, k16, v16, T, H, D, 0, 0, out); });
        const double own = best([&] { return i8::run(q, q16, k16, v16, T, H, 0, 0, T, out); });
        sycl::free(x, q);
        const int k = kit <= own ? 1 : 0;
        std::fprintf(stderr, "xestrata: the int8 attention: %s (4096 rows of 24 heads: comfy-kitchen's %.2f ms, "
                             "XeStrata's own %.2f)\n", k ? "comfy-kitchen's" : "XeStrata's own", 1e3 * kit, 1e3 * own);
        return k;
    }) == 1;
}
#endif

int flash_attn_i8_kind(void* stream) { return i8::kind_of(core::Runtime::get().stream(stream)); }

int flash_attn_i8_chunk() { return i8::CH; }

bool flash_attn_i8(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                   int64_t r0, int64_t first, uint16_t* out16, void* stream, const FlashAttnI8Stats* st) {
    if (T <= r0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
#if STRATA_HIP_KITCHEN
    // comfy-kitchen's (AMD GPUs with WMMA): its own statistics (K smoothed by an anchor key, V's columns' scales)
    if (kitchen_attn_first(q) && hip_kitchen::attn(q, q16, k16, v16, T, heads, D, r0, first, out16)) {
        if (!stream) q.wait();
        return true;
    }
#endif
    if (!i8::run(q, q16, k16, v16, T, heads, r0, first, T, out16, st)) return false;
    if (!stream) q.wait();
    return true;
}

bool flash_attn_i8_causal(const uint16_t* q16, const QsaAttnPools& pools, int64_t page_size, int64_t Tq, int64_t Tk,
                          int64_t heads, int64_t kv_heads, int64_t head_dim, int64_t pos0, uint16_t* out16,
                          void* stream) {
    if (Tq <= 0) return true;
    if (pools.page_table == nullptr || (head_dim != 128 && head_dim != 256)) return false;
    i8::Paged k, v;
    k.table = v.table = pools.page_table;
    k.kv_heads = v.kv_heads = kv_heads;
    k.ps = v.ps = page_size;
    k.hd = v.hd = head_dim;
    k.f16 = pools.k_pool, v.f16 = pools.v_pool;
    if (k.f16 == nullptr) k.i8 = pools.k_q, k.sc = pools.k_scale, k.q4 = k.i8 == nullptr ? pools.k_q4 : nullptr;
    if (v.f16 == nullptr) v.i8 = pools.v_q, v.sc = pools.v_scale, v.q4 = v.i8 == nullptr ? pools.v_q4 : nullptr;
    auto none = [](const i8::Paged& p) { return p.f16 == nullptr && p.i8 == nullptr && p.q4 == nullptr; };
    if (none(k) || none(v)) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    const bool ok = head_dim == 128 ? i8::run_causal<128>(q, q16, k, v, Tq, Tk, heads, kv_heads, pos0, out16)
                                    : i8::run_causal<256>(q, q16, k, v, Tq, Tk, heads, kv_heads, pos0, out16);
    if (!ok) return false;
    if (!stream) q.wait();
    return true;
}

bool flash_attn_i8_causal_prepare(int64_t Tq, int64_t Tk, int64_t heads, int64_t kv_heads, int64_t head_dim,
                                  void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (i8::kind_of(q) != 1 && i8::kind_of(q) != 2) return false;
    return i8::causal_bufs_of(q).grow(Tk, kv_heads * head_dim, kv_heads, Tq, heads * head_dim, heads, true);
}

void flash_attn_i8_causal_release(void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    i8::CausalOf& of = core::Runtime::of_queue(q).queue_item<i8::CausalOf>(q);
    std::lock_guard<std::mutex> g(of.mu);
    q.wait();
    of.b.release();
}

uint64_t flash_attn_i8_prefix_bytes(int64_t P, int64_t heads) { return i8::Prefix::bytes(P, heads); }

bool flash_attn_i8_prefix(const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* prefix,
                          void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (P <= 0 || !i8::make_prefix(q, k16, v16, P, heads, prefix)) return false;
    if (!stream) q.wait();
    return true;
}

bool flash_attn_i8_prefixed(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                            const void* prefix, int64_t P, const float* v_from, const float* v_to, uint16_t* out16,
                            void* stream) {
    if (T <= 0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (P <= 0 || !i8::run_prefixed(q, q16, k16, v16, T, heads, prefix, P, v_from, v_to, out16)) return false;
    if (!stream) q.wait();
    return true;
}

}  // namespace strata::kernels
