// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn.cpp - the DiT's attention on the GPU's matrix engines (qi_kernels.hpp: qi_attention).
//
// The 16-bit attention is the kernel the plain models' prompt shares (strata/kernels/flash_attn.hpp, on the
// matrix engines); Q, K and V hold their rows rounded up to a multiple of 128 (qi_attn_rows), the rows past T zero.
// A GPU without matrix engines, or STRATA_NO_XMX=1, takes its FMA form.  The int8 attention below
// (SageAttention's scheme) is the image rows' own.
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/elementwise.hpp"
#include "../kernels/xe/attn_parts.hpp"
#include "../kernels/xe/amd_wmma.hpp"
#include "../kernels/xe/device_target.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace strata::image {

namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using kernels::xe::attn::at;      // the attention kernels' shared parts (attn_parts.hpp)
using kernels::xe::attn::built;
using kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using kernels::f16_gpu::f32_from_f16;

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
// scales
struct Bufs {
    int8_t *q = nullptr, *k = nullptr, *vt = nullptr;
    float *sq = nullptr, *sk = nullptr, *sv = nullptr, *mean = nullptr, *part = nullptr, *sw = nullptr;
    int64_t cap_t = 0, cap_c = 0;
    bool grow(int64_t T, int64_t C, int64_t heads) {
        const int64_t Tp = (T + CH - 1) / CH * CH;
        if (Tp <= cap_t && C <= cap_c) return true;
        for (void* p : {(void*) q, (void*) k, (void*) vt, (void*) sq, (void*) sk, (void*) sv, (void*) mean, (void*) part,
                        (void*) sw})
            if (p != nullptr) strata::gpu::free(p);
        *this = Bufs{};
        if (!strata::gpu::alloc_device(&q, (size_t) (Tp * C)) || !strata::gpu::alloc_device(&k, (size_t) (Tp * C)) ||
            !strata::gpu::alloc_device(&vt, (size_t) (Tp * C)) || !strata::gpu::alloc_device(&sq, (size_t) (Tp * heads) * 4) ||
            !strata::gpu::alloc_device(&sk, (size_t) (Tp * heads) * 4) || !strata::gpu::alloc_device(&sv, (size_t) C * 4) ||
            !strata::gpu::alloc_device(&mean, (size_t) C * 4) || !strata::gpu::alloc_device(&sw, (size_t) C * 4) ||
            !strata::gpu::alloc_device(&part, (size_t) 2 * SLICES * C * 4))
            return false;
        cap_t = Tp;
        cap_c = C;
        return true;
    }
};

Bufs& bufs_of(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<Bufs>> of;
    return *of.get(q.get_device(), [] { return std::make_shared<Bufs>(); });
}

// NVIDIA's mma.sync from sm_80 on, in PTX (NvKernel): m16n8k32 s8 . s8 into s32, ldmatrix for the B operands out of
// local memory and cp.async for the chunks into it
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
#define STRATA_NV_I8 1
inline void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
inline uint32_t smem_u32(const void* p) {
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
inline void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
inline void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
inline void cp_commit() { asm volatile("cp.async.commit_group;"); }
inline uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
template<int N>
inline void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
// 2^x flushing a denormal result to 0: sycl::native::exp2 is ex2.approx.f32 without .ftz, which sm_89 runs as a check,
// two multiplies and the instruction (20% of the kernel's instructions, RTX 4070, 2026-11-05); a probability below
// 2^-126 is 0 in int8 either way
inline float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#else
#define STRATA_NV_I8 0
#endif

// Which of the int8 kernels the device takes: 1 Kernel (RDNA3, RDNA4), 2 NvKernel (NVIDIA from sm_80 on, where the
// binary the runtime picked for it carries the kernel: an older target's code JIT-compiled for it does not), else 0
int kind_of(sycl::queue& q) {
    static core::PerDevice<int> of;
    return of.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        if (strata::kernels::xe::amd_wmma(d)) return 1;
        if (d.get_backend() != sycl::backend::ext_oneapi_cuda) return 0;
        const auto a = static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>());
        if (a < static_cast<uint64_t>(syclex::architecture::nvidia_gpu_sm_80) || a >= 0x0200000000000000ull) return 0;
        int* f = sycl::malloc_device<int>(1, q);
        if (f == nullptr) return 0;
        int v = 0;
        q.single_task([=] { *f = STRATA_NV_I8; }).wait();
        q.memcpy(&v, f, sizeof v).wait();
        sycl::free(f, q);
        return v == 1 ? 2 : 0;
    });
}

// the keys' sums and the values' largest magnitudes over slices of the keys: part [2][SLICES][C]
void key_stats(sycl::queue& q, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t C, float* part,
               float* mean, float* sv) {
    q.parallel_for(sycl::range<2>((size_t) SLICES, (size_t) C), [=](sycl::id<2> id) {
        const int64_t sl = (int64_t) id[0], c = (int64_t) id[1];
        const int64_t t0 = T * sl / SLICES, t1 = T * (sl + 1) / SLICES;
        float sum = 0.f, mx = 0.f;
        for (int64_t t = t0; t < t1; ++t) {
            sum += f32_from_f16(k16[t * C + c]);
            mx = sycl::fmax(mx, sycl::fabs(f32_from_f16(v16[t * C + c])));
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
        mean[c] = sum / (float) T;
        sv[c] = mx > 0.f ? mx / 127.f : 1.f;
    });
}

// Q's and K's rows (a (row, head)'s 128 values, less `mean` for K) as int8 with their scales (Q's times `qmul`), and V
// as int8 by its (head, dim) scales, transposed into vt [heads][Tp / CH][D][CH] (a chunk's V^T in one piece, which the
// attention reads at once: a dim's CH bytes Tp apart were 2% slower on the RX 9060 XT): one kernel, a work-group a head
// and RB keys, V^T built in local memory so its rows go out RB bytes at a time; rows and keys past T up to Tp zero.
// q16 null: the keys and values alone (make_prefix).  NvKernel's V^T (kind_of 2) holds each 16 keys in the order its
// P operand takes them: at 4 i + j the key 2 i + (j & 1) + 8 (j >> 1).
void prep_i8(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* mean,
             const float* sv, int64_t T, int64_t Tp, int64_t heads, float qmul, int8_t* q8, int8_t* k8, int8_t* vt,
             float* sq, float* sk) {
    const bool with_q = q16 != nullptr, nv = kind_of(q) == 2;
    const int64_t nc = Tp / CH;
    constexpr int RB = 128, WG = 256, TLD = RB + 4;   // V^T's rows in the tile: 4 lanes' dims a bank apart
    const int64_t C = heads * D;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> tile(sycl::range<1>((size_t) (D * TLD)), h);
        // the heads the fast dimension: the groups at work together read whole rows, not one head's slice of
        // rows C * 2 bytes apart
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) ((Tp + RB - 1) / RB), (size_t) (heads * WG)),
                                         sycl::range<2>(1, WG)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t hd = (int64_t) it.get_group(1), t0 = (int64_t) it.get_group(0) * RB;
            const int lid = (int) it.get_local_id(1);
            sycl::sub_group sg = it.get_sub_group();
            const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            // Q and K: a wave a row at a time, 4 dims a lane; V into the tile by its scales
            float isv[4], mn[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                isv[u] = 1.f / sv[(int64_t) hd * D + (int64_t) lane * 4 + u];
                mn[u] = mean[(int64_t) hd * D + (int64_t) lane * 4 + u];
            }
            for (int tt = wave; tt < sycl::min<int64_t>(RB, Tp - t0); tt += WG / 32) {
                const int64_t t = t0 + tt, base = t * C + (int64_t) hd * D + (int64_t) lane * 4;
                const bool in = t < T;
                const sycl::vec<uint16_t, 4> z(0);
                const bool inq = in && with_q;
                const sycl::vec<uint16_t, 4> xq = inq ? *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(q16 + base)
                                                      : z,
                                             xk = in ? *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(k16 + base) : z,
                                             xv = in ? *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(v16 + base) : z;
                float vq[4], vk[4];
                float mq = 0.f, mk = 0.f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    vq[u] = f32_from_f16(xq[u]);
                    vk[u] = in ? f32_from_f16(xk[u]) - mn[u] : 0.f;
                    tile[(lane * 4 + u) * TLD + tt] = (int8_t) sycl::rint(f32_from_f16(xv[u]) * isv[u]);
                    mq = sycl::fmax(mq, sycl::fabs(vq[u]));
                    mk = sycl::fmax(mk, sycl::fabs(vk[u]));
                }
                mq = sycl::reduce_over_group(sg, mq, sycl::maximum<float>());
                mk = sycl::reduce_over_group(sg, mk, sycl::maximum<float>());
                const float cq = mq > 0.f ? mq / 127.f : 1.f, ck = mk > 0.f ? mk / 127.f : 1.f;
                const float iq = 1.f / cq, ik = 1.f / ck;
                uint32_t wq = 0, wk = 0;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    wq |= (uint32_t) (uint8_t) (int8_t) sycl::rint(vq[u] * iq) << (8 * u);
                    wk |= (uint32_t) (uint8_t) (int8_t) sycl::rint(vk[u] * ik) << (8 * u);
                }
                if (with_q) *reinterpret_cast<uint32_t*>(q8 + base) = wq;
                *reinterpret_cast<uint32_t*>(k8 + base) = wk;
                if (lane == 0) {
                    if (with_q) sq[t * heads + hd] = cq * qmul;
                    sk[t * heads + hd] = ck;
                }
            }
            sycl::group_barrier(it.get_group());
            const int nk = (int) sycl::min<int64_t>(RB, Tp - t0);
            for (int i = lid; i < D * RB / 4; i += WG) {   // V^T along the keys, 4 a work-item
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
                const int64_t at = (((int64_t) hd * nc + (t0 + tt) / CH) * D + d) * CH + (t0 + tt) % CH;
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

struct Kernel {
    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<int8_t, 1> kl, vl;       // two chunks' K [CH][KLD] and V^T [D][VLD]
    sycl::local_accessor<float, 1> skl, svl;      // two chunks' key scales [CH], the head's V scales [D] (and the
                                                  // second segment's turn [D] after them)

    // the work-group as an attribute, not a property: intel/llvm bounds an AMD kernel's registers by it from the
    // attribute only (see iq_mmq.cpp), and the 32 queries a wave need more than the 192 VGPRs of a 1024-lane group
    [[sycl::reqd_work_group_size(1, NW * 32), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    __attribute__((always_inline)) void operator()(sycl::nd_item<2> it) const {
#if STRATA_AMD_WMMA
        using i8v = int __attribute__((ext_vector_type(8)));
        using fv8 = float __attribute__((ext_vector_type(8)));
        using u4 = sycl::vec<uint32_t, 4>;
#if defined(__GFX12__)
        using frag = int __attribute__((ext_vector_type(2)));
        constexpr int KOFF = 8;
#else
        using frag = int __attribute__((ext_vector_type(4)));
        constexpr int KOFF = 0;
#endif
        auto wmma = [](const frag& a, const frag& b, const i8v& c) {
#if defined(__GFX12__)
            return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
            return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#endif
        };
        sycl::sub_group sg = it.get_sub_group();
        const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int lo = lane & 15, hi = lane >> 4, lid = (int) it.get_local_id(1);
        auto row_of = [&](int e) {   // the accumulator's row of element e (the lane's column is lo)
#if defined(__GFX12__)
            return 8 * hi + e;
#else
            return 2 * e + hi;
#endif
        };
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        // the wave's QT tiles of 16 queries: my0 + 16 qt + lo
        const int64_t my0 = r0 + (int64_t) it.get_group(1) * NW * 16 * QT + (int64_t) wave * 16 * QT;
        const bool live = my0 < row_end;
        int8_t* K0 = kl.get_multi_ptr<sycl::access::decorated::no>().get();
        int8_t* V0 = vl.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = lid; i < D; i += NW * 32) {
            svl[i] = sv[h * D + i];
            if (b.T > 0) svl[D + i] = b.sw[h * D + i];
        }
        // the queries as B operands of S^T (column lo: a query, K values kk * 16 + hi KOFF ..), their scales
        frag qb[QT][D / 16];
        float sql[QT], m[QT], l[QT];
        fv8 o[QT][D / 16];   // O^T: rows the dims nt * 16 + row_of(e), column the lane's query
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
            const int64_t qr = sycl::min<int64_t>(my0 + 16 * qt + lo, Tp - 1);
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
        // chunk ci: the first segment's ci, else the second's ci - na
        const int64_t na = Tp / CH, nch = na + b.Tp / CH;
        auto fetch = [&](int64_t ci) {
            const bool sb = ci >= na;
            const int64_t j0 = (sb ? ci - na : ci) * CH, lim = sb ? b.T : T, nc = sb ? b.Tp / CH : na;
            const int8_t *kp = sb ? b.k : k8, *vp = sb ? b.vt : vt;
            const float* sp = sb ? b.sk : sk;
#pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                const int i = lid + p * NW * 32;
                if (i >= KV16) break;
                const int kr = i / (D / 16), c = i % (D / 16) * 16;
                rk[p] = *reinterpret_cast<const u4*>(kp + (j0 + kr) * C + h * D + c);
                const int d = i / (CH / 16), cv = i % (CH / 16) * 16;
                rv[p] = *reinterpret_cast<const u4*>(vp + ((h * nc + j0 / CH) * D + d) * CH + cv);
            }
            if (lid < CH) rs = j0 + lid < lim ? sp[(j0 + lid) * heads + h] : 0.f;
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
          if (live) {   // a wave past the rows shares the loads, nothing else
            const bool sb = ci >= na;
            const int64_t j0 = (sb ? ci - na : ci) * CH, lim = sb ? b.T : T;
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
                        s[qt][nt][e] = in ? (float) c[qt][e] * ks : -INFINITY;
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
                const float off = 6.98868468677217f - cmx[qt];   // log2 127: the values times 127 in exp2's argument
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
// an SM, they spilled: 4.9 ms)
constexpr int NQ = 1;             // query tiles of 16 a warp
constexpr int NNW = 8;            // warps a work-group
constexpr int SC = 2;             // the layout's chunks a step
constexpr int NK = SC * CH;       // keys a step
constexpr int NVLD = NK + 16;     // a dim's row of V^T in local memory (bytes; 8 rows of an ldmatrix in distinct banks)

struct NvKernel {
    const int8_t *q8, *k8, *vt;
    const float *sq, *sk, *sv;
    uint16_t* out16;
    int64_t T, Tp, heads, r0, first, row_end;
    Seg b;
    sycl::local_accessor<int8_t, 1> kl, vl;       // two steps' K [NK][KLD] and V^T [D][NVLD]
    sycl::local_accessor<float, 1> skl, svl;      // two steps' key scales and their -kMagicF multiples [NK][2], the
                                                  // head's V scales [D] (and the second
                                                  // segment's turn [D] after them)

    [[sycl::reqd_work_group_size(1, NNW * 32), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    void operator()(sycl::nd_item<2> it) const {
#if STRATA_NV_I8
        constexpr int kMagic = 0x4b400000;      // 1.5 * 2^23's bits
        constexpr float kMagicF = 12582912.f;   // 1.5 * 2^23
        sycl::sub_group sg = it.get_sub_group();
        const int wave = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int g = lane >> 2, t4 = lane & 3, lid = (int) it.get_local_id(1);
        const int64_t h = (int64_t) it.get_group(0), C = heads * D;
        const int64_t my0 = r0 + (int64_t) it.get_group(1) * NNW * 16 * NQ + (int64_t) wave * 16 * NQ;
        const bool live = my0 < row_end;
        int8_t* K0 = kl.get_multi_ptr<sycl::access::decorated::no>().get();
        int8_t* V0 = vl.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = lid; i < D; i += NNW * 32) {
            svl[i] = sv[h * D + i];
            if (b.T > 0) svl[D + i] = b.sw[h * D + i];
        }
        // the members as values (the lambdas below take them so, not through the kernel object)
        const int8_t *const ka = k8, *const kb2 = b.k, *const va = vt, *const vb2 = b.vt;
        const float *const ska = sk, *const skb = b.sk;
        const int64_t Ta = T, Tb = b.T;
        // the steps: the first segment's, then the second's, each from a chunk of CH keys
        const int64_t na = Tp / CH, nb = b.Tp / CH, sa = (na + SC - 1) / SC, nsteps = sa + (nb + SC - 1) / SC;
        auto step_of = [=](int64_t si, bool& sb, int64_t& c0, int& nc) {
            sb = si >= sa;
            c0 = (sb ? si - sa : si) * SC;
            nc = (int) sycl::min<int64_t>(SC, (sb ? nb : na) - c0);
        };
        // a step's K rows and V^T's chunks into buffer bf (16 bytes a copy), its key scales into a register: piece p
        // of a work-item is K's row kr0 + p KR at column kc0, and V^T's chunk p at dim vd0, keys vc0 (offsets from a
        // base each, not arrays: those were held in registers the kernel had none left for)
        constexpr int NT = NNW * 32, KP = NK * D / 16 / NT, VP = D * NK / 16 / NT, KR = NT / (D / 16);
        static_assert(KP * NT == NK * D / 16 && NT % (D / 16) == 0 && NT == D * CH / 16 && VP == NK / CH,
                      "whole pieces a work-item");
        const int kr0 = lid / (D / 16), vd0 = lid / (CH / 16);
        const uint32_t ks_at = (uint32_t) (kr0 * KLD + lid % (D / 16) * 16),
                       vs_at = (uint32_t) (vd0 * NVLD + lid % (CH / 16) * 16);
        const int64_t kg_at = (int64_t) kr0 * C + (int64_t) (lid % (D / 16)) * 16,
                      vg_at = (int64_t) vd0 * CH + (int64_t) (lid % (CH / 16)) * 16;
        const uint32_t ks0 = smem_u32(K0), vs0 = smem_u32(V0);
        float rs = 0.f;
        auto issue = [&](int64_t si, int bf) {
            bool sb;
            int64_t c0;
            int nc;
            step_of(si, sb, c0, nc);
            const int8_t* kp = (sb ? kb2 : ka) + c0 * CH * C + h * D + kg_at;
            const int8_t* vp = (sb ? vb2 : va) + (h * (sb ? nb : na) + c0) * D * CH + vg_at;
            const uint32_t kb = ks0 + (uint32_t) bf * NK * KLD + ks_at, vb = vs0 + (uint32_t) bf * D * NVLD + vs_at;
#pragma unroll
            for (int p = 0; p < KP; ++p)
                if (kr0 + p * KR < nc * CH) cp16(kb + (uint32_t) (p * KR * KLD), kp + (int64_t) p * KR * C);
#pragma unroll
            for (int p = 0; p < VP; ++p)
                if (p < nc) cp16(vb + (uint32_t) (p * CH), vp + (int64_t) p * D * CH);
            cp_commit();
            const int64_t key = c0 * CH + lid;
            if (lid < NK) rs = lid < nc * CH && key < (sb ? Tb : Ta) ? (sb ? skb : ska)[key * heads + h] : 0.f;
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
                const int64_t qr = sycl::min<int64_t>(my0 + (16 * qt + g + 8 * r), Tp - 1);
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
        if (lid < NK) {
            skl[(size_t) 2 * lid] = rs;
            skl[(size_t) 2 * lid + 1] = -kMagicF * rs;
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
            cp_wait<0>();
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
                const uint32_t kb = smem_u32(K0) + (uint32_t) bf * NK * KLD + koff,
                               vb = smem_u32(V0) + (uint32_t) bf * D * NVLD + voff;
                const float* sk_ = &skl[(size_t) bf * 2 * NK];
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
                // the scores, P and its sums; a step that ends past the segment's last key (edge) masks the keys
                // after it
                auto scores = [&](auto edge, int qt) __attribute__((always_inline)) {
                    float cmx[2] = {-INFINITY, -INFINITY};
                    float s[NK / 8][4];
#pragma unroll
                    for (int j = 0; j < NK / 8; ++j) {
                        // the two keys' scales and their -kMagicF multiples
                        const sycl::float4 ks = *reinterpret_cast<const sycl::float4*>(sk_ + (ptrdiff_t) 2 * (j * 8 + 2 * t4));
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int k = 2 * (e & 1);
                            s[j][e] = sycl::fma(sycl::bit_cast<float>(acc[qt][j][e]), ks[k], ks[k + 1]);
                            if constexpr (decltype(edge)::value)
                                if (j * 8 + 2 * t4 + (e & 1) >= kmax) s[j][e] = -INFINITY;
                            cmx[e >> 1] = sycl::fmax(cmx[e >> 1], s[j][e]);
                        }
                    }
                    float off[2], sum[2] = {0.f, 0.f};
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        cmx[r] = sycl::fmax(cmx[r], sycl::permute_group_by_xor(sg, cmx[r], 1));
                        cmx[r] = sycl::fmax(cmx[r], sycl::permute_group_by_xor(sg, cmx[r], 2)) * sql[qt][r];
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
                        ps[qt][r] = m[qt][r] == -INFINITY ? 0.f : ex2(cmx[r] - m[qt][r]) * (1.f / 127.f);
                        off[r] = 6.98868468677217f - cmx[r];
                    }
                    uint32_t w[NK / 8][2];   // tile j's row r's two values, 0 .. 127, in bytes 0 and 1
#pragma unroll
                    for (int j = 0; j < NK / 8; ++j)
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const int e0 = 2 * r;
                            const float p0 = ex2(sycl::fma(s[j][e0], sql[qt][r], off[r])),
                                        p1 = ex2(sycl::fma(s[j][e0 + 1], sql[qt][r], off[r]));
                            sum[r] += p0 + p1;
                            // rounded to the nearest (even) integer by the add, its low byte the value
                            w[j][r] = prmt(sycl::bit_cast<uint32_t>(p0 + kMagicF),
                                           sycl::bit_cast<uint32_t>(p1 + kMagicF), 0x0040u);
                        }
                    // A of P . V a 32 keys: row g's keys 4 t4 .. in V^T's order from tiles 4 kc and 4 kc + 1, then
                    // 16 + 4 t4 .. from 4 kc + 2 and 4 kc + 3; row g + 8's the same
#pragma unroll
                    for (int kc = 0; kc < NK / 32; ++kc)
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const int j0 = 4 * kc;
                            pa[qt][kc][r] = prmt(w[j0][r], w[j0 + 1][r], 0x5410u);
                            pa[qt][kc][2 + r] = prmt(w[j0 + 2][r], w[j0 + 3][r], 0x5410u);
                        }
#pragma unroll
                    for (int r = 0; r < 2; ++r) l[qt][r] += sum[r] * ps[qt][r];
                };
#pragma unroll
                for (int qt = 0; qt < NQ; ++qt) {
                    if (kmax == NK) scores(std::false_type{}, qt);
                    else scores(std::true_type{}, qt);
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
                            mma_s8(c[qt][0], pa[qt][kc], vf[0], vf[1]);
                            mma_s8(c[qt][1], pa[qt][kc], vf[2], vf[3]);
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
            if (si + 1 < nsteps && lid < NK) {   // the next step's key scales (its buffer was read last step)
                skl[(size_t) (bf ^ 1) * 2 * NK + (size_t) 2 * lid] = rs;
                skl[(size_t) (bf ^ 1) * 2 * NK + (size_t) 2 * lid + 1] = -kMagicF * rs;
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

void launch(sycl::queue& q, const Bufs& b, const float* sv, int64_t T, int64_t Tp, int64_t heads, int64_t r0,
            int64_t first, int64_t row_end, uint16_t* out16, const Seg& seg = {});

// The rows r0 .. row_end - 1 (which see every key) into out16 (its row 0 query row `first`) through the int8 kernel;
// false where it is not built for the device or its buffers do not fit (the caller takes the 16-bit path).
bool run(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
         int64_t r0, int64_t first, int64_t row_end, uint16_t* out16) {
    if (kind_of(q) == 0) return false;
    Bufs& b = bufs_of(q);
    const int64_t C = heads * D, Tp = (T + CH - 1) / CH * CH;
    if (!b.grow(T, C, heads)) return false;
    key_stats(q, k16, v16, T, C, b.part, b.mean, b.sv);
    prep_i8(q, q16, k16, v16, b.mean, b.sv, T, Tp, heads, 1.4426950408889634f / std::sqrt((float) D),   // log2 e / sqrt D
            b.q, b.k, b.vt, b.sq, b.sk);
    launch(q, b, b.sv, T, Tp, heads, r0, first, row_end, out16);
    return true;
}

// the kernel on the buffers prepared: T keys (then seg's), query rows r0 .. row_end - 1, the output at V's scales sv
void launch(sycl::queue& q, const Bufs& b, const float* sv, int64_t T, int64_t Tp, int64_t heads, int64_t r0,
            int64_t first, int64_t row_end, uint16_t* out16, const Seg& seg) {
    if (kind_of(q) == 2) {
        const int64_t rows = (int64_t) NNW * 16 * NQ;
        const size_t groups = (size_t) ((row_end - r0 + rows - 1) / rows);
        q.submit([&](sycl::handler& h) {
            NvKernel k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
                       sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * NK * KLD)), h),
                       sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (2 * D * NVLD)), h),
                       sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (4 * NK)), h),
                       sycl::local_accessor<float, 1>(sycl::range<1>((size_t) (2 * D)), h)};
            h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * NNW * 32}, {1, (size_t) NNW * 32}), k);
        });
        return;
    }
    const size_t groups = (size_t) ((row_end - r0 + (int64_t) NW * 16 * QT - 1) / ((int64_t) NW * 16 * QT));
    q.submit([&](sycl::handler& h) {
        Kernel k{b.q, b.k, b.vt, b.sq, b.sk, sv, out16, T, Tp, heads, r0, first, row_end, seg,
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

}  // namespace i8

}  // namespace

const char* qi_attention_path() {
    static const char* names[] = {"matrix engines 8x16x16", "matrix engines 8x8x16", "matrix engines 16x16x16",
                                  "WMMA 16x16x16"};
    const int p = kernels::flash_attn_path(&core::Runtime::get().device());
    return p < 0 ? "plain (FP32)" : names[p];
}

namespace {
thread_local bool g_no_i8 = false;   // the 16-bit path's run while the int8 one is timed against it

// qi_attention's choices of the int8 kernel, a GPU's: (image rows, keys) / 1024 -> int8
struct Choices {
    std::mutex mu;
    std::map<std::pair<int64_t, int64_t>, bool> of;
};
Choices& choices_of(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<Choices>> of;
    return *of.get(q.get_device(), [] { return std::make_shared<Choices>(); });
}
int forced_i8() {   // STRATA_IMAGE_ATTN_I8: 1 always, 0 never, else -1 (timed)
    static const int forced = [] {
        const char* v = std::getenv("STRATA_IMAGE_ATTN_I8");
        return v == nullptr || *v == 0 ? -1 : (int) (std::strtol(v, nullptr, 10) != 0);
    }();
    return forced;
}
}  // namespace

bool qi_attention_i8(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                     int64_t r0, int64_t first, uint16_t* out16, void* stream) {
    if (T <= r0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (!i8::run(q, q16, k16, v16, T, heads, r0, first, T, out16)) return false;
    if (!stream) q.wait();
    return true;
}

int qi_attention_i8_chunk() { return i8::CH; }

uint64_t qi_attention_i8_prefix_bytes(int64_t P, int64_t heads) { return i8::Prefix::bytes(P, heads); }

bool qi_attention_i8_prefix(const uint16_t* k16, const uint16_t* v16, int64_t P, int64_t heads, void* prefix,
                            void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (P <= 0 || !i8::make_prefix(q, k16, v16, P, heads, prefix)) return false;
    if (!stream) q.wait();
    return true;
}

bool qi_attention_i8_prefixed(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                              int64_t heads, const void* prefix, int64_t P, const float* v_from, const float* v_to,
                              uint16_t* out16, void* stream) {
    if (T <= 0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (P <= 0 || !i8::run_prefixed(q, q16, k16, v16, T, heads, prefix, P, v_from, v_to, out16)) return false;
    if (!stream) q.wait();
    return true;
}

bool qi_attention_i8_preferred(int64_t rows, int64_t keys, int64_t heads, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    const sycl::device dev = q.get_device();
    if (i8::kind_of(q) == 0 || kernels::flash_attn_path(&dev) < 0) return false;
    if (forced_i8() >= 0) return forced_i8() == 1;
    rows = std::max<int64_t>(rows, 2048);   // qi_attention times from 2048 rows on
    keys = std::max(keys, rows);
    Choices& ch = choices_of(q);
    const auto key = std::make_pair((rows + 1023) / 1024, (keys + 1023) / 1024);
    {
        std::lock_guard<std::mutex> lock(ch.mu);
        if (const auto it = ch.of.find(key); it != ch.of.end()) return it->second;
    }
    // not timed yet: qi_attention times it, on rows of zeros (the kernels' speed does not ask what the values are)
    const int64_t C = heads * D, TP = qi_attn_rows(keys);
    uint16_t *qkv = nullptr, *out = nullptr;
    if (!strata::gpu::alloc_device(&qkv, (size_t) (3 * TP * C) * 2) ||
        !strata::gpu::alloc_device(&out, (size_t) (rows * C) * 2)) {
        if (qkv != nullptr) strata::gpu::free(qkv);
        return false;
    }
    bool use = false;
    try {
        strata::gpu::memset_async(qkv, 0, (size_t) (3 * TP * C) * 2, stream);
        qi_attention(qkv, qkv + TP * C, qkv + 2 * TP * C, keys, heads, D, 0, out, stream, nullptr,
                     keys - rows, true);
        q.wait();
        std::lock_guard<std::mutex> lock(ch.mu);
        const auto it = ch.of.find(key);
        use = it != ch.of.end() && it->second;
    } catch (...) {
        strata::gpu::free(qkv);
        strata::gpu::free(out);
        throw;
    }
    strata::gpu::free(qkv);
    strata::gpu::free(out);
    return use;
}

void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, void* stream,
                  const int32_t* key_lim, int64_t first, bool int8,
                  const std::vector<std::pair<int64_t, int64_t>>* spans) {
    if (T <= first) return;
    if (head_dim != D) throw std::invalid_argument("qi_attention: head_dim 128");
    // the image rows (they see every key) through the int8 kernel where it is the faster, else the GPU maker's kernel
    // where there is one; the prefix's own rows, a few dozen, by the kernel below
    int64_t row_end = T;
    const int64_t r0 = std::max(first, n_text);
    // the int8 kernel: STRATA_IMAGE_ATTN_I8=1 always, 0 never; else the image rows timed in it and in the 16-bit kernel
    // that would take them (the GPU maker's, else the engine's own), once a GPU and a shape (the rows and the keys, in
    // steps of 1024: a generation's and an edit's differ), from 2048 image rows on (fewer: not taken)
    bool i8_done = false;
    if (int8 && T - r0 >= 256) {
        sycl::queue& qq = core::Runtime::get().stream(stream);
        const int forced = forced_i8();
        Choices& ch = choices_of(qq);
        const auto key = std::make_pair((T - r0 + 1023) / 1024, (T + 1023) / 1024);
        auto run_i8 = [&] { return i8::run(qq, q16, k16, v16, T, heads, r0, first, T, out16); };
        int use = forced;
        if (use < 0) {
            std::lock_guard<std::mutex> lock(ch.mu);
            const auto it = ch.of.find(key);
            if (it != ch.of.end()) use = it->second;
            else if (T - r0 >= 2048 && i8::kind_of(qq) != 0) {
                auto timed = [&](const std::function<bool()>& f) {
                    if (!f()) return -1.0;
                    qq.wait();
                    const auto t0 = std::chrono::steady_clock::now();
                    f();
                    qq.wait();
                    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                };
                const double i8ms = timed(run_i8);
                double f16ms = timed([&] {   // the maker's kernel on the image rows, else the engine's on all asked
                    return qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim, out16 + (r0 - first) * heads * head_dim,
                                   stream);
                });
                if (f16ms < 0.0) {
                    g_no_i8 = true;
                    f16ms = timed([&] {
                        qi_attention(q16, k16, v16, T, heads, head_dim, n_text, out16, stream, key_lim, first);
                        return true;
                    });
                    g_no_i8 = false;
                }
                // int8 unless it is clearly the slower: a near tie (the RX 9060 XT, Qwen-Image's 4096 rows, 2026-10-29:
                // 6.58 against 6.66 ms) picked one or the other from start to start, and with it the image
                use = i8ms > 0.0 && i8ms < f16ms * 1.05;
                ch.of[key] = use;
                std::fprintf(stderr, "strata-image: the attention's image rows %s at %lld rows of %lld keys (int8 "
                                     "%.2f ms, 16-bit %.2f ms)\n", use ? "in int8" : "in 16 bits", (long long) (T - r0),
                             (long long) T, i8ms, f16ms);
            } else {
                use = 0;
            }
        }
        if (use == 1 && !g_no_i8) i8_done = run_i8();
    }
    // the image rows went through the int8 kernel or else the maker's 16-bit one; the text rows are left
    if (i8_done || (T - r0 >= 256 && qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim,
                                             out16 + (r0 - first) * heads * head_dim, stream))) {
        if (first >= n_text) {
            if (!stream) core::Runtime::get().stream(stream).wait();
            return;
        }
        row_end = n_text;
    }
    sycl::queue& q = core::Runtime::get().stream(stream);
    const sycl::device dev = q.get_device();
    const int p = kernels::flash_attn_path(&dev);
    // The sub-groups a work-group: more share the keys' tiles in the caches, fewer leave the GPU freer to overlap
    // work-groups.  The fastest of the most that fit, a half and a quarter of it, timed once a GPU on the first
    // attention of 2048 rows or more (the images' own inputs; a small first image would not show it): 1024 x 1024
    // block, the B70 4 (11.88 -> 11.44 ms), the RTX 4070 2 (11.55 -> 10.53 ms).  Until then, the most.
    static core::PerDevice<std::shared_ptr<std::atomic<int>>> tuned_of;
    std::atomic<int>& tuned = *tuned_of.get(q.get_device(), [] { return std::make_shared<std::atomic<int>>(0); });
    // rows f .. e - 1 by the shared kernel (flash_attn.hpp), the dense layout; `fma`: its FMA form (no matrix engines)
    auto run = [&](int nsg, int64_t f, int64_t e, bool fma = false) {
        kernels::FlashAttnArgs a;
        a.q = q16;
        a.q_ld = heads * D;
        a.k = k16;
        a.v = v16;
        a.kv_ld = heads * D;
        a.n_head = a.n_kv = heads;
        a.head_dim = D;
        a.rows = a.keys = T;
        a.n_causal = n_text;
        a.key_lim = key_lim;
        a.first = f;
        a.row_end = e;
        a.out = out16 + (f - first) * heads * head_dim;
        a.out_ld = heads * D;
        a.scale = 1.0f / std::sqrt((float) D);
        a.fma = fma;
        return kernels::flash_attn(a, stream, nsg);
    };
    // The references' rows (spans) as image rows past a context of b keys, through the image rows' kernels (the int8
    // one or the GPU maker's, timed as the image rows' are; an edit's 4096 rows of a reference, 4104 keys: 22.8 ms a
    // block in the kernel below, 6.3 in AMD's on the RX 9060 XT); the rows between them, the text's, by the kernel below
    std::vector<std::pair<int64_t, int64_t>> segs;   // the rows left to the kernel below
    {
        int64_t f = first;
        if (p >= 0 && spans != nullptr)
            for (const auto& [a, b] : *spans) {
                if (a < f || b > std::min(row_end, n_text)) continue;   // a span asked only in part: the kernel below
                qi_attention(q16, k16, v16, b, heads, head_dim, a, out16 + (a - first) * heads * head_dim,
                             stream, nullptr, a, int8);
                if (a > f) segs.emplace_back(f, a);
                f = b;
            }
        if (row_end > f) segs.emplace_back(f, row_end);
    }
    const int most = p >= 0 ? kernels::flash_attn_nsg_most(&dev, D) : 0;
    int nsg = tuned.load();
    const auto big = std::max_element(segs.begin(), segs.end(), [](const auto& x, const auto& y) {
        return x.second - x.first < y.second - y.first;
    });
    if (most > 0 && nsg == 0 && big != segs.end() && big->second - big->first >= 2048) {
        q.wait();
        double best = 1e30;
        std::string said;
        for (int n = most; n >= std::max(1, most / 4); n /= 2) {
            run(n, big->first, big->second);   // its kernel built, its caches warm
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            run(n, big->first, big->second);
            q.wait();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            said += (said.empty() ? "" : ", ") + std::to_string(n) + ": " + std::to_string(ms).substr(0, 5) + " ms";
            if (ms < best) { best = ms; nsg = n; }
        }
        tuned.store(nsg);
        std::fprintf(stderr, "strata-image: the attention takes %d sub-groups a work-group (timed at %lld rows of %lld "
                             "keys: %s)\n", nsg, (long long) (big->second - big->first), (long long) T, said.c_str());
    }
    if (nsg == 0) nsg = most;
    bool done = most > 0;
    for (const auto& [f, end] : segs)
        if (done) done = run(nsg, f, end);
    if (!done && !run(0, first, row_end, true))
        throw std::invalid_argument("qi_attention: no form of the attention takes the shape");
    if (!stream) q.wait();
}

}  // namespace strata::image
