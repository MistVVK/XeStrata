// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/yskn_mma.cpp - the decode's .yskn product on the int8 matrix units (yskn_mma.hpp,
// strata/kernels/yskn.hpp), written anew in SYCL from YASAKANI's lnord.cu and a8_prologue.cuh as the DP4a form
// (src/kernels/xe/yskn.cpp) is, on the matrix parts the engine already shares: joint_matrix's int8 tiles (NVIDIA's
// tensor cores, the shape from the device's report as mma_gemm.cpp takes it), Intel's XMX (the VNNI-packed B of
// iq_mmq.cpp's intel_body), RDNA3's and RDNA4's int8 WMMA (amd_wmma.hpp, the layouts iq_mmq.cpp's and attn_parts.hpp
// write out).  The same integer sums as the DP4a form, the same bits: a work-group decodes its 64 output rows' codes
// through the int8 table into local memory, the matrix units take the exact int32 dot of each 512-group, and the
// group's fixed-point scale (yskn_sum.hpp) adds it in int64 - the form changes the multiply, not the sums.
//
// A work-group: 64 output rows (WN) by all the window's rows (up to kYsknMmaRows = 16, zero past the last), the
// columns divided among n_workers as the DP4a form's (a work-group a (row tile, worker) pair, its int64 partial sums
// to part[worker][row][output]).  K advances a unit (256 values) at a time, two a 512-group: the work-group decodes
// the unit's codes (a lane half a row's unit, the words the device order gives) and copies the activations into local
// memory, the sub-groups multiply the half into their int32 tiles, and after the group's two halves the tiles go to
// local memory row by row, where every lane scales its (row, output) elements and adds them in int64.  The store
// through local memory is what keeps the three matrix forms' accumulator layouts out of the sums: the tail reads
// (row, output) by index, the same on every device.
#include "yskn_mma.hpp"

#include "amd_wmma.hpp"
#include "device_caps.hpp"
#include "device_target.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/trellis.hpp"
#include "strata/kernels/yskn_sum.hpp"
#include "strata/platform/env.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_yskn_mma
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace ys = strata::kernels::yskn;
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)

// AMD's device compile has joint_matrix on CDNA's matrix cores only: RDNA3 and RDNA4 take their int8 WMMA instead
// (STRATA_AMD_WMMA), the other devices joint_matrix (as iq_mmq.cpp's guards)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define YSKN_MMA_JOINT 0
#define YSKN_MMA_WMMA STRATA_AMD_WMMA
#else
#define YSKN_MMA_JOINT 1
#define YSKN_MMA_WMMA 0
#endif
// Intel's XMX shapes (the VNNI-packed B): Intel's device compile and the host's, never NVIDIA's or AMD's
#define YSKN_MMA_INTEL (YSKN_MMA_JOINT && !STRATA_DEVICE_NOT_INTEL)

constexpr int kGroup = 512;
constexpr int kStates = 4096;
constexpr int WN = 64;                 // output rows a work-group (the device order's rows are a multiple of 64)
constexpr int LANES = 128;             // the work-group's work-items
constexpr int KH = 256;                // K a half: one unit
constexpr int LDA = KH + 16;           // the activations' local row stride (16-byte multiples, rows apart in the banks)
constexpr int LDB = KH + 16;           // the weights' (the 16 x 16 forms' column-major B)
constexpr int ACC_N = kYsknMmaRows * WN / LANES;   // a lane's int64 accumulators (8)
constexpr int up128(int b) { return (b + 127) / 128 * 128; }
// the local memory: the int8 table, the activations (16 rows), the weights (the larger of the two B layouts), the
// int32 tiles on their way out (32-byte aligned for the matrix loads)
constexpr int LUT_OFF = 0, A_OFF = up128(LUT_OFF + kStates * 2), B_OFF = up128(A_OFF + kYsknMmaRows * LDA),
              C_OFF = up128(B_OFF + WN * LDB), LOCAL = up128(C_OFF + kYsknMmaRows * WN * 4);

template<typename T>
inline auto local_as(sycl::multi_ptr<uint8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}

// the state of step j: the shared trellis window (kernels/trellis.hpp, the DP4a form's and the reader's own)
using strata::kernels::trellis::state_of;

// the kernel's fields (the matrix form's state is the Ops', below)
struct Fields {
    const uint32_t* words;
    const uint16_t* lut8;
    const int8_t* xq;
    const float* alpha;
    int64_t* part;
    int rows, cols, T, nw;
    sycl::local_accessor<uint8_t, 1> lm;
};

// The shared skeleton: the table and the rows' factors, then each 512-group's two halves (the codes decoded and the
// activations copied into local memory, the matrix units multiplying), the tiles out through local memory and the
// int64 tail.  Ops: the form's tiles (zero, half, store), its VNNI saying how the decoded weights are laid out
template<class Ops, int K>
inline void gemv_body(sycl::nd_item<1> it, const Fields& f, Ops& ops) {
    const int lid = (int) it.get_local_id(0);
    const auto sg = it.get_sub_group();
    const int gid = (int) it.get_group(0), tile = gid / f.nw, wk = gid % f.nw, r0 = tile * WN;
    const auto base = f.lm.template get_multi_ptr<sycl::access::decorated::no>();
    auto lut = local_as<uint16_t>(base + LUT_OFF);
    auto al = local_as<int8_t>(base + A_OFF);
    auto bl = local_as<int8_t>(base + B_OFF);
    [[maybe_unused]] auto bw = local_as<uint32_t>(base + B_OFF);   // the VNNI layout's words (the XMX form only)
    auto cp = local_as<int32_t>(base + C_OFF);
    for (int i = lid; i < kStates; i += LANES) lut[i] = f.lut8[i];
    const int G = f.cols / kGroup;
    float fac[kYsknMmaRows];
    for (int t = 0; t < kYsknMmaRows; ++t) {
        float m = 0.f;
        if (t < f.T)
            for (int g = 0; g < G; ++g) m = sycl::fmax(m, f.alpha[t * G + g]);
        fac[t] = ys::fixed_factor(m);
    }
    int64_t acc[ACC_N] = {};
    const int gpw = G / f.nw;
    for (int g = wk * gpw; g < (wk + 1) * gpw; ++g) {
        ops.zero(sg);
        for (int h = 0; h < 2; ++h) {
            sycl::group_barrier(it.get_group());   // the last half's tiles are read, the table loaded
            // the unit's codes: a lane half a row's unit (64 steps), the words the device order gives, the int8 pairs
            // from the table; the two halves of a row write the row's two halves (no overlap)
            const int s = 2 * g + h, n = lid / 2, j0 = (lid % 2) * 64, r = r0 + n;
            uint32_t u[4 * K];
#pragma unroll
            for (int c = 0; c < K; ++c) {
                const sycl::vec<uint32_t, 4> v =
                    *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(f.words + (((size_t) s * K + c) * f.rows + r) * 4);
                u[4 * c] = v[0];
                u[4 * c + 1] = v[1];
                u[4 * c + 2] = v[2];
                u[4 * c + 3] = v[3];
            }
#pragma unroll
            for (int jj = 0; jj < 32; ++jj) {
                const int j = j0 + jj;
                const uint32_t wv = (uint32_t) lut[state_of<K>(u, 2 * j)] |
                                    ((uint32_t) lut[state_of<K>(u, 2 * j + 1)] << 16);
                if constexpr (Ops::VNNI) bw[(size_t) (j0 / 2 + jj) * WN + n] = wv;   // [K / 4][output]
                else *reinterpret_cast<uint32_t*>(bl.get() + (size_t) n * LDB + 2 * j0 + 4 * jj) = wv;   // [output][K]
            }
            // the activations: the window's rows, zero past the last (the matrix tiles' rows past T multiply zeros)
            for (int v = lid; v < kYsknMmaRows * (KH / 16); v += LANES) {
                const int rr = v / (KH / 16), c = v % (KH / 16) * 16;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (rr < f.T)
                    val = *reinterpret_cast<const u32x4*>(f.xq + (size_t) rr * f.cols + (size_t) g * kGroup + h * KH + c);
                *reinterpret_cast<u32x4*>(al.get() + (size_t) rr * LDA + c) = val;
            }
            sycl::group_barrier(it.get_group());
            ops.half(sg, al, bl);
        }
        ops.store(sg, cp);
        sycl::group_barrier(it.get_group());
        // the group's int32 dots scaled and added in int64 (yskn_sum.hpp, the DP4a form's own): a lane its (row,
        // output) elements by index, the accumulator layouts out of it
        for (int i = 0; i < ACC_N; ++i) {
            const int idx = lid + i * LANES;
            if (idx >= f.T * WN) break;
            const int t = idx / WN, nn = idx % WN;
            acc[i] += ys::group_term(cp[(size_t) t * WN + nn], ys::fixed_scale(f.alpha[t * G + g], fac[t]));
        }
    }
    for (int i = 0; i < ACC_N; ++i) {
        const int idx = lid + i * LANES;
        if (idx >= f.T * WN) break;
        const int t = idx / WN, nn = idx % WN;
        f.part[((size_t) wk * f.T + t) * f.rows + r0 + nn] = acc[i];
    }
}

// ---------------------------------------------------------------- the forms' tiles
#if YSKN_MMA_INTEL
// Intel's XMX: 8 x IN x 32 on IN lanes (IN 16 Xe2, 8 Xe-HPG), B VNNI-packed ([K / 4][output], a word 4 K values of one
// output), the sub-group a 8 x IN tile of the 16 x WN (a row tile rt, a column tile ct).  The accumulator stores row by
// row, so its layout is not read (iq_mmq.cpp's intel_body reads it by lane; the tail here does not need to)
template<int IN>
struct IsOps {
    static constexpr bool VNNI = true;
    static constexpr int IM = 8, IK = 32;
    mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, IM, IN> ci;
    int rt, ct;
    explicit IsOps(int sgid) : rt(sgid / (WN / IN)), ct(sgid % (WN / IN)) {}
    void zero(sycl::sub_group sg) { mx::joint_matrix_fill(sg, ci, 0); }
    template<class P> void half(sycl::sub_group sg, P al, P bl) {
#pragma unroll
        for (int kk = 0; kk < KH; kk += IK) {
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, IM, IK, mx::layout::row_major> a;
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, IK, IN, mx::layout::ext_intel_packed> b;
            mx::joint_matrix_load(sg, a, al + (size_t) rt * IM * LDA + kk, LDA);
            mx::joint_matrix_load(sg, b, bl + (size_t) (kk / 4) * WN * 4 + ct * IN * 4, WN * 4);
            mx::joint_matrix_mad(sg, ci, a, b, ci);
        }
    }
    void store(sycl::sub_group sg, sycl::multi_ptr<int32_t, sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no> cp) {
        mx::joint_matrix_store(sg, ci, cp + (size_t) rt * IM * WN + ct * IN, WN, mx::layout::row_major);
    }
};
#endif

#if YSKN_MMA_JOINT
// joint_matrix's 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores from sm_72; a device reporting the shape): B
// column-major ([output][K]), the sub-group a 16 x 16 tile (a column tile ct, all 16 rows)
struct JmOps {
    static constexpr bool VNNI = false;
    mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, 16, 16> ci;
    int ct;
    explicit JmOps(int sgid) : ct(sgid) {}
    void zero(sycl::sub_group sg) { mx::joint_matrix_fill(sg, ci, 0); }
    template<class P> void half(sycl::sub_group sg, P al, P bl) {
#pragma unroll
        for (int kk = 0; kk < KH; kk += 16) {
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, 16, 16, mx::layout::row_major> a;
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, 16, 16, mx::layout::col_major> b;
            mx::joint_matrix_load(sg, a, al + kk, LDA);
            mx::joint_matrix_load(sg, b, bl + (size_t) ct * 16 * LDB + kk, LDB);
            mx::joint_matrix_mad(sg, ci, a, b, ci);
        }
    }
    void store(sycl::sub_group sg, sycl::multi_ptr<int32_t, sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no> cp) {
        mx::joint_matrix_store(sg, ci, cp + (size_t) ct * 16, WN, mx::layout::row_major);
    }
};
#endif

#if YSKN_MMA_WMMA
// RDNA3's and RDNA4's int8 WMMA, 16 x 16 x 16 on 32 lanes (amd_wmma.hpp): lane l row l % 16 of A, column l % 16 of B
// and of the accumulator, its rows row_of(l / 16, e); the same tiles as JmOps in the instructions' own layouts
struct WmOps {
    static constexpr bool VNNI = false;
    xe::wmma::iv8 ci;
    int ct, lo, hi;
    WmOps(int sgid, int lane) : ct(sgid), lo(lane & 15), hi(lane >> 4) {}
    void zero(sycl::sub_group) { ci = xe::wmma::iv8{0, 0, 0, 0, 0, 0, 0, 0}; }
    template<class P> void half(sycl::sub_group, P al, P bl) {
        using frag = xe::wmma::s8f;
        constexpr int KOFF = xe::wmma::KOFF;
#pragma unroll
        for (int kk = 0; kk < KH; kk += 16) {
            const frag a = *reinterpret_cast<const frag*>(al.get() + (size_t) lo * LDA + kk + hi * KOFF);
            const frag b = *reinterpret_cast<const frag*>(bl.get() + (size_t) (ct * 16 + lo) * LDB + kk + hi * KOFF);
            ci = xe::wmma::mad_s8(a, b, ci);
        }
    }
    void store(sycl::sub_group, sycl::multi_ptr<int32_t, sycl::access::address_space::local_space,
                                              sycl::access::decorated::no> cp) {
        for (int e = 0; e < 8; ++e) cp[(size_t) xe::wmma::row_of(hi, e) * WN + ct * 16 + lo] = ci[e];
    }
};
#endif

// ---------------------------------------------------------------- the kernels (a form and the codes' K)
template<int IN, int K>
struct KernelIs {
    Fields f;
    void operator()(sycl::nd_item<1> it) const {
#if YSKN_MMA_INTEL
        IsOps<IN> ops((int) it.get_sub_group().get_group_linear_id());
        gemv_body<IsOps<IN>, K>(it, f, ops);
#else
        (void) it;
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(IN)>}; }
};
template<int K>
struct KernelJm {
    Fields f;
    [[sycl::reqd_work_group_size(LANES, 1, 1), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    void operator()(sycl::nd_item<1> it) const {
#if YSKN_MMA_JOINT
        JmOps ops((int) it.get_sub_group().get_group_linear_id());
        gemv_body<JmOps, K>(it, f, ops);
#else
        (void) it;
#endif
    }
};
template<int K>
struct KernelWm {
    Fields f;
    [[sycl::reqd_work_group_size(LANES, 1, 1), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    void operator()(sycl::nd_item<1> it) const {
#if YSKN_MMA_WMMA
        WmOps ops((int) it.get_sub_group().get_group_linear_id(), (int) it.get_sub_group().get_local_linear_id());
        gemv_body<WmOps, K>(it, f, ops);
#else
        (void) it;
#endif
    }
};

template<int K> using KernelIs16 = KernelIs<16, K>;
template<int K> using KernelIs8 = KernelIs<8, K>;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

template<class KT>
sycl::event launch(sycl::queue& q, const Fields& f) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) LOCAL), h);
        Fields ff = f;
        ff.lm = lm;
        const size_t groups = (size_t) (f.rows / WN) * f.nw;
        h.parallel_for(sycl::nd_range<1>(groups * LANES, LANES), KT{ff});
    });
}

#define YSKN_MMA_K(KT)                                                                  \
    case 4: return launch<KT<4>>(q, f);                                                 \
    case 5: return launch<KT<5>>(q, f);                                                 \
    case 6: return launch<KT<6>>(q, f);                                                 \
    case 7: return launch<KT<7>>(q, f);                                                 \
    default: return launch<KT<8>>(q, f)

// the form's kernel builds for this device (its registers, its sub-group size), asked once
template<class KT>
bool builds(sycl::queue& q) {
    try {
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()},
                                                                       {sycl::get_kernel_id<KT>()});
        return true;
    } catch (const sycl::exception&) {
        return false;
    }
}

// the device's int8 matrix combination (sint8 . sint8 into sint32) of m x n x k
bool reported(const std::vector<mx::combination>& combos, size_t m, size_t n, size_t k) {
    for (const auto& c : combos) {
        const bool mm = c.msize == m || (c.msize == 0 && c.max_msize >= m);
        const bool nn = c.nsize == n || (c.nsize == 0 && c.max_nsize >= n);
        const bool kk = c.ksize == k || (c.ksize == 0 && c.max_ksize >= k);
        if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
            c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && mm && nn && kk)
            return true;
    }
    return false;
}

// the choice kept for the device: the form (yskn_mma_shape) and the row boundary the load times
struct Choice {
    int shape = 0;
    int boundary = kYsknMmaRows + 1;   // the smallest window rows taking the matrix form (past the most: never)
    bool timed = false;
};
Choice& choice_of(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<Choice>> per_device;
    return *per_device.get(q.get_device(), [&q] {
        auto c = std::make_shared<Choice>();
        c->shape = xe::yskn_mma_shape(q);
        return c;
    });
}
int mma_env() {
    static const int v = [] {
        const char* e = strata::env_get("YSKN_MMA");
        return e == nullptr || e[0] == '\0' ? -1 : (int) (std::strtol(e, nullptr, 10) != 0);
    }();
    return v;
}

}  // namespace

namespace xe {

int yskn_mma_shape(sycl::queue& q) {
    static core::PerDevice<int> per_device;
    return per_device.get(q.get_device(), [&q] {
        if (no_xmx()) return 0;
        const sycl::device d = q.get_device();
        if (d.get_info<sycl::info::device::local_mem_size>() < (size_t) LOCAL ||
            d.get_info<sycl::info::device::max_work_group_size>() < (size_t) LANES)
            return 0;
        if (xe::amd_wmma(d)) return builds<KernelWm<6>>(q) ? 3 : 0;   // RDNA3's and RDNA4's int8 WMMA
        const auto combos = matrix_combinations(d);
        // Intel's XMX (the fast shapes there), then the 16 x 16 x 16 a device reports (NVIDIA's tensor cores)
        if (xe::has_sub_group(d, 16) && reported(combos, 8, 16, 32) && builds<KernelIs16<6>>(q)) return 1;
        if (xe::has_sub_group(d, 8) && reported(combos, 8, 8, 32) && builds<KernelIs8<6>>(q)) return 2;
        if (xe::has_sub_group(d, 32) && reported(combos, 16, 16, 16) && builds<KernelJm<6>>(q)) return 3;
        return 0;
    });
}

sycl::event yskn_mma_gemv(sycl::queue& q, const YsknDev& w, const int8_t* xq, const float* alpha, int T, int64_t* part,
                          int shape) {
    if (!w.words || !w.lut8 || !xq || !alpha || !part || T <= 0 || T > kYsknMmaRows || w.rows <= 0 ||
        w.rows % WN != 0 || w.cols <= 0 || w.cols % kGroup != 0 || w.n_workers <= 0 ||
        (w.cols / kGroup) % w.n_workers != 0 || w.bits < 4 || w.bits > 8)
        throw core::DeviceError("yskn_mma_gemv: invalid .yskn matrix");
    Fields f{w.words, w.lut8, xq, alpha, part, w.rows, w.cols, T, w.n_workers, {}};
    switch (shape) {
        case 1: switch (w.bits) { YSKN_MMA_K(KernelIs16); }
        case 2: switch (w.bits) { YSKN_MMA_K(KernelIs8); }
        default:
            if (xe::amd_wmma(q.get_device())) {
                switch (w.bits) { YSKN_MMA_K(KernelWm); }
            } else {
                switch (w.bits) { YSKN_MMA_K(KernelJm); }
            }
    }
}

}  // namespace xe

bool yskn_mma_usable(void* stream) { return choice_of(queue_for(stream)).shape != 0; }

bool yskn_mma_take(int T, void* stream) {
    Choice& c = choice_of(queue_for(stream));
    if (c.shape == 0) return false;
    const int e = mma_env();
    if (e >= 0) return e == 1;
    return T >= c.boundary;
}

void yskn_mma_proj(const YsknDev& w, const float* x, int64_t ldx, int T, float* y, int64_t ldy, void* scratch,
                   void* stream) {
    if (T <= 0 || T > kYsknMmaRows) throw core::DeviceError("yskn_mma_proj: invalid row count");
    const YsknScratch s = yskn_scratch_parts(w, scratch);
    auto& q = queue_for(stream);
    yskn_act(w, x, ldx, T, s.xq, s.alpha, stream);
    const auto e = xe::yskn_mma_gemv(q, w, s.xq, s.alpha, T, s.part, choice_of(q).shape);
    if (!stream) core::Runtime::get().wait(e, "yskn_mma_gemv");
    yskn_finish(w, s.part, s.alpha, T, y, ldy, stream);
}

void yskn_mma_tune(const YsknDev& w, void* scratch, void* stream) {
    auto& q = queue_for(stream);
    Choice& c = choice_of(q);
    if (c.shape == 0 || c.timed) return;
    c.timed = true;
    if (mma_env() >= 0) return;   // forced: no boundary to time
    const YsknScratch s = yskn_scratch_parts(w, scratch);
    const size_t act_bytes = (size_t) kYsknMmaRows * w.cols + (size_t) kYsknMmaRows * (w.cols / kGroup) * 4;
    q.memset(s.xq, 0, act_bytes).wait();   // zero activations: zero sums, real times
    using Clock = std::chrono::steady_clock;
    const auto time = [&](const std::function<void()>& f) {
        f();
        q.wait_and_throw();   // the kernel's build
        const auto t0 = Clock::now();
        for (int i = 0; i < 3; ++i) f();
        q.wait_and_throw();
        return std::chrono::duration<double>(Clock::now() - t0).count() / 3;
    };
    try {
        for (int T : {1, 2, 4, 8, 16}) {
            // the DP4a form: a launch of up to 8 rows (16: two, the sums overwritten - timed, not read)
            const double dp = time([&] {
                if (T <= kYsknMaxRows) yskn_gemv(w, s.xq, s.alpha, T, s.part, stream);
                else {
                    yskn_gemv(w, s.xq, s.alpha, kYsknMaxRows, s.part, stream);
                    yskn_gemv(w, s.xq, s.alpha, T - kYsknMaxRows, s.part, stream);
                }
            });
            const double mm = time([&] { (void) xe::yskn_mma_gemv(q, w, s.xq, s.alpha, T, s.part, c.shape); });
            if (mm > 0 && dp > 0 && mm * 1.05 < dp) {
                c.boundary = T;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the .yskn decode's forms: %s\n", e.what());
    }
    if (c.boundary <= kYsknMmaRows)
        std::fprintf(stderr, "xestrata plain: the .yskn decode takes the matrix form at %d rows and more (timed; "
                             "XESTRATA_YSKN_MMA=0 keeps DP4a)\n", c.boundary);
    else
        std::fprintf(stderr, "xestrata plain: the .yskn decode stays in DP4a (the matrix form timed no faster; "
                             "XESTRATA_YSKN_MMA=1 forces it)\n");
}

}  // namespace strata::kernels