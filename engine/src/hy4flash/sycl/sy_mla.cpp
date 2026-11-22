// sy_mla.cpp - SYCL kernels of the Hy4-Flash MLA attention (see sy_mla.h).
//   * Prefill (M >= 16 on XMX): XMX joint_matrix prep_q, Flash-MLA attend_xmx (in-SLM V transpose, 0 Vt copy), vup_xmx
//   * Decode/Verify (M <= 16): Quantized S_K kv_b kernels (3x less bandwidth, 0 persistent fp16 wk_b/wv_b) + batch-invariant split-K attend
//   * Device-resident StepParams (*_dev) variants for SYCL command_graph capture/replay
#include "sy_mla.h"

#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace hy4s {

int g_mla_target_blocks = 56;
int g_mla_batch_invariant = 0;

namespace xm = sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr int HG = 16;
constexpr float NEG = -1e30f;
constexpr int PART_STRIDE = HG * MLA_KV_LORA + 32;

using vh8 = sycl::vec<half, 8>;
using vh4 = sycl::vec<half, 4>;
using vh2 = sycl::vec<half, 2>;
using vf4 = sycl::vec<float, 4>;

inline bool mla_use_xmx() {
    static const bool on = [] {
        const char* e = std::getenv("HY4S_MLA_XMX");
        if (e && e[0] == '0') return false;
#ifndef HY4S_DISABLE_SG8
        if (dev_caps().has_xmx_sg8) return true;
#endif
#ifdef HY4S_ENABLE_SG16_XMX
        if (dev_caps().has_xmx_sg16) return true;
#endif
        return false;
    }();
    return on;
}

inline void rope_pair(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    const float freq = 1.0f / sycl::pow(theta, (float)(2 * i) / 64.0f);
    const float ang = (float)pos * freq;
    float cs;
    const float sn = sycl::sincos(ang, &cs);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

inline void ldh8(const half* p, float (&f)[8]) {
    const vh8 v = *reinterpret_cast<const vh8*>(p);
    const sycl::vec<float, 8> w = v.convert<float>();
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = w[i];
}

int max_keys_of(const MlaAttnArgs& a) {
    if (a.mode == MLA_CAUSAL) return a.p0 + a.M;
    if (a.mode == MLA_WINDOW) return std::min(a.window, a.p0 + a.M);
    return a.kmax;
}

inline int key_slot(const MlaAttnArgs& a, const TreePos& tp, int t, int lo, int j) {
    int lp;
    if (a.mode == MLA_CAUSAL) lp = j;
    else if (a.mode == MLA_WINDOW) lp = lo + j;
    else lp = a.keys[(size_t)t * a.kmax + j];
    if (tp.n > 0 && lp >= a.p0) {
        lp = a.p0 + (int)tp.anc[t][(lp - a.p0) & 7];
    }
    return a.ring > 0 ? (lp % a.ring) : lp;
}

// ---- KV cache write (supports host p0, device StepParams* d_step, and Tree-8) ----
void write_kv_impl(const float* kv_a, const float* norm_w, half* cache, int M, int p0_host, const StepParams* d_step, bool use_dfly_a, int ring, float theta, float eps, TreePos tp_host = {}) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(32, h);
        h.parallel_for(sycl::nd_range<1>((size_t)M * 256, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = d_step ? (use_dfly_a ? d_step->a : d_step->p0) : p0_host;
            const bool tree_on = !use_dfly_a && (d_step ? (d_step->tree_on != 0) : (tp_host.n > 0));
            const float* row = kv_a + (size_t)t * MLA_ROW;
            const float v0 = row[2 * tid], v1 = row[2 * tid + 1];
            const float ss = wg_sum(it, v0 * v0 + v1 * v1, sh.get_multi_ptr<sycl::access::decorated::no>());
            const float inv = 1.0f / sycl::sqrt(ss / (float)MLA_KV_LORA + eps);
            const int phys_pos = p0 + t;
            const int rope_pos = p0 + (tree_on ? (int)(d_step ? d_step->tree.depth[t] : tp_host.depth[t]) : t);
            const int slot = ring > 0 ? phys_pos % ring : phys_pos;
            half* dst = cache + (size_t)slot * MLA_ROW;
            vh2 o;
            o[0] = (half)(v0 * inv * norm_w[2 * tid]);
            o[1] = (half)(v1 * inv * norm_w[2 * tid + 1]);
            *reinterpret_cast<vh2*>(dst + 2 * tid) = o;
            if (tid < 32) {
                float o0, o1;
                rope_pair(row[MLA_KV_LORA + 2 * tid], row[MLA_KV_LORA + 2 * tid + 1], rope_pos, tid, theta, o0, o1);
                vh2 r;
                r[0] = (half)o0;
                r[1] = (half)o1;
                *reinterpret_cast<vh2*>(dst + MLA_KV_LORA + 2 * tid) = r;
            }
        });
    });
}

// ---- q absorb + rope (FP16 wk_b, decode/small-M) ----
template <int TT>
static void prep_q_t(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0_host, const StepParams* d_step, bool use_dfly_a, float theta) {
    const int ntt = (M + TT - 1) / TT;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(TT * MLA_NOPE, h);
        sycl::local_accessor<float, 1> red(3 * TT * 128, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, 4, (size_t)ntt * 256), sycl::range<3>(1, 1, 256)), [=](sycl::nd_item<3> it) REQD_SG32 {
            const int hh = (int)it.get_group(0), cq = (int)it.get_group(1), t0 = (int)it.get_group(2) * TT;
            const int tid = (int)it.get_local_id(2);
            const int p0 = d_step ? (use_dfly_a ? d_step->a : d_step->p0) : p0_host;
            for (int idx = tid; idx < TT * MLA_NOPE; idx += 256) {
                const int tt = idx / MLA_NOPE, i = idx % MLA_NOPE, t = t0 + tt;
                sq[idx] = t < M ? q_b[((size_t)t * NH + hh) * MLA_QHEAD + i] : 0.f;
            }
            sycl::group_barrier(it.get_group());
            const int isub = tid >> 6, cp = tid & 63;
            const int col = cq * 128 + 2 * cp;
            float a0[TT], a1[TT];
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) a0[tt] = a1[tt] = 0.f;
            const half* wrow = wk_b + ((size_t)hh * MLA_NOPE + isub * 48) * MLA_KV_LORA + col;
#pragma unroll 4
            for (int i = 0; i < 48; ++i) {
                const vh2 w = *reinterpret_cast<const vh2*>(wrow + (size_t)i * MLA_KV_LORA);
                const float wx = (float)w[0], wy = (float)w[1];
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    const float qv = sq[tt * MLA_NOPE + isub * 48 + i];
                    a0[tt] += qv * wx;
                    a1[tt] += qv * wy;
                }
            }
            if (isub > 0) {
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    red[((isub - 1) * TT + tt) * 128 + 2 * cp] = a0[tt];
                    red[((isub - 1) * TT + tt) * 128 + 2 * cp + 1] = a1[tt];
                }
            }
            sycl::group_barrier(it.get_group());
            if (isub == 0) {
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    const int t = t0 + tt;
                    if (t < M) {
                        float s0 = a0[tt], s1 = a1[tt];
#pragma unroll
                        for (int k = 0; k < 3; ++k) {
                            s0 += red[(k * TT + tt) * 128 + 2 * cp];
                            s1 += red[(k * TT + tt) * 128 + 2 * cp + 1];
                        }
                        vh2 o;
                        o[0] = (half)s0;
                        o[1] = (half)s1;
                        *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + col) = o;
                    }
                }
            }
            if (cq == 0) {
                for (int idx = tid; idx < TT * 32; idx += 256) {
                    const int tt = idx / 32, i = idx % 32, t = t0 + tt;
                    if (t >= M) continue;
                    const float* qp = q_b + ((size_t)t * NH + hh) * MLA_QHEAD + MLA_NOPE + 2 * i;
                    float o0, o1;
                    rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
                    vh2 o;
                    o[0] = (half)o0;
                    o[1] = (half)o1;
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + MLA_KV_LORA + 2 * i) = o;
                }
            }
        });
    });
}

// ---- q absorb + rope XMX (prefill M >= 16): GEMM [M, 192] x [192, 512] per head using 8x8x16 XMX ----
#ifndef HY4S_DISABLE_SG8
static void prep_q_xmx(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta) {
    const int nbm = (M + 63) / 64;
    const int nbn = MLA_KV_LORA / 64;  // 8
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(64 * 64, h);  // [64 tokens][64 k] row-major
        sycl::local_accessor<half, 1> sB(64 * 64, h);  // VNNI-2 packed [32][128] for 64 k x 64 n
        sycl::local_accessor<float, 1> sC(64 * 64, h); // epilogue staging for tail M
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, nbn, (size_t)nbm * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(8)]] {
            const int hh = (int)it.get_group(0), bn = (int)it.get_group(1), bm = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, sg_m = sgi >> 1, sg_n = sgi & 1;

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sa_ptr = &sA[0];
            half* sb_ptr = &sB[0];
            auto sA_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sa_ptr);
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            // K = 192 = 3 blocks of 64
            for (int k0 = 0; k0 < MLA_NOPE; k0 += 64) {
                // Load sA[64][64] from q_b[(bm*64 + r), hh, k0..k0+63]
                for (int idx = tid; idx < 64 * 64; idx += 128) {
                    const int r = idx >> 6, k = idx & 63;
                    const int t = bm * 64 + r;
                    sa_ptr[idx] = t < M ? (half)q_b[((size_t)t * NH + hh) * MLA_QHEAD + k0 + k] : (half)0.0f;
                }
                // Load wk_b[hh][k0..k0+63][bn*64..bn*64+63] -> VNNI-2 sB[(k>>1)*128 + n*2 + (k&1)]
                for (int idx = tid; idx < 32 * 64; idx += 128) {
                    const int kp = idx >> 6, n = idx & 63;  // k pair (2*kp, 2*kp+1)
                    const half w0 = wk_b[((size_t)hh * MLA_NOPE + k0 + 2 * kp) * MLA_KV_LORA + bn * 64 + n];
                    const half w1 = wk_b[((size_t)hh * MLA_NOPE + k0 + 2 * kp + 1) * MLA_KV_LORA + bn * 64 + n];
                    *reinterpret_cast<vh2*>(sb_ptr + kp * 128 + n * 2) = vh2(w0, w1);
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    xm::joint_matrix_load(sg, ma, sA_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)(sg_n * 4 + j) * 16, (size_t)128);
                        xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            float* sc_ptr = &sC[0];
            auto sC_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc_ptr);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xm::joint_matrix_store(sg, mc[j], sC_mp + (size_t)(sg_m * 8) * 64 + (sg_n * 4 + j) * 8, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            for (int idx = tid; idx < 64 * 32; idx += 128) {
                const int r = idx >> 5, cp = idx & 31;
                const int t = bm * 64 + r;
                if (t < M) {
                    vh2 o;
                    o[0] = (half)sc_ptr[r * 64 + 2 * cp];
                    o[1] = (half)sc_ptr[r * 64 + 2 * cp + 1];
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + bn * 64 + 2 * cp) = o;
                }
            }
            if (bn == 0) {
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int r = idx >> 5, i = idx & 31;
                    const int t = bm * 64 + r;
                    if (t >= M) continue;
                    const float* qp = q_b + ((size_t)t * NH + hh) * MLA_QHEAD + MLA_NOPE + 2 * i;
                    float o0, o1;
                    rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + MLA_KV_LORA + 2 * i) = vh2((half)o0, (half)o1);
                }
            }
        });
    });
}
#endif

// ---- q absorb + rope XMX SG16 (Xe2 Battlemage prefill M >= 16: 8x16x16 DPAS) ----
#ifdef HY4S_ENABLE_SG16_XMX
static void prep_q_xmx_sg16(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta) {
    const int nbm = (M + 63) / 64;
    const int nbn = MLA_KV_LORA / 64;  // 8
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(64 * 64, h);
        sycl::local_accessor<half, 1> sB(64 * 64, h);
        sycl::local_accessor<float, 1> sC(64 * 64, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, nbn, (size_t)nbm * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int hh = (int)it.get_group(0), bn = (int)it.get_group(1), bm = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4;  // 0..7 (each SG16 owns 8 M-rows x 64 N-cols = 4 tiles of 8x16)

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sa_ptr = &sA[0];
            half* sb_ptr = &sB[0];
            auto sA_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sa_ptr);
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            for (int k0 = 0; k0 < MLA_NOPE; k0 += 64) {
                for (int idx = tid; idx < 64 * 64; idx += 128) {
                    const int r = idx >> 6, k = idx & 63;
                    const int t = bm * 64 + r;
                    sa_ptr[idx] = t < M ? (half)q_b[((size_t)t * NH + hh) * MLA_QHEAD + k0 + k] : (half)0.0f;
                }
                for (int idx = tid; idx < 32 * 64; idx += 128) {
                    const int kp = idx >> 6, n = idx & 63;
                    const half w0 = wk_b[((size_t)hh * MLA_NOPE + k0 + 2 * kp) * MLA_KV_LORA + bn * 64 + n];
                    const half w1 = wk_b[((size_t)hh * MLA_NOPE + k0 + 2 * kp + 1) * MLA_KV_LORA + bn * 64 + n];
                    *reinterpret_cast<vh2*>(sb_ptr + kp * 128 + n * 2) = vh2(w0, w1);
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    xm::joint_matrix_load(sg, ma, sA_mp + (size_t)(sgi * 8) * 64 + s * 16, (size_t)64);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)j * 32, (size_t)128);
                        xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            float* sc_ptr = &sC[0];
            auto sC_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc_ptr);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xm::joint_matrix_store(sg, mc[j], sC_mp + (size_t)(sgi * 8) * 64 + j * 16, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            for (int idx = tid; idx < 64 * 32; idx += 128) {
                const int r = idx >> 5, cp = idx & 31;
                const int t = bm * 64 + r;
                if (t < M) {
                    vh2 o;
                    o[0] = (half)sc_ptr[r * 64 + 2 * cp];
                    o[1] = (half)sc_ptr[r * 64 + 2 * cp + 1];
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + bn * 64 + 2 * cp) = o;
                }
            }
            if (bn == 0) {
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int r = idx >> 5, i = idx & 31;
                    const int t = bm * 64 + r;
                    if (t >= M) continue;
                    const float* qp = q_b + ((size_t)t * NH + hh) * MLA_QHEAD + MLA_NOPE + 2 * i;
                    float o0, o1;
                    rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + MLA_KV_LORA + 2 * i) = vh2((half)o0, (half)o1);
                }
            }
        });
    });
}
#endif

// ---- Quantized S_K q absorb + rope (decode/verify M <= 16; bit-identical to split_kvb -> fp16 -> prep_q; supports Tree-8!) ----
// 8 WGs per head (2 superblocks x 4 64-weight subgroups) x 64 threads/WG -> 4x higher occupancy, ZERO inner-loop barriers!
template <int FMT, int TT>
static void prep_q_sk_t(const uint8_t* W_kvb, const float* q_b, half* qa, int M, int NH, int p0_host, const StepParams* d_step, bool use_dfly_a, float theta, TreePos tp_host = {}) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int ntt = (M + TT - 1) / TT;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(TT * MLA_NOPE, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, 8, (size_t)ntt * 64), sycl::range<3>(1, 1, 64)), [=](sycl::nd_item<3> it) REQD_SG32 {
            const int hh = (int)it.get_group(0), grp8 = (int)it.get_group(1), t0 = (int)it.get_group(2) * TT;
            const int sbi = grp8 >> 2, g = grp8 & 3;
            const int tid = (int)it.get_local_id(2);  // 0..63 within 64-weight subgroup g
            const int c_in_sb = (g << 6) + tid;      // 0..255
            const int p0 = d_step ? (use_dfly_a ? d_step->a : d_step->p0) : p0_host;
            const bool tree_on = !use_dfly_a && (d_step ? (d_step->tree_on != 0) : (tp_host.n > 0));
            for (int idx = tid; idx < TT * MLA_NOPE; idx += 64) {
                const int tt = idx / MLA_NOPE, i = idx % MLA_NOPE, t = t0 + tt;
                sq[idx] = t < M ? q_b[((size_t)t * NH + hh) * MLA_QHEAD + i] : 0.f;
            }
            sycl::group_barrier(it.get_group());
            float acc[TT];
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) acc[tt] = 0.f;
            const uint8_t* wbase = W_kvb + ((size_t)hh * 448 * 2 + (size_t)sbi) * SB;
            for (int i = 0; i < MLA_NOPE; ++i) {
                const uint8_t* blk = wbase + (size_t)i * 2 * SB;
                const float D = h2f_bits(ld16(blk));
                const float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
                const float w = (float)(half)(((float)sk_q<FMT>(blk + 6, c_in_sb) - sk_half<FMT>()) * eff);
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) acc[tt] += sq[tt * MLA_NOPE + i] * w;
            }
            const int col = sbi * 256 + c_in_sb;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                const int t = t0 + tt;
                if (t < M) qa[((size_t)t * NH + hh) * MLA_ROW + col] = (half)acc[tt];
            }
            if (grp8 == 0) {
                for (int idx = tid; idx < TT * 32; idx += 64) {
                    const int tt = idx / 32, i = idx % 32, t = t0 + tt;
                    if (t >= M) continue;
                    const int rope_pos = p0 + (tree_on ? (int)(d_step ? d_step->tree.depth[t] : tp_host.depth[t]) : t);
                    const float* qp = q_b + ((size_t)t * NH + hh) * MLA_QHEAD + MLA_NOPE + 2 * i;
                    float o0, o1;
                    rope_pair(qp[0], qp[1], rope_pos, i, theta, o0, o1);
                    *reinterpret_cast<vh2*>(qa + ((size_t)t * NH + hh) * MLA_ROW + MLA_KV_LORA + 2 * i) = vh2((half)o0, (half)o1);
                }
            }
        });
    });
}

// ---- v up-projection (FP16 wv_b, decode/small-M) ----
template <int TT>
static void vup_t(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH) {
    const int ntt = (M + TT - 1) / TT;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cs(TT * MLA_KV_LORA, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, 4, (size_t)ntt * 256), sycl::range<3>(1, 1, 256)), [=](sycl::nd_item<3> it) REQD_SG32 {
            const int hh = (int)it.get_group(0), jq = (int)it.get_group(1), t0 = (int)it.get_group(2) * TT;
            const int tid = (int)it.get_local_id(2);
            for (int idx = tid; idx < TT * MLA_KV_LORA; idx += 256) {
                const int tt = idx / MLA_KV_LORA, c = idx % MLA_KV_LORA, t = t0 + tt;
                cs[idx] = t < M ? (float)ctx[((size_t)t * NH + hh) * MLA_KV_LORA + c] : 0.f;
            }
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const int w = tid >> 5, lane = tid & 31;
            const float* csp = cs.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            for (int k = 0; k < 8; ++k) {
                const int j = jq * 64 + w + 8 * k;
                const half* row = wv_b + ((size_t)hh * MLA_VHEAD + j) * MLA_KV_LORA;
                float wf[2][8];
                ldh8(row + lane * 8, wf[0]);
                ldh8(row + 256 + lane * 8, wf[1]);
                float acc[TT];
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    float s = 0.f;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const vf4 x0 = *reinterpret_cast<const vf4*>(csp + tt * MLA_KV_LORA + r * 256 + lane * 8);
                        const vf4 x1 = *reinterpret_cast<const vf4*>(csp + tt * MLA_KV_LORA + r * 256 + lane * 8 + 4);
                        s += x0[0] * wf[r][0] + x0[1] * wf[r][1] + x0[2] * wf[r][2] + x0[3] * wf[r][3] + x1[0] * wf[r][4] + x1[1] * wf[r][5] + x1[2] * wf[r][6] + x1[3] * wf[r][7];
                    }
                    acc[tt] = s;
                }
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    const float v = sycl::reduce_over_group(sg, acc[tt], sycl::plus<float>());
                    const int t = t0 + tt;
                    if (lane == 0 && t < M) {
                        float r = v;
                        if (gate) r *= 1.0f / (1.0f + sycl::exp(-gate[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j]));
                        o[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j] = (half)r;
                    }
                }
            }
        });
    });
}

// ---- v up-projection XMX (prefill M >= 16): GEMM [M, 512] x [256, 512]^T + sigmoid(gate) ----
#ifndef HY4S_DISABLE_SG8
static void vup_xmx(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH) {
    const int nbm = (M + 63) / 64;
    const int nbn = MLA_VHEAD / 64;  // 4
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(64 * 64, h);
        sycl::local_accessor<half, 1> sB(64 * 64, h);
        sycl::local_accessor<float, 1> sC(64 * 64, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, nbn, (size_t)nbm * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(8)]] {
            const int hh = (int)it.get_group(0), bn = (int)it.get_group(1), bm = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, sg_m = sgi >> 1, sg_n = sgi & 1;

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sa_ptr = &sA[0];
            half* sb_ptr = &sB[0];
            auto sA_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sa_ptr);
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            // K = 512 = 8 blocks of 64
            for (int k0 = 0; k0 < MLA_KV_LORA; k0 += 64) {
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int r = idx >> 5, kp = idx & 31;
                    const int t = bm * 64 + r;
                    vh2 v = t < M ? *reinterpret_cast<const vh2*>(ctx + ((size_t)t * NH + hh) * MLA_KV_LORA + k0 + 2 * kp) : vh2((half)0.f, (half)0.f);
                    *reinterpret_cast<vh2*>(sa_ptr + r * 64 + 2 * kp) = v;
                }
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int n = idx >> 5, kp = idx & 31;
                    const vh2 w = *reinterpret_cast<const vh2*>(wv_b + ((size_t)hh * MLA_VHEAD + bn * 64 + n) * MLA_KV_LORA + k0 + 2 * kp);
                    *reinterpret_cast<vh2*>(sb_ptr + kp * 128 + n * 2) = w;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    xm::joint_matrix_load(sg, ma, sA_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)(sg_n * 4 + j) * 16, (size_t)128);
                        xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            float* sc_ptr = &sC[0];
            auto sC_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc_ptr);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xm::joint_matrix_store(sg, mc[j], sC_mp + (size_t)(sg_m * 8) * 64 + (sg_n * 4 + j) * 8, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            for (int idx = tid; idx < 64 * 64; idx += 128) {
                const int r = idx >> 6, n = idx & 63;
                const int t = bm * 64 + r;
                if (t < M) {
                    const int j = bn * 64 + n;
                    float v = sc_ptr[idx];
                    if (gate) v *= 1.0f / (1.0f + sycl::exp(-gate[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j]));
                    o[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j] = (half)v;
                }
            }
        });
    });
}
#endif

// ---- v up-projection XMX SG16 (Xe2 Battlemage prefill M >= 16: 8x16x16 DPAS) ----
#ifdef HY4S_ENABLE_SG16_XMX
static void vup_xmx_sg16(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH) {
    const int nbm = (M + 63) / 64;
    const int nbn = MLA_VHEAD / 64;  // 4
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(64 * 64, h);
        sycl::local_accessor<half, 1> sB(64 * 64, h);
        sycl::local_accessor<float, 1> sC(64 * 64, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, nbn, (size_t)nbm * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int hh = (int)it.get_group(0), bn = (int)it.get_group(1), bm = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4;  // 0..7 (each SG16 owns 8 M-rows x 64 N-cols = 4 tiles of 8x16)

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sa_ptr = &sA[0];
            half* sb_ptr = &sB[0];
            auto sA_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sa_ptr);
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            for (int k0 = 0; k0 < MLA_KV_LORA; k0 += 64) {
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int r = idx >> 5, kp = idx & 31;
                    const int t = bm * 64 + r;
                    vh2 v = t < M ? *reinterpret_cast<const vh2*>(ctx + ((size_t)t * NH + hh) * MLA_KV_LORA + k0 + 2 * kp) : vh2((half)0.f, (half)0.f);
                    *reinterpret_cast<vh2*>(sa_ptr + r * 64 + 2 * kp) = v;
                }
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int n = idx >> 5, kp = idx & 31;
                    const vh2 w = *reinterpret_cast<const vh2*>(wv_b + ((size_t)hh * MLA_VHEAD + bn * 64 + n) * MLA_KV_LORA + k0 + 2 * kp);
                    *reinterpret_cast<vh2*>(sb_ptr + kp * 128 + n * 2) = w;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    xm::joint_matrix_load(sg, ma, sA_mp + (size_t)(sgi * 8) * 64 + s * 16, (size_t)64);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)j * 32, (size_t)128);
                        xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            float* sc_ptr = &sC[0];
            auto sC_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc_ptr);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xm::joint_matrix_store(sg, mc[j], sC_mp + (size_t)(sgi * 8) * 64 + j * 16, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            for (int idx = tid; idx < 64 * 64; idx += 128) {
                const int r = idx >> 6, n = idx & 63;
                const int t = bm * 64 + r;
                if (t < M) {
                    const int j = bn * 64 + n;
                    float v = sc_ptr[idx];
                    if (gate) v *= 1.0f / (1.0f + sycl::exp(-gate[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j]));
                    o[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j] = (half)v;
                }
            }
        });
    });
}
#endif

// ---- Quantized S_K v up-projection (decode/verify M <= 16; bit-identical to split_kvb -> fp16 -> vup) ----
template <int FMT, int TT>
static void vup_sk_t(const uint8_t* W_kvb, const half* ctx, const float* gate, half* o, int M, int NH) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int ntt = (M + TT - 1) / TT;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cs(TT * MLA_KV_LORA, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, 4, (size_t)ntt * 256), sycl::range<3>(1, 1, 256)), [=](sycl::nd_item<3> it) REQD_SG32 {
            const int hh = (int)it.get_group(0), jq = (int)it.get_group(1), t0 = (int)it.get_group(2) * TT;
            const int tid = (int)it.get_local_id(2);
            for (int idx = tid; idx < TT * MLA_KV_LORA; idx += 256) {
                const int tt = idx / MLA_KV_LORA, c = idx % MLA_KV_LORA, t = t0 + tt;
                cs[idx] = t < M ? (float)ctx[((size_t)t * NH + hh) * MLA_KV_LORA + c] : 0.f;
            }
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const int w = tid >> 5, lane = tid & 31;
            const float* csp = cs.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            const int g = lane >> 3, j0 = (lane & 7) << 3;  // 8 consecutive weights inside a 64-weight subgroup
            for (int k = 0; k < 8; ++k) {
                const int j = jq * 64 + w + 8 * k;
                const uint8_t* rp = W_kvb + ((size_t)hh * 448 + 192 + j) * 2 * SB;
                float wf[2][8];
#pragma unroll
                for (int sbi = 0; sbi < 2; ++sbi) {
                    const uint8_t* blk = rp + (size_t)sbi * SB;
                    const float D = h2f_bits(ld16(blk));
                    const float eff = D * ((float)blk[2 + g] / 255.0f);
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                        wf[sbi][q] = (float)(half)(((float)sk_q<FMT>(blk + 6, (g << 6) + j0 + q) - sk_half<FMT>()) * eff);
                }
                float acc[TT];
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    float s = 0.f;
#pragma unroll
                    for (int sbi = 0; sbi < 2; ++sbi) {
                        const vf4 x0 = *reinterpret_cast<const vf4*>(csp + tt * MLA_KV_LORA + sbi * 256 + (g << 6) + j0);
                        const vf4 x1 = *reinterpret_cast<const vf4*>(csp + tt * MLA_KV_LORA + sbi * 256 + (g << 6) + j0 + 4);
                        s += x0[0] * wf[sbi][0] + x0[1] * wf[sbi][1] + x0[2] * wf[sbi][2] + x0[3] * wf[sbi][3] +
                             x1[0] * wf[sbi][4] + x1[1] * wf[sbi][5] + x1[2] * wf[sbi][6] + x1[3] * wf[sbi][7];
                    }
                    acc[tt] = s;
                }
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    const float v = sycl::reduce_over_group(sg, acc[tt], sycl::plus<float>());
                    const int t = t0 + tt;
                    if (lane == 0 && t < M) {
                        float r = v;
                        if (gate) r *= 1.0f / (1.0f + sycl::exp(-gate[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j]));
                        o[(size_t)t * NH * MLA_VHEAD + (size_t)hh * MLA_VHEAD + j] = (half)r;
                    }
                }
            }
        });
    });
}

// ---- Scalar FMA Flash-MLA (exact batch-invariant decode/verify + non-XMX fallback) ----
template <int TKT>
static void attend_t(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a_in,
                     const StepParams* d_step, bool use_dfly_a, float* part, int S, int tps) {
    constexpr int HPT = TKT / 32;
    constexpr int KPL = TKT / 16;
    const int NHG = a_in.NH / HG;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(HG * MLA_ROW, h);
        sycl::local_accessor<float, 1> sS(HG * TKT, h);
        sycl::local_accessor<float, 1> sPt(TKT * HG, h);
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(HG, h), sl(HG, h), sc(HG, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(a_in.M, S, (size_t)NHG * 512), sycl::range<3>(1, 1, 512)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            MlaAttnArgs a = a_in;
            if (d_step) a.p0 = use_dfly_a ? d_step->a : d_step->p0;
            const TreePos tp = (!use_dfly_a && d_step && d_step->tree_on) ? d_step->tree : a_in.tree;
            const int t = (int)it.get_group(0), sp = (int)it.get_group(1), hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            float* sqp = sq.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            float* sSp = sS.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            float* sPp = sPt.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            {
                const half* qbase = qa + ((size_t)t * a.NH + hg * HG) * MLA_ROW;
                for (int e = tid * 8; e < HG * MLA_ROW; e += 512 * 8) {
                    float f[8];
                    ldh8(qbase + e, f);
#pragma unroll
                    for (int i = 0; i < 8; ++i) sqp[e + i] = f[i];
                }
            }
            const int pos = a.p0 + (tp.n > 0 ? (int)tp.depth[t] : t);
            int nk, lo = 0;
            if (a.mode == MLA_CAUSAL) nk = pos + 1;
            else if (a.mode == MLA_WINDOW) {
                lo = std::max(0, pos - a.window + 1);
                nk = pos - lo + 1;
            } else nk = a.nkeys[t];
            const int ntile_q = (nk + TKT - 1) / TKT;
            const int tile0 = sp * tps, tile1 = std::min(tile0 + tps, ntile_q);
            if (tid < HG) {
                sm[tid] = sp == 0 ? sinks[hg * HG + tid] : NEG;
                sl[tid] = sp == 0 ? 1.0f : 0.0f;
            }
            float acc[HG];
#pragma unroll
            for (int r = 0; r < HG; ++r) acc[r] = 0.f;
            sycl::group_barrier(grp);
            const int kk = tid & (TKT - 1), hgrp = tid / TKT;
            for (int tile = tile0; tile < tile1; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk - j0);
                const bool valid = kk < cnt;
                const int slot = valid ? key_slot(a, tp, t, lo, j0 + kk) : 0;
                if (hgrp == 0) sSlot[kk] = slot;
                const half* kr = cache + (size_t)slot * MLA_ROW;
                float s[HPT];
#pragma unroll
                for (int i = 0; i < HPT; ++i) s[i] = 0.f;
                const float* qrow = sqp + (size_t)(hgrp * HPT) * MLA_ROW;
#pragma unroll 2
                for (int d = 0; d < MLA_ROW; d += 8) {
                    float kf[8];
                    ldh8(kr + d, kf);
#pragma unroll
                    for (int i = 0; i < HPT; ++i) {
                        const vf4 q0 = *reinterpret_cast<const vf4*>(qrow + (size_t)i * MLA_ROW + d);
                        const vf4 q1 = *reinterpret_cast<const vf4*>(qrow + (size_t)i * MLA_ROW + d + 4);
                        s[i] += q0[0] * kf[0] + q0[1] * kf[1] + q0[2] * kf[2] + q0[3] * kf[3] + q1[0] * kf[4] + q1[1] * kf[5] + q1[2] * kf[6] + q1[3] * kf[7];
                    }
                }
#pragma unroll
                for (int i = 0; i < HPT; ++i) sSp[(hgrp * HPT + i) * TKT + kk] = valid ? s[i] * a.scale : NEG;
                sycl::group_barrier(grp);
                if (tid < 256) {
                    const int r = tid >> 4, pl = tid & 15;
                    float sv[KPL];
                    float tmax = NEG;
#pragma unroll
                    for (int i = 0; i < KPL; ++i) {
                        sv[i] = sSp[r * TKT + pl * KPL + i];
                        tmax = sycl::fmax(tmax, sv[i]);
                    }
#pragma unroll
                    for (int m = 1; m < 16; m <<= 1) tmax = sycl::fmax(tmax, sycl::permute_group_by_xor(sg, tmax, m));
                    const float mo = sm[r];
                    const float nm = sycl::fmax(mo, tmax);
                    const float corr = sycl::exp(mo - nm);
                    float psum = 0.f;
#pragma unroll
                    for (int i = 0; i < KPL; ++i) {
                        const float p = sv[i] > -1e29f ? sycl::exp(sv[i] - nm) : 0.f;
                        sPp[(pl * KPL + i) * HG + r] = p;
                        psum += p;
                    }
#pragma unroll
                    for (int m = 1; m < 16; m <<= 1) psum += sycl::permute_group_by_xor(sg, psum, m);
                    const float lold = sl[r];
                    sycl::group_barrier(sg);
                    if (pl == 0) {
                        sl[r] = lold * corr + psum;
                        sm[r] = nm;
                        sc[r] = corr;
                    }
                }
                sycl::group_barrier(grp);
#pragma unroll
                for (int r = 0; r < HG; ++r) acc[r] *= sc[r];
                for (int j = 0; j < cnt; ++j) {
                    const int sl_ = sSlot[j];
                    const float v = (float)cache[(size_t)sl_ * MLA_ROW + tid];
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const vf4 p4 = *reinterpret_cast<const vf4*>(sPp + j * HG + g * 4);
                        acc[g * 4 + 0] += p4[0] * v;
                        acc[g * 4 + 1] += p4[1] * v;
                        acc[g * 4 + 2] += p4[2] * v;
                        acc[g * 4 + 3] += p4[3] * v;
                    }
                }
                sycl::group_barrier(grp);
            }
            if (S == 1) {
#pragma unroll
                for (int r = 0; r < HG; ++r) ctx[((size_t)t * a.NH + hg * HG + r) * MLA_KV_LORA + tid] = (half)(acc[r] / sl[r]);
            } else {
                float* base = part + ((size_t)(t * NHG + hg) * S + sp) * PART_STRIDE;
#pragma unroll
                for (int r = 0; r < HG; ++r) base[r * MLA_KV_LORA + tid] = acc[r];
                if (tid < HG) {
                    base[HG * MLA_KV_LORA + tid] = sm[tid];
                    base[HG * MLA_KV_LORA + HG + tid] = sl[tid];
                }
            }
        });
    });
}

// ---- Tiled XMX Flash-MLA (Alchemist SG=8, 8x8x16 DPAS for both Q@K^T and P@V, in-SLM V transpose, 0 Vt copy!) ----
// 128 threads = 16 SG8 sub-groups; 1 workgroup = 1 query t, HG=16 heads, split sp, TKT=64 keys per tile.
// Total SLM: 18 KiB (sQ) + 8 KiB (sKV) + 4 KiB (sS/sP) + 4 KiB (sTmp) + 0.5 KiB = 34.5 KiB!
#ifndef HY4S_DISABLE_SG8
static void attend_xmx(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a_in,
                       const StepParams* d_step, bool use_dfly_a, float* part, int S, int tps) {
    constexpr int TKT = 64;
    const int NHG = a_in.NH / HG;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQ(HG * MLA_ROW, h);       // 18 KiB
        sycl::local_accessor<half, 1> sKV(TKT * 64, h);          // 8 KiB VNNI-2 [32][128]
        sycl::local_accessor<float, 1> sS(HG * TKT, h);          // 4 KiB scores [16][64]
        sycl::local_accessor<half, 1> sP(HG * TKT, h);           // 2 KiB probs [16][64]
        sycl::local_accessor<float, 1> sTmp(16 * 64, h);         // 4 KiB (16 SG x 8x8)
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(HG, h), sl(HG, h), sc(HG, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(a_in.M, S, (size_t)NHG * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(8)]] {
            MlaAttnArgs a = a_in;
            if (d_step) a.p0 = use_dfly_a ? d_step->a : d_step->p0;
            const int t = (int)it.get_group(0), sp = (int)it.get_group(1), hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, lane = tid & 7;
            const int h_tile = sgi >> 3;   // 0 or 1 (8 heads each)
            const int kc_tile = sgi & 7;   // 0..7 (8 keys or 8 latent cols each)

            half* sq_ptr = &sQ[0];
            half* skv_ptr = &sKV[0];
            float* sS_ptr = &sS[0];
            half* sP_ptr = &sP[0];
            float* my_tmp = &sTmp[sgi * 64];

            auto sQ_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sq_ptr);
            auto sKV_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(skv_ptr);
            auto sS_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sS_ptr);
            auto sP_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sP_ptr);
            auto tmp_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_tmp);

            // Load Q[16][576] into SLM (128 threads x 8 halves = 1024 halves per iter)
            {
                const half* qbase = qa + ((size_t)t * a.NH + hg * HG) * MLA_ROW;
                for (int e = tid * 8; e < HG * MLA_ROW; e += 128 * 8)
                    *reinterpret_cast<vh8*>(sq_ptr + e) = *reinterpret_cast<const vh8*>(qbase + e);
            }

            const int pos = a.p0 + t;
            int nk, lo = 0;
            if (a.mode == MLA_CAUSAL) nk = pos + 1;
            else if (a.mode == MLA_WINDOW) {
                lo = std::max(0, pos - a.window + 1);
                nk = pos - lo + 1;
            } else nk = a.nkeys[t];

            const int ntile_q = (nk + TKT - 1) / TKT;
            const int tile0 = sp * tps, tile1 = std::min(tile0 + tps, ntile_q);
            if (tid < HG) {
                sm[tid] = sp == 0 ? sinks[hg * HG + tid] : NEG;
                sl[tid] = sp == 0 ? 1.0f : 0.0f;
            }

            // Output accumulator in registers: 8 col-blocks x 8 head-rows (lane owns column cb*64 + kc_tile*8 + lane)
            float o_reg[8][8];
#pragma unroll
            for (int cb = 0; cb < 8; ++cb)
#pragma unroll
                for (int r = 0; r < 8; ++r) o_reg[cb][r] = 0.0f;

            sycl::group_barrier(grp);

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc;

            for (int tile = tile0; tile < tile1; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk - j0);
                if (tid < TKT) sSlot[tid] = (tid < cnt) ? key_slot(a, a.tree, t, lo, j0 + tid) : 0;
                sycl::group_barrier(grp);

                // ---- Phase 1: S[16][64] = Q[16][576] @ K[64][576]^T ----
                xm::joint_matrix_fill(sg, mc, 0.0f);
                const int k_row = tid >> 1;      // 0..63
                const int d_half = tid & 1;      // 0 or 1 (32 dims per thread)
                const int my_slot = sSlot[k_row];
                const bool k_valid = (k_row < cnt);

                for (int d0 = 0; d0 < MLA_ROW; d0 += 64) {
                    const half* kr = cache + (size_t)my_slot * MLA_ROW + d0 + d_half * 32;
                    half* dst = skv_ptr + (d_half * 16) * 128 + k_row * 2;
                    if (k_valid) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const vh8 v = *reinterpret_cast<const vh8*>(kr + q * 8);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 0) * 128) = vh2(v[0], v[1]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 1) * 128) = vh2(v[2], v[3]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 2) * 128) = vh2(v[4], v[5]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 3) * 128) = vh2(v[6], v[7]);
                        }
                    } else {
                        const vh2 z((half)0.f, (half)0.f);
#pragma unroll
                        for (int p = 0; p < 16; ++p) *reinterpret_cast<vh2*>(dst + p * 128) = z;
                    }
                    sycl::group_barrier(grp);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQ_mp + (size_t)(h_tile * 8) * MLA_ROW + d0 + s * 16, (size_t)MLA_ROW);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * 128 + (size_t)kc_tile * 16, (size_t)128);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    sycl::group_barrier(grp);
                }

                xm::joint_matrix_store(sg, mc, sS_mp + (size_t)(h_tile * 8) * TKT + kc_tile * 8, (size_t)TKT, xm::layout::row_major);
                sycl::group_barrier(grp);

                // ---- Phase 2: Online softmax (16 subgroups = 1 subgroup per head row, 8 lanes x 8 keys) ----
                {
                    const int r = sgi;
                    float sv[8];
                    float tmax = NEG;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int k = lane * 8 + i;
                        sv[i] = (k < cnt) ? (sS_ptr[r * TKT + k] * a.scale) : NEG;
                        tmax = sycl::fmax(tmax, sv[i]);
                    }
                    tmax = sycl::reduce_over_group(sg, tmax, sycl::maximum<float>());
                    const float mo = sm[r];
                    const float nm = sycl::fmax(mo, tmax);
                    const float corr = sycl::exp(mo - nm);
                    float psum = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float p = sv[i] > -1e29f ? sycl::exp(sv[i] - nm) : 0.f;
                        sP_ptr[r * TKT + lane * 8 + i] = (half)p;
                        psum += p;
                    }
                    psum = sycl::reduce_over_group(sg, psum, sycl::plus<float>());
                    if (lane == 0) {
                        sl[r] = sl[r] * corr + psum;
                        sm[r] = nm;
                        sc[r] = corr;
                    }
                }
                sycl::group_barrier(grp);

                float corrs[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) corrs[r] = sc[h_tile * 8 + r];

                // ---- Phase 3: O[16][512] += P[16][64] @ V[64][512] (in-SLM V transpose, 8 col-blocks of 64) ----
                // 128 threads load 32 pairs of keys x 64 cols: thread tid loads key pair (2*kp, 2*kp+1) for 16 cols?
                // Even cleaner: 128 threads = 32 key-pairs (kp = tid >> 2) x 4 col-quarters (cq = tid & 3, 16 cols each)!
                const int kp = tid >> 2;         // 0..31 -> keys (2*kp, 2*kp+1)
                const int cq = tid & 3;          // 0..3 -> 16 cols [cq*16 .. cq*16+15]
                const int k0_v = 2 * kp, k1_v = 2 * kp + 1;
                const int sl0 = sSlot[k0_v], sl1 = sSlot[k1_v];
                const bool v0_ok = k0_v < cnt, v1_ok = k1_v < cnt;

                for (int cb = 0; cb < 8; ++cb) {
                    const int c0 = cb * 64 + cq * 16;
                    vh8 r0a, r0b, r1a, r1b;
                    if (v0_ok) {
                        r0a = *reinterpret_cast<const vh8*>(cache + (size_t)sl0 * MLA_ROW + c0);
                        r0b = *reinterpret_cast<const vh8*>(cache + (size_t)sl0 * MLA_ROW + c0 + 8);
                    } else {
                        r0a = vh8((half)0.f);
                        r0b = vh8((half)0.f);
                    }
                    if (v1_ok) {
                        r1a = *reinterpret_cast<const vh8*>(cache + (size_t)sl1 * MLA_ROW + c0);
                        r1b = *reinterpret_cast<const vh8*>(cache + (size_t)sl1 * MLA_ROW + c0 + 8);
                    } else {
                        r1a = vh8((half)0.f);
                        r1b = vh8((half)0.f);
                    }
                    half* dst = skv_ptr + kp * 128 + (cq * 16) * 2;
#pragma unroll
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<vh2*>(dst + i * 2) = vh2(r0a[i], r1a[i]);
#pragma unroll
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<vh2*>(dst + (8 + i) * 2) = vh2(r0b[i], r1b[i]);
                    sycl::group_barrier(grp);

                    xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sP_mp + (size_t)(h_tile * 8) * TKT + s * 16, (size_t)TKT);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * 128 + (size_t)kc_tile * 16, (size_t)128);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    xm::joint_matrix_store(sg, mc, tmp_mp, (size_t)8, xm::layout::row_major);
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int r = 0; r < 8; ++r) o_reg[cb][r] = o_reg[cb][r] * corrs[r] + my_tmp[r * 8 + lane];
                    sycl::group_barrier(grp);
                }
            }

            // Write final output (S == 1 -> ctx directly; S > 1 -> split-K partial buffer)
            if (S == 1) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float inv_l = 1.0f / sl[h_tile * 8 + r];
                    half* out_row = ctx + ((size_t)t * a.NH + hg * HG + h_tile * 8 + r) * MLA_KV_LORA;
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) out_row[cb * 64 + kc_tile * 8 + lane] = (half)(o_reg[cb][r] * inv_l);
                }
            } else {
                float* base = part + ((size_t)(t * NHG + hg) * S + sp) * PART_STRIDE;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    float* dst_row = base + (h_tile * 8 + r) * MLA_KV_LORA;
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) dst_row[cb * 64 + kc_tile * 8 + lane] = o_reg[cb][r];
                }
                if (tid < HG) {
                    base[HG * MLA_KV_LORA + tid] = sm[tid];
                    base[HG * MLA_KV_LORA + HG + tid] = sl[tid];
                }
            }
        });
    });
}

// ---- Multi-Query Tiled XMX Flash-MLA (prefill M >= 16, CAUSAL / WINDOW, S == 1): 4 queries x 16 heads = 64 rows per WG! ----
// 512 threads = 64 SG8 (8 M-tiles x 8 N-tiles): 4x LESS KV cache global bandwidth + full 64x64 XMX tiles + 33.7 KiB SLM + zero spills!
static void attend_xmx_mq(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a_in) {
    constexpr int TKT = 64;
    constexpr int KV_LD = 136;  // bank-conflict-free VNNI-2 stride (128 + 8 halves)
    const int NHG = a_in.NH / HG;
    const int nqb = (a_in.M + 3) / 4;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQP(64 * 64, h);        // 8 KiB (Q tile in Phase 1, P probs in Phase 2/3)
        sycl::local_accessor<half, 1> sKV(32 * KV_LD, h);     // 8.7 KiB VNNI-2 K/V tile
        sycl::local_accessor<float, 1> sS(64 * 64, h);        // 16 KiB (scores in Phase 1/2, reused as 64*64 sTmp in Phase 3)
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(64, h), sl(64, h), sc(64, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(nqb, 1, (size_t)NHG * 512), sycl::range<3>(1, 1, 512)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(8)]] {
            const MlaAttnArgs a = a_in;
            const int t0 = (int)it.get_group(0) * 4, hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, lane = tid & 7;  // sgi: 0..63
            const int sg_m = sgi >> 3;                 // 0..7 (8 rows: q_local = sg_m>>1, h_half = sg_m&1)
            const int sg_n = sgi & 7;                  // 0..7 (8 cols)
            const int q_local = sg_m >> 1, h_half = sg_m & 1;

            half* sqp = &sQP[0];
            half* skv = &sKV[0];
            float* sSp = &sS[0];
            float* my_tmp = sSp + sgi * 64;

            auto sQP_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sqp);
            auto sKV_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(skv);
            auto sS_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sSp);
            auto tmp_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_tmp);

            const int pos_min = a.p0 + t0;
            const int pos_max = a.p0 + std::min(t0 + 3, a.M - 1);
            const int lo_u = (a.mode == MLA_WINDOW) ? std::max(0, pos_min - a.window + 1) : 0;
            const int nk_u = pos_max - lo_u + 1;
            const int ntile = (nk_u + TKT - 1) / TKT;

            if (tid < 64) {
                sm[tid] = sinks[hg * HG + (tid & 15)];
                sl[tid] = 1.0f;
            }
            float o_reg[8][8];
#pragma unroll
            for (int cb = 0; cb < 8; ++cb)
#pragma unroll
                for (int r = 0; r < 8; ++r) o_reg[cb][r] = 0.0f;

            sycl::group_barrier(grp);

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc;

            const int qr_ld = tid >> 3, qc8_ld = tid & 7;  // for loading sQP[64][64] (8 halves/thread)
            const int q_ld_t = t0 + (qr_ld >> 4);
            const int q_ld_h = hg * HG + (qr_ld & 15);
            const bool q_ld_ok = q_ld_t < a.M;

            const int k_row = tid >> 3, d8 = tid & 7;      // for loading sKV[32][KV_LD] (8 halves/thread)
            const int kp_v = tid >> 4, cq_v = tid & 15;    // for Phase 3 V load: 32 key-pairs x 16 col-quads (4 cols/thread)

            for (int tile = 0; tile < ntile; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk_u - j0);
                if (tid < TKT) {
                    const int k_abs = lo_u + j0 + tid;
                    sSlot[tid] = (tid < cnt) ? (a.ring > 0 ? (k_abs % a.ring) : k_abs) : 0;
                }
                sycl::group_barrier(grp);

                // ---- Phase 1: S[64][64] = Q[64][576] @ K[64][576]^T ----
                xm::joint_matrix_fill(sg, mc, 0.0f);
                const int my_slot = sSlot[k_row];
                const bool k_valid = (k_row < cnt);

                for (int d0 = 0; d0 < MLA_ROW; d0 += 64) {
                    *reinterpret_cast<vh8*>(sqp + qr_ld * 64 + qc8_ld * 8) =
                        q_ld_ok ? *reinterpret_cast<const vh8*>(qa + ((size_t)q_ld_t * a.NH + q_ld_h) * MLA_ROW + d0 + qc8_ld * 8) : vh8((half)0.f);
                    const vh8 kv = k_valid ? *reinterpret_cast<const vh8*>(cache + (size_t)my_slot * MLA_ROW + d0 + d8 * 8) : vh8((half)0.f);
                    half* kdst = skv + (d8 * 4) * KV_LD + k_row * 2;
                    *reinterpret_cast<vh2*>(kdst + 0 * KV_LD) = vh2(kv[0], kv[1]);
                    *reinterpret_cast<vh2*>(kdst + 1 * KV_LD) = vh2(kv[2], kv[3]);
                    *reinterpret_cast<vh2*>(kdst + 2 * KV_LD) = vh2(kv[4], kv[5]);
                    *reinterpret_cast<vh2*>(kdst + 3 * KV_LD) = vh2(kv[6], kv[7]);
                    sycl::group_barrier(grp);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQP_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * KV_LD + (size_t)sg_n * 16, (size_t)KV_LD);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    sycl::group_barrier(grp);
                }

                xm::joint_matrix_store(sg, mc, sS_mp + (size_t)(sg_m * 8) * 64 + sg_n * 8, (size_t)64, xm::layout::row_major);
                sycl::group_barrier(grp);

                // ---- Phase 2: Online softmax (64 subgroups = 1 SG8 per row r = sgi) ----
                {
                    const int r = sgi;
                    const int tq = t0 + (r >> 4);
                    const int pos_q = a.p0 + tq;
                    const int lo_q = (a.mode == MLA_WINDOW) ? std::max(0, pos_q - a.window + 1) : 0;
                    float sv[8];
                    float tmax = NEG;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int k = lane * 8 + i;
                        const int k_abs = lo_u + j0 + k;
                        const bool ok = (k < cnt) && (tq < a.M) && (k_abs >= lo_q) && (k_abs <= pos_q);
                        sv[i] = ok ? (sSp[r * 64 + k] * a.scale) : NEG;
                        tmax = sycl::fmax(tmax, sv[i]);
                    }
                    tmax = sycl::reduce_over_group(sg, tmax, sycl::maximum<float>());
                    const float mo = sm[r];
                    const float nm = sycl::fmax(mo, tmax);
                    const float corr = sycl::exp(mo - nm);
                    float psum = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float p = sv[i] > -1e29f ? sycl::exp(sv[i] - nm) : 0.f;
                        sqp[r * 64 + lane * 8 + i] = (half)p;
                        psum += p;
                    }
                    psum = sycl::reduce_over_group(sg, psum, sycl::plus<float>());
                    if (lane == 0) {
                        sl[r] = sl[r] * corr + psum;
                        sm[r] = nm;
                        sc[r] = corr;
                    }
                }
                sycl::group_barrier(grp);

                float corrs[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) corrs[r] = sc[sg_m * 8 + r];

                // ---- Phase 3: O[64][512] += P[64][64] @ V[64][512] (reusing sS as 64*64 sTmp!) ----
                const int k0_v = 2 * kp_v, k1_v = 2 * kp_v + 1;
                const int sl0 = sSlot[k0_v], sl1 = sSlot[k1_v];
                const bool v0_ok = k0_v < cnt, v1_ok = k1_v < cnt;

                for (int cb = 0; cb < 8; ++cb) {
                    const int c0 = cb * 64 + cq_v * 4;
                    const vh4 r0 = v0_ok ? *reinterpret_cast<const vh4*>(cache + (size_t)sl0 * MLA_ROW + c0) : vh4((half)0.f);
                    const vh4 r1 = v1_ok ? *reinterpret_cast<const vh4*>(cache + (size_t)sl1 * MLA_ROW + c0) : vh4((half)0.f);
                    half* vdst = skv + kp_v * KV_LD + (cq_v * 4) * 2;
                    *reinterpret_cast<vh2*>(vdst + 0) = vh2(r0[0], r1[0]);
                    *reinterpret_cast<vh2*>(vdst + 2) = vh2(r0[1], r1[1]);
                    *reinterpret_cast<vh2*>(vdst + 4) = vh2(r0[2], r1[2]);
                    *reinterpret_cast<vh2*>(vdst + 6) = vh2(r0[3], r1[3]);
                    sycl::group_barrier(grp);

                    xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQP_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * KV_LD + (size_t)sg_n * 16, (size_t)KV_LD);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    xm::joint_matrix_store(sg, mc, tmp_mp, (size_t)8, xm::layout::row_major);
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int r = 0; r < 8; ++r) o_reg[cb][r] = o_reg[cb][r] * corrs[r] + my_tmp[r * 8 + lane];
                    sycl::group_barrier(grp);
                }
            }

            const int t_out = t0 + q_local;
            if (t_out < a.M) {
                float inv_l[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) inv_l[r] = 1.0f / sl[sg_m * 8 + r];
#pragma unroll
                for (int cb = 0; cb < 8; ++cb) {
                    const int col = cb * 64 + sg_n * 8 + lane;
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                        ctx[((size_t)t_out * a.NH + hg * HG + h_half * 8 + r) * MLA_KV_LORA + col] = (half)(o_reg[cb][r] * inv_l[r]);
                }
            }
        });
    });
}
#endif

// ---- Xe2 Battlemage SG=16 Tiled XMX Flash-MLA (8x16x16 DPAS for both Q@K^T and P@V, in-SLM V transpose) ----
#ifdef HY4S_ENABLE_SG16_XMX
static void attend_xmx_sg16(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a_in,
                            const StepParams* d_step, bool use_dfly_a, float* part, int S, int tps) {
    constexpr int TKT = 64;
    const int NHG = a_in.NH / HG;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQ(HG * MLA_ROW, h);       // 18 KiB
        sycl::local_accessor<half, 1> sKV(TKT * 64, h);          // 8 KiB VNNI-2 [32][128]
        sycl::local_accessor<float, 1> sS(HG * TKT, h);          // 4 KiB scores [16][64]
        sycl::local_accessor<half, 1> sP(HG * TKT, h);           // 2 KiB probs [16][64]
        sycl::local_accessor<float, 1> sTmp(8 * 128, h);         // 4 KiB (8 SG16 x 8x16)
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(HG, h), sl(HG, h), sc(HG, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(a_in.M, S, (size_t)NHG * 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            MlaAttnArgs a = a_in;
            if (d_step) a.p0 = use_dfly_a ? d_step->a : d_step->p0;
            const int t = (int)it.get_group(0), sp = (int)it.get_group(1), hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4, lane = tid & 15;
            const int h_tile = sgi >> 2;   // 0 or 1 (8 heads each)
            const int kc_tile = sgi & 3;   // 0..3 (16 keys or 16 latent cols each)

            half* sq_ptr = &sQ[0];
            half* skv_ptr = &sKV[0];
            float* sS_ptr = &sS[0];
            half* sP_ptr = &sP[0];
            float* my_tmp = &sTmp[sgi * 128];

            auto sQ_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sq_ptr);
            auto sKV_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(skv_ptr);
            auto sS_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sS_ptr);
            auto sP_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sP_ptr);
            auto tmp_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_tmp);

            {
                const half* qbase = qa + ((size_t)t * a.NH + hg * HG) * MLA_ROW;
                for (int e = tid * 8; e < HG * MLA_ROW; e += 128 * 8)
                    *reinterpret_cast<vh8*>(sq_ptr + e) = *reinterpret_cast<const vh8*>(qbase + e);
            }

            const int pos = a.p0 + t;
            int nk, lo = 0;
            if (a.mode == MLA_CAUSAL) nk = pos + 1;
            else if (a.mode == MLA_WINDOW) {
                lo = std::max(0, pos - a.window + 1);
                nk = pos - lo + 1;
            } else nk = a.nkeys[t];

            const int ntile_q = (nk + TKT - 1) / TKT;
            const int tile0 = sp * tps, tile1 = std::min(tile0 + tps, ntile_q);
            if (tid < HG) {
                sm[tid] = sp == 0 ? sinks[hg * HG + tid] : NEG;
                sl[tid] = sp == 0 ? 1.0f : 0.0f;
            }

            float o_reg[8][8];
#pragma unroll
            for (int cb = 0; cb < 8; ++cb)
#pragma unroll
                for (int r = 0; r < 8; ++r) o_reg[cb][r] = 0.0f;

            sycl::group_barrier(grp);

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc;

            for (int tile = tile0; tile < tile1; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk - j0);
                if (tid < TKT) sSlot[tid] = (tid < cnt) ? key_slot(a, a.tree, t, lo, j0 + tid) : 0;
                sycl::group_barrier(grp);

                xm::joint_matrix_fill(sg, mc, 0.0f);
                const int k_row = tid >> 1;
                const int d_half = tid & 1;
                const int my_slot = sSlot[k_row];
                const bool k_valid = (k_row < cnt);

                for (int d0 = 0; d0 < MLA_ROW; d0 += 64) {
                    const half* kr = cache + (size_t)my_slot * MLA_ROW + d0 + d_half * 32;
                    half* dst = skv_ptr + (d_half * 16) * 128 + k_row * 2;
                    if (k_valid) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const vh8 v = *reinterpret_cast<const vh8*>(kr + q * 8);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 0) * 128) = vh2(v[0], v[1]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 1) * 128) = vh2(v[2], v[3]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 2) * 128) = vh2(v[4], v[5]);
                            *reinterpret_cast<vh2*>(dst + (4 * q + 3) * 128) = vh2(v[6], v[7]);
                        }
                    } else {
                        const vh2 z((half)0.f, (half)0.f);
#pragma unroll
                        for (int p = 0; p < 16; ++p) *reinterpret_cast<vh2*>(dst + p * 128) = z;
                    }
                    sycl::group_barrier(grp);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQ_mp + (size_t)(h_tile * 8) * MLA_ROW + d0 + s * 16, (size_t)MLA_ROW);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * 128 + (size_t)kc_tile * 32, (size_t)128);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    sycl::group_barrier(grp);
                }

                xm::joint_matrix_store(sg, mc, sS_mp + (size_t)(h_tile * 8) * TKT + kc_tile * 16, (size_t)TKT, xm::layout::row_major);
                sycl::group_barrier(grp);

                // Online softmax: 8 SG16 subgroups x 2 head rows/SG16, 4 keys/lane
                for (int dr = 0; dr < 2; ++dr) {
                    const int r = sgi * 2 + dr;
                    float sv[4];
                    float tmax = NEG;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int k = lane * 4 + i;
                        sv[i] = (k < cnt) ? (sS_ptr[r * TKT + k] * a.scale) : NEG;
                        tmax = sycl::fmax(tmax, sv[i]);
                    }
                    tmax = sycl::reduce_over_group(sg, tmax, sycl::maximum<float>());
                    const float mo = sm[r];
                    const float nm = sycl::fmax(mo, tmax);
                    const float corr = sycl::exp(mo - nm);
                    float psum = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float p = sv[i] > -1e29f ? sycl::exp(sv[i] - nm) : 0.f;
                        sP_ptr[r * TKT + lane * 4 + i] = (half)p;
                        psum += p;
                    }
                    psum = sycl::reduce_over_group(sg, psum, sycl::plus<float>());
                    if (lane == 0) {
                        sl[r] = sl[r] * corr + psum;
                        sm[r] = nm;
                        sc[r] = corr;
                    }
                }
                sycl::group_barrier(grp);

                float corrs[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) corrs[r] = sc[h_tile * 8 + r];

                const int kp = tid >> 2;
                const int cq = tid & 3;
                const int k0_v = 2 * kp, k1_v = 2 * kp + 1;
                const int sl0 = sSlot[k0_v], sl1 = sSlot[k1_v];
                const bool v0_ok = k0_v < cnt, v1_ok = k1_v < cnt;

                for (int cb = 0; cb < 8; ++cb) {
                    const int c0 = cb * 64 + cq * 16;
                    vh8 r0a, r0b, r1a, r1b;
                    if (v0_ok) {
                        r0a = *reinterpret_cast<const vh8*>(cache + (size_t)sl0 * MLA_ROW + c0);
                        r0b = *reinterpret_cast<const vh8*>(cache + (size_t)sl0 * MLA_ROW + c0 + 8);
                    } else {
                        r0a = vh8((half)0.f);
                        r0b = vh8((half)0.f);
                    }
                    if (v1_ok) {
                        r1a = *reinterpret_cast<const vh8*>(cache + (size_t)sl1 * MLA_ROW + c0);
                        r1b = *reinterpret_cast<const vh8*>(cache + (size_t)sl1 * MLA_ROW + c0 + 8);
                    } else {
                        r1a = vh8((half)0.f);
                        r1b = vh8((half)0.f);
                    }
                    half* dst = skv_ptr + kp * 128 + (cq * 16) * 2;
#pragma unroll
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<vh2*>(dst + i * 2) = vh2(r0a[i], r1a[i]);
#pragma unroll
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<vh2*>(dst + (8 + i) * 2) = vh2(r0b[i], r1b[i]);
                    sycl::group_barrier(grp);

                    xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sP_mp + (size_t)(h_tile * 8) * TKT + s * 16, (size_t)TKT);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * 128 + (size_t)kc_tile * 32, (size_t)128);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    xm::joint_matrix_store(sg, mc, tmp_mp, (size_t)16, xm::layout::row_major);
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int r = 0; r < 8; ++r) o_reg[cb][r] = o_reg[cb][r] * corrs[r] + my_tmp[r * 16 + lane];
                    sycl::group_barrier(grp);
                }
            }

            if (S == 1) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float inv_l = 1.0f / sl[h_tile * 8 + r];
                    half* out_row = ctx + ((size_t)t * a.NH + hg * HG + h_tile * 8 + r) * MLA_KV_LORA;
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) out_row[cb * 64 + kc_tile * 16 + lane] = (half)(o_reg[cb][r] * inv_l);
                }
            } else {
                float* base = part + ((size_t)(t * NHG + hg) * S + sp) * PART_STRIDE;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    float* dst_row = base + (h_tile * 8 + r) * MLA_KV_LORA;
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) dst_row[cb * 64 + kc_tile * 16 + lane] = o_reg[cb][r];
                }
                if (tid < HG) {
                    base[HG * MLA_KV_LORA + tid] = sm[tid];
                    base[HG * MLA_KV_LORA + HG + tid] = sl[tid];
                }
            }
        });
    });
}

// ---- Multi-Query Tiled XMX Flash-MLA for Xe2 Battlemage SG=16 (4 queries x 16 heads = 64 rows per WG, 512 thr = 32 SG16) ----
static void attend_xmx_mq_sg16(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a_in) {
    constexpr int TKT = 64;
    constexpr int KV_LD = 136;
    const int NHG = a_in.NH / HG;
    const int nqb = (a_in.M + 3) / 4;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQP(64 * 64, h);
        sycl::local_accessor<half, 1> sKV(32 * KV_LD, h);
        sycl::local_accessor<float, 1> sS(64 * 64, h);  // 16 KiB: reused as 32 * 128 sTmp in Phase 3!
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(64, h), sl(64, h), sc(64, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(nqb, 1, (size_t)NHG * 512), sycl::range<3>(1, 1, 512)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const MlaAttnArgs a = a_in;
            const int t0 = (int)it.get_group(0) * 4, hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4, lane = tid & 15;  // sgi: 0..31
            const int sg_m = sgi >> 2;                  // 0..7 (8 rows each)
            const int sg_n = sgi & 3;                   // 0..3 (16 cols each)
            const int q_local = sg_m >> 1, h_half = sg_m & 1;

            half* sqp = &sQP[0];
            half* skv = &sKV[0];
            float* sSp = &sS[0];
            float* my_tmp = sSp + sgi * 128;

            auto sQP_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sqp);
            auto sKV_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(skv);
            auto sS_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sSp);
            auto tmp_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_tmp);

            const int pos_min = a.p0 + t0;
            const int pos_max = a.p0 + std::min(t0 + 3, a.M - 1);
            const int lo_u = (a.mode == MLA_WINDOW) ? std::max(0, pos_min - a.window + 1) : 0;
            const int nk_u = pos_max - lo_u + 1;
            const int ntile = (nk_u + TKT - 1) / TKT;

            if (tid < 64) {
                sm[tid] = sinks[hg * HG + (tid & 15)];
                sl[tid] = 1.0f;
            }
            float o_reg[8][8];
#pragma unroll
            for (int cb = 0; cb < 8; ++cb)
#pragma unroll
                for (int r = 0; r < 8; ++r) o_reg[cb][r] = 0.0f;

            sycl::group_barrier(grp);

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc;

            const int qr_ld = tid >> 3, qc8_ld = tid & 7;
            const int q_ld_t = t0 + (qr_ld >> 4);
            const int q_ld_h = hg * HG + (qr_ld & 15);
            const bool q_ld_ok = q_ld_t < a.M;

            const int k_row = tid >> 3, d8 = tid & 7;
            const int kp_v = tid >> 4, cq_v = tid & 15;

            for (int tile = 0; tile < ntile; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk_u - j0);
                if (tid < TKT) {
                    const int k_abs = lo_u + j0 + tid;
                    sSlot[tid] = (tid < cnt) ? (a.ring > 0 ? (k_abs % a.ring) : k_abs) : 0;
                }
                sycl::group_barrier(grp);

                xm::joint_matrix_fill(sg, mc, 0.0f);
                const int my_slot = sSlot[k_row];
                const bool k_valid = (k_row < cnt);

                for (int d0 = 0; d0 < MLA_ROW; d0 += 64) {
                    *reinterpret_cast<vh8*>(sqp + qr_ld * 64 + qc8_ld * 8) =
                        q_ld_ok ? *reinterpret_cast<const vh8*>(qa + ((size_t)q_ld_t * a.NH + q_ld_h) * MLA_ROW + d0 + qc8_ld * 8) : vh8((half)0.f);
                    const vh8 kv = k_valid ? *reinterpret_cast<const vh8*>(cache + (size_t)my_slot * MLA_ROW + d0 + d8 * 8) : vh8((half)0.f);
                    half* kdst = skv + (d8 * 4) * KV_LD + k_row * 2;
                    *reinterpret_cast<vh2*>(kdst + 0 * KV_LD) = vh2(kv[0], kv[1]);
                    *reinterpret_cast<vh2*>(kdst + 1 * KV_LD) = vh2(kv[2], kv[3]);
                    *reinterpret_cast<vh2*>(kdst + 2 * KV_LD) = vh2(kv[4], kv[5]);
                    *reinterpret_cast<vh2*>(kdst + 3 * KV_LD) = vh2(kv[6], kv[7]);
                    sycl::group_barrier(grp);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQP_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * KV_LD + (size_t)sg_n * 32, (size_t)KV_LD);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    sycl::group_barrier(grp);
                }

                xm::joint_matrix_store(sg, mc, sS_mp + (size_t)(sg_m * 8) * 64 + sg_n * 16, (size_t)64, xm::layout::row_major);
                sycl::group_barrier(grp);

                // Phase 2: Online softmax (32 SG16s x 2 rows = 64 rows, 4 keys/lane)
                for (int dr = 0; dr < 2; ++dr) {
                    const int r = sgi * 2 + dr;
                    const int tq = t0 + (r >> 4);
                    const int pos_q = a.p0 + tq;
                    const int lo_q = (a.mode == MLA_WINDOW) ? std::max(0, pos_q - a.window + 1) : 0;
                    float sv[4];
                    float tmax = NEG;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int k = lane * 4 + i;
                        const int k_abs = lo_u + j0 + k;
                        const bool ok = (k < cnt) && (tq < a.M) && (k_abs >= lo_q) && (k_abs <= pos_q);
                        sv[i] = ok ? (sSp[r * 64 + k] * a.scale) : NEG;
                        tmax = sycl::fmax(tmax, sv[i]);
                    }
                    tmax = sycl::reduce_over_group(sg, tmax, sycl::maximum<float>());
                    const float mo = sm[r];
                    const float nm = sycl::fmax(mo, tmax);
                    const float corr = sycl::exp(mo - nm);
                    float psum = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float p = sv[i] > -1e29f ? sycl::exp(sv[i] - nm) : 0.f;
                        sqp[r * 64 + lane * 4 + i] = (half)p;
                        psum += p;
                    }
                    psum = sycl::reduce_over_group(sg, psum, sycl::plus<float>());
                    if (lane == 0) {
                        sl[r] = sl[r] * corr + psum;
                        sm[r] = nm;
                        sc[r] = corr;
                    }
                }
                sycl::group_barrier(grp);

                float corrs[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) corrs[r] = sc[sg_m * 8 + r];

                const int k0_v = 2 * kp_v, k1_v = 2 * kp_v + 1;
                const int sl0 = sSlot[k0_v], sl1 = sSlot[k1_v];
                const bool v0_ok = k0_v < cnt, v1_ok = k1_v < cnt;

                for (int cb = 0; cb < 8; ++cb) {
                    const int c0 = cb * 64 + cq_v * 4;
                    const vh4 r0 = v0_ok ? *reinterpret_cast<const vh4*>(cache + (size_t)sl0 * MLA_ROW + c0) : vh4((half)0.f);
                    const vh4 r1 = v1_ok ? *reinterpret_cast<const vh4*>(cache + (size_t)sl1 * MLA_ROW + c0) : vh4((half)0.f);
                    half* vdst = skv + kp_v * KV_LD + (cq_v * 4) * 2;
                    *reinterpret_cast<vh2*>(vdst + 0) = vh2(r0[0], r1[0]);
                    *reinterpret_cast<vh2*>(vdst + 2) = vh2(r0[1], r1[1]);
                    *reinterpret_cast<vh2*>(vdst + 4) = vh2(r0[2], r1[2]);
                    *reinterpret_cast<vh2*>(vdst + 6) = vh2(r0[3], r1[3]);
                    sycl::group_barrier(grp);

                    xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        xm::joint_matrix_load(sg, ma, sQP_mp + (size_t)(sg_m * 8) * 64 + s * 16, (size_t)64);
                        xm::joint_matrix_load(sg, mb, sKV_mp + (size_t)(s * 8) * KV_LD + (size_t)sg_n * 32, (size_t)KV_LD);
                        xm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    xm::joint_matrix_store(sg, mc, tmp_mp, (size_t)16, xm::layout::row_major);
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int r = 0; r < 8; ++r) o_reg[cb][r] = o_reg[cb][r] * corrs[r] + my_tmp[r * 16 + lane];
                    sycl::group_barrier(grp);
                }
            }

            const int t_out = t0 + q_local;
            if (t_out < a.M) {
                float inv_l[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) inv_l[r] = 1.0f / sl[sg_m * 8 + r];
#pragma unroll
                for (int cb = 0; cb < 8; ++cb) {
                    const int col = cb * 64 + sg_n * 16 + lane;
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                        ctx[((size_t)t_out * a.NH + hg * HG + h_half * 8 + r) * MLA_KV_LORA + col] = (half)(o_reg[cb][r] * inv_l[r]);
                }
            }
        });
    });
}
#endif

void launch_split_reduce(const float* part, half* ctx, int M, int NH, int S) {
    const int NHG = NH / HG;
    Q().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(M, (size_t)NH * 256), sycl::range<2>(1, 256)), [=](sycl::nd_item<2> it) {
            const int t = (int)it.get_group(0), hh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
            const int hg = hh / HG, r = hh % HG;
            const float* base = part + (size_t)(t * NHG + hg) * S * PART_STRIDE;
            float mx = NEG;
            for (int s = 0; s < S; ++s) mx = sycl::fmax(mx, base[(size_t)s * PART_STRIDE + HG * MLA_KV_LORA + r]);
            float L = 0.f, a0 = 0.f, a1 = 0.f;
            for (int s = 0; s < S; ++s) {
                const float* bs = base + (size_t)s * PART_STRIDE;
                const float w = sycl::exp(bs[HG * MLA_KV_LORA + r] - mx);
                L += w * bs[HG * MLA_KV_LORA + HG + r];
                a0 += w * bs[r * MLA_KV_LORA + 2 * tid];
                a1 += w * bs[r * MLA_KV_LORA + 2 * tid + 1];
            }
            const float il = 1.0f / L;
            vh2 o;
            o[0] = (half)(a0 * il);
            o[1] = (half)(a1 * il);
            *reinterpret_cast<vh2*>(ctx + ((size_t)t * NH + hh) * MLA_KV_LORA + 2 * tid) = o;
        });
    });
}

}  // namespace

size_t mla_cache_bytes(int slots) { return (size_t)slots * MLA_ROW * sizeof(half); }

void mla_choose_splits(const MlaAttnArgs& a, int& TK, int& S, int& tps) {
    const int base = ((g_mla_batch_invariant && a.M <= 16) ? 1 : a.M) * (a.NH / HG);
    const int mk = std::max(1, max_keys_of(a));
    // For XMX prefill (M >= 16), TKT is fixed at 64 so split count S is computed for TK=64
    if (a.M >= 16 && mla_use_xmx()) {
        TK = 64;
        const int ntiles = std::max(1, (mk + TK - 1) / TK);
        const int target = mk > 4096 ? 2 * g_mla_target_blocks : g_mla_target_blocks;
        int s = (target + base - 1) / base;
        s = std::max(1, std::min(s, ntiles));
        tps = (ntiles + s - 1) / s;
        S = (ntiles + tps - 1) / tps;
        return;
    }
    const int target = mk > 4096 ? 2 * g_mla_target_blocks : g_mla_target_blocks;
    TK = 32;
    for (int cand : {128, 64}) {
        if ((long)base * ((mk + cand - 1) / cand) >= target) {
            TK = cand;
            break;
        }
    }
    const int ntiles = std::max(1, (mk + TK - 1) / TK);
    int s = (target + base - 1) / base;
    s = std::max(1, std::min(s, ntiles));
    tps = (ntiles + s - 1) / s;
    S = (ntiles + tps - 1) / tps;
}

size_t mla_workspace_bytes(const MlaAttnArgs& a) {
    int TK, S, tps;
    mla_choose_splits(a, TK, S, tps);
    if (S <= 1) return 0;
    return (size_t)a.M * (a.NH / HG) * S * PART_STRIDE * sizeof(float);
}

void mla_write_kv(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps, TreePos tp) {
    (void)cache_slots;
    write_kv_impl(kv_a, norm_w, cache, M, p0, nullptr, false, ring, theta, eps, tp);
}

void mla_write_kv_dev(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, const StepParams* d_step, bool use_dfly_a, int ring, float theta, float eps) {
    (void)cache_slots;
    write_kv_impl(kv_a, norm_w, cache, M, 0, d_step, use_dfly_a, ring, theta, eps);
}

void mla_compact_kv(half* cache, int cache_slots, int p0, int ring, TreePos src) {
    (void)cache_slots;
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int tid = (int)it.get_local_id(0);
        for (int k = 0; k < src.n; ++k) {
            const int sidx = (int)src.depth[k];
            if (sidx == k) continue;
            const int dst_slot = ring > 0 ? (p0 + k) % ring : (p0 + k);
            const int src_slot = ring > 0 ? (p0 + sidx) % ring : (p0 + sidx);
            half* dst = cache + (size_t)dst_slot * MLA_ROW;
            const half* sr = cache + (size_t)src_slot * MLA_ROW;
            for (int d = tid; d < MLA_ROW; d += 256) dst[d] = sr[d];
            sycl::group_barrier(it.get_group());
        }
    });
}

void mla_prep_q(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta) {
    if (M >= 16 && mla_use_xmx()) {
#ifndef HY4S_DISABLE_SG8
        if (dev_caps().has_xmx_sg8) {
            prep_q_xmx(q_b, wk_b, qa, M, NH, p0, theta);
            return;
        }
#endif
#ifdef HY4S_ENABLE_SG16_XMX
        if (dev_caps().has_xmx_sg16) {
            prep_q_xmx_sg16(q_b, wk_b, qa, M, NH, p0, theta);
            return;
        }
#endif
    }
    if (M == 1) prep_q_t<1>(q_b, wk_b, qa, M, NH, p0, nullptr, false, theta);
    else prep_q_t<8>(q_b, wk_b, qa, M, NH, p0, nullptr, false, theta);
}

void mla_prep_q_dev(const float* q_b, const half* wk_b, half* qa, int M, int NH, const StepParams* d_step, bool use_dfly_a, float theta) {
    if (M == 1) prep_q_t<1>(q_b, wk_b, qa, M, NH, 0, d_step, use_dfly_a, theta);
    else prep_q_t<8>(q_b, wk_b, qa, M, NH, 0, d_step, use_dfly_a, theta);
}

void mla_prep_q_sk(int fmt, const void* W_kvb, const float* q_b, half* qa, int M, int NH, int p0, const StepParams* d_step, bool use_dfly_a, float theta, TreePos tp) {
    const uint8_t* w = (const uint8_t*)W_kvb;
    auto disp = [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M == 1) prep_q_sk_t<F, 1>(w, q_b, qa, M, NH, p0, d_step, use_dfly_a, theta, tp);
        else if (M <= 4) prep_q_sk_t<F, 4>(w, q_b, qa, M, NH, p0, d_step, use_dfly_a, theta, tp);
        else prep_q_sk_t<F, 8>(w, q_b, qa, M, NH, p0, d_step, use_dfly_a, theta, tp);
    };
    switch (fmt) {
        case FMT_S4: disp(std::integral_constant<int, FMT_S4>{}); break;
        case FMT_S5: disp(std::integral_constant<int, FMT_S5>{}); break;
        case FMT_S6: disp(std::integral_constant<int, FMT_S6>{}); break;
        case FMT_S8: disp(std::integral_constant<int, FMT_S8>{}); break;
        default: fprintf(stderr, "mla_prep_q_sk: unsupported fmt %d\n", fmt); abort();
    }
}

void mla_vup(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH) {
    if (M >= 16 && mla_use_xmx()) {
#ifndef HY4S_DISABLE_SG8
        if (dev_caps().has_xmx_sg8) {
            vup_xmx(ctx, wv_b, gate, o, M, NH);
            return;
        }
#endif
#ifdef HY4S_ENABLE_SG16_XMX
        if (dev_caps().has_xmx_sg16) {
            vup_xmx_sg16(ctx, wv_b, gate, o, M, NH);
            return;
        }
#endif
    }
    if (M == 1) vup_t<1>(ctx, wv_b, gate, o, M, NH);
    else vup_t<8>(ctx, wv_b, gate, o, M, NH);
}

void mla_vup_sk(int fmt, const void* W_kvb, const half* ctx, const float* gate, half* o, int M, int NH) {
    const uint8_t* w = (const uint8_t*)W_kvb;
    auto disp = [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M == 1) vup_sk_t<F, 1>(w, ctx, gate, o, M, NH);
        else if (M <= 4) vup_sk_t<F, 4>(w, ctx, gate, o, M, NH);
        else vup_sk_t<F, 8>(w, ctx, gate, o, M, NH);
    };
    switch (fmt) {
        case FMT_S4: disp(std::integral_constant<int, FMT_S4>{}); break;
        case FMT_S5: disp(std::integral_constant<int, FMT_S5>{}); break;
        case FMT_S6: disp(std::integral_constant<int, FMT_S6>{}); break;
        case FMT_S8: disp(std::integral_constant<int, FMT_S8>{}); break;
        default: fprintf(stderr, "mla_vup_sk: unsupported fmt %d\n", fmt); abort();
    }
}

void mla_attend_dev(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a,
                    const StepParams* d_step, bool use_dfly_a, int TK, int S, int tps, void* ws, size_t ws_bytes) {
    if (S > 1) {
        const size_t need = (size_t)a.M * (a.NH / HG) * S * PART_STRIDE * sizeof(float);
        if (ws == nullptr || ws_bytes < need) {
            fprintf(stderr, "mla_attend_dev: workspace too small (%zu < %zu)\n", ws_bytes, need);
            abort();
        }
    }
    float* part = static_cast<float*>(ws);
    bool ran_xmx = false;
    if (a.M >= 16 && mla_use_xmx() && TK == 64 && a.tree.n == 0) {
#ifndef HY4S_DISABLE_SG8
        if (dev_caps().has_xmx_sg8) {
            if (S == 1 && !d_step && (a.mode == MLA_CAUSAL || a.mode == MLA_WINDOW))
                attend_xmx_mq(qa, cache, sinks, ctx, a);
            else
                attend_xmx(qa, cache, sinks, ctx, a, d_step, use_dfly_a, part, S, tps);
            ran_xmx = true;
        }
#endif
#ifdef HY4S_ENABLE_SG16_XMX
        if (!ran_xmx && dev_caps().has_xmx_sg16) {
            if (S == 1 && !d_step && (a.mode == MLA_CAUSAL || a.mode == MLA_WINDOW))
                attend_xmx_mq_sg16(qa, cache, sinks, ctx, a);
            else
                attend_xmx_sg16(qa, cache, sinks, ctx, a, d_step, use_dfly_a, part, S, tps);
            ran_xmx = true;
        }
#endif
    }
    if (!ran_xmx) {
        if (TK == 128) attend_t<128>(qa, cache, sinks, ctx, a, d_step, use_dfly_a, part, S, tps);
        else if (TK == 64) attend_t<64>(qa, cache, sinks, ctx, a, d_step, use_dfly_a, part, S, tps);
        else attend_t<32>(qa, cache, sinks, ctx, a, d_step, use_dfly_a, part, S, tps);
    }
    if (S > 1) launch_split_reduce(part, ctx, a.M, a.NH, S);
}

void mla_attend(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes) {
    int TK, S, tps;
    mla_choose_splits(a, TK, S, tps);
    mla_attend_dev(qa, cache, sinks, ctx, a, nullptr, false, TK, S, tps, ws, ws_bytes);
}

}  // namespace hy4s
