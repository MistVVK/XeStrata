// sy_indexer.h - compressed IndexPool of the 12 sparse layers, SYCL.
// Includes:
//   * XMX-accelerated score_xmx (Alchemist SG8) & score_xmx_sg16 (Battlemage Xe2 SG16) for prefill (T >= 16)
//   * 4-pass 8-bit radix top-K selection (8x faster than 33-round binary search)
//   * Device-resident StepParams variants (*_dev) with fixed grid geometry for SYCL command_graph replay
//   * Tree-8 speculative verification kernels (pool_tree, q_rope_tree, score_tree, select_tree, compact_ivg)
#pragma once
#include "sy_common.h"

namespace hy4s {
namespace idx {

namespace xm = sycl::ext::oneapi::experimental::matrix;

inline void rope_pair_i(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    const float freq = 1.0f / sycl::pow(theta, (float)(2 * i) / 64.0f);
    const float ang = (float)pos * freq;
    const float sn = sycl::sin(ang), cs = sycl::cos(ang);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

inline uint32_t f2key(float f) {
    const uint32_t b = sycl::bit_cast<uint32_t>(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// store iv, ig (fp32 [T][128] each) of tokens p0..p0+T-1 into the fp16 cache ring
inline void store_impl(const float* iv, const float* ig, half* ivg, int p0_host, const StepParams* d_step, int T, int iring) {
    Q().parallel_for(NR(T, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int p0 = d_step ? d_step->p0 : p0_host;
        const int t = (int)it.get_group(0), d = (int)it.get_local_id(0);
        ivg[(size_t)((p0 + t) % iring) * 256 + d] = (half)iv[(size_t)t * 128 + d];
        ivg[(size_t)((p0 + t) % iring) * 256 + 128 + d] = (half)ig[(size_t)t * 128 + d];
    });
}
inline void store(const float* iv, const float* ig, half* ivg, int p0, int T, int iring) {
    store_impl(iv, ig, ivg, p0, nullptr, T, iring);
}
inline void store_dev(const float* iv, const float* ig, half* ivg, const StepParams* d_step, int T, int iring) {
    store_impl(iv, ig, ivg, 0, d_step, T, iring);
}

// pool completed groups g in [g_lo, g_lo + n)
inline void pool(const half* ivg, const float* ape, const float* knw, half* pooled, int g_lo, int n, float theta, float eps, int iring) {
    Q().parallel_for(NR(n, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        float* sv = lmem<float, 128>(it);
        const int g = g_lo + (int)it.get_group(0), d = (int)it.get_local_id(0);
        float l[4], mx = -3.0e38f;
        for (int j = 0; j < 4; ++j) {
            l[j] = (float)ivg[(size_t)((4 * g + j) % iring) * 256 + 128 + d] + ape[j * 128 + d];
            mx = sycl::fmax(mx, l[j]);
        }
        float s = 0;
        for (int j = 0; j < 4; ++j) {
            l[j] = sycl::native::exp(l[j] - mx);
            s += l[j];
        }
        float acc = 0;
        for (int j = 0; j < 4; ++j) acc += (l[j] / s) * (float)ivg[(size_t)((4 * g + j) % iring) * 256 + d];
        const float ss = wg_sum(it, acc * acc, sh);
        const float inv = sycl::rsqrt(ss / 128.0f + eps);
        float v = acc * inv * knw[d];
        sv[d] = v;
        sycl::group_barrier(it.get_group());
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            const float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1];
            float o0, o1;
            rope_pair_i(x0, x1, 4 * g, i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        pooled[(size_t)g * 128 + d] = (half)v;
    });
}

// Device-param fixed-grid pool for command_graph replay: launches nmax = (T + 3) / 4 groups and early-exits when g > g_hi
inline void pool_dev(const half* ivg, const float* ape, const float* knw, half* pooled, const StepParams* d_step, int T, float theta, float eps, int iring) {
    const int nmax = (T + 3) / 4;
    Q().parallel_for(NR(nmax, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int p0 = d_step->p0;
        const int g_lo = p0 >> 2, g_hi = ((p0 + T) >> 2) - 1;
        const int g = g_lo + (int)it.get_group(0);
        if (g > g_hi) return;
        float* sh = lmem<float, 32>(it);
        float* sv = lmem<float, 128>(it);
        const int d = (int)it.get_local_id(0);
        float l[4], mx = -3.0e38f;
        for (int j = 0; j < 4; ++j) {
            l[j] = (float)ivg[(size_t)((4 * g + j) % iring) * 256 + 128 + d] + ape[j * 128 + d];
            mx = sycl::fmax(mx, l[j]);
        }
        float s = 0;
        for (int j = 0; j < 4; ++j) {
            l[j] = sycl::native::exp(l[j] - mx);
            s += l[j];
        }
        float acc = 0;
        for (int j = 0; j < 4; ++j) acc += (l[j] / s) * (float)ivg[(size_t)((4 * g + j) % iring) * 256 + d];
        const float ss = wg_sum(it, acc * acc, sh);
        const float inv = sycl::rsqrt(ss / 128.0f + eps);
        float v = acc * inv * knw[d];
        sv[d] = v;
        sycl::group_barrier(it.get_group());
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            const float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1];
            float o0, o1;
            rope_pair_i(x0, x1, 4 * g, i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        pooled[(size_t)g * 128 + d] = (half)v;
    });
}

// Tree-8 temporary per-node pooled keys for up to 2 groups g in [p0/4, (p0+depth[t])/4) along node t's ancestor chain.
// tmp_pooled: half [T * 2][128]. Works for both eager and command_graph (d_step != nullptr).
inline void pool_tree(const half* ivg, const float* ape, const float* knw, half* tmp_pooled, int p0_host, TreePos tp_host,
                      const StepParams* d_step, int T, float theta, float eps, int iring) {
    Q().parallel_for(NR((size_t)T * 2, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int gid = (int)it.get_group(0);
        const int t = gid >> 1, k = gid & 1, d = (int)it.get_local_id(0);
        const int p0 = d_step ? d_step->p0 : p0_host;
        const TreePos tp = d_step ? d_step->tree : tp_host;
        const int g0 = p0 >> 2;
        const int grp_t = (p0 + (int)tp.depth[t]) >> 2;
        const int g = g0 + k;
        if (g >= grp_t) return;
        float* sh = lmem<float, 32>(it);
        float* sv = lmem<float, 128>(it);
        int sl[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lp = 4 * g + j;
            const int phys = lp < p0 ? lp : p0 + (int)tp.anc[t][(lp - p0) & 7];
            sl[j] = phys % iring;
        }
        float l[4], mx = -3.0e38f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            l[j] = (float)ivg[(size_t)sl[j] * 256 + 128 + d] + ape[j * 128 + d];
            mx = sycl::fmax(mx, l[j]);
        }
        float s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            l[j] = sycl::native::exp(l[j] - mx);
            s += l[j];
        }
        float acc = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc += (l[j] / s) * (float)ivg[(size_t)sl[j] * 256 + d];
        const float ss = wg_sum(it, acc * acc, sh);
        const float inv = sycl::rsqrt(ss / 128.0f + eps);
        float v = acc * inv * knw[d];
        sv[d] = v;
        sycl::group_barrier(it.get_group());
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            const float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1];
            float o0, o1;
            rope_pair_i(x0, x1, 4 * g, i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        tmp_pooled[(size_t)(t * 2 + k) * 128 + d] = (half)v;
    });
}

// iq: fp32 [T][2048] (16 heads x 128) -> rope on dims 64..127 -> half [T][2048]
inline void q_rope_impl(const float* iq, half* out, int p0_host, const StepParams* d_step, float theta, int T, TreePos tp_host = {}) {
    Q().parallel_for(NR((size_t)T * 16, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int p0 = d_step ? d_step->p0 : p0_host;
        const bool tree_on = d_step ? (d_step->tree_on != 0) : (tp_host.n > 0);
        const int g = (int)it.get_group(0), t = g >> 4, h = g & 15, d = (int)it.get_local_id(0);
        const int pos = p0 + (tree_on ? (int)(d_step ? d_step->tree.depth[t] : tp_host.depth[t]) : t);
        const float* q = iq + (size_t)t * 2048 + h * 128;
        float v = q[d];
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            float o0, o1;
            rope_pair_i(q[64 + 2 * i], q[64 + 2 * i + 1], pos, i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        out[(size_t)t * 2048 + h * 128 + d] = (half)v;
    });
}
inline void q_rope(const float* iq, half* out, int p0, float theta, int T) { q_rope_impl(iq, out, p0, nullptr, theta, T); }
inline void q_rope_dev(const float* iq, half* out, const StepParams* d_step, float theta, int T) { q_rope_impl(iq, out, 0, d_step, theta, T); }
inline void q_rope_tree(const float* iq, half* out, int p0, TreePos tp, const StepParams* d_step, float theta, int T) {
    q_rope_impl(iq, out, p0, d_step, theta, T, tp);
}

// ---- XMX-accelerated IndexPool score (prefill T >= 16 on Alchemist SG=8) ----
#ifndef HY4S_DISABLE_SG8
inline void score_xmx(const half* iq, const float* hw, const half* pooled, float* sc, int p0, int G, int ldS, float hscale, int T) {
    const int nb = (G + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQ(16 * 128, h);   // 4 KiB
        sycl::local_accessor<half, 1> sK(64 * 128, h);   // 16 KiB VNNI-2 [64][128]
        sycl::local_accessor<float, 1> sD(16 * 64, h);   // 4 KiB dots
        sycl::local_accessor<float, 1> shw(16, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nb * T * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0), t = gid / nb, gb = gid % nb, tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, h_tile = sgi >> 3, g_tile = sgi & 7;
            const int grp = (p0 + t) >> 2;
            const int g0 = gb * 64;
            if (g0 >= grp) {
                if (tid < 64 && g0 + tid < G) sc[(size_t)t * ldS + g0 + tid] = -INFINITY;
                return;
            }
            half* sq_ptr = &sQ[0];
            half* sk_ptr = &sK[0];
            float* sd_ptr = &sD[0];
            for (int e = tid * 8; e < 2048; e += 128 * 8)
                *reinterpret_cast<sycl::vec<half, 8>*>(sq_ptr + e) = *reinterpret_cast<const sycl::vec<half, 8>*>(iq + (size_t)t * 2048 + e);
            if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
            const int gl = tid >> 1, kh = tid & 1;
            const int g = g0 + gl;
            half* dst = sk_ptr + (kh * 32) * 128 + gl * 2;
            if (g < grp && g < G) {
                const half* pr = pooled + (size_t)g * 128 + kh * 64;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const sycl::vec<half, 8> v = *reinterpret_cast<const sycl::vec<half, 8>*>(pr + q * 8);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 0) * 128) = sycl::vec<half, 2>(v[0], v[1]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 1) * 128) = sycl::vec<half, 2>(v[2], v[3]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 2) * 128) = sycl::vec<half, 2>(v[4], v[5]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 3) * 128) = sycl::vec<half, 2>(v[6], v[7]);
                }
            } else {
                const sycl::vec<half, 2> z((half)0.f, (half)0.f);
#pragma unroll
                for (int p = 0; p < 32; ++p) *reinterpret_cast<sycl::vec<half, 2>*>(dst + p * 128) = z;
            }
            sycl::group_barrier(it.get_group());

            auto sQ_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sq_ptr);
            auto sK_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sk_ptr);
            auto sD_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sd_ptr);
            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc;
            xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                xm::joint_matrix_load(sg, ma, sQ_mp + (size_t)(h_tile * 8) * 128 + s * 16, (size_t)128);
                xm::joint_matrix_load(sg, mb, sK_mp + (size_t)(s * 8) * 128 + (size_t)g_tile * 16, (size_t)128);
                xm::joint_matrix_mad(sg, mc, ma, mb, mc);
            }
            xm::joint_matrix_store(sg, mc, sD_mp + (size_t)(h_tile * 8) * 64 + g_tile * 8, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            if (tid < 64) {
                const int gg = g0 + tid;
                if (gg < G) {
                    if (gg >= grp) {
                        sc[(size_t)t * ldS + gg] = -INFINITY;
                    } else {
                        float res = 0.0f;
#pragma unroll
                        for (int hh = 0; hh < 16; ++hh) res += sycl::fmax(sd_ptr[hh * 64 + tid], 0.0f) * shw[hh];
                        sc[(size_t)t * ldS + gg] = res;
                    }
                }
            }
        });
    });
}
#endif

// ---- XMX-accelerated IndexPool score (prefill T >= 16 on Xe2 Battlemage SG=16, 8x16x16 DPAS) ----
#ifdef HY4S_ENABLE_SG16_XMX
inline void score_xmx_sg16(const half* iq, const float* hw, const half* pooled, float* sc, int p0, int G, int ldS, float hscale, int T) {
    const int nb = (G + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sQ(16 * 128, h);   // 4 KiB
        sycl::local_accessor<half, 1> sK(64 * 128, h);   // 16 KiB VNNI-2 [64][128]
        sycl::local_accessor<float, 1> sD(16 * 64, h);   // 4 KiB dots
        sycl::local_accessor<float, 1> shw(16, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nb * T * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gid = (int)it.get_group(0), t = gid / nb, gb = gid % nb, tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4, h_tile = sgi >> 2, g_tile = sgi & 3;  // 2 H-tiles (8 heads) x 4 G-tiles (16 groups)
            const int grp = (p0 + t) >> 2;
            const int g0 = gb * 64;
            if (g0 >= grp) {
                if (tid < 64 && g0 + tid < G) sc[(size_t)t * ldS + g0 + tid] = -INFINITY;
                return;
            }
            half* sq_ptr = &sQ[0];
            half* sk_ptr = &sK[0];
            float* sd_ptr = &sD[0];
            for (int e = tid * 8; e < 2048; e += 128 * 8)
                *reinterpret_cast<sycl::vec<half, 8>*>(sq_ptr + e) = *reinterpret_cast<const sycl::vec<half, 8>*>(iq + (size_t)t * 2048 + e);
            if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
            const int gl = tid >> 1, kh = tid & 1;
            const int g = g0 + gl;
            half* dst = sk_ptr + (kh * 32) * 128 + gl * 2;
            if (g < grp && g < G) {
                const half* pr = pooled + (size_t)g * 128 + kh * 64;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const sycl::vec<half, 8> v = *reinterpret_cast<const sycl::vec<half, 8>*>(pr + q * 8);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 0) * 128) = sycl::vec<half, 2>(v[0], v[1]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 1) * 128) = sycl::vec<half, 2>(v[2], v[3]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 2) * 128) = sycl::vec<half, 2>(v[4], v[5]);
                    *reinterpret_cast<sycl::vec<half, 2>*>(dst + (4 * q + 3) * 128) = sycl::vec<half, 2>(v[6], v[7]);
                }
            } else {
                const sycl::vec<half, 2> z((half)0.f, (half)0.f);
#pragma unroll
                for (int p = 0; p < 32; ++p) *reinterpret_cast<sycl::vec<half, 2>*>(dst + p * 128) = z;
            }
            sycl::group_barrier(it.get_group());

            auto sQ_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sq_ptr);
            auto sK_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sk_ptr);
            auto sD_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sd_ptr);
            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc;
            xm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                xm::joint_matrix_load(sg, ma, sQ_mp + (size_t)(h_tile * 8) * 128 + s * 16, (size_t)128);
                xm::joint_matrix_load(sg, mb, sK_mp + (size_t)(s * 8) * 128 + (size_t)g_tile * 32, (size_t)128);
                xm::joint_matrix_mad(sg, mc, ma, mb, mc);
            }
            xm::joint_matrix_store(sg, mc, sD_mp + (size_t)(h_tile * 8) * 64 + g_tile * 16, (size_t)64, xm::layout::row_major);
            sycl::group_barrier(it.get_group());

            if (tid < 64) {
                const int gg = g0 + tid;
                if (gg < G) {
                    if (gg >= grp) {
                        sc[(size_t)t * ldS + gg] = -INFINITY;
                    } else {
                        float res = 0.0f;
#pragma unroll
                        for (int hh = 0; hh < 16; ++hh) res += sycl::fmax(sd_ptr[hh * 64 + tid], 0.0f) * shw[hh];
                        sc[(size_t)t * ldS + gg] = res;
                    }
                }
            }
        });
    });
}
#endif

// Standard IndexPool score (exact fp32 accumulation for decode/verify & non-XMX; dispatches to score_xmx / score_xmx_sg16 for prefill T >= 16)
inline void score(const half* iq, const float* hw, const half* pooled, float* sc, int p0, int G, int ldS, float hscale, int T) {
    const bool allow_xmx = !(std::getenv("HY4S_MLA_XMX") && std::getenv("HY4S_MLA_XMX")[0] == '0');
#ifndef HY4S_DISABLE_SG8
    if (T >= 16 && allow_xmx && dev_caps().has_xmx_sg8) {
        score_xmx(iq, hw, pooled, sc, p0, G, ldS, hscale, T);
        return;
    }
#endif
#ifdef HY4S_ENABLE_SG16_XMX
    if (T >= 16 && allow_xmx && dev_caps().has_xmx_sg16) {
        score_xmx_sg16(iq, hw, pooled, sc, p0, G, ldS, hscale, T);
        return;
    }
#endif
    const int nb = (G + 63) / 64;
    Q().parallel_for(NR((size_t)nb * T, 64), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sq = lmem<float, 16 * 128>(it);
        float* shw = lmem<float, 16>(it);
        const int gid = (int)it.get_group(0), t = gid / nb, gb = gid % nb, tid = (int)it.get_local_id(0);
        const int g = gb * 64 + tid;
        for (int i = tid; i < 16 * 128; i += 64) sq[i] = (float)iq[(size_t)t * 2048 + i];
        if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
        sycl::group_barrier(it.get_group());
        if (g >= G) return;
        const int grp = (p0 + t) >> 2;
        float res;
        if (g >= grp) {
            res = -INFINITY;
        } else {
            const half* pr = pooled + (size_t)g * 128;
            float pk[128];
            for (int i = 0; i < 128; ++i) pk[i] = (float)pr[i];
            res = 0;
            for (int h = 0; h < 16; ++h) {
                float dsum = 0;
                for (int k = 0; k < 128; ++k) dsum += sq[h * 128 + k] * pk[k];
                res += sycl::fmax(dsum, 0.0f) * shw[h];
            }
        }
        sc[(size_t)t * ldS + g] = res;
    });
}

// Device-param fixed-grid score for command_graph replay (supports both linear and Tree-8 with tmp_pooled!)
inline void score_dev_impl(const half* iq, const float* hw, const half* pooled, const half* tmp_pooled, float* sc,
                           int p0_host, TreePos tp_host, const StepParams* d_step, int ldS, float hscale, int T) {
    constexpr int NB_FIXED = 128;  // 128 groups/token covers 8K pools in 1 pass, 128K pools in 4 passes
    Q().parallel_for(NR((size_t)NB_FIXED * T, 64), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sq = lmem<float, 16 * 128>(it);
        float* shw = lmem<float, 16>(it);
        const int p0 = d_step ? d_step->p0 : p0_host;
        const bool tree_on = d_step ? (d_step->tree_on != 0) : (tp_host.n > 0);
        const TreePos tp = d_step ? d_step->tree : tp_host;
        int max_d = T - 1;
        if (tree_on) {
            max_d = 0;
            for (int i = 0; i < tp.n; ++i) max_d = sycl::max(max_d, (int)tp.depth[i]);
        }
        const int G = sycl::max(1, (p0 + max_d + 1) >> 2);
        const int nb = (G + 63) >> 6;
        const int gid = (int)it.get_group(0), t = gid / NB_FIXED, gb0 = gid % NB_FIXED, tid = (int)it.get_local_id(0);
        if (gb0 >= nb) return;
        for (int i = tid; i < 16 * 128; i += 64) sq[i] = (float)iq[(size_t)t * 2048 + i];
        if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
        sycl::group_barrier(it.get_group());
        const int g0 = p0 >> 2;
        const int grp = (p0 + (tree_on ? (int)tp.depth[t] : t)) >> 2;
        for (int gb = gb0; gb < nb; gb += NB_FIXED) {
            const int g = gb * 64 + tid;
            if (g >= G) break;
            float res;
            if (g >= grp) {
                res = -INFINITY;
            } else {
                const half* pr = (tree_on && tmp_pooled && g >= g0)
                                     ? (tmp_pooled + (size_t)(t * 2 + (g - g0)) * 128)
                                     : (pooled + (size_t)g * 128);
                float pk[128];
                for (int i = 0; i < 128; ++i) pk[i] = (float)pr[i];
                res = 0;
                for (int h = 0; h < 16; ++h) {
                    float dsum = 0;
                    for (int k = 0; k < 128; ++k) dsum += sq[h * 128 + k] * pk[k];
                    res += sycl::fmax(dsum, 0.0f) * shw[h];
                }
            }
            sc[(size_t)t * ldS + g] = res;
        }
    });
}
inline void score_dev(const half* iq, const float* hw, const half* pooled, float* sc, const StepParams* d_step, int ldS, float hscale, int T) {
    score_dev_impl(iq, hw, pooled, nullptr, sc, 0, {}, d_step, ldS, hscale, T);
}
inline void score_tree(const half* iq, const float* hw, const half* pooled, const half* tmp_pooled, float* sc, int p0, TreePos tp,
                       const StepParams* d_step, int ldS, float hscale, int T) {
    score_dev_impl(iq, hw, pooled, tmp_pooled, sc, p0, tp, d_step, ldS, hscale, T);
}

// per token: select top n_top pools via 4-pass 8-bit radix select (8x fewer passes than 33-round binary search; supports linear & Tree-8 + d_step!)
inline void select_impl(const float* score, int ldS, int p0_host, const StepParams* d_step, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T, TreePos tp_host = {}) {
    Q().submit([&](sycl::handler& ch) {
        sycl::local_accessor<int, 1> hist(256, ch);
        sycl::local_accessor<int, 1> o_gt(256, ch);
        sycl::local_accessor<int, 1> o_eq(256, ch);
        sycl::local_accessor<int, 1> red(16, ch);
        sycl::local_accessor<int, 1> si(3, ch);       // [0]=need, [1]=n_gt, [2]=cnt_sel
        sycl::local_accessor<uint32_t, 1> su(2, ch);  // [0]=prefix, [1]=thr
        ch.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int p0 = d_step ? d_step->p0 : p0_host;
            const bool tree_on = d_step ? (d_step->tree_on != 0) : (tp_host.n > 0);
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int pos = p0 + (tree_on ? (int)(d_step ? d_step->tree.depth[t] : tp_host.depth[t]) : t);
            const int grp = pos >> 2;
            int32_t* kl = keys + (size_t)t * kmax;
            if (grp <= n_top) {
                for (int i = tid; i <= pos; i += 256) kl[i] = i;
                if (tid == 0) nkeys[t] = pos + 1;
                return;
            }
            const float* sc = score + (size_t)t * ldS;
            if (tid == 0) { su[0] = 0u; si[0] = n_top; }
            sycl::group_barrier(it.get_group());
            for (int pass = 0; pass < 4; ++pass) {
                const int shift = 24 - 8 * pass;
                const uint32_t mask = (pass == 0) ? 0u : (~0u << (shift + 8));
                hist[tid] = 0;
                sycl::group_barrier(it.get_group());
                const uint32_t pre = su[0];
                for (int g = tid; g < grp; g += 256) {
                    const uint32_t k = f2key(sc[g]);
                    if ((k & mask) == pre) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> hbin(hist[(k >> shift) & 255u]);
                        hbin.fetch_add(1);
                    }
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    int acc = 0, d = 255;
                    const int need = si[0];
                    for (; d > 0; --d) {
                        if (acc + hist[d] >= need) break;
                        acc += hist[d];
                    }
                    si[0] = need - acc;
                    su[0] = pre | ((uint32_t)d << shift);
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) { su[1] = su[0]; si[2] = 0; }
            sycl::group_barrier(it.get_group());
            const uint32_t thr = su[1];
            auto sg_obj = it.get_sub_group();
            int c = 0;
            for (int g = tid; g < grp; g += 256) c += (f2key(sc[g]) > thr) ? 1 : 0;
            c = sycl::reduce_over_group(sg_obj, c, sycl::plus<int>());
            if (sg_obj.get_local_linear_id() == 0) red[sg_obj.get_group_linear_id()] = c;
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                int tot = 0;
                const int nsg = (int)sg_obj.get_group_linear_range();
                for (int w = 0; w < nsg; ++w) tot += red[w];
                si[1] = tot;
            }
            sycl::group_barrier(it.get_group());
            const int n_gt = si[1];
            const int n_eq_need = n_top - n_gt;
            {
                const int chunk = (grp + 255) / 256;
                const int g0 = tid * chunk, g1 = sycl::min(grp, g0 + chunk);
                int cg = 0, ce = 0;
                for (int g = g0; g < g1; ++g) {
                    const uint32_t k = f2key(sc[g]);
                    cg += k > thr ? 1 : 0;
                    ce += k == thr ? 1 : 0;
                }
                o_gt[tid] = cg;
                o_eq[tid] = ce;
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    int sg = 0, se = 0;
                    for (int i = 0; i < 256; ++i) {
                        const int a = o_gt[i], b = o_eq[i];
                        o_gt[i] = sg;
                        o_eq[i] = se;
                        sg += a;
                        se += b;
                    }
                    si[2] = n_gt + sycl::min(se, n_eq_need);
                }
                sycl::group_barrier(it.get_group());
                int sg = o_gt[tid], se = o_eq[tid];
                for (int g = g0; g < g1; ++g) {
                    const uint32_t k = f2key(sc[g]);
                    int s = -1;
                    if (k > thr) s = sg++;
                    else if (k == thr) { if (se < n_eq_need) s = n_gt + se; ++se; }
                    if (s >= 0)
                        for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
                }
            }
            sycl::group_barrier(it.get_group());
            const int npool = si[2];
            const int nown = pos - 4 * grp + 1;
            for (int i = tid; i < nown; i += 256) kl[4 * npool + i] = 4 * grp + i;
            if (tid == 0) nkeys[t] = 4 * npool + nown;
        });
    });
}
inline void select(const float* score, int ldS, int p0, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    select_impl(score, ldS, p0, nullptr, n_top, kmax, keys, nkeys, T);
}
inline void select_dev(const float* score, int ldS, const StepParams* d_step, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    select_impl(score, ldS, 0, d_step, n_top, kmax, keys, nkeys, T);
}
inline void select_tree(const float* score, int ldS, int p0, TreePos tp, const StepParams* d_step, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    select_impl(score, ldS, p0, d_step, n_top, kmax, keys, nkeys, T, tp);
}

// Compact accepted tree-branch ivg rows: for k = 0..src.n-1 copy physical slot (p0 + src.depth[k]) % iring -> (p0 + k) % iring.
inline void compact_ivg(half* ivg, int p0, TreePos src, int iring) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int tid = (int)it.get_local_id(0);
        for (int k = 0; k < src.n; ++k) {
            const int sidx = (int)src.depth[k];
            if (sidx == k) continue;
            ivg[(size_t)((p0 + k) % iring) * 256 + tid] = ivg[(size_t)((p0 + sidx) % iring) * 256 + tid];
            sycl::group_barrier(it.get_group());
        }
    });
}

}  // namespace idx
}  // namespace hy4s
