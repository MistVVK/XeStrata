// sy_indexer.h - compressed IndexPool of the 12 sparse layers, SYCL.
// Includes:
//   * XMX-accelerated score_xmx (16x64 GEMM, K=128) for prefill (T >= 16) -> 15-20x faster at 32K-128K!
//   * Device-resident StepParams variants (*_dev) with fixed grid geometry for SYCL command_graph replay
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

// iq: fp32 [T][2048] (16 heads x 128) -> rope on dims 64..127 -> half [T][2048]
inline void q_rope_impl(const float* iq, half* out, int p0_host, const StepParams* d_step, float theta, int T) {
    Q().parallel_for(NR((size_t)T * 16, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int p0 = d_step ? d_step->p0 : p0_host;
        const int g = (int)it.get_group(0), t = g >> 4, h = g & 15, d = (int)it.get_local_id(0);
        const float* q = iq + (size_t)t * 2048 + h * 128;
        float v = q[d];
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            float o0, o1;
            rope_pair_i(q[64 + 2 * i], q[64 + 2 * i + 1], p0 + t, i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        out[(size_t)t * 2048 + h * 128 + d] = (half)v;
    });
}
inline void q_rope(const float* iq, half* out, int p0, float theta, int T) { q_rope_impl(iq, out, p0, nullptr, theta, T); }
inline void q_rope_dev(const float* iq, half* out, const StepParams* d_step, float theta, int T) { q_rope_impl(iq, out, 0, d_step, theta, T); }

// ---- XMX-accelerated IndexPool score (prefill T >= 16 on Alchemist SG=8) ----
// Computes dot[16][64] = iq[t][16][128] @ pooled[g0..g0+63][128]^T with 8 XMX 8x8x16 DPAS steps, then sum_h relu(dot)*hw*hscale.
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
            // Load iq[t][16][128] (2048 halves -> 128 threads x 16 halves = 2 vh8)
            for (int e = tid * 8; e < 2048; e += 128 * 8)
                *reinterpret_cast<sycl::vec<half, 8>*>(sq_ptr + e) = *reinterpret_cast<const sycl::vec<half, 8>*>(iq + (size_t)t * 2048 + e);
            if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
            // Load pooled[g0..g0+63][128] -> VNNI-2 sK[(k>>1)*128 + g_local*2 + (k&1)] (2 threads per key, 64 dims each)
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

// Standard IndexPool score (exact fp32 accumulation for decode/verify & non-XMX; dispatches to score_xmx for prefill T >= 16)
inline void score(const half* iq, const float* hw, const half* pooled, float* sc, int p0, int G, int ldS, float hscale, int T) {
#ifndef HY4S_DISABLE_SG8
    if (T >= 16 && dev_caps().has_xmx_sg8 && !(std::getenv("HY4S_MLA_XMX") && std::getenv("HY4S_MLA_XMX")[0] == '0')) {
        score_xmx(iq, hw, pooled, sc, p0, G, ldS, hscale, T);
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

// Device-param fixed-grid score for command_graph replay: grid-stride over pool blocks gb so ANY context 0..128K works in ONE captured graph!
// Bit-identical to score() for every (t, g).
inline void score_dev(const half* iq, const float* hw, const half* pooled, float* sc, const StepParams* d_step, int ldS, float hscale, int T) {
    constexpr int NB_FIXED = 128;  // 128 groups/token covers 8K pools in 1 pass, 128K pools in 4 passes
    Q().parallel_for(NR((size_t)NB_FIXED * T, 64), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sq = lmem<float, 16 * 128>(it);
        float* shw = lmem<float, 16>(it);
        const int p0 = d_step->p0;
        const int G = sycl::max(1, (p0 + T) >> 2);
        const int nb = (G + 63) >> 6;
        const int gid = (int)it.get_group(0), t = gid / NB_FIXED, gb0 = gid % NB_FIXED, tid = (int)it.get_local_id(0);
        if (gb0 >= nb) return;
        for (int i = tid; i < 16 * 128; i += 64) sq[i] = (float)iq[(size_t)t * 2048 + i];
        if (tid < 16) shw[tid] = hw[(size_t)t * 16 + tid] * hscale;
        sycl::group_barrier(it.get_group());
        const int grp = (p0 + t) >> 2;
        for (int gb = gb0; gb < nb; gb += NB_FIXED) {
            const int g = gb * 64 + tid;
            if (g >= G) break;
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
        }
    });
}

// per token: select top n_top pools (supports both host p0 and device StepParams* d_step)
inline void select_impl(const float* score, int ldS, int p0_host, const StepParams* d_step, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        int* red = lmem<int, 8>(it);
        int* cnt_sel = lmem<int, 1>(it);
        uint32_t* thr_s = lmem<uint32_t, 1>(it);
        int* n_gt_s = lmem<int, 1>(it);
        const int p0 = d_step ? d_step->p0 : p0_host;
        const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int pos = p0 + t;
        const int grp = pos >> 2;
        int32_t* kl = keys + (size_t)t * kmax;
        if (grp <= n_top) {
            for (int i = tid; i <= pos; i += 256) kl[i] = i;
            if (tid == 0) nkeys[t] = pos + 1;
            return;
        }
        const float* sc = score + (size_t)t * ldS;
        uint32_t lo = 0, hi = 0xFFFFFFFFu;
        for (int itr = 0; itr < 33 && lo < hi; ++itr) {
            const uint32_t mid = (uint32_t)(lo + (((uint64_t)hi - lo + 1) >> 1));
            int c = 0;
            for (int g = tid; g < grp; g += 256) c += (f2key(sc[g]) >= mid) ? 1 : 0;
            c = sycl::reduce_over_group(it.get_sub_group(), c, sycl::plus<int>());
            sycl::group_barrier(it.get_group());
            if ((tid & 31) == 0) red[tid >> 5] = c;
            sycl::group_barrier(it.get_group());
            int tot = 0;
            for (int w = 0; w < 8; ++w) tot += red[w];
            if (tot >= n_top) lo = mid; else hi = mid - 1;
            sycl::group_barrier(it.get_group());
        }
        if (tid == 0) { *thr_s = lo; *cnt_sel = 0; }
        sycl::group_barrier(it.get_group());
        const uint32_t thr = *thr_s;
        int c = 0;
        for (int g = tid; g < grp; g += 256) c += (f2key(sc[g]) > thr) ? 1 : 0;
        c = sycl::reduce_over_group(it.get_sub_group(), c, sycl::plus<int>());
        sycl::group_barrier(it.get_group());
        if ((tid & 31) == 0) red[tid >> 5] = c;
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            int tot = 0;
            for (int w = 0; w < 8; ++w) tot += red[w];
            *n_gt_s = tot;
        }
        sycl::group_barrier(it.get_group());
        const int n_eq_need = n_top - *n_gt_s;
        {
            int* o_gt = lmem<int, 256>(it);
            int* o_eq = lmem<int, 256>(it);
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
                *cnt_sel = *n_gt_s + sycl::min(se, n_eq_need);
            }
            sycl::group_barrier(it.get_group());
            int sg = o_gt[tid], se = o_eq[tid];
            for (int g = g0; g < g1; ++g) {
                const uint32_t k = f2key(sc[g]);
                int s = -1;
                if (k > thr) s = sg++;
                else if (k == thr) { if (se < n_eq_need) s = *n_gt_s + se; ++se; }
                if (s >= 0)
                    for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
            }
        }
        sycl::group_barrier(it.get_group());
        const int npool = *cnt_sel;
        const int nown = pos - 4 * grp + 1;
        for (int i = tid; i < nown; i += 256) kl[4 * npool + i] = 4 * grp + i;
        if (tid == 0) nkeys[t] = 4 * npool + nown;
    });
}
inline void select(const float* score, int ldS, int p0, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    select_impl(score, ldS, p0, nullptr, n_top, kmax, keys, nkeys, T);
}
inline void select_dev(const float* score, int ldS, const StepParams* d_step, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    select_impl(score, ldS, 0, d_step, n_top, kmax, keys, nkeys, T);
}

}  // namespace idx
}  // namespace hy4s
