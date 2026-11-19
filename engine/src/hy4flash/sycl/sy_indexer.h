// sy_indexer.h - compressed IndexPool of the 12 sparse layers, SYCL (port of run/hy4_indexer.hip.h).
// kpool 4, 16 index heads x 128 dims, top-(topk/4 - 1) pools + the current group.
// Per layer state: ivg cache half [ctx][256] (iv | ig of every token), pooled keys half [ctx/4][128] (normed, rope at the group's first position).
#pragma once
#include "sy_common.h"

namespace hy4s {
namespace idx {

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

// store iv, ig (fp32 [T][128] each) of tokens p0..p0+T-1 into the fp16 cache rows
inline void store(const float* iv, const float* ig, half* ivg, int p0, int T, int iring) {  // ivg is a ring of iring (multiple of 4) token rows: only the tokens of the groups still to be pooled are needed
    Q().parallel_for(NR(T, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int t = (int)it.get_group(0), d = (int)it.get_local_id(0);
        ivg[(size_t)((p0 + t) % iring) * 256 + d] = (half)iv[(size_t)t * 128 + d];
        ivg[(size_t)((p0 + t) % iring) * 256 + 128 + d] = (half)ig[(size_t)t * 128 + d];
    });
}

// pool completed groups g in [g_lo, g_lo + n): softmax over the 4 tokens of (ig + ape), weighted sum of iv, rmsnorm(key_norm), rope at 4g on dims 64..127
inline void pool(const half* ivg, const float* ape /*[4][128]*/, const float* knw /*[128]*/, half* pooled, int g_lo, int n, float theta, float eps, int iring) {
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

// iq: fp32 [T][2048] (16 heads x 128) -> rope on dims 64..127 of every head at position p0+t -> half [T][2048]
inline void q_rope(const float* iq, half* out, int p0, float theta, int T) {
    Q().parallel_for(NR((size_t)T * 16, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
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

// score[t][g] = sum_h relu(iq[t][h] . pooled[g]) * hw[t][h] * hscale, for g < grp(t) = (p0+t)/4, else -inf.  group = (64 pools, 1 token) = (gb*T + t), 64 threads.
inline void score(const half* iq, const float* hw, const half* pooled, float* sc, int p0, int G, int ldS, float hscale, int T) {
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

// per token: select the top n_top pools among g < grp (all of them if grp <= n_top), emit the key list (cache slots = positions):
// the 4 tokens of each selected pool, then the own group's tokens 4*grp..t.  keys [T][kmax], nkeys [T].  256 threads.
inline void select(const float* score, int ldS, int p0, int n_top, int kmax, int32_t* keys, int32_t* nkeys, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        int* red = lmem<int, 8>(it);
        int* cnt_sel = lmem<int, 1>(it);
        uint32_t* thr_s = lmem<uint32_t, 1>(it);
        int* n_gt_s = lmem<int, 1>(it);
        const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int pos = p0 + t;
        const int grp = pos >> 2;
        int32_t* kl = keys + (size_t)t * kmax;
        if (grp <= n_top) {  // everything before the own group plus the own group: causal
            for (int i = tid; i <= pos; i += 256) kl[i] = i;
            if (tid == 0) nkeys[t] = pos + 1;
            return;
        }
        const float* sc = score + (size_t)t * ldS;
        uint32_t lo = 0, hi = 0xFFFFFFFFu;  // find max thr with count(key >= thr) >= n_top
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
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> cnt(*cnt_sel);
        for (int g = tid; g < grp; g += 256) {
            if (f2key(sc[g]) > thr) {
                const int s = cnt.fetch_add(1);
                for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
            }
        }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            int s = *cnt_sel, need = n_eq_need;
            for (int g = 0; g < grp && need > 0; ++g) {
                if (f2key(sc[g]) == thr) {
                    for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
                    ++s;
                    --need;
                }
            }
            *cnt_sel = s;
        }
        sycl::group_barrier(it.get_group());
        const int npool = *cnt_sel;
        const int nown = pos - 4 * grp + 1;
        for (int i = tid; i < nown; i += 256) kl[4 * npool + i] = 4 * grp + i;
        if (tid == 0) nkeys[t] = 4 * npool + nown;
    });
}

}  // namespace idx
}  // namespace hy4s
