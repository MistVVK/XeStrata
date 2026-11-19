// sy_mla.cpp - SYCL kernels of the Hy4-Flash MLA attention (see sy_mla.h). Plain FMA kernels with local-memory staging, fp32 accumulation, split-K flash decode.
#include "sy_mla.h"

#include <algorithm>

namespace hy4s {

int g_mla_target_blocks = 56;

namespace {

constexpr int HG = 16;   // heads per work-group (the 16 rows share the key set of one query)
constexpr float NEG = -1e30f;

using vh8 = sycl::vec<half, 8>;
using vh4 = sycl::vec<half, 4>;
using vh2 = sycl::vec<half, 2>;
using vf4 = sycl::vec<float, 4>;

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

// tile size TK in {128, 64, 32} (largest one that still gives >= g_mla_target_blocks work-groups), number of splits S and tiles per split
void choose_splits(const MlaAttnArgs& a, int& TK, int& S, int& tps) {
    const int base = a.M * (a.NH / HG);
    const int mk = std::max(1, max_keys_of(a));
    const int target = mk > 4096 ? 2 * g_mla_target_blocks : g_mla_target_blocks;  // long causal decode wants more split-K work-groups
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

constexpr int PART_STRIDE = HG * MLA_KV_LORA + 32;  // floats per (query, head group, split): acc[16][512] | m[16] | l[16]

inline int key_slot(const MlaAttnArgs& a, int t, int lo, int j) {
    if (a.mode == MLA_CAUSAL) return a.ring > 0 ? j % a.ring : j;
    if (a.mode == MLA_WINDOW) return (lo + j) % a.ring;
    return a.keys[(size_t)t * a.kmax + j];
}

}  // namespace

size_t mla_cache_bytes(int slots) { return (size_t)slots * MLA_ROW * sizeof(half); }

size_t mla_workspace_bytes(const MlaAttnArgs& a) {
    int TK, S, tps;
    choose_splits(a, TK, S, tps);
    if (S <= 1) return 0;
    return (size_t)a.M * (a.NH / HG) * S * PART_STRIDE * sizeof(float);
}

// ---- KV cache write: grid M, 256 threads (2 latent elements each)
void mla_write_kv(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps) {
    (void)cache_slots;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(32, h);
        h.parallel_for(sycl::nd_range<1>((size_t)M * 256, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const float* row = kv_a + (size_t)t * MLA_ROW;
            const float v0 = row[2 * tid], v1 = row[2 * tid + 1];
            const float ss = wg_sum(it, v0 * v0 + v1 * v1, sh.get_multi_ptr<sycl::access::decorated::no>());
            const float inv = 1.0f / sycl::sqrt(ss / (float)MLA_KV_LORA + eps);
            const int pos = p0 + t;
            const int slot = ring > 0 ? pos % ring : pos;
            half* dst = cache + (size_t)slot * MLA_ROW;
            vh2 o;
            o[0] = (half)(v0 * inv * norm_w[2 * tid]);
            o[1] = (half)(v1 * inv * norm_w[2 * tid + 1]);
            *reinterpret_cast<vh2*>(dst + 2 * tid) = o;
            if (tid < 32) {
                float o0, o1;
                rope_pair(row[MLA_KV_LORA + 2 * tid], row[MLA_KV_LORA + 2 * tid + 1], pos, tid, theta, o0, o1);
                vh2 r;
                r[0] = (half)o0;
                r[1] = (half)o1;
                *reinterpret_cast<vh2*>(dst + MLA_KV_LORA + 2 * tid) = r;
            }
        });
    });
}

// ---- q absorb + rope. grid (NH, 4 column quarters, token tiles of TT), 256 threads = 4 i-subranges (48 rows each) x 64 column pairs
template <int TT>
static void prep_q_t(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta) {
    const int ntt = (M + TT - 1) / TT;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(TT * MLA_NOPE, h);
        sycl::local_accessor<float, 1> red(3 * TT * 128, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(NH, 4, (size_t)ntt * 256), sycl::range<3>(1, 1, 256)), [=](sycl::nd_item<3> it) REQD_SG32 {
            const int hh = (int)it.get_group(0), cq = (int)it.get_group(1), t0 = (int)it.get_group(2) * TT;
            const int tid = (int)it.get_local_id(2);
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

void mla_prep_q(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta) {
    if (M == 1) prep_q_t<1>(q_b, wk_b, qa, M, NH, p0, theta);
    else prep_q_t<8>(q_b, wk_b, qa, M, NH, p0, theta);
}

// ---- v up-projection. grid (NH, 4 row quarters of 64, token tiles), 256 threads = 8 sub-groups, each owns rows w, w+8, ... of its quarter
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

void mla_vup(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH) {
    if (M == 1) vup_t<1>(ctx, wv_b, gate, o, M, NH);
    else vup_t<8>(ctx, wv_b, gate, o, M, NH);
}


// ---- attention: grid (M, S, NH/16), 512 threads (sub-group size 16). One work-group = one query, 16 heads, one key range (split), keys in tiles of TKT.
// Per tile: scores S[16][TKT] (item = one key x TKT/32 heads, q tile in local memory, K rows straight from the cache), online softmax (one sub-group per head row),
// then P V with item = one latent column x 16 heads (acc[16] in registers; V = the first 512 halves of the key row).
template <int TKT>
static void attend_t(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, float* part, int S, int tps) {
    constexpr int HPT = TKT / 32;     // heads per item in the score phase
    constexpr int KPL = TKT / 16;     // keys per lane in the softmax phase
    const int NHG = a.NH / HG;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(HG * MLA_ROW, h);
        sycl::local_accessor<float, 1> sS(HG * TKT, h);
        sycl::local_accessor<float, 1> sPt(TKT * HG, h);
        sycl::local_accessor<int, 1> sSlot(TKT, h);
        sycl::local_accessor<float, 1> sm(HG, h), sl(HG, h), sc(HG, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(a.M, S, (size_t)NHG * 512), sycl::range<3>(1, 1, 512)), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int t = (int)it.get_group(0), sp = (int)it.get_group(1), hg = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            float* sqp = sq.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            float* sSp = sS.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            float* sPp = sPt.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            {  // query tile
                const half* qbase = qa + ((size_t)t * a.NH + hg * HG) * MLA_ROW;
                for (int e = tid * 8; e < HG * MLA_ROW; e += 512 * 8) {
                    float f[8];
                    ldh8(qbase + e, f);
#pragma unroll
                    for (int i = 0; i < 8; ++i) sqp[e + i] = f[i];
                }
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
            float acc[HG];
#pragma unroll
            for (int r = 0; r < HG; ++r) acc[r] = 0.f;
            sycl::group_barrier(grp);
            const int kk = tid & (TKT - 1), hgrp = tid / TKT;
            for (int tile = tile0; tile < tile1; ++tile) {
                const int j0 = tile * TKT;
                const int cnt = std::min(TKT, nk - j0);
                const bool valid = kk < cnt;
                const int slot = valid ? key_slot(a, t, lo, j0 + kk) : 0;
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
                if (tid < 256) {  // online softmax: sub-group = head row r, 16 lanes x KPL keys (the other 16 sub-groups idle)
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

void mla_attend(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes) {
    int TK, S, tps;
    choose_splits(a, TK, S, tps);
    if (S > 1 && (ws == nullptr || ws_bytes < mla_workspace_bytes(a))) {
        fprintf(stderr, "mla_attend: workspace too small (%zu < %zu)\n", ws_bytes, mla_workspace_bytes(a));
        abort();
    }
    float* part = static_cast<float*>(ws);
    if (TK == 128) attend_t<128>(qa, cache, sinks, ctx, a, part, S, tps);
    else if (TK == 64) attend_t<64>(qa, cache, sinks, ctx, a, part, S, tps);
    else attend_t<32>(qa, cache, sinks, ctx, a, part, S, tps);
    if (S > 1) {
        const int NH = a.NH, NHG = a.NH / HG;
        Q().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(a.M, (size_t)NH * 256), sycl::range<2>(1, 256)), [=](sycl::nd_item<2> it) {
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
}

}  // namespace hy4s
