// sy_vision.h - the Hy4-Flash image path on the Intel Arc (SYCL): the 27-layer ViT + rmsnorm_mlp_adapter of Hy4-Flash-Vision.strata and the LM-side pieces a prefill with pictures needs
// (embedding scatter, 3D rope for MLA / indexer, Engram mask). SYCL port of the HY4-VISION hy4_vision.hip.h (f-hy4-vision); the host parts (picture decode, PIL-exact resize, spans,
// 3D positions) are reused unchanged from run/hy4_vision.h. Design and numbers: phaseA-tests/hy4/HY4-VISION-DESIGN.md.
//
// Memory on an 8 GB card: the 934 MB shard is read into PINNED HOST memory on the first picture and stays there; every phase (embedding, each of the 27 layers, the adapter) copies only
// the tensors it needs into a small device arena (~80 MB) in front of its kernels. fp16 weights, fp32 residual stream and accumulation, fp16 GEMM inputs. The GEMM runs on the XMX engines.
#pragma once
#include <atomic>
#include <chrono>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include "hy4_io.h"
#include "hy4_vision.h"
#include "sy_common.h"

namespace hy4s {
namespace vit {

constexpr int VA_HD = 72;  // head dimension (1152 / 16)
inline double tnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
enum { EPI_H = 0, EPI_GELU_TANH = 1, EPI_GELU_ERF = 2, EPI_RES_ADD = 3, EPI_RES_SET_POS = 4, EPI_F32_ROWMAP = 5 };

inline float gelu_tanh_f(float x) { return 0.5f * x * (1.0f + sycl::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
inline float gelu_erf_f(float x) { return 0.5f * x * (1.0f + sycl::erf(x * 0.7071067811865476f)); }

// rgb u8 [h][w][3] -> fp16 [N][768] = (C, 16, 16) flattened per patch, (v/255 - 0.5)/0.5, patches in raster order
inline void patchify(const uint8_t* rgb, half* pix, int w, int gw, int N) {
    Q().parallel_for(NR(N, 256), [=](sycl::nd_item<1> it) {
        const int pidx = (int)it.get_group(0);
        const int py = pidx / gw, px = pidx % gw;
        for (int e = (int)it.get_local_id(0); e < 768; e += 256) {
            const int c = e >> 8, dy = (e >> 4) & 15, dx = e & 15;
            const float v = (float)rgb[((size_t)(py * 16 + dy) * w + px * 16 + dx) * 3 + c] * (1.0f / 255.0f);
            pix[(size_t)pidx * 768 + e] = (half)((v - 0.5f) / 0.5f);
        }
    });
}
// learned position grid (row 0 = cls, rows 1.. = 128x128, fp16) resampled bilinearly to gh x gw (torch F.interpolate scale_factor (n+0.1)/128, align_corners=False)
inline void posemb(const half* tab, float* out, int gh, int gw, int edge, float sy, float sx, int D) {
    Q().parallel_for(NR((size_t)gh * gw, 256), [=](sycl::nd_item<1> it) {
        const int p = (int)it.get_group(0);
        const int y = p / gw, x = p % gw;
        float fy = sy * ((float)y + 0.5f) - 0.5f, fx = sx * ((float)x + 0.5f) - 0.5f;
        if (fy < 0.f) fy = 0.f;
        if (fx < 0.f) fx = 0.f;
        int y0 = (int)fy, x0 = (int)fx;
        const int y1 = y0 < edge - 1 ? y0 + 1 : y0, x1 = x0 < edge - 1 ? x0 + 1 : x0;
        if (y0 > edge - 1) y0 = edge - 1;
        if (x0 > edge - 1) x0 = edge - 1;
        const float ly = fy - (float)y0, lx = fx - (float)x0;
        const half* t = tab + (size_t)D;  // skip the cls row
        for (int c = (int)it.get_local_id(0); c < D; c += 256) {
            const float a = (float)t[((size_t)y0 * edge + x0) * D + c], b = (float)t[((size_t)y0 * edge + x1) * D + c];
            const float cc = (float)t[((size_t)y1 * edge + x0) * D + c], d = (float)t[((size_t)y1 * edge + x1) * D + c];
            out[(size_t)p * D + c] = (1.f - ly) * ((1.f - lx) * a + lx * b) + ly * ((1.f - lx) * cc + lx * d);
        }
    });
}
// LayerNorm (biased variance, affine): fp32 [N][D] -> fp16
inline void ln(const float* x, const float* w, const float* b, half* out, int D, float eps, int N) {
    Q().parallel_for(NR(N, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t r = it.get_group(0);
        const int tid = (int)it.get_local_id(0);
        const float* xr = x + r * D;
        float s = 0;
        for (int i = tid; i < D; i += 256) s += xr[i];
        const float mean = wg_sum(it, s, sh) / (float)D;
        float v = 0;
        for (int i = tid; i < D; i += 256) {
            const float d = xr[i] - mean;
            v += d * d;
        }
        const float inv = sycl::rsqrt(wg_sum(it, v, sh) / (float)D + eps);
        for (int i = tid; i < D; i += 256) out[r * D + i] = (half)((xr[i] - mean) * inv * w[i] + b[i]);
    });
}
// RMSNorm of the ViT output written into the 2x2-merge matrix A [(gh/2)*(gw/2)][D*4], column c*4 + dy*2 + dx
inline void rms_merge(const float* x, const float* w, half* A, int gw, int D, float eps, int N) {
    Q().parallel_for(NR(N, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int p = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int y = p / gw, xx = p % gw;
        const float* xr = x + (size_t)p * D;
        float ss = 0;
        for (int i = tid; i < D; i += 256) ss += xr[i] * xr[i];
        const float inv = sycl::rsqrt(wg_sum(it, ss, sh) / (float)D + eps);
        const size_t cell = (size_t)(y >> 1) * (gw >> 1) + (xx >> 1);
        const int sub = (y & 1) * 2 + (xx & 1);
        for (int i = tid; i < D; i += 256) A[cell * (size_t)(D * 4) + (size_t)i * 4 + sub] = (half)(xr[i] * inv * w[i]);
    });
}

// Y = X[M][K] (fp16, rows padded to a multiple of 64) * W[N][K]^T (fp16) + bias, XMX 8x8x16 (Alchemist) or portable SG32 fallback (Xe2/iGPU), K % 16 == 0.
template <int EPI>
inline void gemm(const half* X, const half* W, const float* bias, half* Yh, float* Yf, const float* pe, int M, int N, int K, int ow) {
#ifndef HY4S_DISABLE_SG8
    if (dev_caps().has_xmx_sg8) {
    namespace xm = sycl::ext::oneapi::experimental::matrix;
    const int nbm = (M + 63) / 64, nbn = (N + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sB(64 * 64, h);
        sycl::local_accessor<float, 1> sE(8 * 64, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 64, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, lane = tid & 7;
            const int m0 = bm * 64 + sgi * 8;
            const bool active = m0 < M;
            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[8];
            for (int j = 0; j < 8; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);
            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);
            const int n_row = bn * 64 + tid;
            for (int k0 = 0; k0 < K; k0 += 64) {
                const int kl = sycl::min(64, K - k0);  // multiple of 16
                if (n_row < N) {
                    const half* wp = W + (size_t)n_row * K + k0;
                    for (int kk = 0; kk < kl; kk += 2) {
                        sb_ptr[(kk >> 1) * 128 + tid * 2 + 0] = wp[kk];
                        sb_ptr[(kk >> 1) * 128 + tid * 2 + 1] = wp[kk + 1];
                    }
                } else {
                    for (int kk = 0; kk < kl; kk += 2) {
                        sb_ptr[(kk >> 1) * 128 + tid * 2 + 0] = (half)0.f;
                        sb_ptr[(kk >> 1) * 128 + tid * 2 + 1] = (half)0.f;
                    }
                }
                sycl::group_barrier(it.get_group());
                if (active) {
                    for (int s = 0; s < (kl >> 4); ++s) {
                        auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(const_cast<half*>(X + (size_t)m0 * K + k0 + s * 16));
                        xm::joint_matrix_load(sg, ma, xp, (size_t)K);
                        for (int j = 0; j < 8; ++j) {
                            xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + j * 16, (size_t)128);
                            xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            // epilogue: each accumulator tile goes through a per-sub-group local 8x8 so every lane can apply the element-wise epilogue with its (row, column)
            float* se = &sE[sgi * 64];
            auto se_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(se);
            for (int j = 0; j < 8; ++j) {
                if (active) xm::joint_matrix_store(sg, mc[j], se_mp, (size_t)8, xm::layout::row_major);
                sycl::group_barrier(sg);
                const int n = bn * 64 + j * 8 + lane;
                if (active && n < N) {
                    const float bv = bias ? bias[n] : 0.f;
                    for (int i = 0; i < 8; ++i) {
                        const int m = m0 + i;
                        if (m >= M) continue;
                        const float v = se[i * 8 + lane] + bv;
                        if constexpr (EPI == EPI_H) Yh[(size_t)m * N + n] = (half)v;
                        else if constexpr (EPI == EPI_GELU_TANH) Yh[(size_t)m * N + n] = (half)gelu_tanh_f(v);
                        else if constexpr (EPI == EPI_GELU_ERF) Yh[(size_t)m * N + n] = (half)gelu_erf_f(v);
                        else if constexpr (EPI == EPI_RES_ADD) Yf[(size_t)m * N + n] += v;
                        else if constexpr (EPI == EPI_RES_SET_POS) Yf[(size_t)m * N + n] = v + pe[(size_t)m * N + n];
                        else Yf[(size_t)(m + m / ow) * N + n] = v;  // EPI_F32_ROWMAP: merged cell m = oy*ow + ox -> row oy*(ow+1) + ox
                    }
                }
                sycl::group_barrier(sg);
            }
        });
    });
    return;
    }
#endif
    // Portable SG32 tiled GEMM fallback (Xe2 Battlemage / Core Ultra iGPU)
    const int nbm = (M + 31) / 32, nbn = (N + 31) / 32;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 33, h);
        sycl::local_accessor<float, 1> sB(32 * 33, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int gid = (int)it.get_group(0), bn = gid % nbn, bm = gid / nbn, tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4, tn = (tid & 15) * 2;
            float c[4][2] = {};
            for (int k0 = 0; k0 < K; k0 += 32) {
                for (int idx = tid; idx < 32 * 32; idx += 128) {
                    const int r = idx >> 5, k = idx & 31;
                    const int gm = bm * 32 + r, gn = bn * 32 + r;
                    sA[r * 33 + k] = (gm < M && k0 + k < K) ? (float)X[(size_t)gm * K + k0 + k] : 0.f;
                    sB[r * 33 + k] = (gn < N && k0 + k < K) ? (float)W[(size_t)gn * K + k0 + k] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                for (int k = 0; k < 32; ++k) {
                    const float a0 = sA[(tm + 0) * 33 + k], a1 = sA[(tm + 1) * 33 + k], a2 = sA[(tm + 2) * 33 + k], a3 = sA[(tm + 3) * 33 + k];
                    const float b0 = sB[(tn + 0) * 33 + k], b1 = sB[(tn + 1) * 33 + k];
                    c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[1][0] += a1 * b0; c[1][1] += a1 * b1;
                    c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[3][0] += a3 * b0; c[3][1] += a3 * b1;
                }
                sycl::group_barrier(it.get_group());
            }
            for (int r = 0; r < 4; ++r) {
                const int m = bm * 32 + tm + r;
                if (m >= M) continue;
                for (int j = 0; j < 2; ++j) {
                    const int n = bn * 32 + tn + j;
                    if (n >= N) continue;
                    const float v = c[r][j] + (bias ? bias[n] : 0.f);
                    if constexpr (EPI == EPI_H) Yh[(size_t)m * N + n] = (half)v;
                    else if constexpr (EPI == EPI_GELU_TANH) Yh[(size_t)m * N + n] = (half)gelu_tanh_f(v);
                    else if constexpr (EPI == EPI_GELU_ERF) Yh[(size_t)m * N + n] = (half)gelu_erf_f(v);
                    else if constexpr (EPI == EPI_RES_ADD) Yf[(size_t)m * N + n] += v;
                    else if constexpr (EPI == EPI_RES_SET_POS) Yf[(size_t)m * N + n] = v + pe[(size_t)m * N + n];
                    else Yf[(size_t)(m + m / ow) * N + n] = v;
                }
            }
        });
    });
}

// full (non-causal) softmax attention per head, head_dim 72, fp16 q/k/v [N][D] (head h at columns 72h..), online softmax in fp32, one work-item per query, 64 queries per group, key tiles of 16
inline void attention(const half* q, const half* k, const half* v, half* o, int N, int D, float scale) {
    const int NH = D / VA_HD, nbq = (N + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> Ks(16 * VA_HD, h), Vs(16 * VA_HD, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbq * NH * 64, 64), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int gid = (int)it.get_group(0), hh = gid / nbq, qb = gid % nbq, tid = (int)it.get_local_id(0);
            const int qi = qb * 64 + tid;
            float qv[VA_HD], acc[VA_HD];
            for (int d = 0; d < VA_HD; ++d) {
                qv[d] = qi < N ? (float)q[(size_t)qi * D + hh * VA_HD + d] * scale : 0.f;
                acc[d] = 0.f;
            }
            float mrun = -3.0e38f, lrun = 0.f;
            for (int kb = 0; kb < N; kb += 16) {
                sycl::group_barrier(it.get_group());
                for (int e = tid; e < 16 * VA_HD; e += 64) {
                    const int j = e / VA_HD, d = e % VA_HD, kj = kb + j;
                    Ks[e] = kj < N ? (float)k[(size_t)kj * D + hh * VA_HD + d] : 0.f;
                    Vs[e] = kj < N ? (float)v[(size_t)kj * D + hh * VA_HD + d] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                const int nj = sycl::min(16, N - kb);
                float s[16];
                float mt = -3.0e38f;
                for (int j = 0; j < 16; ++j) {
                    float dot = 0.f;
                    for (int d = 0; d < VA_HD; ++d) dot += qv[d] * Ks[j * VA_HD + d];
                    s[j] = j < nj ? dot : -3.0e38f;
                    mt = sycl::fmax(mt, s[j]);
                }
                const float mn = sycl::fmax(mrun, mt);
                const float corr = sycl::native::exp(mrun - mn);
                lrun *= corr;
                for (int d = 0; d < VA_HD; ++d) acc[d] *= corr;
                for (int j = 0; j < 16; ++j) {
                    const float p = j < nj ? sycl::native::exp(s[j] - mn) : 0.f;
                    lrun += p;
                    for (int d = 0; d < VA_HD; ++d) acc[d] += p * Vs[j * VA_HD + d];
                }
                mrun = mn;
            }
            if (qi < N) {
                const float inv = 1.0f / lrun;
                for (int d = 0; d < VA_HD; ++d) o[(size_t)qi * D + hh * VA_HD + d] = (half)(acc[d] * inv);
            }
        });
    });
}
inline void fill_newline(float* rows, const float* nl, int oh, int ow, int D) {
    Q().parallel_for(NR(oh, 256), [=](sycl::nd_item<1> it) {
        float* dst = rows + ((size_t)it.get_group(0) * (ow + 1) + ow) * D;
        for (int i = (int)it.get_local_id(0); i < D; i += 256) dst[i] = nl[i];
    });
}

// ---- LM side
// image rows -> the residual stream: x [T][4][3072] fp32 = the 4 iHC streams all start as the embedding
inline void scatter_rows(float* x, const float* rows, const int32_t* map, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0);
        const int r = map[t];
        if (r < 0) return;
        for (int d = (int)it.get_local_id(0); d < HID; d += 256) {
            const float v = rows[(size_t)r * HID + d];
            for (int h = 0; h < NHC; ++h) x[((size_t)t * NHC + h) * HID + d] = v;
        }
    });
}
struct RopeDev {
    const int32_t* tab;  // [3][stride]: (t, h, w) position of token index < tail_from
    int stride, tail_from, delta;
};
inline int rope_pos(const RopeDev& r, int idx, int axis) { return idx < r.tail_from ? r.tab[(size_t)axis * r.stride + idx] : idx + r.delta; }
inline void rope_pair_v(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    const float freq = 1.0f / sycl::pow(theta, (float)(2 * i) / 64.0f);
    const float ang = (float)pos * freq;
    const float sn = sycl::sin(ang), cs = sycl::cos(ang);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}
// MLA query: qa[t][h][512 + 2i ..] = rope(q_b[t][h][192 + 2i ..], mrope position of token p0 + t)
inline void fix_q(const float* q_b, half* qa, int M, int NH, int p0, float theta, RopeDev rd) {
    Q().parallel_for(NR((size_t)M * NH, 32), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0) / NH, h = (int)it.get_group(0) % NH, i = (int)it.get_local_id(0);
        const float* qp = q_b + ((size_t)t * NH + h) * 256 + 192 + 2 * i;
        float o0, o1;
        rope_pair_v(qp[0], qp[1], rope_pos(rd, p0 + t, hy4::mrope_axis(i)), i, theta, o0, o1);
        qa[((size_t)t * NH + h) * 576 + 512 + 2 * i] = (half)o0;
        qa[((size_t)t * NH + h) * 576 + 512 + 2 * i + 1] = (half)o1;
    });
}
// MLA key: cache[slot][512 + 2i ..] = rope(kv_a[t][512 + 2i ..], mrope position of token p0 + t); slot = (p0 + t) % ring, or p0 + t without a ring
inline void fix_k(const float* kv_a, half* cache, int M, int p0, int ring, float theta, RopeDev rd) {
    Q().parallel_for(NR(M, 32), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0), i = (int)it.get_local_id(0);
        const int pos = p0 + t;
        const int slot = ring > 0 ? pos % ring : pos;
        float o0, o1;
        rope_pair_v(kv_a[(size_t)t * 576 + 512 + 2 * i], kv_a[(size_t)t * 576 + 512 + 2 * i + 1], rope_pos(rd, pos, hy4::mrope_axis(i)), i, theta, o0, o1);
        cache[(size_t)slot * 576 + 512 + 2 * i] = (half)o0;
        cache[(size_t)slot * 576 + 512 + 2 * i + 1] = (half)o1;
    });
}
// indexer query: out[t][h][d] = iq[t][h][d], dims 64..127 roped at the mrope position of token p0 + t
inline void fix_iq(const float* iq, half* out, int T, int p0, float theta, RopeDev rd) {
    Q().parallel_for(NR((size_t)T * 16, 128), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0) >> 4, h = (int)it.get_group(0) & 15, d = (int)it.get_local_id(0);
        const float* q = iq + (size_t)t * 2048 + h * 128;
        float v = q[d];
        if (d >= 64) {
            const int i = (d - 64) >> 1;
            float o0, o1;
            rope_pair_v(q[64 + 2 * i], q[64 + 2 * i + 1], rope_pos(rd, p0 + t, hy4::mrope_axis(i)), i, theta, o0, o1);
            v = (d & 1) ? o1 : o0;
        }
        out[(size_t)t * 2048 + h * 128 + d] = (half)v;
    });
}
// pooled indexer keys were roped at 4g; rotate pair i of dims 64..127 by (mrope position of token 4g - 4g) * freq_i
inline void fix_pool(half* pooled, int g_lo, int n, float theta, RopeDev rd) {
    Q().parallel_for(NR(n, 32), [=](sycl::nd_item<1> it) {
        const int g = g_lo + (int)it.get_group(0), i = (int)it.get_local_id(0);
        const int pn = rope_pos(rd, 4 * g, hy4::mrope_axis(i));
        if (pn == 4 * g) return;
        const float freq = 1.0f / sycl::pow(theta, (float)(2 * i) / 64.0f);
        const float ang = (float)(pn - 4 * g) * freq;
        const float sn = sycl::sin(ang), cs = sycl::cos(ang);
        half* p = pooled + (size_t)g * 128 + 64 + 2 * i;
        const float x0 = (float)p[0], x1 = (float)p[1];
        p[0] = (half)(x0 * cs - x1 * sn);
        p[1] = (half)(x0 * sn + x1 * cs);
    });
}

}  // namespace vit

// =============================================================================================================================================================================
struct VisionEncoder {
    hy4::VisionShardInfo vi;
    uint8_t* h_w = nullptr;  // Section 1 of the shard in pinned host memory
    uint8_t* arena = nullptr;
    size_t arena_cap = 0, arena_pos = 0;
    bool loaded = false;
    double t_load = 0;
    bool gelu_tanh_ffn = true;
    const std::atomic<bool>* cancel = nullptr;  // set by serve: STOP during the encode ends it at the next layer
    std::string dump_dir;  // debug: residual stream after the embedding and after every layer (fp32 [N][1152] .npy)
    void dump_x(const std::string& name, const float* dx, int N, int D) {
        wait();
        std::vector<float> hh((size_t)N * D);
        d2h(hh.data(), dx, hh.size() * 4);
        wait();
        std::string sh = "(" + std::to_string(N) + "," + std::to_string(D) + ")";
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
        const size_t tot = 10 + hdr.size() + 1, pad = (64 - tot % 64) % 64;
        hdr += std::string(pad, ' ') + "\n";
        FILE* f = fopen((dump_dir + "/" + name + ".npy").c_str(), "wb");
        if (!f) return;
        fwrite("\x93NUMPY", 1, 6, f);
        const unsigned char ver[2] = {1, 0};
        fwrite(ver, 1, 2, f);
        const uint16_t hl = (uint16_t)hdr.size();
        fwrite(&hl, 2, 1, f);
        fwrite(hdr.data(), 1, hdr.size(), f);
        fwrite(hh.data(), 4, hh.size(), f);
        fclose(f);
    }

    bool load(const std::string& path, std::string& err) {
        if (loaded) return true;
        if (!hy4::vision_probe(path, vi, err)) return false;
        gelu_tanh_ffn = vi.c.gelu_tanh_ffn;
        if (const char* g = std::getenv("HY4_VIT_GELU")) gelu_tanh_ffn = std::string(g) != "erf";
        const double t0 = vit::tnow();
        h_w = halloc<uint8_t>(vi.sec1_len + 4096);
        hy4::AlignedFile af;
        if (!af.open(path, false)) { err = "cannot open the vision shard"; return false; }
        if (af.read_at(vi.sec1_off, h_w, vi.sec1_len) < vi.sec1_len) { err = "short read of the vision shard"; af.close(); return false; }
        af.close();
        arena_cap = 96ull << 20;
        arena = dalloc<uint8_t>(arena_cap);
        loaded = true;
        t_load = vit::tnow() - t0;
        fprintf(stderr, "[vision] shard in pinned host memory: %.2f GB in %.2f s (device arena %.0f MiB)\n", vi.sec1_len / 1e9, t_load, arena_cap / 1048576.0);
        return true;
    }
    void arena_reset() { arena_pos = 0; }
    const uint8_t* dev(const std::string& n) {
        const hy4::VitTensor& t = vi.get(n);
        const size_t at = (arena_pos + 255) & ~(size_t)255;
        if (at + t.bytes > arena_cap) {
            fprintf(stderr, "[vision] device arena too small for %s\n", n.c_str());
            exit(3);
        }
        Q().memcpy(arena + at, h_w + t.off, t.bytes);
        arena_pos = at + t.bytes;
        return arena + at;
    }
    const half* wh(const std::string& n) { return reinterpret_cast<const half*>(dev(n)); }
    const float* wf(const std::string& n) { return reinterpret_cast<const float*>(dev(n)); }

    void gemm(int epi, const half* X, const half* W, const float* bias, half* Yh, float* Yf, const float* pe, int M, int N, int K, int ow = 1) {
        switch (epi) {
            case vit::EPI_H: vit::gemm<vit::EPI_H>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_GELU_TANH: vit::gemm<vit::EPI_GELU_TANH>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_GELU_ERF: vit::gemm<vit::EPI_GELU_ERF>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_RES_ADD: vit::gemm<vit::EPI_RES_ADD>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_RES_SET_POS: vit::gemm<vit::EPI_RES_SET_POS>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            default: vit::gemm<vit::EPI_F32_ROWMAP>(X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
        }
    }

    // rgb: host RGB u8 [h][w][3] (h, w multiples of 32). rows_out: device fp32 [(h/32)*(w/32+1)][out_hidden]. Allocates and frees its own workspace.
    bool encode(const uint8_t* rgb, int h, int w, float* rows_out, std::string& err) {
        if (!loaded) {
            err = "vision shard not loaded";
            return false;
        }
        const hy4::VitCfg& c = vi.c;
        const int P = c.patch, D = c.hidden, NH = c.heads, FF = c.ffn, OD = c.out_hidden;
        if (h % (P * c.merge) || w % (P * c.merge) || h < 32 || w < 32) {
            err = "picture size must be a multiple of 32";
            return false;
        }
        const int gh = h / P, gw = w / P, N = gh * gw, oh = gh / 2, ow = gw / 2, M2 = oh * ow;
        if (D / NH != vit::VA_HD) {
            err = "unsupported head dim";
            return false;
        }
        const int Np = (N + 63) & ~63, M2p = (M2 + 63) & ~63;  // rows padded to the GEMM tile
        uint8_t* d_rgb = nullptr;
        half *d_pix = nullptr, *d_h = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_ff = nullptr, *d_A = nullptr, *d_mid = nullptr;
        float *d_x = nullptr, *d_pe = nullptr;
        const size_t bN = (size_t)Np * D;
        auto free_ws = [&]() {
            wait();
            for (void* q : {(void*)d_rgb, (void*)d_pix, (void*)d_x, (void*)d_pe, (void*)d_h, (void*)d_q, (void*)d_k, (void*)d_v, (void*)d_ff, (void*)d_A, (void*)d_mid})
                if (q) dfree(q);
        };
        try {
            d_rgb = dalloc<uint8_t>((size_t)h * w * 3);
            d_pix = dalloc<half>((size_t)Np * 768);
            d_x = dalloc<float>(bN);
            d_pe = dalloc<float>(bN);
            d_h = dalloc<half>(bN);
            d_q = dalloc<half>(bN);
            d_k = dalloc<half>(bN);
            d_v = dalloc<half>(bN);
            d_ff = dalloc<half>((size_t)Np * FF);
            d_A = dalloc<half>((size_t)M2p * D * 4);
            d_mid = dalloc<half>((size_t)M2p * D * 4);
        } catch (...) {
            free_ws();
            err = "out of device memory for the vision workspace";
            return false;
        }
        dzero(d_pix, (size_t)Np * 768 * 2);
        dzero(d_h, bN * 2);
        dzero(d_A, (size_t)M2p * D * 4 * 2);
        dzero(d_mid, (size_t)M2p * D * 4 * 2);
        h2d(d_rgb, rgb, (size_t)h * w * 3);
        vit::patchify(d_rgb, d_pix, w, gw, N);
        arena_reset();
        const float sy = (float)(1.0 / (((double)gh + 0.1) / (double)c.pos_edge)), sx = (float)(1.0 / (((double)gw + 0.1) / (double)c.pos_edge));
        vit::posemb(wh("vit.embeddings.position_embedding.weight"), d_pe, gh, gw, c.pos_edge, sy, sx, D);
        arena_reset();  // (the position table is consumed by the kernel above before the next copy: in-order queue)
        gemm(vit::EPI_RES_SET_POS, d_pix, wh("vit.embeddings.patch_embedding.weight"), wf("vit.embeddings.patch_embedding.bias"), nullptr, d_x, d_pe, N, D, 768);
        if (!dump_dir.empty()) dump_x("emb0", d_x, N, D);
        const float scale = 1.0f / std::sqrt((float)(D / NH));
        for (int il = 0; il < c.layers; ++il) {
            if (cancel && cancel->load()) {
                err = "cancelled";
                free_ws();
                return false;
            }
            arena_reset();
            const std::string p = "vit.layers." + std::to_string(il) + ".";
            vit::ln(d_x, wf(p + "input_layernorm.weight"), wf(p + "input_layernorm.bias"), d_h, D, c.eps, N);
            gemm(vit::EPI_H, d_h, wh(p + "self_attn.q_proj.weight"), wf(p + "self_attn.q_proj.bias"), d_q, nullptr, nullptr, N, D, D);
            gemm(vit::EPI_H, d_h, wh(p + "self_attn.k_proj.weight"), wf(p + "self_attn.k_proj.bias"), d_k, nullptr, nullptr, N, D, D);
            gemm(vit::EPI_H, d_h, wh(p + "self_attn.v_proj.weight"), wf(p + "self_attn.v_proj.bias"), d_v, nullptr, nullptr, N, D, D);
            vit::attention(d_q, d_k, d_v, d_h, N, D, scale);
            gemm(vit::EPI_RES_ADD, d_h, wh(p + "self_attn.o_proj.weight"), wf(p + "self_attn.o_proj.bias"), nullptr, d_x, nullptr, N, D, D);
            vit::ln(d_x, wf(p + "post_attention_layernorm.weight"), wf(p + "post_attention_layernorm.bias"), d_h, D, c.eps, N);
            gemm(gelu_tanh_ffn ? vit::EPI_GELU_TANH : vit::EPI_GELU_ERF, d_h, wh(p + "mlp.dense_h_to_4h.weight"), wf(p + "mlp.dense_h_to_4h.bias"), d_ff, nullptr, nullptr, N, FF, D);
            gemm(vit::EPI_RES_ADD, d_ff, wh(p + "mlp.dense_4h_to_h.weight"), wf(p + "mlp.dense_4h_to_h.bias"), nullptr, d_x, nullptr, N, D, FF);
            if (!dump_dir.empty()) dump_x("L" + std::to_string(il), d_x, N, D);
        }
        // adapter
        arena_reset();
        vit::rms_merge(d_x, wf("vit.perceive.norm.weight"), d_A, gw, D, c.eps, N);
        gemm(vit::EPI_GELU_ERF, d_A, wh("vit.perceive.proj.0.weight"), wf("vit.perceive.proj.0.bias"), d_mid, nullptr, nullptr, M2, D * 4, D * 4);
        arena_reset();
        gemm(vit::EPI_F32_ROWMAP, d_mid, wh("vit.perceive.proj.2.weight"), wf("vit.perceive.proj.2.bias"), nullptr, rows_out, nullptr, M2, OD, D * 4, ow);
        vit::fill_newline(rows_out, wf("vit.perceive.image_newline"), oh, ow, OD);
        free_ws();
        return true;
    }
};

// Per-request picture state: image rows on the device, the spans of the prompt they fill, the (t, h, w) position table. Owned by serve / main, pointed to by Runner::vis.
struct VisionCtx {
    std::vector<hy4::ImageSpan> spans;
    hy4::RopePos pos;
    std::vector<uint8_t> mask;  // per token index of the prompt: 1 where the token is an image row (Engram contribution is off there)
    float* d_rows = nullptr;    // [R][3072] fp32
    int n_rows = 0;
    int32_t* d_tab = nullptr;   // [3][tab_n]
    int tab_n = 0;
    int32_t* d_map = nullptr;
    int map_cap = 0;
    double t_encode = 0;
    int image_token = 120120;

    ~VisionCtx() { release(); }
    void release() {
        if (d_rows) dfree(d_rows);
        if (d_tab) dfree(d_tab);
        if (d_map) dfree(d_map);
        d_rows = nullptr;
        d_tab = nullptr;
        d_map = nullptr;
        n_rows = tab_n = map_cap = 0;
        spans.clear();
        mask.clear();
        pos = hy4::RopePos();
    }
    bool active() const { return !spans.empty(); }

    // ids = the whole (expanded) prompt, recs = its pictures in order. Encodes every picture (enc must be loaded) into d_rows.
    bool setup(VisionEncoder& enc, const std::vector<int32_t>& ids, const std::vector<hy4::ImageRec>& recs, int max_chunk, std::string& err) {
        release();
        image_token = enc.vi.c.image_token;
        if (!hy4::build_image_spans(ids, image_token, recs, spans, pos, err)) return false;
        int total = 0;
        for (const hy4::ImageSpan& s : spans) total += s.n;
        n_rows = total;
        d_rows = dalloc<float>((size_t)total * HID);
        const double t0 = vit::tnow();
        for (size_t k = 0; k < recs.size(); ++k) {
            if (!enc.encode(recs[k].rgb.data(), recs[k].h, recs[k].w, d_rows + (size_t)spans[k].row0 * HID, err)) return false;
        }
        t_encode = vit::tnow() - t0;
        mask.assign(ids.size(), 0);
        for (const hy4::ImageSpan& s : spans)
            for (int j = 0; j < s.n; ++j) mask[s.start + j] = 1;
        tab_n = (int)ids.size();
        d_tab = dalloc<int32_t>((size_t)3 * tab_n);
        for (int a = 0; a < 3; ++a) h2d(d_tab + (size_t)a * tab_n, pos.p[a].data(), (size_t)tab_n * 4);
        map_cap = max_chunk;
        d_map = dalloc<int32_t>((size_t)map_cap);
        wait();
        return true;
    }

    vit::RopeDev rd() const { return vit::RopeDev{d_tab, tab_n, tab_n, pos.delta}; }

    // hook 1: after the text embedding of chunk [p0, p0 + T): overwrite the image positions with the image rows
    void embed(float* x, int p0, int T) {
        if (!active()) return;
        std::vector<int32_t> map(T, -1);
        bool any = false;
        for (const hy4::ImageSpan& s : spans) {
            const int a = std::max(p0, s.start), b = std::min(p0 + T, s.start + s.n);
            for (int i = a; i < b; ++i) {
                map[i - p0] = s.row0 + (i - s.start);
                any = true;
            }
        }
        if (!any) return;
        h2d(d_map, map.data(), (size_t)T * 4);
        vit::scatter_rows(x, d_rows, d_map, T);
    }
    // hook 2: positions of the chunk are not index + 0 any more (a picture is in the context): redo the rope of q (qa rope part) and of the new keys with the 3D positions
    void fixq(const float* q_b, half* qa, int M, int NH, int p0, float theta) { vit::fix_q(q_b, qa, M, NH, p0, theta, rd()); }
    void fixk(const float* kv_a, half* cache, int M, int p0, int ring, float theta) { vit::fix_k(kv_a, cache, M, p0, ring, theta, rd()); }
    void iq_rope(const float* iq, half* out, int T, int p0, float theta) { vit::fix_iq(iq, out, T, p0, theta, rd()); }
    void pool_fix(half* pooled, int g_lo, int n, float theta) { vit::fix_pool(pooled, g_lo, n, theta, rd()); }
    // Engram: the contribution is masked at image positions
    void mask_engram(uint8_t* valid /*[T][16]*/, int p0, int T) const {
        for (int t = 0; t < T; ++t)
            if (p0 + t < (int)mask.size() && mask[p0 + t]) std::memset(valid + (size_t)t * 16, 0, 16);
    }
    bool needs_fix(int p0, int T) const { return active() && (pos.delta != 0 || p0 < tab_n); }
};

}  // namespace hy4s
