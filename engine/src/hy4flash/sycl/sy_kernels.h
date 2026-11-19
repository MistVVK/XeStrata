// sy_kernels.h - elementwise / norm / iHC / router / engram / embed / head kernels of the Hy4-Flash graph, SYCL (Arc).
// Port of run/hy4_kernels.hip.h. Residual stream x is fp32 [T][4][3072]. GEMM inputs are fp16, GEMM outputs fp32.
// Every function enqueues on Q() and returns at once (no sync, no allocation).
#pragma once
#include "sy_common.h"

namespace hy4s {

// ---- embedding: S8_K row -> 4 identical hc streams (fp32). W: S8_K [V][3072] raw, row = 12 super-blocks of 262 B.
inline void embed_hc(const uint8_t* W, const int32_t* ids, float* x, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int t = (int)it.get_group(0);
        const uint8_t* row = W + (size_t)ids[t] * 12 * 262;
        for (int d = (int)it.get_local_id(0); d < HID; d += 256) {
            const int sb = d >> 8, i = d & 255, g = i >> 6;
            const uint8_t* blk = row + sb * 262;
            const float D = h2f_bits(ld16(blk));
            const float eff = D * ((float)blk[2 + g] / 255.0f);
            const float v = (float)(int8_t)blk[6 + i] * eff;
            for (int h = 0; h < NHC; ++h) x[((size_t)t * NHC + h) * HID + d] = v;
        }
    });
}

// ---- RMSNorm of `rows` rows of width D: in fp32 -> out fp32 (outf) and/or fp16 (outh); w may be null. block = 32*nwarp threads (D <= block*k)
inline void rms_rows(const float* in, const float* w, float* outf, half* outh, int D, float eps, int rows, int block = 256) {
    Q().parallel_for(NR(rows, block), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t r = it.get_group(0);
        const float* x = in + r * D;
        const int tid = (int)it.get_local_id(0);
        float ss = 0;
        for (int i = tid; i < D; i += block) ss += x[i] * x[i];
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / D + eps);
        for (int i = tid; i < D; i += block) {
            const float v = x[i] * inv * (w ? w[i] : 1.0f);
            if (outf) outf[r * D + i] = v;
            if (outh) outh[r * D + i] = (half)v;
        }
    });
}

// ---- iHC mixes: per token, rms of the 12288 flat state, nout dot products with hc_fn [nout][12288] -> pre[4], post[4] (layers: nout 8; head: nout 4, pre only)
inline void hc_mix(const float* x, const float* fn, const float* scale, const float* base, float* pre, float* post, int nout, float eps, float hc_eps, float hc_mag, int T) {
    Q().parallel_for(NR(T, 512), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* xt = x + (size_t)t * NHC * HID;
        constexpr int K = NHC * HID;
        float ss = 0, acc[8];
        for (int j = 0; j < 8; ++j) acc[j] = 0;
        for (int k = tid; k < K; k += 512) {
            const float v = xt[k];
            ss += v * v;
            for (int j = 0; j < 8; ++j)
                if (j < nout) acc[j] += v * fn[(size_t)j * K + k];
        }
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / K + eps);
        float m[8];
        for (int j = 0; j < 8; ++j) {
            m[j] = 0;
            if (j < nout) m[j] = wg_sum(it, acc[j], sh) * inv;
        }
        if (tid == 0) {
            for (int h = 0; h < NHC; ++h) {
                pre[(size_t)t * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
                if (nout == 8) post[(size_t)t * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
            }
        }
    });
}

// ---- cur = sum_h x[h]*pre[h]; then rmsnorm(cur, w) -> fp16 (GEMM input) and optionally fp32
inline void hc_reduce_norm(const float* x, const float* pre, const float* w, half* outh, float* outf, float eps, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t t = it.get_group(0);
        const int tid = (int)it.get_local_id(0);
        const float* xt = x + t * NHC * HID;
        const float* p = pre + t * NHC;
        const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
        constexpr int PER = HID / 256;
        float v[PER];
        float ss = 0;
        for (int i = 0; i < PER; ++i) {
            const int d = tid + i * 256;
            const float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
            v[i] = c;
            ss += c * c;
        }
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / HID + eps);
        for (int i = 0; i < PER; ++i) {
            const int d = tid + i * 256;
            const float o = v[i] * inv * (w ? w[d] : 1.0f);
            if (outh) outh[t * HID + d] = (half)o;
            if (outf) outf[t * HID + d] = o;
        }
    });
}

// ---- x[h] += sub * post[h]   (sub fp32 [T][3072])
inline void hc_post(float* x, const float* sub, const float* post, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const size_t t = it.get_group(0);
        const float* s = sub + t * HID;
        const float* p = post + t * NHC;
        float* xt = x + t * NHC * HID;
        for (int d = (int)it.get_local_id(0); d < HID; d += 256) {
            const float sv = s[d];
            for (int h = 0; h < NHC; ++h) xt[h * HID + d] += sv * p[h];
        }
    });
}

// ---- out = rmsnorm(a (+ b), w)  fp32 (b optional), D = 3072
inline void add_rms(const float* a, const float* b, const float* w, float* out, int D, float eps, int rows) {
    Q().parallel_for(NR(rows, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t o = it.get_group(0) * (size_t)D;
        const int tid = (int)it.get_local_id(0);
        float ss = 0;
        for (int i = tid; i < D; i += 256) {
            const float v = a[o + i] + (b ? b[o + i] : 0.0f);
            ss += v * v;
        }
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / D + eps);
        for (int i = tid; i < D; i += 256) {
            const float v = a[o + i] + (b ? b[o + i] : 0.0f);
            out[o + i] = v * inv * w[i];
        }
    });
}

inline void f2h(const float* in, half* out, size_t n) {
    Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) out[i] = (half)in[i];
    });
}

// ---- SwiGLU: h = silu(g) * u, fp32 in -> fp16 out
inline void swiglu_h(const float* g, const float* u, half* out, size_t n) {
    Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) out[i] = (half)(siluf_(g[i]) * u[i]);
    });
}

// ---- router: logits fp32 [T][512] -> top-16 by sigmoid+bias; weights = p/(sum p + 1e-20)*scale. One group (256 thr) per token.
inline void router_topk(const float* logits, const float* bias, int32_t* idx, float* w, int E, int K, float scale, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sp = lmem<float, 512>(it);
        float* ss = lmem<float, 512>(it);
        float* rv = lmem<float, 8>(it);
        int* ri = lmem<int, 8>(it);
        int* chosen = lmem<int, 16>(it);
        const size_t t = it.get_group(0);
        const int tid = (int)it.get_local_id(0);
        const float* lg = logits + t * E;
        for (int e = tid; e < E; e += 256) {
            const float p = sigmoidf_(lg[e]);
            sp[e] = p;
            ss[e] = p + bias[e];
        }
        sycl::group_barrier(it.get_group());
        for (int k = 0; k < K; ++k) {
            float bv = -3.0e38f;
            int bi = 0x7fffffff;
            for (int e = tid; e < E; e += 256)
                if (ss[e] > bv || (ss[e] == bv && e < bi)) { bv = ss[e]; bi = e; }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = shfl_xor(it, bv, o);
                const int oi = shfl_xor(it, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            const int wv = tid >> 5;
            if ((tid & 31) == 0) { rv[wv] = bv; ri[wv] = bi; }
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                float b2 = rv[0];
                int i2 = ri[0];
                for (int q = 1; q < 8; ++q)
                    if (rv[q] > b2 || (rv[q] == b2 && ri[q] < i2)) { b2 = rv[q]; i2 = ri[q]; }
                chosen[k] = i2;
                ss[i2] = -3.0e38f;
            }
            sycl::group_barrier(it.get_group());
        }
        if (tid == 0) {
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += sp[chosen[k]];
            for (int k = 0; k < K; ++k) {
                idx[t * K + k] = chosen[k];
                w[t * K + k] = sp[chosen[k]] / (sum + 1e-20f) * scale;
            }
        }
    });
}

// ---- argmax over logits [V] (one row) -> token id; 1 group of 1024
inline void argmax_row(const float* logits, int V, int32_t* out) {
    Q().parallel_for(NR(1, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sv = lmem<float, 32>(it);
        int* si = lmem<int, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float bv = -3.0e38f;
        int bi = 0;
        for (int i = tid; i < V; i += 1024)
            if (logits[i] > bv) { bv = logits[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(it, bv, o);
            const int oi = shfl_xor(it, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            float b2 = sv[0];
            int i2 = si[0];
            for (int q = 1; q < 32; ++q)
                if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
            *out = i2;
        }
    });
}

// ---- Engram: raw rows [T][16][100 B] (f32 scale + 96 int8) -> normed rows r[T][1536] fp16: r[t][c*96+i] = rmsnorm96(row) * lookup_norm[c][i] * valid[t][c]
inline void engram_rows(const uint8_t* raw, const float* lnw /*[16][96]*/, const uint8_t* valid /*[T][16]*/, half* r, int T) {
    Q().parallel_for(NR(T, 512), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int t = (int)it.get_group(0), c = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const uint8_t* row = raw + ((size_t)t * 16 + c) * 100;
        float scale;
        std::memcpy(&scale, row, 4);
        float v[3], ss = 0;
        for (int j = 0; j < 3; ++j) {
            v[j] = (float)(int8_t)row[4 + lane + 32 * j] * scale;
            ss += v[j] * v[j];
        }
        ss = sg_sum(it, ss);
        const float inv = sycl::rsqrt(ss / 96.0f + 1.1920929e-7f);
        const float vm = valid[(size_t)t * 16 + c] ? 1.0f : 0.0f;
        for (int j = 0; j < 3; ++j) r[(size_t)t * 1536 + c * 96 + lane + 32 * j] = (half)(v[j] * inv * lnw[c * 96 + lane + 32 * j] * vm);
    });
}

// ---- Engram apply: kv [5][T][3072] fp32 (without bias), x [T][4][3072]. group per (token, stream j) = t*4 + j.
inline void engram_apply(float* x, const float* kv, const float* kvb /*[5][3072]*/, const float* egate /*[4][3072]*/, const float* pre_gamma, const float* hcs /*[4][3072]*/,
                         const uint8_t* valid, int T) {
    Q().parallel_for(NR((size_t)T * 4, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int g = (int)it.get_group(0), t = g >> 2, j = g & 3, tid = (int)it.get_local_id(0);
        const float EPS = 1.1920929e-7f;
        float* xs = x + ((size_t)t * NHC + j) * HID;
        const float* key = kv + ((size_t)j * T + t) * HID;
        const float* val = kv + ((size_t)4 * T + t) * HID;
        float sx = 0, sk = 0, sv = 0;
        for (int d = tid; d < HID; d += 256) {
            const float a = xs[d], k = key[d] + kvb[j * HID + d], v = val[d] + kvb[4 * HID + d];
            sx += a * a; sk += k * k; sv += v * v;
        }
        sx = wg_sum(it, sx, sh);
        sk = wg_sum(it, sk, sh);
        sv = wg_sum(it, sv, sh);
        const float xi = sycl::rsqrt(sx / HID + EPS), ki = sycl::rsqrt(sk / HID + EPS), vi = sycl::rsqrt(sv / HID + EPS);
        float dot = 0;
        for (int d = tid; d < HID; d += 256) {
            const float a = xs[d], k = key[d] + kvb[j * HID + d];
            dot += (a * xi) * (k * ki) * egate[j * HID + d];
        }
        dot = wg_sum(it, dot, sh) / sycl::sqrt((float)HID);
        const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
        const float gate = sigmoidf_(dot < 0 ? -mag : mag);
        int anyv = 0;
        for (int c = 0; c < 16; ++c) anyv |= valid[(size_t)t * 16 + c] ? 1 : 0;
        const float anyf = anyv ? 1.0f : 0.0f;
        for (int d = tid; d < HID; d += 256) {
            const float v = val[d] + kvb[4 * HID + d];
            xs[d] += (v * vi * pre_gamma[d]) * (gate + hcs[j * HID + d]) * anyf;
        }
    });
}

}  // namespace hy4s
