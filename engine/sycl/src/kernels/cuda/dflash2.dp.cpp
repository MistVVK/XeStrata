// sycl/src/kernels/cuda/dflash2.dp.cpp - the small kernels of the DFlash2 drafter on SYCL (Intel Arc).  Hand port of src/kernels/cuda/dflash2.cu for the
// Q4_0 drafter pack; the interface is include/strata/kernels/dflash2.hpp (a `stream` is a sycl::queue*, null = the default in-order queue).
//
// Differences from the CUDA file, all on purpose:
//   * Q4_0 pack only (df2_gemm throws for a Q8 matrix, df2_q8_tile is absent).
//   * DF-03, active columns: the GEMM is instantiated per column count NC = 1..8 (acc[R][NC]); the arithmetic of a column does not depend on NC (same steps,
//     same fma order, same shuffle tree), so the first n columns of an 8-column launch are bitwise the n-column launch.  Alchemist has 128 GRFs a thread:
//     at 32 lanes a float is 4 GRFs, so acc[8][8] cannot live in registers; fewer columns is the difference between spilling and not.
//   * conv_rms_quant takes 256 threads (a work-group is at most 512 on Alchemist), the sum of the squares reduces over 8 sub-group partials, not 32.
//   * attention: no register staging of the next K/V tile (loads go to local memory directly).
//   * math: sycl::exp / rsqrt / rint (precise) where CUDA uses __expf / rsqrtf.
// Everything is asynchronous on the queue, allocates nothing and can be recorded into a sycl graph.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HD = kDf2Hd;
#ifndef DF2_ATTN_CHUNK_AUTO
#define DF2_ATTN_CHUNK_AUTO 64
#endif
#ifndef DF2_ATTN_AUTO
#define DF2_ATTN_AUTO 2
#endif
int g_gemm_variant = 0;   // bench only (df2_gemm_variant)
int g_attn_chunk = 0;     // bench only (df2_attention_chunk)
int g_attn_variant = 0;   // 0: STRATA_DF2_ATTN or the default; 1: the CUDA file kernel (128 work-items); 2: 256 work-items, O[2][8]

#define DF2_SG32 [[sycl::reqd_sub_group_size(32)]]
#define DF2_SG16 [[sycl::reqd_sub_group_size(16)]]

using sycl::nd_item;
inline float bf16f(uint16_t h) { return sycl::bit_cast<float>((uint32_t) h << 16); }
inline float xorf(const sycl::sub_group& sg, float v, int o) { return sycl::permute_group_by_xor(sg, v, (unsigned) o); }
inline int xori(const sycl::sub_group& sg, int v, int o) { return sycl::permute_group_by_xor(sg, v, (unsigned) o); }
inline void bar(const nd_item<1>& it) { sycl::group_barrier(it.get_group()); }
inline void bar(const nd_item<2>& it) { sycl::group_barrier(it.get_group()); }

// one 32-block of one row to the planar activation layout (a sub-group: lane = element)
inline void quant_block(const sycl::sub_group& sg, float v, int blk, int col, int K, int8_t* xq, float* xd) {
    const int lane = (int) sg.get_local_linear_id();
    float amax = sycl::fabs(v);
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, xorf(sg, amax, o));
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q = (int) sycl::rint(v * inv);
    xq[(size_t) col * K + blk * 32 + lane] = (int8_t) q;
    if (lane == 0) xd[(size_t) col * (K / 32) + blk] = d;
}

// ---- the grouped dynamic 2-tap convolution over the block's rows
void conv_launch(sycl::queue* q, const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out, int rows, int H, int group) {
    const int n = rows * H;
    q->parallel_for(sycl::nd_range<1>((size_t) ((n + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= n) return;
        const int b = i / H, c = i % H;
        const int groups = H / group;
        float acc = 0.0f;
        #pragma unroll
        for (int o = 0; o < 2; ++o) {
            if (b - o < 0) continue;
            const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
            acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
        }
        out[i] = resid ? resid[i] + acc : acc;
    });
}

inline void rope_cs(double inv, int32_t pos, float& c, float& s) {
    double ang = (double) pos * inv;
    const double two_pi = 6.283185307179586476925286766559;
    ang -= two_pi * sycl::rint(ang / two_pi);   // [-pi, pi]
    s = sycl::sin((float) ang);
    c = sycl::cos((float) ang);
}

// ---- norm + rope of one (row, head), 128 work-items: out = rope(rmsnorm(in) * gamma)
inline float head_norm(const nd_item<2>& it, const sycl::local_accessor<float, 1>& red, float x, float eps) {
    const sycl::sub_group sg = it.get_sub_group();
    const int d = (int) it.get_local_id(1);
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    bar(it);
    const float tot = red[0] + red[1] + red[2] + red[3];
    return tot;
}

}  // namespace

void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out, int rows, int H, int group, void* stream) {
    conv_launch(strata::q_of(stream), x, dyn, dyn_stride, base, resid, out, rows, H, group);
}

void df2_rope_inv(float theta, double* inv, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>(64, 64), [=](nd_item<1> it) {
        const int i = (int) it.get_local_id(0);
        inv[i] = sycl::pow((double) theta, -(double) (2 * i) / (double) HD);
    });
}

void df2_rope_table(const int32_t* pos, int rows, float theta, float* table, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>((size_t) rows * 64, 64), [=](nd_item<1> it) {
        const int r = (int) it.get_group(0), i = (int) it.get_local_id(0);
        const double inv = sycl::pow((double) theta, -(double) (2 * i) / (double) HD);
        float c, s;
        rope_cs(inv, pos[r], c, s);
        table[(r * 2 + 0) * 64 + i] = c;
        table[(r * 2 + 1) * 64 + i] = s;
    });
}

void df2_rope_table_inv(const int32_t* pos, int rows, const double* inv_tab, float* table, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>((size_t) rows * 64, 64), [=](nd_item<1> it) {
        const int r = (int) it.get_group(0), i = (int) it.get_local_id(0);
        float c, s;
        rope_cs(inv_tab[i], pos[r], c, s);
        table[(r * 2 + 0) * 64 + i] = c;
        table[(r * 2 + 1) * 64 + i] = s;
    });
}

void df2_rope_table_cpu(const int32_t* pos, int rows, float theta, float* table) {
    const double two_pi = 6.283185307179586476925286766559;
    for (int i = 0; i < 64; ++i) {
        const double inv = std::pow((double) theta, -(double) (2 * i) / (double) HD);
        for (int r = 0; r < rows; ++r) {
            double ang = (double) pos[r] * inv;
            ang -= two_pi * std::nearbyint(ang / two_pi);   // [-pi, pi]
            table[(r * 2 + 0) * 64 + i] = (float) std::cos((float) ang);
            table[(r * 2 + 1) * 64 + i] = (float) std::sin((float) ang);
        }
    }
}

void df2_norm_rope(const float* in, int in_stride, const float* gamma, const int32_t* pos, const float* rope_table, float* out, int slot_mod, int rows, int heads,
                   float eps, void* stream) {
    sycl::queue* q = strata::q_of(stream);
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(4, h), y(HD, h);
        h.parallel_for(sycl::nd_range<2>({(size_t) heads, (size_t) rows * HD}, {1, HD}), [=](nd_item<2> it) DF2_SG32 {
            const int hh = (int) it.get_group(0), r = (int) it.get_group(1), d = (int) it.get_local_id(1);
            const float x = in[(size_t) r * in_stride + hh * HD + d];
            const float tot = head_norm(it, red, x, eps);
            const float v = x * sycl::rsqrt(tot / (float) HD + eps) * gamma[d];
            float res = v;
            if (rope_table != nullptr) {
                y[d] = v;
                bar(it);
                const int i = d & 63;
                const float c = rope_table[(r * 2 + 0) * 64 + i], s = rope_table[(r * 2 + 1) * 64 + i];
                res = d < 64 ? v * c - y[d + 64] * s : v * c + y[d - 64] * s;
            }
            const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
            out[slot_mod > 0 ? ((size_t) hh * slot_mod + slot) * HD + d : ((size_t) slot * heads + hh) * HD + d] = res;
        });
    });
}

void df2_copy_rows(const float* in, float* out, const int32_t* pos, int slot_mod, int rows, int cols, void* stream) {
    const int n = rows * cols;
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>((size_t) ((n + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= n) return;
        const int r = i / cols, c = i % cols;
        const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
        out[slot_mod > 0 ? ((size_t) (c / HD) * slot_mod + slot) * HD + c % HD : (size_t) slot * cols + c] = in[i];
    });
}

// ---- attention.  grid (8 K/V heads, kDf2MaxChunks), 128 work-items.  The 32 queries (4 heads x 8 rows) of a K/V head share each key tile;
// S = Q K^T in 4 x 4 register tiles (64 work-items), the online softmax per query, O += P V in 4 x 8 register tiles (128 work-items).
namespace {
constexpr int AS4 = 33;   // local row stride of the Q / K / V tiles in float4 (132 floats: conflict-free for the 4 x 4 tiles)
constexpr int PS4 = 9;    // stride of the transposed probabilities P[key][query] in float4 (36 floats)
}

void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk, const int32_t* anchor_pos, int ring_slots,
                   int window, float* part, void* xq, float* xd, void* stream, const int64_t* tag, const int32_t* epoch_ptr) {
    sycl::queue* qu = strata::q_of(stream);
    static const int env_v = [] { const char* e = std::getenv("STRATA_DF2_ATTN"); return e ? std::atoi(e) : 0; }();
    const int av = g_attn_variant != 0 ? g_attn_variant : env_v != 0 ? env_v : DF2_ATTN_AUTO;
    static const int env_ck = [] { const char* e = std::getenv("STRATA_DF2_ATTN_CHUNK"); return e ? std::atoi(e) : 0; }();
    const int CK = g_attn_chunk != 0 ? g_attn_chunk : env_ck == 128 ? 128 : DF2_ATTN_CHUNK_AUTO;   // keys per work-group: 64 (33 chunks) or 128 (17)
    const int nch = CK == 128 ? kDf2MaxChunks : kDf2PartChunks;
    if (av == 2) {
        qu->submit([&](sycl::handler& h) {
            sycl::local_accessor<sycl::float4, 1> Qs(32 * AS4, h), Ks(32 * AS4, h), Ps(32 * PS4, h);
            sycl::local_accessor<float, 1> Sc(32 * 33, h), m_s(32, h), l_s(32, h), corr_s(32, h);
            sycl::local_accessor<unsigned char, 1> kvalid(32, h);
            h.parallel_for(sycl::nd_range<2>({(size_t) nch, (size_t) kDf2Kv * 256}, {1, 256}), [=](nd_item<2> it) DF2_SG16 {
                const sycl::sub_group sg = it.get_sub_group();
                const int chunk = (int) it.get_group(0), kvh = (int) it.get_group(1), tid = (int) it.get_local_id(1);
                float* Psf = reinterpret_cast<float*>(&Ps[0]);
                const int s = *anchor_pos;
                const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
                const int nctx = s - lo;
                const int nkeys = nctx + kDf2Rows;
                const int k_begin = chunk * CK;
                const int64_t ep = tag != nullptr ? ((int64_t) *epoch_ptr << 32) : 0;
                float* const pbase = part + (((size_t) chunk * kDf2Kv + kvh) * 32) * 130;
                if (k_begin >= nkeys) {
                    for (int i = tid; i < 32 * 130; i += 256) pbase[i] = (i % 130 == 0) ? -INFINITY : 0.0f;
                    return;
                }
                const float scale = sycl::rsqrt((float) HD);
#pragma unroll
                for (int u = 0; u < 4; ++u) {   // Q: 32 queries x 128, float4
                    const int i = tid + u * 256;
                    const int a = i / (HD / 4), d4 = i % (HD / 4);
                    const int row = a >> 2, g = a & 3;
                    sycl::float4 v = *reinterpret_cast<const sycl::float4*>(q + ((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d4 * 4);
                    Qs[a * AS4 + d4] = v * scale;
                }
                if (tid < 32) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }
                float O[2][8];
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) O[i][j] = 0.0f;
                const int qg_pv = tid >> 4, dg_pv = tid & 15;   // PV: queries qg*2, qg*2+1, dims dg*8 .. +7
                const int qg = tid >> 4, kg = tid & 15;         // S: queries qg*2 .. +1, keys kg*2 .. +1
                const int k_end = sycl::min(k_begin + CK, nkeys);
                auto load_tile = [&](int k0, const float* ring, const float* blk) {
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int i = tid + u * 256, t = i / (HD / 4), d4 = i % (HD / 4), k = k0 + t;
                        sycl::float4 v(0.f, 0.f, 0.f, 0.f);
                        if (k < nctx) v = *reinterpret_cast<const sycl::float4*>(ring + ((size_t) kvh * ring_slots + (lo + k) % ring_slots) * HD + d4 * 4);
                        else if (k < nkeys) v = *reinterpret_cast<const sycl::float4*>(blk + ((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d4 * 4);
                        Ks[t * AS4 + d4] = v;
                    }
                };
                for (int k0 = k_begin; k0 < k_end; k0 += 32) {
                    bar(it);
                    load_tile(k0, kring, kblk);
                    if (tid < 32) {
                        const int k = k0 + tid;
                        kvalid[tid] = (tag == nullptr || k >= nctx || k >= nkeys) ? 1 : (tag[(lo + k) % ring_slots] == (ep | (int64_t) (lo + k)) ? 1 : 0);
                    }
                    bar(it);
                    {   // S tile: 2 queries x 2 keys per work-item
                        float acc[2][2];
#pragma unroll
                        for (int i = 0; i < 2; ++i)
#pragma unroll
                            for (int j = 0; j < 2; ++j) acc[i][j] = 0.0f;
#pragma unroll 4
                        for (int d4 = 0; d4 < HD / 4; ++d4) {
                            sycl::float4 qv[2], kv[2];
#pragma unroll
                            for (int i = 0; i < 2; ++i) qv[i] = Qs[(qg * 2 + i) * AS4 + d4];
#pragma unroll
                            for (int j = 0; j < 2; ++j) kv[j] = Ks[(kg * 2 + j) * AS4 + d4];
#pragma unroll
                            for (int i = 0; i < 2; ++i)
#pragma unroll
                                for (int j = 0; j < 2; ++j) acc[i][j] += qv[i].x() * kv[j].x() + qv[i].y() * kv[j].y() + qv[i].z() * kv[j].z() + qv[i].w() * kv[j].w();
                        }
#pragma unroll
                        for (int i = 0; i < 2; ++i)
#pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                const int qi = qg * 2 + i, t = kg * 2 + j, k = k0 + t, row = qi >> 2;
                                bool ok = k < nkeys;
                                if (ok && k < nctx) ok = (lo + k) > (s + row - window) && kvalid[t];
                                Sc[qi * 33 + t] = ok ? acc[i][j] : -INFINITY;
                            }
                    }
                    bar(it);
                    {   // online softmax: query = tid / 8, 4 keys each, 8 lanes per query
                        const int qi = tid >> 3, sub = tid & 7;
                        float v[4], tmax = -INFINITY;
#pragma unroll
                        for (int u = 0; u < 4; ++u) { v[u] = Sc[qi * 33 + sub * 4 + u]; tmax = sycl::fmax(tmax, v[u]); }
                        tmax = sycl::fmax(tmax, xorf(sg, tmax, 1));
                        tmax = sycl::fmax(tmax, xorf(sg, tmax, 2));
                        tmax = sycl::fmax(tmax, xorf(sg, tmax, 4));
                        const float m_old = m_s[qi];
                        const float mn = sycl::fmax(m_old, tmax);
                        const float corr = (m_old == -INFINITY) ? 0.0f : sycl::exp(m_old - mn);
                        float psum = 0.0f;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const float p = v[u] == -INFINITY ? 0.0f : sycl::exp(v[u] - mn);
                            Psf[(sub * 4 + u) * (PS4 * 4) + qi] = p;
                            psum += p;
                        }
                        psum += xorf(sg, psum, 1);
                        psum += xorf(sg, psum, 2);
                        psum += xorf(sg, psum, 4);
                        if (sub == 0) { l_s[qi] = l_s[qi] * corr + psum; m_s[qi] = mn; corr_s[qi] = corr; }
                    }
                    bar(it);
                    load_tile(k0, vring, vblk);
                    bar(it);
                    {
                        const float cr[2] = {corr_s[qg_pv * 2], corr_s[qg_pv * 2 + 1]};
#pragma unroll
                        for (int i = 0; i < 2; ++i)
#pragma unroll
                            for (int j = 0; j < 8; ++j) O[i][j] *= cr[i];
#pragma unroll 4
                        for (int t = 0; t < 32; ++t) {
                            const float p0 = Psf[t * (PS4 * 4) + qg_pv * 2], p1 = Psf[t * (PS4 * 4) + qg_pv * 2 + 1];
                            const sycl::float4 va = Ks[t * AS4 + dg_pv * 2], vb = Ks[t * AS4 + dg_pv * 2 + 1];
                            const float vv[8] = {va.x(), va.y(), va.z(), va.w(), vb.x(), vb.y(), vb.z(), vb.w()};
#pragma unroll
                            for (int j = 0; j < 8; ++j) { O[0][j] = sycl::fma(p0, vv[j], O[0][j]); O[1][j] = sycl::fma(p1, vv[j], O[1][j]); }
                        }
                    }
                }
                bar(it);
                if (tid < 32) { pbase[tid * 130] = m_s[tid]; pbase[tid * 130 + 1] = l_s[tid]; }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    float* o = pbase + (size_t) (qg_pv * 2 + i) * 130 + 2 + dg_pv * 8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) o[j] = O[i][j];
                }
            });
        });
    } else
    qu->submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::float4, 1> Qs(32 * AS4, h), Ks(32 * AS4, h), Ps(32 * PS4, h);
        sycl::local_accessor<float, 1> Sc(32 * 33, h), m_s(32, h), l_s(32, h), corr_s(32, h);
        sycl::local_accessor<unsigned char, 1> kvalid(32, h);
        h.parallel_for(sycl::nd_range<2>({(size_t) nch, (size_t) kDf2Kv * 128}, {1, 128}), [=](nd_item<2> it) DF2_SG16 {
            const sycl::sub_group sg = it.get_sub_group();
            const int chunk = (int) it.get_group(0), kvh = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            float* Psf = reinterpret_cast<float*>(&Ps[0]);
            const int s = *anchor_pos;
            const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
            const int nctx = s - lo;
            const int nkeys = nctx + kDf2Rows;
            const int k_begin = chunk * CK;
            const int64_t ep = tag != nullptr ? ((int64_t) *epoch_ptr << 32) : 0;
            float* const pbase = part + (((size_t) chunk * kDf2Kv + kvh) * 32) * 130;
            if (k_begin >= nkeys) {
                for (int i = tid; i < 32 * 130; i += 128) pbase[i] = (i % 130 == 0) ? -INFINITY : 0.0f;
                return;
            }
            const float scale = sycl::rsqrt((float) HD);
            for (int i = tid; i < 32 * HD / 4; i += 128) {   // Q: 32 queries x 128, float4
                const int a = i / (HD / 4), d4 = i % (HD / 4);
                const int row = a >> 2, g = a & 3;
                sycl::float4 v = *reinterpret_cast<const sycl::float4*>(q + ((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d4 * 4);
                Qs[a * AS4 + d4] = v * scale;
            }
            if (tid < 32) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }
            float O[4][8];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) O[i][j] = 0.0f;
            const int qg_pv = tid >> 4, dg_pv = tid & 15;   // PV: queries qg*4 .. +3, dims dg*8 .. +7
            const int k_end = sycl::min(k_begin + CK, nkeys);
            // a tile is 32 keys x 128 dims = 1024 float4: eight per work-item
            auto load_tile = [&](int k0, const float* ring, const float* blk) {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int i = tid + u * 128, t = i / (HD / 4), d4 = i % (HD / 4), k = k0 + t;
                    sycl::float4 v(0.f, 0.f, 0.f, 0.f);
                    if (k < nctx) v = *reinterpret_cast<const sycl::float4*>(ring + ((size_t) kvh * ring_slots + (lo + k) % ring_slots) * HD + d4 * 4);
                    else if (k < nkeys) v = *reinterpret_cast<const sycl::float4*>(blk + ((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d4 * 4);
                    Ks[t * AS4 + d4] = v;
                }
            };
            for (int k0 = k_begin; k0 < k_end; k0 += 32) {
                bar(it);   // Ks free (the last tile's V reads done); Qs / m_s written on the first pass
                load_tile(k0, kring, kblk);
                if (tid < 32) {
                    const int k = k0 + tid;
                    kvalid[tid] = (tag == nullptr || k >= nctx || k >= nkeys) ? 1 : (tag[(lo + k) % ring_slots] == (ep | (int64_t) (lo + k)) ? 1 : 0);
                }
                bar(it);
                if (tid < 64) {   // S tile: 4 queries x 4 keys per work-item
                    const int qg = tid >> 3, kg = tid & 7;
                    float acc[4][4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
                    #pragma unroll 4
                    for (int d4 = 0; d4 < HD / 4; ++d4) {
                        sycl::float4 qv[4], kv[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) qv[i] = Qs[(qg * 4 + i) * AS4 + d4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) kv[j] = Ks[(kg * 4 + j) * AS4 + d4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) acc[i][j] += qv[i].x() * kv[j].x() + qv[i].y() * kv[j].y() + qv[i].z() * kv[j].z() + qv[i].w() * kv[j].w();
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int qi = qg * 4 + i, t = kg * 4 + j, k = k0 + t, row = qi >> 2;
                            bool ok = k < nkeys;
                            if (ok && k < nctx) ok = (lo + k) > (s + row - window) && kvalid[t];   // s + row - j < window, and the slot holds that row
                            Sc[qi * 33 + t] = ok ? acc[i][j] : -INFINITY;
                        }
                }
                bar(it);
                {   // online softmax: query = tid / 4, 8 keys each, 4 lanes per query
                    const int qi = tid >> 2, sub = tid & 3;
                    float v[8], tmax = -INFINITY;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) { v[u] = Sc[qi * 33 + sub * 8 + u]; tmax = sycl::fmax(tmax, v[u]); }
                    tmax = sycl::fmax(tmax, xorf(sg, tmax, 1));
                    tmax = sycl::fmax(tmax, xorf(sg, tmax, 2));
                    const float m_old = m_s[qi];
                    const float mn = sycl::fmax(m_old, tmax);
                    const float corr = (m_old == -INFINITY) ? 0.0f : sycl::exp(m_old - mn);
                    float psum = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const float p = v[u] == -INFINITY ? 0.0f : sycl::exp(v[u] - mn);
                        Psf[(sub * 8 + u) * (PS4 * 4) + qi] = p;
                        psum += p;
                    }
                    psum += xorf(sg, psum, 1);
                    psum += xorf(sg, psum, 2);
                    if (sub == 0) { l_s[qi] = l_s[qi] * corr + psum; m_s[qi] = mn; corr_s[qi] = corr; }
                }
                bar(it);   // Ps, corr_s complete; every read of Ks (the S tile) done
                load_tile(k0, vring, vblk);
                bar(it);
                {
                    const float cr[4] = {corr_s[qg_pv * 4], corr_s[qg_pv * 4 + 1], corr_s[qg_pv * 4 + 2], corr_s[qg_pv * 4 + 3]};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) O[i][j] *= cr[i];
                    #pragma unroll 4
                    for (int t = 0; t < 32; ++t) {
                        const sycl::float4 p4 = Ps[t * PS4 + qg_pv];
                        const sycl::float4 va = Ks[t * AS4 + dg_pv * 2], vb = Ks[t * AS4 + dg_pv * 2 + 1];
                        const float pv[4] = {p4.x(), p4.y(), p4.z(), p4.w()};
                        const float vv[8] = {va.x(), va.y(), va.z(), va.w(), vb.x(), vb.y(), vb.z(), vb.w()};
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) O[i][j] = sycl::fma(pv[i], vv[j], O[i][j]);
                    }
                }
            }
            bar(it);
            if (tid < 32) { pbase[tid * 130] = m_s[tid]; pbase[tid * 130 + 1] = l_s[tid]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float* o = pbase + (size_t) (qg_pv * 4 + i) * 130 + 2 + dg_pv * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = O[i][j];
            }
        });
    });
    qu->parallel_for(sycl::nd_range<1>((size_t) kDf2Kv * 32 * HD, HD), [=](nd_item<1> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int a = (int) it.get_group(0);   // kvh * 32 + qi
        const int kvh = a >> 5, qi = a & 31, d = (int) it.get_local_id(0);
        float M = -INFINITY;
        for (int c = 0; c < nch; ++c) M = sycl::fmax(M, part[(((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130]);
        float L = 0.0f, o = 0.0f;
        for (int c = 0; c < nch; ++c) {
            const float* p = part + (((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130;
            const float mc = p[0];
            const float w = (mc == -INFINITY) ? 0.0f : sycl::exp(mc - M);
            L += p[1] * w;
            o = sycl::fma(p[2 + d], w, o);
        }
        const int row = qi >> 2, g = qi & 3;
        // the output row `row`, head kvh * 4 + g, dims d: quant block (kvh * 4 + g) * 4 + d / 32 of the 4096-wide row
        quant_block(sg, L > 0.0f ? o / L : 0.0f, (kvh * kDf2Gqa + g) * (HD / 32) + (d >> 5), row, 32 * HD, static_cast<int8_t*>(xq), xd);
    });
}

// ---- Q4_0 GEMM for up to 8 activation columns (the drafter's matrices); NC = the columns that are computed (DF-03)
//
// Weights are TILED (df2_q4_tile makes them from ggml Q4_0 blocks): a group of 8 rows is one unit; per 512 values of the reduction axis (a "step", 16
// Q4_0 blocks) the group stores the 8 rows' nibbles one after the other (2 KB), a lane (block * 2 + half) takes 8 bytes = 16 values as two words.  Word 0
// holds values 0-3 in its four low nibbles and 4-7 in its four high nibbles, word 1 holds 8-11 and 12-15, so (w & 0x0F0F0F0F) and ((w >> 4) & 0x0F0F0F0F)
// are the int8 words dp4a wants, as unsigned q = 0..15 (value = (q - 8) * d).  sum((q - 8) x) = sum(q x) - 8 sum(x).  The fp16 block scales follow the
// nibble plane (n_out * n_in / 2 bytes) in the same tile order.  Activations: xq (int8 [col][n_in]) and xd (float [col][n_in / 32]).
namespace {
struct Df2Seg { const int8_t* q; const sycl::half* d; float* y; int n_out; int row0; };
struct Df2Segs { Df2Seg s[3]; int n; };

inline Df2Seg df2_pick(const Df2Segs& segs, int row_g) {
    Df2Seg sg = segs.s[0];
    if (segs.n > 1 && row_g >= segs.s[1].row0) sg = segs.s[1];
    if (segs.n > 2 && row_g >= segs.s[2].row0) sg = segs.s[2];
    return sg;
}

template <int WARPS, int R, int UNR, int NC>
void gemm4_launch(sycl::queue* q, const Df2Segs& segs, const int8_t* xq, const float* xd, int n_in, int rows) {
    constexpr int SUBS = 8 / R;
    const size_t groups = (size_t) (rows / 8) * SUBS / WARPS;
    q->parallel_for(sycl::nd_range<1>(groups * WARPS * 32, WARPS * 32), [=](nd_item<1> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int warp = (int) (it.get_local_id(0) >> 5), lane = (int) sg.get_local_linear_id();
        const int wi = (int) it.get_group(0) * WARPS + warp;
        const int row_g = (wi / SUBS) * 8, sub = wi % SUBS;
        const Df2Seg sgm = df2_pick(segs, row_g);
        const int row0 = row_g - sgm.row0;
        const int nb = n_in / 32, steps = nb / 16;
        const sycl::uint2* wq = reinterpret_cast<const sycl::uint2*>(sgm.q + (size_t) row0 * (n_in / 2)) + lane + sub * R * 32;
        const sycl::half* wd = sgm.d + (size_t) row0 * nb + (lane >> 1) + sub * R * 16;
        const sycl::int4* xv = reinterpret_cast<const sycl::int4*>(xq) + lane;
        float acc[R][NC];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
        for (int st = 0; st < steps; ++st) {
            int w[R][4];
            float dw[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const sycl::uint2 raw = wq[(size_t) st * 256 + r * 32];
                // bytewise (q - 8) for q = 0..15: ((q | 0x80) - 8) ^ 0x80, no borrow across bytes; sum((q - 8) x) is the integer the CUDA kernel gets as sum(q x) - 8 sum(x)
                w[r][0] = (int) ((((raw.x() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                w[r][1] = (int) (((((raw.x() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                w[r][2] = (int) ((((raw.y() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                w[r][3] = (int) (((((raw.y() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                dw[r] = (float) wd[(size_t) st * 128 + r * 16];
            }
            float xs[NC];
            #pragma unroll
            for (int c = 0; c < NC; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
            #pragma unroll
            for (int c = 0; c < NC; ++c) {
                const sycl::int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    int isum = STRATA_DP4A(w[r][0], x.x(), 0);
                    isum = STRATA_DP4A(w[r][1], x.y(), isum);
                    isum = STRATA_DP4A(w[r][2], x.z(), isum);
                    isum = STRATA_DP4A(w[r][3], x.w(), isum);
                    acc[r][c] = sycl::fma(dw[r] * xs[c], (float) isum, acc[r][c]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int c = 0; c < NC; ++c) {
                float v = acc[r][c];
                v += xorf(sg, v, 16);
                v += xorf(sg, v, 8);
                v += xorf(sg, v, 4);
                v += xorf(sg, v, 2);
                v += xorf(sg, v, 1);
                if (lane == 0) sgm.y[(size_t) c * sgm.n_out + row0 + sub * R + r] = v;
            }
    });
}

template <int WARPS, int R, int UNR>
void gemm4_nc(sycl::queue* q, const Df2Segs& segs, const int8_t* xq, const float* xd, int n_in, int rows, int ncols) {
    switch (ncols) {
        case 1: gemm4_launch<WARPS, R, UNR, 1>(q, segs, xq, xd, n_in, rows); break;
        case 2: gemm4_launch<WARPS, R, UNR, 2>(q, segs, xq, xd, n_in, rows); break;
        case 3: gemm4_launch<WARPS, R, UNR, 3>(q, segs, xq, xd, n_in, rows); break;
        case 4: gemm4_launch<WARPS, R, UNR, 4>(q, segs, xq, xd, n_in, rows); break;
        case 5: gemm4_launch<WARPS, R, UNR, 5>(q, segs, xq, xd, n_in, rows); break;
        case 6: gemm4_launch<WARPS, R, UNR, 6>(q, segs, xq, xd, n_in, rows); break;
        case 7: gemm4_launch<WARPS, R, UNR, 7>(q, segs, xq, xd, n_in, rows); break;
        default: gemm4_launch<WARPS, R, UNR, 8>(q, segs, xq, xd, n_in, rows); break;
    }
}


// The same GEMM on sub-groups of 16 (SIMD16: a variable takes 2 GRFs, not 4, so R rows a lane fit without spilling): the 32 lanes of the CUDA warp are two
// sub-groups, `half` 0 for lanes 0-15 and `half` 1 for lanes 16-31 of the same row group, each lane doing the work of its virtual lane l = half * 16 + l16 exactly
// (same steps, same fma order).  The xor-16 stage of the shuffle tree is acc[l] + acc[l + 16], an add that commutes, so half 1 hands its sums over through local
// memory and half 0 finishes the tree (8, 4, 2, 1): bitwise the result of gemm4_launch.
template <int R, int NC, int PAIRS, int UNR>
void gemm4p_launch(sycl::queue* q, const Df2Segs& segs, const int8_t* xq, const float* xd, int n_in, int rows) {
    constexpr int SUBS = 8 / R;
    const size_t pairs = (size_t) (rows / 8) * SUBS;
    q->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> slm((size_t) PAIRS * R * NC * 16, hd);
        hd.parallel_for(sycl::nd_range<1>((pairs / PAIRS) * PAIRS * 32, PAIRS * 32), [=](nd_item<1> it) DF2_SG16 {
            const sycl::sub_group sg = it.get_sub_group();
            const int l16 = (int) sg.get_local_linear_id();
            const int sgi = (int) (it.get_local_id(0) >> 4);
            const int pair = sgi >> 1, half = sgi & 1;
            const int lane = half * 16 + l16;
            const size_t gp = it.get_group(0) * PAIRS + pair;
            const int row_g = (int) (gp / SUBS) * 8, sub = (int) (gp % SUBS);
            const Df2Seg sgm = df2_pick(segs, row_g);
            const int row0 = row_g - sgm.row0;
            const int nb = n_in / 32, steps = nb / 16;
            const sycl::uint2* wq = reinterpret_cast<const sycl::uint2*>(sgm.q + (size_t) row0 * (n_in / 2)) + lane + sub * R * 32;
            const sycl::half* wd = sgm.d + (size_t) row0 * nb + (lane >> 1) + sub * R * 16;
            const sycl::int4* xv = reinterpret_cast<const sycl::int4*>(xq) + lane;
            float acc[R][NC];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
            for (int st = 0; st < steps; ++st) {
                int w[R][4];
                float dw[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const sycl::uint2 raw = wq[(size_t) st * 256 + r * 32];
                    w[r][0] = (int) ((((raw.x() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                    w[r][1] = (int) (((((raw.x() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                    w[r][2] = (int) ((((raw.y() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                    w[r][3] = (int) (((((raw.y() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                    dw[r] = (float) wd[(size_t) st * 128 + r * 16];
                }
                float xs[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const sycl::int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int isum = STRATA_DP4A(w[r][0], x.x(), 0);
                        isum = STRATA_DP4A(w[r][1], x.y(), isum);
                        isum = STRATA_DP4A(w[r][2], x.z(), isum);
                        isum = STRATA_DP4A(w[r][3], x.w(), isum);
                        acc[r][c] = sycl::fma(dw[r] * xs[c], (float) isum, acc[r][c]);
                    }
                }
            }
            if (half == 1) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int c = 0; c < NC; ++c) slm[((pair * R + r) * NC + c) * 16 + l16] = acc[r][c];
            }
            sycl::group_barrier(it.get_group());
            if (half == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        float v = acc[r][c] + slm[((pair * R + r) * NC + c) * 16 + l16];   // the xor-16 stage
                        v += xorf(sg, v, 8);
                        v += xorf(sg, v, 4);
                        v += xorf(sg, v, 2);
                        v += xorf(sg, v, 1);
                        if (l16 == 0) sgm.y[(size_t) c * sgm.n_out + row0 + sub * R + r] = v;
                    }
            }
        });
    });
}

template <int R, int PAIRS, int UNR>
void gemm4p_nc(sycl::queue* q, const Df2Segs& segs, const int8_t* xq, const float* xd, int n_in, int rows, int ncols) {
    switch (ncols) {
        case 1: gemm4p_launch<R, 1, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 2: gemm4p_launch<R, 2, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 3: gemm4p_launch<R, 3, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 4: gemm4p_launch<R, 4, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 5: gemm4p_launch<R, 5, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 6: gemm4p_launch<R, 6, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        case 7: gemm4p_launch<R, 7, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
        default: gemm4p_launch<R, 8, PAIRS, UNR>(q, segs, xq, xd, n_in, rows); break;
    }
}

// ---- planar quantization / norm kernels
void quant_rows_launch(sycl::queue* q, const float* x, int K, int ncols, int8_t* xq, float* xd) {
    q->parallel_for(sycl::nd_range<2>({(size_t) ncols, (size_t) ((K / 32 + 7) / 8) * 256}, {1, 256}), [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int col = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5), lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= K) return;
        quant_block(sg, x[(size_t) col * K + blk * 32 + lane], blk, col, K, xq, xd);
    });
}
}  // namespace

#ifndef DF2_Q4_AUTO
#define DF2_Q4_AUTO(rows) ((rows) <= 1024 ? 40 : 41)   // SIMD16 pair kernels (bitwise the same results): 2 rows a lane for the big matrices, 1 for the 640-row ones
#endif

void df2_gemm(const Df2Mat* mats, int n, const void* xq, const float* xd, int n_in, int ncols, void* stream) {
    if (n < 1 || n > 3 || n_in % 512 != 0 || ncols < 1 || ncols > 8) throw std::invalid_argument("df2_gemm: unsupported shape");
    Df2Segs segs;
    int rows = 0;
    for (int i = 0; i < n; ++i) {
        if (mats[i].n_out % 16 != 0) throw std::invalid_argument("df2_gemm: n_out must be a multiple of 16");
        if (!mats[i].q4) throw std::invalid_argument("df2_gemm: the SYCL drafter runs the Q4_0 pack only");
        const int8_t* qw = static_cast<const int8_t*>(mats[i].w);
        segs.s[i] = Df2Seg{qw, reinterpret_cast<const sycl::half*>(qw + (size_t) mats[i].n_out * n_in / 2), mats[i].y, mats[i].n_out, rows};
        rows += mats[i].n_out;
    }
    segs.n = n;
    const int8_t* xqp = static_cast<const int8_t*>(xq);
    sycl::queue* q = strata::q_of(stream);
    static const int env_v = [] { const char* e = std::getenv("STRATA_DF2_GEMM4"); return e ? std::atoi(e) : 0; }();
    const int v = g_gemm_variant != 0 ? g_gemm_variant : env_v != 0 ? env_v : DF2_Q4_AUTO(rows);
    switch (v) {   // the ids are the CUDA file's (W warps a block, R rows a warp)
        case 10: gemm4_nc<4, 4, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 11: gemm4_nc<4, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 12: gemm4_nc<4, 1, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 24: gemm4_nc<1, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 25: gemm4_nc<1, 4, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 28: gemm4_nc<1, 1, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 34: gemm4_nc<2, 1, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 17: gemm4_nc<2, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 40: gemm4p_nc<1, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;   // sub-groups of 16, two per row group (R rows a lane, PAIRS pairs a work-group)
        case 41: gemm4p_nc<2, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 42: gemm4p_nc<4, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        case 43: gemm4p_nc<8, 2, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;
        default: gemm4_nc<1, 1, 2>(q, segs, xqp, xd, n_in, rows, ncols); break;   // 28
    }
}

void df2_gemm_variant(int v) { g_gemm_variant = v; }
void df2_attention_variant(int v) { g_attn_variant = v; }
void df2_attention_chunk(int ck) { g_attn_chunk = ck; }

void df2_quant_rows(const float* x, int K, int ncols, void* xq, float* xd, void* stream) {
    quant_rows_launch(strata::q_of(stream), x, K, ncols, static_cast<int8_t*>(xq), xd);
}

// RMS norm of row `col` (N values) then the planar quantization of the normalized values; also writes them when xn != null.  One group of 256 per row.
void df2_rms_quant(const float* h, const float* gamma, float* xn, int rows, int N, float eps, void* xq, float* xd, void* stream) {
    sycl::queue* q = strata::q_of(stream);
    int8_t* xqi = static_cast<int8_t*>(xq);
    q->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int col = (int) it.get_group(0), tid = (int) it.get_local_id(0), warp = tid >> 5, lane = tid & 31;
            float ss = 0.0f;
            for (int i = tid; i < N; i += 256) { const float v = h[(size_t) col * N + i]; ss += v * v; }
            for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
            if (lane == 0) red[warp] = ss;
            bar(it);
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[w];
            const float scale = sycl::rsqrt(tot / (float) N + eps);
            for (int blk = warp; blk * 32 < N; blk += 8) {
                const int i = blk * 32 + lane;
                const float v = h[(size_t) col * N + i] * scale * gamma[i];
                if (xn) xn[(size_t) col * N + i] = v;
                quant_block(sg, v, blk, col, N, xqi, xd);
            }
        });
    });
}

// conv (as df2_conv, no residual) then the planar quantization: grid (ceil(H / 32 / 8), rows), 256 work-items
void df2_conv_quant(const float* x, const float* dyn, int dyn_stride, const float* base, int rows, int H, int group, void* xq, float* xd, void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    strata::q_of(stream)->parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((H / 32 + 7) / 8) * 256}, {1, 256}), [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int b = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5), lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= H) return;
        const int c = blk * 32 + lane, groups = H / group;
        float acc = 0.0f;
        #pragma unroll
        for (int o = 0; o < 2; ++o) {
            if (b - o < 0) continue;
            const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
            acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
        }
        quant_block(sg, acc, blk, b, H, xqi, xd);
    });
}

// h += conv(x) (the finish convolution with its residual), then RMS norm of the new h with `gamma` -> xn (optional) and the planar quantization.
// One group of 256 per row: a sub-group owns whole 32-blocks (80 of them for 2560, ten each), so the quantization needs no exchange.
void df2_conv_rms_quant(const float* x, const float* dyn, int dyn_stride, const float* base, float* h, const float* gamma, float* xn, int rows, int H, int group,
                        float eps, void* xq, float* xd, void* stream) {
    constexpr int MAXJ = 12;   // H <= 3072 -> 96 blocks -> 12 per sub-group
    if (H > 3072) throw std::invalid_argument("df2_conv_rms_quant: H too large");
    sycl::queue* q = strata::q_of(stream);
    int8_t* xqi = static_cast<int8_t*>(xq);
    q->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int b = (int) it.get_group(0), tid = (int) it.get_local_id(0), warp = tid >> 5, lane = tid & 31;
            const int groups = H / group, nb = H / 32;
            float v[MAXJ];
            float ss = 0.0f;
            #pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int blk = warp + j * 8;
                v[j] = 0.0f;
                if (blk < nb) {
                    const int c = blk * 32 + lane;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int o = 0; o < 2; ++o) {
                        if (b - o < 0) continue;
                        const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
                        acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
                    }
                    v[j] = h[(size_t) b * H + c] + acc;
                    h[(size_t) b * H + c] = v[j];
                    ss += v[j] * v[j];
                }
            }
            for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
            if (lane == 0) red[warp] = ss;
            bar(it);
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[w];
            const float scale = sycl::rsqrt(tot / (float) H + eps);
            #pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int blk = warp + j * 8;
                if (blk < nb) {
                    const int c = blk * 32 + lane;
                    const float y = v[j] * scale * gamma[c];
                    if (xn) xn[(size_t) b * H + c] = y;
                    quant_block(sg, y, blk, b, H, xqi, xd);
                }
            }
        });
    });
}

// q and k of the block: per (row, head) RMS norm and rope; group y < q_heads is a q head (-> qo, row-major rows), the rest k heads (-> ko)
void df2_norm_rope_qk(const float* q, const float* k, const float* gq, const float* gk, const float* table, float* qo, float* ko, int rows, int q_heads, int k_heads,
                      float eps, void* stream) {
    sycl::queue* qu = strata::q_of(stream);
    qu->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(4, hd), y(HD, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) (q_heads + k_heads), (size_t) rows * HD}, {1, HD}), [=](nd_item<2> it) DF2_SG32 {
            const int hh = (int) it.get_group(0), r = (int) it.get_group(1), d = (int) it.get_local_id(1);
            const bool isq = hh < q_heads;
            const int h = isq ? hh : hh - q_heads, heads = isq ? q_heads : k_heads;
            const float* in = isq ? q : k;
            const float* gamma = isq ? gq : gk;
            float* out = isq ? qo : ko;
            const float x = in[((size_t) r * heads + h) * HD + d];
            const float tot = head_norm(it, red, x, eps);
            const float v = x * sycl::rsqrt(tot / (float) HD + eps) * gamma[d];
            y[d] = v;
            bar(it);
            const int i = d & 63;
            const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
            out[((size_t) r * heads + h) * HD + d] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
        });
    });
}

// the context update's ring write: K norm + rope of (row, head) and the V copy, both into the head-major ring slot pos % slots
void df2_kv_ring(const float* kin, const float* vin, const float* gk, const int32_t* pos, const float* table, float* kring, float* vring, int slots, int rows, int heads,
                 float eps, void* stream, int64_t* tag, const int32_t* epoch_ptr) {
    sycl::queue* qu = strata::q_of(stream);
    qu->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(4, hd), y(HD, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) heads, (size_t) rows * HD}, {1, HD}), [=](nd_item<2> it) DF2_SG32 {
            const int h = (int) it.get_group(0), r = (int) it.get_group(1), d = (int) it.get_local_id(1);
            const float x = kin[((size_t) r * heads + h) * HD + d];
            const float tot = head_norm(it, red, x, eps);
            const float v = x * sycl::rsqrt(tot / (float) HD + eps) * gk[d];
            y[d] = v;
            bar(it);
            const int i = d & 63;
            const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
            const int slot = (int) (pos[r] % slots);
            const size_t at = ((size_t) h * slots + slot) * HD + d;
            kring[at] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
            vring[at] = vin[((size_t) r * heads + h) * HD + d];
            if (tag != nullptr && h == 0 && d == 0) tag[slot] = ((int64_t) *epoch_ptr << 32) | (int64_t) pos[r];
        });
    });
}

// silu(gate) * up then the planar quantization: grid (ceil(I / 32 / 8), rows), 256 work-items
void df2_swiglu_quant(const float* gate, const float* up, int rows, int I, void* xq, float* xd, void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    strata::q_of(stream)->parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((I / 32 + 7) / 8) * 256}, {1, 256}), [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int b = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5), lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= I) return;
        const size_t i = (size_t) b * I + blk * 32 + lane;
        const float g = gate[i];
        quant_block(sg, g / (1.0f + sycl::exp(-g)) * up[i], blk, b, I, xqi, xd);
    });
}

// ggml Q4_0 blocks (18 bytes: fp16 d, 16 bytes with value i in the low nibble of byte i and value i + 16 in the high nibble) of a [n_out][n_in] matrix -> the
// nibble plane (n_out * n_in / 2 bytes, the layout above) and the tiled d plane; one work-item per block
void df2_q4_tile(const void* blocks_, void* qs_, void* dpl_, int n_in, int64_t count, void* stream) {
    const uint8_t* blocks = static_cast<const uint8_t*>(blocks_);
    uint8_t* qs = static_cast<uint8_t*>(qs_);
    uint16_t* dpl = static_cast<uint16_t*>(dpl_);
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>((size_t) ((count + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int64_t b = (int64_t) it.get_global_id(0);
        if (b >= count) return;
        const int nb = n_in / 32;
        const int64_t row = b / nb;
        const int blk = (int) (b % nb);
        const int64_t g = row / 8;
        const int r = (int) (row % 8), st = blk / 16, bi = blk % 16;
        const uint8_t* src = blocks + b * 18;
        dpl[(g * 8 * nb) + (int64_t) st * 128 + r * 16 + bi] = (uint16_t) src[0] | ((uint16_t) src[1] << 8);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {   // values half * 16 .. +15 of the block
            uint32_t wd[2];
            #pragma unroll
            for (int w = 0; w < 2; ++w) {   // word w holds values w * 8 .. w * 8 + 7 of the half: byte p = (value w*8+p, value w*8+4+p) as (low, high) nibble
                uint32_t v = 0;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint8_t b_lo = src[2 + w * 8 + p], b_hi = src[2 + w * 8 + 4 + p];
                    const uint32_t q_lo = half == 0 ? (b_lo & 15u) : (b_lo >> 4), q_hi = half == 0 ? (b_hi & 15u) : (b_hi >> 4);
                    v |= (q_lo | (q_hi << 4)) << (p * 8);
                }
                wd[w] = v;
            }
            uint8_t* o = qs + g * 8 * (n_in / 2) + (int64_t) st * 2048 + r * 256 + (bi * 2 + half) * 8;
            reinterpret_cast<sycl::uint2*>(o)[0] = sycl::uint2(wd[0], wd[1]);
        }
    });
}

void df2_q8_tile(const void*, void*, void*, int, int64_t, void*) {
    throw std::invalid_argument("df2_q8_tile: the SYCL drafter runs the Q4_0 pack only");
}

// ---- top-16
namespace {
constexpr int TK_CHUNK_MAX = 4096;
inline bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }
}

void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok, void* stream) {
    if ((n + kDf2TopBlocks - 1) / kDf2TopBlocks > TK_CHUNK_MAX) throw std::invalid_argument("df2_topk: too many logits per row");
    float* cval = reinterpret_cast<float*>(scratch);
    int32_t* cidx = reinterpret_cast<int32_t*>(scratch + (size_t) rows * kDf2TopBlocks * kDf2TopK);
    sycl::queue* q = strata::q_of(stream);
    q->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> vals(TK_CHUNK_MAX, hd), wv(8, hd);
        sycl::local_accessor<int, 1> wi(8, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) kDf2TopBlocks * 256}, {1, 256}), [=](nd_item<2> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int row = (int) it.get_group(0), blk = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            const int cs = (n + kDf2TopBlocks - 1) / kDf2TopBlocks;
            const int c0 = blk * cs, c1 = sycl::min(c0 + cs, n);
            for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[(size_t) row * n + c0 + i] : -INFINITY;
            bar(it);
            for (int k = 0; k < kDf2TopK; ++k) {
                float bv = -INFINITY;
                int bi = 0x7fffffff;
                for (int i = tid; i < cs; i += 256)
                    if (better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = xorf(sg, bv, o);
                    const int oi = xori(sg, bi, o);
                    if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
                }
                if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; }
                bar(it);
                if (tid == 0) {
                    float v = wv[0];
                    int ix = wi[0];
                    for (int w = 1; w < 8; ++w)
                        if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
                    const size_t o = ((size_t) row * kDf2TopBlocks + blk) * kDf2TopK + k;
                    cval[o] = v;
                    cidx[o] = v == -INFINITY ? -1 : c0 + ix;
                    if (v != -INFINITY) vals[ix] = -INFINITY;
                }
                bar(it);
            }
        });
    });
    q->submit([&](sycl::handler& hd) {
        constexpr int NC = kDf2TopBlocks * kDf2TopK;
        sycl::local_accessor<float, 1> sv(NC, hd), wv(8, hd);
        sycl::local_accessor<int, 1> si(NC, hd), wi(8, hd), wp(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int row = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            for (int i = tid; i < NC; i += 256) { sv[i] = cval[(size_t) row * NC + i]; si[i] = cidx[(size_t) row * NC + i]; }
            bar(it);
            for (int k = 0; k < kDf2TopK; ++k) {
                float bv = -INFINITY;
                int bi = 0x7fffffff, bp = -1;
                for (int i = tid; i < NC; i += 256)
                    if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = xorf(sg, bv, o);
                    const int oi = xori(sg, bi, o);
                    const int op = xori(sg, bp, o);
                    if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
                }
                if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; wp[tid >> 5] = bp; }
                bar(it);
                if (tid == 0) {
                    float v = wv[0];
                    int ix = wi[0], p = wp[0];
                    for (int w = 1; w < 8; ++w)
                        if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
                    if (p >= 0) si[p] = -1;
                    val[row * kDf2TopK + k] = v;
                    tok[row * kDf2TopK + k] = p >= 0 ? (idx_map ? idx_map[ix] : ix) : 0;
                }
                bar(it);
            }
        });
    });
}

// ---- the selector's walk: one group of 256 work-items
void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb, int rank, int rows,
                const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream, bool cb_q4) {
    if (cb_q4) throw std::invalid_argument("df2_select: Q4_0 selector codebooks are not on SYCL (the loader keeps the bf16 pack)");
    if (rank != 256) throw std::invalid_argument("df2_select: the selector rank must be 256");
    sycl::queue* q = strata::q_of(stream);
    q->submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> vp(256, hd), sc(kDf2TopK, hd);
        sycl::local_accessor<int, 1> prev(1, hd);
        hd.parallel_for(sycl::nd_range<1>(256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0), w = tid >> 5, lane = tid & 31;
            if (tid == 0) prev[0] = *anchor_tok;
            bar(it);
            for (int i = 0; i < rows; ++i) {
                vp[tid] = bf16f(pred_cb[(size_t) prev[0] * rank + tid]) * hp[(size_t) i * rank + tid];
                bar(it);
                for (int j = w * 2; j < w * 2 + 2; ++j) {
                    const int t = tok[i * kDf2TopK + j];
                    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4*>(succ_cb + (size_t) t * rank + lane * 8);
                    const uint32_t u[4] = {raw.x(), raw.y(), raw.z(), raw.w()};
                    float acc = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        acc = sycl::fma(sycl::bit_cast<float>(u[e] << 16), vp[lane * 8 + e * 2], acc);
                        acc = sycl::fma(sycl::bit_cast<float>(u[e] & 0xffff0000u), vp[lane * 8 + e * 2 + 1], acc);
                    }
                    for (int o = 16; o > 0; o >>= 1) acc += xorf(sg, acc, o);
                    if (lane == 0) sc[j] = unary[i * kDf2TopK + j] + acc;
                }
                bar(it);
                if (tid == 0) {
                    int bj = 0;
                    float bv = sc[0];
                    for (int j = 1; j < kDf2TopK; ++j)
                        if (sc[j] > bv) { bv = sc[j]; bj = j; }
                    float z = 0.0f;
                    for (int j = 0; j < kDf2TopK; ++j) z += sycl::exp(sc[j] - bv);
                    out_tok[i] = tok[i * kDf2TopK + bj];
                    out_prob[i] = 1.0f / z;
                    prev[0] = out_tok[i];
                }
                bar(it);
            }
        });
    });
}

void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((n_embd + 255) / 256) * 256}, {1, 256}), [=](nd_item<2> it) {
        const int64_t i = (int64_t) it.get_group(1) * 256 + (int64_t) it.get_local_id(1);
        const int r = (int) it.get_group(0);
        if (i >= n_embd) return;
        float s = 0.0f;
        for (int c = 0; c < hc; ++c) s += R[((size_t) r * hc + c) * n_embd + i];
        out[(size_t) r * out_stride + i] = s / (float) hc;
    });
}

void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject, int64_t inj_stride, float* out, int64_t out_stride, int rows,
                int64_t n_embd, int hc, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((n_embd + 255) / 256) * 256}, {1, 256}), [=](nd_item<2> it) {
        const int64_t i = (int64_t) it.get_group(1) * 256 + (int64_t) it.get_local_id(1);
        const int r = (int) it.get_group(0);
        if (i >= n_embd) return;
        float wsum = 0.0f, rs = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const float x = inject[(size_t) r * inj_stride + c] / (float) hc;
            wsum += 2.0f / (1.0f + sycl::exp(-x));
            rs += R[(size_t) r * R_stride + (size_t) c * n_embd + i];
        }
        out[(size_t) r * out_stride + i] = (rs + bo[(size_t) r * bo_stride + i] * wsum) / (float) hc;
    });
}

}  // namespace strata::kernels
