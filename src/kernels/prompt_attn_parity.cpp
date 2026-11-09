// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/prompt_attn_parity.cpp - the plain models' prompt attention (`prompt_attn`, prompt_attn.hpp) against an
// FP64 host reference (GPU, synthetic, no model):
//
//   Qwen3.5-9B's 16/4 heads of 256 on FP16 pools behind a shuffled page table, a chunk of queries after cells already
//   in the pools, its length a whole number of neither kernel's query blocks; every query attends causally to the
//   cells up to its own.  The kernel is the device's choice: the WMMA one on RDNA3 and RDNA4, the FMA one elsewhere
//   and under STRATA_PROMPT_ATTN_MMA=0 (CTest runs both ways).  The output agrees with FP64 to 1e-4 of its scale
//   (the FMA kernel's FP32), or 5e-3 where the WMMA kernel rounds Q and the probabilities to FP16 (a wrong head
//   or cell is off by the output's whole scale).
//
//   PLaMo 3 Translate's sliding-window layers (24/4, 16/1 and 32/4 heads of 128): a query sees the last `window` cells
//   only, the history longer than the window; and through a ring of pages (the page table's entries repeating, as
//   PlainModel's sliding-window layers hold their cells), whose cells before the window hold newer values.
//
//   The same through the pools overload in each of its formats (FP16, INT8 with an FP16 scale per 64 values, Q4_0,
//   INT8 K with Q4_0 V): the reference reads the values the codes stand for, so what is checked is the kernels'
//   decoding and arithmetic, not the quantization; 5e-3 of the scale (the matrix engines' FP16).  A device without
//   a kernel for those pools, and STRATA_PROMPT_ATTN_MMA=0, leave the quantized forms to the decode kernel: skipped.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/platform/env.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;

template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

// The pools overload's formats (prompt_attn.hpp); -1: the FP16 pools of the plain overload
enum { kPlain = -1, kF16 = 0, kI8 = 1, kQ4 = 2, kK8V4 = 3 };

// a pool plane as INT8 codes and FP16 scales per 64 (as kv_q8.hpp rounds them), into `deq`
void to_q8(const std::vector<float>& x, std::vector<int8_t>& codes, std::vector<uint16_t>& scales,
           std::vector<double>& deq) {
    codes.assign(x.size(), 0);
    scales.assign(x.size() / 64, 0);
    deq.assign(x.size(), 0.0);
    for (size_t g = 0; g < x.size() / 64; ++g) {
        float amax = 0.f;
        for (size_t i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(x[g * 64 + i]));
        scales[g] = k::f16_from_f32(amax / 127.0f);
        const float sf = k::f32_from_f16(scales[g]);
        for (size_t i = 0; i < 64; ++i) {
            const int q = sf > 0.f ? std::clamp((int) std::nearbyint(x[g * 64 + i] / sf), -127, 127) : 0;
            codes[g * 64 + i] = (int8_t) q;
            deq[g * 64 + i] = (double) q * sf;
        }
    }
}

// the same plane as Q4_0 blocks (kv_q4.hpp: d = the value of largest magnitude / -8, codes (x / d + 8.5) truncated)
void to_q4(const std::vector<float>& x, std::vector<uint8_t>& q4, std::vector<double>& deq) {
    const size_t nb = x.size() / k::QK4_0;
    q4.assign(nb * sizeof(k::block_q4_0), 0);
    deq.assign(x.size(), 0.0);
    for (size_t b = 0; b < nb; ++b) {
        const float* v = x.data() + b * k::QK4_0;
        float amax = 0.f, mval = 0.f;
        for (int j = 0; j < k::QK4_0; ++j)
            if (std::fabs(v[j]) > amax) { amax = std::fabs(v[j]); mval = v[j]; }
        const uint16_t d16 = k::f16_from_f32(mval / -8.0f);
        const float d = k::f32_from_f16(d16), id = d != 0.f ? 1.0f / d : 0.f;
        k::block_q4_0 blk{};
        blk.d = d16;
        for (int j = 0; j < k::QK4_0; ++j) {
            const int q = std::min(15, (int) (v[j] * id + 8.5f));
            if (j < 16) blk.qs[j] = (uint8_t) (blk.qs[j] | q);
            else blk.qs[j - 16] = (uint8_t) (blk.qs[j - 16] | (q << 4));
            deq[b * k::QK4_0 + (size_t) j] = (double) (q - 8) * d;
        }
        std::memcpy(q4.data() + b * sizeof(k::block_q4_0), &blk, sizeof(blk));
    }
}

int check(const char* name, int64_t nh, int64_t nkv, int64_t hd, int64_t pos0, int64_t T, int fmt = kPlain,
          int64_t window = 0, int64_t ring = 0) {
    par::Device dev;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = nh; s.n_head_kv = nkv; s.head_dim = hd;
    s.page_size = 64;   // the plain models' pages (plain_model.cpp): FP16 pools take the shared kernel (flash_attn.hpp)
    const int64_t PS = s.page_size, cells = pos0 + T, pages = (cells + PS - 1) / PS + 1;
    const int64_t held = ring > 0 ? ring : pages, rows = held * nkv * PS;   // the pools' pages
    const int64_t G = nh / nkv;
    std::mt19937 rng(91 + (unsigned) (nh + hd + T));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> kf((size_t) (rows * hd)), vf((size_t) (rows * hd));
    for (auto& x : kf) x = nd(rng) * 1.5f;
    for (auto& x : vf) x = nd(rng);
    // the values the pools hold: FP16, or what the codes stand for
    std::vector<uint16_t> kh(kf.size()), vh(vf.size());
    std::vector<double> kd(kf.size()), vd(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        kh[i] = k::f16_from_f32(kf[i]);
        vh[i] = k::f16_from_f32(vf[i]);
        kd[i] = k::f32_from_f16(kh[i]);
        vd[i] = k::f32_from_f16(vh[i]);
    }
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ksc, vsc;
    std::vector<uint8_t> k4, v4;
    if (fmt == kI8 || fmt == kK8V4) to_q8(kf, kq, ksc, kd);
    if (fmt == kI8) to_q8(vf, vq, vsc, vd);
    if (fmt == kQ4) to_q4(kf, k4, kd);
    if (fmt == kQ4 || fmt == kK8V4) to_q4(vf, v4, vd);
    std::vector<int32_t> perm((size_t) held), table((size_t) pages);
    for (int64_t i = 0; i < held; ++i) perm[(size_t) i] = (int32_t) i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = perm[(size_t) (i % held)];
    std::vector<float> q((size_t) (T * nh * hd));
    for (auto& x : q) x = nd(rng) * 2.0f;

    // FP64 reference: query t (position pos0 + t), head h reads KV head h / G over cells 0 .. pos0 + t (with a window,
    // from pos0 + t + 1 - window on).  In a ring a cell's row holds the last position written there: the cells the
    // window sees are those (the random rows stand for whichever position wrote them last)
    std::vector<double> ref(q.size(), 0.0);
    double scale_out = 0.0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < nh; ++h) {
            const int64_t kvh = h / G, n = pos0 + t + 1, c0 = window > 0 ? std::max<int64_t>(0, n - window) : 0;
            const float* qr = q.data() + (t * nh + h) * hd;
            std::vector<double> sc((size_t) n);
            double m = -1e300;
            for (int64_t c = c0; c < n; ++c) {
                const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
                double d = 0.0;
                for (int64_t i = 0; i < hd; ++i) d += (double) qr[i] * kd[(size_t) (row * hd + i)];
                sc[(size_t) c] = d / std::sqrt((double) hd);
                m = std::max(m, sc[(size_t) c]);
            }
            double l = 0.0;
            for (int64_t c = c0; c < n; ++c) {
                double& v = sc[(size_t) c];
                v = std::exp(v - m);
                l += v;
            }
            double* out = ref.data() + (t * nh + h) * hd;
            for (int64_t c = c0; c < n; ++c) {
                const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
                for (int64_t i = 0; i < hd; ++i) out[i] += sc[(size_t) c] / l * vd[(size_t) (row * hd + i)];
            }
        }
    for (double v : ref) scale_out = std::max(scale_out, std::fabs(v));

    const int32_t* d_table = up(dev, table);
    const float* d_q = up(dev, q);
    float* d_attn = dev.alloc<float>(q.size());
    k::QsaAttnPools pools;
    pools.page_table = d_table;
    if (fmt == kPlain || fmt == kF16) {
        pools.k_pool = up(dev, kh);
        pools.v_pool = up(dev, vh);
    }
    if (!kq.empty()) { pools.k_q = up(dev, kq); pools.k_scale = up(dev, ksc); }
    if (!vq.empty()) { pools.v_q = up(dev, vq); pools.v_scale = up(dev, vsc); }
    if (!k4.empty()) pools.k_q4 = up(dev, k4);
    if (!v4.empty()) pools.v_q4 = up(dev, v4);
    const char* mma = strata::env_get("PROMPT_ATTN_MMA");
    const bool fma = mma != nullptr && std::strtol(mma, nullptr, 10) == 0;
    const bool took = fmt == kPlain ? k::prompt_attn(d_q, pools.k_pool, pools.v_pool, d_table, pos0, T, s, d_attn, nullptr,
                                                     window)
                                    : k::prompt_attn(d_q, pools, pos0, T, s, d_attn, nullptr, window);
    if (!took) {
        if (fmt > kF16) {   // the decode kernel's pools here
            std::printf("  %-34s no prompt kernel for these pools here: skipped\n", name);
            return 0;
        }
        std::printf("  %-34s the kernel does not take this shape: FAIL\n", name);
        return 1;
    }
    std::vector<float> attn(q.size());
    par::copy_out(attn.data(), d_attn, attn.size());

    double err = 0.0;
    for (size_t i = 0; i < attn.size(); ++i) err = std::max(err, std::fabs((double) attn[i] - ref[i]));
    // the device may take the WMMA kernel, the pools overload the matrix engines'
    const double rel = err / scale_out, tol = fma && fmt <= kF16 ? 1e-4 : 5e-3;
    std::printf("  %-34s %lld queries after %lld cells, vs FP64 %.3e of the scale (%s, at most %.0e) %s\n", name,
                (long long) T, (long long) pos0, rel, fma ? "the FMA kernel" : "the device's kernel", tol,
                rel <= tol ? "ok" : "FAIL");
    return rel <= tol ? 0 : 1;
}
}  // namespace

int main() {
    int bad = 0;
    // 45 queries: not a whole number of the FMA kernel's 4 nor the WMMA kernel's 16; 37 cells before them
    bad += check("Qwen3.5-9B (16/4 x 256)", 16, 4, 256, 37, 45);
    bad += check("pools FP16 (16/4 x 256)", 16, 4, 256, 37, 45, kF16);
    bad += check("pools INT8 (16/4 x 256)", 16, 4, 256, 37, 45, kI8);
    bad += check("pools Q4_0 (16/4 x 256)", 16, 4, 256, 37, 45, kQ4);
    bad += check("pools K8V4 (16/4 x 256)", 16, 4, 256, 37, 45, kK8V4);
    // a window of 96 cells (whole pages and not), 300 cells before 70 queries; a ring of 3 pages (192 cells: the
    // window and a chunk of 70, the page the window starts in)
    bad += check("PLaMo 2B window (24/4 x 128)", 24, 4, 128, 300, 70, kPlain, 96);
    bad += check("PLaMo 0.8B window (16/1 x 128)", 16, 1, 128, 300, 70, kPlain, 96);
    bad += check("PLaMo 8B window (32/4 x 128)", 32, 4, 128, 300, 70, kPlain, 96);
    bad += check("PLaMo 2B pools window (24/4 x 128)", 24, 4, 128, 300, 70, kF16, 96);
    bad += check("PLaMo 2B window, ring (24/4 x 128)", 24, 4, 128, 300, 70, kF16, 96, 3);
    std::printf("prompt_attn: %d failures\n", bad);
    return bad ? 1 : 0;
}
