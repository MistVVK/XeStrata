// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/dflash2_parity.cpp - the DFlash2 kernels (kernels/dflash2.hpp) against CPU references (GPU, synthetic, no
// model).  The quantization is checked byte for byte; the products and norms within fp32's rounding of a different
// summation order; the attention's planar output within one quantization step; the top-k and the selector's picks
// exactly.  Skipped (77) without a GPU.
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/draft_head.hpp"
#include "../kernels/parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;
namespace k = strata::kernels;

int g_failed = 0;
void report(const char* what, int64_t bad, double worst) {
    std::printf("  %-22s %lld out of bound, worst %.3g\n", what, (long long) bad, worst);
    g_failed += bad != 0;
}

std::mt19937 rng(7);   // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the inputs reproducible
float rnd(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

// ---- references
float bf16f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t bf16bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
uint16_t f16bits(float f) {
    // fp32 -> fp16 (round to nearest even), the pack's scale storage
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u, exp = (u >> 23) & 0xffu, man = u & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp > 142) return (uint16_t) (sign | 0x7c00u);
    if (exp < 113) return (uint16_t) sign;
    uint32_t e = exp - 127 + 15, m = man >> 13;
    const uint32_t r = man & 0x1fffu;
    if (r > 0x1000u || (r == 0x1000u && (m & 1))) { m++; if (m == 0x400u) { m = 0; e++; } }
    return (uint16_t) (sign | (e << 10) | m);
}
float f16f(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    uint32_t u;
    if (exp == 0) u = sign | (man ? 0x38000000u : 0u);   // subnormal: small enough to be 0 here
    else if (exp == 31) u = sign | 0x7f800000u | (man << 13);
    else u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

// ggml quantize_row_q8_0 / q4_0 of one row block (the pack tool's rounding)
void quant_q8_block(const float* x, uint8_t* out) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0f;
    const uint16_t db = f16bits(d);
    out[0] = (uint8_t) db; out[1] = (uint8_t) (db >> 8);
    const float inv = d != 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; ++i) out[2 + i] = (uint8_t) (int8_t) std::nearbyint(x[i] * inv);
}
void quant_q4_block(const float* x, uint8_t* out) {
    int idx = 0; float mx = 0;
    for (int i = 0; i < 32; ++i) if (std::fabs(x[i]) > std::fabs(mx)) { mx = x[i]; idx = i; }
    (void) idx;
    const float d = mx / -8.0f;
    const uint16_t db = f16bits(d);
    out[0] = (uint8_t) db; out[1] = (uint8_t) (db >> 8);
    const float inv = d != 0 ? 1.0f / d : 0.0f;
    int q[32];
    for (int i = 0; i < 32; ++i) q[i] = std::min(15, (int) (x[i] * inv + 8.5f));
    for (int i = 0; i < 16; ++i) out[2 + i] = (uint8_t) (q[i] | (q[i + 16] << 4));
}

// the planar activation quantization (df2_quant_rows' reference)
void quant_planar(const float* x, int K, int ncols, std::vector<int8_t>& xq, std::vector<float>& xd) {
    xq.assign((size_t) ncols * K, 0);
    xd.assign((size_t) ncols * (K / 32), 0);
    for (int c = 0; c < ncols; ++c)
        for (int b = 0; b < K / 32; ++b) {
            const float* row = x + (size_t) c * K + b * 32;
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(row[i]));
            const float d = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
            xd[(size_t) c * (K / 32) + b] = d;
            for (int i = 0; i < 32; ++i) xq[(size_t) c * K + b * 32 + i] = (int8_t) std::nearbyint(row[i] * inv);
        }
}

// ggml blocks of a random [n_out][n_in] matrix
std::vector<uint8_t> ggml_blocks(int n_out, int n_in, bool q4, std::vector<float>& w) {
    w.resize((size_t) n_out * n_in);
    for (auto& v : w) v = rnd(-0.5f, 0.5f);
    const int bs = q4 ? 18 : 34;
    std::vector<uint8_t> out((size_t) n_out * (n_in / 32) * bs);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < n_in / 32; ++b) {
            if (q4) quant_q4_block(w.data() + (size_t) r * n_in + b * 32, out.data() + ((size_t) r * (n_in / 32) + b) * bs);
            else quant_q8_block(w.data() + (size_t) r * n_in + b * 32, out.data() + ((size_t) r * (n_in / 32) + b) * bs);
        }
    return out;
}

// one dequantized ggml weight (the kernel reads the quantized blocks, so the reference must too)
float deq(const std::vector<uint8_t>& blocks, int n_in, bool q4, int r, int i) {
    const int b = i / 32, j = i % 32;
    const int bs = q4 ? 18 : 34;
    const uint8_t* blk = blocks.data() + ((size_t) r * (n_in / 32) + b) * bs;
    const float d = f16f((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
    if (q4) {
        const uint8_t byte = blk[2 + (j & 15)];
        const int q = j < 16 ? (byte & 15) : (byte >> 4);
        return (float) (q - 8) * d;
    }
    return (float) (int8_t) blk[2 + j] * d;
}

// dequantized ggml row dot the planar activations (the GEMM's reference, fp64 sums)
void gemm_ref(const std::vector<uint8_t>& blocks, int n_out, int n_in, bool q4, const std::vector<int8_t>& xq,
              const std::vector<float>& xd, int ncols, std::vector<float>& y) {
    y.assign((size_t) ncols * n_out, 0);
    for (int c = 0; c < ncols; ++c)
        for (int r = 0; r < n_out; ++r) {
            double acc = 0;
            for (int b = 0; b < n_in / 32; ++b) {
                double s = 0;
                for (int i = 0; i < 32; ++i)
                    s += (double) deq(blocks, n_in, q4, r, b * 32 + i) * xq[(size_t) c * n_in + b * 32 + i];
                acc += (double) xd[(size_t) c * (n_in / 32) + b] * s;
            }
            y[(size_t) c * n_out + r] = (float) acc;
        }
}

int check_rel(const float* got, const float* ref, size_t n, double tol, double& worst) {
    int64_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double) got[i] - ref[i]), bound = tol * std::max(1.0, std::fabs((double) ref[i]));
        worst = std::max(worst, e / bound);
        if (e > bound) ++bad;
    }
    return (int) bad;
}

// ---- the tests
void test_quant_rows(void* s) {
    const int K = 2560, ncols = 8;
    std::vector<float> x((size_t) ncols * K);
    for (auto& v : x) v = rnd(-3, 3);
    std::vector<int8_t> xq_ref; std::vector<float> xd_ref;
    quant_planar(x.data(), K, ncols, xq_ref, xd_ref);
    parity::Device dev;
    const float* dx = dev.upload(x);
    auto* dxq = dev.alloc<int8_t>(xq_ref.size());
    auto* dxd = dev.alloc<float>(xd_ref.size());
    k::df2_quant_rows(dx, K, ncols, dxq, dxd, s);
    parity::sync();
    std::vector<int8_t> xq(xq_ref.size()); std::vector<float> xd(xd_ref.size());
    parity::copy_out(xq.data(), dxq, xq.size());
    parity::copy_out(xd.data(), dxd, xd.size());
    const int bad_q = xq == xq_ref ? 0 : 1, bad_d = xd == xd_ref ? 0 : 1;
    report("quant_rows", bad_q + bad_d, bad_q + bad_d);
}

void test_gemm(void* s, bool q4) {
    struct Case { int n_out, n_in, ncols; };
    const Case cases[] = {{2560, 2560, 8}, {1024, 2560, 3}, {7680, 2560, 8}, {2560, 7680, 1}, {640, 2560, 8},
                          {256, 2560, 7}, {2560, 12800, 2}, {256, 512, 9}, {256, 512, 15}, {256, 512, 16}};
    for (const Case& c : cases) {
        std::vector<float> w;
        const std::vector<uint8_t> blocks = ggml_blocks(c.n_out, c.n_in, q4, w);
        std::vector<float> x((size_t) c.ncols * c.n_in);
        for (auto& v : x) v = rnd(-2, 2);
        std::vector<int8_t> xq; std::vector<float> xd;
        quant_planar(x.data(), c.n_in, c.ncols, xq, xd);
        std::vector<float> y_ref;
        gemm_ref(blocks, c.n_out, c.n_in, q4, xq, xd, c.ncols, y_ref);
        parity::Device dev;
        const uint8_t* dblocks = dev.upload(blocks);
        const size_t plane = q4 ? (size_t) c.n_out * c.n_in / 2 : (size_t) c.n_out * c.n_in;
        auto* dtile = dev.alloc<uint8_t>(plane + (size_t) c.n_out * (c.n_in / 32) * 2);
        if (q4) k::df2_q4_tile(dblocks, dtile, dtile + plane, c.n_in, (int64_t) c.n_out * (c.n_in / 32), s);
        else k::df2_q8_tile(dblocks, dtile, dtile + plane, c.n_in, (int64_t) c.n_out * (c.n_in / 32), s);
        const int8_t* dxq = dev.upload(xq);
        const float* dxd = dev.upload(xd);
        auto* dy = dev.alloc<float>(y_ref.size());
        k::Df2Mat mat{dtile, dy, c.n_out, q4};
        k::df2_gemm(&mat, 1, dxq, dxd, c.n_in, c.ncols, s);
        parity::sync();
        std::vector<float> y(y_ref.size());
        parity::copy_out(y.data(), dy, y.size());
        double worst = 0;
        const int bad = check_rel(y.data(), y_ref.data(), y.size(), 2e-5, worst);
        char name[64];
        std::snprintf(name, sizeof name, "gemm_%s %dx%d c%d", q4 ? "q4" : "q8", c.n_out, c.n_in, c.ncols);
        report(name, bad, worst);
    }
}

void test_conv(void* s) {
    const int rows = 8, H = 2560, group = 16, groups = H / group;
    std::vector<float> x((size_t) rows * H), dyn((size_t) rows * 640), base(2 * H), resid((size_t) rows * H);
    for (auto& v : x) v = rnd(-1, 1);
    for (auto& v : dyn) v = rnd(-0.2f, 0.2f);
    for (auto& v : base) v = rnd(-0.5f, 0.5f);
    for (auto& v : resid) v = rnd(-1, 1);
    std::vector<float> ref((size_t) rows * H);
    for (int b = 0; b < rows; ++b)
        for (int c = 0; c < H; ++c) {
            float acc = 0;
            for (int o = 0; o < 2 && b - o >= 0; ++o)
                acc += (base[o * H + c] + dyn[(size_t) b * 640 + o * groups + c / group]) * x[(size_t) (b - o) * H + c];
            ref[(size_t) b * H + c] = resid[(size_t) b * H + c] + acc;
        }
    parity::Device dev;
    const float *dx = dev.upload(x), *ddyn = dev.upload(dyn), *dbase = dev.upload(base), *dresid = dev.upload(resid);
    auto* dout = dev.alloc<float>(ref.size());
    k::df2_conv(dx, ddyn, 640, dbase, dresid, dout, rows, H, group, s);
    parity::sync();
    std::vector<float> out(ref.size());
    parity::copy_out(out.data(), dout, out.size());
    double worst = 0;
    report("conv", check_rel(out.data(), ref.data(), ref.size(), 2e-5, worst), worst);
}

void test_rms_swiglu(void* s) {
    const int rows = 8, N = 2560;
    const float eps = 1e-6f;
    std::vector<float> h((size_t) rows * N), gamma(N), gate((size_t) rows * N), up((size_t) rows * N);
    for (auto& v : h) v = rnd(-2, 2);
    for (auto& v : gamma) v = rnd(0.5f, 1.5f);
    for (auto& v : gate) v = rnd(-3, 3);
    for (auto& v : up) v = rnd(-2, 2);
    // rms_quant reference
    std::vector<float> xn_ref((size_t) rows * N);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < N; ++i) ss += (double) h[(size_t) r * N + i] * h[(size_t) r * N + i];
        const float scale = (float) (1.0 / std::sqrt(ss / N + eps));
        for (int i = 0; i < N; ++i) xn_ref[(size_t) r * N + i] = h[(size_t) r * N + i] * scale * gamma[i];
    }
    std::vector<int8_t> xq_ref; std::vector<float> xd_ref;
    quant_planar(xn_ref.data(), N, rows, xq_ref, xd_ref);
    parity::Device dev;
    const float *dh = dev.upload(h), *dgamma = dev.upload(gamma);
    auto* dxn = dev.alloc<float>(xn_ref.size());
    auto* dxq = dev.alloc<int8_t>(xq_ref.size());
    auto* dxd = dev.alloc<float>(xd_ref.size());
    k::df2_rms_quant(dh, dgamma, dxn, rows, N, eps, dxq, dxd, s);
    parity::sync();
    std::vector<float> xn(xn_ref.size()), xd(xd_ref.size());
    std::vector<int8_t> xq(xq_ref.size());
    parity::copy_out(xn.data(), dxn, xn.size());
    parity::copy_out(xq.data(), dxq, xq.size());
    parity::copy_out(xd.data(), dxd, xd.size());
    double worst = 0;
    int bad = check_rel(xn.data(), xn_ref.data(), xn.size(), 2e-5, worst);
    bad += xq == xq_ref ? 0 : 1;
    report("rms_quant", bad, worst);
    // swiglu_quant reference
    std::vector<float> sw((size_t) rows * N);
    for (size_t i = 0; i < sw.size(); ++i) sw[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    quant_planar(sw.data(), N, rows, xq_ref, xd_ref);
    const float *dgate = dev.upload(gate), *dup = dev.upload(up);
    k::df2_swiglu_quant(dgate, dup, rows, N, dxq, dxd, s);
    parity::sync();
    parity::copy_out(xq.data(), dxq, xq.size());
    parity::copy_out(xd.data(), dxd, xd.size());
    // silu's exp may differ by 1 ULP (sycl vs std), flipping one code at a rounding boundary: compare the
    // dequantized values within one quantization step (a draft only proposes; the verify window decides)
    int64_t sw_bad = 0; double sw_worst = 0;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < N / 32; ++b) {
            const float d = xd[(size_t) r * (N / 32) + b], dr = xd_ref[(size_t) r * (N / 32) + b];
            for (int i = 0; i < 32; ++i) {
                const size_t at = (size_t) r * N + b * 32 + i;
                const double e = std::fabs((double) xq[at] * d - (double) xq_ref[at] * dr);
                const double bound = 1.5 * std::max(std::fabs(d), std::fabs(dr)) + 1e-6;
                sw_worst = std::max(sw_worst, e / bound);
                if (e > bound) ++sw_bad;
            }
        }
    report("swiglu_quant", sw_bad, sw_worst);
}

void test_rope(void* s) {
    (void) s;
    const int rows = 8;
    const float theta = 1000000.0f;
    std::vector<int32_t> pos = {0, 1, 7, 2047, 4096, 32768, 100000, 262143};
    std::vector<float> table((size_t) rows * 128);
    k::df2_rope_table_cpu(pos.data(), rows, theta, table.data());
    int64_t bad = 0; double worst = 0;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < 64; ++i) {
            const double inv = std::pow((double) theta, -(double) (2 * i) / 128.0);
            double ang = (double) pos[r] * inv;
            const double two_pi = 6.283185307179586476925286766559;
            ang -= two_pi * std::nearbyint(ang / two_pi);
            const float c = (float) std::cos(ang), sn = (float) std::sin(ang);
            const double e1 = std::fabs(table[(r * 2 + 0) * 64 + i] - c), e2 = std::fabs(table[(r * 2 + 1) * 64 + i] - sn);
            worst = std::max(worst, std::max(e1, e2));
            if (e1 > 1e-6 || e2 > 1e-6) ++bad;
        }
    report("rope_table_cpu", bad, worst);
}

void test_norm_rope_kv(void* s) {
    const int rows = 8, NH = 32, NKV = 8, HD = k::kDf2Hd, slots = 2064;
    const float eps = 1e-6f, theta = 1000000.0f;
    std::vector<float> q((size_t) rows * NH * HD), kk((size_t) rows * NKV * HD), gq(HD), gk(HD);
    for (auto& v : q) v = rnd(-1, 1);
    for (auto& v : kk) v = rnd(-1, 1);
    for (auto& v : gq) v = rnd(0.5f, 1.5f);
    for (auto& v : gk) v = rnd(0.5f, 1.5f);
    std::vector<int32_t> pos = {5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> table((size_t) rows * 128);
    k::df2_rope_table_cpu(pos.data(), rows, theta, table.data());
    // reference: rms + rope per (row, head)
    auto norm_rope = [&](const float* in, const float* gamma, int heads, int r, int h, float* out) {
        const float* x = in + ((size_t) r * heads + h) * HD;
        double ss = 0;
        for (int d = 0; d < HD; ++d) ss += (double) x[d] * x[d];
        const float scale = (float) (1.0 / std::sqrt(ss / HD + eps));
        float v[128];
        for (int d = 0; d < HD; ++d) v[d] = x[d] * scale * gamma[d];
        for (int d = 0; d < HD; ++d) {
            const int i = d & 63;
            const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
            out[d] = d < 64 ? v[d] * c - v[d + 64] * sn : v[d] * c + v[d - 64] * sn;
        }
    };
    std::vector<float> qo_ref(q.size()), ko_ref(kk.size());
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < NH; ++h) norm_rope(q.data(), gq.data(), NH, r, h, qo_ref.data() + ((size_t) r * NH + h) * HD);
        for (int h = 0; h < NKV; ++h) norm_rope(kk.data(), gk.data(), NKV, r, h, ko_ref.data() + ((size_t) r * NKV + h) * HD);
    }
    parity::Device dev;
    const float *dq = dev.upload(q), *dk = dev.upload(kk), *dgq = dev.upload(gq), *dgk = dev.upload(gk);
    const float* dtable = dev.upload(table);
    auto* dqo = dev.alloc<float>(q.size());
    auto* dko = dev.alloc<float>(kk.size());
    k::df2_norm_rope_qk(dq, dk, dgq, dgk, dtable, dqo, dko, rows, NH, NKV, eps, s);
    parity::sync();
    std::vector<float> qo(q.size()), ko(kk.size());
    parity::copy_out(qo.data(), dqo, qo.size());
    parity::copy_out(ko.data(), dko, ko.size());
    double worst = 0;
    int bad = check_rel(qo.data(), qo_ref.data(), qo.size(), 2e-5, worst);
    bad += check_rel(ko.data(), ko_ref.data(), ko.size(), 2e-5, worst);
    report("norm_rope_qk", bad, worst);
    // kv_ring: the k above lands in the head-major ring at pos % slots, v copied, tags stamped
    std::vector<float> vin((size_t) rows * NKV * HD);
    for (auto& v : vin) v = rnd(-1, 1);
    auto* dkring = dev.alloc<float>((size_t) NKV * slots * HD);
    auto* dvring = dev.alloc<float>((size_t) NKV * slots * HD);
    auto* dtag = dev.alloc<int64_t>(slots);
    auto* dpos = dev.upload(pos);
    auto* depoch = dev.alloc<int32_t>(1);
    const int32_t epoch = 3;
    parity::copy_in(depoch, &epoch, 1);
    k::df2_kv_ring(dk, dev.upload(vin), dgk, dpos, dtable, dkring, dvring, slots, rows, NKV, eps, s, dtag, depoch);
    parity::sync();
    std::vector<float> kring((size_t) NKV * slots * HD), vring(kring.size());
    std::vector<int64_t> tag(slots);
    parity::copy_out(kring.data(), dkring, kring.size());
    parity::copy_out(vring.data(), dvring, vring.size());
    parity::copy_out(tag.data(), dtag, tag.size());
    bad = 0; worst = 0;
    for (int r = 0; r < rows; ++r) {
        const int slot = pos[r] % slots;
        for (int h = 0; h < NKV; ++h) {
            const size_t at = ((size_t) h * slots + slot) * HD;
            bad += check_rel(kring.data() + at, ko_ref.data() + ((size_t) r * NKV + h) * HD, HD, 2e-5, worst);
            for (int d = 0; d < HD; ++d)
                if (vring[at + d] != vin[((size_t) r * NKV + h) * HD + d]) ++bad;
        }
        if (tag[slot] != (((int64_t) epoch << 32) | pos[r])) ++bad;
    }
    report("kv_ring", bad, worst);
}

void test_attention(void* s, int rows) {
    const int NH = 32, NKV = k::kDf2Kv, HD = k::kDf2Hd;
    const int slots = 2064, window = 2048;
    const int anchor = 3000;   // a position past the window: the ring wraps
    std::vector<float> q((size_t) rows * NH * HD), kring((size_t) NKV * slots * HD), vring(kring.size());
    std::vector<float> kblk((size_t) rows * NKV * HD), vblk(kblk.size());
    for (auto& v : q) v = rnd(-0.5f, 0.5f);
    for (auto& v : kring) v = rnd(-0.5f, 0.5f);
    for (auto& v : vring) v = rnd(-0.5f, 0.5f);
    for (auto& v : kblk) v = rnd(-0.5f, 0.5f);
    for (auto& v : vblk) v = rnd(-0.5f, 0.5f);
    const int lo = std::max(0, anchor - (window - 1));
    std::vector<int64_t> tag(slots, -1);
    for (int j = lo; j < anchor; ++j) tag[j % slots] = ((int64_t) 3 << 32) | j;
    // a few slots hold the wrong row (a stale epoch): the attention must skip them
    for (int j = lo; j < anchor; j += 97) tag[j % slots] = ((int64_t) 2 << 32) | j;
    // reference: per (row, head) softmax over the visible keys
    std::vector<float> ref((size_t) rows * NH * HD);
    const float scale = 1.0f / std::sqrt((float) HD);
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / k::kDf2Gqa;
            const float* qr = q.data() + ((size_t) r * NH + h) * HD;
            std::vector<float> scores;
            std::vector<const float*> keys, vals;
            for (int j = lo; j < anchor; ++j) {
                if (anchor + r - j >= window) continue;
                if (tag[j % slots] != (((int64_t) 3 << 32) | j)) continue;
                scores.push_back(0);
                keys.push_back(kring.data() + ((size_t) kvh * slots + j % slots) * HD);
                vals.push_back(vring.data() + ((size_t) kvh * slots + j % slots) * HD);
            }
            for (int b = 0; b < rows; ++b) {
                scores.push_back(0);
                keys.push_back(kblk.data() + ((size_t) b * NKV + kvh) * HD);
                vals.push_back(vblk.data() + ((size_t) b * NKV + kvh) * HD);
            }
            float m = -INFINITY;
            for (size_t t = 0; t < keys.size(); ++t) {
                double acc = 0;
                for (int d = 0; d < HD; ++d) acc += (double) qr[d] * keys[t][d];
                scores[t] = (float) acc * scale;
                m = std::max(m, scores[t]);
            }
            double l = 0;
            for (auto& sc : scores) { sc = std::exp(sc - m); l += sc; }
            for (int d = 0; d < HD; ++d) {
                double acc = 0;
                for (size_t t = 0; t < keys.size(); ++t) acc += (double) scores[t] * vals[t][d];
                ref[((size_t) r * NH + h) * HD + d] = (float) (acc / l);
            }
        }
    parity::Device dev;
    const float *dq = dev.upload(q), *dkring = dev.upload(kring), *dvring = dev.upload(vring);
    const float *dkblk = dev.upload(kblk), *dvblk = dev.upload(vblk);
    const int64_t* dtag = dev.upload(tag);
    auto* danchor = dev.alloc<int32_t>(1);
    auto* depoch = dev.alloc<int32_t>(1);
    const int32_t epoch = 3;
    parity::copy_in(danchor, &anchor, 1);
    parity::copy_in(depoch, &epoch, 1);
    auto* dpart = dev.alloc<float>((size_t) k::kDf2PartChunks * NKV * rows * 4 * 130);
    const int K = NH * HD;
    auto* dxq = dev.alloc<int8_t>((size_t) rows * K);
    auto* dxd = dev.alloc<float>((size_t) rows * (K / 32));
    auto run = [&](int variant) {
        k::df2_attention_variant(variant);
        k::df2_attention(dq, dkring, dvring, dkblk, dvblk, danchor, slots, window, dpart, dxq, dxd, s, dtag, depoch, rows);
        parity::sync();
        std::vector<int8_t> xq((size_t) rows * K);
        std::vector<float> xd((size_t) rows * (K / 32));
        parity::copy_out(xq.data(), dxq, xq.size());
        parity::copy_out(xd.data(), dxd, xd.size());
        k::df2_attention_variant(0);
        return std::pair<std::vector<int8_t>, std::vector<float>>(std::move(xq), std::move(xd));
    };
    const auto v1 = run(1);
    const auto v2 = rows == k::kDf2Rows ? run(2) : v1;
    if (rows == k::kDf2Rows)
        report("attention variants (bytes)", std::memcmp(v1.first.data(), v2.first.data(), v1.first.size()) != 0 ||
                                                std::memcmp(v1.second.data(), v2.second.data(), v1.second.size() * sizeof(float)) != 0,
               0);
    // dequantize and compare within one quantization step (the fp order differs, a code may differ by 1)
    auto check = [&](const char* name, const std::vector<int8_t>& xq, const std::vector<float>& xd) {
        int64_t bad = 0; double worst = 0;
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < K / 32; ++b) {
                const float d = xd[(size_t) r * (K / 32) + b];
                for (int i = 0; i < 32; ++i) {
                    const size_t at = (size_t) r * K + b * 32 + i;
                    const float got = xq[at] * d;
                    const double e = std::fabs((double) got - ref[at]), bound = 1.5 * std::fabs(d) + 1e-5;
                    worst = std::max(worst, e / bound);
                    if (e > bound) ++bad;
                }
            }
        report(name, bad, worst);
    };
    check(rows == k::kDf2Rows ? "attention (8-query groups)" : "attention (16 rows)", v1.first, v1.second);
    if (rows == k::kDf2Rows) check("attention (256 work-items)", v2.first, v2.second);
}

void test_topk(void* s) {
    const int rows = 7, n = 55328;
    std::vector<float> logits((size_t) rows * n);
    for (auto& v : logits) v = rnd(-10, 10);
    std::vector<int32_t> idx_map(n);
    for (int i = 0; i < n; ++i) idx_map[i] = (i * 7 + 3) % n;   // a shuffled vocabulary map
    std::vector<float> val_ref((size_t) rows * k::kDf2TopK);
    std::vector<int32_t> tok_ref((size_t) rows * k::kDf2TopK);
    for (int r = 0; r < rows; ++r) {
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + k::kDf2TopK, order.end(), [&](int a, int b) {
            const float va = logits[(size_t) r * n + a], vb = logits[(size_t) r * n + b];
            return va > vb || (va == vb && a < b);
        });
        for (int j = 0; j < k::kDf2TopK; ++j) {
            val_ref[(size_t) r * k::kDf2TopK + j] = logits[(size_t) r * n + order[j]];
            tok_ref[(size_t) r * k::kDf2TopK + j] = idx_map[order[j]];
        }
    }
    parity::Device dev;
    const float* dlogits = dev.upload(logits);
    const int32_t* dmap = dev.upload(idx_map);
    auto* dscratch = dev.alloc<uint32_t>((size_t) rows * k::kDf2TopBlocks * k::kDf2TopK * 2);
    auto* dval = dev.alloc<float>(val_ref.size());
    auto* dtok = dev.alloc<int32_t>(tok_ref.size());
    k::df2_topk(dlogits, n, dmap, rows, dscratch, dval, dtok, s);
    parity::sync();
    std::vector<float> val(val_ref.size());
    std::vector<int32_t> tok(tok_ref.size());
    parity::copy_out(val.data(), dval, val.size());
    parity::copy_out(tok.data(), dtok, tok.size());
    report("topk", (val == val_ref ? 0 : 1) + (tok == tok_ref ? 0 : 1), 0);
}

// The draft head selector: equal scores may choose any distinct IDs, including across chunks.
void test_shared_topk(void* s) {
    constexpr int n = 8193;
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = (float) (i % 257);
    std::vector<float> expected = v;
    std::sort(expected.begin(), expected.end(), std::greater<float>());
    parity::Device dev;
    const float* dv = dev.upload(v);
    for (int count : {1, 16, 64}) {
        auto* ids = dev.alloc<int32_t>(count);
        auto* scratch = dev.alloc<uint8_t>(k::topk_scratch_bytes(n, count));
        k::topk(dv, n, count, ids, scratch, s);
        parity::sync();
        std::vector<int32_t> got((size_t) count);
        parity::copy_out(got.data(), ids, got.size());
        int64_t bad = 0;
        std::vector<bool> seen(n, false);
        for (int i = 0; i < count; ++i) {
            const int id = got[(size_t) i];
            if (id < 0 || id >= n || seen[(size_t) id]) { ++bad; continue; }
            seen[(size_t) id] = true;
            bad += v[(size_t) id] != expected[(size_t) i];
        }
        char name[64];
        std::snprintf(name, sizeof name, "shared topk %d ties", count);
        report(name, bad, 0);
    }
}

void test_select(void* s, bool q4) {
    const int rows = 7, rank = 256, vocab = 4096;
    std::vector<float> hp((size_t) rows * rank), unary((size_t) rows * k::kDf2TopK);
    for (auto& v : hp) v = rnd(-0.5f, 0.5f);
    for (auto& v : unary) v = rnd(-2, 2);
    std::vector<int32_t> tok((size_t) rows * k::kDf2TopK);
    for (auto& v : tok) v = (int) (rnd(0, (float) vocab));
    std::vector<float> pred_w((size_t) vocab * rank), succ_w((size_t) vocab * rank);
    for (auto& v : pred_w) v = rnd(-0.5f, 0.5f);
    for (auto& v : succ_w) v = rnd(-0.5f, 0.5f);
    // the codebooks: bf16 rows, or Q4_0 rows in block order
    std::vector<uint16_t> pred_cb((size_t) vocab * rank), succ_cb((size_t) vocab * rank);
    if (q4) {
        std::vector<uint8_t> pb((size_t) vocab * (rank / 32) * 18), sb(pb.size());
        for (int r = 0; r < vocab; ++r)
            for (int b = 0; b < rank / 32; ++b) {
                quant_q4_block(pred_w.data() + (size_t) r * rank + b * 32, pb.data() + ((size_t) r * (rank / 32) + b) * 18);
                quant_q4_block(succ_w.data() + (size_t) r * rank + b * 32, sb.data() + ((size_t) r * (rank / 32) + b) * 18);
            }
        std::memcpy(pred_cb.data(), pb.data(), pb.size());
        std::memcpy(succ_cb.data(), sb.data(), sb.size());
        // the reference reads the quantized values back
        auto deq = [&](const std::vector<uint8_t>& bytes, int r, int v) {
            const uint8_t* blk = bytes.data() + (size_t) r * (rank / 32) * 18 + (v / 32) * 18;
            const float d = f16f((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
            const int j = v % 32;
            const int q = j < 16 ? (blk[2 + j] & 15) : (blk[2 + j - 16] >> 4);
            return (float) (q - 8) * d;
        };
        for (int r = 0; r < vocab; ++r)
            for (int v = 0; v < rank; ++v) {
                pred_w[(size_t) r * rank + v] = deq(pb, r, v);
                succ_w[(size_t) r * rank + v] = deq(sb, r, v);
            }
    } else {
        for (size_t i = 0; i < pred_w.size(); ++i) {
            pred_cb[i] = bf16bits(pred_w[i]);
            succ_cb[i] = bf16bits(succ_w[i]);
            pred_w[i] = bf16f(pred_cb[i]);
            succ_w[i] = bf16f(succ_cb[i]);
        }
    }
    // reference walk
    const int32_t anchor = 17;
    std::vector<int32_t> tok_ref(rows);
    std::vector<float> prob_ref(rows);
    int prev = anchor;
    for (int i = 0; i < rows; ++i) {
        float sc[k::kDf2TopK];
        for (int j = 0; j < k::kDf2TopK; ++j) {
            const int t = tok[(size_t) i * k::kDf2TopK + j];
            double acc = 0;
            for (int v = 0; v < rank; ++v)
                acc += (double) succ_w[(size_t) t * rank + v] * (pred_w[(size_t) prev * rank + v] * hp[(size_t) i * rank + v]);
            sc[j] = unary[(size_t) i * k::kDf2TopK + j] + (float) acc;
        }
        int bj = 0;
        for (int j = 1; j < k::kDf2TopK; ++j) if (sc[j] > sc[bj]) bj = j;
        double z = 0;
        for (int j = 0; j < k::kDf2TopK; ++j) z += std::exp((double) sc[j] - sc[bj]);
        tok_ref[i] = tok[(size_t) i * k::kDf2TopK + bj];
        prob_ref[i] = (float) (1.0 / z);
        prev = tok_ref[i];
    }
    parity::Device dev;
    const float *dhp = dev.upload(hp), *dunary = dev.upload(unary);
    const int32_t* dtok = dev.upload(tok);
    const uint16_t* dpred = dev.upload(pred_cb);
    const uint16_t* dsucc = dev.upload(succ_cb);
    auto* danchor = dev.alloc<int32_t>(1);
    parity::copy_in(danchor, &anchor, 1);
    auto* dout_tok = dev.alloc<int32_t>(rows);
    auto* dout_prob = dev.alloc<float>(rows);
    k::df2_select(dhp, dunary, dtok, dpred, dsucc, rank, rows, danchor, dout_tok, dout_prob, s, q4);
    parity::sync();
    std::vector<int32_t> out_tok(rows);
    std::vector<float> out_prob(rows);
    parity::copy_out(out_tok.data(), dout_tok, rows);
    parity::copy_out(out_prob.data(), dout_prob, rows);
    double worst = 0;
    const int bad = (out_tok == tok_ref ? 0 : 1) + check_rel(out_prob.data(), prob_ref.data(), rows, 1e-4, worst);
    report(q4 ? "select_q4" : "select_bf16", bad, worst);
}

void test_taps(void* s) {
    const int rows = 8, hc = 4;
    const int64_t N = 2560;
    std::vector<float> R((size_t) rows * hc * N), bo((size_t) rows * N), inj((size_t) rows * hc);
    for (auto& v : R) v = rnd(-1, 1);
    for (auto& v : bo) v = rnd(-1, 1);
    for (auto& v : inj) v = rnd(-2, 2);
    std::vector<float> ref_gr((size_t) rows * N), ref_rows((size_t) rows * N);
    for (int r = 0; r < rows; ++r)
        for (int64_t d = 0; d < N; ++d) {
            double rs = 0, ws = 0, plain = 0;
            for (int c = 0; c < hc; ++c) {
                const float x = inj[(size_t) r * hc + c] / (float) hc;
                ws += 2.0 / (1.0 + std::exp(-(double) x));
                rs += R[((size_t) r * hc + c) * N + d];
                plain += R[((size_t) r * hc + c) * N + d];
            }
            ref_gr[(size_t) r * N + d] = (float) ((rs + bo[(size_t) r * N + d] * ws) / hc);
            ref_rows[(size_t) r * N + d] = (float) (plain / hc);
        }
    parity::Device dev;
    const float *dR = dev.upload(R), *dbo = dev.upload(bo), *dinj = dev.upload(inj);
    auto* dout = dev.alloc<float>(ref_gr.size());
    k::df2_tap_gr(dR, hc * N, dbo, N, dinj, hc, dout, N, rows, N, hc, s);
    parity::sync();
    std::vector<float> out(ref_gr.size());
    parity::copy_out(out.data(), dout, out.size());
    double worst = 0;
    report("tap_gr", check_rel(out.data(), ref_gr.data(), out.size(), 2e-5, worst), worst);
    k::df2_tap_rows(dR, dout, N, N, hc, rows, s);
    parity::sync();
    parity::copy_out(out.data(), dout, out.size());
    report("tap_rows", check_rel(out.data(), ref_rows.data(), out.size(), 2e-5, worst), worst);
}

}  // namespace

int main() {
    try { core::Runtime::get(); } catch (const std::exception& e) {
        std::printf("dflash2_parity: no GPU (%s) - skipped\n", e.what());
        return 77;
    }
    void* s = parity::stream();
    test_quant_rows(s);
    test_gemm(s, true);
    test_gemm(s, false);
    test_conv(s);
    test_rms_swiglu(s);
    test_rope(s);
    test_norm_rope_kv(s);
    test_attention(s, k::kDf2Rows);
    test_attention(s, 16);
    test_topk(s);
    test_shared_topk(s);
    test_select(s, false);
    test_select(s, true);
    test_taps(s);
    std::printf("dflash2_parity: %s\n", g_failed ? "FAILED" : "all within bound");
    return g_failed ? 1 : 0;
}