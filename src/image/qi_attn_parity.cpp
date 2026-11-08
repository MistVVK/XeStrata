// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn_parity.cpp - the image attention (qi_attention, its 16-bit path, and qi_attention_i8) against an
// FP64 host reference (GPU, synthetic, no model):
//
//   Qwen-Image-2.1's 32 heads of 128.  Short rows: 64 text rows (their keys up to themselves, the 16-bit path's
//   alone) before 512 image rows, which see every key; Q and K about 1 a value, K with a mean on some dims (int8's
//   smoothing takes it off), V about 1;
//   1. the 16-bit path's RMS error is within 1e-2 of the output's RMS;
//   2. the int8 kernel's (where it is built for the GPU; skipped elsewhere) within three times the error its rounding
//      should make (int8_bound), an estimate from the inputs; a head or a key out of place would be far past both
//      (the reference against itself with the heads shifted is checked to be);
//   3. the int8 kernel row by row (row_bound), on these rows and on long ones: an edit's 8192 keys, the last 4096 the
//      image rows (64 of them checked), with scores spread as Qwen-Image-2.1-Turbo's are (long_inputs).  An RMS over
//      every row misses an error that only some rows make, or that a long row's many small probabilities make.
//
//   qi_attn_parity --bench T [N_TEXT]   times both on T rows' image rows (no reference; not a CTest case)
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "../kernels/parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace si = strata::image;

namespace {
namespace par = strata::parity;

constexpr int64_t H = 32, D = 128, C = H * D;

template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

void inputs(int64_t T, std::vector<uint16_t>& q, std::vector<uint16_t>& kk, std::vector<uint16_t>& v) {
    std::mt19937 rng(1201);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    std::normal_distribution<float> nd(0.f, 1.f);
    q.resize((size_t) (T * C));
    kk.resize(q.size());
    v.resize(q.size());
    for (auto& x : q) x = k::f16_from_f32(nd(rng));
    for (size_t i = 0; i < kk.size(); ++i) kk[i] = k::f16_from_f32(nd(rng) + (i % D < 8 ? 2.f : 0.f));
    for (auto& x : v) x = k::f16_from_f32(nd(rng));
}

// Long rows' inputs, their scores spread as Qwen-Image-2.1-Turbo's (its edit at 1024x1024, seed 11, Q and K of
// blocks 0, 10, 20 and 31 of the 4th step and block 15 of the 1st, 32 rows a head, 2026-10-27): a row's scores spread
// by 1.3 to 5 (median of the heads), the share of its probability under half of int8's step 4 to 9% against its
// largest and 0.1 to 0.5% against each 32 keys' largest, n_eff 19 to 28.  Normal scores of that spread would leave a
// key or two: these are a bulk (Q 1.75 a value) less a per-key offset, exponential, along 32 of the dims (spread 4.4,
// 10 to 15% and 0.7% under the steps, n_eff about 300).  V follows the offset on those dims (a key's V goes with how
// far its score is down, as the model's do: P rounded against the row's largest, dropping those keys, made images
// LPIPS 0.09 to 0.27 apart); a V independent of its key would hide that loss, whatever the probabilities.
void long_inputs(int64_t T, std::vector<uint16_t>& q, std::vector<uint16_t>& kk, std::vector<uint16_t>& v) {
    constexpr int64_t TAIL = 32;
    constexpr double QS = 1.75, G = 2.0, LAM = 4.0;   // Q's scale, Q on the tail dims, the offset's mean in score units
    const double b = LAM * std::sqrt((double) D) / ((double) TAIL * G);
    std::mt19937 rng(1202);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    std::normal_distribution<float> nd(0.f, 1.f);
    std::exponential_distribution<float> ex(1.f);
    q.resize((size_t) (T * C));
    kk.resize(q.size());
    v.resize(q.size());
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            const float e = ex(rng);
            for (int64_t i = 0; i < D; ++i) {
                const size_t x = (size_t) (t * C + h * D + i);
                q[x] = k::f16_from_f32(i < TAIL ? (float) G : (float) QS * nd(rng));
                kk[x] = k::f16_from_f32(nd(rng) - (i < TAIL ? (float) b * e : 0.f));
                v[x] = k::f16_from_f32(nd(rng) + (i < TAIL ? 0.5f * e : 0.f));
            }
        }
}

// The RMS error the int8 kernel's rounding should make on these inputs, from its scales: a value rounded to a step of
// s (its row's or its column's largest over 127) is off by s / sqrt(12) on average.  The score q . k / sqrt(D) takes
// sqrt(D) such errors of Q times K's values and of K times Q's: sigma_s = sqrt(rms_q^2 e_k^2 + rms_k^2 e_q^2) (K less
// its mean, as the kernel takes it), a probability off by sigma_s relative; V's own rounding e_v; P's, 127 steps of the
// chunk's largest, 1 / (127 sqrt(12)) relative.  An output averages these independent errors over its keys with their
// probabilities as weights: their sizes times sqrt(sum p^2) = 1 / sqrt(n_eff), n_eff the keys an output effectively
// reads (from the FP64 reference's probabilities, `inv_neff` their mean of sum p^2), each times V's RMS.
// The mean square error a key's Q, K and V rounding puts on an output, before the average over its keys (sigma_s^2 times
// V's mean square, and V's own rounding), and V's mean square.
struct Unit {
    double qkv, rv;
};
Unit int8_unit(const std::vector<uint16_t>& qh, const std::vector<uint16_t>& kh, const std::vector<uint16_t>& vh,
               int64_t T) {
    std::vector<double> mean((size_t) C, 0.0), vmax((size_t) C, 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            mean[(size_t) c] += k::f32_from_f16(kh[(size_t) (t * C + c)]) / (double) T;
            vmax[(size_t) c] = std::max(vmax[(size_t) c], (double) std::fabs(k::f32_from_f16(vh[(size_t) (t * C + c)])));
        }
    double eq = 0.0, ek = 0.0, rq = 0.0, rk = 0.0, rv = 0.0;   // mean squares
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            double mq = 0.0, mk = 0.0;
            for (int64_t i = 0; i < D; ++i) {
                const size_t e = (size_t) (t * C + h * D + i);
                const double qv = k::f32_from_f16(qh[e]), kv = k::f32_from_f16(kh[e]) - mean[(size_t) (h * D + i)];
                mq = std::max(mq, std::fabs(qv));
                mk = std::max(mk, std::fabs(kv));
                rq += qv * qv;
                rk += kv * kv;
                rv += (double) k::f32_from_f16(vh[e]) * k::f32_from_f16(vh[e]);
            }
            eq += (mq / 127) * (mq / 127) / 12 * D;
            ek += (mk / 127) * (mk / 127) / 12 * D;
        }
    const double n = (double) (T * C);
    eq /= n, ek /= n, rq /= n, rk /= n, rv /= n;
    double ev = 0.0;
    for (double m : vmax) ev += (m / 127) * (m / 127) / 12 / (double) C;
    return {(rq * ek + rk * eq) * rv + ev, rv};
}

double int8_bound(const Unit& u, double inv_neff) {
    const double ep = 1.0 / (127.0 * std::sqrt(12.0));
    return std::sqrt((u.qkv + ep * ep * u.rv) * inv_neff);
}

// Row by row: a (row, head)'s output against FP64, within three times what its rounding should make there.  Q's, K's
// and V's, as int8_unit's averaged by the row's own sum p^2; P's not estimated but made: the row's FP64 probabilities
// rounded as the kernel rounds them (127 steps of each chunk's largest, qi_attention_i8_chunk), the output they give
// against the exact one.  A row whose small probabilities the kernel drops more than that is past it.  Returns the
// failures; prints the largest error relative to its bound and to its output.
int row_bound(const char* name, const std::vector<uint16_t>& qh, const std::vector<uint16_t>& kh,
              const std::vector<uint16_t>& vh, int64_t T, const std::vector<int64_t>& rows,
              const std::vector<double>& got, const Unit& u) {
    const int64_t ch = si::qi_attention_i8_chunk();
    double worst = 0.0, worst_rel = 0.0;
    int64_t wt = 0, wh = 0;
    std::vector<double> sc((size_t) T), o((size_t) D), oh((size_t) D);
    for (int64_t t : rows)
        for (int64_t h = 0; h < H; ++h) {
            double m = -1e300;
            for (int64_t j = 0; j < T; ++j) {
                double d = 0.0;
                for (int64_t i = 0; i < D; ++i)
                    d += (double) k::f32_from_f16(qh[(size_t) (t * C + h * D + i)]) * k::f32_from_f16(kh[(size_t) (j * C + h * D + i)]);
                sc[(size_t) j] = d / std::sqrt((double) D);
                m = std::max(m, sc[(size_t) j]);
            }
            double l = 0.0, lh = 0.0, p2 = 0.0;
            for (auto& x : sc) { x = std::exp(x - m); l += x; }
            std::fill(o.begin(), o.end(), 0.0);
            std::fill(oh.begin(), oh.end(), 0.0);
            for (int64_t j0 = 0; j0 < T; j0 += ch) {
                const int64_t j1 = std::min(T, j0 + ch);
                double cm = 0.0;
                for (int64_t j = j0; j < j1; ++j) cm = std::max(cm, sc[(size_t) j]);
                for (int64_t j = j0; j < j1; ++j) {
                    const double p = sc[(size_t) j], ph = std::rint(p / cm * 127.0) * cm / 127.0;
                    p2 += (p / l) * (p / l);
                    lh += ph;
                    for (int64_t i = 0; i < D; ++i) {
                        const double vv = k::f32_from_f16(vh[(size_t) (j * C + h * D + i)]);
                        o[(size_t) i] += p * vv;
                        oh[(size_t) i] += ph * vv;
                    }
                }
            }
            double e = 0.0, ep = 0.0, r = 0.0;
            for (int64_t i = 0; i < D; ++i) {
                const double x = o[(size_t) i] / l, d = got[(size_t) (t * C + h * D + i)] - x;
                e += d * d;
                ep += (oh[(size_t) i] / lh - x) * (oh[(size_t) i] / lh - x);
                r += x * x;
            }
            const double bound = 3.0 * std::sqrt(u.qkv * p2 + ep / (double) D);
            const double ratio = std::sqrt(e / (double) D) / bound;
            if (ratio > worst) worst = ratio, wt = t, wh = h;
            worst_rel = std::max(worst_rel, std::sqrt(e / r));
        }
    std::printf("  int8 kernel, %-22s the largest error %.2f of its bound (row %lld, head %lld; %.2e of its output) %s\n",
                name, worst, (long long) wt, (long long) wh, worst_rel, worst <= 1.0 ? "ok" : "FAIL");
    return worst > 1.0;
}

// The long rows (long_inputs): 8192 keys, the int8 kernel on the last 4096 rows, 64 of them checked row by row
int check_long(par::Device& dev) {
    const int64_t T = 8192, r0 = 4096;
    std::vector<uint16_t> qh, kh, vh;
    long_inputs(T, qh, kh, vh);
    uint16_t *q = up(dev, qh), *kd = up(dev, kh), *v = up(dev, vh);
    uint16_t* out = dev.alloc<uint16_t>((size_t) (T * C));
    if (!si::qi_attention_i8(q, kd, v, T, H, r0, 0, out, nullptr)) {
        std::printf("  int8 kernel, long rows: its buffers do not fit (FAIL)\n");
        return 1;
    }
    std::vector<uint16_t> g((size_t) (T * C));
    par::copy_out(g.data(), out, g.size());
    std::vector<double> got(g.size());
    for (size_t i = 0; i < g.size(); ++i) got[i] = k::f32_from_f16(g[i]);
    std::vector<int64_t> rows;
    rows.reserve(64);
    for (int64_t i = 0; i < 64; ++i) rows.push_back(r0 + i * (T - r0) / 64 + i % 7);
    return row_bound("long rows (64 of 4096):", qh, kh, vh, T, rows, got, int8_unit(qh, kh, vh, T));
}

int check() {
    par::Device dev;
    const int64_t n_text = 64, T = n_text + 512;
    std::vector<uint16_t> qh, kh, vh;
    inputs(T, qh, kh, vh);
    // FP64 reference: a text row t sees keys 0 .. t, an image row all T
    std::vector<double> ref((size_t) (T * C), 0.0);
    double inv_neff = 0.0;   // the image rows' mean of sum p^2
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            const int64_t keys = t < n_text ? t + 1 : T;
            std::vector<double> sc((size_t) keys);
            double m = -1e300;
            for (int64_t j = 0; j < keys; ++j) {
                double d = 0.0;
                for (int64_t i = 0; i < D; ++i)
                    d += (double) k::f32_from_f16(qh[(size_t) (t * C + h * D + i)]) * k::f32_from_f16(kh[(size_t) (j * C + h * D + i)]);
                sc[(size_t) j] = d / std::sqrt((double) D);
                m = std::max(m, sc[(size_t) j]);
            }
            double l = 0.0, p2 = 0.0;
            for (auto& x : sc) { x = std::exp(x - m); l += x; }
            for (double x : sc) p2 += (x / l) * (x / l);
            if (t >= n_text) inv_neff += p2 / (double) ((T - n_text) * H);
            for (int64_t j = 0; j < keys; ++j)
                for (int64_t i = 0; i < D; ++i)
                    ref[(size_t) (t * C + h * D + i)] += sc[(size_t) j] / l * k::f32_from_f16(vh[(size_t) (j * C + h * D + i)]);
        }
    double ref_rms = 0.0;
    for (int64_t i = 0; i < T * C; ++i) ref_rms += ref[(size_t) i] * ref[(size_t) i];
    ref_rms = std::sqrt(ref_rms / (double) (T * C));
    uint16_t *q = up(dev, qh), *kd = up(dev, kh), *v = up(dev, vh);
    uint16_t* out = dev.alloc<uint16_t>((size_t) (T * C));
    // the RMS error over rows r0 .. T - 1 of `got` (head h's column taken from head h + shift there)
    auto rms_err = [&](const std::vector<double>& got, int64_t r0, int64_t shift) {
        double e = 0.0;
        for (int64_t t = r0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t i = 0; i < D; ++i) {
                    const double d = got[(size_t) (t * C + (h + shift) % H * D + i)] - ref[(size_t) (t * C + h * D + i)];
                    e += d * d;
                }
        return std::sqrt(e / (double) ((T - r0) * C));
    };
    auto fetch = [&] {
        std::vector<uint16_t> g((size_t) (T * C));
        par::copy_out(g.data(), out, g.size());
        std::vector<double> d(g.size());
        for (size_t i = 0; i < g.size(); ++i) d[i] = k::f32_from_f16(g[i]);
        return d;
    };
    int fails = 0;
    si::qi_attention(q, kd, v, T, H, D, n_text, out, nullptr);
    const double e16 = rms_err(fetch(), 0, 0) / ref_rms;
    std::printf("  16-bit path (%s)          RMS error %.3e of the output's (at most 1e-02) %s\n", si::qi_attention_path(),
                e16, e16 <= 1e-2 ? "ok" : "FAIL");
    fails += e16 > 1e-2;
    const Unit unit = int8_unit(qh, kh, vh, T);
    const double bound = 3.0 * int8_bound(unit, inv_neff) / ref_rms;
    // a head out of place: the reference against itself with its heads shifted by one is far past the bound
    const double swapped = rms_err(ref, n_text, 1) / ref_rms;
    std::printf("  a head out of place                RMS error %.3e (past the int8 bound %.3e) %s\n", swapped, bound,
                swapped > 10 * bound ? "ok" : "FAIL");
    fails += swapped <= 10 * bound;
    if (si::qi_attention_i8(q, kd, v, T, H, n_text, 0, out, nullptr)) {
        const std::vector<double> got = fetch();
        const double e8 = rms_err(got, n_text, 0) / ref_rms;
        std::printf("  int8 kernel, the image rows        RMS error %.3e of the output's (at most %.3e) %s\n", e8, bound,
                    e8 <= bound ? "ok" : "FAIL");
        fails += e8 > bound;
        std::vector<int64_t> rows;
        for (int64_t t = n_text; t < T; ++t) rows.push_back(t);
        fails += row_bound("512 short rows:", qh, kh, vh, T, rows, got, unit);
        fails += check_long(dev);
    } else {
        std::printf("  int8 kernel: not on this GPU (skipped)\n");
    }
    std::printf("qi_attn: %d failures\n", fails);
    return fails ? 1 : 0;
}

int bench(int64_t T, int64_t n_text) {
    par::Device dev;
    std::vector<uint16_t> qh, kh, vh;
    inputs(T, qh, kh, vh);
    uint16_t *q = up(dev, qh), *kd = up(dev, kh), *v = up(dev, vh);
    uint16_t* out = dev.alloc<uint16_t>((size_t) (T * C));
    // the median of 21 runs after one (a run's time on this GPU varies by a fifth from one to the next)
    auto ms = [&](auto&& f) {
        f();
        std::vector<double> t;
        for (int i = 0; i < 21; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            f();
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        return t[10];
    };
    // the image rows in 16 bits: the GPU maker's kernel where it takes them (as qi_attention would), else qi_attention
    const bool maker = si::qi_sdpa(q, kd, v, n_text, T, T, H, D, out, nullptr);
    uint16_t one = 0;   // read back to wait for the maker's kernel, which returns as it is queued
    const double f16 = maker ? ms([&] {
        si::qi_sdpa(q, kd, v, n_text, T, T, H, D, out, nullptr);
        par::copy_out(&one, out, 1);
    })
                             : ms([&] { si::qi_attention(q, kd, v, T, H, D, n_text, out, nullptr); });
    std::printf("%lld rows (%lld before the image rows): 16-bit %s %.3f ms", (long long) T, (long long) n_text,
                maker ? "(the maker's kernel, the image rows)" : "(qi_attention, every row)", f16);
    if (si::qi_attention_i8(q, kd, v, T, H, n_text, 0, out, nullptr))
        std::printf(", int8 kernel (the image rows) %.3f ms\n",
                    ms([&] { si::qi_attention_i8(q, kd, v, T, H, n_text, 0, out, nullptr); }));
    else
        std::printf(", int8 kernel not on this GPU\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // the 16-bit path as qi_attention takes it without the int8 kernel (which this program calls by itself)
#if defined(_WIN32)
    _putenv_s("XESTRATA_IMAGE_ATTN_I8", "0");
#else
    setenv("XESTRATA_IMAGE_ATTN_I8", "0", 1);
#endif
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0)
        return bench(std::strtoll(argv[2], nullptr, 10), argc >= 4 ? std::strtoll(argv[3], nullptr, 10) : 0);
    std::printf("qi_attn: Qwen-Image's 32 heads of 128, 64 text rows and 512 image rows (and long rows, 8192 keys), against FP64\n");
    return check();
}
