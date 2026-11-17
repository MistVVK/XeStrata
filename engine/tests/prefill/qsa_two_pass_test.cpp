// tests/prefill/qsa_two_pass_test.cpp - STRATA_SELECT_TWO_PASS without a GPU (qsa_two_pass.hpp).
//
// 1. TF32: the host copy of cvt.rna.tf32.f32 rounds to nearest, ties away from zero.
// 2. The bound: for random and adversarial indexer queries and keys, the single-pass score S1 and the 3xTF32 score S3
//    are computed from the same TF32 operands as the kernels, with each MMA modelled two ways - exact then rounded to
//    nearest (the most accurate hardware), and aligned to the largest addend with every addend and the result
//    truncated to 24 bits (no guard bits, the least accurate model) - and |S1 - S3| must stay within eps.  The exact
//    cross term |qh.kl + ql.kh| must stay within its Cauchy-Schwarz part.
// 3. The cut-off rule: rows of 3xTF32 scores, single-pass scores anywhere inside [S3 - eps, S3 + eps] (pushed across
//    the threshold where possible), ties at the threshold, every tail weight, short and long rows (the sampled
//    candidate threshold), the identity: a host copy of the band kernel picks the blocks to score again, and the
//    top-k of the resulting row (block_topk_kernel's rule) must be the top-k of the 3xTF32 row, id for id.
// Exits non-zero on the first failure.  No GPU, no model.
#include "strata/kernels/qsa_two_pass.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace tp = strata::kernels::two_pass;

namespace {

int g_fail = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);       \
            std::fprintf(stderr, __VA_ARGS__);                              \
            std::fprintf(stderr, "\n");                                     \
            ++g_fail;                                                       \
            if (g_fail > 20) std::exit(1);                                  \
        }                                                                   \
    } while (0)

float up(double x) {   // the smallest float >= x (x finite and in range)
    float f = (float) x;
    if ((double) f < x) f = std::nextafter(f, std::numeric_limits<float>::infinity());
    return f;
}
float down(double x) {
    float f = (float) x;
    if ((double) f > x) f = std::nextafter(f, -std::numeric_limits<float>::infinity());
    return f;
}

// Shewchuk's exact summation (Python's math.fsum): the sum of doubles, correctly rounded
double fsum(const std::vector<double>& xs) {
    std::vector<double> p;
    for (double x : xs) {
        size_t i = 0;
        for (double y : p) {
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            const double hi = x + y, lo = y - (hi - x);
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        p.resize(i);
        p.push_back(x);
    }
    double s = 0.0;
    for (size_t i = p.size(); i-- > 0;) s += p[i];   // the partials are non-overlapping: this is the rounded sum
    return s;
}

// ---- 1. TF32
void test_tf32() {
    auto bits = [](float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; };
    CHECK(tp::tf32_rna_host(1.0f) == 1.0f, "1.0");
    CHECK(tp::tf32_rna_host(1.0f + 0x1p-11f) == 1.0f + 0x1p-10f, "a tie rounds away from zero");
    CHECK(tp::tf32_rna_host(-(1.0f + 0x1p-11f)) == -(1.0f + 0x1p-10f), "a negative tie rounds away from zero");
    CHECK(tp::tf32_rna_host(1.0f + 0x1p-12f) == 1.0f, "below the tie rounds down");
    CHECK((bits(tp::tf32_rna_host(3.14159265f)) & 0x1fffu) == 0u, "the low 13 bits are zero");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (int i = 0; i < 100000; ++i) {
        const float x = nd(rng) * std::ldexp(1.0f, (int) (rng() % 40) - 20);
        const float h = tp::tf32_rna_host(x);
        CHECK(std::fabs((double) x - (double) h) <= std::ldexp(std::fabs((double) x), -11), "TF32 error %g", (double) x);
    }
}

// ---- 2. the bound
struct Op { float h, l; };   // tf32(x), tf32(x - tf32(x)): the kernels' split
Op split(float x) {
    const float h = tp::tf32_rna_host(x);
    return {h, tp::tf32_rna_host(x - h)};
}

// one MMA: the accumulator c plus 8 exact products
float mma_rn(float c, const double* p) {
    std::vector<double> t(p, p + 8);
    t.push_back(c);
    return (float) fsum(t);
}
float trunc_to(double v, int e) {   // toward zero to a multiple of 2^e
    const double q = std::ldexp(1.0, e);
    return (float) (std::trunc(v / q) * q);
}
float mma_trunc(float c, const double* p) {
    double t[9];
    for (int i = 0; i < 8; ++i) t[i] = p[i];
    t[8] = c;
    int emax = std::numeric_limits<int>::min();
    for (double v : t)
        if (v != 0.0) { int e; std::frexp(v, &e); emax = std::max(emax, e); }
    if (emax == std::numeric_limits<int>::min()) return 0.0f;
    std::vector<double> a;
    for (double v : t) {   // every addend kept to 24 bits below the largest one's leading bit, toward zero
        const double q = std::ldexp(1.0, emax - 24);
        a.push_back(std::trunc(v / q) * q);
    }
    const double s = fsum(a);
    if (s == 0.0) return 0.0f;
    int e;
    std::frexp(s, &e);
    return trunc_to(s, e - 24);   // the result truncated to FP32's 24 bits
}

struct Pair { std::vector<float> q, k; };   // q: 4 heads x 128, k: 128

// The scores the kernels compute, per model: S1 (qh*kh, 16 MMAs per head) and S3 (ql*kh, qh*kl, qh*kh per step).
void scores(const Pair& x, bool trunc, float& s1, float& s3, double cross[4], double hn[4][2], double kn[2]) {
    std::vector<Op> k(128);
    for (int i = 0; i < 128; ++i) k[i] = split(x.k[i]);
    double kh2 = 0, kl2 = 0;
    for (int i = 0; i < 128; ++i) { kh2 += (double) k[i].h * k[i].h; kl2 += (double) k[i].l * k[i].l; }
    kn[0] = std::sqrt(kh2);
    kn[1] = std::sqrt(kl2);
    s1 = s3 = 0.0f;
    for (int h = 0; h < 4; ++h) {
        std::vector<Op> q(128);
        for (int i = 0; i < 128; ++i) q[i] = split(x.q[h * 128 + i]);
        double qh2 = 0, ql2 = 0;
        for (int i = 0; i < 128; ++i) { qh2 += (double) q[i].h * q[i].h; ql2 += (double) q[i].l * q[i].l; }
        hn[h][0] = std::sqrt(qh2);
        hn[h][1] = std::sqrt(ql2);
        std::vector<double> d;
        for (int i = 0; i < 128; ++i) { d.push_back((double) q[i].h * k[i].l); d.push_back((double) q[i].l * k[i].h); }
        cross[h] = fsum(d);
        float c1 = 0.0f, c3 = 0.0f;
        for (int st = 0; st < 16; ++st) {
            double p_hh[8], p_lh[8], p_hl[8];
            for (int j = 0; j < 8; ++j) {
                const int i = 8 * st + j;
                p_hh[j] = (double) q[i].h * k[i].h;
                p_lh[j] = (double) q[i].l * k[i].h;
                p_hl[j] = (double) q[i].h * k[i].l;
            }
            c1 = trunc ? mma_trunc(c1, p_hh) : mma_rn(c1, p_hh);
            c3 = trunc ? mma_trunc(c3, p_lh) : mma_rn(c3, p_lh);   // the kernel's order: al*bh, ah*bl, ah*bh
            c3 = trunc ? mma_trunc(c3, p_hl) : mma_rn(c3, p_hl);
            c3 = trunc ? mma_trunc(c3, p_hh) : mma_rn(c3, p_hh);
        }
        s1 += c1 > 0.0f ? c1 : 0.0f;
        s3 += c3 > 0.0f ? c3 : 0.0f;
    }
}

// eps as the band kernel forms it (norms rounded up, the coefficients and eps rounded up)
float eps_of(const double hn[4][2], const double kn[2]) {
    double a = 0, c = 0;
    for (int h = 0; h < 4; ++h) { a += hn[h][0]; c += hn[h][1]; }
    a = up(a) * (1 + 1e-12);
    c = up(c) * (1 + 1e-12);
    double P, Q;
    tp::coefficients(a, c, P, Q);
    const float Pf = up(P), Qf = up(Q), kh = up(kn[0] * (1 + 1e-12)), kl = up(kn[1] * (1 + 1e-12));
    return up((double) Pf * kl + (double) Qf * kh + (double) tp::kAbs);
}

void test_bound() {
    std::mt19937 rng(20261024);
    std::normal_distribution<float> nd;
    std::uniform_int_distribution<int> small(-3, 3), expo(-12, 12), coin(0, 9);
    // the worst rounding: mantissas just below a TF32 tie, so |x - tf32(x)| is nearly 2^-11 |x|
    auto worst = [&](float sign) {
        uint32_t b = 0x3f800000u | ((rng() & 0x3ffu) << 13) | 0x0fffu;   // ...0 1111 1111 1111 in the dropped bits
        float f;
        std::memcpy(&f, &b, 4);
        return sign * std::ldexp(f, expo(rng) / 4);
    };
    const char* names[] = {"normal", "wide magnitudes and zeros", "small integers", "aligned worst rounding (q = k)",
                           "worst rounding, signs of k"};
    for (int mode = 0; mode < 5; ++mode) {
        double worst_ratio[2] = {0, 0}, worst_cross = 0;
        const int N = mode >= 3 ? 1500 : 3000;
        for (int n = 0; n < N; ++n) {
            Pair x;
            x.q.resize(512);
            x.k.resize(128);
            for (int i = 0; i < 128; ++i) {
                float v;
                if (mode == 0) v = nd(rng);
                else if (mode == 1) v = coin(rng) == 0 ? 0.0f : std::ldexp(nd(rng), expo(rng));
                else if (mode == 2) v = (float) small(rng);
                else v = worst(nd(rng) > 0 ? 1.0f : -1.0f);
                x.k[i] = v;
            }
            for (int h = 0; h < 4; ++h)
                for (int i = 0; i < 128; ++i) {
                    float v;
                    if (mode == 0) v = nd(rng);
                    else if (mode == 1) v = coin(rng) == 0 ? 0.0f : std::ldexp(nd(rng), expo(rng));
                    else if (mode == 2) v = (float) small(rng);
                    else if (mode == 3) v = x.k[i];   // every head aligned with k, lo parts too: Cauchy-Schwarz is tight
                    else v = worst(x.k[i] > 0 ? 1.0f : -1.0f);   // independent worst roundings, positive dot products
                    x.q[h * 128 + i] = v;
                }
            for (int model = 0; model < 2; ++model) {
                float s1, s3;
                double cross[4], hn[4][2], kn[2];
                scores(x, model == 1, s1, s3, cross, hn, kn);
                const float e = eps_of(hn, kn);
                const double d = std::fabs((double) s1 - (double) s3);
                CHECK(d <= (double) e, "%s, %s model: |S1 - S3| = %.9g > eps %.9g", names[mode],
                      model ? "truncating" : "rounding", d, (double) e);
                worst_ratio[model] = std::max(worst_ratio[model], d / (double) e);
                if (model == 0)
                    for (int h = 0; h < 4; ++h) {
                        const double cb = hn[h][0] * kn[1] + hn[h][1] * kn[0];
                        CHECK(std::fabs(cross[h]) <= cb * (1 + 1e-12) + 1e-300, "%s: cross term %.9g > %.9g", names[mode],
                              std::fabs(cross[h]), cb);
                        if (cb > 0) worst_cross = std::max(worst_cross, std::fabs(cross[h]) / cb);
                    }
            }
        }
        std::printf("bound, %-36s: largest |S1 - S3| / eps %.4f (rounding MMA), %.4f (truncating MMA); "
                    "cross term / its bound %.4f\n", names[mode], worst_ratio[0], worst_ratio[1], worst_cross);
    }
}

// ---- 3. the cut-off rule
constexpr int R = 4;
uint32_t key(float s) { return tp::order_key_host(s); }

// the largest key k with (sum of weights of entries with key >= k) >= width (block_topk_kernel's threshold)
uint32_t kth(std::vector<std::pair<uint32_t, int>> kw, int64_t width) {
    std::sort(kw.begin(), kw.end(), [](auto& a, auto& b) { return a.first > b.first; });
    int64_t c = 0;
    for (auto& [k, w] : kw) {
        c += w;
        if (c >= width) return k;
    }
    return 0u;   // fewer cells than the width: the radix scan ends at digit 0 every pass
}

// block_topk_kernel's ids: cells of blocks above the threshold, and the first (width - above) tied cells by index
std::vector<int32_t> topk(const std::vector<float>& sc, int64_t n_kv, int64_t n_bid, int64_t width) {
    std::vector<int32_t> out;
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j) out.push_back((int32_t) j);
        return out;
    }
    auto weight = [&](int64_t b) { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    std::vector<std::pair<uint32_t, int>> kw;
    for (int64_t b = 0; b <= n_bid; ++b)
        if (weight(b)) kw.push_back({key(sc[b]), weight(b)});
    const uint32_t thr = kth(kw, width);
    int64_t above = 0;
    for (int64_t b = 0; b <= n_bid; ++b)
        if (weight(b) && key(sc[b]) > thr) above += weight(b);
    int64_t eq_left = width - above;
    for (int64_t b = 0; b <= n_bid; ++b) {
        const int w = weight(b);
        if (!w) continue;
        const uint32_t k = key(sc[b]);
        if (k > thr) for (int c = 0; c < w; ++c) out.push_back((int32_t) (b * R + c));
        else if (k == thr) for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out.push_back((int32_t) (b * R + c));
    }
    return out;
}

struct Band { int mode = 0; std::vector<int64_t> again; int64_t cand = 0; };

// the band kernel on the host (tp_band_kernel, qsa_select.cu): the same steps and constants
Band band(const std::vector<float>& s1, const std::vector<float>& eps, int64_t n_kv, int64_t n_bid, int64_t width) {
    Band r;
    if (n_kv <= width) return r;
    const int64_t nb = n_bid + 1;
    const int w_tail = (int) (n_kv - n_bid * R);
    auto interval = [&](int64_t b, float& lo, float& hi, float& e) {
        e = eps[b];
        lo = down((double) s1[b] - (double) e);
        hi = up((double) s1[b] + (double) e);
    };
    float F = -std::numeric_limits<float>::infinity();
    if (nb > tp::kCandCap) {
        int64_t stride = tp::kSampleStride;
        while ((nb + stride - 1) / stride > tp::kCandCap) stride *= 2;
        const int64_t ns = (nb + stride - 1) / stride;
        std::vector<std::pair<uint32_t, int>> kw;
        for (int64_t i = 0; i < ns; ++i) {
            const int64_t b = i * stride;
            uint32_t k = 0u;
            if (b < n_bid) { float lo, hi, e; interval(b, lo, hi, e); k = key(lo); }
            else if (w_tail > 0) k = key(s1[b]);
            kw.push_back({k, 1});
        }
        const int64_t ks = 2 * ((width + R * stride - 1) / (R * stride)) + 16;
        const uint32_t fk = kth(kw, std::min(ks, ns));
        if (fk != 0u) F = tp::key_float_host(fk);
    }
    std::vector<int64_t> cand;
    int64_t wf = 0;
    float E = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        float lo, hi, e = 0.0f;
        int w;
        if (b < n_bid) { w = R; interval(b, lo, hi, e); }
        else { w = w_tail; if (!w) continue; lo = hi = s1[b]; }
        if (lo >= F) wf += w;
        if (hi >= F) { cand.push_back(b); E = std::max(E, e); }
    }
    r.cand = (int64_t) cand.size();
    if ((int64_t) cand.size() > tp::kCandCap || wf < width) { r.mode = 2; return r; }
    std::vector<std::pair<uint32_t, int>> kw;
    for (int64_t b : cand) kw.push_back({key(s1[b]), b < n_bid ? R : w_tail});
    const float T1 = tp::key_float_host(kth(kw, width));
    const float lo_b = std::max(down((double) T1 - (double) E), F), hi_b = up((double) T1 + (double) E);
    int res[8] = {};
    for (int64_t b : cand) {
        if (b >= n_bid) continue;
        float lo, hi, e;
        interval(b, lo, hi, e);
        if (hi >= lo_b && lo <= hi_b) { r.again.push_back(b); ++res[b & 7]; }
    }
    for (int x : res) if (x > tp::kResCap) { r.mode = 2; return r; }
    r.mode = 1;
    return r;
}

void test_rule() {
    std::mt19937 rng(424242);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const int64_t sizes[] = {300, 512, 600, 1500, 4000, 5119, 5120, 5121, 9000, 20000, 65000};
    const char* kinds[] = {"continuous", "small integers (ties)", "a plateau at the threshold", "eps wide against the spread",
                           "heavy tail", "every score equal"};
    int64_t rows = 0, full = 0, again = 0, cand = 0;
    for (int kind = 0; kind < 6; ++kind) {
        const int64_t rows0 = rows, full0 = full, again0 = again;
        for (int64_t n_bid : sizes) {
            for (int tail = 0; tail < 4; ++tail) {
                for (int rep = 0; rep < (n_bid > 20000 ? 2 : 4); ++rep) {
                    const int64_t n_kv = n_bid * R + tail, width = std::min<int64_t>(n_kv, 2048 + R - 1);
                    std::vector<float> s3((size_t) n_bid + 1), s1((size_t) n_bid + 1), eps((size_t) n_bid + 1);
                    // kind 0: continuous; 1: small integers (ties); 2: a plateau at the threshold; 3: eps wide
                    // against the spread; 4: heavy tail; 5: every score equal
                    for (int64_t b = 0; b < n_bid; ++b) {
                        double v;
                        if (kind == 1) v = (double) (int) (ud(rng) * 40);
                        else if (kind == 2) v = ud(rng) < 0.3 ? 7.0 : 7.0 + nd(rng);
                        else if (kind == 4) v = std::exp(2.0 * nd(rng));
                        else if (kind == 5) v = 3.0;
                        else v = std::max(0.0, 2.0 + nd(rng));
                        s3[b] = (float) v;
                        const double scale = kind == 3 ? 0.3 : (kind == 1 ? 0.6 : 0.002);
                        eps[b] = (float) (scale * (0.5 + ud(rng)));
                    }
                    s3[n_bid] = tail ? 1e9f + (float) ud(rng) : (float) ud(rng);   // the tail: exact, +1e9 with cells
                    eps[n_bid] = 0.0f;
                    // the 3xTF32 threshold, to push single-pass scores across it
                    const std::vector<int32_t> ref = topk(s3, n_kv, n_bid, width);
                    std::vector<std::pair<uint32_t, int>> kw;
                    for (int64_t b = 0; b <= n_bid; ++b) {
                        const int w = b < n_bid ? R : tail;
                        if (w) kw.push_back({key(s3[b]), w});
                    }
                    const float T = n_kv > width ? tp::key_float_host(kth(kw, width)) : 0.0f;
                    for (int64_t b = 0; b < n_bid; ++b) {
                        // S1 inside [S3 - eps, S3 + eps]: towards and across the threshold where it can, else random
                        const double e = eps[b] * (1.0 - 1e-6);
                        double d;
                        if (rep % 2 == 0) d = (s3[b] >= T ? -e : e);
                        else d = (2.0 * ud(rng) - 1.0) * e;
                        s1[b] = (float) ((double) s3[b] + d);
                        if (std::fabs((double) s1[b] - (double) s3[b]) > eps[b]) s1[b] = s3[b];
                    }
                    s1[n_bid] = s3[n_bid];
                    const Band bd = band(s1, eps, n_kv, n_bid, width);
                    std::vector<float> row = s1;
                    if (bd.mode == 2) row = s3;
                    for (int64_t b : bd.again) row[b] = s3[b];
                    const std::vector<int32_t> got = topk(row, n_kv, n_bid, width);
                    CHECK(got == ref, "kind %d, %lld blocks, tail %d, rep %d: ids differ (mode %d, %zu scored again)", kind,
                          (long long) n_bid, tail, rep, bd.mode, bd.again.size());
                    ++rows;
                    full += bd.mode == 2;
                    again += (int64_t) bd.again.size();
                    cand += bd.cand;
                }
            }
        }
        std::printf("rule, %-28s: %3lld rows, %3lld scored again in full, %.1f blocks scored again per row\n", kinds[kind],
                    (long long) (rows - rows0), (long long) (full - full0),
                    rows > rows0 ? (double) (again - again0) / (double) (rows - rows0) : 0.0);
    }
    std::printf("rule: %lld rows, ids identical in every one; %lld scored again in full (the list or the bound could not "
                "settle them), %.1f blocks scored again and %.0f candidates per row\n", (long long) rows, (long long) full,
                rows ? (double) again / rows : 0.0, rows ? (double) cand / rows : 0.0);
    // the guard bites: keep the single-pass score of one block whose interval reaches the threshold, and the ids move
    {
        const int64_t n_bid = 1000, n_kv = n_bid * R, width = 2048 + R - 1;
        std::vector<float> s3((size_t) n_bid + 1), s1((size_t) n_bid + 1), eps((size_t) n_bid + 1, 0.01f);
        for (int64_t b = 0; b < n_bid; ++b) s3[b] = 1.0f + 0.001f * (float) b;   // distinct, rising
        s3[n_bid] = 0.0f;
        eps[n_bid] = 0.0f;
        const std::vector<int32_t> ref = topk(s3, n_kv, n_bid, width);
        // the 513th block from the top (block 487) sits at the threshold; its single-pass score 0.009 lower
        for (int64_t b = 0; b <= n_bid; ++b) s1[b] = s3[b];
        s1[487] = s3[487] - 0.009f;
        s1[486] = s3[486] + 0.009f;
        const Band bd = band(s1, eps, n_kv, n_bid, width);
        CHECK(bd.mode == 1 && std::find(bd.again.begin(), bd.again.end(), 487) != bd.again.end(),
              "block 487 must be scored again (mode %d)", bd.mode);
        std::vector<float> row = s1;
        for (int64_t b : bd.again) row[b] = s3[b];
        CHECK(topk(row, n_kv, n_bid, width) == ref, "the guarded row must select today's ids");
        CHECK(topk(s1, n_kv, n_bid, width) != ref, "without the re-score the ids must differ (the test bites)");
    }
}

}  // namespace

int main() {
    test_tf32();
    test_bound();
    test_rule();
    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_fail);
        return 1;
    }
    std::puts("PASS: qsa_two_pass_test (TF32 split, the bound under two MMA models, the cut-off rule)");
    return 0;
}
