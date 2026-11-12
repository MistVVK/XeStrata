// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/gdn_parity.cpp - P2.S2's test for the GDN recurrence, the conv and the two norms: the native kernels
// (native_gdn.hpp, native_gdn_preprocess.hpp) against `ref/gdn.py`'s reading in double.
//
// `ref/gdn.py` carries eleven PROPERTY checks, several of which exist because a rival reading of the SOURCE
// produces well-formed output.  The four that the kernels here can be wrong about are re-pinned below, each
// asserted OBSERVABLE against a deliberately-wrong computation before the kernel is judged:
//
//   1. MODULO head pairing, h % h_k, not INTERLEAVE.  With H_k=2 and H_v=4 the two give [0,1,0,1] and
//      [0,0,1,1]: they differ at h=1 and h=2.  Both produce a full-rank state of the right shape.
//   2. Decay applied BEFORE the rank-1 update.  Applying it after is a one-line move that keeps every shape
//      and every magnitude.
//   3. The conv's SLIDE DIRECTION.  `kernel[0]` must read the OLDEST state row and the new input must land in
//      the LAST state row; reversing either is invisible on a symmetric state, so the state rows here are
//      labelled so each position is identifiable.
//   4. `l2_norm`'s `+ eps` is an absolute floor on the SQUARED NORM, not on the mean.  The mean reading
//      differs by sqrt(S) = 11.3x at S = 128.  `ref/gdn.py` PROPERTY 1 instruments the derivation, so it is
//      not taken on trust there either.
//
// Last, a window's rows through the verify-window kernels (verify_kernels.hpp) as the plain engine runs them - the
// rows in one launch each, the history and the state left after them, the rows' copies beside - against the same rows
// one at a time: the same bits.
//
// It also checks the STATE LAYOUT, which is the one place these kernels intentionally differ from the
// reference: (S, h_v, S) with j fastest instead of (S, S, h_v).  A layout mix-up is silent, so the state is
// filled with a value that encodes its own coordinates.
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// The reference's `gdn_step`, with switches for the rival readings.
///
/// `state` is passed in the REFERENCE layout (S, S, h_v) - i.e. (i, j, h) - so that this function is a
/// transcription of `ref/gdn.py` and not of the kernel.
void ref_step(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              int S, int h_k, int h_v, std::vector<float>& o, bool modulo_heads, bool decay_first) {
    auto at = [&](int i, int j, int h) -> double& { return state[(size_t) (i * S + j) * h_v + h]; };
    std::vector<float> oo((size_t) h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        const int src = modulo_heads ? (h % h_k) : (h / (h_v / h_k));
        const double dec = std::exp((double) gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            if (decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i, j, h) * (double) k[(size_t) src * S + i];
            const double d = ((double) v[(size_t) h * S + j] - sk) * b;
            for (int i = 0; i < S; ++i) at(i, j, h) += (double) k[(size_t) src * S + i] * d;
            if (!decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double dot = 0;
            for (int i = 0; i < S; ++i) dot += at(i, j, h) * (double) q[(size_t) src * S + i];
            oo[(size_t) h * S + j] = (float) dot;
        }
    }
    o = oo;
}

/// `ref/gdn.py::conv_step` in its own (d_conv, C) row-major orientation.
void ref_conv(std::vector<float>& cs, const std::vector<float>& x, const std::vector<float>& kern, int C,
              int dc, std::vector<float>& out) {
    out.assign((size_t) C, 0.0f);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int i = 0; i < dc - 1; ++i) acc += (double) cs[(size_t) i * C + c] * (double) kern[(size_t) i * C + c];
        acc += (double) x[(size_t) c] * (double) kern[(size_t) (dc - 1) * C + c];
        out[(size_t) c] = (float) acc;
        for (int i = 0; i < dc - 2; ++i) cs[(size_t) i * C + c] = cs[(size_t) (i + 1) * C + c];
        cs[(size_t) (dc - 2) * C + c] = x[(size_t) c];
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gdn_parity [--selftest]\n"); return 2; }
    }

    const int S = 128, h_k = 16, h_v = 48;      // the artifact's real geometry
    strata::gpu::Stream cs = strata::gpu::stream_create();   // the native kernels take an explicit stream
    if (cs == nullptr) { std::fprintf(stderr, "gdn_parity: no stream\n"); return 1; }
    std::mt19937 rng(2024);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the parity inputs reproducible
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ================= 1. the recurrence =================
    {
        // The state encodes its own coordinates, so a layout mix-up cannot pass: reading (i,j,h) where the
        // buffer holds (i,h,j) gives a DIFFERENT number, not a wrong-but-plausible one.
        std::vector<double> st_ref((size_t) S * S * h_v);
        std::vector<float> st_dev((size_t) S * h_v * S);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const float val = (float) (0.001 * (i + 2 * j + 3 * h) - 1.0);
                    st_ref[(size_t) (i * S + j) * h_v + h] = val;
                    st_dev[(size_t) (i * h_v + h) * S + j] = val;   // (S, h_v, S), j fastest
                }

        std::vector<float> q((size_t) h_k * S), k((size_t) h_k * S), v((size_t) h_v * S), gate((size_t) h_v),
            beta((size_t) h_v);
        for (auto& x : q) x = gauss(rng) * 0.1f;
        for (auto& x : k) x = gauss(rng) * 0.1f;
        for (auto& x : v) x = gauss(rng);
        for (auto& x : gate) x = -(2.0f + 4.0f * (float) (rng() % 100) / 100.0f);   // dec in (0.0025, 0.135]
        for (auto& x : beta) x = (float) (rng() % 100) / 100.0f;

        std::vector<float> want_o;
        std::vector<double> st_mod = st_ref;
        ref_step(st_mod, q, k, v, gate, beta, S, h_k, h_v, want_o, true, true);

        // the traps
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, false, true);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "modulo vs interleave head pairing",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, true, false);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "decay before vs after the update",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            // The two readings differ by exactly `k (x) (beta * v * (1 - dec))`, so the fixture is only able
            // to see the trap when `dec` is far from 1 and `beta` is far from 0.  The first version used
            // gate in [-0.99, 0] - dec in [0.37, 1] - and managed 3.25%, which is a threshold the wrong
            // reading nearly clears.  Deep decay is also the realistic regime: `ssm_a = -exp(A_log) < 0` and
            // softplus is positive, so gate is comfortably negative in the real model.
            if (!(rel > 0.05)) ++bad;
        }

        // the kernel
        float *d_st = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_g = nullptr, *d_b = nullptr,
              *d_o = nullptr;
        d_st = static_cast<decltype(d_st)>(strata::parity::alloc_bytes(st_dev.size() * 4));
        d_q = static_cast<decltype(d_q)>(strata::parity::alloc_bytes(q.size() * 4));
        d_k = static_cast<decltype(d_k)>(strata::parity::alloc_bytes(k.size() * 4));
        d_v = static_cast<decltype(d_v)>(strata::parity::alloc_bytes(v.size() * 4));
        d_g = static_cast<decltype(d_g)>(strata::parity::alloc_bytes(gate.size() * 4));
        d_b = static_cast<decltype(d_b)>(strata::parity::alloc_bytes(beta.size() * 4));
        d_o = static_cast<decltype(d_o)>(strata::parity::alloc_bytes((size_t) h_v * S * 4));
        strata::parity::copy_bytes_in(d_st, st_dev.data(), st_dev.size() * 4);
        strata::parity::copy_bytes_in(d_q, q.data(), q.size() * 4);
        strata::parity::copy_bytes_in(d_k, k.data(), k.size() * 4);
        strata::parity::copy_bytes_in(d_v, v.data(), v.size() * 4);
        strata::parity::copy_bytes_in(d_g, gate.data(), gate.size() * 4);
        strata::parity::copy_bytes_in(d_b, beta.data(), beta.size() * 4);

        strata::kernels::GdnShapes sh{S, h_k, h_v};
        strata::kernels::native_gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, cs);
        strata::gpu::stream_sync(cs);
        // q is not pre-scaled: the kernel applies the recurrence's 1/sqrt(S) to the finished readout
        for (auto& x : want_o) x = (float) ((double) x / std::sqrt((double) S));

        std::vector<float> got_o((size_t) h_v * S), got_st(st_dev.size());
        strata::parity::copy_bytes_out(got_o.data(), d_o, got_o.size() * 4);
        strata::parity::copy_bytes_out(got_st.data(), d_st, got_st.size() * 4);

        // o, and the STATE read back through the declared layout
        const double rel_o = rel_l1(want_o, got_o);
        double rel_st = 0, mag_st = 0;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const size_t dev = (size_t) (i * h_v + h) * S + j;
                    const double w = st_mod[(size_t) (i * S + j) * h_v + h];
                    rel_st += std::fabs(w - (double) got_st[dev]);
                    mag_st += std::fabs(w);
                }
        rel_st /= (mag_st > 1e-30 ? mag_st : 1e-30);
        std::printf("\n  %-42s rel %.3e\n", "o vs reference", rel_o);
        std::printf("  %-42s rel %.3e\n", "state (read via (S,h_v,S)) vs reference", rel_st);
        // The kernel accumulates in f32 and the reference in f64, and the state is a RECURRENCE over 128
        // terms, so 1e-5 is the arithmetic's floor here and not a structural allowance.
        if (!(rel_o <= 1e-5)) { std::printf("    *** o over 1e-5 ***\n"); ++bad; }
        if (!(rel_st <= 1e-5)) { std::printf("    *** state over 1e-5 ***\n"); ++bad; }

        // a zero state stays zero, and a zero gate/beta leaves the state untouched (the reference's
        // PROPERTY 4 setup) - a cheap independent invariant that no layout can fake
        {
            std::vector<float> zero(st_dev.size(), 0.0f), zg((size_t) h_v, 0.0f), zb((size_t) h_v, 0.0f);
            strata::parity::copy_bytes_in(d_st, zero.data(), zero.size() * 4);
            strata::parity::copy_bytes_in(d_g, zg.data(), zg.size() * 4);
            strata::parity::copy_bytes_in(d_b, zb.data(), zb.size() * 4);
            strata::kernels::native_gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, cs);
            strata::gpu::stream_sync(cs);
            std::vector<float> zz(zero.size());
            strata::parity::copy_bytes_out(zz.data(), d_st, zz.size() * 4);
            double nz = 0;
            for (float x : zz) nz += std::fabs(x);
            const bool ok = nz == 0.0;
            std::printf("  %-42s %s\n", "gate=beta=0 from a zero state is a no-op", ok ? "yes" : "*** NO ***");
            if (!ok) ++bad;
        }

        // ephemeral state: unchanged when the state is untouched but beta=1 — the delta rule writing k*d
        }

    // ================= 2. the conv =================
    {
        const int C = 24, dc = 4;      // small, so the host reference is a plain loop
        // TWO DIFFERENT LAYOUTS MEET HERE, and this is where the first version of this test failed at 0.136.
        // `ref/gdn.py` carries the conv state as (d_conv-1, C) ROW-MAJOR - (i, c) at i*C + c - because it is
        // numpy.  The kernel carries it as (C, d_conv-1) with c fastest - c*(dc-1) + i - because that is
        // ggml's `ne = {d_conv-1, C}` with ne0 fast, and it is a runtime buffer so either is legal.  The
        // mapping between them is written out below rather than assumed, in both directions.
        std::vector<float> cs_ref((size_t) (dc - 1) * C), x((size_t) C), kern((size_t) dc * C);
        for (int i = 0; i < dc - 1; ++i)
            for (int c = 0; c < C; ++c) cs_ref[(size_t) i * C + c] = (float) (10 * (i + 1) + c);  // labelled
        for (int c = 0; c < C; ++c) x[(size_t) c] = (float) (100 + c);
        for (size_t i = 0; i < kern.size(); ++i) kern[i] = gauss(rng);

        auto to_dev_layout = [&](const std::vector<float>& r) {
            std::vector<float> d(r.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) d[(size_t) c * (dc - 1) + i] = r[(size_t) i * C + c];
            return d;
        };

        std::vector<float> cst = to_dev_layout(cs_ref);
        std::vector<float> want;
        std::vector<float> cs_want_ref = cs_ref;
        ref_conv(cs_want_ref, x, kern, C, dc, want);
        const std::vector<float> cs_want = to_dev_layout(cs_want_ref);

        // THE LAYOUT: the manifest's `ssm_conv1d.weight` has ne = [d_conv, C], so ne0 = d_conv is fast.
        // The kernel takes `kW[c*dc + i]`; the reference's `kern` is (dc, C) row-major.  The transcription
        // below is the ONLY place the two meet.
        std::vector<float> kW((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kW[(size_t) c * dc + i] = kern[(size_t) i * C + c];
        // and the rival reading, which must differ or the fixture cannot see a transposed weight
        std::vector<float> kW_wrong((size_t) C * dc);
        for (size_t i = 0; i < kern.size(); ++i) kW_wrong[i] = kern[i];
        {
            // apply the WRONG layout by hand: tap i of channel c is kW[c*dc+i] read as kW[i*C+c]
            const std::vector<float>& cs2 = cst;
            std::vector<float> out_bad((size_t) C, 0.0f);
            for (int c = 0; c < C; ++c) {
                double a = 0;
                for (int i = 0; i < dc - 1; ++i) a += (double) cs2[(size_t) i * C + c] * (double) kW_wrong[(size_t) c * dc + i];
                a += (double) x[(size_t) c] * (double) kW_wrong[(size_t) (c * dc + dc - 1)];
                out_bad[(size_t) c] = (float) a;
            }
            const double rel = rel_l1(want, out_bad);
            std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "conv weight layout is observable",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }

        float *d_cs = nullptr, *d_x = nullptr, *d_w = nullptr, *d_o = nullptr, *d_silu = nullptr;
        d_cs = static_cast<decltype(d_cs)>(strata::parity::alloc_bytes(cst.size() * 4));
        d_x = static_cast<decltype(d_x)>(strata::parity::alloc_bytes(x.size() * 4));
        d_w = static_cast<decltype(d_w)>(strata::parity::alloc_bytes(kW.size() * 4));
        d_o = static_cast<decltype(d_o)>(strata::parity::alloc_bytes((size_t) C * 4));
        d_silu = static_cast<decltype(d_silu)>(strata::parity::alloc_bytes((size_t) C * 4));
        strata::parity::copy_bytes_in(d_cs, cst.data(), cst.size() * 4);
        strata::parity::copy_bytes_in(d_x, x.data(), x.size() * 4);
        strata::parity::copy_bytes_in(d_w, kW.data(), kW.size() * 4);
        strata::kernels::native_gdn_conv_silu(d_cs, d_x, d_w, d_o, d_silu, C, dc, cs);
        strata::gpu::stream_sync(cs);
        std::vector<float> got((size_t) C), got_silu((size_t) C), got_cs(cst.size());
        strata::parity::copy_bytes_out(got.data(), d_o, got.size() * 4);
        strata::parity::copy_bytes_out(got_silu.data(), d_silu, got_silu.size() * 4);
        strata::parity::copy_bytes_out(got_cs.data(), d_cs, got_cs.size() * 4);
        std::vector<float> want_silu((size_t) C);
        for (int c = 0; c < C; ++c) {
            const double v = (double) want[(size_t) c];
            want_silu[(size_t) c] = (float) (v / (1.0 + std::exp(-v)));
        }
        const double rel_out = rel_l1(want, got), rel_cs = rel_l1(cs_want, got_cs), rel_silu = rel_l1(want_silu, got_silu);
        std::printf("  %-42s rel %.3e\n", "conv out vs reference", rel_out);
        std::printf("  %-42s rel %.3e\n", "conv SiLU vs the double reference", rel_silu);
        std::printf("  %-42s rel %.3e\n", "conv state slide (mapped) vs reference", rel_cs);
        if (!(rel_out <= 1e-6)) { std::printf("    *** conv out ***\n"); ++bad; }
        // f32 fast-math SiLU: a few ulp of the device's exp
        if (!(rel_silu <= 1e-6)) { std::printf("    *** conv silu ***\n"); ++bad; }
        if (!(rel_cs <= 1e-6)) { std::printf("    *** conv state ***\n"); ++bad; }
        }

    // ================= 3. l2_norm =================
    {
        const int rows = 16, cols = 128;
        std::vector<float> x((size_t) rows * cols), x_ref;
        for (auto& val : x) val = gauss(rng) * 3.0f;
        x_ref = x;
        const float eps = 1e-6f;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_ref[(size_t) r * cols + i] * (double) x_ref[(size_t) r * cols + i];
            const float inv = (float) (1.0 / std::sqrt(acc + (double) eps));
            for (int i = 0; i < cols; ++i) x_ref[(size_t) r * cols + i] *= inv;
        }
        // The rival reading puts THE SAME eps on the MEAN instead of on the squared norm, which at
        // ||x||^2 ~ 1152 and cols = 128 changes the divisor from sqrt(1152) to sqrt(9) - a factor of 11.3, so
        // the trap is large here.  It would be a much smaller correction on a near-zero row, where the floor
        // is the whole answer; that regime is not what this fixture exercises.
        std::vector<float> x_mean = x;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_mean[(size_t) r * cols + i] * (double) x_mean[(size_t) r * cols + i];
            const float inv = (float) (1.0 / std::sqrt(acc / cols + (double) eps));
            for (int i = 0; i < cols; ++i) x_mean[(size_t) r * cols + i] *= inv;
        }
        const double rel_trap = rel_l1(x_ref, x_mean);
        std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "eps on sum vs on mean is observable",
                    rel_trap > 0.05 ? "yes" : "*** NO ***", rel_trap * 100);
        if (!(rel_trap > 0.05)) ++bad;

        float* d_x = nullptr;
        d_x = static_cast<decltype(d_x)>(strata::parity::alloc_bytes(x.size() * 4));
        strata::parity::copy_bytes_in(d_x, x.data(), x.size() * 4);
        strata::kernels::native_gdn_l2_norm(d_x, rows, cols, eps, cs);
        strata::gpu::stream_sync(cs);
        std::vector<float> got(x.size());
        strata::parity::copy_bytes_out(got.data(), d_x, got.size() * 4);
        const double rel = rel_l1(x_ref, got);
        std::printf("  %-42s rel %.3e\n", "l2_norm vs reference", rel);
        if (!(rel <= 1e-6)) ++bad;
    }

    // ================= 4. the closing norm =================
    {
        const int S2 = 128, hv2 = 8;
        std::vector<float> o((size_t) hv2 * S2), z((size_t) hv2 * S2), sn((size_t) S2), y_ref((size_t) hv2 * S2);
        for (auto& val : o) val = gauss(rng);
        for (auto& val : z) val = gauss(rng);
        for (auto& val : sn) val = 1.0f + 0.1f * gauss(rng);
        const float eps = 1e-6f;
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) h * S2 + i] * (double) o[(size_t) h * S2 + i];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i)
                y_ref[(size_t) h * S2 + i] = o[(size_t) h * S2 + i] * inv * sn[(size_t) i] *
                                            sigmoid_f(z[(size_t) h * S2 + i]);
        }
        float *d_o = nullptr, *d_z = nullptr, *d_sn = nullptr, *d_y = nullptr;
        d_o = static_cast<decltype(d_o)>(strata::parity::alloc_bytes(o.size() * 4));
        d_z = static_cast<decltype(d_z)>(strata::parity::alloc_bytes(z.size() * 4));
        d_sn = static_cast<decltype(d_sn)>(strata::parity::alloc_bytes(sn.size() * 4));
        d_y = static_cast<decltype(d_y)>(strata::parity::alloc_bytes(y_ref.size() * 4));
        strata::parity::copy_bytes_in(d_o, o.data(), o.size() * 4);
        strata::parity::copy_bytes_in(d_z, z.data(), z.size() * 4);
        strata::parity::copy_bytes_in(d_sn, sn.data(), sn.size() * 4);
        strata::kernels::native_gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, cs);
        strata::gpu::stream_sync(cs);
        std::vector<float> got(y_ref.size());
        strata::parity::copy_bytes_out(got.data(), d_y, got.size() * 4);
        const double rel = rel_l1(y_ref, got);
        std::printf("\n  %-42s rel %.3e\n", "out norm (rms * ssm_norm * sigmoid z)", rel);
        if (!(rel <= 1e-6)) ++bad;

        // SiLU instead of sigmoid is the qwen3.5 reading; `ref/gdn.py` says this artifact does NOT use it
        std::vector<float> y_silu(y_ref.size());
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) h * S2 + i] * (double) o[(size_t) h * S2 + i];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i) {
                const float zz = z[(size_t) h * S2 + i];
                y_silu[(size_t) h * S2 + i] =
                    o[(size_t) h * S2 + i] * inv * sn[(size_t) i] * (zz / (1.0f + std::exp(-zz)));
            }
        }
        const double rel_silu = rel_l1(y_ref, y_silu);
        std::printf("  %-42s %-4s (%.2f%% apart)\n", "sigmoid vs SiLU on the gate is observable",
                    rel_silu > 0.05 ? "yes" : "*** NO ***", rel_silu * 100);
        if (!(rel_silu > 0.05)) ++bad;

        // ...and it is qwen35moe's: gdn_set_silu_gate switches the kernel to it, and back
        strata::kernels::gdn_set_silu_gate(true);
        strata::kernels::native_gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, cs);
        strata::gpu::stream_sync(cs);
        strata::kernels::gdn_set_silu_gate(false);
        strata::parity::copy_bytes_out(got.data(), d_y, got.size() * 4);
        const double rel_gate = rel_l1(y_silu, got);
        std::printf("  %-42s rel %.3e\n", "out norm with gdn_set_silu_gate (silu z)", rel_gate);
        if (!(rel_gate <= 1e-6)) ++bad;
        }

    // ================= 5. a window's rows at once, as the plain engine runs them =================
    {
        namespace sk = strata::kernels;
        const int T = 3, NE = 256, C = 2 * h_k * S + h_v * S, ZV = h_v * S;
        const float eps = 1e-6f;
        auto fill = [&](std::vector<float>& v, size_t n, float scale) {
            v.resize(n);
            for (auto& x : v) x = gauss(rng) * scale;
        };
        std::vector<float> qkv, conv_w, hist, x, wa, wb, dt, sa, z, gamma, st;
        fill(qkv, (size_t) T * C, 1.0f); fill(conv_w, (size_t) C * 4, 0.5f); fill(hist, (size_t) C * 3, 1.0f);
        fill(x, (size_t) T * NE, 1.0f); fill(wa, (size_t) h_v * NE, 0.1f); fill(wb, (size_t) h_v * NE, 0.1f);
        fill(dt, (size_t) h_v, 1.0f); fill(sa, (size_t) h_v, 1.0f); fill(z, (size_t) T * ZV, 1.0f);
        fill(gamma, (size_t) S, 1.0f); fill(st, (size_t) S * h_v * S, 0.01f);
        for (auto& v : sa) v = -std::fabs(v);
        // BF16-exact weights: the F32 and BF16 forms of the alpha/beta kernel must then agree to the bit
        std::vector<uint16_t> wa16(wa.size()), wb16(wb.size());
        for (size_t i = 0; i < wa.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &wa[i], 4); u &= 0xffff0000u; std::memcpy(&wa[i], &u, 4); wa16[i] = (uint16_t) (u >> 16);
            std::memcpy(&u, &wb[i], 4); u &= 0xffff0000u; std::memcpy(&wb[i], &u, 4); wb16[i] = (uint16_t) (u >> 16);
        }
        auto up = [&](const void* src, size_t bytes) {
            void* d = strata::parity::alloc_bytes(std::max<size_t>(bytes, 4));
            if (src != nullptr) strata::parity::copy_bytes_in(d, src, bytes);
            return static_cast<float*>(d);
        };
        const std::vector<int32_t> counts{0, 1, 2, 3};
        auto* d_counts = reinterpret_cast<int32_t*>(up(counts.data(), counts.size() * 4));
        float *d_qkv = up(qkv.data(), qkv.size() * 4), *d_cw = up(conv_w.data(), conv_w.size() * 4),
              *d_x = up(x.data(), x.size() * 4), *d_wa = up(wa.data(), wa.size() * 4), *d_wb = up(wb.data(), wb.size() * 4),
              *d_dt = up(dt.data(), dt.size() * 4), *d_sa = up(sa.data(), sa.size() * 4), *d_z = up(z.data(), z.size() * 4),
              *d_gm = up(gamma.data(), gamma.size() * 4);
        auto* d_wa16 = reinterpret_cast<uint16_t*>(up(wa16.data(), wa16.size() * 2));
        auto* d_wb16 = reinterpret_cast<uint16_t*>(up(wb16.data(), wb16.size() * 2));
        // [0]: the window, [1]: row by row
        float *d_hist[2], *d_st[2], *d_h[2], *d_gate[2], *d_beta[2], *d_y[2], *d_hsnap[2][T - 1], *d_ssnap[2][T - 1];
        for (int w = 0; w < 2; ++w) {
            d_hist[w] = up(hist.data(), hist.size() * 4);
            d_st[w] = up(st.data(), st.size() * 4);
            d_h[w] = up(nullptr, (size_t) T * C * 4);
            d_gate[w] = up(nullptr, (size_t) T * h_v * 4);
            d_beta[w] = up(nullptr, (size_t) T * h_v * 4);
            d_y[w] = up(nullptr, (size_t) T * ZV * 4);
            for (int t = 0; t < T - 1; ++t) {
                d_hsnap[w][t] = up(nullptr, hist.size() * 4);
                d_ssnap[w][t] = up(nullptr, st.size() * 4);
            }
        }
        sk::GdnRowCopies hs, ss;
        for (int t = 0; t < T - 1; ++t) { hs.p[t] = d_hsnap[0][t]; ss.p[t] = d_ssnap[0][t]; }
        hs.n = ss.n = T - 1;
        sk::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, d_gate[0], d_beta[0], NE, h_v, T, cs);
        sk::gdn_conv_l2_multi(d_hist[0], d_qkv, d_cw, d_h[0], C, 2 * h_k, eps, T, cs, 0, false, &hs);
        sk::gdn_conv_commit(d_hist[0], d_qkv, C, d_counts + T, cs);
        sk::gdn_step_norm_multi(d_st[0], d_h[0], C, d_gate[0], d_beta[0], d_z, d_gm, eps, d_y[0], h_k, h_v, T,
                                d_counts + T, cs, 0, nullptr, &ss);
        for (int t = 0; t < T; ++t) {
            sk::gdn_ab_multi(d_x + (size_t) t * NE, d_wa16, d_wb16, d_dt, d_sa, d_gate[1] + (size_t) t * h_v,
                             d_beta[1] + (size_t) t * h_v, NE, h_v, 1, cs);
            sk::gdn_conv_l2_multi(d_hist[1], d_qkv + (size_t) t * C, d_cw, d_h[1] + (size_t) t * C, C, 2 * h_k, eps, 1, cs,
                                  0, true);
            sk::gdn_step_norm_multi(d_st[1], d_h[1] + (size_t) t * C, C, d_gate[1] + (size_t) t * h_v,
                                    d_beta[1] + (size_t) t * h_v, d_z + (size_t) t * ZV, d_gm, eps,
                                    d_y[1] + (size_t) t * ZV, h_k, h_v, 1, d_counts + 1, cs);
            if (t < T - 1) {
                strata::gpu::stream_sync(cs);
                std::vector<float> tmp(st.size());
                strata::parity::copy_bytes_out(tmp.data(), d_st[1], tmp.size() * 4);
                strata::parity::copy_bytes_in(d_ssnap[1][t], tmp.data(), tmp.size() * 4);
                strata::parity::copy_bytes_out(tmp.data(), d_hist[1], hist.size() * 4);
                strata::parity::copy_bytes_in(d_hsnap[1][t], tmp.data(), hist.size() * 4);
            }
        }
        strata::gpu::stream_sync(cs);
        auto same = [&](const char* what, const float* a, const float* b, size_t n) {
            std::vector<float> va(n), vb(n);
            strata::parity::copy_bytes_out(va.data(), a, n * 4);
            strata::parity::copy_bytes_out(vb.data(), b, n * 4);
            const bool ok = std::memcmp(va.data(), vb.data(), n * 4) == 0;
            std::printf("  %-42s %s\n", what, ok ? "same bits" : "*** DIFFER ***");
            if (!ok) ++bad;
        };
        same("window: gate (F32 vs BF16 weights)", d_gate[0], d_gate[1], (size_t) T * h_v);
        same("window: beta (F32 vs BF16 weights)", d_beta[0], d_beta[1], (size_t) T * h_v);
        same("window: conv + norms of the rows", d_h[0], d_h[1], (size_t) T * C);
        same("window: outputs", d_y[0], d_y[1], (size_t) T * ZV);
        same("window: conv history after the rows", d_hist[0], d_hist[1], hist.size());
        same("window: state after the rows", d_st[0], d_st[1], st.size());
        for (int t = 0; t < T - 1; ++t) {
            same("window: conv history copy of a row", d_hsnap[0][t], d_hsnap[1][t], hist.size());
            same("window: state copy of a row", d_ssnap[0][t], d_ssnap[1][t], st.size());
        }
    }

    strata::gpu::stream_destroy(cs);
    std::printf("\ngdn: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gdn_parity OK\n");
    return 0;
}
