// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gdn_chunk_parity.cpp - the chunked DeltaNet scan (gdn_chunk.cpp) against an FP64 recurrence on the
// CPU: calls of 1, 16, 17 and 64 tokens (a chunk's ends) in form 0 with the output norm, and one call in each other
// form the GPU takes; Qwen3.8's shape.  The line (upstream's docs/GDN_CONTRACT.md, the tier of another precision):
// max |d| / max |ref| over the raw outputs, the normalized FP16 outputs and the state at most the larger of a floor and
// 3x the token-by-token kernels' own (kernels.cpp, run beside it), every value finite.  Exits 77 (a CTest skip) where
// the GPU takes no form of the scan.
#include "gdn_chunk.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "../kernels/parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S;
constexpr double kFloorY = 2e-3, kFloorState = 1e-4;   // GDN_CONTRACT.md's floors
constexpr float kEps = 1e-6f;

struct Inputs {
    int64_t T;
    std::vector<float> h, gate, beta, z, gamma, state0;
};

Inputs make(int64_t T, std::mt19937& rng) {
    Inputs in;
    in.T = T;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    in.h.resize((size_t) (T * C));
    for (auto& v : in.h) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) {
        for (int hd = 0; hd < 2 * HK; ++hd) {   // gdn_conv's L2 norm of the q and k heads
            float* x = in.h.data() + t * C + (int64_t) hd * S;
            double ss = 0;
            for (int r = 0; r < S; ++r) ss += (double) x[r] * x[r];
            for (int r = 0; r < S; ++r) x[r] = (float) (x[r] / std::sqrt(ss + 1e-6));
        }
        for (int r = 2 * HK * S; r < C; ++r) in.h[(size_t) (t * C + r)] *= 0.5f;
    }
    in.gate.resize((size_t) (T * HV));
    in.beta.resize(in.gate.size());
    for (size_t i = 0; i < in.gate.size(); ++i) {
        in.gate[i] = -std::fabs(nd(rng)) * 0.1f;   // softplus(..) * ssm_a: <= 0
        in.beta[i] = 1.0f / (1.0f + std::exp(-nd(rng)));
    }
    in.z.resize((size_t) (T * HV * S));
    for (auto& v : in.z) v = nd(rng);
    in.gamma.resize(S);
    for (auto& v : in.gamma) v = 1.0f + 0.1f * nd(rng);
    in.state0.resize((size_t) S * HV * S);
    for (auto& v : in.state0) v = 0.05f * nd(rng);
    return in;
}

// the recurrence in FP64: y = o / sqrt(S), y16's values before their rounding (sigmoid(z) gate), the state [S][HV][S]
void reference(const Inputs& in, std::vector<double>& y, std::vector<double>& yn, std::vector<double>& st) {
    st.assign(in.state0.begin(), in.state0.end());
    y.assign((size_t) (in.T * HV * S), 0.0);
    yn.assign(y.size(), 0.0);
    std::vector<double> kv(S), o(S);
    for (int64_t t = 0; t < in.T; ++t)
        for (int vh = 0; vh < HV; ++vh) {
            const float* q = in.h.data() + t * C + (int64_t) (vh % HK) * S;
            const float* k = q + (int64_t) HK * S;
            const float* v = in.h.data() + t * C + (int64_t) (2 * HK + vh) * S;
            const double g = std::exp((double) in.gate[(size_t) (t * HV + vh)]), b = in.beta[(size_t) (t * HV + vh)];
            auto at = [&](int r, int col) -> double& { return st[((size_t) r * HV + vh) * S + col]; };
            for (int col = 0; col < S; ++col) {
                double a = 0;
                for (int r = 0; r < S; ++r) a += at(r, col) * k[r];
                kv[col] = a;
            }
            double ss = 0;
            for (int col = 0; col < S; ++col) {
                const double delta = ((double) v[col] - g * kv[col]) * b;
                double a = 0;
                for (int r = 0; r < S; ++r) {
                    at(r, col) = g * at(r, col) + k[r] * delta;
                    a += at(r, col) * q[r];
                }
                o[col] = a / std::sqrt((double) S);
                ss += o[col] * o[col];
            }
            const double rs = 1.0 / std::sqrt(ss / S + kEps);
            for (int col = 0; col < S; ++col) {
                const size_t i = (size_t) (t * HV + vh) * S + col;
                y[i] = o[col];
                yn[i] = o[col] * rs * in.gamma[col] / (1.0 + std::exp(-(double) in.z[i]));
            }
        }
}

template<class T> double nmax(const std::vector<T>& a, const std::vector<double>& ref, bool& finite) {
    double d = 0, m = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double x = (double) a[i];
        if (!std::isfinite(x)) finite = false;
        d = std::max(d, std::fabs(x - ref[i]));
        m = std::max(m, std::fabs(ref[i]));
    }
    return m > 0 ? d / m : d;
}

struct Out {
    std::vector<float> y, st, y16;
};

// form -1: the token-by-token kernels (gdn_recurrence, XESTRATA_GDN_CHUNKED=0); else the scan in that form
Out run(const Inputs& in, int form) {
    using namespace strata;
    parity::Device dev;
    const float* dh = dev.upload(in.h);
    const float* dg = dev.upload(in.gate);
    const float* db = dev.upload(in.beta);
    const float* dz = dev.upload(in.z);
    const float* dgm = dev.upload(in.gamma);
    float* dst = dev.upload(in.state0);
    const size_t ny = (size_t) (in.T * HV * S);
    float* dy = dev.alloc<float>(ny);
    uint16_t* dy16 = dev.alloc<uint16_t>(ny);
    bool normed = true;
    if (form < 0) {
        prefill::gdn_recurrence(dst, dh, dg, db, dz, dgm, kEps, dy, dy16, in.T, parity::stream(), HK, HV);
    } else {
        const prefill::GdnChunkNorm nm{dz, dgm, kEps, false, dy16};
        normed = prefill::gdn_chunk_run(core::Runtime::get().compute(), form, HK, HV, dst, dh, dg, db, dy, &nm, in.T);
    }
    parity::sync();
    Out o;
    o.st.resize(in.state0.size());
    parity::copy_out(o.st.data(), dst, o.st.size());
    if (!normed || form < 0) {   // y is written where the scan did not normalize (and by the token-by-token kernels)
        o.y.resize(ny);
        parity::copy_out(o.y.data(), dy, ny);
    }
    if (normed) {
        std::vector<uint16_t> h(ny);
        parity::copy_out(h.data(), dy16, ny);
        o.y16.resize(ny);
        for (size_t i = 0; i < ny; ++i) o.y16[i] = strata::kernels::f32_from_f16(h[i]);
    }
    return o;
}

}  // namespace

int main() {
    // the token-by-token kernels for the comparison: gdn_recurrence must not choose the scan itself
#if defined(_WIN32)
    _putenv_s("XESTRATA_GDN_CHUNKED", "0");
#else
    setenv("XESTRATA_GDN_CHUNKED", "0", 1);
#endif
    const auto& forms = strata::prefill::gdn_chunk_forms(strata::core::Runtime::get().compute(), HK, HV);
    if (forms.empty()) {
        std::printf("the chunked DeltaNet scan does not run on this GPU: skipped\n");
        return 77;
    }
    struct Case { int form; int64_t T; };
    std::vector<Case> cases = {{0, 1}, {0, 16}, {0, 17}, {0, 64}};
    for (const int f : forms)
        if (f != 0) cases.push_back({f, 17});
    std::mt19937 rng(20261114);
    int fail = 0;
    for (const Case& k : cases) {
        const Inputs in = make(k.T, rng);
        std::vector<double> ry, rn, rs;
        reference(in, ry, rn, rs);
        bool ok = true, iok = true;
        const Out inc = run(in, -1), o = run(in, k.form);
        const double iy = nmax(inc.y, ry, iok), i16 = nmax(inc.y16, rn, iok), is = nmax(inc.st, rs, iok);
        const double es = nmax(o.st, rs, ok);
        double ey;
        bool good = ok && es <= std::max(kFloorState, 3 * is);
        if (!o.y16.empty()) {
            ey = nmax(o.y16, rn, ok);
            good = good && ok && ey <= std::max(kFloorY, 3 * i16);
        } else {
            ey = nmax(o.y, ry, ok);
            good = good && ok && ey <= std::max(kFloorY, 3 * iy);
        }
        std::printf("form %d, %lld tokens: %s %.2e, state %.2e (token by token %.2e, %.2e)%s\n", k.form,
                    (long long) k.T, o.y16.empty() ? "y" : "y16", ey, es, o.y16.empty() ? iy : i16, is,
                    good ? "" : "  FAIL");
        if (!good) ++fail;
    }
    std::printf(fail ? "%d FAILED\n" : "all passed\n", fail);
    return fail ? 1 : 0;
}
