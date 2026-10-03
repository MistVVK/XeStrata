// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gr_parity.cpp - P2.S2's test for the gated residual / hyper-connection.
//
// `ref/gr.py` opens by listing the details "a prose reading gets wrong", and every one of them is a reading
// that has the RIGHT SHAPES and produces a PLAUSIBLE number.  So each is computed here the wrong way round and
// required to differ materially BEFORE the kernel is judged against the right one.  A test that only does the
// second half passes against either reading:
//
//   1. PER-STREAM RMSNorm.  The source comment reads ambiguously; the code reduces over ne[0] = n_embd, i.e.
//      one RMS per stream.  The rival reading is one RMS over the whole hc*n_embd stack.
//   2. `/ hc` INSIDE the silu: `lo = silu(proj / hc)`.  Moving it outside is a one-line change that keeps
//      every shape.
//   3. SiLU on `lo` and SIGMOID on the gate, a few lines apart in the same function.  Swapping them is the
//      obvious slip, and both are monotone saturating functions so the magnitudes stay comparable.
//   4. `mean` over the stream axis, not `sum`.  A factor of hc = 4, which is exactly the kind of error that
//      looks like a scale problem rather than a structural one.
//   5. ACTIVATION PRECISION. The historical BF16-rounded CPU contract and the pinned CUDA single-token
//      FP32-activation contract must be distinguishable. Both are tested against explicit references;
//      neither is selected merely because the weight is BF16. The scalar fixture isolates xn and lo
//      rounding from reduction order, and graph replay must preserve the precision chosen at capture.
//   6. `gr_write`'s `2*sigmoid`, which centres the gate on 1 so a ZERO injection is a plain residual add.
//      Asserted as a property, not as a value, because that is what the source comment claims.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/fused_gr.hpp"

#include "parity_device.hpp"
#include "strata/core/gpu.hpp"
#include <stdexcept>

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string(operation) + ": " + strata::gpu::last_error());
}


/// The same rule the kernel uses, on the host - `ref/quant.py::bf16`.
float to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    float o;
    std::memcpy(&o, &i, 4);
    return o;
}

uint16_t bf16_bits(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t) (i >> 16);
}

struct Opts {
    bool per_stream_norm = true;   ///< false: one RMS over the whole stack (WRONG)
    bool scale_inside_silu = true; ///< false: silu(proj)/hc instead of silu(proj/hc) (WRONG)
    bool sigmoid_on_gate = true;   ///< false: silu on the gate and sigmoid on lo (WRONG)
    bool mean_over_streams = true; ///< false: sum over streams instead of mean (WRONG)
    bool round_activation = true;  ///< false: pinned CUDA single-token BF16 MMVF contract
};

/// `ref/gr.py::gr_read`, transcribed with switches for the rival readings.
void reference(const std::vector<float>& R, const std::vector<float>& w_norm,
               const std::vector<float>& w_down, const std::vector<float>& w_up,
               const std::vector<float>& w_inject, float eps, long long n_embd, long long hc, long long hc_lr,
               const Opts& o, std::vector<float>& mixed, std::vector<float>& inject) {
    const long long hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    if (o.per_stream_norm) {
        for (long long c = 0; c < hc; ++c) {
            double ms = 0;
            for (long long d = 0; d < n_embd; ++d) ms += (double) R[(size_t) (c * n_embd + d)] * R[(size_t) (c * n_embd + d)];
            ms /= (double) n_embd;
            const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                xn[i] = R[i] * rs * w_norm[i];
            }
        }
    } else {
        double ms = 0;
        for (long long i = 0; i < hc_dim; ++i) ms += (double) R[(size_t) i] * R[(size_t) i];
        ms /= (double) hc_dim;
        const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
        for (long long i = 0; i < hc_dim; ++i) xn[(size_t) i] = R[(size_t) i] * rs * w_norm[(size_t) i];
    }

    std::vector<float> act((size_t) hc_dim);
    for (long long i = 0; i < hc_dim; ++i) act[(size_t) i] = o.round_activation ? to_bf16(xn[(size_t) i]) : xn[(size_t) i];

    // lo
    std::vector<float> lo((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_down[(size_t) (k * hc_dim + i)];
        const float p = (float) a;
        if (o.sigmoid_on_gate) lo[(size_t) k] = o.scale_inside_silu ? (p / (float) hc) / (1.0f + std::exp(-(p / (float) hc)))
                                                                  : (p / (1.0f + std::exp(-p))) / (float) hc;
        else                   lo[(size_t) k] = o.scale_inside_silu ? 1.0f / (1.0f + std::exp(-(p / (float) hc)))
                                                                    : (1.0f / (1.0f + std::exp(-p))) / (float) hc;
    }
    std::vector<float> lq((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k)
        lq[(size_t) k] = o.round_activation ? to_bf16(lo[(size_t) k]) : lo[(size_t) k];

    // gate and the gated mean
    mixed.assign((size_t) n_embd, 0.0f);
    for (long long d = 0; d < n_embd; ++d) {
        float m = 0.0f;
        for (long long c = 0; c < hc; ++c) {
            const long long i = c * n_embd + d;
            double a = 0;
            for (long long k = 0; k < hc_lr; ++k) a += (double) lq[(size_t) k] * (double) w_up[(size_t) (i * hc_lr + k)];
            const float g = (float) a;
            const float s = o.sigmoid_on_gate ? 1.0f / (1.0f + std::exp(-g)) : g / (1.0f + std::exp(-g));
            m += xn[(size_t) i] * s;
        }
        mixed[(size_t) d] = o.mean_over_streams ? m / (float) hc : m;
    }

    inject.assign((size_t) hc, 0.0f);
    for (long long c = 0; c < hc; ++c) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_inject[(size_t) (c * hc_dim + i)];
        inject[(size_t) c] = (float) a;
    }
}

/// Normalised L1 difference.  `mag_out` returns the mean |a|, for reporting.
double rel_diff(const std::vector<float>& a, const std::vector<float>& b, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / (double) (a.empty() ? 1 : a.size());
    return d / (mag > 1e-30 ? mag : 1e-30);
}

/// Relative error against the magnitude of the TERMS, not of the result.
///
/// `R + block_out * w` cancels wherever block_out*w is close to -R, and a plain |want-got|/|want| then
/// reports the CONDITION NUMBER instead of the arithmetic.  This is the same metric mistake the project has
/// now made four times: rounds 169 (Q4_K), 189 (RoPE), 194 (shared_expert) and this one.  For a sum, the
/// denominator is the sum of the term magnitudes.
double rel_terms(double want, double got, double term_a, double term_b) {
    const double den = std::fabs(term_a) + std::fabs(term_b);
    return std::fabs(want - got) / (den > 1e-30 ? den : 1e-30);
}

const char* activation_mode_name(int mode) {
    return mode == 2 ? "native pinned GR" : (mode == 1 ? "FP32" : "BF16");
}
void select_activation_mode(int mode) {
    strata::kernels::gr_set_native_mmvf(mode == 2);
    // In mode 2 the precision-only switch is deliberately false: native MMVF must imply FP32 by itself.
    strata::kernels::gr_set_fp32_activations(mode == 1);
}

// A diagonal GR fixture makes activation precision observable without summation-order ambiguity.
// Two-wide matrices satisfy native MMVF's pair ABI, while each nonzero dot has only one nonzero product.
// At R=1 and eps=0, xn=gamma exactly; any rounding in xn or lo comes from the selected activation contract.
int scalar_activation_contract() {
    using namespace strata::kernels;
    const GrShapes sh{2, 1, 2};
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t* d_weights = nullptr;
    void* d_scratch = nullptr;
    d_R = static_cast<decltype(d_R)>(strata::parity::alloc_bytes(2 * sizeof(float)));
    d_norm = static_cast<decltype(d_norm)>(strata::parity::alloc_bytes(2 * sizeof(float)));
    d_mixed = static_cast<decltype(d_mixed)>(strata::parity::alloc_bytes(2 * sizeof(float)));
    d_inject = static_cast<decltype(d_inject)>(strata::parity::alloc_bytes(sizeof(float)));
    d_weights = static_cast<decltype(d_weights)>(strata::parity::alloc_bytes(10 * sizeof(uint16_t)));
    d_scratch = static_cast<decltype(d_scratch)>(strata::parity::alloc_bytes(gr_workspace_bytes(sh)));
    GrWorkspace ws;
    gr_workspace_init(sh, d_scratch, ws);
    const float ones[] = {1.0f, 1.0f};
    const uint16_t weights[] = {
        bf16_bits(0.5f), 0, 0, 0,             // down: lo[0]=silu(xn[0]/2), lo[1]=0
        bf16_bits(1.5f), 0, bf16_bits(1.5f), 0, // both gate rows consume lo[0]
        bf16_bits(2.0f), 0                     // injection: 2*xn[0]
    };
    strata::parity::copy_bytes_in(d_R, ones, sizeof(ones));
    strata::parity::copy_bytes_in(d_weights, weights, sizeof(weights));
    namespace sx = sycl::ext::oneapi::experimental;
    auto& cq = strata::core::Runtime::get().compute();
    void* stream = &cq;
    int bad = 0;
    for (float gamma : {1.0f, 1.00390625f}) {
        const float gammas[] = {gamma, gamma};
        strata::parity::copy_bytes_in(d_norm, gammas, sizeof(gammas));
        float mixed_by_mode[3][2] = {}, inject_by_mode[3] = {};
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            strata::parity::sync();
            sx::command_graph graph(cq.get_context(), cq.get_device());
            graph.begin_recording(cq);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            graph.end_recording(cq);
            auto executable = graph.finalize();
            // Changing both host options after capture cannot change the graph's selected kernels.
            select_activation_mode((mode + 1) % 3);
            // Warmup results must not make a missing/no-op captured launch pass.
            // Poison the outputs and intermediate workspace before replay.
            cq.memset(d_mixed, 0xa5, 2 * sizeof(float));
            cq.memset(d_inject, 0xa5, sizeof(float));
            cq.memset(d_scratch, 0xa5, ws.bytes);
            cq.ext_oneapi_graph(executable);
            strata::parity::sync();
            strata::parity::copy_bytes_out(mixed_by_mode[mode], d_mixed, 2 * sizeof(float));
            strata::parity::copy_bytes_out(&inject_by_mode[mode], d_inject, sizeof(float));
            const float activation = mode ? gamma : to_bf16(gamma);
            const float projection = activation * 0.5f;
            const float lo = projection / (1.0f + std::exp(-projection));
            const float gate = (mode ? lo : to_bf16(lo)) * 1.5f;
            const float expected_mixed = gamma / (1.0f + std::exp(-gate));
            const float expected_inject = activation * 2.0f;
            // Each mode has its own activation contract and bounded scalar-formula
            // check. Native fast exp/div need not match ordinary FP32 postops bit
            // for bit: native_gr_postops_parity establishes that arithmetic with
            // actual pinned CUDA graphs. The nonzero projection/injection here
            // has only one exactly representable product and remains exact.
            const bool ok = std::fabs(mixed_by_mode[mode][0] - expected_mixed) <= 2e-7f &&
                            mixed_by_mode[mode][0] == mixed_by_mode[mode][1] &&
                            inject_by_mode[mode] == expected_inject;
            std::printf("  scalar gamma %.8f %s activation/capture %s (mixed %.9f, ref %.9f)\n",
                        gamma, activation_mode_name(mode), ok ? "pass" : "FAIL",
                        mixed_by_mode[mode][0], expected_mixed);
            if (!ok) ++bad;

            const float sentinel = -73.25f;
            strata::parity::copy_bytes_in(d_inject, &sentinel, sizeof(float));
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, nullptr, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            strata::parity::sync();
            float final_mixed[2] = {}, untouched = 0.0f;
            strata::parity::copy_bytes_out(final_mixed, d_mixed, sizeof(final_mixed));
            strata::parity::copy_bytes_out(&untouched, d_inject, sizeof(float));
            const bool final_ok = std::memcmp(final_mixed, mixed_by_mode[mode], sizeof(final_mixed)) == 0 && untouched == sentinel;
            std::printf("  scalar final mixer %s %s\n", activation_mode_name(mode), final_ok ? "pass" : "FAIL");
            if (!final_ok) ++bad;
        }
        if (std::fabs(mixed_by_mode[1][0] - mixed_by_mode[0][0]) <= 1e-5f ||
            (gamma != 1.0f && inject_by_mode[0] == inject_by_mode[1])) {
            std::printf("  scalar precision fixture is not observable\n");
            ++bad;
        }
        if (inject_by_mode[1] != inject_by_mode[2]) {
            std::printf("  native MMVF changed the exact scalar FP32 injection projection\n");
            ++bad;
        }
    }
    select_activation_mode(0);
    return bad;
}

int fused_multi_lds_parity(const float* d_norm, const uint16_t* d_down, const uint16_t* d_up,
                           const uint16_t* d_inject, float eps) {
    using namespace strata::kernels;
    constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, T = kFusedGrMaxT;
    std::mt19937 rng(0x6f8a);  // NOLINT(bugprone-random-generator-seed): reproducible parity fixture.
    std::normal_distribution<float> normal(0.0f, 0.3f);
    std::vector<float> r((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC);
    for (auto& x : r) x = normal(rng);
    for (auto& x : bo) x = normal(rng);
    for (auto& x : inj) x = normal(rng);

    float *d_r = nullptr, *d_r_out = nullptr, *d_bo = nullptr, *d_inj = nullptr;
    float *d_lo = nullptr, *d_rs = nullptr, *d_inj_out = nullptr, *d_mixed = nullptr, *d_xn = nullptr;
    check(strata::gpu::alloc_device(&d_r, r.size() * sizeof(float)), "multi R");
    check(strata::gpu::alloc_device(&d_r_out, r.size() * sizeof(float)), "multi R_out");
    check(strata::gpu::alloc_device(&d_bo, bo.size() * sizeof(float)), "multi bo");
    check(strata::gpu::alloc_device(&d_inj, inj.size() * sizeof(float)), "multi inj");
    check(strata::gpu::alloc_device(&d_lo, (size_t) T * LR * sizeof(float)), "multi lo");
    check(strata::gpu::alloc_device(&d_rs, (size_t) T * HC * sizeof(float)), "multi rs");
    check(strata::gpu::alloc_device(&d_inj_out, (size_t) T * HC * sizeof(float)), "multi injection");
    check(strata::gpu::alloc_device(&d_mixed, (size_t) T * N * sizeof(float)), "multi mixed");
    check(strata::gpu::alloc_device(&d_xn, (size_t) T * D * sizeof(float)), "multi xn");
    check(strata::gpu::copy(d_r, r.data(), r.size() * sizeof(float)), "multi copy R");
    check(strata::gpu::copy(d_bo, bo.data(), bo.size() * sizeof(float)), "multi copy bo");
    check(strata::gpu::copy(d_inj, inj.data(), inj.size() * sizeof(float)), "multi copy inj");

    std::vector<FusedGrArgs> args(T);
    for (int t = 0; t < T; ++t) {
        auto& a = args[t];
        a.R = d_r + (size_t) t * D;
        a.R_out = d_r_out + (size_t) t * D;
        a.apply = true;
        a.bo_prev = d_bo + (size_t) t * N;
        a.inj_prev = d_inj + (size_t) t * HC;
        a.w_norm = d_norm;
        a.w_down = d_down;
        a.w_up = d_up;
        a.w_inject = d_inject;
        a.eps = eps;
        a.lo = d_lo + (size_t) t * LR;
        a.rs = d_rs + (size_t) t * HC;
        a.inject_out = d_inj_out + (size_t) t * HC;
        a.mixed = d_mixed + (size_t) t * N;
    }

    struct Snapshot {
        std::vector<float> r_out, lo, rs, inject, mixed;
    };
    auto snapshot = [&]() {
        Snapshot s;
        s.r_out.resize((size_t) T * D);
        s.lo.resize((size_t) T * LR);
        s.rs.resize((size_t) T * HC);
        s.inject.resize((size_t) T * HC);
        s.mixed.resize((size_t) T * N);
        check(strata::gpu::copy(s.r_out.data(), d_r_out, s.r_out.size() * sizeof(float)), "multi read R_out");
        check(strata::gpu::copy(s.lo.data(), d_lo, s.lo.size() * sizeof(float)), "multi read lo");
        check(strata::gpu::copy(s.rs.data(), d_rs, s.rs.size() * sizeof(float)), "multi read rs");
        check(strata::gpu::copy(s.inject.data(), d_inj_out, s.inject.size() * sizeof(float)), "multi read inject");
        check(strata::gpu::copy(s.mixed.data(), d_mixed, s.mixed.size() * sizeof(float)), "multi read mixed");
        return s;
    };
    auto same = [&](const Snapshot& a, const Snapshot& b) {
        return std::memcmp(a.r_out.data(), b.r_out.data(), a.r_out.size() * sizeof(float)) == 0 &&
               std::memcmp(a.lo.data(), b.lo.data(), a.lo.size() * sizeof(float)) == 0 &&
               std::memcmp(a.rs.data(), b.rs.data(), a.rs.size() * sizeof(float)) == 0 &&
               std::memcmp(a.inject.data(), b.inject.data(), a.inject.size() * sizeof(float)) == 0 &&
               std::memcmp(a.mixed.data(), b.mixed.data(), a.mixed.size() * sizeof(float)) == 0;
    };

    strata::gpu::Stream stream = nullptr;
    check(((stream = strata::gpu::stream_create()) != nullptr), "multi stream");
    // Max T forces the HIP kernel's full dynamic-LDS request: 8 * 1280 * sizeof(float) = 40 KiB.
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(strata::gpu::stream_sync(stream), "multi max-T sync");
    const Snapshot multi = snapshot();
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(strata::gpu::stream_sync(stream), "single reference sync");
    const Snapshot single = snapshot();
    int bad = 0;
    if (!same(multi, single)) {
        std::printf("  fused GR multi max-T differs from single-token calls\n");
        ++bad;
    }

    strata::gpu::Graph* graph = nullptr;
    strata::gpu::Graph* graph_exec = nullptr;
    check(strata::gpu::begin_capture(stream), "multi graph begin");
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(strata::gpu::end_capture(stream, &graph), "multi graph end");
    graph_exec = graph; graph = nullptr;
    check(strata::gpu::graph_launch(graph_exec, stream), "multi graph initial replay");
    check(strata::gpu::stream_sync(stream), "multi graph initial sync");
    const Snapshot captured = snapshot();
    if (!same(multi, captured)) {
        std::printf("  fused GR multi captured graph differs from direct max-T call\n");
        ++bad;
    }

    // Reuse the same captured pointers with new payloads; then independently run the single-token path again.
    for (size_t i = 0; i < r.size(); ++i) r[i] = -0.7f * r[i] + 0.001f * (float) (i % 17);
    for (size_t i = 0; i < bo.size(); ++i) bo[i] = -0.4f * bo[i] + 0.02f;
    for (size_t i = 0; i < inj.size(); ++i) inj[i] += 0.3f;
    check(strata::gpu::copy_async(d_r, r.data(), r.size() * sizeof(float), stream), "multi replay R");
    check(strata::gpu::copy_async(d_bo, bo.data(), bo.size() * sizeof(float), stream), "multi replay bo");
    check(strata::gpu::copy_async(d_inj, inj.data(), inj.size() * sizeof(float), stream), "multi replay inj");
    check(strata::gpu::graph_launch(graph_exec, stream), "multi graph changed replay");
    check(strata::gpu::stream_sync(stream), "multi graph changed sync");
    const Snapshot replay = snapshot();
    if (std::memcmp(multi.mixed.data(), replay.mixed.data(), multi.mixed.size() * sizeof(float)) == 0) {
        std::printf("  fused GR graph replay ignored changed inputs\n");
        ++bad;
    }
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(strata::gpu::stream_sync(stream), "changed single reference sync");
    if (!same(replay, snapshot())) {
        std::printf("  fused GR changed graph replay differs from single-token calls\n");
        ++bad;
    }

    std::printf("  fused GR multi max-T=8 LDS launch and changing graph replay %s\n",
                bad == 0 ? "pass" : "FAIL");
    strata::gpu::graph_destroy(graph_exec);
    strata::gpu::graph_destroy(graph);
    strata::gpu::stream_destroy(stream);
    strata::gpu::free(d_xn);
    strata::gpu::free(d_mixed);
    strata::gpu::free(d_inj_out);
    strata::gpu::free(d_rs);
    strata::gpu::free(d_lo);
    strata::gpu::free(d_inj);
    strata::gpu::free(d_bo);
    strata::gpu::free(d_r_out);
    strata::gpu::free(d_r);
    return bad;
}

}  // namespace

int main(int argc, char** argv) try {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gr_parity [--selftest]\n"); return 2; }
    }

    const long long n_embd = 256, hc = 4, hc_lr = 32;
    const long long hc_dim = hc * n_embd;
    const float eps = 1e-6f;
    std::mt19937 rng(1234);  // NOLINT(bugprone-random-generator-seed): reproducible parity fixture.
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Weights are BF16-VALUED f32, which is what the pack holds for a BF16 source type.  Generating plain f32
    // weights would make the kernel (which stores 16 bits) differ from the reference by the rounding itself,
    // and the test would then be measuring the fixture.
    auto bf16_weight = [&](size_t n, float sigma) {
        std::vector<float> v(n);
        for (auto& x : v) x = to_bf16(gauss(rng) * sigma);
        return v;
    };
    std::vector<float> R((size_t) hc_dim);
    for (auto& x : R) x = gauss(rng);
    // STREAM SCALES 1, 4, 16, 64.  Without them trap 1 is only ~5% observable, because four streams of i.i.d.
    // normal noise have nearly the same RMS by accident - so a whole-stack RMS and a per-stream RMS agree to
    // within the noise, and the fixture would be asserting a distinction it cannot see.  Giving the streams
    // different magnitudes is what makes "per-stream" a claim with content.
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) R[(size_t) (c * n_embd + d)] *= std::pow(4.0f, (float) c);
    // gamma stored as (1 + w), so values near 1 - a reader expecting a plain scale is the one this catches
    std::vector<float> w_norm((size_t) hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<float> w_down = bf16_weight((size_t) (hc_lr * hc_dim), 0.2f);
    std::vector<float> w_up = bf16_weight((size_t) (hc_dim * hc_lr), 0.2f);
    std::vector<float> w_inject = bf16_weight((size_t) (hc * hc_dim), 0.05f);

    // NO PERMUTATION: the kernel takes both matrices exactly as the manifest (and `ref/gr.py`) store them.
    // `w_down` is (hc_lr, hc_dim) row-major and `w_up` is (hc_dim, hc_lr) row-major, so the bit-copy below is
    // the whole conversion.  An earlier version of this kernel wanted both TRANSPOSED - the test built them
    // that way and asserted the orientation observable - and that is gone because the warp-per-row mapping
    // makes it unnecessary.  What replaces it is the general orientation trap further down: a weight array
    // read with the wrong index expression must still produce a visibly different answer.
    std::vector<uint16_t> q_down(w_down.size()), q_up(w_up.size()), q_inject(w_inject.size());
    for (size_t i = 0; i < w_down.size(); ++i) q_down[i] = bf16_bits(w_down[i]);
    for (size_t i = 0; i < w_up.size(); ++i) q_up[i] = bf16_bits(w_up[i]);
    for (size_t i = 0; i < w_inject.size(); ++i) q_inject[i] = bf16_bits(w_inject[i]);

    // ---- device side
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t *d_down = nullptr, *d_up = nullptr, *d_inj = nullptr;
    d_R = static_cast<decltype(d_R)>(strata::parity::alloc_bytes(R.size() * sizeof(float)));
    d_norm = static_cast<decltype(d_norm)>(strata::parity::alloc_bytes(w_norm.size() * sizeof(float)));
    d_down = static_cast<decltype(d_down)>(strata::parity::alloc_bytes(q_down.size() * sizeof(uint16_t)));
    d_up = static_cast<decltype(d_up)>(strata::parity::alloc_bytes(q_up.size() * sizeof(uint16_t)));
    d_inj = static_cast<decltype(d_inj)>(strata::parity::alloc_bytes(q_inject.size() * sizeof(uint16_t)));
    d_mixed = static_cast<decltype(d_mixed)>(strata::parity::alloc_bytes((size_t) n_embd * sizeof(float)));
    d_inject = static_cast<decltype(d_inject)>(strata::parity::alloc_bytes((size_t) hc * sizeof(float)));
    strata::parity::copy_bytes_in(d_R, R.data(), R.size() * sizeof(float));
    strata::parity::copy_bytes_in(d_norm, w_norm.data(), w_norm.size() * sizeof(float));
    strata::parity::copy_bytes_in(d_down, q_down.data(), q_down.size() * sizeof(uint16_t));
    strata::parity::copy_bytes_in(d_up, q_up.data(), q_up.size() * sizeof(uint16_t));
    strata::parity::copy_bytes_in(d_inj, q_inject.data(), q_inject.size() * sizeof(uint16_t));

    const strata::kernels::GrShapes sh{n_embd, hc, hc_lr};
    // DEVICE memory: the workspace is written by the kernel.
    void* d_ws_raw = nullptr;
    d_ws_raw = static_cast<decltype(d_ws_raw)>(strata::parity::alloc_bytes(strata::kernels::gr_workspace_bytes(sh)));
    strata::kernels::GrWorkspace ws;
    strata::kernels::gr_workspace_init(sh, d_ws_raw, ws);
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
    std::vector<float> got_mixed((size_t) n_embd), got_inject((size_t) hc);
    strata::parity::copy_bytes_out(got_mixed.data(), d_mixed, got_mixed.size() * sizeof(float));
    strata::parity::copy_bytes_out(got_inject.data(), d_inject, got_inject.size() * sizeof(float));

    int bad = 0;

    // ---- the traps, asserted OBSERVABLE before the kernel is judged against the right reading
    Opts right;
    std::vector<float> want_mixed, want_inject;
    reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, right, want_mixed, want_inject);

    struct Trap { const char* name; Opts o; double floor; };
    const Trap traps[] = {
        {"per-stream RMSNorm vs whole-stack", {false, true, true, true, true}, 0.05},
        {"/hc inside the silu vs outside", {true, false, true, true, true}, 0.05},
        {"SiLU on lo / sigmoid on gate, swapped", {true, true, false, true, true}, 0.05},
        {"mean over streams vs sum", {true, true, true, false, true}, 0.50},
        {"BF16 vs FP32 activation contracts", {true, true, true, true, false}, 1e-4},
    };
    for (const Trap& t : traps) {
        std::vector<float> m, i;
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, t.o, m, i);
        const double rel = rel_diff(want_mixed, m);
        const bool observable = rel > t.floor;
        std::printf("  %-40s %-4s (%.4f%% apart, floor %.3f%%)\n", t.name, observable ? "yes" : "*** NO ***",
                    rel * 100, t.floor * 100);
        if (!observable) {
            std::printf("      *** the fixture cannot see this trap, so passing proves nothing about it ***\n");
            ++bad;
        }
    }

    // ---- the kernel against the correct reference
    double mag = 0;
    const double rel_mixed = rel_diff(want_mixed, got_mixed, &mag);
    std::printf("\n  %-40s worst %.3e (mean |ref| %.4f)\n", "mixed vs reference", rel_mixed, mag);

    // The tolerance is set by SUMMATION ORDER, not by the structure: both sides round the activation to bf16
    // and both weights are bf16-valued, so every product is exact in f32 and only the order differs.  A
    // structural error is several orders of magnitude larger - the trap floors above say so.
    if (!(rel_mixed <= 1e-4)) {
        std::printf("    *** over 1e-4, which summation order does not explain - look for a structural bug ***\n");
        ++bad;
    }

    const double rel_inject = rel_diff(want_inject, got_inject);
    std::printf("  %-40s rel %.3e\n", "inject vs reference", rel_inject);
    if (!(rel_inject <= 1e-4)) {
        // Say WHICH elements and by how much: a whole-array L1 hides whether this is one stream wrong (an
        // indexing bug) or all of them slightly wrong (a precision bug), and those need different responses.
        std::printf("    want:");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) want_inject[(size_t) c]);
        std::printf("\n    got :");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) got_inject[(size_t) c]);
        std::printf("\n");
        ++bad;
    }

    // Same geometry and weights with the CUDA single-token activation contract.  The scalar fixture
    // below is the independent precision check; this case exercises multi-stream reductions and indexing.
    {
        Opts fp32;
        fp32.round_activation = false;
        std::vector<float> wm, wi, gm((size_t) n_embd), gi((size_t) hc);
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, fp32, wm, wi);
        strata::kernels::gr_set_fp32_activations(true);
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        strata::parity::copy_bytes_out(gm.data(), d_mixed, gm.size() * sizeof(float));
        strata::parity::copy_bytes_out(gi.data(), d_inject, gi.size() * sizeof(float));
        strata::kernels::gr_set_fp32_activations(false);
        const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
        const bool ok = rm <= 1e-4 && ri <= 1e-4;
        std::printf("  FP32 activations vs reference: %s (mixed %.3e, inject %.3e)\n",
                    ok ? "pass" : "FAIL", rm, ri);
        if (!ok) ++bad;
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        strata::parity::copy_bytes_out(gm.data(), d_mixed, gm.size() * sizeof(float));
        strata::parity::copy_bytes_out(gi.data(), d_inject, gi.size() * sizeof(float));
        const bool restored = std::memcmp(gm.data(), got_mixed.data(), gm.size() * sizeof(float)) == 0 &&
                              std::memcmp(gi.data(), got_inject.data(), gi.size() * sizeof(float)) == 0;
        std::printf("  restoring default activation contract: %s\n", restored ? "byte-identical" : "FAIL");
        if (!restored) ++bad;
    }

    // ---- THE WEIGHT ORIENTATION MUST BE OBSERVABLE.  The kernel now takes both matrices exactly as the
    // manifest stores them, so there is no permutation for a loader to forget - but the INDEX EXPRESSIONS
    // inside the kernel are still a place to be wrong, and a wrong one gives right-shaped, wrong-valued
    // output.  Feed transposed copies through the same code path and require the answer to differ.
    {
        uint16_t *d_down_bad = nullptr, *d_up_bad = nullptr;
        d_down_bad = static_cast<decltype(d_down_bad)>(strata::parity::alloc_bytes(q_down.size() * sizeof(uint16_t)));
        d_up_bad = static_cast<decltype(d_up_bad)>(strata::parity::alloc_bytes(q_up.size() * sizeof(uint16_t)));
        std::vector<uint16_t> tr_down(q_down.size()), tr_up(q_up.size());
        for (long long k = 0; k < hc_lr; ++k)
            for (long long i = 0; i < hc_dim; ++i)
                tr_down[(size_t) (i * hc_lr + k)] = q_down[(size_t) (k * hc_dim + i)];
        for (long long i = 0; i < hc_dim; ++i)
            for (long long k = 0; k < hc_lr; ++k)
                tr_up[(size_t) (k * hc_dim + i)] = q_up[(size_t) (i * hc_lr + k)];
        strata::parity::copy_bytes_in(d_down_bad, tr_down.data(), tr_down.size() * 2);
        strata::parity::copy_bytes_in(d_up_bad, tr_up.data(), tr_up.size() * 2);
        std::vector<float> bad_mixed((size_t) n_embd);
        float* d_bad = nullptr;
        d_bad = static_cast<decltype(d_bad)>(strata::parity::alloc_bytes(bad_mixed.size() * sizeof(float)));
        strata::kernels::gr_read(d_R, d_norm, d_down_bad, d_up_bad, d_inj, eps, sh, ws, d_bad, d_inject,
                                 nullptr);
        strata::parity::copy_bytes_out(bad_mixed.data(), d_bad, bad_mixed.size() * sizeof(float));
        const double rel = rel_diff(want_mixed, bad_mixed);
        const bool visible = rel > 0.05;
        std::printf("  %-40s %s (%.1f%% apart)\n", "weight orientation is observable",
                    visible ? "yes" : "*** NO ***", rel * 100);
        if (!visible) ++bad;
        }

    // ---- the FINAL mixer passes w_inject = nullptr, and then nothing may be written
    float sentinel = -12345.0f;
    strata::parity::copy_bytes_in(d_inject, &sentinel, sizeof(float));
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, nullptr, eps, sh, ws, d_mixed, d_inject, nullptr);
    float after = 0.0f;
    strata::parity::copy_bytes_out(&after, d_inject, sizeof(float));
    const bool untouched = (after == sentinel);
    std::printf("  %-40s %s\n", "null w_inject writes nothing", untouched ? "yes" : "*** NO ***");
    if (!untouched) ++bad;

    // ---- gr_write: the `2*sigmoid` centring, asserted as the PROPERTY the source comment claims
    std::vector<float> block_out((size_t) n_embd);
    for (auto& x : block_out) x = gauss(rng);
    std::vector<float> zero_inj((size_t) hc, 0.0f);
    float *d_Rw = nullptr, *d_bo = nullptr, *d_zi = nullptr, *d_outw = nullptr;
    d_Rw = static_cast<decltype(d_Rw)>(strata::parity::alloc_bytes(R.size() * sizeof(float)));
    d_bo = static_cast<decltype(d_bo)>(strata::parity::alloc_bytes(block_out.size() * sizeof(float)));
    d_zi = static_cast<decltype(d_zi)>(strata::parity::alloc_bytes(zero_inj.size() * sizeof(float)));
    d_outw = static_cast<decltype(d_outw)>(strata::parity::alloc_bytes(R.size() * sizeof(float)));
    strata::parity::copy_bytes_in(d_Rw, R.data(), R.size() * sizeof(float));
    strata::parity::copy_bytes_in(d_bo, block_out.data(), block_out.size() * sizeof(float));
    strata::parity::copy_bytes_in(d_zi, zero_inj.data(), zero_inj.size() * sizeof(float));

    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    std::vector<float> got_write((size_t) hc_dim);
    strata::parity::copy_bytes_out(got_write.data(), d_outw, got_write.size() * sizeof(float));

    double worst_plain = 0;
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d];
            worst_plain = std::fmax(worst_plain, rel_terms(want, (double) got_write[i], (double) R[i],
                                                           (double) block_out[(size_t) d]));
        }
    const bool plain = worst_plain < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "zero injection -> plain residual add",
                plain ? "yes" : "*** NO ***", worst_plain);
    if (!plain) ++bad;

    // the rival reading: sigmoid without the 2 halves the update, and must be visible.
    //
    // The comparison is on the DELTA `out - R`, not on `out`.  The claim is about the gate WEIGHT, and `R`
    // now reaches 64 in the last stream, so measuring `R + block_out*w` against the wrong reading dilutes a
    // 100% error in `w` down to 2.27% - which is how this check first passed as "not observable" while being
    // perfectly observable.  Measure the quantity the claim is about.
    {
        std::vector<float> got_delta((size_t) hc_dim), want_delta((size_t) hc_dim), wrong_delta((size_t) hc_dim);
        for (long long c = 0; c < hc; ++c)
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                got_delta[i] = (float) ((double) got_write[i] - (double) R[i]);
                want_delta[i] = block_out[(size_t) d];        // 2*sigmoid(0) = 1.0
                wrong_delta[i] = block_out[(size_t) d] * 0.5f; // 1*sigmoid(0) = 0.5
            }
        const double rel = rel_diff(want_delta, wrong_delta);
        const double rel_got = rel_diff(want_delta, got_delta);
        const bool visible = rel > 0.05 && rel_got < 0.05;
        std::printf("  %-40s %s (readings %.0f%% apart, kernel within %.2e)\n",
                    "1*sigmoid vs 2*sigmoid is observable", visible ? "yes" : "*** NO ***", rel * 100, rel_got);
        if (!visible) ++bad;
    }

    // a non-zero injection: every stream receives the SAME block output, differing only by weight
    std::vector<float> inj((size_t) hc);
    for (auto& x : inj) x = gauss(rng) * 3.0f;
    strata::parity::copy_bytes_in(d_zi, inj.data(), inj.size() * sizeof(float));
    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    strata::parity::copy_bytes_out(got_write.data(), d_outw, got_write.size() * sizeof(float));
    double worst_stream = 0;
    for (long long c = 0; c < hc; ++c) {
        const double w = 2.0 / (1.0 + std::exp(-(double) inj[(size_t) c] / (double) hc));
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d] * w;
            worst_stream = std::fmax(worst_stream, rel_terms(want, (double) got_write[i], (double) R[i],
                                                             (double) block_out[(size_t) d] * w));
        }
    }
    const bool streams_ok = worst_stream < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "every stream adds the same block output",
                streams_ok ? "yes" : "*** NO ***", worst_stream);
    if (!streams_ok) ++bad;

    // ---- THE REAL DIMENSIONS. Exercise all activation/projection variants at n_embd=2560, hc=4, hc_lr=320.
    // Workspace sizing and every full weight row must agree with the model geometry.
    {
        const long long rn = 2560, rhc = 4, rlr = 320, rdim = rhc * rn;
        std::mt19937 rrng(99);  // NOLINT(bugprone-random-generator-seed): reproducible parity fixture.
        std::normal_distribution<float> rg(0.0f, 1.0f);
        auto rbf16 = [&](size_t n, float sigma) {
            std::vector<float> v(n);
            for (auto& x : v) x = to_bf16(rg(rrng) * sigma);
            return v;
        };
        std::vector<float> rR((size_t) rdim);
        for (auto& x : rR) x = rg(rrng);
        for (long long c = 0; c < rhc; ++c)
            for (long long d = 0; d < rn; ++d) rR[(size_t) (c * rn + d)] *= std::pow(4.0f, (float) c);
        std::vector<float> rnorm((size_t) rdim);
        for (auto& x : rnorm) x = 1.0f + 0.1f * rg(rrng);
        std::vector<float> rdown = rbf16((size_t) (rlr * rdim), 0.2f);
        std::vector<float> rup = rbf16((size_t) (rdim * rlr), 0.2f);
        std::vector<float> rinj = rbf16((size_t) (rhc * rdim), 0.05f);

        std::vector<uint16_t> qd(rdown.size()), qu(rup.size()), qi(rinj.size());
        for (size_t i = 0; i < rdown.size(); ++i) qd[i] = bf16_bits(rdown[i]);
        for (size_t i = 0; i < rup.size(); ++i) qu[i] = bf16_bits(rup[i]);
        for (size_t i = 0; i < rinj.size(); ++i) qi[i] = bf16_bits(rinj[i]);

        float *dR = nullptr, *dN = nullptr, *dM = nullptr, *dI = nullptr;
        uint16_t *dD = nullptr, *dU = nullptr, *dJ = nullptr;
        dR = static_cast<decltype(dR)>(strata::parity::alloc_bytes(rR.size() * 4));
        dN = static_cast<decltype(dN)>(strata::parity::alloc_bytes(rnorm.size() * 4));
        dD = static_cast<decltype(dD)>(strata::parity::alloc_bytes(qd.size() * 2));
        dU = static_cast<decltype(dU)>(strata::parity::alloc_bytes(qu.size() * 2));
        dJ = static_cast<decltype(dJ)>(strata::parity::alloc_bytes(qi.size() * 2));
        dM = static_cast<decltype(dM)>(strata::parity::alloc_bytes((size_t) rn * 4));
        dI = static_cast<decltype(dI)>(strata::parity::alloc_bytes((size_t) rhc * 4));
        strata::parity::copy_bytes_in(dR, rR.data(), rR.size() * 4);
        strata::parity::copy_bytes_in(dN, rnorm.data(), rnorm.size() * 4);
        strata::parity::copy_bytes_in(dD, qd.data(), qd.size() * 2);
        strata::parity::copy_bytes_in(dU, qu.data(), qu.size() * 2);
        strata::parity::copy_bytes_in(dJ, qi.data(), qi.size() * 2);

        const strata::kernels::GrShapes rsh{rn, rhc, rlr};
        void* rws_raw = nullptr;
        rws_raw = static_cast<decltype(rws_raw)>(strata::parity::alloc_bytes(strata::kernels::gr_workspace_bytes(rsh)));
        strata::kernels::GrWorkspace rws;
        strata::kernels::gr_workspace_init(rsh, rws_raw, rws);
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            strata::kernels::gr_read(dR, dN, dD, dU, dJ, eps, rsh, rws, dM, dI, nullptr);
            std::vector<float> gm((size_t) rn), gi((size_t) rhc);
            strata::parity::copy_bytes_out(gm.data(), dM, gm.size() * 4);
            strata::parity::copy_bytes_out(gi.data(), dI, gi.size() * 4);

            Opts precision;
            precision.round_activation = mode == 0;
            std::vector<float> wm, wi;
            reference(rR, rnorm, rdown, rup, rinj, eps, rn, rhc, rlr, precision, wm, wi);
            const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
            const bool ok = rm <= 1e-4 && ri <= 1e-4;
            std::printf("\n  real dims 2560/4/320 %s %s (mixed %.3e, inject %.3e)\n",
                        activation_mode_name(mode), ok ? "pass" : "*** FAIL ***", rm, ri);
            if (!ok) ++bad;
        }
        bad += fused_multi_lds_parity(dN, dD, dU, dJ, eps);
        select_activation_mode(0);
        }

    bad += scalar_activation_contract();
    std::printf("\ngr_read/gr_write: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gr_parity OK\n");
    return 0;
}
 catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
