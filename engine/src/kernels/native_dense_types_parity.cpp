// src/kernels/native_dense_types_parity.cpp - round 0214: the dense GGUF tensor types of the new Unsloth Qwen3.8-Flash-Next
// quants (UD-IQ1_M .. UD-IQ4_XS) through native_mmvq, on the real shapes, against ggml's own dequantizer.
//
//     build/native_dense_types_parity [--bench]
//
// Types: Q5_0, Q6_K, Q5_K, Q4_K, Q3_K, Q8_0, IQ4_NL, IQ4_XS, Q2_0 (the dense attention / SSM / shared-expert / head tensors
// and the token embedding of the files).  Each is quantized from random floats by ggml (an importance matrix of ones where
// the type needs one), then y = W x on the GPU (the engine's q8_1 activation, native_mmvq) is compared with a double
// reference built from ggml's to_float of the same blocks and the float x: the only difference is the activation rounding,
// so the relative L1 error must stay under 2e-2 (measured values are printed).  Also checked: ncols = 1..4 columns (the
// exact multi-column layout) against single-column calls, bit for bit, and a zero activation giving a zero output.
// --bench prints the single-column decode time of the type at 2560 x 8192 (a fused QKV-sized tensor), microseconds and GB/s.
// Exit 77 without a GPU.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
int g_fail = 0;

struct Case { const char* name; ggml_type t; int n_in; };

std::vector<uint8_t> make_weights(ggml_type t, int n_out, int n_in, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) n_out * n_in);
    for (int r = 0; r < n_out; ++r) {
        const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
        for (int c = 0; c < n_in; ++c) w[(size_t) r * n_in + c] = scale * nd(rng);
    }
    const std::vector<float> imatrix((size_t) n_in, 1.0f);
    std::vector<uint8_t> q((size_t) n_out * ggml_row_size(t, n_in));
    ggml_quantize_chunk(t, w.data(), q.data(), 0, n_out, n_in, ggml_quantize_requires_imatrix(t) ? imatrix.data() : nullptr);
    return q;
}

double rel(const std::vector<float>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs(b[i]); }
    return n / (d + 1e-30);
}

void run(const Case& c, int n_out, bool bench, cudaStream_t s, int NC = 4) {
    const int n_in = c.n_in;
    if (!K::native_mmvq_supported((int) c.t)) {
        std::printf("%-7s: native_mmvq_supported = false  FAIL\n", c.name);
        ++g_fail;
        return;
    }
    const auto w = make_weights(c.t, n_out, n_in, 17 + (int) c.t);
    const size_t row = ggml_row_size(c.t, n_in);
    // float reference from ggml's own dequantizer
    std::vector<float> wf((size_t) n_out * n_in);
    const auto* tr = ggml_get_type_traits(c.t);
    for (int r = 0; r < n_out; ++r) tr->to_float(w.data() + (size_t) r * row, wf.data() + (size_t) r * n_in, n_in);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NC * n_in);
    for (auto& v : x) v = nd(rng);
    std::fill(x.begin() + (size_t) (NC - 1) * n_in, x.end(), 0.0f);   // the last column: all zero
    std::vector<double> ref((size_t) NC * n_out);
    for (int j = 0; j < NC; ++j)
        for (int r = 0; r < n_out; ++r) {
            double acc = 0;
            for (int k = 0; k < n_in; ++k) acc += (double) wf[(size_t) r * n_in + k] * x[(size_t) j * n_in + k];
            ref[(size_t) j * n_out + r] = acc;
        }
    uint8_t* dw;
    float *dx, *dy;
    void* dxq;
    cudaMalloc((void**) &dw, w.size());
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMalloc((void**) &dy, (size_t) NC * n_out * 4);
    cudaMalloc(&dxq, K::native_q8_1_bytes(n_in, NC));
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    K::native_quantize_q8_1(dx, dxq, n_in, NC, s);
    std::vector<float> multi((size_t) NC * n_out), single((size_t) NC * n_out);
    K::native_mmvq((int) c.t, dw, dxq, dy, n_in, n_out, NC, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(multi.data(), dy, multi.size() * 4, cudaMemcpyDeviceToHost);
    for (int j = 0; j < NC; ++j)
        K::native_mmvq((int) c.t, dw, (const uint8_t*) dxq + (size_t) j * (n_in / 32) * 36, dy + (size_t) j * n_out, n_in, n_out, 1, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(single.data(), dy, single.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> m3(multi.begin(), multi.begin() + (size_t) (NC - 1) * n_out);
    std::vector<double> r3(ref.begin(), ref.begin() + (size_t) (NC - 1) * n_out);
    const double e = rel(m3, r3);
    const bool same = std::memcmp(multi.data(), single.data(), multi.size() * 4) == 0;
    double zmax = 0;
    for (int r = 0; r < n_out; ++r) zmax = std::max(zmax, (double) std::fabs(multi[(size_t) (NC - 1) * n_out + r]));
    const bool finite = std::all_of(multi.begin(), multi.end(), [](float v) { return std::isfinite(v); });
    const bool ok = e < 2e-2 && same && zmax == 0.0 && finite;
    std::printf("%-7s %6d x %-5d rel L1 vs ggml dequant %.2e  %d-col == 1-col bitwise: %s  zero column max %.1g  %s\n", c.name,
                n_out, n_in, e, NC, same ? "yes" : "NO", zmax, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
    if (bench) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        // a pool of copies past the last-level cache, so each call reads cold weights
        const int pool = std::max(2, (int) (((size_t) 400 << 20) / w.size()));
        uint8_t* dp;
        cudaMalloc((void**) &dp, (size_t) pool * w.size());
        for (int i = 0; i < pool; ++i) cudaMemcpy(dp + (size_t) i * w.size(), w.data(), w.size(), cudaMemcpyHostToDevice);
        std::vector<double> us;
        for (int rep = 0; rep < 7; ++rep) {
            for (int i = 0; i < 10; ++i) K::native_mmvq((int) c.t, dp + (size_t) (i % pool) * w.size(), dxq, dy, n_in, n_out, 1, s);
            cudaStreamSynchronize(s);
            double tot = 0;
            const int it = 100;
            for (int i = 0; i < it; ++i) {
                float ms = 0;
                cudaEventRecord(e0, s);
                K::native_mmvq((int) c.t, dp + (size_t) (i % pool) * w.size(), dxq, dy, n_in, n_out, 1, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                tot += ms;
            }
            us.push_back(tot / it * 1e3);
        }
        std::sort(us.begin(), us.end());
        std::printf("BENCH %-7s %5d x %-5d decode (1 col): median %.1f us, %.0f GB/s\n", c.name, n_out, n_in, us[3],
                    (double) w.size() / us[3] / 1e3);
        cudaFree(dp);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dxq);
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool bench = false, head = false;
    for (int i = 1; i < argc; ++i) {
        bench = bench || std::string(argv[i]) == "--bench";
        head = head || std::string(argv[i]) == "--head";
    }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no GPU: skipped\n"); return 77; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    if (head) {   // --head: the output head's real shape (2560 x 248,320, the head types of the Flash-Next files) at 1 to 8 columns (a verify window's rows), #6030
        const Case heads[] = {{"Q8_0", GGML_TYPE_Q8_0, 2560}, {"Q5_K", GGML_TYPE_Q5_K, 2560}, {"Q6_K", GGML_TYPE_Q6_K, 2560}};
        for (const Case& c : heads)
            for (int nc : {2, 5, 8}) run(c, 248320, false, s, nc);
        std::printf("native_dense_types_parity --head: %d failures\n", g_fail);
        return g_fail ? 1 : 0;
    }
    // n_in 2560 is the model width (a multiple of 256 and 64); the down-shaped tensors run at 640 (32-block types only) and 2048
    const Case cases[] = {{"Q5_0", GGML_TYPE_Q5_0, 2560}, {"Q6_K", GGML_TYPE_Q6_K, 2560}, {"Q5_K", GGML_TYPE_Q5_K, 2560},
                          {"Q4_K", GGML_TYPE_Q4_K, 2560}, {"Q3_K", GGML_TYPE_Q3_K, 2560}, {"Q8_0", GGML_TYPE_Q8_0, 2560},
                          {"IQ4_NL", GGML_TYPE_IQ4_NL, 2560}, {"IQ4_XS", GGML_TYPE_IQ4_XS, 2560}, {"Q2_0", GGML_TYPE_Q2_0, 2560}};
    for (const Case& c : cases) run(c, bench ? 8192 : 1024, bench, s);
    const Case narrow[] = {{"Q5_0", GGML_TYPE_Q5_0, 640}, {"Q8_0", GGML_TYPE_Q8_0, 640}, {"IQ4_NL", GGML_TYPE_IQ4_NL, 640},
                           {"Q2_0", GGML_TYPE_Q2_0, 640}, {"Q6_K", GGML_TYPE_Q6_K, 2048}, {"Q5_K", GGML_TYPE_Q5_K, 2048}};
    for (const Case& c : narrow) run(c, 2560, false, s);
    std::printf("native_dense_types_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
