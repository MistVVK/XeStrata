// src/kernels/native_dense_parity.cpp - the dense GGUF projections of the Unsloth v3.1 Qwen3.8-Flash-Next sets
// (Q5_0, Q6_K, Q5_K, Q4_K, Q8_0, IQ4_XS, IQ4_NL, Q2_0 ...) on synthetic weights quantized by ggml, no model files.
//
//     build/native_dense_parity [--only=NAME]
//
// Decode: the GPU's q8_1 quantizer and native_mmvq (1, 2, 3, 4 and 8 columns) against a double-precision product of
// ggml's own dequantized weights (to_float) and the FP32 activations.  The only difference is the activations'
// q8_1 rounding, so the relative L1 error has to stay under 2e-2 (it is a few 1e-3; 4e-2 for Q4_0 / Q5_0, see run_case); columns of a multi-column call
// are bitwise the single-column call (native_mmvq_multi_exact, the default).
// Prompt (the dequantize-to-FP16/BF16 route that serves a type without an MMQ instance): dequant_f32 against
// to_float (relative error 1e-6), dequant_f16 and dequant_bf16 against the rounded to_float.
// Shapes: the model's hidden width 2560 (attention, SSM and shared-expert inputs, the head) and 640 (the shared
// expert's and the experts' FF width) for the formats with a block of 64 values or fewer.
#include "strata/kernels/dequant_bf16.hpp"
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

namespace k = strata::kernels;

namespace {
struct T { const char* name; ggml_type type; };
const T kTypes[] = {
    {"Q2_0", GGML_TYPE_Q2_0},   {"Q4_0", GGML_TYPE_Q4_0}, {"Q5_0", GGML_TYPE_Q5_0},   {"Q8_0", GGML_TYPE_Q8_0},
    {"IQ4_NL", GGML_TYPE_IQ4_NL}, {"IQ4_XS", GGML_TYPE_IQ4_XS}, {"Q3_K", GGML_TYPE_Q3_K}, {"Q4_K", GGML_TYPE_Q4_K},
    {"Q5_K", GGML_TYPE_Q5_K},   {"Q6_K", GGML_TYPE_Q6_K},
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("  CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

double rel_l1(const std::vector<float>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs(b[i]); }
    return n / (d + 1e-30);
}

uint16_t bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// returns the number of failed checks
int run_case(const T& t, int n_in, int n_out, int seed) {
    int fails = 0;
    const int64_t blck = ggml_blck_size(t.type);
    if (n_in % blck != 0) return 0;
    std::mt19937 rng(1000 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) n_out * n_in);
    for (int r = 0; r < n_out; ++r) {
        const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
        for (int c = 0; c < n_in; ++c) w[(size_t) r * n_in + c] = scale * nd(rng);
    }
    const size_t rb = ggml_row_size(t.type, n_in);
    std::vector<uint8_t> blob((size_t) n_out * rb);
    const std::vector<float> imatrix((size_t) n_in, 1.0f);
    ggml_quantize_chunk(t.type, w.data(), blob.data(), 0, n_out, n_in,
                        ggml_quantize_requires_imatrix(t.type) ? imatrix.data() : nullptr);
    const auto* tr = ggml_get_type_traits(t.type);
    std::vector<float> wd((size_t) n_out * n_in);
    for (int r = 0; r < n_out; ++r) tr->to_float(blob.data() + (size_t) r * rb, wd.data() + (size_t) r * n_in, n_in);

    constexpr int NC = 8;
    std::vector<float> x((size_t) NC * n_in);
    for (auto& v : x) v = nd(rng);
    for (int j = 0; j < NC; ++j) x[(size_t) j * n_in + (size_t) (j * 37 % n_in)] *= 12.f;   // an outlier per column
    std::vector<double> ref((size_t) NC * n_out);
    for (int j = 0; j < NC; ++j)
        for (int r = 0; r < n_out; ++r) {
            double a = 0;
            for (int c = 0; c < n_in; ++c) a += (double) wd[(size_t) r * n_in + c] * x[(size_t) j * n_in + c];
            ref[(size_t) j * n_out + r] = a;
        }

    cudaStream_t s;
    cudaStreamCreate(&s);
    void *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *dy = nullptr, *df = nullptr;
    uint16_t *dh = nullptr, *db = nullptr;
    const size_t welems = (size_t) n_out * n_in;
    bool ok = ck(cudaMalloc(&dw, blob.size()), "w") && ck(cudaMalloc((void**) &dx, x.size() * 4), "x") &&
              ck(cudaMalloc(&dxq, k::native_q8_1_bytes(n_in, NC)), "xq") && ck(cudaMalloc((void**) &dy, (size_t) NC * n_out * 4), "y") &&
              ck(cudaMalloc((void**) &df, welems * 4), "f32") && ck(cudaMalloc((void**) &dh, welems * 2), "f16") &&
              ck(cudaMalloc((void**) &db, welems * 2), "bf16");
    if (!ok) return 1;
    cudaMemcpy(dw, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);

    char line[256];
    std::string res;
    // decode: native_mmvq at 1, 2, 3, 4, 8 columns
    std::vector<float> y1((size_t) n_out);
    // Q4_0 / Q5_0 subtract an offset (8 / 16) times the q8_1 block's sum of the ORIGINAL activations (llama.cpp CUDA's form, the
    // same in ggml-cuda), so the activations' rounding shows 3x larger on these random zero-mean outputs: 1.6-2.3e-2 measured
    const double tol = (t.type == GGML_TYPE_Q4_0 || t.type == GGML_TYPE_Q5_0) ? 4e-2 : 2e-2;
    double worst = 0;
    bool cols_equal = true;
    if (!k::native_mmvq_supported((int) t.type)) {
        std::printf("%-8s n_in %-5d native_mmvq: NOT SUPPORTED\n", t.name, n_in);
        ++fails;
    } else {
        for (int nc : {1, 2, 3, 4, 8}) {
            cudaMemset(dy, 0xff, (size_t) NC * n_out * 4);
            k::native_quantize_q8_1(dx, dxq, n_in, nc, s);
            k::native_mmvq((int) t.type, dw, dxq, dy, n_in, n_out, nc, s);
            cudaStreamSynchronize(s);
            std::vector<float> got((size_t) nc * n_out);
            cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
            std::vector<double> rr(ref.begin(), ref.begin() + (ptrdiff_t) got.size());
            const double e = rel_l1(got, rr);
            worst = std::max(worst, e);
            if (!(e < tol) || !std::isfinite(e)) ++fails;
            if (nc == 1) y1.assign(got.begin(), got.end());
            else if (std::memcmp(got.data(), y1.data(), y1.size() * 4) != 0) cols_equal = false;
        }
        if (!cols_equal) ++fails;
        std::snprintf(line, sizeof line, "decode rel %.2e (1-8 cols), col0 same as 1 col: %s;", worst, cols_equal ? "yes" : "NO");
        res += line;
    }
    // prompt dequantizers
    if (!k::dequant_bf16_supported((int) t.type)) {
        res += " dequant: NOT SUPPORTED";
        ++fails;
    } else {
        std::vector<float> gf(welems);
        std::vector<uint16_t> gh(welems), gb(welems);
        k::dequant_f32((int) t.type, dw, 0, n_out, n_in, df, s);
        k::dequant_f16((int) t.type, dw, 0, n_out, n_in, dh, s);
        k::dequant_bf16((int) t.type, dw, 0, n_out, n_in, db, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(gf.data(), df, welems * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gh.data(), dh, welems * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(gb.data(), db, welems * 2, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        size_t d32 = 0, d16 = 0, dbf = 0;
        for (size_t i = 0; i < welems; ++i) {
            num += std::fabs((double) gf[i] - wd[i]);
            den += std::fabs((double) wd[i]);
            d32 += std::memcmp(&gf[i], &wd[i], 4) != 0;
            d16 += gh[i] != ggml_fp32_to_fp16(wd[i]);
            dbf += gb[i] != bf16_rne(wd[i]);
        }
        const double e32 = num / (den + 1e-30);
        // f32: a rounding at most (fused multiply-adds); f16/bf16: the rounded to_float, or one step off where the f32 differs
        const bool ok32 = e32 < 1e-6;
        const bool ok16 = d16 <= d32 && dbf <= d32;
        if (!ok32 || !ok16) ++fails;
        std::snprintf(line, sizeof line, " dequant f32 rel %.1e (%zu of %zu bits differ), f16 %zu / bf16 %zu differ", e32, d32, welems,
                      d16, dbf);
        res += line;
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy); cudaFree(df); cudaFree(dh); cudaFree(db);
    cudaStreamDestroy(s);
    std::printf("%-8s n_in %-5d n_out %-4d %s  %s\n", t.name, n_in, n_out, res.c_str(), fails ? "FAIL" : "ok");
    return fails;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::printf("no CUDA device\n"); return 77; }
    int fails = 0, seed = 0;
    for (const T& t : kTypes) {
        if (only && std::strcmp(only, t.name) != 0) continue;
        for (int n_in : {2560, 640, 4096, 512, 1024, 2048, 3072}) {
            if (n_in == 640 && ggml_blck_size(t.type) > 64) continue;   // (K-quants and IQ4_XS: 256-value blocks; 640 is not a multiple)
            if (n_in == 640 && ggml_blck_size(t.type) == 256) continue;
            fails += run_case(t, n_in, 512, seed++);
        }
    }
    std::printf("native_dense_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
