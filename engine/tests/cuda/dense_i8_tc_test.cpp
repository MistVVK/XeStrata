// tests/cuda/dense_i8_tc_test.cpp - Parity & throughput test for clean-room MIT INT8 TC dense GEMM (sm_80+).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Validates dense_i8_tc_mmvq (T = 1, 4, 8, 16) against native_mmvq AND dense_i8_tc_prefill (T = 64, 512, 2048)
// against dequant_f16 + FP16 cuBLAS across Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), IQ4_XS (23).
#include "strata/kernels/dense_i8_tc.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_i8_tc_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

void fill_weights(int ggml_type, int n_in, int n_out, std::vector<uint8_t>& buf, std::mt19937& rng) {
    buf.resize(strata::kernels::native_mmvq_weight_bytes(ggml_type, n_in, n_out));
    if (ggml_type == 8) {
        const int nb = n_in / 32;
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* p = buf.data() + ((size_t) r * nb + b) * 34u;
                const uint16_t d = __half_as_ushort(__float2half_rn(0.005f + 0.01f * ((float) (rng() & 255) / 255.0f)));
                p[0] = (uint8_t) (d & 0xffu); p[1] = (uint8_t) (d >> 8);
                for (int j = 0; j < 32; ++j) p[2 + j] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
            }
        }
    } else if (ggml_type == 12) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 144u;
                const half2 dm = __floats2half2_rn(0.004f + 0.006f * ((float) (rng() & 255) / 255.0f),
                                                   0.003f + 0.005f * ((float) (rng() & 255) / 255.0f));
                std::memcpy(p, &dm, 4);
                for (int j = 4; j < 144; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    } else if (ggml_type == 13) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 176u;
                const half2 dm = __floats2half2_rn(0.003f + 0.004f * ((float) (rng() & 255) / 255.0f),
                                                   0.002f + 0.004f * ((float) (rng() & 255) / 255.0f));
                std::memcpy(p, &dm, 4);
                for (int j = 4; j < 176; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    } else if (ggml_type == 14) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 210u;
                for (int j = 0; j < 192; ++j) p[j] = (uint8_t) (rng() & 0xffu);
                for (int s = 0; s < 16; ++s) p[192 + s] = (uint8_t) (int8_t) ((int) (rng() % 63) - 31);
                const uint16_t d = __half_as_ushort(__float2half_rn(0.003f + 0.006f * ((float) (rng() & 255) / 255.0f)));
                p[208] = (uint8_t) (d & 0xffu); p[209] = (uint8_t) (d >> 8);
            }
        }
    } else if (ggml_type == 23) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 136u;
                const uint16_t d = __half_as_ushort(__float2half_rn(0.001f + 0.002f * ((float) (rng() & 255) / 255.0f)));
                p[0] = (uint8_t) (d & 0xffu); p[1] = (uint8_t) (d >> 8);
                for (int j = 2; j < 136; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    }
}

void test_decode_type(int ggml_type, const char* name, int n_in, int n_out, bool bench, cudaStream_t s) {
    std::mt19937 rng(9000u + (unsigned) ggml_type);
    std::vector<uint8_t> h_w;
    fill_weights(ggml_type, n_in, n_out, h_w, rng);
    std::vector<float> h_x((size_t) 16 * (size_t) n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : h_x) v = nd(rng);

    void *d_w = nullptr, *d_xq = nullptr;
    float *d_x = nullptr, *d_ref = nullptr, *d_got = nullptr;
    ck(cudaMalloc(&d_w, h_w.size()), "malloc w");
    ck(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_xq, 2u * strata::kernels::native_q8_1_bytes(n_in, 8)), "malloc xq");
    ck(cudaMalloc(&d_ref, (size_t) 16 * (size_t) n_out * sizeof(float)), "malloc ref");
    ck(cudaMalloc(&d_got, (size_t) 16 * (size_t) n_out * sizeof(float)), "malloc got");
    ck(cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice), "H2D w");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    for (int T : {1, 4, 8, 16}) {
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int nt = std::min(8, T - t0);
            strata::kernels::native_quantize_q8_1(
                d_x + (size_t) t0 * n_in,
                reinterpret_cast<uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                n_in, nt, s);
        }
        for (int t0 = 0; t0 < T; t0 += 4) {
            const int nt = std::min(4, T - t0);
            strata::kernels::native_mmvq(
                ggml_type, d_w,
                reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                d_ref + (size_t) t0 * n_out, n_in, n_out, nt, s);
        }
        if (!strata::kernels::dense_i8_tc_mmvq(ggml_type, d_w, d_xq, d_got, n_in, n_out, T, s)) {
            std::fprintf(stderr, "dense_i8_tc_mmvq returned false for %s T=%d\n", name, T);
            std::exit(1);
        }
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> h_ref((size_t) T * n_out), h_got((size_t) T * n_out);
        ck(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H ref");
        ck(cudaMemcpy(h_got.data(), d_got, h_got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H got");

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < h_ref.size(); ++i) {
            const double df = (double) h_got[i] - (double) h_ref[i];
            num += df * df;
            den += (double) h_ref[i] * (double) h_ref[i];
        }
        const double rel_rms = den > 0.0 ? std::sqrt(num / den) : 0.0;
        if (!std::isfinite(rel_rms) || rel_rms > 1e-4) {
            std::fprintf(stderr, "FAIL decode %s %dx%d T=%d: rel_rms=%.6e\n", name, n_out, n_in, T, rel_rms);
            std::exit(1);
        }

        if (bench && (T == 4 || T == 8 || T == 16)) {
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");
            constexpr int NITER = 30;
            for (int i = 0; i < 5; ++i)
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_ref + (size_t) t0 * n_out, n_in, n_out, std::min(4, T - t0), s);
            ck(cudaEventRecord(e0, s), "r0");
            for (int i = 0; i < NITER; ++i)
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_ref + (size_t) t0 * n_out, n_in, n_out, std::min(4, T - t0), s);
            ck(cudaEventRecord(e1, s), "r1"); ck(cudaEventSynchronize(e1), "s1");
            float ms_ref = 0.0f; ck(cudaEventElapsedTime(&ms_ref, e0, e1), "el"); ms_ref /= NITER;

            for (int i = 0; i < 5; ++i) strata::kernels::dense_i8_tc_mmvq(ggml_type, d_w, d_xq, d_got, n_in, n_out, T, s);
            ck(cudaEventRecord(e0, s), "r0");
            for (int i = 0; i < NITER; ++i) strata::kernels::dense_i8_tc_mmvq(ggml_type, d_w, d_xq, d_got, n_in, n_out, T, s);
            ck(cudaEventRecord(e1, s), "r1"); ck(cudaEventSynchronize(e1), "s1");
            float ms_tc = 0.0f; ck(cudaEventElapsedTime(&ms_tc, e0, e1), "el"); ms_tc /= NITER;

            std::printf("  DECODE %-6s %dx%d T=%2d: rel_rms=%.2e | mmvq=%.3f ms  i8_tc=%.3f ms  speedup=%.2fx\n",
                        name, n_out, n_in, T, rel_rms, ms_ref, ms_tc, ms_ref / ms_tc);
            ck(cudaEventDestroy(e0), "d0"); ck(cudaEventDestroy(e1), "d1");
        } else {
            std::printf("  DECODE %-6s %dx%d T=%2d: rel_rms=%.2e OK\n", name, n_out, n_in, T, rel_rms);
        }
    }

    ck(cudaFree(d_w), "f"); ck(cudaFree(d_x), "f"); ck(cudaFree(d_xq), "f");
    ck(cudaFree(d_ref), "f"); ck(cudaFree(d_got), "f");
}

void test_prefill_type(int ggml_type, const char* name, int K, int N, int T, bool bench, cudaStream_t s) {
    std::mt19937 rng(19000u + (unsigned) ggml_type + (unsigned) T);
    std::vector<uint8_t> h_w;
    fill_weights(ggml_type, K, N, h_w, rng);
    std::vector<uint16_t> h_x((size_t) T * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : h_x) v = __half_as_ushort(__float2half_rn(nd(rng)));

    void *d_w = nullptr, *d_scratch = nullptr;
    uint16_t* d_x = nullptr;
    float *d_ref = nullptr, *d_got = nullptr;
    const size_t sc_bytes = std::max<size_t>((size_t) N * K * 2u, strata::kernels::dense_i8_tc_prefill_scratch_bytes(T, K));
    ck(cudaMalloc(&d_w, h_w.size()), "mw");
    ck(cudaMalloc(&d_x, h_x.size() * 2u), "mx");
    ck(cudaMalloc(&d_scratch, sc_bytes), "ms");
    ck(cudaMalloc(&d_ref, (size_t) T * N * sizeof(float)), "mr");
    ck(cudaMalloc(&d_got, (size_t) T * N * sizeof(float)), "mg");
    ck(cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice), "H2D w");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2u, cudaMemcpyHostToDevice), "H2D x");

    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, (int64_t) N * K, err)) {
        std::fprintf(stderr, "Gemm::init failed: %s\n", err.c_str());
        std::exit(1);
    }
    gemm.native(d_x, ggml_type, d_w, d_ref, T, N, K, N, 0.0f);
    if (!strata::kernels::dense_i8_tc_prefill(d_x, ggml_type, d_w, d_got, T, N, K, N, 0.0f, d_scratch, sc_bytes, s)) {
        std::fprintf(stderr, "dense_i8_tc_prefill returned false for %s\n", name);
        std::exit(1);
    }
    ck(cudaStreamSynchronize(s), "sync");

    std::vector<float> h_ref((size_t) T * N), h_got((size_t) T * N);
    ck(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H r");
    ck(cudaMemcpy(h_got.data(), d_got, h_got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H g");

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        const double df = (double) h_got[i] - (double) h_ref[i];
        num += df * df;
        den += (double) h_ref[i] * (double) h_ref[i];
    }
    const double rel_rms = den > 0.0 ? std::sqrt(num / den) : 0.0;
    if (!std::isfinite(rel_rms) || rel_rms > 0.015) {
        std::fprintf(stderr, "FAIL prefill %s %dx%d T=%d: rel_rms=%.6f\n", name, N, K, T, rel_rms);
        std::exit(1);
    }

    if (bench) {
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");
        constexpr int NITER = 15;
        for (int i = 0; i < 3; ++i) gemm.native(d_x, ggml_type, d_w, d_ref, T, N, K, N, 0.0f);
        ck(cudaEventRecord(e0, s), "r0");
        for (int i = 0; i < NITER; ++i) gemm.native(d_x, ggml_type, d_w, d_ref, T, N, K, N, 0.0f);
        ck(cudaEventRecord(e1, s), "r1"); ck(cudaEventSynchronize(e1), "s1");
        float ms_ref = 0.0f; ck(cudaEventElapsedTime(&ms_ref, e0, e1), "el"); ms_ref /= NITER;

        for (int i = 0; i < 3; ++i)
            strata::kernels::dense_i8_tc_prefill(d_x, ggml_type, d_w, d_got, T, N, K, N, 0.0f, d_scratch, sc_bytes, s);
        ck(cudaEventRecord(e0, s), "r0");
        for (int i = 0; i < NITER; ++i)
            strata::kernels::dense_i8_tc_prefill(d_x, ggml_type, d_w, d_got, T, N, K, N, 0.0f, d_scratch, sc_bytes, s);
        ck(cudaEventRecord(e1, s), "r1"); ck(cudaEventSynchronize(e1), "s1");
        float ms_tc = 0.0f; ck(cudaEventElapsedTime(&ms_tc, e0, e1), "el"); ms_tc /= NITER;

        std::printf("  PREFILL %-6s %dx%d T=%4d: rel_rms=%.4f | deq+cublas=%.3f ms  i8_tc=%.3f ms  speedup=%.2fx\n",
                    name, N, K, T, rel_rms, ms_ref, ms_tc, ms_ref / ms_tc);
        ck(cudaEventDestroy(e0), "d0"); ck(cudaEventDestroy(e1), "d1");
    } else {
        std::printf("  PREFILL %-6s %dx%d T=%4d: rel_rms=%.4f OK\n", name, N, K, T, rel_rms);
    }

    ck(cudaFree(d_w), "f"); ck(cudaFree(d_x), "f"); ck(cudaFree(d_scratch), "f");
    ck(cudaFree(d_ref), "f"); ck(cudaFree(d_got), "f");
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0 || !strata::kernels::dense_i8_tc_supported()) {
        std::printf("dense_i8_tc_test: sm_80+ CUDA GPU required (skipped)\n");
        return 77;
    }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    const std::pair<int, const char*> types[] = {
        {8, "Q8_0"}, {12, "Q4_K"}, {13, "Q5_K"}, {14, "Q6_K"}, {23, "IQ4_XS"}
    };
    for (auto [ty, nm] : types) {
        test_decode_type(ty, nm, 2560, 8192, bench, s);
    }
    for (auto [ty, nm] : types) {
        test_prefill_type(ty, nm, 2560, 8192, bench ? 2048 : 131, bench, s);
    }
    ck(cudaStreamDestroy(s), "destroy");
    std::printf("dense_i8_tc_test: ok\n");
    return 0;
}
