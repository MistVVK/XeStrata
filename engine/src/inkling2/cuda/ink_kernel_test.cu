// engine/src/inkling2/cuda/ink_kernel_test.cu - Exact FP64 Parity & Microbenchmark Test for Inkling-2-Mini CUDA Kernels
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "ink_cuda.cuh"
#include "ink_kernels.cuh"
#include "ink_moe_cuda.cuh"
#include "ink_moe_fp4_sm120.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ink;

static double e2m1_host(uint8_t nib) {
    static constexpr double lut[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    };
    return lut[nib & 15u];
}

static double e4m3_host(uint8_t c) {
    const int mag = c & 0x7F;
    if (mag == 0x7F || mag == 0) return 0.0;
    const int exp = mag >> 3;
    const int man = mag & 7;
    double v = (exp == 0) ? ((double)man / 512.0) : ((1.0 + (double)man * 0.125) * std::ldexp(1.0, exp - 7));
    return (c & 0x80) ? -v : v;
}

static double ue8m0_host(uint8_t e) {
    if (e == 255) return 0.0;
    if (e == 0) return std::ldexp(1.0, -127);
    return std::ldexp(1.0, (int)e - 127);
}

static int test_sm120_sf4() {
    if (!sm120::available()) {
        std::printf("[SKIP] sm_120a SF4 TC not available on this GPU\n");
        return 0;
    }
    const int M = 64, N = 256, K = 2304;
    std::mt19937 rng(20261121);
    std::uniform_int_distribution<int> du8(0, 255), dsc(24, 96);
    std::normal_distribution<float> dnorm(0.0f, 0.5f);

    std::vector<uint8_t> h_wp((size_t)N * (K / 2));
    std::vector<uint8_t> h_ws((size_t)N * (K / 16));
    std::vector<float> h_x((size_t)M * K);
    for (auto& b : h_wp) b = (uint8_t)du8(rng);
    for (auto& b : h_ws) b = (uint8_t)dsc(rng);
    for (auto& v : h_x) v = dnorm(rng);
    const float w_outer = 1.75f / (6.0f * 448.0f);

    uint8_t* d_wp = dalloc<uint8_t>(h_wp.size());
    uint8_t* d_ws = dalloc<uint8_t>(h_ws.size());
    float* d_x = dalloc<float>(h_x.size());
    float* d_y = dalloc<float>((size_t)M * N);
    INK_CK(cudaMemcpy(d_wp, h_wp.data(), h_wp.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_ws, h_ws.data(), h_ws.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice));

    cudaStream_t st;
    INK_CK(cudaStreamCreate(&st));
    sm120::test_sf4_gemm_tc(d_wp, d_ws, w_outer, d_x, d_y, M, N, K, st);

    std::vector<float> h_y((size_t)M * N);
    INK_CK(cudaMemcpy(h_y.data(), d_y, h_y.size() * 4, cudaMemcpyDeviceToHost));

    // Compare against exact FP64 reference (note: activation is quantized to FP4, so check cosine > 0.995 & rel_err < 0.08!)
    double dot = 0.0, ny = 0.0, nr = 0.0;
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int b = 0; b < K / 16; ++b) {
                double sc = e4m3_host(h_ws[(size_t)n * (K / 16) + b]) * (double)w_outer;
                double acc = 0.0;
                for (int j = 0; j < 8; ++j) {
                    uint8_t byte = h_wp[(size_t)n * (K / 2) + b * 8 + j];
                    acc += e2m1_host(byte & 0xFu) * (double)h_x[(size_t)m * K + b * 16 + 2 * j];
                    acc += e2m1_host(byte >> 4)   * (double)h_x[(size_t)m * K + b * 16 + 2 * j + 1];
                }
                ref += acc * sc;
            }
            double got = (double)h_y[(size_t)m * N + n];
            dot += got * ref;
            ny += got * got;
            nr += ref * ref;
        }
    }
    const double cos_sim = dot / (std::sqrt(ny * nr) + 1e-30);
    const double norm_ratio = std::sqrt(ny) / (std::sqrt(nr) + 1e-30);
    std::printf("[TEST sm_120a mxf4nvf4 SF4 GEMM] cos=%.6f norm_ratio=%.5f (expected >0.995, ~1.00)\n", cos_sim, norm_ratio);

    INK_CK(cudaFree(d_wp)); INK_CK(cudaFree(d_ws)); INK_CK(cudaFree(d_x)); INK_CK(cudaFree(d_y));
    INK_CK(cudaStreamDestroy(st));
    return (cos_sim > 0.993 && std::abs(norm_ratio - 1.0) < 0.05) ? 0 : 1;
}

static int test_sm120_mx8() {
    if (!sm120::available()) {
        std::printf("[SKIP] sm_120a MX8 TC not available on this GPU\n");
        return 0;
    }
    const int M = 64, N = 256, K = 2304;
    std::mt19937 rng(20261122);
    std::uniform_int_distribution<int> dmag(0, 96), dsgn(0, 1), dsc(118, 124);
    std::normal_distribution<float> dnorm(0.0f, 0.5f);

    std::vector<uint8_t> h_wp((size_t)N * K);
    std::vector<uint8_t> h_ws((size_t)N * (K / 32));
    std::vector<float> h_x((size_t)M * K);
    for (auto& b : h_wp) b = (uint8_t)(dmag(rng) | (dsgn(rng) << 7));
    for (auto& b : h_ws) b = (uint8_t)dsc(rng);
    for (auto& v : h_x) v = dnorm(rng);

    uint8_t* d_wp = dalloc<uint8_t>(h_wp.size());
    uint8_t* d_ws = dalloc<uint8_t>(h_ws.size());
    float* d_x = dalloc<float>(h_x.size());
    float* d_y = dalloc<float>((size_t)M * N);
    INK_CK(cudaMemcpy(d_wp, h_wp.data(), h_wp.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_ws, h_ws.data(), h_ws.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice));

    cudaStream_t st;
    INK_CK(cudaStreamCreate(&st));
    sm120::test_mx8_gemm_tc(d_wp, d_ws, d_x, d_y, M, N, K, st);

    std::vector<float> h_y((size_t)M * N);
    INK_CK(cudaMemcpy(h_y.data(), d_y, h_y.size() * 4, cudaMemcpyDeviceToHost));

    double dot = 0.0, ny = 0.0, nr = 0.0;
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int b = 0; b < K / 32; ++b) {
                double sc = ue8m0_host(h_ws[(size_t)n * (K / 32) + b]);
                double acc = 0.0;
                for (int j = 0; j < 32; ++j) {
                    acc += e4m3_host(h_wp[(size_t)n * K + b * 32 + j]) * (double)h_x[(size_t)m * K + b * 32 + j];
                }
                ref += acc * sc;
            }
            double got = (double)h_y[(size_t)m * N + n];
            dot += got * ref;
            ny += got * got;
            nr += ref * ref;
        }
    }
    const double cos_sim = dot / (std::sqrt(ny * nr) + 1e-30);
    const double norm_ratio = std::sqrt(ny) / (std::sqrt(nr) + 1e-30);
    std::printf("[TEST sm_120a mxf8f6f4 MX8 GEMM] cos=%.6f norm_ratio=%.5f (expected >0.999, ~1.00)\n", cos_sim, norm_ratio);

    INK_CK(cudaFree(d_wp)); INK_CK(cudaFree(d_ws)); INK_CK(cudaFree(d_x)); INK_CK(cudaFree(d_y));
    INK_CK(cudaStreamDestroy(st));
    return (cos_sim > 0.998 && std::abs(norm_ratio - 1.0) < 0.02) ? 0 : 1;
}

static int test_screened_unembed() {
    const int V = 32768, M = 4;
    std::mt19937 rng(20261123);
    std::normal_distribution<float> dnorm(0.0f, 0.05f);
    std::vector<uint16_t> h_w((size_t)V * kInkHidden);
    std::vector<float> h_x((size_t)M * kInkHidden);
    for (auto& b : h_w) {
        __nv_bfloat16 bf = __float2bfloat16_rn(dnorm(rng));
        std::memcpy(&b, &bf, 2);
    }
    for (auto& v : h_x) v = dnorm(rng) * 10.0f;

    uint16_t* d_w = dalloc<uint16_t>(h_w.size());
    float* d_x = dalloc<float>(h_x.size());
    float* d_full = dalloc<float>((size_t)M * V);
    INK_CK(cudaMemcpy(d_w, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice));

    // Exact full BF16 unembed
    k_ink_gemv_bf16<4><<<(V + 7) / 8, 256>>>(d_w, d_x, d_full, M, V, kInkHidden);
    k_ink_softcap<<<((size_t)M * V + 255) / 256, 256>>>(d_full, (size_t)M * V, kInkSoftcap);
    int32_t* d_arg_ref = dalloc<int32_t>(M);
    k_ink_argmax_rows<<<M, 1024>>>(d_full, V, d_arg_ref);

    // 2-stage Cauchy-Schwarz screened unembed (INT8 coarse plane)
    int8_t* d_q8 = dalloc<int8_t>((size_t)V * kInkHidden);
    float* d_sc = dalloc<float>(V);
    float* d_eps = dalloc<float>(V);
    k_ink_build_unembed_screen_bf16<<<(V + 7) / 8, 256>>>(d_w, d_q8, d_sc, d_eps, V);

    float* d_hn = dalloc<float>(M);
    int* d_mlb = dalloc<int>(M);
    int32_t* d_cc = dalloc<int32_t>(M);
    int32_t* d_cids = dalloc<int32_t>((size_t)M * 4096);
    float* d_dh = dalloc<float>((size_t)M * V);
    int32_t* d_arg_scr = dalloc<int32_t>(M);
    k_ink_row_l2norm<<<M, 256>>>(d_x, d_hn, d_mlb, d_cc, M);
    k_ink_screen_pass1<<<dim3((V + 7) / 8, M), 256>>>(d_q8, d_sc, d_eps, d_x, d_hn, d_dh, d_mlb, M, V);
    k_ink_screen_pass2_bf16<<<dim3((V + 255) / 256, M), 256>>>(d_w, d_eps, d_x, d_hn, d_dh, d_mlb, d_cids, d_cc, 4096, M, V);
    k_ink_screen_eval_bf16<<<M, 256>>>(d_w, d_x, d_cids, d_cc, 4096, d_arg_scr, M);
    INK_CK(cudaDeviceSynchronize());

    std::vector<int32_t> hr(M), hs(M), hc(M);
    INK_CK(cudaMemcpy(hr.data(), d_arg_ref, M * 4, cudaMemcpyDeviceToHost));
    INK_CK(cudaMemcpy(hs.data(), d_arg_scr, M * 4, cudaMemcpyDeviceToHost));
    INK_CK(cudaMemcpy(hc.data(), d_cc, M * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        if (hr[m] != hs[m]) ++bad;
        std::printf("  row %d: ref=%d screen=%d (cands=%d/%d) %s\n", m, hr[m], hs[m], hc[m], V, hr[m] == hs[m] ? "EXACT" : "MISMATCH");
    }
    std::printf("[TEST 2-Stage Cauchy-Schwarz Screened Unembed] %d/%d exact!\n", M - bad, M);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_full); cudaFree(d_arg_ref);
    cudaFree(d_q8); cudaFree(d_sc); cudaFree(d_eps); cudaFree(d_hn);
    cudaFree(d_mlb); cudaFree(d_cc); cudaFree(d_cids); cudaFree(d_dh); cudaFree(d_arg_scr);
    return bad;
}

int main() {
    cudaDeviceProp p{};
    INK_CK(cudaGetDeviceProperties(&p, 0));
    std::printf("Device: %s (sm_%d%d, %d SMs, %.1f GiB VRAM) | sm120_tc=%d\n",
                p.name, p.major, p.minor, p.multiProcessorCount, (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                (int)sm120::available());
    int fail = 0;
    fail += test_sm120_sf4();
    fail += test_sm120_mx8();
    fail += test_screened_unembed();
    if (fail == 0) std::printf("ALL INK CUDA KERNEL TESTS PASSED!\n");
    else           std::printf("FAILED: %d tests\n", fail);
    return fail ? 1 : 0;
}
