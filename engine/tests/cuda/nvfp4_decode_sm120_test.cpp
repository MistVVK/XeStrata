// tests/cuda/nvfp4_decode_sm120_test.cpp - Parity & benchmark for Blackwell sm_120a hardware NVFP4 grouped decode.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/nvfp4_decode_sm120.hpp"

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

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void run_case(int n_groups, int entries_per_group, bool bench) {
    using namespace strata::kernels;
    constexpr int kN = 2560;
    constexpr int kFF = 640;
    constexpr int kTypeNVFP4 = 40;
    const NativeExpertLayout L = native_expert_layout(kTypeNVFP4, kTypeNVFP4, kN, kFF);
    const int total_entries = n_groups * entries_per_group;
    const int n_tok = std::max(1, entries_per_group);

    std::mt19937 rng(42 + n_groups * 7 + entries_per_group);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> sc_dist(16, 80); // valid non-NaN UE4M3 scales
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    std::vector<uint8_t> h_blobs((size_t) n_groups * L.bytes);
    for (int g = 0; g < n_groups; ++g) {
        uint8_t* bp = h_blobs.data() + (size_t) g * L.bytes;
        const size_t nblk = L.tail_off / 36;
        for (size_t b = 0; b < nblk; ++b) {
            for (int i = 0; i < 4; ++i) bp[b * 36 + i] = (uint8_t) sc_dist(rng);
            for (int i = 4; i < 36; ++i) bp[b * 36 + i] = (uint8_t) byte_dist(rng);
        }
        float* tail = reinterpret_cast<float*>(bp + L.tail_off);
        tail[0] = 0.015f;
        tail[1] = 0.018f;
        tail[2] = 0.012f;
        tail[3] = 0.0f;
    }

    std::vector<float> h_x((size_t) n_tok * kN);
    for (float& v : h_x) v = fdist(rng);

    std::vector<int32_t> h_grp_start(n_groups + 1);
    std::vector<int32_t> h_ent_dst(total_entries);
    std::vector<int32_t> h_ent_tok(total_entries);
    for (int g = 0; g < n_groups; ++g) {
        h_grp_start[g] = g * entries_per_group;
        for (int j = 0; j < entries_per_group; ++j) {
            const int idx = g * entries_per_group + j;
            h_ent_dst[idx] = idx;
            h_ent_tok[idx] = j % n_tok;
        }
    }
    h_grp_start[n_groups] = total_entries;

    uint8_t* d_blobs = nullptr;
    unsigned long long* d_grp_ptr = nullptr;
    int32_t *d_grp_start = nullptr, *d_n_groups = nullptr, *d_ent_dst = nullptr, *d_ent_tok = nullptr;
    float *d_x = nullptr, *d_out_ref = nullptr, *d_out_new = nullptr;
    void *d_xq = nullptr, *d_scratch = nullptr;

    ck(cudaMalloc(&d_blobs, h_blobs.size()), "malloc blobs");
    ck(cudaMemcpy(d_blobs, h_blobs.data(), h_blobs.size(), cudaMemcpyHostToDevice), "h2d blobs");

    std::vector<unsigned long long> h_grp_ptr(n_groups);
    for (int g = 0; g < n_groups; ++g) h_grp_ptr[g] = (unsigned long long) (d_blobs + (size_t) g * L.bytes);

    ck(cudaMalloc(&d_grp_ptr, (size_t) n_groups * sizeof(unsigned long long)), "malloc ptr");
    ck(cudaMalloc(&d_grp_start, (size_t) (n_groups + 1) * sizeof(int32_t)), "malloc start");
    ck(cudaMalloc(&d_n_groups, sizeof(int32_t)), "malloc ng");
    ck(cudaMalloc(&d_ent_dst, (size_t) total_entries * sizeof(int32_t)), "malloc dst");
    ck(cudaMalloc(&d_ent_tok, (size_t) total_entries * sizeof(int32_t)), "malloc tok");
    ck(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_xq, (size_t) n_tok * (kN / 32) * 36), "malloc xq");
    const size_t fa = ((size_t) total_entries * kFF * sizeof(float) + 255) & ~(size_t) 255;
    const size_t scr_bytes = 3 * fa + (size_t) total_entries * (kFF / 32) * 36 + 4096;
    ck(cudaMalloc(&d_scratch, scr_bytes), "malloc scratch");
    ck(cudaMalloc(&d_out_ref, (size_t) total_entries * kN * sizeof(float)), "malloc ref");
    ck(cudaMalloc(&d_out_new, (size_t) total_entries * kN * sizeof(float)), "malloc new");

    ck(cudaMemcpy(d_grp_ptr, h_grp_ptr.data(), (size_t) n_groups * sizeof(unsigned long long), cudaMemcpyHostToDevice), "h2d ptr");
    ck(cudaMemcpy(d_grp_start, h_grp_start.data(), (size_t) (n_groups + 1) * sizeof(int32_t), cudaMemcpyHostToDevice), "h2d start");
    ck(cudaMemcpy(d_n_groups, &n_groups, sizeof(int32_t), cudaMemcpyHostToDevice), "h2d ng");
    ck(cudaMemcpy(d_ent_dst, h_ent_dst.data(), (size_t) total_entries * sizeof(int32_t), cudaMemcpyHostToDevice), "h2d dst");
    ck(cudaMemcpy(d_ent_tok, h_ent_tok.data(), (size_t) total_entries * sizeof(int32_t), cudaMemcpyHostToDevice), "h2d tok");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d x");

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    for (int t0 = 0; t0 < n_tok; t0 += 4) {
        const int nt = std::min(4, n_tok - t0);
        native_quantize_q8_1(d_x + (size_t) t0 * kN, (uint8_t*) d_xq + (size_t) t0 * (kN / 32) * 36, kN, nt, st);
    }

    // Reference: existing scalar/dp4a grouped path (override sm120 hardware decode off).
    void* hq_ptr = (uint8_t*) d_scratch + 3 * fa;
    nvfp4_decode_sm120_set_override(1);
    if (!nvfp4_decode_sm120_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                    n_groups, total_entries, d_xq, hq_ptr, d_out_new, st)) {
        std::fprintf(stderr, "nvfp4_decode_sm120_grouped returned false\n");
        std::exit(1);
    }
    nvfp4_decode_sm120_set_override(0);
    native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                          n_groups, total_entries, d_xq, d_scratch, d_out_ref, st);
    nvfp4_decode_sm120_set_override(1);
    ck(cudaStreamSynchronize(st), "sync");

    std::vector<float> h_ref((size_t) total_entries * kN), h_new((size_t) total_entries * kN);
    ck(cudaMemcpy(h_ref.data(), d_out_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h ref");
    ck(cudaMemcpy(h_new.data(), d_out_new, h_new.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h new");

    double num = 0.0, den = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        const double d = (double) h_new[i] - (double) h_ref[i];
        num += d * d;
        den += (double) h_ref[i] * (double) h_ref[i];
        max_abs = std::max(max_abs, std::abs(d));
    }
    const double rel_rms = std::sqrt(num / std::max(den, 1e-30));
    std::printf("  NVFP4 G=%2d E/G=%2d: rel_rms=%.6f max_abs=%.6f %s\n",
                n_groups, entries_per_group, rel_rms, max_abs, rel_rms < 5e-3 ? "OK" : "FAIL");
    if (rel_rms >= 5e-3) std::exit(1);

    if (bench) {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "ev a");
        ck(cudaEventCreate(&b), "ev b");
        const int iters = 40;
        nvfp4_decode_sm120_set_override(0);
        for (int i = 0; i < 5; ++i)
            native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                  n_groups, total_entries, d_xq, d_scratch, d_out_ref, st);
        ck(cudaEventRecord(a, st), "rec a");
        for (int i = 0; i < iters; ++i)
            native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                  n_groups, total_entries, d_xq, d_scratch, d_out_ref, st);
        ck(cudaEventRecord(b, st), "rec b");
        ck(cudaEventSynchronize(b), "sync b");
        float ms_ref = 0.0f;
        ck(cudaEventElapsedTime(&ms_ref, a, b), "elap ref");
        ms_ref /= iters;

        nvfp4_decode_sm120_set_override(1);

        for (int i = 0; i < 5; ++i)
            nvfp4_decode_sm120_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                       n_groups, total_entries, d_xq, hq_ptr, d_out_new, st);
        ck(cudaEventRecord(a, st), "rec a2");
        for (int i = 0; i < iters; ++i)
            nvfp4_decode_sm120_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                       n_groups, total_entries, d_xq, hq_ptr, d_out_new, st);
        ck(cudaEventRecord(b, st), "rec b2");
        ck(cudaEventSynchronize(b), "sync b2");
        float ms_new = 0.0f;
        ck(cudaEventElapsedTime(&ms_new, a, b), "elap new");
        ms_new /= iters;
        std::printf("  BENCH NVFP4 G=%2d E/G=%2d: ref=%.3f ms  sm120_hw=%.3f ms  speedup=%.2fx\n",
                    n_groups, entries_per_group, ms_ref, ms_new, ms_ref / ms_new);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }

    cudaStreamDestroy(st);
    cudaFree(d_blobs); cudaFree(d_grp_ptr); cudaFree(d_grp_start); cudaFree(d_n_groups);
    cudaFree(d_ent_dst); cudaFree(d_ent_tok); cudaFree(d_x); cudaFree(d_xq);
    cudaFree(d_scratch); cudaFree(d_out_ref); cudaFree(d_out_new);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
    strata::kernels::nvfp4_decode_sm120_set_override(1);
    if (!strata::kernels::nvfp4_decode_sm120_available()) {
        std::printf("nvfp4_decode_sm120_test: skipping (requires Blackwell sm_120a/sm_121a)\n");
        return 77;
    }
    // Note: pass STRATA_NVFP4_DEC_SM120=0 only around ref? No, nvfp4_decode_sm120_available caches env once.
    // Force native_expert_grouped to run the reference path via native_expert_set_mode(0) + g_grouped_v1?
    // Actually native_expert_grouped dispatches to nvfp4_decode_sm120_grouped unless STRATA_NVFP4_DEC_SM120=0,
    // so run this test by setting native_expert_set_grouped_v1(true) before calling native_expert_grouped?
    // Wait: let's check if native_expert_set_grouped_v1 exists or expose a direct env/flag!
    run_case(10, 1, bench);
    run_case(24, 2, bench);
    run_case(32, 4, bench);
    run_case(32, 8, bench);
    std::printf("nvfp4_decode_sm120_test: ok\n");
    return 0;
}
