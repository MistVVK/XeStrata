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
#include <cuda_fp16.h>

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
        tail[2] = (1 + g % 4) * 0.00005f;
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

    // Compare against the mode 0 DP4A result using identical blobs and activations.
    void* hq_ptr = (uint8_t*) d_scratch + 3 * fa;
    ck(cudaMemsetAsync(d_out_new, 0xff, (size_t) total_entries * kN * sizeof(float), st), "poison tensor core output");
    nvfp4_decode_sm120_set_override(1);
    if (!nvfp4_decode_sm120_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                    n_groups, total_entries, d_xq, hq_ptr, d_out_new, st)) {
        std::fprintf(stderr, "nvfp4_decode_sm120_grouped returned false\n");
        std::exit(1);
    }
    nvfp4_decode_sm120_set_override(0);
    native_expert_set_mode(0, 0);
    ck(cudaMemsetAsync(d_out_ref, 0xff, (size_t) total_entries * kN * sizeof(float), st), "poison reference output");
    native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                          n_groups, total_entries, d_xq, d_scratch, d_out_ref, st);
    nvfp4_decode_sm120_set_override(1);
    ck(cudaStreamSynchronize(st), "sync");

    std::vector<float> h_ref((size_t) total_entries * kN), h_new((size_t) total_entries * kN);
    ck(cudaMemcpy(h_ref.data(), d_out_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h ref");
    ck(cudaMemcpy(h_new.data(), d_out_new, h_new.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h new");

    auto check = [&](const char* path, double rel_tol) {
        double num = 0.0, den = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < h_ref.size(); ++i) {
            const double d = (double) h_new[i] - (double) h_ref[i];
            num += d * d;
            den += (double) h_ref[i] * (double) h_ref[i];
            max_abs = std::max(max_abs, std::abs(d));
        }
        const double rel_rms = den > 0.0 ? std::sqrt(num / den) : 0.0;
        const bool ok = den > 0.0 && std::isfinite(rel_rms) && rel_rms < rel_tol;
        std::printf("  NVFP4 %s G=%2d E/G=%2d: rel_rms=%.6f max_abs=%.6f %s\n",
                    path, n_groups, entries_per_group, rel_rms, max_abs, ok ? "OK" : "FAIL");
        if (!ok) std::exit(1);
    };
    check("sm120 tensor core", 5e-3);

    // Both grouped layouts route the 640-column down projection to the 8-lane kernel.
    nvfp4_decode_sm120_set_override(0);
    for (int mode : {8, 7}) {
        native_expert_set_mode(mode, 0);
        const int poison = mode == 7 ? 0x7f : 0xff;
        ck(cudaMemsetAsync(d_out_new, poison, h_new.size() * sizeof(float), st), "poison grouped output");
        native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                              n_groups, total_entries, d_xq, d_scratch, d_out_new, st);
        ck(cudaStreamSynchronize(st), "grouped sync");
        ck(cudaMemcpy(h_new.data(), d_out_new, h_new.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h grouped");
        check(mode == 7 ? "grouped mode 7" : "grouped mode 8", 1e-5);
    }
    native_expert_set_mode(-1, 0);

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

void fill_valid_quant_bytes(int ty, uint8_t* dst, size_t n_rows, size_t cols, std::mt19937& rng) {
    auto put_h = [](uint8_t* p, float v) {
        const uint16_t h = __half_as_ushort(__float2half_rn(v));
        p[0] = (uint8_t) (h & 0xFFu); p[1] = (uint8_t) (h >> 8);
    };
    const size_t row_b = strata::kernels::iq_row_bytes(ty, (int64_t) cols);
    for (size_t r = 0; r < n_rows; ++r) {
        uint8_t* row = dst + r * row_b;
        if (ty == 40) {
            const size_t nb = cols / 64;
            for (size_t b = 0; b < nb; ++b) {
                for (int i = 0; i < 4; ++i) row[b * 36 + i] = (uint8_t) (24 + (rng() % 48));
                for (int i = 4; i < 36; ++i) row[b * 36 + i] = (uint8_t) (rng() & 0xFFu);
            }
        } else if (ty == 8) {
            const size_t nb = cols / 32;
            for (size_t b = 0; b < nb; ++b) {
                put_h(row + b * 34, 0.005f + 0.005f * ((float) (rng() & 255) / 255.0f));
                for (int i = 2; i < 34; ++i) row[b * 34 + i] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
            }
        } else if (ty == 20) {
            const size_t nb = cols / 32;
            for (size_t b = 0; b < nb; ++b) {
                put_h(row + b * 18, 0.004f + 0.004f * ((float) (rng() & 255) / 255.0f));
                for (int i = 2; i < 18; ++i) row[b * 18 + i] = (uint8_t) (rng() & 0xFFu);
            }
        } else if (ty == 42) {
            const size_t nb = cols / 64;
            for (size_t b = 0; b < nb; ++b) {
                put_h(row + b * 18, 0.01f + 0.01f * ((float) (rng() & 255) / 255.0f));
                for (int i = 2; i < 18; ++i) row[b * 18 + i] = (uint8_t) (rng() & 0xFFu);
            }
        } else if (ty == 10) {
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * 84;
                for (int i = 0; i < 80; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
                put_h(bp + 80, 0.005f + 0.005f * ((float) (rng() & 255) / 255.0f));
                put_h(bp + 82, 0.004f + 0.004f * ((float) (rng() & 255) / 255.0f));
            }
        } else if (ty == 11) {
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * 110;
                for (int i = 0; i < 108; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
                put_h(bp + 108, 0.004f + 0.004f * ((float) (rng() & 255) / 255.0f));
            }
        } else if (ty == 12 || ty == 13) {
            const size_t bs = ty == 12 ? 144u : 176u;
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * bs;
                put_h(bp + 0, 0.004f + 0.004f * ((float) (rng() & 255) / 255.0f));
                put_h(bp + 2, 0.003f + 0.003f * ((float) (rng() & 255) / 255.0f));
                for (size_t i = 4; i < bs; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
            }
        } else if (ty == 14) {
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * 210;
                for (int i = 0; i < 192; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
                for (int i = 192; i < 208; ++i) bp[i] = (uint8_t) (int8_t) ((int) (rng() % 63) - 31);
                put_h(bp + 208, 0.003f + 0.004f * ((float) (rng() & 255) / 255.0f));
            }
        } else if (ty == 18 || ty == 21 || ty == 23) {
            const size_t bs = ty == 18 ? 98u : ty == 21 ? 110u : 136u;
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * bs;
                put_h(bp + 0, 0.003f + 0.003f * ((float) (rng() & 255) / 255.0f));
                for (size_t i = 2; i < bs; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
            }
        } else if (ty == 29) {
            const size_t nb = cols / 256;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* bp = row + b * 56;
                for (int i = 0; i < 48; ++i) bp[i] = (uint8_t) (rng() & 0xFFu);
                const uint16_t dh = __half_as_ushort(__float2half_rn(0.01f));
                for (int k = 0; k < 4; ++k) {
                    const uint16_t sc = (uint16_t) ((rng() & 0x0FFFu) | (((dh >> (4 * k)) & 0xFu) << 12));
                    bp[48 + 2 * k] = (uint8_t) (sc & 0xFFu);
                    bp[48 + 2 * k + 1] = (uint8_t) (sc >> 8);
                }
            }
        }
    }
}

void run_grouped_pair_case(const char* label, int gu_type, int d_type, int n_embd, int n_ff,
                           int n_groups, int epg, size_t blob_shift, int had_block,
                           float sg, float su, float sd, bool bench) {
    using namespace strata::kernels;
    if (!native_expert_supported(gu_type, d_type, n_embd, n_ff)) {
        std::printf("  %-24s SKIP (not compiled in)\n", label);
        return;
    }
    NativeExpertLayout L = native_expert_layout(gu_type, d_type, n_embd, n_ff);
    L.had_block = had_block;
    const int total_entries = n_groups * epg;
    const int n_tok = std::max(1, epg);
    const size_t stride = ((L.bytes + blob_shift + 63) / 64) * 64;

    std::mt19937 rng(7000u + (unsigned) gu_type * 31u + (unsigned) d_type * 13u + (unsigned) had_block);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    std::vector<uint8_t> h_blobs((size_t) n_groups * stride, 0);
    for (int g = 0; g < n_groups; ++g) {
        uint8_t* bp = h_blobs.data() + (size_t) g * stride + blob_shift;
        fill_valid_quant_bytes(gu_type, bp, (size_t) n_ff, (size_t) n_embd, rng);
        fill_valid_quant_bytes(gu_type, bp + L.up_off, (size_t) n_ff, (size_t) n_embd, rng);
        fill_valid_quant_bytes(d_type, bp + L.down_off, (size_t) n_embd, (size_t) n_ff, rng);
        if (L.tail_off) {
            const float tail[4] = {sg, su, sd, 0.0f};
            std::memcpy(bp + L.tail_off, tail, 16);
        }
    }

    std::vector<float> h_x((size_t) n_tok * n_embd);
    for (float& v : h_x) v = fdist(rng);

    std::vector<int32_t> h_grp_start(n_groups + 1), h_ent_dst(total_entries), h_ent_tok(total_entries);
    for (int g = 0; g < n_groups; ++g) {
        h_grp_start[g] = g * epg;
        for (int j = 0; j < epg; ++j) {
            const int idx = g * epg + j;
            h_ent_dst[idx] = idx;
            h_ent_tok[idx] = j % n_tok;
        }
    }
    h_grp_start[n_groups] = total_entries;

    uint8_t* d_blobs = nullptr;
    unsigned long long* d_grp_ptr = nullptr;
    int32_t *d_grp_start = nullptr, *d_n_groups = nullptr, *d_ent_dst = nullptr, *d_ent_tok = nullptr;
    float *d_x = nullptr, *d_out0 = nullptr, *d_out8 = nullptr;
    void *d_xq = nullptr, *d_scratch = nullptr;
    ck(cudaMalloc(&d_blobs, h_blobs.size()), "malloc blobs");
    ck(cudaMemcpy(d_blobs, h_blobs.data(), h_blobs.size(), cudaMemcpyHostToDevice), "h2d blobs");
    std::vector<unsigned long long> h_grp_ptr(n_groups);
    for (int g = 0; g < n_groups; ++g) h_grp_ptr[g] = (unsigned long long) (d_blobs + (size_t) g * stride + blob_shift);

    ck(cudaMalloc(&d_grp_ptr, (size_t) n_groups * sizeof(unsigned long long)), "malloc ptr");
    ck(cudaMalloc(&d_grp_start, (size_t) (n_groups + 1) * sizeof(int32_t)), "malloc start");
    ck(cudaMalloc(&d_n_groups, sizeof(int32_t)), "malloc ng");
    ck(cudaMalloc(&d_ent_dst, (size_t) total_entries * sizeof(int32_t)), "malloc dst");
    ck(cudaMalloc(&d_ent_tok, (size_t) total_entries * sizeof(int32_t)), "malloc tok");
    ck(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_xq, (size_t) n_tok * (n_embd / 32) * 36), "malloc xq");
    ck(cudaMalloc(&d_scratch, native_expert_scratch_bytes(total_entries, n_ff) + 4096), "malloc scratch");
    ck(cudaMalloc(&d_out0, (size_t) total_entries * n_embd * sizeof(float)), "malloc out0");
    ck(cudaMalloc(&d_out8, (size_t) total_entries * n_embd * sizeof(float)), "malloc out8");

    ck(cudaMemcpy(d_grp_ptr, h_grp_ptr.data(), (size_t) n_groups * 8, cudaMemcpyHostToDevice), "h2d ptr");
    ck(cudaMemcpy(d_grp_start, h_grp_start.data(), (size_t) (n_groups + 1) * 4, cudaMemcpyHostToDevice), "h2d start");
    ck(cudaMemcpy(d_n_groups, &n_groups, 4, cudaMemcpyHostToDevice), "h2d ng");
    ck(cudaMemcpy(d_ent_dst, h_ent_dst.data(), (size_t) total_entries * 4, cudaMemcpyHostToDevice), "h2d dst");
    ck(cudaMemcpy(d_ent_tok, h_ent_tok.data(), (size_t) total_entries * 4, cudaMemcpyHostToDevice), "h2d tok");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice), "h2d x");

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    quantize_q8_1_rows(d_x, n_tok, n_embd, d_xq, st);

    nvfp4_decode_sm120_set_override(0);
    native_expert_set_mode(0, 0);
    ck(cudaMemsetAsync(d_out0, 0xff, (size_t) total_entries * n_embd * 4, st), "poison 0");
    native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok, n_groups, total_entries, d_xq, d_scratch, d_out0, st);

    native_expert_set_mode(8, 0);
    ck(cudaMemsetAsync(d_out8, 0x7f, (size_t) total_entries * n_embd * 4, st), "poison 8");
    native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok, n_groups, total_entries, d_xq, d_scratch, d_out8, st);
    ck(cudaStreamSynchronize(st), "sync");

    std::vector<float> h0((size_t) total_entries * n_embd), h8((size_t) total_entries * n_embd);
    ck(cudaMemcpy(h0.data(), d_out0, h0.size() * 4, cudaMemcpyDeviceToHost), "d2h 0");
    ck(cudaMemcpy(h8.data(), d_out8, h8.size() * 4, cudaMemcpyDeviceToHost), "d2h 8");

    double num = 0.0, den = 0.0;
    size_t nonfinite = 0, bit_diff = 0;
    for (size_t i = 0; i < h0.size(); ++i) {
        if (!std::isfinite(h0[i]) || !std::isfinite(h8[i])) ++nonfinite;
        if (std::memcmp(&h0[i], &h8[i], 4) != 0) ++bit_diff;
        const double d = (double) h8[i] - (double) h0[i];
        num += d * d;
        den += (double) h0[i] * (double) h0[i];
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    const bool ok = nonfinite == 0 && den > 0.0 && std::isfinite(rel) && rel < 1e-5;
    if (!bench) {
        std::printf("  %-24s G=%2d E/G=%d shift=%zu had=%d: rel=%.2e bitdiff=%zu nonfin=%zu %s\n",
                    label, n_groups, epg, blob_shift, had_block, rel, bit_diff, nonfinite, ok ? "OK" : "FAIL");
    } else {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "a"); ck(cudaEventCreate(&b), "b");
        auto time_mp = [&](int mode, int phase) -> float {
            native_expert_set_mode(mode, phase);
            for (int i = 0; i < 5; ++i)
                native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok, n_groups, total_entries, d_xq, d_scratch, d_out8, st);
            ck(cudaEventRecord(a, st), "ra");
            constexpr int IT = 40;
            for (int i = 0; i < IT; ++i)
                native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok, n_groups, total_entries, d_xq, d_scratch, d_out8, st);
            ck(cudaEventRecord(b, st), "rb");
            ck(cudaEventSynchronize(b), "sb");
            float ms = 0.0f; ck(cudaEventElapsedTime(&ms, a, b), "el");
            return ms / IT * 1000.0f;
        };
        const float full0 = time_mp(0, 0), full8 = time_mp(8, 0);
        const float down0 = time_mp(0, 2), down8 = time_mp(8, 2);
        std::printf("  %-24s rel=%.1e | full m0=%6.1fus m8=%6.1fus (%.2fx) | down m0=%5.1fus m8=%5.1fus (%.2fx) %s\n",
                    label, rel, full0, full8, full0 / full8, down0, down8, down0 / down8, ok ? "OK" : "FAIL");
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    native_expert_set_mode(-1, 0);
    cudaStreamDestroy(st);
    cudaFree(d_blobs); cudaFree(d_grp_ptr); cudaFree(d_grp_start); cudaFree(d_n_groups);
    cudaFree(d_ent_dst); cudaFree(d_ent_tok); cudaFree(d_x); cudaFree(d_xq);
    cudaFree(d_scratch); cudaFree(d_out0); cudaFree(d_out8);
    if (!ok) std::exit(1);
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
    run_case(10, 1, bench);
    run_case(24, 2, bench);
    run_case(24, 3, bench);
    run_case(32, 4, bench);
    run_case(32, 8, bench);

    // NVFP4 Mission 1 edge cases: s=0.003f, mixed gu/d with unaligned tail (2B shift), Hadamard-folded NVFP4
    run_grouped_pair_case("NVFP4 s=0.003", 40, 40, 2560, 640, 16, 2, 0, 0, 0.003f, 0.003f, 0.003f, false);
    run_grouped_pair_case("NVFP4 had=64", 40, 40, 2560, 640, 16, 2, 0, 64, 0.003f, 0.003f, 0.003f, false);
    run_grouped_pair_case("MIX Q8_0/NVFP4 sh=2", 8, 40, 2560, 640, 16, 2, 2, 0, 99.0f, 99.0f, 0.003f, false);
    run_grouped_pair_case("MIX NVFP4/Q8_0 sh=2", 40, 8, 2560, 640, 16, 2, 2, 0, 0.003f, 0.003f, 99.0f, false);
    run_grouped_pair_case("MIX NVFP4/IQ4_NL sh=2", 40, 20, 2560, 640, 16, 2, 2, 0, 0.003f, 0.003f, 99.0f, false);

    // Unsloth Mission 2 parity & A/B timing (mode 0 vs mode 8, full & down-only)
    run_grouped_pair_case("Q8_0/Q8_0 ff640", 8, 8, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q4_K/Q4_K ff512", 12, 12, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q5_K/Q5_K ff512", 13, 13, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q6_K/Q6_K ff512", 14, 14, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("IQ4_XS/IQ4_XS ff512", 23, 23, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("IQ3_XXS/IQ3_XXS ff512", 18, 18, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("IQ3_S/IQ3_S ff512", 21, 21, 2560, 512, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("IQ4_NL/IQ4_NL ff640", 20, 20, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q3_K/Q8_0 ff640", 11, 8, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q2_K/Q8_0 ff640", 10, 8, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("IQ1_M/IQ4_NL ff640", 29, 20, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("Q2_0/Q2_0 ff640", 42, 42, 2560, 640, 24, 2, 0, 0, 1.f, 1.f, 1.f, bench);
    run_grouped_pair_case("NVFP4/NVFP4 ff640", 40, 40, 2560, 640, 24, 2, 0, 0, 0.003f, 0.003f, 0.003f, bench);

    std::printf("nvfp4_decode_sm120_test: ok\n");
    return 0;
}
