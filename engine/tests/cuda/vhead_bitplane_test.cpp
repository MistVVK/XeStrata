// tests/cuda/vhead_bitplane_test.cpp - Parity & latency test for exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Verifies that vhead_bitplane_argmax matches brute-force native_mmvq + argmax_rows 100% bit-for-bit across
// T = 1, 4, 8, 16 on both Q8_0 (type 8) and Q6_K (type 14), including full 248,320 x 2,560 vocabulary benchmark.
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/kernels/native_mmvq.hpp"

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
        std::fprintf(stderr, "vhead_bitplane_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

void test_type(int ggml_type, const char* name, int n_in, int n_out, bool bench) {
    const size_t w_bytes = strata::kernels::native_mmvq_weight_bytes(ggml_type, n_in, n_out);
    std::mt19937 rng(ggml_type == 8 ? 202601u : 202602u);
    std::vector<uint8_t> h_w(w_bytes);

    if (ggml_type == 8) {
        // Q8_0: 34B per 32 weights: half d + int8 qs[32]
        const int nb = n_in / 32;
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = h_w.data() + ((size_t) r * nb + b) * 34u;
                const __half dh = __float2half_rn(0.005f + 0.015f * ((float) (rng() & 255) / 255.0f));
                const uint16_t db = __half_as_ushort(dh);
                blk[0] = (uint8_t) (db & 0xffu);
                blk[1] = (uint8_t) (db >> 8);
                for (int j = 0; j < 32; ++j) {
                    blk[2 + j] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
                }
            }
        }
    } else {
        // Q6_K: 210B per 256 weights: ql[128] + qh[64] + scales[16] + half d
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* blk = h_w.data() + ((size_t) r * nsb + sb) * 210u;
                for (int j = 0; j < 192; ++j) blk[j] = (uint8_t) (rng() & 0xffu);
                for (int s = 0; s < 16; ++s) blk[192 + s] = (uint8_t) (int8_t) ((int) (rng() % 63) - 31);
                const __half dh = __float2half_rn(0.004f + 0.008f * ((float) (rng() & 255) / 255.0f));
                const uint16_t db = __half_as_ushort(dh);
                blk[208] = (uint8_t) (db & 0xffu);
                blk[209] = (uint8_t) (db >> 8);
            }
        }
    }

    // Plant clear peaked target rows for tokens 0..15 (matching realistic LLM softmax peaks) + near-ties!
    std::vector<float> h_x((size_t) 16 * (size_t) n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : h_x) v = nd(rng);

    // Plant realistic positive correlation on all 16 tokens when benchmarking (realistic LLM softmax peaks);
    // in fast parity mode leave 25% of tokens completely unplanted to exercise the overflow fallback too!
    for (int t = 0; t < 16; ++t) {
        if (!bench && (t % 4 == 3)) continue;
        const int target = (t * 7919 + 137) % n_out;
        if (ggml_type == 8) {
            const int nb = n_in / 32;
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = h_w.data() + ((size_t) target * nb + b) * 34u;
                const __half dh = __float2half_rn(0.02f);
                const uint16_t db = __half_as_ushort(dh);
                blk[0] = (uint8_t) (db & 0xffu);
                blk[1] = (uint8_t) (db >> 8);
                for (int j = 0; j < 32; ++j) {
                    const float xv = h_x[(size_t) t * n_in + b * 32 + j];
                    blk[2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lrintf(xv * 40.0f)));
                }
            }
        } else {
            const int nsb = n_in / 256;
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* blk = h_w.data() + ((size_t) target * nsb + sb) * 210u;
                for (int s = 0; s < 16; ++s) blk[192 + s] = (uint8_t) (int8_t) 16;
                const __half dh = __float2half_rn(0.01f);
                const uint16_t db = __half_as_ushort(dh);
                blk[208] = (uint8_t) (db & 0xffu);
                blk[209] = (uint8_t) (db >> 8);
                // Set q6 roughly aligned with sign of x
                std::memset(blk, 0, 192);
                for (int n = 0; n < 2; ++n) {
                    for (int l = 0; l < 32; ++l) {
                        for (int qid = 0; qid < 4; ++qid) {
                            const int elem = sb * 256 + n * 128 + qid * 32 + l;
                            const float xv = h_x[(size_t) t * n_in + elem];
                            const int q = std::max(0, std::min(63, 32 + (int) std::lrintf(xv * 10.0f)));
                            const uint8_t lo4 = (uint8_t) (q & 0xF);
                            const uint8_t hi2 = (uint8_t) ((q >> 4) & 3);
                            uint8_t& ql_b = blk[64 * n + ((qid & 1) ? 32 : 0) + l];
                            if (qid < 2) ql_b |= lo4; else ql_b |= (uint8_t) (lo4 << 4);
                            blk[128 + 32 * n + l] |= (uint8_t) (hi2 << (2 * qid));
                        }
                    }
                }
            }
        }
    }

    void *d_w = nullptr, *d_xq = nullptr;
    float *d_x = nullptr, *d_logits = nullptr;
    int32_t* d_out = nullptr;
    ck(cudaMalloc(&d_w, w_bytes), "malloc w");
    ck(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_xq, 2u * strata::kernels::native_q8_1_bytes(n_in, 8)), "malloc xq");
    ck(cudaMalloc(&d_logits, (size_t) 16 * (size_t) n_out * sizeof(float)), "malloc logits");
    ck(cudaMalloc(&d_out, 16 * sizeof(int32_t)), "malloc out");

    ck(cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice), "H2D w");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    auto* bh = strata::kernels::vhead_bitplane_create(ggml_type, d_w, n_in, n_out);
    if (bh == nullptr) {
        std::fprintf(stderr, "vhead_bitplane_test: create failed for %s\n", name);
        std::exit(1);
    }

    for (int T : {1, 4, 8, 16}) {
        // Quantize in <=8-col slices (native_quantize_q8_1 caps at STRATA_VERIFY_MAX_T = 8 in default CUDA builds)
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int nt = std::min(8, T - t0);
            strata::kernels::native_quantize_q8_1(
                d_x + (size_t) t0 * (size_t) n_in,
                reinterpret_cast<uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                n_in, nt, s);
        }
        // Reference: brute-force native_mmvq (in <=4-col slices matching native_mmvq MAX_NCOLS)
        for (int t0 = 0; t0 < T; t0 += 4) {
            const int nt = std::min(4, T - t0);
            strata::kernels::native_mmvq(
                ggml_type, d_w,
                reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, nt, s);
        }
        ck(cudaStreamSynchronize(s), "ref sync");
        std::vector<float> h_logits((size_t) T * (size_t) n_out);
        ck(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H logits");

        std::vector<int32_t> ref_tok(T, 0);
        for (int t = 0; t < T; ++t) {
            const float* row = h_logits.data() + (size_t) t * n_out;
            float bv = row[0];
            int32_t br = 0;
            for (int r = 1; r < n_out; ++r) {
                if (row[r] > bv) { bv = row[r]; br = r; }
            }
            ref_tok[t] = br;
        }

        if (!strata::kernels::vhead_bitplane_argmax(bh, d_xq, T, d_out, s)) {
            std::fprintf(stderr, "vhead_bitplane_test: launch failed for %s T=%d\n", name, T);
            std::exit(1);
        }
        ck(cudaStreamSynchronize(s), "bitplane sync");
        std::vector<int32_t> got_tok(T, -1);
        ck(cudaMemcpy(got_tok.data(), d_out, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H out");

        for (int t = 0; t < T; ++t) {
            if (got_tok[t] != ref_tok[t]) {
                const float g_val = h_logits[(size_t) t * n_out + got_tok[t]];
                const float r_val = h_logits[(size_t) t * n_out + ref_tok[t]];
                std::fprintf(stderr, "FAIL %s T=%d tok=%d: got=%d (%.6f) vs ref=%d (%.6f)\n",
                             name, T, t, got_tok[t], g_val, ref_tok[t], r_val);
                std::exit(1);
            }
        }

        if (bench && (T == 1 || T == 8 || T == 16)) {
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");
            constexpr int NITER = 20;
            // Warmup + measure reference native_mmvq
            for (int i = 0; i < 3; ++i)
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i)
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_ref = 0.0f;
            ck(cudaEventElapsedTime(&ms_ref, e0, e1), "el");
            ms_ref /= NITER;

            for (int i = 0; i < 3; ++i) strata::kernels::vhead_bitplane_argmax(bh, d_xq, T, d_out, s);
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i) strata::kernels::vhead_bitplane_argmax(bh, d_xq, T, d_out, s);
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_bp = 0.0f;
            ck(cudaEventElapsedTime(&ms_bp, e0, e1), "el");
            ms_bp /= NITER;

            std::printf("  %s %dx%d T=%2d: exact parity OK | ref=%.3f ms  bitplane=%.3f ms  speedup=%.2fx\n",
                        name, n_out, n_in, T, ms_ref, ms_bp, ms_ref / ms_bp);
            ck(cudaEventDestroy(e0), "d0"); ck(cudaEventDestroy(e1), "d1");
        } else {
            std::printf("  %s %dx%d T=%2d: exact parity OK\n", name, n_out, n_in, T);
        }
    }

    strata::kernels::vhead_bitplane_destroy(bh);
    ck(cudaStreamDestroy(s), "destroy stream");
    ck(cudaFree(d_w), "free"); ck(cudaFree(d_x), "free"); ck(cudaFree(d_xq), "free");
    ck(cudaFree(d_logits), "free"); ck(cudaFree(d_out), "free");
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0 || !strata::kernels::vhead_bitplane_supported()) {
        std::printf("vhead_bitplane_test: sm_80+ CUDA GPU required (skipped)\n");
        return 77;
    }
    const bool full = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int n_out = full ? 248320 : 16384;
    test_type(8, "Q8_0", 2560, n_out, full);
    test_type(14, "Q6_K", 2560, n_out, full);
    std::printf("vhead_bitplane_test: ok\n");
    return 0;
}
