// tests/cuda/prefill_fp4_tc_test.cpp - parity and throughput check for Blackwell (sm_120a / sm_121a) mxf4nvf4 prompt experts.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Validates strata::prefill::fp4_tc::quantize_act + fp4_tc::experts against:
//   1. Exact FP64 dequantized MoE reference across edge token counts (1, 17, 64, 65, 517), including an all-zero token
//      and 2-byte-shifted expert blobs;
//   2. Strata's W4A8 int8 fused path (moe_fused_iq.cu) on the same synthetic 512-expert layer, reporting relative RMS
//      error and paired GPU latency.
// Exits with code 77 (skipped) when no CUDA GPU is present or when fp4_tc::available() is false (non-sm_120/sm_121 or
// built without 120a/121a).
#include "strata/prefill/dense_fp4_tc.hpp"
#include "strata/prefill/moe_fp4_tc.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/quant/types.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace fused = strata::prefill::fused;
namespace fp4_tc = strata::prefill::fp4_tc;

constexpr int T_NVFP4 = (int) strata::quant::Type::NVFP4;
constexpr int N = fused::kMoeEmbd;
constexpr int FF = fused::kMoeFF;
constexpr int K = 10;
constexpr int BS = 36;   // sizeof(block_nvfp4)

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill_fp4_tc_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

float ue4m3_true_val(uint8_t x) {
    if (x == 0 || x == 0x7f || x == 0xff) return 0.0f;
    const int exp = (x >> 3) & 0xf;
    const int man = x & 0x7;
    if (exp == 0) return std::ldexp((float) man, -9);
    return std::ldexp(1.0f + (float) man / 8.0f, exp - 7);
}

uint8_t encode_ue4m3_host(float x, float& decoded) {
    if (!(x > 0.0f)) { decoded = 0.0f; return 0; }
    const float c = std::min(x, 448.0f);
    uint8_t best = 0;
    float best_val = 0.0f;
    float best_err = std::fabs(c);
    for (int code = 1; code < 0x7f; ++code) {
        const float v = ue4m3_true_val((uint8_t) code);
        const float err = std::fabs(c - v);
        if (err < best_err || (err == best_err && (code & 1) == 0)) {
            best_err = err;
            best_val = v;
            best = (uint8_t) code;
        }
    }
    decoded = best_val;
    return best;
}

float quantize_e2m1_host(float v) {
    static constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool neg = v < 0.0f;
    const float a = std::fabs(v);
    int best = 0;
    float best_err = a;
    for (int i = 1; i < 8; ++i) {
        const float err = std::fabs(a - lut[i]);
        if (err < best_err || (err == best_err && (i & 1) == 0)) {
            best_err = err;
            best = i;
        }
    }
    return neg ? -lut[best] : lut[best];
}

void quantize_act_nvfp4_host(const float* in, int cols, float* out) {
    for (int b = 0; b < cols; b += 64) {
        float am_64 = 0.0f;
        float am_sub[4] = {};
        for (int sub = 0; sub < 4; ++sub) {
            float m = 0.0f;
            for (int j = 0; j < 16; ++j) m = std::max(m, std::fabs(in[b + 16 * sub + j]));
            am_sub[sub] = m;
            am_64 = std::max(am_64, m);
        }
        const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
        for (int sub = 0; sub < 4; ++sub) {
            float s_sub = 0.0f;
            (void) encode_ue4m3_host(g_blk > 0.0f ? (am_sub[sub] / (g_blk * 6.0f)) : 0.0f, s_sub);
            const float denom = g_blk * s_sub;
            const float inv = denom > 0.0f ? (1.0f / denom) : 0.0f;
            for (int j = 0; j < 16; ++j) {
                out[b + 16 * sub + j] = denom > 0.0f ? (quantize_e2m1_host(in[b + 16 * sub + j] * inv) * denom) : 0.0f;
            }
        }
    }
}

float e2m1_true_val(uint8_t nibble) {
    static constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = lut[nibble & 0x7];
    return (nibble & 0x8) ? -m : m;
}

void dequant_nvfp4_row(const uint8_t* row, int cols, float* out) {
    const int nb = cols / 64;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* bp = row + (size_t) b * BS;
        for (int sub = 0; sub < 4; ++sub) {
            const float sc = ue4m3_true_val(bp[sub]);
            const uint8_t* qs = bp + 4 + 8 * sub;
            for (int j = 0; j < 8; ++j) {
                out[b * 64 + 16 * sub + j] = sc * e2m1_true_val(qs[j] & 0x0f);
                out[b * 64 + 16 * sub + 8 + j] = sc * e2m1_true_val(qs[j] >> 4);
            }
        }
    }
}

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    ~DevBuf() { if (p) (void) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <typename T> T* as() const { return reinterpret_cast<T*>(p); }
};

// Exercise the first row beyond the old 65535-CTA grid limit without allocating a full MoE layer.
void run_long_quantize_case(cudaStream_t s) {
    constexpr int blocks_per_row = N / 64;
    constexpr int rows = (65535 * 8) / blocks_per_row + 1;
    const size_t block_bytes = fused::act_bytes(1, 64);
    const size_t row_bytes = fused::act_bytes(1, N);
    const int endpoints[] = {0, rows - 1};
    DevBuf x((size_t) rows * N * sizeof(float));
    DevBuf packed(rows * row_bytes);
    DevBuf reference(2 * row_bytes);
    ck(cudaMemsetAsync(x.p, 0, (size_t) rows * N * sizeof(float), s), "long quantize input");
    ck(cudaMemsetAsync(packed.p, 0xff, rows * row_bytes, s), "long quantize sentinel");

    std::vector<float> samples(2 * N);
    std::vector<uint8_t> expected(2 * row_bytes), actual(2 * row_bytes);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < N; ++k) {
            const int sign = (k + i) % 2 == 0 ? 1 : -1;
            samples[i * N + k] = sign * (1 + (k * 7 + i * 11) % 37) * (1 + k / 64) / 32.0f;
        }
        float* row = x.as<float>() + (size_t) endpoints[i] * N;
        ck(cudaMemcpyAsync(row, samples.data() + i * N, N * sizeof(float), cudaMemcpyHostToDevice, s), "long quantize endpoint");
        fp4_tc::quantize_act(row, 1, N, reference.as<uint8_t>() + i * row_bytes, s);
    }
    fp4_tc::quantize_act(x.as<float>(), rows, N, packed.p, s);
    ck(cudaMemcpyAsync(expected.data(), reference.p, expected.size(), cudaMemcpyDeviceToHost, s), "short quantize results");
    for (int i = 0; i < 2; ++i) {
        ck(cudaMemcpyAsync(actual.data() + i * row_bytes, packed.as<uint8_t>() + endpoints[i] * row_bytes,
                           row_bytes, cudaMemcpyDeviceToHost, s), "long quantize results");
    }
    ck(cudaStreamSynchronize(s), "long quantize sync");
    // Each 64-value block writes 32 payload bytes, four scale codes and an outer float scale.
    for (int i = 0; i < 2; ++i) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t offset = i * row_bytes + b * block_bytes;
            if (std::memcmp(actual.data() + offset, expected.data() + offset, 40) != 0) {
                std::fprintf(stderr, "prefill_fp4_tc_test: activation block mismatch at row %d block %d\n", endpoints[i], b);
                std::exit(1);
            }
        }
    }
}

void run_parity_case(int T, int E, size_t blob_shift, cudaStream_t s) {
    const size_t gu_row = (size_t) (N / 64) * BS;
    const size_t d_row = (size_t) (FF / 64) * BS;
    const size_t up_off = (size_t) FF * gu_row;
    const size_t down_off = (size_t) (2 * FF) * gu_row;
    const size_t tail_off = down_off + (size_t) N * d_row;
    const size_t blob_bytes = tail_off + 16;
    const size_t stride = ((blob_bytes + 63) / 64) * 64;

    std::mt19937 rng(1200u + (unsigned) T * 17u + (unsigned) blob_shift);
    std::uniform_real_distribution<float> xdist(-1.5f, 1.5f);

    std::vector<uint8_t> h_blobs((size_t) E * stride + 64, 0);
    for (int e = 0; e < E; ++e) {
        uint8_t* bp = h_blobs.data() + (size_t) e * stride + blob_shift;
        // Gate + Up + Down blocks with finite non-NaN UE4M3 scales (0x18..0x48) and random E2M1 nibbles.
        const size_t total_blocks = (2 * (size_t) FF * (N / 64)) + ((size_t) N * (FF / 64));
        for (size_t b = 0; b < total_blocks; ++b) {
            uint8_t* blk = bp + b * BS;
            for (int sub = 0; sub < 4; ++sub) blk[sub] = (uint8_t) (0x24 + (rng() % 28));
            for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) (rng() & 0xffu);
        }
        const float scales[4] = {0.012f + 0.001f * (float) (e % 5),
                                 0.015f + 0.001f * (float) (e % 3),
                                 0.011f + 0.001f * (float) (e % 7),
                                 0.0f};
        std::memcpy(bp + tail_off, scales, 16);
    }

    std::vector<float> h_x((size_t) T * N, 0.0f);
    const int zero_tok = T >= 2 ? (T / 2) : -1;
    for (int t = 0; t < T; ++t) {
        if (t == zero_tok) continue;
        for (int c = 0; c < N; ++c) h_x[(size_t) t * N + c] = xdist(rng);
    }

    const int64_t rows = (int64_t) T * K;
    std::vector<int32_t> h_ids((size_t) rows);
    for (int t = 0; t < T; ++t) {
        int base = (int) (rng() % E);
        for (int k = 0; k < K; ++k) h_ids[(size_t) t * K + k] = (base + k * 3 + 1) % E;
    }

    DevBuf d_blobs(h_blobs.size());
    DevBuf d_x(h_x.size() * sizeof(float));
    DevBuf d_ids(h_ids.size() * sizeof(int32_t));
    DevBuf d_gu(fused::group_bytes(rows, E));
    DevBuf d_slot((size_t) rows * sizeof(int32_t));
    DevBuf d_src((size_t) rows * sizeof(int32_t));
    DevBuf d_xq(fused::act_bytes(T, N));
    DevBuf d_hq(fused::act_bytes(rows, FF));
    DevBuf d_dm((size_t) rows * N * sizeof(float));

    ck(cudaMemcpyAsync(d_blobs.p, h_blobs.data(), h_blobs.size(), cudaMemcpyHostToDevice, s), "H2D blobs");
    ck(cudaMemcpyAsync(d_x.p, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, s), "H2D x");
    ck(cudaMemcpyAsync(d_ids.p, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s), "H2D ids");
    ck(cudaMemsetAsync(d_dm.p, 0, (size_t) rows * N * sizeof(float), s), "zero dm");

    fp4_tc::quantize_act(d_x.as<float>(), T, N, d_xq.p, s);
    fused::group(d_ids.as<int32_t>(), rows, K, E, d_gu.p, d_slot.as<int32_t>(), d_src.as<int32_t>(), s);

    const fused::NativeGeom ng{T_NVFP4, T_NVFP4, gu_row, d_row, up_off, down_off};
    for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
        fused::Batch b;
        b.e0 = e0;
        b.e1 = std::min(E, e0 + fused::kMaxBatch);
        for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = d_blobs.as<uint8_t>() + (size_t) e * stride + blob_shift;
        fp4_tc::experts(b, ng, E, rows, d_gu.p, d_xq.p, d_src.as<int32_t>(), d_hq.p, d_dm.as<float>(), s);
    }
    ck(cudaStreamSynchronize(s), "sync");

    std::vector<int32_t> h_slot((size_t) rows);
    std::vector<float> h_dm((size_t) rows * N);
    ck(cudaMemcpy(h_slot.data(), d_slot.p, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H slot");
    ck(cudaMemcpy(h_dm.data(), d_dm.p, h_dm.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H dm");

    // Host FP64 references:
    //  1) exact W4A4 quantized FP64 model (matches mxf4nvf4 hardware arithmetic up to FP32 rounding -> must be < 5e-3!)
    //  2) unquantized FP64 reference (measures W4A4 double-E2M1 quantization noise floor)
    std::vector<float> w_buf((size_t) std::max(N, FF));
    std::vector<float> x_q4((size_t) N);
    std::vector<float> h_f32((size_t) FF);
    std::vector<float> h_q4((size_t) FF);
    std::vector<double> h_mid((size_t) FF);
    double num_sq = 0.0, den_sq = 0.0;
    double q4_num_sq = 0.0, q4_den_sq = 0.0;
    const int64_t step = rows <= 80 ? 1 : (rows / 40);
    for (int64_t r = 0; r < rows; r += step) {
        const int t = (int) (r / K);
        const int e = h_ids[(size_t) r];
        const uint8_t* bp = h_blobs.data() + (size_t) e * stride + blob_shift;
        float tail[4];
        std::memcpy(tail, bp + tail_off, 16);
        const float* xrow = h_x.data() + (size_t) t * N;
        quantize_act_nvfp4_host(xrow, N, x_q4.data());

        for (int f = 0; f < FF; ++f) {
            dequant_nvfp4_row(bp + (size_t) f * gu_row, N, w_buf.data());
            double g_acc = 0.0, g_q4 = 0.0;
            for (int c = 0; c < N; ++c) {
                g_acc += (double) w_buf[(size_t) c] * (double) xrow[c];
                g_q4 += (double) w_buf[(size_t) c] * (double) x_q4[(size_t) c];
            }
            dequant_nvfp4_row(bp + up_off + (size_t) f * gu_row, N, w_buf.data());
            double u_acc = 0.0, u_q4 = 0.0;
            for (int c = 0; c < N; ++c) {
                u_acc += (double) w_buf[(size_t) c] * (double) xrow[c];
                u_q4 += (double) w_buf[(size_t) c] * (double) x_q4[(size_t) c];
            }
            const double gt = g_acc * (double) tail[0];
            const double up = u_acc * (double) tail[1];
            h_mid[(size_t) f] = (gt / (1.0 + std::exp(-gt))) * up;
            const double gt4 = g_q4 * (double) tail[0];
            const double up4 = u_q4 * (double) tail[1];
            h_f32[(size_t) f] = (float) ((gt4 / (1.0 + std::exp(-gt4))) * up4);
        }
        quantize_act_nvfp4_host(h_f32.data(), FF, h_q4.data());
        const float* got = h_dm.data() + (size_t) h_slot[(size_t) r] * N;
        for (int o = 0; o < N; ++o) {
            dequant_nvfp4_row(bp + down_off + (size_t) o * d_row, FF, w_buf.data());
            double d_acc = 0.0, d_q4 = 0.0;
            for (int f = 0; f < FF; ++f) {
                d_acc += (double) w_buf[(size_t) f] * h_mid[(size_t) f];
                d_q4 += (double) w_buf[(size_t) f] * (double) h_q4[(size_t) f];
            }
            const double ref = d_acc * (double) tail[2];
            const double diff = (double) got[o] - ref;
            num_sq += diff * diff;
            den_sq += ref * ref;
            const double ref4 = d_q4 * (double) tail[2];
            const double diff4 = (double) got[o] - ref4;
            q4_num_sq += diff4 * diff4;
            q4_den_sq += ref4 * ref4;
        }
    }
    const double rel_rms = den_sq > 0.0 ? std::sqrt(num_sq / den_sq) : 0.0;
    const double rel_rms_q4 = q4_den_sq > 0.0 ? std::sqrt(q4_num_sq / q4_den_sq) : 0.0;
    std::printf("  T=%4d E=%3d shift=%zu: vs W4A4-FP64 model rel RMS = %.6f | vs unquant-FP64 rel RMS = %.4f\n",
                T, E, blob_shift, rel_rms_q4, rel_rms);
    if (!std::isfinite(rel_rms_q4) || rel_rms_q4 > 0.01) {
        std::fprintf(stderr, "prefill_fp4_tc_test: rel_rms_q4 %.6f exceeded W4A4 model parity tolerance (0.01)\n", rel_rms_q4);
        std::exit(1);
    }
    if (!std::isfinite(rel_rms) || rel_rms > 0.22) {
        std::fprintf(stderr, "prefill_fp4_tc_test: rel_rms %.4f exceeded unquantized W4A4 bound (0.22)\n", rel_rms);
        std::exit(1);
    }
    if (zero_tok >= 0) {
        for (int k = 0; k < K; ++k) {
            const float* z = h_dm.data() + (size_t) h_slot[(size_t) zero_tok * K + k] * N;
            for (int o = 0; o < N; ++o) {
                if (z[o] != 0.0f) {
                    std::fprintf(stderr, "prefill_fp4_tc_test: zero token output z[%d] = %g (want 0)\n", o, z[o]);
                    std::exit(1);
                }
            }
        }
    }
}

void run_bench_case(int T, int E, cudaStream_t s) {
    const size_t gu_row = (size_t) (N / 64) * BS;
    const size_t d_row = (size_t) (FF / 64) * BS;
    const size_t up_off = (size_t) FF * gu_row;
    const size_t down_off = (size_t) (2 * FF) * gu_row;
    const size_t tail_off = down_off + (size_t) N * d_row;
    const size_t stride = ((tail_off + 16 + 63) / 64) * 64;

    std::mt19937 rng(777u + (unsigned) T);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<uint8_t> h_blobs((size_t) E * stride, 0);
    for (int e = 0; e < E; ++e) {
        uint8_t* bp = h_blobs.data() + (size_t) e * stride;
        const size_t total_blocks = (2 * (size_t) FF * (N / 64)) + ((size_t) N * (FF / 64));
        for (size_t b = 0; b < total_blocks; ++b) {
            uint8_t* blk = bp + b * BS;
            for (int sub = 0; sub < 4; ++sub) blk[sub] = 0x38;
            for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) ((e * 13 + b + q) & 0xffu);
        }
        const float scales[4] = {0.012f, 0.015f, 0.011f, 0.0f};
        std::memcpy(bp + tail_off, scales, 16);
    }
    std::vector<float> h_x((size_t) T * N);
    for (float& v : h_x) v = xdist(rng);
    const int64_t rows = (int64_t) T * K;
    std::vector<int32_t> h_ids((size_t) rows);
    for (int64_t r = 0; r < rows; ++r) h_ids[(size_t) r] = (int32_t) (rng() % E);

    DevBuf d_blobs(h_blobs.size());
    DevBuf d_x(h_x.size() * sizeof(float));
    DevBuf d_ids(h_ids.size() * sizeof(int32_t));
    DevBuf d_gu(fused::group_bytes(rows, E));
    DevBuf d_slot((size_t) rows * sizeof(int32_t));
    DevBuf d_src((size_t) rows * sizeof(int32_t));
    DevBuf d_xq(fused::act_bytes(T, N));
    DevBuf d_hq(fused::act_bytes(rows, FF));
    DevBuf d_dm((size_t) rows * N * sizeof(float));

    ck(cudaMemcpyAsync(d_blobs.p, h_blobs.data(), h_blobs.size(), cudaMemcpyHostToDevice, s), "H2D blobs");
    ck(cudaMemcpyAsync(d_x.p, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, s), "H2D x");
    ck(cudaMemcpyAsync(d_ids.p, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s), "H2D ids");
    fused::group(d_ids.as<int32_t>(), rows, K, E, d_gu.p, d_slot.as<int32_t>(), d_src.as<int32_t>(), s);

    const fused::NativeGeom ng{T_NVFP4, T_NVFP4, gu_row, d_row, up_off, down_off};
    auto run_w4a4 = [&]() {
        fp4_tc::quantize_act(d_x.as<float>(), T, N, d_xq.p, s);
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0;
            b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = d_blobs.as<uint8_t>() + (size_t) e * stride;
            fp4_tc::experts(b, ng, E, rows, d_gu.p, d_xq.p, d_src.as<int32_t>(), d_hq.p, d_dm.as<float>(), s);
        }
    };
    auto run_w4a8 = [&]() {
        fused::quantize_act_native(d_x.as<float>(), T, N, d_xq.p, s);
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0;
            b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = d_blobs.as<uint8_t>() + (size_t) e * stride;
            fused::experts_native(b, ng, E, rows, d_gu.p, d_xq.p, d_src.as<int32_t>(), d_hq.p, d_dm.as<float>(), s);
        }
    };

    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), "ev0"); ck(cudaEventCreate(&ev1), "ev1");
    for (int i = 0; i < 2; ++i) run_w4a4();
    ck(cudaEventRecord(ev0, s), "ev0");
    for (int i = 0; i < 6; ++i) run_w4a4();
    ck(cudaEventRecord(ev1, s), "ev1");
    ck(cudaEventSynchronize(ev1), "sync");
    float ms4 = 0.0f; ck(cudaEventElapsedTime(&ms4, ev0, ev1), "el"); ms4 /= 6.0f;

    for (int i = 0; i < 2; ++i) run_w4a8();
    ck(cudaEventRecord(ev0, s), "ev0");
    for (int i = 0; i < 6; ++i) run_w4a8();
    ck(cudaEventRecord(ev1, s), "ev1");
    ck(cudaEventSynchronize(ev1), "sync");
    float ms8 = 0.0f; ck(cudaEventElapsedTime(&ms8, ev0, ev1), "el"); ms8 /= 6.0f;

    std::printf("  BENCH T=%4d E=%3d: W4A8=%.3f ms  W4A4(mxf4nvf4)=%.3f ms  speedup=%.2fx\n",
                T, E, ms8, ms4, ms8 / ms4);
    ck(cudaEventDestroy(ev0), "d0"); ck(cudaEventDestroy(ev1), "d1");
}

void run_dense_fp4_parity_case(int T, int N_out, int K_in, float beta, cudaStream_t s) {
    namespace dense_fp4 = strata::prefill::dense_fp4_tc;
    if (!dense_fp4::available()) {
        std::fprintf(stderr, "prefill_fp4_tc_test: dense_fp4_tc::available() is false on fp4_tc-capable device\n");
        std::exit(1);
    }
    const int ldy = N_out + 17;   // non-aligned stride to verify bounds and ldy handling
    const size_t w_row_bytes = (size_t) (K_in / 64) * BS;
    std::mt19937 rng(4400u + (unsigned) T * 31u + (unsigned) N_out * 7u);
    std::uniform_real_distribution<float> xdist(-1.5f, 1.5f);

    std::vector<uint8_t> h_w((size_t) N_out * w_row_bytes);
    for (int r = 0; r < N_out; ++r) {
        for (int b = 0; b < K_in / 64; ++b) {
            uint8_t* blk = h_w.data() + (size_t) r * w_row_bytes + (size_t) b * BS;
            for (int sub = 0; sub < 4; ++sub) blk[sub] = (uint8_t) (0x24 + (rng() % 28));
            for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) (rng() & 0xffu);
        }
    }

    std::vector<uint16_t> h_x16((size_t) T * K_in);
    std::vector<float> h_xf((size_t) T * K_in);
    for (size_t i = 0; i < h_x16.size(); ++i) {
        const __half hv = __float2half(xdist(rng));
        uint16_t bits = 0;
        std::memcpy(&bits, &hv, 2);
        h_x16[i] = bits;
        h_xf[i] = __half2float(hv);
    }

    const float kSentinel = 12345.25f;
    std::vector<float> h_y_init((size_t) T * ldy, kSentinel);
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N_out; ++n) h_y_init[(size_t) t * ldy + n] = xdist(rng) * 3.0f;
    }

    DevBuf d_w(h_w.size());
    DevBuf d_x(h_x16.size() * sizeof(uint16_t));
    DevBuf d_y(h_y_init.size() * sizeof(float));
    ck(cudaMemcpyAsync(d_w.p, h_w.data(), h_w.size(), cudaMemcpyHostToDevice, s), "dense H2D w");
    ck(cudaMemcpyAsync(d_x.p, h_x16.data(), h_x16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, s), "dense H2D x");
    ck(cudaMemcpyAsync(d_y.p, h_y_init.data(), h_y_init.size() * sizeof(float), cudaMemcpyHostToDevice, s), "dense H2D y");

    const bool ok = dense_fp4::gemm_f16_nvfp4(d_x.as<uint16_t>(), d_w.p, d_y.as<float>(), T, N_out, K_in, ldy, beta, s);
    if (!ok) {
        std::fprintf(stderr, "prefill_fp4_tc_test: dense_fp4_tc::gemm_f16_nvfp4 returned false for T=%d N=%d K=%d\n",
                     T, N_out, K_in);
        std::exit(1);
    }
    ck(cudaStreamSynchronize(s), "dense sync");

    std::vector<float> h_y(h_y_init.size());
    ck(cudaMemcpy(h_y.data(), d_y.p, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost), "dense D2H y");

    for (int t = 0; t < T; ++t) {
        for (int n = N_out; n < ldy; ++n) {
            if (h_y[(size_t) t * ldy + n] != kSentinel) {
                std::fprintf(stderr, "prefill_fp4_tc_test: dense OOB write at t=%d n=%d (got %g)\n",
                             t, n, h_y[(size_t) t * ldy + n]);
                std::exit(1);
            }
        }
    }

    std::vector<float> w_row((size_t) K_in);
    std::vector<float> x_q4((size_t) K_in);
    double q4_num = 0.0, q4_den = 0.0, raw_num = 0.0, raw_den = 0.0;
    const int n_step = N_out <= 320 ? 1 : (N_out / 128);
    const int t_step = T <= 65 ? 1 : (T / 32);
    for (int t = 0; t < T; t += t_step) {
        const float* xr = h_xf.data() + (size_t) t * K_in;
        quantize_act_nvfp4_host(xr, K_in, x_q4.data());
        for (int n = 0; n < N_out; n += n_step) {
            dequant_nvfp4_row(h_w.data() + (size_t) n * w_row_bytes, K_in, w_row.data());
            double acc4 = 0.0, acc_raw = 0.0;
            for (int k = 0; k < K_in; ++k) {
                acc4 += (double) w_row[(size_t) k] * (double) x_q4[(size_t) k];
                acc_raw += (double) w_row[(size_t) k] * (double) xr[k];
            }
            const double base = (beta == 0.0f) ? 0.0 : (double) beta * (double) h_y_init[(size_t) t * ldy + n];
            const double ref4 = base + acc4;
            const double ref_raw = base + acc_raw;
            const double got = (double) h_y[(size_t) t * ldy + n];
            q4_num += (got - ref4) * (got - ref4);
            q4_den += ref4 * ref4;
            raw_num += (got - ref_raw) * (got - ref_raw);
            raw_den += ref_raw * ref_raw;
        }
    }
    const double rel4 = q4_den > 0.0 ? std::sqrt(q4_num / q4_den) : 0.0;
    const double rel_raw = raw_den > 0.0 ? std::sqrt(raw_num / raw_den) : 0.0;
    std::printf("  DENSE T=%4d N=%4d K=%4d beta=%.1f: vs W4A4-FP64 rel RMS = %.6f | vs unquant-FP64 = %.4f\n",
                T, N_out, K_in, beta, rel4, rel_raw);
    if (!std::isfinite(rel4) || rel4 > 1e-3) {
        std::fprintf(stderr, "prefill_fp4_tc_test: dense rel4 %.6f exceeded tolerance (1e-3)\n", rel4);
        std::exit(1);
    }
    if (!std::isfinite(rel_raw) || rel_raw > 0.22) {
        std::fprintf(stderr, "prefill_fp4_tc_test: dense rel_raw %.4f exceeded bound (0.22)\n", rel_raw);
        std::exit(1);
    }
}

void run_dense_fp4_bench_case(int T, int N_out, int K_in, cudaStream_t s) {
    namespace dense_fp4 = strata::prefill::dense_fp4_tc;
    const size_t w_bytes = (size_t) N_out * (size_t) (K_in / 64) * BS;
    DevBuf d_w(w_bytes);
    DevBuf d_x((size_t) T * K_in * sizeof(uint16_t));
    DevBuf d_y((size_t) T * N_out * sizeof(float));
    ck(cudaMemsetAsync(d_w.p, 0x38, w_bytes, s), "bench memset w");
    ck(cudaMemsetAsync(d_x.p, 0x3c, (size_t) T * K_in * sizeof(uint16_t), s), "bench memset x");
    ck(cudaMemsetAsync(d_y.p, 0, (size_t) T * N_out * sizeof(float), s), "bench memset y");

    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), "ev0"); ck(cudaEventCreate(&ev1), "ev1");
    for (int i = 0; i < 3; ++i) dense_fp4::gemm_f16_nvfp4(d_x.as<uint16_t>(), d_w.p, d_y.as<float>(), T, N_out, K_in, N_out, 0.0f, s);
    ck(cudaEventRecord(ev0, s), "ev0");
    for (int i = 0; i < 20; ++i) dense_fp4::gemm_f16_nvfp4(d_x.as<uint16_t>(), d_w.p, d_y.as<float>(), T, N_out, K_in, N_out, 0.0f, s);
    ck(cudaEventRecord(ev1, s), "ev1");
    ck(cudaEventSynchronize(ev1), "sync");
    float ms = 0.0f; ck(cudaEventElapsedTime(&ms, ev0, ev1), "el"); ms /= 20.0f;
    const double tflops = (2.0 * (double) T * (double) N_out * (double) K_in) / ((double) ms * 1e9);
    std::printf("  DENSE BENCH T=%4d N=%4d K=%4d: %.3f ms (%.1f TFLOPS)\n", T, N_out, K_in, ms, tflops);
    ck(cudaEventDestroy(ev0), "d0"); ck(cudaEventDestroy(ev1), "d1");
}

void run_long_dense_case(int T, cudaStream_t s) {
    namespace dense_fp4 = strata::prefill::dense_fp4_tc;
    constexpr int N_out = 256;
    constexpr int K_in = N;
    const size_t w_row_bytes = (size_t) (K_in / 64) * BS;
    std::mt19937 rng(77000u + (unsigned) T);
    std::uniform_real_distribution<float> fdist(-1.25f, 1.25f);

    std::vector<uint8_t> h_w((size_t) N_out * w_row_bytes);
    for (int n = 0; n < N_out; ++n) {
        uint8_t* row = h_w.data() + (size_t) n * w_row_bytes;
        for (int b = 0; b < K_in / 64; ++b) {
            uint8_t* blk = row + (size_t) b * BS;
            for (int sub = 0; sub < 4; ++sub) blk[sub] = (uint8_t) (0x24 + (rng() % 24));
            for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) (rng() & 0xffu);
        }
    }

    // Fill a few distinct probe rows (including 0, T/2, T-1, an all-zero row, and an Inf/NaN edge row).
    std::vector<uint16_t> h_x16((size_t) T * K_in, 0);
    std::vector<float> h_probe[3];
    const int probe_rows[3] = {0, T / 2, T - 1};
    for (int p = 0; p < 3; ++p) {
        h_probe[p].resize((size_t) K_in);
        for (int k = 0; k < K_in; ++k) {
            const float v = __half2float(__float2half(fdist(rng)));
            h_probe[p][(size_t) k] = v;
            h_x16[(size_t) probe_rows[p] * K_in + k] = __half_as_ushort(__float2half(v));
        }
    }
    // Edge row 1: all zeros; edge row 2: Inf/NaN elements (must sanitize to finite 0, never NaN).
    h_x16[(size_t) 2 * K_in + 0] = 0x7C00u; // +Inf
    h_x16[(size_t) 2 * K_in + 1] = 0xFC00u; // -Inf
    h_x16[(size_t) 2 * K_in + 2] = 0x7E00u; // NaN

    DevBuf d_w(h_w.size());
    DevBuf d_x(h_x16.size() * sizeof(uint16_t));
    DevBuf d_y((size_t) T * N_out * sizeof(float));
    ck(cudaMemcpyAsync(d_w.p, h_w.data(), h_w.size(), cudaMemcpyHostToDevice, s), "long dense H2D w");
    ck(cudaMemcpyAsync(d_x.p, h_x16.data(), h_x16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, s), "long dense H2D x");
    ck(cudaMemsetAsync(d_y.p, 0xff, (size_t) T * N_out * sizeof(float), s), "long dense poison y");

    if (!dense_fp4::gemm_f16_nvfp4(d_x.as<uint16_t>(), d_w.p, d_y.as<float>(), T, N_out, K_in, N_out, 0.0f, s)) {
        std::fprintf(stderr, "prefill_fp4_tc_test: long dense_fp4_tc failed at T=%d\n", T);
        std::exit(1);
    }
    ck(cudaStreamSynchronize(s), "long dense sync");

    std::vector<float> h_y((size_t) T * N_out);
    ck(cudaMemcpy(h_y.data(), d_y.p, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost), "long dense D2H y");

    size_t nonfinite = 0;
    for (size_t i = 0; i < h_y.size(); ++i)
        if (!std::isfinite(h_y[i])) ++nonfinite;
    if (nonfinite > 0) {
        std::fprintf(stderr, "prefill_fp4_tc_test: long dense T=%d produced %zu non-finite outputs!\n", T, nonfinite);
        std::exit(1);
    }

    std::vector<float> w_row((size_t) K_in), x_q4((size_t) K_in);
    double num = 0.0, den = 0.0;
    for (int p = 0; p < 3; ++p) {
        quantize_act_nvfp4_host(h_probe[p].data(), K_in, x_q4.data());
        for (int n = 0; n < N_out; n += 4) {
            dequant_nvfp4_row(h_w.data() + (size_t) n * w_row_bytes, K_in, w_row.data());
            double acc = 0.0;
            for (int k = 0; k < K_in; ++k) acc += (double) w_row[(size_t) k] * (double) x_q4[(size_t) k];
            const double got = (double) h_y[(size_t) probe_rows[p] * N_out + n];
            num += (got - acc) * (got - acc);
            den += acc * acc;
        }
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("  LONG DENSE T=%5d N=%d K=%d: nonfinite=%zu rel_vs_W4A4_FP64=%.6f OK\n", T, N_out, K_in, nonfinite, rel);
    if (!std::isfinite(rel) || rel > 1e-3) {
        std::fprintf(stderr, "prefill_fp4_tc_test: long dense T=%d rel %.6f > 1e-3\n", T, rel);
        std::exit(1);
    }
}

void run_long_moe_case(int T, cudaStream_t s) {
    constexpr int E = 8;
    constexpr int K_top = 1;
    const size_t gu_row = (size_t) (N / 64) * BS;
    const size_t d_row = (size_t) (FF / 64) * BS;
    const size_t up_off = (size_t) FF * gu_row;
    const size_t down_off = (size_t) (2 * FF) * gu_row;
    const size_t tail_off = down_off + (size_t) N * d_row;
    const size_t blob_bytes = tail_off + 16;
    const size_t stride = ((blob_bytes + 63) / 64) * 64;

    std::mt19937 rng(88000u + (unsigned) T);
    std::uniform_real_distribution<float> xdist(-1.2f, 1.2f);
    std::vector<uint8_t> h_blobs((size_t) E * stride, 0);
    for (int e = 0; e < E; ++e) {
        uint8_t* bp = h_blobs.data() + (size_t) e * stride;
        const size_t total_blocks = (2 * (size_t) FF * (N / 64)) + ((size_t) N * (FF / 64));
        for (size_t b = 0; b < total_blocks; ++b) {
            uint8_t* blk = bp + b * BS;
            for (int sub = 0; sub < 4; ++sub) blk[sub] = (uint8_t) (0x24 + (rng() % 24));
            for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) (rng() & 0xffu);
        }
        const float scales[4] = {0.003f, 0.003f, 0.003f, 0.0f};
        std::memcpy(bp + tail_off, scales, 16);
    }

    std::vector<float> h_x((size_t) T * N, 0.0f);
    const int probe_toks[3] = {0, T / 2, T - 1};
    for (int p = 0; p < 3; ++p)
        for (int c = 0; c < N; ++c) h_x[(size_t) probe_toks[p] * N + c] = xdist(rng);
    // Edge token 1: all zeros; edge token 2: Inf/NaN elements (must sanitize to 0, zero NaNs in dm).
    h_x[(size_t) 2 * N + 0] = std::numeric_limits<float>::infinity();
    h_x[(size_t) 2 * N + 1] = -std::numeric_limits<float>::infinity();
    h_x[(size_t) 2 * N + 2] = std::numeric_limits<float>::quiet_NaN();

    const int64_t rows = (int64_t) T * K_top;
    std::vector<int32_t> h_ids((size_t) rows);
    for (int t = 0; t < T; ++t) h_ids[(size_t) t] = t % E;

    DevBuf d_blobs(h_blobs.size());
    DevBuf d_x(h_x.size() * sizeof(float));
    DevBuf d_ids(h_ids.size() * sizeof(int32_t));
    DevBuf d_gu(fused::group_bytes(rows, E));
    DevBuf d_slot((size_t) rows * sizeof(int32_t));
    DevBuf d_src((size_t) rows * sizeof(int32_t));
    DevBuf d_xq(fused::act_bytes(T, N));
    DevBuf d_hq(fused::act_bytes(rows, FF));
    DevBuf d_dm((size_t) rows * N * sizeof(float));

    ck(cudaMemcpyAsync(d_blobs.p, h_blobs.data(), h_blobs.size(), cudaMemcpyHostToDevice, s), "long moe H2D blobs");
    ck(cudaMemcpyAsync(d_x.p, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, s), "long moe H2D x");
    ck(cudaMemcpyAsync(d_ids.p, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s), "long moe H2D ids");
    ck(cudaMemsetAsync(d_dm.p, 0xff, (size_t) rows * N * sizeof(float), s), "long moe poison dm");

    fp4_tc::quantize_act(d_x.as<float>(), T, N, d_xq.p, s);
    fused::group(d_ids.as<int32_t>(), rows, K_top, E, d_gu.p, d_slot.as<int32_t>(), d_src.as<int32_t>(), s);
    const fused::NativeGeom ng{T_NVFP4, T_NVFP4, gu_row, d_row, up_off, down_off};
    fused::Batch b;
    b.e0 = 0; b.e1 = E;
    for (int e = 0; e < E; ++e) b.blob[e] = d_blobs.as<uint8_t>() + (size_t) e * stride;
    fp4_tc::experts(b, ng, E, rows, d_gu.p, d_xq.p, d_src.as<int32_t>(), d_hq.p, d_dm.as<float>(), s);
    ck(cudaStreamSynchronize(s), "long moe sync");

    std::vector<int32_t> h_slot((size_t) rows);
    std::vector<float> h_dm((size_t) rows * N);
    ck(cudaMemcpy(h_slot.data(), d_slot.p, h_slot.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "long moe D2H slot");
    ck(cudaMemcpy(h_dm.data(), d_dm.p, h_dm.size() * sizeof(float), cudaMemcpyDeviceToHost), "long moe D2H dm");

    size_t nonfinite = 0;
    for (size_t i = 0; i < h_dm.size(); ++i)
        if (!std::isfinite(h_dm[i])) ++nonfinite;
    if (nonfinite > 0) {
        std::fprintf(stderr, "prefill_fp4_tc_test: long moe T=%d produced %zu non-finite outputs!\n", T, nonfinite);
        std::exit(1);
    }

    // Verify exact W4A4-FP64 parity on the 3 probe tokens (0, T/2, T-1).
    std::vector<float> w_buf((size_t) std::max(N, FF)), x_q4((size_t) N), h_f32((size_t) FF), h_q4((size_t) FF);
    double num = 0.0, den = 0.0;
    for (int p = 0; p < 3; ++p) {
        const int t = probe_toks[p];
        const int e = h_ids[(size_t) t];
        const uint8_t* bp = h_blobs.data() + (size_t) e * stride;
        const float* xrow = h_x.data() + (size_t) t * N;
        quantize_act_nvfp4_host(xrow, N, x_q4.data());
        for (int f = 0; f < FF; ++f) {
            dequant_nvfp4_row(bp + (size_t) f * gu_row, N, w_buf.data());
            double g_q4 = 0.0;
            for (int c = 0; c < N; ++c) g_q4 += (double) w_buf[(size_t) c] * (double) x_q4[(size_t) c];
            dequant_nvfp4_row(bp + up_off + (size_t) f * gu_row, N, w_buf.data());
            double u_q4 = 0.0;
            for (int c = 0; c < N; ++c) u_q4 += (double) w_buf[(size_t) c] * (double) x_q4[(size_t) c];
            const double gt4 = g_q4 * 0.003;
            const double up4 = u_q4 * 0.003;
            h_f32[(size_t) f] = (float) ((gt4 / (1.0 + std::exp(-gt4))) * up4);
        }
        quantize_act_nvfp4_host(h_f32.data(), FF, h_q4.data());
        const float* got = h_dm.data() + (size_t) h_slot[(size_t) t] * N;
        for (int o = 0; o < N; o += 16) {
            dequant_nvfp4_row(bp + down_off + (size_t) o * d_row, FF, w_buf.data());
            double d_q4 = 0.0;
            for (int f = 0; f < FF; ++f) d_q4 += (double) w_buf[(size_t) f] * (double) h_q4[(size_t) f];
            const double ref = d_q4 * 0.003;
            num += ((double) got[o] - ref) * ((double) got[o] - ref);
            den += ref * ref;
        }
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("  LONG MOE   T=%5d E=%d: nonfinite=%zu rel_vs_W4A4_FP64=%.6f OK\n", T, E, nonfinite, rel);
    if (!std::isfinite(rel) || rel > 5e-3) {
        std::fprintf(stderr, "prefill_fp4_tc_test: long moe T=%d rel %.6f > 5e-3\n", T, rel);
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        std::printf("prefill_fp4_tc_test: no CUDA device (skipped)\n");
        return 77;
    }
    if (!fp4_tc::available()) {
        std::printf("prefill_fp4_tc_test: fp4_tc not available on this device/build (skipped)\n");
        return 77;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "cudaStreamCreate");
    run_long_quantize_case(s);
    run_parity_case(1, 16, 0, s);
    run_parity_case(17, 16, 2, s);
    run_parity_case(65, 32, 0, s);
    run_parity_case(517, 64, 2, s);
    run_dense_fp4_parity_case(33, 192, 256, 0.0f, s);
    run_dense_fp4_parity_case(65, 320, 512, 1.0f, s);
    run_dense_fp4_parity_case(130, 2560, 2560, 0.0f, s);
    run_long_dense_case(24576, s);
    run_long_dense_case(45056, s);
    run_long_moe_case(24576, s);
    run_long_moe_case(45056, s);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        run_bench_case(512, 512, s);
        run_bench_case(2048, 512, s);
        run_bench_case(8192, 512, s);
        run_dense_fp4_bench_case(512, 2560, 2560, s);
        run_dense_fp4_bench_case(2048, 2560, 2560, s);
        run_dense_fp4_bench_case(2048, 6144, 2560, s);
        run_dense_fp4_bench_case(8192, 6144, 2560, s);
    }
    ck(cudaStreamDestroy(s), "cudaStreamDestroy");
    std::printf("prefill_fp4_tc_test: ok\n");
    return 0;
}
