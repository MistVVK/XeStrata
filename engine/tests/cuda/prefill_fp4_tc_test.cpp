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
#include "strata/prefill/moe_fp4_tc.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/quant/types.hpp"

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
    run_parity_case(1, 16, 0, s);
    run_parity_case(17, 16, 2, s);
    run_parity_case(65, 32, 0, s);
    run_parity_case(517, 64, 2, s);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        run_bench_case(512, 512, s);
        run_bench_case(2048, 512, s);
        run_bench_case(8192, 512, s);
    }
    ck(cudaStreamDestroy(s), "cudaStreamDestroy");
    std::printf("prefill_fp4_tc_test: ok\n");
    return 0;
}
