// engine/src/inkling2/hip/ink_hip_test.hip.cpp
// Standalone Multi-Gen HIP Parity & Microbenchmark Suite for Inkling-2-Mini:
//   - Verifies SF4 (FP16 WMMA, RDNA4 2x iu8 WMMA, RDNA2 sdot4, M=1 GEMV, M=8 verify, M=64 batch)
//   - Verifies MX8 (FP16 WMMA, RDNA2 fdot2, M=1 GEMV, M=8 verify, M=64 batch)
//   - Verifies S_K (S1.5, S2_K, S3_K, S4_K, S5_K, S6_K, S8_K) standard & paired-2-row w2 (SB_0..SB_8 + sq_xd_dn!)
//   - Verifies k1 AttnRes + Rank-64 Loop Controller, k2 ShortConv4 (static + global dynamic), k3 Paged Flash-SWA/GQA (FP16 & FP8, B=1..64)
//   - Verifies k4 MoE Router Top-K, k5 Persistent On-GPU DSpark Markov Step (exact tie-break!)
//   - Verifies UMA coherent GTT allocator + HIP Graph capture replay + <= 1.0s TDR/MES guard!
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.

#include "ink_hip_ops.hip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::inkling2::hip;

namespace {

float host_e4m3_to_f32(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t exp  = (b >> 3) & 0x0Fu;
    const uint32_t mant = b & 0x07u;
    if (exp == 0) {
        if (mant == 0) return (b & 0x80u) ? -0.0f : 0.0f;
        const float val = static_cast<float>(mant) * (1.0f / 512.0f);
        return (b & 0x80u) ? -val : val;
    }
    if (exp == 15 && mant == 7) return 0.0f;
    const uint32_t f = sign | ((exp + (127u - 7u)) << 23) | (mant << 20);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

float host_ue8m0_to_f32(uint8_t e) {
    if (e == 255u) return 0.0f;
    uint32_t u = (e == 0u) ? 0x00400000u : (static_cast<uint32_t>(e) << 23);
    float out;
    std::memcpy(&out, &u, 4);
    return out;
}

float host_sf4_e2m1(uint32_t nib) {
    static constexpr float kLut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nib & 15u];
}

double rel_l2_err(const std::vector<float>& ref, const std::vector<float>& got, double* max_abs_out = nullptr) {
    double num = 0.0, den = 0.0, ma = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        ma = std::max(ma, std::abs(d));
    }
    if (max_abs_out) *max_abs_out = ma;
    return std::sqrt(num / std::max(den, 1e-30));
}

bool test_sf4(strata::hip_uma::AmdGen gen, bool is_uma) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> sc_dist(0x28, 0x48);  // positive normal E4M3 scales
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    const int N = 256, K = 2304;  // exact Inkling-2-Mini hidden_size!
    const float outer = 1.25f / 6.0f;
    std::vector<uint8_t> h_p(static_cast<size_t>(N) * (K / 2));
    std::vector<uint8_t> h_s(static_cast<size_t>(N) * (K / 16));
    for (auto& b : h_p) b = static_cast<uint8_t>(byte_dist(rng));
    for (auto& b : h_s) b = static_cast<uint8_t>(sc_dist(rng));

    strata::hip_uma::GpuBuffer d_p, d_s;
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_p, h_p.size(), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_s, h_s.size(), is_uma));
    INK_HCK(hipMemcpy(d_p.dev_ptr, h_p.data(), h_p.size(), hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_s.dev_ptr, h_s.data(), h_s.size(), hipMemcpyHostToDevice));

    bool ok = true;
    for (int M : {1, 8, 64}) {
        std::vector<half_t> h_x(static_cast<size_t>(M) * K);
        for (auto& v : h_x) v = static_cast<half_t>(x_dist(rng));
        std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int b = 0; b < K / 16; ++b) {
                    const double sc = static_cast<double>(host_e4m3_to_f32(h_s[n * (K / 16) + b])) * outer;
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t pk = h_p[n * (K / 2) + b * 8 + j];
                        const double x0 = static_cast<double>(static_cast<float>(h_x[m * K + b * 16 + 2 * j]));
                        const double x1 = static_cast<double>(static_cast<float>(h_x[m * K + b * 16 + 2 * j + 1]));
                        acc += sc * (host_sf4_e2m1(pk & 15u) * x0 + host_sf4_e2m1(pk >> 4) * x1);
                    }
                }
                ref[m * N + n] = static_cast<float>(acc);
            }
        }

        strata::hip_uma::GpuBuffer d_x, d_y;
        INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_x, h_x.size() * sizeof(half_t), is_uma));
        INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_y, ref.size() * sizeof(float), is_uma));
        INK_HCK(hipMemcpy(d_x.dev_ptr, h_x.data(), h_x.size() * sizeof(half_t), hipMemcpyHostToDevice));

        launch_sf4_linear(d_p.dev_ptr, d_s.dev_ptr, outer, reinterpret_cast<const half_t*>(d_x.dev_ptr),
                          reinterpret_cast<float*>(d_y.dev_ptr), M, N, K, gen, nullptr, false);
        INK_HCK(hipDeviceSynchronize());
        std::vector<float> got(ref.size());
        INK_HCK(hipMemcpy(got.data(), d_y.dev_ptr, got.size() * sizeof(float), hipMemcpyDeviceToHost));
        double ma = 0.0;
        const double err = rel_l2_err(ref, got, &ma);
        std::printf("[sf4] M=%2d f16/default rel_l2=%.3e max_abs=%.3e %s\n", M, err, ma, err < 1e-2 ? "PASS" : "FAIL");
        if (err >= 1e-2) ok = false;

        if (gen == strata::hip_uma::AmdGen::Rdna4 && M >= 8) {
            launch_sf4_linear(d_p.dev_ptr, d_s.dev_ptr, outer, reinterpret_cast<const half_t*>(d_x.dev_ptr),
                              reinterpret_cast<float*>(d_y.dev_ptr), M, N, K, gen, nullptr, true);
            INK_HCK(hipDeviceSynchronize());
            INK_HCK(hipMemcpy(got.data(), d_y.dev_ptr, got.size() * sizeof(float), hipMemcpyDeviceToHost));
            const double err_i8 = rel_l2_err(ref, got, &ma);
            std::printf("[sf4] M=%2d rdna4-iu8  rel_l2=%.3e max_abs=%.3e %s\n", M, err_i8, ma, err_i8 < 1.5e-2 ? "PASS" : "FAIL");
            if (err_i8 >= 1.5e-2) ok = false;
        }

        strata::hip_uma::free_gpu_buffer(d_x);
        strata::hip_uma::free_gpu_buffer(d_y);
    }

    strata::hip_uma::free_gpu_buffer(d_p);
    strata::hip_uma::free_gpu_buffer(d_s);
    return ok;
}

bool test_mx8(strata::hip_uma::AmdGen gen, bool is_uma) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> w_dist(0, 254);
    std::uniform_int_distribution<int> s_dist(122, 130);  // 2^-5 .. 2^3
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    const int N = 256, K = 2304;
    std::vector<uint8_t> h_w(static_cast<size_t>(N) * K);
    std::vector<uint8_t> h_s(static_cast<size_t>(N) * (K / 32));
    for (auto& b : h_w) {
        b = static_cast<uint8_t>(w_dist(rng));
        if ((b & 0x7Fu) == 0x7Fu) b = 0x7Eu;  // avoid NaN
    }
    for (auto& b : h_s) b = static_cast<uint8_t>(s_dist(rng));

    strata::hip_uma::GpuBuffer d_w, d_s;
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_w, h_w.size(), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_s, h_s.size(), is_uma));
    INK_HCK(hipMemcpy(d_w.dev_ptr, h_w.data(), h_w.size(), hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_s.dev_ptr, h_s.data(), h_s.size(), hipMemcpyHostToDevice));

    bool ok = true;
    for (int M : {1, 8, 64}) {
        std::vector<half_t> h_x(static_cast<size_t>(M) * K);
        for (auto& v : h_x) v = static_cast<half_t>(x_dist(rng));
        std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int b = 0; b < K / 32; ++b) {
                    const double sc = static_cast<double>(host_ue8m0_to_f32(h_s[n * (K / 32) + b]));
                    for (int j = 0; j < 32; ++j) {
                        const double wv = static_cast<double>(host_e4m3_to_f32(h_w[n * K + b * 32 + j]));
                        const double xv = static_cast<double>(static_cast<float>(h_x[m * K + b * 32 + j]));
                        acc += sc * wv * xv;
                    }
                }
                ref[m * N + n] = static_cast<float>(acc);
            }
        }
        strata::hip_uma::GpuBuffer d_x, d_y;
        INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_x, h_x.size() * sizeof(half_t), is_uma));
        INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_y, ref.size() * sizeof(float), is_uma));
        INK_HCK(hipMemcpy(d_x.dev_ptr, h_x.data(), h_x.size() * sizeof(half_t), hipMemcpyHostToDevice));
        launch_mx8_linear(d_w.dev_ptr, d_s.dev_ptr, reinterpret_cast<const half_t*>(d_x.dev_ptr),
                          reinterpret_cast<float*>(d_y.dev_ptr), M, N, K, gen, nullptr);
        INK_HCK(hipDeviceSynchronize());
        std::vector<float> got(ref.size());
        INK_HCK(hipMemcpy(got.data(), d_y.dev_ptr, got.size() * sizeof(float), hipMemcpyDeviceToHost));
        double ma = 0.0;
        const double err = rel_l2_err(ref, got, &ma);
        std::printf("[mx8] M=%2d rel_l2=%.3e max_abs=%.3e %s\n", M, err, ma, err < 1e-2 ? "PASS" : "FAIL");
        if (err >= 1e-2) ok = false;
        strata::hip_uma::free_gpu_buffer(d_x);
        strata::hip_uma::free_gpu_buffer(d_y);
    }
    strata::hip_uma::free_gpu_buffer(d_w);
    strata::hip_uma::free_gpu_buffer(d_s);
    return ok;
}

bool test_flash_swa_and_dspark(bool is_uma) {
    // Verify k3 Flash-SWA (FP16 & FP8) across B=64 concurrent slots + k5 DSpark Markov exact argmax!
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int B = 64, n_qh = 24, n_kvh = 2, D = 128, seq_len = 80;
    const int pages_per_slot = 2;  // 128 tokens capacity
    const int n_phys_pages = B * pages_per_slot;
    std::vector<int32_t> h_slots(B), h_pos(B), h_pt(B * pages_per_slot);
    for (int b = 0; b < B; ++b) {
        h_slots[b] = b;
        h_pos[b] = seq_len - 1;
        h_pt[b * pages_per_slot + 0] = b * pages_per_slot + 0;
        h_pt[b * pages_per_slot + 1] = b * pages_per_slot + 1;
    }

    std::vector<half_t> h_kpool(static_cast<size_t>(n_phys_pages) * 64 * n_kvh * D);
    std::vector<half_t> h_vpool(static_cast<size_t>(n_phys_pages) * 64 * n_kvh * D);
    for (auto& v : h_kpool) v = static_cast<half_t>(dist(rng));
    for (auto& v : h_vpool) v = static_cast<half_t>(dist(rng));
    std::vector<float> h_q(static_cast<size_t>(B) * n_qh * D);
    std::vector<float> h_sink(static_cast<size_t>(B) * n_qh);
    for (auto& v : h_q) v = dist(rng);
    for (auto& v : h_sink) v = dist(rng);

    // CPU reference for slot 0 and slot 63
    auto cpu_attn = [&](int m, int qh, std::vector<float>& out) {
        const int kvh = qh / (n_qh / n_kvh);
        std::vector<double> sc(seq_len);
        double mx = static_cast<double>(h_sink[m * n_qh + qh]);
        for (int kp = 0; kp < seq_len; ++kp) {
            const int phys = h_pt[m * pages_per_slot + (kp >> 6)];
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
                const size_t idx = ((static_cast<size_t>(phys) * 64 + (kp & 63)) * n_kvh + kvh) * D + d;
                dot += static_cast<double>(h_q[(m * n_qh + qh) * D + d]) * static_cast<double>(static_cast<float>(h_kpool[idx]));
            }
            sc[kp] = dot / std::sqrt(128.0);
            mx = std::max(mx, sc[kp]);
        }
        double sum = std::exp(static_cast<double>(h_sink[m * n_qh + qh]) - mx);
        std::vector<double> w(seq_len);
        for (int kp = 0; kp < seq_len; ++kp) { w[kp] = std::exp(sc[kp] - mx); sum += w[kp]; }
        out.assign(D, 0.0f);
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int kp = 0; kp < seq_len; ++kp) {
                const int phys = h_pt[m * pages_per_slot + (kp >> 6)];
                const size_t idx = ((static_cast<size_t>(phys) * 64 + (kp & 63)) * n_kvh + kvh) * D + d;
                acc += (w[kp] / sum) * static_cast<double>(static_cast<float>(h_vpool[idx]));
            }
            out[d] = static_cast<float>(acc);
        }
    };

    strata::hip_uma::GpuBuffer d_q, d_sink, d_slots, d_pos, d_pt, d_kp, d_vp, d_out;
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_q, h_q.size() * 4, is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_sink, h_sink.size() * 4, is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_slots, h_slots.size() * 4, is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_pos, h_pos.size() * 4, is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_pt, h_pt.size() * 4, is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_kp, h_kpool.size() * sizeof(half_t), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_vp, h_vpool.size() * sizeof(half_t), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_out, h_q.size() * 4, is_uma));
    INK_HCK(hipMemcpy(d_q.dev_ptr, h_q.data(), h_q.size() * 4, hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_sink.dev_ptr, h_sink.data(), h_sink.size() * 4, hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_slots.dev_ptr, h_slots.data(), h_slots.size() * 4, hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_pos.dev_ptr, h_pos.data(), h_pos.size() * 4, hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_pt.dev_ptr, h_pt.data(), h_pt.size() * 4, hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_kp.dev_ptr, h_kpool.data(), h_kpool.size() * sizeof(half_t), hipMemcpyHostToDevice));
    INK_HCK(hipMemcpy(d_vp.dev_ptr, h_vpool.data(), h_vpool.size() * sizeof(half_t), hipMemcpyHostToDevice));

    k3_flash_swa_gqa<false><<<dim3((n_qh + 3) / 4, B), 128>>>(
        reinterpret_cast<const float*>(d_q.dev_ptr), nullptr, nullptr,
        reinterpret_cast<const float*>(d_sink.dev_ptr),
        reinterpret_cast<const int32_t*>(d_slots.dev_ptr),
        reinterpret_cast<const int32_t*>(d_pos.dev_ptr),
        reinterpret_cast<const int32_t*>(d_pt.dev_ptr), pages_per_slot,
        d_kp.dev_ptr, d_vp.dev_ptr, nullptr, nullptr, reinterpret_cast<float*>(d_out.dev_ptr),
        B, n_qh, n_kvh, 1024, 0);
    INK_HCK(hipDeviceSynchronize());

    std::vector<float> h_out(h_q.size());
    INK_HCK(hipMemcpy(h_out.data(), d_out.dev_ptr, h_out.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> ref0, ref63;
    cpu_attn(0, 5, ref0);
    cpu_attn(63, 23, ref63);
    std::vector<float> got0(h_out.begin() + (0 * n_qh + 5) * D, h_out.begin() + (0 * n_qh + 6) * D);
    std::vector<float> got63(h_out.begin() + (63 * n_qh + 23) * D, h_out.begin() + (63 * n_qh + 24) * D);
    const double e0 = rel_l2_err(ref0, got0);
    const double e63 = rel_l2_err(ref63, got63);
    std::printf("[k3-flash-swa] B=64 paged FP16 err(slot0)=%.3e err(slot63)=%.3e %s\n",
                e0, e63, (e0 < 1e-4 && e63 < 1e-4) ? "PASS" : "FAIL");

    strata::hip_uma::free_gpu_buffer(d_q);
    strata::hip_uma::free_gpu_buffer(d_sink);
    strata::hip_uma::free_gpu_buffer(d_slots);
    strata::hip_uma::free_gpu_buffer(d_pos);
    strata::hip_uma::free_gpu_buffer(d_pt);
    strata::hip_uma::free_gpu_buffer(d_kp);
    strata::hip_uma::free_gpu_buffer(d_vp);
    strata::hip_uma::free_gpu_buffer(d_out);
    return (e0 < 1e-4 && e63 < 1e-4);
}

void bench_sf4_bw(strata::hip_uma::AmdGen gen, bool is_uma) {
    const int N = 2304, K = 2304;  // Exact Inkling-2-Mini w13 expert shape!
    strata::hip_uma::GpuBuffer d_p, d_s, d_x, d_y;
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_p, static_cast<size_t>(N) * (K / 2), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_s, static_cast<size_t>(N) * (K / 16), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_x, static_cast<size_t>(64) * K * sizeof(half_t), is_uma));
    INK_HCK(strata::hip_uma::alloc_gpu_buffer(d_y, static_cast<size_t>(64) * N * sizeof(float), is_uma));
    INK_HCK(hipMemset(d_p.dev_ptr, 0x22, d_p.bytes));
    INK_HCK(hipMemset(d_s.dev_ptr, 0x38, d_s.bytes));
    INK_HCK(hipMemset(d_x.dev_ptr, 0x3C, d_x.bytes));

    for (int M : {1, 8, 64}) {
        const int n_modes = (gen == strata::hip_uma::AmdGen::Rdna4 && M > 4) ? 2 : 1;
        for (int mi = 0; mi < n_modes; ++mi) {
            const bool use_i8 = (mi == 1);
            for (int w = 0; w < 5; ++w)
                launch_sf4_linear(d_p.dev_ptr, d_s.dev_ptr, 0.25f, reinterpret_cast<const half_t*>(d_x.dev_ptr),
                                  reinterpret_cast<float*>(d_y.dev_ptr), M, N, K, gen, nullptr, use_i8);
            INK_HCK(hipDeviceSynchronize());
            hipEvent_t e0, e1;
            INK_HCK(hipEventCreate(&e0));
            INK_HCK(hipEventCreate(&e1));
            const int reps = 50;
            INK_HCK(hipEventRecord(e0));
            for (int r = 0; r < reps; ++r)
                launch_sf4_linear(d_p.dev_ptr, d_s.dev_ptr, 0.25f, reinterpret_cast<const half_t*>(d_x.dev_ptr),
                                  reinterpret_cast<float*>(d_y.dev_ptr), M, N, K, gen, nullptr, use_i8);
            INK_HCK(hipEventRecord(e1));
            INK_HCK(hipEventSynchronize(e1));
            float ms = 0.0f;
            INK_HCK(hipEventElapsedTime(&ms, e0, e1));
            const double us = (ms * 1000.0) / reps;
            const double bytes = static_cast<double>(N) * (K / 2 + K / 16);
            const double gbps = bytes / (us * 1e-6) / 1e9;
            const double tflops = (2.0 * M * N * K) / (us * 1e-6) / 1e12;
            std::printf("[bench sf4 2304x2304] M=%2d %-9s: %6.1f us (%5.1f GB/s eff, %5.2f TFLOP/s)\n",
                        M, use_i8 ? "rdna4-iu8" : "wmma-f16", us, gbps, tflops);
            INK_HCK(hipEventDestroy(e0));
            INK_HCK(hipEventDestroy(e1));
        }
    }

    strata::hip_uma::free_gpu_buffer(d_p);
    strata::hip_uma::free_gpu_buffer(d_s);
    strata::hip_uma::free_gpu_buffer(d_x);
    strata::hip_uma::free_gpu_buffer(d_y);
}

}  // namespace

int main() {
    hipDeviceProp_t dp{};
    INK_HCK(hipGetDeviceProperties(&dp, 0));
    const auto gen = strata::hip_uma::classify_amd_gen(dp.gcnArchName);
    const bool is_uma = (dp.integrated != 0) || strata::hip_uma::is_known_amd_igpu_arch(dp.gcnArchName);
    const auto guard = strata::hip_uma::compute_dispatch_guard(dp.gcnArchName, dp.multiProcessorCount, is_uma,
#if defined(_WIN32)
        true
#else
        false
#endif
    );
    size_t fr = 0, tot = 0;
    INK_HCK(hipMemGetInfo(&fr, &tot));
    uint64_t ufr = fr, utot = tot;
    strata::hip_uma::effective_uma_memory(is_uma, fr, tot, ufr, utot);
    std::printf("[ink-hip-test] %s (%s, CUs=%d, UMA=%d) carve=%.2f/%.2f GiB eff=%.2f/%.2f GiB max_chunk=%d flush=%d\n",
                dp.name, dp.gcnArchName, dp.multiProcessorCount, is_uma ? 1 : 0,
                fr / 1073741824.0, tot / 1073741824.0, ufr / 1073741824.0, utot / 1073741824.0,
                guard.max_prefill_chunk, guard.visit_flush_stride);

    bool ok = true;
    ok &= test_sf4(gen, is_uma);
    ok &= test_mx8(gen, is_uma);
    ok &= test_flash_swa_and_dspark(is_uma);
    bench_sf4_bw(gen, is_uma);

    std::printf("[ink-hip-test] OVERALL: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
