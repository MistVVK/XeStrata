// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Shape-specific BF16 GEMM parity and timing tool. Kept separate from kernel instrumentation.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct Buffers {
    uint16_t* x = nullptr;
    uint16_t* w = nullptr;
    float* y = nullptr;
    ~Buffers() { strata::gpu::free(x); strata::gpu::free(w); strata::gpu::free(y); }
};

bool upload(void* dst, const void* src, size_t bytes) {
    auto& rt = strata::core::Runtime::get();
    try { rt.wait(rt.copy_async(dst, src, bytes)); return true; }
    catch (const std::exception& e) { std::fprintf(stderr, "copy: %s\n", e.what()); return false; }
}

bool download(void* dst, const void* src, size_t bytes) {
    auto& rt = strata::core::Runtime::get();
    try {
        auto ready = rt.compute().ext_oneapi_submit_barrier();
        rt.wait(rt.copy_async(dst, src, bytes, {ready}));
        return true;
    } catch (const std::exception& e) { std::fprintf(stderr, "copy: %s\n", e.what()); return false; }
}

bool allocate(Buffers& b, int64_t T, int64_t N, int64_t K) {
    return strata::gpu::alloc_device(&b.x, (size_t) (T * K) * sizeof(uint16_t)) &&
           strata::gpu::alloc_device(&b.w, (size_t) (N * K) * sizeof(uint16_t)) &&
           strata::gpu::alloc_device(&b.y, (size_t) (T * N) * sizeof(float));
}

bool bench(int64_t T, int64_t N, int64_t K, int rounds, bool compare = false, bool half = false) {
    Buffers b;
    if (!allocate(b, T, N, K)) return false;
    std::vector<uint16_t> x((size_t) (T * K)), w((size_t) (N * K));
    std::vector<float> y((size_t) (T * N), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (uint16_t) (0x3f00u + (i * 37u % 0x100u));
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint16_t) (0x3e80u + (i * 19u % 0x180u));
    if (!upload(b.x, x.data(), x.size() * 2) || !upload(b.w, w.data(), w.size() * 2) ||
        !upload(b.y, y.data(), y.size() * 4)) return false;
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(nullptr, 0, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return false; }
    for (int i = 0; i < 2; ++i) {
        if (half) gemm.f16(b.x, b.w, b.y, T, N, K, N);
        else gemm.bf16(b.x, b.w, b.y, T, N, K, N);
    }
    strata::gpu::stream_sync(nullptr);
    if (compare) {
        std::vector<double> samples[3];
        const char* names[] = {"thin", "oneMKL", "rows"};
        for (int i = 0; i < rounds; ++i) for (int mode = 0; mode < 3; ++mode) {
            if (mode == 0) { unset_env("STRATA_NO_BLAS"); if (!half) set_env("STRATA_BF16_THIN_XMX", "1"); }
            if (mode == 1) { unset_env("STRATA_NO_BLAS"); unset_env("STRATA_BF16_THIN_XMX"); }
            if (mode == 2) { set_env("STRATA_NO_BLAS", "1"); unset_env("STRATA_BF16_THIN_XMX"); }
            const auto start = Clock::now();
            if (mode == 2)
                strata::kernels::gemm_rows(half ? strata::kernels::XmxType::f16 : strata::kernels::XmxType::bf16,
                                           b.x, b.w, b.y, T, N, K, N, nullptr);
            else if (half)
                gemm.f16(b.x, b.w, b.y, T, N, K, N);
            else
                gemm.bf16(b.x, b.w, b.y, T, N, K, N);
            strata::gpu::stream_sync(nullptr);
            samples[mode].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        }
        for (int mode = 0; mode < 3; ++mode) {
            auto& us = samples[mode]; std::sort(us.begin(), us.end());
            std::printf("SAMPLE mode=%s T=%lld N=%lld K=%lld us=", names[mode], (long long) T, (long long) N, (long long) K);
            for (size_t i = 0; i < us.size(); ++i) std::printf("%s%.2f", i ? "," : "", us[i]);
            const double median = us[us.size() / 2];
            const double tflops = 2.0 * (double) T * (double) N * (double) K / median / 1.0e6;
            std::printf(" median=%.2f us tflops=%.3f\n", median, tflops);
        }
        return true;
    }
    std::vector<double> us;
    for (int i = 0; i < rounds; ++i) {
        const auto start = Clock::now();
        gemm.bf16(b.x, b.w, b.y, T, N, K, N);
        strata::gpu::stream_sync(nullptr);
        us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
    }
    std::sort(us.begin(), us.end());
    std::printf("SAMPLE T=%lld N=%lld K=%lld us=", (long long) T, (long long) N, (long long) K);
    for (size_t i = 0; i < us.size(); ++i) std::printf("%s%.2f", i ? "," : "", us[i]);
    std::printf(" median=%.2f\n", us[us.size() / 2]);
    return true;
}

bool parity(int64_t only_t = 0, int64_t only_n = 0, int64_t only_k = 0, int64_t only_ldy = 0, bool tiny = false) {
    // Pin the reference implementation to the existing non-BLAS row/XMX dispatch. The feature flag is read per call.
    set_env("STRATA_NO_BLAS", "1");
    unset_env("STRATA_BF16_THIN_XMX");
    const std::vector<int64_t> ts = only_t ? std::vector<int64_t>{only_t}
        : tiny ? std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8} : std::vector<int64_t>{1, 7, 32, 33, 65};
    const std::vector<int64_t> ks = only_t ? std::vector<int64_t>{only_k}
        : tiny ? std::vector<int64_t>{16, 31, 32, 33, 48, 64} : std::vector<int64_t>{16, 31, 32, 33, 64};
    auto& rt = strata::core::Runtime::get();
    size_t cases = 0;
    for (int64_t T : ts) for (int64_t N = 1; N < (only_t ? only_n + 1 : tiny ? 5 : 16); ++N) {
      if (only_t && N != only_n) continue;
      for (int64_t K : ks) for (int64_t ldy : only_t ? std::vector<int64_t>{only_ldy}
                                                       : tiny ? std::vector<int64_t>{N, N + 3}
                                                              : std::vector<int64_t>{N + ((N + T) % 2 ? 3 : 0)}) {
        const size_t y_elems = (size_t) ((T - 1) * ldy + N);
        Buffers b;
        if (!strata::gpu::alloc_device(&b.x, (size_t) (T * K) * 2) ||
            !strata::gpu::alloc_device(&b.w, (size_t) (N * K) * 2) ||
            !strata::gpu::alloc_device(&b.y, y_elems * 4)) return false;
        std::vector<uint16_t> x((size_t) (T * K)), w((size_t) (N * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = (uint16_t) (0x3d00u + (i * 11u % 0x300u));
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint16_t) (0x3c80u + (i * 17u % 0x280u));
        std::vector<float> initial(y_elems);
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = (float) ((int) (i % 9) - 4) * 0.125f;
        if (!upload(b.x, x.data(), x.size() * 2) || !upload(b.w, w.data(), w.size() * 2)) return false;
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init(nullptr, 0, error)) return false;
        std::vector<float> ref(y_elems), got(y_elems);
        if (!upload(b.y, initial.data(), y_elems * 4)) return false;
        unset_env("STRATA_BF16_THIN_XMX");
        gemm.bf16(b.x, b.w, b.y, T, N, K, ldy);
        if (!download(ref.data(), b.y, y_elems * 4)) return false;
        if (!upload(b.y, initial.data(), y_elems * 4)) return false;
        set_env("STRATA_BF16_THIN_XMX", "0");
        gemm.bf16(b.x, b.w, b.y, T, N, K, ldy);
        if (!download(got.data(), b.y, y_elems * 4) || std::memcmp(ref.data(), got.data(), y_elems * 4) != 0) {
            std::fprintf(stderr, "default changed T=%lld N=%lld K=%lld ldy=%lld\n", (long long) T, (long long) N,
                         (long long) K, (long long) ldy);
            return false;
        }
        if (N < 16 && K >= 32 && K % 32 == 0 && T > 0) {
            if (!upload(b.y, initial.data(), y_elems * 4)) return false;
            set_env("STRATA_BF16_THIN_XMX", "1");
            gemm.bf16(b.x, b.w, b.y, T, N, K, ldy);
            if (!download(got.data(), b.y, y_elems * 4)) return false;
            for (size_t i = 0; i < y_elems; ++i) {
                const float tol = 0.002f + 0.002f * std::fabs(ref[i]);
                if (std::fabs(ref[i] - got[i]) > tol) {
                    std::fprintf(stderr, "thin mismatch T=%lld N=%lld K=%lld ldy=%lld at %zu: %.8g %.8g\n",
                                 (long long) T, (long long) N, (long long) K, (long long) ldy, i, ref[i], got[i]);
                    return false;
                }
            }
        }
        ++cases;
      }
    }
    // Accumulation is intentionally outside the dedicated non-accumulating fast path.
    if (only_t)
        std::printf("PASS parity cases=%zu; default and explicit-off are byte-identical; targeted shape checked\n", cases);
    else if (tiny)
        std::printf("PASS parity cases=%zu; default and explicit-off are byte-identical; T=1..8, N=1..4, K boundaries, ldy=N/padded\n", cases);
    else
        std::printf("PASS parity cases=%zu; default and explicit-off are byte-identical; widths 1..15, K boundaries, ldy=N/padded\n", cases);
    (void) rt;
    return true;
}

bool guard() {
    if (!strata::kernels::xmx_available(strata::kernels::XmxType::bf16)) {
        std::puts("SKIP guard: BF16 matrix engines unavailable");
        return true;
    }
    constexpr int64_t K = 32;
    constexpr size_t guard_floats = 64;
    const std::vector<int64_t> ts{1, 7, 8, 31, 32, 33, 31393};
    for (int64_t T : ts) {
        const int64_t y_rows = ((T + 63) / 64) * 64;
        const size_t y_elems = (size_t) y_rows * 16 + guard_floats;
        Buffers b;
        if (!strata::gpu::alloc_device(&b.x, (size_t) T * K * sizeof(uint16_t)) ||
            !strata::gpu::alloc_device(&b.w, (size_t) 16 * K * sizeof(uint16_t)) ||
            !strata::gpu::alloc_device(&b.y, y_elems * sizeof(float))) return false;
        std::vector<uint16_t> x((size_t) T * K, 0x3f80u), w((size_t) 16 * K, 0x3f80u);
        std::vector<float> y(y_elems, -12345.0f);
        if (!upload(b.x, x.data(), x.size() * sizeof(uint16_t)) ||
            !upload(b.w, w.data(), w.size() * sizeof(uint16_t)) || !upload(b.y, y.data(), y.size() * sizeof(float)))
            return false;
        strata::kernels::xmx_gemm_bf16_narrow(b.x, b.w, b.y, T, K, nullptr);
        if (!download(y.data(), b.y, y.size() * sizeof(float))) return false;
        for (size_t i = (size_t) y_rows * 16; i < y.size(); ++i) {
            if (y[i] != -12345.0f) {
                std::fprintf(stderr, "guard overwritten T=%lld at float %zu\n", (long long) T, i);
                return false;
            }
        }
    }
    std::printf("PASS guard cases=%zu; 64-row scratch boundary plus %zu-float tail unchanged\n", ts.size(), guard_floats);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--parity") == 0) return parity() ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--tiny-parity") == 0) return parity(0, 0, 0, 0, true) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--guard") == 0) return guard() ? 0 : 1;
    if (argc == 6 && std::strcmp(argv[1], "--case") == 0)
        return parity(std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10),
                      std::strtoll(argv[4], nullptr, 10), std::strtoll(argv[5], nullptr, 10)) ? 0 : 1;
    if (argc >= 5 && std::strcmp(argv[1], "--bench") == 0)
        return bench(std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10),
                     std::strtoll(argv[4], nullptr, 10), argc > 5 ? (int) std::strtol(argv[5], nullptr, 10) : 5) ? 0 : 1;
    if (argc >= 5 && std::strcmp(argv[1], "--ab") == 0)
        return bench(std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10),
                     std::strtoll(argv[4], nullptr, 10), argc > 5 ? (int) std::strtol(argv[5], nullptr, 10) : 5, true)
                   ? 0 : 1;
    if (argc >= 5 && std::strcmp(argv[1], "--ab-f16") == 0)
        return bench(std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10),
                     std::strtoll(argv[4], nullptr, 10), argc > 5 ? (int) std::strtol(argv[5], nullptr, 10) : 5,
                     true, true) ? 0 : 1;
    std::fprintf(stderr, "usage: %s --parity | --tiny-parity | --guard | --case T N K ldy | --bench/--ab/--ab-f16 T N K [rounds]\n", argv[0]);
    return 2;
}
