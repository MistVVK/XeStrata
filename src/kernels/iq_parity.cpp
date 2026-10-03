// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/iq_parity.cpp - plan v0.3 P6: the i-quant kernels against gguf-py on real rows.
//
//     python tools/iq_fixture.py --out logs/iq_fixture && build/iq_parity logs/iq_fixture
//
// Dequant must match gguf-py's values to fp32 rounding; the MMVQ dot (q8_1 activations) must match the float
// matrix-vector product within the activation rounding (a few 1e-3 relative).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "strata/core/runtime.hpp"

#include <memory>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {
using strata::core::Runtime;

// Device buffers live until the harness ends; copies go through the runtime's queues in order.
struct Device {
    std::vector<std::unique_ptr<strata::core::DeviceArena>> arenas;
    template<class T> T* alloc(size_t bytes) {
        arenas.push_back(std::make_unique<strata::core::DeviceArena>(bytes));
        return static_cast<T*>(arenas.back()->base());
    }
};
void upload(void* dst, const void* src, size_t bytes) {
    auto& runtime = Runtime::get();
    runtime.wait(runtime.copy_async(dst, src, bytes), "iq_parity upload");
}
void download(void* dst, const void* src, size_t bytes) {
    auto& runtime = Runtime::get();
    const auto produced = runtime.compute().ext_oneapi_submit_barrier();
    runtime.wait(runtime.copy_async(dst, src, bytes, {produced}), "iq_parity download");
}
}  // namespace

int main(int argc, char** argv) try {
    const std::string dir = argc > 1 ? argv[1] : "logs/iq_fixture";
    const char* names[] = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "Q2_0", "Q3_K"};
    int failures = 0;
    void* s = &Runtime::get().compute();
    Device dev;
    for (const char* nm : names) {
        std::FILE* f = std::fopen((dir + "/" + nm + ".bin").c_str(), "rb");
        std::FILE* g = std::fopen((dir + "/" + nm + ".f32").c_str(), "rb");
        if (!f || !g) { std::printf("%-8s missing fixture\n", nm); ++failures; continue; }
        int hdr[3];
        std::fread(hdr, 4, 3, f);
        const int type = hdr[0], rows = hdr[1], cols = hdr[2];
        std::vector<uint8_t> raw;
        { uint8_t buf[1 << 16]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) raw.insert(raw.end(), buf, buf + n); }
        std::vector<float> ref((size_t) rows * cols);
        std::fread(ref.data(), 4, ref.size(), g);
        std::fclose(f);
        std::fclose(g);
        void* dw = dev.alloc<uint8_t>(raw.size());
        float* dq = dev.alloc<float>(ref.size() * 4);
        upload(dw, raw.data(), raw.size());
        double dq_err = 0.0;
        if (strata::kernels::iq_supported(type) && ((size_t) rows * cols) % 256 == 0) {
            strata::kernels::iq_dequant_f32(type, dw, (int64_t) rows * cols, dq, s);
            std::vector<float> got(ref.size());
            download(got.data(), dq, got.size() * 4);
            double num = 0, den = 0;
            for (size_t i = 0; i < ref.size(); ++i) { num += std::fabs(got[i] - ref[i]); den += std::fabs(ref[i]); }
            dq_err = num / (den + 1e-30);
        }
        // the dot, two columns
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) 2 * cols);
        for (auto& v : x) v = nd(rng);
        float* dx = dev.alloc<float>(x.size() * 4);
        void* xq = dev.alloc<uint8_t>((size_t) 2 * cols / 32 * 36);
        float* dy = dev.alloc<float>((size_t) 2 * rows * 4);
        upload(dx, x.data(), x.size() * 4);
        strata::kernels::quantize_q8_1_rows(dx, 2, cols, xq, s);
        try {
            strata::kernels::native_mmvq(type, dw, xq, dy, cols, rows, 2, s);
        } catch (const std::exception& e) { std::printf("%-8s mmvq: %s\n", nm, e.what()); ++failures; continue; }
        std::vector<float> y((size_t) 2 * rows);
        download(y.data(), dy, y.size() * 4);
        double num = 0, den = 0;
        for (int c = 0; c < 2; ++c)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int k = 0; k < cols; ++k) acc += (double) ref[(size_t) r * cols + k] * x[(size_t) c * cols + k];
                num += std::fabs(y[(size_t) c * rows + r] - acc);
                den += std::fabs(acc);
            }
        const double mm_err = num / (den + 1e-30);
        const bool ok = dq_err < 1e-6 && mm_err < 2e-2;
        std::printf("%-8s type %2d %4d x %5d  dequant rel %.2e  mmvq rel %.2e  %s\n", nm, type, rows, cols, dq_err, mm_err,
                    ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }
    std::printf("iq_parity: %d failures\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "iq_parity: %s\n", e.what());
    return 1;
}
