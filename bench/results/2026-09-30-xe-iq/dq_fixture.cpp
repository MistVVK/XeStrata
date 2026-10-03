// Scratch check: GPU dequant of the fixture rows the MMVQ-less formats skip in iq_parity (not committed).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    auto& q = strata::core::Runtime::get().compute();
    int fails = 0;
    for (const char* nm : {"IQ4_NL", "IQ4_XS", "Q2_0", "Q3_K"}) {
        const std::string d = argv[1];
        FILE* f = std::fopen((d + "/" + nm + ".bin").c_str(), "rb"); FILE* g = std::fopen((d + "/" + nm + ".f32").c_str(), "rb");
        int h[3]; std::fread(h, 4, 3, f);
        std::vector<uint8_t> raw; uint8_t buf[65536]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) raw.insert(raw.end(), buf, buf + n);
        std::vector<float> ref((size_t) h[1] * h[2]); std::fread(ref.data(), 4, ref.size(), g);
        auto* dw = sycl::malloc_device<uint8_t>(raw.size(), q); auto* dq = sycl::malloc_device<float>(ref.size(), q);
        q.memcpy(dw, raw.data(), raw.size()).wait();
        strata::kernels::iq_dequant_f32(h[0], dw, (int64_t) ref.size(), dq, nullptr);
        std::vector<float> got(ref.size()); q.memcpy(got.data(), dq, got.size() * 4).wait();
        double num = 0, den = 0; for (size_t i = 0; i < ref.size(); ++i) { num += std::fabs(got[i] - ref[i]); den += std::fabs(ref[i]); }
        std::printf("%-7s %d x %d  dequant rel %.2e\n", nm, h[1], h[2], num / (den + 1e-30)); fails += num / (den + 1e-30) >= 1e-6;
    }
    return fails;
}
