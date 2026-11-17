#include "strata/prefill/gemm.hpp"
#include "strata/sycl_queue.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int64_t T = 3, N = 5, K = 32, LDY = N + 3;
constexpr float guard = -123.0f;

uint16_t bf16(float value) {
    return uint16_t(std::bit_cast<uint32_t>(value) >> 16);
}

bool run(bool native, int64_t ldx, float beta, int64_t scratch_rows) {
    auto& queue = dpct::get_in_order_queue();
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(&queue, scratch_rows * K, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    std::vector<uint16_t> x(size_t(T * ldx), native ? 0x7e00 : 0x7fc0);
    std::vector<uint16_t> weight(size_t(N * K));
    std::vector<uint8_t> blocks(size_t(N * 34));
    const uint16_t half_x[T] = {0x3800, 0x3c00, 0x3e00};
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k)
            x[size_t(t * ldx + k)] = native ? half_x[t] : bf16(float(t + 1) / 2);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            weight[size_t(n * K + k)] = bf16(float(n + 1) / 4);
            blocks[size_t(n * 34 + 2 + k)] = uint8_t(n + 1);
        }
        const uint16_t scale = 0x2800;  // FP16 1/32, exactly representable.
        std::memcpy(blocks.data() + n * 34, &scale, sizeof(scale));
    }
    std::vector<float> y(size_t(T * LDY), guard);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) y[size_t(t * LDY + n)] = 0.25f;
    auto* dx = sycl::malloc_device<uint16_t>(x.size(), queue);
    auto* dw = sycl::malloc_device<uint16_t>(weight.size(), queue);
    auto* db = sycl::malloc_device<uint8_t>(blocks.size(), queue);
    auto* dy = sycl::malloc_device<float>(y.size(), queue);
    queue.memcpy(dx, x.data(), x.size() * sizeof(uint16_t));
    queue.memcpy(dw, weight.data(), weight.size() * sizeof(uint16_t));
    queue.memcpy(db, blocks.data(), blocks.size());
    queue.memcpy(dy, y.data(), y.size() * sizeof(float)).wait();
    if (native) gemm.native(dx, 8 /* GGML_TYPE_Q8_0 */, db, dy, T, N, K, LDY, beta, ldx);
    else gemm.bf16(dx, dw, dy, T, N, K, LDY, beta, ldx);
    queue.memcpy(y.data(), dy, y.size() * sizeof(float)).wait();
    bool good = true;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < LDY; ++n) {
            const float expected = n < N
                ? float(t + 1) / 2 * float(n + 1) * (native ? 1.0f : 8.0f) + beta * 0.25f
                : guard;
            const float actual = y[size_t(t * LDY + n)];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5f) {
                std::fprintf(stderr, "%s stride=%lld beta=%.0f scratch=%lld [%lld,%lld]: %.9g != %.9g\n",
                             native ? "native" : "bf16", (long long)ldx, beta, (long long)scratch_rows,
                             (long long)t, (long long)n, actual, expected);
                good = false;
            }
        }
    sycl::free(dx, queue);
    sycl::free(dw, queue);
    sycl::free(db, queue);
    sycl::free(dy, queue);
    return good;
}
}  // namespace

int main() try {
    for (int64_t stride : {K, K + 8})
        for (float beta : {0.0f, 1.0f}) {
            if (!run(false, stride, beta, N)) return 1;
            for (int64_t scratch_rows : {N, int64_t(2)})
                if (!run(true, stride, beta, scratch_rows)) return 1;
        }
    std::puts("GEMM tight/padded input, padded output, beta 0/1, and native row slices: PASS");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
