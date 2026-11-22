#include "strata/prefill/bmg_gate_up.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    namespace gu = strata::prefill::bmg_gate_up;
    auto& queue = dpct::get_in_order_queue();
    if (!gu::supported(queue)) return 77;
    constexpr int k = 2560, n = 1280, max_rows = 769, guard = 128;
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(&queue, 0, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto operand = [&](size_t count) {
        std::vector<uint16_t> values(count);
        for (auto& v : values) v = sycl::bit_cast<uint16_t>(sycl::half(dist(rng)));
        auto* dst = sycl::malloc_device<uint16_t>(count, queue);
        queue.memcpy(dst, values.data(), count * sizeof(uint16_t)).wait_and_throw();
        return dst;
    };
    auto* x = operand(max_rows * k);
    auto* w = operand(n * k);
    auto* projected = sycl::malloc_device<float>(max_rows * n, queue);
    const size_t size = max_rows * (n / 2) + 2 * guard;
    auto* hidden = sycl::malloc_device<uint16_t>(size, queue);
    auto* reference = sycl::malloc_device<uint16_t>(size, queue);
    std::vector<uint16_t> got(size), expected(size);
    bool ok = true;
    std::vector<int> shapes{0, 1, 7, 16, 17, 31, 32, 33, 128, 129, 160, 192, 193, 224, 255, 256, 257, 384, 385, 416, 447, 448, 449, 512, 513, 640, 767, 768, 769};
    for (int rows = 34; rows < 128; ++rows) shapes.push_back(rows);
    for (int rows : shapes) {
        queue.fill(hidden, uint16_t(0x7d55), size);
        queue.fill(reference, uint16_t(0x7d55), size);
        bool used = gu::run(queue, x, w, hidden + guard, rows);
        const bool wanted = rows >= 33 && rows <= 768;
        if (used) {
            gemm.f16(x, w, projected, rows, n, k);
            strata::prefill::swiglu_interleaved(projected, reference + guard, rows, &queue);
        }
        queue.memcpy(got.data(), hidden, size * sizeof(uint16_t));
        queue.memcpy(expected.data(), reference, size * sizeof(uint16_t));
        queue.wait_and_throw();
        if (used != wanted || got != expected) {
            std::fprintf(stderr, "Battlemage gate/up output or route differs at %d rows\n", rows);
            ok = false;
        }
    }
    sycl::free(reference, queue);
    sycl::free(hidden, queue);
    sycl::free(projected, queue);
    sycl::free(w, queue);
    sycl::free(x, queue);
    return ok ? 0 : 1;
}
