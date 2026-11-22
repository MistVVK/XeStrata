#include "bmg_expert_window_plan.hpp"
#include "strata/prefill/bmg_gate_up.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/sycl_queue.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {
bool check(sycl::queue& q, strata::prefill::Gemm& gemm, int k, int n, bool gate) {
    constexpr int capacity = 1539;
    std::mt19937 rng(35923);
    auto upload = [&](size_t length) {
        std::vector<uint16_t> host(length);
        for (auto& value : host) {
            const float v = std::ldexp(float(int(rng() % 4097) - 2048), -int(rng() % 13) - 3);
            value = sycl::bit_cast<uint16_t>(sycl::half(v));
        }
        auto* device = sycl::malloc_device<uint16_t>(length, q);
        q.memcpy(device, host.data(), length * sizeof(uint16_t)).wait_and_throw();
        return device;
    };
    auto* input = upload(size_t(capacity) * k);
    auto* weights = upload(size_t(n) * k);
    auto* split = sycl::malloc_device<float>(capacity * n, q);
    auto* merged = sycl::malloc_device<float>(capacity * n, q);
    std::vector<float> a(capacity * n), b(capacity * n);
    bool ok = true;
    const std::pair<int, int> pairs[] = {{32, 33}, {96, 97}, {128, 129}, {192, 193}, {768, 769},
                                         {16, 16}, {64, 65}, {100, 30},  {384, 384}, {769, 770}};
    for (auto [left, right] : pairs) {
        const int rows = left + right;
        gemm.f16(input, weights, merged, rows, n, k);
        gemm.f16(input, weights, split, left, n, k);
        gemm.f16(input + left * k, weights, split + left * n, right, n, k);
        const size_t bytes = size_t(rows) * n * sizeof(float);
        q.memcpy(a.data(), split, bytes);
        q.memcpy(b.data(), merged, bytes).wait_and_throw();
        const bool equal = std::memcmp(a.data(), b.data(), bytes) == 0;
        const auto allowed = strata::prefill::bmg_window::merge_products(left, right);
        const bool expected = gate ? allowed.gate : allowed.down;
        std::printf("%s %d+%d allowed=%d byte_equal=%d\n", gate ? "gate/up" : "down", left, right, expected, equal);
        ok &= equal == expected;
    }
    sycl::free(input, q);
    sycl::free(weights, q);
    sycl::free(merged, q);
    sycl::free(split, q);
    return ok;
}
} // namespace
int main() {
    auto& q = dpct::get_in_order_queue();
    if (!strata::prefill::bmg_gate_up::supported(q))
        return 77;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(&q, 0, err))
        return 1;
    const bool gate = check(q, gemm, 2560, 1280, true);
    const bool down = check(q, gemm, 640, 2560, false);
    return gate && down ? 0 : 1;
}
