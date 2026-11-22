#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <string_view>

namespace gdn_test {
// Report the median of nine paired samples. Alternate order and wait on the same queue on both sides.
template <class A, class B> void measure(sycl::queue& q, const char* name, int rows, A previous, B tiled) {
    for (int i = 0; i < 3; ++i) {
        previous();
        tiled();
    }
    q.wait_and_throw();
    auto sample = [&](auto launch) {
        const auto start = std::chrono::steady_clock::now();
        launch();
        q.wait_and_throw();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::array<double, 9> old_ms, new_ms;
    for (size_t i = 0; i < old_ms.size(); ++i) {
        if (i % 2) {
            new_ms[i] = sample(tiled);
            old_ms[i] = sample(previous);
        } else {
            old_ms[i] = sample(previous);
            new_ms[i] = sample(tiled);
        }
    }
    std::sort(old_ms.begin(), old_ms.end());
    std::sort(new_ms.begin(), new_ms.end());
    std::printf("%s rows=%d previous_ms=%.6f tiled_ms=%.6f\n", name, rows, old_ms[4], new_ms[4]);
}
} // namespace gdn_test
