#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdint>
using clk = std::chrono::steady_clock;
extern SYCL_EXTERNAL int __spirv_SDotAccSatKHR(int, int, int, int);
extern SYCL_EXTERNAL int __spirv_SDotKHR(int, int, int);
template<int MODE> inline int dot(int a, int b, int c) {
    if constexpr (MODE == 0) { for (int k = 0; k < 4; ++k) c += (int)(int8_t)(a >> 8*k) * (int)(int8_t)(b >> 8*k); return c; }
    else if constexpr (MODE == 1) { return __spirv_SDotAccSatKHR(a, b, c, 0); } else { return c + __spirv_SDotKHR(a, b, 0); }
}
int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const size_t n = 1ull << 30;   // bytes
    auto* a = sycl::malloc_device<sycl::uint4>(n / 16, q); auto* out = sycl::malloc_device<unsigned>(1 << 20, q);
    { std::vector<uint32_t> h(n / 4); uint32_t x = 12345; for (auto& v : h) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; v = x; } q.memcpy(a, h.data(), n).wait(); }
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = clk::now();
        for (int it = 0; it < 10; ++it)
            q.parallel_for(sycl::nd_range<1>(1 << 20, 256), [=](sycl::nd_item<1> i) {
                unsigned s = 0; const size_t g = i.get_global_id(0);
                for (size_t k = g; k < n / 16; k += (1 << 20)) { auto v = a[k]; s ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
                out[g] = s;
            });
        q.wait();
        std::printf("read-only kernel: %.1f GB/s\n", 10.0 * n / std::chrono::duration<double>(clk::now() - t0).count() / 1e9);
    }
    auto* ints = sycl::malloc_device<int>(1 << 20, q); q.memset(ints, 3, 4 << 20).wait();
    auto bench = [&](auto tag) {
        constexpr int M = decltype(tag)::value;
        const auto t0 = clk::now();
        q.parallel_for(sycl::nd_range<1>(1 << 20, 256), [=](sycl::nd_item<1> i) {
            int acc = 0, x = ints[i.get_global_id(0)], y = x ^ 0x5a5a5a5a;
            for (int k = 0; k < 4096; ++k) { acc = dot<M>(x, y, acc); x += 0x01010101; }
            ints[i.get_global_id(0)] = acc;
        }).wait();
        std::printf("dp4a mode %d: %.1f Gdot/s\n", M, (1 << 20) * 4096.0 / std::chrono::duration<double>(clk::now() - t0).count() / 1e9);
    };
    for (int r = 0; r < 2; ++r) { bench(std::integral_constant<int,0>{}); bench(std::integral_constant<int,1>{}); bench(std::integral_constant<int,2>{}); }
}
