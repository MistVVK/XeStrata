#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t gib = argc > 1 ? std::atoll(argv[1]) : 8;
    const int mode = argc > 2 ? std::atoi(argv[2]) : 0;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("max_mem_alloc_size %.2f GiB\n", q.get_device().get_info<sycl::info::device::max_mem_alloc_size>() / 1073741824.0);
    const size_t bytes = gib << 30;
    auto* p = (uint8_t*) sycl::malloc_device(bytes, q);
    std::printf("malloc %zu GiB -> %p\n", gib, (void*) p);
    if (!p) return 1;
    const auto t0 = std::chrono::steady_clock::now();
    if (mode == 0) q.memset(p, 0, bytes).wait();
    else if (mode == 2) {
        auto* p2 = (uint8_t*) sycl::malloc_device(bytes, q);
        std::printf("second malloc -> %p\n", (void*) p2);
        q.memcpy(p2, p, bytes).wait();
        std::printf("memcpy of %zu GiB done\n", gib);
    } else {
        const size_t chunk = 1ull << 30;
        for (size_t o = 0; o < bytes; o += chunk) q.memset(p + o, 0, std::min(chunk, bytes - o));
        q.wait();
    }
    std::printf("memset done in %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    q.single_task([=] { p[bytes - 1] = 7; }).wait();
    uint8_t v = 0;
    q.memcpy(&v, p + bytes - 1, 1).wait();
    std::printf("last byte %d\n", v);
}
