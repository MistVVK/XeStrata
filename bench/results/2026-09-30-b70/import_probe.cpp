// Scratch probe: register one large ordinary mapping for device copies and measure H2D from it.
#include <sycl/sycl.hpp>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstring>
namespace exp = sycl::ext::oneapi::experimental;
int main(int argc, char** argv) {
    const size_t bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 51ull) * 1000 * 1000 * 1000 / 4096 * 4096;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) dev = d;
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order{});
    uint8_t* p = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { std::puts("mmap failed"); return 1; }
    madvise(p, bytes, MADV_HUGEPAGE);
    std::memset(p, 1, bytes);
    const size_t chunk = 707788800;
    void* d = sycl::malloc_device(chunk, dev, ctx);
    auto rate = [&](const char* tag) {
        q.memcpy(d, p + bytes - chunk, chunk).wait();
        auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off + chunk <= bytes; off += chunk) q.memcpy(d, p + off, chunk);
        q.wait();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-22s %.2f GB/s over %.1f GB\n", tag, (double)(bytes / chunk * chunk) / s / 1e9, bytes / 1e9);
    };
    rate("unregistered mmap");
    auto t0 = std::chrono::steady_clock::now();
    try { exp::prepare_for_device_copy(p, bytes, ctx); }
    catch (const std::exception& e) { std::printf("prepare_for_device_copy failed: %s\n", e.what()); return 1; }
    std::printf("registration took %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    rate("registered mmap");
    rate("registered mmap");
    FILE* f = std::fopen("/proc/self/status", "r"); char line[256];
    while (f && std::fgets(line, sizeof line, f)) if (!std::strncmp(line, "VmLck", 5) || !std::strncmp(line, "VmPin", 5)) std::fputs(line, stdout);
    exp::release_from_device_copy(p, ctx);
    sycl::free(d, ctx);
    return 0;
}
