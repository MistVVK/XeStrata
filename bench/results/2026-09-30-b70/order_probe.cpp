// Scratch probe: register an untouched mapping, fill it afterwards, and check what the device receives.
#include <sycl/sycl.hpp>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
namespace exp = sycl::ext::oneapi::experimental;
int main() {
    const size_t bytes = 40ull << 30, chunk = 707788800;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) dev = d;
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order{});
    uint8_t* p = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(p, bytes, MADV_HUGEPAGE);
    auto t0 = std::chrono::steady_clock::now();
    exp::prepare_for_device_copy(p, bytes, ctx);
    std::printf("register untouched 40 GiB: %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (size_t i = 0; i < bytes; i += 4096) p[i] = (uint8_t)(i >> 12);   // write after registration
    std::memset(p + bytes - chunk, 0xA5, chunk);
    uint8_t* d = (uint8_t*)sycl::malloc_device(chunk, dev, ctx);
    std::vector<uint8_t> back(chunk);
    size_t bad = 0;
    for (size_t off : {size_t(0), bytes / 2 / 4096 * 4096, bytes - chunk}) {
        q.memcpy(d, p + off, chunk).wait();
        q.memcpy(back.data(), d, chunk).wait();
        bad += std::memcmp(back.data(), p + off, chunk) != 0;
    }
    t0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off + chunk <= bytes; off += chunk) q.memcpy(d, p + off, chunk);
    q.wait();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("content mismatches: %zu of 3 ranges; H2D %.2f GB/s\n", bad, (double)(bytes / chunk * chunk) / s / 1e9);
    FILE* f = std::fopen("/proc/self/smaps_rollup", "r"); char line[256];
    while (f && std::fgets(line, sizeof line, f)) if (std::strstr(line, "AnonHugePages")) std::fputs(line, stdout);
    exp::release_from_device_copy(p, ctx);
    return bad != 0;
}
