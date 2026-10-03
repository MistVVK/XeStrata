// Scratch probe: host USM allocation limits and CPU read bandwidth of the backing kinds.
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static sycl::device pick() {
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) return d;
    throw std::runtime_error("no L0 GPU");
}

static double read_gbps(const uint8_t* p, size_t bytes, int threads, int iters) {
    std::atomic<uint64_t> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] {
        size_t per = bytes / threads / 32 * 32, off = per * t;
        __m256i acc = _mm256_setzero_si256();
        for (int it = 0; it < iters; ++it)
            for (size_t i = 0; i < per; i += 128) {
                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*)(p + off + i)));
                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*)(p + off + i + 32)));
                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*)(p + off + i + 64)));
                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*)(p + off + i + 96)));
            }
        sink += (uint64_t)_mm256_extract_epi64(acc, 0);
    });
    for (auto& t : ts) t.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)bytes * iters / s / 1e9;
}

// Random 64-byte reads spread over the whole region: the TLB-sensitive access of a CPU expert matvec over many blobs.
static double random_gbps(const uint8_t* p, size_t bytes, int threads, size_t reads) {
    std::atomic<uint64_t> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] {
        uint64_t x = 0x9E3779B97F4A7C15ull * (t + 1), acc = 0;
        const size_t lines = bytes / 4096;
        for (size_t i = 0; i < reads; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            const uint8_t* q = p + (x % lines) * 4096;   // one 4 KB page per read, a new page nearly every time
            for (int k = 0; k < 4096; k += 64) acc += *(const uint64_t*)(q + k);
        }
        sink += acc;
    });
    for (auto& t : ts) t.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)reads * threads * 4096 / s / 1e9;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "bw";
    auto dev = pick();
    sycl::context ctx(dev);
    if (mode == "limits") {
        std::printf("global_mem_size %llu  max_mem_alloc_size %llu\n",
                    (unsigned long long)dev.get_info<sycl::info::device::global_mem_size>(),
                    (unsigned long long)dev.get_info<sycl::info::device::max_mem_alloc_size>());
        // cumulative: many 6.4 GB host allocations until 51 GB or failure
        std::vector<void*> v;
        size_t each = 6400ull << 20, total = 0;
        while (total < 51ull * 1000 * 1000 * 1000) {
            void* p = sycl::aligned_alloc_host(4096, each, ctx);
            if (!p) { std::printf("cumulative: failed after %zu allocations (%.1f GB)\n", v.size(), total / 1e9); break; }
            std::memset(p, 1, each);
            v.push_back(p); total += each;
        }
        if (total >= 51ull * 1000 * 1000 * 1000) std::printf("cumulative: %zu allocations, %.1f GB OK\n", v.size(), total / 1e9);
        // GPU can copy from the last one
        sycl::queue q(ctx, dev);
        void* d = sycl::malloc_device(each, dev, ctx);
        auto t0 = std::chrono::steady_clock::now();
        q.memcpy(d, v.back(), each).wait();
        std::printf("copy from last allocation: %.2f GB/s\n", each / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);
        sycl::free(d, ctx);
        for (void* p : v) sycl::free(p, ctx);
        return 0;
    }
    const size_t bytes = 8ull << 30;   // 8 GiB per backing
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    struct B { const char* name; uint8_t* p; };
    std::vector<B> bs;
    bs.push_back({"host USM", (uint8_t*)sycl::aligned_alloc_host(4096, bytes, ctx)});
    void* m4 = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(m4, bytes, MADV_NOHUGEPAGE);
    bs.push_back({"mmap 4K (THP off)", (uint8_t*)m4});
    void* mt = mmap(nullptr, bytes + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t* mta = (uint8_t*)(((uintptr_t)mt + (2 << 20) - 1) & ~((uintptr_t)(2 << 20) - 1));
    madvise(mta, bytes, MADV_HUGEPAGE);
    bs.push_back({"mmap THP (madvise)", mta});
    for (auto& b : bs) std::memset(b.p, 1, bytes);
    for (int round = 0; round < 3; ++round)
        for (auto& b : bs)
            std::printf("round %d  %-20s seq %6.2f GB/s   random-4K %6.2f GB/s\n", round, b.name,
                        read_gbps(b.p, bytes, threads, 2), random_gbps(b.p, bytes, threads, 200000));
    // AnonHugePages of this process
    FILE* f = std::fopen("/proc/self/smaps_rollup", "r"); char line[256];
    while (f && std::fgets(line, sizeof line, f)) if (std::strstr(line, "AnonHugePages")) std::fputs(line, stdout);
    return 0;
}
