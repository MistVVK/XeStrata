// Which host->GPU signalling does a spinning Xe kernel observe?  mode:
//   0 host USM flag, atomic_ref system seq_cst load
//   1 host USM flag, volatile load + system acquire fence
//   2 shared USM flag, atomic_ref system load
//   3 device USM flag, written by a host->device memcpy on a second in-order queue
// Each mode: 20 round trips; the kernel gives up after ~2 s of spinning (clock-free bound on iterations).
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int mode = argc > 1 ? std::atoi(argv[1]) : 0;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue t{q.get_context(), q.get_device(), sycl::property::queue::in_order{}};
    auto* ring = sycl::malloc_host<uint32_t>(1, q);
    uint32_t* flag = mode == 2 ? sycl::malloc_shared<uint32_t>(1, q)
                   : mode == 3 ? sycl::malloc_device<uint32_t>(1, q) : sycl::malloc_host<uint32_t>(1, q);
    auto* gaveup = sycl::malloc_host<uint32_t>(1, q);
    auto* src = sycl::malloc_host<uint32_t>(1, q);
    int ok = 0;
    double worst = 0;
    for (uint32_t seq = 1; seq <= 2000; ++seq) {
        *ring = 0; *gaveup = 0;
        if (mode == 3) q.memset(flag, 0, 4).wait(); else *flag = 0;
        q.single_task([=] {
            sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space> r(*ring), f(*flag);
            r.store(seq);
            uint64_t spins = 0;
            for (;;) {
                uint32_t v;
                if (mode == 1) {
                    v = *reinterpret_cast<volatile uint32_t*>(flag);
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                } else {
                    v = f.load();
                }
                if (v == seq) break;
                if (++spins > 200000000ull) { *gaveup = 1; break; }
            }
        });
        const auto t0 = clk::now();
        while (*reinterpret_cast<volatile uint32_t*>(ring) != seq)
            if (clk::now() - t0 > std::chrono::seconds(5)) { std::printf("mode %d: ring never seen\n", mode); return 2; }
        const auto t1 = clk::now();
        if (mode == 3) { *src = seq; t.memcpy(flag, src, 4); }
        else { std::atomic_thread_fence(std::memory_order_seq_cst); *reinterpret_cast<volatile uint32_t*>(flag) = seq; }
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(clk::now() - t1).count();
        if (!*gaveup) { ++ok; worst = std::max(worst, us); }
        if (seq == 1) std::printf("mode %d seq 1: %s after %.1f us\n", mode, *gaveup ? "GAVE UP" : "seen", us);
        t.wait();
    }
    std::printf("mode %d: %d/2000 round trips observed, worst answer->done %.1f us\n", mode, ok, worst);
    return ok == 2000 ? 0 : 1;
}
