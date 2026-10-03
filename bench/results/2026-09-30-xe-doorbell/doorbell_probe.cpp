// Doorbell visibility probe on the B70: (1) a kernel writes a host-USM flag mid-kernel and the host sees it while the
// kernel is still running; (2) the kernel then spins on another host-USM flag until the host writes it; (3) the same
// round trip repeated, with latencies; (4) the same inside a SYCL graph replay.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto* ring = sycl::malloc_host<uint32_t>(1, q);
    auto* answer = sycl::malloc_host<uint32_t>(1, q);
    auto* payload = sycl::malloc_host<float>(256, q);
    auto* dev_out = sycl::malloc_device<float>(256, q);
    auto launch = [&](sycl::queue& qq, uint32_t seq) {
        qq.single_task([=] {
            sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space> r(*ring), a(*answer);
            r.store(seq);
            uint64_t spins = 0;
            for (;;) { const uint32_t v = *reinterpret_cast<volatile uint32_t*>(answer); sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); if (v == seq || ++spins > 200000000ull) break; }
        });
        qq.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { dev_out[i] = payload[i] * 2.0f; });
    };
    int bad = 0;
    std::vector<double> ring_us, total_us;
    for (uint32_t seq = 1; seq <= 2000; ++seq) {
        *ring = 0; *answer = 0;
        const auto t0 = clk::now();
        launch(q, seq);
        auto* vr = reinterpret_cast<volatile uint32_t*>(ring);
        while (*vr != seq) {
            if (clk::now() - t0 > std::chrono::seconds(3)) { std::printf("seq %u: ring never seen\n", seq); return 1; }
        }
        const auto t1 = clk::now();
        if (seq <= 2) std::printf("seq %u ring seen\n", seq);
        for (int i = 0; i < 256; ++i) payload[i] = float(seq + i);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile uint32_t*>(answer) = seq;
        if (seq <= 2) std::printf("seq %u answered\n", seq);
        q.wait();
        if (seq <= 2) std::printf("seq %u kernel done\n", seq);
        const auto t2 = clk::now();
        std::vector<float> out(256);
        q.memcpy(out.data(), dev_out, 256 * 4).wait();
        for (int i = 0; i < 256; ++i) if (out[i] != 2.0f * float(seq + i)) { ++bad; break; }
        ring_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        total_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
    }
    std::sort(ring_us.begin(), ring_us.end());
    std::sort(total_us.begin(), total_us.end());
    std::printf("direct: 2000 round trips, %d payload errors; launch->ring seen median %.1f us p99 %.1f; answer->done "
                "median %.1f us p99 %.1f\n", bad, ring_us[1000], ring_us[1979], total_us[1000], total_us[1979]);

    // the same through a recorded graph
    sx::command_graph g(q.get_context(), q.get_device());
    uint32_t* seq_dev = sycl::malloc_host<uint32_t>(1, q);
    g.begin_recording(q);
    q.single_task([=] {
        sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space> r(*ring), a(*answer), s(*seq_dev);
        const uint32_t v = s.load();
        r.store(v);
        uint64_t spins = 0;
        for (;;) { const uint32_t w = *reinterpret_cast<volatile uint32_t*>(answer); sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); if (w == v || ++spins > 200000000ull) break; }
    });
    q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { dev_out[i] = payload[i] * 2.0f; });
    g.end_recording(q);
    auto exec = g.finalize();
    int gbad = 0;
    ring_us.clear();
    for (uint32_t seq = 1; seq <= 2000; ++seq) {
        *ring = 0; *answer = 0; *seq_dev = seq;
        const auto t0 = clk::now();
        q.ext_oneapi_graph(exec);
        auto* vr = reinterpret_cast<volatile uint32_t*>(ring);
        while (*vr != seq) {
            if (clk::now() - t0 > std::chrono::seconds(3)) { std::printf("graph seq %u: ring never seen\n", seq); return 1; }
        }
        ring_us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
        for (int i = 0; i < 256; ++i) payload[i] = float(seq * 3 + i);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile uint32_t*>(answer) = seq;
        q.wait();
        std::vector<float> out(256);
        q.memcpy(out.data(), dev_out, 256 * 4).wait();
        for (int i = 0; i < 256; ++i) if (out[i] != 2.0f * float(seq * 3 + i)) { ++gbad; break; }
    }
    std::sort(ring_us.begin(), ring_us.end());
    std::printf("graph: 2000 replays, %d payload errors; launch->ring seen median %.1f us p99 %.1f\n", gbad, ring_us[1000],
                ring_us[1979]);
    return bad || gbad;
}
