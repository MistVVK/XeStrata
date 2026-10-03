// Global timestamps written by Level Zero commands, eagerly and inside a recorded SYCL graph.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <chrono>
#include <cstdio>
#include <variant>
namespace x = sycl::ext::oneapi::experimental;

static void stamp(sycl::queue& q, uint64_t* dst) {
    auto nq = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
    ze_command_list_handle_t imm = std::holds_alternative<ze_command_list_handle_t>(nq) ? std::get<ze_command_list_handle_t>(nq) : nullptr;
    q.submit([&](sycl::handler& h) {
        h.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
            ze_command_list_handle_t cl;
            if (ih.ext_codeplay_has_graph()) {
                cl = ih.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_level_zero>();
            } else {
                if (!imm) { std::printf("not an immediate list\n"); return; }
                cl = imm;
            }
            ze_result_t r = zeCommandListAppendWriteGlobalTimestamp(cl, dst, nullptr, 0, nullptr);
            if (r != ZE_RESULT_SUCCESS) std::printf("append: 0x%x\n", (unsigned) r);
        });
    });
}
static void busy(sycl::queue& q, float* p, int iters) {
    q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
        float v = p[i];
        for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1e-7f;
        p[i] = v;
    });
}
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    auto d = q.get_device();
    auto zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
    ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2};
    zeDeviceGetProperties(zd, &p);
    std::printf("timerResolution %llu (cycles/s for DEVICE_PROPERTIES_1_2), timestampValidBits %u, kernelTimestampValidBits %u\n",
                (unsigned long long) p.timerResolution, p.timestampValidBits, p.kernelTimestampValidBits);
    const double ns_per_tick = 1e9 / (double) p.timerResolution;
    auto* ts = sycl::malloc_device<uint64_t>(8, q);
    auto* f = sycl::malloc_device<float>(1024, q);
    q.memset(ts, 0, 64).wait();
    uint64_t h[8];
    // eager
    auto t0 = std::chrono::steady_clock::now();
    stamp(q, ts + 0); busy(q, f, 2000000); stamp(q, ts + 1);
    q.wait();
    double wall = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    q.memcpy(h, ts, 64).wait();
    std::printf("eager: %llu %llu -> %.3f ms (host wall %.3f ms)\n", (unsigned long long) h[0], (unsigned long long) h[1],
                (double) (h[1] - h[0]) * ns_per_tick / 1e6, wall / 1e6);
    // graph
    x::command_graph g{q.get_context(), d};
    g.begin_recording(q);
    stamp(q, ts + 2); busy(q, f, 2000000); stamp(q, ts + 3);
    g.end_recording(q);
    auto e = g.finalize();
    for (int rep = 0; rep < 2; ++rep) {
        t0 = std::chrono::steady_clock::now();
        q.ext_oneapi_graph(e);
        q.wait();
        wall = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        q.memcpy(h, ts, 64).wait();
        std::printf("graph replay %d: %llu %llu -> %.3f ms (host wall %.3f ms)\n", rep, (unsigned long long) h[2],
                    (unsigned long long) h[3], (double) (h[3] - h[2]) * ns_per_tick / 1e6, wall / 1e6);
    }
}
