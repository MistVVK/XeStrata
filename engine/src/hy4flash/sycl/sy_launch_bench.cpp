// sy_launch_bench.cpp - kernel launch cost on the Arc A750 / Level Zero: host submit time and GPU completion time of N dependent tiny kernels.
#include <chrono>
#include <cstdio>

#include "sy_common.h"

using namespace hy4s;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    float* a = dalloc<float>(1 << 20);
    dzero(a, 4 << 20);
    wait();
    for (int variant = 0; variant < 3; ++variant) {
        for (int n : {200, 2000}) {
            const double t0 = now();
            for (int i = 0; i < n; ++i) {
                if (variant == 0) Q().single_task([=]() { a[0] += 1.0f; });
                else if (variant == 1) Q().parallel_for(NR(8, 256), [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1.0f; });
                else Q().parallel_for(NR(512, 256), [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1.0f; });
            }
            const double t1 = now();
            wait();
            const double t2 = now();
            printf("%-34s n=%-5d submit %.2f us/kernel, total (submit+drain) %.2f us/kernel\n", variant == 0 ? "single_task" : variant == 1 ? "parallel_for 8 groups x 256" : "parallel_for 512 groups x 256", n, (t1 - t0) / n * 1e6, (t2 - t0) / n * 1e6);
        }
    }
    // dependent chain through memory is implied by the in-order queue; now the same with explicit sub-group size 32 kernels reading 4 MB
    for (int n : {1000}) {
        const double t0 = now();
        for (int i = 0; i < n; ++i)
            Q().parallel_for(NR(1024, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += a[(it.get_global_id(0) * 4 + k) & ((1 << 20) - 1)];
                a[it.get_global_id(0)] = s;
            });
        wait();
        printf("sg32 1024x256 reading 4 floats/thread: %.2f us/kernel\n", (now() - t0) / n * 1e6);
    }
#if HY4S_HAS_GRAPH
    if (dev_caps().has_graph) {
        for (int n : {100, 1000}) {
            try {
                sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
                cg.begin_recording(Q());
                for (int i = 0; i < n; ++i)
                    Q().parallel_for(NR(8, 256), [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1.0f; });
                cg.end_recording(Q());
                auto exec = cg.finalize();
                Q().ext_oneapi_graph(exec);
                wait();
                const double t0 = now();
                for (int r = 0; r < 10; ++r) Q().ext_oneapi_graph(exec);
                const double t1 = now();
                wait();
                const double t2 = now();
                printf("%-34s n=%-5d submit %.2f us/kernel, total %.2f us/kernel (10 replays)\n",
                       "command_graph 8x256", n, (t1 - t0) / (10.0 * n) * 1e6, (t2 - t0) / (10.0 * n) * 1e6);
            } catch (const sycl::exception& e) {
                printf("command_graph n=%d failed: %s\n", n, e.what());
            }
        }
    } else {
        printf("command_graph not supported on this device/runtime\n");
    }
#endif
    return 0;
}
