// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// event_record_probe: what marking an in-order queue's end costs the next wait, the mark a timestamp
// (submit_profiling_tag, what gpu::event_record always submitted) or a barrier (ext_oneapi_submit_barrier).  N rounds of
// a single-task kernel and a mark, the marks' events held until after the wait (as the prompt path holds its ring's)
// or dropped at once; then the time to submit them and to wait for the queue (a barrier's wait, as Runtime::finish).
// Medians of 5.  Argument: N (384, the prompt path's streamed ring on the B70).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static void run(int n) {
    sycl::device d{sycl::gpu_selector_v};
    sycl::queue q{d, sycl::property::queue::in_order{}};
    int* p = sycl::malloc_device<int>(16, q);
    std::vector<sycl::event> held(n);
    std::printf("%s, %s, N=%d\n", d.get_info<sycl::info::device::name>().c_str(),
                d.get_platform().get_info<sycl::info::platform::name>().c_str(), n);
    for (int tag = 1; tag >= 0; --tag)
        for (int hold = 1; hold >= 0; --hold) {
            std::vector<double> sub, wait;
            for (int r = 0; r < 6; ++r) {
                q.wait();
                const auto t0 = Clock::now();
                for (int k = 0; k < n; ++k) {
                    q.single_task([=] { p[0] += 1; });
                    sycl::event e = tag ? sycl::ext::oneapi::experimental::submit_profiling_tag(q)
                                        : q.ext_oneapi_submit_barrier();
                    if (hold) held[k] = e;
                }
                const auto t1 = Clock::now();
                q.ext_oneapi_submit_barrier().wait();
                const auto t2 = Clock::now();
                held.assign(n, sycl::event{});
                if (r == 0) continue;   // the first round builds the kernel
                sub.push_back(ms(t0, t1));
                wait.push_back(ms(t1, t2));
            }
            std::printf("%-9s %-14s submit %6.2f ms, wait %6.2f ms\n", tag ? "timestamp" : "barrier",
                        hold ? "events held" : "events dropped", med(sub), med(wait));
        }
}
int main(int argc, char** argv) {
    try {
        run(argc > 1 ? (int) std::strtol(argv[1], nullptr, 10) : 384);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
