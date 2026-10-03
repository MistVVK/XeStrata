#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <cstdio>
namespace x = sycl::ext::oneapi::experimental;
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("aspect queue_profiling_tag %d\n", q.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag));
    auto* f = sycl::malloc_device<float>(1024, q);
    x::command_graph g{q.get_context(), q.get_device()};
    g.begin_recording(q);
    try {
        auto e0 = x::submit_profiling_tag(q);
        q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { f[i] = 1.f; });
        auto e1 = x::submit_profiling_tag(q);
        g.end_recording(q);
        auto ex = g.finalize();
        q.ext_oneapi_graph(ex).wait();
        std::printf("tags in graph: %llu\n", (unsigned long long) (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                                     e0.get_profiling_info<sycl::info::event_profiling::command_end>()));
    } catch (const std::exception& e) { std::printf("graph: %s\n", e.what()); }
}
