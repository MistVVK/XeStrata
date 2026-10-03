#include <sycl/sycl.hpp>
#include <cstdio>
namespace x = sycl::ext::oneapi::experimental;
int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto d = q.get_device();
    std::printf("%s\n", d.get_info<sycl::info::device::name>().c_str());
    std::printf("clock_sub_group %d clock_work_group %d clock_device %d\n", d.has(sycl::aspect::ext_oneapi_clock_sub_group),
                d.has(sycl::aspect::ext_oneapi_clock_work_group), d.has(sycl::aspect::ext_oneapi_clock_device));
    std::printf("profiling timer resolution %zu ns\n", d.get_info<sycl::info::device::profiling_timer_resolution>());
    auto* buf = sycl::malloc_shared<unsigned long long>(4, q);
    auto e0 = q.single_task([=] { buf[0] = 1; });
    auto e1 = q.single_task([=] { buf[1] = 2; });
    q.wait();
    auto s0 = e0.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto s1 = e1.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::printf("event profiling: %llu ns across two tasks\n", (unsigned long long) (s1 - s0));
    if (d.has(sycl::aspect::ext_oneapi_clock_sub_group)) {
        q.single_task([=] { buf[2] = x::clock<x::clock_scope::sub_group>(); }).wait();
        std::printf("sub_group clock %llu\n", buf[2]);
    }
}
