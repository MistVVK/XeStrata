// sy_probe.cpp - toolchain probe for the Hy4 SYCL lane: device info, a wg_sum kernel, a sub-group-32 kernel, half conversion, a 1 GiB alloc.
#include "sy_common.h"

using namespace hy4s;

int main() {
    auto& q = Q();
    auto d = q.get_device();
    printf("device %s\n", d.get_info<sycl::info::device::name>().c_str());
    printf("global mem %.2f GiB, max alloc %.2f GiB, max wg %zu, local mem %zu KiB, EUs %u\n", d.get_info<sycl::info::device::global_mem_size>() / 1073741824.0,
           d.get_info<sycl::info::device::max_mem_alloc_size>() / 1073741824.0, d.get_info<sycl::info::device::max_work_group_size>(),
           d.get_info<sycl::info::device::local_mem_size>() / 1024, d.get_info<sycl::info::device::max_compute_units>());
    printf("fp16 %d fp64 %d\n", (int)d.has(sycl::aspect::fp16), (int)d.has(sycl::aspect::fp64));
    printf("sub-group sizes:");
    for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) printf(" %u", (unsigned)s);
    printf("\n");
    const int N = 4096;
    float* x = dalloc<float>(N);
    float* y = dalloc<float>(8);
    std::vector<float> hx(N);
    for (int i = 0; i < N; ++i) hx[i] = (float)(i % 7) - 3.0f;
    h2d(x, hx.data(), N * 4);
    // 4 groups of 1024 threads, each sums 1024 values with wg_sum
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(32, h);
        h.parallel_for(sycl::nd_range<1>(4 * 1024, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
            float v = x[it.get_global_id(0)];
            float s = wg_sum(it, v, sh.get_multi_ptr<sycl::access::decorated::no>());
            if (it.get_local_id(0) == 0) y[it.get_group(0)] = s;
        });
    });
    float hy[8] = {};
    d2h(hy, y, 16);
    wait();
    double ref[4] = {};
    for (int i = 0; i < N; ++i) ref[i / 1024] += hx[i];
    printf("wg_sum: %.1f %.1f %.1f %.1f (expect %.1f %.1f %.1f %.1f)\n", hy[0], hy[1], hy[2], hy[3], ref[0], ref[1], ref[2], ref[3]);
    // half round trip
    half hh = (half)3.14159f;
    printf("half 3.14159 -> %.5f, bits %04x\n", (float)hh, (unsigned)sycl::bit_cast<uint16_t>(hh));
    // big alloc
    try {
        uint8_t* big = sycl::malloc_device<uint8_t>((size_t)3 << 30, q);
        printf("3 GiB alloc: %s\n", big ? "ok" : "null");
        if (big) sycl::free(big, q);
        uint8_t* big2 = sycl::malloc_device<uint8_t>((size_t)5 << 30, q);
        printf("5 GiB alloc: %s\n", big2 ? "ok" : "null");
        if (big2) sycl::free(big2, q);
    } catch (const sycl::exception& e) {
        printf("alloc exception: %s\n", e.what());
    }
    return 0;
}
