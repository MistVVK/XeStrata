// include/strata/uc_probe.hpp - test-only in-process kernel timer for the "uncached host memory slows every later
// kernel" question on Arc Alchemist (STRATA_UC_PROBE=1; does nothing when unset).
//
// uc_probe("tag") times four small kernels back to back on the engine's own queue and prints one line to stderr:
//   head   the engine's native_quantize_q8_1 + native_mmvq (Q8_0, 2048 x 16384, 7 columns), a memory-bound GEMV
//   q6k    native_mmvq Q6_K 2560 x 10240, 4 columns (a GDN projection of the decode window)
//   fma    256 work-groups of 128 items, 4096 dependent FMAs each (ALU bound, no memory)
//   tiny   one work-group of 64 items, 400 dependent adds (launch/latency bound)
// STRATA_UC_POISON=fp64|uc_read|uc_write|cached_read|shared_write runs one deliberately poisoning kernel
// right after the first probe, so the effect can be reproduced inside the engine process.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata {
namespace ucprobe {
inline bool on() { static const bool v = [] { const char* e = std::getenv("STRATA_UC_PROBE"); return e && e[0] && e[0] != '0'; }(); return v; }

struct Bufs {
    sycl::queue* q;
    uint8_t* w8; uint8_t* w6; float* x; void* xq; float* y; float* fo; int* ti;
    explicit Bufs(sycl::queue& qq) : q(&qq) {
        const size_t b8 = strata::kernels::native_mmvq_weight_bytes(8, 2048, 16384);
        const size_t b6 = strata::kernels::native_mmvq_weight_bytes(14, 2560, 10240);
        w8 = (uint8_t*) sycl::malloc_device(b8, qq); w6 = (uint8_t*) sycl::malloc_device(b6, qq);
        x = sycl::malloc_device<float>((size_t) 7 * 2560, qq);
        xq = sycl::malloc_device(strata::kernels::native_q8_1_bytes(2560, 7) + 256, qq);
        y = sycl::malloc_device<float>((size_t) 7 * 16384, qq);
        fo = sycl::malloc_device<float>(256 * 128, qq); ti = sycl::malloc_device<int>(64, qq);
        qq.memset(w8, 0, b8); qq.memset(w6, 0, b6); qq.memset(x, 0, (size_t) 7 * 2560 * 4); qq.memset(y, 0, (size_t) 7 * 16384 * 4);
        qq.memset(fo, 0, 256 * 128 * 4); qq.memset(ti, 0, 256); qq.wait();
    }
    ~Bufs() { q->wait(); sycl::free(w8, *q); sycl::free(w6, *q); sycl::free(x, *q); sycl::free(xq, *q); sycl::free(y, *q); sycl::free(fo, *q); sycl::free(ti, *q); }
};
template <class F> double time_us(sycl::queue& q, F&& f, int n) {
    for (int i = 0; i < 5; ++i) f();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    q.wait();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}
inline void poison(sycl::queue& q, const std::string& how) {
    if (how == "fp64") {
        double* d = sycl::malloc_device<double>(64, q);
        q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { d[i] = sycl::sqrt((double) i[0] + 0.5) * 1.0000001; }).wait();
        sycl::free(d, q);
    } else if (how == "uc_read" || how == "uc_write") {
        int* hp = (int*) strata::host_malloc_polled(4096, q);
        int* dv = sycl::malloc_device<int>(64, q);
        std::memset(hp, 0, 4096);
        if (how == "uc_read") q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { dv[i] = hp[i]; }).wait();
        else q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { hp[i] = (int) i[0]; }).wait();
        sycl::free(dv, q);   // hp is leaked on purpose: a test-only probe
    } else if (how == "cached_read") {
        int* hc = (int*) sycl::malloc_host(4096, q);
        int* dv = sycl::malloc_device<int>(64, q);
        std::memset(hc, 0, 4096);
        q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { dv[i] = hc[i]; }).wait();
        sycl::free(dv, q);
    } else if (how == "shared_write") {
        int* sh = (int*) sycl::malloc_shared(4096, q);
        q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { sh[i] = (int) i[0]; }).wait();
    }
}
}  // namespace ucprobe

inline void uc_probe(const char* tag) {
    if (!ucprobe::on()) return;
    static int calls = 0;
    sycl::queue& q = dpct::get_in_order_queue();
    {
        ucprobe::Bufs b(q);
        namespace k = strata::kernels;
        double head = 1e30, q6 = 1e30, fma = 1e30, tiny = 1e30;
        for (int rep = 0; rep < 3; ++rep) {   // the first pass warms the GPU clock; the line shows the best of three
        const double head_ = ucprobe::time_us(q, [&] {
            k::native_quantize_q8_1(b.x, b.xq, 2048, 7, &q);
            k::native_mmvq(8, b.w8, b.xq, b.y, 2048, 16384, 7, &q);
        }, 40);
        const double q6_ = ucprobe::time_us(q, [&] {
            k::native_quantize_q8_1(b.x, b.xq, 2560, 4, &q);
            k::native_mmvq(14, b.w6, b.xq, b.y, 2560, 10240, 4, &q);
        }, 80);
        float* fo = b.fo;
        const double fma_ = ucprobe::time_us(q, [&] {
            q.parallel_for(sycl::nd_range<1>(256 * 128, 128), [=](sycl::nd_item<1> it) {
                float a = 1.0f + 1e-3f * (float) it.get_global_id(0);
                for (int i = 0; i < 4096; ++i) a = a * 1.0000001f + 1e-7f;
                fo[it.get_global_id(0)] = a;
            });
        }, 40);
        int* ti = b.ti;
        const double tiny_ = ucprobe::time_us(q, [&] {
            q.parallel_for(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> it) {
                int a = (int) it.get_global_id(0);
                for (int i = 0; i < 400; ++i) a = a * 3 + i;
                ti[it.get_global_id(0)] = a;
            });
        }, 200);
        head = std::min(head, head_); q6 = std::min(q6, q6_); fma = std::min(fma, fma_); tiny = std::min(tiny, tiny_);
        }
        double fma_g = -1, tiny_g = -1, empty_w = -1;
        float* fo = b.fo; int* ti = b.ti;
        try {   // GPU stamps of the same two kernels (needs a profiling queue: STRATA_KPROF=1), and the host cost of an empty launch
            auto gpu_us = [&](auto&& launch, int n) {
                std::vector<sycl::event> ev; ev.reserve(n);
                for (int i = 0; i < 5; ++i) launch();
                q.wait();
                for (int i = 0; i < n; ++i) ev.push_back(launch());
                q.wait();
                double g = 0;
                for (auto& e : ev) g += (double) (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>());
                return g / 1e3 / n;
            };
            fma_g = gpu_us([&] { return q.parallel_for(sycl::nd_range<1>(256 * 128, 128), [=](sycl::nd_item<1> it) {
                float a = 1.0f + 1e-3f * (float) it.get_global_id(0);
                for (int i = 0; i < 4096; ++i) a = a * 1.0000001f + 1e-7f;
                fo[it.get_global_id(0)] = a; }); }, 40);
            tiny_g = gpu_us([&] { return q.parallel_for(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> it) {
                int a = (int) it.get_global_id(0);
                for (int i = 0; i < 400; ++i) a = a * 3 + i;
                ti[it.get_global_id(0)] = a; }); }, 200);
        } catch (...) {}
        {
            for (int i = 0; i < 5; ++i) q.single_task([] {});
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) q.single_task([] {});
            const double submit = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
            q.wait();
            empty_w = submit;
        }
        {
            std::lock_guard<std::mutex> lk(strata::device_alloc_mutex());
            size_t tot = 0; for (const auto& kv : strata::device_alloc_map()) tot += kv.second;
            std::fprintf(stderr, "UCPROBE %-14s   registered device allocations: %zu (%.2f GiB)\n", tag, strata::device_alloc_map().size(), (double) tot / 1073741824.0);
        }
        std::fprintf(stderr, "UCPROBE %-14s   gpu-stamps: fma %7.1f us  tiny %7.1f us   host cost of an empty launch %6.1f us\n", tag, fma_g, tiny_g, empty_w);
        std::fprintf(stderr, "UCPROBE %-14s head %8.1f us  q6k %8.1f us  fma %8.1f us  tiny %7.1f us\n", tag, head, q6, fma, tiny);
    }
    if (calls++ == 0) {
        if (const char* p = std::getenv("STRATA_UC_POISON")) {
            if (p[0]) { ucprobe::poison(q, p); std::fprintf(stderr, "UCPROBE poisoned with %s\n", p); uc_probe("after-poison"); }
        }
    }
}
}  // namespace strata
