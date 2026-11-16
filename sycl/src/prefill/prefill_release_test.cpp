// sycl/src/prefill/prefill_release_test.cpp - #5405: Prefill::release() must not destroy the device's default in-order queue.
//
// The prompt path's K/V copy queue starts as dpct's default in-order queue (borrowed). release() destroyed it, which freed
// dpct's _q_in_order, so every later get_in_order_queue() ran on a freed queue: the frees at exit, and on a small card the
// whole engine after init_prompt_paths() retried with a smaller chunk (generate.cpp: sp.reset()). This test builds a Prefill,
// resets it (release() on a fresh object) and destroys it (release() again), then:
//   1. makes 8 new dpct queues: a freed default queue's memory comes back as one of them at the same address (glibc hands a
//      just-freed block of the same size out again), which is the use-after-free itself; a live default queue cannot be reused;
//   2. copies and runs a kernel on the default queue.
// Needs one SYCL device (the Arc, or a CPU device); exits 77 (skipped) without one.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <vector>

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        (void) dpct::get_current_device();
    } catch (const std::exception& e) {
        std::printf("prefill_release_test: no SYCL device (%s): skipped\n", e.what());
        return 77;
    }
    sycl::queue* before = &dpct::get_in_order_queue();
    std::printf("device: %s; default in-order queue at %p\n", before->get_device().get_info<sycl::info::device::name>().c_str(),
                (void*) before);
    {
        strata::prefill::Prefill p;
        std::printf("Prefill made; reset()\n");
        p.reset();   // release() on a fresh prompt path, then a new one
        std::printf("reset() done; destroying it\n");
    }                // and release() again in the destructor
    std::printf("destroyed\n");
    int fail = 0;
    std::vector<sycl::queue*> fresh;
    for (int i = 0; i < 8; ++i) fresh.push_back(dpct::get_current_device().create_queue(true));
    for (sycl::queue* q : fresh)
        if (q == before) {
            std::printf("  a new queue took the default queue's address %p: release() freed the default queue\n", (void*) q);
            ++fail;
        }
    for (sycl::queue*& q : fresh) dpct::get_current_device().destroy_queue(q);
    if (fail) {   // the default queue is gone: do not use it
        std::printf("prefill_release_test: FAILED\n");
        return 1;
    }
    sycl::queue& q = dpct::get_in_order_queue();
    if (&q != before) { std::printf("  the default queue moved (%p -> %p)\n", (void*) before, (void*) &q); ++fail; }
    try {
        const size_t n = 4096;
        std::vector<int> host(n), back(n, -1);
        for (size_t i = 0; i < n; ++i) host[i] = (int) (i * 7 + 3);
        int* d = sycl::malloc_device<int>(n, q);
        q.memcpy(d, host.data(), n * sizeof(int)).wait();
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { d[i] += 1; }).wait();
        q.memcpy(back.data(), d, n * sizeof(int)).wait();
        sycl::free(d, q);
        for (size_t i = 0; i < n; ++i)
            if (back[i] != host[i] + 1) { ++fail; std::printf("  wrong value at %zu: %d\n", i, back[i]); break; }
    } catch (const std::exception& e) {
        std::printf("  the default queue failed after release(): %s\n", e.what());
        ++fail;
    }
    std::printf("prefill_release_test: %s\n", fail ? "FAILED" : "the default in-order queue is alive and works after release(): ok");
    return fail ? 1 : 0;
}
