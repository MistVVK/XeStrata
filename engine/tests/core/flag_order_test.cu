// Flag B behind queued copies (PR #5096): an empty plan must not release a layer whose uploads are still in flight.
// A delayed host function stands in for a slow DMA on the copy stream (it writes the "staging" word a while after it was
// queued); a kernel plays the graph (waits for flag B >= value, then reads staging).  Run on every device found.
#include "strata/core/flag_order.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using strata::core::FlagRaise;
using strata::core::flag_raise_max;
using strata::core::raise_flag_b_ordered;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
#define CK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); return 1; } } while (0)

struct SlowCopy { volatile uint32_t* staging; uint32_t word; int ms; };
static void slow_copy(void* p) {
    SlowCopy* c = (SlowCopy*) p;
    std::this_thread::sleep_for(std::chrono::milliseconds(c->ms));
    *c->staging = c->word;   // the final writer of the buffer
}

__global__ void consumer(volatile uint32_t* flag, uint32_t want, volatile uint32_t* staging, volatile uint32_t* seen) {
    while (*flag < want) {}
    *seen = *staging;        // what the graph's PCIe group would read
}

static uint32_t rd(volatile uint32_t* p) { return __atomic_load_n((uint32_t*) p, __ATOMIC_SEQ_CST); }

static int run_device(int dev) {
    CK(cudaSetDevice(dev));
    cudaStream_t copy, cs;
    CK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    volatile uint32_t *flag_h, *staging_h, *seen_h;
    CK(cudaHostAlloc((void**) &flag_h, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &staging_h, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &seen_h, 64, cudaHostAllocMapped));
    volatile uint32_t *flag_d, *staging_d, *seen_d;
    CK(cudaHostGetDevicePointer((void**) &flag_d, (void*) flag_h, 0));
    CK(cudaHostGetDevicePointer((void**) &staging_d, (void*) staging_h, 0));
    CK(cudaHostGetDevicePointer((void**) &seen_d, (void*) seen_h, 0));
    FlagRaise slots[4] = {};
    SlowCopy c0{staging_h, 0xAAu, 150};

    // 1. nonempty plan (layer 0) followed by an empty plan (layer 1) while the copy is delayed
    *flag_h = 0; *staging_h = 0; *seen_h = 0;
    CK(cudaLaunchHostFunc(copy, slow_copy, &c0));                                // layer 0's upload, in flight
    CK(raise_flag_b_ordered(true, copy, &slots[0], (uint32_t*) flag_h, 1));      // its completion flag (non-empty plan)
    CK(raise_flag_b_ordered(true, copy, &slots[1], (uint32_t*) flag_h, 2));      // layer 1: empty plan, same copy stream
    CHECK(rd(flag_h) == 0);                                                      // neither flag may rise before the delayed copy
    consumer<<<1, 1, 0, cs>>>(flag_d, 1, staging_d, seen_d);                     // the graph's layer-0 group
    CK(cudaStreamSynchronize(cs));
    CHECK(rd(seen_h) == 0xAAu);                                                  // the consumer saw the final writer's bytes
    CK(cudaStreamSynchronize(copy));
    CHECK(rd(flag_h) == 2);

    // 2. the negative control: a direct raise (the old empty-plan path) does release the consumer early
    *flag_h = 0; *staging_h = 0; *seen_h = 0;
    CK(cudaLaunchHostFunc(copy, slow_copy, &c0));
    flag_raise_max((uint32_t*) flag_h, 2);                                       // what an unordered empty plan did
    consumer<<<1, 1, 0, cs>>>(flag_d, 1, staging_d, seen_d);
    CK(cudaStreamSynchronize(cs));
    CHECK(rd(seen_h) == 0);                                                      // read before the copy landed: the bug the ordered raise removes
    CK(cudaStreamSynchronize(copy));

    // 3. nothing queued in this window: the flag rises at once
    *flag_h = 0;
    CK(raise_flag_b_ordered(false, copy, &slots[2], (uint32_t*) flag_h, 3));
    CHECK(rd(flag_h) == 3);

    // 4. the flag only rises: a late host function of an earlier layer does not lower it
    *flag_h = 5;
    CK(raise_flag_b_ordered(true, copy, &slots[3], (uint32_t*) flag_h, 4));
    CK(cudaStreamSynchronize(copy));
    CHECK(rd(flag_h) == 5);

    // 5. an idle copy stream with copies_queued still set (the window's earlier copies finished): raised through the stream
    *flag_h = 0;
    CK(raise_flag_b_ordered(true, copy, &slots[0], (uint32_t*) flag_h, 6));
    CK(cudaStreamSynchronize(copy));
    CHECK(rd(flag_h) == 6);

    // 6. through a graph: the consumer is captured and replayed twice, a fresh delayed copy each time
    {
        cudaGraph_t graph; cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
        consumer<<<1, 1, 0, cs>>>(flag_d, 7, staging_d, seen_d);
        CK(cudaStreamEndCapture(cs, &graph));
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int rep = 0; rep < 2; ++rep) {
            *flag_h = 0; *staging_h = 0; *seen_h = 0;
            SlowCopy c{staging_h, 0x55u + (uint32_t) rep, 60};
            CK(cudaLaunchHostFunc(copy, slow_copy, &c));
            CK(raise_flag_b_ordered(true, copy, &slots[1], (uint32_t*) flag_h, 7));
            CK(cudaGraphLaunch(exec, cs));
            CK(cudaStreamSynchronize(cs));
            CHECK(rd(seen_h) == 0x55u + (uint32_t) rep);
            CK(cudaStreamSynchronize(copy));
        }
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
    }
    CK(cudaStreamDestroy(copy));
    CK(cudaStreamDestroy(cs));
    CK(cudaFreeHost((void*) flag_h)); CK(cudaFreeHost((void*) staging_h)); CK(cudaFreeHost((void*) seen_h));
    return 0;
}

int main() {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n == 0) { std::puts("flag_order_test: no device, skipped"); return 0; }
    for (int d = 0; d < n; ++d) {
        const int rc = run_device(d);
        if (rc != 0) return rc;
        std::printf("flag_order_test: device %d ok\n", d);
    }
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("flag_order_test: ok");
    return 0;
}
