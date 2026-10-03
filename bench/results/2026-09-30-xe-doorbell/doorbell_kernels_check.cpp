// The library's doorbell kernels (elementwise.hpp) through the protocol the session uses, directly and replayed from
// a SYCL graph: doorbell_publish copies x/ids/weights to host USM and rings; the host polls the ring, checks the
// payload, writes an answer vector and the flag; doorbell_wait spins; copy_from_mapped and copy_i32_from_mapped bring
// the answer back.  Build as the other probes (see ../2026-09-30-xe-native/README.md) without the ggml libraries.
#include "strata/core/runtime.hpp"
#include "strata/kernels/elementwise.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

int main() {
    auto& rt = strata::core::Runtime::get();
    sycl::queue* st = rt.create_stream();
    const int n = 2560, kk = 10;
    auto* x = sycl::malloc_device<float>(n, *st);
    auto* ids = sycl::malloc_device<int32_t>(kk, *st);
    auto* w = sycl::malloc_device<float>(kk, *st);
    auto* back = sycl::malloc_device<float>(n, *st);
    auto* back_i = sycl::malloc_device<int32_t>(16, *st);
    auto* hx = sycl::malloc_host<float>(n, *st);
    auto* hids = sycl::malloc_host<int32_t>(kk, *st);
    auto* hw = sycl::malloc_host<float>(kk, *st);
    auto* hseq = sycl::malloc_host<uint32_t>(1, *st);
    auto* hflag = sycl::malloc_host<uint32_t>(1, *st);
    auto* hans = sycl::malloc_host<float>(n, *st);
    auto* hans_i = sycl::malloc_host<int32_t>(16, *st);
    *hseq = 0;
    *hflag = 0;
    auto body = [&] {
        k::doorbell_publish(x, ids, w, n, kk, hx, hids, hw, hseq, st);
        k::doorbell_wait(hflag, hseq, st);
        k::copy_from_mapped(back, hans, n, st);
        k::copy_i32_from_mapped(back_i, hans_i, 16, st);
    };
    sx::command_graph g(st->get_context(), st->get_device());
    g.begin_recording(*st);
    body();
    g.end_recording(*st);
    auto exec = g.finalize();
    int bad = 0;
    for (int it = 0; it < 400; ++it) {
        const bool graph = it % 2;
        std::vector<float> xs(n), ws(kk);
        std::vector<int32_t> is(kk);
        for (int i = 0; i < n; ++i) xs[i] = float(it * 7 + i);
        for (int i = 0; i < kk; ++i) { is[i] = it + i; ws[i] = 0.5f * (it + i); }
        st->memcpy(x, xs.data(), n * 4);
        st->memcpy(ids, is.data(), kk * 4);
        st->memcpy(w, ws.data(), kk * 4).wait();
        const uint32_t want = *hseq + 1;
        if (graph) st->ext_oneapi_graph(exec); else body();
        const auto t0 = std::chrono::steady_clock::now();
        while (*reinterpret_cast<volatile uint32_t*>(hseq) != want)
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) { std::printf("ring %u never seen\n", want); return 2; }
        bool payload = true;
        for (int i = 0; i < n; ++i) payload &= hx[i] == xs[i];
        for (int i = 0; i < kk; ++i) payload &= hids[i] == is[i] && hw[i] == ws[i];
        for (int i = 0; i < n; ++i) hans[i] = -float(it + i);
        for (int i = 0; i < 16; ++i) hans_i[i] = it * 100 + i;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile uint32_t*>(hflag) = want;
        rt.finish(*st);
        std::vector<float> got(n);
        std::vector<int32_t> got_i(16);
        st->memcpy(got.data(), back, n * 4);
        st->memcpy(got_i.data(), back_i, 16 * 4).wait();
        bool answer = true;
        for (int i = 0; i < n; ++i) answer &= got[i] == -float(it + i);
        for (int i = 0; i < 16; ++i) answer &= got_i[i] == it * 100 + i;
        if (!payload || !answer) {
            ++bad;
            std::printf("round %d (%s): payload %s, answer %s\n", it, graph ? "graph" : "direct", payload ? "ok" : "WRONG",
                        answer ? "ok" : "WRONG");
        }
    }
    std::printf("doorbell_kernels_check: 400 rounds (200 direct, 200 graph replays), %d failures\n", bad);
    return bad ? 1 : 0;
}
