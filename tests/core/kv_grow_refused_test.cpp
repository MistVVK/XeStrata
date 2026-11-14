// The elastic K/V when the expert cache cannot give it slots (kv_grow_rules.hpp): the pools of a QSA state made elastic
// at init map only their first cells; the K/V then grows from new VRAM (the take that returns 0, as generate.cpp's
// kvg_ensure does when the cache has nothing to give) to the cells a long prompt needs and on to the whole context, and
// every cell it maps is writable and reads back.  GPU with virtual memory needed; skips with 77 otherwise.
#include "strata/core/geometry.hpp"
#include "strata/core/kv_grow_rules.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/vmm.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// cells [0, cells) of one [page][kv_head][page_size][w] pool are its first pages, contiguous
size_t pool_bytes(int64_t cells, int64_t page_size, int64_t heads, int64_t w) {
    return (size_t) ((cells + page_size - 1) / page_size) * (size_t) (heads * page_size * w);
}

bool fill_and_check(const QsaState& st, const strata::kernels::QsaShapes& s, int64_t cells, uint8_t v) {
    const int64_t scb = (s.head_dim / 64) * 2;
    const std::pair<void*, int64_t> arrays[] = {{st.k_q, s.head_dim}, {st.v_q, s.head_dim}, {st.k_scale, scb}, {st.v_scale, scb}};
    for (const auto& [p, w] : arrays) {
        const size_t n = pool_bytes(cells, s.page_size, s.n_head_kv, w);
        if (cudaMemset(p, v, n) != cudaSuccess) return false;
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        std::vector<uint8_t> h(4096);   // the last bytes: the far end of the mapped cells
        if (cudaMemcpy(h.data(), (const uint8_t*) p + (n - h.size()), h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        for (const uint8_t b : h)
            if (b != v) return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (cudaFree(nullptr) != cudaSuccess || !vmm_available()) {
        std::printf("kv_grow_refused_test: no CUDA device with virtual memory - skipped\n");
        return 77;
    }
    const ModelGeometry g;   // the model's QSA shapes
    const strata::kernels::QsaShapes s = qsa_shapes(g);
    const int64_t max_cells = 65536, init = 4096;
    qsa_set_kv_int8(true);
    qsa_set_kv_elastic(true, init);
    CHECK(qsa_kv_elastic());
    void* base = nullptr;
    CHECK(cudaMalloc(&base, qsa_state_bytes(g, max_cells)) == cudaSuccess);
    QsaState st;
    CHECK(qsa_state_init(g, max_cells, base, st) > 0);
    CHECK(st.kv_elastic >= 0 && st.k_q != nullptr && st.v_scale != nullptr);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const int64_t cells0 = qsa_kv_elastic_cells();
    std::printf("kv_grow_refused_test: first mapping %lld of %lld cells (%.1f of %.1f MiB)\n", (long long) cells0,
                (long long) max_cells, (double) qsa_kv_elastic_mapped_bytes() / 1048576.0,
                (double) qsa_kv_elastic_full_bytes() / 1048576.0);
    CHECK(cells0 >= init && cells0 < max_cells);
    CHECK(fill_and_check(st, s, cells0, 0x11));

    // the refused state of BLOCK 6: --expert-cache 100 (128 slots, the floor), pools elastic
    const KvGrowStart r = kv_grow_start(qsa_kv_elastic(), true, true, true, 128, 128);
    CHECK(r.on && !r.from_cache);
    // a 20,000-token prompt (kvg_ensure(n + 256)): its cells are not mapped yet, then grown from new VRAM
    const int64_t want = 20000 + 256, step = 8192;
    const int64_t target = kv_grow_target(want, step, max_cells);
    CHECK(target >= want);
    CHECK(qsa_kv_elastic_need(target) > 0);
    int64_t fresh = 0;
    CHECK(qsa_kv_elastic_grow(target, [&] { ++fresh; return (VmmChunk) 0; }));
    CHECK(fresh > 0 && qsa_kv_elastic_need(target) == 0);
    const int64_t cells1 = qsa_kv_elastic_cells();
    CHECK(cells1 >= target);
    CHECK(fill_and_check(st, s, cells1, 0x22));
    // --full-card (not in ctest: it takes every free byte of the card for a moment): with no new VRAM either, the grow
    // reports false (generate.cpp's callers then end the request with their "no VRAM is left" line), the cells mapped
    // so far stay mapped and usable, the device has no sticky error, and the grow succeeds once memory is free again
    bool full_card = false;
    for (int i = 1; i < argc; ++i) full_card = full_card || std::string(argv[i]) == "--full-card";
    if (full_card) {
        std::vector<void*> hog;
        for (const size_t block : {(size_t) 256 << 20, (size_t) 2 << 20}) {
            void* p = nullptr;
            while (cudaMalloc(&p, block) == cudaSuccess) hog.push_back(p);
            (void) cudaGetLastError();
        }
        size_t free_b = 0, total_b = 0;
        (void) cudaMemGetInfo(&free_b, &total_b);
        std::printf("kv_grow_refused_test: full card: %zu blocks taken, %.1f MiB free\n", hog.size(), (double) free_b / 1048576.0);
        const bool grew = qsa_kv_elastic_grow(max_cells, [] { return (VmmChunk) 0; });
        (void) cudaGetLastError();
        std::printf("kv_grow_refused_test: full card: the grow to %lld cells returned %s; %lld cells mapped\n",
                    (long long) max_cells, grew ? "true" : "false", (long long) qsa_kv_elastic_cells());
        CHECK(!grew);
        CHECK(qsa_kv_elastic_cells() >= cells1);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(fill_and_check(st, s, cells1, 0x44));
        for (void* p : hog) (void)cudaFree(p);
    }
    // the whole context: every cell of --max-context can be mapped the same way
    CHECK(qsa_kv_elastic_grow(max_cells, [] { return (VmmChunk) 0; }));
    CHECK(qsa_kv_elastic_cells() >= max_cells && qsa_kv_elastic_need(max_cells) == 0);
    CHECK(fill_and_check(st, s, max_cells, 0x33));
    std::printf("kv_grow_refused_test: grown from new VRAM to %lld then %lld cells (%lld new chunks for the first grow)\n",
                (long long) cells1, (long long) qsa_kv_elastic_cells(), (long long) fresh);
    CHECK(cudaGetLastError() == cudaSuccess);
    if (failed) {
        std::fprintf(stderr, "kv_grow_refused_test: %d failed\n", failed);
        return 1;
    }
    std::printf("kv_grow_refused_test: ok\n");
    return 0;
}
