// The prompt path's K/V append under KV streaming (strata/core/kv_host_dma.hpp): CUDA copies the chunk's new cells to
// the host copy by DMA by default, HIP keeps the kernel's own stores unless asked, STRATA_KV_HOST_DMA=1 / 0 forces
// either way, and a layer without a staging pool always keeps the kernel's stores.
#include "strata/core/kv_host_dma.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    // unset: the backend's default (CUDA true, HIP false)
    CHECK(kv_host_dma_on(nullptr, true));
    CHECK(!kv_host_dma_on(nullptr, false));
    // "1" / "0" force it on both backends
    CHECK(kv_host_dma_on("1", false));
    CHECK(kv_host_dma_on("1", true));
    CHECK(!kv_host_dma_on("0", true));
    CHECK(!kv_host_dma_on("0", false));
    // HIP as before this header: only a value that starts with '1' turns it on
    CHECK(kv_host_dma_on("1yes", false));
    CHECK(!kv_host_dma_on("", false));
    CHECK(!kv_host_dma_on("yes", false));
    CHECK(!kv_host_dma_on("2", false));
    // CUDA: anything but a leading '0' keeps the default on
    CHECK(kv_host_dma_on("", true));
    CHECK(kv_host_dma_on("yes", true));
    CHECK(!kv_host_dma_on("0n", true));
    // one append: the switch, a streamed (staged) layer, and its staging pool; any one missing -> the kernel's stores
    CHECK(kv_host_dma_use(true, true, true));
    for (int on = 0; on < 2; ++on)
        for (int staged = 0; staged < 2; ++staged)
            for (int pool = 0; pool < 2; ++pool)
                if (!(on && staged && pool)) CHECK(!kv_host_dma_use(on != 0, staged != 0, pool != 0));
    if (failed) {
        std::fprintf(stderr, "kv_host_dma_test: %d failed\n", failed);
        return 1;
    }
    std::printf("kv_host_dma_test: ok\n");
    return 0;
}
