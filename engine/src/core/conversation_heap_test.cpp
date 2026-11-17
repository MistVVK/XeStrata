// #5697: an evicted parked conversation must give its RAM back to the system, because the physical-RAM gate that
// evicted it reads MemAvailable right after.  With glibc the K/V segments of a parked conversation land in the heap
// once the dynamic mmap threshold has risen (it does at the first freed segment), and free() hands back only the top of
// the heap: an entry below a newer one stayed resident, so the gate evicted everything and still refused.
//
// The test parks three conversations, evicts the oldest (the one below the others) and reads the process RSS from
// /proc/self/statm before and after.  Linux with glibc only; elsewhere it says SKIPPED.  It also prints what the
// eviction costs in time.
#include "strata/core/conversation_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__) && defined(__GLIBC__)
#include <unistd.h>

using namespace strata::core;

namespace {
constexpr size_t MiB = size_t(1) << 20;
constexpr size_t layers = 32, part = 8 * MiB;              // 32 layers x (K + V) x 8 MiB = 512 MiB per conversation

size_t rss_bytes() {
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long size = 0, resident = 0;
    const int n = std::fscanf(f, "%lu %lu", &size, &resident);
    std::fclose(f);
    return n == 2 ? size_t(resident) * size_t(sysconf(_SC_PAGESIZE)) : 0;
}

SavedConversation conversation(int32_t id, uint8_t fill) {
    SavedConversation s;
    s.live.ids = {id, id + 1, id + 2};
    s.live.gdn.resize(64, fill);
    s.kv.resize(layers);
    for (auto& layer : s.kv) {
        layer.k.resize(part, fill);                        // written: the pages are resident, as a parked image's are
        layer.v.resize(part, fill);
    }
    return s;
}
}

int main() {
    // What the engine's first freed segment does: a freed mmapped block of this size raises glibc's mmap threshold,
    // so the segments allocated after it come from the heap.
    {
        std::vector<uint8_t> warm(16 * MiB);
        std::memset(warm.data(), 1, warm.size());
    }
    ConversationCache cache(size_t(8) << 30, 4);
    for (int32_t i = 0; i < 3; ++i)
        if (!cache.put(conversation(100 * (i + 1), uint8_t(i + 1)))) { std::fprintf(stderr, "FAIL: park %d\n", i); return 1; }
    const size_t entry = cache.bytes() / 3;
    const size_t before = rss_bytes();
    const auto t0 = std::chrono::steady_clock::now();
    const bool evicted = cache.evict_oldest();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t after = rss_bytes();
    const size_t returned = before > after ? before - after : 0;
    std::printf("conversation_heap_test: parked 3 x %zu MiB, RSS %zu MiB -> %zu MiB after evicting the oldest "
                "(%zu MiB returned to the system), eviction took %.2f ms\n",
                entry / MiB, before / MiB, after / MiB, returned / MiB, ms);
    if (!evicted || cache.size() != 2) { std::fprintf(stderr, "FAIL: evict_oldest\n"); return 1; }
    if (returned < entry * 8 / 10) {
        std::fprintf(stderr, "FAIL: the evicted conversation's RAM stayed with the process (%zu of %zu MiB returned)\n",
                     returned / MiB, entry / MiB);
        return 1;
    }
    // the second eviction: the heap holds the third conversation above it, nothing else freed
    const auto t1 = std::chrono::steady_clock::now();
    cache.evict_oldest();
    std::printf("conversation_heap_test: the second eviction took %.2f ms, RSS now %zu MiB\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(), rss_bytes() / MiB);
    std::printf("conversation_heap_test: ok\n");
    return 0;
}
#else
int main() {
    std::printf("SKIPPED: conversation_heap_test needs Linux with glibc\n");
    return 0;
}
#endif
