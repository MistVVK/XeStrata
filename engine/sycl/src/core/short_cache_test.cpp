// sycl/src/core/short_cache_test.cpp - the sizing of a start short of VRAM (strata/core/short_cache.hpp), on the 8 GB Arc A750's figures from the engine's logs.
#include "strata/core/short_cache.hpp"

#include <cstdio>

using namespace strata::core;
static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-110s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    const int64_t MiB = 1 << 20;
    const int64_t blob = 2330000 + 0;   // the pack's largest blob (2.33 MB)
    const int64_t head = 73 * MiB;
    // --batch 2 + MTP, 8192 tokens, 1024 MiB reserve given: the engine's auto sizing left 0 slots (2.46 GiB free, chunk 2048)
    const int64_t free8k = (int64_t) (2.46 * 1024) * MiB;
    check(slots_with_own_chunk(free8k, 1024, 2048, head, 0, blob, 24576) == 0, "8K, reserve 1024, chunk 2048: the sizing left no slot (the refused start)");
    ShortCache a = size_short_cache(free8k, 1024, true, 300, 2048, false, 256, 128, head, 0, blob, 24576);
    check(a.chunk == 1024 && a.reserve_mib == 1024 && a.slots >= 128, "... the chunk goes 2048 -> 1024, the typed reserve stays, a working cache is left");
    // the same with 128K sessions of 0.73 GiB (KV window 32768): 0.92 GiB free
    const int64_t free128 = (int64_t) (0.92 * 1024) * MiB;
    ShortCache b = size_short_cache(free128, 300, true, 300, 1024, false, 256, 128, head, 0, blob, 24576);
    check(b.chunk == 256 && b.reserve_mib == 300 && b.slots > 0 && b.slots < 128, "128K, window 32768: the smallest chunk, a cache below 128 slots but a start");
    // a typed reserve is lowered only after the chunk is as small as it goes
    ShortCache c = size_short_cache((int64_t) (1.36 * 1024) * MiB, 1024, true, 300, 8192, false, 256, 128, head, 0, blob, 24576);
    check(c.chunk == 256 && c.reserve_mib < 1024 && c.reserve_mib >= 300 && c.slots >= 128, "128K, window 20480, reserve 1024: chunk 256 first, then the reserve comes down only as far as needed");
    // the person's own chunk is not touched
    ShortCache d = size_short_cache(free8k, 1024, true, 300, 2048, true, 256, 128, head, 0, blob, 24576);
    check(d.chunk == 2048, "a chunk the person asked for stays");
    ShortCache e = size_short_cache(free8k, 1024, false, 300, 2048, true, 256, 128, head, 0, blob, 24576);
    check(e.reserve_mib == 1024, "a reserve that was not asked for is not touched here (the engine's own reserve rule runs before)");
    // impossible: nothing left even at the smallest chunk and reserve
    ShortCache f = size_short_cache(100 * MiB, 300, true, 300, 2048, false, 256, 128, head, 0, blob, 24576);
    check(f.slots == 0 && f.chunk == 256, "a card with 100 MiB free: no slot at the smallest chunk - the one plain refusal");
    // the KV window
    auto need = [&](int64_t w) { return (w == 0 ? 3 * 1930LL : w == 65536 ? 3 * 1050LL : w == 32768 ? 3 * 750LL : 3 * 585LL) * MiB + 3000 * MiB; };
    check(pick_kv_resident(131072, 12000 * MiB, need) == 0, "a card that holds the whole context keeps it in VRAM (no streaming)");
    check(pick_kv_resident(131072, 8500 * MiB, need) == 65536, "the largest window that fits is taken");
    check(pick_kv_resident(131072, 6100 * MiB, need) == 32768, "... the next one down");
    check(pick_kv_resident(131072, 4000 * MiB, need) == 20480, "none fits: the engine's smallest window");
    check(pick_kv_resident(32768, 4000 * MiB, need) == 20480 && pick_kv_resident(20480, 100 * MiB, need) == 20480, "a window is never larger than the context asked for");
    std::printf("%s\n", g_fail ? "SHORT-CACHE: FAIL" : "SHORT-CACHE: ok");
    return g_fail ? 1 : 0;
}
