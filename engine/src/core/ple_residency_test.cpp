#include "strata/core/ple_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t GiB = 1ull << 30;
}

int main() {
    const uint64_t table = 26ull * GiB + 870ull * (1ull << 20);   // IQ3_S's table, 26.8 GiB
    const uint64_t arena = 51ull * GiB + 100ull * (1ull << 20);   // IQ3_S's 51.1 GiB of experts
    // the owner PC, 2026-10-26: 45.2 GiB available when the table is opened, 63 GiB total.  The 0.2.7 first policy
    // (table + 8 GiB) admitted it and the expert load then failed with error 1453.  Not admitted now.
    check(!ple_resident_admit(45ull * GiB + (1ull << 28), table, arena, 8 * GiB), "64 GB PC with the IQ3_S arena: not admitted");
    check(!ple_resident_admit(40ull * GiB, table, arena, 8 * GiB), "40 GiB: not admitted");
    // STRATA_DEFAULTS=026: the policy does not run (v0.2.6 had none); an explicit STRATA_PLE_AUTO_RAM wins either way
    check(ple_auto_ram_allowed(nullptr, false), "policy on by default");
    check(!ple_auto_ram_allowed(nullptr, true), "policy off under STRATA_DEFAULTS=026");
    check(ple_auto_ram_allowed("1", true), "STRATA_PLE_AUTO_RAM=1 wins over 026");
    check(!ple_auto_ram_allowed("0", false), "STRATA_PLE_AUTO_RAM=0 is the old off switch");
    check(!ple_auto_ram_allowed("0", true), "STRATA_PLE_AUTO_RAM=0 under 026");
    // a PC that holds the table, the arena and 8 GiB
    check(ple_resident_admit(table + arena + 8 * GiB, table, arena, 8 * GiB), "exactly table + arena + headroom is admitted");
    check(!ple_resident_admit(table + arena + 8 * GiB - 1, table, arena, 8 * GiB), "one byte below is not");
    check(ple_resident_admit(240ull * GiB, table, arena, 8 * GiB), "a 256 GB box: admitted");
    // a smaller model's arena on a mid PC (Q2_0: 34 GiB arena): 26.8 + 34 + 8 = 68.8
    check(ple_resident_admit(80ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 on 96 GB: admitted");
    check(!ple_resident_admit(60ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 with 60 GiB available: not admitted");
    // unknowns fail closed
    check(!ple_resident_admit({}, table, arena, 8 * GiB), "unknown telemetry");
    check(!ple_resident_admit(240ull * GiB, 0, arena, 8 * GiB), "unknown table size");
    check(!ple_resident_admit(240ull * GiB, table, 0, 8 * GiB), "unknown arena size");
    check(!ple_resident_admit(std::numeric_limits<uint64_t>::max(), table, std::numeric_limits<uint64_t>::max(), 8 * GiB),
          "sizes cannot overflow");
    check(!ple_resident_admit(0, table, arena, 0), "zero available");
    // #4518: after the load, does the table's lock leave the automatic conversation cache no room?  (min free 2560 MiB, cache 40% of the
    // rest, 8192 MiB at most, 1024 MiB at least)
    {
        const int64_t floor_mib = 2560;
        check(!ple_lock_blocks_cache(std::nullopt, table, floor_mib), "unknown telemetry: leave the lock");
        check(!ple_lock_blocks_cache(40ull * GiB, 0, floor_mib), "unknown table size: leave the lock");
        check(ple_lock_blocks_cache(4ull * GiB, table, floor_mib), "the reporter: 4 GiB left with the table locked (cache off), 30 GiB without");
        check(ple_lock_blocks_cache(10ull * GiB, table, floor_mib), "10 GiB left: cache 1.9 GiB with the lock, 8 GiB without");
        check(!ple_lock_blocks_cache(40ull * GiB, table, floor_mib), "40 GiB left: the cache is already at its 8 GiB cap");
        check(!ple_lock_blocks_cache(23ull * GiB, table, floor_mib), "23 GiB: the cap is just reached with the lock");
        check(ple_lock_blocks_cache(22ull * GiB, table, floor_mib), "22 GiB: the lock costs the cache some of its 8 GiB");
        check(!ple_lock_blocks_cache(1ull * GiB, 1ull << 20, floor_mib), "a tiny table cannot make a cache");
        check(!ple_lock_blocks_cache(std::numeric_limits<uint64_t>::max(), table, floor_mib), "no overflow");
    }
    // #4846 (owner 11-01): RAM still short at the arena after the wait and the give-back: the --mmap-experts source, never a refusal,
    // never registering anyway.  A hand-set --ple-io ram or a shared arena file is respected (warn only).
    {
        const uint64_t need = arena + (1ull << 20);   // the arena and one blob
        const uint64_t none = std::numeric_limits<uint64_t>::max();   // no cgroup limit
        check(arena_ram_verdict(need, need, none, true) == ArenaRamVerdict::Load, "RAM exactly there: load");
        check(arena_ram_verdict(need, 240ull * GiB, none, true) == ArenaRamVerdict::Load, "plenty: load");
        check(arena_ram_verdict(need, need - 1, none, true) == ArenaRamVerdict::FallBack, "one byte short: file source");
        check(arena_ram_verdict(need, 20ull * GiB, none, true) == ArenaRamVerdict::FallBack, "20 GiB of 51: file source");
        check(arena_ram_verdict(need, 20ull * GiB, none, false) == ArenaRamVerdict::Warn, "20 GiB of 51, fixed by hand: warn only");
        check(arena_ram_verdict(need, 240ull * GiB, 30ull * GiB, true) == ArenaRamVerdict::FallBack, "a 30 GiB container limit: file source, not a refusal");
        check(arena_ram_verdict(need, 240ull * GiB, 30ull * GiB, false) == ArenaRamVerdict::Refuse, "a 30 GiB container limit, fixed by hand: as before");
        check(arena_ram_verdict(need, 240ull * GiB, need, true) == ArenaRamVerdict::Load, "a limit of exactly the arena: load");
        check(arena_ram_verdict(0, 0, 0, true) == ArenaRamVerdict::Load, "an unpriced arena loads");
        check(arena_ram_fallback_allowed(nullptr, false, false), "fallback on by default");
        check(!arena_ram_fallback_allowed(nullptr, true, false), "a hand-set --ple-io ram is kept");
        check(!arena_ram_fallback_allowed(nullptr, false, true), "a shared arena file is kept");
        check(!arena_ram_fallback_allowed("0", false, false), "STRATA_ARENA_RAM_FALLBACK=0 is the old behaviour");
        check(arena_ram_fallback_allowed("1", false, false), "STRATA_ARENA_RAM_FALLBACK=1");
        check(arena_ram_fallback_allowed("", false, false), "an empty value is unset");
        check(!arena_ram_fallback_allowed("1", true, false), "the user's --ple-io ram wins over the env value");
    }
    // #5327: a shortfall the kernel can serve by dropping the box's own file cache loads through it (0.1.55's
    // warn-and-proceed), one it cannot (anonymous RAM: #4051's host) keeps the #4846 fallback
    {
        const uint64_t need = 31ull * GiB + 687000000ull;   // the reporter's Q2_0 arena: 31.64 GiB
        check(arena_shortfall_served_by_file_cache(need, 1ull * GiB, 38ull * GiB),
              "12.9 GiB available, the model's files hot in the cache: load through the file cache");
        check(!arena_shortfall_served_by_file_cache(need, 300ull << 20, 6ull * GiB),
              "#4051's host: an exiting engine's anonymous RAM is not the cache: keep the fallback");
        check(arena_shortfall_served_by_file_cache(need, need, 0), "free pages alone cover it");
        check(!arena_shortfall_served_by_file_cache(need, 0, need - 1), "one byte short even with the cache");
        check(!arena_shortfall_served_by_file_cache(0, 0, 0), "an unpriced arena is not this decision");
        check(arena_shortfall_served_by_file_cache(std::numeric_limits<uint64_t>::max(), 5, std::numeric_limits<uint64_t>::max()),
              "no overflow");
        check(arena_shortfall_served_by_file_cache(need, 1ull * GiB, 6ull * GiB, 32ull * GiB),
              "a reserved hugetlb pool that holds the whole arena: load");
        check(!arena_shortfall_served_by_file_cache(need, 1ull * GiB, 6ull * GiB, need - 1),
              "a pool one page short serves none of it (MAP_HUGETLB is all-or-nothing): keep the fallback");
        uint64_t free_b = 1, drop_b = 1, huge_b = 1;
        check(meminfo_free_and_droppable(
                  "MemTotal:       1000 kB\nMemFree:          10 kB\nMemAvailable:    100 kB\nBuffers:           5 kB\n"
                  "Cached:         400 kB\nSwapCached:       0 kB\nActive(file):   200 kB\nDirty:            2 kB\n"
                  "Shmem:          30 kB\nSReclaimable:    15 kB\nHugePages_Total: 16384\nHugePages_Free:  16384\n"
                  "Hugepagesize:    2048 kB\n",
                  free_b, drop_b, &huge_b) && free_b == 10 * 1024 && drop_b == (400 + 5 + 15 - 30 - 2) * 1024 &&
                  huge_b == 16384ull * 2048 * 1024,
              "free, droppable and huge pages parsed (Shmem and Dirty are not droppable)");
        check(meminfo_free_and_droppable("MemFree: 1 kB\nCached: 2 kB\n", free_b, drop_b) && free_b == 1024 && drop_b == 2 * 1024,
              "Buffers/SReclaimable/Shmem/Dirty are optional");
        check(!meminfo_free_and_droppable("MemFree: 1 kB\n", free_b, drop_b), "no Cached line: unknown, no credit");
        check(!meminfo_free_and_droppable("Cached: 1 kB\n", free_b, drop_b), "no MemFree line: unknown, no credit");
        // the loopholes a review of #5531 found: locked file pages (the PLE table's mlock) sit in Cached but the kernel cannot
        // drop them (MemAvailable leaves them out); reserved huge pages are counted free but spoken for; only a 2 MiB pool
        // is the arena's pool
        check(meminfo_free_and_droppable("MemFree: 10 kB\nCached: 4000 kB\nBuffers: 5 kB\nMlocked: 3000 kB\nDirty: 2 kB\nShmem: 30 kB\n",
                                         free_b, drop_b) &&
                  drop_b == (4000 + 5 - 30 - 2 - 3000) * 1024,
              "Mlocked pages are not droppable");
        check(meminfo_free_and_droppable("MemFree: 1 kB\nCached: 100 kB\nMlocked: 5000 kB\n", free_b, drop_b) && drop_b == 0,
              "Mlocked above the cache floors at 0");
        check(meminfo_free_and_droppable("MemFree: 1 kB\nCached: 1 kB\nHugePages_Free: 16384\nHugePages_Rsvd: 16000\nHugepagesize: 2048 kB\n",
                                         free_b, drop_b, &huge_b) &&
                  huge_b == 384ull * 2048 * 1024,
              "reserved huge pages are not free for the arena");
        check(meminfo_free_and_droppable("MemFree: 1 kB\nCached: 1 kB\nHugePages_Free: 16\nHugePages_Rsvd: 20\nHugepagesize: 2048 kB\n",
                                         free_b, drop_b, &huge_b) &&
                  huge_b == 0,
              "more reserved than free: none");
        check(meminfo_free_and_droppable("MemFree: 1 kB\nCached: 1 kB\nHugePages_Free: 64\nHugepagesize: 1048576 kB\n",
                                         free_b, drop_b, &huge_b) &&
                  huge_b == 0,
              "a 1 GiB default pool is not the 2 MiB pool the arena maps");
        // #4051's other shape: an engine still exiting holds the PLE table mlocked.  Cached has it, MemAvailable does not.
        {
            const uint64_t need_x = 47ull * GiB;
            check(meminfo_free_and_droppable("MemFree: " + std::to_string(2 * GiB / 1024) + " kB\nCached: " + std::to_string(55 * GiB / 1024) +
                                             " kB\nMlocked: " + std::to_string(27 * GiB / 1024) + " kB\n",
                                             free_b, drop_b) &&
                      !arena_shortfall_served_by_file_cache(need_x, free_b, drop_b),
                  "a locked PLE table in the cache does not serve the arena: keep the fallback");
        }
    }
    // --ram-keep-gib W (R5070): ONE setting for both RAM decisions at the start.  The arena is used only if W GiB stay available after it;
    // otherwise the RAM copy leaves max(W, the safety headroom: 8 GiB Windows, 4 GiB elsewhere), unless STRATA_RESIDENT_HEADROOM_GIB is set.
    {
        const uint64_t need = 46ull * GiB + 860ull * (1ull << 20) + 2660000ull;   // IQ3_S: 46.84 GiB of experts and the spare blob
        const uint64_t none = std::numeric_limits<uint64_t>::max();
        const uint64_t win = fallback_safety_headroom(true), lin = fallback_safety_headroom(false);
        check(win == 8 * GiB && lin == 4 * GiB, "safety headroom: 8 GiB Windows, 4 GiB elsewhere");
        check(ram_keep_from_gib(0.0) == 0 && ram_keep_from_gib(-3.0) == 0 && ram_keep_from_gib(8.0) == 8 * GiB, "W in GiB");
        check(ram_keep_from_gib(1e9) == kRamKeepMax, "a typo cannot ask for more than 1 TiB");
        check(arena_need_with_keep(need, 0) == need && arena_need_with_keep(need, 8 * GiB) == need + 8 * GiB, "need plus W");
        check(arena_need_with_keep(none - 5, 100) == none, "need plus W cannot overflow");
        // W = 0 is the behaviour before the setting: the arena exactly when arena_ram_verdict says Load, the copy leaves the safety headroom
        const double frees[] = {10.0, 24.0, 30.0, 36.0, 40.0, 44.6, 46.5, 48.0, 48.9, 49.1, 50.0, 54.9, 60.0, 100.0};
        for (const double f : frees) {
            const uint64_t avail = (uint64_t) (f * (double) GiB);
            for (const uint64_t safety : {win, lin}) {
                const RamKeepPlan p0 = ram_keep_plan(need, avail, 0, std::nullopt, safety);
                check(p0.arena == (arena_ram_verdict(need, avail, none, true) == ArenaRamVerdict::Load), "W=0: the arena whenever it fits, as before");
                check(p0.headroom == safety, "W=0: the safety headroom");
                check(p0.budget == (p0.arena ? 0 : (avail > safety + GiB ? avail - safety : 0)), "W=0: the budget is what is available less the safety headroom");
                uint64_t prev_budget = none;
                bool prev_arena = true;
                for (const uint64_t w : {0ull, 5ull * GiB, 7ull * GiB, 8ull * GiB, 16ull * GiB}) {
                    const RamKeepPlan p = ram_keep_plan(need, avail, w, std::nullopt, safety);
                    check(p.arena == (avail >= need + w), "the arena only if W GiB stay available after it");
                    if (p.arena) check(avail - need >= w, "an arena start leaves at least W free");
                    check(p.headroom == (w > safety ? w : safety), "the headroom is max(W, safety)");
                    if (!p.arena && p.budget > 0) check(avail - p.budget >= p.headroom, "a RAM copy leaves at least the headroom free");
                    check(!p.arena || prev_arena, "a larger W never turns the fallback into the arena");
                    check(p.budget <= prev_budget || prev_arena, "a larger W never grows the copy");
                    prev_arena = p.arena;
                    prev_budget = p.arena ? none : p.budget;
                }
                // the environment variable wins over both W and the safety headroom, even when smaller
                const RamKeepPlan e = ram_keep_plan(need, avail, 8 * GiB, std::optional<uint64_t>(2 * GiB), safety);
                check(e.headroom == 2 * GiB, "STRATA_RESIDENT_HEADROOM_GIB still wins");
                check(e.arena == (avail >= need + 8 * GiB), "the environment does not move the arena decision");
            }
        }
        // the owner's PC (63.1 GB, 12 GB card, IQ3_S): 48.9 GiB free at the start
        const uint64_t f489 = (uint64_t) (48.9 * (double) GiB), f446 = (uint64_t) (44.6 * (double) GiB);
        check(ram_keep_plan(need, f489, 0, std::nullopt, win).arena, "48.9 GiB free, W=0: the arena");
        check(!ram_keep_plan(need, f489, 8 * GiB, std::nullopt, win).arena, "48.9 GiB free, W=8: not the arena (2 GiB left)");
        const RamKeepPlan w8 = ram_keep_plan(need, f489, 8 * GiB, std::nullopt, win);
        check(w8.headroom == 8 * GiB && w8.budget > 39 * GiB && w8.budget < 41 * GiB, "48.9 GiB free, W=8: a 40.9 GiB RAM copy, 8 GiB left");
        const RamKeepPlan w5 = ram_keep_plan(need, f489, 5 * GiB, std::nullopt, win);
        check(!w5.arena && w5.headroom == 8 * GiB, "48.9 GiB free, W=5 on Windows: 48.9 < 46.8 + 5, so the copy, and the safety headroom of 8 GiB wins over 5");
        check(ram_keep_plan(need, f489, 5 * GiB, std::nullopt, lin).headroom == 5 * GiB, "W=5 on Linux: the copy leaves 5 GiB");
        check(!ram_keep_plan(need, f446, 0, std::nullopt, win).arena && ram_keep_plan(need, f446, 0, std::nullopt, win).budget > 36 * GiB,
              "44.6 GiB free, W=0: the fallback with a 36.6 GiB copy");
    }
    // A HARD memory limit just above the arena (R5070, found on l620: a 49 GiB cgroup limit, the 46.84 GiB arena chosen, the engine killed by the
    // cgroup OOM killer at the start).  With the fallback allowed the arena needs its headroom under the limit too.
    {
        const uint64_t need = 46ull * GiB + 860ull * (1ull << 20) + 2660000ull;   // IQ3_S: 46.84 GiB of experts and the spare blob
        const uint64_t head4 = 4 * GiB, head8 = 8 * GiB;
        auto verdict = [&](uint64_t limit, uint64_t keep, uint64_t headroom, bool fallback) {
            return arena_ram_verdict(arena_hard_limit_need(need, keep, headroom, fallback), ~uint64_t{0}, limit, fallback);
        };
        check(arena_hard_limit_need(need, 0, head4, true) == need + head4, "fallback allowed: the arena needs its headroom under the limit");
        check(arena_hard_limit_need(need, 8 * GiB, head4, true) == need + 8 * GiB, "the RAM to keep counts when it is larger than the headroom");
        check(arena_hard_limit_need(need, 0, head4, false) == need, "fallback not allowed: the comparison stays as it was");
        check(arena_hard_limit_need(need, 8 * GiB, head4, false) == need + 8 * GiB, "fallback not allowed: the RAM to keep is still asked");
        check(verdict(49ull * GiB, 0, head4, true) == ArenaRamVerdict::FallBack, "49 GiB limit, 4 GiB headroom: the copy, not an arena that the OOM killer ends");
        check(verdict(49ull * GiB, 0, head8, true) == ArenaRamVerdict::FallBack, "49 GiB limit, 8 GiB headroom (Windows): the copy");
        check(verdict(51ull * GiB, 0, head4, true) == ArenaRamVerdict::Load, "51 GiB limit, 4 GiB headroom: the arena");
        check(verdict(need + head4, 0, head4, true) == ArenaRamVerdict::Load, "a limit of exactly arena + headroom: the arena");
        check(verdict(need + head4 - 1, 0, head4, true) == ArenaRamVerdict::FallBack, "one byte under: the copy");
        check(verdict(need - 1, 0, head4, true) == ArenaRamVerdict::FallBack, "a limit under the arena: the copy, as before");
        check(verdict(49ull * GiB, 0, head4, false) == ArenaRamVerdict::Load, "fallback not allowed, 49 GiB limit: as before (no new refusal)");
        check(verdict(45ull * GiB, 0, head4, false) == ArenaRamVerdict::Refuse, "fallback not allowed, limit under the arena: refused as before");
        check(verdict(60ull * GiB, 8 * GiB, head4, true) == ArenaRamVerdict::Load, "60 GiB limit, 8 GiB to keep: the arena");
        check(verdict(54ull * GiB, 8 * GiB, head4, true) == ArenaRamVerdict::FallBack, "54 GiB limit, 8 GiB to keep: the copy");
        check(verdict(std::numeric_limits<uint64_t>::max(), 0, head4, true) == ArenaRamVerdict::Load, "no limit: the arena");
    }
    std::printf("ple_residency_test: %d checks passed\n", checks);
    return 0;
}
