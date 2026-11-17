// The prompt path's NVIDIA pick (src/pf_pick.hpp), on the CPU alone:
//   1. pf_pick is the table the engine used before: the largest of 8192 x 4, 8192 x 2, 4096 x 4, 4096 x 2 that fits, else 4096 x 2;
//   2. without the arena (FNX_ARENA=0, the launcher's setting: arena_pool -1) nothing is taken off a card's free VRAM, and no pool
//      entry is read;
//   3. with the arena the largest pool's slots are taken off (#5496): a card that looked roomy picks a smaller path.
// Exit 0 when all pass; each failure printed.
#include <cstdio>
#include <string>

#include "../src/model.hpp"
#include "../src/pf_pick.hpp"

static int fails = 0;
static void check(bool ok, const std::string& m) {
    if (!ok) {
        std::printf("FAIL: %s\n", m.c_str());
        ++fails;
    }
}

int main() {
    constexpr size_t MiB = 1ull << 20, GiB = 1ull << 30;
    const size_t margin = 3 * GiB;
    // 1. the table
    check(fnx::pf_pick(margin + 4148 * MiB, margin).rows == 8192 && fnx::pf_pick(margin + 4148 * MiB, margin).pass == 4, "8192 x 4 fits exactly");
    check(fnx::pf_pick(margin + 4147 * MiB, margin).pass == 2 && fnx::pf_pick(margin + 4147 * MiB, margin).rows == 8192, "one MiB short: 8192 x 2");
    check(fnx::pf_pick(margin + 3028 * MiB - 1, margin).rows == 4096 && fnx::pf_pick(margin + 3028 * MiB - 1, margin).pass == 4, "4096 x 4");
    check(fnx::pf_pick(margin + 1673 * MiB - 1, margin).rows == 4096 && fnx::pf_pick(margin + 1673 * MiB - 1, margin).pass == 2, "4096 x 2");
    check(fnx::pf_pick(0, margin).rows == 4096 && fnx::pf_pick(0, margin).pass == 2, "nothing fits: the smallest");

    // 2. no arena: no pool, nothing off (the engine read slot_bytes[-1] / n[-1] here before the guard)
    fnx::ExpertCache off;
    off.arena_pool = -1;
    off.n[0] = 9000;
    off.slot_bytes[0] = 1382400;
    check(fnx::pf_arena_unallocated(off, false) == 0, "FNX_ARENA=0: 0 bytes");
    check(fnx::pf_arena_unallocated(off, true) == 0, "arena on but no arena pool: 0 bytes");
    fnx::ExpertCache bad;
    bad.arena_pool = 7;   // out of the pools: never read
    check(fnx::pf_arena_unallocated(bad, true) == 0, "an arena pool out of range: 0 bytes");

    // 3. the arena's pool taken off: a 24 GB card with 20 GiB free before the arena, 11,000 slots of 1.38 MB (14.2 GiB) in the arena pool
    fnx::ExpertCache on;
    on.arena_pool = 1;
    on.n[1] = 11000;
    on.slot_bytes[1] = 1382400;
    const size_t arena = fnx::pf_arena_unallocated(on, true);
    check(arena == 11000ull * 1382400ull, "the arena pool's slots");
    check(fnx::pf_arena_unallocated(on, false) == 0, "the same cache with FNX_ARENA=0: 0 bytes");
    const size_t free_now = 20 * GiB;
    const fnx::PfPick before = fnx::pf_pick(free_now, margin);
    const fnx::PfPick after = fnx::pf_pick(fnx::pf_free_after_arena(free_now, arena), margin);
    check(before.rows == 8192 && before.pass == 4, "without the reserve the card looks roomy: 8192 x 4");
    check(after.rows == 4096 && after.pass == 4, "with it: 4096 x 4 (2,906 MiB left beside the 3 GiB margin)");
    check(fnx::pf_free_after_arena(5 * GiB, 6 * GiB) == 0, "an arena larger than the free VRAM: 0, not a wrap");

    std::printf("pf_pick_test: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}
