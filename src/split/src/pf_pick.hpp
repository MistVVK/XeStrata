// The prompt path's rows and chunks a pass on NVIDIA (Engine::Engine): the largest of 8192 x 4, 8192 x 2, 4096 x 4,
// 4096 x 2 whose buffers fit in the smallest card's free VRAM beside the K/V pool and the rest. Plain C++ (no CUDA), so
// tools/pf_pick_test.cpp checks it on the CPU.
//
// #5496 / PR #5667: with FNX_ARENA the largest expert pool's slots are allocated AFTER this pick (Engine::arena_fill), so
// cudaMemGetInfo still counts them as free; on cards where they are GiBs the pick took 8192 x 4 and the arena then moved
// experts to RAM to make room (3x RTX 3090: a pool fell under the window's 320 slots and the engine did not start). The
// pick now sees each card's free VRAM less those slots. Without the arena (FNX_ARENA=0, which the launcher sets) the
// card has no arena pool (arena_pool -1) and nothing is taken off.
#pragma once

#include <cstddef>

namespace fnx {

struct PfPick {
    int rows, pass;
    size_t mib;   // the prompt path's buffers on a card, measured (FNX_PF_TIMING "prefill buffers on card")
};

// largest first
inline constexpr PfPick kPfPicks[] = {{8192, 4, 4148}, {8192, 2, 3028}, {4096, 4, 1673}, {4096, 2, 1112}};

// the largest pick whose buffers fit beside `margin` in `free_min` bytes; the smallest when none does
inline PfPick pf_pick(size_t free_min, size_t margin) {
    for (const PfPick& p : kPfPicks)
        if ((p.mib << 20) + margin <= free_min) return p;
    return kPfPicks[sizeof kPfPicks / sizeof kPfPicks[0] - 1];
}

// the bytes the arena's pool will still take on a card: 0 without the arena or without an arena pool
template <class EC>
size_t pf_arena_unallocated(const EC& ec, bool arena) {
    constexpr int pools = (int) (sizeof ec.n / sizeof ec.n[0]);
    if (!arena || ec.arena_pool < 0 || ec.arena_pool >= pools || ec.n[ec.arena_pool] <= 0) return 0;
    return ec.slot_bytes[ec.arena_pool] * (size_t) ec.n[ec.arena_pool];
}

// a card's free VRAM as the pick sees it: what is free now less what the arena will take (0 when it takes more)
inline size_t pf_free_after_arena(size_t free_now, size_t arena_bytes) { return free_now > arena_bytes ? free_now - arena_bytes : 0; }

}  // namespace fnx
