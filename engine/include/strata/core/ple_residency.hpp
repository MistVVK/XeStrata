#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

#include "strata/core/conversation_memory.hpp"

namespace strata::core {

/// The 0.2.7 residency policy for the PLE (n-gram) table: keep the whole table in RAM, or leave it on direct reads?
///
/// The table is not the only thing that wants RAM.  The engine copies every routed expert into a pinned host arena at
/// start (about 51 GiB for IQ3_S), after the table has been opened, and an unbuffered read into that arena needs the
/// pages locked.  A locked table takes its share of the lockable memory first, so on a 64 GB PC (Windows, RTX 5070,
/// 2026-10-26) the policy that only looked at "table + 8 GiB available" admitted the table and the expert load then
/// failed with ERROR_WORKING_SET_QUOTA (1453).  The table is therefore admitted only when RAM holds the table AND the
/// arena the engine will still fill AND the headroom, all measured against what is available BEFORE either is
/// allocated.  Unknown memory telemetry, an unknown table size or an unknown arena size admit nothing (the direct
/// reads of 0.2.6 stay).
///
/// `experts_ram_bytes`: the native expert layout's total (every layer's blob x experts), 0 when it is not known.
inline bool ple_resident_admit(std::optional<uint64_t> available, uint64_t table_bytes, uint64_t experts_ram_bytes,
                               uint64_t headroom_bytes) {
    if (!available || table_bytes == 0 || experts_ram_bytes == 0) return false;
    if (table_bytes > UINT64_MAX - experts_ram_bytes) return false;
    return conversation_memory_admit(available, table_bytes + experts_ram_bytes, headroom_bytes);
}

/// May the residency policy run at all?  STRATA_PLE_AUTO_RAM when it is set (0 off, anything else on: an explicit value wins), otherwise on
/// except under STRATA_DEFAULTS=026 (`v026`: v0.2.6 had no such policy, the direct reads stayed).
inline bool ple_auto_ram_allowed(const char* env_value, bool v026) {
    if (env_value != nullptr) return std::atoi(env_value) != 0;
    return !v026;
}

/// #4518: the table was locked at start on the policy's word (table + arena + headroom fit), but what the rest of the engine takes after
/// that (the KV pools, the pinned staging, the GPUs' host buffers) is not known then.  On a 128 GB two-GPU Windows PC it left less than the
/// 5 GiB the automatic conversation cache needs, and the cache turned itself off.  The cache is worth more than the lock (parking a 100 K
/// conversation is 1.1 s against 30 s to read it again, the lock was measured at 2-3% of prompt speed), so once the cache is to be sized:
/// does giving the lock back make the automatic cache bigger?  `available` is the RAM the cache sees now (the table locked), the table's
/// pages count as available again once unlocked (clean file pages).  Unknown telemetry: no.
inline bool ple_lock_blocks_cache(std::optional<uint64_t> available, uint64_t table_bytes, int64_t min_free_mib) {
    if (!available || table_bytes == 0 || *available > UINT64_MAX - table_bytes) return false;
    return auto_conversation_cache_mib(*available + table_bytes, min_free_mib) > auto_conversation_cache_mib(available, min_free_mib);
}

/// #4846 (owner 11-01): what the expert arena does when the RAM is short.  The arena is loaded into anonymous memory; with less RAM
/// available than it needs (after the wait for an engine still exiting, and after the engine gave back the locks it took by itself) the
/// load pushes the host into reclaim, and a 60 GiB host was seen to stall until it was power-cycled (#4051, #4846).  The engine still
/// starts, never refuses: it reads the experts from the file like --mmap-experts for this run and says so.
///   Load      the RAM is there (or the arena is not priced: `need` 0)
///   FallBack  short, and the fallback is allowed: use the --mmap-experts expert source, register nothing
///   Warn      short, fallback not allowed (the user fixed the choice by hand): load as asked, with the warning
///   Refuse    a hard memory limit (a cgroup / container) below the arena, fallback not allowed: certain to be killed while loading
enum class ArenaRamVerdict { Load, FallBack, Warn, Refuse };

inline ArenaRamVerdict arena_ram_verdict(uint64_t need, uint64_t available, uint64_t hard_limit, bool fallback_allowed) {
    if (need == 0) return ArenaRamVerdict::Load;
    if (hard_limit < need) return fallback_allowed ? ArenaRamVerdict::FallBack : ArenaRamVerdict::Refuse;
    if (available < need) return fallback_allowed ? ArenaRamVerdict::FallBack : ArenaRamVerdict::Warn;
    return ArenaRamVerdict::Load;
}

/// May the engine swap the arena for the --mmap-experts source on its own?  Not when the user fixed the RAM by hand: `--ple-io ram`
/// asked for the PLE table locked (`ple_ram_by_hand`: the user's word, not the residency policy's), or a shared arena file
/// (--shared-expert-arena) was named.  STRATA_ARENA_RAM_FALLBACK=0 turns it off (testers: the old behaviour, to compare).
inline bool arena_ram_fallback_allowed(const char* env_value, bool ple_ram_by_hand, bool shared_arena) {
    if (ple_ram_by_hand || shared_arena) return false;
    if (env_value != nullptr && env_value[0] != '\0') return std::atoi(env_value) != 0;
    return true;
}

/// #5327: may a short arena still load, as 0.1.55 did?  The false alarm `available < need` raises on a host that can
/// load is a reserved hugetlb pool: MemAvailable does not count free huge pages at all, and a launcher may reserve the
/// pool for exactly this arena, so the reserve itself takes MemAvailable below the need while every page the arena
/// maps sits free in the pool (the 0.1.54/0.1.55 "expert arena read through the file cache", decode at full speed for
/// hours).  The arena's mapping is MAP_HUGETLB or nothing (pinned.cu), so the free pool is credited only when it holds
/// the WHOLE arena; a smaller pool serves none of it.  Free pages plus the droppable file pages
/// (`meminfo_free_and_droppable`) are credited too, but that is within a small margin of what MemAvailable already
/// counts (it keeps back the zone low watermark and reserves; this does not), so it only bridges a small shortfall.
/// The shortfall that must NOT load is the one backed by memory the kernel cannot take without swapping another
/// program's anonymous RAM - #4051's host (an engine that was still exiting) wedged in reclaim until it was
/// power-cycled, which is what the #4846 refusal is for.  Subtraction form, so no overflow.
inline bool arena_shortfall_served_by_file_cache(uint64_t need, uint64_t free_bytes, uint64_t droppable_file_bytes,
                                                 uint64_t huge_free_bytes = 0) {
    if (need == 0) return false;
    if (huge_free_bytes >= need) return true;
    if (free_bytes >= need) return true;
    return droppable_file_bytes >= need - free_bytes;
}

/// #5327: MemFree and the file pages the kernel can drop for the arena without touching swap or any program's anonymous
/// memory, from a /proc/meminfo text: MemFree, and Cached + Buffers + SReclaimable - Shmem - Dirty - Mlocked, floored at 0
/// (Shmem is swap-backed and counted inside Cached; Dirty needs a writeback before its pages are reusable; Mlocked pages
/// are file pages too - the PLE table's mlock is a mapping of the model file, the same for a program that is still
/// exiting - but the kernel cannot drop them: MemAvailable leaves them out, so this must as well).
/// `huge_free_bytes` (optional) gets (HugePages_Free - HugePages_Rsvd) x Hugepagesize, the pool the arena's MAP_HUGETLB try
/// takes (pages another mapping has reserved are spoken for), and only where Hugepagesize is the 2 MiB the arena asks for.
/// False when MemFree or Cached is absent (no credit: the refusal stands).
inline bool meminfo_free_and_droppable(const std::string& meminfo_text, uint64_t& free_bytes, uint64_t& droppable_file_bytes,
                                       uint64_t* huge_free_bytes = nullptr) {
    free_bytes = droppable_file_bytes = 0;
    if (huge_free_bytes) *huge_free_bytes = 0;
    bool have_free = false, have_cached = false;
    uint64_t cached = 0, buffers = 0, sreclaimable = 0, shmem = 0, dirty = 0, mlocked = 0, huge_pages = 0, huge_rsvd = 0, huge_size = 0;
    std::istringstream in(meminfo_text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (!(fields >> key >> value)) continue;
        fields >> unit;
        const uint64_t bytes = unit == "kB" ? value * 1024 : value;
        if (key == "MemFree:") { free_bytes = bytes; have_free = true; }
        else if (key == "Cached:") { cached = bytes; have_cached = true; }
        else if (key == "Buffers:") buffers = bytes;
        else if (key == "SReclaimable:") sreclaimable = bytes;
        else if (key == "Shmem:") shmem = bytes;
        else if (key == "Dirty:") dirty = bytes;
        else if (key == "Mlocked:") mlocked = bytes;
        else if (key == "HugePages_Free:") huge_pages = value;
        else if (key == "HugePages_Rsvd:") huge_rsvd = value;
        else if (key == "Hugepagesize:") huge_size = bytes;
    }
    if (!have_free || !have_cached) return false;
    const uint64_t held_a = shmem > UINT64_MAX - dirty ? UINT64_MAX : shmem + dirty;
    const uint64_t held = held_a > UINT64_MAX - mlocked ? UINT64_MAX : held_a + mlocked;
    droppable_file_bytes = cached + buffers + sreclaimable > held ? cached + buffers + sreclaimable - held : 0;
    if (huge_free_bytes && huge_size == (2ull << 20)) {   // the arena maps MAP_HUGE_2MB: another pool size is not its pool
        const uint64_t pages = huge_pages > huge_rsvd ? huge_pages - huge_rsvd : 0;
        *huge_free_bytes = pages > UINT64_MAX / huge_size ? UINT64_MAX : pages * huge_size;
    }
    return true;
}

/// --ram-keep-gib W (default 0; STRATA_RAM_KEEP_GIB is the same for a program that cannot pass the flag): the RAM the person wants left
/// free for their own work.  One setting for both RAM decisions at the start:
///   1. the expert arena is used only if W GiB are still available after it (`arena_need_with_keep`: its need plus W against the RAM available);
///   2. otherwise the RAM copy of the hottest experts (the budget form) leaves max(W, the platform's safety headroom) free, unless
///      STRATA_RESIDENT_HEADROOM_GIB is set: that value wins.
/// W = 0 is the behaviour before the setting existed: the arena whenever it fits, and the safety headroom.  The safety headroom is not
/// optional: on Windows the copy is read through a mapped view whose pages are counted in the process, and the page trim that gives them
/// back needs room (available RAM fell to 0.9 GiB during the copy of a 34.6 GiB budget with less).  8 GiB on Windows, 4 GiB elsewhere.
inline constexpr uint64_t kGiB = 1073741824ull;
inline constexpr uint64_t kRamKeepMax = 1024ull * kGiB;   // a typo cannot ask for more than any PC has

inline uint64_t ram_keep_from_gib(double gib) {
    if (!(gib > 0.0)) return 0;
    if (gib >= (double) (kRamKeepMax / kGiB)) return kRamKeepMax;
    return (uint64_t) (gib * (double) kGiB);
}

inline uint64_t fallback_safety_headroom(bool windows) { return windows ? 8ull * kGiB : 4ull * kGiB; }

/// The RAM the arena (plus its spare blob) must find available: its need and the RAM to keep free, no overflow.
inline uint64_t arena_need_with_keep(uint64_t need, uint64_t keep) {
    return keep > UINT64_MAX - need ? UINT64_MAX : need + keep;
}

/// What the arena needs under a HARD memory limit (a cgroup, a container, a job object, --memory-limit-mib).  The arena is pinned: it is
/// never given back to make room, and the engine's own memory (about 3 GiB: buffers, the draft head, the PLE and PLE row cache, the CUDA
/// context) comes on top, so a limit just above the arena kills the engine at the start (cgroup OOM, found on l620: a 49 GiB limit and a
/// 46.84 GiB arena).  When the fallback is allowed the arena needs its headroom under the limit as well, max(keep, the fallback headroom);
/// when it is not allowed (a hand-set --ple-io ram, a shared arena file, STRATA_ARENA_RAM_FALLBACK=0) there is nothing to fall back to, so
/// the comparison stays as it was (the arena and the RAM to keep).
inline uint64_t arena_hard_limit_need(uint64_t base_need, uint64_t keep, uint64_t headroom, bool fallback_allowed) {
    return arena_need_with_keep(base_need, fallback_allowed ? (keep > headroom ? keep : headroom) : keep);
}

/// The headroom the RAM-copy fallback leaves free.  `env_headroom`: STRATA_RESIDENT_HEADROOM_GIB when it is set (wins, even below the safety headroom).
inline uint64_t fallback_headroom(std::optional<uint64_t> env_headroom, uint64_t keep, uint64_t safety) {
    if (env_headroom) return *env_headroom;
    return keep > safety ? keep : safety;
}

/// The RAM copy's budget: what is available less the headroom; 0 (no copy: the plain file source) when less than 1 GiB would be left to it.
inline uint64_t fallback_budget(uint64_t available, uint64_t headroom) {
    return (headroom <= UINT64_MAX - kGiB && available > headroom + kGiB) ? available - headroom : 0;
}

/// Both decisions at once, for the tests and the startup line.
struct RamKeepPlan { bool arena; uint64_t headroom; uint64_t budget; };
inline RamKeepPlan ram_keep_plan(uint64_t arena_need, uint64_t available, uint64_t keep, std::optional<uint64_t> env_headroom, uint64_t safety) {
    RamKeepPlan p{};
    p.arena = arena_ram_verdict(arena_need_with_keep(arena_need, keep), available, ~uint64_t{0}, true) == ArenaRamVerdict::Load;
    p.headroom = fallback_headroom(env_headroom, keep, safety);
    p.budget = p.arena ? 0 : fallback_budget(available, p.headroom);
    return p;
}

} // namespace strata::core
