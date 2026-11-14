#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now - the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as the kernel reports it, less the group's clean file cache, which the kernel gives back at
// memory.high or the limit before it kills anything (cache_credit_from_stat below).  Under a limit below MemTotal (a
// container) every clean page on the evictable file lists counts (active_file + inactive_file) except this process's
// own resident file mappings (the PLE table, mapped model files, the libraries: the engine reads them, so planning their
// RAM for something else would push them out); otherwise, and with STRATA_MEM_CREDIT_INACTIVE=1, only inactive_file
// does, as before.  The lists are list state, not page types
// (cgroup-v2 memory.stat): mlocked pages are on the unevictable list and shmem (the pinned complements, the KV pool)
// is on the anon lists, so neither is ever credited.  A memory.stat that is missing or unparsable gives no credit (it
// does not make the sample unknown).
// Pinned, locked and shared memory stays charged in full: a hard-limit kill is worse than a skipped park.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// makes the sample unknown (fail closed); a file that is absent (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
    std::string self_status = "/proc/self/status";
};

/// One group's reclaimable file cache from its memory.stat text (v2 names, or v1's total_* names), each less the dirty
/// and writeback pages and never more than `current`:
///   inactive  inactive_file (the rule outside containers, and the earlier rule everywhere);
///   lists     inactive_file + active_file: every clean page the kernel can evict for the group.
/// No inactive_file line, or a malformed inactive / dirty / writeback value: both 0.  A malformed or missing active_file:
/// lists = inactive.
struct CacheCredit { uint64_t inactive = 0; uint64_t lists = 0; };
CacheCredit cache_credit_from_stat(const std::string& stat, uint64_t current, bool v1);

/// STRATA_MEM_CREDIT_INACTIVE=1: the inactive-only credit everywhere (the way back for testing).
bool mem_credit_inactive_only(const char* env_value);

/// The file bytes this process holds locked in RAM (the PLE table when its lock worked; 0 = none).  Locked pages are in
/// RssFile but on the unevictable list, not on the file lists: they are added back to own_evictable_mapped_bytes().
void set_locked_own_file_bytes(uint64_t bytes);
uint64_t locked_own_file_bytes();

/// This process's resident file mappings that sit on the file lists: RssFile from a /proc/self/status text, less the
/// locked bytes above (floor 0).  nullopt when the text has no readable RssFile line.
std::optional<uint64_t> own_evictable_mapped_bytes(const std::string& status_text, uint64_t locked_file_bytes);

/// The credit a cgroup limit gets.  Under a limit below MemTotal (a container: /proc/meminfo is the host's) `lists` less
/// this process's own evictable mappings (`own_mapped`; floor 0); when those cannot be told, `inactive`.  Otherwise
/// (no MemTotal, a limit at or above it, or STRATA_MEM_CREDIT_INACTIVE=1) `inactive`, as before.
uint64_t cache_credit_for_limit(const CacheCredit& credit, uint64_t limit, std::optional<uint64_t> mem_total, bool inactive_only,
                                std::optional<uint64_t> own_mapped);
/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

/// #3493: the conversation cache budget (MiB) --serve uses when --conversation-cache-mib was not given, from the RAM that is available
/// once the model is loaded: 40% of what is left above the parking floor, at most 8192 (four 100K-token agent conversations were
/// parked in 8 GiB in the report), in steps of 256, and 0 (off) when less than 1024 would be left to give.  Unknown telemetry: 0.
inline int64_t auto_conversation_cache_mib(std::optional<uint64_t> available, int64_t min_free_mib) {
    if (!available) return 0;
    const int64_t usable = (int64_t) (*available >> 20) - min_free_mib;
    if (usable <= 0) return 0;
    int64_t mib = usable * 2 / 5;
    if (mib > 8192) mib = 8192;
    mib -= mib % 256;
    return mib >= 1024 ? mib : 0;
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
