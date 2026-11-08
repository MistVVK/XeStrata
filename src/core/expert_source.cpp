// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/platform/thread_name.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// a 64-bit seek (as in pinned.cu): the 32-bit `fseek` wraps past 4 GiB, and the spelling differs per platform
#ifndef STRATA_FSEEK64
#if defined(_WIN32)
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif
#endif

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's counters can race memory.current, so bound them to charged usage first.  The whole clean page
    // cache is reclaimable, as MemAvailable counts active file pages too, less what the kernel keeps (upstream
    // 82c80ac4d: a container whose own reads left active cache was judged short of RAM, 1.2 GiB available instead of
    // about 11 in a 32 GiB group).
    uint64_t reclaimable = std::min(std::max(stat.file, stat.inactive_file), stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    reclaimable -= std::min(reclaimable / 2, limit / 64);

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err) {
    const size_t blobs = (size_t) (n_layers * n_expert);
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers || offsets.size() != blobs) {
        err = "append_stage_lend_regions: the plan does not match the geometry";
        return -1;
    }
    std::vector<std::pair<int32_t, int32_t>> sorted;
    for (const auto& region : regions) {   // the whole request is checked before the plan is touched
        for (const auto& pair : region)
            if (pair.first < 0 || pair.first >= n_layers || pair.second < 0 || pair.second >= n_expert) {
                err = "append_stage_lend_regions: a lend-region expert is outside the model";
                return -1;
            }
        sorted = region;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            err = "append_stage_lend_regions: a stage's lend region repeats an expert";
            return -1;
        }
    }
    int64_t kept = 0;
    for (const auto& region : regions) {
        for (const auto& pair : region) {
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if (offsets[index] != kNoCacheComplement) continue;   // already in the copy
            const uint64_t b = layer_blob_bytes[(size_t) pair.first];
            if (b > cap_bytes - bytes) break;                     // this stage's rest keeps the file fallback
            offsets[index] = bytes;
            bytes += b;
            ++kept;
        }
    }
    return kept;
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available) {
    waited = 0.0;
    if (!sample(available)) return false;
    double next_note = 0.0;
    while (available < need && waited < max_wait_s) {
        if (progress && waited >= next_note) {
            progress(waited, available);
            next_note = waited + note_s;
        }
        sleep(step_s);
        waited += step_s;
        if (!sample(available)) return false;
    }
    return available >= need;
}

}  // namespace detail

namespace {

// A whole file mapped read-only: the descriptor is the C runtime's on both systems, the view MapViewOfFile's on
// Windows and mmap's elsewhere.
#if defined(_WIN32)
int open_read_only(const std::string& path) { return ::_open(path.c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT); }
bool file_bytes(int fd, uint64_t& bytes) {
    struct _stat64 st{};
    if (::_fstat64(fd, &st) != 0 || st.st_size < 0) return false;
    bytes = (uint64_t) st.st_size;
    return true;
}
const uint8_t* map_read_only(int fd, uint64_t bytes) {
    HANDLE mapping = CreateFileMappingW((HANDLE) ::_get_osfhandle(fd), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) return nullptr;
    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, (SIZE_T) bytes);
    CloseHandle(mapping);   // the view keeps the mapping
    return (const uint8_t*) view;
}
void unmap_read_only(const uint8_t* view, uint64_t) { UnmapViewOfFile(view); }
void close_file(int fd) { ::_close(fd); }
#else
int open_read_only(const std::string& path) { return ::open(path.c_str(), O_RDONLY); }
bool file_bytes(int fd, uint64_t& bytes) {
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0) return false;
    bytes = (uint64_t) st.st_size;
    return true;
}
const uint8_t* map_read_only(int fd, uint64_t bytes) {
    void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_SHARED, fd, 0);
    return view == MAP_FAILED ? nullptr : (const uint8_t*) view;
}
void unmap_read_only(const uint8_t* view, uint64_t bytes) { munmap((void*) view, (size_t) bytes); }
void close_file(int fd) { ::close(fd); }
#endif

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false, file = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        } else if (key == "file") {   // optional (every v2 kernel has it)
            if (file) return false;
            file = true;
            stat.file = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

}  // namespace

namespace detail {

bool host_available_memory(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    m = HostMemory{};
#if defined(_WIN32)
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.available = (uint64_t) status.ullAvailPhys;
    return m.available > 0;
#else
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups(self_cgroup);
    if (!groups) return false;
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v1_path;   // the memory controller's group (cgroup v1), when there is no v2 line
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) {
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
            continue;
        }
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    if (!v2 && !v1_path.empty()) {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory controller's group and its ancestors.  An
        // unlimited group says a number near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone, as with no cgroup at all.
        const std::filesystem::path mroot = root / "memory";
        auto path = (mroot / v1_path.substr(v1_path.rfind('/', 0) == 0 ? 1 : 0)).lexically_normal();
        while (path.string().rfind(mroot.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                bytes = std::min(bytes, usage < cap ? cap - usage : 0);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
            path = path.parent_path();
        }
    }
    m.available = bytes;
    return true;
#endif
}

bool resident_pin_fits(uint64_t copy, uint64_t in_place, uint64_t headroom, uint64_t available, std::string& why) {
    const double gib = 1073741824.0;
    const uint64_t left = available > copy ? available - copy : 0;
    const bool fits = available > 0 && left >= headroom && left - headroom >= in_place;
    char b[512];
    std::snprintf(b, sizeof b,
                  fits ? "the %.2f GiB RAM copy is page-locked: of %.2f GiB available, %.2f GiB stays beside it for the "
                         "%.2f GiB still read from the model's files and the %.0f GiB headroom"
                       : "the %.2f GiB RAM copy stays pageable (#11): of %.2f GiB available, %.2f GiB stays beside it, "
                         "less than the %.2f GiB still read from the model's files and the %.0f GiB headroom (memory "
                         "page-locked there faulted the GPU's copies when the files paged; XESTRATA_RESIDENT_PIN=1 locks it "
                         "anyway)",
                  (double) copy / gib, (double) available / gib, (double) left / gib, (double) in_place / gib,
                  (double) headroom / gib);
    why = b;
    return fits;
}

namespace {
// -1: not set yet, read from STRATA_RAM_RESERVE_MIB at the first ram_reserve_bytes
std::atomic<int64_t> g_ram_reserve{-1};
}  // namespace

uint64_t ram_reserve_bytes() {
    int64_t v = g_ram_reserve.load();
    if (v < 0) {
        const char* e = strata::env_get("RAM_RESERVE_MIB");
        const long long mib = e != nullptr ? std::strtoll(e, nullptr, 10) : 0;
        int64_t expected = -1;
        g_ram_reserve.compare_exchange_strong(expected, mib > 0 ? (int64_t) mib << 20 : 0);
        v = g_ram_reserve.load();
    }
    return (uint64_t) v;
}

void set_ram_reserve_bytes(uint64_t bytes) { g_ram_reserve.store((int64_t) bytes); }

std::string page_cache_hint(const std::string& meminfo_text) {
    const auto kib = [&](const char* key) -> uint64_t {
        std::istringstream in(meminfo_text);
        std::string line;
        const std::string want = std::string(key) + ":";
        while (std::getline(in, line)) {
            std::istringstream f(line);
            std::string k;
            uint64_t v = 0;
            if (f >> k >> v && k == want) return v;
        }
        return 0;
    };
    const uint64_t cache = (kib("Cached") + kib("Buffers")) * 1024;
    if (cache < (1ull << 30)) return "";
    char msg[420];
    std::snprintf(msg, sizeof msg,
                  "; %.1f GiB of the RAM is file cache (earlier runs' model files, which Linux gives back on demand - "
                  "'free' showing little is normal). If the start still fails, free it with: "
                  "sync; echo 1 | sudo tee /proc/sys/vm/drop_caches   (or start the server with "
                  "XESTRATA_DROP_CACHE_ON_EXIT=1 so it releases its model files when it stops)",
                  (double) cache / 1073741824.0);
    return msg;
}

}  // namespace detail

namespace {

bool available_memory_bytes(uint64_t& bytes) {
    detail::HostMemory m;
    if (!detail::host_available_memory(m)) return false;
    bytes = m.available;
    return bytes > 0;
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

#if defined(__linux__)
namespace {
/// Bytes of [p, p + n) whose pages are in the page cache (mincore, page-granular); 0 when the call fails.
uint64_t resident_in(const uint8_t* p, uint64_t n) {
    if (n == 0) return 0;
    static thread_local std::vector<unsigned char> vec;
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095, e = ((uintptr_t) p + (uintptr_t) n + 4095) & ~(uintptr_t) 4095;
    vec.resize((size_t) ((e - a) / 4096));
    if (mincore((void*) a, (size_t) (e - a), vec.data()) != 0) return 0;
    uint64_t pages = 0;
    for (unsigned char c : vec) pages += c & 1;
    return std::min<uint64_t>(n, pages * 4096);
}
/// A cheap residency probe: the first, last and two inner pages; the bytes of the range when all four are cached, else 0.
uint64_t resident_sampled(const uint8_t* p, uint64_t n) {
    if (n == 0) return 0;
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095, e = ((uintptr_t) p + (uintptr_t) n - 1) & ~(uintptr_t) 4095;
    const uintptr_t pages = (e - a) / 4096;
    const uintptr_t at[4] = {a, a + (pages / 3) * 4096, a + (pages * 2 / 3) * 4096, e};
    for (uintptr_t x : at) {
        unsigned char v = 0;
        if (mincore((void*) x, 4096, &v) != 0 || (v & 1) == 0) return 0;
    }
    return n;
}
/// The whole range with pread (a short read continues); false on an error or an end of file.
bool pread_full(int fd, uint8_t* dst, uint64_t n, uint64_t off) {
    uint64_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, dst + done, (size_t) (n - done), (off_t) (off + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        done += (uint64_t) r;
    }
    return true;
}
/// The room the page cache has for the experts: in a memory-limited cgroup (v2; the tightest limit of the group and its
/// ancestors) the limit less what is not page cache (the group's memory.current less its "file" pages); otherwise
/// MemAvailable (free + reclaimable cache).  0 when unknown.
uint64_t gate_cache_room() {
    auto read_u64 = [](const std::string& path, uint64_t& v) {
        std::ifstream f(path);
        std::string s;
        if (!(f >> s) || s == "max") return false;
        v = std::strtoull(s.c_str(), nullptr, 10);
        return v > 0;
    };
    std::string rel;
    {
        std::ifstream g("/proc/self/cgroup");
        std::string line;
        while (std::getline(g, line))
            if (line.rfind("0::", 0) == 0) { rel = line.substr(3); break; }
    }
    uint64_t best = 0;
    if (!rel.empty()) {
        const std::string base = "/sys/fs/cgroup";
        for (std::string at = rel;;) {
            const std::string dir = base + (at == "/" ? std::string() : at);
            uint64_t mx = 0, cur = 0, file = 0;
            if (read_u64(dir + "/memory.max", mx) && read_u64(dir + "/memory.current", cur)) {
                std::ifstream st(dir + "/memory.stat");
                std::string k;
                uint64_t v = 0;
                while (st >> k >> v)
                    if (k == "file") { file = v; break; }
                const uint64_t used = cur > file ? cur - file : 0;
                const uint64_t room = mx > used ? mx - used : 0;
                if (best == 0 || room < best) best = room;
            }
            if (at.empty() || at == "/") break;
            const size_t cut = at.find_last_of('/');
            at = cut == 0 || cut == std::string::npos ? std::string("/") : at.substr(0, cut);
        }
    }
    if (best > 0) return best;
    std::ifstream info("/proc/meminfo");
    std::string key, unit;
    uint64_t value = 0;
    while (info >> key >> value >> unit)
        if (key == "MemAvailable:") return value * 1024;
    return 0;
}
}  // namespace
#endif

int FileExpertSource::spans(int64_t layer, int64_t expert, Span* out) const {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return 0;
    if (!role_ptr_.empty()) {
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) expert * per);
            const size_t f = (size_t) role_file_[i];
            out[r] = {p, per, f < io_fds_.size() ? io_fds_[f] : -1, (uint64_t) (p - maps_[f].base), at};
            at += per;
        }
        return 3;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return 0;
    out[0] = {b, layer_blob_bytes_[(size_t) layer], io_fds_.empty() ? -1 : io_fds_[0], (uint64_t) (b - base_), 0};
    return 1;
}

uint64_t FileExpertSource::cached_bytes(int64_t layer, int64_t expert, uint64_t& total, bool exact) const {
    total = 0;
#if defined(__linux__)
    Span sp[3];
    const int ns = spans(layer, expert, sp);
    uint64_t res = 0;
    for (int i = 0; i < ns; ++i) {
        total += sp[i].n;
        res += exact ? resident_in(sp[i].p, sp[i].n) : resident_sampled(sp[i].p, sp[i].n);
    }
    return res;
#else
    (void) layer; (void) expert; (void) exact;
    return 0;
#endif
}

uint64_t FileExpertSource::expert_bytes() const {
    uint64_t total = 0;
    for (uint64_t b : layer_blob_bytes_) total += b * (uint64_t) n_expert_;
    return total;
}

double FileExpertSource::cache_room_ratio() const {
#if defined(__linux__)
    const uint64_t total = expert_bytes(), room = gate_cache_room();
    return total == 0 || room == 0 ? 0.0 : (double) room / (double) total;
#else
    return 0.0;
#endif
}

FileExpertSource::IoCounters FileExpertSource::io_counters() const {
    IoCounters c;
    c.cached_bytes = io_cached_.load(std::memory_order_relaxed);
    c.uncached_bytes = io_uncached_.load(std::memory_order_relaxed);
    c.pread_bytes = io_pread_bytes_.load(std::memory_order_relaxed);
    c.pread_us = io_pread_us_.load(std::memory_order_relaxed);
    c.pread_n = io_pread_n_.load(std::memory_order_relaxed);
    c.pf_jobs = io_pf_jobs_.load(std::memory_order_relaxed);
    c.pf_demand = io_pf_demand_.load(std::memory_order_relaxed);
    c.pf_read_blobs = io_pf_blobs_.load(std::memory_order_relaxed);
    c.pf_resident_skips = io_pf_skips_.load(std::memory_order_relaxed);
    c.pf_dropped = io_pf_dropped_.load(std::memory_order_relaxed);
    c.pf_used = io_pf_used_.load(std::memory_order_relaxed);
    c.pf_unused = io_pf_unused_.load(std::memory_order_relaxed);
    c.pf_bytes = io_pf_bytes_.load(std::memory_order_relaxed);
    c.pf_used_bytes = io_pf_used_bytes_.load(std::memory_order_relaxed);
    c.crit_us = io_crit_us_.load(std::memory_order_relaxed);
    c.crit_n = io_crit_n_.load(std::memory_order_relaxed);
    return c;
}

void FileExpertSource::set_io_prefetch(bool on, int threads) {
    io_stop();
#if defined(__linux__)
    if (!on) return;
    if (base_ == nullptr || !direct_.empty()) return;   // unbuffered readers batch their own reads
    std::vector<std::string> paths = paths_;
    if (paths.empty()) {
        if (bin_path_.empty()) return;
        paths.push_back(bin_path_);
    }
    std::vector<int> fds;
    for (const std::string& path : paths) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            for (int f : fds) ::close(f);
            return;
        }
        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);   // read exactly the range asked for, no read-ahead beyond it
        fds.push_back(fd);
    }
    io_fds_ = std::move(fds);
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    }
    if (!warm_stamp_) warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    io_threads_ = threads > 0 ? threads : 8;
    if (const char* v = strata::env_get("IO_PF_AHEAD")) io_ahead_ = std::strtol(v, nullptr, 10) != 0;
    io_fill_ = role_ptr_.empty();   // experts.bin: fill the page cache, the kernels read the mapping
    gate_on_ = true;                // the decode gate comes with io prefetch; STRATA_IO_PF_GATE=0 turns it off
    if (const char* m = strata::env_get("IO_PF_GATE"); m != nullptr && m[0] != '\0') gate_on_ = std::strtol(m, nullptr, 10) != 0;
    if (const char* m = strata::env_get("IO_PF_GATE_LO")) gate_lo_ = std::clamp(std::strtod(m, nullptr), 0.0, 1.0);
    if (const char* m = strata::env_get("IO_PF_GATE_HI")) gate_hi_ = std::clamp(std::strtod(m, nullptr), gate_lo_, 1.0);
    if (const char* m = strata::env_get("IO_PF_GATE_USEFUL")) gate_min_useful_ = std::clamp(std::strtod(m, nullptr), 0.0, 1.0);
    if (io_fill_ && !pf_mark_) {
        pf_mark_.reset(new std::atomic<uint8_t>[(size_t) (n_layers_ * n_expert_)]);
        for (int64_t i = 0; i < n_layers_ * n_expert_; ++i) pf_mark_[(size_t) i].store(0, std::memory_order_relaxed);
    }
    io_pf_ = true;
    io_stats_ = true;
    io_quit_ = false;
    for (int i = 0; i < io_threads_; ++i) io_workers_.emplace_back([this] { io_worker(); });
#else
    (void) on; (void) threads;
    (void) io_threads_; (void) gate_lo_; (void) gate_hi_; (void) gate_min_useful_;   // the Linux path's settings
#endif
}

void FileExpertSource::io_stop() {
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        io_quit_ = true;
        io_q_.clear();
    }
    io_cv_.notify_all();
    for (std::thread& t : io_workers_)
        if (t.joinable()) t.join();
    io_workers_.clear();
#if defined(__linux__)
    for (int f : io_fds_) ::close(f);
#endif
    io_fds_.clear();
    io_pf_ = false;
    io_quit_ = false;
}

void FileExpertSource::io_enqueue(int64_t layer, int64_t expert, bool ahead) {
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (io_q_.size() >= 512) {   // the predictions run ahead of the reads: the oldest are the stalest
            io_q_.pop_front();
            io_pf_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        io_q_.emplace_back((int32_t) layer, (int32_t) expert | (ahead ? kIoAhead : 0));
    }
    io_cv_.notify_one();
}

// An I/O thread: takes a (layer, expert), the newest first, and skips it when its pages are cached or it is already
// staged.  The GGUF in place reads the whole blob into a stage buffer (copy_from_files); experts.bin reads what is not
// cached into a scratch buffer, so its pages are cached when the kernels read the mapping.
void FileExpertSource::io_worker() {
    strata::platform::name_this_thread("xs-io");
    for (;;) {
        int32_t layer, expert;
        bool ahead;
        {
            std::unique_lock<std::mutex> lk(io_mu_);
            io_cv_.wait(lk, [&] { return io_quit_ || !io_q_.empty(); });
            if (io_quit_) return;
            layer = io_q_.back().first;   // the newest first: it is the layer that comes next
            expert = io_q_.back().second & ~kIoAhead;
            ahead = (io_q_.back().second & kIoAhead) != 0;
            io_q_.pop_back();
        }
        io_pf_jobs_.fetch_add(1, std::memory_order_relaxed);
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
        if (complement_ready_ && resident_blob(index) != nullptr) continue;
        if (!override_.empty() && override_[index] != nullptr) continue;
#if defined(__linux__)
        if (io_fill_) {
            static thread_local std::vector<uint8_t> scratch;
            Span sp[3];
            const int ns = spans(layer, expert, sp);
            uint64_t got = 0;
            for (int i = 0; i < ns; ++i) {
                if (sp[i].fd < 0 || resident_sampled(sp[i].p, sp[i].n) >= sp[i].n) continue;
                if (scratch.size() < sp[i].n) scratch.resize((size_t) sp[i].n);
                const auto t0 = std::chrono::steady_clock::now();
                if (pread_full(sp[i].fd, scratch.data(), sp[i].n, sp[i].off)) {
                    io_pread_bytes_.fetch_add(sp[i].n, std::memory_order_relaxed);
                    io_pread_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                                           std::memory_order_relaxed);
                    io_pread_n_.fetch_add(1, std::memory_order_relaxed);
                    got += sp[i].n;
                }
            }
            if (got > 0) {
                io_pf_blobs_.fetch_add(1, std::memory_order_relaxed);
                io_pf_bytes_.fetch_add(got, std::memory_order_relaxed);
                if (pf_mark_ && ahead) pf_mark_[index].store(1, std::memory_order_relaxed);   // scored at the demand
            } else {
                io_pf_skips_.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
#endif
        uint64_t total = 0;
        if (cached_bytes(layer, expert, total, false) >= total) {
            io_pf_skips_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        size_t v = 0;
        bool fill = false;
        uint8_t* dst = nullptr;
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            if (stage_blob_ == 0) continue;
            const bool have = claim_stage((int64_t) layer * n_expert_ + expert, v, fill, /*ahead=*/true);
            if (have || !fill) continue;   // already staged or being read, or no buffer
            stage_pf_[v] = ahead ? 1 : 0;
            dst = stage_buf_[v].get();
        }
        const bool ok = copy_from_files(layer, expert, dst);
        io_pf_blobs_.fetch_add(1, std::memory_order_relaxed);
        if (ok) {
            file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
            io_pf_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            stage_busy_[v] = 0;
            if (!ok) {
                stage_of_.erase(stage_key_[v]);
                stage_key_[v] = -1;
                stage_pf_[v] = 0;
            }
        }
        stage_cv_.notify_all();
    }
}

// The decode gate: whether this thread reads for the decode (the dispatch's thread, and the fetch threads it starts).
namespace { thread_local bool tl_decode = false; }
FileExpertSource::DecodeScope::DecodeScope() : prev(tl_decode) { tl_decode = true; }
FileExpertSource::DecodeScope::~DecodeScope() { tl_decode = prev; }
bool FileExpertSource::in_decode() { return tl_decode; }

// Once a decode window: the share of the window's demand bytes the page cache held (an EWMA, reported only).
void FileExpertSource::gate_tick() {
    if (!gate_on_) return;
    const uint64_t c = gate_c_.load(std::memory_order_relaxed), u = gate_u_.load(std::memory_order_relaxed);
    const uint64_t dc = c - gate_prev_c_, du = u - gate_prev_u_;
    gate_prev_c_ = c;
    gate_prev_u_ = u;
    if (!gate_open_.load(std::memory_order_relaxed)) ++gate_closed_w_;
    if (dc + du < (64ull << 20)) return;   // too few bytes read this window to judge
    const double h = (double) dc / (double) (dc + du);
    gate_ewma_ = gate_ewma_ < 0.0 ? h : 0.8 * gate_ewma_ + 0.2 * h;
}

// Every 16 decode windows: the page cache's room for the experts, and whether the read-ahead pays (the share of its
// bytes a layer then used, 64 MB at least read ahead to judge).  Upstream measured a closed gate's case: an 18 GiB RAM
// tier read 2.5 GB ahead in a run, 0% used, and decoded 9..16% slower with the read-ahead than without.  Closed for not
// paying, the gate opens again for a probe every 256 windows.
void FileExpertSource::gate_capacity() {
#if defined(__linux__)
    if (!gate_on_ || (gate_win_++ % 16) != 0) return;
    const uint64_t total = expert_bytes(), room = gate_cache_room();
    if (total == 0 || room == 0) return;   // unknown: the state stays
    gate_ratio_ = (double) room / (double) total;
    const uint64_t pf = io_pf_bytes_.load(std::memory_order_relaxed), used = io_pf_used_bytes_.load(std::memory_order_relaxed);
    const uint64_t dpf = pf - gate_prev_pf_, dused = used - gate_prev_used_;
    gate_prev_pf_ = pf;
    gate_prev_used_ = used;
    if (dpf >= (64ull << 20)) gate_useful_ = (double) dused / (double) dpf;
    const bool open = gate_open_.load(std::memory_order_relaxed);
    const bool room_ok = open ? gate_ratio_ >= gate_lo_ : gate_ratio_ > gate_hi_;
    bool want = room_ok;
    if (room_ok && open && gate_win_ > 16 && (dpf < (128ull << 20) || gate_useful_ < gate_min_useful_)) {
        want = false;
        gate_useless_ = true;
        gate_useless_since_ = gate_win_;
    } else if (room_ok && !open && gate_useless_) {
        want = gate_win_ - gate_useless_since_ >= 256;   // the probe
        if (want) gate_useless_ = false;
    }
    if (!room_ok) gate_useless_ = false;
    if (want != open) { gate_open_.store(want, std::memory_order_relaxed); ++gate_flips_; }
#endif
}


bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";
    if (layout.native && !gguf_.empty() && !std::filesystem::exists(path)) {
        // CS-T: no experts.bin - the model's GGUF shards, read in place
        blobs_ = (int64_t) blob_count;
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        layer_offsets_ = std::move(layer_offsets);
        layer_blob_bytes_ = std::move(layer_blob_bytes);
        if (!open_gguf(err)) { close(); return false; }
        record_inputs(path, gguf_, layout, true);
        return true;
    }

    const int fd = open_read_only(path);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    uint64_t have = 0;
    if (!file_bytes(fd, have)) { close_file(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (have != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) have, (unsigned long long) want);
        close_file(fd);
        err = buf;
        return false;
    }
    const uint8_t* view = map_read_only(fd, want);
    if (view == nullptr) { close_file(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    bin_path_ = path;
    base_ = view;
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    record_inputs(path, gguf_, layout, false);
    return true;
}

bool FileExpertSource::register_on(int device, std::string& err) {
    if (registered_on(device) || (!complement_ready_ && xstage_ == nullptr)) return true;
    // layer_devices: a part in the device's own host USM needs no registration, and another maker's part holds
    // layers this device does not run
    const bool by_device = !complement_chunks_.empty() && complement_chunks_[0].owner >= 0;
    std::vector<std::pair<void*, size_t>> blocks;
    if (!complement_chunks_.empty()) {
        for (const ComplementChunk& c : complement_chunks_)
            if (!by_device || (c.owner != device && strata::gpu::same_backend(c.owner, device)))
                blocks.emplace_back(c.host, (size_t) (c.end - c.start));
    } else if (complement_arena_ != nullptr && complement_bytes_ > 0)
        blocks.emplace_back(complement_arena_, (size_t) complement_bytes_);
    // the exchange buffers: another maker's host USM is not registered here, and their copies stay pageable
    if (xstage_ != nullptr && xstage_cap_ > 0 && !(xstage_pinned_ && !strata::gpu::same_backend(xstage_dev_, device)))
        blocks.emplace_back(xstage_, (size_t) xstage_cap_ * (size_t) xstage_blob_);
    std::vector<void*> regs;
    for (const auto& [p, n] : blocks) {
        if (!strata::gpu::host_register_on(device, p, n)) {
            for (void* r : regs) strata::gpu::host_unregister_on(device, r);
            err = std::string("registering the RAM copy on GPU ") + std::to_string(device) + ": " +
                  strata::gpu::last_error();
            return false;
        }
        regs.push_back(p);
    }
    stage_reg_.emplace_back(device, std::move(regs));
    return true;
}

bool FileExpertSource::pin_by_device(std::vector<uint64_t>& offsets, std::vector<ComplementChunk>& chunks,
                                     std::string& note) const {
    if (layer_dev_.empty()) return false;
    const int here = current_device();
    auto owner_of = [&](size_t i) {
        const size_t l = i / (size_t) n_expert_;
        return l < layer_dev_.size() && layer_dev_[l] >= 0 ? layer_dev_[l] : here;
    };
    std::vector<int> owners;
    for (size_t i = 0; i < offsets.size(); ++i)
        if (offsets[i] != kNoComplement && std::find(owners.begin(), owners.end(), owner_of(i)) == owners.end())
            owners.push_back(owner_of(i));
    if (owners.size() < 2) return false;
    std::vector<size_t> order;   // the blobs in their offsets' order, which each device's part keeps
    for (size_t i = 0; i < offsets.size(); ++i)
        if (offsets[i] != kNoComplement) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return offsets[a] < offsets[b]; });
    std::vector<uint64_t> packed(offsets.size(), kNoComplement);
    std::vector<ComplementChunk> got;
    uint64_t at = 0;
    bool ok = true;
    for (size_t k = 0; k < owners.size() && ok; ++k) {
        const uint64_t start = at;
        for (const size_t i : order)
            if (owner_of(i) == owners[k]) {
                packed[i] = at;
                at += layer_blob_bytes_[i / (size_t) n_expert_];
            }
        const OnDevice on(owners[k]);
        void* h = nullptr;
        void* d = nullptr;
        ok = strata::gpu::alloc_host(&h, (size_t) (at - start));
        if (h != nullptr) got.push_back({start, at, (uint8_t*) h, nullptr, owners[k]});
        ok = ok && strata::gpu::device_pointer(&d, h) && d != nullptr;
        if (ok) got.back().device = (const uint8_t*) d;
    }
    if (!ok) {
        std::fprintf(stderr, "FileExpertSource: a part of the page-locked copy for GPU %d was refused (%s): one copy\n",
                     got.empty() ? owners[0] : got.back().owner, strata::gpu::last_error());
        for (const ComplementChunk& c : got) (void) strata::gpu::free(c.host);
        return false;
    }
    chunks.swap(got);
    offsets.swap(packed);
    note = "page-locked and mapped, one part per GPU's runtime (" + std::to_string(chunks.size()) +
           ": each GPU's layers in its own host USM)";
    return true;
}

bool FileExpertSource::registered_on(int device) const {
    for (const auto& o : stage_reg_)
        if (o.first == device) return true;
    return false;
}

void FileExpertSource::close() {
    io_stop();
    if (complement_arena_ || xstage_) strata::gpu::device_sync();
    for (auto& o : stage_reg_) {   // before the memory goes, and after that GPU's copies from it
        const OnDevice on(o.first);
        (void) strata::gpu::device_sync();
        for (void* p : o.second) strata::gpu::host_unregister_on(o.first, p);
    }
    stage_reg_.clear();
    inputs_.clear();
    if (!complement_chunks_.empty()) {
        for (const ComplementChunk& c : complement_chunks_) (void) strata::gpu::free(c.host);
        complement_chunks_.clear();
    } else if (complement_arena_ != nullptr) {
        if (complement_pinned_ && !complement_partial_) (void) strata::gpu::free(complement_arena_);
        else {
            if (complement_partial_) (void) strata::gpu::host_unregister(complement_arena_);
            if (complement_locked_ > 0)
                strata::platform::unlock_resident((uint8_t*) complement_arena_ + complement_lock_off_, complement_locked_);
            std::free(complement_arena_);
        }
    }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) strata::gpu::free(xstage_);
        else std::free(xstage_);
    }
    xstage_ = nullptr;
    xstage_pinned_ = false;
    xstage_cap_ = 0;
    xstage_blob_ = 0;
    override_.clear();
    staged_.clear();
    exchange_storage_.clear();
    exchanges_ = 0;
    file_reads_.store(0);
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_partial_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    if (!maps_.empty()) {
        for (Map& m : maps_) {
            if (m.base != nullptr) unmap_read_only(m.base, m.bytes);
            if (m.fd >= 0) close_file(m.fd);
        }
        maps_.clear();
        base_ = nullptr;   // one of the views above
    }
    role_ptr_.clear();
    role_bytes_.clear();
    role_file_.clear();
    paths_.clear();
    direct_.clear();
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_used_.clear();
        stage_busy_.clear();
        stage_pf_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
        stage_seq_ = 0;
        epoch_ = 0;
        last_layer_ = -1;
        stage_grew_ = false;
    }
    ram_reads_.store(0);
    warm_stamp_.reset();
    warm_hits_.store(0);
    warm_count_.store(0);
    pf_mark_.reset();
    zc_keys_.clear();
    zc_count_.store(0);
    bin_path_.clear();
    io_fill_ = false;
    io_stats_ = false;
    io_ahead_ = true;
    io_cached_.store(0); io_uncached_.store(0); io_pread_bytes_.store(0); io_pread_us_.store(0); io_pread_n_.store(0);
    io_pf_jobs_.store(0); io_pf_demand_.store(0); io_pf_blobs_.store(0); io_pf_skips_.store(0); io_pf_dropped_.store(0);
    io_pf_used_.store(0); io_pf_unused_.store(0); io_pf_bytes_.store(0); io_pf_used_bytes_.store(0);
    io_crit_us_.store(0); io_crit_n_.store(0);
    gate_on_ = false;
    gate_open_.store(true);
    gate_c_.store(0); gate_u_.store(0);
    gate_prev_c_ = gate_prev_u_ = gate_prev_pf_ = gate_prev_used_ = 0;
    gate_win_ = gate_useless_since_ = gate_closed_w_ = gate_flips_ = 0;
    gate_ewma_ = gate_ratio_ = -1.0;
    gate_useful_ = 1.0;
    gate_useless_ = false;
    file_read_bytes_.store(0);
    file_blob_bytes_.store(0);
    file_us_.store(0);
    if (base_ != nullptr) unmap_read_only(base_, mapped_bytes_);
    if (fd_ >= 0) close_file(fd_);
    fd_ = -1;
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}


bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

// ================================ CS-T: THE GGUF SHARDS IN PLACE ================================
//
// A native pack without experts.bin: every file native_experts.txt names is mapped (MapViewOfFile / mmap, no
// flag - the same retention argument as experts.bin above), and an expert's blob [gate rows | up rows | down rows]
// is three slices of three tensors, possibly in two shards (UD-Q4_K_XL's layer 11).  Nothing is read at open; a
// blob is assembled when it is asked for (`blob`, into a small pool of buffers) or copied where it is needed
// (`copy_blob`: the RAM copy, the prompt path's pinned stager buffers).
bool FileExpertSource::open_gguf(std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!check_experts_gguf(gguf_, lay, err)) { err = "FileExpertSource: " + err; return false; }
    const size_t cut = gguf_.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1);
    std::map<std::string, size_t> index;
    role_ptr_.assign((size_t) (3 * n_layers_), nullptr);
    role_bytes_.assign((size_t) (3 * n_layers_), 0);
    role_file_.assign((size_t) (3 * n_layers_), 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.down_off - fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * l + r);
            const std::string path = lay.gguf_file.size() > i && !lay.gguf_file[i].empty() ? dir + lay.gguf_file[i]
                                                                                            : gguf_;
            auto it = index.find(path);
            if (it == index.end()) {
                Map m;
                const int fd = open_read_only(path);
                uint64_t size = 0;
                if (fd < 0 || !file_bytes(fd, size) || size == 0) {
                    if (fd >= 0) close_file(fd);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                const uint8_t* view = map_read_only(fd, size);
                if (view == nullptr) { close_file(fd); err = "FileExpertSource: cannot map " + path; return false; }
                m.fd = fd;
                m.bytes = size;
                m.base = view;
                maps_.push_back(m);
                paths_.push_back(path);
                it = index.emplace(path, maps_.size() - 1).first;
            }
            const Map& m = maps_[it->second];
            role_file_[i] = (int) it->second;
            const uint64_t at = lay.gguf_off[i], bytes = per[r] * (uint64_t) n_expert_;
            if (at > m.bytes || bytes > m.bytes - at) {   // check_experts_gguf proved it; the mapping must agree
                err = "FileExpertSource: an expert span runs past the end of " + path;
                return false;
            }
            role_ptr_[i] = m.base + (size_t) at;
            role_bytes_[i] = per[r];
        }
    }
    base_ = maps_.front().base;       // "opened"; mapped_blob answers nullptr in this mode
    warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    if (!direct_.empty()) {   // #286: from the drive; a failed read falls back to the mapping below
        const Fill f{0, layer, expert, dst};
        if (read_direct(&f, 1)) return true;
    }
#if defined(__linux__)
    if (io_stats_) {   // what the page cache held of the blob (and, in the decode, for the gate)
        Span sp[3];
        const int ns = spans(layer, expert, sp);
        for (int i = 0; i < ns; ++i) {
            const uint64_t res = resident_in(sp[i].p, sp[i].n);
            io_cached_.fetch_add(res, std::memory_order_relaxed);
            io_uncached_.fetch_add(sp[i].n - res, std::memory_order_relaxed);
            if (gate_on_ && tl_decode) {
                gate_c_.fetch_add(res, std::memory_order_relaxed);
                gate_u_.fetch_add(sp[i].n - res, std::memory_order_relaxed);
            }
        }
    }
#endif
    if (!role_ptr_.empty()) {
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            const uint8_t* src = role_ptr_[i] + (size_t) ((uint64_t) expert * per);
#if !defined(_WIN32)
            // read, not copied from the mapping: a slice of megabytes in one request instead of a fault a page (the
            // resident copy of GLM-5.3-Flash's 88 GiB from the drive: 0.65 GB/s through the mapping, 2026-10-19)
            // with the I/O path on, through its own descriptor: POSIX_FADV_RANDOM reads exactly the slice, where the
            // mapping's descriptor would read ahead past it (and that read-ahead advice is the mapping's as well)
            const Map& m = maps_[(size_t) role_file_[i]];
            const int fd = (size_t) role_file_[i] < io_fds_.size() ? io_fds_[(size_t) role_file_[i]] : m.fd;
            if (fd >= 0) {
                uint64_t done = 0;
                while (done < per) {
                    const ssize_t n = ::pread(fd, dst + at + done, (size_t) (per - done),
                                              (off_t) ((uint64_t) (src - m.base) + done));
                    if (n <= 0) break;
                    done += (uint64_t) n;
                }
                if (done == per) { at += per; continue; }
            }
#endif
            std::memcpy(dst + at, src, (size_t) per);
            at += per;
        }
        return true;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return false;
    std::memcpy(dst, b, (size_t) layer_blob_bytes_[(size_t) layer]);
    return true;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during `kStageAge` layers (begin_layer) or 256 assemblies, whichever comes
// first, and never while it is being filled: the pool computes a layer's misses before it starts the next, and a
// fill (the GPU cache at startup, an adaptive swap, a helper GPU) copies the blob right away.
//
// `claim_stage` finds or reserves the buffer of `key` (stage_mu_ held): true when the blob is already there (or
// being filled by another thread - the caller then waits), false when the caller must fill buffer `v`.
bool FileExpertSource::claim_stage(int64_t key, size_t& v, bool& fill, bool ahead) {
    constexpr uint64_t kStageSeq = 256;
    const uint64_t seq = ++stage_seq_;
    auto it = stage_of_.find(key);
    fill = false;
    if (it != stage_of_.end()) {
        v = it->second;
        stage_epoch_[v] = epoch_;
        stage_used_[v] = seq;
        if (!ahead && stage_pf_[v]) {   // a layer asked for what was read ahead
            stage_pf_[v] = 0;
            io_pf_used_.fetch_add(1, std::memory_order_relaxed);
            io_pf_used_bytes_.fetch_add(layer_blob_bytes_[(size_t) (key / n_expert_)], std::memory_order_relaxed);
        }
        return true;
    }
    v = stage_buf_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < stage_buf_.size(); ++i)
        if (!stage_busy_[i] && (stage_epoch_[i] + kStageAge <= epoch_ || stage_used_[i] + kStageSeq <= seq) &&
            stage_used_[i] < oldest) {
            oldest = stage_used_[i];
            v = i;
        }
    if (v == stage_buf_.size()) {
        stage_buf_.emplace_back(new (std::nothrow) uint8_t[(size_t) stage_blob_]);
        if (!stage_buf_.back()) { stage_buf_.pop_back(); return false; }
        stage_key_.push_back(-1);
        stage_epoch_.push_back(0);
        stage_used_.push_back(0);
        stage_busy_.push_back(0);
        stage_pf_.push_back(0);
        if (stage_buf_.size() == 512 && !stage_grew_) {
            stage_grew_ = true;
            std::fprintf(stderr, "FileExpertSource: %zu blobs assembled from the GGUF are in use at once (%.2f GiB)\n",
                         stage_buf_.size(), (double) stage_buf_.size() * (double) stage_blob_ / 1073741824.0);
        }
    } else {
        stage_of_.erase(stage_key_[v]);
        if (stage_pf_[v]) { stage_pf_[v] = 0; io_pf_unused_.fetch_add(1, std::memory_order_relaxed); }   // never asked for
    }
    stage_key_[v] = key;
    stage_epoch_[v] = epoch_;
    stage_used_[v] = seq;
    stage_busy_[v] = 1;
    stage_of_[key] = v;
    fill = true;
    return false;
}

// Fills buffer `v` (reserved by claim_stage) outside the lock, then publishes it.
bool FileExpertSource::fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = copy_from_files(layer, expert, dst);
    publish_stage(v, layer, ok, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

void FileExpertSource::publish_stage(size_t v, int64_t layer, bool ok, double us) {
    file_us_.fetch_add((uint64_t) us, std::memory_order_relaxed);
    if (ok) {
        file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        file_blob_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_busy_[v] = 0;
        if (!ok) {
            stage_of_.erase(stage_key_[v]);
            stage_key_[v] = -1;
        }
    }
    stage_cv_.notify_all();
}

const uint8_t* FileExpertSource::staged_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    size_t v = 0;
    bool fill = false;
    uint8_t* dst = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    auto crit = [&] {
        if (!io_pf_) return;
        io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    };
    {
        std::unique_lock<std::mutex> lk(stage_mu_);
        const bool have = claim_stage(key, v, fill);
        if (!have && !fill) return nullptr;
        dst = stage_buf_[v].get();
        if (have) {
            // another thread (a prefetch, a read ahead, the adaptive tier) is filling it: wait for that
            stage_cv_.wait(lk, [&] { return !stage_busy_[v] || stage_key_[v] != key; });
            if (stage_key_[v] != key) return nullptr;   // its fill failed
            lk.unlock();
            crit();
            return dst;
        }
    }
    const bool ok = fill_stage(v, layer, expert, dst);
    crit();
    return ok ? dst : nullptr;
}

void FileExpertSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    if (n <= 0 || layer < 0 || layer >= n_layers_) return;
    if (io_fill_ && io_pf_ && direct_.empty()) {   // experts.bin: the layer's uncached blobs go to the I/O threads; the
        for (int64_t i = 0; i < n; ++i) {          // kernels that reach one before it lands wait on its pages
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (pf_mark_ && pf_mark_[index].exchange(0, std::memory_order_relaxed)) {   // read ahead, now asked for
                io_pf_used_.fetch_add(1, std::memory_order_relaxed);
                io_pf_used_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
            }
            if (complement_ready_ && resident_blob(index) != nullptr) continue;
            if (!override_.empty() && override_[index] != nullptr) continue;
            uint64_t total = 0;
            if (cached_bytes(layer, e, total, false) >= total) continue;
            io_pf_demand_.fetch_add(1, std::memory_order_relaxed);
            io_enqueue(layer, e, false);
        }
        return;
    }
    if (!staged()) return;
    std::vector<Fill> todo;
    const auto t0 = std::chrono::steady_clock::now();
    claim_todo(layer, experts, n, todo, false);
    fill_many(todo);
    if (io_pf_ && !todo.empty()) {
        io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    }
}

void FileExpertSource::claim_todo(int64_t layer, const int64_t* experts, int64_t n, std::vector<Fill>& todo, bool zc) {
    // zero-copy (upstream 149, STRATA_COLD_OVERLAP=4): the GGUF in place, mapped and buffered - an expert the page
    // cache holds whole is computed from the mapping (its three tensors' rows), not assembled into a stage buffer
    // first.  Decided here, before the lock.
    zc_keys_.clear();
    std::vector<char> zc_ok;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (zc && !role_ptr_.empty() && direct_.empty() && lay.native && lay.fmt[(size_t) layer].tail_off == 0) {
        zc_ok.assign((size_t) n, 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && resident_blob(index) != nullptr) continue;   // in the RAM copy
            if (!override_.empty() && override_[index] != nullptr) continue;
            uint64_t total = 0;
            const uint64_t c = cached_bytes(layer, e, total, true);
            zc_ok[(size_t) i] = total > 0 && c >= total ? 1 : 0;   // no residency query (not Linux): never
        }
    }
    std::lock_guard<std::mutex> lk(stage_mu_);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ && resident_blob(index) != nullptr)
            continue;                                     // in the RAM copy
        if (!override_.empty() && override_[index] != nullptr) continue;
        if (!zc_ok.empty() && zc_ok[(size_t) i] && stage_of_.find(layer * n_expert_ + e) == stage_of_.end()) {
            zc_keys_.push_back(layer * n_expert_ + e);   // read from the mapping by the kernels themselves
            continue;
        }
        size_t v = 0;
        bool fill = false;
        if (!claim_stage(layer * n_expert_ + e, v, fill) && fill) {
            todo.push_back({v, layer, e, stage_buf_[v].get()});
            if (warm_stamp_) {
                const uint32_t s = warm_stamp_[index].load(std::memory_order_relaxed);
                if (s != 0 && (uint64_t) s + 3 >= epoch_ + 1) warm_hits_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

int64_t FileExpertSource::prefetch_begin(int64_t layer, const int64_t* experts, int64_t n) {
    if (n <= 0 || layer < 0 || layer >= n_layers_) return 0;
    if (!staged() || (io_fill_ && io_pf_)) {
        prefetch(layer, experts, n);
        return 0;
    }
    std::vector<Fill> todo;
    const auto t0 = std::chrono::steady_clock::now();
    claim_todo(layer, experts, n, todo, true);
    fill_many(todo);
    if (io_pf_ && !todo.empty()) {
        io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
}

bool FileExpertSource::zc_roles(int64_t layer, int64_t expert, const uint8_t** p) {
    const int64_t key = layer * n_expert_ + expert;
    if (std::find(zc_keys_.begin(), zc_keys_.end(), key) == zc_keys_.end()) return false;
    for (int r = 0; r < 3; ++r) {
        const size_t i = (size_t) (3 * layer + r);
        p[r] = role_ptr_[i] + (size_t) ((uint64_t) expert * role_bytes_[i]);
    }
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    if (complement_ready_) file_reads_.fetch_add(1, std::memory_order_relaxed);
    ++reads_;
    zc_count_.fetch_add(1, std::memory_order_relaxed);
    if (io_stats_) io_cached_.fetch_add(bytes, std::memory_order_relaxed);
    if (gate_on_ && tl_decode) gate_c_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

void FileExpertSource::prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) {
    if (direct_.empty() || pairs == nullptr || n <= 0) return;   // unbuffered only: the mapped fill stays as it was
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < std::min<int64_t>(n, 64); ++i) {
            const int64_t l = pairs[i].first, e = pairs[i].second;
            if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(l * n_expert_ + e, v, fill) && fill) todo.push_back({v, l, e, stage_buf_[v].get()});
        }
    }
    fill_many(todo);
}

bool FileExpertSource::advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
    if (base_ == nullptr || !direct_.empty() || !strata::platform::read_ahead_enabled()) return false;
    for (int64_t i = 0; pairs != nullptr && i < n; ++i) {
        const int64_t l = pairs[i].first, e = pairs[i].second;
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
        if (role_ptr_.empty()) {   // experts.bin: the blob is one contiguous range
            if (const uint8_t* b = mapped_blob(l, e)) strata::platform::advise_willneed(b, layer_blob_bytes_[(size_t) l]);
            continue;
        }
        for (int r = 0; r < 3; ++r) {
            const size_t k = (size_t) (3 * l + r);
            strata::platform::advise_willneed(role_ptr_[k] + (size_t) ((uint64_t) e * role_bytes_[k]), role_bytes_[k]);
        }
    }
    return true;
}

void FileExpertSource::fill_many(const std::vector<Fill>& todo) {
    if (todo.empty()) return;
    if (!direct_.empty()) {
        // #286: overlapped batches - every role window of 16 blobs in flight at once; a big batch (the profile
        // fill) on up to 4 threads, a decode layer's few misses on this one
        std::atomic<size_t> next{0};
        auto work = [&] {
            for (size_t at; (at = next.fetch_add(16)) < todo.size();) {
                const size_t k = std::min<size_t>(16, todo.size() - at);
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = read_direct(todo.data() + at, k);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (size_t i = at; i < at + k; ++i) {
                    const Fill& f = todo[i];
                    // a failed batch: each blob again on its own (copy_from_files falls back to the mapping)
                    if (ok) publish_stage(f.v, f.layer, true, us / (double) k);
                    else (void) fill_stage(f.v, f.layer, f.e, f.dst);
                }
            }
        };
        const size_t nt = std::min<size_t>(4, (todo.size() + 15) / 16);
        std::vector<std::thread> th;
        for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
        return;
    }
    // the page faults of a mapped read are one outstanding request each: several threads keep the SSD's queue full
    std::atomic<size_t> next{0};
    const bool dec = tl_decode;   // the decode gate counts the reads of the threads a decode layer starts
    auto work = [&] {
        tl_decode = dec;
        for (size_t i; (i = next.fetch_add(1)) < todo.size();)
            (void) fill_stage(todo[i].v, todo[i].layer, todo[i].e, todo[i].dst);
    };
    const size_t nt = std::min<size_t>(todo.size(), (size_t) fetch_threads_);
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}

bool FileExpertSource::set_unbuffered(uint64_t ram_bytes, std::string& why) {
    (void) ram_bytes;
    why = "through the file cache (not Windows)";
    return false;
}

bool FileExpertSource::read_direct(const Fill* fills, size_t n) const {
    (void) fills; (void) n;
    return false;
}

void FileExpertSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    if ((role_ptr_.empty() && !io_pf_) || n <= 0 || layer < 0 || layer >= n_layers_) return;
    uint32_t stamp;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stamp = (uint32_t) epoch_ + 1;
    }
    for (int64_t j = 0; j < n; ++j) {
        const int64_t e = experts[j];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
            continue;                                                    // in the RAM copy
        if (warm_stamp_) warm_stamp_[index].store(stamp, std::memory_order_relaxed);
        warm_count_.fetch_add(1, std::memory_order_relaxed);
        if (io_pf_ && direct_.empty() && gate_open()) {   // read it ahead on the I/O threads (instead of an madvise hint)
            if (io_ahead_) io_enqueue(layer, e, true);
            continue;
        }
        if (role_ptr_.empty()) continue;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            strata::platform::advise_willneed(role_ptr_[i] + (size_t) ((uint64_t) e * role_bytes_[i]), role_bytes_[i]);
        }
    }
}

RouterLookahead::~RouterLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err) {
    if (src == nullptr || !src->warms()) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (!strata::kernels::cpu::cpu_avx2_ok()) { err = "RouterLookahead: the router needs AVX2"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] { run(); });
    return true;
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::run() {
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int64_t> want;
    for (;;) {
        int64_t layer, nt;
        const int32_t* host_res;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || pending_; });
            if (quit_) return;
            pending_ = false;
            busy_ = true;
            layer = layer_;
            nt = n_tok_;
            host_res = host_res_;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t layer0 = layer;
        const int depth = src_->read_ahead_open() ? depth_ : 1;   // the decode gate closed: the next layer only
        for (int dj = 0; dj < depth && layer0 + dj < (int64_t) routers_.size(); ++dj) {
        layer = layer0 + dj;
        want.clear();
        strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                  x_.data(), (int) nt, logits.data());
        for (int64_t t = 0; t < nt; ++t) {
            const float* lt = logits.data() + (size_t) (t * n_expert_);
            for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
            std::partial_sort(order.begin(), order.begin() + k_, order.end(),
                              [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
            for (int j = 0; j < k_; ++j) {
                const int64_t e = order[(size_t) j];
                if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;   // on the GPU
                if (std::find(want.begin(), want.end(), e) == want.end()) want.push_back(e);
            }
        }
        src_->warm(layer, want.data(), (int64_t) want.size());
        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        }
        busy_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
    }
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) ids;
    (void) k;
    if (!staged() && !io_pf_) return;
    bool new_window = false;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        if (layer != last_layer_) {
            ++epoch_;
            new_window = layer < last_layer_;
            last_layer_ = layer;
        }
    }
    if (new_window && gate_on_ && tl_decode) { gate_capacity(); gate_tick(); }
}

bool FileExpertSource::transient(int64_t layer, int64_t expert) const {
    if (!staged() || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && resident_blob(index) != nullptr)
        return false;
    return override_.empty() || override_[index] == nullptr;
}

double FileExpertSource::cached_share(int64_t samples) const {
#if defined(__linux__)
    if (role_ptr_.empty() || samples <= 0 || n_layers_ <= 0 || n_expert_ <= 0) return -1.0;
    // an even spread over the (layer, expert) grid, every page of each of the expert's three slices through mincore
    const int64_t total = (int64_t) n_layers_ * (int64_t) n_expert_;
    const int64_t n = std::min<int64_t>(samples, total);
    uint64_t pages = 0, resident = 0;
    std::vector<unsigned char> vec;
    for (int64_t s = 0; s < n; ++s) {
        const int64_t idx = std::min<int64_t>(total - 1, (s * total) / n + (s * 7919) % std::max<int64_t>(1, total / n));
        const int64_t layer = idx / n_expert_, expert = idx % n_expert_;
        if (!transient(layer, expert)) continue;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            if (role_ptr_[i] == nullptr || role_bytes_[i] == 0) continue;
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) expert * role_bytes_[i]);
            const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
            const size_t len = (size_t) ((uintptr_t) p + role_bytes_[i] - a), np = (len + pg - 1) / pg;
            vec.assign(np, 0);
            if (mincore((void*) a, len, vec.data()) != 0) return -1.0;
            pages += np;
            for (unsigned char v : vec) resident += v & 1;
        }
    }
    return pages ? (double) resident / (double) pages : -1.0;
#else
    (void) samples;
    return -1.0;   // no cheap residency query for a file mapping here: the caller keeps its SSD assumption
#endif
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (base_ == nullptr || dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    if (complement_ready_) {
        const uint8_t* held =
            resident_blob(index);
        if (held == nullptr && !override_.empty()) held = override_[index];
        if (held != nullptr) {
            std::memcpy(dst, held, (size_t) bytes);
            return true;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, expert, dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (!role_ptr_.empty()) return nullptr;   // the GGUF in place: no contiguous blob in any file
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, Pin pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
    err.clear();
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    // The GPU cache's experts and their slots
    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    std::vector<int32_t> pair_slot;
    primary_gpu_pairs.reserve((size_t) cache.resident());
    pair_slot.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const int32_t slot = cache.slot_of(layer, expert);
            if (slot == kNotResident) continue;
            primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
            pair_slot.push_back(slot);
        }
    }
    return pin_complement(cache.slots(), primary_gpu_pairs, pair_slot, err, pin, additional_gpu_pairs, lend_from_slot,
                          headroom_bytes, budget_bytes, rank);
}

bool FileExpertSource::pin_complement(
    int64_t n_slots, const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<int32_t>& pair_slot, std::string& err, Pin pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
    err.clear();
    const auto pin_t0 = std::chrono::steady_clock::now();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (primary_gpu_pairs.size() != pair_slot.size()) {
        err = "FileExpertSource: a slot for each GPU expert";
        return false;
    }
    const bool sync = strata::gpu::device_sync();
    if (sync != true) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + strata::gpu::last_error();

        return false;
    }

    // the bytes each slot's expert takes here (for the lend region below)
    std::vector<uint64_t> slot_bytes((size_t) std::max<int64_t>(n_slots, 0), 0);
    for (size_t i = 0; i < pair_slot.size(); ++i)
        if (pair_slot[i] >= 0 && pair_slot[i] < n_slots)
            slot_bytes[(size_t) pair_slot[i]] = layer_blob_bytes_[(size_t) primary_gpu_pairs[i].first];
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;
    const bool what_fits = budget_bytes == kResidentWhatFits;   // #467: the soft mode's second try
    const int64_t lend_asked = lend_from_slot;   // #11: the lent slots whose experts stay on the files are read again
    int64_t budget_lent = 0, budget_lendable = 0;   // the lend region's slots kept by a budget (below), and its size
    uint64_t budget_physical = 0;   // #403: the RAM reading a budget was sized from (0: no budget)
    if (budget_bytes > 0) {
        // CS-T: a RAM budget.  The complement's experts in `rank` order (the expert profile, hottest first) while
        // they fit, the rest left on the mapped files; clamped to what the RAM has room for.
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for --resident-budget-gib";
            return false;
        }
        budget_physical = physical;
        const uint64_t room = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (budget_bytes > room) {
            // #403: 256 MiB under the room, so the engine's own allocations after this reading still leave the
            // headroom (a budget clamped to exactly the room failed the safety check below on a reading a few MB
            // lower).  An unclamped budget is unchanged.
            const uint64_t margin = 256ull << 20;
            const uint64_t clamped = room > margin ? room - margin : 0;
            if (what_fits)
                std::fprintf(stderr, "FileExpertSource: RAM room for the complement: %.2f GiB (%.2f GiB available "
                                     "minus %.0f GiB headroom and a 0.25 GiB margin)\n",
                             (double) clamped / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "FileExpertSource: --resident-budget-gib %.2f is more than the RAM has room for "
                                     "(%.2f GiB available minus %.0f GiB headroom and a 0.25 GiB margin): %.2f GiB\n",
                             (double) budget_bytes / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0, (double) clamped / 1073741824.0);
            budget_bytes = clamped;
        }
        std::vector<uint64_t> ranked(offsets.size(), kNoComplement);
        uint64_t at = 0;
        int64_t held = 0;
        int64_t wanted = 0;   // the ranked experts no GPU holds
        if (rank != nullptr)
            for (const auto& pr : *rank) {
                if (pr.first < 0 || pr.second < 0 || pr.first >= n_layers_ || pr.second >= n_expert_) continue;
                const size_t i = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                if (offsets[i] == kNoComplement || ranked[i] != kNoComplement) continue;   // on a GPU, or twice
                ++wanted;
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) continue;
                ranked[i] = at;
                at += b;
                ++held;
            }
        // #467: none of them fits - the caller's plain mmap fallback, not an empty copy.  None to keep (the GPUs of a
        // layer split hold every ranked expert) is not that: the copy goes on empty, and keeps the stages' lend regions.
        if (what_fits && wanted > 0 && held == 0) {
            err = "FileExpertSource: the RAM has no room for any expert of the complement";
            return false;
        }
        std::fprintf(stderr, "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files\n",
                     (double) budget_bytes / 1073741824.0, (long long) held, (double) bytes / 1073741824.0);
        // The prompt path's lend region under a budget.  Until now a budget turned the loan off (`lend_from_slot = -1`
        // below), so every lent slot's expert was read from the pack during every prompt chunk and again when the slot
        // was refilled after the prompt - even with room left in the budget.  Measured on an RX 6900 XT 16 GB with
        // --resident-budget-gib 71 and UD-Q4_K_XL: the copy kept all 63.47 GiB of the experts outside the cache, 7.5
        // GiB of the budget stayed unused, and every agent turn read 0.5-2.6 GB of the 1,231 lent slots' experts (3.6
        // GiB) from the GGUF (a fresh 17K-token prompt 24 GB).  As the unbudgeted path does, take the lend region's
        // experts from the last slot down (the order the borrowing takes them) while they fit in what the budget has
        // left; the ones past it keep the file fallback, and `rank` keeps its precedence (the hottest experts first).
        if (lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty()) {
            budget_lendable = n_slots - lend_from_slot;
            std::vector<size_t> order;   // the pairs in the lend region, highest slot first
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] >= lend_from_slot && pair_slot[i] < n_slots) order.push_back(i);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pair_slot[a] > pair_slot[b]; });
            const uint64_t at0 = at;
            for (size_t i : order) {
                const auto& pr = primary_gpu_pairs[i];
                const size_t idx = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) break;   // a contiguous tail of the region: a short prompt lends only the last few
                if (ranked[idx] != kNoComplement) continue;
                ranked[idx] = at;
                at += b;
                ++budget_lent;
            }
            std::fprintf(stderr, "FileExpertSource: the prompt path's lend region: %lld of its %lld slots keep their experts "
                                 "in RAM too (%.2f GiB of the budget's remaining %.2f GiB)\n",
                         (long long) budget_lent, (long long) budget_lendable, (double) (at - at0) / 1073741824.0,
                         (double) (budget_bytes - at0) / 1073741824.0);
        }
        offsets.swap(ranked);
        bytes = at;
        lend_from_slot = -1;
    }

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty();
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    uint64_t physical = budget_physical;
    if (bytes > 0 || lend) {
        // #403: with a budget, the reading it was sized from - a second reading a few MB lower (the engine's own
        // allocations, the file cache) failed a budget the first one had clamped.  (A budget turns `lend` off.)
        if (physical == 0 && !available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        budget = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (bytes > budget) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM (%.2f GiB) minus the "
                          "%.0f GiB safety headroom",
                          (double) bytes / 1073741824.0, (double) physical / 1073741824.0,
                          (double) headroom_bytes / 1073741824.0);
            err = message;
#if defined(__linux__)
            if (std::ifstream mi("/proc/meminfo"); mi) {
                std::stringstream mt;
                mt << mi.rdbuf();
                err += detail::page_cache_hint(mt.str());
            }
#endif
            return false;
        }
    }
    // The prompt path's lend region: its slots' experts are streamed from here during a prompt and copied back into
    // their slots after it, so the ones that fit are kept here too (from the last slot down: a short prompt lends
    // only the last few).  The rest keep the mapped-file fallback.
    int64_t keep_from = n_slots;
    if (lend) {
        keep_from = detail::choose_resident_keep_from(slot_bytes, bytes, budget, lend_from_slot);
        if (keep_from < 0) keep_from = n_slots;
        if (keep_from < n_slots) {
            std::vector<std::pair<int32_t, int32_t>> core;
            core.reserve(primary_gpu_pairs.size());
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] < keep_from) core.push_back(primary_gpu_pairs[i]);
            if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, core,
                                                    additional_gpu_pairs, offsets, bytes, err)) return false;
        }
    }

    // A layer split: every stage's lend region (`stage_lend_regions`, set before the call).  The prompt path borrows
    // the tail slots of every part's cache, and with `additional_gpu_pairs` the single lend region above is off, so
    // without this the borrowed experts sat in neither VRAM nor RAM during a prompt and came back through page faults
    // from the mapped files (upstream aea9dc3e, an RTX 2000 Ada + 5060 Ti, 17/48 split, IQ3_S, one 73K-token prompt:
    // 3.7 GiB read from the NVMe, against 1.1 GiB in the arena mode, and 1.5 s of a 3.9 s chunk on host staging).
    // The regions walk each stage's highest slots first, as the borrowing does; the whole copy, complement and
    // regions together, stays within one budget, and experts past it keep the file fallback they had.
    int64_t stage_lent = 0;
    uint64_t stage_lent_bytes = 0;
    if (!stage_lend_regions_.empty()) {
        if (budget == std::numeric_limits<uint64_t>::max()) {   // #403: the one reading, taken now
            if (physical == 0 && !available_memory_bytes(physical)) {
                err = "FileExpertSource: cannot determine available RAM for the stage lend regions";
                return false;
            }
            budget = physical > headroom_bytes ? physical - headroom_bytes : 0;
        }
        uint64_t cap = budget;   // a RAM budget (--resident-budget-gib) caps the whole copy, regions included
        if (budget_bytes > 0 && budget_bytes != kResidentWhatFits && budget_bytes < cap) cap = budget_bytes;
        const uint64_t before = bytes;
        stage_lent = detail::append_stage_lend_regions(n_layers_, n_expert_, layer_blob_bytes_, offsets, bytes, cap,
                                                       stage_lend_regions_, err);
        if (stage_lent < 0) return false;
        stage_lent_bytes = bytes - before;
    }
    // #11: page-locked only when the RAM beside the copy holds what is still read from the files - the experts neither
    // on a GPU nor in the copy, and the prompt's lent slots' experts the copy does not keep
    bool lock = pin == Pin::on;
    if (pin == Pin::automatic && bytes > 0) {
        std::vector<char> on_gpu((size_t) (n_layers_ * n_expert_), 0);
        using Pairs = std::vector<std::pair<int32_t, int32_t>>;
        for (const Pairs* pairs : {static_cast<const Pairs*>(&primary_gpu_pairs), &additional_gpu_pairs})
            for (const auto& [l, e] : *pairs)
                if (l >= 0 && e >= 0 && l < n_layers_ && e < n_expert_) on_gpu[(size_t) l * (size_t) n_expert_ + (size_t) e] = 1;
        uint64_t in_place = 0;
        for (size_t i = 0; i < on_gpu.size(); ++i)
            if (!on_gpu[i] && offsets[i] == kNoComplement) in_place += layer_blob_bytes_[i / (size_t) n_expert_];
        if (lend_asked >= 0)
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i) {
                const auto [l, e] = primary_gpu_pairs[i];
                if (pair_slot[i] >= lend_asked && offsets[(size_t) l * (size_t) n_expert_ + (size_t) e] == kNoComplement)
                    in_place += layer_blob_bytes_[(size_t) l];
            }
        if (physical == 0 && !available_memory_bytes(physical)) physical = 0;
        const uint64_t cma = strata::platform::cma_free_bytes();   // not for pinned pages (upstream #3698)
        std::string why;
        lock = detail::resident_pin_fits(bytes, in_place, headroom_bytes, physical > cma ? physical - cma : 0, why);
        std::fprintf(stderr, "FileExpertSource: %s\n", why.c_str());
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    uint64_t locked = 0;
    uint64_t partial_pin = 0;   ///< CS-T: a registered prefix of a locked arena
    uint64_t lock_off = 0;      ///< where the working-set lock starts (after the registered prefix)
    std::string note;
    std::vector<ComplementChunk> chunks;   ///< the arena in parts when one allocation was refused
    bool heap = false;                     ///< the arena is std::malloc's (the pageable copy)
    auto release = [&]() {
        if (arena == nullptr) return;
        if (heap) {
            if (partial_pin > 0) (void) strata::gpu::host_unregister(arena);
            if (locked > 0) strata::platform::unlock_resident((uint8_t*) arena + lock_off, locked);
            std::free(arena);
        } else if (!chunks.empty()) {
            for (const ComplementChunk& c : chunks) (void) strata::gpu::free(c.host);
            chunks.clear();
        } else (void) strata::gpu::free(arena);
        arena = nullptr;
    };
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, lock ? "page-locked" : "pageable resident");
        std::fflush(stderr);
        if (lock) {
            // #38, upstream #3681: page-locked memory the driver took while the kernel reclaimed the model files' page
            // cache came slowly (a 71 GiB copy took 43 minutes after a few starts, 59 s after a reboot) and read slowly
            // from the GPU; the reclaim done first, on ordinary memory, leaves the driver free pages
            strata::platform::make_free_ram(bytes + (2ull << 30));
            const bool by_device = pin_by_device(offsets, chunks, note);
            const bool allocated = !by_device && strata::gpu::alloc_host(&arena, (size_t) bytes);
            if (by_device) {
                pinned_ok = true;
            } else if (allocated == true) {
                void* alias = nullptr;
                const bool aliased = strata::gpu::device_pointer(&alias, arena);
                if (aliased == true && alias != nullptr) {
                    device = (const uint8_t*) alias;
                    pinned_ok = true;
                    note = "page-locked and mapped";
                } else {
                    note = std::string("no device alias (") + strata::gpu::last_error() + ")";

                    (void) strata::gpu::free(arena);
                    arena = nullptr;
                }
            } else {
                // Refused: a single host allocation has a ceiling below the RAM (the B70's Level Zero took 30 GiB and
                // refused 32, 2026-10-11), so the same bytes in parts, each page-locked and mapped, cut between
                // blobs; the experts stay where the PCIe share reads them.  Only when the parts are refused too: the
                // same bytes in ordinary memory, copied by the CPU instead of by DMA, not locked in the working set
                // (on the B70 the locked copy read a 32K prompt 1.6% slower than the plain one).
                note = std::string("page-locking refused (") + strata::gpu::last_error() + ")";
                arena = nullptr;
                std::vector<std::pair<uint64_t, uint64_t>> blobs;   // (offset, bytes), in arena order
                for (size_t i = 0; i < offsets.size(); ++i)
                    if (offsets[i] != kNoComplement)
                        blobs.emplace_back(offsets[i], layer_blob_bytes_[i / (size_t) n_expert_]);
                std::sort(blobs.begin(), blobs.end());
                const char* parts_env = strata::env_get("PIN_PARTS");   // 0: one allocation or none (A/B)
                const int max_parts = parts_env != nullptr ? (int) std::strtol(parts_env, nullptr, 10) : 32;
                for (int parts = 2; parts <= max_parts && chunks.empty(); ++parts) {
                    std::vector<ComplementChunk> got;
                    uint64_t start = 0;
                    bool ok = true;
                    for (size_t b = 0; b <= blobs.size() && ok; ++b) {
                        const uint64_t at = b < blobs.size() ? blobs[b].first : bytes;
                        const bool cut = b == blobs.size() ||
                                         at - start >= (bytes + (uint64_t) parts - 1) / (uint64_t) parts;
                        if (!cut || at == start) continue;
                        ComplementChunk c{start, at, nullptr, nullptr};
                        void* h = nullptr;
                        void* d = nullptr;
                        ok = strata::gpu::alloc_host(&h, (size_t) (at - start));
                        if (ok) c.host = (uint8_t*) h;
                        ok = ok && strata::gpu::device_pointer(&d, h) && d != nullptr;
                        if (h != nullptr) got.push_back(c);
                        if (ok) got.back().device = (const uint8_t*) d;
                        start = at;
                    }
                    if (ok) chunks.swap(got);
                    else
                        for (const ComplementChunk& c : got) (void) strata::gpu::free(c.host);
                }
                if (!chunks.empty()) {
                    pinned_ok = true;
                    note = "page-locked and mapped in " + std::to_string(chunks.size()) + " parts (one " +
                           std::to_string(bytes) + "-byte allocation refused)";
                }
            }
        }
        if (!chunks.empty()) {
            arena = chunks[0].host;
            device = chunks[0].device;
        } else if (arena == nullptr) {
            arena = std::malloc((size_t) bytes);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            heap = true;
        }
        host = (const uint8_t*) arena;
    }

    // Copied layer by layer on a few threads: the page faults of the mapped file are the cost, and they overlap.
#if !defined(_WIN32)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        err = "FileExpertSource: cannot determine page size for mapped-page release";
        release();
        return false;
    }
#endif
    std::atomic<int64_t> next_layer{0}, layers_done{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    const auto copy_t0 = std::chrono::steady_clock::now();
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        for (;;) {
            const int64_t layer = next_layer.fetch_add(1);
            if (layer >= n_layers_ || failed.load()) return;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            std::vector<Fill> batch;   // #286, unbuffered: 32 blobs' reads in flight at once, merged where near
            auto flush = [&]() {
                if (batch.empty()) return true;
                bool ok = read_direct(batch.data(), batch.size());
                for (size_t i = 0; !ok && i < batch.size(); ++i)
                    if (!copy_from_files(batch[i].layer, batch[i].e, batch[i].dst)) return false;
                copied.fetch_add(blob_bytes * (uint64_t) batch.size());
                batch.clear();
                return true;
            };
            if (direct_.empty()) {   // mapped reads: ask for the layer's blobs before copying them
                std::vector<std::pair<int32_t, int32_t>> ahead;
                for (int64_t expert = 0; expert < n_expert_; ++expert)
                    if (offsets[(size_t) layer * (size_t) n_expert_ + (size_t) expert] != kNoComplement)
                        ahead.emplace_back((int32_t) layer, (int32_t) expert);
                (void) advise_pairs(ahead.data(), (int64_t) ahead.size());
            }
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
                const uint64_t offset = offsets[index];
                if (offset == kNoComplement) continue;
                if (offset > bytes || blob_bytes > bytes - offset) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                uint8_t* dst = chunks.empty() ? (uint8_t*) host + (size_t) offset : chunk_at(chunks, offset);
                if (!direct_.empty()) {
                    batch.push_back({0, layer, expert, dst});
                    if (batch.size() == 32 && !flush()) {
                        fail("FileExpertSource: an expert could not be read while building the cache complement");
                        return;
                    }
                    continue;
                }
                if (!copy_from_files(layer, expert, dst)) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                copied.fetch_add(blob_bytes);
            }
            if (!flush()) {
                fail("FileExpertSource: an expert could not be read while building the cache complement");
                return;
            }
#if !defined(_WIN32)
            // a copied layer's pages handed back; Windows leaves them to its memory manager, as upstream does
            if (role_ptr_.empty()) {   // experts.bin; the GGUF in place leaves its pages to the OS
            const uint64_t layer_offset = layer_offsets_[(size_t) layer];
            const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
            const uint64_t layer_end = layer_offset + layer_bytes;
            const uint64_t page = (uint64_t) page_size;
            const uint64_t advice_start = layer_offset - layer_offset % page;
            const uint64_t end_remainder = layer_end % page;
            const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
            const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
            if (advice_end > advice_start &&
                madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
                fail("FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer));
                return;
            }
            if (posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED) != 0) {
                fail("FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer));
                return;
            }
            }
#endif
            const int64_t done = layers_done.fetch_add(1) + 1;
            if (done % 8 == 0 || done == n_layers_)
                std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB, %.0f s)\n",
                             (long long) done, (long long) n_layers_, (double) copied.load() / 1073741824.0,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count());
        }
    };
    {
        const int threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    std::fflush(stderr);
    if (failed.load()) {
        err = fail_msg;
        release();
        return false;
    }

    complement_arena_ = arena;
    complement_chunks_ = std::move(chunks);
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = (pinned_ok || partial_pin > 0) && bytes > 0;
    complement_pin_limit_ = pinned_ok ? bytes : partial_pin;
    complement_partial_ = !pinned_ok && partial_pin > 0;
    complement_locked_ = locked;
    complement_lock_off_ = lock_off;
    complement_lent_slots_ = lend ? n_slots - keep_from : budget_lent;   // a budget keeps its own tail of the region
    complement_lent_slots_ += stage_lent;   // and the split stages' regions: the pairs the copy kept of them
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB in %.0f s%s%s\n",
                 complement_pinned_ ? "mapped pinned" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_t0).count(),
                 note.empty() ? "" : "; ", note.c_str());
    const int64_t lendable_all = lend ? n_slots - lend_from_slot : budget_lendable;   // 0: no lend region at all
    if (lendable_all > 0)
        std::fprintf(stderr, "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s\n", (long long) (lend ? n_slots - keep_from : budget_lent), (long long) lendable_all,
                     (lend ? n_slots - keep_from : budget_lent) < lendable_all
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (lendable_all > 0 && (lend ? n_slots - keep_from : budget_lent) < lendable_all) {
        // below full coverage the uncovered experts are read from the files during a long prompt, which routes
        // through nearly all of them, so the reads repeat (upstream 14d9d2e1, f780961c, #1389: 18% uncovered cost 26%
        // of the prompt speed on a gfx1100; short prompts barely touch them).  A warning, not a refusal.
        const long long lendable = (long long) lendable_all;
        const long long missing = lendable - (long long) (lend ? n_slots - keep_from : budget_lent);
        std::fprintf(stderr, "FileExpertSource: WARNING: %lld of %lld lendable slots (%.0f%%) will read their experts "
                             "from the files during a long prompt; expect a slower prompt read on 50K+ token prompts "
                             "(one measurement: 18%% uncovered cost 26%%).  %s and restart; the line above should read "
                             "N of N\n",
                     missing, lendable, 100.0 * (double) missing / (double) std::max<long long>(1, lendable),
                     lend ? "Free RAM" : "Raise --resident-budget-gib (the lend region comes after the ranked experts)");
    }
    if (!stage_lend_regions_.empty()) {
        int64_t offered = 0;
        for (const auto& region : stage_lend_regions_) offered += (int64_t) region.size();
        std::fprintf(stderr, "FileExpertSource: the split stages lend %lld slots to the prompt path: %lld keep their "
                             "experts in RAM too (%.2f GiB)%s\n",
                     (long long) offered, (long long) stage_lent, (double) stage_lent_bytes / 1073741824.0,
                     stage_lent < offered ? " (the rest are read from the file when lent: not enough RAM for them)" : "");
    }
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

int64_t FileExpertSource::resident_count() const {
    return (int64_t) std::count_if(complement_offsets_.begin(), complement_offsets_.end(),
                                  [](uint64_t off) { return off != kNoComplement; });
}

bool FileExpertSource::has_resident(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return resident_blob(index) != nullptr;
}

bool FileExpertSource::reserve_exchanges(int64_t n, std::string& err, int64_t bounce) {
    err.clear();
    if (n <= xstage_cap_) return true;
    if (!staged_.empty()) { err = "FileExpertSource: exchange buffers are in use"; return false; }
    if (exchange_storage_.active()) {   // the first allocation may now hold live resident experts
        err = "FileExpertSource: reserve the most exchange buffers before the rotation is on";
        return false;
    }
    uint64_t blob = 0;
    for (const uint64_t b : layer_blob_bytes_) blob = std::max(blob, b);
    if (blob == 0 || n <= 0) { err = "FileExpertSource: no expert geometry for the exchange buffers"; return false; }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) strata::gpu::free(xstage_);
        else std::free(xstage_);
        xstage_ = nullptr;
        xstage_cap_ = 0;
    }
    if ((uint64_t) n > std::numeric_limits<size_t>::max() / blob) {
        err = "FileExpertSource: the exchange buffers' size overflows";
        return false;
    }
    const char* rotate_env = strata::env_get("EXCHANGE_ROTATE");
    const bool requested = rotate_env != nullptr && std::strcmp(rotate_env, "1") == 0;
    // layers of several block sizes (a GGUF's mixed i-quants): spares of every size, see ExchangeStorage
    bounce = std::max<int64_t>(0, std::min(bounce, n));
    const uint64_t rotate_total =
        detail::ExchangeStorage::exchange_bytes_needed(layer_blob_bytes_, (size_t) n, (size_t) bounce);
    const bool eligible = requested && rotate_total != 0 && rotate_total <= std::numeric_limits<size_t>::max() &&
                          complement_pinned_ && !complement_partial_ &&
                          complement_host_ != nullptr && complement_device_ != nullptr && complement_bytes_ > 0 &&
                          complement_chunks_.empty();
    const size_t total = eligible ? (size_t) rotate_total : (size_t) n * (size_t) blob;
    void* p = nullptr;
    if (strata::gpu::alloc_host(&p, total) == true && p != nullptr) {
        xstage_pinned_ = true;
        xstage_dev_ = current_device();
    } else {

        p = std::malloc(total);
        xstage_pinned_ = false;
        if (p == nullptr) { err = "FileExpertSource: cannot allocate the exchange buffers"; return false; }
    }
    xstage_ = (uint8_t*) p;
    xstage_cap_ = n;
    xstage_blob_ = blob;
    if (requested) {
        std::string reason = "it needs a fully page-locked RAM copy in one part";
        void* device = nullptr;   // host USM: the device reads it at the same address
        if (eligible && xstage_pinned_ && strata::gpu::device_pointer(&device, p) && device != nullptr) {
            try {
                std::vector<uint64_t> sizes(complement_offsets_.size(), 0);
                for (size_t i = 0; i < sizes.size(); ++i) {
                    const size_t layer = n_expert_ > 0 ? i / (size_t) n_expert_ : 0;
                    sizes[i] = layer < layer_blob_bytes_.size() ? layer_blob_bytes_[layer] : 0;
                }
                if (exchange_storage_.initialize_sized(complement_offsets_, sizes, (uint8_t*) complement_host_,
                                                       complement_device_, complement_bytes_, xstage_,
                                                       (const uint8_t*) device, (uint64_t) total, (size_t) n,
                                                       (size_t) bounce, reason))
                    std::fprintf(stderr, "FileExpertSource: exchange buffer rotation on: %lld buffers in %zu block "
                                         "sizes (largest %llu bytes, %.2f GiB page-locked); no host copy at a commit\n",
                                 (long long) n, exchange_storage_.block_sizes(), (unsigned long long) blob,
                                 (double) total / 1073741824.0);
            } catch (const std::bad_alloc&) { reason = "its tables could not be allocated"; }
        } else if (eligible) {
            reason = "the exchange buffers could not be page-locked";
        }
        if (!exchange_storage_.active())
            std::fprintf(stderr, "FileExpertSource: exchange buffer rotation is off (%s): the copying exchanges\n",
                         reason.c_str());
    }
    return true;
}

uint8_t* FileExpertSource::exchange_buffer(int64_t q, int64_t layer) const {
    if (xstage_ == nullptr || q < 0 || q >= xstage_cap_) return nullptr;
    if (exchange_storage_.active())
        return layer >= 0 && layer < n_layers_
                   ? exchange_storage_.spare((size_t) q, (size_t) layer_blob_bytes_[(size_t) layer]).host
                   : exchange_storage_.spare((size_t) q).host;
    return xstage_ + (size_t) q * (size_t) xstage_blob_;
}

bool FileExpertSource::stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q) {
    if (out < 0 || out >= n_expert_ || !has_resident(layer, in) || has_resident(layer, out) ||
        exchange_buffer(q, layer) == nullptr)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (override_.empty()) override_.assign((size_t) blobs_, nullptr);
    if (override_[i_out] != nullptr) return false;
    for (const Exchange& x : staged_)
        if (x.in == i_in || x.q == q) return false;
    override_[i_out] = exchange_buffer(q, layer);
    staged_.push_back({i_in, i_out, q, layer_blob_bytes_[(size_t) layer]});
    return true;
}

int64_t FileExpertSource::commit_exchanges() {
    int64_t n = 0;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        if (exchange_storage_.active()) {
            if (!exchange_storage_.commit(x.in, x.out, (size_t) x.q, src, (size_t) x.bytes)) {
                // admitting the GPU swap after a failed RAM ownership update would lose an expert
                std::fprintf(stderr, "FileExpertSource: invalid exchange rotation commit; refusing corrupt residency\n");
                std::abort();
            }
            ++n;
            override_[x.out] = nullptr;
            continue;
        }
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr) {
            std::memcpy(complement_at(at), src, (size_t) x.bytes);
            if (detail::exchange_cache_complement(complement_offsets_, x.in, x.out)) ++n;
        }
        override_[x.out] = nullptr;
    }
    staged_.clear();
    exchanges_ += n;
    return n;
}

bool FileExpertSource::replace_resident(int64_t layer, int64_t in, int64_t out) {
    if (!complement_ready_ || complement_host_ == nullptr || exchange_storage_.active() || !staged_.empty() ||
        layer < 0 || layer >= n_layers_ || in < 0 || out < 0 || in >= n_expert_ || out >= n_expert_ || in == out)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i_in >= complement_offsets_.size() || i_out >= complement_offsets_.size()) return false;
    if (!override_.empty() && (override_[i_in] != nullptr || override_[i_out] != nullptr)) return false;
    const uint64_t at = complement_offsets_[i_out], bytes = layer_blob_bytes_[(size_t) layer];
    if (at == kNoComplement || complement_offsets_[i_in] != kNoComplement || at > complement_bytes_ ||
        bytes > complement_bytes_ - at) return false;
    // read beside the copy first: a failed read leaves `out` where it was
    if (ram_adapt_buf_.size() < (size_t) bytes) ram_adapt_buf_.resize((size_t) bytes);
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, in, ram_adapt_buf_.data())) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    std::memcpy(complement_at(at), ram_adapt_buf_.data(), (size_t) bytes);
    if (!detail::exchange_cache_complement(complement_offsets_, i_out, i_in)) return false;
    ++ram_adapts_;
    return true;
}

bool FileExpertSource::replace_resident_staged(int64_t layer, int64_t in, int64_t out) {
    if (!complement_ready_ || complement_host_ == nullptr || exchange_storage_.active() || !staged_.empty() ||
        layer < 0 || layer >= n_layers_ || in < 0 || out < 0 || in >= n_expert_ || out >= n_expert_ || in == out)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i_in >= complement_offsets_.size() || i_out >= complement_offsets_.size()) return false;
    if (!override_.empty() && (override_[i_in] != nullptr || override_[i_out] != nullptr)) return false;
    const uint64_t at = complement_offsets_[i_out], bytes = layer_blob_bytes_[(size_t) layer];
    if (at == kNoComplement || complement_offsets_[i_in] != kNoComplement || at > complement_bytes_ ||
        bytes > complement_bytes_ - at || bytes > stage_blob_) return false;
    {
        // held across the copy: claim_stage cannot hand the buffer to another blob meanwhile
        std::lock_guard<std::mutex> lk(stage_mu_);
        const auto it = stage_of_.find((int64_t) i_in);
        if (it == stage_of_.end() || stage_busy_[it->second] || stage_key_[it->second] != (int64_t) i_in) return false;
        std::memcpy(complement_at(at), stage_buf_[it->second].get(), (size_t) bytes);
    }
    if (!detail::exchange_cache_complement(complement_offsets_, i_out, i_in)) return false;
    ++ram_adapts_;
    ++ram_staged_;
    return true;
}

void FileExpertSource::commit_copies() {
    if (exchange_storage_.active()) return;   // STRATA_EXCHANGE_ROTATE: ownership moves in commit_flip, nothing to copy
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_.empty() ? nullptr : override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr)
            std::memcpy(complement_at(at), src, (size_t) x.bytes);
    }
}

int64_t FileExpertSource::commit_flip() {
    int64_t n = 0;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        if (exchange_storage_.active()) {   // STRATA_EXCHANGE_ROTATE: the same ownership transfer as commit_exchanges
            if (!exchange_storage_.commit(x.in, x.out, (size_t) x.q, src, (size_t) x.bytes)) {
                std::fprintf(stderr, "FileExpertSource: invalid exchange rotation commit; refusing corrupt residency\n");
                std::abort();
            }
            ++n;
            override_[x.out] = nullptr;
            continue;
        }
        const uint64_t at = complement_offsets_[x.in];
        // the same test as commit_copies: an exchange whose bytes were copied is the one that flips
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr && detail::exchange_cache_complement(complement_offsets_, x.in, x.out))
            ++n;
        override_[x.out] = nullptr;
    }
    staged_.clear();
    exchanges_ += n;
    return n;
}

const uint8_t* FileExpertSource::resident_blob(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || complement_host_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ ||
        expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (index >= complement_offsets_.size()) return nullptr;
    return resident_blob(index);   // through the rotation's ownership table when STRATA_EXCHANGE_ROTATE is on
}

const uint8_t* FileExpertSource::resident_blob(size_t index) const {
    if (exchange_storage_.active()) return exchange_storage_.resident(index).host;
    if (!complement_chunks_.empty())
        return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement
                   ? complement_at(complement_offsets_[index]) : nullptr;
    return detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* result = nullptr;
    bool from_files = true;
    if (complement_ready_) {
        result = resident_blob(index);
        if (result != nullptr) {
            ram_reads_.fetch_add(1, std::memory_order_relaxed);
            from_files = false;
        } else if (!override_.empty() && override_[index] != nullptr) {
            result = override_[index];
            from_files = false;
        }
    }
    if (from_files) {
        if (staged()) {
            result = staged_blob(layer, expert);          // counts its bytes
        } else {
            result = mapped_blob(layer, expert);
            if (result != nullptr) {
                file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
#if defined(__linux__)
                if (io_stats_) {   // what the page cache held when the kernels were handed the mapping
                    uint64_t total = 0;
                    const uint64_t c = cached_bytes(layer, expert, total, false);
                    io_cached_.fetch_add(c, std::memory_order_relaxed);
                    io_uncached_.fetch_add(total - c, std::memory_order_relaxed);
                    if (gate_on_ && tl_decode) {
                        gate_c_.fetch_add(c, std::memory_order_relaxed);
                        gate_u_.fetch_add(total - c, std::memory_order_relaxed);
                    }
                }
#endif
            }
        }
        if (complement_ready_ && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (exchange_storage_.active()) return exchange_storage_.resident(index).host != nullptr;
    if (index >= complement_offsets_.size() || complement_offsets_[index] == kNoComplement) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (exchange_storage_.active()) return exchange_storage_.resident(index).device;
    if (!complement_chunks_.empty()) {
        const uint64_t off = complement_offsets_[index];
        const ComplementChunk& c = *chunk_of(complement_chunks_, off);
        return c.device + (size_t) (off - c.start);
    }
    return complement_device_ + (size_t) complement_offsets_[index];
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`) are fixed
// arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is refused at
// run time rather than written past them (upstream 64cfe3c)
constexpr int64_t kMaxWindowEntries = 128;
static_assert((int64_t) strata::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

namespace {
// The CPU side's calibration calls: up to 8 experts a call, taken in turn from 64 of one layer that are in RAM
// (pinned: blob() reads no file) and not in VRAM, one token each, in the layer's native format.  64 experts are past
// any CPU cache (8 were not: their calls ran 3x faster than a decode's), so the calls read memory as a decode does.
// `layer` < 0 picks the first such layer.
struct CpuCalJobs {
    static constexpr int kCall = 8, kRing = 64;
    strata::kernels::cpu::NativeFmt f{};
    std::vector<float> x, out;
    strata::kernels::cpu::NativeAct act;
    const uint8_t* ring[kRing] = {};
    int cursor = 0;
    strata::kernels::cpu::ExpertJobMulti jobs[kCall];
    void next(int k) {   // the next k experts of the ring into jobs[0, k)
        for (int j = 0; j < k; ++j) {
            jobs[j].blob = ring[cursor];
            cursor = (cursor + 1) % kRing;
        }
    }
};
bool cpu_cal_jobs(ExpertDispatch& d, int64_t n_layers, int64_t& layer, CpuCalJobs& cj, std::string& why) {
    using namespace strata::kernels::cpu;
    const ExpertLayout& lay = expert_layout();
    if (!lay.native) { why = "not a native pack"; return false; }
    if (d.pool == nullptr || d.src == nullptr || d.host_res == nullptr) { why = "no CPU pool or residency table"; return false; }
    const int64_t l0 = layer >= 0 ? layer : 0, l1 = layer >= 0 ? layer + 1 : n_layers;
    layer = -1;
    for (int64_t l = l0; l < l1 && layer < 0; ++l) {
        int n = 0;
        for (int64_t e = 0; e < d.n_expert && n < CpuCalJobs::kRing; ++e)
            if (d.host_res[(size_t) l * (size_t) d.n_expert + (size_t) e] < 0 && d.src->pinned(l, e)) {
                const uint8_t* b = d.src->blob(l, e);
                if (b != nullptr) cj.ring[n++] = b;
            }
        if (n == CpuCalJobs::kRing) layer = l;
    }
    if (layer < 0) { why = "no layer has 64 experts in RAM outside VRAM"; return false; }
    cj.f = lay.fmt[(size_t) layer];
    const int64_t H = cj.f.n_embd;
    cj.x.assign((size_t) H, 0.0f);
    cj.out.assign((size_t) CpuCalJobs::kCall * (size_t) H, 0.0f);
    for (int64_t i = 0; i < H; ++i) cj.x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
    native_quant_act(cj.f, cj.x.data(), cj.act.bytes);
    for (int j = 0; j < CpuCalJobs::kCall; ++j) {
        cj.jobs[j] = ExpertJobMulti{};
        cj.jobs[j].nt = 1;
        cj.jobs[j].nact[0] = cj.act.bytes;
        cj.jobs[j].out[0] = cj.out.data() + (size_t) j * (size_t) H;
    }
    return true;
}
}  // namespace

bool measure_cpu_expert_ms(ExpertDispatch& d, int64_t n_layers, double& c0, double& c, int64_t& layer, std::string& why) {
    CpuCalJobs cj;
    if (!cpu_cal_jobs(d, n_layers, layer, cj, why)) return false;
    constexpr int kCall = CpuCalJobs::kCall;
    // three rounds, each after 100 ms of calls that keep the clocks up (an idle CPU takes tens of ms to raise them; a
    // decode keeps them up): 1 and 8 experts a call in turns, 32 calls each, the median; the lowest round.  One round's
    // mean read an expert 4-8x slow in 2 of 32 starts on the RX 9060 XT's machine (0.18 and 0.39 against 0.05 ms, the
    // same beside the link: the whole round slow), and the split then sent the missed experts over PCIe for most of a
    // 256-token decode
    const auto now = [] { return std::chrono::steady_clock::now(); };
    double ms[2] = {1e30, 1e30};
    for (int round = 0; round < 3; ++round) {
        for (const auto t0 = now(); now() - t0 < std::chrono::milliseconds(100);) {
            cj.next(kCall);
            d.pool->run_split_multi_native(cj.f, cj.jobs, kCall);
        }
        double one[2][32];
        for (int rep = 0; rep < 32; ++rep)
            for (int w = 0; w < 2; ++w) {
                const int k = w == 0 ? 1 : kCall;
                cj.next(k);
                const auto t0 = now();
                d.pool->run_split_multi_native(cj.f, cj.jobs, k);
                one[w][rep] = std::chrono::duration<double, std::milli>(now() - t0).count();
            }
        for (int w = 0; w < 2; ++w) {
            std::nth_element(one[w], one[w] + 16, one[w] + 32);
            ms[w] = std::min(ms[w], one[w][16]);
        }
    }
    c = (ms[1] - ms[0]) / (kCall - 1);
    c0 = std::max(0.0, ms[0] - c);
    if (!(c > 0.0)) { why = "the CPU times did not grow with the experts"; return false; }
    return true;
}

int pcie_split_m(const PcieSplitTimes& t, bool overlap, int f, int h) {
    if (f <= 0) return 0;
    const double hits = h > 0 ? t.q0 + t.q * h : 0.0;
    auto cost = [&](int m) {
        const int k = f - m;
        const bool both = m > 0 && k > 0;
        const double fetch = m > 0 ? (both ? t.pb0 + t.pb * m : t.p0 + t.p * m) : 0.0;
        const double after = m > 0 ? (h > 0 ? t.q * m : t.q0 + t.q * m) : 0.0;   // the fetched experts' compute
        const double gpu = (overlap ? std::max(hits, fetch) : hits + fetch) + after;
        const double cpu = k > 0 ? (both ? t.cb0 + t.cb * k : t.c0 + t.c * k) : 0.0;
        return std::max(gpu, cpu);
    };
    const double all_cpu = cost(0);
    double best = all_cpu;
    int pick = 0;
    for (int m = 1; m <= f; ++m)
        if (const double c = cost(m); c < best) { best = c; pick = m; }
    return best <= 0.9 * all_cpu ? pick : 0;
}

namespace {
// The split's CPU times come from start-up calls of one token an expert, alone on the pool.  A decode's calls ran
// slower than those times gave: 1.7x on the RX 9060 XT (0.049 against ~0.085 ms an expert; decode 41 tok/s against
// upstream's 45 at a fixed share), and 3.5x beside a busy host, where the split gave the CPU nearly every missed expert
// (26 against 40 tok/s).  The decode's own pool calls are set against the times the split read for them, and the CPU's
// times are scaled by the ratio when it is off by more than 15%: after 512 calls with experts on the CPU, then 1,024 and
// 2,048 more, three updates at most (as upstream's PcieAuto refines its share), by up to 8x an update (4x left a
// start-up reading 8x slow at 2x after the first update).  STRATA_PCIE_SPLIT_REFINE=0: the start-up times stay.
void refine_split(ExpertDispatch& d, bool both, int experts, double ms) {
    static const bool on = [] {
        const char* v = strata::env_get("PCIE_SPLIT_REFINE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    ExpertDispatch::SplitRefine& r = d.split_refine;
    if (!on || r.updates >= 3) return;
    PcieSplitTimes& t = *d.pcie_split;
    r.ms += ms;
    r.ms_model += both ? t.cb0 + t.cb * experts : t.c0 + t.c * experts;
    ++r.calls;
    r.experts += experts;
    if (r.calls < (int64_t) 512 << r.updates || r.experts < 300) return;
    const double ratio = r.ms_model > 0.0 ? r.ms / r.ms_model : 1.0;
    ++r.updates;
    r = ExpertDispatch::SplitRefine{0.0, 0.0, 0, 0, r.updates};
    if (!(ratio > 0.0) || (ratio >= 1.0 / 1.15 && ratio <= 1.15)) return;
    const double f = std::min(8.0, std::max(0.125, ratio));
    t.c0 *= f;
    t.c *= f;
    t.cb0 *= f;
    t.cb *= f;
    std::fprintf(stderr, "xestrata: PCIe split: the decode's CPU pool calls took %.2fx the start-up times; an expert on "
                         "the CPU is now %.3f ms (%.3f beside the link), update %d of 3\n", ratio, t.c, t.cb, r.updates);
}

// STRATA_ROUTE_TAIL_SKIP: the rank from which a missed expert that every token of the window routes that low is
// skipped (upstream b4e3407b).  An explicit environment value is read once; the serve start-up (generate.cpp) sets the
// default where it applies.
std::atomic<int> g_tail_skip_rank{-1};   // -1: not resolved yet
}  // namespace
int tail_skip_rank() {
    int r = g_tail_skip_rank.load(std::memory_order_relaxed);
    if (r < 0) {
        const char* v = strata::env_get("ROUTE_TAIL_SKIP");
        r = v != nullptr ? (int) std::strtol(v, nullptr, 10) : 0;
        r = r > 0 && r < 10 ? r : 0;
        g_tail_skip_rank.store(r, std::memory_order_relaxed);
    }
    return r;
}
void set_tail_skip_rank(int r) { g_tail_skip_rank.store(r > 0 && r < 10 ? r : 0, std::memory_order_relaxed); }

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (k < 1 || k > kMaxWindowEntries / n_tok) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if (d.lookahead) d.lookahead->submit(d.layers, x_f, n_tok, d.host_res);
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    // the activation width: a native layer's own (models other than Qwen3.8), else the S2 kernels' compiled H
    const int64_t W = native ? lay.fmt[(size_t) d.layers].n_embd : (int64_t) H;
    if (native && d.nact_multi.size() < (size_t) MAXT) d.nact_multi.resize((size_t) MAXT);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = strata::env_get("POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    const FileExpertSource::DecodeScope decode_scope;   // the file tier's decode gate: this thread reads for the decode
    // zero-copy (upstream 149, STRATA_COLD_OVERLAP=4): unset, on where the source's Linux file-tier defaults apply
    static const int cold_env = [] {
        const char* v = strata::env_get("COLD_OVERLAP");
        return v == nullptr || v[0] == '\0' ? -1 : (std::strtol(v, nullptr, 10) == 4 ? 4 : 0);
    }();
    d.cold_mode = cold_env >= 0 ? cold_env : (d.src->linux_file_defaults() ? 4 : 0);
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty() && d.usage_by_window && !d.route_rec.empty() && d.layers < d.route_rec_layers &&
        n_tok <= d.route_rec_t) {
        int32_t* rec = d.route_rec.data() + (size_t) d.layers * (size_t) (d.route_rec_t * k);
        std::copy(ids, ids + n_tok * k, rec);
    } else if (!d.usage.empty() && (!d.usage_by_window || !d.route_rec.empty())) {
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    }
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe, 2 a helper GPU, -2 skipped
    std::fill_n(kind, kMaxWindowEntries, -1);
    int split_m = -1;                      // the measured split's PCIe count for this layer (-1: no split)
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        // a helper GPU (RemoteExperts) holding the expert computes it in its own VRAM once begin() claims the row: like
        // the peer tier's, such an expert is no miss of this plan and never part of the PCIe share, which would read it
        // over the primary's link instead (2x RX 6900 XT, IQ3_S, --remote-expert-opt with the probed share 0.39: ~4
        // helper experts per layer-window moved to PCIe, 59 ms per window against 34 ms)
        auto helper_holds = [&](int32_t e) {
            for (int r = 0; r < d.remote_count; ++r)
                if (d.remote[r]->holds(d.layers, e)) return true;
            return false;
        };
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0 &&
                    !helper_holds(e)) ++nmiss;
            }
        }
        // STRATA_ROUTE_TAIL_SKIP=R (upstream 65268d99; opt-in, changes the output): a missed expert routed at rank >= R
        // (the smallest weights of the top-k) contributes zero to that token, without renormalisation (the others keep
        // the weight the router gave them); one that EVERY token of the window routes that low is neither copied nor
        // computed.  Less PCIe and CPU work.  Per entry, not per expert: upstream zeroed a token's low-ranked expert
        // only when no other token of the window needed it, so a token's output depended on the window's other rows
        // (window-invariant output, #152).  An expert a helper GPU holds is computed there, as a resident one is on
        // this GPU, and the misses it does not hold are skipped as without it: a helper used to turn the skip off, and
        // the CPU then computed more rows than without the helper (#53: decode 15% slower, 2x RTX 4070 Ti, no P2P).
        const int tail_rank = tail_skip_rank();
        // STRATA_DISK_SKIP_FROM=R (Project Maya's STRATA_GLM_DISK_SKIP_FROM, #109; opt-in, changes the output): the
        // same for a missed expert the RAM copy does not hold either, read from the model file (`transient`), routed at
        // rank >= R by every token of the window: the SSD is not waited for.  Maya also skips such an expert by its
        // route weight; the dispatch has the ids only, so the rank alone here
        static const int disk_rank = [] {
            const char* v = strata::env_get("DISK_SKIP_FROM");
            return v != nullptr ? std::max(0, (int) std::strtol(v, nullptr, 10)) : 0;
        }();
        bool skip_d[kMaxWindowEntries];
        for (int q = 0; q < nd; ++q) skip_d[q] = false;
        if ((tail_rank > 0 && tail_rank < k) || (disk_rank > 0 && disk_rank < k)) {
            int minr[kMaxWindowEntries];
            for (int q = 0; q < nd; ++q) minr[q] = 99;
            for (int64_t i = 0; i < n; ++i) {
                int q = 0;
                while (q < nd && distinct[q] != first_of[i]) ++q;
                if (q < nd) minr[q] = std::min(minr[q], (int) (i % k));
            }
            for (int q = 0; q < nd; ++q) {
                const int32_t e = ids[distinct[q]];
                if (e < 0 || e >= d.n_expert) continue;
                if (d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) continue;   // resident
                if (helper_holds(e)) continue;
                const bool tail = tail_rank > 0 && minr[q] >= tail_rank;
                const bool disk = disk_rank > 0 && minr[q] >= disk_rank && d.src->transient(d.layers, e);
                if (!tail && !disk) continue;
                skip_d[q] = true;
                --nmiss;
                ++d.tail_experts;
            }
        }
        // STRATA_FREE_MISSES=1 (measuring only, the output is wrong; after nextsycl's NS_FREE_MISSES): a missed expert
        // is computed on the GPU from the slot of a resident expert of the layer instead of fetched or computed on the
        // CPU, so a run times the decode as if every expert were in VRAM - the bound the CPU / PCIe split works toward
        static const bool free_misses = [] {
            const char* v = strata::env_get("FREE_MISSES");
            return v != nullptr && std::strtol(v, nullptr, 10) != 0;
        }();
        int32_t standin = -1;
        if (free_misses && d.remote_count == 0) {
            for (int32_t e = 0; e < d.n_expert && standin < 0; ++e)
                standin = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
            if (standin >= 0) nmiss = 0;
        }
        // the layer's RAM can be read over PCIe (its first expert may be in VRAM, outside a RAM copy of the others)
        const bool pcie_ok = d.pcie_num > 0 && d.src->pcie_layer(d.layers);
        // from the times measured at start (pcie_split_m); else the share pcie_num/256
        int m = !pcie_ok ? 0
                : d.pcie_split != nullptr ? pcie_split_m(*d.pcie_split, d.pcie_overlap, nmiss, std::max(0, nd - nmiss))
                                          : (nmiss * d.pcie_num) >> 8;
        // STRATA_PCIE_MIN1 (opt-in, upstream ab6bce05): the share pcie_num/256 rounds to 0 at 1-2 misses a layer, so the
        // PCIe lane idles on the typical decode step; with two misses or more one goes over PCIe (the last in routing
        // order; only where its blob is pinned, as the fetch below checks).  Not with the measured split.
        static const bool pcie_min1 = [] {
            const char* v = strata::env_get("PCIE_MIN1");
            return v != nullptr && std::strtol(v, nullptr, 10) != 0;
        }();
        if (pcie_min1 && pcie_ok && d.pcie_split == nullptr && m < 1 && nmiss >= 2) m = 1;
        if (pcie_ok && d.pcie_split != nullptr) split_m = m;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            if (skip_d[q]) {                       // STRATA_ROUTE_TAIL_SKIP: neither copied nor computed
                for (int64_t i = i0; i < n; ++i)
                    if (first_of[i] == i0) kind[i] = -2;
                continue;
            }
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot < 0) slot = standin;   // STRATA_FREE_MISSES
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else if (helper_holds(e)) {
                    // left to the helper: kind -1 until RemoteExperts::begin() claims the row (see above)
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        // pinned() first: blob() is a file read with the GGUF in place (upstream cacffa9)
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            // a tail entry of a missed expert (above); STRATA_DISK_SKIP_FROM's of one read from the file, per entry too
            const bool on_disk = kd != 0 && disk_rank > 0 && d.src->transient(d.layers, e);
            const bool held_by_helper = kd != 0 && e >= 0 && e < d.n_expert && helper_holds(e);
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0)
                    kind[i] = kd != 0 && !held_by_helper &&
                                      ((tail_rank > 0 && i % k >= tail_rank) || (on_disk && i % k >= disk_rank))
                                  ? -2 : kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0 && kind[i] == 1) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    // the verify window's copy_rows_from_mapped zeroes the rows the GPU computed itself (STRATA_DEC_BATCH, the
    // default); STRATA_VERIFY_DEVICE_PLAN's copy_or_zero_from_mapped copies every row, so there the host still zeroes
    // them.  A layer with no row for the CPU quantizes no activation and fetches nothing (upstream a36be1db)
    static const bool dec_batch = [] {
        const char* v = strata::env_get("DEC_BATCH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    static const bool device_plan = [] {
        const char* v = strata::env_get("VERIFY_DEVICE_PLAN");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    const bool gpu_zeroes_hits = d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap && dec_batch && !device_plan;
    const bool any_cpu = std::any_of(kind, kind + n, [](int32_t v) { return v == -1; });   // -2: skipped, no work
    const auto c1 = std::chrono::steady_clock::now();
    if (!any_cpu) {
    } else if (native) {
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * W, d.nact_multi[(size_t) t].bytes);
    } else {
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * W, (int) W, d.act_multi[(size_t) t]);
    }
    const auto c2 = std::chrono::steady_clock::now();
    if (any_cpu) {   // the experts the CPU computes, fetched together (the GGUF in place reads them on several threads)
        static thread_local std::vector<int64_t> miss;
        miss.clear();
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (kind[i] == -1 && ids[i] >= 0 && ids[i] < d.n_expert &&
                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())
                miss.push_back(ids[i]);
        if (d.cold_mode == 4 && native) (void) d.src->prefetch_begin(d.layers, miss.data(), (int64_t) miss.size());
        else d.src->prefetch(d.layers, miss.data(), (int64_t) miss.size());
    }
    if (d.cold_mode == 4 && d.zc_fmt.size() < d.jobs_multi.size()) d.zc_fmt.resize(d.jobs_multi.size());   // jobs keep pointers
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * W;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] == -2) {            // STRATA_ROUTE_TAIL_SKIP: a skipped tail expert contributes zero
                std::memset(row, 0, (size_t) H * sizeof(float));
                ++d.tail_skipped;
                continue;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                else ++d.offload_entries;                       // #588: PCIe or another GPU
                if (!(gpu_zeroes_hits && kind[i] <= 1)) std::memset(row, 0, (size_t) W * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            // #45: a job takes MAXT tokens, and an id routed twice within a token (the same expert in two of its k
            // slots) can give one expert more: a ninth wrote past act/out/nact into fmt and the pool's per-job arrays,
            // and the CPU pool faulted.  The expert gets another job then (jobs_multi has n_tok * k of them).
            if (jo >= 0 && d.jobs_multi[(size_t) jo].nt >= MAXT) {
                static std::atomic<bool> said{false};
                if (!said.exchange(true))
                    std::fprintf(stderr, "xestrata: expert %lld of layer %lld is routed more than %d times in a window "
                                         "of %lld tokens (an id repeated within a token); it is computed in two "
                                         "jobs (#45, said once)\n",
                                 (long long) e, (long long) d.layers, MAXT, (long long) n_tok);
                jo = -1;
            }
            if (jo < 0) {
                const uint8_t* zp[3] = {nullptr, nullptr, nullptr};
                const bool zc = d.cold_mode == 4 && native && d.src->zc_roles(d.layers, e, zp);
                const uint8_t* b = zc ? zp[0] : d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
                nj.fmt = nullptr;
                if (zc) {   // the gate rows are the blob; up and down at their distance in the mapping
                    strata::kernels::cpu::NativeFmt& f = d.zc_fmt[(size_t) jo];
                    f = lay.fmt[(size_t) d.layers];
                    f.up_off = (size_t) ((uintptr_t) zp[1] - (uintptr_t) zp[0]);     // modular: the tensors' order
                    f.down_off = (size_t) ((uintptr_t) zp[2] - (uintptr_t) zp[0]);   // in the file does not matter
                    nj.fmt = &f;
                    ++d.cold_zc;
                }
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = native ? nullptr : &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi[(size_t) t].bytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (njobs > 0) {
        if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
        else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    if (split_m >= 0 && njobs > 0 && d.remote_count == 0) refine_split(d, split_m > 0, njobs, ms(c3, c4));
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    strata::gpu::Stream cs = stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (!strata::gpu::memset_async(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) ||
            !strata::gpu::copy_async(d.d_slot, d.h_slot.data(), list_bytes, cs) ||
            !strata::gpu::copy_async(d.d_dst, d.h_dst.data(), list_bytes, cs)) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) strata::gpu::event_record((strata::gpu::Event*) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) strata::gpu::event_query((strata::gpu::Event*) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (strata::gpu::event_query((strata::gpu::Event*) d.hit_done)) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

namespace {

void hash_u64(uint64_t& h, uint64_t v) {
    h = fnv1a64(reinterpret_cast<const uint8_t*>(&v), sizeof v, h);
}

void hash_text(uint64_t& h, const std::string& s) {
    h = fnv1a64((const uint8_t*) s.data(), (uint64_t) s.size(), h);
}

bool hash_small_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    hash_text(h, path.filename().string());
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        hash_u64(h, 0);
        return true;
    }
    hash_u64(h, 1);
    std::vector<uint8_t> buf(64u << 10);
    for (;;) {
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = f.gcount();
        if (n > 0) h = fnv1a64(buf.data(), (uint64_t) n, h);
        if (f.eof()) break;
        if (!f) {
            err = "ArenaExpertSource: cannot hash pack metadata " + path.string();
            return false;
        }
    }
    return true;
}

bool hash_sampled_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "ArenaExpertSource: cannot sample pack source " + path.string();
        return false;
    }
    const std::streamoff end = f.tellg();
    if (end < 0) {
        err = "ArenaExpertSource: cannot size pack source " + path.string();
        return false;
    }
    const uint64_t bytes = (uint64_t) end;
    hash_text(h, path.filename().string());
    hash_u64(h, bytes);
    constexpr uint64_t sample = 64u << 10;
    const uint64_t starts[3] = {0, bytes / 2, bytes > sample ? bytes - sample : 0};
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(sample, bytes));
    for (uint64_t off : starts) {
        if (buf.empty()) break;
        const uint64_t at = std::min<uint64_t>(off, bytes - (uint64_t) buf.size());
        f.clear();
        f.seekg((std::streamoff) at);
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        if ((size_t) f.gcount() != buf.size()) {
            err = "ArenaExpertSource: short read while hashing pack source " + path.string();
            return false;
        }
        hash_u64(h, at);
        h = fnv1a64(buf.data(), (uint64_t) buf.size(), h);
    }
    return true;
}

bool shared_arena_pack_hash(const std::string& pack_dir, const std::string& experts_path,
                            const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay,
                            uint64_t& out, std::string& err) {
    uint64_t h = 1469598103934665603ull;
    hash_text(h, "strata-shared-expert-arena-pack-v1");
    hash_u64(h, (uint64_t) lay.n_layers);
    hash_u64(h, (uint64_t) lay.n_expert);
    hash_u64(h, lay.total);
    hash_u64(h, lay.max_blob);
    hash_u64(h, lay.native ? 1 : 0);

    const std::filesystem::path pack(pack_dir);
    for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"}) {
        if (!hash_small_file(pack / name, h, err)) return false;
    }

    if (std::filesystem::exists(experts_path)) {
        if (!hash_sampled_file(experts_path, h, err)) return false;
    } else if (!gguf.empty()) {
        // Native packs may read experts straight from one or more GGUF shards.  Sample every distinct source
        // file named by native_experts.txt; this keeps the fingerprint cheap while still tying it to the model
        // bytes rather than only to an equal-size layout.
        const std::filesystem::path first(gguf);
        std::vector<std::filesystem::path> sources{first};
        for (const std::string& name : lay.gguf_file) {
            if (name.empty()) continue;
            const std::filesystem::path p = first.parent_path() / name;
            if (std::find(sources.begin(), sources.end(), p) == sources.end()) sources.push_back(p);
        }
        for (const auto& p : sources) {
            if (!hash_sampled_file(p, h, err)) return false;
        }
    }

    out = h == 0 ? 1 : h;
    return true;
}

}  // namespace

namespace {
/// The GGUF file that holds role `r` (0 gate, 1 up, 2 down) of layer `l`: a name beside the --native shard
/// (native_experts.txt v3 per layer, v4 per role), or the --native shard itself.
std::string expert_gguf_file(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int r) {
    const size_t i = (size_t) (3 * l + r);
    if (lay.gguf_file.size() <= i || lay.gguf_file[i].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[i];
}
}  // namespace

void ExpertSource::record_inputs(const std::string& pack_experts, const std::string& gguf,
                                 const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf) {
    static const char* roles[3] = {"gate", "up", "down"};
    inputs_.clear();
    if (!from_gguf) { inputs_.push_back({"pack experts.bin", pack_experts}); return; }
    // the same resolution open_gguf / load_experts_gguf use: every (layer, role) tensor and the file it is read from
    for (int64_t l = 0; l < lay.n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            inputs_.push_back({"expert blk." + std::to_string(lay.block_of(l)) + ".ffn_" + roles[r],
                               expert_gguf_file(gguf, lay, l, r)});
}

bool expert_layout_from_gguf(const std::string& gguf, const std::vector<int64_t>& moe_blocks, int64_t n_expert,
                             int64_t n_embd, int64_t n_ff, std::string& err, bool folded_ok) {
    using namespace strata::kernels::cpu;
    static const char* roles[3] = {"gate", "up", "down"};
    if (moe_blocks.empty() || n_expert <= 0 || n_embd <= 0 || n_ff <= 0) {
        err = "experts from the GGUF: the geometry is empty";
        return false;
    }
    try {
        const strata::GgufModel model = strata::GgufModel::open(gguf);
        if (!folded_ok && model.shard(0).metadata().count("prism.hadamard.version")) {   // APR: hadamard.txt comes with a pack only
            err = "experts from the GGUF: its experts are Hadamard-folded (prism.hadamard); make a pack with "
                  "tools/iq_pack.py";
            return false;
        }
        const int64_t n_layers = (int64_t) moe_blocks.size();
        ExpertLayout L;
        L.native = true;
        L.n_layers = n_layers;
        L.n_expert = n_expert;
        L.fmt.resize((size_t) n_layers);
        L.offset.assign((size_t) n_layers, 0);
        L.bytes.assign((size_t) n_layers, 0);
        L.gguf_off.assign((size_t) (3 * n_layers), 0);
        L.gguf_file.assign((size_t) (3 * n_layers), std::string());
        L.block = moe_blocks;
        L.max_blob = 0;
        uint64_t at = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const strata::TensorInfo* t[3] = {};
            size_t shard[3] = {};
            for (int r = 0; r < 3; ++r) {
                const std::string name = "blk." + std::to_string(moe_blocks[(size_t) l]) + ".ffn_" + roles[r] +
                                         "_exps.weight";
                t[r] = model.find(name, &shard[r]);
                const uint64_t cols = (uint64_t) (r < 2 ? n_embd : n_ff), rows = (uint64_t) (r < 2 ? n_ff : n_embd);
                if (t[r] == nullptr || t[r]->shape.size() != 3 || t[r]->shape[0] != cols || t[r]->shape[1] != rows ||
                    t[r]->shape[2] != (uint64_t) n_expert) {
                    err = "experts from the GGUF: " + name + " is missing or not [" + std::to_string(cols) + ", " +
                          std::to_string(rows) + ", " + std::to_string(n_expert) + "]";
                    return false;
                }
                if (!model.in_bounds(*t[r], shard[r])) {
                    err = "experts from the GGUF: " + name + " runs past the end of its file (a truncated shard?)";
                    return false;
                }
            }
            // gate and up may differ in type (native_mixed_gu), as long as they read the same activation
            NativeFmt& f = L.fmt[(size_t) l];
            if (!native_fmt((int) t[0]->type, (int) t[2]->type, n_embd, n_ff, f, err, (int) t[1]->type)) {
                std::string where = "experts from the GGUF: block ";
                where += std::to_string(moe_blocks[(size_t) l]);
                where += ": ";
                err.insert(0, where);
                return false;
            }
            const uint64_t per[3] = {f.up_off, f.up_row * (uint64_t) n_ff, f.bytes - f.down_off};
            for (int r = 0; r < 3; ++r) {
                if (strata::tensor_payload_bytes(*t[r]) != per[r] * (uint64_t) n_expert) {
                    err = "experts from the GGUF: block " + std::to_string(moe_blocks[(size_t) l]) + "'s " + roles[r] +
                          " tensor is not " + std::to_string(per[r]) + " B per expert";
                    return false;
                }
                const strata::GgufFile& file = model.shard(shard[r]);
                L.gguf_off[(size_t) (3 * l + r)] = file.data_start() + t[r]->offset;
                const std::string& path = file.path();
                const size_t cut = path.find_last_of("/\\");
                L.gguf_file[(size_t) (3 * l + r)] = cut == std::string::npos ? path : path.substr(cut + 1);
            }
            L.offset[(size_t) l] = at;
            L.bytes[(size_t) l] = f.bytes;
            L.max_blob = std::max<uint64_t>(L.max_blob, f.bytes);
            at += (uint64_t) f.bytes * (uint64_t) n_expert;
        }
        L.total = at;
        expert_layout_set(L);
        return true;
    } catch (const std::exception& e) {
        err = std::string("experts from the GGUF: ") + e.what();
        return false;
    }
}

std::vector<std::string> native_layer_problems(bool mixed_gu) {
    using namespace strata::kernels::cpu;
    std::vector<std::string> out;
    const ExpertLayout& lay = expert_layout();
    for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
        const NativeFmt& f = lay.fmt[(size_t) l];
        const std::string at = "MoE layer " + std::to_string(l) + " (block " + std::to_string(lay.block_of(l)) + ")";
        const int up = f.up_type >= 0 ? f.up_type : f.gu_type;
        if (!mixed_gu && native_mixed_gu(f)) {
            out.push_back(at + " has gate and up of different types (" + strata::ggml_type_name((uint32_t) f.gu_type) +
                          ", " + strata::ggml_type_name((uint32_t) up) + "), which this engine's experts do not take");
        } else if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)) {
            out.push_back(at + "'s experts are " + strata::ggml_type_name((uint32_t) f.gu_type) + "/" +
                          strata::ggml_type_name((uint32_t) up) + "/" + strata::ggml_type_name((uint32_t) f.d_type) +
                          " (gate/up/down), which this engine has no GPU kernels for");
        }
    }
    return out;
}

bool check_experts_gguf(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    if (lay.gguf_off.size() != (size_t) (3 * lay.n_layers)) {
        err = "native_experts.txt has no GGUF offsets (a pack older than v2): repack it with tools/iq_pack.py";
        return false;
    }
    try {
        std::map<std::string, std::unique_ptr<strata::GgufFile>> files;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.down_off - fm.up_off, blob - fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string path = expert_gguf_file(gguf, lay, l, r);
                auto& f = files[path];
                if (!f) f = std::make_unique<strata::GgufFile>(path);
                const std::string name = "blk." + std::to_string(lay.block_of(l)) + ".ffn_" + roles[r] + "_exps.weight";
                const strata::TensorInfo* t = f->find(name);
                // up may be of a type of its own (native_mixed_gu: Ornith-1.5's GSQ-RCO files)
                const uint64_t want_type = (uint64_t) (r == 0 ? fm.gu_type
                                                       : r == 1 ? (fm.up_type >= 0 ? fm.up_type : fm.gu_type)
                                                                : fm.d_type);
                // GGUF order: dim 0 is the row (the input), dim 1 the rows, dim 2 the experts
                const uint64_t cols = (uint64_t) (r < 2 ? fm.n_embd : fm.n_ff);
                const uint64_t rows = (uint64_t) (r < 2 ? fm.n_ff : fm.n_embd);
                const uint64_t bytes = per[r] * (uint64_t) lay.n_expert;
                const uint64_t payload = f->file_size() - f->data_start();
                std::string why;
                if (t == nullptr) why = "is not in it";
                else if (t->type != want_type)
                    why = std::string("is ") + t->type_name() + ", the pack says type " + std::to_string(want_type);
                else if (t->shape.size() != 3 || t->shape[0] != cols || t->shape[1] != rows ||
                         t->shape[2] != (uint64_t) lay.n_expert)
                    why = "is not [" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
                          std::to_string(lay.n_expert) + "]";
                else if (strata::tensor_payload_bytes(*t) != bytes)
                    why = "is not " + std::to_string(per[r]) + " B per expert";
                else if (f->data_start() + t->offset != lay.gguf_off[(size_t) (3 * l + r)])
                    why = "starts at byte " + std::to_string(f->data_start() + t->offset) + ", the pack says " +
                          std::to_string(lay.gguf_off[(size_t) (3 * l + r)]);
                else if (t->offset > payload || bytes > payload - t->offset)
                    why = "runs past the end of the file (a truncated shard?)";
                if (!why.empty()) {
                    err = "the pack's native_experts.txt does not match the model: ";
                    err += name; err += " in "; err += path; err += " "; err += why;
                    err += " - repack with tools/iq_pack.py from this model's shards";
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("native experts from the GGUF: ") + e.what();
        return false;
    }
}

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, const std::vector<uint8_t*>& layer_dst,
                            const strata::kernels::cpu::ExpertLayout& lay, int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    // a layer's tensors may sit in other shards of the model (native_experts.txt v3/v4): names beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l, int r) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) (3 * l + r)].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) (3 * l + r)];
    };
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.down_off - fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string name = file_of(l, r);
                if (name != open_name) {
                    f.close();
                    f.clear();
                    f.open(name, std::ios::binary);
                    if (!f) { bad = true; return; }
                    open_name = name;
                }
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(layer_dst[(size_t) l] + (lay.blob_offset(l, (int64_t) e) - lay.layer_offset(l)) + at[r],
                                    buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

namespace {
// A shared arena an earlier engine left with units dropped (PinnedArena::mark_shared_dropped): only the experts whose
// bit is set in `missing` (layer * n_expert + expert) read again, from experts.bin (`path`, in the arena's layout) or
// from the GGUF's gate, up and down tensors (`gguf`), a run of adjacent experts at a time.  st.bytes: what was read.
LoadStats load_expert_units(const std::string& path, const std::string& gguf, bool from_gguf,
                            const std::vector<uint8_t*>& layer_dst, const strata::kernels::cpu::ExpertLayout& lay,
                            const std::vector<uint8_t>& missing, int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    auto dropped = [&](int64_t l, int64_t e) {
        const uint64_t u = (uint64_t) (l * lay.n_expert + e);
        return u / 8 < missing.size() && (missing[u / 8] >> (u % 8) & 1) != 0;
    };
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l, int r) -> std::string {
        if (!from_gguf) return path;
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) (3 * l + r)].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) (3 * l + r)];
    };
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    std::atomic<uint64_t> bytes{0};
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        auto read_at = [&](const std::string& name, uint64_t off, uint64_t n) {
            if (name != open_name) {
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                open_name = f ? name : std::string();
            }
            buf.resize((size_t) n);
            f.seekg((std::streamoff) off);
            f.read((char*) buf.data(), (std::streamsize) n);
            return (bool) f && (uint64_t) f.gcount() == n;
        };
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const uint64_t blob = lay.blob_bytes(l);
            uint8_t* dst = layer_dst[(size_t) l];
            for (int64_t e0 = 0; e0 < lay.n_expert;) {
                if (!dropped(l, e0)) { ++e0; continue; }
                int64_t e1 = e0 + 1;
                while (e1 < lay.n_expert && dropped(l, e1)) ++e1;
                const uint64_t n = (uint64_t) (e1 - e0);
                if (!from_gguf) {
                    if (!read_at(path, lay.blob_offset(l, e0), n * blob)) { bad = true; return; }
                    std::memcpy(dst + (lay.blob_offset(l, e0) - lay.layer_offset(l)), buf.data(), (size_t) (n * blob));
                } else {
                    const auto& fm = lay.fmt[(size_t) l];
                    const uint64_t per[3] = {fm.up_off, fm.down_off - fm.up_off, blob - fm.down_off};
                    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
                    for (int r = 0; r < 3; ++r) {
                        if (!read_at(file_of(l, r), lay.gguf_off[(size_t) (3 * l + r)] + (uint64_t) e0 * per[r],
                                     n * per[r])) { bad = true; return; }
                        for (uint64_t k = 0; k < n; ++k)
                            std::memcpy(dst + (lay.blob_offset(l, e0 + (int64_t) k) - lay.layer_offset(l)) + at[r],
                                        buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
                bytes += n * blob;
                e0 = e1;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.ok = false;
        st.error = "short read or unreadable file while reading the dropped experts again";
        return st;
    }
    st.bytes = bytes.load();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}
}  // namespace

LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst,
                            const strata::kernels::cpu::ExpertLayout& lay, int threads) {
    std::vector<uint8_t*> layers;
    layers.reserve((size_t) lay.n_layers);
    for (int64_t l = 0; l < lay.n_layers; ++l) layers.push_back(dst + lay.layer_offset(l));
    return load_experts_gguf(gguf, layers, lay, threads);
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::register_on(int device, std::string& err) {
    return arena_ == nullptr || static_cast<PinnedArena*>(arena_)->register_on(device, err);
}

void ArenaExpertSource::hold_registration(int device) {
    if (arena_ != nullptr) static_cast<PinnedArena*>(arena_)->hold_registration(device);
}

bool ArenaExpertSource::registered_on(int device) const {
    return arena_ != nullptr && static_cast<PinnedArena*>(arena_)->registered_on(device);
}

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, const std::string& shared_arena_file) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want && got != want + (uint64_t) blob) {   // + one blob: STRATA_ARENA_MMAP's padded file
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // STRATA_ARENA_MMAP=1 (a small-RAM machine whose GPUs hold most experts): the arena is the pack's experts.bin
    // mapped READ-ONLY - not locked, not pinned.  The page cache keeps what the CPU pool and the prompt path
    // actually read (the experts no VRAM cache holds) and gives back the rest under memory pressure; a page
    // dropped is re-read from the file, so a result never depends on what is resident.  The GPUs then get no
    // mapped alias: run with --pcie-frac 0.  The first start writes experts.bin (arena layout, padded by one blob
    // so a whole-slot copy may start at any expert), later ones map it.
#if defined(_WIN32)
    static const bool arena_mmap = false;   // POSIX mmap/madvise: Linux only for now
#else
    static const bool arena_mmap = [] { const char* v = strata::env_get("ARENA_MMAP"); return v && v[0] == '1'; }();
    if (arena_mmap) {
        const uint64_t file_bytes = want + (uint64_t) blob;
        uint64_t have = 0;
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (f) have = (uint64_t) f.tellg();
        }
        if (have == file_bytes) {
            const int fd = ::open(path.c_str(), O_RDONLY);
            void* v = fd >= 0 ? mmap(nullptr, (size_t) file_bytes, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
            if (fd >= 0) ::close(fd);
            if (v == MAP_FAILED) { err = "ArenaExpertSource: mmap of " + path + " failed"; return false; }
            map_ = v;
            map_bytes_ = file_bytes;
            base_ = (const uint8_t*) v;
            pinned_bytes_ = 0;
            dev_slice_.clear();
            slice_bytes_ = 0;
            blobs_ = n_layers * n_expert;
            n_expert_ = n_expert;
            reads_ = 0;
            note_ = "mapped read-only from " + path + " (XESTRATA_ARENA_MMAP: not locked, not pinned)";
            gib_per_s_ = 0.0;
            load_seconds_ = load_read_s_ = load_copy_s_ = 0.0;
            record_inputs(path, gguf_, lay, from_gguf);   // session files: the mapped experts.bin is a model input
            return true;
        }
    }
#endif
    // #633: THE RAM BEFORE THE ALLOCATION.  On Linux the arena is an anonymous mapping that succeeds whatever the host
    // has; its pages are committed as the load writes them, so a container whose memory limit is below the arena was
    // killed by the OOM killer part-way through the load, without a message.  A hard cgroup limit below it is
    // certain to end that way: refused, with the numbers.  Less RAM available than the arena (another program, an
    // engine still exiting) is only a warning - the OS may make room - per the recommend-not-force rule.
    ram_warning_.clear();
    // (a shared arena, --shared-expert-arena, may already be in RAM for another engine: not checked)
    if (detail::HostMemory hm; shared_arena_file.empty() && detail::host_available_memory(hm)) {
        const uint64_t need = want + (uint64_t) blob;
        const double gib = 1073741824.0;
        char buf[512];
        if (hm.cgroup_limit < need) {
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: the expert arena needs %.2f GiB of RAM but this process's memory limit "
                          "(cgroup memory.max / memory.limit_in_bytes) is %.2f GiB: it would be killed while loading. "
                          "Raise the container's limit, or run with less RAM: --mmap-experts with "
                          "--resident-budget-gib N keeps only the hottest experts in RAM",
                          (double) need / gib, (double) hm.cgroup_limit / gib);
            err = buf;
            return false;
        }
        if (hm.available < need) {
            // upstream 07d5ecfac (#4051): the shortfall is often an engine that was still exiting, a few seconds from
            // giving its RAM back: wait for it, not for ever, and say so.  STRATA_ARENA_RAM_WAIT_S changes the seconds
            // (0 loads at once).  The load then goes on either way - it only warns.
            double wait_s = 20.0;
            if (const char* w = strata::env_get("ARENA_RAM_WAIT_S"); w != nullptr && w[0] != '\0') {
                char* end = nullptr;
                const double v = std::strtod(w, &end);
                if (end != w && *end == '\0' && v >= 0.0 && v <= 3600.0) wait_s = v;
            }
            double waited = 0.0;
            uint64_t avail = hm.available;
            if (wait_s > 0.0) {
                const bool came = detail::wait_for_ram(
                    need, wait_s, 0.25, 5.0,
                    [](uint64_t& a) {
                        detail::HostMemory h;
                        if (!detail::host_available_memory(h)) return false;
                        a = h.available;
                        return true;
                    },
                    [](double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); },
                    [&](double w, uint64_t a) {
                        std::fprintf(stderr, "xestrata generate: the expert arena needs %.2f GiB of RAM and %.2f GiB is "
                                             "available; waiting up to %.0f s for it (another engine still exiting?) - "
                                             "%.0f s so far (XESTRATA_ARENA_RAM_WAIT_S changes this)\n",
                                     (double) need / gib, (double) a / gib, wait_s, w);
                    },
                    waited, avail);
                if (came)
                    std::fprintf(stderr, "xestrata generate: %.2f GiB of RAM is available now (after %.1f s): loading the "
                                         "expert arena\n", (double) avail / gib, waited);
                hm.available = avail;
            }
            if (hm.available < need) {
                std::snprintf(buf, sizeof buf,
                              "the expert arena needs %.2f GiB of RAM but %.2f GiB is available (%.2f GiB short%s): the "
                              "load may swap or be stopped by the OS%s. Close other programs, or run with less RAM: "
                              "--mmap-experts with --resident-budget-gib N",
                              (double) need / gib, (double) hm.available / gib, (double) (need - hm.available) / gib,
                              waited > 0.0 ? ", after waiting" : "",
                              hm.available < need / 2 ? " - with this little room the whole machine can stall in "
                                                        "memory reclaim until it is restarted; on Linux a systemd "
                                                        "MemoryHigh= below the RAM keeps one program from taking all "
                                                        "of it" : "");
                ram_warning_ = buf;
            }
        }
    }

    // The registrations for device copies end at expert starts, so no expert straddles two: a layer larger than the
    // device's largest allocation is cut inside (IQ3_XXS's on the RX 9060 XT; a copy across two registrations failed
    // there with an invalid value).  The arena is one blob longer than the file: a copy of a whole VRAM slot (the
    // largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes, cuts;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
        for (int64_t e = 0; e < n_expert; ++e) cuts.push_back(lay.blob_offset(l, e));
    }
    bounds.push_back(want);
    cuts.push_back(want);
    // STRATA_ARENA_HOST_USM=1: every layer in its own host USM block, which GPU kernels can read (the PCIe share).
    // Without it the one registered mapping, moved into host USM later only when the measured split reads experts
    // over PCIe and the runtime's registration is for copies only (Level Zero's): make_kernel_readable.
    static const bool host_usm = [] { const char* e = strata::env_get("ARENA_HOST_USM"); return e && *e == '1'; }();
    if (!shared_arena_file.empty() && host_usm) {
        err = "shared arena cannot use layer-separated host USM"; return false;
    }
    uint64_t pack_hash = 0;
    if (!shared_arena_file.empty() && !shared_arena_pack_hash(pack_dir, path, gguf_, lay, pack_hash, err)) return false;
    PinnedArena* a = nullptr;
    try {
        a = shared_arena_file.empty()
            ? new PinnedArena(want + (uint64_t) blob, bounds, (uint64_t) blob, host_usm, cuts)
            : new PinnedArena(want + (uint64_t) blob, bounds, 0, shared_arena_file, pack_hash, cuts);
    } catch (const std::exception& e) {
        // a runtime without the registration (intel/llvm's CUDA adapter has no urUSMImportExp) takes host USM
        if (!shared_arena_file.empty() || host_usm) {
            err = std::string("ArenaExpertSource: ") + e.what();
            return false;
        }
        std::fprintf(stderr, "xestrata: %s; the expert arena takes host USM instead\n", e.what());
        try {
            a = new PinnedArena(want + (uint64_t) blob, bounds, (uint64_t) blob, /*host_usm=*/true);
        } catch (const std::exception& e2) {
            err = std::string("ArenaExpertSource: ") + e.what() + "; host USM: " + e2.what();
            return false;
        }
    }
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    std::vector<uint8_t*> ldst;
    ldst.reserve((size_t) n_layers);
    for (int64_t l = 0; l < n_layers; ++l) ldst.push_back(a->at(lay.layer_offset(l)));
    LoadStats st;
    uint64_t reloaded = 0;
    try {
        const bool fill = a->begin_shared_population();
        if (fill && !a->shared_missing().empty()) {
            // an engine before this one dropped its coldest experts from the shared arena (drop_cold): those alone
            st = load_expert_units(path, gguf_, from_gguf, ldst, lay, a->shared_missing(), threads);
            reloaded = st.bytes;
            if (st.ok) { st.bytes = want; a->publish_shared(); }
        } else if (fill) {
            st = from_gguf ? load_experts_gguf(gguf_, ldst, lay, threads)
                           : load_experts_ranges(path, ldst, loff, lbytes, threads, /*chunk=*/8u << 20);
            if (st.ok && st.bytes == want) a->publish_shared();
        } else { st.bytes = want; st.layers = (uint64_t) n_layers; }
    } catch (const std::exception& e) {
        delete a; err = std::string("ArenaExpertSource: ") + e.what(); return false;
    }
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    // the first start: write experts.bin for the mapped starts after this one - only when the drive has room for it
    // and 2 GiB more (a full drive fails other writes too); otherwise this start says so and runs pinned as before
    std::error_code space_ec;
    const uint64_t free_disk = arena_mmap && from_gguf ? (uint64_t) std::filesystem::space(
        std::filesystem::path(path).parent_path(), space_ec).available : 0;
    const bool room = !space_ec && free_disk >= want + (uint64_t) blob + (2ull << 30);
    if (arena_mmap && from_gguf && !room)
        std::fprintf(stderr, "xestrata generate: XESTRATA_ARENA_MMAP: NOT writing %s - %.1f GiB free on that drive, it needs "
                             "%.1f GiB plus 2 GiB to spare; this start keeps the arena in RAM\n", path.c_str(),
                     (double) free_disk / 1073741824.0, (double) (want + (uint64_t) blob) / 1073741824.0);
    if (arena_mmap && from_gguf && room) {
        const std::string tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && std::fwrite(a->data(), 1, (size_t) want, f) == (size_t) want;
        if (ok) {
            std::vector<uint8_t> pad((size_t) blob, 0);
            ok = std::fwrite(pad.data(), 1, pad.size(), f) == pad.size();
        }
        if (f) ok = std::fclose(f) == 0 && ok;
        if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) std::remove(tmp.c_str());
        std::fprintf(stderr, "xestrata generate: XESTRATA_ARENA_MMAP: %s %s (the next start maps it)\n",
                     ok ? "wrote" : "could NOT write", path.c_str());
    }
    a->hold_in_place();   // filled: locking or pinning it now does not zero it before the load (#42)
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    // Windows: CUDA gives registered memory a device address of its own, not the host's (the RTX 3050 Laptop read the
    // host address and faulted), so a registration there is for copies only, as Level Zero's: make_kernel_readable
    // moves the arena into host USM when the PCIe share needs it.  Host USM is the same address on both sides.
#if defined(_WIN32)
    if (a->registered_bytes > 0 && !a->layer_base.empty()) {
#else
    if (a->registered_bytes > 0) {
#endif
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (!strata::gpu::device_pointer(&d, (void*) a->at(off))) {
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    if (reloaded > 0) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "; the rest was in the shared arena: %.2f GiB of experts an engine before this one "
                                       "dropped read again in %.1f s", (double) reloaded / 1073741824.0, st.seconds);
        note_ += buf;
    }
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    record_inputs(path, gguf_, lay, from_gguf);
    return true;
}

bool ArenaExpertSource::drop_cold(uint64_t want_bytes, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                  std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t& dropped,
                                  std::string& err) {
    ranges.clear();
    dropped = 0;
    if (arena_ == nullptr) { err = "the experts are not in an arena"; return false; }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const int64_t units = lay.n_layers * lay.n_expert;
    // the coldest first: the pairs the profile does not rank (the last layer first), then the ranked ones from the end
    std::vector<uint8_t> seen((size_t) units, 0);
    for (const auto& [l, e] : ranked)
        if (l >= 0 && l < lay.n_layers && e >= 0 && e < lay.n_expert) seen[(size_t) (l * lay.n_expert + e)] = 1;
    std::vector<int64_t> order;
    order.reserve((size_t) units);
    for (int64_t u = units - 1; u >= 0; --u)
        if (!seen[(size_t) u]) order.push_back(u);
    for (auto it = ranked.rbegin(); it != ranked.rend(); ++it)
        if (it->first >= 0 && it->first < lay.n_layers && it->second >= 0 && it->second < lay.n_expert) {
            const int64_t u = (int64_t) it->first * lay.n_expert + it->second;
            if (seen[(size_t) u] == 1) { seen[(size_t) u] = 2; order.push_back(u); }
        }
    std::vector<uint8_t> bits((size_t) ((units + 7) / 8), 0);
    for (int64_t u : order) {
        if (dropped >= want_bytes) break;
        bits[(size_t) (u / 8)] |= (uint8_t) (1u << (u % 8));
        dropped += lay.blob_bytes(u / lay.n_expert);
    }
    if (dropped == 0) return true;
    auto* a = (PinnedArena*) arena_;
    if (!a->mark_shared_dropped(bits, err)) { dropped = 0; return false; }
    // the file's ranges, adjacent experts as one (the server punches them out of the file once this engine ended)
    const uint64_t head = PinnedArena::shared_header_bytes();
    for (int64_t u = 0; u < units; ++u) {
        if (!(bits[(size_t) (u / 8)] >> (u % 8) & 1)) continue;
        const int64_t l = u / lay.n_expert, e = u % lay.n_expert;
        const uint64_t off = head + lay.blob_offset(l, e), n = lay.blob_bytes(l);
        if (!ranges.empty() && ranges.back().first + ranges.back().second == off) ranges.back().second += n;
        else ranges.emplace_back(off, n);
    }
    return true;
}

bool arena_drop_command(ArenaExpertSource& arena, const std::string& line,
                        const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& answer, std::string& err) {
    const std::string head = "ARENA DROP ";
    char* end = nullptr;
    const long long mib = line.rfind(head, 0) == 0 ? std::strtoll(line.c_str() + head.size(), &end, 10) : -1;
    if (mib < 0 || end == nullptr || *end != ' ' || end[1] == '\0') {
        err = "expected: ARENA DROP <mib> <file>";
        return false;
    }
    const std::string file = end + 1;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t dropped = 0;
    if (!arena.drop_cold((uint64_t) mib << 20, ranked, ranges, dropped, err)) return false;
    std::ofstream out(file, std::ios::trunc);
    for (const auto& [off, n] : ranges) out << off << ' ' << n << '\n';
    out.close();
    if (!out) { err = "ARENA DROP: cannot write " + file; return false; }
    answer = "ARENA dropped_mib=" + std::to_string(dropped >> 20) + " ranges=" + std::to_string(ranges.size());
    return true;
}

void ArenaExpertSource::close() {
#if !defined(_WIN32)
    if (map_ != nullptr) munmap(map_, (size_t) map_bytes_);
#endif
    map_ = nullptr;
    map_bytes_ = 0;
    inputs_.clear();
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

void ArenaExpertSource::prefetch(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = lay.blob_offset(layer, expert) & ~(uint64_t) 4095;
    const uint64_t end = lay.blob_offset(layer, expert) + lay.blob_bytes(layer);
    madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_WILLNEED);
#endif
}

// Only the pages wholly inside the blob: a page shared with a neighbour the CPU may still read stays.  MADV_DONTNEED
// unmaps them from this process; they stay in the page cache as clean, unmapped pages - the first the kernel takes
// back under pressure, and a minor fault away when the CPU needs the expert again (an adaptive swap evicts it).
// Dropping them from the cache too (POSIX_FADV_DONTNEED) made every eviction a re-read from the disk: ~200 major
// faults a second in decode, +8 ms a window.  The memory the arena used to hold was never the cache itself but
// ROCclr's pin-in-place locks on it (see main) - locked pages are the ones the kernel cannot take back.
uint64_t ArenaExpertSource::release(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
    return 0;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = (lay.blob_offset(layer, expert) + 4095) & ~(uint64_t) 4095;
    const uint64_t end = (lay.blob_offset(layer, expert) + lay.blob_bytes(layer)) & ~(uint64_t) 4095;
    if (end <= off) return 0;
    if (madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_DONTNEED) != 0) return 0;
    return end - off;
#endif
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

bool ArenaExpertSource::make_kernel_readable(std::string& err) {
    if (!dev_slice_.empty()) return true;
    if (arena_ == nullptr) { err = "the arena is a file mapping"; return false; }
    auto* a = static_cast<PinnedArena*>(arena_);
    // issue #10: the RAM available before each layer moves (UINT64_MAX when it cannot be read)
    const auto ram = [] {
        detail::HostMemory hm;
        return detail::host_available_memory(hm) ? hm.available : UINT64_MAX;
    };
    if (!a->to_host_usm(strata::kernels::cpu::expert_layout().max_blob, err, ram)) return false;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    slice_bytes_ = a->slice_bytes;
    for (uint64_t off : a->slice_starts) {
        void* d = nullptr;
        if (!strata::gpu::device_pointer(&d, (void*) a->at(off))) {
            dev_slice_.clear();
            err = "the host USM blocks have no device address";
            return false;
        }
        dev_slice_.push_back((const uint8_t*) d);
    }
    note_ = a->note;
    return true;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    const uint64_t off = strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
    return arena_ != nullptr ? ((const PinnedArena*) arena_)->at(off) : base_ + off;   // else STRATA_ARENA_MMAP's map
}

const uint8_t* ArenaExpertSource::blob_at(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    const uint64_t off = strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
    return arena_ != nullptr ? ((const PinnedArena*) arena_)->at(off) : base_ + off;
}

}  // namespace strata::core
