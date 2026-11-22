// SPDX-License-Identifier: MIT
// include/strata/platform/cgroup_mem.hpp - Linux cgroup v1/v2 memory ceiling with ancestor walk and reclaimable
// file-cache credit (portable C++17 header, testable on any OS with synthetic fixture directories).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::platform::cgroup_mem {

inline constexpr uint64_t kGiB = 1073741824ULL;               // 2^30 bytes (never mix with 1e9!)
inline constexpr uint64_t kUnlimited = ~uint64_t{0};
inline constexpr uint64_t kMaxSaneLimit = uint64_t{1} << 50;  // 1 PiB: v1's "unlimited" sits near PAGE_ALIGN(LONG_MAX)

struct CgroupRoom {
    bool limited = false;                  ///< true when a cgroup limit meaningfully bounds host RAM
    uint64_t limit_bytes = kUnlimited;     ///< tightest memory.max / memory.limit_in_bytes across leaf..root
    uint64_t available_bytes = kUnlimited; ///< tightest (limit - max(0, current - reclaimable_file)) across leaf..root
};

namespace detail {

inline bool read_u64_file(const std::string& path, uint64_t& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string tok;
    if (!(f >> tok)) return false;
    if (tok == "max") {
        out = kUnlimited;
        return true;
    }
    char* end = nullptr;
    const unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
    if (end == tok.c_str() || *end != '\0') return false;
    out = (v >= kMaxSaneLimit) ? kUnlimited : static_cast<uint64_t>(v);
    return true;
}

inline uint64_t reclaimable_from_stat(const std::string& stat_path, bool v2) {
    std::ifstream f(stat_path);
    if (!f) return 0;
    uint64_t inactive_file = 0, active_file = 0, file_mapped = 0, file_dirty = 0, file_writeback = 0;
    bool saw_total_inactive = false;
    std::string key;
    uint64_t val = 0;
    while (f >> key >> val) {
        if (v2) {
            if (key == "inactive_file") inactive_file = val;
            else if (key == "active_file") active_file = val;
            else if (key == "file_mapped") file_mapped = val;
            else if (key == "file_dirty") file_dirty = val;
            else if (key == "file_writeback") file_writeback = val;
        } else {
            if (key == "total_inactive_file") { inactive_file = val; saw_total_inactive = true; }
            else if (key == "inactive_file" && !saw_total_inactive) inactive_file = val;
            else if (key == "total_mapped_file" || key == "mapped_file") file_mapped = std::max(file_mapped, val);
            else if (key == "total_dirty" || key == "dirty") file_dirty = std::max(file_dirty, val);
            else if (key == "total_writeback" || key == "writeback") file_writeback = std::max(file_writeback, val);
        }
    }
    // Conservative credit: clean unmapped inactive file cache (plus active_file when unmapped on v2) that the kernel
    // reclaims before invoking OOM.
    const uint64_t gross = v2 ? (inactive_file + active_file) : inactive_file;
    const uint64_t held = file_mapped + file_dirty + file_writeback;
    return gross > held ? (gross - held) : 0;
}

inline std::vector<std::string> ancestor_chain(const std::string& rel_subpath) {
    std::vector<std::string> out;
    std::string clean = rel_subpath;
    while (!clean.empty() && clean.back() == '/') clean.pop_back();
    while (!clean.empty() && clean != "/") {
        out.push_back(clean);
        const auto slash = clean.rfind('/');
        if (slash == std::string::npos || slash == 0) break;
        clean.resize(slash);
    }
    out.emplace_back("");   // mount root last
    return out;
}

inline void tighten_dir(const std::string& dir, bool v2, CgroupRoom& room) {
    const std::string lim_file = dir + (v2 ? "/memory.max" : "/memory.limit_in_bytes");
    uint64_t lim = kUnlimited;
    if (!read_u64_file(lim_file, lim) || lim == kUnlimited || lim == 0) return;
    room.limit_bytes = std::min(room.limit_bytes, lim);

    const std::string cur_file = dir + (v2 ? "/memory.current" : "/memory.usage_in_bytes");
    uint64_t cur = 0;
    if (!read_u64_file(cur_file, cur)) {
        room.available_bytes = std::min(room.available_bytes, lim);
        return;
    }
    const uint64_t reclaimable = reclaimable_from_stat(dir + "/memory.stat", v2);
    const uint64_t eff_used = cur > reclaimable ? (cur - reclaimable) : 0;
    const uint64_t avail = lim > eff_used ? (lim - eff_used) : 0;
    room.available_bytes = std::min(room.available_bytes, avail);
}

}  // namespace detail

/// Probes cgroup v2 (`0::/slice`) and v1 (`N:memory:/slice`) ancestor chains under `cgroup_root`.
inline CgroupRoom probe_cgroup_room(uint64_t host_total_bytes,
                                    const std::string& self_cgroup = "/proc/self/cgroup",
                                    const std::string& cgroup_root = "/sys/fs/cgroup") {
    CgroupRoom room;
    std::string v2_rel;
    std::string v1_rel;
    bool have_v2_line = false, have_v1_line = false;

    if (std::ifstream cg(self_cgroup); cg) {
        std::string line;
        while (std::getline(cg, line)) {
            const auto c1 = line.find(':');
            if (c1 == std::string::npos) continue;
            const auto c2 = line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            const std::string id = line.substr(0, c1);
            const std::string subsys = line.substr(c1 + 1, c2 - c1 - 1);
            const std::string path = line.substr(c2 + 1);
            if (id == "0" && subsys.empty()) {
                v2_rel = path;
                have_v2_line = true;
            } else {
                std::stringstream ss(subsys);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (item == "memory") {
                        v1_rel = path;
                        have_v1_line = true;
                    }
                }
            }
        }
    }

    if (have_v2_line) {
        for (const std::string& rel : detail::ancestor_chain(v2_rel)) {
            detail::tighten_dir(cgroup_root + rel, /*v2=*/true, room);
        }
    } else {
        detail::tighten_dir(cgroup_root, /*v2=*/true, room);
    }

    if (have_v1_line) {
        for (const std::string& rel : detail::ancestor_chain(v1_rel)) {
            detail::tighten_dir(cgroup_root + "/memory" + rel, /*v2=*/false, room);
            detail::tighten_dir(cgroup_root + rel, /*v2=*/false, room);
        }
    } else {
        detail::tighten_dir(cgroup_root + "/memory", /*v2=*/false, room);
    }

    if (room.limit_bytes != kUnlimited && room.limit_bytes > (uint64_t{1} << 28) &&
        (host_total_bytes == 0 || room.limit_bytes + kGiB < host_total_bytes)) {
        room.limited = true;
    }
    return room;
}

inline bool in_cgroup_memory_limit(uint64_t host_total_bytes = 0,
                                   const std::string& self_cgroup = "/proc/self/cgroup",
                                   const std::string& cgroup_root = "/sys/fs/cgroup") {
    if (host_total_bytes == 0) {
        if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long kb = 0;
            while (std::fgets(line, sizeof(line), f)) {
                if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
                    host_total_bytes = kb * 1024ULL;
                    break;
                }
            }
            std::fclose(f);
        }
    }
    return probe_cgroup_room(host_total_bytes, self_cgroup, cgroup_root).limited;
}

/// Converts a GiB figure to exact bytes using 2^30 (fixing decimal 1e9 under-allocation).
constexpr uint64_t gib_to_bytes(double gib) noexcept {
    return gib > 0.0 ? static_cast<uint64_t>(gib * static_cast<double>(kGiB)) : 0ULL;
}

}  // namespace strata::platform::cgroup_mem
