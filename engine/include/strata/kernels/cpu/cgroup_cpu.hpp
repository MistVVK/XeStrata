// include/strata/kernels/cpu/cgroup_cpu.hpp - the CPU time a process may use under cgroup limits (Docker --cpus, a Kubernetes or RunPod
// limit, a systemd CPUQuota=), as a number of CPUs.  Header only and independent of the real /sys: the caller passes the cgroup mount
// and the text of /proc/self/cgroup, so a test can build both in a temp directory (cgroup_cpu_test.cpp).
//
// Why not just read <mount>/cpu.max: that file is the process's own group only when the container has its own cgroup namespace (the
// Docker default on a cgroup v2 host).  Without one - the host's namespace, a systemd service or scope, an older runtime - the mount
// shows the host's tree and the limit sits in the process's group or in a parent of it, so the root of the mount says nothing.  The
// limit that applies is the smallest quota on the way from the process's group up to the root, as for memory (conversation_memory.cpp).
#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

namespace cgroup_cpu_detail {

/// "a/b/c" -> {"a","b","c"}; empty parts dropped.  False for a path with "..", which no cgroup path has.
inline bool split_path(const std::string& path, std::vector<std::string>& out) {
    out.clear();
    std::string part;
    std::istringstream in(path);
    while (std::getline(in, part, '/')) {
        if (part.empty() || part == ".") continue;
        if (part == "..") return false;
        out.push_back(part);
    }
    return true;
}

inline bool read_line(const std::string& file, std::string& line) {
    std::ifstream f(file);
    if (!f) return false;
    std::getline(f, line);
    return true;
}

/// cgroup v2 cpu.max in `dir`: "max 100000" (no limit) or "<quota> <period>" -> CPUs, 0 for no limit or unreadable.
inline double v2_quota(const std::string& dir) {
    std::string line;
    if (!read_line(dir + "/cpu.max", line)) return 0.0;
    std::istringstream in(line);
    std::string quota;
    long long period = 0;
    if (!(in >> quota >> period) || quota == "max" || period <= 0) return 0.0;
    char* end = nullptr;
    const double q = std::strtod(quota.c_str(), &end);
    return end != quota.c_str() && q > 0 ? q / (double) period : 0.0;
}

/// cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us in `dir`: -1 means no limit.
inline double v1_quota(const std::string& dir) {
    std::string a, b;
    if (!read_line(dir + "/cpu.cfs_quota_us", a) || !read_line(dir + "/cpu.cfs_period_us", b)) return 0.0;
    const long long quota = std::atoll(a.c_str()), period = std::atoll(b.c_str());
    return quota > 0 && period > 0 ? (double) quota / (double) period : 0.0;
}

/// The smallest limit in `base`, `base/p1`, `base/p1/p2`, ... `base/p1/.../pn`, read with `one`; 0 when none has one.
template <class Reader>
double tightest_on_the_way(const std::string& base, const std::vector<std::string>& parts, Reader one) {
    double best = 0.0;
    std::string dir = base;
    for (size_t i = 0;; ++i) {
        const double q = one(dir);
        if (q > 0.0 && (best <= 0.0 || q < best)) best = q;
        if (i >= parts.size()) break;
        dir += "/" + parts[i];
    }
    return best;
}

}  // namespace cgroup_cpu_detail

/// CPUs the cgroup limits allow this process: 0 when no limit applies.  `cgroup_root` is where the cgroup files are mounted (/sys/fs/cgroup),
/// `self_cgroup` the text of /proc/self/cgroup ("0::/path" for cgroup v2, "N:cpu,cpuacct:/path" for v1; both kinds of line may be there).
/// The group's own path is read under the mount and so is every parent of it, down to the mount's root (which is the container's own group
/// when it has a cgroup namespace of its own, and where the old reading stopped).  A path that is not there is skipped.
inline double cgroup_cpu_quota_at(const std::string& cgroup_root, const std::string& self_cgroup) {
    using namespace cgroup_cpu_detail;
    double best = 0.0;
    auto take = [&best](double q) { if (q > 0.0 && (best <= 0.0 || q < best)) best = q; };
    bool any_line = false;
    std::istringstream in(self_cgroup);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t c1 = line.find(':');
        const size_t c2 = c1 == std::string::npos ? std::string::npos : line.find(':', c1 + 1);
        if (c2 == std::string::npos) continue;
        const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
        std::vector<std::string> parts;
        if (!split_path(line.substr(c2 + 1), parts)) continue;
        any_line = true;
        if (controllers.empty()) {                 // cgroup v2 (the unified hierarchy)
            take(tightest_on_the_way(cgroup_root, parts, v2_quota));
        } else {                                   // cgroup v1: the line that lists the cpu controller
            std::istringstream names(controllers);
            std::string name;
            bool has_cpu = false;
            while (std::getline(names, name, ',')) has_cpu = has_cpu || name == "cpu";
            if (!has_cpu) continue;
            for (const char* mount : {"/cpu", "/cpu,cpuacct", "/cpuacct,cpu"})
                take(tightest_on_the_way(cgroup_root + mount, parts, v1_quota));
        }
    }
    if (!any_line) {   // no /proc/self/cgroup: the mount's own root, v2 then v1 (what the engine read before)
        take(v2_quota(cgroup_root));
        take(v1_quota(cgroup_root + "/cpu"));
    }
    return best;
}

}  // namespace strata::kernels::cpu
